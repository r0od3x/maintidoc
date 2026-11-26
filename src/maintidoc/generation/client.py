"""Vision-language model clients.

One protocol, three implementations: Ollama for real use, a scripted stub for
tests, and a recording wrapper for capturing fixtures.

Generation is constrained by the `AnswerV1` JSON schema rather than asked for
politely in the prompt. A prompt can request a shape; constrained decoding
enforces it, and anything that still comes back malformed is a hard failure
rather than something to parse hopefully.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class GenerationError(RuntimeError):
    """The model was unreachable, too slow, or returned nothing usable."""


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    system: str
    user: str
    images: tuple[bytes, ...] = ()
    schema: dict[str, Any] | None = None
    num_ctx: int = 4096
    num_predict: int = 900
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    model: str
    elapsed_ms: float = 0.0
    done_reason: str | None = None

    @property
    def looks_truncated(self) -> bool:
        """A response cut off at the token limit is not a refusal to answer.

        Worth distinguishing: truncation is fixed by raising `num_predict`,
        while an empty answer usually means the model spent its budget
        thinking.
        """
        return self.done_reason == "length"


@runtime_checkable
class VlmClient(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    def available(self) -> bool: ...


class OllamaVlmClient:
    """Local Ollama. Nothing leaves the machine.

    `think=False` matters for the Qwen family: left on, the model can spend
    its whole generation budget reasoning and return empty content — the
    failure the prototype notebook papered over with a retry loop.
    """

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://127.0.0.1:11434",
        timeout: int = 240,
        keep_alive: str = "5m",
    ) -> None:
        self.name = model
        self.host = host
        self.timeout = timeout
        self.keep_alive = keep_alive

    def _client(self):
        from ollama import Client

        return Client(host=self.host, timeout=self.timeout)

    def available(self) -> bool:
        try:
            models = self._client().list()
        except Exception:
            return False
        names = {getattr(entry, "model", "") for entry in getattr(models, "models", [])}
        return self.name in names

    def installed_models(self) -> list[str]:
        try:
            models = self._client().list()
        except Exception:
            return []
        return sorted(
            name for name in (getattr(e, "model", "") for e in models.models) if name
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        import time

        message: dict[str, Any] = {"role": "user", "content": request.user}
        if request.images:
            message["images"] = list(request.images)

        started = time.perf_counter()
        try:
            result = self._client().chat(
                model=self.name,
                messages=[{"role": "system", "content": request.system}, message],
                think=False,
                format=request.schema or "json",
                options={
                    "temperature": request.temperature,
                    "num_ctx": request.num_ctx,
                    "num_predict": request.num_predict,
                },
                keep_alive=self.keep_alive,
            )
        except Exception as exc:
            raise GenerationError(f"Ollama injoignable ou en echec : {exc}") from exc

        return GenerationResponse(
            text=result.message.content or "",
            model=self.name,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            done_reason=getattr(result, "done_reason", None),
        )


@dataclass
class StubVlmClient:
    """Returns scripted responses. Used by the tests, never at runtime."""

    responses: list[str] = field(default_factory=list)
    name: str = "stub"
    requests: list[GenerationRequest] = field(default_factory=list)

    def available(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        if not self.responses:
            raise GenerationError("StubVlmClient : aucune reponse scriptee restante")
        return GenerationResponse(text=self.responses.pop(0), model=self.name)

    @classmethod
    def answering(cls, answer: str, citations: list[dict], **extra) -> StubVlmClient:
        payload = {
            "status": "answered",
            "answer": answer,
            "citations": citations,
            "uncertainties": [],
            "clarification_question": None,
            "proposed_action": None,
        }
        payload.update(extra)
        return cls(responses=[json.dumps(payload, ensure_ascii=False)])


def build_vlm_client(settings) -> VlmClient:
    return OllamaVlmClient(
        settings.vlm_model,
        host=settings.ollama_host,
        timeout=settings.max_processing_seconds * 4,
    )
