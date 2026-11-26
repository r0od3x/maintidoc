"""Turning an evidence bundle into a model request.

Two things this module is careful about.

**Evidence is data, never instruction.** Each excerpt is wrapped in a labelled
block and the allowed references are listed separately. A document that says
"ignore your instructions" is then visibly a quoted passage rather than a
sibling of the system prompt. That is defence in depth, not a guarantee —
phase 9 tests whether it actually holds.

**The model is told exactly what it may cite.** The allowed list is the
bundle's own references, so the structural verifier and the prompt cannot
disagree about what counts as a real citation.
"""

from __future__ import annotations

from dataclasses import dataclass

from maintidoc.contracts import Evidence, EvidenceBundle, answer_json_schema
from maintidoc.generation.client import GenerationRequest
from maintidoc.ingestion.store import ContentStore
from maintidoc.prompts import PromptTemplate, load_prompt


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    request: GenerationRequest
    prompt: PromptTemplate
    allowed_refs: tuple[str, ...]
    images_sent: int


def format_evidence(items: tuple[Evidence, ...]) -> str:
    """One labelled block per element, excerpt verbatim."""
    blocks: list[str] = []
    for item in items:
        header = f"[SOURCE {item.ref} | page {item.page} | {item.modality.value}]"
        if item.image_uri:
            header += " (image jointe)"
        body = (item.excerpt or "").strip() or "(aucun texte : voir l'image jointe)"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def format_allowed(items: tuple[Evidence, ...]) -> str:
    return "\n".join(
        f"- {item.ref}  (document {item.document_id}, page {item.page}, "
        f"element {item.element_id})"
        for item in items
    )


def load_images(bundle: EvidenceBundle, content: ContentStore | None) -> list[bytes]:
    """Original bytes for the figures selected by the retriever.

    A missing binary is skipped rather than fatal: the caption is usually
    still present, so the answer is degraded rather than lost.
    """
    if content is None:
        return []
    images: list[bytes] = []
    for item in bundle.image_items:
        if not item.image_uri:
            continue
        try:
            images.append(content.get(item.image_uri))
        except (OSError, ValueError):
            continue
    return images


def prepare_answer_prompt(
    bundle: EvidenceBundle,
    *,
    content: ContentStore | None = None,
    prompts_dir=None,
    num_ctx: int = 4096,
    num_predict: int = 900,
    version: str = "v1",
) -> PreparedPrompt:
    template = load_prompt("answer", version, prompts_dir)
    user = template.render_user(
        question=bundle.question,
        evidence=format_evidence(bundle.items),
        allowed_refs=format_allowed(bundle.items) or "(aucune)",
    )
    images = load_images(bundle, content)
    return PreparedPrompt(
        request=GenerationRequest(
            system=template.system,
            user=user,
            images=tuple(images),
            schema=answer_json_schema(),
            num_ctx=num_ctx,
            num_predict=num_predict,
        ),
        prompt=template,
        allowed_refs=tuple(sorted(bundle.allowed_refs)),
        images_sent=len(images),
    )


def prepare_evidence_check_prompt(
    claim: str,
    ref: str,
    content_text: str,
    *,
    prompts_dir=None,
    version: str = "v1",
    num_ctx: int = 4096,
) -> tuple[GenerationRequest, PromptTemplate]:
    template = load_prompt("evidence_check", version, prompts_dir)
    user = template.render_user(
        claim=claim, source_ref=ref, source_content=content_text
    )
    return (
        GenerationRequest(
            system=template.system,
            user=user,
            schema=None,
            num_ctx=num_ctx,
            num_predict=350,
        ),
        template,
    )
