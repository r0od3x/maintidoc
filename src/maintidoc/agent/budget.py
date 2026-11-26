"""Budgets, enforced by the code rather than requested in a prompt.

The brief fixes them (§9): at most 3 MCP calls, at most 6 model calls, at most
60 seconds of automatic processing, excluding time waiting for a human.

They are counters, not advice. An agent asked politely to stop looping will
sometimes loop anyway; a counter that raises cannot be talked out of it.

Time spent waiting for human approval is explicitly excluded — a technician
reading a draft ticket for two minutes has not consumed any compute, and
letting the clock run there would make the limit punish the human.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """A limit was reached. The run must stop and say so."""

    def __init__(self, resource: str, limit: int | float) -> None:
        self.resource = resource
        self.limit = limit
        super().__init__(f"Budget depasse : {resource} (limite {limit})")


@dataclass
class BudgetTracker:
    max_mcp_calls: int = 3
    max_model_calls: int = 6
    max_seconds: float = 60.0

    mcp_calls: int = 0
    model_calls: int = 0
    retrieval_rounds: int = 0
    max_retrieval_rounds: int = 3

    _started: float = field(default_factory=time.perf_counter)
    _paused_at: float | None = None
    _paused_total: float = 0.0

    # --- consumption ------------------------------------------------------

    def spend_mcp(self) -> None:
        if self.mcp_calls >= self.max_mcp_calls:
            raise BudgetExceeded("appels MCP", self.max_mcp_calls)
        self.mcp_calls += 1

    def spend_model(self) -> None:
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded("appels modele", self.max_model_calls)
        self.model_calls += 1

    def spend_retrieval(self) -> None:
        if self.retrieval_rounds >= self.max_retrieval_rounds:
            raise BudgetExceeded("tours de recuperation", self.max_retrieval_rounds)
        self.retrieval_rounds += 1

    # --- the clock --------------------------------------------------------

    @property
    def elapsed(self) -> float:
        now = time.perf_counter()
        paused = self._paused_total
        if self._paused_at is not None:
            paused += now - self._paused_at
        return now - self._started - paused

    def check_time(self) -> None:
        if self.elapsed > self.max_seconds:
            raise BudgetExceeded("secondes de traitement", self.max_seconds)

    def pause(self) -> None:
        """Stop the clock while a human is deciding."""
        if self._paused_at is None:
            self._paused_at = time.perf_counter()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._paused_total += time.perf_counter() - self._paused_at
            self._paused_at = None

    # --- reporting --------------------------------------------------------

    @property
    def remaining_mcp(self) -> int:
        return max(self.max_mcp_calls - self.mcp_calls, 0)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "mcp_calls": self.mcp_calls,
            "model_calls": self.model_calls,
            "retrieval_rounds": self.retrieval_rounds,
            "elapsed_seconds": round(self.elapsed, 3),
        }

    @classmethod
    def from_settings(cls, settings) -> BudgetTracker:
        return cls(
            max_mcp_calls=settings.max_mcp_calls,
            max_model_calls=settings.max_model_calls,
            max_seconds=float(settings.max_processing_seconds),
        )
