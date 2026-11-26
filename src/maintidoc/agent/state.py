"""Agent state.

A `TypedDict` because that is what LangGraph merges between nodes; the values
inside it are the Pydantic contracts from earlier phases, so nothing is
re-modelled here.

The state carries the budget tracker itself rather than a copy of its counters.
Passing counts around would let two branches each believe they had spent
nothing.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from maintidoc.agent.budget import BudgetTracker
from maintidoc.contracts import (
    AnswerV1,
    ApprovalToken,
    EvidenceBundle,
    Principal,
    TicketDraft,
)


class Outcome(TypedDict):
    """What the run produced, for the caller."""

    status: str
    detail: str


class AgentState(TypedDict):
    # --- input ---
    question: str
    principal: Principal
    visual_query: NotRequired[str | None]
    equipment_id: NotRequired[str | None]
    wants_ticket: NotRequired[bool]

    # --- accumulated ---
    budget: BudgetTracker
    bundle: NotRequired[EvidenceBundle | None]
    answer: NotRequired[AnswerV1 | None]
    history: NotRequired[list[dict[str, Any]]]
    open_tickets: NotRequired[list[dict[str, Any]]]
    draft: NotRequired[TicketDraft | None]
    approval: NotRequired[ApprovalToken | None]
    ticket_id: NotRequired[str | None]

    # --- bookkeeping ---
    trace: NotRequired[list[str]]
    warnings: NotRequired[list[str]]
    outcome: NotRequired[Outcome | None]
    grounded: NotRequired[bool]


def initial_state(
    question: str,
    principal: Principal,
    budget: BudgetTracker,
    *,
    equipment_id: str | None = None,
    visual_query: str | None = None,
    wants_ticket: bool = False,
) -> AgentState:
    return AgentState(
        question=question,
        principal=principal,
        visual_query=visual_query,
        equipment_id=equipment_id,
        wants_ticket=wants_ticket,
        budget=budget,
        bundle=None,
        answer=None,
        history=[],
        open_tickets=[],
        draft=None,
        approval=None,
        ticket_id=None,
        trace=[],
        warnings=[],
        outcome=None,
        grounded=False,
    )
