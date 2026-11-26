"""Maintenance records exposed through MCP.

Everything here is fictitious, as the brief requires. The shapes still matter:
they are what the agent reads, and what a human approves before a ticket is
created.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TicketStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"


class Equipment(BaseModel):
    """An equipment item and the group allowed to see its records.

    The access group lives with the equipment on the *server*. The agent may
    claim anything; the server decides from this.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    equipment_id: str = Field(min_length=1, max_length=64)
    label: str
    family: str
    acl_group: str


class Intervention(BaseModel):
    """A past intervention, read-only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intervention_id: str
    equipment_id: str
    performed_at: datetime
    summary: str
    outcome: str


class Ticket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str
    equipment_id: str
    title: str
    description: str
    status: TicketStatus = TicketStatus.OPEN
    created_by: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    idempotency_key: str | None = None


class ToolCallRecord(BaseModel):
    """Audit line for one tool call.

    Metadata only — who, what, when, allowed or refused. No document content
    and no token value, per the brief's logging requirement (§9).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str
    user_id: str
    equipment_id: str | None
    allowed: bool
    reason: str = ""
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
