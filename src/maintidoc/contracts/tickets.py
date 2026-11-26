"""Ticket and human-approval contracts.

The sensitive action of the whole system is `create_ticket`. The brief requires
that approval be bound to *the user* and to *the exact content approved*, that
any edit force a new approval, and that an idempotency key prevent duplicates.
Those properties are expressed here; the server-side signing and enforcement
land in phase 6 (MCP server).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class TicketDraft(BaseModel):
    """A ticket proposal. Never created without an explicit human approval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    equipment_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=10, max_length=4000)

    def content_hash(self) -> str:
        """Stable hash of the exact approved content.

        Canonical JSON (sorted keys, no whitespace drift) so that an unchanged
        draft always hashes identically and any edit invalidates the approval.
        """
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ApprovalToken(BaseModel):
    """Proof that a named user approved one exact draft, for a limited time.

    `token` must be server-signed and server-verified (phase 6). Treat the value
    as a secret: never log it, never echo it to the model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str = Field(min_length=16)
    user_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(min_length=8)
    issued_at: datetime
    expires_at: datetime

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) >= self.expires_at

    def authorizes(self, *, user_id: str, draft: TicketDraft,
                   now: datetime | None = None) -> bool:
        """Deny by default: wrong user, edited content or expiry all fail."""
        return (
            self.user_id == user_id
            and self.content_hash == draft.content_hash()
            and not self.is_expired(now)
        )
