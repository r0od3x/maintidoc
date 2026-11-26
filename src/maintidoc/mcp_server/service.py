"""The three maintenance tools.

    get_equipment_history(equipment_id)
    get_open_tickets(equipment_id)
    create_ticket(equipment_id, title, description, approval_token)

The protocol layer is deliberately thin and separate (`server.py`). Everything
that matters for safety lives here, so it can be tested without speaking MCP.

Two rules hold for every call:

**The caller's claims are never trusted.** A call carries a `user_id`; the
server looks that user's groups up in its own tables. An agent that could
assert its own permissions would make the check decorative — and the brief is
explicit that authorisation is verified server-side at each call (§6).

**Refusals are indistinguishable.** An unknown equipment and a forbidden one
return the same refusal, so the tool cannot be used to enumerate what exists.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from maintidoc.contracts import ApprovalToken, TicketDraft
from maintidoc.mcp_server.approvals import ApprovalError, ApprovalSigner
from maintidoc.mcp_server.models import Intervention, Ticket, ToolCallRecord
from maintidoc.mcp_server.repository import MaintenanceRepository

REFUSAL = "Equipement inconnu ou acces non autorise"


class ToolError(Exception):
    """A tool call that must not proceed."""


class AccessRefused(ToolError):
    pass


@dataclass(frozen=True, slots=True)
class CreatedTicket:
    ticket: Ticket
    created: bool  # False when an idempotent replay returned the existing one


class MaintenanceService:
    def __init__(
        self,
        repository: MaintenanceRepository,
        signer: ApprovalSigner,
        *,
        history_limit: int = 10,
    ) -> None:
        self.repository = repository
        self.signer = signer
        self.history_limit = history_limit

    # --- authorisation ----------------------------------------------------

    def _authorise(self, tool: str, user_id: str, equipment_id: str):
        """Resolve the equipment and check the user may see it.

        Returns the equipment, or raises. The audit line is written either
        way: a refusal is exactly what an investigation needs to see.
        """
        equipment = self.repository.equipment(equipment_id)
        groups = self.repository.groups_of(user_id)
        allowed = equipment is not None and equipment.acl_group in groups

        self.repository.record(
            ToolCallRecord(
                tool=tool,
                user_id=user_id,
                equipment_id=equipment_id,
                allowed=allowed,
                reason="" if allowed else REFUSAL,
            )
        )
        if not allowed:
            # Same message whether it does not exist or is not permitted.
            raise AccessRefused(REFUSAL)
        return equipment

    # --- read-only tools --------------------------------------------------

    def get_equipment_history(
        self, *, user_id: str, equipment_id: str, limit: int | None = None
    ) -> list[Intervention]:
        self._authorise("get_equipment_history", user_id, equipment_id)
        return self.repository.interventions(
            equipment_id, limit or self.history_limit
        )

    def get_open_tickets(self, *, user_id: str, equipment_id: str) -> list[Ticket]:
        self._authorise("get_open_tickets", user_id, equipment_id)
        return self.repository.open_tickets(equipment_id)

    # --- the sensitive action ---------------------------------------------

    def create_ticket(
        self,
        *,
        user_id: str,
        draft: TicketDraft,
        approval: ApprovalToken | None,
        now: datetime | None = None,
    ) -> CreatedTicket:
        """Create a ticket, only against a valid single-use approval.

        Order matters: authorisation first, then the approval. Checking the
        token first would let an unauthorised caller learn whether a given
        approval is valid.
        """
        self._authorise("create_ticket", user_id, draft.equipment_id)

        if approval is None:
            self._refuse("create_ticket", user_id, draft.equipment_id,
                         "Aucune approbation humaine fournie")
            raise ApprovalError("Aucune approbation humaine fournie")

        try:
            self.signer.verify(approval, user_id=user_id, draft=draft, now=now)
        except ApprovalError as exc:
            self._refuse("create_ticket", user_id, draft.equipment_id, str(exc))
            raise

        signature = approval.token
        existing = self.repository.ticket_by_idempotency_key(approval.idempotency_key)
        if existing is not None:
            # Same approved request arriving twice: return what was created,
            # do not create a second ticket.
            return CreatedTicket(existing, created=False)

        if self.repository.approval_was_consumed(signature):
            self._refuse("create_ticket", user_id, draft.equipment_id,
                         "Approbation deja utilisee")
            raise ApprovalError("Approbation deja utilisee : rejeu refuse")

        ticket = Ticket(
            ticket_id=f"TCK-{uuid.uuid4().hex[:10].upper()}",
            equipment_id=draft.equipment_id,
            title=draft.title,
            description=draft.description,
            created_by=user_id,
            created_at=now or datetime.now(UTC),
            idempotency_key=approval.idempotency_key,
        )
        stored = self.repository.insert_ticket(ticket)
        created = stored.ticket_id == ticket.ticket_id

        if created and not self.repository.consume_approval(
            signature, user_id, stored.ticket_id
        ):
            # Lost a race: another caller consumed it first.
            raise ApprovalError("Approbation deja utilisee : rejeu refuse")

        return CreatedTicket(stored, created=created)

    def _refuse(self, tool: str, user_id: str, equipment_id: str, reason: str) -> None:
        self.repository.record(
            ToolCallRecord(
                tool=tool, user_id=user_id, equipment_id=equipment_id,
                allowed=False, reason=reason,
            )
        )

    # --- for the human approval step --------------------------------------

    def prepare_approval(
        self, *, user_id: str, draft: TicketDraft, now: datetime | None = None
    ) -> ApprovalToken:
        """Issue a token for a draft the user has just been shown.

        Authorisation is checked here too: there is no point presenting an
        approval screen for equipment the user may not touch.
        """
        self._authorise("prepare_approval", user_id, draft.equipment_id)
        return self.signer.issue(user_id=user_id, draft=draft, now=now)
