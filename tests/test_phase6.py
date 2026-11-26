"""MCP maintenance server: authorisation, approval, idempotency, audit.

Everything is driven through `MaintenanceService` and the tool router, not
through the MCP transport: the safety properties live in the service, and
testing them there keeps them independent of the protocol.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from maintidoc.contracts import TicketDraft
from maintidoc.mcp_server import (
    TOOL_DEFINITIONS,
    AccessRefused,
    ApprovalError,
    ApprovalSigner,
    MaintenanceRepository,
    MaintenanceService,
    MaintenanceToolRouter,
    seed,
)
from maintidoc.mcp_server.seed import equipment_id_for

SAN = equipment_id_for("SAN-LAB 2000")
UPS = equipment_id_for("UPS-LAB 500")
NOW = datetime(2026, 11, 20, 10, 0, tzinfo=UTC)


@pytest.fixture
def repository() -> MaintenanceRepository:
    repo = MaintenanceRepository()
    seed(repo)
    yield repo
    repo.close()


@pytest.fixture
def service(repository: MaintenanceRepository) -> MaintenanceService:
    return MaintenanceService(repository, ApprovalSigner("secret-de-test", ttl_seconds=900))


def _draft(title: str = "Avertissement thermique a verifier") -> TicketDraft:
    return TicketDraft(
        equipment_id=SAN,
        title=title,
        description="Voyant orange fixe constate lors de la ronde documentaire.",
    )


# --- authorisation, re-checked server side --------------------------------


def test_a_technician_reads_only_their_own_equipment(service: MaintenanceService):
    assert service.get_equipment_history(user_id="tech-san", equipment_id=SAN)
    with pytest.raises(AccessRefused):
        service.get_equipment_history(user_id="tech-san", equipment_id=UPS)


def test_a_user_without_groups_is_refused_everything(service: MaintenanceService):
    for call in (service.get_equipment_history, service.get_open_tickets):
        with pytest.raises(AccessRefused):
            call(user_id="visiteur", equipment_id=SAN)


def test_the_caller_cannot_assert_its_own_permissions(
    service: MaintenanceService, repository: MaintenanceRepository
):
    """Groups come from the server's tables, never from the call.

    There is no argument through which a caller could claim a group: the only
    way to gain access is for the server's own data to grant it.
    """
    with pytest.raises(AccessRefused):
        service.get_open_tickets(user_id="tech-nrj", equipment_id=SAN)

    repository.grant("tech-nrj", ["maintenance_san"])
    assert service.get_open_tickets(user_id="tech-nrj", equipment_id=SAN) is not None


def test_unknown_and_forbidden_equipment_are_indistinguishable(service: MaintenanceService):
    """Otherwise the tool enumerates what exists."""
    with pytest.raises(AccessRefused) as forbidden:
        service.get_open_tickets(user_id="tech-san", equipment_id=UPS)
    with pytest.raises(AccessRefused) as missing:
        service.get_open_tickets(user_id="tech-san", equipment_id="N-EXISTE-PAS")
    assert str(forbidden.value) == str(missing.value)


# --- the acceptance criterion ---------------------------------------------


def test_a_replayed_approval_is_refused(service: MaintenanceService):
    """First creation succeeds; the same token used again for new content is
    refused rather than creating a second ticket."""
    draft = _draft()
    approval = service.prepare_approval(user_id="tech-san", draft=draft, now=NOW)
    first = service.create_ticket(
        user_id="tech-san", draft=draft, approval=approval, now=NOW
    )
    assert first.created

    # Same signature, different idempotency key: a genuine replay attempt.
    forged = approval.model_copy(update={"idempotency_key": "autre-cle-0001"})
    with pytest.raises(ApprovalError):
        service.create_ticket(
            user_id="tech-san", draft=draft, approval=forged, now=NOW
        )


def test_an_expired_approval_is_refused(service: MaintenanceService):
    draft = _draft()
    approval = service.prepare_approval(user_id="tech-san", draft=draft, now=NOW)
    later = NOW + timedelta(hours=2)
    with pytest.raises(ApprovalError, match="expiree"):
        service.create_ticket(
            user_id="tech-san", draft=draft, approval=approval, now=later
        )


def test_an_edited_ticket_invalidates_the_approval(service: MaintenanceService):
    """The user approved exact words; anything else needs approving again."""
    approved = _draft()
    approval = service.prepare_approval(user_id="tech-san", draft=approved, now=NOW)
    edited = _draft(title="Remplacement du controleur A")

    with pytest.raises(ApprovalError, match="contenu du ticket a change"):
        service.create_ticket(
            user_id="tech-san", draft=edited, approval=approval, now=NOW
        )


# --- other approval failures ----------------------------------------------


def test_creation_without_any_approval_is_refused(service: MaintenanceService):
    with pytest.raises(ApprovalError, match="Aucune approbation"):
        service.create_ticket(
            user_id="tech-san", draft=_draft(), approval=None, now=NOW
        )


def test_another_user_cannot_use_an_approval(service: MaintenanceService):
    draft = _draft()
    approval = service.prepare_approval(user_id="tech-san", draft=draft, now=NOW)
    # chef-maint is allowed on this equipment, but did not approve this draft.
    with pytest.raises(ApprovalError, match="emise pour"):
        service.create_ticket(
            user_id="chef-maint", draft=draft, approval=approval, now=NOW
        )


def test_a_forged_signature_is_refused(service: MaintenanceService):
    draft = _draft()
    approval = service.prepare_approval(user_id="tech-san", draft=draft, now=NOW)
    tampered = approval.model_copy(update={"token": "v1.deadbeef.0" * 2})
    with pytest.raises(ApprovalError):
        service.create_ticket(
            user_id="tech-san", draft=draft, approval=tampered, now=NOW
        )


def test_a_token_from_another_secret_is_refused(repository: MaintenanceRepository):
    """An approval minted elsewhere must not be accepted here."""
    attacker = MaintenanceService(repository, ApprovalSigner("mauvais-secret"))
    legitimate = MaintenanceService(repository, ApprovalSigner("secret-de-test"))
    draft = _draft()
    approval = attacker.prepare_approval(user_id="tech-san", draft=draft, now=NOW)
    with pytest.raises(ApprovalError, match="Signature"):
        legitimate.create_ticket(
            user_id="tech-san", draft=draft, approval=approval, now=NOW
        )


def test_an_empty_secret_is_rejected_at_startup():
    """A blank secret would make every token forgeable."""
    with pytest.raises(ValueError, match="Secret"):
        ApprovalSigner("")


def test_authorisation_is_checked_before_the_approval(service: MaintenanceService):
    """Otherwise an unauthorised caller learns whether a token is valid."""
    draft = TicketDraft(
        equipment_id=UPS,
        title="Tentative hors perimetre",
        description="Cette demande porte sur un equipement non autorise.",
    )
    with pytest.raises(AccessRefused):
        service.create_ticket(
            user_id="tech-san", draft=draft, approval=None, now=NOW
        )


# --- idempotency ----------------------------------------------------------


def test_the_same_approved_request_creates_one_ticket(service: MaintenanceService):
    draft = _draft()
    approval = service.prepare_approval(user_id="tech-san", draft=draft, now=NOW)

    first = service.create_ticket(user_id="tech-san", draft=draft, approval=approval, now=NOW)
    second = service.create_ticket(user_id="tech-san", draft=draft, approval=approval, now=NOW)

    assert first.created is True
    assert second.created is False
    assert first.ticket.ticket_id == second.ticket.ticket_id

    open_tickets = service.get_open_tickets(user_id="tech-san", equipment_id=SAN)
    assert sum(t.ticket_id == first.ticket.ticket_id for t in open_tickets) == 1


def test_a_new_approval_creates_a_distinct_ticket(service: MaintenanceService):
    draft = _draft()
    first = service.create_ticket(
        user_id="tech-san", draft=draft,
        approval=service.prepare_approval(user_id="tech-san", draft=draft, now=NOW),
        now=NOW,
    )
    second = service.create_ticket(
        user_id="tech-san", draft=draft,
        approval=service.prepare_approval(user_id="tech-san", draft=draft, now=NOW),
        now=NOW,
    )
    assert first.ticket.ticket_id != second.ticket.ticket_id


# --- audit ----------------------------------------------------------------


def test_every_call_is_audited_including_refusals(
    service: MaintenanceService, repository: MaintenanceRepository
):
    service.get_open_tickets(user_id="tech-san", equipment_id=SAN)
    with pytest.raises(AccessRefused):
        service.get_open_tickets(user_id="tech-nrj", equipment_id=SAN)

    trail = repository.audit_trail()
    assert any(entry.allowed for entry in trail)
    assert any(not entry.allowed for entry in trail)


def test_the_audit_records_no_content_or_token(
    service: MaintenanceService, repository: MaintenanceRepository
):
    """Metadata and access events only, per the brief's logging rule."""
    draft = _draft()
    approval = service.prepare_approval(user_id="tech-san", draft=draft, now=NOW)
    service.create_ticket(user_id="tech-san", draft=draft, approval=approval, now=NOW)

    serialised = json.dumps(
        [e.model_dump(mode="json") for e in repository.audit_trail()], ensure_ascii=False
    )
    assert draft.description not in serialised
    assert approval.token not in serialised


# --- the tool surface -----------------------------------------------------


def test_the_three_tools_of_the_brief_are_exposed(service: MaintenanceService):
    names = {definition["name"] for definition in TOOL_DEFINITIONS}
    assert names == {"get_equipment_history", "get_open_tickets", "create_ticket"}


def test_identity_is_not_a_tool_argument():
    """A model that could choose its own user_id would defeat every check."""
    for definition in TOOL_DEFINITIONS:
        properties = definition["inputSchema"]["properties"]
        assert "user_id" not in properties
        assert definition["inputSchema"]["additionalProperties"] is False


def test_the_router_reports_a_refusal_as_data(service: MaintenanceService):
    router = MaintenanceToolRouter(service, user_id="tech-san")
    payload = json.loads(router.call("get_open_tickets", {"equipment_id": UPS}))
    assert "error" in payload


def test_the_router_rejects_an_unknown_tool(service: MaintenanceService):
    router = MaintenanceToolRouter(service, user_id="tech-san")
    assert "error" in json.loads(router.call("exfiltrate_everything", {}))


def test_the_router_creates_through_the_full_approval_path(service: MaintenanceService):
    router = MaintenanceToolRouter(service, user_id="tech-san")
    draft = _draft()
    # Issued at real time: the router does not take a clock, so a token minted
    # at a fixed past instant would legitimately be expired by now.
    approval = service.prepare_approval(user_id="tech-san", draft=draft)
    payload = json.loads(
        router.call(
            "create_ticket",
            {
                "equipment_id": draft.equipment_id,
                "title": draft.title,
                "description": draft.description,
                "approval_token": approval.model_dump_json(),
            },
        )
    )
    assert payload["created"] is True
    assert payload["ticket"]["ticket_id"].startswith("TCK-")


def test_the_router_refuses_creation_without_a_token(service: MaintenanceService):
    router = MaintenanceToolRouter(service, user_id="tech-san")
    payload = json.loads(
        router.call(
            "create_ticket",
            {
                "equipment_id": SAN,
                "title": "Sans approbation",
                "description": "Tentative de creation directe sans validation humaine.",
                "approval_token": "",
            },
        )
    )
    assert "error" in payload
