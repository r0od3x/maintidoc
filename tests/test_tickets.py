"""Approval is bound to the user and to the exact approved content."""

from datetime import UTC, datetime, timedelta

from maintidoc.contracts import ApprovalToken, TicketDraft

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


def _draft(title: str = "Voyant orange controleur A") -> TicketDraft:
    return TicketDraft(
        equipment_id="SAN-LAB-2000",
        title=title,
        description="Avertissement constate sur le panneau avant, a verifier sur site.",
    )


def _token(draft: TicketDraft, user_id: str = "tech-01") -> ApprovalToken:
    return ApprovalToken(
        token="signature-serveur-a-remplacer-phase-6",
        user_id=user_id,
        content_hash=draft.content_hash(),
        idempotency_key="req-0001-abcd",
        issued_at=NOW,
        expires_at=NOW + timedelta(minutes=15),
    )


def test_content_hash_is_stable():
    assert _draft().content_hash() == _draft().content_hash()


def test_approval_authorizes_the_exact_draft():
    draft = _draft()
    assert _token(draft).authorizes(user_id="tech-01", draft=draft, now=NOW)


def test_edited_draft_invalidates_the_approval():
    token = _token(_draft())
    edited = _draft(title="Remplacement du controleur A")
    assert not token.authorizes(user_id="tech-01", draft=edited, now=NOW)


def test_another_user_cannot_reuse_the_approval():
    draft = _draft()
    assert not _token(draft).authorizes(user_id="tech-99", draft=draft, now=NOW)


def test_expired_approval_is_refused():
    draft = _draft()
    later = NOW + timedelta(minutes=30)
    assert not _token(draft).authorizes(user_id="tech-01", draft=draft, now=later)
