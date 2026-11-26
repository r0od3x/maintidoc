"""Issuing and verifying approval tokens.

The brief's requirements for the one sensitive action (§6):

  * the user sees and approves the **exact** content before creation;
  * approval is bound to that user and that content;
  * any modification forces a new approval;
  * an idempotency key prevents duplicate creation.

The token is an HMAC over those bindings, signed with a server secret. It is
not a bearer capability the agent can mint: the agent can carry a token, it
cannot forge one. Verification is done by the server on every call, never by
the caller.

Single use is enforced separately, in the repository, because replay
protection needs state and a signature does not carry any.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from maintidoc.contracts import ApprovalToken, TicketDraft

VERSION = "v1"


class ApprovalError(Exception):
    """The token is absent, forged, expired, replayed, or for other content."""


@dataclass(frozen=True, slots=True)
class ApprovalSigner:
    """Signs and verifies approval tokens with a server-side secret."""

    secret: str
    ttl_seconds: int = 900

    def __post_init__(self) -> None:
        if not self.secret:
            # A blank secret would make every token forgeable, so refuse to
            # start rather than appear to protect anything.
            raise ValueError("Secret d'approbation vide : configurer MAINTIDOC_JWT_SECRET")

    def _signature(self, user_id: str, content_hash: str, idempotency_key: str,
                   expires_at: datetime, nonce: str) -> str:
        payload = "|".join(
            [VERSION, user_id, content_hash, idempotency_key,
             expires_at.isoformat(), nonce]
        )
        return hmac.new(
            self.secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def issue(
        self,
        *,
        user_id: str,
        draft: TicketDraft,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> ApprovalToken:
        moment = now or datetime.now(UTC)
        expires_at = moment + timedelta(seconds=self.ttl_seconds)
        key = idempotency_key or secrets.token_hex(8)
        nonce = secrets.token_hex(8)
        content_hash = draft.content_hash()
        signature = self._signature(user_id, content_hash, key, expires_at, nonce)
        return ApprovalToken(
            token=f"{VERSION}.{nonce}.{signature}",
            user_id=user_id,
            content_hash=content_hash,
            idempotency_key=key,
            issued_at=moment,
            expires_at=expires_at,
        )

    def verify(
        self,
        token: ApprovalToken,
        *,
        user_id: str,
        draft: TicketDraft,
        now: datetime | None = None,
    ) -> None:
        """Raise `ApprovalError` unless everything binds. Returns None on success."""
        moment = now or datetime.now(UTC)

        parts = token.token.split(".")
        if len(parts) != 3 or parts[0] != VERSION:
            raise ApprovalError("Jeton d'approbation malforme")
        _version, nonce, signature = parts

        expected = self._signature(
            token.user_id, token.content_hash, token.idempotency_key,
            token.expires_at, nonce,
        )
        # Constant-time: a timing oracle on the signature would let a token be
        # forged byte by byte.
        if not hmac.compare_digest(expected, signature):
            raise ApprovalError("Signature d'approbation invalide")

        if token.user_id != user_id:
            raise ApprovalError(
                f"Approbation emise pour {token.user_id}, utilisee par {user_id}"
            )
        if token.content_hash != draft.content_hash():
            raise ApprovalError(
                "Le contenu du ticket a change depuis l'approbation : "
                "une nouvelle validation humaine est requise"
            )
        if token.is_expired(moment):
            raise ApprovalError("Approbation expiree")
