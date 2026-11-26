"""Authentication and role-based access control.

JWT via PyJWT rather than hand-rolled signing. The brief makes security a
deliverable, and the classic JWT failures — `alg: none`, HS/RS confusion,
non-constant-time comparison — are exactly what a vetted library already
handles.

Identity is established **once**, at the edge, and turned into a `Principal`.
Nothing downstream re-derives permissions from a request body: the groups the
retriever and the MCP server see come from the verified token, never from
anything the caller typed.

Users are an in-memory fixture here. A real deployment swaps `UserDirectory`
for OIDC; the rest of the code does not change, because it only ever sees a
`Principal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import jwt
from pydantic import BaseModel, ConfigDict

from maintidoc.contracts import Principal

ALGORITHM = "HS256"
ISSUER = "maintidoc"


class AuthError(Exception):
    """Authentication failed. Deliberately uninformative to the caller."""


class TokenResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    groups: list[str]


@dataclass(frozen=True, slots=True)
class DemoUser:
    user_id: str
    password: str
    groups: tuple[str, ...]


# Fictitious accounts, matching the corpus and MCP access groups so one policy
# governs documents and tools alike.
DEMO_USERS: tuple[DemoUser, ...] = (
    DemoUser("tech-san", "san", ("maintenance", "maintenance_san")),
    DemoUser("tech-nrj", "nrj", ("maintenance", "maintenance_energie")),
    DemoUser(
        "chef-maint", "chef",
        ("maintenance", "maintenance_san", "responsable_maintenance"),
    ),
    DemoUser("visiteur", "visiteur", ()),  # no group: everything is refused
)


@dataclass
class UserDirectory:
    users: dict[str, DemoUser] = field(default_factory=dict)

    @classmethod
    def demo(cls) -> UserDirectory:
        return cls({user.user_id: user for user in DEMO_USERS})

    def authenticate(self, user_id: str, password: str) -> DemoUser:
        import hmac

        user = self.users.get(user_id)
        # Compare even when the user is unknown, so a wrong name and a wrong
        # password take the same time and are indistinguishable.
        reference = user.password if user else "\x00invalid"
        ok = hmac.compare_digest(reference, password)
        if not user or not ok:
            raise AuthError("Identifiants invalides")
        return user


@dataclass(frozen=True, slots=True)
class TokenService:
    secret: str
    ttl_seconds: int = 3600

    def __post_init__(self) -> None:
        if not self.secret:
            raise ValueError(
                "Secret JWT vide : configurer MAINTIDOC_JWT_SECRET avant de demarrer"
            )

    def issue(self, user: DemoUser, *, now: datetime | None = None) -> TokenResponse:
        moment = now or datetime.now(UTC)
        payload = {
            "iss": ISSUER,
            "sub": user.user_id,
            "groups": list(user.groups),
            "iat": int(moment.timestamp()),
            "exp": int((moment + timedelta(seconds=self.ttl_seconds)).timestamp()),
        }
        return TokenResponse(
            access_token=jwt.encode(payload, self.secret, algorithm=ALGORITHM),
            expires_in=self.ttl_seconds,
            user_id=user.user_id,
            groups=list(user.groups),
        )

    def principal(self, token: str) -> Principal:
        """Verify a token and turn it into a `Principal`.

        The algorithm is pinned: accepting whatever the header claims is the
        classic JWT forgery route.
        """
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[ALGORITHM],
                issuer=ISSUER,
                options={"require": ["exp", "sub", "iss"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("Jeton expire") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError("Jeton invalide") from exc

        groups = payload.get("groups") or []
        if not isinstance(groups, list):
            raise AuthError("Jeton malforme")
        return Principal(
            user_id=str(payload["sub"]), groups=frozenset(str(g) for g in groups)
        )


def require_groups(principal: Principal, required: set[str]) -> bool:
    """RBAC check for routes needing a specific role, beyond document ACLs."""
    return bool(principal.groups.intersection(required))
