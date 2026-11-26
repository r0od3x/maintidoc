"""Phase 8 — API FastAPI, authentification et RBAC."""

from maintidoc.api.app import create_app
from maintidoc.api.deps import Runtime
from maintidoc.api.security import (
    DEMO_USERS,
    AuthError,
    DemoUser,
    TokenService,
    UserDirectory,
)

__all__ = [
    "DEMO_USERS",
    "AuthError",
    "DemoUser",
    "Runtime",
    "TokenService",
    "UserDirectory",
    "create_app",
]
