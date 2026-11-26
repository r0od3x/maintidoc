"""Who is asking, and what they may read.

The brief orders the retrieval pipeline deliberately (§4.2): identity and
permissions come *first*, before any search. Filtering afterwards is not the
same thing — a document the user may not read must never be scored at all,
because even its rank leaks that a relevant document exists.

`groups` is a frozenset so a principal cannot be mutated after the access
decision has been taken on it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Principal(BaseModel):
    """An authenticated user and the access groups granted to them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    groups: frozenset[str] = frozenset()
    # Optional narrowing by the caller, never a widening.
    families: frozenset[str] | None = None

    @property
    def can_read_anything(self) -> bool:
        """False for a principal with no group: deny by default."""
        return bool(self.groups)

    def may_read(self, acl_groups: tuple[str, ...]) -> bool:
        return bool(self.groups.intersection(acl_groups))
