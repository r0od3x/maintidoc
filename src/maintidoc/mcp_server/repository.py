"""Storage for the maintenance service.

SQLite rather than PostgreSQL, because no server is available here. The
interface is what matters: `MaintenanceRepository` is the seam PostgreSQL
slots into in phase 10, and the SQL is plain enough to port.

Two tables exist purely for safety properties:

  * `consumed_approvals` makes an approval single-use. A signature alone
    cannot prevent replay, since it carries no state.
  * the unique index on `idempotency_key` makes duplicate creation impossible
    at the storage layer, not merely unlikely at the application layer.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from maintidoc.mcp_server.models import (
    Equipment,
    Intervention,
    Ticket,
    TicketStatus,
    ToolCallRecord,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS equipment (
    equipment_id TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    family       TEXT NOT NULL,
    acl_group    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    user_id   TEXT NOT NULL,
    group_name TEXT NOT NULL,
    PRIMARY KEY (user_id, group_name)
);
CREATE TABLE IF NOT EXISTS interventions (
    intervention_id TEXT PRIMARY KEY,
    equipment_id    TEXT NOT NULL,
    performed_at    TEXT NOT NULL,
    summary         TEXT NOT NULL,
    outcome         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id       TEXT PRIMARY KEY,
    equipment_id    TEXT NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    status          TEXT NOT NULL,
    created_by      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    idempotency_key TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS tickets_idempotency
    ON tickets(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE TABLE IF NOT EXISTS consumed_approvals (
    token_signature TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    ticket_id       TEXT NOT NULL,
    consumed_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tool         TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    equipment_id TEXT,
    allowed      INTEGER NOT NULL,
    reason       TEXT NOT NULL,
    at           TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MaintenanceRepository:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    # --- reference data ---------------------------------------------------

    def add_equipment(self, equipment: Equipment) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO equipment VALUES (?, ?, ?, ?)",
            (equipment.equipment_id, equipment.label, equipment.family, equipment.acl_group),
        )
        self.connection.commit()

    def equipment(self, equipment_id: str) -> Equipment | None:
        row = self.connection.execute(
            "SELECT * FROM equipment WHERE equipment_id = ?", (equipment_id,)
        ).fetchone()
        return Equipment(**dict(row)) if row else None

    def grant(self, user_id: str, groups: Iterable[str]) -> None:
        self.connection.executemany(
            "INSERT OR IGNORE INTO users VALUES (?, ?)",
            [(user_id, group) for group in groups],
        )
        self.connection.commit()

    def groups_of(self, user_id: str) -> frozenset[str]:
        """The server's own view of a user's groups.

        Deliberately not taken from the caller: an agent that could assert its
        own permissions would make the whole check decorative.
        """
        rows = self.connection.execute(
            "SELECT group_name FROM users WHERE user_id = ?", (user_id,)
        ).fetchall()
        return frozenset(row["group_name"] for row in rows)

    # --- interventions ----------------------------------------------------

    def add_intervention(self, intervention: Intervention) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO interventions VALUES (?, ?, ?, ?, ?)",
            (
                intervention.intervention_id,
                intervention.equipment_id,
                intervention.performed_at.isoformat(),
                intervention.summary,
                intervention.outcome,
            ),
        )
        self.connection.commit()

    def interventions(self, equipment_id: str, limit: int = 10) -> list[Intervention]:
        rows = self.connection.execute(
            "SELECT * FROM interventions WHERE equipment_id = ? "
            "ORDER BY performed_at DESC LIMIT ?",
            (equipment_id, limit),
        ).fetchall()
        return [
            Intervention(
                intervention_id=row["intervention_id"],
                equipment_id=row["equipment_id"],
                performed_at=datetime.fromisoformat(row["performed_at"]),
                summary=row["summary"],
                outcome=row["outcome"],
            )
            for row in rows
        ]

    # --- tickets ----------------------------------------------------------

    def _ticket_from_row(self, row: sqlite3.Row) -> Ticket:
        return Ticket(
            ticket_id=row["ticket_id"],
            equipment_id=row["equipment_id"],
            title=row["title"],
            description=row["description"],
            status=TicketStatus(row["status"]),
            created_by=row["created_by"],
            created_at=datetime.fromisoformat(row["created_at"]),
            idempotency_key=row["idempotency_key"],
        )

    def open_tickets(self, equipment_id: str) -> list[Ticket]:
        rows = self.connection.execute(
            "SELECT * FROM tickets WHERE equipment_id = ? AND status != ? "
            "ORDER BY created_at DESC",
            (equipment_id, TicketStatus.CLOSED.value),
        ).fetchall()
        return [self._ticket_from_row(row) for row in rows]

    def ticket_by_idempotency_key(self, key: str) -> Ticket | None:
        row = self.connection.execute(
            "SELECT * FROM tickets WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return self._ticket_from_row(row) if row else None

    def insert_ticket(self, ticket: Ticket) -> Ticket:
        try:
            self.connection.execute(
                "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ticket.ticket_id,
                    ticket.equipment_id,
                    ticket.title,
                    ticket.description,
                    ticket.status.value,
                    ticket.created_by,
                    ticket.created_at.isoformat(),
                    ticket.idempotency_key,
                ),
            )
            self.connection.commit()
        except sqlite3.IntegrityError:
            # The unique index fired: another call already created this ticket.
            self.connection.rollback()
            if ticket.idempotency_key:
                existing = self.ticket_by_idempotency_key(ticket.idempotency_key)
                if existing:
                    return existing
            raise
        return ticket

    # --- replay protection ------------------------------------------------

    def consume_approval(self, signature: str, user_id: str, ticket_id: str) -> bool:
        """Record an approval as used. False if it had already been used."""
        try:
            self.connection.execute(
                "INSERT INTO consumed_approvals VALUES (?, ?, ?, ?)",
                (signature, user_id, ticket_id, _now()),
            )
            self.connection.commit()
        except sqlite3.IntegrityError:
            self.connection.rollback()
            return False
        return True

    def approval_was_consumed(self, signature: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM consumed_approvals WHERE token_signature = ?", (signature,)
        ).fetchone()
        return row is not None

    # --- audit ------------------------------------------------------------

    def record(self, entry: ToolCallRecord) -> None:
        self.connection.execute(
            "INSERT INTO audit (tool, user_id, equipment_id, allowed, reason, at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                entry.tool,
                entry.user_id,
                entry.equipment_id,
                int(entry.allowed),
                entry.reason,
                entry.at.isoformat(),
            ),
        )
        self.connection.commit()

    def audit_trail(self, limit: int = 50) -> list[ToolCallRecord]:
        rows = self.connection.execute(
            "SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            ToolCallRecord(
                tool=row["tool"],
                user_id=row["user_id"],
                equipment_id=row["equipment_id"],
                allowed=bool(row["allowed"]),
                reason=row["reason"],
                at=datetime.fromisoformat(row["at"]),
            )
            for row in rows
        ]
