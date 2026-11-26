"""Fictitious maintenance data, aligned with the synthetic corpus.

Equipment ids and access groups match `maintidoc.synth`, so the access control
tested in phase 4 (documents) and here (tools) is the same policy rather than
two unrelated ones that happen to agree.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from maintidoc.mcp_server.models import Equipment, Intervention, Ticket, TicketStatus
from maintidoc.mcp_server.repository import MaintenanceRepository
from maintidoc.synth.catalog import ALL_MODELS

USERS = {
    "tech-san": ("maintenance", "maintenance_san"),
    "tech-nrj": ("maintenance", "maintenance_energie"),
    "chef-maint": ("maintenance", "maintenance_san", "responsable_maintenance"),
    "visiteur": (),  # no group at all: every call must be refused
}

SUMMARIES = (
    "Controle periodique de facade",
    "Verification des journaux de supervision",
    "Analyse d'un avertissement signale par la ronde",
    "Controle apres coupure programmee",
)
OUTCOMES = (
    "Aucune anomalie retenue",
    "Avertissement confirme, dossier transmis au niveau 2",
    "Ecart documentaire corrige dans l'inventaire",
)


def equipment_id_for(label: str) -> str:
    return label.replace(" ", "-").upper()


def seed(repository: MaintenanceRepository, *, seed_value: int = 20261120) -> int:
    """Populate equipment, users, interventions and a couple of open tickets."""
    rng = random.Random(seed_value)

    for user_id, groups in USERS.items():
        repository.grant(user_id, groups)

    now = datetime.now(UTC)
    count = 0
    for model in ALL_MODELS:
        equipment_id = equipment_id_for(model.label)
        # One access group per equipment, taken from the corpus policy.
        repository.add_equipment(
            Equipment(
                equipment_id=equipment_id,
                label=model.label,
                family=model.family,
                acl_group=model.acl_groups[-1],
            )
        )
        count += 1
        for index in range(3):
            repository.add_intervention(
                Intervention(
                    intervention_id=f"{model.code}-INT-{index + 1:03d}",
                    equipment_id=equipment_id,
                    performed_at=now - timedelta(days=rng.randint(5, 300)),
                    summary=rng.choice(SUMMARIES),
                    outcome=rng.choice(OUTCOMES),
                )
            )

    first = equipment_id_for(ALL_MODELS[0].label)
    repository.insert_ticket(
        Ticket(
            ticket_id="TCK-SEED00001",
            equipment_id=first,
            title="Avertissement thermique a confirmer",
            description=(
                "Voyant orange fixe signale lors de la ronde. Verification "
                "documentaire effectuee, confirmation sur site attendue."
            ),
            status=TicketStatus.OPEN,
            created_by="chef-maint",
        )
    )
    return count
