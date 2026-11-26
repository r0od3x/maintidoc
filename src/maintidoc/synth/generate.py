"""Synthetic corpus generation.

Produces real PDFs — ruled tables, vector schematics, raster front panels,
captioned figures, running footers — plus the manifest entries that declare
them and a ground-truth file recording where each fact was written.

That last part matters more than it looks. The generator knows the page a LED
table landed on, so the evaluation of phase 9 can be built from facts rather
than from someone re-reading forty documents and annotating them by hand.

Everything is seeded: the same seed yields byte-identical PDFs, so the
ingestion stays idempotent and element ids stay stable across regenerations.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from maintidoc.synth.catalog import (
    ALL_MODELS,
    ESCALATION,
    EXCLUSIONS,
    FAMILY_LABELS,
    FAQ,
    GLOSSARY,
    INTERVENTION_OUTCOMES,
    INTERVENTION_SUBJECTS,
    PRECHECKS,
    STANDARD_STEPS,
    SUPERVISION_POINTS,
    EquipmentModel,
)
from maintidoc.synth.layout import DocumentBuilder, TableSpec

LED_RGB = {
    "vert": (46, 160, 67),
    "orange": (214, 134, 22),
    "rouge": (200, 50, 45),
    "bleu": (47, 109, 200),
}

DOCUMENT_KINDS = ("manuel", "alarmes", "procedure", "note")


@dataclass
class FactRecord:
    """Where a checkable fact was written. Feeds the phase 9 test set."""

    document_id: str
    model: str
    family: str
    kind: str
    page: int
    subject: str
    expected: str


@dataclass
class GeneratedDocument:
    document_id: str
    path: Path
    model: str
    family: str
    kind: str
    title: str
    pages: int
    acl_groups: tuple[str, ...]
    facts: list[FactRecord] = field(default_factory=list)

    def manifest_entry(self, relative_to: Path) -> dict:
        return {
            "path": self.path.relative_to(relative_to).as_posix(),
            "document_id": self.document_id,
            "document_version": "1.0",
            "equipment_family": self.family,
            "acl_groups": list(self.acl_groups),
            "title": self.title,
            "language": "fr",
        }


# --- drawing --------------------------------------------------------------


def front_panel_png(model: EquipmentModel, rng: random.Random) -> bytes:
    """Raster front panel, so the corpus exercises the embedded-image path."""
    width, height = 760, 240
    image = Image.new("RGB", (width, height), (238, 238, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [20, 30, width - 20, height - 30], fill=(60, 63, 70), outline=(30, 30, 34), width=3
    )
    draw.text((36, 46), model.label, fill=(235, 235, 235))

    for index, led in enumerate(model.leds):
        x = 48 + index * 96
        y = 120
        draw.ellipse([x, y, x + 34, y + 34], fill=LED_RGB.get(led.colour, (150, 150, 150)))
        draw.text((x, y + 42), led.colour, fill=(225, 225, 225))

    for slot in range(6):
        x = 430 + slot * 48
        draw.rectangle([x, 96, x + 38, 180], fill=(88, 92, 100), outline=(40, 40, 44))
        if rng.random() < 0.3:
            draw.ellipse([x + 14, 164, x + 24, 174], fill=LED_RGB["orange"])

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def draw_topology(model: EquipmentModel):
    """Vector schematic: no embedded image, so the extractor must render it."""

    def _draw(page: pymupdf.Page, rect: pymupdf.Rect) -> None:
        width, height = rect.width, rect.height
        left = pymupdf.Rect(
            rect.x0 + 10, rect.y0 + height * 0.18, rect.x0 + width * 0.26, rect.y0 + height * 0.46
        )
        right = pymupdf.Rect(
            rect.x1 - width * 0.26, rect.y0 + height * 0.18, rect.x1 - 10, rect.y0 + height * 0.46
        )
        centre = pymupdf.Rect(
            rect.x0 + width * 0.37, rect.y0 + height * 0.58, rect.x1 - width * 0.37, rect.y1 - 12
        )
        for box, label in ((left, "Chemin A"), (right, "Chemin B"), (centre, model.label)):
            page.draw_rect(box, color=(0.15, 0.15, 0.2), width=1.1)
            page.insert_text((box.x0 + 8, box.y0 + 16), label, fontsize=8)
        for box in (left, right):
            page.draw_line(
                pymupdf.Point((box.x0 + box.x1) / 2, box.y1),
                pymupdf.Point((centre.x0 + centre.x1) / 2, centre.y0),
                color=(0.25, 0.25, 0.3),
                width=0.9,
            )
        page.draw_line(
            pymupdf.Point(rect.x0 + 10, rect.y0 + 8),
            pymupdf.Point(rect.x1 - 10, rect.y0 + 8),
            color=(0.7, 0.7, 0.75),
            width=0.6,
        )

    return _draw


# --- document bodies ------------------------------------------------------


def _led_table(model: EquipmentModel) -> TableSpec:
    return TableSpec(
        headers=["Voyant", "Etat", "Signification", "Action documentaire"],
        rows=[[led.colour, led.state, led.meaning, led.action] for led in model.leds],
    )


def _facts_for_leds(
    document_id: str, model: EquipmentModel, kind: str, page: int
) -> list[FactRecord]:
    return [
        FactRecord(
            document_id=document_id,
            model=model.label,
            family=model.family,
            kind=kind,
            page=page,
            subject=f"{led.phrase} sur {model.label}",
            expected=led.meaning,
        )
        for led in model.leds
    ]


def _glossary_table(model: EquipmentModel) -> TableSpec:
    return TableSpec(
        headers=["Terme", "Definition documentaire"],
        rows=[[term, definition] for term, definition in GLOSSARY[model.family]],
    )


def _supervision_table(model: EquipmentModel) -> TableSpec:
    return TableSpec(
        headers=["Point de mesure", "Frequence", "Source"],
        rows=[list(row) for row in SUPERVISION_POINTS[model.family]],
    )


def _history_table(model: EquipmentModel, rng: random.Random) -> TableSpec:
    rows = []
    for _ in range(4):
        rows.append(
            [
                f"{model.code}-INT-{rng.randint(100, 999)}",
                f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}",
                rng.choice(INTERVENTION_SUBJECTS),
                rng.choice(INTERVENTION_OUTCOMES),
            ]
        )
    rows.sort(key=lambda row: row[1])
    return TableSpec(headers=["Reference", "Date", "Objet", "Conclusion documentaire"], rows=rows)


def _faq_section(builder: DocumentBuilder, model: EquipmentModel) -> None:
    builder.heading("Questions frequentes")
    for question, answer in FAQ[model.family]:
        builder.heading(question, size=9.5)
        builder.paragraph(answer)


def _escalation_section(builder: DocumentBuilder) -> None:
    builder.heading("Niveaux d'escalade")
    builder.paragraph(
        "L'escalade suit trois niveaux. Un assistant documentaire se situe en "
        "amont du niveau 1 : il prepare l'information, il ne decide pas."
    )
    builder.table(
        TableSpec(
            headers=["Niveau", "Acteur", "Perimetre"],
            rows=[list(row) for row in ESCALATION],
        )
    )


def build_manual(model: EquipmentModel, document_id: str, rng: random.Random) -> DocumentBuilder:
    builder = DocumentBuilder(
        title=f"Manuel de maintenance {model.label}",
        footer_label=document_id,
    )
    builder.cover(
        subtitle=f"{FAMILY_LABELS[model.family].capitalize()} — documentation de reference",
        lines=[
            f"Equipement : {model.label} (fictif)",
            f"Perimetre : {model.summary}",
            f"Reference : {document_id}  |  Revision 1.0",
        ],
    )

    builder.heading("1. Presentation et perimetre")
    builder.paragraph(
        f"Le {model.label} est {model.summary}. Ce manuel decrit la lecture des indicateurs "
        f"et les controles documentaires associes. Il ne decrit aucune operation de "
        f"remplacement et ne se substitue pas a la notice du constructeur."
    )
    builder.paragraph(
        "Les ensembles suivants sont mentionnes dans ce document : "
        + ", ".join(model.components)
        + "."
    )
    builder.heading("Regles prealables", size=10.5)
    builder.bullets(list(PRECHECKS))

    builder.figure(
        f"facade du {model.label} et position des indicateurs. Les couleurs sont "
        f"conventionnelles et ne doivent pas etre transposees a un autre modele.",
        height=180.0,
        image=front_panel_png(model, rng),
    )

    builder.heading("2. Indicateurs et interpretation")
    builder.paragraph(
        f"Le tableau ci-dessous est la reference unique pour le {model.label}. "
        f"La signification d'une couleur depend du modele : un meme voyant peut "
        f"designer un etat different sur un autre equipement."
    )
    led_page = builder.table(_led_table(model))

    builder.figure(
        f"chemins redondants du {model.label}. Le schema illustre des liaisons "
        f"logiques et non un cablage reel.",
        height=200.0,
        draw=draw_topology(model),
    )

    builder.heading("3. Supervision et points de mesure")
    builder.paragraph(
        f"La supervision du {model.label} repose sur les points ci-dessous. Les "
        f"valeurs relevees servent de reference documentaire : elles ne "
        f"declenchent aucune action automatique."
    )
    builder.table(_supervision_table(model))
    builder.paragraph(
        "Un ecart ponctuel entre un releve de supervision et l'etat affiche en "
        "facade ne constitue pas une anomalie. La correlation se fait sur la "
        "periode complete, journaux a l'appui."
    )

    builder.heading("4. Procedure de controle non intrusive")
    builder.paragraph(
        "La procedure suivante n'implique aucune manipulation du materiel. "
        "Elle prepare un compte rendu exploitable par un technicien habilite."
    )
    builder.bullets([f"{index}. {step}" for index, step in enumerate(STANDARD_STEPS, start=1)])
    builder.paragraph(
        "Chaque etape est consignee. Une etape non realisee est signalee comme "
        "telle plutot qu'omise : un compte rendu incomplet mais honnete reste "
        "exploitable, un compte rendu lacunaire ne l'est pas."
    )

    _escalation_section(builder)

    builder.heading("5. Historique documentaire (fictif)")
    builder.paragraph(
        f"Les interventions ci-dessous sont fictives et illustrent la tracabilite "
        f"attendue pour le {model.label}."
    )
    builder.table(_history_table(model, rng))

    builder.heading("6. Glossaire")
    builder.table(_glossary_table(model))

    _faq_section(builder, model)

    builder.heading("7. Exclusions")
    builder.bullets(list(EXCLUSIONS))

    builder.facts = _facts_for_leds(document_id, model, "manuel", led_page)  # type: ignore[attr-defined]
    return builder


def build_alarm_sheet(
    model: EquipmentModel, document_id: str, rng: random.Random
) -> DocumentBuilder:
    builder = DocumentBuilder(
        title=f"Fiche d'alarmes {model.label}",
        footer_label=document_id,
        footer_note="Fiche de reference fictive",
    )
    builder.new_page()
    builder.heading(f"Fiche d'alarmes — {model.label}")
    builder.paragraph(
        f"Cette fiche recense les alarmes documentees pour le {model.label}. "
        f"Elle complete le manuel de maintenance et ne s'applique a aucun autre modele."
    )
    led_page = builder.table(_led_table(model))

    builder.heading("Codes evenementiels")
    codes = [
        [f"{model.code}-{rng.randint(100, 999)}", component, "consigner et comparer au journal"]
        for component in model.components
    ]
    builder.table(
        TableSpec(headers=["Code", "Ensemble concerne", "Suite documentaire"], rows=codes)
    )

    builder.heading("Ensembles concernes")
    for component in model.components:
        builder.heading(component.capitalize(), size=9.5)
        builder.paragraph(
            f"Le {component} du {model.label} est suivi par la supervision. Un "
            f"evenement portant sur cet ensemble est consigne avec son code, son "
            f"horodatage et l'etat des indicateurs au moment du releve. Aucune "
            f"manipulation n'est decrite ici."
        )

    _escalation_section(builder)

    builder.heading("Limites d'interpretation")
    builder.paragraph(
        "Une ressemblance visuelle avec un autre equipement ne vaut pas "
        "confirmation. En cas de doute sur le modele, demander une photographie "
        "de la plaque signaletique avant toute conclusion."
    )
    builder.paragraph(
        f"Ce tableau est propre au {model.label}. Les modeles voisins de la meme "
        f"famille possedent leur propre fiche : les transposer conduirait a un "
        f"contresens de diagnostic."
    )
    builder.table(_glossary_table(model))
    builder.facts = _facts_for_leds(document_id, model, "alarmes", led_page)  # type: ignore[attr-defined]
    return builder


def build_procedure(model: EquipmentModel, document_id: str, rng: random.Random) -> DocumentBuilder:
    builder = DocumentBuilder(
        title=f"Procedure de controle {model.label}",
        footer_label=document_id,
        footer_note="Procedure pedagogique fictive",
    )
    builder.new_page()
    builder.heading(f"Procedure {model.code}-PROC-01")
    builder.paragraph(
        f"Controle periodique non intrusif du {model.label}. Aucune etape de cette "
        f"procedure ne modifie la configuration de l'equipement."
    )
    builder.heading("Prerequis", size=10.5)
    builder.bullets(list(PRECHECKS))
    builder.heading("Deroulement", size=10.5)
    builder.bullets([f"{index}. {step}" for index, step in enumerate(STANDARD_STEPS, start=1)])

    builder.figure(
        f"circuit documentaire applique au {model.label} : observation, comparaison "
        f"au tableau de reference, puis validation humaine avant toute action.",
        height=170.0,
        draw=draw_topology(model),
    )
    builder.heading("Criteres d'arret")
    builder.bullets(
        [
            "Plaque signaletique illisible : demander une nouvelle photographie.",
            "Modele non confirme : ne pas appliquer le tableau d'un autre equipement.",
            "Voyant rouge : escalader sans manipulation.",
            "Documents contradictoires : signaler l'ecart sans trancher.",
        ]
    )

    builder.heading("Detail des etapes")
    for index, step in enumerate(STANDARD_STEPS, start=1):
        builder.heading(f"Etape {index}", size=9.5)
        builder.paragraph(
            f"{step} Cette etape est documentaire : elle n'implique aucune "
            f"manipulation du {model.label} ni de son environnement electrique. "
            f"Le resultat est consigne meme lorsqu'il est negatif."
        )

    _escalation_section(builder)

    builder.heading("Compte rendu attendu")
    builder.paragraph(
        "Le compte rendu mentionne l'identifiant de l'equipement, la date et "
        "l'heure du controle, l'etat de chaque indicateur, les references des "
        "documents consultes et les points restes incertains."
    )
    builder.table(
        TableSpec(
            headers=["Rubrique", "Contenu attendu"],
            rows=[
                ["Identification", f"Modele {model.label} et numero de serie releve"],
                ["Observations", "Etat de chaque indicateur au moment du releve"],
                ["Sources", "References documentaires consultees"],
                ["Incertitudes", "Points non confirmes, explicitement listes"],
            ],
        )
    )
    builder.facts = []  # type: ignore[attr-defined]
    return builder


def build_note(model: EquipmentModel, document_id: str, rng: random.Random) -> DocumentBuilder:
    builder = DocumentBuilder(
        title=f"Note technique {model.label}",
        footer_label=document_id,
        footer_note="Note technique fictive",
    )
    builder.new_page()
    builder.heading(f"Note technique — {model.label}")
    builder.paragraph(
        f"Rappel : {model.summary}. Cette note precise un point frequemment confondu "
        f"entre modeles d'une meme famille."
    )
    orange = next(
        (led for led in model.leds if led.colour == "orange" and led.state == "fixe"), None
    )
    if orange is not None:
        builder.heading("Point de vigilance", size=10.5)
        note_page = builder.paragraph(
            f"Sur le {model.label}, le {orange.phrase} signifie : {orange.meaning}. "
            f"Cette signification est propre a ce modele. L'appliquer a un autre "
            f"equipement conduirait a un contresens de diagnostic."
        )
        builder.facts = [  # type: ignore[attr-defined]
            FactRecord(
                document_id=document_id,
                model=model.label,
                family=model.family,
                kind="note",
                page=note_page,
                subject=f"{orange.phrase} sur {model.label}",
                expected=orange.meaning,
            )
        ]
    else:
        builder.facts = []  # type: ignore[attr-defined]

    builder.figure(
        f"reperage des ensembles du {model.label} cites dans cette note.",
        height=160.0,
        image=front_panel_png(model, rng),
    )
    builder.heading("Modeles a ne pas confondre")
    builder.paragraph(
        f"Les modeles suivants appartiennent a la meme famille que le "
        f"{model.label} et presentent des indicateurs d'aspect identique. Leur "
        f"signification differe : consulter la fiche du modele concerne. Les "
        f"significations ne sont volontairement pas reproduites ici."
    )
    siblings = [
        other.label
        for other in ALL_MODELS
        if other.family == model.family and other.label != model.label
    ]
    builder.table(
        TableSpec(
            headers=["Modele voisin", "Document de reference"],
            rows=[[label, f"Fiche d'alarmes {label}"] for label in siblings],
        )
    )

    builder.heading("Reference croisee")
    builder.paragraph(
        f"Voir le manuel de maintenance du {model.label} pour le tableau complet "
        f"des indicateurs, et la fiche d'alarmes pour les codes evenementiels. En "
        f"cas de divergence entre deux documents, signaler la contradiction "
        f"plutot que de retenir l'un des deux."
    )
    _faq_section(builder, model)
    return builder


BUILDERS = {
    "manuel": build_manual,
    "alarmes": build_alarm_sheet,
    "procedure": build_procedure,
    "note": build_note,
}

TITLES = {
    "manuel": "Manuel de maintenance",
    "alarmes": "Fiche d'alarmes",
    "procedure": "Procedure de controle",
    "note": "Note technique",
}


# --- orchestration --------------------------------------------------------


def generate_corpus(
    output_dir: Path,
    *,
    seed: int = 20261119,
    kinds: tuple[str, ...] = DOCUMENT_KINDS,
    models: tuple[EquipmentModel, ...] = ALL_MODELS,
) -> list[GeneratedDocument]:
    generated: list[GeneratedDocument] = []
    for model in models:
        for kind in kinds:
            document_id = f"{model.code}-{kind.upper()[:4]}"
            # Seeded per document, so adding one does not perturb the others.
            rng = random.Random(f"{seed}:{document_id}")
            builder = BUILDERS[kind](model, document_id, rng)
            path = output_dir / model.family / f"{model.slug}_{kind}.pdf"
            pages = builder.save(path)
            generated.append(
                GeneratedDocument(
                    document_id=document_id,
                    path=path,
                    model=model.label,
                    family=model.family,
                    kind=kind,
                    title=f"{TITLES[kind]} {model.label}",
                    pages=pages,
                    acl_groups=model.acl_groups,
                    facts=list(getattr(builder, "facts", [])),
                )
            )
    return generated


def write_manifest(
    documents: list[GeneratedDocument],
    manifest_path: Path,
    corpus_root: Path,
    *,
    corpus_version: str,
    keep: list[dict] | None = None,
) -> None:
    """Rewrite the manifest, preserving any hand-declared entries."""
    import yaml

    entries = list(keep or []) + [d.manifest_entry(corpus_root) for d in documents]
    payload = {"corpus_version": corpus_version, "documents": entries}
    header = (
        "# Genere par `python -m maintidoc.synth`. Les entrees hors generateur\n"
        "# (manuel SAN-LAB 2000 d'origine) sont conservees en tete de liste.\n"
    )
    manifest_path.write_text(
        header + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_facts(documents: list[GeneratedDocument], path: Path) -> int:
    """Ground truth for phase 9: fact, equipment, and the page it was put on."""
    facts = [asdict(fact) for document in documents for fact in document.facts]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(facts)
