"""Content pools for the synthetic corpus.

Every model is fictitious. The design goal is not realism for its own sake but
*discriminability*: the same phrase must mean different things on different
equipment, so that retrieving the right document is the hard part.

The clearest case is the steady orange indicator. It appears in all ten models
and means something different in each. A system that answers "que signifie le
voyant orange fixe ?" without first pinning down the model cannot be right by
accident — which is exactly the hard negative the brief asks the reranker to
learn (§5: "distinguer une alarme de temperature d'un voyant similaire
appartenant a un autre equipement").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LedCode:
    colour: str
    state: str
    meaning: str
    action: str

    @property
    def phrase(self) -> str:
        return f"voyant {self.colour} {self.state}"


@dataclass(frozen=True, slots=True)
class EquipmentModel:
    family: str
    label: str  # "SAN-LAB 2200"
    code: str  # "SAN-2200"
    summary: str
    components: tuple[str, ...]
    leds: tuple[LedCode, ...]
    acl_groups: tuple[str, ...]

    @property
    def slug(self) -> str:
        return self.code.lower().replace("-", "_")


SHARED_GREEN = LedCode(
    "vert",
    "fixe",
    "fonctionnement nominal, aucune action requise",
    "aucune action ; consigner l'etat lors de la ronde",
)
SHARED_RED = LedCode(
    "rouge",
    "fixe",
    "defaut bloquant necessitant une intervention habilitee",
    "escalader immediatement ; ne rien manipuler sans accord",
)

SAN_MODELS: tuple[EquipmentModel, ...] = (
    EquipmentModel(
        family="san",
        label="SAN-LAB 2000",
        code="SAN-2000",
        summary="baie de stockage a deux controleurs et deux fabrics Fibre Channel",
        components=("controleur A", "controleur B", "enceinte disques", "ports FC"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "avertissement thermique sur le controleur A",
                "verifier la ventilation et la temperature ambiante du local",
            ),
            LedCode(
                "orange",
                "clignotant",
                "groupe de disques en mode degrade",
                "consulter l'inventaire des disques avant toute action",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance", "maintenance_san"),
    ),
    EquipmentModel(
        family="san",
        label="SAN-LAB 2200",
        code="SAN-2200",
        summary="baie de stockage a quatre ports hotes et cache protege",
        components=("controleur A", "controleur B", "module de cache", "ports hotes"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "reconstruction du groupe de disques en cours",
                "ne pas extraire de disque tant que la reconstruction dure",
            ),
            LedCode(
                "bleu",
                "clignotant",
                "module de cache en cours de synchronisation",
                "attendre la fin de la synchronisation avant tout arret",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance", "maintenance_san"),
    ),
    EquipmentModel(
        family="san",
        label="SAN-LAB 3000",
        code="SAN-3000",
        summary="baie haute densite avec double alimentation et supervision etendue",
        components=("controleur A", "controleur B", "alimentations redondantes", "fond de panier"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "chemin Fibre Channel degrade sur une fabric",
                "comparer les WWPN et le zoning aux configurations approuvees",
            ),
            LedCode(
                "orange",
                "clignotant",
                "micrologiciel en cours de mise a jour",
                "ne pas couper l'alimentation pendant la mise a jour",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance", "maintenance_san"),
    ),
    EquipmentModel(
        family="san",
        label="SAN-RACK 120",
        code="SAN-R120",
        summary="tiroir de stockage d'extension raccorde a une baie principale",
        components=("module d'extension", "cables SAS", "alimentation redondante"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "alimentation redondante indisponible",
                "verifier la presence des deux cordons et leur depart electrique",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance", "maintenance_san"),
    ),
    EquipmentModel(
        family="san",
        label="SAN-RACK 240",
        code="SAN-R240",
        summary="tiroir d'extension double capacite avec supervision deportee",
        components=("module d'extension", "cables SAS", "carte de supervision"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "micrologiciel en attente de validation",
                "planifier la validation avec l'equipe stockage",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance_san", "responsable_maintenance"),
    ),
)

UPS_MODELS: tuple[EquipmentModel, ...] = (
    EquipmentModel(
        family="onduleur",
        label="UPS-LAB 500",
        code="UPS-500",
        summary="onduleur monophase a double conversion pour baie informatique",
        components=("module batterie", "redresseur", "onduleur", "bypass manuel"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "fonctionnement sur batterie, reseau absent",
                "verifier l'alimentation amont et l'autonomie restante",
            ),
            LedCode(
                "orange",
                "clignotant",
                "test de batterie en cours",
                "attendre la fin du test avant d'interpreter les autres voyants",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance", "maintenance_energie"),
    ),
    EquipmentModel(
        family="onduleur",
        label="UPS-LAB 800",
        code="UPS-800",
        summary="onduleur monophase avec module de ventilation remplacable",
        components=("module batterie", "module de ventilation", "bypass automatique"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "defaut de ventilation du module de puissance",
                "controler les grilles et signaler sans ouvrir le capot",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance", "maintenance_energie"),
    ),
    EquipmentModel(
        family="onduleur",
        label="UPS-RACK 1500",
        code="UPS-R1500",
        summary="onduleur rackable triphase avec supervision reseau",
        components=("armoire batterie", "carte reseau", "bypass automatique"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "batterie en fin de vie annoncee par la supervision",
                "planifier le remplacement, ne pas intervenir sur la batterie",
            ),
            LedCode(
                "bleu",
                "fixe",
                "supervision reseau active",
                "aucune action ; verifier la remontee des alarmes",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance_energie", "responsable_maintenance"),
    ),
    EquipmentModel(
        family="onduleur",
        label="UPS-RACK 3000",
        code="UPS-R3000",
        summary="onduleur rackable forte puissance a bypass automatique",
        components=("armoire batterie", "bypass automatique", "filtre d'entree"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "bypass automatique actif, charge non protegee",
                "retablir le mode normal apres analyse de la cause",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance", "maintenance_energie"),
    ),
    EquipmentModel(
        family="onduleur",
        label="ONDULEUR-LAB 300",
        code="OND-300",
        summary="onduleur compact de laboratoire a autonomie reduite",
        components=("batterie interne", "prise de sortie protegee"),
        leds=(
            SHARED_GREEN,
            LedCode(
                "orange",
                "fixe",
                "surcharge de sortie detectee",
                "retirer une charge non prioritaire et reverifier",
            ),
            SHARED_RED,
        ),
        acl_groups=("maintenance", "maintenance_energie"),
    ),
)

ALL_MODELS: tuple[EquipmentModel, ...] = SAN_MODELS + UPS_MODELS

FAMILY_LABELS = {"san": "baie de stockage SAN", "onduleur": "onduleur"}

PRECHECKS = (
    "Verifier l'identifiant de l'equipement et son numero de serie.",
    "Confirmer la version du micrologiciel aupres de la supervision.",
    "Controler les autorisations d'acces au local et a l'equipement.",
    "S'assurer qu'une sauvegarde ou une protection equivalente est en place.",
)

STANDARD_STEPS = (
    "Enregistrer la date, l'heure et l'identifiant de l'equipement concerne.",
    "Photographier le panneau de facade sans deplacer l'equipement.",
    "Relever l'etat de chaque voyant et le comparer au tableau de reference.",
    "Consulter les journaux de supervision sur la periode concernee.",
    "Verifier la coherence entre le symptome observe et la documentation.",
    "Preparer un compte rendu avant toute demande d'intervention.",
)

EXCLUSIONS = (
    "Aucun remplacement a chaud n'est decrit dans ce document.",
    "Aucune modification de configuration en production n'est autorisee ici.",
    "Le diagnostic reste a la charge d'un technicien habilite.",
)


# --- Supplementary content pools -----------------------------------------
#
# These exist to give each document enough substance that retrieval has to
# discriminate *within* a document, not only between documents. A one-page
# manual makes Recall@5 look good for the wrong reason.

GLOSSARY = {
    "san": (
        ("LUN", "Volume logique expose aux hotes ; identifiant et taille consignes"),
        ("Zoning", "Regles de visibilite des ports dans une fabric"),
        ("WWPN", "Identifiant de port Fibre Channel, compare a l'inventaire"),
        ("MPIO", "Gestion des chemins multiples et basculement selon configuration"),
        ("Fabric", "Reseau Fibre Channel commute reliant hotes et baie"),
    ),
    "onduleur": (
        ("Bypass", "Chemin direct reseau-charge, sans protection onduleur"),
        ("Autonomie", "Duree de maintien estimee a charge constante"),
        ("Double conversion", "Conversion permanente alternatif-continu-alternatif"),
        ("Charge", "Puissance appelee par les equipements proteges"),
        ("Redresseur", "Etage d'entree alimentant l'onduleur et la batterie"),
    ),
}

SUPERVISION_POINTS = {
    "san": (
        ("Etat des controleurs", "toutes les 5 minutes", "journal de supervision"),
        ("Etat des chemins", "toutes les 5 minutes", "console de stockage"),
        ("Capacite des volumes", "toutes les heures", "rapport de capacite"),
        ("Temperature d'entree", "toutes les 15 minutes", "sonde de baie"),
    ),
    "onduleur": (
        ("Tension d'entree", "toutes les minutes", "carte de supervision"),
        ("Niveau de charge", "toutes les 5 minutes", "afficheur de facade"),
        ("Autonomie estimee", "toutes les 15 minutes", "carte de supervision"),
        ("Temperature batterie", "toutes les 15 minutes", "sonde interne"),
    ),
}

ESCALATION = (
    ("Niveau 1", "Technicien de ronde", "Observation, photographie, compte rendu"),
    ("Niveau 2", "Equipe specialisee", "Analyse des journaux et correlation"),
    ("Niveau 3", "Responsable de maintenance", "Decision d'intervention et validation"),
)

FAQ = {
    "san": (
        (
            "Un voyant orange impose-t-il un arret immediat ?",
            "Non. L'arret n'est decide qu'apres analyse des journaux et validation "
            "par un technicien habilite. Le voyant signale un etat, pas une consigne.",
        ),
        (
            "Peut-on comparer deux baies de modeles differents ?",
            "Les tableaux d'indicateurs ne sont pas transposables d'un modele a "
            "l'autre. Confirmer le modele avant toute lecture du tableau.",
        ),
        (
            "Que faire si la plaque signaletique est illisible ?",
            "Demander une nouvelle photographie. Sans identification du modele, "
            "aucune interpretation documentaire n'est valide.",
        ),
    ),
    "onduleur": (
        (
            "Le passage sur batterie est-il une panne ?",
            "Pas necessairement. Il traduit l'absence du reseau amont. La cause "
            "se determine par les journaux, pas par le voyant seul.",
        ),
        (
            "Peut-on intervenir sur une batterie signalee en fin de vie ?",
            "Non. Le remplacement releve d'une intervention habilitee et planifiee.",
        ),
        (
            "Un bypass actif protege-t-il la charge ?",
            "Non. En bypass, la charge est alimentee sans protection onduleur.",
        ),
    ),
}

INTERVENTION_SUBJECTS = (
    "Controle periodique de facade",
    "Verification des journaux de supervision",
    "Analyse d'un avertissement signale par la ronde",
    "Controle apres coupure programmee",
    "Revue documentaire avant audit",
)

INTERVENTION_OUTCOMES = (
    "Aucune anomalie retenue ; compte rendu archive",
    "Avertissement confirme ; dossier transmis au niveau 2",
    "Ecart documentaire corrige dans l'inventaire",
    "Observation close apres verification des journaux",
)
