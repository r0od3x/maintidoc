"""Generateur de corpus synthetique (outil de developpement, phase 2).

Produit de vrais PDF pour deux familles d'equipement, avec des significations
d'indicateurs volontairement divergentes d'un modele a l'autre : c'est ce qui
rend la discrimination necessaire a la recherche documentaire.
"""

from maintidoc.synth.catalog import ALL_MODELS, EquipmentModel, LedCode
from maintidoc.synth.generate import (
    GeneratedDocument,
    generate_corpus,
    write_facts,
    write_manifest,
)

__all__ = [
    "ALL_MODELS",
    "EquipmentModel",
    "GeneratedDocument",
    "LedCode",
    "generate_corpus",
    "write_facts",
    "write_manifest",
]
