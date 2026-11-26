# Fiche de limites — MaintiDoc

Livrable 5 du cahier des charges. À lire **avant** toute utilisation, et à
joindre à toute démonstration.

Version 0.1.0 · corpus 2026.11 · rédigée le 20 novembre 2026.

---

## 1. Ce que MaintiDoc est

Une **aide à la lecture documentaire**. Il retrouve des passages et des figures
dans un corpus autorisé, en cite les références, et prépare un brouillon de
ticket que seul un humain peut valider.

## 2. Ce que MaintiDoc n'est pas

- Il **ne pose aucun diagnostic**. Une réponse sourcée reste une citation de
  document, pas une conclusion technique.
- Il **ne pilote aucun équipement** et n'émet aucune commande.
- Il **ne remplace ni les procédures de sécurité, ni un technicien habilité**.
- Il **ne crée aucun ticket** de sa propre initiative.

## 3. Périmètre couvert

| Élément | Valeur |
|---|---|
| Familles d'équipement | 2 — baies SAN, onduleurs |
| Modèles | 10, **tous fictifs** |
| Documents | 41 (1 manuel d'origine, 40 générés) |
| Éléments indexés | 317 |
| Langue | français |

**Tous les équipements, tickets, interventions et utilisateurs sont fictifs.**
Aucune donnée réelle, aucune donnée personnelle.

## 4. Ce qui a été mesuré

| Propriété | Résultat |
|---|---|
| Contrôle d'accès documentaire | 20/20 scénarios, **aucune fuite** |
| Confinement des injections (3 canaux) | 20/20 |
| Citation inventée détectée | oui (contrôle structurel) |
| Citation réelle non soutenante détectée | oui (contrôle sémantique) |
| Rejeu / expiration / contenu modifié d'une approbation | refusés |
| Création sans approbation humaine | impossible (interruption du graphe) |
| Budgets (3 MCP, 6 modèle, 60 s) | appliqués par compteurs |

## 5. Ce qui n'a PAS été mesuré

Ces points sont **ouverts**, pas acquis. Les présenter comme validés serait
faux.

| Point | Pourquoi | Ce qu'il faut pour conclure |
|---|---|---|
| Qualité et fidélité des réponses | aucun VLM exécuté (Ollama absent) | installer Ollama, relancer l'évaluation |
| Obéissance à une injection | idem — seul le confinement est mesuré | idem |
| Taux de citations soutenues (cible ≥ 90 %) | idem | idem |
| Gain du reranking fin | entraînement jamais lancé | mémoire suffisante, puis `training train` |
| `docker compose up` | aucun moteur Docker disponible | construire et démarrer la pile |
| CI GitHub Actions | jamais exécutée | pousser sur un dépôt GitHub |
| OCR réel | binaire Tesseract non fonctionnel sur la machine | réinstaller Tesseract |

Les métriques de recherche publiées (Recall@5 0,43) proviennent du **mode
allégé, BM25 seul**. Ce n'est pas la performance du système complet.

## 6. Limites techniques connues

- **Discrimination du français dans l'index visuel.** Deux requêtes françaises
  sans rapport obtiennent 0,977 de similarité cosinus, contre 0,787 pour la
  paire anglaise équivalente. Les figures sont donc atteintes principalement
  par leurs légendes, dans l'index texte.
- **Mémoire.** BGE-M3 et le reranker ne tiennent pas ensemble sur la machine de
  développement (`MemoryError`). Le VLM est plus gros encore.
- **Latence.** Le reranking de 20 candidats prend ~18 s sur CPU, avant même
  l'appel au modèle. La cible p95 < 30 s du cahier des charges n'est pas tenable
  dans cette configuration.
- **Politique d'accès du corpus.** Un groupe partagé `maintenance` couvre 259
  des 317 éléments : un technicien qui le détient lit les deux familles. Seuls
  les documents `responsable_maintenance` sont réellement cloisonnés.
- **PostgreSQL remplacé par SQLite**, MinIO par le système de fichiers local.
- **Vérificateur sémantique lexical** disponible en repli : approximation
  faible, qui s'abstient plutôt que de trancher.

## 7. Conditions d'utilisation

1. Ne jamais présenter une réponse comme sourcée si `is_sourced` est faux.
2. Ne jamais appliquer le tableau d'indicateurs d'un modèle à un autre : dix
   modèles emploient la même formule pour dix significations différentes.
3. Toute création de ticket passe par une lecture humaine du contenu exact.
4. Les secrets (`MAINTIDOC_JWT_SECRET`, mots de passe) sont fournis par
   l'environnement. Un secret vide empêche le démarrage.
5. Le corpus étant fictif, aucune réponse ne doit être appliquée à un
   équipement réel.

## 8. Ce que l'absence d'échec ne prouve pas

Le jeu de test compte 150 scénarios. Qu'aucun ne tombe en échec **ne garantit
pas** la sécurité du système : il garantit que ces 150 cas se comportent comme
attendu, dans cette configuration, avec ce corpus. Les cas difficiles
demandent une analyse manuelle, et les vérifications non exécutées ne comptent
pour rien.
