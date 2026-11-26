# MaintiDoc

Assistant documentaire **multimodal et agentique** pour la maintenance SAN.

Un technicien pose une question en français, éventuellement avec une photo.
MaintiDoc répond à partir de la documentation autorisée, **cite les pages et
figures réellement récupérées**, consulte l'historique d'interventions via MCP
et peut préparer un ticket — dont la création exige une **approbation humaine
explicite**.

> Aide documentaire uniquement. MaintiDoc ne pilote aucun équipement, ne
> garantit aucun diagnostic et ne remplace ni les procédures de sécurité ni un
> technicien habilité. Le corpus de démonstration est **fictif**.

Cahier des charges : [`docs/cahier_des_charges.pdf`](docs/cahier_des_charges.pdf) ·
Feuille de route : [`docs/ROADMAP.md`](docs/ROADMAP.md)

## État : 7 phases sur 10 — projet en cours

Le socle documentaire est construit et testé : le corpus est ingéré et indexé,
la récupération complète fonctionne, la génération est contrainte et ses
citations vérifiées, et l'agent orchestre le tout avec une validation humaine
avant toute action sensible. **Les trois dernières phases restent à faire.**

| Phase | Objet | État |
|---|---|---|
| 1 | Fondations, contrats, prompts versionnés | **fait** |
| 2 | Corpus et ingestion | **fait** |
| 3 | Indexation et stockage persistant | **fait** |
| 4 | Récupération (ACL → hybride → RRF → reranker) | **fait** |
| 5 | Génération sourcée et vérification des citations | **fait** (VLM non exécuté) |
| 6 | Serveur MCP et actions sensibles | **fait** |
| 7 | Orchestration LangGraph et validation humaine | **fait** |
| 8 | API, interface, authentification et RBAC | *à faire* |
| 9 | Évaluation et tests d'injection | *à faire* |
| 10 | Fine-tuning du reranker, MLOps et déploiement | *à faire* |

### Ce qui vient ensuite

- **Phase 8** — exposer le graphe derrière une API FastAPI (JWT, RBAC) et une
  interface Streamlit. Aujourd'hui l'agent s'utilise en ligne de commande.
- **Phase 9** — construire les ~150 scénarios du cahier des charges (§10) et
  mesurer : Recall@5, nDCG@5, MRR@10, validité des citations, résistance aux
  injections. Les défenses sont écrites ; **elles ne sont pas encore mesurées**.
- **Phase 10** — adapter le reranker sur des négatifs difficiles (le corpus est
  déjà construit pour cela : le *voyant orange fixe* signifie dix choses
  différentes selon le modèle), puis MLflow, Docker Compose et livraison.

## Démarrage

```bash
uv sync --group dev          # environnement reproductible (uv.lock)
uv run pytest -q             # 192 tests hors modèles, ~23 s
uv run ruff check .
cp .env.example .env         # puis renseigner les valeurs locales
```

Les tests qui chargent de vrais modeles sont marques `models` et **exclus par
defaut**. Ils se lancent un module a la fois :

```bash
uv run pytest -m models tests/test_retrieval_models.py   # 8 tests
uv run pytest -m models tests/test_phase4_models.py      # 4 tests
```

Les reunir dans un meme processus epuise la memoire (voir plus bas).

Aucun service externe n'est requis : contrats, configuration, prompts et
ingestion sont testables hors ligne.

### Corpus

Le corpus compte **41 documents sur deux familles d'equipement** (baies SAN et
onduleurs), soit 317 elements indexables :

| Source | Documents | Contenu |
|---|---|---|
| `corpus/raw/san_lab_2000/` | 1 | Manuel SAN-LAB 2000 d'origine, 9 pages |
| `corpus/raw/genere/` | 40 | Manuels, fiches d'alarmes, procédures et notes, générés |

Les 40 documents générés couvrent 10 modèles fictifs. Leurs tableaux
d'indicateurs sont **volontairement divergents** : le *voyant orange fixe*
existe sur les 10 modèles et y signifie 10 choses différentes. Répondre
correctement impose donc d'identifier le modèle d'abord — c'est le négatif
difficile que le cahier des charges demande d'apprendre au reranker (§5).

```bash
uv run python -m maintidoc.synth generate     # deterministe, graine fixe
```

La génération est reproductible à l'octet près : regénérer puis réingérer ne
produit aucun travail. Elle écrit aussi `eval/datasets/corpus_facts.json`, qui
enregistre pour chaque fait la page où il a été écrit — la vérité de terrain
de la phase 9 vient du générateur, pas d'une annotation manuelle de 40
documents.

### Ingestion du corpus

```bash
uv sync --extra ingestion
uv run python -m maintidoc.ingestion ingest          # idempotent
uv run python -m maintidoc.ingestion ingest --ocr    # si Tesseract est installe
uv run python -m maintidoc.ingestion stats
uv run python -m maintidoc.ingestion show SAN-MAN-001 --page 2
```

Les sorties vont dans `corpus/derived/` (ignoré par Git, entièrement
reconstructible). Relancer `ingest` sur un corpus inchangé ne fait rien : la
clé est le SHA-256 des octets source. Les identifiants d'élément restent
stables d'une exécution à l'autre, sans quoi toute citation déjà enregistrée
dans une évaluation deviendrait fausse.

## Structure

```
src/maintidoc/
  contracts/      modèles Pydantic partagés (voir ci-dessous)
  config.py       configuration et secrets, lus dans l'environnement
  prompts.py      chargement des prompts versionnés + empreinte SHA-256
  verification.py contrôle structurel des citations
  versions.py     empreinte de version d'une exécution
  ingestion/      phase 2 : manifeste, extraction, stockage, idempotence
  synth/          générateur de corpus synthétique (outil de développement)
  retrieval/      phases 3-4 : embeddings, Qdrant, BM25, fusion RRF, reranker
  generation/     phase 5 : appel VLM, vérification structurelle et sémantique
  mcp_server/     phase 6 : outils MCP, approbation, idempotence, audit
  agent/          phase 7 : graphe LangGraph, budgets, interruption humaine
prompts/          answer_v1.yaml, visual_analysis_v1.yaml,
                  evidence_check_v1.yaml, ticket_draft_v1.yaml
corpus/raw/       documents sources (manuel SAN-LAB 2000)
notebooks/        prototype d'origine, conservé comme référence
legacy/           application Streamlit du prototype, gelée
docs/             cahier des charges, feuille de route, schémas
```

## Les contrats

Tout transite par des modèles Pydantic, importables depuis
`maintidoc.contracts` :

| Modèle | Rôle |
|---|---|
| `Element` | unité indexée du corpus, avec provenance et `acl_groups` |
| `ElementRef` | identité citable : `document_id:pN:element_id` |
| `Evidence` / `EvidenceBundle` | ce qui a été récupéré **et transmis** au modèle |
| `AnswerV1` | réponse structurée (§7 du cahier des charges) |
| `TicketDraft` / `ApprovalToken` | action sensible liée à un utilisateur et à un contenu exact |
| `RunVersions` | modèle, prompts, reranker, index et corpus d'une exécution |

`Element.extraction` (ajouté en phase 2) distingue trois provenances :
`native` (couche texte ou image incorporée), `ocr` (texte reconnu) et `render`
(zone rendue pour une figure vectorielle). Du texte issu d'OCR n'a pas la même
fiabilité qu'une couche texte, et l'évaluation doit pouvoir les séparer.

`AnswerV1` remplace le schéma JSON écrit à la main dans le notebook. Le même
objet contraint la génération, valide la sortie et sert de corps de réponse à
l'API :

```python
from maintidoc.contracts import AnswerV1, answer_json_schema

result = client.chat(model=..., messages=..., format=answer_json_schema())
answer = AnswerV1.model_validate_json(result.message.content)
```

Trois règles sont imposées par le code, parce qu'un prompt ne peut pas les
garantir :

- `answered` sans citation → rejeté ;
- `needs_clarification` sans question de clarification → rejeté ;
- action proposée sur une réponse non étayée → rejeté.

## Vérification des citations

`maintidoc.verification.verify_structural` établit qu'une citation désigne un
élément **réellement récupéré et transmis**. Cela écarte les références
inventées.

Cela ne démontre **pas** que le passage cité soutient l'affirmation. L'objectif
de 90 % de citations soutenues est une propriété sémantique, vérifiée en
phase 5 (`prompts/evidence_check_v1.yaml`). Une réponse ne doit jamais être
présentée comme « sourcée » sur la seule foi du contrôle structurel.

## Décisions figées en phase 1

- **Base vectorielle : Qdrant.** Le cahier des charges indique Qdrant, un schéma
  du prototype indiquait PGVector. Qdrant est retenu, notamment pour le
  filtrage par `acl_groups` dans la charge utile.
- **Modèle vision-langage : `qwen2.5vl:7b` par défaut.** Le prototype
  référençait `qwen3.8:27b`, qui n'est pas un tag Ollama réel et n'est pas
  multimodal : les images jointes n'étaient pas vues par le modèle. Vérifier le
  tag disponible avec `ollama list`.
- **Encodeur visuel multilingue** (`jina-clip-v2`), afin qu'une question en
  français interroge directement les images. Le CLIP ViT-B/32 du prototype
  obligeait le technicien à saisir une requête en anglais.
- **Secrets exclusivement dans l'environnement**, jamais dans Git, les prompts
  ou les journaux (`SecretStr`, `.env` ignoré par Git).

### Indexation

```bash
uv sync --extra retrieval
uv run python -m maintidoc.retrieval build              # modèles réels
uv run python -m maintidoc.retrieval build --fake       # hors ligne, plomberie seule
uv run python -m maintidoc.retrieval stats
uv run python -m maintidoc.retrieval search "voyant orange sur le SAN-LAB 2000"
uv run python -m maintidoc.retrieval search "chemins redondants" --space visual
uv run python -m maintidoc.retrieval search "voyant orange" --groups maintenance_energie
```

Deux collections Qdrant, parce que les deux espaces n'ont ni la même dimension
ni le même modèle :

| Collection | Contenu | Modèle |
|---|---|---|
| `maintidoc_text` | texte, tableaux, **légendes de figures** | BGE-M3 |
| `maintidoc_visual` | figures et illustrations | CLIP ViT-B/32 + tour texte multilingue |

Une figure légendée est indexée **dans les deux** sous la même référence : elle
se trouve par sa légende comme par son contenu, et la fusion de la phase 4 les
réunit au lieu de compter la figure deux fois.

**Qdrant tourne en mode local** — même moteur, même API, sans serveur ni
Docker. L'URL d'un serveur reste utilisable (`--qdrant-url`) et sera la
configuration de la phase 10.

#### Choix de modèles et licences

| Rôle | Modèle | Licence |
|---|---|---|
| Texte | `BAAI/bge-m3` | MIT |
| Image | `sentence-transformers/clip-ViT-B-32` | Apache-2.0 |
| Requête visuelle | `sentence-transformers/clip-ViT-B-32-multilingual-v1` | Apache-2.0 |

La tour texte multilingue est alignée sur l'espace image de CLIP : **une
question en français interroge directement l'index visuel**. Le prototype
imposait une seconde zone de saisie en anglais, la tour texte de CLIP ViT-B/32
étant anglophone.

`jinaai/jina-clip-v2` obtient de meilleurs scores et tient dans un seul modèle,
mais sa licence CC-BY-NC-4.0 interdit l'usage commercial : retenu comme option,
pas comme défaut, la revue de licences du cahier des charges (§3) devant
trancher.

#### Résultats mesurés

Index réel : **316 points texte** (BGE-M3, 1024d) et **46 points visuels**
(CLIP, 512d) sur 41 documents, construits en 302 s sur CPU.

Ce qui fonctionne :

- une question française retrouve la bonne figure **par sa légende**, dans
  l'index texte, sans aucune ressaisie en anglais ;
- « Que signifie le voyant orange fixe sur le SAN-LAB 2000 ? » place un
  document SAN-LAB 2000 en **premier**, devant les neuf autres modèles qui
  emploient la même formule — la discrimination par modèle fonctionne ;
- une requête française et sa traduction anglaise sont alignées
  (cosinus 0,85 à 0,99).

#### Limite mesurée : discrimination du français dans l'index visuel

`clip-ViT-B-32-multilingual-v1` projette le français dans un cône étroit. Deux
requêtes françaises **sans rapport** obtiennent un cosinus de **0,977**, là où
la paire anglaise équivalente est à **0,787**. L'alignement entre langues est
correct ; c'est la séparation *à l'intérieur* du français qui est mauvaise.

Conséquence : interrogé seul, l'index visuel classe les requêtes françaises de
façon presque identique quelle que soit la question. En pratique les figures
sont atteintes par leurs légendes dans l'index texte, que BGE-M3 traite bien.

Trois suites possibles, à trancher :

1. accepter la limite — les légendes portent déjà la recherche de figures ;
2. passer à `jinaai/jina-clip-v2`, nettement meilleur en multilingue, si la
   licence CC-BY-NC-4.0 est acceptable pour le projet ;
3. compter sur le reranker de la phase 4 pour réparer l'ordre.

Un test verrouille cette mesure : si un changement de modèle améliore la
séparation du français, il échoue et la limite doit être retirée d'ici.

Une requête française reste par ailleurs parfois en échec sans reranking
(« Quel schéma montre le panneau avant et les voyants ? » remonte du texte de
supervision plutôt que FIG-01). C'est précisément ce que la phase 4 doit
corriger.

#### Traçabilité des index

Chaque construction ajoute une entrée à `corpus/derived/index_runs.jsonl` :
modèles, dimensions, versions de corpus et d'index, nombre de points, durée.
Sans ce journal, deux résultats d'évaluation ne sont pas comparables — une
variation de rappel pourrait venir du reranker, du corpus ou d'un modèle
d'embedding changé sans qu'on sache lequel.

Changer de modèle change la dimension : la collection existante est alors
refusée plutôt que corrompue, et une réindexation explicite est exigée
(`--recreate`).

### Récupération

```bash
uv run python -m maintidoc.retrieval ask "Que signifie le voyant orange fixe sur le SAN-LAB 2000 ?"
uv run python -m maintidoc.retrieval ask "..." --no-rerank          # baseline
uv run python -m maintidoc.retrieval ask "..." --groups maintenance_energie
```

Le pipeline suit l'ordre imposé par le cahier des charges (§4.2) :

```
identité et permissions
  → filtrage du corpus autorisé
  → recherche dense (BGE-M3) + lexicale (BM25) + visuelle (CLIP)
  → fusion RRF
  → reranking (bge-reranker-v2-m3)
  → dossier de preuves borné
```

La sortie est un `EvidenceBundle` — le contrat écrit en phase 1. La phase 5 le
consomme tel quel, et `verify_structural` contrôle déjà les citations contre
lui : les deux moitiés se rejoignent sans adaptateur.

**Le contrôle d'accès est décidé une fois, en tête, et poussé dans chaque
branche comme filtre.** Aucune branche ne peut l'élargir. Un document interdit
n'est jamais *scoré* : le filtrer après coup laisserait fuir, par le rang et
par la latence, l'existence d'un document pertinent. Un principal sans groupe
se voit refuser la recherche (`AccessDenied`) au lieu de tout recevoir.

#### Trois branches, pas deux

Le cahier des charges demande texte et visuel. Une branche **lexicale BM25**
est ajoutée parce que ce corpus se joue sur des identifiants exacts :
`SAN-LAB 2000` et `SAN-LAB 2200` diffèrent d'un chiffre et ne signifient pas la
même chose, or les embeddings denses sont précisément faibles là-dessus. La
fusion RRF absorbe une troisième branche sans coût.

#### Fusion au niveau élément

Le prototype fusionnait **par page**, ce qui se tenait pour un manuel de neuf
pages. Sur 41 documents c'est trop grossier : une page porte plusieurs éléments
sans rapport, et tous remontent parce qu'un seul correspond, saturant le budget
de preuves.

La page est conservée comme signal faible : un élément dont la page est
soutenue par une *autre* branche reçoit un petit bonus, calibré pour ne jamais
dépasser un véritable accord entre deux branches.

#### Pourquoi le reranker n'est pas optionnel

Sans lui, sur la question type du corpus, la fusion place une note
**SAN-RACK 240** en tête, le bon document SAN-LAB 2000 n'arrivant qu'en
troisième position. Dix modèles emploient la même formule ; seul un
cross-encoder, qui lit question et passage ensemble, tranche. C'est aussi la
référence sur laquelle la phase 10 mesurera son fine-tuning
(`--no-rerank` donne la baseline).

### Génération sourcée

```bash
uv run python -m maintidoc.generation check        # etat d'Ollama et du modele
uv run python -m maintidoc.generation answer "Que signifie le voyant orange ... ?"
uv run python -m maintidoc.generation answer "..." --lexical-check   # juge lexical
```

La génération est **contrainte par le schéma `AnswerV1`**, pas demandée
poliment dans le prompt. Les preuves sont présentées en blocs étiquetés
`[SOURCE ...]`, et les références autorisées listées à part : un document qui
contient « ignore les instructions précédentes » se lit alors comme une
citation, non comme une consigne. C'est une défense en profondeur, pas une
garantie — la phase 9 la met à l'épreuve.

#### Deux vérifications, car aucune ne remplace l'autre

| Contrôle | Question posée | Attrape |
|---|---|---|
| Structurel (phase 1) | cette référence a-t-elle été transmise ? | citation **inventée** |
| Sémantique (phase 5) | ce passage soutient-il l'affirmation ? | citation **réelle mais hors sujet** |

Une réponse affirmative que plus aucune citation ne soutient est **rétrogradée
en `insufficient_evidence`**, et toute action proposée est retirée : un ticket
ne doit jamais s'appuyer sur une réponse dont les sources ne tiennent pas.

`is_sourced` n'est vrai que si **les deux** contrôles ont été exécutés et
réussis. Un contrôle sémantique absent ou en échec ne vaut pas approbation :
le vérificateur s'abstient (`unknown`) plutôt que de supposer le soutien.

#### Deux vérificateurs sémantiques

- `LlmGroundingChecker` — lit affirmation et passage ensemble via
  `prompts/evidence_check_v1.yaml`. C'est celui qui compte.
- `LexicalGroundingChecker` — recouvrement de mots pleins. Approximation
  faible, conservée parce qu'elle fonctionne **sans aucun modèle** : elle
  détecte de façon fiable une citation sans rapport avec l'affirmation, et
  rend le pipeline testable hors ligne. Elle s'abstient dans la zone
  intermédiaire au lieu d'inventer un verdict.

#### Limite : le VLM n'a jamais été exécuté

Ollama n'est pas installé sur cette machine. Le client, le prompt, le décodage
contraint et les deux vérifications sont écrits et testés contre un modèle
factice scripté, mais **aucune génération réelle n'a eu lieu**. Le critère
d'acceptation de la phase porte sur la vérification, qui est prouvée hors
ligne ; la qualité des réponses, elle, reste à mesurer.

Pour l'activer :

```bash
ollama pull qwen2.5vl:7b
uv run python -m maintidoc.generation check
```

Le modèle `qwen3.8:27b` du prototype n'existe pas et n'est pas multimodal :
les images jointes n'étaient pas lues.

### Serveur MCP et action sensible

Trois outils, ceux du cahier des charges : `get_equipment_history`,
`get_open_tickets`, `create_ticket(..., approval_token)`.

La couche protocole est volontairement mince ([`server.py`](src/maintidoc/mcp_server/server.py)).
Tout ce qui engage la sécurité vit dans le service, donc testable sans parler
MCP.

**Les affirmations de l'appelant ne sont jamais crues.** Un appel porte un
`user_id` ; le serveur lit lui-même les groupes de cet utilisateur dans ses
propres tables. `user_id` n'est d'ailleurs **pas un argument d'outil** : un
modèle qui pourrait choisir son identité annulerait tout le contrôle.

**Un refus est indistinguable.** Équipement inexistant et équipement interdit
renvoient le même message, faute de quoi l'outil sert à énumérer le parc.

#### Le jeton d'approbation

HMAC signé côté serveur, liant **l'utilisateur**, **l'empreinte du contenu
exact** et une **clé d'idempotence**. Vérifié en temps constant.

| Tentative | Résultat |
|---|---|
| Jeton rejoué | refusé (`consumed_approvals`) |
| Jeton expiré | refusé |
| Contenu modifié après approbation | refusé — nouvelle validation exigée |
| Jeton d'un autre utilisateur | refusé |
| Signature forgée, ou autre secret | refusé |
| Même demande approuvée deux fois | **un seul** ticket, l'existant est renvoyé |

L'autorisation est vérifiée **avant** l'approbation : l'ordre inverse
permettrait à un appelant non autorisé d'apprendre qu'un jeton est valide.

Un secret vide fait échouer le démarrage plutôt que de donner l'illusion
d'une protection.

L'audit n'enregistre que des métadonnées — qui, quoi, quand, autorisé ou non.
Ni contenu documentaire, ni valeur de jeton.

> **Stockage** : SQLite au lieu de PostgreSQL, faute de serveur disponible ici.
> L'interface `MaintenanceRepository` est la couture que PostgreSQL remplace en
> phase 10 ; le SQL est volontairement portable.

### Workflow agentique

```bash
uv run python -m maintidoc.agent ask "Que signifie le voyant orange ... ?" \
    --equipment SAN-LAB-2000 --ticket --seed
```

```
check_access → retrieve → [consult_history] → compose → verify
             → answer | clarify | draft_ticket
             → interrupt(validation humaine) → create_ticket
```

**Les budgets sont des compteurs, pas des consignes.** Le cahier des charges
fixe 3 appels MCP, 6 appels modèle, 60 secondes : ils sont appliqués par le
code. Un agent à qui l'on demande poliment d'arrêter de boucler boucle parfois
quand même ; un compteur qui lève ne se laisse pas convaincre.

**Le temps d'attente humaine n'est pas facturé.** L'horloge est mise en pause
pendant la décision : un technicien qui lit attentivement un brouillon ne
consomme aucun calcul, et compter ce temps reviendrait à le sanctionner.

**L'interruption est réelle**, adossée à un checkpointer : l'exécution
s'arrête, l'état est persisté, et rien n'est créé tant qu'aucune décision
n'arrive. « Pas de création sans approbation » devient structurel plutôt que
dépendant de la bonne conduite de l'agent.

Un brouillon modifié par l'opérateur devient le contenu réellement créé — et,
puisque l'empreinte change, il est réapprouvé.

Une réponse en `insufficient_evidence` ou `needs_clarification` **ne peut pas**
donner lieu à un ticket : une affirmation que ses sources ne soutiennent pas ne
devient pas une demande d'intervention.

### Contrainte mémoire mesurée

Sur la machine de développement, **BGE-M3 et le reranker ne tiennent pas
ensemble en mémoire** : charger `bge-reranker-v2-m3` après `bge-m3` lève un
`MemoryError` (et, via le fichier de pagination Windows, une `OSError 1455`).
Les deux pèsent environ 568 millions de paramètres chacun, en float32.

C'est précisément l'hypothèse matérielle que le cahier des charges (§11)
demande de valider par mesure — avec ici un résultat négatif, à consigner tel
quel plutôt qu'à contourner en silence.

Conséquences, déjà prises en compte :

- **toutes les branches de recherche sont optionnelles.** Une configuration
  lexicale + reranker répond sans charger BGE-M3 et tient en mémoire ;
- **les modèles se chargent une fois et se partagent.** La première version des
  tests construisait un `Retriever` par test, rechargeant BGE-M3 à chaque fois ;
  le service de la phase 8 devra charger au démarrage, jamais par requête ;
- les tests modèles sont séparés par module.

Pistes pour la suite, à trancher avant la phase 5 (le VLM est plus gros
encore) : augmenter le fichier de pagination, quantifier les modèles, passer
le reranker en ONNX, ou le sortir dans un service distinct.

## Ingestion : ce qui a été corrigé

| Prototype | Phase 2 |
|---|---|
| Figure = page entre 2 et 6 | Figure = ce qu'une légende désigne, légendes repliées comprises |
| Figures vectorielles invisibles | Zone rendue quand `get_images()` ne retourne rien |
| Cellules de tableau indexées deux fois | Les blocs situés dans un tableau sont exclus du corps de texte |
| En-têtes et pieds de page indexés | Détectés dans les marges, numéro de page normalisé |
| Un document, identifiants implicites | Manifeste déclarant `document_id`, version et `acl_groups` |
| Réindexation complète à chaque lancement | Idempotence sur le SHA-256, identifiants stables |
| 1 document, 24 éléments | 41 documents, 317 éléments, 2 familles |

L'OCR (`--ocr`) exige Tesseract et `pytesseract` sur la machine. Il est
desactivé par défaut et le pipeline signale explicitement une page pauvre en
texte natif quand aucun moteur n'est disponible, plutôt que de la perdre en
silence.

## Sécurité

Les documents, les images et les réponses d'outils sont traités comme des
**données**, jamais comme des instructions. Les budgets du cahier des charges
(3 appels MCP, 6 appels modèle, 60 s) sont des paramètres de configuration,
appliqués par le code aux phases 6 et 7.
