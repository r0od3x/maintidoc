# Feuille de route — 10 phases

Chaque phase a un critère d'acceptation vérifiable. Les phases 2 → 5 forment
une chaîne stricte. La phase 6 (serveur MCP) est un service indépendant et peut
être menée en parallèle dès le début. Le jeu de test de la phase 9 s'écrit
pendant les phases 2 à 5, pas après.

## 1 — Fondations et contrats ✅

Monorepo, dépendances figées (`uv.lock`), secrets dans l'environnement
uniquement. Contrats Pydantic écrits **avant** le code qui les consomme :
`Element`, `Evidence`, `EvidenceBundle`, `Citation`, `AnswerV1`, `TicketDraft`,
`ApprovalToken`, `RunVersions`. Prompts versionnés dans `prompts/`, chargés avec
leur empreinte SHA-256.

*Acceptation* : le schéma JSON écrit à la main dans le notebook est remplacé par
`AnswerV1`, importable par le notebook comme par l'API. ✅ 34 tests.

## 2 — Corpus et ingestion ✅

Passer de 1 document à **30-50 documents sur deux familles d'équipements** (la
seconde famille peut être synthétique, produite comme l'a été SAN-LAB 2000).
PyMuPDF plus OCR local pour les pages scannées et pour le texte **contenu dans
les figures**. Fichiers dans MinIO, métadonnées dans PostgreSQL.

*Acceptation* : ingestion idempotente sur le SHA-256 ✅ ; l'heuristique
positionnelle du prototype (`kind = "schema" if 2 <= page <= 6`) est remplacée
par une détection de figures et de légendes ✅ (FIG-01 à FIG-05 retrouvées par
leur légende, photo de couverture classée comme illustration).

Corpus porté à **41 documents sur deux familles** (SAN et onduleurs) par un
générateur déterministe, `maintidoc.synth`. Les significations d'indicateurs
divergent d'un modèle à l'autre, ce qui force la recherche à discriminer. Le
générateur émet aussi la vérité de terrain consommée par la phase 9.

**Limite connue** : l'OCR est implémenté et testé avec un moteur factice, mais
n'a jamais tourné contre un vrai Tesseract — à vérifier avant de s'appuyer
dessus pour des documents scannés.

## 3 — Indexation et stockage persistant ✅

Qdrant, deux collections : texte (BGE-M3) et visuel (encodeur multilingue).
Version d'index inscrite dans les métadonnées ; la réindexation est un travail
tracé.

*Acceptation* : une question en français récupère la bonne figure sans
traduction manuelle en anglais ✅ — par la légende, dans l'index texte BGE-M3.

**Limite mesurée** : l'index visuel seul discrimine mal le français
(deux requêtes françaises sans rapport à 0,977 de cosinus, contre 0,787 pour la
paire anglaise). L'alignement entre langues est correct ; la séparation interne
au français ne l'est pas. Voir le README pour les options.

## 4 — Récupération ✅

Dans l'ordre imposé par le cahier des charges : identité et permissions →
filtrage du corpus **avant** récupération (filtre Qdrant sur `acl_groups`) →
recherches texte et visuelle → fusion RRF → reranking (`bge-reranker-v2-m3`).
Fusion au niveau **élément**, la page servant de signal de regroupement : la
fusion par page du prototype perd en précision dès que le corpus grandit.

*Acceptation* : un utilisateur du groupe B ne peut pas récupérer un document du
groupe A, et le test le démontre ✅ — filtrage avant scoring dans les trois
branches, refus explicite pour un principal sans groupe.

Une branche lexicale BM25 a été ajoutée aux deux branches demandées : le corpus
se joue sur des identifiants exacts que les embeddings denses distinguent mal.

**Contrainte mesurée** : BGE-M3 et le reranker ne tiennent pas ensemble en
mémoire sur la machine de développement (`MemoryError`). Les branches sont donc
toutes optionnelles, et les modèles doivent être chargés une fois puis
partagés. À arbitrer avant la phase 5, dont le VLM est plus gros encore.

## 5 — Génération sourcée et vérification ✅ (vérification) · VLM non exécuté

Appel du VLM avec les preuves présentées comme données non fiables, sortie
contrainte par `AnswerV1`. Double contrôle : vérification structurelle
(`verify_structural`, phase 1) **et** vérification sémantique du soutien de
chaque affirmation (`evidence_check_v1`). Une affirmation non soutenue fait
basculer la réponse en `insufficient_evidence`.

*Acceptation* : une citation inventée **et** une citation réelle mais non
soutenante sont toutes deux détectées ✅ — la première par le contrôle
structurel, la seconde par le contrôle sémantique, avec rétrogradation
automatique en `insufficient_evidence`.

**Limite** : Ollama n'étant pas installé, aucune génération réelle n'a eu lieu.
Le pipeline est testé contre un modèle factice scripté. La vérification est
prouvée ; la qualité des réponses reste à mesurer (phase 9).

## 6 — Serveur MCP et actions sensibles ✅

`get_equipment_history`, `get_open_tickets`, `create_ticket(..., approval_token)`
sur une base de tickets fictive. Autorisation revérifiée **côté serveur à chaque
appel** : l'agent n'est jamais de confiance. Le jeton d'approbation lie
l'utilisateur et l'empreinte du contenu exact ; toute modification l'invalide ;
une clé d'idempotence évite les doublons.

*Acceptation* : un jeton rejoué, expiré, ou dont le contenu a changé est refusé ✅
— plus le jeton d'un autre utilisateur, la signature forgée et le secret
étranger. PostgreSQL est remplacé par SQLite (aucun serveur disponible) ;
`MaintenanceRepository` est la couture de remplacement.

## 7 — Orchestration LangGraph ✅

Nœuds : `check_access → analyze_request → retrieve → maybe_mcp_history →
compose → verify → (answer | clarify | draft_ticket) → interrupt(approbation) →
create_ticket`. Budgets appliqués par le code : **≤ 3 appels MCP, ≤ 6 appels
modèle, ≤ 60 s**. Checkpointer pour reprendre après l'interruption humaine.

*Acceptation* : une boucle de récupération provoquée s'arrête sur budget ✅ —
compteurs appliqués par le code, horloge mise en pause pendant l'attente
humaine, interruption réelle adossée à un checkpointer.

## 8 — API, interface, authentification ✅

FastAPI avec OIDC/JWT et RBAC devant l'ensemble. Streamlit (évolution de
`legacy/app_streamlit_prototype.py`) : conversation, photo, panneau des sources,
et écran d'approbation affichant le contenu **exact** du ticket à créer.
Discord/Telegram restent facultatifs.

*Acceptation* : le scénario de démonstration complet passe par l'API, non par
des cellules de notebook ✅ — question, réponse sourcée, historique MCP, ticket
préparé, approuvé, créé une seule fois, le tout en HTTP.

## 9 — Évaluation et tests d'injection ✅ (obéissance non mesurée)

Environ 150 scénarios selon le tableau du cahier des charges : 40 documentaires,
30 visuels, 20 MCP, 20 sources absentes ou contradictoires, 20 contrôle d'accès,
20 injections. Les injections passent par **trois canaux** : le texte des
documents, le texte **inscrit dans les images** (chemin OCR) et les réponses du
serveur MCP. Mesures : Recall@5, nDCG@5, MRR@10, taux de citations soutenues,
abstentions justifiées, latences p50 et p95.

*Acceptation* : la suite tourne sans interface en CI et produit un rapport
comparable d'une exécution à l'autre ✅ — 150 scénarios, rapport JSON portant
les versions qui l'ont produit.

**Limite** : sans générateur, l'obéissance aux injections, la fidélité des
réponses et le taux de citations soutenues sont rapportés `non vérifiable`,
jamais « réussi ». Le confinement des injections, lui, est mesuré : 20/20.

## 10 — Fine-tuning du reranker, MLOps, livraison ✅ (entraînement non lancé)

800 à 1 500 triplets (question, passage pertinent, négatif difficile) — les
négatifs difficiles doivent provenir de la **même famille d'équipements**, sinon
l'expérience ne mesure rien. Séparation par document ou par famille, jamais
aléatoire. Comparaison en trois temps : sans reranker, reranker d'origine,
reranker adapté, journalisée dans MLflow. Puis Docker Compose (API, Streamlit,
Qdrant, PostgreSQL, MinIO, MLflow, MCP, Ollama), CI, procédure de retour
arrière, fiche de limites, scénario de démonstration.

*Acceptation* : partiellement atteinte, et l'écart est signalé plutôt que
masqué.

- 910 triplets, 75 % de négatifs difficiles, séparation par modèle sans fuite ✅
- harnais de comparaison à trois bras, delta négatif rapporté tel quel ✅
- suivi MLflow, dégradation explicite si absent ✅
- Docker Compose, Dockerfile, CI, fiche de limites, scénario de démonstration,
  procédure de retour arrière ✅ écrits
- **entraînement jamais lancé** : le reranker ne tient pas en mémoire ici,
  aucun gain n'est revendiqué ❌
- **`docker compose up` jamais exécuté** : aucun moteur Docker disponible ❌
