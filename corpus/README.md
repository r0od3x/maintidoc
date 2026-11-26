# Corpus

```
corpus/
  raw/<equipement>/    documents sources, versionnés
  derived/             sorties d'ingestion (ignoré par Git, reconstructible)
```

`raw/san_lab_2000/` contient le manuel pédagogique **fictif** du prototype
(9 pages, FIG-01 à FIG-05, SAN-MAN-001 révision 1.1).

Phase 2 porte le corpus à 30-50 documents sur **deux familles d'équipements**.
Chaque document reçoit un `document_id` stable, une version, un SHA-256 et des
`acl_groups` : ces champs sont la base des citations vérifiables et du contrôle
d'accès, et ne peuvent pas être ajoutés après coup.

Aucune donnée personnelle, aucun document sous licence restrictive.
