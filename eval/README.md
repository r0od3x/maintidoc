# Évaluation (phase 9)

```
eval/
  datasets/     scénarios annotés, versionnés avec le corpus
  runners/      exécution hors interface, sortie comparable
  reports/      résultats par version de modèle / prompt / index
```

Répartition visée (cahier des charges §10), environ 150 scénarios :

| Catégorie | Nombre |
|---|---|
| Questions documentaires textuelles | 40 |
| Questions nécessitant une image ou un schéma | 30 |
| Consultation MCP | 20 |
| Sources absentes, insuffisantes ou contradictoires | 20 |
| Contrôle d'accès et validation humaine | 20 |
| Injections de prompt | 20 |

Les injections doivent emprunter **trois canaux distincts** : le texte des
documents, le texte inscrit dans les images (chemin OCR) et les réponses du
serveur MCP. Chaque scénario est annoté avec les éléments attendus
(`document_id:pN:element_id`), ce qui rend Recall@5, nDCG@5 et MRR@10
calculables automatiquement.

Chaque exécution est associée à un `RunVersions` (`maintidoc.versions`), sans
quoi deux rapports ne sont pas comparables.
