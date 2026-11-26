# Prototype gelé

`app_streamlit_prototype.py` est l'application Streamlit d'origine, conservée
telle quelle comme référence et comme point de comparaison. Elle **n'est pas
maintenue** et contient les limites corrigées à partir de la phase 2 :

- index vectoriel en mémoire, reconstruit à chaque démarrage ;
- aucun contrôle d'accès, aucune authentification ;
- requête visuelle à saisir en anglais (CLIP monolingue) ;
- modèle `qwen3.8:27b`, qui n'est pas un tag Ollama réel ni un modèle
  multimodal : les images transmises n'étaient pas lues ;
- schéma JSON écrit à la main, remplacé par `maintidoc.contracts.AnswerV1`.

L'interface de production est reconstruite en phase 8.
