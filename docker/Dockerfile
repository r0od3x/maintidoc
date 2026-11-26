# Image unique, deux roles (api, ui), choisis par la commande.
# Base slim : le modele VLM vit dans le conteneur Ollama, pas ici.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# libgomp : requis par torch et onnxruntime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 tesseract-ocr tesseract-ocr-fra \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Les dependances changent moins souvent que le code : couche separee.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev \
      --extra ingestion --extra retrieval --extra agent \
      --extra mcp --extra api --extra ui

COPY prompts ./prompts
COPY corpus/manifest.yaml ./corpus/manifest.yaml

# Utilisateur non privilegie : rien ici n'a besoin de root.
RUN useradd --create-home --uid 10001 maintidoc \
 && chown -R maintidoc:maintidoc /app
USER maintidoc

ENV PATH="/app/.venv/bin:$PATH"

FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "maintidoc.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ui
EXPOSE 8501
CMD ["streamlit", "run", "src/maintidoc/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
