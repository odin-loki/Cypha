"""
Environment-driven defaults for CyphaStudio (API listen, registry root, CORS).

Variables (all optional):

- **CYPHA_REGISTRY_ROOT** — model registry directory (default ``~/.cypha/models``). The default
  FastAPI ASGI app (``uvicorn cypha_studio.server.api:app``) uses **``ModelRegistry(registry_root())``**
  so ``/models``, ``/load``, and ``/register`` hit this tree.
- **CYPHA_API_HOST** — REST bind address (default ``127.0.0.1``).
- **CYPHA_API_PORT** — REST port (default ``7749``).
- **CYPHA_CORS_ORIGINS** — comma-separated origins, or ``*`` for allow-all (default ``*``).
- **CYPHA_CSV_CHUNK_ROWS** — if set to a positive integer, ``CSVDataset.from_file`` streams
  the CSV in chunks of that size (lower peak memory for very large files). Unset = load whole
  file as before.
- **CYPHA_REGRESSION_HEAD** — optional path to ``regression_head.json`` for FastAPI ``/predict``
  (scalar mixture regression overlay; mirrors native ``cypha_rest``). See ``docs/studio/CYPHA_ENV.md``.
- **CYPHA_BRANCH_A_EPISTEMIC_THRESHOLD** — epistemic variance gate for ``POST /route/*`` (default ``0.5``).
- **CYPHA_BRANCH_A_N_TRAIN** — 20 Newsgroups samples for lazy Branch A router training (default ``1200``).
- **CYPHA_BRANCH_A_EMBED_BACKEND** — ``auto`` | ``sentence_transformers`` | ``hashing`` for text embedder.
- **CYPHA_BRANCH_A_CHECKPOINT** — checkpoint base path (``.json`` + ``.npz`` pair) to load/skip retrain.
- **CYPHA_BRANCH_A_AUTO_SAVE** — when ``1``, save checkpoint after training (default off).
- **CYPHA_OLLAMA_URL** — Ollama base URL for ``fallback_llm`` generation (default ``http://127.0.0.1:11434``).
- **CYPHA_OLLAMA_MODEL** — Ollama model tag (default ``mistral``).

CLI flags in ``cypha_studio/main.py`` override these when explicitly parsed after ``os.environ``
is applied to defaults at parse time.

**Human-readable reference:** ``docs/studio/CYPHA_ENV.md`` (REST routes, production notes).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List


def registry_root() -> str:
    return os.environ.get("CYPHA_REGISTRY_ROOT", "~/.cypha/models")


def registry_root_expanded() -> Path:
    return Path(os.path.expanduser(registry_root()))


def api_default_host() -> str:
    return os.environ.get("CYPHA_API_HOST", "127.0.0.1")


def api_default_port() -> int:
    return int(os.environ.get("CYPHA_API_PORT", "7749"))


def cors_allow_origins() -> List[str]:
    raw = os.environ.get("CYPHA_CORS_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parts if parts else ["*"]


def csv_read_chunk_rows() -> int:
    """0 = default (buffer full CSV); else stream in chunks of this many rows."""
    raw = os.environ.get("CYPHA_CSV_CHUNK_ROWS", "").strip()
    if not raw:
        return 0
    try:
        n = int(raw)
    except ValueError:
        return 0
    if n <= 0:
        return 0
    return n


def branch_a_checkpoint_base() -> str:
    """Base path for Branch A router checkpoint (``.json`` + ``.npz`` siblings)."""
    raw = os.environ.get("CYPHA_BRANCH_A_CHECKPOINT", "").strip()
    if raw:
        return os.path.expanduser(raw)
    return os.path.expanduser("~/.cypha/branch_a_router")


def branch_a_auto_save() -> bool:
    return os.environ.get("CYPHA_BRANCH_A_AUTO_SAVE", "").strip().lower() in ("1", "true", "yes")
