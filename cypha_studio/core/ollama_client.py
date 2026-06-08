"""Minimal Ollama HTTP client for Branch A fallback generation."""

from __future__ import annotations

import os
from typing import Any

import httpx


def ollama_base_url() -> str:
    return os.environ.get("CYPHA_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def ollama_model() -> str:
    return os.environ.get("CYPHA_OLLAMA_MODEL", "mistral").strip() or "mistral"


def ollama_timeout_s() -> float:
    raw = os.environ.get("CYPHA_OLLAMA_TIMEOUT_S", "120").strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 120.0


def ollama_available(*, base_url: str | None = None, timeout_s: float = 3.0) -> bool:
    """Return True if Ollama responds on ``/api/tags``."""
    url = (base_url or ollama_base_url()).rstrip("/")
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(f"{url}/api/tags")
            return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def ollama_generate(
    prompt: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    system: str | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """
    Call Ollama ``POST /api/generate``.

    Returns ``{"text", "model", "provider", "latency_ms", "done"}``.
    Raises ``RuntimeError`` on HTTP or connection failure.
    """
    import time

    url = (base_url or ollama_base_url()).rstrip("/")
    model_name = model or ollama_model()
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": bool(stream),
    }
    if system:
        payload["system"] = system

    t0 = time.perf_counter()
    with httpx.Client(timeout=ollama_timeout_s()) as client:
        if stream:
            parts: list[str] = []
            with client.stream("POST", f"{url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json

                    row = json.loads(line)
                    chunk = row.get("response", "")
                    if chunk:
                        parts.append(str(chunk))
                    if row.get("done"):
                        break
            text = "".join(parts)
        else:
            r = client.post(f"{url}/api/generate", json=payload)
            r.raise_for_status()
            body = r.json()
            text = str(body.get("response", ""))

    return {
        "provider": "ollama",
        "model": model_name,
        "text": text,
        "latency_ms": (time.perf_counter() - t0) * 1000.0,
        "done": True,
    }
