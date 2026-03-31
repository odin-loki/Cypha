"""cypha_studio.env_config — registry, API defaults, CORS."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def clean_env(monkeypatch):
    keys = (
        "CYPHA_REGISTRY_ROOT",
        "CYPHA_API_HOST",
        "CYPHA_API_PORT",
        "CYPHA_CORS_ORIGINS",
        "CYPHA_CSV_CHUNK_ROWS",
    )
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    yield
    for k, v in saved.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


def test_registry_root_default(clean_env):
    from cypha_studio import env_config

    assert env_config.registry_root() == "~/.cypha/models"
    assert "cypha" in str(env_config.registry_root_expanded()).lower()


def test_registry_root_override(clean_env, monkeypatch):
    monkeypatch.setenv("CYPHA_REGISTRY_ROOT", "/tmp/cypha_models")
    from cypha_studio import env_config

    assert env_config.registry_root() == "/tmp/cypha_models"


def test_api_defaults(clean_env, monkeypatch):
    from cypha_studio.env_config import api_default_host, api_default_port

    assert api_default_host() == "127.0.0.1"
    assert api_default_port() == 7749
    monkeypatch.setenv("CYPHA_API_HOST", "0.0.0.0")
    monkeypatch.setenv("CYPHA_API_PORT", "9000")
    assert api_default_host() == "0.0.0.0"
    assert api_default_port() == 9000


def test_cors_allow_origins(clean_env, monkeypatch):
    from cypha_studio.env_config import cors_allow_origins

    assert cors_allow_origins() == ["*"]
    monkeypatch.setenv("CYPHA_CORS_ORIGINS", "http://a.local,https://b.app")
    assert cors_allow_origins() == ["http://a.local", "https://b.app"]


def test_csv_read_chunk_rows(clean_env, monkeypatch):
    from cypha_studio.env_config import csv_read_chunk_rows

    assert csv_read_chunk_rows() == 0
    monkeypatch.setenv("CYPHA_CSV_CHUNK_ROWS", "50000")
    assert csv_read_chunk_rows() == 50000
    monkeypatch.setenv("CYPHA_CSV_CHUNK_ROWS", "0")
    assert csv_read_chunk_rows() == 0
    monkeypatch.setenv("CYPHA_CSV_CHUNK_ROWS", "nope")
    assert csv_read_chunk_rows() == 0
