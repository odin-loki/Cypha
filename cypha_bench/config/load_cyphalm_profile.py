"""Load domain-specific CyphaLM profiles for cypha_bench."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = BENCH_ROOT / "config"

PROFILE_ALIASES: dict[str, str] = {
    "llm": "profiles/cyphalm_llm.json",
    "default": "profiles/cyphalm_llm.json",
    "d04": "profiles/cyphalm_d04_gutenberg.json",
    "gutenberg": "profiles/cyphalm_d04_gutenberg.json",
    "d17": "profiles/cyphalm_d17_wikitext.json",
    "wikitext": "profiles/cyphalm_d17_wikitext.json",
}

LEGACY_PROFILE = CONFIG_DIR / "cyphalm_profile.json"


def resolve_cyphalm_profile_path(name: str | None = None) -> Path:
    """Resolve profile by alias, env CYPHALM_PROFILE, or legacy cyphalm_profile.json."""
    key = name or os.environ.get("CYPHALM_PROFILE") or os.environ.get("CYPHA_LM_PROFILE")
    if key:
        rel = PROFILE_ALIASES.get(str(key).lower().strip())
        if rel:
            return CONFIG_DIR / rel
        p = Path(key)
        if p.exists():
            return p
        candidate = CONFIG_DIR / key
        if candidate.exists():
            return candidate
        candidate = CONFIG_DIR / "profiles" / key
        if candidate.exists():
            return candidate
    if LEGACY_PROFILE.exists():
        return LEGACY_PROFILE
    return CONFIG_DIR / PROFILE_ALIASES["llm"]


def load_cyphalm_profile_file(path: Path | None = None) -> dict[str, Any]:
    p = path or resolve_cyphalm_profile_path()
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if "_meta" in data:
        data = {k: v for k, v in data.items() if not k.startswith("_")}
    return data


def write_cyphalm_profile(
    params: dict[str, Any],
    *,
    profile: str,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write tuned params to profiles/<name>.json and sync legacy cyphalm_profile.json if llm."""
    rel = PROFILE_ALIASES.get(profile, f"profiles/cyphalm_{profile}.json")
    path = CONFIG_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(params)
    if meta:
        payload["_meta"] = meta
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if profile in ("llm", "default"):
        LEGACY_PROFILE.write_text(json.dumps({k: v for k, v in payload.items() if k != "_meta"}, indent=2) + "\n", encoding="utf-8")
    return path
