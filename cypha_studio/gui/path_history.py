"""Recent dataset paths + last browse directory (QSettings)."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings

KEY_LAST_DIR = "paths/last_dataset_dir"
KEY_RECENT = "paths/recent_datasets_json"


def studio_settings() -> QSettings:
    return QSettings("Cypha", "CyphaStudio")


def dataset_dialog_start_dir() -> str:
    s = studio_settings()
    v = s.value(KEY_LAST_DIR, "")
    return v if isinstance(v, str) else str(v or "")


def dataset_dialog_start_dir_preferred(prefs=None) -> str:
    """Prefer ``StudioPreferences.dataset_dialog_start_dir`` when it exists on disk."""
    if prefs is not None:
        d = (getattr(prefs, "dataset_dialog_start_dir", "") or "").strip()
        if d and Path(d).is_dir():
            return d
    return dataset_dialog_start_dir()


def _load_recent(s: QSettings) -> list[str]:
    raw = s.value(KEY_RECENT, "[]")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        return list(json.loads(str(raw)))
    except (json.JSONDecodeError, TypeError):
        return []


def recent_dataset_paths(max_n: int = 12) -> list[str]:
    return _load_recent(studio_settings())[:max_n]


def record_dataset_opened(path: str, max_recent: int = 12) -> None:
    p = Path(path)
    s = studio_settings()
    s.setValue(KEY_LAST_DIR, str(p.parent.resolve()))
    cur = _load_recent(s)
    path_str = str(p.resolve()) if p.exists() else str(path)
    nxt = [path_str] + [x for x in cur if x != path_str]
    s.setValue(KEY_RECENT, json.dumps(nxt[:max_recent]))
