"""
Persistent Studio preferences (QSettings) and effective paths/chunk sizes.

Environment variables still win when a GUI override is empty / zero — see
effective_* helpers and in-dialog notes for CYPHA_* names.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from ..env_config import csv_read_chunk_rows, registry_root


@dataclass
class StudioPreferences:
    """User-editable defaults for CyphaStudio (saved under QSettings)."""

    inference_use_gh: bool = True
    inference_ood_threshold: float = 3.0
    inference_chi: float = 1.0
    inference_psi: float = 1.0

    branch_a_routing_enabled: bool = False
    branch_a_epistemic_threshold: float = 0.5

    registry_root_override: str = ""
    csv_chunk_rows_override: int = 0
    dataset_dialog_start_dir: str = ""

    ui_font_pt: int = 0

    def effective_registry_root(self) -> str:
        r = (self.registry_root_override or "").strip()
        if r:
            return r
        return registry_root()

    def effective_csv_chunk_rows(self) -> int:
        if self.csv_chunk_rows_override > 0:
            return self.csv_chunk_rows_override
        return csv_read_chunk_rows()

    def to_settings_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_settings_dict(cls, d: Dict[str, Any]) -> StudioPreferences:
        b = StudioPreferences()
        return cls(**{fn.name: d.get(fn.name, getattr(b, fn.name)) for fn in fields(cls)})


def _qsettings():
    from PySide6.QtCore import QSettings

    return QSettings("Cypha", "CyphaStudio")


def load_studio_preferences() -> StudioPreferences:
    s = _qsettings()
    d: Dict[str, Any] = {}
    for f in fields(StudioPreferences):
        key = f"studio/{f.name}"
        default = getattr(StudioPreferences(), f.name)
        val = s.value(key, default)
        if f.type is bool and not isinstance(val, bool):
            val = str(val).lower() in ("1", "true", "yes")
        elif f.type is int and not isinstance(val, int):
            try:
                val = int(val)
            except (TypeError, ValueError):
                val = default
        elif f.type is float and not isinstance(val, float):
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = default
        elif f.type is str and val is not None and not isinstance(val, str):
            val = str(val)
        d[f.name] = val
    return StudioPreferences.from_settings_dict(d)


def save_studio_preferences(prefs: StudioPreferences) -> None:
    s = _qsettings()
    for f in fields(StudioPreferences):
        s.setValue(f"studio/{f.name}", getattr(prefs, f.name))


def apply_preferences_to_inference_state(state: Any) -> None:
    """Update live engine/session from ``state.preferences`` (if loaded)."""
    p = getattr(state, "preferences", None)
    if p is None:
        return
    eng = getattr(state, "engine", None)
    if eng is not None and hasattr(eng, "set_ood_threshold"):
        eng.set_ood_threshold(float(p.inference_ood_threshold))
    sess = getattr(state, "session", None)
    if sess is not None and hasattr(sess, "set_gh_params"):
        sess.set_gh_params(float(p.inference_chi), float(p.inference_psi))
