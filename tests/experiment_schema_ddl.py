"""Load canonical ExperimentDB DDL from ``cypha_studio/core/experiment.py`` without importing ``cypha_studio``."""
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT_PY = _REPO_ROOT / "cypha_studio" / "core" / "experiment.py"


def experiment_schema_sql() -> str:
    """Return ``_SCHEMA`` text (strip + trailing newline) from ``experiment.py``."""
    if not _EXPERIMENT_PY.is_file():
        raise FileNotFoundError(_EXPERIMENT_PY)
    tree = ast.parse(_EXPERIMENT_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or t.id != "_SCHEMA":
            continue
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return v.value.strip() + "\n"
    raise RuntimeError("_SCHEMA not found in experiment.py")
