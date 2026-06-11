#!/usr/bin/env python3
"""
Emit SQLite DDL for ExperimentDB (``cypha_studio.core.experiment._SCHEMA``).

Use for native bootstrapping, ``sqlite3 my.db < experiment.sql``, or diffing against docs.

  python scripts/export_experiment_schema_sql.py
  python scripts/export_experiment_schema_sql.py -o native/sql/experiment_schema.sql
  python scripts/export_experiment_schema_sql.py -o artifacts/experiment_schema.sql
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT_PY = _ROOT / "cypha_studio" / "core" / "experiment.py"


def _schema_from_experiment_py() -> str:
    """Read ``_SCHEMA`` from ``experiment.py`` without importing ``cypha_studio`` (no numpy)."""
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Write DDL to this file (UTF-8); default: stdout",
    )
    args = p.parse_args()

    text = _schema_from_experiment_py()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
