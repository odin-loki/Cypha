#!/usr/bin/env python3
"""
Emit SQLite DDL for ExperimentDB (``cypha_studio.core.experiment._SCHEMA``).

Use for native bootstrapping, ``sqlite3 my.db < experiment.sql``, or diffing against docs.

  python scripts/export_experiment_schema_sql.py
  python scripts/export_experiment_schema_sql.py -o artifacts/experiment_schema.sql
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Write DDL to this file (UTF-8); default: stdout",
    )
    args = p.parse_args()

    from cypha_studio.core.experiment import _SCHEMA

    text = _SCHEMA.strip() + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
