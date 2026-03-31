"""Smoke-test ``scripts/export_regression_head.py`` (MoE sidecar JSON)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = _ROOT / "docs" / "port" / "schemas" / "regression_head.schema.json"


def test_regression_head_json_schema_file_is_valid_json() -> None:
    import json

    assert _SCHEMA.is_file(), _SCHEMA
    data = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    assert data.get("title")
    assert "experts" in (data.get("properties") or {})


def test_export_regression_head_cli_writes_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "rh.json"
    r = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "export_regression_head.py"), "-o", str(out), "--steps", "25", "--seed", "1"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data.get("schema") == 1
    ex = data.get("experts")
    assert isinstance(ex, dict) and len(ex) >= 1
    for _lbl, row in ex.items():
        assert isinstance(row, dict)
        assert "mu" in row and "var_ema" in row
        assert isinstance(row["mu"], (int, float))
        assert isinstance(row["var_ema"], (int, float))
