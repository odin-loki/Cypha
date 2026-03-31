"""
Wrap ``cypha_studio/test_cypha_studio.py`` so CI runs it via pytest.

The script exits 1 on any failure and prints a summary.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "cypha_studio" / "test_cypha_studio.py"


@pytest.mark.slow
def test_cypha_studio_script_all_pass():
    assert _SCRIPT.is_file(), _SCRIPT
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        env=env,
    )
    assert r.returncode == 0, (
        f"stdout:\n{r.stdout}\n\nstderr:\n{r.stderr}"
    )
