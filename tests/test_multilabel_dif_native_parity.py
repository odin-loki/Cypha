"""``multilabel_dif_parity`` vs ``parity_fixtures/multilabel_dif/``. CTest: ``native_multilabel_dif``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable

_FIX = _ROOT / "parity_fixtures" / "multilabel_dif"


def test_multilabel_dif_parity_subprocess():
    if not (_FIX / "sidecar.json").is_file():
        pytest.skip("run scripts/generate_multilabel_dif_fixture.py")
    r = run_native_executable(
        "multilabel_dif_parity",
        [_FIX],
        timeout=120,
        env_override="CYPHA_MULTILABEL_DIF_PARITY_BIN",
    )
    if r is None:
        pytest.skip("multilabel_dif_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
