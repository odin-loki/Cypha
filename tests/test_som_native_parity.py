"""``som_parity`` smoke. CTest: ``native_som_parity``. Override: ``CYPHA_SOM_PARITY_BIN``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.native_subprocess import run_native_executable


def test_som_parity_subprocess():
    r = run_native_executable(
        "som_parity",
        [],
        timeout=30,
        env_override="CYPHA_SOM_PARITY_BIN",
    )
    if r is None:
        pytest.skip("som_parity not built")
    assert r.returncode == 0, (r.stdout, r.stderr)
