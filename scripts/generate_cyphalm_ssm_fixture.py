#!/usr/bin/env python3
"""Generate CyphaLM SSM one-step parity fixture for native/cyphalm_ssm_parity.cpp."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_lm.temporal.cellai_ssm import CellAISSM  # noqa: E402


def main() -> None:
    cfg = dict(
        d_input=64,
        d_state=128,
        tau_fast=10.0,
        tau_slow=100.0,
        n_layers=2,
        seed=42,
        use_spectral_pde=True,
        use_multiscale=True,
        use_sparse_hebbian=True,
    )
    ssm = CellAISSM(**cfg)
    rng = np.random.default_rng(123)
    e_t = rng.standard_normal(64).astype(np.float64)
    ctx = ssm.step(e_t)

    w_fast = [np.asarray(w, dtype=np.float64).ravel().tolist() for w in ssm.W_fast]
    w_slow = [np.asarray(w, dtype=np.float64).ravel().tolist() for w in ssm.W_slow]

    out_dir = _REPO / "parity_fixtures" / "cyphalm_ssm"
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "config": cfg,
        "e_t": e_t.tolist(),
        "context": np.asarray(ctx, dtype=np.float64).tolist(),
        "context_dim": int(ssm.context_dim),
        "W_fast": w_fast,
        "W_slow": w_slow,
    }
    path = out_dir / "sidecar.json"
    path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} (context_dim={sidecar['context_dim']})")


if __name__ == "__main__":
    main()
