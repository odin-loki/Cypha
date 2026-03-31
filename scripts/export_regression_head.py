#!/usr/bin/env python3
"""
Write ``regression_head.json`` (scalar MoE sidecar) from a short ``DIFRegressor`` demo train.

Same schema as native ``cypha_rest --regression-json`` and FastAPI ``CYPHA_REGRESSION_HEAD``.
Expert keys are routing labels (``_e0``, …) — for production, train a regressor aligned with
your classifier's class names or map keys before deploy.

Usage::

    python scripts/export_regression_head.py -o artifacts/regression_head.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Cypha import DIFRegressor, VectorEncoder  # noqa: E402


def experts_from_difreg(reg: DIFRegressor) -> dict:
    out: dict = {}
    for k, mu_arr in reg._expert_mu.items():
        mu = np.asarray(mu_arr, dtype=np.float64).ravel()
        mu0 = float(mu[0]) if mu.size > 0 else 0.0
        out[str(k)] = {"mu": mu0, "var_ema": float(reg._expert_var.get(k, 0.0))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export regression_head.json from DIFRegressor demo train")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_ROOT / "artifacts" / "regression_head_demo.json",
        help="Output JSON path (default: artifacts/regression_head_demo.json)",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for encoder + stream")
    ap.add_argument("--steps", type=int, default=80, help="Online train steps")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    reg = DIFRegressor(VectorEncoder(6), field_dim=40, n_experts=6, rng=rng)
    for i in range(max(1, args.steps)):
        reg.train_step(rng.standard_normal(6), float(rng.normal(0, 2.0)))

    payload = {"schema": 1, "experts": experts_from_difreg(reg)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(payload['experts'])} experts)")


if __name__ == "__main__":
    main()
