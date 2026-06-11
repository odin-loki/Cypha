#!/usr/bin/env python3
"""
Emit parity fixtures for native ``PreprocessorState::fit_from_design_matrix``:

- ``parity_fixtures/preprocessor_fit/`` — ``scale=True``, PCA
- ``parity_fixtures/preprocessor_fit_no_scale/`` — ``scale=False``, PCA
- ``parity_fixtures/preprocessor_fit_rff/`` — ``scale=True``, PCA + RFF

Native Jacobi PCA vs NumPy SVD: sign-aligned in ``preprocessor_fit_parity``.
RFF weights use NumPy ``default_rng(seed)`` — reproduced natively via ``NumpyDefaultRng``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cypha_studio.core.dataset import Preprocessor


def _write_case(
    out_dir: Path,
    *,
    scale: bool,
    pca_dim: int,
    rff_dim: int | None,
    rff_gamma: float,
    seed_fit: int,
    n_rows: int,
    n_cols: int,
    rng: np.random.Generator,
    n_probes: int,
    probe_scale: float,
) -> None:
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float64)
    pre = Preprocessor(scale=scale, pca_dim=pca_dim, rff_dim=rff_dim, rff_gamma=rff_gamma, seed=seed_fit)
    pre.fit(X)
    state = pre.save_state()

    probes = []
    for _ in range(n_probes):
        x = rng.standard_normal(n_cols).astype(np.float64) * probe_scale
        probes.append({"x": x.tolist(), "expected": pre.transform_one(x).tolist()})

    design = {
        "fixture_schema": 1,
        "scale": scale,
        "pca_dim": pca_dim,
        "rff_dim": rff_dim,
        "seed": seed_fit,
        "rff_gamma": rff_gamma,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rowmajor": X.ravel(order="C").tolist(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    (out_dir / "expected_preprocessor.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (out_dir / "probe.json").write_text(json.dumps({"probes": probes}, indent=2), encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(2026)
    _write_case(
        _ROOT / "parity_fixtures" / "preprocessor_fit",
        scale=True,
        pca_dim=3,
        rff_dim=None,
        rff_gamma=1.0,
        seed_fit=99,
        n_rows=75,
        n_cols=6,
        rng=rng,
        n_probes=6,
        probe_scale=0.7,
    )
    print(f"Wrote {_ROOT / 'parity_fixtures' / 'preprocessor_fit'}/")

    rng2 = np.random.default_rng(8844)
    _write_case(
        _ROOT / "parity_fixtures" / "preprocessor_fit_no_scale",
        scale=False,
        pca_dim=2,
        rff_dim=None,
        rff_gamma=1.0,
        seed_fit=101,
        n_rows=55,
        n_cols=7,
        rng=rng2,
        n_probes=5,
        probe_scale=0.65,
    )
    print(f"Wrote {_ROOT / 'parity_fixtures' / 'preprocessor_fit_no_scale'}/")

    rng3 = np.random.default_rng(5511)
    _write_case(
        _ROOT / "parity_fixtures" / "preprocessor_fit_rff",
        scale=True,
        pca_dim=3,
        rff_dim=8,
        rff_gamma=0.5,
        seed_fit=99,
        n_rows=75,
        n_cols=6,
        rng=rng3,
        n_probes=6,
        probe_scale=0.7,
    )
    print(f"Wrote {_ROOT / 'parity_fixtures' / 'preprocessor_fit_rff'}/")


if __name__ == "__main__":
    main()
