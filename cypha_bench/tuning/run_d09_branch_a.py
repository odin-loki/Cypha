#!/usr/bin/env python3
"""Run D09 Branch A (frozen semantic embeddings + Gutenberg OOD epistemic)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "cypha_bench"))

from cypha_bench.adapters.branch_a_documents import run_branch_a_documents
from cypha_bench.common.metrics import save_table
from cypha_bench.common.paths import scale

_OUT = _REPO / "cypha_bench" / "config" / "d09_branch_a_summary.json"


def main() -> int:
    backend = os.environ.get("CYPHA_BRANCH_A_BACKEND", "auto")
    n_samples = int(os.environ.get("CYPHA_BRANCH_A_N_SAMPLES", str(scale(2000, 800))))
    freeze = os.environ.get("CYPHA_BRANCH_A_FREEZE_ENC", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

    print(
        f"[D09 Branch A] n_samples={n_samples} backend={backend} freeze_enc={freeze}",
        flush=True,
    )
    branch_a = run_branch_a_documents(
        n_samples=n_samples,
        backend=backend,
        freeze_encoder_proj=freeze,
    )
    print(
        f"  acc={branch_a['cypha_accuracy']:.4f} "
        f"ood_epi={branch_a['gutenberg_ood']['mean_epistemic_ood']:.6f} "
        f"in_epi={branch_a['gutenberg_ood']['mean_epistemic_in']:.6f}",
        flush=True,
    )

    prior_path = _REPO / "cypha_bench" / "report" / "tables" / "d09.json"
    payload: dict = {}
    if prior_path.exists():
        payload = json.loads(prior_path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", {})
    if isinstance(experiments, list):
        experiments = {}
    summary = experiments.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    summary["branch_a_frozen_embed"] = branch_a
    experiments["summary"] = summary
    payload["experiments"] = experiments
    save_table("d09_documents", {"domain": "d09_documents", **summary, "branch_a_frozen_embed": branch_a})

    _OUT.write_text(json.dumps(branch_a, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
