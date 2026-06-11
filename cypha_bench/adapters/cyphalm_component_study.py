"""Systematic CyphaLM component ablation — isolate and combine subsystems."""

from __future__ import annotations

import time
from typing import Any

from cypha_bench.adapters.cyphalm_bench import (
    bigram_baseline_bpc,
    eval_held_out_bpc,
    load_cyphalm_config,
    make_cyphalm,
    trigram_baseline_bpc,
)
from cypha_bench.common.paths import is_fast

# ---------------------------------------------------------------------------
# Phase A — Architecture paths (mutually exclusive context_mode)
# Each cell answers: what does this pipeline variant achieve alone?
# ---------------------------------------------------------------------------
ARCHITECTURE_CELLS: list[dict[str, Any]] = [
    {
        "id": "arch_full",
        "phase": "architecture",
        "label": "Full: SSM -> DIF (epistemic) -> GRIA",
        "overrides": {"context_mode": "full"},
    },
    {
        "id": "arch_gria_ngram",
        "phase": "architecture",
        "label": "GRIA+ngram: SSM + embed history -> GRIA (DIF zeroed in input)",
        "overrides": {"context_mode": "gria_ngram", "ngram_context": 2},
    },
    {
        "id": "arch_ssm_only",
        "phase": "architecture",
        "label": "SSM only: field_x -> GRIA, no DIF",
        "overrides": {"context_mode": "ssm_only"},
    },
    {
        "id": "arch_no_dif",
        "phase": "architecture",
        "label": "No DIF routing: DIF mean only, no epistemic term",
        "overrides": {"context_mode": "ablation_no_dif"},
    },
    {
        "id": "arch_no_ssm",
        "phase": "architecture",
        "label": "No SSM: n-gram embed stack only",
        "overrides": {"context_mode": "ablation_no_ssm", "ngram_context": 2},
    },
]

# ---------------------------------------------------------------------------
# Phase B — Single-toggle ablations on gria_ngram (current best architecture)
# Each cell: remove or change ONE subsystem vs profile baseline.
# ---------------------------------------------------------------------------
GRIA_NGRAM_BASE: dict[str, Any] = {
    "context_mode": "gria_ngram",
    "ngram_context": 2,
}


def _toggle(id_: str, label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {**GRIA_NGRAM_BASE, **overrides}
    return {"id": id_, "phase": "toggle", "label": label, "overrides": merged}


TOGGLE_CELLS: list[dict[str, Any]] = [
    _toggle("toggle_baseline", "Profile gria_ngram baseline (no extra overrides)", {}),
    _toggle("toggle_no_bptt", "Remove BPTT (bptt_steps=0)", {"bptt_steps": 0}),
    _toggle("toggle_no_laplace", "Remove Laplace bias prior", {"laplace_smoothing": 0.0}),
    _toggle("toggle_dif_offline", "DIF predict-only (online=False)", {"online": False}),
    _toggle("toggle_frozen_alpha", "Frozen GRIA alpha (alpha_learnable=False)", {"alpha_learnable": False}),
    _toggle("toggle_no_multiscale", "SSM single-scale (use_multiscale=False)", {"use_multiscale": False}),
    _toggle("toggle_spectral_pde", "SSM spectral PDE path on", {"use_spectral_pde": True}),
    _toggle("toggle_sparse_hebbian", "SSM sparse Hebbian on", {"use_sparse_hebbian": True}),
    _toggle("toggle_train_ssm", "Hebbian SSM update (train_ssm=True, no BPTT)", {"train_ssm": True, "bptt_steps": 0}),
    _toggle("toggle_ngram1", "Shorter n-gram window (ngram_context=1)", {"ngram_context": 1}),
    _toggle("toggle_ngram3", "Longer n-gram window (ngram_context=3)", {"ngram_context": 3}),
    _toggle("toggle_ngram4", "Longest n-gram window (ngram_context=4)", {"ngram_context": 4}),
    _toggle("toggle_experts4", "Warm-start 4 DIF experts", {"n_experts": 4}),
    _toggle("toggle_tau_tight", "Tighter memory (tau_fast=0.5, tau_slow=10)", {"tau_fast": 0.5, "tau_slow": 10.0}),
    _toggle("toggle_no_lr_decay", "Constant GRIA LR across epochs (gria_lr_decay=1)", {"gria_lr_decay": 1.0}),
    _toggle("toggle_same_order_e1", "Single forward pass (train_epochs=1)", {"train_epochs": 1}),
]

# ---------------------------------------------------------------------------
# Phase C — SSM feature combinatorics (2³ = 8) on gria_ngram
# ---------------------------------------------------------------------------
def _ssm_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for spectral in (False, True):
        for multiscale in (False, True):
            for hebbian in (False, True):
                flags = {
                    "use_spectral_pde": spectral,
                    "use_multiscale": multiscale,
                    "use_sparse_hebbian": hebbian,
                }
                tag = "".join(
                    [
                        "S" if spectral else "s",
                        "M" if multiscale else "m",
                        "H" if hebbian else "h",
                    ]
                )
                cells.append(
                    {
                        "id": f"ssm_{tag}",
                        "phase": "ssm_combo",
                        "label": f"SSM flags spectral={spectral} multiscale={multiscale} hebbian={hebbian}",
                        "overrides": {**GRIA_NGRAM_BASE, **flags},
                    }
                )
    return cells


SSM_COMBO_CELLS = _ssm_cells()

# ---------------------------------------------------------------------------
# Phase D — Baseline origins + upgrade stacks
# ---------------------------------------------------------------------------
UPGRADE_CELLS: list[dict[str, Any]] = [
    {
        "id": "origin_factory_defaults",
        "phase": "upgrade",
        "label": "AI factory defaults (context_mode=full, no BPTT, 1 epoch)",
        "overrides": {
            "context_mode": "full",
            "train_epochs": 1,
            "bptt_steps": 0,
            "view_schedule": "same_order",
            "laplace_smoothing": 1.0,
            "gria_lr": 0.05,
            "train_ssm": False,
            "online": True,
        },
        "profile": None,
    },
    {
        "id": "origin_bench_default",
        "phase": "upgrade",
        "label": "Bench DEFAULT_CYPHALM_CONFIG (pre-profile)",
        "overrides": {},
        "profile": None,
        "use_bench_defaults_only": True,
    },
    {
        "id": "upgrade_profile_d17",
        "phase": "upgrade",
        "label": "Current D17 profile (gria_ngram + BPTT + 2 epochs)",
        "overrides": {},
        "profile": "d17",
    },
    {
        "id": "upgrade_gria_tau_tight",
        "phase": "upgrade",
        "label": "gria_ngram + tight memory",
        "overrides": {**GRIA_NGRAM_BASE, "tau_fast": 0.5, "tau_slow": 10.0},
    },
    {
        "id": "upgrade_gria_schedule_b",
        "phase": "upgrade",
        "label": "gria_ngram + schedule_b multi-view",
        "overrides": {**GRIA_NGRAM_BASE, "view_schedule": "schedule_b"},
    },
    {
        "id": "upgrade_full_tau_tight",
        "phase": "upgrade",
        "label": "Full pipeline + tight memory",
        "overrides": {"context_mode": "full", "tau_fast": 0.5, "tau_slow": 10.0},
    },
    {
        "id": "upgrade_full_bptt_laplace_off",
        "phase": "upgrade",
        "label": "Full + BPTT64 + no Laplace",
        "overrides": {"context_mode": "full", "bptt_steps": 64, "laplace_smoothing": 0.0},
    },
    {
        "id": "upgrade_stack_best",
        "phase": "upgrade",
        "label": "Stack: gria_ngram + schedule_b + tau_tight + ngram3",
        "overrides": {
            **GRIA_NGRAM_BASE,
            "view_schedule": "schedule_b",
            "tau_fast": 0.5,
            "tau_slow": 10.0,
            "ngram_context": 3,
            "gria_lr_decay": 1.0,
        },
    },
]

FAST_PHASES = {"architecture", "toggle"}
FAST_CELL_IDS = {
    "arch_full",
    "arch_gria_ngram",
    "arch_no_ssm",
    "toggle_baseline",
    "toggle_no_bptt",
    "toggle_no_laplace",
    "toggle_dif_offline",
    "origin_factory_defaults",
    "upgrade_profile_d17",
}


def all_cells(*, fast: bool = False) -> list[dict[str, Any]]:
    cells = ARCHITECTURE_CELLS + TOGGLE_CELLS + SSM_COMBO_CELLS + UPGRADE_CELLS
    if fast:
        cells = [c for c in cells if c["phase"] in FAST_PHASES or c["id"] in FAST_CELL_IDS]
    return cells


def eval_cell(
    corpus,
    cell: dict[str, Any],
    *,
    n_train: int,
    n_eval: int,
    default_profile: str = "d17",
) -> dict[str, Any]:
    """Train and score one ablation cell."""
    profile = cell.get("profile", default_profile)
    overrides = dict(cell.get("overrides") or {})

    if cell.get("use_bench_defaults_only"):
        merged = load_cyphalm_config(overrides, profile=None)
    else:
        merged = load_cyphalm_config(overrides, profile=profile if profile else None)

    model, cfg = make_cyphalm(merged, profile=None)
    limit = min(n_train, len(corpus.train_ids) - 1)
    train_ids = corpus.train_ids[: limit + 1]

    t0 = time.perf_counter()
    model.train_sequence(train_ids)
    train_s = time.perf_counter() - t0

    bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=n_eval)
    bigram = bigram_baseline_bpc(train_ids, corpus.eval_ids, corpus.vocab_size)
    trigram = trigram_baseline_bpc(train_ids, corpus.eval_ids, corpus.vocab_size)

    snap = {
        k: getattr(cfg, k)
        for k in (
            "context_mode",
            "ngram_context",
            "bptt_steps",
            "laplace_smoothing",
            "online",
            "alpha_learnable",
            "use_spectral_pde",
            "use_multiscale",
            "use_sparse_hebbian",
            "train_ssm",
            "train_epochs",
            "view_schedule",
            "gria_lr",
            "tau_fast",
            "tau_slow",
            "n_experts",
            "gria_lr_decay",
        )
    }

    return {
        "id": cell["id"],
        "phase": cell["phase"],
        "label": cell["label"],
        "held_out_bpc": float(bpc),
        "bigram_bpc": float(bigram),
        "trigram_bpc": float(trigram),
        "delta_vs_bigram": float(bpc - bigram),
        "delta_vs_trigram": float(bpc - trigram),
        "n_train": int(limit),
        "n_eval": int(n_eval),
        "train_seconds": float(train_s),
        "config": snap,
    }


def run_component_study(
    corpus,
    *,
    n_train: int,
    n_eval: int,
    phases: set[str] | None = None,
    fast: bool | None = None,
    default_profile: str = "d17",
) -> dict[str, Any]:
    """Run all ablation cells and aggregate by phase."""
    use_fast = is_fast() if fast is None else fast
    cells = all_cells(fast=use_fast)
    if phases:
        cells = [c for c in cells if c["phase"] in phases]

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    for cell in cells:
        print(f"[{cell['phase']}] {cell['id']}: {cell['label']}")
        try:
            row = eval_cell(
                corpus,
                cell,
                n_train=n_train,
                n_eval=n_eval,
                default_profile=default_profile,
            )
            results.append(row)
            print(
                f"  -> bpc={row['held_out_bpc']:.4f} "
                f"d_bi={row['delta_vs_bigram']:+.4f} ({row['train_seconds']:.0f}s)"
            )
        except Exception as exc:
            print(f"  -> FAIL: {exc}")
            results.append(
                {
                    "id": cell["id"],
                    "phase": cell["phase"],
                    "label": cell["label"],
                    "error": str(exc),
                }
            )

    ok = [r for r in results if "held_out_bpc" in r]
    by_phase: dict[str, list] = {}
    for r in ok:
        by_phase.setdefault(r["phase"], []).append(r)

    phase_best: dict[str, Any] = {}
    for phase, rows in by_phase.items():
        phase_best[phase] = min(rows, key=lambda x: x["held_out_bpc"])

    global_best = min(ok, key=lambda x: x["held_out_bpc"]) if ok else None

    # Key comparisons
    comparisons: dict[str, Any] = {}
    by_id = {r["id"]: r for r in ok}
    if "arch_full" in by_id and "arch_gria_ngram" in by_id:
        comparisons["full_minus_gria_ngram_bpc"] = (
            by_id["arch_full"]["held_out_bpc"] - by_id["arch_gria_ngram"]["held_out_bpc"]
        )
    if "origin_factory_defaults" in by_id and "upgrade_profile_d17" in by_id:
        comparisons["profile_gain_vs_factory_bpc"] = (
            by_id["origin_factory_defaults"]["held_out_bpc"]
            - by_id["upgrade_profile_d17"]["held_out_bpc"]
        )
    if "toggle_baseline" in by_id and "toggle_no_bptt" in by_id:
        comparisons["bptt_contribution_bpc"] = (
            by_id["toggle_no_bptt"]["held_out_bpc"] - by_id["toggle_baseline"]["held_out_bpc"]
        )

    return {
        "corpus": corpus.source,
        "corpus_train_tokens": len(corpus.train_ids),
        "n_train": n_train,
        "n_eval": n_eval,
        "fast": use_fast,
        "cells_run": len(cells),
        "cells_ok": len(ok),
        "elapsed_s": time.perf_counter() - t0,
        "results": results,
        "by_phase": {k: sorted(v, key=lambda x: x["held_out_bpc"]) for k, v in by_phase.items()},
        "phase_best": phase_best,
        "global_best": global_best,
        "comparisons": comparisons,
    }
