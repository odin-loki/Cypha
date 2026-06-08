"""Domain 17 — CyphaLM integration (WikiText / Gutenberg)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BENCH = Path(__file__).resolve().parents[1]
_REPO = _BENCH.parent
for path in (_REPO, _BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bench_common import finalize_domain

from cypha_bench.adapters.char_lstm_baseline import char_lstm_baseline_bpc
from cypha_bench.adapters.cyphalm_ablations import run_lm_ablations
from cypha_bench.adapters.cyphalm_component_study import run_component_study
from cypha_bench.adapters.cyphalm_bench import (
    bigram_baseline_bpc,
    cyphalm_bench_limits,
    cyphalm_dif_metrics,
    eval_held_out_bpc,
    fivegram_baseline_bpc,
    fourgram_baseline_bpc,
    load_cyphalm_config,
    make_cyphalm,
    prepare_lm_corpus,
    require_real_corpus,
    train_with_learning_curve,
    trigram_baseline_bpc,
)
from cypha_bench.common.metrics import save_figure
from cypha_bench.common.paths import is_fast


def _load_corpus():
    limits = cyphalm_bench_limits()
    corpus = prepare_lm_corpus(prefer_wikitext=True, max_train_chars=limits["max_corpus_chars"])
    require_real_corpus(corpus.source, domain="D17")
    return corpus


def _lm_baseline_metrics(train_slice, test_ids, vocab_size, limits) -> dict:
    metrics = {
        "bigram_bpc": bigram_baseline_bpc(train_slice, test_ids, vocab_size),
        "trigram_bpc": trigram_baseline_bpc(train_slice, test_ids, vocab_size),
    }
    if _phase1c_mode():
        return metrics
    if not is_fast():
        metrics["fourgram_bpc"] = fourgram_baseline_bpc(train_slice, test_ids, vocab_size)
        metrics["fivegram_bpc"] = fivegram_baseline_bpc(train_slice, test_ids, vocab_size)
        metrics["char_lstm_bpc"] = char_lstm_baseline_bpc(
            train_slice,
            test_ids,
            vocab_size,
            n_train_steps=limits["n_train"],
        )
    return metrics


def _phase1c_mode() -> bool:
    return os.environ.get("CYPHA_BENCH_FULL_CORPUS", "0") == "1"


def experiment_17a_bpc():
    limits = cyphalm_bench_limits()
    corpus = _load_corpus()
    if len(corpus.train_ids) < 512:
        return {"skipped": True, "source": corpus.source}

    model, cfg = make_cyphalm(profile="d17")
    n_train = min(limits["n_train"], len(corpus.train_ids) - 1)
    train_slice = corpus.train_ids[: n_train + 1]

    print(
        f"[17A] n_train={n_train} context_mode={cfg.context_mode} "
        f"phase1c={_phase1c_mode()}",
        flush=True,
    )
    t0 = time.perf_counter()
    if _phase1c_mode():
        train_out = model.train_sequence(train_slice)
        losses = train_out.get("loss", np.asarray([], dtype=np.float64))
        tail = max(1, min(5000, len(losses)))
        online_train_bpc = (
            float(np.mean(losses[-tail:]) / np.log(2)) if len(losses) else float("nan")
        )
        held_out_bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=limits["n_eval"])
        curve = {
            "steps": [n_train],
            "held_out_bpc": [held_out_bpc],
            "expert_count": [0],
            "trained_steps": n_train,
            "online_train_bpc": online_train_bpc,
            "final_train_bpc": online_train_bpc,
        }
    else:
        curve = train_with_learning_curve(
            model,
            corpus.train_ids,
            corpus.eval_ids,
            n_train=n_train,
            snapshot_every=limits["snapshot_every"],
            n_eval_snapshot=min(200, limits["n_eval"]),
        )
        held_out_bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=limits["n_eval"])
    print(f"[17A] train+eval done in {time.perf_counter() - t0:.0f}s bpc={held_out_bpc:.4f}", flush=True)

    baselines = _lm_baseline_metrics(train_slice, corpus.eval_ids, corpus.vocab_size, limits)
    bigram_bpc = baselines["bigram_bpc"]
    trigram_bpc = baselines["trigram_bpc"]

    if _phase1c_mode():
        ablations = {"skipped": True, "reason": "phase1c — see cyphalm_component_ablation.json"}
    else:
        ablation_modes = (
            ["full", "gria_ngram"]
            if is_fast()
            else ["full", "gria_ngram", "ssm_only", "ablation_no_dif", "ablation_no_ssm"]
        )
        ablations = run_lm_ablations(corpus, limits, ablation_modes, profile="d17")

    out = {
        "cyphalm_bpc": held_out_bpc,
        "online_train_bpc": curve["online_train_bpc"],
        "final_train_bpc": curve["final_train_bpc"],
        "bigram_bpc": bigram_bpc,
        "trigram_bpc": trigram_bpc,
        **baselines,
        "delta_vs_bigram": held_out_bpc - bigram_bpc,
        "delta_vs_trigram": held_out_bpc - trigram_bpc,
        "ablations": ablations,
        "source": corpus.source,
        "split": corpus.split,
        "train_tokens": n_train,
        "learning_curve": curve,
        "device": model.device,
        "profile": load_cyphalm_config(profile="d17"),
        "cypha_dif": cyphalm_dif_metrics(model),
    }
    if getattr(model, "lstm_head", None) is not None:
        out["hybrid_blend_logit"] = float(getattr(model, "_hybrid_blend_logit", 0.0))
    return out


def experiment_17b_alpha_spectrum():
    limits = cyphalm_bench_limits()
    corpus = _load_corpus()
    model, _ = make_cyphalm(profile="d17")
    n = min(limits["n_train"], len(corpus.train_ids) - 1)
    if _phase1c_mode():
        n = min(n, 40_000)
    print(f"[17B] training n={n} for alpha spectrum", flush=True)
    model.train_sequence(corpus.train_ids[: n + 1])
    profile = model.compression_profile()
    expert_alpha = np.asarray(profile.get("expert_alpha_spectrum", []), dtype=np.float64)
    if expert_alpha.size == 0:
        frac_edge = float("nan")
    else:
        frac_edge = float(np.mean(np.abs(expert_alpha - 0.5) < 0.1))
    return {
        "mean_alpha": float(profile.get("mean_alpha", float("nan"))),
        "fraction_edge_of_chaos": frac_edge,
        "n_experts": int(profile.get("n_experts", 0)),
    }


def _eval_view_schedule_bpc(
    view_schedule: str,
    corpus,
    *,
    n_train: int,
    n_eval: int,
) -> float:
    """Train CyphaLM with a view-schedule preset and return held-out BPC."""
    merged = load_cyphalm_config({"view_schedule": view_schedule}, profile="d17")
    model, _ = make_cyphalm(merged, profile=None)
    train_slice = corpus.train_ids[: min(n_train + 1, len(corpus.train_ids))]
    model.train_sequence(train_slice)
    return eval_held_out_bpc(model, corpus.eval_ids, n_eval=n_eval)


def experiment_17e_multi_view():
    """Compare multi-view schedules vs same-order baseline on WikiText."""
    limits = cyphalm_bench_limits()
    corpus = _load_corpus()
    if len(corpus.train_ids) < 512:
        return {"skipped": True, "source": corpus.source}

    n_train = min(limits["n_train"], len(corpus.train_ids) - 1)
    n_eval = limits["n_eval"]

    same_order_bpc = _eval_view_schedule_bpc(
        "same_order", corpus, n_train=n_train, n_eval=n_eval
    )
    schedule_a_bpc = _eval_view_schedule_bpc(
        "schedule_a", corpus, n_train=n_train, n_eval=n_eval
    )

    out: dict = {
        "same_order_bpc": same_order_bpc,
        "schedule_a_bpc": schedule_a_bpc,
        "delta_schedule_a": schedule_a_bpc - same_order_bpc,
        "n_train": n_train,
        "source": corpus.source,
        "view_block_size": int(load_cyphalm_config(profile="d17").get("view_block_size", 512)),
    }

    if not is_fast():
        schedule_b_bpc = _eval_view_schedule_bpc(
            "schedule_b", corpus, n_train=n_train, n_eval=n_eval
        )
        out["schedule_b_bpc"] = schedule_b_bpc
        out["delta_schedule_b"] = schedule_b_bpc - same_order_bpc

    return out


def experiment_17g_beat_bigram_schedule_b() -> dict:
    """schedule_b at 70k/150k train steps — beat-bigram gate for Phase 1."""
    limits = cyphalm_bench_limits()
    max_chars = limits["max_corpus_chars"]
    if not is_fast():
        max_chars = max(max_chars, 10_000_000)
    corpus = prepare_lm_corpus(prefer_wikitext=True, max_train_chars=max_chars)
    require_real_corpus(corpus.source, domain="D17")
    if len(corpus.train_ids) < 512:
        return {"skipped": True, "source": corpus.source}

    env_grid = os.environ.get("CYPHA_BEAT_BIGRAM_N_TRAIN", "").strip()
    if env_grid:
        n_train_grid = [int(x.strip()) for x in env_grid.split(",") if x.strip()]
    elif is_fast():
        n_train_grid = [min(8000, limits["n_train"], len(corpus.train_ids) - 1)]
    else:
        n_train_grid = [70000, 150000]

    n_eval = limits["n_eval"]
    merged = load_cyphalm_config({"view_schedule": "schedule_b"}, profile="d17")
    runs: list[dict] = []
    for n_train in n_train_grid:
        limit = min(n_train, len(corpus.train_ids) - 1)
        t0 = time.perf_counter()
        model, _ = make_cyphalm(merged, profile=None)
        train_slice = corpus.train_ids[:limit]
        model.train_sequence(train_slice)
        cyphalm_bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=n_eval)
        train_seconds = time.perf_counter() - t0
        bigram_bpc = bigram_baseline_bpc(train_slice, corpus.eval_ids, corpus.vocab_size)
        runs.append(
            {
                "n_train": limit,
                "cyphalm_bpc": float(cyphalm_bpc),
                "bigram_bpc": float(bigram_bpc),
                "delta_vs_bigram": float(cyphalm_bpc - bigram_bpc),
                "train_seconds": train_seconds,
            }
        )

    return {
        "view_schedule": "schedule_b",
        "runs": runs,
        "fast_reduced": is_fast(),
        "n_eval": n_eval,
        "source": corpus.source,
    }


def experiment_17h_component_ablation() -> dict:
    """Systematic subsystem ablation — architecture, toggles, SSM combos, upgrades."""
    limits = cyphalm_bench_limits()
    max_chars = limits["max_corpus_chars"]
    if not is_fast():
        max_chars = max(max_chars, 10_000_000)
    corpus = prepare_lm_corpus(prefer_wikitext=True, max_train_chars=max_chars)
    require_real_corpus(corpus.source, domain="D17")
    if len(corpus.train_ids) < 512:
        return {"skipped": True, "source": corpus.source}

    n_train = limits["n_train"]
    n_eval = limits["n_eval"]
    out = run_component_study(
        corpus,
        n_train=n_train,
        n_eval=n_eval,
        fast=is_fast(),
        default_profile="d17",
    )
    if not is_fast():
        from cypha_bench.tuning.cyphalm_component_ablation import _OUT

        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return {
        "n_train": n_train,
        "n_eval": n_eval,
        "cells_run": out.get("cells_run"),
        "global_best": out.get("global_best"),
        "comparisons": out.get("comparisons"),
        "phase_best": out.get("phase_best"),
        "fast_reduced": is_fast(),
        "source": corpus.source,
    }


def experiment_17f_iteration_view_sweep() -> dict:
    """Train-length × view-schedule sweep (optimal iterations per presentation mode)."""
    from cypha_bench.tuning.cyphalm_view_iteration_sweep import (
        FAST_N_TRAIN,
        TRAINING_MODES,
        _OUT,
        run_sweep,
    )

    if is_fast():
        grid = FAST_N_TRAIN
        n_eval = 300
    else:
        grid = [2000, 4000, 8000, 12000, 16000, 24000, 32000, 40000]
        n_eval = 2000

    out = run_sweep(n_train_grid=grid, n_eval=n_eval, modes=TRAINING_MODES)
    if not is_fast():
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return {
        "grid_n_train": grid,
        "n_eval": n_eval,
        "best_per_mode": out.get("best_per_mode"),
        "winner_by_n_train": out.get("winner_by_n_train"),
        "global_best": out.get("global_best"),
        "elapsed_s": out.get("elapsed_s"),
        "output_path": str(_OUT) if not is_fast() else None,
    }


def experiment_17k_long_range_context() -> dict:
    """Cypha Tests Phase 1: long-range context, sequential vs shuffled, SSM ablation."""
    from cypha_bench.adapters.cyphalm_long_range import run_long_range_suite
    from cypha_bench.tuning.cyphalm_long_range_suite import _OUT

    limits = cyphalm_bench_limits()
    corpus = _load_corpus()
    n_train = min(limits["n_train"], len(corpus.train_ids) - 1)
    n_eval = limits["n_eval"]
    fast = is_fast()

    if _phase1c_mode():
        prior = _BENCH / "config" / "cyphalm_long_range_300k.json"
        if prior.exists():
            print(f"[17K] loading prior long-range artifact {prior}", flush=True)
            out = json.loads(prior.read_text(encoding="utf-8"))
            return {
                "n_train": out.get("n_train", n_train),
                "n_eval": n_eval,
                "held_out_bpc": out.get("held_out_bpc"),
                "context_length_bpc": out.get("context_length_bpc"),
                "reset_interval_bpc": out.get("reset_interval_bpc"),
                "sequential_vs_shuffled": out.get("sequential_vs_shuffled"),
                "ssm_ablation_sequential": out.get("ssm_ablation_sequential"),
                "phase1c_stub": True,
                "source": out.get("corpus"),
            }

    out = run_long_range_suite(
        corpus,
        n_train=n_train,
        n_eval=n_eval,
        profile="d17",
        fast=fast,
        skip_ablation=_phase1c_mode(),
    )
    if not fast:
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        from cypha_bench.adapters.cyphalm_long_range import save_long_range_figures

        save_long_range_figures(out)
    return {
        "n_train": n_train,
        "n_eval": n_eval,
        "held_out_bpc": out.get("held_out_bpc"),
        "context_length_bpc": out.get("context_length_bpc"),
        "reset_interval_bpc": out.get("reset_interval_bpc"),
        "sequential_vs_shuffled": out.get("sequential_vs_shuffled"),
        "ssm_ablation_sequential": out.get("ssm_ablation_sequential"),
        "output_path": str(_OUT) if not fast else None,
        "source": corpus.source,
    }


def experiment_17i_view_learnable() -> dict:
    """Upgrade V2 Track A: fixed vs learnable view embeddings."""
    from cypha_bench.tuning.cyphalm_upgrade_v2_sweep import _OUT, run_sweep

    limits = cyphalm_bench_limits()
    n_train = min(limits["n_train"], 300_000) if not is_fast() else min(limits["n_train"], 40_000)
    n_eval = limits["n_eval"]
    out = run_sweep(n_train=n_train, n_eval=n_eval, profile="d17")
    if not is_fast():
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return {
        "n_train": n_train,
        "n_eval": n_eval,
        "bigram_bpc": out.get("bigram_bpc"),
        "cells": out.get("cells"),
        "learnable_minus_baseline_bpc": out.get("learnable_minus_baseline_bpc"),
        "best": out.get("best"),
        "output_path": str(_OUT) if not is_fast() else None,
    }


def experiment_17j_hybrid_lstm() -> dict:
    """Model-class C2: GRIA-only vs hybrid GRIA + char-LSTM."""
    from cypha_bench.tuning.cyphalm_hybrid_lstm_sweep import _OUT, CELLS
    import cypha_bench.tuning.cyphalm_hybrid_lstm_sweep as sweep_mod

    limits = cyphalm_bench_limits()
    n_train = min(limits["n_train"], 40_000 if is_fast() else 300_000)
    n_eval = limits["n_eval"]
    corpus = _load_corpus()
    train_slice = corpus.train_ids[: min(n_train, len(corpus.train_ids) - 1)]
    from cypha_bench.adapters.char_lstm_baseline import char_lstm_baseline_bpc
    from cypha_bench.adapters.cyphalm_bench import bigram_baseline_bpc

    rows = []
    for cell in CELLS:
        rows.append(
            sweep_mod._run_cell(
                corpus, cell=cell, n_train=n_train, n_eval=n_eval, profile="d17"
            )
        )
    bigram = bigram_baseline_bpc(train_slice, corpus.eval_ids, corpus.vocab_size)
    char_lstm = char_lstm_baseline_bpc(
        train_slice, corpus.eval_ids, corpus.vocab_size, n_train_steps=len(train_slice)
    )
    for row in rows:
        row["delta_vs_bigram"] = row["held_out_bpc"] - bigram
    gria = next(r for r in rows if r["cell_id"] == "gria_ngram")
    hybrid = next(r for r in rows if r["cell_id"] == "hybrid_gria_lstm")
    out = {
        "n_train": n_train,
        "cells": rows,
        "char_lstm_baseline_bpc": float(char_lstm),
        "hybrid_minus_gria_bpc": float(gria["held_out_bpc"] - hybrid["held_out_bpc"]),
    }
    if not is_fast():
        _OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return {**out, "output_path": str(_OUT) if not is_fast() else None}


def experiment_17d_online_adaptation():
    from cypha_bench.adapters.cyphalm_bench import encode_text
    from cypha_bench.common.paths import DATA_DIR

    corpus = _load_corpus()
    holmes = DATA_DIR / "gutenberg" / "sherlock_holmes.txt"
    if holmes.exists():
        text_b = holmes.read_text(encoding="utf-8", errors="ignore")[:20_000]
        ids_b = encode_text(text_b, corpus.char2id)
    else:
        ids_b = list(reversed(corpus.train_ids[:4000]))

    model, _ = make_cyphalm(profile="d17")
    warm = min(2000, len(corpus.train_ids) - 1)
    model.train_sequence(corpus.train_ids[:warm])
    pre = model.train_sequence(ids_b[:500])
    bpc_before = float(np.sum(pre["loss"]) / max(len(pre["loss"]), 1) / np.log(2))
    model.train_sequence(ids_b[500:3500])
    post = model.train_sequence(ids_b[3500:4000])
    bpc_after = float(np.sum(post["loss"]) / max(len(post["loss"]), 1) / np.log(2))
    return {
        "bpc_ood_before_adapt": bpc_before,
        "bpc_ood_after_adapt": bpc_after,
        "bpc_improvement": bpc_before - bpc_after,
    }


def _save_learning_figure(result: dict) -> None:
    curve = result.get("learning_curve") or {}
    steps = curve.get("steps") or []
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, curve["held_out_bpc"], marker="o", label="CyphaLM held-out")
    bb = result.get("bigram_bpc")
    tb = result.get("trigram_bpc")
    if bb is not None:
        ax.axhline(bb, color="gray", linestyle="--", label=f"Bigram ({bb:.2f})")
    if tb is not None:
        ax.axhline(tb, color="orange", linestyle=":", label=f"Trigram ({tb:.2f})")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Bits per character")
    ax.set_title(f"D17 CyphaLM ({result.get('source', '?')})")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, "fig17_learning_curve")
    plt.close(fig)


def run() -> dict:
    try:
        exp17a = experiment_17a_bpc()
        if not exp17a.get("skipped"):
            _save_learning_figure(exp17a)
        experiments: dict = {
            "17A_bits_per_character": exp17a,
            "17B_alpha_spectrum": experiment_17b_alpha_spectrum(),
            "17D_online_adaptation": experiment_17d_online_adaptation(),
        }
        if _phase1c_mode():
            print("[D17] phase1c: core experiments only (17A, 17B, 17D, 17K)", flush=True)
            experiments["17K_long_range_context"] = experiment_17k_long_range_context()
        else:
            experiments.update(
                {
                    "17E_multi_view": experiment_17e_multi_view(),
                    "17F_iteration_view_sweep": experiment_17f_iteration_view_sweep(),
                    "17G_beat_bigram_schedule_b": experiment_17g_beat_bigram_schedule_b(),
                    "17H_component_ablation": experiment_17h_component_ablation(),
                    "17K_long_range_context": experiment_17k_long_range_context(),
                    "17I_view_learnable": experiment_17i_view_learnable(),
                    "17J_hybrid_lstm": experiment_17j_hybrid_lstm(),
                }
            )
    except ImportError as exc:
        experiments = {
            "skipped": True,
            "reason": f"cypha_lm not available: {exc}",
        }
    except RuntimeError as exc:
        experiments = {
            "skipped": True,
            "reason": str(exc),
        }
    return finalize_domain("d17", experiments)


if __name__ == "__main__":
    print(run())
