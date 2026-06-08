"""CyphaLM long-range context and sequential tests (Cypha Tests.txt Phase 1)."""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from cypha_bench.adapters.cyphalm_bench import (
    eval_bpc_by_context_length,
    eval_held_out_bpc,
    make_cyphalm,
)
from cypha_bench.common.paths import scale


def _bpc_from_losses(losses: list[float]) -> float:
    if not losses:
        return float("nan")
    return float(np.mean(losses) / np.log(2))


def eval_reset_interval_bpc(
    model,
    test_ids: list[int],
    reset_intervals: tuple[int, ...] = (0, 8, 32, 64, 128, 256, 512),
    *,
    n_eval: int = 2000,
) -> dict[str, float]:
    """Eval BPC when context is reset every N tokens (0 = never reset mid-stream)."""
    n = min(n_eval, len(test_ids) - 1)
    out: dict[str, float] = {}
    for interval in reset_intervals:
        losses: list[float] = []
        model.reset_context()
        for i in range(n):
            if interval > 0 and i > 0 and i % interval == 0:
                model.reset_context()
            pred = model.predict_next(int(test_ids[i]))
            nxt = int(test_ids[i + 1])
            losses.append(float(-pred["log_probs"][nxt]))
        key = "never" if interval == 0 else str(interval)
        out[key] = _bpc_from_losses(losses)
    return out


def eval_shuffled_stream_bpc(
    model,
    test_ids: list[int],
    *,
    n_eval: int = 2000,
    block_size: int = 512,
    seed: int = 42,
) -> dict[str, float]:
    """Sequential vs block-shuffled eval on same token multiset (Cypha Tests 1A)."""
    n = min(n_eval, len(test_ids) - 1)
    slice_ids = test_ids[: n + 1]

    def _eval_stream(ids: list[int]) -> float:
        losses: list[float] = []
        model.reset_context()
        for i in range(len(ids) - 1):
            pred = model.predict_next(int(ids[i]))
            nxt = int(ids[i + 1])
            losses.append(float(-pred["log_probs"][nxt]))
        return _bpc_from_losses(losses)

    forward_bpc = _eval_stream(slice_ids)

    blocks = [
        slice_ids[i : i + block_size] for i in range(0, len(slice_ids), block_size) if slice_ids[i : i + block_size]
    ]
    rng = random.Random(seed)
    order = list(range(len(blocks)))
    rng.shuffle(order)
    shuffled = [tok for idx in order for tok in blocks[idx]]
    shuffled_bpc = _eval_stream(shuffled[: len(slice_ids)])

    return {
        "forward_bpc": forward_bpc,
        "block_shuffled_bpc": shuffled_bpc,
        "delta_shuffled_minus_forward": shuffled_bpc - forward_bpc,
    }


def eval_context_length_extended(
    model,
    test_ids: list[int],
    *,
    context_lengths: tuple[int, ...] | None = None,
    n_positions: int | None = None,
) -> dict[str, float]:
    """BPC vs warm-up window; extended grid for long-range probe."""
    if context_lengths is None:
        context_lengths = (8, 16, 32, 64, 128, 256, 512, 1024)
    return eval_bpc_by_context_length(
        model,
        test_ids,
        context_lengths=context_lengths,
        n_positions=n_positions or scale(120, 40),
    )


def eval_ssm_ablation_sequential(
    corpus,
    *,
    n_train: int,
    n_eval: int = 2000,
    profile: str = "d17",
) -> dict[str, Any]:
    """Cypha Tests 1A/1C: does SSM field help sequential LM?"""
    modes = [
        ("gria_ngram", {"context_mode": "gria_ngram"}),
        ("ssm_only", {"context_mode": "ssm_only"}),
        ("ablation_no_ssm", {"context_mode": "ablation_no_ssm", "ngram_context": 3}),
    ]
    limit = min(n_train, len(corpus.train_ids) - 1)
    train_ids = corpus.train_ids[: limit + 1]
    rows: dict[str, Any] = {}
    for label, overrides in modes:
        model, _ = make_cyphalm(overrides, profile=profile)
        t0 = time.perf_counter()
        model.train_sequence(train_ids)
        bpc = eval_held_out_bpc(model, corpus.eval_ids, n_eval=n_eval)
        rows[label] = {
            "held_out_bpc": float(bpc),
            "train_seconds": time.perf_counter() - t0,
        }
    g = rows.get("gria_ngram", {}).get("held_out_bpc")
    s = rows.get("ssm_only", {}).get("held_out_bpc")
    n = rows.get("ablation_no_ssm", {}).get("held_out_bpc")
    if g is not None and n is not None:
        rows["ssm_contribution_bpc"] = float(n - g)  # positive => SSM helps vs embed-only
    if g is not None and s is not None:
        rows["dif_ngram_contribution_bpc"] = float(s - g)
    return rows


def run_long_range_suite(
    corpus,
    *,
    n_train: int,
    n_eval: int = 2000,
    profile: str = "d17",
    fast: bool = False,
    skip_ablation: bool = False,
) -> dict[str, Any]:
    """Full long-range + sequential test battery on a trained model."""
    limit = min(n_train, len(corpus.train_ids) - 1)
    train_ids = corpus.train_ids[: limit + 1]
    test_ids = corpus.eval_ids

    model, cfg = make_cyphalm(profile=profile)
    t0 = time.perf_counter()
    model.train_sequence(train_ids)
    train_s = time.perf_counter() - t0

    ctx_lengths = (8, 32, 64, 128, 256) if fast else (8, 16, 32, 64, 128, 256, 512, 1024)
    reset_grid = (0, 32, 128, 512) if fast else (0, 8, 32, 64, 128, 256, 512)

    held_out = eval_held_out_bpc(model, test_ids, n_eval=n_eval)
    out: dict[str, Any] = {
        "corpus": corpus.source,
        "n_train": limit,
        "n_eval": n_eval,
        "profile": profile,
        "context_mode": getattr(cfg, "context_mode", None),
        "train_seconds": train_s,
        "held_out_bpc": float(held_out),
        "context_length_bpc": eval_context_length_extended(
            model, test_ids, context_lengths=ctx_lengths, n_positions=scale(60, 25) if fast else None
        ),
        "reset_interval_bpc": eval_reset_interval_bpc(
            model, test_ids, reset_intervals=reset_grid, n_eval=n_eval
        ),
        "sequential_vs_shuffled": eval_shuffled_stream_bpc(model, test_ids, n_eval=n_eval),
    }
    if not fast and not skip_ablation:
        out["ssm_ablation_sequential"] = eval_ssm_ablation_sequential(
            corpus, n_train=n_train, n_eval=n_eval, profile=profile
        )
    return out


def save_long_range_figures(result: dict[str, Any], *, prefix: str = "fig17_long_range") -> None:
    """Write context-length and reset-interval curves to report/figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cypha_bench.common.metrics import save_figure

    ctx = result.get("context_length_bpc") or {}
    if ctx:
        xs = sorted(int(k) for k in ctx)
        ys = [ctx[str(x)] for x in xs]
        n_train = result.get("n_train", "")
        tag = f"_{n_train}" if n_train else ""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="o", label="CyphaLM")
        held = result.get("held_out_bpc")
        if held is not None:
            ax.axhline(float(held), color="gray", linestyle="--", label=f"Stream eval ({held:.2f})")
        ax.set_xlabel("SSM warm-up length (tokens)")
        ax.set_ylabel("Held-out BPC")
        ax.set_title(f"Long-range context @ n_train={result.get('n_train', '?')}")
        ax.set_xscale("log", base=2)
        ax.legend()
        plt.tight_layout()
        save_figure(fig, f"{prefix}{tag}_context_bpc")
        plt.close(fig)

    reset = result.get("reset_interval_bpc") or {}
    if reset:
        xs = [0 if k == "never" else int(k) for k in reset]
        pairs = sorted(zip(xs, reset.values()), key=lambda t: t[0])
        xs_o = [p[0] if p[0] > 0 else 1 for p in pairs]
        ys = [p[1] for p in pairs]
        n_train = result.get("n_train", "")
        tag = f"_{n_train}" if n_train else ""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs_o, ys, marker="o")
        ax.set_xscale("symlog", linthresh=8)
        ax.set_xlabel("Reset interval (tokens; 1 = never)")
        ax.set_ylabel("Held-out BPC")
        ax.set_title("BPC vs memory reset interval")
        plt.tight_layout()
        save_figure(fig, f"{prefix}{tag}_reset_interval")
        plt.close(fig)
