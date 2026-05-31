"""Domain 04 — character-level language modelling with CyphaLM.

Pipeline: Izaac embedding → CellAI SSM → CyphaDIF expert field → GRIA projection.

Experiments:
  - Held-out BPC + training curve
  - BPC vs context length (CellAI SSM warm-up)
  - CyphaDIF expert routing during generation
  - Save/restore checkpoint fidelity
  - Sampling strategy comparison (greedy, temperature, top-k, top-p, uncertainty-gated)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cypha_bench.adapters.cyphalm_bench import (
    DEFAULT_CYPHALM_CONFIG,
    bigram_baseline_bpc,
    build_char_vocab,
    compare_sampling_strategies,
    cyphalm_bench_limits,
    cyphalm_dif_metrics,
    decode_ids,
    encode_text,
    eval_bpc_by_context_length,
    eval_held_out_bpc,
    eval_held_out_bpc_preserve_training,
    eval_save_restore_fidelity,
    expert_routing_trace,
    load_char_corpus,
    make_cyphalm,
)
from cypha_bench.common.metrics import save_figure, save_table


def _train_with_snapshots(
    model,
    train_ids: list[int],
    test_ids: list[int],
    *,
    n_train: int,
    snapshot_every: int,
    n_eval_snapshot: int,
) -> dict:
    steps: list[int] = []
    bpc_curve: list[float] = []
    expert_curve: list[int] = []
    epistemic_curve: list[float] = []
    alpha_curve: list[float] = []
    train_losses: list[float] = []

    model.reset_context()
    limit = min(n_train, len(train_ids) - 1)
    for t in range(limit):
        metrics = model.train_step(int(train_ids[t]), int(train_ids[t + 1]))
        train_losses.append(float(metrics["loss"]))
        trained = t + 1
        if trained % snapshot_every == 0 or trained == limit:
            snap_bpc = eval_held_out_bpc_preserve_training(
                model, test_ids, n_eval=n_eval_snapshot
            )
            dif = cyphalm_dif_metrics(model)
            steps.append(trained)
            bpc_curve.append(snap_bpc)
            expert_curve.append(int(dif["n_experts"]))
            epistemic_curve.append(float(metrics["epistemic_var"]))
            alpha_curve.append(float(metrics["alpha_gria"]))

    tail = max(1, len(train_losses) // 10)
    online_train_bpc = float(np.mean(train_losses) / np.log(2)) if train_losses else float("nan")
    final_train_bpc = float(np.mean(train_losses[-tail:]) / np.log(2)) if train_losses else float("nan")

    return {
        "steps": steps,
        "held_out_bpc": bpc_curve,
        "expert_count": expert_curve,
        "epistemic_var": epistemic_curve,
        "alpha_gria": alpha_curve,
        "trained_steps": limit,
        "online_train_bpc": online_train_bpc,
        "final_train_bpc": final_train_bpc,
    }


def _run_cyphalm_char_lm(text: str) -> dict:
    limits = cyphalm_bench_limits()
    vocab_size = int(DEFAULT_CYPHALM_CONFIG["vocab_size"])
    char2id, id2char = build_char_vocab(text, vocab_size=vocab_size - 1)
    ids = encode_text(text, char2id)
    if len(ids) < 512:
        return {"skipped": True, "reason": "corpus too short after encoding"}

    split = int(len(ids) * 0.8)
    train_ids = ids[:split]
    test_ids = ids[split:]

    model, _ = make_cyphalm()
    train_metrics = _train_with_snapshots(
        model,
        train_ids,
        test_ids,
        n_train=limits["n_train"],
        snapshot_every=limits["snapshot_every"],
        n_eval_snapshot=min(200, limits["n_eval"]),
    )

    final_bpc = eval_held_out_bpc(model, test_ids, n_eval=limits["n_eval"])
    bigram_bpc_val = bigram_baseline_bpc(train_ids, test_ids, vocab_size)
    dif_final = cyphalm_dif_metrics(model)

    prompt_len = min(limits["prompt_len"], len(test_ids) - 1)
    prompt_ids = test_ids[:prompt_len]

    context_bpc = eval_bpc_by_context_length(model, test_ids)
    routing = expert_routing_trace(model, prompt_ids, max_tokens=limits["max_generate"])
    sampling = compare_sampling_strategies(
        model, prompt_ids, id2char, max_tokens=limits["max_generate"]
    )

    with tempfile.TemporaryDirectory() as tmp:
        save_restore = eval_save_restore_fidelity(
            model,
            test_ids,
            Path(tmp) / "d04_cyphalm_ckpt",
            n_eval=min(200, limits["n_eval"]),
        )

    return {
        "model": "CyphaLM",
        "pipeline": "Izaac -> CellAI SSM -> CyphaDIF -> GRIA",
        "vocab_size": len(char2id) + 1,
        "eval_method": "held_out_20pct_suffix",
        **train_metrics,
        "final_bpc": final_bpc,
        "bigram_bpc": bigram_bpc_val,
        "cypha_dif": dif_final,
        "context_length_bpc": context_bpc,
        "expert_routing": routing,
        "save_restore": save_restore,
        "sampling_strategies": sampling,
        "generation_preview": {
            "prompt_text": decode_ids(prompt_ids, id2char)[:120],
            "greedy": sampling.get("greedy", {}).get("sample_text", ""),
            "top_p": sampling.get("top_p_0.92", {}).get("sample_text", ""),
        },
    }


def _save_figures(lm_metrics: dict, source: str) -> None:
    if not lm_metrics.get("steps"):
        return

    # Fig 1: training BPC + expert growth (existing)
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        lm_metrics["steps"],
        lm_metrics["held_out_bpc"],
        label="CyphaLM held-out",
        marker="o",
    )
    bb = lm_metrics.get("bigram_bpc")
    if bb is not None and not np.isnan(bb):
        axes[0].axhline(bb, color="gray", linestyle="--", label=f"Bigram ({bb:.2f})")
    axes[0].set_xlabel("Training tokens")
    axes[0].set_ylabel("Bits per character")
    axes[0].set_title(f"CyphaLM char LM ({source})")
    axes[0].legend()
    axes[1].plot(lm_metrics["steps"], lm_metrics["expert_count"])
    axes[1].set_xlabel("Training tokens")
    axes[1].set_ylabel("Expert count")
    axes[1].set_title("CyphaDIF expert growth")
    plt.tight_layout()
    save_figure(fig1, "fig04_char_lm_training")
    plt.close(fig1)

    # Fig 2: BPC vs context length
    ctx = lm_metrics.get("context_length_bpc") or {}
    if ctx:
        xs = sorted(int(k) for k in ctx)
        ys = [ctx[str(x)] for x in xs]
        fig2, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel("Held-out BPC")
        ax.set_title("Perplexity vs context window")
        ax.set_xscale("log", base=2)
        plt.tight_layout()
        save_figure(fig2, "fig04_context_bpc")
        plt.close(fig2)

    # Fig 3: expert routing during generation
    routing = lm_metrics.get("expert_routing") or {}
    dom = routing.get("dominant_expert_per_step") or []
    epi = routing.get("epistemic_var_per_step") or []
    if dom:
        fig3, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        axes[0].step(range(len(dom)), dom, where="mid")
        axes[0].set_ylabel("Dominant CyphaDIF expert")
        axes[0].set_title("Expert routing during greedy generation")
        if epi:
            axes[1].plot(epi)
            axes[1].set_ylabel("Epistemic variance")
        axes[1].set_xlabel("Generation step")
        plt.tight_layout()
        save_figure(fig3, "fig04_expert_routing")
        plt.close(fig3)

    # Fig 4: sampling strategy BPC comparison
    sampling = lm_metrics.get("sampling_strategies") or {}
    if sampling:
        names = list(sampling.keys())
        bpcs = [sampling[n].get("mean_bpc", float("nan")) for n in names]
        toks = [sampling[n].get("tokens_generated", 0) for n in names]
        fig4, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(names))
        ax.bar(x, bpcs, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("Mean BPC (generated tokens)")
        ax.set_title("Sampling strategy comparison")
        for i, t in enumerate(toks):
            ax.text(i, bpcs[i], str(t), ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        save_figure(fig4, "fig04_sampling_strategies")
        plt.close(fig4)


def run() -> dict:
    try:
        text, source = load_char_corpus(prefer_gutenberg=True)
        lm_metrics = _run_cyphalm_char_lm(text)
    except ImportError as exc:
        return {
            "domain": "d04_generation_language",
            "skipped": True,
            "reason": f"cypha_lm not available: {exc}",
        }

    metrics = {
        "domain": "d04_generation_language",
        "corpus_source": source,
        "char_lm": lm_metrics,
        "final_bpc": lm_metrics.get("final_bpc"),
        "bigram_bpc": lm_metrics.get("bigram_bpc"),
        "save_restore_parity_ok": (lm_metrics.get("save_restore") or {}).get("parity_ok"),
    }

    if not lm_metrics.get("skipped"):
        _save_figures(lm_metrics, source)

    save_table("d04_generation_language", metrics)
    return metrics


if __name__ == "__main__":
    import json

    result = run()
    print(json.dumps(result, indent=2, default=str)[:6000])
