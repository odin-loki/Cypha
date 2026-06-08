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

import json
import os
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

from cypha_bench.adapters.char_lstm_baseline import char_lstm_baseline_bpc
from cypha_bench.adapters.cyphalm_ablations import run_lm_ablations
from cypha_bench.adapters.cyphalm_bench import (
    bigram_baseline_bpc,
    compare_sampling_strategies,
    cyphalm_bench_limits,
    cyphalm_dif_metrics,
    decode_ids,
    eval_bpc_by_context_length,
    eval_held_out_bpc,
    eval_save_restore_fidelity,
    expert_routing_trace,
    fivegram_baseline_bpc,
    fourgram_baseline_bpc,
    load_cyphalm_config,
    make_cyphalm,
    prepare_lm_corpus,
    require_real_corpus,
    train_with_learning_curve,
    trigram_baseline_bpc,
)
from cypha_bench.common.metrics import save_figure, save_table
from cypha_bench.common.paths import is_fast


def _lm_baseline_metrics(train_slice, test_ids, vocab_size, limits) -> dict:
    metrics = {
        "bigram_bpc": bigram_baseline_bpc(train_slice, test_ids, vocab_size),
        "trigram_bpc": trigram_baseline_bpc(train_slice, test_ids, vocab_size),
    }
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


def _run_cyphalm_char_lm(corpus) -> dict:
    limits = cyphalm_bench_limits()
    train_ids = corpus.train_ids
    test_ids = corpus.eval_ids

    model, _ = make_cyphalm(profile="d04")
    train_metrics = train_with_learning_curve(
        model,
        train_ids,
        test_ids,
        n_train=limits["n_train"],
        snapshot_every=limits["snapshot_every"],
        n_eval_snapshot=min(200, limits["n_eval"]),
    )

    n_train = train_metrics["trained_steps"]
    train_slice = train_ids[:n_train]
    final_bpc = eval_held_out_bpc(model, test_ids, n_eval=limits["n_eval"])
    baselines = _lm_baseline_metrics(train_slice, test_ids, corpus.vocab_size, limits)
    bigram_bpc_val = baselines["bigram_bpc"]
    trigram_bpc_val = baselines["trigram_bpc"]
    dif_final = cyphalm_dif_metrics(model)

    ablation_modes = (
        ["full", "gria_ngram"]
        if is_fast()
        else ["full", "gria_ngram", "ssm_only", "ablation_no_dif", "ablation_no_ssm"]
    )
    if os.environ.get("CYPHA_BENCH_SKIP_LM_ABLATIONS", "0") == "1":
        sweep_path = _REPO / "cypha_bench" / "config" / "cyphalm_hybrid_lstm_d04_300k.json"
        if sweep_path.exists():
            sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
            ablation_bpc = {
                row["cell_id"]: row["held_out_bpc"]
                for row in sweep.get("cells", [])
                if row.get("cell_id") in {"gria_ngram", "hybrid_gria_lstm"}
            }
            ablation_bpc.setdefault("hybrid_gria_lstm", final_bpc)
            ablations = {
                "stub_from_sweep": sweep_path.name,
                "n_train": sweep.get("n_train"),
                "bigram_bpc": sweep.get("bigram_bpc"),
                "char_lstm_bpc": sweep.get("char_lstm_baseline_bpc"),
                "ablation_bpc": ablation_bpc,
            }
        else:
            ablations = {"skipped": True, "reason": "CYPHA_BENCH_SKIP_LM_ABLATIONS without sweep artifact"}
    else:
        ablations = run_lm_ablations(corpus, limits, ablation_modes, profile="d04")

    prompt_len = min(limits["prompt_len"], len(test_ids) - 1)
    prompt_ids = test_ids[:prompt_len]

    context_bpc = eval_bpc_by_context_length(model, test_ids)
    routing = expert_routing_trace(model, prompt_ids, max_tokens=limits["max_generate"])
    sampling = compare_sampling_strategies(
        model, prompt_ids, corpus.id2char, max_tokens=limits["max_generate"]
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
        "vocab_size": len(corpus.char2id) + 1,
        "eval_method": corpus.split,
        "device": model.device,
        "profile": load_cyphalm_config(profile="d04"),
        **train_metrics,
        "final_bpc": final_bpc,
        "bigram_bpc": bigram_bpc_val,
        "trigram_bpc": trigram_bpc_val,
        **baselines,
        "delta_vs_bigram": final_bpc - bigram_bpc_val,
        "delta_vs_trigram": final_bpc - trigram_bpc_val,
        "ablations": ablations,
        "cypha_dif": dif_final,
        "context_length_bpc": context_bpc,
        "expert_routing": routing,
        "save_restore": save_restore,
        "sampling_strategies": sampling,
        "generation_preview": {
            "prompt_text": decode_ids(prompt_ids, corpus.id2char)[:120],
            "greedy": sampling.get("greedy", {}).get("sample_text", ""),
            "top_p": sampling.get("top_p_0.92", {}).get("sample_text", ""),
        },
    }


def _save_figures(lm_metrics: dict, source: str) -> None:
    if not lm_metrics.get("steps"):
        return

    fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        lm_metrics["steps"],
        lm_metrics["held_out_bpc"],
        label="CyphaLM held-out",
        marker="o",
    )
    bb = lm_metrics.get("bigram_bpc")
    tb = lm_metrics.get("trigram_bpc")
    if bb is not None and not np.isnan(bb):
        axes[0].axhline(bb, color="gray", linestyle="--", label=f"Bigram ({bb:.2f})")
    if tb is not None and not np.isnan(tb):
        axes[0].axhline(tb, color="orange", linestyle=":", label=f"Trigram ({tb:.2f})")
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
        limits = cyphalm_bench_limits()
        corpus = prepare_lm_corpus(prefer_wikitext=False, max_train_chars=limits["max_corpus_chars"])
        require_real_corpus(corpus.source, domain="D04")
        lm_metrics = _run_cyphalm_char_lm(corpus)
        source = corpus.source
    except ImportError as exc:
        return {
            "domain": "d04_generation_language",
            "skipped": True,
            "reason": f"cypha_lm not available: {exc}",
        }
    except RuntimeError as exc:
        return {
            "domain": "d04_generation_language",
            "skipped": True,
            "reason": str(exc),
        }

    metrics = {
        "domain": "d04_generation_language",
        "corpus_source": source,
        "char_lm": lm_metrics,
        "final_bpc": lm_metrics.get("final_bpc"),
        "bigram_bpc": lm_metrics.get("bigram_bpc"),
        "trigram_bpc": lm_metrics.get("trigram_bpc"),
        "fourgram_bpc": lm_metrics.get("fourgram_bpc"),
        "fivegram_bpc": lm_metrics.get("fivegram_bpc"),
        "char_lstm_bpc": lm_metrics.get("char_lstm_bpc"),
        "ablations": lm_metrics.get("ablations"),
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
