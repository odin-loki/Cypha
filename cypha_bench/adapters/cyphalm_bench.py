"""Shared CyphaLM helpers for cypha_bench language-model domains (D04, D17)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from cypha_bench.common.paths import scale

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_CYPHALM_CONFIG: dict[str, Any] = {
    "vocab_size": 128,
    "d_embed": 64,
    "d_state": 128,
    "tau_fast": 1.0,
    "tau_slow": 20.0,
    "ssm_layers": 2,
    "field_dim": 160,
    "max_experts": 128,
    "alpha_init": 0.5,
    "context_length": 256,
    "seed": 42,
    "gria_lr": 0.06,
    "online": True,
}


def cyphalm_bench_limits() -> dict[str, int]:
    """Scale corpus / train / eval sizes for full vs CYPHA_BENCH_FAST runs."""
    return {
        "max_corpus_chars": scale(80_000, 20_000),
        "n_train": scale(40_000, 3_000),
        "n_eval": scale(2_000, 300),
        "snapshot_every": scale(2_000, 500),
        "max_generate": scale(120, 40),
        "prompt_len": scale(64, 24),
    }


def make_cyphalm(config: dict[str, Any] | None = None):
    from cypha_lm.config import CyphaLMConfig
    from cypha_lm.model.cypha_lm import CyphaLM

    cfg = CyphaLMConfig(**(config or DEFAULT_CYPHALM_CONFIG))
    return CyphaLM(cfg), cfg


def load_char_corpus(
    max_chars: int | None = None,
    *,
    prefer_gutenberg: bool = True,
) -> tuple[str, str]:
    """Load text for char-LM benchmarks. Returns (text, source_name)."""
    from cypha_bench.common.paths import DATA_DIR

    if max_chars is None:
        max_chars = cyphalm_bench_limits()["max_corpus_chars"]

    if prefer_gutenberg:
        for name in ("moby_dick.txt", "alice.txt", "sherlock_holmes.txt"):
            path = DATA_DIR / "gutenberg" / name
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
                return text, name

    wt = DATA_DIR / "wikitext2" / "wikitext-2" / "wiki.train.tokens"
    if wt.exists():
        return wt.read_text(encoding="utf-8")[:max_chars], "wikitext2"

    rng = np.random.default_rng(42)
    alphabet = "abcdefghijklmnopqrstuvwxyz .,\n"
    synthetic = "".join(rng.choice(list(alphabet)) for _ in range(max_chars))
    return synthetic, "synthetic"


def build_char_vocab(text: str, vocab_size: int = 128) -> tuple[dict[str, int], dict[int, str]]:
    """Map chars to ids (1..N); 0 reserved for unknown."""
    chars = sorted(set(text))[: vocab_size - 1]
    char2id = {c: i + 1 for i, c in enumerate(chars)}
    id2char = {i + 1: c for i, c in enumerate(chars)}
    id2char[0] = "?"
    return char2id, id2char


def encode_text(text: str, char2id: dict[str, int]) -> list[int]:
    unk = 0
    return [char2id.get(c, unk) for c in text]


def decode_ids(ids: list[int], id2char: dict[int, str]) -> str:
    return "".join(id2char.get(int(i), "?") for i in ids)


def bigram_baseline_bpc(
    train_ids: list[int],
    test_ids: list[int],
    vocab_size: int,
) -> float:
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for a, b in zip(train_ids, train_ids[1:]):
        if 0 <= a < vocab_size and 0 <= b < vocab_size:
            counts[a, b] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True) + 1e-12
    probs = counts / row_sums
    bits: list[float] = []
    for a, b in zip(test_ids, test_ids[1:]):
        if 0 <= a < vocab_size and 0 <= b < vocab_size:
            p = max(float(probs[a, b]), 1e-12)
            bits.append(-np.log2(p))
    return float(np.mean(bits)) if bits else float("nan")


def eval_held_out_bpc(model, test_ids: list[int], n_eval: int | None = None) -> float:
    """Sequential held-out BPC: predict token t, score token t+1."""
    n = min(n_eval or len(test_ids) - 1, len(test_ids) - 1)
    if n <= 0:
        return float("nan")
    model.reset_context()
    losses: list[float] = []
    for i in range(n):
        pred = model.predict_next(int(test_ids[i]))
        nxt = int(test_ids[i + 1])
        lp = pred["log_probs"]
        losses.append(float(-lp[nxt]))
    return float(np.mean(losses) / np.log(2))


def snapshot_model_state(model) -> dict[str, Any]:
    return {
        "ssm": model.ssm.get_state(),
        "dif": model.dif.get_state(),
        "gria": model.gria.get_state(),
        "token_counts": np.asarray(model._token_counts, dtype=np.float64).copy(),
        "proj_ssm": np.asarray(model._proj_ssm, dtype=np.float64).copy(),
        "proj_dif": np.asarray(model._proj_dif, dtype=np.float64).copy(),
        "proj_embed": np.asarray(model._proj_embed, dtype=np.float64).copy(),
    }


def restore_model_state(model, snap: dict[str, Any]) -> None:
    model.ssm.set_state(snap["ssm"])
    model.dif.set_state(snap["dif"])
    model.gria.set_state(snap["gria"])
    model._token_counts = np.asarray(snap["token_counts"], dtype=np.float64).copy()
    model._proj_ssm = np.asarray(snap["proj_ssm"], dtype=np.float64).copy()
    model._proj_dif = np.asarray(snap["proj_dif"], dtype=np.float64).copy()
    model._proj_embed = np.asarray(snap["proj_embed"], dtype=np.float64).copy()


def eval_held_out_bpc_preserve_training(model, test_ids: list[int], n_eval: int) -> float:
    """Held-out BPC without disturbing weights or training-time SSM context."""
    snap = snapshot_model_state(model)
    bpc = eval_held_out_bpc(model, test_ids, n_eval=n_eval)
    restore_model_state(model, snap)
    model.reset_context()
    return bpc


def cyphalm_dif_metrics(model) -> dict[str, Any]:
    """CyphaDIF integration stats exposed through CyphaLM."""
    profile = model.compression_profile()
    return {
        "n_experts": int(profile.get("n_experts", 0)),
        "mean_alpha": float(profile.get("mean_alpha", float("nan"))),
        "mean_epistemic_var": float(profile.get("mean_epistemic_var", float("nan"))),
        "mean_aleatoric_var": float(profile.get("mean_aleatoric_var", float("nan"))),
        "lossless_fraction": float(profile.get("lossless_fraction", float("nan"))),
    }


def eval_bpc_by_context_length(
    model,
    test_ids: list[int],
    context_lengths: tuple[int, ...] | None = None,
    *,
    n_positions: int | None = None,
) -> dict[str, float]:
    """Mean held-out BPC at each context window length (SSM warm-up replay)."""
    if context_lengths is None:
        context_lengths = (8, 16, 32, 64, 128, 256)
    n_pos = n_positions or scale(80, 30)
    out: dict[str, float] = {}
    for ctx in context_lengths:
        if ctx >= len(test_ids) - 2:
            continue
        losses: list[float] = []
        end = min(ctx + n_pos, len(test_ids) - 1)
        for i in range(ctx, end):
            model.reset_context()
            for t in test_ids[i - ctx : i]:
                model.predict_next(int(t))
            pred = model.predict_next(int(test_ids[i]))
            nxt = int(test_ids[i + 1])
            losses.append(float(-pred["log_probs"][nxt]))
        out[str(ctx)] = float(np.mean(losses) / np.log(2)) if losses else float("nan")
    return out


def eval_save_restore_fidelity(
    model,
    test_ids: list[int],
    checkpoint_path: str | Path,
    *,
    n_eval: int = 200,
) -> dict[str, Any]:
    """Round-trip CyphaLM save/load and compare held-out BPC + log-prob parity."""
    from cypha_lm.model.cypha_lm import CyphaLM

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = str(path.with_suffix(""))

    n = min(n_eval, len(test_ids) - 1)
    probe_ids = test_ids[:n]
    model.reset_context()
    before_lps: list[np.ndarray] = []
    for tid in probe_ids:
        pred = model.predict_next(int(tid))
        before_lps.append(np.asarray(pred["log_probs"], dtype=np.float64))

    bpc_before = eval_held_out_bpc(model, test_ids, n_eval=n)

    model.save(base)
    loaded = CyphaLM.load(base)

    loaded.reset_context()
    after_lps: list[np.ndarray] = []
    max_diff = 0.0
    for tid, lp_before in zip(probe_ids, before_lps):
        pred = loaded.predict_next(int(tid))
        lp_after = np.asarray(pred["log_probs"], dtype=np.float64)
        after_lps.append(lp_after)
        max_diff = max(max_diff, float(np.max(np.abs(lp_before - lp_after))))

    bpc_after = eval_held_out_bpc(loaded, test_ids, n_eval=n)
    return {
        "bpc_before": bpc_before,
        "bpc_after": bpc_after,
        "bpc_delta": abs(bpc_before - bpc_after),
        "logprob_max_abs_diff": max_diff,
        "parity_ok": max_diff < 1e-9 and abs(bpc_before - bpc_after) < 1e-6,
        "checkpoint_base": base,
    }


def compare_sampling_strategies(
    model,
    prompt_ids: list[int],
    id2char: dict[int, str],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """Benchmark greedy, temperature, top-k, top-p, and uncertainty-gated decoding."""
    from cypha_lm.model.generation import autoregressive_decode

    prompt = [int(t) for t in prompt_ids]
    strategies = {
        "greedy": {"strategy": "greedy", "temperature": 0.0},
        "temperature_0.9": {"strategy": "temperature", "temperature": 0.9},
        "top_k_20": {"strategy": "top_k", "temperature": 0.9, "top_k": 20},
        "top_p_0.92": {"strategy": "top_p", "temperature": 0.9, "top_p": 0.92},
        "uncertainty_gated": {
            "strategy": "uncertainty_gated",
            "temperature": 0.9,
            "epistemic_threshold": 0.5,
        },
    }
    results: dict[str, Any] = {}
    for name, kwargs in strategies.items():
        out = autoregressive_decode(model, prompt, max_tokens, **kwargs)  # type: ignore[arg-type]
        steps = out["per_step"]
        mean_bpc = float("nan")
        losses = [s["loss"] for s in steps if s.get("token_id") is not None]
        if losses:
            mean_bpc = float(np.mean(losses) / np.log(2))
        results[name] = {
            "tokens_generated": len(out["generated_ids"]),
            "mean_bpc": mean_bpc,
            "halted_on_uncertainty": bool(out.get("halted_on_uncertainty", False)),
            "sample_text": decode_ids(out["generated_ids"], id2char)[:160],
        }
    return results


def expert_routing_trace(
    model,
    prompt_ids: list[int],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """CyphaDIF expert routing during greedy generation."""
    from cypha_lm.model.generation import autoregressive_decode

    out = autoregressive_decode(
        model,
        [int(t) for t in prompt_ids],
        max_tokens,
        strategy="greedy",
        temperature=0.0,
    )
    steps = out["per_step"]
    dominant = [int(s.get("dominant_expert", 0)) for s in steps if s.get("token_id") is not None]
    active = [int(s.get("active_experts", 0)) for s in steps if s.get("token_id") is not None]
    epi = [float(s["epistemic_var"]) for s in steps if s.get("token_id") is not None]
    return {
        "dominant_expert_per_step": dominant,
        "active_experts_per_step": active,
        "epistemic_var_per_step": epi,
        "unique_experts_used": sorted(set(dominant)),
        "mean_active_experts": float(np.mean(active)) if active else float("nan"),
    }
