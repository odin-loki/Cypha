"""Shared CyphaLM helpers for cypha_bench language-model domains (D04, D17)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cypha_bench.common.paths import is_fast, scale

_REPO = Path(__file__).resolve().parents[2]
_BENCH = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cypha_bench.config.load_cyphalm_profile import (
    load_cyphalm_profile_file,
    resolve_cyphalm_profile_path,
)

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
    "device": "auto",
    "use_spectral_pde": False,
    "use_sparse_hebbian": False,
    "use_multiscale": True,
    "n_experts": 0,
    "train_ssm": False,
    "ssm_lr": 0.001,
    "view_schedule": "same_order",
    "view_block_size": 512,
}


def load_cyphalm_config(
    overrides: dict[str, Any] | None = None,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Merge defaults ← domain profile JSON ← overrides."""
    cfg = dict(DEFAULT_CYPHALM_CONFIG)
    path = resolve_cyphalm_profile_path(profile)
    if path.exists():
        cfg.update(load_cyphalm_profile_file(path))
    if overrides:
        cfg.update(overrides)
    return cfg


def cyphalm_bench_limits() -> dict[str, int]:
    """Scale corpus / train / eval sizes for full vs CYPHA_BENCH_FAST runs."""
    if os.environ.get("CYPHA_BENCH_FULL_CORPUS", "0") == "1":
        n_train = int(os.environ.get("CYPHA_BENCH_FULL_N_TRAIN", "300000"))
        return {
            "max_corpus_chars": 10_000_000,
            "n_train": n_train,
            "n_eval": scale(8_000, 500),
            "snapshot_every": scale(10_000, 500),
            "max_generate": scale(120, 40),
            "prompt_len": scale(64, 24),
        }
    return {
        "max_corpus_chars": scale(80_000, 20_000),
        "n_train": scale(40_000, 3_000),
        "n_eval": scale(2_000, 300),
        "snapshot_every": scale(2_000, 500),
        "max_generate": scale(120, 40),
        "prompt_len": scale(64, 24),
    }


def make_cyphalm(
    config: dict[str, Any] | None = None,
    *,
    profile: str | None = None,
):
    from cypha_lm.config import CyphaLMConfig
    from cypha_lm.model.cypha_lm import CyphaLM

    merged = load_cyphalm_config(config, profile=profile)
    cfg = CyphaLMConfig(**merged)
    return CyphaLM(cfg), cfg


def _allow_synthetic_corpus() -> bool:
    return is_fast() or os.environ.get("CYPHA_BENCH_ALLOW_SYNTHETIC", "0") == "1"


def require_real_corpus(source: str, *, domain: str) -> None:
    if source != "synthetic":
        return
    if _allow_synthetic_corpus():
        return
    raise RuntimeError(
        f"{domain}: corpus fell back to synthetic text. Install real data:\n"
        "  python cypha_bench/setup/acquire_data.py\n"
        "Or set CYPHA_BENCH_FAST=1 / CYPHA_BENCH_ALLOW_SYNTHETIC=1 for dev runs."
    )


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

    if not _allow_synthetic_corpus():
        raise RuntimeError(
            "No Gutenberg or WikiText-2 corpus found under cypha_bench/data/. "
            "Run: python cypha_bench/setup/acquire_data.py"
        )
    rng = np.random.default_rng(42)
    alphabet = "abcdefghijklmnopqrstuvwxyz .,\n"
    synthetic = "".join(rng.choice(list(alphabet)) for _ in range(max_chars))
    return synthetic, "synthetic"


def _read_corpus_file(path: Path, max_chars: int | None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None:
        return text[:max_chars]
    return text


@dataclass
class LMCorpus:
    source: str
    train_ids: list[int]
    eval_ids: list[int]
    test_ids: list[int]
    char2id: dict[str, int]
    id2char: dict[int, str]
    vocab_size: int
    split: str


def prepare_lm_corpus(
    *,
    prefer_wikitext: bool = False,
    max_train_chars: int | None = None,
) -> LMCorpus:
    """
    Build train/eval/test id lists.

    WikiText-2: official train / valid / test files when present.
    Gutenberg: single book, 80/20 train/eval (test = eval tail).
    """
    from cypha_bench.common.paths import DATA_DIR

    limits = cyphalm_bench_limits()
    if max_train_chars is None:
        max_train_chars = limits["max_corpus_chars"]

    wt_dir = DATA_DIR / "wikitext2" / "wikitext-2"
    wt_train = wt_dir / "wiki.train.tokens"
    wt_valid = wt_dir / "wiki.valid.tokens"
    wt_test = wt_dir / "wiki.test.tokens"

    if prefer_wikitext and wt_train.exists() and wt_valid.exists():
        train_text = _read_corpus_file(wt_train, max_train_chars)
        valid_text = _read_corpus_file(wt_valid, limits["max_corpus_chars"] // 4)
        test_text = (
            _read_corpus_file(wt_test, limits["max_corpus_chars"] // 4)
            if wt_test.exists()
            else valid_text
        )
        combined = train_text + valid_text + test_text
        vocab_size = int(load_cyphalm_config()["vocab_size"])
        char2id, id2char = build_char_vocab(combined, vocab_size=vocab_size - 1)
        return LMCorpus(
            source="wikitext2",
            train_ids=encode_text(train_text, char2id),
            eval_ids=encode_text(valid_text, char2id),
            test_ids=encode_text(test_text, char2id),
            char2id=char2id,
            id2char=id2char,
            vocab_size=vocab_size,
            split="wikitext_official",
        )

    text, source = load_char_corpus(max_chars=max_train_chars, prefer_gutenberg=not prefer_wikitext)
    require_real_corpus(source, domain="LM corpus")
    vocab_size = int(load_cyphalm_config()["vocab_size"])
    char2id, id2char = build_char_vocab(text, vocab_size=vocab_size - 1)
    ids = encode_text(text, char2id)
    if len(ids) < 512:
        raise ValueError("corpus too short after encoding")
    split_at = int(len(ids) * 0.8)
    train_ids = ids[:split_at]
    eval_ids = ids[split_at:]
    return LMCorpus(
        source=source,
        train_ids=train_ids,
        eval_ids=eval_ids,
        test_ids=eval_ids,
        char2id=char2id,
        id2char=id2char,
        vocab_size=vocab_size,
        split="holdout_20pct",
    )


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


def _ngram_counts(train_ids: list[int], n: int, vocab_size: int) -> dict[tuple[int, ...], np.ndarray]:
    counts: dict[tuple[int, ...], np.ndarray] = {}
    for i in range(len(train_ids) - n):
        ctx = tuple(train_ids[i : i + n])
        nxt = int(train_ids[i + n])
        if any(t < 0 or t >= vocab_size for t in ctx) or not (0 <= nxt < vocab_size):
            continue
        if ctx not in counts:
            counts[ctx] = np.zeros(vocab_size, dtype=np.float64)
        counts[ctx][nxt] += 1.0
    return counts


def ngram_baseline_bpc(
    train_ids: list[int],
    test_ids: list[int],
    vocab_size: int,
    n: int = 2,
) -> float:
    """Character n-gram baseline BPC (n=2 bigram, n=3 trigram)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        counts = np.zeros(vocab_size, dtype=np.float64)
        for t in train_ids:
            if 0 <= t < vocab_size:
                counts[t] += 1.0
        probs = (counts + 1.0) / (counts.sum() + vocab_size)
        bits: list[float] = []
        for t in test_ids:
            if 0 <= t < vocab_size:
                bits.append(-np.log2(max(float(probs[t]), 1e-12)))
        return float(np.mean(bits)) if bits else float("nan")

    table = _ngram_counts(train_ids, n - 1, vocab_size)
    global_counts = np.zeros(vocab_size, dtype=np.float64)
    for t in train_ids:
        if 0 <= t < vocab_size:
            global_counts[t] += 1.0
    global_probs = (global_counts + 1.0) / (global_counts.sum() + vocab_size)

    bits: list[float] = []
    for i in range(len(test_ids) - n + 1):
        ctx = tuple(test_ids[i : i + n - 1])
        nxt = int(test_ids[i + n - 1])
        if not (0 <= nxt < vocab_size):
            continue
        if ctx in table:
            row = table[ctx]
            probs = (row + 1.0) / (row.sum() + vocab_size)
            p = max(float(probs[nxt]), 1e-12)
        else:
            p = max(float(global_probs[nxt]), 1e-12)
        bits.append(-np.log2(p))
    return float(np.mean(bits)) if bits else float("nan")


def bigram_baseline_bpc(
    train_ids: list[int],
    test_ids: list[int],
    vocab_size: int,
) -> float:
    return ngram_baseline_bpc(train_ids, test_ids, vocab_size, n=2)


def trigram_baseline_bpc(
    train_ids: list[int],
    test_ids: list[int],
    vocab_size: int,
) -> float:
    return ngram_baseline_bpc(train_ids, test_ids, vocab_size, n=3)


def fourgram_baseline_bpc(
    train_ids: list[int],
    test_ids: list[int],
    vocab_size: int,
) -> float:
    return ngram_baseline_bpc(train_ids, test_ids, vocab_size, n=4)


def fivegram_baseline_bpc(
    train_ids: list[int],
    test_ids: list[int],
    vocab_size: int,
) -> float:
    return ngram_baseline_bpc(train_ids, test_ids, vocab_size, n=5)


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
    from cypha_lm.array_backend import asnumpy

    return {
        "ssm": model.ssm.get_state(),
        "dif": model.dif.get_state(),
        "gria": model.gria.get_state(),
        "token_counts": np.asarray(model._token_counts, dtype=np.float64).copy(),
        "proj_ssm": asnumpy(model._proj_ssm),
        "proj_dif": asnumpy(model._proj_dif),
        "proj_embed": asnumpy(model._proj_embed),
        "proj_ngram": asnumpy(model._proj_ngram),
    }


def restore_model_state(model, snap: dict[str, Any]) -> None:
    xp = model._backend.xp
    model.ssm.set_state(snap["ssm"])
    model.dif.set_state(snap["dif"])
    model.gria.set_state(snap["gria"])
    model._token_counts = np.asarray(snap["token_counts"], dtype=np.float64).copy()
    model._proj_ssm = xp.asarray(snap["proj_ssm"], dtype=xp.float64)
    model._proj_dif = xp.asarray(snap["proj_dif"], dtype=xp.float64)
    model._proj_embed = xp.asarray(snap["proj_embed"], dtype=xp.float64)
    if "proj_ngram" in snap:
        model._proj_ngram = xp.asarray(snap["proj_ngram"], dtype=xp.float64)


def eval_held_out_bpc_preserve_training(model, test_ids: list[int], n_eval: int) -> float:
    """Held-out BPC without disturbing weights or training-time SSM context."""
    snap = snapshot_model_state(model)
    bpc = eval_held_out_bpc(model, test_ids, n_eval=n_eval)
    restore_model_state(model, snap)
    model.reset_context()
    return bpc


def train_with_learning_curve(
    model,
    train_ids: list[int],
    eval_ids: list[int],
    *,
    n_train: int,
    snapshot_every: int,
    n_eval_snapshot: int,
) -> dict[str, Any]:
    """Online train with periodic held-out BPC snapshots (D04/D17)."""
    steps: list[int] = []
    bpc_curve: list[float] = []
    expert_curve: list[int] = []
    train_losses: list[float] = []

    model.reset_context()
    limit = min(n_train, len(train_ids) - 1)
    log_every = max(5000, snapshot_every)
    for t in range(limit):
        metrics = model.train_step(int(train_ids[t]), int(train_ids[t + 1]))
        train_losses.append(float(metrics["loss"]))
        trained = t + 1
        if trained % log_every == 0:
            print(f"[train] {trained}/{limit} loss={train_losses[-1]:.4f}", flush=True)
        if trained % snapshot_every == 0 or trained == limit:
            snap_bpc = eval_held_out_bpc_preserve_training(
                model, eval_ids, n_eval=n_eval_snapshot
            )
            steps.append(trained)
            bpc_curve.append(snap_bpc)
            expert_curve.append(int(metrics.get("active_experts", 0)))

    tail = max(1, len(train_losses) // 10)
    online_train_bpc = float(np.mean(train_losses) / np.log(2)) if train_losses else float("nan")
    final_train_bpc = float(np.mean(train_losses[-tail:]) / np.log(2)) if train_losses else float("nan")

    return {
        "steps": steps,
        "held_out_bpc": bpc_curve,
        "expert_count": expert_curve,
        "trained_steps": limit,
        "online_train_bpc": online_train_bpc,
        "final_train_bpc": final_train_bpc,
    }


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
    max_diff = 0.0
    for tid, lp_before in zip(probe_ids, before_lps, strict=False):
        pred = loaded.predict_next(int(tid))
        lp_after = np.asarray(pred["log_probs"], dtype=np.float64)
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
