#!/usr/bin/env python3
"""CyphaLM CPU vs CUDA benchmark with optional component breakdown."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from cypha_lm.array_backend import asnumpy, cuda_available
from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM


def _bench_train(model: CyphaLM, n_steps: int, warmup: int) -> float:
    ids = (list(range(1, model.config.vocab_size)) * ((n_steps + warmup) // model.config.vocab_size + 2))
    model.reset_context()
    for i in range(warmup):
        model.train_step(ids[i], ids[i + 1])
    if model.device == "cuda":
        model._backend.sync()
    t0 = time.perf_counter()
    for i in range(warmup, warmup + n_steps):
        model.train_step(ids[i], ids[i + 1])
    if model.device == "cuda":
        model._backend.sync()
    return time.perf_counter() - t0


def _bench_predict(model: CyphaLM, n_steps: int, warmup: int) -> float:
    ids = list(range(1, model.config.vocab_size)) * 10
    for i in range(warmup):
        model.predict_next(ids[i % len(ids)])
    if model.device == "cuda":
        model._backend.sync()
    t0 = time.perf_counter()
    for i in range(n_steps):
        model.predict_next(ids[i % len(ids)])
    if model.device == "cuda":
        model._backend.sync()
    return time.perf_counter() - t0


def _bench_components(model: CyphaLM, n_steps: int) -> dict[str, float]:
    """Rough per-stage timing (CUDA sync after each stage on GPU)."""
    xp = model._backend.xp
    ids = list(range(1, model.config.vocab_size)) * 10
    acc = {"embed": 0.0, "ssm": 0.0, "proj": 0.0, "dif": 0.0, "gria": 0.0, "gria_train": 0.0}
    model.reset_context()
    for i in range(n_steps):
        tid, nxt = int(ids[i % len(ids)]), int(ids[(i + 1) % len(ids)])
        t0 = time.perf_counter()
        e = model.embed.embed(tid)
        if model.device == "cuda":
            model._backend.sync()
        acc["embed"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        ctx = model.ssm.step(e)
        if model.device == "cuda":
            model._backend.sync()
        acc["ssm"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        field_x = model._project_context(ctx)
        if model.device == "cuda":
            model._backend.sync()
        acc["proj"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        dif_out = model.dif.predict(asnumpy(field_x))
        acc["dif"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        v = model._gria_input(field_x, dif_out)
        log_probs = model.gria.forward(v)
        if model.device == "cuda":
            model._backend.sync()
        acc["gria"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        gw, ga, gb = model.gria.cross_entropy_gradients(v, nxt)
        model.gria.update_weights(gw, model.config.gria_lr)
        model.gria.update_alpha(ga, model.config.gria_lr)
        model.gria.update_bias(gb, model.config.gria_lr)
        if model.device == "cuda":
            model._backend.sync()
        acc["gria_train"] += time.perf_counter() - t0

        if model.config.online:
            target_vec = model._proj_embed @ model.embed.embed(nxt)
            model.dif.train_step(asnumpy(field_x), asnumpy(target_vec))

    return {k: v / n_steps * 1000 for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="CyphaLM CPU vs CUDA benchmark")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--breakdown", action="store_true")
    args = ap.parse_args()

    base = dict(
        vocab_size=128,
        d_embed=64,
        d_state=128,
        field_dim=160,
        ssm_layers=2,
        max_experts=128,
        gria_lr=0.06,
        online=True,
    )

    print(f"CyphaLM benchmark — {args.steps} train steps (warmup {args.warmup})")
    print(f"Config: {base}\n")

    results: dict[str, dict[str, float]] = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not cuda_available():
            print("CUDA unavailable — skipping GPU runs")
            break
        cfg = CyphaLMConfig(**base, device=device)
        model = CyphaLM(cfg)
        train_s = _bench_train(model, args.steps, args.warmup)
        pred_s = _bench_predict(model, min(args.steps, 1000), min(args.warmup, 100))
        results[device] = {
            "train_ms_per_step": train_s / args.steps * 1000,
            "train_tok_per_s": args.steps / train_s,
            "predict_ms_per_step": pred_s / min(args.steps, 1000) * 1000,
        }
        print(f"[{device}] train: {results[device]['train_ms_per_step']:.3f} ms/step "
              f"({results[device]['train_tok_per_s']:.1f} tok/s)")
        print(f"[{device}] predict: {results[device]['predict_ms_per_step']:.3f} ms/step")
        if args.breakdown and device == "cuda":
            bd = _bench_components(model, min(500, args.steps))
            print("  breakdown (ms/step):", ", ".join(f"{k}={v:.3f}" for k, v in bd.items()))

    if "cpu" in results and "cuda" in results:
        ratio = results["cpu"]["train_ms_per_step"] / results["cuda"]["train_ms_per_step"]
        print(f"\nGPU train speedup vs CPU: {ratio:.2f}x")
        if ratio < 1.0:
            print("  (GPU slower — typical for small sequential ops + CPU DIF sync)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
