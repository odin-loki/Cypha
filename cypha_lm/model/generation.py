"""Autoregressive decoding strategies for CyphaLM."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import numpy as np

from cypha_lm.model.cypha_lm import CyphaLM

DecodeStrategy = Literal["greedy", "temperature", "top_k", "top_p", "uncertainty_gated"]


def _consume_prompt(model: CyphaLM, prompt_ids: list[int]) -> None:
    model.reset_context()
    for tid in prompt_ids[:-1]:
        model._forward_context(int(tid))


def _sample_token(
    model: CyphaLM,
    log_probs: np.ndarray,
    *,
    strategy: DecodeStrategy,
    temperature: float,
    top_k: int,
    top_p: float,
    pred: dict[str, Any],
) -> int:
    temp = max(float(temperature), 1e-6)
    if strategy == "greedy" or temperature <= 1e-6:
        return int(pred["top_k_tokens"][0])

    rng = model._rng
    lp = np.asarray(log_probs, dtype=np.float64)

    if strategy == "top_k":
        kk = min(int(top_k), lp.size)
        idx = np.argpartition(lp, -kk)[-kk:]
        sub = lp[idx]
        probs = np.exp((sub - np.max(sub)) / temp)
        probs /= probs.sum() + 1e-12
        return int(rng.choice(idx, p=probs))

    if strategy == "top_p":
        order = np.argsort(lp)[::-1]
        probs = np.exp(lp[order] / temp)
        probs /= probs.sum() + 1e-12
        cum = np.cumsum(probs)
        cutoff = int(np.searchsorted(cum, float(top_p), side="right"))
        cutoff = max(cutoff, 1)
        nucleus = order[:cutoff]
        n_probs = probs[:cutoff]
        n_probs /= n_probs.sum() + 1e-12
        return int(rng.choice(nucleus, p=n_probs))

    # temperature (full vocab) — default for uncertainty_gated after gate check
    probs = np.exp(lp / temp)
    probs /= probs.sum() + 1e-12
    return int(rng.choice(model.config.vocab_size, p=probs))


def _step_record(pred: dict[str, Any], token_id: int, loss: float) -> dict[str, Any]:
    routing = pred.get("routing_probs")
    return {
        "token_id": int(token_id),
        "loss": float(loss),
        "epistemic_var": float(pred["epistemic_var"]),
        "aleatoric_var": float(pred["aleatoric_var"]),
        "active_experts": int(pred.get("active_experts", 0)),
        "dominant_expert": int(pred.get("dominant_expert", 0)),
        "routing_probs": list(routing) if routing is not None else [],
    }


def autoregressive_decode(
    model: CyphaLM,
    prompt_ids: list[int],
    max_tokens: int,
    *,
    strategy: DecodeStrategy = "temperature",
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    epistemic_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Unified decode with per-step CyphaDIF routing trace.

    Returns generated token ids plus ``per_step`` metrics (epistemic variance,
    dominant expert index, full routing probability vector).
    """
    _consume_prompt(model, prompt_ids)
    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    halted = False
    last = int(prompt_ids[-1]) if prompt_ids else 0

    for _ in range(max_tokens):
        pred = model.predict_next(last)
        ep = float(pred["epistemic_var"])
        if strategy == "uncertainty_gated" and epistemic_threshold is not None:
            if ep > float(epistemic_threshold):
                halted = True
                steps.append(
                    {
                        "token_id": None,
                        "loss": float("nan"),
                        "epistemic_var": ep,
                        "aleatoric_var": float(pred["aleatoric_var"]),
                        "active_experts": int(pred.get("active_experts", 0)),
                        "dominant_expert": int(pred.get("dominant_expert", 0)),
                        "routing_probs": list(pred.get("routing_probs") or []),
                        "halted": True,
                    }
                )
                break

        log_probs = np.asarray(pred["log_probs"], dtype=np.float64)
        tid = _sample_token(
            model,
            log_probs,
            strategy=strategy if strategy != "uncertainty_gated" else "temperature",
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pred=pred,
        )
        loss = float(-log_probs[tid])
        generated.append(tid)
        steps.append(_step_record(pred, tid, loss))
        last = tid

    return {
        "generated_ids": generated,
        "per_step": steps,
        "halted_on_uncertainty": halted,
        "strategy": strategy,
    }


def stream_generate(
    model: CyphaLM,
    prompt_ids: list[int],
    max_tokens: int,
    *,
    strategy: DecodeStrategy = "temperature",
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    epistemic_threshold: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one JSON-serializable chunk per generated token (for SSE / NDJSON streaming)."""
    _consume_prompt(model, prompt_ids)
    last = int(prompt_ids[-1]) if prompt_ids else 0
    index = 0

    for _ in range(max_tokens):
        pred = model.predict_next(last)
        ep = float(pred["epistemic_var"])
        if strategy == "uncertainty_gated" and epistemic_threshold is not None and ep > float(
            epistemic_threshold
        ):
            yield {
                "index": index,
                "done": True,
                "halted_on_uncertainty": True,
                "epistemic_var": ep,
                "active_experts": int(pred.get("active_experts", 0)),
                "dominant_expert": int(pred.get("dominant_expert", 0)),
            }
            return

        log_probs = np.asarray(pred["log_probs"], dtype=np.float64)
        tid = _sample_token(
            model,
            log_probs,
            strategy=strategy if strategy != "uncertainty_gated" else "temperature",
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pred=pred,
        )
        chunk = _step_record(pred, tid, float(-log_probs[tid]))
        chunk["index"] = index
        chunk["done"] = False
        yield chunk
        last = tid
        index += 1

    yield {"index": index, "done": True, "halted_on_uncertainty": False}


def greedy_decode(model: CyphaLM, prompt_ids: list[int], max_tokens: int) -> list[int]:
    return autoregressive_decode(
        model, prompt_ids, max_tokens, strategy="greedy", temperature=0.0
    )["generated_ids"]


def temperature_sample(
    model: CyphaLM,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
) -> list[int]:
    return autoregressive_decode(
        model, prompt_ids, max_tokens, strategy="temperature", temperature=temperature
    )["generated_ids"]


def top_k_sample(
    model: CyphaLM,
    prompt_ids: list[int],
    max_tokens: int,
    k: int,
    temperature: float,
) -> list[int]:
    return autoregressive_decode(
        model,
        prompt_ids,
        max_tokens,
        strategy="top_k",
        temperature=temperature,
        top_k=k,
    )["generated_ids"]


def top_p_sample(
    model: CyphaLM,
    prompt_ids: list[int],
    max_tokens: int,
    p: float,
    temperature: float,
) -> list[int]:
    """Nucleus (top-p) sampling."""
    return autoregressive_decode(
        model,
        prompt_ids,
        max_tokens,
        strategy="top_p",
        temperature=temperature,
        top_p=p,
    )["generated_ids"]


def uncertainty_gated_sample(
    model: CyphaLM,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    epistemic_threshold: float,
) -> dict:
    """Sample until epistemic variance exceeds threshold (anti-hallucination pause)."""
    out = autoregressive_decode(
        model,
        prompt_ids,
        max_tokens,
        strategy="uncertainty_gated",
        temperature=temperature,
        epistemic_threshold=epistemic_threshold,
    )
    epistemic_trace = [s["epistemic_var"] for s in out["per_step"]]
    return {
        "generated_ids": out["generated_ids"],
        "epistemic_trace": epistemic_trace,
        "halted_on_uncertainty": out["halted_on_uncertainty"],
        "per_step": out["per_step"],
    }
