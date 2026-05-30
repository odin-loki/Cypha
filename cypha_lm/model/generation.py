"""Autoregressive decoding strategies for CyphaLM."""

from __future__ import annotations

import numpy as np

from cypha_lm.model.cypha_lm import CyphaLM


def _consume_prompt(model: CyphaLM, prompt_ids: list[int]) -> None:
    model.reset_context()
    for tid in prompt_ids[:-1]:
        model._forward_context(int(tid))


def greedy_decode(model: CyphaLM, prompt_ids: list[int], max_tokens: int) -> list[int]:
    _consume_prompt(model, prompt_ids)
    out: list[int] = []
    last = int(prompt_ids[-1]) if prompt_ids else 0
    for _ in range(max_tokens):
        pred = model.predict_next(last)
        tid = int(pred["top_k_tokens"][0])
        out.append(tid)
        last = tid
    return out


def temperature_sample(
    model: CyphaLM,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
) -> list[int]:
    _consume_prompt(model, prompt_ids)
    rng = model._rng
    temp = max(float(temperature), 1e-6)
    out: list[int] = []
    last = int(prompt_ids[-1]) if prompt_ids else 0
    for _ in range(max_tokens):
        pred = model.predict_next(last)
        log_probs = np.asarray(pred["log_probs"], dtype=np.float64)
        probs = np.exp(log_probs / temp)
        probs /= probs.sum() + 1e-12
        tid = int(rng.choice(model.config.vocab_size, p=probs))
        out.append(tid)
        last = tid
    return out


def top_k_sample(
    model: CyphaLM,
    prompt_ids: list[int],
    max_tokens: int,
    k: int,
    temperature: float,
) -> list[int]:
    _consume_prompt(model, prompt_ids)
    rng = model._rng
    temp = max(float(temperature), 1e-6)
    out: list[int] = []
    last = int(prompt_ids[-1]) if prompt_ids else 0
    for _ in range(max_tokens):
        pred = model.predict_next(last)
        log_probs = np.asarray(pred["log_probs"], dtype=np.float64)
        kk = min(int(k), log_probs.size)
        idx = np.argpartition(log_probs, -kk)[-kk:]
        sub = log_probs[idx]
        probs = np.exp((sub - np.max(sub)) / temp)
        probs /= probs.sum() + 1e-12
        choice = int(rng.choice(idx, p=probs))
        out.append(choice)
        last = choice
    return out


def uncertainty_gated_sample(
    model: CyphaLM,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    epistemic_threshold: float,
) -> dict:
    """
    Sample until epistemic variance exceeds threshold (anti-hallucination pause).
    """
    _consume_prompt(model, prompt_ids)
    rng = model._rng
    temp = max(float(temperature), 1e-6)
    generated: list[int] = []
    epistemic_trace: list[float] = []
    halted = False
    last = int(prompt_ids[-1]) if prompt_ids else 0
    for _ in range(max_tokens):
        pred = model.predict_next(last)
        ep = float(pred["epistemic_var"])
        epistemic_trace.append(ep)
        if ep > float(epistemic_threshold):
            halted = True
            break
        log_probs = np.asarray(pred["log_probs"], dtype=np.float64)
        probs = np.exp(log_probs / temp)
        probs /= probs.sum() + 1e-12
        tid = int(rng.choice(model.config.vocab_size, p=probs))
        generated.append(tid)
        last = tid
    return {
        "generated_ids": generated,
        "epistemic_trace": epistemic_trace,
        "halted_on_uncertainty": halted,
    }
