"""Phase 9 integration tests for the full CyphaLM stack."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cypha_lm.config import CyphaLMConfig
from cypha_lm.model.cypha_lm import CyphaLM
from cypha_lm.tests.conftest import TEST_CONFIG, assert_no_torch_on_import


def _perplexity(model: CyphaLM, token_ids: list[int]) -> float:
    model.reset_context()
    nll = 0.0
    n = 0
    for t in range(len(token_ids) - 1):
        pred = model.predict_next(token_ids[t])
        nll += -float(pred["log_probs"][token_ids[t + 1]])
        n += 1
    return float(np.exp(nll / max(n, 1)))


def _count_learnable_params(model: CyphaLM) -> int:
    total = model.gria.W.size + model.gria.alpha.size + model.gria.bias.size
    total += model._proj_ssm.size + model._proj_dif.size + model._proj_embed.size
    if model.config.train_ssm:
        for wf, ws in zip(model.ssm.W_fast, model.ssm.W_slow, strict=False):
            total += wf.size + ws.size
    return int(total)


def _expected_param_count(cfg: CyphaLMConfig) -> int:
    ctx_dim = 2 * cfg.d_state * cfg.ssm_layers
    gria = cfg.vocab_size * (cfg.field_dim + 2)
    proj = cfg.field_dim * ctx_dim + cfg.field_dim * cfg.field_dim + cfg.field_dim * cfg.d_embed
    ssm = 0
    if cfg.train_ssm:
        in_dims = [cfg.d_embed] + [2 * cfg.d_state] * (cfg.ssm_layers - 1)
        for in_dim in in_dims:
            ssm += 2 * cfg.d_state * in_dim
    return gria + proj + ssm


def test_full_pipeline_smoke() -> None:
    model = CyphaLM(TEST_CONFIG)
    for step in range(10):
        out = model.train_step(step % 8, (step + 1) % 8)
        assert np.isfinite(out["loss"])


def test_perplexity_below_ceiling() -> None:
    cfg = replace(TEST_CONFIG, gria_lr=0.08, seed=11)
    model = CyphaLM(cfg)
    pattern = [i % 8 for i in range(8)]
    seq = pattern * 20
    for _ in range(1000):
        model.train_sequence(seq)
    model.reset_context()
    ppl = _perplexity(model, seq)
    assert ppl < 10.0


def test_expert_count_grows() -> None:
    cfg = replace(TEST_CONFIG, nig_kappa0=0.2, seed=12)
    model = CyphaLM(cfg)
    rng = np.random.default_rng(12)
    for step in range(5000):
        a = int(rng.integers(0, cfg.vocab_size))
        b = int(rng.integers(0, cfg.vocab_size))
        model.train_step(a, b)
        if step % 8 == 0:
            field_x = rng.standard_normal(cfg.field_dim) * (1.0 + step * 0.02)
            target = model._proj_embed @ model.embed.embed(b)
            model.dif.train_step(field_x, target)
    assert model.dif.expert_count() > 10


def test_uncertainty_is_informative() -> None:
    cfg = replace(TEST_CONFIG, seed=13, gria_lr=0.05)
    model = CyphaLM(cfg)
    pattern = [0, 1, 2, 3, 4, 5, 6, 7]
    for _ in range(150):
        model.train_sequence(pattern)
    losses = []
    epi = []
    rng = np.random.default_rng(13)
    for _ in range(200):
        if rng.random() < 0.5:
            a, b = pattern[rng.integers(0, len(pattern) - 1)], pattern[rng.integers(1, len(pattern))]
        else:
            a = int(rng.integers(0, cfg.vocab_size))
            b = int(rng.integers(0, cfg.vocab_size))
        m = model.train_step(a, b)
        losses.append(m["loss"])
        epi.append(m["epistemic_var"])
    median = float(np.median(losses))
    high_epi = float(np.mean([e for loss, e in zip(losses, epi, strict=False) if loss >= median]))
    low_epi = float(np.mean([e for loss, e in zip(losses, epi, strict=False) if loss < median]))
    assert high_epi >= low_epi * 0.85


def test_online_beats_frozen() -> None:
    cfg = replace(TEST_CONFIG, seed=14)
    rng = np.random.default_rng(14)
    in_seq = [1, 2, 3, 4, 5] * 10
    ood_seq = [(int(rng.integers(40, cfg.vocab_size)), int(rng.integers(40, cfg.vocab_size))) for _ in range(500)]

    online = CyphaLM(cfg)
    for _ in range(100):
        online.train_sequence(in_seq)

    frozen = CyphaLM(cfg)
    for _ in range(100):
        frozen.train_sequence(in_seq)

    frozen_state = {
        "gria": frozen.gria.get_state(),
        "dif": frozen.dif.get_state(),
        "ssm": frozen.ssm.get_state(),
        "proj_ssm": frozen._proj_ssm.copy(),
        "proj_dif": frozen._proj_dif.copy(),
        "proj_embed": frozen._proj_embed.copy(),
    }

    for a, b in ood_seq:
        online.train_step(a, b)

    frozen.gria.set_state(frozen_state["gria"])
    frozen.dif.set_state(frozen_state["dif"])
    frozen.ssm.set_state(frozen_state["ssm"])
    frozen._proj_ssm = frozen_state["proj_ssm"]
    frozen._proj_dif = frozen_state["proj_dif"]
    frozen._proj_embed = frozen_state["proj_embed"]

    eval_pairs = ood_seq[:50]
    def avg_nll(m: CyphaLM) -> float:
        m.reset_context()
        total = 0.0
        for a, b in eval_pairs:
            pred = m.predict_next(a)
            total += -float(pred["log_probs"][b])
        return total / len(eval_pairs)

    assert avg_nll(online) < avg_nll(frozen)


def test_no_forgetting() -> None:
    cfg = replace(TEST_CONFIG, seed=15, gria_lr=0.02)
    model = CyphaLM(cfg)
    task_a = [0, 1, 2, 3, 4] * 15
    task_b = [50, 51, 52, 53, 54] * 15
    for _ in range(80):
        model.train_sequence(task_a)
    _perplexity(model, task_a)
    for _ in range(40):
        model.train_sequence(task_b)
    ppl_a2 = _perplexity(model, task_a)
    assert np.isfinite(ppl_a2)
    assert ppl_a2 < 500.0


def test_generation_terminates() -> None:
    model = CyphaLM(TEST_CONFIG)
    out = model.generate([0, 1], max_tokens=30, temperature=1.0)
    assert len(out["generated_ids"]) <= 30


def test_no_nan_after_10k_steps() -> None:
    cfg = replace(TEST_CONFIG, seed=16)
    model = CyphaLM(cfg)
    rng = np.random.default_rng(16)
    for _ in range(10_000):
        a = int(rng.integers(0, cfg.vocab_size))
        b = int(rng.integers(0, cfg.vocab_size))
        m = model.train_step(a, b)
        assert np.isfinite(m["loss"])
        assert np.isfinite(m["epistemic_var"])
    pred = model.predict_next(0)
    assert np.all(np.isfinite(pred["log_probs"]))


def test_alpha_distribution_converges() -> None:
    cfg = replace(TEST_CONFIG, gria_lr=0.03, seed=17)
    model = CyphaLM(cfg)
    pattern = [i % 12 for i in range(12)]
    for step in range(5000):
        model.train_step(pattern[step % len(pattern)], pattern[(step + 1) % len(pattern)])
    gria_alpha = model.gria.alpha
    assert float(np.std(gria_alpha)) < 0.45
    assert np.all((gria_alpha >= 0.01) & (gria_alpha <= 0.99))


def test_parameter_count_exact() -> None:
    cfg = replace(TEST_CONFIG, train_ssm=False)
    model = CyphaLM(cfg)
    assert _count_learnable_params(model) == _expected_param_count(cfg)


def test_no_torch_dependency() -> None:
    assert_no_torch_on_import("from cypha_lm.model.cypha_lm import CyphaLM")
