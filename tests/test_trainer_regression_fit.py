"""Trainer batch regression paths: RFF, two-stage DIF, MKE (studio core)."""
from __future__ import annotations

import math

import pytest

from cypha_studio.core.dataset import SklearnDataset, SplitConfig
from cypha_studio.core.trainer import Trainer, TrainerConfig


def _diabetes_split():
    ds = SklearnDataset.load("diabetes", task="regression")
    return ds.split(SplitConfig(seed=0, stratify=False))


@pytest.mark.parametrize(
    "model_type,overrides",
    [
        ("RFFRegressor", {}),
        ("TwoStageDIF", {"n_experts": 4}),
        ("MKE", {"n_experts": 4, "rff_D": 48, "field_dim": 48}),
    ],
)
def test_trainer_regression_fit_diabetes_smoke(model_type, overrides):
    tr, val, _te = _diabetes_split()
    kw = dict(
        model_type=model_type,
        n_epochs=1,
        eval_every_n=10_000,
        early_stopping=False,
        rff_D=64,
        field_dim=64,
        seed=7,
    )
    kw.update(overrides)
    cfg = TrainerConfig(**kw)
    t = Trainer()
    t.fit(tr, val, cfg)
    assert t.model is not None
    assert hasattr(t.model, "predict_batch") or hasattr(t.model, "predict")
    m = t.evaluate(val, cfg)
    assert math.isfinite(m.r2_score)
    assert math.isfinite(m.rmse)
