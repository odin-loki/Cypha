"""Trainer.evaluate after wiring _model/_preprocessor (registry compare path)."""
from __future__ import annotations

from cypha_studio.core.dataset import Preprocessor, SklearnDataset, SplitConfig
from cypha_studio.core.trainer import Trainer, TrainerConfig


def test_evaluate_manual_model_matches_trained_trainer():
    ds = SklearnDataset.load("iris")
    tr, val, te = ds.split(SplitConfig(seed=42))
    pre = Preprocessor()
    pre.fit(tr.X)
    tr.preprocessor = pre
    val.preprocessor = pre
    te.preprocessor = pre
    cfg = TrainerConfig(
        feat_dim=32,
        field_dim=32,
        n_epochs=1,
        eval_every_n=9999,
        seed=42,
    )
    trained = Trainer()
    trained.fit(tr, val, cfg)

    fresh = Trainer()
    fresh._model = trained.model
    fresh._preprocessor = pre
    m1 = trained.evaluate(te, cfg)
    m2 = fresh.evaluate(te)
    assert abs(m1.accuracy - m2.accuracy) < 1e-5
    assert abs(m1.macro_f1 - m2.macro_f1) < 1e-5
