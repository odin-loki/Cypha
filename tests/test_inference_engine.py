"""InferenceEngine agrees with raw CyphaDIF for classification."""
from __future__ import annotations

import numpy as np
import pytest

from Cypha import CyphaDIF, VectorEncoder
from cypha_studio.core.inference import InferenceEngine


def _trained_clf(seed: int = 7):
    rng = np.random.default_rng(seed)
    clf = CyphaDIF(
        encoder=VectorEncoder(4),
        field_dim=32,
        rng=np.random.default_rng(seed + 1),
    )
    for _ in range(80):
        lbl = str(rng.integers(0, 3))
        o = np.zeros(4, dtype=np.float64)
        o[int(lbl)] = 2.0
        clf.train_step(rng.normal(0, 0.25, 4) + o, lbl)
    return clf


def test_inference_predict_matches_infer():
    clf = _trained_clf()
    eng = InferenceEngine(clf, preprocessor=None)
    rng = np.random.default_rng(99)
    for _ in range(15):
        x = rng.normal(0, 0.4, 4)
        p = eng.predict(x, use_gh=False)
        pred, conf = clf.infer(x)
        assert p.label == pred
        assert abs(p.confidence - conf) < 1e-9


def test_inference_all_scores_argmax_matches_label():
    clf = _trained_clf()
    eng = InferenceEngine(clf, preprocessor=None)
    rng = np.random.default_rng(100)
    for _ in range(10):
        x = rng.normal(0, 0.5, 4)
        p = eng.predict(x, use_gh=False)
        assert p.all_scores, "expected LLR breakdown"
        best = max(p.all_scores, key=lambda k: p.all_scores[k])
        assert best == p.label
