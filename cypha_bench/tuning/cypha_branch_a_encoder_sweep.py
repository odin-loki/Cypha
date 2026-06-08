#!/usr/bin/env python3
"""Compare RFF vs VectorEncoder on frozen sentence-transformer inputs (Branch A)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from Cypha import CyphaDIF, RFFEncoder, VectorEncoder
from cypha_bench.adapters.frozen_text_embeddings import embed_texts
from cypha_bench.common.metrics import evaluate_classification, online_train_classifier, standardize_train_test
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.domains.d09_documents import _load_20news


def _train_cypha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    encoder_kind: str,
    seed: int = 42,
) -> tuple[CyphaDIF, float]:
    d = int(x_train.shape[1])
    if encoder_kind == "rff":
        enc = RFFEncoder(d, D=256, gamma=1.0 / max(d, 1), seed=seed)
    elif encoder_kind == "vector":
        enc = VectorEncoder(d)
    else:
        raise ValueError(encoder_kind)
    enc._frozen = True
    model = CyphaDIF(encoder=enc, rng=np.random.default_rng(seed))
    model.encoder._frozen = True

    class _Wrap:
        def __init__(self, dif: CyphaDIF) -> None:
            self.dif = dif

        def train_step(self, x, label) -> float:
            return float(self.dif.train_step(x, str(label)))

        def predict(self, x):
            full = self.dif.infer_full(x)
            return str(full.get("label", "?")), full

    wrap = _Wrap(model)
    passes = max(1, int(classification_params(load_profile()).get("n_epochs", 1)))
    t0 = time.perf_counter()
    online_train_classifier(wrap, x_train, y_train, label_fn=str, passes=passes)
    return model, time.perf_counter() - t0


def run_sweep(*, n_samples: int, backend: str, seed: int = 42) -> dict:
    _, _, _, y, texts = _load_20news(n_samples)
    y = np.asarray(y)
    x, embed_meta = embed_texts(texts, backend=backend)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    x_train, x_test = standardize_train_test(x_train, x_test)

    rows = {}
    for kind in ("vector", "rff"):
        d = int(x_train.shape[1])
        if kind == "rff":
            enc_obj = RFFEncoder(d, D=256, gamma=1.0 / max(d, 1), seed=seed)
        else:
            enc_obj = VectorEncoder(d)
        enc_dim = int(enc_obj.dim)
        model, train_s = _train_cypha(x_train, y_train, encoder_kind=kind, seed=seed)

        class _Pred:
            def __init__(self, dif: CyphaDIF) -> None:
                self.dif = dif

            def predict(self, xi):
                full = self.dif.infer_full(xi)
                pred = str(full.get("label", "__unknown__"))
                probs_dict = full.get("probs") or {}
                with self.dif.memory._lock:
                    labels = list(self.dif.memory._classes.keys())
                probs = np.array([float(probs_dict.get(l, 0.0)) for l in labels], dtype=np.float64)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                ep = float(full.get("entropy", 0.0))
                return pred, probs, ep

        scores = evaluate_classification(_Pred(model), x_test, y_test, label_fn=str)
        rows[kind] = {
            "accuracy": float(scores["accuracy"]),
            "train_seconds": train_s,
            "encoder_dim": enc_dim,
        }

    lr = LogisticRegression(max_iter=2000, random_state=seed)
    lr.fit(x_train, y_train)
    rows["logreg"] = {"accuracy": float(lr.score(x_test, y_test))}

    return {
        "n_samples": n_samples,
        "input_dim": int(x.shape[1]),
        "embedding": embed_meta,
        "encoders": rows,
        "vector_minus_rff_pp": float((rows["vector"]["accuracy"] - rows["rff"]["accuracy"]) * 100),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Branch A encoder sweep (RFF vs VectorEncoder)")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--write", action="store_true")
    ap.add_argument(
        "--out",
        default=str(_REPO / "cypha_bench" / "config" / "cypha_branch_a_encoder_sweep.json"),
    )
    args = ap.parse_args()

    out = run_sweep(n_samples=args.n_samples, backend=args.backend)
    print(json.dumps(out, indent=2))
    if args.write:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
