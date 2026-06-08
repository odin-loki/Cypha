"""Branch A text router — frozen embeddings + CyphaDIF epistemic gate."""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from cypha_bench.adapters.bench_models import BenchClassifier
from cypha_bench.adapters.frozen_text_embeddings import embed_texts
from cypha_bench.common.metrics import online_train_classifier, standardize_train_test
from cypha_bench.config.load_profile import classification_params, load_profile
from cypha_bench.domains.d09_documents import _load_20news

from ..env_config import branch_a_auto_save, branch_a_checkpoint_base
from .ollama_client import ollama_generate


def branch_a_epistemic_threshold() -> float:
    raw = os.environ.get("CYPHA_BRANCH_A_EPISTEMIC_THRESHOLD", "0.5").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.5


def branch_a_train_samples() -> int:
    raw = os.environ.get("CYPHA_BRANCH_A_N_TRAIN", "1200").strip()
    try:
        return max(100, int(raw))
    except ValueError:
        return 1200


def branch_a_embedding_backend() -> str:
    return os.environ.get("CYPHA_BRANCH_A_EMBED_BACKEND", "auto").strip() or "auto"


def encode_prompt_chars(text: str, *, vocab_size: int = 128) -> list[int]:
    """Map prompt text to token ids for char-level CyphaLM (matches Studio chat)."""
    chars = sorted(set(text))[: max(vocab_size - 1, 1)]
    char2id = {c: i + 1 for i, c in enumerate(chars)}
    return [char2id.get(c, 0) for c in text]


def decode_generated_ids(ids: list[int], prompt: str, *, vocab_size: int = 128) -> str:
    """Decode char ids produced after a prompt (best-effort)."""
    chars = sorted(set(prompt))[: max(vocab_size - 1, 1)]
    id2char = {i + 1: c for i, c in enumerate(chars)}
    id2char[0] = "?"
    return "".join(id2char.get(int(t), "?") for t in ids)


def checkpoint_paths(base: str | Path) -> tuple[Path, Path]:
    """Return ``(meta.json, arrays.npz)`` paths for a checkpoint base."""
    root = Path(os.path.expanduser(str(base)))
    return root.with_suffix(".json"), root.with_suffix(".npz")


def default_checkpoint_base() -> Path:
    return Path(branch_a_checkpoint_base())


class BranchARouter:
    """
    Train CyphaDIF on frozen text embeddings (20 Newsgroups) and route queries.

    Low epistemic variance → ``cypha_route`` (CyphaLM or classify-only).
    High epistemic variance → ``fallback_llm`` (Ollama).
    """

    def __init__(
        self,
        *,
        epistemic_threshold: float | None = None,
        n_train_samples: int | None = None,
        backend: str | None = None,
        seed: int = 42,
        checkpoint_base: str | Path | None = None,
    ) -> None:
        self.epistemic_threshold = (
            branch_a_epistemic_threshold() if epistemic_threshold is None else float(epistemic_threshold)
        )
        self.n_train_samples = branch_a_train_samples() if n_train_samples is None else int(n_train_samples)
        self.backend = backend or branch_a_embedding_backend()
        self.seed = seed
        self._model: BenchClassifier | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._embedding_backend: str | None = None
        self._train_seconds: float | None = None
        self._checkpoint_base = (
            Path(os.path.expanduser(str(checkpoint_base)))
            if checkpoint_base is not None
            else default_checkpoint_base()
        )

    @property
    def checkpoint_base(self) -> Path:
        return self._checkpoint_base

    def save_checkpoint(self, base: str | Path | None = None) -> Path:
        """Persist router weights + standardisation stats (``.json`` + ``.npz``)."""
        if not self.is_trained:
            raise RuntimeError("router is not trained")
        assert self._model is not None and self._mean is not None and self._std is not None

        root = Path(os.path.expanduser(str(base))) if base is not None else self._checkpoint_base
        meta_path, npz_path = checkpoint_paths(root)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        dif_blob = pickle.dumps(self._model.dif.save_state(), protocol=4)
        np.savez_compressed(
            npz_path,
            mean=self._mean.astype(np.float64),
            std=self._std.astype(np.float64),
            dif_pkl=np.frombuffer(dif_blob, dtype=np.uint8),
        )
        meta = {
            "version": 1,
            "n_train_samples": int(self.n_train_samples),
            "input_dim": int(self._mean.shape[0]),
            "backend": self.backend,
            "embedding_backend": self._embedding_backend,
            "seed": int(self.seed),
            "train_seconds": self._train_seconds,
            "n_classes": int(self._model.expert_count()),
            "npz_file": npz_path.name,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        self._checkpoint_base = root
        return meta_path

    def load_checkpoint(self, base: str | Path | None = None) -> dict[str, Any]:
        """Load router from checkpoint base path."""
        root = Path(os.path.expanduser(str(base))) if base is not None else self._checkpoint_base
        meta_path, npz_path = checkpoint_paths(root)
        if not meta_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(f"Branch A checkpoint missing: {root}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays = np.load(npz_path, allow_pickle=False)
        mean = np.asarray(arrays["mean"], dtype=np.float64)
        std = np.asarray(arrays["std"], dtype=np.float64)
        dif_state = pickle.loads(arrays["dif_pkl"].tobytes())

        input_dim = int(meta.get("input_dim", mean.shape[0]))
        seed = int(meta.get("seed", self.seed))
        model = BenchClassifier(input_dim, seed=seed)
        model.dif.load_state(dif_state)
        model.dif.encoder._frozen = True

        self._model = model
        self._mean = mean
        self._std = std
        self._embedding_backend = meta.get("embedding_backend")
        self._train_seconds = meta.get("train_seconds")
        self.n_train_samples = int(meta.get("n_train_samples", self.n_train_samples))
        self.backend = str(meta.get("backend", self.backend))
        self.seed = seed
        self._checkpoint_base = root
        return meta

    def try_load_checkpoint(self, base: str | Path | None = None) -> bool:
        try:
            self.load_checkpoint(base)
            return True
        except (FileNotFoundError, ValueError, KeyError, pickle.UnpicklingError):
            return False

    @property
    def is_trained(self) -> bool:
        return self._model is not None and self._mean is not None and self._std is not None

    def train(self) -> dict[str, Any]:
        """Fit router on 20 Newsgroups (frozen encoder projection)."""
        _, _, _, y, texts = _load_20news(self.n_train_samples)
        y = np.asarray(y)
        x, embed_meta = embed_texts(texts, backend=self.backend)
        x_train, _, y_train, _, = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
        x_train, _ = standardize_train_test(x_train, x_train)
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0] = 1.0

        passes = max(1, int(classification_params(load_profile()).get("n_epochs", 1)))
        t0 = time.perf_counter()
        model = BenchClassifier(x_train.shape[1], seed=self.seed)
        model.dif.encoder._frozen = True
        online_train_classifier(model, x_train, y_train, label_fn=str, passes=passes)
        elapsed = time.perf_counter() - t0

        self._model = model
        self._mean = mean
        self._std = std
        self._embedding_backend = str(embed_meta.get("backend", self.backend))
        self._train_seconds = elapsed
        if branch_a_auto_save():
            try:
                self.save_checkpoint()
            except OSError:
                pass
        return {
            "n_train_samples": int(self.n_train_samples),
            "input_dim": int(x_train.shape[1]),
            "embedding_backend": self._embedding_backend,
            "train_seconds": elapsed,
            "n_classes": int(model.expert_count()),
        }

    def ensure_trained(self) -> None:
        if self.is_trained:
            return
        if self.try_load_checkpoint():
            return
        self.train()

    def route(self, text: str, *, epistemic_threshold: float | None = None) -> dict[str, Any]:
        """Embed query, classify, and decide cypha_route vs fallback_llm."""
        self.ensure_trained()
        assert self._model is not None and self._mean is not None and self._std is not None

        threshold = self.epistemic_threshold if epistemic_threshold is None else float(epistemic_threshold)
        vec, meta = embed_texts([text], backend=self.backend)
        x = (vec[0] - self._mean) / self._std
        label, probs, epistemic = self._model.predict(x)
        conf = float(np.max(probs)) if len(probs) else 0.0
        abstain = epistemic > threshold
        return {
            "label": label,
            "confidence": conf,
            "epistemic_var": float(epistemic),
            "abstain": abstain,
            "embedding_backend": meta.get("backend", self._embedding_backend),
            "action": "fallback_llm" if abstain else "cypha_route",
        }

    def dispatch_generate(
        self,
        text: str,
        lm_engine: Any | None = None,
        *,
        epistemic_threshold: float | None = None,
        max_tokens: int = 128,
        ollama_model: str | None = None,
        ollama_system: str | None = None,
        cypha_lm_strategy: str = "top_p",
        cypha_lm_temperature: float = 0.9,
    ) -> dict[str, Any]:
        """
        Route query then generate with CyphaLM (in-domain) or Ollama (OOD abstain).
        """
        route = self.route(text, epistemic_threshold=epistemic_threshold)
        out: dict[str, Any] = {"route": route, "generation": None}

        if route["abstain"]:
            try:
                gen = ollama_generate(
                    text,
                    model=ollama_model,
                    system=ollama_system or (
                        "You are a helpful assistant. The user's query was flagged as "
                        "out-of-domain for the Cypha router; answer directly and concisely."
                    ),
                )
            except Exception as exc:
                out["generation"] = {
                    "provider": "ollama",
                    "error": str(exc),
                    "text": "",
                }
                return out
            out["generation"] = gen
            return out

        if lm_engine is None:
            out["generation"] = {
                "provider": "none",
                "text": "",
                "reason": "CyphaLM not loaded; routing only (in-domain).",
            }
            return out

        vocab = int(getattr(lm_engine.model.config, "vocab_size", 128))
        prompt_ids = encode_prompt_chars(text, vocab_size=vocab)
        gen_out = lm_engine.generate(
            prompt_ids,
            max_tokens=int(max_tokens),
            strategy=cypha_lm_strategy,
            temperature=float(cypha_lm_temperature),
            top_p=0.92,
        )
        generated = gen_out.get("generated_ids", [])
        out["generation"] = {
            "provider": "cypha_lm",
            "text": decode_generated_ids(generated, text, vocab_size=vocab),
            "generated_ids": [int(t) for t in generated],
            "latency_ms": float(gen_out.get("latency_ms", 0.0)),
            "n_tokens": int(gen_out.get("n_tokens", len(generated))),
        }
        return out

    def summary(self) -> dict[str, Any]:
        meta_path, _ = checkpoint_paths(self._checkpoint_base)
        return {
            "trained": self.is_trained,
            "n_train_samples": self.n_train_samples,
            "epistemic_threshold": self.epistemic_threshold,
            "embedding_backend": self._embedding_backend,
            "train_seconds": self._train_seconds,
            "n_classes": self._model.expert_count() if self._model else 0,
            "checkpoint_base": str(self._checkpoint_base),
            "checkpoint_exists": meta_path.is_file(),
        }
