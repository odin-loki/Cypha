from __future__ import annotations

import math
import os
from typing import Any, Union

import numpy as np

from Cypha import CyphaDIF, DIFRegressor, TieredContextBuffer, VectorEncoder, RFFEncoder

from cypha_bench.config.load_profile import (
    architecture_params,
    classification_params,
    load_profile,
    regression_params,
    select_classification_regime,
)
from cypha_bench.config.algorithm_variants import apply_algorithm_variants, load_algorithm_variants


def _profile_enabled() -> bool:
    return os.environ.get("CYPHA_BENCH_USE_PROFILE", "1").strip().lower() not in ("0", "false", "no")


def _field_dim(input_dim: int, field_dim: int, *, regression: bool = False) -> int:
    if regression:
        return max(64, int(field_dim))
    return max(32, min(int(field_dim), input_dim * 4))


# Diagnostic finding 2026-05-30: for small-dim inputs (≤30 features), RFF with D=256
# gives +6.5pp on 2-class linear (0.731 vs 0.667) and +2.1pp on iris (0.884 vs 0.863).
# VectorEncoder still wins for dim>30 (e.g. digits 64D: VE=0.918 vs RFF=0.826).
_RFF_DIM_THRESHOLD = 30
_RFF_D = 256


def _make_encoder(input_dim: int, seed: int):
    """Auto-select encoder: RFF for small-dim inputs, VectorEncoder otherwise."""
    if input_dim <= _RFF_DIM_THRESHOLD:
        # gamma = 1/sqrt(d) is a good heuristic for standardized data
        gamma = 1.0 / math.sqrt(max(input_dim, 1))
        return RFFEncoder(input_dim, D=_RFF_D, gamma=gamma, seed=seed)
    return VectorEncoder(input_dim)


def _build_classifier(
    input_dim: int,
    seed: int,
    profile: dict[str, Any] | None = None,
    *,
    regime: str | None = None,
) -> CyphaDIF:
    prof = profile or load_profile()
    if regime is None:
        regime = select_classification_regime(input_dim)
    p = classification_params(prof, regime=regime)
    arch = architecture_params(prof, regime)
    rng = np.random.default_rng(seed)
    encoder = _make_encoder(input_dim, seed)
    fd = encoder.dim if input_dim <= _RFF_DIM_THRESHOLD else _field_dim(input_dim, int(p.get("field_dim", 128)))
    clf = CyphaDIF(
        encoder=encoder,
        field_dim=fd,
        enc_lr=float(p.get("enc_lr", 0.002)),
        delta_lr=float(p.get("delta_lr", 0.05)),
        world_lr=float(p.get("world_lr", 0.008)),
        mdl_lambda=float(p.get("mdl_lambda", 0.001)),
        context_win=int(p.get("context_win", 32)),
        replay_ratio=float(p.get("replay_ratio", arch.get("replay_ratio", 0.30))),
        rng=rng,
        use_kernel_llr=bool(p.get("use_kernel_llr", False)),
    )
    clf.temperature = float(p.get("temperature", 1.0))
    clf.ood_sigma = float(p.get("ood_sigma", arch.get("ood_sigma", 15.0)))
    variants = load_algorithm_variants(prof)
    clf.deliberation_lo = float(variants.get("deliberation_lo", 0.45))
    clf.deliberation_hi = float(variants.get("deliberation_hi", 0.55))
    enc_mode = os.environ.get("CYPHA_ENCODER_UPDATE", "contrastive").strip().lower()
    if enc_mode in ("hebbian", "contrastive"):
        clf.encoder_update_mode = enc_mode
    return clf


def _build_regressor(input_dim: int, seed: int, profile: dict[str, Any] | None = None) -> DIFRegressor:
    prof = profile or load_profile()
    regime = "regression"
    p = regression_params(prof)
    arch = architecture_params(prof, regime)
    rng = np.random.default_rng(seed)
    variants = load_algorithm_variants(prof)
    fd = _field_dim(input_dim, int(p.get("field_dim", 128)), regression=True)
    reg = DIFRegressor(
        encoder=VectorEncoder(input_dim),
        field_dim=fd,
        n_experts=int(p.get("n_experts", 8)),
        target_lr=float(p.get("target_lr", 0.06)),
        replay_ratio=float(p.get("replay_ratio", arch.get("replay_ratio", 0.30))),
        rng=rng,
        cold_start_steps=int(variants.get("cold_start_steps", 20)),
        min_experts_floor=int(variants.get("min_experts_floor", 4)),
        reg_hash_routing=bool(variants.get("reg_hash_routing", False)),
        use_soft_mixture=True,
        use_linear_head=True,
    )
    reg.clf.enc_lr = float(p.get("enc_lr", 0.002))
    reg.clf.delta_lr = float(p.get("delta_lr", 0.05))
    reg.clf.world_lr = float(p.get("world_lr", 0.01))
    reg.clf.mdl_lambda = float(p.get("mdl_lambda", 0.001))
    reg.clf.context = TieredContextBuffer(short_window=int(p.get("context_win", 32)))
    reg.clf.temperature = float(p.get("temperature", 1.05))
    reg.clf.ood_sigma = float(p.get("ood_sigma", arch.get("ood_sigma", 15.0)))
    reg.clf.deliberation_lo = float(variants.get("deliberation_lo", 0.45))
    reg.clf.deliberation_hi = float(variants.get("deliberation_hi", 0.55))
    return reg


class BenchClassifier:
    """CyphaDIF classifier wrapper using tuned everyday profile."""

    def __init__(
        self,
        input_dim: int,
        seed: int = 42,
        profile: dict[str, Any] | None = None,
        *,
        use_profile: bool | None = None,
    ) -> None:
        self.input_dim = input_dim
        if use_profile is None:
            use_profile = _profile_enabled()
        self._profile = profile if use_profile else None
        self._use_profile = use_profile
        self._seed = seed
        self._regime = select_classification_regime(input_dim) if use_profile else None
        self._rng = np.random.default_rng(seed)
        self.dif = (
            _build_classifier(input_dim, seed, profile, regime=self._regime)
            if use_profile
            else CyphaDIF(encoder=_make_encoder(input_dim, seed), rng=self._rng)
        )

    def train_step(self, x: Any, label: Union[str, int]) -> float:
        return float(self.dif.train_step(x, str(label)))

    def predict(self, x: Any) -> tuple[str, np.ndarray, float]:
        full = self.dif.infer_full(x)
        pred = str(full.get("label", "__unknown__"))
        probs_dict = full.get("probs") or {}
        with self.dif.memory._lock:
            labels = list(self.dif.memory._classes.keys())
        if not labels:
            labels = list(probs_dict.keys())
        probs = np.array([float(probs_dict.get(lbl, 0.0)) for lbl in labels], dtype=np.float64)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        ep_var = float(full.get("entropy", 0.0))
        self._last_aleatoric = float(1.0 - full.get("confidence", 0.0))
        return pred, probs, ep_var

    def aleatoric_var(self, x: Any) -> float:
        self.predict(x)
        return float(getattr(self, "_last_aleatoric", 0.0))

    def expert_count(self) -> int:
        with self.dif.memory._lock:
            return len(self.dif.memory._classes)

    def alpha_per_expert(self) -> np.ndarray:
        with self.dif.memory._lock:
            classes = list(self.dif.memory._classes.values())
        if not classes:
            return np.array([], dtype=np.float64)
        alphas = []
        for cls in classes:
            delta = cls.delta_mu
            spread = float(np.std(delta)) if delta.size else 0.0
            alpha = float(np.clip(1.0 - spread / (spread + 1.0), 0.0, 1.0))
            alphas.append(alpha)
        return np.asarray(alphas, dtype=np.float64)

    def mean_alpha(self) -> float:
        alpha = self.alpha_per_expert()
        return float(alpha.mean()) if alpha.size else 0.5

    def reset(self) -> None:
        if self._use_profile:
            self.dif = _build_classifier(self.input_dim, self._seed, self._profile, regime=self._regime)
        else:
            self.dif = CyphaDIF(encoder=VectorEncoder(self.input_dim), rng=self._rng)


class BenchRegressor:
    """DIFRegressor wrapper using tuned everyday profile."""

    def __init__(
        self,
        input_dim: int,
        seed: int = 42,
        profile: dict[str, Any] | None = None,
        *,
        use_profile: bool | None = None,
    ) -> None:
        self.input_dim = input_dim
        if use_profile is None:
            use_profile = _profile_enabled()
        self._profile = profile if use_profile else None
        self._use_profile = use_profile
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.reg = (
            _build_regressor(input_dim, seed, profile)
            if use_profile
            else DIFRegressor(encoder=VectorEncoder(input_dim), rng=self._rng)
        )

    def train_step(self, x: Any, y: Union[float, np.ndarray]) -> float:
        return float(self.reg.train_step(x, y))

    def predict(self, x: Any) -> tuple[float, float, float]:
        y_pred, uncertainty = self.reg.predict(x)
        scalar = float(np.asarray(y_pred, dtype=np.float64).ravel()[0])
        ep_var = float(uncertainty ** 2)
        al_var = float(max(0.0, ep_var * 0.25))
        return scalar, ep_var, al_var

    def expert_count(self) -> int:
        return len(self.reg._expert_mu)

    def alpha_per_expert(self) -> np.ndarray:
        n = self.expert_count()
        if n == 0:
            return np.array([], dtype=np.float64)
        return np.full(n, self.mean_alpha(), dtype=np.float64)

    def mean_alpha(self) -> float:
        n = max(self.expert_count(), 1)
        return float(np.clip(0.5 + 0.1 * math.log1p(n), 0.0, 1.0))

    def finalize_training(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._use_profile and hasattr(self.reg, "fit_linear_head"):
            self.reg.fit_linear_head(X, y)

    def reset(self) -> None:
        if self._use_profile:
            self.reg = _build_regressor(self.input_dim, self._seed, self._profile)
        else:
            self.reg = DIFRegressor(encoder=VectorEncoder(self.input_dim), rng=self._rng)
