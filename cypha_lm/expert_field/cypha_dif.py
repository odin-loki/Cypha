"""DIF routing with dynamic NIG expert field."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

from cypha_lm.config import CyphaLMConfig
from cypha_lm.expert_field.nig_expert import NIGExpert

NOVELTY_THRESHOLD = 0.05
ACTIVE_EXPERT_THRESHOLD = 0.001
_ENTROPY_BINS = 16


def _histogram_entropy(values: np.ndarray, n_bins: int = _ENTROPY_BINS) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=n_bins, density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


@dataclass
class _DIFExpert:
    input_nig: NIGExpert
    output_nig: NIGExpert
    activation_history: list[np.ndarray] = field(default_factory=list)

    def record_activation(self, x: np.ndarray, max_history: int = 256) -> None:
        self.activation_history.append(x.copy())
        if len(self.activation_history) > max_history:
            self.activation_history.pop(0)

    def input_entropy(self) -> float:
        if not self.activation_history:
            return 1.0
        stacked = np.stack(self.activation_history, axis=0)
        per_dim = [_histogram_entropy(stacked[:, d]) for d in range(stacked.shape[1])]
        return float(np.mean(per_dim))


class CyphaDIF:
    """
    DIF routing + NIG expert field with dynamic expert growth.

    Routes context vectors via LLR softmax. Each expert tracks per-dimension
    NIG over input (routing) and output (field prediction). Creates a new
    expert when max routing probability falls below the novelty threshold.
    """

    def __init__(self, config: CyphaLMConfig) -> None:
        self.config = config
        self.field_dim = config.field_dim
        self.max_experts = config.max_experts
        self.novelty_threshold = NOVELTY_THRESHOLD

        self._kappa0 = config.nig_kappa0
        self._alpha0 = config.nig_alpha0
        self._beta0 = config.nig_beta0

        self._input_dim: int | None = None
        self._experts: list[_DIFExpert] = []
        self._warm_start_experts = max(0, int(config.n_experts))

    def _make_expert(self, input_dim: int | None = None) -> _DIFExpert:
        dim = input_dim if input_dim is not None else self._input_dim
        if dim is None:
            raise RuntimeError("Cannot create expert before input dimension is known")
        return _DIFExpert(
            input_nig=NIGExpert(
                self._kappa0, self._alpha0, self._beta0, mu0=0.0, dim=dim
            ),
            output_nig=NIGExpert(
                self._kappa0, self._alpha0, self._beta0, mu0=0.0, dim=self.field_dim
            ),
        )

    def _ensure_input_dim(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._input_dim is None:
            self._input_dim = x.shape[0]
            if self._warm_start_experts > 0 and not self._experts:
                for _ in range(self._warm_start_experts):
                    self._experts.append(self._make_expert())
        elif x.shape[0] != self._input_dim:
            raise ValueError(
                f"Expected input dim {self._input_dim}, got {x.shape[0]}"
            )

    def _prior_log_prob(self, x: np.ndarray) -> float:
        prior = NIGExpert(
            self._kappa0, self._alpha0, self._beta0, mu0=0.0, dim=self._input_dim
        )
        return prior.prior_predictive_log_prob(x)

    def _llr(self, x: np.ndarray) -> np.ndarray:
        prior_lp = self._prior_log_prob(x)
        llrs = np.array(
            [expert.input_nig.predictive_log_prob(x) - prior_lp for expert in self._experts],
            dtype=np.float64,
        )
        return llrs

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        total = exp_logits.sum()
        if total <= 0 or not np.isfinite(total):
            k = len(logits)
            return np.full(k, 1.0 / k, dtype=np.float64)
        return exp_logits / total

    def _active_expert_count(self, probs: np.ndarray) -> int:
        active = int(np.sum(probs > ACTIVE_EXPERT_THRESHOLD))
        if self._warm_start_experts > 0 and len(self._experts) == self._warm_start_experts:
            spread = float(np.max(probs) - np.min(probs)) if probs.size else 0.0
            if spread < 0.05 or active < self._warm_start_experts:
                active = max(active, self._warm_start_experts)
        return active

    def _maybe_grow(self, x: np.ndarray, probs: np.ndarray) -> np.ndarray:
        llrs = self._llr(x)
        max_llr = float(np.max(llrs)) if llrs.size else -np.inf
        novel = probs.size == 0 or np.max(probs) < self.novelty_threshold
        if max_llr < np.log(self.novelty_threshold + 1e-12):
            novel = True
        if novel and len(self._experts) < self.max_experts:
            self._experts.append(self._make_expert())
            probs = self._softmax(self._llr(x))
        return probs

    def route(self, x: np.ndarray) -> np.ndarray:
        """
        Returns routing probabilities p(k|x), shape (K,).
        Creates a new expert if novelty threshold exceeded.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        self._ensure_input_dim(x)

        if not self._experts:
            self._experts.append(self._make_expert())

        if len(self._experts) == 1:
            probs = np.array([1.0], dtype=np.float64)
            return self._maybe_grow(x, probs)

        probs = self._softmax(self._llr(x))
        probs = self._maybe_grow(x, probs)
        return probs

    def train_step(self, x: np.ndarray, y: Union[float, np.ndarray]) -> None:
        """Online update: route x, update winning expert's NIG posterior."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if y_arr.shape != (self.field_dim,):
            raise ValueError(f"Expected target shape ({self.field_dim},)")

        probs = self.route(x)
        winner = int(np.argmax(probs))
        expert = self._experts[winner]
        expert.input_nig.update(x)
        expert.output_nig.update(y_arr)
        expert.record_activation(x)

    def predict(self, x: np.ndarray) -> dict:
        """
        Returns mixture prediction over the expert field.

        mean          : np.ndarray (field_dim,) — vector for GRIA projection
        epistemic_var : float — aggregated parameter uncertainty
        aleatoric_var : float — aggregated irreducible noise
        routing_probs : np.ndarray (K,)
        active_experts: int — experts with p(k|x) > 0.01
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        probs = self.route(x)

        if len(self._experts) == 1:
            expert = self._experts[0]
            mean = np.asarray(expert.output_nig.predict()[0], dtype=np.float64).ravel()
            epistemic = np.asarray(expert.output_nig.epistemic_variance(), dtype=np.float64)
            aleatoric = np.asarray(expert.output_nig.aleatoric_variance(), dtype=np.float64)
            epistemic_var = float(np.mean(epistemic))
            aleatoric_var = float(np.mean(aleatoric))
            return {
                "mean": mean,
                "epistemic_var": epistemic_var,
                "aleatoric_var": aleatoric_var,
                "routing_probs": probs,
                "active_experts": 1,
            }

        means = np.stack(
            [np.asarray(expert.output_nig.predict()[0], dtype=np.float64) for expert in self._experts],
            axis=0,
        )
        mean = probs @ means

        epistemic = np.stack(
            [np.asarray(expert.output_nig.epistemic_variance(), dtype=np.float64) for expert in self._experts],
            axis=0,
        )
        aleatoric = np.stack(
            [np.asarray(expert.output_nig.aleatoric_variance(), dtype=np.float64) for expert in self._experts],
            axis=0,
        )

        if epistemic.ndim == 1:
            epistemic = epistemic[:, np.newaxis]
        if aleatoric.ndim == 1:
            aleatoric = aleatoric[:, np.newaxis]

        epistemic_var = float(np.sum(probs[:, np.newaxis] * epistemic))
        aleatoric_var = float(np.sum(probs[:, np.newaxis] * aleatoric))
        active_experts = self._active_expert_count(probs)

        return {
            "mean": mean,
            "epistemic_var": epistemic_var,
            "aleatoric_var": aleatoric_var,
            "routing_probs": probs,
            "active_experts": active_experts,
        }

    def alpha_per_expert(self) -> np.ndarray:
        """
        GRIA alpha per expert via Grand Unified Law entropy ratio:
        alpha_k = 1 - H(f(X_k)) / H(X_k)
        """
        if not self._experts:
            return np.array([], dtype=np.float64)

        alphas = []
        for expert in self._experts:
            h_x = max(expert.input_entropy(), 1e-8)
            h_f = max(expert.output_nig.predictive_entropy(), 1e-8)
            alpha = 1.0 - h_f / h_x
            alphas.append(float(np.clip(alpha, 0.0, 1.0)))
        return np.asarray(alphas, dtype=np.float64)

    def expert_count(self) -> int:
        return len(self._experts)

    def reset(self) -> None:
        saved_input_dim = self._input_dim
        warm = self._warm_start_experts
        self._experts.clear()
        self._input_dim = None
        self._warm_start_experts = warm
        if warm > 0 and saved_input_dim is not None:
            self._input_dim = saved_input_dim
            for _ in range(warm):
                self._experts.append(self._make_expert())
        elif warm == 0:
            self._input_dim = saved_input_dim if self.config.n_experts > 0 else None

    def get_state(self) -> dict:
        return {
            "input_dim": self._input_dim,
            "experts": [
                {
                    "input_nig": expert.input_nig.state_dict(),
                    "output_nig": expert.output_nig.state_dict(),
                    "activation_history": [h.tolist() for h in expert.activation_history],
                }
                for expert in self._experts
            ],
        }

    def set_state(self, state: dict) -> None:
        self._input_dim = state.get("input_dim")
        self._experts.clear()
        for ex in state.get("experts", []):
            if self._input_dim is None:
                break
            expert = self._make_expert()
            expert.input_nig.load_state_dict(ex["input_nig"])
            expert.output_nig.load_state_dict(ex["output_nig"])
            expert.activation_history = [
                np.asarray(h, dtype=np.float64) for h in ex.get("activation_history", [])
            ]
            self._experts.append(expert)
