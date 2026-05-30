"""Normal-Inverse-Gamma posterior for a single DIF expert."""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy import special, stats


class NIGExpert:
    """
    Normal-Inverse-Gamma posterior with optional per-dimension tracking.

    Posterior: p(mu, sigma^2) = NIG(mu_n, kappa_n, alpha_n, beta_n)
    Predictive: StudentT(mu_n, beta_n*(kappa_n+1)/(alpha_n*kappa_n), 2*alpha_n)

    Update (one observation y per dimension):
        kappa_n = kappa0 + 1
        mu_n    = (kappa0 * mu0 + y) / kappa_n
        alpha_n = alpha0 + 0.5
        beta_n  = beta0 + (kappa0 * (y - mu0)^2) / (2 * kappa_n)

    Epistemic variance:  beta_n / (kappa_n * (alpha_n - 1))
    Aleatoric variance:  beta_n / (alpha_n - 1)
    """

    def __init__(
        self,
        kappa0: float,
        alpha0: float,
        beta0: float,
        mu0: Union[float, np.ndarray] = 0.0,
        dim: int = 1,
    ) -> None:
        if kappa0 <= 0 or alpha0 <= 1 or beta0 <= 0:
            raise ValueError("Require kappa0 > 0, alpha0 > 1, beta0 > 0")
        if dim < 1:
            raise ValueError("dim must be >= 1")

        self.dim = dim
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)

        if dim == 1:
            self.mu0 = float(mu0)
            self.mu_n = float(mu0)
        else:
            mu_arr = np.full(dim, float(mu0), dtype=np.float64) if np.isscalar(mu0) else np.asarray(
                mu0, dtype=np.float64
            ).copy()
            if mu_arr.shape != (dim,):
                raise ValueError(f"mu0 must be scalar or shape ({dim},)")
            self.mu0 = mu_arr.copy()
            self.mu_n = mu_arr.copy()

        self.kappa_n = np.full(dim, kappa0, dtype=np.float64)
        self.alpha_n = np.full(dim, alpha0, dtype=np.float64)
        self.beta_n = np.full(dim, beta0, dtype=np.float64)
        self._n_updates = 0

    def update(self, y: Union[float, np.ndarray]) -> None:
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if y_arr.shape != (self.dim,):
            raise ValueError(f"Expected observation shape ({self.dim},)")

        mu0 = np.atleast_1d(np.asarray(self.mu_n, dtype=np.float64)).copy()
        kappa0 = self.kappa_n.copy()
        alpha0 = self.alpha_n.copy()
        beta0 = self.beta_n.copy()

        self.kappa_n = kappa0 + 1.0
        self.mu_n = (kappa0 * mu0 + y_arr) / self.kappa_n
        self.alpha_n = alpha0 + 0.5
        self.beta_n = beta0 + (kappa0 * (y_arr - mu0) ** 2) / (2.0 * self.kappa_n)
        if self.dim == 1:
            self.mu_n = float(self.mu_n[0])
        self._n_updates += 1

    def _as_scalar_or_array(self, arr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.dim == 1:
            if np.isscalar(arr):
                return float(arr)
            return float(np.asarray(arr).ravel()[0])
        return np.asarray(arr, dtype=np.float64).copy()

    def predict(self) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Returns (mean, epistemic_variance, aleatoric_variance)."""
        epistemic = self.beta_n / (self.kappa_n * (self.alpha_n - 1.0))
        aleatoric = self.beta_n / (self.alpha_n - 1.0)
        return (
            self._as_scalar_or_array(self.mu_n),
            self._as_scalar_or_array(epistemic),
            self._as_scalar_or_array(aleatoric),
        )

    def predictive_log_prob(self, y: Union[float, np.ndarray]) -> float:
        """Log predictive density p(y | posterior), summed over dimensions."""
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if y_arr.shape != (self.dim,):
            raise ValueError(f"Expected observation shape ({self.dim},)")

        scale_sq = self.beta_n * (self.kappa_n + 1.0) / (self.alpha_n * self.kappa_n)
        scale = np.sqrt(np.maximum(scale_sq, 1e-12))
        df = 2.0 * self.alpha_n

        log_probs = stats.t.logpdf(y_arr, df=df, loc=self.mu_n, scale=scale)
        return float(np.sum(log_probs))

    def prior_predictive_log_prob(self, y: Union[float, np.ndarray]) -> float:
        """Log predictive density under the prior hyperparameters."""
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if y_arr.shape != (self.dim,):
            raise ValueError(f"Expected observation shape ({self.dim},)")

        mu0 = self.mu0 if self.dim > 1 else np.array([self.mu0])
        kappa0 = np.full(self.dim, self.kappa0)
        alpha0 = np.full(self.dim, self.alpha0)
        beta0 = np.full(self.dim, self.beta0)

        scale_sq = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0)
        scale = np.sqrt(np.maximum(scale_sq, 1e-12))
        df = 2.0 * alpha0
        log_probs = stats.t.logpdf(y_arr, df=df, loc=mu0, scale=scale)
        return float(np.sum(log_probs))

    def epistemic_variance(self) -> Union[float, np.ndarray]:
        return self._as_scalar_or_array(
            self.beta_n / (self.kappa_n * (self.alpha_n - 1.0))
        )

    def aleatoric_variance(self) -> Union[float, np.ndarray]:
        return self._as_scalar_or_array(self.beta_n / (self.alpha_n - 1.0))

    def predictive_entropy(self) -> float:
        """Differential entropy proxy of the Student-t predictive (per dim, summed)."""
        scale_sq = self.beta_n * (self.kappa_n + 1.0) / (self.alpha_n * self.kappa_n)
        df = 2.0 * self.alpha_n
        # Entropy of Student-t: H = log(B) + (df+1)/2 * (psi((df+1)/2) - psi(df/2)) + log(sqrt(df*pi))
        # Use scipy approximation via sampling-free formula
        ent = (
            0.5 * np.log(np.maximum(scale_sq, 1e-12))
            + 0.5 * (df + 1.0) * (special.digamma((df + 1.0) / 2.0) - special.digamma(df / 2.0))
            + np.log(np.sqrt(df * np.pi))
        )
        return float(np.sum(ent))

    def reset(self) -> None:
        if self.dim == 1:
            self.mu_n = float(self.mu0)
        else:
            self.mu_n = self.mu0.copy()
        self.kappa_n.fill(self.kappa0)
        self.alpha_n.fill(self.alpha0)
        self.beta_n.fill(self.beta0)
        self._n_updates = 0

    def state_dict(self) -> dict:
        return {
            "mu_n": self.mu_n.copy() if self.dim > 1 else float(self.mu_n),
            "kappa_n": self.kappa_n.copy(),
            "alpha_n": self.alpha_n.copy(),
            "beta_n": self.beta_n.copy(),
            "n_updates": self._n_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mu_n = state["mu_n"]
        self.kappa_n = np.asarray(state["kappa_n"], dtype=np.float64)
        self.alpha_n = np.asarray(state["alpha_n"], dtype=np.float64)
        self.beta_n = np.asarray(state["beta_n"], dtype=np.float64)
        self._n_updates = int(state["n_updates"])
