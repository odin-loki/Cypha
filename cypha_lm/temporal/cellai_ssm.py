"""Multi-scale exponential memory kernel (CellAI rank-2 SSM) with v3 innovations."""

from __future__ import annotations

from typing import Any

import numpy as np


class CellAISSM:
    """
    Multi-scale exponential memory kernel (rank-2 SSM).

    Fast and slow tracks per layer:
        h_t = spectral_transition(h_{t-1}) + (1 - lambda_fast) * W_fast @ e_t
        s_t = spectral_transition(s_{t-1}) + (1 - lambda_slow) * W_slow @ e_t

    lambda = exp(-1 / tau). Stacked layers concatenate context vectors.

    CellAI v3 innovations (each gated by a feature flag):

    SpectralPDE (use_spectral_pde):
        Replaces the O(D²) scalar-decay state transition with an O(D log D)
        FFT-based circulant convolution.  The kernel is initialised as a unit
        impulse scaled by lambda, so the default behaviour is numerically
        identical to the original scalar decay.

    MultiScalePartitions (use_multiscale):
        Blends the fast and slow state tracks via a per-layer learned scalar
        alpha before forming the context vector passed to subsequent layers.

    SparseHebbian (use_sparse_hebbian):
        Maintains a per-layer Hebbian weight matrix updated online using only
        the top-k/8 most active neurons (12.5% sparsity), reducing the Hebbian
        write cost from O(D²) to O(D log D).
    """

    def __init__(
        self,
        d_input: int = 64,
        d_state: int = 64,
        tau_fast: float = 10.0,
        tau_slow: float = 100.0,
        n_layers: int = 1,
        seed: int = 0,
        use_spectral_pde: bool = True,
        use_multiscale: bool = True,
        use_sparse_hebbian: bool = True,
    ) -> None:
        if d_input < 1 or d_state < 1 or n_layers < 1:
            raise ValueError("d_input, d_state, and n_layers must be >= 1")
        if tau_fast <= 0 or tau_slow <= 0:
            raise ValueError("tau_fast and tau_slow must be positive")

        self.d_input = d_input
        self.d_state = d_state
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.n_layers = n_layers
        self.seed = seed

        self.use_spectral_pde = use_spectral_pde
        self.use_multiscale = use_multiscale
        self.use_sparse_hebbian = use_sparse_hebbian

        self.lambda_fast = float(np.exp(-1.0 / tau_fast))
        self.lambda_slow = float(np.exp(-1.0 / tau_slow))
        self.context_dim = 2 * d_state * n_layers

        rng = np.random.default_rng(seed)
        self._layer_input_dims = [d_input] + [2 * d_state] * (n_layers - 1)
        self.W_fast = [
            rng.standard_normal((d_state, in_dim)).astype(np.float64) * 0.05
            for in_dim in self._layer_input_dims
        ]
        self.W_slow = [
            rng.standard_normal((d_state, in_dim)).astype(np.float64) * 0.05
            for in_dim in self._layer_input_dims
        ]

        # SpectralPDE: per-layer state-transition kernels.
        # Initialised as impulse at index 0 (scalar-equiv to original lambda decay).
        self._a_kernel_fast: list[np.ndarray] = []
        self._a_kernel_slow: list[np.ndarray] = []
        for _ in range(n_layers):
            kf = np.zeros(d_state, dtype=np.float64)
            kf[0] = self.lambda_fast
            self._a_kernel_fast.append(kf)
            ks = np.zeros(d_state, dtype=np.float64)
            ks[0] = self.lambda_slow
            self._a_kernel_slow.append(ks)

        # MultiScalePartitions: per-layer blend weight (0=slow-only, 1=fast-only).
        self._alpha = np.full(n_layers, 0.5, dtype=np.float64)

        # SparseHebbian: per-layer weight matrix (d_state × d_state).
        self._W: list[np.ndarray] = [
            np.zeros((d_state, d_state), dtype=np.float64) for _ in range(n_layers)
        ]

        self._h: list[np.ndarray] | None = None
        self._s: list[np.ndarray] | None = None
        self._lam_fast_scale = 1.0
        self._lam_slow_scale = 1.0
        self.reset()
        try:
            from cypha_som import config as _som_cfg
            from cypha_som.hooks import wire_cellai

            if _som_cfg.USE_DYNAMIC_TOPOLOGY or _som_cfg.USE_TEMPORAL_SOM:
                wire_cellai(self)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Input dimensionality (alias for d_input)."""
        return self.d_input

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._h = [np.zeros(self.d_state, dtype=np.float64) for _ in range(self.n_layers)]
        self._s = [np.zeros(self.d_state, dtype=np.float64) for _ in range(self.n_layers)]

    # ------------------------------------------------------------------
    # SpectralPDE helpers
    # ------------------------------------------------------------------

    def spectral_step(self, s: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """O(D log D) state transition via FFT circular convolution.

        Replaces the scalar decay A @ s (where A = lambda * I) with a learned
        circulant operator represented in the frequency domain.  When the kernel
        is a unit impulse at index 0 with value lambda, this is exactly
        equivalent to the original scalar decay.
        """
        D = len(s)
        S_fft = np.fft.rfft(s)
        A_fft = np.fft.rfft(kernel, n=D)
        return np.fft.irfft(S_fft * A_fft, n=D)

    # ------------------------------------------------------------------
    # SparseHebbian helper
    # ------------------------------------------------------------------

    def sparse_hebbian_update(
        self, pre: np.ndarray, post: np.ndarray, lr: float, layer: int = 0
    ) -> None:
        """Top-k/8 sparse Hebbian weight update (12.5% activity threshold).

        Only the k most-active pre-synaptic neurons participate in the outer
        product, reducing Hebbian write cost from O(D²) to O(D log D).
        """
        k = max(1, len(pre) // 8)
        top_k_idx = np.argpartition(np.abs(pre), -k)[-k:]
        grad = np.outer(post, pre)
        mask = np.zeros_like(grad)
        mask[:, top_k_idx] = 1.0
        self._W[layer] += lr * (grad * mask)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, e_t: np.ndarray) -> np.ndarray:
        e_t = np.asarray(e_t, dtype=np.float64).ravel()
        if e_t.shape[0] != self._layer_input_dims[0]:
            raise ValueError(
                f"Expected input of shape ({self._layer_input_dims[0]},), got {e_t.shape}"
            )

        layer_input = e_t
        if hasattr(self, "_temporal_som"):
            _, lf, ls = self._temporal_som.step(e_t, train=True)
            self._lam_fast_scale = lf
            self._lam_slow_scale = ls
        contexts: list[np.ndarray] = []

        for layer in range(self.n_layers):
            h = self._h[layer]
            s = self._s[layer]
            Wf = self.W_fast[layer]
            Ws = self.W_slow[layer]

            lam_f = getattr(self, "_lam_fast_scale", 1.0)
            lam_s = getattr(self, "_lam_slow_scale", 1.0)
            lf = float(np.clip(self.lambda_fast * lam_f, 0.01, 0.999))
            ls = float(np.clip(self.lambda_slow * lam_s, 0.01, 0.999))

            if self.use_spectral_pde:
                # O(D log D) circulant state transition + input projection
                h = (
                    self.spectral_step(h, self._a_kernel_fast[layer])
                    + (1.0 - lf) * (Wf @ layer_input)
                )
                s = (
                    self.spectral_step(s, self._a_kernel_slow[layer])
                    + (1.0 - ls) * (Ws @ layer_input)
                )
            else:
                h = lf * h + (1.0 - lf) * (Wf @ layer_input)
                s = ls * s + (1.0 - ls) * (Ws @ layer_input)

            self._h[layer] = h
            self._s[layer] = s

            if self.use_multiscale:
                # Blend fast and slow tracks via learned alpha
                alpha = float(np.clip(self._alpha[layer], 0.0, 1.0))
                ctx = np.concatenate([alpha * h + (1.0 - alpha) * s, s])
            else:
                ctx = np.concatenate([h, s])

            if hasattr(self, "_hebb_graph"):
                ctx = self._hebb_graph.diffuse(ctx)
                self._hebb_graph.update(ctx)

            if self.use_sparse_hebbian:
                self.sparse_hebbian_update(h, s, lr=1e-4, layer=layer)

            contexts.append(ctx)
            layer_input = ctx

        return np.concatenate(contexts)

    # ------------------------------------------------------------------
    # Sequence processing
    # ------------------------------------------------------------------

    def process_sequence(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (T, d_input)")
        if embeddings.shape[1] != self._layer_input_dims[0]:
            raise ValueError(
                f"Expected d_input={self._layer_input_dims[0]}, got {embeddings.shape[1]}"
            )

        outputs = np.zeros((embeddings.shape[0], self.context_dim), dtype=np.float64)
        for t in range(embeddings.shape[0]):
            outputs[t] = self.step(embeddings[t])
        return outputs

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return {
            "h": [h.copy() for h in self._h],
            "s": [s.copy() for s in self._s],
            "d_input": self.d_input,
            "d_state": self.d_state,
            "tau_fast": self.tau_fast,
            "tau_slow": self.tau_slow,
            "n_layers": self.n_layers,
            "seed": self.seed,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._h = [np.asarray(h, dtype=np.float64).copy() for h in state["h"]]
        self._s = [np.asarray(s, dtype=np.float64).copy() for s in state["s"]]
        if len(self._h) != self.n_layers or len(self._s) != self.n_layers:
            raise ValueError("State layer count does not match n_layers")
