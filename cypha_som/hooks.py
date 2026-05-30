"""Integration hooks for CyphaDIF and CellAISSM."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cypha_som import config
from cypha_som.discriminative_feedback import DiscriminativeFeedback
from cypha_som.gng_expert import GNGExpertManager
from cypha_som.gria_controller import GRIAController
from cypha_som.som_encoder import OnlineSOMEncoder


class CyphaSOMHooks:
    def __init__(self, feat_dim: int) -> None:
        self.feat_dim = int(feat_dim)
        self.gng: Optional[GNGExpertManager] = None
        self.som: Optional[OnlineSOMEncoder] = None
        self.gria: Optional[GRIAController] = None
        self.feedback = DiscriminativeFeedback(beta=0.1)
        self._last_bmu: int = 0
        if config.USE_GNG:
            self.gng = GNGExpertManager(feat_dim)
        if config.USE_GRIA_CONTROLLER and self.gng is not None:
            self.gria = GRIAController()

    def post_encode(self, h: np.ndarray, train: bool = True) -> np.ndarray:
        """Optional SOM on latent h (when encoder is linear and SOM not in wrapper)."""
        if not config.USE_SOM_ENCODER:
            return h
        if self.som is None:
            self.som = OnlineSOMEncoder(self.feat_dim, k=8, T=5000)
        return self.som.encode(h, train=train)

    def on_train_step(
        self,
        clf: Any,
        h: np.ndarray,
        post_llrs: Dict[str, float],
    ) -> None:
        """GRIA structural control after a train step (GNG runs in CyphaDIF.train_step)."""
        if self.gria is not None and self.gng is not None:
            act = np.array(list(post_llrs.values()), dtype=np.float64) if post_llrs else h
            self.gria.push(h, act)
            self.gria.act(self._last_bmu, self.gng)

    def modulate_encoder_update(
        self,
        clf: Any,
        dW: np.ndarray,
    ) -> np.ndarray:
        if not config.USE_DISCRIM_FEEDBACK:
            return dW
        with clf.memory._lock:
            K = len(clf.memory._label_order)
            if K == 0:
                return dW
            D = clf.memory._D_buf[:K].copy()
            inv_v = clf.memory.world.inv_v.copy()
        d = self.feedback.compute_d(D, inv_v)
        return self.feedback.modulate(dW, d)

    def merge_context(
        self, base: Dict[str, float], extra: Dict[str, float]
    ) -> Dict[str, float]:
        if not extra:
            return base
        out = dict(base)
        for k, v in extra.items():
            out[k] = out.get(k, 0.0) + v
        return out


def wire_cellai(ssm: Any) -> None:
    """Attach dynamic topology / temporal SOM to CellAISSM if flags set."""
    from cypha_som.hebbian_topology import DynamicHebbianGraph
    from cypha_som.temporal_som import TemporalSOM

    if config.USE_DYNAMIC_TOPOLOGY:
        n = ssm.d_state * 2
        ssm._hebb_graph = DynamicHebbianGraph(n)
    if config.USE_TEMPORAL_SOM:
        ssm._temporal_som = TemporalSOM(M=8, L_max=16)
