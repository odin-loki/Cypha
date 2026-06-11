"""Live GRIA alpha measurement across projection and experts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from cypha_lm.model.cypha_lm import CyphaLM


class AlphaSpectrumAnalyser:
    """Snapshot and track alpha evolution during training."""

    def __init__(self, model: CyphaLM) -> None:
        self.model = model

    def snapshot(self) -> dict:
        gria_alpha = self.model.gria.alpha.copy()
        expert_alpha = self.model.dif.alpha_per_expert()
        all_alpha = np.concatenate([gria_alpha, expert_alpha]) if expert_alpha.size else gria_alpha
        near = float(np.mean(np.abs(all_alpha - 0.5) < 0.1))
        return {
            "gria_projection_alpha": gria_alpha,
            "expert_alpha": expert_alpha,
            "mean_alpha": float(np.mean(all_alpha)),
            "fraction_near_edge_of_chaos": near,
        }

    def track(self, n_steps: int, train_data: list[int]) -> pd.DataFrame:
        import pandas as pd

        records: list[dict] = []
        data = list(train_data)
        if not data:
            data = [0]
        for step in range(n_steps):
            a = step % max(len(data) - 1, 1)
            if len(data) > 1:
                self.model.train_step(data[a], data[a + 1])
            snap = self.snapshot()
            snap["step"] = step
            records.append(snap)
            if step > 0 and step % 100 == 0:
                pass
        rows = []
        for r in records:
            rows.append(
                {
                    "step": r["step"],
                    "mean_alpha": r["mean_alpha"],
                    "fraction_near_edge_of_chaos": r["fraction_near_edge_of_chaos"],
                    "gria_alpha_mean": float(np.mean(r["gria_projection_alpha"])),
                    "expert_alpha_mean": float(np.mean(r["expert_alpha"]))
                    if r["expert_alpha"].size
                    else np.nan,
                }
            )
        return pd.DataFrame(rows)
