"""Lossy / lossless compression split measurement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cypha_lm.model.cypha_lm import CyphaLM


class CompressionProfiler:
    """
    Estimates distributional (lossy) vs epistemic residual (lossless) information split.
    """

    def measure(self, model: "CyphaLM", test_sequences: list[list[int]]) -> dict:
        per_token: list[float] = []
        losses: list[float] = []
        unigram = model.gria.bias.copy()
        unigram_losses: list[float] = []

        for seq in test_sequences:
            model.reset_context()
            ids = [int(t) for t in seq]
            for t in range(len(ids) - 1):
                pred = model.predict_next(ids[t])
                nxt = ids[t + 1]
                lp = float(pred["log_probs"][nxt])
                losses.append(-lp)
                total = pred["epistemic_var"] + pred["aleatoric_var"] + 1e-12
                per_token.append(float(pred["epistemic_var"] / total))
                unigram_losses.append(float(-unigram[nxt]))

        per_arr = np.asarray(per_token, dtype=np.float64)
        loss_arr = np.asarray(losses, dtype=np.float64)
        uni_arr = np.asarray(unigram_losses, dtype=np.float64)
        lossless = float(np.mean(per_arr)) if per_arr.size else 0.0
        lossy = float(1.0 - lossless)
        bits_model = float(np.mean(loss_arr / np.log(2.0))) if loss_arr.size else 0.0
        bits_uni = float(np.mean(uni_arr / np.log(2.0))) if uni_arr.size else 1.0
        ratio = bits_uni / (bits_model + 1e-12)
        return {
            "lossy_fraction": lossy,
            "lossless_fraction": lossless,
            "per_token_profile": per_arr,
            "compression_ratio": ratio,
            "mean_bits_per_token": bits_model,
            "unigram_bits_per_token": bits_uni,
        }
