# AdaptivePredictorMixer (living)

**Status:** slice-3 shipped (2026-07-19) — match voting + hidden-kNN blend + warm-model gating  
**Design sources:** CMIX/PAQ / Nacrith-style ensemble; Cypha explore agents 2026-07-19

## What shipped

| Item | Detail |
|------|--------|
| API | `native/include/cypha/cyphalm/adaptive_predictor_mixer.hpp` |
| Wire-in | Codec-only (`predictive_codec`); not inside `predict_next` |
| Experts | Neural; Laplace n-grams (count-gated); longest-match (multi-hit vote); token-hash kNN; optional bias/SSE |
| Hidden kNN | `CyphaLMModel::hidden_knn_log_probs` blended into neural (`use_hidden_knn`, mix 0.15) |
| Online adapt | `PredictiveCodecOptions::online_adapt` → `adapt_after_predict` (encode=decode from same checkpoint) |
| Options | `use_mixer`, expert toggles, `online_adapt_lr_scale` (default 0.25) |
| Pin safety | Mixer does not change D17 `eval_bpc`; production pin is stacked LSTM **2.816** |

## Defaults that matter

| Knob | Default | Why |
|------|---------|-----|
| `neural_prior` | 6.0 | Softmax prior so cold experts do not swamp a warm LSTM |
| `neural_weight_floor` | 0.55 | Hard floor on neural mix weight |
| `min_context_count` | 16 | N-grams stay off until context is supported |
| Match gate | real match only | Uniform fallback must not enter the mix |
| kNN gate | `knn_max_dist2` | Distant neighbors stay off |
| `use_bias_expert` | **false** | Diluted warm WikiText before enough steps |

## Measured

| Setting | mix BPC | neural BPC |
|---------|---------|------------|
| Cold repetitive (smoke) | ~2.02 | ~6.00 |
| WikiText 8k/2k Hybrid | **3.257** | 3.623 |
| WikiText 40k/4k Hybrid | **2.903** | 3.085 |

Regression note: first match/kNN/bias wiring without gates pushed mix to ~4.8 @ 8k. Gating + neural floor restored gains.

## Encode/decode contract

When `online_adapt=true`, compress mutates weights. Decompress must start from the **same checkpoint** as encode (save → compress → load → decompress). Smoke covers this.

## Next slices

1. Optional A1/A2 = separate GRIA vs LSTM heads into the mixer
2. Attention-over-memory expert (retrieval beyond token-hash kNN)
3. Promote bias/SSE once it wins an ablation on 40k+

See also [`ARCHITECTURE_BPC_LEARNINGS_2026-07-19.md`](ARCHITECTURE_BPC_LEARNINGS_2026-07-19.md).
