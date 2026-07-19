# H15 axiom NaN fix — 2026-07-18

**Symptom:** `bench/results/cell_sweep/variant_H15.json` has `"bpc": null` after 300k overnight (aggregator skipped H15 → 24/25 rows).

**Root cause:** `apply_axiom_gate` allowed raw `tanh ∈ [-1,1]` on LSTM **control** gates (i/f/o). Negative forget/input gates explode cell state → NaN log-probs → `eval_bpc` returns NaN.

**Fix:** Control gates map tanh → `[0,1]` via `0.5*(tanh+1)`; candidate gate `g` maps sigmoid/eml → `[-1,1]`. Backward derivatives updated to match (`char_lstm.cpp`).

**Follow-up:** Remeasured **3.982 BPC** @ `n_train=5000` (2026-07-18); local cell-sweep CSV is 25/25. Optional: re-run H15-only @ 300k to replace the overnight null row (not required for lock).
