# Optimality P3 — GMM hard-split warm-start — 2026-07-18

**API:** `class_gmm_hard_split_warmstart` in `class_gmm.hpp` / `.cpp`  
**Smoke:** `class_gmm_p3_smoke` (now reports `xor_linear_mean_warm`)

## Results (3 seeds, linear XOR, no kernels)

| Mode | Mean acc |
|------|----------|
| GMM OFF | 0.512 |
| GMM ON (online EM) | 0.505 |
| GMM ON + hard-split warm-start | 0.505 |

Target ≥0.75: **not met**. Warm-start identical to online-only within noise.

## Verdict

**REJECT** for default-on. Keep `use_class_gmm=false`. Kernel RFF remains the working nonlinear XOR path (~76%). Further GMM work needs a different representation (e.g. warm-start in input space before encode, or explicit lobe labels) — not queued this wave.
