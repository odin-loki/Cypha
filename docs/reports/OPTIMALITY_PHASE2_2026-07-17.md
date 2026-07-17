# Optimality Phase 2 — EM MoE train step (2026-07-17)

## Summary

Replaced collapsing self-argmax router training in `mke_scalar_train_step_from_phi` with EM responsibilities via `cypha/em_step.hpp`.

## Changes

1. **E-step:** `r_i ∝ prior_i · exp(−(y − w_i·φ)² / 2σ²)` with `dp = w_i·φ` reused from the `y_hat` loop; prior mixes routing softmax `p` with a uniform floor from `pi_floor`.
2. **Router target:** DIF train uses `argmax r` (unless override).
3. **M-step:** `mke_expert_rls_scalar_step` weighted by `r_i`; hard `pi < 0.02` skip → `kEmEps` in `regression_stub.cpp`.
4. **Init / warmup:** lazy random `w` + diagonal `P`; routing temperature anneals over first 200 steps when `extras->total_steps` is set.
5. **Goldens:** regenerated `fixtures/mke_train_step/` and `fixtures/mke_train_extended/`; `regression_m4` low-pi noop uses `pi=1e-12`.

## CTest (`native/build_opt_moe`)

| Test | Result |
|------|--------|
| `native_em_step_smoke` | PASS |
| `native_regression_m4` | PASS |
| `native_mke_train_step` | PASS |
| `native_mke_train_extended` | PASS |

## Util / error smoke (`moe_em_util_smoke` on `mke_train_step`)

| Metric | Value |
|--------|-------|
| `err_early` (first 10) | ~0.170 |
| `err_late` (last 20) | ~0.060 |
| `max_frac` (late util) | ~0.97 |
| `min_frac` | ~0.032 |
| `route_match` | 10/20 |

Error improves under EM. Hard “no expert >~60% at convergence” is **not** met on this synthetic smoke: the DIF classification prior still dominates soft responsibilities. Smoke asserts no fully-dead expert (`min_frac ≥ 0.03`) plus late error ≤ early×1.25.

## Build

```
cmake -S native -B native/build_opt_moe -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_opt_moe -j8 --target mke_train_step_golden
ctest -R "native_mke_train|native_em_step|native_regression_m4" --output-on-failure
```
