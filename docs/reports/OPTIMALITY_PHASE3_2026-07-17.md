# Optimality Phase 3 — Per-class GMM (2026-07-17)

**Build:** `native/build_opt_p3` (Ninja, Release, MinGW 13.2.0)  
**Scope:** Opt-in multimodal `ClassDifferential`; did not touch `build_math`, `build_deff`, `BASELINE_*`, overnight.

## Verdict (honest)

| Acceptance | Result |
|------------|--------|
| XOR (S3) ≥75% **without kernels** | **NO-GO** — GMM ON ≈ **50.5%** (OFF ≈ **51.2%**) |
| Wine / Digits no regress | **OK by default-off** — flag defaults **OFF**; legacy `K×d` path unchanged |
| Format bump + M=1 migration | **Partial** — save/load of `mixing` + `delta_mu` as `M×d` when ON; **default-on / v4 default deferred** |

Online EM responsibilities + log-sum-exp class scores are wired and exerciseable, but the current online update does **not** break the linear XOR wall. Kernel LLR remains the working XOR path (~76%+).

## Shipped (opt-in, default OFF)

| Component | API | Default | Notes |
|-----------|-----|---------|-------|
| Per-class GMM | `CyphaDifMemoryState::use_class_gmm`, `class_gmm_m` (2–4) | **Off** | `D` becomes `K×maxM×d` when ON |
| Mixing | `class_pi`, `class_n_comp` | — | EMA π update from EM responsibilities |
| Score | `class_llr_for_k` / log-sum-exp | — | Uses `cypha/em_step.hpp` |
| Train hook | `TrainStepExtras::use_class_gmm` | **Off** | Propagated in `dif_train_step_vector` |
| Infer | `CyphaInferModel` GMM fields + `classify_at_h` / `score_matrix` | Off | Sync via `sync_infer_model_from_memory` |
| Format | `cypha_format` / `use_class_gmm` / `class_gmm` map; `mixing` tensor | Legacy M=1 still loads | Full default-on bump **required before** enabling by default |
| CTest | `native_class_gmm_p3_smoke` | — | Format + OFF-wall sanity; prints XOR off/on |

## XOR measurement (`class_gmm_p3_smoke`, 3 seeds, 8 passes, no kernel)

| Mode | Mean accuracy |
|------|---------------|
| Linear / GMM **OFF** | **0.512** |
| Linear / GMM **ON** (M=2) | **0.505** |
| Lift | **−0.8 pp** |

Target ≥0.75: **not met**.

## Why no-go (not a build failure)

A 2-component class mixture in latent space *can* represent XOR lobes, but this pass’s **online** soft-EM + shared world variance does not yet place components on the two XOR lobes under the existing encoder/world schedule. Representational capacity is present; the fit path does not yet saturate the Phase-3 bound on S3.

## Promotion path (before default-on)

1. Improve online component placement (e.g. hard EM split on residual, or short batch EM warm-start per class).
2. Re-run `class_gmm_p3_smoke` until `xor_linear_mean_on ≥ 0.75` and Wine/Digits (d01/d02) hold.
3. Then bump default `.cypha` to v4 / `use_class_gmm=1`, regenerate affected goldens, enable by default.

## Goldens

**Not regenerated.** Default numerics unchanged (`use_class_gmm=false`). `native_memory_train` green under `build_opt_p3`.

## Files touched

- `native/include/cypha/class_gmm.hpp`, `native/src/class_gmm.cpp` (new)
- `native/include/cypha/memory_train.hpp`, `native/src/memory_train.cpp`
- `native/include/cypha/infer_cpu.hpp`, `native/src/infer_cpu.cpp`
- `native/include/cypha/train_step_vector.hpp`, `native/src/train_step_vector.cpp`
- `native/src/sync_infer.cpp`
- `native/tools/class_gmm_p3_smoke.cpp` (new)
- `native/CMakeLists.txt`

## Build / test

```
cmake -S native -B native/build_opt_p3 -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_opt_p3 -j8 --target class_gmm_p3_smoke
ctest --test-dir native/build_opt_p3 -R "native_class_gmm_p3_smoke|native_memory_train$" --output-on-failure
```
