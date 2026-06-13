# Intelligence Stats — implementation report (2026-06-13)

## Scope

Read all five papers from `docs/research/intelligence_stats/`, implement Phase 1 in C++, test, and profile against Cypha native stack.

## What was implemented (C++23)

| Component | Path | Tests |
|-----------|------|-------|
| NIG statistic state | `native/include/cypha/intelligence/nig_statistic_state.hpp` | `intelligence_profiler_smoke` |
| Profile enum (7 stats) | `native/include/cypha/intelligence/profile_statistic.hpp` | — |
| Measurers (D_eff, C, r_eu, α) | `native/include/cypha/intelligence/measurers.hpp` | smoke |
| Intelligence profiler | `native/include/cypha/intelligence/intelligence_profiler.hpp` | smoke + CTest |
| Smoke CLI | `native/tools/intelligence_profiler_smoke.cpp` | `native_intelligence_profiler_smoke` |

### Formulas ported

- **D_eff:** participation ratio `(Σλ)² / Σλ²`, normalized by dimension count
- **C:** `1 − ECE` (10-bin expected calibration error)
- **r_eu:** `σ²_e / (σ²_e + σ²_a)`
- **α:** `1 − H(output)/H(input)` (histogram entropy)
- **NIG update:** conjugate normal-inverse-gamma per Paper I §4.2
- **κ (criticality):** `1 − (1/7) Σ|P_i − P*_i|` with target vector from Paper IV
- **Health signal:** diagonal Mahalanobis distance from Welford baseline

## What is not yet implemented

- Paper II: navigation loss, failure-mode table, Bayesian retrain decision
- Paper III: full test bench (`measure_*` on external models), dominance partial order
- Paper IV: `EpistemicThreshold`, `SelfCorrectingInferencer`, profile-guided cell regularizers
- Paper V: `SoftWorldModel`, causal discovery, simulation loop
- CyphaLM live hook: profiler during `cyphalm_bench_native` inference
- REST/diagnostics JSON export for profile matrix

## XOR / kernel LLR (related fix)

Parallel work on the linear XOR ceiling:

- **`score_matrix()`** now calls `_blend_kernel_llr_matrix()` when `use_kernel_llr=True` (was infer-only).
- Benchmark: `scripts/benchmark_xor_kernel_llr.py` — compare linear vs kernel on S3 XOR.

Run:
```bash
```

**Results (2026-05-31, Nyström whitening, M=256, replay off, 5 seeds, 8 passes, blend=1.0):** linear **50.5%**, kernel **61.1%** (Δ **+10.6 pp**). Native XOR bench (Python-matched LRs/temperature/permutation, 3 seeds): **+9.3 pp** vs Python **+9.7 pp** on same config.

## Folder organization

| Before | After |
|--------|-------|
| `Intelligence Stats/*.md` (repo root) | `docs/research/intelligence_stats/` |
| `cypha_som/` (active experiment) | **Removed** — archive: `docs/archive/failed_experiments/cypha_som/` |

## Build & test commands

```powershell
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --target intelligence_profiler_smoke
ctest --test-dir native/build -R native_intelligence_profiler_smoke --output-on-failure
```

## Next steps (priority)

1. Wire profiler into `cypha_diagnostics_run` Phase 5 (profile JSON on reference.cypha infer)
2. ~~Persist `KernelMemory` in sidecar JSON for native XOR training parity~~ — parity fixture + C++ port done; Python `save_state` / v3 binary + CTest `native_kernel_snapshot_roundtrip` (2026-05-31)
3. Add `d_profile` bench domain exporting P-space CSV
4. Implement Paper IV epistemic threshold on CyphaLM generation halt
