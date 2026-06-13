# Intelligence Stats — implementation report (2026-06-13)

## Scope

Read all five papers from `docs/research/intelligence_stats/`, implement Phase 1 in C++, test, and profile against Cypha native stack.

## What was implemented (C++17)

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
python scripts/benchmark_xor_kernel_llr.py -o artifacts/profiles/xor_kernel_llr.json
```

**Results (2026-06-13, 3 seeds, 4 passes, blend=0.5):** linear **49.8%**, kernel **49.4%** (Δ −0.3 pp) — reservoir RBF does not yet close the diagnostic ceiling (~83% sklearn RBF on latents).

**With blend=1.0, 8 passes:** linear **50.2%**, kernel **52.1%** (Δ **+1.9 pp**) — modest gain; full Nyström / RFF path still required per Paper I / FUTURE §0a.

## Folder organization

| Before | After |
|--------|-------|
| `Intelligence Stats/*.md` (repo root) | `docs/research/intelligence_stats/` |
| `cypha_som/` (active experiment) | Documented as failed: `docs/archive/failed_experiments/cypha_som/` |

## Build & test commands

```powershell
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --target intelligence_profiler_smoke
ctest --test-dir native/build -R native_intelligence_profiler_smoke --output-on-failure
```

## Next steps (priority)

1. Wire profiler into `cypha_diagnostics_run` Phase 5 (profile JSON on reference.cypha infer)
2. Persist `KernelMemory` in sidecar JSON for native XOR training parity
3. Add `d_profile` bench domain exporting P-space CSV
4. Implement Paper IV epistemic threshold on CyphaLM generation halt
