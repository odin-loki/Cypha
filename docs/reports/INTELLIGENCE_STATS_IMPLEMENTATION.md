# Intelligence Stats — implementation report

**Last updated:** 2026-06-13  
**Papers:** [`docs/research/intelligence_stats/`](../research/intelligence_stats/README.md)

## Phase 1 — shipped (C++23)

| Component | Path | CTest |
|-----------|------|-------|
| NIG statistic state | `native/include/cypha/intelligence/nig_statistic_state.hpp` | smoke |
| 7-stat profile enum | `profile_statistic.hpp` | — |
| Measurers (D_eff, C, r_eu, α, σ, L, τ) | `measurers.hpp/cpp` | smoke + papers |
| Intelligence profiler | `intelligence_profiler.hpp/cpp` | smoke |
| Profile JSON export | `intelligence_profile_json.cpp` | REST `/intelligence/profile` |
| Epistemic threshold (Paper IV) | `epistemic_threshold.hpp/cpp` | papers |
| Soft world monitor (Paper V) | `soft_world_monitor.hpp/cpp` | papers |
| Papers II–V scenarios | `intelligence_profiler_papers.cpp` | `native_intelligence_profiler_papers` |

## Phase 2 — shipped (2026-06-13)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Self-correcting infer loop (Paper IV) | `self_correcting_infer.hpp/cpp` | papers + REST `"self_correct": true` |
| Profile from reference fixture | `profile_from_model.hpp/cpp` | `native_intelligence_bench_smoke` |
| Full report JSON (navigation loss, failure modes, landscape κ) | `intelligence_profile_report_json` | bench **d18** |
| Bench domain **d18** | `bench_domains.cpp` → `run_d18_intelligence_profile` | `cypha_bench_run --domain-tag d18` |
| CLI export | `cypha_intelligence_bench` | smoke |
| REST | `GET /intelligence/report` | manual |
| Diagnostics phase 5 | `cypha_diagnostics_run --phases 5` | inline profiler κ |

### Formulas (all native)

- **D_eff:** participation ratio on latent activations  
- **C:** `1 − ECE` (10-bin)  
- **r_eu:** `σ²_e / (σ²_e + σ²_a)`  
- **α:** `1 − H(output)/H(input)`  
- **κ:** `1 − (1/7) Σ|P_i − P*_i|`  
- **Navigation loss (Paper II):** `||P − P*||²` toward critical targets  
- **Failure modes (Paper II):** marginal threshold flags on P-vector  

## Still planned

- **CyphaLM live hook:** update profiler during `cyphalm_bench_native` token eval (entropy → α)
- **Paper IV cell regularizers** in CyphaLM training (profile-guided loss terms)
- **Paper V** causal graph + simulation loop (beyond `SoftWorldMonitor` maturation signal)
- **Cell hypothesis testbench** — 28 variants ([spec](../research/upgrades/CELL_HYPOTHESIS_TESTBENCH.md))
- **RPSM Option A/B** — matrix refactor + sequence layer ([spec](../research/upgrades/RPSM_COMBINED_SPEC.md))

## Commands

```powershell
cmake --build native/build --target intelligence_profiler_smoke intelligence_profiler_papers cypha_intelligence_bench
ctest --test-dir native/build -R "native_intelligence" --output-on-failure
cypha_intelligence_bench --repo . --out bench/report/tables/d18_intelligence_profile.json
cypha_bench_run --domain-tag d18
curl http://127.0.0.1:8099/intelligence/report
```
