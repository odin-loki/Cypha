# Intelligence Stats — implementation report

**Last updated:** 2026-06-14  
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

## Phase 3 — shipped (2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| CyphaLM live profiler hook | `cyphalm_intelligence_hook.hpp/cpp` | `native_cyphalm_bench_intelligence_profile` |
| Qt self-correct toggle | `shell_main.cpp` predict tab | manual |
| Paper V causal graph stub | `causal_graph.hpp/cpp` | papers + `/intelligence/report` |
| Paper IV profile-guided loss | `profile_guided_loss.hpp/cpp` | `cyphalm_train --profile-guided-loss` |
| Cell hypothesis sweep | `cypha_cell_hypothesis_sweep`, bench **d19** | `native_cell_hypothesis_sweep_smoke` |

## Still planned

- Inject **profile_guided_loss** into per-step CyphaLM `train_step` (needs cell implementations for H02, H06–H22)
- **Full 28-variant** cell hypothesis overnight sweep @ 300k tokens
- **Paper V** full causal simulation loop (beyond `CausalGraphMonitor` stub)
- **RPSM Option B** — sequence layer in CyphaLM

## Commands

```powershell
cmake --build native/build --target intelligence_profiler_papers cypha_intelligence_bench cypha_cell_hypothesis_sweep cyphalm_bench_native
ctest --test-dir native/build -R "native_intelligence|native_cell_hypothesis|native_cyphalm_bench_intelligence" --output-on-failure
cyphalm_bench_native --profile d17 --mode hybrid --n-train 120 --n-eval 40 --intelligence-profile
cypha_cell_hypothesis_sweep --smoke
curl http://127.0.0.1:8099/intelligence/report
```
