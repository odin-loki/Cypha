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

## Phase 4 — shipped (2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| EWC regularizer stub | `ewc_regularizer.hpp/cpp` | `native_ewc_smoke` |
| Curriculum sampler | `curriculum.hpp/cpp` | `native_curriculum_smoke` |
| Paper V simulation REST | `GET /intelligence/simulation` | manual |
| REST `/update` batch + curriculum | `cypha_rest.cpp` | `native_rest_*` |
| Qt curriculum checkbox | `shell_main.cpp`, `bulk_train_worker` | manual |
| Epistemic halt on generate | `cyphalm_generation.cpp` | REST `/generate` |
| Federated merge stub | `federated_aggregate.hpp/cpp`, `cypha_federated_merge` | `native_federated_merge_smoke` |
| RPSM sequence layer (Option B) | `rpsm_sequence_layer.hpp/cpp` | `native_rpsm_sequence_smoke`, `native_cyphalm_bench_rpsm_smoke` |
| Cell variants H02–H14 | `cypha_cell_hypothesis.hpp/cpp`, `eml_activation.hpp` | `native_cell_hypothesis_tier2_smoke` |
| WikiText D17 full profile | `d17_wikitext_full_profile.json` | `native_d17_wikitext_smoke` |

## Phase 5 — shipped (2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Qt epistemic halt on generate | `shell_main.cpp` CyphaLM tab | manual |
| D17 overnight bench profile | `d17_wikitext_overnight_profile.json` | `native_d17_wikitext_overnight_smoke` |
| `--overnight` / `CYPHA_BENCH_OVERNIGHT` | `cyphalm_bench_native.cpp`, `cypha_bench_run.cpp` | smoke + manual 300k |
| RPSM hierarchy + global memory | `rpsm_sequence_layer.hpp/cpp` | `native_rpsm_hierarchy_smoke` |
| RPSM batched LLR default | `infer_cpu.cpp` (`CYPHA_USE_RPSM_LLR=0` opt-out) | `native_rpsm_batched_llr_smoke` |
| NIG-state cell H06 | `nig_state_cell.hpp/cpp` | `native_cell_hypothesis_tier2_smoke` |
| Profile-guided loss in train backprop | `cyphalm_model.cpp`, `profile_guided_loss.cpp` | `native_cyphalm_train_smoke` |
| EWC D16B zero-forgetting probe | `ewc_d16b_smoke.cpp`, REST `ewc_lambda` | `native_ewc_d16b_smoke` |
| Federated coordinator | `cypha_federated_coordinator.cpp` | `native_federated_coordinator_smoke` |
| Paper V simulation loop | `causal_graph.cpp`, `intelligence_rest_routes.cpp` | `native_intelligence_profiler_papers` |
| Release notes v2.3.5 template | `scripts/create_release_notes.ps1` | manual |

## Phase 6 — shipped (2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| 28-variant overnight sweep | `cypha_cell_hypothesis_sweep --overnight-sweep` | `native_cell_hypothesis_overnight_smoke` |
| Cell modules H09–H22 | `gria_gated_mixture`, `reversible_ssm_cell`, `mdl_forget`, `axiom_activation`, `ca_state_cell`, etc. | `native_cell_hypothesis_tier3_smoke` |
| RPSM train loop | `rpsm_sequence_layer::train_step` | `native_rpsm_train_smoke` |
| CyphaLM EWC | `cyphalm_ewc_regularizer.hpp/cpp` | `native_ewc_cyphalm_smoke` |
| Federated worker HTTP | `cypha_federated_worker.cpp` | `native_federated_worker_smoke` |
| Bench domain **d20** | `bench_domains.cpp` | `cypha_bench_run --domain 20` |
| D17 overnight runner | `scripts/run_d17_overnight.ps1` | manual |
| Release notes v2.3.6 template | `scripts/create_release_notes.ps1` | manual |

## Still planned

- **RPSM production** — end-to-end CyphaLM rpsm mode training at 300k scale
- **EWC** full Fisher across all CyphaLM parameters (today: W_ih/W_hh diagonal stub)
- **Federated TLS** — `--tls-cert`/`--tls-key` when OpenSSL linked in httplib
- **GitHub Release** publish via `gh` (needs auth)
- **D17 300k + 28-variant overnight** — wired; run manually with `CYPHA_BENCH_OVERNIGHT=1`

## Commands

```powershell
cmake --build native/build --target intelligence_profiler_papers cypha_intelligence_bench cypha_cell_hypothesis_sweep cyphalm_bench_native
ctest --test-dir native/build -R "native_intelligence|native_cell_hypothesis|native_cyphalm_bench_intelligence" --output-on-failure
cyphalm_bench_native --profile d17 --mode hybrid --n-train 120 --n-eval 40 --intelligence-profile
cypha_cell_hypothesis_sweep --smoke
cypha_cell_hypothesis_sweep --overnight-sweep-smoke
cypha_cell_hypothesis_sweep --overnight-sweep   # CYPHA_BENCH_OVERNIGHT=1 → 300k
curl http://127.0.0.1:8099/intelligence/report
```
