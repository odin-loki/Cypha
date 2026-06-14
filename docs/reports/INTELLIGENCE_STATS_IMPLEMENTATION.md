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

## Phase 7 — shipped (2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Baseline lock (D17 hybrid @ 300k) | `bench/BASELINE_LOCK.json` | manual compare |
| Overnight mini-bench smoke | `cyphalm_bench_native --overnight` | `native_overnight_mini_smoke` |
| H16 SR gate laws | `sr_gate_laws.hpp/cpp` | `native_sr_gate_laws_smoke` |
| RPSM d21 end-to-end | `bench d21`, `run_rpsm_overnight.ps1` | `native_d21_rpsm_smoke` |
| CyphaLM EWC embed+head | `cyphalm_ewc_regularizer.cpp` | `native_ewc_cyphalm_smoke` |
| Federated TLS (optional) | `CYPHA_ENABLE_OPENSSL=ON` | `native_federated_tls_smoke` (skip w/o OpenSSL) |
| Release publish helper | `scripts/publish_release.ps1` | manual (`gh auth login`) |
| Release notes v2.3.7 template | `scripts/create_release_notes.ps1` | manual |

## Phase 8 — shipped (2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Cross-domain intelligence bench **d22** | `bench_domains.cpp` → `run_d22_intelligence_cross_profile` | `cypha_bench_run --domain 22` or `--domain-tag d22` |
| D16 EWC probe helper | `run_d16_ewc_probe` (shared with d16 16B) | nested in d22 |
| D22 profile config | `bench/config/d22_intelligence_cross_profile.json` | manual |
| Cross-profile JSON report | `bench/report/tables/d22_intelligence_cross_profile.json` | d22 run |
| Hybrid EWC (SSM/GRIA α) | `cyphalm_ewc_regularizer.hpp/cpp`, `HybridEwcGradStub` | `native_ewc_hybrid_smoke` |
| B0 ngram count prior | `cyphalm_model.cpp` (`ngram_count_table_`, `uses_ngram_count_path`) | hybrid / GRIA-ngram train path |
| H01 α forget gate | `char_lstm.cpp` (`forget_gate_scale`), `cyphalm_config.hpp` `use_alpha_forget_gate` | cell hypothesis H01 |
| CyphaLM checkpoint ngram table | `cyphalm_checkpoint.cpp` save/load `ngram_count_table` | B0 count path round-trip |
| Baseline lock CLI | `tools/cypha_baseline_lock.cpp` | `native_baseline_lock_smoke` |
| Baseline lock PS wrapper | `scripts/update_baseline_lock.ps1` | manual (`--run d17\|d21\|cell-sweep`) |
| CI federated TLS mirror | `scripts/ci_federated_tls_linux.sh` | optional CI `federated_tls` job |
| CTest wiring smoke | `native_d22_cross_smoke` | `ctest -R native_d22` |
| Release notes v2.3.8 template | `scripts/create_release_notes.ps1` | manual |

**CI gate:** **96 CTests** (`ctest -R native_`); optional **`federated_tls`** job runs **`native_federated_tls_smoke`** with **`-DCYPHA_ENABLE_OPENSSL=ON`**.

## Phase 9 — shipped (v2.3.9, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Hybrid EWC weight Fisher (GRIA **U**/**V** + SSM **W_fast**) | `cyphalm_ewc_regularizer.hpp/cpp` — extend `HybridEwcRegularizer` + `HybridEwcGradStub` (`dU`, `dV`, `d_W_fast`) | `native_ewc_weights_smoke` |
| EWC checkpoint persistence | `cyphalm_checkpoint.cpp` — save/load `ewc_anchor` + `ewc_fisher` blocks in `checkpoint.json` | round-trip in `native_ewc_weights_smoke` |
| Unified overnight runner | `scripts/run_overnight_all.ps1` — chains D17, d21 RPSM, cell sweep, `update_baseline_lock.ps1` | manual |
| Overnight lock validation bench **d23** | `bench_domains.cpp` → `run_d23_overnight_lock_validation` | `cypha_bench_run --domain-tag d23` |
| D23 profile config | `bench/config/d23_overnight_lock_profile.json` | manual |
| D23 lock report | `bench/report/tables/d23_overnight_lock_validation.json` | d23 run |
| Release notes v2.3.9 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 9 shipped):** **98 CTests** (`ctest -R native_`); +2 smokes: **`native_d23_overnight_lock_smoke`**, **`native_ewc_weights_smoke`**.

## Phase 10 — shipped (v2.3.10, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Hybrid EWC bias + W_slow Fisher | `cyphalm_ewc_regularizer.hpp/cpp` — extend Phase 9 weight Fisher (`d_gria_bias`, `d_ssm_w_slow`; anchor/Fisher on GRIA **bias** + SSM **W_slow**) | `native_ewc_weights_smoke` |
| Baseline lock `--run all` | `tools/cypha_baseline_lock.cpp` — single CLI chains D17 + d21 + cell-sweep | `native_baseline_lock_smoke` |
| Baseline lock PS wrapper | `scripts/update_baseline_lock.ps1` | manual (`--run d17\|d21\|cell-sweep\|all`) |
| Production lock validation bench **d24** | `bench_domains.cpp` → `run_d24_production_lock_validation` | `cypha_bench_run --domain-tag d24` |
| D24 profile config | `bench/config/d24_production_lock_profile.json` | manual |
| Federated TLS Windows CI mirror | `scripts/ci_federated_tls_windows.ps1` | optional CI `federated_tls` job on Windows |
| Release notes v2.3.10 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 10 shipped):** **99 CTests** (`ctest -R native_`); +1 smoke: **`native_d24_production_lock_smoke`**; extended **`native_ewc_weights_smoke`** (GRIA bias + SSM W_slow Fisher).

## Phase 11 — shipped (v2.3.11, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| WikiText-2 download scripts | `scripts/download_wikitext2.ps1`, `scripts/download_wikitext2.sh` | manual |
| WikiText layout docs | `bench/data/wikitext2/README.md` | manual |
| Gutenberg fallback for d17/d21 | `cyphalm_corpus.cpp` — `gutenberg_fallback` when WikiText absent (Moby Dick preferred) | `native_corpus_smoke` |
| Corpus load smoke CLI | `tools/corpus_smoke.cpp` | `native_corpus_smoke` |
| Bench domain **d25** corpus readiness | `bench_domains.cpp` → `run_d25_corpus_readiness` | `cypha_bench_run --domain-tag d25` |
| D25 profile config | `bench/config/d25_corpus_readiness_profile.json` | manual |
| D25 readiness report | `bench/report/tables/d25_corpus_readiness.json` | d25 run |
| Overnight `-Fast` without WikiText | `run_d17_overnight.ps1`, `run_rpsm_overnight.ps1`, `run_overnight_all.ps1`, `update_baseline_lock.ps1` — propagate `-Fast` + `CYPHA_BENCH_FAST=1` | manual |
| Release notes v2.3.11 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 11 shipped):** **101 CTests** (`ctest -R native_`); +1 smoke: **`native_d25_corpus_smoke`**; also **`native_corpus_smoke`** (direct `load_bench_corpus` probe).

## Still planned

- **D17 300k + 28-variant overnight** — use **`scripts/run_overnight_all.ps1`**; fill `bench/BASELINE_LOCK.json` → `overnight_results` via **`scripts/update_baseline_lock.ps1 -Run all`**
- **GitHub Release** publish via `gh auth login` + `scripts/publish_release.ps1`
- **RPSM @ 300k production benchmark** — d21 wired; full overnight not executed in CI

## Commands

```powershell
cmake --build native/build --target intelligence_profiler_papers cypha_intelligence_bench cypha_cell_hypothesis_sweep cyphalm_bench_native
ctest --test-dir native/build -R "native_intelligence|native_cell_hypothesis|native_cyphalm_bench_intelligence" --output-on-failure
cyphalm_bench_native --profile d17 --mode hybrid --n-train 120 --n-eval 40 --intelligence-profile
cypha_cell_hypothesis_sweep --smoke
cypha_cell_hypothesis_sweep --overnight-sweep-smoke
cypha_cell_hypothesis_sweep --overnight-sweep   # CYPHA_BENCH_OVERNIGHT=1 → 300k
cypha_bench_run --domain-tag d22
pwsh -File scripts/update_baseline_lock.ps1 -Run d17 -Fast
pwsh -File scripts/run_overnight_all.ps1 -Fast          # Phase 9/11: unified overnight; -Fast works without WikiText
cypha_bench_run --domain-tag d23                        # Phase 9: overnight lock validation (shipped)
cypha_baseline_lock --run all --fast --lock-file bench/BASELINE_LOCK.json  # Phase 10: all lock runs
cypha_bench_run --domain-tag d24                        # Phase 10: production lock validation (shipped)
pwsh -File scripts/ci_federated_tls_windows.ps1         # Phase 10: Windows TLS smoke mirror (shipped)
pwsh -File scripts/download_wikitext2.ps1               # Phase 11: fetch WikiText-2 into bench/data/
bash scripts/download_wikitext2.sh                      # Phase 11: Linux/CI equivalent
corpus_smoke                                            # Phase 11: d17+d21 load_bench_corpus probe
cypha_bench_run --domain-tag d25                        # Phase 11: corpus readiness validation
ctest --test-dir native/build -R "native_d23|native_ewc_weights" --output-on-failure
ctest --test-dir native/build -R "native_d24|native_ewc_weights" --output-on-failure  # Phase 10
ctest --test-dir native/build -R "native_d25|native_corpus" --output-on-failure     # Phase 11
bash scripts/ci_federated_tls_linux.sh   # optional TLS smoke (skip without OpenSSL)
curl http://127.0.0.1:8099/intelligence/report
```
