# Intelligence Stats — implementation report

**Last updated:** 2026-06-23  
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

## Phase 12 — shipped everywhere (v2.3.12, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Medium overnight tier | `-Medium` on `run_d17_overnight.ps1`, `run_rpsm_overnight.ps1`, `run_overnight_all.ps1`, `update_baseline_lock.ps1` — 5k train / 256 eval, real corpus | manual |
| `cypha_baseline_lock --medium` | `tools/cypha_baseline_lock.cpp` — `status=medium_smoke` in lock JSON | `native_baseline_lock_smoke` |
| Bench domain **d26** medium overnight validation | `bench_domains.cpp` → `run_d26_medium_overnight_validation` | `cypha_bench_run --domain-tag d26` |
| D26 profile config | `bench/config/d26_medium_overnight_profile.json` | manual |
| D26 validation report | `bench/report/tables/d26_medium_overnight_validation.json` | d26 run |
| Baseline lock PS validator | `scripts/validate_baseline_lock.ps1` (`-LockFile`, `-Strict`) | manual |
| Baseline lock C++ validator | `tools/baseline_lock_smoke.cpp` → `baseline_lock_validate` | `native_baseline_lock_validate_smoke` |
| Release notes preview | `scripts/publish_release.ps1 -DryRun` / `-NotesOnly` | manual |
| Optional CI corpus job | `.github/workflows/ci.yml` → `corpus_and_d25` | optional (`continue-on-error`) |
| Release notes v2.3.12 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 12 shipped everywhere):** **103 CTests** (`ctest -R native_`); +2 smokes: **`native_d26_medium_overnight_smoke`**, **`native_baseline_lock_validate_smoke`**.

## Phase 13 — shipped (v2.3.13, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Production overnight tier | `-Production` on `run_d17_overnight.ps1`, `run_rpsm_overnight.ps1`, `run_overnight_all.ps1`, `update_baseline_lock.ps1` — 300k train / 2000 eval, real corpus (mutually exclusive with `-Fast`/`-Medium`) | manual |
| `cypha_baseline_lock --production` | `tools/cypha_baseline_lock.cpp` — `status=production`, `CYPHA_BENCH_FULL_CORPUS=1`, `CYPHA_BENCH_OVERNIGHT=1`, `CYPHA_BENCH_FULL_N_TRAIN=300000` | `native_baseline_lock_smoke` |
| Dedicated production runner | `scripts/run_production_overnight.ps1` — chains `run_overnight_all.ps1 -Production`, logs to `bench/results/production_overnight_<timestamp>.log` | manual |
| Bench domain **d27** production overnight validation | `bench_domains.cpp` → `run_d27_production_lock_validation` | `cypha_bench_run --domain-tag d27` |
| D27 profile config | `bench/config/d27_production_lock_profile.json` | manual |
| D27 validation report | `bench/report/tables/d27_production_lock_validation.json` | d27 run |
| Baseline lock PS production validator | `scripts/validate_baseline_lock.ps1 -Production` — when `n_train >= 300000`, require `status=production` or `completed`, BPC within 0.05 of 2.873 pin | manual |
| Baseline lock C++ production validator | `tools/baseline_lock_smoke.cpp` → `baseline_lock_validate --production` | `native_baseline_lock_validate_smoke` |
| Release notes v2.3.13 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 13 shipped):** **104 CTests** (`ctest -R native_`); +1 smoke: **`native_d27_production_lock_smoke`**. Full 300k production overnight is **not** run in CI — maintainer-only via **`run_production_overnight.ps1`**.

## Phase 14 — shipped (v2.3.14, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Baseline lock status validator fix | `scripts/validate_baseline_lock.ps1`, `tools/baseline_lock_smoke.cpp` — accept **`medium_smoke`** and **`production`** in addition to **`fast_smoke`** / **`completed`** (fixes production/medium overnight lock validation) | `native_baseline_lock_validate_smoke` |
| Cell sweep artifact path | `cypha_cell_hypothesis_sweep`, `cypha_baseline_lock --output-dir`, `bench_paths::results_dir()` — default overnight sweep output **`bench/results/cell_sweep`** (was repo-root **`results/`**) | manual |
| Bench domain **d28** unified overnight completion validation | `bench_domains.cpp` → `run_d28_overnight_complete_validation` | `cypha_bench_run --domain-tag d28` |
| D28 profile config | `bench/config/d28_overnight_complete_profile.json` | manual |
| D28 validation report | `bench/report/tables/d28_overnight_complete_validation.json` | d28 run |
| Post-overnight finalize script | `scripts/finalize_production_overnight.ps1` — **`validate_baseline_lock.ps1 -Production`**, d27 + d28 bench domains, lock section summary | manual (chained from **`run_production_overnight.ps1`**) |
| Overnight script wiring | `run_overnight_all.ps1`, `update_baseline_lock.ps1`, `run_production_overnight.ps1` — cell-sweep **`OutputDir`** defaults to **`bench/results/cell_sweep`** | manual |
| Release notes v2.3.14 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 14 shipped):** **106 CTests** (`ctest -R native_`); +2 smokes: **`native_d28_overnight_complete_smoke`**, **`native_baseline_lock_validate_production_status`**. d28 validates cross-section consistency: **`overnight_results`**, **`rpsm_results`**, and **`cell_sweep_results`** must share **`n_train`** / **`n_eval`**; if **`n_train < 300000`**, reports **`pending_overnight_complete`** (smoke pass); if **≥ 300k**, require **`status=production`** or **`completed`** on all three sections.

## Phase 15 — shipped (v2.3.15, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d29** release readiness validation | `bench_domains.cpp` → `run_d29_release_readiness_validation` | `cypha_bench_run --domain-tag d29` |
| D29 profile config | `bench/config/d29_release_readiness_profile.json` | manual |
| D29 validation report | `bench/report/tables/d29_release_readiness_validation.json` | d29 run |
| Validate-all env hooks | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_OVERNIGHT_COMPLETE=1`** (d28 after lock validate), **`CYPHA_VALIDATE_RELEASE_READINESS=1`** (d29), **`CYPHA_VALIDATE_PRODUCTION=1`** (production lock validate), **`CYPHA_STRICT_TEST_COUNT=1`** (fail when count ≠ 107) | manual |
| Lock commit helper | `scripts/commit_production_lock.ps1` — chains **`finalize_production_overnight.ps1`**, stage/commit **`bench/BASELINE_LOCK.json`** @ 300k (`-DryRun` / `-Force`; never pushes) | manual |
| Production overnight watcher | `scripts/watch_production_overnight.ps1` — log growth, process PIDs, lock section summary; stall warn after 30m | manual |
| Release notes v2.3.15 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 15 shipped):** **107 CTests** (`ctest -R native_`); +1 smoke: **`native_d29_release_readiness_smoke`**. d29 validates schema + production tier (d27) + overnight-complete (d28) + release script presence; **`pending_release`** when production/overnight gates pending (smoke pass); **`release_ready`** when both **`production_validated`** and **`overnight_complete_validated`**.

## Phase 16 — shipped (v2.3.16, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d30** artifact path hygiene validation | `bench_domains.cpp` → `run_d30_artifact_hygiene_validation` | `cypha_bench_run --domain-tag d30` |
| D30 profile config | `bench/config/d30_artifact_hygiene_profile.json` | manual |
| D30 validation report | `bench/report/tables/d30_artifact_hygiene_validation.json` | d30 run |
| Legacy results migration | `scripts/migrate_legacy_results.ps1` (`-DryRun`, `-RemoveLegacy`) | manual |
| Overnight progress logging | `run_d17_overnight.ps1` → `bench/results/overnight_d17_<timestamp>.log`; stderr **`[cyphalm]`** / **`[cell_sweep]`** | manual |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_ARTIFACT_HYGIENE=1`** (d30 when profile exists) | manual |
| Release notes v2.3.16 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 16 shipped):** **108 CTests** (`ctest -R native_`); +1 smoke: **`native_d30_artifact_hygiene_smoke`**. d30 validates legacy repo-root **`results/`** path detection in **`cell_sweep_results.artifact_path`**, **`bench/results/.gitkeep`** presence; **`hygiene_ok`** or **`legacy_artifact_path`** (smoke pass).

## Phase 17 — shipped (v2.3.17, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d31** post-overnight pipeline validation | `bench_domains.cpp` → `run_d31_post_overnight_pipeline_validation` | `cypha_bench_run --domain-tag d31` |
| D31 profile config | `bench/config/d31_post_overnight_pipeline_profile.json` | manual |
| D31 validation report | `bench/report/tables/d31_post_overnight_pipeline_validation.json` | d31 run |
| Poll + finalize automation | `scripts/poll_and_finalize_overnight.ps1` | manual |
| Legacy cleanup wrapper | `scripts/cleanup_legacy_results.ps1` (+ `migrate_legacy_results.ps1 -ArchiveLegacy`) | manual |
| Production overnight chain | `run_production_overnight.ps1` → `finalize_production_overnight.ps1` → `commit_production_lock.ps1 -DryRun` | manual |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE=1`** (d31 when profile exists) | manual |
| Release notes v2.3.17 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 17 shipped):** **109 CTests** (`ctest -R native_`); +1 smoke: **`native_d31_post_overnight_pipeline_smoke`**. d31 validates d27→d30 chain + pipeline script presence (`poll_and_finalize_overnight.ps1`, `finalize_production_overnight.ps1`, `commit_production_lock.ps1`, `migrate_legacy_results.ps1`); **`pipeline_ok`** (smoke pass).

## Phase 18 — shipped (v2.3.18, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d32** production complete validation | `bench_domains.cpp` → `run_d32_production_complete_validation` | `cypha_bench_run --domain-tag d32` |
| D32 profile config | `bench/config/d32_production_complete_profile.json` | manual |
| D32 validation report | `bench/report/tables/d32_production_complete_validation.json` | d32 run |
| Unified post-overnight validator | `scripts/validate_production_complete.ps1` — baseline lock + finalize + d31/d30 | manual |
| Background poll + finalize | `scripts/start_poll_finalize_background.ps1` | manual |
| Cell sweep progress sidecar | `cypha_cell_hypothesis_sweep` → `overnight_progress.log` | manual |
| Release publish gh auth preflight | `scripts/publish_release.ps1` | manual |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_PRODUCTION_COMPLETE=1`** (d32 when profile exists) | manual |
| Release notes v2.3.18 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 18 shipped):** **110 CTests** (`ctest -R native_`); +1 smoke: **`native_d32_production_complete_smoke`**. d32 validates production tier + script presence (`validate_production_complete.ps1`, `start_poll_finalize_background.ps1`); **`pending_production_complete`** when **`n_train < 300000`** (smoke pass); **`production_complete_validated`** when **≥ 300k** and all gates pass.

## Phase 19 — shipped (v2.3.19, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d33** release publish validation | `bench_domains.cpp` → `run_d33_release_publish_validation` | `cypha_bench_run --domain-tag d33` |
| D33 profile config | `bench/config/d33_release_publish_profile.json` | manual |
| D33 validation report | `bench/report/tables/d33_release_publish_validation.json` | d33 run |
| Release publish smoke gate | `scripts/verify_release_publish.ps1` — production complete + d33 + `publish_release.ps1 -DryRun` | manual |
| Poll BuildDir auto-detect | `poll_and_finalize_overnight.ps1`, `start_poll_finalize_background.ps1` | manual |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_RELEASE_PUBLISH=1`** (d33 when profile exists) | manual |
| Release notes v2.3.19 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 19 shipped):** **111 CTests** (`ctest -R native_`); +1 smoke: **`native_d33_release_publish_smoke`**. d33 validates publish script presence + production/overnight-complete tiers; **`pending_release_publish`** when **`n_train < 300000`** (smoke pass); **`release_publish_ready`** when **≥ 300k** and all gates pass.

## Phase 20 — shipped (v2.3.20, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d34** repo smoke hygiene validation | `bench_domains.cpp` → `run_d34_repo_smoke_hygiene_validation` | `cypha_bench_run --domain-tag d34` |
| D34 profile config | `bench/config/d34_repo_smoke_hygiene_profile.json` | manual |
| D34 validation report | `bench/report/tables/d34_repo_smoke_hygiene_validation.json` | d34 run |
| Repo smoke cleanup helper | `scripts/cleanup_repo_smoke_artifacts.ps1` — remove repo-root `d##_smoke.json` spill files | manual |
| Poll heartbeat logging | `poll_and_finalize_overnight.ps1` — per-cycle **HEARTBEAT** (timestamp, process count, lock `n_train`) | manual |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_REPO_SMOKE_HYGIENE=1`** (d34 when profile exists) | manual |
| Release notes v2.3.20 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 20 shipped):** **112 CTests** (`ctest -R native_`); +1 smoke: **`native_d34_repo_smoke_hygiene_smoke`**. d34 validates repo-root smoke JSON leak detection + cleanup script presence; **`repo_root_smoke_ok`** or **`repo_root_smoke_leak`** (smoke pass).

## Phase 21 — shipped (v2.3.21, 2026-06-14)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d35** lock commit pipeline validation | `bench_domains.cpp` → `run_d35_lock_commit_pipeline_validation` | `cypha_bench_run --domain-tag d35` |
| D35 profile config | `bench/config/d35_lock_commit_pipeline_profile.json` | manual |
| D35 validation report | `bench/report/tables/d35_lock_commit_pipeline_validation.json` | d35 run |
| Production pipeline smoke gate | `scripts/verify_production_pipeline.ps1` — production complete + release publish + repo smoke cleanup preview + optional d35 | manual |
| Overnight watch variant progress | `scripts/watch_production_overnight.ps1` — **`done/28`** cell sweep progress; poll dedupe + heartbeat fix | manual |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE=1`** (d35 when profile exists) | manual |

**CI gate (Phase 21 shipped):** **113 CTests** (`ctest -R native_`); +1 smoke: **`native_d35_lock_commit_pipeline_smoke`**. d35 validates post-overnight commit toolchain script presence (`commit_production_lock.ps1`, `finalize_production_overnight.ps1`, `poll_and_finalize_overnight.ps1`, `validate_production_complete.ps1`); **`pending_lock_commit`** when **`n_train < 300000`** (smoke pass); **`lock_commit_ready`** when **≥ 300k** and production + overnight-complete gates pass.

## Phase 22 — shipped (v2.3.22)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d36** production pipeline E2E validation | `bench_domains.cpp` → `run_d36_pipeline_e2e_validation` | `cypha_bench_run --domain-tag d36` |
| D36 profile config | `bench/config/d36_pipeline_e2e_profile.json` | manual |
| D36 validation report | `bench/report/tables/d36_pipeline_e2e_validation.json` | d36 run |
| Post-overnight maintainer wrapper | `scripts/run_post_overnight.ps1` — poll/finalize/commit + **`verify_production_pipeline.ps1`** | manual |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_PIPELINE_E2E=1`** (d36 when profile exists) | manual |

**CI gate (Phase 22 shipped):** **114 CTests** (`ctest -R native_`); +1 smoke: **`native_d36_pipeline_e2e_smoke`**. d36 validates full maintainer overnight→publish toolchain script presence + **d27–d35** bench profiles; **`pending_pipeline_e2e`** when **`n_train < 300000`** (smoke pass); **`pipeline_e2e_ready`** when **≥ 300k** and production + overnight-complete gates pass.

## Phase 23 — shipped (v2.3.23)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d37** overnight lock refresh validation | `bench_domains.cpp` → `run_d37_lock_refresh_validation` | `cypha_bench_run --domain-tag d37` |
| D37 profile config | `bench/config/d37_lock_refresh_profile.json` | manual |
| D37 validation report | `bench/report/tables/d37_lock_refresh_validation.json` | d37 run |
| In-flight artifact migrate | `scripts/migrate_inflight_overnight_artifacts.ps1` — merge repo-root **`results/`** spill into **`bench/results/cell_sweep/`** | manual |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_LOCK_REFRESH=1`** (d37 when profile exists) | manual |
| Offline release notes | `scripts/publish_release.ps1` — **`-NotesPath`** for offline **`gh release create`** workflow | manual |

**CI gate (Phase 23 shipped):** **115 CTests** (`ctest -R native_`); +1 smoke: **`native_d37_lock_refresh_smoke`**. d37 validates post-overnight baseline lock update toolchain (`update_baseline_lock.ps1`, migrate scripts, `finalize_production_overnight.ps1`); **`pending_lock_refresh`** when **`n_train < 300000`** (smoke pass); **`lock_refresh_ready`** when **≥ 300k** and production + overnight-complete gates pass.

## Phase 24 — prep (v2.3.24)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d38** production overnight completion certificate | `bench_domains.cpp` → `run_d38_overnight_certificate_validation` | `cypha_bench_run --domain-tag d38` |
| D38 profile config | `bench/config/d38_overnight_certificate_profile.json` | manual |
| D38 validation report | `bench/report/tables/d38_overnight_certificate_validation.json` | d38 run |
| Poll auto-commit | `scripts/poll_and_finalize_overnight.ps1` — **`-AutoCommit`** (post-finalize **`commit_production_lock.ps1 -Force`** when **`n_train >= 300000`**) | manual |
| Background poll auto-commit | `scripts/start_poll_finalize_background.ps1` — **`-AutoCommit`** passthrough | manual |
| Variant stall detector | `scripts/watch_production_overnight.ps1` — **`-StallMinutes`**, **`-LogFile`** | manual |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE=1`** (d38 when profile exists) | manual |

**CI gate (Phase 24 prep):** **115 CTests** today; **116** when **`native_d38_overnight_certificate_smoke`** merges (+1). d38 validates full 300k overnight completion (overnight/RPSM/cell-sweep alignment, **≥ 28** variants); **`pending_overnight_certificate`** when **`n_train < 300000`** (smoke pass); **`overnight_certificate_ready`** when **≥ 300k** and production + overnight-complete + variant gates pass.

## Phase 25 — shipped (v2.3.25, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| CyphaLM token-stream monitor | `lm_intelligence_monitor.hpp/cpp` | `native_intelligence_lm_monitor_smoke` |
| Profile completeness validation | `profile_completeness.hpp/cpp` | REST `/intelligence/profile`, bench `--intelligence-profile` |
| REST live profiler report | `GET /intelligence/report?source=live` | manual |
| REST predict real measurers | `cypha_rest.cpp` `POST /predict` | `native_rest_*` |
| Bench domain **d39** | `bench_domains.cpp` → `run_d39_intelligence_monitor_profile_validation` | `cypha_bench_run --domain-tag d39` |
| D39 profile config | `bench/config/d39_intelligence_monitor_profile.json` | manual |
| D39 validation report | `bench/report/tables/d39_intelligence_monitor_profile_validation.json` | d39 run |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_INTELLIGENCE_MONITOR=1`** (d39 when profile exists) | manual |

**Bugfixes (Phase 25 completion):** `lm_intelligence_monitor` — τ from **`batch.tau`** (not **`batch.sequence`** OOB); **`embed_dim != field_dim`** guard on **`batch.input`**; `cyphalm_model` — skip **`update_profiler_from_lm_token`** when monitor active; d39 — throw on subprocess exit **≠ 0**.

**Full-process monitoring (train / eval / generate / REST / bench):**

| Process | Hook | Stats measured |
|---------|------|----------------|
| CyphaLM train | `update_profiler_from_lm_step` via `LmIntelligenceMonitor` | α, D_eff, σ, L, τ, r_eu, C |
| CyphaLM eval BPC | `eval_bpc(..., profiler*)` | α, r_eu, C (token stream) |
| CyphaLM generate | `generate_decode(..., profiler*, monitor*)` | α, r_eu (epistemic halt path) |
| REST `/predict` | `json_predict_impl` measurers | α, r_eu, C (optional label), τ (self-correct) |
| REST `/intelligence/profile` | `profile_completeness_to_json` | completeness + κ |
| REST `/intelligence/report?source=live` | session `g_profiler` | full report + completeness |
| `cyphalm_bench_native --intelligence-profile` | eval profiler + completeness JSON | all 7 stats |

**CI gate (Phase 25 shipped):** **117 CTests** (`ctest -R native_`); +1 smoke: **`native_d39_intelligence_monitor_smoke`**. d39 validates 7-stat monitoring toolchain file presence + `cyphalm_bench_native --intelligence-profile` stdout; **`pending_profile_monitor`** when completeness incomplete (smoke pass); **`profile_monitor_ready`** when **`profile_completeness.all_complete == true`**.

## Phase 26 — shipped (v2.3.26, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| CyphaLM math integration | `cyphalm_math_integration.hpp/cpp` | `native_d40_math_integration_smoke` |
| Profile-guided navigation loss | `profile_guided_loss.hpp/cpp` (`navigation_loss_total`, 7-stat λ) | wired in `--math-integration` train path |
| Bench domain **d40** | `bench_domains.cpp` → `run_d40_math_integration_validation` | `cypha_bench_run --domain-tag d40` |
| D40 profile config | `bench/config/d40_math_integration_profile.json` | manual |
| D40 validation report | `bench/report/tables/d40_math_integration_validation.json` | d40 run |
| Math integration bench runner | `scripts/run_math_integration_bench.ps1` | manual (regenerates **`bench/MATH_INTEGRATION_REPORT.md`**) |
| Validate-all env hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_MATH_INTEGRATION=1`** (d40 when profile exists) | manual |

**Bugfixes (Phase 26 completion):** d40 — throw on subprocess exit **≠ 0** (same gate as d39).

**CI gate (Phase 26 shipped):** **118 CTests** (`ctest -R native_`); +1 smoke: **`native_d40_math_integration_smoke`**. d40 validates math-integration sources + `cyphalm_bench_native --math-integration --intelligence-profile` stdout; **`pending_math_integration`** when incomplete (smoke pass); **`math_integration_ready`** when exit 0 + **`profile_completeness.all_complete`** + finite BPC.

## Phase 27 — shipped (v2.3.27, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Math integration scale gate **d41** | `bench_domains.cpp` → `run_d41_math_integration_scale_validation` | `native_d41_math_integration_scale_smoke` |
| Scale profile | `bench/config/d41_math_integration_scale_profile.json` | 5000 train / 256 eval; ΔBPC / Δκ vs baseline |
| Navigation loss warmup | `cyphalm_config.hpp` `navigation_loss_warmup_steps`; `cyphalm_model.cpp` | ramps 7-stat loss over 200 steps |
| Hybrid blend navigation nudge | `cyphalm_model.cpp` | LSTM/GRIA blend gets profile-guided gradient |
| Profile curriculum re-enabled | `cyphalm_math_integration.cpp` | hardest-first prescan + replay train |
| Cell sweep κ overlay | `cypha_cell_hypothesis_sweep.cpp` `--intelligence-profile` | per-variant κ + completeness |
| LM self-correct REST wiring | `cyphalm_rest_routes.cpp` | `self_correct` auto-enables `epistemic_halt` |
| LM self-correct smoke | `tools/lm_self_correct_smoke.cpp` | `native_lm_self_correct_smoke` |
| Validate-all hook | `scripts/cypha_native_validate_all.ps1` — **`CYPHA_VALIDATE_MATH_INTEGRATION_SCALE=1`** | d41 when profile exists |

**CI gate (Phase 27 shipped):** **121 CTests** (`ctest -R native_`); +1 smoke: **`native_d41_math_integration_scale_smoke`**; +1: **`native_lm_self_correct_smoke`**. d41 records **`delta_bpc`** / **`delta_kappa`** at 5k scale; **`math_integration_scale_ready`** when both subprocesses exit 0 with complete profiles + finite BPC.

## Phase 28 — shipped (v2.3.28, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| CharLSTM navigation loss | `cyphalm_model.cpp` train_step CharLstm branch | `native_navigation_loss_char_lstm_smoke` |
| Adaptive λ from κ | `profile_guided_loss.cpp` `scale_profile_guided_loss_config` | `native_intelligence_profiler_papers` |
| d20/d22 κ ranking | `bench_domains.cpp`, `cypha_cell_hypothesis_sweep.cpp` | `native_d22_cross_smoke` |
| Production math gate **d42** | `run_d42_math_integration_production_validation` | `native_d42_math_integration_production_smoke` |
| Overnight math arm | `scripts/run_d17_overnight.ps1 -MathIntegration` | maintainer |
| Lock stub | `bench/BASELINE_LOCK.json` `math_integration_results` | d42 reads pending/production |
| Validate-all hook | `CYPHA_VALIDATE_MATH_INTEGRATION_PRODUCTION=1` | d42 when profile exists |

**CI gate (Phase 28 shipped):** **123 CTests** (`ctest -R native_`); +1: **`native_d42_math_integration_production_smoke`**; +1: **`native_navigation_loss_char_lstm_smoke`**. d42 **`pending_math_integration_production`** when lock stub; **`math_integration_production_ready`** at 300k tier with complete profiles.

## Phase 29 — shipped (v2.3.29, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Hybrid LSTM logit nav grads | `profile_guided_loss` `d_logit_uniform`; `cyphalm_model.cpp` hybrid `dWy`/`dby` | `native_navigation_loss_hybrid_smoke` |
| CharLSTM logit nav grads | `cyphalm_model.cpp` CharLstm `dby` nudge | `native_navigation_loss_char_lstm_smoke` |
| κ×BPC Pareto ranking | `build_pareto_ranked_variants`, d20/d22 `best_pareto_variant` | `native_d22_cross_smoke` |
| d17-math baseline lock | `cypha_baseline_lock --run d17-math` | `update_baseline_lock.ps1 -Run d17-math` |
| Production math pipeline | `run_production_overnight.ps1 -MathIntegration`, finalize d42 | maintainer |

**CI gate (Phase 29 shipped):** **124 CTests** (`ctest -R native_`); +1: **`native_navigation_loss_hybrid_smoke`**.

## Phase 30 — shipped (v2.3.30, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Direct LSTM weight nav | `char_lstm.cpp` `backward_step(..., logit_nudge)` | nav smokes |
| Pareto default cell winner | `cypha_baseline_lock.cpp`, overnight sweep `--intelligence-profile` | `native_d22_cross_smoke` |
| Math lock gate **d43** | `run_d43_math_integration_lock_validation` | `native_d43_math_integration_lock_smoke` |
| H04 kernel LLR stub | `cyphalm_dif.cpp` field RBF proxy blend | `native_kernel_llm_h04_smoke` |
| Validate-all hook | `CYPHA_VALIDATE_MATH_INTEGRATION_LOCK=1` | d43 when profile exists |

**CI gate (Phase 30 shipped):** **126 CTests** (`ctest -R native_`); +1: **`native_d43_math_integration_lock_smoke`**; +1: **`native_kernel_llm_h04_smoke`**.

## Phase 31 — shipped (v2.3.31, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Nyström kernel in CyphaDIF | `cyphalm_dif.cpp` `KernelMemory` route/train/snapshot | `native_kernel_llm_h04_smoke` |
| κ trajectory λ schedule | `profile_guided_loss.cpp` `scale_profile_guided_loss_from_trajectory` | `native_intelligence_profiler_papers` |
| Math preset trajectory | `cyphalm_math_integration.cpp` `use_kappa_trajectory_lambdas` | d40–d43 reports |
| Kernel CyphaLM gate **d44** | `run_d44_kernel_nystrom_cyphalm_validation` | `native_d44_kernel_nystrom_cyphalm_smoke` |
| Validate-all hook | `CYPHA_VALIDATE_KERNEL_NYSTROM_CYPHALM=1` | d44 when profile exists |

**CI gate (Phase 31 shipped):** **127 CTests** (`ctest -R native_`); +1: **`native_d44_kernel_nystrom_cyphalm_smoke`**.

## Phase 32 — shipped (v2.3.32, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Per-stat deviation λ | `scale_profile_guided_loss_by_stat_deviation` | `native_intelligence_profiler_papers` |
| Unified λ resolver | `resolve_adaptive_profile_guided_config` | `cyphalm_model.cpp` train + export |
| Enriched math export | `export_math_integration_report` — `stat_deltas`, `kappa_trajectory`, `navigation_config` | d40–d45 |
| Cell sweep Pareto lock | `cypha_baseline_lock` `--intelligence-profile` + full Pareto object | d43 |
| Variant gate fix | `kExpectedCellSweepVariants` = 25 | d38 |
| Lipschitz grad symmetry | `compute_profile_guided_loss_grad` | nav smokes |
| Per-stat gate **d45** | `run_d45_per_stat_navigation_validation` | `native_d45_per_stat_navigation_smoke` |
| Validate-all hook | `CYPHA_VALIDATE_PER_STAT_NAVIGATION=1` | d45 when profile exists |

**CI gate (Phase 32 shipped):** **128 CTests** (`ctest -R native_`); +1: **`native_d45_per_stat_navigation_smoke`**.

## Phase 33 — shipped (v2.3.33, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| τ forget gate (Paper IV) | `cyphalm_model.cpp` `hybrid_forget_gate_scale` | `native_tau_forget_gate_smoke` |
| Kernel LLR in math preset | `cyphalm_math_integration.cpp` | d46 export |
| Span tune | `per_stat_deviation_span = 1.0` | d46 |
| Math stack gate **d46** | `run_d46_math_stack_upgrade_validation` | `native_d46_math_stack_upgrade_smoke` |
| Config JSON | `cyphalm_config.cpp` nav + kernel keys | checkpoint profiles |

**CI gate (Phase 33 shipped):** **130 CTests** (`ctest -R native_`); +1: **`native_d46_math_stack_upgrade_smoke`**; +1: **`native_tau_forget_gate_smoke`**.

## Phase 34 — shipped (v2.3.34, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| κ ceiling λ scheduler | `profile_guided_loss.cpp` `use_kappa_ceiling_lambdas` | `intelligence_profiler_papers` |
| LSTM hidden D_eff nudge | `cyphalm_model.cpp` + `char_lstm.cpp` `hidden_nudge` | hybrid train backprop |
| Span CLI | `cyphalm_bench_native --per-stat-deviation-span` | d47 ablation |
| Math preset Phase 34 | `cyphalm_math_integration.cpp` | export `navigation_config` |
| Span ablation gate **d47** | `run_d47_span_ablation_validation` | `native_d47_span_ablation_smoke` |
| Validate-all hook | `CYPHA_VALIDATE_SPAN_ABLATION=1` | d47 when profile exists |

**CI gate (Phase 34 shipped):** **131 CTests** (`ctest -R native_`); +1: **`native_d47_span_ablation_smoke`**.

## Phase 35 — shipped (v2.3.35, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| Eigenvalue D_eff PR | `measurers.cpp` `CovarianceEigenvalue` | `intelligence_profiler_papers` |
| Stronger κ ceiling | `kappa_ceiling_strength` / `min_scale` | math preset + export |
| r_eu forget gate | `hybrid_forget_gate_scale` + `use_reu_forget_gate` | hybrid forward |
| κ target CLI | `--kappa-lambda-target`, `--kappa-ceiling-strength` | d48 ablation @ 5k |
| κ ceiling gate **d48** | `run_d48_kappa_ceiling_ablation_validation` | `native_d48_kappa_ceiling_ablation_smoke` |
| Validate-all hook | `CYPHA_VALIDATE_KAPPA_CEILING_ABLATION=1` | d48 when profile exists |

**CI gate (Phase 35 shipped):** **132 CTests** (`ctest -R native_`); +1: **`native_d48_kappa_ceiling_ablation_smoke`**.

## Phase 36 — shipped (v2.3.36, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| κ trajectory ceiling | `scale_profile_guided_loss_from_trajectory` | `intelligence_profiler_papers` |
| κ excess grad nudge (no margin) | `kappa_excess_grad_nudge` + model backprop | hybrid train |
| Soft r_eu forget | `reu_forget_gate_blend` | `hybrid_forget_gate_scale` |
| Ceiling grid CLI | `--kappa-ceiling-min-scale`, opt-in flags | d49 @ 5k |
| Joint gate **d49** | `run_d49_ceiling_grid_joint_validation` | `native_d49_ceiling_grid_joint_smoke` |

**CI gate (Phase 36 shipped):** **133 CTests** (`ctest -R native_`); +1: **`native_d49_ceiling_grid_joint_smoke`**.

## Phase 37 — shipped (v2.3.37, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|-------------|
| κ excess grad margin | `profile_guided_loss.cpp` `kappa_excess_grad_nudge(..., margin)` | activates when κ > target + margin |
| Separate excess nudge flag | `cyphalm_config.hpp` `use_kappa_excess_grad_nudge` | math preset + `--disable-kappa-excess-grad-nudge` |
| Margin / scale CLI | `cyphalm_bench_native --kappa-excess-grad-margin`, `--kappa-excess-grad-scale` | d50 @ 5k |
| Math preset Phase 37 | `cyphalm_math_integration.cpp` — margin **0.02**, scale **0.35** | export `navigation_config` |
| Pinned bench seed | `cyphalm_bench_native --bench-seed N`, `CYPHA_BENCH_SEED` | JSON `bench_seed` in bench output |
| Joint lock gate **d50** | `run_d50_math_joint_lock_validation` | `native_d50_math_joint_lock_smoke` |
| Lock repro | vs `bench/BASELINE_LOCK.json` → `math_integration_results` | BPC ±0.025, κ ±0.03 @ seed **42** |
| Validate-all hook | `CYPHA_VALIDATE_MATH_JOINT_LOCK=1` | d50 when profile exists |

**CI gate (Phase 37 shipped):** **134 CTests** (`ctest -R native_`); +1: **`native_d50_math_joint_lock_smoke`**.

## Phase 38 — prep (v2.3.38, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Ablation winners in preset | `cyphalm_math_integration.cpp` — κ target **0.83**, min_scale **0.40** | d48/d49 winners merged |
| κ-adaptive kernel blend | `scale_kernel_blend_from_kappa` + `CyphaDIF::set_runtime_kernel_blend` | hybrid train routing |
| Opt-in lever gate **d51** | `run_d51_opt_in_lever_joint_validation` | `native_d51_opt_in_lever_joint_smoke` |

**CI gate (Phase 38 shipped):** **135 CTests** (`ctest -R native_`); +1: **`native_d51_opt_in_lever_joint_smoke`**.

## Phase 39 — shipped (v2.3.39, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| r_eu forget in preset | `cyphalm_math_integration.cpp` `use_reu_forget_gate = true` | d51 winner merged |
| Pinned seed d49 | `kMathIntegrationBenchSeed = 42` in ceiling grid | d49 joint refresh |
| Preset ship lock **d52** | `run_d52_preset_ship_lock_validation` | `native_d52_preset_ship_lock_smoke` |
| Validate-all hook | `CYPHA_VALIDATE_PRESET_SHIP_LOCK=1` | d52 when profile exists |

**CI gate (Phase 39 shipped):** **136 CTests** (`ctest -R native_`); +1: **`native_d52_preset_ship_lock_smoke`**.

## Phase 40 — prep (v2.3.40, 2026-06-23)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| κ-aware navigation warmup | `scale_navigation_warmup_from_kappa` | damp nav loss ramp when κ high |
| Production preset ship lock **d53** | `run_d53_production_preset_ship_lock_validation` | `native_d53_production_preset_ship_lock_smoke` |
| Production tier gate | lock `n_train >= 300000` + `status=production` | `pending_*` @ 5k until overnight fill |
| Finalize hook | `finalize_production_overnight.ps1` d53 after d42 | maintainer workflow |
| Validate-all hook | `CYPHA_VALIDATE_PRODUCTION_PRESET_SHIP_LOCK=1` | d53 when profile exists |

**CI gate (Phase 40 prep):** **137 CTests** (`ctest -R native_`); +1: **`native_d53_production_preset_ship_lock_smoke`**.

## Phase 41 — prep (v2.3.41, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Nav warmup CLI ablation | `cyphalm_bench_native --kappa-navigation-warmup-strength/floor`, `--disable-kappa-navigation-warmup` | d55 grid |
| Production math certificate **d54** | `run_d54_production_math_certificate_validation` | `native_d54_production_math_certificate_smoke` |
| Nav warmup grid **d55** | `run_d55_nav_warmup_grid_joint_validation` | `native_d55_nav_warmup_grid_joint_smoke` |
| Hybrid BPC gate @ 300k | lock math BPC ≤ `overnight_results.bpc` + **0.05** | d54 production tier |
| Finalize hook | `finalize_production_overnight.ps1` d54 after d53 | maintainer workflow |
| Validate-all hooks | `CYPHA_VALIDATE_PRODUCTION_MATH_CERTIFICATE=1`, `CYPHA_VALIDATE_NAV_WARMUP_GRID_JOINT=1` | d54/d55 when profiles exist |

**CI gate (Phase 41 prep):** **139 CTests** (`ctest -R native_`); +2: **`native_d54_production_math_certificate_smoke`**, **`native_d55_nav_warmup_grid_joint_smoke`**.

## Phase 42 — prep (v2.3.42, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Cell sweep math preset | `cypha_cell_hypothesis_sweep --math-integration` | applies full navigation preset per variant |
| Overnight / lock parity | `run_d17_overnight.ps1`, `cypha_baseline_lock` cell-sweep | `--math-integration` when `-MathIntegration` |
| Cell sweep math joint **d56** | `run_d56_cell_sweep_math_integration_validation` | `native_d56_cell_sweep_math_integration_smoke` |
| Lock field | `cell_sweep_results.math_integration_enabled` | set when math sweep used |
| Validate-all hook | `CYPHA_VALIDATE_CELL_SWEEP_MATH_INTEGRATION=1` | d56 when profile exists |

**CI gate (Phase 42 prep):** **140 CTests** (`ctest -R native_`); +1: **`native_d56_cell_sweep_math_integration_smoke`**.

## Phase 43 — prep (v2.3.43, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Production cell sweep certificate **d57** | `run_d57_production_cell_sweep_math_certificate_validation` | `native_d57_production_cell_sweep_math_certificate_smoke` |
| Finalize hook | `finalize_production_overnight.ps1` d56 + d57 after d54 | maintainer workflow |
| Lock script | `update_baseline_lock.ps1 -MathIntegration` | sets `CYPHA_OVERNIGHT_MATH_INTEGRATION=1`; `-Run all` also runs `d17-math` |
| Hybrid BPC gate | cell `b2_bpc` ≤ `overnight_results.bpc` + **0.05** @ 300k math tier | d57 production tier |
| Validate-all hook | `CYPHA_VALIDATE_PRODUCTION_CELL_SWEEP_MATH_CERTIFICATE=1` | d57 when profile exists |

**CI gate (Phase 43 prep):** **141 CTests** (`ctest -R native_`); +1: **`native_d57_production_cell_sweep_math_certificate_smoke`**.

## Phase 44 — prep (v2.3.44, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Cell sweep **best_pareto_variant** export | `cypha_cell_hypothesis_sweep` stdout + manifest | lock fill for Pareto winner |
| Overnight all **-MathIntegration** cell sweep | `run_overnight_all.ps1` | explicit `-MathIntegration` on cell-sweep lock |
| Unified math complete **d58** | `run_d58_production_overnight_math_complete_validation` | `native_d58_production_overnight_math_complete_smoke` |
| Cross-checks | math/cell/overnight tier alignment, hybrid BPC gates (d54+d57), Pareto present | production tier |
| Finalize hook | `finalize_production_overnight.ps1` d58 after d57 | maintainer workflow |
| Validate-all hook | `CYPHA_VALIDATE_PRODUCTION_OVERNIGHT_MATH_COMPLETE=1` | d58 when profile exists |

**CI gate (Phase 44 prep):** **142 CTests** (`ctest -R native_`); +1: **`native_d58_production_overnight_math_complete_smoke`**.

## Phase 45 — prep (v2.3.45, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Kernel blend floor CLI | `cyphalm_bench_native` `--kappa-kernel-blend-floor`, `--disable-kappa-kernel-blend-scale` | subprocess ablation |
| Kernel blend floor grid **d59** | `run_d59_kernel_blend_floor_grid_joint_validation` | `native_d59_kernel_blend_floor_grid_joint_smoke` |
| Grid floors | {0.05, 0.08, 0.12} @ 5k seed 42 | preset ship **0.08** |
| Validate-all hook | `CYPHA_VALIDATE_KERNEL_BLEND_FLOOR_GRID_JOINT=1` | d59 when profile exists |

**CI gate (Phase 45 prep):** **143 CTests** (`ctest -R native_`); +1: **`native_d59_kernel_blend_floor_grid_joint_smoke`**.

## Phase 46 — prep (v2.3.46, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Excess grad margin subprocess CLI | `run_math_integration_bench_subprocess` margin/scale/nudge | d60 grid |
| Excess grad margin grid **d60** | `run_d60_excess_grad_margin_grid_joint_validation` | `native_d60_excess_grad_margin_grid_joint_smoke` |
| Grid margins | {0.01, 0.02, 0.04} @ 5k seed 42 | preset ship **0.02** / scale **0.35** |
| Validate-all hook | `CYPHA_VALIDATE_EXCESS_GRAD_MARGIN_GRID_JOINT=1` | d60 when profile exists |

**CI gate (Phase 46 prep):** **144 CTests** (`ctest -R native_`); +1: **`native_d60_excess_grad_margin_grid_joint_smoke`**.

## Phase 47 — prep (v2.3.47, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Excess grad scale grid **d61** | `run_d61_excess_grad_scale_grid_joint_validation` | `native_d61_excess_grad_scale_grid_joint_smoke` |
| Grid scales | {0.25, 0.35, 0.50} @ 5k seed 42 | preset ship **0.35** |
| Validate-all hook | `CYPHA_VALIDATE_EXCESS_GRAD_SCALE_GRID_JOINT=1` | d61 when profile exists |

**CI gate (Phase 47 prep):** **145 CTests** (`ctest -R native_`); +1: **`native_d61_excess_grad_scale_grid_joint_smoke`**.

## Phase 48 — prep (v2.3.48, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Stack complete **d62** | `run_d62_math_ablation_stack_complete_validation` | `native_d62_math_ablation_stack_complete_smoke` |
| Table audit | d47, d48, d49, d50, d51, d52, d55, d59, d60, d61 | `*_joint_ready` on all → `math_ablation_stack_complete_ready` |
| Subprocess joint | baseline + math @ 5k seed 42 | fallback `math_ablation_stack_joint_ready` |
| Finalize hook | `finalize_production_overnight.ps1` d59–d62 after d58 | maintainer workflow |
| Validate-all hook | `CYPHA_VALIDATE_MATH_ABLATION_STACK_COMPLETE=1` | d62 when profile exists |

**CI gate (Phase 48 prep):** **146 CTests** (`ctest -R native_`); +1: **`native_d62_math_ablation_stack_complete_smoke`**.

## Phase 49 — prep (v2.3.49, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| r_eu forget blend CLI | `cyphalm_bench_native` `--reu-forget-gate-blend` | hybrid forget gate scale |
| r_eu forget blend grid **d63** | `run_d63_reu_forget_blend_grid_joint_validation` | `native_d63_reu_forget_blend_grid_joint_smoke` |
| Grid blends | {0.0, 0.25, 0.50} @ 5k seed 42 | preset ship **0.25** |
| Validate-all hook | `CYPHA_VALIDATE_REU_FORGET_BLEND_GRID_JOINT=1` | d63 when profile exists |

**CI gate (Phase 49 prep):** **147 CTests** (`ctest -R native_`); +1: **`native_d63_reu_forget_blend_grid_joint_smoke`**.

## Phase 50 — prep (v2.3.50, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| κ trajectory window CLI | `cyphalm_bench_native` `--kappa-trajectory-window` | EMA window for trajectory λ |
| Trajectory window grid **d64** | `run_d64_kappa_trajectory_window_grid_joint_validation` | `native_d64_kappa_trajectory_window_grid_joint_smoke` |
| Grid windows | {8, 16, 32} @ 5k seed 42 | preset ship **16** |
| d62 stack refresh | audits **12** tables (adds d63+d64) | `math_ablation_stack_complete_ready` |
| Validate-all hook | `CYPHA_VALIDATE_KAPPA_TRAJECTORY_WINDOW_GRID_JOINT=1` | d64 when profile exists |

**CI gate (Phase 50 prep):** **148 CTests** (`ctest -R native_`); +1: **`native_d64_kappa_trajectory_window_grid_joint_smoke`**.

## Phase 51 — prep (v2.3.51, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Navigation loss warmup CLI | `cyphalm_bench_native` `--navigation-loss-warmup-steps` | 7-stat loss ramp |
| Nav loss warmup grid **d65** | `run_d65_navigation_loss_warmup_grid_joint_validation` | `native_d65_navigation_loss_warmup_grid_joint_smoke` |
| Grid steps | {100, 200, 400} @ 5k seed 42 | preset ship **200** |
| Validate-all hook | `CYPHA_VALIDATE_NAVIGATION_LOSS_WARMUP_GRID_JOINT=1` | d65 when profile exists |

**CI gate (Phase 51 prep):** **149 CTests** (`ctest -R native_`); +1: **`native_d65_navigation_loss_warmup_grid_joint_smoke`**.

## Phase 52 — prep (v2.3.52, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Free energy beta CLI | `cyphalm_bench_native` `--free-energy-beta` | epistemic variance penalty |
| Free energy beta grid **d66** | `run_d66_free_energy_beta_grid_joint_validation` | `native_d66_free_energy_beta_grid_joint_smoke` |
| Grid betas | {0.005, 0.01, 0.02} @ 5k seed 42 | preset ship **0.01** |
| d62 stack refresh | audits **14** tables (adds d65+d66) | `math_ablation_stack_complete_ready` |
| Validate-all hook | `CYPHA_VALIDATE_FREE_ENERGY_BETA_GRID_JOINT=1` | d66 when profile exists |

**CI gate (Phase 52 prep):** **150 CTests** (`ctest -R native_`); +1: **`native_d66_free_energy_beta_grid_joint_smoke`**.

## Phase 53 — prep (v2.3.53, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Base kernel blend CLI | `cyphalm_bench_native` `--kernel-blend` | DIF/kernel LLR mix anchor |
| Kernel blend grid **d67** | `run_d67_kernel_blend_grid_joint_validation` | `native_d67_kernel_blend_grid_joint_smoke` |
| Grid blends | {0.15, 0.25, 0.40} @ 5k seed 42 | preset ship **0.25** (vs d59 κ floor) |
| Validate-all hook | `CYPHA_VALIDATE_KERNEL_BLEND_GRID_JOINT=1` | d67 when profile exists |

**CI gate (Phase 53 prep):** **151 CTests** (`ctest -R native_`); +1: **`native_d67_kernel_blend_grid_joint_smoke`**.

## Phase 54 — prep (v2.3.54, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Nyström kernel_m CLI | `cyphalm_bench_native` `--kernel-m` | landmark count |
| Kernel_m grid **d68** | `run_d68_kernel_m_grid_joint_validation` | `native_d68_kernel_m_grid_joint_smoke` |
| Grid kernel_m | {32, 64, 128} @ 5k seed 42 | preset ship **64** |
| d62 stack refresh | audits **16** tables (adds d67+d68) | `math_ablation_stack_complete_ready` |
| Validate-all hook | `CYPHA_VALIDATE_KERNEL_M_GRID_JOINT=1` | d68 when profile exists |

**CI gate (Phase 54 prep):** **152 CTests** (`ctest -R native_`); +1: **`native_d68_kernel_m_grid_joint_smoke`**.

## Phase 55 — prep (v2.3.55, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| GRIA/LSTM hybrid blend logit CLI | `cyphalm_bench_native` `--hybrid-blend-logit` | sigmoid mix prior |
| Hybrid blend logit grid **d69** | `run_d69_hybrid_blend_logit_grid_joint_validation` | `native_d69_hybrid_blend_logit_grid_joint_smoke` |
| Grid hybrid_blend_logit | {0.0, 0.5, 1.0} @ 5k seed 42 | preset ship **0.5** |
| Validate-all hook | `CYPHA_VALIDATE_HYBRID_BLEND_LOGIT_GRID_JOINT=1` | d69 when profile exists |

**CI gate (Phase 55 prep):** **153 CTests** (`ctest -R native_`); +1: **`native_d69_hybrid_blend_logit_grid_joint_smoke`**.

**Validated @ 5k:** `hybrid_blend_logit_grid_joint_ready` — ΔBPC **−0.017** (best **−0.0174** @ logit 0.0), Δκ **+0.044** (joint ✓).

## Phase 56 — prep (v2.3.56, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| H12 MDL forget max norm CLI | `cyphalm_bench_native` `--mdl-forget-max-norm` | L2 cap via `mdl_forget_project` |
| MDL forget max norm grid **d70** | `run_d70_mdl_forget_max_norm_grid_joint_validation` | `native_d70_mdl_forget_max_norm_grid_joint_smoke` |
| Grid mdl_forget_max_norm | {2.0, 4.0, 8.0} @ 5k seed 42 | preset ship **4.0** |
| d62 stack refresh | audits **18** tables (adds d69+d70) | `math_ablation_stack_complete_ready` |
| Validate-all hook | `CYPHA_VALIDATE_MDL_FORGET_MAX_NORM_GRID_JOINT=1` | d70 when profile exists |

**CI gate (Phase 56 prep):** **154 CTests** (`ctest -R native_`); +1: **`native_d70_mdl_forget_max_norm_grid_joint_smoke`**.

**Validated @ 5k:** `mdl_forget_max_norm_grid_joint_ready` — grid tied ΔBPC **−0.017**, Δκ **+0.044** (joint ✓). d62 **18/18** tables joint-ready.

## Phase 57 — prep (v2.3.57, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Kernel LLR lr scale CLI | `cyphalm_bench_native` `--kernel-lr-scale` | Nyström memory update rate |
| Kernel lr scale grid **d71** | `run_d71_kernel_lr_scale_grid_joint_validation` | `native_d71_kernel_lr_scale_grid_joint_smoke` |
| GRIA alpha_init CLI | `cyphalm_bench_native` `--alpha-init` | Dirichlet prior |
| Alpha init grid **d72** | `run_d72_alpha_init_grid_joint_validation` | `native_d72_alpha_init_grid_joint_smoke` |
| Grid kernel_lr_scale | {0.5, 1.0, 2.0} @ 5k seed 42 | preset ship **1.0** |
| Grid alpha_init | {0.3, 0.5, 0.7} @ 5k seed 42 | preset ship **0.5** |
| Validate-all hooks | `CYPHA_VALIDATE_KERNEL_LR_SCALE_GRID_JOINT=1`, `CYPHA_VALIDATE_ALPHA_INIT_GRID_JOINT=1` | d71/d72 when profiles exist |

**CI gate (Phase 57 prep):** **156 CTests** (`ctest -R native_`); +2: **`native_d71_*`**, **`native_d72_*`**.

**Validated @ 5k:** d71 grid tied ΔBPC **−0.017**, Δκ **+0.044**; d72 best **α=0.3** ΔBPC **−0.0174** (preset **0.5** retained).

## Phase 58 — prep (v2.3.58, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Navigation hybrid blend lr CLI | `cyphalm_bench_native` `--hybrid-blend-lr` | κ-navigation logit nudge rate |
| Hybrid blend lr grid **d73** | `run_d73_hybrid_blend_lr_grid_joint_validation` | `native_d73_hybrid_blend_lr_grid_joint_smoke` |
| Grid hybrid_blend_lr | {0.005, 0.01, 0.02} @ 5k seed 42 | preset ship **0.01** |
| d62 stack refresh | audits **21** tables (adds d71–d73) | `math_ablation_stack_complete_ready` |
| Validate-all hook | `CYPHA_VALIDATE_HYBRID_BLEND_LR_GRID_JOINT=1` | d73 when profile exists |

**CI gate (Phase 58 prep):** **157 CTests** (`ctest -R native_`); +1: **`native_d73_hybrid_blend_lr_grid_joint_smoke`**.

**Validated @ 5k:** d73 `hybrid_blend_lr_grid_joint_ready` — lr {0.005, 0.01, 0.02} ΔBPC **−0.014** to **−0.019**, Δκ **+0.044** (joint ✓). d62 **21/21** tables joint-ready.

## Phase 59 — prep (v2.3.59, 2026-06-19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| DIF n_experts CLI | `cyphalm_bench_native` `--n-experts` | OOD branching width |
| n_experts grid **d74** | `run_d74_n_experts_grid_joint_validation` | `native_d74_n_experts_grid_joint_smoke` |
| Priority replay slots CLI | `cyphalm_bench_native` `--max-memory-slots` | compressive memory capacity |
| max_memory_slots grid **d75** | `run_d75_max_memory_slots_grid_joint_validation` | `native_d75_max_memory_slots_grid_joint_smoke` |
| Compressive interval CLI | `cyphalm_bench_native` `--compress-interval` | SSM/memory compression cadence |
| compress_interval grid **d76** | `run_d76_compress_interval_grid_joint_validation` | `native_d76_compress_interval_grid_joint_smoke` |
| Grid n_experts | {4, 8, 12} @ 5k seed 42 | preset ship **8** |
| Grid max_memory_slots | {128, 256, 512} @ 5k seed 42 | preset ship **256** |
| Grid compress_interval | {8, 16, 32} @ 5k seed 42 | preset ship **16** |
| d62 stack refresh | audits **24** tables (adds d74–d76) | `math_ablation_stack_complete_ready` |
| Validate-all hooks | `CYPHA_VALIDATE_N_EXPERTS_GRID_JOINT=1`, etc. | d74–d76 when profiles exist |

**CI gate (Phase 59 prep):** **160 CTests** (`ctest -R native_`); +3: **`native_d74_*`**, **`native_d75_*`**, **`native_d76_*`**.

**Validated @ 5k:** all three structural grids joint-ready (ΔBPC **−0.017**, Δκ **+0.044**); preset ship cells retained. **All `apply_math_integration_preset` tunables now have CLI + grid ablation.**

## Still planned

- **Phase 60 production math certificate** — d53–d58 `production_*_ready` via **`scripts/run_production_overnight.ps1 -MathIntegration`**
- **D17 300k + 25-variant production overnight** — maintainer workflow; fill **`bench/BASELINE_LOCK.json`** Pareto cell winner via **`update_baseline_lock.ps1 -Run cell-sweep -Production`**
- **GitHub Release publish** — preview with **`scripts/publish_release.ps1 -DryRun`**; actual **`gh release create`** still requires **`gh auth login`**
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
pwsh -File scripts/run_overnight_all.ps1 -Medium          # Phase 12: 5k train, real WikiText/gutenberg
cypha_bench_run --domain-tag d26                        # Phase 12: medium overnight lock validation (shipped)
pwsh -File scripts/validate_baseline_lock.ps1             # Phase 12: schema + d17 pin check
pwsh -File scripts/validate_baseline_lock.ps1 -Strict     # Phase 12: reject fast_smoke-only overnight
baseline_lock_validate --lock-file bench/BASELINE_LOCK.json  # Phase 12: C++ validator CLI
pwsh -File scripts/publish_release.ps1 -Tag v2.3.12 -DryRun  # Phase 12: notes preview, no gh
ctest --test-dir native/build -R "native_d26|native_baseline_lock_validate" --output-on-failure  # Phase 12
pwsh -File scripts/run_production_overnight.ps1             # Phase 13: 300k production overnight (maintainer)
pwsh -File scripts/run_overnight_all.ps1 -Production          # Phase 13: 300k train, real WikiText/gutenberg
cypha_bench_run --domain-tag d27                            # Phase 13: production overnight lock validation (shipped)
pwsh -File scripts/validate_baseline_lock.ps1 -Production   # Phase 13: require production/completed @ 300k
baseline_lock_validate --lock-file bench/BASELINE_LOCK.json --production  # Phase 13: C++ production validator
ctest --test-dir native/build -R "native_d27" --output-on-failure  # Phase 13
pwsh -File scripts/finalize_production_overnight.ps1              # Phase 14: post-overnight validate + d27/d28
cypha_bench_run --domain-tag d28                                    # Phase 14: unified overnight completion validation (shipped)
ctest --test-dir native/build -R "native_d28" --output-on-failure  # Phase 14
cypha_bench_run --domain-tag d29                                    # Phase 15: release readiness validation (shipped)
pwsh -File scripts/watch_production_overnight.ps1 -Once             # Phase 15: production overnight watcher snapshot
pwsh -File scripts/commit_production_lock.ps1 -DryRun               # Phase 15: post-overnight lock commit preview
$env:CYPHA_VALIDATE_RELEASE_READINESS = "1"                         # Phase 15: d29 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d29" --output-on-failure  # Phase 15
cypha_bench_run --domain-tag d30                                    # Phase 16: artifact path hygiene validation (shipped)
pwsh -File scripts/migrate_legacy_results.ps1 -DryRun               # Phase 16: preview legacy results/ migration
$env:CYPHA_VALIDATE_ARTIFACT_HYGIENE = "1"                          # Phase 16: d30 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d30" --output-on-failure  # Phase 16
cypha_bench_run --domain-tag d31                                    # Phase 17: post-overnight pipeline validation (shipped)
pwsh -File scripts/poll_and_finalize_overnight.ps1 -Once            # Phase 17: poll snapshot (exit 1 if still running)
pwsh -File scripts/cleanup_legacy_results.ps1 -DryRun               # Phase 17: preview migrate + remove legacy results/
$env:CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE = "1"                   # Phase 17: d31 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d31" --output-on-failure  # Phase 17
cypha_bench_run --domain-tag d32                                    # Phase 18: production complete validation (shipped)
pwsh -File scripts\validate_production_complete.ps1 -AllowPending   # Phase 18: smoke when lock below 300k
pwsh -File scripts\start_poll_finalize_background.ps1               # Phase 18: detached poll → finalize
$env:CYPHA_VALIDATE_PRODUCTION_COMPLETE = "1"                       # Phase 18: d32 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d32" --output-on-failure  # Phase 18
cypha_bench_run --domain-tag d33                                    # Phase 19: release publish validation (shipped)
pwsh -File scripts\verify_release_publish.ps1                       # Phase 19: production complete + d33 + publish -DryRun
$env:CYPHA_VALIDATE_RELEASE_PUBLISH = "1"                           # Phase 19: d33 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d33" --output-on-failure  # Phase 19
cypha_bench_run --domain-tag d34                                    # Phase 20: repo smoke hygiene validation (shipped)
pwsh -File scripts\cleanup_repo_smoke_artifacts.ps1 -DryRun         # Phase 20: preview repo-root smoke JSON cleanup
$env:CYPHA_VALIDATE_REPO_SMOKE_HYGIENE = "1"                        # Phase 20: d34 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d34" --output-on-failure  # Phase 20
cypha_bench_run --domain-tag d35                                    # Phase 21: lock commit pipeline validation (shipped)
pwsh -File scripts\verify_production_pipeline.ps1 -AllowPending     # Phase 21: unified production pipeline smoke
$env:CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE = "1"                      # Phase 21: d35 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d35" --output-on-failure  # Phase 21
cypha_bench_run --domain-tag d36                                    # Phase 22: production pipeline E2E validation (shipped)
pwsh -File scripts\run_post_overnight.ps1 -AllowPending             # Phase 22: poll/finalize + production verify
$env:CYPHA_VALIDATE_PIPELINE_E2E = "1"                              # Phase 22: d36 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d36" --output-on-failure  # Phase 22
cypha_bench_run --domain-tag d37                                    # Phase 23: overnight lock refresh validation (shipped)
pwsh -File scripts\migrate_inflight_overnight_artifacts.ps1 -DryRun # Phase 23: preview in-flight results/ migration
$env:CYPHA_VALIDATE_LOCK_REFRESH = "1"                              # Phase 23: d37 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d37" --output-on-failure  # Phase 23
cypha_bench_run --domain-tag d38                                    # Phase 24: overnight completion certificate (prep)
pwsh -File scripts\poll_and_finalize_overnight.ps1 -AutoCommit       # Phase 24: auto-commit when n_train >= 300k
$env:CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE = "1"                     # Phase 24: d38 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d38" --output-on-failure  # Phase 24
cypha_bench_run --domain-tag d39                                    # Phase 25: intelligence monitor validation (shipped)
curl "http://127.0.0.1:8099/intelligence/report?source=live"        # Phase 25: live session profiler report
$env:CYPHA_VALIDATE_INTELLIGENCE_MONITOR = "1"                      # Phase 25: d39 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d39|native_intelligence_lm_monitor" --output-on-failure  # Phase 25
cypha_bench_run --domain-tag d40                                    # Phase 26: math integration validation (shipped)
pwsh -File scripts/run_math_integration_bench.ps1                   # Phase 26: regenerate MATH_INTEGRATION_REPORT.md
$env:CYPHA_VALIDATE_MATH_INTEGRATION = "1"                          # Phase 26: d40 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d40" --output-on-failure  # Phase 26
cypha_bench_run --domain-tag d41                                    # Phase 27: math integration scale validation (shipped)
$env:CYPHA_VALIDATE_MATH_INTEGRATION_SCALE = "1"                    # Phase 27: d41 in cypha_native_validate_all.ps1
cypha_cell_hypothesis_sweep --smoke --intelligence-profile          # Phase 27: per-variant κ overlay
cypha_bench_run --domain-tag d42                                    # Phase 28: math integration production gate (shipped)
pwsh -File scripts/run_d17_overnight.ps1 -MathIntegration         # Phase 28: overnight with navigation loss
$env:CYPHA_VALIDATE_MATH_INTEGRATION_PRODUCTION = "1"             # Phase 28: d42 in cypha_native_validate_all.ps1
ctest --test-dir native/build -R "native_d41|native_d42|native_navigation_loss_char_lstm" --output-on-failure  # Phase 28
bash scripts/ci_federated_tls_linux.sh   # optional TLS smoke (skip without OpenSSL)
curl http://127.0.0.1:8099/intelligence/report
```
