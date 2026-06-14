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

## Phase 19 — prep (v2.3.19)

| Component | Path | CTest / bench |
|-----------|------|---------------|
| Bench domain **d33** release publish validation | `bench_domains.cpp` → `run_d33_release_publish_validation` | `cypha_bench_run --domain-tag d33` |
| D33 profile config | `bench/config/d33_release_publish_profile.json` | manual |
| D33 validation report | `bench/report/tables/d33_release_publish_validation.json` | d33 run |
| Release publish smoke gate | `scripts/verify_release_publish.ps1` — production complete + d33 + `publish_release.ps1 -DryRun` | manual |
| Poll BuildDir auto-detect | `poll_and_finalize_overnight.ps1`, `start_poll_finalize_background.ps1` | manual |
| Release notes v2.3.19 template | `scripts/create_release_notes.ps1` | manual |

**CI gate (Phase 19 prep):** **110 CTests** today; **111** when **`native_d33_release_publish_smoke`** merges (+1). d33 validates publish script presence + production/overnight-complete tiers; **`pending_release_publish`** when **`n_train < 300000`** (smoke pass); **`release_publish_ready`** when **≥ 300k** and all gates pass.

## Still planned

- **D17 300k + 28-variant production overnight** — **in progress** (maintainer workflow via **`scripts/run_production_overnight.ps1`** → **`finalize_production_overnight.ps1`** → **`commit_production_lock.ps1`**; monitor with **`watch_production_overnight.ps1`**); **`-Medium`** for 5k real-corpus smoke; fill **`bench/BASELINE_LOCK.json`** via **`scripts/update_baseline_lock.ps1 -Run all -Production`**
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
cypha_bench_run --domain-tag d33                                    # Phase 19: release publish validation (prep)
pwsh -File scripts\verify_release_publish.ps1                       # Phase 19: production complete + d33 + publish -DryRun
bash scripts/ci_federated_tls_linux.sh   # optional TLS smoke (skip without OpenSSL)
curl http://127.0.0.1:8099/intelligence/report
```
