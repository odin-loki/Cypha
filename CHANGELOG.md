# Changelog

All notable changes to Cypha are recorded here. The project follows a
**milestone** release model: each entry corresponds to a named engineering
milestone or a significant self-contained change.

---

## [Unreleased]

### Removed
- **`Cypha Possible Upgrades/`** root folder — content backported to [`docs/research/upgrades/`](docs/research/upgrades/README.md).
- **Python runtime decommissioned (P7):** `Cypha.py`, `cypha_studio/`, `cypha_core/`, `cypha_accel/`, `bench/` (Python package removed), `cypha_lm/`, and related packages removed from the product path. Native C++ (`cypha_core` library, `cypha_rest`, `cypha_qt_shell`, `cypha_bench_run`) is the sole runtime.
- **pytest CI gate:** ~274 pytest tests no longer run in CI; validation is CTest-only.
- **`run_all.py`:** replaced by `cypha_bench_run`.
- **CUDA CI jobs:** **`windows_cuda_msvc`** and **`linux_cuda`** removed from `.github/workflows/ci.yml`; CUDA remains an optional local build.

### Changed
- **CI gate:** native-only — **101 CTests** (`ctest -R native_`) across **two blocking jobs** (`build_and_test`, `mingw_cross`); optional **`federated_tls`** job (**`-DCYPHA_ENABLE_OPENSSL=ON`**, `native_federated_tls_smoke`, `continue-on-error`; local mirrors **`scripts/ci_federated_tls_linux.sh`**, **`scripts/ci_federated_tls_windows.ps1`**).
- **`CYPHA_ACCEL_GPU_MIN_BATCH_ROWS` default:** **1** (was 16) — CUDA used for all batch sizes n≥1 when a GPU is available.
- **Docs:** README, CONTRIBUTING, docs hub, NATIVE_QUICKSTART, PORT_FULL_STACK, RESEARCH_STATUS, FUTURE, and C++ framework plan updated for native-first workflow.
- **C++23** standard for native build (`native/CMakeLists.txt`).
- **`cyphalm_parity`:** Windows subprocess fix (`CreateProcess` instead of `std::system`).
- **Intelligence Stats papers** moved to `docs/research/intelligence_stats/`.
- **cypha_som** removed; documented as failed experiment under `docs/archive/failed_experiments/cypha_som/`.
- **C++2023 migration plan:** `docs/native/migration/CPLUSPLUS_2023_MASTER_PLAN.md`.
- **Repo layout (Phase A):** `cypha_bench/` → `bench/`, `parity_fixtures/` → `fixtures/`, `install/` → `packaging/`.
- **Release install scripts:** under `packaging/` (`install_release_linux.sh`, `install_release_windows.ps1`).

### Added
- **Kernel LLR XOR pair features:** d03_xor + smoke now use `xor_pair` kernel path — **97.8%** kernel acc (3 seeds, 8 passes); closes diagnostic 32 pp gap.
- **RPSM Option A scaffold:** `PsiMatrices` + `rpsm_score_matrix_batched` + CTest `native_rpsm_batched_llr_smoke`.
- **Intelligence Stats Phase 3:** CyphaLM profiler hook (`--intelligence-profile`), Qt self-correct checkbox, `CausalGraphMonitor`, `profile_guided_loss`, cell hypothesis sweep (`d19`, `cypha_cell_hypothesis_sweep`).
- **Intelligence Stats Phase 4:** EWC regularizer stub + curriculum sampler; `GET /intelligence/simulation`; REST `/update` batch+curriculum; Qt curriculum checkbox; epistemic halt on `/generate`; federated JSON merge (`cypha_federated_merge`).
- **RPSM Option B scaffold:** `rpsm_sequence_layer`, CyphaLM `--mode rpsm`, profile-guided loss in `train_step`; CTests `native_rpsm_sequence_smoke`, `native_cyphalm_bench_rpsm_smoke`.
- **Cell hypothesis H02–H14:** `cypha_cell_hypothesis` + EML activation in char-LSTM; all 28 variants runnable; Tier-2 smoke (`native_cell_hypothesis_tier2_smoke`).
- **WikiText D17 full profile:** `d17_wikitext_full_profile.json` + `CYPHA_BENCH_FULL_CORPUS`; CTest `native_d17_wikitext_smoke`.
- **WikiText D17 overnight profile:** `d17_wikitext_overnight_profile.json`, `--overnight` / `CYPHA_BENCH_OVERNIGHT=1` on `cyphalm_bench_native` and `cypha_bench_run`; CTest `native_d17_wikitext_overnight_smoke`.
- **Qt CyphaLM epistemic halt:** generate tab checkbox mirrors REST `/generate` `epistemic_halt` (Paper IV r_eu gate).
- **Intelligence Stats Phase 5:** RPSM hierarchy (`W_up`/`W_down`, global memory, Izaac init); RPSM batched LLR default path; NIG-state cell H06 + OOD branching H14; profile-guided loss in CyphaLM train-step backprop; EWC D16B smoke + REST `ewc_lambda`; federated coordinator CLI; Paper V multi-step simulation loop.
- **Intelligence Stats Phase 6:** 28-variant `--overnight-sweep`; bench **d20**; `scripts/run_d17_overnight.ps1`.
- **Intelligence Stats Phase 7:** `bench/BASELINE_LOCK.json`; `scripts/publish_release.ps1`; `native_overnight_mini_smoke`.
- **Intelligence Stats Phase 8:** bench **d22** cross-domain intelligence profile (d18 + d16 EWC + d20 cell sweep); `d22_intelligence_cross_profile.json`; **`cypha_baseline_lock`** + `scripts/update_baseline_lock.ps1`; hybrid EWC (SSM/GRIA α + B0 ngram count path, H01 α forget gate); CTests `native_d22_cross_smoke`, `native_ewc_hybrid_smoke`, `native_baseline_lock_smoke`.
- **Intelligence Stats Phase 9 (v2.3.9) — shipped:** hybrid EWC **weight** Fisher on GRIA **U**/**V** + SSM **W_fast** (extends Phase 8 α Fisher); **`scripts/run_overnight_all.ps1`** unified overnight runner (D17 + d21 RPSM + 28-variant cell sweep + baseline-lock refresh); bench **d23** overnight lock validation; EWC anchor/Fisher persistence in CyphaLM **`checkpoint.json`**; CTests `native_d23_overnight_lock_smoke`, `native_ewc_weights_smoke`.
- **Intelligence Stats Phase 10 (v2.3.10) — shipped:** hybrid EWC Fisher on GRIA **bias** + SSM **W_slow** (extends Phase 9 weight Fisher); bench **d24** production lock validation; **`cypha_baseline_lock --run all`** (D17 + d21 + cell-sweep in one CLI); Windows TLS CI mirror **`scripts/ci_federated_tls_windows.ps1`**; CTests `native_d24_production_lock_smoke`, `native_ewc_weights_smoke` (**99 total**).
- **Intelligence Stats Phase 11 (v2.3.11) — shipped:** WikiText-2 download scripts (**`scripts/download_wikitext2.ps1`**, **`scripts/download_wikitext2.sh`**) + **`bench/data/wikitext2/README.md`**; **Gutenberg fallback** for d17/d21 when WikiText absent (`gutenberg_fallback` source tag in **`cyphalm_corpus.cpp`**); **`corpus_smoke`** CLI + bench **d25** corpus readiness validation; overnight **`-Fast`** propagates **`CYPHA_BENCH_FAST=1`** through **`run_d17_overnight.ps1`**, **`run_rpsm_overnight.ps1`**, **`run_overnight_all.ps1`**, **`update_baseline_lock.ps1`** (synthetic corpus without WikiText); CTests `native_d25_corpus_smoke`, `native_corpus_smoke` (**101 total**).
- **H16 SR gate laws:** `sr_gate_laws.hpp/cpp` — fit closed-form forget-gate laws, apply in char_lstm; `native_sr_gate_laws_smoke`.
- **RPSM d21 bench:** end-to-end CyphaLM rpsm `train_sequence`; bench **d21**; `scripts/run_rpsm_overnight.ps1`; `native_d21_rpsm_smoke`.
- **CyphaLM EWC full:** embed + lm_head Fisher (diagonal grad²); extended `native_ewc_cyphalm_smoke`.
- **Federated TLS option:** `CYPHA_ENABLE_OPENSSL=ON` for HTTPS coordinator/worker; `native_federated_tls_smoke` (skipped without OpenSSL).
- **Hybrid EWC:** `HybridEwcRegularizer` — diagonal Fisher on SSM multiscale **α** + GRIA per-token **α** (char-LSTM embed/head Fisher retained); B0 **`ngram_count_table`** online prior when `ngram_context > 0`; H01 **`use_alpha_forget_gate`** scales char-LSTM forget gate by mean GRIA α; CTest `native_ewc_hybrid_smoke`.
- **`cypha_baseline_lock`:** CLI merges D17/d21/cell-sweep BPC into **`bench/BASELINE_LOCK.json`**; wrapper **`scripts/update_baseline_lock.ps1`**; CTest `native_baseline_lock_smoke`.
- **CyphaLM checkpoint:** save/load **`ngram_count_table`** in checkpoint JSON (B0 ngram count path round-trip).
- **CTests +3 (Phase 8):** `native_d22_cross_smoke`, `native_ewc_hybrid_smoke`, `native_baseline_lock_smoke` (**96 total** at Phase 8).
- **CTests +2 (Phase 9, shipped):** `native_d23_overnight_lock_smoke`, `native_ewc_weights_smoke` (**98 total**).
- **CTests +1 (Phase 10):** `native_d24_production_lock_smoke` (**99 total**); extended `native_ewc_weights_smoke` (GRIA bias + SSM W_slow Fisher).
- **CTests +2 (Phase 11):** `native_d25_corpus_smoke`, `native_corpus_smoke` (**101 total**).
- **CTests +3 (Phase 7):** `native_sr_gate_laws_smoke`, `native_d21_rpsm_smoke`, `native_federated_tls_smoke` (**93 total** at Phase 7).
- **Preprocessor:** auto `auto_rff_gamma_cv` when tabular dim ≤30 and `rff_gamma` left at default 1.0.
- **Release notes script:** `scripts/create_release_notes.ps1`; release workflow hook.
- **Web UI tabs:** CyphaLM generate, Experiments (`/models`), Intelligence report (`/intelligence/report`).
- **GGUF export:** tensor blobs for `enc_W`, `world.mu`, class `D`/`D_T`, `inv_v`, `llr_bias` from `.cypha`.
- **ONNX export smoke:** CTest `native_onnx_export_smoke` writes valid graph from `reference.cypha`.
- **Phase B native layout:** `native/apps/`, `native/tests/parity/`, `cmake/CyphaApps.cmake` + `CyphaParity.cmake`.
- **Web UI (§4):** vanilla SPA at `GET /` served by `cypha_rest`; CTest `native_rest_ui_smoke`.
- **Fixture generators:** `cypha_fixture_gen` for batch_llr, memory_train, preprocessor, train_step_vector, quantile_dif_train, regression_m4.
- **`cypha_onnx_export`:** header-only ONNX ModelProto writer for VectorEncoder+LLR inference path.
- **`cyphalm_train`:** native CyphaLM training CLI + checkpoint save; CTest `native_cyphalm_train_smoke`.
- **`cyphalm_ssm_diagnose`:** CellAI/SSM probe tool; `--ssm-diagnose` on `cypha_bench_run` for d10/d17.
- **Intelligence Profiler Papers II–V:** extended measurers + CTest `native_intelligence_profiler_papers`.
- **Kernel LLR tuning:** diverse landmarks, whitening fallback, M=512 tuned profile (XOR ~71% kernel acc).
- **`cypha_kernel_tune`:** grid search wrapper for XOR kernel bench.
- **Packaging (§3):** `packaging/build_appimage.sh`, `packaging/build_windows_bundle.ps1`; release workflow AppImage + native-only assets.
- **Qt dark theme + model card editor (§2d–e);** threaded bulk REST `/update`.
- **`auto_rff_gamma_cv`:** RFF bandwidth grid CV in native preprocessor; Qt Fit Preprocessor gamma mode combo; CTest `native_preprocessor_rff_gamma_cv`.
- **`cypha_fixture_gen`:** native fixture regeneration CLI (`--fixture batch_llr`); CTest `native_fixture_gen_list`.
- **Multi-model REST:** registry map, optional `"model"` on predict/update, `--preload-registry`; CTest `native_rest_multi_model`.
- **Qt bulk train worker:** background `QThread` with live loss chart and cancel.
- **Qt chart UX (§2b):** zoom, pan, tooltips on loss charts.
- **Experiment compare (§2c):** multi-run loss overlay in Experiments panel.
- **`/uncertainty-rank`:** active-learning entropy ranking; CTest `native_rest_uncertainty_rank`.
- **Auto-γ RFF (§0b):** native `PreprocessorState::auto_rff_gamma`; bench + Qt shell wired.
- **Nyström kernel LLR (native):** `KernelMemory` in C++ with train/infer wiring, `.cypha` persistence, XOR bench (`xor_kernel_bench`, CTest `native_xor_kernel_bench_smoke`), bench domain **`d03_xor`** (`cypha_bench_run --domain-tag d03_xor`), opt-in profile `bench/config/kernel_llr_profile.json`.
- **Intelligence Profiler (C++):** `native/include/cypha/intelligence/` — NIG statistic states, 7-stat measurers, κ, health signal; CTest `native_intelligence_profiler_smoke`.
- **Research upgrades hub:** RPSM, nonlinear boundary, cell hypothesis specs under `docs/research/upgrades/`.
- **`cypha_fixture_gen`:** all **32** sidecar parity fixtures regenerable; CTest `native_fixture_gen_list`.
- **`cypha_parity_run`:** unified parity driver `--fixture NAME`; CTest `native_parity_run_list`.
- **`bench_domains.cpp`:** domain runners split from `cypha_bench_run` CLI.
- **`mt19937_rng`:** renamed from `numpy_default_rng` (parity MT19937).
- **REST schema contract:** `native_rest_schema_contract` replaces pytest `test_api_contract`.
- **Studio Web UI tabs:** Update, Models/Load, LM generate, Session RNG, Metrics chart.
- **`embed_static_ui`:** C++ tool replaces Python embed script.
- **Kernel XOR pair features:** raw XOR polynomial kernel path (~97% acc smoke); `kernel_x` / `kernel_features` API.
- **`bench/config/auto_rff_gamma_cv_profile.json`** for d01 RFF CV preprocessor.
- **Diagnostics phase 5:** intelligence profiler inline + SSM recommendations JSON.
- **`GET /intelligence/profile`** and **`GET /intelligence/report`** on `cypha_rest` (report = reference fixture P-space).
- **Qt:** chunked CSV train, MKE bulk worker, uncertainty sort checkbox, Qt Charts `QRectF` fix.
- **`cypha_export_gguf`:** GGUF header + manifest stub; CTest `native_export_gguf_help`.

---

## [2.2.8] — 2026-06-11 · All four CI jobs blocking

### Changed
- **`linux_cuda`** CI job is now **blocking** (was `continue-on-error`); Jimver **`cudart`/`thrust`** + **`CUDAToolkit_ROOT`**; failure log artifact.
- Docs: CONTRIBUTING, VERIFY_PLAN §5, ROADMAP, ACCEL_CUDA, framework plan, README test counts synced to **four blocking CI jobs**.
- `cyphalm_checkpoint_parity`: log BPC delta on Python checkpoint load.
- Native CMake project version **2.2.8**.

---

## [2.2.7] — 2026-06-11 · MSVC CUDA CI green

### Fixed
- **MSVC + CUDA CI (Windows):** green on GHA — `ilammy/msvc-dev-cmd`, explicit `CUDA_PATH` from Jimver output, Ninja **1.12.1** pin, `thrust` subpackage, optional CUDA VS integration copy; failure log artifact.

### Changed
- **`windows_cuda_msvc`** is now a **blocking** CI job (was `continue-on-error`).
- Native CMake project version **2.2.7**.

---

## [2.2.6] — 2026-06-11 · Hybrid checkpoint atol + MSVC CUDA CI hardening

### Fixed
- **MSVC + CUDA CI:** `vswhere` + `vcvars64.bat` + Ninja with explicit `cl`/`nvcc` (Jimver nvcc-only install; no Nsight VSE hang).
- **CyphaLM hybrid checkpoint parity:** tighten `atol_bpc` from 0.35 → **0.02** (measured native load delta ~0.002 BPC).

### Changed
- Native CMake project version **2.2.6**; CyphaLM tracker marks `proj_dif` GRIA wiring complete.
- `bench/README.md`: note on when to commit `report/` baseline snapshots.

---

## [2.2.5] — 2026-06-11 · Docs sync + MSVC CUDA CI fix

### Changed
- Verify hub, ROADMAP, README test counts aligned with green CI (~274 pytest, 52 CTest).
- **MSVC + CUDA CI:** `ilammy/msvc-dev-cmd` + Ninja single-config (configure/build/ctest in one step so `cl`/`nvcc` env persists).

---

## [2.2.4] — 2026-06-11 · CI green (Linux CTest + pytest)

### Fixed
- **Linux CI CTest:** CyphaLM checkpoint sidecars use portable relative `checkpoint.json` paths; `cyphalm_checkpoint_parity` resolves paths against the sidecar directory.
- **Linux CI pytest:** restore empty Studio GUI shim modules as re-exports from `widgets.py`; extend CTest↔pytest registry; add bench/tune/diagnostics/SOM/GH infer subprocess tests; Qt shell `--help` documents PNG/SVG/CSV export keywords.
- **Linux CI Qt:** compile Qt targets on ubuntu-latest; exclude headless-incompatible GUI CTest exec (`native_qt_shell_smoke`, `native_qt_stub_load_reference`).

### Changed
- **`scripts/ci_native_linux.sh`:** skip Qt GUI CTests when `CYPHA_BUILD_QT=1` on headless hosts (matches CI).

---

## [2.2.3] — 2026-05-31 · Release packaging hardening

### Fixed
- **Windows release verify:** fail-fast packaging when required PE binaries are missing; clearer verify step; build `--target all`; rm staging before unzip check.
- **MSVC + CUDA CI:** try VS 2026 then VS 2022 generators; job marked non-blocking (`continue-on-error`).
- **Ruff CI:** pin `ruff>=0.9,<0.10` for reproducible lint.
- **CI MinGW job:** replace PE CTest (cannot run on Linux) with artifact existence smoke.
- **Ruff UP038:** `isinstance(exc, ValueError | TypeError)` for ruff 0.9 pin compatibility.
- **Linux CI Qt/CTest:** install XCB/EGL/GL runtime deps; compile Qt targets; exclude GUI exec tests from CTest on headless runners.

---

## [2.2.2] — 2026-05-31 · CI / release pipeline fixes

### Fixed
- **MinGW cross-compile:** vendored zlib via CMake FetchContent when system `ZLIB` is missing (fixes `cypha_bench_run` on release Windows PE builds).
- **Ruff CI:** auto-format + targeted `per-file-ignores`; all lint paths green.
- **MSVC + CUDA CI:** auto-detect VS 2026 vs 2022 generator on `windows-latest`.

---

## [2.2.1] — 2026-06-11 · Qt Studio dialogs + doc sync

### Added
- **Qt shell:** Settings dialog (QSettings, Studio-compatible prefs) and Confusion Matrix view after labeled batch predict.

### Changed
- Master plan and quickstart marked **v2.5 complete**; validation gate unchanged (52 CTests, 155 pytest).

---

## [2.2.0] — 2026-06-11 · Full C++ framework release

### Added
- **Full native production framework** — bench (`cypha_bench_run`, `cypha_bench_report`), tune (`cypha_tune_run`), diagnostics (`cypha_diagnostics_run`), and REST **`/dif/*`** routes in **`cypha_rest`**; Python required only for fixture generation and research prototyping.
- **`cypha_qt_shell`** — Qt Studio shell with **9 tabs** (Data, Model, Train, Predict, Registry, Server, Experiments, CyphaLM, Help).
- **Native validation gate** — `scripts/cypha_native_validate_all.ps1` orchestrates CTest, pytest parity, bench smoke, tune dry-run, and REST contract checks.

### Changed
- **CyphaLM 300k baseline lock** — native hybrid **2.897 BPC** on D17 @ 300k (Python reference 2.873; Δ +0.024 ~0.8%).
- **Test coverage** — **52 CTests** (`native_*`) and **155 pytest** cases green on the full native gate.

### Installers
- Release bundles via `scripts/package_release_windows.sh` / `scripts/package_release_linux.sh`; see [`docs/native/NATIVE_QUICKSTART.md`](docs/native/NATIVE_QUICKSTART.md).

---

## [1.1.0] — 2026-05-31 · CyphaLM native release

### Added
- **`cypha_lm_native`** — C++ CyphaLM Tiers 0–2–4 (hybrid **2.892 BPC @ 300k** vs Python 2.873).
- Native REST LM routes in **`cypha_rest`**: `/lm/load`, `/lm/metrics`, `/lm/predict_next`, `/generate`, `/generate/stream`.
- Checkpoint save/load with DIF + SSM state; Python GRIA import via `load_from_full_w`.
- GitHub Release workflow: Linux `.tar.gz` + Windows `.zip` installer bundles (`scripts/package_release_*.sh`).

### Installers
- **Linux:** `cypha-1.1.0-linux-x86_64.tar.gz` → `bash install.sh`
- **Windows:** `cypha-1.1.0-windows-x86_64.zip` → `powershell -File install.ps1`

---

## [Unreleased] — 2026-06-10 CyphaLM native Tier 0–2–4 integration

### Added
- **`cypha_lm_native`** static library — `native/src/cyphalm/*.cpp` (CMake `GLOB`), OpenMP optional.
- **`cyphalm_bench_native`** — BPC bench CLI (`--mode`, `--profile d17|d04`, `--n-train`, `--n-eval`, `--threads`).
- **`cyphalm_parity`** — meta-runner for native CyphaLM parity tools.
- **`scripts/generate_cyphalm_native_fixtures.py`** — one-time Python → `fixtures/cyphalm_*/sidecar.json`.
- **`tests/test_cyphalm_native_parity.py`** — subprocess parity (skip if binary missing).
- **`native/include/cypha/cyphalm/cyphalm_config.hpp`** — unified config + bench mode mapping.
- PORT_CONTRACT **§4b** and `CYPHALM_NATIVE_UPGRADE_MASTER.md` integration / build notes.

---

## [Unreleased] — 2026-06-08 Branch A frozen embeddings

### Added
- **`frozen_text_embeddings.py`** — MiniLM or hashing frozen text vectors.
- **`cypha_branch_a_sweep.py`** — CyphaDIF on frozen ST vs TF-IDF on 20 Newsgroups.

### Changed
- Branch A @ 2k samples: CyphaDIF + frozen MiniLM **62.5%** vs TF-IDF path **34.0%**.
- D09 integration: `run_d09_branch_a.py`, Gutenberg OOD epistemic **6.3×** in-domain.

### Added (follow-up)
- `branch_a_documents.py`, `run_d09_branch_a.py`, `scripts/demo_branch_a_route.py`.
- **REST Branch A routing:** `POST /route/text`, `POST /route/generate`, Ollama fallback client.
- **Studio chat:** Branch A mode in Settings → Inference; `BranchADispatchWorker` routes text in chat.
- **Router checkpoint:** `save_checkpoint` / `load_checkpoint`, `scripts/save_branch_a_router.py`, `POST /route/save`.
- **Encoder sweep:** VectorEncoder **59.5%** vs RFF **4.5%** @ 384-d MiniLM — keep VectorEncoder.

---

## [Unreleased] — 2026-06-08 Cypha Tests Phase 2A encoder

### Added
- **`EncoderProjection.hebbian_update`** and `CyphaDIF.encoder_update_mode` (`contrastive` | `hebbian`).
- **`cypha_encoder_phase2a_sweep.py`** — D01 + 20 Newsgroups contrastive vs Hebbian benchmark.

### Changed
- Phase 2A baseline: competitive Hebbian **underperforms** contrastive on all 4 tasks; keep contrastive default.

---

## [Unreleased] — 2026-06-08 char_lstm mode + Phase 2 Hebbian

### Added
- **`context_mode=char_lstm`** — LSTM-only path in `CyphaLM` (C1 model-class).
- **`cyphalm_hebbian_phase2_sweep.py`** — Cypha Tests 2C sparse Hebbian toggle.
- **`docs/CYPHA_TESTS_PHASE2.md`** — Phase 2 experiment map (2A–2C).

### Changed
- Hybrid sweep supports `--cells` filter; third cell `char_lstm`.
- **Char-LSTM @ 300k:** **2.876 BPC** (≈ hybrid 2.873); Hebbian SSM **neutral @ 40k**.

---

## [Unreleased] — 2026-06-08 char 1A + D04 refresh

### Added
- **Char-level shuffle probe** (Cypha Tests 1A) in `eval_shuffled_stream_bpc`.
- **`run_d04_hybrid_refresh.py`** — D04 @ 300k with ablation stub from sweep.

### Changed
- **D04 @ 300k hybrid:** **2.993 BPC** (Moby Dick); figures and `d17.json` 17K char-shuffle fields updated.
- **Cypha Tests 1A:** passes @ char shuffle (+4.54 BPC @ 300k hybrid); block shuffle still flat.

---

## [Unreleased] — 2026-06-07 CyphaLM hybrid + long-range

### Added
- **`hybrid_gria_lstm`** context mode — GRIA + char-LSTM dual head with online blend (`cypha_lm/model/char_lstm_head.py`).
- **Long-range context suite** — `cyphalm_long_range.py`, `cyphalm_long_range_suite.py`, D17 **17K**, `docs/CYPHALM_LONG_RANGE_TESTS.md`.
- **Upgrade V2** — learnable views (`view_embed.py`), gated n-gram fusion (`ngram_fusion.py`); sweeps + D17 **17I/17J**.
- **Phase 1c runner** — `run_d17_phase1c.py`; 300k train cap via `cyphalm_bench_limits()`.
- **Profiles** — `cyphalm_d17_hybrid.json`; D17/D04/llm defaults → `hybrid_gria_lstm`.

### Changed
- **D17 @ 300k:** hybrid **2.873 BPC** (Phase 1c 17A) — beats bigram and char-LSTM bench.
- **`d17.json`**, **`BASELINE_REPORT.md`**, research docs refreshed for hybrid + 17K long-range.

---

## [Unreleased] — 2026-05-31 CyphaLM + LLM features

### Added
- **D04 rewritten for CyphaLM** — char-LM domain now runs Izaac → CellAI SSM → CyphaDIF → GRIA (not raw CyphaDIF + CharNgramEncoder).
- **`bench/adapters/cyphalm_bench.py`** — shared LM helpers for D04/D17: BPC eval, context-length curve, save/restore fidelity, sampling comparison, expert routing trace.
- **D04 experiments:** BPC vs context length, CyphaDIF expert routing during generation, checkpoint round-trip parity, sampling strategy bar chart (`fig04_context_bpc`, `fig04_expert_routing`, `fig04_sampling_strategies`).
- **CyphaLM generation:** `top_p_sample` (nucleus), unified `autoregressive_decode`, `stream_generate` SSE chunks; `predict_next` exposes `routing_probs`, `dominant_expert`, `active_experts`.
- **CyphaStudio LM REST (FastAPI-only):** `POST /lm/load`, `GET /lm/metrics`, `POST /lm/predict_next`, `POST /generate`, `POST /generate/stream` (SSE with epistemic gating).
- **`cypha_studio/core/lm_engine.py`** — `LMEngine` wrapper for CyphaLM inference and streaming.
- **`CYPHA_LM_CHECKPOINT`** env var — auto-load CyphaLM at FastAPI startup.
- **`tests/test_lm_api.py`** — generation utilities + REST route tests.
- **`examples/lm_generate_body.json`**, `curl_lm_generate_stream.sh/.ps1`.
- **`docs/port/PORT_CONTRACT.md` §4** — CyphaLM REST contract.
- **CyphaStudio GUI CyphaLM chat** — File → Load CyphaLM…; streamed char generation in chat with expert/epistemic status (`lm_generation_worker.py`, `chat_widget.py`).
- **`scripts/generate_demo_lm_checkpoint.py`** — trains a tiny demo checkpoint for Studio/REST smoke tests (`examples/demo_cyphalm/README.md`).
- **CI** — `pip install -e cypha_lm/`, scoped ruff, `cypha_lm/model/tests/` in pytest matrix.
- **Installers** — Windows/Linux scripts run `test_lm_api` and document demo checkpoint generation.

### Changed
- **`cypha_lm/model/cypha_lm.py`** — `generate()` accepts `strategy`, `top_k`, `top_p`; adds `stream_generate()`.
- **Documentation:** `cypha_lm/README.md`, `bench/README.md`, `cypha_studio/README.md`, `docs/studio/CYPHA_ENV.md`, `examples/README.md`, `docs/RESEARCH_STATUS.md`.
- **D10 time-series tuning** — ECG passes 4→8, encoder window 50→32, n_fft 10→16.
- **`DEFAULT_CYPHALM_CONFIG`** — `gria_lr: 0.06`, `online: True` for bench training.
- **Full D04/D17 benchmark refresh** — updated figures, tables, and `BASELINE_REPORT.md`.

---

## [Unreleased] — 2026-05-31 polish pass

### Added
- **`docs/RESEARCH_STATUS.md`** — canonical research journal: 17-domain benchmark table,
  confirmed properties, hard limits, hypothesis ledger, and forward research map.
- **`docs/reports/`** — permanent archive of investigation reports; all bench and diagnostic
  reports copied here (`BENCH_TUNING_REPORT.md`, `BENCH_ARCH_TUNING_REPORT.md`,
  `BENCH_ARCH_RESCORE_REPORT.md`, `BENCH_UPGRADE_REPORT.md`, `BENCH_PAPER.md`).
- **`CHANGELOG.md`** (this file).
- **Package READMEs:** `cypha_som/README.md`, `cypha_accel/README.md`;
  major rewrite of `cypha_lm/README.md` (architecture table, D04 bug warning,
  known limitations, configuration guide, empirical results).
- **`bench/README.md`** — 17-domain structure, run instructions, single-domain usage.
- **`examples/`** — `README.md`, `cypha_update_body.json`, `cypha_load_body.json`,
  `cypha_adapt_temperature_body.json`, `curl_predict.sh`, `curl_predict.ps1`.
- **`packaging/`** — `install_windows.ps1`, `install_linux.sh`, `README.md`.
- **`docs/studio/CYPHA_STUDIO_MASTER_PLAN.md`** — historical stub fixing broken links.
- `cypha_diagnostics/README.md` — explains package purpose and confirmed findings.
- Proper `[project]` metadata in `pyproject.toml` (name, version, description,
  dependencies, optional extras `studio`, `gpu`, `dev`).
- `cypha_lm/` submodule `__init__.py` docstrings (embeddings, temporal, model,
  expert_field, projection, analysis).
- `native/CMakeLists.txt` VERSION field (`0.1.0`).
- CI `concurrency` group (cancel redundant PR runs), `PYTHON_VERSION` env var,
  job/step names for clarity.
- `Makefile` full `.PHONY` declaration and `help` target.

### Fixed
- **D04 benchmark bug** — `d04_generation_language.py` was indexing `probs[char_id]`
  into a label-ordered probability array, yielding a nonsensical 33.2 bpc floor.
  Fixed to map via `memory._classes.keys()` order. All documentation references to
  "33.2 bpc CyphaLM failure" updated to reflect this was a benchmark bug.
- **D04 clarification propagated** to `CHANGELOG [1.0.0]`, `DIAGNOSTIC_REPORT.md`,
  `docs/FUTURE.md`, `README.md`, `cypha_lm/README.md`.
- `bench/BASELINE_REPORT.md` header clarified: this is the **post-diagnostic
  tuned** run, not a default-parameters baseline.
- `docs/port/PORT_FULL_STACK.md` M6 ExperimentDB API checkbox marked complete.
- 36 `pytest.importorskip` calls in `test_api_contract.py` and
  `test_cypha_rest_smoke.py` now include `reason=` strings.

### Changed
- **`docs/port/PORT_CONTRACT.md`** updated: Kernel LLR (Python-only), deliberation
  (Python-only, default OFF), `gh_infer` vs native `use_gh` gap, `/session/rng` routes,
  `field_sr_vec` key, fixture staleness note (2026-05-30 additions don't affect fixtures
  when generated with defaults).
- Moved `results/SOM_UPGRADE_REPORT.md` → `docs/reports/`.
- Moved `cypha_diagnostics/DIAGNOSTIC_REPORT.md` → `docs/reports/`.
- `scripts/README.md` extended with 15+ previously undocumented scripts (parity
  fixture generators, CyphaLM/SOM eval runners, analysis utilities, PS extras).
- `docs/verify/VERIFY_PLAN.md` — updated from single Ubuntu CI job to 2 jobs;
  `.sh` script notes.
- `docs/verify/VERIFICATION_STATUS.md` — 2 CI jobs, Not-in-CI table for
  `cypha_som/tests/` + `cypha_lm/`, D04 note, full 24-dir fixture inventory,
  `native_generation` in CTest list.
- `docs/verify/ROADMAP.md` — Phase 5 now lists Kernel LLR as top priority;
  Phase 6 entry added for CyphaLM (D17: 4.50 bpc).
- `CONTRIBUTING.md` — install scripts, `cypha_lm`/`cypha_som` test commands,
  2-job CI, `RESEARCH_STATUS.md` link.
- `docs/README.md` — research status section, bench report index, benchmark commands.
- `fixtures/README.md` — missing fixtures added (memory_train, preprocessor,
  f_field, regression_m4, rff_regression, two_stage_*, generation, registry_register).
- `benchmark.py` and `benchmark_baseline.py` — expanded docstrings and `--help`.
- `native/README.md` — M6 experiments section updated to reference CRUD parity.
- `.gitignore` — added MNIST `*.gz` files; removed `*.sh` rule; deduplicated entries.

### Removed
- Stale planning files: `CyphaDIF_TestBench.md`, `CyphaLM_Plan.md`,
  `cypha_diagnostic_plan.md`, `cypha_som_upgrades.md` — outcomes live in
  implemented packages and `docs/reports/`.
- Raw experiment JSON from `results/` and `cypha_diagnostics/results/`
  (ephemeral; regenerate via `scripts/run_som_upgrade_eval.py` and
  `cypha_diagnostics/run_diagnostics.py`).

---

## [1.0.0] — 2026-05-30 · commit `1dbfa13`

**Cypha comprehensive upgrade — diagnostics, CellAI, KernelLLR, multi-pass**

### Highlights
- **Three root-cause bugs found and fixed** via the full diagnostic plan
  (`cypha_diagnostics/run_diagnostics.py`):
  - Bug 1: Deliberation band `[0.4, 0.6]` was masking ~40% of predictions as
    `__unknown__` on binary problems. Fix: `deliberation_lo=1.0, deliberation_hi=0.0`
    (disabled). Effect: +23.5 pp on S1_2class_linear; regression R² −0.007 → 0.756.
  - Bug 2: `delta_lr=0.06` was too aggressive. Fix: `delta_lr=0.03`. Effect: +4 pp
    on R3_digits (0.882 → 0.922).
  - Bug 3: `VectorEncoder` inadequate for `input_dim ≤ 30`. Fix: auto-select
    `RFFEncoder(D=256)` for small inputs. Effect: +14 pp on S1_2class over
    `VectorEncoder` alone.
- **+23.5 pp** on linearly-separable 2-class benchmark; **+20.5 pp** on digits.
- SOM upgrade evaluation completed (`scripts/run_som_upgrade_eval.py`): all six
  upgrades benchmarked. Verdict: default flags remain OFF (U2 hurts accuracy;
  U1 doubles latency with no benefit; U3/U5/U6 are neutral on classification).
- Multi-pass training (`n_epochs` from profile) used in D01 domain loop: ~3 pp gain.

### Architecture — confirmed limits
- **XOR / nonlinear boundaries:** FDR=0.001, kernel(h)=0.835 vs linear(h)=0.512.
  Gap of 32.3 pp is a hard LLR-linearity ceiling. **Requires Kernel LLR (Nyström)**
  to close. Highest-priority future upgrade.
- **D04 char-LM metric is a benchmark bug** — D04 runs `CyphaDIF + CharNgramEncoder`
  (not CyphaLM); "33.2 bpc" is caused by wrong probability indexing (`probs[next_idx]`
  indexes by char ID into a label-ordered array, hitting the `1e-10` floor → -log2(1e-10)
  = 33.2). The SGD "0.66 bpc" is cherry-picked from step 1000 (final SGD is 1.51 bpc).
  Real CyphaLM evaluation: **D17 held-out BPC = 4.50** (bigram baseline 3.69).
- **CellAI / D10 ECG:** 17–20% accuracy on 5-class time-series; temporal SSM domain not yet tuned.

### Files changed
- `bench/config/everyday_profile.json` — deliberation disabled, delta_lr=0.03.
- `bench/adapters/bench_models.py` — auto-RFF for `input_dim ≤ 30`.
- `bench/domains/d01_statistical_baselines.py` — multi-pass with `n_epochs`.
- `cypha_diagnostics/` — new diagnostic package (`run_diagnostics.py`, `apply_upgrades.py`).
- `cypha_som/` — SOM/GNG/GRIA/Hebbian/temporal hooks (all flags OFF by default).
- `benchmark_baseline.py` — baseline runner for SOM upgrade evaluation.
- `scripts/run_som_upgrade_eval.py`, `scripts/run_cypha_lm_report.py`,
  `scripts/merge_final_profile.py` — new utility scripts.

---

## [0.9.0] — 2026-03-31 · commit `8945c95`

**Refactor, docs cleanup, Qt streaming training thread**

### Highlights
- Qt shell streaming training: `QThread` worker emitting `lossReported` and
  `valAccReported` signals; live loss chart updates every N steps.
- Documentation pass across all `docs/` subdirectories.
- Minor refactors to `Cypha.py` and `cypha_studio/` for consistency.

---

## [0.1.0] — 2026-03-31 · commit `127bbe9`

**Initial commit — Cypha native port M1–M6 complete**

### Summary

First committed state of the project. All six native port milestones signed off:

| Milestone | Description |
|-----------|-------------|
| M1 | Inference kernel: encode + LLR + GH gate + softmax vs `fixtures/`. |
| M2 | Registry + preprocessor: fit (scale/PCA), transform, CSV load. |
| M3 | Online `train_step`: DIF, GH, replay, NIG, context, OOD. |
| M4 | Regression stack: MKE / RFF / two-stage / ridge / EMA. |
| M5 | `cypha_rest` native server + Qt shell (`cypha_qt_shell`). |
| M6 | Experiments DB (SQLite amalgamation) + Qt M6 panel. |

- **188 pytest + 33 CTest** cases across 13 named parity fixtures.
- Python reference (`Cypha.py`, `cypha_studio/`) serving as golden spec.
- `cypha_lm/` research package (LM stack, embeddings, SSM, experts).
- `bench/` evaluation harness (17 domains, encoders, reports).
- `cypha_som/` optional SOM/GNG/GRIA hooks.
- `cypha_accel/` CuPy-accelerated LLR / projection / NIG helpers.
- GitHub Actions CI: **four blocking jobs** — Linux CTest + pytest, MinGW PE, MSVC + CUDA, GCC + CUDA.

---

[Unreleased]: https://github.com/odin-loki/Cypha/compare/v2.2.8...HEAD
[2.2.8]: https://github.com/odin-loki/Cypha/compare/v2.2.7...v2.2.8
[2.2.7]: https://github.com/odin-loki/Cypha/compare/v2.2.6...v2.2.7
[2.2.6]: https://github.com/odin-loki/Cypha/compare/v2.2.5...v2.2.6
[2.2.5]: https://github.com/odin-loki/Cypha/compare/v2.2.4...v2.2.5
[2.2.4]: https://github.com/odin-loki/Cypha/compare/v2.2.3...v2.2.4
[2.2.3]: https://github.com/odin-loki/Cypha/compare/v2.2.2...v2.2.3
[2.2.2]: https://github.com/odin-loki/Cypha/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/odin-loki/Cypha/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/odin-loki/Cypha/releases/tag/v2.2.0
[1.1.0]: https://github.com/odin-loki/Cypha/releases/tag/v1.1.0
[1.0.0]: https://github.com/odin-loki/Cypha/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/odin-loki/Cypha/compare/v0.1.0...v0.9.0
[0.1.0]: https://github.com/odin-loki/Cypha/releases/tag/v0.1.0
