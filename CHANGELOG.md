# Changelog

All notable changes to Cypha are recorded here. The project follows a
**milestone** release model: each entry corresponds to a named engineering
milestone or a significant self-contained change.

---

## [Unreleased] — 2026-05-31 CyphaLM + LLM features

### Added
- **D04 rewritten for CyphaLM** — char-LM domain now runs Izaac → CellAI SSM → CyphaDIF → GRIA (not raw CyphaDIF + CharNgramEncoder).
- **`cypha_bench/adapters/cyphalm_bench.py`** — shared LM helpers for D04/D17: BPC eval, context-length curve, save/restore fidelity, sampling comparison, expert routing trace.
- **D04 experiments:** BPC vs context length, CyphaDIF expert routing during generation, checkpoint round-trip parity, sampling strategy bar chart (`fig04_context_bpc`, `fig04_expert_routing`, `fig04_sampling_strategies`).
- **CyphaLM generation:** `top_p_sample` (nucleus), unified `autoregressive_decode`, `stream_generate` SSE chunks; `predict_next` exposes `routing_probs`, `dominant_expert`, `active_experts`.
- **CyphaStudio LM REST (FastAPI-only):** `POST /lm/load`, `GET /lm/metrics`, `POST /lm/predict_next`, `POST /generate`, `POST /generate/stream` (SSE with epistemic gating).
- **`cypha_studio/core/lm_engine.py`** — `LMEngine` wrapper for CyphaLM inference and streaming.
- **`CYPHA_LM_CHECKPOINT`** env var — auto-load CyphaLM at FastAPI startup.
- **`tests/test_lm_api.py`** — generation utilities + REST route tests.
- **`examples/lm_generate_body.json`**, `curl_lm_generate_stream.sh/.ps1`.
- **`docs/port/PORT_CONTRACT.md` §4** — CyphaLM REST contract.

### Changed
- **`cypha_lm/model/cypha_lm.py`** — `generate()` accepts `strategy`, `top_k`, `top_p`; adds `stream_generate()`.
- **Documentation:** `cypha_lm/README.md`, `cypha_bench/README.md`, `cypha_studio/README.md`, `docs/studio/CYPHA_ENV.md`, `examples/README.md`, `docs/RESEARCH_STATUS.md`.

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
- **`cypha_bench/README.md`** — 17-domain structure, run instructions, single-domain usage.
- **`examples/`** — `README.md`, `cypha_update_body.json`, `cypha_load_body.json`,
  `cypha_adapt_temperature_body.json`, `curl_predict.sh`, `curl_predict.ps1`.
- **`install/`** — `install_windows.ps1`, `install_linux.sh`, `README.md`.
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
- `cypha_bench/BASELINE_REPORT.md` header clarified: this is the **post-diagnostic
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
- `parity_fixtures/README.md` — missing fixtures added (memory_train, preprocessor,
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
- `cypha_bench/config/everyday_profile.json` — deliberation disabled, delta_lr=0.03.
- `cypha_bench/adapters/bench_models.py` — auto-RFF for `input_dim ≤ 30`.
- `cypha_bench/domains/d01_statistical_baselines.py` — multi-pass with `n_epochs`.
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
| M1 | Inference kernel: encode + LLR + GH gate + softmax vs `parity_fixtures/`. |
| M2 | Registry + preprocessor: fit (scale/PCA), transform, CSV load. |
| M3 | Online `train_step`: DIF, GH, replay, NIG, context, OOD. |
| M4 | Regression stack: MKE / RFF / two-stage / ridge / EMA. |
| M5 | `cypha_rest` native server + Qt shell (`cypha_qt_shell`). |
| M6 | Experiments DB (SQLite amalgamation) + Qt M6 panel. |

- **188 pytest + 33 CTest** cases across 13 named parity fixtures.
- Python reference (`Cypha.py`, `cypha_studio/`) serving as golden spec.
- `cypha_lm/` research package (LM stack, embeddings, SSM, experts).
- `cypha_bench/` evaluation harness (17 domains, encoders, reports).
- `cypha_som/` optional SOM/GNG/GRIA hooks.
- `cypha_accel/` CuPy-accelerated LLR / projection / NIG helpers.
- GitHub Actions CI: Linux native build (CTest) + pytest on every push.

---

[Unreleased]: https://github.com/odin-loki/Cypha/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/odin-loki/Cypha/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/odin-loki/Cypha/compare/v0.1.0...v0.9.0
[0.1.0]: https://github.com/odin-loki/Cypha/releases/tag/v0.1.0
