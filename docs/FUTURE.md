# Future directions

**One Cypha.** Public type `cypha::Cypha` (classify + regress + latent sample + tokens). Native port (M1-M6 + P7) is complete -- inference, training, REST, Qt shell, experiments DB, parity fixtures -- CI **214 CTests** (see `scripts/cypha_native_validate_all.ps1`). Python runtime removed.

**Forward path:** living sequence spine **Hybrid GRIA+LSTM L2+Wave2 BPTT (2.664 BPC lock)** under `cypha::Cypha`, with **predictive arithmetic coding** (model probs → entropy coder) as the text compress/generate path. U06 PGM→Wy remains opt-in. Cutover: [`reports/ONE_CYPHA_CUTOVER.md`](reports/ONE_CYPHA_CUTOVER.md). Dated reports: [`archive/`](archive/README.md).

**Last updated:** 2026-07-19

---

## 0 -- Evidence-confirmed upgrades (post-diagnostic, highest priority)

These come directly from the 2026-05-30 diagnostic run
([`archive/reports/DIAGNOSTIC_REPORT.md`](archive/reports/DIAGNOSTIC_REPORT.md)). Every
item has a measured effect size; nothing is speculative.

**RPSM roadmap (Option A + B):** see [10](#10--rpsm-matrix-refactor--sequence-layer-closed) and [`docs/research/upgrades/`](research/upgrades/README.md). Option B is **STOP / deprioritized** (2026-07-18).

### 0a — Kernel LLR via Nyström approximation (SHIPPED — tuning continues)

**Evidence:** FDR=0.001 on XOR; `linear(h)=0.512` (chance) vs `kernel(h)=0.835`.
Nonlinearity gap = **32.3 pp** — unreachable with the current linear LLR discriminant.

**Shipped (2026-05-31):** Whitened Nyström features in native C++ (`native/src/kernel_memory.cpp`):
1. Reservoir landmarks (**M=256** default) with median-γ RBF bandwidth.
2. `φ(h) = K(h, landmarks) · K(landmarks, landmarks)^{-1/2}` via Cholesky whitening.
3. Online softmax gradient on `φ(h)`; blended into `score_matrix` when kernel enabled.

**Measured gain (3 seeds, 8 passes, blend=1.0, M=256, replay off, 2026-06):** native linear **49.9%** → **59.2%** (+ **+9.3 pp**). RFF auto-γ (2026-07-11) closes the sklearn RBF ceiling gap to **~2.7pp** at `rff_dim=4096` (was ~18pp at Nyström M=256 default). `.cypha` kernel keys via `patch_kernel_into_root`; Qt shell + native `cypha_rest` train/infer/save/load wired. Bench domain **`d03_xor`** (`cypha_bench_run --domain-tag d03_xor`). Opt-in profile: `bench/config/kernel_llr_profile.json`. Full validate includes d03_xor fast smoke + REST kernel body test.

> **P7 note:** Python `cypha_core` / `KernelMemory` removed; native path is authoritative. Full fix taxonomy: [`research/upgrades/NONLINEAR_BOUNDARY.md`](research/upgrades/NONLINEAR_BOUNDARY.md).

### 0b — Auto-gamma for RFF bandwidth (SHIPPED — bench + studio + native fit)

**Evidence:** `D_rff` sweep showed high variance between D=256 and D=512, suggesting
the gamma bandwidth is not well-tuned for all datasets.

**Shipped (2026-06):** Native `PreprocessorState::auto_rff_gamma` + `estimate_rff_gamma_median_pairwise` in `fit_from_design_matrix`. Bench: native `BenchClassifier.prepare_encoder_from_data` wired via online train. Qt shell trainer: auto-γ before online loop. CTest: **`native_preprocessor_fit`** (RFF fixture under `fixtures/preprocessor_fit_rff/`). **`auto_rff_gamma_cv`** grid CV also shipped.

> **P7 note:** Python `RFFEncoder.fit(X)` removed with `cypha_core`.

**Expected gain:** +2–4 pp on small-dimensional datasets (re-benchmark d01 small tasks to confirm).

### 0c — D10/D17 CellAI SSM investigation

**Evidence:**
- **D10 ECG (resolved, 2026-07-18):** D10A default **90.11%** on real UCR ECG5000 with enriched features; legacy `CYPHA_D10_ECG_ENRICH=0` -> **85.96%**; synthetic fallback was **60.67%**. Scored path uses the `cypha_core` DIF expert-routing classifier, not `CellAISSM`. Archive: [`D10_ECG_SSM_DIAGNOSIS_2026-07-11.md`](archive/reports/D10_ECG_SSM_DIAGNOSIS_2026-07-11.md), [`D10_ECG5000_GT90_ATTEMPT_2026-07-18.md`](archive/reports/D10_ECG5000_GT90_ATTEMPT_2026-07-18.md), [`D10_ECG5000_REAL_DATA_2026-07-18.md`](archive/reports/D10_ECG5000_REAL_DATA_2026-07-18.md).
- D17 sequence: **hybrid_gria_lstm L2 + Wave2 BPTT @ 300k = 2.664 BPC** is the **living production pin** (`bench/BASELINE_LOCK.json`). Default spine: Hybrid + predictive AC -- [`ONE_CYPHA_CUTOVER.md`](reports/ONE_CYPHA_CUTOVER.md).
- **D04 "33.2 bpc" was a benchmark bug** -- do not use it as evidence. Native D04 runs the sequence stack via `cypha_bench_run --domain 4`.

**What to do:** Instrument native SSM state (`cyphalm_ssm_diagnose`, `--ssm-diagnose` on bench) to verify that:
- State norms do not collapse or explode over long sequences.
- Multi-scale decay rates (τ_fast=1.0, τ_slow=20.0) are appropriate for the domain.
- Output projections are properly connected to the expert routing head.

**D10 status:** the above three checks were run against D10's SSM anyway for completeness (`cyphalm_ssm_diagnose --domain d10`): no collapse/explosion (verdict: pass), and no connectivity check applies since D10 has no routing head to disconnect from in the first place. **This instrumentation remains relevant for D17/CyphaLM** (which genuinely uses the SSM -> GRIA routing path); it is no longer an open question for D10.

> **P7 note:** Python `CellAISSM` / `cypha_lm` packages removed; instrument native SSM via `cyphalm_bench_native` and CTests under `native_cyphalm_*`. Cell hypothesis sweep **closed** (2026-07-18): H19 **~2.921 BPC** best cell; hybrid **2.664** (Wave2 BPTT) is the living pin. Living sequence default is Hybrid GRIA+LSTM L2 -- [`CELL_SWEEP_SUMMARY_2026-07-18.md`](archive/reports/CELL_SWEEP_SUMMARY_2026-07-18.md), [`ONE_CYPHA_CUTOVER.md`](reports/ONE_CYPHA_CUTOVER.md).

---

## 1 — CUDA GPU path (`cypha::accel`)

**Status:** Native **CUDA** in `native/src/accel_cuda.cu` (pooled device memory + Bessel table for GH–NIG gate) plus `accel_backend.cpp`. Without `-DCYPHA_ENABLE_CUDA=ON`, the same APIs use **ISO C++** `std::thread`. **`infer_cpu`** routes **`batch_encode`**, **`score_matrix_use_field`**, and **`world_gate_vector_use_field`** through **`cypha::accel`** (CUDA when batch rows ≥ **`CYPHA_ACCEL_GPU_MIN_BATCH_ROWS`**, default **1** — GPU used for all n≥1 when available). **`cuda_smoke`** checks encode / score / softmax / tanh gate / NIG gate vs references; **`--bench`** compares CUDA vs CPU refs when a GPU is present.

**Windows (native MSVC):** install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), then:
```powershell
cd native
cmake --preset windows-msvc-release -DCYPHA_ENABLE_CUDA=ON
cmake --build --preset windows-msvc-release-build
.\build-windows-msvc\Release\cuda_smoke.exe
```

**WSL2 / Linux:** install NVIDIA driver on Windows + CUDA toolkit inside WSL (`nvidia-cuda-toolkit` or NVIDIA's `.run` installer). Use preset **`wsl-gcc-release`** and add `-DCYPHA_ENABLE_CUDA=ON`. Set `-DCMAKE_CUDA_ARCHITECTURES=` to your GPU (e.g. `89`, `86`, `75`).

**Not supported:** MinGW cross-compiles cannot enable `CYPHA_ENABLE_CUDA` (CMake will error).

**CI — local-only (formal, 2026-07-17):** CUDA is **not** compiled or tested in GitHub Actions. The former **`windows_cuda_msvc`** and **`linux_cuda`** jobs were removed in v2.2.8 and **will not return** on hosted runners (no GPU fleet). Validate on a **self-hosted** or local CUDA box with `-DCYPHA_ENABLE_CUDA=ON`, then run **`native_cuda_smoke`** and **`native_score_batch`**. CPU-only CI (**Linux CTest**, **`windows_msvc`**) is the release gate; MinGW is not a gate. See [`docs/native/ACCEL_CUDA.md`](native/ACCEL_CUDA.md).

**Performance:** profile with `./cuda_smoke --bench` on GPU; small batches may be CPU-faster due to launch overhead.

---

## 2 — Qt shell richer UX

The Qt shell (`cypha_qt_shell`) covers the full training/inference/registry workflow. Items **2a–2e shipped** (2026 Q2); see [`native/qt/README.md`](../native/qt/README.md).

### 2a — Streaming training progress — SHIPPED
Bulk native train runs on a **background `QThread`**. Live loss chart + rolling accuracy drain every 80 ms; cancel via atomic flag; final state synced on finish.

### 2b — Chart interactivity — SHIPPED
Painted `SimpleLossChart` and optional **`QChartView`** (`-DCYPHA_QT_CHARTS=ON`): mouse-over tooltip, pan, scroll-wheel zoom (clamped to data extents). PNG/SVG/CSV export.

### 2c — Experiment run comparison view — SHIPPED
Experiments panel **"Compare selected runs"** overlays `metrics_history` loss curves for 2+ runs (colour per run).

### 2d — Model card editor — SHIPPED
**`card.json`** dialog edits name, description, tags, task before registry register.

### 2e — Dark theme — SHIPPED
**Fusion** palette toggle in Settings; persisted to `~/.cypha/shell_settings.json` and QSettings.

**Remaining Qt polish (optional):** richer compare-run statistics; experiment diff export; additional chart series types.

---

## 3 — Packaged standalone binary

Goal: a single distributable executable (`cypha_qt_shell`) with no external runtime dependencies.

**Status (2026-07-18): SHIPPED** — release **v2.3.25** (One Cypha) includes packaging; scripts below are the maintained workflow.

**Linux AppImage:**
1. Build with `DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_QT=ON`.
2. Use `linuxdeploy` + `linuxdeploy-plugin-qt` to bundle Qt .so files.
3. Optionally bundle a stripped `cypha_rest` as a sidecar.

Scripts: [`packaging/build_appimage.sh`](../../packaging/build_appimage.sh).

**Windows `.exe`:**
1. Native MSVC/Qt build + `windeployqt` via [`native/scripts/package_windows_qt.ps1`](../native/scripts/package_windows_qt.ps1).
2. Bundle script: [`packaging/build_windows_bundle.ps1`](../../packaging/build_windows_bundle.ps1).

Release bundles: [`packaging/`](../../packaging/) install scripts + `scripts/package_release_*.sh`.

---

## 4 — Web UI (browser-based Studio)

Native **`cypha_rest`** (cpp-httplib; see [`PORT_CONTRACT.md`](port/PORT_CONTRACT.md) §3) exposes the full REST API.

### Minimal SPA — SHIPPED (2026 Q2)
Vanilla JS SPA at **`GET /`** — health, predict, update, models, session RNG. Static assets under `native/tools/static/`; optional embed via `-DCYPHA_EMBED_STATIC_UI=ON`. CTest **`native_rest_ui_smoke`**.

**Remaining (expansion):** live training charts, experiment browser, richer model registry UX.

**Option B — htmx + server-side HTML:** not started; lower priority now that minimal SPA ships.

---

## 5 — Multi-model serving in `cypha_rest`

**Current:** `cypha_rest` serves one model at a time (load-then-predict; hot-swap via `POST /load`).

**Target architecture:**
```
cypha_rest --registry <root> --listen 0.0.0.0:8765
  GET  /models          → list all registered models
  POST /predict         → body: { "model": "<name>/<version>", "input": [...] }
  POST /update          → body: { "model": "...", "input": [...], "correct_label": "..." }
  POST /load            → hot-swap active model (for backward compat)
```

**Implementation sketch:**
- `g_models: std::unordered_map<std::string, CyphaInferModel>` — keyed by `name/version`.
- Per-model `std::mutex` for train vs infer serialisation.
- `POST /load` without a body → load all models in the registry at startup.
- Optional LRU eviction (keep N most-recently-used models in RAM).

**Benefit:** one process can serve an A/B test or a production-then-staging model pair without two separate server instances.

---

## 6 — Curriculum / active learning

The current training loop is purely online with a simple replay buffer. Higher-quality training on skewed datasets can use:

**Curriculum:**
- Sort training examples by current model confidence (hardest first, then randomise within a window).
- Implemented as a reordering pass over the CSV before `dif_train_classify_sequence`.

**Active learning (uncertainty sampling):**
- Sort unlabelled pool by `entropy(softmax(LLR))` — highest entropy = most uncertain.
- Expose `GET /uncertainty-rank` on `cypha_rest` that returns the top-N row indices from a provided feature matrix.

Both require only additions to the existing hot path — no changes to `CyphaInferModel` or the binary format.

**Status (2026-07-12): both SHIPPED, opt-in / default-off.** The claim above holds exactly as
written: neither feature touches `CyphaInferModel`, `CyphaDifMemoryState`, or the `.cypha`
binary format — both are pure additions (new free functions + one new REST route + one new
env-gated branch in the bench training loop).

- **Curriculum ordering.** `cypha::curriculum_order_ascending_confidence` (hardest-first by max
  softmax confidence) already existed (used by the CyphaLM token-curriculum pilot and by
  `/uncertainty-rank`'s `"curriculum"` mode); this pass added the missing "randomise within a
  window" half of the spec as `cypha::curriculum_order_windowed(confidences, n_rows, window, rng)`
  in `native/include/cypha/curriculum.hpp` / `native/src/curriculum.cpp`. It is wired into the
  actual CSV-driven DIF training hot path — `train_eval_vectors` in
  `native/src/bench/bench_domains.cpp`, which every tabular/vision bench domain (D03, D08, ...)
  funnels through as its `dif_train_step_vector`-per-row loop (the practical equivalent of
  `dif_train_classify_sequence` for CSV data; that function itself is currently only exercised by
  parity fixtures, not by any bench CLI) — via a new opt-in env gate, **`CYPHA_CURRICULUM_WINDOW`**
  (integer window size; unset or `<= 0` = off, byte-identical to the pre-existing per-pass shuffle,
  verified by re-running D03 twice with the var unset and diffing the output table — identical).
  When set, each pass re-scores every training row with the model's *current* confidence
  (`batch_llr_from_x` + `softmax_batch_reference`, both pre-existing), sorts hardest-first, and
  locally shuffles within contiguous chunks of `window` rows using a dedicated
  `std::mt19937` seeded independently of the main training RNG. Falls back to the original
  per-pass shuffle on the very first pass of a cold-start online model (no labels registered yet,
  so there is no confidence signal). New CTest `native_curriculum_window_smoke` covers determinism
  (same seed ⇒ identical order), the window-boundary invariant (each window is a permutation of
  its own hardest-first slice, not the whole array), and that windowing actually reorders relative
  to the strict hardest-first baseline.
- **Active learning / uncertainty ranking.** This was already fully shipped (see `d8271c4`/
  `5d24b02`, predating this doc's "future" listing — §6 had not been updated to reflect it): `GET`
  and `POST /uncertainty-rank` on `cypha_rest`, request body `{"rows": [[...], ...], "top_n": N,
  "temperature": T?, "curriculum": bool?}`, response
  `{"indices": [...], "entropies": [...], "confidences": [...], "top_n": N}` — `indices` sorted by
  descending `entropy(softmax(LLR/temperature))` (most uncertain first) by default, or by ascending
  confidence (hardest-first) when `"curriculum": true`. Covered by CTest `native_rest_uncertainty_rank`
  (integration test that starts `cypha_rest.exe` and hits the live endpoint). One audit fix this
  pass: the endpoint's entropy computation was a hand-duplicated copy of
  `cypha::row_entropy_from_probs` (`native/include/cypha/infer_cpu.hpp`, also used by the
  library's own `cypha::uncertainty_rank_indices` helper); `cypha_rest.cpp` now calls the shared
  function directly instead of reimplementing it, per this section's original design intent to
  reuse existing softmax/entropy math.
- **Small-scale sanity check (plumbing correctness, not a hyperparameter search):** D03 (iris +
  wine, `CYPHA_BENCH_FAST=1`, single epoch, seed 42) with `CYPHA_CURRICULUM_WINDOW` unset vs. `=8`:
  iris accuracy 0.867 → 0.967 (improved), wine 1.00 → 1.00 (unchanged, already at ceiling). Neutral-
  to-positive on this tiny synthetic slice, as expected for a plumbing check — not a claim of a
  general accuracy win.
- Full relevant suite green: `ctest --test-dir native/build_curriculum -R
  "curriculum|uncertainty|dif_train"` → 5/5 passed; full non-overnight suite
  (`ctest -E "overnight|cuda_bench|cell_hypothesis_overnight"`) → 164/164 passed (1 skipped,
  OpenSSL-gated `native_federated_tls_smoke`, expected).

---

## 7 — Export formats

### ONNX export
Export the inference path (encode → LLR → softmax) as an ONNX graph so the model can run in PyTorch, TensorFlow, or ONNX Runtime without `cypha_rest`.

- `batch_encode` maps to a matmul + `tanh`/`cos` non-linearity.
- `score_matrix_use_field` maps to a matmul + optional NIG field gates.
- **`cypha_onnx_export`** header-only writer shipped; full pipeline integration is future work.

**Caveat:** ONNX export only covers inference, not online training. For production serving without `cypha_rest`, ONNX is useful. For adaptive models that need to keep learning from new data, the native binary remains the right choice.

### GGUF / llama.cpp format
Longer-horizon: pack `enc_W`, `F_field`, class centroid tensors into a GGUF container so the model can be loaded by llama.cpp-style inference tools. Requires writing a GGUF serialiser (native).

---

## 8 — Distributed / federated training

The current model trains on a single machine with a single replay buffer. For multi-device or multi-process training:

**Parameter averaging:**
- Each worker trains independently for N steps.
- Workers share `world_mu` and class stats via a coordinator (Redis pub/sub or gRPC).
- The coordinator averages the updates and broadcasts back.
- Matches the existing `merge_state_into_root_for_save` / `from_cypha_root` round-trip.

**Federated learning:**
- Each client trains on private data; only gradient-equivalent stats (not raw data) are sent.
- `world_mu` deltas + class distribution changes are the federated payload.
- No raw samples leave the device.

Both require new network coordination code outside the native training core — the local training math in the C++ `cypha_core` library stays unchanged.

---

## 9 — Deprecations and clean-up (P7 complete)

| Item | Status (P7) |
|------|-------------|
| Legacy sigmoid (`gh_chi <= 0`) | **Removed** — `kInferWorldGateApiVersion=2`; throws when `gh_chi`/`gh_psi` ≤ 0 |
| Python FastAPI / PySide6 Studio (`cypha_studio/`) | **Removed** — `cypha_rest` + `cypha_qt_shell` are authoritative |
| `cypha_accel/` CuPy path | **Removed** — native `cypha::accel` (CUDA / parallel CPU) |
| `cypha_core`, `bench/` (Python), `cypha_lm/` Python packages | **Removed** — native binaries only (`bench/` configs via `cypha_bench_run`) |
| pytest CI gate (~274 tests) | **Removed** - **214 CTests** (`ctest -R native_`; see `scripts/cypha_native_validate_all.ps1` for the current authoritative count) gate releases |
| `run_all.py` bench orchestrator | **Removed** — `cypha_bench_run` |

---

## 10 -- RPSM matrix refactor + sequence layer (closed)

**Status:** Option A **shipped**; Option B **STOP / deprioritized (2026-07-18)** -- see [`RPSM_UPGRADE_PLAN.md`](archive/plans/RPSM_UPGRADE_PLAN.md), [`RPSM_SMALL_TIER_GATE_2026-07-18.md`](archive/reports/RPSM_SMALL_TIER_GATE_2026-07-18.md), and [`CYPHA_BILL_OF_WORK.md`](../CYPHA_BILL_OF_WORK.md) (open items). Do **not** treat "beat 2.873 via RPSM" as an open plan.

**Historical target:** Beat `hybrid_gria_lstm` D17 BPC **2.873** @ 300k -- **not met** by RPSM Option B. Hybrid is the living default; further gains go through predictive AC + hybrid quality under `cypha::Cypha`.

| Track | What | Status |
|-------|------|--------|
| **Option A** | Classifier matrix refactor (batched LLR/GEMM) | **Shipped** |
| **Option B** | RPSM sequence layer | **STOP** |

**Parallel track:** [Cell hypothesis testbench](research/upgrades/CELL_HYPOTHESIS_TESTBENCH.md) -- **closed**; H19 **~2.921 BPC** best cell -- [`CELL_SWEEP_SUMMARY_2026-07-18.md`](archive/reports/CELL_SWEEP_SUMMARY_2026-07-18.md).

**Native milestone:** Option A shipped; see [`CYPHA_FULL_CPP_FRAMEWORK_PLAN.md`](native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md).

---

## Horizon summary

| When | What | Evidence |
|------|------|----------|
| **Now -- living** | Hybrid **2.664 BPC** + predictive AC | Product spine under `cypha::Cypha` |
| **Now -- tuning** | Kernel LLR (Nystrom + RFF) -- 0a | Shipped; RFF auto-gamma closes XOR gap to **~2.7pp** |
| **Now -- shipped** | D10 ECG -- 0c | **90.11%** ECG5000 (2026-07-18) |
| **Production pin** | Hybrid D17 **2.664 BPC** (L2 + Wave2 BPTT) | Living default target @ 300k WikiText-2 |
| **Now -- shipped** | Qt UX 2a-2e; minimal Web UI -- 4 | Threaded train, charts, REST SPA |
| **Shipped (v2.3.25)** | Packaged AppImage / Windows bundle -- 3 | One Cypha release |
| **Done (policy)** | CUDA local-only validation -- 1 | No hosted GPU CI |
| **Human-only** | Paper arXiv upload | `arxiv_bundle` ready -- [`CYPHA_BILL_OF_WORK.md`](../CYPHA_BILL_OF_WORK.md) |
| **2-4 months** | Multi-model `cypha_rest` -- 5 | Deployment |
| **Closed** | RPSM Option B / cell sweep -- 10 | STOP; see archive |
| **4-8 months** | ONNX integration -- 7 | Curriculum/uncertainty already shipped |
| **Longer** | Federated; GGUF -- 8 / upgrades | Research |
