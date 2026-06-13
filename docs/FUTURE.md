# Future directions

Cypha's native port (M1ùM6 + P7) is complete ù inference, training, REST server, Qt shell, experiments DB, and parity fixtures all pass CI (**53 CTests**). Python runtime packages removed. This document records the most valuable next engineering directions, from near-term (months) to longer-horizon (quarters).

---

## ù0 ù Evidence-confirmed upgrades (post-diagnostic, highest priority)

These come directly from the 2026-05-30 diagnostic run
([`docs/reports/DIAGNOSTIC_REPORT.md`](reports/DIAGNOSTIC_REPORT.md)). Every
item has a measured effect size; nothing is speculative.

### ù0a ù Kernel LLR via Nystrùm approximation (SHIPPED ù tuning continues)

**Evidence:** FDR=0.001 on XOR; `linear(h)=0.512` (chance) vs `kernel(h)=0.835`.
Nonlinearity gap = **32.3 pp** ù unreachable with the current linear LLR discriminant.

**Shipped (2026-05-31):** Whitened Nystrùm features in native C++ (`native/src/kernel_memory.cpp`):
1. Reservoir landmarks (**M=256** default) with median-? RBF bandwidth.
2. `?(h) = K(h, landmarks) ù K(landmarks, landmarks)^{-1/2}` via Cholesky whitening.
3. Online softmax gradient on `?(h)`; blended into `score_matrix` when `use_kernel_llr=True`.

**Measured gain (3 seeds, 8 passes, blend=1.0, M=256, replay off, 2026-06):** native linear **49.9%** ? **59.2%** (? **+9.3 pp**). Still below sklearn RBF ceiling (~79% on raw XOR splits). `.cypha` kernel keys via `patch_kernel_into_root`; Qt shell + native `cypha_rest` train/infer/save/load wired. Bench domain **`d03_xor`** (`cypha_bench_run --domain-tag d03_xor`). Opt-in profile: `bench/config/kernel_llr_profile.json`. Full validate includes d03_xor fast smoke + REST kernel body test.

> **P7 note:** Python `cypha_core` / `KernelMemory` removed; native path is authoritative.

### ù0b ù Auto-gamma for RFF bandwidth (SHIPPED ù bench + studio + native fit)

**Evidence:** `D_rff` sweep showed high variance between D=256 and D=512, suggesting
the gamma bandwidth is not well-tuned for all datasets.

**Shipped (2026-06):** Native `PreprocessorState::auto_rff_gamma` + `estimate_rff_gamma_median_pairwise` in `fit_from_design_matrix`. Bench: native `BenchClassifier.prepare_encoder_from_data` wired via online train. Qt shell trainer: auto-? before online loop. CTest: **`native_preprocessor_fit`** (RFF fixture under `fixtures/preprocessor_fit_rff/`).

> **P7 note:** Python `RFFEncoder.fit(X)` / `tests/test_rff_auto_gamma.py` removed with `cypha_core`.

**Expected gain:** +2ù4 pp on small-dimensional datasets (re-benchmark d01 small tasks to confirm).

### ù0c ù D10/D17 CellAI SSM investigation

**Evidence:**
- D10 ECG: 17ù20% accuracy on 5-class temporal classification (chance = 20%); CellAI/SSM
  integration not yet tuned for this domain.
- D17 CyphaLM held-out BPC = 4.50 vs bigram baseline 3.69 ù LM stack produces a real
  distribution but is weaker than bigram on held-out text.
- **D04 "33.2 bpc" was a benchmark bug** (wrong probability indexing in the legacy Python D04 domain, not a CyphaLM failure) ù do not use it as evidence. Native D04 runs the full CyphaLM stack via `cypha_bench_run --domain 4`.

**What to do:** Run Phase 5 of the diagnostic plan on the CellAI SSM independently
of the CyphaDIF classifier. Instrument native SSM state to verify that:
- State norms do not collapse or explode over long sequences.
- Multi-scale decay rates (?_fast=1.0, ?_slow=20.0) are appropriate for the domain.
- Output projections are properly connected to the expert routing head.
- Fix native D04 probability indexing before interpreting historical pre-fix numbers.

> **P7 note:** Python `CellAISSM` / `cypha_lm` packages removed; instrument native SSM via `cyphalm_bench_native` and CTests under `native_cyphalm_*`.

---

## ù1 ù CUDA GPU path (`cypha::accel`)

**Status:** Native **CUDA** in `native/src/accel_cuda.cu` (pooled device memory + Bessel table for GHùNIG gate) plus `accel_backend.cpp`. Without `-DCYPHA_ENABLE_CUDA=ON`, the same APIs use **ISO C++** `std::thread`. **`infer_cpu`** routes **`batch_encode`**, **`score_matrix_use_field`**, and **`world_gate_vector_use_field`** through **`cypha::accel`** (CUDA when batch rows ? **`CYPHA_ACCEL_GPU_MIN_BATCH_ROWS`**, default 16). **`cuda_smoke`** checks encode / score / softmax / tanh gate / NIG gate vs references; **`--bench`** compares CUDA vs CPU refs when a GPU is present.

**Windows (native MSVC):** install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), then:
```powershell
cd native
cmake --preset windows-msvc-release -DCYPHA_ENABLE_CUDA=ON
cmake --build --preset windows-msvc-release-build
.\build-windows-msvc\Release\cuda_smoke.exe
```

**WSL2 / Linux:** install NVIDIA driver on Windows + CUDA toolkit inside WSL (`nvidia-cuda-toolkit` or NVIDIA's `.run` installer). Use preset **`wsl-gcc-release`** and add `-DCYPHA_ENABLE_CUDA=ON`. Set `-DCMAKE_CUDA_ARCHITECTURES=` to your GPU (e.g. `89`, `86`, `75`).

**Not supported:** MinGW cross-compiles cannot enable `CYPHA_ENABLE_CUDA` (CMake will error).

**CI:** **`windows_cuda_msvc`** and **`linux_cuda`** are **blocking** jobs on every push ù Jimver nvcc install, compile with `-DCYPHA_ENABLE_CUDA=ON`, run **`native_cuda_smoke`** and **`native_score_batch`**. Runners without a GPU still pass (CPU thread fallback); **`native_cuda_bench`** is optional when hardware is present. See [`docs/native/ACCEL_CUDA.md`](native/ACCEL_CUDA.md).

**Performance:** profile with `./cuda_smoke --bench` on GPU; small batches may be CPU-faster due to launch overhead.

---

## ù2 ù Qt shell richer UX

The current Qt shell (`cypha_qt_shell`) covers the full training/inference/registry workflow. Remaining UX gaps, ordered by value:

### 2a ù Streaming training progress (high value)
The current bulk-native-train loop calls `QCoreApplication::processEvents()` per row but writes the final chart update only at the end. For large CSVs (>10k rows) the window appears frozen.

- **Fix:** Move bulk training to a `QThread` worker; emit `lossReported(step, loss)` and `valAccReported(step, acc)` signals; update chart and progress panel on the main thread via `QMetaObject::invokeMethod`.
- Also enables a **live loss chart** that updates every N steps during training.

### 2b ù Chart interactivity (medium value)
The painted `SimpleLossChart` is read-only. Add:
- Mouse-over tooltip showing `(step, loss)` at the nearest point.
- Click-drag pan and scroll-wheel zoom (clamp to data extents).
- Optional: `QChartView` with `-DCYPHA_QT_CHARTS=ON` already does this when Qt Charts is installed.

### 2c ù Experiment run comparison view (medium value)
The M6 Experiments panel lists runs. Add a "Compare runs" view that plots the `metrics_history` loss curves for multiple selected runs on a single `SimpleLossChart`, with one colour per run. Uses `experiment_db_crud::compare_runs`.

### 2d ù Model card editor (low-medium value)
Allow editing `card.json` fields (name, description, tags, task) from within the shell before registering a model.

### 2e ù Dark theme (polish)
`QApplication::setStyle("Fusion")` with a dark palette. Or expose a settings JSON (`~/.cypha/shell_settings.json`) with a theme toggle.

---

## ù3 ù Packaged standalone binary

Goal: a single distributable executable (`cypha_qt_shell`) with no external runtime dependencies.

**Linux AppImage:**
1. Build with `DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_QT=ON`.
2. Use `linuxdeploy` + `linuxdeploy-plugin-qt` to bundle Qt .so files.
3. Optionally bundle a stripped `cypha_rest` as a sidecar.

**Windows `.exe` (single file, no installer):**
1. MinGW cross-build (`cmake --preset mingw-w64-cross`) already produces statically-linked binaries without `libgcc`/`libstdc++` dependencies.
2. Qt itself is a shared DLL dependency ù either distribute the Qt DLLs alongside, or switch to a static Qt build.
3. Bundle with `windeployqt` for an easily-distributed folder; package with NSIS or WiX for a `.msi`.

Release bundles: [`packaging/`](../../packaging/) install scripts + `scripts/package_release_*.sh`.

**macOS:**
Build natively on macOS with Qt 6 from Homebrew; `macdeployqt` produces a `.app` bundle.

---

## ù4 ù Web UI (browser-based Studio)

Native **`cypha_rest`** (cpp-httplib; see [`PORT_CONTRACT.md`](port/PORT_CONTRACT.md) ù3) exposes the full REST API. A browser front-end would complement the Qt shell for headless server deployments.

**Option A ù Minimal SPA (React/Vue):**
- Single-page app that calls `/health`, `/predict`, `/update`, `/models`, `/session/rng`.
- Served as a static bundle embedded in `cypha_rest` (cpp-httplib handles static file serving).
- Build step: `npm run build` ? `native/tools/static/` ? CMake `target_compile_definitions(cypha_rest PRIVATE CYPHA_EMBED_STATIC_UI)`.

**Option B ù htmx + server-side HTML (simpler):**
- No JS build step; small templates rendered by cpp-httplib.
- Fast to prototype; harder to make interactive for live loss charts.

**Priority:** Option A is higher value if there's a need for a GUI on servers that can't run Qt (headless Linux boxes, cloud VMs). Not needed if the Qt shell + `cypha_rest` CLI covers all deployment targets.

---

## ù5 ù Multi-model serving in `cypha_rest`

**Current:** `cypha_rest` serves one model at a time (load-then-predict; hot-swap via `POST /load`).

**Target architecture:**
```
cypha_rest --registry <root> --listen 0.0.0.0:8765
  GET  /models          ? list all registered models
  POST /predict         ? body: { "model": "<name>/<version>", "input": [...] }
  POST /update          ? body: { "model": "...", "input": [...], "correct_label": "..." }
  POST /load            ? hot-swap active model (for backward compat)
```

**Implementation sketch:**
- `g_models: std::unordered_map<std::string, CyphaInferModel>` ù keyed by `name/version`.
- Per-model `std::mutex` for train vs infer serialisation.
- `POST /load` without a body ? load all models in the registry at startup.
- Optional LRU eviction (keep N most-recently-used models in RAM).

**Benefit:** one process can serve an A/B test or a production-then-staging model pair without two separate server instances.

---

## ù6 ù Curriculum / active learning

The current training loop is purely online with a simple replay buffer. Higher-quality training on skewed datasets can use:

**Curriculum:**
- Sort training examples by current model confidence (hardest first, then randomise within a window).
- Implemented as a reordering pass over the CSV before `dif_train_classify_sequence`.

**Active learning (uncertainty sampling):**
- Sort unlabelled pool by `entropy(softmax(LLR))` ù highest entropy = most uncertain.
- Expose `GET /uncertainty-rank` on `cypha_rest` that returns the top-N row indices from a provided feature matrix.

Both require only additions to the existing hot path ù no changes to `CyphaInferModel` or the binary format.

---

## ù7 ù Export formats

### ONNX export
Export the inference path (encode ? LLR ? softmax) as an ONNX graph so the model can run in PyTorch, TensorFlow, or ONNX Runtime without `cypha_rest`.

- `batch_encode` maps to a matmul + `tanh`/`cos` non-linearity.
- `score_matrix_use_field` maps to a matmul + optional NIG field gates.
- Native ONNX serialiser is future work; Python `torch.onnx.export` path removed with `cypha_core` (P7).

**Caveat:** ONNX export only covers inference, not online training. For production serving without `cypha_rest`, ONNX is useful. For adaptive models that need to keep learning from new data, the native binary remains the right choice.

### GGUF / llama.cpp format
Longer-horizon: pack `enc_W`, `F_field`, class centroid tensors into a GGUF container so the model can be loaded by llama.cpp-style inference tools. Requires writing a GGUF serialiser (native).

---

## ù8 ù Distributed / federated training

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

Both require new network coordination code outside the native training core ù the local training math in the C++ `cypha_core` library stays unchanged.

---

## ù9 ù Deprecations and clean-up (P7 complete)

| Item | Status (P7) |
|------|-------------|
| Legacy sigmoid (`gh_chi <= 0`) | Remove with a version bump + changelog entry; unused in-tree |
| Python FastAPI / PySide6 Studio (`cypha_studio/`) | **Removed** ù `cypha_rest` + `cypha_qt_shell` are authoritative |
| `cypha_accel/` CuPy path | **Removed** ù native `cypha::accel` (CUDA / parallel CPU) |
| `cypha_core`, `bench/` (Python), `cypha_lm/` Python packages | **Removed** ó native binaries only (`bench/` configs via `cypha_bench_run`) |
| pytest CI gate (~274 tests) | **Removed** ù **53 CTests** (`ctest -R native_`) gate releases |
| `run_all.py` bench orchestrator | **Removed** ù `cypha_bench_run` |

---

## Horizon summary

| When | What | Evidence |
|------|------|----------|
| **Now ù highest priority** | Kernel LLR (Nystrùm) ù ù0a | Diagnostic: 32.3 pp gap on XOR; FDR=0.001 |
| **Now** | Auto-gamma RFF bandwidth ù ù0b | Diagnostic: sweep variance; expected +2ù4 pp |
| **Now** | D10/D17 CellAI SSM investigation ù ù0c | D10: 17ù20% ECG; D17: 4.50 bpc above bigram |
| **Weeks** | Qt shell streaming training thread; chart interactivity ù ù2a/2b | UX |
| **Weeks** | Packaged AppImage (Linux) / NSIS installer (Windows) ù ù3 | Distribution |
| **1ù2 months** | CUDA CI matrix job; second CUDA stream ù ù1 | Performance |
| **2ù4 months** | Multi-model `cypha_rest`; Web UI (SPA or htmx) ù ù4/5 | Deployment |
| **4ù8 months** | Curriculum / active learning; ONNX export ù ù6/7 | Research |
| **Longer** | Federated training; GGUF; full Vulkan/CUDA backend ù ù8 | Research |
