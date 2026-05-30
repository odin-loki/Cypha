# Future directions

Cypha's native port (M1–M6) is complete — inference, training, REST server, Qt shell, experiments DB, and parity fixtures all pass CI. This document records the most valuable next engineering directions, from near-term (months) to longer-horizon (quarters).

---

## §1 — CUDA GPU path (`cypha::accel`)

**Status:** Native **CUDA** in `native/src/accel_cuda.cu` (pooled device memory + Bessel table for GH–NIG gate) plus `accel_backend.cpp`. Without `-DCYPHA_ENABLE_CUDA=ON`, the same APIs use **ISO C++** `std::thread`. **`infer_cpu`** routes **`batch_encode`**, **`score_matrix_use_field`**, and **`world_gate_vector_use_field`** through **`cypha::accel`** (CUDA when batch rows ≥ **`CYPHA_ACCEL_GPU_MIN_BATCH_ROWS`**, default 16). **`cuda_smoke`** checks encode / score / softmax / tanh gate / NIG gate vs references; **`--bench`** compares CUDA vs CPU refs when a GPU is present.

**Windows (native MSVC):** install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), then:
```powershell
cd native
cmake --preset windows-msvc-release -DCYPHA_ENABLE_CUDA=ON
cmake --build --preset windows-msvc-release-build
.\build-windows-msvc\Release\cuda_smoke.exe
```

**WSL2 / Linux:** install NVIDIA driver on Windows + CUDA toolkit inside WSL (`nvidia-cuda-toolkit` or NVIDIA’s `.run` installer). Use preset **`wsl-gcc-release`** and add `-DCYPHA_ENABLE_CUDA=ON`. Set `-DCMAKE_CUDA_ARCHITECTURES=` to your GPU (e.g. `89`, `86`, `75`).

**Not supported:** MinGW cross-compiles cannot enable `CYPHA_ENABLE_CUDA` (CMake will error).

**Performance:** profile with `./cuda_smoke --bench` on GPU; small batches may be CPU-faster due to launch overhead.

---

## §2 — Qt shell richer UX

The current Qt shell (`cypha_qt_shell`) covers the full training/inference/registry workflow. Remaining UX gaps, ordered by value:

### 2a — Streaming training progress (high value)
The current bulk-native-train loop calls `QCoreApplication::processEvents()` per row but writes the final chart update only at the end. For large CSVs (>10k rows) the window appears frozen.

- **Fix:** Move bulk training to a `QThread` worker; emit `lossReported(step, loss)` and `valAccReported(step, acc)` signals; update chart and progress panel on the main thread via `QMetaObject::invokeMethod`.
- Also enables a **live loss chart** that updates every N steps during training.

### 2b — Chart interactivity (medium value)
The painted `SimpleLossChart` is read-only. Add:
- Mouse-over tooltip showing `(step, loss)` at the nearest point.
- Click-drag pan and scroll-wheel zoom (clamp to data extents).
- Optional: `QChartView` with `-DCYPHA_QT_CHARTS=ON` already does this when Qt Charts is installed.

### 2c — Experiment run comparison view (medium value)
The M6 Experiments panel lists runs. Add a "Compare runs" view that plots the `metrics_history` loss curves for multiple selected runs on a single `SimpleLossChart`, with one colour per run. Uses `experiment_db_crud::compare_runs`.

### 2d — Model card editor (low-medium value)
Allow editing `card.json` fields (name, description, tags, task) from within the shell before registering a model.

### 2e — Dark theme (polish)
`QApplication::setStyle("Fusion")` with a dark palette. Or expose a settings JSON (`~/.cypha/shell_settings.json`) with a theme toggle.

---

## §3 — Packaged standalone binary

Goal: a single distributable executable (`cypha_qt_shell`) with no external runtime dependencies.

**Linux AppImage:**
1. Build with `DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_QT=ON`.
2. Use `linuxdeploy` + `linuxdeploy-plugin-qt` to bundle Qt .so files.
3. Optionally bundle a stripped `cypha_rest` as a sidecar.

**Windows `.exe` (single file, no installer):**
1. MinGW cross-build (`cmake --preset mingw-w64-cross`) already produces statically-linked binaries without `libgcc`/`libstdc++` dependencies.
2. Qt itself is a shared DLL dependency — either distribute the Qt DLLs alongside, or switch to a static Qt build.
3. Bundle with `windeployqt` for an easily-distributed folder; package with NSIS or WiX for a `.msi`.

**macOS:**
Build natively on macOS with Qt 6 from Homebrew; `macdeployqt` produces a `.app` bundle.

---

## §4 — Web UI (browser-based Studio)

The `cypha_studio/server/api.py` FastAPI layer already exposes a complete REST API. A browser front-end would replace the PySide6 GUI for headless server deployments.

**Option A — Minimal SPA (React/Vue):**
- Single-page app that calls `/health`, `/predict`, `/update`, `/models`, `/session/rng`.
- Served as a static bundle embedded in `cypha_rest` (cpp-httplib handles static file serving).
- Build step: `npm run build` → `native/tools/static/` → CMake `target_compile_definitions(cypha_rest PRIVATE CYPHA_EMBED_STATIC_UI)`.

**Option B — htmx + server-side HTML (simpler):**
- No JS build step; small templates rendered by cpp-httplib.
- Fast to prototype; harder to make interactive for live loss charts.

**Priority:** Option A is higher value if there's a need for a GUI on servers that can't run Qt (headless Linux boxes, cloud VMs). Not needed if the Qt shell + `cypha_rest` CLI covers all deployment targets.

---

## §5 — Multi-model serving in `cypha_rest`

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

## §6 — Curriculum / active learning

The current training loop is purely online with a simple replay buffer. Higher-quality training on skewed datasets can use:

**Curriculum:**
- Sort training examples by current model confidence (hardest first, then randomise within a window).
- Implemented as a reordering pass over the CSV before `dif_train_classify_sequence`.

**Active learning (uncertainty sampling):**
- Sort unlabelled pool by `entropy(softmax(LLR))` — highest entropy = most uncertain.
- Expose `GET /uncertainty-rank` on `cypha_rest` that returns the top-N row indices from a provided feature matrix.

Both require only additions to the existing hot path — no changes to `CyphaInferModel` or the binary format.

---

## §7 — Export formats

### ONNX export
Export the inference path (encode → LLR → softmax) as an ONNX graph so the model can run in PyTorch, TensorFlow, or ONNX Runtime without `cypha_rest`.

- `batch_encode` maps to a matmul + `tanh`/`cos` non-linearity.
- `score_matrix_use_field` maps to a matmul + optional NIG field gates.
- Python: use `torch.onnx.export` with a traced wrapper around `CyphaDIF.infer`.
- Native: would require an ONNX protobuf serialiser; simpler to do from Python.

**Caveat:** ONNX export only covers inference, not online training. For production serving without `cypha_rest`, ONNX is useful. For adaptive models that need to keep learning from new data, the native binary remains the right choice.

### GGUF / llama.cpp format
Longer-horizon: pack `enc_W`, `F_field`, class centroid tensors into a GGUF container so the model can be loaded by llama.cpp-style inference tools. Requires writing a GGUF serialiser (or adapting an existing Python binding).

---

## §8 — Distributed / federated training

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

Both require new network coordination code outside `cypha_core` — the local training math stays unchanged.

---

## §9 — Deprecations and clean-up (when ready)

| Item | Action |
|------|--------|
| Legacy sigmoid (`gh_chi <= 0`) | Remove with a version bump + changelog entry; unused in-tree |
| Python FastAPI server in production | Keep as golden reference; mark clearly as "reference only" once Qt shell covers all Studio workflows |
| `cypha_accel/` CuPy path | Keep for Python-side profiling; native `cypha::accel` covers batch hot paths |
| `test_cypha.py` (root) | Fold into `pytest tests/` with a `@slow` marker; remove the custom runner |
| `cypha_studio/test_cypha_studio.py` | Same — already mirrored by `tests/test_cypha_studio_runner.py`; keep as a convenience script |

---

## Horizon summary

| When | What |
|------|------|
| **Weeks** | Optional: second CUDA stream / pinned host memory for HPC serving |
| **1–2 months** | Qt shell streaming training thread; chart interactivity; packaged AppImage |
| **2–4 months** | Multi-model `cypha_rest`; Web UI (SPA or htmx) |
| **4–8 months** | Curriculum / active learning; ONNX export |
| **Longer** | Federated training; GGUF; full Vulkan/CUDA backend |
