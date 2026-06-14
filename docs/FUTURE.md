# Future directions

Cypha's native port (M1-M6 + P7) is complete - inference, training, REST server, Qt shell, experiments DB, and parity fixtures all pass CI (**115 CTests**; **116** when d38 merges). Python runtime packages removed. This document records the most valuable next engineering directions, from near-term (months) to longer-horizon (quarters).

---

## �0 � Evidence-confirmed upgrades (post-diagnostic, highest priority)

These come directly from the 2026-05-30 diagnostic run
([`docs/reports/DIAGNOSTIC_REPORT.md`](reports/DIAGNOSTIC_REPORT.md)). Every
item has a measured effect size; nothing is speculative.

**RPSM roadmap (Option A + B):** see [�10](#10--rpsm-matrix-refactor--cyphalm-sequence-layer) and [`docs/research/upgrades/`](research/upgrades/README.md).

### �0a � Kernel LLR via Nystr�m approximation (SHIPPED � tuning continues)

**Evidence:** FDR=0.001 on XOR; `linear(h)=0.512` (chance) vs `kernel(h)=0.835`.
Nonlinearity gap = **32.3 pp** � unreachable with the current linear LLR discriminant.

**Shipped (2026-05-31):** Whitened Nystr�m features in native C++ (`native/src/kernel_memory.cpp`):
1. Reservoir landmarks (**M=256** default) with median-? RBF bandwidth.
2. `?(h) = K(h, landmarks) � K(landmarks, landmarks)^{-1/2}` via Cholesky whitening.
3. Online softmax gradient on `?(h)`; blended into `score_matrix` when kernel enabled.

**Measured gain (3 seeds, 8 passes, blend=1.0, M=256, replay off, 2026-06):** native linear **49.9%** ? **59.2%** (+ **+9.3 pp**). Still below sklearn RBF ceiling (~79% on raw XOR splits). `.cypha` kernel keys via `patch_kernel_into_root`; Qt shell + native `cypha_rest` train/infer/save/load wired. Bench domain **`d03_xor`** (`cypha_bench_run --domain-tag d03_xor`). Opt-in profile: `bench/config/kernel_llr_profile.json`. Full validate includes d03_xor fast smoke + REST kernel body test.

> **P7 note:** Python `cypha_core` / `KernelMemory` removed; native path is authoritative. Full fix taxonomy: [`research/upgrades/NONLINEAR_BOUNDARY.md`](research/upgrades/NONLINEAR_BOUNDARY.md).

### �0b � Auto-gamma for RFF bandwidth (SHIPPED � bench + studio + native fit)

**Evidence:** `D_rff` sweep showed high variance between D=256 and D=512, suggesting
the gamma bandwidth is not well-tuned for all datasets.

**Shipped (2026-06):** Native `PreprocessorState::auto_rff_gamma` + `estimate_rff_gamma_median_pairwise` in `fit_from_design_matrix`. Bench: native `BenchClassifier.prepare_encoder_from_data` wired via online train. Qt shell trainer: auto-? before online loop. CTest: **`native_preprocessor_fit`** (RFF fixture under `fixtures/preprocessor_fit_rff/`). **`auto_rff_gamma_cv`** grid CV also shipped.

> **P7 note:** Python `RFFEncoder.fit(X)` removed with `cypha_core`.

**Expected gain:** +2�4 pp on small-dimensional datasets (re-benchmark d01 small tasks to confirm).

### �0c � D10/D17 CellAI SSM investigation

**Evidence:**
- D10 ECG: 17�20% accuracy on 5-class temporal classification (chance = 20%); CellAI/SSM
  integration not yet tuned for this domain.
- D17 CyphaLM: **hybrid_gria_lstm @ 300k = 2.873 BPC** (beats bigram); GRIA-only stack still weaker.
- **D04 "33.2 bpc" was a benchmark bug** (wrong probability indexing in the legacy Python D04 domain, not a CyphaLM failure) � do not use it as evidence. Native D04 runs the full CyphaLM stack via `cypha_bench_run --domain 4`.

**What to do:** Instrument native SSM state (`cyphalm_ssm_diagnose`, `--ssm-diagnose` on bench) to verify that:
- State norms do not collapse or explode over long sequences.
- Multi-scale decay rates (?_fast=1.0, ?_slow=20.0) are appropriate for the domain.
- Output projections are properly connected to the expert routing head.

> **P7 note:** Python `CellAISSM` / `cypha_lm` packages removed; instrument native SSM via `cyphalm_bench_native` and CTests under `native_cyphalm_*`. Cell hypothesis sweep: [`research/upgrades/CELL_HYPOTHESIS_TESTBENCH.md`](research/upgrades/CELL_HYPOTHESIS_TESTBENCH.md).

---

## �1 � CUDA GPU path (`cypha::accel`)

**Status:** Native **CUDA** in `native/src/accel_cuda.cu` (pooled device memory + Bessel table for GH�NIG gate) plus `accel_backend.cpp`. Without `-DCYPHA_ENABLE_CUDA=ON`, the same APIs use **ISO C++** `std::thread`. **`infer_cpu`** routes **`batch_encode`**, **`score_matrix_use_field`**, and **`world_gate_vector_use_field`** through **`cypha::accel`** (CUDA when batch rows ? **`CYPHA_ACCEL_GPU_MIN_BATCH_ROWS`**, default **1** � GPU used for all n?1 when available). **`cuda_smoke`** checks encode / score / softmax / tanh gate / NIG gate vs references; **`--bench`** compares CUDA vs CPU refs when a GPU is present.

**Windows (native MSVC):** install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), then:
```powershell
cd native
cmake --preset windows-msvc-release -DCYPHA_ENABLE_CUDA=ON
cmake --build --preset windows-msvc-release-build
.\build-windows-msvc\Release\cuda_smoke.exe
```

**WSL2 / Linux:** install NVIDIA driver on Windows + CUDA toolkit inside WSL (`nvidia-cuda-toolkit` or NVIDIA's `.run` installer). Use preset **`wsl-gcc-release`** and add `-DCYPHA_ENABLE_CUDA=ON`. Set `-DCMAKE_CUDA_ARCHITECTURES=` to your GPU (e.g. `89`, `86`, `75`).

**Not supported:** MinGW cross-compiles cannot enable `CYPHA_ENABLE_CUDA` (CMake will error).

**CI:** CUDA is **not** gated in GitHub Actions — build and test locally with `-DCYPHA_ENABLE_CUDA=ON`, then run **`native_cuda_smoke`** and **`native_score_batch`**. The former **`windows_cuda_msvc`** and **`linux_cuda`** jobs were removed. See [`docs/native/ACCEL_CUDA.md`](native/ACCEL_CUDA.md).

**Performance:** profile with `./cuda_smoke --bench` on GPU; small batches may be CPU-faster due to launch overhead.

---

## �2 � Qt shell richer UX

The Qt shell (`cypha_qt_shell`) covers the full training/inference/registry workflow. Items **�2a�2e shipped** (2026 Q2); see [`native/qt/README.md`](../native/qt/README.md).

### 2a � Streaming training progress ? SHIPPED
Bulk native train runs on a **background `QThread`**. Live loss chart + rolling accuracy drain every 80 ms; cancel via atomic flag; final state synced on finish.

### 2b � Chart interactivity ? SHIPPED
Painted `SimpleLossChart` and optional **`QChartView`** (`-DCYPHA_QT_CHARTS=ON`): mouse-over tooltip, pan, scroll-wheel zoom (clamped to data extents). PNG/SVG/CSV export.

### 2c � Experiment run comparison view ? SHIPPED
Experiments panel **"Compare selected runs"** overlays `metrics_history` loss curves for 2+ runs (colour per run).

### 2d � Model card editor ? SHIPPED
**`card.json�`** dialog edits name, description, tags, task before registry register.

### 2e � Dark theme ? SHIPPED
**Fusion** palette toggle in Settings; persisted to `~/.cypha/shell_settings.json` and QSettings.

**Remaining Qt polish (optional):** richer compare-run statistics; experiment diff export; additional chart series types.

---

## �3 � Packaged standalone binary

Goal: a single distributable executable (`cypha_qt_shell`) with no external runtime dependencies.

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

## �4 � Web UI (browser-based Studio)

Native **`cypha_rest`** (cpp-httplib; see [`PORT_CONTRACT.md`](port/PORT_CONTRACT.md) �3) exposes the full REST API.

### Minimal SPA ? SHIPPED (2026 Q2)
Vanilla JS SPA at **`GET /`** � health, predict, update, models, session RNG. Static assets under `native/tools/static/`; optional embed via `-DCYPHA_EMBED_STATIC_UI=ON`. CTest **`native_rest_ui_smoke`**.

**Remaining (expansion):** live training charts, experiment browser, CyphaLM `/generate` chat pane, richer model registry UX.

**Option B � htmx + server-side HTML:** not started; lower priority now that minimal SPA ships.

---

## �5 � Multi-model serving in `cypha_rest`

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
- `g_models: std::unordered_map<std::string, CyphaInferModel>` � keyed by `name/version`.
- Per-model `std::mutex` for train vs infer serialisation.
- `POST /load` without a body ? load all models in the registry at startup.
- Optional LRU eviction (keep N most-recently-used models in RAM).

**Benefit:** one process can serve an A/B test or a production-then-staging model pair without two separate server instances.

---

## �6 � Curriculum / active learning

The current training loop is purely online with a simple replay buffer. Higher-quality training on skewed datasets can use:

**Curriculum:**
- Sort training examples by current model confidence (hardest first, then randomise within a window).
- Implemented as a reordering pass over the CSV before `dif_train_classify_sequence`.

**Active learning (uncertainty sampling):**
- Sort unlabelled pool by `entropy(softmax(LLR))` � highest entropy = most uncertain.
- Expose `GET /uncertainty-rank` on `cypha_rest` that returns the top-N row indices from a provided feature matrix.

Both require only additions to the existing hot path � no changes to `CyphaInferModel` or the binary format.

---

## �7 � Export formats

### ONNX export
Export the inference path (encode ? LLR ? softmax) as an ONNX graph so the model can run in PyTorch, TensorFlow, or ONNX Runtime without `cypha_rest`.

- `batch_encode` maps to a matmul + `tanh`/`cos` non-linearity.
- `score_matrix_use_field` maps to a matmul + optional NIG field gates.
- **`cypha_onnx_export`** header-only writer shipped; full pipeline integration is future work.

**Caveat:** ONNX export only covers inference, not online training. For production serving without `cypha_rest`, ONNX is useful. For adaptive models that need to keep learning from new data, the native binary remains the right choice.

### GGUF / llama.cpp format
Longer-horizon: pack `enc_W`, `F_field`, class centroid tensors into a GGUF container so the model can be loaded by llama.cpp-style inference tools. Requires writing a GGUF serialiser (native).

---

## �8 � Distributed / federated training

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

Both require new network coordination code outside the native training core � the local training math in the C++ `cypha_core` library stays unchanged.

---

## �9 � Deprecations and clean-up (P7 complete)

| Item | Status (P7) |
|------|-------------|
| Legacy sigmoid (`gh_chi <= 0`) | Remove with a version bump + changelog entry; unused in-tree |
| Python FastAPI / PySide6 Studio (`cypha_studio/`) | **Removed** � `cypha_rest` + `cypha_qt_shell` are authoritative |
| `cypha_accel/` CuPy path | **Removed** � native `cypha::accel` (CUDA / parallel CPU) |
| `cypha_core`, `bench/` (Python), `cypha_lm/` Python packages | **Removed** � native binaries only (`bench/` configs via `cypha_bench_run`) |
| pytest CI gate (~274 tests) | **Removed** - **115 CTests** (`ctest -R native_`; **116** when d38 merges) gate releases |
| `run_all.py` bench orchestrator | **Removed** � `cypha_bench_run` |

---

## �10 � RPSM matrix refactor + CyphaLM sequence layer

**Status:** Planned � specs in [`docs/research/upgrades/`](research/upgrades/README.md).  
**Target:** Beat `hybrid_gria_lstm` D17 BPC **2.873** @ 300k.

Two composed workstreams (neither replaces the other):

| Track | What | Depends on |
|-------|------|------------|
| **Option A** | CyphaDIF matrix refactor (?_mu / ?_var, batched LLR/GEMM) | Existing parity suite + new batched fixtures |
| **Option B** | RPSM sequence layer � CyphaDIF at level 0, hierarchy above | Option A + [RPSM_IMPLEMENTATION.md](research/upgrades/RPSM_IMPLEMENTATION.md) |

**Execution order:**

1. Option A � matrix refactor (parity-validated, no behaviour change)  
2. Nystr�m kernel LLR into A � **partially shipped** (�0a); tuning continues  
3. Option B � RPSM sequence layer in native CyphaLM  
4. Global memory � Izaac episodic store + working memory + GMM world model  
5. D17 benchmark vs hybrid baseline  

**Parallel track:** [Cell hypothesis testbench](research/upgrades/CELL_HYPOTHESIS_TESTBENCH.md) � 28 recurrent-cell variants.

**Native milestone:** Option A is the next major `cypha_core` refactor � see [`CYPHA_FULL_CPP_FRAMEWORK_PLAN.md`](native/CYPHA_FULL_CPP_FRAMEWORK_PLAN.md).

---

## Horizon summary

| When | What | Evidence |
|------|------|----------|
| **Now � tuning** | Kernel LLR (Nystr�m) � �0a | Shipped; ~18 pp gap vs sklearn RBF on XOR |
| **Now � shipped** | Auto-gamma RFF � �0b | Native fit + Qt + bench |
| **Now** | D10/D17 CellAI SSM � �0c | D10: 17�20% ECG; hybrid D17 **2.873 BPC** |
| **Now � shipped** | Qt UX �2a�2e; minimal Web UI �4 | Threaded train, charts, compare runs, dark theme, REST SPA |
| **Weeks** | Packaged AppImage / Windows bundle � �3 | `packaging/` scripts shipped |
| **Ongoing** | CUDA CI + profiling � �1 | Blocking jobs green |
| **2�4 months** | Multi-model `cypha_rest` � �5 | Deployment |
| **4�8 months** | RPSM Option A ? B � �10 | Beat 2.873 BPC target |
| **4�8 months** | Curriculum / active learning; ONNX � �6/7 | Research |
| **Longer** | Cell hypothesis sweep; federated; GGUF � �8 / upgrades | Research |
