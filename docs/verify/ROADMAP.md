# Roadmap — reference → profiled → GPU experiments → native port

The **Python tree** is the spec: tests, parity blobs, fixture generators. The **product hot path** is in C++/Qt. Python stays for research and golden output generation.

---

## Where we are (current)

**All milestones M1–M6 are complete.** The full native stack is built and CI-gated:

| Block | Status |
|-------|--------|
| Inference kernel (encode + LLR + GH + softmax) | ✅ parity vs `expected.npz` |
| Registry + preprocessor contract (fit, transform, CSV load) | ✅ parity fixtures |
| Online `train_step` (DIF, GH, replay, NIG, context, OOD) | ✅ parity fixtures |
| Regression stack (MKE/RFF/two-stage/ridge/EMA) | ✅ parity fixtures |
| `cypha_rest` native server | ✅ JSON-compatible with FastAPI reference |
| Qt shell (`cypha_qt_shell`) | ✅ full training/inference/registry/plots/experiments UI |
| Preprocessor fit from Qt (scale + PCA, no Python) | ✅ `fit_from_design_matrix` + save `preprocessor.json` |
| Dataset panel (column picker, preview, val split) | ✅ |
| Experiments DB (SQLite, M6) | ✅ `experiment_db_crud`, Qt M6 panel |
| Deterministic REST replay (`/session/rng`) | ✅ cross-runtime state transfer |
| Autoregressive / generation path | ✅ native `generation_parity`, CTest |
| Test suite | ✅ **52 CTest** + **~274 pytest** (GitHub Actions CI — all four jobs green incl. MSVC CUDA) |

---

## Phase 0 — Debugging baseline ✅

- `Cypha.py` core: 54 deterministic checks + parity fixtures.
- Studio pipeline: 48 checks + pytest API smoke.
- Benchmark + sklearn CV as regression smoke.

---

## Phase 1 — Locking behaviour before native ✅

All tests committed and green:

- `InferenceEngine` vs raw `CyphaDIF.infer` — same label + conf.
- `score_matrix(..., use_field=True/False)` — finite LLR, consistent shapes.
- Preprocessor + trainer edge cases.

---

## Phase 2 — Profile on real data (CPU) ✅

Profiling infrastructure complete:

- `python scripts/profile_real_datasets.py` — sklearn tabular; cumtime identifies GEMM, RFF, NIG bottlenecks.
- `scripts/gpu_microbench.py` + `scripts/gpu_fullbench.py` — CuPy/CPU timing for encode+LLR+softmax+gate.

---

## Phase 3 — GPU (Python accel path) ✅

`cypha_accel` provides CuPy-accelerated `fused_score_llr`, `project_features`, `softmax_rows_llr` — CPU NumPy fallback when CuPy is absent. GPU microbench and full pipeline bench are scripted.

Native **`cypha::accel`**: optional **CUDA** (`native/src/accel_cuda.cu`, `-DCYPHA_ENABLE_CUDA=ON`) or **parallel CPU** (`std::thread` in `accel_backend.cpp`). CI builds without CUDA; **`cuda_smoke`** still passes on CPU threads.

---

## Phase 4 — Native port (M1–M6) ✅

All milestones complete — see [`PORT_FULL_STACK.md`](../port/PORT_FULL_STACK.md) for the full per-milestone record.

---

## Phase 5 — Current engineering horizon

The native hot path is complete. **Highest priority:** **Kernel LLR (Nyström)** — closes the ~32 pp gap on nonlinear boundaries (XOR FDR=0.001: kernel h=0.835 vs linear h=0.512). Confirmed in diagnostics and [`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md). Python-only today; native port follows once the reference kernel path is frozen.

Secondary priorities (depth and distribution):

1. **CUDA in CI (optional)** — add a matrix job with NVIDIA runner + `-DCYPHA_ENABLE_CUDA=ON` to exercise `cuda_smoke --bench`. See [`docs/FUTURE.md`](../FUTURE.md) §1.

2. **Qt shell polish** — streaming progress updates during long CSV training, chart zoom/pan, optional dark theme, export to ONNX. See [`docs/FUTURE.md`](../FUTURE.md) §2.

3. **Packaged binary** — single-binary Qt shell (static Qt + `cypha_core`) with no runtime dependencies; Windows `.msi` / Linux AppImage. See [`docs/FUTURE.md`](../FUTURE.md) §3.

4. **Web UI** — lightweight REST front-end replacing PySide6 for headless server deployments. See [`docs/FUTURE.md`](../FUTURE.md) §4.

5. **Multi-model serving** — `cypha_rest` serving N models in parallel; hot-swap without restart. See [`docs/FUTURE.md`](../FUTURE.md) §5.

---

## Phase 6 — CyphaLM research prototype

**`cypha_lm/`** — language-model stack (Izaac embeddings, NIG experts, GRIA projection, CellAI SSM). Not in CI; run `pytest cypha_lm/ -v` locally.

| Metric | Value | Notes |
|--------|-------|-------|
| **D17 held-out BPC** | **4.50** | Gutenberg; bigram baseline 3.69 |
| **D04 “33.2 bpc”** | benchmark bug | Wrong prob indexing on `CyphaDIF + CharNgramEncoder` — not CyphaLM |

See [`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md) § Language model and [`CHANGELOG.md`](../../CHANGELOG.md) §1.0.0 (D04 note).

---

## Doc index

| File | Purpose |
|------|---------|
| [VERIFICATION_STATUS.md](VERIFICATION_STATUS.md) | Test counts, coverage snapshot, known gaps |
| [PORT_CONTRACT.md](../port/PORT_CONTRACT.md) | Frozen binary/REST contracts |
| [PORT_FULL_STACK.md](../port/PORT_FULL_STACK.md) | Per-milestone record (M1–M6) |
| [FUTURE.md](../FUTURE.md) | Future directions in depth |
| [RESEARCH_STATUS.md](../RESEARCH_STATUS.md) | Benchmark journal, D04/D17, Kernel LLR priority |
| [VERIFY_PLAN.md](VERIFY_PLAN.md) | WSL, benchmark, cProfile workflow |
| [MAINTENANCE.md](MAINTENANCE.md) | When to regen fixtures, rebuild native, sync DDL |
