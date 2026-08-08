# Roadmap — reference → profiled → GPU experiments → native port

The **committed parity fixtures** and **PORT_CONTRACT** are the spec. The **product hot path** is C++/Qt.

---

## Where we are (current)

**All milestones M1–M6 are complete.** The full native stack is built and CI-gated:

| Block | Status |
|-------|--------|
| Inference kernel (encode + LLR + GH + softmax) | ✅ parity vs `expected.npz` |
| Registry + preprocessor contract (fit, transform, CSV load) | ✅ parity fixtures |
| Online `train_step` (DIF, GH, replay, NIG, context, OOD) | ✅ parity fixtures |
| Regression stack (MKE/RFF/two-stage/ridge/EMA) | ✅ parity fixtures |
| `cypha_rest` native server | ✅ JSON per **PORT_CONTRACT** §3 |
| Qt shell (`cypha_qt_shell`) | ✅ training/inference/registry/plots/experiments UI |
| Preprocessor fit from Qt (scale + PCA) | ✅ `fit_from_design_matrix` + save `preprocessor.json` |
| Experiments DB (SQLite, M6) | ✅ `experiment_db_crud`, Qt M6 panel |
| Autoregressive / generation path | ✅ native `generation_parity`, CTest |
| Test suite | ✅ **214 CTests** (`ctest -R native_`) — GitHub Actions two blocking jobs (`build_and_test`, `windows_msvc`) |

---

## Phase 0 — Debugging baseline ✅

- `cypha_core`: deterministic CTests + parity fixtures.
- REST + Qt: native smokes.
- **`cypha_bench_run`** + sklearn CV as regression smoke.

---

## Phase 1 — Locking behaviour before native ✅

Committed CTests green: inference vs fixtures, `score_matrix` modes, preprocessor + trainer edge cases.

---

## Phase 2 — Profile on real data (CPU) ✅

- Native bench/tune binaries identify GEMM, RFF, NIG bottlenecks.
- **`xor_kernel_bench`** — kernel LLR XOR profile.

---

## Phase 3 — GPU (native accel) ✅

Native **`cypha::accel`**: optional **CUDA** (`-DCYPHA_ENABLE_CUDA=ON`) or parallel CPU. **CUDA is local-only** — no GitHub Actions CUDA jobs; validate on your own machine before merging accel changes.

---

## Phase 4 — Native port (M1–M6) ✅

All milestones complete — see [`PORT_FULL_STACK.md`](../port/PORT_FULL_STACK.md).

---

## Phase 5 — Current engineering horizon

Kernel LLR (Nyström) shipped in C++ with **`xor_kernel_bench`** (+10.6 pp XOR gain at M=256). Secondary priorities: Qt polish, packaged binaries, web UI, multi-model serving — [`docs/FUTURE.md`](../FUTURE.md). (CUDA CI is not on the horizon — local-only policy.)

---

## Phase 6 -- Sequence research (hybrid production + predictive AC)

Native sequence stack lives in **`cypha_core`** (`cypha_lm_native` is an INTERFACE alias). Bench CLI: **`cyphalm_bench_native`** (default `--mode hybrid`). Local gate: `ctest --test-dir native/build -R 'native_cyphalm|native_predictive_codec' --output-on-failure`.

| Metric | Value | Notes |
|--------|-------|-------|
| **Living sequence default** | **Hybrid GRIA+LSTM** | `Cypha::init_default_sequence` → `apply_hybrid_production_recipe` -- [`ONE_CYPHA_CUTOVER.md`](../reports/ONE_CYPHA_CUTOVER.md) |
| **D17 hybrid BPC @ 300k** | **2.664** (L2 + Wave2 BPTT lock) / prior L1 **2.873** | Living production target; predictive AC codes under the same `predict_next` |
| **D04 "33.2 bpc"** | benchmark bug | Wrong prob indexing on legacy D04 path -- ignore as evidence |

See [`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md) and [`CHANGELOG.md`](../../CHANGELOG.md).

---

## Doc index

| File | Purpose |
|------|---------|
| [VERIFICATION_STATUS.md](VERIFICATION_STATUS.md) | Test counts, coverage snapshot, known gaps |
| [PORT_CONTRACT.md](../port/PORT_CONTRACT.md) | Frozen binary/REST contracts |
| [PORT_FULL_STACK.md](../port/PORT_FULL_STACK.md) | Per-milestone record (M1–M6) |
| [FUTURE.md](../FUTURE.md) | Future directions in depth |
| [VERIFY_PLAN.md](VERIFY_PLAN.md) | WSL, benchmark workflow |
| [MAINTENANCE.md](MAINTENANCE.md) | When to update fixtures, rebuild native, sync DDL |
