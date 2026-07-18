# Verification status — how “debugged” is this?

Honest snapshot for **native production**. “Debugged” here means *automated checks + known contracts*, not formal proof.

**Relationship to [`VERIFY_PLAN.md`](VERIFY_PLAN.md):** this file is the **snapshot** (what runs, counts, gaps). The verify plan is the **checklist** (commands, WSL, profiling workflow). **When to regen / rebuild:** [`MAINTENANCE.md`](MAINTENANCE.md).

## Automated coverage

| Layer | What runs | Count / notes |
|--------|-----------|----------------|
| **`ctest -R native_`** | `cypha_core` math, encoders, regressors, save/load, parity fixtures, REST/Qt smokes | **52+** CTests (see `ctest -N` on your build tree) |
| **`cypha_parity`** | Reload `.cypha`, numeric targets vs `expected.npz`, Tier-1 context | CTest **`native_parity`** |
| **`native/scripts/smoke_cypha_rest_mingw.ps1`** | Subprocess `cypha_rest` vs parity fixtures | REST smoke (set **`CYPHA_REST_BIN`**) |
| **`cypha_diagnostics_run`** | Phases 1–4 orchestration over parity exes | CTest **`native_diagnostics_run`** |
| **`cypha_bench_run`** | Multi-domain benchmark regression | Manual / validate script smoke |
| **`scripts/cypha_native_validate_all.ps1`** | Windows full gate: rebuild + CTest + bench + tune + REST | Local release bar |

**GUI smoke (optional Qt build):** `ctest --test-dir native/build -R native_qt --output-on-failure` with **`QT_QPA_PLATFORM=offscreen`**.

GitHub Actions **CI** (`.github/workflows/ci.yml`): **two blocking jobs** — **`build_and_test`** (Ubuntu: cmake + **`ctest -R native_`**) and **`windows_msvc`** (native MSVC Release on `windows-latest`).

**CyphaLM (research):** `ctest --test-dir native/build -R native_cyphalm --output-on-failure` locally when changing LM native code.

**Not covered automatically:** interactive GUI workflows, long-run memory leaks, multi-thread stress, real KDD-scale files unless you supply data.

## Contracts frozen for the port

- **[`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md)** — `.cypha` v3, LLR/softmax/GH/temperature, REST JSON.
- **`fixtures/`** — committed golden models + vectors; native code must reproduce within tolerance.

### Parity fixture inventory

See [`fixtures/README.md`](../../fixtures/README.md) and the table in prior revisions — **24+ directories** gated by **`ctest -R native_*`**.

## Known gaps (before you trust production scale)

1. **Full GPU training** — not implemented; optional CUDA accel for fused LLR when built with **`-DCYPHA_ENABLE_CUDA=ON`**.
2. **Real-data profiling** — use native bench/tune binaries on your CSV dumps.
3. **CUDA CI** — not in GitHub Actions; build locally with `-DCYPHA_ENABLE_CUDA=ON` and run **`native_cuda_smoke`** / **`native_score_batch`**; device benchmarks remain manual.

## Green bar (keep this clean)

- [x] GitHub Actions **CI green** on `main` — two blocking jobs
- [x] **`ctest -R native_`** green on CI and local validate scripts
- [x] Committed **`fixtures/`** match native parity tools
- [ ] Optional: GPU box profile after LM/accel changes (see [`docs/FUTURE.md`](../FUTURE.md))

## Full stack replacement — COMPLETE

All milestones M1–M7 are complete. See **[`PORT_FULL_STACK.md`](../port/PORT_FULL_STACK.md)** and **[`docs/FUTURE.md`](../FUTURE.md)** for next horizons.

**Regression contract:** native CI consumes **`fixtures/`** and JSON/binary artifacts only — no Python runtime. **[`PREPROCESSOR_CONTRACT.md`](../port/PREPROCESSOR_CONTRACT.md)** freezes `preprocessor.json` next to `model.cypha`.
