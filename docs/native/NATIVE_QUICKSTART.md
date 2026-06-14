# Native quick start (v2.4 / v2.5)

One-page guide: **install → validate → bench → tune → REST**. Native C++ is the sole production runtime.

**Deeper docs:** [`native/README.md`](../../native/README.md) (all targets), [`PORT_CONTRACT.md`](../port/PORT_CONTRACT.md) (API + bench §6), [`CYPHALM_NATIVE_BUILD.md`](CYPHALM_NATIVE_BUILD.md) (CyphaLM build).

---

## 1. Build (developers)

Use a build directory **outside OneDrive** (cloud sync locks object files).

```powershell
# Windows — from repo root
cmake -S native -B C:\Temp\cypha_full_cpp_build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build C:\Temp\cypha_full_cpp_build --parallel
```

```bash
# Linux / WSL
cmake -S native -B /tmp/cypha_full_cpp_build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build /tmp/cypha_full_cpp_build --parallel
```

Optional: **`-DCYPHA_ENABLE_CUDA=ON`** (MSVC or Linux + NVIDIA), **`-DCYPHA_BUILD_QT=ON`** (Studio shell), **`-DCYPHA_BUILD_EXPERIMENT_DB=ON`** (SQLite experiments).

---

## 2. Install (release bundle)

Prebuilt installers: **[GitHub Releases `v2.2.8`](https://github.com/odin-loki/Cypha/releases/tag/v2.2.8)** (`cypha-*-linux-x86_64.tar.gz`, `cypha-*-windows-x86_64.zip`).

After download, or after local packaging with `scripts/package_release_windows.sh` / `scripts/package_release_linux.sh`:

```powershell
# Windows — from extracted tarball
powershell -ExecutionPolicy Bypass -File packaging\install_release_windows.ps1
```

```bash
# Linux
bash packaging/install_release_linux.sh
```

Adds **`cypha_rest`**, **`cypha_bench_run`**, **`cypha_bench_report`**, **`cypha_tune_run`**, **`cypha_diagnostics_run`** to PATH. Parity tools (if bundled) live in **`bin/dev/`**. See [`packaging/README.md`](../../packaging/README.md).

---

## 3. Validate

**Full gate (recommended):**

```powershell
powershell -File scripts\cypha_native_validate_all.ps1
```

Runs: Release build → **`ctest -R native_`** (107 CTests) → REST contract smoke → bench smoke (d01, d04, d17) → **`--report-only`** figures → tune dry-run (all smoke configs). Pass **`-TuneSmoke`** for a live **`cypha_tune_run`** sweep.

**Manual subset:**

```bash
ctest --test-dir C:/Temp/cypha_full_cpp_build -R native_ --output-on-failure
C:/Temp/cypha_full_cpp_build/cypha_diagnostics_run --fixtures fixtures --exe-dir C:/Temp/cypha_full_cpp_build
```

Set **`CYPHA_*_BIN`** env vars to point smoke harnesses at your build tree (see `scripts/cypha_native_validate_all.ps1`).

---

## 4. Bench

From repo root (so **`bench/report/`** paths resolve):

```bash
cypha_bench_run --list-domains
cypha_bench_run --domain 1          # d01 only
cypha_bench_run --from-domain 1     # d01 … d17
cypha_bench_run --report-only       # cross-domain + BASELINE_REPORT.md + figures
```

Fast smoke: **`CYPHA_BENCH_FAST=1`** (smaller subsamples).

Outputs:

| Path | Content |
|------|---------|
| `bench/report/tables/dXX.json` | Per-domain results |
| `bench/report/summary.json` | Roll-up |
| `bench/BASELINE_REPORT.md` | Human-readable report |
| `bench/report/figures/` | Figure JSON + PNG (`figures_manifest.json`) |

Report-only without re-running domains: **`cypha_bench_report --output ./bench_report`**.

---

## 5. Tune

Native sweep configs under **`bench/config/`**:

| Config | Runner |
|--------|--------|
| `cyphalm_hybrid_lstm_tune_smoke.json` | `cyphalm_bench_native` |
| `cyphalm_d17_phase1c_tune_smoke.json` | `cyphalm_bench_native` |
| `cypha_branch_a_encoder_tune_smoke.json` | `cypha_bench_run` |

```powershell
# Dry-run all smoke sweeps (default in validate_all)
powershell -File scripts\cypha_tune_smoke.ps1 -DryRun -MaxCells 4

# Live smoke (writes results under bench/artifacts/tuning/)
powershell -File scripts\cypha_tune_smoke.ps1 -Write -MaxCells 4

# Or single sweep
cypha_tune_run --config bench/config/cyphalm_hybrid_lstm_tune_smoke.json --dry-run
cypha_tune_run --config path/to/sweep.json --write --max-cells 4
```

Sweep JSON specifies **`runner`** (`cyphalm_bench_native` or `cypha_bench_run`), **`defaults`**, and explicit **`cells`** or cartesian **`grid`**. Boolean cell args (e.g. **`analysis`**) emit flag-only CLI tokens. Results JSON written under **`bench/artifacts/tuning/`** (gitignored).

---

## 5b. CyphaLM training (native)

Train a checkpoint from corpus text (profiles **`d17`** / **`d04`** load bench JSON from **`bench/config/profiles/`**):

```bash
# WikiText-style profile (requires bench/data/wikitext2/... or use synthetic smoke)
cyphalm_train --profile d17 --corpus bench/data/wikitext2/wikitext-2/wiki.train.tokens \
  --epochs 2 --out bench/artifacts/checkpoints/d17_run/

# Gutenberg char-LM (d04)
cyphalm_train --profile d04 --corpus bench/data/gutenberg/moby_dick.txt \
  --epochs 2 --out bench/artifacts/checkpoints/d04_run/

# Fast smoke (synthetic tokens — same path CTest uses)
cyphalm_train --profile d04 --epochs 1 --synthetic-tokens 512 \
  --max-train-steps 128 --out /tmp/cyphalm_train_smoke/
```

Writes **`checkpoint.json`** + **`checkpoint.npz`** under **`--out`**. Load in REST: **`cypha_rest --cyphalm-checkpoint path/to/checkpoint.json`**.

```bash
ctest --test-dir C:/Temp/cypha_full_cpp_build -R native_cyphalm_train_smoke --output-on-failure
```

---

## 6. REST

```bash
cypha_rest --listen 127.0.0.1:8099 \
  --cypha fixtures/reference.cypha \
  --f-field-json fixtures/f_field.json
```

**CyphaDIF:** `GET /health`, `GET /ready`, `POST /predict`, `POST /update`, `GET /models`, `POST /load`, `POST /register`, …

**CyphaLM:** `POST /lm/load`, `GET /lm/metrics`, `POST /lm/predict_next`, `POST /generate`, `POST /generate/stream`

**Branch A** (optional `--branch-a-json branch_a_router.json`): `GET /route/health`, `POST /route/text`, `POST /route/generate`, `POST /route/save`

```bash
curl -s http://127.0.0.1:8099/ready
curl -s http://127.0.0.1:8099/predict -H 'Content-Type: application/json' \
  -d '{"input":[0,0,0,0,0,0,0,0],"use_gh":true}'
```

Optional regression sidecar: **`--regression-json regression_head.json`**. Registry: **`--registry ~/.cypha/models`**.

**Qt Studio shell** (separate build with **`-DCYPHA_BUILD_QT=ON`**): see [`native/qt/README.md`](../../native/qt/README.md) and `scripts/run_cypha_qt_windows.ps1`.

---

## 7. Diagnostics

```bash
cypha_diagnostics_run --fixtures fixtures --exe-dir C:/Temp/cypha_full_cpp_build
cypha_diagnostics_run --phases 1,2 --list
```

Orchestrates parity executables + inline checks; writes JSON under **`cypha_diagnostics/results/`** (gitignored).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Slow / failed links on Windows | Build under `C:\Temp\`, not OneDrive |
| `ctest` cannot find fixtures | Run from configured build; MinGW cross-build rewrites `/mnt/c/` paths |
| Bench cannot find repo | Set **`CYPHA_REPO_ROOT`** to checkout root |
| CUDA on Windows | MSVC only — see [`ACCEL_CUDA.md`](ACCEL_CUDA.md) |
| Missing embed/retrieval CTests | Fixture sidecar not generated yet — CTest auto-disabled |

**Regenerate parity fixtures (offline, optional):** domain-specific generators under `scripts/` when `.cypha` or sidecar format changes.
