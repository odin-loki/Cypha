# Real-data profiling pass - 2026-07-17

**Scope:** Bill of Work section 7 housekeeping - log bench/tune timing + metrics on a real CSV under `bench/data/`. Did not touch `build_math`, `build_deff`, `BASELINE_*`, or overnight.

## Sample data

| File | Rows | Loader |
|------|------|--------|
| `bench/data/iris.csv` | 132 | `load_tabular_dataset("iris")` -> `data_source=csv` when present |

Wine in D03 still uses synthetic fallback (no `wine.csv` in tree); iris confirms the CSV ingest path on real UCI measurements.

## Build

```powershell
cmake -S native -B native/build_realprof -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_realprof --target cypha_bench_run cypha_tune_run
```

## Bench run (cypha_bench_run --domain 3)

| Setting | Value |
|---------|-------|
| `CYPHA_BENCH_FAST` | `1` |
| `CYPHA_REPO_ROOT` | scratch temp tree (no writes to repo `bench/BASELINE_*`) |
| Wall time | **4.503s** |

### D03 metrics (from `bench/report/tables/d03.json`)

| Dataset | Source | Accuracy |
|---------|--------|----------|
| iris | csv | 0.8518518518518519 |
| wine | synthetic | 1.0 |

## Tune smoke (cypha_tune_run)

| Setting | Value |
|---------|-------|
| Config | `bench/config/real_data_profile_tune_smoke.json` |
| Cells | d03_default, d03_fast_repeat |
| Wall time | **8.99s** |

## Reproduce

```powershell
scripts/run_real_data_profile.ps1
# or: scripts/run_real_data_profile.ps1 -SkipBuild -BuildDir native/build_realprof
```

Generated: 2026-07-17 21:15:57 +10:00