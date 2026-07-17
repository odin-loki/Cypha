# Sample efficiency curve (MC5/MG5) — 2026-07-17

**Scope:** Bill of Work Addendum 2, build-order item 3.

## What shipped

| ID | Deliverable | Path |
|----|-------------|------|
| MC5 | BPC vs `n_train` curve runner | `cyphalm_sample_efficiency_curve` |
| MG5 | JSON curve table format + validation | `bench/results/sample_efficiency_curve.json` |
| — | Profile config | `bench/config/sample_efficiency_curve_profile.json` |
| — | CTest smoke | `native_sample_efficiency_curve_smoke` |

The curve tool invokes **`cyphalm_bench_native`** once per tier (no new training loop). Each point records `n_train`, `bpc`, and a null `accuracy` placeholder for future classification-tier sweeps via `cypha_bench_run`.

## JSON curve table format

```json
{
  "curve_id": "sample_efficiency",
  "metric": "bpc",
  "runner": "cyphalm_sample_efficiency_curve",
  "profile": "d17",
  "mode": "hybrid",
  "bench_seed": 42,
  "fast": true,
  "tiers_requested": [80, 160, 240],
  "points": [
    {"n_train": 80, "n_eval": 64, "bpc": 7.03, "accuracy": null},
    {"n_train": 160, "n_eval": 64, "bpc": 6.98, "accuracy": null},
    {"n_train": 240, "n_eval": 64, "bpc": 6.94, "accuracy": null}
  ]
}
```

Default tiers: **500 / 2000 / 5000** (`CYPHA_BENCH_FAST=1` → **80 / 160 / 240**).

## Example curve (D17 hybrid, `CYPHA_BENCH_FAST=1`, synthetic corpus)

| n_train | BPC |
|---------|-----|
| 80 | *(run below)* |
| 160 | *(run below)* |
| 240 | *(run below)* |

## Build & test

```powershell
cmake -S native -B native/build_curves -DCMAKE_BUILD_TYPE=Release -DCYPHA_BUILD_EXPERIMENT_DB=OFF -G Ninja
cmake --build native/build_curves --target cyphalm_sample_efficiency_curve sample_efficiency_curve_smoke
ctest --test-dir native/build_curves -R native_sample_efficiency_curve_smoke
$env:CYPHA_BENCH_FAST=1; native/build_curves/cyphalm_sample_efficiency_curve.exe --write-table
```

Smoke asserts **≥2** curve points and finite `bpc` on every point.

## Notes

- **`accuracy`** is reserved for tabular domains (D01/D03) when a per-tier `cypha_bench_run` hook lands; LM curves use **`bpc`** only today.
- Use **`--write-table`** to mirror output into `bench/report/tables/sample_efficiency_curve.json`.
