# Cell sweep summary aggregation tool — 2026-07-17

**Script:** `scripts/aggregate_cell_sweep_summary.ps1`  
**Output:** `bench/results/cell_sweep/summary.csv`  
**Scope:** Read-only scan of cell-sweep artifacts; does **not** modify `bench/BASELINE_LOCK.json` or overnight processes.

---

## Purpose

Populate `summary.csv` from per-variant JSON artifacts (`variant_*.json`) and compare each run's BPC to the locked cell-sweep baselines:

| Baseline | Model | BPC |
|----------|-------|-----|
| B0 | 4-gram (bigram tier) | **3.478** |
| B1 | char-LSTM | **2.979** |
| B2 | hybrid_gria_lstm | **2.873** |

Negative `vs_B*` values mean the variant beats that baseline (lower BPC is better).

---

## CSV columns

```csv
variant,bpc,vs_B0,vs_B1,vs_B2
```

- **variant** — variant ID (e.g. `B0`, `H14`)
- **bpc** — measured bits-per-character from the variant JSON
- **vs_B0** — `bpc − 3.478`
- **vs_B1** — `bpc − 2.979`
- **vs_B2** — `bpc − 2.873`

---

## How to run

From repo root:

```powershell
# Default: scan bench/results/cell_sweep (+ repo-root results/ spill if present), write summary.csv
pwsh -File scripts/aggregate_cell_sweep_summary.ps1

# Dry-run: same scan and CSV write; prints mode banner (safe during in-flight overnight)
pwsh -File scripts/aggregate_cell_sweep_summary.ps1 -DryRun

# Custom output directory
pwsh -File scripts/aggregate_cell_sweep_summary.ps1 -OutputDir bench/results/cell_sweep
```

The script prints `rows=<N> hash=<sha256>` on completion.

---

## Input layout

Scanned paths (newest file wins when the same variant ID appears in both):

1. `bench/results/cell_sweep/variant_*.json` (primary)
2. `results/variant_*.json` (optional in-flight overnight spill)

Each JSON must contain `id` and numeric `bpc`. Unparseable or BPC-missing files are skipped with a warning.

---

## Partial / empty sweeps

- **No JSON found:** writes a header-only CSV (`variant,bpc,vs_B0,vs_B1,vs_B2`).
- **Partial sweep (e.g. H15–H18 incomplete):** includes only variants with valid JSON; missing IDs are omitted (no placeholder rows).
- Exit code **0** in all cases unless a fatal parameter/path error occurs.

During an active overnight run, variant JSON may lag progress lines in `overnight_progress.log` (artifacts flush at sweep completion). Re-run after finalize for the full matrix.

---

## Related

- Native sweep writer: `native/tools/cypha_cell_hypothesis_sweep.cpp` (`write_overnight_artifacts`)
- Baseline reference: `docs/research/upgrades/CELL_HYPOTHESIS_TESTBENCH.md`
- Overnight health: `docs/reports/OVERNIGHT_HEALTH_2026-07-17.md`
- Spill merge: `scripts/migrate_inflight_overnight_artifacts.ps1`
