# Cell-sweep summary — post-overnight (2026-07-18)

**Tool:** `scripts/aggregate_cell_sweep_summary.ps1`  
**Baselines:** B0=3.478 · B1=2.979 · B2=**2.873** (canonical hybrid)  
**Local CSV:** `bench/results/cell_sweep/summary.csv` (gitignored; 25 rows)

## Artifact status

| Variant | Issue |
|---------|-------|
| **H15** | Overnight `bpc: null` (axiom NaN). Remeasured **3.982** @ `n_train=5000` after NaN fix — **not** a 300k overnight re-row. |
| Count | **25/25** rows with BPC in local CSV after 2026-07-18 reaggregate |

## Highlights (lower BPC better)

| Variant | BPC | vs B2 (2.873) | Note |
|---------|-----|---------------|------|
| **H19** | **2.921** | +0.048 | Best hypothesis-tier row in CSV |
| B1 | 2.887 | +0.014 | Char-LSTM artifact row (not locked B1) |
| H22 | 3.077 | +0.204 | |
| H01 | 3.073 | +0.200 | |
| **H15** | **3.982** | +1.109 | Finite @5k post-fix; optional 300k re-run |
| H16/H17 | ~4.61 | +1.74 | |
| H18/H20/H21 | ~11.5 | +8.6 | Collapsed / failed configs |
| B0 / B2 rows in CSV | 11.58 / 6.81 | — | Sweep-local baselines, **not** lock pins |

**No hypothesis beat canonical hybrid 2.873.** Closest: H19 at 2.921 (+48 mBPC).

## How to regenerate

```powershell
powershell -File scripts/aggregate_cell_sweep_summary.ps1
```
