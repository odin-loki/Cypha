# Unified-context BPC tournament — smoke winner (2026-07-18)

**Decision:** User accepted the **smoke ranking** instead of the full 40k screen / 300k crown.

**Protocol (smoke):** `cyphalm_bench_native --profile d17 --cell-variant X --n-train 200 --n-eval 50 --threads 1 --bench-seed 42` with `CYPHA_BENCH_FAST=1` (synthetic/fast corpus). Lower BPC is better.

**Binary:** `native/build-pgm/cyphalm_bench_native.exe` (unified-context U01–U10 wired).

## Smoke table

| Rank | Variant | Spine | BPC | Δ vs B2 |
|-----:|---------|-------|----:|--------:|
| 1 | **U06** | PGM→logits (`PgmWy`) | **6.0129** | **−0.1946** |
| 2 | U07 | Bank→LSTM | 6.1651 | −0.0424 |
| 3 | U10 | UnifiedBuffer→LSTM | 6.1686 | −0.0388 |
| 4 | U04 | PGM→LSTM | 6.1693 | −0.0382 |
| 5 | U09 | PGM-large→LSTM | 6.1693 | −0.0381 |
| 6 | U01 | Field→LSTM | 6.1716 | −0.0359 |
| 7 | U08 | Memory→LSTM | 6.1743 | −0.0332 |
| 8 | U03 | LSTM-only | 6.1763 | −0.0311 |
| 9 | **B2** | hybrid (control) | 6.2074 | 0 |
| 10 | U05 | PGM→GRIA | 7.5237 | +1.3163 |
| 11 | U02 | Field→GRIA | 7.5347 | +1.3272 |

## Winner

**U06 — PGM→logits** (single context = PGM `h`, single readout = `Wy` on PGM dim).

- Best smoke BPC among B2 + U01–U10.
- Best unified spine: same (U06).
- GRIA-only spines (U02/U05) are clearly worse at this budget.

## Caveats

- Smoke is **not** the D17 300k pin protocol. Absolute BPC (~6.0) is not comparable to the locked hybrid **2.873 @ 300k**.
- Relative ordering may shift at 40k/300k; smoke only picks a research default among the unified spines.
- Production pin remains **B2** until a real 300k crown beats 2.873.

## How to promote later

```powershell
powershell -File scripts\run_unified_context_tournament.ps1 -BuildDir native/build-pgm -SkipBuild
```

Or pin U06 alone:

```text
cyphalm_bench_native --profile d17 --cell-variant U06 --n-train 300000 --n-eval 2000 --threads 1 --bench-seed 42
```
