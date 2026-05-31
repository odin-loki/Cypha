# `cypha_diagnostics/`

Empirical diagnostic package for CyphaDIF. Runs a structured multi-phase
investigation to identify bottlenecks and root-cause bugs, then optionally
applies the evidence-backed fixes.

The permanent report from the 2026-05-30 run lives at
[`docs/reports/DIAGNOSTIC_REPORT.md`](../docs/reports/DIAGNOSTIC_REPORT.md).

---

## Scripts

| Script | Purpose |
|--------|---------|
| `run_diagnostics.py` | Full diagnostic pipeline (Phases 1–5): baseline, encoder quality, NIG calibration, online dynamics, best configs. Writes JSON phase results to `results/` (gitignored). |
| `apply_upgrades.py` | Applies the three diagnostic-confirmed fixes: deliberation disabled, `delta_lr=0.03`, auto-RFF for `input_dim ≤ 30`. Writes the patched profile to `cypha_bench/config/everyday_profile.json`. |

---

## Running the diagnostics

```bash
# From repo root (venv active with requirements-verify.txt + cypha_bench deps)
python cypha_diagnostics/run_diagnostics.py
```

Phase output JSON is written to `cypha_diagnostics/results/` (gitignored — raw
experiment data). The permanent readable summary must be written manually to
`docs/reports/DIAGNOSTIC_REPORT.md` after reviewing the output.

```bash
# Apply confirmed fixes to the bench profile
python cypha_diagnostics/apply_upgrades.py
```

---

## What the diagnostic checks

| Phase | Question answered |
|-------|------------------|
| Phase 1 — Baseline | Are the benchmarks saturated? How much headroom exists? |
| Phase 2 — Encoder quality | Is the RFF encoder producing well-separated latents? (FDR, silhouette, nonlinearity gap) |
| Phase 3 — NIG calibration | Are the class-conditional Gaussians well-calibrated? (ECE, delta_lr sweep) |
| Phase 4 — Online dynamics | Catastrophic forgetting? Label noise robustness? Convergence speed? |
| Phase 5 — Best configs | Grid of best hyperparameter combinations; top-5 report. |

---

## Confirmed findings (2026-05-30)

1. **Deliberation band `[0.4, 0.6]`** was suppressing ~40% of binary predictions
   as `__unknown__`. Fix: disabled. Effect: **+23.5 pp** on S1_2class_linear;
   regression R² −0.007 → 0.756.

2. **`delta_lr=0.06`** too aggressive. Fix: `delta_lr=0.03`. Effect: **+4 pp**
   on R3_digits.

3. **`VectorEncoder` inadequate for `input_dim ≤ 30`**. Fix: auto-select
   `RFFEncoder(D=256)`. Effect: **+14 pp** on S1_2class over `VectorEncoder` alone.

**Remaining hard limit:** XOR / nonlinear boundaries — FDR=0.001, gap=32.3 pp.
Requires Kernel LLR (Nyström). See [`docs/FUTURE.md §0a`](../docs/FUTURE.md).
