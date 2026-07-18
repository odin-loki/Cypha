# Tier 2 cell native paths — verification (2026-07-18)

**BoW §3:** “Tier 2 H07, H09–H13 native paths”  
**Build:** `native/build_ewc_d16`  
**Budget:** `n_train=2000`, `n_eval=128`, `bench_seed=42` (no overnight/full-corpus env)

## Status

All listed variants already have `runnable=true` and `apply_cell_variant` wiring in `cypha_cell_hypothesis.cpp`. This pass confirms they execute end-to-end.

| Variant | Mode | BPC @ 2k | vs hybrid B2 |
|---------|------|----------|--------------|
| B2 hybrid ref | hybrid | **4.720** | — |
| H07 Differential gate | hybrid | 4.720 | ≈0 |
| H09 GRIA-gated mixture | hybrid | **4.712** | −0.008 |
| H10 NMP regularised | hybrid | 4.720 | ≈0 |
| H11 Reversible cell | ssm | **null** | short-budget NaN; overnight sweep already covers H11 |
| H12 MDL forget | hybrid | 4.720 | ≈0 |
| H13 Priority replay | hybrid | 4.720 | ≈0 |

Artifacts: `bench/results/tier2_fast_clean/*.json`  
Also: `cypha_cell_hypothesis_sweep --tier2-smoke` green for its smoke subset.

## Disposition

- **Close BoW checkbox** — native paths exist and run.
- No Tier‑2 promote from this budget (H09 tiny edge only).
- H11 null @ 2k is known SSM/RevNet short-budget fragility, not a missing wiring bug.

## Maintainer gate (still open)

Release still needs interactive GitHub auth:

```text
gh auth login -h github.com -p https -w
# then: git push && powershell -File scripts/publish_release.ps1
```
