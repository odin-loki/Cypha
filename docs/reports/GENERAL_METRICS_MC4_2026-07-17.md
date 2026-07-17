# General metrics MC4 (margin distribution) — 2026-07-17

**Scope:** Bill of Work Addendum 2, MC4 — per-sample distance from the decision boundary.

| ID | Metric | Implementation | JSON field |
|----|--------|----------------|------------|
| MC4 | Mean / p50 / p10 of \|top1−top2\| logit margin | `cypha::bench::logit_margin_top2` + `margin_distribution` | `cypha_scores.margin_mean` / `margin_p50` / `margin_p10` |

Wired in `clf_metrics_native` (`bench_domains.cpp`) from batched LLR rows.

## Smoke fixture

Margins `{2, 2, 1, 10}` → mean **3.75**, p50 **2.0**, p10 **1.3** (linear percentile).

```
bench_metrics_smoke … margin_mean=3.7500 margin_p50=2.0000 margin_p10=1.3000 … PASS
```

Build: `native/build_mc4` (or any tree that rebuilds `bench_metrics_smoke` / `cypha_bench_native`).
