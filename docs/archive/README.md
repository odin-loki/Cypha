# Documentation archive

Historical reports, studies, and mega-plans. **Do not treat these as the product spine.**

## Living vs archive

| Kind | Where | Rule |
|------|-------|------|
| **Living** | `docs/` hub, `docs/verify/`, `docs/port/`, `docs/native/`, `docs/RESEARCH_STATUS.md`, `docs/FUTURE.md`, `docs/reports/ONE_CYPHA_CUTOVER.md` | Current product truth for **One Cypha** (`cypha::Cypha`: classify + regress + tokens) |
| **Archive** | `docs/archive/**` | Dated experiments, cutover notes, CyphaLM studies, closed roadmaps -- keep for history; update living docs instead of editing here |

Living sequence default: **PGM->Wy (U06)**. Hybrid GRIA+LSTM D17 **2.873 BPC** is a **historical pin** only (`bench/BASELINE_LOCK.json`).

## Layout

```
docs/archive/
  README.md                 # this policy
  failed_experiments/       # e.g. cypha_som post-mortem
  reports/                  # dated investigation / closeout reports
  reports/one_cypha/        # PGM / unified-context / large-context / v2.3.25 release notes
  studies/                  # CyphaLM algorithm studies, multi-view plans, doc-refresh notes
  plans/                    # mega plans and research roadmaps
```

## What stayed living under `docs/reports/`

- [`../reports/ONE_CYPHA_CUTOVER.md`](../reports/ONE_CYPHA_CUTOVER.md) -- cutover inventory and phase status for `cypha::Cypha`

## How to add new archive material

1. Prefer `git mv` from living paths so history is preserved.
2. Put dated one-off writeups in `reports/` (or `reports/one_cypha/` if sequence/cutover-specific).
3. Put long-lived study docs in `studies/`; multi-month roadmaps in `plans/`.
4. Leave a one-line pointer in the living doc that used to own the topic.
5. Do **not** delete archive files when superseding -- demote the claim in living docs instead.

## Entry points

| Need | Start here |
|------|------------|
| Product hub | [`../README.md`](../README.md) |
| Research journal | [`../RESEARCH_STATUS.md`](../RESEARCH_STATUS.md) |
| Cutover | [`../reports/ONE_CYPHA_CUTOVER.md`](../reports/ONE_CYPHA_CUTOVER.md) |
| Failed SOM experiment | [`failed_experiments/cypha_som/README.md`](failed_experiments/cypha_som/README.md) |
