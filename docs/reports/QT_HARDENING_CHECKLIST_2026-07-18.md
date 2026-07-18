# Qt shell manual hardening checklist — 2026-07-18

**Build:** `-DCYPHA_BUILD_QT=ON -DCYPHA_BUILD_EXPERIMENT_DB=ON`  
**Plan:** Phase A4 of [`BACKLOG_EXECUTION_PLAN_2026-07-18.md`](BACKLOG_EXECUTION_PLAN_2026-07-18.md)

## Manual QA

| # | Step | Pass? |
|---|------|-------|
| 1 | Launch `cypha_qt_shell`, Help tab loads without crash | |
| 2 | Load a `.cypha` model; Predict on valid features | |
| 3 | Native train one-step / cancel mid-stream train | |
| 4 | Experiments: 0 runs selected → muted compare hint | |
| 5 | Select 1 run → Compare enabled; summary mentions single-run | |
| 6 | Select 2+ runs with `metrics_history` → Compare overlays curves + stats table | |
| 7 | **Export compare…** → CSV and JSON write readable files | |
| 8 | Clear compare → chart empty, export disabled, hint restored | |
| 9 | Dark theme readable on charts + tables | |
| 10 | Model card editor save/reload | |

## Shipped this pass

- Compare → **Export compare…** (CSV / JSON) from the stats table (`shell_main.cpp`).

## Still optional

- Extra compare chart series (rolling accuracy)
- Help tab path refresh
- REST endpoint discovery panel
