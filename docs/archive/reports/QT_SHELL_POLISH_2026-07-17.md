# Qt shell polish — compare empty-state hints (2026-07-17)

**Scope:** Bill of Work §5 Qt shell (optional). One bounded UX slice on the Experiments compare pane. Did not touch `build_math`, `build_deff`, `BASELINE_*`, or overnight.

## Shipped

| Change | Where |
|--------|-------|
| Selection-aware compare hint (`0` / `1` / `N` runs) | `experiment_update_compare_hint()` in `native/qt/src/shell_main.cpp` |
| Clearer empty chart copy | `SimpleLossChart` compare-mode placeholder |
| Tooltip on summary label | Acc/F1/R²/RMSE + `metrics_history` loss requirement |
| Clear resets compare series + restores hint | Compare Clear button path |

Builds on compare stats from `7578f13`. Source-only this slice (no Qt rebuild required to land).

## Manual check

Open `cypha_qt_shell` → Experiments → select 0/1/2+ runs → confirm hint text updates → Compare → Clear restores empty hint.
