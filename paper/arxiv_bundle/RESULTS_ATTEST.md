# Results snapshot (bundle provenance) — 2026-07-18

Companion to the paper draft. Canonical lock pin unchanged.

| Claim | Value | Source |
|-------|-------|--------|
| D17 hybrid BPC (canonical) | **2.873** | `bench/BASELINE_LOCK.json` |
| D10A ECG5000 accuracy | **90.11%** (legacy 85.96% via `CYPHA_D10_ECG_ENRICH=0`) | `docs/reports/D10_ECG5000_GT90_ATTEMPT_2026-07-18.md` |
| Cell H15 @ 300k (post axiom fix) | **5.262** BPC, κ≈0.872 — **not a promote** | `docs/reports/H15_300K_RERUN_2026-07-18.md` |
| Best cell-sweep hypothesis vs hybrid | H19 **2.921** (+0.048 vs 2.873) | `docs/reports/CELL_SWEEP_SUMMARY_2026-07-18.md` |

Honest product caveats (also in paper): shared-model continual learning open; zero forgetting = per-task isolation (D16F); GRIA blend LSTM-dominated (~99.6%).
