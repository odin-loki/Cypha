# Results snapshot (bundle provenance) — 2026-08-16

Companion to the paper draft. Living production pin is `bench/BASELINE_LOCK.json`.

| Claim | Value | Source |
|-------|-------|--------|
| D17 hybrid BPC (production) | **2.664** (L2 + Wave2 BPTT @ 300k) | `bench/BASELINE_LOCK.json` |
| Prior D17 pins (historical) | L1 **2.873** / SGD L2 **2.816** | lock `prior_l1_pin_bpc` / `prior_sgd_l2_pin_bpc` |
| D10A ECG5000 accuracy | **90.11%** (legacy 85.96% via `CYPHA_D10_ECG_ENRICH=0`) | `docs/archive/reports/D10_ECG5000_GT90_ATTEMPT_2026-07-18.md` |
| Cell H15 @ 300k (post axiom fix) | **5.262** BPC, κ≈0.872 — **not a promote** | `docs/archive/reports/H15_300K_RERUN_2026-07-18.md` |
| Best historical cell-sweep hypothesis | H19 **2.921** (status=`historical`; living spine is Hybrid 2.664) | `docs/archive/reports/CELL_SWEEP_SUMMARY_2026-07-18.md` |

Honest product caveats (also in paper): shared-model continual learning open; zero forgetting = per-task isolation (D16F); GRIA blend LSTM-dominated (~99.6%).
