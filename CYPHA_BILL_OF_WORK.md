# Cypha — open work

Living checklist only. Completed July 2026 continuum / optimality / RPSM / cell-sweep closeout lives in the archive — do not treat those files as current pins.

| Kind | Where |
|------|-------|
| Research journal | [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) |
| Forward path | [`docs/FUTURE.md`](docs/FUTURE.md) |
| Docs hub | [`docs/README.md`](docs/README.md) |
| Competition card | [`MODEL_CARD.md`](MODEL_CARD.md) |
| Historical BoW (2026-07-18) | [`docs/archive/reports/CYPHA_BILL_OF_WORK_2026-07-18.md`](docs/archive/reports/CYPHA_BILL_OF_WORK_2026-07-18.md) |
| Historical optimality plan | [`docs/archive/plans/CYPHA_OPTIMALITY_PLAN_2026-07-18.md`](docs/archive/plans/CYPHA_OPTIMALITY_PLAN_2026-07-18.md) |
| Historical forecasting design | [`docs/archive/plans/CYPHA_FORECASTING_PLAN_2026-08-08.md`](docs/archive/plans/CYPHA_FORECASTING_PLAN_2026-08-08.md) |

**Authoritative production pin:** Hybrid L2 + Wave2 BPTT **2.664 BPC** @ 300k (`bench/BASELINE_LOCK.json`). Prior L1 2.873 / SGD L2 2.816 are historical.

**Latest release:** **[v2.4.0](https://github.com/odin-loki/Cypha/releases/tag/v2.4.0)** (2026-08-16). Notes: [`docs/reports/RELEASE_V2_4_0_2026-08-16.md`](docs/reports/RELEASE_V2_4_0_2026-08-16.md).

## Open

- [x] **Cell sweep wrap-up** — B2 and H06 rerun 2026-08-16 @ 300k / eval 2k (isolated `bench/results/cell_sweep_rerun/`). Both **3.681 BPC** with math-integration (same as H14; not a promote vs Hybrid 2.664). Archived at `data/archive/cell_sweep/variant_B2.json` + `variant_H06.json`. Best new row remains U03 **2.822**.
- [x] **Paper PDF body** — regenerated 2026-08-16; living pin **2.664** in `paper/CyphaLM_paper.md` + `paper/arxiv_bundle/` (HTML/PDF). L1 2.873 kept as historical.
- [ ] **arXiv / venue upload** — human; bundle at `paper/arxiv_bundle/` and on the v2.4.0 GitHub Release (zip refreshed 2026-08-16 with the 2.664 PDF). See `paper/arxiv_bundle/SUBMIT_CHECKLIST.md`.

## Future (not blocking)

- **CPU SIMD via [xsimd](https://github.com/xtensor-stack/xsimd)** — portable AVX2/AVX-512/NEON kernels for `score_matrix` / matvec / softmax. Training stays on CPU; this is the speed path that is actually worth it. Spec: [`docs/FUTURE.md`](docs/FUTURE.md) §1b.

## Closed (do not reopen)

Optimality P0–P9, RPSM Option B, 28-variant July cell sweep, v2.3.24 / v2.3.25 publish, D10A 90.11%, XOR latent RFF default, forecasting Phases 1–9, **GPU training** (CUDA infer is optional; device training is slower than CPU on this workload — not a gap). Details: archive BoW + [`docs/archive/README.md`](docs/archive/README.md).
