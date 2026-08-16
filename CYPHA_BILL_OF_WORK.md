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

## Open

- [ ] **Linux CI green** — current run is [140](https://github.com/odin-loki/Cypha/actions/runs/31923605267) on `b3d7902` (docs/data move). Prior UTF-8-fix run was cancelled by that push. Last fully green `build_and_test` was v2.3.25.
- [ ] **Cut v2.4.0** — bump `native/CMakeLists.txt` + script defaults; tag after CI is green. Notes template is in `scripts/create_release_notes.ps1`.
- [ ] **Finish cell sweep gaps** — 12 variants at 300k (H14, H23, U01–U10). July 300k rows stay historical. Runner: `scripts/run_cell_sweep_parallel.ps1`.
- [ ] **Paper PDF body** — `paper/arxiv_bundle/CyphaLM_paper.md` still cites 2.873; lock + `MODEL_CARD.md` + `RESULTS_ATTEST.md` are submission truth until the PDF is regenerated.
- [ ] **arXiv / venue upload** — human; bundle at `paper/arxiv_bundle/`.

## Future (not blocking)

- **CPU SIMD via [xsimd](https://github.com/xtensor-stack/xsimd)** — portable AVX2/AVX-512/NEON kernels for `score_matrix` / matvec / softmax. Training stays on CPU; this is the speed path that is actually worth it. Spec: [`docs/FUTURE.md`](docs/FUTURE.md) §1b.

## Closed (do not reopen)

Optimality P0–P9, RPSM Option B, 28-variant July cell sweep, v2.3.24 / v2.3.25 publish, D10A 90.11%, XOR latent RFF default, forecasting Phases 1–9, **GPU training** (CUDA infer is optional; device training is slower than CPU on this workload — not a gap). Details: archive BoW + [`docs/archive/README.md`](docs/archive/README.md).
