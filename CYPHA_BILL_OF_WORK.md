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
- [ ] **Numerical audit — 4 reachable defects in the inference gate, 13 latent** — full report: [`docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md`](docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md). Recorded, **not fixed**, by decision. The four reachable ones: **R1** `infer_cpu.cpp:1165-1169` applies `world_gate` twice on the kernel-LLR path (`confidence = disc·gate²`), a ~40% understatement at gate 0.6, settable from the REST request body and able to flip a label through the abstain band; **R2** `infer_cpu.cpp:1271-1280` builds the anomaly score by dividing a latent-variance-scaled `r_eff` by a dimensionless `mahal_ema` — `use_gh` defaults to **true**, so this is the default `/predict` path, and on the shipped fixture all four GH cases score 0.0 instead of 0.18–1.44; **R3** `infer_cpu.cpp:865-885` drops any sample at confidence exactly `1.0` from ECE binning and scores an all-NaN evaluation as a perfect 0.0 — a perfectly confident, perfectly wrong classifier reports ECE 0.000000; **R4** `infer_cpu.cpp:903-950` never scores the incumbent temperature, so `/adapt_temperature` can install a worse one and report success (measured 1.44× regression), and `n_grid=1` stores `T_min` with no search. R2 and R4 are 1:1 ports whose parity is stated in `docs/port/PORT_CONTRACT.md`, so fixing them is a contract decision, not a silent patch. No test covers any of the four.

## Future (not blocking)

- **CPU SIMD via [xsimd](https://github.com/xtensor-stack/xsimd)** — portable AVX2/AVX-512/NEON kernels for `score_matrix` / matvec / softmax. Training stays on CPU; this is the speed path that is actually worth it. Spec: [`docs/FUTURE.md`](docs/FUTURE.md) §1b.

## Closed (do not reopen)

Optimality P0–P9, RPSM Option B, 28-variant July cell sweep, v2.3.24 / v2.3.25 publish, D10A 90.11%, XOR latent RFF default, forecasting Phases 1–9, **GPU training** (CUDA infer is optional; device training is slower than CPU on this workload — not a gap). Details: archive BoW + [`docs/archive/README.md`](docs/archive/README.md).
