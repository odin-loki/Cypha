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
- [ ] **Numerical audit — R1 fixed, R3/R4 pinned, 3 decisions open** — full report: [`docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md`](docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md). Every finding is now hand-verified; L10 did not survive that pass and is marked unresolved. **Done:** R1 (`world_gate` applied twice on the kernel-LLR path, a 45.8% confidence understatement) fixed and guarded by CTest `native_kernel_gate_invariant`; R3/R4 pinned by CTest `native_ported_defects_pinned`, which asserts their defective behaviour on purpose so a future change is a deliberate contract decision. **Open, and each needs a human call, not a patch:** **R2** `infer_cpu.cpp:1271-1280` divides a latent-variance-scaled `r_eff` by a dimensionless `mahal_ema` on the default `/predict` path — its reference (the FastAPI `InferenceEngine`) is not in the archive and `PORT_CONTRACT.md:75` pins the pairing, so there is nothing to check a change against; **R3** `infer_cpu.cpp:865-885` drops confidence exactly `1.0` from ECE binning and scores an all-NaN evaluation as a perfect 0.0 (Python `Cypha.py:137`); **R4** `infer_cpu.cpp:903-950` never scores the incumbent temperature (Python `Cypha.py:3129`) — a parity-preserving improvement would be to return the chosen ECE alongside `T` so callers can reject a regression. Diverging on R3/R4 means updating `PORT_CONTRACT.md`. Thirteen latent findings (L5–L13, N1–N4) are recorded with measurements; the sharpest are **L9** (183,085% interpolation error in the Bessel table's first cell) and **L5** (`K₀/K₁` negative on 44.1% of its domain).

## Future (not blocking)

- **CPU SIMD via [xsimd](https://github.com/xtensor-stack/xsimd)** — portable AVX2/AVX-512/NEON kernels for `score_matrix` / matvec / softmax. Training stays on CPU; this is the speed path that is actually worth it. Spec: [`docs/FUTURE.md`](docs/FUTURE.md) §1b.

## Closed (do not reopen)

Optimality P0–P9, RPSM Option B, 28-variant July cell sweep, v2.3.24 / v2.3.25 publish, D10A 90.11%, XOR latent RFF default, forecasting Phases 1–9, **GPU training** (CUDA infer is optional; device training is slower than CPU on this workload — not a gap). Details: archive BoW + [`docs/archive/README.md`](docs/archive/README.md).
