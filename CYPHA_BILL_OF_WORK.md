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

**Latest release:** **[v2.5.0](https://github.com/odin-loki/Cypha/releases/tag/v2.5.0)** (2026-09-20). Prior notes: [`docs/reports/RELEASE_V2_4_0_2026-08-16.md`](docs/reports/RELEASE_V2_4_0_2026-08-16.md).

## Open

- [x] **Cell sweep wrap-up** — B2 and H06 rerun 2026-08-16 @ 300k / eval 2k (isolated `bench/results/cell_sweep_rerun/`). Both **3.681 BPC** with math-integration (same as H14; not a promote vs Hybrid 2.664). Archived at `data/archive/cell_sweep/variant_B2.json` + `variant_H06.json`. Best new row remains U03 **2.822**.
- [x] **Paper PDF body** — regenerated 2026-08-16; living pin **2.664** in `paper/CyphaLM_paper.md` + `paper/arxiv_bundle/` (HTML/PDF). L1 2.873 kept as historical.
- [ ] **arXiv / venue upload** — human; bundle at `paper/arxiv_bundle/` and on the v2.5.0 GitHub Release (zip refreshed with the living PDF). See `paper/arxiv_bundle/SUBMIT_CHECKLIST.md`.
- [x] **Numerical audit — R1–R4 and N1–N4, L5–L9 fixed; L11–L13 recorded** — full report: [`docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md`](docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md). Every finding is hand-verified; L10 did not survive that pass. **Inference gate:** **R1** `world_gate` applied twice on the kernel-LLR path (45.8% confidence understatement) → `native_kernel_gate_invariant`; **R2** the OOD anomaly score divided a latent-variance-scaled `r_eff` by a dimensionless `mahal_ema`, so on any model whose latent variance sat below `mahal_ema` the flag never fired — now `max(0, r_eff/r_base − 1)`, dimensionless and scale-invariant; **R3** ECE binning dropped confidence exactly `1.0` and scored an all-NaN evaluation as a perfect `0.0`, which won the temperature search outright since it minimises ECE; **R4** `adapt_temperature_ece` never scored the incumbent, so it could install a worse temperature and report success. R2/R3/R4 → `native_ported_defects_pinned`, verified red-then-green against the shipped function. The Python-parity objection to fixing R3/R4 was void: Python was decommissioned at P7 (`CHANGELOG.md:55`). **Bessel/GIG kernels:** N1–N4 and L5–L9 turned out to be one defect seen nine ways — `K₂/K₁` was the primitive and `K₀/K₁` derived from it, when `K₂/K₁` carries a `2/x` pole and `K₀/K₁` is bounded in `[0,1)`. Inverting that via the exact recurrence `K₂/K₁ = 2/x + K₀/K₁` took the worst error from **183,084% to 1.9e-6**, put `K₀/K₁` back inside `[0,1)` (it was **negative on 44.1%** of the score-match domain), and let all four wrong GIG-moment limit branches be deleted rather than corrected. → `native_bessel_gig_limits`. One golden fixture was regenerated: it pinned Python-parity values carrying the same error, and the new values are 8–15× closer to `scipy`. The CUDA device path carried the same four defects and now mirrors the host (verified bit-identical by transcribing the device functions to host C++; **not compile-tested — no CUDA toolchain in this environment**), with a text-level drift guard that needs no GPU. Fixing the kernels also exposed that the **Phase 7 acceptance gate was inverted**: `native_gate_score_match_p7_smoke` asserted only `loglik_score_match >= loglik_lut` with no reference value, so it rewarded whichever backend overstated the likelihood — it had been passing by a margin of +0.159 while score-match sat **8.5× further from the `scipy` truth** than the LUT. It now pins the exact held-out log-likelihood and asserts both backends track it. Constants and the previously-unreproducible Bessel table both have provenance now: [`scripts/gen_gig_k0k1_fit.py`](scripts/gen_gig_k0k1_fit.py). **Recorded, not fixed:** **L11**'s NaN asymmetry (CUDA `fmax(NaN, 0) = 0` vs the CPU's `std::max(NaN, 0) = NaN`, which needs a design call on what a NaN Mahalanobis should mean; its `pool_ensure` stale-capacity half **is** fixed), **L12** (`field_diag_a` allocates before its `fd <= 0` guard), **L13** (`infer_at_h` hardcodes `gh_chi = gh_psi = 1.0`, silently discarding caller options on the non-GH path).

## Future (not blocking)

- **CPU SIMD via [xsimd](https://github.com/xtensor-stack/xsimd)** — portable AVX2/AVX-512/NEON kernels for `score_matrix` / matvec / softmax. Training stays on CPU; this is the speed path that is actually worth it. Spec: [`docs/FUTURE.md`](docs/FUTURE.md) §1b.

## Closed (do not reopen)

Optimality P0–P9, RPSM Option B, 28-variant July cell sweep, v2.3.24 / v2.3.25 publish, D10A 90.11%, XOR latent RFF default, forecasting Phases 1–9, **GPU training** (CUDA infer is optional; device training is slower than CPU on this workload — not a gap). Details: archive BoW + [`docs/archive/README.md`](docs/archive/README.md).
