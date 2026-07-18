# P4 / P5 continual-learning decision — 2026-07-18

**Plan:** Phase B of [`BACKLOG_EXECUTION_PLAN_2026-07-18.md`](BACKLOG_EXECUTION_PLAN_2026-07-18.md)  
**FAST suite:** `CYPHA_BENCH_FAST=1` `cypha_bench_run --domain-tag d16` + `ewc_d16b_sweep` + D03 pilots

## P4 — Multi-view CyphaDIF

| Experiment | Result | Verdict |
|------------|--------|---------|
| Index-reorder (D03 Phase 2.1 / 16G) | Prior negatives | **CLOSED** — do not enable in production |
| **16I replay-interleave** | `forgetting_delta_0_22=+0.075`, `_0_50=+0.228` vs RR (worse) | **FAIL** gate (≥5pp reduction) |
| **DIF-V1 class-block** (`CYPHA_D03_CLASS_BLOCK=1`) | iris 0.85→0.70; wine 1.0→0.75 | **FAIL** |
| **DIF-V2 curriculum** (`CYPHA_CURRICULUM_WINDOW=8`) | iris 0.85→**0.926**; wine held 1.0 | **PASS** (≥1pp on iris) |

**Product:** Keep curriculum as opt-in env for `train_eval_vectors` callers. Do **not** port LM-style reorder or elevated `replay_ratio` as D16 defaults. Isolation (16F) remains the zero-forgetting path.

## P5 — Shared-model CL / EWC

| Setting | forgetting_score | Notes |
|---------|------------------|-------|
| FAST baseline (sweep) | 0.0345 | Lower than legacy everyday 0.813 (budget differ) |
| EWC λ=2.0, no world-field | **0.0** | Best this sweep; B/C acc held |
| EWC λ=2.0 + world-field protect | 1.0 | Catastrophic (unchanged finding) |

**Decision:** Ship **D16F isolation-only** as the supported product claim. EWC λ≈2.0 is an **optional research overlay** when shared-model training is required; world-field Fisher protection stays off. Routing redesign spike deferred unless a product requirement demands shared-model near-zero forgetting under heavy continual load.

## BoW updates

- §1 P4: index-reorder closed; DIF-V3 16I measured negative; curriculum opt-in kept.
- §1 P5: accept isolation-only for product; EWC modest/optional.
