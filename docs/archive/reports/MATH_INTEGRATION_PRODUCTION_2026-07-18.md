# Math-integration production status — 2026-07-18

**Lock:** `a552aee` / `bench/BASELINE_LOCK.json`  
**Plan:** [`BACKLOG_EXECUTION_PLAN_2026-07-18.md`](BACKLOG_EXECUTION_PLAN_2026-07-18.md) Phase A1

## Locked 300k numbers

| Arm | BPC | κ |
|-----|-----|---|
| Baseline (overnight) | **2.864** | 0.837 |
| Math-integration | **3.073** | 0.860 |
| Δ | **+0.209 BPC** (worse) | **+0.023 κ** |

`math_integration_results.status = production`, `n_train = 300000`. Joint BPC gate (ΔBPC < 0) **fails** at production scale even though κ moves toward the target band.

## Open research (unchanged)

| Item | Status |
|------|--------|
| Flat ablations @ FAST/5k (identical ΔBPC) | Still open — insufficient resolution; do not trust presets from 5k alone |
| Scale sign-flip (worse@500, better@5k, worse@300k) | Documented by lock; needs medium-tier sweep {500,5k,50k,300k} before retune |
| κ held-out transfer | No dedicated transfer tests beyond eval split |
| Eigenvalue `D_eff` alone | Prior A/B **+0.096 ΔBPC** (harmful); r_eu forget gate stays in preset |

## d53–d58 results (2026-07-18, `build_d38`)

| Domain | `validation_status` | Notes |
|--------|---------------------|-------|
| d53 | `preset_ship_production_wiring_ready` | `lock_joint_ok=false` (ΔBPC +0.209 @ 300k); subprocess @ smoke still joint-OK |
| d58 | `production_overnight_math_wiring_ready` | `lock_joint_ok=false`; `production_lock_ready=false` (joint BPC gate) |

Not a full joint math certificate — expected while BPC regresses at production scale.

## Product implication

Math-integration remains **opt-in research**. Do not promote as default overnight path until a retuned preset wins joint BPC at ≥50k (preferably 300k).
