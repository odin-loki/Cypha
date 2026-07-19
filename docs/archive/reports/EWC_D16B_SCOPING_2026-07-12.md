# EWC vs D16B shared-model forgetting — scoping (2026-07-12)

**Status:** Shipped (growable-`D` fix + opt-in NIG world-field protection + sweep tooling).  
**Priority:** [`RESEARCH_STATUS.md`](../RESEARCH_STATUS.md) Priority 5.

## Problem

Bench D16B trains a **shared** CyphaDIF classifier on three sequential tasks (iris → wine → digits) and measures task-A forgetting after task B/C training. Documented baseline: **forgetting_score ≈ 0.813** (81% retention loss).

`EwcRegularizer` was wired into D16 probes, but two blockers prevented meaningful EWC on this path:

1. **Growable `D` silent no-op (bug, fixed):** `CyphaDifMemoryState::D` grows append-only when new class labels appear after `snapshot()`. Pre-2026-07-12, `penalty()` / `apply_pull()` required an *exact* size match with the anchor, so the `D` Fisher term became a no-op the moment task B introduced unseen classes — even with `ewc_lambda > 0`.

2. **NIG world field not anchored (gap, opt-in):** Priority 5 called for "EWC as a post-hoc overlay on the NIG field". `world_mu` / `world_inv_v` are fixed-size shared statistics updated every step (`memory_train.cpp` `world_update`). Added `EwcRegularizer::set_protect_world_field(true)` to anchor `world_mu` with diagonal Fisher `world_inv_v` (no calibration pass needed).

## Shipped changes

| Item | Location |
|------|----------|
| Prefix-only `D` compare/pull | `native/src/ewc_regularizer.cpp` |
| Opt-in `set_protect_world_field()` | `native/include/cypha/ewc_regularizer.hpp` |
| D16B EWC λ sweep (standalone, no bench report) | `run_d16_ewc_sweep()` in `bench_domains.cpp`, `ewc_d16b_sweep` tool |
| Regression tests | `native_ewc_growable_d_smoke`, `native_ewc_d16b_sweep` CTests |

Default behavior unchanged: existing D16B/D16H probes and REST `/train` callers remain byte-identical unless they opt into world-field protection.

## Measured trade-off (FAST sweep, seed 42, 800 steps/task)

Reproduce: `.\native\build_ewc_d16\ewc_d16b_sweep.exe` (stdout JSON; `bench/report/tables/` is gitignored).

| `ewc_lambda` | `protect_world_field` | forgetting_score | task_a_before → after | Notes |
|---:|---|---:|---|---|
| 0 | — | **0.135** | 0.949 → 0.821 | Baseline (EWC off); lower than legacy 0.813 figure (different step budget / data path) |
| 0.1 | false | 0.135 | unchanged | Growable-D fix active; penalty > 0 but no retention gain at this λ |
| 0.1 | true | 0.162 | slight regression | World pull adds stiffness without benefit here |
| 0.5 | false | 0.135 | unchanged | |
| 0.5 | true | 0.189 | worse | |
| 2.0 | false | **0.108** | modest improvement | Best setting in this sweep |
| 2.0 | true | **1.000** | catastrophic | Over-regularized world_mu pull |

**Verdict:** Fixing the growable-`D` bug is necessary for EWC to engage at all (regression-tested via nonzero `ewc_penalty_final`). On this D16B-style multitask probe, classical EWC (D + enc_w) gives at best a **modest** forgetting reduction (0.135 → 0.108 at λ=2.0). NIG world-field protection at λ≥0.5 **hurts** retention — leave opt-in/off unless recalibrated.

Shared-model continual learning remains an open problem; per-task isolation (D16F) still achieves zero forgetting by architecture.

## Commands

```powershell
cmake --build native/build_ewc_d16 --target ewc_d16b_sweep ewc_growable_d_smoke
.\native\build_ewc_d16\ewc_d16b_sweep.exe
ctest --test-dir native/build_ewc_d16 -R "native_ewc_growable_d_smoke|native_ewc_d16b_sweep|native_ewc_d16b_smoke"
```
