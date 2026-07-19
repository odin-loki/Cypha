# Multi-view early-stop policy (CyphaLM + D16) — 2026-07-17

**Scope:** Bill of Work §1 **P4** — document when to stop / which schedule to use for GRIA-only multi-view runs, and record the D16 **16G** task-block-shuffle regression. Did not touch `build_math`, `build_deff`, `BASELINE_*`, or overnight.

**Canonical plan:** [`docs/MULTI_VIEW_TRAINING_PLAN.md`](../MULTI_VIEW_TRAINING_PLAN.md)  
**Sweep artifacts:** `bench/config/cyphalm_view_iteration_sweep.json` (2k–40k), `bench/config/cyphalm_convergence_limit.json` (40k–250k)

---

## 1. CyphaLM early-stop policy (GRIA-only / pre-hybrid)

There is **no automatic early-stop hook** in native CyphaLM today. Training length is controlled manually:

| Knob | Where | Effect |
|------|-------|--------|
| `--n-train N` | `cyphalm_bench_native` | Hard cap on training tokens (default **40 000**) |
| `view_schedule` | profile JSON / `CyphaLMConfig` | `"same_order"`, `"schedule_a"`, `"schedule_b"`, … |
| `train_epochs` | profile JSON | Repeats the resolved view list (`same_order` = forward × N) |
| `recommended_n_train` | profile `_meta` | Documentation only (not enforced) |

**Policy (from 32-run iteration sweep + convergence extension):**

| Training budget | Recommended schedule | Stop rule | Evidence |
|-----------------|---------------------|-----------|----------|
| **≤ 24 000 tokens** | **`schedule_b`** (`forward → block_shuffle → rotated`) | Stop at budget cap; do **not** switch to same-order mid-run | `schedule_b` wins at every grid point ≤ 32k; @ 24k BPC **3.474** vs `same_order_e2` **3.648** (−0.174) |
| **40 000 tokens (full short budget)** | **`same_order` + `train_epochs: 2`** | **Hard stop @ 40k** — BPC rises after 50k on same-order | Convergence sweep: `same_order_e2` peaks @ 40k (**4.094** BPC), regresses @ 50k |
| **Bigram-target smoke (12k–16k)** | **`schedule_b`**, one macro-pass | Early-stop when held-out BPC crosses above bigram | `same_order_e1` crosses **above** bigram between ~12k–16k; multi-view stays below longer |
| **Extended GRIA-only (70k–300k)** | **`schedule_b`** only if continuing past 40k | Stop **`schedule_b` @ ~300k** (regresses @ 400k); local min ~150k (**3.958**) | `cyphalm_convergence_limit.json` + continue sweep |

**Do not confuse with production hybrid:** default D17 profile (`hybrid_gria_lstm`, `recommended_n_train: 300000`, `view_schedule: schedule_b`) already beat bigram @ 300k (**2.873 BPC**). The table above governs **GRIA-only / view ablation** runs and short-budget tuning — not the locked hybrid overnight path.

### Example commands

```powershell
# Short budget — schedule_b @ 24k (policy default for ≤24k)
cyphalm_bench_native --profile d17 --mode gria_ngram --n-train 24000 --n-eval 2000 `
  --view-schedule schedule_b --train-epochs 2

# Full 40k short cap — same_order × 2 epochs
cyphalm_bench_native --profile d17 --mode gria_ngram --n-train 40000 --n-eval 2000 `
  --view-schedule same_order --train-epochs 2
```

Profile `_meta.multiview_policy` in `bench/config/profiles/cyphalm_d17_wikitext.json` mirrors this table for humans/tools; it is **not** parsed by the trainer.

---

## 2. D16 16G — task-block-shuffle regression

### What 16G measures

`run_d16()` experiment **16G** (`native/src/bench/bench_domains.cpp`) compares two multitask stream policies after a short iris warm-start:

| Stream | Behavior |
|--------|----------|
| **`round_robin`** | Fixed task order `{iris, wine, digits}` each macro-epoch; one sample per task per round |
| **`task_block_shuffle`** | Shuffles `{iris, wine, digits}` task-block order each macro-epoch; still one sample per task per round |

Both use the same step budget (`bench_scale(3000, 1500)` FAST / full).

### Observed result (FAST, 2026-05-31)

Logged in [`MULTI_VIEW_TRAINING_PLAN.md`](../MULTI_VIEW_TRAINING_PLAN.md) Phase 2a:

| Metric | `round_robin` | `task_block_shuffle` |
|--------|---------------|----------------------|
| Mean per-task accuracy | **~0.81** | **~0.58** |
| Task-A forgetting score | baseline | **0.0** (misleading) |
| Failure mode | — | **wine / digits collapse** — low forgetting score because iris retention looks fine while other tasks never train evenly |

**Verdict:** `task_block_shuffle` is a **negative control**, not a candidate for production multitask scheduling. Forgetting score alone is insufficient — always inspect **per-task accuracy** (`per_task_accuracy` in 16G JSON).

### Why no code fix shipped

The regression is reproducible and intentional as a harness sanity check (destructive reorder **should** hurt). A one-line switch back to round-robin would erase the comparison. No opt-in mechanism (env gate, profile flag) changes the underlying DIF expert-growth dynamics that make task-block permutation harmful on this probe.

Corroborated independently by D03 view-schedule pilot (`docs/reports/MULTI_VIEW_DIF_PHASE2_PLAN.md` §6): index reordering on tabular DIF **hurts** accuracy, worst for `rotated`.

### Next experiment (BoW P4 open item)

1. **Do not wire `task_block_shuffle` into `everyday_profile.json` or REST `/train`.** Keep 16G as bench-only negative control.
2. **Try DIF-V3 replay interleave** (`PriorityReplayBuffer` as its own view) — not yet exercised in 16-series; distinct mechanism from task-block permutation.
3. **Production multitask path:** prefer **D16F per-task isolation** (zero forgetting by architecture) or **EWC overlay (16H / D16B λ sweep)** for shared-model retention — not view reordering.
4. Re-bench 16G at **production step count** (non-FAST) before any D16 view-schedule port; confirm RR baseline holds.

---

## 3. BoW P4 disposition

| Item | Status |
|------|--------|
| Document early-stop policy | **Done** — this report + profile `_meta` + CLI default comment |
| Port multi-view to CyphaDIF (D16) | **Open** — D03 opt-in pilot only; D16 remains ad hoc 16D/16G |
| Fix 16G task-block-shuffle regression | **Open** — documented negative result; next experiment = replay-interleave (§2), not block-shuffle tuning |

---

## References

- `docs/MULTI_VIEW_TRAINING_PLAN.md` — Phase 1 sweep recommendations (lines 325, 340)
- `docs/FINDINGS_CYPHALM_TRAINING.md` — convergence limits, same-order @ 40k cap
- `docs/reports/MULTI_VIEW_DIF_PHASE2_PLAN.md` — D03 pilot + 16G negative context
- `native/src/bench/bench_domains.cpp` — `16G_view_streams` (~2388)
- `native/tools/cyphalm_bench_native.cpp` — `--n-train` default 40k
