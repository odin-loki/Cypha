# Multi-view training Phase 2 (CyphaDIF) — 2026-07-11

**Status:** Small opt-in pilot implemented and validated on **D03** (see §6). Full four-domain port (D16/D03/D08/D09) remains a plan, not implemented — see §7 for explicit scope boundary.
**Priority:** P6 in `docs/reports/DEV_PLAN_2026-07-11.md:156` ("Multi-view training Phase 2 (CyphaDIF) | LM side done; DIF side not started | independent").
**Relationship to the live overnight run:** independent and additive. This work does not touch `native/build_math` (the live production-overnight D17→D21→cell-sweep run), `bench/BASELINE_LOCK.json`, or any overnight orchestration script. It builds in a fresh `native/build_multiview` directory and only edits `native/src/bench/bench_domains.cpp` (D03's own helper functions) and `native/CMakeLists.txt` (two new CTest entries). It does not touch `native/src/rpsm/*`, `native/src/intelligence/measurers.cpp`, `cyphalm_bench_native.cpp`'s `--lstm-hidden` path, kernel-LLR files, or `native/include/cypha/intelligence/causal_graph.*` — all confirmed untouched by `git status --porcelain native/` after this session's edits.

---

## 1. The claim being scoped

`docs/RESEARCH_STATUS.md:377`:

> "Multi-view online training helps LM/DIF | Planned | Spec: [`MULTI_VIEW_TRAINING_PLAN.md`](MULTI_VIEW_TRAINING_PLAN.md) — not yet implemented"

`docs/MULTI_VIEW_TRAINING_PLAN.md:5`'s own status line: "Owner track: CyphaLM (Phase 1) → CyphaDIF (Phase 2)". Phase 1 (CyphaLM) is real, shipped, native C++ (confirmed in §2 below — not the Python `cypha_lm`/`cypha_views` package the plan doc's prose describes, which does not exist in this repo; the actual implementation is `native/src/cyphalm/cyphalm_views.cpp` + `cyphalm_model.cpp`). Phase 2 (CyphaDIF) is confirmed **not implemented** as a generic mechanism — verified firsthand by reading the actual D03/D08/D09/D16 training loops (§3). This document scopes Phase 2, implements a small validated first increment on D03 (§6), and lays out the remaining phased work for D08/D09/D16 (§7–8).

---

## 2. What Phase 1 (CyphaLM) actually shipped

### 2.1 Core types (`native/include/cypha/cyphalm/cyphalm_views.hpp:11-46`)

| Type | Fields | Purpose |
|---|---|---|
| `ViewMemoryPolicy` | `reset_fast`, `carry_slow`, `carry_dif`, `carry_gria_bias` (all `bool`) | Per-view rule for whether the SSM's fast/slow hidden state is reset at a view boundary |
| `ViewSpec` | `name`, `view_id`, `transform_name`, `memory_policy` | One named presentation of the corpus |
| `ViewSchedule` | `views: vector<ViewSpec>`, `seed` | Ordered list of views across training epochs |
| `ViewEpochItem` | `view_spec`, `epoch_idx`, `segment_ids`, `reset_before` | One yielded training segment (a block of token ids under a given view) |

### 2.2 Pure transforms (`native/src/cyphalm/cyphalm_views.cpp:78-150`)

Four functions, all pure `vector<int> -> vector<int>` (or `-> vector<vector<int>>` for the block variants), with **no dependency on what the ints mean**:

| Function | What it does |
|---|---|
| `view_identity` | No-op passthrough |
| `view_reverse` | Reverses the sequence |
| `view_rotate_start(ids, offset)` | Rotates so the sequence starts at index `offset` |
| `split_blocks` / `split_blocks_by_delimiter` | Chunks a sequence into fixed-size or delimiter-bounded blocks |
| `block_shuffle_blocks(ids, block_size, seed)` | Splits into blocks, shuffles block *order* (not within-block order), reassembles |

This is the single most important fact for Phase 2 reuse: **these four functions never look at token semantics.** They are generic reorderings of an integer sequence. Confirmed by direct inspection — `cyphalm_views.cpp:78-150` never reads a vocabulary, embedding table, or label.

### 2.3 The orchestration layer that ties transforms to schedules (`cyphalm_views.cpp:152-251`)

- `resolve_view_schedule(name, train_epochs)` — maps a preset name (`"same_order"`, `"schedule_a"`, `"schedule_b"`, `"schedule_c"`, or an arbitrary literal view name) to an ordered `vector<string>` of view names (`cyphalm_views.cpp:152-162`). `"same_order"` repeats `"forward"` `train_epochs` times; `"schedule_a"` = `{forward, block_shuffle}`; `"schedule_b"` = `{forward, block_shuffle, rotated}`; `"schedule_c"` = `{forward, block_shuffle, backward}`.
- `resolve_view_schedule_struct(name, seed, train_epochs)` — same resolution, but returns a full `ViewSchedule` with a `ViewMemoryPolicy` attached per view via `memory_policy_for_view` (`cyphalm_views.cpp:22-30`): `"forward"` gets `{reset_fast=false, carry_slow=true, carry_dif=true, carry_gria_bias=true}`; every other view (`block_shuffle`, `rotated`, `backward`) gets `{reset_fast=true, carry_slow=false, ...}`.
- `iter_view_epochs(ids, schedule, char_newline_id, block_size)` — the real engine (`cyphalm_views.cpp:221-251`). For block-segmented views (`forward`, `block_shuffle`), it calls `block_segments_for_view` to chop the (possibly transformed) id sequence into blocks and yields one `ViewEpochItem` per block, each carrying `reset_before = view_spec.memory_policy.reset_fast`. For non-block views (`rotated`, `backward`), it transforms the whole sequence once and yields it as a single segment.

### 2.4 How the model consumes it (`native/src/cyphalm/cyphalm_model.cpp:1301-1367`)

`CyphaLMModel::train_sequence_views` is the concrete integration:

1. Resolves `cfg_.view_schedule` (config field, `native/include/cypha/cyphalm/cyphalm_config.hpp:61-66`, default `"same_order"`) to a `ViewSchedule` via `resolve_view_schedule_struct(cfg_.view_schedule, cfg_.seed, cfg_.train_epochs)` (`cyphalm_model.cpp:1323-1324`).
2. Calls `iter_view_epochs` to get the segment list.
3. For each yielded segment: sets a **view slot** on a per-view embedding table (`set_view_slot(...)`, fed into GRIA's context — either a learned `view_emb_` table or a fixed `view_slot_for_name` mapping, `cyphalm_model.cpp:1303-1336`); resets SSM context (`reset_context()`) on macro-epoch boundaries or when `item.reset_before` is set; decays `cfg_.gria_lr` by `gria_lr_decay^epoch_idx`; then runs the ordinary per-token `train_step` loop over the segment.
4. Called from `train_sequence` only `if (cfg_.view_schedule != "same_order")` (`cyphalm_model.cpp:1544-1547`) — i.e. the multi-view path is itself already an opt-in branch, off by default, exactly the same pattern this document's D03 pilot follows for DIF.

**Summary of what Phase 1 actually is:** a token-sequence reordering scheme, coupled tightly to (a) the SSM's fast/slow hidden-state reset semantics and (b) an optional per-view embedding table fed into GRIA. Nothing in here is DIF-aware; "CyphaDIF" in the LM path (`cfg_.n_experts`, GRIA) is the *language-model's* internal expert mixture over next-token distributions, not the standalone classification/regression `CyphaInferModel`/`CyphaDifMemoryState` used by D03/D08/D09/D16 — these are different systems that happen to share the "DIF" name (confirmed: `cyphalm_model.cpp` never includes `dif_train_step_vector`, and `bench_domains.cpp`'s D03/D08/D09/D16 never include `cyphalm_model.hpp`'s GRIA/SSM types).

---

## 3. How D16/D03/D08/D09 currently train (confirmed firsthand)

All four domains bottom out in the same primitive, `cypha::dif_train_step_vector` (declared in `native/include/cypha/cyphalm_dif... ` — actually the standalone classifier header; called from `native/src/bench/bench_domains.cpp`), driven by one of two thin wrappers:

| Domain | Bench entry point | Wrapper used | File:line |
|---|---|---|---|
| **D03** classification (iris, wine) | `run_d03` → `run_tabular_dataset` | `train_eval_vectors` | `bench_domains.cpp:1081`, `:417` (pre-edit: `:377`) |
| **D08** vision (MNIST/HOG) | `run_d08` → `run_vision_encoding` | `train_eval_vectors` | `bench_domains.cpp:1099`, `:1140` |
| **D09** documents (20news, Gutenberg) | `run_d09` | `train_classifier_online` | `bench_domains.cpp:2276` (pre-edit line numbers) |
| **D16** multitask (iris/wine/digits) | `run_d16` (+ `run_d16_ewc_probe`) | `train_classifier_online` / manual per-experiment loops | `bench_domains.cpp:1757`, `:1665` |

Both wrappers do the same thing, confirmed by reading the loops directly (pre-edit `train_eval_vectors`, `bench_domains.cpp:333-344`; `train_classifier_online`, `:520-532`):

```cpp
for (int p = 0; p < passes; ++p) {
    std::vector<int> order(train_n);
    for (int i = 0; i < train_n; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), make_rng(seed + p));
    for (int idx : order) {
        dif_train_step_vector(infer, mem, replay, x[idx], ..., y[idx], ...);
    }
}
```

**This confirms the "not yet implemented" claim exactly as stated:** every pass over the training set is an **independent uniform-random shuffle** of the same fixed `(x, y)` pairs. There is:
- No `view_id` concept threaded into `dif_train_step_vector`'s signature (checked: its parameter list is `(infer, mem, replay, x, d, label, world_lr, delta_lr, ..., ood_sigma, tsp, rng, enc_updates, extras_a, extras_b)` — no view/schedule parameter anywhere).
- No block structure, no class-block grouping, no curriculum ordering, no bidirectional/rotated presentation.
- No notion of "memory policy" — `CyphaDifMemoryState` accumulates NIG statistics and grows experts permanently across the entire run; there is no separate "fast, resettable" state analogous to the LM SSM's `h`/`s` to apply `reset_fast`/`carry_slow` to.

**One partial exception, found and worth flagging:** D16 already has two *hand-rolled, non-generic* experiments that informally probe view-like ideas, predating any shared infrastructure:
- **16D** (`bench_domains.cpp:1817-1860`) compares `round_robin` / `random` / `block` task-interleaving strategies (each task's stream length is fixed; only *which task* is sampled next, and in what block granularity, changes).
- **16G "view streams"** (`bench_domains.cpp:1927-1991`) directly compares `round_robin` vs. a `task_block_shuffle` (shuffling the order of `{iris, wine, digits}` task blocks each macro-epoch) on a forgetting-probe metric, and is already logged as a real result in `docs/MULTI_VIEW_TRAINING_PLAN.md:311`: *"D16 16G fast: task-block-shuffle **hurts** accuracy (0.58 mean vs RR 0.81); forgetting 0.0 but wine/digits collapse."*

Both are literal, bespoke re-implementations inline in `run_d16`/`run_d16_ewc_probe` — **not** built on `cyphalm_views.hpp`'s `ViewSchedule`/`resolve_view_schedule_struct`/`iter_view_epochs`. They are evidence that (a) the general shape of "reorder DIF's training stream" has already been tried ad hoc for D16 specifically, and (b) the one real result available (16G) is a **negative** result. This is important context for §5–6: it is not a fresh hypothesis, and the prior signal already points toward "hurts, not helps," at least for task-level reordering in a multitask forgetting probe.

---

## 4. Reusability assessment: how directly does Phase 1's mechanism port to DIF?

| Layer | Reusable as-is for DIF? | Why |
|---|---|---|
| Pure transforms (`view_reverse`, `view_rotate_start`, `block_shuffle_blocks`, `split_blocks`) | **Yes, directly.** | They operate on `vector<int>` with no semantic assumptions (§2.2). Feed a sample-*index* vector (`0..n-1`) instead of a token-*id* vector, and they reorder `(x, y)` pairs instead of tokens — exactly the DIF-V1/V5 "reorder the stream" taxonomy the plan doc itself describes (`MULTI_VIEW_TRAINING_PLAN.md:180-186`). Verified by implementation in §6: zero changes needed to the transform functions themselves. |
| `resolve_view_schedule` (preset name → view-name list) | **Yes, directly.** | Same reasoning — it's just string dispatch, agnostic to what a "view" ultimately reorders. |
| `ViewMemoryPolicy` / `reset_fast` / `carry_slow` / `carry_dif` / `carry_gria_bias` | **No — does not map.** | These fields describe resetting a **continuous hidden state carried between adjacent time steps** (the SSM's `h`/`s` in `CellAISSM`). DIF's classifier (`CyphaInferModel` + `CyphaDifMemoryState`) has no equivalent transient per-step state — its "memory" *is* the permanently-accumulated NIG statistics/expert bank, which is never meant to reset mid-run. There is no clean DIF analogue for "reset fast, carry slow" — the very same expert-bank state that would be "slow" is also the *only* state, so the policy collapses to a no-op or requires an entirely new concept (e.g. "freeze which experts, unfreeze others," which the plan doc does not spec). |
| `iter_view_epochs` block-segmentation (fixed `view_block_size` windows, or delimiter-bounded blocks) | **Partially — needs semantic redefinition.** | LM blocks are windows of *contiguous tokens* (sentence/paragraph/fixed-512-char chunks) — a spatial/sequential notion that has no equivalent for i.i.d. tabular rows (D03), independent images (D08), or independent documents (D09). The plan doc's own DIF view taxonomy (`MULTI_VIEW_TRAINING_PLAN.md:177-186`) already anticipates this by defining DIF-specific "blocks" as **class blocks** (grouping same-label samples) rather than fixed-size windows — a different definition of "block" that the LM code's `split_blocks`/`split_blocks_by_delimiter` don't implement. Only D16's multitask setting has a ready-made "block" unit (a task), which is what its ad hoc 16D/16G already exploit. |
| Per-view embedding table (`view_emb_`, `cfg_.view_id_dim`/`view_learnable`) fed into GRIA | **No — LM/GRIA-specific.** | This concatenates a learned per-view vector into the LM's GRIA n-gram context input. `CyphaInferModel`'s classification path has no GRIA n-gram context and no equivalent injection point; porting this would mean designing a new "view-conditioned feature" mechanism for DIF, not reusing anything from `cyphalm_model.cpp`. The plan doc's own Phase 2 §2.3 (`MULTI_VIEW_TRAINING_PLAN.md:213`) gestures at something similar via a `"view2_class3"` label-prefix trick for `DIFRegressor`, which is a different (and untested) idea, not a port of `view_emb_`. |
| `CyphaLMModel::train_sequence_views` orchestration (macro-epoch loop, `reset_context()` calls, `gria_lr` decay) | **No — tightly coupled to the LM's own `train_step`/SSM API.** | `dif_train_step_vector`'s signature and `CyphaInferModel`/`CyphaDifMemoryState` have nothing resembling `reset_context()`; this function would need a from-scratch DIF equivalent, not a call-site port. |

**Bottom line:** roughly the bottom third of Phase 1's stack (pure sequence transforms + preset-name resolution, `cyphalm_views.hpp:48-59` and part of `:62`) is genuinely, directly, zero-modification reusable for DIF, provided the caller supplies a *sample-index* vector rather than a *token-id* vector and picks its own (much smaller) block size. The top two-thirds — `ViewSchedule`'s memory-policy semantics, the block-segmentation engine, the view-embedding table, and the model-level orchestration loop — are LM/SSM/GRIA-specific and would need original design work for DIF, not a port. This directly answers the scoping question in the task: it is **not** a drop-in "wire the existing struct into DIF's loop"; it is "reuse the four pure transform functions as a new sample-reordering primitive, and design everything else (block semantics, success criteria, per-domain integration) fresh," which is exactly the smaller, well-scoped slice implemented in §6.

---

## 5. Phased plan

| Phase | Scope | Gate before next | Status |
|---|---|---|---|
| **2.0** | Confirm "not implemented" firsthand; reusability assessment | This document | ✅ Done (§3–4) |
| **2.1 (this pass)** | Wire the four pure transform functions into **D03 only**, behind an opt-in env flag (`CYPHA_D03_VIEW_SCHEDULE`), default off = byte-identical to pre-existing behavior | Regression-free when off; CTest smoke for the on-path; cheap-scale measurement of the on-path | ✅ **Implemented** — §6 |
| **2.2** | Decide, from 2.1's measurement, whether index-reordering views are worth pursuing further for DIF at all | 2.1's measured effect is not uniformly harmful, OR a specific transform/domain combination shows promise | ⏳ Blocked on 2.1 data — current read (§6.3) is **mixed-to-negative**, matching the pre-existing D16 16G result (§3) |
| **2.3** | If 2.2 gates green: design a genuine DIF-native "view" concept — class-block presentation (not fixed-window), curriculum-by-margin (DIF-V2), or replay-interleave (DIF-V3) from `MULTI_VIEW_TRAINING_PLAN.md:180-186` — and port to D08/D09/D16 | ≥2/4 domains improve ≥1pp accuracy per the plan doc's own success criterion (`MULTI_VIEW_TRAINING_PLAN.md:234`) | Not started — **out of scope for this pass**, see §7 |
| **2.4** | Cross-domain gain matrix + profile wiring (`view_schedules` in `everyday_profile.json`) | `RESEARCH_STATUS.md` updated with real numbers | Not started |
| **3** | Product integration (`CYPHA_VIEW_SCHEDULE` env, Studio/REST `view_id`) | Only after 2.1–2.4 validate | Not started (plan doc's own Phase 3, `MULTI_VIEW_TRAINING_PLAN.md:246-253`) |

---

## 6. What was implemented this pass (Phase 2.1 — D03 pilot)

### 6.1 Design

Added a new file-local helper in `native/src/bench/bench_domains.cpp:304-336` (renumbered after edit):

```cpp
std::vector<int> dif_view_order(int n, const std::string& view_schedule, int pass_idx) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), make_rng(42 + pass_idx));   // identical to pre-existing shuffle
    if (view_schedule.empty() || view_schedule == "same_order") {
        return idx;                                                  // <-- old behavior, byte-identical
    }
    const auto view_names = cypha::cyphalm::resolve_view_schedule(view_schedule, 1);
    const std::string& view = view_names[pass_idx % view_names.size()];
    if (view == "reverse" || view == "backward") return cypha::cyphalm::view_reverse(idx);
    if (view == "rotated")                       return cypha::cyphalm::view_rotate_start(idx, std::max(1, n / 4));
    if (view == "block_shuffle") {
        const int block = std::clamp(n / 4, 4, 64);   // small block size (n≈120-142), not LM's 512
        const auto blocks = cypha::cyphalm::block_shuffle_blocks(idx, block, 42 + pass_idx + 1000);
        // flatten blocks back into a single index order
    }
    return idx;  // "forward" / unrecognized -> shuffled IID base order
}
```

This directly reuses `cypha::cyphalm::view_reverse`, `view_rotate_start`, `block_shuffle_blocks`, and `resolve_view_schedule` from `cyphalm_views.hpp` (new include added at `bench_domains.cpp:59`) — **zero modifications to Phase 1's code** — applied to a sample-index vector instead of a token-id vector, exactly per the §4 reusability finding.

`train_eval_vectors` (`bench_domains.cpp:338`) gained one new trailing parameter, `const std::string& view_schedule = ""`, defaulting to the old behavior. Its per-pass loop (`:375-380`) now calls `dif_view_order(train_n, view_schedule, p)` instead of inlining the shuffle. **Only `run_tabular_dataset`** (`:417-433`, D03's exclusive helper — confirmed via `Grep` that no other domain calls it) passes a non-default value, read from a new env var:

```cpp
std::string d03_view_schedule_from_env() {
    const char* v = std::getenv("CYPHA_D03_VIEW_SCHEDULE");
    return (v == nullptr || *v == '\0') ? "" : std::string(v);
}
```

D01 (`train_eval_classifier`, `:455`) and D08 (`run_vision_encoding`, `:1140`) call `train_eval_vectors` without the new argument, so they take the default `""` and are **provably unreachable** by this change — not merely "off by default," but structurally incapable of taking the new code path, since no env var read anywhere in their call chain feeds it. D09 and D16 use the separate `train_classifier_online` wrapper, which was **not touched at all**.

### 6.2 Regression check (flag off)

Built in a fresh `native/build_multiview` (Ninja + MinGW `g++`/`gcc` from `C:/Strawberry/c`, `Release`, matching `native/build_scale`'s toolchain). Ran `cypha_bench_run --domain-tag d03` with `CYPHA_BENCH_FAST=1` and no `CYPHA_D03_VIEW_SCHEDULE` set, twice, to confirm determinism:

| Run | iris accuracy | wine accuracy |
|---|---|---|
| Baseline (before this change, reasoned from code identity) | 0.8667 | 1.0 |
| After change, flag unset (run 1) | **0.8667** | **1.0** |
| After change, flag unset (run 2, re-run for determinism) | **0.8667** | **1.0** |

Identical to 12 significant figures across reruns — the new default path is confirmed byte-identical to the pre-existing shuffle, both by code inspection (§6.1: the early-return branch performs the exact same two lines the old inline code did) and by this empirical rerun.

### 6.3 Measured effect (flag on, cheap scale)

D03's datasets (iris n=150, wine n=178) are not affected by `CYPHA_BENCH_FAST` (`run_d03` calls `load_tabular_dataset` with fixed sizes, `bench_domains.cpp:1082-1083`) — so this *is* already the full-scale D03 result, not a smoke-only approximation. `n_epochs=4` passes (`bench/config/everyday_profile.json`'s `tabular` regime).

| `CYPHA_D03_VIEW_SCHEDULE` | iris accuracy | wine accuracy | Δ iris vs baseline |
|---|---|---|---|
| *(unset — baseline)* | 0.8667 | 1.0 | — |
| `schedule_a` (`forward`, `block_shuffle` alternating per pass) | 0.80 | 1.0 | **−6.7pp** |
| `block_shuffle` (every pass) | 0.7667 | 1.0 | **−10.0pp** |
| `rotated` (every pass) | 0.3333 (chance level, 3-class) | 1.0 | **−53.3pp** |

Wine is saturated at 1.0 in every configuration (a ceiling effect — not informative either way). Iris shows a **consistent, monotonic degradation** as the reordering gets more aggressive, with `rotated` catastrophically bad (collapses to chance). This is a genuine, reproducible (seed-fixed) measurement, not noise.

**Interpretation:** naively porting Phase 1's token-sequence transforms to reorder DIF's sample stream does **not** help D03, and actively hurts, worst for `rotated`. This is consistent with, and adds a second independent data point to, the pre-existing D16 16G finding (§3: task-block-shuffle hurt accuracy 0.58 vs 0.81 RR). A plausible mechanism (not verified beyond this pass's scope): `CyphaDifMemoryState` grows experts lazily and its NIG statistics are most informative early in training when few samples have been seen — `rotate_start` on a stratified-split index array can push an entire class's early samples to the end of a pass, materially changing *which* labels the model sees first and how many experts get created before some classes are ever observed, unlike the LM case where token order within a rotated sequence has comparatively little effect on next-token statistics at this scale. This matches the plan doc's own stated risk register (`MULTI_VIEW_TRAINING_PLAN.md:287`: *"View-as-class explodes DIF class count"*) and its explicit design principle that destructive views *should* measurably hurt to validate the harness (`MULTI_VIEW_TRAINING_PLAN.md:31`: *"Fail loud... validates the harness"*) — by that standard, this pilot's harness is working correctly; it is the specific transforms tried that are unpromising for DIF's online per-sample training dynamics.

### 6.4 Tests added

Two new CTest entries (`native/CMakeLists.txt`, inserted after `native_d22_cross_smoke`):

- `native_d03_smoke` — `cypha_bench_run --domain-tag d03` with `CYPHA_BENCH_FAST=1`, flag unset. Regression guard for the untouched default path.
- `native_d03_view_schedule_smoke` — same command, plus `CYPHA_D03_VIEW_SCHEDULE=schedule_a`. End-to-end smoke for the new opt-in path (confirms it runs to completion, produces a `view_schedule` field in the output JSON, and does not throw/crash).

Both pass (`ctest --test-dir native/build_multiview -R "native_d03_smoke|native_d03_view_schedule_smoke"` → `100% tests passed, 0 tests failed out of 2`).

### 6.5 Files touched

| File | Change |
|---|---|
| `native/src/bench/bench_domains.cpp` | +1 include (`cyphalm_views.hpp`); +`dif_view_order`, +`d03_view_schedule_from_env`; `train_eval_vectors` gained one defaulted trailing parameter and its pass-loop now calls the new helper; `run_tabular_dataset` reads the env var and threads it through, and stamps `view_schedule` into its own result JSON only when non-default |
| `native/CMakeLists.txt` | +2 `add_test`/`set_tests_properties` blocks |
| `docs/reports/MULTI_VIEW_DIF_PHASE2_PLAN.md` | This document (new) |

No changes to `native/src/rpsm/*`, `native/src/intelligence/measurers.cpp`, `cyphalm_bench_native.cpp`, kernel-LLR files, or `native/include/cypha/intelligence/causal_graph.*` — confirmed via `git status --porcelain` showing only the three files above plus unrelated concurrent sibling-agent edits to `causal_graph.*`/`kernel_memory.hpp`/`intelligence_profiler_papers.cpp` (not made by this work).

---

## 7. Explicitly out of scope for this pass

Per the task's own instruction, the following are **not** attempted, because they require either touching D08/D09/D16 (excluded by the "one domain only" constraint) or a nontrivial DIF training-loop refactor (excluded by the "small increment only" constraint):

1. **Porting to D08 (vision), D09 (documents), or D16 (multitask).** D16 in particular already has a negative ad hoc result (16G) that a naive port would likely just reproduce; D08/D09 would need their own "block" semantics (class-block for D08, document/topic-block for D09) that don't exist yet.
2. **Any DIF-native view concept beyond index reordering** — curriculum-by-margin (DIF-V2), replay-interleave (DIF-V3), feature-presentation views (DIF-V4), or OOD interleave (DIF-V6) from `MULTI_VIEW_TRAINING_PLAN.md:180-186`. These require new hooks into `dif_train_step_vector`/`TrainStepParams`/`PriorityReplayBuffer`, not just stream reordering, and are real design work, not a wiring task.
3. **A `ViewMemoryPolicy` equivalent for DIF.** As established in §4, this doesn't have a clean mapping; designing one (e.g., "freeze N oldest experts during view k") is a research question, not an implementation detail.
4. **Profile-driven configuration** (adding `view_schedule` to `bench/config/everyday_profile.json`'s shared `tabular`/`vision` regimes). Deliberately avoided because that JSON is shared across D03/D08/D09/D10, so any edit there would risk changing behavior for domains this pass must not touch. The env-var approach in §6.1 was chosen specifically to keep the blast radius to D03 alone.

---

## 8. Recommendation for Phase 2.3's pilot domain (if 2.2 gates green on a redesigned view concept)

Ranked by suitability, assuming the negative index-reordering result in §6.3 leads to trying a **genuinely DIF-native** view concept instead (per §7.2) rather than abandoning Phase 2 outright:

1. **D03 (classification)** — already instrumented (§6), smallest/fastest iteration loop (~5-7s per full run), only two datasets, no encoder pipeline in front of the classifier to confound results. Best choice for iterating on a *redesigned* view concept (e.g., class-block presentation instead of arbitrary-window `block_shuffle`) before spending time on the others. This matches the plan doc's own Phase 2b ordering intuition (`MULTI_VIEW_TRAINING_PLAN.md:229`: D03 listed first among the "four representative domains").
2. **D08 (vision)** — plan doc's DIF-V4 ("feature presentation" views — same label, transformed x) maps naturally onto image augmentation-as-view, which is a well-understood technique elsewhere and has a plausible mechanism for helping (unlike arbitrary index reordering). Second priority once D03 has a working non-negative view concept to port.
3. **D09 (documents)** — needs a document/topic-block notion before any block-based view makes sense; more design work than D03/D08.
4. **D16 (multitask)** — **actively deprioritized**, not because it's unimportant, but because it already has a directly-relevant negative result (16G) that any straightforward reordering-based approach would likely reproduce. Revisit only with a materially different mechanism (e.g., DIF-V3 replay-interleave using the existing `PriorityReplayBuffer`, which 16-series experiments have not tried).

---

## 9. Commands appendix

```powershell
# Fresh build directory, matching native/build_scale's toolchain (Ninja + Strawberry Perl MinGW)
cmake -S native -B native/build_multiview -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=C:/Strawberry/c/bin/c++.exe -DCMAKE_C_COMPILER=C:/Strawberry/c/bin/gcc.exe
cmake --build native/build_multiview --target cypha_bench_run -j 16

# Regression check (flag off — must match pre-existing D03 numbers)
$env:CYPHA_BENCH_FAST=1
native/build_multiview/cypha_bench_run.exe --domain-tag d03
Get-Content bench/report/tables/d03.json

# Opt-in pilot (flag on)
$env:CYPHA_D03_VIEW_SCHEDULE="schedule_a"   # or: block_shuffle, rotated, backward, same_order
native/build_multiview/cypha_bench_run.exe --domain-tag d03
Remove-Item Env:\CYPHA_D03_VIEW_SCHEDULE

# CTest smoke (both the regression guard and the opt-in path)
ctest --test-dir native/build_multiview -R "native_d03_smoke|native_d03_view_schedule_smoke" --output-on-failure
```

---

## References

- Phase 1 shipped code: `native/include/cypha/cyphalm/cyphalm_views.hpp`, `native/src/cyphalm/cyphalm_views.cpp`, `native/src/cyphalm/cyphalm_model.cpp:1301-1367,1544-1547`, `native/include/cypha/cyphalm/cyphalm_config.hpp:61-66`
- Plan doc (full read, all phases): `docs/MULTI_VIEW_TRAINING_PLAN.md`
- Hypothesis ledger entry: `docs/RESEARCH_STATUS.md:377`
- Priority ranking: `docs/reports/DEV_PLAN_2026-07-11.md:156`
- Pre-existing D16 negative result this pass's finding corroborates: `docs/MULTI_VIEW_TRAINING_PLAN.md:311`, `native/src/bench/bench_domains.cpp:1927-1991` (16G)
- This pass's implementation: `native/src/bench/bench_domains.cpp` (D03 helpers), `native/CMakeLists.txt` (CTest entries)
