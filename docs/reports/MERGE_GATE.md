# Merge M4: the gate's structure experts, grouped

**Odin Loch** · 2026-11-06, checks rerun 2026-11-12 · branch `wip/merge-gate` · GENERATION_PLAN §13.40 M4

**Outcome first.** The refactor is exact. Every expert's score is bit-identical, and the dev GEN files of the planner (the default, text-only store `e9_text`) and of `--k5-struct` are equal under `cmp`. Gate time falls by about 43% under the planner and about half under `--k5-struct` (the median of every run). Peak RAM falls 26 MB (planner) and 291 MB (`--k5-struct`). Wall time does not move beyond this PC's noise: the gate is a small part of either writer. The code grows by 30 lines (19 code lines).

**2026-11-12.** The branch was merged with main 9cbd277 (Q1 and Q2 as flags; the discourse planner, Q3t, is the default; O5). Both dev GEN checks were run again on the merged tree, old and new started together: both are still equal under `cmp` (section "Recheck on main 9cbd277" below).

All changes are in `native/include/cypha/lm/structure/experts.hpp`. No other file changed. The branch was merged with main 34da0ec (M1, M2, the text-only planner store) before the checks below, and every check was run again on the merged tree.

## Where the time went

A harness (`gate_bench`) runs the experts on H's drafts (K5_STRUCTURE.md: a real next sentence and 8 K5 drafts per boundary). Here it used the first 2,000 boundaries, one thread, and contexts that grow by one sentence the way the writers grow them. Lap timers split `prepare` (once per context) and `features` (once per candidate) by expert. Old and new ran at the same time, one thread each. Microseconds per call:

| Part | S10 old | S10 new | S210 old | S210 new |
|---|---:|---:|---:|---:|
| **prepare**, all | 247 | 255 | 323 | 322 |
| · LDA inference of the context's topics | 214 | 224 | 279 | 280 |
| · names, ids, classes of the context | 14 | 13 | 20 | 18 |
| · entity grid of the context | 15 | 15 | 18 | 18 |
| **features**, all | 127 | **63 (−51%)** | 261 | **116 (−56%)** |
| · class n-gram | 76 | **28 (−63%)** | 187 | **66 (−65%)** |
| · novelty: longest run in K5's training text | 17.5 | **7.4 (−58%)** | 25.6 | **9.9 (−61%)** |
| · topic | 13.1 | 9.4 | 15.9 | 11.8 |
| · relations | 6.8 | 5.4 | 13.9 | 11.7 |
| · entity grid | 5.8 | 5.1 | 7.9 | 6.7 |
| · novelty: longest run shared with the context | 3.3 | 3.2 | 4.5 | 4.4 |
| · rhythm, reference, transitions, time | 4.3 | 3.8 | 6.4 | 5.1 |

S10 holds the planner's statistics; S210 holds `--k5-struct`'s. The writers call `prepare` once per sentence and `features` 8 times. So in the old code the class n-gram was about half of the gate with S10 and 62% with S210. LDA inference was a sixth and an eighth.

The class n-gram looked up two tables per level: the context's count and number of followers in `cctx_`, then the n-gram's count in `cng_`. Both were binary searches over tens of millions of entries. A probe of the 415,793 class queries of those 2,000 boundaries showed that all of them reach level 2, nearly all level 3, 98% level 4 and 84% level 5 (S210). A context has 256, 198, 100 and 39 followers on average at levels 2 to 5.

## What was grouped

1. **The class n-gram's two tables are one.** A context's count is the sum of its n-grams' counts, and its number of followers is how many there are. Its n-grams lie together in `cng_` (sorted by order, then context, then last class). A check over both statistics files found every one of the 1,742,138 (S10) and 15,290,751 (S210) stored contexts equal to the sum and count of its n-grams. So `cctx_` (42 MB for S10, 367 MB for S210) is no longer held. In its place:
   - a dense index of the contexts of one and two classes (512 + 2¹⁸ entries, 4.2 MB): the first n-gram, how many, and their total;
   - for each 3-gram and 4-gram, where its followers start (4 bytes each: 6.8 MB for S10, 61 MB for S210).
   A context of 3 or 4 classes is found from its first two classes by stepping down to each next class's followers, with a short search each time. Its total is summed over its followers as an integer. The n-gram's own count is a short search inside the context's followers. The formula, and its order of operations, are unchanged.
   The file format is unchanged: `save` writes the context table again from `cng_`, byte for byte, and `load` skips it.
2. **Reference and time share one pass over the candidate's tokens.** Reference read every token's pronoun class twice (agreement, then the first he/she); time read every token again for years.
3. **Topic counts content words and sums their log ratio in one pass.** It called `lda.hpp`'s `content_count` and `log_ratio`, each looking up every token's content-word id. The sum is the same expression in the same order.
4. **Relations looks a name run up once.** The co-occurrence part and the slot part each joined, hashed and looked up the same name. `Stats::build` (counting the slots) and `features` (scoring them) had two copies of the same slot walk. They now share one (`Stats::slot_walk`).
5. **Novelty's training-run sets get a bucket table.** This is an index, not a merge. Each order's sorted list of n-gram hashes gets the start of each value of the hash's top 16 bits (65,537 entries per order, 3.1 MB in all). A lookup searches about 20 hashes instead of 1.3 million, with the same answer.
6. `Context::names` and `Context::name_str` were filled or cleared but never read. They are gone.

**Not grouped, and why:**
- **Tokenising and class-mapping a candidate once.** This already holds. No expert tokenises: the candidate arrives as K5's tokens and ids. Only the class expert maps ids to classes, once per token. The name runs are found once per candidate and serve relations and reference. The entity grid's roles are found once and serve both grid features.
- **Class, rhythm and transitions over one class-code sequence.** In this code only the class expert reads class codes. Rhythm reads the length bucket and the rendered bytes; transitions read the first tokens. There is no shared pass to make.
- **Topic with the entity grid.** Both read the candidate's tokens, but their per-token work lives in `lda.hpp` and `entity_grid.hpp` (content-word ids against capitalised runs, determiners and stems). Sharing it would mean copying both into `experts.hpp`. After the change they cost 15 to 18 µs per candidate together. Not done.
- **LDA inference in `prepare`** is now the largest single cost of the gate. It is a Gibbs sampler in `lda.hpp`, seeded by the context's words. It cannot be shared or reordered and stay exact. Not touched.
- **Tried and dropped:**
  - a branch-free binary search with prefetching. On the class queries it was no faster than `std::lower_bound` at levels 4 and 5 (slower at level 4 with S210).
  - keeping each 3-gram's and 4-gram's follower total beside where its followers start, so that a context's total is read, not summed. It was exact, but it barely helped: in three runs the class expert was 2–11% faster with S210 and 0–8% slower with S10. It needs 4 more bytes per entry (61 MB more for S210). The sum runs over about 140 neighbouring entries, which is cheap.
- **Not tried:** carrying the class n-gram's position from one token to the next. That would cut the class expert further, but it needs a stateful interface and about 20 more lines.

## Exactness

Old is main 34da0ec (`~/m4r/build_old`, run from a copy of main); new is this branch merged with it (`~/build_m4`, run from a copy of the worktree). Same compiler and flags (GCC 13.3, Release). Each pair was started together, 4 threads each.

| Check | Result |
|---|---|
| Planner (`sample --k5`, the default, store `e9_text`), dev GEN file: `scripts/gen_eval.sh dev … --prompts 250 --k5 fuzzy/train.txt` | **equal** (`cmp`, 200,254 bytes); `sample` stdout equal; the SCRIBE v2 and v3.1 tables and the summary equal apart from the run's label. SCRIBE v2 0.511, v3.1 0.511, both. |
| The same GEN file against the checked-in P1t file (`bench/k5_planner/P1t_s0.gen.txt`, CRs stripped) | **equal** |
| `--k5-struct ~/k5s_data/runs/S210.Rnok5.chosen.model` (W5, as MERGE_MIXERS.md used), dev GEN file | **equal** (`cmp`, 200,125 bytes); stdout equal; the SCRIBE tables and summary equal apart from the label. SCRIBE v2 0.453, v3.1 0.483, both. |
| The same GEN file against the checked-in W5 file (`bench/k5_structure/W5_s0.gen.txt`, CRs stripped) | **equal** |
| The timer builds' GEN files (13 planner and `--k5-struct` pairs), old against new and against the plain builds' | equal |
| Every feature of every candidate as a raw double, and the selectors, on H's 10,572 boundaries × 9 candidates (1,945,248 doubles), S10 and S210, fresh contexts (as `k5_structure features`) and growing contexts (as the writers) | **equal** (`cmp`), all four |
| `k5_structure features` on H with S210 (the meta-mixer's training input) | **equal** (`cmp`), and equal to K5_STRUCTURE.md's `H.S210.feat` |
| `S10.stats` loaded and saved again | **equal** (`cmp`): the context table written again is byte-identical |
| Statistics built from the same 5 MB of S | equal in every section, key and count. The files differ only in the 4 padding bytes of each 16-byte (key, count) pair. The old code writes those from uninitialised memory too: two runs of the old code differ in the same bytes. |
| Selftests | `k5_structure` ok (old and new), `k5_planner` ok, `cyphalm_trie` ok, `scribe` 120 of 120 |

There is no floating-point reordering anywhere. Every sum is taken over the same terms in the same order, and the sums that are whole numbers are exact.

Before the merge with main (2026-11-05, against main 7265b3e, with the `e9_aug` store and W1's model `S210.R.chosen.model`), the same GEN checks were also equal.

## Recheck on main 9cbd277 (2026-11-12)

Main had moved on since 34da0ec: the discourse planner (Q3t) is the default writer, and Q1 and Q2 came in as flags. `experts.hpp` did not change on main, so the merge was clean. Old is main 9cbd277 (`~/j3/main`, built to `~/j3/b_main`); new is this branch merged with it, 799743a (`~/j3/m4`, built to `~/j3/b_m4`). Same compiler (GCC 13.3) and flags (CMake Release). The two pairs ran at the same time, each pair's old and new started together, 4 threads each (`K5_THREADS=4`). The PC's load average was about 15.

| Check | Result |
|---|---|
| Default writer, the discourse planner on `e9_text`: `scripts/gen_eval.sh dev … --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt` | GEN file (200,638 bytes) and `sample` stdout **equal** (`cmp`); SCRIBE v2 and v3.1 tables and the summary equal apart from the run's label. SCRIBE v2 0.606 [0.572, 0.642], copy32 0.0385, both. |
| The same GEN file against the checked-in `bench/k5_planner/Q3t_s0.gen.txt` (CRs stripped) | equal apart from the MODEL line (as §13.48 found for main itself) |
| `--k5-struct ~/k5s_data/runs/S210.Rnok5.chosen.model` (W5), dev GEN file | GEN file (200,125 bytes) and stdout **equal** (`cmp`); SCRIBE tables and summary equal apart from the label |
| The same GEN file against `bench/k5_structure/W5_s0.gen.txt` (CRs stripped) | **equal** |

Sample step under `/usr/bin/time -v`:

| Writer | Wall, old → new | CPU (user), old → new | K5 "generated … in", old → new | Peak RSS, old → new |
|---|---:|---:|---:|---:|
| Discourse planner (default) | 2:41.3 → 2:41.5 | 571.4 → 570.7 s | 140.3 → 139.6 s | 630 MB → 604 MB (−26 MB) |
| `--k5-struct` | 3:45.4 → 3:41.3 | 816.5 → 802.9 s (−1.7%) | 201.4 → 197.3 s | 1,685 MB → 1,379 MB (−306 MB, −18%) |

The gate timers below were not run again: `experts.hpp` is the same file, and the gate is under 1% of either writer's time.

## Gate time

The same timer was added to the old and new code: the time inside `Experts::prepare` and `Experts::features`, summed over threads. The first series measured wall time only. The later series also measured each thread's own CPU time (`CLOCK_THREAD_CPUTIME_ID`), which does not grow when other jobs take the thread off its core. A third timer split the planner's work into its phases. Old and new ran `sample` side by side on 4 threads each, with `gen_eval.sh`'s flags, the start order swapped each time. The PC was shared with other jobs throughout: its load average was 12 to 100.

Each cell is prepare + features = the gate, in seconds summed over the 4 threads. Runs are in time order.

| Writer | Run (started first) | Old, wall | New, wall | Change | Old → new, CPU | Change |
|---|---|---:|---:|---:|---:|---:|
| Planner | 1 (old) | 0.065 + 0.392 = 0.457 | 0.066 + 0.197 = 0.263 | **−42.5%** | | |
| Planner | 2 (new) | 0.065 + 0.469 = 0.534 | 0.064 + 0.237 = 0.301 | **−43.6%** | | |
| Planner | 3 (old) | 0.230 + 1.963 = 2.193 | 0.194 + 1.896 = 2.090 | −4.7% ¹ | | |
| Planner | 4 (old) | 0.060 + 0.428 = 0.489 | 0.062 + 0.219 = 0.281 | **−42.5%** | 0.494 → 0.286 | **−42.1%** |
| Planner | 5 (new) | 0.055 + 0.387 = 0.442 | 0.057 + 0.195 = 0.251 | **−43.2%** | 0.447 → 0.256 | **−42.7%** |
| Planner | 6 (old) | 0.067 + 0.465 = 0.532 | 0.067 + 0.230 = 0.297 | **−44.2%** | 0.538 → 0.302 | **−43.9%** |
| `--k5-struct` | 1 (old) | 1.110 + 7.465 = 8.575 | 1.084 + 3.203 = 4.286 | **−50.0%** | | |
| `--k5-struct` | 2 (new) | 1.160 + 7.898 = 9.058 | 1.203 + 9.051 = 10.254 | +13.2% ² | | |
| `--k5-struct` | 3 (old) | 1.099 + 8.082 = 9.181 | 1.079 + 3.067 = 4.146 | **−54.8%** | 7.953 → 4.197 | **−47.2%** |
| `--k5-struct` | 4 (new) | 1.193 + 8.572 = 9.765 | 1.373 + 7.193 = 8.566 | −12.3% ² | 9.675 → 5.325 | **−45.0%** |
| `--k5-struct` | 5 (old) | 1.365 + 11.307 = 12.672 | 1.364 + 29.487 = 30.851 | +143.5% ³ | 12.765 → 30.909 | +142.1% ³ |
| `--k5-struct` | 6 (new) | 2.410 + 22.830 = 25.240 | 2.216 + 9.565 = 11.781 | **−53.3%** | 21.157 → 10.508 | **−50.3%** |
| `--k5-struct` | 7 (old) | 1.800 + 14.327 = 16.127 | 1.845 + 5.815 = 7.660 | **−52.5%** | 15.528 → 7.586 | **−51.1%** |
| `--k5-struct` | 8 (new) | 1.284 + 8.274 = 9.558 | 1.316 + 3.602 = 4.919 | **−48.5%** | 9.646 → 4.991 | **−48.3%** |
| `--k5-struct` | 9 (old) | 1.342 + 8.620 = 9.962 | 1.342 + 3.594 = 4.936 | **−50.5%** | 10.055 → 5.008 | **−50.2%** |

**Medians.** Planner: wall −42.9% (6 runs), CPU −42.7% (3 runs). `--k5-struct`: wall −50.0% (9 runs), CPU −48.3% (7 runs). In 11 of the 15 runs the gate fell by 42–55%. The PC's other jobs held up the other four:
1. Both sides' gate times grew four times over in that run (the load rose as another session started a large build). Old and new were hit alike.
2. The new run's threads were blocked: 21,344 and 11,380 voluntary context switches, against about 2,000 in every other run. Its peak RSS was 1,337 MB, against about 1,390 MB in its other runs, with 23,000–33,000 more minor faults: the kernel took pages from it and it faulted them back. That costs wall time, not CPU time: in run 4 the gate's CPU time still fell 45%.
3. The new run's gate CPU time grew seven times over (`features` 29.5 s, against 3.1–4.1 s in runs 3, 4, 8 and 9). Its `prepare` took the same CPU time as old's (1.37 s each), the rest of its sampling used the same CPU time as old's (920 and 918 s), and its faults and context switches were normal. Old's gate also slowed (12.8 s, against 8 to 10 s). The cause was not found. About 20 minutes later Windows had 6.6 GB of its 64 GB free, and the WSL VM was not fully resident (46 of its 50 GB). Paging by the host stalls a guest's memory reads without any fault inside the guest. That is a likely cause, not a proven one. The four runs after it were normal.

**Notes.**
- On dev the planner plans 77 of the 250 prompts. It calls `prepare` 191 times and `features` 1,422 times. `--k5-struct` calls them 2,247 and 17,976 times.
- `features` falls by about half. `prepare` does not move, because LDA inference is most of it.
- The class n-gram also scores the planner's skeletons (step 2, outside the gate). That step fell from 1.70–2.22 to 0.62–0.87 thread-seconds (−60% to −67%, all six planner runs).
- The class-query probe (the 415,793 class queries of 2,000 boundaries, one thread, old and new together, load about 75): one lookup went from 6.28 to 1.88 µs (S10) and from 17.89 to 4.34 µs (S210), with the same sum to the last digit. On 2026-11-05, at a lower load: 3.30 to 1.01 µs and 7.08 to 2.02 µs.

## Wall time and peak RAM

The exactness runs above (`gen_eval.sh`, old and new started together on 4 threads each, load average about 20), sample step under `/usr/bin/time -v`:

| Writer | Sample wall, old → new | Sampling alone ("generated … in"), old → new | Sample peak RSS, old → new |
|---|---:|---:|---:|
| Planner | 2:59.1 → 2:54.9 | 157.1 → 152.2 s | 628 MB → 602 MB (−26 MB, −4.1%) |
| `--k5-struct` | 3:21.7 → 3:15.6 | 178.9 → 172.5 s | 1,684 MB → 1,393 MB (−291 MB, −17.3%) |

- The gate is a small part of either writer: under 0.1% of the planner's sampling CPU time (0.25–0.54 s of 537–640 s) and about 1% of `--k5-struct`'s (4–10 s of 700–980 s in the normal runs). K5's own drafting and filling is the rest. So halving the gate cannot move the wall clock.
- Sampling time alone, over all 16 old/new pairs: the planner −3.1% to +2.7%; `--k5-struct` −3.6% to +1.1%, apart from the run of note 3 (+10.4%).
- Peak RSS in every timer run: the planner 628–636 MB → 600–610 MB; `--k5-struct` 1,646–1,684 MB → 1,337–1,393 MB.

**The planner's few percent, explained.** The first runs (2026-11-05) found the new planner binary 0.8–5.7% slower in every pairing, and left it open. The phase timer answers it.
- The planner's phases that run none of `experts.hpp` (the prompts that plain K5 writes, and K5's own gate after the plan) moved by −0.9% to +3.5% in the six planner runs, like the total.
- Its expert phases fell in every run: the skeleton step by 60–67%, the gate by 42–44% (bar run 3).
- Plain K5 alone (`--no-k5-plan`: no planner, no structure statistics, no experts) differed between the two binaries by +0.6% and −1.0% (the timer builds) and −4.0% and −3.9% (the plain builds), with identical GEN files.
- So those few percent come from the binaries' layout or from the PC, not from the merge. On the merged tree the plain planner pair was 3.1% faster with the new binary.

## Lines of code

"Code" counts lines that are not blank and not only a `//` comment.

| File | Lines before | Lines after | Code before | Code after |
|---|---:|---:|---:|---:|
| `lm/structure/experts.hpp` | 921 | 951 (+30) | 814 | 833 (+19, +2.3%) |

The context table's build code and the duplicated loops went. The class index, the bucket table and the shared slot walk came in.

## §13.40 M4 predictions, scored

Registered: "M4: exact; gate time down at least 20%."

| Prediction | Result | Score |
|---|---|---|
| exact | every feature bit-identical on all of H (S10, S210, fresh and growing contexts); both dev GEN files equal under `cmp`, and equal to the checked-in P1t and W5 files | **hit** |
| gate time down at least 20% | the gate's own time (prepare + features, summed over threads): median −42.9% wall and −42.7% CPU under the planner, −50.0% wall and −48.3% CPU under `--k5-struct` | **hit** |

**The keep rule** (§13.40: kept if exact and it lowers RAM or time or code with nothing worse): it is exact, and it lowers gate time and RAM. It adds 19 code lines (2.3%). That is the one thing worse, and the lead decides whether it counts. Rechecked on main 9cbd277 (2026-11-12): still exact, peak RAM −26 MB (planner) and −306 MB (`--k5-struct`).

## Files

- `native/include/cypha/lm/structure/experts.hpp` (the only code change).
- The 2026-11-12 recheck: `/mnt/d/cypha_llm/merge/m4/r1112/` (`P_*` the default writer, `S_*` `--k5-struct`, `old` and `new`, with `.sample.time`).
- Runs on Odin's PC, `/mnt/d/cypha_llm/merge/m4/rerun/`: `gen/` (the `gen_eval` outputs of the exactness runs, with `.sample.time`), `gate/` and `gate2/` (the timer runs), `control/` (plain K5 and the class-query probe), `harness/` (the feature checks, the profile, the selftests), `v2/` (the dropped follower-total variant). The first runs, before the merge with main, are in `/mnt/d/cypha_llm/merge/m4/` (`P_*`, `S_*`, `gate/`, `layout/`).
- Not committed (scratch tools): the harness `gate_bench.cpp` and its lap-timer variant, the timer patches (`mktimer2.py`, `m4_mkplantimer.py`), the class-query probe, the run scripts.
