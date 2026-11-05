# Merge M4: the gate's structure experts, grouped

**Odin Loch** · 2026-11-05 · branch `wip/merge-gate` · GENERATION_PLAN §13.40 M4

**Outcome first.** The refactor is exact: every expert's score is bit-identical, and the planner's and `--k5-struct`'s dev GEN files are equal under `cmp`. Gate time falls 40% under the planner and 50% under `--k5-struct`. Peak RAM falls 27 MB (planner) and 289 MB (`--k5-struct`). Wall time does not move: the gate is under 2% of either writer's work. The code grows by 30 lines (19 code lines).

All changes are in `native/include/cypha/lm/structure/experts.hpp`. No other file changed.

## Where the time went

A harness ran the experts on H's drafts (K5_STRUCTURE.md: a real next sentence and 8 K5 drafts per boundary). It used the first 2,000 boundaries, one thread, and contexts that grow by one sentence the way the writers grow them. Lap timers split `prepare` (once per context) and `features` (once per candidate) by expert. Base code, microseconds per call:

| Part | S10 (planner's statistics) | S210 (`--k5-struct`'s) |
|---|---:|---:|
| **prepare**, all | 260 | 282 |
| · LDA inference of the context's topics | 224 | 243 |
| · names, ids, classes of the context | 17 | 18 |
| · entity grid of the context | 15 | 16 |
| **features**, all | 156 | 234 |
| · class n-gram | **95** | **167** |
| · novelty: longest run in K5's training text | 24 | 24 |
| · topic | 15 | 15 |
| · relations | 7.6 | 12.7 |
| · entity grid | 6.1 | 6.8 |
| · novelty: longest run shared with the context | 3.4 | 3.6 |
| · rhythm, reference, transitions, time | 4.6 | 5.1 |

The writers call `prepare` once per sentence and `features` 8 times. So the class n-gram was about half of the gate (62% with S210), and LDA inference a sixth.

The class n-gram looked up two tables per level: the context's count and number of followers in `cctx_`, then the n-gram's count in `cng_`. Both were binary searches over tens of millions of entries. A second probe (on the 415,793 class queries of those 2,000 boundaries) showed that all of them reach level 2, nearly all level 3, 98% level 4 and 84% level 5 (S210). A context has 256, 198, 100 and 39 followers on average at levels 2 to 5.

## What was grouped

1. **The class n-gram's two tables are one.** A context's count is the sum of its n-grams' counts, and its number of followers is how many there are. Its n-grams lie together in `cng_` (sorted by order, then context, then last class). A check over both statistics files found every one of the 1,742,138 (S10) and 15,290,751 (S210) stored contexts equal to the sum and count of its n-grams. So `cctx_` (42 MB for S10, 367 MB for S210) is no longer held. In its place:
   - a dense index of the contexts of one and two classes (512 + 2¹⁸ entries, 4.2 MB): the first n-gram, how many, and their total;
   - for each 3-gram and 4-gram, where its followers start (4 bytes each: 6.8 MB for S10, 61 MB for S210).
   A context of 3 or 4 classes is found from its first two classes by stepping down to each next class's followers, with a short search each time. Its total is summed over its followers as an integer. The n-gram's own count is a short search inside the context's followers. The formula, and its order of operations, are unchanged.
   The file format is unchanged: `save` regenerates the context table from `cng_` byte for byte, and `load` skips it.
2. **Reference and time share one pass over the candidate's tokens.** Reference read every token's pronoun class twice (agreement, then the first he/she); time read every token again for years.
3. **Topic counts content words and sums their log ratio in one pass.** It called `lda.hpp`'s `content_count` and `log_ratio`, each looking up every token's content-word id. The sum is the same expression in the same order.
4. **Relations looks a name run up once.** The co-occurrence part and the slot part each joined, hashed and looked up the same name. `Stats::build` (counting the slots) and `features` (scoring them) had two copies of the same slot walk. They now share one (`Stats::slot_walk`).
5. **Novelty's training-run sets get a bucket table.** This is an index, not a merge. Each order's sorted list of n-gram hashes gets the start of each value of the hash's top 16 bits (65,537 entries per order, 3.1 MB in all). A lookup searches about 20 hashes instead of 1.3 million, with the same answer.
6. `Context::names` and `Context::name_str` were filled or cleared but never read. They are gone.

**Not grouped, and why:**
- **Class, rhythm and transitions over one class-code sequence.** In this code only the class expert reads class codes. Rhythm reads the length bucket and the rendered bytes; transitions read the first tokens. There is no shared pass to make.
- **Topic with the entity grid.** Both read the candidate's tokens, but their per-token work lives in `lda.hpp` and `entity_grid.hpp` (content-word ids against capitalised runs, determiners and stems). Sharing it would mean copying both into `experts.hpp`. After the change they cost about 13 µs per candidate together. Not done.
- **LDA inference in `prepare`** is now the largest single cost of the gate. It is a Gibbs sampler in `lda.hpp`, seeded by the context's words, and cannot be shared or reordered and stay exact. Not touched.
- **Tried and dropped:** a branch-free binary search with prefetching. On the class queries it was no faster than `std::lower_bound` at levels 4 and 5 (slower at level 4 with S210).
- **Not tried:** carrying the class n-gram's position from one token to the next. That would cut the class expert further, but it needs a stateful interface and about 20 more lines.

## Exactness

Every check compares the old code (main 7265b3e) with the new.

| Check | Result |
|---|---|
| Every feature of every candidate as a raw double, and the selectors, on H's 10,572 boundaries × 9 candidates (1,945,248 doubles), S10 and S210, fresh contexts (as `k5_structure features`) and growing contexts (as the writers) | **equal** (`cmp`), all four |
| Planner (`sample --k5`, the default), dev GEN file, `scripts/gen_eval.sh dev … --k5 fuzzy/train.txt` | **equal** (`cmp`, 200,295 bytes); stdout equal; SCRIBE v2 0.532 both |
| `--k5-struct` (W1's model `S210.R.chosen.model`), dev GEN file | **equal** (`cmp`, 200,397 bytes); stdout equal; SCRIBE v2 0.411 both |
| The same GEN files from the timer builds (planner 3 times, `--k5-struct` twice), and plain K5 (`--no-k5-plan`) twice | equal |
| `k5_structure features` on H with S210 (the meta-mixer's training input) | **equal** (`cmp`) |
| `S10.stats` loaded and saved again | **equal** (`cmp`): the regenerated context table is byte-identical |
| Statistics built from the same 5 MB of S | equal in every key, count and section. The files differ only in the 4 padding bytes of each 16-byte (key, count) pair. The old code writes those from uninitialised memory too: two runs of the old code differ in the same bytes. |
| Selftests | `k5_structure` ok (old and new), `k5_planner` ok, `cyphalm_trie` ok, `scribe` 119/119 |

There is no floating-point reordering anywhere. Every sum is taken over the same terms in the same order, and integer-valued sums are exact.

## Gate time, wall time, RAM, code

**Gate time.** The same timer was added to the old and new code: the time inside `Experts::prepare` and `Experts::features`, summed over threads. Old and new ran `sample` side by side on 4 threads each, with the same flags as `gen_eval.sh`. The machine was shared (load average 17 to 28).

| Writer | Run | Old: prepare + features = gate (s) | New: prepare + features = gate (s) | Gate time |
|---|---|---:|---:|---:|
| Planner | 1 | 0.125 + 1.163 = 1.287 | 0.115 + 0.654 = 0.769 | **−40.2%** |
| Planner | 2 | 0.086 + 0.682 = 0.768 | 0.087 + 0.341 = 0.427 | **−44.4%** |
| Planner | 3 (new started first) | 0.103 + 0.903 = 1.006 | 0.107 + 0.459 = 0.565 | **−43.8%** |
| `--k5-struct` | 1 | 1.109 + 8.109 = 9.218 | 1.110 + 3.544 = 4.654 | **−49.5%** |
| `--k5-struct` | 2 | 1.252 + 10.266 = 11.517 | 1.265 + 4.600 = 5.864 | **−49.1%** |

- The planner calls `prepare` 258 times and `features` 1,925 times on dev (93 of 250 prompts take a plan path). `--k5-struct` calls them 2,300 and 18,400 times.
- `features` falls 44–56%. `prepare` does not move, because LDA inference is most of it.
- The class n-gram also scores the planner's skeletons (step 2, outside the gate). On the same 415,793 class queries a single lookup went from 3.30 to 1.01 µs (S10) and from 7.08 to 2.02 µs (S210), with the same sum to the last digit.

**Wall time and peak RAM** of the sample step (the exactness runs, `gen_eval.sh`, old and new started together on 4 threads each):

| Writer | Sample wall, old → new | Sample peak RSS, old → new |
|---|---:|---:|
| Planner | 2:18.7 → 2:21.9 | 769 MB → 742 MB (−27 MB, −3.5%) |
| `--k5-struct` | 5:11.5 → 5:10.6 | 1,849 MB → 1,559 MB (−289 MB, −15.6%) |

Sampling time alone (`sample`'s own "generated … in" line, the K5 build and loading left out), every old/new pairing, 4 threads each:

| Pairing | Old (s) | New (s) | Change |
|---|---:|---:|---:|
| Planner, timer builds, run 1 | 189.9 | 200.7 | +5.7% |
| Planner, timer builds, run 2 | 143.8 | 147.6 | +2.6% |
| Planner, timer builds, run 3 (new started first) | 167.8 | 171.3 | +2.1% |
| Planner, plain builds (new started first) | 166.5 | 167.8 | +0.8% |
| `--k5-struct`, timer builds, run 1 | 188.4 | 190.9 | +1.3% |
| `--k5-struct`, timer builds, run 2 | 219.7 | 221.5 | +0.8% |
| Control: plain K5 (`--no-k5-plan`, none of `experts.hpp` runs) | 212.7 / 191.4 | 213.1 / 190.7 | +0.2% / −0.4% |

- The gate is a small part of either writer. Under the planner it is 0.2% of the sampling threads' time; under `--k5-struct` it is 1.2%. K5's own drafting and filling is the rest. So halving the gate cannot move the wall clock.
- In every planner pairing the new binary was a little slower (0.8–5.7%), even though its gate and its class lookups are faster. The plain-K5 control shows no difference, so it is not the binary as a whole. PHASE_TBD

**Code.** `experts.hpp` goes from 921 to 951 lines, or from 814 to 833 lines that are not blank or comments. The context table's build code and the duplicated loops went. The class index, the bucket table and the shared slot walk came in.

**Per expert after the change** (the harness, base and new started together on one thread each, 2,000 boundaries, µs per call):

| Part | S10 old | S10 new | S210 old | S210 new |
|---|---:|---:|---:|---:|
| prepare | 486 | 479 | 463 | 448 |
| · LDA inference | 388 | 390 | 376 | 364 |
| features | 468 | 211 (−55%) | 591 | 240 (−59%) |
| · class n-gram | 298 | 113 (−62%) | 433 | 144 (−67%) |
| · novelty: training runs | 79 | 26 (−67%) | 61 | 23 (−62%) |
| · topic | 40 | 33 | 35 | 25 |
| · relations | 21 | 17 | 33 | 27 |
| · entity grid | 13 | 10 | 13 | 10 |
| · reference and time | 3.3 | 2.5 | 3.1 | 2.5 |

This profile ran late in the session, with the machine's load average near 50, so its numbers are two to three times the first profile's. The old and new columns ran at the same time, so they compare fairly with each other. The entity grid's code did not change; its small drop is the machine, not the merge.

## §13.40 M4 predictions, scored

- **Exact: hit.** Every feature is bit-identical on H, and both dev GEN files are equal under `cmp`.
- **Gate time down at least 20%: hit.** It fell 40–44% under the planner (three runs) and 49–50% under `--k5-struct` (two runs).

By the keep rule of §13.40, the merge is exact and lowers time and RAM. It also adds 30 lines (2.3% more code lines). That is the one thing worse, and the lead decides whether it counts.

## Files

- `native/include/cypha/lm/structure/experts.hpp` (the only code change).
- Runs: `/mnt/d/cypha_llm/merge/m4/` on Odin's PC: `P_old`, `P_new`, `S_old`, `S_new` (gen_eval outputs, `.sample.time`), and `gate/` (the timer runs).
- Not committed (scratch tools): the harness `gate_bench.cpp`, its lap-timer variant, the timer patch, the class-query probe.
