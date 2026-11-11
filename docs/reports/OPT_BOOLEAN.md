# Boolean minimisation in the hot loops (O3)

2026-11-06 · Odin Loch

**Odin:** "Keep looking. Come up with more. Try optimisation techniques, do it in boolean and run expresso on it, variable elimination on the gpu stuff like that. Go crazy on it."

Branch `wip/opt-boolean`, merged with main f26ea25 (M1 and M2 exact; the planner on by default with the text-only store `e9_text`; Q3's discourse flag). Plan: [GENERATION_PLAN §13.41](../GENERATION_PLAN.md), job O3. Odin's PC (WSL, Xeon Gold 6242, GCC 13.3, Release). Other jobs (O5, M4, T1) ran on the PC at the same time. No GPU. Nothing was downloaded.

## In short

- O3 found **15 Boolean decisions** in the hot loops of `context8k` and K5 sampling. It wrote each one's truth table and minimised it with our own minimiser, `scripts/bool_min.py` (Quine–McCluskey with Petrick's method, and an Espresso-style loop for wide functions).
- **The minimised forms are not what the code now uses.** On a CPU a 256-byte table load, a bit test, or a count of compares is cheaper than any sum of products. Minimisation told us how big each function is; it did not give faster code.
- What did move the time was removing **unpredictable branches**: the trie's count quantiser, the similar-context expert's top-k and its list lookup. That is if-conversion, not minimisation.
- **Exact:** every check is `cmp`-identical, old against new. (Details below.)
- **Time:** TBD.

## Where the branch misses were

The old binary (main, built with `-g`), `context8k` on the small setup, sampled with our own `perf_event_open` sampler (`branch-misses` every 20,011, `branches` every 1,000,003, cycles every 4,000,037). Inside the binary: 9.70 × 10⁹ misses of 2.39 × 10¹¹ branches, 4.1%.

| Where | Share of misses | Share of branches | Miss rate |
|---|---:|---:|---:|
| the trie's count quantiser (`trie::quantize`, `used_count`), in `fill_experts` | 33.2% | 11.1% | 12.1% |
| the similar-context search (`SimilarContext::search`: `partial_sort`'s heap, `upper_bound`) | 24.0% | 6.6% | 14.7% |
| byte KN's build (`std::sort` of pairs) | 13.2% | 7.3% | 7.4% |
| the copy expert's window scan (`copy_pred`) | 3.9% | 13.8% | 1.1% |
| the trie's `gather` | 3.5% | 1.4% | 10.5% |
| the online n-gram | 1.6% | 2.3% | 2.8% |

Cycles: 44% in the trie's `fill_experts`, mostly waiting on memory (the follower lists are linked lists). A miss costs about 15–20 cycles, so all misses together cost about 5–7% of the cycles. That is the ceiling for anything done to branches here.

K5 sampling (the planner's default, dev prompts, same sampler): 70% of the branch misses and 28% of the cycles are in one `std::partial_sort` heap, the nucleus in `K5::step_word` (`word_kn.hpp`). That is O6's job (the lazy exact nucleus), so O3 left it alone. The repeat ban (`RepeatBan::banned`) takes 18% of the cycles and 8% of the misses.

## The functions found and their minimised sizes

`python3 -I scripts/bool_min.py cypha --md` (the full table, with owners and how often each runs, is its output). Each output bit is minimised alone, with the rows that cannot occur as don't-cares (a length past 8,192, a count of 1 that is also past 0.8, and so on). Sizes are products / literals of a two-level sum of products. "Canonical" is one product per ON row. Up to 16 inputs the minimum is exact (Quine–McCluskey, then Petrick's method or branch and bound on the cyclic core); the wide ones are Espresso's loop, checked against the function. `bool_min.py selftest` checks both against brute force on random functions of 2–5 inputs (Espresso was above the minimum on 7 of 300).

| Function | Where | Inputs | Don't-care rows | Canonical SOP | Minimal SOP, per output bit | Minimal SOP | Code now |
|---|---|---:|---:|---:|---|---:|---|
| `byte_class` | `online/mix_core.hpp:46` | 8 | 0 | 259/2072 | 8/40 7/43 13/57 | 28/140 | 256-byte table (done) |
| `copy_bucket` | `lm/copy_expert.hpp:79` | 7 | 65 | 150/1050 | 4/8 4/5 4/4 | 12/17 | 66-entry table with a clamp (done) |
| `len_bucket` | `tools/lm/cyphalm_trie.cpp:878` | 14 | 8191 | 16334/228676 | 3/24 9/13 9/9 | 21/46 | n < 4 ? 0 : min(6, bit_width(n) - 2) (patch) |
| `count_bucket` | `tools/lm/cyphalm_trie.cpp:879` | 5 | 0 | 46/230 | 2/4 3/3 | 5/7 | (t > 1) + (t > 3) + (t > 15) (patch) |
| `sim_bucket` | `lm/similar_context.hpp:702` | 11 | 1888 | 408/4488 | 1/2 12/50 8/27 22/116 13/59 7/31 4/12 | 67/297 | sums of compare results (done) |
| `cls_bucket` | `lm/class_context.hpp:137` | 7 | 0 | 312/2184 | 2/4 3/3 1/1 1/1 | 7/9 | order * 4 + (t > 1) + (t > 3) + (t > 15) (done) |
| `copy_state` | `online/full_mixer.hpp:394` | 3 | 0 | 5/15 | 2/4 1/2 | 3/6 | unchanged: already minimal, and the branch skips a log |
| `bit_position` | `online/online_ngram.hpp:162` | 8 | 1 | 614/4912 | 4/10 4/8 4/4 | 12/22 | bit_width(c0) - 1 (patch) |
| `match_log2` | `online/meta_mixer.hpp:182` | 14 | 8191 | 18284/255976 | 44/240 24/119 12/71 7/14 | 87/444 | min(15, bit_width(ig_n + 1) - 1) (done) |
| `byte_equal` | `lm/copy_expert.hpp:32` | 16 | 0 | 256/4096 | 256/4096 | 256/4096 | copy: SSE2 compares into a 64-bit mask, positions that cannot win skipped (done); the n-gram's slot claim: masks (patch) |
| `sentence_end` | `lm/word_kn.hpp:1433` | 9 | 0 | 4/36 | 4/36 | 4/36 | size 1 and a 256-bit set (patch) |
| `markup` | `lm/word_kn.hpp:1436` | 8 | 0 | 11/88 | 7/51 | 7/51 | 256-bit set (patches) |
| `fact_preference` | `lm/planner/writer.hpp:514` | 5 | 12 | 6/30 | 2/5 | 2/5 | unchanged: already minimal |
| `candidate_order` | `lm/similar_context.hpp:982` | 3 | 2 | 3/9 | 2/3 | 2/3 | one 64-bit key per pair; radix sort when k ≥ n/4, else nth_element or partial_sort on keys (done) |
| `used_count_choice` | `lm/packed_trie.hpp:156` | 32 | 0 | – | 31/37 6/167 | 37/204 | both values made, one kept by a mask; the ladder by counted compares (done) |

What the table says:
- Most functions were already small as written. `copy_state`, `fact_preference` and `candidate_order` were already minimal. `count_bucket` and `cls_bucket` minimise to 7–9 literals, about the size of the chain of three compares they came from.
- Equality of two bytes (`byte_equal`) and the four sentence stops do not shrink at all in two levels.
- The functions that do shrink a lot (`len_bucket` 16,334 → 21 products, `match_log2` 18,284 → 87) are threshold chains on an integer, and their best form is not a sum of products but `bit_width`, one instruction.
- `byte_class` shrinks from 259 to 28 products, but a table of 256 bytes is one load and the products are 140 literals.

So no minimised sum of products went into the code. Each function took the form that is cheapest on this CPU: a table, a bit set, a count of compares, or `bit_width`.

## What was changed

**In place** (files no other job owns now):

| File | Change |
|---|---|
| `lm/packed_trie.hpp` | `quantize`: the binary search over the 1.08 ladder becomes a count of 12 compares inside one octave (a 33-entry table by bit width gives the steps below it); the small-count case is picked by a mask. `used_count`: a mask, no branch. `gather`: every follower is written at `n`, and `n` moves on by `c != 0`. The old search stays as `quantize_search`, the reference. |
| `lm/similar_context.hpp` | `similar_top_k`: each (dot, slot) pair becomes one 64-bit key, so the two-part compare is one integer compare; then a radix sort when k is at least a quarter of the list (the default: k 8192), else `nth_element` and `sort`, or `partial_sort` for k under 256. `upper_index`: a binary search with no branch on the data (Khuong and Morin). `bucket`: each chain of compares becomes a sum of compare results. |
| `lm/class_context.hpp` | `bucket`: a sum of three compares. |
| `lm/copy_expert.hpp` | `copy_pred`: a window position can beat the best match only if its byte before and its byte n before both match, so SSE2 compares test 64 positions at once into a bit mask, and only the set bits are measured, highest first, as before. `copy_bucket`: a 66-entry table. |
| `lm/gate_class.hpp` (new) | the mixers' byte class as a 256-byte table built at compile time from the old chain; the gates' markup and stop tests as 256-bit sets. |
| `online/mix_core.hpp` | `byte_class` reads the table; `path_bits` adds the half by a mask. |
| `online/full_mixer.hpp` | the final gate's string compare (`final_gate == "copy"`, once a byte) is made once, at construction. |
| `online/meta_mixer.hpp` | the match-length log is `bit_width`, not a loop of shifts. |
| `tools/lm/opt_boolean_smoke.cpp` | checks every changed function against its old form (ctest `native_opt_boolean_smoke`). |
| `scripts/bool_min.py`, `scripts/opt_boolean_tables.py` | the minimiser and the truth tables. |

M1 and M2 have landed, so the mixer changes that were patches before are now made in place (M2 moved `byte_class` and `path_bits` into `mix_core.hpp`).

**As patches** (files other jobs own now; `docs/reports/opt_boolean_patches/`, each against main f26ea25, for the owner to apply):

| Patch | Owner | Change |
|---|---|---|
| `o2_cyphalm_trie.patch` | O2 (`tools/lm/cyphalm_trie.cpp`) | InfiniGram's length bucket from `bit_width`; the count bucket as a sum of compares |
| `o5_online_ngram.patch` | O5 (`online/online_ngram.hpp`) | the bit position from `bit_width`; `bit ? p : 1 - p` picked from a pair; a slot's claim by masks |
| `o6_word_kn.patch` | O6 (`lm/word_kn.hpp`) | K5's gate: the sentence stop and the markup count by byte sets |
| `q2_planner_writer.patch` | Q2 (`lm/planner/writer.hpp`) | the planner gate's markup count by a byte set |

Left as they were: the full mix's copy state (already minimal, and its branch skips a log) and the planner's fact preference (already minimal, and it runs once per prompt).

## Resumed (2026-11-12, before any run) · Odin Loch

- `origin/main` (9cbd277: O5, Q1/Q2 flags, the planner's fact files) is merged in. No conflict.
- **The patches.** O5 has landed on main, so `online/online_ngram.hpp` has no running owner. Its patch no longer applied (O5 rewrote the file), so it was redone by hand: the slot claim by masks, and `bit ? p : 1 - p` picked from a pair, in `code()`. O5's bit-position loop is gone from main, so that part has nothing to change. It is now in place on this branch.
- **Left as patches:** `o2_cyphalm_trie.patch` (`tools/lm/cyphalm_trie.cpp`: O2 runs now and rewrites that file), `o6_word_kn.patch` (`lm/word_kn.hpp`: O6, and §13.48 runs now) and `q2_planner_writer.patch` (`lm/planner/writer.hpp`: §13.48 runs now; it no longer applies either).
- **Checks** (old: main 9cbd277 from `git archive`; new: this branch; GCC 13, Release, Ninja). Each pair started together, each run on its own 4 CPUs by `taskset`:
  - small `context8k`: the default, `--no-meta-mix`, `--no-full-mix`;
  - dev GEN files (`cyphalm_trie sample`, as `scripts/gen_eval.sh dev` runs it, 250 prompts, `--k5 fuzzy/train.txt`): the planner (the default) and `--no-k5-plan`, GEN file and stdout;
  - `ctest` `native_opt_boolean_smoke`;
  - one full default `context8k` in its own turn of `flock ~/.cypha_full.lock`: `context8k_bpc` 1.4962 and stdout `cmp` equal to the old full run's (run with O2's pair).
- Exact means `cmp` of the whole stdout. If any check differs, O3 is reported failed.

## Exactness

TBD-EXACT

## Time and branch misses, old against new

Each pair was started together, each run pinned to its own 4 CPUs (two cores with both hyperthreads: 48–51 and 52–55). Counters from our own `perf_event_open` counter (`pstat`), user mode, all threads. The PC was loaded by other jobs (load 16–30 on 64 CPUs), so each comparison is within a pair, not across pairs.

**`context8k`, small setup, default** (old: main f26ea25; new: this branch):

| | old | new | change |
|---|---:|---:|---:|
| wall | 580.5 s | 538.9 s | −7.2% |
| user CPU | 713.1 s | 655.1 s | −8.1% |
| cycles | 2.122 × 10¹² | 1.958 × 10¹² | −7.8% |
| instructions | 2.061 × 10¹² | 1.966 × 10¹² | −4.6% |
| branches | 2.793 × 10¹¹ | 2.123 × 10¹¹ | −24.0% |
| branch misses | 1.080 × 10¹⁰ | 0.449 × 10¹⁰ | **−58.5%** |
| miss rate | 3.87% | 2.11% | |
| peak RAM | 1,337,944 KB | 1,338,492 KB | +0.04% |

Where it went (the full mix's own thread timers, same pair): the static rows, where the trie's quantiser runs, 30.5 → 10.3 busy seconds (−66%); the similar-context thread 69.7 → 48.7 s (−30%). The full-mix stage itself only fell 123.9 → 117.6 s (−5%), because it waits for its slowest threads, the LSTM (115 → 107 s) and the mixer (117 → 112 s), which O3 did not touch.

**The same, old (again) against all** (all: this branch plus the four patches). This pair ran while other projects' benchmarks pushed the PC's load to 75–80, so both took about 17 minutes:

| | old | all | change |
|---|---:|---:|---:|
| wall | 1,037.0 s | 978.0 s | −5.7% |
| user CPU | 1,207.7 s | 1,113.8 s | −7.8% |
| cycles | 3.334 × 10¹² | 3.046 × 10¹² | −8.7% |
| instructions | 2.067 × 10¹² | 1.982 × 10¹² | −4.1% |
| branch misses | 1.081 × 10¹⁰ | 0.441 × 10¹⁰ | −59.2% |

The patches add little: all made 4.41 × 10⁹ misses, new 4.49 × 10⁹ (miss counts move little with load). The online n-gram's claim and bit pick, the InfiniGram buckets and the K5 gate are not where the misses were.

TBD-TIME3

**The first form failed on time.** The first top-k (a sorted buffer, filled by counting) was branch-free but did work in proportion to k times the list. The default expert keeps k = 8192 of about 10,000 candidates, so it doubled the instructions (2.06 → 4.23 × 10¹²) and made `context8k` 29% slower (492 → 637 s), while cutting branch misses 55%. The output was still identical. It was replaced by the radix sort before any other check. A scratch benchmark of the methods on synthetic lists (one CPU; every method's output checked against `partial_sort`), n 12,000 and k 8,192: `partial_sort` on pairs 1,179 µs a query, on packed keys 893 µs, `nth_element` and `sort` 444 µs, radix sort 230 µs. For k = 32 the radix sort is the slowest (233 µs against 32 µs), so a small k keeps `partial_sort` on the keys, and k from 256 takes `nth_element` and `sort`.

## The O3 prediction, scored

TBD-SCORE

## Where Boolean minimisation would buy something

**Here, almost nothing.** The decisions in Cypha's hot loops are small functions of one byte, one count or a few compares. On a CPU:
- a function of up to about 12 input bits is one load from a table, whatever its sum of products (the byte class: 28 products and 140 literals, or one load);
- a threshold chain is a sum of compares (the buckets);
- the cost that matters is a branch the predictor cannot guess, and that is removed by computing both sides and choosing by a mask, or by changing the algorithm (the quantiser, the top-k), not by a smaller formula.

The time is in arithmetic and memory: the trie's linked follower lists, the LSTM, the mixers' floating point. That is why §13.41 predicted under 3%.

**Where it would buy something:**
- **Bit-sliced evaluation.** When one Boolean function is applied to many items at once (64 per word, 512 per AVX-512 register), each product is one AND across all of them, so a minimal cover is a direct cut in instructions. The copy expert's window scan is the one place in Cypha shaped like this, and there the function is just two byte compares.
- **Hardware.** On an FPGA or ASIC the gate count is the cost. A CyphaLM context selector or gate on an FPGA would be worth minimising (and multi-level synthesis, as in ABC, would do better than two-level Espresso).
- **Large learned rule sets.** If a gate or filter were learned as a decision tree or rule list over many binary features (a gibberish filter, a fact filter), Espresso with don't-cares from unseen feature combinations could shrink it. Nothing like that runs in a hot loop today.

**Where the remaining branch misses are** (leads for other jobs, not Boolean minimisation):
- K5 sampling: the nucleus's `partial_sort` heap (70% of K5's misses) is O6's lazy exact nucleus. The same packed-key trick applies if O6 keeps a sort.
- `context8k`: byte KN's build sorts its pairs with `std::sort` (13% of the old misses); a radix sort on packed keys would remove most of them, if the order is total.
- K5's repeat ban (18% of K5's cycles) is hash lookups, not logic.
