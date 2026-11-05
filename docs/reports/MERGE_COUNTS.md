# Merge M1: one count index and one Kneser–Ney core

**Odin Loch** · 2026-11-05 · branch `wip/merge-counts` · GENERATION_PLAN §13.40 M1

**Outcome first.** The merge is exact. K5's counts, its 12-word copy cap, its 13-token run check and its word positions now come from one suffix array over the training ids. SCRIBE's MATCH reads the same index type over the pack's bytes. Byte KN and word KN share one Kneser–Ney core. Every output checked is the same byte for byte as main's binary: K5 alone and the planner on all 250 GEN-A dev prompts (GEN files, GEN_EVAL lines, SCRIBE v2 and v3.1 reports, summaries), the selftests, and byte KN's users on the small setup. K5's peak RAM falls by half (297 MB to 146 MB); with the planner it falls from 769 MB to 631 MB. Time does not change. Code grows by about 290 lines outside the tests.

Byte KN can also take its counts from a suffix array over its bytes. That gives the same tables and builds 5–12 times faster, but it needs 2–5 times the RAM while it builds. So it stays behind a switch (`CYPHA_BKN_COUNTS=index`), and byte KN's default is unchanged.

## What was merged

**One count index** (`native/include/cypha/lm/count_index.hpp`, new). `cidx::CountIndex<S>` is a suffix array over bytes or 32-bit ids, built by libsais. It answers:
- the count of any string, of any length (as infini-gram does);
- the symbols after a string, in ascending order, each with its count;
- continuation counts: the distinct symbols before a string (Kneser–Ney's N1+(• c w));
- the longest prefix of a pattern that occurs (matching statistics).

**One Kneser–Ney core** (same header, `kn::`). Two functions:
- `kn::discounts`: Chen and Goodman's D1, D2, D3+ from an order's counts of counts. The two estimators differ only when a count of counts is zero; a flag keeps each one's rule.
- `kn::context`: one context's values and gamma from its entries' counts, with byte KN's singleton pruning as an option. It does the same sums in the same order as both old estimators.

**K5** (`lm/word_kn.hpp`).
- `WordKN` builds its counts from the suffix array. The entries are the same as before: the top order counts occurrences; each order below counts distinct left words; order 1 counts distinct words before each word. The old tables counted only n-grams that end at position 4 or later, and the index keeps that rule.
- Contexts seen at least 64 times are kept as tables (values, gamma, n(c)). The rest are counted from the suffix array when the walk asks for them, with the same arithmetic. That is why the RAM falls: the old tables held 4.0 million entries (163 MB); the kept tables and the index hold 18 MB. `CYPHA_K5_KEEP=N` changes the threshold; any value gives the same text.
- The class model (order 3 over 258 classes) keeps every context and drops its suffix array after the build.
- The copy cap (`NgramSet`) is a view on the index: "do the last 12 ids followed by w occur?" is a binary search. Before, it was a sorted set of 1.35 million 64-bit window hashes, and another set for the 13-token run check.
- The prompt association reads each word's positions from its suffix-array interval. The old position lists are gone.
- The planner's fill (`lm/planner/writer.hpp`) and `lm/k5_mcmc.hpp` hash windows themselves (`prefix(ctx, h)`, `has(h)`). M1 leaves those files alone (the fill belongs to M3). So `NgramSet` still answers the hash calls, and builds the hash set only on their first use. K5 alone never builds it.

**Byte KN** (`lm/byte_kn.hpp`). The discounts and each context's values come from the shared core. The counts still come from BKN2's grouped parts by default. `build_index` counts them from a byte suffix array instead; `CYPHA_BKN_COUNTS=index` selects it.

**SCRIBE's MATCH** (`eval/scribe.hpp`, `PackIndex`). It is now a `CountIndex<std::uint8_t>` over the pack. The old one was a sorted array without the suffixes that start at a document's end ('\n'). The new one answers a pattern that starts with '\n' the way the old one did: one byte of match, no count past it. Matches are still capped at 64 bytes.

**Build.** `scribe` and `k5_planner` now compile `third_party/libsais/libsais.c` (CMake). The header declares the two libsais calls it uses, so targets that include it without building an index need nothing new.

## Commands

```
rsync -a --exclude .git <worktree>/ ~/m1_src/
cmake -S ~/m1_src/native -B ~/build_m1s -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ~/build_m1s --target cyphalm_trie scribe k5_planner
# each pair: old (~/build_lead, cwd ~/lead_src) and new (~/build_m1s, cwd ~/m1_src) started together, both through
# ~/m1_src/scripts/gen_eval.sh, every binary under /usr/bin/time -v
export WIKI=/mnt/d/cypha_llm/enwik8 ENWIK8=/mnt/d/cypha_llm/enwik8 SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32 K5_THREADS=4
gen_eval.sh dev k5 $OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt --no-k5-plan
gen_eval.sh dev plan $OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt
cyphalm_trie context8k $ENWIK8 $DOLLY --train 1048576 --sample 1179648            # and with CYPHA_BKN_COUNTS=index
cyphalm_trie sample $ENWIK8 OUT.gen.txt --prompts 40 --train 1048576 --sample 1179648   # the byte sampler
```

## Exactness checks (old = main 7265b3e's binaries in `~/build_lead`)

Every check is `cmp` of whole files, unless it says otherwise.

| Check | Result |
|---|---|
| `sample --k5 train.txt --no-k5-plan`, 250 dev prompts, via `gen_eval.sh`: GEN file, stdout (k5_cont_bpb, copy32, verbatim13) | same |
| … its SCRIBE v2 report, v3.1 report (MATCH) and summary, new `scribe` against old `scribe` | same |
| The default (planner), 250 dev prompts: GEN file and stdout | same (twice: one pair with gen_eval, one without) |
| … its SCRIBE v2 report, v3.1 report and summary | same |
| Old-header vs new-header harness: 20,000 contexts' distributions (`kn().dist`, `model_dist` with the class floor, with dynamic counts), 160,000 `word_prob` values, 200,000 copy-cap answers, 3,000 `has_run13` answers | same; the index cap also equals the hash cap at every one |
| K5's build counts: entries per order (473,074 / 998,428 / 1,243,711 / 1,319,212) and windows (1,351,348) | same |
| `cyphalm_trie selftest`, `scribe selftest` (120 checks), `k5_planner selftest`, `cypha_report_card selftest` | pass |
| The branch merged with today's main (M3's fill in `planner/writer.hpp` already there): `git merge-tree` clean; `cyphalm_trie` and `k5_planner` build, their selftests pass | pass |
| Every other target that includes the changed headers builds: `cyphalm_tok_gen`, `thought_bindings_test`, `thought_k5`, `thought_k6`, `thought_k6_test` | pass |
| Byte KN tables, BKN2 parts against `build_index` (`same()` plus the counts of counts and sizes before pruning), 8 cases: random bytes; enwik8 5 kB at K 1 and 2; 1 MB at K 6 with and without pruning, K 8; 10 MB | same in all 8 |
| `context8k` small setup (`--train 1048576 --sample 1179648`): stdout, old against new, and old against new with `CYPHA_BKN_COUNTS=index` | same, same |
| The byte sampler (BKN2), `sample --prompts 40` on the small setup: GEN file and stdout, the same three | same, same |

A trap on the PC: `~/lead_src/scripts/gen_eval.sh` is older than main's. It lacks §13.39's `--plan-drop 75497472:100000000`, so a planner read through it drops 253 facts instead of 2,383, and its text differs. The first final planner pair ran the old side through that copy and differed for that reason only. Both sides then ran main's `gen_eval.sh`, and every file was the same. (The final K5-alone pair also ran the old side through the old copy; the flag reaches only the planners, and those files were the same.) (`~/lead_src`'s fact store also has CRLF in six side files; the outputs show it does not matter.)

New selftests: the index's counts and longest matches against a direct scan; K5's distributions the same with every context kept, the default, and none kept; the index cap against the hash cap; SCRIBE's pack index on random patterns (some starting with '\n', some longer than 64 bytes) against a brute-force scan.

## RAM, time and code

Old and new were started together with the same thread count (4 each, so 8 in all). Other jobs shared the PC.

**K5 and the planner** (`sample`, 250 dev prompts):

| Read | wall old → new | user CPU old → new | peak RAM old → new |
|---|---:|---:|---:|
| K5 alone, final pair (with gen_eval) | 3:03.6 → 3:02.2 | 649.9 → 644.8 s | 297 → 146 MB (−51%) |
| K5 alone, first pair | 3:28.4 → 3:29.0 | 683.5 → 686.0 s | 298 → 146 MB (−51%) |
| The planner, final pair (with gen_eval) | 5:02.4 → 4:59.6 | 1,001.8 → 990.4 s | 769 → 631 MB (−18%) |
| The planner, first pair | 2:22.6 → 2:22.1 | 501.9 → 501.2 s | 770 → 633 MB (−18%) |
| Harness (one thread: K5's build and 20,000 contexts) | 37.3 → 37.9 s | — | 274 → 73 MB (−73%) |

K5's build takes the same time (16.1 s before, 16.6 s after, with the 256 exchange classes; the old run had the PC to itself). K5's own estimate of its model size falls from 163.2 MB to 18.4 MB. With the planner, the planner's statistics dominate the RAM; the planner's fill also builds the hash set of windows (about 22 MB) through the old calls.

**SCRIBE** (the ladders of the two final pairs; only v3.1 builds MATCH's pack index):

| Ladder | wall old → new (K5 pair; planner pair) | peak RAM old → new |
|---|---:|---:|
| v2 | 12.5 → 12.5 s; 15.6 → 17.7 s | 399 → 400 MB; 400 → 400 MB |
| v3.1 | 18.2 → 18.3 s; 23.3 → 24.2 s | 508 → 469 MB; 511 → 470 MB (−8%) |

v2 does not touch the pack index, so its 2-second spread in the planner pair is the shared PC. The new pack index sorts on one core (libsais); the old one sorted 64-byte keys on 4 threads with a buffer per bucket.

**Byte KN's build** (enwik8, K 6, BKN2's default pruning 4:2:1; one process per build):

| Text | BKN2 parts: time, peak RAM | suffix array: time, peak RAM |
|---|---:|---:|
| 1 MB | 0.77 s, 15 MB | 0.16 s, 27 MB |
| 7.3 MB (the default `--train`) | 8.6 s, 40 MB | 1.4 s, 119 MB |
| 40 MB | 108 s, 95 MB | 9.3 s, 459 MB |

The suffix array needs 4 bytes per text byte, plus the text and its neighbours' common prefixes, plus every context of an order before pruning. That is more than BKN2's bounded parts. So byte KN keeps BKN2's counts by default.

Byte KN's users on the small setup (the three binaries started together):

| Run | wall old / new / index | peak RAM old / new / index |
|---|---:|---:|
| `context8k` | 11:34 / 11:27 / 9:37 | 1,336 / 1,336 / 1,335 MB |
| byte sampler, 40 prompts (second read) | 56.2 / 55.4 / 55.9 s | 30.2 / 30.2 / 36.3 MB |

Byte KN is a small part of `context8k`: its build there takes under a second, and the process peaks elsewhere. The wall-time spread is the shared PC (a first read of the byte sampler, while the gen_eval pairs ran, gave 61 / 62 / 111 s; the second read, with the same outputs, is above). In the byte sampler the suffix-array path raises the peak by 6 MB (+20%).

**Lines of code** (total lines; code lines leave out blank and comment-only lines):

| File or part | lines before | lines after | code before | code after |
|---|---:|---:|---:|---:|
| `lm/word_kn.hpp` | 2,137 | 2,307 | 1,899 | 2,037 |
| … `WordKN` | 172 | 236 | 161 | 216 |
| … `NgramSet` (the copy cap) | 49 | 75 | 45 | 68 |
| … `self_test` | 298 | 352 | 287 | 340 |
| `lm/byte_kn.hpp` | 478 | 565 | 415 | 491 |
| `lm/count_index.hpp` (new) | 0 | 288 | 0 | 218 |
| `eval/scribe.hpp` | 4,229 | 4,179 | 3,660 | 3,605 |
| … `PackIndex` | 131 | 42 | 127 | 37 |
| all four files | 6,844 | 7,339 | 5,974 | 6,351 |

Code grows by 377 lines, about 290 outside the tests. The index (218) and byte KN's switchable suffix-array path (about 80) are the new code. SCRIBE's pack index loses 90. Two follow-ups would remove more: the planner's fill could call the exact cap (`prefix(ctx, range)`, `completes`), and then the hash calls in `NgramSet` could go; and byte KN's suffix-array path could go if it is not wanted.

## The §13.40 M1 predictions

| Prediction | Measured | Score |
|---|---|---|
| Exact | every output above is the same under `cmp` | **hit** |
| K5's peak RAM down at least 30% (the count tables go) | K5 alone −51% (297 → 146 MB); with the planner −18% (769 → 631 MB), where the planner's own statistics dominate | **hit** |
| Sampling time within +50% (suffix-array lookups are slower than hash lookups) | K5 alone −0.8% and +0.3% wall; the planner −0.9% and −0.4% | **hit** |

The old tables were not hash tables: they were sorted arrays searched by binary search. Most contexts at orders 4 and 5 are seen once or twice, so counting them from the suffix array costs about what the old search did. The frequent contexts are kept as tables.

## Recommendation

Keep the merge as the one path for K5's counts, its copy cap and SCRIBE's MATCH: it is exact, it halves K5's RAM, and it costs no time. It is already the default on this branch; there is no old path left for K5 or SCRIBE. Keep byte KN's default as it was (BKN2 counts, now on the shared Kneser–Ney core); its suffix-array counts are exact and much faster to build but not lighter.

## Files and data

- Code: `native/include/cypha/lm/count_index.hpp` (new), `lm/word_kn.hpp`, `lm/byte_kn.hpp`, `eval/scribe.hpp`, `native/CMakeLists.txt`. `native/tools/lm/k5_structure.cpp`, built by hand, now needs `third_party/libsais/libsais.c` too.
- Runs on Odin's PC: `/mnt/d/cypha_llm/merge/m1/` (`ref/` the first old reads, `dev1/` the first new read, `time1/` the first pairs, `final/` the final K5 pair, `final2/` the final planner pair, `bkn/` and `bkn2/` byte KN's users).
- Not touched: `lm/structure/*`, `lm/planner/*`, `online/*`, `native/tools/trie_stream_train.cpp`, `scripts/run_train.sh`. No `malloc_trim`.
