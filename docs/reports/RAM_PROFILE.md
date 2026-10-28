# CyphaLM's RAM, and cutting it

**Odin Loch** · 2026-10-28 · PLAN §10 (budgets) and Odin's goal of low RAM

(Draft. The profile below is measured; the decision rule and predictions were written before any change was tried. Results are added under them.)

**Question.** Where does CyphaLM's memory go at the 8 MiB protocol, and how much of it can be cut without losing BPC, text quality or answers?

## The profile (measured 2026-10-28, before any change)

The qa model: trie trained on wiki train, then two Dolly passes; InfiniGram and the class-context expert (shape classes, tail 6) over the same training text, 18.6 MB (wiki train 7.0 MB + chat 11.6 MB). `VmRSS` / `VmHWM` read after each stage (scratch harness `memprof.cpp`).

| Stage | Resident | Peak so far |
|---|---:|---:|
| start | 3.5 MB | 3.5 MB |
| wiki 8 MiB read | 11.8 MB | 11.8 MB |
| trie trained on wiki | 47.7 MB | 47.7 MB |
| trie trained on chat (2 passes) | 117.5 MB | 117.5 MB |
| match table compacted | 105.6 MB | 117.0 MB |
| training text copied (for InfiniGram and the class expert) | 124.2 MB | 124.2 MB |
| InfiniGram built in memory | 201.2 MB (+77.0) | 216.9 MB |
| class expert built, 2 orders | 172.3 MB (+48.1 over 124.2) | **470.0 MB** |

| Component | Size |
|---|---|
| Trie nodes, 581,141 × 28 B, and followers, 1,756,608 × 12 B | 35.6 MB in use (the vectors reserve 104 MB of address space, not resident) |
| The same trie in the packed layout (Synth accounting, not yet a structure) | 10.7 MB |
| Match table after compaction | 2.7 MB |
| InfiniGram, built in memory (text copy + bit-packed suffix array) | 77.0 MB |
| Class expert table (1,956,595 contexts, 3,016,632 followers) | 44.2 MB |
| Class expert build: every (context, byte) event sorted at once, 8 B × orders × training bytes | about 300 MB transient (2 orders); more with 3 |

**What dominates.** The peak (470 MB) is the class expert's build. Resident after everything is about 200 MB: InfiniGram 77 MB, class table 44 MB, trie 36 MB, the texts the rest.

## Decision rule (fixed before the changes)

A memory change is adopted only if:
1. **Lossless changes** (a new build or a new layout meant to give the same numbers) reproduce every BPC to the printed digit.
2. **Lossy changes** (quantised counts, shorter keys, pruning) change fresh-text BPC by at most +0.002 over trie + InfiniGram + class, surface S does not fall (interval of the change not entirely below 0), and the dev Q/A tally does not change.
3. The saving is measured, resident and peak, on the same harness.

## Predictions (registered before the changes)

| # | Change | Prediction |
|---|---|---|
| M1 | Class expert built in chunks (sort and merge per chunk) | Identical table; build peak falls from 470 MB to under 250 MB |
| M2 | Compact class table: 32-bit keys, 1-byte quantised counts (the trie's quantiser), totals summed at lookup | Table 44 → under 20 MB; fresh BPC change under +0.001 |
| M3 | Contexts seen once dropped from the class table | A further 30–50% off the table; fresh BPC change +0.001 to +0.005 |
| M4 | A packed read-only trie for serving, built from the trained model, predictions equal to Synth mode | Trie 36 → about 11 MB resident, BPC identical to the digit |
| M5 | InfiniGram as a bidirectional FM-index (BWT of the text and of the reversed text, each in a wavelet matrix), no suffix array and no text copy. Registered after M1–M3 were read | Every query result identical (n, total, all 256 counts) on the holdout; index 77 → under 45 MB; query time at most 3× the suffix array's |
| M6 | In `cyphalm_trie qa`, build the class expert and the FM-index before the trie is trained (neither reads the trie), and let the index take the training text. Registered after M1–M5 were read | Every qa number identical; peak 303 → under 200 MB with 3 orders |
| M7 | A stored packed trie file (`PackedModel::save` / `load`): magic, version, sizes and a checksum, with the six mixture weights and the scoring split, so a served model is self-contained. Registered before it was written | Loaded model bit-identical (every array, every holdout distribution, BPC to the bit); file within 1 KB of the packed bytes; load under 0.2 s for the qa model (8.1 MB); resident after load within 2 MB of the packed bytes + the 8 MiB text + the empty process |
| M8 | A stored FM-index file (`FmInfiniGram::save` / `load`), so serving never builds the index. Registered before it was written | Load gives the same answers; file within 1% of the index's bytes; load under 0.5 s for 19.6 MB of text; peak while loading under the index's size + 10 MB |
| M9 | M4, M5 and M6 in `cyphalm_trie`: the packed trie; FmInfiniGram with cursors in place of InfiniGram everywhere; in `qa` the class table is built from T, then the index consumes T, and the chat bytes leave the text buffer; class spec flags. Registered before the tool was changed | Every printed number identical to the build before (times excepted); `cls shape:6:3:cp2` gives the M2/M3 row's fresh trie + IG + class 2.085255; peak: `qa` A0 263 → under 160 MB, `context8k` 229 → under 160 MB, `qa --cls shape:6:3:cp2` under 200 MB (M6's bound) |

## Results (2026-10-28)

### M1: the class expert built in chunks (lossless)

Each 2 MB chunk of training text has its (context, byte) events sorted and counted on its own; the counted runs are then merged into the table. Measured on the qa model (wiki train + chat, 18.6 MB), scratch harness `memprof.cpp`.

| Orders | Contexts / followers | Build peak before | Build peak after |
|---|---|---:|---:|
| 2 | 1,956,595 / 3,016,632 (same) | 470.0 MB | **244.2 MB** |
| 3 | 3,464,156 / 5,208,733 (same) | 684.0 MB | **326.2 MB** |

- **Same numbers to the digit.** `cyphalm_trie cls enwik8 shape:6 tags_lex_wiki.tsv:6` (B4's decision run) prints the same file with the new header, every BPC, interval and surface score included. `cyphalm_trie selftest` passes.
- **Prediction M1: hit** (identical table; 2-order peak under 250 MB).
- **Then each order merged on its own** (2026-10-28, after the whole-model table below showed the build was still the peak). The merge held every order's runs twice, and the heap fragmented as the runs grew; chunk size changed the peak unevenly (2^21: 166 MB over the text, 2^19: 203, 2^17: 168). Now each order's runs are merged separately, buffers are sized once, and the chunk is 2^17. Scratch harness `ccpeak.cpp`, 3 orders, compact, seen once dropped, on the qa training text:

| Build | Chunk | Memory over the text at the peak | Build time |
|---|---:|---:|---:|
| All orders merged together | 2^21 | 165.8 MB | |
| All orders merged together | 2^19 | 202.8 MB | |
| Each order on its own | 2^21 | 175.6 MB | 13.8 s |
| Each order on its own | 2^19 | 156.0 MB | 14.4 s |
| Each order on its own | **2^17** | **131.8 MB** | 16.7 s |
| Each order on its own | 2^16 | 143.8 MB | 19.0 s |

  Same table: every prediction bit for bit on every third holdout byte (`cceq.cpp`), contexts and followers equal, for 2 and 3 orders, shape classes and E1 tags, default, compact, dropped, and both: 16 of 16. B4's decision run again prints the same file.

### M2 and M3: a compact table, and contexts seen once dropped (lossy)

Wiki model (the B4 protocol), fresh text read once, trie + InfiniGram + class; paired block bootstrap. Surface S on 200 prompts, paired.

| Table | Contexts | Table size | Fresh BPC | Change | Surface S | Change in S |
|---|---:|---:|---:|---:|---:|---:|
| 2 orders | 1,242,013 | 29.1 MB | 2.087926 | | | |
| 2 orders, compact | 1,241,820 | 13.6 MB | 2.087946 | +0.000019 [−0.000004, +0.000044] | | |
| 2 orders, seen once dropped | 533,829 | 14.2 MB | 2.088891 | +0.000965 [+0.00079, +0.00115] | | |
| 3 orders | 2,160,001 | 50.2 MB | 2.084349 | | 0.9097 | |
| 3 orders, compact | 2,159,451 | 23.5 MB | 2.084431 | +0.000082 [+0.000059, +0.000103] | 0.9073 | −0.0024 [−0.0090, +0.0041] |
| 3 orders, compact, seen once dropped | 885,985 | **10.8 MB** | 2.085255 | +0.000906 [+0.000685, +0.001118] | 0.9067 | −0.0030 [−0.0097, +0.0033] |

- **Compact** keeps the top 32 bits of each context's 56-bit key and one quantised byte a count (the trie's 1.08 ladder); a context's total is summed at lookup. About 1 context in 4,000 to 6,000 merges with another.
- **Dropped** means a context whose followers total 1.
- **Both pass rule 2 on BPC and S:** at most +0.001 BPC, S intervals not below 0. The dev Q/A tally is read when the expert is wired into `cyphalm_trie qa` (the Q/A tool takes the table as a spec; the flags are added then).
- **Prediction M2: hit** (table halved; change +0.00002 / +0.00008, under +0.001).
- **Prediction M3: miss, both ways in our favour.** The table fell 51–54% (predicted 30–50%) and BPC rose 0.0009–0.0010 (predicted 0.001 to 0.005).

### M5: InfiniGram over an FM-index (lossless)

`native/include/cypha/cyphalm/fm_infinigram.hpp` and `native/src/cyphalm/fm_infinigram.cpp`. The index keeps the BWT of the training text and of the reversed text, each in a wavelet tree, and nothing else. A query grows the context's suffix to the left one byte a step; the reversed text's rows are kept in step, and its BWT gives the next-byte counts. A cursor lets a caller that reads one byte at a time extend the last match by one step.

| Read (qa training text, 19.6 MB) | Suffix array (now) | FM-index |
|---|---:|---:|
| Index in memory | 77 MB (text 18.6 + suffix array 58.3) | **26.0 MB** (text not kept) |
| Build time, one core | 1.3–1.9 s | 4.7–5.5 s |
| Holdout query, hint chain (µs) | 4.7 | 7.6 alone, **5.0 with the cursor** |
| Copying training text, 3 × 3,000 bytes (µs a byte) | 1.8 | 301 alone, **1.1 with the cursor** |
| Answers that differ from InfiniGram's | | **0** of 1,048,576 holdout, 299,594 with min_total 2 and 16, 9,000 copy, 200,000 with a misused cursor |

- **First try, a wavelet matrix** (8 bits a symbol): 39.6 MB, holdout 9.0 µs (6.7 with the cursor). The alphabetic tree shaped by weight needs 5.3 bits a symbol, so it is smaller and faster.
- **On plain enwik8** (`fm_infinigram_test --bench enwik8 22000000`, 19.25 MB indexed, the CMake build's default flags): 75.7 → 26.2 MB; 7.3 µs a query → 17.0 alone, **2.9 with the cursor**.
- **CTest `native_fm_infinigram_test`:** 158,380 query checks against InfiniGram on random texts (alphabets of 1, 2, 4, 26 and 256 bytes, byte 0 included, with and without repeats), with and without hints, min_total up to 4, cursors used in order and misused. A planted bug in the `$` row's handling fails 56,390 of them. The cursor's first version checked only the matched bytes; the test failed it (96 checks), and the check now covers the bytes up to the length bound.
- **Build peak** is about 7 bytes a text byte from a copy and 6 when the text is moved in; the suffix array's is about 5.
- **Prediction M5: hit.** Identical answers; 26 MB, under 45; query time 1.6–2.3× alone and 0.4–1.06× with the cursor (bound 3×).
- **`match_prefix` too, without the position:** the longest prefix of a string that occurs, and its count (a prefix grows to the right, one step on the reversed BWT). `cyphalm_trie qa --parse` keeps a second InfiniGram over the lower-cased training text for A1c's familiarity statistics, and they use only the length and the count, so that index can be an FM-index as well (another 77 → 26 MB). The CTest checks it: 178,380 checks in all.
- **Not done yet:** a stored FM file (the index is built at load, 5 s for 19.6 MB), a position for `match_prefix` (sampled suffix-array entries; nothing in the qa path needs it), and the wiring into `cyphalm_trie`.

### The whole qa model, before and after

Scratch harness `memprof3.cpp`: the qa model (trie on wiki train + two Dolly passes, match table compacted), then the class expert and InfiniGram over the training text T (wiki train + chat, 19.6 MB). `malloc_trim` before each reading, so freed memory is not counted. The trie is still in its training form here (M4 below replaces it).

| Configuration | Class table | InfiniGram | Resident at the end | Peak |
|---|---:|---:|---:|---:|
| Now: suffix array, class 2 orders (B4's setting) | 44.2 MB | 77 MB | 187.1 MB | 220.7 MB |
| Now: suffix array, class 3 orders (the B4B candidate) | 77.7 MB | 77 MB | 227.5 MB | 302.9 MB |
| FM-index, class 2 orders compact | 20.7 MB | 26.0 MB | **113.7 MB** | 231.0 MB |
| FM-index, class 3 orders compact, seen once dropped | 18.2 MB | 26.0 MB | **109.7 MB** | 302.9 MB |

- **Resident falls by 39–52%.** The trie in its training form plus the texts (75.9 MB after training) is now the largest part.
- **The peak did not fall.** It is the class build (3 orders) or the index build (2 orders: the FM build needs about 6 bytes a text byte against the suffix array's 5). Compact storage and dropping contexts change the finished table, not the build's merge buffers.

### M4: a packed read-only trie for serving (lossless)

`PackedModel` and `pack(model)` in `native/include/cypha/cyphalm/packed_trie.hpp`. `pack()` reads a trained `Model` after `compact_match()` and keeps only what Synth mode reads:
- **Order 0 and order 1:** the count tables, as trained (264 KB). The KT experts use full counts.
- **Nodes:** only those Synth can reach, that is a node with a follower seen at least twice, or a node on the path to one (`measure()`'s "useful" nodes). Ids are breadth first: 0 is the root, 1 + b the order-1 node of byte b.
- **Children:** consecutive ids, sorted by byte. A node keeps a 1-byte edge and the 4-byte id of its first child (orders 0–3). There are no child or sibling links.
- **Followers:** a node of order 2–4 keeps the 4-byte start of its followers. A follower seen at least twice is 2 bytes: the symbol and its count on the 1.08 ladder.
- **Match:** the compacted match (contexts seen at least twice), bucketed by the key's high 16 bits. An entry is 2 bytes of key, the 4-byte follower offset (for the follower limit) and the predicted byte, so the packed form never reads the training text.

`fill_experts` and `mix` are now one template body, used by both forms: the same arithmetic in the same order. Only the walk, the follower gather and the match lookup differ. The packed form holds Synth mode only (another mode stops the program). Its online match finds nothing, as in a `Model` after `compact_match()`. `Model::release()` frees the nodes, the followers and the match after packing. `score_holdout`, `heldout` and `sample_continuation` take either form. The header grows from 1,293 to 1,612 lines: `PackedModel` 117, `pack()` 95, the shared template and its two callers 116 (mostly moved out of `Model`), `release()` 9, the self-test check 53. No new tool.

**Parity, bit for bit.** Every comparison is a `memcmp` of the doubles: E0–E4, the match flag and byte, and `mix()`'s distribution and u. Two models, trained as the tool trains them: the bpc model (wiki train) and the chat/qa model (wiki train, then two Dolly passes: 15,015 turns, 12,213,449 bytes).

| Model | Positions | Follower limit | Match fires | Differ |
|---|---:|---|---:|---:|
| bpc | 1,048,576 (whole holdout) | train | 1,007,095 | 0 |
| bpc | 1,048,576 | all followers | 1,007,095 | 0 |
| bpc | 137,519 (every 61st training byte) | all followers | 134,564 | 0 |
| bpc | 137,519 | the position itself | 16,475 | 0 |
| chat/qa | 1,048,576 | train | 218,022 | 0 |
| chat/qa | 1,048,576 | all followers | 1,019,572 | 0 |
| chat/qa | 337,739 | all followers | 335,493 | 0 |
| chat/qa | 337,739 | the position itself | 3,590 | 0 |

| Model | Read | Training form | Packed form |
|---|---|---|---|
| bpc | backoff / mixture BPC | 2.386293 / 2.297794 | 2.386293 / 2.297794 |
| bpc | the same, full precision | 2.3862928313539933 / 2.2977939389859512 | identical |
| bpc | synth_bpc, mean u, match rate | 2.386293, 0.486240, 0.960441 | identical |
| chat/qa | backoff / mixture BPC | 2.364556 / 2.344571 | 2.364556 / 2.344571 |
| chat/qa | the same, full precision | 2.3645558122752401 / 2.3445707246454637 | identical |
| chat/qa | synth_bpc, mean u, match rate | 2.364556, 0.470982, 0.207922 | identical |

`self_test()` has a new fixed-input check: a 654-byte text with counts above 64 (quantised) and counts of 1 (dropped). The two forms are compared bitwise at all 685 prefixes of the text plus 30 unseen bytes, with the follower limit alternating between all followers and the first half. The online match is checked too: after `compact_match()` neither form matches, and the experts agree. The sizes are golden: 306 nodes, 54 followers, 18 matches, 313 positions with a match. It adds about 7 ms to `self_test()` (fastest run: 7.3 ms before, 14.4 ms after).

**Sizes.** MB is 2^20 bytes, as in the profile.

| | bpc model | chat/qa model |
|---|---:|---:|
| Training form in use (nodes 28 B, followers 12 B, match 8 B, tables) | 26,423,648 B (25.20 MB) | 40,404,292 B (38.53 MB) |
| of which nodes and followers | 24,760,032 B (23.61 MB) | 37,351,244 B (35.62 MB) |
| Synth accounting (`measure()` + order-1 + match) | 5,385,965 B (5.14 MB) | 11,181,177 B (10.66 MB) |
| **Packed form** (`PackedModel::bytes()`) | **4,111,378 B (3.92 MB)** | **8,106,219 B (7.73 MB)** |
| of which trie (edges, child and follower starts, followers) | 2,361,340 B | 5,140,428 B (4.90 MB) |
| of which match | 1,485,566 B | 2,701,319 B |
| of which order-0 and order-1 tables (and the struct) | 264,472 B | 264,472 B |
| Trie nodes, training form (all orders, with the root) | 404,601 | 581,141 |
| Trie nodes kept, orders 2–4 (plus all 256 of order 1) | 200,479 | 426,567 |
| Followers, training form / kept | 1,119,267 / 570,398 | 1,756,608 / 1,313,874 |
| Matches | 174,774 | 348,453 |

- The packed form is 23.7% (bpc) and 27.5% (chat/qa) below the Synth accounting. Kept nodes and followers equal the accounting's counts exactly.
- The accounting charges every node an 8-byte header and every child 4–5 bytes. The breadth-first layout needs 9 bytes for a node of order 2 or 3 and 5 bytes for a node of order 4.
- The match costs 7 bytes an entry plus 256 KB of buckets, against 8 bytes an entry in the accounting.

**Resident memory** (`VmRSS` after each stage; the packing peak is `VmHWM`, reset just before packing; scratch harness `packprof.cpp`, built like `memprof.cpp`).

| Stage | bpc model | chat/qa model |
|---|---:|---:|
| wiki 8 MiB read | 11.82 MB | 11.83 MB |
| trained | 47.71 MB | 96.21 MB |
| match compacted (the training form serves) | 37.16 MB | 75.20 MB |
| packed, both forms held | 42.22 MB | 76.59 MB |
| training form freed (**the packed form serves**) | **17.28 MB** | **41.04 MB** |
| the same after `malloc_trim(0)` | 16.14 MB | 32.47 MB |
| peak while packing | 42.22 MB | 76.59 MB |
| peak while training | 49.55 MB | 96.21 MB |

- **Saving.** Serving from the packed form lowers resident memory by 19.88 MB (bpc) and 34.16 MB (chat/qa). The C library keeps some freed heap; `malloc_trim(0)` returns it, for 21.02 MB and 42.73 MB in all.
- **Peak.** Packing never sets the peak. It adds 4.74 MB (bpc) and 1.33 MB (chat/qa) to the resident size just before it, because it allocates each array once, at its final size.
- **Packing time.** 0.18–0.28 s (bpc, three runs) and 0.51–0.63 s (chat/qa, two runs).

**The whole qa model with M4.** The harness of "The whole qa model, before and after" (`memprof3.cpp`, `malloc_trim` before each reading), with the trie packed and its training form released right after compaction. There, packing takes the trie step from 75.9 to 44.0 MB.

| Configuration | Resident at the end, before → with M4 | Peak, before → with M4 |
|---|---:|---:|
| FM-index, class 2 orders compact | 113.7 → **78.5 MB** | 231.0 → 206.1 MB |
| FM-index, class 3 orders compact, seen once dropped | 109.7 → **76.1 MB** | 302.9 → 234.5 MB |

The peaks fall too, because the class table is now built after the training form is freed.

**Serving speed.** `fill_experts` per byte over the whole holdout, two alternating runs per form in each harness run. The machine was shared (load 6–8 on 4 cores), so the spread is wide.

| Model | Training form | Packed form | Fastest vs fastest |
|---|---:|---:|---:|
| bpc (6 runs a form) | 10,579–20,267 ns | 2,295–4,304 ns | 4.6× |
| chat/qa (4 runs a form) | 17,205–21,292 ns | 3,059–4,222 ns | 5.6× |

The training form walks linked sibling and follower lists. The packed form reads short sorted arrays.

**The tool** (measured on a patched build; the patch reaches `main` with the next `cyphalm_trie` change, after the answering sweep that is editing that file). `cyphalm_trie` packs right after training and frees the training form in `bpc`, `chat`, `probe`, `context8k`, `sample`, `cls` and `qa`. `bpc --negative` keeps the training form, because its PPMD, chain and inheritance rows need it. In `qa`, the wiki-stage reads (`wiki_lm`) use the training form before the chat passes and the packed form after them. `selftest` also checks that the packed form gives the golden BPC bit for bit. Each subcommand was run with the old build and the new build on the same data; standard output must match byte for byte. Wall time is on the shared machine.

| Subcommand | Standard output | Other files | Wall s, old / new | Peak MB, old / new |
|---|---|---|---:|---:|
| `bpc` | identical (22 lines) | | 301 / 226 | 51.7 / 51.4 |
| `bpc --negative` | identical (30 lines) | | 129 / 106 | 67.1 / 67.5 |
| `chat` | identical (10 lines: 2.344571, mix_stored 11,181,225) | | 1,237 / 1,420 | 103.4 / 103.3 |
| `sample` | (none) | both files identical (sha256 `fcc2b9c4…`, `8592fb7e…`, as in B1) | 177 / 222 | 51.7 / 51.8 |
| `context8k` | identical (221 lines) | | 1,555 / 1,181 | 229.3 / 194.1 |
| `cls shape:6` | identical except `build_s`, the class table's build time (2.5 / 5.0 s) | | 145 / 209 | 264.0 / 242.9 |
| `qa --dev` (A0) | identical (98 lines) | rows identical | 654 / 521 | 263.4 / 232.0 |
| `qa --dev --leak --parse (candidates) --dk --fam-stat ent_title --copy --cls shape` | identical (226 lines) | rows identical | 592 / 546 | 477.1 / 445.4 |

`probe` was not run: its change is the same `sample_continuation` swap as `chat`'s.

- **Peak.** Where training sets the peak (`bpc`, `chat`, `sample`), it does not move. Where InfiniGram or the class table is built after training, it is now built after the training form is freed: the peak falls by 35.2 MB (`context8k`), 31.4 MB (`qa`), 31.7 MB (`qa` with the class expert) and 21.1 MB (`cls`).
- **Prediction M4: hit.** Every BPC and every distribution is identical to the bit. The trie fell from 35.6 to 4.9 MB, the whole model from 38.5 to 7.7 MB, and resident memory by 34.2 MB (42.7 MB with `malloc_trim`). The size beat the prediction (about 11 MB), which came from the Synth accounting.
- **Not done yet:** a stored packed file. With a load path, a served model would never hold the training form at all.

### M6: the build order

Scratch harness `memprof6.cpp`: the texts are read and T (wiki train + the `<|doc|>` boundary + chat, the same bytes as the tool's T) is built first; then the class expert; then the FM-index, which takes T; then the trie is trained. Final `class_context.hpp` (per-order merge, chunk 2^17).

| Configuration | Class built: resident / peak | Index built: resident / peak | Trie trained: resident | Peak, now order → M6 order |
|---|---:|---:|---:|---:|
| FM-index, class 3 orders compact, seen once dropped | 61.2 / 186.8 MB | 68.1 / 186.8 MB | 119.6 MB | 302.9 → **186.8 MB** |
| FM-index, class 2 orders compact | 63.7 / 155.3 MB | 70.5 / 208.2 MB | 122.0 MB | 231.0 → 208.2 MB |
| Suffix array, class 2 orders (B4's setting) | 87.2 / 155.6 MB | 145.6 / 161.7 MB | 195.7 MB | 220.7 → 211.4 MB |

- **Prediction M6, on the harness: hit** (186.8 MB with 3 orders, under 200). The qa numbers are read when the tool is wired.
- **The FM build sets the 2-order peak.** It needs the text, its BWT and libsais's 4-byte-a-byte work array at once (about 6 bytes a text byte), and the heap fragments; one run added 144 MB, another at most 125.
- **The registered method does not fit `qa` as it is.** There, the chat text depends on the wiki trie (A1c's familiarity tokens are chosen with it), so the chat text cannot be built before training. M4 gives the same effect another way: the tool packs the trie and frees its training form before InfiniGram and the class expert are built, and the packed form never reads the training text. The harness with M4 (the M4 section) gives 234.5 MB with the earlier class build; with the per-order build the class step adds 132 MB instead of 166–203, so that peak falls further.

### M7: a stored packed trie file

`PackedModel::save` and `PackedModel::load`, each for a path or a stream. The file holds the packed form, the six mixture weights and the scoring split (`train`, `sample`, `alpha`, `agree_floor`, `match_cap`), so a served model is self-contained. The format:
- little-endian, no padding: "CYPHAPK" and a zero byte, version 1, then the split, the weights and the count tables;
- each array as a 64-bit count and its elements;
- last, the FNV-1a 64-bit hash of every byte before it.

`load()` checks the magic and the version, every count against the bytes left, the hash, and that the arrays fit together (`consistent()`: sizes, rising starts, every start inside its array). A damaged file throws `std::runtime_error`. `operator==` compares every table and array. The code is 199 lines. `self_test()` round-trips its fixed-input model in memory (528,449 bytes) and checks that a copy with one bit flipped, and a copy cut short, are refused. That adds 3.6 ms (fastest of 15 runs: 14.4 → 18.0 ms).

Scratch harness `packfile.cpp`. `build` trains as the tool does, packs, saves and reloads. `serve` is a fresh process: it loads the file, reads the 8 MiB text and scores the holdout.

| | bpc model | chat/qa model |
|---|---:|---:|
| Packed form in memory (`bytes()`) | 4,111,378 B | 8,106,219 B |
| **File** | **4,111,298 B** | **8,106,139 B** |
| Save | 6.8 ms | 13.7 ms |
| **Load** in a fresh process (three more loads) | **9.2 ms** (8.2–9.2) | **17.0 ms** (17.5–18.8) |
| Building instead: train, then pack | 54.7 s, then 0.24 s | 369.6 s, then 0.48 s |
| Fresh process, empty | 3.50 MB | 3.56 MB |
| **After load** | **7.86 MB** | **11.72 MB** |
| After reading the 8 MiB text | 15.91 MB | 19.78 MB |
| After scoring the holdout (also the peak) | 16.22 MB | 20.09 MB |
| Building instead: training form / packed and released / peak | 37.27 / 17.27 / 49.80 MB | 75.20 / 40.91 / 95.93 MB |

- **Identical.** The loaded model equals the saved one: every array, the weights and the split. All 1,048,576 holdout positions give the same experts, match and `mix()` distribution, bit for bit.
- **BPC from the file, in a fresh process:** bpc model 2.3862928313539933 (backoff) and 2.2977939389859512 (mixture); chat/qa model 2.3645558122752401 and 2.3445707246454637. These are the training form's numbers.
- **The file is the packed form.** It is 80 bytes smaller than `bytes()`, which also counts the vectors' headers.
- **Resident after load** is the empty process plus the packed form, and 0.43–0.44 MB more (3.50 + 3.92 = 7.42 against 7.86 MB; 3.56 + 7.73 = 11.29 against 11.72 MB). With the text it is 0.49 MB above packed form + text + empty process, for both models.
- **Serving never holds the training form or the chat text.** The chat model's serving peak is 20.09 MB, against 95.93 MB when it is trained in the same process. The build process keeps 40.91 MB after release, because it still holds the chat text.
- Loads were timed with the file in the page cache.
- **Prediction M7: hit.** Bit-identical; the file is 80 bytes under the packed bytes; load 9–19 ms (under 0.2 s); resident 0.49 MB above packed form + text + empty process (within 2 MB).
- **Not done yet:** `cyphalm_trie` options to save a model and serve from a file. They come with the tool's integration.

### M8: a stored FM-index file

`FmInfiniGram::save(path)` and `FmInfiniGram::load(path)`: "FMIG", version 1, the body's size and an FNV-1a checksum, then the arrays as they are. Scratch harness `fmstore.cpp` on the qa training text (19.6 MB):

| Read | Value |
|---|---:|
| Build, then save | 3.7 s, then 0.08 s |
| File | 27,227,720 B; the index in memory is 26.0 MB |
| Load | 0.06 s |
| Peak while loading, over the process before it | +26.0 MB (the index itself) |
| Holdout answers after loading | the same: hash `9652039fc4d09c09` over every holdout query's n, total and count, built and loaded |

- **Prediction M8: hit** (same answers; file within 1%; load 0.06 s; peak + 26 MB, under the index + 10 MB).
- A served model that loads the index never pays the build's 6 bytes a text byte. CTest `native_fm_infinigram_test` checks the round trip on every tenth random text.
