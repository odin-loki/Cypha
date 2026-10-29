# CyphaLM's RAM, and cutting it

**Odin Loch** · 2026-10-28 · PLAN §10 (budgets) and Odin's goal of low RAM

(The profile below is measured; the decision rule and the predictions were written before any change was tried, M5–M9 each before its own run. Results are added under them.)

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

### M6 and M5 in the tool (M9)

`cyphalm_trie` now serves from the packed trie and from the FM-index. The change is +266 / −117 lines over main `8528663` (with A4's `--sim`). `native/CMakeLists.txt` builds `fm_infinigram.cpp` in place of `infinigram.cpp`.
- **M4:** every subcommand packs the trie after training and frees the training form. `bpc --negative` keeps it.
- **M5:** InfiniGram is `FmInfiniGram` everywhere: qa's index, A1c's lowered index (`match_prefix`), `context8k`, `cls`, `dktext` and `Lm`. The holdout loops and `Lm` carry a cursor.
- **M6 order in `qa`:** the trie is packed first. Then T is built and the class table is built from it. Then the index takes T. It copies T instead when `--sim` reads it (`src=all`). The chat bytes leave the text buffer, and the chat string and the Dolly turns are released. The per-byte holdout arrays are allocated after both builds.
- **Class specs:** `TABLE[:tail[:orders[:flags]]]`, orders 2 or 3, flags `c` (compact) and `pN` (min_total N), for example `shape:6:3:cp2`. The fields are read from the right, so `TABLE` and `TABLE:tail` parse as before, and a Windows drive colon stays in the path.
- **M7:** `chat` and `sample` take `--save FILE` (store the trained model) and `--model FILE` (serve it without training). A served `chat` prints no `mix_stored` line, because that line measures the training form.

**Parity.** Each subcommand was run with the old build and the new build. Old means main before the change: `02fc5318`, and `8528663` for the `--sim` rows. Standard output is identical, line for line, except timing fields: `cls`'s `build_s`, A4's `score_s`, `p50_us` and `p99_us`, and qa's `LATENCY` lines (decode times, added by A4). Rows files, sample files and `DK.tsv` are byte-identical. Peak RSS is `VmHWM`, polled every 0.2 s. Peaks of the same code varied by up to 5 MB between runs.

| Subcommand | Output | Peak, old → new |
|---|---|---:|
| `bpc` | identical (22 lines) | 52.5 → 52.0 MB |
| `bpc --negative` | identical (30 lines) | 67.9 → 68.0 MB |
| `chat` | identical (10 lines) | 103.9 → 103.6 MB |
| `sample` | both files identical | 52.4 → 52.2 MB |
| `context8k` | identical (221 lines) | 229.8 → **165.1–169.0 MB** |
| `dktext` | identical (24 lines), `DK.tsv` identical | 321.8 → 275.9 MB |
| `cls shape:6` | identical (10 lines) | 260.5 → 223.0–228.4 MB |
| `qa` A0 | identical (98 lines), rows identical | 264.0 → **170.7–174.8 MB** |
| `qa` C1L (A1d's default) | identical (264 lines), rows identical | 367.7 → **199.1 MB** |
| `qa --cls shape:6` | identical (99 lines), rows identical | 359.9 → 219.7 MB |
| `qa`, class with 3 orders (old: main built with 3 orders; new: `shape:6:3`) | identical but for the spec label (98 lines), rows identical | 413.1 → 294.4 MB |
| `qa --cls shape:6 --sim S` (`src=all`) | identical (491 lines), rows identical | 847.6 → 717.2 MB |
| `qa --cls shape:6 --sim S,src=wiki` | identical (491 lines), rows identical | 556.2 → 406.0 MB |
| `cls shape:6 --sim S --sim-fresh --sim-surface` | identical (15 lines) | 524.0 → 486.0 MB |

S is A4's decision spec, `vec=ppmi_noeval,sif=0,W=2,k=2048,drop=0,lists=2048,probe=32`. The merge with A4 changed only A4's own lines and one line (the index copies T for `src=all`). So `bpc`, `chat`, `sample` and `dktext` were run before the merge, and `qa`, `cls`, `context8k` and the `--sim` rows after it. `bpc`, `cls shape:6` and `context8k` were run again with the final build, which is byte-identical to the CMake build of the repo.

**New paths.**

| Run | Result | Peak |
|---|---|---:|
| `cls shape:6:3:cp2` | M2/M3's row exactly: 885,985 contexts, 10,787,584 table bytes, fresh trie + IG + class **2.085255**, surface **0.9067** | 219.8 MB |
| `qa --cls shape:6:3:cp2` | 1,562,354 contexts, 19,115,288 bytes; holdout2 trie + IG + class 1.994082 | 234.9 MB |
| `chat --model chat.pk` | identical to `chat` but for the `mix_stored` line; 12 s against 476 s | 20.9 MB (training: 103.9) |
| `sample --model sample.pk` | both files identical to `sample`'s | 18.3 MB (training: 52.4) |

The qa class table has one context more than the `memprof3` harness's (1,562,353), because the tool's T has the `<|doc|>` boundary.

**Where the peaks are now.** Scratch profile of `qa` A0: `VmRSS` every 0.25 s, with stage markers on stderr.
- Training peaks at 107.6 MB. After packing, 90.4 MB stays resident, 42 MB of it the per-byte holdout arrays.
- The FM build on T (19.6 MB) adds about 128 MB. That was 218.4 MB with the arrays allocated first. With the arrays allocated after the builds, it is 170.7–174.8 MB.
- In `qa --cls shape:6:3:cp2` the class build sets the peak instead (234.9 MB). The index build then fits in the freed heap.
- **The heap keeps what the builds free.** After the index build of `qa` A0, 144.1 MB stays resident. `malloc_trim(0)` returns 43.5 MB of it (to 100.6 MB). The tool does not call it, because it is glibc-only.
- **Releasing buffers does not always lower the peak.** In `qa`, releasing the chat string and the turns lowered the peak by 24 MB (198.8 → 174.8). In `context8k`, releasing the chat string raised it by 16–17 MB (165.1 → 181.7, 169.1 → 185.9), so `context8k` keeps it.

**Predictions.**
- **M9:**
  - every printed number identical: **hit**;
  - `cls shape:6:3:cp2` gives 2.085255: **hit**;
  - `qa` A0 peak under 160 MB: **miss** (170.7–174.8, from 264.0);
  - `context8k` under 160 MB: **miss** (165.1–169.0, from 229.8);
  - `qa --cls shape:6:3:cp2` under 200 MB: **miss** (234.9).
- **M6 on the tool:**
  - every qa number identical: **hit**;
  - peak under 200 MB with 3 orders: **miss**.
  - What was compared. "Now" is the old tool with a 3-order class table, built from main with orders set to 3: 413.1 MB. The same table on the new tool (`shape:6:3`) takes 294.4 MB; compact with contexts seen once dropped (`shape:6:3:cp2`), 234.9 MB. With 2 orders it is 359.9 → 219.7 MB. The 303 MB in M6's prediction came from the `memprof3` harness.
- **Why the peaks stay above 200 MB.** In `qa` the class table and the index cannot be built before training, because A1c's tokens need the wiki trie before the chat text exists. So their builds, about 130–170 MB each over what is resident, come after training and on top of the packed model and the holdout data.

### Odin's rulings (2026-10-28)

- The smaller class table (M2 + M3, three orders) is the default: `class_context.hpp`'s `Options` and a flagless `cyphalm_trie` class spec.
- Returning freed build memory to the OS (`malloc_trim`, glibc only): not wanted. The tool stays portable.
- The similar-context expert is a default in `qa`, `sample` and `context8k` (Odin; A4 report, A4b). Its 32-d int8 vectors and int8 keys add 42 MB resident (its first version added 142 MB); `qa` A0's peak 175 → 187 MB.

## M10–M12: `qa` served from stored files (registered 2026-10-28, before any run)

**Question.** Serving should never train or build (HANDOFF §6.5). `qa`'s peak with three-order classes is 235 MB (M9), and the target is 200 MB at most. The builds come after training, because A1c's tokens need the wiki trie. Can `qa` load what it needs from files instead?

**What `qa` reads after training.**
- The packed chat-stage trie and its weights: every holdout read and every answer.
- The wiki-stage weights. After M4, A1c's `wiki_lm` reads the packed chat-stage trie with these weights.
- InfiniGram over T, the class table over T, and A1c's lowered index (`--parse`).
- The familiarity statistic's cut and the follower limit.
- The similar-context expert. It is built from its 2.2 MB vector file over wiki train, which serving reads anyway.

**What cannot be served from a file.** Before the chat passes, `qa` reads the wiki-stage trie in its training form, with its live match table. The packed form does not hold that table (M4). That model gives:
- the build's calibration and analysis lines (`FAMSTAT`, `ENTITY`, `FAMCAL`, `FAMTOK`);
- the tokens of the training text (`fam_dolly`), and the chat composition (`chat_turns`);
- the `question_lp` and `ent_lp` token statistics.

So a served model takes the token statistics that read only the lowered index: `ent_count`, `ent_link`, `ent_bold`, `ent_title`, `ent_igfrac` and `ent_tb`. C1L uses `ent_title`. The build's calibration lines are printed at build time only.

**Design.**
- **`qa --save DIR`:** the normal run, and after the builds it writes these files:
  - `model.pk`: the packed trie, its weights and split (M7);
  - `ig.fm`: InfiniGram over T (M8);
  - `low.fm`: the lowered index (with `--parse`);
  - `cls.cc`: the class table (with `--cls`, M10);
  - `qa.meta`: the record of the options the model was built from, the wiki-stage weights, the follower limit and the cuts, with a checksum.
- **`qa --model DIR`:** loads them. It reads the 8 MiB wiki text (the holdout contexts and the grading) and not Dolly. The only thing it builds is the similar-context expert.
  - It refuses a store whose record differs from the options given: wiki bytes, split, Dolly, `--dk`, `--dk-ratio`, `--fam`, `--fam-stat`, `--fam-train`, `--parse`, the shards and the class spec.
  - It refuses `--fam-only`, a token statistic that reads the model, and a similar-context spec with `src=all`.
- **Names.** `chat` and `sample` already use `--save` and `--model` for a stored packed trie (M7); for `qa` they take a directory. `--from` is taken by `dktext`.
- **M10:** `ClassContext::save` / `load`. The file holds "CYPHACC" and a zero byte, version 1, the options, the word-class map (sorted) and the arrays (each a u64 count and its elements), then an FNV-1a 64-bit hash of every byte before it. A damaged or cut file throws.
- **M12:** a stored similar-context index, only if measurement says it pays.

**Decision rule (lossless; fixed before the runs).**
1. A build run with `--save` prints exactly what the tool without `--save` prints, and writes the same rows file. Timing fields are excepted: `build_s`, the `LATENCY` lines, and `SIM_LAT`'s `p50_us` and `p99_us`.
2. A `--model` run prints the build run's lines without the build-only ones (`FAMSTAT`, `ENTITY`, `FAMCAL`, `FAMTOK`, `fam_dolly`, `chat_turns`), in the same order, timing fields excepted. Its rows file is byte-identical.
3. The saving is measured: peak RSS (`VmHWM`, polled each second), wall time and file sizes, on the same machine.
4. Anything lossy follows rule 2 at the top of this report. None is planned.

**Configurations.**
- **A0:** `qa` with `--dev` and `--leak`, the default similar-context expert.
- **E:** A0 with `--cls shape --copy --copy-min 8`: what the echo fix may make the default.
- **C1L+E:** A1d's C1L options with `--cls shape --copy-min 8`.

**Predictions (registered before the runs).**

| # | Change | Prediction |
|---|---|---|
| M10 | Class table file | Loaded table equal in every array; every prediction identical; file within 1 KB of `bytes()` plus the word-class map; load under 0.1 s for the qa table (19 MB); a flipped bit and a cut file refused |
| M11a | `qa --model`, A0 | Peak 140–175 MB (the build run: 186.9, Q0S); wall time under half the build run's |
| M11b | `qa --model`, E | Peak 160–190 MB |
| M11c | `qa --model`, C1L+E | Peak under 200 MB: the target |
| M11d | `qa --save` | Same peak as without it (within the 5 MB run-to-run spread) and the same output |
| M11e | The stored files | Each within 1% of the part's bytes in memory; all parts load in under 1 s |
| M12 | A stored similar-context index | Pays only if its build is over 20% of a `--model` run's wall time or sets its peak. Then the same neighbours, and a load under 0.2 s |

### M13: the order of `qa`'s last steps (registered 2026-10-28, before its runs)

**Why.** The first runs (results below) show where the memory goes after the builds. A served A0 run peaked at 182.9 MB:
- the loaded parts and the 8 MiB text are 46 MB;
- the per-byte holdout arrays add 40 MB at once, and more later;
- the similar-context expert is then built on top of them: +88 MB at its build, 40 MB of it the finished index.

The build run of E peaks at its class build (245 MB). Afterwards, the arrays and the similar-context build reach about 210 MB.

**Change (lossless).**
1. The similar-context expert is built right after InfiniGram, before the holdout arrays exist. Its lines print where they did.
2. Each per-byte array is made only when it is read: the copy expert's with `--copy`, the class expert's with `--cls`.
3. Each array is freed after its last read: InfiniGram's probabilities and buckets once the trie + InfiniGram mixture is made, and the class expert's once its mixture is made.

**Predictions (registered before the runs).** Peaks are `VmHWM` in kB / 1000, as in M9 and A4b.

| # | Run | Prediction |
|---|---|---|
| M13a | Every run | Every printed number and the rows file identical to the M11 tool's (timing fields excepted) |
| M13b | `qa --model`, A0 | 182.9 → under 140 MB |
| M13c | `qa --model`, E | Under 170 MB |
| M13d | `qa --model`, C1L+E | Under 200 MB |
| M13e | `qa --save`, A0 | 187.1 → under 178 MB. The index build (174.7 MB) then sets the peak |
| M13f | `qa --save`, E | Unchanged within 5 MB: the class build (245 MB) sets it |

### M14: the lowered index freed after its last read (registered 2026-10-28, before its runs)

**Why.** Served C1L+E peaked at 209.0 MB under M13 (results below), over the 200 MB target. Its loaded parts are 98 MB. A1c's lowered index is 26 MB of them. With an entity statistic in the token (C1L's `ent_title`), the token step chooses and keeps (`chosen`) the entity of every dev and test question. After that step nothing reads the index.

**Change (lossless).** `entity_of` reads its kept choice first. After the token step, the index is freed when the token uses an entity statistic. With `--save` it is kept, because the store writes it later.

**Predictions (registered before the runs).**

| # | Run | Prediction |
|---|---|---|
| M14a | Every run | Every printed number and the rows file identical (timing fields excepted) |
| M14b | `qa --model`, C1L+E | 209.0 → under 190 MB |
| M14c | `qa`, C1L+E, no `--save` | 275.1 → under 255 MB: the class build, which sets the peak, comes after the index is freed |

### Results: M10–M14 (2026-10-28)

**How it was run.**
- `cyphalm_trie qa` on the 8 MiB protocol, the default similar-context expert in every configuration.
- The unchanged tool is main `6ad7198`, built from a frozen copy.
- Peaks are `VmHWM` (the kernel's exact high-water mark) in kB / 1000, as in M9 and A4b. A scratch script also polled `VmRSS` every 0.25 s next to the tool's progress lines on stderr, to find the phase.
- The machine was shared with two other agents' `qa` runs (load 4–6 on 4 cores), so wall times are noisy. Peaks are not.
- Parity: standard output compared line by line, with only the timing fields masked (`build_s`, `LATENCY`, `SIM_LAT`'s p50 and p99). Rows files compared byte for byte.

#### M10: the class table file

- `ClassContext::save` / `load` and `operator==`: +145 lines in `class_context.hpp`.
- CTest `native_class_context_test` (new): 979 checks.
  - 16 settings: shape classes or a word table, 2 or 3 orders, full or compact storage, min_total 1 or 2.
  - For each: build, save, load. The table must be equal in every array, and every prediction must be the same at every third position of the training text and of an unseen text.
  - 48 one-bit flips, 6 cuts and an appended byte per file must each be refused.
  - A planted bug (load drops the stored min_total) fails 9 checks.
- The qa tables:

| Table | Contexts | `bytes()` | File | Load |
|---|---:|---:|---:|---:|
| E (wiki train + chat) | 1,562,354 | 19,115,288 | 19,115,392 | 0.043 s |
| C1L+E (with the don't-know and shard turns) | 1,596,160 | 19,537,894 | 19,537,998 | 0.037 s |

- **Prediction M10: hit.** Equal tables and predictions; files 104 bytes over `bytes()`; loads under 0.1 s; damage refused.

#### M11: `qa --save DIR` and `qa --model DIR`

- **Code.** In `cyphalm_trie.cpp`:
  - a store record and its helpers (`QaStore`, `qa_record`, `save_qa`, `load_qa`);
  - `cmd_qa`: skip training, Dolly and the builds when serving; write the store after the builds.
  - The chat block is wrapped and re-indented.
- **Refusals** (each exits 2 with a message): `--model` with `--save`, `--fam-only`, `--fam` with `question_lp` or `ent_lp`, and `--sim` with `src=all`. A store whose record differs from the options given is refused with both records printed. A missing or damaged file is refused.

**Parity.**

| Configuration | Unchanged tool → `--save` | `--save` → `--model` |
|---|---|---|
| A0 | 496 lines identical; rows identical (521 lines, sha256 `8aa710fa…`) | 495 lines identical; rows identical. Dropped: `chat_turns` |
| E | 594 lines identical; rows identical (1,041 lines, `b927e4a8…`) | 593 lines identical; rows identical. Dropped: `chat_turns` |
| C1L+E | 661 lines identical; rows identical (1,041 lines, `f2389a5a…`) | 634 lines identical; rows identical. Dropped 27: `FAMSTAT` ×8, `FAMCAL` ×8, `FAMTOK` ×8, `ENTITY`, `fam_dolly`, `chat_turns` |

- The served C1L+E run still prints `fam_thresholds`, `fam_dk`, `fam_dev` and the shard line, identical to the build's. Its dev tokens come from the stored cut and the loaded lowered index.
- Stores written by two different runs (M11 and M13 tools) are byte-identical.

**The stores.**

| Part | A0 | E | C1L+E | Load (C1L+E) |
|---|---:|---:|---:|---:|
| `model.pk` | 8,106,139 | 8,106,139 | 8,360,801 | 0.020 s |
| `ig.fm` | 27,227,720 | 27,227,720 | 28,664,760 | 0.052 s |
| `low.fm` | | | 25,772,480 | 0.045 s |
| `cls.cc` | | 19,115,392 | 19,537,998 | 0.037 s |
| `qa.meta` | 443 | 455 | 623 | |
| **All** | **35.3 MB** | **54.4 MB** | **82.3 MB** | **0.15 s** |

Each file is within 0.05% of its part's `bytes()`. Loads are the fastest of five, with the files in the page cache.

#### M13 and M14

- **M13.** The similar-context expert is built before the per-byte holdout arrays. Each array is made only when read, and freed after its last read. +30 / −20 lines.
- **M14.** `entity_of` reads its kept choice first. The lowered index is freed after the token step when the token uses an entity statistic. +4 lines.
- **Parity.** Every M13 and M14 run printed the same lines as the M11 run of its configuration (timing masked), with identical rows: build and served, A0, E and C1L+E.

#### Peaks and wall times

| Configuration | Unchanged tool | `--save`, M11 | `--model`, M11 | `--model`, M13 (two runs) | `--save`, M13 | `--model`, M14 (two runs) |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 186.8 MB, 922 s | 187.1, 1,017 s | 182.9, 190 s | **138.4 / 138.4**, 376 / 195 s | 175.2, 1,109 s | (as M13) |
| E | 245.3 / 232.1 MB (two runs), 778 / 1,083 s | 225.8, 1,148 s | 201.5, 328 s | **169.4 / 169.3**, 365 / 245 s | 225.8, 802 s | (as M13) |
| C1L+E | 275.1 MB, 1,236 s | 302.2, 1,059 s | 241.3, 235 s | 209.0 / 209.1, 226 / 215 s | 269.2, 703 s | **189.1 / 189.2**, 205 / 217 s |

A0 and E take no token statistic, so M14 leaves them as M13 has them. C1L+E built without `--save` under M14 peaked at 284.5 MB (730 s), with every line and the rows identical to the unchanged tool's.

**Where the memory goes.**
- **Unchanged tool.**
  - A0: the similar-context build on top of the holdout arrays (186.8); the index build reaches 174.7.
  - E and C1L+E: the class build right after training (245 / 275).
- **Served, M11.** The stored parts are 35–82 MB of files; the 8 MiB text and the question files come on top (about 100 MB in all for C1L+E). The per-byte holdout arrays then take 40 MB at once, 68 MB in all. The similar-context build then adds about 90 MB on top of them, 40 MB of it the finished index.
- **Served, M13.**
  - The similar-context build comes first and needs nothing else.
  - The arrays come after it.
  - The peak comes after that. The similar-context and copy steps each read six per-byte arrays at once (44 MB), and answering adds a little more.
- **Served, M14.** C1L+E without the lowered index after the token step: 189 MB.
- **Build runs vary.** The class build's peak changes from run to run with the same code and data: 245.3 and 232.1 MB for the unchanged tool on E. Whether the C library keeps the build's freed memory resident afterwards also changes: 224.6 MB resident after the build in the unchanged tool's C1L+E run, 268.8 MB in the M11 `--save` run. In the M14 build run the class build started from 92 MB resident (the unchanged tool's: 135 MB) and still peaked at 284.5 MB: it added about 190 MB, against about 140 MB in the unchanged tool's run. The cause was not found. The program runs on one thread, so the likely candidate is where the C library's heap lands from run to run. Served runs repeat to 0.1 MB.
- **The similar-context build** takes 26–51 s of a served run: 12–16% of its wall time in all 11 served runs. After M13 it does not set the peak. So M12 (a stored similar-context index) was not built.

#### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| M10 | Equal table, same predictions, file within 1 KB, load under 0.1 s, damage refused | **Hit** |
| M11a | `--model` A0: peak 140–175 MB; wall under half the build run's | **Miss** on the peak (182.9); **hit** on time (190 against 1,017 s) |
| M11b | `--model` E: 160–190 MB | **Miss** (201.5) |
| M11c | `--model` C1L+E: under 200 MB | **Miss** (241.3) |
| M11d | `--save`: same peak within 5 MB | A0 **hit** (+0.3). E and C1L+E **miss**: the build's own spread is wider (above) |
| M11e | Files within 1%; all parts load in under 1 s | **Hit** (0.05%; 0.15 s) |
| M12 | A stored similar-context index only if it pays | **Not built**: 12–16% of a served run's wall time, and not the peak after M13 |
| M13a | Identical lines and rows | **Hit** |
| M13b | `--model` A0 under 140 MB | **Hit** (138.4) |
| M13c | `--model` E under 170 MB | **Hit** (169.4) |
| M13d | `--model` C1L+E under 200 MB | **Miss** (209.0) |
| M13e | `--save` A0 under 178 MB | **Hit** (175.2) |
| M13f | `--save` E unchanged within 5 MB | **Hit** (225.8 and 225.8) |
| M14a | Identical lines and rows | **Hit** |
| M14b | `--model` C1L+E under 190 MB | **Hit** (189.1 / 189.2) |
| M14c | C1L+E without `--save`: 275.1 → under 255 MB | **Miss** (284.5): the index is freed, but this run's class build added more (below) |

**Decision.** By the rule (lossless, measured), M10, M11, M13 and M14 are adopted.
- `qa` served from a store never trains or builds, apart from the similar-context index.
- Every configuration measured is under the 200 MB target when served: A0 138, E 169, C1L+E 189 MB.
- It is 3–5 times faster than building on this machine (190–376 s against 703–1,148 s).
- Building a model still peaks at the class build (226–302 MB, varying by run). Building is a one-time step.

**Still open.**
- **Building** still peaks at the class build: 226–302 MB, varying by run.
- **A served model cannot use `question_lp` or `ent_lp` tokens.** They read the wiki-stage trie with its live match table, which no stored form holds.
- **`sample` and `chat`** serve a stored trie (M7). When the class expert becomes a default there, their `--save` / `--model` should carry its table too (`ClassContext::save` / `load`).
- **Memory-mapping the stored files** was not tried. The data-structure cuts were enough for the target.

## M15: the class build's peak when building (registered 2026-10-28, before any run)

**Question.** Building a `qa` model with B4c's defaults (class + similar-context + copy) peaks at the class build. C1L+E built at 264–275 MB, and the same code and data varied 226–302 MB between runs (M10–M14). Served, it is 189 MB. Why does the build's peak vary, and can it be made lower and steady without changing a number?

**What is known.** The class build adds about 132–190 MB at its peak (M1, M14). It started from 92–135 MB resident in the M14 runs. Whether the C library keeps the build's freed memory resident afterwards also changed between runs (224.6 against 268.8 MB after the build).

**Decision rule (fixed before the runs).**
1. **Lossless.** Every printed number and the rows file byte-identical to the unchanged tool (main `e8748a2`) for A0 (`--no-cls --no-copy`), E (the defaults) and C1L+E. Timing fields are excepted: `build_s`, the `LATENCY` lines, `SIM_LAT`'s p50 and p99. The class table must be equal to the unchanged build's (`operator==`; the stored `cls.cc` files byte-identical).
2. **Peaks** are `VmHWM` in kB / 1000, at least two runs per configuration and tool. A change is kept if it lowers the built peak of E and C1L+E by more than the unchanged tool's spread between repeat runs, or makes the spread 5 MB or less without raising the mean.
3. **No OS-specific calls in the code** (no `malloc_trim`, `mallopt`, `madvise`): Odin, "don't worry about the OS". Allocator settings may be changed by environment variable in test runs only, to test the cause.
4. **Lossy changes** follow rule 2 at the top of this report. None is planned.

**Target.** C1L+E built at 220 MB or less, with repeat runs within 5 MB.

**Predictions (registered before the runs).**

| # | Prediction |
|---|---|
| M15a | The unchanged tool's repeat runs of E differ by more than 10 MB at the peak (M14: 245.3 and 232.1) |
| M15b | The peak of E and C1L+E is reached inside the class build, in every run |
| M15c | The spread comes from the C library's heap. With large blocks forced into their own mappings by environment variable (`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072`), repeat runs agree within 5 MB and the peak falls by at least 20 MB |
| M15d | Most of the build's extra memory is the chunk-by-chunk merge: each chunk reserves a new array of (the order's runs + the chunk's events) while the old one is alive, on top of every order's runs. A build whose arrays never regrow cuts the build's addition by at least a third |

The change itself is chosen after the profile, and its predictions are registered before its runs (as M13 and M14 were).

### M15e: the class build in key buckets (registered 2026-10-28, before its runs)

**Why.** A harness that builds only the default class table over an approximate T (wiki train + raw Dolly, 20.6 MB) shows where the build's memory goes:
- the distinct (context, byte) pairs are 1.28 M, 1.66 M and 2.14 M for the three orders: 5.09 M in all, 61 MB as 12-byte runs;
- the table is 18.8 MB;
- yet the build adds 151 MB to the peak (36 → 187 MB), and 73 MB stays resident afterwards for a 19 MB table.

With the C library's mmap threshold fixed by environment variable (test runs only), the same build adds 103 MB and leaves 42 MB resident. So about 48 MB of the peak is heap left behind by the merge. Every chunk reserves a new run array a little longer than the order's old one, and frees the old one, which is too short to reuse. The harness repeats to 0.1 MB alone, but with 80 MB held before the build it peaks at 269 MB. So the spread in `qa` comes from the heap the build starts from.

**Change (lossless).** In `ClassContext::build`:
1. Every order's runs go into one array, since the table adds up a (key, byte) counted in two orders anyway.
2. The runs are split into 256 buckets by the key's top 8 bits. A chunk's sorted events fall in each bucket as one stretch, and only that bucket's runs (about 0.25 MB) are copied by its merge.
3. The table is filled bucket by bucket, and each bucket is freed once read.

**Predictions (registered before the runs).**

| # | Prediction |
|---|---|
| M15e1 | The class table is byte-identical (stored files) in the harness and in `qa` (A0, E, C1L+E); `native_class_context_test` passes unchanged; every printed number and the rows file identical |
| M15e2 | Harness: the build adds under 100 MB to the peak (151 before) |
| M15e3 | Harness: with 80 MB held before the build, the addition is the same within 5 MB |
| M15e4 | `qa` built: E under 210 MB; C1L+E 220 MB or less (the target); repeat runs within 5 MB |
| M15e5 | The build's time is within +20% of the unchanged build's |
| M15e6 | The class build no longer sets `qa`'s peak in E. The next highest is the FM-index build or the similar-context build |

### Results: M15 (2026-10-28)

**How it was run.**
- `cyphalm_trie qa` on the 8 MiB protocol. The unchanged tool is main `e8748a2`, and the changed one is that commit plus M15e.
- Configurations, with B4c's defaults: **A0** (`--no-cls --no-copy`), **E** (the defaults: class + similar-context + copy) and **C1L+E** (A1d's C1L options).
- Peaks are `VmHWM` in kB / 1000. A wrapper also polled `VmRSS` every 0.25 s next to the tool's progress lines, stamped with their times, to place each peak in a phase.
- The machine was shared with other agents' runs, so wall times are noisy.
- A harness (scratch, not committed) built only the default class table over an approximate T: wiki train plus the raw Dolly file, 20.6 MB.

**The cause.**
- The unchanged build kept one run array per order and, for every 131,072-byte chunk, reserved a new array of (the order's runs + the chunk's events) before freeing the old one.
- In the harness, that build adds 151 MB to the peak for a 19 MB table, and leaves 73 MB resident after it.
- With the C library's mmap threshold fixed by environment variable, the same build adds 103 MB and leaves 42 MB. In `qa` (E), the unchanged tool under that setting peaks at 169.3 MB instead of 225.8, one run. So about 50 MB of the build's peak was heap: freed arrays too short to reuse, held resident.
- With 80 MB held before the build, the harness peaks at 269 MB (+164): the build's cost depends on the heap it starts from. This is the spread M10–M14 saw between runs of different tool versions.
- A first bucketed build still added 110 MB in the harness. Its run arrays held 92 MB of capacity for 61 MB of runs: the spare merge array passes from bucket to bucket and never shrinks, so every bucket grew to the largest one's room. Sizing each merge to the distinct events it can add, and dropping a spare array more than 1/16 too long, fixed it. That was before any `qa` run of the change.

**The change.** `ClassContext::build` (`class_context.hpp`, +126 / −109 lines):
- every order's runs in one array, split into 256 buckets by the event's top 8 bits;
- a compact table's runs keep 32-bit keys, the bits below the bucket's (56 bits in a u64 otherwise);
- each merge sized to the distinct events it can add;
- the table filled bucket by bucket, each bucket freed once read.

`class_context_test` gains 64 checks: every one of its 16 settings, built in chunks of 1, 7, 997 or 2^20 bytes, must equal the 4,096-byte build. A planted merge bug (old counts lost across chunks) fails all 64. The 979 earlier checks did not catch it.

**Parity.**
- **Tables.** The harness tables are byte-identical to the unchanged build's in 11 settings:
  - compact and full storage;
  - 2 and 3 orders;
  - min_total 1 to 3;
  - chunks of 1 to 131,072 bytes;
  - texts of 0 to 20.6 MB.
- **`qa` stores.** The stores written by `qa --save` are byte-identical, file for file:
  - E: `cls.cc` 19,115,392 bytes, `ig.fm`, `model.pk`, `qa.meta`;
  - C1L+E: `cls.cc` 19,537,998 bytes, `low.fm`, `ig.fm`, `model.pk`, `qa.meta`.
- **Output.** Every A0, E and C1L+E run, with and without `--save`, printed the same lines as the unchanged tool (timing masked) with byte-identical rows.

**Peaks** (MB; repeat runs listed).

| Configuration | Unchanged tool | Changed tool | Where the changed tool peaks |
|---|---:|---:|---|
| A0 | 175.3 | 175.3 | the similar-context build (no class table) |
| E | 225.8 / 225.9 / 225.8 (`--save`) | **194.0 / 194.1** (`--save`) | the FM-index build after the class build |
| C1L+E | 259.3 | **235.0** | the FM-index build |
| C1L+E with `--save` (the lowered index kept for the store) | 264.0 | 260.7 | the FM-index build, with the lowered index's 26 MB |

- **The class build itself** now adds 22 MB in E (110.4 → 132.5) and 19 MB in C1L+E (92 → 111): its small bucket arrays reuse heap the training freed. Before, it added 115 MB (E: 110.5 → 225.8).
- **The class build's time** is unchanged: 11.4 s against 11.2 s (E), and 11.6 s against 12.1 s (C1L+E). In the harness, alternating runs gave 12.2–14.2 s against 11.6–15.2 s.
- **The next peak is the FM-index build** (`FmInfiniGram` over T, 21 MB). libsais needs a 4-byte suffix-array scratch per byte (85 MB). The text and the first wavelet tree are held while the second BWT is made. In the harness that build adds 114 MB. In `qa` C1L+E it goes 111 → 215 (first BWT) → 235 MB (second BWT).
  - Building the index first does not help: the class build needs T, so the index would have to copy it.
  - A served model (M11) loads the index and never builds it.

#### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| M15a | Unchanged E repeats differ by more than 10 MB | **Miss.** 225.8, 225.9 and 225.8: the same binary repeats on this machine. The spread M10–M14 saw was between different tools' runs and options (`--save`, M13, M14), which change the heap the class build starts from |
| M15b | The peak of E and C1L+E is inside the class build | **Hit.** At its end in every unchanged run; the index build's first second reaches the same level within 0.1 MB |
| M15c | With a fixed mmap threshold, repeats agree within 5 MB and the peak falls 20 MB or more | **Hit** on the fall (harness 187.4 → 126.7; `qa` E 225.8 → 169.3, one run). Repeats only measured in the harness (within 0.1 MB) |
| M15d | The merge is most of the build's extra memory; arrays that never regrow cut the addition by a third | **Hit.** 151 → 73 MB in the harness (−52%) |
| M15e1 | Byte-identical table, lines and rows; the CTest passes | **Hit** |
| M15e2 | Harness: the build adds under 100 MB | **Hit** (73). The first bucketed version added 110, fixed before the `qa` runs (above) |
| M15e3 | Harness with 80 MB held: the same addition within 5 MB | **Hit** (73.1 and 73.1; unchanged: 151 and 164) |
| M15e4 | `qa` E under 210 MB; C1L+E 220 or less; repeats within 5 MB | E **hit** (194.0, 194.1); C1L+E **miss** (235.0: the index build) |
| M15e5 | The build's time within +20% | **Hit** |
| M15e6 | The class build no longer sets the peak in E; the index or the similar-context build does | **Hit:** the FM-index build |

**Decision.** By the rule (lossless, measured; the E peak falls 31.8 MB, far beyond the unchanged tool's 0.1 MB spread), M15e is adopted.
- Building a `qa` model peaks at 194 MB (E) and 235 MB (C1L+E), down from 226 and 259.
- Repeat runs agree within 0.1 MB.
- The class build no longer sets any peak. The 220 MB target for C1L+E is missed because of the FM-index build.

**Still open.**
- **The FM-index build** sets the built peak: libsais's 4-byte scratch per byte. A lower-memory BWT, or freeing what is resident at that moment, is the next step. A served model never builds it.
- **The heap.** The C library holds memory freed by earlier steps. Without code changes, a fixed mmap threshold takes the unchanged tool's E build to 169 MB. That is an environment setting, not code (Odin: "don't worry about the OS"), so it is recorded here, not adopted.

## M16: the FM-index build's peak (registered 2026-10-28, before any run)

**Question.** After M15, building a `qa` model peaks inside the FM-index build: C1L+E went 111 → 215 MB (first BWT) → 235 MB (second BWT), E peaked at 194 MB. libsais needs 4 bytes of suffix-array scratch per text byte (T is about 21 MB, so 85 MB). Can the build's peak come down without changing a byte of the index?

**Where the memory goes (from the code).** The build holds, at its first BWT, the text, a copy for the BWT and the scratch: about 6 bytes a text byte. Each wavelet tree is built by copying each node's sequence into a left and a right part, so the build makes and frees n bytes per tree level, and the C library keeps some of it. At the second BWT it holds the reversed text, the scratch and the first tree. In `qa` the class table (19.5 MB) is also resident, since it is built from T first.

**Change (lossless), in three parts.**
1. **Both BWTs in place.** The forward BWT overwrites the text. The reversed text for the second BWT is rebuilt from the first tree by one LF walk (from the empty suffix back to the start, which yields the text backwards), written into the same buffer. No copy of the text is kept.
2. **Trees filled without copies.** The tree's shape depends only on the byte counts, so it is made first (the same split rule and the same node order). Then one pass over the sequence appends each byte's bit to every node on its path. Nothing but the tree's own bits is allocated.
3. **In `qa`, the index before the class table.** `FmInfiniGram::text()` gives the text back from the index (one LF walk over the reversed text's tree, which yields the text in order). The index is built from T first, then the class table from the text the index returns. The class table is no longer resident during the scratch. When the similar-context expert reads T (`src=all`), T is copied into the index as before and kept.

**Decision rule (fixed before the runs).**
1. **Lossless.** The index is byte-identical: the stored `ig.fm` and `low.fm` files equal the unchanged tool's. Every printed number and the rows file are byte-identical to the unchanged tool (wip/integ2 `95b5883`: main `e8748a2` with S4, S6, M15 and A4c) for A0 (`--no-cls --no-copy`), E (the defaults) and C1L+E. Timing fields are excepted (`build_s`, `LATENCY`, `SIM_LAT`'s p50 and p99). `native_fm_infinigram_test` passes unchanged, and new checks show the new build path gives the same index as the old one and that `text()` returns the text.
2. **Peaks** are `VmHWM` in kB / 1000, at least two runs per configuration and tool. A change is kept if it lowers the built peak of E and C1L+E by more than the repeat spread.
3. **No OS-specific calls** (Odin: "don't worry about the OS").
4. **Time.** A `qa` build is at most 10% slower.

**Target.** C1L+E built at 200 MB or less, with repeat runs within 2 MB.

**Predictions (registered before the runs).**

| # | Prediction |
|---|---|
| M16a | The unchanged tool (`95b5883`) peaks inside the FM-index build in E and C1L+E, at the second BWT: C1L+E 225–240 MB, E 185–200 MB |
| M16b | Harness (the index alone over a 21 MB text): the new build's peak addition is at least 20 MB below the old one's |
| M16c | The index is byte-identical from both build paths; `text()` returns the text exactly; `native_fm_infinigram_test` passes unchanged |
| M16d | With the index built before the class table, C1L+E built is 200 MB or less (the target) and E 180 MB or less; repeats within 2 MB |
| M16e | The two LF walks and the tree fill add under 10 s to a `qa` build, and the index build alone is at most 50% slower than before |
| M16f | In C1L, the lowered index's build (the 203.9 MB phase in M15's trace) falls by at least 20 MB |

### Results: M16 (2026-10-29)

**How it was run.**
- `cyphalm_trie qa` on the 8 MiB protocol. The unchanged tool is wip/integ2 `95b5883`; the changed one is `ec0d773` (that commit plus M16). Configurations as in M15, with B4c's and A4c's defaults: **A0** (`--no-cls --no-copy`), **E** (the defaults) and **C1L+E**, and `context8k` with its defaults.
- Peaks are `VmHWM` in kB / 1000. M15's wrapper polled `VmRSS` every 0.25 s next to the tool's stamped progress lines, to place each peak in a phase.
- Runs alternated the two tools, one at a time, on a machine shared with three other agents, so wall times are noisy.
- A harness (scratch, not committed) built only the index over wiki train plus the raw Dolly file (20.6 MB), moved in.

**Harness: the index alone (20.6 MB of text).**

| Build | Peak added over the text | Resident after | Build time | `text()` |
|---|---:|---:|---:|---:|
| Before M16 | 114.4 / 114.5 MB | index + 21 MB of heap the tree build left | 2.70 / 2.76 s | — |
| M16 | 94.8 / 94.8 MB | the index (28.9 MB) and nothing else | 3.52 / 3.61 s | 5.6 / 5.8 s |

- A first version rebuilt the reversed text by one LF walk through the forward tree. It used the same memory but took 8–9 s, since every step is a cache miss per tree level. libsais's inverse BWT does the same in place in about a second with the same 4-byte scratch as a BWT, so the build uses it; `text()` keeps the walk, which needs no scratch.

**Parity.**
- **Index.** `native_fm_infinigram_test` gains golden hashes: the stored index of 12 fixed texts (1 to 300,000 bytes, 1 to 256 symbols, half zeros or none), written by the build before M16, through both constructors. The M16 build gives the same bytes. A planted bug that only renumbers the tree's nodes (every query still right) fails 9 of the 12. `text()` returns each text exactly. 191,909 checks in all.
- **`qa` stores.** `qa --save` writes byte-identical stores with both tools: E (`cls.cc`, `ig.fm`, `model.pk`, `qa.meta`) and C1L+E (the same plus `low.fm`).
- **Output.** Every A0, E and C1L+E run, with and without `--save`, printed the unchanged tool's lines (timing masked) with byte-identical rows. `context8k` printed the unchanged tool's 224 lines (timing masked).
- All 9 CTests from HANDOFF §4.4 pass.

**Peaks** (MB; repeat runs listed).

| Configuration | Unchanged tool | M16 | Where M16 peaks |
|---|---:|---:|---|
| A0 | 175.3 | **132.2** | the index build (the scratch over T) |
| E | 194.0 / 194.2 (`--save`) | **169.9 / 170.1** (`--save`) | the holdout arrays after the similar-context build |
| C1L+E | 235.2 | **184.6 / 184.5** | the lowered index's build (A1c's token step) |
| C1L+E with `--save` (the lowered index kept for the store) | 260.8 | **204.5** | after the store: the lowered index is still resident |
| `context8k` | 225.8 | **202.3** | after the class table (the unchanged tool: the index build) |

- **Where the unchanged tool peaked.** In C1L+E at the second BWT (215.2 at the first, 235.2 at the second), in E at both (194.0), and in A0 inside the index build too (175.3; M15 had placed A0's peak at the similar-context build).
- **The index build now** reaches 132.3 MB in E (it was the 194.0 peak) and 178.5 MB in C1L+E. After it the process drops to 68.4 MB (E), since the in-place build leaves no heap behind.
- **The lowered index's build** (C1L only, before the chat text exists) falls 204.3 → 184.6 MB and is now C1L+E's peak.

**Time.** The index build takes about 1 s more (the inverse BWT and the one-pass tree fill), and `text()` about 6 s once per run with the class table. Whole runs took the same or less (E 551 / 554 s, C1L+E 571 / 646 s, A0 458 / 486 s, against the unchanged tool), within the machine's noise.

#### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| M16a | The unchanged tool peaks inside the index build at the second BWT: C1L+E 225–240, E 185–200 MB | **Hit.** C1L+E 235.2 at the second BWT; E 194.0, reached at both BWTs |
| M16b | Harness: the new build's peak addition at least 20 MB lower | **Miss, by 0.4 MB:** 114.4 → 94.8 (−19.6) |
| M16c | The index byte-identical from both paths; `text()` exact; the CTest passes | **Hit** (golden hashes, `qa` stores, 191,909 checks) |
| M16d | C1L+E built 200 MB or less, E 180 or less, repeats within 2 MB | **Hit.** C1L+E 184.6 / 184.5; E 169.9 / 170.1 |
| M16e | Under 10 s added to a `qa` build; the index build at most 50% slower | **Hit.** About 7 s of added work (1 s in the index, 6 s `text()`); the index build +33% (2.7 → 3.6 s) |
| M16f | The lowered index's build in C1L falls by at least 20 MB | **Miss, by 0.3 MB:** 204.3 → 184.6 (−19.7) |

**Decision.** By the rule (lossless; E −24 MB, C1L+E −51 MB, A0 −43 MB, context8k −24 MB, far beyond the 0.2 MB repeat spread; time within 10%), M16 is adopted.
- A built `qa` model now peaks at 132 MB (A0), 170 MB (E) and 185 MB (C1L+E), against 175, 194 and 235. The 200 MB target for C1L+E is met.
- `FmInfiniGram::text()` gives a stored or built index's text back, so code that needs T can take it from the index instead of keeping a copy.

**Still open.**
- **`--save` keeps the lowered index** (26 MB) to the end of the run, so C1L+E with `--save` peaks at 204.5 MB. Freeing it once the store is written would take that under 200 (M14's rule; not done here).
- **The libsais scratch** (4 bytes a text byte) is now the index build's whole cost above its text and one tree. Lower would need a blockwise BWT with a merge; at 185 MB built and under 200 served, it is not needed now.

## M17: the lowered index freed once the store is written (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · follows M14 and M16's "Still open"

**Question.** With `qa --save`, M14 keeps the lowered index (A1c's FM index over the lowered training bytes, about 26 MB) resident, because the store writes it. It then stays to the end of the run, so C1L + defaults with `--save` peaks at 204.5 MB (M16), after the store is written. Can it be freed as soon as `save_qa` has written it?

**The change.** One line after `save_qa`: when `--save` is set, the token carries an entity statistic (`fam_k > 0`) and `--cards` is not set, free the lowered index. That is M14's own condition with `--save` now handled; M14's argument holds unchanged (every dev and test entity was chosen before and is kept in `chosen`, and nothing after the token step reads the index; `--cards` reads it, so it is kept there).

**Decision rule.**
- Lossless: the stores (every file), the rows file and stdout (timing masked) byte-identical to the unchanged tool, for C1L + defaults and for the defaults alone, both with `--save`.
- Adopted if lossless and C1L + defaults with `--save` falls by more than the 0.2 MB repeat spread; time within 10%.

**Predictions.**

| # | Prediction |
|---|---|
| M17a | C1L + defaults with `--save` peaks at 186 MB or less (unchanged tool: about 204.5); the new peak is the lowered index's build, as without `--save` (about 184.6) |
| M17b | The defaults alone with `--save` (no `--parse`, so no lowered index) are unchanged within 0.5 MB (about 170) |
| M17c | Every store file, the rows and stdout (timing masked) are byte-identical in both configurations |

### Results: M17 (2026-10-29)

Peaks are VmHWM in MB (kB / 1000); the unchanged tool is main aca60a5, the new one adds the one line. All runs with `--save`.

| Configuration | Unchanged tool | M17 | Time (s), unchanged / M17 |
|---|---:|---:|---|
| C1L + defaults | 204.5 | **186.7 / 186.7** (two runs) | 614 / 700, 578 |
| Defaults alone (no `--parse`: no lowered index) | 170.1 | 170.1 | 558 / 687 |

- **Lossless.** In both configurations every store file (`model.pk`, `ig.fm`, `low.fm`, `cls.cc`, `qa.meta`), the rows file and stdout (timing masked) are byte-identical to the unchanged tool's.
- **Time.** The machine was shared with other runs. The defaults alone, where the new line never runs, slowed by as much as C1L did on its first run (558 → 687 s), and the repeat C1L run was faster than the unchanged one (578 against 614 s). Freeing an index cannot add work, so these are load, not the change.
- **Where the peak is now.** 186.7 MB is about 2 MB above the same run without `--save` (184.6, M16), whose peak is the lowered index's build; the store's writing sits near it. The 26 MB the lowered index held after the store are gone.

#### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| M17a | C1L + defaults with `--save` 186 MB or less | **Miss, by 0.7 MB:** 186.7 (from 204.5, −17.8) |
| M17b | The defaults alone unchanged within 0.5 MB | **Hit:** 170.1 / 170.1 |
| M17c | Stores, rows and stdout byte-identical | **Hit**, both configurations |

**Decision.** Lossless, and C1L + defaults with `--save` falls 17.8 MB, far beyond the 0.2 MB repeat spread (two runs 0.05 MB apart): M17 is adopted. Every built and served `qa` configuration is now under 190 MB.

## BKN (2026-10-29)

Packed byte Kneser–Ney 6:4:2:1 is 13.8 MB on the wiki bytes and 24.3 MB on `qa`'s training text; its build alone peaks at 69 MB. Inside `qa` it adds 23.7 MB served and 99.8 MB to the built peak (186.7 → 286.5 MB), because the build ran while the index, class table and lowered index were resident, on a second copy of the text. BKN2 moves the build after the lowered index is freed ([report](BYTE_KN.md)).
