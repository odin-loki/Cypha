# B4b: which next-byte experts become CyphaLM's defaults

**Odin Loch** · 2026-10-28 · PLAN B4 and B9 · follows [B4_CLASS_CONTEXT_EXPERT.md](B4_CLASS_CONTEXT_EXPERT.md) and [A1C_ENTITY_FAMILIARITY_AND_COPY.md](A1C_ENTITY_FAMILIARITY_AND_COPY.md)

(The decision rule and the predictions below were written before the runs. Results and the decision are added under them.)

**Question.** Two new experts each lower CyphaLM's BPC a lot: the class-context expert (B4: −0.078 on fresh text with shape classes, −0.091 with E1 tags) and the copy expert (A1c: −0.146 on the holdout's second half, chat model). Do they add up, which class table is best, and which of them become defaults?

## Decision rule (fixed before the runs)

An expert, or a setting of one, becomes a default only if all of these hold:
1. **BPC.** On fresh enwik8 text `[8388608, 9437184)`, read once, it lowers trie + InfiniGram BPC with LCI ≥ 0.95 (block bootstrap, 4 KiB blocks).
2. **Text.** Surface S on the 200-prompt protocol (B4) does not fall: the 95% interval of the change is not entirely below zero.
3. **Answers.** In the tool-profile grid, coverage at 2% wrong on dev does not fall, and the wrong rate's Wilson upper bound does not rise.
4. **Cost.** Inside the PLAN §10 budgets (whole language system ≤ 4 GB resident; Q/A p50 ≤ 200 ms on one core), with its build time and RAM recorded.

Settings (tail length, class table) are chosen on the protocol holdout; only the chosen ones are read on fresh text.

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| L1 | The class and copy experts stack. On fresh text, the gain of both together over trie + InfiniGram is 70–100% of the sum of their single gains: the copy expert works on repeats inside the document, the class expert on short within-word and word-class contexts |
| L2 | With both on, trie + InfiniGram on fresh text is 1.95–2.03 BPC (alone: 2.166) |
| L3 | S-1b's k-means word classes (256 or 1,024) beat shape classes by −0.00 to −0.02 BPC on fresh text, no more than the E1 tags do (−0.013) |
| L4 | Tags × word classes (one product table) beat the better single table by less than 0.005 BPC |
| L5 | The copy expert raises surface S less than the class expert does (S counts real words, and copying repeats words the document already has, which the loop term L can penalise) |

---

## Results, first read (run 2026-10-28)

Wiki-only `bpc` model (train bytes only). Class expert and copy expert as in B4 and A1c; mixing order trie → InfiniGram → class → copy, each lambda set fit over the distribution it is mixed into. Holdout: lambdas on the first half, read on the second. Fresh text `[8388608, 9437184)`: lambdas on the whole holdout, read once. Intervals: 4 KiB block bootstrap (2,000 / 4,000 resamples). Surface S: B4's protocol (200 prompts, trie base, no InfiniGram).

| Model | Holdout BPC | Fresh BPC | Fresh gain over trie + IG (95%) | Surface S | S change over trie (95%) |
|---|---:|---:|---|---:|---|
| trie | 2.278066 | 2.367826 | | 0.7612 | |
| trie + IG | 2.072500 | 2.166032 | | | |
| + class (shape:6) | 1.998603 | 2.087926 | −0.0781 [−0.0810, −0.0751] | 0.9052 | +0.1440 [+0.1341, +0.1538] |
| + class (E1 tags:6) | 1.982232 | 2.075081 | −0.0910 [−0.0939, −0.0879] | 0.8766 | +0.1154 [+0.1052, +0.1256] |
| + copy | 1.922956 | **1.934289** | −0.2317 [−0.2713, −0.1957] | 0.7086 | **−0.0526 [−0.0654, −0.0402]** |
| + class (shape) + copy | 1.859114 | 1.869297 | −0.2967 [−0.3363, −0.2613] | 0.8398 | +0.0786 [+0.0656, +0.0910] |
| + class (tags) + copy | 1.844656 | **1.857687** | −0.3083 [−0.3469, −0.2738] | 0.8475 | +0.0862 [+0.0742, +0.0977] |

- **They stack.** On fresh text, class (shape) + copy gains 96% of the sum of the single gains (−0.297 against −0.310); with tags, 95%.
- **Copy lowers surface S**, alone (−0.053) and on top of the class expert (−0.065 shape, −0.029 tags). Its samples are full of word fragments: *"Chan the [[salty, Alabama prote and othere early also a famous legal [[parland is fall internich's…"*. In generated text, short matches inside the model's own output are unreliable, while in real text they are mostly repeated words.
- **Copy on fresh text is worth far more than on the holdout's second half** (−0.232 against −0.150): the fresh megabyte has more repetition inside its articles.
- Peak RSS of the whole harness: 305 MB (shape), 341 MB (tags).

**Predictions:** L1 (stacking 70–100%) **hit**: 96% and 95%. L2 (both together 1.95–2.03 on fresh text) **miss**: 1.869 / 1.858, better than predicted, because copy's fresh gain was underestimated. L5 (copy raises S less than class) **miss**: copy lowers S.

**By the decision rule, copy as it is cannot be a default** (rule 2: S falls). The class expert with shape classes passes rules 1 and 2; rule 3 is read below. Tags pass rule 1 but fail rule 2 against shape classes (S −0.029, B4).

### Variant: copy only on long matches (registered before the run)

In real text, a long match inside the document is almost always a true repeat; in generated text, short matches are what produce fragments. So the copy expert is tried with its lambda forced to 0 below a minimum match length m (its other lambdas as fit).

| # | Prediction |
|---|---|
| L6 | With m = 8, copy keeps at least 70% of its fresh-text BPC gain on top of the class expert (shape), and surface S no longer falls (the interval of the change over class alone includes 0 or is above it) |
| L7 | With m = 12 or 16, S does not fall and at least 50% of the gain is kept; m = 6 still lowers S |

**Result (run 2026-10-28).** Class expert (shape:6) on, copy lambdas zeroed below m bytes; everything else as the first read.

| Copy minimum match m | Fresh BPC, trie + IG + class + copy | Copy gain over class (95%) | Share of the m = 0 gain | Surface S, trie + class + copy | S change over class (95%) | Rule 2 |
|---|---:|---|---:|---:|---|---|
| 0 (as A1c) | 1.869297 | −0.2186 [−0.2566, −0.1839] | 100% | 0.8398 | −0.0654 [−0.0763, −0.0550] | fail |
| 6 | 1.909310 | −0.1786 [−0.2144, −0.1467] | 82% | 0.8973 | −0.0080 [−0.0153, −0.0008] | fail |
| **8** | **1.937724** | **−0.1502 [−0.1842, −0.1199]** | **69%** | **0.9068** | **+0.0016 [−0.0035, +0.0069]** | **pass** |
| 12 | 1.976828 | −0.1111 [−0.1427, −0.0839] | 51% | 0.9052 | 0 (copy never fires) | pass |
| 16 | 2.003554 | −0.0844 [−0.1125, −0.0603] | 39% | 0.9052 | 0 | pass |

- **Why m ≥ 12 has no effect on generated text:** the no-repeat rule already bans continuing any 11-byte context that occurred before, so a copy of 12 or more bytes can never be sampled. On real text those long matches still pay (−0.111 at m = 12).
- **m = 8 is the largest gain that passes rule 2.** With the class expert, CyphaLM on fresh text goes from 2.166 (trie + IG) to **1.938** BPC, with surface S 0.907 (trie alone 0.761; human continuations 0.870).
- **Predictions:** L6 **miss** on the gain (69% against ≥ 70%), hit on S. L7: m = 6 still lowers S (hit), m = 12 keeps 51% with S unchanged (hit), m = 16 keeps 39% (miss).
- **Rule 3 (answers)** for class + copy (m = 8) is being read in `qa` (the answering-variants run, A1D/A2B). Note: in A2, the copy model (m = 0) lost both gated right answers (raw right 6, kept 0), so this read matters.

### Which class table (run 2026-10-28; `cyphalm_trie cls`, one run, fresh text read once)

The word classes are S-1b's k-means classes over Cypha's own learned vectors (50,000 words; [S1B report](S1B_LEARNED_CONTEXTS.md)). The product tables map a word to (E1 tag, k-means class) where both tables have it (20,832 words); other words fall to shape classes.

| Table, tail 6 | Contexts | Table | Fresh BPC, trie + IG + class | Change vs shape (95%) | Surface S | S change vs shape (95%) |
|---|---:|---:|---:|---|---:|---|
| shape only | 1,242,013 | 29 MB | 2.087926 | | **0.9052** | |
| E1 tags | 2,709,055 | 61 MB | **2.075081** | −0.0128 [−0.0147, −0.0110] | 0.8766 | −0.0286 [−0.0370, −0.0206] |
| k-means 256 | 5,628,017 | 124 MB | 2.092801 | +0.0049 [+0.0002, +0.0093] | 0.8263 | −0.0789 [−0.0893, −0.0682] |
| k-means 1,024 | 6,331,989 | 138 MB | 2.091095 | +0.0032 [−0.0022, +0.0081] | 0.8130 | −0.0922 [−0.1026, −0.0819] |
| tags × k-means 256 | 6,253,503 | 137 MB | 2.081371 | −0.0066 [−0.0116, −0.0018] | 0.8173 | −0.0880 [−0.0967, −0.0789] |
| tags × k-means 1,024 | 6,644,921 | 145 MB | 2.084709 | −0.0032 [−0.0083, +0.0016] | 0.8176 | −0.0876 [−0.0971, −0.0784] |

- **Word classes from vectors do not help the expert.** They are no better than shape classes on BPC, and they cost 8–9 points of surface S. Classes that group words by meaning spread each context's counts over words that do not share spelling patterns, so the expert's within-word predictions get worse while its word-to-word ones gain little.
- **Tags help BPC but cost S**; product tables are worse than tags alone on both.
- **Predictions:** L3 (k-means beat shape by 0.00–0.02) **miss**: they are worse or level. L4 (a product beats the better single table by under 0.005) **hit**: it does not beat it at all.
- **Decision (rules 1–2): the default table is shape classes only** (no table). Tags and word classes fail rule 2. This is also the tool-profile result for the tagger's and the word vectors' class-context adapters (see TOOL_PROFILE_P2_GRID).

### Variant: the copy expert's window (registered before the run)

The copy expert looks back 4,096 bytes. Articles are longer than that, so a longer window may find more true repeats. Tried with m = 8 and the class expert (shape:6), windows 4,096, 8,192 and 16,384 bytes (the search is linear in the window, so the cost grows with it).

| # | Prediction |
|---|---|
| L8 | A 16,384-byte window adds at most 0.02 BPC of gain over 4,096 on fresh text, and surface S does not change (generated texts are 400 bytes, inside any window) |

**Result (run 2026-10-28).** m = 8, class expert (shape:6) on.

| Copy window | Fresh BPC, trie + IG + class + copy | Copy gain over class (95%) | S change over class (95%) | Copy cost per byte: mean / p50 / p99 | One 80-byte answer |
|---|---:|---|---|---|---:|
| 4,096 | 1.937724 | −0.1502 [−0.1842, −0.1199] | +0.0016 [−0.0035, +0.0069] | 7.9 / 7.1 / 22.8 µs | 0.6 ms |
| 8,192 | 1.922851 | −0.1651 [−0.1995, −0.1345] | +0.0004 [−0.0047, +0.0054] | 16.2 / 14.1 / 42.9 µs | 1.3 ms |
| **16,384** | **1.912150** | **−0.1758 [−0.2108, −0.1449]** | −0.0007 [−0.0058, +0.0043] | 30.0 / 27.3 / 67.1 µs | 2.4 ms |

- The cost is the linear search over the window, timed on 200,000 fresh-text positions, one core. Even at 16,384 bytes an answer costs 2.4 ms, far inside the 200 ms budget (PLAN §10). In Q/A the prompt and answer fit inside 4,096 bytes, so the window changes nothing there.
- **Prediction L8: miss** on the gain (−0.026 over 4,096, more than the 0.02 predicted); hit on S.
- **Chosen: 16,384 bytes, m = 8.** CyphaLM with the class expert (shape) and this copy expert reads fresh text at **1.912** BPC (trie + InfiniGram: 2.166), with surface S unchanged by the copy expert.

### Variant: a third word order in the class expert (registered before the run)

`class_context.hpp` gains an option `orders = 3`: a third context, (c3, c2, c1, tail), above the two it had. With the default `orders = 2`, counts and hashes are unchanged (the parity line below checks it).

| # | Prediction |
|---|---|
| L9 | With shape classes, the third order adds −0.005 to −0.02 BPC on fresh text over two orders, and surface S moves by less than ±0.01 |

**Result (run 2026-10-28; fresh text read once; surface S on B4's protocol).**

| Shape classes, tail 6 | Contexts | Table | Fresh BPC, trie + IG + class | Change (95%) | Surface S | S change (95%) |
|---|---:|---:|---:|---|---:|---|
| 2 orders (default) | 1,242,013 | 29 MB | 2.087926 (parity: equal to B4) | | 0.9052 (parity) | |
| 3 orders | 2,160,001 | 50 MB | 2.084349 | −0.0036 [−0.0051, −0.0022] | 0.9097 | +0.0045 [−0.0026, +0.0115] |

- **Prediction L9: miss** on the size of the gain (−0.0036, below the −0.005 to −0.02 predicted); hit on S.
- The third order passes rules 1 and 2 at +21 MB of table and a slightly longer build. It goes into the final combined read (rule 3, answers) with the copy expert, and becomes the default only if that read holds.

### SCRIBE on the new experts' text (reported only; run 2026-10-28)

Surface S saturates: it scores the experts' word salad at or above human text. SCRIBE v1 has not passed its own ladder, so it decides nothing (PLAN §6.4), but it sees the word salad. Its fresh ladder (SCRIBE v1 protocol: 250 prompts × T 0.6 / 0.8 / 1.0, 600 bytes, cleaned; human text from [64, 72) MiB), run on three sample files written exactly as `cyphalm_trie sample` writes them. The trie-only file is byte-identical to `sample`'s.

| Text | n | SCRIBE [95%] | Surface S (this ladder's cleaning) |
|---|---:|---|---:|
| held-out human | 500 | 0.535 [0.517, 0.553] | 0.909 |
| sentences shuffled across documents | 500 | 0.230 [0.220, 0.241] | 0.899 |
| words shuffled within sentences | 500 | 0.111 [0.104, 0.118] | 0.920 |
| trie | 623 | 0.028 [0.025, 0.032] | 0.734 |
| trie + class (shape, tail 6, 3 orders) | 647 | **0.074 [0.068, 0.081]** | 0.857 |
| trie + class + copy (m = 8, 16,384) | 641 | 0.072 [0.066, 0.080] | 0.843 |

- **The class expert lifts SCRIBE 2.6 times** (0.028 → 0.074), with disjoint intervals. The copy expert changes nothing on it.
- **It is still far from human text** (0.535), and below shuffled human words (0.111). The text is better word salad, not sense.

### Rule 3, answers, and the decision (2026-10-28)

Rule 3 was read in `qa` by the answering-variants runs ([A1D/A2B](A1D_A2B_ANSWERING_VARIANTS.md), step 6). Dev, chat form, 5-fold CV gate at 2% wrong:

| Configuration | Without the class expert: raw right; kept right / wrong | With it | Rule 3 |
|---|---|---|---|
| No fact shard: trie + IG (+ class) | 2; 0 / 0 | 2; 0 / 0 | holds |
| No fact shard: + copy (with the class expert m 8; the reference is A1c's copy, m 0) | 1; 0 / 2 | 1; 0 / 1 | holds |
| D4 fact shard: trie + IG (+ class) | 7; 2 / 0 | 6; 2 / 1 | **fails**: a wrong answer gets through |
| D4 fact shard: + copy (m 8) | 6; 0 / 0 | 2; 0 / 0 | **fails** on raw answers: the model echoes the question (*"Avicenna born?"*) |
| C1L's setting (token, shard): + copy, shape / tags | 6 raw right | 4 / 5 | fails on raw answers |

**Decision: the defaults do not change.** CyphaLM's default stays trie + InfiniGram. The class expert (shape classes, tail 6, 3 orders) and the copy expert (m = 8, window 16,384) pass rules 1 and 2 and cost little after the RAM work (the class table is 18 MB compact with contexts seen once dropped; RAM_PROFILE M2/M3). But with the fact shard, which is how CyphaLM answers anything today, they make it echo the question or let a wrong answer through, so they fail rule 3.

- **They stay options:** `cls`, `qa --cls`, `qa --copy --copy-min 8`. For text alone they are the best measured setting: fresh text 2.166 → 1.912 BPC, surface S 0.907, SCRIBE 0.028 → 0.072.
- **What would change this:** a fix for the echo (a copy weight that depends on where it is, inside the answer or not; flagged for Odin in A1D/A2B), then rule 3 read again.
- **The similar-context expert (A4) is in the same place.** It passes rules 1, 2 and 4 (fresh text −0.061 over trie + IG + class, S unchanged), and rule 3 without the shard, but with the shard it lets look-alike wrong answers through or breaks the shard's answers. It stays an option ([A4 report](A4_SIMILAR_CONTEXT_EXPERT.md)).
