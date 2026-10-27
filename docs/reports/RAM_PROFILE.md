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
