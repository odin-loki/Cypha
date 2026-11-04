# Online learning upgrades in the content mixer

2026-11-04 · Odin Loch

**Odin:** "How can we upgrade the online learning too? … Do all of it and make sure you document everything we have tried and the results. Do it all on my PC."

This report holds every arm, rule, prediction and result of this work, failures included. It was written as the work went: each part's arms, rules and predictions were registered here and committed before its runs.

## Setup

- Branch `wip/online-upgrades`, from `origin/main` at c7dd132.
- Odin's PC: Xeon Gold 6242 (32 cores, 64 threads), 47 GB RAM, WSL. Other sessions ran on the PC at the same time (a wide GLN, K5 work, a paq8px run, a VM), so the load was 16–30 throughout. Each `context8k` run took one core; arms ran side by side, so their wall times compare with each other, not with the cloud's.
- Build: `cmake -DCMAKE_BUILD_TYPE=Release` of `native/`, target `cyphalm_trie` (-O3).
- Data: `/mnt/d/cypha_llm/enwik8` and `/mnt/d/cypha_llm/corpus/labelled/dolly/databricks-dolly-15k.jsonl`, as R0/R1 of §13.12 used.
- Every run: `/usr/bin/time -v cyphalm_trie context8k ENWIK8 DOLLY [flags]`. Wall time and peak RSS come from `time -v`.
- The score is `online_final_bpc` (bits per byte on the 1 MiB wiki holdout's second half). The chain line is the last chain stage's line. Answers (`TALLY` lines) must match the baseline's exactly.

## What was built

Every change is a `context8k` flag. With no new flag, the output is the old default's (checked: see the results).

1. **`--ngram`: an online n-gram expert** (`native/include/cypha/online/online_ngram.hpp`, `OnlineNgram`).
   - PAQ-style. Each byte is 8 bit predictions, with the partial byte as context.
   - Each order k has a hashed table of bit counts: fixed point (×128, 16 bits), halved past a cap, as the bitwise GLN's counts.
   - Each slot has an 8-bit check. A slot held by another context reads as unseen and is taken over.
   - A small logistic mixer joins the orders' bit predictions in the stretch domain. Its weight set is chosen by the bit position and the longest order seen.
   - P(byte) is the product of the 8 mixed bit probabilities. Each order alone also gives a P(byte), the product of its own 8.
   - It pretrains its counts on wiki train (`data[0, kTrain − 1 MiB)`), then predicts and learns (counts and mixer) over wiki train's last MiB and the whole holdout. Each byte is predicted before it is learned.
   - In `context8k`: a chain stage before the LSTM (a lambda per InfiniGram bucket, learning online), and final-layer inputs (the mixed probability and each order's).
   - Flags: `--ngram-orders` (default `1,2,3,4,5,6,8,12`), `--ngram-tb` (table bits, default 22), `--ngram-cap` (60), `--ngram-lr` (0.005).
2. **`--doc-boost B --doc-span N`: document-aware steps.** The holdout is Wikipedia pages. For the first N bytes after each `<page>`, every online step is multiplied by B: the chain stages' exponentiated-gradient steps (main, slow and fast chains) and the final layer's step. Elsewhere the step is unchanged.
3. **`--lstm-size H --lstm-window T`: a bigger LSTM.** `ByteLSTM` is now `ByteLSTMT<64>`: the hidden size is a template parameter and the BPTT window a constructor argument. `ByteLSTM` (64, 20) is unchanged, so the GLN and every other user are unchanged. `context8k` takes H = 64, 128 or 256.

### Tuning the n-gram expert before registering (first half only)

A scratch harness ran the n-gram alone: counts on wiki train, online over its last MiB, scored on the holdout's **first half** only. The scored second half was not read. Bits per byte, first half:

| Orders | Table bits | Check | Cap | Mixer step | Bits per byte | MB |
|---|---:|---|---:|---:|---:|---:|
| 0–6 | 22 | no | 255 | 0.02 | 2.0904 | 85 |
| 0–6 | 22 | no | 127 | 0.02 | 2.0754 | 85 |
| 0–6 | 22 | no | 60 | 0.02 | 2.0550 | 85 |
| 0–6 | 22 | no | 30 | 0.02 | 2.0380 | 85 |
| 0–6 | 22 | no | 60 | 0.05 | 2.1263 | 85 |
| 0–6 | 22 | no | 60 | 0.005 | 2.0320 | 85 |
| 0–6, 8, 12, 16 | 22 | no | 60 | 0.02 | 2.0282 | 135 |
| 0–6 | 24 | no | 60 | 0.02 | 1.9515 | 337 |
| 0–6 | 22 | yes | 60 | 0.005 | 1.9116 | 106 |
| 0–6 | 22 | yes | 60 | 0.002 | 1.9103 | 106 |
| 0–6 | 22 | yes | 30 | 0.005 | 1.9075 | 106 |
| 0–6 | 23 | yes | 60 | 0.005 | 1.8758 | 211 |
| 0–6 | 24 | yes | 60 | 0.005 | 1.8561 | 421 |
| 1–6, 8 | 22 | yes | 60 | 0.005 | 1.8863 | 127 |
| 0–6, 8, 12 | 22 | yes | 60 | 0.005 | 1.8661 | 148 |
| **1–6, 8, 12** | **22** | **yes** | **60** | **0.005** | **1.8703** | **148** |
| 1–4, 6, 8, 12, 16 | 22 | yes | 60 | 0.005 | 1.8960 | 148 |
| 1–6, 8, 12, 16 | 22 | yes | 60 | 0.005 | 1.8683 | 169 |
| 1–6, 8, 12, 16, 24 | 22 | yes | 60 | 0.005 | 1.8685 | 190 |
| 1–6, 8, 12, 16 | 22 | yes | 30 | 0.003 | 1.8647 | 169 |
| 1–6, 8, 12, 16 (4 MiB online warm-up) | 22 | yes | 60 | 0.005 | 1.8683 | 169 |

- Hash collisions cost most. The 8-bit check gains 0.12 at the same size; bigger tables gain more but cost RAM.
- A lower cap (faster adaptation) and a smaller mixer step help a little.
- Orders past 12 add almost nothing. Order 0 adds almost nothing.
- The default chosen: orders 1–6, 8, 12; 22 table bits; check; cap 60; step 0.005; 1 MiB online warm-up. About 148 MB.
- For comparison on the same text: the LSTM (64) scores about 2.65 alone and the bitwise GLN 1.94.

### The LSTM's speed (scratch harness, 64 KiB of the holdout, this PC under load)

| Hidden | Window | Bits per byte (first 64 KiB) | Seconds per 64 KiB | Per MiB (estimate) |
|---:|---:|---:|---:|---:|
| 64 | 20 | 3.871 | 10.3 | 165 s |
| 128 | 50 | 4.020 | 25.7 | 410 s |
| 256 | 100 | 4.385 | 100.8 | 1,610 s |
| 256 | 20 | 3.484 | 111.0 | 1,780 s |

A longer window means fewer Adam steps (one per window), so it learns slower at first.

## Phase 1: registered before the runs

**Arms** (each `context8k ENWIK8 DOLLY` plus the flags; one core each, side by side):

| Arm | Flags |
|---|---|
| B0 | (none: this PC's baseline, the current default) |
| N1 | `--ngram` |
| N2 | `--ngram --ngram-tb 24` |
| D1 | `--doc-boost 3 --doc-span 1024` |
| D2 | `--doc-boost 2 --doc-span 4096` |
| L1 | `--lstm-size 128 --lstm-window 50` |
| L2 | `--lstm-size 256 --lstm-window 100` |
| L3 | `--lstm-size 256 --lstm-window 20` |

**Rules** (each read against B0's `online_final_bpc` on this PC; every arm's `TALLY` lines must equal B0's; `selftest` must pass):

1. **A cheap change** (wall time within +10% of B0, peak RSS within +10%) wins if it is at least 0.003 bits per byte below B0.
2. **A costly change** wins if it is at least 0.01 below B0, within its own limits:
   - the n-gram: wall within +25% of B0, peak RSS within +700 MB;
   - the bigger LSTM: wall within +100% of B0 (the bigger LSTM is slow by design, as in PAQ8 and cmix), RSS within +50 MB.
3. Among winners of the same kind, the lowest score is chosen, unless a cheaper one is within 0.003 of it. So N2 is chosen over N1 only if it is at least 0.003 below N1; L2 or L3 over L1 only if at least 0.003 below L1.
4. The doc-boost variants are cheap: D1 or D2 wins by rule 1; if both win, the lower one.
5. **Phase 2** (registered then, before its run) combines the winners in one run.

**Predictions:**
- N1: `ngram_alone_bpc` near 1.87. `online_final_bpc` −0.012 against B0 (between −0.005 and −0.020). It overlaps Kneser–Ney, the trie and InfiniGram, but it learns online and holds hashed counts as the GLN did, and the GLN gained −0.034 before the LSTM existed.
- N2: about 0.004 below N1.
- D1 and D2: within ±0.003 of B0. The doc cache already resets at each page, and exponentiated gradient already moves fast.
- L1: −0.003 to −0.008. L2: 0 to −0.006 (few updates). L3: −0.008 to −0.015 (the LSTM alone about 0.15 better).
- Every arm's `TALLY` lines equal B0's.
