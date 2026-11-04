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

## Phase 1: first results (B0, N1, N2, D1, D2, L1; L2 and L3 were still running)

| Arm | `online_final_bpc` | Against B0 | Chain line | Alone | Wall | Peak RSS | Answers |
|---|---:|---:|---:|---:|---:|---:|---|
| B0 | 1.6232 | — | 1.6570 | LSTM 2.6526 | 55:27 | 255 MB | — |
| N1 `--ngram` | 1.5986 | −0.0246 | 1.6296 | n-gram 1.8281 | 56:02 (+1%) | 404 MB (+149) | match |
| N2 `--ngram --ngram-tb 24` | **1.5913** | **−0.0319** | 1.6188 | n-gram 1.7555 | 56:09 (+1%) | 825 MB (+570) | match |
| D1 `--doc-boost 3 --doc-span 1024` | 1.6246 | +0.0014 | 1.6570 | — | 54:54 | 256 MB | match |
| D2 `--doc-boost 2 --doc-span 4096` | 1.6239 | +0.0007 | 1.6569 | — | 55:11 | 257 MB | match |
| L1 `--lstm-size 128 --lstm-window 50` | 1.6132 | −0.0100 | 1.6492 | LSTM 2.4568 | 57:42 (+4%) | 244 MB | match |

"Answers" compares every `GEN` and `TALLY` line with B0's. The doc boost ran over 159 page starts in the holdout.

- B0 on this PC gives 1.6232 (the cloud gave 1.6238 for the same build).
- **The n-gram wins.** N1 is −0.0246 (0.01 needed), wall +1%, RSS +149 MB. N2 is −0.0319, 0.0073 below N1 (0.003 needed), wall +1%, RSS +570 MB (700 allowed). By rule 3, **N2 is chosen**. The n-gram costs about a minute (counts and online pass), not more.
- **The doc boost loses.** Both variants are slightly worse. Faster steps at a page start add noise more than they adapt. The doc cache already resets at each page.
- **L1 is exactly at its threshold:** −0.0100 printed to 4 decimals, with 0.01 needed. It is held as "not decided" until L2 and L3 are read: if one of them wins it is chosen anyway; if neither wins, L1 is read at the printed precision (−0.0100 meets "at least 0.01").
- Predictions: N1 (−0.012, range −0.005 to −0.020) missed: the gain is twice as large. N2 (0.004 below N1) missed: 0.0073. D1 and D2 within ±0.003 held. L1 (−0.003 to −0.008) missed: −0.010.

## Phase 2: registered before its runs

L2 and L3 were still running when this was written, so the combined arm is launched for each possible LSTM outcome at once. Only the one the phase 1 rules pick is read as **C**; the others are reported as extras.

| Arm | Flags | Read as C when |
|---|---|---|
| C-L1 | `--ngram --ngram-tb 24 --lstm-size 128 --lstm-window 50` | L1 is the LSTM chosen |
| C-L3 | `--ngram --ngram-tb 24 --lstm-size 256 --lstm-window 20` | L3 is the LSTM chosen |
| C-L2 | `--ngram --ngram-tb 24 --lstm-size 256 --lstm-window 100` | L2 is the LSTM chosen |
| L4 | `--lstm-size 128 --lstm-window 20` | an extra arm (not eligible for a default; it only shows whether the window or the size matters) |

**Rules for the defaults:**
1. C becomes `context8k`'s default if its `online_final_bpc` is at least 0.003 below N2's, and its answers match B0's. Otherwise N2 alone becomes the default (`--ngram --ngram-tb 24`) and the LSTM stays 64/20.
2. The new default is built in (no flag needed) and run once with no flags. It must reproduce C's (or N2's) output line for line. `--no-ngram` and `--lstm-size 64 --lstm-window 20` then restore B0's lines, and that run must reproduce B0 line for line.
3. Doc boost stays a flag, off.

**Predictions:** C is about the sum of its parts less a little: N2 plus 70% of the LSTM's gain (the n-gram and the bigger LSTM both add context the chain lacked, and they overlap). C-L1 near 1.584. L4 between L1 and B0 (window 20 learns faster but the model is small).

## Phase 1: L2 and L3

| Arm | `online_final_bpc` | Against B0 | Chain line | LSTM alone | LSTM pass | Wall | Peak RSS | Answers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| L1 `--lstm-size 128 --lstm-window 50` | 1.6132 | −0.0100 | 1.6492 | 2.4568 | 380 s | 57:42 (+4%) | 244 MB | match |
| L2 `--lstm-size 256 --lstm-window 100` | 1.6097 | −0.0135 | 1.6447 | 2.3262 | 1,194 s | 1:11:51 (+30%) | 255 MB | match |
| L3 `--lstm-size 256 --lstm-window 20` | **1.6026** | **−0.0206** | 1.6384 | 2.2755 | 1,402 s | 1:14:59 (+35%) | 253 MB | match |

The 64/20 LSTM's pass took 153 s in the same runs.

- **L3 wins.** It is −0.0206 (0.01 needed), wall +35% (100% allowed), RSS unchanged. It is 0.0106 below L1 and 0.0071 below L2 (0.003 needed), so by rule 3 **L3 is chosen**. L1's tie no longer matters.
- **The window matters as much as the size.** At hidden 256, a window of 20 beats a window of 100 by 0.007: with one Adam step per window, the long window learns from five times fewer steps on 1 MiB. cmix's horizon of 100 suits far more text than this.
- Predictions: L2 (0 to −0.006) missed: −0.0135. L3 (−0.008 to −0.015) missed: −0.0206. The bigger LSTM is worth more than predicted.

## Phase 2: results

All runs started together. The PC's load fell during these runs, so their wall times are not comparable with phase 1's. VB and V0 (the old and the new default, same binary, started together) give the fair time comparison.

| Arm | `online_final_bpc` | Against B0 | Chain line | n-gram alone | LSTM alone | Wall | Peak RSS | Answers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C-L1 (N2 + LSTM 128/50) | 1.5860 | −0.0372 | 1.6161 | 1.7555 | 2.4568 | 36:01 | 706 MB | match |
| C-L2 (N2 + LSTM 256/100) | 1.5848 | −0.0384 | 1.6141 | 1.7555 | 2.3262 | 46:29 | 706 MB | match |
| **C-L3 (N2 + LSTM 256/20) = C** | **1.5793** | **−0.0439** | **1.6103** | 1.7555 | 2.2755 | 48:32 | 707 MB | match |
| L4 `--lstm-size 128 --lstm-window 20` (extra) | 1.6118 | −0.0114 | 1.6477 | — | 2.4221 | 35:46 | 249 MB | match |

- **C passes.** C (C-L3) is 1.5793: 0.0120 below N2's 1.5913 (0.003 needed). Its answers match B0's. **By rule 1, the n-gram (24 table bits) and the LSTM at 256/20 become `context8k`'s defaults.**
- The parts add up almost fully: N2 alone gave −0.0319, L3 alone −0.0206; together −0.0439, which is 85% of the sum (−0.0525).
- The prediction for C (about N2 plus 70% of the LSTM's gain, so near 1.577) held roughly: 1.5793. C-L1 predicted near 1.584: 1.5860. L4 between L1 and B0 missed: at 1.6118 it is slightly better than L1 (1.6132), so at hidden 128 the shorter window also helps.

## The new default, verified

`context8k` with no new flag now runs the n-gram (`--ngram-tb 24`, orders 1–6, 8, 12) and the LSTM at 256/20. `--no-ngram` and `--lstm-size 64` restore the old default. `bench/fuzzy/glnx_prep.sh` now passes `--no-ngram` too, so the dump keeps the frozen chain's columns.

| Run | Flags | `online_final_bpc` | Wall | Peak RSS | Output |
|---|---|---:|---:|---:|---|
| V0 | (none: the new default) | 1.5793 (chain 1.6103) | 45:13 | 828 MB | **byte for byte C-L3's** |
| VB | `--no-ngram --lstm-size 64` | 1.6232 (chain 1.6570) | 28:02 | 247 MB | **byte for byte B0's** |

- Both outputs were compared whole with `cmp`, every `GEN` and `TALLY` line included.
- **The cost, honestly:** in the same conditions the new default takes **+61% wall time** (45:13 against 28:02) and about **+580 MB peak RSS** (828 MB against 247 MB). Almost all the time is the 256 LSTM (about 1,100–1,400 s for 1 MiB, against 153 s at 64). Almost all the memory is the n-gram's tables (589 MB at 24 bits). Peak RSS of the same configuration varied between runs (707 MB for C-L3, 828 MB for V0), likely from when the tables' pages were touched under memory pressure.
- `cyphalm_trie selftest` passes on the new build.
- **The GLN and the old LSTM are unchanged:** `gln-train` (tb 18) and `lstm-train` on the same 400 KB of text wrote byte-identical model files with the old binary and the new one.

## All arms against this PC's baseline

| Arm | Flags | `online_final_bpc` | Against B0 | Chain line | Wall | Peak RSS | Answers match B0 | Verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| B0 | (old default) | 1.6232 | — | 1.6570 | 55:27 | 255 MB | — | baseline |
| N1 | `--ngram` | 1.5986 | −0.0246 | 1.6296 | 56:02 | 404 MB | yes | wins; N2 chosen over it |
| N2 | `--ngram --ngram-tb 24` | 1.5913 | −0.0319 | 1.6188 | 56:09 | 825 MB | yes | **wins** |
| D1 | `--doc-boost 3 --doc-span 1024` | 1.6246 | +0.0014 | 1.6570 | 54:54 | 256 MB | yes | loses |
| D2 | `--doc-boost 2 --doc-span 4096` | 1.6239 | +0.0007 | 1.6569 | 55:11 | 257 MB | yes | loses |
| L1 | `--lstm-size 128 --lstm-window 50` | 1.6132 | −0.0100 | 1.6492 | 57:42 | 244 MB | yes | at threshold; L3 chosen |
| L2 | `--lstm-size 256 --lstm-window 100` | 1.6097 | −0.0135 | 1.6447 | 1:11:51 | 255 MB | yes | wins; L3 chosen |
| L3 | `--lstm-size 256 --lstm-window 20` | 1.6026 | −0.0206 | 1.6384 | 1:14:59 | 253 MB | yes | **wins** |
| L4 | `--lstm-size 128 --lstm-window 20` | 1.6118 | −0.0114 | 1.6477 | 35:46\* | 249 MB | yes | extra |
| C-L1 | N2 + LSTM 128/50 | 1.5860 | −0.0372 | 1.6161 | 36:01\* | 706 MB | yes | extra |
| C-L2 | N2 + LSTM 256/100 | 1.5848 | −0.0384 | 1.6141 | 46:29\* | 706 MB | yes | extra |
| C-L3 | N2 + LSTM 256/20 | **1.5793** | **−0.0439** | 1.6103 | 48:32\* | 707 MB | yes | **the new default** |
| V0 | (new default) | 1.5793 | −0.0439 | 1.6103 | 45:13\* | 828 MB | yes | = C-L3 byte for byte |
| VB | `--no-ngram --lstm-size 64` | 1.6232 | 0 | 1.6570 | 28:02\* | 247 MB | yes | = B0 byte for byte |

\* Phase 2, at a lower load than phase 1. Compare V0 with VB for the time cost (+61%).

On this PC the frozen chain before §13.15 gave 1.6971 (R0). The default went from 1.6232 to 1.5793 bits per byte with these upgrades, with no GLN.

## What did not work

- **Document-aware steps** (D1, D2): both slightly worse (+0.0014, +0.0007). Faster learning at a page start adds noise. The doc cache already resets at each page, and exponentiated gradient already moves fast. A reset of the final layer's weights at each page was not tried: a weaker version of the same idea (more steps) already lost.
- **A long BPTT window** (100 or 50): worse than 20 at the same size, as one Adam step per window gives fewer steps.
- In tuning (first half only): no slot check (0.12 worse alone), a cap of 255 (slow to adapt), a mixer step of 0.05, orders past 12.

## Risks before merging

- **Time and memory.** The default `context8k` now takes about +61% wall time and about +580 MB. Anything that runs `context8k` many times (`scripts/cypha_report_card.sh`, the GLNX dump) pays it. `--lstm-size 64` saves most of the time (L4 shows 128/20 keeps half the LSTM gain at a quarter of the cost); `--ngram-tb 22` saves 440 MB for 0.007.
- **Merge conflicts.** Another session adds a whole-distribution mixer to `context8k` at the same time. This branch touches `cmd_context8k` (the doc-boost lines after `g_online_mix`; the n-gram stage and the LSTM size before and in the LSTM stage; the final layer's inputs, `q[48]` and `fin.update`), `content_mixer.hpp` (`g_step_mult`, `eg_stage`, `OnlineFinal::update`'s optional step factor), and `byte_lstm.hpp` (`ByteLSTM` is now `ByteLSTMT<64>`).
- **The final layer's input count.** With the n-gram the final layer has 9 more inputs (35 in all). A new mixer that reads the same inputs must size its arrays for that.
- **Only `context8k` changed.** `sample`, `qa` and answers do not read the n-gram or the bigger LSTM. Whether they help writing is untested.
- **One holdout.** Every number is the same 1 MiB of enwik8 (its second half). The n-gram's settings were tuned on the first half of the same holdout, not on other text.
- **Wall times on a shared PC.** Other sessions loaded the PC throughout (load 16–117), so wall times are noisy. The same-run comparisons (V0 against VB) are the fair ones.

# Optimisation

2026-11-05 · Odin Loch

**Odin:** "See if you can't optimise all those upgrades before going in."

The aim: make the n-gram and the LSTM cheaper with **byte-for-byte identical output**. Branch `wip/opt-online`, from `main` at 20701dd. Only `byte_lstm.hpp` and `online_ngram.hpp` change, plus a selftest check and two CMake lines (`cyphalm_trie` links `Threads::Threads`).

## Where the time went (main, before any change)

No `perf` on this WSL. gprof missed most of the time (it cannot see libm or memory stalls). So the phases were timed with clocks in a scratch copy of the header, over 8 KiB of the holdout, one core, PC quiet.

| LSTM 256/20, main | Seconds per 8 KiB | Share |
|---|---:|---:|
| Forward (1,024 gate rows × 272) | 2.65 | 38% |
| Backward (softmax again, output and gate gradients, input error) | 2.64 | 38% |
| Adam (once per 20 bytes) | 1.11 | 16% |
| `dist` (256 output rows × 256) | 0.58 | 8% |
| **Total** | **7.0** | |

- Each dot product is one long chain of dependent adds, so the forward pass waits on add latency.
- The backward pass streams the 2.2 MB gate weights and their 2.2 MB gradient once per byte.
- Adam did not vectorise: its square root needs `-fno-math-errno`, and the vector-of-vectors indexing hides that the arrays don't overlap.
- The LSTM 64 (the GLN's, and `--lstm-size 64`): 0.77 s per 8 KiB.

| n-gram (orders 1–6, 8, 12; 24 table bits), main | Seconds |
|---|---:|
| Pretrain (counts over 6 MiB) | 17.5 |
| Online over 1 MiB + 64 KiB, with `dist` on every 4th byte | 7.35 |

Each lookup touched three arrays (count of ones, count, check), so three cache misses per order per bit.

## What changed (all exact)

**The LSTM:**
1. **Dots across rows.** `lstm_rows` puts two rows in one SSE2 register, one lane per row, eight rows at a time. Each row is summed in its own order (j = 0, 1, …). There is no FMA (the default build has no `-march`) and no reassociation, so every row's value is the scalar loop's. It is used for the gates and the output logits, on the row-major weights, with no transposed copy.
2. **Gradients once per window.** The backward pass keeps each step's errors (`da`, `p`). After the window, each weight's gradient is summed over the steps in the same order as before (last step first) and Adam is applied at once. The gradient arrays are no longer streamed every byte.
3. **Adam vectorised.** It works on plain arrays with `__restrict` and SSE2 (`_mm_div_pd`, `_mm_sqrt_pd`). IEEE division and square root give the same result in SIMD as in scalar code, and `std::clamp` is written out with the same comparisons. Scalar code is used without SSE2.
4. **No recomputation.** The softmax `dist` computes is reused by BPTT when it is for the same state (not after `restore`), and tanh(c) is kept from the forward pass.
5. **Threads.** A small spinning pool (`LstmPool`): 4 threads by default from hidden 128, 1 below (so the GLN's and `sample`'s LSTM 64 run on one), and `CYPHA_LSTM_THREADS` or `set_threads` overrides. The gate rows, the output rows, the update's rows and the `dh` units are split across threads. Each row is computed by one thread in its original order, so the output is the same for any count.
   - The input error (`dxh`, a sum over all 1,024 rows for each of 272 columns) stays on one thread. A column split and an exact row-order wavefront were both measured slower than one thread on this PC: every thread then reads all of `W`, or waits on the others.

**The n-gram:**
1. **One 5-byte slot** holds the count of ones, the count and the check (packed). One cache line per lookup instead of three, the same memory (`bytes()` unchanged).
2. **Prefetch.** All 8 bits' keys of a byte are known before the counts are read (the byte is given), so they are computed and prefetched first. `dist` prefetches 8 partial bytes ahead.
3. **Pretrain on threads by order.** The orders' tables are independent: each thread fills its own tables in the original order (`CYPHA_NGRAM_THREADS`, 4 by default).

## Exactness checks

- **Scratch harness, old header against new**, a hash of every `dist` output and `code` value:
  - LSTM 64/20, 128/50, 128/20 and 256/20, on 1, 2, 3, 4 and 8 threads;
  - `dist` called on every byte, every 3rd or every 7th byte;
  - the writer's save, step without learning, `dist` and restore, every 997 bytes.
  - All hashes equal.
- **The n-gram:** 24, 20 and 22 table bits, check on and off, `dist` every 4th byte, reads without learning. All hashes equal, and `bytes()` is equal.
- **`gln-train`** (tb 18) and **`lstm-train`** on 400 KB of enwik8 write byte-identical files with main's binary and the new one. `gln-train` also runs faster (48.5 s → 32.9 s), as the GLN holds an LSTM 64.
- **`cyphalm_trie selftest`** passes. It now also checks an LSTM 128's hash over 400 bytes of learning, on 1 and on 4 threads, against main's value.

## Micro-benchmarks (8 KiB of the holdout; the PC's load varied from 3 to 13, so ±20%)

| LSTM | main | New, 1 thread | New, 4 threads |
|---|---:|---:|---:|
| 256/20 | 7.0–7.2 s | 3.7 s | 1.7 s |
| 128/20 | 2.1 s | — | 0.50 s |
| 64/20 | 0.77 s | 0.41 s | (1 thread by default) |

| n-gram, 24 table bits | main | New |
|---|---:|---:|
| Pretrain | 17.5 s | 2.9 s (6.6 s on 1 thread) |
| Online 1 MiB + 64 KiB, with `dist` | 7.35 s | 3.16 s |

## A non-exact arm (registered before its run)

The memory is now the main cost: the n-gram's 589 MB cannot shrink without changing the numbers.
- **O1:** `--ngram-tb 23` (the new binary): about 294 MB less.
- **Rule:** adopt only if `online_final_bpc` is at most 0.001 above the default's (1.5793) and the saving is at least 200 MB.
- **Prediction:** +0.003 to +0.004 (24 bits against 22 gave −0.0073), so it fails. A float32 LSTM was not tried: the exact changes already cut the LSTM's time by 4×.
