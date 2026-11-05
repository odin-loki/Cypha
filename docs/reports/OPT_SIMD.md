# O5: the byte LSTM on AVX-512 and the n-gram's lookups (GENERATION_PLAN §13.41)

2026-11-06 · Odin Loch

**Odin:** "Keep looking. Come up with more. Try optimisation techniques … Go crazy on it."

The job (§13.41, O5): make the byte LSTM and the online n-gram expert faster, with exactly the same output. Branch `wip/opt-simd`. Work began on 2026-11-05 from main 6530cb2 and stopped on Odin's word (§13.44). It resumed on 2026-11-06 (§13.45): the branch first merged main c10935b (M1 and M2, both exact). Only two files change: `native/include/cypha/online/byte_lstm.hpp` and `native/include/cypha/online/online_ngram.hpp`.

Odin's PC: Xeon Gold 6242 (AVX-512), WSL, GCC 13.3, the Release build (-O3, no `-march`). Other sessions (Q1–Q3, T1) ran on the PC throughout; the load was 10–31.

## What changed

### The LSTM (`byte_lstm.hpp`)

- **Lanes across outputs.** With AVX-512 every matrix product runs across its outputs, eight lanes at a time. Each lane holds one output's running sum and adds its terms in the old order (j = 0, 1, …). So each output is the scalar loop's value, bit for bit.
- **No fused multiply-add.** Each product goes through an empty `asm` before its add, so the compiler cannot fuse the two, whatever the flags. The new binary has no FMA instruction (`objdump`: 0).
- **Weights laid out for the kernels.** The gates read a copy of W in tiles of 8 units (4 gates × 8 lanes per column). The logits read a copy of Wo in tiles of 32 outputs. BPTT's sums down the columns (dh, and the input's error dxh) read copies in blocks of 8 columns. After each Adam step the changed rows are copied again. These are plain copies of the values.
- **BPTT as one job.** Each thread owns some hidden units (their dh and gate errors) and the matching columns of dxh. A step needs one barrier. Before, dxh ran on one thread.
- **Gradients four rows at a time.** Each weight row's gradient over the window is still summed last step first, as before; four rows share each column load.
- **Adam eight at a time.** Division and square root are IEEE in any width. Adam skips dividing by its bias corrections once they are exactly 1 (x / 1 = x).
- **The window's steps live in place.** Before, each step was copied twice.
- **Chosen at run time:** the CPU (and OS) has AVX-512F, and `CYPHA_LSTM_AVX512` is not 0. Otherwise the SSE2 and scalar code runs, as before.
- Memory: the copies add 5.5 MB for the LSTM 256.

### The n-gram (`online_ngram.hpp`)

The same hashing, the same slots, the same replacement. Less work per lookup:
- `dist()` computes each key once. Before, it hashed twice: once to prefetch, once to read.
- The parts of the slot hash that do not depend on the partial byte are multiplied once a byte. XOR is associative, so the key is the same.
- The orders' context hashes are kept until the history moves.
- A slot's probability and stretch, for counts that are whole multiples of 128 (a slot never halved), come from a table made at the start by the same expressions.
- `dist()` forms the 256 byte products down the bit tree. Shared prefixes are multiplied once; each byte's product is still 1 × bit 0's × bit 1's × …, in that order.

**Tried and not kept** (each exact; timings in "The experts alone"):
- **2 MB pages** (`madvise(MADV_HUGEPAGE)` on the tables). Counting got 8–20% faster. But this WSL compacts memory to make huge pages when a program asks for them. On the busy PC, making the tables then took 1.7–15 s, against 0.4–2.5 s without. The small `context8k` counts nothing before its online pass, so it would only lose. The tables stay plain vectors, as on main.
- **Prefetching one byte ahead** (§13.41's wording). In `pretrain` the text is known, so byte i + 1's slots were prefetched while byte i was counted. `code()` ended by prefetching the next context's first slot. Neither changed the time. Each byte already prefetches its 8 × 8 slots before it reads them, which keeps the memory system busy. And the next call comes at once, so there is no gap to hide a load in.

## Exactness checks

Old: main c10935b, built from a clean copy (`git archive`). New: this branch (main merged in). Same compiler and flags. Every `context8k` pair was started together, with the same threads. Exact means `cmp` of the whole stdout (every `GEN` and `TALLY` line included).

The small setup is `cyphalm_trie context8k /mnt/d/cypha_llm/enwik8 /mnt/d/cypha_llm/corpus/labelled/dolly/databricks-dolly-15k.jsonl --train 1048576 --sample 1179648`.

| Check | Threads (LSTM, n-gram) | What was compared | Result |
|---|---|---|---|
| small, default | default (4, 4) | stdout, 260 lines; `context8k_bpc 1.7501` | `cmp` equal |
| small, `--no-meta-mix` | default | stdout, 245 lines; 1.7633 | `cmp` equal |
| small, `--no-full-mix` | default | stdout, 233 lines; 1.8384 | `cmp` equal |
| small, `--no-ngram --lstm-size 64` | default (1) | stdout, 258 lines; 1.7849 | `cmp` equal |
| small, default | 1, 1 | stdout, 260 lines | `cmp` equal |
| small, default, new with `CYPHA_LSTM_AVX512=0` (the fallback) | default | stdout against the old run's | `cmp` equal |
| small, `--no-full-mix`, new with `CYPHA_LSTM_AVX512=0` | 1, 1 | stdout against the old run's (4 threads) | `cmp` equal |
| `lstm-train` on 1 MiB of K5's text (`fuzzy/train.txt`) | 1 | the LST1 file (996,480 bytes); bpc 2.76178 | `cmp` equal |
| `gln-train` on 1.2 MB of enwik8, tb 18 (the GLN holds an LSTM 64) | 1 | the model file (7,730,304 bytes); bpc 2.05661 | `cmp` equal |
| `sample --k5 fuzzy/train.txt --no-k5-plan --k5-gate 1 --k5-lstm L_old.lst`, 20 dev prompts, `K5_THREADS=2` (the LSTM's save, restore and steps without learning) | 1 | GEN file (41 lines), stdout | `cmp` equal |
| `cyphalm_trie selftest` (it hashes an LSTM 128 over 400 bytes on 1 and 4 threads against main's value) | — | pass, old and new, also with `CYPHA_LSTM_AVX512=0`; output equal but its own timing line | pass |
| The experts alone (the harness below) | 1 and 4 | a hash of every number they return: n-gram (pretrain, `code`, `dist`); LSTM 64, 128, 256 on 1 and 4 threads, AVX-512 on and off | all hashes equal |
| **full `context8k`, default** (old, then new, each in its own turn of `flock ~/.cypha_full.lock`) | default | stdout, 260 lines; **`context8k_bpc 1.4962`**, the default's | `cmp` equal |

**O5 is exact.** The new binary has no FMA instruction. Without AVX-512 it gives the same numbers too.

## Stage times (small setup, old and new started together)

The LSTM and the n-gram each run on their own thread inside the full mix, and the full mix prints how long each thread was busy. With `--no-full-mix` each expert prints its own seconds (the n-gram's include making its tables and its 1 MiB warm-up). Other sessions loaded the PC (load 13–31).

| Run | Threads | LSTM stage, old → new | n-gram stage, old → new | Wall, old → new | Peak RSS, old → new |
|---|---|---:|---:|---:|---:|
| default | default (4, 4) | 52.4 → 23.1 s (**−56%**) | 10.2 → 5.6 s (**−45%**) | 513 → 510 s | 1,303 → 1,309 MB |
| `--no-meta-mix` | default | 45.0 → 17.4 s (−61%) | 9.8 → 5.1 s (−48%) | 521 → 517 s | 905 → 908 MB |
| `--no-full-mix` | default | 33.1 → 14.2 s (−57%) | 3.43 → 3.30 s (−4%) | 412 → 397 s | 671 → 672 MB |
| `--no-ngram --lstm-size 64` | default (1) | 10.3 → 6.0 s (−42%) | — | 445 → 448 s | 668 → 668 MB |
| default | 1, 1 | 95.3 → 80.9 s (−15%) | 9.7 → 5.1 s (−47%) | 556 → 551 s | 1,303 → 1,307 MB |

- **The wall time hardly moves.** The LSTM is not what the run waits for. In the small default run the full mix's main thread (about 80 s of mixing) is the slowest part. The full mix's own time stayed 81.4 → 81.5 s. Only where the LSTM was the slowest producer did the mix finish sooner: `--no-meta-mix`'s full mix took 47.2 → 41.1 s (it waited 16.3 → 8.0 s for its producers).
- **The n-gram's gain is in `dist()`.** The full mix asks for the whole distribution every byte; `--no-full-mix` only calls `code()`, which changed little.
- **One thread, LSTM 256: only −15%.** Its weights (2.2 MB, read twice a byte) do not fit one core's 1 MB L2, so one core waits on memory either way. On 4 threads each core's quarter of the weights nearly fits, and the lanes pay off. The smaller LSTMs fit on one core: the LSTM 64 −42% (above), the LSTM 128 −42% (harness, below).
- The other users: `lstm-train` 87.0 → 45.7 s; `gln-train` 139.0 → 116.3 s; K5's generation with `--k5-lstm` 62.8 → 50.0 s (its wall 79.9 → 67.4 s). Each pair ran together.
- User CPU in the default small run: 712 → 663 s; system 139 → 83 s.

## The experts alone

A harness (`o5_bench.cpp`) drives each expert as `context8k` does, on enwik8, and hashes every number it returns. Old and new headers ran together, unless marked. Seconds.

**The n-gram** (orders 1–6, 8, 12; tb 24; cap 60): pretrain over 6 MiB, `code()` over 1 MiB, then `dist()` and `code()` on every byte of 128 KiB.

| Pretrain threads | Make the tables | Pretrain | Warm-up `code()` | `dist` + `code` | Total |
|---|---:|---:|---:|---:|---:|
| 1 | 0.39 → 0.39 | 6.07 → 6.05 | 2.44 → 2.35 | 8.92 → 5.78 | 17.8 → 14.6 |
| 1 | 4.70 → 4.70 | 9.30 → 8.59 | 3.27 → 3.73 | 12.65 → 7.67 | 29.9 → 24.7 |
| 1 | 5.92 → 6.53 | 10.49 → 10.00 | 3.65 → 3.39 | 12.60 → 8.05 | 32.7 → 28.0 |
| 4 | 1.83 → 2.23 | 3.62 → 3.49 | 3.84 → 3.83 | 13.34 → 7.14 | 22.6 → 16.7 |
| 4 | 0.58 → 0.58 | 3.53 → 3.10 | 3.39 → 2.47 | 11.14 → 7.08 | 18.6 → 13.2 |

The two arms that were dropped, each against the same code without it, run together unless marked (hashes equal):

| Arm | Make the tables | Pretrain | Warm-up | `dist` + `code` |
|---|---:|---:|---:|---:|
| 2 MB pages, together (no pages → pages) | 0.94 → 5.00, 0.38 → 1.69 | 6.69 → 5.33, 6.57 → 5.18 | 2.40 → 2.01, 2.32 → 2.19 | 5.77 → 5.87, 5.66 → 5.64 |
| 2 MB pages, one at a time, 4 threads, memory free | 0.81 → 0.53, 0.87 → 0.49 | 2.71 → 2.37, 2.44 → 2.36 | 3.09 → 2.50, 3.25 → 2.59 | 6.68 → 7.22, 6.76 → 6.26 |
| one byte ahead, together (without → with) | 1.53 → 1.59, 0.63 → 0.76, 0.49 → 0.50 | 10.21 → 11.00, 9.01 → 8.72, 8.79 → 8.93 | 3.23 → 2.96, 3.32 → 3.20, 3.53 → 3.78 | 6.70 → 7.36, 7.45 → 7.50, 6.84 → 7.33 |

With three runs at once, the 2 MB pages took 11.1 and 15.1 s to make.

**The LSTM** (`dist` then `step` with learning, every byte, 16 KiB of enwik8):

| Hidden | Threads | Main | New | Change |
|---|---|---:|---:|---:|
| 256 | 1 | 10.85 | 9.12 | −16% |
| 256 | 4 | 5.55 | 2.47 | −56% |
| 128 | 1 | 3.09 | 1.79 | −42% |
| 128 | 4 | 1.64 | 0.80 | −51% |
| 64 | 1 | 0.83 | 0.66 | −21% |
| 64 | 1 (128 KiB, twice) | 9.43, 9.79 | 5.94, 6.47 | −37%, −34% |

The fallback (`CYPHA_LSTM_AVX512=0`) against main, together: LSTM 256 on 1 thread 10.41 → 9.85, on 4 threads 5.48 → 5.22; LSTM 64 on 1 thread (128 KiB) 9.43 → 10.13 and 9.79 → 10.99, so 7–12% slower. Only a CPU without AVX-512 runs it.

## The full run

`cyphalm_trie context8k /mnt/d/cypha_llm/enwik8 /mnt/d/cypha_llm/corpus/labelled/dolly/databricks-dolly-15k.jsonl`, default threads. Old from 04:21, new from 05:35 (another session's full run held the lock in between). **These ran one after the other, so their times carry the load's noise.**

| | Old | New |
|---|---:|---:|
| `context8k_bpc` | 1.4962 (meta_mix M1Gs) | **1.4962**, stdout `cmp` equal |
| `lstm_alone_bpc`, `ngram_alone_bpc` | 2.2755, 1.7555 | the same |
| LSTM thread busy | 354.9 s | 141.2 s (−60%) |
| n-gram thread busy (online part) | 112.6 s | 44.9 s (−60%) |
| similar-context thread busy | 775.1 s | 735.1 s |
| main thread mixing; waited for producers | 595.0 s; 247.5 s | 522.5 s; 226.7 s |
| full mix's own seconds | 845.9 | 759.9 |
| Wall | 2,026 s | 2,194 s |
| User, system CPU | 3,714 s, 1,035 s | 3,439 s, 562 s |
| Peak RSS | 1,520 MB | 1,448 MB |

- The full run reproduces the default exactly: `context8k_bpc` 1.4962.
- The LSTM and the n-gram threads did 60% less work. The run still waits on the similar-context expert (its thread was busy 735–775 s while the main thread waited about 230 s), so the wall time did not fall. The new run's longer wall is most likely load: it ran later, beside other sessions' runs. The full runs are a check of the output, not a timing test.
- The peak RSS difference is not O5's: the small pairs, run together, show the LSTM's copies adding 4–6 MB.

## The predictions (§13.41)

| Prediction | Result | |
|---|---|---|
| O5 exact | every check `cmp` equal; the full run gives `context8k_bpc` 1.4962 | **hit** |
| The LSTM stage down at least 40% | −56% in the default small run (LSTM 256, its default 4 threads); −61% and −57% without the meta mix and without the full mix; −42% for the LSTM 64. On one thread the LSTM 256 is only −15%. (Full run, one after the other: −60%.) | **hit** (at the default threads) |
| The n-gram stage down at least 10% | −45% in the default small run (−48% without the meta mix); −4% with `--no-full-mix`, where no whole distributions are asked for. (Full run: −60%.) | **hit** (default) |

The method registered for the n-gram, prefetching one byte ahead, gave nothing. The gain came from doing less work per lookup instead.

## What to use

- **Keep it, on by default.** It is exact, and the experts' own time falls by about half at the default threads. The PC's AVX-512 path is chosen at run time. `CYPHA_LSTM_AVX512=0` turns it off (same numbers).
- **Expect little change in a run's wall time.** The LSTM and the n-gram were not what `context8k` waited for. The full mix's main thread and the similar-context expert are; O2 (expert pipelining) and O8 (SIMD across the meta mix's weight sets) aim there.
- Threads: on one thread the LSTM 256 gains only 15%; on its default 4 it gains 56%. Two threads were not measured.

## Files

- Code: `native/include/cypha/online/byte_lstm.hpp`, `native/include/cypha/online/online_ngram.hpp`.
- Runs on the PC: `/mnt/d/cypha_llm/opt/o5/` (`S_*` small pairs with `.cmp`, `F_*` the full runs and `F.cmp`, `L_*`, `G_*`, `K_*` the other users and `others.cmp`, `small.log`, `full.log`). Run scripts and the harness: `/mnt/d/cypha_llm/opt/o5/scripts/` (`k8.sh`, `pair.sh`, `small.sh`, `others.sh`, `full.sh`, `o5_bench.cpp`); the harness's outputs: `/mnt/d/cypha_llm/opt/o5/bench/`. The 2026-11-05 runs that were stopped: `stopped_20261105/`, `stopped_20261106/`.
