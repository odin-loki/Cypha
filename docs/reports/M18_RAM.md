# M18: `qa` back under the RAM line

**Odin Loch** · 2026-11-12 · PLAN Track P, M18 · branch `wip/m18-ram` (from main `dfae57de`)

(Registered before any code change. Results go under "Results" below.)

## The goal

Every `qa` configuration at least 10 MB under 200 MB: peak VmHWM at most 190 MiB (MiB = kB / 1024). Outputs must not change. J5 (QA_RECHECK.md) measured, on `fb79cc93`'s code: built 196.3, built with `--save` 200.6, served 189.7, `a0` 176.3 MiB.

## Step 1: why the peaks rose

J5 found the peaks 2.0–8.8 MiB above M18's R1 and R2 (`ac9ac07`, 2026-11-01), with the same outputs. I built main's code-touching commits from `ac9ac07` to `dfae57de` (Release, the same flags) and ran each one as R1 runs: C1L with `--save`, pinned to 4 CPUs, VmHWM polled every 0.25 s. All ran side by side on the PC at the same time.

| Commit | What it brought | Peak kB | MiB |
|---|---|---:|---:|
| `ac9ac07` | R1's commit | 199,084 | 194.4 |
| `eaab8205` | modules reorganised | 199,496 | 194.8 |
| `975a99bd` | K5 in `sample` | 199,324 | 194.7 |
| `775396e9` | full mix on in `context8k` | 199,296 | 194.6 |
| `394a3d51` | meta mix | 199,712 | 195.0 |
| `7bf5e659` | discourse planner | 200,764 | 196.1 |
| `4b255e01` | O5 | 200,012 | 195.3 |
| `12c14760` | O1, O6, G16 merges | 200,200 | 195.5 |
| `d17bd76d` | O3 Boolean | 200,112 | 195.4 |
| `dfae57de` | main today | 200,708 | 196.0 |

Every run's rows file equals R1's byte for byte.

**The cause.** No commit raised the peak. From `ac9ac07` to `dfae57de` it moved 1.6 MiB, up and down, with no step. Most of J5's rise is not in the code:

- The same binary gives different peaks on different runs. `ac9ac07` read 191.8 MiB on 2026-11-01 and 194.4 today. J5's `--save` run of today's code read 200.6; mine of the same code reads 196.0, with the same rows.
- VmHWM comes from the kernel's RSS counter. On this kernel (6.6, WSL) that counter is kept per CPU and summed late, so it can be off by a few MiB on a busy 64-thread PC. `VmRSS` read 1.3–2.3 MB below the exact page count (`smaps_rollup`) on the runs above.
- Where glibc puts each big array (its own mapping, or the heap) depends on what was freed before it. A freed heap array often stays resident. So small changes in order move the peak by a few MiB.

So the gate cannot be met by undoing a commit. It needs real cuts with room to spare.

## Where the peak sits

On `dfae57de` (profile above) the peak is the end of the holdout pass, after the similar-context expert is built: 197.1 MB during its holdout pass, 200.7 MB when the doc cache is built. The early peak, the chat training and its packing, is 183.7 MB (179.4 MiB).

At the late peak the per-byte holdout arrays are 56 MiB (the holdout is 1 MiB; each `double` array is 8 MiB, each `int` array 4 MiB):

- `pt`, `pinf`, `pig`, `pcp`, `bk`, `cbk` are all allocated (and zero-filled) at the top of the holdout pass, though `pig`, `pcp` and `cbk` are first written much later;
- `bt0` and `bi0` are copies of `pt` and `pig` with the class expert mixed in, and `pt` and `pig` stay resident beside them;
- the similar-context pass adds `ps` and `sb`.

## The cuts (all in `cmd_qa`, `native/tools/lm/cyphalm_trie.cpp`)

Every cut does the same arithmetic in the same order. Only where values live and when they are freed changes.

1. **Allocate each array where it is first written.** `pig` is made after InfiniGram's lambdas, written into `pinf`'s memory (`pinf` is not read again). `pcp` and `cbk` are made at the copy stage.
2. **Mix in place.** `bt0` and `bi0` take `pt`'s and `pig`'s memory (a move), and the class expert is mixed into them there. No copy of either is made.
3. **The copy-alone lines first.** With the class expert on, two lines read `pt` and `pig` after the copy fit: `holdout2_trie_copy_bpc` and `cls_given_copy_lci`. Their numbers are computed before the class stage, from the same arrays, and printed where they are printed now. The copy arrays used there are freed, and filled again at the copy stage (they depend only on the holdout bytes).

Nothing in `online/similar_context.hpp` or the mixers changes (T1b on `wip/tune-adapt` hooks those). No shared header changes. No `malloc_trim`, no allocator setting.

After the cuts the per-byte arrays are at most 32 MiB at a time (the Kneser–Ney mix: `pt`, `pinf`, `bk` and its own two), 28 MiB through the similar-context and copy stages, against 56 MiB now.

## The rule

- **Exact.** For each configuration, old and new binary: the rows file byte-identical; the whole stdout equal except timing lines (`LATENCY`, `SIM_LAT`, `build_s`, `seconds`); every `holdout*_bpc` line equal under `cmp` after those lines are removed. Any change in an output means the cut is not kept.
- **Peaks.** Every configuration (built C1L, built C1L `--save`, served `--model`, `a0`) at most 190 MiB, measured as the card does (4 CPUs, `nice -n 5`) and also pinned to one CPU. Old and new binaries run side by side.
- **Kept** only if every configuration is at most 190 MiB with every output identical.
- Small `context8k` (`--train 1048576 --sample 1179648`) is run old against new as a check; it does not go through `cmd_qa`. The writer (K5) does not go through `cmd_qa` either.

## Predictions

| # | Prediction |
|---|---|
| P1 | Every output of every configuration is identical (rows, answers, BPC lines) |
| P2 | Built with `--save`: 176–184 MiB. The peak moves to the chat training (about 179 MiB) |
| P3 | Built (no `--save`): 176–184 MiB |
| P4 | Served (`--model`): 165–178 MiB |
| P5 | `a0`: at most 176 MiB |
| P6 | The time of a `qa` run changes by less than 2% (the copy arrays are filled one more time) |

## Results

(to come)
