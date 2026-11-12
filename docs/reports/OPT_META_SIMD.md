# O8: the meta mix's weight sets on AVX-512

**Odin Loch** · 2026-11-12 · branch `wip/opt-meta-simd` · GENERATION_PLAN §13.41 O8

**Outcome first.** Exact everywhere, the full run included (`context8k_bpc` 1.4962). AVX-512, as registered, is no faster (meta-mix time +3.9%): it lowers this Xeon's clock. The same layout with SSE2 (amendment O8b, registered before its runs) cuts meta-mix time 14.8% and the full mix's own time 8%. Prediction −50%: **miss**. Kept with SSE2 as the default kernel; not on main.

## Registered before any run (2026-11-12)

§13.41 registered O8: "the 14 × 5 weight sets are independent mixers doing the same operations, so one AVX-512 lane per set keeps each set's order. Exact; meta-mix time down at least 50%." This section adds the details §13.41 left open. It is committed and pushed before any timing or check runs.

**What changes.** Only `native/include/cypha/online/meta_mixer.hpp`.
- M1 runs at 5 rates (m1r = 1, scales 0.25, 0.5, 1, 2, 4). Each rate is a two-layer logistic mixer with 14 first-layer weight sets (the 14 selectors). All five read the same inputs and the same 14 contexts. Only their weights and rates differ.
- The five mixers become one bank (`M1Bank`). For one selector and one context, the 5 rates' weights lie side by side, input after input. One AVX-512 vector of 8 doubles holds the 5 rates of one selector (3 lanes unused). 4 selectors run at once, so 4 sums are in flight.
- Each lane adds its products in the scalar order, from 0, input 0 first. A float weight is widened to double, as the scalar code does. The product is kept apart from the sum (an empty `asm` barrier, as O5's LSTM kernels do), so no fused multiply-add can form.
- The weight update is done the same way: each lane computes `w += float(err × x)` in its own lane.
- The second layer, squash and stretch stay scalar, in their order.
- The kernel is compiled for AVX-512F only and chosen at run time. `CYPHA_META_AVX512=0` runs the scalar loops over the same layout.
- The memory is the same: the bank holds exactly the five mixers' weights. `bytes()` reports the same number.
- Not changed: M2A's four extra bitwise mixers (in `full_mixer.hpp`'s `BitMixer`), the masters, the APMs.

**Checks** (exact means `cmp` of the whole stdout, old binary against new; old is main 9cbd277, new is this branch; same compiler and flags, GCC 13.3, CMake Release):
1. small `context8k` (`--train 1048576 --sample 1179648`), default;
2. the same with `--no-meta-mix`;
3. the same with `--no-full-mix`;
4. the small default with `CYPHA_META_AVX512=0` (the new binary's scalar path) against old;
5. the planner's and `--no-k5-plan`'s dev GEN files (`scripts/gen_eval.sh dev`; `sample` never builds a meta mixer, so these must be equal);
6. one full default `context8k` under `flock ~/.cypha_full.lock`: `context8k_bpc` 1.4962 must reproduce, and stdout must equal the O5 full run's (`/mnt/d/cypha_llm/opt/o5/F_new`, main's context8k code is unchanged since).

**Measure of meta-mix time** (fixed now): the full mix's own seconds (the sum of its six parts, the "mixer s:" line on stderr; it leaves out waiting for producers) with the meta mix, minus the same without it (`--no-meta-mix`). Four small runs are started together: old and new, each with and without the meta mix, 4 threads each. The pair is run 3 times; the median change is scored. Wall time and peak RSS are reported beside it.

**Prediction** (§13.41, unchanged): exact; meta-mix time down at least 50%.
- Expected by this registration: the M1 bank is most of the meta mix's cost (70 dot products of about 30 inputs per bit, against 16 for M2A), so −50% is in reach only if gathers are not needed; the side-by-side layout avoids them. My own guess: −40% to −60%.

**Machine:** threads as O5 ran them (the LSTM 4, the n-gram 4, the defaults); no run starts while `free -g` shows under 10 GB available; the full run takes `flock ~/.cypha_full.lock`.

## First runs (2026-11-12, 10:04–10:41)

Old is main 9cbd277 (`~/j3/b_main`), new is eb84f91 (`~/j3/b_o8`). The PC's load average was 15–30 (other sessions).

**Exact.** All `cmp` equal:
- small `context8k` default, 3 rounds, and `--no-meta-mix`, 3 rounds (stdout, old against new);
- small `--no-full-mix` (old against new);
- the new binary's scalar path (`CYPHA_META_AVX512=0`), small default, against old;
- the planner's dev GEN file and `--no-k5-plan`'s (GEN file, `sample` stdout; SCRIBE tables and summaries equal apart from the label).

**But slower.** The full mix's own seconds (sum of the six parts), 4 runs started together each round:

| Round | Old, default | New, default | Old, no meta | New, no meta | Meta-mix time, old → new |
|---|---:|---:|---:|---:|---:|
| 1 | 79.2 | 81.9 | 34.6 | 34.8 | 44.6 → 47.1 (+5.6%) |
| 2 | 69.1 | 71.4 | 30.3 | 30.0 | 38.8 → 41.4 (+6.7%) |
| 3 | 74.9 | 83.9 | 30.1 | 33.5 | 44.8 → 50.4 (+12.5%) |

- The bit loop (where M1 runs) fell 10–14% in rounds 1 and 2. But the parts that run no new code grew: the geometric mixers ("geo") by 4–6 s, the chain and the final layer by 5–15%.
- The likely cause: AVX-512 lowers this Xeon's clock while its wide units are busy, and the whole thread then runs slower. The meta mix's dot products are short (about 30 inputs), so the clock drops for a small gain.
- So the registered O8 is exact but **meta-mix time +6.7% (median)**, not −50%.

## Amendment O8b (registered 2026-11-12, before its runs)

- **O8b:** the same bank and layout, with SSE2 kernels (two rates a 128-bit vector; 3 vectors for the 5 rates). SSE2 is part of every x86-64 CPU and does not lower the clock. `CYPHA_META_SIMD=scalar|sse2|avx512` chooses the kernel (all three must give the same numbers).
- **Runs:** 3 rounds of 5 small runs started together: old default, new default with `avx512`, with `sse2`, with `scalar` (the new layout, no vectors), and old `--no-meta-mix` (the meta-free baseline for all, since the code without the meta mix is the same). Meta-mix time = default − baseline. Exact: `cmp` of each new stdout with old's.
- The kernel kept is the fastest of the three, if it is faster than old; it becomes the default. O8's rule and prediction stand (exact; meta-mix time down at least 50%). My guess for SSE2: −10% to −25%. The meta mix is not mostly M1's dot products.

## O8b results (2026-11-12, 10:42–11:03)

New is c770565 (`~/j3/b_o8b`), the kernel chosen by `CYPHA_META_SIMD`. 5 runs started together each round; the PC's load average was about 15–25.

**Exact.** In every round the stdout of `avx512`, `sse2` and `scalar` equals old's (`cmp`; `context8k_bpc` 1.7501 on the small setup).

The full mix's own seconds (sum of the six parts); meta-mix time = default − old `--no-meta-mix`:

| Round | Baseline (no meta) | Old | AVX-512 | SSE2 | Scalar, new layout |
|---|---:|---:|---:|---:|---:|
| 1 | 31.5 | 69.7 → meta 38.2 | 71.2 → 39.7 (+3.9%) | 63.9 → 32.4 (−15.2%) | 64.4 → 32.9 (−13.9%) |
| 2 | 31.7 | 69.5 → 37.8 | 72.2 → 40.5 (+7.1%) | 63.9 → 32.2 (−14.8%) | 64.5 → 32.8 (−13.2%) |
| 3 | 31.0 | 68.5 → 37.5 | 68.2 → 37.2 (−0.8%) | 63.4 → 32.4 (−13.6%) | 63.8 → 32.8 (−12.5%) |
| **Median** | | | **+3.9%** | **−14.8%** | **−13.2%** |

- The full mix's own time with SSE2: 69.5 → 63.9 s (−8%, median). Wall time does not move (6:25–6:52 for all): the full mix's main thread is not what the small run waits for at the end. Peak RSS is the same (1,340 MB).
- **Most of the gain is the layout, not the vectors.** The scalar loops on the new layout are only 1–2 points behind SSE2. Putting the 5 rates' weights side by side means one context's 5 rows are one run of memory, not 5 rows in 5 tables.
- AVX-512 is no faster than old: the clock drop costs what the wide lanes save.
- The M1 bank is not most of the meta mix. Its dot products are about 70 × 30 multiply-adds a bit. The rest (M2A's four mixers, the M2G geometric mixers, five SSE stages, the masters, about 90 logs and exps a bit) does not change.
- **SSE2 is the default kernel** (4935cee). AVX-512 and scalar stay behind `CYPHA_META_SIMD`.

## The full run (2026-11-12, 11:33–12:02, under the lock)

`cyphalm_trie context8k /mnt/d/cypha_llm/enwik8 …/databricks-dolly-15k.jsonl`, default threads, the default kernel (SSE2), binary 4935cee (`~/j3/b_o8c`).
- **stdout equals the O5 full run's** (`/mnt/d/cypha_llm/opt/o5/F_new.out`, 260 lines, `cmp`): **`context8k_bpc` 1.4962**.
- Wall 28:51, peak RSS 1,562 MB; the full mix's own time 376.6 s. O5's run (2026-11-06) took 36:34 and 507 s, but it ran on another day beside other jobs, so these are not compared.

## Summary of checks

| Check | Result |
|---|---|
| small default, 3 rounds (eb84f91, AVX-512) | `cmp` equal |
| small `--no-meta-mix`, 3 rounds | equal |
| small `--no-full-mix` | equal |
| small default, scalar path (`CYPHA_META_AVX512=0`) | equal |
| small default with `avx512`, `sse2`, `scalar`, 3 rounds each (c770565) | equal |
| the planner's dev GEN file; `--no-k5-plan`'s | equal (GEN file, stdout; SCRIBE tables apart from the label) |
| full default `context8k` (4935cee, SSE2) | equal to O5's full run; `context8k_bpc` 1.4962 |

## Lines of code

`meta_mixer.hpp`: 412 → 655 lines (+243). The bank and its three kernels (scalar, SSE2, AVX-512) and their comments came in; about 20 lines of the five-mixer code went.

## §13.41 O8 prediction, scored

| Prediction | Result | Score |
|---|---|---|
| exact | every check `cmp` equal; the full run reproduces 1.4962 | **hit** |
| meta-mix time down at least 50% | AVX-512 (as registered): **+3.9%** (median, 3 rounds; +6.7% in the first 3); SSE2 (O8b): −14.8%; scalar on the new layout: −13.2% | **miss** |
| Mine (O8b): SSE2 −10% to −25% | −14.8% | **hit** |

**Kept, with SSE2** (the §13.40 rule: exact, and it lowers time with nothing worse): meta-mix time −15%, the full mix's own time −8%, the same RAM. It adds 243 lines. The wall time of a run does not move, because the run waits on the similar-context expert, not on the mix. Not on main: the lead merges.

**What the meta mix's time is.** M1's 70 dot products were not most of it. The rest is about 90 logs and exps a bit (stretch and squash for M1's 70 sets, M2A, five SSE stages, the masters), M2A's four mixers, and the M2G geometric mixers per byte. A next step would be a cheaper exact stretch of the same squash (the stretch of squash(d) is d, up to the clamp and rounding, so it cannot be skipped exactly), or fewer meta arms.

## Files

- Code: `native/include/cypha/online/meta_mixer.hpp` only.
- Runs on Odin's PC, `/mnt/d/cypha_llm/opt/o8/`: `R{1,2,3}_*` (first rounds), `nfm_*`, `def_scalar`, `small.log`; `b/` (the O8b rounds, `b.log`); `gen/` (the GEN pairs); `full/` (the full run, `full.log`, `F_o8.cmp`). Scripts: `~/j3/o8small.sh`, `~/j3/o8b.sh`, `~/j3/genpair.sh`, `~/j3/full.sh`, `~/j3/k8.sh`.
