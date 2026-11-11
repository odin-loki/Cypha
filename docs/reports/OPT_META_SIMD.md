# O8: the meta mix's weight sets on AVX-512

**Odin Loch** · 2026-11-12 · branch `wip/opt-meta-simd` · GENERATION_PLAN §13.41 O8

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
