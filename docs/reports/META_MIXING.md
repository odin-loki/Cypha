# Meta mixing over the full mix

2026-11-05 · Odin Loch

**Odin:** "We have a mixer and we could possibly do meta mixing. Do a google search and see what you come up with and do that too."

Branch `wip/meta-mix`, from main e9b9823. Odin's PC (WSL). Plan: [GENERATION_PLAN §13.32](../GENERATION_PLAN.md). Data: `/mnt/d/cypha_llm/enwik8` and `/mnt/d/cypha_llm/corpus/labelled/dolly/databricks-dolly-15k.jsonl`.

## Why

`context8k`'s headline is the full mix's arm ABs: 1.5051 bits per byte on this PC ([FULL_DISTRIBUTION_MIXING.md](FULL_DISTRIBUTION_MIXING.md)). ABs is one bitwise logistic mixer (A, four weight sets and a second layer), one geometric mixer (B), SSE after each, and a small mixer joining the two per bit. Each part has one learning rate, picked on the first half.

The search (§13.32) found three ways to mix the mixers:
- **PAQ8 stacks mixers.** Many first-layer weight sets, each picked by a different small context, and a second layer joining them before SSE.
- **MetaGrad and Squint** run one learner at many learning rates. A master weights them by their loss, so no rate needs tuning.
- **Fixed share** (Herbster and Warmuth) follows whichever expert is best now. Text changes topic page by page, so the best arm may change too.

## The design

Code: `native/include/cypha/online/meta_mixer.hpp` (new). It is a read-only hook on the full mix (`fullmix::StepHook`, a few lines in `full_mixer.hpp`):
- it sees each byte's inputs, and every full-mix arm's probability of each bit before any arm learns the bit;
- it changes nothing in the full mix. With no hook (the default), the full mix runs as before.

`context8k --meta-mix` turns it on. `--meta-mix-params k=v,...` sets it. It needs the full mix on. It prints `meta_mix_arm` lines and a `meta_mix_bpc` line after the full mix's lines. It does not change `context8k_bpc`.

**M1: PAQ's second layer, wider.** One more bitwise mixer over A's inputs (every expert, the chain's stages, Fn, copy's input, a bias). It has up to 12 first-layer weight sets, each picked by its own context:

| # | Selector | Sets |
|---|---|---:|
| 0 | the partial byte (as A) | 256 |
| 1 | InfiniGram's bucket × bit position (as A) | 256 |
| 2 | copy's bucket × copy's state × bit position (as A) | 256 |
| 3 | the previous byte's class × the partial byte (as A) | 1,280 |
| 4 | the previous byte × the partial byte (order 1) | 65,536 |
| 5 | matched length (log2 of InfiniGram's length) × copy's bucket × bit position | 1,024 |
| 6 | InfiniGram's bucket × the partial byte | 8,192 |
| 7 | the two previous bytes' classes × bit position | 200 |
| 8 | the LSTM's confidence on this bit (sign, 8 levels) × bit position | 128 |
| 9 | similar-context's bucket × bit position | 968 |
| 10 | Kneser–Ney's bucket × bit position | 512 |
| 11 | a hash of the two previous bytes and the partial byte (order 2) | 65,536 |

- The plan named "LSTM-state sign bits". The LSTM's hidden state is not among the full mix's inputs (nor in its dump), so selector 8 uses the sign and size of the LSTM's own stretched bit probability instead.
- A second layer joins the sets' stretched outputs. Its weights are picked by bit position, or by the partial byte (`l2=1`).
- Then the same SSE as the full mix (**M1s**), then a per-bit mixer with Bs, as ABs joins As and Bs (**M1Bs**).
- Layer-1 weights are float32 (A's are double), to halve the memory of the two 65,536-set tables.

**M2: a mixture over learning rates.**
- **M2A:** A at its rates × {1/4, 1/2, 1, 2, 4}. The 1× member is the full mix's own A. A master joins them per bit: exponential weights with fixed share (`m2m=0`), or a small logistic mixer (`m2m=1`). Then SSE (**M2As**).
- **M2G:** the geometric mixer B at its rate × {1/4, 1/2, 1, 2, 4}; the 1× member is B itself. A master joins their whole byte distributions per byte (exponential weights with fixed share). Then SSE (**M2Gs**).
- **M2ABs:** M2As and M2Gs, mixed per bit as ABs.

**M3: a master over the arms.** Its arms are Fs, As, Bs, ABs, M1s, M1Bs, M2As, M2Gs and M2ABs.
- **M3:** fixed share per bit: p = Σ wᵢ pᵢ; then wᵢ ← wᵢ · pᵢ(bit)^η, normalised, then w ← (1 − α) w + α / 9.
- **M3L:** a logistic mixer over the same arms' stretched probabilities (weights by bit position, or by partial byte), starting at ABs. **M3Ls:** M3L, then SSE.

Every arm is bitwise, so its byte probability is the product of its 8 bit probabilities and is exactly normalised. Every bit is predicted before anything learns it.

**Tuning.** `cyphalm_trie meta-mix-replay FILE FULL_MIX_PARAMS META_PARAMS first-half` runs the full mix and the meta mixers over a `context8k --full-mix-dump`, and stops at the holdout's half. It never reads the second half.

## Checks before tuning

- **D0:** main e9b9823, `context8k --full-mix-dump`. Its output equals N1's (main's default in [FULL_DISTRIBUTION_MIXING.md](FULL_DISTRIBUTION_MIXING.md)) byte for byte (`cmp`): ABs 1.5051. It ran at a load of 30–70 from other sessions: 4048 s, 1149 MB. The dump is 27.9 GB.
- **Small setup** (`--train 1048576 --sample 1179648`, not the scored holdout):
  - this branch without `--meta-mix` gives main's output byte for byte (`cmp`);
  - with `--meta-mix`, the only change is the new `meta_mix` lines, added after the full mix's lines. Every other line is unchanged.
- **Replay sanity:** with only A's four selectors (`sel=0:1:2:3`), M1 equals A (1.5449 on the first half). M1's float32 weights do not change A to 4 decimals.

## Tuning (replay, first half only)

`meta-mix-replay d0.fmx "" PARAMS first-half`, six at a time, 230–330 s each. Bits per byte on the holdout's **first half**. The full mix's own arms are the same in every row: F 1.5978, Fs 1.5747, As 1.5396, Bs 1.5408, **ABs 1.5272**.

| Run | Settings | M1s | M1Bs | M1Gs | M2As | M2Gs | M2ABs | M3 | M3L | M3s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T00 | first defaults (sel 0–10, l2 0, m1lr1 0.001, m3a 1e-3) | 1.5299 | 1.5234 | | 1.5360 | 1.5385 | 1.5252 | 1.5246 | 1.5282 | |
| T01 | m3a 1e-4 | | | | | | | 1.5230 | | |
| T02 | m3a 1e-2 | | | | | | | 1.5260 | | |
| T03 | sel 0:1:2:3 (A's own) | 1.5396 | 1.5272 | | | | | 1.5269 | 1.5292 | |
| T04 | l2 1 | 1.5295 | 1.5231 | | | | | 1.5243 | 1.5275 | |
| T05 | m2m 1 (logistic master over rates) | | | | 1.5359 | | 1.5265 | 1.5251 | | |
| T10 | m1lr1 0.0005 | 1.5319 | 1.5265 | | | | | 1.5256 | | |
| T11 | m1lr1 0.002 | 1.5329 | 1.5218 | | | | | 1.5238 | 1.5272 | |
| T12 | m1lr2 0.002 | 1.5351 | 1.5247 | | | | | 1.5252 | | |
| T13 | l2 1, m3a 1e-4 | 1.5295 | 1.5231 | | | | | 1.5225 | | |
| T14 | sel without 4 (order 1) | 1.5316 | 1.5243 | | | | | 1.5252 | | |
| T15 | sel 0–11 (+ order 2) | 1.5285 | 1.5228 | | | | | 1.5241 | | |
| T20 | sel 0–11, l2 1, m1lr1 0.002, m3a 1e-4 | 1.5284 | 1.5196 | | | | | 1.5196 | 1.5249 | 1.5250 |
| T21 | T20 + m1r 1 (M1 over rates) | 1.5257 | 1.5186 | | | | | 1.5192 | 1.5240 | 1.5246 |
| T22 | T20 + M3L rates 0.003/0.002 | 1.5284 | 1.5196 | | | | | 1.5196 | 1.5290 | |
| T23 | T20, m1lr1 0.003 | 1.5323 | 1.5190 | | | | | 1.5191 | 1.5241 | 1.5243 |
| T24 | T20 + M2 shares 1e-4 | | | | 1.5348 | 1.5390 | 1.5257 | 1.5196 | | |
| T25 | T20 + m3eta 0.7 | | | | | | | 1.5193 | | 1.5246 |
| T30 | T21 (new code, M1Gs added) | 1.5257 | 1.5186 | 1.5180 | | | | 1.5192 | 1.5240 | 1.5246 |
| T31 | T30 + sel 12 (order 3) | 1.5247 | 1.5180 | 1.5174 | | | | 1.5187 | 1.5233 | |
| T32 | T30 + sel 13 (word) | 1.5253 | 1.5183 | 1.5177 | | | | 1.5189 | 1.5236 | |
| **T33** | **T30 + sel 12 and 13** | 1.5245 | 1.5179 | **1.5173** | 1.5360 | 1.5385 | 1.5252 | 1.5185 | 1.5231 | 1.5239 |
| T34 | T30, m1lr1 0.003 | 1.5266 | 1.5183 | 1.5176 | | | | 1.5189 | | |
| T35 | T30 + M3L from M1Bs, lr 0.002 | | | | | | | | 1.5231 | |
| T40 | T33, scales 1/8–2 | 1.5238 | 1.5190 | 1.5191 | 1.5349 | 1.5392 | 1.5270 | 1.5196 | | |
| T41 | T33 + m2geta 0.5 | | 1.5179 | 1.5174 | | 1.5382 | | 1.5185 | | |
| T42 | T33 + m2gshare 0.01 | | 1.5179 | 1.5173 | | 1.5384 | | 1.5185 | | |
| T43 | T33, scales 1/2–2 | 1.5248 | 1.5179 | 1.5176 | 1.5371 | 1.5393 | 1.5259 | 1.5185 | | |
| T44 | T33, l2 0 | 1.5252 | 1.5189 | 1.5184 | | | | 1.5196 | | |
| T45 | T33 + m1lr2 0.001 | 1.5259 | 1.5183 | 1.5177 | | | | 1.5190 | | |

(Blank: the same as the row above it, or not affected by that row's change. Every run's full output is in `/mnt/d/cypha_llm/meta_mix/tune/`.)

What tuning showed:
- **M1, PAQ's wider first layer, is the gain.** Each new selector adds a little: order 1, order 2, order 3 and the word hash most. Weights by partial byte in the second layer help (l2 1).
- M1 wants a faster first layer than A (0.002 against 0.001), though M1 alone is then worse. It is better once mixed with B.
- **M1 over rates (m1r 1)** gains 0.001 more. The master spreads its weight over ×1/2, ×1 and ×2.
- **M2 alone gains little:** M2ABs 1.5252 against ABs 1.5272 (−0.002). The rate tuned on the first half was already near the best rate.
- **M3, fixed share over the arms, follows M1Bs** (weight 0.8–0.9) and ends up 0.001 behind it. It does not find a better arm by switching. A small share works best (1e-4).
- **The logistic master M3L, and SSE after a master (M3Ls, M3s), lose** to the simple 2-input mixers.
- **Best: M1Gs**, M1s and M2Gs mixed per bit, as ABs mixes As and Bs: 1.5173, −0.0099 against ABs.

**Chosen** (by the headline arm's first half): headline **M1Gs**, with T33's settings. They are now the defaults in `metamix::Params`: sel 0–13, l2 1, m1lr1 0.002, m1r 1, scales {1/4, 1/2, 1, 2, 4}, m3a 1e-4; the rest as before.

## Registration

Written before any second-half read of a meta-mix arm.

**Runs**, side by side at the same load:
- **MA:** main e9b9823, `context8k` (its default; ABs 1.5051).
- **R1:** this branch, `context8k --meta-mix` (the defaults above). One run, the registered read.
- **R0:** this branch, `context8k` (no flag). Its whole output must equal MA's (`cmp`).

**Rule.** M1Gs becomes `context8k`'s default, with `context8k_bpc` taken from it, if all hold:
1. R1's M1Gs second-half bits are at least 0.005 below 1.5051 (so at most 1.5001).
2. R1's TALLY and GEN lines, and every other line before the `meta_mix` lines, equal MA's. The full mix's replica checks hold (arm F equals `online_final_bpc`; `full_mix_check` 0).
3. R0's output equals MA's byte for byte.
4. `cyphalm_trie selftest` passes.
5. Wall time and peak RSS are stated beside MA's.

If it fails, the meta mix stays behind `--meta-mix`, off by default. Every arm is reported either way.

**Predictions** (second half, against ABs 1.5051):
- P1: M1Gs is between −0.007 and −0.012 (first half −0.0099). The rule holds (about 75%).
- P2: M1Bs is within 0.001 of M1Gs.
- P3: M3 is within 0.002 of M1Bs, and does not beat M1Gs.
- P4: M2ABs is between −0.001 and −0.004.
- P5: M3L and M3Ls do not beat M1Gs.
- P6: the cost: wall within +15% of MA's (the meta mixers run on the main thread, which waits for the LSTM); peak RSS about +400 to +600 MB.
