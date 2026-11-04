# Whole-distribution mixing for the content mixer, the PAQ way

2026-11-04 · Odin Loch

**Odin:** "Do all of it and make sure you document everything we have tried and the results. Do it all on my PC."

Branch `wip/full-mix`. Odin's PC (WSL), one core per run. Data: `/mnt/d/cypha_llm/enwik8` and `/mnt/d/cypha_llm/chat/dolly.jsonl` (the files §13.12's R0 and R1 used; the Dolly file is byte-identical to `corpus/labelled/dolly/databricks-dolly-15k.jsonl`).

## Why

`cyphalm_trie context8k` mixes only each expert's probability of the true byte: one number per expert per byte.
- That scores a linear mixture correctly.
- It rules out PAQ's strongest tools: logistic mixing in the stretch domain, and adaptive probability maps (APM, also called SSE) after the mixer. Both are nonlinear. They need whole distributions to stay normalised.
- It also means the mixture that scores is not the one that writes.

So here every expert gives its whole next-byte distribution for every holdout byte, and the mixers work on those.

## The design

Code: `native/include/cypha/online/full_mixer.hpp` (new). The hook in `cmd_context8k` is one call after the final layer. `context8k --full-mix` turns it on; without the flag nothing changes.

**1. The inputs, streamed.** One more pass over the holdout. For each byte it builds every expert's whole distribution, as the chain built them:
- the trie's 5 experts and their mixture (`fill_experts`, `mix_dist`);
- byte Kneser–Ney (`ByteKN::dist`);
- InfiniGram at caps 4, 16, 64, 512 and 8192 (count ÷ total);
- class (`ClassContext::predict`) and similar-context (`SimilarContext::predict`);
- copy (its predicted byte and match length);
- the document cache (`DocTopic::Stream::predict`);
- the byte LSTM (`ByteLSTM::dist`), learning online as in the chain.

The similar-context and document experts and the LSTM are built again inside the hook. They are deterministic, so they give the chain's numbers. Nothing is kept per byte, so the pass adds little memory.

**2. The chain and the final layer on whole distributions (the replica).**
- Every chain stage is replayed on all 256 bytes: cur = λ·expert + (1 − λ)·cur.
- The λs learn from the true byte with the same arithmetic as `content_mixer.hpp` (exponentiated gradient, the slow and fast chains, the doc stage's λs as `mix_doc_into` fit them).
- So the replica's probability of the true byte equals the chain's. The hook checks this on every byte.
- The online final layer is replayed the same way. Its weights learn from the true byte exactly as `OnlineFinal` does, so arm F must print `online_final_bpc`.
- F does not sum to 1: absent experts read as zeros, and every input has a floor of 1e-9. **Fn** is F divided by its sum. Fn is a proper distribution, so it could also write.

**3. The PAQ arms.** All run side by side on the same inputs, each learning online from the first byte. Every byte, and every bit, is predicted before it is learned.
- **(a) Bitwise logistic mixing (A).**
  - Each distribution becomes the byte's 8 binary decisions, most significant bit first. Bit j's probability is the distribution's mass on the partial byte's 1-branch over the partial byte's mass.
  - The inputs are those bit probabilities, stretched: ln(p ÷ (1 − p)).
  - Four weight sets, each chosen by its own small context: the partial byte (256); InfiniGram's bucket × bit position; copy's bucket × copy's state (none, disagrees, expects 0, expects 1) × bit position; the previous byte's class × the partial byte.
  - Each set gives squash(w · x). A second layer, with weights by bit position, mixes their stretched outputs (PAQ8's two layers).
  - Every weight learns by w += lr · (bit − p) · x.
  - Copy has its own input: the bit it predicts, while the partial byte still agrees with its byte, at a confidence learned per match-length bucket and bit position.
  - The inputs are every expert, the chain after each stage, the slow and fast chains, and Fn. Every weight set starts with weight 1 on Fn and 0 elsewhere, so A starts exactly at Fn.
- **SSE after a mixer** (the "s" arms). Four APMs per bit, each keyed by:
  - the partial byte (order 0);
  - the previous byte and the partial byte (order 1);
  - a hash of the two previous bytes and the partial byte (order 2, 2^18 contexts);
  - InfiniGram's bucket and the partial byte.

  Each APM has 33 points over the stretched probability in [−16, 16]. It interpolates the two nearest, and both move toward the bit. A small learned mixer then combines the mixer's probability and the four APMs. It starts at the mixer's own.
- **(b) Byte-level geometric mixing (B).**
  - p(s) ∝ ∏ D_j(s)^w_j over the normalised, floored (1e-6) distributions, normalised over the 256 bytes.
  - There are 80 weight sets, chosen as the final layer chooses them. The weights learn by gradient on log loss: Δw_j = −lr · (E_p[log D_j] − log D_j(y)).
  - It starts at Fn. Then comes the same bitwise SSE (Bs).
- Every bitwise arm gives P(byte) as the product of its 8 bit probabilities, so it is exactly normalised. The selftest checks this: after some learning, a bitwise mixer with SSE gives the 256 bytes probabilities that sum to 1 within 1e-9.

**The arms printed** (`full_mix_arm NAME first_half X second_half Y`):

| Arm | What |
|---|---|
| F | the final layer's replica at the true byte (must equal `online_final_bpc`) |
| Fn | F normalised |
| Fs | Fn, then SSE |
| A | the bitwise logistic mixer over every input |
| As | A, then SSE |
| Rs | the bitwise mixer over the raw experts only (no chain, no final layer; it starts at the trie mixture), then SSE |
| B | geometric mixing |
| Bs | B, then SSE |
| ABs | As and Bs, mixed per bit by a small learned mixer |

`full_mix_bpc` prints the headline arm's second half.

**Tuning without the scored half.** `context8k --full-mix-dump FILE` writes the inputs (15 distributions per byte as float32, about 16 GB). `cyphalm_trie full-mix-replay FILE PARAMS` runs the same FullMixer over them in minutes. Learning rates and other settings are chosen on the holdout's **first half** only. The second half, which every `context8k` line scores, is read for the registered runs.

## Registration

Written before any result below was seen.

**Runs:**
- **B0:** `context8k` as it is (origin/main c7dd132, unchanged code). Its `online_final_bpc` is this PC's baseline.
- **X0** (development): `context8k --full-mix --full-mix-dump x0.fmx`, the default settings. It checks the plumbing and writes the dump for tuning:
  - every line B0 prints, X0 prints identically (the TALLY lines included);
  - arm F equals B0's `online_final_bpc`;
  - the replica chain's probability of the true byte equals the chain's on every byte (`full_mix_check`).
  - X0's second-half numbers are not used to choose anything.
- **Tuning:** `full-mix-replay` on X0's dump. Settings and the headline arm are chosen on the first half only.
- **X1** (the registered read): `context8k --full-mix` with the chosen settings built in as the defaults. One run.

**Rule.** `--full-mix`'s headline arm becomes `context8k`'s headline line if all hold:
1. X1's `full_mix_bpc` is at least 0.005 below B0's `online_final_bpc`.
2. Every TALLY and GEN line is identical to B0's (the answers do not change).
3. `cyphalm_trie selftest` passes.
4. The wall time and peak memory are stated beside it. There is no limit on them, but they are part of the decision for `sample` (below).

**Predictions** (bits per byte on the second half, against F = B0's `online_final_bpc`):
- P1: F matches `online_final_bpc` to 4 decimals, and the chain replica matches to 1e-12.
- P2: Fn − F is between −0.005 and 0. F's missing mass is small; most experts are normalised.
- P3: Fs − F is between −0.005 and −0.015. SSE after a good mixer gains 1–3% in PAQ.
- P4: As − F is between −0.015 and −0.035. This is the headline prediction.
- P5: Bs is within 0.01 of As.
- P6: Rs (no chain and no final layer as inputs) is within ±0.03 of F.

## B0: this PC's baseline

`context8k` as it is (origin/main c7dd132). It ran beside X0 while the PC was heavily loaded by another project's builds (load 35–90 on 64 threads), so wall times are slow and noisy.

| | Bits per byte (second half) |
|---|---:|
| Last chain line (`…_doc_lstm_bpc`) | 1.6570 |
| `online_final_bpc` | **1.6232** |

Wall 3403 s; peak RSS 250 MB. The cloud printed 1.6238 for the same code; this PC gives 1.6232.

## X0: the plumbing (development run)

`context8k --full-mix --full-mix-dump x0.fmx`, with the first settings (lr1 0.002, lr2 0.005, lrb 0.002, APM rate 0.02, SSE combined by a learned mixer).
- **Every line B0 prints, X0 prints identically:** 0 differing lines outside the `full_mix` lines. The TALLY and GEN lines are unchanged.
- **Arm F is 1.6232,** the same as B0's `online_final_bpc`. The replica of the final layer on whole distributions is exact at the true byte.
- **The replica chain matches the chain** on every byte (the maximum difference printed as 0.0000 at 4 decimals; X1 prints it in full).
- F's whole distribution sums to 0.99983 on average. The missing mass is the zero experts and the floors.
- Wall 4868 s (B0 + 1465 s); the full-mix pass alone took 1616 s, most of it the experts computed again. Peak RSS 446 MB (+196 MB; the mixers' state is 178 MB, most of it APM tables). The dump is 18.3 GB.
- P1 held.

First half only (the second half was not read until X1):

| Arm | F | Fn | Fs | A | As | Rs | B | Bs | ABs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X0, first half | 1.6333 | 1.6329 | 1.6028 | 1.6012 | 1.5966 | 1.6599 | 1.6706 | 1.6416 | 1.5813 |

## Tuning (replay, first half only)

`full-mix-replay x0.fmx PARAMS`, 190–225 s each, six at a time. With X0's settings the replay gives X0's numbers to 4 decimals. Bits per byte on the holdout's **first half** (F 1.6333, Fn 1.6329 in every row):

| Settings | Fs | A | As | Rs | B | Bs | ABs |
|---|---:|---:|---:|---:|---:|---:|---:|
| X0's (lr1 0.002, lr2 0.005, lrb 0.002, apm 0.02, ssef 1) | 1.6028 | 1.6012 | 1.5966 | 1.6599 | 1.6706 | 1.6416 | 1.5813 |
| lr1 0.001 | 1.6030 | 1.5872 | 1.5829 | 1.6606 | 1.6706 | 1.6404 | 1.5741 |
| lr1 0.005 | 1.6057 | 1.6537 | 1.6453 | 1.6977 | 1.6706 | 1.6469 | 1.6030 |
| lr1 0.01 | 1.6124 | 1.7401 | 1.7205 | 1.7687 | 1.6706 | 1.6562 | 1.6303 |
| lrb 0.0005 | 1.6028 | 1.6012 | 1.5966 | 1.6599 | 1.6057 | 1.5891 | 1.5739 |
| lrb 0.0002 | 1.6028 | 1.6012 | 1.5966 | 1.6599 | 1.6010 | 1.5828 | 1.5728 |
| lr1 0.0005, lrb 0.0002 | 1.6044 | 1.5846 | 1.5804 | 1.6782 | 1.6010 | 1.5831 | 1.5735 |
| lr1 0.0002, lrb 0.0002 | 1.6079 | 1.5887 | 1.5850 | 1.7193 | 1.6010 | 1.5856 | 1.5796 |
| lr1 0.001, lrb 0.0001 | 1.6030 | 1.5872 | 1.5829 | 1.6606 | 1.6031 | 1.5826 | 1.5726 |
| lr1 0.001, lrb 0.0002, lr2 0.002 | 1.6023 | 1.5829 | 1.5764 | 1.6539 | 1.6010 | 1.5816 | 1.5673 |
| … lr2 0.01 | 1.6043 | 1.5950 | 1.5934 | 1.6716 | 1.6010 | 1.5837 | 1.5781 |
| … lr2 0.002, apm 0.01 | 1.6070 | 1.5872 | 1.5847 | 1.6641 | 1.6010 | 1.5851 | 1.5736 |
| … lr2 0.001 | 1.6021 | 1.5816 | 1.5742 | 1.6519 | 1.6010 | 1.5814 | 1.5658 |
| … lr2 0.0005 | 1.6020 | 1.5811 | 1.5731 | 1.6511 | 1.6010 | 1.5813 | 1.5651 |
| … lr2 0.002, apm 0.03 | 1.6002 | 1.5829 | 1.5755 | 1.6519 | 1.6010 | 1.5801 | 1.5661 |
| … lr2 0.002, ssef 0 (fixed average) | 1.6084 | 1.5829 | 1.5757 | 1.6544 | 1.6010 | 1.5824 | 1.5647 |
| … lr2 0.002, order-2 APM 2^20 (+415 MB) | 1.6022 | 1.5829 | 1.5764 | 1.6539 | 1.6010 | 1.5816 | 1.5672 |
| … lr2 0.002, decay 1e-4 | 1.6023 | 1.5783 | 1.5695 | 1.6660 | 1.6007 | 1.5803 | 1.5686 |
| … lr2 0.0005, apm 0.03 | 1.5999 | 1.5811 | 1.5721 | 1.6489 | 1.6010 | 1.5798 | 1.5640 |
| … lr2 0.0005, apm 0.03, decay 1e-4 | 1.5999 | 1.5788 | 1.5678 | 1.6629 | 1.6007 | 1.5785 | 1.5663 |
| … lr2 0.0005, apm 0.03, decay 3e-5 | 1.5999 | 1.5769 | 1.5672 | 1.6507 | 1.6001 | 1.5785 | 1.5641 |
| lr1 0.002, lr2 0.0005, apm 0.03, decay 1e-4 | 1.5997 | 1.5760 | 1.5669 | 1.6394 | 1.6007 | 1.5789 | 1.5647 |
| **lr1 0.001, lrb 0.0002, lr2 0.0005, apm 0.03, ssef 0** | 1.6050 | 1.5811 | 1.5728 | 1.6507 | 1.6010 | 1.5800 | **1.5621** |
| lr1 0.001, lrb 0.0002, lr2 0.0002, apm 0.03 | 1.5999 | 1.5809 | 1.5716 | 1.6488 | 1.6010 | 1.5798 | 1.5637 |

What tuning showed:
- Every mixer wants small steps. X0's rates were 2–10× too large. The geometric mixer was worse than Fn until lrb fell to 0.0002.
- A bigger order-2 APM (2^20 contexts, +415 MB) gains nothing.
- A decaying step helps As alone but not ABs.
- Mixing As and Bs per bit (ABs) is the best arm in every row: the two mixers make different errors.

**Chosen for X1** (by the headline arm's first half): **ABs**, with lr1 0.001, lr2 0.0005, lrb 0.0002, APM rate 0.03, SSE combined by a fixed average (ssef 0), order-2 APM 2^18, no decay. They are now the defaults in `Params`. On the first half ABs is 1.5621, against F's 1.6333 (−0.071).

Before X1 the code was read again for any use of the true byte before its prediction:
- the chain's and the final layer's weights learn only after the whole distribution is formed;
- each bit's inputs, copy's state and the APM contexts read only the partial byte and earlier bytes;
- the selftest checks that a bitwise arm's 256 probabilities sum to 1.

## X1: the registered read

`context8k --full-mix` with the chosen defaults, one run. The first X1 was killed when WSL restarted, before it printed any `full_mix` line. It was rerun exactly as registered (same binary, same flags). The PC was less loaded this time (load about 15).

Bits per byte:

| Arm | First half | **Second half** | Against F (second half) |
|---|---:|---:|---:|
| F: the final layer's replica | 1.6333 | 1.6232 | — |
| Fn: F normalised | 1.6329 | 1.6230 | −0.0002 |
| Fs: Fn + SSE | 1.6050 | 1.5926 | −0.0306 |
| A: bitwise logistic mixer | 1.5811 | 1.5715 | −0.0517 |
| As: A + SSE | 1.5728 | 1.5618 | −0.0614 |
| Rs: raw experts only, bitwise + SSE | 1.6507 | 1.6105 | −0.0127 |
| B: geometric mixing | 1.6010 | 1.5870 | −0.0362 |
| Bs: B + SSE | 1.5800 | 1.5655 | −0.0577 |
| **ABs: As and Bs mixed per bit (headline)** | 1.5621 | **1.5499** | **−0.0733** |

- `full_mix_bpc 1.5499 (arm ABs)`. B0's `online_final_bpc` is 1.6232.
- The replica chain equals the chain on every byte (maximum difference 0), and F equals `online_final_bpc` (1.6232).
- Every line outside the `full_mix` lines is identical to B0's, the TALLY and GEN lines included. Selftest passes.
- **Wall:** 1834 s in all; the full-mix pass took 796 s of it. That is about +77% over the rest of the run, most of it the experts computed a second time. B0's 3403 s ran under far heavier load, so the two totals are not comparable.
- **Memory:** peak RSS 472 MB, against B0's 250 MB (+222 MB). The mixers hold 178 MB, mostly the four SSE stages' APM tables.
- There is no bootstrap interval: the per-byte bits are not kept. The gain is about 18 times the widths of earlier intervals on this holdout (about ±0.004).

**The rule holds.** All four conditions are met:
1. 1.5499 is 0.0733 below 1.6232 (at least 0.005 needed).
2. The answers are unchanged.
3. Selftest passes.
4. The wall time and memory are stated above.

So `--full-mix`'s ABs line becomes `context8k`'s headline bits per byte. The flag stays off by default here: it costs the extra pass and 222 MB, and the cloud session decides the default at merge.

**Predictions:**
- P1 (F and the chain replica exact): **held**.
- P2 (Fn − F between −0.005 and 0): **held** (−0.0002).
- P3 (Fs − F between −0.005 and −0.015): **missed**. SSE alone gains −0.031, twice the top of the range.
- P4 (As − F between −0.015 and −0.035): **missed**. The gain is −0.061, nearly twice the top.
- P5 (Bs within 0.01 of As): **held** (0.0037).
- P6 (Rs within ±0.03 of F): **held** (−0.013).

**Reading.**
- Normalising the linear mixture gains almost nothing: its mass is already 0.9998.
- The gain is nonlinear. SSE alone, after the linear final layer, takes 0.031 bits per byte. Logistic mixing with SSE takes 0.061. Mixing that with the geometric mixer takes 0.073.
- The bitwise mixer over the raw experts alone (Rs) beats the whole linear chain and final layer by 0.013. The chain and final layer are still worth having as inputs: As beats Rs by 0.049.

## Onto main's new default

X1 passed, so `origin/main` (6aed3eb) was merged into `wip/full-mix` as a merge commit (4c3954b), as the cloud session asked.

**What main changed in context8k:**
- an online n-gram expert (`OnlineNgram`, orders 1–12) as a chain stage before the LSTM, and as 9 more final-layer inputs (its mixed probability, then each of its 8 orders);
- the byte LSTM at hidden 256 (it was 64).

The final layer now has 35 inputs. Its `online_final_bpc` is 1.5793 on this PC, by the cloud session's report.

**What the replica needed:**
- **The n-gram's whole distributions.** `OnlineNgram::code()` predicts and learns in one call. A new `dist()` gives its mixed and per-order distributions over all 256 bytes from the same state, learning nothing: a slot that another context holds reads as unseen, as `claim()` would leave it. The hook then calls `code()` to learn the byte, and checks that `dist()` gave the same probability (`ngram_dist_max_abs_diff`).
- The n-gram stage in the replica chain, and its 9 inputs in the replica final layer, in main's order.
- The LSTM at `--lstm-size` and `--lstm-window`.
- The PAQ mixers read the n-gram too: its mixed distribution and each order's are 9 more inputs (33 in all; 25 raw).
- The dump is FMX2 (the n-gram's 9 distributions after the LSTM's). `--full-mix` refuses `--doc-boost` (off by default), since the replica does not scale the steps.

**Check run (X1m), before X2 is registered.** It runs side by side with M0, main's own binary, and the PC is nearly idle (load about 2).
- X1m: `context8k --full-mix --full-mix-dump x1m.fmx` on the merged code.
- M0: `context8k` from origin/main 6aed3eb, unchanged.
- **Checks:**
  - every line M0 prints, X1m prints identically;
  - arm F equals M0's `online_final_bpc` exactly;
  - the replica chain equals the chain;
  - the n-gram's `dist()` equals `code()`.
- Only these and the first half are read from X1m; its second half is not read before X2.

**Check run results** (M0 and X1m side by side; load about 2):

| | M0 (main) | X1m (merged, `--full-mix`, dump) |
|---|---:|---:|
| `online_final_bpc` | 1.5793 (chain 1.6103) | 1.5793 (chain 1.6103) |
| Wall | 1726 s | 3526 s (the pass 1750 s, with the 28 GB dump) |
| Peak RSS | 874 MB | 1097 MB |

- 0 lines differ outside the `full_mix` lines.
- Arm F is 1.5793, the same as `online_final_bpc`.
- The replica chain, the chain after copy and the n-gram's `dist()` all match exactly (maximum differences 0).
- The replica's mass is 0.99989.

First half only:

| | F | Fn | Fs | A | As | Rs | B | Bs | ABs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X1m, first half | 1.5978 | 1.5975 | 1.5747 | 1.5449 | 1.5396 | 1.5970 | 1.5557 | 1.5408 | 1.5272 |

## X2: registered (before the run)

- **Run:** `context8k --full-mix` on the merged code (4c3954b), the same settings as X1, no new tuning. One run, alone on the PC as far as other sessions allow.
- **X2 repeats X1m's arithmetic.** The code is deterministic, and the dump changes nothing. So its second half will equal X1m's, which was written but has not been read. X2 is run anyway, for a clean wall time without the dump.
- **Rule** (X1's, against the new default). `--full-mix`'s ABs line becomes `context8k`'s headline on main's default if all hold:
  1. X2's `full_mix_bpc` is at least 0.005 below M0's `online_final_bpc` (1.5793);
  2. every TALLY and GEN line is identical to M0's;
  3. selftest passes;
  4. the wall time and peak memory are stated.
- **Prediction P7.** X2's `full_mix_bpc` is 1.502–1.512 (−0.067 to −0.077 against 1.5793). On the first half the gain is −0.071, as X1's was (−0.071 first half, −0.073 second). The n-gram adds an order-mixing model that the bitwise mixer partly duplicated before, so the gain should not grow.
- **P8.** As − F and Bs − F are each within 0.01 of X1's (−0.061 and −0.058).

## What it would take for `sample` to write from the same mixture

Not built: `sample` is unchanged, and K5 is the writer. This is what it would need.

1. **A non-learning query.** Today `FullMixer::step` predicts, scores and learns in one call, along the true byte's path. A writer needs `dist(ByteIn, P[256])` that learns nothing. For the bitwise arms that means the 255 bit probabilities of the partial-byte tree, each from the mixers and APMs without updating them. Or it samples bit by bit, so only the 8 nodes on the sampled path are needed (about 8/255 of the cost). Both are exact: a bitwise arm is a proper distribution by construction (the selftest checks it). Geometric mixing (B) already gives P[256].
2. **The experts' whole distributions at every written byte.** `sample` already builds most of them (the trie, KN, InfiniGram 8192, class, similar-context, copy, doc). It would also need:
   - InfiniGram at caps 4–512;
   - the online n-gram's and the LSTM's distributions over the prompt and the text written so far, as `WriterExpert` does for the GLN and LSTM (save and restore their state around each continuation);
   - the chain replica with online λs, and the final layer's weights.
3. **Its learned state, carried over from the holdout and frozen.** After `context8k` (or `sample`'s own pass over the holdout), the mixers' weights, the APM tables, copy's confidences, the chain's λs and the final layer's weights would be kept and read without learning while writing (178 MB, mostly APM tables).
4. **The writing problems seen before still apply.**
   - The mixture is fitted for log loss on raw enwik8, markup and all. §13.15–13.18 found that experts which help prediction (the GLN, the LSTM, the online λs) lowered SCRIBE when sampled from.
   - SSE sharpens confident predictions. That may suit temperature 0.6, or may sharpen markup.
   - The test would be §13.16's: all 250 GEN-A dev prompts, SCRIBE v2, non-inferior to W0 (byte sampler), and compared with K5.
5. **Cost.** The pass costs about 0.8–1.7 ms per byte here (most of it the experts, which `sample` builds anyway). The mixers themselves cost about 0.2 ms per byte (the replay runs 1 MiB in about 200 s). Writing 600 bytes × 250 prompts would take a few minutes.

## X2: the result

`context8k --full-mix` on the merged code. The PC was nearly idle (load about 1–2). Bits per byte:

| Arm | First half | **Second half** | Against F (second half) |
|---|---:|---:|---:|
| F: the final layer's replica | 1.5978 | 1.5793 | — |
| Fn: F normalised | 1.5975 | 1.5793 | 0.0000 |
| Fs: Fn + SSE | 1.5747 | 1.5545 | −0.0248 |
| A: bitwise logistic mixer | 1.5449 | 1.5247 | −0.0546 |
| As: A + SSE | 1.5396 | 1.5189 | −0.0604 |
| Rs: raw experts only, bitwise + SSE | 1.5970 | 1.5484 | −0.0309 |
| B: geometric mixing | 1.5557 | 1.5312 | −0.0481 |
| Bs: B + SSE | 1.5408 | 1.5169 | −0.0624 |
| **ABs (headline)** | 1.5272 | **1.5051** | **−0.0742** |

- `full_mix_bpc 1.5051 (arm ABs)`, against M0's `online_final_bpc` 1.5793.
- F equals `online_final_bpc`. The replica chain and the n-gram's `dist()` match exactly.
- 0 lines differ from M0 outside the `full_mix` lines (TALLY and GEN included). Selftest passes.
- X2's arm lines equal X1m's, as registered.
- **Wall:** 3434 s, against M0's 1726 s (+99%). The full-mix pass took 1699 s, most of it the experts computed again (the n-gram's whole distribution and the 256-wide LSTM are the new costs).
- **Memory:** peak RSS 1098 MB, against M0's 874 MB (+224 MB).

**The rule holds.** All four conditions are met:
1. 1.5051 is 0.0742 below 1.5793 (at least 0.005 needed).
2. The answers are unchanged.
3. Selftest passes.
4. The wall time and memory are stated above.

**Predictions:**
- P7 (1.502–1.512): **held** (1.5051).
- P8 (As − F and Bs − F within 0.01 of X1's): **held** (−0.0604 against −0.0614; −0.0624 against −0.0577).

**Reading.**
- The gain does not shrink on the stronger default: −0.073 on the old default, −0.074 on the new.
- The raw-input bitwise mixer (Rs) now beats the linear final layer by 0.031 (0.013 before). With the n-gram's orders as inputs, it has more to mix.

## Summary

| Default | `online_final_bpc` | `full_mix_bpc` (ABs) | Gain | Extra wall | Extra memory |
|---|---:|---:|---:|---:|---:|
| Old (B0; X1) | 1.6232 | 1.5499 | −0.0733 | +796 s pass (+77%) | +222 MB |
| Main's, with the n-gram and LSTM 256 (M0; X2) | 1.5793 | **1.5051** | **−0.0742** | +1699 s pass (+99%) | +224 MB |

**What became of each idea:**
- Normalising the linear mixture: no gain (its mass is already 0.9998).
- SSE after the linear final layer: −0.025 to −0.031.
- Bitwise logistic mixing with SSE: −0.060.
- Geometric mixing with SSE: −0.058 to −0.062. It needs a small step (lrb 0.0002); at 0.002 it was worse than the linear mixture.
- Both, mixed per bit: −0.073 to −0.074.
- A bigger order-2 APM and decaying steps: no gain.

**What should become default.** By the rule, ABs's line becomes `context8k`'s headline bits per byte. The flag is off in this branch. Turning it on doubles `context8k`'s wall time, so that is the cloud session's choice at merge.
- One way to keep the cost down: fold the full-mix pass into the final layer's loop. That loop already calls `fill_experts`, class and copy for every byte; the extra InfiniGram, similar-context, LSTM and n-gram passes could keep their whole distributions instead of only the true byte's.

## Risks before merging

- **One holdout.** Every number is the same 1 MiB of enwik8. The settings were tuned on its first half, and the scored second half follows it. The first and second halves agree (−0.071 and −0.073), but no other text has been read.
- **No interval.** The per-byte bits are not kept, so there is no bootstrap. The gain is far larger than earlier intervals on this holdout (about ±0.004), but it is not measured here.
- **Cost.** `context8k --full-mix` doubles the wall time and adds about 220 MB. Anything that runs `context8k` often (`scripts/cypha_report_card.sh`, the GLNX dump) would pay it if it became the default.
- **The replica must follow the chain.** F checks this on every run: any change to the chain or the final layer (a new stage, a new input, a new step rule) must be mirrored in `full_mixer.hpp`, or arm F stops equalling `online_final_bpc`. `--doc-boost` and `--gln` are refused for this reason. `--final-gate lstm` is refused too.
- **`OnlineNgram::dist`** is new, in another session's file. It matched `code()` exactly on all 1 MiB, but two nodes of one byte sharing a hashed slot could make them differ in rare cases.
- **Prediction, not writing.** Nothing here changes `sample`. Earlier experts that helped prediction lowered SCRIBE when sampled from (§13.15–13.18). See the section above.
- **Files.** The dumps are on this PC, `~/fullmix_data/x0.fmx` (18 GB) and `x1m.fmx` (28 GB), for more replay tuning. Run outputs: `/mnt/d/cypha_llm/full_mix/`.

## Optimisation and default

2026-11-05 · Odin Loch

**Odin:** "See if you can't optimise all those upgrades before going in." The task: make `--full-mix` cheap, then turn it on by default. Branch `wip/opt-fullmix`, from main 20701dd.

### Where the time went (before)

X2's full-mix pass took about 1700 s on one thread, on top of `context8k`'s own 1726 s. It computed every expert again:
- **a second byte LSTM** (hidden 256), trained from scratch over the holdout: about 900 s, half the pass;
- a second online n-gram, pretrained again (24 s, and 588 MB at its peak);
- a second similar-context expert (14 s to build);
- the static experts again: the trie's 5, Kneser–Ney, InfiniGram at 5 caps, class, copy;
- the mixers: about 150–180 s. Replay profile, s per 1 MiB: chain 4, final layer 15, bit inputs 13, **geometric mixing 83** (8,448 logs a byte), bitwise loop 21, learning 10.

### What changed

- **One LSTM and one n-gram.** With the full mix on, the n-gram's and the LSTM's byte loops run inside the full mix's row loop, on `cmd_context8k`'s own instances. The LSTM's `dist()` is read before its `step()`; the n-gram's `dist()` is read before its `code()`. The chain's `plstm` and `pngram` columns, and their bits, are computed by the same calls in the same order, so they are bit-identical. Their lines and chain stages then follow in the old order.
- **Rows computed ahead, on threads.** Rows come in blocks of 512. While the main thread mixes block b, producer threads compute block b + 1:
  - the static experts, split over 3 threads, each with its own InfiniGram cursors (a cursor only saves time; it never changes an answer);
  - similar-context on 1 thread, in order;
  - the LSTM on 1 thread;
  - the n-gram on 1 thread.

  That is 7 threads in all. Every value is computed by the same code as before, and every sum keeps its order.
- **Similar-context is still built again.** It learns online keys as it reads, so the chain's instance, which has read the whole holdout, would see the future. A fresh one reads the holdout in order: 14 s and about 20 MB.
- **Output order.** The full mix now flushes `std::cout` before it prints. Before, its lines could come out before `online_final_bpc` (X1's file has them at the end, for that reason).
- **Not changed:** the mixers themselves. The geometric mixer's logs are the mixer's largest part, but in the default run the main thread already waits for the LSTM (below), so speeding them up would not shorten the wall time.

### The default

- `context8k` runs the full mix by default. `--no-full-mix` turns it off.
- Where it cannot replay the chain, it switches itself off with one stderr note, and the old lines print as before. That is `--gln`, `--doc-boost`, `--final-gate lstm`, `--frozen-mix`, `--no-online-final` or `--bkn-way a`.
- **A headline line, last:** `context8k_bpc X (SOURCE)`, the best line the run printed. That is the full mix's ABs, else `online_final_bpc`, else the last chain line. `online_final_bpc` and every other line stay.
- `bench/fuzzy/glnx_prep.sh` passes `--no-full-mix`. Its frozen-chain flags would switch the full mix off anyway. The flag changes its manifest key, so its next run redoes the dump once.
- `cyphalm_trie` now links `Threads::Threads` (CMake).

### The checks (exact, as the rule asks)

Runs in two waves; each wave's runs side by side under the same load (load about 8–10). Peak RSS from `/usr/bin/time`.

| Run | What | Wall | Peak RSS | Lines |
|---|---|---:|---:|---|
| M0b | main 20701dd, `context8k` | 2063 s | 874 MB | the reference (equal to M0) |
| X2b | main 20701dd, `--full-mix` | 3773 s | 1057 MB | the old full mix (equal to X2) |
| **N1** | this branch, `context8k` (full mix on) | **2075 s** | **1161 MB** | every arm line = X2's (ABs **1.5051**); every other line = M0b's |
| N0 | this branch, `--no-full-mix` | 1824 s | 875 MB | = M0b's, line for line, plus `context8k_bpc 1.5793 (online_final)` |
| N2 | this branch, `--no-ngram --lstm-size 64` (full mix on) | 1425 s | 515 MB | = X1's lines (ABs 1.5499); only the order of X1's arm lines differs (see above) |
| R64 | main 20701dd, `--no-ngram --lstm-size 64 --full-mix` | 1838 s | 472 MB | = X1's |

Wave 1 is M0b, X2b and N1; wave 2 is N0, N2 and R64. N0's 1824 s against M0b's 2063 s is load: the two waves ran at different times.

Selftest passes. On a small setup first (`--train 1048576 --sample 1179648`), the old and new full mix also gave the same lines (651 s against 419 s).

**The rule holds.**
- ABs stays 1.5051, and every arm line of X2 is unchanged.
- Every other line, GEN and TALLY included, is M0's.
- `--no-ngram --lstm-size 64` reproduces X1.
- Selftest passes.

### The profile after (N1)

Busy seconds by thread over the row loop (952 s of wall):

| Static experts (3 threads, summed) | Similar-context | LSTM | n-gram | Mixers (main thread) | Main thread waiting |
|---:|---:|---:|---:|---:|---:|
| 89 | 578 | **949** | 68 | 181 | 769 |

The mixers' part: chain 9, final layer 20, bit inputs 15, geometric 91, bitwise loop 26, learning 11.

- **The LSTM is the critical path.** The row loop takes as long as the LSTM alone did in M0 (943 s), so the full mix now costs about **+1% wall** (2075 s against 2063 s), against +83% before (X2b).
- **Memory is +287 MB** against M0 (1161 against 874 MB), and +104 MB against X2b. Before, the second n-gram was built after the first was freed. Now the one n-gram, the mixers' state (178 MB, mostly APM tables), the row blocks (48 MB) and the second similar-context expert are all live at once.
- **With `--lstm-size 64`**, similar-context (480 s, one thread, in order) becomes the critical path: 1425 s against 1838 s for main's `--full-mix` (−22%).

### What each caller now pays

- **`context8k` with no flags:** about the same wall as before (+1%), +287 MB, and 7 threads for about 16 minutes instead of 1.
- **`scripts/cypha_report_card.sh`:** its `ctx` part runs `context8k "$E" "$D" $F`, so it gets the full mix (the same cost as above).
  - Its reader (`cypha_report_card.cpp`, `kCtxPref`) still takes the old chain lines as `context8k`'s headline. It does not read `online_final_bpc` or `context8k_bpc`.
  - That was left as it is. Putting `context8k_bpc` first in `kCtxPref` would make the card show the headline, but it would change the card's numbers, so it is the cloud session's call.
- **`bench/fuzzy/glnx_prep.sh`:** unchanged cost (`--no-full-mix`).
- **`qa`, `sample`, `chat`:** unchanged. The full mix is only in `context8k`.

### Risks before merging

- **Threads.** `context8k` now uses 7 threads by default (3 static, similar-context, LSTM, n-gram, main). On a busy machine its wall depends on getting them. `--no-full-mix` gives the one-thread run.
- **Memory.** +287 MB at the peak. Halving the blocks (256 rows) would save 24 MB. The APM tables (138 MB) are the rest, and shrinking them would change the numbers.
- **Another session is speeding up `byte_lstm.hpp` and `online_ngram.hpp`** with byte-identical output and the same interfaces. The full mix calls only `dist()`, `step()`, `code()` and the n-gram's `dist()`, so a merge should not conflict. The arm lines are the check: N1's must stay X2's.
- **The replica still has to follow the chain.** Arm F checks it on every run (it must equal `online_final_bpc`).

### Files

The 46 GB of tuning dumps (`~/fullmix_data/x0.fmx`, `x1m.fmx`) were deleted after these runs; `full-mix-replay` can remake them with `--full-mix-dump`. The run outputs stay in `/mnt/d/cypha_llm/full_mix/`.
