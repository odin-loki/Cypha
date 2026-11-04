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
