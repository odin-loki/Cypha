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
