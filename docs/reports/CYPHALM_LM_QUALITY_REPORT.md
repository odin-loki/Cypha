# CyphaLM as a language model: next-byte distribution quality

**Date:** 2026-09-23
**Harness:** `native/tools/cyphalm_lm_quality.cpp`
**Pretraining:** enwik8, first 8 MiB, mem 22, one online pass (`gate24` and `lean` tiers)
**Held-out:** enwik8 from byte 96,000,000 (wiki, in-domain; never seen in pretraining)
and *Alice in Wonderland* (`alice29.txt` from byte 20,000; plain English, out of domain).
16 KiB per eval. Raw JSON: [`lm_quality/`](lm_quality/).

The compression reports measure bits to code a file from scratch. For LLM use
the question is different: after pretraining, how good is the full next-byte
distribution on text the model has not learned from, and can we decode it into
good text?

## Method

At every held-out byte the harness takes the full 256-way distribution
(`serve_next_byte_log_probs`), scores the true byte (NLL, rank, top-1
confidence, entropy), and then lets the model read the true byte. That is
"online" (in-context) evaluation: the model adapts to the text it is reading, as
it would to a prompt. `--frozen-eval` turns learning off, so it measures what
the pretrained model knows on its own.

Generations continue from a 256-byte held-out prompt. Each is scored by a
**judge**: a fresh copy of the model reads the prompt, then scores the
continuation with learning off (bits/byte). The true continuation is scored the
same way as the **reference**. Good text should sit near the reference on both
judge bits and distinct 4-grams (`d4`). Very low judge bits with low `d4` mean a
copy loop; high judge bits mean gibberish.

## Distribution quality (held-out, 16 KiB)

| | wiki gate24 | wiki lean | Alice gate24 | Alice lean |
|---|---:|---:|---:|---:|
| NLL, bits/byte (online) | 1.7947 | 1.7950 | 2.0946 | **2.0852** |
| NLL, frozen (pretrained only) | — | 2.0576 | — | 2.8871 |
| top-1 / top-5 | 63.1% / 88.1% | 63.1% / 88.1% | 59.4% / 85.9% | 59.6% / 85.8% |
| mean entropy, bits | 1.866 | 1.863 | 2.133 | 2.126 |
| ECE of top-1 confidence | 1.4% | 1.4% | 1.2% | 1.1% |
| NLL-optimal temperature | 1.0 | 1.0 | 1.0 | 1.0 |
| bit-greedy byte = argmax byte | 84.9% | 85.2% | 86.0% | 86.3% |

- **Calibrated as-is.** Reliability bins track the diagonal (e.g. lean wiki:
  confidence 0.25 → accuracy 0.26, 0.55 → 0.56, 0.97 → 0.97; slightly
  underconfident around 0.75–0.85), and rescaling by any temperature only
  hurts. No calibration layer is needed.
- **In-context learning is most of the value**: reading the text is worth
  0.26 bits/byte in-domain and 0.80 on plain English. This is hp's online
  learning acting as in-context adaptation.
- `lean` (the smaller, faster tier) is as good in-domain and slightly better
  out of domain.
- **Greedy was not greedy.** The old O(8) greedy walked the bit tree bit by bit
  and picked a different byte from the argmax 14–16% of the time here (20–24% on
  smaller models). `DecodeParams::exact_greedy` (default on) now takes the true
  argmax.

### Frozen scoring (serve default)

Scoring from the model as it stands at the byte boundary, with no learning
inside the hypothetical byte, instead of hp's compression semantics:

| | exact NLL | frozen NLL | Δ | exact ms | frozen ms | speedup |
|---|---:|---:|---:|---:|---:|---:|
| wiki gate24 | 1.7947 | 1.7996 | +0.0049 | 8.69 | 4.15 | 2.1× |
| wiki lean | 1.7950 | 1.7983 | +0.0033 | 7.47 | 3.12 | 2.4× |
| Alice gate24 | 2.0946 | 2.1023 | +0.0077 | 8.52 | 3.91 | 2.2× |
| Alice lean | 2.0852 | 2.0904 | +0.0052 | 7.48 | 3.04 | 2.5× |

`CyphaLMConfig::hp_frozen_scoring` defaults on (`CYPHA_HP_FROZEN_SCORING=0` for
exact). Training and `eval_bpc` are unaffected. The distribution still sums to
exactly 1, and scoring leaves the model untouched (`hp_frozen_smoke`).

## Generation

Lean, 8 MiB pretrain, 300 bytes. Reference: wiki judge 1.92–1.97 / d4 0.89;
Alice judge 2.42–2.79 / d4 0.89–0.91.

| decode | wiki judge / d4 | Alice judge / d4 | what it looks like |
|---|---|---|---|
| greedy (learn from output) | 0.79–0.91 / 0.03–0.29 | 0.36–0.42 / 0.14 | loops a phrase from the prompt |
| greedy, no-repeat 16 | 3.39 / 0.20 | **0.46 / 0.75** | Alice: copies real later text; wiki: breaks words |
| greedy, frozen, no-repeat 16 | 1.65 / 0.29 | 1.28 / 0.40 | loops with variations |
| top-p 0.9, T 0.8 (learn) | 2.09–2.54 / 0.12–0.23 | 0.92–1.45 / 0.05–0.08 | collapses into a repeated fragment |
| min-p 0.1, T 0.8, learn | 0.99 / 0.09 | 0.66 / 0.14 | loops |
| min-p 0.1, T 0.8, frozen | 4.55 / 0.92 | 4.13 / 0.95 | letter salad |
| stop self-indexing (`index_output=false`) | no change | no change | loops live in the context models too |

Two failure modes:

1. **With online learning on its own output,** the model reinforces whatever it
   just wrote and falls into loops. Greedy on any fixed-context model cycles
   anyway.
2. **Frozen,** it does not loop, but sampled text drifts into letter salad. For
   text sampled from a calibrated model, judge bits should be near the model's
   entropy (~2–2.7). They come out at 4+, because after a few sampled bytes the
   context is one hp has never counted, and its hashed-context statistics carry
   no information there. hp does not generalise across contexts.

So the next-byte distributions are excellent where the model has evidence
(which includes everything in the prompt), and near-uninformative in novel
contexts. Better decoding can't fix that. The model needs a component that
generalises, which is why a small LSTM expert was added
([below](#lstm-expert)).

## LSTM expert

`ByteLstm` (embed → LSTM(H) → softmax over 256 bytes, truncated BPTT 20, Adam)
trains online next to hp. `ByteMixGate` mixes the two distributions per byte:
w = σ(θ·[1, H_hp, H_lstm, max log p_hp, max log p_lstm]), learned online by
log-loss. Where hp is sharp it keeps hp; where hp is flat and the LSTM is not,
the mix can move toward the LSTM.

Pretraining with the LSTM on 8 MiB is running; results will be added here. At 300 KB of pretraining the LSTM alone is far weaker than hp (Alice 3.78 vs 2.75 bits/byte) and the mix gains nothing yet (2.750); the gate correctly keeps w ≈ 0.87 on hp.

## Decode controls added

| `DecodeParams` | default | effect |
|---|---|---|
| `exact_greedy` | on | greedy = argmax of the full distribution |
| `learn_from_output` | on | learn from generated bytes (off = context only) |
| `index_output` | on | index generated bytes for match copying |
| `min_p` | 0 | drop bytes with p < min_p · p_max |
| `no_repeat_ngram` / `no_repeat_window` | 0 / 256 | ban bytes that repeat an n-byte sequence in the window |

## Reproduce

```bash
Q=native/build/cyphalm_lm_quality
$Q --tier lean --train enwik8 --train-bytes 8388608 --save /tmp/pre_lean
$Q --load /tmp/pre_lean.json --eval enwik8 --eval-offset 96000000 --eval-bytes 16384 --compare-scoring
$Q --load /tmp/pre_lean.json --eval alice29.txt --eval-offset 20000 --eval-bytes 16384 --frozen-eval --gen-bytes 0
```
