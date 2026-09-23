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

**Bug found and fixed first.** Every decoder primed the prompt through
`prime_serve_context`, which called `reset_context()`. That replaced the hp
predictor with an untrained one, so generation (and REST `/generate`) ignored
all training and ran on a model that had read only the prompt. A first round
of measurements made under that bug looked like "letter salad, hp cannot
generalise"; that conclusion was wrong and is withdrawn (its raw data is kept
as `g2_*` / `g3_*`). Priming now uses `reset_stream()`, which keeps every
learned table (`hp_frozen_smoke` checks it).

With the fix, lean pretrained on 8 MiB, 300 bytes from a 256-byte held-out
prompt:

| decode | wiki judge / d4 | Alice judge / d4 |
|---|---|---|
| **reference (true continuation)** | 1.97 / 0.89 | 2.79 / 0.91 |
| greedy | 0.99 / 0.04 | 0.42 / 0.03 |
| greedy, no-repeat 8 | 1.48 / 0.52 | 1.25 / 0.62 |
| top-p 0.9, T 0.8, learning from output (old default) | 1.29 / 0.59 | 0.71 / 0.23 |
| min-p 0.1, T 0.8, learning from output | 1.33 / 0.58 | 0.67 / 0.25 |
| **min-p 0.1, T 0.8, frozen output (new default)** | **1.45 / 0.89** | **1.69 / 0.87** |
| min-p 0.2, T 1.0, frozen | 1.51 / 0.90 | 1.35 / 0.78 |

Samples:

- wiki, new default: *"…with there a link which space of such that time
  serialisms several council of the medical announcing the married the school…"*
- Alice, new default: *"…winter out the Mouse said in [[Algeria]].'
  Maryland able of it way in a map of struggle in the introduction…"*
- wiki, greedy: *"…the set the set the set the set…"*

What this shows:

- **The model now writes words** at the same distinct-4-gram diversity as real
  text (0.87–0.89). The sentences don't hold together yet, and on Alice it
  drifts into wiki style (its pretraining domain). Judge bits sit below the
  reference because the samples are, by construction, what the model finds
  likely.
- **Learning from its own output makes it loop** (d4 0.23–0.59). Keeping
  learned statistics fixed while it writes (`learn_from_output=false`) removes
  the loops. Prompt bytes are still learned.
- **Greedy always cycles**, as argmax decoding of any finite-context model will.
  No-repeat helps diversity but breaks words at byte level. Use sampling.
- New `DecodeParams` defaults: temperature 0.8, `min_p` 0.1,
  `learn_from_output` false.

## LSTM expert (tried, removed)

hp predicts only from contexts it has counted. A recurrent model generalises
across contexts, so a small float32 byte LSTM (byte → LSTM(128) → softmax,
truncated BPTT 20, Adam; 0.17 ms/byte) was trained online next to hp. The two
distributions were mixed per byte by an online-learned gate on their entropies
and peaks. Pretraining: the same 8 MiB, about an hour for the LSTM.

| held-out | hp | LSTM alone | gated mix |
|---|---:|---:|---:|
| wiki NLL, bits/byte | 1.7950 | 2.6606 | 1.7984 (+0.003) |
| Alice NLL, bits/byte | 2.0852 | 3.9156 | 2.0887 (+0.004) |
| wiki / Alice top-1 | 63.1% / 59.6% | 48.1% / 46.0% | 62.9% / 59.7% |

The gate learns to trust hp (w ≈ 0.95), and generations from the mix look
like hp's own. An LSTM this small, trained this briefly, is never better than
hp often enough to pay for mixing. cmix-class gains need far larger LSTMs
trained far longer than this CPU budget allows. **Removed** along with the
`index_output` decode switch, which measured no effect. Recover both from
commit `ee1325c` (`native/include/cypha/cyphalm/byte_lstm.hpp`,
`native/src/cyphalm/byte_lstm.cpp`). Raw data: `lm_quality/m_*.json`.

## Decode controls added

| `DecodeParams` | default | effect |
|---|---|---|
| `exact_greedy` | on | greedy = argmax of the full distribution |
| `learn_from_output` | **off** | learn from generated bytes (off = context only) |
| `min_p` | **0.1** | drop bytes with p < min_p · p_max |
| `temperature` | **0.8** | |
| `no_repeat_ngram` / `no_repeat_window` | 0 / 256 | ban bytes that repeat an n-byte sequence in the window |

## Reproduce

```bash
Q=native/build/cyphalm_lm_quality
$Q --tier lean --train enwik8 --train-bytes 8388608 --save /tmp/pre_lean
$Q --load /tmp/pre_lean.json --eval enwik8 --eval-offset 96000000 --eval-bytes 16384 --compare-scoring
$Q --load /tmp/pre_lean.json --eval alice29.txt --eval-offset 20000 --eval-bytes 16384 --frozen-eval --gen-bytes 0
```
