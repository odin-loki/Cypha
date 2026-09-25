# CyphaLM as a language model: next-byte distribution quality

**Date:** 2026-09-23 to 2026-09-24 (last commit covered: `2a003bf`)
**Harness:** `native/tools/cyphalm_lm_quality.cpp`
**Pretraining:** enwik8, first 8 MiB, mem 22, one online pass (`gate24` and `lean` tiers);
later sections state their own (95 MB, shards, `slim`)
**Held-out:** enwik8 from byte 96,000,000 (wiki, in-domain; never seen in pretraining)
and *Alice in Wonderland* (`alice29.txt` from byte 20,000; plain English, out of domain).
16 KiB per eval unless a table says 8 KiB. Raw JSON: [`lm_quality/`](lm_quality/).
Tools, flags, config fields, env vars, file formats, changelog and negative
results: [Reference](#reference) at the end.

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
  `learn_from_output` false. (Word lookahead, added later, now runs on top of
  these by default: see *Serving improvements*.)

## Scaling with pretraining data

Lean, 95 MB of enwik8 (bytes 0–95,000,000; ~2.3 h at ~11 kB/s, 1.08 GB) against
the 8 MiB pretrain. The held-out text is unchanged: wiki starts at byte 96,000,000,
past all training bytes.

| held-out | 8 MiB | 95 MB | Δ |
|---|---:|---:|---:|
| wiki NLL, bits/byte | 1.7950 | 1.7312 | −0.064 |
| wiki top-1 / top-5 | 63.1% / 88.1% | 64.3% / 88.6% | |
| Alice NLL, bits/byte | 2.0852 | 2.0587 | −0.027 |
| Alice top-1 / top-5 | 59.6% / 85.8% | 60.1% / 86.2% | |
| ECE, wiki / Alice | 1.4% / 1.1% | 1.6% / 1.0% | |
| online training bpc | 1.6099 | 1.5058 | |

About 12× the data buys a real but modest gain, mostly in-domain. Generation
with the default decoder becomes more on-topic without becoming coherent. wiki:
*"…in size is a necessary to have reduced insects chambers are communicantly…"*
(judge 1.26, d4 0.86); reference judge 1.74, d4 0.91. Greedy and
learn-from-output decoding still loop. Raw data: `lm_quality/s95_*.json`.

## Serving improvements (second round)

Three changes to how the pretrained model is served, each measured on the
lean 8 MiB model with held-out text.

### Keep the byte history when starting a prompt

`prime_serve_context` began a new stream with `reset_stream()`, which also
wiped the byte ring (32 MB of the most recent text) that the match models copy
from. Frozen held-out NLL, 8 KiB:

| start of eval | wiki | Alice |
|---|---:|---:|
| continue the training stream | 2.0285 | 2.9380 |
| `reset_stream()` (old priming) | 2.0606 (+0.032) | 2.9585 (+0.021) |
| `reset_stream(keep_history=true)` (new priming) | 2.0285 | 2.9379 |

Priming now resets per-stream contexts but keeps the history.

### Adapt to the prompt at half the trained mixer rate

`hp::Predictor::set_serve_adaptation` scales the mixer learning rates at serve
time (idempotent; checkpoints always store trained rates). Online held-out NLL
with the mixer at a fraction of its trained rate:

| text (8-16 KiB) | ×1 (trained) | **×0.5** | ×0.25 | ×1.5 | ×2 | ×3 |
|---|---:|---:|---:|---:|---:|---:|
| enwik8 @96M | 1.8259 | **1.8212** | 1.8214 | 1.8325 | 1.8450 | 1.8728 |
| Alice | 2.2532 | **2.2467** | 2.2460 | 2.2638 | 2.2809 | 2.3199 |
| enwik8 @97.5M (check) | 1.7517 | **1.7466** | | | | |
| lcet10 (check) | 1.6252 | **1.6237** | | | | |
| plrabn12 (check) | 2.2750 | **2.2666** | | | | |

Faster adaptation is worse everywhere and half speed is better on all five
texts (−0.0015 to −0.008). Removing the mixer's small-error skip adds nothing
reliable. `CyphaLMConfig::hp_serve_mixer_lr_scale = 0.5` (env
`CYPHA_HP_SERVE_MIXER_LR_SCALE`) is applied by `CyphaLMModel::set_serve_mode`:
generation turns it on, training entry points turn it off.

These rates were integers with a floor of 1: ×0.5 turned the layer-1 rates
2 / 3 / 4 into 1 / 1 / 2 and ×0.25 turned all of them into 1, so the two
columns above are nearly the same setting. On the lr1_scale 40 shards of
winners v2 and v3 (layer-1 rates 1-2) ×0.5 changed only the final layer and
the two rate-2 sets. Since the serve-state fix the effective rates run in
1/16ths (`hp::MixerNet`, update `>> 18` instead of `>> 14`), bit-identical
at ×1 (8 MiB gate24 online bpc and the v3 wiki 1.5927 unchanged); ×0.5 now
halves every rate. Wiki 16 KiB at ×0.5 (`--serve-lr 2`), floored → real:
v3 winner 1.5942 → 1.5954 (×1: 1.5927); gate24 table bits 18 with the
upstream mixer gains, 8 MiB: 1.8142 → 1.8173. On v3 ×0.5 is worse than ×1
either way, so the 0.5 serving default needs re-measuring for lr1_scale 40
models (upgrade plan candidate 4).

### Word lookahead decoding

Byte sampling writes real words but invents some ("communicantly") and loses
the thread. Word lookahead samples K candidate words (up to and including the
next delimiter) with the normal byte-level settings, rewinds after each, and
keeps the candidate with the highest mean log-probability that does not repeat
a 12-byte sequence of the recent text (without that ban, best-of-K loops:
distinct 4-grams 0.23–0.53).

Rewinding must be exact and cheap. `hp::StreamRewind` saves the predictor's
inline state (~170 KB) and records heap writes in an undo frame that also
covers byte boundaries. Learned tables are not written while learning is off,
so that is everything that changes. `hp_stream_rewind_smoke` checks that full
checkpoints are byte-identical after 60 random rewinds, with bit-tree and serve
scoring nested inside, and that 1000 later distributions match an untouched
copy. A first version copied the model (1.1 GB) per candidate and ran at
~400 ms/byte. With rewinds, and with candidates sharing each prefix's
distribution (they start from the same state and often agree), generation
latency from `cyphalm_generate` (300 bytes, lean, one core) is:

| decode | ms/byte |
|---|---:|
| byte sampling | 1.8 |
| word lookahead K 4 | 5.6 |
| word lookahead K 8 (default) | 12.7 |

400 bytes from the held-out prompts (judge bits / distinct 4-grams):

| decode | wiki | Alice |
|---|---|---|
| reference (true continuation) | 1.68 / 0.86 | 2.17 / 0.92 |
| byte sampling (min-p 0.1, T 0.8) | 1.10 / 0.68 | 1.59 / 0.86 |
| word lookahead K 4 | 0.86 / 0.69 | 1.22 / 0.73 |
| **word lookahead K 8 (new default)** | 0.70 / 0.74 | 0.96 / 0.72 |

Samples, K 8:

- wiki: *"…organizations of the abdominal spiracle and associated trachea of
  caterpillars in the administration of the insects are the internal existing
  the trachea are often the tracheole cell basement membrane. Since they are
  considered the heart relationship between the spiracles…the hemolymph seeps
  back into the heart."*
- Alice: *"…and began to explain the direction and began in the supported by
  the Dodo said to herself, and was high time with supporter the said the
  obligations and began wrapping itself up very career and discovered the pair
  of white kid…"*
- byte sampling, same wiki prompt: *"…the tracheal the classicized from office
  in the tracheal of the trachea little and members of series of the
  exoskeleton the exoskeleton the exoskeleton…"*

Phrases now hold together over several words and stay on topic. Whole
sentences still don't. Judge bits fall well below the reference because the
decoder picks what the model finds likely, so read them with distinct
4-grams. `DecodeParams::word_candidates` (default 8; 0 = off),
`word_no_repeat` (12). CLI `--word-candidates`; REST `word_candidates`,
`word_no_repeat`; streaming emits the bytes once a word is chosen.

## Ensembles of shard models

The largest gain in held-out quality this round came from ensembles, not a
bigger model. hp models trained independently on **disjoint** slices of the
data and mixed geometrically (normalised weighted mean of log probabilities)
beat one model trained on all of those slices at once. Held-out NLL, 8 KiB,
lean tier; shards are 8 MiB slices of enwik8 at 0, 16, 32, 48 and 80 MB:

| model | data | RAM | wiki | Alice | top-1 wiki |
|---|---|---:|---:|---:|---:|
| 1 × 8 MiB | 8 MiB | 1.1 GB | 1.8259 | 2.2532 | 63.1% |
| 1 × 16 MiB (both slices) | 16 MiB | 1.1 GB | 1.7746 | 2.2704 | 64.0% |
| **2 × 8 MiB ensemble** | 16 MiB | 2.2 GB | **1.7583** | **2.2060** | 64.8% |
| 4 × 8 MiB ensemble | 32 MiB | 4.4 GB | 1.7248 | 2.1906 | 64.9% |
| 5 × 8 MiB ensemble | 40 MiB | 5.5 GB | 1.7168 | 2.1841 | 65.0% |
| 1 × 95 MB | 95 MB | 1.1 GB | 1.7608 | 2.2158 | 64.1% |
| 1 × 95 MB, table bits 24 | 95 MB | 2.0 GB | 1.7234 (16 KiB) | 2.0605 (16 KiB) | |
| 95 MB + 5 × 8 MiB | | 6.6 GB | **1.6978** | **2.1642** | 65.4% |
| 2 × 8 MiB, table bits 20 | 16 MiB | 1.3 GB | 1.7624 | 2.2082 | 64.7% |
| 4 × 8 MiB, table bits 20 | 32 MiB | 2.6 GB | 1.7302 | 2.1925 | 64.9% |
| **11 × 8.6 MB, table bits 20 (all 95 MB)** | 95 MB | 7.2 GB | **1.6994** | **2.1797** | 65.4% |
| **95 MB + those 11 shards** | 95 MB | 8.3 GB | **1.6860** | **2.1636** | 65.4% |

The last two rows use learned mixing weights (below). The 95 MB split
into 11 slices of 8.6 MB, each trained as a small-table model, beats one lean
model trained on the same 95 MB by 0.061 on wiki and 0.036 on Alice. One of
those small slice models alone scores 1.8527 / 2.2855. Adding the 95 MB model to
the 11 shards gives the best distribution measured: wiki 1.686, Alice 2.164,
top-1 65.4%, ECE 2.1% / 1.3%.

- **Diversity, not capacity.** One model on 16 MiB gains 0.05 on wiki and
  *loses* on Alice; two models on the same 16 MiB gain 0.07 and 0.05. Four
  8 MiB shards beat the model trained on 95 MB. Quadrupling the 95 MB model's
  tables gains only 0.008 (wiki, 16 KiB: 1.7312 → 1.7234) and nothing on Alice,
  so that model is not table-bound either.
- **At equal memory**, two table-bits-20 shards (1.3 GB) beat one table-bits-22
  model (1.1 GB) by 0.064 on wiki and 0.045 on Alice.
- Returns diminish per shard (2 → 4 → 5 shards: −0.034, −0.008), but shards
  that cover more data keep helping (11 shards over all 95 MB: 1.699). Each
  member costs ~3 ms/byte of CPU.
- Weights: linear mixing is worse than geometric (1.7352 vs 1.7248), and
  fixed-share switching is worse still. **Learned weights** are the default
  (`hp_ensemble_learning_rate` 0.01). The mixture's log loss drives an
  exponentiated-gradient step each time a scored byte is read with learning
  on: every held-out byte in the harness, never a generated byte. Generation
  primed prompts without scoring them, so until the prompt-scoring fix it
  served the start weights (for every stage: ensemble, ∞-gram, session,
  neural); it now scores the last 512 prompt bytes (`prompt_score_bytes`)
  and restores the weights after each request. For 95 MB + 8 MiB
  they settle near 0.65 / 0.35, which beats equal weights and every fixed
  weight tried (wiki 1.7210 vs 1.7248 equal and 1.7213 best fixed; Alice
  2.1729 vs 2.1759 and 2.1747). For equal-size shards they stay within 0.001
  of equal.
- Speed: members score on worker threads. On 4 cores, 11 members take 8.2
  instead of 26 ms/byte (3.2×), with identical results
  (`CYPHA_HP_ENSEMBLE_THREADS=0` turns it off).
- Calibration: ECE rises from 1.4% to ~2.5% and the best temperature moves to
  about 0.9 on wiki (−0.004) but stays 1.0 on Alice. Not worth a knob.
- Shards train in parallel on separate cores, so the recipe scales training
  time and quality together.

**Building one.** `cyphalm_shard_train` splits a corpus into N shards, trains
them on worker threads, and writes the checkpoints plus an `ensemble.json`
manifest. `load_cyphalm_model` reads a manifest as one model, so
`cyphalm_generate --load`, the REST server and the harnesses serve it
unchanged:

```bash
cyphalm_shard_train --train enwik8 --bytes 95000000 --shards 11 --tier lean --table-bits 20 --threads 4 --out /tmp/ens
cyphalm_generate --load /tmp/ens/ensemble.json --prompt "..."
```

`CyphaLMModel::add_ensemble_member(model, weight)` attaches pretrained models
for serving. Every serve path fans out (scoring, context advance, learning
switch, stream reset, serve mixer rate), and `hp::StreamRewind` covers every
member, so word lookahead works on ensembles. `cyphalm_ensemble_smoke` checks
the mix against a hand-computed blend. CLI: `cyphalm_generate --ensemble
CKPT[:W]`; harness: `cyphalm_lm_quality --member CKPT`.

**Generation from ensembles.** Mixing flattens the confident, copied runs a
single model produces (a single 8 MiB model with lookahead sometimes quotes
training text verbatim: *"The art of putting together a set is hard to put
into words, but the tunes must flow from one…"*). With word lookahead on top,
ensemble samples lean to safe function words (*"…the information and
commission and the prototype and the original sources and the southern…"*).
Ensembles are the better *distribution*. For free generation a single model,
or an ensemble with lower `word_candidates`, reads better. Proposing candidate
words from the primary model and choosing by the ensemble did not help
(2-shard, wiki: judge 1.272 / d4 0.735 vs 1.158 / 0.789), so it was not kept. Generation
benchmark (8 wiki prompts, `cyphalm_gen_bench`, judge lean 16 MiB):

| generator | judge bits (ref 2.13) | distinct 4-grams (ref 0.80) | ms/byte |
|---|---:|---:|---:|
| lean 8 MiB, byte sampling | 1.745 | 0.859 | 1.9 |
| lean 8 MiB, lookahead K 8 | 1.128 | 0.824 | 7.5–14 |
| 95 MB, byte sampling | 1.925 | 0.902 | 3.2 |
| 95 MB, lookahead K 8 | 1.642 | 0.810 | 15 |
| 2-shard ensemble, K 8 | 1.158 | 0.789 | 37 |
| 5-shard ensemble, K 8 | 1.195 | 0.769 | 92 |
| 11-shard ensemble, byte sampling | 1.764 | 0.898 | 12.8 |
| 11-shard ensemble, K 3 | 1.465 | 0.815 | 22 |
| 11-shard ensemble, K 8 | 1.267 | 0.839 | 66 |

The first row comes from the first benchmark run (`gb_lean_k0_wiki.json`,
reference 2.104 / 0.783), the rest from the second (`gb2_*`, `gb11_*`,
reference 2.134 / 0.803); in the lookahead row, 7.5 ms/byte is from the first
run (`gb_lean_k8`: 1.197 / 0.827) and 14 from the second.
The judge was trained on the first 16 MiB, so it favours models trained there.
Compare decoders on one model, not models against each other (held-out NLL
does that). ms/byte figures come from a loaded 4-core box.

## ∞-gram expert (after infini-gram)

Related work: infini-gram (Liu et al. 2024, arXiv:2401.17377) builds an
n-gram LM with *unbounded* n over trillions of tokens. A suffix array finds,
for the current context, the longest suffix that occurs anywhere in the
corpus, and counts the tokens after every occurrence. Interpolated with neural
LMs it cut perplexity by up to 73%. kNN-LM (Khandelwal et al. 2020) and
StateSMix / Nacrith (neural models plus n-gram context mixing) point the same
way: a big exact memory of the training text complements a parametric model.

hp already has match models, but each follows only the **most recent**
occurrence of a context. The ∞-gram counts **every** occurrence in the whole
corpus.

**Implementation.** `InfiniGram` builds a suffix array (SA-IS, linear time)
over the training bytes. For 95 MB that is 20 s and a 475 MB file (5 bytes per
byte; the packed IGR2 format from the speed round makes it 416 MB), mapped
read-only and shared. A query bisects on the suffix length and
reads next-byte counts from the SA range with one binary search per distinct
next byte: 0.03 ms. `HpSequenceBackend::set_infinigram` serves

p = w0 · p_model + w1 · p_longest + w2 · p_reliable

where p_longest counts bytes after the longest matching suffix, and p_reliable
does the same for the longest suffix seen at least 16 times. Weights are
learned online (exponentiated gradient) per bucket of (match length, count,
model confidence) while learning is on. Context comes from the predictor's
byte history, so exact rewinds and word lookahead cover it.
`cyphalm_infinigram_smoke` checks queries against brute force and the
expert's normalisation, observe and rewind.

Held-out, 16 KiB (index over the same 95 MB the models trained on; eval text
outside it):

| model | wiki | Alice | lcet10 | top-1 wiki |
|---|---:|---:|---:|---:|
| slim 95 MB model | 1.7436 | 2.0828 | 1.5957 | 63.9% |
| **+ ∞-gram** | **1.6925** | **2.0654** | **1.5572** | 65.1% |
| 11 slim shards (ensemble) | 1.6951 | 2.0411 | 1.5356 | 64.8% |
| **11 slim shards + ∞-gram** | **1.6562** | **2.0290** | **1.5141** | 65.8% |

- One slim model plus the index (496 MB + 475 MB shared) now matches the
  11-model ensemble on wiki.
- On the ensemble, the index still takes another 0.039 off wiki.
- The gain is largest in-domain (the index is Wikipedia). On plain English
  (Alice) it is 0.012–0.017.
- Calibration stays within 1–2.3% ECE.

- **Mixing buckets** also split on whether the model's and the longest
  match's top bytes agree: wiki 1.6925 → 1.6890, Alice 2.0654 → 2.0645,
  lcet10 1.5572 → 1.5574 (kept).
- **Tried:** a fourth backoff level (the longest suffix seen at least 256
  times) is ~0.001 worse everywhere, so three parts stay. Starting weights
  learned on 256 KB of held-out wiki (`--ig-weights-out`, loaded from
  `X.igr.weights.json`) barely matter online (−0.0006) and are domain-bound
  when nothing adapts (frozen: wiki −0.003, lcet10 −0.009, Alice +0.010), so
  none ship by default. Those weights were learned before the agreement split
  below doubled the buckets (128 → 256): `igw_trained.json` holds 384 values,
  and `set_infinigram_weights` now needs 768, so a weights file from before
  `2d30e6a` makes `attach_infinigram` throw. Delete or relearn it.
- **Generation** (8 wiki prompts, K 8): distinct 4-grams rise from 0.827 to
  0.879 at the same judge score. The ∞-gram lets the decoder follow long
  verbatim runs of the training text (*"…(Colossians 1:15) to the image of
  Caesar on a Roman coin (Matthew 22:20)…"*). That is more fluent, and it is
  recall.

`cyphalm_infinigram_build --text CORPUS --bytes N --out X.igr`; serve with
`--infinigram X.igr` (`cyphalm_generate`, `cyphalm_lm_quality`,
`cyphalm_gen_bench`) or `"infinigram"` in an ensemble manifest.

## RAM

Everything a served model holds, and what each cut costs. Footprints are
resident memory with tables copied into RAM (`CYPHA_HP_MMAP=0`); held-out NLL
in bits/byte on 8 KiB of wiki / Alice (and *lcet10*, a second book, where
shown).

**Where the memory is** (lean, table bits 22, 8 MiB model, 1071 MB): context
model hash tables ~70% (seven of 50 MB), byte-match position tables 13 × 17
MB, Hebbian word associations 46 MB, byte history 32 MB, discovery pool 25
MB, everything else (mixer, APMs, DMC) ~8 MB. Tables are 96–99% full after
pretraining, so demand-zero pages do not help. The saving has to come from
smaller slots, smaller tables or fewer of them.

### Cuts that cost (almost) nothing

| cut | 8 MiB model | cost wiki / Alice |
|---|---|---|
| **packed context slots**: 16-bit (10-bit state + 6-bit checksum) instead of 24-bit | 1071 → 819 MB | +0.0007 / +0.0048 |
| **Hebbian tables** folded to 16 bits | −44 MB | +0.000 / −0.001 |
| **discovery pool** folded to 16 bits | −24 MB | +0.002 / +0.002 |
| **mapped loading**: serve tables straight from the checkpoint file | private RAM 1066 → 164 MB (frozen serving) | none |

- **Packed slots** changed hp itself. Checkpoint format v3 converts older ones
  on load. Compressing 2 MiB from scratch costs +0.0014 bpc (0.08%).
- **Mapped loading** needs no retraining (`hp::MapScope`, default on): tables
  are `mmap`ed copy-on-write from the `.hpbin`. After 4 KB of text, frozen
  serving holds 164 MB private plus 905 MB of clean file-backed pages. Those
  are shared by every process serving that model and reclaimable by the
  kernel. With learning on while reading a prompt, private memory grows with
  each page written (624 MB after 4 KB). Loading takes 0.1 s instead of 4.2 s,
  and results are identical.

### Cuts that trade quality

- **Serve-time drop of low-value context models.** Each context model was
  dropped from a trained model one at a time (`--drop`, freeing its table).
  Thirteen changed held-out NLL by less than the eval noise (±0.002). Dropping
  all thirteen frees 185 MB: 819 → 634 MB for +0.004 / +0.007 / +0.009
  (wiki / Alice / lcet10). Training without them from the start (`slim` tier)
  is no better than dropping them afterwards (8 MiB, 576 MB: 1.8341 vs
  1.8327). So existing lean models convert after training. The tier still
  halves training memory.
- **Table folding** (`CyphaLMModel::fold_hp_tables`, `--fold`) shrinks trained
  tables as if they had been trained smaller: the dropped index bit moves into
  the slot checksum. Context tables are the expensive ones to fold (22 bits:
  +0.013 / +0.020). Match tables are cheap for a model trained on 8 MiB
  (16 bits: +0.009 / +0.004) but not for one trained on 95 MB (+0.020 /
  +0.013), where they index far more text. `slim` therefore keeps match
  tables at lean's size.
- **Byte-match models** can be dropped too (`Config::match_drop`,
  `--match-drop`). On the 95 MB model, where match tables are 38% of slim, five
  of the thirteen (orders 4, 5, 6 and skip-2 / skip-3) each moved held-out
  NLL by ≤0.001. Dropping all five: 577 → 496 MB for +0.0009 / −0.0003 /
  −0.0001. Three more cost another 48 MB for +0.005 on wiki. `slim` now drops
  the five.
- **Merging shards** into one model (`--merge`, `merge_shard_tables`): two
  8.6 MB shards merged into one table set gain 0.039 over one shard at the
  same RAM, but merging all eleven collapses (2.006 wiki), because one set of
  tables cannot hold them.

- **Freezing the big tables while serving** keeps pages shared, but in-context
  learning lives largely in the context tables. Slim 8 MiB model, 16 KB of
  text: all tables learning, 1.8049 / 2.1050 and 493 MB private. Tables of
  23+ bits frozen, 1.8412 / 2.1507 and 239 MB private. All frozen, 1.8839 /
  2.2745 and 194 MB private (match, history and pool writes remain). Not kept.

### RAM / quality frontier

Held-out wiki NLL (lower is better) against resident footprint:

| config | footprint | wiki | Alice |
|---|---:|---:|---:|
| 95 MB model, packed + context 22 + match/pool 16 | 343 MB | 1.8577 | 2.2961 |
| 8 MiB model, slim | 576 MB | 1.8341 | 2.2685 |
| **95 MB model, slim (incl. 5 match models dropped)** | **496 MB** | **1.7738** | **2.2442** |
| 95 MB model, slim drop + pool/Hebbian 16 | 577 MB | 1.7729 | 2.2445 |
| 95 MB model, packed | 819 MB | 1.7660 | 2.2263 |
| 95 MB model, unpacked (before) | 1071 MB | 1.7608 | 2.2158 |
| 5 shards, context 22 + match/pool 16 | 810 MB | 1.7717 | 2.2301 |
| 3 shards, match/pool 16 | 1.03 GB | 1.7629 | 2.2178 |
| 5 shards, match/pool 16 | 1.72 GB | 1.7399 | 2.2039 |
| **11 shards, slim drop + pool/Hebbian 16** | **4.0 GB** | **1.7069** | **2.1936** |
| 11 shards trained slim (`cyphalm_shard_train --tier slim`) | 3.9 GB | 1.7058 | 2.1949 |
| **11 slim shards, 5 match models dropped** | **3.55 GB** | **1.7085** | **2.1957** |
| **11 slim shards at table bits 18, 5 match models dropped** | **2.29 GB** | **1.7268** | **2.2004** |
| **95 MB slim + 11 slim shards** | **4.5 GB** | **1.6916** | **2.1812** |
| 11 shards, packed | 5.04 GB | 1.7009 | 2.1834 |
| 11 shards, unpacked (before) | 7.2 GB | 1.6994 | 2.1797 |

- **Up to ~1 GB,** one model trained on all the data is the best use of
  memory. Slim plus packing takes the 95 MB model from 1071 to 496 MB (−54%)
  for +0.013 wiki / +0.028 Alice.
- **Above ~1 GB,** ensembles of shard models win. The 11-shard ensemble drops
  from 7.2 GB to 3.55 GB (−51%) for +0.009 / +0.016. Trained at table bits
  18, it drops to 2.29 GB (−68%) for +0.027 / +0.021, still well ahead of the
  single 95 MB model (1.7608 / 2.2158 at 1.07 GB).
- **With mapped loading,** a single model's tables stay shareable file-backed
  pages when served frozen. Learning from the prompt writes pages, though, and
  an ensemble's members all learn. After 8 KB of prompt, the 11 slim shards
  hold 3.6 GB private of 3.9 GB (the slim 95 MB model alone: 470 MB of 577).
  Keeping the members frozen while only the primary learns cuts that to 1.3 GB
  but costs +0.06 wiki and +0.11 Alice (members still write match positions).
  Not kept: in-context learning is worth the memory.
- **The best ensemble before this work** (95 MB + 11 shards, unpacked, 8.3 GB:
  1.6860 / 2.1636) is now 1.6916 / 2.1812 at 4.5 GB.

## Other results this round

- **Two training epochs hurt.** Epoch 2 trains at 0.004 bpc because the match
  models replay the memorised text, and held-out NLL gets worse (wiki
  1.8259 → 1.8545, Alice 2.2532 → 2.2821). One pass is right for hp.
- **Serve-time StateMap limits and APM rates** (slower or faster adaptation
  of the context models and APMs) moved wiki and Alice in opposite directions
  by ≤0.004. Not kept.
- **Two checkpoint bugs fixed.** A loaded or copied predictor lost its
  word-match window (the rebind replaced it with the full ring mask), and
  checkpoints left out the sentence memory. Both made a reloaded model predict
  differently from the saved one; format v2 now round-trips byte-identically.
  Re-measured, the held-out numbers above are unchanged.
- **Vocabulary.** hp checkpoints made with the default config had
  `vocab_size` 128, which truncated serve distributions to ASCII (no UTF-8 in
  generation). `apply_hp_production_recipe` now always uses 256.

## Speed round and the winner

Ideas adapted from related work:
- **min-p sampling and lpaq / zpaq** spend effort only where probability is.
  Here that becomes pruning of hp's next-byte bit tree.
- **infini-gram mini** (FM-index, arXiv:2506.12229) shrinks suffix-array
  storage. Here that becomes bit-packing the SA entries.
- **paq8px** multithreads models ahead of a serial mixer. Here the ensemble
  members already score on threads.

**Bit-tree pruning** (`hp_tree_prune`, default 1e-4). A next-byte
distribution walks the 255-node bit tree through the whole mixer. Subtrees
under 1e-4 probability are not expanded, and their mass is spread evenly over
their bytes, so the distribution stays normalised. Slim 95 MB + ∞-gram, 8 KiB:

| prune | ms per distribution | wiki | Alice | lcet10 |
|---|---:|---:|---:|---:|
| exact | 12.8 | 1.7068 | 2.2128 | 1.5427 |
| 1e-5 | 3.7–4.2 | 1.7068 | 2.2124 | 1.5427 |
| **1e-4** | **2.3–2.8** | **1.7067** | **2.2125** | **1.5432** |
| 1e-3 | 1.3–1.8 | 1.7089 | 2.2163 | 1.5447 |

**Packed ∞-gram index** (IGR2): suffix-array entries at ceil(log2 n) bits,
27 for 95 MB. 475 → 416 MB, identical results. A smaller corpus is not a good
trade: a 48 MB index costs +0.011 on wiki, 24 MB +0.027.

**The winner** (`models/cyphalm_winner/`, rebuilt by
`scripts/build_cyphalm_winner.sh`), 16 KiB held-out:

| manifest | wiki | Alice | lcet10 | top-1 / top-5 wiki | ECE | ms/dist | RAM private + shared |
|---|---:|---:|---:|---|---:|---:|---|
| `winner.json`: 11 slim shards + ∞-gram | **1.6516** | **2.0258** | **1.5124** | 65.9% / 89.4% | 1.2% | 2.3 | 3.3 + 0.6 GB |
| `winner_light.json`: slim 95 MB + ∞-gram | 1.6909 | 2.0639 | 1.5566 | 65.1% / 89.2% | 1.0% | 1.8 | 0.44 + 0.45 GB |

Against the lean 95 MB model at the start of this work (1.7608 / 2.2158 on 8
KiB, 1.1 GB, ~8 ms per distribution), the light winner is better on every
text at under 1 GB and ~4× faster. The full winner cuts another 0.04 on wiki.

The two manifests (paths relative to the manifest; `winner.json` lists
`shard_0.json` … `shard_10.json` in order):

```json
{"cyphalm_ensemble": 1, "members": [{"checkpoint": "shard_0.json"}, …, {"checkpoint": "shard_10.json"}],
 "learning_rate": 0.01, "infinigram": "enwik8_95m.igr"}
{"cyphalm_ensemble": 1, "members": [{"checkpoint": "slim95.json"}], "infinigram": "enwik8_95m.igr"}
```

Provenance: the measured `slim95` checkpoint records tier `lean` in its config
(`win_winner_light_*.json`, `sl_s95_*.json`). It is the lean 95 MB model cut
to slim after training (`--drop`, `--fold`). `build_cyphalm_winner.sh`
instead trains `--tier slim` from scratch. The two routes measured equal at
8 MiB (1.8341 trained slim vs 1.8327 converted) but were not compared at
95 MB. Both winner rows were scored at the harness defaults (trained mixer
rate, see [Measurement conventions](#measurement-conventions)).

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
| `word_candidates` | **8** | word lookahead: best of K sampled words (0 = byte sampling only) |
| `word_no_repeat` | 12 | word lookahead rejects words that repeat a sequence this long |
| `prompt_score_bytes` | **512** | composite models: score the last N context bytes before generating, so the mixing weights adapt to the prompt (0 = start weights, the behaviour before this fix) |
| `restore_mixing` | on | put every mixing weight back when the request ends |

Serve-time model settings: `hp_frozen_scoring` (on), `hp_serve_mixer_lr_scale`
(0.5), `hp_tree_prune` (1e-4), learned ensemble and ∞-gram weights (adapted
over the scored prompt, restored per request), and priming keeps the byte
history. Full list: [Reference](#reference).

## Reproduce

```bash
Q=native/build/cyphalm_lm_quality
$Q --tier lean --train enwik8 --train-bytes 8388608 --save /tmp/pre_lean
$Q --load /tmp/pre_lean.json --eval enwik8 --eval-offset 96000000 --eval-bytes 16384 --compare-scoring
$Q --load /tmp/pre_lean.json --eval alice29.txt --eval-offset 20000 --eval-bytes 16384 --frozen-eval --gen-bytes 0
# ensembles: shards trained on disjoint slices, then mixed
$Q --tier lean --train enwik8 --train-offset 48000000 --train-bytes 8388608 --save /tmp/pre_48m
$Q --load /tmp/pre_lean.json --member /tmp/pre_48m.json --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
$Q --tier lean --train enwik8 --train-bytes 16777216 --save /tmp/pre_16 --gen-bytes 0   # the judge
native/build/cyphalm_gen_bench --load /tmp/pre_lean.json --judge /tmp/pre_16.json --text enwik8 --offset 96000000 --prompts 8
# serving: history on reset, mixer rate (quarters of trained), word lookahead
$Q --load /tmp/pre_lean.json --reset-stream full --frozen-eval --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
$Q --load /tmp/pre_lean.json --serve-lr 2 --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
$Q --load /tmp/pre_lean.json --eval enwik8 --eval-offset 96000000 --eval-bytes 16384 --gen-bytes 400 --only-default --word-k 8
# ensembles through the library: shard trainer, manifest, learned weights
native/build/cyphalm_shard_train --train enwik8 --bytes 95000000 --shards 11 --tier slim --table-bits 20 --threads 4 --out /tmp/s11
$Q --load /tmp/s11/ensemble.json --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
$Q --load /tmp/pre_lean.json --member /tmp/pre_48m.json --ensemble-lr 0 --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
native/build/cyphalm_generate --load /tmp/pre_lean.json --ensemble /tmp/pre_48m.json:0.5 --prompt "..." --max-bytes 300
# RAM: serve-time conversion of a lean model to slim (13 context models, pool/Hebbian 16 bits, 5 match models)
$Q --load /tmp/pre_lean.json --drop 0x628426bc0 --fold 0,0,16,16 --match-drop 0xd06 --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
$Q --load /tmp/pre_lean.json --fold 22,16,16 --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0   # context 22, match/pool 16
$Q --load /tmp/s11/shard_0.json --merge /tmp/s11/shard_1.json --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
CYPHA_HP_MMAP=0 $Q --load /tmp/pre_lean.json --frozen-eval --eval enwik8 --eval-offset 96000000 --eval-bytes 4096 --gen-bytes 0  # copied tables
$Q --tier lean --train enwik8 --train-bytes 8388608 --epochs 2 --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
# ∞-gram index, starting weights, bit-tree pruning
$Q --tier lean --train enwik8 --train-bytes 95000000 --save /tmp/lean95 --gen-bytes 0   # ~2.3 h
$Q --load /tmp/lean95.json --drop 0x628426bc0 --fold 0,0,16,16 --save /tmp/slim95 --gen-bytes 0   # cut to slim
native/build/cyphalm_infinigram_build --text enwik8 --bytes 95000000 --out /tmp/enwik8_95m.igr
$Q --load /tmp/slim95.json --infinigram /tmp/enwik8_95m.igr --eval enwik8 --eval-offset 96000000 --eval-bytes 16384 --gen-bytes 0
$Q --load /tmp/slim95.json --infinigram /tmp/enwik8_95m.igr --eval enwik8 --eval-offset 95000000 --eval-bytes 262144 --gen-bytes 0 --ig-weights-out /tmp/w.json
$Q --load /tmp/slim95.json --infinigram /tmp/enwik8_95m.igr --tree-prune 1e-3 --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
CYPHA_HP_TREE_PRUNE=0 $Q --load /tmp/slim95.json --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0   # exact tree
native/build/cyphalm_gen_bench --load /tmp/slim95.json --infinigram /tmp/enwik8_95m.igr --judge /tmp/pre_16.json --text enwik8 --prompts 8
# the winner
scripts/build_cyphalm_winner.sh enwik8 /tmp/winner native/build 4
$Q --load /tmp/winner/winner.json --eval enwik8 --eval-offset 96000000 --eval-bytes 16384 --gen-bytes 0
```

The ∞-gram, pruning and winner lines match the configurations behind the raw
JSON; the paths are placeholders. `slim95.json` in the report is the
converted lean 95 MB model (see *Speed round and the winner*).

## Winner v2: smaller weights, upstream gains, neural expert

**Where the 4 GB went.** `HP_CKPT_SIZES=1` prints each component's checkpoint size. A slim shard is 338 MB:
- six context-model tables of 33.5 MB each;
- the order-0 match table (48 MB);
- the byte ring (8 MB);
- smaller tables.

How full the tables are varies widely. `skip4_` is 0.8% used, `o3_` 7.7%, `word_` 13%, `o4_` 23%, while `wbi_` is 94% and `sen_` 100%.

**Occupancy fold** (`Predictor::fold_auto`, `--fold-auto OCC`). Each context, match, pool and Hebbian table is halved while its projected occupancy stays at or under OCC. Old winner, 16 KiB:

| OCC | freed | wiki | Alice |
|---|---:|---:|---:|
| — | — | 1.6510 | 2.0257 |
| 0.4 | 0.80 GB | 1.6520 | 2.0269 |
| 0.6 | 1.17 GB | 1.6530 | 2.0298 |
| 0.8 | 1.60 GB | 1.6597 | 2.0323 |

Dropping whole tables costs more. `o3_` +0.0125 and `o4_` +0.0149 on wiki; `sentst_` is the exception (+0.002 wiki, +0.0001 Alice).

**Upstream compression gains** (CompressionAlgorithm; ported as config fields, checkpoint v4):
- layer-1 rate scale 40;
- layer-1 dot scale 0.75;
- mixer skip 56;
- layer-1 skip 80.

On gate24 8 MiB, the rate scale alone saves 6,741 B (upstream 7,038). These settings do **not** help the slim tier (one 8.6 MB shard: wiki 1.8178 → 1.8177, frozen worse). On lean they do, and lean beats gate24 at every size after folding:

| one 8.6 MB shard | size after fold | wiki | Alice |
|---|---:|---:|---:|
| slim (old winner shard), unfolded | 322 MB | 1.8178 | 2.1395 |
| lean + upstream, OCC 0.6 | 326 MB | **1.8106** | **2.1169** |
| lean + upstream, OCC 0.8 | 268 MB | 1.8193 | 2.1235 |
| gate24 + upstream, OCC 0.6 | 380 MB | 1.8101 | 2.1179 |

The upstream wiki context models (bold/italic, sentence position, capitals × paragraph, `<ref>` group, state transitions, two cross contexts) are ported behind `hp_extra_cms` / `CYPHA_HP_EXTRA_CMS` (default 0, bit-identical when off; checkpoint v5 when on). They save 5,252 B on gate24 8 MiB, but on a lean shard only −0.0008 wiki / −0.0027 Alice for +4% size, so they are off in the winner.

**Neural expert** (`ByteLstmExpert`, manifest key `neural`, `--neural`). The 1-hour byte LSTM from the comparison (3.4M params, 13 MB) runs in C++ and matches PyTorch to 1e-6 bits/byte. It is mixed after the ∞-gram step as w·p + (1−w)·p_nn, with w learned per (confidence, agreement) bucket. Its matrices are held as bf16: 0.66 ms per byte on one idle core (2.28 ms as float32; the step is bound by reading the weights), same held-out NLL.

**Winner v2** (`models/cyphalm_winner/`, 16 KiB held-out, private + mapped RAM):

| manifest | wiki | Alice | lcet10 | RAM |
|---|---:|---:|---:|---|
| old winner (11 slim shards + ∞-gram) | 1.6510 | 2.0257 | 1.5117 | 3.3 + 0.6 GB |
| `winner_full.json`: 11 lean+upstream shards, OCC 0.8, no `sentst_`, ∞-gram, LSTM | **1.6158** | **2.0113** | **1.5038** | 2.7 + 0.4 GB |
| **`winner.json` (default)**: first 4 of those shards | 1.6192 | 2.0191 | 1.5160 | **1.0 + 0.4 GB** |
| `winner_light.json`: slim 95 MB + ∞-gram + LSTM | 1.6393 | 2.0466 | 1.5388 | 0.46 + 0.44 GB |
| old light (no LSTM) | 1.6909 | 2.0639 | 1.5566 | 0.44 + 0.45 GB |

The 4-shard default is within 0.003 (wiki) to 0.012 (lcet10) of the full ensemble at 37% of its private RAM, and is better than the old 11-shard winner on every text. The 11 shards take 2,737 MB on disk (old 3,544 MB; the default's 4 take about 1 GB) and train in 24.0 min on 4 cores; the LSTM takes 60 min. Before folding, the 11 lean shards score 1.6465 / 2.0102 / 1.5064 with ∞-gram and no LSTM, at about 5 GB.

**Just-in-time ∞-gram index.** The suffix array no longer needs to be stored. libsais (induced sorting, Apache-2.0, vendored in `native/third_party/libsais`) sorts the 95 MB corpus in 5.6 s on one core of this machine, even while it is busy (the old SA-IS code took 20–29 s). The published libsais benchmark gives 1.2 s for enwik8 on one desktop core. `InfiniGram::open` takes either a stored IGR file or the plain corpus plus `infinigram_bytes`. Given the corpus, it builds the same bit-packed index in memory at load time. The winner manifests now point at `enwik8` with 95,000,000 bytes:
- held-out numbers are identical (4-shard wiki 1.6191);
- load time is 7.4 s in total, index included;
- the 416 MB index file is gone;
- the same ~0.4 GB of RAM moves from mapped to private.

**The index as a cache.** An in-memory index costs seconds, so it can be built for any text, not only the corpus.

- **Tracing** (`cyphalm_trace`, after OLMoTrace). For every position of a text it finds the longest corpus match and its location, via `InfiniGram::match_prefix`. It reports:
  - the spans of at least `--min-len` bytes;
  - the share of the text they cover.

  A 16 KiB text traces in 0.2–0.7 s. Raw data: [`lm_quality/trace/`](lm_quality/trace/).

  | text | share in ≥32-byte corpus matches | longest match | what matches |
  |---|---:|---:|---|
  | wiki held-out (16 KiB) | 4.2% | 133 | XML page headers, taxobox templates |
  | Alice (16 KiB) | 0.4% | 61 | one sentence quoted in Wikipedia |
  | lcet10 (16 KiB) | 1.5% | 35 | whitespace runs |
  | true continuations (12 × 400 B) | 16% | 202 | |
  | CyphaLM, word lookahead | **28%** | 67 | |
  | CyphaLM, byte sampling | 3.7% | 55 | |
  | LSTM | 8.9% | 67 | |
  | Transformer | 1.3% | 58 | |

  The held-out texts are not contaminated. Word lookahead copies whole corpus phrases more than any other decoder, which is part of why every judge rated its output so probable.
- **Session cache** (`set_session_cache`, `--session-cache`, manifest `"session_cache": true`). An index over the text the stream has read is rebuilt at 1 KiB, on each doubling, then every 64 KiB. Since the serve-state fix it covers the last 1 MiB only (`window`, manifest `"session_window"`, 0 = all; rebuilds at most window / 16 apart, so each costs at most 1 MiB), bytes read under `hp::StreamRewind` lookahead never trigger a rebuild (before, each word candidate past a threshold rebuilt the whole history), and `reset_stream(false)` / `reset()` start an empty session. Its longest-match counts are mixed in with a weight learned per (length, count) bucket. Mixing every match hurts (4-shard winner: wiki +0.012, lcet10 +0.008), because hp's match models and online learning already cover short repeats. Using only matches of 16 bytes or more is neutral to slightly positive (wiki ±0, Alice −0.0004, lcet10 −0.0010). It is off by default; it is for long sessions that paste or repeat long text.
- **Draft bytes from the index** (speculative decoding) does not pay here:
  - a Transformer verifies k drafted tokens in one parallel pass, but hp reads bytes serially, so checking a draft costs as much as generating it;
  - the geometric ensemble mix needs every member's full distribution to normalise, so even a single-byte check needs the whole distribution.

  Not implemented.
- **Replacing hp's match models with the index** does not work either: with the ∞-gram attached, dropping the order-0 match model costs +0.0017 wiki / +0.0015 Alice, and dropping `smatch` costs +0.0013 / +0.0005 (drop sweep above). The mixer uses the match models' recency signal, which the index lacks.

## Winner v3: two neural experts that adapt while reading

**Reproducible rebuild.** `scripts/build_cyphalm_winner.sh` was run end to end on this machine (148 min wall time).
- The rebuilt shards reproduce the published scores exactly when served with the published LSTM: 4-shard wiki 1.6191, 11-shard 1.6157, against 1.6192 / 1.6158.
- The rebuilt LSTM was weaker (4-shard 1.6220 / 2.0222 / 1.5181). Its training stopped on a 60-minute wall-clock budget, and a concurrent compile left it with 65.9 MB read instead of 81.1 MB.
- `byte_lm.py train --budget-bytes` now stops on bytes read instead, and the script uses the published runs' byte counts.

Raw data: [`lm_quality/v3/`](lm_quality/v3/).

**Transformer expert** (`ByteGptExpert`, "BGT1", `byte_lm.py export`). The 1-hour 3.35M-parameter Transformer from the comparison runs in C++:
- pre-LN, exact GELU, a KV cache and bf16 matrices;
- a full 512-byte window re-primes on its last 256 bytes, as the PyTorch stepper does;
- it matches PyTorch fp32 to 1e-4 bits/byte (1.85854 vs 1.85866 on 1,500 wiki bytes);
- 1.46 ms/byte on one core.

An offline estimate from the saved distributions showed the LSTM and the Transformer complement each other: old winner + both, wiki 1.5941 against 1.6115 with the LSTM alone.

**Several experts, one mix** (`add_neural`, manifest `"neural": [...]`, repeatable `--neural`). After the ∞-gram step the served distribution is the linear mix w₀·p + Σ wᵢ·p_nn,i. The weights are learned by exponentiated gradient per (model confidence × agreement with expert 0) bucket.

**Output-layer adaptation** (`set_neural_adaptation`, `--neural-adapt`, manifest `neural_adapt`). This is dynamic evaluation of the last layer only. Each stream keeps its own float copy of the experts' output layer (LSTM output weights and bias; the Transformer's tied embedding) and takes one SGD step on each read byte's log loss, only while learning is on.

In PyTorch on an 8 KiB tuning slice, the LSTM alone scores 1.770 static, 1.638 with full dynamic evaluation (SGD 0.3 per 128 bytes) and 1.684 with the output layer only. So the output layer carries two-thirds of the gain at a fraction of the cost: about 131K multiply-adds per byte, and 0.5 MB of state per stream for the LSTM.

**Tuning on separate slices.** Rates were tuned on enwik8 @97,000,000, Alice @100,000 and lcet10 @200,000, 16 KiB each. The test texts were not used.

| 4-shard winner + … (tuning slices) | wiki | Alice | lcet10 |
|---|---:|---:|---:|
| no neural expert | 1.3083 | 1.9472 | 1.6578 |
| LSTM, mixing rate 0.05 | 1.2449 | 1.9361 | 1.6404 |
| LSTM + Transformer, 0.05 | 1.2216 | 1.9371 | 1.6354 |
| … + output adaptation 0.002 | **1.2132** | **1.9242** | **1.6232** |
| … + output adaptation 0.005 | 1.2197 | 1.9219 | 1.6222 |

Mixing rates 0.02 / 0.1 / 0.3 were within ±0.001 of 0.05, or worse at 0.3.

**Winner v3** (held-out test texts, 16 KiB; neural rate 0.05, adaptation 0.002):

| manifest | wiki | Alice | lcet10 | top-1 wiki | private RAM |
|---|---:|---:|---:|---:|---:|
| v2 `winner.json` (4 shards + LSTM) | 1.6192 | 2.0191 | 1.5160 | | 1.0 GB |
| **v3 `winner.json`** (4 shards + LSTM + Transformer, adapting) | **1.5927** | **1.9992** | **1.4984** | 66.7% | 1.0 GB |
| v3 `winner_full.json` (11 shards) | 1.5920 | 1.9929 | 1.4889 | 66.7% | 2.7 GB |
| v3 `winner_light.json` (one slim model) | 1.5997 | 2.0192 | 1.5186 | 66.5% | 0.46 GB |
| v2 `winner_light.json` | 1.6393 | 2.0466 | 1.5388 | | 0.46 GB |

The light model (0.46 GB) now beats the v2 11-shard winner on wiki. The 13.4 MB Transformer adds about 1.5 ms per byte on one core.

## Against neural baselines

[`CYPHALM_VS_NEURAL_LM.md`](CYPHALM_VS_NEURAL_LM.md) sets the winner against a byte-level Transformer (3.35M params) and an LSTM (3.43M params). Each trains for at most 1 hour on the same 4 cores and the same 95 MB, and all are scored by one code path.

| wiki / Alice / lcet10, bits/byte | CyphaLM | Transformer | LSTM |
|---|---|---|---|
| adapting to the text (reading / dynamic eval) | **1.651 / 2.026 / 1.512** | 1.776 / 2.685 / 2.028 | 1.739 / 2.427 / 2.015 |
| no adaptation (frozen / static) | 1.819 / **2.695 / 1.970** | **1.805** / 3.500 / 2.466 | 1.901 / 3.257 / 2.438 |
| training time | 17 min | 60 min | 60 min |

The Transformer is ahead only with no adaptation on in-domain text. CyphaLM copies from context (0.05 bits/byte on a repeated passage, against 1.2–1.8 for the neural models) and generates real words (99.4%). The neural models are about 300× smaller to serve and better at predicting markup and punctuation.

## Reference

State at commit `2a003bf` (2026-09-24). Everything below is read from the code
and the commit messages; numbers are the ones measured above.

### Measurement conventions

- Held-out slices: wiki = enwik8 @ 96,000,000; Alice = `alice29.txt` @ 20,000;
  lcet10 = `lcet10.txt` @ 50,000; checks: enwik8 @ 97,500,000 and
  `plrabn12.txt` @ 50,000. The harness scores `--eval-bytes` from the offset;
  the next `--prompt-bytes` (256) are the generation prompt.
- **Mixer rate.** `cyphalm_lm_quality` scores at the *trained* mixer rate
  (`--serve-lr 4`, quarters) and does not call `set_serve_mode`. Serving
  (`generate_decode`, REST) runs at 0.5× (`hp_serve_mixer_lr_scale`). Every
  held-out NLL in this report is at 1× unless it says ×0.5 (the serve-rate
  table, `aq_2_*`, `val_2_*`). The 0.5× rate was 0.0015–0.008 better there.
- **Stream.** `--reset-stream none` (default): evaluation continues the
  pretraining stream. Serving primes with `reset_stream(keep_history=true)`,
  measured identical for frozen NLL (2.0285 both, `rs_*`).
- **Scoring.** Frozen scoring on (default since `f9c24d0`) except the
  exact-vs-frozen table. Bit-tree pruning follows the checkpoint config
  (`hp_tree_prune`, default 1e-4 since `d7694b5`; checkpoints saved before it
  have no key and load with 1e-4). Earlier sections were measured before
  pruning existed, i.e. exact. `--tree-prune P` only overrides for P > 0; for
  exact scoring set `CYPHA_HP_TREE_PRUNE=0`.
- Timings (ms/byte, ms/distribution) come from a shared 4-core box under load
  unless marked solo. Compare within a table only.
- RSS in the raw JSON: `rss_mb_*` (total), `rss_anon_mb_*` (private),
  `rss_file_mb_*` (file-backed, shareable), `rss_hwm_mb` (peak).
- **Composite models** (members, ∞-gram, neural experts or session cache).
  Before the harness-fix commit, `ms_per_byte` in their JSON counted a second
  full scoring per byte (the bit-greedy check re-scored the mixture); tables
  here use `distribution_ms`, which was unaffected. `bit_greedy_equals_argmax`
  is now `null` for them, since serve greedy is the mix's argmax by
  construction, and the generation grid is skipped because it reloads the
  saved primary alone. Earlier manifest runs with `--serve-lr`/`--serve-skip`
  set them on the primary only, and `--neural-lr` did not reach manifest
  experts.
- **Served mixture.** Before the prompt-scoring fix, generation on composite
  models used the start mixing weights (priming scored nothing), while the
  harness NLLs above use weights learned over the held-out text;
  `--freeze-mixing` measures the start-weight mixture: v3 winner, wiki
  16 KiB, 1.6175 against 1.5927 learned (+0.025; top-1 66.6%). Scoring
  512 prompt bytes adds ~1.9 s per request on the v3 winner (3.7 ms per
  distribution, shared box). Byte beam search ranked with the primary
  predictor alone until the same fix.
- **Fractional serve rates.** `--serve-lr Q` below 4 scales in 1/16ths of a
  rate since the serve-state fix; before, each rate floored at 1, so ×0.5 was
  a no-op on the layer-1 sets of lr1_scale 40 shards. ×1 (the default) is
  bit-identical either way.

### Tools

| tool | purpose | flags (default) |
|---|---|---|
| `cyphalm_lm_quality` | pretrain or load, score held-out next-byte distributions, generate | see below |
| `cyphalm_generate` | generate from a cold model, checkpoint or manifest | see below |
| `cyphalm_gen_bench` | continue N held-out prompts, score with a fixed judge (learning off, exact rewind of predictors, neural experts and session cache). A plain-text `--infinigram` corpus indexed whole is warned about on stderr, more loudly when it is the `--text` file and reaches `--offset` (the model could quote the references) | `--load CKPT` · `--member CKPT` (repeatable) · `--infinigram IDX\|CORPUS` · `--infinigram-bytes N` (0 = all; plain corpus only) · `--judge CKPT` · `--text FILE` · `--offset` (96000000) · `--prompts` (8) · `--prompt-bytes` (256) · `--gen-bytes` (200) · `--stride` (8192) · `--word-candidates` (8) · `--temperature` (0.8) · `--min-p` (0.1) · `--seed` (1234) · `--prompt-score-bytes` (512; `DecodeParams::prompt_score_bytes`) |
| `cyphalm_shard_train` | split `--bytes` from `--offset` into N equal shards (remainder dropped), train one model per shard on threads, write `shard_<i>.json/.hpbin` + `ensemble.json`. `--threads` below 1 exits 2; a shard that fails (unreadable or empty slice, save error) is reported and the run exits 1 without a manifest | `--train FILE` · `--bytes N` · `--offset` (0) · `--shards` (4) · `--threads` (all cores, ≥ 1) · `--tier` (lean) · `--table-bits` (20) · `--out DIR` |
| `cyphalm_trace` | spans of a text found verbatim in the corpus (∞-gram index), JSON on stdout. An unreadable `--text` exits 1 before the index is built | `--corpus IGR\|TEXT` · `--corpus-bytes` (0 = all) · `--text FILE` · `--offset` (0) · `--bytes` (0 = rest) · `--min-len` (32) · `--top` (20) |
| `cyphalm_infinigram_build` | SA-IS suffix array over a corpus slice, IGR2 file | `--text FILE` · `--bytes N` · `--offset` (0) · `--out X.igr` |
| `scripts/build_cyphalm_winner.sh` | rebuild the winner checkpoints + index | `ENWIK8 OUT_DIR [BUILD_DIR] [THREADS]` |

`cyphalm_lm_quality`:

| flag | default | effect |
|---|---|---|
| `--train FILE` `--train-bytes N` `--train-offset N` | —, 0, 0 | pretrain (one online pass per epoch) |
| `--epochs N` | 1 | passes over the training slice (2 hurt, see below) |
| `--tier NAME` `--table-bits N` | gate24, 22 | new model's tier / table bits |
| `--save BASE` / `--load CKPT.json` | | write `BASE.json` + `BASE.hpbin` / load a checkpoint or ensemble manifest |
| `--member CKPT` | | attach an ensemble member (repeatable, equal weights 1/(n+1)) |
| `--neural FILE` | | attach a byte LSTM (BLM1) / Transformer (BGT1) expert (repeatable) |
| `--neural-lr R` | manifest `neural_learning_rate`, else 0.1 | mixing-weight rate of every expert, manifest ones included; the effective rate is written as `neural_learning_rate` |
| `--neural-adapt LR` | manifest | output-layer SGD rate of the experts (0 = frozen) |
| `--ensemble-lr R` | config (0.01) | ensemble weight learning rate; 0 = fixed |
| `--freeze-mixing` | off | no mixing weight learns (ensemble, ∞-gram, session, neural; `set_mixing_learning(false)`) while the models still learn: scores the start-weight mixture that generation served before prompt scoring |
| `--merge CKPT` | | merge an equal-data shard's tables into `--load` (`merge_shard_tables`, repeatable). Exits 1 when table sizes differ (folded or dropped models): merge before folding |
| `--fold CM,MATCH,POOL[,HEBB]` | 0 = keep | fold trained tables to these bits (load path; also every `--member`) |
| `--fold-auto OCC` | 0 = off | fold each table while its projected occupancy stays ≤ OCC (`fold_auto`, members too); OCC outside (0, 1) exits 2 |
| `--drop MASK` | 0 | drop context models, bit i = `hp::Predictor::CmId` i (slim's extra 13: `0x628426bc0`) |
| `--match-drop MASK` | 0 | drop byte-match models, bit k over `match_[0..8], smatch, skipk, skip3, skip4` (slim: `0xd06`) |
| `--infinigram X.igr` | | attach the ∞-gram expert (loads `X.igr.weights.json` if present) |
| `--ig-weights-out FILE` | | after eval, write the learned ∞-gram weights (name it `X.igr.weights.json` to auto-load) |
| `--tree-prune P` | checkpoint config | override bit-tree pruning (only P > 0) |
| `--eval FILE` `--eval-offset N` `--eval-bytes N` | —, 0, 16384 | held-out slice |
| `--prompt-bytes N` | 256 | generation prompt: the bytes right after the scored slice |
| `--frozen-eval` | off | learning off while scoring (pretrained knowledge only) |
| `--compare-scoring` | off | exact and frozen distributions per byte |
| `--reset-stream none\|full\|keep` | none | new stream before eval: none, wipe history, keep history |
| `--serve-lr Q` `--serve-skip S` | 4, −1 | mixer rate × Q/4, small-error skip threshold (−1 = trained), on the primary and every member |
| `--gen-bytes N` | 200 | generation length (0 = none; skipped for composite models: members, ∞-gram, experts, session cache; use `cyphalm_generate --load` or `cyphalm_gen_bench`) |
| `--temperature T` `--top-p P` | 0.8, 0.9 | byte-level generation grid |
| `--only-default` | off | only the default byte-level decoder in the grid |
| `--word-k K` | 0 | also generate with word lookahead at K/2 and K (library decoder) |
| `--dump-dist FILE` | | float32 natural-log P, 256 per held-out byte (working tree, not in `2a003bf`) |

Flags that act on a loaded model (`--member`, `--ensemble-lr`, `--merge`,
`--fold`, `--drop`, `--match-drop`, `--fold-auto`, `--session-cache`,
`--neural`, `--neural-lr`, `--neural-adapt`, `--infinigram`,
`--infinigram-bytes`, `--ig-weights-out`, `--freeze-mixing`) exit with
status 2 without `--load`.

`cyphalm_generate` flags added in this work (older ones: `docs/native/CYPHALM_SERVE.md`):
`--load CKPT.json` (checkpoint or manifest; cold model otherwise, table bits
16), `--tier NAME`, `--min-p P` (0.1), `--no-repeat N` (0), `--learn-from-output`
(off), `--word-candidates K` (8; 0 or 1 = byte sampling), `--ensemble
CKPT.json[:W]` (repeatable; no `:W` = equal shares 1/(n+1)), `--infinigram
X.igr|CORPUS`, `--infinigram-bytes N` (plain corpus: index only its first N
bytes; 0 = all, with a stderr warning, since a whole corpus may hold the
held-out text), `--neural FILE` (repeatable), `--neural-lr R` (mixing rate of
every expert; default: the manifest's `neural_learning_rate`, else 0.1 —
`--neural` used to reset a manifest's rate to 0.1), `--session-cache`,
`--prompt-score-bytes N` (512; 0 = serve the start mixing weights),
`--no-restore-mixing` (keep the weights the prompt taught).
Temperature defaults to 0.8. Word lookahead runs for every non-beam
strategy while `learn_from_output` is off, so plain greedy needs
`--word-candidates 0`.

### Config fields (`CyphaLMConfig`, saved in the checkpoint JSON `config`)

| field | default | env | effect |
|---|---|---|---|
| `hp_table_bits` | 22 | | table size (mem) |
| `hp_lossy_tier` | "" (gate24) | `CYPHA_HP_LOSSY_TIER` | tier name: `lean`, `balanced`, `compact`, `small`, `tiny`, `slim` |
| `hp_cm_drop` | 0 | | bit i drops context model i |
| `hp_cm_bits_cap` / `hp_match_bits_cap` / `hp_pool_bits_cap` / `hp_hebb_bits_cap` | 0 = none | | cap context / match / pool / Hebbian tables |
| `hp_pool_slots` | 0 = 12 | | discovery-pool slots kept |
| `hp_gate_drop`, `hp_mixer_skip` | 0 | | mixer weight-set drop, update skip (no tier uses them) |
| `hp_match_drop` | 0 | | bit k drops byte-match model k |
| `hp_frozen_scoring` | true | `CYPHA_HP_FROZEN_SCORING` (0/1) | score with learning off inside the hypothetical byte |
| `hp_tree_prune` | 1e-4 | `CYPHA_HP_TREE_PRUNE` (0 = exact) | bit-tree subtrees below this are not expanded |
| `hp_serve_mixer_lr_scale` | 0.5 | `CYPHA_HP_SERVE_MIXER_LR_SCALE` | serve mixer rate, applied by `set_serve_mode`; resolution 1/16 of each rate (floored at 1 before the serve-state fix) |
| `hp_ensemble_learning_rate` | 0.01 | | ensemble weights' exponentiated-gradient rate |
| `hp_lossy_mem`, `hp_serve_compact`, `hp_prune_cold_min_n` | 0, false, 0 | `CYPHA_HP_LOSSY_MEM`, `CYPHA_HP_SERVE_COMPACT`, `CYPHA_HP_PRUNE_COLD_MIN_N` | older RAM levers ([plan](CYPHALM_LOSSY_LLM_PLAN.md)) |
| `vocab_size` | 256 | | `apply_hp_production_recipe` now always sets 256 |

Tier contents (`apply_hp_lossy_tier`): `lean` drops 8 wiki context models,
keeps 8 pool slots, match 22 / pool 20 bits; `balanced` / `compact` /
`small` / `tiny` add context caps 23 / 22 / 21 / 20 (small: match 21; tiny:
match 20, pool 18); `slim` = lean's drops + 13 more context models
(`slim_cm_drop()`), pool and Hebbian 16 bits, match 22 bits, match drop
`0xd06`. Env vars are applied in `CyphaLMModel::init_components`, so they
override the values stored in a loaded checkpoint.

Other env vars: `CYPHA_HP_MMAP=0` copies tables into RAM instead of mapping
the `.hpbin` (Linux only; tables ≥ 64 KiB are mapped). `CYPHA_HP_ENSEMBLE_THREADS=0`
scores members serially. `CYPHA_HP_LEGACY_BYTE_LOGPROBS=1` uses the old
256-clone scoring. REST: `CYPHALM_CHECKPOINT` / `CYPHA_LM_CHECKPOINT` /
`CYPHA_SEQUENCE_CHECKPOINT` or `cypha_rest --cyphalm-checkpoint` auto-load a
checkpoint or manifest at start.

### Decoding (`DecodeParams`, REST `POST /generate` and `/generate/stream`)

REST body fields that map to the table in *Decode controls added*:
`temperature`, `top_k`, `top_p`, `min_p`, `word_candidates`, `word_no_repeat`,
`no_repeat_ngram`, `no_repeat_window`, `learn_from_output`, `exact_greedy`,
`prompt_score_bytes`, `restore_mixing`, `seed` (42). Unspecified fields take
the `DecodeParams` defaults (before `ee1325c` REST hard-coded temperature 0.9). `POST /sequence/load
{"checkpoint_path": ...}` loads a checkpoint or ensemble manifest.

### Library API added

| API | what |
|---|---|
| `CyphaLMModel::reset_stream(keep_history)` | new stream, learned tables kept (priming uses `true`) |
| `CyphaLMModel::set_serve_mode(on)` | serve mixer rate on / off (generation on, training off) |
| `CyphaLMModel::add_ensemble_member(model, w)` | attach a pretrained model for serving |
| `CyphaLMModel::attach_infinigram(path)` | ∞-gram expert (+ `path.weights.json`) |
| `CyphaLMModel::fold_hp_tables(cm, match, pool, drop, hebb, match_drop)` | fold / drop on the model and every member |
| `load_cyphalm_model` / `save_cyphalm_ensemble_manifest` | checkpoints and manifests |
| `HpSequenceBackend::set_frozen_scoring`, `set_tree_prune`, `set_learning`, `set_ensemble_learning_rate`, `ensemble_weights`, `infinigram_weights`, `set_infinigram_weights`, `neural_learning_rate` | serve knobs |
| `HpSequenceBackend::is_composite` | any mixing stage active; `log_prob_byte`, `serve_greedy_next_byte`, `serve_sample_next_byte` then use the full mix, and reuse the distribution `next_byte_log_probs` served at this position (as `observe_next_byte` does) |
| `hp::Predictor::set_learning`, `learned_digest`, `set_serve_adaptation`, `fold_tables`, `reset_stream_state` | hp side |
| `hp::Predictor::tables_match(src)` | every mergeable table (context, Hebbian, pool, mixer) the same size as in `src`. `merge_shard_tables` and `transfer_tables_from` return `false` and change nothing when it is false; `hp::merge_predictor_tables` returns `MergeStatus::ConfigMismatch` |
| `hp::Predictor::fold_auto(occ)` / `HpSequenceBackend::fold_auto(occ)` | occupancy fold; outside (0, 1) the hp side folds nothing and the backend throws `std::invalid_argument` |
| `hp::StreamRewind` | exact rewind of one or more predictors across bytes |
| `hp::MixerNet::set_rate_scale`, `rate_q4`, `layer1_rate_q4` | serve rate scale; effective rates in 1/16ths (`kMixerRateFrac`), checkpoints keep trained rates |
| `HpSequenceBackend::set_session_cache(on, eta, window)` | session cache over the last `window` bytes (default `kSessionWindow` 1 MiB, 0 = all); `session_size` (bytes read), `session_held`, `session_indexed`, `session_builds` |
| `HpSequenceBackend::mixing_state`, `set_mixing_state`, `set_mixing_learning` | snapshot / restore every mixing weight (ensemble, ∞-gram, session, neural; members too); freeze their learning |
| `CyphaLMModel::serve_observe(token)` | score with the full served distribution, then advance (prompt scoring) |
| `generate_beam` | beam search on the full served distribution, replayed under `hp::StreamRewind` with learning off; shared bytes committed per `learn_from_output`; losses from the searched distribution |
| `HpSequenceBackend::reset()` | cold model: also ∞-gram / neural weights back to their start values, experts re-initialised, session emptied |
| `InfiniGram::is_index_file(path)` | stored index (IGR1/IGR2) or plain corpus |
| `hp::MapScope` | map large tables from the `.hpbin` on load |
| `InfiniGram::build` / `InfiniGram(path)` / `query` | index build, load, longest-suffix counts |

### File formats

| file | format |
|---|---|
| `BASE.json` | `{"algorithm": "hp", "config": {...CyphaLMConfig...}, "hp_checkpoint": "BASE.hpbin", "train_step_count", "note"}` |
| `BASE.hpbin` | magic `HPCP` + version. v1: original. v2 (`39ddaf9`): + sentence memory. v3 (`74a2a13`): context slots packed to 16 bits (10-bit state + 6-bit checksum, was 16 + 8). v4: mixer layer-1 scale and skip. v5: the optional upstream context models (`hp_extra_cms` ≠ 0 only). Older versions convert on load; saves are v4 (v5 with extra context models). Folded models store their caps in the JSON and reload at the smaller size; per-table sizes (`fold_auto`) come from the file, and a model saved dropped loads dropped. `load_cyphalm_model` throws on a bad magic, an unknown version, a truncated file, a mixer or table whose shape does not match the JSON config, and a missing `.hpbin` next to a JSON that names one (`hp_checkpoint` / `"algorithm": "hp"`; legacy JSON-only checkpoints still load). `HP_CKPT_SIZES=1` prints each component's size on save (byte-match `match_[]` and word-match `wmatch_[]` separately). |
| ensemble manifest | `{"cyphalm_ensemble": 1, "members": [{"checkpoint": "a.json", "weight": w?}, ...], "learning_rate": r?, "infinigram": "x.igr"?, "note"?}`. First member is the primary. Paths are relative to the manifest. Member weights default to 1/N each (the primary keeps the rest). `learning_rate` overrides `hp_ensemble_learning_rate`. |
| `X.igr` (IGR2, `2b8b1b9`) | `"IGR2"`, uint64 n, uint64 bits (= ceil(log2 n), 27 for 95 MB), n text bytes, zero pad to 8, suffix array bit-packed little-endian at `bits` each, 8 bytes pad. `mmap`ed read-only and shared. IGR1 (`46592ff`): `"IGR1"`, uint64 n, text, pad, n uint32 entries. Not on Windows. |
| `X.igr.weights.json` | `{"weights": [3 per bucket × 256 buckets], "learned_on", "offset", "note"}` from `--ig-weights-out` |

∞-gram expert constants: 3 parts (model, longest suffix, longest suffix with
≥ 16 occurrences), starting weights 0.8 / 0.1 / 0.1, learning rate 0.3,
256 buckets = 8 match-length × 4 count × 4 model-confidence × 2 top-byte
agreement.

### Tests (CTest)

| test | checks |
|---|---|
| `native_hp_bit_tree_smoke` | bit-tree scoring equals fresh-copy scoring; no speculative leak |
| `native_hp_frozen_smoke` | frozen scoring normalised and leak-free; generation keeps the trained model |
| `native_hp_stream_rewind_smoke` | checkpoints byte-identical after 60 random rewinds; 1000 later distributions match |
| `native_hp_fold_smoke` | folded + dropped model saves, reloads identically, checkpoint shrinks |
| `native_hp_checkpoint_roundtrip_smoke` | lossy knobs survive save/load; reloaded predictor continues exactly |
| `native_hp_checkpoint_robust_smoke` | bad magic / version, truncated and missing `.hpbin` throw; mixer shape and table mask checks; dropped context / match models reload dropped (and live ones into dropped builds); folded context / pool / Hebbian merges refused unchanged; `fold_auto` range |
| `native_cyphalm_shard_train_threads`, `native_cyphalm_shard_train_worker_error` | `--threads 0` is an error; a failing shard is reported, no manifest |
| `native_cyphalm_trace_missing_text` | a missing `--text` is reported before indexing |
| `native_cyphalm_lm_quality_fold_auto_range` | `--fold-auto` outside (0, 1) is an error |
| `native_cyphalm_ensemble_smoke` | mix equals a hand-computed blend; manifest load equals direct build; lookahead leaves members unchanged; `log_prob_byte` / greedy / sampling read the served mix, cold or reused, without changing what is learned |
| `native_cyphalm_neural_smoke` | LSTM / Transformer experts against a reference; mix normalised; `log_prob_byte` / greedy use the neural and session mix |
| `native_cyphalm_lm_quality_load_only_flags` | a load-only harness flag without `--load` is an error |
| `native_cyphalm_infinigram_smoke` | index queries against brute force; expert normalisation, observe, rewind |
| `native_cyphalm_serve_state_smoke` | serve rate scale in 1/16ths (×1 bit-identical, ×0.5 ≠ ×0.25 on lr1_scale 40, checkpoint keeps trained rates); session cache window, doubling cadence, no rebuild under `StreamRewind`, exact rewind, `reset_stream(false)` clears; `reset()` scores like a fresh twin (∞-gram, adapting expert, session); top-p at T 1e-5 equals greedy |
| `native_cyphalm_serve_mixing_smoke` | manifest-style composite: no prompt scoring leaves the start weights; scoring adapts ensemble / ∞-gram / neural weights with the learned tables unchanged, `restore_mixing` puts them back; `set_mixing_learning(false)` freezes them; beam width 1 starts with the served argmax (not the primary's), equals greedy bytes and losses; beam learns only the prompt, or the output with `learn_from_output` |
| `native_cyphalm_generate_infinigram_whole`, `native_cyphalm_generate_infinigram_bytes` | `cyphalm_generate` warns about a plain corpus indexed whole; `--infinigram-bytes` silences it |

### Changelog

| commit | change | details |
|---|---|---|
| `68bcfdf` | vendored hp reduced to gate24 (289 flags resolved, 15,706 → 6,687 lines) | [strip](CYPHALM_HP_GATE24_STRIP.md) |
| `9c14efd` | serve leak fixed (DMC, word-match undo); bit-tree DFS ~1.85× faster; lossy `hp::Config` knobs | [lossy](CYPHALM_LOSSY_MIXER_REPORT.md) |
| `8ada3c0` | demand-zero table pages (construct ~55 ms, ~80 MB) | lossy |
| `a88a5ab` | tiers `lean` … `tiny` | lossy |
| `02aefd6` | context-slot prefetch removed (noise) | lossy |
| `ae92146` | `cyphalm_lm_quality`; `exact_greedy`; `Predictor::set_learning`, `learned_digest` | Distribution quality |
| `7eaff1e`, `f9c24d0` | frozen scoring, then default on; `min_p`, `no_repeat_ngram`; byte LSTM (later removed) | Frozen scoring |
| `e6cc556`, `01f836a`, `f29bba6` | first report; LSTM mixture generation; LSTM speed-up | LSTM expert |
| `d05523b` | fix: generation replaced the trained predictor with an untrained one | Generation |
| `8ede50f` | decode defaults T 0.8, min-p 0.1, no learning from output | Generation |
| `bf34a87`, `ee1325c` | `cyphalm_generate --load` and decode flags; REST decode fields | Tools |
| `bbcb383` | byte LSTM and `index_output` removed | LSTM expert |
| `95ff0a4` | 95 MB scaling | Scaling |
| `b134076` | priming keeps byte history; `--reset-stream`, `--serve-lr`, `--serve-skip`, `--epochs`, two-model `--ensemble` | Serving improvements |
| `73ef7e6` | serve mixer rate 0.5; `--word-k` prototype (model copy) | Serving improvements |
| `a8ce4e7` | `hp::StreamRewind`; word lookahead, default K 8 | Serving improvements |
| `4137015` | report round 2 | |
| `fba6453` | `--train-offset`, ensemble weight grid; StateMap / APM raw data | Other results |
| `bbb3b65` | lookahead shares prefix distributions (37 → 12.7 ms/byte) | Serving improvements |
| `936b0e4` | serve-time ensembles; `--ensemble` (generate), `--member` (harness) | Ensembles |
| `2587ef1` | `cyphalm_gen_bench` | Ensembles |
| `39ddaf9` | checkpoint v2 (sentence memory, word-match window); vocab 256 | Other results |
| `9d1f295`, `7cbc201` | report: ensembles, epochs; primary-proposal lookahead not kept | Ensembles |
| `0f290f4` | learned ensemble weights; threaded member scoring; `--ensemble-lr`; two-model `--ensemble` path removed | Ensembles |
| `75cc6da`, `9f7893c` | report: 11 shards; generation benchmark | Ensembles |
| `cc02ccc` | ensemble manifests; `cyphalm_shard_train` | Ensembles |
| `74a2a13` | table folding; mapped loading; packed slots, checkpoint v3 | RAM |
| `a5a7784` | serve-time context-model drop; Hebbian fold; `--merge`; `hp_fold_smoke` | RAM |
| `8ae22c6`, `0462ddb`, `c0587fb` | `slim` tier; frontier; frozen tables / members not kept | RAM |
| `3aafeee`, `4bcea2b` | droppable match models; `slim` drops five; folding reaches members | RAM |
| `46592ff`, `29f1947`, `317c645`, `2d30e6a` | ∞-gram expert, IGR1, weights save/load, backoff / weights / generation measured, agreement buckets | ∞-gram |
| `d7694b5` | bit-tree pruning 1e-4 | Speed round |
| `2b8b1b9` | IGR2 packed index | Speed round |
| `c02a6d0`, `2a003bf` | winner manifests, rebuild script; report | Speed round |

### Negative results

| tried | result | code |
|---|---|---|
| byte LSTM expert + learned gate | +0.003 / +0.004 wiki / Alice (`m_*`) | removed `bbcb383`; recover from `ee1325c` (`byte_lstm.hpp/.cpp`, harness `--lstm-hidden`) |
| `index_output` (no match copying from generated bytes) | no effect | removed `bbcb383`; recover from `ee1325c` (`DecodeParams::index_output`, `Predictor::set_history_indexing`) |
| learning from own output while generating | loops, d4 0.23–0.59 (`g4_*`) | kept as `learn_from_output` (off) |
| greedy, greedy + no-repeat | cycles; no-repeat breaks words (`g4_*`) | kept as options |
| bug-era generations ("letter salad") | withdrawn, caused by `d05523b`'s bug (`g2_*`, `g3_*`) | fixed |
| serve mixer rate ≠ 0.5, no small-error skip | ×0.25 ≈ ×0.5, faster worse; skip 0 no reliable gain (`ad_*`, `aq_*`). ×0.25 then floored most layer-1 rates at 1 (see *Adapt to the prompt*) | kept as `--serve-lr`, `--serve-skip` |
| serve-time StateMap limits, APM rates | ≤ 0.004, wiki and Alice move opposite (`lim_*`) | harness code never committed |
| two training epochs | +0.029 wiki (`ep2_*`) | `--epochs` kept |
| table bits 24 | 8 MiB: 1.7962 vs 1.7950 (`t24_*`); 95 MB: −0.008 wiki, none on Alice (`t24_95_*`) | config only |
| linear, fixed-share and grid ensemble weights | linear 1.7352 vs geometric 1.7248; fixed-share worse (`ens_*`, `div_*`) | removed `0f290f4`; recover `git show 0f290f4^:native/tools/cyphalm_lm_quality.cpp` |
| word best-of-K by sum / mean without ban / SIR / soft ban (model-copy prototype, ~400 ms/byte) | sum 0.52 / 0.229, mean 0.54 / 0.441, SIR 1.351 / 0.845 judge / d4 (`wk_*`, `wk2_*`) | replaced by the library decoder in `a8ce4e7`; recover from `73ef7e6` |
| candidate words from the primary, chosen by the ensemble | 1.272 / 0.735 vs 1.158 / 0.789 | not committed; no raw JSON |
| training `slim` from scratch vs converting | equal (1.8341 vs 1.8327) | both work |
| folding context tables; match tables of a 95 MB model | +0.013 / +0.020 at 22 bits; +0.020 / +0.013 at 16 bits (`fd_*`, `pa_*`) | `--fold` kept |
| merging all 11 shards into one table set | 2.006 wiki (`mg11_*`) | `--merge` kept |
| freezing big tables while serving | +0.036 to +0.079 wiki (`fz_*`) | harness code never committed |
| frozen ensemble members | +0.06 wiki, +0.11 Alice (`slf_*`) | harness code never committed |
| ∞-gram as one fixed-λ linear mix (first analysis) | best λ 0.20: 1.7000 wiki vs 1.6925 for the shipped 3-part mix (`ig_*`, `ig2_*`, `igl_*`) | never committed |
| ∞-gram fourth backoff level (≥ 256 occurrences) | ~0.001 worse (`ig4_*`) | never committed |
| ∞-gram pretrained starting weights | −0.0006 online; frozen domain-bound (`igw_*`) | `--ig-weights-out` kept, none shipped |
| smaller ∞-gram corpus (48 / 24 MB) | +0.011 / +0.027 wiki (`isz_*`) | index size only |
| pruning at 1e-3 | +0.002 wiki, +0.004 Alice (`pr_1e-3_*`) | 1e-4 kept |
| ensemble temperature | ~0.9 on wiki (−0.004), 1.0 on Alice | not added |
| context-slot prefetch | 237.0 s vs 239.6 s (noise) | removed `02aefd6`; recover from `9c14efd` |

### Raw JSON index (`lm_quality/`)

| prefix | experiment |
|---|---|
| `q_g_*`, `q_l_*` | gate24 / lean 8 MiB distribution quality, frozen, exact vs frozen scoring |
| `g2_*`, `g3_*` | bug-era generations (withdrawn); `g4_*`: decode grid after the fix |
| `m_*` | byte LSTM mixture |
| `s95_*`, `t24_*`, `t24_95_*` | 95 MB scaling; table bits 24 |
| `rs_*`, `ad_*`, `aq_*`, `val_*`, `lim_*` | stream reset; serve mixer rate (half units / quarters); checks; StateMap / APM |
| `wk_*`, `wk2_*`, `wl_*` | word best-of-K prototype variants; library lookahead K 4 / 8 |
| `div_*`, `ens*`, `ep2_*` | two-model ensembles vs 16 MiB; weight schemes; two epochs |
| `ne_*`, `fx_*`, `elr*`, `sm_*`, `e11*`, `thr_*` | library ensembles; re-measured after the checkpoint fixes; weight learning rate; table-bits-20 shards; 11 shards; threads on / off |
| `gb_*`, `gb2_*`, `gb11_*`, `gbig_*` | generation benchmark runs 1 and 2; 11 shards; ∞-gram generation |
| `mm_*`, `pk_*`, `cmp_*` | mapped loading; packed slots (8 MiB eval, 2 MiB compress old / new) |
| `fd_*`, `dr_*`, `dc_*`, `cb_*`, `ef_*`, `pa_*`, `a3_*`, `slim8_*` | fold screen; per-context-model drop (bit N); drop sets; fold+drop combos; 11-shard fold presets; packed frontier; serve-time slim conversion; trained slim |
| `mg*`, `fz_*`, `sl_*`, `slf_*`, `md_*`, `mdc_*`, `t18b_*` | merge; frozen tables; slim ensembles mapped; frozen members; per-match-model drop (bit N); match drop sets (A = slim); 11 slim shards at table bits 18 / 20 |
| `ig_*`, `ig2_*`, `igl_*`, `eig_*`, `ig4_*`, `igag*`, `igw_*`, `isz_*`, `igv2_*` | ∞-gram: fixed λ; shipped mix; ensemble + index; 4th level; agreement buckets (`igag8` shipped); starting weights; index size; IGR2 |
| `pr_*`, `win_*` | bit-tree pruning; the winner manifests |
