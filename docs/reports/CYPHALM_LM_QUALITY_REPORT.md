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
  on, i.e. while reading prompts, never while generating. For 95 MB + 8 MiB
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

The judge was trained on the first 16 MiB, so it favours models trained there.
Compare decoders on one model, not models against each other (held-out NLL
does that). ms/byte figures come from a loaded 4-core box.

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
| **95 MB model, slim drop + pool/Hebbian 16** | **577 MB** | **1.7729** | **2.2445** |
| 95 MB model, packed | 819 MB | 1.7660 | 2.2263 |
| 95 MB model, unpacked (before) | 1071 MB | 1.7608 | 2.2158 |
| 5 shards, context 22 + match/pool 16 | 810 MB | 1.7717 | 2.2301 |
| 3 shards, match/pool 16 | 1.03 GB | 1.7629 | 2.2178 |
| 5 shards, match/pool 16 | 1.72 GB | 1.7399 | 2.2039 |
| **11 shards, slim drop + pool/Hebbian 16** | **4.0 GB** | **1.7069** | **2.1936** |
| 11 shards trained slim (`cyphalm_shard_train --tier slim`) | 3.9 GB | 1.7058 | 2.1949 |
| **95 MB slim + 11 slim shards** | **4.5 GB** | **1.6916** | **2.1812** |
| 11 shards, packed | 5.04 GB | 1.7009 | 2.1834 |
| 11 shards, unpacked (before) | 7.2 GB | 1.6994 | 2.1797 |

- **Up to ~1 GB,** one model trained on all the data is the best use of
  memory. Slim plus packing takes the 95 MB model from 1071 to 577 MB (−46%)
  for +0.012 wiki / +0.029 Alice.
- **Above ~1 GB,** ensembles of shard models win. The 11-shard ensemble drops
  from 7.2 to 4.0 GB (−44%) for +0.008 / +0.014.
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

Serve-time model settings: `hp_frozen_scoring` (on), `hp_serve_mixer_lr_scale`
(0.5), and priming keeps the byte history.

## Reproduce

```bash
Q=native/build/cyphalm_lm_quality
$Q --tier lean --train enwik8 --train-bytes 8388608 --save /tmp/pre_lean
$Q --load /tmp/pre_lean.json --eval enwik8 --eval-offset 96000000 --eval-bytes 16384 --compare-scoring
$Q --load /tmp/pre_lean.json --eval alice29.txt --eval-offset 20000 --eval-bytes 16384 --frozen-eval --gen-bytes 0
# ensembles: shards trained on disjoint slices, then mixed
$Q --tier lean --train enwik8 --train-offset 48000000 --train-bytes 8388608 --save /tmp/pre_48m
$Q --load /tmp/pre_lean.json --member /tmp/pre_48m.json --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
native/build/cyphalm_gen_bench --load /tmp/pre_lean.json --judge /tmp/pre_16.json --text enwik8 --offset 96000000 --prompts 8
# serving: history on reset, mixer rate (quarters of trained), word lookahead
$Q --load /tmp/pre_lean.json --reset-stream full --frozen-eval --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
$Q --load /tmp/pre_lean.json --serve-lr 2 --eval enwik8 --eval-offset 96000000 --eval-bytes 8192 --gen-bytes 0
$Q --load /tmp/pre_lean.json --eval enwik8 --eval-offset 96000000 --eval-bytes 16384 --gen-bytes 400 --only-default --word-k 8
```
