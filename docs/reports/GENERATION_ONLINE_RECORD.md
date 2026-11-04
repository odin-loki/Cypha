# Generation and online learning: everything tried, 1–4 November 2026

*Odin Loch · 2026-11-04*

Odin asked: "Make sure you document everything we have tried and the results." This is that record. The detail, rules and predictions for each item are in [GENERATION_PLAN.md](../GENERATION_PLAN.md) §13.10–13.24 and the reports linked below. Every rule was registered before its run. Choices were made on dev; each adopted change got at most one test read.

## Where things stand

| Task | What CyphaLM uses now | Score |
|---|---|---|
| Writing | **K5, upgraded** (nucleus 0.8, 256 exchange classes, the prompt-association sentence gate), 12-word copy cap: `sample --k5 CLEAN.txt` | GEN-A test SCRIBE v2 **0.524** [0.493, 0.552] (gate provisional: needs a judge of another kind); plain K5 0.305 |
| Writing, scale | Plain K5 on 330 MB of clean enwik9: test 0.380. The upgraded K5 does not gain from it (dev 0.479 against 0.500 on 7 MB), so the writer stays on 7 MB | test 0.380 (plain K5 only) |
| Writing, bytes | The byte sampler (`sample`), no LSTM, no GLN | dev 0.174 at T 0.6; test 0.147 |
| Prediction, `--full-mix` | the default below, then whole-distribution mixing with SSE (PAQ-style); off by default (twice the time) | **1.5051** bits per byte, about 1.1 GB |
| Prediction | `context8k`: online chain (exponentiated gradient), online n-gram expert, byte LSTM at 256/20, slow and fast chains, online final layer with the copy gate | **1.5793** bits per byte (holdout second half, Odin's PC), 828 MB |
| Prediction, earlier on 4 November | the same without the n-gram, LSTM at 64 (`--no-ngram --lstm-size 64`) | 1.6238 (cloud), 1.6232 (PC), 247–269 MB |
| Prediction before 1 November | the frozen chain | 1.6979 |
| Answers | unchanged by all of the above | — |

The two headline facts:
- Writing on SCRIBE v2 test: 0.147 (bytes) → 0.305 (K5) → 0.524 (upgraded K5). The last step is provisional: SCRIBE v3's new judges do not confirm it (§3).
- Prediction fell by 0.119 bits per byte by default (1.6971 → 1.5793 on Odin's PC), and by 0.192 with `--full-mix` (1.5051).

They came from different changes. What predicts better did not write better, in every experiment.

## 1. Writing (SCRIBE v2 on GEN-A dev; human text scores 0.923)

| Experiment | Section | Best result | Decision |
|---|---|---|---|
| Byte sampler (BKN2 default, T 0.6) | §13.16 | 0.174 [0.137, 0.209] | Kept as the byte writer |
| **K5** (Python bake-off, then C++) | §13.7, §13.19 | **0.353** [0.318, 0.384]; **test 0.305** | The writer |
| K5 tuning: nucleus 0.8 | [K5_UPGRADES](K5_UPGRADES.md) | 0.395 (4 seeds +0.057) | **Adopted** |
| K5: word cache; dynamic counts | [K5_UPGRADES](K5_UPGRADES.md) | 0.114–0.394 (bits down to 1.488) | No: lower bits, lower SCRIBE |
| K5: orders 6 and 7 | [K5_UPGRADES](K5_UPGRADES.md) | 0.349 / 0.409 (noise) | No |
| K5: 256 exchange classes as backoff | [K5_UPGRADES](K5_UPGRADES.md) | 0.444 (4 seeds +0.029) | **Adopted** |
| K5: sentence gate, 8 candidates, prompt association | [K5_UPGRADES](K5_UPGRADES.md) | 0.498; **test 0.524** | **Adopted** (provisional: judge caution) |
| K5 on clean enwik9: 7 / 30 / 100 / 300 / 330 MB | [K5_SCALE](K5_SCALE.md) | 0.350 / 0.375 / 0.363 / 0.382 / 0.390 (4-seed mean); **test 0.380** | **Adopted** as the training text |
| K5 with a 65,536-word vocabulary | [K5_SCALE](K5_SCALE.md) | −0.010 to −0.016 | No |
| Upgraded K5 on 330 MB (C1) | [K5_SCALE](K5_SCALE.md), §13.29 | 0.479 against 0.500 on 7 MB (4 seeds) | No: the writer stays on 7 MB |
| 10 fuzzy experts mixed into K5 | §13.7 | all lower (ALL: 0.218) | Stopped for writing |
| Self-organising structure (exchange classes etc.) | §13.9 | exchange classes 0.327 | Not adopted for writing |
| GLN mixed into the byte sampler | §13.15 | 0.065 (10 prompts; W0 0.256) | No: pulled the text into XML |
| GLN by product of experts / gate / clean prose (250 prompts) | §13.16 | gate 0.117 (T 1.0), geo 0.090, geo+clean 0.153 | None non-inferior |
| GLN trained on 1 GiB of C4 | §13.17 | gate 0.143, geo 0.111 | None non-inferior |
| Online lambdas in the sampler | §13.22 | 0.147 | No (−0.027) |
| Byte LSTM in the sampler | §13.18 | gate 0.155 (non-inferior); linear 0.119; geo 0.119 | Was default briefly; taken out (Odin: "go with K5") |
| K5 + byte LSTM | §13.24 | geo 0.149; linear 0.106; DExperts' prior fix 0.221 | No: a length bias, then blander text |
| The 1 TB trie in the byte sampler | §13.27 | 0.138 (T 0.6) | No (rule 0.154) |
| The wider GLN (1 GiB, 128 × 6 then 16 × 4), gate mix | §13.27 | 0.143 | No (reported only) |
| GLNB writing (1 November) | §13.27 | 0.046 | No |

What the writing results say:
- **Whole words counted on clean prose** is the one change that moved writing a lot: K5. Clean text alone on bytes failed, and tokens alone on raw XML failed (GEN-A, T4).
- **Mixing in any second model lowered SCRIBE**: fuzzy experts, GLN, LSTM, online weights. Predictors hedge, and samples drawn from a hedged mixture are bland. The gate (contrastive decoding's plausibility constraint) limits the damage and helps at T 1.0, but never beats the plain writer at its best temperature.
- **A byte model inside a word model has a length bias:** it favours short, common words. Correcting for the model's own prior (DExperts' form) fixes the length but not the score.

## 2. Prediction (bits per byte, `context8k`'s holdout second half)

| Experiment | Section | Result | Decision |
|---|---|---|---|
| Frozen chain | — | 1.6979 (cloud), 1.6971 (Odin's PC) | The old default |
| Online lambdas, AdaGrad (R1) | §13.12 | 1.6966 (PC) | Not default (rule 0.01 missed) |
| Why: AdaGrad's step set by a few bytes (top 0.1% held 60–99.9% of the squared gradients) | §13.15 | — | The spike |
| **Spike fix: exponentiated gradient** (dump) | §13.15 | 1.6920; 1.6692 with short copies in scoring | **Adopted** |
| Clipped gradients; log-odds updates; RMS | §13.15 | worse or no better | Not adopted |
| Online chain (C1) | §13.15 | 1.6714 | **Default** |
| + online final layer (C2) | §13.15 | 1.6543 | **Default** |
| + GLN, bitwise GLN with an LSTM (C3) | §13.15 | 1.6214 | Default 3–4 Nov, then off (Odin) |
| GLN + the mixer's feedback (C4, recurrent) | §13.15 | 1.6213 | No gain |
| GLN trained on 1 GiB of C4 (P1) | §13.17 | 1.6403 against 1.6206 on the PC | Worse: wrong domain |
| Wider GLN on the same 1 GiB (P2b) | §13.27 | 1.6359 | Still worse than the 7 MiB GLN |
| The 1 TB trie (B1 against B0, the C2 chain without the GLN) | §13.27 | 1.6473 against 1.6534 | **Recommended model file** (rule 0.005); not yet run on the current default |
| **Byte LSTM as a chain stage and input (D1)** | §13.18 | 1.6311 | **Default** |
| Two learning rates + copy gate (D2) | §13.21 | 1.6489 | **Default** |
| **D1 + D2 (D3)** | §13.21 | **1.6238** | **The default now** |
| The LSTM's state as the final layer's gate (D5) | §13.24 | 1.6345 | No |
| *Odin's PC from here; its D3 (B0) is 1.6232* | [ONLINE_LEARNING_UPGRADES](ONLINE_LEARNING_UPGRADES.md) | | |
| Online n-gram expert, PAQ-style, 22 / 24 table bits (N1, N2) | §13.26 | 1.5986 / 1.5913 | N2 **adopted** |
| Larger online steps after each page start (D1, D2) | §13.26 | 1.6246, 1.6239 | No |
| Bigger LSTM: 128/50, 256/100, 256/20, 128/20 | §13.26 | 1.6132, 1.6097, **1.6026**, 1.6118 | 256/20 **adopted** |
| **N2 + LSTM 256/20 (C)** | §13.26 | **1.5793** (+61% time, 828 MB) | **The default now** |
| Whole-distribution mixing with SSE, ABs, on the old default (X1) | §13.29 | 1.5499 against 1.6232 | Rule holds |
| **The same on the new default (X2)** | §13.29 | **1.5051** against 1.5793 (−0.074; twice the time, +224 MB) | Rule holds; flag `--full-mix`, off until Odin decides |

Alone, as experts:
- the GLN scores 1.94 bits per byte (7 MiB of counts, the fixed LSTM);
- the GLN on 1 GiB of C4 scores 2.18 on Wikipedia;
- the byte LSTM scores 2.53–2.65.

Each is weak alone and still useful in the mixture.

What the prediction results say:
- **The online mixer was crippled by its optimiser, not by being online.** Once the step could not be frozen by one spike, learning online beat frozen lambdas by 0.027.
- **Weak, different models help a mixture.** The LSTM gains 0.023 for 4 MB; the GLN gained 0.033 for 120 MB.
- **More gates do not help** (gate24, GLNX, D5). A better choice of gate context (copy × byte class × long match) helps a little.
- **Scale on the wrong text hurts.** 1 GiB of web text made the GLN worse on Wikipedia.
- **The biggest single gain was an old idea:** a PAQ-style bitwise n-gram learning online (−0.032). With the bigger LSTM it gives −0.044, 85% of the two gains added.
- **Faster learning at page starts adds noise.** The online steps already move fast enough.
- **The biggest gain of the week was in how experts are combined, not in a new expert:** PAQ's nonlinear mixing and SSE on whole distributions gave −0.074 on top of everything else.

## 3. The judge: SCRIBE v3 ([SCRIBE_V3](SCRIBE_V3.md), §13.26)

Odin: "Make another two metrics with different maths that complement each other and take the average of all 3. Make it fool proof."

| Judge | Maths | Fresh ladder (5 pairs) |
|---|---|---|
| v2 | word vectors, ECDF z-scores, four axes | 5 of 5 |
| SURPRISE | code length under a model of the pack, two-sided | 5 of 5 |
| MATCH | suffix-array matches with the pack, LZ76, prompt echo | 4 of 5 (human > sentences shuffled 0.940) |
| **v3** = mean of the three | | **5 of 5** |

- **Not fool-proof yet.** Of 9 attacks, 5 are caught (copying, echo, loops, co-occurrence collages from the pack).
- Missed: keyword stuffing (into human text 0.825, into generic text 0.457), a generic paragraph (0.930), copied prompt sentences then a continuation (0.948). In the last two, one or two judges catch it and the mean dilutes it; `min3` and `disagree` show it.
- Sentences reordered within a text are invisible to all three.
- **The K5 sentence gate is not confirmed.** Old K5 → upgraded K5: v2 +0.145, SURPRISE −0.106, MATCH −0.009, v3 +0.011 (inside the intervals). SURPRISE finds the gated text too predictable.

**SCRIBE v3.1** ([SCRIBE_V31](SCRIBE_V31.md), §13.28): the geometric mean of the three judges, plus detectors for keyword stuffing and bland text.
- Fresh read (enwik9): ladder 5 of 5, registered attacks **9 of 9** (v3: 5 of 9), human 0.904.
- **8 new attacks held out until the fresh read: 2 of 8 caught.** Stuffing at one word in 7, a template with the prompt's names, a title swap and loosely related sentences get through.
- K5 gate: v3.1 +0.048, most of it from v2. Still not confirmed by a judge of another kind.
- v3 does not confirm the old K5's gain from 330 MB of text either (v2 +0.046, v3 +0.001). Both of 4 November's writing gains rest on v2 alone.

## 4. Smaller findings and fixes

- GLNB's LSTM predicted from a state one byte stale. Fixed (§13.13).
- The dump column bug (35 → 36) and R2's crash (`--online-final` read arrays only `--dump-experts` filled) were both fixed (§13.12).
- K5's port ordered tied words by frequency, which cost about 0.03 SCRIBE. Ties are now broken by a salted hash (§13.19).
- The 10-prompt "fast mode" screen swings by about 0.2 SCRIBE for the same sampler. Use it to catch failures, never to rank arms (§13.16).

## 5. Still running or never collected

| Item | Where | State |
|---|---|---|
| Wider 3-level GLN on 1 GiB (§13.23) | Odin's PC | **Done**: 1.6359, writing 0.143; GLN stays off (§13.27) |
| The 1 TB trie with the online mixer (§13.20) | Odin's PC | **Done**: −0.0061 (recommended file); writing 0.138, no (§13.27) |
| R3, the half-space test, GLNB's writing (1 Nov) | Odin's PC | **Collected**: R3 1.6971 / 1.6678; best half-spaces 4 → 6 as data grows; GLNB writing 0.046 (§13.27) |
| K5 upgrades | Odin's PC | **Done**: test 0.524 → [K5_UPGRADES.md](K5_UPGRADES.md) |
| K5 on clean Wikipedia from enwik9 | Odin's PC | **Done**: test 0.380 → [K5_SCALE.md](K5_SCALE.md) |
| Upgraded K5 on the 330 MB text (the two together) | Odin's PC | **Done**: 0.479 against 0.500, rule fails; stays on 7 MB (§13.29) |
| The 1 TB trie under the current default (n-gram, LSTM 256/20) | Odin's PC | Not run |
| A judge of another kind for the sentence gate (Odin's blind ratings) | Odin | Next |
| Online n-gram expert, document-aware learning, a bigger LSTM | Odin's PC | **Done**: 1.5793 → [ONLINE_LEARNING_UPGRADES.md](ONLINE_LEARNING_UPGRADES.md) |
| SCRIBE v3 | cloud | **Done**: ladder 5 of 5, attacks 5 of 9 → [SCRIBE_V3.md](SCRIBE_V3.md) |
| SCRIBE v3.1: catch keyword stuffing and generic paragraphs; fresh read on enwik9 | cloud | **Done**: registered attacks 9 of 9, held-out 2 of 8 → [SCRIBE_V31.md](SCRIBE_V31.md) |
| Whole-distribution mixing with APM/SSE (PAQ-style) | Odin's PC | **Done**: 1.5051 on the new default (−0.074); behind `--full-mix` → [FULL_DISTRIBUTION_MIXING.md](FULL_DISTRIBUTION_MIXING.md) |
| `--full-mix` on by default (twice `context8k`'s time) | Odin | To decide |

This record is updated as those land.
