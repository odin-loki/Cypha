# Generation and online learning: everything tried, 1–4 November 2026

*Odin Loch · 2026-11-04*

Odin asked: "Make sure you document everything we have tried and the results." This is that record. The detail, rules and predictions for each item are in [GENERATION_PLAN.md](../GENERATION_PLAN.md) §13.10–13.24 and the reports linked below. Every rule was registered before its run. Choices were made on dev; each adopted change got at most one test read.

## Where things stand

| Task | What CyphaLM uses now | Score |
|---|---|---|
| Writing | **K5**: word-level Kneser–Ney of order 5 on 7 MB of clean prose (`sample --k5 CLEAN.txt`), 12-word copy cap | GEN-A test SCRIBE v2 **0.305** [0.276, 0.338]; dev 0.353 |
| Writing, bytes | The byte sampler (`sample`), no LSTM, no GLN | dev 0.174 at T 0.6; test 0.147 |
| Prediction | `context8k`: online chain (exponentiated gradient), byte LSTM stage, slow and fast chains, online final layer with the copy gate | **1.6238** bits per byte (holdout second half), 269 MB |
| Prediction before 1 November | the frozen chain | 1.6979 |
| Answers | unchanged by all of the above | — |

The two headline facts:
- Writing roughly doubled: 0.147 → 0.305 on test.
- Prediction fell by 0.074 bits per byte.

They came from different changes. What predicts better did not write better, in every experiment.

## 1. Writing (SCRIBE v2 on GEN-A dev; human text scores 0.923)

| Experiment | Section | Best result | Decision |
|---|---|---|---|
| Byte sampler (BKN2 default, T 0.6) | §13.16 | 0.174 [0.137, 0.209] | Kept as the byte writer |
| **K5** (Python bake-off, then C++) | §13.7, §13.19 | **0.353** [0.318, 0.384]; **test 0.305** | **The recommended writer** |
| 10 fuzzy experts mixed into K5 | §13.7 | all lower (ALL: 0.218) | Stopped for writing |
| Self-organising structure (exchange classes etc.) | §13.9 | exchange classes 0.327 | Not adopted for writing |
| GLN mixed into the byte sampler | §13.15 | 0.065 (10 prompts; W0 0.256) | No: pulled the text into XML |
| GLN by product of experts / gate / clean prose (250 prompts) | §13.16 | gate 0.117 (T 1.0), geo 0.090, geo+clean 0.153 | None non-inferior |
| GLN trained on 1 GiB of C4 | §13.17 | gate 0.143, geo 0.111 | None non-inferior |
| Online lambdas in the sampler | §13.22 | 0.147 | No (−0.027) |
| Byte LSTM in the sampler | §13.18 | gate 0.155 (non-inferior); linear 0.119; geo 0.119 | Was default briefly; taken out (Odin: "go with K5") |
| K5 + byte LSTM | §13.24 | geo 0.149; linear 0.106; DExperts' prior fix 0.221 | No: a length bias, then blander text |

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
| **Byte LSTM as a chain stage and input (D1)** | §13.18 | 1.6311 | **Default** |
| Two learning rates + copy gate (D2) | §13.21 | 1.6489 | **Default** |
| **D1 + D2 (D3)** | §13.21 | **1.6238** | **The default now** |
| The LSTM's state as the final layer's gate (D5) | §13.24 | 1.6345 | No |

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

## 3. Smaller findings and fixes

- GLNB's LSTM predicted from a state one byte stale. Fixed (§13.13).
- The dump column bug (35 → 36) and R2's crash (`--online-final` read arrays only `--dump-experts` filled) were both fixed (§13.12).
- K5's port ordered tied words by frequency, which cost about 0.03 SCRIBE. Ties are now broken by a salted hash (§13.19).
- The 10-prompt "fast mode" screen swings by about 0.2 SCRIBE for the same sampler. Use it to catch failures, never to rank arms (§13.16).

## 4. Still running or never collected

| Item | Where | State |
|---|---|---|
| Wider 3-level GLN on 1 GiB (§13.23) | Odin's PC | Training; its online bits on C4 are 1.93–1.98 against the narrow GLN's 2.004 |
| The 1 TB trie with the online mixer (§13.20) | Odin's PC | Asked; not yet reported |
| R3, the half-space test, GLNB's writing (1 Nov) | Odin's PC, `/mnt/d/cypha_llm/r3_glnb/` | Never collected; asked again with §13.23 |
| K5 upgrades: tuning, online cache and counts, order 6, classes, sentence gate, content plan | Odin's PC | Running → [K5_UPGRADES.md](K5_UPGRADES.md) |
| K5 on clean Wikipedia from enwik9 (7 MB → hundreds of MB) | Odin's PC | Running → [K5_SCALE.md](K5_SCALE.md) |
| Online n-gram expert, document-aware learning, a bigger LSTM | Odin's PC | Running → [ONLINE_LEARNING_UPGRADES.md](ONLINE_LEARNING_UPGRADES.md) |
| Whole-distribution mixing with APM/SSE (PAQ-style) | Odin's PC | Running → [FULL_DISTRIBUTION_MIXING.md](FULL_DISTRIBUTION_MIXING.md) |

This record is updated as those land.
