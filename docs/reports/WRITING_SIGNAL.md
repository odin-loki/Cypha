# The signal coherent text needs: which text signals agree with the blind raters

**Odin Loch** · 2026-11-12 · registered before any comparison with the raters · branch `wip/writing-signal` (from main fb79cc93) · GENERATION_PLAN §13.14 (PLAN ledger 103), §13.31–13.36, §13.48–13.54

## Why

- §13.14 asks which measurable signal writing needs. Prediction (bits per byte) is the wrong target for writing.
- SCRIBE has disagreed with the blind raters again and again. It rose for topic stuffing (§13.34), it did not see the 337 MB gain (§13.48), and it rose most for R2, which the raters marked down (§13.53).
- So a signal that picks what the raters pick is worth having: to screen arms before a rating round, and later as a target to optimise.

## The data

- **Judgments:** every committed blind round in `bench/blind/` that has packets, the raters' replies and a key. 14 rounds: `coh_long`, `coh_pair`, `fill_pair`, `gq_pair`, `ng_pair`, `p1t_pair`, `planner_pair`, `q1t_pair`, `q3_long`, `q3_pair`, `tw_pair` (pairwise choice) and `llm_rating`, `llm_rating2`, `llm_rating3` (1–10 scores). `q1_pair` and `q2_pair` have no committed replies and are left out.
- **One judgment** is one rater's verdict on two texts. In a pair round it is the choice. In a rate round, any two texts of one item with different scores make a pair, the higher score preferred.
- **The texts are taken from the packets**, as the raters saw them (cut near 460 characters, or 1,500 for the long rounds).
- **Left out:** pairs of identical texts (714; the arms wrote the same text) and rater ties (1,805, almost all from the rate rounds' floor at 2/10).
- **Used:** 1,431 writer-against-writer judgments and 1,200 human-against-writer judgments (rate rounds only).
- **The human-against-writer gap** is measured on the 250 GEN-A dev prompts: the human continuation against every dev GEN file of `bench/gen_quality`, `bench/tune_write`, `bench/k5_coherence` and `bench/k5_planner` (42 files), all cut near 460 characters. Today's default is L1 (`bench/k5_coherence/runs/L1.gen.txt`).

## The signals (registered)

`scripts/surprise_profile.py` (new). The surprise signals use a word trigram Kneser–Ney model (interpolated, discount 0.75) over `fuzzy/train.txt` (7 MB, K5's own text; 1.36 M tokens, 31,668 words). The prompt's last two words are the first context. A content word has 3+ letters and is not on a list of 160 function words. "Human median" is the median over the 250 human dev continuations.

| # | Signal | Reads better when | What it is |
|---|---|---|---|
| 1 | `surp_mean` | lower | mean surprise per word (bits) |
| 2 | `surp_sd` | higher | spread of per-word surprise |
| 3 | `surp_ac1` | higher | lag-1 autocorrelation of surprise (burstiness) |
| 4 | `surp_cf` | higher | surprise of content words minus function words |
| 5 | `burst_cw` | higher | share of content words that repeat an earlier one |
| 6 | `commit_text` | higher | share of sentences 2+ that reuse a content word of an earlier sentence |
| 7 | `commit_prompt` | higher | share of sentences that reuse a content word of the prompt |
| 8 | `ent_reuse` | higher | share of mid-sentence capitalised words already seen |
| 9 | `sl_mean_h` | lower | distance of mean sentence length from the human median (log) |
| 10 | `sl_cv_h` | lower | distance of sentence-length variation from the human median |
| 11 | `fw_js` | lower | Jensen–Shannon distance of function-word use from human text |
| 12 | `fw_share_h` | lower | distance of the function-word share from the human median |
| 13 | `rep3` | lower | share of word trigrams repeated within the text |
| 14 | `junk` | lower | byte escapes, spaced hyphens and orphan letters per word |
| 15 | `oov` | lower | share of words outside train.txt's vocabulary |
| 16 | SCRIBE v2 | higher | per text, `scribe prepare --mode continuation` then `score` (as the ladder) |
| 17 | SCRIBE v3.1 | higher | the same run |

Check before registration: SCRIBE v2 per text, averaged over the cut dev texts, is close to the ladder's figure (L1 0.628 against 0.610; D0 0.624 against 0.606). Features and SCRIBE were computed before this registration. No signal has been compared with a rater's verdict, and the human-against-writer gap has not been computed.

## Measures

- **Gap:** each signal's mean on human text and on the writer's text, and the AUC: the chance that a human text's value is above a writer text's (raw values). Against all 42 dev files pooled, and against L1.
- **Agreement:** the share of judgments where the signal, read in its registered direction, picks the rater's choice. A signal tie counts half. 95% bootstrap over clusters of round × prompt (2,000 resamples). Reported for writer-against-writer pairs (the main measure), per round, and for human-against-writer pairs.
- **Against SCRIBE:** each signal's agreement minus SCRIBE v2's on the same pairs, with a paired cluster bootstrap.
- **Ceiling:** in pair rounds, two raters judged each pair. Their agreement (both non-tie) is the most a signal could expect.

## Rule

- A signal **tracks the raters** if its writer-against-writer agreement is above 0.5 and its 95% interval excludes 0.5.
- It **beats SCRIBE** if its agreement minus SCRIBE v2's has a 95% interval above 0.
- 17 signals are tested, so about one could pass by chance. Per-round results are read for consistency, not ruled on.
- Nothing is adopted or changed by this report. A signal that passes is a candidate screen for the next rating round, and a candidate writing target (§13.14, "What to optimise").

## Predictions (registered)

**Gap** (AUC human above L1; above 0.5 means human text has the higher value):
- `surp_mean` 0.60–0.75 (human text is more surprising); `surp_sd` 0.60–0.70; `surp_ac1` 0.55–0.65; `surp_cf` about 0.60.
- `burst_cw` 0.65–0.80; `commit_text` 0.70–0.85 (K5_COHERENCE measured 0.46 against 0.15 for the previous sentence); `commit_prompt` 0.65–0.75; `ent_reuse` 0.65–0.80.
- the human-distance signals (`sl_mean_h`, `sl_cv_h`, `fw_js`, `fw_share_h`) at most 0.45 (human text is nearer its own median);
- `rep3` 0.45–0.55; `junk` at most 0.30; `oov` 0.50–0.60;
- SCRIBE v2 about 0.85.

**Agreement, writer against writer:**
- best: `junk` about 0.62 and `commit_text` about 0.60; then `surp_mean`, `ent_reuse`, `fw_js` and `sl_mean_h` about 0.57–0.58;
- `burst_cw`, `commit_prompt`, `surp_cf` and `rep3` about 0.52–0.53 (prompt reuse rewards stuffing, as in §13.34);
- `surp_sd` and `surp_ac1` about 0.50: the surprise profile separates human from writer, but not one writer from another;
- SCRIBE v2 about 0.52, its interval touching 0.5; v3.1 about 0.53;
- at least two text signals track the raters, and at least one beats SCRIBE v2;
- raters against each other about 0.80.

**Human against writer:** SCRIBE v2 about 0.85; `junk` about 0.85; `commit_text` about 0.75.

## Results (2026-11-12; registration 51532b0b pushed first)

Full tables: `bench/writing_signal/gap.md`, `agree.md`, `explore.md` (and `.json`). Run: `python3 -I scripts/surprise_profile.py all` (about 2 minutes, 0.4 GB).

### Outcome

- **One text signal tracks the raters: mean surprise, read as "lower reads better".** 0.588 [0.544, 0.630] over 1,431 writer-against-writer judgments.
- **SCRIBE v2 tracks them too, just:** 0.547 [0.508, 0.584]. SCRIBE v3.1 does not (0.473 [0.431, 0.513]).
- **No signal beats SCRIBE v2 on all rounds.** Mean surprise is +0.041 [−0.021, +0.098] above it.
- **On today's writer (K5 from 337 MB) it does** (exploratory, not ruled on): mean surprise 0.643 [0.596, 0.689], SCRIBE v2 0.566 [0.514, 0.618], difference +0.079 [+0.007, +0.150] (594 judgments: `tw_pair`, `ng_pair`, `coh_pair`, `coh_long`).
- **But it points away from human text.** Human text is far more surprising than any writer's (AUC 0.93 against L1). In the 1,200 human-against-writer judgments, lower surprise picks the rater's choice 0.005 of the time.
- **The human-like signals separate human from writer, but not one writer from another.** Commitment, content-word burstiness, prompt reuse, sentence rhythm and function-word use all have gap AUCs of 0.75–0.85. Their writer-against-writer agreement is 0.47–0.54.
- Raters agree with each other 0.840 [0.804, 0.873]. No signal is close.

### Human against writer (dev; AUC = chance a human text's value is higher)

| Signal | human | all 42 files | L1 (today) | AUC vs all | AUC vs L1 |
|---|---:|---:|---:|---:|---:|
| mean surprise (bits/word) | 8.92 | 6.83 | 7.40 | 0.949 | 0.929 |
| surprise spread | 5.41 | 4.37 | 4.84 | 0.898 | 0.822 |
| surprise autocorrelation | −0.06 | −0.06 | −0.04 | 0.504 | 0.447 |
| content minus function surprise | 6.90 | 5.25 | 6.07 | 0.828 | 0.715 |
| content-word burstiness | 0.143 | 0.063 | 0.067 | 0.789 | 0.771 |
| commitment to earlier sentences | 0.668 | 0.262 | 0.279 | 0.825 | 0.806 |
| commitment to the prompt | 0.537 | 0.221 | 0.201 | 0.789 | 0.807 |
| entity reuse | 0.214 | 0.108 | 0.107 | 0.627 | 0.621 |
| mean sentence length (words) | 19.7 | 15.0 | 12.7 | | |
| sentence-length distance from human | 0.19 | 0.43 | 0.49 | 0.214 | 0.155 |
| function-word share | 0.453 | 0.532 | 0.505 | | |
| function-word JS distance | 0.362 | 0.311 | 0.319 | 0.788 | 0.750 |
| repeated trigrams | 0.009 | 0.004 | 0.004 | 0.575 | 0.573 |
| junk per word | 0.011 | 0.029 | 0.034 | 0.313 | 0.289 |
| words outside train.txt | 0.058 | 0.012 | 0.014 | 0.870 | 0.848 |
| SCRIBE v2 | 0.886 | 0.608 | 0.628 | 0.783 | 0.761 |
| SCRIBE v3.1 | 0.805 | 0.547 | 0.555 | 0.806 | 0.786 |

- Human text is more surprising, with a wider spread, and the gap sits on content words (§13.14's guess: surprise rises at real choice points).
- Human sentences are longer (19.7 words against 12.7) and reuse earlier words more than twice as often (0.67 against 0.28).
- One surprise: a human text's function words are *further* from the pooled human profile than a writer's. K5's function words are average; each human text has its own.

### Agreement with the raters, writer against writer (registered direction)

| Signal | agreement [95%] | minus SCRIBE v2 [95%] | human vs writer |
|---|---|---|---|
| **mean surprise (lower)** | **0.588 [0.544, 0.630]** | +0.041 [−0.021, +0.098] | 0.005 |
| sentence-length variation (near human) | 0.535 [0.494, 0.578] | −0.013 [−0.069, +0.043] | 0.737 |
| function-word JS (lower) | 0.530 [0.484, 0.574] | −0.018 [−0.079, +0.042] | 0.150 |
| junk (lower) | 0.525 [0.482, 0.565] | −0.025 [−0.079, +0.034] | 0.682 |
| content minus function surprise | 0.519 [0.474, 0.563] | −0.029 [−0.087, +0.030] | 0.990 |
| surprise autocorrelation | 0.515 [0.473, 0.556] | −0.033 [−0.088, +0.023] | 0.672 |
| sentence length (near human) | 0.515 [0.474, 0.553] | −0.033 [−0.088, +0.023] | 0.759 |
| entity reuse | 0.496 [0.464, 0.532] | −0.051 [−0.103, +0.001] | 0.763 |
| commitment to earlier sentences | 0.491 [0.452, 0.529] | −0.057 [−0.106, −0.007] | 0.880 |
| repeated trigrams (lower) | 0.484 [0.466, 0.501] | −0.064 [−0.106, −0.018] | 0.349 |
| commitment to the prompt | 0.477 [0.436, 0.517] | −0.070 [−0.121, −0.016] | 0.822 |
| function-word share (near human) | 0.476 [0.431, 0.525] | −0.072 [−0.130, −0.012] | 0.849 |
| words outside train.txt (lower) | 0.476 [0.463, 0.489] | −0.071 [−0.110, −0.030] | 0.050 |
| content-word burstiness | 0.474 [0.433, 0.515] | −0.073 [−0.121, −0.023] | 0.883 |
| surprise spread | 0.455 [0.408, 0.501] | −0.092 [−0.154, −0.030] | 0.998 |
| **SCRIBE v2** | **0.547 [0.508, 0.584]** | — | 0.851 |
| SCRIBE v3.1 | 0.473 [0.431, 0.513] | −0.074 [−0.110, −0.040] | 0.879 |

### Why the pooled numbers are small: the rounds disagree

Per round (`agree.md`), the signals flip sign with what the arms changed:
- **Variants of one writer** (sampling, gate, topic and link choosers: `llm_rating`, `llm_rating2`, `tw_pair`, `ng_pair`, `coh_pair`, `coh_long`; 991 judgments): lower surprise agrees **0.738 [0.700, 0.774]**, in every one of the six rounds (0.58–1.00). SCRIBE v2: 0.547. Surprise spread agrees 0.296 (reversed).
- **Planner and fill rounds** (`p1t_pair`, `planner_pair`, `q1t_pair`, `q3_pair`, `q3_long`, `fill_pair`; 218): reversed. Lower surprise 0.367; spread 0.725, commitment to earlier sentences 0.702, SCRIBE v2 0.706. The planned text names people, places and dates. That is more surprising, more committed, and the raters prefer it.
- **`gq_pair`** (K5 from 7 MB against 337 MB; 222): lower surprise 0.14. The 337 MB text uses words the 7 MB model rarely saw. Without `gq_pair`, lower surprise agrees 0.671 [0.629, 0.710], SCRIBE v2 0.576.
- **A combination does not travel.** A logistic fit on all 17 signals' differences, tested on the round it did not see, agrees 0.467 [0.426, 0.507] (text signals alone 0.420). The weights learned on some rounds point the wrong way on others.

### What it means

- **Between variants of one writer, the raters reward fluency, and low surprise under a fixed small model measures it.** This is the best rater proxy found for screening sampler and chooser settings on today's writer: 0.643 against SCRIBE v2's 0.566.
- **It is not a writing target.** Pushed on, it moves text away from human text (§13.14 warned of this: what predicts best does not write best). Human text is more surprising, and lower surprise picks against the human every time.
- **The raters reward content when the content is new** (the planner's facts, the 337 MB counts). Surprise then rises, and the human-like signals (commitment, spread) agree with the raters instead.
- **The human-like statistics are not what separates today's writers.** Commitment, burstiness and sentence rhythm show the human gap clearly (AUC about 0.8), but the writers the raters compare barely differ on them. Closing that gap needs a writer that commits (a plan for the paragraph, §13.51–13.53), not a chooser.
- **SCRIBE v3.1** does no better than chance on today's writer (0.468 [0.412, 0.524]). SCRIBE v2 is the better of the two for rating screens.

### Predictions

- Gap: `burst_cw` 0.65–0.80 **hit** (0.771); `commit_text` 0.70–0.85 **hit** (0.806); `junk` at most 0.30 **hit** (0.289); human-distance signals at most 0.45 **hit** for 3 of 4 (`fw_js` **miss**, 0.750). Direction right but larger than predicted, so **miss**: `surp_mean` (0.929 against 0.60–0.75), `surp_sd` (0.822), `surp_cf` (0.715), `commit_prompt` (0.807), `oov` (0.848). **Miss**: `surp_ac1` (0.447 against 0.55–0.65), `ent_reuse` (0.621), `rep3` (0.573), SCRIBE v2 (0.761 against about 0.85).
- Agreement: `surp_mean` about 0.57–0.58 **hit** (0.588). `junk` about 0.62 **miss** (0.525); `commit_text` about 0.60 **miss** (0.491); `ent_reuse`, `fw_js`, `sl_mean_h` about 0.57 **miss** (0.50–0.53). `burst_cw`, `commit_prompt`, `surp_cf`, `rep3` about 0.52 **hit** in substance (0.47–0.52). `surp_sd` and `surp_ac1` about 0.50 **hit** (0.455, 0.515). SCRIBE v2 about 0.52, interval touching 0.5: **miss** (0.547, interval just above 0.5). SCRIBE v3.1 about 0.53 **miss** (0.473).
- At least two signals track the raters: **miss** (one). At least one beats SCRIBE v2: **miss** (on all rounds; exploratory yes on today's writer).
- Raters against each other about 0.80: **hit** (0.840).
- Human against writer: SCRIBE v2 about 0.85 **hit** (0.851); `junk` about 0.85 **miss** (0.682); `commit_text` about 0.75 **miss** (0.880, higher).

### Limits

- The surprise model is a word trigram over 7 MB of K5's own text. The writer counts from that text and 330 MB more, so the model is not neutral. A model over independent text was not tried here.
- The judgments are not independent: two raters per pair, many pairs per prompt. The intervals resample round × prompt clusters, which covers most of it.
- The per-round splits and the 337 MB subset were chosen after seeing the per-round table. They are exploratory, not ruled on.
- Nothing is adopted. A next rating round could register mean surprise as a screen beside SCRIBE v2 and test it on fresh pairs.
