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

## Results

(to come, after this registration is committed and pushed)
