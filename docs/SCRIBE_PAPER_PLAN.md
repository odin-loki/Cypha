# SCRIBE paper plan: v2, its meta-evaluation and the comparisons

**Odin Loch** · 2026-10-28

This is the plan for a paper on SCRIBE, the generated-text metric of [`PLAN.md`](PLAN.md) §6, for upload to Zenodo. It covers a v2 of the metric, a much wider meta-evaluation than v0 and v1 had, comparisons with the standard metrics (neural and statistical), and profiling. Predictions are registered in §8 before any v2 run. Results go in [`reports/SCRIBE_V2.md`](reports/SCRIBE_V2.md) and the paper in [`../paper/scribe/`](../paper/scribe/).

---

## 1. Where SCRIBE stands

- **v0** passed 1 of 6 adjacent pairs of its control ladder ([report](reports/SCRIBE_V0.md)).
- **v1** passes 3 of 6 on a fresh ladder ([report](reports/SCRIBE_V1.md)). Reproduced exactly in a new container on 2026-10-28 (same means, same accuracies to the printed digit).
- **The three v1 failures.**
  1. "trie > sentences shuffled": the order-4 trie writes word salad, so this rung order cannot hold on it. A ladder problem, not a metric problem.
  2. "sentences shuffled > words shuffled" (0.829): the S gate and a bag-of-words C pull in opposite directions.
  3. "words shuffled > random lexicon words" (0.722): the S cap throws away the C information that separates them.
- **Also open:** human samples have a wide spread (one weak sub-score drags a sample down), E is not calibrated across regions, and R, B, V2 (human agreement) and V3 (attacks) were never run.

## 2. Questions the paper answers

| # | Question | Test |
|---|---|---|
| Q1 | Can a metric built from counts alone order a control ladder of degraded text? | Open-text ladder, fresh region, PLAN 6.4 pass rule |
| Q2 | Does it generalise to degradations it was not designed on? | Held-out corruption families and dose–response curves |
| Q3 | Does it rank real language models in the known order? | Pythia training checkpoints, Pythia sizes, GPT-2 sizes × decoding, n-gram orders, CyphaLM |
| Q4 | Does it agree with people? | MAUVE human evaluation (pairwise, WebText), HANNA (stories, 6 criteria), OpenMEVA MANS (stories, overall) |
| Q5 | Can it be fooled? | PLAN 6.4 V3 attacks |
| Q6 | Is it stable? | Length, seed, genre (Wikipedia, web, books, stories), reference size, pack size |
| Q7 | What does each part add, and what does it cost? | Ablations; time and RAM per component |
| Q8 | How does it compare with the standard metrics on every test above? | Same texts scored by every baseline in §6 |

Q1–Q8 are asked of SCRIBE v2 and of every baseline on the same texts.

## 3. Data and splits

All splits are fixed here, before v2 is designed. MiB = 2²⁰ bytes. enwik8 is 10⁸ bytes (95.37 MiB); enwik9 is the first 10⁹ bytes of the same 2006 Wikipedia dump.

### 3.1 Wikipedia (enwik9)

| Range | Use |
|---|---|
| [0, 7) MiB | CyphaLM trie training text. SCRIBE never reads it |
| [7, 8) MiB | Trie holdout: prompts for trie samples (as in v1) |
| [8, 16) MiB | Training text for the n-gram generators (§5.3). Disjoint from SCRIBE's pack |
| [16, 48) MiB | **SCRIBE pack**: lexicon, word trigram, IDF, PPMI-SVD vectors, tag n-gram text (as in v1) |
| [48, 56) MiB | **Human reference**: the empirical CDFs, the length pmf |
| [56, 64) MiB | **Dev ladder A** (v0 and v1's development ladder, seed 20261027) |
| [64, 72) MiB | **Dev ladder B** (v1's fresh ladder; read by v1, so it is development data now) |
| [72, 95.37) MiB | **v2 fresh ladder** (never read by any SCRIBE). Also the prompts for the fresh model ladders |
| [128, 384) MiB | Pack-size profile only (32, 64, 128, 256 MiB packs) |
| [900, 953) MiB | Spare fresh region, kept for a v3 if v2 fails (§9) |

### 3.2 Other genres

| Genre | Pack | Reference | Test |
|---|---|---|---|
| Web (GPT-2 output dataset, OpenAI) | `webtext.train` subset (about 40 MB) | `webtext.valid` (5,000 docs) | `webtext.test` (5,000) plus the 8 GPT-2 test files (small, medium, large, xl × pure sampling, top-k 40) |
| Books (Project Gutenberg) | about 80 novels | 20 novels | 20 novels (10 dev, 10 fresh), split by book |
| Stories (WritingPrompts) | WritingPrompts train | WritingPrompts valid | HANNA and OpenMEVA-WP stories, whose prompts come from the WritingPrompts test split |

### 3.3 Human judgments

| Set | What people rated | Size |
|---|---|---|
| MAUVE human evaluation (Pillutla et al. 2021) | Pairwise: which WebText continuation is more interesting (q1), makes more sense (q2), is more likely human (q3). 9 systems: human and GPT-2 {small, medium, large, xl} × {nucleus, pure sampling} | 3,240 pairs, texts included |
| HANNA (Chhun et al. 2022, 2024) | 1,056 WritingPrompts stories from 10 systems and humans; 3 raters × 6 criteria (relevance, coherence, empathy, surprise, engagement, complexity). Ships 72 automatic metrics and 4 LLM judges | 19,008 ratings |
| OpenMEVA (Guan et al. 2021) | MANS: overall quality of ROC and WritingPrompts stories; AUTOS: perturbation tests | about 2,000 rated stories |

No rating set was made for Cypha. PLAN 6.4 V2 asked for 300 samples rated by Odin; that is replaced here by published ratings and stays open.

### 3.4 Independence (PLAN V5)

- SCRIBE's pack and reference never overlap a test text, a prompt or a generator's training text.
- The neural baselines are external yardsticks only. None of them is part of SCRIBE.
- Neural models are used to *write* model-ladder text (§5.3); that is data, not the metric.

## 4. SCRIBE v2: what changes

The design is chosen on dev ladders A and B only, then frozen, then run once on the fresh region.

| v1 problem | v2 candidates (decided on dev, by measurement) |
|---|---|
| Gates throw away information (failures 2 and 3) | Replace caps by a single **human-typicality deviance**: each raw statistic becomes a signed z-score through the human ECDF (Gaussian copula), with a parametric tail past the ECDF's range; one-sided statistics are penalised only on the bad side; the deviance is a weighted sum of squared bad-side z-scores, so several moderate deviations add up and one extreme deviation dominates. The score is a monotone map of the deviance, 1 at the human median |
| Wide human spread | The deviance only counts the tails, so a human sample near the median on every statistic scores near 1 |
| Bag-of-words coherence cannot see word order | Word-order statistics: S2 tag plausibility (E1 tagger plus a tag 4-gram trained on UD English EWT), and local word-order fluency |
| Coherence statistics | C1 adjacent and all-pairs sentence cosine, C0 word cohesion, C3 drift; candidate C2 entity grid |
| Integrity | I1 verbatim copy against the pack (suffix array) and I4 prompt echo for prompted tasks |
| E not calibrated | E null drawn per region with between-region spread; E reported only, unless it calibrates on held-out regions |
| Relevance (R) | R1 matched-pair probability from SIF cosine, R3 swap rank, R4 key-term coverage, R5 informativeness. Validated on HANNA relevance and a Q/A ladder from Dolly |
| Belief (B) | Out of scope for this paper unless time allows; stated as a limit |

v0 and v1 stay runnable (`--profile v0|v1`) so every table can show all three versions.

## 5. Tests

### 5.1 Control ladders (PLAN 6.4 V1)

Rungs, best to worst: held-out human > sentences shuffled across documents > words shuffled within sentences > random lexicon words > random letters > one phrase repeated. Pass rule unchanged: pairwise accuracy ≥ 0.95 over all (upper, lower) pairs, ties one half, and disjoint 95% bootstrap intervals of the rung means. The PLAN's "model" rung is tested separately in §5.3, because the rung order "model > shuffled sentences" is only meaningful for a model that writes sentences.

Run on: Wikipedia dev A, dev B, fresh; WebText; books (dev, fresh). Standard (400–600 B) and short (60–260 B) samples.

### 5.2 Held-out corruptions and dose–response (not used in the design)

Each family applied at strengths 0, 10, 25, 50, 75, 100% to fresh human text:
word substitution with random lexicon words; adjacent word swaps; sentence order reversal within a sample; sentence repetition; noun substitution across documents; character noise (typos); truncation of sentences; insertion of off-topic sentences; plus the OpenMEVA AUTOS perturbations.
Score: Spearman of score with strength, and the share of strength steps that lower the mean score.

### 5.3 Model ladders (known order)

| Ladder | Known order | Texts |
|---|---|---|
| Pythia-410M training checkpoints (steps 1k, 4k, 16k, 64k, 143k) | later > earlier | 250 dev and 250 fresh prompts each, nucleus 0.95 |
| Pythia sizes (70M, 160M, 410M, 1B, 1.4B) | larger > smaller | same prompts |
| GPT-2 decoding (GPT-2 large) | reported, partly known: greedy and beam degenerate; nucleus ≈ human; pure sampling at T = 1.3 incoherent | greedy, T = 0.7, top-k 40, nucleus 0.95, pure T = 1.0, T = 1.3 |
| GPT-2 output dataset | larger > smaller within each decoding | 8 × 5,000 web samples |
| KN word n-grams (orders 1–5) | higher order > lower | trained on [8, 16) MiB |
| CyphaLM (order-4 trie; trie + InfiniGram) at T = 0.6, 0.8, 1.0 | placed by measurement | as v1 |

System-level: Spearman and Kendall of the mean score against the known order, with bootstrap intervals. Sample-level: pairwise accuracy between adjacent systems.

### 5.4 Human agreement (PLAN 6.4 V2)

- MAUVE human evaluation: per-pair agreement of each metric with the people's choice (q1, q2, q3), and system-level Spearman against Bradley–Terry scores fitted to the people's choices.
- HANNA: story-level Kendall τ with the mean rating of each criterion; system-level Kendall; Williams-style bootstrap tests for differences between metrics.
- OpenMEVA MANS: story-level Pearson and Kendall with the overall score, as in the OpenMEVA paper.

### 5.5 Attacks (PLAN 6.4 V3)

Copied human text from the pack; the prompt echoed; keyword stuffing; a generic "it depends" answer; random dates; a high-frequency function-word loop; lists of real sentences from one document in order (a "fluent but unprompted" attack); text from the same genre but another topic.

### 5.6 Stability (PLAN 6.4 V4)

Correlation with length on human text; seed variance (five seeds); genre transfer (each genre's ladder with its own reference, and with the Wikipedia reference); reference size (100 to 1,900 samples); pack size (8 to 256 MiB).

### 5.7 Profiling

Time per sample (single thread), pack build time, peak RAM, per-component share of time; the same for each baseline on the same machine (4 cores, no GPU).

## 6. Baselines (same texts)

| Family | Metrics |
|---|---|
| Cypha's own | Surface S = H·D·L; SCRIBE v0, v1 |
| Statistical, per sample | KN word-trigram perplexity (the pack's), distinct-1..4, rep-n and seq-rep-4 (Welleck et al. 2020), gzip compression ratio, mean word length, type–token ratio |
| Statistical, per set | Zipf coefficient (Holtzman et al. 2020), self-BLEU (Zhu et al. 2018), MAUVE on SIF vectors |
| Neural, per sample | GPT-2 perplexity (small and large); grammatical acceptability from a RoBERTa CoLA classifier; sentence-embedding coherence (all-MiniLM-L6-v2, adjacent cosine); the OpenAI GPT-2 output detector's P(human); an LLM judge (Qwen2.5-1.5B-Instruct, probability-weighted 1–10 score, G-Eval style) |
| Neural, per set | MAUVE with GPT-2-large features (Pillutla et al. 2021) |
| Shipped with HANNA | 72 automatic metrics (BLEU, ROUGE, BERTScore, BARTScore, MoverScore, …) and 4 LLM judges |

## 7. Workstreams and file ownership

| Stream | Owner | Files |
|---|---|---|
| SCRIBE library and tool | main session | `native/include/cypha/eval/scribe.hpp`, `native/tools/scribe.cpp` (replaces `scribe_ladder.cpp`), CMake block |
| Datasets | agent | `scripts/scribe_eval/data.py` |
| Generations | agent | `scripts/scribe_eval/generate.py` |
| Baselines and LLM judge | agent | `scripts/scribe_eval/baselines.py`, `scripts/scribe_eval/judge.py` |
| Meta-evaluation statistics | agent | `scripts/scribe_eval/meta.py` |
| Report and paper | main session | `docs/reports/SCRIBE_V2.md`, `paper/scribe/` |

Every text that is scored is one JSONL line: `{"id", "set", "system", "text", "prompt"?, "ref"?, "meta"?}`. Every metric writes `{"id", "<metric>": value, ...}` lines for the same ids.

## 8. Registered predictions (written before any v2 run)

Scored hit or miss in the report.

| # | Prediction |
|---|---|
| P1 | SCRIBE v2 passes all 5 adjacent pairs of the open-text control ladder on the fresh Wikipedia region (accuracy ≥ 0.95 and disjoint intervals) |
| P2 | Surface S passes at most 2 of those 5 pairs |
| P3 | GPT-2 perplexity fails "random letters > one phrase repeated" (a repeated phrase is very probable) and ranks the repeated phrase above held-out human text |
| P4 | Set-level MAUVE (GPT-2-large features) orders the 6 rungs correctly as sets |
| P5 | SCRIBE v2 passes at least 4 of 5 pairs on WebText and on books, each with its own reference |
| P6 | Dose–response: SCRIBE v2's mean falls monotonically with strength (Spearman ≤ −0.9) for at least 7 of the 9 held-out corruption families |
| P7 | Pythia-410M checkpoints: SCRIBE v2's system means follow training steps with at most one adjacent swap (Spearman ≥ 0.9 over 5 checkpoints) |
| P8 | Pythia sizes: Spearman ≥ 0.8 between SCRIBE v2 and size (5 sizes) |
| P9 | GPT-2 output dataset: SCRIBE v2 orders the four sizes correctly within at least one decoding family; MAUVE within both |
| P10 | GPT-2 greedy decoding scores lowest of the decoding ladder under SCRIBE v2, and highest under GPT-2 perplexity |
| P11 | MAUVE human evaluation: system-level Spearman between SCRIBE v2 and the q3 (human-like) Bradley–Terry scores ≥ 0.6; per-pair agreement with q3 above 0.55 |
| P12 | HANNA: story-level Kendall τ between SCRIBE v2 (story reference) and mean human coherence ≥ 0.15, and system-level Kendall ≥ 0.6 |
| P13 | OpenMEVA MANS: story-level Pearson with the overall score between 0.1 and 0.4 (the range of published reference-free metrics) |
| P14 | V4: \|Pearson(SCRIBE v2, bytes)\| ≤ 0.1 on held-out human text in each length bucket and pooled |
| P15 | Scoring WebText human text against the Wikipedia reference drops its human parity below 0.8: SCRIBE is genre-relative |
| P16 | Attacks: copied pack text scores at human parity ≥ 0.9 without I1 and below 0.5 with I1 in generation mode; keyword stuffing and the function-word loop score below 0.2 parity |
| P17 | Cost: after the pack is built, SCRIBE v2 scores at least 1,000 standard samples per second on one core, with peak RAM under 500 MB for the 32 MiB pack; at least 100 times faster than computing GPT-2-large MAUVE features on the same CPU |
| P18 | Ablations on the fresh ladder: without C, "human > sentences shuffled" falls below 0.95; without S3, "sentences shuffled > words shuffled" falls below 0.95 |
| P19 | The LLM judge (1.5B) fails at least one adjacent pair of the control ladder |
| P20 | On HANNA, SCRIBE v2 correlates with coherence more than BLEU and ROUGE-L (which need a reference) at story level |

Amended 2026-10-28, before any v2 run: the checkpoint ladder uses Pythia-410M (5 checkpoints of 0.9 GB) instead of Pythia-1.4B (4 of 2.8 GB), to fit the container's disk and CPU; P7 was rewritten for 5 checkpoints.

## 9. Kill criteria and reading rules

- The fresh Wikipedia region is read once for v2. Every read is reported.
- If v2 fails 2 or more control pairs on the fresh region, that is the v2 result. A v3 may be designed on dev ladders only and is then read once on the spare region [900, 953) MiB.
- No metric weight or threshold is chosen on human-judgment data, model ladders, or held-out corruptions. Those are test data for every metric.
- Baselines run with their published defaults; nothing is tuned for them either.

## 10. Paper

- Markdown source in `paper/scribe/SCRIBE_paper.md`, tables only (no images), built to HTML and PDF for Zenodo, with `zenodo.json` metadata, an abstract file, and the result tables as CSV.
- Licence: paper text CC BY 4.0; code under the repository's AGPL-3.0 (and the commercial licence).
- Every number in the paper comes from a committed tool and is listed in the report with its command.
