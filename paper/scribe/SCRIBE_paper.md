---
title: "SCRIBE: Scoring Generated Text Against Human Text With Counts Alone"
author: "Odin Loch"
date: "28 October 2026"
---

# SCRIBE: Scoring Generated Text Against Human Text With Counts Alone

**Odin Loch**

<p class="meta">Version 1.0 · 28 October 2026<br>Code and data recipes: the Cypha repository, github.com/odin-loki/Cypha (<code>native/include/cypha/eval/scribe.hpp</code>, <code>native/tools/scribe.cpp</code>, <code>scripts/scribe_eval/</code>)</p>

## Abstract

<!-- ABSTRACT -->

---

## 1. Introduction

Most measures of a language model say how well it predicts human text: perplexity, or bits per character. They do not say whether the text the model writes is any good. A model can compress text well and still write word salad. The usual ways to rate generated text use a second large neural network: perplexity under another language model, similarity in an embedding space, MAUVE, a trained classifier, or a language model asked for a rating. These need a downloaded model and, to be fast, a GPU. They also grade a model with machinery of the same kind as the model under test, with its own blind spots and biases.

This paper asks how far counts alone can go. A count-based metric is cheap and transparent, and it can be rebuilt for a new genre from a few megabytes of human text. It is also easy to get wrong. A simple surface score `S = H·D·L` (the share of words in a vocabulary, times a diversity term, times a loop penalty) shows how. On fresh Wikipedia text it rates real passages with their words shuffled (0.925) and random dictionary words (0.895) about as high as the real passages (0.915). A metric that cannot tell word salad from prose cannot judge a language model.

This paper describes SCRIBE, the metric built to replace it, and the tests it had to pass before it was allowed to judge anything. SCRIBE computes about twenty statistics of a text from counts: n-gram probabilities, word classes, word co-occurrence vectors, repetition and overlap. It converts each statistic into a signed distance from human text of the same genre and length, read from a sample of human reference texts, and combines the distances into one deviance with a fixed rule. The score is high when a text is typical of human text on every statistic, and falls with the evidence that it is not.

The design rule of this work is that a metric must pass control ladders before it is used. A control ladder is a sequence of texts whose order nobody disputes: real text, then the same text with its sentences shuffled across documents, then with its words shuffled inside sentences, then random dictionary words, then random letters, then one phrase repeated. A metric passes a rung pair when it orders at least 95% of all (upper, lower) pairs correctly and the confidence intervals of the rung means do not overlap. Two earlier versions of SCRIBE failed this test: on the fresh ladder of this paper, v1 passes 2 of 5 pairs and the surface score 1 of 5. This paper presents v2, which was designed on two development ladders and then frozen, and which passes all five pairs on fresh text it had never seen.

Passing a ladder is necessary, not sufficient. We therefore test SCRIBE v2, and standard metrics next to it, on: held-out degradations it was not designed on; language models whose quality order is known (n-gram orders, and GPT-2 sizes under two decoding methods); three published sets of human judgments (the MAUVE human evaluation, HANNA and OpenMEVA); question answering; adversarial attacks; stability with length, seed, genre, reference size and training-text size; ablations of every part; and cost. Twenty predictions were registered before the first v2 run. Each is scored hit or miss, and four could not be tested.

**Contributions.**

1. A complete, neural-network-free metric for generated text, SCRIBE v2, defined by human-relative z-scores and a fixed combination rule, with a header-only C++ implementation that scores about a thousand texts per second on one CPU core.
2. A context-gain statistic: the log-probability of each word given its context minus its log-probability from frequency alone. It separates word salad from rare-topic human text far better than raw n-gram probability (Section 4).
3. A design history that keeps the failures: v0 and v1, every combination that did not pass, and why the final structure, with categorical form failures and graded coherence, was needed.
4. A broad meta-evaluation of SCRIBE and of standard metrics on the same texts, with pre-registered predictions, bootstrap intervals, and the cases where SCRIBE fails.

## 2. Related work

**Reference-based metrics.** BLEU (Papineni et al., 2002), ROUGE (Lin, 2004), METEOR (Banerjee and Lavie, 2005) and chrF (Popović, 2015) count n-gram overlap with a reference. BERTScore (Zhang et al., 2020), MoverScore (Zhao et al., 2019), BLEURT (Sellam et al., 2020) and BARTScore (Yuan et al., 2021) compare texts through neural representations. All of them need a reference written for the same input, and their validity for open-ended generation, where many different texts are good, is weak (Novikova et al., 2017; Reiter, 2018; Gehrmann et al., 2023).

**Reference-free and distributional metrics.** Perplexity under a large language model rewards predictable text and is known to prefer degenerate, repetitive output (Holtzman et al., 2020). Diversity statistics such as distinct-n (Li et al., 2016), self-BLEU (Zhu et al., 2018), repetition rates (Welleck et al., 2020) and the Zipf coefficient (Holtzman et al., 2020) each capture one property. MAUVE (Pillutla et al., 2021) compares the distribution of model text with human text in the embedding space of a large language model and correlates well with human judgments of model families. Meister and Cotterell (2021) compare model text and human text on statistical tendencies such as type–token ratios and length distributions; SCRIBE shares their premise that human text sets the norm, but scores individual texts. HUSE (Hashimoto et al., 2019) combines human judgments with model probabilities to estimate how distinguishable model text is from human text. GRUEN (Zhu and Bhat, 2020) and UNION (Guan and Huang, 2020) learn quality from perturbations with neural encoders.

**Language models as judges.** Prompting a large language model for a rating (G-Eval, Liu et al., 2023; Zheng et al., 2023) gives strong correlations with people, at a large cost and with the model's own biases; HANNA (Chhun et al., 2022, 2024) measures this for story generation.

**Coherence.** The entity grid (Barzilay and Lapata, 2008) models how entities move between sentences; sentence-order and cohesion models (Lapata, 2003) score local coherence. SCRIBE's lexical cohesion statistic is a count-only form of the entity-grid idea, and its sentence vectors follow the SIF construction of Arora et al. (2017) on PPMI-SVD word vectors (Levy et al., 2015).

**Meta-evaluation.** OpenMEVA (Guan et al., 2021) and the perturbation checklists of Sai et al. (2021), following CheckList (Ribeiro et al., 2020), test metrics with controlled perturbations; HANNA and the MAUVE human evaluation supply human ratings. Control ladders are a simple, strict version of perturbation testing: every rung must be ordered, with a pass threshold set in advance.

**Detection.** Detectors of machine text, such as GLTR (Gehrmann et al., 2019), the GPT-2 output detector (Solaiman et al., 2019) and DetectGPT (Mitchell et al., 2023), exploit the fact that sampled text is more predictable than human text. Ippolito et al. (2020) show that the decoding settings that fool people most are the easiest for detectors. This tension appears in SCRIBE's results (Section 6.3).

## 3. SCRIBE v2

### 3.1 Principles

1. **Human text is the norm.** Every statistic is read as a distance from human text of the same genre and length bucket, never as an absolute value. A text that matches human text on every statistic scores near the top.
2. **Independence.** Everything SCRIBE learns comes from human text disjoint from the texts it scores and from the training text of any model it scores. It never uses the model under test.
3. **Counts only.** No neural network is trained or run. The statistics come from n-gram counts, word classes, co-occurrence counts and their factorisation.
4. **No use before passing.** A version is used to judge models only after it passes its control ladder on text it was not designed on.

### 3.2 The pack and the reference

SCRIBE needs two disjoint sets of human text of the target genre:

- **The pack** trains the statistics: a lexicon (words seen at least twice), an interpolated Kneser–Ney word trigram (Kneser and Ney, 1995; Chen and Goodman, 1999), a skeleton trigram over word classes, sentence-level IDF, unigram frequencies, the set of word 8-grams, and 300-dimensional PPMI-SVD word vectors with SIF sentence embeddings (weight `a/(a + p(w))`, `a = 10⁻³`, first principal component removed).
- **The reference** supplies the human distribution of every statistic. It is cut into samples by the same rule used for the texts being scored: consecutive whole sentences, 400–600 bytes with at least three sentences (standard bucket), or 60–260 bytes (short bucket). For Wikipedia, the pack is 32 MiB of enwik8 and the reference has 1,909 standard and 1,729 short samples.

### 3.3 Statistics

Table 1 lists the statistics. "Side" says which direction is bad: `+` means lower than human is bad, `−` means higher is bad, `±` means either.

**Table 1.** SCRIBE v2 statistics (all from counts).

| Axis | Statistic | Definition | Side |
|---|---|---|---|
| L | s1 lexical validity | share of word tokens that are in the lexicon or numeric | + |
| S | s2g skeleton gain | mean over tokens of `log p(c_t | c_{t−2}, c_{t−1}) − log p(c_t)` for the class sequence: the 150 most frequent words keep their identity, other words map to one of 9 suffix classes, numbers to one class; trigram and unigram trained on the pack | + |
| S | s3g context gain | mean over in-lexicon words of `log p_KN(w_t | w_{t−2}, w_{t−1}) − log p(w_t)`: how much the two preceding words predict each word, beyond its frequency | + |
| S | s4 well-formedness | pass rate of six checks: valid UTF-8, sentence-initial capitals, terminal punctuation, balanced brackets, paired quotes, no cut-off last word | + |
| C | c0 word cohesion | mean cosine over all pairs of distinct content words (sentence IDF ≥ 3, known to the vectors) | + |
| C | c1 sentence cohesion | mean cosine over all pairs of SIF sentence vectors (two halves of the text when it has one sentence) | + |
| C | c1a adjacent cohesion | mean cosine of adjacent sentence vectors | + |
| C | c2 lexical cohesion | mean over all pairs of sentences of the IDF mass of their shared content words, over the smaller sentence's IDF mass | + |
| C | c3 topic drift | Jensen–Shannon divergence between the content-word distributions of the two halves | − |
| I | loop8 | share of 8-byte windows seen earlier in the text | − |
| I | rep | share of sentences that repeat an earlier sentence | − |
| I | d1–d4 | distinct-1 to distinct-4 over word n-grams | ± |
| I | i1 verbatim copy | share of tokens inside a word 8-gram that occurs in the pack | ± |
| R | r1 prompt cosine | cosine of the SIF vectors of prompt and text | + |
| R | r4 key-term coverage | IDF-weighted share of the prompt's content words that the text uses | + |
| R | r5 informativeness | share of the text's content tokens not in the prompt | ± |
| R | i4 prompt echo | share of the text's word 4-grams that occur in the prompt | − |

Also computed and reported, but not scored: raw word-trigram and skeleton log-probabilities (s3, s2), the worst sentence's log-probability, sentence lengths, and mean word rarity. Coherence statistics are one-sided in v2: more cohesion and less drift are never penalised, because cohesion that comes from repetition is caught by the integrity axis.

### 3.4 From a statistic to evidence

For each statistic `k` and length bucket, let `F` be the mid-rank empirical CDF of the reference values. Inside the 2nd to 98th percentile the z-score is `z = Φ⁻¹(F(x))`. Beyond it the tail is extended linearly, `z = −2.054 − (q₀.₀₂ − x)/σ_lo` below and `z = 2.054 + (x − q₀.₉₈)/σ_hi` above, with `σ_lo = (q₀.₁₆ − q₀.₀₂)/1.059` and `σ_hi = (q₀.₉₈ − q₀.₈₄)/1.059`, floored at a small per-statistic minimum (0.15 for s4, so that one failed check of six counts as moderate evidence). The linear tail keeps the distance of values far outside the human range, where an ECDF would saturate.

The bad-side magnitude is `u = max(0, −z)` for side `+`, `max(0, z)` for side `−`, and `|z|` for `±`. The evidence is `e = max(0, u − 1)`: a dead zone of one standard unit forgives the ordinary spread of human text. A statistic that at least 95% of reference samples have, and the text lacks (for example, no word is known to the vectors), counts as `u = 4`.

### 3.5 Combination

The evidence of each axis is summed: `L`, `S`, `C`, `I` and `R`. The deviance is

```
D = 30·sat(L − θ_L) + 30·sat(S − θ_S) + 3·max(0, I − θ_I) + C + w_R·R
sat(x) = 1 − exp(−max(0, x)/4)
SCRIBE = exp(−D/5)
```

where `θ_L`, `θ_S` and `θ_I` are the 99th percentiles of `L`, `S` and `I` over the reference samples of the bucket, and `w_R` is 0 in the open-continuation profile and 8 in the prompted profile (whose relevance dead zone is 0.5 instead of 1).

The structure encodes three statements, each tested in Section 4:

- **Form failures are categorical.** A text is either made of words (L) or not, and either in an order people write (S) or not. Human text almost never fails either way, so each counts only beyond what 99% of human samples show, and each saturates: two texts that are both word salad are not told apart by how badly they fail syntax.
- **Degeneration is the worst failure.** Repetition (I) counts beyond the human 99th percentile, with weight 3, and does not saturate.
- **Coherence is graded.** Human text varies continuously in cohesion and less cohesive is worse at every level, so coherence counts from the dead zone on, with no threshold. Inside the word-salad class, coherence orders the texts.

### 3.6 Profiles, long texts and short texts

The **open-continuation** profile (`v2`) scores a text on its own. The **prompted** profile (`v2p`) adds the relevance axis, read against human prompt–continuation pairs of the genre. Texts longer than the reference samples are cut into consecutive windows of whole sentences (closing a window before 600 bytes once it holds 300) and score the mean over windows. For corpora whose units are short whole texts, such as five-sentence ROC stories, whole reference documents serve as samples.

### 3.7 Implementation

SCRIBE is a header-only C++20 library using only the standard library, with a command-line tool for document cleaning, prompt extraction, sampling, preparation of model output, control ladders and scoring. Building the Wikipedia pack (32 MiB) takes about 10 s; the PPMI-SVD vectors take about 50 s once. Scoring needs about 415 MB of memory and runs at roughly a thousand texts per second on one core (Section 6.8). The tool reproduces the earlier v1 ladder to the printed digit.

## 4. How the design was found

SCRIBE was built in three versions, each tested on a control ladder before anything else. The rule throughout: a design choice is made on development ladders only; a frozen version is then read once on fresh text.

### 4.1 v0 and v1: percentiles, geometric means and gates

The PLAN that defined SCRIBE converted every statistic to a percentile among human samples (one-sided `min(1, 2F)` or two-sided `1 − |2F − 1|`), took the geometric mean of the sub-scores of each axis and of the axes, and capped a sample at its syntax score when that fell under 0.2 ("word salad cannot be rescued by other axes"). v0 implemented this with a lexicon, a word trigram, bag-of-words cohesion and diversity statistics. On its ladder, which also had a rung of samples from a byte-level count language model, it passed 1 of 6 pairs. v1 added PPMI-SVD sentence vectors, all-pairs cohesion, an integrity gate and length parity, and passed 3 of 6. On the fresh control ladder of this paper it passes 2 of 5 (Section 6.1). Its failures pointed at the combination rather than the statistics: the syntax cap threw away the coherence information that separates shuffled words from random words, and a geometric mean of about eight sub-scores, each roughly uniform on human text, let one weak sub-score pull a human sample down.

### 4.2 v2: from percentiles to evidence

v2 replaced percentiles by the robust z-scores of Section 3.4, so that a value far outside the human range keeps its distance. The search that followed used dev ladder A (enwik8 [56, 64) MiB) for choices and dev ladder B ([64, 72) MiB) as a check, never the fresh region. Its steps, each measured:

1. **Sum of squared excess z over all statistics, coherence two-sided:** 2 of 5 pairs on ladder A. One unbalanced quote (a single failed well-formedness check) placed a human sample at z ≈ −5.
2. **Coherence one-sided.** The all-pairs sentence cosine alone separates human from shuffled sentences at 0.989 when only low cohesion counts, against 0.972 two-sided.
3. **No additive weighting passes.** Ordering "shuffled sentences > shuffled words" needs syntax weighted above coherence; but that also amplifies the syntax noise of human samples and breaks "human > shuffled sentences". Across several thousand additive settings (squared or linear evidence; sum, mean or maximum within an axis; axis weights from 1 to 4), none passed all five pairs on ladder A; the best minimum pair accuracy was 0.926.
4. **Categorical form and integrity.** Counting syntax and integrity only beyond a high human quantile, and coherence from the dead zone on, gave 4 of 5 pairs. The remaining conflict was inside word salad: how badly a text fails syntax varied more than the coherence signal that orders shuffled words above random words.
5. **Context gain.** Raw n-gram probability is low for word salad and also for human text about rare topics. Subtracting each word's frequency-only log-probability leaves the part the context explains. Shuffled words moved from a median bad-side z of 3.1–3.5 (raw probability) to 6.8–8.8 (gain), with none inside the human 99.5th percentile. Samples from a byte-level count language model moved from 19% to 96% beyond the human 99th percentile.
6. **Two form failures.** Once syntax saturates, real words and non-words received the same penalty. Splitting lexical validity (L) from syntax (S), each saturating on its own, passed all five pairs on both development ladders.

The parameters of the final structure are not a knife-edge. Of 2,592 settings of the thresholds, cap, softness, integrity weight and dead zone tried on both development ladders, 1,729 pass all five pairs on both, and every one-step change of a frozen parameter passes. Section 6.7 repeats the main changes on the fresh region. The design was then frozen and committed before the fresh region was read (repository commit `c7fd176`).

The prompted profile's relevance axis was designed afterwards, on development prompts from [56, 64) MiB, against a prompted ladder: true continuation > another document's true continuation > half echo of the prompt > the prompt echoed. Relevance graded from z = 0 orders it best, but then penalises half of all human continuations; a dead zone of 0.5 with weight 8 keeps accuracies of 0.963, 0.967 and 1.000 on that ladder.

## 5. Evaluation protocol

All splits were fixed before v2 was designed and are listed in the study plan in the repository (`docs/SCRIBE_PAPER_PLAN.md`), which also holds the twenty predictions registered before the first v2 run (Section 6.9).

### 5.1 Data

**Table 4.** Genres, packs, references and test texts. MiB = 2²⁰ bytes; enwik8 and enwik9 are the first 10⁸ and 10⁹ bytes of the March 2006 English Wikipedia dump (Mahoney, 2011).

| Genre | Pack | Reference | Development | Test |
|---|---|---|---|---|
| Wikipedia | enwik8 [16, 48) MiB, 2,600 articles, 17.5 MB of prose | [48, 56) MiB, 1,909 + 1,729 samples | ladders A [56, 64) and B [64, 72) MiB | fresh ladder [72, 95.37) MiB; model-ladder prompts from the same range |
| Web | WebText train, 14,936 documents, 32.2 MB of prose | WebText valid, 5,000 documents | none (no design choice) | WebText test and the GPT-2 output dataset (Radford et al., 2019; Solaiman et al., 2019); the MAUVE human evaluation |
| Books | 80 Project Gutenberg novels, 53.5 MB | 20 novels | none | 10 novels (split by book) |
| Stories | WritingPrompts train (Fan et al., 2018), 14,068 stories | WritingPrompts valid | none | HANNA and OpenMEVA-WP stories (prompts from the WritingPrompts test split) |
| ROC | OpenMEVA ROC train (Mostafazadeh et al., 2016), 87,175 stories | OpenMEVA ROC dev, 4,782 stories (first sentence as prompt) | none | OpenMEVA-ROC stories |

Every pack and reference is disjoint from every test text; documents identical to a test text were removed from the packs. The n-gram generators of Section 5.2 train on enwik8 [8, 16) MiB, apart from every SCRIBE pack, and their prompts come from the fresh region. The question-answering test uses Dolly 15k (Conover et al., 2023): two thirds of the rows form the pack, and the rest alternate between reference and test.

### 5.2 Tests

1. **Control ladders.** Rungs, best first: held-out human samples; sentences shuffled across documents; words shuffled within sentences; random lexicon words (frequency-weighted, in place); random letters (same lengths); the first six words repeated as a sentence. Every control keeps the sample's sentence lengths, terminal punctuation and capitals. 500 standard human samples per ladder. A pair passes at ordering accuracy ≥ 0.95 over all (upper, lower) pairs, ties counting one half (this accuracy is the AUROC), and disjoint 95% bootstrap intervals of the two rung means (1,000 resamples). Run on the fresh Wikipedia region (read once), WebText and books, and on WebText against the Wikipedia reference.
2. **Held-out corruptions.** Nine families, none used in the design: random-word substitution, adjacent word swaps, sentence reordering within the text, sentence repetition, content-word swaps from other documents, typos, truncated sentences, off-topic sentences and dropped words, each at strengths 0, 10, 25, 50, 75 and 100% on 200 fresh human samples (10,800 texts).
3. **Model ladders**, whose order is known in advance. Word n-gram generators (Kneser–Ney, orders 1–5) continue 250 openings of 100–300 bytes from the fresh Wikipedia region; the human rung is the true continuation of the same openings. The GPT-2 output dataset (Radford et al., 2019) has four model sizes from 117M to 1.5B parameters, each under pure sampling and top-k 40, with 1,000 samples each and 1,000 WebText texts. A planned ladder of Pythia training checkpoints and sizes (Biderman et al., 2023) and of GPT-2 decoding settings was not run. Generating it on the available CPU would have taken about 20 hours.
4. **Question answering.** 600 Dolly test questions, each with four answers, best first: the human answer; the human answer to another question of the same category; the question echoed to the answer's length; and the human answer with its words shuffled inside each sentence. Scored with the prompted profile against a Dolly pack and reference.
5. **Human judgments.** The MAUVE human evaluation (Pillutla et al., 2021): 3,240 pairwise judgments of WebText continuations from human writers and eight GPT-2 configurations, on three questions (more interesting, makes more sense, more likely written by a human). HANNA (Chhun et al., 2022): 1,056 stories from ten systems and humans, three raters per story on six criteria, shipped with 72 automatic metrics and five LLM judges. OpenMEVA MANS (Guan et al., 2021): 1,000 ROC and 1,000 WritingPrompts stories rated for overall quality.
6. **Attacks** on 250 fresh Wikipedia prompts: copied pack text, the prompt echoed, keyword stuffing, a fluent generic paragraph with and without the prompt's topic words inserted, randomised numbers, inserted negations, a function-word loop, and off-prompt human text.
7. **Stability:** correlation with length; five seeds; reference size from 50 to 1,000 samples (three draws each); pack size from 8 to 128 MiB.
8. **Ablations** of every axis and every statistic, and of each design decision.
9. **Cost:** time and memory per component and per metric, on one 4-core CPU without a GPU.

### 5.3 Baselines

All baselines score exactly the texts SCRIBE scores, or a fixed subset of them stated with each result. **Earlier versions:** surface S and SCRIBE v1. **Statistical:** distinct-1–4, repetition rates rep-2–4 and seq-rep-4 (Welleck et al., 2020), their product as a diversity score, type–token ratio and gzip compression ratio; per system, self-BLEU and the Zipf coefficient. **Neural:** GPT-2 (124M) perplexity (`openai-community/gpt2`); grammatical acceptability, the mean probability of "acceptable" over sentences from a RoBERTa-base CoLA classifier (`textattack/roberta-base-CoLA`; Warstadt et al., 2019); sentence-embedding coherence, the mean cosine of adjacent sentences under all-MiniLM-L6-v2 (Reimers and Gurevych, 2019); the RoBERTa GPT-2 output detector's probability of human text (Solaiman et al., 2019); and set-level MAUVE (Pillutla et al., 2021) with GPT-2 features. **Shipped with HANNA:** 72 automatic metrics (among them BLEU, ROUGE, METEOR, chrF, BERTScore, MoverScore, BaryScore, BARTScore and SUPERT) and the ratings of LLM judges (ChatGPT, Mistral-7B, Llama-13B, Beluga-13B). **Published:** the MAUVE scores of the MAUVE human evaluation. Baselines run with their published defaults; nothing is tuned for any metric on test data. The neural baselines are slow on a 4-core CPU, so they read only the first 256 GPT-2 tokens of each text and run on fixed subsets. GPT-2 large, conditional perplexity and an LLM judge of our own were planned but not run.

### 5.4 Statistics

Rung and system means carry 95% percentile bootstrap intervals (1,000 resamples). Ordering accuracy counts ties as one half. Agreement with pairwise human choices carries Wilson intervals; Bradley–Terry strengths are fitted to the human choices with a small ridge (Bradley and Terry, 1952). Correlations with ratings are Pearson, Spearman and Kendall's tau-b (Kendall, 1945) with item bootstrap intervals; differences between metrics use a paired bootstrap test with Benjamini–Hochberg adjustment (Benjamini and Hochberg, 1995). The statistics module reproduces the MAUVE paper's published system-level correlations exactly (Spearman 0.810, 0.857 and 0.952 for the three questions) when given the published MAUVE scores.

## 6. Results

<!-- RESULTS -->

## 7. Discussion

**What counts can see.** SCRIBE v2's statistics see whether a text is made of words, whether its words follow each other the way they do in human text once their frequency is accounted for, whether its sentences share vocabulary and topic, whether it repeats itself, whether it copies its training text, and how it overlaps with a prompt. With those, it orders every rung of the control ladder in three genres, follows seven of nine held-out degradations, catches the repetition, echo, stuffing and copying attacks, and puts word-salad generators (n-gram models) far below human text.

**What counts cannot see.** Four blind spots showed up, each a direct consequence of a design choice:

1. *Sentence order.* All cohesion statistics are symmetric in the order of sentences, so reordering the sentences of a text costs nothing (Table 8). An ordered statistic, such as the entity grid's transition model or the probability of discourse connectives given the previous sentence, is the obvious addition.
2. *Truncation with punctuation.* A sentence cut in half and closed with a full stop is still well formed and still fluent word by word; the context gain ignores the end-of-sentence token. Scoring the end-of-sentence event would catch it.
3. *Truth.* Random numbers and inserted negations keep their human parity. SCRIBE has no belief axis; the PLAN's B axis needs a fact store and claim extraction, which this version does not have.
4. *Too typical.* SCRIBE v2 never penalises text for being more predictable or more cohesive than human text. Truncated sampling (top-k 40) produces exactly such text, and v2 scores it above WebText itself (Section 6.3). An exploratory two-sided variant puts human text first and roughly orders the top-k models by size while still passing the ladder, but it does not agree better with people on the MAUVE pairs. Whether a quality metric should penalise excess typicality is a value judgment: detectors exploit it (Gehrmann et al., 2019; Ippolito et al., 2020), people often prefer it, and typical-set decoding (Meister et al., 2023) treats it as a flaw.

**Passing the ladder is not the same as ranking fluent texts.** v2's dead zones make it a strict judge of whether a text is human-like at all: most human samples score near 1, and every degradation falls far below. Inside the range of fluent text it has little resolution, and on HANNA's graded ratings SCRIBE v1, whose percentile scores keep the ordinary variation of human text, correlates better with people than v2 does. A deployment might use both: v2 as the gate that a model must pass, and a graded within-human-range score to rank models that pass it.

**Where the line falls.** The control ladder puts "sentences shuffled across documents" just below human text. On Wikipedia every n-gram generator falls far below that rung (means 0.001–0.089 against 0.354). On web text every GPT-2 configuration scores above it (0.71–0.97 against 0.43). So the ladder gives a model builder a concrete first target: a model whose samples do not score above shuffled sentences has not yet learned to write prose, however well it compresses text.

**Cost.** SCRIBE needs no GPU and no downloaded model. It scores about a thousand texts per second on one core with 0.4 GB of memory. That is over 200 times faster than GPT-2 small perplexity on two cores of the same CPU, and several hundred times faster than the other neural baselines (Section 6.8). It can run inside a training loop, on every checkpoint, or on every generated answer.

## 8. Limitations

- **English only, four genres.** Every pack and reference is English: Wikipedia, web text, novels and short stories. Another language or genre needs its own pack and reference (100 or more reference samples per length bucket and a few MiB of pack text were enough here).
- **Reference dependence.** SCRIBE measures distance from its reference. A reference of the wrong genre lowers human parity (web text against the Wikipedia reference keeps 0.90 of it), and SCRIBE cannot rate text that is better than its reference in ways the statistics do not see.
- **Blind spots.** Sentence order, truncation with punctuation, factual errors and excess typicality (Section 7). The relevance axis is lexical: it separates right from wrong answers on Dolly and on Wikipedia prompts, but not good from bad stories for a creative prompt (HANNA relevance).
- **Graded agreement.** Item-level correlations with people are modest (Kendall up to about 0.2 on stories), like most metrics in Table 11; SCRIBE is stronger at system level and at ordering clear differences than at ranking fluent texts.
- **Design data.** The control ladder's corruption types were used in design (on development text), so the fresh ladder tests generalisation to new text, not to new kinds of damage. The held-out families, the model ladders and the human judgments test the latter, and they show the limits above.
- **Compute.** Everything ran on one 4-core CPU without a GPU. The neural baselines read only the first 256 GPT-2 tokens of each text and ran on fixed subsets. No LLM judge of our own was run, so the comparison with LLM judges rests on the judges shipped with HANNA. The model ladders stop at GPT-2 XL (1.5B parameters), and the planned Pythia and decoding ladders were not run, so three registered predictions on them are untested.
- **No rating study of our own.** The study plan's own rating test (300 samples rated by the author) was not run; published ratings stand in for it.

## 9. Conclusion

<!-- CONCLUSION -->

## Reproducibility

<!-- REPRO -->

## References

Arora, S., Liang, Y., and Ma, T. (2017). A simple but tough-to-beat baseline for sentence embeddings. *ICLR*.

Banerjee, S., and Lavie, A. (2005). METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. *ACL Workshop on Intrinsic and Extrinsic Evaluation Measures*.

Barzilay, R., and Lapata, M. (2008). Modeling local coherence: An entity-based approach. *Computational Linguistics*, 34(1).

Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1).

Biderman, S., et al. (2023). Pythia: A suite for analyzing large language models across training and scaling. *ICML*.

Bradley, R. A., and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4).

Caccia, M., Caccia, L., Fedus, W., Larochelle, H., Pineau, J., and Charlin, L. (2020). Language GANs falling short. *ICLR*.

Chen, S. F., and Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. *Computer Speech and Language*, 13(4).

Chhun, C., Colombo, P., Suchanek, F. M., and Clavel, C. (2022). Of human criteria and automatic metrics: A benchmark of the evaluation of story generation. *COLING*.

Chhun, C., Suchanek, F. M., and Clavel, C. (2024). Do language models enjoy their own stories? Prompting large language models for automatic story evaluation. *Transactions of the ACL*, 12.

Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.

Fan, A., Lewis, M., and Dauphin, Y. (2018). Hierarchical neural story generation. *ACL*.

Gehrmann, S., Strobelt, H., and Rush, A. M. (2019). GLTR: Statistical detection and visualization of generated text. *ACL: System Demonstrations*.

Gehrmann, S., Clark, E., and Sellam, T. (2023). Repairing the cracked foundation: A survey of obstacles in evaluation practices for generated text. *Journal of Artificial Intelligence Research*, 77.

Guan, J., and Huang, M. (2020). UNION: An unreferenced metric for evaluating open-ended story generation. *EMNLP*.

Guan, J., Zhang, Z., Feng, Z., Liu, Z., Ding, W., Mao, X., Fan, C., and Huang, M. (2021). OpenMEVA: A benchmark for evaluating open-ended story generation metrics. *ACL*.

Hashimoto, T. B., Zhang, H., and Liang, P. (2019). Unifying human and statistical evaluation for natural language generation. *NAACL*.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. (2020). The curious case of neural text degeneration. *ICLR*.

Ippolito, D., Duckworth, D., Callison-Burch, C., and Eck, D. (2020). Automatic detection of generated text is easiest when humans are fooled. *ACL*.

Kendall, M. G. (1945). The treatment of ties in ranking problems. *Biometrika*, 33(3).

Kneser, R., and Ney, H. (1995). Improved backing-off for m-gram language modeling. *ICASSP*.

Lapata, M. (2003). Probabilistic text structuring: Experiments with sentence ordering. *ACL*.

Levy, O., Goldberg, Y., and Dagan, I. (2015). Improving distributional similarity with lessons learned from word embeddings. *Transactions of the ACL*, 3.

Li, J., Galley, M., Brockett, C., Gao, J., and Dolan, B. (2016). A diversity-promoting objective function for neural conversation models. *NAACL*.

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. *ACL Workshop on Text Summarization Branches Out*.

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., and Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *EMNLP*.

Mahoney, M. (2011). Large Text Compression Benchmark. https://mattmahoney.net/dc/text.html.

Meister, C., and Cotterell, R. (2021). Language model evaluation beyond perplexity. *ACL*.

Meister, C., Pimentel, T., Wiher, G., and Cotterell, R. (2023). Locally typical sampling. *Transactions of the ACL*, 11.

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., and Finn, C. (2023). DetectGPT: Zero-shot machine-generated text detection using probability curvature. *ICML*.

Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., Kohli, P., and Allen, J. (2016). A corpus and cloze evaluation for deeper understanding of commonsense stories. *NAACL*.

Novikova, J., Dušek, O., Cercas Curry, A., and Rieser, V. (2017). Why we need new evaluation metrics for NLG. *EMNLP*.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL*.

Pillutla, K., Swayamdipta, S., Zellers, R., Thickstun, J., Welleck, S., Choi, Y., and Harchaoui, Z. (2021). MAUVE: Measuring the gap between neural text and human text using divergence frontiers. *NeurIPS*.

Popović, M. (2015). chrF: Character n-gram F-score for automatic MT evaluation. *WMT*.

Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI technical report.

Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP*.

Reiter, E. (2018). A structured review of the validity of BLEU. *Computational Linguistics*, 44(3).

Ribeiro, M. T., Wu, T., Guestrin, C., and Singh, S. (2020). Beyond accuracy: Behavioral testing of NLP models with CheckList. *ACL*.

Sai, A. B., Dixit, T., Sheth, D. Y., Mohan, S., and Khapra, M. M. (2021). Perturbation CheckLists for evaluating NLG evaluation metrics. *EMNLP*.

Sellam, T., Das, D., and Parikh, A. (2020). BLEURT: Learning robust metrics for text generation. *ACL*.

Solaiman, I., et al. (2019). Release strategies and the social impacts of language models. arXiv:1908.09203.

Warstadt, A., Singh, A., and Bowman, S. R. (2019). Neural network acceptability judgments. *Transactions of the ACL*, 7.

Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., and Weston, J. (2020). Neural text generation with unlikelihood training. *ICLR*.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158).

Yuan, W., Neubig, G., and Liu, P. (2021). BARTScore: Evaluating generated text as text generation. *NeurIPS*.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *ICLR*.

Zhao, W., Peyrard, M., Liu, F., Gao, Y., Meyer, C. M., and Eger, S. (2019). MoverScore: Text generation evaluating with contextualized embeddings and Earth Mover distance. *EMNLP*.

Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS Datasets and Benchmarks*.

Zhu, W., and Bhat, S. (2020). GRUEN for evaluating linguistic quality of generated text. *Findings of EMNLP*.

Zhu, Y., Lu, S., Zheng, L., Guo, J., Zhang, W., Wang, J., and Yu, Y. (2018). Texygen: A benchmarking platform for text generation models. *SIGIR*.


## Appendix A. Every parameter of SCRIBE v2

**Table A1.** Parameters, with the stage at which each was fixed. "Inherited" means it comes from v0 or v1 unchanged; "dev" means it was chosen on development ladders A and B (or, for relevance, development prompts).

| Parameter | Value | Fixed |
|---|---|---|
| Text cleaning (Wikipedia) | entities, tags, `<ref>`, templates, tables and links stripped; prose lines of ≥ 60 bytes kept | inherited |
| Text cleaning (other JSONL) | prose lines of ≥ 20 bytes starting like a sentence and ending in . ! or ? | v2 |
| Sentence splitter, tokenizer | abbreviation and initial rules; lowercased words; hyphens split; digit groups kept | inherited |
| Lexicon | words seen ≥ 2 times in the pack | inherited |
| Word trigram | interpolated Kneser–Ney, one discount per order `n1/(n1 + 2 n2)`, out-of-vocabulary penalty 8 nats (only in s3, not scored) | inherited |
| Skeleton classes | 150 most frequent pack words keep their identity; suffix classes -ing, -ed, -ly, nominal (-tion -sion -ment -ness -ity -ism), adjectival (-al -ive -ous -ful -able -ible -ic -ish), -er/-or, plural -s, other alphabetic, non-alphabetic, number | v2 |
| Context gains s2g, s3g | mean over in-lexicon tokens of trigram minus unigram log-probability; s3g needs ≥ 3 such tokens | dev |
| Content words | sentence IDF ≥ 3 (i.e. in fewer than about 5% of pack sentences) | inherited |
| Word vectors | PPMI-SVD, window ±5 with 1/distance weights, context smoothing 0.75, 50,000 words, 300 dimensions, `U·S^0.5`, rows centred and normalised | inherited (C1) |
| Sentence vectors | SIF, `a = 10⁻³`, first principal component of 60,000 pack sentences removed | inherited |
| Short-text cohesion | a text with fewer than two embeddable sentences is cut into two halves of its tokens (≥ 8 tokens) | inherited |
| 8-gram copy set | all word 8-grams of the pack, hashed to 64 bits | v2 |
| Length buckets | short < 300 bytes ≤ standard; reference samples 60–260 and 400–600 bytes; a bucket with < 20 reference samples falls back to the other | inherited / v2 |
| z-score | mid-rank ECDF inside the 2nd–98th percentile; linear tails with scales from the 2nd–16th and 84th–98th percentiles | dev |
| Tail floors | s1 0.01; s2g, s3g 0.05; s4 0.15; c0, c2 0.005; c1, c1a, c3 0.01; loop8 0.01; rep 0.05; d1, d2 0.01; d3, d4 0.005; i1 0.01; r1 0.01; r4, r5 0.02; i4 0.01 | dev (s4), otherwise first choice |
| Missing statistic | counts as `u = 4` if ≥ 95% of reference samples of the bucket have it (not for relevance) | dev |
| Sides | Table 1; coherence one-sided | dev |
| Dead zone | 1 (relevance 0.5) | dev |
| Thresholds θ_L, θ_S, θ_I | 99th percentile over the reference samples of the bucket | dev |
| Saturation | height 30, softness 4, for L and S | dev |
| Weights | integrity 3, coherence 1, relevance 0 (open) or 8 (prompted) | dev |
| Score | `exp(−D/5)` | dev (monotone; affects no ordering) |
| Long texts | windows of whole sentences: a window closes before 600 bytes once it holds 300; a last window under 300 bytes joins the previous one; the score is the mean over windows | v2 |

**Exploratory option (not v2).** `twosided=1` also penalises s2g, s3g and every coherence statistic when they are higher than human text; it is reported only in Sections 6.3 and 6.4 and was decided after those results were seen.