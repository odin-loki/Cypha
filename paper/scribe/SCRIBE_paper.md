---
title: "SCRIBE v3: Three Count-Based Judges of Generated Text"
author: "Odin Loch"
date: "4 November 2026"
---

# SCRIBE v3: Three Count-Based Judges of Generated Text

**Odin Loch**

<p class="meta">Version 2.0 · 4 November 2026 · updates version 1.0 (28 October 2026, SCRIBE v2)<br>Code: <code>SCRIBE_code_v2.0.zip</code> · Result tables: <code>SCRIBE_results_v2.0.zip</code> · both published with this paper</p>

## Abstract

SCRIBE is a metric for generated text built from counts alone, with no neural network. Version 1.0 described SCRIBE v2, which reads about twenty statistics of a text against human reference text and passes a control ladder of degraded text where standard per-text metrics fail. This version adds two judges of other kinds. SURPRISE reads the text's code length word by word under a count model of the pack (a Kneser–Ney trigram with spelling models, a cache and a match model), two-sided, so text more predictable than human text loses as surely as salad. MATCH reads matching statistics against a suffix array of the pack, the text's Lempel–Ziv complexity and echo of a prompt. Neither uses word vectors or co-occurrence. They were built because a writer, K5, gained +0.145 on v2 from a sentence gate that picks sentences by word co-occurrence with the prompt, while v2's coherence axis also rests on co-occurrence. The new judges do not confirm that gain (SURPRISE −0.106) or a later gain from more training text. SCRIBE v3, the plain mean of v2, SURPRISE and MATCH, passed its fresh control ladder but caught only 5 of 9 registered attacks, because one judge's catch was diluted. SCRIBE v3.1, the recommended form, is the geometric mean of v2 and new versions of the judges that add statistics for words that fit neither neighbour and for bland text. On two fresh Wikipedia regions it passes all five ladder pairs (0.966–0.968 on the hardest) and all 9 registered attacks (v2: 5), catching copied prompt sentences, keyword stuffing at one word in five or more, and generic paragraphs. It ranks human web text above every GPT-2 top-k system (Spearman +0.90; v2 −0.70), fixing v2's "too typical" blind spot in the mean, and it sees relevance to a story prompt (HANNA Kendall 0.169; v2 0.017). It scores about 590 texts per second on one CPU core. It is not fool-proof: it passes only 2 and 3 of 8 held-out attacks; stuffing at one word in seven, a template with the prompt's names, a title swap and loosely related fluent sentences get through. Sentence order, truncated sentences and truth are invisible to all three judges; it does not rank models that all write salad, passes 3 of 5 pairs on novels and 4 of 5 on web text, cannot tell a right answer from another question's on Dolly, and needs 500 reference samples per bucket. Its agreement with people is about v2's. Of twenty predictions registered before this version's reads, twelve hit, five hit in part and three missed.

---

## 1. Introduction

Version 1.0 of this paper described SCRIBE v2, a metric for generated text built from counts alone (Loch, 2026). It reads about twenty statistics of a text against human reference text of the same genre and length, and combines them with a fixed rule: categorical evidence for texts that are not made of words or not in a human word order, graded evidence for coherence, and a separate penalty for degeneration. v2 passed every pair of a control ladder on fresh Wikipedia text, where no standard per-text metric did. Its stated blind spots were sentence order, truncated sentences, factual errors, and text more predictable than human text: it rated GPT-2 top-k samples above human web text, and it could not rank models that all write word salad.

This version adds two judges and a way to combine them. It was prompted by a result that could not be checked with v2 alone.

**The problem that started v3.** CyphaLM's best writer, a word-level Kneser–Ney model of order 5 called K5, was upgraded in a series of registered steps (Section 3.1). The largest step was a sentence gate: K5 writes eight candidates for each sentence and keeps the one whose words co-occur most with the prompt's words in K5's own training text. On SCRIBE v2 this raised K5's score from 0.353 to 0.498 (+0.145), most of it on v2's coherence axis. But v2's coherence axis also rests on co-occurrence: its sentence vectors are PPMI-SVD vectors, a factorisation of word co-occurrence counts. The gate reads nothing of SCRIBE. Still, both measure the same kind of thing, so part of the gain might be two co-occurrence measures agreeing rather than better text. A judge of another kind was needed.

**What this version adds.**

1. **SURPRISE**, a judge from information theory: the code length of the text, word by word, under a model of the pack (a Kneser–Ney trigram with spelling models, a cache and a match model), read two-sided, so that text more predictable than human text loses as surely as salad.
2. **MATCH**, a judge from string matching: matching statistics of the text against a suffix array of the pack, the text's own Lempel–Ziv complexity, and echo of the prompt.
3. **SCRIBE v3**, the plain mean of v2, SURPRISE and MATCH. It passed its control ladder on fresh text but failed its registered attack rule (5 of 9 attacks), because one judge's catch was diluted by two judges that saw nothing.
4. **SCRIBE v3.1**, the recommended form: the geometric mean of v2 and new versions of the two judges, SURPRISE 3.1 and MATCH 3.1, which add statistics for words that fit neither neighbour (keyword stuffing) and a group for text that is too bland. It passes its ladder and all nine registered attacks on two fresh regions.
5. **A re-run of version 1.0's meta-evaluation** with v3 and v3.1 next to v2, wherever it could be re-run: genre ladders, the GPT-2 output dataset, held-out corruptions, n-gram generators, question answering, attacks, human judgments, stability, K5 and cost. Twenty predictions for these reads were registered before they were run (Section 9.12).

Neither new judge uses word vectors or any co-occurrence statistic. Neither confirmed the sentence gate's gain: v3 moved by +0.011, inside its intervals, and SURPRISE fell by −0.106.

**Why the paper presents v3.1 and tells the v3 story inside it.** v3.1 is the form to use: it is the only one of the three combinations that passes both the ladder and the registered attacks. But v3.1 was designed after v3's fresh read had shown which attacks v3 missed, so its first fresh read is not blind to the kinds of attack it was built against. Showing v3 in full, with its failed rule, makes that history visible, and the held-out attacks (written before any v3.1 design work) and a second fresh region read for this paper are the checks that do not depend on it.

**What did not change.** v2 is unchanged and frozen, and v3.1 contains it as one of its three judges. Profiles v1, v2, v3 and v31 of the released tool reproduce their earlier outputs byte for byte. Everything in version 1.0 about v2 still holds; this paper repeats only what is needed to read v3.

## 2. Related work

Version 1.0 reviews reference-based metrics, reference-free and distributional metrics, fluency from language-model probability, language models as judges, coherence models and meta-evaluation (its Section 2). This section adds the work behind the two new judges.

**Code length and surprisal.** The information content of a text under a model is its code length (Shannon, 1948); a text whose words are much more or much less predictable than human text's is atypical. Per-word surprisal under a language model predicts reading difficulty (Hale, 2001; Levy, 2008), and its distribution, not only its mean, separates human text from sampled text: GLTR (Gehrmann et al., 2019) and DetectGPT (Mitchell et al., 2023) use it to detect machine text, and locally typical sampling (Meister et al., 2023) argues that human text keeps the information per word near its expected value. SURPRISE reads these quantities two-sided, against human text, under a count model. Its model adds a unigram cache (Kuhn and De Mori, 1990) and a match model in the style of PPM and PAQ compressors (Cleary and Witten, 1984; Mahoney, 2005), with Witten–Bell spelling models for unknown words (Witten and Bell, 1991).

**String matching.** Matching statistics, the longest prefix of each suffix of a text that occurs in another text, come from the suffix-tree literature (Chang and Lawler, 1994) and are computed here with a suffix array (Manber and Myers, 1993). Long matches with a training corpus measure copying (Carlini et al., 2021; McCoy et al., 2023; Liu et al., 2024, whose ∞-gram counts n-grams of unbounded length the same way). Lempel–Ziv complexity (Lempel and Ziv, 1976) measures how compressible a text is in itself; loops make it fall and noise makes it rise.

**Pointwise mutual information.** SURPRISE 3.1's word fit is minus the pointwise mutual information of a word with its two-sided context (Church and Hanks, 1990), computed from the pack's trigram. It is close in spirit to the context gain of v2 (the per-word SLOR of Pauls and Klein, 2012, and Lau et al., 2017), but it asks whether the sentence reads better *without* the word, which is what an inserted keyword does.

**Combining judges.** Ensembles of metrics are usually averaged or learned (e.g. Sellam et al., 2020). A plain mean dilutes a confident catch by one member. A geometric mean of scores of the form exp(−D/τ) adds the members' deviances instead, which is the combination of independent pieces of evidence in log space.

## 3. The challenges

This section lists every problem met between version 1.0 and this paper, and what was done about each. The numbers come from the reports of each step (`docs/reports/` in the code archive) and from Section 9.

### 3.1 The co-occurrence circularity found through K5

K5's upgrades (`K5_UPGRADES.md`) were registered one step at a time, each adopted only if it raised SCRIBE v2 on 250 development prompts by at least 0.01 over 4 fresh seeds and at 3 or 4 of them, without more copying. Nucleus 0.8, 256 word classes as the backoff floor, and the 8-candidate sentence gate by prompt association were adopted. The gate gave the largest gain: v2 0.444 → 0.498 at the first seed, +0.079 over four, nearly all on the coherence axis (D_C 4.15 → 3.41). On one test read, the upgraded K5 scored 0.524 against K5's 0.305.

The circularity is this. The gate scores a candidate sentence by the positive PMI of its words with the prompt's words, counted within 10 words in K5's training text. v2's coherence statistics c0, c1 and c1a are cosines of PPMI-SVD vectors, which are co-occurrence within 5 words in SCRIBE's pack. Different text, different window, same kind of quantity. A sentence chosen for co-occurrence with the prompt will look cohesive to a co-occurrence measure even when a reader would not find it so. The upgrade report flagged this itself and asked for a judge of another kind.

The same question came up for scale. K5 trained on 330 MB of clean Wikipedia from enwik9 gained +0.046 on v2 over the 7 MB text (4-seed means, `K5_SCALE.md`). The upgraded K5 did not gain from it at all (−0.021).

**What was done.** SURPRISE and MATCH use no word vectors and no co-occurrence statistic. They see cohesion only as exact repetition: content words that a cache of the text's other sentences predicts (SURPRISE), and strings shared between sentences, weighted by how rare they are in the pack (MATCH). On the gate, SURPRISE fell by 0.106 and MATCH by 0.009, against v2's +0.145 (Section 9.10). On the old K5's scale gain, v3 moved by +0.001 against v2's +0.046. The gains rest on v2 alone. Whether they are real is a question for a human read; counts of a different kind do not confirm them.

### 3.2 v2's known blind spots

Version 1.0 listed them. Here is what v3 and v3.1 do about each.

- **Text more predictable than human text.** v2 counts only the bad side of each statistic, so it never penalises text for being too fluent or too cohesive. It rated GPT-2 top-k 40 samples (0.968–0.972) above WebText (0.925). SURPRISE and MATCH are two-sided by construction, and v3.1 adds a bland group. On new draws of the GPT-2 output dataset, v3 and v3.1 put WebText above all four top-k systems, where v2 puts it below all four (Section 9.4). **Fixed in the mean; the per-text margin is small.**
- **Models that all write salad.** v2's form axes saturate, so n-gram orders 3 to 5 tie. SURPRISE and MATCH saturate less (Section 9.4). **Partly.**
- **Sentence order.** No statistic of any judge depends on the order of a text's own sentences beyond adjacent pairs. A probe for it in v3.1 (`w_lead`, reported only) ranks human text above reordered text at 0.556 on development data. **Not fixed.**
- **Truncated sentences.** A sentence cut in half and closed with a full stop stays fluent word by word. v3.1's word fit sees the sentence end, but on the dose test it falls less with truncation than v2 does (Section 9.3). **Not fixed.**
- **Factual errors.** No judge has a notion of truth: random numbers and inserted negations keep human parity under all three (Section 9.6). **Not fixed.**

### 3.3 Dilution in a plain mean of three

v3 was the plain mean of v2, SURPRISE and MATCH. Its attack rule required v3 to rank human text above each of nine registered attacks with accuracy ≥ 0.95, and SURPRISE or MATCH to catch each. On its fresh read the rule passed 5 of 9. Two of the misses were caught by a judge and lost in the mean:

- the prompt's own sentences followed by the true continuation (`echo_copy`): MATCH alone 1.000, v3 0.948;
- v2's generic, contentless paragraph: SURPRISE 0.970 and MATCH 0.967, v3 0.930.

A text that one judge rejects with a deviance of several hundred still keeps two thirds of its score if the other two rate it human. **What was done.** v3.1 takes the geometric mean. Each judge scores exp(−D/τ), so the geometric mean is exp(−(D_v2/5 + D_S/τ_S + D_M/τ_M)/3): the deviances add, and a confident catch drives the score to zero whatever the other two say. Power means with negative exponents and mixtures of the mean and the minimum did no better on development data (`SCRIBE_V31.md`, design log arm 3). The price is the other side of the coin: a human text that one judge rejects is not rescued either.

On the new fresh read of this paper, the combination ablation shows both sides (Section 9.11): the plain mean of the v3.1 judges passes 7 of 9 registered attacks (copied prompt sentences 0.942, stuffed generic text 0.935), the geometric mean 9 of 9.

### 3.4 MATCH failing a ladder pair

v3's rule 1 required SURPRISE alone, MATCH alone and the v3 mean each to pass all five pairs on the fresh ladder. MATCH failed "human > sentences shuffled across documents" (0.940). Without co-occurrence, MATCH sees whether sentences belong together only through exact strings they share, and on development data it already had the thinnest margins (0.953, 0.955). The v3 mean passed (0.981).

**What was done.** Nothing in MATCH: a judge that is weak on one pair can still be useful in a combination, and v3.1's rule asks only the combination to pass the ladder. The new judges are weaker alone in v3.1 (SURPRISE 3.1 passes 3–4 of 5 on fresh Wikipedia, MATCH 3.1 3 of 5) because each failure mode a judge learns adds false positives on odd human texts. Only the geometric mean passes 5 of 5. **v3.1 should be read as one score.**

### 3.5 Keyword stuffing caught by rate, not pattern

Neither v3 judge caught keyword stuffing (another document's continuation with the prompt's five rarest content words inserted after every 4th word): v3 0.825 on fresh data. Each insertion moves several statistics by 2 to 3 standard deviations, and human text reaches that on single statistics, so the 99th-percentile thresholds absorb it. A statistic that caught it in v3 (`w_anti`, words the left context makes less likely) cost SURPRISE a ladder pair and was kept as report-only.

**What was done.** v3.1 asks of each word whether its sentence reads better without it. `w_fit` is the code length of the sentence with the word, minus the code length with the word deleted, minus the word's own unigram code length: minus the PMI of the word with its two-sided context. `m_bridge` counts words whose two pack bigrams (with the left and right neighbour) are both absent while the bigram of the two neighbours is present. Both go into a saturating group F. v3.1 catches the registered stuffing (0.953 on its fresh read, 0.975 on this paper's). **But it catches stuffing by its rate, not its pattern.** On the held-out attacks one inserted word in 3 is caught (0.984–0.995), one in about 6 at random gaps is not (0.83–0.88), and one in 8 is not (0.59–0.64). The fit and bridge statistics are averages over the text; at one word in 8 they move by about 1.8 standard deviations, inside the human range.

### 3.6 The fresh data ran out

Each version is read once on a region of text it has never seen. v2 used enwik8 [72, 95.37) MiB, the last fresh part of enwik8; v3 read the same region (v3's own judges had never seen it, but v2's verdict on it was known). For v3.1 nothing was left in enwik8.

**What was done.** v3.1 read enwik9 [100, 124) MiB, past enwik8's 10⁸ bytes (enwik9 starts with enwik8, checked with `cmp`). Wikipedia dumps hold some pages more than once, and a copy of a pack page in the test region would look like copying. A near-duplicate filter (`--dedup`) was registered: a test document is dropped when more than half of its bytes lie inside full 64-byte matches with the pack's or the reference's text. It was counted before the read with `scribe dedup`, which scores nothing: 2 of 2,106 documents dropped. This paper reads one more region, enwik9 [900, 924) MiB, the start of the v2 study plan's spare region (1 of 4,115 documents dropped). The rest of the spare region, [924, 953) MiB, is left for a later version.

K5's 330 MB training text also comes from enwik9 past enwik8. That does not touch SCRIBE's own ladders and attacks, whose pack and reference are enwik8 [16, 56) MiB.

### 3.7 Test-once discipline

Every version followed the same order: design on development ladders A (enwik8 [56, 64) MiB) and B ([64, 72) MiB) and their attack batteries; freeze, with the rules and predictions in a dated commit; read the fresh region once; change nothing after. Two refinements were needed:

- **Held-out attacks.** v3.1 was designed after seeing which attacks v3 missed, so its registered attacks are not a blind test. Eight more attacks were written and committed before any v3.1 design work and run only by `scribe attacks --heldout`, which was never run on development data. They are reported pass or fail, with no rule.
- **One registered seed for re-draws.** The v2 study did not record the seeds of some of its test sets (the web and books ladders, the GPT-2 samples, the corruptions, the n-gram prompts and the attack set), so those texts cannot be rebuilt byte for byte. For this paper they are drawn again with one registered seed, and v2 is re-scored on the new draws next to v3 and v3.1. Runs whose seeds were recorded reproduce exactly (Section 8).

### 3.8 The cost

v3 builds a suffix array over the pack (84 MB for the 17.5 MB Wikipedia pack, 0.9 s on 4 threads) and two spelling models, and scores each text three times. In the version reports' timings (dev A battery, a busy machine), v2 scored about 1,400 texts a second on one core, v3 about 520–560 and v3.1 about 450; on this paper's idle machine, 1,749, 730 and 587 (Section 9.13). That is still about a hundred times faster than GPT-2 small perplexity on the same CPU, with no GPU and no downloaded model.

## 4. SCRIBE v2, briefly

v2 is unchanged; version 1.0 defines it fully (its Section 3 and Appendix A). In short: a **pack** of human text trains a lexicon, a word trigram, a skeleton trigram over word classes, IDF and PPMI-SVD word vectors; a disjoint **reference** of human text supplies the distribution of every statistic per length bucket. Each statistic becomes a robust z-score against the reference (a mid-rank ECDF with linear tails); its bad side, beyond a dead zone of one unit, is evidence. The axes are summed and combined as

```
D_v2 = 30·sat(L − θ_L) + 30·sat(S − θ_S) + 3·max(0, I − θ_I) + C        sat(x) = 1 − exp(−max(0, x)/4)
SCRIBE v2 = exp(−D_v2/5)
```

with thresholds θ at the reference's 99th percentiles. L is lexical validity, S the context gains and well-formedness, C coherence (one-sided), I integrity (repetition, diversity, copying). Section 6 lists every statistic.

## 5. SCRIBE v3: two new judges

Both new judges use v2's pack and reference, v2's z-score (`QuantileZ`), and v2's rule for a statistic that is missing where 95% of human samples have it (z = 4 on its bad side). Unlike v2, **every statistic is two-sided**: each side of z feeds one group.

### 5.1 SURPRISE

**The model.** The pack's interpolated Kneser–Ney word trigram, proper over its vocabulary. A word it calls unknown is spelled by a character trigram (Witten–Bell) trained on the pack's once-seen word types; a number by one trained on the pack's number types. So every string has a probability. On top, the text's own history: a unigram cache, p = 0.9·p_static + 0.1·c(w)/|H|, and a match model: if the last k ≥ 2 words occurred earlier in the text, the word that followed them gets weight k/(k + 1). The surprisal of a word is s(w) = −log₂ p(w) bits.

**The statistics** (Section 6 gives each one's side and failure): the mean, standard deviation and lag-1 autocorrelation of s; the share of words below 4.35 bits and above 17.45 bits (the 25th and 90th percentiles of the reference's words); the information the trigram context gives (unigram bits minus trigram bits per word); the bits spent spelling unknown words; and cohesion as code length, the bits a sentence's content words (unigram surprisal ≥ 8 bits) save when a cache holds the text's other sentences.

### 5.2 MATCH

**The index.** A suffix array over the pack's cleaned text, documents joined by a newline, sorted on the first 64 bytes of each suffix. From each byte of the text, the longest prefix of the rest that occurs in the pack (up to 64 bytes): one binary search inside the bucket of the first two bytes. The same array counts how often a string occurs.

**The statistics:** the median and 90th percentile of the match lengths; the share of bytes inside matches of ≥ 32 bytes (copying) and only inside matches of ≤ 4 bytes (novel junk); the text's own LZ76 complexity, phrases × log₂ n / n; the share of word starts whose match covers three or more whole words (word order); cohesion as shared strings, the longest match (≥ 5 bytes) from each word start with the text's other sentences, each covered byte weighted by the matched string's information in the pack, log₂((N + 1)/(count + 1))/log₂(N + 1), above a floor of 0.45 so that strings like " the " count nothing; and, with a prompt, the share of bytes inside ≥ 8-byte and ≥ 32-byte matches with the prompt (echo).

### 5.3 The score form

| Group | What | Evidence | Term of D |
|---|---|---|---|
| L | not words | max(0, u − 4) | 30·sat(L − θ_L) |
| S | not word order | max(0, u − 1) | 30·sat(S − θ_S) |
| C | sentences unrelated, or off the prompt | max(0, u) | w_C·C |
| G | too predictable, graded | max(0, u − d_G) | w_G·G |
| X | too predictable, beyond each statistic's own human 99th percentile | Σ max(0, e_k − θ_k), e_k = max(0, u − 1) | w_P·X |

Here u is the bad side's |z|, sat(x) = 1 − exp(−max(0, x)/κ), θ are 99th percentiles over the reference, and score = exp(−D/τ). L's dead zone is wide (4) because random letters sit at z 17–33 while human tails reach 2–3. The too-predictable side has two parts because one group sum with one threshold let Wikipedia boilerplate, whose long pack matches make a heavy human tail, hide every other predictable signal. Settings (dead zones of L, S, C, G, X; cap and κ; w_C, w_G, w_P; τ): SURPRISE 4, 1, 0, 1.5, 1; 30 and 2; 4, 1, 8; 20. MATCH 4, 1, 0, 2, 1; 30 and 0.25; 6, 1, 20; 30. τ is set so that dev A's human samples average 0.909 under each judge, as under v2; it changes no order within a judge.

**SCRIBE v3** = (v2 + SURPRISE + MATCH)/3. The tool also reports `min3`, the least of the three, and flags a text `disagree` when the largest minus the smallest exceeds 0.5.

## 6. SCRIBE v3.1

**SCRIBE v3.1** = (v2 · SURPRISE 3.1 · MATCH 3.1)^(1/3) = exp(−(D_v2/5 + D_S/27 + D_M/32)/3).

SURPRISE 3.1 and MATCH 3.1 keep every statistic of v3's judges and add:

- **F, foreign words.** `w_fit`, `w_fithi` and `w_anti` (SURPRISE 3.1); `m_bridge` (MATCH 3.1). Saturating like L and S: 60·(1 − exp(−max(0, F − θ_F)/2)), dead zone 1, θ_F the reference's 99th percentile. Insertion evidence must saturate: a linear F made SURPRISE rank random letters above random words.
- **B, bland.** The low sides of mean surprisal, its spread, the share of rare words, the spelling bits and the word fit, and the high sides of the share of very predictable words and of the context's information (SURPRISE 3.1; these sides were in v3's too-predictable group). Linear beyond the reference's 97.5th percentile: 10·max(0, B − θ_B), no dead zone.
- **Prompt statistics.** `w_pgain`, the bits a unigram cache of the prompt's words saves on the text's content words; `m_pcoh`, the prompt's strings in the text weighted by pack information. Two-sided: too much is echo or stuffing, too little is off the prompt (counted in C from zero, like all cohesion).
- **Copied prompt sentences** as 64-byte matches (`m_echo64`), replacing v3's 32-byte echo in MATCH 3.1. On development data 3 of 200 human continuations repeat a 32–48-byte name or phrase of their prompt, which v3's statistic charged with a deviance of several hundred; none has a 64-byte match, and every copied-prompt text keeps a share of at least 0.18.
- **L caps at 90** (S stays at 30, F at 60), so that non-words cost more than broken order and foreign words together.

The design was chosen on dev A and its battery only, and dev B was read once after the freeze. The trade-off is real and is recorded in `SCRIBE_V31.md`: on dev A, no setting of the new groups gave more than about 0.96 on both the first ladder pair and the worst attack, and the bland group is the most tuned part (at the 99th percentile instead of the 97.5th it would have missed the generic paragraph on dev A).

## 7. What measures what

Table 1 lists every statistic of the three judges. "High" and "low" say where each side of a statistic's z-score goes: a group (L lexical, S order, C cohesion or prompt, I integrity, P too predictable (v3's G and X), F foreign words, B bland), or "—" when that side is not counted. v2's statistics are one-sided except where marked ±. A text is charged only on the bad side and only beyond the group's dead zone or threshold (Sections 4–6).

**Table 1.** Every statistic of SCRIBE v2, SURPRISE, MATCH and the v3.1 additions.

| Statistic | Judge | What it measures | Maths | High side | Low side | The failure it catches |
|--------|-----|--------|------------------|----|----|---------|
| s1 lexical validity | v2 | real words | share of word tokens in the lexicon (seen ≥ 2 times) or numeric | — | L | random letters, non-words |
| s2g skeleton gain | v2 | word-class order | mean over tokens of `log p(c_t | c_t−2, c_t−1) − log p(c_t)`, 150 frequent words + 10 classes | — | S | shuffled words, salad |
| s3g context gain | v2 | word order | mean of `log p_KN(w_t | w_t−2, w_t−1) − log p(w_t)`, per-word SLOR | — | S | shuffled words, salad |
| s4 well-formedness | v2 | sentence form | pass rate of 6 checks (UTF-8, capitals, final punctuation, brackets, quotes, cut word) | — | S | broken sentences |
| c0 word cohesion | v2 | topic of the words | mean cosine of all pairs of content-word PPMI-SVD vectors | — | C | random words on many topics |
| c1 sentence cohesion | v2 | topic of the sentences | mean cosine of all pairs of SIF sentence vectors | — | C | sentences from other documents |
| c1a adjacent cohesion | v2 | local flow | mean cosine of consecutive sentence vectors | — | C | unrelated neighbours |
| c2 lexical cohesion | v2 | shared content words | mean over sentence pairs of shared IDF mass / smaller sentence's | — | C | sentences from other documents |
| c3 topic drift | v2 | drift | Jensen–Shannon divergence of the content words of the two halves | C | — | a change of topic |
| loop8 | v2 | loops | share of 8-byte windows seen earlier in the text | I | — | loops, repeated phrases |
| rep | v2 | repetition | share of sentences repeating an earlier one | I | — | repeated sentences |
| d1–d4 | v2 | diversity | distinct-1 to distinct-4 | I (±) | I (±) | degenerate or noisy word use |
| i1 verbatim copy | v2 | copying | share of tokens in a word 8-gram of the pack | I (±) | I (±) | copied training text |
| r1, r4, r5, i4 | v2p | relevance (prompted profile only) | prompt cosine; surprisal-weighted key-term coverage; new content; 4-gram echo | i4: R | r1, r4: R; r5 ± | off-prompt answers, echo |
| w_mean | SURPRISE | predictability | mean surprisal, bits per word | L | P (3.1: B) | salad (high); bland, copied text (low) |
| w_sd | SURPRISE | spread | standard deviation of surprisal | L | P (3.1: B) | salad; monotone text |
| w_ac1 | SURPRISE | burstiness | lag-1 autocorrelation of surprisal | P | S | formulaic runs; broken order |
| w_lo | SURPRISE | very predictable words | share of words below the reference words' 25th percentile (4.35 bits) | P (3.1: B) | S | bland, copied text; salad |
| w_hi | SURPRISE | very surprising words | share above the 90th percentile (17.45 bits) | L | P (3.1: B) | salad; text with no rare words |
| w_ctx | SURPRISE | what the context tells | unigram bits − trigram bits, per word | P (3.1: B) | S | formulaic text; shuffled words |
| w_spell | SURPRISE | non-words | bits spent spelling unknown words, per word | L | P (3.1: B) | random letters, typos |
| w_coh | SURPRISE | cohesion as code length | bits a sentence's content words save from a cache of the other sentences | P | C | repetition; sentences from other documents |
| w_anti | SURPRISE | words the context resists | share of words the left context makes > 2 bits less likely | — (3.1: F) | — | inserted words (3.1) |
| m_med | MATCH | typical match | median longest pack match from each byte | P | — | copied, pack-like text |
| m_p90 | MATCH | long matches | 90th percentile of the match lengths | P | S | copying; broken order |
| m_long | MATCH | copying | share of bytes in pack matches ≥ 32 bytes | P | — | copied training text |
| m_short | MATCH | novel junk | share of bytes only in matches ≤ 4 bytes | L | P | random letters; over-common text |
| m_lz | MATCH | self-compressibility | LZ76 phrases × log₂ n / n | C (weight 0.5, dead zone 2) | P | noise; loops |
| m_tri | MATCH | word order as strings | share of word starts whose match covers ≥ 3 whole words | P | S | copying; shuffled words |
| m_coh | MATCH | cohesion as shared strings | bytes matched with the other sentences, weighted by pack information above 0.45 | P | C | repetition; sentences from other documents |
| m_echo | MATCH | prompt echo | share of bytes in ≥ 8-byte matches with the prompt | P | C | echo; off-prompt text |
| m_echo32 | MATCH | copied prompt text | share in ≥ 32-byte matches with the prompt | P (3.1: —) | — | copied prompt sentences (v3) |
| w_fit | SURPRISE 3.1 | does the word belong | code length of the sentence with the word − without it − the word's unigram code length (−PMI with the two-sided context), mean | F | B | keyword stuffing; bland text |
| w_fithi | SURPRISE 3.1 | words that do not belong | share of words with fit above the reference's 90th percentile (1.18 bits) | F | P | keyword stuffing |
| w_pgain | SURPRISE 3.1 | use of the prompt's words | bits a unigram cache of the prompt saves on content words, per word | P | C | stuffing, echo; off-prompt text |
| m_bridge | MATCH 3.1 | inserted words as strings | share of words whose two pack bigrams are absent while the neighbours' bigram is present | F | — | keyword stuffing |
| m_pcoh | MATCH 3.1 | the prompt's strings | bytes matched with the prompt from word starts (≥ 4 bytes), weighted by pack information | P | C | echo; off-prompt text |
| m_echo64 | MATCH 3.1 | copied prompt sentences | share of bytes in full 64-byte matches with the prompt | P | — | copied prompt sentences, not names |
| w_lead | SURPRISE 3.1 | sentence order (probe) | first sentence's prompt gain − the others' | — | — | reported only; does not work (0.556) |

Read together, the three judges see three views of the same five questions:

| Question | v2 (counts and vectors) | SURPRISE (code length) | MATCH (strings) |
|------|------|------|------|
| Are these words? | s1 | w_spell, w_hi, w_mean | m_short |
| In a human word order? | s2g, s3g, s4 | w_ctx, w_lo, w_ac1 | m_tri, m_p90 |
| Do the sentences belong together? | c0–c3 (word vectors: co-occurrence) | w_coh (repeated content words) | m_coh, m_lz (repeated rare strings) |
| Is it degenerate or copied? | loop8, rep, d1–d4, i1 | the cache and match model; w_mean, w_lo too low | m_long, m_med, m_lz too low |
| Is it too predictable or bland? | not seen | the P side; B in 3.1 | the P side |
| Does every word belong? (3.1) | not seen | w_fit, w_fithi, w_anti | m_bridge |
| With a prompt: echo, relevance | v2p only (r1, r4, r5, i4) | w_pgain (3.1) | m_echo, m_echo64, m_pcoh |

Only the third row differs in kind between the judges: v2 measures cohesion through co-occurrence, the other two only through exact repetition. That is the point of the design (Section 3.1), and also why the new judges are weaker on "human > sentences shuffled" (Section 3.4).

## 8. Evaluation protocol

### 8.1 Regions of text and when each was read

**Table 2.** Wikipedia regions (MiB = 2²⁰ bytes; enwik8 is the first 10⁸ bytes of enwik9, the March 2006 English Wikipedia dump; Mahoney, 2011). Pack [16, 48) and reference [48, 56) of enwik8 throughout.

| Region | Role | Read by |
|----|----------|-------|
| enwik8 [8, 16) | training text of the n-gram generators | none of the judges |
| enwik8 [56, 64), [64, 72) | development ladders A and B and their attack batteries; A also the GEN-A prompts of K5 | design of v2, v3 and v3.1 |
| enwik8 [72, 95.37) | fresh for v2 (28 October), read again by v3's fresh verdict (4 November) | v2, v3 once each; re-used in this paper for reports only |
| enwik9 [100, 124) | fresh for v3.1, `--dedup` (2 of 2,106 documents dropped) | v3.1 once |
| enwik9 [128, 256) | packs of 8–128 MiB for the pack-size test | as packs only |
| enwik9 [900, 924) | **fresh for this paper**, `--dedup` (1 of 4,115 dropped) | v2, v3 and v3.1 once, 4 November |
| enwik9 [924, 953) | spare, kept for a later version | none |

The other genres are those of version 1.0 (its Table 2): WebText from the GPT-2 output dataset (Radford et al., 2019), 80 + 20 + 10 Project Gutenberg novels split by book, WritingPrompts (Fan et al., 2018) for HANNA and OpenMEVA-WP, OpenMEVA's ROC splits (Mostafazadeh et al., 2016), and Dolly 15k (Conover et al., 2023).

### 8.2 Registrations

Each version has a freeze record written before its fresh read, in the code archive:

- **v3** (`SCRIBE_V3.md`): rule 1, SURPRISE alone, MATCH alone and v3 each pass all five ladder pairs on dev A, dev B and fresh; rule 2, v3 ranks human text above each of 9 registered attacks at ≥ 0.95 and SURPRISE or MATCH catches each.
- **v3.1** (`SCRIBE_V31.md`, commit 28ff62c; held-out attacks in 5defdb5 before any design work): rule 1, v3.1 passes all five pairs on dev A, dev B and fresh; rule 2, v3.1 ≥ 0.95 on each of the 9 registered attacks; rule 3, the 8 held-out attacks reported with no rule; rule 4, the fresh human mean within 0.05 of v3's.
- **This paper** (`SCRIBE_V3_PAPER.md`, commit 3e3bc6e): thirteen reads (R1–R13) and twenty predictions (Q1–Q20), committed before any of them was run, with every command in `scripts/scribe_eval/runs/v3_paper.sh`. No setting was chosen on these reads.

### 8.3 Tests

1. **Control ladders** as in version 1.0: 500 human samples of 400–600 bytes, then sentences shuffled across documents, words shuffled within sentences, random lexicon words, random letters, one phrase repeated. A pair passes at accuracy ≥ 0.95 over all (upper, lower) pairs, ties one half (the AUROC), with disjoint 95% bootstrap intervals of the rung means.
2. **Attack batteries** (v3's, built by `scribe attacks`): 200 prompt / continuation pairs from the test region, one per document. Nine registered attacks (Table 6), three extras, and v3.1's eight held-out attacks. Accuracy is again over all (human, attack) pairs.
3. **v2's attack set** (`attacks.py`, version 1.0's Table 9) on 250 new prompts, with paired wins as in version 1.0.
4. **The re-runs of version 1.0's meta-evaluation**: web and books ladders, the GPT-2 output dataset (1,000 samples per system), nine held-out corruption families at six strengths on 200 samples, word n-gram generators of orders 1–5, the Dolly question-answering ladder, HANNA, the MAUVE human evaluation, OpenMEVA MANS, seeds, reference sizes, pack sizes, length and cost. The statistics are version 1.0's (`meta.py`, 1,000 bootstrap resamples).
5. **K5's saved samples**: every saved seed of the old K5 (7 MB and 330 MB of training text, and three sizes between) and of the upgraded K5 (7 MB and 330 MB), on the GEN-A development ladder.

Every score file is written by `scribe … --profile v31`, which also writes v2's and v3's fields, so the three metrics are always read on the same texts. Texts with a prompt are scored open (prompt removed) and with the prompt where that differs.

The neural baselines of version 1.0 (GPT-2 perplexity, CoLA, MiniLM coherence, the GPT-2 output detector, MAUVE) and HANNA's shipped metrics were not re-run. Where a table compares with them, their numbers are version 1.0's, on version 1.0's texts.

## 9. Results

### 9.1 Control ladders

**Table 3.** Wikipedia control ladders: pairs passed of 5 (first pair: human > sentences shuffled across documents). Bold: the registered fresh verdict of that version.

| Ladder | v2 | SURPRISE | MATCH | v3 | SURPRISE 3.1 | MATCH 3.1 | v3.1 |
|---|---|---|---|---|---|---|---|
| dev A (design) | 5 (0.974) | 5 (0.952) | 5 (0.953) | 5 (0.982) | 4 (0.915) | 3 (0.948) | 5 (0.959) |
| dev B (check) | 5 (0.983) | 5 (0.969) | 5 (0.955) | 5 (0.988) | 5 (0.961) | 4 (0.953) | 5 (0.982) |
| enwik8 [72, 95.37), v2's and v3's fresh | **5 (0.975)** | **5 (0.950)** | **4 (0.940)** | **5 (0.981)** | | | 5 (0.965) |
| enwik9 [100, 124), v3.1's fresh | 5 (0.976) | 4 (0.935) | 4 (0.940) | 5 (0.979) | 4 (0.924) | 3 (0.935) | **5 (0.968)** |
| enwik9 [900, 924), this paper's fresh | **5 (0.974)** | 5 (0.951) | 3 (0.941) | **5 (0.982)** | 3 (0.934) | 3 (0.935) | **5 (0.966)** |

The v3.1 row of enwik8 [72, 95.37) is a report read of a region whose v3 attack results v3.1 was designed against (Section 3.7).

Every combination passes every Wikipedia ladder it was read on. The new judges alone do not: MATCH fails v3's registered rule 1 (0.940), and v3.1's judges each fail one to two pairs on each fresh region, mostly the first pair and "words shuffled > random words". On this paper's fresh region, rung means under v3.1 are 0.908 (human), 0.551, 0.034, 0.018, 0.002 and 0.000, and 0.940 for short human samples. v3.1's first pair is the thinnest of the three combinations on every Wikipedia ladder (0.959–0.982 against v2's 0.974–0.983 and v3's 0.979–0.988).

**Table 4.** Control ladders of other genres (new draws, seed 20261104; each genre its own pack and reference; last row: web text against the Wikipedia pack and reference). Pairs passed, accuracies of the two hard pairs, and the human mean. Version 1.0's v2 results on its own draws in brackets.

| Ladder | v2 | v3 | v3.1 |
|---|---|---|---|
| WebText | 5; 0.959, 0.987; 0.918 (5; 0.966, 0.985; 0.923) | 5; 0.965, 0.971; 0.912 | **4**; 0.948, 0.979; 0.903 |
| Books | 3; 0.782, 0.882; 0.910 (3; 0.785, 0.894; 0.906) | 3; 0.837, 0.797; 0.902 | 3; 0.821, 0.803; 0.903 |
| WebText, Wikipedia reference | 4; 0.890, 0.998; 0.839 (4; 0.879, 0.997; 0.821) | 4; 0.909, 0.961; 0.855 | 4; 0.880, 0.987; 0.836 |

On web text v3.1 fails the first pair (0.948), the only ladder in this paper where a combination fails. Its human mean is lower than v2's: web text has lists, forms and boilerplate that the bland and foreign-word groups charge. On novels all three fail the same two pairs. The new judges separate human text from sentences shuffled across novels better than v2 (0.837 and 0.821 against 0.782), because sentences from other novels bring other names, but they separate shuffled words from random words worse (0.80 against 0.88). Against the wrong genre's reference, v3.1 keeps 0.926 of its own-genre human score, v2 0.914.

### 9.2 Attacks

**Table 5.** The nine registered attacks on three fresh regions: how many the metric ranks below human text at accuracy ≥ 0.95. (v3's own rule also needed SURPRISE or MATCH to catch each attack; under it v3 passed 5, 5 and 7.)

| Region | v2 | v3 | v3.1 | worst v3.1 attack |
|---|---|---|---|---|
| enwik8 [72, 95.37) (v3's fresh) | 5 | 5 | (not read as fresh) | |
| enwik9 [100, 124) (v3.1's fresh) | 5 | 6 | **9** | `stuff_human` 0.953 |
| enwik9 [900, 924) (this paper's fresh) | 5 | 7 | **9** | `generic`, `shuffle_docs` 0.970 |

**Table 6.** The battery on this paper's fresh region: mean score, with the accuracy P(human > attack) in brackets. The three extras and the eight held-out attacks have no rule.

| Attack | Construction | v2 | v3 | v3.1 |
|------|----------------|----|----|----|
| human | the true continuation | 0.900 | 0.888 | 0.845 |
| `copy_pack` | a 400–600 B window of pack text | 0.105 (1.000) | 0.050 (1.000) | 0.000 (1.000) |
| `echo_repeat` | the prompt repeated | 0.000 (1.000) | 0.002 (1.000) | 0.000 (1.000) |
| `echo_copy` | the prompt's sentences, then the true continuation | 0.883 (0.506) | 0.599 (0.966) | 0.000 (1.000) |
| `stuff_human` | another document's continuation, the prompt's 5 rarest content words after every 4th word | 0.408 (0.841) | 0.532 (0.880) | 0.254 (0.975) |
| `stuff_generic` | the same insertion into v2's generic paragraph | 0.969 (0.336) | 0.877 (0.560) | 0.377 (0.976) |
| `loop` | the continuation's first two sentences repeated | 0.000 (1.000) | 0.007 (1.000) | 0.000 (1.000) |
| `generic` | v2's fixed generic paragraph | 0.835 (0.760) | 0.654 (0.955) | 0.462 (0.970) |
| `shuffle_docs` | each sentence from another document | 0.357 (0.979) | 0.517 (0.987) | 0.430 (0.970) |
| `cooc_pack` | pack sentences with the most PPMI-SVD association with the prompt (a crude K5 gate) | 0.130 (0.992) | 0.079 (1.000) | 0.000 (1.000) |
| `reorder` (extra) | the continuation's own sentences reordered | 0.891 (0.511) | 0.885 (0.508) | 0.841 (0.509) |
| `cooc_out` (extra) | the co-occurrence attack from text outside the pack | 0.886 (0.447) | 0.832 (0.587) | 0.792 (0.569) |
| `offprompt` (extra) | another document's continuation | 0.900 (0.500) | 0.846 (0.652) | 0.700 (0.821) |
| `ho_stuff7_rare` | 5 rarest words after every 7th word | 0.803 (0.559) | 0.855 (0.470) | 0.704 (0.643) |
| `ho_stuff3_freq` | 5 most frequent content words after every 3rd word | 0.093 (0.988) | 0.352 (0.990) | 0.125 (**0.995**) |
| `ho_stuff_rand` | 5 rarest words at random gaps of 2–8 words | 0.619 (0.699) | 0.711 (0.705) | 0.454 (0.878) |
| `ho_generic2` | one of four other generic paragraphs | 0.838 (0.664) | 0.636 (0.897) | 0.453 (**0.953**) |
| `ho_template` | a five-sentence template with the prompt's names and words swapped in | 1.000 (0.265) | 0.919 (0.467) | 0.846 (0.529) |
| `ho_echo_scatter` | the prompt's sentences scattered between the continuation's | 0.374 (0.794) | 0.375 (0.986) | 0.000 (**1.000**) |
| `ho_title_swap` | another document's continuation with its title entity replaced by the prompt's | 0.900 (0.510) | 0.867 (0.588) | 0.788 (0.650) |
| `ho_list_related` | other documents' sentences that share the most rare words with the prompt | 0.633 (0.814) | 0.684 (0.882) | 0.665 (0.827) |

On both of v3.1's fresh reads the held-out attacks pass 2 of 8 and 3 of 8 under v3.1 (v3: 2 and 2; v2: 1 and 1). The two reads agree on which fail: stuffing below one word in five, the template, the title swap and loosely related sentences. One generic paragraph of the four in `ho_generic2` is not caught (version 1.0's own generic paragraph is).

What v3.1 adds over v2 on these attacks is concrete:

- **Copied prompt sentences** go to zero (`echo_copy`, `ho_echo_scatter`: v2 0.506 and 0.794, v3.1 1.000).
- **Keyword stuffing** at one word in 3 to 5 is caught (0.975–0.995), where v2 catches it only when the inserted words also break the grammar (0.841, 0.336).
- **Bland, contentless paragraphs** are caught (v2 0.760, v3.1 0.970), and three of four new ones.
- **Off-prompt fluent text** is half-seen (v2 0.500, v3.1 0.821), through MATCH 3.1's prompt statistics.

What none of the three sees: sentence order (0.509), fluent sentences from elsewhere with some shared words (`cooc_out` 0.569, `ho_list_related` 0.827, `ho_title_swap` 0.650), and a template that repeats the prompt's names (0.529; v2 rates it 1.000).

### 9.3 Held-out corruptions

**Table 7.** Dose–response on 200 new samples of enwik8 [72, 95.37) MiB (10,800 texts): means at strengths 0, 25, 50 and 100%, and Spearman between strength and score over the six strength means (texts in brackets). A family is followed at ≤ −0.9 over the means.

| Family | v2 | v3 | v3.1 |
|---|---|---|---|
| word substitution | 0.932, 0.133, 0.004, 0.001; −1.000 (−0.944) | 0.921, 0.509, 0.199, 0.137; −1.000 (−0.929) | 0.917, 0.205, 0.036, 0.019; −1.000 (−0.946) |
| typos | 0.932, 0.423, 0.021, 0.000; −1.000 (−0.941) | 0.921, 0.683, 0.226, 0.101; −1.000 (−0.917) | 0.917, 0.585, 0.061, 0.005; −1.000 (−0.930) |
| sentence repetition | 0.932, 0.321, 0.005, 0.000; −1.000 (−0.921) | 0.921, 0.553, 0.101, 0.003; −1.000 (−0.906) | 0.917, 0.359, 0.008, 0.000; −1.000 (−0.913) |
| adjacent word swaps | 0.932, 0.079, 0.005, 0.003; −1.000 (−0.899) | 0.921, 0.460, 0.231, 0.190; −1.000 (−0.876) | 0.917, 0.131, 0.040, 0.032; −1.000 (−0.888) |
| dropped words | 0.932, 0.486, 0.210, 0.011; −1.000 (−0.861) | 0.921, 0.720, 0.631, 0.154; −1.000 (−0.859) | 0.917, 0.554, 0.349, 0.039; −1.000 (−0.872) |
| content-word swaps | 0.932, 0.817, 0.525, 0.079; −1.000 (−0.826) | 0.921, 0.851, 0.667, 0.285; −1.000 (−0.823) | 0.917, 0.802, 0.508, 0.104; −1.000 (−0.854) |
| off-topic sentences | 0.932, 0.737, 0.496, 0.365; −0.943 (−0.788) | 0.921, 0.806, 0.660, 0.571; −0.943 (−0.773) | 0.917, 0.801, 0.655, 0.558; −0.943 (−0.762) |
| truncated sentences | 0.932, 0.908, 0.881, 0.916; −0.657 (−0.103) | 0.921, 0.900, 0.881, 0.938; −0.143 (−0.016) | 0.917, 0.895, 0.876, 0.936; −0.143 (−0.019) |
| sentence reordering | 0.932, 0.929, 0.935, 0.930; −0.348 (−0.012) | 0.921, 0.919, 0.922, 0.919; +0.086 (−0.008) | 0.917, 0.914, 0.918, 0.914; −0.486 (−0.011) |
| **followed** | **7 of 9** | **7 of 9** | **7 of 9** |

All three follow the same seven families and miss the same two. v2 follows them with these new draws as it did with version 1.0's (7 of 9). v3.1's correlations over texts are within 0.03 of v2's on every followed family: stronger on three (substitution, dropped words, content-word swaps), weaker on four. v3, the plain mean, falls less steeply: SURPRISE and MATCH keep a graded score of about 0.2–0.3 for word salad, so v3 at full damage is far above v2 and v3.1. Off-topic sentences level off higher under v3 and v3.1 (0.56) than under v2 (0.37): fluent sentences from other documents are, to the new judges, mostly fluent sentences.

### 9.4 Model ladders

**The GPT-2 output dataset and v2's "too typical" blind spot.** Table 8 is the clearest single improvement of v3.

**Table 8.** The GPT-2 output dataset (new draw: 1,000 samples per system, cut as human samples are, WebText pack and reference). Means; Spearman with the expected order over the five systems of each decoding (human, then larger first); and P(WebText > the system) over all text pairs.

| System | v2 | v3 | v3.1 | SURPRISE | SURPRISE 3.1 | v2: P(WebText >) | v3.1: P(WebText >) |
|---|---|---|---|---|---|---|---|
| WebText (human) | 0.917 | 0.912 | 0.907 | 0.914 | 0.926 | | |
| xl-1542M, pure | 0.836 | 0.845 | 0.842 | 0.855 | 0.872 | 0.639 | 0.664 |
| large-762M, pure | 0.815 | 0.827 | 0.824 | 0.834 | 0.858 | 0.648 | 0.686 |
| medium-345M, pure | 0.715 | 0.753 | 0.740 | 0.778 | 0.787 | 0.747 | 0.781 |
| small-117M, pure | 0.753 | 0.775 | 0.771 | 0.795 | 0.815 | 0.733 | 0.779 |
| xl-1542M, top-k 40 | **0.971** | 0.903 | 0.886 | 0.847 | 0.841 | **0.376** | 0.511 |
| large-762M, top-k 40 | **0.964** | 0.884 | 0.857 | 0.815 | 0.799 | **0.384** | 0.567 |
| medium-345M, top-k 40 | **0.967** | 0.884 | 0.872 | 0.814 | 0.823 | **0.391** | 0.567 |
| small-117M, top-k 40 | **0.975** | 0.841 | 0.800 | 0.720 | 0.701 | **0.359** | 0.648 |
| Spearman, pure sampling | 0.900 | 0.900 | 0.900 | 0.900 | 0.900 | | |
| Spearman, top-k 40 | **−0.700** | **0.900** | **0.900** | **1.000** | 0.900 | | |

Version 1.0's draw gave v2 the same pattern (top-k 0.968–0.972 above WebText's 0.925; Spearman 0.900 and −0.800). Under v3 and v3.1, WebText is first under both decodings, every top-k system scores below it, and the four sizes are nearly in order (medium and large swap under v3.1). SURPRISE alone orders all five top-k systems exactly. The cause is the two-sided reading: top-k samples are more predictable than human text under the pack's trigram (fewer rare words, too even a surprisal), and SURPRISE and the bland group charge exactly that. This settles, for this test, the question version 1.0 left open (its Section 7): penalising excess typicality puts human text first without breaking any control ladder.

The per-text margin is small. A single top-k text is ranked below a single WebText text only 51–65% of the time under v3.1. The improvement is in the order of the system means, which is what a model comparison uses.

Under pure sampling nothing changes: all three order XL > large > small > medium, with medium below small as in version 1.0. GPT-2 medium's pure samples are noisier than small's under every count statistic.

**n-gram generators.** Word n-grams of orders 1–5 trained on enwik8 [8, 16) MiB continue 250 new prompts.

**Table 9.** n-gram generators (about 250 continuations each, scored open). Means, and the accuracy against the next rung down.

| Generator | v2 | v3 | v3.1 | SURPRISE 3.1 |
|---|---|---|---|---|
| human continuation | 0.906 (0.991) | 0.900 (0.979) | 0.892 (0.975) | 0.916 (0.941) |
| 5-gram | 0.085 (0.480) | 0.456 (0.487) | 0.267 (0.489) | 0.587 (0.498) |
| 4-gram | 0.100 (0.505) | 0.463 (0.501) | 0.281 (0.503) | 0.571 (0.513) |
| 3-gram | 0.100 (0.726) | 0.470 (0.672) | 0.279 (0.732) | 0.580 (0.667) |
| 2-gram | 0.034 (0.982) | 0.407 (0.978) | 0.159 (0.991) | 0.407 (0.996) |
| 1-gram | 0.001 | 0.148 | 0.019 | 0.030 |
| Spearman over six systems | 0.829 | 0.771 | 0.829 | 0.943 |

Every generator is far below human text under all three, and orders 3, 4 and 5 are tied under all three. v2's blind spot "models that all write salad" is not fixed: the saturating form groups that tie them are in all three judges. (Version 1.0's draw gave v2 0.943; on this draw the 5-gram scores a little below the 4-gram under every metric.) Under v3 the n-gram texts keep 0.41–0.47 (v3's shuffled-sentence rung is 0.560): the plain mean compresses the scale, which makes it a poor gate.

### 9.5 Question answering

**Table 10.** Dolly (600 questions, four answers each), scored with the question as the prompt. Accuracy of the correct answer against each wrong one.

| Metric | correct > another question's answer | > question echoed | > words shuffled |
|---|---|---|---|
| v2p (prompted v2, version 1.0's best) | **0.961** | **0.957** | 0.864 |
| v2 (open) | 0.521 | 0.786 | 0.939 |
| v3 | 0.470 | 0.926 | 0.907 |
| v3.1 | 0.536 | **0.952** | 0.752 |
| SURPRISE 3.1 | 0.679 | 0.894 | 0.933 |

v3.1 catches the echoed question but cannot tell the right answer from a fluent answer to another question. The reason is a design limit, not a defect: the prompt statistics of v3 and v3.1 (MATCH's echo, `w_pgain`, `m_pcoh`) are read against prompt/continuation pairs that the tool cuts from inside the reference documents, while `--ref-prompts`, the genre's real prompt–answer pairs, feeds only v2p. For Wikipedia continuations the two are the same kind of pair. For an instruction and its answer they are not, and Dolly's short answers yield only 299 such pairs. Word-shuffled answers also slip under v3.1 (0.752): many Dolly answers are lists, whose order MATCH 3.1 barely sees (0.539 alone). **For question answering, v2p stays the tool.**

### 9.6 Version 1.0's attack set

**Table 11.** `attacks.py` on 250 new prompts of enwik8 [72, 95.37) MiB, scored with the prompt. Mean, and the share of prompts where the true continuation beats the attack (ties one half), as version 1.0's Table 9.

| Attack | v2 | v3 | v3.1 |
|---|---|---|---|
| true continuation | 0.906 | 0.873 | 0.831 |
| copied pack text | 0.103 (0.992) | 0.048 (1.000) | 0.000 (1.000) |
| another document's continuation | 0.906 (0.484) | 0.834 (0.636) | 0.688 (0.808) |
| prompt echoed | 0.000 (1.000) | 0.002 (1.000) | 0.000 (1.000) |
| keyword stuffing (every 3rd word replaced) | 0.015 (1.000) | 0.103 (0.992) | 0.020 (0.992) |
| generic paragraph | 0.835 (0.792) | 0.656 (0.920) | 0.464 (**0.964**) |
| generic paragraph with the prompt's topic words | 0.911 (0.704) | 0.953 (0.320) | 0.947 (**0.288**) |
| randomised numbers | 0.906 (0.498) | 0.872 (0.494) | 0.830 (0.516) |
| inserted negations | 0.908 (0.466) | 0.882 (0.420) | 0.842 (0.450) |
| function-word loop | 0.000 (1.000) | 0.030 (1.000) | 0.003 (1.000) |

v3.1 catches the plain generic paragraph that v2 misses, and is fooled worse than v2 by the same paragraph with the prompt's rare words put in as its topic: v3.1 scores it 0.947, above the true continuations (0.831). It is the same weakness as the held-out template (Table 6): the prompt statistics charge a shortfall of the prompt's words at every level but an excess only far out, so a text that repeats the prompt's words looks more on-topic than a human continuation. Wrong numbers and inserted negations are invisible to all three: none has a notion of truth.

### 9.7 Agreement with people

**Table 12.** HANNA (960 machine-written stories from ten systems; human stories excluded). Story-level Kendall τ with the mean human rating of each criterion; system-level Kendall for coherence. The v2, v3 and v3.1 rows are re-run here (story-pack vectors rebuilt; v2 differs from version 1.0 by at most 0.004); the other rows are version 1.0's Table 11.

| Metric | RE | CH | EM | SU | EG | CX | CH, system |
|---|---|---|---|---|---|---|---|
| SCRIBE v2 (re-run) | 0.017 | 0.193 | 0.127 | 0.144 | 0.170 | 0.227 | 0.689 |
| SCRIBE v3 | 0.112 | **0.213** | 0.128 | 0.195 | 0.216 | 0.300 | **0.778** |
| SCRIBE v3.1 | 0.109 | 0.209 | 0.127 | 0.192 | 0.215 | 0.294 | 0.689 |
| SCRIBE v3.1, with the prompt | **0.169** | **0.233** | 0.136 | 0.204 | 0.237 | 0.311 | 0.600 |
| MATCH 3.1, with the prompt | 0.191 | 0.236 | 0.143 | 0.205 | 0.253 | 0.306 | 0.689 |
| min of 3 | 0.125 | 0.222 | 0.130 | 0.208 | 0.232 | 0.319 | 0.600 |
| SCRIBE v1 (version 1.0) | 0.074 | 0.232 | 0.183 | 0.209 | 0.262 | 0.322 | 0.600 |
| rep-4 | 0.044 | 0.194 | 0.098 | 0.185 | 0.199 | 0.264 | 0.378 |
| BLEU | 0.074 | 0.110 | 0.133 | 0.106 | 0.139 | 0.189 | 0.333 |
| BERTScore F1 | 0.132 | 0.139 | 0.169 | 0.123 | 0.192 | 0.241 | 0.556 |
| ChatGPT | 0.156 | 0.202 | 0.147 | 0.111 | 0.181 | 0.171 | 0.600 |
| Beluga-13B | 0.203 | 0.282 | 0.259 | 0.216 | 0.290 | 0.304 | 0.733 |

Open, v3 and v3.1 correlate a little more with coherence than v2 (0.213 and 0.209 against 0.193; not significant, Benjamini–Hochberg p 0.42 and 0.52), and more with every criterion but empathy. With the prompt, v3.1 reaches 0.233 on coherence (p 0.14). The largest change is relevance: v2 cannot see it (0.017); v3.1 reaches 0.109 open and 0.169 with the prompt, between BERTScore F1 (0.132) and Beluga-13B (0.203) and above ChatGPT (0.156), without a reference story. MATCH 3.1 alone, with the prompt, reaches 0.191. The prompt statistics of v3.1 work on stories, where a prompt and its story are related as a Wikipedia passage and its continuation are, unlike Dolly's instructions and answers (Section 9.5). At system level v3 has the highest coherence correlation in the table (0.778, above the best LLM judges' 0.733), though with ten systems the intervals are wide; v3.1 open equals v2 (0.689), and with the prompt it is lower (0.600).

**Table 13.** The MAUVE human evaluation (pairwise judgments of WebText continuations; q1 more interesting, q2 makes more sense, q3 more likely written by a person; the 1,666–1,695 judgments over 25 seconds). Per-pair agreement, and system Spearman against Bradley–Terry scores over the eight GPT-2 configurations.

| Metric | q1 pair | q2 pair | q3 pair | q1 system | q2 system | q3 system |
|---|---|---|---|---|---|---|
| SCRIBE v2 (re-run) | 0.543 | 0.538 | 0.541 | 0.762 | 0.786 | 0.881 |
| SCRIBE v3 | 0.550 | 0.547 | **0.559** | 0.762 | 0.786 | 0.881 |
| SCRIBE v3.1 | 0.547 | 0.547 | 0.556 | 0.690 | 0.762 | 0.833 |
| SURPRISE | 0.529 | 0.535 | 0.552 | **0.786** | **0.905** | **0.976** |
| SCRIBE v2 (version 1.0) | 0.547 | 0.534 | 0.541 | 0.714 | 0.762 | 0.857 |
| MAUVE, GPT-2 large features (published) | | | | 0.810 | 0.857 | 0.952 |

Per pair, v3 and v3.1 agree with people a little more often than v2 (by 0.004–0.018 open), still close to chance. With the prompt, v3.1 agrees more (0.559, 0.558 and 0.575) and its system correlations equal v2's (0.762, 0.786, 0.881). Open, at system level v3 equals v2 and v3.1 is a little lower; SURPRISE alone tracks all three questions best, above published MAUVE on q2 and q3. With eight systems the intervals are wide (version 1.0: q3 0.21 to 0.88), so none of these system differences is clear.

**OpenMEVA MANS** (1,000 machine stories each, overall quality, story-level Pearson): ROC v2 0.189, v3 0.186, v3.1 0.161; WritingPrompts v2 0.050, v3 0.089, v3.1 0.030 open and 0.113 with the prompt. GPT-2 perplexity reached 0.207 and 0.301 in version 1.0. The new judges do not change version 1.0's conclusion: on these stories, single-text count metrics agree with overall quality ratings only weakly. On ROC stories the prompt changes nothing, because five-sentence stories yield no prompt pairs for v3's reference.

### 9.8 Stability

- **Seeds.** Five more draws of the enwik8 fresh ladder: v2, v3 and v3.1 pass 5 of 5 on every seed. v3.1's first pair ranges from 0.956 to 0.983 (v2 0.969–0.990, as in version 1.0).
- **Reference size** (three draws each). v3.1 needs more reference samples than v2. Pairs passed for v2, v3, v3.1: at 50 samples per bucket 4/4/4, 4/4/5 and 4/4/3; at 100, 5/5/5, 5/5/5 and 5/4/4; at 200, 5/5/5, 5/5/5 and 5/5/4; at 500 and 1,000 every draw of every metric passes. v3.1 counts more statistics beyond 97.5th and 99th percentiles in more groups, and with 100–200 samples those percentiles rest on a handful of texts. **Use at least 500 reference samples per bucket with v3.1.**
- **Pack size.** Packs of 8, 16, 32, 64 and 128 MiB of enwik9, each with its own word vectors and suffix array: all three pass 5 of 5 at every size (v3.1's first pair 0.951–0.961; v2's 0.971–0.983, the same as version 1.0).
- **Length.** On held-out human text, Pearson's correlation of v3.1 with length in bytes is 0.033 (standard bucket) and −0.038 (short); pooled −0.136, because short texts score higher on average (0.951 against 0.902). v2: 0.009, 0.041, −0.124.

### 9.9 Reproduction of version 1.0

Every run on an enwik range with a recorded seed reproduces version 1.0's v2 numbers exactly: the fresh ladder (0.975, 0.999, 0.998, 0.998, 1.000), five seeds, the reference-size draws and five pack sizes. Profiles v1, v2 and v3 of the released tool reproduce their earlier outputs byte for byte (checked with `cmp` when v3.1 was added; no code changed since). The sets of other genres give v2 numbers within 0.004 of version 1.0's (HANNA, Dolly), because their word vectors were rebuilt. Six test sets of version 1.0 had no recorded seed and were drawn again (Section 3.7); on the new draws v2 reproduces version 1.0's pattern in every case, with differences of up to 0.02 in means.

### 9.10 The K5 cross-checks

**Table 14.** K5's saved development samples (GEN-A ladder, 250 prompts; means over the 4 saved seeds of each arm). The old K5 and the upgraded K5 (nucleus 0.8, 256 word classes, the 8-candidate co-occurrence gate), each on 7 MB and on 330 MB of clean Wikipedia.

| Arm | v2 | SURPRISE | MATCH | v3 | SURPRISE 3.1 | MATCH 3.1 | v3.1 |
|---|---|---|---|---|---|---|---|
| old K5, 7 MB | 0.344 | 0.709 | 0.581 | 0.544 | 0.638 | 0.446 | 0.429 |
| old K5, 330 MB | 0.390 | 0.675 | 0.569 | 0.545 | 0.588 | 0.438 | 0.449 |
| upgraded K5, 7 MB | 0.500 | 0.595 | 0.578 | 0.558 | 0.536 | 0.461 | 0.476 |
| upgraded K5, 330 MB | 0.479 | 0.596 | 0.565 | 0.547 | 0.483 | 0.454 | 0.448 |
| **the gate's gain** (old → upgraded, 7 MB) | **+0.156** | −0.114 | −0.003 | +0.014 | −0.102 | +0.015 | +0.047 |
| **scale, old K5** (7 → 330 MB) | **+0.046** | −0.034 | −0.012 | +0.001 | −0.050 | −0.008 | +0.020 |
| **scale, upgraded K5** (7 → 330 MB) | −0.021 | +0.001 | −0.013 | −0.011 | −0.053 | −0.007 | −0.028 |
| human text, same ladder | 0.923 | 0.921 | 0.922 | 0.922 | 0.930 | 0.924 | 0.919 |
| sentences shuffled, same ladder | 0.344 | 0.665 | 0.661 | 0.557 | 0.736 | 0.677 | 0.546 |

(Single seeds of the published cross-check: gate v2 +0.145, v3 +0.011, SURPRISE −0.106, v3.1 +0.048, all reproduced here.)

Neither judge without co-occurrence confirms either gain. SURPRISE and SURPRISE 3.1 fall in both cases: the gated text is more predictable than human text (7.4 bits a word against human text's 9.5) and the 330 MB text more pack-like. MATCH does not move. v3.1 shows about a third of v2's gate gain and less than half of its scale gain, and all of it comes from v2: in a geometric mean, v2's large relative change (0.344 → 0.500 is +45%) counts more than in a plain mean. All K5 versions stay far below human text and below shuffled human sentences on v3.1. Cohesion measured without co-occurrence is far below human text for every K5 version (SURPRISE's `w_coh` −0.10 and −0.07 bits a word for old and gated K5 against +1.09 for human text, `SCRIBE_V3.md`).

So the circularity of Section 3.1 is real in this sense: the two gains rest on v2's co-occurrence-based axis alone. That does not prove they are not real. It means counts of other kinds do not confirm them, and a judge of a third kind (a person) is needed.

### 9.11 Which combination

**Table 15.** The combination ablation on this paper's fresh region: ladder pairs, registered attacks and held-out attacks with accuracy ≥ 0.95, and the weakest value of each.

| Combination | Ladder | weakest pair | Registered attacks | weakest attack | Held-out |
|----------|---|---|---|-----|---|
| **v3.1: geometric mean of v2, SURPRISE 3.1, MATCH 3.1** | **5** | 0.966 | **9** | 0.970 | 3 |
| without v2 | 3 | 0.903 | 9 | 0.958 | 3 |
| without SURPRISE 3.1 | 5 | 0.973 | 6 | 0.586 (`stuff_generic`) | 2 |
| without MATCH 3.1 | 5 | 0.966 | 8 | 0.546 (`echo_copy`) | 2 |
| plain mean of the three v3.1 judges | 4 | 0.860 | 7 | 0.935 (`stuff_generic`) | 3 |
| minimum of the three v3.1 judges | 5 | 0.959 | 9 | 0.956 (`shuffle_docs`) | 2 |
| v3: plain mean of v2, SURPRISE, MATCH | 5 | 0.976 | 7 | 0.560 (`stuff_generic`) | 2 |

Each judge carries something. Without v2 the ladder breaks (only v2 sees co-occurrence cohesion, and its form axes order word salad); without SURPRISE 3.1 stuffing and the generic paragraph get through; without MATCH 3.1 copied prompt sentences do. The plain mean dilutes. The minimum performs about as well as the geometric mean here; the geometric mean was chosen on development data (`SCRIBE_V31.md`, arms 2, 3 and 17), and this paper does not change it.

### 9.12 Registered predictions

**Table 16.** The twenty predictions of this paper, registered before its reads (`SCRIBE_V3_PAPER.md`), scored.

| | Prediction (short form) | Result | Verdict |
|--|----------------|----------|---|
| Q1 | new fresh ladder: v2, v3, v3.1 pass 5 of 5; v3.1's first pair thinnest, 0.95–0.98; MATCH 3.1 alone fails a pair | 5, 5, 5; 0.966; MATCH 3.1 fails 2 | hit |
| Q2 | v3.1 ≥ 8 of 9 registered attacks; v3's rule 4–6 of 9 | 9; v3's rule 7 | partly |
| Q3 | v3.1 1–3 of 8 held-out; named five fail, echo scatter passes | 3; as named | hit |
| Q4 | v3.1 human mean in [0.854, 0.954] | 0.908 | hit |
| Q5 | ablation: no v2 fails a pair; no MATCH 3.1 loses echo attacks; plain mean passes fewer attacks | 2 pairs; 0.546 and fail; 7 of 9 | hit |
| Q6 | web: v2 5, v3 5, v3.1 4–5 (first pair weakest); v3.1 human 0.85–0.92, below v2 | 5, 5, 4; 0.903 < 0.918 | hit |
| Q7 | books: v2's first pair < 0.9; v3, v3.1 better on it; one of them passes 4 of 5 | 0.782; 0.837, 0.821; both 3 of 5 | partly |
| Q8 | web against the Wikipedia reference: v3.1 keeps less of its score than v2 | 0.926 against 0.914 | miss |
| Q9 | GPT-2 top-k: v2 rates every top-k system above WebText; SURPRISE, SURPRISE 3.1 and v3.1 put WebText first; v3.1 Spearman ≥ 0.6; v3 WebText above ≥ 3 of 4 | all as predicted; +0.900; 4 of 4 | hit |
| Q10 | GPT-2 pure: v3.1 Spearman ≥ 0.8 | 0.900 | hit |
| Q11 | corruptions: v3, v3.1 follow 7–8 of 9; reordering by none; v3.1 falls more than v2 with truncation, not to −0.9 | 7, 7; none; less (−0.143 against −0.657) | partly |
| Q12 | HANNA: v3.1 coherence 0.17–0.26, one of v3, v3.1 above v2; system ≥ 0.5; relevance with the prompt above open and above 0.05 | 0.209, yes; 0.689; 0.169 > 0.109 | hit |
| Q13 | MAUVE: v3.1 per pair 0.51–0.57; q3 system ≥ 0.6; not above v2 on ≥ 2 of 3 | open: 0.547–0.556; 0.833; below on 3 (with the prompt q3 per pair 0.575, just above) | hit |
| Q14 | OpenMEVA: v3.1 ROC 0.08–0.25, WP < 0.15 | 0.161, 0.030 | hit |
| Q15 | n-grams: human > 5-gram ≥ 0.95 under v3.1; v3 orders 3–5 better than v2; v3 Spearman ≥ 0.9 | 0.975; no; 0.771 | miss |
| Q16 | Dolly: v3.1 correct > echo and > shuffled ≥ 0.95, > other 0.80–0.95 | 0.952, 0.752, 0.536 | miss |
| Q17 | v2's attacks: v3.1 catches copy, echo, stuffing, loop, generic; fails generic + topic; numbers and negations 0.45–0.55 for all | as predicted; v3's negation 0.420 | partly |
| Q18 | stability: v3.1 5 of 5 on ≥ 4 of 5 seeds; every draw passes at 500 and 1,000 and fails at 50; packs 16–128 pass; \|length Pearson\| ≤ 0.15 per bucket | 5 of 5 seeds; yes; yes; 0.033, 0.038 | hit |
| Q19 | K5: the gate figures reproduce; old K5's scale gain under v3.1 in [0, 0.03]; upgraded K5 in [−0.03, 0] | yes; +0.020; −0.028 | hit |
| Q20 | cost: v3.1 350–550 texts/s, < 600 MB, 2–4 times slower than v2; web 200–450/s, < 1 GB | 587/s, 489 MB, 3.0 times; 400/s, 768 MB | partly |

Twelve predictions hit, five hit in part and three missed. The registered rules of v3 and v3.1, scored in their own reports, are in Table 3 and Table 5: v3's rule 1 failed for MATCH and its rule 2 passed 5 of 9; v3.1's four rules all held. The misses say: books are hard for every judge (Q7); v3.1 depends on genre less than feared (Q8); no judge ranks models that all write salad (Q15); and v3.1's prompt statistics do not fit question answering (Q16).

### 9.13 Cost

**Table 17.** Time and peak memory of `scribe score`, one core of an idle 4-core Xeon (2.8 GHz), no GPU. Wikipedia pack 17.5 MB of prose; WebText pack 32.2 MB.

| Profile | texts/s, Wikipedia | peak memory | texts/s, WebText | peak memory |
|---|---|---|---|---|
| v2 | 1,749 | 385 MB | 1,213 | 649 MB |
| v3 | 730 | 481 MB | 464 | 749 MB |
| v3.1 | 587 | 489 MB | 400 | 768 MB |

v3.1's own statistics cost 0.96 ms a text with the Wikipedia pack (1.50 ms with WebText's). Setup adds the suffix array (0.9 s for the Wikipedia pack on 4 threads, 2.0 s for the 35.7 MB story pack, about 5 bytes of memory per pack byte) and the spelling models (0.4–0.8 s), and for a large reference the v3 features of every reference sample (27 s for WritingPrompts' 89,000). In version 1.0, on a shared machine, v2 ran at 1,054 texts a second, GPT-2 small perplexity at 4.7 and MAUVE featurisation at 10. v3.1 remains about a hundred times faster than GPT-2 perplexity on the same CPU.

## 10. Why v3.1 is better, and where it is not

**Better, by measurement.**

1. **It passes what v2 passes.** Every Wikipedia ladder (5 of 5 on three fresh regions, five seeds and five pack sizes), the same 7 of 9 held-out corruption families, and the same order under GPT-2 pure sampling. v2 is one of its judges, so v3.1 keeps v2's strengths on form and on co-occurrence cohesion.
2. **It catches attacks v2 does not.** 9 of 9 registered attacks on two fresh regions, against v2's 5 of 9: copied prompt sentences, keyword stuffing at one word in five or more, and bland generic paragraphs.
3. **It fixes v2's "too typical" blind spot in the mean.** WebText is ranked above every GPT-2 top-k system (Spearman +0.900 against v2's −0.700), without a ladder failure.
4. **It sees relevance to a story prompt.** HANNA relevance 0.169 against v2's 0.017.
5. **It gives a second and third opinion that do not share v2's co-occurrence basis**, which is what showed that K5's gate and scale gains rest on v2 alone.
6. **It is still cheap**: about 590 texts a second on one core, 0.5 GB.

**Not better, said plainly.**

1. **It is not fool-proof.** On two fresh reads it passes 2 and 3 of 8 held-out attacks. Keyword stuffing at one word in 7 or 8, or at random gaps, gets through (0.64–0.88). A template that repeats the prompt's names gets through (0.53), and the generic paragraph with the prompt's topic words scores above human text (0.288 wins). A title swap and loosely related fluent sentences get through (0.65, 0.83).
2. **Sentence order is invisible** to all three judges (reordering 0.51; Spearman of the dose means −0.49 at best).
3. **Truth is not checked.** Random numbers and inserted negations keep human parity.
4. **Truncated sentences** are not seen, by any judge.
5. **Models that all write salad** are not ranked (n-gram orders 3–5 tie).
6. **Its margins are thin.** Its first ladder pair is the weakest of the three combinations on every Wikipedia ladder (0.966–0.968 on the fresh regions), it fails that pair on web text (0.948), and it needs at least 500 reference samples per bucket.
7. **Books stay at 3 of 5** for all three.
8. **Its prompt statistics fit continuations, not answers.** On Dolly it cannot tell the right answer from another question's (0.536); v2p can (0.961).
9. **Its judges are weak alone.** SURPRISE 3.1 and MATCH 3.1 each fail one or two ladder pairs on every fresh region. v3.1 is one score.
10. **It agrees with people about as well as v2.** Per-pair agreement on the MAUVE judgments rises by under 0.02, and story-level coherence correlations by under 0.02 on HANNA; neither difference is significant.

**How to use it.** v3.1 as the gate that a writer must pass, with v2, `min3` and the two judges' deviance terms as the diagnosis of how it fails; v2p for question answering; and a person, or a judge of yet another kind, for any gain that only one judge sees.

## 11. Limitations

- **The same genres and language as version 1.0.** English only; Wikipedia, web text, novels, stories and Dolly answers.
- **The v3.1 design saw v3's fresh results.** Its registered attacks were chosen after v3's read. The held-out attacks and this paper's fresh region are the blind checks.
- **Re-drawn test sets.** Six test sets of version 1.0 had no recorded seed and were drawn again; v2 is compared with v3 and v3.1 on the new draws, and version 1.0's numbers are quoted next to them.
- **Baselines not re-run.** The neural baselines and HANNA's shipped metrics are quoted from version 1.0, on its texts.
- **Prompt references.** v3's prompt statistics use prompt/continuation pairs cut from reference documents; there is no option for the genre's real prompt–answer pairs (Section 9.5).
- **Development data.** The control-ladder damage types and the registered attacks were used in design; the held-out corruptions, held-out attacks, model ladders and human judgments were not.
- **No human rating of our own.** The K5 question in particular needs one.

## 12. Conclusion

Version 1.0 showed that counts alone can tell prose from broken text, if each statistic is read against human text and the evidence is combined with the right structure. This version shows two more things.

First, a second and third count-based judge, built from code lengths and string matches and sharing no co-occurrence statistic with v2, are worth having. On their own they are weaker than v2. Combined by a geometric mean, so that a confident catch is not diluted, they catch copied prompt sentences, keyword stuffing and bland text that v2 misses, and they put human web text above GPT-2 top-k samples, the blind spot version 1.0 ended on. And they disagree with v2 where it matters: neither confirms K5's gains from a co-occurrence gate or from more text, both of which v2 rewarded.

Second, being harder to fool is not being fool-proof. Each new failure a judge learns to see adds false positives on odd human texts, the margins at the 0.95 line are thin, and the held-out attacks still get through 5 or 6 times out of 8: stuffing at a lower rate, templates and fluent sentences from elsewhere. Sentence order, truncation and truth remain invisible. The next steps follow from those: a statistic of sentence order, a spacing-aware stuffing statistic, prompt statistics read against the genre's real prompt–answer pairs, and a human read of the K5 samples.

## Reproducibility

**Code.** All code is in `SCRIBE_code_v2.0.zip` under the GNU AGPL v3; its `README.md` maps every file to a section. The metric is `native/include/cypha/eval/scribe.hpp`, header-only C++20 with no dependency beyond the standard library; `--profile v31` is SCRIBE v3.1, `v3` is v3, `v2` is v2. The tool is `native/tools/scribe.cpp`; `scribe selftest` runs 119 checks. Word vectors come from `native/tools/ppmi_svd.cpp`.

**This version's runs.** `scripts/scribe_eval/runs/v3_paper.sh all` reruns every new read of this paper (about 50 minutes on 4 cores), and `scripts/scribe_eval/v3_report.py` writes every table of Section 9 into `results_v2.0/`. The tables as published are in `SCRIBE_results_v2.0.zip`, with version 1.0's tables next to them.

**Data.** All data is public: enwik8 and enwik9 (Mahoney, 2011), the GPT-2 output dataset, Project Gutenberg, WritingPrompts, HANNA, OpenMEVA, the MAUVE human evaluation and Dolly 15k; `data.py` downloads and converts each, with SHA-256 hashes. K5's samples are in the Cypha repository (`bench/k5_scale/`).

**Determinism.** Every random choice is seeded. This version's draws use seed 20261104; the fresh region is enwik9 [900, 924) MiB.

## References

Arora, S., Liang, Y., and Ma, T. (2017). A simple but tough-to-beat baseline for sentence embeddings. *ICLR*.

Carlini, N., Tramèr, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, Ú., Oprea, A., and Raffel, C. (2021). Extracting training data from large language models. *USENIX Security Symposium*.

Chang, W. I., and Lawler, E. L. (1994). Sublinear approximate string matching and biological applications. *Algorithmica*, 12(4–5).

Chhun, C., Colombo, P., Suchanek, F. M., and Clavel, C. (2022). Of human criteria and automatic metrics: A benchmark of the evaluation of story generation. *COLING*.

Church, K. W., and Hanks, P. (1990). Word association norms, mutual information, and lexicography. *Computational Linguistics*, 16(1).

Cleary, J. G., and Witten, I. H. (1984). Data compression using adaptive coding and partial string matching. *IEEE Transactions on Communications*, 32(4).

Conover, M., Hayes, M., Mathur, A., Xie, J., Wan, J., Shah, S., Ghodsi, A., Wendell, P., Zaharia, M., and Xin, R. (2023). Free Dolly: Introducing the world's first truly open instruction-tuned LLM. Databricks blog; dataset databricks-dolly-15k.

Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.

Fan, A., Lewis, M., and Dauphin, Y. (2018). Hierarchical neural story generation. *ACL*.

Gehrmann, S., Strobelt, H., and Rush, A. M. (2019). GLTR: Statistical detection and visualization of generated text. *ACL: System Demonstrations*.

Guan, J., Zhang, Z., Feng, Z., Liu, Z., Ding, W., Mao, X., Fan, C., and Huang, M. (2021). OpenMEVA: A benchmark for evaluating open-ended story generation metrics. *ACL*.

Hale, J. (2001). A probabilistic Earley parser as a psycholinguistic model. *NAACL*.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. (2020). The curious case of neural text degeneration. *ICLR*.

Ippolito, D., Duckworth, D., Callison-Burch, C., and Eck, D. (2020). Automatic detection of generated text is easiest when humans are fooled. *ACL*.

Kneser, R., and Ney, H. (1995). Improved backing-off for m-gram language modeling. *ICASSP*.

Kuhn, R., and De Mori, R. (1990). A cache-based natural language model for speech recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 12(6).

Lau, J. H., Clark, A., and Lappin, S. (2017). Grammaticality, acceptability, and probability: A probabilistic view of linguistic knowledge. *Cognitive Science*, 41(5).

Lempel, A., and Ziv, J. (1976). On the complexity of finite sequences. *IEEE Transactions on Information Theory*, 22(1).

Levy, O., Goldberg, Y., and Dagan, I. (2015). Improving distributional similarity with lessons learned from word embeddings. *Transactions of the ACL*, 3.

Levy, R. (2008). Expectation-based syntactic comprehension. *Cognition*, 106(3).

Liu, J., Min, S., Zettlemoyer, L., Choi, Y., and Hajishirzi, H. (2024). Infini-gram: Scaling unbounded n-gram language models to a trillion tokens. *COLM*.

Loch, O. (2026). SCRIBE: Scoring generated text against human text with counts alone. Version 1.0. Zenodo. (This record's previous version.)

Mahoney, M. (2005). Adaptive weighing of context models for lossless data compression. Florida Institute of Technology technical report CS-2005-16.

Mahoney, M. (2011). Large Text Compression Benchmark. https://mattmahoney.net/dc/text.html.

Manber, U., and Myers, G. (1993). Suffix arrays: A new method for on-line string searches. *SIAM Journal on Computing*, 22(5).

McCoy, R. T., Smolensky, P., Linzen, T., Gao, J., and Celikyilmaz, A. (2023). How much do language models copy from their training data? Evaluating linguistic novelty in text generation using RAVEN. *Transactions of the ACL*, 11.

Meister, C., Pimentel, T., Wiher, G., and Cotterell, R. (2023). Locally typical sampling. *Transactions of the ACL*, 11.

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., and Finn, C. (2023). DetectGPT: Zero-shot machine-generated text detection using probability curvature. *ICML*.

Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., Kohli, P., and Allen, J. (2016). A corpus and cloze evaluation for deeper understanding of commonsense stories. *NAACL*.

Pauls, A., and Klein, D. (2012). Large-scale syntactic language modeling with treelets. *ACL*.

Pillutla, K., Swayamdipta, S., Zellers, R., Thickstun, J., Welleck, S., Choi, Y., and Harchaoui, Z. (2021). MAUVE: Measuring the gap between neural text and human text using divergence frontiers. *NeurIPS*.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI technical report.

Sellam, T., Das, D., and Parikh, A. (2020). BLEURT: Learning robust metrics for text generation. *ACL*.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3).

Solaiman, I., et al. (2019). Release strategies and the social impacts of language models. arXiv:1908.09203.

Witten, I. H., and Bell, T. C. (1991). The zero-frequency problem: Estimating the probabilities of novel events in adaptive text compression. *IEEE Transactions on Information Theory*, 37(4).

Version 1.0's reference list covers every other work cited there.

## Appendix A. Every parameter of SURPRISE, MATCH and v3.1

**Table A1.** Parameters of the new judges (`V3Params` in `scribe.hpp`; `set_v31()` for v3.1). "dev A" means chosen on development ladder A and its battery; v2's parameters are version 1.0's Appendix A.

| Parameter | v3 | v3.1 | Fixed |
|-----|----------|-------|---|
| SURPRISE model | pack KN trigram; Witten–Bell character trigrams for unknown words (once-seen types) and numbers; unigram cache λ 0.1; match model k ≥ 2, weight k/(k + 1) | same | dev A |
| Content words for `w_coh` | unigram surprisal ≥ 8 bits | same | dev A |
| `w_lo`, `w_hi` thresholds | 25th and 90th percentiles of the reference's words (4.35, 17.45 bits for Wikipedia) | same | dev A |
| `w_fithi` threshold | | 90th percentile of the reference words' fit (1.18 bits) | dev A |
| MATCH index | suffix array of the pack, first 64 bytes, bucketed by the first two bytes; matches up to 64 bytes | same | dev A |
| `m_coh`, `m_pcoh` | matches ≥ 5 (prompt: 4) bytes; byte weight log₂((N + 1)/(count + 1))/log₂(N + 1) above 0.45 | same | dev A |
| `m_lz` in C | weight 0.5, dead zone 2 | same | dev A |
| z-score, missing statistic | v2's `QuantileZ`; z = 4 on the bad side when ≥ 95% of the reference has it | same | v2 |
| Dead zones L, S, C, G, X (SURPRISE / MATCH) | 4, 1, 0, 1.5, 1 / 4, 1, 0, 2, 1 | same | dev A |
| L and S caps; κ (SURPRISE / MATCH) | 30, 30; 2 / 0.25 | L 90, S 30 | dev A |
| w_C, w_G, w_P (SURPRISE / MATCH) | 4, 1, 8 / 6, 1, 20 | same | dev A |
| F | | cap 60, κ 2, dead zone 1, threshold the 99th percentile | dev A |
| B (SURPRISE only) | | 10 per unit beyond the 97.5th percentile, no dead zone | dev A |
| Thresholds of L, S, X | 99th percentiles | same | dev A |
| τ (SURPRISE / MATCH) | 20 / 30 | 27 / 32 | dev A's human mean 0.909 |
| Echo reference | up to 1,000 prompt / continuation pairs cut from the reference documents (seed 20261104) | same | v3 |
| Combination | (v2 + S + M)/3; `min3`; `disagree` at a spread > 0.5 | (v2 · S · M)^(1/3) | dev A |
| Reported only | `w_anti`, `m_words` | `w_lead`, `m_echo32` | |

## Appendix B. Changes from version 1.0

- New: SURPRISE, MATCH, SCRIBE v3 and SCRIBE v3.1 (Sections 5–7); the challenges between the versions (Section 3); two more fresh reads (enwik9 [100, 124) and [900, 924) MiB); the attack batteries and held-out attacks; the K5 cross-checks; a re-run of the meta-evaluation with v3 and v3.1; twenty more registered predictions.
- Unchanged: SCRIBE v2 and every number version 1.0 reports about it. Its five post-freeze defects (version 1.0, Section 6.10) are still unfixed in v2, and therefore in v3.1, which contains it; fixing them would change v2's definition.
- Not repeated here: version 1.0's baselines, ablations of v2 and design history of v2, which remain in version 1.0.
