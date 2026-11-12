# Decoder research: what the record and the literature say about a mixer decoder

**Odin Loch** · 2026-11-12 · research for D4, the mixer decoder (`wip/k5-mixdec`, `docs/DECODER.md` on that branch) · reading and writing only; no code changed, nothing run

Odin's ask: "a mixer decoder and an architecture like a LLM architecture… we have a learner/encoder and we need a decoder… do the reverse of compression." D4 is building it now. Its design, as briefed: a word-level product of experts. K5 proposes words. The byte mixer scores each word's spelling exactly. The combined distribution is decoded by arithmetic decoding. The learner is primed on the prompt and on a retrieved article.

This note gathers the evidence that should shape that design. Part 1 is the project's own record, read on `origin/main` at 4629f68c. Part 2 is the literature. Section references (§) are to [GENERATION_PLAN.md](../GENERATION_PLAN.md) unless a report is named.

---

## Summary: ranked design recommendations

Each item names its evidence. "Readers" means the blind pairwise Claude raters of §13.38's rule (win share against the baseline, 95% interval, both rater pairs agreeing).

**1. Enter the mixer as a context term, not as a raw product.** Score a word by the mixer's context-conditioned probability divided by its prior (the PMI form, `p_mix(w | context) / p_mix(w | " ")`), raised to a small weight, normalised over K5's candidates. Never multiply K5 by the raw spelling probability.
- This exact design has been run once, with the byte LSTM as the byte model (§13.24). K5 proposed words inside its nucleus; the LSTM scored each word's bytes after the prompt and the text so far; the two were combined as `p_K5^(1−w) × p_LSTM^w`. SCRIBE v2 fell from 0.353 to 0.149 (w 0.25) and 0.018 (w 0.5). The mean word fell from 4.31 to 3.48 letters and distinct words halved (0.171 → 0.076). Linear mixing fell to 0.106.
- The cause is a length and frequency bias. A product of byte probabilities is a whole word probability, so short common words win. K5 already holds that prior, so the product counts it twice.
- The PMI fix (DExperts' form) restored word length (4.45) and raised SCRIBE to 0.221 (w 0.5), but still lost to K5 alone (0.353). Copying rose (0.035 → 0.055). The text got "smoother" (surface S 0.96) and blander.
- The literature says the same: products and guidance work only with fractional weights and a negative (prior) expert (contrastive decoding, DExperts, classifier-free guidance; Part 2 §2.2).
- The mixer is far stronger than that LSTM (1.49 against 2.53 bits per byte alone). So D4 may do better. But the record predicts that a strong byte model mostly sharpens, and sharpening is what readers dislike (item 3).

**2. Truncate after combining. Arithmetic decoding at temperature 1 is plain ancestral sampling.** Decoding random bits through an arithmetic decoder draws each symbol with exactly the model's probability. It adds exactness, not quality.
- Every CyphaLM writer needed truncation. The byte sampler scores 0.174 at T 0.6 and 0.073 at T 1.0 (§13.16). Tokens: the nucleus cut was the lever (0.119 → 0.192 on dev, T4c). K5's nucleus moved 0.9 → 0.8 (adopted, K5_UPGRADES) and 0.7 read better again (N7, 0.670, TUNE_WRITE).
- Literature: nucleus, eta/epsilon, typical and min-p sampling all exist because untruncated tails ruin samples (Part 2 §2.3).
- So: keep arithmetic decoding as the random-number mechanism if Odin wants "the reverse of compression", but decode a truncated, tempered distribution (nucleus at word level, re-tuned on the combined distribution).

**3. Never pick by the mixer's likelihood.** No best-of-K by mean log-probability, no beam search, no "most probable word".
- Word lookahead (CYPHALM_LM_QUALITY_REPORT, CYPHALM_VS_NEURAL_LM, 2026-09) did exactly this: K 8 sampled candidate words, the one with the best mean log-probability was kept. It wrote 99.4% real words. But its text sat at 0.70–0.90 bits per byte under the judges against 1.68–1.86 for real text: it was far more predictable than human text, and still incoherent ("…the internal existing the trachea are often the tracheole cell basement membrane…"). Without a repeat ban, best-of-K looped.
- WRITING_SIGNAL (§13.57): lower surprise agrees with readers between settings of one writer (0.738), but when an arm adds content, readers prefer the more surprising text (0.367, 0.14). Human text is far more surprising than CyphaLM's (AUC 0.93).
- The structure mixer (§13.36) picked the real sentence among K5 drafts at 0.113 bits, but in writing it chose the oddest draft; readers rated it 2.00 like K5.
- Literature: the likelihood trap (Zhang 2021), MBR with weak likelihood–quality correlation (Freitag 2022), beam degeneration (Holtzman 2020).

**4. Use the mixer where it is strong: as a plausibility veto on spelling and names.** The mixer is excellent at copying, spelling and rare strings (CYPHALM_VS_NEURAL_LM: it wins on lowercase letters and digits; 0.05 bits per byte on a repeated passage). Use it to drop candidates it finds implausible (contrastive decoding's adaptive plausibility constraint: keep words with at least α of the best candidate's mixer score), then sample K5's own distribution inside that set.
- §13.16 tried this gate for the GLN in the byte sampler. It cut the bad tail and won at T 1.0 (0.117 against 0.073), but not at the best temperature. It is the least harmful way in.
- This also covers out-of-vocabulary words. K5 writes `<unk>` never; the byte model could spell names K5 lacks (§13.11 planned it; §13.19 left it unbuilt).

**5. Prime on the prompt and the retrieved article, then freeze. Never learn from the decoder's own output.**
- Learning from its own output made CyphaLM loop (distinct 4-grams 0.23–0.59, CYPHALM_LM_QUALITY_REPORT). An unfrozen document cache looped on its own words (GEN-C: "offering … offering"). K5's cache and dynamic counts lowered bits to 1.488 and lowered SCRIBE (K5_UPGRADES); on the fluent writer they did not move readers (0.500, K5_CACHE2).
- Priming on retrieved text is the one content move that readers rewarded at the word level: K1, K5's counts overlaid with the BM25-nearest article, 0.860 and 0.880 against HLT (§13.66). So priming the mixer on the same article is well motivated. But the gain came from the writer's counts, not from a better byte predictor. D4 should keep K1's overlay in K5 and treat the mixer's priming as extra.
- Keep the byte history when priming (resetting it cost 0.02–0.03 bits per byte, CYPHALM_LM_QUALITY_REPORT).
- Watch copying: copy32 rose with every retrieval strength (0.232 → 0.275 at k 1, 0.501 at rel 1.5; §13.66). A primed mixer is a strong copy model.

**6. Keep every stage that moved readers, and run the decoder under them.** What moved readers: K5's counts from 337 MB (0.940, §13.48), retrieval (0.860/0.880, §13.66), the planner (0.635, §13.38), the discourse writer (0.590, §13.46), gate 16 and nucleus 0.7 (0.670 each, §13.51), the LDA term in the gate (0.610/0.700, §13.53). What did not: every change that reshaped local next-word or next-byte probabilities (caches, dynamic counts, fuzzy experts, LSTM, GLN, MBR, MCMC, entity grid, topic rescaling, paragraph-plan term, first-sentence gate, web counts). D4 changes the word distribution under all of them, so the gate, nucleus and LDA weight may need re-tuning, and the arm must be compared with K1, not with plain K5.

**7. Judge by readers, against K1, at 460 and 1,500 characters.** Register the arms, rules and predictions first. SCRIBE and bits are screens only. SCRIBE rose where readers fell three times (topic rescaling §13.34, R2 §13.53, C4 counts §13.67). Bits fell where writing fell in every experiment (§13.11, GENERATION_ONLINE_RECORD). End with a 1–10 round: today's writer is 2.28 of 10 and human text 8.54 (§13.69). That is the bar a new decoder must move.

**8. Budget the cost before the arms.** The mixer is the slow part. CyphaLM's serve stack takes about 2.4 ms per next-byte distribution (CYPHALM_VS_NEURAL_LM). Word lookahead K 8 cost 12.7 ms per byte against 1.8 ms for byte sampling. The LSTM inside K5 cost 125 ms a word, about 90× K5 alone (§13.24). Inference, not measured: scoring 50 nucleus words of 6 bytes with prefix sharing is a few hundred byte distributions a word, about 0.5–1 s a word; with gate 16's 16 drafts a sentence, the 250 dev prompts would take days on one thread. Score only K5's truncated top words, share prefixes, freeze learning while scoring and rewind exactly (the older hp serve stack has `hp::StreamRewind`; a quick search found no rewind in `cyphalm_trie`'s `context8k` path, so check before relying on one), and apply it to the chosen draft path only if needed.

### Where the evidence contradicts the current D4 design

1. **"K5 proposes words, the byte mixer scores their spelling exactly", as a product.** This is §13.24's KL1–KL3 with a stronger byte model. Every product weight tried lost to K5 alone (0.149, 0.018 against 0.353); the PMI form lost too (0.221). The product double-counts the word prior and favours short words. Use the PMI form with a small weight, or a plausibility veto, and test w = 0 as an arm.
2. **"Decoded by arithmetic decoding of the combined distribution."** If that means temperature 1 without truncation, the record and the literature both predict worse text: the byte sampler at T 1.0 scored 0.073 against 0.174 at T 0.6. Arithmetic decoding is fine only as the sampler under a truncated, tempered distribution.
3. **The spelling score must include the word's end.** `P(" word" + delimiter | context)`, not the prefix probability: otherwise "the" also counts every "then" and "there". T3's exact byte–token bridge (TOKENS_T3_T4) already computes exact prefix and cover probabilities, and is the place to reuse.
4. **Priming the learner** agrees with the evidence only if the learner stops learning once the decoder starts writing. Learning from its own output loops.
5. **If D4 uses draft-and-verify (speculative style) to be exact, the output is the verifier's distribution.** Exact acceptance (min(1, q/p)) returns the mixer's own word distribution, and K5's contribution vanishes. The mixer's own writing has never been read: `sample` never used the 1.49 stack (§13.29), and the older byte sampler on the same family of experts scored 0.147–0.174 (word salad), against K1's 0.706 test SCRIBE.
6. **What the current design does not contradict:** word-level units (the biggest single writing gain, §13.11), K5 as the proposer, exact scoring, and priming on retrieved text (the second-largest reader gain, §13.66).

---

## Part 1. The project's own record

### 1a. Every attempt to write text, and why it failed or worked

Scores are SCRIBE v2 on GEN-A unless noted (human 0.923; sentences shuffled across documents 0.344; GPT-2 small 0.767; §2). "dev" and "test" are GEN-A's splits.

**Bytes.**

| Attempt | Section / report | Result | Why |
|---|---|---|---|
| Trie only, 8 MiB; 1 TB | §2, Track A | 0.020; 0.028 | Order-4 bytes see 0.6 of a word; word salad |
| Similar-context expert in `sample` (A4b) | A4_SIMILAR_CONTEXT_EXPERT | 0.014 → 0.061 | More reach helps from a very low base |
| Class expert (B4c) | B4C_ECHO_GATE | 0.061 → 0.025 (disjoint) while BPC fell | "Better bits do not mean better text"; kept out of `sample` |
| Copy expert (B4c) | B4C_ECHO_GATE | 0.061 → 0.056 (overlap) | Kept |
| Clean prose as training text (GEN-A) | §3 | dev 0.079 → 0.069; copy 0.041 → 0.005 | Markup was a small part; sentence form is the rest |
| Sentence gate, best of 4 by own fluency − markup (GEN-B) | GEN_B_SENTENCE_GATE | test 0.072 → 0.090 (overlap); D_S −8%, D_C +10%; 4.1× CPU | Picks form, loses topic; candidates too poor |
| Document cache and triggers (GEN-C) | GEN_C_TOPIC_CACHE | ≤ 0.060 against 0.063; BPC −0.043; D_C −23 to −35% | Topical words land in broken byte syntax; an unfrozen cache loops on its own output |
| Content-plan expert from the fact store (GEN-E) | GEN_E_CONTENT_PLAN | test 0.072 → 0.071; covered BPC −0.020 | A byte expert cannot hold a phrase: "was born in China" for Cheshire |
| Byte Kneser–Ney 6:4:2:1 (BKN/BKN2) | BYTE_KN | dev 0.079 → 0.125; test 0.147 at T 0.6 | The largest byte-level gain: smoothing and reach |
| Temperatures below 0.6 (TEMPS) | TEMPS_BYTE_SAMPLER | 0.5 ties (0.157); 0.4 0.125; copy 0.069 → 0.130 | T 0.6 is the byte sampler's best; lower copies and invents |
| Online lambdas in writing | §13.22 | −0.027 | Writing is set by the sampler, not small weight changes |
| GLN, linear | §13.15–13.16 | 0.256 → 0.020 at T 0.6 (10 prompts) | A confident wrong peak ("contributor" → ">") sharpened by T 0.6 |
| GLN as product of experts (`geo`), gate (`gate`), clean prose (250 prompts) | §13.16 | 0.090, 0.117 (T 1.0), 0.153 against 0.174 | None non-inferior; the gate helps only at T 1.0 |
| GLN on 1 GiB of C4 | §13.17, §13.27 | gate 0.143, geo 0.111 | Wrong domain; bits worse too |
| Byte LSTM in the sampler | §13.18 | gate 0.155 (non-inferior), linear 0.119, geo 0.119 | Taken out (Odin: "go with K5") |
| 1 TB trie in the sampler | §13.27 | 0.138 against 0.174 | More short context does not make sentences |
| GLNB writing | §13.27 | 0.046 | — |
| hp gate24 + word lookahead K 8 (2026-09) | CYPHALM_VS_NEURAL_LM, CYPHALM_LM_QUALITY_REPORT | 99.4% real words; judge bits 0.70–0.90 against real 1.68–1.86; "far from fluent" | Fixes invented words, not sentences; picks the most probable, so bland; best-of-K loops without a ban |
| Greedy; learning from own output | CYPHALM_LM_QUALITY_REPORT | "the set the set the set"; distinct 4-grams 0.23–0.59 | Argmax of a finite-context model cycles; online learning on own output loops |

**Tokens and words.**

| Attempt | Section / report | Result | Why |
|---|---|---|---|
| Whole-token sampling (T4, T4b) | TOKENS_T3_T4 | 0.112 / 0.115 against 0.106 | Tokens alone do not lift SCRIBE |
| Tokens + nucleus 0.9 + repeat ban (T4c, NRK0) | §2, §3 | test 0.187; invented words 6.9% (subword pieces) | The nucleus is the lever; the ∞-gram copies |
| **K5**: word Kneser–Ney order 5 on 7 MB clean prose, nucleus 0.9, repeat ban | §13.11, §13.19 | dev 0.344–0.353; test 0.305 | Whole words + clean prose: the largest jump |
| 10 fuzzy experts, self-organising experts mixed into K5 | §13.11 | all lower SCRIBE by 0.013–0.33 while bits fell 0.0004–0.032 | "Weights fit to predict spread the mass the sampler draws from" |
| K5 + byte LSTM, product / linear / PMI | §13.24 | 0.149 / 0.106 / 0.221 against 0.353 | Length bias; blander, more copying |
| K5 upgrades: nucleus 0.8, 256 exchange classes, 8-draft gate | §13.25, K5_UPGRADES | test 0.524; readers 2.09 against 1.77 of 10 (+0.32) | Small reader gain; still "sentence-shaped but not connecting" |
| MBR, entity grid, LDA rescaling, MCMC, structure meta-mixer | §13.34, §13.36 | all about 2 of 10; topic rescaling −0.22 / −0.42 | Choosing among K5's drafts cannot fix K5; its drafts hold no good sentence |
| Planner: plan → skeleton → fill → gate | §13.37–13.39 | readers 0.635 against K5; test 0.561 | First sentences on the subject; fill invents details |
| Discourse planner, text-only store (Q3t) | §13.46 | readers 0.590 against P1t; test 0.631 | — |
| K5 counts from 337 MB (W1b) | §13.48 | readers **0.940** against D0 | The free text became grammatical. The largest reader gain |
| Gate 16; nucleus 0.7 | §13.51, TUNE_WRITE | readers 0.670 each against W1b | G16 adopted; the pair NG ties G16 (0.510) |
| LDA topic term in the gate (L1) | §13.53, K5_COHERENCE | readers 0.610 (460 chars), 0.700 (1,500) | Adopted; the link to the previous sentence (R2) read worse |
| Honest fill (H1b, HLT) | §13.48, §13.63, HONEST_DATES2 | fact precision 0.437 → 0.935; readers 0.540 (not worse) | Adopted for honesty, not reading |
| Retrieval overlay (K1) | §13.66, K5_RETRIEVE | readers **0.860 / 0.880** against HLT; test SCRIBE 0.706 | Content from the subject's neighbourhood |
| Retrieval refresh (RG) | §13.68, K5_RETRIEVE2 | 0.620 [0.440, 0.780] | Not adopted (interval includes 0.5) |
| C4 web counts | §13.67, K5_C4 | not rated; first- and second-person share 0.08 → 0.52–0.62 | The web's voice leaks in |

**The pattern.** Two moves made big jumps: whole words counted on clean prose (K5), and more of the right text (337 MB of Wikipedia prose, then the retrieved article). Truncation (nucleus, temperature) mattered at every level. Every attempt to make writing better by adding a predictor to the sampling distribution failed.

### 1b. Every expert's effect on bits against its effect on writing

"How blended" names the combination inside the sampler.

| Expert | Bits | Writing | How blended | Ref |
|---|---|---|---|---|
| Class expert | −0.08 to −0.09 BPC (qa holdout) | 0.061 → 0.025 | linear λ per bucket, fit on the holdout, frozen | B4C |
| Similar-context | −0.061 BPC | 0.014 → 0.061 (up) | linear λ per bucket | A4 |
| Byte Kneser–Ney | −0.18 to −0.19 BPC | 0.079 → 0.125 (up) | linear λ per bucket (way b) | BYTE_KN |
| Document cache + triggers | −0.043 BPC | 0.063 → ≤ 0.060; loops when unfrozen | linear λ per bucket, after the default experts | GEN_C |
| Content plan | −0.020 BPC on covered prompts | 0.072 → 0.071 | linear λ per alignment-length bucket, last | GEN_E |
| GLN | −0.033 BPC in `context8k` | 0.174 → 0.090–0.153 | linear; geometric `chain^(1−λ) × GLN^λ`; plausibility gate α 0.1 | §13.16 |
| Byte LSTM | −0.023 BPC | 0.174 → 0.119–0.155 | linear; geometric; gate | §13.18, §13.22 |
| 1 TB trie | −0.006 BPC | 0.174 → 0.138 | replaces the trie | §13.27 |
| Online lambdas | −0.027 BPC (in scoring) | −0.027 SCRIBE | the chain's weights learn online, then frozen for writing | §13.22 |
| 10 fuzzy experts into K5 | −0.0004 to −0.032 bits per byte | −0.013 to −0.33 (ALL: best bits, worst text, 0.218) | linear, weights fit to predict | §13.11 |
| Byte LSTM into K5 | — | 0.353 → 0.018–0.221 | geometric over K5's nucleus; linear; PMI | §13.24 |
| K5 word cache | to 1.488 bits per byte (−0.061) | 0.114–0.359 against 0.394 | linear, before temperature | K5_UPGRADES |
| K5 dynamic counts | small (1.528–1.534 against the cache arms' 1.489–1.507) | −0.011 over 4 seeds | added into Kneser–Ney counts | K5_UPGRADES |
| K5 orders 6–7; 65,536 words | small | none; lower | — | K5_UPGRADES, K5_SCALE |
| Full mix, meta mix (prediction only) | −0.074, −0.009 | never used by `sample` | — | §13.29, §13.35 |

**The repeated finding.** In prediction a weak model with different errors helps a mixture. In writing, every sampled symbol the extra model pulls in carries its errors, and the hedged mixture spreads mass the sampler then draws from (§13.11, §13.16, GENERATION_ONLINE_RECORD §1). Linear blends were worst. Geometric blends sharpened toward short common words. The gate (plausibility constraint) did least harm and helped only at T 1.0.

**One caution for D4.** The 1.49 stack (`context8k`: online chain, n-gram, LSTM 256, full mix, meta mix, 1 TB trie) has never written a byte. `sample` uses the older trie + byte Kneser–Ney + similar-context + copy stack with frozen lambdas, at about 2.0 bits per byte on GEN-A continuations (TEMPS). So D4 is the first writing test of the strong mixer. The record gives a strong prior, not a measurement.

### 1c. What moved readers, and what did not

Blind pairwise raters, four per round, 50 prompts (25 for long readings). A win share above 0.5 with an interval excluding 0.5 "reads better".

| Change | Against | Win share [95%] | Moved readers? | Ref |
|---|---|---|---|---|
| K5 counts from 337 MB (W1b) | D0 | **0.940** [0.860, 1.000] | **Yes** | §13.48 |
| Same + honest fill (WHb) | D0 | 0.930 | Yes | §13.48 |
| Retrieval overlay k 1 (K1), 460 / 1,500 chars | HLT | **0.860 / 0.880** | **Yes** | §13.66 |
| Retrieval k 4 (K4), 460 / 1,500 | HLT | 0.820 / 0.940 | Yes (kept as a flag) | §13.66 |
| Gate 16 drafts (G16); nucleus 0.7 (N7) | W1b | 0.670 / 0.670 | Yes | §13.51 |
| LDA term in the gate (L1), 460 / 1,500 | G16 | 0.610 / 0.700 | Yes | §13.53 |
| Planner (P1) | K5 | 0.635 | Yes (partly on Wikidata seeds, §13.46) | §13.38 |
| P1t (text-only store) | K5 | 0.610 | Yes | §13.46 |
| Discourse planner (Q3t) | P1t | 0.590 | Yes | §13.46 |
| Sentence gate (upgraded K5), 1–10 scale | old K5 | +0.32 points | Yes, small | §13.31 |
| Honest life dates (HLT) | L1 | 0.540 [0.500, 0.590] | No (not worse; adopted for fact precision 0.935) | §13.63 |
| H1b honest fill | D0 | 0.560 | No | §13.48 |
| Retrieval refresh (RG) | K1 | 0.620 [0.440, 0.780] | No (leans better) | §13.68 |
| First sentence, 64 drafts (K64) | L1 | 0.600 [0.500, 0.700] | No (leans better) | §13.60 |
| N7 + G16 | G16 | 0.510 | No | §13.51 |
| O6 (order 6) | W1b | 0.580 [0.460, 0.690] | No | §13.51 |
| Word cache, dynamic counts | HLT | 0.500–0.560 | No | §13.65 |
| Paragraph-plan term | L1 | 0.470–0.520 | No | §13.62 |
| Link to the previous sentence (R2) | G16 | 0.390 / 0.460 | Worse | §13.53 |
| Dynamic beam fill F1 / F2 | P1 | 0.420 / 0.530 | Worse / no | §13.40 |
| Topic rescaling B 1 / 1.5 / 2 | K5 | 0.00 / −0.22 / −0.42 points | No / worse | §13.34 |
| MBR, MCMC + entity grid | K5 | +0.04 / −0.05 points | No | §13.34 |
| Structure meta-mixer picks | K5 | 0.000 | No (floor) | §13.36 |

**What it says.** Readers reward content and fluency from data: more of the right text in the counts, the subject's own neighbourhood, real facts in the first sentences, and more drafts to choose from. They do not reward reshaped local probabilities (caches, terms, smoothing), and they punish keyword stuffing. Text signals: only lower mean surprise tracks readers overall (0.588), and only between settings of one writer; when content is added, readers prefer the more surprising text (§13.57). The gap to human text is commitment (0.67 against 0.28 of sentences reuse an earlier word) and sentence length (19.7 against 12.7 words). The writer's link to the previous sentence is 0.204 against 0.508 for humans (§13.59).

### 1d. The absolute reading (§13.69)

Four blind raters rated 1–10, 50 prompts, 460 characters, two ratings per text:

| Arm | Mean [95%] | Rated 6 or more | Minus human |
|---|---|---|---|
| human | 8.54 [8.32, 8.73] | 99 of 100 | — |
| K1, today's default | 2.28 [2.16, 2.39] | 0 of 100 | −6.26 [−6.48, −6.03] |
| Q3t | 1.82 | 0 | −6.72 |
| plain K5, 7 MB | 1.66 | 0 | −6.88 |

K1 against K5 is +0.62 [+0.49, +0.75]. Readers place today's writer between "word salad" and "sentence-shaped but not connecting". SCRIBE v2's 0.706 overstates it. Earlier rounds agree: K5 2.09 (§13.31), 2.18 (§13.34), 2.00 (§13.36), human 8.4–9.0.

**For D4.** The decoder has to move this number, not SCRIBE or bits. A pairwise win against K1 of the size of W1b (0.94) or K1 (0.86) has moved the 1–10 scale by about half a point. A new decoder that only changes word choice inside K5's nucleus should be expected to move it less.

### 1e. Costs by stage

| Stage | Time | RAM | Ref |
|---|---|---|---|
| `context8k` default (1.4962), pipeline on, 6 CPUs | 712 s (was 1,960 s) | 1.91 GB | §13.54 |
| `context8k` with the 1 TB trie (1.4900) | 7:39 | 2.92 GB; file 992 MB | §13.55 |
| Full mix (folded in) | +1% time | +287 MB | §13.30 |
| Meta mix | +5.9% | +405 MB | §13.35 |
| Online n-gram + LSTM 256 | +61% before optimisation | 828 MB peak (n-gram tables) | §13.26 |
| LSTM 256 alone | 937 s → 218 s after SIMD | — | §13.30 |
| GLN on 1 GiB | 1 h 20 min training | 1.6 GB; 403 MB file | §13.17 |
| CyphaLM serve, one next-byte distribution | 2.4 ms (4 threads) | 3.3 GB + 0.6 GB mapped | CYPHALM_VS_NEURAL_LM |
| Byte sampling / word lookahead K 8 (lean) | 1.8 / 12.7 ms per byte | — | CYPHALM_LM_QUALITY_REPORT |
| Byte sampler, 1,500 samples (trie + sim) | 292 s | 82.7 MB | A4 |
| GEN-B gate, 4 candidates | 4.1× CPU per byte | — | GEN_B |
| K5 on 7 MB: build; 250 samples, 4 threads | 1 s; 9 s (1.1 ms a word) | model 119 MB, peak 220 MB | §13.19 |
| K5 + LSTM scoring every nucleus word | 125 ms a word (about 90×) | — | §13.24 |
| Planner default on 7 MB (D0) | 2:20 for 250 dev prompts | 0.64 GB | §13.48 |
| K5 on 337 MB (W1b, today's base) | build about 140 s; 6:45 for 250 prompts on 6 threads | 11.3 GB | §13.48 |
| K5 + 150–600 MB of C4 | — | 16–31 GB | §13.67 |
| Lazy exact nucleus (O6) | K5 sampling −33% to −70% | +7 MB | §13.50 |

**For D4.** Today's writer already needs 11.3 GB. The mixer adds about 2–3 GB. The time cost is per scored byte, so the number of candidate words scored per step and the number of gate drafts set the bill (summary item 8).

---

## Part 2. The literature

Drafted by a research subagent with web search, then checked against the record. Links marked [V] were fetched and the title and year matched; [U] are from memory and not verified. Numbers come only from fetched abstracts. "Inference" marks reasoning, not a paper's finding. Where the project's own record bears on a prediction, it is noted under "For CyphaLM".

### 2.1. Generation from compression models

#### 2.1a. Arithmetic decoding as sampling

Mechanism. An arithmetic decoder splits the current interval in proportion to the model's probabilities. If you feed it uniformly random bits, it picks each symbol with exactly the model's probability. So "decompress random bits" is ancestral sampling at temperature 1 from the model. It is the same as sampling from the softmax. It adds nothing beyond sampling, except exact bookkeeping of the bits spent.

Sources.
- Mahoney's PAQ pages and ZPAQ spec describe the predictor/coder split, which makes this possible: https://www.mattmahoney.net/dc/paq.html [V as a page; it does not describe text generation]. I searched for a Mahoney writeup on generating text from PAQ/cmix and PPM and found none. I cannot cite one. Shannon's 1948 n-gram "approximations to English" is the classic sampling-from-a-count-model demo (from memory, [U]).
- Delétang et al. 2023, "Language Modeling Is Compression": https://arxiv.org/abs/2309.10668 [V]. The abstract says any compressor (like gzip) can be turned into a conditional generative model. I only saw the abstract. I do not remember or verify the sample quality they report, so I make no claim about what gzip samples look like. Their main empirical content is compression rates of LLMs on text, image, audio, not generation quality.

What it predicts for CyphaLM. Byte-level sampling at temperature 1 has to survive every byte. Each byte has a small tail of wrong choices. A word has 5 to 8 bytes, so error compounds into invented words. This matches the "byte salad". Inference: the fix the literature offers is truncation (§2.3), not a better arithmetic coder.

Main risk. A model that is well calibrated for compression is calibrated for the average over the data. Sampling from it at T=1 produces text with the entropy of the data, with no repair of earlier mistakes. Small models drift off-distribution quickly (exposure bias). This is the main known failure of pure ancestral sampling (Holtzman 2020, below).

#### 2.1b. kNN-LM: interpolation, and generation quality

Mechanism. kNN-LM linearly interpolates the base LM's distribution with a distribution from nearest neighbours in a datastore of hidden states: p = λ p_kNN + (1−λ) p_LM. It is a mixture, not a product.

- Khandelwal et al. 2020, "Generalization through Memorization": https://arxiv.org/abs/1911.00172 [V]. Reported WikiText-103 perplexity 15.79, 2.9 points better than the base LM, with no extra training.
- Wang et al. 2023, "kNN-LM Does Not Improve Open-ended Text Generation": https://arxiv.org/abs/2305.14625 [V via search result; my first guess at the ID was wrong]. Finding: kNN-LM lowers perplexity but gives no matching gain in open-ended generation quality, by automatic metrics (MAUVE) and by human evaluation. Reasons given in the paper summary I found: interpolation makes most tokens worse and a small number of tokens much better, so perplexity improves on average. Retrieval entropy also grows faster than the base LM's entropy as the generated text gets longer. So retrieval is less reliable when the queries are model-generated.

Prediction for CyphaLM. This is the closest published analogue of your experience: bits improve, samples do not. The gain is concentrated on a few tokens (rare words, names, copies), and generation wants the average token to be good. Inference: a better byte mixer will help K5 candidates on rare and spelled-out words, but not the plot of the text.

Main risk. If the combined model helps perplexity through rare-token wins only, sampled text will look unchanged or blander. Measure per-token (per-word) win rates, not only total bits.

#### 2.1c. infini-gram

- Liu et al. 2024, "Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens": https://arxiv.org/abs/2401.17377 [V]. Abstract: 5 trillion tokens; next-token accuracy 47% for the unbounded n-gram; millisecond latency; combining with neural LMs "greatly reduces" perplexity (no number in the abstract).
- I did not verify any open-ended generation result in that paper. From memory, the paper focuses on analysis of text and on perplexity, not on generation quality. Treat that as unverified.

Prediction. Large count models are strong for accuracy but this is about next-token accuracy, which says little about sample quality.

#### 2.1d. n-gram plus neural mixtures for generation

Linear interpolation is the standard (kNN-LM, infini-gram, cache models). The geometric alternative is in §2.2. I found no verified paper showing an n-gram + neural mixture improves human-rated generation. The only verified generation result (Wang 2023 above) is negative.

---

### 2.2. Combining models at decode time

#### 2.2a. Product of experts and log-linear interpolation

Mechanism. A product of experts multiplies the experts' probabilities and renormalizes: p(x) ∝ Π p_i(x)^{w_i}. A single expert can veto. Each expert only needs to be right about what is wrong. Log-linear (geometric) interpolation of LMs is the same idea with weights tuned on held-out data.

- Hinton 2002, "Training Products of Experts by Minimizing Contrastive Divergence", Neural Computation 14(8). No arXiv version. DOI https://doi.org/10.1162/089976602760128018 [U].
- Klakow 1998, "Log-linear interpolation of language models", ICSLP 1998. No arXiv link; I could not find a stable URL [U]. From memory: it reported lower perplexity than linear interpolation in some settings. I cannot verify the size.

Prediction. Inference: a PoE with weights summing to 1 is a geometric mean and does not sharpen. A PoE with weights summing to more than 1 sharpens, like lowering temperature. In your design, both K5 and the byte mixer already model the same word-choice question. Mixer-scoring a word's spelling in context also contains information on whether that word is likely there. So a product of K5 and the mixer's word probability is a double-count unless the weights are fractional. With weights of 1 and 1 you get a sharper, blander distribution. This matches your earlier finding that stronger experts make text blander.

Main risk. The normalizer. The product is only defined up to the normalizer over all words. With K5 proposing N candidates, you normalize over N candidates only. That makes it a self-normalized importance-style approximation, and it depends on N and on which words K5 happens to propose.

#### 2.2b. Contrastive decoding and DExperts

- Li et al. 2022/2023, "Contrastive Decoding: Open-ended Text Generation as Optimization": https://arxiv.org/abs/2210.15097 [V]. Mechanism: score = log p_large − log p_small, restricted to tokens the large model finds plausible. It removes what a small model finds easy (generic, repetitive text). I did not fetch the numbers.
- Liu et al. 2021, "DExperts": https://arxiv.org/abs/2105.03023 [V]. Mechanism: logits of a base model plus α (expert logits − anti-expert logits), at decode time. Controlled generation (toxicity, sentiment).

Prediction. These are the literature's direct answer to your "blander" problem, and the sign is opposite to a plain product. They use a weaker model as a negative. Inference: K5 is weak on spelling and generic; its generic predictions could be subtracted, with the mixer as the strong model. But the plausibility constraint (top-p/top-alpha of the strong model) is essential. Without it, contrastive scores favor odd words, which is exactly what invented words are.

Main risk. A tail-seeking objective plus a spelling scorer will prefer rare, odd, or misspelled strings. It needs a hard floor from the strong model.

#### 2.2c. Classifier-free guidance for LMs

- Sanchez et al. 2023, "Stay on topic with Classifier-Free Guidance": https://arxiv.org/abs/2306.17806 [V]. Abstract numbers: humans preferred GPT4All outputs with CFG 75% of the time over the baseline; gains comparable to a model with twice the parameters; LLaMA-7B reached state of the art on LAMBADA, ahead of PaLM-540B.
- Mechanism: p ∝ p(x | prompt)^γ / p(x | no prompt)^(γ−1). It is a PoE with a negative expert (the unprompted model). This is the closest published form of "priming on the prompt".

Prediction. For CyphaLM: guidance toward the prompt and the retrieved article could be done as the ratio of a primed and an unprimed learner. This keeps the topic without over-sharpening generic text. Risk: guidance scales above about 1 increase repetition and reduce diversity (this is from memory of the paper, I did not verify it). Also the unprimed model must be kept around, which doubles the cost.

#### 2.2d. Speculative decoding and rejection sampling

- Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding": https://arxiv.org/abs/2211.17192 [V]. 2X to 3X speedup on T5-XXL, and outputs sampled from the same distribution as the standard method (the abstract says identical).
- Chen et al. 2023, "Accelerating Large Language Model Decoding with Speculative Sampling": https://arxiv.org/abs/2302.01318 [V title and year].
- Mechanism: a small model drafts tokens from p. The large model scores them with q. Accept a token with probability min(1, q/p). On rejection, resample from the normalized residual max(0, q − p). The final token is exactly distributed as q.

Fit for "K5 drafts, mixer verifies". Good in structure: K5 is p, the mixer is q. But two caveats (inference).
1. The exactness guarantee targets q, the verifier. The output is the mixer's distribution, not a product. K5's contribution disappears from the final distribution. It only affects speed. So this gives you the mixer's own word distribution, i.e. the byte salad again, just faster. It does not give you a PoE.
2. To get q as a word distribution you need it normalized over all words. The mixer can score any given word, but the normalizer over all words is unknown. Rejection sampling needs normalized q, or at least a bound on q/p.
   - You can do rejection sampling against a target q̃ ∝ p_K5^a · p_mixer^b if you know M with q̃/p_K5 ≤ M. Acceptance is then q̃/(M p_K5). That gives exact samples from the PoE restricted to K5's support. M can be estimated, and an under-estimated M makes the sampler approximate.
   - Cheaper: draw N words from K5, weight each by the mixer ratio, resample one. This is sampling-importance-resampling. It is approximate for finite N and exact as N grows.

Main risk. Exact sampling from the PoE only means you sample the target you specified. It does not mean the target is good (see §2.3 and §2.6).

For CyphaLM (from the record). The mixer's own distribution has never been read as text: `sample` never used the 1.49 stack (Part 1 §1b). Its nearest relative, the byte sampler, writes word salad (0.147–0.174 SCRIBE v2; Part 1 §1a). Exact draft-and-verify against the mixer alone should be expected to throw away what K5 brings. Sampling-importance-resampling over K5's nucleus, with the PMI-form weight, keeps K5 as the base.

---

### 2.3. Decoding algorithms and human ratings

For each: mechanism, source, result, prediction.

- **Nucleus (top-p)**. Holtzman et al. 2020, "The Curious Case of Neural Text Degeneration": https://arxiv.org/abs/1904.09751 [V title and year]. Sample only from the smallest set of tokens whose total probability exceeds p. Finding (from memory; I did not fetch it): beam search and greedy produce repetitive, bland text; pure sampling produces incoherent text from the unreliable tail; human text does not have high per-token probability. Prediction: this describes both of your failures. Byte salad is the untruncated tail. Blandness is the high-probability trap. Apply truncation at the word level, after combining, and at the byte level inside the spelling scorer.
- **Trading off diversity and quality**. Zhang et al. 2020/2021: https://arxiv.org/abs/2004.10450 [V via search; first-guess ID was wrong]. Result: across decoding methods, when quality matters nucleus does best among the evaluated methods; all are similar when diversity matters. It confirms the "likelihood trap": high-likelihood sequences are often low quality. Proposes selective sampling.
- **Typical sampling**. Meister et al. 2022/2023, "Locally Typical Sampling": https://arxiv.org/abs/2202.00666 [V title and year]. Keep the tokens whose surprisal is closest to the model's expected surprisal (entropy). Claimed to match nucleus quality with less repetition (from memory; numbers unverified). Prediction: a natural fit because you know your bits. The target is the corpus's bits per word.
- **Mirostat**. Basu et al. 2020/2021: https://arxiv.org/abs/2007.14966 [V title and year]. A feedback loop adjusts the truncation to keep the surprisal of the output at a target. Prediction: good fit for your "bits improve, text blandens" problem. Fix a target bits-per-word of generated text near that of the real corpus (K5's 1.5 bpc on its corpus). The loop prevents the sharpening drift from stronger experts. Risk: it controls the average, and not the quality of individual words.
- **Min-p**. Nguyen et al. 2024/2025: https://arxiv.org/abs/2407.01082 [V]. Cutoff is a fraction of the top token's probability. Abstract: improved quality and diversity at high temperature on GPQA, GSM8K, AlpacaEval Creative Writing; human evaluators preferred it. I have not checked later critiques of this paper; treat its human-eval claim as one paper's result.
- **Eta / epsilon sampling**. Hewitt et al. 2022, "Truncation Sampling as Language Model Desmoothing": https://arxiv.org/abs/2210.15191 [V title and year]. Idea: the model smooths probability mass onto the tail, so truncate by an absolute probability floor (epsilon) or an entropy-dependent floor (eta). Prediction: strongly relevant. Your byte mixer is a smoothed model with a thin uniform-ish floor, and word probabilities multiply byte floors. An absolute floor on the word-level combined probability is the cheap equivalent.
- **Beam search**. Meister et al. 2020, "If beam search is the answer, what was the question?": https://arxiv.org/abs/2010.02650 [V title and year]. Argues beam search works for translation because it enforces uniform information density, not just because it finds high-probability strings. Together with Holtzman it says: max-likelihood search is wrong for open-ended text. Do not use beam search over your PoE.
- **MBR decoding**. Freitag et al. 2022, "High Quality Rather than High Model Probability: Minimum Bayes Risk Decoding with Neural Metrics": https://arxiv.org/abs/2111.09388 [V]. Result: model probability and human-judged quality correlate only weakly; MBR over samples with BLEURT gave significant human-evaluation gains; selected outputs have much lower model likelihood. Suzgun et al. 2022, "Follow the Wisdom of the Crowd: Effective Text Generation via MBR Decoding": https://arxiv.org/abs/2211.07634 [V]. Result: 3 to 7 points ROUGE/BLEU gains on summarization, data-to-text, translation, style transfer; open-ended generation also motivated. Prediction: for sentence-level overgeneration, pick the sample that agrees most with the other samples under a similarity (e.g. n-gram overlap), not the one with the highest likelihood. Risk: MBR favours the consensus (the bland middle). Use it for robustness, not for liveliness.
- Also relevant (not fetched): Ippolito et al. 2019, "Comparison of Diverse Decoding Methods from Conditional Language Models", https://arxiv.org/abs/1906.06362 [U]. MAUVE for evaluating open-ended output: Pillutla et al. 2021, https://arxiv.org/abs/2102.01454 [V title and year].

---

### 2.4. Retrieval for count models; cache LMs; copy mechanisms

- **Cache LMs**. Kuhn and De Mori 1990, "A cache-based natural language model for speech recognition", IEEE TPAMI [U, no link]. Mixture of a static n-gram and a model of recent words. Grave et al. 2016/2017, "Improving Neural Language Models with a Continuous Cache": https://arxiv.org/abs/1612.04426 [V title and year]. A cache of recent hidden states and next words, mixed in with the base LM at test time. I did not fetch the numbers. Prediction: priming the learner on the prompt and article is the classic cache LM; it is known to help perplexity by capturing topical repetition (rare words that recur). Your online mixer already behaves like a cache (adaptive state), so the byte mixer primed on an article is a strong cache. Risk: caches favour copying. Sampled text may quote the article. The same effect gives you hallucinated-looking copies of names, which is fine for coherence but bad for novelty.
- **RETRO**. Borgeaud et al. 2021/2022, "Improving language models by retrieving from trillions of tokens": https://arxiv.org/abs/2112.04426 [V]. Retrieval database of 2 trillion tokens; comparable to GPT-3 and Jurassic-1 on the Pile with 25 times fewer parameters (abstract). Mechanism: the input is split into chunks, nearest chunks are retrieved, and chunked cross-attention conditions on them. Prediction: confirms that retrieval gives a perplexity gain. The abstract gives no generation-quality claim, and I did not verify one. The neural cross-attention has no direct count-model counterpart. The nearest equivalent for you is priming (the learner state sees the chunk).
- **Copy mechanisms**. Pointer-generator (See et al. 2017, from memory [U]). They mix a generation distribution with a copy distribution. Inference: your byte match models in the mixer are copy mechanisms. They give the largest bit savings and are also the main source of repetition in sampled text, because a sampled continuation of a copied run keeps matching itself.
- Main risk across this topic: Wang 2023 (above) found retrieval entropy grows as generated context drifts from real text. Priming on a real retrieved article helps most at the start and decays as the generated text replaces it.

For CyphaLM (from the record). Here the record is more positive than Wang 2023. Retrieval as a count overlay in K5 (the nearest article's n-grams, about a third of each order's mass) was the second-largest reader gain measured: 0.860 and 0.880 against HLT (§13.66). The decay Wang predicts also shows: refreshing the overlay every 500 bytes (RG) leaned better on long text but did not pass (0.620 [0.440, 0.780], §13.68). Copying rose with retrieval strength (copy32 0.232 → 0.275 → 0.501). An unfrozen cache that reads the writer's own output looped (GEN-C) and did not move readers (§13.65).

---

### 2.5. Planning, overgeneration, reranking

- **Overgenerate and rank**. Langkilde and Knight 1998, "Generation that exploits corpus-based statistical knowledge" (the Nitrogen system), COLING-ACL 1998 [U, no link verified]. A grammar produces a lattice of possible realizations; an n-gram LM picks the best path. It established that a cheap generator plus a statistical ranker works. Caveat: it picks the max-likelihood path, which is right for sentence realization with fixed content and wrong for open-ended text (Holtzman, Freitag).
- **Noisy channel and reranking**. From memory: noisy-channel reranking combines p(y|x) and p(x|y) p(y). I did not verify a specific paper. Best-of-N with a scorer is the general form. The danger is over-optimizing the scorer (Goodhart): the more samples, the more the pick exploits the scorer's blind spots. No verified number here.
- **Plan and write**. Yao et al. 2019, "Plan-And-Write: Towards Better Automatic Storytelling": https://arxiv.org/abs/1811.05701 [V title and year]. First generate a storyline of keywords, then write the text conditioned on it. Reported better story quality (from memory; I did not verify the numbers). Prediction: for K5, a keyword or sentence-topic plan from the retrieved article is the cheap way to get global coherence that neither sampler provides.
- **Repetition**. Welleck et al. 2019, "Neural Text Generation with Unlikelihood Training": https://arxiv.org/abs/1908.04319 [V title and year]. Training-time penalty for repeated tokens, which a count-model decoder cannot use. A decode-time analogue is a repetition penalty or an n-gram block on the combined distribution. Holtzman 2020 notes repetition is self-reinforcing: the more a phrase repeats, the more probable another repeat (from memory).

---

### 2.6. The calibration problem: lower perplexity, worse samples

Evidence that matches your experience.
1. Wang et al. 2023 (kNN-LM): perplexity improves, MAUVE and human ratings do not. Verified in abstract summary.
2. Zhang et al. 2020/2021: the "likelihood trap", high-likelihood outputs are often low quality. Verified.
3. Freitag et al. 2022: model probability and human quality correlate weakly; the best outputs have much lower likelihood. Verified.
4. Holtzman 2020: human text sits in a middle band of probability, not at the mode. From memory.

Mechanism (inference). Perplexity is an average over all positions. A new expert (match model, indirect context, SSE stage) lowers it by sharpening the distribution in predictable spots (copied strings, boilerplate, suffixes). At sampling time, these sharpened spots are sampled correctly but are not where the interesting choices are. At open choice points, the new expert adds no information and may slightly flatten or sharpen. Then after blending into a sampler, the sharper distributions combine with truncation to give more mode-seeking output, which reads as bland. In addition, a lower-bits model is trained to be right on typical text, not to be a good generator when conditioned on its own previous samples.

Implication. Use a different dev metric from bits: a generation metric (MAUVE-like divergence, distinct-n, repetition rate, human or pairwise-judge preference), computed on samples at fixed truncation. Your pairwise blind packets (W1b vs N7/G16) already do the right thing. The literature (Pillutla 2021, Zhang 2021) supports a frontier view: report quality vs diversity at several truncation settings, not one setting.

---

### 2.7. Ranked design implications for the decoder

1. Truncate after combining, at the word level, using nucleus, epsilon, or typical rules. The literature agrees that untruncated sampling gives incoherent tails and that the byte salad is this. Use an absolute probability floor (eta/epsilon) since your models are heavily smoothed.
2. Use fractional PoE weights (sum near 1) or contrastive form, not a plain product with weights 1 and 1. A product of two models that both predict words double-counts and sharpens. Tune the weights on a generation metric, not on bits.
3. Add a closed-loop target on bits per word (Mirostat-style or typical sampling), with the target at the real corpus's rate. It counters the sharpening that stronger experts cause.
4. Treat K5 as a proposal and the mixer as a verifier through self-normalized resampling with N candidates, or rejection sampling with an estimated bound M. Know that exact speculative-style verification gives the verifier's distribution, not a PoE.
5. Evaluate on sample quality (MAUVE, repetition, pairwise human or judge), not bits. Report quality-diversity frontiers.
6. For sentence-level overgeneration, rerank with MBR or typicality rather than max-likelihood, to avoid the likelihood trap and Goodhart effects on the scorer.
7. Use priming as a cache, and optionally CFG-style guidance (primed over unprimed) for the topic. Watch for copying and for the drift of retrieval reliability as generation lengthens.
8. Add a cheap global plan (topic keywords or a sentence outline from the retrieved article) if local sampling lacks coherence.

---

### 2.8. Where the literature contradicts or warns against the planned design

1. Exact byte-spelling scores in a plain product. The product of byte probabilities is a full word probability, not just a spelling check. It includes word choice. So it double-counts with K5. Also, byte products favour short words unless the end-of-word byte is included. Inference, not from a paper. The literature on product and contrastive forms (Hinton; Li 2022; Sanchez 2023) says weights and plausibility constraints decide everything.
2. Arithmetic decoding of the combined distribution. It is only temperature-1 ancestral sampling. Nothing in the literature supports this being good without truncation (Holtzman; Zhang; Hewitt). The "reverse of compression" framing gives exactness, not quality. Delétang et al. state the construction but I did not verify any claim of good samples from it.
3. Speculative-style exactness. Exact only to the verifier. If you want the PoE target, you need the normalizer or a bound, otherwise it is approximate.
4. Trusting bits as the selection metric. Directly contradicted by Wang 2023, Zhang 2021, Freitag 2022 (all verified above). This matches your experience that bits-improving experts blur samples.
5. Priming on a retrieved article. kNN-LM and cache findings say it helps perplexity and often helps topicality, but Wang 2023 found no generation-quality gain and retrieval entropy rising as text grows. Expect verbatim copying and a decaying effect. Not a contradiction of the plan, but a warning that it may not show up in human ratings.
6. Max-likelihood reranking (Nitrogen-style) and beam search. Both are warned against for open-ended text (Holtzman 2020; Meister 2020; Freitag 2022).
7. Unverified items to avoid relying on: the sample quality of gzip/PPM/PAQ generation in Delétang et al.; any generation result in infini-gram; the numbers for Klakow 1998, Nitrogen, Plan-and-Write, CFG repetition effects, typical-sampling gains.
