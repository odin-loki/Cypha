# Coherent text: the research and development plan for CyphaLM's generation

**Odin Loch** · 2026-11-01 · **the plan for generation** · PLAN ledger 98 · follows [`reports/GEN_TRACK.md`](reports/GEN_TRACK.md) and [`TOKENS_PLAN.md`](TOKENS_PLAN.md)

Odin's aim: CyphaLM writes coherent text, not only answers. This plan looks at everything tried so far and every source of data we hold. It sets out what to do next, in order, with measures, gates and predictions written before any run.

---

## 0. In one page

**Where we are.** CyphaLM's text is word salad with real words. On GEN-A's frozen test prompts:
- The best measured sampler, words with Kneser–Ney and a nucleus (T4c's NRK0), scores SCRIBE v2 **0.187 [0.155, 0.217]**.
- Today's byte default (byte Kneser–Ney, similar-context, copy) scores 0.147 [0.115, 0.181] at T 0.6.
- Sentences shuffled across documents score 0.344, and human text 0.923.

**What holds it back.** Sentence form. Of SCRIBE's distance D ≈ 23 on the default's samples, sentence form (D_S) is about 18. Every fix tried so far moved one axis and cost another:

| Fix | Helped | Hurt |
|---|---|---|
| Clean text | took the markup out | sentence form unchanged |
| Sentence gate | sentence form | topic |
| Topic cache | topic | sentence form |
| Tokens | whole words | more invented subword words |

Coherence gains (the cache cut the coherence distance by 23–35%) stay invisible while sentences are broken.

**Why.** CyphaLM sees about 4 bytes (0.6 of a word) through its trie, and about 16 bytes (2 words) through its long-match experts. Nothing in it plans a sentence, and nothing decides what the next sentence is about.

The data limits it too:
- Generation trains on 7 MiB of Wikipedia XML.
- The 1 TB corpus holds no Wikipedia.
- An order-4 byte trie is already full after about 1 GB of text of one kind.

SCRIBE's own model ladder sets the bar. Count n-gram models of orders 1–5 score 0.001–0.089, and GPT-2 small scores 0.767.

**The plan.** Five phases, each a rung of a generation ladder (§1), each with gates.

| Phase | Rung | What |
|---|---|---|
| 0 | — | Foundations: clean prose data, a second benchmark, judges that cannot be gamed, the token model inside `sample` |
| 1 | G2 | Grammatical sentences: word-level models on 10–1,000 times more clean prose; a part-of-speech skeleton model; overgenerate and rank |
| 2 | G3 | Paragraphs that stay on topic: word-level topic cache, entity continuity, sentence-to-sentence coherence in the ranker |
| 3 | G4 | Content: a fact plan realised at the word level, fact checks against the store, "doesn't know" in generation |
| 4 | — | Learning: Cypha's classifier learns the decoding and ranking choices from reward, refereed by LCI |
| 5 | — | Scale on Odin's PC: 100 GB+ word models, packed, with a domain router |

**The honest expectation.**
- Phase 1 alone should take the score from 0.15–0.19 to about 0.25–0.35, near the shuffled-sentence rung.
- Phase 2 is where coherence starts to count.
- Getting past GPT-2 small (0.77) without a neural network at runtime is not something anyone has shown. We will measure how far counts, skeletons, plans and ranking go, and say so at each rung.

**First steps** (§11): build the clean prose corpora on Odin's PC; the axis table and paired re-read of the runs we already have; the token model inside `sample`; the data-scaling read (CT1.1).

**Odin's calls** (§12): the paired per-prompt gate; neural yardsticks or released neural text; the verbatim copy cap at scale; the RAM budget for generation.

---

## 1. Goal and how it is measured

**Goal.** CyphaLM writes text a reader takes as coherent English on the prompt's subject. The rules stay:
- CyphaLM writes every byte (rule 10).
- There is no neural network at runtime (PLAN §1).
- Not knowing is said, never filled with a fluent guess (rule 9).

### 1.1 The generation ladder

A rung is claimed only when its test passes on a frozen test split, read once.

| Rung | Ability | Test | Status |
|---|---|---|---|
| G0 | Real words, no structure | SCRIBE v2 < 0.1; invented words < 5% | **Have** (byte default 0.115 pooled, invented 3.1%) |
| G1 | Locally grammatical: phrases of 3–5 words read well | SCRIBE ≥ 0.20; invented words < 1%; D_S at least 25% below today's | Near (NRK0 0.187, but invented 6.9%) |
| G2 | Grammatical sentences, unrelated to each other | SCRIBE ≥ 0.30, near the shuffled-sentence rung (0.344); D_S within human range in at least half the samples | Not yet |
| G3 | Paragraphs that stay on the prompt's subject | SCRIBE ≥ 0.50; D_C at most 1.5 times the human median; Odin's blind ratings prefer it to G2 text | Not yet |
| G4 | Coherent and factual | SCRIBE ≥ 0.70 (GPT-2 small: 0.767); fact precision ≥ 0.9 on statements the store can check; no contradictions with the store | Not yet |
| G5 | Human-like | SCRIBE ≥ 0.90 (human 0.923); Odin cannot tell it from human text above chance | Stretch |

### 1.2 The measures, always reported together

- **SCRIBE v2** with its 95% interval: pooled, and for each temperature.
- **Its axes:** D_L (word validity), D_S (sentence form), D_C (coherence), D_I (integrity).
- **Surface S.**
- **Continuation bits per byte.** The human continuation scored under the sampler's mixture, paired.
- **Copy share.** The share of generated bytes in a 32-byte span of the training text.
- **SCRIBE's i1.** Word 8-grams found in SCRIBE's pack.
- **Invented-word rate.**
- **Mean length kept, cost** (CPU per byte), **and peak RAM.**
- **Answers.** `qa` must not change: generation tools stay off inside answers.

### 1.3 Judges that cannot be gamed

Several steps below add scorers to the sampler (skeleton models, coherence, ranking). If a scorer reads the same statistics as SCRIBE, the text can learn to please SCRIBE without getting better. So:

1. **SCRIBE's data never enters generation.** SCRIBE's pack (enwik8 [16, 48) MiB), its reference ([48, 56)), its development ladders ([56, 72)) and GEN-A's benchmark ([72 MiB, end)) are kept out of every generation training text. They are also kept out of any scorer the sampler uses.
2. **Different definitions.** A sampler-side scorer may not use SCRIBE's skeleton classes or its vectors (`vec32` on the pack). Ours use the E1 tagger's UD tags and vectors trained on other text.
3. **A second benchmark** in another genre (CT0.3), scored against its own reference. A gain must show on both.
4. **Odin's blind ratings** at each rung: 20 prompts, two texts each, unlabelled and in random order. Which reads better, or are they the same? About 10 minutes. This is the final check of a rung. It is not a gate on single changes.
5. **An outside yardstick**, if Odin allows it (§12): GPT-2 perplexity of the generated text, offline, reported only.

---

## 2. Where we are (measured)

**The reference rungs** (SCRIBE v2, GEN-A's ladder and SCRIBE's study):

| Text | SCRIBE v2 |
|---|---:|
| Human Wikipedia continuations | 0.923 |
| GPT-2 top-k 40 (released outputs) | 0.968–0.972 |
| GPT-2 pure sampling, small to XL | 0.706–0.826 |
| Sentences shuffled across documents | 0.344 |
| Count n-gram models, orders 1–5 (SCRIBE study) | 0.001–0.089 |
| Words shuffled | 0.003 |

**CyphaLM** (GEN-A's prompts unless noted):

| Sampler | Split | SCRIBE v2 [95%] | Surface S | Copy share | Invented words |
|---|---|---|---|---|---|
| Trie only, 8 MiB (1 TB report) | dev | 0.020 [0.014, 0.027] | 0.737 | 0.003 | — |
| Trie only, 1 TB | dev | 0.028 [0.021, 0.035] | 0.796 | 0.000 | — |
| Default before byte Kneser–Ney | test, T 0.6 | 0.106 [0.078, 0.135] | 0.901 | 0.056 | 0.024 |
| **Default today** (byte Kneser–Ney) | test, T 0.6 | **0.147 [0.115, 0.181]** | 0.926 | 0.077 | 0.014 |
| Default today, pooled over T 0.6 / 0.8 / 1.0 | test | 0.115 [0.098, 0.133] | 0.893 | 0.046 | 0.031 |
| Full stack with the 1 TB trie | dev, pooled | 0.113 [0.097, 0.131] | 0.892 | 0.036 | — |
| **Words: BPE 4,096, Kneser–Ney, nucleus 0.9, repeat ban (NRK0)** | test, T 1.0 | **0.187 [0.155, 0.217]** | 0.809 | 0.065 | 0.069 |

**Where the distance comes from** (the default's dev samples before byte Kneser–Ney, GEN-A):
- D_S (sentence form) 18.2;
- D_C (coherence) 4.7;
- D_I 0.0.

SCRIBE = exp(−D/5), so D must fall to about 5.3 for the shuffled-sentence rung and to about 0.4 for human text. Sentence form has to come first.

---

## 3. Everything tried, and what each taught

| Item | Change | Result | What it taught |
|---|---|---|---|
| A4b | Similar-context expert in `sample` | SCRIBE 0.014 → 0.061 | Longer-range byte statistics help a lot from a very low base |
| B4b / B4c | Class and copy experts | BPC −0.08 to −0.09 in `qa`'s holdout; SCRIBE about 0.06 | Better bits do not mean better text |
| GEN-A | Clean prose as training text (`--clean`) | Dev 0.079 → 0.069 (not adopted); copy 0.041 → 0.005; surface S +0.023 | The markup was a small part of the problem. Sentence form is the rest |
| GEN-B | Best of 4 sentences by fluency − markup | Test 0.072 → 0.090 (overlap); D_S −8%, D_C +10%; 4.1× CPU | Ranking fixes form but not topic. The candidates are too poor and too few |
| GEN-C | Document cache and triggers (bytes) | Dev ≤ 0.060 against 0.063; BPC −0.043; D_C −23 to −35% | Topic works, but topical words land in broken byte syntax. An unfrozen cache loops on its own output |
| GEN-C2 | The cache as a default in `context8k` and `qa`'s holdout | BPC better (holdout 1.6983) | Kept for prediction, not for `sample` |
| GEN-E | Content-plan expert from the fact store (bytes) | Test 0.072 → 0.071; on covered prompts BPC −0.020 | The plan carries knowledge, but a byte expert cannot hold a phrase. It writes the frame and drifts on the value |
| BKN / BKN2 | Byte Kneser–Ney 6:4:2:1 in the mixture | Dev 0.079 → 0.125; test 0.147 at T 0.6 | **The largest single gain.** Smoothing and context length matter |
| TEMPS | Temperatures below 0.6 | 0.5 ties 0.157; 0.4 falls to 0.125; copy rises 0.069 → 0.130 | T 0.6 is the byte sampler's best |
| T0–T2 | Tokenizers by counting; token Kneser–Ney | BPE 4,096: 1.905 BPC against the byte model's 2.073; byte Kneser–Ney order 12: 1.883 | On 8 MiB, the gain is smoothing and length, not tokens. Word-start boundaries matter. Small vocabularies win at this size |
| T3 | Exact byte–token bridge | Exact to 5e-16 | Token and byte experts can share one mixture |
| T4 / T4b | Whole-token sampling | Test 0.112 / 0.115 against 0.106 (overlap) | Tokens alone do not lift SCRIBE |
| T4c | Tokens + nucleus 0.9 + repeat ban, no ∞-gram | Test **0.187** against 0.106 (gate met) | The nucleus cut is the lever. The ∞-gram copies |
| T4d | T4c against the byte default with Kneser–Ney | 0.187 against 0.147 (overlap); the byte model predicts 0.014 BPC better | Byte Kneser–Ney caught up. Tokens invent 5 times more words (subword pieces) |
| Track A | 1 TB order-4 byte trie | Alone 0.020 → 0.028; in the stack 0.135 → 0.113 (overlap) | More of the same short context does not help generation. Within a domain, order 4 is full after about 1 GB |
| hp gate24 + word lookahead (2026-07, neural comparison) | Sampling with word lookahead | 99.4% real words; most probable text under three judges; far from fluent | Finishing whole words fixes invented words, not sentences |

**Five lessons.**
1. **Smoothing and reach are what have worked** (Kneser–Ney, similar-context). Every step that added reach raised SCRIBE.
2. **Sentence form is the bottleneck.** No change has moved D_S by more than 10%.
3. **Axes trade off at the byte level.** Topic, plan and gate signals fight broken syntax. They need a word-level writer to land in.
4. **The data is small and the wrong shape.** Word statistics from 7 MiB are sparse. More clean Wikipedia prose is the cheapest untested lever.
5. **Decoding matters** (the nucleus cut: 0.119 → 0.192 on dev for tokens). But it reshapes a model. It cannot add structure the model lacks.

---

## 4. Diagnosis: why the text is not coherent

1. **Reach.** The trie's 4 bytes and the experts' 16 bytes are 0.6 and 2 words. A grammatical English sentence needs agreement across 5–15 words (subject and verb, clause structure). Count models smooth over short contexts and cannot see that far.
2. **No sentence plan.** Nothing chooses a sentence's shape (subject, verb, object, clauses) before its words. N-gram text is locally plausible and globally shapeless. That is what D_S measures.
3. **No discourse plan.** Nothing decides what the next sentence is about, or keeps the entities of the prompt. That is D_C, hidden today behind D_S.
4. **Sparse statistics.** 7 MiB gives 1.19 M words and 66,939 distinct words, 32,092 of them seen once. A word 5-gram model needs orders of magnitude more text before its counts generalise. Unlike the byte trie's order 4, word models are not full at 1 GB.
5. **Wrong data shape.** The prompts are clean Wikipedia prose. The training text is Wikipedia XML with markup (GEN-A), and the 1 TB corpus is web, code, papers and Q&A with no Wikipedia.
6. **One unit at a time.** Byte sampling makes 5–7 choices a word, and tokens 2–3. Each choice is a chance to drift (TOKENS_PLAN §1).

**What follows.** Fix sentence form first, with more clean prose at the word level, an explicit skeleton of each sentence, and overgenerate-and-rank (Langkilde and Knight, 1998, did exactly this with n-grams). Then add the topic and plan signals that already work in bits but fail at the byte level. Then content.

---

## 5. Rules for this plan

- **Rule 10.** A component may add training text, add an expert to CyphaLM's mixture, propose candidates that CyphaLM's own distribution accepts, or gate and rank CyphaLM's own candidates. CyphaLM writes every byte. Nothing pastes text CyphaLM did not generate.
- **No neural network at runtime.** Neural text or neural scores enter only offline and only with Odin's ruling (§12).
- **Register first.** Rules and predictions are committed before any run. Choices are made on dev. Each change reads a frozen test split once, logged in GEN_TRACK's Reads table.
- **The gate for a default.** SCRIBE v2 up with disjoint 95% intervals (or the paired per-prompt gate, if Odin allows it, §12), and copy share at most the comparator's + 0.02. Surface S, BPC and the axes are reported with it. Answers unchanged.
- **Judges that cannot be gamed** (§1.3).
- **Kill criteria.** A line of work that misses its gate twice stops (PLAN rule 6), with a report.
- **Cost.** Heavy builds and runs are scripts that run on Odin's PC without a Claude session (`scripts/…`, outputs on `D:`). Claude time goes on design, code and reading results.
- **The trainer files.** `native/tools/trie_stream_train.cpp` and `scripts/run_train.sh` are not touched without Odin's word.

---

## 6. Data

### 6.1 What we hold

| Data | Size | Where | Use for generation | Notes |
|---|---|---|---|---|
| enwik8 | 100 MB Wikipedia XML (2006) | cloud, PC | Today's training slice [0, 8 MiB); SCRIBE's regions [16, 72) MiB; GEN-A's benchmark [72 MiB, end) | Only [0, 16) MiB is free for training |
| enwik9 | 1 GB Wikipedia XML (2006) | PC | [100 MB, 1 GB) is free: about 400–500 MB of clean prose after SCRIBE's cleaner | The first 100 MB is enwik8. Keep it out |
| English Wikipedia dump (2026) | 5.9 GB `.part`, bz2 | PC | About 20 GB of clean modern prose once finished and read | Needs a bz2 reader. Holds modern versions of the benchmark's 500 pages: drop them by title |
| 1 TB corpus | C4 625 GB, GitHub 224 GB, arXiv 94 GB, Stack Exchange 80 GB, 40 books | PC | C4 is web prose (the second benchmark's genre); books are clean narrative | Non-ASCII was dropped. Code and LaTeX must stay out of a prose model |
| Gutenberg | 40 books | PC | Narrative prose | More can be fetched (gzip requests work) |
| Dolly 15k | chat | cloud, PC | Not prose; the answer models | — |
| F2 fact store, Wikidata tables (D4, S15, S8e), DBpedia infobox, GeoNames | — | cloud | The content plan (GEN-E, CT3) and fact checks | — |
| UD English EWT | treebank | cloud | Trains the tagger behind the skeleton model (CT1.4) | E1 tagger 92.1% UPOS; perceptron 94.8% |
| PPMI-SVD vectors, similar-context keys | — | cloud | Coherence scorers (CT2.3) | SCRIBE's `vec32` is not used by the sampler |
| Packed 1 TB trie | 992 MB | git LFS | An expert in the stack | Did not help generation |
| GEN-A benchmark, SCRIBE ladders, GPT-2 output dataset, HANNA / OpenMEVA / MAUVE judgments | — | cloud | Evaluation only | Never trained on |

### 6.2 What to build (CT0.4)

A script on Odin's PC writes clean prose corpora, one document per line, with a manifest (source, licence line, sha256, bytes):
1. **W1:** enwik9 [100 MB, 1 GB), through SCRIBE's cleaner.
2. **W2:** the 2026 English Wikipedia: finish the download, add a bz2 reader, clean it, and drop every page whose title is one of GEN-A's 500 benchmark pages or lies in SCRIBE's regions.
3. **S1:** Simple English Wikipedia (download): short, plain sentences. A stepping stone for grammar.
4. **B1:** Gutenberg books: fetch more, strip headers.
5. **C1:** C4 prose, shards not used by the second benchmark.

Every corpus is checked against the benchmark: no 32-byte span of a benchmark continuation may appear in it (GEN-A's contamination rule). The JSON reader's `\uXXXX` fix (TRIE_1TB_PROFILE "Next" 2) comes first, so accented names survive.

---

## 7. The research programme

Each item is registered before it runs (rule, arms, predictions), chosen on dev, and read on test once. "Where" says what runs it: the cloud container, or Odin's PC by script.

### Phase 0: foundations

| ID | Item | Why | Measure / gate | Where, cost |
|---|---|---|---|---|
| CT0.1 | **Axis table** for every arm already run (D_L, D_S, D_C, D_I, per temperature), from the saved SCRIBE exports | Know which axis each change moves; target D_S | Report only | Cloud, minutes |
| CT0.2 | **Paired per-prompt SCRIBE.** Recompute T4d's BK0 against NRK0, prompt by prompt, from the saved samples | Two independent intervals waste power; the arms answer the same prompts | Report the paired difference and its interval. The gate change is Odin's call (§12) | Cloud, minutes |
| CT0.3 | **A second benchmark**: web prose prompts and continuations from an unseen C4 shard, a SCRIBE reference from other unseen shards, frozen 250 + 250 rows | A gain must not be Wikipedia-only or SCRIBE-specific | SCRIBE's control ladder must pass on it (5 of 5, as WebText did) before use | PC script + cloud check |
| CT0.4 | **Clean prose corpora** W1, W2, S1, B1, C1 (§6.2) | Lesson 4: the data is small and the wrong shape | Manifest, hashes, contamination check | PC script, hours |
| CT0.5 | **Odin's blind-rating sheet**: 20 prompts, A/B, random order, written to a file he fills in | The final check of each rung | — | Cloud, minutes; Odin, 10 min a rung |
| CT0.6 | **The token model inside `cyphalm_trie sample`** (T4c's integration step): the token Kneser–Ney model and the bridge as part of `sample`, with the byte experts beside them | Every later word-level step needs one sampler | Without the flag, files byte-identical; with it, T4c's NRK0 reproduced; RAM measured | Cloud, the largest code item |

### Phase 1: grammatical sentences (target G2)

| ID | Item | Why, evidence | Measure / gate | Where, cost |
|---|---|---|---|---|
| CT1.1 | **The data-scaling read.** Word-level Kneser–Ney (BPE with word boundaries, vocabulary grown with the data) trained on W1 at 7 MiB, 64 MiB and 512 MiB, then W2 at 4 GB and 20 GB. Each sampled with T4c's decoder on GEN-A's dev split | Lesson 4. Large n-gram models get more fluent with data (Brants et al., 2007). Order 4 bytes saturate; word 5-grams should not | SCRIBE, D_S, BPC and invented words against size, with a fitted curve. No gate: it sets the data for every later step | PC script; the 20 GB build needs GBs of RAM |
| CT1.2 | **Words and bytes in one mixture.** Token Kneser–Ney + byte Kneser–Ney + similar-context + copy through the bridge, sampling whole tokens (T10's mixture, for generation) | T4d: the byte experts predict better, the token sampler writes better. Each lacks the other's strength | Gate against the better of BK0 and NRK0 | Cloud after CT0.6 |
| CT1.3 | **Superwords** (T5: merges across spaces, SuperBPE / BoundlessBPE) | Each context order reaches further | SCRIBE over plain tokens, gate | Cloud or PC |
| CT1.4 | **The skeleton model.** Tag the training prose with the tagger (UD tags). Learn a tag n-gram (order 6–8: only 17 tags, so long contexts are dense). Generate words from a factored model, P(word \| previous words, its tag) × P(tag \| previous tags), with generalised backoff (Bilmes and Kirchhoff, 2003, factored language models) | Lessons 2 and 3. A sentence's shape lives in its tags. 17 tags give dense counts at orders a word model cannot reach | D_S at least 20% below CT1.2's, and SCRIBE up with the gate. The tagger and its tags are independent of SCRIBE's skeleton classes | Cloud for design; PC to tag W1 / W2 |
| CT1.5 | **Overgenerate and rank, version 2.** For each sentence, k candidates (8–32) from the CT1.2 / CT1.4 sampler, ranked by the word model's length-normalised log-probability + the skeleton model + a markup penalty. The weights are chosen on dev (Langkilde and Knight, 1998) | GEN-B's best of 4 at the byte level moved D_S −8%. Word candidates are better, and more of them can be drawn | Gate, cost per byte reported. The ranker uses no SCRIBE statistic | Cloud |
| CT1.6 | **Decoding at the word level.** Nucleus, min-p, locally typical sampling (Meister et al., 2023), contrastive search with count vectors (Su et al., 2022), temperature and the repeat ban, on CT1.2's model | The nucleus cut was T4c's lever. Contrastive search penalises the degenerate repetition count models fall into | Dev grid, one test read for the chosen setting | Cloud |
| CT1.7 | **Phrase drafts** (T7). The ∞-gram over W1 / W2 proposes several tokens. CyphaLM's mixture accepts each token or stops the draft (speculative style) | "Copy is all you need" (Lan et al., 2023): phrase units read fluently. The acceptance keeps rule 10 | Gate; copy share and SCRIBE's i1; a verbatim cap (§12) | Cloud, after CT1.1 |

### Phase 2: paragraphs that stay on topic (target G3)

| ID | Item | Why, evidence | Measure / gate | Where |
|---|---|---|---|---|
| CT2.1 | **The topic cache at the word level** (T6; GEN-C moved to words): cached content words and trigger pairs (Rosenfeld, 1996), frozen to the prompt so they cannot loop on their own output | GEN-C cut the coherence distance by 23–35% but broke byte syntax. At the word level a cached word is written whole | Gate; D_C | Cloud |
| CT2.2 | **Entity continuity.** Track the prompt's entities (the tagger's proper nouns and E2's entities). An expert learns from W1 / W2, by counts, how entities carry across sentences (the entity grid, Barzilay and Lapata, 2008; centering, Grosz et al., 1995): re-mention, pronoun, subject or object | Wikipedia paragraphs keep their subject; ours lose it within a sentence | Gate; the share of sentences that mention a prompt entity, against human continuations | Cloud |
| CT2.3 | **Coherence in the ranker.** Add sentence-to-sentence similarity to CT1.5's ranker, with vectors trained on W1 (not SCRIBE's `vec32`) | Lexical cohesion (Halliday and Hasan, 1976) is what makes a paragraph one paragraph | Gate; D_C; the second benchmark must agree | Cloud |
| CT2.4 | **Sentence openings.** A count model of how a sentence starts, given the previous one (connectives, dates, "He", "The …"), learned from W1 | Discourse structure lives at sentence starts | Gate | Cloud |
| CT2.5 | **Long memory over Wikipedia.** The FM-index ∞-gram over W2 at the word level, for long matches, under the copy cap | Reach of dozens of words where the text is common | Gate; copy share; i1 | PC build, cloud read |

### Phase 3: content (target G4)

| ID | Item | Why, evidence | Measure / gate | Where |
|---|---|---|---|---|
| CT3.1 | **The content plan at the word level** (GEN-E → T9). The store plans facts about the prompt's entity. F2's frames, filled with the entity and value, become phrase drafts that CyphaLM accepts or stops (CT1.7's mechanism) | GEN-E: the plan makes the human text cheaper (−0.020 bits per byte on covered prompts), but a byte expert drifts off the value | Gate; fact precision (CT3.4); coverage of prompts | Cloud |
| CT3.2 | **Fact checks in the ranker.** A candidate sentence that states a fact the store contradicts is dropped. One the store cannot support is hedged or not written | Rule 9 in generation: a fluent wrong fact is worse than none | Contradictions with the store → 0; fact precision | Cloud |
| CT3.3 | **Paragraph plans.** The order of facts and topics learned from Wikipedia's leads (counts over relation order: birth, career, death; place, history, economy) | Classic generation pipelines plan content first (Reiter and Dale, 2000) | Gate; Odin's ratings | Cloud |
| CT3.4 | **Fact precision.** Statements in generated text, found by F2's patterns, graded by S8e's grader against Wikidata | A G4 measure that SCRIBE cannot give (SCRIBE is blind to wrong numbers and negations) | Report; a G4 gate | Cloud |

### Phase 4: learning

| ID | Item | Why | Measure / gate |
|---|---|---|---|
| CT4.1 | **Cypha's classifier learns the decoder.** Ranking weights, when to stop a draft, when to hedge: learned from reward on a dev reward set. Reward = the independent judges of §1.3 (the second benchmark's reference, the skeleton and fact checks), never SCRIBE itself. LCI referees each change | Track G's self-improvement, applied to generation | LCI ≥ 0.95; SCRIBE read on test once |
| CT4.2 | **Learning from Odin's ratings.** Preference counts over his A/B choices feed the ranker's weights | His judgment is the target | Held-out ratings |

### Phase 5: scale on Odin's PC

| ID | Item | Measure |
|---|---|---|
| CT5.1 | Word-level models on 100 GB+ of prose (W2 + C1 + B1), packed like the byte trie (M4), with registered predictions (SCRIBE, bits per byte, distinct n-grams, RAM) | Checkpoints on the CT1.1 curve |
| CT5.2 | A domain router (Track A's A8): Wikipedia, web and book models, chosen per prompt by CyphaLM's own bits | Both benchmarks |

---

## 8. Predictions (registered 2026-11-01)

| # | Prediction | Falsified if |
|---|---|---|
| P-CT1 | CT1.1: dev SCRIBE of the word model rises with data, by at least +0.03 per 8× of clean wiki prose up to 512 MiB, and reaches at least 0.25 at 512 MiB, with copy share within BK0's + 0.02 | it rises by less, or falls |
| P-CT2 | CT1.1: invented words fall below 2% by 512 MiB as the vocabulary grows | 2% or more |
| P-CT3 | CT1.2: the combined mixture beats both BK0 and NRK0 on dev SCRIBE, but by less than 0.04 | it loses to either, or wins by 0.04 or more |
| P-CT4 | CT1.4: the skeleton model lowers D_S by at least 20% against CT1.2 at the same data size | under 20% |
| P-CT5 | CT1.5: ranking 16 word-level candidates adds at least +0.05 SCRIBE over 1 candidate, at about 4–16× the CPU | under +0.05 |
| P-CT6 | Phase 1 reaches G1 on test. It does not reach G2 (0.30) without Phase 2 | G2 reached in Phase 1, or G1 missed |
| P-CT7 | The topic cache and entity continuity (CT2.1–CT2.2) raise SCRIBE only once D_S is within human range in at least half the samples. Before that, D_C falls and SCRIBE stays within ±0.02 | SCRIBE rises by 0.03 or more while D_S is still out of range |
| P-CT8 | The 1 TB byte trie and more C4 web text do not raise SCRIBE on the Wikipedia benchmark, but help on the web benchmark (CT0.3) | they raise wiki SCRIBE with disjoint intervals, or do not help on web |
| P-CT9 | Without a neural network, no arm passes 0.5 (G3) on test before Phase 3 | one does |
| P-CT10 | Answers in `qa` are unchanged by every item here (generation tools stay off inside answers) | any answer changes |

---

## 9. Kill criteria and decision points

- **Phase 1 stops** if CT1.1 shows SCRIBE flat with data (under +0.02 per 8×) **and** CT1.4 misses its gate twice. That would mean counts over words and tags cannot make grammatical sentences at our scale. The next question then goes to Odin: Track H (offline, verified distillation) or a different representation.
- **Phase 2 waits** until G1 is reached. Coherence measured over broken sentences misleads (lesson 3, P-CT7).
- **Any item** that misses its gate twice stops, with its flag kept and a report.
- **Each rung** ends with a dated report, a test read, Odin's blind ratings and the report card.

---

## 10. Order, dependencies and cost

```
CT0.1, CT0.2, CT0.5 (cloud, minutes)          CT0.4 clean corpora (PC script)        CT0.3 web benchmark (PC + cloud)
            \                                         |                                      |
             +------------- CT0.6 token model in sample (cloud) ----------+                  |
                                              |                            \                 |
                                   CT1.1 data scaling (PC) ---------------> CT1.2 one mixture
                                              |                                  |
                          CT1.4 skeleton (tag W1 on PC)   CT1.6 decoding   CT1.3 superwords
                                              \                |                /
                                               +-------- CT1.5 rank ---------+--- CT1.7 drafts
                                                              |
                                                  G1 / G2 test read + Odin's ratings
                                                              |
                               CT2.1 cache, CT2.2 entities, CT2.3 coherence, CT2.4 openings, CT2.5 memory
                                                              |
                                          CT3.1 plan, CT3.2 fact checks, CT3.3 paragraph plans, CT3.4
                                                              |
                                                    CT4 learning, CT5 scale
```

| Item | Machine time | Claude effort | Notes |
|---|---|---|---|
| CT0.1, CT0.2, CT0.5 | minutes | low | From saved runs |
| CT0.3 | an hour | medium | Prompt maker exists (`scribe prompts`) |
| CT0.4 | hours on the PC | medium (one script) | Runs without Claude |
| CT0.6 | — | **high** | The biggest code change; parity and RAM checks |
| CT1.1 | hours to a day on the PC | medium | One script per size; runs without Claude |
| CT1.2, CT1.3, CT1.6 | about an hour per arm | medium | Dev grids |
| CT1.4 | tagging W1 / W2: hours on the PC | high | New model code |
| CT1.5, CT1.7 | 4–16× sampling CPU | medium | |
| Phase 2 | about an hour per arm | medium each | |
| Phase 3 | about an hour per arm | high | Reuses F2, GEN-E, S8e |
| Phase 4, 5 | days on the PC | high | |

**RAM.** `sample` peaks at 232.5 MB today, and the token tool at about 380 MB. Word models on 512 MiB to 20 GB of text need GBs. PLAN §10's budget for the whole language system is 4 GB resident with large stores memory-mapped. Generation's own budget is Odin's call (§12). It does not touch `qa`'s 200 MB rule.

---

## 11. First steps

1. **CT0.4 on Odin's PC:** the clean-corpus script for W1 (enwik9 [100 MB, 1 GB)) first; W2 once the bz2 reader and the download are done.
2. **CT0.1 and CT0.2 in the cloud:** the axis table and the paired re-read of BK0 against NRK0, from saved runs. Minutes, no new runs.
3. **CT0.6:** register and build the token model inside `sample`.
4. **CT1.1:** register its predictions (P-CT1, P-CT2) with the script, then run 7 MiB → 64 MiB → 512 MiB on the PC.
5. **CT0.3** in parallel: the web benchmark from an unseen C4 shard.

---

## 12. What Odin decides

1. **The paired per-prompt gate** (CT0.2). It has more power than two independent intervals. It changes the generation gate, so it is your call.
2. **Neural text or scores, offline** (Track H). Options, each yes or no:
   - (a) GPT-2 perplexity of generated text as an outside yardstick, reported only;
   - (b) OpenAI's released GPT-2 output dataset (no inference) as training text;
   - (c) TinyStories (text written by GPT models) as a narrow-domain stepping stone.

   All are offline. None runs a network inside CyphaLM. (b) and (c) make CyphaLM learn from machine-written text, so they are distillation.
3. **The verbatim copy cap at scale.** With 20 GB of Wikipedia, long matches become possible. Proposed: no generated span of more than 12 consecutive words verbatim from the training text, beside the copy-share rule.
4. **Generation's RAM budget.** Proposed: 1 GB for `sample` at Phase 1, 4 GB (PLAN §10) at Phase 5, with large tables memory-mapped.
5. **Simple English Wikipedia** (S1) as a stepping-stone domain, with its own SCRIBE reference: yes or no.

---

## References

- Brants, Popat, Xu, Och and Dean (2007). Large language models in machine translation. EMNLP.
- Bilmes and Kirchhoff (2003). Factored language models and generalized parallel backoff. NAACL.
- Langkilde and Knight (1998). Generation that exploits corpus-based statistical knowledge. ACL.
- Barzilay and Lapata (2008). Modeling local coherence: an entity-based approach. Computational Linguistics.
- Grosz, Joshi and Weinstein (1995). Centering: a framework for modeling the local coherence of discourse. Computational Linguistics.
- Halliday and Hasan (1976). Cohesion in English. Longman.
- Rosenfeld (1996). A maximum entropy approach to adaptive statistical language modeling. Computer Speech and Language.
- Reiter and Dale (2000). Building Natural Language Generation Systems. Cambridge.
- Holtzman, Buys, Du, Forbes and Choi (2020). The curious case of neural text degeneration. ICLR.
- Meister, Pimentel, Wiher and Cotterell (2023). Locally typical sampling. TACL.
- Su, Lan, Wang, Yogatama, Kong and Collier (2022). A contrastive framework for neural text generation. NeurIPS.
- Lan, Cai, Wang, Huang and Mao (2023). Copy is all you need. ICLR.
- Eldan and Li (2023). TinyStories: how small can language models be and still speak coherent English?
- Chen and Goodman (1999). An empirical study of smoothing techniques for language modeling. Computer Speech and Language.
- Liu, Min, Zettlemoyer, Choi and Hajishirzi (2024). Infini-gram: scaling unbounded n-gram language models to a trillion tokens. COLM.
- The repo's own reports: [GEN_TRACK](reports/GEN_TRACK.md), [GEN_B](reports/GEN_B_SENTENCE_GATE.md), [GEN_C](reports/GEN_C_TOPIC_CACHE.md), [GEN_E](reports/GEN_E_CONTENT_PLAN.md), [TEMPS](reports/TEMPS_BYTE_SAMPLER.md), [BYTE_KN](reports/BYTE_KN.md), [TOKENS_T0_T2](reports/TOKENS_T0_T2.md), [TOKENS_T3_T4](reports/TOKENS_T3_T4.md), [SCRIBE_V2](reports/SCRIBE_V2.md), [CYPHALM_1TB_MODEL](reports/CYPHALM_1TB_MODEL.md), [CYPHALM_VS_NEURAL_LM](reports/CYPHALM_VS_NEURAL_LM.md), [TOKENS_PLAN](TOKENS_PLAN.md).
