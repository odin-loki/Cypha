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

---

## 13. Fuzzy matching: the neural diagnostic and the port (registered 2026-11-01, before any run)

**Odin Loch** · 2026-11-01 · PLAN ledger 99

### 13.1 Odin's rulings on §12

1. **The gate stays as it is:** two independent 95% intervals. No paired gate for defaults.
2. **Neural:** the NN plan below is approved ("Do it"). Small neural models are trained offline as instruments. The port may use a learned linear map at runtime (route 2) and states distilled offline from a neural model (route 3). No neural network runs inside CyphaLM. No Qwen.
3. **The copy cap:** no generated run of more than 12 consecutive words verbatim from the training text, beside the copy-share rule.
4. **RAM:** no hard budget, but keep it down. A change that adds memory is taken only for a big or medium win, meaning its registered gate is met. A near-tie that costs memory is dropped.
5. Simple English Wikipedia: not decided yet.

### 13.2 The idea

Neural language models do fuzzy pattern matching. Each context becomes a learned state, and contexts with similar states get similar predictions, even when the words never occurred together. CyphaLM does exact matching: it needs the exact string, and when the string is unseen it drops the oldest byte.

Our own results point the same way:
- The 1 TB run saturates by about 1 GB, because exact long contexts are too rare to count.
- The two biggest SCRIBE gains both share evidence across contexts: similar-context 0.014 → 0.061, Kneser–Ney 0.079 → 0.125.
- In the earlier neural comparison, the neural models won on structure (punctuation, markup and spaces) and CyphaLM won on letters and copying.

Fuzzy matching has two parts:
- **similarity** (what counts as alike);
- **a state** that carries the whole past (what the match is made on).

N0 measures which part buys more text quality, so the port starts with the right one.

### 13.3 N0: the diagnostic

**Data (W1).** enwik9 bytes [100,000,000, 1,000,000,000), cleaned by SCRIBE's cleaner (`scribe docs`), one page per line.
- A page is dropped if it holds any 32-byte span of a GEN-A dev or test continuation.
- SCRIBE's regions and GEN-A's benchmark are in enwik8 (enwik9's first 100 MB), so they are out by construction.
- **Train:** the first 128 MiB of clean prose. **Val:** the next 4 MiB, for early stopping only.

**Tokens.** One BPE vocabulary with word boundaries, 8,192 tokens, learned by `cyphalm_tok` on the training text and shared by every arm.

**Arms.**

| Arm | Model | Context |
|---|---|---|
| K5 | Modified Kneser–Ney, order 5 (`cyphalm_tok_gen`, T4c's model) | 4 tokens |
| K8 | The same, order 8 (control: counts with a longer window) | 7 tokens |
| M5 | Feedforward net over the same 4 tokens (Bengio et al., 2003): embeddings 256, hidden 1,024, softmax over the vocabulary | 4 tokens |
| TR | Transformer: 6 layers, d 384, 6 heads, pre-LN, tied embeddings, context 256 tokens | 256 tokens |

M5 and TR each train on the same number of tokens (two passes over train), AdamW, cosine schedule. The best checkpoint is kept by val loss.

**Decoding, the same for every arm.** T4c's decoder: nucleus 0.9 and the repeat ban, at T 1.0 and T 0.8. The same 250 GEN-A **dev** prompts, 600 bytes, fixed seeds. No test read: N0 changes no default.

**Measures.**
- SCRIBE v2, pooled and for each temperature, with intervals, and its axes (D_L, D_S, D_C, D_I);
- surface S, invented words and mean length;
- copy share against the 128 MiB training text, and the 12-word verbatim cap;
- bits per byte on val and on the GEN-A dev continuations after their prompts. Every arm is scored in bytes: total bits divided by bytes.

**What it settles.**
- Δsim = M5 − K5 (SCRIBE pooled, dev): what similarity buys at the same window.
- Δstate = TR − M5: what the long state buys.
- Both come with a bootstrap over prompts, paired by prompt (a diagnostic, not a gate).
- The larger one, if its interval excludes 0, is ported first. If neither does, the result is reported and the port waits for Odin.

**What the instruments may not do.**
- They are never part of CyphaLM, and never a default.
- Their text is never training text.
- They are deleted or archived after the report.

**Predictions.**

| # | Prediction | Falsified if |
|---|---|---|
| N1 | Bits per byte on val: M5 below K5 by at least 0.05; TR below M5 by at least 0.15 | either gap is smaller |
| N2 | K8's SCRIBE is within ±0.02 of K5's: counts cannot use the longer window | outside |
| N3 | M5's SCRIBE is at least K5's + 0.05, with D_S at least 15% lower | smaller |
| N4 | TR's SCRIBE is at least M5's + 0.10, mostly from D_S and D_C | smaller |
| N5 | TR's dev SCRIBE is in [0.30, 0.60] | outside |
| N6 | Δstate > Δsim | Δsim ≥ Δstate |
| N7 | The neural arms copy less than the count arms (32-byte copy share) | they copy as much or more |

**Where, cost.** Odin's PC by script: data about 1 h, tokens and K5 / K8 about 1 h, M5 a few hours, TR 4–8 h, sampling and SCRIBE about 1 h. Outputs go to `D:\cypha_llm\nn_diag\`. Code goes on branch `wip/nn-diag` until it is reviewed.

### 13.4 The port (registered when N0 is read)

- **P1, similarity first** (if Δsim wins): fuzzy backoff at the word level.
  - Back off from exact tokens to tags, classes and vectors, not only by dropping the oldest token (factored backoff, Bilmes and Kirchhoff, 2003).
  - Find the nearest training positions by a count-derived state of the last few words: the similar-context expert, moved to words.
- **P2, state first** (if Δstate wins): a running state of the whole past, and next tokens taken from the training positions whose state is nearest. The state is built from:
  - a topic vector;
  - an entity register;
  - the part-of-speech and clause history.

  It is either learned to predict (route 2: a linear map, e.g. CCA between past and future features) or distilled (route 3: TR's hidden states clustered into a few thousand discrete states, with transitions learned as counts).
- **Either way:** exact matching stays for facts and copying, where it is better (rule 9: a fuzzy match blends facts). Fuzzy matching is for form.

### 13.5 Q0: a yardstick from Qwen 3.5 9B (Odin, 2026-11-01, ledger 100)

Odin runs Qwen 3.5 9B locally (Ollama). His ruling: it is used **only as a yardstick**. "No Qwen inference" is lifted for this use alone.
- No table or state is taken from it, no text of it is trained on, and nothing of it enters CyphaLM. Uses B, C and D of the discussion are declined.
- **What is run:** its plain continuations (raw completion, no chat template, thinking off) of the same 250 GEN-A **dev** prompts. Nucleus 0.9 at T 0.8 and T 1.0, cut at 600 bytes, scored by the same SCRIBE ladder, axes and copy share as N0's arms.
- **When:** after N0's training, so it does not compete for the CPU.
- **Caveat:** Qwen has read Wikipedia, likely including modern versions of these pages. Its score is a reference point, not a fair comparison, and its copy share against our training text does not measure its memory.

### 13.6 Fuzzy matching without a network: the candidates

Odin: "There are better fuzzy approaches. There is a plethora of techniques." The ones that fit CyphaLM, by what they port. Each is registered on N0's bench (W1, the shared tokens, the shared decoder, SCRIBE) before it runs, against K5.

| Part ported | Technique | What it does |
|---|---|---|
| Fuzzy keys | **Approximate matching in the FM-index**: backtracking search with up to k substitutions (as in read aligners), substitutions weighted by a learned word-substitution matrix (which words replace each other in the same contexts, from counts) | The ∞-gram finds training text that matches the context with a few words changed |
| Fuzzy keys | **Gapped and skip contexts** (skip-grams; PAQ's sparse contexts) | Matches that skip a word or two |
| Fuzzy keys | **Class and factored backoff** (Brown clusters by the exchange algorithm; generalised parallel backoff, Bilmes and Kirchhoff, 2003) | Back off to the word's class or tag instead of forgetting it |
| Fuzzy keys | **Locality-sensitive hashing** (SimHash, MinHash) of context features | Fast nearest-context lookup |
| Similarity tuned to predict | **Memory-based learning** (TiMBL, IGTree; Daelemans and van den Bosch, 2005): nearest stored examples under a distance whose feature weights are the information gain of each context position for the next word | A learned "what counts as alike", from counts |
| Similarity tuned to predict | **Decision-tree and random-forest language models** (Bahl et al., 1989; Xu and Jelinek, 2004): trees ask questions about any word of the history, chosen to predict | Generalises across contexts, reaches beyond the window |
| Similarity tuned to predict | **Spectral and CCA context states** (Dhillon, Foster and Ungar, eigenwords): an SVD of the covariance of past and future features | Context vectors whose closeness means similar futures (route 2) |
| Similarity tuned to predict | **Indirect contexts** (PAQ): contexts grouped by what followed them recently, not by their spelling | Contexts that behave alike share evidence |
| The long state | **Sparse distributed memory with counters** (Kanerva, 1988). Attention approximates it (Bricken and Pehlevan, 2021) | A count-based content-addressable memory: the closest thing to attention without a network |
| The long state | **Hyperdimensional state vectors** (holographic reduced representations; random indexing, already in `ppmi_svd`): the past bound into one vector with position and decay | A running state of the whole past, by algebra, with no training |
| The long state | **Head-word contexts from syntax** (the structured language model, Chelba and Jelinek, 2000): predict from the exposed heads of the partial parse, using E1's tags and K6's chunks | Grammar across clauses |
| The long state | **Latent-state models** (HMMs at scale, spectral learning: Hsu, Kakade and Zhang, 2009) | A learned discrete state, by EM or spectral methods |

**The first to try, by fit and cost:**
1. Approximate FM-index matching: it reuses our index.
2. Memory-based kNN with information-gain weights.
3. A random-forest language model.
4. Sparse distributed memory and hyperdimensional states, for the long state.

N0's result decides which group goes first.

### 13.7 The fuzzy bake-off (Odin, 2026-11-01: "You can do all of it at once. Try it on 8 MB of data in Python"; registered before any run)

**Tool.** `bench/fuzzy/fuzzy_bake.py`, numpy only, run in the cloud container. It is an instrument for choosing what to build in C++. Nothing in it becomes a default.

**Data.**
- enwik8 [0, 16) MiB, cleaned by SCRIBE's cleaner. That is about 8 MB of prose, outside SCRIBE's regions and GEN-A's benchmark.
- Pages holding a 32-byte span of a GEN-A continuation are dropped.
- Every 20th page is held out (val). Its pages alternate between fitting weights and scoring.

**Units.** Words, numbers and punctuation. The vocabulary is words seen at least twice, up to 32,000. Rarer words are `<unk>`, which is never sampled.

**Arms.** Each fuzzy model is mixed with K5 by one weight fit on val. ALL mixes every model, with weights fit by EM over the models available at each position.

| Arm | Technique (§13.6) |
|---|---|
| K5 | Interpolated modified Kneser–Ney, order 5 (the baseline) |
| F1 | Class substitution: windows of 2–4 words with one word replaced by its class (512 classes, k-means on our own PPMI-SVD vectors) |
| F2 | Gapped word contexts (7 skip patterns) |
| F3 | SimHash buckets of a context vector, at 10 / 14 / 18 / 22 bits |
| F4 | Memory-based learning, IB1-IG: neighbours by information-gain-weighted word and class matches at 4 positions |
| F5 | Random class forest: 6 random partitions of the history. An approximation of Xu and Jelinek: the trees are not grown by likelihood |
| F6 | PAQ indirect contexts (the last two followers of the context) |
| F7 | CCA context states with kNN over 600,000 stored training states |
| F8a | Sparse distributed memory with counters (8,192 locations), addressed by the last 3 words |
| F8b | The same, addressed by a decayed state of the whole past (40 words) beside the last word |
| ALL | All of the above with K5 |

**Measures.**
- Bits per byte on val's scoring half, with each arm's difference from K5 and a bootstrap interval over pages.
- Each model's mixture weight and coverage.
- Generation: GEN-A's 250 **dev** prompts, T 1.0, nucleus 0.9, the 11-byte repeat ban lifted to words, 600 bytes.
- SCRIBE v2 (the same ladder command as `gen_eval.sh`), copy share (32-byte spans of the training text), the share of samples with a 13-word verbatim run, and continuation bits per byte.
- No test read.

**Rule (a screen).** A technique goes forward to C++ if both hold:
- K5 + F lowers val bits per byte with its 95% interval below 0, by at least 0.01;
- its dev SCRIBE point estimate is at least K5's + 0.02, with copy share within K5's + 0.02.

ALL is reported against K5 by the same measures.

**Predictions.**

| # | Prediction |
|---|---|
| B1 | ALL lowers val bits per byte against K5 by at least 0.05 |
| B2 | The largest single gain in bits comes from F4 or F1 |
| B3 | F8b lowers SCRIBE's coherence distance the most of the single arms |
| B4 | No single arm raises dev SCRIBE by more than +0.05 over K5 at 8 MB |
| B5 | ALL raises dev SCRIBE over K5 by at least +0.03 (point estimate) |
| B6 | K5's dev SCRIBE is in [0.10, 0.20] |
| B7 | At least 3 techniques pass the screen |

### 13.8 Two more arms, and the bake-off moves to Odin's PC (registered 2026-11-01, before the full run)

**Odin:** "You can implement a NN in fuzzy logic. Let's analyse that. Stop the bake-off and run it on my PC if it's going to take long. Start each job on a single core."

- **The run.** The cloud run was stopped before any arm finished, and no score was read. `bench/fuzzy/run_bake.sh` runs prep, then one job per arm, each pinned to its own core (`taskset`, one BLAS thread), then the report. Outputs go to `D:\cypha_llm\fuzzy\`.
- **F9: a fuzzy rule base** (zero-order Takagi–Sugeno).
  - Rules have the form "IF w-1 is A_i AND w-2 is A_j THEN next ~ P_ij".
  - The fuzzy sets are fuzzy c-means prototypes (256) of our word vectors. Each word belongs to its 3 nearest sets, with c-means memberships.
  - Firing is the product of memberships. Each rule's output distribution is learned by counting next words, weighted by firing strength, with no gradient. Inference is the firing-weighted average.
  - This is a normalised radial-basis-function network built by counting: the neural form, without backpropagation.
- **F10: fuzzy ART** (Carpenter, Grossberg and Rosen, 1991): a self-organising category learner.
  - The input is the vectors of w-1 and w-2, complement-coded.
  - A new category is created whenever no existing one matches at vigilance 0.7. That was 0.8 at first, but a smoke test hit the 3,000-category cap. No score was read.
  - Each category counts next words, and a context reads its 3 best categories.
- **Predictions.**
  - B8: F9 gains more bits than F1 (soft memberships beat hard classes).
  - B9: F10 forms fewer than 3,000 categories and gains less than F9.

### 13.9 Self-organising structure without a network (registered 2026-11-01, before any run)

**Odin:** "The part where you said the NN self-organises its whole structure to learn the data was the important part. How can we replicate that, and test for it?" Then: "Do all of it at once and run it on my PC."

**The idea.** A neural network reshapes every part of itself to lower one number, its prediction loss. A count model can do the same with local moves on its structure, keeping a move only if it lowers bits:
- **Classes:** words move between classes when that raises the class-bigram likelihood (the exchange algorithm; Kneser and Ney, 1993; Martin, Liermann and Ney, 1998).
- **Contexts:** each context is kept only if it saves more bits than it costs to store (minimum description length).
- **Categories:** a category that predicts the true next word worse than the unigram is skipped and split, so new categories form where errors are (fuzzy ARTMAP's match tracking).
- **Mixture weights:** by EM, as now.

Running the result stays lookups and counts. **Tool:** `bench/fuzzy/so_bench.py`. It uses the bake-off's data, vocabulary, sampler and SCRIBE ladder. Each job runs on one core on Odin's PC.

| Test | Compares | Measured on |
|---|---|---|
| SO1 | Exchange classes against k-means classes (512 each), each used by K5 + a class trigram + class-substitution contexts | Val bits per byte, paired; writing |
| SO2 | At the same stored-pair budget (30% of K5's pairs): K5 pruned by count against contexts from all 80 patterns chosen by bits saved | Val bits, paired; writing |
| SO3 | Fuzzy ARTMAP categories (error-driven) against k-means categories with the same number, each mixed with K5 | Val bits, paired; writing |
| SO4 | The whole learner (exchange classes → contexts chosen by bits under the 30% budget → ARTMAP), alone and with K5; and its learning curve at 1, 2, 4 and 8 MB against K5's, with vectors and classes relearned at each size | Val bits; the curve; writing |
| SO5 | A domain shift: 8 chunks of other text (SO_SHIFT, web text on Odin's PC), each scored and then learned. FIXED (wiki classes and contexts, counts updated) against REORG (2 exchange passes and contexts re-chosen per chunk), with K5 beside them | Bits per byte per chunk |
| SO6 | Writing: SCRIBE v2 on GEN-A's 250 dev prompts for every SO1–SO4 arm, with K5 from the bake-off | SCRIBE, surface S, copy share, 13-word runs |

**Rule (a screen, as the bake-off's).** A self-organising move goes forward to C++ if its paired difference in val bits favours it with the 95% interval clear of 0, and its dev SCRIBE is not below its comparator's by more than 0.02. SO4 and SO5 are reported against their predictions.

**Predictions.**

| # | Prediction |
|---|---|
| S1 | Exchange classes beat k-means classes by at least 0.005 bits per byte (SO1) |
| S2 | At the same budget, contexts chosen by bits beat count pruning (SO2) |
| S3 | ARTMAP categories beat k-means categories at the same number (SO3) |
| S4 | The whole learner with K5 beats K5 by at least 0.02 bits per byte at 8 MB; alone, at 30% of K5's pairs, it is within +0.03 of K5 |
| S5 | The whole learner's advantage over K5 grows from 1 to 8 MB (the self-organising signature) |
| S6 | REORG beats FIXED on the shift's mean bits per byte, and the gap widens over the chunks |
| S7 | No SO arm raises dev SCRIBE over K5 by more than +0.02 at 8 MB |

**Changes before the full run (2026-11-01, from the smoke test on 1 MB; no full-run number was read).**
- SO2 and SO4 now fill the pair budget, taking contexts by net bits saved per stored pair, best first. The registered rule took only contexts with a positive net saving: at 1 MB that kept about 9,000 of 161,000 pairs, which broke "equal size".
- The pattern model's fallback to the unigram is a weight learned by EM, not a fixed 1%.
- **A known confound:** the pattern model uses maximum-likelihood counts mixed by EM, while K5 uses Kneser–Ney discounting. So SO2 measures structure and smoothing together. On the smoke slice, the chosen contexts score 1.615 bits per byte against 1.518 for pruned K5.
- `bench/fuzzy/run_so.sh` runs the 11 jobs on cores 53–63 after the bake-off finishes. SO5's shift text is the first 8 MiB of the unseen C4 shard (`c4_heldout.txt`).

### 13.10 GLNX: a gated linear network on CyphaLM's whole model (exploration, 2026-11-01)

**Odin:**
- "Make the content mixer a 2-layer gated linear network, gated on a side-information vector of expert losses and fuzzy features."
- "Use my whole model. It's meant to be the first layer."
- "Try it all at once ... on 1 meg of data and then we will reverse engineer it. I just wanna see it."
- "Keep doing the quick fast-mode check first: small streams and 10 prompts, to profile it and iterate it real quick."

This is exploration. Nothing is registered as a gate, and no default changes.

- **Layer 1 is CyphaLM's default stack.** `cyphalm_trie context8k ... --dump-experts FILE` writes one row per byte of the 1 MiB wiki holdout:
  - each expert's probability of the true byte, with its bucket: the trie's 5 experts and match, the trie mixture, byte Kneser–Ney, InfiniGram at 5 caps, class, similar-context, copy, the document parts;
  - the content mixer's chain after each stage.

  Without the flag, `context8k`'s output is unchanged.
- **Layer 2 is `bench/fuzzy/glnx.py`:**
  - gated linear mixers over those 24 inputs, each gate chosen by half-spaces of a side-information vector (buckets, match lengths, recent losses, surprise, previous byte's class, position in the word);
  - weights learned online by exponentiated gradient on log loss;
  - every gate starts at the content mixer's output;
  - it learns on the holdout's first half (where the content mixer fit its lambdas), then reads the second half prequentially;
  - `--fast` uses 64 KiB of each half.
- **Generation (10 prompts)** needs layer 2 inside `cyphalm_trie sample`. That comes once the bits show what layer 2 does.
- **The word-level version** written first, with experts I picked (log-bilinear, reservoir, linear state), is replaced by this one. Odin: "You didn't use my model."

**The data preprocessing phase (Odin, 2026-11-01: "Save that as a data preprocessing phase to save time").** Layer-2 work runs in two phases:

1. **Preprocess once:** `bench/fuzzy/glnx_prep.sh OUT`.
   - It runs CyphaLM's whole model (`context8k --dump-experts`) over the 1 MiB wiki holdout.
   - It saves `OUT/experts.f32`: one row of 36 float32 values per byte, about 150 MB. The columns are listed in `glnx.py`'s `COLS`.
   - It writes `OUT/manifest.txt`: the binary's hash, the command, the row count, the file's sha256, and the content mixer's chain lines.
   - It takes about 20 minutes on 4 cores. It is skipped when the manifest matches the binary and the flags (`FORCE=1` redoes it).
   - **Redo it whenever an expert or the content mixer changes,** because the dump is a snapshot of layer 1.
2. **Iterate on layer 2:** `python3 bench/fuzzy/glnx.py OUT/experts.f32 --fast`.
   - It uses 64 KiB of each half, which takes seconds.
   - Without `--fast` it reads the whole holdout, which takes minutes.
   - Every layer-2 idea is tried here first.

The dump holds only each expert's probability of the true byte, which is all a linear layer 2 needs to learn and score. Generation needs layer 2 inside `cyphalm_trie sample`.

### 13.11 Everything checked, and the decision (2026-11-01)

**Odin:** "Double check everything we did and why. Then make a decision. Check the whole repo." Also: "Don't undo my work."

**What was done, and why.**

| Step | Why | Result |
|---|---|---|
| The plan (§0–12) | Coherent text. Sentence form binds (D_S about 18 of D about 23) | The ladder G0–G5. Phases CT0–CT5 |
| N0, the neural diagnostic (§13.3), on Odin's PC | Which half of a network's fuzzy matching matters: similarity or the long state | Still running. Its report is pending |
| The fuzzy bake-off (§13.7), on Odin's PC | Non-neural fuzzy matching, 10 techniques, each mixed into word-level Kneser–Ney (K5) | Results below (`bench/fuzzy/results_pc_2026-11-01/`) |
| The self-organising bench (§13.9), on Odin's PC | Structure chosen by prediction: exchange classes, contexts chosen by bits, ARTMAP, the whole learner, a domain shift | Results below |
| GLNX (§13.10), in the cloud | Odin's content mixer as a gated linear network over his whole model | Fast mode below |

**What the results say.** All on dev, with no test read.

1. **Word-level counts on clean prose write far better than anything before.**
   - K5 is Kneser–Ney order 5 over words on about 7 MB of clean enwik8 prose. It samples whole words at T 1.0, nucleus 0.9, with the repeat ban.
   - It scores GEN-A dev **SCRIBE v2 0.344 [0.311, 0.379]**, with surface S 0.928 and copy share 0.040. That is the shuffled-sentence rung (G2's level).
   - The byte default scores 0.157 on dev at T 0.6, and the token sampler 0.192.
   - The lever is the combination of whole-word units and clean prose. Clean text alone on bytes (GEN-A) and tokens alone on raw XML (T4) each failed.
   - 1.2% of K5's samples hold a verbatim run of more than 12 words, which breaks Odin's copy cap.
2. **Every expert mixed into K5 improves bits and makes the text worse.**
   - All 10 fuzzy arms, and every self-organising arm with K5, lower val bits per byte by 0.0004–0.032. They lower SCRIBE by 0.013–0.33.
   - ALL predicts best (−0.015) and writes worst (0.218).
   - Weights fit to predict spread the mass that the sampler draws from.
3. **The best structural move is exchange classes:** −0.032 bits per byte against k-means classes (SO1), the largest single gain. Contexts chosen by bits (SO2), ARTMAP (SO3) and re-organising on a shift (SO5) did not help.
4. **The content mixer is right; freezing it is what is wrong.**
   - On GLNX's fast slice, Odin's chain with its per-bucket lambdas learning online (C0o) gains −0.050 bits per byte [−0.064, −0.037].
   - One gated layer gains −0.041, and two layers −0.043. More gates add nothing, as gate24 found.
   - The full clean dump, with the document stage, is being rebuilt to confirm this.

**Predictions, scored.**
- Bake-off:
  - B1 miss (ALL −0.015, not −0.05);
  - B2 miss (F2 is the largest single gain, −0.0083);
  - B3 not scored (the axes were not computed);
  - B4 hit;
  - B5 miss;
  - B6 miss (K5 0.344, far above [0.10, 0.20]);
  - B7 miss (none passed the screen);
  - B8 miss (F9 −0.0008 against F1 −0.0030);
  - B9 miss on the gain (F10 −0.0010, more than F9).
- Self-organising bench:
  - S1 hit (−0.032);
  - S2 and S3 miss;
  - S4 miss (−0.014 with K5; alone +0.105);
  - S5 hit (−0.004, −0.010, −0.012, −0.014 from 1 to 8 MB);
  - S6 miss (a tie);
  - S7 hit.

**The decision** (open to Odin's veto):
1. **Generation: build K5's design into CyphaLM.**
   - Whole words, Kneser–Ney counts, clean prose, the nucleus and repeat ban, inside `cyphalm_trie sample`.
   - The byte model spells any word outside the vocabulary, so nothing is dropped as `<unk>`. The 12-word cap is enforced.
   - Register it, then make one GEN-A test read against the byte default (BK0, test 0.147 at T 0.6).
   - **No other expert is mixed into the sampler** until it is shown on dev to raise SCRIBE, not only bits.
2. **Prediction: keep Odin's content mixer and stop freezing it.**
   - Once the full dump confirms C0o, the chain's per-bucket lambdas learn online in `cyphalm_trie` (AdaGrad per bucket, as C0o). Registered first; answers unchanged.
   - Exchange classes are a candidate for the class expert's classes.
3. **Stopped for generation:** the fuzzy experts, the self-organising structure moves, and extra gates or layers in the mixer. Their code and results stay in `bench/fuzzy/` as the record. Nothing is deleted.
4. **Waiting:** N0 and Qwen's yardstick (Q0) will show how far neural models on the same data and decoder sit above K5's 0.344.

### 13.12 The online content mixer in `cyphalm_trie` (registered 2026-11-01, before the C++ runs)

**Odin:** "I definitely want to make the mixer online." Then: "Test it and decide whether to make it default, and then write it up."

**The change** (`cyphalm_trie`, `context8k`). Two flags; without them the output is unchanged.
- `--online-mix`: every stage of the content mixer's chain learns its lambda per bucket online, instead of fitting it on the holdout's first half and freezing it. The stages are Kneser–Ney, InfiniGram 8192, class, similar-context, copy and the document parts.
  - Each byte is predicted with the current lambda. The lambda then follows its stage's log-loss gradient: AdaGrad per bucket, start 0.05, step 0.05, kept in [0, 0.995]. Copy's short-match buckets stay 0.
  - The frozen fit is still computed and returned to the answer path and the samplers, so answers do not change.
- `--online-final`: one gated linear mixer after the chain, over 24 values per byte (every expert and every chain stage).
  - Its weight sets are chosen by InfiniGram's bucket × the previous byte's class: 80 sets, gated by a small context as gate24 is.
  - Weights are learned by exponentiated gradient (step 0.05), and every set starts at the chain.

**Evidence before the C++ run** (`glnx.py` on the full dump, second half, block bootstrap):
- the online chain: −0.0172 [−0.0213, −0.0134];
- one online layer over the frozen chain: −0.0291 [−0.0334, −0.0254];
- two layers: −0.0287.

**Arms** (`context8k` on enwik8 + Dolly, the standard command): R0 as it is; R1 `--online-mix`; R2 `--online-mix --online-final`.

**Rule.** The online mixer becomes the default of `context8k` (and of `qa`'s holdout lines, with `--frozen-mix` to restore) if all hold:
1. R1's final chain line is below R0's by at least 0.01 bits per byte. The glnx bootstrap above gives the interval for the same bytes.
2. R1's and R0's answer tallies are identical, and so is every line outside the chain.
3. The wall time is within +10% of R0's.
4. `cyphalm_trie selftest` passes.

`--online-final` is added to the default if R2's `online_final_bpc` is below R1's final chain line by at least 0.005.

**Predictions.**
- O1: R1's final chain line is 1.675–1.690 (R0 1.6979).
- O2: R2's `online_final_bpc` is at least 0.008 below R1's final line.
- O3: the answer tallies are identical.

**The two-layer experiment on the online chain** (2026-11-01, `glnx.py --online-inputs`, fast mode on the 150 MB dump). Odin: "Try the experiment we just did with the online learning on the 2-layer one."

The inputs are the raw experts plus the chain's stages as the online chain computes them while it learns. Results, in bits per byte:

| Arm | Fast slice | Against the online chain [95%] |
|---|---:|---|
| C0: the chain, frozen | 1.7786 | — |
| D0: the chain, online | 1.7557 | — |
| D1: one online layer on top (gate24-style small context) | **1.7369** | **−0.019 [−0.023, −0.015]** |
| D2: two layers, half-space gates | 1.7425 | −0.013 |
| D3: two layers, each neuron gated by its own small context | 1.7385 | −0.017 |
| D4: D3 at half the step size | 1.7463 | −0.009 |
| D5: D2 at half the step size | 1.7443 | −0.011 |

One online layer on top of the online chain is the best, and a second layer adds nothing. D1 is the C++ test's R2 (`--online-mix --online-final`).

**Why the better predictors wrote worse (a reading, not yet measured).**
- Log loss rewards a smooth, hedged distribution. Sampling from it gives average, non-committal text.
- Human text commits where language is constrained, and its surprise rises only at real choice points (Holtzman et al., 2020; Meister et al., 2023, typical sampling).
- Next measures: the per-word surprise profile of human, K5 and mixture text; and sampling from a sharpened mixture (geometric mixing, typical sampling) while the hedged mixture still scores and answers.

**The C++ results** (Odin's PC, 2026-11-01, one core per arm, the standard `context8k` command):

| Arm | Final chain line | `online_final_bpc` | Wall | Peak RSS |
|---|---:|---:|---:|---:|
| R0, as it is | 1.6971 | — | 915 s | 216 MB |
| R1, `--online-mix` | 1.6966 | — | 909 s | 216 MB |
| R2, `--online-mix --online-final` | crashed | — | — | — |
| R2b, R2 plus `--dump-experts` | 1.6966 | **1.6674** | 935 s | 321 MB |

- The answers do not change. Outside the bits lines, R0 and R1 differ in 0 lines, and R1 and R2b in 0 lines. Selftest passes.
- **R2's crash.** `--online-final` read per-byte arrays that only `--dump-experts` filled. Fixed: both flags now fill them. They are also stored as float, since the dump and the final layer only read them as float. That halves their memory, and no output changes.

**Why R1 gains so little.** In Python the online chain gained 0.017 bits per byte. In C++ it gains 0.0005.
- The Python chain let copy's short matches back in. The C++ chain keeps their lambdas at 0, as B4B chose: short copies lowered generated text's surface S from 0.907 to 0.840.
- With the same clip in Python, the online chain scores 1.7027, worse than the frozen 1.6979.
- So the Python gain came from short copies, not from learning online.

**Decision.**
- `--online-mix` does not become the default. Rule 1 fails: −0.0005 against the 0.01 needed. O1 missed (1.6966 is outside 1.675–1.690). O3 held.
- `--online-final` passes its own rule: −0.029 against the 0.005 needed. The answers are identical, the wall time is +2%, and selftest passes. O2 held (−0.029 is at least 0.008).
- But it was run on top of the online chain, and the default chain stays frozen. So R3 checks it on the frozen chain before it becomes the default.
- The final layer reads every expert, short copies included, and only scores. Answers and samplers still read the frozen chain. So its bits measure prediction, not writing, and the final chain line stays the line the sampler matches.

**R3** (registered 2026-11-01, before the run; this container, one core per arm, all three side by side):
- R0c: `context8k` as it is.
- R3: `context8k --online-final`, on the frozen chain.
- R2c: `context8k --dump-experts`. Its dump must match `experts.f32` (sha256 `manifest.txt`) byte for byte. This shows the float arrays change nothing.
- **Moved to Odin's PC** (Odin, before any result: "Run them both on my PC. Too long"). The PC's chain line differs from this container's (1.6971 against 1.6979), so its dump cannot match `experts.f32`. Instead R2o, the binary from `28dbab6` (before the float change) with `--dump-experts`, runs beside R2c, and the two dumps must match byte for byte. The four arms run side by side, one core each.

**Rule.** `--online-final` becomes the default of `context8k` (`--no-online-final` restores the old output) if all hold:
1. R3's `online_final_bpc` is below its own final chain line by at least 0.005.
2. Every line of R3 outside `online_final_bpc` and `online_mix` matches R0c's.
3. R3's wall time is within +10% of R0c's.
4. Selftest passes, and R2c's dump matches.

**Prediction.** O4: R3's `online_final_bpc` is 1.663–1.673. The Python layer on the frozen chain gave 1.6688; this machine's chain line is 1.6979. O5: R3's peak memory is under 290 MB.

### 13.13 GLNB: a bitwise gated linear network with an LSTM inside it (exploration, 2026-11-01)

**Odin:** "Make a hybrid model of a proper bitwise GLN with an LSTM capability", in fast mode and in Python. The LSTM is ruled in for this experiment. No bit-level dump and no separate LSTM byte model.

`bench/fuzzy/glnb.py`. Every byte is 8 binary predictions, most significant bit first, with the partial byte as PAQ keeps it.
- **Base inputs:** bitwise context models of orders 0–5. They hold hashed counts from enwik8's first 7 MiB (CyphaLM's training text) and update online.
- **The LSTM inside:** a byte LSTM (hidden 64) that learns online from scratch on the stream (truncated BPTT, Adam).
  - Its state is part of every layer-1 neuron's gate.
  - Its probability of each bit is one more input.
- **The GLN:**
  - layer-1 neurons (16, 32 or 64), each a gated geometric mixer with weights chosen per bit position by half-spaces of the LSTM state and the inputs;
  - 4 layer-2 neurons;
  - a final neuron per bit position;
  - an APM.

  Every neuron learns online from its own log loss.
- **The read:** the same bytes as `glnx.py --fast`. It learns on 64 KiB of the holdout's first half, then predicts, scores and learns on 64 KiB of its second half. Arms: H16, H32, H64, and H32 without the LSTM.

**Results** (2026-11-01, this container, one run each, so no intervals). Bits per byte on the 64 KiB scored:

| Arm | Layer-1 neurons | LSTM | Bits per byte | Time |
|---|---:|---|---:|---:|
| H16 | 16 | yes | 2.0835 | 145 s |
| H32 | 32 | yes | **2.0821** | 154 s |
| H64 | 64 | yes | 2.0964 | 192 s |
| H32 without the LSTM | 32 | no | 2.0945 | 122 s |
| The content mixer, frozen (same bytes) | — | — | 1.7786 | — |
| The content mixer, online (same bytes) | — | — | 1.7557 | — |

**Reading.**
- GLNB is about 0.30 bits per byte behind CyphaLM's content mixer on the same bytes.
  - Its inputs are hashed bit counts of orders 0–5 from 7 MiB of text.
  - The mixer's inputs are CyphaLM's whole model: the trie, InfiniGram, class, similar-context, copy and the document parts. The gap is the inputs, not the mixing.
- The LSTM helps a little: 2.0945 to 2.0821 (−0.012). With one run and no interval, that is not yet shown.
- More neurons do not help. 64 is worse than 32, and 16 is about as good. A small one is enough, as Odin said.
- It is not adopted. It only decides something once it reads CyphaLM's experts, and that is the bit-level dump Odin ruled out for this experiment.

**Context capacity against data** (registered 2026-11-01, before the runs). Odin cited Veness et al.: GLNs approximate any continuous function "so long as the context capacity is sufficiently expressive".
- **The test:** H32 with 2, 4, 6 and 8 half-spaces per layer-1 neuron (4, 16, 64 and 256 contexts per bit position). Each is scored on 64 KiB and on 512 KiB, after learning on as much of the first half. `glnb.py ENWIK8 hs HALF_SPACES BYTES`.
- **The frozen chain on the same bytes:** 1.7786 on 64 KiB and 1.6979 on 512 KiB.
- **Reading rule:**
  - "Data is the limit" if the best half-space count on 512 KiB is higher than on 64 KiB, and beats 4 half-spaces there by at least 0.01.
  - "Inputs are the limit" if, at both sizes, every count is within 0.01 of the others.
  - Anything else is mixed.
- **Prediction:** inputs are the limit. On 512 KiB every arm stays at least 0.2 bits per byte behind the chain.

**A bug in GLNB's LSTM** (found 2026-11-01, after the results above). The LSTM read each byte's predecessor only after predicting the byte, so every prediction came from a state one byte behind. Fixed: the state now reads every byte before the one it predicts. H16 to H64 and the half-space test (sent to the PC before the fix) ran with the old LSTM; the writing test below reports the fixed H32's bits per byte beside them.

**Does GLNB write better?** (Odin, 2026-11-01: "See if GLNB generates better. We haven't checked that. Remember prediction isn't generation." Registered before the run; fast mode on Odin's PC.)
- **Sampler** (`glnb.py ENWIK8 gen ARM TEMP NPROMPTS OUT`): the byte sampler's rules. The model gives all 256 byte probabilities (255 bit predictions). Bytes that would repeat what followed the same 11-byte window earlier are banned. Then temperature and nucleus 0.9; 600 bytes; no learning while writing.
- **Arms:** H32 with the fixed LSTM, and H32 without the LSTM, each at T 1.0 and T 0.6. Each learns as the read does (128 KiB of the holdout) first.
- **Prompts:** GEN-A dev's first 10.
- **On the same 10 prompts:** the byte default (`cyphalm_trie sample`, T 0.6, its best), and K5 (the bake-off's text).
- **Score:** SCRIBE v2 with surface S (GEN_TRACK's ladder command).
- **Rule:** this is a screen. GLNB goes to all 250 dev prompts if its best arm's SCRIBE is at least 0.05 above the byte default's on the same 10 prompts.
- **Prediction:** GLNB writes worse than the byte default (SCRIBE below it, surface S lower). It reads 5 bytes of context and a small LSTM, against the byte default's whole model.

### 13.14 Research item: the signal coherent text needs, and optimising to extract it (Odin, 2026-11-01)

**Odin:** "Add a note to the documentation to research the signal needed to extract to get coherent text. We can optimise to extract that."

**Why.** The bake-off, the self-organising bench and GLNX all found the same thing: what predicts best does not write best.
- Mixtures fitted for log loss are smooth and hedged, so they are never badly surprised.
- Sampling from them gives average, non-committal text.
- Human text commits where language is constrained, and its surprise rises only at real choice points (§13.12).

So prediction (bits per byte) is the wrong target for writing. We need to know what signal writing needs, then optimise the model to extract that signal.

**What to research.** Each is measured on human continuations (GEN-A dev), K5's text and the mixtures' text:
1. **The surprise profile:** the distribution of per-byte and per-word surprise, its variance, and its autocorrelation (burstiness). Where it rises: word starts, content words, sentence starts.
2. **Entropy against surprise:** whether human text sits near the local entropy (typical sampling; Meister et al., 2023) and model text does not.
3. **Commitment:** the probability mass the top choice holds inside words and phrases, against at choice points.
4. **Which of these track SCRIBE's axes** (sentence form D_S, coherence D_C) across every arm already scored (the bake-off, SO, N0, GEN-A to GEN-E, T4).
5. **Where each expert helps or hurts the text:** per position class, which experts sharpen the right choices and which only spread mass.

**What to optimise.** Once the signal is known:
- **A generation objective, separate from log loss.** For example, match human text's surprise profile, or reward commitment at constrained positions. This is not SCRIBE itself: it must not be gamed (§1.3), and it is checked on the second benchmark and by Odin's ratings.
- **Mixer weights for writing.** The online content mixer (or the GLN) gets a second weight set, or a gate, learned for that objective. The hedged mixture keeps scoring and answering.
- **The sampler:** temperature per position, typical sampling, or geometric mixing (where the experts must agree), each tuned on dev for the objective.

**Measures and rules:** as §1.2 and §5, with dev for choices and one test read per change.

### 13.15 The spike in the content mixer's online learning, and the GLN in the mixers (2026-11-03)

**Odin:** "Fix the spike in the learning on the content mixer. It's overfiltered and the signal isn't learning on text." And: "Have the GLN feed into the mixers as well, feed that into the online learning. So basically it's going to be recurrent and online on every part."

**The spike (measured on the dump, before any fix).** `online_mix` learns each lambda by AdaGrad. AdaGrad divides each step by the square root of every squared gradient the bucket has seen. The gradient is (expert − mixture) ÷ mixture, so a byte the mixture gave almost nothing to makes a huge one.

| Stage | Largest gradient | Median | Share of the squared-gradient sum from the top 0.1% of bytes |
|---|---:|---:|---:|
| Kneser–Ney | 160 | 0.18 | 60% |
| InfiniGram 8192 | 200 | 0.52 | 14% |
| Class | 2,238 | 0.29 | 97% |
| Similar-context | 2,784 | 0.41 | 91% |
| Copy | 12 | 0.21 | 3% |
| Document cache | 29,088 | 1.00 | 99.9% |

So in four of six stages a few bytes set the step size for the rest of the stream. After one spike, a bucket barely learns from the text that follows.

**What was tried** (second half of the dump, bits per byte; the frozen chain is 1.6979):

| Update | Copy's short matches off (as now) | Copy's short matches on |
|---|---:|---:|
| AdaGrad (as now) | 1.6994 | 1.6776 |
| AdaGrad, gradients clipped to ±2 | 1.7383 | 1.7338 |
| Decaying average of squared gradients (RMS) | 1.6991 | 1.6768 |

- Clipping the gradients is much worse. The spikes carry real signal; they should move the lambda, not freeze it.
- **The registered fix** (before its result): learn each lambda in log-odds, lambda = sigmoid(theta), as PAQ's mixers learn in the stretch domain. The log-loss gradient in theta is lambda (1 − lambda) (expert − mixture) ÷ mixture, which always lies in [−1, 1]. Arms: plain steps (0.02 and 0.005) and AdaGrad on theta (step 0.1). Exponentiated gradient (step 0.02, as the final layer) is a fourth arm.
- **Rule:** the best arm replaces `online_mix`'s update in C++ if it beats AdaGrad by at least 0.003 bits per byte on the dump in both columns. Answers and samplers are unchanged by it (they read the frozen fit).
- **Result** (the registered arms, same bytes):

  | Update | Short copies off | Short copies on |
  |---|---:|---:|
  | AdaGrad (as now) | 1.6994 | 1.6776 |
  | Log-odds, step 0.02 | 1.6979 | 1.6758 |
  | Log-odds, step 0.005 | 1.7101 | 1.6882 |
  | Log-odds, AdaGrad | 1.6984 | 1.6765 |
  | **Exponentiated gradient, step 0.02** | **1.6920** | **1.6692** |

  - Exponentiated gradient beats AdaGrad by 0.0074 and 0.0084, so by the rule it replaces `online_mix`'s update. Each lambda is the first of two weights; each weight is multiplied by exp(0.02 × its expert's probability ÷ the mixture's), capped at 30 in the exponent, then the pair is renormalised with a floor of 1e-4. A spike can move the weights a lot at once, but it never shrinks later steps.
  - Log-odds removes the spikes too, but learns too slowly; the gain is under 0.002.
  - The step size hardly matters (chosen on the dump, after the rule): 0.01 gives 1.6935 and 1.6708; 0.05 gives 1.6913 and 1.6692. It stays 0.02.
- **Overfiltering.** Copy's short matches (below 8 bytes) are held at 0 (B4B, for writing quality). In the online chain that costs 0.022 bits per byte (1.6994 against 1.6776). The online path only scores, so it can take short copies. The sampler keeps B4B's rule until a writing test says otherwise: prediction is not generation.

**The GLN in the mixers** (registered before the runs). `bench/fuzzy/gln_mix.py` on the dump, with `glnb.py … stream` giving GLNB's probability of every holdout byte. GLNB is the bitwise GLN with the fixed LSTM, 32 neurons; it learns online over the whole holdout.
- M0: the online chain (the log-odds update, short copies on).
- M1: M0 plus GLNB as one more chain stage (lambda per InfiniGram bucket, online).
- M2: M0 plus the online final layer (`--online-final`).
- M3: M1 plus the final layer, with GLNB as one more input.
- M4: M3 with GLNB fed back from the mixer. Two more side values gate every GLNB neuron: the mixer's recent loss, and GLNB's recent loss against it. So the GLN reads the mixer and the mixer reads the GLN: one recurrent loop, online on every part.
- **Predictions:** M1 − M0 is −0.005 to −0.015 (GLNB alone is far worse, but it is a different kind of model, and a mixture gains from that). M3 − M2 is about −0.005. M4 − M3 is at most −0.002.
- **Rule:** GLNB is ported to C++ (the online module) as a chain stage if M1 − M0 ≤ −0.005 with its 95% interval below 0. Then a writing test (SCRIBE on GEN-A dev) decides whether the sampler reads it.

**In C++** (2026-11-03, after the module reorganisation). Both are in the online module:
- `online_mix` (`native/include/cypha/online/content_mixer.hpp`) now learns by exponentiated gradient. In the online path, copy's short matches are no longer held at 0. The frozen lambdas that answers and samplers read keep B4B's clip.
- `BitwiseGLN` (`native/include/cypha/online/bitwise_gln.hpp`) is GLNB in C++. On the holdout's second half it scores 1.9435 bits per byte (Python 1.9422 with feedback; C++ with feedback 1.9412). It streams 1 MiB in about 4 minutes and holds 101 MB; its counts are 16-bit fixed point.
- `cyphalm_trie context8k --gln` adds it as the chain's last stage (a lambda per InfiniGram bucket) and as one more input of `--online-final`, before the chain. `--gln-feedback` closes the loop with the chain's recent loss. It prints `gln_alone_bpc`.

**C++ runs** (registered before them; this container, one core per arm, side by side):
- C0: `context8k` as it is.
- C1: `--online-mix`.
- C2: `--online-mix --online-final`.
- C3: `--online-mix --online-final --gln`.
- C4: `--online-mix --online-final --gln-feedback`.
- C5: `--online-mix --gln` (the GLN stage without the final layer).

**Rules for the defaults of `context8k`** (each needs identical answer tallies to C0 and a passing selftest):
1. `--online-mix` becomes the default if C1's last chain line is at least 0.01 below C0's, with wall time within +10% (§13.12's rule).
2. `--online-final` becomes the default if C2's `online_final_bpc` is at least 0.005 below C1's last chain line.
3. `--gln` becomes the default if C3's `online_final_bpc` is at least 0.02 below C2's (a big win, since it adds about 100 MB), with wall time within +35% of C2's.
4. `--gln-feedback` replaces `--gln` if C4 is at least 0.002 below C3.

**Predictions** (from the dump): C1 near 1.669; C2 near 1.654; C3 and C4 near 1.62; C5 near 1.643.

None of this changes what the sampler writes. That needs the GLN and the online mixer in `sample`, and a SCRIBE test on GEN-A dev.

**The GLN replicates in C++, and the loop adds nothing** (`gln_mix.py` on the C++ GLN's output, same bytes):

| Arm | Bits per byte | Against [95%] |
|---|---:|---|
| M0: the online chain (exponentiated gradient, short copies on) | 1.6692 | the frozen chain 1.6979: −0.0287 [−0.0356, −0.0224] |
| M1: M0 plus the GLN stage | 1.6425 | M0: **−0.0267 [−0.0308, −0.0227]** |
| M2: M0 plus the final layer | 1.6538 | M0: −0.0154 [−0.0169, −0.0139] |
| M3: M1 plus the final layer, the GLN an input | **1.6198** | M2: **−0.0341 [−0.0400, −0.0288]** |
| M4: M3 with the mixer's feedback into the GLN | 1.6197 | M3: −0.0001 [−0.0003, +0.0001] |

- By the rule, the GLN goes into C++ as a chain stage (M1 − M0 is far below −0.005). The registered read on Python's GLNB follows when its stream finishes.
- Against today's default (the frozen chain), M3 is −0.078 bits per byte.
- The feedback loop does not help: the GLN already sees the same text the mixer sees. It stays a flag (`--gln-feedback`), off.
- The predictions were too small: M1 − M0 was predicted at −0.005 to −0.015 and is −0.027; M3 − M2 was predicted at −0.005 and is −0.034. M4 − M3 (predicted at most −0.002) is 0.

**Does it write better? The writing screen** (registered before the run). `cyphalm_trie sample --gln` mixes the GLN's whole byte distribution in last, with one lambda fit on the holdout's first half over the sampler's own chain. For each prompt the GLN reads the prompt and every byte written, learning nothing, and returns to its state after.
- W0: `sample` as it is. W1: `sample --gln`. GEN-A dev's first 10 prompts, T 0.6, 0.8 and 1.0, scored by SCRIBE v2 (GEN_TRACK's ladder).
- **Rule:** a screen. The GLN goes to all 250 dev prompts if W1's best temperature beats W0's best by at least 0.03 SCRIBE on these 10 prompts.
- **Prediction:** no gain (the difference is under 0.03). The GLN helps prediction most where the text is most predictable, and smoothing in a fifth model's mass is what has lowered SCRIBE before (§13.11).

**Writing screen result** (2026-11-03, this container). SCRIBE v2 on GEN-A dev's first 10 prompts, by temperature:

| Arm | T 0.6 | T 0.8 | T 1.0 | Surface S at T 0.6 |
|---|---:|---:|---:|---:|
| W0: `sample` | **0.256** [0.112, 0.425] | 0.075 | 0.032 | 0.937 |
| W1: `sample --gln` (lambda 0.245) | 0.020 [0.008, 0.034] | 0.031 | 0.065 | 0.928 |

- The GLN makes the writing much worse: the best arm falls from 0.256 to 0.065. By the rule it does not go to the 250 prompts, and `sample` stays without it. The prediction (no gain) was right in direction but too mild.
- It pulls the text into markup. It learned on raw enwik8, XML and all, and one lambda gives it a quarter of the mass at every byte. Prompt 0 at T 0.6 with it: "The and [[Protestant and have been contributor>\n <comment>*</comment>\n <text xml:space="preserve">{{Otherusesabout|…".
- So the GLN stays in prediction (`context8k --gln`) and out of writing. Two ways it could enter writing later, each to be registered: learn on clean prose (SCRIBE's cleaner) rather than raw enwik8; and a lambda per position class (word starts, inside words) learned for writing, as §13.14 asks.

**C++ results** (2026-11-03, this container, one core per arm; C0 to C3 side by side, then C4 and C5). Bits per byte on the holdout's second half:

| Arm | Last chain line | `online_final_bpc` | `gln_alone_bpc` | Wall | Peak RSS |
|---|---:|---:|---:|---:|---:|
| C0: as it was | 1.6979 | — | — | 1296 s | 214 MB |
| C1: `--online-mix` | 1.6714 | — | — | 1278 s | 214 MB |
| C2: C1 + `--online-final` | 1.6714 | 1.6543 | — | 1339 s | 251 MB |
| C3: C2 + `--gln` | 1.6456 | **1.6214** | 1.9468 | 1569 s | 371 MB |
| C4: C2 + `--gln-feedback` | 1.6455 | 1.6213 | 1.9446 | 1636 s | 371 MB |
| C5: C1 + `--gln` | 1.6456 | — | 1.9468 | 1567 s | 287 MB |

- Every arm's answer tallies match C0's, and selftest passes.
- **Rule 1 holds:** C1 is 0.0265 below C0, and its wall time is 1% shorter. `--online-mix` is the default of `context8k`.
- **Rule 2 holds:** C2's final layer is 0.0171 below C1's chain. `--online-final` is the default.
- **Rule 3 holds:** C3 is 0.0329 below C2, with wall time +17% (limit +35%). The GLN is the default. It adds about 120 MB.
- **Rule 4 fails:** C4 is 0.0001 below C3. The feedback loop stays a flag.
- **Every prediction held:** C1 1.6714 (near 1.669), C2 1.6543 (near 1.654), C3 and C4 1.6214 and 1.6213 (near 1.62), C5 1.6456 (near 1.643).
- The registered Python read agrees. GLNB as a chain stage gives M1 − M0 = −0.0261 [−0.0302, −0.0222]; the feedback loop gives M4 − M3 = 0.0000 [−0.0002, +0.0002].

**The new default of `context8k`** reports 1.6214 bits per byte, against 1.6979 before: −0.0765. `--frozen-mix --no-online-final --no-gln` gives the old lines. `qa`, `sample` and every answer are unchanged. `glnx_prep.sh` passes those three flags, so the dump keeps the frozen chain's columns.

**What did not change: writing.** The sampler still reads the frozen chain, without the GLN, because the GLN lowered SCRIBE (above). Odin asked for "online on every part". The online parts now cover prediction. For writing, online learning still has to be shown to help on GEN-A, as §13.14 asks.
