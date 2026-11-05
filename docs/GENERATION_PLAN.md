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

### 13.16 Reconciling the GLN with the sampler (registered 2026-11-03, before the runs)

**Odin:** "The GLN shouldn't drop SCRIBE. It hasn't been trained at scale. With the GLN the algorithm can scale. Find a way to reconcile the GLN and the change that gave the +2 to SCRIBE without the GLN. They are incompatible." Then: "Find some simple but elegant way to reconcile them. Do a google search."

**The clash.** Temperature 0.6 is what lifts the byte sampler: on the screen's 10 prompts, 0.032 at T 1.0 becomes 0.256 at T 0.6. With the GLN it does the opposite: 0.065 at T 1.0 becomes 0.020 at T 0.6.
- The GLN reads 5 bytes and a small LSTM, so it is sometimes confidently wrong. After "contributor" it puts most of its mass on ">", because in raw enwik8 the word mostly sits in XML headers.
- Mixed in linearly with a quarter of the weight, that one peak can outrank the chain's mass spread over good words.
- Temperature 0.6 raises every probability to the power 1/0.6, which sharpens that peak until it wins. Nucleus 0.9 then keeps it.

**The fix from the literature.** Contrastive decoding (Li et al., 2023) uses an *adaptive plausibility constraint*: the strong model says which tokens are possible, namely those with at least α (0.1) of its top token's probability, and only those are scored. DExperts (Liu et al., 2021) combines models as a product of experts, so a token needs every expert's support. Both stop a weaker model from putting mass where the strong one says no. Here the strong model is the chain, which writes well at T 0.6; the GLN is the one that learns and scales.

**Arms** (`sample --gln-mix MODE`, GEN-A dev's first 10 prompts, the screen's seeds, T 0.6, 0.8 and 1.0, SCRIBE v2):
- R1 `after`: the temperature sharpens the chain only, then the GLN is mixed in at its lambda.
- R2 `geo`: a product of experts, chain^(1 − λ) × GLN^λ, then the temperature.
- R3 `gate`: the plausibility constraint. Candidates are the bytes the chain gives at least 0.1 of its top byte; the GLN and the chain mix inside that set; then the temperature. α = 0.1 is the paper's value, not tuned.
- R4 `gate` with `--gln-clean`: the GLN learns clean prose instead of raw enwik8. Its counts come from the 7 MB of SCRIBE-cleaned enwik8 that K5 used, then it learns online over that text's last MiB, then reads the holdout without learning.
- R5 `geo` with `--gln-clean`.
- References, same prompts and seeds: W0 (no GLN) 0.256 at T 0.6; W1 (`lin`) 0.065 at best.

**Rule.** An arm reconciles the two if its best temperature scores at least W0's best minus 0.03 (≥ 0.226) while the GLN is in. The best such arm goes to all 250 dev prompts against W0. It becomes the sampler's default only on that read's rules (GEN_TRACK: dev SCRIBE with 95% intervals, copy share within the limit).

**Predictions.** R3 (`gate`) reconciles them: 0.22–0.30 at T 0.6. R2 (`geo`) is above W1 but below W0. R1 (`after`) stays near W1. Clean prose (R4, R5) adds at least 0.02 to its pair.

**Screen results** (10 prompts, SCRIBE v2 by temperature; the GLN's lambda in brackets):

| Arm | T 0.6 | T 0.8 | T 1.0 |
|---|---:|---:|---:|
| W0: no GLN | **0.256** [0.112, 0.425] | 0.075 | 0.032 |
| W1: `lin` (0.245) | 0.020 | 0.031 | 0.065 |
| R1: `after` (0.245) | 0.157 | 0.195 | 0.065 |
| R2: `geo` (0.245) | 0.192 [0.024, 0.410] | 0.018 | 0.011 |
| R3: `gate` (0.245) | 0.130 [0.032, 0.258] | 0.080 | 0.088 |
| R4: `gate` + clean (0.015) | 0.131 | 0.097 | 0.118 |
| R5: `geo` + clean (0.015) | 0.049 [0.017, 0.101] | 0.252 | **0.263** [0.071, 0.542] |

- Every new mix beats the plain linear one (W1's best is 0.065; theirs are 0.13–0.26). The clash is real, and the way the GLN is mixed in matters.
- By the rule, R5 reconciles (0.263 ≥ 0.226). But its GLN has lambda 0.015, so it writes almost as W0 does. Yet it scores 0.049 at T 0.6 against W0's 0.256. On 10 prompts the same sampler can swing by 0.2, so this screen cannot rank the arms.
- The prediction failed: the gate (R3) scores 0.130, not 0.22–0.30.
- The clean GLN's lambda is 0.015 because it is fit on the raw holdout, XML and all, where a model of clean prose does badly. For writing, the lambda should be fit on the kind of text to be written.

**The 250-prompt read** (registered before it). W0, R2, R3 and R5 on all 250 GEN-A dev prompts, T 0.6, 0.8 and 1.0, side by side. R5 goes by the rule; R2 and R3 go as exploration, since the screen cannot separate them.
- **Rule:** a GLN arm reconciles the two if its best temperature is at least W0's best minus 0.02 (non-inferior), with copy share within GEN_TRACK's limit. Among those, the one with the highest SCRIBE becomes `sample`'s way of mixing the GLN, while `--gln` stays off by default until the GLN is trained at scale (Odin: "it hasn't been trained at scale").
- **Prediction:** R2 and R5 are non-inferior; R3 is not.

**250-prompt result** (2026-11-03, all of GEN-A dev, SCRIBE v2 [95%] by temperature):

| Arm | T 0.6 | T 0.8 | T 1.0 |
|---|---:|---:|---:|
| W0: no GLN | **0.174** [0.137, 0.209] | 0.155 | 0.073 [0.049, 0.098] |
| R2: `geo` (0.245) | 0.090 [0.067, 0.116] | 0.075 | 0.062 |
| R3: `gate` (0.245) | 0.096 | 0.102 | **0.117** [0.090, 0.147] |
| R5: `geo` + clean (0.015) | 0.153 [0.121, 0.186] | 0.147 | 0.099 |

- **No arm reconciles them by the rule.** R5 is the closest: 0.021 below W0, against the 0.02 allowed. Its interval overlaps W0's almost entirely, but its GLN carries 1.5% of the weight, so it shows only that a nearly absent GLN does no harm. R2 and R3 are clearly below W0. `sample` keeps writing without the GLN.
- **Predictions:** R3 not non-inferior held. R2 non-inferior failed. R5 non-inferior failed by 0.001.
- **The gate does what it should, at T 1.0.** There it beats W0 (0.117 against 0.073), and its scores change least with temperature: it cuts the implausible tail that T 1.0 lets in. But at T 0.6 the chain alone is sharper and better.
- **What this says.** In prediction, a weaker model helps a mixture because its errors differ. In writing, every sampled byte the GLN pulls in carries its errors. Alone, the GLN scores 1.94 bits per byte against the chain's 1.67; it learned from 7 MB of text. At that size no way of mixing it lets it carry real weight without lowering SCRIBE.
- **Next**, as Odin says, is the GLN trained at scale: on clean prose, far more of it, on Odin's PC.
  - Read both its bits per byte alone and the 250-prompt SCRIBE with `geo` and `gate`.
  - Fit its writing lambda on clean prose, not on the raw holdout.
  - The registered rule above stays the test.

### 13.17 The GLN trained on 1 GB (registered 2026-11-04, before the run; Odin's PC)

**Odin:** "It hasn't been trained at scale. With the GLN the algorithm can scale." Then: "My PC is running. Run that test on 1 GB and check it." On the GPU: "They are slow. Don't use them yet." So it runs on the CPU only.

**Tools** (`cyphalm_trie`, this change):
- `gln-train TEXT OUT [--bytes N] [--tb N] [--online-mb N]` builds the GLN's bit counts over the first N bytes of TEXT, at 2^tb slots per order. The whole network (the LSTM, every neuron and the APM) then learns online over the last `--online-mb` MiB. The GLN is saved to OUT (format "GLN1").
- `context8k --gln-load FILE` and `sample --gln-load FILE` read it instead of building the GLN from wiki train's 7 MiB.
- The gates' fixed parts (the LSTM state and the extra values) are now dotted once per byte, not once per bit. The output is identical (1.9435 on the holdout) and 28% faster.

**The run** (Odin's PC, one core per job):
1. G1: `gln-train` over the first 1 GiB of the web text the 1 TB trie learned (no Wikipedia), `--tb 24 --online-mb 32`.
2. Prediction: `context8k` as it is (the 7 MiB GLN), and `context8k --gln-load G1`, side by side.
3. Writing, all 250 GEN-A dev prompts, SCRIBE v2: W0 (`sample`), `--gln-load G1 --gln-mix geo` and `--gln-load G1 --gln-mix gate`.

**Rules:**
- G1 becomes `context8k`'s GLN if its `online_final_bpc` is at least 0.005 below the 7 MiB GLN's on the same machine.
- Writing: §13.16's rule. A GLN arm reconciles the two if its best temperature is at least W0's best minus 0.02.

**Predictions:**
- G1's online bits per byte, on its own last MiB of web text, end below 2.0.
- On the enwik8 holdout, `gln_alone_bpc` is 1.80–1.95: web text is not Wikipedia, but the GLN keeps learning online over the holdout.
- `online_final_bpc` falls by 0.005–0.015.
- In writing, `geo` is non-inferior and `gate` is not.

**Result** (Odin's PC, 2026-11-04; from `D:\cypha_llm\gln1g\report.txt`):
- **Text:** the first 1 GiB of RedPajama's C4 (shards 00000 and 00001).
- **Training:** `gln-train … --tb 24 --online-mb 32` took 1 h 20 min on one core, with a 1.6 GB peak, and wrote a 403 MB file. Its online bits per byte on the last MiB were 2.004, so the prediction "below 2.0" just missed.
- **Prediction** on the enwik8 holdout, side by side:

  | | 7 MiB GLN (P0) | 1 GiB GLN (P1) |
  |---|---:|---:|
  | GLN alone, bits per byte | 1.9455 | 2.1751 |
  | `online_final_bpc` | **1.6206** | 1.6403 |

  The 1 GiB GLN is worse. By the rule the 7 MiB GLN stays `context8k`'s; both predictions for this step missed. C4 is web text and the holdout is Wikipedia with its XML; 32 MiB of online learning on C4 moved the LSTM and the neurons toward the web, and the holdout's 1 MiB does not move them back.
- **Writing** (SCRIBE v2, 250 dev prompts, best temperature): W0 0.174 at T 0.6, the same as §13.16, so the setup reproduces.
  - `geo` scores 0.111 (0.090 with the 7 MiB GLN): it fails the rule (≥ 0.154). The prediction that `geo` holds up missed.
  - `gate` scores 0.143 (0.117 with the 7 MiB GLN): it fails, as predicted. At T 1.0 it still beats W0, 0.122 against 0.073.
- **Reading.** Scale on the wrong kind of text does not help. The GLN gained on writing (both mixes rose by 0.02–0.03) but lost on prediction, because the test is Wikipedia and the training was the web. The next step is the same test at scale on in-domain text: clean Wikipedia prose that does not overlap enwik8 (for example enwik9 beyond its first 100 MB, cleaned by SCRIBE's cleaner), with the online pass on that text too.

### 13.18 The GLN off by default; a plain byte LSTM instead (2026-11-04)

**Odin:** "Whoa. I was wrong about the GLN." Then: "I want you to turn GLN off by default. I'm thinking maybe just a simple LSTM. Give that a whiz instead of the GLN. It helped the original CyphaLM and that's what PAQ8 uses."

**The GLN is off by default** in `context8k`. The default is now the online chain plus the online final layer (C2): 1.6543 bits per byte, 251 MB. `--gln` turns the GLN back on. Answers are unchanged, as before.

**The LSTM** (`native/include/cypha/online/byte_lstm.hpp`) is the byte LSTM that was inside the GLN, now on its own: embedding 16, hidden 64, a softmax over 256 bytes, truncated BPTT over 20 bytes, Adam. It gives a whole next-byte distribution in one step, so it costs little in scoring and in writing.

**Fast check** (registered before it): the LSTM streams over the 1 MiB holdout, learning online, then `gln_mix.py` scores it in place of the GLN.
- L0: no warm-up. L2: first learning online over wiki train's last 2 MiB.
- Read: the LSTM alone; M1 (the online chain with the LSTM as its last stage) against M0; M3 (plus the final layer, the LSTM an input) against M2.
- **Rule:** the LSTM goes into `cyphalm_trie` (`context8k --lstm`, `sample --lstm`) if M1 − M0 ≤ −0.005 with its 95% interval below 0. Then a C++ run decides the default (registered then), and the 250-prompt writing test decides `sample`.
- **Predictions:** the LSTM alone scores 2.2–2.5 bits per byte. M1 − M0 is −0.005 to −0.015, less than the GLN's −0.027, since the GLN held hashed counts as well. The 2 MiB warm-up helps by at least 0.02 alone.

### 13.19 K5 in `cyphalm_trie sample` (2026-11-04)

**Odin** (on §13.11's decision 1): "do all that other stuff too." So K5 now runs inside CyphaLM.

**What was built.**
- `native/include/cypha/lm/word_kn.hpp` (namespace `cypha::cyphalm::wkn`). It ports the arm K5 of `bench/fuzzy/fuzzy_bake.py`:
  - the tokeniser TOKRE, with Python's `\s`;
  - the vocabulary: 32,000 entries, tokens seen at least twice;
  - the estimator KN: interpolated modified Kneser–Ney of order 5, with the same discounts, continuation counts and backoff;
  - the sampler `generate_one`: the 3,000 most probable words cut at 0.995, the word-level repeat ban, nucleus 0.9 at T 1.0, the `piece()` spacing and the 600-byte stop.
- The 12-word cap. A word may not complete 13 consecutive tokens of the training text, prompt included. The header keeps a hash set of every 13-token window. If the cap leaves no candidate, the most probable word that it allows is written.
- Words outside the vocabulary are `<unk>` in the context, as in Python. `<unk>` is never written. The byte model does not spell new words. That is not built: a sample never needs it, and it was not shown to keep SCRIBE.
- `cyphalm_trie sample … --k5 TEXT [--k5-cap N]`: K5 writes every GEN line (`GEN\ttrie_T1.0\t…`), and the trie and its experts are not built. TEXT is clean prose, a document per line. `GEN_EVAL` prints K5's continuation bits, copy32 and the share of samples with a 13-token training run. The hook is a few lines in `cyphalm_trie.cpp`.
- The selftest checks the tokeniser, `piece()`, the repeat ban and the cap. It also checks that KN sums to one, and that one probability equals a direct count.

**Checks against Python** (its own `train.txt`, `vocab.json` and `train.npy`):
- The vocabulary and the id stream are identical (1,356,468 ids), and so are the entries per order.
- The probabilities agree to about 13 digits.
- The continuation bits per byte on GEN-A dev are 1.549997, the same as `res_K5.json`.
- `fuzzy_bake.py`'s own `generate_one`, run in this container, gives Odin's PC numbers exactly: copy32 0.0402, 13-word runs 0.012, SCRIBE 0.344 [0.311, 0.379]. So SCRIBE and its vectors here match the PC's.

**One bug found and fixed: the order of ties.**
- Many tail words have the same probability, for example those with one left context and no stored context.
- The 0.9 nucleus splits such a block at 55% of steps (25 dev prompts, Python).
- numpy leaves ties in no particular order. The first port ordered them by id, which is frequency, so the most frequent words of a tied block got in. Those are words that follow one word only, such as "Angeles" and "Shrugged", or sentence openers such as "Instead" and "Moreover", which only follow a full stop.
- That port scored 0.286, 0.296, 0.312 and 0.310 on four seeds. Ties are now ordered by a hash salted per step.

**Dev result** (GEN-A dev, 250 prompts, SCRIBE v2, seeds 20261101 + prompt index as in Python):

| | SCRIBE v2 [95%] | surface S | copy32 | samples with a 13-token run |
|---|---|---|---|---|
| Python K5 (`res_K5.json`) | 0.344 [0.311, 0.379] | 0.928 | 0.040 | 0.012 |
| **K5 in `cyphalm_trie` (cap 12)** | **0.353 [0.321, 0.385]** | 0.928 | 0.035 | **0** |
| the same, `--k5-cap 0` | 0.353 [0.321, 0.386] | 0.928 | 0.035 | 0 |

- **Spread over seeds.** The C++ K5 scores 0.309–0.353 over 8 seeds (mean 0.330). Python scores 0.344, 0.344 and 0.310 over 3 (mean 0.333). They are the same sampler.
- **The cap.** It fired at 2 of 31,895 words on the dev seeds and changed 2 samples. SCRIBE did not move. Over 8 seeds, 11 of 2,000 uncapped samples hold a 13-token run (0.55%; Python 3 of 750). With the cap, 0 of 2,000 do.
- **Speed and memory.** Building takes 1 s. The 250 samples take 9 s on 4 threads (about 1.1 ms a word a thread). Python takes about 200 s. The model is 119 MB, and the run peaks at 220 MB.

**Next.** K5 is at its Python level on dev and keeps the cap. The one GEN-A test read against the byte default (BK0, test 0.147 at T 0.6) is the parent's to register and make.

### 13.20 The 1 TB trie with the online content mixer (registered 2026-11-04, before the run; Odin's PC)

**Odin:** "Run a big model on the PC. See what happens with the BPC and text generation." Then: "Do all that other stuff too."

**Arms** (Odin's PC, one core each, side by side; no GLN, so the default chain is online with the final layer):
- B0: `context8k ENWIK8 DOLLY --no-gln` (the trie trained on wiki's 7 MiB).
- B1: `context8k ENWIK8 DOLLY --no-gln --model TRI2_PACKED` (the 1 TB trie, packed, 992 MB).
- S1: `sample ENWIK8 … --prompts 250 --model TRI2_PACKED` on GEN-A dev, SCRIBE v2; against W0 (0.174 at T 0.6, the same machine, §13.17).

**Rules:**
- The 1 TB trie becomes the recommended `context8k` model if B1's `online_final_bpc` is at least 0.005 below B0's. It is a file, not a code default.
- In writing, it becomes the recommended `sample` model if S1's best temperature is at least W0's best minus 0.02 and its copy share is within GEN_TRACK's limit.

**Predictions:**
- B1 is 1.640–1.650 (B0 near 1.654). The old frozen stack gained 0.007 from the 1 TB trie (1.6906 against 1.6979), and the online chain keeps that.
- S1 is 0.15–0.20: a bigger trie of web text does not by itself make sentences.

### 13.21 Content-mixer upgrades without the GLN (registered 2026-11-04, before the runs)

**Odin:** "What upgrades did you have for just the content mixer that scored ~1.65?" Then: "Do all that other stuff too."

`bench/fuzzy/mixer_upgrades.py` on the dump, as `cyphalm_trie`'s defaults (the online chain by exponentiated gradient, short copies on, the online final layer):
- U0: the default, for the same run's reference (the dump gives 1.6538).
- U1: two learning rates. The chain also runs at a slow step (0.005) and a fast one (0.08). The final layer gets both chains' outputs as two more inputs and picks between them, as PAQ runs mixers at several rates.
- U2: the same 80 weight sets in the final layer, chosen by copy's bucket (8) × the previous byte's class (5) × whether InfiniGram matched at least 16 bytes (2). Today they are chosen by InfiniGram's bucket × the previous byte's class. More gates are not tried (gate24).
- U3: U1 and U2 together.

**Rule:** an arm goes to C++ if it beats U0 by at least 0.003 bits per byte with its 95% interval below 0. Then a C++ run decides the default on §13.15's rules.

**Predictions:** U1 is −0.003 to −0.010. U2 is within ±0.003. U3 is about U1.

### 13.22 The online mixer in writing (registered 2026-11-04, before the run)

**Odin:** "Online on every part." So far the online mixer only scores. `sample` still writes with lambdas fit once on the holdout's first half and frozen.

`sample --online-lambdas`: every chain stage fit through `mix_into` (Kneser–Ney, class, copy, …) learns online over the whole holdout. The sampler then writes with the lambdas' last state. Copy's short matches stay at 0 in writing (B4B).

**Arms:** W0 (`sample`) and SO (`sample --online-lambdas`), all 250 GEN-A dev prompts, SCRIBE v2, side by side on this container.

**Rule:** `--online-lambdas` becomes `sample`'s default if SO's best temperature is at least W0's best minus 0.02 (no worse, within noise), with copy share within GEN_TRACK's limit.

**Prediction:** SO is within ±0.02 of W0. The lambdas move little once learned, and writing is set more by the sampler than by small changes in the weights.

**§13.18 and §13.21 fast-check results** (the dump; bits per byte on the second half):

| Arm | Bits per byte | Against [95%] |
|---|---:|---|
| LSTM alone (no warm-up; 2 MiB warm-up) | 2.6447; 2.5326 | — |
| The online chain + the LSTM stage | 1.6547 | the chain 1.6692: **−0.0145 [−0.0182, −0.0111]** |
| + the final layer, the LSTM an input | **1.6298** | the final layer 1.6538: −0.0240 [−0.0296, −0.0191] |
| U1: two learning rates | 1.6512 | U0 1.6538: −0.0026 [−0.0030, −0.0022] |
| U2: final-layer sets by copy × byte class × long match | 1.6493 | U0: **−0.0046 [−0.0052, −0.0038]** |
| U3: U1 + U2 | **1.6470** | U0: **−0.0068 [−0.0077, −0.0058]** |

- **The LSTM passes its rule** (M1 − M0 ≤ −0.005, interval below 0). It goes to C++ as `--lstm`. It gives about 70% of the GLN's gain (−0.024 against −0.034 with the final layer), holds about 1 MB against the GLN's 101 MB, and runs faster.
- The 2 MiB warm-up helps the LSTM alone (−0.11) but not the mixture (−0.0002). It is left out.
- **Predictions:**
  - The LSTM alone at 2.2–2.5 missed; it scored 2.53–2.64.
  - M1 − M0 between −0.005 and −0.015 held (−0.0145).
  - The warm-up gaining at least 0.02 held alone, but not in the mixture.
- **U2 and U3 pass, U1 alone does not.** Predictions: U1 −0.003 to −0.010 missed (−0.0026); U2 within ±0.003 missed (−0.0046); U3 near U1 missed (−0.0068). U3 goes to C++ as `--final-gate copy --two-rates`.

**C++ runs** (registered before them; this container, one core each, side by side):
- D0: `context8k` (the default: online chain and final layer, no GLN).
- D1: `--lstm`.
- D2: `--final-gate copy --two-rates`.
- D3: `--lstm --final-gate copy --two-rates`.

**Rules** (each also needs answer tallies identical to D0's and selftest passing):
- `--lstm` becomes the default if D1's `online_final_bpc` is at least 0.01 below D0's, with wall time within +35%.
- `--final-gate copy --two-rates` becomes the default if D2 is at least 0.003 below D0, with wall time within +10%.

**Predictions:** D1 near 1.630; D2 near 1.647; D3 near 1.623.

**The LSTM in writing** (registered before the run). `sample --lstm`: the byte LSTM learns online over the holdout. One lambda is fit over the sampler's chain on the first half. Its whole distribution is mixed in last; while writing it reads the prompt and every byte written, learning nothing. `--lstm-mix gate|geo` uses §13.16's mixing.
- **Arms:** LS1 `--lstm` (linear), LS2 `--lstm-mix gate`, LS3 `--lstm-mix geo`. All 250 GEN-A dev prompts, SCRIBE v2. Against W0 from §13.22's run (same machine and seeds).
- **Rule:** the best arm that is non-inferior (best temperature ≥ W0's best − 0.02, copy share within the limit) becomes `sample`'s default way of writing with the LSTM in, as Odin asked for online learning on every part.
- **Prediction:** the LSTM is weaker than the GLN (2.64 bits per byte alone), so its lambda is small. LS1 and LS2 are non-inferior; LS3 is not.

### 13.23 A wider GLN at 1 GB (registered 2026-11-04, before the run; Odin's PC)

**Odin:** "Try one more GLN at 1 GB on my PC. Set the width optimally of the GLN. It rises per data. 3 levels is best. I just want to see."

**The width.** The GLN has three levels: layer 1, layer 2, and an output neuron per bit position, then an APM. It was 32 neurons × 16 weight sets per bit position, then 4 × 4. That suits 64 KiB. At 1 GiB of counts and 32 MiB of online learning, each weight set sees far more bytes, so the width rises:
- layer 1 is 128 neurons × 64 sets (6 half-spaces);
- layer 2 is 16 neurons × 16 sets (4 half-spaces);
- the output stays one neuron per bit position.

Even so, each layer-1 weight set sees about half a million bits. The cost is about 1.7× per byte, mostly in the gates. The LSTM is unchanged. `gln-train` takes `--neurons --hs --neurons2 --hs2`. The defaults are unchanged (1.9435 on the holdout), and the file format is now "GLN2".

**The run** (the same 1 GiB of C4 as §13.17, so only the width changes):
- GW: `gln-train text1g.txt glnw.gln --tb 24 --online-mb 32 --neurons 128 --hs 6 --neurons2 16 --hs2 4`.
- P2: `context8k --no-gln --gln-load glnw.gln` (the default mixer; against §13.17's P0 1.6206 and P1 1.6403).
- G3: `sample --gln-load glnw.gln --gln-mix gate`, 250 dev prompts, SCRIBE v2 (against the narrow GLN's 0.143 and W0's 0.174).
- And §13.11's half-space read, which ran on the PC on 1 November and was never collected, is read from its report.

**No rule** ("I just want to see"). It is reported only; a default would need its own registered run.

**Predictions:**
- GW's online bits per byte on C4's last MiB are at least 0.03 below the narrow GLN's 2.004.
- On the enwik8 holdout it is still worse than the 7 MiB GLN, because the domain is wrong.
- G3 is within 0.02 of 0.143.

**K5's one test read** (registered 2026-11-04, before it; §13.11's decision). `sample --k5 /root/work/fz/train.txt` on GEN-A's 250 test prompts, SCRIBE v2 (GEN_TRACK's ladder), read once.
- **Against:** the byte default BK0 on test, 0.147 [0.115, 0.181] at T 0.6 (T4d's read).
- **Rule:** K5 becomes CyphaLM's recommended writer (`scripts/gen_eval.sh` and the docs use `--k5`) if its test interval lies wholly above BK0's (lower bound above 0.181), with no sample holding a run of more than 12 copied words. The byte sampler stays for answers, `chat` and anything without the clean text.
- **Prediction:** 0.30–0.37, as on dev.

**Result** (read once, 2026-11-04, this container): K5 on GEN-A test scores **SCRIBE v2 0.305 [0.276, 0.338]**, surface S 0.927 [0.923, 0.932], copy32 0.0335, and **0 of 250** samples hold a run of more than 12 copied words (the cap fired once in 31,773 words). It wrote the 250 continuations in 18 s.
- Against BK0's 0.147 [0.115, 0.181] the intervals are disjoint: the lower bound 0.276 is above 0.181. **The rule holds. K5 is CyphaLM's recommended writer.**
  - To write: `cyphalm_trie sample ENWIK8 OUT --prompts-file … --k5 CLEAN.txt`.
  - The clean text comes from SCRIBE's cleaner: `scribe docs enwik:ENWIK8:0:16 --out DIR`, as `fuzzy_bake.py prep` makes it.
  - The byte sampler stays for answers, `chat`, and anywhere without that text.
- The prediction (0.30–0.37) held.
- This is the shuffled-sentence rung on SCRIBE's ladder (G2's level, §5). The byte default sits below it. Human text scores 0.923.

**C++ results, §13.18 and §13.21** (2026-11-04, this container, D0–D3 side by side). Every arm's answer tallies match D0's. D0 matches C2 line for line (the refactors change nothing).

| Arm | `online_final_bpc` | Against D0 | Wall | Peak RSS |
|---|---:|---:|---:|---:|
| D0: the default | 1.6543 | — | 1474 s | 251 MB |
| D1: `--lstm` | 1.6311 | −0.0232 | 1556 s (+6%) | 255 MB |
| D2: `--final-gate copy --two-rates` | 1.6489 | −0.0054 | 1493 s (+1%) | 265 MB |
| D3: both | **1.6238** | **−0.0305** | 1544 s (+5%) | 269 MB |

- The LSTM alone scores 2.6526; the chain with it, 1.6579.
- **Both rules hold.** The LSTM (−0.023, at least 0.01 needed; +6% wall) and the two rates with the copy gate (−0.005, at least 0.003 needed; +1% wall) become `context8k`'s defaults. The default now reports 1.6238, against 1.6979 frozen before §13.15: **−0.074**, with no GLN. It takes 269 MB, against 371 MB with the GLN.
  - `--no-lstm`, `--one-rate` and `--final-gate ig` restore D0's lines.
  - `glnx_prep.sh` passes `--no-lstm --one-rate` with its flags, so the dump keeps the frozen chain's columns.
- Predictions: D1 near 1.630 (1.6311), D2 near 1.647 (1.6489) and D3 near 1.623 (1.6238) all held.

**§13.22 result: the online mixer in writing.** All 250 dev prompts, by temperature:

| Arm | T 0.6 | T 0.8 | T 1.0 |
|---|---:|---:|---:|
| W0: `sample` | **0.174** [0.137, 0.209] | 0.155 | 0.073 |
| SO: `sample --online-lambdas` | 0.147 [0.117, 0.180] | 0.127 | 0.084 |

- SO's best is 0.027 below W0's, beyond the 0.02 allowed. **`--online-lambdas` stays a flag.** The prediction (within ±0.02) missed.
- The lambdas learned online over the holdout's end fit that text, not the prompts.
- The writer that wins is K5 (§13.19) in any case, and it has no lambdas.

**§13.18 writing result** (all 250 dev prompts; the LSTM's lambda is 0.055):

| Arm | T 0.6 | T 0.8 | T 1.0 |
|---|---:|---:|---:|
| W0: no LSTM | **0.174** [0.137, 0.209] | 0.155 | 0.073 |
| LS1: linear | 0.110 | 0.119 | 0.068 |
| LS2: the plausibility gate | 0.137 | **0.155** [0.122, 0.188] | **0.133** [0.102, 0.164] |
| LS3: product of experts | 0.119 | 0.100 | 0.078 |

- **LS2 is non-inferior:** 0.155 ≥ 0.154. By the rule, `sample` writes with the LSTM in by default, mixed in through the gate. Online learning reaches writing, and SCRIBE does not measurably fall.
- At T 1.0 the gate is far better than W0 (0.133 against 0.073), as with the GLN: it cuts the tail that T 1.0 lets in.
- `--no-lstm` gives W0's text; `--lstm-mix lin|geo` gives the other mixes.
- **Predictions:** LS1 non-inferior missed; LS2 non-inferior held; LS3 not non-inferior held.
- The byte sampler is now CyphaLM's second writer; K5 (§13.19) writes best.

### 13.24 The LSTM wired into everything, K5 included (registered 2026-11-04, before the runs)

**Odin:** "Any way we can get K5 to work with the LSTM? Let's treat the LSTM like we treated the GLN and wire it into everything and reuse it as much as possible and see what we can get."

**Where the LSTM already is:** inside the GLN (its state gates every neuron; its bit probability is an input); in `context8k` as the chain's last stage and a final-layer input (the default, §13.18); in `sample` through the plausibility gate (the default, §13.18).

**New:**
1. **K5 with the LSTM** (`sample --k5 TEXT --k5-lstm FILE [--k5-lstm-mix geo|lin] [--k5-lstm-w W]`). K5 still picks the candidate words and keeps its nucleus 0.9: the plausibility gate. Inside the nucleus the byte LSTM scores each word by the bytes it would write (spacing included), after the prompt and everything written so far. The two combine either as a product of experts, p_K5^(1 − w) × p_LSTM^w, or linearly, (1 − w) p_K5 + w p_LSTM, each normalised over the nucleus. The LSTM reads the prompt and every written word, learning nothing while it writes.
   - The LSTM learns K5's clean prose first: `cyphalm_trie lstm-train /root/work/fz/train.txt clean.lstm`, the whole 7 MB online, saved as "LST1".
2. **The LSTM's state picks the final layer's weight sets** (`context8k --final-gate lstm`), as the GLN's gates read it. The 80 sets are the signs of 4 fixed random projections of the state that predicts the byte (16) × the previous byte's class (5). It replaces the copy × class × long-match context.

**Arms:**
- Writing, all 250 GEN-A dev prompts, SCRIBE v2:
  - K5: K5 alone, rerun with this build;
  - KL1: geo, w 0.25;
  - KL2: geo, w 0.5;
  - KL3: lin, w 0.25.
- Prediction: D5 is `context8k --final-gate lstm`, against D3's 1.6238.

**Rules:**
- The best KL arm that is non-inferior to K5 (at least K5's dev score minus 0.02) makes the LSTM part of K5 by default. It then gets one test read, and stays if it is at least K5's test 0.305 minus 0.02.
- `--final-gate lstm` becomes the default if D5 is at least 0.003 below D3.

**Predictions:**
- KL1 is non-inferior (0.33–0.37): the LSTM only reranks words K5 already allows, and it has learned the same prose.
- KL2 and KL3 fall below K5.
- D5 is within ±0.003 of D3.

**Results** (2026-11-04, this container, 250 dev prompts):

| Arm | SCRIBE v2 [95%] | Surface S | copy32 | Mean word length | Distinct words |
|---|---:|---:|---:|---:|---:|
| K5 | **0.353** [0.318, 0.384] | 0.928 | 0.035 | 4.31 | 0.171 |
| KL1: geo, w 0.25 | 0.149 [0.120, 0.181] | 0.970 | 0.004 | 3.48 | 0.076 |
| KL2: geo, w 0.5 | 0.018 | 0.958 | 0.001 | — | — |
| KL3: lin, w 0.25 | 0.106 | 0.935 | 0.017 | — | — |

- No KL arm is non-inferior, so K5 stays without the LSTM. All three predictions missed.
- **Why: a length bias.** A byte model scores a word by the product of its bytes' probabilities, so short, common words win. With the LSTM, the mean word is 3.5 letters against 4.3, and half as many distinct words are used ("as a group of the same name; most of the north to the …").
- The LSTM is about 90× slower inside K5: 125 ms a word, because it scores every word in the nucleus.
- **D5** (`--final-gate lstm`) scores 1.6345 against D3's 1.6238, so it is worse; the copy × class × long-match context stays. The prediction (within ±0.003) missed.

**The length-bias fix** (registered before the run). The byte LSTM should add only what the context tells it, not its own prior on words. DExperts' form: p_K5 × (p_LSTM(word | context) ÷ p_LSTM(word | a space))^w, normalised over K5's nucleus (`--k5-lstm-mix pmi`). The prior is the LSTM's probability of the word's bytes after reading one space, so it cancels the length and frequency bias, which K5 already weighs.
- **Arms:** KL4 pmi w 0.5; KL5 pmi w 1.0 (250 dev prompts).
- **Rule:** as above (non-inferior to K5's 0.353, then one test read).
- **Prediction:** KL4 is non-inferior, and its mean word length and distinct-word share are within 5% of K5's.

**The length-bias fix: result** (250 dev prompts):

| Arm | SCRIBE v2 [95%] | Surface S | copy32 | Mean word length | Distinct words |
|---|---:|---:|---:|---:|---:|
| K5 | **0.353** [0.318, 0.384] | 0.928 | 0.035 | 4.31 | 0.171 |
| KL4: pmi, w 0.5 | 0.221 [0.189, 0.257] | 0.961 | 0.055 | 4.45 | 0.158 |
| KL5: pmi, w 1.0 | 0.173 [0.152, 0.197] | 0.960 | 0.056 | 4.74 | 0.140 |

- The fix removes the length bias: the mean word is back to 4.45 letters. It raises SCRIBE from 0.149 (KL1) to 0.221. But no arm is non-inferior to K5, so **K5 stays without the LSTM**.
- The prediction missed: KL4's word length is within 5% of K5's (+3%), but its distinct-word share is 8% lower and its SCRIBE is far lower.
- With the LSTM, surface S rises above K5's (0.96 against 0.93), and copying rises too (0.055 against 0.035). The LSTM pulls K5 towards smooth phrases it has seen. SCRIBE's other axes fall.
- K5's output without `--k5-lstm` is unchanged by the hook: the rerun matches §13.19's text line for line (0.353).

**Where the LSTM stays, by the measurements:**
- In prediction, `context8k`'s chain and final layer: the default (−0.023).
- In the byte sampler, through the gate: the default, non-inferior.
- Not in K5: every mix tried lowers SCRIBE. Not as the final layer's gate context either: +0.011 bits per byte.

The flags stay (`--k5-lstm`, `--k5-lstm-mix`, `--final-gate lstm`, `lstm-train`) for a stronger LSTM, such as one trained at scale on Odin's PC.

**The LSTM out of writing** (Odin, 2026-11-04: "Take the LSTM out, let's just go with K5"). `sample` writes without the LSTM again (W0's text); `--lstm` or `--lstm-mix MODE` puts it back. K5 is the writer. The LSTM stays in `context8k`'s prediction, where it gains 0.023 bits per byte.

### 13.25 The record, and the work moved to Odin's PC (2026-11-04)

**Odin:** "Do all of it and make sure you document everything we have tried and the results. Use sub agents." Then: "Do it all on my PC."

- **The record** of everything tried for generation and online learning, 1–4 November, with every result and decision: [reports/GENERATION_ONLINE_RECORD.md](reports/GENERATION_ONLINE_RECORD.md).
- **Four sessions on Odin's PC**, each registering its rules before its runs and writing its own report:
  1. K5 upgrades: tuning; online cache and counts; order 6 and word classes; a sentence gate; a content plan. Branch `wip/k5-upgrades`, report [reports/K5_UPGRADES.md](reports/K5_UPGRADES.md).
  2. K5 on clean Wikipedia from enwik9, past enwik8's bytes, checked for overlap with GEN-A and SCRIBE. Branch `wip/k5-scale`, report [reports/K5_SCALE.md](reports/K5_SCALE.md).
  3. Online learning in the mixer: an online n-gram expert, document-aware learning, a bigger LSTM. Branch `wip/online-upgrades`, report [reports/ONLINE_LEARNING_UPGRADES.md](reports/ONLINE_LEARNING_UPGRADES.md).
  4. Whole-distribution mixing with APM/SSE, PAQ-style. Branch `wip/full-mix`, report `docs/reports/FULL_DISTRIBUTION_MIXING.md` on that branch (not merged yet).
- The cloud session merges each branch and adds its results here and to the record.

**Results from Odin's PC** (2026-11-04; merged to main):

| Session | What changed | GEN-A dev | GEN-A test (one read) | Report |
|---|---|---|---|---|
| K5 upgrades | Nucleus 0.8; 256 exchange classes as K5's backoff floor; an 8-candidate sentence gate that keeps the sentence whose words co-occur most with the prompt's in K5's own training text | 0.498 [0.469, 0.528]; 4 seeds 0.493–0.505 against K5's 0.312–0.362 | **0.524 [0.493, 0.552]** against 0.305 | [K5_UPGRADES.md](reports/K5_UPGRADES.md) |
| K5 at scale | 330 MB of clean Wikipedia from enwik9 past enwik8, overlap with GEN-A and SCRIBE removed (0 on recheck) | 4-seed mean 0.390 against 0.344 | **0.380 [0.353, 0.406]** against 0.305 | [K5_SCALE.md](reports/K5_SCALE.md) |

- **Both are adopted by their registered rules.**
  - K5's default is now the upgraded sampler. `--k5-base k5` writes §13.19's text line for line, checked in the cloud.
  - The recommended training text is `train_all.txt`, 330 MB. It needs about 7 GB of RAM and a 65 s build. (For the old K5 only: the upgraded K5 does not gain from it, §13.29.)
  - The cloud check of the new default gives 0.498 [0.469, 0.528] on dev, the same as the PC, with copy32 0.038 and 0 samples over the 12-word cap.
- **What failed (K5 upgrades):**
  - The word cache and dynamic counts lower K5's bits the most (to 1.488), but they lower SCRIBE. Prediction is not generation, again.
  - Orders 6 and 7 add nothing.
  - A 65,536-word vocabulary lowers SCRIBE.
  - The content plan was skipped: GEN-E's fact store is in the cloud, not on the PC.
- **Copying at scale.** On 330 MB, copy32 rises to 0.17: 17% of written bytes match some 32-byte span of the training text. The 12-word cap still holds.
- **A caution on the judge (raised by the PC session).** The sentence gate scores co-occurrence with the prompt. SCRIBE's coherence axis also rests on co-occurrence (word vectors), and most of the gate's gain is on that axis (D_C 4.15 → 3.41). Part of the gain may be two co-occurrence measures agreeing rather than better text. By §1.3 it needs a judge of another kind: Odin's own blind ratings, or the second benchmark. Until then the gate's gain is provisional.
- **Not yet run:** the two together (the upgraded K5 on the 330 MB text).
- **Seeds:** `--k5-seed S` is absolute; a value below 20261101 is an offset, so both reports' seeds reproduce.

### 13.26 Online learning upgrades and SCRIBE v3 (2026-11-04; results)

**Online learning** (Odin's PC; [reports/ONLINE_LEARNING_UPGRADES.md](reports/ONLINE_LEARNING_UPGRADES.md)). Rules and predictions were registered in the report before each phase.

| Arm | What | Bits per byte | Against the old default |
|---|---|---:|---:|
| B0 | the old default (D3) on the PC | 1.6232 | — |
| N2 | an online n-gram expert, PAQ-style (orders 1–6, 8, 12; 24 table bits) | 1.5913 | −0.0319 |
| L3 | the byte LSTM at hidden 256, window 20 | 1.6026 | −0.0206 |
| D1, D2 | larger online steps after each page start | 1.6246, 1.6239 | +0.0014, +0.0007 |
| **C (N2 + L3)** | **the new default** | **1.5793** | **−0.0439** |

- **Adopted by rule 1.** `context8k` now runs the n-gram and the 256/20 LSTM. Answers are unchanged.
- The old default is `--no-ngram --lstm-size 64`. It reproduces B0 byte for byte; the new default reproduces C byte for byte.
- **The cost:** +61% wall time (45 against 28 minutes) and peak RSS 828 MB against 247 MB. The time is the bigger LSTM; the memory is the n-gram's tables. `--ngram-tb 22` saves 440 MB for 0.007 bits per byte.
- **What failed:** document-aware steps, and long BPTT windows (50, 100).
- **Untested:** whether the n-gram or the bigger LSTM help `sample` or K5. By every earlier result (§13.24), they will not.
- The cloud build of the merge passes `cyphalm_trie selftest`.

**SCRIBE v3** ([reports/SCRIBE_V3.md](reports/SCRIBE_V3.md)). Odin: "Make another two metrics with different maths that complement each other and take the average of all 3. Make it fool proof."

- **SURPRISE**: the text's code length under a model of the pack (Kneser–Ney trigram, spelling models, a cache and a match model). Two-sided: too predictable loses as surely as salad.
- **MATCH**: matching statistics against a suffix array of the pack, the text's own LZ76 complexity, and echo of the prompt.
- **v3** is the mean of v2, SURPRISE and MATCH. It also reports the minimum of the three and a `disagree` flag.
- **The ladder (fresh data, read once):** v3 passes all 5 pairs, SURPRISE all 5, MATCH 4 (human > sentences shuffled 0.940).
- **Not fool-proof yet.** The attack rule passed 5 of 9:
  - caught: copying, echo, loops, co-occurrence collages from the pack;
  - missed: copied prompt sentences then a continuation (v3 0.948; MATCH alone catches it), keyword stuffing into human text (0.825) and into a generic paragraph (0.457), a generic paragraph (0.930; SURPRISE and MATCH catch it, the mean dilutes it);
  - invisible to all three: sentences reordered within a text.
- **K5 cross-check:** old K5 → upgraded K5 is v2 +0.145, SURPRISE −0.106, MATCH −0.009, v3 +0.011 (inside the intervals). **The new judges do not confirm the sentence gate's gain.** SURPRISE finds the gated text too predictable: 7.4 bits a word against human text's 9.5. The gate stays provisional (§13.25) until Odin's blind ratings.
- v2 is unchanged and stays the score of record for comparisons with earlier results. v3, `min3` and `disagree` are reported next to it from now on.
- No fresh enwik8 range is left for another frozen read of a SCRIBE change. A v3.1 can take its fresh read from enwik9 past enwik8. SCRIBE's design never read it. K5's 330 MB training text comes from there, which does not matter for the judge's own ladder and attacks.

### 13.27 Results: the 1 TB trie (§13.20), the wider GLN (§13.23) and 1 November's runs (2026-11-04)

Collected from the reports on Odin's PC: `D:\cypha_llm\big1tb\report.txt`, `D:\cypha_llm\glnwide\report.txt` (with 1 November's report appended word for word) and `D:\cypha_llm\gln1g\report.txt`. Every job exited cleanly before the PC's WSL restart.

**§13.20, the 1 TB trie.** The binary was the morning's default without the GLN (C2: online chain and final layer, no LSTM, no n-gram).

| | B0: 7 MiB wiki trie | B1: 1 TB trie |
|---|---:|---:|
| Trie only | 2.3161 | 3.1146 |
| Trie + BKN | 2.0278 | 2.0408 |
| Last chain line | 1.6704 | 1.6638 |
| **`online_final_bpc`** | **1.6534** | **1.6473** |
| Wall, peak RSS | 18:15, 251 MB | 13:50, 1.22 GB |

- **B1 is 0.0061 lower: the rule (0.005) holds.** The 1 TB trie is the recommended `context8k` model file. It is a file, not a code default.
- The prediction (B1 1.640–1.650) held.
- Answers are the same for both.
- The two packed 1 TB files (`trie_1tb.pk`, `model2/trie_model2.pk`) differ only in the trie-only line (3.1146, 3.1132).
- **Not yet measured:** the 1 TB trie under the current default (n-gram and LSTM 256/20). The n-gram already covers much of what a bigger trie adds, so the gain may shrink.

**Writing with the 1 TB trie** (S1, `sample`, SCRIBE v2 on 250 dev prompts):

| T | S1 | S1, model 2 |
|---|---|---|
| 0.6 | **0.138** [0.106, 0.173] | 0.141 [0.110, 0.176] |
| 0.8 | 0.118 | 0.114 |
| 1.0 | 0.082 | 0.093 |

- The best is 0.138, below the rule's 0.154 (W0's 0.174 − 0.02). **Not recommended for writing.** The prediction (0.15–0.20) missed.
- Its text still drifts into wiki markup and lists.

**§13.23, the wider GLN at 1 GiB** (reported only; no rule).
- GW trained in 4 h 02 min on one core: 1.58 GB peak, a 409.5 MB file. Its online bits per byte on C4 held at 1.97–1.99. The last MiB is 1.990, against the narrow GLN's 2.004.
  - 0.014 lower. **The prediction (at least 0.03 lower) missed.**
- On the enwik8 holdout (P2b, `--gln-load` without `--no-gln`; the registered P2 had `--no-gln`, which switches the GLN off, and so matched B0 exactly):

| | 7 MiB GLN (P0) | Narrow, 1 GiB (P1) | Wide, 1 GiB (P2b) |
|---|---:|---:|---:|
| GLN alone | 1.9455 | 2.1751 | 2.1289 |
| `online_final_bpc` | **1.6206** | 1.6403 | 1.6359 |

  - Wider helps a little (−0.004 against P1) but stays worse than the 7 MiB GLN. **The prediction held.**
- Writing (G3, the gate mix): 0.143 at T 0.6, 0.131 at T 0.8, 0.124 at T 1.0. The same as the narrow GLN's best (0.143). **The prediction held.** W0 is 0.174 and K5 0.524.
- **The GLN stays off.** Width and scale on web text do not fix the domain.

**1 November's runs, collected at last:**
- **R3:** frozen chain 1.6971, final layer 1.6678. The two binaries' dumps are identical.
- **Half-spaces in the GLN** (bits per byte):

| Half-spaces | 64 KiB | 512 KiB |
|---:|---:|---:|
| 2 | 2.0960 | 2.0173 |
| 4 | **2.0821** | 2.0021 |
| 6 | 2.0854 | **1.9967** |
| 8 | 2.0943 | 1.9982 |

  - The best count rises with data, from 4 at 64 KiB to 6 at 512 KiB. §13.23's choice of 6 for layer 1 fits.
- **GLNB writing:** best 0.046, against K5's 0.241 in the same run.

### 13.28 SCRIBE v3.1 (2026-11-04; results)

Odin's ask was "fool proof". v3 missed 4 of 9 attacks (§13.26), so v3.1 went after them. Full record: [reports/SCRIBE_V31.md](reports/SCRIBE_V31.md).

- **What changed:**
  - **The geometric mean of the three judges** (v2, SURPRISE 3.1, MATCH 3.1) instead of the plain mean. One judge's confident catch is no longer diluted.
  - **New statistics for words that fit neither neighbour,** to catch keyword stuffing.
  - **A group for text that is too bland,** to catch the generic paragraph.
  - **Copied prompt sentences counted as 64-byte matches,** not repeated names.
- **Protocol:**
  - Designed on dev A and confirmed on dev B, with no change after B.
  - Eight new attacks were written before any design work and never run on development data.
  - Frozen before the fresh read. The fresh read was enwik9 [100, 124) MiB, past enwik8, read once.
  - A registered near-duplicate filter dropped 2 of 2,106 documents.
- **Results (fresh):**
  - Ladder 5 of 5 (worst pair 0.968).
  - **Registered attacks 9 of 9** (v3: 5 of 9). Keyword stuffing is only just caught (0.953).
  - Human text 0.904 (v3: 0.906).
  - All four rules hold.
- **Still not fool-proof: the held-out attacks pass 2 of 8.**
  - Caught: stuffing with frequent words every 3rd word, and prompt sentences scattered through the text.
  - Missed:
    - stuffing every 7th word, or at random gaps;
    - one of four new generic paragraphs;
    - a template that repeats the prompt's names;
    - another document with the prompt's title swapped in;
    - loosely related fluent sentences.
  - Sentence order is still invisible. Truth is not checked.
  - Stuffing is caught by its rate, not its pattern.
- **K5 cross-check:** old K5 → upgraded K5 is v3.1 +0.048, with disjoint intervals. Most of that comes from v2 (+0.145); SURPRISE 3.1 moves −0.090 and MATCH 3.1 +0.008. **The sentence gate's gain is still not confirmed by a judge of another kind.**
- **Cost:** 448 texts a second and 491 MB, against v3's 518 and 493.
- **Byte for byte:** v1, v2 and v3 are unchanged. The agent checked them all; the cloud spot check repeated dev A for v3 and dev B for v2.
- **Use:** report v2 (the score of record), v3.1, and the three parts. A text-level judge built from statistics can be gamed by anything fluent and on-topic. Odin's blind ratings, or a judge of another kind, stay the check for any claimed gain in writing.

### 13.29 Results: whole-distribution mixing, and the upgraded K5 at scale (2026-11-04)

**Whole-distribution mixing, PAQ-style** (Odin's PC; [reports/FULL_DISTRIBUTION_MIXING.md](reports/FULL_DISTRIBUTION_MIXING.md)). `context8k --full-mix` builds each expert's whole 256-byte distribution, replays today's chain and final layer on them exactly (arm F equals `online_final_bpc`, difference 0), then mixes them nonlinearly:
- bitwise logistic mixing with four weight sets;
- geometric mixing with learned exponents;
- adaptive probability maps (SSE) after each.

The headline arm ABs mixes the two per bit. Settings were tuned by replay on the holdout's first half only. Each read was registered before it ran.

| Run | Baseline `online_final_bpc` | ABs | Change | Time | Peak RSS |
|---|---:|---:|---:|---|---|
| X1, the old default (no n-gram, LSTM 64) | 1.6232 | 1.5499 | −0.0733 | +77% | +222 MB |
| **X2, main's default** | **1.5793** | **1.5051** | **−0.0742** | +99% (3434 s against 1726 s) | 1098 MB (+224) |

- **Both rules hold.** Answers and every other line are unchanged. `cyphalm_trie selftest` passes on the cloud build of the merge.
- Normalising the final layer alone gains nothing (1.5793). All the gain comes from the nonlinear mixing: SSE alone −0.025, the bitwise mixer with SSE −0.060, the two together −0.074.
- **It is behind the flag `--full-mix`, off by default.** It doubles `context8k`'s time. Turning it on by default is Odin's call. Folding the pass into the final layer's loop would remove most of the extra time.
- Risks:
  - One 1 MiB holdout, and no bootstrap interval (per-byte bits were not kept).
  - Any change to the chain or the final layer must be copied into `full_mixer.hpp`; arm F shows it if not. `--gln`, `--doc-boost` and `--final-gate lstm` are refused with it.
- `sample` does not use it. Writing from it would need the extra experts at every written byte and frozen learned state, and it would face the SCRIBE test every mixed-in predictor has failed so far.

**The upgraded K5 on 330 MB** (Odin's PC; [reports/K5_SCALE.md](reports/K5_SCALE.md), "The upgraded K5 at scale"). Registered before the reads: adopt it if its 4-seed dev mean beats the upgraded K5 on 7 MB by at least 0.02.

| Arm | v2 (4 seeds) | SURPRISE | MATCH | v3 | copy32 | Build, peak RAM |
|---|---|---:|---:|---:|---:|---|
| U8: upgraded K5, 7 MB | **0.500** (0.496–0.505) | 0.595 | 0.578 | 0.558 | 0.043 | 16 s, 0.30 GB |
| C1: upgraded K5, 330 MB | 0.479 (0.462–0.502) | 0.596 | 0.565 | 0.547 | 0.236 | 170 s, 8.67 GB |

- **The rule fails** (−0.021). No test read. **The upgraded K5 stays on the 7 MB enwik8 text.** This replaces §13.25's recommendation of `train_all.txt`, which was for the old K5.
- Copying rises to a quarter of written bytes (copy32 0.236). The 12-word cap held.
- **SCRIBE v3 does not confirm the old K5's gain from more text either:** v2 +0.046, v3 +0.001, SURPRISE −0.034. With §13.26's finding on the sentence gate, both of 4 November's writing gains rest on v2 alone.

### 13.30 The upgrades optimised, and the full mix on by default (2026-11-04)

**Odin:** "See if you can't optimise all those upgrades before going in." Two sessions on Odin's PC. Each change had to leave every number unchanged, checked with `cmp` on the whole `context8k` output.

**The LSTM and the n-gram** ([reports/ONLINE_LEARNING_UPGRADES.md](reports/ONLINE_LEARNING_UPGRADES.md), "Optimisation"):
- The LSTM's dot products run two rows per SSE2 register. Each row keeps its own summation order, so the numbers are exact.
- Gradients are summed once per window, with a vectorised Adam. The softmax is reused.
- A pool of 4 threads runs from hidden size 128 up.
- The n-gram's three arrays are packed into one 5-byte slot, its keys are prefetched, and pretraining runs on threads.
- **Results:**
  - The default `context8k` went from 32:41 to **20:07** (−38%) at the same load. The LSTM 256 went from 937 s to 218 s, the n-gram from 30 s to 7 s. Memory is unchanged.
  - The cloud repeated the check on its own build: 65,536 LSTM-256 distributions and 262,144 n-gram probabilities were bit for bit identical to main's, on 1 and on 4 threads.
- **Failed:** `--ngram-tb 23` saves 294 MB but is 0.0034 worse, against 0.001 allowed.
- `CYPHA_LSTM_THREADS=1` and `CYPHA_NGRAM_THREADS=1` give the same output on a busy machine.

**The full mix** ([reports/FULL_DISTRIBUTION_MIXING.md](reports/FULL_DISTRIBUTION_MIXING.md), "Optimisation and default"):
- It used to train a second 256 LSTM and a second n-gram in its own pass. Now it reads the instances `context8k` already runs, inside their byte loops. Expert rows are computed ahead on 6 threads.
- Similar-context is still built a second time, because the chain's copy has read the whole holdout.

| Run (same load) | Wall | Peak RSS | Lines |
|---|---:|---:|---|
| main, `context8k` | 2063 s | 874 MB | reference |
| main, `--full-mix` | 3773 s | 1057 MB | the old full mix |
| **new default (full mix on)** | **2075 s** | **1161 MB** | full-mix lines = X2's (ABs 1.5051); all others = main's |
| new, `--no-full-mix` | 1824 s | 875 MB | = main's |

- **The full mix is on by default.** It now costs about +1% time instead of +83%, and +287 MB.
  - `--no-full-mix` turns it off.
  - It switches itself off, with one note, under `--gln`, `--doc-boost`, `--final-gate lstm`, `--frozen-mix`, `--no-online-final` and `--bkn-way a`.
  - `glnx_prep.sh` passes `--no-full-mix`.
- **A new last line, `context8k_bpc`, is the headline: 1.5051 on Odin's PC.**
- The LSTM thread is now the bottleneck. The default uses about 7 threads.
- **Open for Odin:** `scripts/cypha_report_card.sh` still reads the old chain line as its headline. Switching it to `context8k_bpc` would change the card's numbers, so it was left as is.

**From 1.6979 to 1.5051 in four days**, with answers unchanged:
- the online chain;
- the online final layer;
- the LSTM;
- the online n-gram;
- the whole-distribution mixing with SSE.

### 13.31 A blind LLM reading of K5's sentence gate (registered 2026-11-04, before the ratings)

**Odin:** "That's too much reading. Just use GPT2 or you rate it."

- **Why not GPT-2:** its perplexity rewards predictable text. That is the flaw SCRIBE v3's SURPRISE found in the gated K5, so GPT-2 would be biased toward the gate.
- **The judges:** four independent Claude raters (subagents). Each gets only the texts in its prompt, no tools and no files, so it cannot see the key.
- **The items:** 50 GEN-A dev prompts: the 10 on Odin's rating page, plus 40 drawn with seed 20261105, leaving out the first sheet's 8. Each prompt has three continuations: the real text, the old K5 and the new (gated) K5. All are cut at a word boundary near 460 characters and shown in random order as A, B and C.
- **Who rates what:** raters 1 and 2 rate prompts 1–25, and raters 3 and 4 rate prompts 26–50. Each rater sees its own order and its own A/B/C labels. Each text gets 1–10 for how well it reads as coherent, on-topic prose.
- **Rule:** the gate's gain counts as confirmed if new K5 minus old K5, averaged over the two raters per prompt and then over the 50 prompts, is above 0, with a 95% bootstrap interval over prompts that excludes 0. Both rater pairs must agree on the sign.
- **Predictions:**
  - human text about 8;
  - both K5s about 2–4;
  - new minus old about +0.5, with the rule holding (about 60%);
  - raters agreeing at r ≥ 0.6 per text.
- Items and key: `bench/blind/llm_rating/`, key base64-encoded.

**§13.31 result** (2026-11-04; 4 blind raters, 150 texts, 2 ratings each):

| | Mean (1–10) | Rated 6 or more |
|---|---:|---:|
| human | **8.50** | 99% |
| new K5 (gate) | 2.09 | 0% |
| old K5 | 1.77 | 0% |

- **The rule holds:** new minus old is +0.32, with 95% interval [0.22, 0.42] over 50 prompts. Both rater pairs are positive (+0.28, +0.36). The gate makes K5 slightly better to a reader.
- **But both K5s read as about 2 out of 10:** "mostly sentence-shaped, but they don't connect or make sense". No K5 text got 6 or more from any rater. The human text came top in 100 of 100 items. Raters agreed closely (r 0.986 per text).
- **Predictions:** human about 8 hit (8.5); K5 2–4 hit at the bottom (1.8–2.1); +0.5 and the rule holding hit in direction, smaller than predicted; agreement hit.
- **What it means:** SCRIBE v2's 0.50 for K5 (human 0.92) overstates how readable it is. To a reader, K5 is still far from prose. Small tuning will not close that gap. It needs a different kind of model (Odin, 2026-11-04: "more math like K5 … but completely different maths").

### 13.32 Meta mixing, and four kinds of maths beside K5 (2026-11-04; work moved to Odin's PC)

**Odin:** "We have a mixer and we could possibly do meta mixing. Do a google search and see what you come up with and do that too." And on the four writers: "Do these as well. I want to see that."

**What the search found for meta mixing:**
- PAQ stacks mixers. Several first-layer mixers each choose their weight set by a different small context, and one more mixer joins them before SSE ([PAQ](https://en.wikipedia.org/wiki/PAQ), [context mixing](https://en.wikipedia.org/wiki/Context_mixing)). PAQ8's mixer maps onto a mixture of experts with a fixed gate ([Knoll and de Freitas](https://arxiv.org/pdf/1108.3298)).
- MetaGrad and Squint run the same learner at many learning rates, and a master weights them by their performance, so no step needs tuning ([MetaGrad, JMLR 2021](https://www.jmlr.org/beta/papers/v22/20-1444.html)).
- Fixed share (switching) follows whichever expert is best right now, which suits text whose topic changes page by page.
- For writing, LLM-Blender and InfoSel rank candidates from several generators and keep the best ([InfoSel](https://arxiv.org/html/2407.12841v1)).

**Three sessions on Odin's PC, each registering rules before its runs:**
1. **Meta mixing** in `context8k`, against today's default 1.5051:
   - a PAQ-style second layer over several context-selected mixers;
   - a mixture over learning rates (MetaGrad style);
   - fixed-share switching over the full mix's arms.
   Branch `wip/meta-mix`.
2. **K5's helpers, by sentence choice:**
   - minimum Bayes risk over many drafts (decision theory);
   - an entity grid (a Markov chain over each sentence's subjects and objects);
   - an LDA topic model (Bayesian), steering K5's words and scoring its sentences;
   - minimum Bayes risk over the pooled drafts of all K5 variants, as meta mixing for writing.
   Branch `wip/k5-discourse`.
3. **Whole-paragraph MCMC:** Metropolis–Hastings over a paragraph by small word edits. Its energy is K5's log probability plus the topic and entity terms from session 2. Branch `wip/k5-mcmc`.

**Judges for writing:**
- SCRIBE v2 and v3.1 on GEN-A dev;
- then four blind Claude raters, as in §13.31, on 50 dev prompts. They are the judge of another kind that §13.31 showed is needed.

The test split is read once per adopted change.

### 13.33 One expert per kind of structure, meta-mixed (2026-11-04; Odin's PC)

**Odin:** "K5 does sentence structure basically. Why don't we work on an algorithm/mixer/expert for each type of structure or relation and metamix them all together. Let's do this."

**The idea:**
- K5 is the syntax expert.
- Every other kind of structure gets its own expert.
- Each expert scores a candidate sentence given the paragraph so far.
- A learned meta-mixer joins the scores and picks the sentence. It does not blend next-word probabilities: §13.15–13.24 showed that blending makes text bland.

**The experts (session 4, branch `wip/k5-structure`):**

| Expert | Structure | Maths |
|---|---|---|
| K5 | word order, syntax | Kneser–Ney 5-gram |
| class sequence | grammar skeleton | exchange-class n-gram |
| topic | what the paragraph is about | LDA (session 2) |
| entity grid | who each sentence is about | Markov chain over roles (session 2) |
| relations | who did what to whom, typed slots (person, place, year) | counts of typed relation patterns |
| reference | pronouns that match their antecedents | name–pronoun statistics |
| time and number | years and numbers consistent with the prompt | simple distributions |
| transitions | how sentences open and link | Markov chain over sentence-opening types |
| rhythm | sentence length, balanced brackets and quotes | ECDFs |
| novelty | no copying of training text, no echo of the prompt | suffix-array match lengths |

**The meta-mixer:**
- An online softmax ranker over the experts' scores, learned by exponentiated gradient, like the content mixer.
- Its weight sets are chosen by a small context (sentence position, prompt length), as in PAQ. A second layer and a mixture over learning rates come from §13.32.
- **Training signal:** clean Wikipedia that K5 never saw (the enwik9 text of K5_SCALE, overlap with GEN-A and SCRIBE already removed). At each sentence boundary, the real next sentence competes with N K5 drafts.
- **Its own metric:** held-out "sentence bits", minus log₂ of the real sentence's probability among the candidates. This is measured for each expert alone and for the mix.

**Judges for writing:**
- SCRIBE v2 and v3.1 on GEN-A dev;
- the four blind Claude raters (§13.31) as the deciding judge;
- one test read only for an adopted arm.

The MCMC sampler (session 3) can use the learned mix as its energy.

### 13.34 Blind reading of the new writers (registered 2026-11-05, before the ratings)

- **Arms**, the seed-0 dev GEN files from `wip/k5-discourse` and `wip/k5-mcmc`:
  - B0, K5's default (= §13.31's new K5);
  - T2, T2b and T2c: LDA topic rescaling at B 1, 1.5 and 2;
  - M3: gate 16 with consensus (MBR);
  - E1b: paragraph MCMC with the entity grid, final temperature 0.8;
  - the human continuation.
- **Raters:** the same 50 dev prompts as §13.31. Four new blind Claude raters, two per prompt, with no tools and no key. Seven texts per prompt in random order, each cut near 460 characters, each rated 1–10.
- **Rule:** an arm reads better than B0 if its mean paired difference from B0 over the 50 prompts (two raters averaged) is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Predictions:**
  - B0 near §13.31's 2.1;
  - T2 +0.3 to +0.8;
  - T2b about the same as T2;
  - T2c below T2b, because stuffing costs readability;
  - M3 about +0.2;
  - E1b about 0;
  - human about 8.5.

**§13.34 result** (2026-11-05; 4 blind raters, 50 prompts × 7 texts, 2 ratings each):

| Arm | Mean (1–10) | Against B0 [95%] | Rater pairs |
|---|---:|---:|---|
| human | **9.04** | | |
| M3, MBR (gate 16 + consensus) | 2.22 | +0.04 [−0.01, +0.10] | +0.08, 0.00 |
| B0, K5 default | 2.18 | — | |
| T2, topic B 1 | 2.18 | 0.00 [−0.06, +0.05] | 0.00, 0.00 |
| E1b, MCMC + entity grid | 2.13 | −0.05 [−0.10, 0.00] | −0.10, 0.00 |
| T2b, topic B 1.5 | 1.96 | **−0.22** [−0.32, −0.12] | −0.22, −0.22 |
| T2c, topic B 2 | 1.76 | **−0.42** [−0.53, −0.32] | −0.34, −0.50 |

- **No arm reads better than B0. The rule fails for all five.**
- **Topic rescaling reads worse the harder it pushes,** while SCRIBE v2 and v3.1 both rose (v3.1: 0.483 → 0.670 → 0.712). This is the keyword-stuffing blind spot of SCRIBE_V31.md, confirmed by readers: the topic model stuffs topic words, and both SCRIBE versions reward them.
- B0 reproduces §13.31 (2.18 against 2.09). Human text scores 9.04.
- **Predictions:**
  - B0 and human hit.
  - T2 missed: 0, not +0.3 to +0.8.
  - T2c below T2b hit.
  - M3 (about +0.2) and E1b (about 0) roughly hit.
- **What it means:** none of these helpers fix what readers see, sentences that don't connect. Two lines remain: the learned structure meta-mixer (§13.33), whose training signal, the real next sentence, cannot be gamed by stuffing; and a SCRIBE that catches topic stuffing.

### 13.35 Result: meta mixing is on by default (2026-11-05; [reports/META_MIXING.md](reports/META_MIXING.md))

- **What won: M1Gs.**
  - **M1:** PAQ's second layer over 14 first-layer weight sets. Each set is chosen by its own context: A's four; order-1, order-2 and order-3 bytes; match length × copy; InfiniGram bucket × partial byte; byte classes; similar-context and Kneser–Ney buckets; a word hash.
  - **The rest:** it runs at 5 learning rates under a fixed-share master, then SSE, then a per-bit mix with the geometric mixer from the rate mixture.
  - Tuned on the holdout's first half only; registered (f846a4d) before any second-half read.
- **Result:** `context8k_bpc` **1.5051 → 1.4962** (−0.0089; the rule needed 0.005), on Odin's PC.
  - Every line before the new ones is unchanged, TALLY and GEN included.
  - `--no-meta-mix` equals main byte for byte.
  - The first and second halves agree (−0.0099, −0.0089).
- **The other arms:**
  - **Mixing over learning rates alone (M2):** small, −0.001 to −0.004. The earlier tuned rates were already near the best.
  - **Fixed share over the arms (M3):** 1.4971. It tracks the best arm and ends slightly behind it.
  - **Gain:** almost all of it comes from the wider first layer.
- **Cost:** +5.9% wall time (3085 s against 2914 s) and +405 MB (1555 MB against 1150 MB), mostly M1's five rate copies. `m1r=0` keeps one copy for about 0.001 less gain.
- **Running total:** 1.6979 on 1 November → **1.4962** now (−0.20 bits per byte), answers unchanged.

### 13.36 Blind reading of the structure meta-mixer's writing (registered 2026-11-05, before the ratings)

- **Arms**, the seed-0 dev GEN files from `wip/k5-structure`:
  - W0, today's K5;
  - W1, the learned mix's best pick among 8 K5 drafts;
  - W3, the mix trained on drafts only;
  - W5, the mix without the k5 expert;
  - the human continuation.
- **Raters:** the same 50 dev prompts as §13.31 and §13.34. Four new blind Claude raters, two per prompt, with no tools and no key. Five texts per prompt in random order, cut near 460 characters.
- **Rule:** as §13.34. An arm reads better than W0 if its mean paired difference over the 50 prompts is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Predictions:**
  - W0 near 2.2;
  - W1 below W0: the mix leans on "improbable under K5", which in writing means the strangest draft;
  - W3 about W0;
  - W5 about W0, within ±0.2;
  - human about 9.

**§13.36 result** (2026-11-05; 4 blind raters, 50 prompts × 5 texts):
- Human text **8.41**.
- W0 (today's K5), W1, W3 and W5 (the structure mixer's writing): **2.00 each, from every rater on every text**. Every difference from W0 is 0.000 [0.000, 0.000].
- **No arm reads better than K5. Nothing is adopted.**
- **A floor effect:**
  - All K5 variants sit at "mostly sentence-shaped, but they don't connect", so this 1–10 scale cannot separate them any more.
  - Telling them apart would need a pairwise forced choice (which of two reads better).
  - The point stands without it: no helper moves K5 off the floor.
- **The structure mixer itself** (K5_STRUCTURE.md):
  - It picks the real next sentence among 8 K5 drafts with 0.113 sentence bits (98%), against 0.415 for K5 alone and 2.72 for uniform weights. More data helped: 0.166 bits with statistics from 10 MB, 0.113 with 210 MB.
  - But it wins mostly because real sentences are *less* probable under K5 than K5's own drafts. In writing, every candidate is a K5 draft, so that cue picks the oddest one.
  - Choosing the draft nearest the real sentence finds almost nothing: 2.992 bits against 3.000 by chance. **K5's drafts hold no good sentence to choose.**
- **Predictions:** W0 near 2.2 hit (2.00); W1 below W0 missed (equal, at the floor); W3 and W5 about W0 hit; human about 9 hit (8.4).

**What the four days of writing work show** (§13.31–13.36):
- **Choosing among K5's sentences cannot fix K5.** Its candidates are all sentence-shaped text that doesn't connect, so the gate, MBR, the entity grid, topic steering, MCMC and the learned structure mixer all land at about 2 out of 10.
- **The next step has to change what writes the sentences,** not what picks among them.

**Code kept, off by default:**
- `lda.hpp`, `entity_grid.hpp`, `mbr.hpp` (`wip/k5-discourse`);
- `k5_mcmc.hpp` (`wip/k5-mcmc`);
- `lm/structure/` (`wip/k5-structure`, `sample --k5-struct`).

K5's default output is byte-identical after the merges: all 250 dev prompts, checked in the cloud.
