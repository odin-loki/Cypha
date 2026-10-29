# Cypha language plan

**Odin Loch** · 2026-10-26 · **the default plan**

This is the forward plan for giving Cypha real language ability. The 1 TB packed-trie run in [`CURRENT.md`](CURRENT.md) carries on as **Track A**, an experiment inside this plan. The measurements that led here are in [`reports/CYPHALM_QA_AND_CONTEXT_TEST.md`](reports/CYPHALM_QA_AND_CONTEXT_TEST.md). Appendix A maps every point raised in the discussion to the section that handles it.

---

## Contents

0. [What the measurements say](#0-what-the-measurements-say)
1. [Goal, decisions, capability ladder](#1-goal-decisions-capability-ladder)
2. [Rules of the work](#2-rules-of-the-work)
3. [Architecture: two machines, two hierarchies, one learning loop](#3-architecture)
4. [The data pack](#4-the-data-pack)
5. [Tracks](#5-tracks)
6. [SCRIBE: the generated-text metric](#6-scribe-the-generated-text-metric)
7. [ECS: Evidence Certainty Score](#7-ecs-evidence-certainty-score)
8. [LCI: Learning Certainty Index](#8-lci-learning-certainty-index)
9. [Frozen evaluation suites](#9-frozen-evaluation-suites)
10. [Profiling and budgets](#10-profiling-and-budgets)
11. [Risks and kill criteria](#11-risks-and-kill-criteria)
12. [Order of work](#12-order-of-work)
- [Appendix A: discussion ledger](#appendix-a-discussion-ledger)
- [Appendix B: options not taken](#appendix-b-options-not-taken)
- [Appendix C: references](#appendix-c-references)

---

## 0. What the measurements say

All of these were measured on 2026-10-26. The protocol is enwik9's first 8 MiB, with `[0, 7340032)` for training and the rest held out, plus Dolly 15k. The full numbers and commands are in the [report](reports/CYPHALM_QA_AND_CONTEXT_TEST.md).

| Finding | Number |
|---|---|
| The three questions `trie_chat` "answered" are Dolly's first, middle and last training turns | Answers match the stored text byte for byte |
| That answer comes from `copied_answer()`, a `std::string::find` of the whole prompt in the training bytes | Fires on 975/975 verbatim prompts, 0/975 with one letter lowercased |
| The order-4 trie mixture cannot see the question. Every prompt ends in `\nAssistant: `, so it sees only `nt: ` | The same text after all 14 prompts tried |
| An order-8192 count model (`InfiniGram`, capped at 8192) on top of the trie | BPC 2.32 → **2.06** with 16 bytes of context. 64, 512 and 8192 bytes add nothing (2.07) |
| How much of an 8K window matches training text | Median **10 bytes**. 1 position in 10⁶ matched more than 1024 |
| Questions with the 8K model | Seen 3/3, one letter changed 3/3, " Thanks." added 0/3, paraphrase 0/3, held-out Dolly 1/30 (that one was "yes"), Wikipedia facts 0/8 |
| Surface score `S` against controls (`surface_controls`) | Random corpus words **0.918** and a real passage with its words shuffled **0.928**, against the real passage at 0.933. The trie's sample scores 0.775. Wrong 8K answers 0.82–0.89, correct copied answers 0.77 |
| Core Cypha's own text features on 20 Newsgroups (2k) | Hashing 16–19%, TF-IDF/SVD 34% (run with `W_enc` learning), MiniLM frozen 62.5% |

Also from the repo's history (survey of `docs/history`, `docs/archive`, `bench/config`):

- **Meaning never came from Cypha itself.** Every in-house text representation was hashed n-grams or surface statistics. The only good text number used meaning borrowed from MiniLM.
- **What was tried and killed by measurement:**
  - Hebbian encoder updates: lost on 4 of 4 tasks.
  - Attention memory (`compressive_memory`, `context_bank`): +0.2 to +0.27 BPC, "KILL".
  - BPE: 4.15 vs 2.87 BPC.
  - A word-sized trie expert: weight 0.0005.
  - Learning `W_enc` on semantic inputs: 49.3% vs 62.5% frozen.
- **What worked:**
  - The Cypha classifier, given good vectors: it beat logistic regression, 62.5% vs 60.3%.
  - Its epistemic out-of-distribution (OOD) signal: 0.17 in domain vs 1.08 out, p ≈ 3×10⁻¹⁰².
  - CyphaLM as a compressor: it beats 3.4M-parameter neural LMs on held-out BPC in a one-hour CPU budget ([`reports/CYPHALM_VS_NEURAL_LM.md`](reports/CYPHALM_VS_NEURAL_LM.md)).

**Conclusion.** CyphaLM is a strong compressor and copier. It has no route from a question to an answer it did not store, and neither a longer byte window nor more data gives it one. Meaning has to come from structure: words, senses, concepts, facts and patterns. Cypha's counting and classifying machinery can learn and use that structure.

---

## 1. Goal, decisions, capability ladder

**Goal.** Good general NLP and language-model functionality from Cypha, with:
- no neural network at runtime;
- low RAM;
- online, dynamic learning;
- selective learning (it chooses what to keep).

**Decisions already made (Odin Loch, 2026-10-26)**

| Decision | Consequence |
|---|---|
| No NN at runtime | NNs appear only as a distillation source, offline and verified (Track H), and as an external yardstick (existing MiniLM and Transformer numbers) |
| Hybrid symbolic + statistical | A data pack gives structure. Counts give coverage and probabilities. The Cypha classifier makes the decisions |
| A data pack instead of learning everything from mass data | Section 4. Sample efficiency with and without the pack is measured (section 8.6) |
| The classifier does RL | Section 3.4 and Track G |
| Learning is dynamic, with recursive self-improvement | Section 3.4. Every self-made change passes the LCI referee (section 8) |
| The 1 TB trie run continues as science | Track A, with predictions registered in advance |
| Everything is profiled | Section 10 and Track P |
| Evaluation contract | Held-out answers + surface score `S` + BPC + SCRIBE + ECS + LCI, reported together. BPC alone is not a generation score |

**Capability ladder.** Each rung has a test. Claims move up a rung only when that test passes.

| Rung | Ability | Test (section 9) | Status |
|---|---|---|---|
| L0 | Return a stored answer to a byte-identical stored question | T-perturb, verbatim rows | Have (exact match only) |
| L1 | Return a stored answer to a reworded stored question, or abstain | T-perturb + T-QA-dolly | Track B7 |
| L2 | Answer a factoid question from facts it holds ("capital of Algeria?") | T-QA-wiki, T-QA-wikidata | Tracks D, E, F |
| L3 | Classify, route and tag text (topic, part of speech, entities, question type) | T-topic, T-syntax | Tracks C, E |
| L4 | Answer from text it has read but not stored as a fact. Compose multi-fact answers | T-QA-squad, T-QA-wikidata held-out facts | Tracks F, G |
| L5 | Open conversation and instruction following | T-dialogue (to be built) | Stretch. Measured, not assumed |

---

## 2. Rules of the work

1. **No claim without a held-out test.** Every claim gets three probes: the verbatim question, a one-letter change, and a paraphrase. This applies to output from any AI assistant (Grok, Claude or anything else) as much as to Cypha. The 2026-10-26 copy result came from a plausible claim that nobody checked.
2. **Every number has a tool and a report.** A tool in `native/tools/` with its paths taken as arguments, and a dated report in `docs/reports/`.
3. **Predict before a big run.** Write the predicted range before the run. Score the prediction afterwards, hit or miss.
4. **Frozen test sets are never trained on.** They live in `bench/eval/` with checksums. Training readers skip `/eval/`, just as `trie_stream_train` skips `/labelled/`.
5. **Profile everything** (section 10). A gain that breaks a RAM or latency budget is not a gain.
6. **Kill criteria are written first** (section 11). A track that misses its gate twice stops.
7. **Name things honestly.** Retrieval is called retrieval. The README already does this for the chat copy. Keep it that way in anything sent to outside reviewers.
8. **Track licences.** Every data source has a licence line in the manifest, and share-alike terms are checked before anything ships under the commercial licence (section 4.3).
9. **Not answering is a feature.** When the evidence is not there, Cypha says it does not know. It never fills the gap with a fluent guess.
   - **This is not a refusal.** A refusal declines a question it could answer. Cypha is saying it doesn't know.
   - **The output has two parts:**
     - a plain message, e.g. "I don't know. I have nothing about Hamlet in what I've read.";
     - a machine tag, e.g. `<|unknown reason=NO_EVIDENCE gap=G-0412|>`.
   - **The tag is for later.** The system reads it, logs the gap in the gap map (demand +1) and queues it for discovery ([DISCOVERY §3.5](DISCOVERY.md#35-the-dont-know-message-and-tag)).
   - **CyphaLM is trained to produce both itself**, the way commercial LLMs are trained on their refusal messages. Decided by Odin Loch, 2026-10-27.
   - A wrong answer costs more than an abstention.
   - Every Q/A result is reported as right / abstained / wrong, and the wrong count is the number to drive down first.
   - A change that raises coverage by adding wrong answers fails LCI on the wrong-answer rate, even if accuracy rises.
   - Decided by Odin Loch, 2026-10-27.
10. **CyphaLM does the task.** Every task, question answering included, is done by CyphaLM.
    - A string search, retrieval index or hand-written lookup that answers in CyphaLM's place is cheating. Its output is not reported as a Cypha result.
    - Other components can measure, or prepare data for CyphaLM. They do not answer for it.
    - A Cypha-classifier gate that decides whether to abstain on CyphaLM's own answer counts as CyphaLM doing the task **for now** (Odin Loch, 2026-10-27). Later, Cypha investigates this itself through discovery mode ([DISCOVERY §8a](DISCOVERY.md#8a-standing-investigation-who-decides-to-abstain)).
    - Removed on 2026-10-27: `copied_answer()` in `trie_chat.cpp`, `qa_infobox.cpp` and `chat_index.cpp` ([record](reports/REMOVED_NON_CYPHALM_QA.md)).
    - Decided by Odin Loch, 2026-10-27.

---

## 3. Architecture

### 3.1 Two machines and a knowledge store

```text
                   text in                                  question in
                      │                                          │
   ┌──────────────────▼──────────────────┐     ┌─────────────────▼────────────────┐
   │ CyphaLM: the counting machine       │     │ Core Cypha: the deciding machine  │
   │ packed trie, InfiniGram, hp gate24  │     │ DIF classifier, online, OOD flag  │
   │ counts → probabilities → features   │◄───►│ tags, parses, links, picks,       │
   │ compression, copying, fluency       │     │ abstains, learns from reward      │
   └──────────────────┬──────────────────┘     └─────────────────┬────────────────┘
                      │                                          │
                      └──────────────►┌──────────────┐◄──────────┘
                                      │ Knowledge     │  lexicon · concepts · facts ·
                                      │ store + ECS   │  patterns · rules (section 3.2)
                                      └──────────────┘
```

- **CyphaLM is a different machine from Core Cypha.** It counts, compresses and extracts probability distributions. In this plan it supplies probabilities, statistical features (co-occurrence, predictive distributions, compression distance) and fluent surface text.
- **Core Cypha makes the discrete decisions**, each with a calibrated confidence and an OOD flag: part-of-speech tag, parser action, entity link, word sense, question type, which answer, and whether to abstain. It is the policy that RL improves.
- **The knowledge store holds structure.** Every item carries its evidence and an Evidence Certainty Score (section 7).

### 3.2 Representation: two hierarchies

**A. Context hierarchy (the text side).** Each level's symbols are IDs from the level below. Each level has its own count model (packed trie or InfiniGram over IDs) and its own vectors (Track C).

| Level | Symbol | Reach of a 4–5 symbol context | Count model | Vector |
|---|---|---|---|---|
| byte | 0–255 | ~4–16 bytes (measured: gains stop at 16) | packed trie + InfiniGram (have) | — |
| word | lexicon ID | 4–6 words, ~25–35 bytes | word trie (B3) | PPMI-SVD, random indexing (C1) |
| class | Brown cluster or WordNet concept ID | generalises across synonyms | class trie (B4) | cluster path bits |
| phrase/chunk | chunk type + head | clause | chunk transitions (E2) | head-word vector |
| sentence | sentence ID | paragraph | entity grid (SCRIBE C2) | SIF vector (C6) |
| document/topic | topic ID | document | topic cache (B5) | topic distribution |

The byte window ran out at about 16 bytes because exact byte strings rarely repeat further than that. Counting over words and classes is how count models reach further: "capital of Algeria" and "capital of Albania" share the class-level context `capital of [COUNTRY]`.

**B. Knowledge hierarchy (the meaning side).** All items use 32-bit IDs and live in packed, memory-mapped tables.

1. **Lexicon.** Surface form → lemma → senses. Each sense has `P(sense | form)` from counts and `P(sense | form, context)` from the Cypha sense classifier. Sources: Wiktionary, Open English WordNet.
2. **Concepts.** An is-a and part-of hierarchy: WordNet hypernyms, Wikidata `instance of`/`subclass of`, and the OpenCyc upper ontology, aligned by a mapping table. Rules and defaults attach high in the hierarchy and are inherited, with exceptions allowed. That inheritance is how the system generalises without a network.
3. **Facts.** `(subject, relation, object, qualifiers)` plus an evidence list and an ECS. Example: `(Algeria, capital, Algiers)`, evidence from an infobox and Wikidata, ECS 0.99.
4. **Patterns.** Bidirectional links between language and relations: "X is the capital of Y", "Y's capital, X", "What is the capital of Y?" ↔ `capital(Y, X)`. Patterns are used to read (text → facts), to understand questions (question → query) and to generate (fact → sentence). Each pattern has a reliability, a Beta posterior over its precision.
5. **Rules.** Inference rules, each with its own reliability:
   - inverse relations: `capital(Y, X) ⇒ capital_of(X, Y)`;
   - transitivity: `part_of`;
   - composition: `born_in(X, C) ∧ city_in(C, K) ⇒ born_in_country(X, K)`;
   - type constraints: a capital is a city.

### 3.3 What the system does once the data model exists

The runtime loop is **read → understand → store → answer → learn**.

1. **Read.**
   1. Tokenise to lexicon IDs, with unknown forms flagged.
   2. Tag parts of speech (E1), then chunk and recognise entities (E2), then parse (E3).
   3. Link mentions to concept IDs (E5).
   4. Match patterns to propose facts.
   5. Score each proposal with ECS.
   6. Store it as admitted or as a candidate (section 7.6).
   7. Update counts at every level.
2. **Understand a question.**
   1. Classify the question type, expected answer type and relation (E4).
   2. Parse the question into a query, e.g. `capital(Algeria, ?)`.
   3. Link its entities.
3. **Answer.** These steps are carried out by CyphaLM (rule 10). How CyphaLM does each one is the next design step, agreed with Odin before it is built.
   1. Look up the fact. If it is missing, infer it through the hierarchy and rules.
   2. If that fails, retrieve passages: BM25 plus encoder similarity.
   3. From the passages, extract candidates of the expected type and score them by redundancy, ECS and similarity.
   4. Pick the best answer, or abstain when Cypha's confidence or OOD flag says so.
   5. Realise the text: relation templates and reversed patterns, with CyphaLM choosing the wording.
4. **Learn.** Every outcome (verified right, verified wrong, abstained, rated) updates:
   - the trust in each pattern, source, rule and sense that contributed;
   - the Cypha decision classifiers that acted.

### 3.4 Learning: dynamic, rewarded and refereed

**What is learned, how, and what checks it**

| Component | Learned by | Signal | Check |
|---|---|---|---|
| Counts (byte, word and class tries; co-occurrence) | Online counting, MDL pruning (drop counts below 2, quantise) | The text itself | Held-out BPC, LCI |
| Word and sentence vectors | PPMI-SVD refresh per checkpoint; random indexing online | Co-occurrence counts | T-words, T-topic |
| Decisions (tags, parser actions, links, senses, question types) | Cypha DIF online updates | Data pack labels first, then rewards | T-syntax, T-QA, calibration |
| Pattern reliability | Beta–Bernoulli updates | Verified hits and misses | ECS audit |
| Source trust | Beta–Bernoulli updates | Verified items per source | ECS audit |
| Fact certainty | Subjective-logic fusion (section 7) | All evidence | ECS calibration |
| Rules and procedures | The self-improvement loop | Proposed, then refereed | LCI ≥ 0.95 |

**Teaching it how to learn: four levels, introduced in order**

- **Level 0, fixed procedures.** We write how it reads, extracts, scores and updates. Nothing learns how to learn yet.
- **Level 1, learned weights.** It learns which patterns, sources, senses and facts to trust, and the classifier parameters.
- **Level 2, learned choices among procedures.** A bandit picks among learning strategies, such as which pattern miner, which admission threshold and which curriculum slice, rewarded by the LCI gain each produced.
- **Level 3, self-proposed procedures and rules.** It generates candidate rules, patterns, features and thresholds. A candidate is promoted only through LCI (section 8).

Each level starts only when the level below it is stable, meaning its LCI dashboards show no drift for two checkpoints.

**RL.** The Cypha classifier is the policy. Actions include which answer, which pattern, which sense, abstain, and explore a new pattern. Rewards:

| Reward | Source | Cost |
|---|---|---|
| R1 prediction | Next symbol at every level (self-supervised) | Free |
| R2 verified Q/A | Questions generated from held-out facts, with answers known | Free once the fact store exists |
| R3 consistency | Type constraints, functional relations, numeric sanity | Free |
| R4 compression gain | An admitted fact or pattern that shortens the description of held-out text (MDL) | Cheap |
| R5 human rating | Odin rates samples (SCRIBE axes) | Expensive; used sparingly and for calibration |

Start with a **contextual bandit** (one decision, immediate reward) using Thompson sampling over the Beta posteriors the system already keeps. Move to multi-step RL only when the bandit plateaus and LCI shows it is the bottleneck.

**Recursive self-improvement loop**

```text
propose  →  sandbox on dev set  →  LCI on the referee set  →  promote or reject  →  log  →  repeat
   ▲                                                                          │
   └────────────── rotate the referee set every N promotions ◄────────────────┘
```

- **Proposal generators:**
  - pattern miner (bootstrapping, section F2);
  - rule miner (relation composition with support and confidence);
  - threshold tuner;
  - feature proposer (new Cypha features from existing statistics);
  - curriculum selector.
- **Guards:**
  - an alpha budget per period (section 8.4);
  - canary non-inferiority;
  - versioned stores with one-command rollback;
  - weekly fact-precision audits with a Wilson lower bound ≥ 0.90, otherwise admission pauses.

**Selective learning ("picky").** An item is admitted only if it pays for itself:
- a count survives packing only if seen at least twice;
- a fact needs ECS ≥ 0.8;
- a pattern needs precision with a lower bound ≥ 0.7 over at least 20 checks;
- a training document must not be a near-duplicate (MinHash Jaccard below 0.8);
- a corpus slice must lower held-out code length per byte (MDL gain > 0).

**Forgetting.** Shared-model continual learning is an open problem here: D16B's forgetting score is ≈ 0.81, while isolated model files show zero forgetting (D16F). So the system keeps one model file per decision type (tagger, parser, linker, question classifier, sense classifier), versions the knowledge store, and runs canary sets on every promotion.

### 3.5 Speculative mode and discovery

**Designed in full in [`DISCOVERY.md`](DISCOVERY.md) (2026-10-27), as Track S.** That document covers:
- knowing what it knows (epistemic states, the competence self-model);
- knowing what it does not know (reason codes, the gap map, estimates of the unseen);
- 20 hypothesis generators;
- scoring by value of information;
- tests with pre-registration;
- 16 discovery processes, the curiosity scheduler, honesty invariants and benchmarks.

The principles in short — hypothetical and speculative thinking is a **separate mode**. It never mixes into answers:

- **What it does.** It proposes hypotheses: facts the rules or analogies suggest, connections between distant concepts, gaps in the store. Each comes with how it was derived (rule chain, analogy, pattern) and what would confirm or refute it.
- **How it is labelled.** Every output is marked as a hypothesis, with its own grade (`Hypothesis`, below Very low in section 7.3). It is stored apart from working knowledge.
- **What it cannot do.** A hypothesis is never used as a premise, never returned as an answer, and never admitted until independent evidence lifts it through ECS like anything else.
- **How it is judged.** The share of hypotheses later confirmed, per generator, is that generator's reliability (a Beta posterior). Generators that never get confirmed are retired.

Answer mode keeps rule 9 unchanged: no evidence, no answer.

---

## 4. The data pack

### 4.1 Sources

Every licence line says "verify at download". It records what we believe today, and the manifest stores what was actually downloaded.

| Source | Gives | Layer | Licence (verify) |
|---|---|---|---|
| **Wikidata** (truthy dump, filtered to entities in our corpora plus the most-linked items) | Facts, types, `instance of`/`subclass of` hierarchy, aliases | Facts, concepts, lexicon aliases | CC0 |
| **Wikipedia infoboxes** (from the dumps we already fetch) | Facts aligned with article text. The 8 MiB slice already contains `capital = [[Algiers]]`, `capital = [[Kabul]]`, `capital =[[Canberra]]` | Facts, pattern seeds | CC BY-SA |
| **Open English WordNet** | Senses, synonym sets (the thesaurus), hypernyms, glosses | Lexicon, concepts | CC BY 4.0 |
| **Wiktionary** via Wiktextract (kaikki.org JSON) | The dictionary: parts of speech, inflections, definitions, synonyms | Lexicon | CC BY-SA |
| **ConceptNet 5** | Common-sense relations (UsedFor, PartOf, CapableOf…) | Facts, concepts | CC BY-SA 4.0 |
| **OpenCyc 4.0** (2012, last release; site shut 2017; SourceForge/OSDN mirrors) | Upper ontology: about 239k concepts and 2M mostly taxonomic facts | Concepts (top levels) | Apache 2.0 |
| **Universal Dependencies English EWT** | Part-of-speech tags and dependency trees | Training labels for E1 and E3 | CC BY-SA 4.0 |
| **Text statistics** from Track A | Word frequencies, n-grams, co-occurrence, Zipf curves | Counts, vectors | Ours |
| **Dolly 15k** | Instruction Q/A pairs | Chat index, question classifier | CC BY-SA 3.0 |
| **OpenAssistant (oasst1)** | Dialogue Q/A pairs, the second Q/A set | Chat index, question classifier | Apache 2.0 |
| **SQuAD v1.1** | Questions with Wikipedia paragraphs | Q/A training and evaluation | CC BY-SA 4.0 |
| **Distilled** (Track H) | Paraphrases, definitions, facts from an open-weight model under a permissive licence | All layers, with a low ECS prior | Per the model's licence |
| Evaluation only: SimLex-999, WordSim-353, MEN, 20 Newsgroups, AG News | Benchmarks | — | Research use; check terms |

**Correction for the record.** The "8 MB of Wikidata" used for the trie was the first 8 MiB of **enwik9**, which is Wikipedia XML text. Wikidata is the separate structured fact base listed above. The chat model (`cyphalm_trie chat`, formerly `trie_chat.cpp`) loads **Dolly only**. OpenAssistant goes in with B7.

### 4.2 Build and format

- `scripts/fetch_corpus.py` gains a `pack` category. Each source records its URL, version, date, licence and SHA-256 in `bench/pack/pack_manifest.json`.
- The builders write packed binary tables (`lexicon.bin`, `concepts.bin`, `facts.bin`, `patterns.bin`) with 32-bit IDs and sorted keys, memory-mapped read-only. The format is versioned like the InfiniGram `IGR2` index.
- The store is append-only: an operation log plus snapshots. Rollback means switching snapshot.

### 4.3 Licence note

CC0, CC BY and Apache are simple for commercial use. CC BY-SA (Wiktionary, ConceptNet, EWT, SQuAD, Dolly) carries share-alike: a redistributed derived data pack must keep the same licence. Before anything ships under [`COMMERCIAL-LICENCE.md`](../COMMERCIAL-LICENCE.md):
- keep share-alike-derived tables separable;
- or ship the builders and let the user build the pack;
- or rebuild those layers from CC0, CC BY or Apache sources only.

### 4.4 Hypothesis H-FIXED: "language is a fixed construct"

This is Odin's hypothesis, and it is tested rather than assumed. On held-out text, as the corpus grows, measure:

- **Pack coverage:**
  - % of tokens with a lexicon entry;
  - % of content words whose sense the classifier assigns with confidence ≥ 0.8;
  - % of entity mentions linked to a concept;
  - % of sentences in which a known pattern fires.
- **Growth:**
  - new word types per million tokens (Heaps' law `V = K·Nᵝ`);
  - new senses, entities and patterns per million tokens.

If coverage stays at least 99% and growth flattens, the hypothesis holds for this domain and the pack carries most of the load. If not, the numbers say exactly how much learning has to supply. The test is reported per domain: Wikipedia, books, web, chat.

---

## 5. Tracks

Each track lists its work items, how they are measured, and the gate to pass. Budgets are in section 10.

### Track A: the 1 TB packed trie (running; science)

This is the plan in [`CURRENT.md`](CURRENT.md): hold out first, shard at about 1 GB with `<|doc|>` breaks, pack and free, merge, store the next byte in the match record past 4 GiB, keep labelled chat separate.

**Report at each checkpoint (8 MiB, 64 MiB, 1 GB, 10 GB, 100 GB, 1 TB):**
- BPC on the enwik 8 MiB holdout and on each category's frozen holdout;
- surface `S`, SCRIBE (S, I and E axes) and held-out answers (T-perturb, T-QA-wiki);
- distinct n-gram counts;
- peak and packed RAM, and wall time.

**Experiments on the model:**

| ID | Experiment | Question it answers |
|---|---|---|
| A1 | Scaling curve: BPC against bytes, with a fitted power law `L(n) = L∞ + a·n^(−b)` | What more data is worth, and the floor `L∞` |
| A2 | Pruning thresholds: drop counts below 2, 3 and 5 | Size against BPC |
| A3 | Mixer weights frozen at the 8 MiB fit, or refit per checkpoint | Does the router need to relearn at scale? |
| A4 | Domain mix: wiki only, wiki + books, everything | Does web and code text help or hurt wiki BPC? |
| A5 | The 8K InfiniGram expert at scale (suffix array per shard) | Does the measured −0.26 BPC hold at 1 TB? |
| A6 | **Bit-count / parameter scaling** ("increase the bit count"): hp gate24 table bits 20→26 against packed-trie bytes, as BPC against model bytes | The size–quality curve, compared like for like |
| A7 | Orders 5 and 6 in packed form | Do higher orders pay their RAM at scale? |
| A8 | Per-domain experts with a learned router | Do specialists beat one big table? |

**Predictions registered 2026-10-26** (to be scored when the run finishes):

| ID | Prediction | Reasoning |
|---|---|---|
| P-A1 | Trie-only BPC on the enwik 8 MiB holdout at 1 TB: **2.00–2.25** | An order-4 model is limited by what 4 bytes predict. Counts saturate early |
| P-A2 | Trie + 8K InfiniGram at 1 TB: **1.85–2.05** | The measured gain came from matches of 8–16 bytes. More text gives more matches, with diminishing returns |
| P-A3 | Distinct 4-grams at 1 TB: **50–200 million** | 3.22M at 10⁹ bytes; the N^0.47 trend gives about 83M; mixed domains push it up. If true, packed size will exceed the 45 MB estimate made for enwik9 |
| P-A4 | Held-out answers to unseen questions: **≤ 2%** at every checkpoint | No mechanism links a question to an unstored answer |
| P-A5 | Surface `S` of samples: unchanged within **±0.05** | `S` measures word validity and repetition, which saturate early |

### Track B: CyphaLM model upgrades

| ID | Upgrade | Measure | Gate |
|---|---|---|---|
| B1 | **Consolidate the trie.** One `native/include/cypha/cyphalm/packed_trie.hpp` + `src/` replaces the 7 forked `trie_*.cpp` tools (1,100–1,600 lines each, 265–730 lines apart) and the 3 test tools. CMake targets, paths as arguments, CTests (self-test, tiny-corpus BPC golden, Q/A probe golden, 8K golden) | Same numbers as the forks to 1e-6 | Exact parity |
| B2 | **8K count-context expert in the mixture.** InfiniGram over training bytes, λ per (match length × count) bucket, fit on a dev slice | Held-out BPC (measured 2.32 → 2.06 at 8 MiB) | LCI ≥ 0.95 at 64 MiB and 1 GB |
| B3 | **Word-level count model.** Trie over lexicon IDs (orders 1–5), mixed into byte prediction through a spelling model `P(bytes | word)`. The old word trie fell to weight 0.0005 because it was an extra byte expert. This one predicts words and conditions bytes on them | BPC, word-level perplexity, T-QA | LCI ≥ 0.95 or kill |
| B4 | **Class-level counts.** Brown clusters (hierarchical, built from counts by mutual information) and WordNet/concept classes. Counts over class sequences generalise across synonyms. This is the statistics-to-semantics bridge | BPC on rare-word contexts, T-words | Beats B3 on rare contexts |
| B5 | **Long-range context without a network.** Document cache and trigger pairs (Kuhn & De Mori; Rosenfeld) plus a topic mixture from C-vectors | BPC on long documents, needle-in-haystack (`cyphalm_needle_haystack`) | LCI ≥ 0.95 |
| B6 | **Wire the trie into serving.** `generate_decode`, `cyphalm_generate` and REST `/generate` call hp today | Latency, RAM, SCRIBE | Within budget |
| B7 | **Fuzzy question handling in CyphaLM.** Reworded and unseen questions must reach the right answer, or an honest abstention, through CyphaLM itself. The BM25 chat index built on 2026-10-27 answered without CyphaLM and was removed under rule 10 ([record](reports/REMOVED_NON_CYPHALM_QA.md)) | T-perturb, T-QA-dolly (held-out), risk–coverage curve | One-letter and "Thanks" variants ≥ 95% of verbatim accuracy, by CyphaLM |
| B8 | **Match record stores the next byte** (corpus past 4 GiB), from CURRENT.md | Parity | Exact |
| B9 | **Copy-from-prompt expert.** A match model over the prompt itself, for closed questions whose answer is in the given context (CyphaLM's reading mode already copies a passage's second occurrence at 0.048 BPC) | Closed-QA exact match on SQuAD-style items | LCI ≥ 0.95 |
| B10 | **Constrained answer decoding.** Stop at the end of an answer, constrain to the expected type, keep the repetition guard (`mark_banned`) | SCRIBE I and R axes | No loss in exact match |
| B11 | **hp gate24 stays as the comparison**, with the A6 bit-scaling curve | — | — |

### Track C: semantics from statistics (the encoders)

**Design: [`SEMANTICS.md`](SEMANTICS.md)** (2026-10-27). Core Cypha becomes the meaning layer: words and concepts as classes, contexts as observations. Its Gaussians, natural-gradient and contrastive updates, MDL shrinkage and uncertainty learn embeddings, directional is-a relations, relation maps with abstention, and senses. These feed CyphaLM as a class-context expert and a similar-context expert. Experiments S-1 to S-6 and their gates are there; the rows below stay as baselines.

**Geometry teacher: [`EMBEDDING_GEOMETRY.md`](EMBEDDING_GEOMETRY.md)** (2026-10-27). Make Cypha's vectors behave like Qwen3-Embedding-8B's: dissect it (weights, outputs, the weight change from its base model), copy its pull–push forces into Cypha's contrastive update, distil its similarity structure pair by pair, and mine pairs and hard negatives with Qwen3.5-9B. Work items QE0–QE9.

| ID | Work | Measure | Gate |
|---|---|---|---|
| C0 | **Fair baselines first.** TF-IDF with `W_enc` frozen (the 34% ran with it learning); plain TF-IDF + logistic regression; the native hashing router (`branch_a_router.hpp`, 16–19%). MiniLM 62.5% is the external yardstick | T-topic (20NG at 2k, same protocol) | Recorded |
| C1 | **Distributional word vectors.** Count word × context-word co-occurrence within ±5 words (distance-weighted) in packed-trie format, dropping singletons. Weight with PPMI, then a randomised truncated SVD to 300 dimensions. Online variant: random indexing (fixed memory, incremental) | T-words (Spearman ρ), T-topic | Clearly above hashing and 34% TF-IDF. Target ≥ 55% on 20NG at 2k |
| C2 | **Predictive-distribution vectors (CyphaLM's own machine).** Represent a word or context by the distributions the count models predict before and after it, compressed by random projection or SVD; similarity by Jensen–Shannon divergence | Same as C1 | Kept only if it beats or complements C1 |
| C3 | **Compression semantics.** Normalised compression distance with hp or the trie as the compressor: `NCD(x, y) = (C(xy) − min(C(x), C(y))) / max(C(x), C(y))`, then kNN classification. Compare with a gzip-kNN baseline that we rerun ourselves (published gzip figures were later found inflated) | T-topic, retrieval recall@k | Kept only if competitive with C1 |
| C4 | **Retrofitting to the pack.** Pull vectors of WordNet synonyms and hypernyms together (Faruqui et al. 2015): symbolic structure improving statistical vectors | T-words | Gain over C1 |
| C5 | **Odin's statistical encoder, tried as asked.** S-1 words-as-classes run 2026-10-28 ([report](reports/C5_WORDS_AS_CLASSES.md)): count level only, far below C1; PPMI-SVD stays. (a) Cypha's DIF machinery for senses and concepts: `WorldPrior` = general meaning, one `ClassDifferential` per sense or concept, disambiguation = Cypha classify, OOD = candidate new sense. (b) The Fisher–Rao `W_enc` projection on top of C1/C2 vectors, frozen and online | Word-sense disambiguation accuracy (WordNet-tagged data), T-topic | Report both. Keep whichever wins |
| C6 | **Sentence vectors.** SIF (frequency-weighted average minus the first principal component, Arora et al. 2017) and DIF-pooled vectors | T-topic, retrieval recall@k, SCRIBE C and R axes | Best of the two is the default |
| C7 | **Into Core Cypha.** Replace the hashing embedder in Branch A routing; rerun 20NG and AG News with `W_enc` frozen | T-topic | Target ≥ 55%, stretch ≥ 60% (MiniLM 62.5%) |

### Track D: data pack and knowledge store

| ID | Work | Measure | Gate |
|---|---|---|---|
| D1 | Manifest and fetcher for every section-4 source (versions, licences, checksums) | Reproducible rebuild | Same hashes twice |
| D2 | Lexicon builder: Wiktionary + Open English WordNet → forms, lemmas, POS, senses, synonyms | Coverage of T-syntax tokens | ≥ 98% of tokens |
| D3 | Concept hierarchy: Wikidata classes + WordNet hypernyms + OpenCyc upper ontology, aligned. Cycle and consistency checks | Alignment precision on 200 hand-checked links | ≥ 90% |
| D4 | Fact store: Wikidata subset + infobox extraction from the Wikipedia dumps, with provenance | Lookup latency, RAM, ECS calibration | Section 10 budgets |
| D5 | Seed patterns: about 50 common relations × a few phrasings each, both directions | Precision on the seed set | ≥ 90% |
| D6 | The H-FIXED coverage and growth report (section 4.4) | Coverage, Heaps' law | Reported per domain |
| D7 | Versioned store: operation log, snapshots, one-command rollback | Rollback test | Passes CTest |

### Track E: the Cypha classifier as a language engine

| ID | Work | Measure | Gate |
|---|---|---|---|
| E1 | **Part-of-speech tagger.** Online Cypha classifier on UD EWT. Features: word ID, prefixes and suffixes, shape, neighbours, lexicon POS candidates from the pack, C-vectors. Baseline: an averaged perceptron we write (Collins 2002) | Accuracy on the EWT test split | ≥ 94%. Below 90% after two design rounds triggers the fallback in section 11 |
| E2 | **Chunker and entity recogniser.** BIO tagging with the same machinery. Entity types PERSON, PLACE, ORG, DATE, NUMBER. Training: UD plus silver labels from Wikipedia links typed through Wikidata | F1 per type | ≥ 80% F1 on the silver test |
| E3 | **Dependency parser.** Transition-based arc-standard parser; Cypha picks each action (Nivre 2008) | UAS / LAS on EWT | UAS ≥ 85% |
| E4 | **Question classifier.** Question type, expected answer type and relation, trained on SQuAD, Dolly, templated questions and distilled paraphrases. OOD → abstain | Accuracy, selective accuracy | ≥ 90% on answer type |
| E5 | **Entity linker.** Mention → concept using `P(entity | mention)` from Wikipedia link anchors + context-vector similarity + type agreement | Linking accuracy | ≥ 85% |

### Track F: relations and question answering

| ID | Work | Measure | Gate |
|---|---|---|---|
| F1 | **Facts answered by CyphaLM.** CyphaLM answers the 8 Wikipedia questions the trie failed, then the 200-question T-QA-wiki. The infobox lookup tool built on 2026-10-27 answered without CyphaLM and was removed under rule 10 ([record](reports/REMOVED_NON_CYPHALM_QA.md)) | T-QA-wiki | ≥ 7/8 by CyphaLM, then T-QA-wiki at 200 |
| F2 | **Pattern learning:** Hearst patterns; distant supervision (align Wikidata facts with sentences, harvest the phrasing between the two entities); bootstrapping (DIPRE, Snowball) with Beta pattern reliability and type constraints; NELL-style coupling against drift | Precision of admitted facts (weekly audit, Wilson CI) | Lower bound ≥ 0.90 |
| F3 | **Q/A pipeline** (section 3.3): parse → lookup or inference → retrieval → typed candidate extraction → AskMSR-style redundancy voting + ECS + similarity → answer or abstain → realise | T-QA-* (exact match, F1), risk–coverage, SCRIBE (Q/A profile) | See section 9 targets |
| F4 | **Composed answers.** Classic NLG pipeline (Reiter & Dale): choose facts → order them → reverse patterns into sentences → CyphaLM picks the wording | SCRIBE (all axes), human ratings | SCRIBE ≥ 0.8 × human-reference SCRIBE |

### Track G: learning, RL and self-improvement

| ID | Work | Measure | Gate |
|---|---|---|---|
| G1 | Every component gets `observe()`/`update()` with logged deltas and snapshots | Replay determinism | Bit-identical replay |
| G2 | Reward plumbing for R1–R5 (section 3.4). Rewards logged in the experiment DB (`native/sql/experiment_schema.sql`) | — | — |
| G3 | Contextual bandit for answer, pattern and sense choice; Thompson sampling over Beta posteriors | Online reward against a fixed policy | LCI ≥ 0.95 |
| G4 | Self-improvement loop with LCI as referee (section 8) | Promotions, yield, regression to the mean | No promotion with LCI < 0.95 |
| G5 | Curriculum and selective learning: pack → simple text → Wikipedia → web, with MDL gates | Learning curves (section 8.6) | — |
| G6 | Forgetting control: canaries, per-decision model files, rollback | Backward transfer (BWT) | BWT ≥ −1 point |
| G7 | Drift audits: 100 newly admitted facts per week, hand- or Wikidata-checked | Wilson lower bound | ≥ 0.90, otherwise pause and roll back |

### Track H: distilling symbolic knowledge from a neural model (offline, verified)

**What to mine and how:** [`GGUF_MINING.md`](GGUF_MINING.md) has the full list.
- **Static, from the weights:** tokenizer priors, embedding geometry and relation offsets, FFN neuron and attention-head circuits, weight spectra.
- **Dynamic, by running the model:** facts, crawls, lexical, taxonomy and commonsense relations, question paraphrases, synthetic corpora, corpus scoring, word senses.
- **Statistics:** every item carries the uncertainty statistics ECS needs.

**The tool:** [`scripts/gguf_mine.py`](../scripts/gguf_mine.py), with a self-test that runs all 18 subcommands.

| ID | Work | Measure | Gate |
|---|---|---|---|
| H1 | Pick an open-weight model under a permissive licence (Apache 2.0 or MIT). It runs offline only, never in Cypha's runtime | — | Licence recorded |
| H2 | Generate question paraphrases (pattern variety for E4 and F2), definitions, relation examples and Q/A pairs | Volume per relation | — |
| H3 | Admit through ECS with a low source prior, Beta(1, 3), until corroborated by Wikidata or text | Hallucination rate per relation | Measured and reported |
| H4 | Measure what distillation adds: the same tests with and without distilled items | LCI | Kept only if LCI ≥ 0.95 |

### Track S: knowing, not knowing, and discovery

The full design, work items S1–S15 and gates are in [`DISCOVERY.md`](DISCOVERY.md) §13. The first steps are honesty steps on the stores that exist now:
- **S1:** reason codes and evidence cards on CyphaLM's answers (the earlier non-CyphaLM Q/A tools were removed under rule 10);
- **S2:** epistemic states on `ecs.hpp`;
- **S5:** the hypothesis store with its firewall;
- **S3:** the gap map.

### Track K: thought processes

A learnable library of human thinking and work processes, plus the AI and data-science processes Cypha uses day to day. It has 144 processes in 25 categories, each a control graph with its source and core maths. Cypha runs, selects and improves them with its own statistics:
- Beta and Dirichlet counts and Welford moments;
- DIF applicability;
- Thompson selection, TD(λ) credit and the value-of-computation stop;
- promotion through LCI.

Design: [`THOUGHT.md`](THOUGHT.md). Library: [`thought/processes.yaml`](thought/processes.yaml). Work items K1–K9 are in THOUGHT.md §12; K1, the library and validator, is done.

### Track M: metrics and the evaluation harness

- **M1.** SCRIBE (section 6), ECS (section 7) and LCI (section 8) as a library, `native/include/cypha/eval/`, with CTests on synthetic cases where the answer is known.
- **M2.** The frozen suites (section 9) with checksums, plus a skip rule in every training reader.
- **M3.** One command, `cypha_report_card`, prints:
  - BPC, surface `S`, the SCRIBE profile, held-out answers, ECS calibration, LCI dashboards and the profile;
  - all with confidence intervals.
- **M4.** Meta-evaluation of SCRIBE (section 6.4) runs before SCRIBE is used to judge anything.

### Track P: profiling

See section 10. It covers every component, every checkpoint, and every promotion.

**Which support tools make the cut** (tagger, pattern learner, word vectors, teacher text, InfiniGram, Cypha abstain gate): [`TOOL_PROFILE_PLAN.md`](TOOL_PROFILE_PLAN.md). Each tool is measured only by what it adds to CyphaLM's own Q/A and BPC (rule 10), for its cost in code and RAM, with LCI as referee.

### Track Q: code health

- **Q1.** Track B1 (consolidation) plus the three test tools from 2026-10-26 folded into CTests.
- **Q2.** No hardcoded `/mnt/d/...` paths in new code: paths are arguments or `CYPHA_*` variables.
- **Q3.** CI runs the small-scale versions of every gate on a tiny fixture corpus.
- **Q4.** README, MODEL_CARD and CURRENT keep the honest wording, and point here.

---

## 6. SCRIBE: the generated-text metric

**S**yntax · **C**oherence · **R**elevance · **I**ntegrity · **B**elief · **E**xpression

Surface `S = H·D·L` stays in the report card, but it rates random corpus words (0.918) above the model (0.775), and wrong answers above correct ones. SCRIBE is built so that cannot happen, and section 6.4 tests it against exactly that. Every part of it is statistical, and nothing in it needs a neural network.

### 6.1 Normalising against human text

Every raw statistic `f` is converted to a score in `[0, 1]` by its percentile among **held-out human reference samples** of the same genre and length bucket. Call the empirical CDF `F_R`.

- **One-sided** (higher is better): `s = min(1, 2·F_R(f(x)))`. It reaches 1 at the human median.
- **Two-sided** (typical is best): `s = 1 − |2·F_R(f(x)) − 1|`. It is 1 at the human median and 0 at either tail.

Every sub-score therefore reads "how human-typical this sample is on this statistic". The reference models used for scoring are trained on text disjoint from the model under test, so the model never grades itself.

### 6.2 The six axes

**S: Syntax (form)**

| Sub-score | Statistic | Side |
|---|---|---|
| S1 lexical validity | Fraction of word tokens in the lexicon, linked to an entity, or numeric | one |
| S2 tag plausibility | Mean log-probability per token of the POS-tag sequence (E1 tagger) under a tag 4-gram trained on UD trees. Word salad has implausible tag sequences | one |
| S3 word fluency | Mean log-probability per word under an independent Kneser–Ney word 3-gram trained on disjoint human text | one |
| S4 well-formedness | Pass rate of checks: sentence-initial capital, terminal punctuation, balanced brackets and quotes, valid UTF-8, no mid-word cut | one |
| S5 parse plausibility | Mean parser action confidence (E3) | one |

**C: Coherence** (only for samples of two or more sentences)

| Sub-score | Statistic | Side |
|---|---|---|
| C1 local cohesion | Cosine similarity of adjacent sentence vectors (C6). Too low means topic jumps, too high means repetition | two |
| C2 entity continuity | Log-likelihood of entity-grid role transitions (Barzilay & Lapata 2008) under a model trained on human text | one |
| C3 topic drift | Jensen–Shannon divergence between the topic distributions of the first and second halves | two |
| C4 reference resolution | Fraction of pronouns with a compatible antecedent in the previous two sentences | one |
| C5 connective fit | After "however" or "but", cohesion should drop; after "therefore" or "so", it should hold. Fraction consistent | one |

**R: Relevance** (prompted tasks only)

| Sub-score | Statistic | Side |
|---|---|---|
| R1 match probability | `P(matched | cos(q, x))` from a logistic fit on matched and mismatched human Q/A pairs | one |
| R2 answer type | Does the first sentence contain an entity of the type E4 expects? Yes 1, partial 0.5, no 0 | — |
| R3 swap rank | Rank of `sim(q, x)` against `sim(q′, x)` over 100 other questions; score = fraction this answer fits worse. Catches a fluent answer to a different question, the failure the 8K model showed | — |
| R4 key-term coverage | IDF-weighted fraction of the question's content words and entities addressed, directly or via a synonym or hypernym from the pack | one |
| R5 informativeness | Fraction of answer content words not in the question. Catches echoing | two |

**I: Integrity** (originality and non-degeneration)

| Sub-score | Statistic | Side |
|---|---|---|
| I1 verbatim copy | Fraction of bytes inside spans of 32 bytes or more that occur verbatim in training (InfiniGram index). In **retrieval** mode it is reported, not penalised. In **generation** mode it is scored two-sided against human text's own copy rate on the same index (people quote too) | two |
| I2 self-repetition | `loop8` and the repeated-sentence rate | one (lower is better) |
| I3 diversity | distinct-1 to distinct-4 | two |
| I4 prompt echo | Fraction of the answer copied from the prompt | one (lower is better) |

**B: Belief** (factuality, using the knowledge store)

Claims in the output are extracted with the pattern layer as triples `c_j`. Each is **supported** (with that fact's ECS `P_j`), **contradicted** (a functional relation with a different value, ECS `P′_j`) or **unknown**.

| Sub-score | Statistic |
|---|---|
| B1 support | `clip01(0.5 + 0.5·(Σ P_j·[supported] − κ·Σ P′_j·[contradicted]) / n_checkable)`, with κ = 2: a confident contradiction costs twice a confident support |
| B2 verifiability | `n_checkable / n_claims`. It is reported beside B, not multiplied in. Low verifiability widens B's interval |
| B3 numeric sanity | Pass rate of constraints: birth before death, plausible ranges by type, unit consistency |
| B4 self-consistency | Agreement of the key answer across k = 5 samples with different seeds (majority fraction). Low agreement means guessing (the SelfCheckGPT idea, statistical) |

**E: Expression** (distributional naturalness, scored over a **set** of samples)

| Sub-score | Statistic |
|---|---|
| E1 Zipf slope | Rank–frequency slope of the sample set against human sets of the same size (two-sided) |
| E2 vocabulary growth | Heaps' curve of types against tokens (two-sided) |
| E3 sentence lengths | `1 − JSD(lengths_gen, lengths_human)` |
| E4 style profile | Function-word and punctuation profile: Burrows' Delta to the human reference, scored `exp(−Δ)` |
| E5 MAUVE-style frontier | Quantise human and generated sentence vectors (C6) into k clusters and compute the area under the divergence frontier (Pillutla et al. 2021), with statistical vectors instead of neural ones |

### 6.3 Combining

- **Axis score.** The geometric mean of its available sub-scores.
- **Per-sample SCRIBE.** The weighted geometric mean of axes S, C, R, I and B:
  `SCRIBE_x = exp( Σ_a w_a·ln(max(ε, A_a)) / Σ_a w_a )`, with ε = 0.01.
- **Gates.**
  - If S < 0.2, the sample is capped at S. If the task is prompted and R < 0.2, it is capped at R. Word salad and wrong-question answers cannot be rescued by other axes.
  - If B1 shows a contradicted claim with `P′ ≥ 0.9`, the sample is capped at 0.3.
- **Set-level SCRIBE.** The mean of `SCRIBE_x`, multiplied by `E^(w_E)`.
- **Human-parity ratio.** SCRIBE of the model divided by SCRIBE of held-out human text under the same profile. This is the headline number.
- **Uncertainty.** A 95% bootstrap interval (1,000 resamples over samples).
- **Always reported with it:** the six-axis profile, which gates fired, and B2. Never the single number alone.

**Task profiles (axis weights)**

| Profile | S | C | R | I | B | E |
|---|---|---|---|---|---|---|
| Open continuation | 2 | 2 | 0 | 1 | 0.5 | 1 |
| Factoid Q/A | 1 | 0.5 | 3 | 1 | 3 | 0.5 |
| Instruction / dialogue | 1.5 | 1.5 | 2.5 | 1 | 2 | 1 |

### 6.4 Meta-evaluation: SCRIBE must pass before it judges anything

**V1. Control ladders.** Each rung must score below the one above it.
- **Open text:** held-out human > model > sentences shuffled > words shuffled within sentences > random lexicon words > random letters > a repeated phrase.
- **Q/A:** correct human answer > the same answer paraphrased > right-type wrong answer > fluent answer to a different question > the question echoed > word salad > empty.
- **Pass condition:** pairwise ordering accuracy ≥ 95% over 500 pairs per adjacent rung, and the bootstrap intervals of adjacent rungs do not overlap.
- Surface `S` is run on the same ladders as the baseline. It already fails "random words < model".

**V2. Human agreement.**
- Odin rates 300 samples (100 open, 100 Q/A, 100 dialogue), 1–5 on each axis plus overall.
- Report Spearman ρ between SCRIBE and the overall rating, overall and per axis. Target ρ ≥ 0.6.
- With a second rater, report Krippendorff's α. The metric is not expected to beat human agreement.

**V3. Adversarial attacks.** Try to score high with junk, and fix SCRIBE if any attack beats the model:
- copied human text (I1);
- the question repeated back (R5, I4);
- keyword stuffing (S2, C1);
- a generic "it depends" (R3, B2);
- random dates (B3).

**V4. Stability.**
- Correlation of SCRIBE with length on human text ≈ 0.
- Variance across seeds.
- The same ladder ordering on books and web, not just Wikipedia.

**V5. Independence.** No component may use the model under test or train on its training text.

---

## 7. ECS: Evidence Certainty Score

How certain is the system that a fact (or pattern, source or rule) is true, given everything it has seen? ECS answers that with a probability, an uncertainty and a grade. It is built from counts with Subjective Logic (Jøsang 2016), which maps cleanly onto Beta distributions.

### 7.1 Evidence items

Each observation `e` that supports or opposes a fact `f` gets a **mass**:

`m_e = ω_e · τ̄_s · π_e · θ_e · δ_e`

| Factor | Meaning | How it is learned |
|---|---|---|
| `τ_s` source trust | Probability that items from source `s` are right. `τ_s ~ Beta(α_s, β_s)`, using the posterior mean τ̄ | +1 to α or β each time an item from `s` is verified (gold overlap with Wikidata, or human audit). Priors: unknown Beta(1,1), Wikidata Beta(50,1), distilled NN Beta(1,3), infobox Beta(20,1) |
| `π_e` extraction confidence | Probability that the extraction is right: pattern precision (Beta posterior mean) × Cypha's calibrated probabilities for each link and type decision used | Pattern Beta updated on every verification; Cypha calibration by temperature or isotonic fit |
| `ω_e` independence | Copies should not count as new evidence. Cluster evidence by near-duplicate text (MinHash Jaccard ≥ 0.8) and source domain. A cluster of `n` items carries total weight `1 + ln n`, shared equally: `ω_e = (1 + ln n)/n` | Clustering at ingest |
| `θ_e` time | For time-varying relations (head of state, population): `exp(−Δt / h_rel)`, half-life `h_rel` learned per relation from Wikidata value change rates. Stable relations (birthplace): 1 | Per relation |
| `δ_e` derivation | For evidence produced by inference: product of premise ECS values × the rule's reliability (Beta posterior). Depth is capped at 3, so long chains decay and cannot snowball, which is NELL's drift lesson | Rule Beta updated when a derived fact is checked |

### 7.2 Fusing the evidence for a fact

- **Positive evidence:** `r = Σ m_e` over supporting items.
- **Negative evidence:** `s = Σ m_e` over opposing items, plus `φ_rel · Σ m_c` over evidence for a *different value* of a functional relation. `φ_rel` is the relation's functionality: the fraction of subjects with one value in Wikidata, e.g. ≈ 0.95 for capital and ≈ 0 for child.
- **Constraint check:** a hard type violation (a capital that is not a settlement) adds `W_c = 5` to `s`. A soft violation adds 1.
- **Prior weight:** `W = 2` (non-informative).
- **Base rate** `a`: the prior probability of a fact of that shape, estimated from counts in the hierarchy (e.g. how often a city named in a country's article is its capital). Default 0.5.

Subjective-logic opinion:

```text
belief       b = r / (r + s + W)
disbelief    d = s / (r + s + W)
uncertainty  u = W / (r + s + W)
ECS(f) = P(f) = b + a·u            (the mean of Beta(r + a·W, s + (1−a)·W))
```

The 90% credible interval comes from that Beta. ECS is reported as `P(f)` with `u` and the interval.

### 7.3 Grades

These are GRADE-style labels for reports and for the admission rules.

| Grade | Condition |
|---|---|
| High | Lower 90% bound ≥ 0.90 |
| Moderate | P ≥ 0.75 and u ≤ 0.3 |
| Low | P ≥ 0.5 |
| Very low | P < 0.5 and d < 0.5 |
| Contradicted | d ≥ 0.5 |

### 7.4 Answer confidence

`conf(answer) = ECS(fact) × P(question parse) × Π P(entity link) × P(answer type)`.

This feeds the abstain decision. The abstain threshold is chosen on the dev set for a target precision, and reported as a risk–coverage curve: accuracy against the fraction of questions answered.

### 7.5 Calibration and audit

- **Calibration.** Reliability diagram and expected calibration error (ECE) on audited facts. Target ECE ≤ 0.05. If it is missed, fit a monotone (isotonic) map from raw P to calibrated P and report both.
- **Weekly admission audit.** Check 100 newly admitted facts. Precision uses the Wilson lower bound. If it falls below 0.90, admission pauses, the store rolls back to the last good snapshot, and the sources and patterns behind the bad facts have their Beta posteriors updated with the misses.

### 7.6 What ECS controls

| Use | Rule |
|---|---|
| Admit to working knowledge | ECS ≥ 0.8 and grade ≥ Moderate |
| Keep as a candidate | 0.5 ≤ ECS < 0.8 (keeps collecting evidence) |
| Use in an answer | `conf(answer)` above the dev-set threshold |
| Use as a premise in inference | Grade High only |
| Pattern admission | Precision lower bound ≥ 0.7 over at least 20 checks |
| Source down-weighting | Automatic through `τ_s` |

---

## 8. LCI: Learning Certainty Index

Did a change really make the system better, will the gain hold, and did anything else get worse? LCI answers that for every change: a count batch, an encoder refresh, a new pattern, a rule, a threshold, or a change the system proposed itself.

### 8.1 Sets

| Set | Use |
|---|---|
| D, dev | Proposals and tuning. Used freely |
| T, referee | The frozen test. Touched only through LCI and answered through Thresholdout (8.5). Rotated every N promotions |
| C, canaries | Earlier capabilities: classifier benches D01, D10A, T-syntax, T-QA from earlier checkpoints |
| X, external | Out-of-domain text and questions (books, web) |

### 8.2 Components

| Symbol | Quantity | Computation |
|---|---|---|
| `P_gain` | Probability the change improves T's primary metric by at least ε | Paired per-item differences, before and after; Bayesian bootstrap (Dirichlet weights, 4,000 draws); `P(Δ > ε)`. ε is the smallest gain that matters: 0.5 points of accuracy, 0.005 BPC, 0.01 SCRIBE |
| `n_eff` | Effective number of independent test items | Cluster-aware (near-duplicates and same-source items merged). If below 200, LCI is capped at 0.5 |
| `r` | Non-inferiority on canaries | For each canary k: `P(Δ_k > −δ_k) ≥ 0.95` with δ_k = 1 point. `r` = 1 if all pass, else 0 |
| `γ` | Generalisation | `G = gain_X / gain_D`, clipped to [0, 1]; `γ = min(1, G / 0.5)`. A big gain on D with none on X is memorisation, the Dolly lesson |
| `c` | Calibration | 1 if ECE rises by no more than 0.01 and the Brier score does not rise; else 0.5 |
| `σ` | Stability | 1 if the gain has the same sign across 3 seeds or data orders; else 0.5 |
| `κ` | Cost | 1 if RAM and latency stay within budget (section 10); else 0 |

**LCI = P_gain × r × γ × c × σ × κ**, a number in [0, 1].

### 8.3 Decision rule

Promote a change only if **LCI ≥ 0.95** and it passes the multiplicity check (8.4). Log every candidate, promoted or rejected, in the experiment DB with its LCI components. Rejected candidates count towards multiplicity.

### 8.4 Multiplicity: many tries must not produce false wins

The self-improvement loop proposes many candidates, and enough tries will make some look good by chance. Two controls:
- **Sequential e-values** (safe anytime-valid inference, Ramdas et al. 2023). Each candidate accumulates an e-value as test batches arrive. It is promoted only if `e ≥ 1/α_i`, with α spent from a weekly budget α_total = 0.05 by a geometric schedule `α_i = α_total·(1−ρ)·ρ^(i−1)`, ρ = 0.9.
- **Batch rounds** use Benjamini–Hochberg at q = 0.05 across the candidates in the round.

### 8.5 Protecting the referee from reuse

Repeatedly querying T leaks it. LCI therefore answers T through **Thresholdout** (Dwork et al. 2015): T reports the dev result unless T and D disagree by more than a noisy threshold, and then it returns T's value plus Laplace noise. T is rotated every N = 20 promotions or when the Thresholdout budget is spent, and the retired T joins the canaries.

### 8.6 Learning dashboards

These are statistical health measures of learning itself, all in the report card.

| Dashboard | Definition | Use |
|---|---|---|
| Learning curve | Metric against data seen, with a fitted `L(n) = L∞ + a·n^(−b)` | Predicts the value of more data; checks Track A |
| Sample efficiency | Data needed to reach a target, **with and without the pack** | Tests "a data pack instead of masses of data" |
| Forgetting | Backward transfer `BWT = mean_k (acc_k,final − acc_k,just learned)` (Lopez-Paz & Ranzato 2017) | Continual-learning health (D16B) |
| Knowledge growth | Facts by ECS grade over time; admission precision with Wilson CI | Drift detection |
| Self-improvement yield | Promoted ÷ proposed; the realised gain of each promotion on the next rotated T | Exposes regression to the mean |
| Calibration over time | ECE and Brier per checkpoint | Confidence stays honest |
| Learning certainty | LCI of each promotion, as a histogram | How sure each step was |

---

## 9. Frozen evaluation suites

All suites live in `bench/eval/` with SHA-256 checksums and a README. Training readers skip `/eval/`.

| Suite | Contents | Metrics | First target |
|---|---|---|---|
| T-perturb | Each Q/A item in four forms: verbatim, one letter changed, " Thanks." appended, synonym swap | Accuracy per form | Changed forms ≥ 95% of verbatim |
| T-QA-wiki | Hand-written questions answerable from the Wikipedia slice, with accepted variants. Starts with today's 8, grows to 200 (Odin writes or reviews) | Exact match, F1, risk–coverage | 7/8 now; ≥ 70% at ≥ 80% coverage at 200 |
| T-QA-dolly | 10% of Dolly, stratified by category, near-duplicates removed, never indexed | Exact match, F1, SCRIBE (Q/A) | Reported; L1 rung |
| T-QA-squad | SQuAD v1.1 dev subset: closed (paragraph given) and open (paragraph not given, article in corpus) | Exact match, F1 | Reported |
| T-QA-wikidata | Templated and paraphrased questions about Wikidata facts **removed from the store**, so they must be read from text | Accuracy, ECS calibration | Reported; L4 rung |
| T-words | SimLex-999, WordSim-353, MEN | Spearman ρ | Within published PPMI-SVD ranges |
| T-topic | 20 Newsgroups (2k protocol, as in Branch A), AG News | Accuracy, macro-F1 | ≥ 55% on 20NG at 2k |
| T-syntax | UD English EWT test split | POS accuracy, UAS, LAS | 94% / 85% UAS |
| T-lm | enwik 8 MiB holdout + one frozen holdout per Track A category | BPC | Track A predictions |
| T-text | Human reference samples per genre and length bucket, plus the section 6.4 control ladders | SCRIBE, surface `S` | Ladders pass |
| T-dialogue | Multi-turn prompts with rated reference replies (to be built) | SCRIBE (dialogue), human rating | Stretch (L5) |
| T-canary | Earlier capabilities for LCI non-inferiority | Per-suite metric | No drop beyond δ |

---

## 10. Profiling and budgets

Every component reports:
- peak RSS, resident and memory-mapped bytes;
- build time;
- p50 and p99 latency;
- throughput;
- disk size.

The tools are `/usr/bin/time -v`, `perf`, heaptrack, and the existing hot-path profilers. Every report card carries the profile.

These are initial budgets, revised by measurement but never silently:

| Component | Budget |
|---|---|
| Whole language system, resident | ≤ 4 GB on a desktop; large stores memory-mapped |
| Core runtime (tagger, parser, question classifier, linker, encoders) | ≤ 1 GB resident |
| Packed trie at 1 TB | Measured in Track A (P-A3 warns it may be GBs) |
| Fact lookup | ≤ 50 µs |
| Word vector lookup / sentence vector | ≤ 10 µs / ≤ 50 µs |
| Tagger / parser throughput | ≥ 20k / ≥ 5k tokens per second per core |
| Q/A end to end | p50 ≤ 200 ms, p99 ≤ 1 s, one core |
| Report card run (small suites) | ≤ 15 min |

---

## 11. Risks and kill criteria

| Risk | Signal | Response |
|---|---|---|
| Cypha DIF struggles with sparse text features (TF-IDF scored 34%) | E1 below 90% after two design rounds | Feed dense C-vectors and hashed features. If it still fails, the averaged perceptron does syntax and Cypha keeps confidence, OOD and routing. **Measured 2026-10-27 ([E4](reports/E4_QUESTION_CLASSIFIER.md)):** the risk is confirmed for question types. On TREC, Cypha (W_enc frozen, online) scored 75.0 coarse and 62–67 fine; the averaged perceptron scored 89.0–91.6 and 84.6. Cypha's confidence was badly under-calibrated, and its OOD flag never fired. The perceptron is the question classifier for now. Fixing Cypha's online update on text is an open research item |
| Pattern coverage too low | T-QA-wiki under 50% after F2 | More distant supervision, distilled paraphrases (H2), retrieval fallback |
| Drift: errors admitted, then used to admit more | Weekly audit lower bound below 0.90 | Pause, roll back, update the source and pattern Betas, tighten thresholds |
| Pipeline error compounding | Per-stage accuracy product far below end-to-end needs | Joint decisions where cheap (tag and chunk together); keep the top-k parses |
| Metric gaming | Any V3 attack beats the model | Fix SCRIBE before using it again |
| Licence conflicts for commercial use | Share-alike in a shipped pack | Section 4.3 options |
| The language-as-fixed-construct hypothesis fails | D6 coverage under 99% with growth not flattening | Learning loops carry more; the pack becomes the prior, not the whole |
| Open conversation (L5) stays out of reach | T-dialogue flat | Report it plainly; L2–L4 still stand on their own |
| Time (a full-time job alongside) | — | Order of work favours quick, decisive tests |

---

## 12. Order of work

1. **Now.** The test tools and report (done 2026-10-26). Track A keeps running; score it at each checkpoint.
2. **Code health.** Q1–Q3 and B1 (consolidated trie, CMake, CTests, paths).
3. **Measurement first.**
   1. Build the T-suites.
   2. SCRIBE v0 with S1, S3, S4, R1–R5 and I1–I4, plus the V1 ladders and the surface-`S` baseline on them.
   3. The ECS and LCI library.
4. **Quick, decisive wins.**
   1. F1: infobox facts + question classifier v0 on the 8 Wikipedia questions.
   2. B7: fuzzy, honest chat index with Dolly + OpenAssistant.
5. **Semantics from statistics.**
   1. C0 baselines.
   2. C1 PPMI-SVD and random indexing.
   3. C2 predictive-distribution vectors, C3 compression distance, C5 Odin's encoder.
   4. C6 sentence vectors, then C7 into Core Cypha.
6. **The classifier as a language engine.** E1 tagger → E2 entities → E4 questions → E5 linker → E3 parser.
7. **The data pack.** D1–D5 and D7, then the D6 H-FIXED report.
8. **Model upgrades.**
   1. B2 8K expert, B3 word level, B4 class level, B5 long range.
   2. B9 copy-from-prompt, B10 decoding.
   3. B6 serve wiring.
9. **Relations and answers.** F2 pattern learning, F3 full pipeline, F4 composed answers.
10. **Learning.** G1–G3, then the G4 self-improvement loop with LCI, plus G5–G7 running throughout.
    **Knowing and discovery (Track S)** runs alongside from step 4 on:
    - S1, S2, S5 and S3 come as soon as the fact store and chat index exist (they do now);
    - the generators follow once F2 and D4 give them a store to work on (order in [`DISCOVERY.md`](DISCOVERY.md) §13).
11. **Distillation.** H1–H4 once ECS and LCI exist to police it.

Each step ends with a dated report and a report card. Promotions to defaults go through LCI.

---

## Appendix A: discussion ledger

Everything raised on 2026-10-26, and where this plan covers it.

| # | Point raised | Where |
|---|---|---|
| 1 | Check CyphaLLM, give feedback; it answered 3 questions from the data | §0, report |
| 2 | Trained on "8 MB of Wikidata" | §4.1 correction: it was enwik9 Wikipedia text; real Wikidata is in the pack |
| 3 | Trained with 2 popular Q/A sets | §4.1: the committed tool loads Dolly only; B7 adds OpenAssistant |
| 4 | "So it predicts the answer from stored answers" | §0: it looks them up; B7 makes lookup honest and fuzzy |
| 5 | Another AI (Grok) gave a wrong picture | Rule 1: three probes for any AI's claim |
| 6 | Aim: good general NLP and LLM functionality | §1 goal and capability ladder |
| 7 | "BPC of 2, look at the textgen metric, it has OK textgen" | §0 surface controls; §6 SCRIBE; surface `S` kept in the report card |
| 8 | "Only thing missing is context. Scale. Increase the bit count. LLMs have 256k context. Test 8K" | §0 8K test; B2; A6 bit-count scaling; §3.2 context hierarchy; B5 long range |
| 9 | "Change that to an order-8K count model, try it" | §0 (InfiniGram capped at 8192); B2; A5 |
| 10 | "BPC is a bad metric for language modelling; we have the surface score" | §1 evaluation contract: BPC measures fit, not generation; surface `S` kept; SCRIBE added |
| 11 | "Remember it" (report held-out answers, surface score and BPC together) | §1, §9, report card M3 |
| 12 | No NNs. Low RAM. It learns and compresses, extracts probability distributions, is picky about data | §1 decisions; C2, C3; §3.4 selective learning; §10 budgets |
| 13 | How do we extract semantics and relations with this algorithm? | Tracks C, D, F |
| 14 | Look at Core Cypha, learn from its lessons | §0 history; §3.1; §11 |
| 15 | CyphaLM is a very different machine to Core Cypha | §3.1; C2 and C3 use CyphaLM's own machinery |
| 16 | "Maybe just do an encoder. The statistical encoder I have is very statistical. Try that" | C5 (DIF senses, `W_enc` frozen and online); C1–C6 |
| 17 | Feedback first, then a plan | This document |
| 18 | "Language is a fixed data construct" | §4.4 H-FIXED test; D6 |
| 19 | A data pack instead of learning from masses of data | §4; sample-efficiency dashboard (§8.6) |
| 20 | Use the classifier to do RL | §3.4 RL; G2, G3 |
| 21 | Dictionary, thesaurus, text statistics, data | §4.1: Wiktionary, Open English WordNet, Track A statistics, Wikidata |
| 22 | Full symbolic, online, classifier, recursive self-improvement | §1 hybrid decision; §3.4; G4 with LCI |
| 23 | Generative models learn what symbolic does; distil the symbolic from an NN | Track H |
| 24 | Agreed on the hybrid approach | §1 |
| 25 | The question is data representation and hierarchy | §3.2 (two hierarchies) |
| 26 | Learning must be dynamic | §3.4, G1 |
| 27 | Once the data model is set up and it learns, what does it do? | §3.3 runtime loop |
| 28 | We need to teach it how to learn too | §3.4 levels 0–3 |
| 29 | Is the Cyc dataset downloadable? | §4.1: OpenCyc 4.0 (Apache, mirrors); full Cyc is not public |
| 30 | Plan from the count-model-context point on; pick what to do; be comprehensive | This document |
| 31 | We will profile all of this | §10, Track P, κ in LCI |
| 32 | Keep the 1 TB run and test its BPC; experiment on the model | Track A, A1–A8, P-A1–P-A5 |
| 33 | Commit to Cypha as the new default plan | This file; README, CURRENT, docs hub and bill of work point here |
| 34 | A metric to rate generated text; go crazy | §6 SCRIBE |
| 35 | A weighted metric for certainty of evidence and of learning; go crazy; mostly statistical | §7 ECS, §8 LCI |
| 36 | Include the model upgrades | Track B (B1–B11) |
| 37 | Earlier advice: held-out eval, CTest for the trie, consolidate the forks, honest wording | Rules 1, 2, 7; B1; Q1–Q4 |
| 38 | Earlier advice: fuzzy retrieval with normalisation | B7 |
| 39 | Earlier advice: part-of-speech tagger test, infobox quick win | E1, F1 |
| 40 | Earlier advice: word-similarity tests, 20NG rerun, PPMI-SVD, random indexing, SIF | §9, C0, C1, C6 |
| 41 | Earlier advice: NELL drift, referee for self-improvement | §3.4, §7.5, G7, §8 |
| 42 | Earlier advice: verify distilled facts; licences | H3; §4.3 |
| 43 | A full list and a script to statistically datamine a GGUF neural model, relations included | Track H; [`GGUF_MINING.md`](GGUF_MINING.md); `scripts/gguf_mine.py` |
| 44 | Keep "it doesn't answer what it doesn't know" as a feature | Rule 9; right / abstained / wrong reporting |
| 45 | Add speculative / hypothetical thinking for discovery later | §3.5 speculative mode, kept apart from answers |
| 46 | Do the speculative thinking now: know what it knows, know what it doesn't, think hard on discovery, map out the discovery processes | [`DISCOVERY.md`](DISCOVERY.md); Track S |
| 47 | Model human thought and work processes across many categories (UML/processes) as "thought" for Cypha, done mathematically with its online learning and statistical feedback | Track K; [`THOUGHT.md`](THOUGHT.md); [`thought/processes.yaml`](thought/processes.yaml) |
| 48 | Cut and merge tools to the minimal and best (keep `gguf_mine.py`); plan to profile the tagger, pattern learner and word vectors with other tools to see what makes the cut | Code-debt cuts (changelog); B1; [`TOOL_PROFILE_PLAN.md`](TOOL_PROFILE_PLAN.md) |
| 49 | Get vectors and statistical association into Cypha, extract them from a trained LLM, and reinterpret Core Cypha | Track C; [`SEMANTICS.md`](SEMANTICS.md); experiment S-1 |
| 50 | Make Cypha's vectors run like Qwen3-Embedding-8B's (how they attract and push away), fully dissect and datamine it, with Qwen3.5-9B as the mining LLM | Track C; [`EMBEDDING_GEOMETRY.md`](EMBEDDING_GEOMETRY.md); QE0–QE9 |
| 51 | Answering routes (2026-10-28): R1 question-form fact text and R3 statement scoring first; R3's frames may come from the question parse and the learned patterns | [`CYPHALM_ANSWERING.md`](CYPHALM_ANSWERING.md) §5; A2, A3 |
| 52 | Teacher (Qwen3.5-9B) Q/A text is allowed only as a separate, leakage-checked, low-trust shard, measured against Cypha's own fact store alone (2026-10-28) | [`CYPHALM_ANSWERING.md`](CYPHALM_ANSWERING.md) §5; TOOL_PROFILE teacher-text shard |
| 53 | The familiarity control token is allowed, on the gate side like the abstain gate; try an entity-level familiarity; keep the token after `Assistant: ` (2026-10-28) | [`CYPHALM_ANSWERING.md`](CYPHALM_ANSWERING.md) §5; A0/A1 report, flags 1–2 |
| 54 | Do not run the Qwen models in the cloud container: only analytics on their weights. Inference and distillation run on Odin's PC (2026-10-28) | [`EMBEDDING_GEOMETRY.md`](EMBEDDING_GEOMETRY.md) §9; GGUF_MINING §14 |
| 55 | "Try a lot of things, profile and record everything, then pick a fact-based decision" (2026-10-28): open design questions are settled by measurement, with the evidence recorded | Every report's Decisions section; this ledger |
| 56 | A paper on the text-generation metric for Zenodo: hard work, cast the net wide, profile it until it is right, many comparisons (2026-10-28) | [`SCRIBE_PAPER_PLAN.md`](SCRIBE_PAPER_PLAN.md); [`reports/SCRIBE_V2.md`](reports/SCRIBE_V2.md); [`../paper/scribe/`](../paper/scribe/) |
| 57 | Reduce CyphaLM's RAM use (2026-10-28) | [`reports/RAM_PROFILE.md`](reports/RAM_PROFILE.md): measured, with a rule fixed before each change. M10–M14 (2026-10-28): `qa` served from a store, under 200 MB in every configuration. M15 (2026-10-28): the class build in key buckets; built `qa` 226 → 194 MB (defaults), 259 → 235 MB (C1L + defaults), lossless  M16 (2026-10-29): the FM index built in place, the class table after it; built `qa` 235 → 185 MB (C1L + defaults), lossless  M17 (2026-10-29): the lowered index freed once `--save` has written it; C1L + defaults with `--save` 205 → 187 MB, lossless |
| 58 | Fix the CI jobs that are not green (2026-10-28) | Done: all six `ci.yml` jobs green on `main` (HANDOFF §4.5) |
| 59 | Make the smaller class table the default; don't worry about returning freed memory to the OS; use the similar-context expert as soon as it is useful (2026-10-28) | `class_context.hpp` defaults (CHANGELOG); the similar-context expert gated off inside answers and made a default where measured useful (A4 report) |
| 60 | Next work: the echo fix, Track S1 and serving from stored files, "all three in parallel" (2026-10-28) | [`reports/B4C_ECHO_GATE.md`](reports/B4C_ECHO_GATE.md) (class and copy off inside answers, defaults where the rules hold); [`reports/S1_REASON_CODES_AND_EVIDENCE.md`](reports/S1_REASON_CODES_AND_EVIDENCE.md); [`reports/RAM_PROFILE.md`](reports/RAM_PROFILE.md) M10–M14 |
| 61 | A reason code describes what CyphaLM has read ("You pick. Most honest."); S1's 82% stands: "If the data isn't there, it isn't there. Don't worry about it." (2026-10-28) | HANDOFF §2 and §5.7; S1 report |
| 62 | Decided by measurement under row 55 (2026-10-28), open to Odin's veto: for text, DISCOVERY §3.3's missing mass counts by page (S4: within 4%, n₁/N undercounts by 20–50%); S6's P0 keeps DISCOVERY §6.1's constraint term, and its parts are judged by one joint test, not one at a time (S6b) | [`reports/S4_UNSEEN_AND_FRONTIER.md`](reports/S4_UNSEEN_AND_FRONTIER.md); [`reports/S6_HYPOTHESIS_SCORING.md`](reports/S6_HYPOTHESIS_SCORING.md) |
| 63 | Delete the 35 hybrid-era `cypha_slow` tests and the code only they use (2026-10-29: "Delete them") | [`reports/SLOW_TEST_TRIAGE.md`](reports/SLOW_TEST_TRIAGE.md) |
| 64 | The text score the decision rules use is SCRIBE v2; surface S is reported alongside (2026-10-29) | [`reports/SCRIBE_V2.md`](reports/SCRIBE_V2.md); HANDOFF §2 rule 5 |
| 65 | Keep `pattern_learn` as Track S's fact and hypothesis tool; it never answers (2026-10-29) | [`TOOL_PROFILE_PLAN.md`](TOOL_PROFILE_PLAN.md); HANDOFF §5 |
| 66 | Decided by measurement under row 55 (2026-10-29), open to Odin's veto: the hypothesis loop uses three-way likelihoods; T1's φ comes from the store; the batch step is BH among the supported (selective; whole-batch e-BH remains the guaranteed alternative); confirmed on S15 dev at 0.956 (S7b) | [`reports/S7_HYPOTHESIS_TESTS.md`](reports/S7_HYPOTHESIS_TESTS.md) |
| 67 | Promotion needs another evidence source: "Get another evidence source. In this situation the result would be to datamine and assess data sources reliability." (2026-10-29) | S8c (running): candidate sources datamined, each one's reliability and independence from Wikidata measured before it may promote |
| 68 | The batch rule is "not up to scratch yet": "Hybrid or something better. Keep working on it." (2026-10-29) | S7c (running) |
| 69 | Fix both hp-wrapper bugs: forecast rollouts keep the trained model (its output may change); a vocabulary under 256 is rejected and the codec test uses 256 (2026-10-29) | HPFIX (running); SLOW_TEST_TRIAGE |
| 70 | Decided by measurement under row 55 (2026-10-29), answering row 68, open to Odin's veto: the batch step is a two-tier hybrid. Known = BH at 0.1 on null-model p-values over the whole batch (FDR ≤ 0.1 if the null likelihoods are right); Believed = `decide`'s other supported, never Known. 23% decided on dev at 0.974; confirmed on a fresh set at 0.954 [0.936, 0.968] (S7c) | [`reports/S7_HYPOTHESIS_TESTS.md`](reports/S7_HYPOTHESIS_TESTS.md) |
| 71 | Decided by measurement under row 55 (2026-10-29), open to Odin's veto: `decide` supports only with a positive independent result (DISCOVERY §7.2; a bug fix: one hypothesis moves in S7–S7c, none in S8a or S8b; S8c read 3 0.667 → 0.750). An evidence source carries a promotion only where its LR+ on the generators' graded errors is ≥ 10 (DBpedia dates, places, links; GeoNames places; not Wikipedia leads, not DBpedia occupations and genres). S8's gate on a fresh confirm-half set: not met, 25 graded at 0.880 [0.700, 0.958]; the 3 misses are S15 name or grain mismatches | [`reports/S8D_PROMOTION.md`](reports/S8D_PROMOTION.md) |
| 72 | The generation track (2026-10-29: "That's ok to do what you suggested"): clean training text, a sentence-level gate, word and topic experts, and a content-plan expert in which the fact store steers through a next-byte expert while CyphaLM writes every byte; off inside answers; each step adopted only on a registered SCRIBE v2 read | GEN-A, GEN-B, GEN-E (running) |
| 73 | Plan CyphaLM over tokens (2026-10-29: "Just queue to do a plan … take tokens next level") | [`TOKENS_PLAN.md`](TOKENS_PLAN.md) |
| 74 | Token plan's questions (2026-10-29: "You pick for the questions"): subwords with word boundaries then superwords; beside the byte model, defaults switch only on T4 and T10; started now as its own tool `cyphalm_tok`; T11 on Odin's PC after T4, never competing with Track A | TOK-A, TOK-B (running) |
| 75 | Bring forecasting to parity with the other Cypha features (2026-10-29: "Make a plan for it. Do docs for it. Go crazy on it. I want to bring this into parity with other Cypha features.") | [`FORECAST_PLAN.md`](FORECAST_PLAN.md) (Track FC); [FC0](reports/FC0_FORECAST_AUDIT.md) done; decisions in FORECAST_PLAN §9, HANDOFF §5.24 |
| 76 | Decided by measurement under row 55 (2026-10-29), answering HANDOFF §5.21, open to Odin's veto: G13's anomalies enter the source stage with its verdict as the prior, never as verdicts (questions-only at 0.5 lost the refutations, 8 of 9 right against 16 of 16); G3 orders its added facts by the link predictor alone (dev recall@1 +0.066 [+0.023, +0.113]); G4 carries no dates as constants (dev MRR −0.001 [−0.004, +0.001]); A6 kept. Dev only; the fresh read waits for S8e's set | [`reports/S11C_ANALOGY_ANOMALY_ABDUCTION.md`](reports/S11C_ANALOGY_ANOMALY_ABDUCTION.md) (S11e) |
| 77 | Decided by measurement under row 55 (2026-10-29), answering HANDOFF §5.12 and §5.14, open to Odin's veto: T2 stays out of the loop (Known-decided +0.14 points against 0.5); Wikidata counts as one source; K4 stays the Known tier; the store's φ is the recommended fusion setting with the default unchanged (Odin's call); S9's single points of failure are a fragile gap kind behind `--gaps-spof`; DISCOVERY §4.4 use 1 dropped under rule 10 | [S9 report "S9b"](reports/S9_WORLDS_AND_FRAGILITY.md), [S7 report "S7d"](reports/S7_HYPOTHESIS_TESTS.md) |
| 78 | Decided by measurement under row 55 (2026-10-29), open to Odin's veto: the low-only don't-know token is the answering form | [`reports/A1D_A2B_ANSWERING_VARIANTS.md`](reports/A1D_A2B_ANSWERING_VARIANTS.md) (ANS) |
| 79 | Decided by measurement under row 55 (2026-10-29), open to Odin's veto: the token's statistic is `ent_bold` (frozen test 0.963 / 0.52, D2 met; `ent_title` 0.635 / 0.94) | ANS |
| 80 | Rule 3 adopted the entity frame + `--copy-dk` for the tag (test: tags name the entity 49 of 50); veto recommended (the message disagrees with the tag in 50 of 50; frame-token echo in 23 of 97); defaults unchanged until Odin decides | ANS |

---

## Appendix B: options not taken

| Option | Why not |
|---|---|
| A neural language model in the runtime (open-weight model + Cypha on top, or a small Transformer trained from scratch) | Decision, 2026-10-26: no NNs at runtime. Kept only as an offline distillation source (Track H) and as an external yardstick |
| Full symbolic only (Cyc-style) | Cyc's decades of hand-built knowledge did not produce general language understanding; coverage is the failure. Hybrid instead |
| A longer byte window as the route to context | Measured: gains stop at about 16 bytes (median match 10). Context grows by counting over words, classes and documents instead (§3.2) |
| More data as the route to question answering | Prediction P-A4; Track A tests it directly |
| Hebbian encoder updates | Lost on 4 of 4 tasks (Phase 2A) |
| Attention memory (`compressive_memory`, `context_bank`) | +0.2 to +0.27 BPC; killed |
| BPE tokens for the byte model | 4.15 vs 2.87 BPC; killed |
| Learning `W_enc` online on semantic vectors | 49.3% vs 62.5% frozen. C5 tests it again on the new statistical vectors, as asked, but frozen is the default |
| RFF on high-dimensional text vectors | 4.5% vs 59.5% with `VectorEncoder` |

---

## Appendix C: references

**Distributional and compression semantics**
- Levy, Goldberg & Dagan (2015). Improving distributional similarity with lessons learned from word embeddings. *TACL*.
- Arora, Liang & Ma (2017). A simple but tough-to-beat baseline for sentence embeddings. *ICLR*.
- Brown et al. (1992). Class-based n-gram models of natural language. *Computational Linguistics*.
- Sahlgren (2005). An introduction to random indexing.
- Halko, Martinsson & Tropp (2011). Finding structure with randomness. *SIAM Review*.
- Faruqui et al. (2015). Retrofitting word vectors to semantic lexicons. *NAACL*.
- Cilibrasi & Vitányi (2005). Clustering by compression. *IEEE Trans. Inf. Theory*.
- Jiang et al. (2023). "Low-resource" text classification: a parameter-free classification method with compressors. *Findings of ACL*. (Rerun it ourselves.)

**Language models and context**
- Kneser & Ney (1995). Improved backing-off for m-gram language modeling. *ICASSP*.
- Kuhn & De Mori (1990). A cache-based natural language model for speech recognition. *IEEE TPAMI*.
- Rosenfeld (1996). A maximum entropy approach to adaptive statistical language modelling. *Computer Speech & Language*.
- Liu et al. (2024). Infini-gram: scaling unbounded n-gram language models to a trillion tokens.

**Syntax**
- Collins (2002). Discriminative training methods for hidden Markov models. *EMNLP*.
- Nivre (2008). Algorithms for deterministic incremental dependency parsing. *Computational Linguistics*.

**Relations, knowledge and question answering**
- Hearst (1992). Automatic acquisition of hyponyms from large text corpora. *COLING*.
- Brin (1998). Extracting patterns and relations from the World Wide Web (DIPRE).
- Agichtein & Gravano (2000). Snowball. *ACM DL*.
- Mintz et al. (2009). Distant supervision for relation extraction without labeled data. *ACL*.
- Carlson et al. (2010). Toward an architecture for never-ending language learning. *AAAI*.
- Mitchell et al. (2018). Never-ending learning. *CACM*.
- Lenat (1995). CYC: a large-scale investment in knowledge infrastructure. *CACM*.
- Dumais, Banko, Brill, Lin & Ng (2002). Web question answering: is more always better? *SIGIR*.

**Generation and text quality**
- Reiter & Dale (2000). *Building Natural Language Generation Systems*.
- Barzilay & Lapata (2008). Modeling local coherence: an entity-based approach. *Computational Linguistics*.
- Pillutla et al. (2021). MAUVE. *NeurIPS*.
- Manakul, Liusie & Gales (2023). SelfCheckGPT. *EMNLP*.
- Burrows (2002). 'Delta': a measure of stylistic difference. *LLC*.

**Evidence and learning certainty**
- Jøsang (2016). *Subjective Logic*.
- Guyatt et al. (2008). GRADE. *BMJ*.
- Wilson (1927). Probable inference, the law of succession, and statistical inference. *JASA*.
- Broder (1997). On the resemblance and containment of documents (MinHash).
- Benjamini & Hochberg (1995). Controlling the false discovery rate. *JRSS B*.
- Ramdas, Grünwald, Vovk & Shafer (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science*.
- Dwork et al. (2015). The reusable holdout. *Science*.
- Lopez-Paz & Ranzato (2017). Gradient episodic memory for continual learning. *NeurIPS*.
