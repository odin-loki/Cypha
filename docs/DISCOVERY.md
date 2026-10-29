# Cypha discovery: what it knows, what it doesn't, and how it finds out

**Odin Loch** · 2026-10-27 · Track S of [PLAN.md](PLAN.md)

This is the map Cypha follows for three things:
- knowing what it knows, and how sure it is;
- knowing what it does not know, why, and roughly how much;
- discovery: forming hypotheses, choosing which to chase, testing them, and turning the survivors into knowledge.

It builds on the evidence score ECS ([PLAN §7](PLAN.md#7-ecs-evidence-certainty-score)) and the learning referee LCI ([PLAN §8](PLAN.md#8-lci-learning-certainty-index)). Both are implemented in `native/include/cypha/eval/`.

---

## Contents

1. [Two modes and the firewall](#1-two-modes-and-the-firewall)
2. [Knowing what it knows](#2-knowing-what-it-knows)
3. [Knowing what it does not know](#3-knowing-what-it-does-not-know)
4. [Hypotheses: the speculative store](#4-hypotheses-the-speculative-store)
5. [Where hypotheses come from: 20 generators](#5-where-hypotheses-come-from-20-generators)
6. [Which hypotheses to chase: scoring](#6-which-hypotheses-to-chase-scoring)
7. [Testing hypotheses](#7-testing-hypotheses)
8. [The discovery processes](#8-the-discovery-processes)
9. [Curiosity: allocating the effort](#9-curiosity-allocating-the-effort)
10. [Talking about it: cards and query forms](#10-talking-about-it-cards-and-query-forms)
11. [Honesty invariants](#11-honesty-invariants)
12. [Measuring all of this](#12-measuring-all-of-this)
13. [Track S: work items and order](#13-track-s-work-items-and-order)
14. [References](#14-references)

---

## 1. Two modes and the firewall

```text
                         ┌──────────────────────────────────────────┐
   question / text ─────►│  ANSWER MODE (default)                    │──► answer + evidence card
                         │  uses: Known and Believed items only      │──► "I don't know" + reason card
                         └───────────────▲──────────────────────────┘
                                         │ promotion only through ECS,
                                         │ with evidence independent of the generator
                         ┌───────────────┴──────────────────────────┐
   "speculate", "what if",│  DISCOVERY MODE                           │──► hypothesis cards
   gaps, anomalies ──────►│  may read everything, writes only to the  │──► tests, research journal
                         │  hypothesis store and the gap map         │
                         └──────────────────────────────────────────┘
```

**The firewall rules**
- **Answer mode** reads only items whose epistemic state is Known or Believed (section 2.1). It never reads a hypothesis, or anything derived from one.
- **Discovery mode** may read every store, but writes only to the hypothesis store, the gap map and the research journal.
- **Taint.** Every item carries a `hypothetical` flag, and anything derived from a hypothetical item inherits it. Answer mode filters on the flag; there are no exceptions.
- **Promotion.** A hypothesis becomes knowledge only when ECS admits it on its own evidence (PLAN §7.6). The generator's support for its own idea is the hypothesis's **prior**, never its evidence.
- **Default.** Answer mode is the default. Discovery output appears only when asked ("speculate…", "what if…", "what don't you know about…") or in the research journal. It is always labelled.

This is PLAN rule 9 made structural: no evidence, no answer. Speculation is allowed, and it is never passed off as an answer.

---

## 2. Knowing what it knows

### 2.1 Epistemic states

Every fact, pattern, rule, concept link and sense carries an ECS opinion `(b, d, u)` with `P = b + a·u` (PLAN §7.2). Its state follows from that opinion:

| State | Condition | May be used for |
|---|---|---|
| **Known** | Grade High (90% lower bound ≥ 0.90) | Answers; premises for inference |
| **Believed** | Grade Moderate or Low | Answers, with its grade stated; not as an inference premise unless Moderate |
| **Contested** | d ≥ 0.5, or degree of conflict with a rival value DC ≥ 0.3 (section 2.3) | Never as an answer; it is reported as contested |
| **Ignorant** | u ≥ 0.8 (almost no evidence either way) | Never |
| **Vacuous** | No evidence at all (u = 1) | Never |
| **Hypothesis** | Lives in the hypothesis store; prior from its generator | Discovery mode only |
| **Refuted** | A hypothesis whose tests went against it (section 7.5) | Blocks regeneration; kept as a negative result |

### 2.2 Two kinds of uncertainty

Cypha keeps two kinds of "not sure" apart, because they call for different actions:

| Kind | Signal | Meaning | Action |
|---|---|---|---|
| **Lack of evidence** (epistemic) | u high | It hasn't seen enough | Look for evidence: gap-closing (process P1) |
| **Mixed evidence** (aleatoric or ambiguous) | u low, P near 0.5, b and d both substantial | Plenty of evidence that genuinely disagrees, or the item is vague | Look for a hidden variable: another time, sense, entity or qualifier (process P4) |

For the classifier's own decisions the same split exists already. The epistemic variance and the OOD flag measure lack of evidence; the class-probability entropy measures ambiguity.

### 2.3 Conflict between beliefs

For two opinions A and B about the same functional slot (two rival capitals, say), Subjective Logic's degree of conflict is:

`DC(A, B) = |P_A − P_B| × (1 − u_A)(1 − u_B)`

That is the projected distance times the conjunctive certainty. It is high only when both opinions are confident and they disagree. DC ≥ 0.3 marks the slot Contested and opens a contradiction gap (section 3.2).

### 2.4 The self-model of competence

Cypha keeps a table about itself. Each cell is a pair of (relation or skill) and (entity type or domain), and holds:

| Field | Statistic |
|---|---|
| `accuracy` | Beta posterior from audited facts and verified Q/A in that cell |
| `coverage` | Fraction of expected slots in the cell with a Known or Believed value (section 3.2) |
| `abstain_rate` | Fraction of questions in the cell answered "I don't know" |
| `overclaim_rate` | Wrong answers given with confidence ≥ 0.9, with a Wilson upper bound |
| `n` | Evidence count behind these numbers |
| `last_audit` | When the cell was last checked |

The table answers questions about Cypha itself: "How good are you at capitals?", "What do you know about chemistry?". It also sets per-cell abstain thresholds and tells the curiosity scheduler where it is weak (section 9).

### 2.5 Introspection queries

The system can be asked about its own knowledge, and asks these of itself on a schedule (process P13):

- Which Known facts rest on a **single source**? This is fragility; see section 7.6.
- Which **rules** are used in derivations but have fewer than 20 verified predictions?
- Which facts are **stale**: past the half-life of their relation (PLAN §7.1 θ)?
- Where has **calibration drifted**, with cell ECE up by more than 0.02 since the last audit?
- Which **sources** have trust posteriors that moved most this period?
- Which **questions** keep being abstained on? That is demand.

---

## 3. Knowing what it does not know

### 3.1 Why it doesn't know: reason codes

Every "I don't know" carries the first reason that applies, in this order, plus the full breakdown:

| Code | Detected by | Statistic |
|---|---|---|
| `OUT_OF_SCOPE` | Cypha OOD flag on the question, or the topic is outside every domain in the store | OOD score above threshold |
| `UNPARSEABLE` | No question pattern matches with P ≥ 0.5 | 1 − P(parse) |
| `AMBIGUOUS_QUESTION` | Two or more relations fit the question, close in probability | Entropy of the relation distribution |
| `UNKNOWN_ENTITY` | No concept links to the mention | Best link P below threshold |
| `AMBIGUOUS_ENTITY` | Two or more concepts fit the mention ("Paris": city, person, myth) | Entropy over link candidates |
| `NO_EVIDENCE` | The slot is empty | Vacuous or Ignorant state |
| `CONFLICT` | Contested slot | DC, or d ≥ 0.5 |
| `LOW_CERTAINTY` | A value exists but conf(answer) is below the cell threshold | conf from PLAN §7.4 |
| `STALE` | The value is past its relation's half-life | θ < 0.5 |
| `BEYOND_DEPTH` | Answerable only by inference deeper than the depth cap (3) | Required depth |
| `HYPOTHESIS_ONLY` | The only candidate is in the hypothesis store | Firewall |

The reason code is itself a prediction, and is audited (section 12.1). A system that says "no evidence" when it actually misread the question doesn't know *why* it doesn't know.

### 3.2 Known unknowns: the gap map

A gap is a place where something should be known and isn't. Each gap has:
- an id and a kind;
- where it is: entity, relation, concept or text region;
- the prior that a value exists;
- its value of information (VOI, section 6.3), its estimated cost and its status.

| Kind | How it is found |
|---|---|
| **Schema gap** | For type T and relation R, the expectation `f(T,R)` is the fraction of type-T entities that have R, taken from the store and from Wikidata in the pack. An entity of type T lacking R is a gap with `P(exists) ≈ f(T,R)`. Example: f(country, capital) ≈ 1, so a country with no capital is a near-certain gap |
| **Demand gap** | Questions answered "I don't know", clustered by (relation, type). Each repeat adds demand |
| **Contradiction gap** | Contested slots (section 2.3) |
| **Certainty gap** | Believed items with Low grade, or Known items resting on a single source |
| **Lexical gap** | Words with no lexicon entry, senses the sense classifier rejects as OOD, and frequent unlinked mentions |
| **Structural gap** | Concepts with no parent, relations with no known inverse, disconnected components, rules never tested |
| **Coverage gap** | Corpus regions where many sentences mention two or more linked entities but no pattern parses them |
| **Prediction gap** | Text regions where CyphaLM's bits per byte is far above its running mean. The model is surprised there, so the store probably lacks what explains them |

### 3.3 How much it does not know: estimating the unseen

Cypha estimates the size of what it hasn't seen, not just its location. These are standard statistics for unseen species, applied to facts, entities, senses and phrasings:

| Estimate | Formula | Use |
|---|---|---|
| **Good–Turing missing mass** | `M̂₀ = n₁ / N`, where n₁ is the number of items seen exactly once and N the total observations. **For text, count by page** (S4, 2026-10-28): n₁ = items found on exactly one page, N = page-item incidences (`unit_missing_mass`). A page repeats what it talks about, so the plain form undercounts what is new by 20–50%; the page form is within 4% on held-out pages | Probability that the **next** item read is new: per relation, per domain, per pattern set. A high M̂₀ means reading more will still teach it things |
| **Chao1 richness** | `Ŝ = S_obs + n₁² / (2 n₂)` (if n₂ = 0: `S_obs + n₁(n₁−1)/2`) | Estimated total number of distinct items, so coverage = `S_obs / Ŝ` |
| **Capture–recapture** (Chapman) | `N̂ = (n_A + 1)(n_B + 1)/(m + 1) − 1`, where n_A and n_B are facts found by two independent sources and m by both | Estimated total facts for a relation; coverage = known / N̂. It underestimates when the sources are correlated, so it is reported as a lower bound |
| **Heaps' law** | `V = K·N^β` fit on the lexicon | Expected new words per million tokens |

**The frontier report.** These numbers go into a periodic report: for each (type, relation) cell, coverage, estimated unseen, certainty and demand. This is Cypha's map of the edge of its knowledge.

### 3.4 Unknown unknowns: signals

What it doesn't know it doesn't know can't be listed, but it leaves traces. These are watched as rates, with alarms on change:

- **Out-of-scope inputs:** the OOD rate of incoming text and questions.
- **Unexplained text:** the unparsed-sentence rate per domain.
- **Vocabulary:** unknown-word rate against the Heaps prediction.
- **Surprise:** surprise spikes in CyphaLM.
- **Uncertain routing:** question types the question classifier is unsure of.
- **Unnamed groups:** clusters found by concept formation (G9) that match no concept.
- **Shifts:** sudden changes in source-trust posteriors.

A rising rate means a region of ignorance has been found, and a new gap is opened for it.

### 3.5 The "don't know" message and tag

When Cypha does not know, it says so in two parts. This is not a refusal: it is saying it lacks the knowledge, not that it won't share it (PLAN rule 9).

| Part | Form | Purpose |
|---|---|---|
| **Message** | Plain text: "I don't know." plus one short reason in everyday words ("I have nothing about Hamlet in what I've read."; "My sources disagree about this.") | For the person asking |
| **Tag** | `<|unknown reason=<CODE> entity="<mention>" relation=<rel> gap=<id>|>`, with CODE from section 3.1 | For later. It is parsed into the gap map (demand +1), opens or updates gap `<id>`, and queues it for gap closing (P1) and the curiosity scheduler (section 9). Shown to users as a small label, or hidden |

**Trained, not bolted on.** CyphaLM is trained to produce the message and the tag itself, the way commercial LLMs are trained on their refusal messages. Training text pairs questions it cannot answer with the right message and tag:
- not-in-slice questions;
- questions about held-out facts;
- contested slots.

The reason code in each training tag comes from the epistemic breakdown (section 3.1).

**What is scored:**
- whether CyphaLM says it doesn't know exactly when it should (abstain recall and precision, PLAN rule 9);
- whether the reason in its tag is the real one (reason-code accuracy, section 12.1);
- whether the tagged gaps get closed later (gaps closed per week, P1).

**For now**, the Cypha-classifier gate (section 8a) may trigger the same message and tag on CyphaLM's behalf.

---

## 4. Hypotheses: the speculative store

### 4.1 The hypothesis record

| Field | Content |
|---|---|
| `id`, `statement` | A triple, a rule, a concept, a numeric law, a causal link or a pattern, in store form plus a sentence |
| `generator` | Which of G1–G20 made it, and its version |
| `derivation` | The chain: premises (with their ECS), rule or analogy used, parameters |
| `prior` | P₀ from the generator's reliability posterior and its own support statistics (section 6.1) |
| `scores` | Plausibility, novelty, impact, testability, cost, VOI (section 6) |
| `tests` | Pre-registered tests, each with its predicted outcome and its disconfirmation search (section 7) |
| `posterior` | Updated after each test (Bayes factor or e-value) |
| `state` | proposed → scored → queued → under test → supported / refuted / parked / irreducible → promoted or retired |
| `taint` | Always hypothetical until promotion |
| `history` | Every update with time and cause |

### 4.2 Lifecycle

```text
 proposed ─► scored ─► queued ─► under test ─┬─► supported ──(ECS admits on independent evidence)──► PROMOTED to knowledge
                │                              ├─► refuted ───► kept as a negative result; blocks regeneration
                └─► below floor: "wild" tier    ├─► parked ────► no feasible test now; re-scored when the store changes
                                               └─► irreducible ─► K tests left it unchanged; stop spending on it
```

### 4.3 Tiers

| Tier | Plausibility | Budget share | Purpose |
|---|---|---|---|
| **Conjecture** | P₀ ≥ 0.3 | 70% | Likely truths not yet confirmed |
| **Speculation** | 0.05 ≤ P₀ < 0.3 | 25% | Real uncertainty, high value if true |
| **Wild** | P₀ < 0.05 | 5% | Exploration: distant recombinations and odd analogies. Kept small and cheap |

The shares are starting values. The scheduler moves them by the confirmed-value rate of each tier (section 9).

### 4.4 Possible worlds: the hypothetical sandbox

For "what if" reasoning and for testing how robust a belief is, Cypha builds a **world**: a copy-on-write overlay of the store plus a set of assumptions, each either added or retracted.

- Rules forward-chain inside the world up to depth 3. The result lists:
  - new consequences;
  - consequences that disappeared;
  - contradictions.
- **A contradiction refutes the assumption set** (reductio). The minimal conflicting subsets are reported.
- Worlds are never written back. They are deleted after the query or test, and their outputs are tainted.

**Uses**
1. Odin's explicit what-if questions, answered as labelled hypotheticals.
2. Consequence tests for hypotheses (section 7.2).
3. Fragility analysis: retract a source and see what flips (section 7.6).

---

## 5. Where hypotheses come from: 20 generators

Each generator is a process with a reliability posterior: a Beta over the share of its hypotheses later confirmed. Generators that keep failing are retired (section 9).

| # | Generator | Method | Output | Main failure mode, and the guard |
|---|---|---|---|---|
| G1 | **Deep deduction** | Forward-chain beyond the answer-mode depth cap, or through Believed premises | Facts that would follow if longer chains held | Error compounding. Prior = product of premise P and rule reliabilities |
| G2 | **Rule induction** | AMIE-style Horn-rule mining over the store: support, head coverage, standard confidence, and PCA confidence (confidence counted only where the subject has some known value) | Rules such as `born_in(X,C) ∧ city_in(C,K) ⇒ nationality(X,K)`, and the facts they predict | Spurious rules from small support. Minimum support 10, PCA confidence ≥ 0.5, BH over the rule batch |
| G3 | **Abduction** | For an unexplained observation (a fact, a text claim, an anomaly), find the minimal set of missing facts that would make it derivable through known rules. Score by description length (fewer, simpler assumptions win) | Explanations | Many explanations fit. All are kept, ranked by MDL, and each is tested separately |
| G4 | **Analogy (structural)** | Structure mapping: find a mapping between two subgraphs that preserves relations (Gentner). Carry over relations present in the source and missing in the target | "Algeria is to Algiers as Albania is to ?" generalised to whole systems | Surface similarity mistaken for structure. Require ≥ 3 aligned relations |
| G5 | **Analogy (vector)** | Offsets in the C-track vectors: `b − a + c`; proportional analogies on relation vectors | Candidate pairs for a relation | Offsets measured weak in static spaces. Low prior until calibrated per relation |
| G6 | **Link prediction (neighbourhood)** | Common neighbours, Jaccard, Adamic–Adar (`Σ 1/log|Γ(z)|` over shared neighbours z) between entities | Likely missing links | Hubs dominate. Degree-normalise |
| G7 | **Link prediction (paths)** | Path Ranking: random-walk probabilities along relation paths as features, combined by the Cypha classifier | Missing facts with path explanations | Paths through generic hubs. Path-length and hub penalties |
| G8 | **Link prediction (factorisation)** | Bilinear factorisation of the relation tensor (RESCAL-style, by alternating least squares; linear algebra, no network) | Scores for every (s, r, o) | Opaque. Used only as a candidate source; explanations come from G7 |
| G9 | **Concept formation** | Cluster entities by relation profile; formal concept analysis over entity × attribute; take lattice nodes with high support that match no existing class. Name them by the shared hypernym, or ask the teacher (G16) | New classes ("landlocked countries", "Baroque composers") | Trivial concepts. Require support ≥ 10 and an information gain for prediction |
| G10 | **Pattern hypotheses** | Bootstrapping: sentences containing both sides of a known fact yield candidate phrasings | New patterns, which yield new facts | Semantic drift. Coupled constraints across relations, as NELL learned |
| G11 | **Open relations** | From sentences with two linked entities that no pattern parses, take the connecting phrase (open IE), cluster phrases, and propose a new relation | New relation types | Noise. Need ≥ 5 entity pairs per phrase cluster |
| G12 | **Literature-based discovery** | Swanson's ABC: A co-occurs with B in one body of text and B with C in another, but A and C are never together. Score `Σ_B w(A,B)·w(B,C)` with w = PMI, excluding pairs that already co-occur. Open form: A only. Closed form: A and C, find the B links | Hidden connections across domains | Generic B terms. IDF-weight B and remove hub terms |
| G13 | **Anomaly explanation** | Values far from their type's distribution (a city larger than its country, a death before a birth), or high-surprise text. Hypotheses: an error, a missing qualifier, a different entity, a genuinely unusual fact | Corrections or "interesting" flags | Real outliers flagged as errors. Both readings are tested |
| G14 | **Contradiction resolution** | For a contested slot, propose hidden variables: time (both true at different dates), sense (two meanings), identity (two entities sharing a name), qualifier ("judicial capital"), source error | Reconciling hypotheses | Over-explaining. MDL plus a test of each variable |
| G15 | **Numeric law discovery** | BACON-style search over numeric attributes of a type. If X rises with Y, try X/Y; if X falls as Y rises, try X·Y. Keep terms that are nearly constant across entities (coefficient of variation < 0.1), check units, and penalise formula length by MDL | Laws and derived quantities (density = mass / volume) | Fitting noise. Test on held-out entities; BH across candidate formulas |
| G16 | **Teacher queries** | Ask a GGUF teacher through `scripts/gguf_mine.py` ([GGUF_MINING.md](GGUF_MINING.md)) | Facts, relations, labels | Hallucination. Prior from the teacher's calibration per relation, Beta(1, 3) by default |
| G17 | **Causal hypotheses** | PC algorithm over attributes of a type (conditional-independence tests), plus "X causes Y" patterns in text. Observational only, and always labelled so | Causal links, as hypotheses | Confounding. Never promoted to Known from observational data alone; graded Believed at most |
| G18 | **Trends and extrapolation** | Time series of a fact's values, run through Cypha's forecasting module | Forecasts, labelled, with intervals | Regime change. Pre-registered and scored when the data arrives |
| G19 | **Generalise and specialise** | Inductive-logic-style refinement of rules: add or drop body atoms, generalise constants to variables, specialise to a subtype where the rule holds better | Better rules | Search blow-up. Beam width and MDL |
| G20 | **Random recombination** | Pair distant concepts from different clusters with a random relation, and keep pairs whose statistical support is better than chance | The wild tier | Mostly junk. Fixed small budget; only the cheapest tests |

---

## 6. Which hypotheses to chase: scoring

### 6.1 Plausibility

`P₀(h) = generator reliability × support term × constraint term`

- **Generator reliability:** the posterior mean of the generator's Beta, per hypothesis kind.
- **Support term:** the generator's own statistic, calibrated. Examples: PCA confidence for G2; the path classifier probability for G7; Σ w(A,B)·w(B,C) mapped through an isotonic fit for G12.
- **Constraint term:** 0 if a hard constraint is violated (type, functional slot already Known with another value), 0.5 if a soft one is, 1 otherwise.

### 6.2 Novelty, impact and interestingness

| Score | Definition |
|---|---|
| **Novelty** | Surprisal under the current store: `−log₂ P_store(h)`, from the link predictors (G6–G8) or base rates. Something the store already strongly predicts is not a discovery, so depth-1 deductions score ≈ 0 |
| **Impact** | Number of other items that would change if h were true: new consequences in a sandbox world (section 4.4), gaps it fills, contradictions it resolves, and the demand on those gaps |
| **Compression gain** | Bits saved in describing the store plus the corpus if h holds: `ΔL = L(store, corpus) − L(store ∪ {h}, corpus)`. A rule that explains 200 facts compresses them. This is Schmidhuber's "interestingness is compression progress", measured with Cypha's own compressors |
| **Bridging** | Whether h connects two components or distant clusters (the change in average shortest-path length, or betweenness) |
| **Interestingness** | `I = z(novelty) + z(impact) + z(ΔL) + z(bridging)`, with z-scores against the current hypothesis pool |

### 6.3 Value of information

For a binary hypothesis with probability p and a test with sensitivity s and specificity c:
- `P(pos) = p·s + (1 − p)(1 − c)`
- `p⁺ = p·s / P(pos)` and `p⁻ = p(1 − s) / (1 − P(pos))`
- `EIG = H(p) − [P(pos)·H(p⁺) + (1 − P(pos))·H(p⁻)]` bits, where `H(p) = −p·log₂p − (1−p)·log₂(1−p)`

The test's s and c are estimated per test kind from its history on known facts (section 7.1).

**Priority** = `EIG × (1 + Impact) × (1 + I⁺) / Cost`, where `I⁺ = max(0, I)` and Cost is the expected compute and data cost (or human time) of the test.

The scheduler (section 9) picks by priority within each tier, plus an exploration bonus.

---

## 7. Testing hypotheses

### 7.1 Test kinds

Cheapest and most automatic come first. Each kind has measured sensitivity and specificity, from running it on known-true and known-false facts.

| Test | What it does | Good for |
|---|---|---|
| T1 **Consistency** | Type, functional and numeric constraints against Known items; sandbox forward-chaining for contradictions | Killing bad hypotheses fast |
| T2 **Consequence (falsification)** | Derive what else must be true if h holds and check those consequences against the store and the corpus. One confident failed consequence counts heavily against h | Rules, laws, abduction |
| T3 **Held-out prediction** | For rules and laws: hide part of the known cases, predict them and score precision (with a Wilson bound) | G2, G15, G19 |
| T4 **Corpus search** | Look for direct statements with pattern matching and the fuzzy index (B7). The search covers **for and against**: the disconfirmation search is mandatory | Facts |
| T5 **Cross-source** | Check other pack sources (Wikidata, WordNet, ConceptNet) not used by the generator | Facts, taxonomy |
| T6 **Compression** | Does adding h reduce code length on held-out text and facts? (MDL) | Rules, concepts, patterns |
| T7 **Teacher** | Query a calibrated GGUF teacher, unless the teacher produced h | Facts, labels |
| T8 **Prospective** | Register a dated prediction; score it when new data arrives | Trends (G18), Track A |
| T9 **Temporal holdout** | The text is a 2006 snapshot; the pack's Wikidata is current. A hypothesis about facts true but unstated in 2006 can be checked against Wikidata as "future" evidence | Every fact generator; the strongest discovery benchmark (section 12.2) |
| T10 **Ask Odin** | Queue for a human, used only for high-VOI hypotheses no automatic test can settle | Anything; cost is human time |

### 7.2 The scientific loop for one hypothesis

1. **Pre-register.** Before any test runs, write to the experiment DB (`native/sql/experiment_schema.sql`):
   - which tests will run;
   - the predicted outcome of each;
   - what would count as refutation.
2. **Design both directions.** Every test plan includes a search for disconfirming evidence, not just confirming.
3. **Run tests cheapest-first.** Stop early when the posterior crosses a decision boundary.
4. **Update the hypothesis posterior.**
   - Each test result has a Bayes factor `BF = P(result | h) / P(result | ¬h)` from that test's sensitivity and specificity.
   - The posterior odds are `O₁ = O₀ × Π BF`.
   - For streams of evidence, the e-process in `lci.hpp` gives anytime-valid decisions.
5. **Decide.**
   - **Supported:** the posterior is above 0.9 and at least one piece of evidence is independent of the generator. It then goes to ECS.
   - **Refuted:** the posterior is below 0.05.
   - **Irreducible:** K = 5 tests have moved the posterior by less than 0.05 in total.
   - Otherwise it waits in the queue.
6. **Log everything**, including refutations. Negative results stop the same hypothesis being regenerated for 10 cycles unless new evidence arrives.

### 7.3 Many hypotheses at once

- **Batch control.** Each batch of decisions from one generator is controlled by Benjamini–Hochberg at q = 0.1, already in `lci.hpp`.
- **Promotions** are controlled by the weekly α budget and e-values (PLAN §8.4).
- **Watching for false discoveries.** The audit sample of promoted discoveries gives a running false-discovery rate. If its Wilson upper bound passes 0.15, promotion pauses.

### 7.4 Promotion into knowledge

A supported hypothesis enters ECS as a candidate fact, with these evidence items:
- the independent test evidence, at full mass;
- the generator's support, as a **prior only**: it sets the base rate `a`, not r.

ECS then grades it like anything else. Promotion records the generator credit, which updates the generator's reliability Beta. That loop is how discovery processes learn which of them to trust.

### 7.5 Refutation

A refuted hypothesis:
- is kept, with its tests;
- counts against its generator's reliability;
- counts against any rule or analogy it came from (that rule's Beta gets a miss);
- triggers G14 if it contradicted something Known, since a confident refutation of a confident derivation means some premise is wrong.

### 7.6 Fragility: how robust is what it knows?

For each Known fact, run a sandbox world (section 4.4) with its top source retracted, and with its weakest premise retracted.
- **Fragility** = the fraction of those worlds in which the fact drops below Known.
- Facts with fragility 1 are single points of failure. They become certainty gaps, and P1 looks for second sources.
- The same analysis by source gives each source's **blast radius**: how many beliefs depend on it alone.

---

## 8. The discovery processes

These are the loops Cypha runs. Each is a named process with a trigger, steps, statistics, outputs, a kill rule and its own metric. They share the stores, the scorer, the tests and the scheduler.

### P1 Gap closing (known unknowns)
- **Trigger:** gap map entries by VOI.
- **Steps:** cheapest source first, the **asking ladder**:
  1. re-read the corpus with current patterns;
  2. inference within Known items;
  3. other pack sources;
  4. the teacher (G16);
  5. Odin (T10).
- **Statistics:** VOI per gap, success rate per ladder rung (a Beta), cost per rung.
- **Output:** filled slots through ECS, or the gap marked unresolvable-now.
- **Kill:** a gap tried at every rung without change is parked for 30 days.
- **Metric:** gaps closed per week, and coverage change in the frontier report.

### P2 Conjecture (generate and test)
- **Trigger:** each generator's scheduled budget.
- **Steps:** generate, score (section 6), queue, test (section 7), promote or refute.
- **Metric:** confirmed hypotheses per unit cost, per generator; false-discovery rate of promotions.

### P3 Anomaly
- **Trigger:**
  - type-distribution outliers;
  - CyphaLM surprise spikes;
  - classifier OOD clusters;
  - unknown-unknown rate alarms (section 3.4).
- **Steps:**
  1. describe the anomaly (what is unusual and by how much, as a z-score or bits);
  2. abduction (G3) and G13 propose explanations;
  3. test each;
  4. a confirmed error becomes a correction, and a confirmed real oddity is flagged "interesting".
- **Metric:** anomalies explained, and the share of them that were errors.

### P4 Contradiction
- **Trigger:** a Contested slot.
- **Steps:**
  1. G14 proposes hidden variables (time, sense, identity, qualifier, source error);
  2. test each;
  3. split or qualify the fact, or down-weight the bad source.
- **Metric:** contested slots resolved, and source trust updates.

### P5 Concept formation
- **Trigger:** periodic, and unnamed-cluster alarms.
- **Steps:**
  1. G9 finds candidate classes;
  2. name them;
  3. test by information gain: does knowing the class predict members' relations better? Use held-out relations (T3).
  4. If it does, add the class to the hierarchy as Believed.
- **Metric:** new classes kept; prediction gain from each.

### P6 Rule induction
- **Trigger:** periodic, over each relation.
- **Steps:** G2 mines rules, G19 refines them, T3 runs held-out prediction, BH controls the batch, and a kept rule gets a Beta reliability.
- **Metric:** rules kept, their held-out precision, and facts they newly predict that were later confirmed.

### P7 Analogy
- **Trigger:** gaps in a target entity that has a well-covered structural analogue.
- **Steps:** G4 and G5 map the structure, transfer missing relations as hypotheses, and test them.
- **Metric:** transfer precision per analogy kind.

### P8 Literature-based discovery
- **Trigger:** on request (seed concept A), and a periodic sweep over high-demand concepts.
- **Steps:**
  1. G12 builds open ABC candidates from co-occurrence statistics;
  2. remove known links;
  3. rank;
  4. T9 temporal holdout: did the A–C link appear in later sources?
- **Metric:** precision@k against later sources, the standard evaluation for this method.

### P9 Law discovery
- **Trigger:** types with at least 30 entities holding at least 3 numeric attributes.
- **Steps:**
  1. G15 searches formulas;
  2. check units;
  3. T3 tests on held-out entities;
  4. BH across formulas;
  5. a kept law becomes a rule computing derived values, with an error model.
- **Metric:** laws kept and their held-out error.

### P10 Open relations
- **Trigger:** the coverage-gap rate in a domain is above its running mean.
- **Steps:**
  1. G11 clusters unparsed connecting phrases;
  2. propose a relation;
  3. ask for its name and type signature (from the teacher or Odin);
  4. seed the patterns;
  5. hand it to F2 pattern learning.
- **Metric:** new relations kept, and the drop in the unparsed rate.

### P11 Hypothetical reasoning
- **Trigger:** what-if questions; fragility analysis; consequence tests.
- **Steps:** build a world (section 4.4), forward-chain, report consequences and contradictions, delete the world.
- **Output:** a labelled hypothetical card (section 10).
- **Metric:** correctness of consequences on test cases with known answers.

### P12 Curiosity scheduler (the meta-process)
Allocates effort across P1–P11, the generators and the tiers. See section 9.

### P13 Self-audit
- **Trigger:** weekly.
- **Steps:**
  1. run the introspection queries (section 2.5);
  2. audit 100 newly admitted facts (PLAN §7.5);
  3. audit 100 abstentions for reason-code accuracy (section 12.1);
  4. recompute the competence table;
  5. recompute calibration per cell;
  6. open gaps for what it finds.
- **Metric:** audit precision with a Wilson bound; calibration drift.

### P14 Research journal
- **Trigger:** weekly, and at each Track A checkpoint.
- **Steps:** write a dated report into `docs/reports/discovery/`. It covers:
  - what was learned, with a per-generator league table;
  - what was refuted;
  - what is contested;
  - the frontier report (section 3.3);
  - the biggest open gaps by VOI;
  - wild-tier hypotheses that survived their first test;
  - questions for Odin.
- **Metric:** Odin's rating of each journal (1–5) feeds R5 in PLAN §3.4.

### P15 Teacher mining
- **Trigger:** gaps whose cheaper rungs failed, and new-concept naming.
- **Steps:** targeted `gguf_mine.py` probes; output enters as hypotheses with the teacher's calibrated prior; independent tests follow.
- **Metric:** teacher-sourced hypotheses confirmed per query, per relation.

### P16 Ask Odin
- **Trigger:** high-VOI hypotheses no automatic test can settle; new relation and concept names; disputed audits.
- **Steps:** a queue with the hypothesis card, the evidence for and against, and a yes / no / unsure / don't care answer. Answers count as evidence from a source with its own trust Beta.
- **Budget:** at most N questions per week. Odin sets N.
- **Metric:** answers per week; how much each answer moved the store.

---

## 8a. Standing investigation: who decides to abstain

**Status.** Decided by Odin Loch, 2026-10-27. For now, a Cypha-classifier gate that decides whether to abstain on CyphaLM's own answer counts as CyphaLM doing the task (PLAN rule 10). That is a starting position, not a settled one.

**Later, Cypha takes this up itself** as a discovery project and records what it finds, using the processes above (P2 conjecture, P3 anomaly, P6 rule induction, P13 self-audit, P14 journal). Questions it should work through:

1. **What does the gate add** over CyphaLM's own probabilities? Measure the wrong-answer rate at fixed coverage both ways. **Measured 2026-10-28** (TOOL_PROFILE P3, [report](reports/TOOL_PROFILE_P2_GRID.md)): with at most 7 right answers in 130, the classifier gate kept more wrong answers than a single-feature threshold (18.0 against 5 over ten configurations) and was deleted; the question stays open for when CyphaLM has more right answers to learn from.
2. **Can CyphaLM learn to judge its own answers?** For example, by training on its own right and wrong answers as text, or by an abstention expert inside its mixture.
3. **Which signals predict a wrong answer?** Entropy, match length, margin, the reason codes. Do they generalise across question types?
4. **Is there a better arrangement** than "generate then gate"? For example, CyphaLM producing the "I don't know" message and tag of section 3.5 by itself, as a learned output, or confidence carried through the mixture.

Each answer is a hypothesis with pre-registered tests (section 7), and changes are promoted only through LCI. The findings go to Odin in the research journal. He decides whether the gate stays, moves into CyphaLM, or goes.

## 9. Curiosity: allocating the effort

The scheduler treats each (process, generator, tier) as an arm of a multi-armed bandit.

- **Reward:** the value confirmed per unit cost = Σ over promotions and resolved gaps of (impact × certainty gained) / cost.
- **Selection:** Thompson sampling over each arm's reward posterior.
- **Learning-progress bonus** (Oudeyer et al.):
  - arms whose recent tests are steadily reducing uncertainty get a bonus;
  - arms stuck at the same uncertainty lose it;
  - this avoids the "noisy TV" trap of chasing the unlearnable.
- **Demand bonus:** gaps that users keep hitting.
- **Floors:** every process keeps at least 2% of budget, so none starves. The wild tier keeps its 5% (section 4.3).
- **Retirement:** a generator whose reliability upper bound falls below 0.05 after at least 200 hypotheses is retired, and listed as retired in the journal.

Arms are promoted or demoted through LCI like any other change (PLAN §8). A new scheduling policy has to beat the old one on confirmed value per cost, without raising the false-discovery rate.

---

## 10. Talking about it: cards and query forms

Every output in either mode is one of these cards, with the data fields kept and the wording generated from them.

**Answer card**
```text
Q: What is the capital of Algeria?
Algiers.                         [Known · P 0.97 (0.92–0.99) · High]
Because: infobox "capital" on the Algeria page; 3 sentences of the form "Algiers, the capital of …"
Independent sources: 2 · Contradictions: none · Fragility: 0.0
Weakest link: entity link "Algeria" (0.99)
Would change my mind: a current source naming another capital
```

**"I don't know" card** (message + tag; section 3.5)
```text
Q: Who wrote Hamlet?
I don't know. I have nothing about Hamlet in what I've read.
<|unknown reason=NO_EVIDENCE entity="Hamlet" relation=author gap=G-0412|>
Why: I have no source about "Hamlet"; it is outside the text I have read.
What I can say: "author" is a well-covered relation for books here (precision 0.90).
What would let me answer: any source about Hamlet.
Logged as gap G-0412 (demand +1).
```

**Hypothesis card** (discovery mode only)
```text
SPECULATION — not knowledge.
Atlas Shrugged was probably first published in English.
From: G2 rule induction: author(B, P) ∧ nationality(P, United States) ⇒ original_language(B, English)
      (support 41, PCA confidence 0.88), applied to author(Atlas Shrugged, Ayn Rand)
Plausibility 0.71 · Novelty 1.9 bits · Impact: fills a schema gap (demand 0)
Tests planned: T4 corpus search (for and against), T5 Wikidata, T1 consistency
Status: queued
```

**Hypothetical card** (what-if)
```text
HYPOTHETICAL — assuming: Berlin is not the capital of Germany.
Consequences (depth ≤ 3): 4 derived facts drop (seat of the Bundestag …); 1 contradiction
with a Known fact: the Bundestag sits in Berlin → the assumption conflicts with Known facts.
Nothing here is stored.
```

**Self-knowledge queries**

| Ask | Returns |
|---|---|
| "What do you know about X?" | The inventory of Known and Believed items for X, grouped by relation, with grades |
| "What don't you know about X?" | Schema gaps for X's type, contested items, single-source items |
| "How sure are you about Y?" | The evidence card for Y |
| "How good are you at Z?" | The competence-table cell for Z (section 2.4) |
| "Where are the edges of your knowledge?" | The frontier report |
| "Speculate about X." | The top hypothesis cards for X, each labelled |
| "What if A?" | A hypothetical card |

---

## 11. Honesty invariants

These are written as CTests and checked on every build:

| # | Invariant |
|---|---|
| I1 | Answer mode never returns an item with the `hypothetical` taint, directly or through derivation |
| I2 | Every answer has at least one evidence item that is not a hypothesis and not from a generator |
| I3 | Whenever conf(answer) is below the cell threshold, the output is an "I don't know" card with a reason code |
| I4 | A hypothesis is promoted only with evidence independent of its generator |
| I5 | Every test is pre-registered before it runs, and its record cannot be edited after the result |
| I6 | Every test plan includes a disconfirmation search |
| I7 | Refuted hypotheses are kept and block regeneration for the cooling period |
| I8 | Discovery output is always labelled (SPECULATION / HYPOTHETICAL) |
| I9 | Sandbox worlds are never written back |
| I10 | Contested items are never returned as answers |

---

## 12. Measuring all of this

### 12.1 Knowing what it knows, and what it doesn't

| Metric | Definition | Target |
|---|---|---|
| Wrong-answer rate | Wrong ÷ all questions, with a Wilson upper bound | ≤ 2%; the number to drive down first (PLAN rule 9) |
| Abstain recall on unanswerables | "I don't know" ÷ questions with no answer in the store (e.g. the not-in-slice half of T-QA-wiki) | ≥ 95% |
| Selective accuracy | Right ÷ answered, with a Wilson lower bound | ≥ 95% |
| Coverage | Answered ÷ answerable | Rises over time without breaking the two above |
| Calibration | ECE and Brier of answer confidence, per competence cell | ECE ≤ 0.05 |
| Discrimination | AUROC of confidence for right against wrong answers | ≥ 0.85 |
| Risk–coverage | Area under the risk–coverage curve (AURC) | Falls over time |
| Overclaim / underclaim | Wrong answers at conf ≥ 0.9 / abstentions where a Known correct answer existed | Both reported |
| Reason-code accuracy | Hand-check 100 "I don't know" cards a week: is the stated reason the real one? | ≥ 90% |
| Missing-mass validity | Hold out one source and predict how many new facts it adds (Good–Turing, capture–recapture), then compare | Within ±25% |

### 12.2 Discovery

| Benchmark | How | Score |
|---|---|---|
| **Rediscovery: facts** | Hide 10% of Known facts at random (and separately, the 10% least connected); run P1, P2 and P7 | Hits@1 / @10 and mean reciprocal rank for link predictors; precision of promotions |
| **Rediscovery: a relation** | Remove a whole relation that has an inverse (capital ↔ capital of) | Does P6 recover the inverse rule and refill it? |
| **Rediscovery: a class** | Remove a class (landlocked countries) | Does P5 find a cluster matching it? (adjusted Rand index) |
| **Rediscovery: laws** | Synthetic attributes with planted laws, then real ones (population density from population and area) | Law found; error on held-out entities |
| **Temporal holdout** | 2006 text against current Wikidata (T9) | Precision of hypotheses about facts true-but-unstated in 2006; recall over a sample of such facts |
| **ABC replication** | Literature-based discovery on time-sliced text where available | Precision@k of A–C links that appear later |
| **Prospective** | Pre-registered predictions (Track A P-A1 to P-A5, trend forecasts) | Hit rate when the data arrives |
| **False-discovery rate** | Audit of promotions | Wilson upper bound ≤ 0.15 |
| **Generator league** | Per generator: confirmations ÷ hypotheses, value per cost, time to confirmation | Ranking; retirements |
| **Journal usefulness** | Odin's weekly 1–5 rating | Trend |

---

## 13. Track S: work items and order

| ID | Work | Gate |
|---|---|---|
| S1 | Reason codes and evidence cards on CyphaLM's answers (every "I don't know" says why; every answer shows its evidence). The non-CyphaLM Q/A tools were removed under PLAN rule 10 | Reason-code accuracy ≥ 90% on 100 hand-checked; I3 CTest. **Run 2026-10-28:** 82% dev, 84.5% test; I3 check passes; `qa --cards` gives evidence cards. The 90% gate is missed and Odin closed S1 there: codes describe what CyphaLM has read, and where the data isn't there, saying so is right ([report](reports/S1_REASON_CODES_AND_EVIDENCE.md)) |
| S2 | Epistemic states on top of `ecs.hpp`: Known / Believed / Contested / Ignorant / Vacuous, the DC conflict measure, and the two kinds of uncertainty | Known-answer CTests |
| S3 | Gap map: schema expectations f(T,R), demand log, contested and fragile lists | Gaps listed with VOI on the F1 fact store |
| S4 | Unseen estimates (Good–Turing, Chao1, capture–recapture, Heaps) and the frontier report | Missing-mass validity within ±25% on a held-out source  **Run 2026-10-28:** gate met; Good–Turing with the page as the unit is within 4% on held-out pages (n₁/N undercounts by 20–50%); `pattern_learn --frontier` writes the frontier report ([report](reports/S4_UNSEEN_AND_FRONTIER.md)) |
| S5 | Hypothesis store, taint, lifecycle, firewall | I1, I4, I7, I8 CTests |
| S6 | Scoring: plausibility, novelty, impact, compression gain, EIG and priority | Ranks known-true above known-false hypotheses (AUROC ≥ 0.75) on a planted set  **Run 2026-10-28:** P0 0.738 [0.717, 0.758] on a fresh planted set, gate missed; the terms are in `hypothesis.hpp` ([report](reports/S6_HYPOTHESIS_SCORING.md)) |
| S7 | Tests T1–T6 with measured sensitivity and specificity, pre-registration in the experiment DB, Bayes-factor and e-value updating | I5, I6 CTests; s and c measured  **Run 2026-10-28:** I5 and I6 CTests pass; s and c measured on test for T1, T3, T4, T5 and T6; T2 cut (no consequence rule fired); the loop supports 56 of 56 correctly ([report](reports/S7_HYPOTHESIS_TESTS.md)) |
| S8 | Generators batch 1: G2 rule induction, G6 and G7 link prediction, G14 contradiction resolution, G16 teacher | Rediscovery Hits@10 reported; promotion precision ≥ 0.9 |
| S9 | Possible-worlds sandbox and the hypothetical card; fragility analysis | I9 CTest; consequence tests pass |
| S10 | Curiosity scheduler (Thompson sampling, learning progress, floors, retirement) | Beats a uniform scheduler on confirmed value per cost, via LCI |
| S11 | Generators batch 2: G9 concept formation, G11 open relations, G12 ABC, G13 anomalies, G3 abduction, G4 and G5 analogy | Rediscovery of a class; ABC precision@k |
| S12 | Generators batch 3: G15 laws, G17 causal (labelled), G18 trends via the forecasting module, G19 refinement, G1 deep deduction, G20 wild tier | Planted-law rediscovery; prospective log |
| S13 | Self-audit (P13), research journal (P14), Ask-Odin queue (P16) through REST and the Qt shell | First journal published; Odin's rating |
| S14 | Self-knowledge queries (section 10 table) | Correct inventories on test entities |
| S15 | Temporal-holdout benchmark (2006 text against current Wikidata) | Benchmark built and frozen in `bench/eval/`  **Built 2026-10-28:** frozen in `bench/eval/temporal_holdout_{dev,test}.tsv`, with its Wikidata cache ([report](reports/S15_TEMPORAL_HOLDOUT.md)) |

**Order.**
1. S1 → S2 → S5 → S3. Honesty first: reason codes, states and the firewall, on the stores that exist now.
   - S1 done 2026-10-28 at 82% (Odin closed the 90% gate); S5 done 2026-10-28 (I1, I4, I7, I8 CTests); S3 done 2026-10-28 (gaps with VOI on the v2 fact store). [Report](reports/S3_S5_GAP_MAP_AND_HYPOTHESES.md).
2. S4 → S6 → S7. Measurement of the unknown, then the hypothesis machinery.
   - S4 done 2026-10-28 (gate met); S6 done 2026-10-28 (gate missed at 0.738; its next change needs S15 or another fresh source); S7 done 2026-10-28 (gates met); S15 built 2026-10-28, ahead of S8.
3. S8 → S9 → S10. The first generators, worlds and the scheduler.
   - S8 started 2026-10-29 (G2, G6, G7, G14; G16 runs on Odin's PC).
4. S11 → S12 → S13 → S14, with S15 built alongside S8 so discovery has a fair test from the start.

**Dependencies on other tracks**
- **D (data pack):** Wikidata for schema expectations, T5 and T9.
- **F (facts and patterns):** the store the generators work on.
- **C (vectors):** G5 and similarity.
- **E (tagger, parser):** G11 and question parsing.
- **H (teacher mining):** G16 and T7.
- **Forecasting module:** G18.
- **Track A:** surprise signals and prospective tests.

---

## 14. References

**Uncertainty, abstention and self-knowledge**
- Jøsang (2016). *Subjective Logic*. Springer. (Opinions, fusion, degree of conflict.)
- Chow (1970). On optimum recognition error and reject tradeoff. *IEEE Trans. Information Theory*.
- Geifman & El-Yaniv (2017). Selective classification for deep neural networks. *NeurIPS*. (Risk–coverage.)
- Hendrycks & Gimpel (2017). A baseline for detecting misclassified and out-of-distribution examples. *ICLR*.
- Kadavath et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.

**Estimating the unseen**
- Good (1953). The population frequencies of species and the estimation of population parameters. *Biometrika*.
- Chao (1984). Nonparametric estimation of the number of classes in a population. *Scandinavian Journal of Statistics*.
- Chapman (1951). Some properties of the hypergeometric distribution with applications to zoological sample censuses. *U. California Publications in Statistics*.

**Discovery systems and scientific method**
- Popper (1959). *The Logic of Scientific Discovery*.
- Peirce (1903). Lectures on pragmatism (abduction).
- Langley, Simon, Bradshaw & Zytkow (1987). *Scientific Discovery: Computational Explorations of the Creative Processes*. MIT Press. (BACON.)
- Lenat (1976). AM: an artificial intelligence approach to discovery in mathematics. Stanford PhD thesis.
- Lenat & Brown (1984). Why AM and EURISKO appear to work. *Artificial Intelligence*.
- Swanson (1986). Fish oil, Raynaud's syndrome, and undiscovered public knowledge. *Perspectives in Biology and Medicine*.
- Rissanen (1978). Modeling by shortest data description. *Automatica*.

**Relations, rules and concepts**
- Galárraga, Teflioudi, Hose & Suchanek (2013). AMIE: association rule mining under incomplete evidence in ontological knowledge bases. *WWW*.
- Lao & Cohen (2010). Relational retrieval using a combination of path-constrained random walks. *Machine Learning*.
- Nickel, Tresp & Kriegel (2011). A three-way model for collective learning on multi-relational data. *ICML*. (RESCAL.)
- Liben-Nowell & Kleinberg (2007). The link-prediction problem for social networks. *JASIST*.
- Adamic & Adar (2003). Friends and neighbors on the Web. *Social Networks*.
- Muggleton (1991). Inductive logic programming. *New Generation Computing*.
- Ganter & Wille (1999). *Formal Concept Analysis*. Springer.
- Banko et al. (2007). Open information extraction from the Web. *IJCAI*.
- Falkenhainer, Forbus & Gentner (1989). The structure-mapping engine. *Artificial Intelligence*.
- Carlson et al. (2010). Toward an architecture for never-ending language learning. *AAAI*. (NELL.)

**Causality**
- Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search*. MIT Press. (PC algorithm.)
- Pearl (2009). *Causality*. Cambridge University Press.

**Curiosity and allocation**
- Oudeyer, Kaplan & Hafner (2007). Intrinsic motivation systems for autonomous mental development. *IEEE Trans. Evolutionary Computation*.
- Schmidhuber (2010). Formal theory of creativity, fun, and intrinsic motivation. *IEEE Trans. Autonomous Mental Development*.
- Thompson (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*.
- Settles (2009). Active learning literature survey. U. Wisconsin TR 1648.
