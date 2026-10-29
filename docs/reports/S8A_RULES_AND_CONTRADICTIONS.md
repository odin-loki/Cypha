# S8a: rule induction (G2) and contradiction resolution (G14)

**Odin Loch** · 2026-10-29 · DISCOVERY §5 (G2, G14), §7, §12.2, §13 S8 · follows [S7_HYPOTHESIS_TESTS.md](S7_HYPOTHESIS_TESTS.md) and [S15_TEMPORAL_HOLDOUT.md](S15_TEMPORAL_HOLDOUT.md)

(The stores, the rule language, the reads, the decision rule and the predictions below were written before any generator ran. Results and decisions are added under them.)

**Question.** Cypha's first generators. G2 mines Horn rules over the store and proposes the facts they predict. G14 looks at contested slots and proposes what hidden variable reconciles them. Do the rules rediscover hidden facts, and does S7's loop promote only true ones?

**Gate (DISCOVERY §13 S8).** Rediscovery Hits@10 reported; promotion precision ≥ 0.9. G16 (teacher) is not part of this step: there is no Qwen inference in the container (PLAN ledger 54).

**Rule 10.** Generators write hypotheses into Cypha's hypothesis store. They never answer a question for CyphaLM, and nothing here reaches `qa` or `chat`. A promoted fact could become CyphaLM training text later only through the documented fact-shard path with its leak checks (D4, S15's `check`).

**Counted before registering** (sizes only, no generator output): the seed store has 22,783 distinct triples over 3,960 subjects, and 401 of its objects are also subjects. The 2006-text store (below) has 8,319 triples over 2,109 subjects, 145 objects that are subjects. Functional slots with more than one stored string: 640 and 195. S15 has 4,059 subjects; 1,758 have a fact in the text store.

## The stores

- **Store K, the Known store** (S6's and S7's): every row of `/root/work/f2/v2/aug/seeds.tsv` with a known relation, all sources (Wikidata included). The rediscovery reads use it.
- **Store T, the 2006-text store:** the seed rows whose source is not `wikidata`, plus the admitted extracted facts of `facts_f2.tsv`. This is what Cypha read in the 2006 text. The S15 read uses it, because S15's truth is current Wikidata: a store seeded from current Wikidata would hold some of S15's unstated and future facts already, and its rules would learn from them.

## G2: the rule language and the miner

AMIE-style closed Horn rules with a head r(X, Y), mined from the store with no label:

| Kind | Body | Example |
|---|---|---|
| **const** | r₁(X, c₁) ⇒ r(X, c₂), r₁ may equal r | `occupation(X, Painter) ⇒ occupation(X, Artist)` |
| **same** | r₁(X, Y) ⇒ r(X, Y), r₁ ≠ r | `location(X, Y) ⇒ located_in(X, Y)` |
| **inverse** | r₁(Y, X) ⇒ r(X, Y) | `predecessor(Y, X) ⇒ successor(X, Y)` |
| **path** | r₁(X, Z) ∧ r₂(Z, Y) ⇒ r(X, Y), each atom in either direction | `birth_place(X, Z) ∧ country(Z, Y) ⇒ …` |

- **Measures** (AMIE): support = distinct (x, y) with the body and the head in the store; head coverage = support / facts of r; standard confidence = support / body pairs; **PCA confidence** = support / body pairs whose x has some value of r.
- **Kept** (DISCOVERY's guard): support ≥ 10, PCA confidence ≥ 0.5, and Benjamini–Hochberg at q = 0.1 over the batch of every candidate with support ≥ 10. A rule's p-value is the binomial upper tail of its support over its PCA body pairs, against p₀ = the mean base rate of the predicted object, P(y | r) with add-one.
- **Predictions:** every (x, r, y) a kept rule derives that is not in the store. Its score is the noisy-or of the PCA confidences of the rules that derive it (dev chooses between noisy-or and the maximum, on MRR; nothing else is tuned).
- **Ranking a slot (s, r, ?):** candidates are every object of r in the store plus every rule prediction for (s, r). Rule score first; ties and unscored candidates by the base rate P(y | r). Values already in the store for (s, r) are filtered out (the standard filtered setting).

## G14: contested slots and hidden variables

- **A contested slot:** a functional relation (φ ≥ 0.9 in pattern_learn's table, or a date relation) whose subject holds two values in store T that cannot both be literally right. Dates conflict when their years differ, or their days differ at day precision. Other values conflict when their normalised strings differ.
- **Hidden variables proposed** (DISCOVERY G14), each with a description length in bits:
  - **granularity / qualifier:** one place contains the other (a `located_in`, `country` or `capital` chain of length ≤ 3 in store K). Both are true at different grain. Cost 1 bit (which is finer).
  - **calendar / precision:** two dates at most 13 days apart (Julian and Gregorian), or the same year at different precision. Same fact. Cost 1 bit.
  - **time:** the relation can change over time (capital, largest_city, currency, location, country). Both were true at different dates. Cost log₂ of the number of values plus 2 bits.
  - **source error:** one value is wrong. Cost log₂ of the number of values (which one) plus 3 bits (an error is rarer than a legal change).
  - **sense / identity** (two meanings, two entities sharing a name) are not modelled: the store's subjects are page titles, so a single subject has one sense by construction. Said, not built.
- **Choice:** the explanation with the fewest bits whose structural precondition holds. For source error, the wrong value is the one T4 does not find on the subject's page when T4 finds the other, else the one with fewer sources in the store, else the later one in the store's order. Each chosen variable is then tested: granularity and calendar by their precondition (store), source error by T4 (page) and T1 (date constraints).
- **Graded against S15** (current Wikidata), slots of S15 subjects only:
  - **both true** when both values match Wikidata values of (s, r);
  - **granularity** when exactly one matches and the other contains it (store K's chains);
  - **error, value v** when exactly one matches and v is the other;
  - **ungraded** otherwise.
  - A granularity or calendar or time explanation is right on "both true" or "granularity"; a source-error explanation is right when it names the wrong value. A baseline keeps the value with more sources.

## The loop (S7's, with G2's hypotheses)

- Every hypothesis is proposed, scored with P0, queued, pre-registered (`PreregLog`, I5, I6), then tested cheapest first and decided as S7 decides.
- **P0** = the rule score (noisy-or PCA), times S6's hard constraint, clamped to [0.02, 0.98] (DISCOVERY 6.1's support and constraint terms; the generator's reliability starts at 1).
- **Tests in the plan:**
  - T1 consistency (S7's), store; 
  - **T2 consequence, back:** if h = r(x, y) holds, every kept rule with h in its body derives a consequence (the other body atom of a path rule from the store). Positive when a consequence is in the store and none is contradicted; negative when a consequence from a rule with PCA ≥ 0.8 lands in a functional slot (φ ≥ 0.9) that holds only other values; else inconclusive. Store, not independent. It earns its place by S7's rule (applicable ≥ 5% and Youden's J interval above 0 on dev) or is cut again;
  - T3 held-out prediction: the best kept rule deriving h is re-measured with a third of its subjects hidden (FNV-1a bits 13+ mod 3 = 1): `rule_holdout_test` on the hidden third. Store, not independent;
  - T6 compression (S7's, ε = 0), store;
  - T4 corpus (S7's `pattern_or_comention`), enwik8 prose of the subject's article, independent;
  - T5 Wikidata (S7's, with S7's eligibility), independent, **only in the rediscovery read**. The S15 read's truth is Wikidata, so T5 is left out there.
- **s and c** of every test are measured on dev pairs of the read (true and false hypotheses, below), add-one, as S7's amended loop did. Test tallies are printed only on the test read.
- **Promotion = supported** (posterior > 0.9 with independent evidence, I4) **and** kept by Benjamini–Hochberg at q = 0.1 **over the generator's batch of decisions** (the hypotheses `decide` supports). This is DISCOVERY 7.3's wording ("each batch of decisions from one generator"). S7 applied BH over every tested hypothesis and found it rejects nothing once s and c are shrunk; both forms are printed, and the gate reads the 7.3 form.

## The reads

**R1, rediscovery on store K** (DISCOVERY §12.2):
- **Hide 10% at random:** K's facts are shuffled with `mt19937_64(20261029)`; the first 10% are hidden. **Hide the least connected 10%:** facts sorted by deg(s) + deg(o) in the entity graph (ties by FNV-1a of the triple); the lowest 10% are hidden. Two separate runs.
- Rules are mined on the reduced store. Every hidden fact is a query (s, r, ?) whose object is ranked: Hits@1, Hits@10 and MRR, with bootstrap intervals over hidden facts, and the share any kept rule covers. Baselines: the base rate alone, and S6's link predictor on the same candidates.
- **Split by subject:** dev when bit 11 of FNV-1a(subject) is 0 (S7's split). Dev is read freely. Test is read once, with `--read-test`.
- **Promotion precision:** G2's hypotheses on dev (test) subjects go through the loop. A promoted fact is graded true when the full store K holds it, false when K holds values for (s, r) and none is o (the partial-completeness assumption), else ungraded.
- s and c pairs: each hidden fact (true) with a false partner by S7's rule (an object of r drawn from the reduced store, not held by s in K, not the same year for a date), `mt19937_64(20261030)`.

**R2, the temporal holdout on store T** (S15):
- Rules are mined on store T. Hypotheses are G2's predictions for S15 subjects.
- **Ranking:** every S15 true fact of category `b` (true by 2006, unstated) or `c` (true only after 2006-03-04) whose relation is one Cypha extracts, whose subject has a fact in store T, and whose value store T does not hold for (s, r): rank its value among (s, r, ?)'s candidates. Hits@1/@10 and MRR per category.
- **Promotion precision:** promoted hypotheses graded true when S15 lists the value for (s, r) (any category), false when S15's false row has it or S15 lists (s, r) values and none matches, else ungraded. A value matches by D4's normalisation (entities) or year and date compatibility (dates).
- **s and c pairs:** S15's own true and false rows (dev), for relations Cypha extracts, subjects with a fact in store T, values store T does not hold.
- **S15's leak note:** Dolly holds some `b` and `c` pairs. Track S's stores are enwik8 only (no Dolly), so nothing here has read them.
- **Dev freely, test once:** the final setting reads `temporal_holdout_test.tsv` once, and that read is logged in the S15 report's "Reads" table.

**G14** runs on store T's contested slots of S15 subjects, graded on dev while building and on test in the same single test read.

## Decision rule (fixed before the runs)

- The gate is read on R2's test read: **G2's promotion precision ≥ 0.9** (point estimate; Wilson 95% interval printed), with at least 10 graded promotions. Fewer than 10 graded promotions: the gate cannot be read and counts as missed.
- R1's precision, Hits@1/@10 and MRR on both reads are reported. Hits@10 has no threshold (the gate says "reported").
- **T2 comes back** only if it earns its place by S7's rule on dev.
- **G14 is kept** if its explanation accuracy on graded test slots beats the more-sources baseline's accuracy at naming the wrong value (on error slots) and is above 0.5 overall; otherwise it is reported and parked.
- Choices made on dev only: noisy-or or max; nothing else.

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| P1 | Kept rules on store K: 200–2,000 const rules; 3–30 same, inverse or path rules |
| P2 | `predecessor` and `successor` are found as each other's inverse in K, with PCA ≥ 0.7 |
| P3 | R1 random hide, dev: G2 Hits@10 0.30–0.55, Hits@1 0.10–0.30, MRR 0.15–0.35; above the base rate on all three |
| P4 | R1 least-connected hide: Hits@10 at least 0.10 below the random hide's |
| P5 | Kept rules cover (derive) 15–40% of the randomly hidden facts |
| P6 | T2 is applicable to 5–25% of R1's dev pairs and earns its place (J's interval above 0) |
| P7 | R1 promotion precision (dev) 0.85–0.98, with 50–2,000 graded promotions |
| P8 | R2: Hits@10 on `b` facts 0.05–0.25; on `c` facts below 0.05 |
| P9 | R2 promotion precision on test 0.80–0.95, with 10–300 graded promotions (T4 is the only independent test there, and a `b` fact is by definition not named on its 2006 page) |
| P10 | BH over every tested hypothesis (S7's form) promotes nothing in R2; the 7.3 form promotes at least 10 |
| P11 | Store T: 60–180 contested slots after the compatibility rule; G14 graded on 30–60% of S15 subjects' slots |
| P12 | G14's explanation accuracy 0.55–0.80; source error is the most common truth |

## Amendments (after the first dev read, before any test read)

The first dev read (2026-10-29) showed two problems with the grading, not with the generator. Both amendments are added beside the registered numbers; the registered gate, its grader and its threshold do not change.

1. **R1's grader cannot grade what G2 gets right.** Under the partial-completeness assumption a prediction is false when store K holds other values for the slot. For taxonomic values that is wrong: K lists Aaliyah's occupations as Actor, Dancer, Model, Musician and Pop singer, so the rule `Pop singer ⇒ Singer` is graded false; so are Björk Songwriter and Eric Clapton Songwriter. And most of G2's promotions on dev (`creator ⇔ author`: "Children of Dune creator Frank Herbert", "Batman author Bob Kane") are ungraded, because K has no value for the slot.
   - **Added: a second grader, S15.** A hypothesis is true when K or S15 (current Wikidata) lists the value; false when S15's false row has it, or S15 lists (s, r) values and none matches. K's own lists no longer count as complete. R1 dev reads S15's dev rows only; the test read adds S15's test rows.
2. **Lexical values form taxonomies.** Exact-value grading (both graders) is not valid for the lexical relations (occupation, genre, nationality, language, party, currency, family, order, class).
   - **Added: every precision is also printed for the non-lexical relations alone** (places, people, works, dates).
- The promoted hypotheses of every read are written to a file, for Odin's hand-check.
- No change to the generator, the tests or the loop.
3. **The rule's null was registered wrong.** The p-value's p₀ was registered as P(y | r) with add-one and one unseen value. With many rare values in a relation, that deflates a frequent value's share (0.66 instead of 0.80 in the planted check), so a rule no better than chance passed BH. Found by the known-answer check for a spurious rule, before any test read. **Corrected:** p₀ is the value's plain share of r's facts. The dev read was run again with it. Head coverage of a const rule also now divides by the facts of its head atom r(·, c₂), as AMIE defines it (reported only; no decision reads it).

## Results (dev 2026-10-29; test read once, commit 10bef62)

**The gate is missed: it cannot be read.** On R2's test read, G2's loop promoted 2 hypotheses and 1 could be graded (false). The registered rule needs at least 10 graded promotions. The 2006-text store is too small for G2 to propose much: 18 rules pass the guard, and they make 46 test hypotheses in all.

### The rules

| Store | Const kept (candidates) | Same | Inverse | Path |
|---|---:|---:|---:|---:|
| K, random hide | 156 (576) | 4 (16) | 2 (4) | 6 (21) |
| K, least-connected hide | 239 (808) | 2 (14) | 3 (6) | 6 (22) |
| T (2006 text) | 10 (46) | 2 (12) | 2 (2) | 4 (4) |

Top rules on K (random hide), by support × PCA:

| # | Rule | Support | PCA |
|---|---|---:|---:|
| 1 | `occupation(X, Poet) ⇒ occupation(X, Writer)` | 75 | 0.806 |
| 2 | `occupation(X, Novelist) ⇒ occupation(X, Writer)` | 66 | 0.857 |
| 3 | `occupation(X, Screenwriter) ⇒ occupation(X, Writer)` | 61 | 0.726 |
| 4 | `occupation(X, Film actor) ⇒ occupation(X, Actor)` | 57 | 0.740 |
| 5 | `occupation(X, Television actor) ⇒ occupation(X, Film actor)` | 46 | 0.836 |
| 6 | `occupation(X, Journalist) ⇒ occupation(X, Writer)` | 49 | 0.742 |
| 7 | `occupation(X, Philosopher) ⇒ occupation(X, Writer)` | 56 | 0.609 |
| 8 | `occupation(X, Actor) ⇒ occupation(X, Film actor)` | 57 | 0.582 |
| 9 | `located_in(X, Z) ∧ country(Z, Y) ⇒ country(X, Y)` | 34 | 0.919 |
| 10 | `occupation(X, science fiction writer) ⇒ occupation(X, Writer)` | 31 | 0.939 |

- The rules with variables are the ones a person would write: `predecessor(Y, X) ⇔ successor(X, Y)` (PCA 1.000 and 0.929), `creator(X, Y) ⇔ author(X, Y)` (1.000, 0.917), `capital ⇔ largest_city` (0.691, 0.594), `family(X, Z) ∧ class(Z, Y) ⇒ class(X, Y)` (0.867), and the `located_in`/`country` chains.
- Most const rules are the occupation taxonomy (a narrower job implies a broader one).

### R1: rediscovery on store K (Hits with bootstrap 95%)

| Hide | Read | Model | Hits@1 | Hits@10 | MRR | Rules cover |
|---|---|---|---|---|---|---:|
| random | dev (1,201) | **G2** | 0.110 [0.092, 0.127] | **0.241** [0.218, 0.264] | 0.158 [0.140, 0.176] | 8.4% |
| | | base rate | 0.059 | 0.204 | 0.108 | |
| | | S6's link predictor | 0.153 | 0.331 | 0.210 | |
| random | **test (1,077)** | **G2** | 0.111 [0.093, 0.132] | **0.230** [0.206, 0.256] | 0.153 [0.135, 0.173] | 7.4% |
| | | base rate | 0.059 | 0.201 | 0.105 | |
| | | S6's link predictor | 0.150 | 0.336 | 0.211 | |
| least connected | test (1,153) | G2 | 0.000 | 0.008 | 0.003 | 0.0% |
| | | S6's link predictor | 0.005 | 0.023 | 0.011 | |

- G2 beats the base rate on all three measures, and loses to S6's link predictor, which scores every pairwise association with no guard. G2's rules are the explanation: they cover few facts, but each one names why.
- The least-connected facts are almost all values that occur once (a spouse, a date). No store-based method can rank a value it has never seen.
- Noisy-or and the maximum tie (dev MRR 0.158 against 0.156); noisy-or was kept.

### The tests on G2's pairs (test, read once; Wilson 95%)

| Test | Applicable | s | c | J | In the loop |
|---|---:|---|---|---|---|
| T1 consistency | 18.2% | 0.728 | 0.959 | +0.687 | kept |
| T2 consequence (back) | 6.6% | 1.000 | 0.034 | +0.034 [−0.027, +0.172] | **cut**: it is never negative |
| T3 held-out rule | 2.8% | 1.000 | 0.000 | 0.000 | cut |
| T6 compression | 74.3% | 0.260 | 0.954 | +0.214 | kept |
| T5 Wikidata | 2.6% | 0.750 | 1.000 | +0.750 | cut: applicable to < 5% (most of K's slots came from Wikidata) |
| T4 corpus | 27.2% | 0.980 | 0.854 | +0.834 | kept |

- **T2 fires now, but S7's framing cuts it.** It is positive on 9.2% of true pairs and 2.7% of false ones on dev (a likelihood ratio of about 3.4), but almost never negative, so its s/c "J" is 0. A three-way likelihood (S7's question 2, S7b's work) would keep it.
- On R2, T2 and T3 never apply: store T's rules are too few.

### Promotion precision (Wilson 95%)

| Read | Grader | Relations | Promoted | Graded | Precision |
|---|---|---|---:|---:|---|
| R1 dev | registered (K) | all | 105 | 24 | 0.458 [0.279, 0.649] |
| R1 dev | S15 (amendment 1) | all | 105 | 16 | 0.750 [0.505, 0.898] |
| R1 test | registered (K) | all | 85 | 13 | 0.538 [0.291, 0.768] |
| R1 test | S15 | all | 85 | 11 | **0.818** [0.523, 0.949] |
| R1 test | either | non-lexical | 76 | 4 | 1.000 [0.510, 1.000] |
| **R2 test (the gate)** | S15 | all | 2 | 1 | 0.000: **not readable** |

- **Why so few graded:** 72 of R1 test's 76 non-lexical promotions are `creator ⇔ author` predictions for works neither K nor S15 covers ("Children of Dune creator Frank Herbert", "Batman author Bob Kane"). They read as right; Odin can hand-check them in `test.r1_test.hyps.tsv` (column `promoted`).
- **What goes wrong:** the independent test that promotes is T4's co-mention variant (S7's choice). A rule's prediction that the page merely names gets through: on R2 dev, "India capital Mumbai" and "Connecticut capital Bridgeport" (from `capital ⇔ largest_city`), "Bermuda located_in United States". With T4's pattern variant (an extraction; dev only, exploratory) nothing is promoted at all: a rule's prediction is, by nature, a fact the text does not state.
- BH over every tested hypothesis (S7's form) promotes nothing in any read; the 7.3 form promotes 105, 85, 5 and 2.

### R2: the temporal holdout (S15, store T)

| Read | Category | n | Hits@1 | Hits@10 | MRR | Rules cover |
|---|---|---:|---|---|---|---:|
| dev | b (true by 2006, unstated) | 1,163 | 0.006 | 0.021 [0.014, 0.030] | 0.012 | 0% |
| dev | c (true only after 2006-03-04) | 146 | 0.000 | 0.075 [0.034, 0.116] | 0.024 | 0% |
| test | b | 1,231 | 0.008 | 0.026 [0.018, 0.035] | 0.013 | 0% |
| test | c | 170 | 0.000 | 0.029 [0.006, 0.059] | 0.014 | 0% |

- No kept rule derives a single S15 `b` or `c` fact. What little ranks well is the base rate.

### G14: contested slots of store T

- Store T has 194 contested slots (functional relations with two incompatible values); 94 are test subjects' slots, 59 graded (14 had three or more values and are not handled).
- The truth is almost always a source error (57 of 59): the 2006 text disagrees with itself, and one side is wrong.
- **Test: accuracy 0.542 [0.417, 0.663].** On error slots G14 names the wrong value 31 of 57 times, the same as keeping the value with more sources (31 of 57). Dev: 0.650, against 0.605 for the baseline.
- **By the registered rule G14 is parked:** it does not beat the baseline on test. Its granularity and calendar explanations were right each time they fired (2 of 2), but they rarely fire.

### Predictions, scored

| # | Result |
|---|---|
| P1 | **Partial.** Const rules 156 on the random hide (predicted 200–2,000: miss), 239 on the least-connected; rules with variables 12 and 11 (hit) |
| P2 | **Hit.** Inverse both ways, PCA 1.000 and 0.929 |
| P3 | **Partial.** Dev Hits@10 0.241 (predicted 0.30–0.55: miss); Hits@1 0.110 and MRR 0.158 in range; above the base rate on all three (hit) |
| P4 | **Hit.** 0.008 against 0.241 |
| P5 | **Miss.** Rules cover 8.4%, not 15–40% |
| P6 | **Miss.** Applicable 6.0%, but cut (never negative) |
| P7 | **Miss.** R1 dev precision 0.458 on 24 graded |
| P8 | **Miss.** b Hits@10 0.021 (dev), below 0.05–0.25; c 0.075 on dev (predicted < 0.05), 0.029 on test |
| P9 | **Miss.** 1 graded promotion |
| P10 | **Partial.** S7's form promotes nothing (hit); the 7.3 form promotes 5 (dev) and 2 (test), not ≥ 10 |
| P11 | **Partial.** 194 contested slots (predicted 60–180: miss); 50% and 63% graded |
| P12 | **Partial.** Accuracy 0.542 on test (predicted 0.55–0.80: miss); source error is the most common truth (hit) |

2 hits, 5 partial, 5 misses.

## Decisions

1. **G2 stays, as a generator of explained hypotheses.** It beats the base rate and its rules are readable. It does not replace S6's link predictor as a scorer.
2. **S8's gate is missed** (not readable on the 2006-text store). R1's promotions are 0.818 by the S15 grader on test, on 11 graded: below 0.9, with a wide interval.
3. **G14 is parked** (no better than "keep the value with more sources").
4. **T2 stays cut** under S7's earn-its-place rule; it should come back with three-way likelihoods.

## For Odin

1. **Promotion needs a stronger independent test than T4's co-mention.** Options: T4's pattern variant (promotes nothing on this text), Wikidata as T5 where the store did not take the slot from Wikidata, or a teacher (G16) on your PC.
2. **The 2006-text store is small** (8,319 facts). A larger extraction (F2 on more of enwik9) would give G2 enough rules for the temporal holdout to be readable.
3. **Hand-check** of the promoted `creator ⇔ author` facts: `test.r1_test.hyps.tsv` (the run files are under `/root/work/s8a/runs`, not committed).

## Reads of the temporal holdout's test half

One read, 2026-10-29, `--read-test`, commit 10bef62, logged in the S15 report.

## How to reproduce

```
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
F=/root/work/f2/v2/aug
./pattern_learn --gen-rules enwik8 $F/seeds.tsv $F/facts_f2.tsv /root/work/f2/wikidata/wd_values.tsv $F/patterns.tsv bench/eval OUT   # dev
./pattern_learn --gen-rules enwik8 $F/seeds.tsv $F/facts_f2.tsv /root/work/f2/wikidata/wd_values.tsv $F/patterns.tsv bench/eval OUT --read-test   # the one test read
./pattern_learn --gen-rules ... OUT --t4-pattern   # the exploratory T4 variant, dev only
```

About 75 s (dev) and 130 s (test), 236–247 MB peak. It writes the rules of each store (`OUT.r1_*.rules.tsv`, `OUT.r2_T.rules.tsv`), the hypotheses with their tests and states (`OUT.r1_*.hyps.tsv`, `OUT.r2_*.hyps.tsv`), G14's slots (`OUT.g14_*.tsv`) and every pre-registration log (`OUT.*.prereg.tsv`).
