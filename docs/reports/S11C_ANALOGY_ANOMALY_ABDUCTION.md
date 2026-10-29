# S11c: anomalies (G13), abduction (G3), and analogy (G4, G5)

**Odin Loch** · 2026-10-29 · DISCOVERY §5 (G3, G4, G5, G13), §6, §7, §12.2, §13 S11 · follows [S8A](S8A_RULES_AND_CONTRADICTIONS.md), [S8B](S8B_LINK_PREDICTION.md), [S8D](S8D_PROMOTION.md) and [S10](S10_CURIOSITY_SCHEDULER.md)

(The design, the reads, the decision rules and the predictions below were written and committed before any S11c generator ran. Results are added under them.)

**Question.** Four more generators for Cypha's discovery track:
- G13 finds values far from what their type allows, and says what is wrong;
- G3 finds the missing facts that would explain an observation no rule derives;
- G4 carries relations over from a structurally similar entity;
- G5 carries them over by word-vector offsets.

Do they find what was hidden or planted, better than a baseline, and does the loop promote only true facts from them?

**Rule 10.** The generators write hypotheses for Cypha's own store. They never answer a question for CyphaLM. Nothing here reaches `qa` or `chat`.

**Counted before registering** (sizes only): store K has 22,783 triples, 79% from Wikidata; store T (the 2006 text) has 8,319. Persons with a birth and a death year: 955 in K, 866 in T. A plain year regex finds a death year before the birth year in 48 of K's and 29 of T's. That count was made while sizing the stores; nothing else about any generator's output was looked at.

## Shared setup

- **Stores.** S8a's: **K** = every seed row (Wikidata included); **T** = the seed rows not from Wikidata plus F2's admitted facts. T's source count of a triple is the number of rows that support it (S8a's `t_sources`).
- **Rules.** S8a's G2 miner and guard (`mine_rules`: support ≥ 10, PCA ≥ 0.5, BH at 0.1), noisy-or.
- **Subject split for planted and rediscovery reads:** dev when bit 11 of `fnv(subject)` is 0 (S8a's R1 split), test otherwise. Dev is read freely; test once, for the final setting, with `--read-test`.
- **The loop** (S8b's, S7's dev point estimates): T1 consistency (s 0.7551, c 0.9184), T6 compression (0.2553, 0.9493), T4 corpus, pattern or co-mention (0.9933, 0.8137; the only independent test). T5 (Wikidata) is left out: Wikidata grades. Pre-registered in `PreregLog`, cheapest first, `decide` with S8d's fix (support needs a positive independent result).
- **The source stage:** `--sources … --fresh` with S8d's calibration (DB-IB carries dates, places and links; GN places; WP-LEAD and DB-IB lexical never carry). **Promoted = supported by `decide` after the source stage**, as S8d.
- **Wilson 95%** intervals on proportions; **2,000 bootstrap resamples** (seed 20261029) on Hits and MRR, and on differences, resampling queries.

## G13: anomalies

**Derived quantities** read from a subject's facts. Years come from `parse_date_str` (BC negative); a subject's birth years are those of `birth_year` and `birth_date`, and likewise for death.

| Kind | Fires when | Facts involved |
|---|---|---|
| A1 death before birth | some death year < some birth year | the birth fact and the death fact |
| A2 lifespan tail | death − birth > the 99.5th percentile of the store's lifespans (one-sided tail p ≤ 0.005, add-one), or > 122 | birth, death |
| A3 year mismatch | a `birth_date`'s year ≠ a `birth_year` (likewise death) | the two facts |
| A4 capital elsewhere | capital(C, X) and country(X, Y) with Y ≠ C, Y ≠ X | capital(C, X), country(X, Y) |
| A5 founder timing | founder(O, P) and founded(O) = y with P born after y, or P's death more than 1 year before y | founder, founded, the person's date |
| A6 spouse gap | \|birth(A) − birth(B)\| for spouse(A, B) beyond the store's 99.5th percentile of spouse gaps | spouse, the two birth facts |
| A7 succession gap | birth-year gap for predecessor or successor pairs beyond the 99.5th percentile of such gaps | the link, the two birth facts |

Percentiles come from the store being read (it describes its own types); a tail needs at least 30 values of its quantity, else the kind does not fire.

**Explanations** (DISCOVERY G13), each with a description length in bits; the cheapest whose precondition holds is the verdict:
- **qualifier** (1 bit): A1–A3, negating one year with \|y\| < 1000 removes the anomaly (a lost BC sign); A4, C contains Y or Y contains C in the store (`located_in` or `country`, chains up to 3: a state's capital in its country).
- **different entity** (2 bits): A4–A7, the linked entity's name is ambiguous: a page title "Name (…)" exists besides it.
- **error in one fact** (log₂ of the facts involved + 3 bits, G14's price).
- **genuinely unusual** (−log₂ of the tail p; infinite for A1 and A3, which no real fact can make).
- **Which fact is wrong**, for an error: the one with fewer source rows; then the one T4 does not find on the subject's page when T4 finds another; then, registered order: the year relation over the date relation (A3), the death fact over the birth fact (A1, A2), the link over the dates (A4–A7).

**Reads.**
- **G13-P, planted, store K.** Seven corruption types, applied together to one copy of K, each to up to 100 facts of dev subjects (test subjects for the test read) drawn by `mt19937_64(20261101)`:
  - C1 century: a birth or death year (or a date's year) ± 100 (sign by hash);
  - C2 transposition: the year's last two digits swapped (only if they differ);
  - C3 BC sign lost: a negative year made positive;
  - C4 swap: a person's birth and death years exchanged (one event, two facts);
  - C5 capital: replaced by another country's capital;
  - C6 spouse: replaced by another person of the store;
  - C7 founder: replaced by another person of the store.
  - A corrupted fact keeps the original's source count. No corruption touches a subject twice.
  - **Measures:** recall (an anomaly involves the corrupted fact), overall and among corruptions that are detectable in principle (the subject has the other fact the kind compares); **naming accuracy** (the verdict names the corrupted fact, among detected); the **baseline** names the fact with fewer sources, then at random (seeded); flags on facts that were not corrupted are counted and reported.
- **G13-R, real, store T.** Every anomaly of store T. A fact is graded against Wikidata (`wd_values.tsv`): right when Wikidata lists its value for (s, r), wrong when it lists other values and not this one, else ungraded. A verdict "error in f" is right when f is graded wrong; "genuinely unusual" or "qualifier" is right when every graded fact involved is right; "different entity" is graded like "error in the link". Split by bit 11, test once.
- **G13's hypotheses for the loop:** each questioned fact of store T's anomalies, "f is true", with P0 = 1 − P(error) (0.2 when the verdict is an error in f, 0.8 when it names another fact or no error), through the source stage. Graded as above.

**Decision rule (G13).** Kept if (1) on G13-P's test read its naming accuracy beats the baseline's (bootstrap interval of the difference above 0) and (2) its error verdicts on G13-R's test subjects have precision ≥ 0.9 with at least 10 graded. Otherwise reported and parked (as G14 was).

## G3: abduction

- **An unexplained observation:** a fact (x, r, y) of the store that no kept rule derives from the other facts (y is not among `RuleSet::candidates(x, r)`).
- **Explanations:** for each kept rule with head r, the facts that must be added for it to derive y:
  - const r₁(X, c₁) ⇒ r(X, y): {r₁(x, c₁)};
  - same r₁(X, Y) ⇒ r(X, Y): {r₁(x, y)};
  - inverse r₁(Y, X) ⇒ r(X, Y): {r₁(y, x)};
  - path: one atom already in the store through a join node z, the other added (one fact); paths with both atoms missing are not proposed (z would be free).
  - Facts already in the store are never proposed; an explanation with no missing fact means the observation was explained.
- **Description length:** bits(E) = Σ over added facts f of −log₂ P(f) − log₂ PCA(rule), with P(f) = S6's link predictor on the store. Explanations are ranked by bits, ties by the rule's support, then the fact's text. Distinct added facts are ranked by their best explanation.
- **G3-P, planted, store K.** Every dev fact O with a derivation in K by a kept rule (mined on K) through one body fact B in the store (the highest-PCA such rule; for a path rule, atom 2 when `fnv(O)` is even). Up to 500 cases are drawn by `mt19937_64(20261102)`, with distinct B and no B that is another case's O. All drawn B are deleted together, rules are mined again on the reduced store, and each O that is now unexplained is a case. **Recall@k** = B among the k best added facts, k = 1, 5, 10.
  - **Baselines,** on the same candidate facts: random order (expected recall@k), the link predictor alone (no rule term), the rule's PCA alone (no fact term).
- **G3's hypotheses for the loop:** in the promotion read below.

**Decision rule (G3).** Kept if on the test read recall@10 ≥ 0.6 and recall@10 beats random order (bootstrap interval of the difference above 0). The link-predictor-alone and PCA-alone orders are reported.

## G4: structural analogy

For a slot (t, r):
- **Analogs:** subjects s ≠ t with a fact of r. A relation r′ ≠ r **aligns** when both have it: *identical* if they share a value, *parallel* otherwise. An analog needs at least 3 aligned relations, one of them identical (DISCOVERY's guard against surface similarity). Its systematicity is the number of aligned relations plus one for each parallel object pair (a, b) whose own facts share a relation (a matched higher-order relation). Candidates are drawn through shared (r′, value) pairs, skipping a pair held by more than 500 subjects; the 20 analogs with the highest systematicity are kept (ties by the rarer shared values, then name).
- **Transfer:** each value a of (s, r) gives candidates for (t, r) by its role in s's structure:
  - *equality:* a is also s's value of r″ ⇒ t's values of r″;
  - *path:* s r₁ z, z r₂ a ⇒ t r₁ z′, z′ r₂ b gives b;
  - *constant:* a itself, weight λ.
  - A candidate's score is the sum over analogs of systematicity × role weight (1, or λ for a constant).
- **λ** is chosen on dev among {0, 0.25, 1} by MRR (λ = 0 is pure structure mapping).

## G5: vector analogy

- **Vectors:** S-1's PPMI-SVD of enwik8 (300 dimensions, 50,000 words; `ppmi_svd`, rebuilt as `data/cyphalm/README.md` says, on all of enwik8) or `data/cyphalm/simvec_ppmi32.bin` (32 dimensions); dev chooses by MRR. An entity's vector is the mean of its lower-cased words' unit vectors (words outside the vocabulary skipped); no word, no vector.
- **Relation offset:** v_r = the mean of unit(b) − unit(a) over the store's facts (a, r, b) with both vectors.
- **Score** of candidate o for (t, r): cosine(unit(t) + v_r, unit(o)).
- **Calibration per relation (dev):** G5 is on for r only if its dev MRR on r beats the base rate's with the bootstrap interval of the difference above 0 (at least 20 dev queries). Off relations get the base rate (G5's low prior, DISCOVERY).

## The reads for G4 and G5

1. **Rediscovery, store K (S8a's R1 random hide):** the same 10% hidden (S8a's `mt19937_64(20261029)` over its triple order), the same dev and test subjects. Every hidden fact is a query (s, r, ?): candidates are r's objects in the reduced store plus the generator's candidates, the slot's other known values filtered. Order: the generator's score, then the base rate, then the string (S8a's `rank_value`). Hits@1, Hits@10, MRR with bootstrap intervals; **base rate and G2 computed in the same run** as baselines; the share of queries the generator scores.
2. **Temporal holdout, store T (S15),** as S8a's R2: S15 `b` and `c` true facts whose relation Cypha extracts, whose subject has a fact in T and whose value T does not hold; Hits@10 and MRR per category. Dev freely; the test half once, with the final setting, logged in S15's Reads table.

**Decision rule (G4, G5; S8b's rule 3).** A generator earns its place if its test MRR on read 1 beats the base rate's with the bootstrap interval of the difference above 0. Otherwise its code goes and the report keeps its numbers.

## Promotion read (all four)

S8's gate (precision ≥ 0.9 with at least 20 graded promotions) is the reference; it is not this step's gate.

- **G4, G5 and G3** on S8d's construction: the slots (s, r) of a half of S15, every store K triple in them hidden, the generators on the reduced store.
  - **dev:** S15 dev's confirm half (bit 17 = 1, S8d's slots; read before by S8d for G2 and G6/G7, never for these generators);
  - **final:** S15 test's half with bit 17 = 0, read once, logged in S15's Reads table (S10 used the bit-17 = 1 half).
  - G4 and G5: the top candidate per slot, as S8b. Prior: an isotonic map from the generator's score to P(true), fitted on read 1's dev hidden slots.
  - G3: the unexplained observations of the reduced store; each distinct added fact that falls in a hidden slot, at its best explanation. Prior: an isotonic map from bits to P(true), fitted on G3-P's dev cases.
- **G13:** its store T hypotheses (above).
- Then the loop, the source stage, and `decide`. Graded with S8c's grader on that half of S15 (G13: on Wikidata, above). Precision per generator and pooled, Wilson 95%.
- The pool files are written in S8d's format (`*.hyps.tsv` with split `fresh`, the pre-registration log, and the source stage's `*.s8d_fresh.tsv` and `*.s8d_fresh_ran.tsv`), so S10's replay can read them later.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| N1 | G13-P dev recall, detectable in principle: C4 ≥ 0.9; C1 0.7–0.95; C2 under 0.3; C5 0.5–0.9; C6 and C7 under 0.4 |
| N2 | G13-P dev recall over all corruptions 0.25–0.5 (most corrupted facts have nothing to be compared with) |
| N3 | G13-P naming accuracy 0.65–0.85; the baseline 0.45–0.6 |
| N4 | G13-R: store T has 40–150 anomalies, A1 and A3 the most common; error-verdict precision 0.6–0.85 (the gate's 0.9 missed) |
| N5 | G3-P dev: 150–500 cases; recall@1 0.3–0.6, recall@10 0.7–0.95; random order's recall@10 under 0.5 |
| N6 | G3's order beats the link predictor alone and PCA alone at recall@1 by at least 0.05 each |
| N7 | G4 dev, read 1: Hits@10 0.22–0.35 and MRR above the base rate's; the share of queries G4 scores 30–70% |
| N8 | λ = 0.25 or 1 is chosen (values carried as constants help) |
| N9 | G5 is on for at most a third of the relations with ≥ 20 dev queries; its overall dev Hits@10 within ±0.03 of the base rate |
| N10 | The 300-dimension vectors are chosen over the 32-dimension ones |
| N11 | Temporal read: G4 and G5 Hits@10 on `b` facts under 0.1 (as G2's 0.02–0.03 and S8b's) |
| N12 | Promotion: fewer than 20 graded promotions per generator on the final read; pooled precision 0.75–0.95; S8's gate not met |
| N13 | G4 earns its place; G5 does not |

## Amendments (on dev, before any test read)

The first dev reads (2026-10-29) showed two faults in the measurement, not in the generators. Both are fixed before any test read; no threshold changes.

1. **A lost BC sign names its year.** The registered qualifier for A1–A3 ("negating one year removes the anomaly") named no fact, and its G13-R grade asked every fact to be right. For a lost BC sign that is wrong: the stored year is the fault. On dev, G13 found 20 of the 31 planted C3 corruptions and named 2.
   - **Changed:** an A1–A3 qualifier names the year whose sign it restores. It is graded like an error in that fact, in both reads. A4's qualifier (one place inside the other) still names no fact and is graded as registered.
2. **G3-P deletes one fact at a time.** Deleting all 500 drawn facts together took support from rules that many cases share (7 cases on `occupation(X, Programmer) ⇒ occupation(X, Computer scientist)` alone). The rules of 108 of the 279 unexplained cases were no longer kept, so the read measured the deletions' damage to the miner, not G3.
   - **Changed:** each case deletes its own B alone, and the rules and the link predictor are rebuilt for it. The cases run on 3 threads; they are independent, so the results do not depend on the threads.

**Clarifications** (no change of rule):
- A4 and A5 have no tail, so "genuinely unusual" costs infinitely many bits for them too, as for A1 and A3.
- `abduce` reads at most 5,000 join nodes per path rule on each side (a hub guard, like G2's `max_join`).
- A6 cannot fire on store K: only 11 spouse pairs there have two birth years, and an add-one tail p needs at least 199 values to reach 0.005. This was found on dev (C6: 0 of 100), and it is reported, not changed.

## Dev results (2026-10-29, after the amendments)

### G13, planted (store K, dev subjects)

600 corruptions; 494 could be seen at all (the subject holds the fact the anomaly compares).

| Type | Planted | Recall | Recall among the detectable | Named right | Baseline (fewer sources) |
|---|---:|---|---|---|---|
| C1 century | 100 | 0.980 | 98 / 99 | 0.745 | 0.490 |
| C2 transposition | 100 | 0.770 | 77 / 98 | 0.571 | 0.545 |
| C3 BC sign lost | 31 | 0.645 | 20 / 24 | 1.000 | 0.350 |
| C4 birth and death swapped | 100 | 1.000 | 100 / 100 | 1.000 | 1.000 |
| C5 capital | 89 | 0.112 | 10 / 10 | 0.800 | 0.200 |
| C6 spouse | 100 | 0.000 | 0 / 87 | — | — |
| C7 founder | 80 | 0.738 | 59 / 76 | 1.000 | 0.525 |
| **All** | **600** | **0.607** [0.567, 0.645] | 364 / 494 = 0.737 | **0.835** [0.794, 0.870] | 0.632 [0.581, 0.680] |

- Naming beats the baseline by +0.203 [+0.159, +0.247].
- The corrupted store has 636 anomalies; 68 involve no planted fact (the store's own).
- C5 is seen only when the new capital has a `country` fact (10 of 89). C6 cannot be seen (the clarification above).

### G13, real (store T, dev subjects, graded by Wikidata)

26 anomalies: 24 year mismatches (A3), 1 death before birth, 1 lifespan tail. Every verdict is "error". Of 21 graded, 13 name a fact Wikidata does not hold: 0.619 [0.409, 0.792]; the fewer-sources baseline 11 / 20. The lifespan tail (711 BC to 585 BC) is a real outlier flagged as an error: Wikidata holds both years.

### G3, planted (store K, dev subjects)

500 cases drawn; 243 are still explained by another rule once their fact is deleted; 257 are read. 17 have no candidate (the rule is no longer kept without the deleted fact).

| Order | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| **G3 (bits)** | 0.568 [0.510, 0.630] | 0.798 | **0.837** [0.790, 0.879] |
| random order | 0.455 | 0.665 | 0.731 |
| link predictor alone | **0.634** | 0.809 | 0.837 |
| rule PCA alone | 0.455 | 0.646 | 0.716 |

- G3 beats random order at recall@10 by +0.105 [+0.070, +0.143], and PCA alone at recall@1 by +0.113 [+0.070, +0.156].
- The link predictor alone beats G3 at recall@1 by 0.066 [0.023, 0.113]: the rule's bits push down the right fact. Nothing is changed for it (no choice was registered here).

### G4 and G5, read 1 (S8a's R1 random hide, dev, 1,201 queries)

| Generator | Hits@1 | Hits@10 | MRR | MRR − base rate | Queries scored |
|---|---|---|---|---|---|
| base rate | 0.059 | 0.204 | 0.108 | — | all |
| G2 (S8a's numbers, reproduced) | 0.110 | 0.241 | 0.158 | +0.050 | all |
| G4, λ = 0 | 0.084 | 0.227 | 0.132 | +0.024 [+0.015, +0.033] | 13.2% |
| **G4, λ = 0.25** | **0.111** | **0.252** | **0.160** | **+0.052** [+0.038, +0.066] | 66.8% |
| G4, λ = 1 | 0.099 | 0.246 | 0.151 | +0.043 | 66.8% |
| **G5, 300 dimensions** | 0.078 | 0.236 | 0.131 | +0.023 [+0.012, +0.035] | 14.0% |
| G5, 32 dimensions | 0.059 | 0.204 | 0.108 | 0 (on for no relation) | 0% |

- **Choices:** λ = 0.25; the 300-dimension vectors. G5 is on for 2 of 14 relations: country (MRR 0.409 against 0.267) and nationality (0.458 against 0.247). On dates it is far below the base rate (years have no useful word vectors).
- Pure structure mapping (λ = 0) scores only 13% of queries: few targets have an analog with an equality or path role for the slot.

### G4 and G5, read 2 (S15 dev, store T)

| Generator | `b` Hits@10 (1,163) | `b` MRR | `c` Hits@10 (146) | `c` MRR |
|---|---|---|---|---|
| base rate | 0.021 | 0.012 | 0.075 | 0.024 |
| G2 | 0.021 | 0.012 | 0.075 | 0.024 |
| **G4** | **0.056** | **0.049** | 0.075 | 0.038 |
| G5 | 0.021 | 0.012 | 0.075 | 0.024 |

G4 scores 37% of the `b` facts and finds 4.4% at rank 1 (the others 0.6%).

### The promotion read on dev (S15 dev's bit-17 = 1 slots; G13: store T's dev subjects)

| Generator | Hypotheses | Graded | Promoted (after the sources) | Graded promotions | Precision | Refuted (graded, share false) |
|---|---:|---:|---:|---:|---|---|
| G3 | 75 | 47 | 11 | 11 | 10 / 11 = 0.909 [0.623, 0.984] | 11 (5, 0.800) |
| G4 | 26 | 21 | 0 | 0 | — | 25 (20, 0.900) |
| G5 | 48 | 48 | 2 | 2 | 2 / 2 | 0 |
| G13 (graded by Wikidata) | 52 | 41 | 7 | 7 | 7 / 7 | 9 (9, 1.000) |
| **All** | **201** | **157** | **20** | **20** | **19 / 20 = 0.950** [0.764, 0.991] | |

- Every promotion rests on a DB-IB positive except two of G3's, which rest on GN (one true, one false).
- G3's promotions are capitals (10) and one death place: the facts its rules need (`capital ⇔ largest_city`).
- G4 proposes few hypotheses here (26 of 1,330 slots): S8d's construction hides every slot of an S15 subject at once, so the target keeps few relations to align. Its dates are refuted (12 of 12 false).
- G13: the source stage refuted 9 questioned facts, all wrong by Wikidata, and supported 7, all right. That is G13's two readings tested, as DISCOVERY asks.

## The test pools, fixed before the reads

Built with the final code (commit 862472a); no test label was read to make them. Slots: S15 test's subjects with bit 17 = 0 (3,436 slots; 4,727 store facts hidden). G13: store T's anomalies of test subjects (bit 11 = 1).

| File | Hypotheses | SHA-256 |
|---|---:|---|
| `a.fresh_g3.hyps.tsv` | 101 | `229b37cc5dcce15e9e409280cb1590b78c0a0097e756f3766bb4e7f95ae591bb` |
| `a.fresh_g13.hyps.tsv` | 38 | `baf92b46e36dcff59d041043cb7c2baac88dd7607d488a89f988defbc4cadf61` |
| `g.fresh_g4.hyps.tsv` | 69 | `5921f5c474450a52497ef8368026977e41263e2b7b4fa6e4c0c1bf88a3b8c2af` |
| `g.fresh_g5.hyps.tsv` | 45 | `e9efcfd2f177f3917d6e8f7cbd3a4e831650a18787e6b3cd608112d2569390bb` |
| `src_g3.s8d_fresh.tsv` (source stage) | | `8ef59f6c2d215ea140679d99dfa07c12e611fe501279dcd108949fb938a0de03` |
| `src_g13.s8d_fresh.tsv` | | `4d35d7354fa74afbbe78f7c96d78de472d3d3d8ed8151b0f384b027d96e06d95` |
| `src_g4.s8d_fresh.tsv` | | `dd0bb334e86924e8bb26a37dd270697deefc27c5592853939750970105a733e4` |
| `src_g5.s8d_fresh.tsv` | | `9e4d4f3356635d90a8070386c4e2bfc02b21677167184309ac7cfc205772e761` |

A second run of `--gen-analogy --fresh-test` wrote the G4 and G5 files again byte for byte. The test reads follow this commit, once each: `--gen-anomaly --read-test` (G13-P, G13-R, G3-P), `--gen-analogy --read-test` (read 1, and read 2 on S15 test), and `--pool-grade` on the four pools.

## Test results (read once, after commit 804d77c)

### G13, planted (store K, test subjects)

| Type | Planted | Recall | Recall among the detectable | Named right | Baseline |
|---|---:|---|---|---|---|
| C1 century | 100 | 0.990 | 99 / 99 | 0.778 | 0.475 |
| C2 transposition | 100 | 0.800 | 79 / 98 | 0.625 | 0.562 |
| C3 BC sign lost | 28 | 0.750 | 21 / 24 | 0.905 | 0.429 |
| C4 swap | 100 | 1.000 | 100 / 100 | 1.000 | 1.000 |
| C5 capital | 83 | 0.108 | 9 / 9 | 1.000 | 0.667 |
| C6 spouse | 100 | 0.000 | 0 / 89 | — | — |
| C7 founder | 92 | 0.707 | 65 / 80 | 1.000 | 0.431 |
| **All** | **603** | **0.620** [0.581, 0.658] | 373 / 499 = 0.747 | **0.856** [0.816, 0.888] | 0.628 [0.578, 0.676] |

Naming beats the baseline by **+0.227 [+0.184, +0.273]**: part 1 of G13's rule holds.

### G13, real (store T, test subjects, graded by Wikidata)

19 anomalies: 15 year mismatches, 3 deaths before births, 1 lifespan tail. Verdicts: 17 errors, 2 lost BC signs. **Error verdicts right 11 of 17 = 0.647 [0.413, 0.827]**; the fewer-sources baseline 12 of 17 = 0.706. Part 2 (≥ 0.9 with ≥ 10 graded) **is not met**. The year mismatches carry it: 7 of 13 right. When a birth date and a birth year disagree, which one is wrong is close to a coin toss for G13's evidence.

### G3, planted (store K, test subjects)

500 drawn; 206 still explained by another rule; 294 read; 21 with no candidate (the rule is not kept without the deleted fact).

| Order | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| **G3 (bits)** | 0.565 [0.507, 0.622] | 0.803 | **0.840** [0.799, 0.881] |
| random order | 0.496 | 0.696 | 0.758 |
| link predictor alone | **0.612** | 0.820 | 0.844 |
| rule PCA alone | 0.514 | 0.714 | 0.759 |

- G3 − random at recall@10: **+0.082 [+0.054, +0.114]**. G3's rule holds: **kept**.
- G3 − PCA alone at recall@1: +0.051 [+0.020, +0.082]. G3 − the link predictor alone at recall@1: **−0.048 [−0.085, −0.010]**: the rule's bits push the right fact down, as on dev.

### G4 and G5, read 1 (test, 1,077 queries)

| Generator | Hits@1 | Hits@10 | MRR | MRR − base rate |
|---|---|---|---|---|
| base rate | 0.059 | 0.201 | 0.105 | — |
| G2 | 0.111 | 0.230 | 0.153 | +0.048 [+0.036, +0.060] |
| **G4, λ = 0.25** | **0.124** | **0.268** [0.244, 0.294] | **0.173** [0.156, 0.194] | **+0.068 [+0.052, +0.085]** |
| **G5, 300 dimensions** | 0.085 | 0.238 [0.212, 0.264] | 0.135 [0.118, 0.152] | **+0.029 [+0.016, +0.043]** |

Both intervals are above 0: **G4 and G5 earn their place**. G4 is the best ranker of the store's generators on read 1 (G2 0.153, S6's link predictor 0.211 in S8a's read is a scorer, not a generator). The other λ values on test, for the record: λ = 0 MRR 0.142, λ = 1 0.167.

### G4 and G5, read 2 (S15 test, store T)

| Generator | `b` Hits@10 (1,231) | `b` MRR | `c` Hits@10 (170) | `c` MRR |
|---|---|---|---|---|
| base rate, G2 | 0.026 | 0.013 | 0.029 | 0.014 |
| **G4** | **0.050** [0.038, 0.063] | **0.042** | **0.041** | 0.022 |
| G5 | 0.027 | 0.013 | 0.029 | 0.014 |

G4 scores 35% of the `b` facts and puts 3.7% at rank 1 (base rate 0.8%). G2 derives none of them, as in S8a.

### The promotion read (test pools, graded once)

| Generator | Hypotheses | Graded | Promoted | Graded promotions | Precision | Refuted (graded, share false) |
|---|---:|---:|---:|---:|---|---|
| G3 | 101 | 50 | 7 | 7 | 7 / 7 | 7 (2, 1.000) |
| G4 | 69 | 58 | 0 | 0 | — | 64 (53, 0.943) |
| G5 | 45 | 45 | 1 | 1 | 1 / 1 | 0 |
| G13 (Wikidata) | 38 | 34 | 5 | 3 | 3 / 3 | 4 (2, 1.000) |
| **All** | **253** | **187** | **13** | **11** | **11 / 11 = 1.000** [0.741, 1.000] | |

- G3's promotions: 6 capitals and 1 death place; 4 rest on a DB-IB positive and 3 on GN.
- G4 promotes nothing; the source stage and the loop refute 64 of its 69 hypotheses, 50 of 53 graded ones rightly. Its dates are all wrong (34 of 34): a date carried as a constant from an analog is not a fact about the target.
- G13: the source stage supported 5 questioned facts (3 graded, all right) and refuted 4 (2 graded, both wrong by Wikidata).
- 11 graded promotions: fewer than the 20 S8's gate needs, so the gate cannot be read on these alone.

## Predictions, scored

| # | Result |
|---|---|
| N1 | **Partial.** C4 1.000 and C6 0 (hits); C1 0.990, C5 1.000 above their ranges; C2 0.786 and C7 0.776 far above "under 0.3" and "under 0.4" |
| N2 | **Miss.** Overall recall 0.607, not 0.25–0.5: more corrupted facts had a partner fact than expected |
| N3 | **Partial.** Naming 0.835 (hit); the baseline 0.632, just above 0.45–0.6 |
| N4 | **Hit.** 45 anomalies in store T (26 dev, 19 test), year mismatches the most common, then deaths before births; test precision 0.647; the gate missed |
| N5 | **Partial.** 257 cases, recall@1 0.568, recall@10 0.837 (hits); random order's recall@10 0.731, not under 0.5 |
| N6 | **Partial.** Beats PCA alone by +0.113 (hit); loses to the link predictor alone by 0.066 (miss) |
| N7 | **Hit.** Hits@10 0.252, MRR above the base rate, 66.8% of queries scored |
| N8 | **Hit.** λ = 0.25 |
| N9 | **Partial.** On for 2 of 14 relations (hit); dev Hits@10 0.236, 0.032 above the base rate (miss by 0.002) |
| N10 | **Hit.** 300 dimensions; the 32-dimension vectors beat the base rate nowhere |
| N11 | **Hit.** `b` Hits@10 0.050 (G4) and 0.027 (G5) |
| N12 | **Partial.** Under 20 graded per generator and the gate not met (hits); pooled precision 1.000, above 0.75–0.95 |
| N13 | **Miss.** G4 earns its place; G5 does too, narrowly |

5 hits, 6 partial, 2 misses.

## Decisions

1. **G13 is parked** by its registered rule: it names planted errors well (0.856 against 0.628), but on store T's real anomalies its error verdicts are right 0.647 of the time, no better than "fewer sources" (0.706). Its anomalies are still good questions: through the source stage, the facts it questions were decided right every time (dev 16 of 16, test 5 of 5 graded).
2. **G3 is kept.** Recall@10 0.840 on test, +0.082 over random order. Its rule term hurts at rank 1 (the link predictor alone is better); nothing is changed after the read.
3. **G4 earns its place.** On test it ranks hidden facts better than G2 (MRR 0.173 against 0.153) and finds twice the base rate's `b` facts in S15. Its hypotheses do not survive promotion: its dates are carried constants.
4. **G5 earns its place, narrowly,** on two relations (country, nationality). On every other relation it is off.
5. **Promotions:** 11 of 11 graded test promotions are true (G3 7, G5 1, G13 3), all on a DB-IB or GN positive. Too few to read S8's gate.

## For Odin

1. **G13 as a question-raiser.** Parked as a corrector, but every fact it questioned was decided right by the sources (21 of 21 graded on dev and test). Use G13's anomalies as hypotheses for the source stage only, never as verdicts? That needs registering before another read.
2. **A6 (spouse gaps)** cannot fire on this store (11 pairs with two birth years). Keep it for a larger store, or cut it?
3. **G3's order.** The link predictor alone beats G3's bits at rank 1 on dev and test. Drop the rule term from the order, read on a fresh set?
4. **G4's dates.** Every carried date was wrong. Stop G4 carrying dates as constants (λ = 0 for date relations)?
5. **S10's pools.** The four pools are in S8d's format (`a.fresh_g3`, `a.fresh_g13`, `g.fresh_g4`, `g.fresh_g5`, with their source stage and `*_ran.tsv`). S10's replay reads only G2's and G6/G7's file names; reading these is S10's change.
6. **S15's test half.** S11c read all of S15 test once (read 2) and graded its bit-17 = 0 half once (the pools). Both test halves have now been used for pools (S10: bit 17 = 1). A future fresh set needs new slots.

## How to reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn   # add -pthread on glibc < 2.34
F=/root/work/f2; E=enwik8; V=ppmi_svd_output/enwik8   # ppmi_svd's 300-dimension vectors (data/cyphalm/README.md)
# dev
./pattern_learn --gen-anomaly $E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval dev/a
./pattern_learn --gen-analogy $E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval $V data/cyphalm/simvec_ppmi32.bin dev/g
# pools (--fresh-dev or --fresh-test), their source stage, their grade
./pattern_learn --gen-anomaly … pool/a --fresh-test ; ./pattern_learn --gen-analogy … pool/g --fresh-test
S="--sources bench/eval/temporal_holdout_dev.tsv tables $F/wikidata/wd_values.tsv $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv"
./pattern_learn $S pool/src_g3 S8D_R1_DEV S8D_R2_DEV S8D_PROMO --fresh pool/a.fresh_g3.hyps.tsv empty.tsv   # likewise g13, g4, g5
./pattern_learn --pool-grade s15 bench/eval/temporal_holdout_test.tsv 0 pool/a.fresh_g3.hyps.tsv pool/src_g3.s8d_fresh.tsv
./pattern_learn --pool-grade wd $F/wikidata/wd_values.tsv -1 pool/a.fresh_g13.hyps.tsv pool/src_g13.s8d_fresh.tsv
# the test reads
./pattern_learn --gen-anomaly … test/a --read-test ; ./pattern_learn --gen-analogy … test/g --read-test
```

- `S8D_R1_DEV`, `S8D_R2_DEV` and `S8D_PROMO` are S8a's and S8b's dev hypothesis files, which S8d's calibration reads (S8d's report).
- Times on the shared machine: `--gen-analogy` 40 s, about 300 MB; `--gen-anomaly` 17 minutes (G3's 500 cases each mine the rules again), about 460 MB. Runs are in `/root/work/s11c/runs` (not committed).
- **Checks:** `native_epistemic_test` passes 1,215 checks (47 new). Five planted bugs each fail it: A1's comparison flipped, the qualifier never holding, `abduce` sorting worst first, `transfer` dropping the equality role, and G5's offset sign flipped. The other 8 CTests of HANDOFF §4.4 pass. S3's normal `--gaps` run is byte-identical (gap map `a77678d8ef7dfdef`, the cells, the demand and the v2 store files).

## S11e: S11c's questions, settled by measurement (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · answers HANDOFF §5.21 under PLAN ledger 55 (decided by measurement, open to Odin's veto); Odin: "Do all"

(The definitions, the reads, the rules and the predictions below were written and committed before any S11e code or run. Results are added under them.)

**Rule 10.** Nothing here answers a question. The generators write hypotheses for Cypha's own store.

**Data.** Dev only, unless S8e's fresh set is frozen in time (`bench/eval/s8e_slots_{dev,test}.tsv`, S8e's report). S8e's own rule says its test part is read once, by its gate. So S11e reads it only if S8e's report allows a second generator read; otherwise every result here is dev, and the fresh read waits. The dev data:
- store T's dev subjects (bit 11 = 0) for G13, graded by Wikidata;
- G3-P's dev cases on store K (S11c's construction, amendment 2);
- read 1's dev queries and read 2 on S15 dev for G4;
- the dev pools on S15 dev's bit-17 = 1 slots (S8d's slots), through S8d's source stage, graded on S15 dev's file as S11c did.

### Item 1: G13 raises questions, never verdicts

- **Now:** a questioned fact enters the source stage with P0 = 0.2 when G13's verdict names it wrong, else 0.8. So the verdict moves the prior.
- **Question-only (`--g13-questions`):** every fact of an anomaly enters the source stage with **P0 = 0.5**, whatever the verdict. Only the sources decide. G13's verdict stays a diagnostic line in its own file.
- **Measures, on dev, both modes on the same facts:** decided (supported or refuted), right decisions (supported and right by Wikidata, refuted and wrong by Wikidata), wrong decisions, supported precision, refuted share false. Wilson intervals.
- **Rule 1:** question-only is adopted if, on dev, its wrong decisions are not more than the verdict mode's, and its right decisions are at least the verdict mode's minus 1.

### Item 2: G3's order without its rule term

- **The order with a weight w:** score(f) = min over f's explanations of [−log₂ P(f) − w · log₂ PCA(rule)], ties by the rule's support, then the fact. w = 1 is S11c's G3; w = 0 orders by the link predictor alone.
- **Measured on G3-P's dev cases** (one run, the same explanations): w ∈ {0, 0.25, 0.5, 1}; recall@1, @5, @10, with bootstrap intervals of the differences from w = 1. A check: the ranks at w = 1 equal S11c's G3 ranks on every case.
- **Rule 2:** the w with the highest dev recall@1 replaces w = 1 only if its recall@1 beats w = 1's with the bootstrap interval of the difference above 0 and its recall@10 is not lower by more than 0.02. Ties go to the larger w.
- **The pool:** G3's dev pool rebuilt with the chosen w; its prior is the isotonic map from that w's score, fitted on G3-P's dev cases at that w. Source stage; promotions and precision against S11c's dev pool (11 promoted, 10 of 11 true).

### Item 3: G4 without carried dates

- **λ_date:** the weight of a value carried as a constant, for date relations (CyphaLM's `is_datey`: date and year types). S11c used λ = 0.25 for every relation.
- **Measured on dev:** read 1 with λ = 0.25 for every relation against λ = 0.25 with λ_date = 0. MRR over all dev queries, on date queries and on the rest, with bootstrap intervals of the differences; the base rate's MRR on date queries beside them. Read 2 on S15 dev (`b` and `c` Hits@10, MRR). The dev pool rebuilt with λ_date = 0: G4's hypotheses by relation, their states after the loop and the source stage, promotions.
- **Rule 3:** λ_date = 0 is adopted if the dev read-1 MRR over all queries does not fall: the bootstrap interval of the difference has its lower end at −0.005 or above.

### Item 4: A6, spouse gaps

- **Measured:** the number of spouse pairs with a birth year for both, in store K and store T. An add-one tail p of 0.005 needs at least 199 of them.
- **Rule 4:** A6 is kept, with its threshold documented, while it is correct and costs nothing. It is cut only if it fires on a store with fewer than 199 values (a bug).

### Item 5: S10 reads the new pools (report only)

- S10's replay (`--schedule`) gains `--extra-dev DIR` and `--wd FILE`: it adds S11c-format pools from DIR to the dev pool. The files are `a.fresh_g3`, `g.fresh_g4`, `g.fresh_g5` (graded on the S15 file, as S10's own) and `a.fresh_g13` (graded by Wikidata), each with its source stage (`src_<g>.s8d_fresh.tsv`, `src_<g>.s8d_fresh_ran.tsv`) and its pre-registration log. A missing generator is skipped.
- **Read, dev only:** S10's four configurations and uniform on (a) S8d's dev pool plus S11c's dev pools (the same slots: S15 dev's bit-17 = 1 half) and (b) S11c's dev pools alone. Report only: S10's gate read waits for a fresh set (S8e's, or one built for it). The test pools of S11c are not read again.
- Without `--extra-dev`, `--schedule` is unchanged byte for byte.

### Predictions (registered before any run)

| # | Prediction |
|---|---|
| E1 | Item 1: question-only decides at least as many graded G13 facts as the verdict mode (16 on dev in S11c), with at most 1 wrong decision; rule 1 adopts it |
| E2 | Item 2: w = 0 or 0.25 has the highest dev recall@1, at least 0.04 above w = 1, with recall@10 within 0.02; rule 2 replaces w = 1 |
| E3 | Item 2: the rebuilt G3 dev pool promotes within ±2 of S11c's 11, with precision ≥ 0.9 |
| E4 | Item 3: on date queries λ = 0.25 is within 0.01 of the base rate's MRR (carried dates do not help ranking); the overall dev MRR changes by less than 0.005; rule 3 adopts λ_date = 0 |
| E5 | Item 3: G4's date hypotheses in the dev pool fall from 12 to at most 3; G4's dev promotions stay 0 |
| E6 | Item 4: fewer than 30 spouse pairs with two birth years in store K, fewer still in T; A6 kept |
| E7 | Item 5: on S8d's dev pool plus S11c's, S10's chosen configuration beats uniform at f = 0.25 by more than 10%; on S11c's pools alone it does not |

### Results: S11e (2026-10-29, dev only)

**No fresh read.** S8e's fresh set was not frozen when S11e finished: its branch holds the registration only (commit 2d17454), with no `bench/eval/s8e_slots_*` files. So every result below is on dev, and the one fresh read of items 1–3 waits for S8e's set. S11c's test pools and S15 test were not read for any choice. Tool: commit 2c07b6f; the dev runs took 53 minutes and 461 MB on a machine running about twenty jobs.

**Parity.** Without the new flags every output is S11c's:
- G3-P's dev cases: G3's ranks at w = 1 equal S11c's on 240 of 240 cases with a candidate;
- the dev pools: `a.fresh_g3`, `g.fresh_g4`, `g.fresh_g5`, their logs, `a.g13p_dev.tsv`, `a.g13r.tsv` and `g.r1_queries.tsv` are byte-identical to S11c's dev files. `a.fresh_g13` differs only in its state column ("queued" against "Queued"), from S11c's own later case fix;
- S11c's G4/G5 test read (`--gen-analogy --read-test`) reproduces line for line, with its query file byte-identical;
- `--schedule` without `--extra-dev` prints S10's dev read line for line (timings aside).

#### Item 1: G13 raises questions, never verdicts

Store T's dev subjects: 52 questioned facts, 41 graded by Wikidata. The source stage as S8d calibrated it.

| Mode | Supported (right) | Refuted (wrong by Wikidata) | Decided | Right decisions | Wrong decisions |
|---|---:|---:|---:|---|---:|
| **Verdict prior** (S11c: P0 0.2 if named wrong, else 0.8) | 7 (7) | 9 (9) | 16 | **16 / 16** [0.806, 1.000] | 0 |
| Question only (P0 0.5) | 9 (8) | 0 | 9 | 8 / 9 [0.565, 0.980] | 1 |

- **Why:** a DB-IB negative moves a fact from 0.5 to 0.074, above `decide`'s refute bound (0.05). From G13's 0.2 it moves to 0.023, and the fact is refuted. So the verdict's prior is what lets the sources refute the 9 wrong years. At 0.5, two facts G13 named wrong are supported by a DB-IB positive instead, one of them wrongly.
- **Rule 1 is not met** (one more wrong decision; 8 right against 16). **G13 keeps its verdict prior.** It stays parked as a corrector: its verdict only sets the prior, and only the sources decide. That is S11c's pool as built, now measured against the alternative.

#### Item 2: G3's order without its rule term

G3-P's dev cases (S11c's construction): 257 read, 240 with a candidate.

| w (rule term's weight) | Recall@1 | Recall@5 | Recall@10 | @1 − (w = 1) | @10 − (w = 1) |
|---|---|---:|---|---|---|
| **0** (the link predictor alone) | **0.634** [0.572, 0.696] | 0.809 | 0.837 [0.790, 0.879] | **+0.066 [+0.023, +0.113]** | +0.000 [−0.012, +0.012] |
| 0.25 | 0.576 [0.514, 0.638] | 0.798 | 0.840 | +0.008 [+0.000, +0.019] | +0.004 |
| 0.5 | 0.572 [0.510, 0.634] | 0.798 | 0.840 | +0.004 [+0.000, +0.012] | +0.004 |
| 1 (S11c) | 0.568 [0.510, 0.630] | 0.798 | 0.837 [0.790, 0.879] | — | — |

- **Rule 2 replaces w = 1 with w = 0.** Rules still decide which facts are candidates; the order among them is by the fact's own bits.
- **The dev pool with w = 0** (75 hypotheses, the same facts): promoted 11, 10 of 11 true (0.909 [0.623, 0.984]), as with w = 1. Refuted 7 (4 of 4 graded false) against w = 1's 11 (4 of 5 false). Only the priors differ (the isotonic map is fitted on each order's own scores).

#### Item 3: G4 without carried dates

Read 1 (S8a's R1 random hide), dev, 1,201 queries; MRR with bootstrap intervals.

| Queries | Base rate | λ = 0.25 (S11c) | λ_date = 0 | λ_date 0 − λ 0.25 |
|---|---:|---|---|---|
| **All** (1,201) | 0.108 | 0.160 (scored 66.8%) | 0.159 (scored 51.3%) | **−0.001 [−0.004, +0.001]** |
| Date relations (263) | 0.038 | 0.099 (scored 76.4%) | 0.093 (scored 5.7%) | −0.006 [−0.017, +0.003] |
| Other relations (938) | 0.127 | 0.177 | 0.177 | 0 |

- On read 1, carried dates do help G4 rank hidden dates (+0.061 [+0.034, +0.093] over the base rate): an analog's year is often near the target's. Without them G4 still beats the base rate on dates (+0.055 [+0.030, +0.085]) through its equality and path roles.
- Read 2 (S15 dev, store T) is unchanged: `b` Hits@10 0.056, MRR 0.049; `c` Hits@10 0.075, MRR 0.038.
- **The dev pool:** 26 → 14 hypotheses. All 12 date hypotheses are gone, and all 12 were false (S11c: 12 of 12). Promotions stay 0. Refuted 25 (18 of 20 graded false) → 8 (4 of 6 false).
- **Rule 3 is met** (lower end −0.004 ≥ −0.005): **λ_date = 0 is adopted.** As a promotion source, a carried date was always wrong; as a ranker, it helps a little.

#### Item 4: A6, spouse gaps

- Store K: 1,013 spouse pairs, 9 with a birth year for both (12 directed facts). Store T: 26 pairs, 2.
- An add-one tail p of 0.005 needs at least 199 values, so A6 cannot fire on either store; it never fired. Rule 4: **A6 is kept**, its threshold documented, for a larger store.

#### Item 5: S10 reads the new pools (report only)

Dev only, f = 0.25 of the pool's cost, 1,000 draws × 5 seeds, as S10. V = confirmed value per 1,000 cost.

| Pool | Hypotheses (slots) | Uniform V | S10's four configurations (random λ 0 / random λ 1 / P0 first λ 0 / P0 first λ 1) | Dev choice, gain over uniform | FDR U / S10 |
|---|---|---:|---|---|---|
| S8d's dev pool (S10's own, reproduced) | 1,398 (1,330) | 2.790 | 4.224 / 4.244 / 6.308 / 6.324 | P0 first λ 0, **+126%** | 0.101 / 0.093 |
| S8d's + S11c's dev pools | 1,599 (1,390) | 7.234 | 8.464 / 8.477 / 9.683 / 9.740 | P0 first λ 0, **+33.9%** (above 1.1 × uniform in 0.905 of draws) | 0.077 / 0.081 |
| S11c's dev pools alone (G3, G4, G5, G13) | 201 (191) | 10.280 | 22.438 / 22.856 / 16.860 / 17.032 | random order λ 0, **+118%** (in 0.986 of draws) | 0.067 / 0.083 |

- S11c's pools are richer per cost than G2's and G6/G7's: uniform's V rises from 2.8 to 7.2 when they are added. Their value sits in a few arms (G3 places, G13 dates, G5 places).
- The combined pool by generator (supported after the sources, true among them): G2 68 (11, 10), G6/G7 1,330 (16, 14), G3 75 (11, 10), G4 26 (0), G5 48 (2, 2), G13 52 (7, 7).
- **The within-arm order flips between pools:** "highest P0 first" wins on S8d's pool and on the combination; random order wins on S11c's pools alone. S10's gate read found the same instability (P0 first won on dev and lost on the gate).
- At f = 1 with retirement off, S10 and uniform tie exactly in 200 of 200 draws on both new pools, as they must.
- **Report only.** S10's gate read waits for a fresh set; S11c's test pools were not read.

#### Predictions, scored

| # | Result |
|---|---|
| E1 | **Miss.** Question-only decided 9 (1 wrong), not ≥ 16 with ≤ 1 wrong; rule 1 keeps the verdict prior |
| E2 | **Hit.** w = 0, recall@1 +0.066, recall@10 +0.000; rule 2 replaces w = 1 |
| E3 | **Hit.** 11 promoted, 10 of 11 true (0.909) |
| E4 | **Partial.** Overall change −0.001 and rule 3 adopts (hits); but on date queries λ = 0.25 beats the base rate by 0.061, not "within 0.01" (miss) |
| E5 | **Hit.** G4's date hypotheses 12 → 0; promotions 0 |
| E6 | **Hit.** 9 pairs in K, 2 in T; A6 kept |
| E7 | **Partial.** On S8d's + S11c's pools S10 beats uniform by 33.9% (hit); on S11c's pools alone it beats it by 118%, not "does not" (miss) |

4 hits, 2 partial, 1 miss.

#### Decisions (decided by measurement under PLAN ledger 55, open to Odin's veto)

1. **G13 keeps its verdict prior.** Its anomalies are questions for the source stage; the verdict sets only the prior (0.2 or 0.8), and only the sources decide. Question-only (0.5) was measured and rejected: it loses the sources' refutations.
2. **G3 orders its added facts by the link predictor alone** (w = 0; `--g3-weights` applies rule 2's choice). Recall@1 +0.066 on dev; recall@10 unchanged.
3. **G4 carries no dates as constants** (λ_date = 0; `--g4-date-lambda`). Ranking is unchanged overall; every carried date it proposed was false.
4. **A6 is kept** for a larger store (it needs 199 spouse pairs with two birth years; this store has 9).
5. **S10 on the new pools:** report only. S10 gains more on richer pools, but the within-arm order is not stable across pools.
- The defaults are left as S11c's so that S11c's reads reproduce. The flags select the adopted settings, and the next pool read (S8e's fresh set) uses them. The one fresh read of items 1–3 happens there.

#### How to reproduce (S11e)

```bash
g++ -O2 -std=c++20 -I native/include -pthread native/tools/pattern_learn.cpp -o pattern_learn
F=/root/work/f2; E=enwik8; V=ppmi_svd_output/enwik8
./pattern_learn --gen-anomaly $E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval pdev/a --fresh-dev --g3-weights --g13-questions
./pattern_learn --gen-analogy $E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval $V data/cyphalm/simvec_ppmi32.bin pdev/g --fresh-dev --g4-date-lambda
# the source stage and the grade, as S11c (pools a.fresh_g3w, a.fresh_g13q, g.fresh_g4d beside S11c's)
./pattern_learn --schedule s10/comb --dev S10_DEV_DIR bench/eval/temporal_holdout_dev.tsv --gaps GAP_MAP --extra-dev S11C_DEV_POOLS --wd $F/wikidata/wd_values.tsv
./pattern_learn --schedule s10/only --dev S10_DEV_DIR bench/eval/temporal_holdout_dev.tsv --gaps GAP_MAP --extra-dev S11C_DEV_POOLS --extra-only --wd $F/wikidata/wd_values.tsv
```

- `S11C_DEV_POOLS` is a directory holding S11c's dev pools with their source stage (`a.fresh_g3`, `g.fresh_g4`, `g.fresh_g5`, `a.fresh_g13`, `src_<g>.s8d_fresh.tsv`, `src_<g>.s8d_fresh_ran.tsv`). Runs are in `/root/work/s11e/runs` (not committed).
- **Checks:** the nine CTests of HANDOFF §4.4 pass. `native_epistemic_test` passes 1,305 checks (12 new: the parts of an explanation's bits, w = 1 equal to S11c's order, a weight that changes the order, a fact counted at its best explanation, a rule with PCA below 1 scaled by w). Three planted bugs each fail it: the weight ignored (2 checks; the first version of the checks missed it, so the PCA-below-1 check was added), the support tie reversed (2 checks), and a fact kept at its worst explanation (1 check). S3's normal `--gaps` run is byte-identical (gap map `a77678d8ef7dfdef`, the cells, the demand and the 7 v2 store files).
