# S8b: link prediction (G6 and G7), rediscovery, promotion, and S6's next change

**Odin Loch** · 2026-10-29 · DISCOVERY §5 (G6, G7), §7.2, §12.2, §13 S8 · follows [S6](S6_HYPOTHESIS_SCORING.md), [S7](S7_HYPOTHESIS_TESTS.md) and [S15](S15_TEMPORAL_HOLDOUT.md)

(The design, the decision rules and the predictions below were written and committed before any run. Results and decisions are added under them.)

**Question.** Can Cypha's store propose facts it does not hold, by its neighbourhoods (G6) and by its relation paths (G7)? How many hidden facts do they find again, how many of their proposals survive S7's tests, and how many of those are true? And does path evidence lift S6's plausibility P0 over its 0.75 gate on a fresh source?

**S8's gate (DISCOVERY §13).** Rediscovery Hits@10 reported; promotion precision ≥ 0.9.

**S6's gate.** Known-true above known-false, AUROC ≥ 0.75 on a planted set. S6b read 0.738 [0.717, 0.758] on its fresh set, so its next change needs new evidence and a new source.

**Rule 10.** G6 and G7 propose hypotheses for Cypha's own store. They never answer a question for CyphaLM, and nothing here reaches `qa` or `chat`. S5's firewall (I1) keeps every hypothesis out of answer mode.

## The stores

- **Full store:** the v2 seed store, S6's and S7's store: 22,783 distinct triples, 3,961 subjects, 34 relations (`f2/v2/aug/seeds.tsv`). 70% of its rows came from Wikidata (D4).
- **2006-text store:** only what the 2006 text states: the seeds from infoboxes, categories and tables, plus F2's extracted facts (`facts_f2.tsv`). 8,319 triples, 2,109 subjects, 31 relations. No Wikidata row.
- S15's truth is current Wikidata. A temporal read on the full store would find facts the store copied from Wikidata, so the S15 discovery read uses the 2006-text store only.

## The generators (fixed before any run)

A query is a slot (s, r). Its candidates are every object of r in the store, minus the slot's other known objects (the filtered setting). Every generator ranks by its score; ties go to the base rate P(o | r), then by mid-rank.

| Generator | Score of candidate o for (s, r) |
|---|---|
| **Base rate** | P(o \| r), S6's `base_rate` |
| **LP** | S6's link predictor `LinkPredictor::prob(s, r, o)` (pairwise and copy rules, m = 1), computed for all candidates in one pass (`LinkScorer`; equal to `prob` to 1e-12, checked) |
| **G6** | neighbourhood of s and o in the entity graph: common neighbours (CN), Jaccard, Adamic–Adar (AA, `Σ 1/ln deg z`) and resource allocation (RA, `Σ 1/deg z`, the degree-normalised form). G6's score is the one of the four with the best dev MRR |
| **G7** | path ranking (PRA). Relation paths of length 1–3 over typed edges in both directions (r and r⁻¹), the slot's own edge left out. Paths with support ≥ 3 among up to 300 training facts of r, top 40 by support × 0.5^(length − 1). Feature: log(1 + 1000 × random-walk probability from s to o along the path). Logistic regression with L2 = 1, on up to 300 facts of r per relation (each with its own edge removed) and 4 negatives each, drawn per triple from r's objects. Hub guard: a walk does not expand an intermediate node with more than 1,000 edges of the step's relation. Each prediction carries its strongest path (weight × feature) as its explanation |
| **Combination** | logistic regression over [logit LP, log(1 + RA), G7's logit, log base rate], trained on dev queries (each true object against 20 drawn candidates); its dev MRR by two-fold cross-fitting |

## The reads

1. **Rediscovery (DISCOVERY §12.2), full store.**
   - *Random:* hide 10% of the triples (seed 20261029). The rest is the reduced store every generator and test reads.
   - *Least connected:* separately, hide the 10% of triples with the lowest deg(s) + deg(o) in the entity graph (ties by FNV-1a).
   - Subjects split by bit 17 of FNV-1a(subject): dev and test. Test is printed only with `--read-test`, read once, for the final setting.
   - Hits@1, Hits@10 and MRR per hidden fact, overall and per relation, with 2,000 bootstrap resamples over hidden facts. A hidden object that no longer occurs as an object of r is unreachable and counts as a miss; the reachable share is reported, and the scores among reachable facts too.
2. **Promotion (S8's gate), on read 1's random hide.**
   - One hypothesis per hidden slot: the chosen generator's top candidate.
   - S7's loop: pre-registration in `PreregLog`, tests cheapest first, Bayes factors from S7's dev point estimates (T1 consistency s 0.7551 c 0.9184; T6 compression, ε 0, s 0.2553 c 0.9493; T4 corpus, pattern or co-mention, s 0.9933 c 0.8137), `decide`, then BH at q 0.1 over the batch.
   - **T5 (Wikidata) is left out**: Wikidata is the truth here, and the store took 70% of its triples from it. T4 is then the only test independent of the generator (I4).
   - Prior: the generator's top score, isotonic-fitted on dev to P(top candidate true), clamped to [0.02, 0.98].
   - Promoted = supported by `decide` (posterior > 0.9 and a positive independent result) and kept by BH.
   - Truth: true if the candidate is a hidden fact of the slot, or Wikidata (`wd_values.tsv`) lists it for (s, r); false if Wikidata has values for (s, r) and none match; otherwise unjudged (counted and reported).
   - *Amended before any run (2026-10-29):* S15 is not used to judge this read. Its test half must stay unread until read 3's single logged read, and for the store's relations it holds the same current Wikidata as `wd_values.tsv`.
   - **Promotion precision** = true / judged among the promoted, with a Wilson interval. The precision of "supported by `decide`" before BH is reported beside it. The gate needs at least 10 judged promotions on test.
3. **Temporal holdout (S15, T9), 2006-text store.**
   - Queries: S15 slots (s, r) with r among the store's relations, s in the store, and no (s, r) fact in the store.
   - Hits@1, Hits@10 and MRR of S15's true values by category (a stated in 2006, b true by 2006 but unstated, c true only after 2006-03-04, u unknown when), overall and among reachable values; values match by S7's rules (dates by year or compatible date, others by normalised name).
   - Promotion through the same loop (T1, T6, T4; no T5), judged by S15: true if an S15 true value of the slot, false otherwise (S15 covers every queried slot).
   - S15's dev half for development; its test half read once and logged in S15's "Reads" table.
4. **S6's next change (S6's gate), S15's true and false pairs, full store.**
   - Per S15 subject, one pair drawn by seed among the eligible: r among the store's relations, s with at least one other fact once the slot is gone, both values known as objects of r in the reduced store. The slot (s, r, ·) of every drawn pair is removed from the store.
   - **S6b's P0:** isotonic(z(logit LP) + z(log(1 + AA))) × the hard constraint.
   - **The one change, P0′:** isotonic(z(logit LP) + z(log(1 + AA)) + z(G7's logit)) × the hard constraint.
   - z pools and isotonic fits on S15's dev pairs; S15's test pairs read once. Pair AUROC with 2,000 bootstrap resamples.
   - **S6's gate is met if P0′ ≥ 0.75 on test.** S6b's P0 on the same test pairs is the baseline.

## Decision rules (fixed before any run)

1. **G6's measure:** the best dev MRR among CN, Jaccard, AA and RA on read 1's random hide.
2. **The generator for reads 2 and 3:** the best dev MRR among base rate, LP, G6, G7 and the combination on read 1's random hide.
3. **A generator earns its place** if its dev MRR beats the base rate with the bootstrap interval of the difference above 0, or if another part uses it (G7's paths explain the combination's proposals). Otherwise its code goes and the report keeps its numbers.
4. **S8's gate:** promotion precision ≥ 0.9 on read 2's test half, with at least 10 judged promotions; fewer is "not measurable", and says so.
5. **S6's gate:** read 4, P0′ ≥ 0.75 on test.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| L1 | Random hide, dev: base rate Hits@10 0.25–0.45; the best generator 0.45–0.70, MRR 0.25–0.45 |
| L2 | The best single generator is G7 or the combination; among G6's measures RA ≥ AA ≥ CN |
| L3 | Least connected: every generator's Hits@10 is under half its random-hide value; under 60% of those hidden facts are reachable |
| L4 | Dates (birth_year, death_year, birth_date, death_date, founded): no generator beats the base rate's Hits@10 by more than 0.02 |
| L5 | Promotion: under 5% of hidden slots are promoted; if at least 10 are judged, precision ≥ 0.9 |
| L6 | Temporal read: under 15% of S15's b and c values are reachable in the 2006-text store; among reachable b and c values, the chosen generator's Hits@10 is 0.3–0.7 |
| L7 | S6's next change: P0′ on S15 test 0.74–0.80, P0′ − P0 between +0.005 and +0.04; the gate is met |
| L8 | Cost: the whole mode runs under 10 minutes, peak under 1 GB |

## Results (runs 2026-10-29)

Tool: `pattern_learn --gen-links` at commit d50fdcd (code committed before the test read). A run takes 16–24 s, peak 388–538 MB. The test halves were read once, in one run (`--read-test`).

### The stores and hides

- Full store: 22,783 triples. Each hide removes 2,278 (10%): the random one in 1,992 slots, the least-connected one in 2,024.
- 1,448 of the random hide's 2,278 hidden objects are still candidates (64%); only 231 of the least-connected (10%).
- 2006-text store: 8,319 triples, 2,109 subjects. S15 gives it 2,296 dev and 2,347 test slots it lacks.

### Read 1: rediscovery (full store, random hide)

| Generator | Dev Hits@1 | Dev Hits@10 | Dev MRR | Test Hits@10 | Test MRR |
|---|---:|---:|---:|---:|---:|
| Base rate | 0.057 | 0.191 [0.168, 0.215] | 0.103 | 0.176 [0.154, 0.200] | 0.098 |
| LP (S6) | 0.153 | 0.327 | 0.211 | 0.301 | 0.188 |
| G6 CN | 0.068 | 0.201 | 0.113 | 0.187 | 0.111 |
| G6 Jaccard | 0.068 | 0.200 | 0.112 | 0.183 | 0.104 |
| G6 AA | 0.069 | 0.202 | 0.1134 | 0.187 | 0.112 |
| G6 RA | 0.069 | 0.202 | 0.1134 | 0.187 | 0.112 |
| G7 paths | 0.168 | 0.338 | 0.225 | 0.321 | 0.200 |
| **Combination** | **0.170** | **0.350 [0.323, 0.377]** | **0.227 [0.206, 0.250]** | **0.334 [0.307, 0.362]** | **0.203 [0.184, 0.224]** |

- Dev picks RA as G6's measure (it ties AA to four digits) and the combination as the generator. The combination's dev numbers are cross-fitted on two folds.
- Its weights, fitted on all of dev: logit LP 0.06, log RA 0.39, **G7 1.08**, log base rate 0.45. Path evidence carries it.
- Over the base rate on dev MRR: LP +0.108 [0.089, 0.126], G6 RA +0.010 [0.003, 0.019], G7 +0.121 [0.102, 0.140], combination +0.124 [0.106, 0.143]. All four earn their place.
- **By relation (dev, combination Hits@10):** occupation 0.60 (base 0.27), nationality 0.69 (0.40), country 0.56 (0.44), genre 0.43 (0.11), located_in 0.28 (0.08), death_place 0.27 (0.16).
  - Entity-valued relations with one value per entity (spouse, predecessor, successor, creator, founder) score 0: their hidden objects are never candidates.
- **Dates** (birth and death year and date, founded; 241 dev and 246 test facts): every generator's Hits@10 sits at 0.04–0.08. Base rate 0.050 dev, 0.053 test; LP 0.079 dev, 0.077 test.
- **Least connected:** 10% reachable, so every generator's Hits@10 is 0.002–0.021 on dev and test (LP and G7 best at about 0.02).

### Read 2: promotion through S7's loop (random hide)

| | Dev | Test |
|---|---|---|
| Hypotheses (top candidates) | 999 | 993 |
| Top candidate true, among judged | 195 / 881 (0.221) | 169 / 887 (0.191) |
| Supported by `decide` | 55: 36 / 53 true (0.679 [0.545, 0.789]) | 56: 31 / 52 true (0.596 [0.461, 0.718]) |
| **Promoted after BH (q 0.1)** | **0** | **0** |
| Refuted | 391: 334 / 344 false (0.971) | 328: 295 / 299 false (0.987) |

- **S8's promotion gate is not measurable: BH promotes nothing.** With T5 out, the largest e-value a hypothesis can reach is T1 × T6 × T4 = 9.2 × 5.0 × 5.3 ≈ 250 (p 0.004). BH over about 1,000 hypotheses needs some 40 of them that strong; only a handful get three positive tests. This is the batch-control strictness S7 reported.
- Before BH, "supported" is right 60–68% of the time, under the 0.9 the gate asks. Refutation is reliable: 97–99% of judged refutations are false facts.
- Example supported hypotheses (test), with G7's path: *Aalborg country Denmark* (located_in), *Alban Berg death_place Vienna* (birth_place), *Baker Island country United States* (located_in > located_in⁻¹ > country). Wrong ones come from coarse paths such as "people with the same death date died in the same place".

### Read 3: the temporal holdout (S15, 2006-text store)

Reachable share of S15's true values (candidates are the store's objects of r):

| Category | Dev | Test |
|---|---:|---:|
| a, stated in 2006 | 274 / 1,331 (0.21) | 234 / 1,266 (0.18) |
| **b, true by 2006, unstated** | **64 / 870 (0.074)** | **82 / 949 (0.086)** |
| **c, true only after 2006-03-04** | **13 / 135 (0.096)** | **10 / 165 (0.061)** |
| u, unknown when | 568 / 1,724 (0.33) | 635 / 1,843 (0.34) |

Hits@10 of the combination (the chosen generator), all targets and among the reachable:

| Category | Dev, all | Dev, reachable | Test, all | Test, reachable |
|---|---:|---:|---:|---:|
| a | 0.126 | 0.613 | 0.112 | 0.607 |
| b | 0.035 | 0.469 (n 64) | 0.037 | 0.427 [0.329, 0.537] (n 82) |
| c | 0.074 | 0.769 (n 13) | 0.042 | 0.700 [0.400, 1.000] (n 10) |
| u | 0.189 | 0.574 | 0.202 | 0.586 |
| all | 0.132 | 0.581 | 0.132 | 0.579 [0.547, 0.609] |

- The base rate's test Hits@10 among reachable values is 0.482. On the test half, G7 and the combination find unstated (b) and future (c) facts at about 0.43–0.70 when the value is a candidate at all.
  - Example: Aleksandr Solzhenitsyn's death place, Moscow (he died in 2008), comes first by "nationality > nationality⁻¹ > birth_place".
- Date values match by S7's rule, so a year-only candidate ("1736") counts for a full date ("1736-07-01"). Some b and c date hits are only a year.
- **Promotion on S15:** of 2,347 test slots, the top candidate is right for 221 (0.094); 82 are supported (43 true, 0.524), and BH again promotes none. 1,012 are refuted, 93% of them rightly.

### Read 4: S6's next change on S15's pairs

- Pairs drawn: 2,442, one per subject. Kept: 2,049 (997 dev, 1,052 test) once both values were known objects of r in the store with every drawn slot removed (16,298 of 22,783 triples left).
- Removing a drawn slot removes all its values, and many slots (occupation, genre) are multi-valued. That is why the store shrinks by 28%.

| Score | Dev AUROC | Test AUROC |
|---|---:|---:|
| S6b's P0 | 0.668 [0.649, 0.687] | 0.656 [0.637, 0.675] |
| **P0′ (+ G7)** | **0.666 [0.648, 0.684]** | **0.658 [0.640, 0.675]** |
| LP alone | 0.640 | 0.639 |
| AA alone | 0.530 | 0.528 |
| G7 alone | 0.562 | 0.560 |
| Hard constraint alone | 0.541 | 0.544 |

- P0′ − P0: dev −0.002 [−0.011, +0.007], test +0.002 [−0.008, +0.011]. **S6's gate is missed** on this source.
- S15's false values are drawn among the same type's values, so they are harder to tell apart than S6's (drawn from all of r's objects): S6b's P0 reads 0.656 here against 0.738 on its own fresh set.
- (The dev-only run, with only dev slots removed, read P0 0.693 and P0′ 0.693 on 1,111 dev pairs.)

### Predictions, scored

| # | Result |
|---|---|
| L1 | **Miss.** Base rate Hits@10 0.191 (predicted 0.25–0.45); best 0.350 (0.45–0.70); MRR 0.227 (0.25–0.45). A third of the hidden objects are not candidates at all |
| L2 | **Hit.** The combination is best (G7 a close second); RA ≥ AA ≥ CN (0.1134, 0.1134, 0.1128) |
| L3 | **Hit.** Least connected: Hits@10 under 0.03 for every generator; 10% reachable |
| L4 | **Miss.** LP beats the base rate on dates by 0.029 (dev) and 0.024 (test), more than 0.02, though every value is under 0.08 |
| L5 | **Half.** No slot is promoted (under 5%); the precision half cannot be read |
| L6 | **Hit.** b and c reachable 6–10%; among reachable b and c on test, Hits@10 0.457 (42 of 92) |
| L7 | **Miss.** P0′ 0.658 (predicted 0.74–0.80); the gain +0.002 (+0.005 to +0.04); the gate is missed |
| L8 | **Hit.** 24 s, 538 MB |

## Decisions

1. **G6 is resource allocation** (dev MRR 0.1134, equal to AA to four digits). G6 earns its place, but barely (+0.010 MRR over the base rate). Its four measures share one function, so the code stays small.
2. **The generator is the combination** of LP, RA, G7 and the base rate; G7 carries most of its weight.
3. **All four generators earn their place** on dev (each beats the base rate with the interval above 0).
4. **S8's gate is not measurable** on test: the loop promotes nothing under BH at q 0.1. Hits@10 is reported (0.334 test). Before BH, supported precision (0.60 test) is under 0.9, so a looser batch rule alone would not meet the gate.
5. **S6's gate stays missed:** path evidence adds nothing measurable to P0 on S15's pairs (+0.002 [−0.008, +0.011]).
6. **No rule-10 question arose.** The generators write hypotheses and test rows only.

## For Odin

- **Batch control blocks every promotion** once T5 is out. S7B is calibrating it. With the tests Cypha has, even a looser rule would promote at about 0.6 precision, so S8's gate needs stronger independent evidence than T4, not only a looser rule.
- **Closed candidate sets cap discovery.** Only 6–9% of S15's unstated and future values are objects the store already holds. Typed page titles (every place page as a candidate for birth_place, and so on) would open it up. That is a design choice for the next change.
- **S6's gate on S15:** the hard negatives of S15 pull P0 to 0.66. Should S6's gate be read on S15's pairs from now on (harder, independent of the store), or kept on store-drawn pairs?

## How to reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
F=/root/work/f2   # D4's and F2's run outputs (see the D4 and F2R reports)
./pattern_learn --gen-links enwik8 $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv \
  $F/v2/aug/patterns.tsv bench/eval out                 # dev only
./pattern_learn --gen-links enwik8 $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv \
  $F/v2/aug/patterns.tsv bench/eval final --read-test   # the single test read (done 2026-10-29)
```

It writes `OUT.rod.tsv` (every hidden fact's rank under each generator), `OUT.promo.tsv` and `OUT.s15.tsv` (each hypothesis with its tests, posterior, state, truth and G7 path), `OUT.gate.tsv` (the pairs' scores) and the pre-registration logs (`*.prereg.tsv`).
