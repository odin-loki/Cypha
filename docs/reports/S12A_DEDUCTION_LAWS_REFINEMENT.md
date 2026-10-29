# S12a: deep deduction (G1), numeric laws (G15), refinement (G19) and the wild tier (G20)

**Odin Loch** · 2026-10-29 · DISCOVERY §5 (G1, G15, G19, G20), §8 (P6, P9), §9, §13 S12 · follows [S8A_RULES_AND_CONTRADICTIONS.md](S8A_RULES_AND_CONTRADICTIONS.md), [S9_WORLDS_AND_FRAGILITY.md](S9_WORLDS_AND_FRAGILITY.md), [S11C_ANALOGY_ANOMALY_ABDUCTION.md](S11C_ANALOGY_ANOMALY_ABDUCTION.md) and [S10_CURIOSITY_SCHEDULER.md](S10_CURIOSITY_SCHEDULER.md)

(The design, the reads, the decision rules and the predictions below were written and committed before any S12a run. Results are added under them.)

**Question.** The first part of S12, generators batch 3. G1 chains G2's rules further than S9's worlds do. G15 searches numeric attributes for laws. G19 refines G2's rules. G20 makes the wild tier. Does G15 rediscover laws that were planted, without finding laws that are not there? What do the other three add?

**S12's gate (DISCOVERY §13).** Planted-law rediscovery, and a prospective log. G17 (causal) and G18 (trends) are not part of this step: the forecasting module's event vocabulary is being reworked (FCAST).

**Rule 10.** The generators write hypotheses and laws into Cypha's own store. They never answer a question, and nothing here reaches `qa` or `chat`.

## What was counted before this was written (sizes only)

- Store K (every seed row): 1,278 subjects with a birth year (from `birth_year` or `birth_date`), 966 with both a birth and a death year, 862 with `founded`. Store T: 1,161, 874 and 115.
- Links between dated entities are rare: at most 13 subjects have a dated spouse, successor, predecessor or founder.
- So no type in the store meets DISCOVERY P9's trigger (at least 30 entities with at least 3 numeric attributes).
- The 2006 text has numeric infobox fields: about 50 country pages carry `area`, `areami²`, `population_estimate`, `density`, `population_densitymi²`, `GDP_PPP`, `GDP_PPP_per_capita` and `HDI`; about 30 place pages carry `population_total`, `area_total`, `area_land`, `area_water` and `elevation`. No law was computed while counting.

## Shared setup

- **Stores** as S8a and S11c: K (every seed row) and T (seed rows not from Wikidata, plus F2's admitted facts), loaded by S11c's `load_stores`.
- **Subject split** for rediscovery reads: dev when bit 11 of FNV-1a(subject) is 0 (S8a's R1 split); test otherwise. Dev is read freely; test once, with `--read-test`.
- **The loop, the source stage, the pools and the grader** are S11c's: T1, T6 and T4 with S7's dev point estimates, `PreregLog`, `decide` with S8d's fix; then `--sources … --fresh` with S8d's calibration; promoted = supported by `decide` after the source stage; graded with S8c's grader on S15 (`--pool-grade`).
- **Intervals:** Wilson 95% on proportions; 2,000 bootstrap resamples (seed 20261029) on differences.

## G15: numeric laws (`native/include/cypha/eval/gen_laws.hpp`)

**Tables.** A table is one type: entities by numeric attributes. Each attribute has a scale:
- **ratio** (positive quantities: areas, populations, money, densities): power products are allowed;
- **interval** (years): only differences of two attributes are allowed. A ratio of two years is never a law (the units check).

**The search (BACON-style).** BACON tries X/Y when X and Y rise together and X·Y when one falls as the other rises, then repeats on the new term. That walk reaches exactly the integer power products. G15 enumerates them directly, so no law depends on the walk's order:
- ratio attributes: every pair with exponents |a|, |b| ≤ 3, and every triple with exponents |a|, |b|, |c| ≤ 2; exponents with greatest common divisor 1, the first one positive;
- interval attributes: every pair difference Y − X; and every ordered pair as an order law (X ≤ Y);
- a formula is searched only on entities holding all its attributes, and only when there are at least 30 of them.

**A law is kept when all of these hold:**
1. **Nearly constant** (DISCOVERY: coefficient of variation < 0.1). For a power product the robust CV (1.4826 · MAD / |median|) of the term is below 0.1, and at least 80% of entities lie within 10% of the median. For a difference Y − X, its robust spread is below 0.1 of the smaller robust spread of X and Y, and at least 80% of entities lie within 3 robust spreads of its median. (Robust: a few typing errors in an infobox must not hide a law; the classic CV is printed too.)
2. **Not chance** (DISCOVERY: BH across candidate formulas). Under independent attributes the variance of log T would be Σ eᵢ² var(log xᵢ) (for a difference, var X + var Y). The p-value is the χ² lower tail of (n − 1) · observed / expected variance (robust variances). Benjamini–Hochberg at q = 0.1 over every formula searched in the table.
3. **Shortest** (DISCOVERY: MDL). Formulas are taken in order of length (attributes, then Σ|eᵢ|). A formula whose exponent vector lies in the span of laws already kept in the table is redundant and dropped.
4. **Held out** (T3). The constant is the median of the term on two thirds of the entities (S8a's hidden third, FNV-1a bits 13+ mod 3 = 1, is held out). On the held-out third at least 80% of entities must lie within the tolerance of rule 1. Too few held-out entities (under 5): the law is kept and marked untested.
- **Order law** X ≤ Y: at least 98% of the entities holding both, Wilson lower bound ≥ 0.95, at least 30 entities; held-out violations at most 5%.

### The gate: planted-law rediscovery

**Tables for planting** (one per replicate, 40 entities):
- **Ratio laws:** synthetic entities with independent attributes X, Y, W1, W2 (each log-normal, log-mean drawn per attribute from N(log 100, 1), log-SD 1), plus Z planted. 5% of the entities get a Z drawn independently (typing errors).
- **Interval laws:** 40 persons of store K with a birth and a death year (dev persons for dev replicates, test persons for test replicates), their real birth and death years, a distractor year W drawn uniformly over the range of their birth years, plus Z planted from the birth year. The 5% contamination applies too.

**The planted laws** (constant c drawn per replicate):

| Kind | Planted | Found when G15 keeps |
|---|---|---|
| RATIO | Z = c · X / Y | the power product Z¹X⁻¹Y¹ |
| PRODUCT | Z = c · X · Y | Z¹X⁻¹Y⁻¹ |
| POWER (Kepler's third law) | Z = c · X^1.5 | Z²X⁻³ |
| SCALE (a unit conversion) | Z = c · X | Z¹X⁻¹ |
| OFFSET | Z = birth + c, c ~ U(10, 60) years | the difference Z − birth |
| ORDER | Z = birth + |N(25, 10)| years | the order law birth ≤ Z |

- Ratio constants c ~ U(0.5, 20). A ratio law is found when the kept power product has the planted exponents (up to sign) and its constant is within 10% of the planted one; OFFSET when the kept difference's constant is within 10% of the noise scale (or 1 year at noise 0) of c.
- **Noise:** σ ∈ {0, 0.02, 0.05, 0.10, 0.20}. Ratio laws: Z times exp(N(0, σ)). Interval laws: Z plus N(0, σ · robust spread of the birth years).
- **Replicates:** 20 per kind and noise level, per split (seeds by split, kind, level and replicate).
- **The null:** each planted table again with Z shuffled across entities (the law broken, the margins kept).
- **A false law:** in a null table, any kept law that involves Z; in any table, any kept law among the independent attributes (X, Y, W1, W2; W with the persons' years). True laws among real attributes (death after birth, for example) are not false.

**Decision rule (S12's gate, read once on test replicates):**
- **Recall ≥ 0.90**, pooled over the six kinds at noise σ ≤ 0.05 (360 planted tables), and
- **false-law rate ≤ 0.05**: the share of null tables with at least one false law (360 null tables at σ ≤ 0.05).
- Both are point estimates, with Wilson intervals printed. Recall at σ = 0.10 and 0.20 is reported, not gated.
- Nothing is chosen on dev: the thresholds are DISCOVERY's or fixed above. Dev runs first to check the code; the test replicates are read once.

### G15's real read (reported, not a gate; one run, nothing chosen)

- **Countries** (2006 infoboxes, enwik8): `area` (km²), `areami²`, `population_estimate`, `density` (`population_density`), `population_densitymi²`, `GDP_PPP`, `GDP_PPP_per_capita`, `HDI`, all ratio. Money read with "million", "billion" and "trillion" (also as a wiki link) and without "$" and commas.
- **Places** (enwik8 infoboxes with `population_total`): `population_total`, `area_total`, `area_land`, `area_water`, `population_density`, `elevation` (elevation is interval: it can be zero or negative).
- **Persons** (store K and store T): birth and death years, interval.
- **The laws a reader would expect** (written now): density ≈ population / area; areami² ≈ area / 2.59; population_densitymi² ≈ 2.59 · density; GDP_PPP_per_capita ≈ GDP_PPP / population; death ≥ birth; a bounded lifespan (death − birth). Every kept law is printed, expected or not.

## G1: deep deduction (`gen_deep.hpp`, S9's worlds)

- **What it is:** forward chaining of the kept G2 rules inside an S9 world, from every fact of the store, up to depth 6 (S9 stops at 3; answer mode chains nothing). A fact first derived at depth d keeps its premises and its chain confidence, the product of the rules' PCA confidences along the chain (DISCOVERY: prior = product of premise P and rule reliabilities). Depth 1 is G2's own prediction. **G1's hypotheses are the facts first derived at depth ≥ 2** (chains through Believed premises).
- **Read 1, rediscovery on store K** (S8a's R1 random hide, rules mined on the reduced store): derived facts by depth; precision with S8a's amended grader (true when the full store K or S15 lists the value; false when S15's false row has it, or S15 lists values for the slot and none matches; else ungraded). Dev subjects freely, test once.
- **Read 2, store T** (S15's temporal read): facts derived for S15 subjects, by depth, graded by S15 (dev freely, test once), by category.

## G19: refinement (`gen_deep.hpp`)

- **Specialisation:** every G2 candidate rule with support ≥ 10 (kept or not) gets one condition atom r′(X, c′) on its subject, chosen among the constant facts held by at least 10 of the rule's supporting subjects. Beam: the 3 best conditions by the refined PCA confidence.
- **Kept** when the refined rule has support ≥ 10, PCA ≥ 0.5, PCA at least 0.1 above its parent's, and passes Benjamini–Hochberg at q = 0.1 over all refinements (the parent's p₀, binomial as G2).
- **Generalisation** (constants to variables) is not built: G2 already mines the variable rules (same, inverse, path) directly. Said, not built.
- **Read** (S8a's R1 random hide): every prediction scored by its rule set's noisy-or. Precision of the top N predictions, N = the number of graded G2 predictions on the split, for G2 alone and for G2 plus G19's kept refinements (equal support: the same N). Gain with a bootstrap interval over predictions. **G19 earns its place** if the test gain's interval is above 0.

## G20: the wild tier (`gen_deep.hpp`)

- **Recombination:** for a slot (s, r), a value o drawn at random from the store's objects of r whose cluster differs from s's. A cluster is (type, country): type by relations (person: birth, death or occupation facts; place: located_in, country or capital; else other), country by `country` or `nationality`, else none.
- **Kept** when S6's link predictor scores (s, r, o) at least twice the base rate P(o | r) (statistical support better than chance).
- **Budget:** 5% of the main pool (S10's wild floor), at least 20 hypotheses, drawn with mt19937_64(20261029).
- **Tests:** the cheapest only: T1 and T6 (store) and T4 (the subject's page), then the source stage.
- **Read:** the confirmed rate (promoted and graded true, over graded) against the main tier's (G1 and G19), Wilson intervals.

## The prospective log

Every pool hypothesis (main and wild) is written with its UTC time, generator, tier, prior, state after the loop and the check registered for it: graded when a later source (a later Wikidata, S8e's slots, or Odin) lists its slot. The log's sha256 is printed and recorded here before any grade is read.

## Pools and the final reads

- **Pools:** S11c's construction. The slots of half of S15 (dev: S15 dev with bit 17 = 1; final: S15 test with bit 17 = 0); every store K triple in those slots hidden; G1, G19 and G20 on the reduced store; the top hypothesis per slot for G1 and G19 (by score), and G20's budget. Then the loop, the source stage and `--pool-grade`.
- **Written in S8d's fresh-set format** (`*.hyps.tsv`, the pre-registration log, the source stage's `*.s8d_fresh.tsv` and `*.s8d_fresh_ran.tsv`), for S10.
- **Why S15 and not S8e's slots:** S8e's fresh set is not frozen yet, and its registration reserves its test part for S8e's own gate read. S15's rule 3 allows each generator one final read; these are G1's, G19's and G20's first. The read is logged in S15's Reads table.
- S8's gate (precision ≥ 0.9 on at least 20 graded promotions) is the reference, not this step's gate.

## Amendment 0 (2026-10-29, before any run)

Checking the registered rules against the planted design, before any code ran, showed three that fail by construction. They are fixed here; nothing else changes.
1. **Order laws could never pass.** With 40 entities and 5% contamination, about 2.5% of pairs violate any order, so "98% with a Wilson lower bound of 0.95" fails even with no noise (a Wilson bound of 0.95 needs about 75 entities with no violation). **Now:** an order law X ≤ Y is kept when at least 95% of the entities holding both have X ≤ Y, its binomial p-value against a coin (P = 0.5) passes the same Benjamini–Hochberg at q = 0.1 as the formulas, there are at least 30 entities, and held-out violations are at most 10%.
2. **ORDER's gap did not scale with its noise.** **Now:** Z = birth + |N(0.25, 0.10)| · S, S the robust spread of the replicate's birth years (the noise is scaled by S too).
3. **OFFSET's tolerance was tighter than a median allows.** **Now:** OFFSET is found when the kept difference's constant is within 3 standard errors of the median (3 · 1.2533 · noise SD / √n) of c, or within 0.5 year at noise 0.

## Amendment 1 (2026-10-29, after the dev read, before any test read)

The dev read showed that amendment 0's null for order laws is wrong. A coin (P = 0.5) is not the null of "no relation": two attributes with shifted margins (a distractor year W against a later year Z) give W ≤ Z for most entities with no link between them, and such orders passed in null tables. **Now:** an order law's p-value is the binomial upper tail of its count against p₀, the share of X ≤ Y over all pairs of different entities (the independence null, as for the formulas). Nothing else changes: thresholds, contamination, noise, replicates and the gate are as registered.
- G15's real read had been run once under amendment 0. It is run again under amendment 1, and both are reported.
- **Noted on dev, not changed:** POWER is not found at σ = 0.05 (the term Z²X⁻³ doubles Z's noise, so its CV sits at the 0.1 bar), and ORDER's recall is capped near 0.7 by the registered contamination (a mistyped year almost always breaks the order, and Bin(40, 0.05) often gives 3 or more). Both follow from the registered design; neither is tuned.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| L1 | Planted gate on test: recall at σ ≤ 0.05 ≥ 0.90 pooled; each kind ≥ 0.85 |
| L2 | Recall at σ = 0.10 between 0.3 and 0.8; at σ = 0.20 below 0.10 |
| L3 | False-law rate on null tables ≤ 0.02; no false law among the independent attributes in more than 2% of tables |
| L4 | Countries: at least 3 of the 4 expected laws kept (density, the two unit conversions, GDP per capita) |
| L5 | Persons (K and T): death ≥ birth kept; the lifespan difference kept |
| L6 | Countries and places: at most 1 kept law outside the expected list |
| D1 | G1 on R1: 50–500 facts first derived at depth ≥ 2; precision at depth 2 within 0.15 of depth 1's |
| D2 | G1 on store T: fewer than 20 facts at depth ≥ 2 for S15 subjects |
| R1 | G19 keeps 10–100 refinements; test gain at equal N between 0.00 and +0.05, interval may include 0 |
| W1 | G20's confirmed rate below the main tier's; at most 1 wild promotion |
| P1 | Pools: fewer than 20 graded promotions per generator; S8's gate cannot be read |

## Dev results (2026-10-29)

Tool: `pattern_learn --gen-laws` and `--gen-deep` at commit e84fcb6 (amendments 0 and 1). Everything here is dev or the real read; no test replicate or test label had been read.

### G15, planted dev replicates (store K's dev persons: 480)

Under amendment 1. Found = the planted law kept with its constant within tolerance; 20 replicates per cell, Wilson 95%.

| Kind | σ 0 | σ 0.02 | σ 0.05 | σ 0.10 | σ 0.20 |
|---|---|---|---|---|---|
| RATIO | 20/20 | 20/20 | 18/20 | 0/20 | 0/20 |
| PRODUCT | 19/20 | 20/20 | 18/20 | 0/20 | 0/20 |
| POWER | 20/20 | 20/20 | 0/20 | 0/20 | 0/20 |
| SCALE | 20/20 | 20/20 | 18/20 | 0/20 | 0/20 |
| OFFSET | 19/20 | 18/20 | 19/20 | 5/20 | 0/20 |
| ORDER | 12/20 | 15/20 | 13/20 | 8/20 | 0/20 |

- **Pooled at σ ≤ 0.05: 309 / 360 = 0.858 [0.819, 0.891].** Below the gate's 0.90 on dev.
- **False laws:** 0 of 360 null tables (amendment 0's coin null had 4 of 360); 1 of 360 planted tables has a false law.
- **Why POWER fails at σ = 0.05:** the term Z²X⁻³ carries twice Z's noise, so its CV sits at the 0.1 bar.
- **Why ORDER stays near 0.7:** the 5% contamination draws a year over the whole birth range, which nearly always breaks the order; Bin(40, 0.05) gives 3 or more such entities about a third of the time, and then fewer than 95% hold.
- **At σ = 0.10 the CV bar (0.1) is the noise itself**, so ratio laws are lost; OFFSET keeps 5 of 20 and ORDER 8 of 20.

### G15's real read (run twice: under amendment 0, then under amendment 1; the same laws both times)

| Table | Entities | Kept | Laws |
|---|---:|---:|---|
| Countries (2006 infoboxes) | 54 | 3 | **area_km² / area_mi² = 2.58999** (robust CV 0.0000, 36 countries); **density_km² / density_mi² = 0.3861** (CV 0.0013); **area_km² · density_km² / population = 0.990** (CV 0.025, 53 countries, all 13 held out within 10%) |
| Places (with `population_total`) | 29 | 0 | none searched: no formula has the 30 entities it needs |
| Persons, store K | 1,307 | 1 | birth ≤ death (962 of 962) |
| Persons, store T | 1,209 | 1 | birth ≤ death (861 of 863) |

- The three density forms through square miles are found and dropped as **redundant**: they follow from the kept laws (MDL).
- **Not kept:** GDP per capita = GDP / population (its term is not nearly constant: the infobox's GDP and population are often from different years); the lifespan (death − birth: median 66 years, robust spread 19.3, against 161 and 180 for birth and death: 0.12, above the 0.1 bar).
- **Nothing kept outside the expected list.**

### G1 on R1 (store K, dev subjects; S8a's amended grader)

| Depth | Derived | Mean chain confidence | Graded | Precision | Non-lexical |
|---:|---:|---:|---:|---|---|
| 1 (G2's own) | 1,693 | 0.736 | 515 | 0.503 [0.460, 0.546] | 22/70 = 0.314 |
| 2 | 406 | 0.541 | 138 | 0.232 [0.169, 0.309] | 9/65 = 0.138 |
| 3 | 205 | 0.501 | 51 | 0.392 [0.270, 0.529] | 2/21 |
| 4 | 248 | 0.502 | 89 | 0.022 [0.006, 0.078] | 2/83 |
| 5 | 198 | 0.457 | 32 | 0.000 [0.000, 0.107] | 0/32 |
| 6 | 184 | 0.393 | 53 | 0.019 [0.003, 0.099] | 1/53 |
| **≥ 2 (G1)** | **1,241** | | **363** | **0.152 [0.118, 0.192]** | |

- Error compounds as DISCOVERY warns, and faster than the chain confidence says: at depth 4–6 the confidence is still 0.4–0.5, the precision near 0.
- On store T (the 2006-text store, S15 dev subjects) the 18 kept rules derive nothing beyond depth 1.

### G19 on R1 (dev)

- 116 refinements tried; **52 kept** (30 rescue a rejected rule, 22 sharpen a kept one).
- Precision at equal N (N = 457 graded G2 predictions): G2 0.523, G2 + G19 0.516; **gain −0.007 [−0.024, +0.009]**. G19 adds 21 graded predictions.

### The dev pools (S15 dev, bit 17 = 1; the source stage; S8c's grader)

| Generator | Hypotheses | Graded | True share | Promoted (after sources) | Loop-supported | Refuted (graded, share false) |
|---|---:|---:|---:|---:|---|---|
| G1 | 38 | 27 | 0.444 | 0 | 6 (6 / 6 true, all lexical) | 3 (2, 1.000) |
| G19 | 0 | | | | | |
| G20 (wild) | 20 | 10 | 0.000 | 0 | 0 | 15 (8, 1.000) |

- No refinement of a rejected rule predicts into the pool's slots, so G19's pool is empty.
- G1's lexical supports cannot be promoted: S8d's calibration lets no source carry a lexical promotion.
- The wild tier is refuted by the source stage, rightly every time it can be graded.

## The test pools, fixed before the test read

Built from S15 test's slots (bit 17 = 0) with `--gen-deep … --fresh-test` (no test label read), with their source stage. The prospective log was written at 2026-10-29T08:35:08Z.

| File | Rows | sha256 |
|---|---:|---|
| `t.fresh_g1.hyps.tsv` | 26 | `b66dbef4188a65e2ab632e0c15ff4311950d8fb4edddcc58fac30f11bc426503` |
| `t.fresh_g19.hyps.tsv` | 0 | `9ee21b6e2a34f65f5e33728758fd4177f0d670606a2cdbc89cb471a8ffde94e9` |
| `t.fresh_g20.hyps.tsv` | 20 | `79653ae8fbfa549f0b3e228cac3e7bcd5ad2073de49942250402649bf244745b` |
| `t.prospective.tsv` (the prospective log) | 46 | `9822e2dbb6d897663b4ac4c76f40df1abea53f8d21ade17641c65c18d7685f1c` |
| `src_g1.s8d_fresh.tsv` | | `707c82ac8f032314916315d1306dfa91446dfd450aaf4304cbe6fdb96776ff45` |
| `src_g20.s8d_fresh.tsv` | | `6d9630dd0954fb654f2170357ca0c240934338e5a90929002a3e85e3e488cf6f` |
| `src_g1.s8d_fresh_ran.tsv` | | `01b193f578ed499206cf5d6d3b94fe90c7f043b34bcc82ed3cdfa0880f939b91` |
| `src_g20.s8d_fresh_ran.tsv` | | `f11152ef330da561d8b1e27a53ef7dd300b54b77f0dc48a471c48f590c1766e2` |

The test reads follow this commit, once each: `--gen-laws --read-test` (the gate's test replicates), `--gen-deep --read-test` (G1 and G19 on R1 test, G1 on S15 test), and `--pool-grade` on the two pools.

## Test results (read once, after commit 6307b17)

### S12's gate: planted-law rediscovery (test replicates; store K's test persons: 482)

| Kind | σ 0 | σ 0.02 | σ 0.05 | σ 0.10 | σ 0.20 | At σ ≤ 0.05 |
|---|---|---|---|---|---|---|
| RATIO | 20/20 | 20/20 | 17/20 | 0/20 | 0/20 | 57/60 = 0.950 |
| PRODUCT | 20/20 | 20/20 | 17/20 | 0/20 | 0/20 | 57/60 = 0.950 |
| POWER | 20/20 | 20/20 | 0/20 | 0/20 | 0/20 | 40/60 = 0.667 |
| SCALE | 20/20 | 20/20 | 18/20 | 0/20 | 0/20 | 58/60 = 0.967 |
| OFFSET | 20/20 | 19/20 | 18/20 | 4/20 | 0/20 | 57/60 = 0.950 |
| ORDER | 16/20 | 12/20 | 10/20 | 9/20 | 2/20 | 38/60 = 0.633 |

- **Recall at σ ≤ 0.05: 307 / 360 = 0.853 [0.812, 0.886].** Below 0.90.
- **False-law rate: 0 / 360 null tables = 0.000 [0.000, 0.011]**; 0 of 360 planted tables hold a false law.
- **S12's gate is not met** (registered rule: recall ≥ 0.90 and false-law rate ≤ 0.05).
- Four kinds pass on their own at σ ≤ 0.05 (0.95–0.97). The two that fail are the two noted on dev: POWER, whose term squares a noisy attribute, and ORDER, whose 5% contamination nearly always breaks the order.

### G1 on R1 (test subjects; S8a's amended grader with S15 test's rows)

| Depth | Derived | Mean chain confidence | Graded | Precision |
|---:|---:|---:|---:|---|
| 1 (G2's own) | 1,605 | 0.752 | 661 | 0.371 [0.335, 0.408] |
| 2 | 368 | 0.575 | 197 | 0.122 [0.083, 0.175] |
| 3 | 213 | 0.545 | 96 | 0.146 [0.089, 0.230] |
| 4 | 191 | 0.532 | 105 | 0.000 [0.000, 0.035] |
| 5 | 171 | 0.484 | 61 | 0.000 [0.000, 0.059] |
| 6 | 183 | 0.433 | 78 | 0.000 [0.000, 0.047] |
| **≥ 2 (G1)** | **1,126** | | **537** | **0.071 [0.052, 0.096]** |

- Store T on S15 test's subjects: 51 facts at depth 1 (0 of 8 graded true), none deeper.

### G19 on R1 (test)

Precision at equal N (N = 580 graded G2 predictions): G2 0.405, G2 + G19 0.416; **gain +0.010 [−0.004, +0.021]**. The interval includes 0, so **G19 does not earn its place** by the registered rule.

### The test pools (graded once; S15 test, bit 17 = 0)

| Generator | Hypotheses | Graded | True share | Promoted (after sources) | Loop-supported | Refuted (graded, share false) |
|---|---:|---:|---:|---:|---:|---|
| G1 | 26 | 17 | 0.176 | 0 | 0 | 5 (2, 0.500) |
| G19 | 0 | | | | | |
| G20 (wild) | 20 | 14 | 0.000 | 0 | 0 | 17 (13, 1.000) |

- No promotions, so S8's gate cannot be read (P1).
- The wild tier's confirmed rate (0 of 14 graded) equals the main tier's (0 of 17): neither promotes. Every graded wild hypothesis is false, and the source stage refuted all 13 of them.
- The prospective log (`t.prospective.tsv`, 46 hypotheses, sha256 recorded above) stays unread until a later source lists its slots.

## Predictions, scored

| # | Prediction | Result |
|---|---|---|
| L1 | Recall at σ ≤ 0.05 ≥ 0.90; each kind ≥ 0.85 | **Miss:** 0.853; POWER 0.667 and ORDER 0.633 |
| L2 | Recall at σ = 0.10 in 0.3–0.8; at σ = 0.20 below 0.10 | **Partial:** 13/120 = 0.108 at 0.10 (miss); 2/120 at 0.20 (hit) |
| L3 | False-law rate ≤ 0.02; independent attributes in ≤ 2% of tables | **Hit:** 0 and 0 |
| L4 | Countries: at least 3 of 4 expected laws | **Hit:** density and the two unit conversions |
| L5 | Persons: death ≥ birth and the lifespan kept | **Partial:** the order is kept in both stores; the lifespan is not (0.12 against 0.1) |
| L6 | At most 1 law outside the expected list | **Hit:** none |
| D1 | G1: 50–500 facts at depth ≥ 2; depth-2 precision within 0.15 of depth 1's | **Miss:** 1,241 (dev); 0.232 against 0.503 |
| D2 | Store T: fewer than 20 facts at depth ≥ 2 for S15 subjects | **Hit:** none |
| R1 | G19: 10–100 kept; test gain 0.00 to +0.05, interval may include 0 | **Hit:** 52 kept; +0.010 [−0.004, +0.021] |
| W1 | G20's confirmed rate below the main tier's; at most 1 wild promotion | **Partial:** no wild promotion (hit); the rates tie at 0 |
| P1 | Pools: fewer than 20 graded promotions per generator; S8's gate not readable | **Hit** |

6 hits, 3 partial, 2 misses.

## Decisions (by the registered rules)

1. **S12's gate is not met:** recall 0.853 against 0.90, with no false law. G15 rediscovers ratio, product, scale and offset laws at noise up to 5% (0.95–0.97). It misses squared terms at 5% noise and orders with typing errors.
2. **G15 stays.** It finds the real laws of the 2006 infoboxes (population density, and the km² and mi² conversions) and keeps nothing false in 720 null and planted test tables.
3. **G1 adds nothing promotable here.** Its precision falls from 0.37 at depth 1 to 0.07 beyond, and to 0 at depth 4–6, while its chain confidence stays at 0.43–0.53. The prior overstates deep chains.
4. **G19 does not earn its place** (gain interval includes 0). The registration did not say its code goes, so it stays; that is for Odin.
5. **G20's wild tier is refuted by the source stage**, rightly each time it can be graded. The prospective log holds the pools' 46 hypotheses for a later check.

## For Odin

1. **POWER laws:** a law whose term squares a noisy attribute fails CV < 0.1 at 5% noise. Measure the CV on the attribute the law predicts (the term's CV divided by that attribute's exponent)? It needs its own registered read.
2. **Orders with typing errors:** keep orders at 90% when they beat independence by a wider margin? Also a registered read.
3. **G1's prior:** cap G1 at S9's depth 3, or calibrate its prior by depth (depth 4–6 were never right here)?
4. **G19:** remove it, as S8b's and S11b's rule 3 did for generators that failed their reads, or keep it for a richer store?
5. **P9's trigger** (30 entities with 3 numeric attributes) holds only for the 54 country infoboxes. A larger numeric extraction (enwik9's infoboxes) would give G15 real material.
6. **G17 and G18** wait for the forecasting module's event vocabulary (FCAST).

## Checks

- `native_epistemic_test`: 1,330 checks, 37 new (`test_laws_deep_S12a`). Six planted bugs each make it fail: the order law's null back to a coin, no redundancy, the held-out check ignored, G19 conditioning on the head's relation, G1's confidence not a product (in worlds.hpp), and G20 without its support rule.
- S3's `--gaps` run is byte-identical: the gap map, its cells and demand files, and the seven store files (10 of 10).
- The test reads ran on the committed tool (e84fcb6 code; 6307b17 only changed a test).
- The nine CTests pass (9 of 9) on this branch: `native_cyphalm_trie_selftest`, `native_fm_infinigram_test`, `native_epistemic_test`, `native_thought_test`, `native_geometry_test`, `native_ecs_lci_test`, `native_scribe_selftest`, `native_cyphalm_neural_smoke`, `native_class_context_test`.

## How to reproduce

```
g++ -O2 -std=c++20 -I native/include -pthread native/tools/pattern_learn.cpp -o pattern_learn
F=/root/work/f2
./pattern_learn --gen-laws enwik8 $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv OUT/laws               # dev replicates and the real read
./pattern_learn --gen-laws enwik8 … OUT/laws --read-test                                            # the gate (once)
./pattern_learn --gen-deep enwik8 $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval OUT/d --fresh-dev   # dev reads and pools
./pattern_learn --gen-deep … OUT/t --fresh-test                                                     # test pools (no label read)
./pattern_learn --gen-deep … OUT/r --read-test                                                      # G1 and G19 test reads (once)
# each pool: S8d's source stage, then the grade
./pattern_learn --sources bench/eval/temporal_holdout_dev.tsv TABLES $F/wikidata/wd_values.tsv $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv OUT/src_g1 S8D_R1_DEV S8D_R2_DEV S8D_PROMO --fresh OUT/t.fresh_g1.hyps.tsv empty.tsv
./pattern_learn --pool-grade s15 bench/eval/temporal_holdout_test.tsv 0 OUT/t.fresh_g1.hyps.tsv OUT/src_g1.s8d_fresh.tsv
```

`TABLES` is S8c's; the `S8D_*` files are S8d's dev hypotheses (see the S8d report). A full `--gen-laws` run takes about 15 s; `--gen-deep` with pools about 40 s.
