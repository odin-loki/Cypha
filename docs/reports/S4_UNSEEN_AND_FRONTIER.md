# S4: estimates of the unseen, and the frontier report

**Odin Loch** · 2026-10-28 · DISCOVERY §3.3, §3.4, §13 S4 · follows [S3_S5_GAP_MAP_AND_HYPOTHESES.md](S3_S5_GAP_MAP_AND_HYPOTHESES.md) and [S1_S2_EPISTEMIC.md](S1_S2_EPISTEMIC.md)

(The rules and predictions below were written before the runs. Results and decisions are added under them.)

**Question.** Cypha should know how much it has not seen, not only where. Do the standard unseen-species estimates, computed on what Cypha has read, predict what it finds when it reads more? And where would reading more teach it the most?

**Gate (DISCOVERY §13).** Missing-mass validity within ±25% on a held-out source.

**Rule 10.** This is Cypha's self-assessment. Nothing here produces an answer for CyphaLM, a gate decision or a tag. It counts what was read and predicts what reading more would bring.

## Rules (fixed before the runs)

### Data

- enwik8, all 8,587 main-namespace articles (`pattern_learn`'s page list), and the v2 fact store as S3 built it (`--cv --wikidata`).
- A page is the sampling unit. A page's place is the byte offset where its text starts.

### Splits

- **Store view (the gate).** Estimation set E: pages starting before byte 50,000,000 (the first half of the dump). Held-out set H: the rest. H is split by a bit of the title's FNV-1a hash into **H-dev** and **H-test**. H-dev chooses; H-test is read once, at the end.
- **CyphaLM's view.** E7: pages starting before byte 7,340,032 (CyphaLM's wiki train). H7: pages starting at or after 8,388,608, split into dev and test the same way.
- **Random split (diagnostic, dev only).** E and H by another hash bit over all pages. It tests the estimators with no drift between E and H.

### Items

| Kind | Item | Observation |
|---|---|---|
| words | a maximal run of ASCII letters in the page's raw text, lower-cased | each run |
| entities | the target of a `[[target]]` or `[[target|text]]` link, normalised (`norm_ent`), without `#section`; namespace and language links skipped | each link |
| facts | (subject, relation, object): a text seed (infobox, table or category) or a stored extracted fact (P ≥ 0.5) | each seed, and each evidence sentence of a stored fact, on the page that holds it |
| values | (relation, object) of the same facts | the same |

### Estimators, computed on E

- **GT0**, DISCOVERY §3.3's Good–Turing missing mass: n₁ / N.
- **GTu**, Good–Turing with the page as the unit: the observations of items seen on exactly one page, over N. It is the leave-one-page-out estimate of the same quantity. Text repeats an item inside a page, so a page-level repeat is no evidence that an item is common.
- **Q1/U**, the incidence form: items on exactly one page over all (page, item) incidences.
- **Chao extrapolation** (Chao et al. 2014), with pages as units: Ŝ(T + t) = S_obs + Q̂₀ [1 − (1 − Q₁/(T Q̂₀ + Q₁))^t], Q̂₀ = ((T−1)/T) Q₁² / (2 Q₂). Also the abundance form, over observations.
- **Heaps** (words): V = K N^β fit on E's growth curve (every 25th page). It predicts the vocabulary after more reading, and the new-word rate dV/dN = β V / N.
- **Chapman** per relation with a kept pattern: source A = slots (subject, relation) with a text seed; source B = slots where a kept pattern fires on the subject's page (any value); m = both.

### Targets, measured on H-dev or H-test

- For GT0 and GTu: the share of the held-out observations whose item is not in E.
- For Q1/U: the share of held-out (page, item) incidences whose item is not in E.
- For Chao: the distinct items in E ∪ H-x.
- For Heaps: the distinct words in E ∪ H-x, and the words target.
- For Chapman: a reference count N_ref = |A ∪ B ∪ W|, where W is the slots with a Wikidata value. Chapman is a lower bound because A and B are correlated (patterns are learned from seeds). The check is N̂ ≤ N_ref.

### Intervals

- The target's sampling error: a bootstrap over the held-out pages (2,000 resamples), E fixed. It gives a 95% interval for the ratio estimate / target.
- The estimate's own error: a delete-a-group jackknife over 20 groups of E's pages, reported as a standard error.

### The gate

- On H-dev, choose between GT0 and GTu: the one with the smaller mean |ln(estimate / target)| over words, entities and facts (relations pooled).
- **S4 passes** if, on H-test (read once), the chosen estimator's point ratio estimate / target is within [0.75, 1.25] for each of words, entities and facts.
- Per relation (facts and values, relations with ≥ 30 held-out observations), the share of relations within ±25% is reported, not gated. CyphaLM's view is reported with the same rule, not gated.

### The frontier report (`pattern_learn --frontier FILE`, with `--gaps`)

One row per (type, relation) cell of S3's map (types: country, person, place, other):

- **known**: filled slots (S3's `with_store`).
- **expected**: entities × P(exists) (S3's chosen estimator of f(T, R)).
- **coverage** = known / max(known, expected); **unseen in the slice** = max(0, expected − known): slots that should exist on pages already read, which better extraction or other sources would fill (P1).
- **certainty**: the mean P over the cell's filled slots (a text seed counts 20/21, an extracted fact its fused P).
- **demand**: CyphaLM's tags on the cell (S3's demand log).
- **reading more**: the chosen estimator of missing mass over the cell's fact observations, times the cell's fact observations per MiB of article text, times 10: the expected new facts per 10 MiB of comparable new text.
- **values**: Chao1 over the relation's distinct values (how many distinct values the relation has, seen and unseen).
- Two rankings: by expected new facts per 10 MiB read ("where reading more teaches the most"), and by unseen in the slice ("where better extraction teaches the most"), each with demand alongside.

### Unknown-unknown signals (§3.4)

Only the vocabulary signal is cheap now: the share of new words in held-out text against the Heaps prediction. The other signals (OOD rate, unparsed sentences, surprise spikes, routing, unnamed clusters, trust shifts) have no producer yet and are listed as not done.

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| P1 | GT0 underestimates the target for words and entities on H-dev (store view) by 10–40%: repeats inside one page make items look common |
| P2 | GTu is within ±15% of the target for words and entities on H-dev |
| P3 | Facts, pooled: the target is ≥ 0.95 (a fact sits on its subject's page); GTu within ±5%; GT0 below GTu |
| P4 | GTu is chosen on dev, and the gate passes on H-test |
| P5 | Values, pooled: GTu within ±25% on dev; per relation, at least half of the relations with ≥ 30 held-out observations within ±25% |
| P6 | Chao extrapolation (pages as units) of distinct words and entities in E ∪ H-x is within ±10%, and under rather than over |
| P7 | Heaps: the vocabulary of E ∪ H-x within ±5%; β between 0.5 and 0.8; the Heaps new-word rate within ±30% of the words target |
| P8 | Chapman(seeds, patterns) ≤ N_ref for every relation with a kept pattern |
| P9 | CyphaLM's view: GTu within ±25% for words and entities; its words target is at least twice the store view's |
| P10 | The top cells by reading more are person cells for birth and death dates and years |

## The dev choice (before the test read)

On H-dev (store view), the ratio estimate / target:

| Kind | GT0 | GTu |
|---|---:|---:|
| words | 0.509 | 0.904 |
| entities | 0.768 | 0.944 |
| facts | 0.555 | 1.000 |

Mean |ln ratio|: GT0 0.509, GTu 0.053. **GTu is chosen.** The test half is read once below with it.

## Results (run 2026-10-28)

Tool: `pattern_learn --frontier` on S3's run (`--cv --wikidata --gaps --demand`). Without `--frontier` the run writes S3's files byte for byte (gap map sha256 a77678d8ef7dfdef, cells eed14241a6d6eff2, demand dd1c8ddac831dcb6; the store files equal `f2/v2/aug`). 34–53 s, peak 226 MB.

Items on all of enwik8: 282,882 word types, 246,292 link entities, 8,306 facts, 4,666 values.

### The gate: missing mass on the held-out test half (read once)

Store view: E = 4,413 pages (the first 50,000,000 bytes); H-test = 2,029 pages. Ratio = GTu / target, with the 95% bootstrap interval over H-test's pages.

| Kind | GTu (jackknife se) | Target: share of H-test's observations that are new | GTu / target | GT0 / target |
|---|---:|---:|---|---:|
| words | 0.02279 (0.00055) | 0.02354 [0.02246, 0.02470] | **0.968** [0.923, 1.015] | 0.545 |
| entities | 0.30735 (0.00428) | 0.31358 [0.30360, 0.32493] | **0.980** [0.946, 1.012] | 0.797 |
| facts | 0.99967 (0.00046) | 1.00000 | **1.000** | 0.555 |
| values (not gated) | 0.61477 (0.00863) | 0.63027 [0.60396, 0.65608] | 0.975 [0.937, 1.018] | 0.420 |

**S4's gate is met.** Good–Turing with the page as the unit is within 4% of what reading more pages actually brings, for words, entities and facts, and the word and entity intervals include 1.

- DISCOVERY's n₁ / N (GT0) is not: it says 1.3% of the next words are new when 2.4% are, and 25% of the next links when 31% are. A page repeats what it talks about, so an item seen five times on one page looks common to GT0 and is not.
- Facts are trivially new: a fact sits on its own subject's page, so every held-out fact observation is a new fact (target 1.000).
- The **random split** (no drift between E and H) gives GTu / target 0.946 (words), 1.014 (entities), 1.000 (facts), 1.026 (values) on dev. So the store view's small shortfall on words (0.904 dev, 0.968 test) is drift: the second half of the dump brings slightly more new words.

**CyphaLM's view** (E = its 536 wiki-train pages; H = the pages from 8,388,608 on), test: words 0.956 [0.933, 0.978], entities 0.895 [0.883, 0.908], facts 1.000, values 0.949. All within ±25%. Its words target is 0.056: one word in 18 of the rest of the dump is a word CyphaLM has never read (2.4× the store view's).

### Per relation (not gated)

| Set | Facts within ±25% | Values within ±25% (relations with ≥ 30 held-out observations) | Outside |
|---|---|---|---|
| store, dev | 17 / 17 | 15 / 17 | occupation 5.94, order 1.31 |
| store, test | 14 / 14 | 13 / 14 | occupation 1.51 |
| CyphaLM, dev | 26 / 26 | 22 / 26 | located_in 0.67, occupation 0.63, order 1.39, class 2.09 |
| CyphaLM, test | 21 / 21 | 18 / 21 | located_in 0.68, language 1.45, occupation 0.52 |

Occupation has 38 values seen hundreds of times; its target is near 0 (0.004 on dev), so a relative error means little there.

### Richness and Heaps

| Check | Store dev | Store test | CyphaLM dev | CyphaLM test |
|---|---:|---:|---:|---:|
| Chao extrapolation (pages), words: predicted / observed distinct in E + H | 0.977 | 0.987 | 0.711 | 0.730 |
| the same, entities | 0.980 | 0.989 | 0.798 | 0.789 |
| the same, observations as draws, words | 0.935 | 0.949 | 0.518 | 0.532 |
| Heaps: vocabulary of E + H, predicted / observed | 1.007 | 1.020 | 1.117 | 1.158 |
| Heaps β | 0.611 | 0.611 | 0.661 | 0.661 |
| Heaps new-word rate / words target | 0.655 | 0.701 | 0.648 | 0.684 |

- In the store view (H about half of E) the page-unit Chao extrapolation is within 3%, and under, as registered. In CyphaLM's view it is asked to extrapolate 536 pages to about 4,500 (8×), past its known range (about 2×), and it falls 20–30% short. Heaps overshoots there by 12–16%.
- Heaps' new-word rate counts a new word once (where the vocabulary grows); the target counts every occurrence of it in the new pages. So the rate runs at 0.65–0.70 of the target even with no drift (random split 0.675). As an unknown-unknown alarm (§3.4) the baseline is that ratio, not 1: store view 0.655, random 0.675, so no drift alarm.

### Chapman per relation (the seven relations with a kept pattern)

| Relation | A text seeds | B patterns fire | m | N̂ (sd) | N_ref = A ∪ B ∪ Wikidata | N̂ / N_ref | Text vs Wikidata: N̂ |
|---|---:|---:|---:|---:|---:|---:|---:|
| birth_place | 362 | 226 | 89 | 915 (65) | 1,231 | 0.743 | 1,293 |
| death_place | 77 | 36 | 4 | 576 (212) | 1,026 | 0.562 | 1,128 |
| birth_date | 113 | 706 | 81 | 982 (54) | 1,068 | 0.919 | 1,119 |
| death_date | 81 | 479 | 57 | 678 (45) | 868 | 0.781 | 907 |
| birth_year | 1,135 | 737 | 714 | 1,172 (5) | 1,318 | 0.889 | 1,334 |
| death_year | 914 | 511 | 498 | 938 (5) | 1,027 | 0.913 | 1,037 |
| language | 121 | 27 | 27 | 121 (0) | 272 | 0.445 | 497 |

- Chapman over the two text sources is a lower bound in all 7 relations (0.44–0.92 of N_ref): the patterns are learned from the seeds, so the two sources are correlated.
- Text against Wikidata are closer to independent. Their Chapman estimates agree with the frontier's expected counts for person cells within 8–17% (birth_place 1,293 against 1,146; birth_date 1,119 against 1,035; death_place 1,128 against 964).

### The frontier report

`pattern_learn --frontier FILE` writes 132 cells (4 types × 33 relations). Over all cells the store holds 7,124 slots and an estimated 9,970 more exist on pages already read (coverage 0.42). By type: person 0.56, country 0.77, place 0.25, other 0.20. Of the 45 cells with at least 50 expected slots, 28 are under half covered.

**One definition was corrected after the dev run** (the estimates above are unchanged by it): S3's P(exists) is the chance that an *empty* store slot has a value (calibrated on empty slots), so the unseen slots are (entities − known) × P(exists), and expected = known + unseen. The registered "entities × P(exists)" gave expected below known in some cells (person birth_year 637 against 1,160 known). The frontier file and the lists below use the corrected definition.

Where reading more teaches the most (new fact observations per 10 MiB of comparable text; in brackets, distinct new facts):

| # | Cell | New per 10 MiB | Coverage | Unseen in the slice | Demand |
|---|---|---:|---:|---:|---:|
| 1 | person birth_date | 268.5 (84.9) | 0.710 | 300 | 31 |
| 2 | person death_date | 171.6 (56.0) | 0.601 | 334 | 16 |
| 3 | person birth_year | 143.0 (133.1) | 0.935 | 81 | 0 |
| 4 | person birth_place | 126.1 (70.9) | 0.435 | 647 | 0 |
| 5 | person occupation | 119.7 (89.0) | 0.480 | 644 | 0 |
| 6 | person nationality | 117.6 (102.7) | 0.647 | 330 | 0 |
| 7 | person death_year | 107.2 (104.1) | 0.960 | 39 | 0 |
| 8 | place located_in | 34.7 (32.9) | 0.680 | 114 | 0 |

Where better extraction (or another source) teaches the most (unseen slots on pages already read):

| # | Cell | Unseen | Coverage | Kept pattern |
|---|---|---:|---:|---|
| 1 | other country | 1,194 | 0.065 | no |
| 2 | person death_place | 855 | 0.113 | yes |
| 3 | other founded | 727 | 0.127 | no |
| 4 | person birth_place | 647 | 0.435 | yes |
| 5 | person occupation | 644 | 0.480 | no |
| 6 | person spouse | 613 | 0.038 | no |
| 7 | place country | 440 | 0.002 | no |

- Ranked by distinct new facts instead, the reading list is the same cells in a different order (birth_year, death_year and nationality first: their facts have one evidence sentence each).
- **Demand** sits on few cells: of CyphaLM's 750 dev tags, 31 land on person birth_date and 16 on death_date; most name a relation the store does not have (`definition`, 333) or an entity with no page (S3's finding).
- Values: Chao1 (pages as units) says person birth_date has about 28,000 distinct values (753 seen) and birth_place about 1,300 (396 seen).

### Unknown-unknown signals (§3.4)

| Signal | State |
|---|---|
| Vocabulary: new-word rate against Heaps | Done: the ratio is 0.65–0.70 with no drift; an alarm is a ratio well below that |
| OOD rate of incoming text and questions | Not done: no OOD detector feeds Cypha yet |
| Unparsed-sentence rate | Not done: needs the sentence parser's failure count per domain |
| Surprise spikes in CyphaLM | Not done |
| Uncertain routing | Not done: the question classifier was removed (E4) |
| Unnamed clusters (G9) | Not done: S11 |
| Source-trust shifts | Not done: no trust posteriors yet |

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| P1 | GT0 under by 10–40% for words and entities | **Miss** on size: under as predicted, but words by 49% (entities 23%) |
| P2 | GTu within ±15% (dev) | **Hit**: 0.904, 0.944 |
| P3 | Facts: target ≥ 0.95, GTu within ±5%, GT0 below | **Hit**: 1.000, 0.9997, 0.555 |
| P4 | GTu chosen; the gate passes on test | **Hit**: 0.968, 0.980, 1.000 |
| P5 | Values pooled within ±25%; half the relations within ±25% | **Hit**: 0.956; 15 / 17 (dev), 13 / 14 (test) |
| P6 | Chao (pages) within ±10%, under | **Hit** in the store view (0.977–0.989); **miss** in CyphaLM's (0.71–0.80, an 8× extrapolation) |
| P7 | Heaps vocabulary within ±5%; β in 0.5–0.8; new-word rate within ±30% | Vocabulary **hit** in the store view (1.007, 1.020), **miss** in CyphaLM's (1.12, 1.16); β **hit**; rate **miss** (0.655 dev, 0.701 test) |
| P8 | Chapman(seeds, patterns) ≤ N_ref, every relation | **Hit**: 7 / 7 |
| P9 | CyphaLM's view: GTu within ±25%; words target ≥ 2× | **Hit**: 0.90–0.96; 2.36× (dev), 2.39× (test) |
| P10 | Top cells by reading are person birth/death dates and years | **Hit**: ranks 1–3 (death_year 7) |

### Checks

- `native_epistemic_test`: 536 checks (20 new: clumped statistics, the page-unit and incidence missing mass, the Chao extrapolation). Three planted bugs are caught: q1 for q1_mass in `unit_missing_mass` (2 checks fail); the (n − 1) / n factor dropped from the extrapolation (6); `units <= 1` and `== 3` in `clumped_stats` (1).

## Decisions

1. **Good–Turing with the page as the unit (GTu) is Cypha's missing-mass estimator for anything read from text.** DISCOVERY §3.3's n₁ / N undercounts what is new by 20–50% on page-clumped text; GTu is within 4% on held-out pages. n₁ / N stays right where the draws really are independent.
2. **S4's gate is met** (store view, test: 0.968, 0.980, 1.000).
3. **The frontier report is `pattern_learn --frontier FILE`**, run after `--gaps`. For facts, reading more always brings new facts (missing mass ≈ 1); the useful numbers are the reading rates per cell, the slots unseen on pages already read, and the values' richness.
4. **The Chao extrapolation is trusted to about 2× the pages read.** Beyond it (CyphaLM's 7 MiB to the whole dump) it undercounts by 20–30%.

## For Odin

- DISCOVERY §3.3 names n₁ / N. The measurement says to use the page as the unit for text. Changing the document's formula is your call; the code has both.
- The demand question stays open (HANDOFF §5.7): most of CyphaLM's tags name relations the store does not have.

## Reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
E=/root/cdata/enwik8; W=/root/work/f2/wikidata/wd_values.tsv; DEV=bench/eval/t_qa_wiki_dev.tsv
A=/root/work/answer; Q=/root/work/qa2/runs
D=$A/d_rows.tsv,$A/d2_rows.tsv,$Q/E1a.rows.tsv,$Q/E1b.rows.tsv,$Q/E1c.rows.tsv,$Q/E1d.rows.tsv
O=/root/work/s4/runs; mkdir -p $O/dev2 $O/test
# dev (UNSEEN / UNSEEN_REL / HEAPS / CHAPMAN / FRONTIER lines on stdout; the frontier in frontier.tsv)
./pattern_learn $E $O/dev2 --cv --wikidata $W --dev $DEV --gaps $O/dev2/gap_map.tsv --demand $D --parse $A/parse_dev.tsv --frontier $O/dev2/frontier.tsv
# the test half (read once; it was read with the first frontier definition, which does not touch the estimates)
./pattern_learn $E $O/test --cv --wikidata $W --dev $DEV --gaps $O/test/gap_map.tsv --demand $D --parse $A/parse_dev.tsv --frontier $O/test/frontier.tsv --frontier-test
g++ -O2 -std=c++20 -I native/include native/tools/epistemic_test.cpp -o epistemic_test && ./epistemic_test   # 536 checks
```

- sha256 prefix of the frontier file: `frontier.tsv` 4306e17a24633044.
