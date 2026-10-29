# S8e: a grader that knows Wikidata's names and places, a fresh set on new slots, and S8's gate read again

**Odin Loch** · 2026-10-29 · DISCOVERY §7.4, §13 S8 · follows [S8D_PROMOTION.md](S8D_PROMOTION.md) · HANDOFF §5.13

(The grader, the set's construction, the rules and the predictions below were written and committed before any S8e read. Results are added under them.)

**Why.** S8d read S8's gate once on a fresh set: 25 graded promotions at 0.880 [0.700, 0.958], so the gate (0.9) was not met. All three false ones name the right place another way:
- Belgium, capital "Brussels": S15 holds City of Brussels, and Wikidata lists "Brussels" among that item's names.
- Bill Bixby, death place "Los Angeles": S15 holds Century City, a district of Los Angeles.
- Hillary Rodham Clinton, birth place "Chicago": S15 holds Edgewater Hospital, in Chicago.

S8c's grader matches S15's one shown name, closed-world. Odin's "Continue. Do all." takes HANDOFF §5.13's recommended path: register a grader that knows Wikidata's names and places inside places, build a fresh set on new slots, and read the gate once more. §5.13's other questions are measured where they can be (Supported or Known; a minimum of positives per source).

**Rule 10.** Everything here grades Cypha's own hypotheses. Nothing answers a question, and nothing reaches `qa`, `chat` or a training text.

## Part 1. The grader (fixed before any read)

### Its data

`scripts/fresh_slots.py grader` fetches from Wikidata's SPARQL endpoint, with D4's fetcher (one request at a time, back-off on 429 and 5xx, every response cached under the run directory), for every item a grade can touch: the value items of S15 (dev and test) and of the fresh set (Part 2), D4's subject items, and their place ancestors.
- **Names:** the English label, every English alias, the enwiki title, every English demonym (P1549) and the taxon name (P225).
- **Place ancestors:** everything reachable upward by located in the administrative territorial entity (P131), location (P276) or country (P17), walked one hop a round (truthy values).
- **Two class flags,** by instance of (P31) followed by subclass of (P279) any number of times:
  - *country-like:* country (Q6256), sovereign state (Q3624078), historical country (Q3024240), country of the United Kingdom (Q3336843);
  - *coarse:* country-like, first-level administrative country subdivision (Q10864048), or continent (Q5107).

### The matching rule

A hypothesis (subject, relation, value h) is graded against its slot's true values in the set's file (label 1), as S8c's grader is, with these additions. All names are compared after S8c's `norm_name`.

1. **Dates and years:** unchanged (S8c's `same_value`).
2. **Equal by name (every relation):** h is one of a true value's names (label, any alias, enwiki title, demonym, taxon name). This takes "Brussels" for City of Brussels, and "American" for the United States of America.
3. **Places containing the truth:** for birth_place, death_place, location and headquarters, h names a place ancestor of a true value that is not coarse. "Los Angeles" for Century City counts; "California" or "United States" does not. For located_in (P131, which is transitive) any ancestor counts, coarse ones too.
4. **A place inside the truth** (h names an item whose ancestors include a true value), for the same relations: **ungraded.** Wikidata cannot say whether the finer place is right.
5. **Country-like places inside the truth:** for country and nationality, h naming a country-like item whose ancestors include a true value counts. "English" (England) for the United Kingdom counts; "California" for the United States does not (not country-like).
6. **Everything else:** names only (rule 2).
7. **False:**
   - h matches a slot's drawn false value (S8c's rule), and no true value;
   - or the slot holds a true value, h matches none, and the relation is single-valued by S8c's list;
   - **or** the relation is one of the fresh set's (nationality, location, predecessor, successor, family, order, class), the slot holds a true value, and h matches none by rules 2–5. Wikidata's lists for these are taken as complete, which makes these slots closed-world.
8. Otherwise ungraded.

**The grader's own error.** After the dev pools exist, 100 graded dev items are drawn with a fixed seed (20261029): 60 where the new grader's verdict differs from S8c's grader or rests on rules 2–5 beyond S15's shown name, and 40 where both agree. Each is checked by hand against Wikidata (names, P131 / P276 / P17, P31). The checks are kept in `docs/reports/S8E_GRADER_AUDIT.tsv`. The grader's error is the share where the hand check disagrees, with a Wilson interval, reported for each stratum.

**S8d's 25 promotions,** graded again with the new grader, are reported only. They were read once already.

## Part 2. The fresh set (fixed before any read)

**Why new slots.** S15 test's two halves each carry a pool read now (S10 on bit 17 = 1, S11c on bit 17 = 0), and S15 dev's confirm half carries S8d's. So the set uses slots S15 never held.

**The slots.** S15's builder (`scripts/temporal_holdout.py`, now with its set name as a parameter; S15's files rebuild from its cache byte for byte) runs on relations S15 did not fetch, for the same subjects (D4's map, D4's T-QA exclusions kept):

| Relation (pattern_learn's) | Wikidata |
|---|---|
| nationality | country of citizenship (P27) |
| location | location (P276) |
| predecessor | replaces (P1365), follows (P155) |
| successor | replaced by (P1366), followed by (P156) |
| family, order, class | the nearest ancestor by parent taxon (P171, truthy) whose taxon rank (P105) is family (Q35409), order (Q36602) or class (Q37517) |

- Direct statements only (an office's successor kept as a qualifier of position held is not read).
- Same columns, categories (a, b, c, u), false partners and page rule as S15. The taxon ranks are timeless.
- Ids `S8E-…`. Split by subject: test if sha256("S8E|" + item)[0] ≥ 128, else dev.
- Written to `bench/eval/s8e_slots_{dev,test}.tsv`, frozen at their commit with their sha256s in `bench/eval/README.md`, with the response cache beside them for audit. The same freezing rule as S15: dev is read freely; test only by this gate's one read.
- No earlier read graded any of these slots: S15 holds none of these properties.

**The pools.** Each generator on the reduced store with every store triple in the part's slots removed, as S8d (G2, G6/G7) and S11c (G3, G4, G5) built theirs, with a new option `--fresh-slots FILE` that takes the slots from a set file (every (subject, relation) of the file whose relation is one of `pattern_learn`'s):
- **G2** (`--gen-rules … --fresh-slots`): S8d's construction (R1's rules, noisy-or; T1, T6, T4).
- **G6/G7** (`--gen-links … --fresh-slots`): S8d's construction (the combination chosen by S8b on dev; T1, T6, T4).
- **G3** (`--gen-anomaly … --fresh-slots`): S11c's pool construction. G13 is parked and left out.
- **G4, G5** (`--gen-analogy … --fresh-slots`): S11c's pool construction (G5 only on country and nationality, as S11c chose).
- The set's labels are not read to build a pool. The test pools' sha256s are added here, and committed, before the read.

## Part 3. Calibration and the minimum of positives (fixed before any read)

**The calibration set.** S8d's (the generators' dev hypotheses on S15's choose half), graded with the new grader, together with the fresh **dev** part's hypotheses from all five pools, graded with the new grader on the dev file. Each distinct (subject, relation, value) once. Then S8d's rule as it stands (PLAN ledger 71): each accepted (source, group) cell with add-one; its own tally when it holds at least 10 graded true and 10 graded false items, else the source's pooled tally; it may carry a promotion only if LR+ ≥ 10. The groups are `pattern_learn`'s: nationality, family, order and class are lexical; predecessor and successor links; location places.

**Variant M, a minimum of positives.** A cell may carry only if, in addition, the tally it uses holds at least 10 positive results on graded items (true or false).

**The choice between them, on dev.** The fresh dev part goes through the source stage with each variant and is graded on the dev file. These items are also in the calibration set, so this read is in-sample and only chooses. **M applies** unless the promotions M removes number at least 5 graded and are at least 0.9 precise; then the variant without the minimum applies.

## Part 4. The gate read (once)

- The five test pools go through S8d's source stage (each from its generator's posterior; DB-IB, WP-LEAD and GN in that order, with the calibration above; `decide` needs a positive independent result), with the variant chosen on dev.
- Graded with the new grader on the test file. Each distinct hypothesis once.
- **S8's gate:** the supported set has at least 20 graded hypotheses and precision at least 0.9.
- Reported beside it, not gates: Known (row 70's tier); the other variant; S8c's grader; by generator, source, relation and group, with Wilson 95% intervals.

## Decision rule

- **Gate met:** propose a ledger row, decided by measurement under row 55 and open to Odin's veto: the new grader for S8's reads, and the variant chosen on dev.
- **Gate not met:** report the counts and where the false promotions come from. Nothing is changed after the read.

## Predictions (registered before any read)

| # | Prediction |
|---|---|
| E1 | S15's files rebuild from S15's cache byte for byte with the parametrised builder |
| E2 | The fresh set holds 2,000–6,000 true facts; nationality is the largest relation (at least 40% of them) |
| E3 | Graded again by the new grader, S8d's three false promotions become true (25 of 25) |
| E4 | The grader's audited error is at most 5% (Wilson upper bound at most 0.15) |
| E5 | DB-IB places and links still carry; DB-IB lexical still does not (LR+ under 10) |
| E6 | Under M, GN places can no longer carry (fewer than 10 positives) |
| E7 | Dev chooses M |
| E8 | The test read has 8–30 graded promotions, at precision at least 0.85; S8's gate is not met for want of 20 graded |
| E9 | The Known tier is empty again |
| E10 | Most test promotions (at least 70%) are predecessor or successor, carried by DB-IB |

## Results

### Before the gate read

Everything in this section was committed, with the test pools' hashes, before the gate read.

**E1 (hit).** S15's builder, now with its set name as a parameter, rebuilds both S15 files from S15's cache byte for byte (`9e1ae1ba…`, `1535dcdd…`).

**The fresh set (E2: hit, just).** Built 2026-10-29 with 393 Wikidata requests. A rebuild from the committed cache alone gives the same two sha256s with 0 requests.

| | dev | test | all |
|---|---:|---:|---:|
| rows (true and false) | 3,428 | 3,922 | 7,350 |
| true facts | 1,714 | 1,961 | 3,675 |
| slots (subject, relation) | 1,369 | 1,383 | 2,752 |
| category a / b / c / u | 234 / 67 / 6 / 1,407 | 206 / 39 / 13 / 1,703 | 440 / 106 / 19 / 3,110 |

True facts by relation: nationality 1,471 (40.0%), predecessor 1,011, successor 466, location 439, order 114, class 94, family 80. So nationality is the largest relation, at 40.0% (the prediction said at least 40%).

**The grader's tables.** Fetched 2026-10-29 with D4's fetcher: 537 requests (136 for place parents, 134 for names, 267 for class flags). One names response came back cut short at 98 KB; it was deleted and fetched again. A rerun from the cache gives the same three files with 0 requests.
- 26,627 items, 26,438 with names (65,665 distinct normalised names);
- 12,998 items with place ancestors (38,068 links);
- 1,005 country-like items, 1,716 coarse ones.
- The three tables are committed as `bench/eval/s8e_grader_tables.tar.gz` (0.77 MB, sha256 `5a99f36513826dbab00406a4c7d53d1dd6b7991471683b6cf0969e9e224ee620`): `names.tsv` `50f6170f…`, `anc.tsv` `69b4e360…`, `flags.tsv` `ac0eed62…`. The raw responses (32 MB) stay in the run directory.

**S8d's 25 promotions graded again (E3: hit).** All three false ones become true: Belgium's capital "Brussels" (rule 2, a name of City of Brussels), Bill Bixby's death place "Los Angeles" (rule 3, Century City is inside it) and Hillary Rodham Clinton's birth place "Chicago" (rule 3). The other 22 stay true. So 25 of 25, against 22 of 25 (0.880) before. This is a report only: those items were read once already.

**The pools.** Every pool was built on its part's slots with their store facts hidden. No label of the set was read.

| Generator | dev hypotheses | test hypotheses | test file sha256 |
|---|---:|---:|---|
| G2 (`s8a.fresh_g2.hyps.tsv`) | 28 | 15 | `d1f9dfcc46d4749ff5d56965bd93e6894544d36277d1aa0deef4cf0768164f2c` |
| G6/G7 (`s8b.fresh_links.hyps.tsv`) | 1,006 | 1,030 | `b492c49c96b5d0398413fe8dc595bc86814a0d8ce2ab452d29e6c67655a660b1` |
| G3 (`a.fresh_g3.hyps.tsv`) | 209 | 179 | `1e68f82676c7e2e91179e253ff3c72d20c6899206cb69353eef082b95d0e5d6b` |
| G4 (`g.fresh_g4.hyps.tsv`) | 513 | 518 | `fafcc91c06074d4bb111b9cca869e252ee2569be910689ac88ee42a4ad4437e9` |
| G5 (`g.fresh_g5.hyps.tsv`) | 571 | 579 | `a6b7ba2fdc3a0339b454afff08a9b68eb7d9cc4aa5dcc270eb090a748f92ca60` |

- Dev files: G2 `32bbc6d5…`, G6/G7 `9121660…`, G3 `4b244949…`, G4 `8c726b8b…`, G5 `107b7bd4…`.
- A second run of `--gen-links` and `--gen-analogy` on the test part wrote the G6/G7, G4 and G5 files (and their pre-registration logs) again byte for byte.
- G2's pools are small (28 and 15) because few fresh slots have a rule candidate: R1's rules rarely conclude these relations. G5 makes only nationality hypotheses (S11c's choice: country and nationality).
- The dev run of `--gen-rules` also wrote S8a's R1 and R2 dev files again; both are byte-identical to S8d's.

**Calibration (Part 3).** S15's choose half gives 342 graded items (108 true; S8c's grader graded 327 of them). The fresh dev pools give 1,858 graded items (366 true). Each (source, group) cell:

| Source | Group | own cell: true (pos) / false (pos) | uses | positives | LR+ | no minimum | M |
|---|---|---|---|---:|---:|---|---|
| DB-IB | dates | 2 (1) / 119 (0) | pooled | 23 | 19.03 | may carry | may carry |
| DB-IB | lexical | 424 (11) / 1,129 (2) | own | 13 | 10.60 | may carry | may carry |
| DB-IB | links | 2 (0) / 268 (0) | pooled | 23 | 19.03 | may carry | may carry |
| DB-IB | places | 46 (8) / 210 (1) | own | 9 | 19.56 | may carry | **may not** |
| WP-LEAD | dates | 2 (0) / 119 (7) | pooled | 173 | 4.21 | may not | may not |
| WP-LEAD | lexical | 424 (78) / 1,129 (66) | own | 144 | 3.13 | may not | may not |
| WP-LEAD | links | 2 (0) / 268 (1) | pooled | 173 | 4.21 | may not | may not |
| WP-LEAD | places | 46 (15) / 210 (6) | own | 21 | 9.94 | may not | may not |
| GN | places | 46 (2) / 210 (0) | own | 2 | 13.04 | may carry | **may not** |

- **E5 (miss).** DB-IB places and links still carry without the minimum, but DB-IB lexical now carries too (LR+ 10.60, just over 10): the closed-world nationality slots add many graded false items on which DB-IB is silent.
- **E6 (hit).** Under M, GN places can no longer carry (2 positives).

**The choice on dev (E7: hit).** Without the minimum, the dev pools give 3 supported items, all locations carried by DB-IB: Eden Project in Cornwall (true), Empire State Building in Manhattan (true) and Finger Lakes in "New York" (graded false). Under M they give none. M removes 3 graded promotions at 0.667, fewer than 5 and under 0.9, so **M applies**. Known is empty under both.
- The Finger Lakes item is right in plain terms: Wikidata's location is the Finger Lakes region, inside New York State. New York State is coarse, so rule 3 does not take it, and the closed-world rule then calls it false. A coarse containing place on a closed-world slot would be better ungraded.

**The grader audit (E4: hit).** 100 graded dev items were drawn with seed 20261029 (`fresh_slots.py audit-draw`): 60 from the 1,854 where the new grade differs from S8c's grader or rests on rules 2–5, and 40 from the 346 others. Each was checked against Wikidata's entity pages (every rank, names, P131 / P276 / P17, P31; `fresh_slots.py audit-info`) and by reading the facts. The checks are in [`S8E_GRADER_AUDIT.tsv`](S8E_GRADER_AUDIT.tsv).

| Stratum | items | hand check opposite | unsure | error, unsure counted (Wilson 95%) | error, opposite only |
|---|---:|---:|---:|---|---|
| A (differs or rules 2–5) | 60 | 1 | 2 | 0.050 [0.017, 0.137] | 0.017 [0.003, 0.089] |
| B (agrees, other basis) | 40 | 1 | 0 | 0.025 [0.004, 0.129] | 0.025 [0.004, 0.129] |
| all | 100 | 2 | 2 | **0.040 [0.016, 0.098]** | 0.020 [0.006, 0.070] |

- Weighted by the strata's sizes, the error with unsure items counted is 0.046.
- The two opposite verdicts:
  - James Joyce, nationality "English": rule 5 says true (England is inside the United Kingdom, one of his citizenships). He was Irish. **Rule 5 can be wrong for a person from another part of the country.**
  - Chet Atkins, occupation "Musician": both graders say false, because S15 drew "Musician" as the slot's false value. He was a musician (Wikidata lists guitarist, jazz musician, country musician). This is an error of S15's drawn false values, not of the new rules.
- The two unsure items are "Italian" for Amerigo Vespucci (Republic of Florence) and Claudio Monteverdi (Republic of Venice). The usual word is Italian, but Italy is not listed and contains those states rather than lying inside them. The closed-world rule calls them false.
- Of the 41 items S8c's grader also graded, the hand check agrees with it on 39. The two others are Chet Atkins (above) and Algernon Swinburne's death place "London", where S15 drew London as the false value although the true value, Putney, is inside it. The new grader gets Swinburne right.

**A further look, not registered (dev only).** Every rule-5 and rule-3 match on dev was read:
- Rule 5 matched 26 dev items. At least 4 are wrong: James Joyce "English", Emma Goldman "Scottish", Gordon Banks "Scottish" and Ian Botham "Scottish". So rule 5 is about 15% wrong where it fires.
- Rule 3 matched 16. One is doubtful: Damascus located in "Turkey", through the Ottoman Syria vilayet (the Ottoman Empire has the alias "Turkey").
- Nothing is changed for the gate: the rules were fixed before the read. A next grader could drop rule 5 for persons or require the place of birth to lie in the named part.

### The gate read (once)

Read once on 2026-10-29, after commit 3d1c627 (which holds the test pools' hashes). The five test pools went through S8d's source stage with the calibration above and variant M, and were graded with the new grader on `s8e_slots_test.tsv`.

| Read | hypotheses (distinct) | graded | supported | graded supported | true | precision (Wilson 95%) | Known |
|---|---:|---:|---:|---:|---:|---|---:|
| **M (the gate)** | 1,915 | 1,911 | **0** | 0 | 0 | none | 0 |
| no minimum (not a gate) | 1,915 | 1,911 | 2 | 2 | 2 | 1.000 [0.342, 1.000] | 0 |
| M, S8c's grader (not a gate) | 1,915 | 22 | 0 | 0 | 0 | none | 0 |

**S8's gate is not met.** No test hypothesis is supported under M, so there are no graded promotions, against the 20 the gate needs. There is no false promotion to trace.
- Without the minimum, two hypotheses are supported, both true: the Baseball Hall of Fame in "Cooperstown, New York" and the Chrysler Building in "Manhattan". Both are G4 locations carried by DB-IB. Under M, DB-IB places cannot carry (9 positives on dev).
- **Known** is empty in every read.
- The refuting side works: 690 test hypotheses were refuted, 688 graded, and 97.8% of those are false.
- By generator: G2 15, G6/G7 1,030, G3 179, G4 518 and G5 579 hypotheses; none supported under M. By group: lexical 1,501, links 264, places 150.
- S8c's grader can grade only 22 of the 1,915 test hypotheses, because it treats these slots as open. The new grader grades 1,911.

**Why nothing is promoted.** A promotion needs a positive result from a source that may carry. On these slots the sources are nearly silent:

| Pool part | DB-IB positive | WP-LEAD positive | GN positive |
|---|---|---|---|
| dev (2,327 rows) | 5 (all location) | 198 (194 nationality, 2 location, 2 successor) | 0 |
| test (2,321 rows) | 4 (all location) | 211 (206 nationality, 4 location, 1 successor) | 0 |

- DB-IB holds no nationality, predecessor, successor or taxon fact for these subjects. Only a few locations.
- WP-LEAD names a nationality often, but it is right only 121 times in 206 on test (0.587). Its lexical LR+ is 3.1, under the bar of 10, so it cannot carry.
- So the gate is limited by the sources' coverage of these relations, not by the grader or by the minimum.

### Predictions scored

| # | Prediction | Result |
|---|---|---|
| E1 | S15 rebuilds byte for byte | **hit** |
| E2 | 2,000–6,000 true facts; nationality at least 40% | **hit**: 3,675; nationality 40.0% |
| E3 | S8d's 3 false promotions become true (25 of 25) | **hit** |
| E4 | audited error at most 5%, Wilson upper at most 0.15 | **hit**: 0.040 [0.016, 0.098] |
| E5 | DB-IB places and links carry; DB-IB lexical does not | **miss**: DB-IB lexical carries too (LR+ 10.60) |
| E6 | under M, GN places cannot carry | **hit** (2 positives) |
| E7 | dev chooses M | **hit** |
| E8 | 8–30 graded test promotions at 0.85 or more; gate not met for want of 20 | **miss**: 0 graded promotions (2 without the minimum). The gate is not met, for want of 20 |
| E9 | Known is empty | **hit** |
| E10 | at least 70% of test promotions are predecessor or successor via DB-IB | **miss**: none; DB-IB never covers those relations here |

7 of 10 hit. The misses share one cause: I expected DB-IB to hold predecessor and successor fields for these subjects, and it holds none.

### Decision

The gate is not met, so under the decision rule nothing changes: no ledger row is proposed, S8's reads keep S8c's grader, and the variant stays a measured option.

What this read does settle (report only):
- The new grader fixes S8d's three false promotions (25 of 25) and grades almost every fresh hypothesis (1,911 of 1,915, against 22 for S8c's grader). Its audited error is 0.040 [0.016, 0.098].
- Its weak spots, found on dev: rule 5 is about 15% wrong where it fires (persons from another part of the country), a coarse containing place on a closed-world slot is called false (Finger Lakes in New York), and S15's drawn false values are sometimes a broader class of a true value (Chet Atkins, "Musician") or contain it (Swinburne, London).
- The minimum of positives (M) removes promotions resting on very few positive results. On these slots it removed all of them, and the two it removed on test were true.
- The binding limit on new slots is source coverage. HANDOFF §5.13's next step is a source that covers nationality, predecessor and successor with LR+ of 10 or more, or new slots where DB-IB has fields.

### Checks

- `native_epistemic_test` passes 1,343 checks (50 new, in `test_wd_grader_S8e`). Seven planted bugs each fail it:
  - rule 3 ignoring the coarse flag (4 checks fail);
  - a finer place counted as a match (5);
  - rule 5 without the country-like flag (1);
  - nationality left out of the closed-world list (2);
  - the finer verdict ignored by `grade_slot` (2);
  - `load` keeping only the first name (2);
  - a drawn false value matched by its shown name only (1).
- The other 8 CTests of HANDOFF §4.4 pass (a fresh cmake build of this branch: `native_cyphalm_trie_selftest`, `native_fm_infinigram_test`, `native_thought_test`, `native_geometry_test`, `native_ecs_lci_test`, `native_scribe_selftest`, `native_cyphalm_neural_smoke`, `native_class_context_test`).
- S3's normal `--gaps` run is byte-identical: gap map `a77678d8ef7dfdef`, its cells and demand files, and the 7 v2 store files.
- S8d's calibration read (`--calibrate`) and its fresh read (`--fresh`) reproduce byte for byte: every output file is identical and the printed lines differ only in the time line. Without `--s8e`, `pattern_learn --sources` is S8d's.
- The grader without tables is S8c's grader (`grade_slot` with no tables; checked in the test).

### Where things are

- Pools (not committed): `/root/work/s8e/pools/{dev,test}` (the generators' outputs, with their logs). The same pools in S8d's fresh-set format for S10, with the source stage's file and cost file: `/root/work/s8e/s10/{dev,test}` (`src.s8d_fresh.tsv`, `src.s8d_fresh_ran.tsv`, `s8a.fresh_g2.*`, `s8b.fresh_links.*`, `a.fresh_g3.*`, `g.fresh_g4.*`, `g.fresh_g5.*`). The dev part's source stage is under M, the chosen variant.
- Runs: `/root/work/s8e/runs/devread3` (dev, calibration, audit draw), `/root/work/s8e/runs/gate` (the gate read), `/root/work/s8e/runs/s3` (S3 parity), `/root/work/s8e/runs/{calib2,read2}` (S8d's reads again).
- Grader tables: `bench/eval/s8e_grader_tables.tar.gz`; raw responses in `/root/work/s8e/grader/run/raw`.
- Audit: `docs/reports/S8E_GRADER_AUDIT.tsv`; the Wikidata pages read for it in `/root/work/s8e/audit`.

### How to reproduce

```bash
g++ -O2 -std=c++20 -I native/include -pthread native/tools/pattern_learn.cpp -o pattern_learn
F=/root/work/f2; E=/root/cdata/enwik8; V=/root/work/sem/out/enwik8; D4=/root/work/s15/d4
# the set (fetches into $RUN/raw the first time; tar -xzf bench/eval/s8e_slots_wikidata_cache.tar.gz -C $RUN for 0 requests)
python3 -I scripts/fresh_slots.py build $E /root/work/f2/basefix/pages.tsv $D4 $F/v2/aug/facts_f2.tsv $F/v2/aug/seeds.tsv $RUN bench/eval
# the grader's tables (or unpack bench/eval/s8e_grader_tables.tar.gz into $G)
python3 -I scripts/fresh_slots.py grader $GRUN $G $D4 bench/eval/temporal_holdout_{dev,test}.tsv bench/eval/s8e_slots_{dev,test}.tsv
# the pools, for P in dev test
A="$E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval"
./pattern_learn --gen-rules $A $P/s8a --fresh-slots bench/eval/s8e_slots_$P.tsv
./pattern_learn --gen-links $A $P/s8b --fresh-slots bench/eval/s8e_slots_$P.tsv
./pattern_learn --gen-anomaly $A $P/a --fresh-slots bench/eval/s8e_slots_$P.tsv
./pattern_learn --gen-analogy $A $V data/cyphalm/simvec_ppmi32.bin $P/g --fresh-slots bench/eval/s8e_slots_$P.tsv
# calibration, the choice on dev, S8d's promotions graded again; add the five test pools and --read-test for the gate read
S="--sources bench/eval/temporal_holdout_dev.tsv tables $F/wikidata/wd_values.tsv $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv"
./pattern_learn $S out/s8e S8D_R1_DEV S8D_R2_DEV S8D_PROMO --s8e $G bench/eval/s8e_slots_dev.tsv bench/eval/s8e_slots_test.tsv \
  --s8e-pool dev G2 dev/s8a.fresh_g2.hyps.tsv --s8e-pool dev G6G7 dev/s8b.fresh_links.hyps.tsv --s8e-pool dev G3 dev/a.fresh_g3.hyps.tsv \
  --s8e-pool dev G4 dev/g.fresh_g4.hyps.tsv --s8e-pool dev G5 dev/g.fresh_g5.hyps.tsv --s8e-regrade S8D_READ.s8d_fresh.tsv
# the audit
python3 -I scripts/fresh_slots.py audit-draw out/s8e.s8e_calib.tsv audit.tsv
python3 -I scripts/fresh_slots.py audit-info $ARUN audit.tsv bench/eval/temporal_holdout_dev.tsv bench/eval/s8e_slots_dev.tsv
```

- `tables` is S8c's; `S8D_R1_DEV`, `S8D_R2_DEV`, `S8D_PROMO` and `S8D_READ` are S8d's files (its report).
- Lines start with `S8E`; the read files are `*.s8e_dev_{nomin,min,ran}.tsv`, `*.s8e_calib.tsv` and, with `--read-test`, `*.s8e_test{,_ran,_other,_oldgrader}.tsv`.
- Times on the shared machine (load about 28 on 4 cores): `--gen-rules` 10–11 minutes, `--gen-links` and `--gen-analogy` about 2 minutes each, `--gen-anomaly` 25–51 minutes (G3's planted read mines the rules 500 times), the `--sources --s8e` read about 5 seconds. The grader's tables took about 40 minutes of throttled requests.

### Changes from the registration

- The audit's hand check marks an item "unsure" when reasonable readers could differ. The headline error counts unsure items as errors; the opposite-only rate is shown beside it.
- `fresh_slots.py` gained `audit-draw` and `audit-info` for the audit. `audit-info` reads Wikidata's entity pages (all ranks) rather than the grader's tables, so the check does not reuse the grader's data.
- With `--fresh-slots`, `--gen-anomaly` still runs G13's planted and real reads before G3 and stops after G3's pool; G13's outputs are not used.
- The registration said the set's response cache is committed beside it; it is (0.42 MB). The grader's raw responses (32 MB) are not; the three tables they make are (0.77 MB).
- The pools were built on this branch's base (`4331619`). On `main`, S11e's variants of G3 and G4 (PLAN ledger 76) sit behind flags (`--g3-weights`, `--g4-date-lambda`), so `main`'s default generators are the ones used here. Other changes on `main` since the base (F2c's `--lexicon-type-check`, off by default; S9b's one-source Wikidata rule) were not run here. S11e's own fresh read uses this set with its variants.
