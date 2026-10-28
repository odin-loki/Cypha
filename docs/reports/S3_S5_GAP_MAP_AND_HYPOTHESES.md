# S3 and S5: the gap map and the hypothesis store

**Odin Loch** · 2026-10-28 · DISCOVERY Track S, items S5 and S3 (order S1 → S2 → S5 → S3)

**Questions.**
1. Can the hypothesis store (record, lifecycle, tiers, taint, history, firewall) be a small header-only library, with invariants I1, I4, I7 and I8 as known-answer CTests?
2. Can the gap map list what the fact store lacks, with VOI: schema, demand, contradiction and certainty gaps?
3. Do CyphaLM's own "I don't know" tags land on real gaps?

**Answers.**
1. **Yes.** `hypothesis.hpp` reuses `epistemic.hpp` (cards, labels, states) and `ecs.hpp` (opinions, fusion, admission). 321 new known-answer checks pass under g++ and clang++. Four planted bugs, one per invariant, each make the test fail.
2. **Yes.** 27,842 gaps on the v2 default store, each with VOI and a status:
   - 26,333 schema gaps;
   - 128 contradiction gaps;
   - 1,342 certainty gaps;
   - 39 demand gaps.

   P(exists) is Wikidata's type-level rate over the store's empty slots. On the held-out Wikidata fifth it predicts 0.0471 against 0.0476 observed. Its Brier score is 0.0301, against 0.0408 for the store's own f.
3. **No.** 0 of 750 dev tags name their question's slot (Wilson 95% upper bound 0.005). The relation is right for 4 of 373. 320 of the 334 tags that name an entity name a training entity that the store has a page for. So tag demand is logged, but it does not rank gaps (weight 0).

PLAN rule 10 holds. Neither store answers anything. Wikidata gives only a type-level rate, never a value. Strings are compared only for grading, on dev. The frozen test file was not opened.

Files:
- [`hypothesis.hpp`](../../native/include/cypha/eval/hypothesis.hpp) (S5);
- [`gap_map.hpp`](../../native/include/cypha/eval/gap_map.hpp) (S3);
- [`epistemic_test.cpp`](../../native/tools/epistemic_test.cpp): the new checks, in CTest `native_epistemic_test`;
- [`pattern_learn.cpp`](../../native/tools/pattern_learn.cpp): `--gaps`.

Spec: [`DISCOVERY.md`](../DISCOVERY.md) sections 1, 2.3, 3.2, 3.5, 4, 6.1, 6.3, 7.2–7.5 and 11. Earlier work: [S1_S2_EPISTEMIC.md](S1_S2_EPISTEMIC.md), [D4](D4_WIKIDATA_SEEDS_AND_FACT_SHARD.md), [F2R](F2R_RAW_BYTE_PATTERNS.md), [A1c](A1C_ENTITY_FAMILIARITY_AND_COPY.md).

---

## Predictions (registered before the first run)

Written into this file before any gap-map run and before the new tests were first compiled. Scored at the end.

**The hypothesis store (S5).**

| # | Prediction |
|---|---|
| H1 | Known-answer checks for I1, I4, I7 and I8, the lifecycle and the tiers all pass under g++ |
| H2 | Each of four planted bugs makes `native_epistemic_test` fail: (a) a derived item does not inherit taint; (b) promotion counts the generator's own evidence; (c) a refuted hypothesis does not block regeneration; (d) the wild tier's card has no label |

**The gap map (S3), on the current fact store (the v2 default run).**

| # | Prediction |
|---|---|
| G1 | Entity types by F2's rules: person 1,700–2,600; country 100–300; place 800–1,600; other 4,000–5,500 |
| G2 | Store f(T,R): f(person, birth_year) ≥ 0.55; f(person, birth_place) 0.15–0.40; f(country, capital) 0.25–0.70; f(person, occupation) 0.40–0.70 |
| G3 | Wikidata's f is above the store's in at least 90% of the cells with at least 30 Wikidata-covered entities and Wikidata f ≥ 0.10 |
| G4 | On the held-out fifth of Wikidata, P(exists) for the store's empty slots is best estimated by Wikidata's rate among the store's empty slots (lowest Brier), then Wikidata's overall rate, then the store's f (worst). The store's f underpredicts the observed rate by at least 0.10 on average |
| G5 | Schema gaps at the floor P(exists) ≥ 0.05: 10,000–40,000. Person relations hold at least 60% of them |
| G6 | Contradiction gaps (contested slots from F2's ECS fusion): 20–150. At most 10% of them are Contested by S2's d ≥ 0.5 rule |
| G7 | Certainty gaps: every stored extracted fact (1,394 with P ≥ 0.5) is Believed with grade Low; there are 0 Known items, so 0 single-source Known items |
| G8 | CyphaLM's dev tags in the six runs that emit them: 35–55% complete; at least 80% of complete tags carry the gap id that their own entity and relation give |
| G9 | Tags that land on their question's real gap (entity = the question's subject, relation = the question's relation, and that slot is a gap): at most 5%. The tag's relation equals the question's relation for 20–60% of tags whose question asks a store relation |
| G10 | Of the in-slice dev questions that ask a store relation, 30–70% ask about a slot that is a schema gap |
| G11 | Contradiction gaps are under 2% of all gaps but at least 30% of the top 100 by VOI |
| G12 | The 16-bit gap id that CyphaLM's tag uses collides: at least 1,000 gaps share their id with another gap |

**Cost.**

| # | Prediction |
|---|---|
| C1 | New headers at most 700 lines; at least 120 new test checks; `pattern_learn.cpp` grows by at most 250 lines |
| C2 | The `--gaps` run takes at most 45 s and 160 MB on one core. Its facts, seeds, patterns and Wikidata check are byte-identical to the v2 default run |

**Rules fixed before the run.**
- P(exists) of a schema gap is the estimator with the lowest Brier score on the held-out Wikidata fifth (G4). Relations with no Wikidata property keep the store's f.
- A schema gap is listed when P(exists) ≥ 0.05, the speculation tier's floor (DISCOVERY §4.3).
- VOI is DISCOVERY §6.3 with a placeholder test (sensitivity 0.9, specificity 0.9, cost 1), Impact = demand on the slot, and no interestingness term until S6.
- Dev only. The frozen test file is not opened.

---

## S5. The hypothesis store

`native/include/cypha/eval/hypothesis.hpp`, namespace `cypha::eval`, header-only, includes `epistemic.hpp`. Nothing in it reads or writes a file.

| Part | Spec | What it provides |
|---|---|---|
| Firewall | 1 | `Mode` (Answer, Discovery) and `StoreKind`; `may_read()`, `may_write()`. `KnowledgeStore`: `add()` makes a derived item inherit any premise's taint; `tainted()` follows the derivation to the observed items and fails closed; `answerable()` and `answer_view()` return only Known or Believed items with no taint, directly or through derivation |
| Record | 4.1 | `Hypothesis`: id, store key, statement, generator and version, derivation (premises with their P, rule, parameters), prior P0, scores, pre-registered tests (`HypTest`: predicted outcome, what refutes it, disconfirmation flag, independence, s and c, result), posterior, state, tier, taint, refutation cycle, history |
| Lifecycle | 4.2 | `HypState` (10 states) and `legal_transition()`. `HypothesisStore::transition()` refuses an illegal move and logs every legal one |
| Tiers | 4.3 | `tier_of(P0)`: conjecture ≥ 0.3, speculation ≥ 0.05, wild below. Budget shares 0.70 / 0.25 / 0.05 |
| Scoring | 6.1, 6.3 | `plausibility_prior()` = reliability × support × constraint. `expected_information_gain(p, s, c)` and `voi_priority()` = EIG (1 + Impact)(1 + I⁺) / Cost. The gap map uses the same two |
| Testing | 7.2 | `bayes_factor()` (s / (1 − c) or (1 − s) / c); `update_posterior()`; `decide()`: supported above 0.9 with independent evidence, refuted below 0.05, irreducible after 5 tests that moved it less than 0.05 in total, else queued |
| Promotion | 7.4, I4 | `promote()`: only independent test results become ECS items, at full mass. The prior is only the base rate `a`. Promoted only if `admission_decision()` says Admit. Then the taint is cleared, the item is written to the fact store, and the generator and rule get a hit |
| Refutation | 7.5, I7 | A refuted record is kept with its tests. Its generator and rule each get a miss. Its key blocks a new proposal for 10 cycles, unless new evidence arrives. The store has no erase |
| Labels | 10, I8 | `discovery_card()` gives the section-10 hypothesis card for a record in any state and tier. `label_discovery_text()` puts the SPECULATION label on any discovery text that lacks one. `is_labelled()` |

### What the tests pin down

All checks are in `native_epistemic_test`. They went from 195 to 516.
- **VOI.** EIG at p 0.5 with s = c = 0.9 is 1 − H(0.9) = 0.5310044. At p 0.2, s 0.9, c 0.8 it is 0.2534771. A perfect test gains H(p). A test with s = 1 − c gains 0.
- **Updating.** BF 9 and 1/9. Prior 0.7 with one positive gives 21/22.
- **Lifecycle.** 19 legal and 18 illegal arrows. Skipping scoring, leaving a final state and undoing a refutation are all refused.
  - An irreducible run: five tests with BF 1.02 move the posterior from 0.5 to 0.5247. That is 0.0247 in total, so after the fifth test the hypothesis is irreducible, and further tests are refused.
  - The history has exactly 14 entries.
- **I1.** A derived item arrives with its flag cleared but with a hypothetical premise: it is tainted at once, and so is an item derived from it.
  - A premise added later as hypothetical still taints the item at answer time.
  - A missing premise and a derivation cycle both count as tainted.
  - Contested, Ignorant and Vacuous items are never answers. The answer view of 14 items is exactly A, B and D3.
- **I4.**
  - **Teacher only.** Three teacher-sourced positives (one mislabelled "independent") push the posterior above 0.999. The hypothesis is never supported and cannot be promoted. The teacher's reliability is untouched.
  - **Rule hypothesis.** A G2 hypothesis with one dependent and one independent positive is supported. ECS then gives r = 1, P 0.8, grade Low: not admitted.
  - **Five independent positives.** ECS gives r = 5 (the dependent one is left out), b 5/7, u 2/7, P 0.9143, Moderate: promoted. The generator and the rule each get a hit.
  - **Derivations.** An item derived before promotion stays tainted; one derived after is clean.
- **I7.** "Sydney is the capital of Australia" is refuted by one negative (0.3 → 1/22). The record is kept with its test. G7 and its rule each get a miss.
  - The same key is refused for cycles 0 to 9, also after the record is retired, and allowed at cycle 10.
  - A different key is allowed at once, and so is the same key with new evidence.
- **I8.** Cards in every tier and state begin with `SPECULATION — not knowledge.`, including an empty record. Free text gets the label once, and labelled text is left alone. The HYPOTHETICAL card counts as labelled; an answer card does not.

**Planted bugs.** Each one was planted in a scratch copy of the headers and the test rebuilt:

| Bug | Checks that fail |
|---|---:|
| (a) a derived item does not inherit taint at `add()` | 4 |
| (a′) `tainted()` does not follow premises | 3 |
| (b) promotion counts the generator's own evidence | 12 |
| (c) a refutation never blocks regeneration | 12 |
| (d) a wild-tier card is written without its label | 8 |

Warnings: none under `-Wall -Wextra -Wpedantic` with g++ 13 and clang++ 18.

---

## S3. The gap map

### Method

`pattern_learn … --gaps FILE` runs after a normal run on the same store (`--cv --wikidata`, the v2 default). Its seeds, patterns, facts and Wikidata check are byte-identical to `/root/work/f2/v2/aug` (7 of 7 files).

1. **Entities and types.** Every main-namespace page except lists and disambiguations: 8,314. The type comes from F2's own sets, looked up by title, in this order:
   - country (country infobox, "Countries in …", US state): 58;
   - person: 1,329;
   - place: 568;
   - other: 6,359.
2. **Filled slots.** A slot is filled by a text seed or by a stored extracted value (P ≥ 0.5). A date slot also fills its year slot.
3. **f(T, R) five ways.**
   - the store's f;
   - Wikidata's rate over covered entities (those with at least one Wikidata value of the 22 properties), and over their empty store slots;
   - Wikidata's rate over all entities (no value counts as none), and over all empty store slots.

   Each is fit on four fifths of the subjects (D4's hash split), then scored on the held-out fifth's empty store slots, where the outcome is "Wikidata has a value". The lowest Brier score picks P(exists), which is then refit on all subjects. Relations with no Wikidata property (largest_city, location, predecessor, successor, owner, nationality, party, family, order, class) keep the store's f.
4. **Schema gaps.** An empty slot with P(exists) ≥ 0.05.
5. **Contradiction gaps.** A stored extracted value of a functional relation whose ECS fusion has competing-value mass (another value, or a conflicting text seed) or constraint mass (an impossible lifespan). p is its P over its P plus the best rival's; a seed counts 20/21.
6. **Certainty gaps.** A stored extracted value that is Believed at grade Low, or Known on one sentence. p = P.
7. **Demand.** Six run files with tags (dev rows only), one per distinct configuration:
   - A1c's D and D′ (`/root/work/answer/d_rows.tsv`, `d2_rows.tsv`);
   - the later frame variants E1a–E1d (`/root/work/qa2/runs/`).

   Parity reruns (P0, and the dev part of the test run) are byte-identical to D′ and left out. Each tag adds demand to its slot's gap: schema first, then contradiction, then certainty. With none of these, it opens a demand gap with p 0.5.
8. **Status.**
   - **Parked:** no automatic rung of the asking ladder (P1) can fill the slot now. Either the relation has no kept pattern (rung 1), or the entity has no page. Rung 3 (other pack sources) is closed by F2's rule that Wikidata values never become facts.
   - **Open** otherwise.
   - **Queued** when a tag named it.
9. **VOI.** This is §6.3 with a placeholder test (s = c = 0.9, cost 1) until S7 measures each test kind.
   - Impact = w × demand. I⁺ = 0 until S6.
   - P1's queue is the gaps that are not parked.
10. **Grading** (dev only). Does a tag name its question's slot?
    - The subject is the in-slice source title, or E1's entity for not-in-slice questions.
    - The relation comes from `q_rels`, D4's question-word map. It is used only here.

### The first run, and why the population changed

The first run scored the estimators only on covered entities. There, Wikidata's rate over covered empty slots won (Brier 0.0546), and the map had **77,015** schema gaps. Most were on "other" pages, with P(exists) 0.05–0.53. Author, founder, inventor, developer and creator alone gave 31,689.

But P(exists) is applied to every entity, and 63% of "other" pages (4,033 of 6,359) have no Wikidata value at all. Most are concepts: Anarchism, Albedo, Binary function. So I added the two all-entity estimators and scored all five on both populations. The selection uses the all-entity population, because that is where the estimate is used.

This change came after seeing the first run. Both runs are reported. The scores below use the registered wording ("the lowest Brier on the held-out fifth"), and where the first run's number differs, both are given.

### P(exists): calibration on the held-out Wikidata fifth

| Estimator (fit on four fifths) | Covered entities: Brier / log-loss / mean predicted | All entities: Brier / log-loss / mean predicted |
|---|---|---|
| Store's f | 0.0879 / 0.602 / 0.023 | 0.0408 / 0.281 / 0.014 |
| Wikidata, covered | 0.0566 / 0.283 / 0.112 | 0.0394 / 0.214 / 0.097 |
| Wikidata, covered, empty slots | **0.0546** / 0.275 / 0.101 | 0.0371 / 0.202 / 0.089 |
| Wikidata, all | 0.0603 / 0.309 / 0.074 | 0.0311 / 0.173 / 0.053 |
| Wikidata, all, empty slots (**chosen**) | 0.0598 / 0.309 / 0.066 | **0.0301** / 0.169 / **0.047** |
| Observed rate | 0.1045 (16,903 slots) | 0.0476 (37,082 slots) |

- **The store's f is the worst on both populations.** It underpredicts by 0.081 (covered) and 0.034 (all), 3.4 times too low on all entities. The store is sparse: most relations have no kept pattern.
- **The chosen estimator is calibrated in the large:** 0.0471 predicted against 0.0476 observed.
- **Caution.** The 302 T-QA dev and test subjects were never sent to Wikidata (D4). They count as "no value", a small downward bias (3.6% of entities).

### f(T, R): the main cells

f_wd is Wikidata's rate over covered entities. P(exists) is the chosen estimator.

| Type | Relation | Entities | f_store | f_wd | P(exists) | Kept pattern | Schema gaps |
|---|---|---:|---:|---:|---:|---|---:|
| country | capital | 58 | 1.000 | 1.000 | 1.000 | no | 0 |
| country | language | 58 | 0.983 | 0.951 | 0.000 | yes | 0 |
| country | founded | 58 | 0.000 | 0.976 | 0.690 | no | 58 |
| person | birth_year | 1,329 | 0.873 | 0.943 | 0.479 | yes | 169 |
| person | birth_date | 1,329 | 0.553 | 0.776 | 0.505 | yes | 594 |
| person | birth_place | 1,329 | 0.376 | 0.917 | 0.780 | yes | 830 |
| person | death_place | 1,329 | 0.082 | 0.790 | 0.701 | yes | 1,220 |
| person | death_date | 1,329 | 0.379 | 0.653 | 0.404 | yes | 826 |
| person | occupation | 1,329 | 0.448 | 0.977 | 0.877 | no | 734 |
| person | spouse | 1,329 | 0.018 | 0.526 | 0.470 | no | 1,305 |
| person | nationality | 1,329 | 0.455 | — | 0.455 (store) | no | 724 |
| place | country | 568 | 0.002 | 0.930 | 0.776 | no | 567 |
| place | located_in | 568 | 0.426 | 0.674 | 0.350 | no | 326 |
| other | country | 6,359 | 0.013 | 0.533 | 0.190 | no | 6,276 |
| other | founded | 6,359 | 0.017 | 0.356 | 0.116 | no | 6,253 |

The full table (132 cells) is `gap_map.tsv.cells.tsv`.

### The gaps

| Kind | Gaps | Open | Queued | Parked |
|---|---:|---:|---:|---:|
| Schema | 26,333 | 4,587 | 4 | 21,742 |
| Contradiction | 128 | 128 | 0 | 0 |
| Certainty | 1,342 | 1,338 | 4 | 0 |
| Demand | 39 | 0 | 1 | 38 |
| **All** | **27,842** | 6,053 | 9 | 21,780 |

- **P1's queue (not parked) holds 6,062 gaps.**
- **Schema gaps by type:**
  - other 12,529 (47.6%);
  - person 10,577 (40.2%);
  - place 3,108;
  - country 119.
- **Schema gaps by relation:**
  - country 6,901;
  - founded 6,871;
  - genre 1,317;
  - spouse 1,305;
  - predecessor 1,231;
  - successor 1,225;
  - death_place 1,220;
  - birth_place 830;
  - death_date 826;
  - occupation 734;
  - nationality 724;
  - birth_date 594;
  - currency 564;
  - language 550;
  - capital 541;
  - death_year 402;
  - located_in 326;
  - birth_year 169;
  - largest_city 3.
- **Parked is the rule, not the exception.** 21,742 of 26,333 schema gaps (82.6%) are in relations with no kept pattern. The 4,591 open or queued ones are persons' dates and places (4,041) and places' languages (550).
- **Open does not mean the next read will close it.** The kept patterns already ran on every article. An open gap closes only when the patterns or the text change.

**The floor, checked on the held-out fifth.** "Confirmed" means Wikidata has a value for the slot:

| Floor on P(exists) | Schema gaps | Expected real (sum of P) | Held-out fifth: confirmed / listed |
|---|---:|---:|---|
| 0.05 (registered) | 26,333 | 7,181 | 1,333 / 4,589 = 0.290 [0.278, 0.304] |
| 0.10 | 22,911 | 6,906 | 1,315 / 4,417 = 0.298 |
| 0.20 | 7,974 | 4,586 | 867 / 1,411 = 0.614 |
| 0.50 | 4,006 | 2,929 | 589 / 794 = 0.742 [0.710, 0.771] |
| 0.80 | 737 | 647 | 128 / 149 = 0.859 [0.794, 0.906] |

The listed gaps are as real as their P says (mean P(exists) 0.273, confirmed 0.290).

**Hand check: 20 random schema gaps**, judged from my own knowledge (`hand20_schema.tsv`). Does a value exist in the world?
- 4 yes (Burroughs' and Rogers' spouses, Elton John's occupation, Gilles Apap's birthplace);
- 2 arguable;
- 2 that exist but were never recorded (Emperor Yōmei's birthplace, Bumin Khan's birth date);
- 12 no.

That is 4–8 of 20 (0.20–0.40) against a mean P(exists) of 0.31 for the sample. The 12 "no" are mostly "other" pages asked for a founding date or a country: Cone, Data compression, Elementary particle. One is a living person's death place (Dava Sobel). One is a type error: Bruno of Querfurt, a person, is typed "other". The queue shows another: "AWK programming language" is typed person.

### Contradiction gaps

- **128 slots:** birth_place 56, birth_date 31, birth_year 30, death_year 7, death_place 3, death_date 1. One more slot (Chad's two official languages) was dropped by the functional-slot rule of §2.3.
- **Contested by S2's d ≥ 0.5 rule: 0.** One sentence of evidence keeps d below 0.5 (largest 0.472).
- **ECS already holds every one back.** All 128 leading values are candidates (P 0.500–0.740); none is admitted.
- **An automatic split:**
  - 29 places nested by name ("Duluth" against "Duluth, Minnesota", the chunk-against-link split F2R found);
  - 30 other places;
  - 63 dates with two or more values;
  - 6 lifespan violations only.
- **A random 30, judged by hand** (`hand30_contradiction.tsv`):

  | Class | Count | Examples |
  |---|---:|---|
  | Granularity: one place inside the other | 14 (0.47 [0.30, 0.64]) | Phoenix / Arizona; Weymouth / Weymouth, Massachusetts |
  | The stored value is wrong | 10 (0.33 [0.19, 0.51]) | Kafka "1852", Enya "1987", George Washington "1708", Ørsted "1778" |
  | The stored value is right, the rival wrong | 5 | Hemingway's death-year seed "1960"; David Thompson's seed "Quebec"; Clarke's "2001" |
  | The 2006 text is wrong | 1 | William Kidd 1654 |

  So 16 of 30 (0.53 [0.36, 0.70]) point at a real error, and 14 are granularity that a `located_in` check would remove. Some pages hold two people ("Gilbert and Sullivan", "Auguste and Louis Lumière"), which is G14's identity variable.
- **71 more functional slots** have two text-seed values (an upper bound, places inside each other included). ECS never sees these, because seeds are not fused.

### Certainty gaps

- All 1,394 stored extracted values are **Believed, grade Low**. No item is Known, so there are no single-source Known items.
- 1,342 slots hold them.
- Their p is high (median 0.950), so their EIG is small (0.116 bits at P 0.95).

### Demand: CyphaLM's own tags (dev)

| Run | Dev rows | Tags | Complete | Gap id matches its own entity and relation |
|---|---:|---:|---:|---:|
| D (A1c) | 1,040 | 106 | 38 | 38 |
| D′ (A1c) | 1,040 | 136 | 56 | 56 |
| E1a | 1,040 | 130 | 60 | 60 |
| E1b | 1,040 | 130 | 61 | 48 |
| E1c | 1,040 | 118 | 46 | 40 |
| E1d | 1,040 | 130 | 61 | 48 |
| **All** | 6,240 | **750** | **322 (0.429 [0.394, 0.465])** | **290 of 322 (0.901 [0.863, 0.929])** |

- **Every tag says `UNKNOWN_ENTITY`.** The 750 tags come from 93 distinct dev questions and name 44 distinct entities.
- **Where they land:**
  - 419 name no slot (cut off before the entity closes, or no relation);
  - 284 land on demand gaps;
  - 28 on schema gaps;
  - 19 on certainty gaps.
- **Clusters by (relation, type):** definition/other 190 (24 entities), definition/unknown 143 (4), location/other 36 (2), birth_date/person 31 (6), death_date/unknown 21, death_date/person 16, founder/other 14, birth_date/unknown 14, author/unknown 12, birth_place/unknown 10, definition/place 9, author/other 7, location/place 7, and five smaller ones.
- **The entities are training entities.** 320 of the 334 tags with an entity (0.958) name a page that the store has. These are A1c's don't-know entities from after the slice: Antares, Discworld, Economy of the Cayman Islands. "UNKNOWN_ENTITY" is true of CyphaLM's training slice and false of the store.

**Do the tags land on a real gap?** (grading only, dev only)

| Step | Tags |
|---|---:|
| Tags | 750 |
| Complete | 322 |
| Their question asks a store relation | 373 |
| The tag's relation is the question's | 4 (1.1% of 373) |
| The tag's entity is the question's subject | **0** |
| The tag names its question's slot | **0** (Wilson upper 0.005) |
| … and that slot is a gap | **0** |

Read looser, "the tag's slot is some gap in the map": 47 of 750 (0.063). This is A1c's finding again, now measured against the store: nothing points the tag at the question's entity.

**The dev questions' own slots.** Question words give the relation; the subject is the title or E1's entity:

| Slot | In slice (100) | Not in slice (30) |
|---|---:|---:|
| No store relation (definitions, numbers, …) | 67 | 5 |
| No page in the store | 0 | 21 |
| Filled | 18 | 3 |
| Schema gap | 6 (first run: 11) | 0 (first run: 1) |
| Empty, below the floor | 9 (first run: 4) | 1 (first run: 0) |

So of the 33 in-slice questions that ask a store relation, 15 ask about an empty slot. Under the chosen estimator, 6 of those are listed gaps. If the demand came from the question rather than the tag, it would land on these slots (see Flags).

### VOI and the queue

- **Top 100 of all gaps:** contradiction 45, demand 39, schema 15, certainty 1. The demand gaps rank high only through p = 0.5. All but one are parked ("definition" is no store relation).
- **Top 100 of P1's queue:** schema 53, contradiction 45, certainty 1, demand 1. Its top 15 are one queued demand gap and 14 contradictions near p 0.5, for example Al Gore's two equal birth places "Washington" and "Washington, D.C." (P 0.725 each).
- **The demand weight, compared.**

  | Weight | Top 100 of the queue | Its top 8 |
  |---|---|---|
  | 0 (chosen) | schema 53, contradiction 45, certainty 1, demand 1 | contradictions and one queued demand gap |
  | 1 (the spec's Impact = demand) | schema 50, contradiction 45, certainty 4, demand 1 | slots the tags named: Carol Kane's and Dean Kamen's death dates (demand 8 each), Blind Blake's birth date (7), Amr Diab's and Coco Chanel's birth dates (7) |

  None of these questions asked about those people.

### Gap ids

- CyphaLM's tag carries a 16-bit id (`G-%04X` of FNV-1a over "entity|relation").
- 27,842 gaps get 22,506 distinct ids, and 9,981 gaps share one. A random hash would give 22,683 distinct ids.
- So the map's key is (kind, entity, relation), and the id is only a label. It is kept so that CyphaLM's tags can be read back.

---

## Decisions (from the evidence)

Odin: "try a lot of things, profile and record everything, then pick a fact-based decision."

1. **P(exists) = Wikidata's rate over the type's empty store slots, over all entities.** Brier 0.0301 against 0.0408 for the store's f, on 37,082 held-out slots. Calibrated: 0.0471 against 0.0476. It is a type-level rate; no Wikidata value enters the map. The tool picks it on every run by the same rule, so a better store can change the choice.
2. **The floor stays at 0.05, as registered.** The table above shows what a higher cut buys: at 0.50 the list is 4,006 gaps and 0.742 confirmed. P1 can take its own cut; VOI orders the list anyway.
3. **Tag demand is logged but not ranked (`--demand-weight 0`, the tool's default).** 0 of 750 dev tags name their question's slot. With weight 1, the queue's top 8 are slots no question asked about. The demand column keeps every count, so nothing is lost if the tags improve.
4. **Contradiction gaps only on functional relations (§2.3).** This drops 1 slot (Chad's two official languages).
5. **P1's queue leaves out parked gaps.** 6,062 of 27,842 gaps have a rung that can run.
6. **The map's key is (kind, entity, relation), not the 16-bit id.** About 36% of gaps would collide on the id.

---

## Predictions, scored

| # | Prediction | Result | |
|---|---|---|---|
| H1 | I1, I4, I7, I8, lifecycle and tiers pass | 516 / 516 checks, g++ and clang++ | **Hit** |
| H2 | Each of four planted bugs fails the test | 4, 12, 12 and 8 failed checks (and 3 for a fifth bug) | **Hit** |
| G1 | person 1,700–2,600; country 100–300; place 800–1,600; other 4,000–5,500 | 1,329; 58; 568; 6,359. F2's rules type far fewer pages than I thought | **Miss** |
| G2 | f(person, birth_year) ≥ 0.55; birth_place 0.15–0.40; f(country, capital) 0.25–0.70; occupation 0.40–0.70 | 0.873; 0.376; **1.000**; 0.448. Every country page has its capital from the infobox | **Miss** (3 of 4) |
| G3 | Wikidata's f above the store's in ≥ 90% of the qualifying cells | 21 of 24 (0.875). The 3 are country cells the store already fills (capital 1.000 = 1.000; language 0.983 > 0.951; currency 0.810 > 0.805) | **Miss** |
| G4 | Order: Wikidata over empty slots best, Wikidata overall, store worst; the store underpredicts by ≥ 0.10 | Order right on the first (covered) population: 0.0546, 0.0566, 0.0879. On all entities the store is also worst. Underprediction 0.081 (covered) and 0.034 (all) | **Miss** (the size) |
| G5 | 10,000–40,000 schema gaps; person relations ≥ 60% | First run 77,015; with the all-entity estimator 26,333. Person 40.2% | **Miss** |
| G6 | 20–150 contradiction gaps; ≤ 10% Contested by d ≥ 0.5 | 128 (129 before the functional rule); 0 | **Hit** |
| G7 | All 1,394 stored extracted values Believed Low; 0 Known | 1,394 Believed Low; 0 Known | **Hit** |
| G8 | 35–55% complete; ≥ 80% of complete tags consistent | 42.9%; 90.1% | **Hit** |
| G9 | ≤ 5% land on their question's gap; relation right for 20–60% | 0%; **1.1%** (4 of 373) | **Miss** (the second part) |
| G10 | 30–70% of in-slice store-relation questions ask about a schema gap | First run 11 of 33 (33%); with the chosen estimator 6 of 33 (18%) | **Miss** |
| G11 | Contradiction gaps < 2% of all, ≥ 30% of the top 100 | 0.46% (0.16% in the first run); 45 of 100 (49 in the first run) | **Hit** |
| G12 | ≥ 1,000 gaps share a 16-bit id | 9,981 (54,860 in the first run) | **Hit** |
| C1 | Headers ≤ 700 lines; ≥ 120 checks; `pattern_learn.cpp` + ≤ 250 | 739; 321; +319 | **Miss** (2 of 3) |
| C2 | ≤ 45 s, ≤ 160 MB; store byte-identical to v2 | 28.3–38.7 s, 132–153 MB; 7 of 7 files identical | **Hit** |

8 hits, 8 misses. The misses are mostly about the store: it is sparser than I thought, F2's type rules type fewer pages, and CyphaLM's tags name the right relation even less often than A1c suggested.

---

## Ambiguities in the spec, and the choices made

1. **Lifecycle arrows the diagram does not draw.** The text adds:
   - under test → queued ("otherwise it waits in the queue");
   - supported → queued (ECS does not admit it yet);
   - parked → scored ("re-scored when the store changes").

   Any open state may be retired, except one under test. Promoted and retired are final. A scored hypothesis may be re-scored in place.
2. **"Moved the posterior by less than 0.05 in total"** is read as the sum of |change| over the tests. A posterior that swings and returns is not irreducible.
3. **Independent of the generator.** It needs the test's flag, and the test's source must not be the generator. A teacher test of a teacher hypothesis never counts, whatever its flag.
4. **"At full mass".** Each independent result is one ECS item (τ = the test's source trust, default 1); a negative result opposes. With W = 2 and a = P0, five independent positives are the least that reach Moderate at P0 = 0.7.
5. **Taint fails closed.** A missing premise, a cycle and a premise that later becomes hypothetical all taint the item.
6. **Answer mode and the gap map.** §1 lets only discovery write the gap map, but §3.5's tag adds demand from answers. So answer mode may write demand to the gap map and nothing else.
7. **"Single-source Known".** No item in the store is Known, so the certainty gaps are the Believed Low items. Every store item rests on one source, the 2006 article, so the definition would otherwise take in all 8,700.
8. **Parked gaps.** §4.2's "no feasible test now" is applied to gaps:
   - rung 1 needs a kept pattern;
   - rung 3 is closed by F2's rule;
   - rungs 4–5 are not run here.
9. **The tag's gap id** is 16 bits wide, so it cannot be the key.

---

## Flags for Odin

1. **Where should demand come from?** CyphaLM's tags name the wrong slot every time (0 of 750). The question itself names the right one:
   - 6 dev questions ask about schema gaps;
   - 9 more ask about empty slots below the floor;
   - 21 ask about subjects with no page.

   Logging demand from the question parse (E1's entity plus a relation) would fix that. But the relation here comes from `q_rels`, a hand-written word map that so far only grades. It never answers, but it would then drive bookkeeping. A1c's flag 2 (a pointer to the entity in the prompt) is the other route. Which do you want?
2. **Wikidata's type-level rate as P(exists).** §3.2 allows "from Wikidata in the pack", and it is the best-calibrated estimate by a wide margin. It is a rate over types and never a value. Please confirm.
3. **The reason code depends on which knowledge counts.** 320 of 334 tags name an entity the store has a page for, yet say `UNKNOWN_ENTITY`. That is right for CyphaLM's 7 MiB slice and wrong for the store. §3.1 should say which one the code describes.
4. **Types are the gap map's weakest part.** 76% of entities are "other", and the hand check found persons typed "other" and a programming language typed person. Finer types would sharpen P(exists) more than anything else here: work, organisation, concept, taxon, from E-track or Wikidata's P31 as a type-level signal.
5. **Seed conflicts never reach ECS.** 71 functional slots have two text-seed values.

---

## Costs

| Item | Value |
|---|---|
| `native/include/cypha/eval/hypothesis.hpp` | 500 lines (new) |
| `native/include/cypha/eval/gap_map.hpp` | 239 lines (new) |
| `native/tools/epistemic_test.cpp` | 515 → 979 lines (+465, −1); checks 195 → 516; 0.00–0.01 s |
| `native/tools/pattern_learn.cpp` | 2,709 → 3,026 lines (+319, −2): header comments and entity setup 50; f(T, R) and calibration 46; schema, contradiction and certainty gaps 54; demand and grading 70; outputs and summary 83; options and usage 16 |
| `native/CMakeLists.txt` | unchanged (the checks are in the existing CTest) |
| `--gaps` run | 28.3–38.7 s, 132–153 MB peak (the tool's own measure), one core, shared machine. The v2 run without it took 28.9 s and 118 MB |
| Outputs | `gap_map.tsv` 27,842 gaps (2.7 MB); `.cells.tsv` 132 cells; `.demand.tsv` 750 tags |

---

## Failure modes and cautions

- **Wikidata is not the world.**
  - A missing Wikidata value counts as "no value", so P(exists) is a lower bound for well-covered relations.
  - The 302 T-QA subjects were never asked, so they count as "no value".
  - On the hand check, 2 gaps "exist but were never recorded" anywhere.
- **One estimator for a mixed type.** "Other" mixes concepts, works and organisations. Its P(exists) is right on average and wrong for most single pages.
- **The VOI test is a placeholder.** With s = c = 0.9, every slot near p 0.5 ties at 0.531 bits. The order inside a tie is by kind and name, not by value.
- **Grading uses a question-word map.** `q_rels` reads relations from question words. A question it misreads is counted as "no store relation".
- **Granularity looks like contradiction.** 14 of 30 sampled contradiction gaps are a place and its region.

---

## Next steps

1. **S4.** The unseen estimates (Good–Turing, Chao1, Chapman with Wikidata as the second source) per (type, relation), and the frontier report from `gap_map.tsv.cells.tsv`.
2. **The asking ladder (P1) needs new rungs** before open gaps can close: pattern classes in the gaps and cross-article evidence (F2's next steps).
3. **A `located_in` check** before a place contest is opened. That would remove about half of the contradiction gaps.
4. **Odin's rulings on flags 1 and 2,** then a demand log from real questions.
5. **S6.** Novelty, impact and compression gain, then a VOI with interestingness.

---

## Reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
cmake -S native -B /root/cb_ecs && cmake --build /root/cb_ecs --target epistemic_test -j1
ctest --test-dir /root/cb_ecs -R native_epistemic_test          # 516 checks
E=/root/cdata/enwik8; W=/root/work/f2/wikidata/wd_values.tsv; DEV=bench/eval/t_qa_wiki_dev.tsv
A=/root/work/answer; Q=/root/work/qa2/runs
D=$A/d_rows.tsv,$A/d2_rows.tsv,$Q/E1a.rows.tsv,$Q/E1b.rows.tsv,$Q/E1c.rows.tsv,$Q/E1d.rows.tsv
O=/root/work/trackS; mkdir -p $O/final $O/run_w1
# the gap map (demand weight 0, the default)
./pattern_learn $E $O/final --cv --wikidata $W --dev $DEV --gaps $O/gap_map.tsv --demand $D --parse $A/parse_dev.tsv
# the comparison with demand weight 1
./pattern_learn $E $O/run_w1 --cv --wikidata $W --dev $DEV --gaps $O/run_w1/gap_map.tsv --demand $D --parse $A/parse_dev.tsv --demand-weight 1
# the store is the v2 default's
for f in facts_f2.tsv facts_f2.evidence.tsv facts_wd_check.tsv seeds.tsv patterns.tsv pages.tsv sample_admitted.tsv; do cmp $O/final/$f /root/work/f2/v2/aug/$f; done
```

- **stdout lines** (also in `stats.txt`):
  - `GAPCAL`: calibration, both populations;
  - `GAPMAP`, `GAPKIND`, `GAPSTATUS`, `GAPREL` and `GAPFLOOR`: the map;
  - `GAPID`: id collisions;
  - `CONTRADICTION` and `STORED_STATE`;
  - `DEMAND` and `DEMAND_CLUSTER`: the tags;
  - `GRADE` and `QSLOT`: dev grading;
  - `TOPQ`, `TOP100_ALL` and `TOP100_QUEUE`: VOI.
- **Outputs:**
  - `gap_map.tsv`: gap_id, kind, status, entity, type, relation, p_exists, f_store, f_wd, p, eig_bits, demand, voi, note; ranked by VOI;
  - `gap_map.tsv.cells.tsv`: f(T, R) all ways;
  - `gap_map.tsv.demand.tsv`: one row per tag, with its grading.
- **The first run** (covered-entity population) is `run1_covered.out`.
- **Hand checks:** `hand20_schema.tsv` (schema gaps, `random.seed(20261028)` over the final map) and `hand30_contradiction.tsv` (contradiction gaps, the same seed).
- **The planted bugs:** copy `native/include` to a scratch directory, make the one-line change named in the S5 table, then rebuild `epistemic_test.cpp` against the copy.
- **sha256 prefixes:**
  - `gap_map.tsv` a77678d8ef7dfdef;
  - `.cells.tsv` eed14241a6d6eff2;
  - `.demand.tsv` dd1c8ddac831dcb6.
