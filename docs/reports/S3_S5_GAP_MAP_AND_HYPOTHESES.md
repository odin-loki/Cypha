# S3 and S5: the gap map and the hypothesis store

**Odin Loch** · 2026-10-28 · DISCOVERY Track S, items S5 then S3 (order S1 → S2 → S5 → S3)

(Draft. Results are filled in after the runs.)

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
