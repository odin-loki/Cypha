# S11a: concept formation (G9) and open relations (G11)

**Odin Loch** · 2026-10-29 · DISCOVERY §5 (G9, G11), §8 (P5, P10), §13 S11 · follows [S8A_RULES_AND_CONTRADICTIONS.md](S8A_RULES_AND_CONTRADICTIONS.md), [S8D_PROMOTION.md](S8D_PROMOTION.md) and [S10_CURIOSITY_SCHEDULER.md](S10_CURIOSITY_SCHEDULER.md)

(The design, the reads, the decision rules and the predictions below were written before any G9 or G11 run. Results are added under them.)

**Question.** Two generators of S11's batch. G9 groups entities by what the store says about them and proposes classes the store does not name. G11 reads sentences that no kept pattern parses and proposes relations Cypha does not extract. Does G9 find a class when its name is taken away? Do G11's phrase clusters line up with real Wikidata properties? And do G9's predictions survive the loop?

**Gate (DISCOVERY §13 S11).** Rediscovery of a class (this report). ABC precision@k is S11b's. Promotion precision is reported with S8's gate (≥ 0.9 on at least 20 graded) as the reference.

**Rule 10.** G9 and G11 write hypotheses into Cypha's hypothesis store and nothing else. They never answer a question, and nothing here reaches `qa` or `chat`.

**No teacher.** G16 is not available (no Qwen inference in the container, PLAN ledger 54). A class is named by its intent (the values it shares), a relation by its phrase.

**Counted before registering** (sizes only, no generator output): store K (S8a's: every seed row with a relation `pattern_learn` knows) has 22,783 triples over 3,960 subjects. With objects normalised by D4's `norm_ent`, its non-date facts give 215 attributes held by at least 10 subjects, over 2,558 subjects. 143 of the attributes have between 15 and 500 subjects.

## G9: concept formation

### The formal context

- **Entities and attributes.** An attribute is (r, v): a non-date relation r of `pattern_learn` (types other than date and year) and a value v = `norm_ent(o)`. An attribute is kept when at least 10 subjects hold it (DISCOVERY's support guard). The entities are the subjects with at least one kept attribute.
- **A1, direct attributes** (the default).
- **A2, A1 plus one hop:** for a fact r(x, z) with r one of birth_place, death_place, location, located_in, and z holding country(z, c) or located_in(z, c) in the store, the attribute (r>country, c) or (r>located_in, c). Kept by the same rule. A2 is a dev choice (below).
- Each entity also records every relation it holds in the store, kept attribute or not (for the PCA counts below).

### Concepts

- Every closed attribute set with at least 10 entities in its extent (formal concept analysis), enumerated by LCM (prefix-preserving closure extension, Uno et al. 2004). The empty intent is left out. The enumeration stops at 500,000 concepts; a stop is reported.

### Candidate classes (G9's output)

- A concept is a **candidate class** when its intent has at least 2 attributes, its extent has at least 10 entities, and its extent's Jaccard with every single attribute's extent is below 0.9. A single attribute is a class the store already names; a conjunction close to one adds nothing.
- Named by its intent ("occupation = composer & nationality = german").

### Class rules: the information-gain test (DISCOVERY P5)

- For a candidate class C and an attribute b outside its intent: support = |ext C ∩ ext b|; PCA body = the members of C that hold any value of b's relation in the store; PCA confidence = support / PCA body.
- **Baseline:** the best single-attribute rule inside the intent: max over i in intent(C) of PCA confidence(i ⇒ b), counted over the whole context.
- **p-value:** P(Bin(PCA body, baseline) ≥ support), the binomial upper tail. A rule is a candidate when support ≥ 10 and PCA confidence ≥ 0.5. It is **kept** by Benjamini–Hochberg at q = 0.1 over every candidate rule (G2's guard, against G9's own baseline). So a kept rule says the class predicts b better than any one of its attributes does.
- A candidate class with at least one kept rule is a **kept class** (P5's "add to the hierarchy as Believed").
- T3 counts, as G2's: support and PCA body among the members in the hidden third (`gen_hidden_third`).

### Read 1, rediscovery of a held-out class (the gate)

- **Held-out classes:** the A1 attributes with 15 to 500 subjects in store K. A class is **dev** when bit 19 of FNV-1a(r + "\x1f" + v) is 0, else **test**. Test classes are read once, with `--read-test`.
- **For each held-out class H:** every fact r(x, o) with `norm_ent(o)` = v is removed from store K. G9 runs on the rest (context, concepts, candidate classes, class rules), exactly as above.
- **Measures**, extents compared by entity name, H's extent being its subjects in the full store K:
  - J_new(H): the best Jaccard of a candidate class's extent with H's. **H is rediscovered when J_new ≥ 0.5.**
  - J_named(H): the best Jaccard of a single attribute's extent (of the reduced store) with H's. **H is redundant when J_named ≥ 0.5**: the store already names a class close to it.
  - J_kept(H): as J_new, over kept classes only.
- **Null:** for each H, a random set of |ext H| entities, drawn from the reduced context's entities that hold a value of r in store K (all its entities if there are too few), with `mt19937_64(20261101 + i)` for the i-th held-out class. Its J_new is read against the same candidate classes as H's.
- **Rate** = rediscovered / held-out classes, with Wilson 95% intervals; the same for the null.
- **The gate is met** when, on the test classes, the rediscovery rate's Wilson lower bound is above the null rate's Wilson upper bound.
- **Dev choice:** A1 or A2, by the dev rediscovery rate. A2 is chosen only if its rate is at least 5 points higher; otherwise A1 stays (simpler). Nothing else is tuned.

### Read 2, promotion through the loop (S8d's pipeline)

- **Slots.** Dev: every (subject, relation) of `temporal_holdout_dev.tsv` with bit 17 of FNV-1a(subject) = 0 (S8d's choose half) and a non-date relation of `pattern_learn`. Test: the same on `temporal_holdout_test.tsv` (bit 17 = 0; S10 read the bit 17 = 1 half). Only subjects and relations are read to make them.
- **Hidden facts.** Every store K triple in those slots is removed. G9 (the attribute set chosen on dev) runs on the reduced store.
- **Hypotheses.** For each slot (s, r) whose subject is in the context: every (s, r, o) derived by a kept class rule whose class contains s, and not held by s. Score = the noisy-or of those rules' PCA confidences (G2's registered combination). P0 = clamp(score × S6's hard constraint, 0.02, 0.98).
- **Tests**, in cost order:
  - T1, S7's consistency;
  - T2 inconclusive (class rules have no consequence rules);
  - T3, the best class rule on the hidden third (`rule_holdout_test`);
  - T6, S7's compression (ε = 0);
  - T4, S7's corpus test (`pattern_or_comention`), the independent one;
  - T5 left out: Wikidata is what S15 grades against.
- **The plan.** Each test's s and c come from the dev hypotheses graded by S15 dev (add-one), with S7's earn-its-place rule (applicable ≥ 5% and Youden's J interval above 0). The test read uses the dev plan.
- **The loop.** S8a's loop: pre-registration (I5, I6), tests cheapest first, `decide` (a positive independent result is needed, S8d), BH among the supported.
- **The source stage.** `pattern_learn --sources … --fresh POOL EMPTY`, with S8d's calibration, unchanged. (s8c labels the pool's rows "G2-fresh"; G9's grading step reads them back as G9's.)
- **Grade.** The S15 rows of the split, by S8a's S15 grader (true when S15 lists the value for (s, r), false when S15's false row has it or S15 lists other values, else ungraded). Each distinct hypothesis once.
- **Reported:** supported after the generator loop, and supported after the source stage (S8d's gate pipeline), each with graded precision and Wilson 95%, by relation group.
- **Reference:** S8's gate, precision ≥ 0.9 on at least 20 graded supported.

## G11: open relations

### Unparsed pairs

- Every enwik8 article (8,587) and every sentence of it (`pattern_learn`'s reader). For every subject mention (`find_subject`, [P] for a sentence-initial pronoun) and every link that is not the page itself, `make_keys` gives the pattern key when the two are 1 to 6 tokens apart.
- A pair is **parsed** when a kept pattern fires on it (`apply_patterns`, the extraction pass's own matcher). Otherwise it is **unparsed** and is kept with its core key ("[S] was educated at [O]", [P] read as [S]).
- **Phrase filter:** the core's middle has at least one token with a letter.

### Clusters

- **C1, the exact core.** One cluster per core key.
- **C2, the connector.** The slot order plus the at most 2 words next to [O], after articles (the, a, an) next to [O] are dropped: "[S] was educated at [O]" and "[S] , educated at [O]" both become "S … educated at O".
- A cluster needs at least 5 distinct (subject, object) pairs (DISCOVERY's guard).

### Grading against Wikidata (S15)

- A pair (s, o) **holds property p** when the split's S15 file has a true row (s, p, v) with `norm_ent(v)` = `norm_ent` of o's link target. A pair is **gradable** when s is a subject of that file. Only S15's 38 properties can be seen; a pair holding another property counts as a miss.
- **Dev.** For each cluster, p* = the property held by the most gradable dev pairs (ties by name); dev precision = pairs holding p* / gradable pairs. A cluster is a **proposed relation** when it has at least 3 gradable dev pairs and dev precision ≥ 0.5.
- **Dev choice, C1 or C2:** by the number of distinct dev pairs that hold their cluster's p* in proposed clusters. C2 is chosen only if it has at least 10% more; otherwise C1 stays.
- **Test read (once).** For each proposed relation, the precision of its dev p* on its test-subject pairs. A proposal **holds** on test with at least 3 gradable test pairs and test precision ≥ 0.5. Pooled precision over all proposals' gradable test pairs, with Wilson 95%. Split by p*: one of `pattern_learn`'s relations (a phrasing the patterns miss) or one of S15's 15 added relations (a relation Cypha does not extract).

### Why G11's pairs do not go through the fact loop

- A G11 pair comes from a sentence on the subject's own page. T4, the loop's only independent test here, reads the same page. Its yes would not be independent of the generator, so the loop would promote on G11's own evidence.
- DISCOVERY P10 sends a new relation to naming (teacher or Odin), pattern seeding and F2. That is where G11's proposals go; this report measures whether they are real.

## Pools for S10

The promotion read writes `OUT.fresh_g9.hyps.tsv` and `OUT.fresh_g9.prereg.tsv` in S8a's fresh-set format. The source stage writes `src.s8d_fresh.tsv` and `src.s8d_fresh_ran.tsv`. Together they are a pool S10's replay can read.

## Reads of S15 test

Two final reads, each once: G9's promotion read on the bit 17 = 0 half (slots, then labels), and G11's proposals on test subjects' rows. Both are logged in S15's Reads table.

## Decision rules

- **G9, rediscovery:** the gate as above. If met, G9 is S11's concept generator, decided by measurement under PLAN ledger 55 and open to Odin's veto. If not, it is reported with its parts and nothing is changed after the read.
- **G9, promotion:** reported against S8's gate. Nothing is adopted from it.
- **G11:** proposals that hold on test are listed for naming (P10). Nothing enters the store.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| E1 | On the full store K with A1: 500 to 5,000 closed concepts, 200 to 3,000 candidate classes |
| E2 | 20 to 300 kept classes (at least one kept class rule) |
| E3 | Dev: A2 is not chosen (within 5 points of A1) |
| E4 | Test: 15% to 45% of held-out classes rediscovered (J_new ≥ 0.5) |
| E5 | The null rate is at most 3%, and the gate is met |
| E6 | At least 15% of test classes are redundant (J_named ≥ 0.5) |
| E7 | Rediscovery by kept classes is less than half of that by candidate classes |
| E8 | Promotion test read: 100 to 2,000 G9 hypotheses; fewer than 20 graded supported after the source stage, so S8's gate cannot be read (S8d's sources do not carry lexical relations) |
| E9 | Supported after the generator loop alone: graded precision below 0.9 |
| E10 | G11: 30,000 to 300,000 unparsed pairs; 100 to 2,000 C1 clusters with at least 5 pairs |
| E11 | Dev chooses C2 |
| E12 | Test: at least 10 proposals hold; pooled test precision of proposals between 0.4 and 0.8 |
| E13 | At least one proposal that holds on test maps to one of S15's added relations (educated_at, award_received, member_of, member_of_sports_team, position_held, employer, …) |

## Dev (2026-10-29)

Runs: `/root/work/s11a/runs/` (`dev.out`, `redisc_dev_A2.out`, `dev_promote.out`). Nothing in this section reads S15 test.

**G9 on the full store K (A1).** 215 attributes over 2,558 entities. 806 closed concepts, 575 candidate classes, 861 candidate class rules of which BH keeps 36, in 25 kept classes. Every kept rule is about occupations ("actor & writer ⇒ screenwriter", PCA 0.717 against its best single attribute's 0.370). The run takes 0.1 s.

**Read 1, dev classes (83).**

| Attribute set | Rediscovered (J_new ≥ 0.5) | Null | Redundant (J_named ≥ 0.5) |
|---|---|---|---|
| A1 | 0 / 83 [0.000, 0.044] | 0 / 83 | 7 (0.084) |
| A2 | 0 / 83 | 0 / 83 | 7 (0.084) |

- **Dev choice: A1.** A2 adds 11 attributes and changes no class's result.
- **The bar is not reached, but the signal is there.** The best J_new values are 0.40 to 0.49: songwriter 0.489 ("composer & musician & singer"), film producer 0.478, director 0.475. On 69 of the 83 classes G9's best class is closer to the held-out class than the random null's; on 11 it is further. Mean J_new 0.187 against the null's 0.061.
- **Why J stays under 0.5:** a candidate class is a conjunction of at least 2 attributes, so its extent is a subset of each of them. A held-out occupation overlaps several others (directors are also producers and screenwriters), and any one conjunction covers only part of it. Countries fail outright: most places hold `country` and little else, so removing the class removes them from the context (Canada: 22 of 51 members left in the context).

**Read 2, dev.** 2,129 dev slots; 410 have a subject left in the context; G9's 13 kept rules on the reduced store predict **none** of them. Every kept rule predicts an occupation from other occupations, and a slot hides all of its subject's occupations.

**G11, dev.** 54,115 (subject mention, link) pairs with a key; kept patterns parse 508; 49,152 are unparsed with a word between.

| Clustering | Clusters (≥ 5 pairs) | Proposed relations | Dev pairs holding their cluster's property |
|---|---|---|---|
| C1, exact core | 486 | 38 | 419 |
| C2, connector | 1,088 | 27 | 395 |

- **Dev choice: C1** (C2 is not 10% better).
- C1's proposals include S15's added relations: "[S] won the [O]" → award_received (9 / 18), "[S] attended [O]" → educated_at (6 / 8), "[S] graduated from [O]" → educated_at (6 / 7), "[S] was educated at [O]" → educated_at (4 / 8). The rest are phrasings of `pattern_learn`'s relations that its kept patterns miss: linked dates ("[S] ( <M> # , [O]" → birth_year, 174 / 177), "[S] died in [O]" → death_place, "[S] is a novel by [O]" → author, "[S] is the county seat of [O]" → located_in, "[S] was founded by [O]" → founder.

### Amendment 1 (after the dev reads, before any test read)

- **Read 1, secondary measures (added; the gate is unchanged).** On the test classes: the sign test of J_new against the null's J (one-sided, ties dropped); the mean J_new − null with a bootstrap 95% interval (2,000 resamples, seed 20261102); the share with J_new ≥ 0.3 against the null's. Dev: 69 against 11, p 1e-11; mean +0.127 [+0.099, +0.155]; 20 (0.241) against 0. These are reported beside the gate and decide nothing.
- **Read 2b, checked on dev only and dropped.** G9 on store T (S8a's 2006-text store) keeps 1 class rule and predicts nothing for S15 dev's bit 17 = 0 subjects. It is not read on test. Its code is in this commit and is removed in the next.
- **Read 2 is read on test as registered.** Dev predicts it will give no hypotheses, so G9 has no promotions and no pool for S10.

### Predictions, after dev (E1–E3, E11 can be scored now)

| # | Result |
|---|---|
| E1 | **Hit:** 806 concepts, 575 candidate classes |
| E2 | **Hit:** 25 kept classes |
| E3 | **Hit:** A1 kept (A2 changes nothing) |
| E11 | **Miss:** C1 chosen |

## Results: the test read (2026-10-29)

One run with `--read-test` at commit 913d2a7 (`/root/work/s11a/runs/test.out`). The final code (the unused grading path removed) reproduces it byte for byte, timings aside.

### G9, read 1: rediscovery of a class (the gate)

| Test classes (60) | Count | Rate [Wilson 95%] |
|---|---:|---|
| Rediscovered (J_new ≥ 0.5) | 0 | 0.000 [0.000, 0.060] |
| Null (random classes) | 0 | 0.000 [0.000, 0.060] |
| Redundant (J_named ≥ 0.5) | 4 | 0.067 |
| Rediscovered by kept classes | 0 | 0.000 |

- **The gate is not met.** G9 rediscovers no held-out class at the registered bar.
- **Secondary (amendment 1):**
  - G9's best class is closer than the null's on 50 classes and further on 3 (sign test p 2.8e-12);
  - mean J_new − null +0.113 [+0.086, +0.142];
  - J_new ≥ 0.3 for 13 classes (0.217) against 0 for the null.
  - So G9's concepts lie near real classes far more often than chance, but never cover one.
- **Closest test classes:**
  - producer 0.464 ("director & film producer");
  - record producer 0.436 ("musician & songwriter");
  - singer-songwriter 0.429 ("guitarist & musician & singer");
  - rock music 0.410 ("composer & musician & singer").
- **Why:** store K's attributes are mostly occupations (105 of 215), and occupations overlap. A conjunction of two or three of them is a subset of each and covers only part of a held-out one. Countries get no close class at all (best J_new 0.094 of 10): most places hold `country` and little else, so removing the class takes them out of the context. Nationalities reach 0.11 to 0.38, genres up to 0.41.

### G9, read 2: promotion

- **Test:** 2,124 slots; 393 slots have a subject left in the context. The reduced store keeps 2 class rules, and they predict **nothing**: 0 hypotheses.
- So G9 has no promotions. S8's gate cannot be read for it, and its pool files are written empty: there is nothing for S10 to schedule.
- The cause is the one found on dev: G9's rules predict an occupation from other occupations, and a hidden slot removes all of a subject's occupations.

### G11: open relations

38 proposals (C1, chosen on dev), each read once on test subjects' pairs:

| Dev property | Proposals | Hold on test | Pooled test pairs holding it | Precision [Wilson 95%] |
|---|---:|---:|---:|---|
| `pattern_learn`'s relations | 34 | 19 | 381 / 426 | 0.894 [0.862, 0.920] |
| S15's added relations | 4 | 2 | 23 / 43 | 0.535 [0.389, 0.675] |
| All | 38 | 21 | 404 / 469 | 0.861 [0.827, 0.890] |

- **New relations for Cypha that hold:** "[S] attended [O]" and "[S] was educated at [O]", both educated_at: 11 / 17 (0.647) and 6 / 10 (0.600).
- **Ones that do not:** "[S] won the [O]" → award_received, 4 / 11 (it also catches team titles such as the FA Cup and the World Series, and names prizes at another grain: "Nobel Prize" for the specific one); "[S] graduated from [O]" → educated_at, 2 / 5.
- **19 of the 34 known-relation proposals are dates** (354 / 370 pairs). They are not new phrasings. The year is a link inside a date ("(January 18, [[1882]] –"), and the kept date patterns already read the whole date through its date candidate (A. A. Milne's birth_date 1882-01-18 is in the v2 store). Only the year's link candidate is left unparsed, and G11 counts it. This breakdown is descriptive; the registered measure is the table above.
- **The other 15 known-relation proposals** (35 / 64 pairs; 6 hold) are phrasings the patterns miss:
  - "[S] was born at [O]" (birth_place 6 / 9);
  - "[S] is the county seat of [O]" (located_in 6 / 6);
  - "[S] is a city located in [O]" (located_in 3 / 3);
  - "[S] was directed by [O]" (director 3 / 5).
  - "[S] died in [O]" fails (7 / 19): it is often followed by a year link ("died in [[1331]]").
  - "[S] is the capital of the [O]" fails (0 / 4).

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| E1 | 500–5,000 concepts, 200–3,000 candidate classes | **Hit:** 806, 575 |
| E2 | 20–300 kept classes | **Hit:** 25 |
| E3 | A2 not chosen | **Hit** |
| E4 | Test rediscovery 15%–45% | **Miss:** 0% |
| E5 | Null ≤ 3%, gate met | **Partial:** the null is 0%; the gate is not met |
| E6 | At least 15% redundant | **Miss:** 6.7% |
| E7 | Kept-class rediscovery under half of candidate-class rediscovery | **Not scorable:** both 0 |
| E8 | 100–2,000 hypotheses; fewer than 20 graded supported | **Miss:** 0 hypotheses |
| E9 | Generator-loop precision below 0.9 | **Not scorable:** no hypotheses |
| E10 | 30,000–300,000 unparsed pairs; 100–2,000 C1 clusters | **Hit:** 49,152; 486 |
| E11 | Dev chooses C2 | **Miss:** C1 |
| E12 | At least 10 proposals hold; pooled precision 0.4–0.8 | **Partial:** 21 hold; precision 0.861, above the range |
| E13 | A proposal for one of S15's added relations holds | **Hit:** educated_at, twice |

5 hits, 2 partial, 4 misses, 2 not scorable.

### Decisions (registered rules)

- **G9 is not adopted** as S11's concept generator: the gate is not met. Its code stays for the next store; nothing is changed after the read.
- **G9 promotion:** no hypotheses, so nothing to adopt.
- **G11:** the 21 proposals that hold are listed for naming (P10). Nothing enters the store. The two educated_at phrasings are the first relation Cypha does not extract that its own reading proposes and Wikidata confirms.

### Checks

- `native_epistemic_test`: 1,237 checks (69 new). LCM is checked against brute force on 6 random contexts; candidate classes, class rules (PCA counts, the baseline, the p-value, BH, T3's hidden third) on a built example; G11's connector, filter, clusters and grade.
- **Planted bugs caught (6):** LCM without the prefix check (crashes, then fails); no duplicate check for candidate classes; the baseline as the minimum; Jaccard ignoring members outside the context; PCA counting every member; articles dropped at the far end of the connector.
- **The nine CTests of HANDOFF §4.4 pass.**
- **S3's `--gaps` run is byte-identical:** gap map, cells, demand and the v2 store files, as `/root/work/s4/runs/test/`.

### How to rebuild

```
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
pattern_learn --gen-concepts /root/cdata/enwik8 /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv \
  /root/work/f2/v2/aug/patterns.tsv bench/eval OUT --attrs A1 [--read-test]
```

Each part runs alone with `--part rediscover|promote|open`. A full run takes about 25 s and 171 MB.

## S11a-b: date links (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · follows the G11 caveat above · the same change is measured for S4b's unparsed rate in [S4_UNSEEN_AND_FRONTIER.md](S4_UNSEEN_AND_FRONTIER.md), "S4b-b"

(The definition, the reads, the rule and the predictions below were written and committed before any run of the changed code. Results are added under them.)

**Question.** 19 of G11's 38 dev proposals name a date relation (birth_year, death_year, founded). Their pairs are links to years or days inside a date, such as "(January 18, [[1882]] –". The reader's date candidate already reads that whole date, and the kept date patterns parse it. Only the year's link is left over as an "unparsed" pair. Should G11 skip a link that sits inside a date candidate? This is decided by measurement (PLAN ledger 55).

**Rule 10.** G11 proposes relations for Cypha's own store. Nothing here answers a question.

### Definition (one for G11 and U)

- **Date candidates** are the reader's own: `find_dates`, the K_DATE spans `gen_cands` already makes. They are a month with a year, a day and month with a year, or a lone year (100–2100, or any number followed by BC).
- **A link sits inside a date candidate** when every token of its display lies inside one date candidate of the same sentence: candidate start ≤ link start and link end ≤ candidate end.
- Examples:
  - In "(January 18, [[1882]] –", the date candidate is "January 18, 1882". "[[1882]]" is inside it. So is "[[January 18]]" when the day is linked.
  - In "in [[1882]], he moved", "[[1882]]" is its own one-token date candidate, so it is inside.
  - "[[War of 1812]]" is not inside: its display has more than the year.
- **The change for G11:** a (subject mention, link) pair whose link sits inside a date candidate is not a pair. It is not counted, not checked against the kept patterns, and not clustered. Nothing else in G11 changes: the keys, the clusters, the dev choice between C1 and C2, the grading.
- The helper is `span_within` in `native/include/cypha/eval/unknown_signals.hpp`, shared with U.

### Reads (dev only)

- **Before:** main's tool at be3315b. **After:** the changed tool. Both run `pattern_learn --gen-concepts … --part open`, without `--read-test`, so only S15 dev is read.
- **Measured:** pairs with a key; parsed and unparsed pairs; clusters and proposals for C1 (core) and C2 (connector); the dev choice; and for the chosen clustering, the date-relation proposals (property birth_date, death_date, birth_year, death_year or founded) and the others, with their pooled dev precision (holding pairs over gradable pairs, Wilson 95%).
- **The 19 identified date proposals** (S11a's chosen C1, dev):

| # | Key | Property | Pairs | Dev |
|---|---|---|---:|---|
| 1 | `[S] ( <M> # , [O]` | birth_year | 381 | 174 / 177 |
| 2 | `[S] ( born <M> # , [O]` | birth_year | 136 | 52 / 60 |
| 3 | `[S] ( <M> # [O]` | birth_year | 41 | 19 / 20 |
| 4 | `[S] died on <M> # , [O]` | death_year | 37 | 19 / 20 |
| 5 | `[S] ( # <M> [O]` | birth_year | 34 | 16 / 18 |
| 6 | `[S] on <M> # , [O]` | birth_year | 28 | 8 / 16 |
| 7 | `[S] was founded in [O]` | founded | 23 | 5 / 9 |
| 8 | `[S] ( # - <M> # , [O]` | death_year | 21 | 7 / 7 |
| 9 | `[S] ( born <M> # [O]` | birth_year | 17 | 7 / 7 |
| 10 | `[S] was born on <M> # , [O]` | birth_year | 16 | 2 / 4 |
| 11 | `[S] tennō ) ( [O]` | birth_year | 11 | 5 / 5 |
| 12 | `[S] tennō ) ( # - [O]` | death_year | 10 | 3 / 4 |
| 13 | `[O] ) , also known as [S]` | death_year | 9 | 5 / 5 |
| 14 | `[S] ( ca . [O]` | birth_year | 7 | 3 / 3 |
| 15 | `[S] , ( <M> # , [O]` | birth_year | 7 | 3 / 3 |
| 16 | `[O] ) , better known as [S]` | death_year | 6 | 5 / 5 |
| 17 | `[S] ( b . <M> # , [O]` | birth_year | 6 | 4 / 4 |
| 18 | `[S] ) ( <M> # , [O]` | birth_year | 6 | 4 / 4 |
| 19 | `[S] ( # <M> , [O]` | birth_year | 5 | 3 / 3 |

### Decision rule (fixed now)

G11 skips links inside date candidates when all three hold on dev:
1. At least 18 of the 19 identified date proposals are gone: no proposal of the chosen clustering has the same key and property.
2. Every non-date proposal of S11a's chosen clustering (C1) is still proposed: same key, same property.
3. The dev choice stays C1. The change is a reader fix. If it moves the choice, it changes G11 by more than the fix, and it is not adopted here; it is reported instead.

Otherwise G11 keeps its current reader, and this section reports why.

**After the choice (no choice depends on it).** If the skip is adopted, G11 has one test read for this change (`--read-test`, S15 test, logged in S15's Reads table). It restates S11a's G11 test numbers for the changed reader. It never revisits the decision.

### Predictions (registered before any run)

| # | Prediction |
|---|---|
| D1 | The skip removes 15–40% of G11's pairs with a key (54,115 before) |
| D2 | 18 or 19 of the 19 identified date proposals are gone |
| D3 | Every non-date C1 proposal stays, and the dev choice stays C1 |
| D4 | The non-date proposals' pooled dev precision rises by 0.01–0.10: year links inside place clusters ("died in [[1894]]") leave them |
| D5 | (U, see S4b-b) |
| D6 | S3's `--gaps` run and G9's lines are byte-identical |

### Results on dev (2026-10-29)

Before: main's tool at be3315b. After: 24727fc's source with one more line in G11's pair loop, `if (ecs::span_within(c.b, c.e, date_spans(cands))) continue;`. That line was measured, not committed (rule 3 failed). Both with `--part open`, without `--read-test`.

| | Before | After |
|---|---:|---:|
| Pairs with a key | 54,115 | 46,475 (−14.1%) |
| Parsed by a kept pattern | 508 | 290 |
| Unparsed, with a word between | 49,152 | 43,512 |
| C1 (core): clusters, proposals, dev pairs holding | 486, 38, 419 | 414, 20, 78 |
| C2 (connector): clusters, proposals, dev pairs holding | 1,088, 27, 395 | 993, 21, 87 |
| Dev choice (C2 needs 10% more) | C1 (395 vs 419) | **C2 (87 vs 78)** |

- **The 19 date proposals:** all 19 are gone from both clusterings.
- **C1 with the choice pinned** (a report-only build, not committed):
  - every one of the 19 non-date proposals stays, with the same key and property;
  - one is new: "[S] received the [O]" → award_received (2 / 4);
  - "died in" loses its year links: 11 / 19 → 11 / 15 (41 → 28 pairs);
  - pooled dev precision of the non-date proposals: 77 / 121 = 0.636 [0.548, 0.717] → 79 / 120 = 0.658 [0.570, 0.737].
- **C2, now the dev choice:** 21 proposals, none of them dates; pooled dev precision 87 / 142 = 0.613 [0.531, 0.689]. Before the skip, C2 (pinned) had 27 proposals, 8 of them dates holding 322 of 459 pairs.
  - Its proposals reach relations C1 never proposed: developer ("developed by", 3 / 5), headquarters ("based in", "headquartered in"), member_of ("fellow of", 4 / 7), country ("territory of", 3 / 4), spouse ("his wife", "wife was") and death_place ("settled in").
  - They were in C2 before the skip too. **C1 won S11a's dev choice because of the date pairs:** they made 344 of C1's 419 holding pairs.

### Decision (registered rule)

- Rule 1 holds: 19 of 19 date proposals are gone.
- Rule 2 holds on C1: every non-date C1 proposal stays.
- **Rule 3 fails:** the dev choice moves from C1 to C2.
- **So G11 keeps its current reader.** The skip is not in G11's code, and S11a's test numbers stand. No G11 test read was made.
- **For Odin:** the date pairs also decided S11a's choice between C1 and C2. A new registered read, the skip with the choice as it falls (C2), with one S15 test read, would settle G11. That is a larger change than this fix, so it is not made here.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| D1 | The skip removes 15–40% of the pairs | **Miss:** 14.1% |
| D2 | 18 or 19 of the 19 date proposals go | **Hit:** 19 |
| D3 | Every non-date C1 proposal stays, and the choice stays C1 | **Partial:** the proposals stay (C1 pinned); the choice moves to C2 |
| D4 | The non-date proposals' pooled dev precision rises by 0.01–0.10 | **Hit** on C1: 0.636 → 0.658 (+0.022) |
| D6 | S3's `--gaps` run and G9's lines byte-identical | **Hit** (G11 unchanged too, since the skip was not adopted) |

D5 is U's (S4b-b).

### Checks

- `--gen-concepts` (all parts, dev only) with the final tool: stdout (timings masked) and its two pool files byte-identical to main's tool.
- S3's `--gaps` run: gap map `a77678d8ef7dfdef`, its cells and demand files and the v2 store files byte-identical.

## S11d: G11 with the skip, and G9 on a richer store (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · follows S11a and S11a-b above · Odin, 2026-10-29: "Continue. Do all." (HANDOFF §5.19)

(The definitions, reads, rules and predictions below were written and committed before any S11d run. Results are added under them.)

**Question.** Two re-reads that S11a-b left open. (a) G11 with the date-link skip, with the clustering choice as S11a's own rule makes it on dev: what holds on test? (b) G9 on a store that knows more than occupations, with S15's added relations as attributes, against classes it has never been tested on: does it rediscover them?

**Rule 10.** Both generators write hypotheses and relation proposals for Cypha's own store. Nothing here answers a question. G11's relation names (P10) need Odin or a teacher: this read lists the dev property of each proposal as a proposed name, as data only. Nothing enters F2.

### (a) G11 with the skip

- **The change:** S11a-b's skip, behind a new flag `--g11-skip-dates`. A (subject mention, link) pair whose link sits inside one of the sentence's date candidates (`date_spans`, `ecs::span_within`) is not a pair. Everything else is S11a's G11: keys, C1 and C2, the ≥ 5-pair guard, the dev grade, the proposal rule, the test grade.
- **Dev choice:** S11a's rule, unchanged: C2 if it has at least 10% more distinct dev pairs holding their cluster's property, else C1. S11a-b measured 87 against 78 on dev, so the rule is expected to pick C2; whatever it picks is read.
- **Test read (once, logged in S15's Reads table):** each proposal's dev property on its test-subject pairs, as S11a. Reported as S11a: proposals, proposals that hold (≥ 3 gradable test pairs and precision ≥ 0.5), pooled precision with Wilson 95%, split by `pattern_learn`'s relations and S15's added relations.
- **Proposed names:** a TSV `OUT.g11_proposals.tsv` (key, proposed name = dev property, kind, dev and test counts). Data only.
- **Baseline for the rule:** S11a's test read without its 19 date proposals: 19 non-date proposals, 8 hold, pooled 58 / 107 = 0.542.
- **Decision rule (fixed now).** The skip becomes G11's default reader when its one test read gives both:
  1. at least 8 proposals that hold; and
  2. pooled test precision at least 0.542.
  Otherwise the flag stays opt-in and S11a's reader stays the default. If adopted, `--g11-keep-dates` gives S11a's reader, and S11a's test read reproduces with it.

### (b) G9 on a richer store (attributes A3)

- **The store.** Store K (S11a's, unchanged) plus the **added facts**: every row of an S15 file with label 1, an entity value (`value_item` not empty) and a relation `pattern_learn` does not know (`rel_index` < 0). These are 14 relations: award_received, chairperson, director_manager, educated_at, employer, head_of_government, head_of_state, headquarters, member_of, member_of_political_party, member_of_sports_team, owned_by, parent_organization, position_held. The value is `norm_ent(value)`, as for store K. The date relation `dissolved` has no entity value and does not enter.
- **Which files.** Dev runs read the added facts from `temporal_holdout_dev.tsv` only. The one test run (`--read-test`) reads them from both files. That reads S15 test's labels for these 14 relations, once, as G9's input; it is logged in S15's Reads table.
- **Attributes A3:** A1 (store K's direct attributes) plus every added fact as an attribute (r, v), kept by the same support guard (at least 10 subjects). Everything else in G9 is S11a's: closed concepts, candidate classes, class rules, P5's test.
- **Fresh held-out classes.** The A3 attributes with 15 to 500 subjects that are not among S11a's 143 held-out classes (the A1 attributes of store K in that range). Store K is unchanged, so these are exactly the added-relation attributes in the range. A fresh class is **dev** when bit 21 of FNV-1a(r + "\x1f" + v) is 0, else **test**. The class sets are computed on the store of the run (dev: added facts from S15 dev; test run: from both files).
- **For each held-out class H:** every fact with relation r and value v is removed, from store K and from the added facts. G9 runs on the rest with A3. J_new, J_named, J_kept and the null are S11a's, with the null's pool drawn from the context entities holding a value of r in store K or the added facts, seed `mt19937_64(20261103 + i)`.
- **The gate (S11a's, unchanged):** on the fresh test classes, the rediscovery rate's (J_new ≥ 0.5) Wilson lower bound is above the null rate's Wilson upper bound.
- **Strict count, registered now as part of the decision.** A rediscovered class that is also redundant (J_named ≥ 0.5) is one the store already names closely. **Strict rediscovery** = J_new ≥ 0.5 and J_named < 0.5.
- **Secondary (decides nothing):** S11a's 60 test classes, read again on the A3 store of the test run (S11a's sign test and mean J_new − null, amendment 1). And S11a's 83 dev classes on the dev runs.
- **Decision rule (fixed now).** G9 with A3 is S11's concept generator, by measurement under PLAN ledger 55 and open to Odin's veto, when the gate is met **and** the strict rate's Wilson lower bound is above the null's Wilson upper bound. Otherwise it is reported and nothing changes. `--attrs A3` is a new option; A1 stays the default either way, so S11a's runs reproduce.
- **No dev choices.** Nothing is tuned on dev: the dev runs check the code and set expectations.

### (b2) G9's promotion read on A3

- S11a's read 2, unchanged except the store: slots (S15 bit 17 = 0 halves, `pattern_learn`'s non-date relations), hidden store K facts, the dev plan, the loop, `decide`, BH among the supported. The added facts stay in the store; they are other relations.
- If G9 makes hypotheses: the pool files in S8d's fresh-set format (`OUT.fresh_g9.hyps.tsv`, its log), S8d's source stage (`--sources … --fresh POOL EMPTY`, S8d's calibration), and the grade after it (`--pool-grade s15 temporal_holdout_test.tsv 0 …`). Reported against S8's gate (≥ 0.9 on at least 20 graded). Nothing is adopted from it.

### Predictions (registered before any run)

| # | Prediction |
|---|---|
| F1 | G11 with the skip: the dev rule picks C2 |
| F2 | G11 test: 12–25 proposals; 6–14 hold |
| F3 | G11 test: pooled precision 0.45–0.70 |
| F4 | G11 test: at least 2 of developer, headquarters, member_of, spouse, country (as proposed names) hold |
| F5 | G11: the rule adopts the skip as the default reader |
| F6 | G9 A3, dev run: 10–30 fresh dev classes |
| F7 | G9 A3, test: 25–70 fresh test classes; rediscovery rate 15–50% |
| F8 | Most rediscovered fresh classes are redundant: the strict count is under half the rediscovered count |
| F9 | The null rate is at most 5%; the gate is met; the strict part fails, so G9 A3 is not adopted |
| F10 | S11a's 60 test classes on A3: 0–3 rediscovered |
| F11 | Read 2 on A3, test: 1–500 hypotheses; fewer than 20 graded supported after the source stage |

### Dev (2026-10-29)

Run: `/root/work/s11d/runs/dev.out` (commit c6227d2, `--attrs A3 --g11-skip-dates`, no `--read-test`). Nothing here reads S15 test.

- **Added facts (S15 dev):** 7,432 over 920 subjects. The A3 context has 278 attributes over 2,613 entities: 806 → 1,086 closed concepts, 575 → 735 candidate classes. Its 22 kept class rules are all occupations, as on A1.
- **Fresh dev classes: 18.** Rediscovered 11 (0.611 [0.386, 0.797]); **strict 0** (0.000 [0.000, 0.176]); null 0; redundant 12.
  - The 11 are all memberships of countries in international bodies (Interpol, UNESCO, WHO, the World Bank's agencies, …). Each is close to another single membership (J_named 0.58–1.00), and a conjunction of three or four memberships reaches it.
  - The others are not reached: the Rock and Roll Hall of Fame (0.444, by "composer & guitarist & musician & singer & songwriter"), the Bavarian Academy of Sciences (0.478), the National Academy of Sciences (0.429), the Royal Society (0.308), the Hollywood Walk of Fame (0.180), the two US parties (0.17, 0.12).
- **S11a's 83 dev classes on A3:** 0 rediscovered, as on A1 (sign test 69 against 13, mean J_new − null +0.126).
- **Read 2 on A3, dev:** 786 of 2,129 slots have their subject in the context; the 8 kept rules (all occupations) predict **none** of them. No hypotheses.
- **G11 with the skip, dev:** 46,475 pairs with a key; C1 414 clusters, 20 proposals, 78 dev pairs holding; C2 993 clusters, 21 proposals, 87. **The rule picks C2** (87 ≥ 1.1 × 78). Its 21 proposals: educated_at ×3 ("attended", "educated at", "graduated from"), death_place ×2 ("died in", "settled in"), headquarters ×3 ("are in", "based in", "headquartered in"), spouse ×3 ("sister ,", "his wife", "wife was"), country ×3 ("territory of", "area of", "language in"), developer, located_in ("seat of"), birth_place ("born at"), founder, member_of ("fellow of"), author ("novel by"), language.
- **Default output unchanged:** `--gen-concepts` without the new flags (dev) gives stdout (timings masked) and pool files byte-identical to main's tool (wip/integ13, 4331619).

### Results: the test read (2026-10-29)

One run with `--attrs A3 --g11-skip-dates --read-test` at commit 64068c5 (`/root/work/s11d/runs/test.out`; 118 s, 176 MB). Logged in S15's Reads table.

#### G11 with the skip (C2, chosen on dev by S11a's rule)

| Proposals for | Proposals | Hold on test | Pooled test pairs holding | Precision [Wilson 95%] |
|---|---:|---:|---:|---|
| `pattern_learn`'s relations | 14 | 4 | 34 / 68 | 0.500 [0.384, 0.616] |
| S15's added relations | 7 | 2 | 30 / 62 | 0.484 [0.364, 0.606] |
| All | 21 | 6 | 64 / 130 | 0.492 [0.408, 0.577] |

- **Proposals that hold** (proposed name = the dev property; data for P10, nothing enters F2):

| Phrase (C2 connector) | Proposed name | Test |
|---|---|---|
| "[S] … attended [O]" | educated_at | 14 / 23 |
| "[S] … educated at [O]" | educated_at | 8 / 13 |
| "[S] … died in [O]" | death_place | 8 / 13 |
| "[S] … born at [O]" | birth_place | 7 / 10 |
| "[S] … seat of [O]" | located_in | 6 / 8 |
| "[S] … territory of [O]" | country | 3 / 3 |

- **Ones that fail:** developer ("developed by", 3 / 14: it also links techniques to their field); headquarters ("based in" 1 / 8, "are in" 2 / 6, "headquartered in" 1 / 3); founder ("founded by", 0 / 5); educated_at ("graduated from", 3 / 7); spouse ("sister ,", 0 / 3). Seven more have under 3 gradable test pairs, among them member_of ("fellow of", 1 / 2) and spouse ("his wife" 0 / 2, "wife was" 1 / 1).
- **Against S11a's read without its date proposals** (8 hold, 58 / 107 = 0.542): fewer hold (6) at a lower pooled precision (0.492). "died in" now holds (8 / 13; it failed at 7 / 19 with its year links), and country ("territory of") is new. The relations C2 reached on dev (developer, headquarters, spouse, member_of) do not hold on test.
- **The rule:** 6 < 8 and 0.492 < 0.542. **The skip stays opt-in** (`--g11-skip-dates`); S11a's reader stays the default.
- The proposals with their names are in `OUT.g11_proposals.tsv`.

#### G9 on A3: fresh classes (the gate)

The test run's store: store K plus 15,015 added facts over 1,877 subjects (S15 dev and test). The A3 context has 370 attributes over 2,685 entities: 1,995 closed concepts, 1,401 candidate classes, 3,300 candidate class rules of which BH keeps 5 (4 kept classes).

| Fresh test classes (42) | Count | Rate [Wilson 95%] |
|---|---:|---|
| Rediscovered (J_new ≥ 0.5) | 14 | 0.333 [0.210, 0.484] |
| **Strict** (and J_named < 0.5) | **2** | **0.048 [0.013, 0.158]** |
| Null (random classes) | 0 | 0.000 [0.000, 0.084] |
| Redundant (J_named ≥ 0.5) | 14 | |
| Rediscovered by kept classes | 1 | |

- **S11a's gate is met** on the fresh classes: 0.210 > 0.084.
- **The strict part is not:** 0.013 < 0.084. **So G9 with A3 is not adopted** (registered rule). `--attrs A3` stays an option; A1 stays the default.
- **What is rediscovered.** 12 of the 14 are memberships of countries in international bodies (the UN, the ITU, the Universal Postal Union, the World Bank, the African Union, …) and one is "ancient Roman senator". Each has a single named attribute close to it (J_named 0.55–1.00): the store already names the class.
- **The two strict ones:**
  - "foreign member of the Royal Society" (award_received), 0.567, by "member_of = National Academy of Sciences & member_of = Royal Society";
  - "International Civil Defence Organization" (member_of), 0.500, at the bar.
- **Near misses:** the Copley Medal 0.455 (by "Academy of Sciences of Turin & Royal Society"), the Göttingen Academy 0.458, the Royal Swedish Academy 0.487, the Prussian Academy 0.475. Awards such as the Nobel Prize in Literature (0.185) or the Legion of Honour (0.12) get no close class.
- **Secondary (amendment 1's measures):** G9's best class is closer than the null's on 34 fresh test classes and further on 8 (sign test p 3.4e-05); mean J_new − null +0.317 [+0.227, +0.408].
- **By relation:** member_of 12 of 23 (1 strict); award_received 1 of 12 (strict); position_held 1 of 3; educated_at 0 of 3; headquarters 0 of 1.

#### Secondary: S11a's classes on A3

- **S11a's 60 test classes:** 1 rediscovered (0.017 [0.003, 0.089]), strictly: "occupation = producer", 0.500, by "award_received = Star on the Hollywood Walk of Fame & occupation = director …". On A1 it was 0.464. Sign test 52 against 4; mean J_new − null +0.120 [+0.093, +0.148] (A1: +0.113 [+0.086, +0.142]).
- **S11a's 83 dev classes:** 1 rediscovered, not strictly: "occupation = statesperson", 0.632, by "occupation = politician & position_held = President of the United States".
- So the added facts move a few occupations near the bar, but the occupation classes stay out of reach.

#### G9's promotion read on A3

- **Dev:** 906 of 2,129 slots have their subject in the context; 4 kept rules; **0 predictions.**
- **Test:** 867 of 2,124 slots; 3 kept rules; **0 hypotheses.**
- With more candidate rules (3,300), BH keeps fewer (5 on the full store), and every kept rule still predicts an occupation from other occupations. A hidden slot removes all of a subject's occupations, so nothing is predicted. **No pool for S10**; the pool files hold only their header.

### Predictions, scored

| # | Prediction | Result |
|---|---|---|
| F1 | The dev rule picks C2 | **Hit** (87 against 78) |
| F2 | 12–25 proposals; 6–14 hold | **Hit:** 21; 6 |
| F3 | Pooled precision 0.45–0.70 | **Hit:** 0.492 |
| F4 | At least 2 of developer, headquarters, member_of, spouse, country hold | **Miss:** only country |
| F5 | The rule adopts the skip | **Miss:** 6 hold, 0.492 |
| F6 | 10–30 fresh dev classes (dev run) | **Hit:** 18 |
| F7 | 25–70 fresh test classes; rediscovery 15–50% | **Hit:** 42; 0.333 |
| F8 | Strict under half of rediscovered | **Hit:** 2 of 14 |
| F9 | Null ≤ 5%; gate met; strict fails; not adopted | **Hit** |
| F10 | S11a's 60 test classes on A3: 0–3 rediscovered | **Hit:** 1 |
| F11 | Read 2, test: 1–500 hypotheses | **Miss:** 0 |

8 hits, 3 misses.

### Decisions (registered rules)

- **G11:** the skip stays opt-in. With it, S11a's own choice rule moves to C2, and C2's proposals hold no better on test than C1's non-date ones. The six that hold are listed above for naming.
- **G9 with A3:** not adopted. The richer store lets G9 pass S11a's gate, but only on classes the store already names; strictly it rediscovers 2 of 42. G9 still makes no hypotheses for hidden slots.
- Nothing else changes. Without the new flags `--gen-concepts` reproduces S11a's test read byte for byte.

### Checks

- **The test read was made once**, at commit 64068c5, before the run was stopped by an interrupt at 07:18 UTC. It was not repeated; the results above are that run as it stands.
- **The 9 CTests** (HANDOFF §4.4) pass on the branch build. `epistemic_test` runs 1,302 checks, 0 failed (S11a's 1,293 plus 9 for S11d).
- **Four planted bugs**, each in `gen_concepts.hpp`, each fails `epistemic_test`:

| Planted bug | Checks failed |
|---|---:|
| Strict ignores redundancy (drops J_named < 0.5) | 3 of 1,302 |
| Adoption rule strict on precision (`>` for `>=`) | 1 of 1,302 |
| Gate compares lower bounds (null's lower for its upper) | 1 of 1,302 |
| Fresh split on hash bit 19, not 21 | 1 of 1,302 |

- **S3's `--gaps` run** with the branch tool: all 10 files byte-identical to S4's (`/root/work/s4/runs/test`); gap map `a77678d8ef7dfdef`.
- **Flags off:** `--gen-concepts --read-test` without `--attrs A3` and `--g11-skip-dates` gives S11a's test read: stdout identical once timings and paths are masked, and its four pool files byte-identical.
