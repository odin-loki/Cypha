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
