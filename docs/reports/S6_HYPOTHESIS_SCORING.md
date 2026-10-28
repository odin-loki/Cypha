# S6: scoring hypotheses (plausibility, novelty, impact, compression gain, bridging, priority)

**Odin Loch** · 2026-10-28 · DISCOVERY §6, §13 S6 · follows [S3_S5_GAP_MAP_AND_HYPOTHESES.md](S3_S5_GAP_MAP_AND_HYPOTHESES.md)

(The rules, the planted set and the predictions below were written before any code was run. Results and decisions are added under them.)

**Question.** Cypha will generate many hypotheses and test few. It needs scores that put likely-true ones first (plausibility) and say which are worth testing (novelty, impact, compression gain, bridging, interestingness, priority). Does plausibility rank known-true hypotheses above known-false ones?

**Gate (DISCOVERY §13).** Known-true above known-false, AUROC ≥ 0.75, on a planted set.

**Rule 10.** This is Cypha's own hypothesis machinery. It never writes an answer for CyphaLM, and nothing here reaches `qa` or `chat`. The firewall of S5 (I1) keeps every hypothesis out of answer mode.

## The store

- The v2 fact store's seeds: `/root/work/f2/v2/aug/seeds.tsv` (D4's run; text seeds from infoboxes, categories and tables, and Wikidata values the D4 run mapped to enwik8 titles).
- Distinct (subject, relation, object) triples of a relation in `pattern_learn`'s table: 22,783 triples, 3,960 subjects.
- The extracted facts (`facts_f2.tsv`) are not used: they are fewer and noisier, and the store is what the generators will read.

## The planted set (fixed before scoring)

1. Subjects with at least two triples, shuffled with `mt19937_64(20261028)`. The first 2,000 are used.
2. For each, its triples are shuffled with the same generator; the first one whose object occurs with that relation for another subject is **held out**. That is the known-true hypothesis.
   - A store-based generator can only propose values the store already has, so facts with an unseen object (a spouse's name, an exact date) cannot be planted. How many are skipped this way is reported.
   - After all hold-outs, a pair whose object no longer occurs in the reduced store is dropped.
3. **The reduced store** is the store without the held-out triples. Every score reads only it.
4. **The known-false partner** of each true (s, r, o): a random triple of relation r is drawn from the reduced store, and its object o′ is used. The draw is by triple, so o′ follows the same frequency distribution as true objects: a frequency term cannot win by construction. A draw is rejected when:
   - o′ = o or o′ = s;
   - (s, r, o′) is in the full store;
   - o′ matches a Wikidata surface for (s, r) in `wd_values.tsv` (case-insensitive);
   - for a date or year relation, o′ has the same year as o.
   Up to 100 draws; a pair with no partner is dropped.
   - Some false partners may still be true facts the store and Wikidata lack. That label noise lowers AUROC, and is accepted.
5. **Split by subject:** dev when bit 7 of FNV-1a(subject) is 0, else test. Every choice (link variant, shrinkage m, isotonic fit, z-score pool) is made on dev. Test is read once.

## The scores

Each hypothesis h = (s, r, o) is scored against the reduced store.

| Score | Definition |
|---|---|
| **Base rate** | P(o \| r), add-one over the relation's values |
| **Association** (link predictor, G2-like pairwise rules) | Body = every fact (r₂, o₂) of s. Rule confidence c = (n_both + m·b)/(n_body + m): n_body = other subjects with (r₂, o₂) and some value of r, n_both = those with (r, o) too, b = base rate. Copy rules for a body fact with the same object: c = P(r and r₂ share a value \| a subject has both). Combined as naive Bayes in log-odds from b (`nb`), or as the max (`max`). m ∈ {1, 2, 5} |
| **Path** | Adamic–Adar of s and o in the reduced store's entity graph (undirected; nodes are subjects and objects) |
| **Constraint** | 0: hard violation (a date slot of s already holds another year; death before birth, or a life longer than 125 years: `pattern_learn`'s rule). 0.5: soft (a functional, non-date relation already has another value for s). 1 otherwise |
| **Plausibility P0** | generator reliability × support × constraint (§6.1). One generator here, so reliability is 1. Support = an isotonic fit (dev) of the chosen link score |
| **Novelty** | −log₂ P_store(h), with P_store the association predictor's probability (§6.2) |
| **Compression gain ΔL** | Two-part code of s's facts under the association model: the bits h saves on coding s's other facts, minus the bits to code h itself |
| **Bridging** | Shortest-path distance between s and o in the reduced graph before h is added, minus 1 (capped at 6; no path = 7). A long distance means h would shorten paths a lot |
| **Impact** | Gaps filled: 1 when r is functional and s has no value for r in the reduced store. Contradictions resolved and demand are 0 here (none planted) |
| **Interestingness I** | z(novelty) + z(impact) + z(ΔL) + z(bridging), z against the dev pool |
| **Priority** | EIG(P0; a test with s = c = 0.9) × (1 + impact)(1 + max(0, I)) / cost, cost 1 (the existing `voi_priority`) |

Link variants for the support term: `nb`, `max`, `path`, and `nb` + `path` (the sum of dev z-scores). Chosen on dev by AUROC; on a tie, the simpler.

## Metrics

- AUROC of each score, known-true as positive, with a 95% cluster bootstrap over subjects (2,000 resamples).
- Within-pair accuracy: the share of pairs whose true member scores above its own false one (ties count half).
- By relation group:
  - dates: birth and death year and date, founded;
  - places: birth_place, death_place, located_in, country, capital, location, largest_city;
  - lexical: occupation, genre, language, nationality, party, currency, class, order, family;
  - people and names: the rest.

## Decision rule (fixed before the runs)

1. **The gate** is P0's AUROC on test ≥ 0.75. Its interval is reported. It is "met with confidence" when the lower bound is ≥ 0.75 too.
2. **A truth term stays in the code** when it carries P0's ranking: removing it lowers dev AUROC by ≥ 0.01, with the bootstrap interval of the drop above 0. Otherwise it is cut.
3. **The value terms** (impact, bridging, I, priority) rank what is worth testing, not what is true. The planted set cannot test them. Each stays only if it is small (about 30 lines or fewer), and is marked untested until the scheduler (S10) can measure confirmed value per cost.
4. Test is read once, after every choice above is fixed on dev.

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| P1 | The base rate's dev AUROC is in [0.45, 0.58]: the sampling matches frequencies |
| P2 | The best association variant's dev AUROC is in [0.65, 0.82] |
| P3 | Path (Adamic–Adar) alone is in [0.55, 0.75] on dev, below the association |
| P4 | The constraint alone is in [0.53, 0.65] on dev (it fires on dates and functional relations only) |
| P5 | P0's test AUROC is in [0.72, 0.86], and the gate is met (≥ 0.75) |
| P6 | Novelty's AUROC is ≤ 0.35: it is the link predictor's surprisal, a truth signal reversed |
| P7 | Bridging's AUROC is in [0.25, 0.45]: true facts sit closer in the graph |
| P8 | Impact's AUROC is exactly 0.500: both members of a pair fill the same slot |
| P9 | ΔL's AUROC is in [0.60, 0.82], at or below P0's |
| P10 | I's AUROC is below 0.5 (it rewards novelty) |
| P11 | P0's AUROC is ≥ 0.85 for dates and in [0.60, 0.80] for lexical relations |
| P12 | The run takes ≤ 60 s and ≤ 300 MB on one core |

---

## Results (run 2026-10-28)

### The planted set

- 2,000 subjects were drawn. 108 had no fact whose object occurs for another subject (a store-based generator could not propose them). 1,892 facts were held out.
- 55 pairs were dropped because their object left the store with another hold-out. No pair lacked a false partner.
- **1,837 pairs: 877 dev, 960 test.** The reduced store has 20,891 triples.
- One run takes 16–21 s and 56 MB on one core (P12).

### Dev: the choices

Every variant, P0 = isotonic(support, dev) × constraint (commit 9a8c6a7, `dev2.out`). AUROC with the 95% pair bootstrap.

| Support | Constraint none | hard | full |
|---|---:|---:|---:|
| association nb, m 1 | 0.7529 | 0.7581 | 0.7514 |
| association nb, m 2 | 0.7511 | 0.7561 | 0.7492 |
| association nb, m 5 | 0.7438 | 0.7494 | 0.7438 |
| association max, m 1 / 2 / 5 | 0.719–0.720 (raw) | | 0.720–0.725 |
| path (Adamic–Adar) | 0.5391 | 0.5729 | 0.5723 |
| nb m 1 + path | 0.7591 | **0.7639** [0.744, 0.784] | 0.7572 |

- **The constraint variant "hard" was added after the first dev read.** The soft 0.5 falls on both members of a pair (the slot is the same for both: 50 and 50 pairs on dev), so it carries no signal within a pair and only lowers the pooled AUROC. The hard part does carry signal: on dev, 79 of 190 false dates and 13 of 190 true dates violate it.
- **The best on dev:** association (naive Bayes, m 1) + path, with the hard constraint: 0.7639.
- **Rule 2, each part removed from that configuration:**

| Removed | Dev AUROC drop (95%) | Rule 2 |
|---|---|---|
| The association rules (support = base rate) | 0.156 [0.134, 0.178] | stays |
| The copy rules (see the S6b correction: this line removed the copy rules **and** path) | 0.0105 [0.0047, 0.0163] | stays |
| Path (Adamic–Adar) | 0.0058 [0.0007, 0.0111] | **cut** (under 0.01) |
| The hard constraint | 0.0048 [0.0007, 0.0091] | **cut** (under 0.01) |

- So the final P0 is the isotonic fit of the association predictor (naive Bayes, m 1, with copy rules). Dev: **0.7529** [0.7317, 0.7743].
- **Disclosed:** inside the pruned model the copy rules carry only 0.0051 [0.0019, 0.0084]. Rule 2 was read as one ablation from the chosen configuration, which is how it was written. Read again on the pruned model, it would cut the copy rules too (dev 0.7478).
- Max rules, other shrinkages, Adamic–Adar and the constraint computation were removed from the code (commit 6b05748). The comparison above reproduces at 9a8c6a7.

### Test, read once: the gate

| Score | Dev AUROC | **Test AUROC (95%)** | Test within-pair |
|---|---:|---:|---:|
| **P0** | 0.7529 | **0.7255 [0.7052, 0.7452]** | 0.7438 |
| P0 without copy rules | 0.7478 | 0.7223 | 0.7375 |
| P0 without rules (base rate only) | 0.5934 | 0.5882 | 0.5990 |
| Base rate | 0.5841 | 0.5863 | 0.6026 |
| Rules' shift alone, logit P − logit b | 0.7477 | 0.7219 | 0.6880 |

**The gate is missed.** P0 ranks known-true above known-false at 0.726 on test, and the whole interval is below 0.75. Dev was 0.753. The two intervals overlap (0.732–0.745), so the drop from dev to test may be chance.

- **What carries the ranking:** the pairwise rules. Without them P0 falls to the base rate's level (test 0.588). The copy rules add 0.003 [0.0005, 0.0060] on test.
- **The frequency artefact is small but real.** False objects are drawn by triple, so subjects with many facts (and rarer objects) supply more of them; true objects come one per subject. The base rate alone reaches 0.586 on test. On the 430 test pairs whose false member is at least as frequent as the true one, P0 is 0.587 [0.554, 0.622]. Beyond frequency, the rules' own shift reads 0.722.

### By relation group (P0)

| Group | Dev pairs | Dev AUROC | Test pairs | Test AUROC (95%) |
|---|---:|---:|---:|---|
| Dates | 190 | 0.559 | 214 | 0.550 [0.516, 0.585] |
| Places | 264 | 0.785 | 266 | 0.767 [0.728, 0.805] |
| Lexical | 394 | 0.823 | 441 | 0.788 [0.758, 0.816] |
| People and names | 29 | 0.867 | 39 | 0.835 [0.730, 0.912] |

- Dates sit near chance. A year is not predicted by a subject's other facts. Only the hard constraint knows that a birth year must match a birth date, or come before a death year. On dev it lifted the date group to 0.65, and it was cut for moving the pooled AUROC by under 0.01.

### The value terms (truth AUROC, test; they are not meant to rank truth)

| Term | Test AUROC | What it says |
|---|---:|---|
| Novelty | 0.274 | Reversed truth, as expected: a fact the store predicts is not news |
| Compression gain ΔL | 0.654 | Partly a truth signal: it includes −L(h) |
| Bridging | 0.347 | True facts sit closer in the graph |
| Impact | 0.500 | Both members fill the same slot, by construction |
| Interestingness I | 0.386 | It rewards novelty and bridging |
| Priority | 0.412 | EIG peaks at P0 = 0.5, so priority favours the uncertain, not the true |

These terms rank what is worth testing. The planted set cannot test that; S10's scheduler can (confirmed value per cost). They stay, small and marked untested: impact 4 lines, the z-scores and I 20, graph distance 17, ΔL 12, novelty 1.

### Predictions, scored

| # | Prediction | Result | |
|---|---|---|---|
| P1 | Base rate dev in [0.45, 0.58] | 0.584 | **miss** (the triple-weighted false draw) |
| P2 | Best association dev in [0.65, 0.82] | 0.746 (raw nb m 1) | hit |
| P3 | Path in [0.55, 0.75], below the association | 0.539 | **miss** (below the association: true) |
| P4 | Constraint alone in [0.53, 0.65] | 0.540 | hit |
| P5 | P0 test in [0.72, 0.86], gate met | 0.726; gate missed | **miss** |
| P6 | Novelty ≤ 0.35 | 0.254 dev, 0.274 test | hit |
| P7 | Bridging in [0.25, 0.45] | 0.329, 0.347 | hit |
| P8 | Impact exactly 0.500 | 0.500 | hit |
| P9 | ΔL in [0.60, 0.82], at or below P0 | 0.680, 0.654 | hit |
| P10 | I below 0.5 | 0.377, 0.386 | hit |
| P11 | Dates ≥ 0.85; lexical in [0.60, 0.80] | dates 0.55; lexical 0.823 dev, 0.788 test | **miss** |
| P12 | ≤ 60 s, ≤ 300 MB | 21 s, 56 MB | hit |

8 hits, 4 misses.

## Decisions

(Decisions 2 and 4 are superseded by S6b below: P0 now has the hard constraint and path, kept by a joint test.)

1. **S6's gate is not met.** The scoring machinery is in place (`hypothesis.hpp`), and the plausibility term built from the store alone reaches 0.726 on test.
2. **P0 = the isotonic fit of the store's pairwise-rule predictor**, generator reliability 1 and no constraint term on this planted set. `plausibility_prior` keeps its constraint argument for generators that have hard constraints.
3. **The value terms stay, untested** until S10.
4. **What would move the gate** (each a new change, with the unused subjects as a fresh test set):
   - **The hard date constraint.** It is cheap and known-right, but rule 2's bar of 0.01 per part cut it. A per-part bar penalises several small parts that together are worth more: path, constraint and copy rules together are worth 0.011 on dev. A joint test is the better rule.
   - **Richer link predictors** (G6–G8 in DISCOVERY): typed path rules (country of the birth place, nationality), and entity vectors from Cypha's own word vectors.
   - **A fresh test set:** 2,904 subjects qualify; 2,000 were used. The other ~900 can test one more change.

## For Odin

- The gate missed by 0.024 on test. Should S6's gate be read on a joint test (all small parts together) rather than per part? Under that rule the hard constraint and path would have stayed.
- Should the hard date constraint stay in P0 as a known-right rule, whatever the planted set says? It is DISCOVERY §6.1's constraint term, and it already runs inside the fusion (`pattern_learn`'s lifespan rule).

## How to reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
# dev only (the test half is not printed)
./pattern_learn --score-planted /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv out
# the single test read
./pattern_learn --score-planted /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv final --read-test
# the dev variants (Adamic-Adar, constraint, max rules, shrinkages): build the same at commit 9a8c6a7
cmake -S native -B build && cmake --build build --target epistemic_test -j1 && build/epistemic_test   # 551 checks
```

- `OUT.planted.tsv` holds every scored pair (test rows only with `--read-test`).
- The planted bugs: copy `native/include`, make one of these changes in `hypothesis.hpp`, and rebuild `epistemic_test` against the copy. Each fails at least one check:
  - shrinkage without m;
  - the copy flag ignored;
  - the self count kept;
  - graph distance off by one;
  - naive Bayes replaced by the last rule;
  - ΔL without its own cost;
  - the z-score's sign flipped.
- The normal `pattern_learn` run is unchanged: S3's `--gaps` command gives the v2 store's 7 files and the 3 gap-map files byte for byte.

---

## S6b: the constraint term back, a joint test, and a fresh test set (2026-10-28)

(Registered before any S6b code was run. Decided under PLAN ledger 55: settle by measurement.)

**Why.** DISCOVERY §6.1 defines P0 with a constraint term, so the hard constraint is part of the specification, not an optional extra. S6's bar of 0.01 per part was this report's own registration. Judging parts one at a time is weaker than a joint test: path, the hard constraint and the copy rules each moved dev AUROC by 0.005–0.011, and S6 cut two of them.

### The change

- **P0 = S6's dev choice before pruning:** an isotonic fit (dev) of z(logit P_assoc) + z(ln(1 + Adamic–Adar)), times the hard constraint. P_assoc is naive Bayes, m 1, with copy rules; the z-scores and the fit use S6's dev half. S6's dev read of this P0 was 0.7639 [0.744, 0.784].
- **One joint test on dev:** the set {path, hard constraint, copy rules} stays when removing all three together lowers dev AUROC by ≥ 0.01, with the bootstrap interval of the drop above 0. Otherwise all three go, and P0 is the association alone without copy rules. S6's dev numbers already imply a drop of about 0.016 (0.7639 against 0.7478). This is a check, not a prediction.
- Each part's own drop is still printed, for the record. It decides nothing.

### The fresh test set (fixed before scoring, read once)

- S6's 960 test pairs were read once, for S6's change. They are not read again: their result led to this revision.
- **Qualifying subjects:** subjects with at least two triples, 2,904 in all. S6's planted set used the first 2,000 in its shuffled order. The fresh set is the remaining 904, by the same planted-set rule:
  1. S6's planted set is built first, with the same random draws, so it does not change.
  2. The same generator then continues. For each remaining subject in the shuffled order, its triples are shuffled, and the first one whose object occurs for another subject is held out.
  3. **The fresh reduced store** is the full store without the fresh hold-outs. S6's hold-outs stay in: they are true facts of the store.
  4. False partners follow S6's rule, drawn from the fresh reduced store with the continuing generator.
  5. Every fresh pair is test. Nothing is fitted on it: the isotonic fit and the z-score pools are S6's dev ones.
- **One read** prints, on the fresh set:
  - P0's AUROC with the 95% pair bootstrap, and its within-pair accuracy;
  - the gate (≥ 0.75), hit or miss;
  - the four relation groups;
  - the joint test's drop, out of sample;
  - the value terms.

### Predictions (S6b)

| # | Prediction |
|---|---|
| Q1 | The joint test keeps the set on dev (drop ≈ 0.016 ≥ 0.01, interval above 0) |
| Q2 | The fresh set has 780–880 pairs |
| Q3 | Fresh P0 AUROC is in [0.71, 0.77]. S6 lost 0.027 from dev to test, so about 0.74 is expected: the gate (≥ 0.75) is more likely missed than met |
| Q4 | Dates on the fresh set are in [0.58, 0.72] (S6 dev with the hard constraint: 0.65; without it, test 0.55) |
| Q5 | Places and lexical are each ≥ 0.72 on the fresh set |
| Q6 | Out of sample, the joint set adds +0.005 to +0.025 AUROC over the association alone without copy rules |

### S6b results (run 2026-10-28)

**The fresh set:** 904 subjects. 42 had no fact with a seen object, so 862 were held out; 12 pairs were dropped because their object left the store. **850 pairs.** The fresh reduced store has 21,921 of 22,783 triples. S6's planted set and its dev numbers are unchanged: the dev pairs are identical, and dev P0 is 0.7639.

**The joint test (dev):** removing path, the hard constraint and the copy rules together costs **0.0161 [0.0086, 0.0234]**. The set stays.

**The fresh set, read once:**

| Score | Fresh AUROC (95%) | Within-pair |
|---|---|---:|
| **P0** (association + path, × hard constraint) | **0.7379 [0.7166, 0.7580]** | 0.7659 |
| The three parts removed together | 0.7196 [0.6974, 0.7414] | 0.7100 |
| P0 minus that (the joint test, out of sample) | **+0.0183 [0.0116, 0.0254]** | |
| Without path | 0.7336 (P0 minus it: +0.0043 [0.0007, 0.0081]) | 0.7606 |
| Without the hard constraint | 0.7279 (+0.0100 [0.0050, 0.0152]) | 0.7247 |
| Without the copy rules | 0.7358 (+0.0021 [0.0004, 0.0041]) | 0.7612 |
| Without the rules (base rate) | 0.5613 (+0.1766 [0.1542, 0.1992]) | 0.5835 |
| Base rate | 0.5579 [0.5350, 0.5808] | 0.5906 |

**The gate is missed again: 0.738 against 0.75.** The interval reaches 0.758, so 0.75 is not ruled out, but the point estimate is below it.

| Group | Fresh pairs | P0 AUROC (95%) | Within-pair |
|---|---:|---|---:|
| **Dates** | 202 | **0.658 [0.620, 0.697]** | 0.723 |
| Places | 227 | 0.763 [0.721, 0.805] | 0.793 |
| Lexical | 388 | 0.793 [0.762, 0.823] | 0.771 |
| People and names | 33 | 0.818 [0.714, 0.906] | 0.788 |

- **The hard constraint earns its place on new data.** It adds +0.010 overall, twice its dev share, and lifts dates from 0.550 (S6's test, without it) to 0.658.
- **Each part's own drop holds out of sample:** path +0.004, constraint +0.010, copy rules +0.002. Together they are worth +0.018.
- **The value terms** read as on S6's test: novelty 0.276, ΔL 0.658, bridging 0.354, impact 0.500, I 0.400, priority 0.468.

**A correction to S6.** S6's ablation line "without copy rules" removed the copy rules and path together (it refit the association alone, without the path term). That 0.0105 is the drop for both. The copy rules alone cost 0.0043 [0.0020, 0.0070] on dev, and 0.0021 on the fresh set. The table above is marked.

### S6b predictions, scored

| # | Prediction | Result | |
|---|---|---|---|
| Q1 | The joint test keeps the set on dev | 0.0161 [0.0086, 0.0234]: kept | hit |
| Q2 | 780–880 fresh pairs | 850 | hit |
| Q3 | Fresh P0 in [0.71, 0.77]; the gate more likely missed | 0.738; missed | hit |
| Q4 | Dates in [0.58, 0.72] | 0.658 | hit |
| Q5 | Places and lexical each ≥ 0.72 | 0.763, 0.793 | hit |
| Q6 | The joint set adds +0.005 to +0.025 out of sample | +0.018 | hit |

6 of 6.

### S6b decisions

1. **P0 is S6's dev choice with the constraint term:** an isotonic fit of association + path, times the hard constraint. The joint test keeps all three parts, and the fresh set confirms each one.
2. **S6's gate (0.75) is still missed: 0.726 on S6's test with the pruned P0, 0.738 on the fresh set with this one.** The store's pairwise rules, a path term and the hard constraint reach about 0.74 on hypotheses they have not seen.
3. **What would move it next** is richer evidence, not more tuning of these parts:
   - typed path rules (G7), such as the country of a birth place against a nationality;
   - entity vectors (G6);
   - dates beyond the hard constraint, such as a death year from a career span.
   Both of Cypha's fresh sets are now used, so the next change needs a new held-out source: the 2006-against-today temporal holdout (S15), or Wikidata facts the store lacks.

### How to reproduce S6b

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
./pattern_learn --score-planted /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv out                 # dev, the joint test
./pattern_learn --score-planted /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv fresh --read-fresh  # the fresh set (read once)
```
