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

---

## S6c: G12's closed form as a support term and as a test kind (registered 2026-10-29, before any run)

**Odin Loch** · 2026-10-29 · HANDOFF §5.20 (Odin: "Continue. Do all.") · follows S6b above and [S11B_ABC_DISCOVERY.md](S11B_ABC_DISCOVERY.md)

(The design, the sets, the rules and the predictions below were written and committed before any S6c run. Results are added under them.)

**Question.** S11b removed G12, but found one part that works: given A and C, the strength of their B links in the 2006 text separates a true value from its false partner (pair AUROC 0.682 on S15 test, popularity 0.472). Does that closed form lift S6's P0 (0.738 on S6b's fresh set, against the 0.75 gate)? And does it earn a place as a test in S7's loop?

**Rule 10.** P0 and the loop rank and test Cypha's own hypotheses. Nothing here answers a question, and nothing reaches `qa` or `chat`.

### The closed form, as used here (fixed now)

- **The text** is S11b's: enwik8's 8,587 articles as documents; each article's entities are its title and the targets of its wiki links, redirects resolved (at most three hops). Years, decades, centuries, days of the year, month names and "List of …" titles are left out of every role.
- **Configuration:** S11b's dev choice, not tuned again: PPMI weights, hub cap H = 200, a link B needs at least 2 documents.
- **One change from S11b, so it can score any pair.** S11b read the closed form only on pairs that never share a document. Here, the documents that hold both A and C are left out before counting: N, n_A, n_C, n_B, n_AB and n_BC are all counted on the other documents. score(A, C) = Σ_B ln(N′/n′_B) · PPMI′(A, B) · PPMI′(B, C), over B ∉ {A, C}. On a pair that never shares a document this is S11b's score exactly. So the score never reads a document where A and C appear together: it is indirect evidence only.
- **Resolving a hypothesis (s, r, o):** s is the article with that title; o is the usable entity with the most documents among those whose surface forms (`norm_ent` of the title and of each redirect title) match `norm_ent(o)`, with at least 2 documents. If either does not resolve, or they resolve to the same entity, the closed form is **not applicable** to that hypothesis. Dates and most years never resolve.
- **Code:** `native/include/cypha/eval/abc_closed.hpp` (the counts and the score, restored from S11b's `gen_abc.hpp` at `0230741`, the closed form only) and `pattern_learn`'s `namespace s6c` (building the text and resolving names).

### Part 1: a support term in S6's P0

- **P0_abc** = isotonic(z(logit P_assoc) + z(ln(1 + AA)) + w · z_abc) × hard constraint. That is S6b's P0 with one term added.
  - z_abc = z(ln(1 + score)) for an applicable hypothesis, with the pool from the applicable dev items; **z_abc = 0 when not applicable** (no evidence either way).
  - The isotonic fit and the other z pools are S6b's (S6's dev half).
- **Dev (choice):** S6's dev pairs (877). w from {0, 0.25, 0.5, 0.75, 1, 1.5, 2}, by dev AUROC; within 0.001, the smaller w. w = 0 is S6b's P0.
- **The fresh set (S6c), fixed now and frozen by its hash before the read:**
  1. Subjects with at least two triples, **except the subjects of S6's dev pairs**, in the store's order, shuffled with `mt19937_64(20261031)`.
  2. Each subject's triples are shuffled with the same generator. The first triple whose object occurs for another subject, and that **no earlier set held out** (S6's planted set, S6b's fresh set, S7's set C), is held out.
  3. The reduced store is the full store without these hold-outs. Every earlier hold-out stays in it as a true fact.
  4. False partners by S6's rule (by triple from the relation in the reduced store, with S6's Wikidata filter), drawn with the continuing generator.
  5. Every pair is read once. Nothing is fitted on it.
  - `--score-planted … --s6c-dump` writes the pairs (no scores) before the read; their SHA-256 goes into this report first.
- **S15's pairs:** S8b's read 4 (one S15 pair per subject, its slot removed, z pools and isotonic fits on S15 dev's pairs) with P0_abc beside S6b's P0, w as chosen on S6's dev. S15 test's pairs read once for this change, logged in S15's Reads table. S15 is used for nothing else here.

### Part 2: a test kind in S7's loop (T7, "indirect links")

- **Outcome on (s, r, o):** not applicable → inconclusive; score ≥ τ → positive; score = 0 (no B links o to s at all) → negative; otherwise inconclusive.
  - τ is a quantile of the applicable dev scores, q ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, chosen on S7's dev (sets U and C, S6's labels) by the mutual information between the outcome and the label, S7's choice statistic.
- **It runs only when T4 is inconclusive.** T4 reads s's own article; T7 never reads a document holding both s and o. They read the same 2006 text, so they are not treated as independent of each other. T7 is run only where T4 is silent, and its three-way likelihoods are measured on dev **given T4 inconclusive**. Then P(T4, T7 | h) = P(T4 | h) · P(T7 | T4 inconclusive, h) exactly, with no independence assumed between them.
- **Not independent evidence.** Links in other articles include the infobox links the store was built from. So T7 is never the independent evidence `decide` needs (I4): it can lower or raise a posterior, never support alone.
- **Dependence on T1 and T6:** on dev, within true and within false hypotheses, the odds ratio of T7 positive against each of T1 and T6 positive is printed.
- **The loop read (once), on the S6c fresh set:** T1 (φ_store), T6, T4 with S7b's three-way likelihoods from S7's dev, the prior S7's P0 (S7's dev fit) on the fresh reduced store; with and without T7 last. T5 is left out: S6's Wikidata filter makes it negative on every false partner by construction. Supported precision, refuted precision, decided share, Brier, with `decide` alone and with S7b's batch step.

### Decision rules (fixed now)

1. **P0 term:** adopted if, on the S6c fresh set, P0_abc − P0 has its paired bootstrap 95% interval above 0, no canary falls, and LCI ≥ 0.95 (PLAN §8):
   - P_gain = P(ΔAUROC > ε = 0.005), Bayesian bootstrap over pairs, 4,000 draws;
   - n_eff = fresh pairs (LCI capped at 0.5 below 200);
   - r: canaries are the four relation groups on the fresh set and S15 test's pairs; r = 1 if each has P(Δ > −0.01) ≥ 0.95;
   - γ = min(1, (fresh gain / dev gain) / 0.5), clipped to [0, 1];
   - c = 1 if P0's Brier score on the fresh set does not rise, else 0.5;
   - σ = 1 if the gain has the same sign in three blocks of subjects (FNV-1a mod 3), else 0.5;
   - κ = 1 if the run stays within 15 minutes and 3 GB.
   - "Adopted" means S6's P0 includes the term (`--score-planted … --abc ENWIK8`). Other modes take it at their next change, so their reads stay reproducible.
   - If w = 0 on dev, the term is not adopted and its fresh read is not made.
2. **T7:** adopted if its dev LR+ (given T4 inconclusive) is at least 3, and on the fresh read the loop's supported and refuted precision with T7 are not below those without it, and the Brier score does not rise.
3. **If neither is adopted,** the code goes in its own commit and this section keeps the numbers. If one is, only what it needs stays.
4. Nothing is changed after a read.

### Predictions (registered before any run)

| # | Prediction |
|---|---|
| E1 | The closed form is applicable to 45–70% of S6's dev hypotheses (dates and many lexical values do not resolve) |
| E2 | Alone, on dev pairs where both members are applicable, its pair AUROC is 0.60–0.75 |
| E3 | w on dev is 0.5, 0.75 or 1; P0_abc's dev AUROC is 0.01–0.04 above S6b's P0 (0.7639) |
| E4 | On the fresh set, P0_abc − P0 is +0.01 to +0.04 with the interval above 0, and P0_abc reaches 0.74–0.78 |
| E5 | Dates do not move (not applicable); places gain the most (+0.03 to +0.08) |
| E6 | On S15 test's pairs, P0_abc − P0 is 0.00 to +0.03 and S6's gate stays missed there |
| E7 | T7 is consulted (T4 inconclusive) on 50–85% of S7's dev hypotheses; its LR+ there is 2–4 |
| E8 | Within each label, the odds ratios of T7 against T1 and T6 are between 0.5 and 2 |
| E9 | On the fresh loop read, T7 changes no supported count (it cannot support) and adds refutations, with refuted precision ≥ 0.80 |
| E10 | The closed form adds under 60 s and under 300 MB to a run |

### S6c results (2026-10-29)

**Odin Loch** · 2026-10-29 · code `0d99896` (all new parts behind flags)

**Outcome.** Neither is adopted. On S6's dev pairs the closed form is at chance (0.504), and every weight above 0 lowers P0. So w = 0, and by rule 1 the fresh read is not made. As T7, its LR+ on dev is 1.28, below rule 2's 3. Its "no links" outcome points the wrong way: it is 5.3 times as likely for a true hypothesis as for a false one. By rule 3 the code stays in its own commit, off unless its flags are given. No frozen set was read.

#### Part 1: the P0 term, on S6's dev pairs

- **Coverage.** The closed form applies to 1,151 of 1,754 dev items (65.6%). Both members of a pair apply in 474 of 877 pairs.
- **Alone,** on those 474 pairs: AUROC 0.504 [0.480, 0.529], pair accuracy 0.524.
- **The weight sweep (P0_abc on all 877 dev pairs):**

| w | AUROC | 95% interval | Pair accuracy |
|---|---|---|---|
| 0 (S6b's P0) | 0.7639 | [0.7442, 0.7837] | 0.7930 |
| 0.25 | 0.7447 | [0.7245, 0.7651] | 0.7794 |
| 0.5 | 0.7291 | [0.7101, 0.7487] | 0.7600 |
| 0.75 | 0.7147 | [0.6954, 0.7345] | 0.7577 |
| 1 | 0.6996 | [0.6797, 0.7187] | 0.7509 |
| 1.5 | 0.6849 | [0.6656, 0.7039] | 0.7360 |
| 2 | 0.6719 | [0.6517, 0.6911] | 0.7263 |

- **Choice:** w = 0. Rule 1: "If w = 0 on dev, the term is not adopted and its fresh read is not made." So the S6c fresh set was not read for P0, and S15 test's pairs were not read (that read was tied to the chosen w). S15's Reads table is unchanged.

#### Part 2: T7, on S7's dev (sets U and C)

- **Consulted.** T4 is inconclusive on 2,253 of 2,950 dev hypotheses (76.4%). The closed form applies to 1,601 of them.
- **τ by mutual information** (given T4 inconclusive):

| q | τ | MI (bits) | True: pos / neg / inc | False: pos / neg / inc |
|---|---|---|---|---|
| 0.5 | 1,529 | 0.0600 | 311 / 196 / 377 | 490 / 56 / 823 |
| 0.6 | 2,357 | 0.0621 | 263 / 196 / 425 | 378 / 56 / 935 |
| 0.7 | 3,870 | 0.0619 | 204 / 196 / 484 | 277 / 56 / 1,036 |
| **0.8** | **6,403** | **0.0624** | 145 / 196 / 543 | 176 / 56 / 1,137 |
| 0.9 | 13,066 | 0.0582 | 70 / 196 / 618 | 91 / 56 / 1,222 |

- **Likelihoods at q = 0.8:** true pos 0.165, neg 0.222, inc 0.613; false pos 0.129, neg 0.042, inc 0.829. LR+ 1.28, LR− 5.35, LR(inc) 0.74. **LR+ is below 3, so rule 2's dev part fails.**
- **The negative outcome is reversed.** A score of 0 (no B link at all) comes up for 196 of 884 true hypotheses (22.2%) and 56 of 1,369 false ones (4.1%). The loop uses measured likelihoods, so it reads this outcome as support. But the test was meant to refute with it.
- **Dependence on T1 and T6** (odds ratio of T7 positive, within each label):

| Label | Against | T7 positive when the other is positive | Otherwise | Odds ratio |
|---|---|---|---|---|
| True | T1 | 0 of 9 | 145 of 875 | 0.26 |
| True | T6 | 30 of 144 | 115 of 740 | 1.44 |
| False | T1 | 0 of 3 | 176 of 1,366 | 0.96 |
| False | T6 | 7 of 46 | 169 of 1,323 | 1.29 |

T1 is rarely positive where T4 is silent (12 hypotheses), so the T1 rows say little.

- **The loop on dev, with T7 last** (for reference only: its likelihoods were measured on these same hypotheses). `decide` alone, S7b's kinds:
  - supported 572, precision 0.976 [0.959, 0.985], with and without T7;
  - refuted 113 → 143, precision 0.965 [0.913, 0.986] → 0.937 [0.885, 0.967];
  - decided share 0.232 → 0.242; Brier 0.156 → 0.141.
  - The same holds without T5 (refuted 109 → 138, precision 0.963 → 0.935) and with S7b's batch step (supported 571 either way).
  - So T7 adds about 30 refutations at lower precision.
- **The fresh loop read was not made.** The registration planned it. But rule 2 needs dev LR+ ≥ 3 *and* the fresh read, and the dev part has failed, so the read could decide nothing. I applied rule 1's logic here too, so the fresh set stays unread for a later change. This is the one departure from the registered plan.

#### Why it fails (dev only; likely cause, not measured directly)

The leave-out rule removes every document that holds both s and o. In S6's and S7's sets, a true value was read from s's own article, and that article links to o. So for a true hypothesis, s's own article is left out. That is the document holding most of s's links. For a false partner the article stays. The score therefore drops for true pairs, often to 0. S11b read the closed form only on pairs that never share a document, so its leave-out never fired: S15's values are mostly not in the 2006 text. On facts read from the same text, the closed form as registered does not work.

A follow-up would first count, on dev, how often the leave-out removes s's own article, by label. Then it could register the closed form only for hypotheses whose pair shares no document. But whether a pair applies would then itself depend on the label. That needs its own design.

#### The S6c fresh set (frozen, unread)

2,027 subjects; 1,106 held out (921 skipped: every eligible triple already held out by an earlier set); 1,071 pairs (35 dropped: object gone). Reduced store 21,677 of 22,783 triples. Checksum (FNV-1a over label, subject, relation, object) `44ead1453d05697c`. No score was read on it. A later change can use it as fresh.

#### Predictions scored

| # | Prediction | Result | Score |
|---|---|---|---|
| E1 | Applicable to 45–70% of dev hypotheses | 65.6% | Hit |
| E2 | Alone, AUROC 0.60–0.75 | 0.504 [0.480, 0.529] | Miss |
| E3 | w 0.5–1; dev +0.01 to +0.04 | w = 0; every w > 0 is lower (0.7447 at 0.25) | Miss |
| E4 | Fresh +0.01 to +0.04, interval above 0 | Not read (rule 1) | Not scored |
| E5 | Dates unmoved; places +0.03 to +0.08 | Not read | Not scored |
| E6 | S15 test 0.00 to +0.03; gate missed | Not read | Not scored |
| E7 | Consulted on 50–85%; LR+ 2–4 | 76.4%; LR+ 1.28 | Miss (share in range) |
| E8 | Odds ratios 0.5–2 | 0.26, 1.44, 0.96, 1.29 | Miss (3 of 4; the miss rests on 9 hypotheses) |
| E9 | Fresh loop: no supported change; refuted precision ≥ 0.80 | Not read | Not scored |
| E10 | Under 60 s and 300 MB added | 14.7 s and +57 MB in `--test-kinds` | Hit |

Two hits, four misses, four not scored.

**Cost.** In `--test-kinds`, building the text and scoring 5,746 hypotheses took 14.7 s; peak memory rose from 266 to 323 MB. In `--score-planted`, the S6c block took 177 s and peak memory rose from 74 to 318 MB. That mode reads the 100 MB text (S6 alone does not) and rebuilds three earlier hold-out sets to exclude them; the split was not timed. The machine ran at a load of about 30 on 4 cores.

#### Checks

- `epistemic_test`: 1,325 checks, 0 failed. S6c adds 32 (`test_abc_closed_S6c`).
- The nine CTests of HANDOFF §4.4 pass.
- Planted bugs in `abc_closed.hpp`, each caught: shared documents not left out (7 checks fail); hub cap exclusive (2); counters not reset between calls (2).
- Flags off, the new build against the base build (`4331619`):
  - `--score-planted`: output and `planted.tsv` byte-identical.
  - `--test-kinds … --s7c`: output identical except the timing line; all five TSVs identical.
  - `--gen-links`: output identical except three timing lines; all six TSVs identical.
  - S3's `--gaps` run: `gap_map.tsv`, cells and demand byte-identical (sha256 `a77678d8ef7dfdef`).

#### How to reproduce S6c

```
g++ -O2 -std=c++20 -I native/include -pthread native/tools/pattern_learn.cpp -o pl
F=f2; W=$F/wikidata/wd_values.tsv
./pl --score-planted $F/v2/aug/seeds.tsv $W OUT --abc enwik8              # Part 1, dev (w sweep)
./pl --test-kinds enwik8 $F/v2/aug/seeds.tsv $W $F/v2/aug/patterns.tsv OUT \
     --phi-wd $F/wikidata/phi.tsv --s15 bench/eval/temporal_holdout_dev.tsv --s7c --s6c   # Part 2, dev
```

`--s6c-dump`, `--read-s6c-fresh` and `--gen-links … --abc-w W` exist but were not run (no read was made).
