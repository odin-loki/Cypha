# S7: testing hypotheses (T1–T6), pre-registration, and Bayes-factor updating

**Odin Loch** · 2026-10-28 · DISCOVERY §7.1–7.3, §13 S7 · follows [S6_HYPOTHESIS_SCORING.md](S6_HYPOTHESIS_SCORING.md)

(The evaluation set, the tests, the decision rule and the predictions below were written before any test was run. Results and decisions are added under them.)

**Question.** Cypha will test the hypotheses it generates. Each test kind needs a measured sensitivity s (positive when h is true) and specificity c (negative when h is false), so its result can move the posterior by a Bayes factor. How good is each test, and does the loop of §7.2 decide well?

**Gates (DISCOVERY §13).**
- The I5 and I6 CTests.
  - I5: every test is pre-registered before it runs, and its record cannot be edited after the result.
  - I6: every test plan includes a disconfirmation search.
- s and c measured for each test kind.

**Rule 10.** This is Cypha's own hypothesis machinery. T4 searches the corpus to test Cypha's hypotheses. It never answers a question for CyphaLM, and nothing in S7 reaches `qa` or `chat`. The firewall of S5 (I1) keeps every hypothesis out of answer mode.

## The evaluation set (fixed before any test runs)

The store, the Wikidata values and the false-partner rule are S6's (the S6 report, "The planted set"), with one change to the partner rule, below.

**Which subjects.** The brief asks for subjects S6 did not use. S6 and S6b used every subject with two or more facts: 2,000 for S6's planted set and the other 904 for the fresh set. What remains:
- **Set U, subjects S6 never used:** the 1,057 subjects with exactly one fact. Their one fact is held out. These hypotheses have no other fact of their subject in the store, so T1, T2 and T6, which read a subject's other facts, cannot fire on them.
- **Set C, a second fact of S6's subjects:** so that every test can be measured, S7 also holds out, for each subject S6 used, a second fact that S6 never read.
  - S6's own held-out pairs are never read. Its held-out facts stay in the store as true facts.
  - The subjects overlap with S6's. The hypotheses do not.
  - Every number below is printed for U, for C and for both. The U numbers are the ones the brief asked for; C is reported beside them.

**Construction.**
1. S6's planted set and fresh set are rebuilt with S6's code and seed, so S6's hold-outs are known exactly.
2. **U:** each one-fact subject whose fact's object occurs for another subject (so a store-based generator could propose it). That fact is held out.
3. **C:** for each subject S6 held a fact out of, in S6's order, its facts are shuffled with `mt19937_64(20261029)`. The first fact that is not S6's hold-out and whose object occurs for another subject is held out.
4. **The reduced store** is the full store without U's and C's hold-outs. Every test reads it.
5. **False partners:** S6's rule (an object drawn by triple from the relation in the reduced store; rejected when it equals o or s, when (s, r, o′) is in the full store, or for a date or year relation when it has o's year), drawn with `mt19937_64(20261030)`, **without** S6's Wikidata filter. A partner Wikidata lists for (s, r) is flagged `wd_true` instead.
   - The filter would make T5 negative on every false partner by construction.
   - s and c of T1, T2, T4 and T6 are read on pairs whose false member is not `wd_true` (S6's labels). T5's c is read on all pairs, and is then a lower bound: a `wd_true` partner may be a true fact.
6. **Split by subject:** dev when bit 11 of FNV-1a(subject) is 0, else test (S6 used bit 7). Every choice is made on dev. Test is read once, with `--read-test`.

**Eligibility, so no test is checked against its own source.**
- **T5 (Wikidata)** is applied to a pair only when the full store has no Wikidata-sourced fact for (s, r), so its true member did not come from Wikidata. Otherwise T5 is inconclusive by rule, for both members.
- **T4 (corpus)** reads only the prose sentences of s's article. Seeds come from infoboxes, categories, tables and Wikidata, none of them prose. The kept patterns were learned from the seeds, so T4's pattern part is mildly optimistic. This is accepted, and said.

## The tests

Each test gives positive, negative or inconclusive. s = P(positive | true, applicable) and c = P(negative | false, applicable). A result moves the posterior by BF = s / (1 − c) when positive, (1 − s) / c when negative, and 1 when inconclusive. Costs (registered units) set the order of the loop.

| Test | Positive | Negative (disconfirmation, I6) | Inconclusive | Source; cost |
|---|---|---|---|---|
| **T1 consistency** | A constraint could have fired and none did | A date conflict (S6's hard constraint), or a functional relation (φ ≥ 0.9, not a date) of s already holding another value, or o of the wrong type for r (a person, place or country relation against the page-type lexicons) | No constraint applies | store; 1 |
| **T2 consequence** | A consequence rule's prediction is in the store | A consequence's slot holds only other values (the predicted relation is functional, φ ≥ 0.9) | No rule fires, or its slot is empty | store; 2 |
| **T6 compression** | ΔL_others > ε | ΔL_others < −ε | \|ΔL_others\| ≤ ε | store; 3 |
| **T5 cross-source** | Wikidata lists o for (s, r) | Wikidata lists (s, r) values, none o | Wikidata has no (s, r), or the pair is not eligible | Wikidata; 5 |
| **T4 corpus** | A kept pattern extracts o for (s, r), or o shares a sentence with a mention of s (variant chosen on dev) | A kept pattern extracts another value for a functional r | Neither | enwik8 prose; 10 |

- **T2's consequence rules** are learned from the reduced store, with no label: if (s, r, o) holds and o has an edge (r₃, x), forward or reverse, then (s, r₂, x) holds; and if (s, r, o) holds then (o, r₂, s) holds. A rule is kept when it fires for at least 20 subjects of the store with confidence ≥ θ.
- **T6's ΔL_others** is S6's compression gain without the cost of h itself: the bits h saves on coding s's other facts. h's own cost is the association probability, which is already in the prior.
- **T3 held-out prediction** tests rules, not facts. Its planted rules are pairwise rules (r₂, o₂) ⇒ (r, o), split three ways by subject inside dev (and, for the test read, inside test):
  - a generator third proposes rules with support ≥ 5 and confidence ≥ 0.5;
  - T3 hides a second third and predicts it: positive when the Wilson 95% lower bound of the hidden precision is ≥ 0.3, negative when its upper bound is < 0.5, else inconclusive (fewer than 3 hidden cases also inconclusive);
  - the last third is the truth: the rule holds when its precision there is ≥ 0.5 (at least 3 cases).
- **Independence (I4):** T4 and T5 read sources other than the store. T1, T2 and T6 read the store that a store-based generator reads, so their evidence is not independent.

**Choices on dev only:** T2's θ ∈ {0.5, 0.7, 0.9}; T6's ε ∈ {0, 0.5, 1, 2} bits; T4's positive rule (pattern only, or pattern or co-mention). Each is chosen by the mutual information between the test's three-way result and the label on dev, in bits per hypothesis.

## The loop (§7.2), read on test

1. **Pre-registration:** each hypothesis's plan (T1, T2, T6, T5, T4, each with its predicted outcome, what would refute h, and its disconfirmation flag) is written to the log before any of its tests runs.
2. **The prior** is S6's P0 without the constraint: an isotonic fit on dev of z(logit P_assoc) + z(ln(1 + Adamic–Adar)), with the z pools on dev. The constraint is T1's job here, so it is not counted twice.
3. **Cheapest first:** T1, T2, T6, T5, T4. Each result updates the posterior with BF from the dev s and c. The loop stops when `decide` says supported (posterior > 0.9 with independent evidence) or refuted (< 0.05).
4. **The batch:** one generator, one batch. Each hypothesis's e-value is the product of its BFs, p = min(1, 1/e), and Benjamini–Hochberg runs at q = 0.1. A hypothesis is supported only when `decide` says so and BH rejects its null.
5. **Read:**
   - supported precision (the share true) and refuted precision (the share false), with Wilson intervals;
   - how many are supported, refuted, irreducible or still queued;
   - the posterior's Brier score against the prior's.

## Decision rule (fixed before the runs)

1. **The gates:** the I5 and I6 CTests pass, and s and c of every test kind are read on test, with Wilson 95% intervals.
2. **A test kind earns its place** when, on dev (both sets), it is applicable to at least 5% of the hypotheses and Youden's J = s + c − 1 is above 0, with the Wilson-based interval of J above 0. Otherwise it is cut: the code goes and the report keeps its numbers. T3 is judged on its planted rules by the same rule.
3. Every choice above is made on dev. Test is read once, after the choices are fixed.

## Predictions (registered before the runs)

| # | Prediction |
|---|---|
| R1 | Set U has 700–950 pairs; set C has 1,800–2,600 pairs |
| R2 | T5 (Wikidata), where applicable: s in [0.55, 0.90], c in [0.80, 0.99]; applicable to 10–45% of eligible pairs |
| R3 | T4 (corpus): s in [0.25, 0.70] and c in [0.60, 0.95] where applicable; applicable to 30–80% of pairs |
| R4 | T1 is applicable to 5–25% of C's pairs and under 3% of U's; where applicable, c in [0.5, 1.0] and s ≥ 0.9 |
| R5 | T2 is applicable to 3–25% of C's pairs; J in [0.0, 0.6] |
| R6 | T6 has the lowest J of the fact tests that stay; J in [0.0, 0.3] |
| R7 | T3 on planted rules: s in [0.6, 0.95], c in [0.2, 0.7] |
| R8 | On test, the loop's supported precision is ≥ 0.85 and its refuted precision ≥ 0.80 |
| R9 | Fewer than half the test hypotheses reach a decision (supported or refuted) |
| R10 | Every test kind passes the earn-its-place rule except possibly T6 |

---

## Results (run 2026-10-28)

Tool: `pattern_learn --test-kinds` (commit ffa6156, before the test read). Dev runs: 18–24 s, peak 204 MB. The test half was read once.

### The set

| | U (one-fact subjects) | C (a second fact of S6's subjects) | Both |
|---|---:|---:|---:|
| Subjects | 1,056 (253 with no seen object) | 2,687 with an S6 pair (442 with no second fact) | |
| Pairs | 726 | 2,147 | 2,873 |
| Dev / test pairs | | | 1,479 / 1,394 |

- 175 hold-outs were dropped because their object left the reduced store; none lacked a false partner.
- The reduced store has 19,735 of 22,783 triples.
- 11 false partners are ones Wikidata lists (`wd_true`).
- 256 test pairs are eligible for T5: the store took nothing for their (s, r) from Wikidata.

### s and c of each test, read once on test

Both sets, S6's labels; T5 on all pairs. Intervals are Wilson 95%; J's interval combines the two.

| Test | Applicable | s | c | J | MI, bits |
|---|---:|---|---|---|---:|
| **T1 consistency** | 0.106 (C 0.143, U 0) | 0.784 [0.711, 0.842] | 0.892 [0.832, 0.932] | +0.676 [+0.542, +0.775] | 0.039 |
| **T4 corpus** (pattern or co-mention) | 0.242 | 0.991 [0.979, 0.996] | 0.830 [0.750, 0.889] | +0.821 [+0.730, +0.885] | 0.213 |
| T4, pattern only (not chosen) | 0.068 | 0.927 [0.857, 0.964] | 1.000 [0.960, 1.000] | +0.927 | 0.055 |
| **T5 Wikidata** | 0.052 (0.285 of eligible pairs) | 0.767 [0.658, 0.849] | 1.000 [0.950, 1.000] | +0.767 [+0.608, +0.849] | 0.030 |
| **T6 compression** (ε 0) | 0.685 (C 0.920, U 0) | 0.197 [0.174, 0.223] | 0.946 [0.930, 0.959] | +0.144 [+0.103, +0.182] | 0.025 |
| **T3 held-out** (98 planted rules) | 0.337 | 1.000 [0.862, 1.000] | 0.667 [0.354, 0.879] | +0.667 [+0.216, +0.879] | 0.175 |
| T2 consequence | cut on dev: no rule fired | | | | |

By set, on test:
- **U** (the subjects S6 never used): T4 s 1.000 [0.957, 1.000], c 0.714 [0.359, 0.918], applicable 0.130; T5 s 0.467 [0.248, 0.699], c 1.000, applicable 0.042. T1 and T6 never apply: a one-fact subject has nothing to check against.
- **C:** T1 as above (it applies only here); T4 s 0.990, c 0.838, applicable 0.280; T5 s 0.845 [0.731, 0.916], c 1.000; T6 as above.

Dev read the same within its intervals: T1 J +0.673, T4 +0.807, T5 +0.659, T6 +0.205, T3 +0.455.

**Choices on dev (mutual information):** T6's ε = 0 bits; T4's positive rule is "pattern or co-mention" (0.213 bits against 0.044 for patterns alone).

**What the tests miss, and why:**
- **T1** is negative on 32 true facts on test. They are slots where the store itself holds two values: `founded` with several Wikidata inception years (Ecuador 1822), or a coarse category place beside a fine one (Erwin Rommel, birth place Baden-Württemberg). The functionality φ is hand-set, and the store does not obey it.
- **T4** reads silence too. It is inconclusive on 0.597 of true and 0.919 of false hypotheses, a Bayes factor of 0.65 for "nothing found" (dev 0.64). The registered loop counts silence as BF 1, so this evidence is unused.
- **T4's false positives** are mostly real mentions of a related value: Hamburg's page mentions India; Anne Brontë's mentions "British".
- **T5's c of 1.000** is close to true by construction. A false partner can be T5-positive only when Wikidata lists it, and among T5-eligible pairs none was.

### T2 was cut

T2's consequence rules (support ≥ 20, confidence ≥ θ) find **no rule** in the reduced store at any θ. The full store has only 4 at θ 0.5: the store has little two-step structure.
- An exploratory read with support ≥ 5, not registered and deciding nothing, found 10 rules. T2 then applied to 0.2% of dev hypotheses (7 true positives, no false).
- By the decision rule the code was removed. The numbers stay here. T2 should come back when generators (S8) add relational structure.

### The loop (§7.2), read on test

Each plan (T1, T6, T5, T4, cheapest first) was registered before its tests ran: 13,790 log entries, chain verified. BH ran at q = 0.1 over the 2,788 test hypotheses.

| | Registered (dev point estimates of s and c) | Amended (add-one estimates) |
|---|---|---|
| Supported by `decide` | 299 | 299 |
| BH rejects the null | 56 | **0** |
| **Supported** | **56, all true: precision 1.000 [0.936, 1.000]** | 0 |
| **Refuted** | **115, 106 false: precision 0.922 [0.858, 0.958]** | 114, precision 0.921 [0.857, 0.958] |
| Decided | 6.1% of 2,785 | 4.1% |
| Brier, posterior / prior | 0.1736 / 0.2197 | 0.1738 / 0.2197 |

**The amendment, and what it showed.** After the dev read, and before the test read, I added the add-one variant. A test with no error on dev has a point estimate c = 1, so a Wikidata match got BF ≈ 6.6 × 10⁵ from `bayes_factor`'s clamp. With add-one, its BF is 55. Then BH rejects nothing:
- an e-value is the product of the tests' Bayes factors;
- with four tests at most about 13,000, and usually under 60;
- BH over about 2,800 hypotheses at q = 0.1 needs e ≥ 28,000 for the first rejection.

The registered version's 56 pass BH only through the clamped Wikidata factor. Its decisions were right (56 of 56), but that rests on c = 1 measured on 82 dev pairs.

### Predictions, scored

| # | Prediction | Result | |
|---|---|---|---|
| R1 | U 700–950 pairs; C 1,800–2,600 | 726; 2,147 | hit |
| R2 | T5: s 0.55–0.90, c 0.80–0.99, applicable to 10–45% of eligible | 0.767; 1.000; 0.285 | partial (c above the range) |
| R3 | T4: s 0.25–0.70, c 0.60–0.95, applicable 30–80% | 0.991; 0.830; 0.242 | miss (s far higher, applicable lower) |
| R4 | T1 applicable 5–25% of C, under 3% of U; c 0.5–1.0, s ≥ 0.9 | 0.143; 0; 0.892; 0.784 | partial (s below 0.9) |
| R5 | T2 applicable to 3–25% of C, J 0–0.6 | no rule fired | miss |
| R6 | T6 lowest J of the fact tests kept, J 0–0.3 | 0.144, lowest | hit |
| R7 | T3: s 0.6–0.95, c 0.2–0.7 | 1.000; 0.667 | partial (s above the range) |
| R8 | Supported precision ≥ 0.85, refuted ≥ 0.80 | 1.000; 0.922 (registered version) | hit |
| R9 | Fewer than half decided | 6.1% | hit |
| R10 | Every test passes the rule except possibly T6 | T2 cut; T6 kept | miss |

4 hits, 3 partial, 3 misses.

## Decisions

1. **The gates are met.**
   - The I5 and I6 CTests pass: 81 new checks in `native_epistemic_test` (654 in all). Eight planted bugs are caught, among them a log that never detects an edit, a plan accepted without a disconfirmation search, a second result for one test, and an inconclusive result counted as evidence.
   - s and c of T1, T3, T4, T5 and T6 are measured on test, with intervals.
2. **T1, T3, T4, T5 and T6 stay. T2 is cut.** T6 is weak (J 0.14) but its interval is above 0.
3. **I5 lives in two places:**
   - `PreregLog` in `hypothesis.hpp`: append-only and hash-chained;
   - the experiment DB's `hypothesis_preregistrations` and `hypothesis_test_results` tables, whose triggers refuse an edit, a deletion, a result without a plan, and a new plan after a result. These were checked once in SQLite.
4. **The loop decides little, and decides well.** On test it supports 56 (all true) and refutes 115 (92% false). It leaves 94% in the queue: these tests are weak on most single facts, and only T4 and T5 are independent.

## For Odin

- **Batch control as specified is too strict for this evidence.** BH over everything tested, with p = 1/e, supports nothing once s and c are not taken as exactly 1. Possible answers:
  - smaller batches (per relation, per generator run);
  - BH only among the hypotheses `decide` supports (not valid FDR control in general);
  - accept that "supported" needs stronger independent tests than the store gives today.
- **Silence is evidence.** T4 finding nothing has BF 0.65. §7.2 could use each test's three-way likelihoods instead of treating inconclusive as BF 1.
- **Set C overlaps S6's subjects.** S6 used every subject with two or more facts, so the brief's "subjects S6 did not use" leaves only U, where T1 and T6 cannot fire. C's hypotheses are new. S6's own pairs were never read.
- **The store's "functional" slots hold several values** (founding years, coarse and fine places). That costs T1 sensitivity. The hand-set φ should be re-estimated, as `pattern_learn`'s comment already says.

## How to reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
# dev only (the test half is not printed)
./pattern_learn --test-kinds enwik8 /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv /root/work/f2/v2/aug/patterns.tsv out
# the single test read
./pattern_learn --test-kinds enwik8 /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv /root/work/f2/v2/aug/patterns.tsv final --read-test
cmake --build build --target epistemic_test && build/epistemic_test   # 654 checks
```

- Writes `OUT.items.tsv` (each hypothesis's results; `state` is the last loop run: [point] on dev, [add1] on test) and the pre-registration logs `OUT.*.prereg.tsv`.
- The seeds, patterns and Wikidata values are D4's run (`/root/work/f2/v2/aug`; the D4 report says how to rebuild them).
- T2's code was removed before the first code commit (ffa6156), so no commit holds it. Its dev numbers above are from the first dev run.

---

## S7b: three-way likelihoods, batch control and φ (2026-10-29)

**Odin Loch** · 2026-10-29 · S7's three questions for Odin, settled by measurement under PLAN ledger 55

(The sets, the variants, the decision rule and the predictions below were written before any S7b run. Results and decisions are added under them.)

**Question.** S7's loop decided little: 6% of test. Three things held it back.
1. Silence was thrown away: an inconclusive result had BF 1, though T4 finding nothing measured BF 0.65.
2. Batch control as specified (BH over the whole batch on p = 1/e) supports nothing once s and c are smoothed.
3. T1 treats slots as functional by a hand-set φ, but the store holds several values in some of them (founding years; coarse and fine birthplaces). That costs T1 sensitivity.

**Rule 10.** As in S7: this is Cypha's own hypothesis machinery. It never produces an answer for CyphaLM, and nothing here reaches `qa` or `chat`.

### The sets (fixed before any run)

- **Choices: S7's dev set** (sets U and C, S6's labels), as S7 read it. S7's test set is spent and is not read again.
- **The confirm set: S15's dev half** (`bench/eval/temporal_holdout_dev.tsv`, true and false pairs with labels). S15's test half is not read; it is kept for the generators' final reads.
  - Only pairs whose relation is one of `pattern_learn`'s relations (`kRel`) and whose subject has an enwik8 page.
  - **Split by subject, fixed now:** the choose half when bit 17 of FNV-1a(subject) is 0, else the confirm half. The choose half is read freely, only to check that the adapter works; no choice is made on it. The confirm half is read once, with `--read-confirm`.
  - **The S15-reduced store:** the full store (S7's seeds) without every true fact of S15's dev half that it holds (matched by normalised key, or by year for year relations and by a compatible date for date relations). Both halves' facts are removed, so no test reads the fact it judges.
  - **The object's form:** an S15 value is replaced by the store's most frequent surface with the same key for that relation, when there is one, so the store's statistics see it.
  - **The primary confirm set:** pairs where both members' objects occur as objects of the relation in the S15-reduced store (what a store-based generator could propose). All pairs are read beside it as a secondary set.
  - **T5 is left out on S15.** S15's labels are current Wikidata, and T5 is current Wikidata (D4's values), so T5 would read the label. The loop on S15 runs T1, T6 and T4; only T4 is independent evidence there.
  - The prior is S7's (the isotonic fit and z pools from S7's dev), applied to features computed on the S15-reduced store.

### The variants (registered alternatives)

**φ for T1.** A relation is functional when φ ≥ 0.9.
- **φ_hand:** `kRel`'s hand-set values (S7 as run).
- **φ_wd:** D4's Wikidata estimate (`/root/work/f2/wikidata/phi.tsv`: the share of subjects with exactly one value); `kRel`'s value where D4 has none.
- **φ_store:** estimated from the store the tests read. Among subjects with at least one value of the relation, the share with exactly one distinct value (distinct years for date relations, distinct normalised keys otherwise).

φ gates two of T1's negatives: the functional conflict (non-date relations, as in S7) and S6's same-kind date conflict (a date slot of s already holding another year: birth, death, founded). The lifespan checks (death before birth, a life over 125 years) and the type checks stay unconditional. With φ_hand, T1 is exactly S7's T1 (every date relation has φ_hand ≥ 0.9).

**Likelihoods.**
- **Two-way (S7):** add-one s and c; positive BF s / (1 − c), negative (1 − s) / c, inconclusive 1.
- **Three-way:** P(outcome | true) and P(outcome | false) for the three outcomes, add-one per row (a Dirichlet(1, 1, 1) mean). BF = P(outcome | true) / P(outcome | false) for every outcome, inconclusive included.
- In both, an inconclusive result is never independent evidence (I4 unchanged): silence can lower or raise a posterior, but it can never be the independent evidence `decide` needs for "supported".

**Batch control.**
- **B0:** BH at q 0.1 over the whole batch on p = min(1, 1/e) (S7). This is exactly e-BH (Wang and Ramdas 2022: reject the k largest e-values where e₍ₖ₎ ≥ m / (q k)), so e-BH is not a separate candidate.
- **B1:** BH at q 0.1 only among the hypotheses `decide` supports. Selective; no FDR guarantee in general.
- **B2:** BH at q 0.1 within each relation's batch (smaller batches).
- **Reference only, not a candidate:** no batch step (`decide` alone).

### Decision rule (fixed before the runs)

1. **φ:** the variant with the highest mutual information between T1's three-way result and the label on S7's dev (S7's choice statistic). Ties go to φ_hand.
2. **Likelihoods × batch control** (2 × 3), on S7's dev with the chosen T1: a configuration qualifies when its supported precision is ≥ 0.90 on at least 10 supported, and its refuted precision is ≥ 0.80. The qualifying configuration that decides the largest share (supported + refuted) wins. Ties (within 0.001) go to two-way before three-way, then B0, B2, B1. If none qualifies, S7's configuration stays.
3. **Confirm:** S15's confirm half, read once, with the chosen configuration and S7's (two-way, B0) beside it. The choice is confirmed when its supported precision on the primary set is ≥ 0.90; its Wilson 95% lower bound is reported with it.

### Predictions (registered before the runs)

| # | Prediction |
|---|---|
| Q1 | φ_store is below 0.9 for birth_place, death_place and founded, and at or above 0.9 for capital, birth_year and death_year |
| Q2 | φ_store is chosen. T1's s on dev rises from 0.755 to ≥ 0.85 and its c falls by at most 0.05. φ_wd moves T1's c more than its s (it makes country and author functional) |
| Q3 | Three-way inconclusive BFs on dev: T4 in [0.55, 0.75], T5 in [0.9, 1.1], T1 in [0.95, 1.3] |
| Q4 | Three-way decides a larger share of dev than two-way under every batch control, by at least 1 point |
| Q5 | Supported counts on dev: B1 ≥ B2 ≥ B0; two-way B0 supports nothing (as in S7) |
| Q6 | The chosen configuration is three-way with B2 or B1, with dev supported precision ≥ 0.95 |
| Q7 | On S15's confirm half (primary set): supported precision ≥ 0.90; under 20% decided; T4 applicable to a smaller share than on S7's dev (S15's b and c facts are not in the 2006 text) |
| Q8 | At least 80% of the confirm half's supported hypotheses are category a (stated in the 2006 text) |

**Amendment (2026-10-29, after the choose-half read, before the confirm read).** The registration re-estimates φ_store on the store the tests read. On S15 that is the S15-reduced store, which has lost 9,393 of the store's 22,783 facts to the hold-out. Multi-valued slots then look functional: country's φ goes from 0.861 to 0.915, capital's from 0.881 to 0.908, director's from 0.877 to 0.909. On the choose half this alone made T1 negative on 46 true and 46 false `country` hypotheses, which carries no information. The hold-out was meant to keep each judged fact away from its tests, not to change the store's structure. So on S15 the loop uses **the φ table chosen on dev, as is**. The registered re-estimate is printed beside it on the confirm read, and the confirmation is judged on the amended form.

### Results (run 2026-10-29)

Tool: `pattern_learn --test-kinds … --phi-wd phi.tsv --s15 temporal_holdout_dev.tsv` (commit ca1ed8b, committed before the confirm read). A run takes 39–58 s at a 266–272 MB peak. S7's own lines and dev items are unchanged byte for byte, and with φ_hand T1 equals S7's T1 on all 5,746 hypotheses. The confirm half was read once (`--read-confirm`).

#### φ for T1 (dev, both sets)

| φ | Applicable | s | c | J | MI, bits |
|---|---:|---|---|---|---:|
| hand (S7) | 0.100 | 0.755 [0.680, 0.818] | 0.918 [0.863, 0.953] | +0.673 | 0.0374 |
| Wikidata (D4) | 0.131 | 0.575 [0.505, 0.643] | 0.938 [0.894, 0.964] | +0.513 | 0.0318 |
| **store** | 0.092 | **0.822 [0.749, 0.878]** | 0.911 [0.851, 0.948] | **+0.733** | **0.0403** |

- **φ_store is chosen.** It is functional (≥ 0.9) where the store really holds one value.
  - Unlike the hand-set table, it makes these not functional: birth_place (0.769: a coarse and a fine place), capital (0.881) and currency (0.808).
  - It makes these functional: author (0.939), location (1.000) and owner (1.000).
  - Date slots stay functional: birth_year 0.919, death_year 0.950, founded 0.938, birth_date 0.974, death_date 0.997.
- **Wikidata's φ is worse than the hand-set one.** Wikidata counts one value per subject for country (0.920), author, director and composer, but the store holds several for them, so T1 starts negating true facts (s 0.755 → 0.575).
- **Only T1 uses this φ.** Fusion's hand-set φ (`kRel`, which builds the store) is unchanged; replacing it changes the store and needs its own read.

#### Three-way likelihoods (dev, with the chosen T1)

| Test | P(+ / − / ? \| true) | P(+ / − / ? \| false) | BF + | BF − | BF ? |
|---|---|---|---:|---:|---:|
| T1 consistency | 0.076 / 0.017 / 0.908 | 0.009 / 0.084 / 0.907 | 8.57 | 0.200 | 1.001 |
| T6 compression | 0.180 / 0.524 / 0.296 | 0.035 / 0.636 / 0.329 | 5.21 | 0.824 | 0.898 |
| T5 Wikidata | 0.037 / 0.020 / 0.943 | 0.001 / 0.056 / 0.943 | 55.0 | 0.349 | 1.000 |
| T4 corpus | 0.400 / 0.003 / 0.597 | 0.014 / 0.057 / 0.929 | 29.4 | 0.059 | **0.643** |

- Silence is evidence for T4: "nothing found" has BF 0.64, as S7 measured.
- The larger gain is elsewhere. S7's two-way factors condition on the test applying, so T4's positive factor was s / (1 − c) = 5.2. But false hypotheses rarely draw any T4 result at all: P(positive | false) is 0.014 over all hypotheses. The three-way positive factor is 29.4.

#### The loop on dev (2,950 hypotheses, one batch)

| Likelihoods | Batch step | Supported (precision, Wilson 95%) | Refuted (precision) | Decided | Qualifies |
|---|---|---|---|---:|---|
| two-way | B0 whole batch | 0 | 103 (0.961) | 0.035 | no |
| two-way | B2 per relation | 111 (0.991 [0.951, 0.998]) | 103 (0.961) | 0.073 | yes |
| two-way | B1 among supported | 237 (0.992 [0.970, 0.998]) | 103 (0.961) | 0.115 | yes |
| three-way | B0 whole batch | 215 (0.995 [0.974, 0.999]) | 113 (0.965) | 0.111 | yes |
| three-way | B2 per relation | 160 (0.988 [0.956, 0.997]) | 113 (0.965) | 0.093 | yes |
| **three-way** | **B1 among supported** | **572 (0.974 [0.957, 0.984])** | **113 (0.965 [0.913, 0.986])** | **0.232** | **yes: chosen** |
| two-way | none (reference) | 342 (0.991) | 103 | 0.151 | — |
| three-way | none (reference) | 573 (0.974) | 113 | 0.233 | — |

- **Chosen by the rule: three-way likelihoods with BH among the supported.** It decides 23.2% of dev, against S7's 3.5% with nothing supported.
- **The batch step does almost nothing here.** Among the hypotheses `decide` supports it keeps 575 of 576. The precision comes from `decide` itself: a posterior above 0.9 from calibrated likelihoods, plus independent evidence. The whole-batch e-BH (three-way B0) is the valid FDR procedure, and supports 215 at 0.995.

#### The confirm read: S15's confirm half, read once

11,590 S15 dev pairs qualify (pattern_learn's relations, enwik8 subjects): 5,754 in the choose half and 5,836 in the confirm half, of which 2,680 are store-reachable. The S15-reduced store holds 13,390 of the store's 22,783 facts (9,393 held out). T5 is left out.

| Set | Configuration | Supported (precision, Wilson 95%) | Refuted (precision) | Decided |
|---|---|---|---|---:|
| **Reachable (primary)** | **three-way, among supported, amended φ** | **977 (0.956 [0.941, 0.967])** | **138 (0.964 [0.918, 0.984])** | **0.208** |
| Reachable | the same, registered φ | 975 (0.956 [0.941, 0.967]) | 138 (0.964) | 0.208 |
| Reachable | S7's (two-way, whole batch) | 0 | 141 (0.957) | 0.026 |
| All pairs | three-way, among supported, amended φ | 2,280 (0.969 [0.961, 0.976]) | 649 (0.928 [0.905, 0.945]) | 0.251 |
| All pairs | S7's (two-way, whole batch) | 0 | 648 (0.929) | 0.056 |

- **The choice is confirmed:** supported precision 0.956, Wilson lower bound 0.941, on a fresh labelled set.
- The posterior's Brier score falls from the prior's 0.238 to 0.198 (primary set).
- The φ amendment changes little in the loop: 977 against 975 supported.
- On S15 the tests behave as on S7's dev, where they apply:
  - T4: s 0.994, c 0.760, applicable 0.209;
  - T6: J +0.16;
  - T1 applies to under 1% of S15's hypotheses (few S15 subjects hold a conflicting value), J +0.29 [−0.36, +0.72].
- **What is supported, by S15 category (primary):** a 432 (stated in the 2006 text), u 538 (no date), b 5, c 2. The loop finds what the 2006 text says. It supports almost nothing true only after 2006, as it should: no test here reads later evidence (T9 is the generators' final read).

#### Predictions, scored

| # | Prediction | Result | |
|---|---|---|---|
| Q1 | φ_store < 0.9 for birth_place, death_place, founded; ≥ 0.9 for capital, birth_year, death_year | birth_place 0.769, death_place 0.951, founded 0.938; capital 0.881, birth_year 0.919, death_year 0.950 | miss (3 of 6) |
| Q2 | φ_store chosen; T1 s 0.755 → ≥ 0.85, c falls ≤ 0.05; φ_wd moves c more than s | chosen; s 0.822; c −0.007; φ_wd moves s (−0.180) more than c (+0.020) | partial |
| Q3 | Inconclusive BFs: T4 0.55–0.75, T5 0.9–1.1, T1 0.95–1.3 | 0.643, 1.000, 1.001 | hit |
| Q4 | Three-way decides ≥ 1 point more under every batch step | +7.6, +2.0, +11.7 points | hit |
| Q5 | Supported B1 ≥ B2 ≥ B0; two-way B0 none | two-way 237 ≥ 111 ≥ 0; three-way B2 160 < B0 215 | partial |
| Q6 | Three-way with B2 or B1, dev precision ≥ 0.95 | three-way B1, 0.974 | hit |
| Q7 | Confirm: precision ≥ 0.90; under 20% decided; T4 applicable less than on dev | 0.956; 20.8%; 0.209 < 0.236 | partial (decided 0.8 points over) |
| Q8 | ≥ 80% of the confirm's supported are category a | 44% (u 55%) | miss |

3 hits, 3 partial, 2 misses. Q8 misread S15's categories: "u" means undated and changeable (occupation, genre, country), not "unstated", and the 2006 text states many of them.

### Decisions (S7b)

1. **T1 uses the store's own φ** (φ_store, the table estimated on the store; on S15 the dev table as is). Fusion's φ in `kRel` is unchanged.
2. **The loop uses three-way likelihoods** (`HypTest::likelihoods`, `likelihoods_from`). An inconclusive result moves the posterior by its measured ratio and is still never independent evidence (I4).
3. **The batch step is BH among the supported** (`bh_reject_e_among`), as the rule chose. It is selective, with no FDR guarantee over the batch. In practice the precision comes from `decide`'s 0.9 threshold on calibrated posteriors, and it held on fresh data (0.956).
4. **Confirmed on S15's confirm half.** The loop decides about 21–25% of fresh hypotheses where S7's decided 3–6% and supported none.
5. **I5 holds:** the plan registers each test's likelihoods; an edit of one breaks the log's chain.
6. 43 new checks in `native_epistemic_test` (697 in all). Four planted bugs are caught (5, 3, 2 and 3 failing checks): silence counted as BF 1, likelihoods left out of the plan, the batch step ignoring the selection, and silence counted as independent evidence.

### For Odin (S7b)

- **Batch control.** The chosen step (BH among the supported) has no FDR guarantee and here changes almost nothing. The valid whole-batch e-BH with three-way likelihoods supports 215 of dev at 0.995 (11% decided against 23%). If a guarantee matters more than reach, switch to it; the code has both.
- **Fusion's φ.** The store's φ differs from the hand-set table in six relations (birth_place, capital, currency, author, location, owner). Using it in fusion (`--phi FILE` exists) would change the store, and needs its own read.

### How to reproduce (S7b)

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
# dev choices and S15's choose half
./pattern_learn --test-kinds enwik8 /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv /root/work/f2/v2/aug/patterns.tsv out \
  --phi-wd /root/work/f2/wikidata/phi.tsv --s15 bench/eval/temporal_holdout_dev.tsv
# the single confirm read
./pattern_learn … final --phi-wd /root/work/f2/wikidata/phi.tsv --s15 bench/eval/temporal_holdout_dev.tsv --read-confirm
cmake --build build --target epistemic_test && build/epistemic_test   # 697 checks
```

The S7b lines start with `S7B`. `OUT.s15_items.tsv` lists the S15 hypotheses of the halves read with their T1, T6 and T4 results and prior. `OUT.s7b_dev.prereg.tsv` and `OUT.s15_confirm.prereg.tsv` are the chosen configuration's pre-registration logs.

## S7c: a batch rule with a guarantee (2026-10-29)

**Odin Loch** · 2026-10-29 · in answer to Odin's ruling on S7b's batch step: "Hybrid or something better. Keep working on it. Its not up to scratch yet."

(The candidates, the sets, the decision rule and the predictions below were written before any S7c run. Results and decisions are added under them.)

**Question.** S7b's batch step (BH among the supported) decides 23% of dev at precision 0.974 but guarantees nothing: it kept 575 of 576. The guaranteed whole-batch e-BH decides 11% at 0.995. Is there a rule with a stated false-discovery guarantee that decides more than 11%, at precision at least S7b's?

**Rule 10.** As in S7 and S7b: Cypha's own hypothesis machinery. It never produces an answer for CyphaLM.

### Common ground

- The loop is S7b's, unchanged: three-way likelihoods from S7's dev, T1 under the store's φ, cheapest first (T1, T6, T5, T4; T5 left out on S15), `decide` (posterior > 0.9 with independent evidence; refuted below 0.05). The batch rule only decides which of `decide`'s supported hypotheses are promoted. Refutations are not batch-controlled (as in S7b).
- q = 0.1 throughout. FDR is the expected share of false hypotheses among those promoted. Counts leave out false members that Wikidata lists (S6's labels), as before.
- **Decided** = (promoted + refuted) / hypotheses, as S7b counted it.

### The candidates (each states what it guarantees)

| # | Rule | Promotes | Guarantee, and what it rests on |
|---|---|---|---|
| K1 | e-BH over the whole batch on the loop's e-values (S7b's B0, three-way) | `decide`'s supported whose null e-BH rejects | FDR ≤ q under any dependence between hypotheses, **if** each e-value is valid: the null likelihoods are right and the tests are independent given the truth |
| K2 | Prior-weighted e-BH: weights w ∝ prior odds p0 / (1 − p0), scaled to mean 1 over the batch; reject the k largest w·e where w₍ₖ₎e₍ₖ₎ ≥ m / (q k) | as K1 | FDR ≤ q · (Σ null weights) / m ≤ q under any dependence, **if** the e-values are valid given the weights (the prior is fixed before any test runs) |
| K3 | **Conformal selection** (Jin and Candès 2023): score = the loop's posterior when `decide` supports, else −1; conformal p = (1 + #{calibration hypotheses that are false and score ≥ s}) / (n + 1), n the whole calibration set; BH at q | the selected (all inside `decide`'s supported, by the score) | **Finite-sample FDR ≤ q when the calibration hypotheses and the batch are exchangeable** (drawn the same way, with labels for the calibration set). Needs no model: not the likelihoods, not calibration, not independence between tests |
| K4 | BH on null-model p-values: p = P(e ≥ e_obs \| false) under the three-way null likelihoods, by enumerating the loop's outcome paths (its stopping rule included) | as K1 | FDR ≤ q **if** the null likelihoods are right and the p-values are PRDS (Benjamini–Yekutieli would cover any dependence) |
| ref | S7b's B1, BH among the supported | | none (reference) |

**The hybrid (Odin's word).** The default becomes two tiers: the winning rule's promotions are **Known** (with its guarantee); `decide`'s other supported hypotheses are **Believed** only, never Known (DISCOVERY §2.1's states). The Believed tier's precision is reported; its only claim is Bayesian (if the posteriors are calibrated, its expected false share is under 0.1, since each posterior is above 0.9).

**Considered, not run:** online e-LOND / e-LORD (the loop decides in batches; a stream rule is weaker on a batch); per-relation batches with α spending (S7b's per-relation BH lost to the whole batch); e-closed testing (its gain over e-BH is small and its closure is costly); conditional-calibration boosting of e-BH (it needs each e-value's null distribution given a sufficient statistic, which this model does not give; K4's enumeration is the nearest form).

### The sets (fixed before any run)

- **S7's dev** (2,950 hypotheses), as S7b read it. K1, K2 and K4 run on the whole dev batch. K3 needs a calibration set: dev is split by bit 13 of FNV-1a(subject) into two folds; each fold is calibrated on the other, and the two selections are counted together. The score's likelihoods and prior were fitted on all of dev, so on dev K3's guarantee is not exact (said, not hidden).
- **S7b's choose half of S15 dev** (read freely by S7b; no choice was made on it): reachable pairs primary, all pairs beside them, T5 left out, the amended φ. K3 folds by bit 13 inside the half; here the score was fitted on S7's dev, not on these hypotheses.
- **The confirm set, C2, never read:** a third fact of S6's subjects. For each subject with an S6 hold-out, in S6's order, its facts are shuffled exactly as S7's set C was (`mt19937_64(20261029)`, the same calls), and the **second** fact that is neither S6's hold-out nor C's, and whose object occurs for another subject, is held out. Nobody has judged these facts: S6 and S7 left them in the store as true facts.
  - The store the tests read is S7's reduced store without C2's facts too.
  - False partners: S7's rule, drawn with `mt19937_64(20261101)`.
  - The tests, the φ table, T6's ε, T4's variant, the prior's fit and the likelihoods are S7b's from dev, applied as is. T5 runs (C2's truth is the store, not Wikidata), with S7's eligibility rule.
  - K3 calibrates C2's two bit-13 folds on each other (the score was fitted on S7's dev, so this is exact). K1, K2 and K4 run on C2 as one batch.
  - C2 is read once, with `--read-c2`, after the choice.

### Checking the guarantees (dev only)

- **Model simulation** (400 batches): each hypothesis gets a prior drawn from dev's priors, a truth drawn with that probability, and outcomes drawn from the three-way likelihoods; the loop runs on it. Two scenarios: **mixed** (truths as drawn) and **planted nulls** (every hypothesis false). Batch and calibration sets each have dev's size and are drawn the same way. The realised FDR is the mean false share among promotions (0 when nothing is promoted), with its Monte Carlo standard error.
- **Real-data resampling** (400 splits): dev's subjects are split at random into a calibration half and a batch half; the rules run on the batch half with the real outcomes. The same on S15's choose half. This checks the guarantees where the model is wrong (dependent tests, smoothed likelihoods).

### Decision rule (fixed before the runs)

1. A candidate **qualifies** when all hold:
   - on S7's dev: at least 10 promoted, and the Wilson 95% lower bound of their precision ≥ 0.90;
   - on S15's choose half (reachable): promoted precision ≥ 0.90 when at least 10 are promoted;
   - realised FDR ≤ 0.10 + 2 standard errors in both simulation scenarios and in both resamplings.
2. Among the qualifiers, the one with the **largest Known-decided share on S7's dev** wins (Known promoted + refuted, over hypotheses). Ties within 0.5 points go to the simpler rule: K1, then K2, K4, K3.
3. The default becomes the hybrid with the winner as its Known tier. If nothing qualifies, S7b's step stays and is relabelled Believed-only (no Known tier).
4. **Confirm** (C2, read once): confirmed when the Known tier's precision has a Wilson 95% lower bound ≥ 0.90. Its Known-decided share against S7b's whole-batch e-BH (11%) and its precision against S7b's (0.974 dev, 0.956 S15 confirm) are reported with it.

### Predictions (registered before the runs)

| # | Prediction |
|---|---|
| V1 | K3 promotes at least 1.5 times K1's 215 on S7's dev, at precision ≥ 0.95; its Known-decided share is ≥ 15% |
| V2 | K2 promotes more than K1 (at least 1.2 times) and fewer than K3 |
| V3 | K4 promotes more than K1 |
| V4 | Model simulation: K1, K2, K3 and K4 all have realised FDR ≤ 0.10 in both scenarios; S7b's B1 exceeds 0.5 under planted nulls |
| V5 | Resampling: K1–K4 have realised FDR ≤ 0.10 on S7's dev and on S15's choose half |
| V6 | The mean e-value of false dev hypotheses is ≤ 1.2 in every prior quartile (K2's condition holds roughly) |
| V7 | The chosen rule is K3 |
| V8 | C2 has 800–2,500 pairs |
| V9 | On C2 the Known tier's precision is ≥ 0.93, Wilson lower bound ≥ 0.90, and its Known-decided share is > 11% |

**Amendment (2026-10-29, before any S7c run).** As registered, K1, K2 and K4 promote `decide`'s supported hypotheses whose null is rejected (R ∩ S). The guarantee covers the rejected set R, and a subset of R need not keep it. So K1, K2 and K4 run on **gated** evidence: e = 0 (K1, K2) or p = 1 (K4) for a hypothesis `decide` does not support. A gated e-value is never larger than the e-value, so it stays valid, and the rejected set lies inside the supported set: the guarantee covers exactly what is promoted. K3's score is gated the same way already. The registered R ∩ S form is printed beside each.

### Results (2026-10-29)

**S7's dev** (2,950 hypotheses; `decide` supports 573 and refutes 113, at 0.965). The R ∩ S forms of K1, K2 and K4 gave the same sets as the gated forms, on every read.

| Rule | Promoted (Known) | Precision [Wilson 95%] | Known-decided | Believed (precision) |
|---|---|---|---|---|
| K1 e-BH | 215 | 0.995 [0.974, 0.999] | 11.1% | 358 (0.961) |
| K2 prior-weighted e-BH | 37 | 1.000 [0.906, 1.000] | 5.1% | 536 (0.972) |
| K3 conformal selection | 573 | 0.974 [0.957, 0.984] | 23.3% | 0 |
| K4 null-model BH | 573 | 0.974 [0.957, 0.984] | 23.3% | 0 |
| ref, S7b's B1 | 572 | 0.974 [0.957, 0.984] | 23.2% | 1 (1.000) |

**S15's choose half.** Reachable (5,236): `decide` supports 901 and refutes 129. K3, K4 and B1 promote all 901, at 0.949 [0.933, 0.962]; Known-decided 19.7%. K1 and K2 promote none, so their Known-decided share is the refutations alone (2.5%). With m = 5,236 and T5 left out, no e-value reaches e-BH's bar. All pairs (11,508): K3 and K4 promote all 2,119 supported, at 0.971; K1 and K2 none.

**The guarantee checks.** Realised FDR, with the mean number promoted per batch in brackets:

| Rule | Sim, mixed | Sim, planted nulls | Resampling, S7 dev | Resampling, S15 choose |
|---|---|---|---|---|
| K1 | 0.000 (0.0) | 0.000 (0.0) | 0.004 (105) | 0.000 (0) |
| K2 | 0.000 (0.6) | 0.000 (0.0) | 0.000 (18) | 0.000 (0) |
| K3 | 0.026 (617) | 0.003 (0.0) | 0.026 (286) | 0.051 (451) |
| K4 | 0.026 (615) | 0.090 ± 0.014 (0.1) | 0.026 (286) | 0.051 (451) |
| ref, B1 | 0.026 (608) | **1.000 (34)** | 0.026 (285) | 0.051 (451) |

Under planted nulls (every hypothesis false), S7b's B1 promotes 34 per batch, all of them false. K3 and K4 promote almost nothing. That is what the guarantee buys. On real data the three promote the same sets, because `decide`'s supported set is already at a 3–5% false share, under q = 0.1. A guaranteed rule only prunes when that share rises.

**Why K1 is low.** In the simulation the tests are independent given the truth. Then a product of Bayes factors rarely reaches e-BH's bar of m / (q k), and K1 promotes nothing. On real dev it promotes 215, because true hypotheses' positive results co-occur (T6 with T4) more often than independence predicts.

**Why K2 is lower still.** The largest e-values belong to low-prior hypotheses: they need a large e to pass 0.9. High-prior ones stop early with a small e. Weighting by prior odds shrinks the first and cannot lift the second, so fewer pass.

**The e-values of false hypotheses (V6).** Mean e by prior quartile on dev: 0.62, 0.73, 1.17 and **1.58**. For false hypotheses with high priors, the product of Bayes factors is not a valid e-value on real data. Plausible false hypotheses pass the tests more often than dev's pooled null likelihoods say. K1, K2 and K4 rest on those likelihoods; K3 does not.

**The choice.** All four candidates qualify. K3 and K4 tie at 23.3% Known-decided on dev, so the registered order (K1, K2, K4, K3) picks **K4**.

**The confirm read (C2, once).** 1,445 pairs, 2,877 hypotheses counted. `decide` supports 679 and refutes 168 (at 0.935).

| Rule | Promoted (Known) | Precision [Wilson 95%] | Known-decided | Believed (precision) |
|---|---|---|---|---|
| K1 | 241 | 0.992 [0.970, 0.998] | 14.2% | 438 (0.934) |
| K2 | 0 | | 5.8% | 679 (0.954) |
| K3 | 679 | 0.954 [0.936, 0.968] | 29.4% | 0 |
| **K4 (chosen)** | **679** | **0.954 [0.936, 0.968]** | **29.4%** | 0 |
| ref, B1 | 679 | 0.954 [0.936, 0.968] | 29.4% | 0 |

The Wilson lower bound is 0.936 ≥ 0.90: **confirmed**. Known-decided is 29.4%, against 11.1% for S7b's whole-batch e-BH on dev. The precision, 0.954, is below dev's 0.974 and level with S15's confirm read (0.956). The confirm run's other lines match the dev run's, timings aside.

### Predictions scored

| # | Result | Held? |
|---|---|---|
| V1 | K3: 573 = 2.7 × 215, at 0.974; Known-decided 23.3% | yes |
| V2 | K2 promoted 37 on dev (0 on C2), fewer than K1 | **no** |
| V3 | K4 573 > K1 215 | yes |
| V4 | All of K1–K4 at or under 0.10 in both scenarios (K4 0.090 under planted nulls is the closest); B1 1.000 | yes |
| V5 | At most 0.051 for K1–K4 on both sets | yes |
| V6 | Top prior quartile 1.58 | **no** |
| V7 | K4, by the tie order; K3 promoted the same sets | **no** |
| V8 | 1,445 pairs | yes |
| V9 | 0.954, lower bound 0.936, Known-decided 29.4% | yes |

Six of nine held.

### Decisions

1. **The default batch step is the hybrid with K4 as its Known tier.** It is BH at q = 0.1 over the whole batch on null-model p-values, gated to `decide`'s supported (`hybrid_null_model_tiers` in `hypothesis.hpp`). `decide`'s other supported hypotheses are Believed and never Known.
2. **What it guarantees.** Among the Known, FDR ≤ 0.1, **if** the three-way null likelihoods (fitted on S7's dev) are right for the batch's false hypotheses and the p-values are PRDS. V6 shows the first is off for the top prior quartile. The real-data resamplings still came in at 0.026 and 0.051, and the planted-null simulation at 0.090.
3. **What changes in numbers.** Nothing, on every read so far: K4 promoted every hypothesis `decide` supported, and the Believed tier was empty. What changes is the claim. S7b's step promoted 34 false hypotheses per all-false batch; K4 promotes 0.1. The share decided stays at S7b's level (23% dev, 29% C2) instead of e-BH's 11%.
4. **Open for Odin: K3 instead of K4.** K3 promoted the same sets on every read. Its guarantee needs no model, only a labelled calibration set drawn like the batch. K4 needs no labelled set at run time, which is why the registered order put it first. Switching changes no number so far.
5. **K1 and K2 are not used.** K2 is dropped. K1 stays in the library (`bh_reject_e`).
6. Refutations stay outside batch control, as in S7b.

**Rule 10.** All of this is Cypha's own hypothesis machinery. It never produces an answer for CyphaLM.

### Checks (CTest)

`epistemic_test`'s S7c block (811 checks in all, 0 failed):

- weighted e-BH against plain e-BH and planted cases;
- conformal p-values (planted cases) and conformal selection's FDR by simulation;
- null-model p-values: known answers, the stopping rule, and super-uniformity under the model;
- the tiers;
- the chosen rule on planted batches (batch of 4: Known; batch of 100: Believed; no null model: Believed);
- the chosen rule's FDR by simulation, mixed and all-false.

Planted bugs, each caught:

- conformal denominator, the +1 and ties;
- weights not rescaled;
- stopping ignored;
- outcomes from the true side;
- all Known;
- BH at q = 1.

The normal `--gaps` run is byte-identical to main's (all ten output files; stdout apart from timings). So is `--test-kinds` without `--s7c`.

### How to reproduce (S7c)

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
# dev, S15's choose half, the simulations, the resamplings and the choice
./pattern_learn --test-kinds enwik8 /root/work/f2/v2/aug/seeds.tsv /root/work/f2/wikidata/wd_values.tsv /root/work/f2/v2/aug/patterns.tsv out \
  --phi-wd /root/work/f2/wikidata/phi.tsv --s15 bench/eval/temporal_holdout_dev.tsv --s7c
# the single confirm read on C2
./pattern_learn … final --phi-wd /root/work/f2/wikidata/phi.tsv --s15 bench/eval/temporal_holdout_dev.tsv --s7c --read-c2
cmake --build build --target epistemic_test && build/epistemic_test   # 811 checks
```

The S7c lines start with `S7C`. `OUT.c2_items.tsv` lists C2's hypotheses with their results, prior, posterior, e-value, null p-value and tier. `OUT.c2.prereg.tsv` is C2's pre-registration log.

## S7d: T2 in the loop, Wikidata as one source, K3 against K4, fusion's φ (2026-10-29)

**Odin Loch** · 2026-10-29 · HANDOFF §5.12 and §5.14 settled by measurement (PLAN ledger 55), open to Odin's veto · S9's part is in [S9_WORLDS_AND_FRAGILITY.md](S9_WORLDS_AND_FRAGILITY.md), "S9b"

(The reads, the rules and the predictions below were written and committed before any S7d run. Results are added under them.)

**Rule 10.** Cypha's own hypothesis machinery. Nothing here answers a question.

### The data (fixed now)

- **S7's dev set** (2,950 hypotheses, both sets U and C), with S7b's loop: T1 under the store's φ, T6, T5, T4, three-way add-one likelihoods from dev, cheapest first, `decide` as on main (S8d: support needs a positive independent result), S7c's hybrid with K4 as the Known tier.
- Its items come from two runs of main's tool: `--test-kinds … --s7c` (T6, T5, T4, P0) and `--worlds` (T1 under the store's φ, T2), joined on pair, label, subject, relation and object. A new mode, `pattern_learn --s7d`, reads them. It first reproduces S7c's dev lines from the joined items; nothing below is read until it does.
- **S15's choose half** (reachable pairs), as S7b read it, with its loop (no T5). It is used only in check 5b below; no choice is made on it.
- **Final reads.** If S8e's fresh set is ready during this work, items 1 and 5 are read once on its test part. If not, they rest on dev and the report says so.

### 1. T2 in the loop

- **The test.** S9's chosen T2 (worlds, depth 1, with the type check) as a test kind "T2 worlds": source `store`, not independent of the generator, cost 2. The plan becomes T1, T2, T6, T5, T4.
- **Its likelihoods.** Three-way add-one from dev, with S6's labels. For the reads they are cross-fitted: each half of dev (bit 13 of FNV-1a(subject)) uses T2's likelihoods from the other half. The other tests keep S7b's.
- **The reads.** Supported, refuted and decided; the Known tier (K4) with its precision and Known-decided share; K4's realised FDR in S7c's model simulation (mixed and planted nulls, 400 batches each) and in the dev resampling (400 splits). The plan without T2 is read beside it the same way.
- **Rule: T2 enters the loop when all hold.**
  - (a) The Known-decided share on dev rises by at least 0.5 points over the plan without T2.
  - (b) The Known precision's Wilson lower bound is at least 0.90, and the precision is not more than 0.01 below the plan without T2.
  - (c) K4's realised FDR is at most 0.10 + 2 SE in both simulations and in the resampling.
  - (d) If S8e's fresh set is read: the Known precision's Wilson lower bound there is at least 0.90.

### 2. Wikidata as one source or per claim (the loop's part)

- **Per claim** (the loop as it is): T5 is independent of a store-based generator, because the claim that judges h is not one of the claims the generator read.
- **One source:** Wikidata is one provider. When the generator's store holds a Wikidata claim about h's subject, T5 is not independent of the generator.
- **The reads, both ways, on dev:** Known, precision and decided; and the Known promotions whose only independent positive result is T5 (T4 not positive), with their precision.
- **Rule** (it protects against one provider carrying a promotion): **Wikidata counts as one source**, unless no Known promotion on dev rests on T5 alone when it is counted per claim. One source then applies to the loop, to fragility and to the blast radius (S9b).

### 5. K3 against K4

S7c found that K3 and K4 promoted the same sets on every read. The reads below look for where they differ.
- **5a. Plausible nulls.** Batches drawn from dev's loop results, with every false member taken from the top prior quartile (where S7c found the false e-values inflated: mean 1.58) and true members drawn at random. False fractions 0.5 and 0.8, batch size 600. K3's calibration set is drawn the same way from the other bit-13 half, so it is exchangeable with the batch. 400 batches per fraction. Realised FDR with its SE for K3, K4 and S7b's step.
- **5b. A calibration set from another distribution.** K3 calibrated on S7's dev, applied to S15's choose half (reachable). Its realised false share, with a bootstrap interval over subjects (2,000 resamples), beside K4's on the same batch. This is how K3 would run in practice: a labelled set made earlier, not drawn with the batch.
- **Rule: switch the Known tier to K3 only when all hold.**
  - (a) In 5a, K4's realised FDR exceeds 0.10 + 2 SE at some fraction while K3's stays at or under 0.10 + 2 SE at both.
  - (b) In 5b, K3's realised false share is at most 0.10, with its interval's upper bound at most 0.15.
  - (c) K3's Known-decided share on dev is within 1 point of K4's.
  - Otherwise K4 stays.

### 6. Fusion's φ

- **The reads.** `pattern_learn`'s normal run (`--cv --wikidata`, S3's command) with fusion's φ as now (`kRel`'s hand-set table) and with the store's φ estimated on the v2 seed store (written by `--worlds --s9b` as `OUT.phi_store.tsv`, passed with the existing `--phi`). Compared: admitted facts, their agreement with Wikidata on held-out subjects, and S3's gap counts by kind.
- **Rule.** The store's φ is the recommended setting for fusion when its agreement with Wikidata is not lower (point estimate) and the admitted facts fall by at most 2%. Either way the default in this change stays byte-identical, because S3's gap map is every change's parity reference. A recommendation to switch is recorded for Odin and the integrator.

### Predictions (registered before the runs)

| # | Prediction |
|---|---|
| D1 | `--s7d` reproduces S7c's dev lines (as changed by S8d: K4 572 at 0.976) from the joined items |
| D2 | T2 moves the Known-decided share by less than 0.5 points: rule 1 fails and T2 stays out |
| D3 | Per claim, more than 50 Known promotions on dev rest on T5 alone: rule 2 picks one source |
| D4 | One source costs at least 20% of dev's Known promotions |
| D5 | In 5a, K4's realised FDR stays at or under 0.10 + 2 SE at both fractions: K4 stays |
| D6 | In 5b, K3's realised false share is under 0.10 |
| D7 | Fusion with the store's φ changes admitted facts by less than 2% and agreement by less than 0.005 |

### Results (run 2026-10-29)

Tool: `pattern_learn --s7d` (commit e4afd6f), on items from main's tool (`--test-kinds … --s7c`, dev only) and from `--worlds --s9b`. A run takes 28 s at 45 MB. S8e's fresh set was not ready, so **every read here is on dev** (and S15's choose half for 5b); no final read was made.

**The reproduction (D1: hit).** The joined items (2,958; none unmatched) give S7c's dev lines exactly, as S8d changed them: K4 promotes 572 at 0.976 [0.959, 0.985], Known-decided 0.232; K1 215, K2 37, K3 572, S7b's step 571.

#### 1. T2 in the loop

| | Known (K4) | Precision [Wilson 95%] | Refuted (precision) | Known-decided |
|---|---|---|---|---|
| Without T2 | 572 | 0.976 [0.959, 0.985] | 113 (0.965) | 0.2322 |
| With T2 (cross-fitted) | 572 | 0.976 [0.959, 0.985] | 117 (0.957) | 0.2336 |

- T2's likelihoods: all of dev BF(positive) 3.35, BF(inconclusive) 0.943, as S9 measured; cross-fitted 2.63 and 4.33 by half.
- T2 changes no supported hypothesis. It refutes 4 more, one of them true.
- K4's realised FDR with T2: 0.026 (mixed), 0.060 ± 0.012 (planted nulls), 0.025 (resampling). All under 0.10 + 2 SE.
- **Rule: (a) fails** (+0.14 points against the 0.5 needed); (b) and (c) hold. **T2 stays out of the loop.** It fires on 5% of dev and is never negative, so it can only nudge posteriors that `decide` already settles.

#### 2. Wikidata as one source or per claim (the loop)

- 1,924 of 2,958 dev items have a subject with a Wikidata claim in the reduced store.
- **Per claim** (the loop as it is): 54 of the 572 Known promotions rest on Wikidata alone. All 54 are true (1.000 [0.934, 1.000]).
- **All 54 have subjects with no Wikidata claim in the store.** So Wikidata judged a subject the generator knew only from other sources. Had the loop gone on to T4, 47 of them would also have had a corpus yes.
- **One source:** 572 Known at 0.976. No promotion differs.
- **Rule: one source.** 54 promotions rest on Wikidata alone when it is counted per claim, so the protective count applies. On S7's dev it costs nothing.
- **Where it bites.** Only where the generator's store holds a Wikidata claim about h's subject and Wikidata is h's only independent yes. The generators' own loops (S8a's and S8b's, and S8d's source stage) still count Wikidata per claim. Wiring the rule into them changes their outputs; it is left for after S8e merges, so S8e's reads are not moved.

#### 5. K3 against K4

**5a. Plausible nulls.** 1,479 true; 659 false with a prior in the top quartile (p0 ≥ 0.473), whose mean e-value is 1.09 and of which `decide` supports 10. 400 batches of 600 per false fraction:

| False fraction | K3 FDR (SE) | K4 FDR (SE) | S7b's step FDR (SE) | Promoted per batch (K3 / K4 / S7b) |
|---|---|---|---|---|
| 0.5 | 0.039 (0.001) | 0.039 (0.001) | 0.039 (0.001) | 117.5 / 117.7 / 117.5 |
| 0.8 | 0.063 (0.003) | 0.054 (0.004) | **0.140 (0.002)** | 35.9 / 25.1 / 52.6 |

K4 stays under its bar even when most of the batch is plausible and false; S7b's selective step does not.

**5b. A calibration set from another distribution.** K3 calibrated on S7's dev, applied to S15's choose half (reachable: 5,236 hypotheses, 833 subjects): 901 promoted, false share 0.051 [0.036, 0.067] (bootstrap over subjects). K4 promotes the same 901.

**Rule: (a) fails** (K4 is never over its bar); (b) and (c) hold. **K4 stays.** The two rules promote the same sets on every real read. K4 needs no labelled set at run time.

#### 6. Fusion's φ

The store's φ on the v2 seed store differs from `kRel`'s hand-set table in most relations. Examples: language 0.59 against 0.20, occupation 0.26 against 0.10, birth_place 0.73, capital 0.88.

| | Hand-set φ (as now) | Store's φ |
|---|---|---|
| Admitted facts (P ≥ 0.8) | 1,216 | 1,214 |
| Candidates (0.5 ≤ P < 0.8) | 178 | 190 |
| Agreement with Wikidata on held-out subjects | 173 / 182 [0.909, 0.974] | 173 / 182 [0.909, 0.974] |
| S3's gaps: certainty / contradiction | 1,342 / 128 | 1,344 / 130 |

- The two admitted facts lost are Chad's language, "arabic" and "Arabic": the store's φ makes language more functional, so French competes harder. Chad has both languages, so that loss is a small error the store's φ introduces.
- **Rule: met** (agreement not lower; admitted −0.16%). **The store's φ is the recommended setting for fusion.** The default is unchanged in this change: S3's gap map is every change's parity reference. Switching it is a decision for Odin and the integrator; it changes the v2 store and S3's map.

### Predictions scored (S7d)

| # | Prediction | Result |
|---|---|---|
| D1 | `--s7d` reproduces S7c's dev lines | **Hit:** exactly |
| D2 | T2 moves Known-decided by < 0.5 points; stays out | **Hit:** +0.14 |
| D3 | > 50 Known promotions rest on T5 alone; one source | **Hit:** 54 |
| D4 | One source costs ≥ 20% of Known | **Miss:** 0% (all 54 are about subjects with no Wikidata claim in the store) |
| D5 | K4 stays under its bar in 5a: K4 stays | **Hit** |
| D6 | K3 across distributions under 0.10 | **Hit:** 0.051 |
| D7 | Fusion with the store's φ: admitted change < 2%, agreement change < 0.005 | **Hit:** −0.16%, 0 |

6 hits, 1 miss.

### Decisions (S7d), open to Odin's veto

1. **T2 stays out of the loop** (rule 1).
2. **Wikidata counts as one source** (rule 2): in fragility and the blast radius (as S9 ran) and in independence. On S7's dev it changes no promotion. Wiring it into the generators' loops is the follow-up.
3. **K4 stays the Known tier** (rule 5). K3 is kept in the library.
4. **The store's φ is the recommended setting for fusion** (rule 6), with the default unchanged until Odin and the integrator switch it.

### How to reproduce (S7d)

```bash
g++ -O2 -std=c++20 -pthread -I native/include native/tools/pattern_learn.cpp -o pattern_learn
./pattern_learn --test-kinds enwik8 seeds.tsv wd_values.tsv patterns.tsv s7 --phi-wd phi.tsv --s15 bench/eval/temporal_holdout_dev.tsv --s7c
./pattern_learn --worlds enwik8 seeds.tsv wd_values.tsv facts_f2.tsv bench/eval/temporal_holdout_dev.tsv s9 --s9b
./pattern_learn --s7d s7.items.tsv s9.dev_items.tsv s9.s9b_dev_items.tsv s7.s15_items.tsv s7d
# fusion's phi: S3's normal command, then the same with --phi s9.phi_store.tsv
```
Runs are in `/root/work/s9b/runs/` (`s7d.out`, `g0`, `gphi`).
