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
