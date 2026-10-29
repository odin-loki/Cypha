# S11b: literature-based discovery (G12, Swanson's ABC)

**Odin Loch** · 2026-10-29 · DISCOVERY §5 (G12), §8 P8, §13 S11 · follows [S8B_LINK_PREDICTION.md](S8B_LINK_PREDICTION.md), [S8D_PROMOTION.md](S8D_PROMOTION.md) and [S10_CURIOSITY_SCHEDULER.md](S10_CURIOSITY_SCHEDULER.md)

(The design, the rules and the predictions below were written and committed before any G12 run. Results are added under them.)

**Question.** In the 2006 text, A appears with B, and B appears with C, but A and C never appear together. Does current Wikidata link A and C more often than chance, and more often than a popularity baseline would guess? That is Swanson's ABC, and DISCOVERY's P8 measures it by precision@k against later sources.

**S11's gate for G12 (DISCOVERY §13).** ABC precision@k. The pass rule is fixed below.

**Rule 10.** G12 proposes hypotheses for Cypha's own store. It never answers a question, and nothing here reaches `qa` or `chat`.

## What was looked at before this was written

While designing, I counted on S15 dev how many true entity-valued facts could be reached at all: the subject is an enwik8 article, the value is a linked entity in at least two articles, and no article holds both. About 3,240 of S15 dev's 16,238 true entity-valued facts pass that (a 250, b 648, c 72, u 2,271), and about 1,020 of those have the value named on the subject's page by S15's page rule. No generator score was computed.

## The text (fixed before any run)

- **Documents.** enwik8's main-namespace articles (`pattern_learn`'s `list_pages`: 8,587 articles, redirects excluded).
- **Entities of an article:** its own title, and the target of every wiki link (`[[target]]`, `[[target|label]]`; a `#section` is dropped).
  - Targets are resolved through enwik8's redirect pages (at most three hops). The first letter is capitalised and `_` becomes a space.
  - A target with a colon followed by a non-space is another namespace or language and is dropped, as `list_pages` does.
- **Entities left out of every role:** years ("1952", "300 BC"), decades ("1990s"), centuries ("19th century"), days of the year ("March 4", "4 March"), month names, and "List of …" titles. They are dates and lists, not things.
- **Counts.** N is the number of articles. n_X is the number of articles holding X. n_XY is the number holding both.
- **Surface forms of an entity:** `norm_ent` of its title, and of each enwik8 redirect title that resolves to it (at least 3 characters).
- **Never together in the 2006 text.** A pair (A, C) is eligible only if all three hold:
  - n_AC = 0;
  - no surface form of C occurs, on word boundaries, in A's article text (`norm_q` of the raw article text);
  - if C is an article, no surface form of A occurs in C's article text.

## G12 (fixed before any run; `native/include/cypha/eval/gen_abc.hpp`)

- **A**, the query: each S15 subject of the half (every S15 subject is an enwik8 article).
- **B**, the link: an entity with 2 ≤ n_B ≤ H that shares an article with A (B ≠ A).
- **C**, the candidate: an entity with n_C ≥ 2 and C ≠ A, eligible as above.
- **Score (open form, A given):** score(A, C) = Σ_B idf(B) · w(A, B) · w(B, C), with idf(B) = ln(N / n_B).
- **Closed form (A and C given):** the same sum, and the three B with the largest terms as the explanation.
- **Weights w.** PPMI(x, y) = max(0, ln(N · n_xy / (n_x · n_y))). Three variants:
  - **PPMI**;
  - **PPMI2**: PPMI where n_xy ≥ 2, else 0 (a pair seen in one article counts for nothing);
  - **LMI**: n_xy · PPMI (local mutual information, weighted by frequency).
- **Hub cap** H: 50 or 200.
- So there are six configurations. One is chosen on dev (below).

## The baselines (fixed before any run)

- **Random:** for each A, candidates drawn uniformly (seed 20261029 + FNV-1a(A)) from the entities with n ≥ 2, kept when eligible.
- **Degree** (popularity): score(A, C) = n_A · n_C, over eligible C. Within one A this ranks by n_C.
- **Ablation, not a baseline:** common neighbours, the number of shared B (cap 200), to see whether the PMI weights matter.

## The reads

1. **Open form, precision@k.**
   - For each method, each query A gives its 10 best eligible candidates. One A cannot fill the list.
   - The pooled list is sorted by score, ties by FNV-1a(A + C). precision@k = hits among the first k, for k = 10, 50, 100 and 500, with Wilson 95% intervals.
   - Also reported: the mean over queries of precision@10 in each query's own list, and the hits by relation and S15 category.
2. **Closed form.**
   - S15's pairs of the half: a true entity-valued fact (A, r, v) and its false partner (A, r, v′). Both values must resolve to entities with n ≥ 2 by their surface forms, and both must be eligible.
   - Each value gets its closed-form score. Reported: the pair AUROC (true value above its false partner), and precision@k over the pooled list of all those values. Its baselines are the true share (random) and n_v (degree).
3. **Typed hypotheses through the loop (promotion).**
   - The chosen configuration's 3 best eligible candidates per query A.
   - **A relation is named** from S15 dev: among the true entity-valued facts whose relation is one of `pattern_learn`'s, take the most common relation for the pair of types (type of subject, type of value). It is used only if it holds at least half of them; otherwise the pair stays an untyped "related" hypothesis and is not tested or promoted.
     - Types come from the 2006 text, as S8b's tests read them: person, country or place by the article's flags and the type lexicons; else other.
   - **P0:** an isotonic fit on dev of the G12 score against S15's grade of (A, r, C), using S8c's grader on graded items only, clamped to [0.02, 0.98].
   - **The loop:** S7's T1 (consistency) and T6 (compression) on S8b's 2006-text store, cheapest first, then `decide`. Neither is independent.
     - **T4 is left out.** G12 picks pairs whose article never names C, so T4 is negative by construction. Its sensitivity and specificity, measured on other generators, do not hold here.
   - Then S8d's source stage (`--sources … --fresh`, S8d's calibration on S15 dev's choose half): DB-IB, WP-LEAD and GN as further tests. A source carries a promotion only where S8d allows it.
   - **Promoted** = supported by `decide`. Graded by S8c's grader on the half's S15 file. Precision with Wilson intervals. S8's gate (at least 20 graded, precision at least 0.9) is the reference, not G12's gate.
   - **Pool files**, in S8d's fresh-set format so S10 can replay them later: `*.abc_{dev,test}.hyps.tsv` (S8b's fresh columns), their pre-registration logs, and the source stage's `*.s8d_fresh.tsv` and `*.s8d_fresh_ran.tsv`.

**Hit (the primary).** (A, C) is a hit if the half's S15 file holds a true fact (A, r, v), r entity-valued (not a date relation), with `norm_ent(v)` equal to a surface form of C. Any category counts: current Wikidata links them, and the 2006 text never put them together. **Strict:** the same, with the category b or c (true by 2006 but unstated, or true only after 2006-03-04). Everything else is a miss. S15 holds only 34 properties, so precision is a lower bound for every method alike.

**Splits.** S15 dev for choices. S15 test read once, for the final read, logged in S15's "Reads" table. Making the test pool reads only the test file's subjects; its labels are read once, with `--read-test`.

## Decision rules (fixed before any run)

1. **Dev choice.** Of the six configurations, the one with the highest dev precision@100 (primary hit). A configuration later in the list (PPMI H 200, PPMI H 50, PPMI2 H 200, PPMI2 H 50, LMI H 200, LMI H 50) replaces an earlier one only if it is more than 0.01 higher.
2. **S11's ABC gate.** LCI ≥ 0.95 on test (PLAN §8.2, adapted as S10 did), against the better of the two baselines, on precision@100:
   - P_gain: the share of 4,000 resamples of the query subjects (with replacement) in which G12's P@100 beats the better baseline's by more than ε = 0.01;
   - n_eff: query subjects with at least one eligible true value; below 200, LCI is capped at 0.5;
   - r: 1 if G12's P@500 is at least the better baseline's minus 0.01 in at least 95% of resamples, else 0;
   - γ = min(1, (test gain / dev gain) / 0.5), gain = G12's P@100 minus the better baseline's; if the dev gain is not positive, γ is 1 when the test gain is positive, else 0;
   - c = 1 (nothing is calibrated by this change; P0's Brier score is reported);
   - σ: 1 if the P@100 difference has the same sign in three disjoint blocks of query subjects (FNV-1a(A) mod 3), else 0.5;
   - κ: 1 if the mode runs in under 15 minutes and peaks under 3 GB, else 0.
3. **G12 earns its place** (S8b's rule 3) if its test P@100 beats the random baseline with the bootstrap 95% interval of the difference above 0. Otherwise its code goes, and this report keeps its numbers.
4. Nothing is changed after the test read.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| B1 | Dev: 1,500–3,000 eligible true values (primary hits possible) over the dev queries, after the text rule |
| B2 | Dev choice: LMI (frequency-weighted) wins; plain PPMI ranks rare entities first and has the lowest P@100 of the three weights |
| B3 | Dev, open form: the chosen G12 configuration's P@100 is 0.02–0.15; the degree baseline's is 0.03–0.15; random under 0.005 |
| B4 | Test: G12 beats random (rule 3 holds) but does not beat the degree baseline by ε: LCI < 0.95, the gate is not met |
| B5 | Closed form, dev: pair AUROC 0.55–0.70 |
| B6 | Among the chosen configuration's dev hits, the strict ones (b or c) are at least 25% |
| B7 | 20–50% of the top-3 candidates can be typed; under 2% of typed hypotheses are supported after the source stage |
| B8 | The mode runs in under 10 minutes and peaks under 2 GB |

## Results on dev (2026-10-29)

Tool: `pattern_learn --gen-abc` at commit d62d653 with one fix to its report code: the first run crashed before printing anything. The by-relation lines indexed the methods with the not-yet-chosen configuration (−1). They now print after the dev choice. Nothing else changed.

### The text

- 8,587 articles, 262,578 entities, 676,980 entity mentions. Built in 5 s.
- Usable entities with at least 2 articles: 75,747.
- **Dev queries:** 1,997 S15 subjects, all of them articles. They hold 16,238 true entity-valued facts.
- **Eligible true values: 2,533**, on 818 queries. A value is eligible when its entity has at least 2 articles, never shares an article with the subject, and neither article names the other.

### Read 1: open form (dev)

Each method's 10 best eligible candidates per query, pooled (about 20,000 picks a method). Wilson 95% intervals.

| Method | P@10 | P@50 | P@100 | P@500 | Mean per-query P@10 | Hits in all picks |
|---|---:|---:|---:|---:|---:|---:|
| G12 PPMI H200 | 0 | 0 | 0 [0, 0.037] | 0.002 | 0.0013 | 26 |
| G12 PPMI H50 | 0 | 0 | 0 | 0.002 | 0.0014 | 28 |
| G12 PPMI2 H200 | 0 | 0 | 0 | 0 | 0.0013 | 26 |
| G12 PPMI2 H50 | 0 | 0 | 0 | 0 | 0.0012 | 24 |
| G12 LMI H200 | 0 | 0.020 | 0.010 [0.002, 0.054] | 0.004 [0.001, 0.014] | 0.0026 | 51 |
| G12 LMI H50 | 0 | 0.020 | 0.010 | 0.002 | 0.0022 | 43 |
| Common neighbours (ablation) | 0 | 0 | 0 | 0.002 | 0.0026 | 52 |
| Degree (n_A · n_C) | 0 | 0 | 0 | 0 | 0.0021 | 42 |
| Random | 0 | 0 | 0 | 0 | 0 | 0 |

- **The open form finds almost nothing at the top of any list.** About 1 pick in 400–800 is a hit, for every non-random method.
- **The dev choice (rule 1) is PPMI H200.** LMI H200 scored 0.010 at P@100, one hit in 100. That is not more than 0.01 above PPMI's 0, so the rule keeps PPMI.
- Against degree on dev, PPMI H200's P@100 difference is +0.000 [+0.000, +0.010]; at P@500, +0.002 [+0.000, +0.006].
- **What the hits are.**
  - PPMI H200: 26 hits, 9 of them strict (8 b, 1 c). By relation: occupation 8, genre 4, headquarters 3.
  - Degree: 42 hits, almost all "country = United States" or "Russia" (category u). Only 2 are strict.
  - Examples of G12's hits:
    - Bob Dylan — Nobel Prize in Literature (c, awarded in 2016);
    - Évariste Galois — Paris (death place, b);
    - Republic of Ireland — UNESCO (member of, b).

### Read 2: closed form (dev)

1,353 S15 pairs where both the true value and its false partner are eligible.

| | G12 PPMI H200 | Degree (n_v) |
|---|---:|---:|
| Pair AUROC | **0.662** | 0.484 |
| P@10 / P@50 / P@100 / P@500 | 0.50 / 0.62 / 0.66 / 0.642 | 0.80 / 0.72 / 0.56 / 0.498 |

- **Given A and C, the B links do separate true values from false ones.** The degree baseline does not (0.484).
- Its best-scored true pairs come with their links, for example "Bob Dylan — Nobel Prize in Literature" through Bessie Smith, Benjamin Zephaniah and Living Colour.

### Read 3: the typed pool and the loop (dev)

- **Naming a relation.** From S15 dev, by the types of subject and value (from the 2006 text). Some pairs of types get none: other–other, other–person, place–other and place–place stay untyped.
- **The pool.** Top-3 candidates: 5,988. 2,216 are typed and 3,772 untyped.
  - 475 of the typed ones are graded, and **4 are true (0.008)**. So the isotonic P0 sits near its floor of 0.02.
- **T1 and T6:** 2,179 refuted, 37 queued.
- **S8d's source stage** (`--sources … --fresh`): 0 supported. 2,191 refuted; of the 269 graded on S15's confirm half, 99.6% are false.

### The test pool, fixed before the read

Built from S15 test's subjects alone; no label was read. 2,062 queries, 6,183 top-3 candidates, 2,235 typed. After T1 and T6: 2,218 refuted, 17 queued. The source stage supported none.

| File | sha256 |
|---|---|
| `g12.abc_dev.hyps.tsv` | `9ced2df70afd7a49fbb6a31557a3a3a5a1808224eb3b341c607adf1d1df94403` |
| `g12.abc_test.hyps.tsv` | `bb2d0bcf389562a1e53333bb863d4b2754102b9a44f7b3b4946241c5220f25f3` |
| `g12.abc_test.prereg.tsv` | `4c0ed9218ffb73acb364e5e90e12c3ffcab5a0b32d8eae9733fdc054664459fd` |
| test source stage `src.s8d_fresh.tsv` | `83464098c4d67df09004fe3e39f699bb19284b30e96a67dae631e3092f8dffff` |
| test source stage `src.s8d_fresh_ran.tsv` | `3ee3fd1c063ace2b154f24c1f1f2f4f93e9232fc1aeab59987131ac90e0495be` |

### Cost

The dev run, which also builds the test pool, takes 670 s and peaks at 443 MB. The machine was shared with other agents' runs.

## The test read (once, after commit 0230741)

One run with `--read-test`. It also rebuilt both pools: the test pool and its log byte for byte against the hashes above, and the dev pool too. It took 798 s and peaked at 435 MB.

### Read 1: open form (test)

2,062 queries; 16,447 true entity-valued facts; **2,478 eligible true values** on 862 queries.

| Method | P@10 | P@50 | P@100 | P@500 | Mean per-query P@10 | Hits in all picks (strict) |
|---|---:|---:|---:|---:|---:|---:|
| **G12 PPMI H200 (chosen)** | 0 | 0 | **0 [0, 0.037]** | 0.002 [0.000, 0.011] | 0.0013 | 26 (8) |
| G12 PPMI H50 | 0 | 0 | 0 | 0.002 | 0.0013 | 26 (8) |
| G12 PPMI2 H200 | 0 | 0 | 0 | 0 | 0.0011 | 22 (7) |
| G12 PPMI2 H50 | 0 | 0 | 0 | 0 | 0.0009 | 18 (7) |
| G12 LMI H200 | 0 | 0 | 0 | 0 | 0.0027 | 56 (18) |
| G12 LMI H50 | 0 | 0 | 0 | 0 | 0.0029 | 59 (20) |
| Common neighbours (ablation) | 0 | 0 | 0 | 0 | 0.0028 | 58 (14) |
| Degree (n_A · n_C) | 0 | 0.020 | **0.010 [0.002, 0.054]** | 0.004 [0.001, 0.014] | 0.0022 | 46 (4) |
| Random | 0 | 0 | **0** | 0 | 0.0001 | 2 (1) |

- **Differences (2,000 resamples of queries).** G12 minus degree: P@100 −0.010 [−0.030, +0.000]; P@500 −0.002 [−0.008, +0.004]. G12 minus random: P@100 +0.000 [+0.000, +0.000]; P@500 +0.002 [+0.000, +0.006].
- **LCI** (rule 2):
  - P_gain 0.0000, n_eff 862, r 1;
  - γ 0 (gain dev +0.000, test −0.010);
  - c 1, σ 0.5 (block signs +1, −1, 0), κ 1 (797 s, 435 MB).
  - **LCI 0.000: S11's ABC gate is not met.**
- **Rule 3:** G12's P@100 does not beat random (+0.000 [+0.000, +0.000]). **G12 does not earn its place, so its code goes** (below).
- **What G12 does find, deep in its lists.**
  - About a third of its hits are strict: true by 2006 but unstated, or true only later.
  - The degree baseline's are mostly "English language" and "United States" (strict 4 of 46).
  - Examples from the test half:
    - Dolly Parton — Rock and Roll Hall of Fame (c, 2022);
    - David Beckham — English Football Hall of Fame (c, 2008);
    - I. M. Pei — Guangzhou (birth place, b);
    - Destroy All Monsters — Akira Ifukube (composer, b).
  - They sit far from the top of the pooled list.

### Read 2: closed form (test)

| | G12 PPMI H200 | Degree (n_v) |
|---|---:|---:|
| Pairs with both values eligible | 1,322 | 1,322 |
| Pair AUROC | **0.682** | 0.472 |
| P@10 / P@50 / P@100 / P@500 | 0.60 / 0.74 / 0.70 / 0.660 | 0.90 / 0.70 / 0.57 / 0.494 |

Given A and C, the B links separate a true value from its false partner on test as on dev (0.662). Degree does not.

### Read 3: the pool (test)

2,235 typed hypotheses, 519 graded on S15 test. **Supported: 0.** Refuted: 2,229; 98.3% of the graded refutations are false. S8's reference gate is not met (no promotion).
- On dev the same: 0 supported, 2,191 refuted (99.2% of 475 graded false).
- **Why nothing is promoted.** Only 4 of 475 graded dev hypotheses were true (0.8%), so the isotonic P0 sits at its floor (0.02). T1 and T6 refute most of them, and a source positive (LR+ about 20) cannot lift 0.02 past 0.9.

## Predictions, scored

| # | Prediction | Result |
|---|---|---|
| B1 | 1,500–3,000 eligible true values on dev | **Hit:** 2,533 |
| B2 | LMI chosen; plain PPMI lowest P@100 | **Miss:** LMI scored highest (0.010), but not by more than 0.01, so rule 1 kept PPMI; PPMI and PPMI2 tied at 0 |
| B3 | Dev P@100: G12 0.02–0.15, degree 0.03–0.15, random < 0.005 | **Miss:** 0, 0 and 0. Open-form precision at the top is near zero for every method |
| B4 | G12 beats random but not degree; gate not met | **Miss in part:** the gate is not met, but G12 does not beat random either (rule 3 fails) |
| B5 | Closed-form AUROC 0.55–0.70 on dev | **Hit:** 0.662 (test 0.682) |
| B6 | Strict hits at least 25% of the chosen configuration's dev hits | **Hit:** 9 of 26 (35%) |
| B7 | 20–50% typed; under 2% supported | **Hit:** 37% typed; 0 supported |
| B8 | Under 10 minutes, under 2 GB | **Miss:** 670 s and 798 s (shared machine); 443 MB |

4 hits, 4 misses.

## Decisions (by the registered rules)

- **S11's ABC gate is not met** (LCI 0.000). Open-form ABC over enwik8's articles, judged by S15, finds almost nothing at the top of its lists: about 1 pick in 800 is a hit, and none in its top 100.
- **G12 does not earn its place (rule 3), so its code goes.** The removal is its own commit after this report. The code that produced these numbers is at commit 0230741, and the commands below rerun there.
- **The closed form is the part that works.** Given A and C, the strength of their B links separates true values from false partners (AUROC 0.68 on test, against 0.47 for popularity). It was not a gate here. Using it would need its own registered read, for example as a support term in S6's P0 or as a test kind (decision for Odin).
- **No pool for S10 from G12.** It promotes nothing, so there is no confirmed value for a scheduler to find.

## How to reproduce (at commit 0230741)

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
E=/root/cdata/enwik8; F=/root/work/f2
./pattern_learn --gen-abc $E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval dev/g12
# S8d's source stage on each pool (S8d's calibration from its dev files; empty.tsv is a header-only G2 file)
O8=/root/work/s8d/runs/after2
S="--sources bench/eval/temporal_holdout_dev.tsv tables $F/wikidata/wd_values.tsv $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv"
for h in dev test; do ./pattern_learn $S src_$h/src $O8/s8a.r1_dev.hyps.tsv $O8/s8a.r2_dev.hyps.tsv $O8/s8b.promo.tsv \
  --fresh empty.tsv dev/g12.abc_$h.hyps.tsv; done
# the test read (once)
./pattern_learn --gen-abc … bench/eval test/g12 --read-test --stage-dev src_dev/src.s8d_fresh.tsv --stage-test src_test/src.s8d_fresh.tsv
```

- `tables` is S8c's (see its report).
- **Checks at 0230741:**
  - `native_epistemic_test` passes 1,224 checks; `test_abc_S11b` adds 56 (weights, the toy corpus's exact scores, the hub cap, the exclusions, the closed form, the text helpers, the pooled list).
  - Six planted bugs are caught:
    - a candidate that shares a document with A;
    - a strict hub cap;
    - idf dropped;
    - scores not reset between queries;
    - a closed form that scores together pairs;
    - min_df ignored for C.
  - The other 8 CTests of HANDOFF §4.4 pass.
  - S3's `--gaps` run is byte-identical: the gap map, its cells and demand files, and the 7 v2 store files.
