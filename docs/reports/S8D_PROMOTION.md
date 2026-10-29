# S8d: promotion needs a positive result, sources calibrated on the generators' errors, and a fresh confirm-half set

**Odin Loch** · 2026-10-29 · DISCOVERY §7.2, §7.4, I4 · follows [S8C_EVIDENCE_SOURCES.md](S8C_EVIDENCE_SOURCES.md)

(The design, the rules and the predictions below were written before any S8d run. Results are added under them.)

**Rule 10.** Everything here is evidence for Cypha's own store and hypotheses. None of it ever answers a question for CyphaLM.

**Why.** S8c found three faults:
- `decide` counted a *negative* independent result as the independent evidence that support needs. All five false confirm-half promotions in S8c came through that.
- WP-LEAD's likelihoods came from S15's drawn false partners. On the generators' own wrong values its positive was right only 25 times in 43 (likelihood ratio about 2.7, not 28).
- Only 56 of the 2,029 dev hypotheses could be graded on S15's confirm half, so S8's gate (precision ≥ 0.9 with ≥ 20 graded promotions) could not be met.

## Part 1. `decide` needs a positive independent result (a bug fix)

DISCOVERY §7.2 says: "Supported: posterior above 0.9 and at least one piece of evidence independent of the generator." Evidence that h is false is not support. The fix:
- **Supported** needs a posterior above 0.9 and at least one test that is done, conclusive, independent, not from the generator, and **positive**.
- A negative independent result still moves the posterior. It still enters ECS as Oppose on promotion. It never lets `decide` support.
- The same change goes into the three places that copy `decide`'s stopping rule: `null_model_pvalue`'s walk (S7c's K4), `pattern_learn`'s S7c simulation, and S8c's read 3.

**Reads rerun on dev, before (main `aca60a5`) and after:**
- S7, S7b and S7c: one `--test-kinds … --phi-wd … --s15 temporal_holdout_dev.tsv --s7c` run (S7's dev loop, S7b's dev loop and S15's choose half, S7c's candidates, simulations and resamplings). Not `--read-confirm`, not `--read-c2`.
- S8a: `--gen-rules` without `--read-test` (R1 and R2 dev).
- S8b: `--gen-links` without `--read-test` (PROMO dev and S15 dev).
- S8c: `--sources` on the S8a and S8b dev files of the same binary. Its read 3 grades on S15's confirm half. That half was already read for these hypotheses. The rerun only measures the fix; nothing is chosen on it.

## Part 2. Each source calibrated on the generators' own errors

**Calibration set.** The generators' existing dev hypotheses (S8a-R1, S8a-R2 and S8b, as `--sources` loads them), with their subject in S15's **choose** half, graded by S15's choose half with S8c's grader (true, false, or ungraded). Each distinct (subject, relation, value) counts once. The confirm half is not used.

**Calibrated likelihoods.** For each (source, relation group) that S8c accepted (DB-IB for dates, places, links and lexical; WP-LEAD for the same four; GN for places), the three-way likelihoods on the calibration set, with S7b's add-one (`likelihoods_from`). A cell with fewer than 10 graded true or 10 graded false items uses the source's pooled cell over its accepted groups instead. These likelihoods replace read 1's in every Bayes factor of that source.

**The rule (registered before any calibration count was made).** A (source, group) may supply the positive result that support needs only when its calibrated positive likelihood ratio, P(positive | true) / P(positive | false), is **at least 10**: S8c's bar, now measured on the generators' errors. A source below the bar still updates the posterior with its calibrated factors. Its test is marked not independent, so it can never carry a promotion.

## Part 3. A fresh set on S15's confirm half, read once

**The slots.** Every (subject, relation) of S15's dev file whose subject is in the confirm half (bit 17 of `pattern_learn`'s `fnv(subject)` = 1) and whose relation is one of `pattern_learn`'s. S15's labels are not read to build the set.

**Hidden facts.** Every triple of the generators' store (S8a's store K, S8b's full store: every seed row, 22,783 triples) in those slots is removed. The generators work on the reduced store:
- **G2 (S8a):** rules mined on the reduced store, combined as R1 chose (noisy-or). Every rule candidate of each hidden slot that the store does not already hold is a hypothesis. P0 as R1. Tests by R1's dev plan (S8a's dev run kept T1, T6 and T4). T5 is left out: Wikidata is what S15 grades against. The loop as R1.
- **G6/G7 (S8b):** the generator S8b chose on dev (by MRR on the random hide) gives each hidden slot its top candidate. The prior is the isotonic map fitted on PROMO's dev items (their scores and hidden-fact truths), as PROMO's own. Tests T1, T6 and T4, S8b's loop.
- Each generator writes its hypotheses with their P0, posterior and state, and no grade. The two files' SHA-256s are added to this report, and committed, before the read.

**The read (once).** `pattern_learn --sources … --fresh G2_FILE LINKS_FILE`:
- Each hypothesis starts from its generator's posterior. The accepted sources follow as further tests, in the order DB-IB, WP-LEAD, GN, each with its calibrated likelihoods. After each result `decide` (with the Part 1 fix) is applied, and testing stops once the hypothesis is no longer queued (§7.2).
- **Promoted** = supported by `decide`. Row 70's Known tier (K4: BH at 0.1 on null-model p-values over the batch, on the source tests) is reported beside it.
- Graded on S15's confirm half with S8c's grader. Precision = true / graded, with Wilson 95% intervals. Reported for all (each distinct hypothesis once), per generator, per source that gave the positive, per relation and per relation group.

**S8's gate** is met when the promoted set has at least 20 graded hypotheses and precision at least 0.9.

S15's test half is never read.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| P1 | S7, S7b, S7c dev: `decide`'s supported set shrinks by under 2% in every read, precision moves by under 0.005, and refuted counts do not change. (T4's and T5's negatives have Bayes factors near 0.01, so a posterior stays above 0.9 after one only from a prior near 0.99.) |
| P2 | S8a's and S8b's own dev loops: supported changes by under 2% |
| P3 | S8c's read 3: the false promotions without any positive source result drop out. The two with a WP-LEAD positive stay. Graded falls from 15 to 10–12, precision 0.80–0.85, and the gate still fails |
| P4 | Calibration: DB-IB passes the bar in every group it is measured in. WP-LEAD fails in every group (LR+ under 5). GN fails (too few positives) |
| P5 | The fresh set has at least 1,000 hypotheses, and the confirm half grades at least 40% of them |
| P6 | Fresh read: at least 20 graded promotions, precision at least 0.9 (Wilson lower bound at least 0.8). S8's gate is met |
| P7 | At least 90% of the fresh promotions rest on a DB-IB positive |
| P8 | Of the relation groups, places has the lowest precision |

## Decision rule (registered)

- If the fresh read meets S8's gate: propose a ledger row, decided by measurement under row 55 and open to Odin's veto. Supported needs a positive independent result. A source can carry a promotion only where its LR+ on the generators' graded errors is at least 10.
- If it does not: report the gate as not met, with the counts and where the false promotions come from. No rule is changed after the read.

## Results before the fresh read (dev)

### Part 1: the fix, before and after

Same commands, main `aca60a5` against this branch. Every line not listed is unchanged. (The runs of main's binary started while the registration was being written; their output was first read after it was committed.)

| Read (dev) | Before | After |
|---|---|---|
| S7's loop, add-one: supported by `decide` | 343 | 342 |
| S7's loop, point estimates: supported by `decide` / kept by BH | 343 / 54 (all true) | 342 / 54 (all true) |
| S7b, three-way, B1 among supported | 572, 0.974 [0.957, 0.984] | 571, 0.975 [0.959, 0.985] |
| S7b, three-way, no batch step | 573, 0.974 | 572, 0.976 |
| S7b, two-way, B1 among supported | 237, 0.992 | 236, 0.996 |
| S7b, refuted (three-way / two-way) | 113 (0.965) / 103 (0.961) | unchanged |
| S7b, S15's choose half (reachable / all) | 901 at 0.949 / 2,119 at 0.971 | unchanged |
| S7c, K4 (row 70's Known) | 573, 0.974 [0.957, 0.984]; Known-decided 0.2325 | 572, 0.976 [0.959, 0.985]; 0.2322 |
| S7c, K1 / K2 | 215 (0.995) / 37 (1.000) | unchanged; K1's Believed 358 → 357 |
| S7c, simulation with planted nulls: K4's FDR | 0.090 ± 0.014 | 0.098 ± 0.015 |
| S7c, simulation with planted nulls: B1 promoted per batch (all false) | 34.1 | 34.4 |
| S7c, resampling of dev: K4's FDR | 0.026 | 0.025 |
| S8a R1 and R2, S8b PROMO and S15 | — | every output file byte-identical |
| **S8c read 3** (confirm half, read before) | **314 supported; 15 graded, 0.667 [0.417, 0.848]** | **309 supported; 12 graded, 0.750 [0.468, 0.911]** |

- The fix changes almost nothing in S7, S7b and S7c. Their independent tests are T5 and T4, and a negative from either has a Bayes factor near 0.01. So "supported after a negative" needs a prior near 0.99.
- S8a and S8b do not change at all. Their only independent test is T4.
- **S8c's read 3 changes.** Three promotions drop: India–Mumbai and Connecticut–Bridgeport (false; DB-IB and GN negative, nothing positive) and Hanover–Germany (true; only GN negative). Three false ones stay, each carried by a WP-LEAD positive: Florida–Jacksonville (from R1 and from R2) and China–Shanghai.

### Part 2: calibration on the generators' errors (S15's choose half)

327 distinct dev hypotheses are graded on the choose half; 91 (0.278) are true.

| Source, group | True: pos / neg / inc | False: pos / neg / inc | Cell used | LR+ calibrated (read 1) | Carries a promotion |
|---|---|---|---|---:|---|
| DB-IB, dates | 1 / 0 / 1 | 0 / 71 / 48 | pooled | 22.9 (680) | **yes** |
| DB-IB, places | 5 / 3 / 15 | 0 / 33 / 54 | own | 20.8 (387) | **yes** |
| DB-IB, links | 0 / 0 / 2 | 0 / 0 / 1 | pooled | 22.9 (210) | **yes** |
| DB-IB, lexical | 11 / 0 / 53 | 1 / 1 / 27 | own | 2.9 (102) | no |
| DB-IB, pooled | 17 / 3 / 71 | 1 / 105 / 130 | — | 22.9 | — |
| WP-LEAD, places | 9 / 0 / 14 | 7 / 0 / 80 | own | 4.3 (24) | no |
| WP-LEAD, lexical | 10 / 0 / 54 | 0 / 0 / 29 | own | 5.3 (28) | no |
| WP-LEAD, dates and links | pooled | pooled | pooled | 3.4 (23, 60) | no |
| GN, places | 2 / 1 / 20 | 0 / 1 / 86 | own | 10.4 (17) | **yes** (2 positives) |

- **WP-LEAD fails everywhere.** Its positive was right 19 times in 33 here. On the dates it was positive on 7 false values and no true one.
- **DB-IB fails on lexical relations.** It was right 11 times in 12 when positive, but it was positive on only 11 of 64 true occupations and genres. With 29 false items, add-one puts its false-positive rate at 2 in 32. Its likelihood ratio, 2.9, is under the bar. The registered rule is applied as written.
- **GN passes on two positives.** Its LR+ is 10.4 against a bar of 10. Its negative's factor is 3.5 (one negative among 23 true, one among 87 false). That is a small-sample artefact: a negative now raises the posterior a little. A negative can never carry a promotion, so it cannot promote anything by itself.
- **S8c's dev set again** (the confirm half was read before; nothing is chosen on it): 60 distinct hypotheses supported, and only 1 is graded (true). The occupation promotions are gone (DB-IB's lexical cell no longer carries), and so are WP-LEAD's.

### The fresh set, fixed before the read

| | |
|---|---|
| S15 dev's confirm-half slots | 3,442 |
| Store facts hidden | 4,979 (reduced store 17,804 of 22,783) |
| Slots whose subject still has facts | 1,330 |
| G2 (rules, noisy-or) | 51 slots with a rule candidate; **68 hypotheses**. Its loop: 24 supported, 44 queued |
| G6/G7 (S8b's choice on dev: the combination) | **1,330 hypotheses**, one per slot. Its loop: 36 supported, 957 refuted |
| All | 1,398 hypotheses |

No S15 label was read to make them. The files and their SHA-256s:
- `s8a.fresh_g2.hyps.tsv`: `4d829a5e72c9dbc6097414ab04c36cba1e8ab5891566833c8d20d320c088fefd`
- `s8b.fresh_links.hyps.tsv`: `8fb06e748a05cf2ae834c54d24e78732a6a8ea3bc1c6b97ef84695e244b3f20e`

A second run of both commands wrote both files again byte for byte. This commit fixes the set; the read follows it, once.

## The fresh read (once, after commit `d544e5e`)

1,398 hypotheses, 1,364 distinct; S15's confirm half grades 1,075 of them (79%). Every subject is in the confirm half.

| Promoted (supported by `decide`) | Graded | Precision [Wilson 95%] |
|---|---:|---|
| G2 | 11 | 10 / 11 = 0.909 [0.623, 0.984] |
| G6/G7 | 16 | 14 / 16 = 0.875 [0.640, 0.965] |
| **All, each distinct once** | **25** | **22 / 25 = 0.880 [0.700, 0.958]** |
| Row 70's Known tier (K4 on the source tests) | 0 | — |

**S8's gate: not met.** 25 graded promotions (at least 20 needed), precision 0.880 (0.9 needed).

| Positive from | Graded | Precision |
|---|---:|---|
| DB-IB | 22 | 20 / 22 = 0.909 [0.722, 0.975] |
| DB-IB and GN | 1 | 1 / 1 |
| GN alone | 2 | 1 / 2 = 0.500 [0.095, 0.905] |

| Relation (all are places) | Graded | Precision |
|---|---:|---|
| capital | 10 | 9 / 10 = 0.900 [0.596, 0.982] |
| death_place | 9 | 8 / 9 = 0.889 [0.565, 0.980] |
| country | 3 | 3 / 3 = 1.000 [0.439, 1.000] |
| birth_place | 3 | 2 / 3 = 0.667 [0.208, 0.939] |

- Nothing outside places was promoted. Dates: 564 of 583 refuted, 99.8% of the graded ones false. Links: 4 of 150 graded. Lexical: DB-IB does not carry there.
- Refuted in all: 1,007, of which 783 graded, 91.1% false.

**The three false promotions** name the place at another grain, or by another name:
- Belgium, capital Brussels (G2; GN positive, DB-IB negative). S15 holds "City of Brussels" (Q239). Wikidata lists "Brussels" among that item's names.
- Bill Bixby, death place Los Angeles (DB-IB positive). S15 holds Century City, a district of Los Angeles.
- Hillary Rodham Clinton, birth place Chicago (DB-IB positive). S15 holds Edgewater Hospital, in Chicago.

S8c met the same kind (Swinburne: London against Putney). S15's grader is closed-world and matches names, not places inside places. By the registered grader the three are false, and the gate is read that way. (Post hoc, not a decision: counted true, the read would be 25 / 25.)

**Why Known is empty.** A hypothesis carried by one DB-IB positive has a null-model p-value of about P(positive | false) on the calibration set: 1 / 90 for places (add-one over 87 false items). BH at 0.1 over 1,398 hypotheses needs p ≤ 0.0018 for 25 rejections. A source positive reaches Known only with calibration sets of thousands of graded generator errors, or with much smaller batches.

## Predictions, scored

| # | Result |
|---|---|
| P1 | **Yes.** At most one hypothesis moves in any S7, S7b or S7c read; precision moves by at most 0.004; refuted counts unchanged |
| P2 | **Yes.** S8a's and S8b's outputs are byte-identical |
| P3 | **Partly.** Graded 12 (predicted 10–12), but precision 0.750 (predicted 0.80–0.85): three WP-LEAD-carried false ones stayed, not two (Florida–Jacksonville comes from R1 and from R2) |
| P4 | **No.** WP-LEAD fails everywhere, as predicted (but lexical's LR+ is 5.3, not under 5). DB-IB fails on lexical relations (2.9). GN passes (10.4) |
| P5 | **Yes.** 1,398 hypotheses; 79% graded |
| P6 | **No.** 25 graded, 0.880 [0.700, 0.958]. The gate is not met |
| P7 | **Yes.** 23 of 25 (92%) rest on a DB-IB positive |
| P8 | **Not testable.** Every promotion is a place |

## Decisions

- **`decide` needs a positive independent result.** A bug fix to match DISCOVERY §7.2. It changes one hypothesis in S7, S7b and S7c, nothing in S8a and S8b, and removes S8c's three negative-only promotions.
- **A source carries a promotion only where its LR+ on the generators' graded errors is at least 10.** On this calibration: DB-IB for dates, places and links; GN for places. Not WP-LEAD, and not DB-IB for lexical relations.
- **S8's gate is not met** on the fresh set: 25 graded, 0.880 [0.700, 0.958].

## For Odin

1. **The gate.** It fails by the registered grader, and all three misses are S15 grain or naming artefacts. The options:
   - record the failure;
   - register a grader that takes Wikidata's names and places inside places, then read a new fresh set (S15's test half stays reserved for the generators' final reads, so it needs new slots);
   - read the gate on another tier.
2. **Known is out of reach for source evidence at this calibration size.** Should S8's gate be read on Supported (as S8c and here) or on Known?
3. **GN carries promotions on two calibration positives** (LR+ 10.4 against a bar of 10). Its two GN-only fresh promotions gave one true and one artefact. Require at least 10 graded positives before a source may carry? That would need registering before another read.
4. **DB-IB on lexical relations** fails the bar with 11 of 12 positives right, because only 29 false lexical items were graded. More graded generator errors on occupations and genres would settle it.

## How to reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
F=/root/work/f2; E=/root/cdata/enwik8
# the dev reads (before: the same at main aca60a5)
./pattern_learn --test-kinds $E $F/v2/aug/seeds.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv out/s7 \
  --phi-wd $F/wikidata/phi.tsv --s15 bench/eval/temporal_holdout_dev.tsv --s7c
./pattern_learn --gen-rules $E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval out/s8a
./pattern_learn --gen-links $E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval out/s8b
S="--sources bench/eval/temporal_holdout_dev.tsv tables $F/wikidata/wd_values.tsv $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv"
./pattern_learn $S out/s8c out/s8a.r1_dev.hyps.tsv out/s8a.r2_dev.hyps.tsv out/s8b.promo.tsv              # S8c's read 3
./pattern_learn $S out/cal out/s8a.r1_dev.hyps.tsv out/s8a.r2_dev.hyps.tsv out/s8b.promo.tsv --calibrate  # calibration
# the fresh set (the same two commands with --fresh-confirm), then its single read
./pattern_learn --gen-rules … bench/eval fresh/s8a --fresh-confirm
./pattern_learn --gen-links … bench/eval fresh/s8b --fresh-confirm
./pattern_learn $S read/s8c out/s8a.r1_dev.hyps.tsv out/s8a.r2_dev.hyps.tsv out/s8b.promo.tsv \
  --fresh fresh/s8a.fresh_g2.hyps.tsv fresh/s8b.fresh_links.hyps.tsv
```

- `tables` is S8c's (see its report). The S8d lines start with `S8D`. `*.s8d_fresh.tsv` lists every fresh hypothesis with each source's outcome, both posteriors, its e-value, the tier and the grade.
- Runs: each dev read takes 1–2 minutes; the fresh read under 2 seconds.
- **Checks:** `native_epistemic_test` passes 1,029 checks (`test_decide_S8d`: a negative-only independent result never supports, at posterior 0.995; negative then positive does; a generator's own source and a source under the bar never carry; an inconclusive one never supports; the null model's walk goes on after a negative; the calibration helpers). The other 8 CTests of HANDOFF §4.4 pass. The normal `pattern_learn … --gaps` run is unchanged: all 10 files byte-identical (7 against `/root/work/f2/v2/aug`, 3 against `/root/work/trackS`).
- **S15's test half was not read.**
