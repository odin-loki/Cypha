# S10: the curiosity scheduler

**Odin Loch** · 2026-10-29 · DISCOVERY §9, §4.3, §8 P2, §13 S10 · follows [S8D_PROMOTION.md](S8D_PROMOTION.md)

(The design, the rules and the predictions below were written before any S10 run. Results are added under them.)

**Rule 10.** The scheduler only decides which of Cypha's own hypotheses get tested next. It never answers a question.

**Gate (DISCOVERY §13).** The scheduler beats a uniform scheduler on confirmed value per cost, via LCI (≥ 0.95, PLAN §8), without raising the false-discovery rate (DISCOVERY §9).

## What there is to schedule

Track S has one discovery process with recorded outcomes: **P2, conjecture** (generate, score, test, promote or refute). Two generators feed it:
- **G2**, rule induction (S8a);
- **G6/G7**, link prediction (S8b).

Every hypothesis goes through the same pipeline, the one S8d registered (PLAN ledger 71):
1. its generator's own loop (T1, T6, T4, cheapest first, until it is no longer queued);
2. the source stage: the calibrated sources (DBpedia infobox, Wikipedia lead, GeoNames) as further tests, from the generator's posterior, until it is no longer queued;
3. `decide` (a positive independent result is needed to support).

The scheduler decides which hypothesis enters this pipeline next. A pipeline run is one **pull**.

## Definitions (fixed before any run)

- **Arm.** DISCOVERY §9 makes an arm a (process, generator, tier). With one process, an arm here is **(generator, relation group, tier)**:
  - generator: G2 or G6/G7;
  - relation group: S8c's groups (dates, places, links, lexical), the slot region;
  - tier: DISCOVERY §4.3 from the hypothesis's P0 (conjecture ≥ 0.3, speculation ≥ 0.05, wild below).
- **Cost of a pull.** The number of tests the pipeline ran on that hypothesis: the generator loop's results (counted in its pre-registration log) plus the source tests run. Every test costs 1.
- **Impact.** S6's `hypothesis_impact`: 1 for the gap it fills (its slot was hidden from the store), plus its slot's demand in S3's gap map.
- **Certainty gained.** max(0, posterior after the source stage − P0).
- **Reward of a pull** (what the scheduler sees, with no grade): impact × certainty gained if `decide` supports the hypothesis, else 0. (DISCOVERY §9: the value confirmed per unit cost. Supported is used, not Known: S8d found the Known tier empty at this calibration size.)
- **Confirmed value** (what the gate reads; the scheduler never sees it): the sum over the supported hypotheses that the grader marks true of impact × certainty gained, minus the same sum over those it marks false. An ungraded promotion adds nothing.
- **V, confirmed value per cost** = confirmed value / cost spent, reported × 1,000.
- **FDR** = graded false promotions / graded promotions.

## The schedulers

**U, the uniform scheduler (the baseline).** Each pull picks an arm uniformly at random among the arms with hypotheses left. Within an arm, hypotheses come in a random order.

**S10, the curiosity scheduler** (`native/include/cypha/eval/curiosity.hpp`):
- **Thompson sampling.** Each arm keeps a Beta(1 + s, 1 + f) posterior of its reward rate, with fractional counts: a pull adds r / R to s and 1 − r / R to f, where R is the largest possible reward. Its expected cost is (cost spent + c₀) / (pulls + 1), with c₀ the mean cost of all pulls so far (1 before the first). Each pull draws p from every arm's Beta. The arm with the largest p · R / expected cost × (1 + λ · LP) is pulled.
- **Learning progress (LP).** Over an arm's last 10 pulls, the mean of max(0, H(P0) − H(posterior)) / cost: bits of uncertainty removed per unit cost. An arm whose tests decide its hypotheses gets the bonus; an arm whose hypotheses stay where they were loses it (the "noisy TV").
- **Demand bonus.** Within an arm, hypotheses on slots with demand go first.
- **Floors.** Once 50 cost units are spent: each generator keeps at least 2% of the cost spent, and the wild tier at least 5%. A generator or tier under its floor gets the next pull (its Thompson-best arm).
- **Retirement.** An arm whose promotion rate has a Wilson 95% upper bound below 0.05 after at least 200 pulls is retired: never pulled again, and out of the floors.
- The scheduler's random numbers are its own (splitmix64 and Marsaglia–Tsang gamma draws), so every platform draws the same.

## The replay

Pulls are replayed on recorded pools. Each hypothesis's outcome (its tests, its posterior, its state and its grade) is fixed and does not depend on what else was tested: a promotion is not fed back into the generators during the replay. So drawing each arm's hypotheses without replacement, in an order that does not look at outcomes, gives each policy the outcomes it would have had. (The limit: in Cypha a promotion would change the store, and later hypotheses with it.)

- **Budget.** A run stops once its cost reaches f × the pool's total cost (the cost of running every hypothesis), or when no hypothesis is left. Primary: **f = 0.25**. Also reported: 0.1, 0.5 and 1.0 (at 1.0 both process everything, so they must tie: a check, not a result).
- **Common random numbers.** For each draw and seed, every arm's random order is fixed, and both policies take hypotheses from the same orders.
- **One draw** = a stratified bootstrap of the pool (each arm resampled with replacement to its own size) and **5 seeds**; each policy's V, FDR and Brier score are pooled over the 5 seeds.

## Choices on dev

**The dev pool** is S8d's fresh set (1,398 hypotheses on S15 dev's confirm-half slots), graded by S15 dev's confirm half. It was read once for S8d's gate. Reading it again here is dev use: nothing on it is a gate.

Two choices are made on dev, by mean V over 1,000 draws at f = 0.25, among four configurations:
- within-arm order: random, or **P0 first** (highest P0 first; S6's plausibility as the priority);
- LP weight λ: 0 or 1.

A configuration is chosen only if its mean V beats the other's by more than 2%; otherwise the simpler one stays (random order, λ = 0). Every other setting above is fixed and not tuned.

## The gate set (fresh, built the way S8d built its set)

S15's test half has been read: by S8b and S8a, each once, for their final reads (S15's Reads table). S15's dev halves have both been read. So the gate needs a fresh set:
- **The slots.** Every (subject, relation) of `bench/eval/temporal_holdout_test.tsv` with bit 17 of `pattern_learn`'s `fnv(subject)` = 1 and a relation of `pattern_learn`'s. Only subjects and relations are read to make it. The other half of the test file (bit 17 = 0) is not touched by S10.
- **Hidden facts.** Every store triple in those slots is removed, and G2 and G6/G7 propose on the reduced store exactly as for S8d's set (`--fresh-test-half`, the same code as `--fresh-confirm` on another slot list).
- **The source stage** runs on the set with S8d's calibration (S15 dev's choose half), as for S8d's set.
- The pool files' SHA-256s are added to this report and committed **before the grade is read**.
- **The read (once).** The scheduler configuration chosen on dev, against U, graded by the test file's rows for those subjects with S8c's grader. It is logged as a read of S15 test (for Odin: S15's freezing rule names generators' final reads; S10's gate is a final read of the scheduler that allocates the generators' work).

## LCI on the gate (PLAN §8.2, fixed before the read)

4,000 draws (as above). Gain Δ = V_S10 − V_U per draw.

| Part | Here |
|---|---|
| P_gain | share of draws with V_S10 > 1.10 × V_U (ε: a 10% gain in confirmed value per cost) |
| n_eff | distinct (subject, relation) slots in the gate pool; below 200, LCI is capped at 0.5 |
| r | non-inferiority of the FDR: r = 1 if FDR_S10 ≤ FDR_U + 0.05 in at least 95% of the draws that have graded promotions under both, else 0. (δ is 5 points, not PLAN's 1: a run promotes a few dozen graded hypotheses, so 1 point is below what it can resolve) |
| γ | G = (gain on the gate) / (gain on dev), each gain = mean V_S10 / mean V_U − 1; γ = min(1, G / 0.5). If the dev gain is not positive, γ = 1 when the gate gain is positive, else 0 |
| c | 1 if the mean Brier score of the graded promotions (posterior against grade) rises by no more than 0.01, else 0.5 |
| σ | 1 if the mean Δ has the same sign in the three blocks of draws (1–1,333, 1,334–2,666, 2,667–4,000), else 0.5 |
| κ | 1 if a scheduling decision takes under 1 ms on average and the scheduler's state is under 1 MB, else 0 |

**LCI = P_gain × r × γ × c × σ × κ.** The gate is met if LCI ≥ 0.95. Only the one chosen configuration is read on the gate, so there is no multiplicity across candidates (PLAN §8.4).

## Decision rule (registered)

- Gate met: the scheduler is adopted as P2's allocator (DISCOVERY §9), decided by measurement under PLAN ledger 55 and open to Odin's veto.
- Gate not met: reported with its LCI parts; nothing is changed after the read.

## Predictions (registered before any run)

| # | Prediction |
|---|---|
| Q1 | The gate pool has 1,000–2,000 hypotheses, at least 75% from G6/G7, and at least 200 distinct slots |
| Q2 | On dev, S10 (whichever configuration is chosen) beats U by at least 30% in V at f = 0.25 |
| Q3 | On the gate, LCI ≥ 0.95: the gate is met |
| Q4 | On the gate, S10's FDR is within 0.05 of U's (pooled over all draws) |
| Q5 | The LP bonus moves dev V by less than 5% (on a fixed pool it has little to find) |
| Q6 | P0-first order raises dev V by at least 10% over random order |
| Q7 | No arm is retired at f = 0.25; at f = 1.0, at least one G6/G7 dates arm is retired |
| Q8 | At least 90% of S10's promotions on the gate are places |
| Q9 | At f = 1.0 with retirement off, S10 and U process the same hypotheses and tie exactly (a check of the replay) |

## Dev, the registered design (2026-10-29)

The dev pool, with each hypothesis's cost from S8d's own logs (S8d's source stage reruns byte for byte with this branch's binary):

| Arm | Hypotheses | Cost | Supported (graded, true) |
|---|---:|---:|---|
| G2 places conjecture | 19 | 96 | 11 (11, 10) |
| G6/G7 places conjecture | 154 | 878 | 14 (14, 12) |
| G6/G7 places speculation | 134 | 712 | 2 (2, 2) |
| 12 other arms (G2 lexical and links; G6/G7 dates, lexical and links in every tier, places wild) | 1,091 | 2,848 | 0 |
| All | 1,398 | 4,534 (3.24 tests each) | 27 |

At f = 0.25, 1,000 draws of 5 seeds:

| Configuration | V, U | V, S10 | Gain | FDR U / S10 |
|---|---:|---:|---:|---|
| random order, λ 0 | 2.790 | 2.568 | −0.079 | 0.101 / 0.100 |
| random order, λ 1 | 2.790 | 2.580 | −0.075 | 0.101 / 0.100 |
| P0 first, λ 0 | 2.790 | 2.519 | −0.097 | 0.101 / 0.134 |
| P0 first, λ 1 | 2.790 | 2.542 | −0.089 | 0.101 / 0.133 |

**The registered scheduler loses to the uniform one.** Its pulls show why: it gives the four wild arms with no promotion 39–41 pulls each, against U's 26, and the productive G6/G7 places conjecture arm 20 against 26. Two faults:
- **The reward scale.** R, the largest possible reward, is 11: one slot in the pool has demand 10 in S3's gap map. A promotion worth 0.6 then adds 0.05 to an arm's Beta, so no arm's posterior moves.
- **The prior.** Beta(1, 1) has mean 0.5. Promotion rates here are about 2%. An arm with no promotion after 20 pulls still draws about 1 / 22, more than the productive arms' real rate, and the cost term then favours the cheap arms. The wild arms' hypotheses cost 2.1 tests; the places conjecture arm's cost 5.7 (they survive their generator loop and reach more sources).

Other dev reads (the registered design): P0-first order raises promotions (75 against 66 per draw) but lowers V, because a likely hypothesis gains less certainty and its FDR is higher. At f = 1.0 with retirement off, U and S10 tie exactly in 200 of 200 draws (Q9's check). At f = 1.0 with retirement, G6/G7 dates wild and lexical wild retire in every run.

## Amendment 1 (on dev, before the gate set exists)

The Thompson draw is changed; nothing else is (arms, cost, value, floors, retirement, LP, the dev choices and their rule, the gate, LCI and the predictions stay as registered):
- **Success and size apart.** An arm's promotion probability q gets a Beta posterior on its promotions (a pull is a success when `decide` supports it). The size of a promotion's reward gets its own mean, m = (the arm's reward + m₀) / (its promotions + 1), with m₀ the mean reward per promotion over all arms so far (0.5 before the first). A large reward on a demanded slot then raises that arm's m, and no longer flattens everyone's Beta.
- **The prior from the pool.** Beta(2ρ, 2(1 − ρ)), with ρ = (all promotions + 1) / (all pulls + 2): every arm starts at the rate seen so far, not at 0.5 (an empirical-Bayes prior of strength 2).
- **The score** = q drawn × m / expected cost × (1 + λ LP).

## Dev with amendment 1

At f = 0.25, 1,000 draws of 5 seeds (the same draws as above):

| Configuration | V, U | V, S10 | Gain | V_S10 > 1.1 V_U | FDR U / S10 |
|---|---:|---:|---:|---:|---|
| random order, λ 0 | 2.790 | 4.224 | +0.514 | 0.934 | 0.101 / 0.110 |
| random order, λ 1 | 2.790 | 4.244 | +0.521 | 0.929 | 0.101 / 0.110 |
| **P0 first, λ 0** | 2.790 | **6.308** | **+1.261** | 0.978 | 0.101 / 0.093 |
| P0 first, λ 1 | 2.790 | 6.324 | +1.266 | 0.980 | 0.101 / 0.094 |

- **The registered rule chooses P0 first, λ 0.** The order is compared first, by the better of its two λ (P0 first beats random by 49%). Then λ within that order: λ 1 is only 0.3% better, inside the 2% margin, so λ 0 stays.
- S10 now gives the productive G6/G7 places conjecture arm 105 pulls per run (U: 26), and the wild arms 13–16 (U: 26).
- Other budgets (P0 first, λ 0): f = 0.1 gain +1.130, f = 0.5 +0.414, f = 1.0 +0.184 (the retired arms' cost is saved). With retirement off, f = 1.0 ties exactly in 200 of 200 draws.
- For reference, one arm with a random order over all hypotheses gives V 2.016 at f = 0.25.

## The gate set (fixed before the read)

Built by `--gen-rules … --fresh-test-half` and `--gen-links … --fresh-test-half`, then `--sources … --fresh` (S8d's calibration). Checks:
- Two runs wrote all four generator files byte for byte.
- The generators' dev outputs in the same runs equal S8d's (`s8a.r1_dev`, `s8a.r2_dev`, `s8b.promo`, `s8b.s15` and their logs).
- `--fresh-confirm` with this branch's binary still writes S8d's fresh set byte for byte.
- The `--sources` run prints S8d's FRESH and GATE lines. They do not apply here: they grade on S15 dev's confirm half, and no gate-set subject is in it.

| | |
|---|---|
| S15 test's bit-17 slots | 3,611 |
| Store facts hidden | 5,288 (reduced store 17,495) |
| Slots whose subject still has facts | 1,376 |
| G2 | 63 hypotheses |
| G6/G7 | 1,376 hypotheses |
| All | 1,439 hypotheses, 1,376 distinct slots, cost 4,793 (3.33 tests each) |
| Supported by the pipeline (no grade read) | 12: G2 places conjecture 8, G6/G7 places speculation 2, G6/G7 places conjecture 1, G6/G7 links conjecture 1 |

The pool was loaded once before the read with S15 dev as its grader (no gate subject is in S15 dev, so nothing was graded), to check that it loads. Nothing was changed after that.

SHA-256 of the files (`/root/work/s10/runs/gate/`):
- `s8a.fresh_g2.hyps.tsv`: `c1c9ac9951494a08966a154721434915f8d14cac027f31edf688eb499f297d57`
- `s8a.fresh_g2.prereg.tsv`: `4f2bd6054022394fa6bcc0ac189d36373fecc82707c6bbf630920e48190bc084`
- `s8b.fresh_links.hyps.tsv`: `0121a6ef9b9cc0d053b1b63a4c97bab19379f757ad68608ccecbb6160313a325`
- `s8b.fresh_links.prereg.tsv`: `ba1dd20bd2795a274717c1b43820fad7640cd7e5f0512c180adcd7e6be211bba`
- `src.s8d_fresh.tsv`: `9c263f8357d36eebfa6c7960eac325ef78e6c2d04b4ae52343700cd9099a9934`
- `src.s8d_fresh_ran.tsv`: `40cfa825bc83397047b5685436e353614962890378789b211016fbb794ec3531`

This commit fixes the set and the configuration (P0 first, λ 0). The read follows it, once.

## The gate read (once, after commit `dc2e632`)

S10 with P0 first and λ 0, against U, at f = 0.25: 4,000 draws of 5 seeds. The gate set's grades: 1,116 of 1,439 hypotheses graded; all 12 supported ones graded, and all 12 true.

| | U | S10 |
|---|---:|---:|
| V, confirmed value per 1,000 cost, mean [95% of draws] | 2.227 [1.254, 3.296] | 2.105 [1.094, 3.357] |
| Promotions per run | 9.31 | 9.01 |
| FDR | 0.000 | 0.000 |

Gain V_S10 − V_U: mean −0.121, 95% of draws in [−0.885, +0.534]; relative gain median −0.044 [−0.380, +0.245].

| LCI part | Value |
|---|---|
| P_gain (V_S10 > 1.1 V_U) | 0.127 |
| n_eff | 1,376 slots (no cap) |
| r | 1 (FDR_S10 ≤ FDR_U + 0.05 in all 4,000 draws) |
| γ | 0 (gain on the gate −0.054, on dev +1.261) |
| c | 1 (Brier rise 0.000) |
| σ | 1 (the three blocks: −0.118, −0.124, −0.121: stably worse) |
| κ | 1 (2.7 µs per pull; state 5 KB) |
| **LCI** | **0.000** |

**S10's gate is not met.**

Where the pulls went (per run):

| Arm | Hypotheses (supported) | U pulls | S10 pulls | U promotions | S10 promotions |
|---|---|---:|---:|---:|---:|
| G2 places conjecture | 23 (8) | 22.4 | 21.5 | 7.77 | 7.15 |
| G6/G7 places conjecture | 101 (1) | 27.0 | 46.4 | 0.26 | 0.85 |
| G6/G7 places speculation | 197 (2) | 26.9 | 19.7 | 0.27 | 0.00 |
| G6/G7 links conjecture | 1 (1) | 1.0 | 1.0 | 1.00 | 1.00 |
| 11 arms with no promotion | 1,117 (0) | 213.7 | 191.9 | 0 | 0 |

Other budgets (1,000 draws): f = 0.1 gain −0.417; f = 0.5 +0.152; f = 1.0 +0.195 (S10 retires G6/G7 dates wild and lexical wild in every run, and saves their cost). No arm is retired at f = 0.25.

### Why it fails

- **The gate set's value sits in one small arm.** 8 of its 12 promotions are in G2 places conjecture, 23 hypotheses. At f = 0.25 the uniform scheduler also takes 22.4 of them. The other 4 promotions are spread over arms of 1–200 hypotheses at rates of 1–2%, which about 350 pulls cannot find reliably. Dev's set had a large productive arm (G6/G7 places conjecture, 14 promotions in 154); this one does not.
- **P0 first, chosen on dev (+49% there), reverses here.** In G2 places conjecture the ten hypotheses with the highest P0 (0.923) are all unsupported; the eight promoted ones have P0 0.745. At f = 0.1, with the same 9 pulls in that arm, S10 promotes 1.0 and U 3.2.

### Post hoc (not a decision)

These were computed on the gate set after its read, to explain it. Nothing is chosen on them.
- **Random order instead of P0 first** (all four configurations on the gate set, 1,000 draws): f = 0.25 gain +0.035, P(V_S10 > 1.1 V_U) 0.137; f = 0.1 gain +0.810 (0.987); f = 0.5 +0.063.
- **The ceiling.** Taking all 12 promotions first and then anything gives V 3.703 at f = 0.25, +66% over U. No scheduler could pass the 10% margin by much on this set.

## Predictions, scored

| # | Prediction | Result |
|---|---|---|
| Q1 | Gate pool 1,000–2,000 hypotheses, ≥ 75% G6/G7, ≥ 200 slots | **Hit:** 1,439, 95.6%, 1,376 |
| Q2 | Dev gain ≥ 30% at f = 0.25 | **Miss as registered** (−7.9%); hit after amendment 1 (+126%) |
| Q3 | Gate met | **Miss:** LCI 0.000 |
| Q4 | Gate FDR within 0.05 of U's | **Hit:** 0.000 and 0.000 |
| Q5 | LP moves dev V by < 5% | **Hit:** 0.5% (registered), 0.3% (amended) |
| Q6 | P0 first raises dev V by ≥ 10% | **Miss as registered** (−1.9%); hit after amendment 1 (+49%); on the gate it hurt |
| Q7 | No retirement at f = 0.25; a G6/G7 dates arm retires at f = 1.0 | **Hit** (dates wild, on dev and gate) |
| Q8 | ≥ 90% of S10's gate promotions are places | **Miss by a point:** 88.9% (the one links promotion is taken in every run) |
| Q9 | U and S10 tie at f = 1.0 with retirement off | **Hit:** 200 of 200 dev draws |

## Decision

- **S10's gate is not met** (LCI 0.000): the scheduler is not adopted as P2's allocator. Nothing was changed after the read.
- `curiosity.hpp` and `pattern_learn --schedule` stay as the measuring tools for the next read. The uniform scheduler stays the default.
- Amendment 1 (made on dev, before the gate set existed) is the scheduler's draw: the registered fractional Beta with R as the largest reward could not learn when one demanded slot sets R.

## For Odin

1. **The gate needs a richer pool.** A scheduler can only beat uniform where value is spread over large arms with different rates. This gate set has 12 promotions in 1,439 hypotheses, and two thirds of them in one 23-hypothesis arm that uniform also exhausts. Read S10 again when the generators give larger pools (after S8's grader question, or with S11's generators), with the budget fixed in advance where arms are not exhausted.
2. **Within-arm order.** P0 first won on dev and lost on the gate. A priority order needs a measured link between P0 and promotion in each arm. Until then, random order within an arm is the safer default.
3. **S15's freezing rule.** This read used S15 test's bit-17 half: subjects and relations to build the set, and the labels once, for the gate. S15's rule names generators' final reads; this was the scheduler's final read. Log it in S15's Reads table, or say if such reads should come from elsewhere. The bit-17 = 0 half of the test file was not touched.

## How to reproduce

```bash
g++ -O2 -std=c++20 -I native/include native/tools/pattern_learn.cpp -o pattern_learn
F=/root/work/f2; E=/root/cdata/enwik8; S8D=/root/work/s8d/runs   # S8d's dev files (after2) and fresh set (fresh)
S="--sources bench/eval/temporal_holdout_dev.tsv tables $F/wikidata/wd_values.tsv $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv"
DEVH="$S8D/after2/s8a.r1_dev.hyps.tsv $S8D/after2/s8a.r2_dev.hyps.tsv $S8D/after2/s8b.promo.tsv"
# the dev pool: S8d's fresh set, its source stage rerun (writes src.s8d_fresh_ran.tsv)
./pattern_learn $S dev/src $DEVH --fresh $S8D/fresh/s8a.fresh_g2.hyps.tsv $S8D/fresh/s8b.fresh_links.hyps.tsv
ln -s $S8D/fresh/s8a.fresh_g2.* $S8D/fresh/s8b.fresh_links.* dev/
# the gate pool
./pattern_learn --gen-rules $E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval gate/s8a --fresh-test-half
./pattern_learn --gen-links $E $F/v2/aug/seeds.tsv $F/v2/aug/facts_f2.tsv $F/wikidata/wd_values.tsv $F/v2/aug/patterns.tsv bench/eval gate/s8b --fresh-test-half
./pattern_learn $S gate/src $DEVH --fresh gate/s8a.fresh_g2.hyps.tsv gate/s8b.fresh_links.hyps.tsv
# dev only, then the gate read
./pattern_learn --schedule dev/sched --dev dev bench/eval/temporal_holdout_dev.tsv --gaps gap_map.tsv
./pattern_learn --schedule gate/sched --dev dev bench/eval/temporal_holdout_dev.tsv --gate gate bench/eval/temporal_holdout_test.tsv --gaps gap_map.tsv
```

- `tables` is S8c's; `gap_map.tsv` is S3's (`pattern_learn … --gaps`), for demand. The S10 lines start with `S10`; `gate/sched.gate_draws.tsv` has each draw's reads.
- Runs: each generator run 1–2 minutes; the source stage about 1 minute; `--schedule` 30 s for dev, 90 s with the gate.
- **Checks:** `native_epistemic_test` passes 1,122 checks (`test_curiosity_S10`, 49 new: the random numbers, the value of a pull, Thompson finding the better arm per unit cost, the uniform shares, floors, retirement, learning progress, amendment 1's prior and reward size, the replay's budget, order and bootstrap, and the tie at full budget). Four planted bugs are caught: a score that ignores cost, a floor with its sign flipped, retirement on the lower bound, and a replay that draws with replacement. The other 8 CTests of HANDOFF §4.4 pass. S3's `--gaps` run is byte-identical (gap map, cells, demand, and the v2 store files). `--fresh-confirm` and S8d's source stage reproduce S8d's files byte for byte.
