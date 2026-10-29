# Forecasting: evaluation protocol

**Odin Loch** · 2026-10-29 · [FORECAST_PLAN](../../FORECAST_PLAN.md) FC5–FC9, FC13 · audit: [FC0](../../reports/FC0_FORECAST_AUDIT.md) §1 C, D

(The tasks, splits, scores, baselines and freezing rule below were written before any FC model was built. Results are added under them, in the reports for each item.)

The VIEWS task follows the VIEWS Prediction Challenge 2023/24 exactly ([Hegre et al. 2024](https://arxiv.org/pdf/2407.11045)), so its numbers can be set beside published ones. The other tasks use the same scoring code.

---

## 1. Rules

FORECAST_PLAN §3 applies in full. In particular:
- the origin rule;
- lagged inputs only;
- temporal splits;
- probabilities that are probabilities;
- a baseline beside every number;
- fail closed;
- the true future read once.

Two rules are specific to evaluation:

1. **Choices on dev, the gate on test.** Every setting (features, K, τ, thresholds, isotonic maps) is chosen on dev. The test split is read once per registered change and logged in §10.
2. **Intervals on differences.** A claim that model A beats B rests on the 95% interval of the paired score difference, from a cluster bootstrap over units (countries, dyads or theaters) with 2,000 resamples. A point difference alone is not a result.

---

## 2. Tasks and splits

| Task | Unit | Target | Train | Dev | Test (read once) | Item |
|---|---|---|---|---|---|---|
| E1 Fatalities | GW country-month | UCDP state-based fatalities, counts | Inputs from 1989 to each origin | Backtest years 2018–2020 | Backtest years 2021–2023 | FC7 |
| E2 Escalation | Dyad-year | A: hostility ≥ 4 MID onset in *t* + 1; B: any MID onset | ≤ 1989 | 1990–2001 | 2002–2014 | FC5 |
| E3 Theater events | Theater-week | Any root 18–20 event present (DATA §4.3) within h ∈ {1, 4, 13} weeks | Weeks starting 2013-04-01 to 2019-12-23 | 2019-12-30 to 2021-12-27 | 2022-01-03 to 2024-12-23 | FC8 |
| E4 Alarms | Theater-week | Documented shocks (§6) | as E3 | as E3 | as E3 | FC9 |
| E5 Planted | Synthetic | Known probabilities (ARCHITECTURE §5.4) | — | — | Every CTest run | FC8, FC13 |

E1 follows VIEWS: each backtest year *Y* has its origin at the end of October *Y* − 1. The targets are the 12 months of *Y*, which are 3 to 14 months after the last observed month (VIEWS's s = 3 … 14).

---

## 3. E1: fatalities, country-month (VIEWS protocol)

| | |
|---|---|
| Submission | 1,000 int32 draws per country-month, each draw ≥ 0 |
| Truth | UCDP GED (state-based, best) for 2018–2023; UCDP Candidate for any window GED does not cover yet |
| Scale | Counts, not log counts |
| Main score | Mean CRPS over all country-months of the year (§9.1) |
| Secondary | IGN on VIEWS's 11 bins (§9.2); MIS at 90% (§9.3) |
| Benchmarks | `bm_exactly_zero`; `bm_last_historical` (Poisson, mean = the last observed month); `bm_conflictology_bootstrap240` (draws from the 240 months before the origin, pooled over all countries, the same for every unit); `bm_conflictology_country12` (draws from the country's own last 12 months) |
| Reported | Per year and over 2018–2023: CRPS, IGN, MIS for Cypha and every benchmark; the paired CRPS difference against each benchmark with its interval; the same restricted to country-months with truth > 0 |
| Gate (FC7) | Mean CRPS below `bm_conflictology_bootstrap240` over 2021–2023, with the interval on the difference below 0 |

---

## 4. E2 and E3: probability tasks

| | E2 escalation | E3 theater events |
|---|---|---|
| Output | P(target) per dyad-year | P(any root 18–20 present within h weeks) per theater-week and h |
| Scores | Brier, log loss, AUC-PR, ECE (§9.4) | Brier, Brier skill score, log loss, ECE |
| Baselines | Base rate from train; persistence; logistic regression on the same inputs | Climatology (the theater's train rate for that h); persistence (present this week); first-order Markov chain over the weekly tokens |
| Gate | Dev Brier skill against the base rate > 0, interval above 0; test read once (FC5) | Test BSS against climatology > 0, interval above 0; reported against persistence and Markov (FC8) |
| Check | Base rates per split, reported so drift between splits is visible | Weekly event totals per split; scores also on counts normalised by `gdelt_week_totals.csv` |

---

## 5. E4: alarms

- **Thresholds.** Each monitor signal's threshold is set on dev weeks, excluding the dev shocks below, so that the expected false-alarm rate is 1 per theater-year. It is then fixed.
- **A false alarm** is an alarm in a theater-week more than 7 days from any listed shock in that theater.
- **Detection delay** is the number of days from the shock date to the first alarm in the shock's theater.
- **Reported:** delay per shock; false alarms per theater-year on test.
- **Gate (FC9):** every test shock detected within 7 days, at ≤ 2 false alarms per theater-year.

---

## 6. Shocks and the true future

### 6.1 Documented shocks

| Date | Theater | Event | Split |
|---|---|---|---|
| 2020-01-03 | MID | US strike in Baghdad that killed Qasem Soleimani | Dev (thresholds are checked here, never fitted to it) |
| 2021-05-10 | MID | Start of the May 2021 Israel–Gaza war | Dev |
| 2022-02-24 | UKR | Russia's full-scale invasion of Ukraine | Test |
| 2022-08-04 | TWN | PLA live-fire exercises around Taiwan after the US House Speaker's visit | Test |
| 2023-10-07 | MID | Hamas attack on Israel and the war that followed | Test |
| 2024-04-13 | MID | Iran's drone and missile attack on Israel | Test |
| 2025-06-13 | MID | Israel's strikes on Iran and the war that followed | After test; a prospective-style check, once |

This list is fixed before FC9 runs. Shocks may be added only for windows after the commit that adds them.

### 6.2 Freezing rule for true-future forecasts

1. **Write.** A forecast for a window that has not started is written to `bench/eval/forecast/prospective/<registered-date>_<task>.csv`, with the bundle manifest's sha256 and the code commit in its header.
2. **Commit.** Its sha256 is added to `bench/eval/README.md` and committed before the window's first period begins.
3. **Score once,** as truth arrives. For E1, that is UCDP Candidate month by month. It is rescored against GED when a GED version covers the window. Both scores are kept, each labelled with its data version.
4. **Never re-register** the same window and task after any truth for it has been seen.
5. **Log** every score in §10 and the item's report.

This is also the prospective log that DISCOVERY T8 asks for (FC15).

---

## 7. What a read reports

Each read is a section in the item's report (`docs/reports/FC<n>_…`), in the house order:
1. the design and predictions, registered first;
2. the result tables, with intervals;
3. the predictions, scored;
4. the decision;
5. questions for Odin;
6. how to reproduce.

Every table carries:
- the data version (manifest sha256);
- the bundle manifest sha256;
- the code commit;
- the split;
- the seed.

---

## 8. Planted bugs the CTests must catch (FC13 gate)

| # | Planted bug | Caught by |
|---|---|---|
| 1 | A forecaster reads a row after the origin | `History` has no accessor for it; a test that tries to read one does not compile, and the runtime variant throws |
| 2 | A feature computed at the target period | The panel audit: every feature column carries the period it was computed at; max ≤ origin < target is checked |
| 3 | `validate_views_holdout`-style use of truth inside a forecast | `score` takes no `History`; a planted forecaster that is handed truth through a global fails the leak test (scaling the truth by 10 must not change its draws) |
| 4 | A UTF-16 CSV, a missing column, `abc` in an int column | Loader errors (FC2) |
| 5 | A training set with one class or a constant column | `forecast_data_probe` and the estimator refuse it |
| 6 | An event id ≥ 256 | `static_assert` and an exhaustive vocabulary test |
| 7 | The old binned `crps()` swapped back in | The scoring golden (§9.5) |
| 8 | An exploration bonus added to a reported probability | Tree invariants: children sum to 1, p ∈ [0, 1], weighted estimate unbiased on the planted chain |
| 9 | A monitor fed a constant | The replay backtest: no alarm, or alarms every week, fails |
| 10 | One byte changed in a bundle component | Bundle load error (sha256) |
| 11 | `/forecast/predict` with no bundle loaded | 503 in `rest_schema_contract` |
| 12 | A tool that prints "OK" on empty input | Each forecast tool's CTest feeds it an empty directory and expects a non-zero exit |

---

## 9. Score definitions and reference values

### 9.1 CRPS from samples

For draws x₁ … xₙ and observation y:

  CRPS = (1/n) Σᵢ |xᵢ − y| − (1/(2n²)) Σᵢ Σⱼ |xᵢ − xⱼ|

With the draws sorted ascending, x₍₁₎ ≤ … ≤ x₍ₙ₎, the second term is (1/n²) Σᵢ (2i − n − 1) x₍ᵢ₎. That makes the cost O(n log n). All draws are equally weighted, as in properscoring and xskillscore, which VIEWS uses.

### 9.2 IGN (VIEWS)

1. **Upsample** the draws to 1,000.
2. **Count** them in 11 bins: {0}, 1–2, 3–5, 6–10, 11–25, 26–50, 51–100, 101–250, 251–500, 501–1000, ≥ 1001.
3. **Add one draw to each bin**, giving 1,011.
4. **Score:** IGN = −log₂ (count in y's bin / 1,011).

### 9.3 MIS at 90%

1. l and u are the 0.05 and 0.95 quantiles of the draws, with linear interpolation (NumPy's default).
2. IS = (u − l) + (2/α)(l − y)·1[y < l] + (2/α)(y − u)·1[y > u], with α = 0.1.
3. MIS is the mean of IS over unit-periods.

### 9.4 Probability scores

| Score | Definition |
|---|---|
| Brier | mean (p − o)² |
| Brier skill score | 1 − Brier / Brier of the reference |
| Log loss | −mean [o ln p + (1 − o) ln(1 − p)], with p clipped to [10⁻⁶, 1 − 10⁻⁶]; the number of clipped forecasts is reported |
| ECE | Σ_b (n_b / N) · \|mean p_b − mean o_b\|. 10 equal-width bins is the headline; equal-mass bins are reported beside it |
| AUC-PR | Average precision |
| PIT | Randomised PIT for count draws; 10-bin histogram, reported with its χ² against uniform |

### 9.5 Reference cases (the scoring golden, FC6)

Each case must match to 1e-9.

| # | Score | Forecast | y | Value |
|---|---|---|---|---|
| 1 | CRPS | One draw at 10 | 480 | 470 |
| 2 | CRPS | One draw at 500 | 480 | 20 |
| 3 | CRPS | One draw at 1,000 | 5,000 | 4,000 |
| 4 | CRPS | One draw each at 0, 10, 50, 100, 500, 1000 | 60 | 75.555556 (680/9) |
| 5 | IGN | 1,000 draws at 0 | 0 | 0.014341 (−log₂ 1001/1011) |
| 6 | IGN | 1,000 draws at 0 | 7 | 9.981567 (log₂ 1011) |
| 7 | MIS | Draws 0, 1, …, 999 | 2,000 | 21,918.1 (l = 49.95, u = 949.05) |

- **Cases 1–4** are FC0's D1 cases, where today's `crps()` gives 100, 0, 500 and 27.78.
- **The rest of the fixture** comes from properscoring and NumPy, run offline, with the sidecar recording the versions.

---

## 10. Reads

| Date | Task | Split | Model | Data sha256 | Commit | Result | Report |
|---|---|---|---|---|---|---|---|

None yet.
