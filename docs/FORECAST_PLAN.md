# Forecasting to parity: plan (Track FC)

**Odin Loch** · 2026-10-29 · from the discussion of 2026-10-29 ("Make a plan for it. Do docs for it. Go crazy on it. I want to bring this into parity with other Cypha features.") · PLAN ledger 75 · audit: [FC0](reports/FC0_FORECAST_AUDIT.md)

This plan takes the forecasting module from a scaffold to the level of the classifier and CyphaLM: readable real data, forecasts with horizons and intervals, proper scores against published baselines, frozen fixtures, pins, a saved model, and a REST contract.

Only FC0 (the audit) and FC1 (honest status) are done. Each later step is a registered experiment: rules and predictions first, dev for choices, the frozen test read once, intervals, and Odin's veto.

The 2026-08-08 design ([archive](archive/plans/CYPHA_FORECASTING_PLAN_2026-08-08.md)) is kept where it holds: Cypha's calibrated estimator at the nodes, a generated scenario tree, and VIEWS as the external check. Its Phases 1–9 stay closed in the bill of work. New work uses FC ids, because Track F (facts) already uses F1–F16.

Supporting documents:
- [ARCHITECTURE](research/forecasting/ARCHITECTURE.md): the target design.
- [DATA](research/forecasting/DATA.md): sources, versions, licences, schemas, provenance, event coding.
- [EVALUATION](research/forecasting/EVALUATION.md): tasks, splits, scores, baselines, freezing.

---

## Contents

1. [Where it stands](#1-where-it-stands)
2. [What parity means here](#2-what-parity-means-here)
3. [The rules this plan keeps](#3-the-rules-this-plan-keeps)
4. [What the field does](#4-what-the-field-does)
5. [The design in one page](#5-the-design-in-one-page)
6. [Work items](#6-work-items)
7. [Predictions](#7-predictions-registered-here-scored-when-the-runs-happen)
8. [Order, dependencies and agents](#8-order-dependencies-and-agents)
9. [Decisions for Odin](#9-decisions-for-odin)
10. [What this plan does not claim](#10-what-this-plan-does-not-claim)
11. [References](#11-references)

---

## 1. Where it stands

From [FC0](reports/FC0_FORECAST_AUDIT.md), measured 2026-10-29 at `09eaa6e`:

| Part | State |
|---|---|
| Committed data | 6 of 7 CSVs are UTF-16; the loaders read none of their rows. 57 identical MID rows, 51 events as one token, 0 UCDP and 0 ICB rows |
| Sequence model on real events | Throws: 9 of 13 event ids are ≥ 256 on a 256-symbol model |
| Escalation estimator | Inputs include the same year's outcome; random split |
| VIEWS score | `cypha_holdout` reads the held-out value; still last on CRPS (32.15 against 18.49) |
| CRPS function | 100 where the exact value is 470 |
| Scenario tree | No time steps; uncertainty fixed at 0; "guided" bonus added after sampling; path probability is one edge |
| Drift monitor | Fed a constant; backtest injects hand-set scores |
| Event coding | "Provide aid" and "yield" coded as threats of force; protest as use of force |
| Serving | Retrains about 845 MB per REST call; returns no forecast |
| Persistence, fixtures, pins, PR gate | None |

What works:
- **Rollouts read the trained model** since `5619cb7`, and a planted cycle is followed (trained 1.0, cold 0.0).
- **The vocabulary and theater types** are clean.
- **The ORF/SORF encoder** builds. Its claims need a recheck (FC16).
- **The 2026-08-08 design** is right about what to use Cypha for.

---

## 2. What parity means here

The bar is what the classifier and CyphaLM have today (FC0 §3). Each row is closed by the item named in it.

| Infrastructure | Forecast now | Forecast at parity | Item |
|---|---|---|---|
| Readable, frozen, real inputs with provenance | Unreadable placeholders | UTF-8 extracts with a sha256 manifest and licences | FC2, FC3 |
| Loaders fail closed | Silent defaults | Missing column, bad cell or wrong encoding is an error naming the file, row and column | FC2 |
| Forecasts have horizons and intervals | None | Samples or probabilities with intervals per unit and horizon | FC5, FC7, FC8 |
| Proper scoring | Wrong CRPS, renamed IGN | CRPS, IGN, MIS, Brier, log loss, ECE matching reference values to 1e-9 | FC6 |
| Published baselines beside every number | Stand-ins | VIEWS's four benchmarks; base rate, persistence, logistic regression, Markov | FC5, FC7, FC8 |
| No leakage, by construction | Leaks in C1–C3 | A forecaster sees history up to an origin and nothing else (a type, not a habit) | FC5, FC7 |
| Fixture directories and goldens | None | `fixtures/forecast_*` with sidecars, plus goldens in `native/tests/regression/` | FC13 |
| Numeric pins with a validator | None | `forecast_*` sections in `bench/BASELINE_LOCK.json`, checked by `baseline_lock_smoke` | FC13 |
| Fast tests in CI; smoke in the PR gate | Nothing that asserts | `native_forecast_*` fast set in `-LE cypha_slow`; probe, scoring and vocabulary in `ci_native_hp_smoke.sh` | FC13 |
| Bench table per PORT_CONTRACT §6 | No `backend` key | `domain`, `backend`, baselines, intervals | FC13 |
| Versioned save and load | None | A forecast bundle with a manifest, a round-trip golden and robustness tests | FC11 |
| REST serves a stored model under the lock | Retrains per call | `/forecast/load`, `/forecast/predict`, `/forecast/ingest` scoring by itself; `/health` and `/metrics` fields | FC12 |
| Contract | None | PORT_CONTRACT §5, `rest_schema_contract` keys, JSON schemas | FC12 |
| Registry card | None | `card.json` with task `forecast` | FC11 |
| Model card with numbers | One line | A section with backtest scores, intervals and limits | FC14 |
| Serves other tracks | Not usable | `Forecaster` interface for DISCOVERY G18 and THOUGHT `estimate`; prospective log (T8) | FC15 |
| Qt shell, Studio | None | Optional, after FC12 (§9.7) | — |

---

## 3. The rules this plan keeps

PLAN rules 1–4, 6 and 7 apply as everywhere: held-out tests, a tool and report for every number, predictions first, frozen sets never trained on, kill criteria first, honest names. HANDOFF rule 6 (Wilson or bootstrap intervals; the frozen test read once per change) applies too. Forecasting adds these.

1. **The origin rule.** A forecast made at origin *t* reads only data whose period is ≤ *t*. This is enforced by a type: forecasters take a `History` cut at the origin (ARCHITECTURE §3). No function takes both history and truth. C1 cannot be written again.
2. **No same-period inputs.** Features are lagged to the origin. A label is never computed from a feature column.
3. **Temporal splits only.** No shuffled splits for evaluation, ever. Train, dev and test are blocks of time (EVALUATION §2).
4. **Probabilities are probabilities.** A reported probability is in [0, 1], and the probabilities of the children of a node sum to 1. Exploration bonuses steer sampling. They are never added to a reported number.
5. **Every number has a baseline beside it.** A score is always reported against the published or simple baselines for that task, with an interval on the difference.
6. **Fail closed.** Degenerate input fails a tool or test, never passes it: zero rows, one class, a constant column, a token outside the vocabulary, or a forecaster with nothing to score.
7. **Provenance.** Every committed data file has a manifest row: source, version, URL, retrieval date, sha256, licence and citation (DATA §3). Placeholder data is named `synthetic_*` and never feeds a reported score.
8. **Name things honestly** (PLAN rule 7). A heuristic is called a baseline. Only a forecast that comes from a Cypha model is called "Cypha".
9. **True future is read once.** A forecast for a window that has not happened is written, hashed and committed before the truth exists, and scored once when it arrives (EVALUATION §6).

---

## 4. What the field does

| Work | Finding | What Cypha takes |
|---|---|---|
| **VIEWS Prediction Challenge 2023/24** ([Hegre et al. 2024](https://arxiv.org/pdf/2407.11045)) | Conflict forecasting as a full distribution over state-based fatalities, country-month and PRIO-GRID-month. 15–1,000 draws per unit-month. CRPS as the main score, IGN on 11 bins and MIS at 90% as secondary. Backtests 2018–2023 from an October origin; true future July 2024 to June 2025. Benchmarks: exactly zero, last historical (Poisson), bootstrap over 240 months | The protocol, scores and baselines for FC6 and FC7, unchanged |
| **Observed Markov Models** for conflict fatalities ([2410.12374](https://arxiv.org/pdf/2410.12374)) | Densities of fatalities from a Markov model over observed states; a challenge entry | A Markov baseline over the same bins (FC7) and over event tokens (FC8) |
| **Forests of Uncertaint(r)ees** ([2512.06210](https://arxiv.org/pdf/2512.06210)) | Tree ensembles that estimate distributions of future conflict; a challenge entry | A strong non-Cypha reference to report against, if its outputs are published |
| **Proper scoring rules** (Gneiting and Raftery 2007) | CRPS and the log score are strictly proper; CRPS from samples is E\|X − y\| − ½ E\|X − X′\| | The exact sample CRPS in FC6, replacing D1 |
| **UCT / PUCT tree search** (Kocsis and Szepesvári 2006) | Allocate rollouts by value plus an exploration bonus | Guidance in FC8, with importance weights so reported probabilities stay unbiased |
| **Importance sampling for rare events** | Biased sampling toward the tail, reweighted by p/q, keeps estimates unbiased and cuts their variance | How "guided" rollouts reach the escalation tail without distorting the numbers |
| **Isotonic calibration** | A monotone map fitted on dev fixes over- and under-confidence without hurting ranking | THOUGHT already uses it for p̂; the same map for tree probabilities (FC8) |
| **Orthogonal random features** (Yu et al., NeurIPS 2016) | Lower kernel approximation error than iid RFF; SORF at O(d log d) | Phase 1's encoder, rechecked in FC16 |
| **UCDP GED 26.1, Candidate 26.0.x** ([UCDP](https://ucdp.uu.se/downloads/)) | Event-level organised violence 1989–2025, CC BY 4.0; monthly candidate releases with about a month's lag (to September 2026 at 26.0.9) | Truth for FC7: GED for backtests, Candidate for the true future |
| **CoW MID 5.0** ([CoW](https://correlatesofwar.org/data-sets/MIDs)) | Disputes and participants 1816–2014; incidents 1993–2014; dyadic MID 4.03 to 2014 | FC5's dyad-year escalation task |
| **ICB v16** ([ICB](https://sites.duke.edu/icbdata/data-collections/)) | 512 crises 1918–2021, 1,131 crisis actors, a dyad file | Great-power involvement in crises: an FC5 feature and a check |
| **GDELT 1.0 and 2.0** ([GDELT](https://gdeltproject.org/data.html)) | Daily event files from 1 April 2013 (1.0); 15-minute updates across 65 translated languages (2.0); CAMEO-coded with root code and QuadClass columns | FC8's weekly theater sequences, and FC9's live feed |

---

## 5. The design in one page

The full design is in [ARCHITECTURE](research/forecasting/ARCHITECTURE.md).

| Layer | What it is | Built on |
|---|---|---|
| Data | Strict UTF-8 loaders, a provenance manifest, panels per unit and period (country-month, dyad-year, theater-week) | `csv_ingest`, new `panel.hpp` |
| History | A panel cut at an origin; the only thing a forecaster may read | New type |
| Escalation estimator | Dyad-year P(hostility ≥ 4 next year), calibrated, with per-row uncertainty and OOD | `cypha::Cypha` classify |
| Fatality forecaster | Country-month draws, 12 months ahead, from Cypha's regressor with its uncertainty | DIF mixture / quantile regressor |
| Event sequences | One byte per (theater, CAMEO root) plus QUIET and week-boundary tokens; an ensemble of K hp models | CyphaLM hp |
| Scenario tree | Time-stepped rollouts; P(level ≥ k within h weeks) with intervals; importance-weighted guidance | `rollout_tree` |
| Scores | Sample CRPS, VIEWS IGN, MIS, Brier, log loss, ECE, PIT; reference fixtures | New `scoring.hpp` |
| Monitor | Sequence surprise, token-distribution shift and classifier anomaly, with thresholds set on dev | `PredictOut`, CyphaLM log-loss |
| Bundle | A saved, versioned forecast model: manifest, component files, calibration, monitor state | `Cypha::save`, `save_cyphalm_model` |
| Serving | Load a bundle; predict; ingest; contract-tested | `cypha_rest` |

---

## 6. Work items

| # | Item | Measure | Gate (registered before the run) |
|---|---|---|---|
| FC0 | **Done 2026-10-29** ([report](reports/FC0_FORECAST_AUDIT.md)). **Audit.** Build, run, probe, read | — | — |
| FC1 | **Done 2026-10-29.** **Honest status.** RESEARCH_STATUS, README, MODEL_CARD, the forecasting README and the 2026-08-08 plan say "under repair" and retract the numbers made on placeholder rows | — | Docs only |
| FC2 | **Data layer that fails closed.** Re-encode the committed CSVs to UTF-8 and add `*.csv` to `.gitattributes`. A CTest rejects non-UTF-8 bytes under `bench/data/**`. Loaders read by header, error on a missing column, unparseable cell or UTF-16 BOM (file, row, column), and sort events by date. UCDP and ICB read by header. `forecast_data_probe` prints rows, date range, class balance, constant columns and the token histogram, and fails on one class or a constant column | Probe output on the committed files; planted bad files | Committed samples: MID 28 rows with 16 escalations, 51 events, 13 tokens; UCDP 7, ICB 6. Each planted fault (UTF-16, missing column, `abc` in an int column, a single class) is an error, never a default |
| FC3 | **Real data, small and frozen.** Replace the placeholders with derived extracts plus `MANIFEST.json`: UCDP GED 26.1 state-based country-month 1989–2025; MID 5.0 dyad-years 1816–2014; ICB v16 crises; GDELT 1.0 theater-week root counts from April 2013. Theaters assigned by an explicit country-code table, not name substrings. Placeholder files are renamed `synthetic_*`. The fetch is ported to `scripts/fetch_forecast_data.sh` as well as the `.ps1`, and fixes G1–G4 | Manifest, sha256, row counts | A CTest verifies every committed file against its manifest row; GDELT root comes from `EventRootCode`; VIEWS-style extracts keep zero months and state-based only |
| FC4 | **Events on a byte model** (decision §9.1). id = 2 + 20 × theater + (root − 1), giving 2–141; QUIET per theater 142–148; WEEK boundary 149; the rest reserved. Level becomes an attribute from DATA §5's map. One stream per theater-week, the top 3 roots by count in severity order, then WEEK. `static_assert` vocab ≤ 256. Sampling is masked to the theater's ids | Exhaustive check over all theater × root; the pipeline on FC3 data | No reachable id ≥ 256; the map fixture matches DATA §5; `forecast_smoke` runs on decoded and on FC3 data without throwing |
| FC5 | **Escalation estimator without leakage.** Target A: onset in year *t* + 1 of a dyadic MID that reaches hostility ≥ 4, for dyads with a MID in the past 10 years. Target B: any MID onset, for politically relevant dyads (DATA §4.2 says why onsets). Features at ≤ *t* only. Split train ≤ 1989, dev 1990–2001, test 2002–2014. Cypha's calibrated probability, with per-row uncertainty and `is_ood`. The theater bank is measured here (FC10) | Brier, log loss, AUC-PR, ECE, with bootstrap intervals by dyad | On dev, Brier skill against the base rate > 0 with the interval above 0; reported against persistence and logistic regression. Test read once |
| FC6 | **Scoring library.** `scoring.hpp`: sample CRPS (exact, sorted, O(n log n)), VIEWS IGN (11 bins, log₂, one added sample per bin), MIS at 90% (linear quantiles), Brier, log loss, ECE, PIT histogram, Wilson and cluster bootstrap. Replaces `crps()` and `ignorance_score()` | Reference fixture computed offline (properscoring) | Every fixture case within 1e-9, including FC0's four D1 cases (470, 20, 4,000, 75.56) |
| FC7 | **VIEWS-protocol fatality forecaster.** Delete `validate_views_holdout` and `forecast_from_mean`. A `Forecaster` writes 1,000 integer draws per country-month for the 12 months after an October origin, backtests 2018–2023. Cypha's regressor on lagged state-based fatalities, conflict history and GDELT counts. VIEWS's four benchmarks reimplemented | Mean CRPS, IGN and MIS per year and overall, with intervals | Choices on 2018–2020. On the test years 2021–2023, read once: mean CRPS below `bootstrap240`, with the bootstrap interval on the difference below 0. Reported against all four benchmarks for all six years. Then a true-future forecast registered (EVALUATION §6) |
| FC8 | **Scenario tree as forecasts.** Time-stepped rollouts per theater-week. Reported: P(level ≥ k within h weeks) with intervals, edge probabilities as visit shares only, path probability as the product. Guidance by PUCT with importance weights. Uncertainty as disagreement across K bootstrap members. Isotonic calibration on dev | Brier, BSS, log loss, ECE on theater-weeks 2022–2024 (test); planted Markov chain | Planted chain: 95% intervals cover the true probability at ≥ 0.93. Real: BSS against climatology > 0 with interval, reported against persistence and a first-order Markov chain |
| FC9 | **Drift and anomaly that read the model.** Three signals: CyphaLM surprise per theater-week, token-distribution shift (PSI) against training, and the estimator's `anomaly_score` / `is_ood`. Thresholds set on dev to a target false-alarm rate. `/forecast/ingest` computes its own scores. `forecast_drift_backtest` replays real weeks | Detection delay on the documented shocks (EVALUATION §6.1: four test shocks 2022–2024 in UKR, TWN and MID); false alarms per theater-year on test | Thresholds set on dev. An alarm in the right theater within 7 days of every test shock, at ≤ 2 false alarms per theater-year |
| FC10 | **Theater bank: measure, then keep or cut.** Trained on both classes, built in memory (no temp files), forgetting measured D16-style. Used by FC5 if kept | Dev Brier per theater against the pooled model; forgetting | Kept only if it beats pooled on dev with an interval; otherwise cut (HANDOFF rule 4) |
| FC11 | **Forecast bundle.** `<name>.forecast/` with `manifest.json` (schema, vocab hash, data manifest hash, origin), `escalation.cypha` and its preprocessor, `fatality.cypha` and its regression head, `seq_k.json`/`.hpbin` for each member, `calibration.json`, `monitor.json`. Save and load. A registry card with task `forecast` | Round trip; corrupted files | Forecasts identical to 1e-12 after reload; bad magic, unknown version and missing files are clean errors |
| FC12 | **Serving.** `POST /forecast/load`, `POST /forecast/predict` (fatality quantiles and draws, theater event probabilities with intervals, top paths), `POST /forecast/ingest`, all under `g_mu`. `/health` and `/metrics` fields. `/forecast/run` leaves the serving binary (§9.6). PORT_CONTRACT §5, JSON schemas, `rest_schema_contract` keys | Contract test; latency | Every documented key present; 400, 404 and 503 paths tested; `/forecast/predict` p95 under 200 ms on a loaded bundle |
| FC13 | **Tests, fixtures, pins, CI** (runs beside every item). `fixtures/forecast_*` with sidecars and READMEs; goldens in `native/tests/regression/`; fast CTests in `-LE cypha_slow`; probe, scoring and vocabulary in the PR gate; `forecast_*` pins in `BASELINE_LOCK.json` with tolerances in the validator; bench domain with `domain`, `backend` and baselines; the five tools fail closed; the `C:/Users` path removed; `cypha_forecast_run` gets a CTest | CTest counts; planted bugs caught | Each planted bug in EVALUATION §8 fails at least one CTest |
| FC14 | **Docs and release.** MODEL_CARD section with backtest scores and limits; README; PORT_CONTRACT; CHANGELOG; RESEARCH_STATUS returns to "Working, benchmarked" only when FC5 and FC7 meet their gates | — | Odin's read |
| FC15 | **Serving other tracks.** The `Forecaster` interface used by DISCOVERY G18 (trends with intervals) and THOUGHT `estimate`; a prospective log for DISCOVERY T8: dated forecasts with hashes, scored when the data arrives | G18's planted trends | G18's gate in DISCOVERY §13 (S12) |
| FC16 | **Phase 1 encoder recheck.** Rerun `orf_encoder_bench` with an eval set large enough for 1 pp intervals; run the full D01 pin with the SORF profile and record it in a report; reconcile the README table; drop the "+30 pp over iid" branch of the promotion rule | XOR and D01 with Wilson intervals | SORF stays the default only if D01 is within 2 pp of 0.9875 under the profile path |

---

## 7. Predictions (registered here, scored when the runs happen)

These are proposed by this plan and stand unless Odin changes them before the item runs.

| # | Prediction |
|---|---|
| P-FC1 | On FC3's real MID dyad-years, the train base rate of Target A (an onset reaching hostility ≥ 4 in *t* + 1, among dyads with a MID in the past 10 years) is below 0.10 |
| P-FC2 | FC5's Cypha estimator beats the base rate on dev Brier, but is within 0.005 Brier of logistic regression on the same lagged features |
| P-FC3 | Before isotonic calibration, FC8's tree probabilities from one hp model are overconfident: ECE > 0.05 on dev. An ensemble of K = 5 lowers it |
| P-FC4 | On theater-weeks, a first-order Markov chain over the same tokens comes within 0.01 Brier of CyphaLM rollouts at h = 1 week. CyphaLM's lead, if any, shows at h = 4 |
| P-FC5 | On the VIEWS cm test years 2021–2023, FC7 beats `bootstrap240` on CRPS, but does not beat `last_historical` in every year |
| P-FC6 | `exactly_zero` stays hard to beat on IGN at cm, because most country-months are zero |
| P-FC7 | FC9 alarms within 2 days of 24 Feb 2022 in the Ukraine theater and within 1 day of 7 Oct 2023 in the Middle East theater |
| P-FC8 | FC10's per-theater models do not beat the pooled model on dev: too few positives per theater |
| P-FC9 | With FC6's exact CRPS, the order on the sample leaderboard is unchanged (`negbin_glmm` first) |
| P-FC10 | FC16: SORF's D01 under the profile path is within 2 pp of 0.9875, and SORF beats iid on XOR by 5 pp or more with disjoint intervals |

---

## 8. Order, dependencies and agents

1. **The spine.** FC2 → FC3 → FC4 → FC6, then FC5 and FC7 in parallel, then FC8 → FC9.
   - FC10 sits inside FC5.
   - FC11 and FC12 come after FC5, FC7 and FC8 have models worth saving.
   - FC13 runs with every item: no item is done without its tests.
   - FC14 is last. FC15 follows FC7. FC16 is independent.
2. **Agents with file ownership** (HANDOFF rule 10):

| Agent | Items | Owns |
|---|---|---|
| FC-DATA | FC2, FC3 | `dispute_data.*`, new `panel.*`, `forecast_data_probe.cpp`, `bench/data/forecast/`, the fetch scripts, `.gitattributes` |
| FC-SCORE | FC6, FC7 | `views_scoring.*`, `views_baselines.*`, `views_holdout.cpp`, new `scoring.hpp`, `fatality_forecaster.*` |
| FC-SEQ | FC4, FC8 | `event_vocabulary.*`, `rollout_tree.*`, `interpretability.*`, new `sequence_ensemble.*` |
| FC-EST | FC5, FC9, FC10 | `node_estimator.*`, `theater_models.*`, `gdelt_monitor.*`, `gdelt_stream.*` |
| FC-SERVE | FC11, FC12 | `forecast_pipeline.*`, new `bundle.*`, the forecast routes in `cyphalm_rest_routes.cpp`, PORT_CONTRACT §5, schemas |

3. **Shared files go through one agent at a time:** `native/CMakeLists.txt`, `bench_domains.cpp`, `BASELINE_LOCK.json`, `ci_native_hp_smoke.sh`. FC-DATA lands first.
4. **Nothing here touches** `trie_stream_train.cpp`, `scripts/run_train.sh` (HANDOFF rule 9) or the GEN and TOK agents' files.
5. **The network.** The cloud container cannot reach UCDP, CoW, Duke or GDELT. FC3's fetch runs on Odin's PC, and the container works from the committed extracts. Licences permit this: UCDP is CC BY 4.0, and GDELT is "free and open". CoW and ICB ask for citation, and the manifest carries it.

---

## 9. Decisions for Odin

Each item has a recommendation. Each will be recorded as a PLAN ledger row when answered.

1. **Event ids on a byte model** (HANDOFF §5.10). The options:
   - **(a) Recommended:** one byte per (theater, root), 140 ids plus specials. The level moves out of the id, where it was redundant.
   - (b) Two bytes per event, with a grammar mask.
   - (c) A model that is not byte-level, which means bringing back a non-hp sequence model.
2. **Time step.** The options:
   - **Recommended:** theater-week for the tree. This gives enough events per step and one step per GDELT week.
   - Theater-day.
   - Month, which would match VIEWS cm.
3. **The headline output.** Recommended: VIEWS-protocol fatality distributions (FC7) are primary, because a published benchmark scores them. Theater event probabilities from the tree (FC8) are second.
4. **Real data in the repo.** Recommended: commit derived extracts with the manifest (FC3), so CI reads real rows. The alternative is fetch-only, with CI on synthetic rows.
5. **The theater bank.** Recommended: measure it (FC10), and cut it if pooled is not worse.
6. **Training over REST.** Recommended: remove `/forecast/run` from `cypha_rest`. Training becomes the CLI `cypha_forecast_run`, which writes a bundle. The alternative is a training route restricted to a configured data root, run as a job.
7. **Qt shell and Studio panels.** Recommended: not until FC12's contract is stable. Then one panel that loads a bundle and shows forecasts.
8. **The PR gate.** Recommended: add the fast forecast set (probe, scoring, vocabulary, bundle round trip; under 30 s together) to `ci_native_hp_smoke.sh`, so forecasting is gated like hp and CyphaLM.
9. **Public wording.** FC1 changed the status lines to "under repair". The v2.4.0 changelog entry and the 2.4.0 release notes stay as history, with a pointer from the archived plan. Recommended: keep it that way until FC14.

---

## 10. What this plan does not claim

1. **Great-power-war examples stay scarce.** Better data and scoring do not create them. Tail forecasts carry wide intervals, and the model card will say so.
2. **A generated tree is association, not causation.** Paths are statistical continuations. The interpretability layer reports them as such.
3. **News volume is not violence.** GDELT counts follow media attention, and its volume changes over time. FC8 scores against GDELT-coded events and, where theaters map to countries, against UCDP fatalities. Event counts are logged and normalised per week.
4. **Beating VIEWS's benchmarks is a first bar, not a claim of state of the art.** The challenge's entries are the real comparison. FC7 reports against any whose forecasts are public.
5. **Live use is out of scope** until FC9's false-alarm rate is measured on held-out weeks.

---

## 11. References

- Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102(477).
- Hegre, H., Vesco, P., et al. (2024). The 2023/24 VIEWS Prediction Challenge: predicting the number of fatalities in armed conflict, with uncertainty. [arXiv 2407.11045](https://arxiv.org/pdf/2407.11045).
- Forecasting densities of fatalities from state-based conflicts using Observed Markov Models. [arXiv 2410.12374](https://arxiv.org/pdf/2410.12374).
- Forests of Uncertaint(r)ees: using tree-based ensembles to estimate probability distributions of future conflict. [arXiv 2512.06210](https://arxiv.org/pdf/2512.06210).
- Kocsis, L. and Szepesvári, C. (2006). Bandit based Monte-Carlo planning. *ECML 2006*.
- Yu, F. X., Suresh, A. T., et al. (2016). Orthogonal random features. *NeurIPS 2016*.
- UCDP downloads: GED 26.1, Candidate 26.0.9, UCDP/PRIO ACD 26.1, CC BY 4.0. [ucdp.uu.se/downloads](https://ucdp.uu.se/downloads/).
- Correlates of War, Militarized Interstate Disputes v5.0 and dyadic MID 4.03. [correlatesofwar.org](https://correlatesofwar.org/data-sets/MIDs).
- International Crisis Behavior, Version 16 (1918–2021). [sites.duke.edu/icbdata](https://sites.duke.edu/icbdata/data-collections/).
- GDELT Project, data and documentation. [gdeltproject.org/data.html](https://gdeltproject.org/data.html).
- CAMEO event codebook, root codes. [parusanalytics.com](https://parusanalytics.com/eventdata/cameo.dir/cameocontents.html).
