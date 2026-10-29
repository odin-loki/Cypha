# FC0: the forecasting module, audited

**Odin Loch** · 2026-10-29 · [FORECAST_PLAN](../FORECAST_PLAN.md) FC0 · PLAN ledger 75 · follows [SLOW_TEST_TRIAGE.md](SLOW_TEST_TRIAGE.md) ("Fix") and HANDOFF §5.10

**Question.** The forecasting module (`native/src/forecast/`, `native/include/cypha/forecast/`) has been marked "Shipped (2026-08)" with Phases 1–9 done since `fd71d5c` (2026-08-08). Does it forecast anything? What stands between it and the testing, pins, persistence and serving that the classifier and CyphaLM have?

**Answer.** It does not forecast anything yet. On the committed data, the loaders read no real rows. All 57 dispute rows come back identical, and all 51 news events collapse to one token. Every number in [`research/forecasting/README.md`](../research/forecasting/README.md) was made from those placeholder rows.

When the same files are decoded correctly, the pipeline throws on its first event outside Taiwan or Global ("token id out of vocab range"). The VIEWS entry called `cypha_holdout` is not a Cypha model, and its forecast reads the held-out value it is scored on. The CRPS function used for every VIEWS score is wrong by up to a factor of 8 on simple cases. The drift backtest cannot fail.

The architecture in the 2026-08-08 design is sound, and parts of it are built: the encoder, the vocabulary, the rollout driver and the theater bank. It is a scaffold. [FORECAST_PLAN](../FORECAST_PLAN.md) takes it to parity.

No code was changed. No test was skipped, relabelled or loosened.

---

## Method

- **Build.** `09eaa6e` (HEAD of `main`, which includes `5619cb7`'s rollout fix), Linux, GCC, `-DCMAKE_BUILD_TYPE=Release`, Ninja. Targets built: `forecast_smoke`, `cypha_forecast_run`, `forecast_drift_backtest`, `views_leaderboard_bench`, `orf_encoder_bench`.
- **Two probes outside the repo**, linked against `libcypha_core.a`:
  - a loader probe that prints what `load_dispute_data` reads, the token ids, the VIEWS leaderboard, and a leak test;
  - a CRPS check that compares `crps()` with the exact CRPS of the same discrete distribution, E|X − y| − ½ E|X − X′|.
- **Runs.** Each tool ran on the committed `bench/data/forecast/`, then on copies decoded to UTF-8 with `iconv`. Each file was decoded by its own byte order (below).
- **Reading.** Every forecast source file, the fetch script, the REST routes, the bench domain and the CTest registrations were read line by line. The file history was read with `git log` over `bench/data/forecast`, `native/src/forecast` and `docs/research/forecasting`.
- **Parity map.** Taken from the classifier's and CyphaLM's fixtures, goldens, pins, CI jobs, REST contract, registry and save/load paths.

---

## 1. Results by class

| Class | Count | What it means |
|---|---|---|
| A. Data never read | 4 | Every result so far was computed on placeholder rows |
| B. Crash when the data is read | 1 | The pipeline cannot run on real events |
| C. Evaluation that sees the answer | 5 | The scores cannot be trusted even after A and B are fixed |
| D. Scoring rules wrong | 3 | The VIEWS numbers are not comparable with VIEWS |
| E. Outputs that are not what the docs say | 7 | Uncertainty, guidance and path probabilities are not computed |
| F. Event coding | 2 | Cooperative events are coded as force |
| G. Fetch script | 4 | The bulk paths would carry the same faults into real data |
| H. Serving | 4 | REST returns no forecasts and retrains on every call |
| I. Phase 1 encoder claims | 2 | The README table does not reproduce |
| J. Parity gaps | 23 rows | §3 |

### A. Data never read

**A1. Six of the seven committed CSVs are UTF-16; the loaders read bytes as UTF-8.**

| File | Encoding (first bytes) | Data lines |
|---|---|---|
| `gdelt_events.csv` | UTF-16LE, no BOM (`y\0e\0`) | 51 |
| `gml_mid.csv` | UTF-16, one byte out of step (`\0y\0e`, decodes as UTF-16BE) | 28 |
| `sample_gdelt.csv` | UTF-16LE | 15 |
| `sample_icb.csv` | UTF-16LE | 6 |
| `sample_mid.csv` | UTF-16LE | 21 |
| `sample_ucdp.csv` | UTF-16LE | 7 |
| `sample_views.csv` | ASCII | 10 |

- **What breaks.** `read_file_or_empty` and `parse_csv_utf8` (`dispute_data.cpp`) do no decoding, so no header matches and `col_index` returns −1 for every column. The MID loader then falls back to defaults: hostility 1, prev_hostility = hostility, great_power 0, escalated = (hostility > prev) = 0.
- **What the loader returns on the committed files:**
  - 57 MID rows (the file has 28 data lines), every one year 0, hostility 1, not escalated;
  - 51 GDELT events, every one token 22 (`GLB_VERBAL_C1`);
  - 0 UCDP rows and 0 ICB rows.
- **How long this has been true.**
  - The samples were UTF-16 in their first commit (`fd71d5c`, 2026-08-08; git stored them as binary).
  - `gml_mid.csv` arrived in `cf69cb5` as a UTF-8 BOM followed by UTF-16LE text.
  - None of the five versions of `dispute_data.cpp` has decoded UTF-16.
- **Why it happened.** `.gitattributes` forces UTF-8 for `*.cpp`, `*.hpp`, `*.md`, `*.json` and `*.ps1`, with a comment that "OneDrive/Windows editors have rewritten forecast/codec files as UTF-16". It does not cover `*.csv`.

**A2. UCDP and ICB never load.** Both loaders skip rows with `r.size() < 6` (`dispute_data.cpp:241`, `:258`), but both files have 5 columns. `ForecastPipeline::run` does not use either dataset anyway.

**A3. The sample data is placeholder, not drawn from the datasets it is named after.**
- `gml_mid.csv` has dyad-years from 2010 to 2024, but MID ends in 2014 (CoW MID 5.0).
- In all 28 rows, `escalated` is 1 exactly when hostility > prev_hostility and hostility ≥ 3.
- In `gdelt_events.csv`, 36 of the 51 rows are three (theater, code) pairs repeated once a month through 2023.

**A4. Events are not in time order.** `gdelt_events.csv` runs 2024, 2025, then 2023. `gdelt_to_token_sequence` keeps file order, so the sequence model trains on a stream whose order is not time.

### B. Crash when the data is read

**B1. 842 event ids on a 256-symbol byte model.**
- The vocabulary is 2 + 7 theaters × 6 levels × 20 CAMEO roots = 842 ids, and theaters are numbered Global, Taiwan, Ukraine, Middle East, Korea, South Asia, Africa. So every id from the Middle East onward is ≥ 362, and Ukraine's are 242–361.
- `apply_hp_production_recipe` sets `vocab_size` to 256, and `token_to_byte` throws on any id ≥ 256 (`cyphalm_model.cpp:149–152`).
- On the decoded data, 9 of the 13 distinct tokens are ≥ 256 (359, 360, 478–481, 575, 694, 716). `forecast_smoke` and `cypha_forecast_run` both end with `terminate … token id out of vocab range` (exit 134).
- HANDOFF §5.10 names this bug. FC0 adds the reason the smoke test still passes: A1 hides it.

### C. Evaluation that sees the answer

**C1. The VIEWS "cypha_holdout" forecast uses the held-out value.**
- `views_holdout.cpp:47` sets the forecast mean to `train_mean * (obs.fatalities > 0 ? 0.85 : 1.0) + obs.fatalities * 0.15`. `obs` is the held-out observation being scored.
- No Cypha model is involved.
- Multiplying the held-out values by 10 moves every forecast:

| Country | Truth | Forecast mean | Truth × 10 | Forecast mean |
|---|---|---|---|---|
| NGA | 60 | 231.4 | 600 | 312.4 |
| PAK | 40 | 228.4 | 400 | 282.4 |
| UKR | 480 | 294.4 | 4,800 | 942.4 |
| SUD | 195 | 251.7 | 1,950 | 514.9 |

Even with the leak, it is last on CRPS: 32.15, against 18.49 for `negbin_glmm`.

**C2. The escalation estimator is given the outcome.**
- `mid_to_features` (`dispute_data.cpp:288–294`) feeds the same year's `hostility` and `prev_hostility` as inputs.
- In the samples, `escalated` is a function of exactly those two (A3), and the bulk fetch defines it that way (G3). The classifier is asked to recompute its label from its inputs.

**C3. Random split, not temporal.** `NodeEstimator::train_on_mid` shuffles the rows (`node_estimator.cpp:35`) and holds out 20%. A model can then train on 2023 and be tested on 2019.

**C4. The drift backtest cannot fail.**
- `forecast_drift_backtest` feeds the monitor 51 events with drift 0.01 and anomaly 0.5. Then it injects one synthetic event dated 2026-02-01 with hand-set drift 0.85 and anomaly 4.5 (`forecast_drift_backtest.cpp:52`). Those values are above the thresholds (0.03, 2.0) by construction.
- No model is involved. The README's "alarms fire on Feb 2025 MID shock events in sample GDELT" is not what the tool checks.
- The tool also carries a fallback path `C:/Users/odinl/OneDrive/Desktop/Cypha/bench/data/forecast` (`:20`).

**C5. Two CTests assert nothing.**
- `views_leaderboard_bench` prints `cypha_best_crps` and always exits 0.
- `native_forecast_gdelt_poll` passes `--once`, which the tool does not parse. Its drift and anomaly come from a rule on the Goldstein score (`goldstein < -8 ? 0.12 : 0.01`, `forecast_gdelt_poll.cpp:53–54`), not from a model.

### D. Scoring rules wrong

**D1. `crps()` is not the CRPS.** It integrates (F(x) − H(x − y))² with the CDF evaluated at each bin's right edge, held over the whole interval to its left. Nothing above the last bin is counted.

| Forecast (bins 0, 10, 50, 100, 500, 1000) | y | `crps()` | Exact CRPS |
|---|---|---|---|
| All mass at 10 | 480 | 100.00 | 470.00 |
| All mass at 500 | 480 | 0.00 | 20.00 |
| All mass at 1000 | 5,000 | 500.00 | 4,000.00 |
| Uniform over the 6 bins | 60 | 27.78 | 75.56 |

**D2. The ignorance score uses VIEWS's name, not its definition.**
- It uses the natural log and the nearest of 6 bin centres.
- VIEWS uses log₂ over 11 bins (0, 1–2, 3–5, 6–10, 11–25, 26–50, 51–100, 101–250, 251–500, 501–1000, 1001+), with one extra sample added to each bin ([Hegre et al. 2024](https://arxiv.org/pdf/2407.11045)).

**D3. Not the VIEWS target, protocol or baselines.**
- **Target.** VIEWS forecasts state-based fatalities. The bulk fetch sums state-based, one-sided and non-state (G4).
- **Protocol.** VIEWS forecasts 12 months from a fixed origin, with 2018–2023 backtests and a true-future window.
- **Submission.** VIEWS takes 15 to 1,000 integer draws per unit-month.
- **Baselines.** VIEWS's are exactly-zero, last-historical (Poisson) and a bootstrap over 240 months.
- **The module.** It scores one holdout month list, with point means turned into 6-bin histograms by an ad hoc formula (`forecast_from_mean`). Its baselines are simplified stand-ins named after the published models.

### E. Outputs that are not what the docs say

**E1. The sequence model reports no uncertainty.** `serve_predict_next` sets `epistemic_var = 0.0` (`cyphalm_model.cpp:190`). So every tree node has `epistemic_var` 0, `leaf_value` 0 and `drift_flag` false. The "NIG-based leaf values" and "drift flags" in the design are constants.

**E2. "Guided" exploration does not guide.**
- Rollouts are plain temperature sampling.
- `active_query_score` is added to `transition_prob` after sampling (`rollout_tree.cpp:200`). That makes `transition_prob` exceed 1 and stop summing to 1 over children.
- `expand_top_k` and `rare_event_boost` are set in the configs and read nowhere.

**E3. "Path probability" is the last edge's score.** `interpretability.cpp:45` sets `path_probability` to the leaf's `transition_prob` (E2), not the product of the edges along the path.

**E4. No time in the tree.**
- A rollout of depth 8 is the next 8 events, at whatever times they happen.
- A forecast with no horizon cannot be scored against what happens by a date.

**E5. Before `5619cb7` (today), the tree sampled from an untrained model.**
- Every rollout and node called `reset_context()`, which built a cold predictor (SLOW_TEST_TRIAGE "Fix").
- The README's "1569 nodes" and the 998-node trees came from that cold model. The tree is now 34 nodes.

**E6. The theater bank learns one class and is not used.**
- `ForecastPipeline::run` trains it only on rows with `great_power && escalated`, all labelled `"escalate"` (`forecast_pipeline.cpp:50`).
- Its models are built for 64 inputs and given 6 raw ones.
- Nothing reads them after training.
- `model_for` writes `.cypha` files into the system temp directory (`theater_models.cpp:21`).

**E7. The drift monitor is fed a constant.**
- `forecast_pipeline.cpp:85` passes `estimator.model().drift_score()` as both the drift and the anomaly argument for every event. That is a training-time EMA, the same for all events.
- `Cypha::predict` already returns a per-row `anomaly_score` and `is_ood` (`cypha.hpp`, `PredictOut`). The pipeline never calls it on an event.

### F. Event coding

**F1. The CAMEO root → level map codes cooperation and protest as force** (`event_vocabulary.cpp:71–77`):

| Roots | CAMEO names | Coded as |
|---|---|---|
| 01–03 | Make public statement, appeal, agree | Verbal |
| 04–06 | Consult, support diplomatically, cooperate | Material |
| 07–08 | Provide aid, yield | ForceThreat |
| 09–14 | Investigate, demand, disapprove, reject, threaten, protest | ForceUse |
| 15–20 | Exhibit military posture, reduce relations, coerce, assault, fight, mass violence | War |

The root names are from the [CAMEO codebook](https://parusanalytics.com/eventdata/cameo.dir/cameocontents.html). A proposed map is in [DATA.md](../research/forecasting/DATA.md) §5.

**F2. The root is taken as `cameo_code / 10`** (`event_vocabulary.cpp:148`, `dispute_data.cpp:304`). That is right for three-digit codes such as 190. It is wrong for GDELT's four-digit codes: 1823 becomes root 182, which is clamped to 20, though it belongs to root 18. It is also wrong for two-digit root-only codes: 19 becomes root 1. GDELT carries `EventRootCode` as its own column.

### G. Fetch script (`scripts/fetch_forecast_data.ps1`)

1. **GDELT bulk** (`Convert-GdeltExportLine`, `Fetch-GdeltBulk`):
   - It takes `EventCode` (field 26), so F2 applies to all of it.
   - It assigns theaters by substring match on actor **names**. `RUS` matches BRUSSELS and `IND` matches INDONESIA. In the C++ fallback `theater_from_dyad`, `chn` is tested first, so a CHN–IND dyad lands in Taiwan.
   - It keeps the first 5,000 rows of one day's file, unsampled and unsorted.
2. **`-Repair`** copies `sample_mid.csv` and `sample_gdelt.csv` over the aliases, but the samples are UTF-16 too (A1). Its header check reads them as UTF-8, fails, and copies again.
3. **MID bulk** (dyadic MID 4.03):
   - It writes theater `GLB`, great_power 1, duration 1 and gdp_ratio 1.0 for every row (line 206).
   - It sets `escalated` to "hostility above the dyad's previous row" (line 205), which is C2 again. Rows stay in file order, capped at 8,000.
4. **VIEWS bulk:**
   - It sums `ucdp_ged_sb`, `_os` and `_ns` (line 255), while the VIEWS target is state-based only.
   - It drops every row with zero fatalities (line 260). Most country-months are zero, and VIEWS scores them.
   - It keeps 2020 onward and holds out the last three months fetched.

### H. Serving (`native/apps/cyphalm_rest_routes.cpp`)

1. **`POST /forecast/run` (:393)**
   - It retrains the whole pipeline from CSV on every request, an hp model of about 845 MB (SLOW_TEST_TRIAGE), with no lock.
   - It reads `data_dir` from the request body as a server-side path.
   - It returns counts and scores, but no forecast.
2. **`POST /forecast/ingest` (:442)** takes `drift` and `anomaly` from the caller. The server scores nothing itself.
3. **No contract.** The forecast routes are not in PORT_CONTRACT, `rest_schema_contract`, `/health` or `/metrics`.
4. **No persistence.** Nothing in the module can be saved or loaded. Every run retrains.

### I. Phase 1 encoder claims

1. **The README table does not reproduce.** The README shows iid 0.438, SORF 0.688 and ORF 0.513 on XOR. Today's `orf_encoder_bench` gives iid 0.500, SORF 0.6125 and ORF 0.5875. At 0.6125 = 49/80, the eval set is about 80 points, so one point is 1.25 pp.
2. **The promotion gate passes while SORF sits 15 pp under the D01 pin on the same harness.**
   - The gate is "XOR +5 pp **and** D01 within 2 pp of the pin, **or** +30 pp over iid".
   - Today: D01 golden iid 0.525 (pin 0.9875), SORF 0.8375, ORF 0.900, so `promote=sorf` by the "or" branch.
   - The README says the full D01 pin was checked under `CYPHA_BENCH_USE_PROFILE=1`. That run is not in any report.

---

## 2. Measured outputs

On the committed files:

```
loader probe:      mid=57 gdelt=51 ucdp=0 icb=0
                   mid[0..2] year=0 host=1 prev=1 gp=0 esc=0;  mid_escalated=0 of 57
                   gdelt_unique_tokens=1 : 22(GLB_VERBAL_C1)
forecast_smoke:    rollout_cycle trained=1 cold=0 · rollout_trained OK nodes=16
                   node_eval_acc=1 · tree_nodes=34 · forecast_smoke OK   (4 s)
forecast_drift_backtest:  baseline_alarms=0 shock_fired=1 shock_drift=0.0513 · OK
views_leaderboard_bench:  conflictology_mean crps=31.9547 · observed_markov 25.2778
                          negbin_glmm 18.4869 · cypha_holdout 32.1492 · n=4 · cypha_best_crps=0 · OK
orf_encoder_bench: xor iid 0.5 · sorf 0.6125 · orf 0.5875
                   d01_golden iid 0.525 (pin~0.9875) · sorf 0.8375 · orf 0.9 · promote=sorf
```

On the same files decoded to UTF-8:

```
loader probe:      mid=28 gdelt=51 ucdp=0 icb=0;  mid_escalated=16 of 28
                   gdelt_unique_tokens=13 : 93 117 213 214 359 360 478 479 480 481 575 694 716
forecast_smoke:    terminate called after throwing an instance of 'std::runtime_error'
                   what():  token id out of vocab range        (exit 134)
cypha_forecast_run: same                                       (exit 134)
```

`node_eval_acc=1` on the committed files is one class, all rows identical. The rollout check is real and passes: a model trained on a planted cycle follows it, and a cold one does not. It is the one part of the smoke that tests forecasting machinery.

---

## 3. Parity: what the classifier and CyphaLM have that forecasting does not

| Infrastructure | Classifier | CyphaLM (hp) | Forecast |
|---|---|---|---|
| Readable frozen inputs | `fixtures/` | Synthetic ids in tests | UTF-16 CSVs the loader cannot read |
| Fixture directories (sidecar, README, before/after) | About 30 | `cyphalm_checkpoint/*` and in-code pins | None |
| Golden binaries in `native/tests/regression/` | About 27 | Round trip, gate24, model, checkpoint | None |
| Numeric assertion with a tolerance | `atol` in sidecars | BPC ±0.05 and latency ceilings | Only "the hand-set shock fires" |
| Determinism and round-trip tests | Save and load | Checkpoint round trip, stream rewind, undo | Same seed, same tree (synthetic part only) |
| Negative tests | REST 400 / 404 / 503 contract | `checkpoint_robust`, CLI error regexes | None |
| In the blocking PR gate | No (optional job) | Yes (Linux, macOS) | No |
| In the `-LE cypha_slow` job | Yes | Yes | `drift_backtest`, `gdelt_poll` (both assert nothing) |
| Bench domain with a CTest | d03, d14 smokes | d17, d25 smokes | `forecast` tag, no CTest |
| Bench table per PORT_CONTRACT §6 (`backend`, baselines) | Yes | Yes | No `backend` key |
| Pin with a validator | D01 in code | d17 lock; gate24 constants | None |
| Model card numbers | D01, D03, D10A | Full section | One line, no numbers |
| PORT_CONTRACT section | §1–3 | §4, §4b | None |
| REST contract test | `rest_schema_contract` | `sequence_loaded` key | None |
| REST serves a stored model | `.cypha`, registry `/load` | Checkpoint by path | Retrains per request |
| REST lock | `g_mu` and per-slot mutex | `g_mu` | None on `/forecast/run` |
| `/health`, `/metrics` fields | Yes | `sequence_loaded` | None |
| Qt shell | Full | Cypha panel | None |
| Studio web UI | `/predict` | Chat over `/generate` | None |
| Registry card | Yes | Path-loaded | None |
| Versioned save and load | `.cypha` v3 | `.json` + `HPCP` `.hpbin` | None |
| Release packaging | Binaries and goldens | Bench and golden | None |
| Bench profile in `bench/config` | `everyday_profile.json` | `profiles/cyphalm_hp_gate24.json` | None |

Two other tracks rely on this module:
- DISCOVERY G18 runs fact time series through it: "forecasts, labelled, with intervals".
- THOUGHT's `estimate` process lists it beside DIF regressors with intervals.

Neither can use it today.

---

## For Odin

1. **Event ids on a byte model** (HANDOFF §5.10). FC4 recommends one byte per (theater, CAMEO root): 7 × 20 = 140 ids plus specials, under 256. The escalation level then becomes an attribute of the root rather than part of the id, which it already is in practice (`from_cameo` derives it). The alternatives are two bytes per event, or a model that is not byte-level. Take the recommendation?
2. **What the headline forecast is.** FC7 makes VIEWS-protocol country-month fatality distributions the primary scored output, because a published benchmark exists to score them against. The scenario tree (FC8) is second and is scored on theater-week event probabilities. Agree?
3. **The status in public docs.** With this report, the module's docs now say "under repair": RESEARCH_STATUS, the forecasting README, MODEL_CARD, README and the 2026-08-08 plan's banner. The v2.4.0 changelog entry stays as history. Is that the right line?
4. **Real data in the repo.** UCDP is CC BY 4.0, GDELT is "100% free and open", and CoW MID and ICB ask for citation. FC3 commits small derived extracts with a provenance manifest and sha256, so CI reads real rows without network access. Or should it fetch only?
5. **The theater bank.** Measure it against a pooled model in FC10, and cut it if pooled is not worse (HANDOFF rule 4)?
6. **Training over REST.** FC12 removes `/forecast/run` from the serving binary, in favour of a stored bundle and `/forecast/predict`. Keep a restricted training route instead?
7. **CTests that assert nothing** (`views_leaderboard_bench`, `forecast_gdelt_poll --once`, `forecast_drift_backtest`). FC13 makes them fail closed rather than deleting them. Agree?

---

## How to reproduce

```bash
# Build the five tools (about 15 min on 2 cores)
cmake -S native -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build forecast_smoke cypha_forecast_run forecast_drift_backtest views_leaderboard_bench orf_encoder_bench

# On the committed files
build/forecast_smoke --data bench/data/forecast
build/forecast_drift_backtest --data bench/data/forecast
build/views_leaderboard_bench --data bench/data/forecast
build/orf_encoder_bench

# Decode each CSV by its own byte order, then run again
mkdir -p /tmp/fc_utf8
for f in bench/data/forecast/*.csv; do
  b=$(od -An -tx1 -N2 "$f" | tr -d ' ')
  case $b in 00*) e=UTF-16BE;; ??00) e=UTF-16LE;; *) e=UTF-8;; esac
  iconv -f $e -t UTF-8 "$f" | tr -d '\r' | sed 's/^\xEF\xBB\xBF//' > /tmp/fc_utf8/$(basename "$f")
done
build/forecast_smoke --data /tmp/fc_utf8          # throws: token id out of vocab range
build/cypha_forecast_run --data /tmp/fc_utf8      # same

# Encoding history
git log --format='%h %ad %s' --date=short -- bench/data/forecast
git show fd71d5c:bench/data/forecast/sample_mid.csv | od -c | head -1
```

- **Probes.** The loader probe and the CRPS check are about 120 lines in total. They become `native/tools/forecast_data_probe.cpp` and a scoring golden in FC2 and FC6.
- **Run times.** The tools run in seconds. The build is the slow part.
- **The decoded data.** It is the same placeholder rows. Decoding shows what the code does with readable input; it does not make the rows real (A3).
