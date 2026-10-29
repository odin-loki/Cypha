# Cypha event forecasting

**Odin Loch** · rewritten 2026-10-29 · [FORECAST_PLAN](../../FORECAST_PLAN.md) · audit: [FC0](../../reports/FC0_FORECAST_AUDIT.md)

> **Under repair (2026-10-29).** The module is a scaffold. Its committed data has never been read (the CSVs are UTF-16; the loaders read UTF-8). On readable data, the sequence model throws on any event outside Taiwan and Global. The VIEWS score reads the held-out value it is scored on, and the CRPS function is wrong. [FC0](../../reports/FC0_FORECAST_AUDIT.md) has the evidence. [FORECAST_PLAN](../../FORECAST_PLAN.md) (Track FC) is the way to parity.

**Numbers retracted.** These were made from placeholder rows the loader could not read, or by code that does not compute what it was named for. No result depends on them.

| Number | Where it appeared | Why it is withdrawn |
|---|---|---|
| Node eval accuracy **1.0** | Smoke, REST, bench | All 57 rows identical, one class (FC0 A1) |
| Scenario tree **1569 nodes** | Smoke | Built on an untrained model before `5619cb7`, from one repeated token (E5, A1) |
| VIEWS CRPS and ignorance, the leaderboard | REST, bench | The forecast reads the holdout (C1); the CRPS is wrong (D1); the ignorance is not VIEWS's (D2) |
| Drift alarms "on Feb 2025 MID shock events" | This README | The backtest injects hand-set scores (C4) |
| XOR 0.438 / 0.688 / 0.513 (iid / SORF / ORF) | This README | Do not reproduce: 0.500 / 0.6125 / 0.5875 on 2026-10-29 (I1) |

---

## Status by part

| Part | Path | State |
|---|---|---|
| ORF/SORF encoder | `native/include/cypha/orthogonal_rff_encoder.hpp` | Built; SORF on in `everyday_profile.json`; promotion rule to recheck (FC16) |
| Event vocabulary | `native/include/cypha/forecast/event_vocabulary.hpp` | Built; 842 ids do not fit the 256-byte model (FC4); CAMEO level map wrong (FC0 F1) |
| Data loaders | `native/include/cypha/forecast/dispute_data.hpp` | Read no rows of the committed files; UCDP and ICB never load (FC2) |
| Escalation estimator | `native/include/cypha/forecast/node_estimator.hpp` | Same-year inputs, shuffled split (FC5) |
| Sequence training | `native/src/forecast/forecast_pipeline.cpp` | Throws on real events (FC4) |
| Rollout tree | `native/include/cypha/forecast/rollout_tree.hpp` | Reads the trained model (since `5619cb7`); no time steps, no uncertainty, bonus added to reported numbers (FC8) |
| Interpretability | `native/include/cypha/forecast/interpretability.hpp` | Path probability is one edge (FC8) |
| Theater bank | `native/include/cypha/forecast/theater_models.hpp` | Trains one class, unused, writes temp files (FC10) |
| VIEWS scoring | `native/include/cypha/forecast/views_scoring.hpp` | CRPS wrong; IGN not VIEWS's (FC6) |
| VIEWS baselines | `native/include/cypha/forecast/views_baselines.hpp` | Simplified stand-ins (FC7) |
| Drift monitor | `native/include/cypha/forecast/gdelt_monitor.hpp` | EMA of a caller's number (FC9) |
| Save and load | — | None (FC11) |
| REST | `native/apps/cyphalm_rest_routes.cpp` | Retrains per call; no forecasts returned (FC12) |

---

## Documents

| Document | What it is |
|---|---|
| [FORECAST_PLAN](../../FORECAST_PLAN.md) | Track FC: parity, rules, work items FC0–FC16, predictions, decisions |
| [ARCHITECTURE](ARCHITECTURE.md) | Target design: types, tokens, tree, estimator, forecaster, monitor, bundle, routes |
| [DATA](DATA.md) | Sources, versions, licences, file schemas, manifest, CAMEO level map, theater rules, fetch |
| [EVALUATION](EVALUATION.md) | Tasks, splits, scores and their exact definitions, baselines, shocks, freezing rule, planted bugs |
| [FC0 audit](../../reports/FC0_FORECAST_AUDIT.md) | What was measured on 2026-10-29, and how to reproduce it |
| [2026-08-08 design](../../archive/plans/CYPHA_FORECASTING_PLAN_2026-08-08.md) | The original design; history |

---

## Running it today

These run, but the numbers they print are not results (see above).

```bash
cmake -S native -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build forecast_smoke cypha_forecast_run forecast_drift_backtest views_leaderboard_bench orf_encoder_bench
build/forecast_smoke --data bench/data/forecast      # passes only because no row is read
build/orf_encoder_bench                              # the encoder bench; real, small eval set
```

- **The one real check.** `forecast_smoke`'s first half (`rollout_check`) trains on a planted cycle and checks that rollouts follow it, learn nothing and are reproducible from a seed. It passes, and it tests real machinery.
- **The rest of the smoke** runs the pipeline on unreadable data.

---

## Data

`bench/data/forecast/` holds placeholder CSVs, not extracts of MID, UCDP, ICB, GDELT or VIEWS. Six of the seven are UTF-16. FC2 re-encodes them and renames them `synthetic_*`. FC3 adds real extracts with a manifest (DATA §3, §4).

`scripts/fetch_forecast_data.ps1 -Bulk` exists, but carries the faults listed in FC0 G1–G4. Do not use its output for results until FC3.

---

## REST

| Route | What it does today |
|---|---|
| `POST /forecast/run` | Retrains the whole pipeline from `data_dir` on every call; returns counts and scores, no forecast |
| `POST /forecast/ingest` | Adds a caller's own `drift` and `anomaly` numbers to an EMA |
| `POST /forecast/monitor/reset` | Resets that EMA |

FC12 replaces these with `/forecast/load`, `/forecast/predict` and a scoring `/forecast/ingest` (ARCHITECTURE §10).
