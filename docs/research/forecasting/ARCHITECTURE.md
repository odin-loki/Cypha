# Forecasting: target architecture

**Odin Loch** · 2026-10-29 · [FORECAST_PLAN](../../FORECAST_PLAN.md) §5 · audit: [FC0](../../reports/FC0_FORECAST_AUDIT.md)

This is the design the FC items build toward. Nothing here is built yet, except where a row says so. Interfaces are proposed and will be settled by the agent that owns the file (FORECAST_PLAN §8).

---

## 1. Shape

```
bench/data/forecast/*.csv + MANIFEST.json
        │  strict UTF-8 loaders, provenance checked           (FC2, FC3)
        ▼
Panels: country-month · dyad-year · theater-week
        │  History = panel cut at an origin                    (rule 1)
        ▼
┌──────────────────────┬────────────────────────┬─────────────────────────┐
│ Escalation estimator │ Fatality forecaster    │ Event sequence ensemble │
│ cypha::Cypha classify│ Cypha DIF regressor    │ K × CyphaLM hp (bytes)  │
│ dyad-year P(≥4, t+1) │ country-month draws    │ theater-week tokens     │
│ FC5                  │ FC7                    │ FC4                     │
└──────────┬───────────┴───────────┬────────────┴────────────┬────────────┘
           │                       │                         ▼
           │                       │              Scenario tree (FC8)
           │                       │              P(level ≥ k within h)
           ▼                       ▼                         ▼
      Forecasts: SampleForecast · EventProbability · ScenarioPath
           │
           ├──► Scoring (FC6): CRPS · IGN · MIS · Brier · log loss · ECE · PIT
           ├──► Monitor (FC9): surprise · token shift · anomaly / OOD
           ├──► Bundle (FC11): save · load · registry card
           └──► Serving (FC12): /forecast/load · /predict · /ingest
```

---

## 2. Where each part comes from

| Part | Today (`09eaa6e`) | Target | Item |
|---|---|---|---|
| `dispute_data.*` | Byte loaders, silent defaults, fixed column counts for UCDP and ICB | Strict loaders by header; errors with file, row and column; date sort | FC2 |
| — | — | `panel.hpp`: typed panels per unit and period, `History` | FC2, FC5, FC7 |
| `event_vocabulary.*` | 842 ids (theater × level × root) | 150 ids: theater × root, QUIET, WEEK; level as an attribute | FC4 |
| `node_estimator.*` | Same-year inputs, shuffled split, accuracy | Lagged inputs, temporal split, calibrated probability, uncertainty, OOD | FC5 |
| `theater_models.*` | One class, unused, temp files | Measured against pooled; kept or cut | FC10 |
| `views_holdout.cpp` | Reads the held-out value | Deleted | FC7 |
| `views_scoring.*` | Wrong CRPS, renamed IGN | Replaced by `scoring.hpp` | FC6 |
| `views_baselines.*` | Stand-ins | VIEWS's four benchmarks | FC7 |
| — | — | `fatality_forecaster.*` | FC7 |
| `rollout_tree.*` | Event-count depth, bonus added to reported numbers | Week steps, PUCT with importance weights, intervals | FC8 |
| `interpretability.*` | Leaf edge as path probability | Product of edges; top paths by probability and by spread | FC8 |
| — | — | `sequence_ensemble.*`: K bootstrap members | FC8 |
| `gdelt_monitor.*` | EMA of a caller's number | Three model signals; thresholds from dev | FC9 |
| `forecast_pipeline.*` | Trains everything per run | Trains and writes a bundle | FC11 |
| — | — | `bundle.*` | FC11 |
| REST forecast routes | Retrain per call, client-supplied drift | Load, predict, ingest; contract | FC12 |
| `orthogonal_rff_encoder.hpp` | Built, SORF default | Rechecked | FC16 |

---

## 3. Core types (proposed)

```cpp
namespace cypha::forecast {

/// A period on one calendar: a month, an ISO week or a year.
struct Period {
  enum class Kind : std::uint8_t { Year, Month, Week } kind;
  int year;
  int index;   // month 1–12, ISO week 1–53, or 0 for Year
  auto operator<=>(const Period&) const = default;
};

/// The last period whose data a forecaster may read (FORECAST_PLAN rule 1).
struct Origin { Period last_observed; };

/// A panel cut at an origin. Built only by Panel::history(origin): rows after the origin
/// are not copied in, so no forecaster can read them.
class History {
 public:
  const Origin& origin() const;
  std::span<const Row> rows(UnitId unit) const;   // every row has period <= origin
  // no accessor returns a row after origin()
};

struct Target { UnitId unit; Period period; int steps_ahead; };

/// Count forecast as integer draws (VIEWS takes 15–1,000 per unit-period).
struct SampleForecast {
  Target target;
  std::vector<std::int32_t> draws;
};

/// "Level ≥ at_least happens in the theater within horizon steps".
struct EventProbability {
  Theater theater;
  EscalationLevel at_least;
  int horizon_steps;
  double p;          // in [0, 1]; never carries an exploration bonus (rule 4)
  double lo, hi;     // 95% interval
  double ess;        // effective sample size behind p
  int members;       // ensemble members that agree within the interval
};

class Forecaster {
 public:
  virtual ~Forecaster() = default;
  virtual std::string name() const = 0;          // "cypha_dif_regressor", "bm_last_historical", …
  virtual bool is_baseline() const = 0;          // rule 8
  virtual std::vector<SampleForecast> forecast(const History& h,
                                               std::span<const Target> targets,
                                               std::mt19937_64& rng) = 0;
};

/// Scoring takes forecasts and truth, never a History (C1 cannot be written).
struct Truth { Target target; double y; };
ScoreTable score(std::span<const SampleForecast> f, std::span<const Truth> y, const ScoreSpec& spec);

}  // namespace cypha::forecast
```

**Why types, not care.** C1, C2 and C3 in FC0 were each one line. The origin rule holds only if a forecaster cannot be handed a future row.
- `History` makes the cut once.
- `score` never sees a `History`.
- A CTest plants a forecaster that tries to read past the origin (EVALUATION §8). It must fail to compile, or throw.

---

## 4. Event tokens on a byte model (FC4)

### 4.1 Ids

| Ids | Meaning |
|---|---|
| 0 | PAD |
| 1 | UNK |
| 2–141 | Event: id = 2 + 20 × theater + (root − 1); theaters 0–6 (Global, Taiwan, Ukraine, Middle East, Korea, South Asia, Africa); roots 1–20 |
| 142–148 | QUIET for theaters 0–6: no event above the count threshold that week |
| 149 | WEEK: the end of one theater-week |
| 150–255 | Reserved; never emitted, never sampled |

- `static_assert(kVocabSize <= 256)`.
- The level is not in the id. It is `level_of_root(root)` from the registered map (DATA §5).
- Changing the map changes no ids, so a saved sequence model stays valid. The vocabulary hash in the bundle (§9) covers the id table only. The map has its own hash.

### 4.2 Streams

- **One stream per theater.** Each week contributes:
  - the up to 3 roots with the most events that week, among roots whose count passes a threshold set on train (DATA §4.3);
  - these in severity order, most severe last;
  - then WEEK.
- A week with none is QUIET, WEEK.
- A rollout's horizon h is h WEEK tokens.
- **Masked sampling.** During rollouts, only that theater's ids, its QUIET and WEEK can be drawn. The other ids get probability 0 and the rest is renormalised.
- **Training.** All theater streams go through one hp model, prefixed by the theater's QUIET token so the model knows which stream it is in. The alternative, one model per theater, is FC10's question.

### 4.3 Why one byte per (theater, root)

- hp predicts bytes and rejects any other vocabulary (`5619cb7`).
- The level was a function of the root in every path that builds sequences, so 6 × 140 ids carried no information beyond 140.
- Two-byte events would double the steps per event and need a grammar.

---

## 5. Scenario tree (FC8)

### 5.1 What is reported

For each theater, level k and horizon h ∈ {1, 4, 13} weeks:

  p̂(k, h) = Σᵢ wᵢ · 1[path i reaches level ≥ k within h weeks] / Σᵢ wᵢ

- With plain sampling, wᵢ = 1. The 95% interval is Wilson on n rollouts.
- With guidance, wᵢ = p(pathᵢ) / q(pathᵢ):
  - p is the model's probability of the path;
  - q is the probability under the guided proposal.
  - The interval uses the effective sample size, ess = (Σw)² / Σw².
- **The ensemble.** Each of the K members gives its own p̂. The reported p̂ is their mean. The interval is the wider of:
  - the Wilson or ess interval on the pooled rollouts;
  - the spread across members, the 2.5th to 97.5th percentile over K bootstrap fits.
- **Calibration.** An isotonic map, fitted on dev, is applied to p̂. The raw value is kept in the output for audit.

### 5.2 Edges and paths

- An edge's probability is the share of the parent's (weighted) visits that took it. Children sum to 1 (rule 4).
- A path's probability is the product of its edges.
- `interpretability` returns:
  - the top paths by probability;
  - the top paths by member spread, where the members disagree most;
  - each path's first week at level ≥ ForceUse.

### 5.3 Guidance

- **Allocation** is by PUCT at each node: value + c · prior · √N_parent / (1 + N_child).
  - The value is the share of rollouts through the child that reach level ≥ k.
  - The prior is the model's probability.
  - `active_query_score` (entropy × (1 − max p)) can replace the prior term as an exploration bonus.
- **The rule.** The bonus decides where rollouts go, and the weights wᵢ undo it in what is reported. Today's code adds the bonus to `transition_prob` (FC0 E2); that is removed.

### 5.4 The planted test

A first-order Markov chain over 10 tokens with known transitions, about 5,000 weeks of it, trains a member. For every k and h, the true probability of reaching level ≥ k within h steps can be computed exactly.
- The tree's 95% intervals must cover it at ≥ 0.93 over 200 (k, h, seed) cases.
- With guidance on, coverage must not fall.

This is the forecasting counterpart of T3's planted chain in TOKENS_PLAN.

---

## 6. Escalation estimator (FC5)

| | |
|---|---|
| Unit | Dyad-year |
| Target A | Onset in *t* + 1 of a dyadic MID whose highest hostility is ≥ 4 (use of force or war), for dyads with a MID in the past 10 years |
| Target B | Onset of any MID in *t* + 1, for all politically relevant dyad-years (a second task) |
| Target C (optional) | Within-dispute escalation to ≥ 4 on incident data (MIDI), 1993–2014 only |
| Why onsets | Dispute-level hostility cannot be spread over a dispute's years before 1993 (DATA §4.2) |
| Inputs at ≤ *t* | Highest hostility of the dyad's disputes ongoing in *t*, and over the past 5 and 10 years; MIDs in the past 10 years; years since the last MID; great-power status of each side (CoW major powers); contiguity; capability ratio (CoW NMC CINC, if FC3 adds it); ICB crisis active in *t* with great-power involvement |
| Model | `cypha::Cypha` classify through the shared preprocessor (ORF/SORF per FC16) |
| Output | P(target), the classifier's `uncertainty`, `anomaly_score`, `is_ood` |
| Calibration | Isotonic on dev |
| Baselines | Base rate (train); persistence (a hostility ≥ 4 MID ongoing in *t*); logistic regression on the same inputs |

---

## 7. Fatality forecaster (FC7)

| | |
|---|---|
| Unit | Gleditsch–Ward country-month |
| Target | UCDP state-based fatalities, best estimate, as counts (no log on the scored scale) |
| Origin and horizon | October of year *Y* − 1; the 12 months of *Y* (VIEWS's s = 3 … 14) |
| Inputs at ≤ origin | log1p fatalities at lags 1, 2, 3, 6, 12; 12- and 36-month sums; months since the last fatality; the neighbourhood sum (CShapes contiguity, if FC3 adds it); GDELT counts of roots 18–20 at lags 1–3 (from April 2013); the region |
| Model | Cypha's DIF mixture regressor: classes are VIEWS's 11 IGN bins of the target. The mixture gives p(bin) and a within-bin mean and variance. Draws: a bin by p(bin), then a value within the bin from a discretised Gaussian, clipped to the bin. Each horizon s is its own head (direct forecasting), so sampled values are never fed back as inputs; a recursive arm is tried on dev only |
| Output | 1,000 int32 draws per country-month |
| Baselines | `bm_exactly_zero`, `bm_last_historical` (Poisson at the last value), `bm_conflictology_bootstrap240` (240 months pooled), `bm_conflictology_country12` (the country's last 12 months) |

This uses parts that already work in Cypha: the DIF regressor's mixture prediction (`predict_mixture_scalar` and `predict_mixture_batch` in `regression.hpp`) and its uncertainty. The fatality forecaster is a client of the classifier. It does not need a new model family.

---

## 8. Monitor (FC9)

| Signal | Definition | Source |
|---|---|---|
| Surprise | Per theater-week, the CyphaLM log-loss of the observed tokens, as a z-score against the dev distribution of weekly log-loss | Sequence ensemble, mean over members |
| Shift | PSI between the root distribution of the last 4 weeks and train's, per theater | Token counts |
| Anomaly | The share of the week's dyad-rows with `is_ood`, and their mean `anomaly_score` | Escalation estimator, `PredictOut` |

- **The alarm rule.** An alarm fires when any signal passes its threshold. Thresholds are set on dev weeks with no documented shock, to a target false-alarm rate (EVALUATION §5).
- **State.** The monitor's state (window, EMAs and thresholds) is part of the bundle.
- **Ingest.** `/forecast/ingest` takes a raw event, updates the theater-week, and returns the three signals and the alarm. The caller no longer sends scores.

---

## 9. The bundle (FC11)

```
<name>.forecast/
  manifest.json            schema 1; created_at; origin; code commit; vocab hash; level-map hash;
                           data MANIFEST sha256; component list with sha256 each
  escalation.cypha         + escalation.preprocessor.json + escalation.calibration.json
  fatality.cypha           + fatality.regression_head.json + fatality.preprocessor.json
  seq_0.json, seq_0.hpbin  … seq_{K-1}.*   (save_cyphalm_model)
  tree.json                tree config, isotonic maps per (theater, k, h)
  monitor.json             thresholds, dev distributions, window state
  card.json                registry card, task "forecast"
```

- **Save and load** compose the existing paths (`Cypha::save`, `save_cyphalm_model`) and add only the manifest.
- **Load checks every sha256 and both hashes.** A mismatch is an error, and nothing is ever re-derived silently.
- **The card's fields:**
  - `task: "forecast"`;
  - `targets` (`fatalities_cm`, `escalation_dyad_year`, `events_theater_week`);
  - `horizons`;
  - `data_versions` (the manifest rows);
  - `backtest` (scores and intervals by task, with baseline scores beside them);
  - `stage`.

The proposed manifest schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://cypha.local/schemas/forecast_bundle.schema.json",
  "title": "Cypha forecast bundle manifest.json (proposed, lands with FC11)",
  "type": "object",
  "required": ["schema", "created_at", "origin", "vocab_hash", "level_map_hash", "data_manifest_sha256", "components"],
  "properties": {
    "schema": { "const": 1 },
    "created_at": { "type": "string", "format": "date-time" },
    "code_commit": { "type": "string" },
    "origin": { "type": "object", "required": ["kind", "year", "index"],
                "properties": { "kind": { "enum": ["year", "month", "week"] }, "year": { "type": "integer" }, "index": { "type": "integer" } } },
    "vocab_hash": { "type": "string" },
    "level_map_hash": { "type": "string" },
    "data_manifest_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
    "components": { "type": "array", "minItems": 1, "items": {
      "type": "object", "required": ["path", "role", "sha256"],
      "properties": { "path": { "type": "string" },
                      "role": { "enum": ["escalation", "fatality", "sequence", "tree", "monitor", "card", "sidecar"] },
                      "sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" } } } }
  }
}
```

---

## 10. Serving (FC12)

| Route | Body | Returns |
|---|---|---|
| `POST /forecast/load` | `{"bundle": "<name>"}`, resolved under a configured root only | `{"loaded": true, "manifest": {…}}` |
| `POST /forecast/predict` | `{"task": "fatalities_cm", "units": ["UKR", …], "months": ["2027-01", …], "quantiles": [0.05, 0.5, 0.95], "draws": false}` | Per unit-month: quantiles, mean, P(y > 0), and draws if asked |
| | `{"task": "events_theater_week", "theaters": ["UKR"], "levels": ["force_use"], "horizons": [1, 4, 13], "paths": 5}` | `EventProbability` rows; top paths with probability and first week at ForceUse |
| `POST /forecast/ingest` | One raw event: date, actor country codes, `EventRootCode`, QuadClass, Goldstein | The three signals, the alarm and the reason |
| `POST /forecast/monitor/reset` | — | As today |
| `GET /health`, `GET /metrics` | — | Adds `forecast_loaded`, the bundle name, `forecast_predictions`, `forecast_ingested` and `forecast_alarms` |

- **Locking.** All routes take `g_mu`.
- **Errors.** No bundle loaded gives 503. An unknown unit or theater gives 404. A malformed body gives 400.
- **Training.** `/forecast/run` leaves the serving binary (FORECAST_PLAN §9.6). Training is `cypha_forecast_run --data … --out <name>.forecast`.

---

## 11. Removed

| What | Why | Item |
|---|---|---|
| `validate_views_holdout` | Reads the truth (FC0 C1) | FC7 |
| `forecast_from_mean` | Ad hoc histogram from a mean (D3) | FC7 |
| `crps`, `ignorance_score` in `views_scoring.cpp` | Wrong (D1, D2) | FC6 |
| Theater temp files | Side effect on the host (E6) | FC10 |
| Client-supplied drift on `/forecast/ingest` | The server must score (H2) | FC9 |
| `C:/Users/odinl/...` fallback | Host path in a tool (C4) | FC13 |
| `expand_top_k`, `rare_event_boost` | Read nowhere (E2) | FC8, FC10 |
