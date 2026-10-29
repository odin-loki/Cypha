# The forecast's event vocabulary on CyphaLM

**Odin Loch** · 2026-10-29 · HANDOFF §5.10 · follows [SLOW_TEST_TRIAGE.md](SLOW_TEST_TRIAGE.md) ("Fix")

(The data, the candidates, the measures, the decision rules and the predictions below were written and committed before any candidate was run. Results are added under them.)

**Question.** The forecast's event vocabulary has 842 ids: PAD, UNK, and 7 theaters × 6 escalation levels × 20 CAMEO roots. The forecast trains a hp-wrapped CyphaLM model, and hp reads bytes. So any event id of 256 or more (any theater after Taiwan) throws `token id out of vocab range` in training. Should events be written as bytes, or should the forecast get a model that is not byte-level? This is decided by measurement (PLAN ledger 55), open to Odin's veto.

**Rule 10.** Forecasting answers no question, so rule 10 does not bind it. Every candidate is still a counting model: hp counts bytes in contexts and mixes them; the n-gram candidate counts events. Nothing here uses a neural network at run time.

## A loader bug found first

The CSVs under `bench/data/forecast/` are UTF-16 (little-endian; `gml_mid.csv` big-endian), with no byte-order mark. `load_dispute_data` read them as bytes, so every field held NULs:
- every GDELT event loaded as CAMEO 0 in the global theater, one token (`GLB_VERBAL_C1`, id 22), 51 times;
- every MID record loaded as zeros (year 0, hostility 1).

So the pipeline trained its sequence model on one repeated token, and its node estimator on constant features. That is why "the smoke data stays below 256" (SLOW_TEST_TRIAGE). `load_dispute_data` now decodes UTF-16 in either byte order, with or without a mark. With the data read correctly, the repo's 51 events include ids 360, 481, 575 and 716, and the pipeline throws as §5.10 says.

## Data

The repo's sample (51 events) is too small to choose a model. GDELT 1.0 daily event exports are fetched with `scripts/forecast_gdelt_window.py`:
- **Source:** `https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip`, the files `scripts/fetch_forecast_data.ps1 -Bulk` already fetches.
- **Conversion:** the same theater mapping as that script (actor names matched against country patterns; everything else global), with one fix: the CAMEO code is `EventBaseCode` (3 digits), not `EventCode`, so the root is the code / 10 for every row. `EventCode` has 4-digit forms ("1823"), whose root would come out as 182 and be clamped to 20.
- **Size:** the first 20,000 converted rows of each day, in file order; 14 days. Each day's zip SHA-256 is in the manifest.
- **Windows (fixed now):**
  - **train:** 2026-09-29 to 2026-10-08 (10 days, 200,000 events);
  - **dev:** 2026-10-09 and 2026-10-10 (40,000 events), for every choice;
  - **test:** 2026-10-11 and 2026-10-12 (40,000 events), read once. For the test read the model trains on train, reads dev with learning on (unscored), then is scored on test.
- **Theaters over the 14 days:** global 222,944; Middle East 22,000; Ukraine 16,961; South Asia 8,014; Taiwan 7,631; Korea 2,450. About one event in five has an id of 256 or more. No row maps to Africa (the script has no pattern for it).

## Candidates

Every candidate gives a distribution over the 842 ids, renormalised over valid ids only, so a sampled event always decodes.

| Name | What it is |
|---|---|
| **hp_pair** (bytes, two per event, by meaning) | One CyphaLM hp model. An event is two bytes: hi = 2 + theater × 6 + level (PAD and UNK: 0 and 1), lo = 64 + CAMEO root − 1 (specials: 64). P(event) = P(hi) × P(lo given hi), each renormalised over valid bytes |
| **hp_base256** (bytes, two per event, by number) | One hp model. hi = id / 256 (0–3), lo = id mod 256. The control for hp_pair: the same model with codes that carry no meaning |
| **hp_theater** (bytes, factorised by theater) | One hp model reads the theater of each event (7 theaters and a special symbol); one hp model per theater reads that theater's own events as its local index (0–119). P(event) = P(theater) × P(local given that theater's stream). Specials are uniform over PAD and UNK |
| **ngram** (not byte-level) | Interpolated Kneser–Ney over event ids, discount 0.75, uniform base over the 842 ids, counts updated online. Order 2, 3 or 4, chosen on dev |
| unigram, uniform | Baselines: ngram at order 1; log2 842 = 9.717 bits |

**hp configuration**, the same for every hp candidate: the production recipe the pipeline uses, at `hp_table_bits` 18 (279 MB a model; the recipe's default 22 is 1.2 GB). hp_theater's eight models use 18 bits for the theater stream and 16 for each theater. Learning is on while scoring (prequential), as when the forecast reads a live stream.

## Measures

- **Bits per event** (primary): mean −log2 P(event) over the scored window, prequential. Intervals: block bootstrap over blocks of 1,000 consecutive events, 2,000 resamples; the same for differences between candidates.
- **On every 20th event:** the full distribution, giving top-1 accuracy, the Brier score over 842 classes, and the expected calibration error of the top-1 probability (15 equal-width bins).
- **Rollout validity:** the share of invalid events among 10,000 masked samples (0 by construction), and among unmasked samples from the raw byte distributions (hp candidates only), at the every-20th positions.
- **Cost:** training and scoring time per event, and peak RSS, each candidate in its own process.

## Decision rules

1. **Dev choice.** The candidate with the lowest dev bits per event. If an hp candidate is within 0.02 bits per event of the ngram candidate, the hp candidate is chosen (it keeps the forecast on CyphaLM's model, as the module was built). ngram's order is chosen on dev the same way (lowest bits per event).
2. **Fitness.** The chosen candidate must give 0 invalid events in 10,000 masked samples, run the pipeline on the repo's data inside the smoke test's limit, and peak no higher than the current pipeline (1.2 GB).
3. **Test read, once.** Every candidate is reported. The dev choice is adopted: it makes all 842 ids work. If on test another candidate beats it by more than 0.02 bits per event, with the bootstrap interval of the difference above 0, the report says so and the choice goes to Odin.

## Predictions

| # | Prediction |
|---|---|
| F1 | hp_pair and ngram are within 0.15 bits per event of each other on dev; ngram is the lower of the two |
| F2 | hp_pair beats hp_base256 by at least 0.05 bits per event on dev (codes with meaning share statistics across theaters) |
| F3 | hp_theater is worse than hp_pair by at least 0.05 bits per event on dev (separate streams lose the context across theaters) |
| F4 | Every context model beats unigram by at least 0.1 bits per event, and unigram beats uniform by at least 3 bits per event |
| F5 | Unmasked samples from the trained hp_pair model are invalid less than 1% of the time |
| F6 | ngram trains and scores at least 100 times faster per event than hp_pair, and peaks under 100 MB |
| F7 | The chosen candidate's top-1 accuracy on test is between 0.10 and 0.30 |

## Results on dev (2026-10-29)

Tool: `forecast_vocab_bench` at commit a18b16d. Train 200,000 events, dev 40,000, every candidate in its own process. The machine was shared (load average 25 to 35 on 4 cores), so times are slower than on a quiet machine.

| Candidate | Bits per event [95%] | Minus ngram order 2 [95%] | Top-1 | Brier | ECE | Raw draws invalid | Train, µs per event | Score, µs per event | Peak |
|---|---|---|---|---|---|---|---:|---:|---:|
| uniform | 9.7177 | +6.1631 | 0.000 | 0.999 | 0.001 | 0 | 0 | 1 | 56 MB |
| unigram (ngram order 1) | 4.6860 [4.6051, 4.7688] | +1.1315 [+1.0660, +1.2002] | 0.203 | 0.926 | 0.019 | 0 | 0.3 | 1.3 | 56 MB |
| **ngram order 2** | **3.5546 [3.4947, 3.6128]** | — | 0.409 | 0.795 | 0.028 | 0 | 0.3 | 1.6 | 56 MB |
| ngram order 3 | 3.6102 [3.5446, 3.6732] | +0.0556 [+0.0472, +0.0647] | 0.402 | 0.801 | 0.029 | 0 | 0.8 | 2.8 | 56 MB |
| ngram order 4 | 3.7296 [3.6580, 3.7985] | +0.1750 [+0.1615, +0.1887] | 0.379 | 0.819 | 0.032 | 0 | 1.9 | 3.8 | 56 MB |
| **hp_pair** | **3.5555 [3.4928, 3.6173]** | **+0.0009 [−0.0117, +0.0122]** | 0.395 | 0.802 | 0.052 | 0.006 | 608 | 2,493 | 297 MB |
| hp_base256 | 3.5873 [3.5254, 3.6479] | +0.0327 [+0.0203, +0.0431] | 0.388 | 0.808 | 0.058 | 0.005 | 477 | 3,373 | 297 MB |
| hp_theater | 3.5756 [3.5117, 3.6372] | +0.0211 [+0.0087, +0.0319] | 0.405 | 0.801 | 0.058 | 0.003 | 682 | 3,474 | 1,283 MB |

- Top-1, Brier and ECE are on every 20th dev event (2,000). Masked samples: 0 invalid of 10,000 for every candidate.
- **Context matters a lot:** one event of context takes unigram's 4.69 bits to 3.55. Consecutive GDELT events often come from the same article and share actors and theater. More context than one event hurts the counts (order 3, order 4).
- **hp_pair and ngram order 2 tie.** The difference is 0.0009 bits per event, with an interval across 0.
- **Codes with meaning help hp a little:** hp_pair beats hp_base256 by 0.032 bits per event, and splitting streams by theater costs 0.020.

### The dev choice (rule 1)

- ngram's order: 2 (lowest of 2, 3 and 4).
- The lowest overall is ngram order 2. hp_pair is within 0.02 bits per event of it, so **rule 1 chooses hp_pair**: it keeps the forecast on CyphaLM's model.

### Fitness (rule 2), on the repo's data

| Pipeline | Peak RSS | CPU time | Wall time (load 27 to 35) |
|---|---:|---:|---:|
| main (reads every event as one token) | 919 MB | — | 37 s |
| hp_pair at table bits 18 (as measured; the new default) | **292 MB** | about 60 s | 272 to 315 s |
| hp_pair at the production recipe's 22 bits | 1,219 MB | 69 s | 506 s |

- The registered hp configuration for every candidate was table bits 18, so the pipeline adopts hp_pair as measured: bits 18 (`forecast_event_model_default`). It peaks at 292 MB, below main's pipeline, and gives the same outputs as the 22-bit model on the repo's data.
- 0 invalid of 10,000 masked samples.
- **Time:** about 60 s of CPU, inside the smoke test's 300 s limit. On this loaded machine the wall time came near the limit. The cost is the rollout tree: each new node takes the full 842-way distribution, which for hp_pair is 44 hp branches (693 nodes on the repo's data). Main's tree had 34 nodes, because every event was one token.
- At the production recipe's 22 bits the pipeline would peak at 1,219 MB, over the registered 1.2 GB. That configuration is not adopted.

### Predictions scored on dev

| # | Prediction | Result |
|---|---|---|
| F1 | hp_pair and ngram within 0.15 bits per event; ngram the lower | **Hit:** 0.0009 apart; ngram order 2 the lower |
| F2 | hp_pair beats hp_base256 by at least 0.05 | **Miss:** by 0.033 |
| F3 | hp_theater worse than hp_pair by at least 0.05 | **Miss:** worse by 0.020 |
| F4 | Context models beat unigram by at least 0.1; unigram beats uniform by at least 3 | **Hit:** by 0.96 to 1.13; by 5.03 |
| F5 | Unmasked samples from trained hp_pair invalid under 1% | **Hit:** 0.6% |
| F6 | ngram at least 100 times faster than hp_pair, under 100 MB | **Hit:** about 2,000 times faster to train and 1,500 times to score; 56 MB |
| F7 | The chosen candidate's top-1 on test is 0.10 to 0.30 | Scored at the test read |

## The test read (once, after commit 5a57f62)

Each candidate trained on train, read dev with learning on (unscored), then was scored on test (2026-10-11 and 10-12, 40,000 events). Load average about 25 to 30.

| Candidate | Bits per event [95%] | Minus hp_pair [95%] | Top-1 | Brier | ECE | Raw draws invalid | Train, µs per event | Score, µs per event | Peak |
|---|---|---|---|---|---|---|---:|---:|---:|
| uniform | 9.7177 | +5.9733 [+5.9280, +6.0159] | 0.000 | 0.999 | 0.001 | 0 | 0 | 1 | 56 MB |
| unigram | 4.9302 [4.8680, 4.9920] | +1.1858 [+1.1291, +1.2428] | 0.201 | 0.930 | 0.019 | 0 | 0.2 | 1 | 56 MB |
| ngram order 2 | 3.7412 [3.6965, 3.7883] | −0.0032 [−0.0106, +0.0042] | 0.382 | 0.815 | 0.010 | 0 | 0.2 | 1 | 56 MB |
| **hp_pair (the dev choice)** | **3.7444 [3.7017, 3.7897]** | — | 0.368 | 0.821 | 0.046 | 0.007 | 523 | 2,760 | 297 MB |
| hp_base256 | 3.7752 [3.7302, 3.8220] | +0.0308 [+0.0251, +0.0365] | 0.365 | 0.823 | 0.047 | 0.007 | 384 | 2,858 | 297 MB |
| hp_theater | 3.7626 [3.7183, 3.8086] | +0.0181 [+0.0105, +0.0250] | 0.378 | 0.821 | 0.050 | 0.003 | 511 | 2,682 | 1,281 MB |

- **Rule 3:** no candidate beats hp_pair by more than 0.02 bits per event with its interval above 0. ngram order 2 ties it again (−0.0032, interval across 0). **The dev choice stands: hp_pair at table bits 18.**
- The test window is harder than dev for every model (about +0.19 bits per event); the ordering is the same.
- **Calibration:** ngram's top-1 probability is better calibrated (ECE 0.010 against hp_pair's 0.046). hp is a little overconfident on events.
- Masked samples: 0 invalid of 10,000 for every candidate.

### Prediction F7, scored

| # | Prediction | Result |
|---|---|---|
| F7 | The chosen candidate's top-1 on test is 0.10 to 0.30 | **Miss:** 0.368. One event of context predicts the next exactly more often than expected, because consecutive GDELT events often repeat (the same article) |

Predictions over all: 4 hits (F1, F4, F5, F6), 3 misses (F2, F3, F7).

## After the test read: a speed change

- The rollout tree takes a full 842-way distribution at every new node, which for hp_pair is 44 hp branches. On the repo's data that is 693 nodes, so `native_forecast_smoke` timed out at 300 s on this loaded machine (load 25).
- hp's bit-tree fan-out skips bytes at or above the size asked for. So hp_pair now asks for bytes 0–43 (hi) and 0–83 (lo), and hp_theater for 0–7 and 0–119; raw draws still take all 256. On one state the values below the cut equal the full fan-out exactly.
- **Not bit-exact over a stream:** a rerun of dev with the change gives 3.555459 bits per event, against 3.555452. 13,914 of 40,000 per-event values moved, by at most 0.0002 bits (mean 7e-6). So the fan-out touches some hp state that a smaller fan-out touches differently. That is about 0.01% of the 0.02 decision margin, and no decision changes. The tables above are the registered runs.
- With it, `native_forecast_smoke` passes in 166 s at load 29; the full distribution is about twice as fast.

## The decision

**The forecast's sequence model is hp_pair: each event is written as two CyphaLM bytes, theater × level then CAMEO root, at hp table bits 18.** All 842 ids train, score and roll out; rollouts only ever produce valid events. It ties a Kneser–Ney event bigram in bits per event on dev and test. It is about 1,500 times slower, but it keeps the forecast on CyphaLM's model, as rule 1 says. It is open to Odin's veto: ngram order 2 is the measured alternative. It is equally accurate, better calibrated, 56 MB, and microseconds per event.

## Checks

- `native_forecast_event_model_test` (new, in CI): 37 checks.
  - What it checks: codes, UTF-16, every kind's normalisation, scoring, sampling, rewind and digest, and Kneser–Ney against a direct count.
  - Five planted bugs each make it fail: Kneser–Ney adding continuations for seen n-grams; no renormalisation over valid bytes; a rewind that does nothing; the CAMEO root off by one; UTF-16 big-endian not detected.
- `native_forecast_smoke` (cypha_slow): every kind trained on a cycle of ids 300, 450, 600 and 750 continues it in rollouts (share 1.0; a cold model 0.0). Learning stays unchanged through rollouts, and the same seed gives the same tree. The pipeline on the repo's data trains on ids up to 797 and builds a 693-node tree.
- `native_forecast_drift_backtest`, `native_forecast_gdelt_poll`, `native_views_leaderboard_bench`: pass on the decoded data.
- The 9 CTests from HANDOFF §4.4 pass.

## Also found

- **The node estimator** now sees real MID features (28 records) and scores 0 of its held-out records. On main it scored 1 on the constant all-zero features. That is outside this change.
- **`scripts/fetch_forecast_data.ps1`** took the CAMEO code from `EventCode`, whose 4-digit codes give a wrong root. It now takes `EventBaseCode`, as `forecast_gdelt_window.py` does.
- No converter maps any event to the Africa theater; its 120 ids are never seen in data.

## How to reproduce

```bash
python3 -I scripts/forecast_gdelt_window.py DATA 20260929 20260930 20261001 20261002 20261003 20261004 20261005 \
  20261006 20261007 20261008 20261009 20261010 20261011 20261012 --max-per-day 20000
TR=$(ls DATA/gdelt_2026092[9].csv DATA/gdelt_20260930.csv DATA/gdelt_2026100[1-8].csv | paste -sd,)
DV=DATA/gdelt_20261009.csv,DATA/gdelt_20261010.csv; TE=DATA/gdelt_20261011.csv,DATA/gdelt_20261012.csv
forecast_vocab_bench --kind ngram --order 2 --train $TR --dev $DV --out dev/ngram2
forecast_vocab_bench --kind hp_pair --hp-bits 18 --train $TR --dev $DV --out dev/hp_pair
forecast_vocab_bench --kind hp_pair --hp-bits 18 --train $TR --dev $DV --test $TE --read-test --out test/hp_pair
python3 -I scripts/forecast_vocab_compare.py dev ngram2 hp_pair --ref ngram2
```

The days' zip SHA-256s are in `FORECAST_EVENT_VOCAB.manifest.tsv`.
