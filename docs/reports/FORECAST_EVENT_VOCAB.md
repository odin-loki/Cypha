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
