# Pipelining context8k's experts (GENERATION_PLAN §13.41 O2)

2026-11-05 · Odin Loch

**Odin:** "Keep looking. Come up with more. Try optimisation techniques … Go crazy on it."

Branch `wip/opt-pipeline`, from `origin/main` at 6530cb2. Odin's PC (WSL). The job: in `context8k` the text is known in advance, and the experts learn from the true bytes, not from the mixer. So each expert can run ahead in its own thread and hand its predictions to the mixer. The output must stay the same, byte for byte.

**Registered prediction (§13.41):** O2 is exact; the full run's wall time falls by at least 30%, bounded by the slowest expert, the LSTM.

## Where the time went before

(filled in below)

## Which experts can run ahead

(filled in below)

## What changed

(filled in below)

## Exactness checks

(filled in below)

## Time and memory

(filled in below)

## The prediction, scored

(filled in below)

## Risks before merging

(filled in below)
