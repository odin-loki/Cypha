# Pipelining context8k's experts (GENERATION_PLAN §13.41 O2)

2026-11-05 · Odin Loch

**Odin:** "Keep looking. Come up with more. Try optimisation techniques … Go crazy on it."

Branch `wip/opt-pipeline`, from `origin/main` at 6530cb2. Odin's PC (WSL). The job: in `context8k` the text is known in advance, and the experts learn from the true bytes, not from the mixer. So each expert can run ahead in its own thread and hand its predictions to the mixer. The output must stay the same, byte for byte.

**Registered prediction (§13.41):** O2 is exact; the full run's wall time falls by at least 30%, bounded by the slowest expert, the LSTM.

## How the checks run (2026-11-12, before any run) · Odin Loch

Resumed after §13.47. `origin/main` (9cbd277) is merged in. Old is main 9cbd277, built from `git archive`; new is this branch. Same compiler (GCC 13), Release, Ninja.

- **CPUs.** The pipeline needs threads by design, so each O2 run gets **6 CPUs** (three cores, both hyperthreads, by `taskset`), not the usual 4. The old binary in each pair gets the same 6, so the pair is fair. This is not in §13.41; it is written here first.
- **Small checks** (`--train 1048576 --sample 1179648`), old and new started together: the default, `--no-meta-mix`, `--no-full-mix`. Plus the new binary with `--no-pipeline` and with `CYPHA_PIPELINE_THREADS=1`, each `cmp` against the old default.
- **Dev GEN files** (`cyphalm_trie sample`, as `scripts/gen_eval.sh dev` runs it, 250 prompts, `--k5 fuzzy/train.txt`): the planner (the default) and `--no-k5-plan`. GEN file and stdout `cmp`. O2 changes only `context8k`, but `sample` is in the same file.
- **Full run:** old and new started together in one turn of `flock ~/.cypha_full.lock`, 6 CPUs each. `context8k_bpc` 1.4962 must reproduce, and the stdouts must be `cmp` equal. The wall times of this pair score the prediction.
- Exact means `cmp` of the whole stdout. If any check differs, O2 is reported failed.

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
