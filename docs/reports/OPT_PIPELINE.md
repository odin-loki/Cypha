# Pipelining context8k's experts (GENERATION_PLAN §13.41 O2)

2026-11-05, finished 2026-11-12 · Odin Loch

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

The old binary (main 9cbd277), small setup, default, on 6 CPUs (its stderr):
- the trie trains on the main thread first; nothing else runs then;
- then, one after another: InfiniGram's index, the class expert (10.2 s), the similar-context expert, built twice (14.6 s and 13.8 s), the online n-gram's pretraining (88.8 s with its tables), the LSTM alone (85.3 s);
- then the full mix (84.8 s), whose main thread mixes (82.1 s busy) while the static, similar-context, LSTM and n-gram threads fill its blocks one ahead;
- 565 s in all, 146% CPU.

## Which experts can run ahead

None of the default experts reads the mixer, its weights or its output. Each learns from the true bytes only. So each can run as soon as what it reads exists:
- **before the trie is trained** (they read only the wiki bytes and the chat): InfiniGram's index and its five caps' passes, the class expert, byte Kneser–Ney, the similar-context expert (one pass, its rows kept), the document experts, the copy expert, the n-gram's pretraining, and the LSTM;
- **after the trie:** the trie's own rows, then the chain and the full mix, which read them all in the old order;
- **the meta mixer** reads only what the full mix hands its hook, so it runs on its own thread, reading copies of each hand-over in order.
- The GLN (`--gln`) reads the chain's loss, so it stays in its place.

## What changed

- `native/include/cypha/lm/expert_pipeline.hpp` (new): the plumbing only. A task on its own thread, a progress count, sparse rows (a 256-byte distribution without its +0.0 entries, every value the computed double), a bounded ring of row blocks, and two loops over independent rows or jobs.
- `native/tools/lm/cyphalm_trie.cpp`, `context8k`: each expert's pass is a job that fills its own arrays. With the pipeline on (the default) the jobs start at once, on their own threads, while the trie trains. The chain, the final layer and the full mix then read the arrays in the old order, with the old arithmetic.
- The full mix's blocks come through a ring of 4 blocks of 512 rows (`--pipeline N`), so its producers run up to 4 blocks ahead.
- The similar-context expert is built once, not twice; the full mix reads its kept rows (15 MB on the small setup).
- Independent passes (InfiniGram's caps, the class expert, Kneser–Ney, the caps' fits) are split over `CYPHA_PIPELINE_THREADS` threads (3).
- `--no-pipeline` runs every pass in place, in the old order.

## Exactness checks

Old: main 9cbd277 (`git archive`); new: this branch, main merged. GCC 13, Release, Ninja. Each pair started together, each run on its own 6 CPUs (`taskset`). Exact means `cmp` of the whole stdout.

| Check | What was compared | Result |
|---|---|---|
| small, default | stdout, 260 lines; `context8k_bpc` 1.7501 | `cmp` equal |
| small, `--no-meta-mix` | stdout, 245 lines; 1.7633 | `cmp` equal |
| small, `--no-full-mix` | stdout, 233 lines; 1.8384 | `cmp` equal |
| small, new `--no-pipeline`, against the old default | stdout | `cmp` equal |
| small, new `CYPHA_PIPELINE_THREADS=1`, against the old default | stdout | `cmp` equal |
| dev GEN, the planner (default), 250 prompts, `--k5 fuzzy/train.txt` | GEN file (501 lines), stdout | `cmp` equal |
| dev GEN, `--no-k5-plan` | GEN file (501 lines), stdout | `cmp` equal |
| **full `context8k`, default**, old and new together in one turn of the lock | stdout; `context8k_bpc` | `cmp` equal; **1.4962** |

**O2 is exact.** Every check is `cmp` equal, and the full run reproduces `context8k_bpc` 1.4962.

## Time and memory

Old and new started together, 6 CPUs each. Other sessions loaded the PC (load 10–45), so compare within a pair only.

| Run | Wall, old → new | User CPU, old → new | Peak RSS, old → new |
|---|---:|---:|---:|
| small, default | 565.0 → 403.7 s (**−28.5%**) | 709.7 → 687.6 s | 1,312 → 1,411 MB (+7.5%) |
| small, `--no-meta-mix` | 511.8 → 376.0 s (−26.5%) | 627.3 → 579.7 s | 914 → 1,003 MB (+9.7%) |
| small, `--no-full-mix` | 420.0 → 281.5 s (−33.0%) | 459.3 → 440.9 s | 671 → 738 MB (+9.9%) |
| **full, default** | 1,960.3 → 712.3 s (**−63.7%**) | 3,178.0 → 2,290.8 s (−28%) | 1,533 → 1,905 MB (+24%) |

- **The full run now waits for the similar-context expert.** Its one pass took 680 s and was ready at 697 s; the trie was packed at 344 s. The old run had the same expert busy 738 s inside the full mix, after the trie, the n-gram and the LSTM had run one by one (full mix 752.9 s, its main thread waited 233 s). The new full mix took 342 s, of which 45 s waiting for the similar-context rows.
- **The small run waits for the trie.** In the small default run the trie was trained and packed at 344 s; every expert's pass was done long before (the slowest, the n-gram's pretraining, 57.6 s, the LSTM 53.5 s). The full mix then took 53.4 s (old 84.8 s): its mixer waited 0.3 s for blocks, and the meta mixer, on its own thread, is no longer in the mixer's loop.
- **The bound is not the LSTM**, as §13.41 guessed. It is the similar-context expert in the full run, and the trie's training in the small run. Both are serial.
- Alone, without the old binary beside it: `--no-pipeline` 607.9 s, `CYPHA_PIPELINE_THREADS=1` 467.9 s (same stdout). These two ran beside each other, not beside the old binary, so they are not timings of record.
- The RAM grows 7–10% on the small setup and 24% on the full run (the experts' arrays and the similar-context rows, 139 MB on the full run, are held at once).

## The prediction, scored

| Prediction (§13.41) | Result | |
|---|---|---|
| O2 exact | every check `cmp` equal; the full run 1.4962 | **hit** |
| the full run's wall time down at least 30%, bounded by the LSTM | full run 1,960 → 712 s, **−63.7%** (pair started together, 6 CPUs each); small −28.5%. The bound is the similar-context expert, not the LSTM | **hit** (the reason given was wrong) |

## Risks before merging

- The pipeline is on by default. `--no-pipeline` gives the old order; both give the same output.
- More threads: each run uses as many CPUs as it is given. On a PC with few cores it gains less.
- The peak RAM rises 7–10% (small) to 24% (full: 1.53 → 1.91 GB).
- O3's patch for this file (`o2_cyphalm_trie.patch`, InfiniGram's buckets) is not in this branch; it would need refreshing against it.

## What to use

Keep it, on by default: exact, and the full `context8k` run is 2.75× faster on 6 CPUs. Merging to main is Odin's call. Next speed lead: the similar-context expert's pass (680 s, serial) now sets the full run's time.
