# T1: tuning the prediction stack's settings (GENERATION_PLAN §13.42, §13.43)

2026-11-12 · Odin Loch

**Odin:** "tuned all these experts/mixers on a meg of data and kept them like that".

Branch `wip/tune-predict`. Driver `scripts/tune_predict.py`. Logs on the PC under `/mnt/d/cypha_llm/tune/t1`.

## Where it stood (2026-11-06, §13.47)

- Small slice (1 MiB to train, scored on enwik8 [1.0625 MiB, 1.125 MiB)): default 1.7501.
- Experts, within the +20% RAM limit: best 1.7337 (`--bkn 10:4:2:1 --doc cache,cand=2048 --lstm-window 12 --ngram-cap 30 --ngram-lr 0.0025`).
- Mixers alone, by replay over the default experts' dump (`sm0.fmx`): best 1.7454.
- 6 MiB slice (6 MiB to train, scored on the second half of enwik8 [6 MiB, 7 MiB)): default S0 1.5796.
- Not done: the end of the expert search, the mixers on the tuned experts, the 6 MiB check, the sensitivity table, the final pair.

## Plan for the rest (registered 2026-11-12, before any run)

§13.42's and §13.43's rules stand. These steps are new. Only the two tuning slices are read for choices; both end before the standard holdout [7 MiB, 8 MiB).

1. **New binary.** Built from this branch after merging main (state of §13.47, O5 included). Check: the default small run with `--full-mix-dump`, as `SM__default` ran. Its stdout must `cmp` equal to the old binary's, and so must the dump. If not, the cached trials are not reused and T1 stops to report.
2. **End of the expert search.** Resume the RAM-limited ladder search where it stopped, from the same start (`--bkn 8:4:2:1 --doc cache,cand=2048 --lstm-window 12 --ngram-cap 30 --ngram-lr 0.0025`, 1.7360). Open directions: `--ngram-cap` down, `--ngram-lr` down, `--lstm-window` down, `--bkn` up. Alternatives: every other `--ngram-orders` set. Finished trials are reused. It runs until no move improves.
3. **Mixers on the tuned experts.** One small run of the tuned experts writes their dump (`sm1.fmx`). The replay search starts from the best mixer point on `sm0.fmx` (1.7454): steps 1.41 then 1.19, at most 4 rounds each, categorical settings in its first round, then one restart from the best (spread 1.5, 3 rounds at step 1.19).
4. **Candidate T\*** = tuned experts + tuned mixers. One small run confirms it (the replay is exact, so it should match).
5. **6 MiB check (one turn of the lock each).** T\* on the 6 MiB slice. If T\* is not below S0 (1.5796), the experts alone (E\*) also run there. The set for the final pair is the lowest of these, if it is below S0. If none is below S0, T1 stops: not kept, no final pair. Also, for the report only: the tuned mixer point replayed over the default 6 MiB dump (`s0.fmx`).
6. **Sensitivity table (§13.43).** At θ\* = T\*: bits per byte at θ\* × {0.5, 0.7, 0.85, 1.15, 1.4, 2}, one setting at a time. Mixer settings by replay over `sm1.fmx`. `--ngram-cap`, `--ngram-lr` and the similar-context temperature by small runs. Written to `bench/tune/t1_sensitivity.tsv`.
7. **Final pair.** The chosen set and today's defaults, full default `context8k` on the standard holdout, 4 threads each, started together in one turn of `~/.cypha_full.lock`. Time = wall clock of each; RAM = peak RSS of each.
8. **If kept** (§13.42: `context8k_bpc` below 1.4962, time +20% at most, RAM +20% at most): the tuned values become `context8k`'s defaults only. `qa`, `sample` and `chat` keep theirs, so answers cannot change. Check: `qa` on the 130 dev questions before and after the change; its rows file must `cmp` equal. Then `context8k` with no flags must `cmp` equal to the tuned run's stdout. The default change stays on this branch; the lead merges.

**Predictions** (new; §13.42's stand: `context8k_bpc` down 0.01 to 0.03, time and RAM within the limits):
- The mixers add 0.002 to 0.005 on top of the tuned experts on the small slice.
- The gain on the 6 MiB slice is smaller than on the small one: between a third and two thirds of it.
- The small slice's lower `--ngram-cap` helps less with more data; it is the setting most likely to hurt on 6 MiB.
