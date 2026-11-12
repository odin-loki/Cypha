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

## Results (2026-11-12)

**Kept.** `context8k_bpc` **1.4962 → 1.4848** (−0.0114) on the standard holdout. Wall time +1%, peak RAM +13%. Answers unchanged.

### The search (1 MB tuning slice, scored on its second half)

| Step | bpc | Peak RAM |
|---|---:|---:|
| Defaults | 1.7501 | 1302 MB |
| Experts, round 1 (start 1.7360) | 1.7305 | 1551 MB |
| Experts, round 2, end | **1.7276** | 1552 MB |
| + mixers by replay over the tuned experts' dump | **1.7201** | — |
| T\* confirmed by a run | 1.7201 | 1475 MB |

- The new binary (main with O5) gives SM__default's stdout and dump unchanged under `cmp`, so the earlier trials stood.
- Tuned experts: `--bkn 10:4:2:1 --doc cache,cand=2048 --lstm-window 5 --ngram-cap 8 --ngram-lr 0.000625 --ngram-orders 1,2,3,4,5,6,7,8,12,16`. `--ngram-cap` and `--lstm-window` stopped at the ends of their ladders.
- Tuned mixers: `--full-mix-params apm=0.0331976,decay=0.0001,eps=2.31807e-05,floor=1.67652e-05,lr1=0.000564081,lr2=0.000150458,lrb=0.00181686 --meta-mix-params m1lr1=0.00545053,m1lr2=0.000200209,m2eta=0.340135,m2geta=0.831624,m2gshare=0.00149035,m2share=0.0014615,scales=0.5:1:2`. The restart from the best point won (1.7201 against the main search's 1.7205).
- Best over the RAM limit: 1.7349 at 1576 MB (11 n-gram orders); not kept.

### The 6 MiB check (dev, before the holdout)

| Run | bpc | Wall | Peak RAM |
|---|---:|---:|---:|
| S0, defaults | 1.5796 | 39:41 | 1546 MB |
| **T\*** | **1.5710** | 26:25 | 1754 MB |
| Tuned mixers alone, replayed over S0's dump | 1.5742 | — | — |

T\* is below S0, so E\* was not needed. The walls are from different loads on the shared PC.

### The final pair (standard holdout, 4 threads each, started together in one turn of the lock)

| | Default | Tuned | Change |
|---|---:|---:|---:|
| `context8k_bpc` | 1.4962 | **1.4848** | −0.0114 |
| Wall | 2910 s | 2938 s | +1.0% |
| CPU (user + system) | 5513 s | 6477 s | +17.5% |
| Peak RAM | 1533 MB | 1732 MB | +13.0% |
| Answer tallies (seen, change_first_letter, add_Thanks, paraphrase, heldout_dolly, wiki_fact) | 3/3, 3/3, 0/3, 0/3, 0/30, 0/8 | the same | — |

The rule holds on all four counts: below 1.4962, time within +20%, RAM within +20%, answers not worse.

### The default change (this branch; the lead merges)

- `context8k_tuned()` in `cyphalm_trie.cpp` gives `context8k` the tuned values for every flag not on the command line. Mixer params given on the command line are read after the tuned ones.
- `qa`, `sample` and `chat` keep their defaults.
- `context8k` with no flags: stdout `cmp` equal to the tuned run of the pair (1.4848).
- `qa`, 130 dev questions, before and after: the rows file is `cmp` equal. Stdout differs only in timing fields (build seconds, latencies).

### Sensitivity (§13.43; `bench/tune/t1_sensitivity.tsv`)

At θ\* on the small slice. Δbpc at ×0.5 / ×0.7 / ×1.4 / ×2:
- steep: `m1lr1` +0.0012 / +0.0002 / +0.0004 / +0.0012; `apm` +0.0005 / +0.0001 / +0.0002 / +0.0010; similar-context T +0.0013 / +0.0005 / +0.0002 / +0.0008;
- `--ngram-cap` still falling: −0.0017 at 4, +0.0021 at 16. It sits at the end of its ladder (8);
- `eps` still falling upward (−0.0006 at ×2);
- flat within ±0.0001: `lr2`, `m1lr2`, `m2eta`, `m2share`, `m2geta`, `m2gshare`, `--ngram-lr`;
- small: `lr1`, `lrb`, `floor`, `decay` (≤ 0.0003).

### Predictions

- §13.42: `context8k_bpc` down 0.01 to 0.03: **hit** (−0.0114). Time and RAM within the limits: **hit** (+1% wall, +13% RAM).
- The mixers add 0.002 to 0.005 on the tuned experts (small slice): **miss** (0.0075, more than predicted).
- The 6 MiB gain is a third to two thirds of the small one: **miss**, just under (0.0086 of 0.0300: 29%). The holdout gain (0.0114) is 38%.
- The lower `--ngram-cap` helps less with more data: **not settled**. It was not tested alone on 6 MiB. The tuned mixers alone give 0.0054 of T\*'s 0.0086 there. So the experts gain much less on 6 MiB than on 1 MB (0.0225 there), which fits the prediction but does not isolate the count limit.
