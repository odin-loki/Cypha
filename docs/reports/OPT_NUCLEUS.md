# O6: K5's lazy exact nucleus (GENERATION_PLAN §13.41)

2026-11-12 · Odin Loch

**Odin:** "Keep looking. Come up with more. Try optimisation techniques … Go crazy on it."

The job (§13.41, wave 3, O6): K5 never builds and sorts its full 32,000-word distribution to draw a word. Exact; K5 sampling time down at least 50%. Branch `wip/opt-nucleus`. Only `native/include/cypha/lm/word_kn.hpp` changes.

The code was written on 2026-11-06 and stopped on Odin's word (§13.47), not verified. It resumed on 2026-11-12. The branch first merged main 9cbd277, then the lead's branch `claude/elated-elgamal-f7e3c6` at 77579f3 (§13.48's `CYPHA_K5_VOCAB_BYTES`, W1b and H1b, which make K5's counts come from 337 MB). Both merges were clean.

## What changed

`step_word` used to build p over the whole vocabulary, sort all of it, and cut the candidates at `cand_mass` or `cand_max`. Now:
- `full_pick` is that old code, unchanged.
- `fast_pick` makes the same choice without the sort. It applies to K5's own walk at temperature 1 (no dynamic counts, no cache, no class mix, no topic factor). Elsewhere `full_pick` runs.
  - A word that no context of the walk holds has q = floor × the walk's back-off weights, rounded as `dist` rounds it. The seen words get the walk's sums in `dist`'s order. So every q is `dist`'s value, bit for bit.
  - z is summed over the vocabulary in id order, as before. Beside it, the mass and count of each band of q (its top 14 bits).
  - From the top band down, the first band that reaches the candidate mass (with a margin of 1e-9) or `cand_max` gives a bound. Only the words above it are sorted, by the old order (p descending, then the salted hash). The cut is then made as before.
  - If the cut is not reached above the bound, or the bound would take a word of q 0, or the 12-word cap leaves no candidate, `fast_pick` changes nothing and `full_pick` makes the choice.
  - The repeat ban checks all candidates against one prepared view of the text (`RepeatBan::Batch`), with the same answers as `banned()`.
- `CYPHA_K5_NUCLEUS=full` uses `full_pick` only. `CYPHA_K5_NUCLEUS=check` runs both picks from the same state and stream at every step, and stops the run at the first difference (words, order, p bit for bit, bans, nucleus, word, piece, log-probability, counts).
- `cyphalm_trie selftest` checks the batch ban against `banned()` on 96,000 random pieces, and `fast_pick` against `full_pick` on two toy models, five option sets and three seeds.

## Checks (registered 2026-11-12, before any run)

Old: the lead's branch at 77579f3. New: this branch. Both built the same way, `g++ -O2` as `scripts/build_1tb_tools.sh` builds them, with the same `CYPHA_SIMVEC_DEFAULT` path. Both run from this branch's checkout, so the planner reads the same files. Each pair is started together, each side pinned to its own 4 logical CPUs, `K5_THREADS=4`.

GEN-A dev, 250 prompts, `scripts/gen_eval.sh dev LABEL OUT --prompts 250 --k5 TEXT` (WIKI, ENWIK8 `/mnt/d/cypha_llm/enwik8`, SCRIBE_VEC `/mnt/d/cypha_llm/scribe/v32`):
1. **7 MB, the planner** (the default): `--k5 /mnt/d/cypha_llm/fuzzy/train.txt`.
2. **7 MB, `--no-k5-plan`.**
3. **337 MB, the planner** (W1b, §13.48): `CYPHA_K5_VOCAB_BYTES=7053155 --k5 train_w1.txt`. `train_w1.txt` is a copy of the lead's `~/gq_runs/train_w1.txt` (336,656,418 bytes, `cmp` equal).
4. **337 MB, `--no-k5-plan`**, the same text.
5. The new binary with `CYPHA_K5_NUCLEUS=check` on 1 and 2 (any difference stops the run).
6. `cyphalm_trie selftest`, old and new.
7. `context8k` small (`--train 1048576 --sample 1179648`): default, `--no-meta-mix`, `--no-full-mix`, old against new.

Exact means `cmp` equal: the GEN file, the `sample` stdout, both SCRIBE reports (v2 and v3.1) and the summary. The `sample` stderr is compared with its timing lines removed (`generated … in … s`, `seconds`, `thread-s`). If any check differs, O6 is reported failed.

Measures: K5's sampling time (stderr's `k5: generated N bytes in X s`), `sample`'s wall time and peak RAM (`/usr/bin/time -v`), and how many steps `fast_pick` made (check mode counts them).

## Results (2026-11-12)

**O6 is exact.** Every check is `cmp` equal. Both setups also reproduce the lead's files: old's 7 MB planner GEN file equals §13.48's D0, and old's 337 MB planner GEN file equals W1b (MODEL line aside).

| Check | What was compared | Result |
|---|---|---|
| 1. 7 MB, the planner | GEN file, stdout, SCRIBE v2 and v3.1, summary; stderr (timing removed, sorted) | `cmp` equal (8 of 8 files) |
| 2. 7 MB, `--no-k5-plan` | the same | `cmp` equal (8 of 8) |
| 3. 337 MB, the planner (W1b) | the same | `cmp` equal (8 of 8) |
| 4. 337 MB, `--no-k5-plan` | the same | `cmp` equal (8 of 8) |
| 5. `CYPHA_K5_NUCLEUS=check` on 1 and 2 | both picks at every step | no difference (rc 0); each count line says `fast_pick` chose at every step so far (5 lines of 65,536 steps); GEN files equal to 1's and 2's |
| 6. `cyphalm_trie selftest` | — | pass, old and new |
| 7. `context8k` small: default, `--no-meta-mix`, `--no-full-mix` | stdout (260, 245, 233 lines; 1.7501, 1.7633, 1.8384) | `cmp` equal |

The stderr logs are compared sorted because `sample` prints its PLAN lines from 4 threads in whatever order they finish.

**Speed** (old and new started together, 4 CPUs each, `K5_THREADS=4`; other sessions loaded the PC, load 20–42):

| Setup | K5 sampling, old → new | `sample` wall, old → new | `sample` user CPU, old → new | Peak RAM, old → new |
|---|---:|---:|---:|---:|
| 7 MB, the planner | 147.6 → 61.3 s (**−58%**) | 172 → 88 s (−49%) | 604 → 205 s (−66%) | 616 → 623 MB |
| 7 MB, `--no-k5-plan` | 201.4 → 66.2 s (**−67%**) | 226 → 91 s (−60%) | 693 → 183 s (−74%) | 144 → 148 MB |
| 337 MB, the planner | 267.4 → 178.7 s (**−33%**) | 465 → 367 s (−21%) | 891 → 505 s (−43%) | 11,015 → 11,022 MB |
| 337 MB, `--no-k5-plan` | 148.0 → 44.8 s (**−70%**) | 333 → 226 s (−32%) | 768 → 352 s (−54%) | 3,748 → 3,754 MB |

- K5 sampling is stderr's line `k5: generated 150000 bytes in X s`. The `sample` wall also holds building K5 (at 337 MB about 3 minutes, unchanged).
- **337 MB with the planner gains least.** There the planner arm takes about 120–130 s more of K5's time than `--no-k5-plan`, in both binaries. That work is not the draw that O6 changes.
- `context8k` does not use K5. Its small runs took the same time (wall 11:27 → 11:33, 9:56 → 9:47, 7:41 → 7:42).
- RAM: about 7 MB more (the word pieces' table and the scratch).

**Prediction** (§13.41): exact; K5 sampling time down at least 50%.
- Exact: **hit.**
- At least 50%: **hit** on the registered setup, today's default (7 MB, the planner): −58%; −67% without the planner. At 337 MB: −70% without the planner, but **−33% with it (a miss there)**.

**Kept.** O6 is always on: it makes the same choice, faster. `CYPHA_K5_NUCLEUS=full` brings back the old path; `=check` runs both. Not on main: that is the lead's call.

Runs: `/mnt/d/cypha_llm/opt/o6/` (`gen/` the pairs with their `.cmp`, `ctx/`, `logs/runs.log`, `bin/` the scripts).
