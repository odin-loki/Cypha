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

## Results

(to come)
