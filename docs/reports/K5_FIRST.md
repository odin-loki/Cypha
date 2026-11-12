# T4: CyphaLM's first sentence stays on the prompt's subject

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/k5-first` (from main c51f6c2b) · GENERATION_PLAN §13.48–13.54, [K5_COHERENCE](K5_COHERENCE.md)

## Why now

- T3 found that the drift starts in the first sentence (K5_COHERENCE.md, reading). In half the prompts read, the first written sentence is already off the subject: "Black powder arms …" → "Various Jews in Eastern European democracies".
- The first gate has only the prompt to go on, and every later chooser keeps the same first sentence.
- On today's files it is worse than T3's table shows. 77 of the 250 dev prompts get a plan, and a plan's first sentence is a fact sentence about the prompt's entity. On the other **173 prompts (plain K5), only 0.110 of first sentences hold a content word of the prompt** (human text: 0.601). 39 of the 50 rated prompts are plain K5.

| L1 (default), `coh_measures.py`, 460 characters | n | 1st sentence on the prompt | 2nd | 3rd | 4th+ | link to the sentence before | 3rd+ use of a word |
|---|---:|---:|---:|---:|---:|---:|---:|
| all prompts | 250 | 0.360 | 0.276 | 0.241 | 0.090 | 0.150 | 0.015 |
| plain K5 prompts | 173 | 0.110 | 0.046 | 0.064 | 0.067 | 0.038 | 0.000 |
| human, plain K5 prompts | 173 | 0.601 | 0.486 | 0.494 | 0.330 | 0.453 | 0.033 |

## What the first gate does today

- K5's context is the whole prompt (`start()` reads every prompt word), so the first sentence already follows the prompt's last words. Seeding with the prompt's last sentence would change nothing. It is not an arm.
- The plain prompts go to the z-scored gate (`gate_z`, since L1): 16 drafts, score = z(mean PPMI with the prompt) + z(LDA topic fit) − 5 × markup. Fluency has weight 0.

## One new flag, off by default

`--k5-first k=N,wp=W,wl=W,req=1` (`word_kn.hpp`, `gate_z`). It acts on the **first sentence only**, and only in the z-scored gate (today's default for plain prompts). A planned prompt's first sentence is the planner's and does not change.
- `k=N`: N drafts for the first sentence instead of 16. Drafts 0–15 are today's drafts (same streams), so N = 64 adds 48 more.
- `wp=W`: W more weight on z(PPMI with the prompt) (1 → 1 + W).
- `wl=W`: W more weight on z(topic) (1 → 1 + W).
- `req=1`: only drafts that hold a content word of the prompt compete, when any draft does. A content word: letters only, 4+ letters, lower-cased, not in `coh_measures.py`'s function-word list, not one of K5's 100 most frequent words. If no draft holds one, all compete as today.
- With every field 0 (the default) nothing is computed, the vectors are the same size, and the choice loop picks as before. The MODEL line is unchanged.
- The 12-word copy cap and the repeat ban are not touched.

## The baseline and the checks

**L1**, today's default: `scripts/gen_eval.sh dev L1 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `CYPHALM_TRIE` and `SCRIBE` built by `g++ -O2` as `scripts/build_1tb_tools.sh` builds them (libsais.o linked into scribe), `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

No arm is read until both hold:
- main's build (c51f6c2b) gives a GEN file equal to `bench/k5_coherence/runs/L1.gen.txt` under `cmp`, MODEL line aside;
- my build with no new flag (L1n) gives the same file. The selftest checks that the flag parses and is off by default.

## The arms (6; each is L1 plus the flag shown)

| Arm | Flag | Idea |
|---|---|---|
| K64 | `--k5-first k=64` | more drafts for the first sentence |
| WP2 | `--k5-first wp=2` | the prompt's PPMI three times as strong in the first gate |
| WL2 | `--k5-first wl=2` | the LDA topic term three times as strong in the first gate |
| RQ | `--k5-first req=1` | the first sentence must hold a prompt content word, when any of the 16 drafts does |
| RQ64 | `--k5-first k=64,req=1` | the same among 64 drafts |
| C | RQ64 + the better of WP2 and WL2 by the screen's order | only if WP2 or WL2 passes the screen; else not run |

Dev, seed 0, 250 prompts. At most 2 arms at once, 6 threads each, no start under 14 GB available.

## The screen (dev)

**Measures:** SCRIBE v2 (95% interval) and v3.1, copy32, verbatim13, wall time, peak RAM, samples unlike L1, and `coh_measures.py` at 460 characters, on all 250 prompts and on the 173 plain prompts.

**The first-sentence measure:** the share of first sentences holding a prompt content word (`coh_measures.py` s1), on the 173 plain prompts. `req` aims at this measure directly, so it alone proves nothing. Beside it, by my reading: of the 12 read prompts, how many first sentences are about the prompt's subject (the same thing, not just a shared word).

**Reading:** dev prompts 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13 (the raters never see them), each arm beside L1, at 460 characters. I am strict about keyword stuffing, which SCRIBE rewards (§13.34, §13.53): a first sentence that drops in a prompt word without being about the subject ("The powder of the Jews …") does not count as on the subject, and an arm that does this often fails the reading.

**An arm passes the screen if all hold:**
- verbatim13 is 0;
- copy32 at most 0.30;
- SCRIBE v2 at least L1's minus 0.03;
- by my reading, it reads at least as well as L1 on the 12 prompts, with no more keyword stuffing.

**Candidates:** at most 3 that pass, chosen by my reading first, then by the first-sentence count on the 12 read prompts, then s1 on the 173 plain prompts. An arm that changes fewer than 15 of the 50 rated prompts is not sent (too few to rate).

## The deciding judge

- `python scripts/blind_rate.py make --out bench/blind/first_pair --ids bench/blind/llm_rating/prompts.json --arm L1=... --arm X=... --mode pair --baseline L1 --seed 20261117`: the 50 prompts, 460 characters (the first sentence is much of what is read there), four fresh blind Claude raters.
- Packets, `key.b64` and `RATER_PROMPT.md` are committed and pushed before the raters run. `KEY.json` stays out of the repository.
- **The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Adopted:** the best arm that reads better by the rule (higher lower bound first, then lower copy32) becomes the default; Odin can veto. Then GEN-A test is read once, for it only.

## Predictions

- Both checks equal L1: **hit expected**.
- Only the plain prompts change: about 173 samples unlike L1 for each arm, fewer for WP2 and WL2 (the best draft often stays the best).
- **K64:** s1 on the plain prompts 0.110 → 0.15–0.20; SCRIBE v2 within ±0.02; wall +20–40%; reads about as L1.
- **WP2:** s1 → 0.15–0.25; SCRIBE v2 +0.01 to +0.03; some first sentences with odd rare words.
- **WL2:** s1 within +0.05 of L1; reads about as L1 (L1's term did not move the prompt share, §13.53).
- **RQ:** s1 → 0.35–0.50 (bounded by how often any of 16 drafts holds a prompt word); stuffing (3rd+ use) up at most 0.005; reads better at the start in 3–5 of the 12 read prompts.
- **RQ64:** s1 → 0.50–0.65; wall +20–40%; the most likely candidate. **Raters prefer RQ64 to L1 at about 0.60.**
- copy32 within ±0.02 of L1 for every arm; verbatim13 0; peak RAM 11.4 GB ± 0.3.
- Later sentences stay as off the subject as today: s2 and later up at most 0.03 (the drift after sentence 1 is T3's problem, not fixed here).

## Results

(Added after the runs, under this line.)

### Amendment (2026-11-12, after RQ ran and before any arm below runs)

- **Checks:** main's build and my build with no new flag (L1n) both equal `bench/k5_coherence/runs/L1.gen.txt` under `cmp`. **Hit.**
- **RQ's word test is too loose.** RQ changed 61 of the 173 plain prompts and raised s1 from 0.110 to 0.428. But the shared words are mostly common: "work", "seen", "above", "however", "used", "name". By my reading of dev 9 and 13 (the two read prompts RQ changed), neither new first sentence is about the prompt's subject. The s1 measure counts the same words, so it overstates RQ.
- **New field, `req=2`:** only the prompt's rare content words count (outside K5's 2000 most frequent words: e.g. powder, peroxide, census, Walpole, Gothenburg, Esperanto). Off by default; `req=0` and `req=1` are unchanged (RQ and RQ64 run as registered).
- **Arm C is dropped** (not run). In its place, still 6 arms: **RS64** `--k5-first k=64,req=2`.
- **Measure added:** s1 with rare words only (the share of plain prompts whose first sentence holds a prompt content word outside K5's 2000 most frequent). My reading count stays the main first-sentence measure.
- **Predictions for RS64:** changes 40–80 of the 173 plain prompts; rare-word s1 from L1's value up by 0.10–0.20; SCRIBE v2 within ±0.02; copy32 up at most 0.02 (rare words pull K5 into the one text that holds them); reads better at the start in 2–4 of the 12 read prompts. It is the most likely candidate now; raters about 0.60.
