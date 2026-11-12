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

### The screen (2026-11-12; dev, seed 0, 250 prompts; files in `bench/k5_first/runs/`)

Two arms at once on a shared PC (6 threads each).

| Arm | Flag | SCRIBE v2 [95%] | v3.1 | copy32 | verbatim13 | wall | peak RAM | samples unlike L1 (of the 50 rated) | screen |
|---|---|---|---|---|---|---|---|---|---|
| L1 | – | 0.610 [0.577, 0.645] | 0.546 | 0.242 | 0 | 5:38 | 11.3 GB | – | baseline |
| K64 | `k=64` | 0.620 [0.589, 0.651] | 0.548 | 0.238 | 0 | 8:19 | 11.3 GB | 135 (30) | **pass, candidate** |
| WP2 | `wp=2` | 0.615 [0.583, 0.648] | 0.545 | 0.240 | 0 | 8:07 | 11.3 GB | 48 (10) | pass, not sent (10 rated changes) |
| WL2 | `wl=2` | 0.615 [0.581, 0.645] | 0.543 | 0.242 | 0 | 8:24 | 11.3 GB | 43 (8) | pass, not sent (8 rated changes) |
| RQ | `req=1` | 0.623 [0.588, 0.655] | 0.557 | 0.237 | 0 | 6:13 | 11.3 GB | 61 (12) | pass, not sent (12 rated changes) |
| RQ64 | `k=64,req=1` | 0.626 [0.595, 0.660] | 0.555 | 0.240 | 0 | 7:59 | 11.3 GB | 153 (37) | **pass (just), candidate** |
| RS64 | `k=64,req=2` | 0.628 [0.598, 0.660] | 0.551 | 0.237 | 0 | 7:16 | 11.3 GB | 137 (32) | **pass, candidate** |

C was dropped by the amendment. RQ's SCRIBE row holds 249 samples.

**`coh_measures.py`** (460 characters; first sentence holds a prompt content word, by position):

| Arm | all 250: s1 | s2 | s3 | link to the sentence before | 3rd+ use | 173 plain: s1 | s2 | s3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 0.360 | 0.276 | 0.241 | 0.150 | 0.015 | 0.110 | 0.046 | 0.064 |
| K64 | 0.400 | 0.276 | 0.241 | 0.144 | 0.015 | 0.168 | 0.046 | 0.064 |
| WP2 | 0.364 | 0.284 | 0.249 | 0.153 | 0.015 | 0.116 | 0.058 | 0.076 |
| WL2 | 0.348 | 0.280 | 0.249 | 0.149 | 0.016 | 0.092 | 0.052 | 0.076 |
| RQ | 0.580 | 0.340 | 0.250 | 0.155 | 0.015 | 0.428 | 0.139 | 0.076 |
| RQ64 | 0.804 | 0.312 | 0.256 | 0.156 | 0.015 | 0.751 | 0.098 | 0.083 |
| RS64 | 0.468 | 0.280 | 0.239 | 0.145 | 0.015 | 0.266 | 0.052 | 0.059 |
| human | 0.632 | 0.488 | 0.510 | 0.460 | 0.032 | 0.601 | 0.486 | 0.494 |

**The shared words are mostly common words.** For RQ64 on the plain prompts the most frequent shared words are "number", "another", "seen", "year", "work", "group". For RS64 (rare words only in the filter) the first sentences still share common ones too ("government", "year", "used"), beside rarer ones ("protons", "granite", "abelian"). So s1 overstates every req arm. The rare-word s1 of the amendment was not computed: K5's word ranks are not at hand outside the binary, and the reading below is the main measure.

**Reading** (dev 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 460 characters, each arm beside L1). First sentences about the prompt's subject:
- **L1: 3 of 12** (dev 5 Whig and Grenville, dev 7 Chalmers in Sweden, dev 11 Zamenhof and Esperanto). Dev 0 is planned (New Zealand people, not Holdens). Dev 3 drops in "Food" twice ("Greece Food writing is preferred to the Food Lion"): that is not on the subject. These five first sentences are the same in every arm.
- **K64: 4 of 12.** It changes dev 1, 2, 4, 9, 10, 12, 13. Dev 13 now opens on religion ("in contrast to the older faith … divinity … Pope Pius XII"), a weak yes. The others are new sentences, as far off the subject as L1's ("The larger southern New Guinea …" for the India census). No keyword drops. Reads as L1.
- **WP2: 3 of 12**, **WL2: 3 of 12.** Few changes. Reads as L1.
- **RQ: 3 of 12.** Dev 9 and 13 change, neither on the subject (dev 9 gets "above" from "discussed above").
- **RQ64: 3 of 12, with two word drops.** Dev 1 "As the size of a small, straight …" (size, from "the size of the granules"); dev 2 "… a senior position" (position, from the grappling prompt). Dev 9 "free trade" is near the gravity model of trade, half a yes. The rest as L1. It reads as L1; the drops are single words, not repetition (3rd+ use 0.015), so it passes, just.
- **RS64: 4 of 12, one word drop.** Mostly K64's sentences (64 drafts, and the rare-word filter often finds nothing). Dev 2 "to hold the dissolved calcite … to escape Nazism" drops two grappling words without the subject. Dev 13 as K64. Reads as L1.
- **None of the arms fixes the start.** In the plain prompts, few of 64 drafts are about the prompt's subject; a filter on shared words mostly picks a draft with a common or dropped-in word. The gain is at most one prompt in 12. Later sentences are as before (s2 and s3 move by under 0.03, except RQ's s2, where the first sentence ends later).

**Candidates for the raters:** K64, RS64 and RQ64 (all pass; they change 30–37 of the 50 rated prompts). By the registered order: reading (all three as L1), then the first-sentence count on the 12 (K64 4, RS64 4, RQ64 3), then plain s1 (RS64 0.266 over K64 0.168): **RS64, K64, RQ64**. WP2, WL2 and RQ change too few rated prompts.

### Packets (built before any rating)

- `bench/blind/first_pair/`: L1 against K64, RS64 and RQ64, the 50 prompts of `bench/blind/llm_rating/prompts.json`, `--mode pair --baseline L1 --seed 20261117`, 460 characters.
- `KEY.json` is kept outside the repository; `key.b64` and `RATER_PROMPT.md` are committed.

### Predictions so far

- Both checks equal L1: **hit**.
- About 173 samples unlike L1 per arm, fewer for WP2 and WL2: **miss** (61–153 for the others; WP2 48, WL2 43).
- K64: s1 0.15–0.20: **hit** (0.168); SCRIBE v2 within ±0.02: **hit** (+0.010); wall +20–40%: **miss** (+47%, shared PC); reads as L1: **hit**.
- WP2: s1 0.15–0.25: **miss** (0.116); SCRIBE v2 +0.01 to +0.03: **miss** (+0.005); odd rare words: **miss** (none seen).
- WL2: s1 within +0.05: **hit** (−0.018); reads as L1: **hit**.
- RQ: s1 0.35–0.50: **hit** (0.428), but by common words; stuffing up at most 0.005: **hit** (0.000); reads better at the start in 3–5 of 12: **miss** (0).
- RQ64: s1 0.50–0.65: **miss** (0.751, higher, by common words); wall +20–40%: **hit** (+42%, about).
- RS64: 40–80 changed: **miss** (137); s1 up 0.10–0.20 (rare-word s1 not computed; plain s1 +0.156); SCRIBE v2 within ±0.02: **hit** (+0.018); copy32 up at most 0.02: **hit** (−0.004); reads better in 2–4 of 12: **miss** (1).
- copy32 within ±0.02, verbatim13 0, peak RAM 11.4 ± 0.3 GB: **hit** for all.
- s2 and later up at most 0.03: **hit**, but for RQ's s2 (+0.093).
- The raters' prediction (RQ64 about 0.60, RS64 about 0.60) waits for the raters.
