# G1: keep the paragraph's words (K5's word cache and dynamic counts, retried on today's writer)

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/k5-cache2` (from main e897970d) · GENERATION_PLAN §13.48–13.64, K5_UPGRADES.md step 2

## Why

- The gap to human text is content continuity. Only 0.204 of the writer's sentences share a content word with the sentence before. Human text has 0.508 (§13.59, PARA_PLAN.md).
- Gate terms and form fixes did not move the readers: R2 read worse (§13.53), K64 did not clear the rule (§13.60), PP read the same (§13.62).
- K5 already has two ways to keep the words of the paragraph (K5_UPGRADES.md step 2):
  - `--k5-cache W`, `--k5-cache2 W2`, `--k5-cache-decay R`: a unigram and bigram cache of the prompt and the text so far, mixed into K5's distribution;
  - `--k5-dyn L`: the prompt's and the written text's n-grams (orders 2–5) added to K5's counts, for that continuation only.
- They were judged once, by SCRIBE only, on the 7 MB K5 (4 November). SCRIBE fell with the cache: the text got ",," and "..". No reader has seen them on today's fluent 337 MB writer.
- **Hypothesis:** on the fluent writer, a small cache lifts the prompt's and the paragraph's content words without breaking sentences. At 7 MB the sentences were already broken.
- No code changes. Only existing flags.

## Setup

- WSL clone `~/kc_src` at main e897970d. `cyphalm_trie` and `scribe` built with `g++ -O2` as `scripts/build_1tb_tools.sh` builds them (`libsais.o` linked into `scribe`), into `~/kc_bin`.
- Env: `CYPHALM_TRIE`, `SCRIBE`, `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.
- **The baseline, HLT** (today's default): `scripts/gen_eval.sh dev HLT OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`. It adds `--k5-more train_all.txt`, the 16-draft gate, the LDA gate term, the discourse planner and the honest fill itself.
- **Check first** (no arm is read until it holds): my build's HLT equals `bench/honest_dates2/runs/HLT.gen.txt` under `cmp`, MODEL line aside.
- At most 2 arms at once, 6 threads each; an arm starts only with at least 14 GB available.

## The arms (at most 6; each is HLT plus the flags shown)

| Arm | Flags | What it tries |
|---|---|---|
| C1 | `--k5-cache 0.02` | a small unigram cache, memory about 70 words (R 0.99) |
| C2 | `--k5-cache 0.05` | the same, stronger (7 MB's smallest weight) |
| C3 | `--k5-cache 0.02 --k5-cache-decay 0.995` | C1 with a longer memory (about 140 words), for the paragraph |
| B1 | `--k5-cache 0.02 --k5-cache2 0.05` | C1 plus the bigram cache: the prompt's word pairs ("New Zealand") |
| D1 | `--k5-dyn 30` | dynamic counts. K5 now counts from 48 times more text than at 7 MB, so L is raised from step 2's 1–10 |
| CD | the passing cache arm (C or B) that reads best, plus `--k5-dyn 30` | the one combination |

- CD runs only if D1 and at least one cache arm pass the screen. Otherwise it is not run.
- The cache's weights are small on purpose. At 7 MB, W 0.05 already cost 0.036 SCRIBE, and the cache's heaviest words are function words and punctuation. A small W mostly lifts rare words: a prompt word with a base probability of 1e-6 gets about 5e-4 at W 0.02, while "the" barely moves.

## The screen (dev, seed 0, 250 prompts)

**Measures:**
- SCRIBE v2 (95% interval) and v3.1, copy32, verbatim13, wall time, peak RAM, samples unlike HLT;
- fact precision: `python3 -I scripts/gen_fact_check.py run --v2 --out bench/k5_cache2/facts HLT=… X=…` (GEN_FACTS amendment A1);
- J2's measures with `bench/para_plan/hook_screen.py --surprise scripts/surprise_profile.py --train /mnt/d/cypha_llm/fuzzy/train.txt` (cut 460): the link to the sentence before (`on_prev`), stuffing (3rd+ use of a content word), top word share, kept_s1–s4, orphan pronouns, and **mean surprise** (`surp_mean`, WRITING_SIGNAL's screen; lower reads better between settings of one writer). Each arm minus HLT with a 95% paired interval. `surprise_profile.py` on main is unchanged since 2124363f.

**Reading:** I read 12 dev prompts the raters never see (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), each arm beside HLT. I look at two things: does a sentence take up the words of the one before or of the prompt, and does it repeat keywords. **Keyword repetition is the known failure** (§13.34: SCRIBE rose for topic stuffing, the raters marked it down). An arm that stays on topic by repeating the prompt's words fails the reading. So does one that writes ",," or broken clauses.

**An arm passes the screen if all hold:**
- verbatim13 is 0;
- copy32 at most 0.30;
- SCRIBE v2 at least HLT's minus 0.03;
- fact precision at least HLT's minus 0.05;
- by my reading, it reads at least as well as HLT on the 12 prompts, with no more keyword repetition.

Mean surprise and `on_prev` are reported, not ruled on.

**Candidates:** at most 2 that pass. Chosen by my reading first, then by `on_prev`, then by SCRIBE v2. An arm that changes fewer than 30 of the 50 rated prompts is not sent (too few to rate).

## The long reading

For HLT and each candidate: `cyphalm_trie sample` on the 250 dev prompts with `--bytes 1800 --k5-max-words 600` and the same seeds (K5_COHERENCE's long setup; `--plan-drop 75497472:100000000`, run from the repository root so the trimmed skeleton file is found). HLT_long is new; no long HLT file exists yet. The same measures at cut 1,500.

## The deciding judge

- `python scripts/blind_rate.py make --out bench/blind/cache_pair --ids bench/blind/llm_rating/prompts.json --arm HLT=… --arm X=… --mode pair --baseline HLT --seed 20261122`: the 50 prompts, four fresh blind Claude raters.
- Long: `--out bench/blind/cache_long --ids bench/blind/q3_long/prompts.json --seed 20261123 --cut 1500`, the 25 prompts K5_COHERENCE used, four more fresh raters.
- Seeds 20261122 and 20261123 were checked: no other round uses them.
- Packets, `key.b64` and `RATER_PROMPT.md` are committed and pushed before the raters run. `KEY.json` stays out of the repository.
- **The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Adopted** if it reads better by the rule on either reading (460 or 1,500) and its win share on the other is at least 0.5. The best such arm (higher lower bound first, then lower copy32) becomes the default; Odin can veto. Then GEN-A test is read once, for it only.

## Predictions

- The check (my HLT equals the committed HLT): **hit expected**.
- **C1:** `on_prev` rises from about 0.20 to 0.23–0.27; stuffing up at most 0.005; SCRIBE v2 within ±0.03; copy32 within ±0.02; fact precision within ±0.03; mean surprise up 0.03–0.15 bits (rare prompt words are surprising to a static trigram). It passes the reading.
- **C2:** `on_prev` 0.26–0.32; stuffing up 0.005–0.015; SCRIBE v2 down 0.01–0.05. It is the most likely to fail the reading for repetition or ",,".
- **C3:** about C1, with `on_prev` +0.01 over C1 at 1,500 characters.
- **B1:** `on_prev` about C1's; more prompt word pairs ("New Zealand"); copy32 within ±0.02.
- **D1:** fewer changes than the cache arms; `on_prev` within ±0.03 of HLT; SCRIBE v2 within ±0.02.
- Wall time up 20–60% for every arm: the cache and dynamic counts turn off O6's fast nucleus. Peak RAM 11.4 GB ± 0.5.
- **Raters:** the best candidate about 0.58 at 460 characters and 0.60 at 1,500. Neither clears the rule at 460 (a cache moves little in 460 characters); one may clear it long.

## Results

(Added after the runs, under this line.)
