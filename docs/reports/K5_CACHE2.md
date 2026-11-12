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

### The screen (2026-11-12; dev, seed 0, 250 prompts; files in `bench/k5_cache2/runs/`)

**Check:** my build's HLT equals `bench/honest_dates2/runs/HLT.gen.txt` under `cmp`, MODEL line aside. **Hit.**

| Arm | SCRIBE v2 [95%] | v3.1 | copy32 | verbatim13 | fact precision [95%] | wall | peak RAM | unlike HLT | screen |
|---|---|---|---|---|---|---|---|---|---|
| HLT | 0.595 [0.563, 0.630] | 0.539 | 0.232 | 0 | 0.935 [0.888, 0.974] | 7:32 | 11.3 GB | – | baseline |
| C1 cache 0.02 | 0.611 [0.579, 0.642] | 0.573 | 0.228 | 0 | 0.912 [0.860, 0.957] | 10:39 | 11.3 GB | 214 | **pass, candidate** |
| C2 cache 0.05 | 0.663 [0.632, 0.696] | 0.665 | 0.195 | 0 | 0.935 [0.888, 0.974] | 10:37 | 11.3 GB | 214 | fail (reading: stuffed, broken) |
| C3 cache 0.02, R 0.995 | 0.621 [0.585, 0.652] | 0.583 | 0.212 | 0 | 0.930 [0.883, 0.970] | 10:03 | 11.3 GB | 214 | pass, not chosen |
| B1 cache 0.02 + bigram 0.05 | 0.698 [0.667, 0.729] | 0.663 | 0.201 | 0 | 0.929 [0.883, 0.970] | 8:27 | 11.3 GB | 215 | fail (reading: stuffed, broken) |
| D1 dynamic counts 30 | 0.614 [0.582, 0.646] | 0.549 | 0.235 | 0 | 0.935 [0.888, 0.974] | 9:59 | 11.3 GB | 153 | pass, not sendable (25 of 50 rated prompts change) |
| CD = C1 + dynamic 30 | 0.642 [0.609, 0.673] | 0.591 | 0.224 | 0 | 0.924 [0.875, 0.967] | 10:56 | 9.2 GB | 214 | **pass, candidate** |

Two arms at once on a shared PC, 6 threads each. The fact floor is 0.885; the SCRIBE floor 0.565.

**J2's measures and mean surprise** (`bench/para_plan/hook_screen.py`, cut 460; `bench/k5_cache2/measures/s460.md`):

| Arm | on_prev [Δ 95%] | stuffing (3rd+ use) [Δ] | mean surprise, bits [Δ] | orphan pronouns | kept_s4+ |
|---|---|---|---|---|---|
| HLT | 0.200 | 0.0154 | 7.399 | 0.240 | 0.307 |
| C1 | 0.214 [+0.014: −0.010, +0.036] | 0.0170 [+0.0016: +0.0004, +0.0028] | 7.466 [+0.067: −0.013, +0.146] | 0.245 | 0.323 |
| C2 | 0.290 [+0.090: +0.063, +0.118] | 0.0253 [+0.0099: +0.0074, +0.0127] | 7.756 [+0.357: +0.267, +0.447] | 0.241 | 0.355 |
| C3 | 0.209 [+0.008: −0.010, +0.026] | 0.0166 [+0.0013: +0.0003, +0.0023] | 7.520 [+0.121: +0.040, +0.204] | 0.231 | 0.312 |
| B1 | 0.284 [+0.083: +0.059, +0.110] | 0.0246 [+0.0093: +0.0066, +0.0121] | 7.535 [+0.137: +0.049, +0.222] | 0.237 | 0.354 |
| D1 | 0.199 [−0.001: −0.012, +0.008] | 0.0157 [+0.0004: −0.0001, +0.0009] | 7.433 [+0.034: −0.014, +0.082] | 0.231 | 0.294 |
| CD | 0.213 [+0.012: −0.008, +0.033] | 0.0171 [+0.0018: +0.0005, +0.0030] | 7.528 [+0.130: +0.052, +0.213] | 0.254 | 0.347 |
| human (§13.59) | 0.508 | | | | |

**Reading** (dev 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, each arm beside HLT):
- **C2 and B1 raise the link by stuffing the prompt's words into broken clauses.** Dev 12 (C2): "Copper is boy", "Basil learns to pay taxes", "His cruel and unusual punishment". Dev 13 (C2): "travel is sect, rather than the account. church in the city … and the Church. The Catholic Church". Dev 9 (B1): "the seat of the diocese of trading", "The empirical, in which". Dev 4 (B1): "(Census 2001). Ethnic diversity." Dev 3 (B1): "the Food and. If that was". This is §13.34's failure. SCRIBE v2 rises most for them (+0.07, +0.10), and the reading fails them.
- **C1 and C3 read about as HLT.** They take up a prompt word now and then in a fitting place. Dev 13 (C1): "the Methodist Episcopal Church … The Christian churches in Scandinavia". Dev 12 (C1): "This legend is confronted". Some odd insertions remain ("his coronation in Census and Statistics", dev 4). No more repetition than HLT by eye. C3's longer memory reads the same as C1 at 460 characters.
- **D1** writes HLT's text word for word until mid-paragraph, then goes its own way. It reads as HLT, but it changes only 25 of the 50 rated prompts at 460 characters, so it is not sent.
- **CD** is C1 for the first sentences and differs from C1 on 30 of 50 rated prompts. It reads about as C1, with a few broken bits ("in Census). The Act in 1978", "population. In the) Atlantic").
- The planned sentences (facts) are the same in every arm. The cache changes the free K5 text after them.

**Candidates:** C1 and CD. All of C1, C3 and CD pass and read about the same. By the registered order (reading, then `on_prev`), C1 (0.214) and CD (0.213) rank above C3 (0.209). They differ from HLT on 43 of the 50 rated prompts each. **The CD rule:** D1 and the cache arms C1 and C3 passed; C1 and C3 read the same, so `on_prev` picked C1.

### The long reading (1,800 bytes, cut 1,500)

`--bytes 1800 --k5-max-words 600`, the same seeds. HLT_long 9:08 and 11.3 GB; C1_long 18:41 (beside other jobs); CD_long 12:29, 11.4 GB. CD_long's first run was killed partway, with every other process on the WSL VM, by something outside this job. It was rerun from scratch and finished.

| Arm | on_prev [Δ 95%] | stuffing [Δ] | mean surprise [Δ] | kept_s4+ | unlike HLT (of 25 rated) |
|---|---|---|---|---|---|
| HLT | 0.109 | 0.0184 | 7.379 | 0.308 | – |
| C1 | 0.119 [+0.010: +0.000, +0.021] | 0.0203 [+0.0019] | 7.450 [+0.071: +0.025, +0.121] | 0.315 | 24 |
| CD | 0.122 [+0.013: +0.002, +0.024] | 0.0217 [+0.0033] | 7.453 [+0.074: +0.029, +0.119] | 0.319 | 24 |

By reading (dev 4 and 12 at 1,500 characters), all three wander by the fourth sentence. C1 and CD hold a loose thread a little more often (dev 12: a woman's mother, her marriages, her husband), but the thread is the writer's own, not the prompt's.

### Packets (built before any rating)

- `bench/blind/cache_pair/`: HLT against C1 and CD, the 50 prompts of `bench/blind/llm_rating/prompts.json`, seed 20261122.
- `bench/blind/cache_long/`: HLT_long against C1_long and CD_long, the 25 long prompts (`prompts.json`, the same list as `q3_long`), seed 20261123, `--cut 1500`.
- `KEY.json` of each is kept outside the repository; `key.b64` and `RATER_PROMPT.md` are committed.

### Predictions so far

- The check equals HLT: **hit**.
- C1: `on_prev` 0.23–0.27 **miss** (0.214); stuffing at most +0.005 **hit** (+0.0016); SCRIBE v2 ±0.03 **hit** (+0.016); copy32 ±0.02 **hit** (−0.004); fact precision ±0.03 **hit** (−0.023); surprise +0.03 to +0.15 **hit** (+0.067); passes the reading **hit**.
- C2: `on_prev` 0.26–0.32 **hit** (0.290); stuffing +0.005 to +0.015 **hit** (+0.0099); SCRIBE v2 down 0.01–0.05 **miss** (+0.068); fails the reading **hit**.
- C3 about C1: **hit**. Its long reading was not run (not a candidate), so "+0.01 over C1 at 1,500" is not scored.
- B1: `on_prev` about C1's **miss** (0.284, stuffed); copy32 ±0.02 **miss** (−0.031).
- D1: fewer changes **hit** (153 against 214); `on_prev` ±0.03 **hit**; SCRIBE v2 ±0.02 **hit** (+0.019).
- Wall time +20–60%: **hit** for C1, C2, C3, D1, CD (+33% to +45%); **miss** for B1 (+12%). Two arms shared the PC with other jobs, so these times are rough.
- Peak RAM 11.4 ± 0.5 GB: **hit**, except CD's dev run (9.2 GB; its long run took 11.4 GB).
- The raters' predictions wait for the raters.

**What it says so far:** the cache does what it is meant to do only when it is strong, and then it stuffs. At a weight the sentences survive (0.02), the link to the sentence before moves by +0.01, a twentieth of the gap to human text. SCRIBE again rewards the stuffed arms most (§13.34).
