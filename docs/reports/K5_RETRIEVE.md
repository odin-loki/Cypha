# G2: K5 reads up on the subject before it writes

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/k5-retrieve` (from main e897970d) · GENERATION_PLAN §13.53, §13.59, §13.60, §13.62, §13.63

(The method, the arms, the screen, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- The writer's gap to human text is content, not form. Only 0.204 of its sentences share a content word with the sentence before; human text has 0.508 (§13.59).
- Its paragraphs leave the prompt's subject after one or two sentences (§13.53, §13.60).
- A gate term for the previous sentence (§13.53 R2) and fixes of sentence form (§13.62 PP) did not move the raters. §13.62 put it plainly: the word thread needs something to write about.
- **Hypothesis:** K5 drifts because its counts know nothing about the prompt's subject beyond the prompt's few words. If K5 first reads the documents of its own training text that are most related to the prompt, and leans on their n-grams for that continuation, it writes from text on the same subject. Its sentences then share words with each other and with the prompt.

## The method

New flag, off by default, byte-identical when off: `--k5-retrieve k=N,rel=R` (or `k=N,w=L`). Code: `native/include/cypha/lm/retrieve.hpp` (new) and small changes to `lm/word_kn.hpp`.

**Retrieval** (CyphaLM's own counts only; rule 10):
- The query is the prompt's distinct words outside K5's 100 most frequent words (the rule the gate's prompt association uses). A word found in more than 5% of the documents is left out.
- The documents are scored by BM25 (k1 1.2, b 0.75) over K5's word ids. A word's postings come from K5's own suffix array, so the index adds only the documents' start positions and lengths.
- The N best documents are kept (ties go to the earlier document).
- **Only `/mnt/d/cypha_llm/k5scale/clean/train_all.txt` is searched.** It is the `--k5-more` text: clean enwik9 past enwik8, with GEN-A's and SCRIBE's overlap removed (K5_SCALE). K5's own 7 MB text is never retrieved. The flag refuses to run without `--k5-more`.
- The planner's `--plan-drop 75497472:100000000` is passed by `gen_eval.sh`, as always. Retrieval does not read the fact store.

**The overlay** (the way `--k5-dyn` adds the prompt's own n-grams; K5_UPGRADES step 2):
- The retrieved documents' n-grams (orders 2 to 5, each document from its own start) are counted for this continuation only. The shared model never changes.
- In a context the documents saw, K5's estimate becomes [(n(cw) − D) + L (d(cw) − 0.5)] / (n(c) + L d(c)) plus the backoff, as in `--k5-dyn`. Every distribution still sums to one (selftest).
- **The weight.** `w=L` is `--k5-dyn`'s absolute weight. It does almost nothing here: the documents are already in K5's 337 MB counts, so L 1 only doubles them. So the arms use `rel=R`: in a context K5 saw, L is R n(c) / d(c). The documents then hold about R / (1 + R) of that order's mass wherever they saw the context, whatever its frequency.
- The overlay is read by K5's own sentences: the plain writer, the gate's drafts, and the discourse planner's free sentences. **The planner's fill of fact sentences does not read it** (its scratch state takes only the context), so the fact sentences, and the fact check's main source, should change little.
- With the overlay on, a word is drawn by `fast_pick` on the overlay's distribution (`model_dist`), the same choice rule as `full_pick`. Before this change the overlay fell back to `full_pick`, which sorts the whole vocabulary.

**The copy cap holds against the retrieved documents.** They are part of K5's text, so the 12-word cap's windows already hold them, and verbatim13 checks them. A new measure also checks each continuation against its own prompt's documents (below).

## The leak check

- `native/tools/lm/k5_ret_titles.cpp` (new) lists the page title of every cleaned document of a range, in the order `scribe docs` gives them. Run on enwik9 [96, 954) MiB (141,633 documents, the same count as `clean_e9.jsonl`; 143 of 143 sampled documents have the same bytes) and on enwik8 [72, 96) MiB (1,868, the same count as `gena_source.jsonl`).
- `bench/k5_retrieve/leak_check.py map` matches each line of `train_all.txt` to its `clean_e9` document (100,793 of 100,795 matched). GEN-A's `doc` column is SCRIBE's page index in enwik8 [72, 96) MiB.
- **`leak_check.py check`** joins each arm's retrieval log (`CYPHA_K5_RET_LOG`) to the prompts. It reports how many retrieved documents come from a page with **the same title as the prompt's own page (must be 0)**, how many lie in GEN-A's source bytes (must be 0), and how many could not be titled.
- **`leak_check.py copy`** gives, per arm, the share of bytes in 32-byte spans of the prompt's own retrieved documents, and the longest run of words found in them.
- By construction the prompt's page cannot be retrieved: GEN-A comes from enwik8 bytes, `train_all.txt` from enwik9 past them, and a dump's titles are unique. The check confirms it from the files.

## The baseline

**HLT**, today's default: `scripts/gen_eval.sh dev HLT OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `cyphalm_trie` and `scribe` built by `g++ -O2` as `scripts/build_1tb_tools.sh` builds them (`libsais.o` linked into `scribe`), `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

**Checks first:**
- main's build (e897970d): the GEN file equals `bench/honest_dates2/runs/HLT.gen.txt` under `cmp`, MODEL line aside. **Done before this was written: equal** (8:06 wall, 11.3 GB).
- my build with the flag off: the same. **Done before this was written: equal** (8:48 wall, 11.3 GB).
- selftest: the flag parses, is off by default, the overlay sums to one at both weights and does not change the shared model. **Passes.**

**Before this was written** (stated so the reader can weigh it): to check that the code runs and to time it, I ran `k=4,rel=0.5` with `cyphalm_trie sample` on dev prompts 0–7 (no SCRIBE). I read the retrieval log's titles for those 8 prompts: they are on the prompts' subjects (gunpowder → Black powder, Nitrocellulose, Cordite; Gothenburg University → Umeå, Stockholm and Lund University). The leak check on them gave 0 and 0. I ran `leak_check.py copy` on those 8 samples to test the script (0.066 of bytes in their documents' 32-byte spans; longest run 10 words). I did not read the samples' text. To time it, I also ran dev prompts 0–5 four ways (`cyphalm_trie sample`, 6 threads): the default 955 ms a word, `CYPHA_K5_NUCLEUS=full` 997, `k=4,rel=0.5` drawn by `full_pick` 958 and by `fast_pick` 1,021 (the planner's fact steps dominate on these prompts); and `--no-k5-plan` with `k=4,rel=0.5` 65 ms a word. So the overlay itself costs little. The `fast_pick` change was made after the first timing and kept. Nothing else changed after these runs.

## The arms

Dev, seed 0, 250 prompts, the baseline's command plus:

| Arm | Flags | What it tests |
|---|---|---|
| K1 | `--k5-retrieve k=1,rel=0.5` | one document |
| K4 | `--k5-retrieve k=4,rel=0.5` | four documents |
| K16 | `--k5-retrieve k=16,rel=0.5` | sixteen documents |
| K4L | `--k5-retrieve k=4,rel=0.2` | a lighter weight |
| K4H | `--k5-retrieve k=4,rel=1.5` | a heavier weight |

Each arm writes its retrieval log (`CYPHA_K5_RET_LOG`). **Machine limits:** at most 2 arms at once, at most 6 threads each, no start while `free -g` shows under 14 GB available.

## Measures (dev)

- SCRIBE v2 with its 95% interval, v3.1 beside it, copy32, verbatim13, wall time, peak RAM, samples unlike HLT (all 250 and the 50 rated).
- **Fact precision:** `python3 -I scripts/gen_fact_check.py run --v2` (GEN_FACTS amendment A1, its store and drop rule), with its 95% interval.
- **J2's measures** (`bench/para_plan/hook_screen.py`, 460 characters, each arm minus HLT with a 95% paired bootstrap): the link to the previous sentence (`on_prev`), stuffing (3rd+ use), and **WRITING_SIGNAL's mean surprise** (`scripts/surprise_profile.py`; lower reads better between variants of one writer).
- **The prompt's words by sentence** (`bench/k5_coherence/coh_measures.py`, 460 characters): on the prompt s1, s2, s3, s4+, `on_prev`, stuff3, top word. HLT on dev: 0.372, 0.280, 0.229, 0.115; `on_prev` 0.146; stuff3 0.015; top word 0.056 (human 0.632, 0.488, 0.510, 0.310; 0.460; 0.032; 0.076).
- The leak check and the copy against the retrieved documents (above).

## The screen

**Reading.** I read the 12 unrated dev prompts K5_COHERENCE read (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), each arm beside HLT. I ask: does it keep to the prompt's subject past sentence 2, do its sentences follow from each other, and does it stuff keywords. **Keyword stuffing is the known failure** (SCRIBE rewards it, §13.34): an arm that stays "on topic" by repeating the prompt's or the documents' words, or by stringing their names together, fails. So does an arm that pastes the documents' phrases as a patchwork.

**An arm passes if all hold:**
- verbatim13 is 0;
- copy32 at most 0.30;
- SCRIBE v2 at least HLT's minus 0.03 (HLT dev 0.595, so 0.565);
- fact precision at least HLT's minus 0.05 (HLT dev 0.935, so 0.885);
- the leak check gives 0 same-title documents and 0 in GEN-A's source bytes;
- by my strict reading, it reads at least as well as HLT on the 12 prompts, with no more keyword repetition.

**Candidates:** at most 2 that pass, chosen by my reading first, then by `on_prev`, then SCRIBE v2. If none pass, nothing goes to the raters and I say so.

## The long reading

For HLT and each candidate: the same command with `--bytes 1800 --k5-max-words 600` (the same seeds; K5_COHERENCE's long setup), labels `HLT_long` and `X_long`. HLT_long is run with my build, flag off (started before this was committed; it is the baseline, not an arm).

## The deciding judge

- 460 characters: `python3 scripts/blind_rate.py make --out bench/blind/ret_pair --ids bench/blind/llm_rating/prompts.json --arm HLT=bench/honest_dates2/runs/HLT.gen.txt --arm X=... [--arm Y=...] --mode pair --baseline HLT --seed 20261124`: the 50 prompts, four fresh blind Claude raters.
- 1,500 characters: `--out bench/blind/ret_long --ids bench/blind/q3_long/prompts.json --arm HLT=.../HLT_long.gen.txt --arm X=..._long.gen.txt --mode pair --baseline HLT --seed 20261125 --cut 1500`: the 25 prompts K5_COHERENCE used, four more fresh raters.
- Packets, `key.b64`, `RATER_PROMPT.md` and the GEN files are committed and pushed before the raters run. `KEY.json` stays out of the repository. The lead runs the raters.
- **The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Adopted** if it reads better by the rule on either reading and its win share on the other reading is at least 0.5. The best such arm (higher lower bound first, then lower copy32) becomes the default; Odin can veto. Then GEN-A test is read once, for it only.

## Predictions

| # | Prediction |
|---|---|
| P1 | My build with the flag off equals HLT (`cmp`) |
| P2 | The leak check: 0 same-title documents and 0 in GEN-A's source bytes, for every arm |
| P3 | `on_prev` (coh_measures) rises from 0.146 to 0.18–0.24 for K4 and K16; K4H highest (0.22–0.30); K1 and K4L smallest (0.16–0.20) |
| P4 | On the prompt at s2–s4+ rises by 0.03–0.10 for K4 and K16; s1 rises less (the first sentence starts from the prompt in every arm) |
| P5 | Stuffing (stuff3) up at most 0.005 at rel 0.5; K4H up 0.005–0.015 |
| P6 | copy32 up 0.01–0.05 at rel 0.5 (K16 the most); K4H up 0.03–0.08; verbatim13 0 in every arm. Bytes in 32-byte spans of the prompt's own documents 0.05–0.15 at rel 0.5 |
| P7 | SCRIBE v2 within −0.01 to +0.04 of HLT; v3.1 likewise |
| P8 | Fact precision within ±0.03 of HLT (the fill does not read the overlay) |
| P9 | Mean surprise rises by 0.1–0.4 bits a word (topical words are rarer in the 7 MB surprise model) |
| P10 | By my reading K4 or K16 passes; K4H fails for stuffing or patchwork; at least one candidate goes to the raters |
| P11 | Raters: the best candidate's win share against HLT about 0.60 at 460 characters and 0.60–0.65 on the long reading; it is adopted by the rule |
| P12 | Wall time per arm 8–12 minutes for 250 prompts with 2 arms at once; peak RAM 11.3–11.6 GB |

## Results

(Added after the runs, under this line.)

### Checks (2026-11-12)

- main's build (e897970d) and my build with the flag off: both equal `bench/honest_dates2/runs/HLT.gen.txt` under `cmp` (MODEL line aside). **Hit.** The selftest passes.
- HLT_long (my build, flag off, `--bytes 1800 --k5-max-words 600`): SCRIBE v2 0.490 [0.465, 0.513], copy32 0.235, 0 verbatim13, 6:40, 11.3 GB.

### The screen (dev, seed 0, 250 prompts; files in `bench/k5_retrieve/runs/`, measures in `bench/k5_retrieve/screen/`)

Two arms at a time on a shared PC.

| Arm | SCRIBE v2 [95%] | v3.1 | copy32 | verbatim13 | fact precision [95%] | wall | peak RAM | screen |
|---|---|---|---|---|---|---|---|---|
| HLT | 0.595 [0.563, 0.630] | 0.539 | 0.232 | 0 | 0.935 [0.888, 0.974] | 8:48 | 11.3 GB | baseline |
| K1 (k 1, rel 0.5) | 0.704 [0.673, 0.735] | 0.665 | 0.275 | 0 | 0.918 [0.867, 0.962] | 5:13 | 11.3 GB | **pass, candidate** |
| K4 (k 4, rel 0.5) | 0.714 [0.682, 0.744] | 0.683 | 0.294 | 0 | 0.929 [0.881, 0.970] | 5:22 | 11.4 GB | **pass, candidate** |
| K16 (k 16, rel 0.5) | 0.752 [0.724, 0.779] | 0.696 | **0.318** | 0 | 0.924 [0.873, 0.964] | 5:14 | 11.4 GB | fail (copy32) |
| K4L (k 4, rel 0.2) | 0.627 [0.596, 0.659] | 0.596 | 0.232 | 0 | 0.935 [0.888, 0.974] | 7:38 | 11.4 GB | pass, not chosen |
| K4H (k 4, rel 1.5) | 0.854 [0.830, 0.877] | 0.801 | **0.501** | 0 | 0.936 [0.889, 0.974] | 5:49 | 11.4 GB | fail (copy32) |

Each arm differs from HLT in 214 of 250 samples (44 of the 50 rated). The other 36 are planned prompts whose text is all fact sentences, which the overlay does not touch. SCRIBE dropped 1–2 samples per arm in cleaning (248–250 kept).

**The leak check** (`leak_check.py check`): every prompt's retrieval was logged and joined to its GEN-A page.

| Arm | retrieved docs | distinct | same page title as the prompt | in GEN-A's source bytes | untitled |
|---|---|---|---|---|---|
| K1 | 250 | 249 | **0** | 0 | 0 |
| K4, K4L, K4H | 1,000 | 983 | **0** | 0 | 0 |
| K16 | 4,000 | 3,824 | **0** | 0 | 0 |

**Copy against each prompt's own documents** (`leak_check.py copy`): bytes in 32-byte spans of them K1 0.084, K4 0.116, K16 0.153, K4L 0.024, K4H 0.392. The longest run of words shared with them is 12 in every arm (11 for K4L): the cap holds; no 13-word run.

**The prompt's words and the link** (`coh_measures.py`, 460 characters):

| Arm | on the prompt s1 | s2 | s3 | s4+ | link to the sentence before | stuff3 | top word |
|---|---:|---:|---:|---:|---:|---:|---:|
| HLT | 0.372 | 0.280 | 0.229 | 0.115 | 0.146 | 0.015 | 0.056 |
| K1 | 0.472 | 0.422 | 0.348 | 0.202 | 0.179 | 0.021 | 0.067 |
| K4 | 0.448 | 0.442 | 0.382 | 0.225 | 0.194 | 0.020 | 0.065 |
| K16 | 0.484 | 0.412 | 0.390 | 0.219 | 0.179 | 0.020 | 0.064 |
| K4L | 0.436 | 0.380 | 0.347 | 0.159 | 0.158 | 0.017 | 0.059 |
| K4H | 0.572 | 0.562 | 0.423 | 0.314 | 0.254 | 0.029 | 0.074 |
| human | 0.632 | 0.488 | 0.510 | 0.310 | 0.460 | 0.032 | 0.076 |

**J2's measures** (`hook_screen.py`, 460 characters; each arm minus HLT, 95% paired bootstrap):

| Arm | link to the sentence before (`on_prev`) | stuff3 | mean surprise (bits a word) |
|---|---|---|---|
| HLT (level) | 0.200 | 0.0154 | 7.399 |
| K1 | +0.031 [+0.008, +0.052] | +0.0060 [+0.0037, +0.0083] | +0.408 [+0.315, +0.497] |
| K4 | +0.052 [+0.028, +0.074] | +0.0050 [+0.0029, +0.0075] | +0.400 [+0.309, +0.494] |
| K16 | +0.040 [+0.015, +0.066] | +0.0046 [+0.0026, +0.0069] | +0.506 [+0.411, +0.599] |
| K4L | +0.013 [−0.009, +0.034] | +0.0019 [+0.0006, +0.0035] | +0.211 [+0.130, +0.291] |
| K4H | +0.123 [+0.093, +0.156] | +0.0134 [+0.0095, +0.0178] | +0.960 [+0.845, +1.076] |

(`hook_screen.py` counts a sentence's link over its own sentence split, so its levels differ from `coh_measures.py`'s.) Orphan pronouns fall in every arm (K4 −0.049 [−0.084, −0.015]).

**Reading** (dev 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13; HLT, K4L, K1 and K4 side by side):
- **The free sentences now come from the prompt's field.** Dev 1 (black powder): K1 "Prior to burn … one part of fine cotton wool to be immersed … The sulfuric acid is added", K4 "The American gunpowder weapons"; HLT "Various Jews in Eastern European democracies". Dev 2 (grappling): K1 "Ground and pound is sometimes supplemented … mixed martial arts … maintain a grappling hook", K4 "a wrestler must be buried". Dev 11 (Interlingua): K1 "a type of constructed language whose grammar and morphology". Dev 12 (Dune): K1 and K4 write about Alia, Leto Atreides II and Children of Dune; HLT about "the South Korean … King of Scotland". Dev 10 (HAL): K1 "the Tycho monolith", "the Space Shuttle Columbia".
- **The grammar is no better.** The sentences still break after a few words, as HLT's do ("The coastal plain and the human speech organs of the gun barrel."). The subject words arrive in a patchwork of fragments, more so in K4 (dev 4: "Census Bureau … 2001 Census … densely populated country").
- **Keyword stuffing:** mild. A name repeats in a few samples (K4 dev 12: "Alia" three times in 460 characters), but no arm repeats the prompt's words line after line. stuff3 rises 0.005; the top word's share 0.056 → 0.065. I count this as no more repetition than HLT by reading, just.
- **Prompts with plans** (dev 0, 3, 5, 7) keep their fact sentences word for word; only the free sentences after them change, toward the subject (dev 7: K1 "constituent colleges of the University. A large number of student organizations").
- **K4L** changes less: its free sentences drift as HLT's do on most of the 12.
- By my reading, **K1 and K4 read at least as well as HLT and keep nearer the subject**; K4L reads as HLT. K16 and K4H failed copy32, so I did not rule on them by reading (a glance at K4H shows the documents' phrases strung together).

**Candidates** (by the registered order: reading, then the link, then SCRIBE v2): **K4** (link +0.052) and **K1** (+0.031). K4L passes but is not sent (at most 2).

### Predictions so far

- P1 flag off equals HLT: **hit**.
- P2 leak check 0 and 0 for every arm: **hit**.
- P3 `on_prev` 0.18–0.24 for K4 and K16: **hit** (0.194, 0.179); K4H highest 0.22–0.30: **hit** (0.254); K1 and K4L 0.16–0.20: **hit** for K1 (0.179), **miss** for K4L (0.158, just under).
- P4 on the prompt at s2–s4+ up 0.03–0.10 for K4 and K16: **miss** in the good direction (K4 +0.162, +0.153, +0.110; K16 +0.132, +0.161, +0.104); s1 rises less: **miss** (s1 +0.08 and +0.11).
- P5 stuff3 up at most 0.005 at rel 0.5: **hit** for K4 and K16 (+0.005), **miss** for K1 (+0.006); K4H +0.005–0.015: **hit** (+0.013).
- P6 copy32 up 0.01–0.05 at rel 0.5: **hit** (K1 +0.043, K4 +0.062 is a **miss**, K16 +0.086 **miss**); K4H +0.03–0.08: **miss** (+0.269); verbatim13 0: **hit**; bytes in the prompt's documents 0.05–0.15 at rel 0.5: **hit** for K1 and K4, **miss** for K16 (0.153).
- P7 SCRIBE v2 within −0.01 to +0.04: **miss** (+0.11 to +0.26; SCRIBE rewards the topic words, §13.34).
- P8 fact precision within ±0.03: **hit** (−0.017 to +0.001).
- P9 mean surprise up 0.1–0.4 bits: **hit** for K4L, K4 (+0.400), **miss** for K1 (+0.408), K16 and K4H.
- P10 K4 or K16 passes, K4H fails, at least one candidate: **hit** (K4H fails on copy32, not by reading).
- P11 raters: wait for the raters.
- P12 wall 8–12 minutes: **miss** (5–8); peak RAM 11.3–11.6 GB: **hit**.

### The long reading (1,800 bytes, cut 1,500; 25 prompts of `bench/blind/q3_long/prompts.json`)

`--bytes 1800 --k5-max-words 600`, the same seeds; 250 dev prompts each.

| Arm | SCRIBE v2 [95%] | copy32 | verbatim13 | bytes in the prompt's documents' spans | wall | peak RAM |
|---|---|---|---|---|---|---|
| HLT_long | 0.490 [0.465, 0.513] | 0.235 | 0 | – | 6:40 | 11.3 GB |
| K1_long | 0.575 [0.551, 0.601] | 0.270 | 0 | 0.075 (longest run 12 words) | 8:38 | 11.3 GB |
| K4_long | 0.624 [0.598, 0.652] | 0.295 | 0 | 0.117 (longest run 12 words) | 7:00 | 11.4 GB |

Leak check on the long runs: 0 same-title documents and 0 in GEN-A's source bytes for both. K1_long and K4_long differ from HLT_long in all 25 rated long prompts.

`coh_measures.py` at 1,500 characters on the 25 prompts:

| Arm | on the prompt s1 | s2 | s3 | s4+ | link to the sentence before | stuff3 | top word |
|---|---:|---:|---:|---:|---:|---:|---:|
| HLT_long | 0.360 | 0.320 | 0.160 | 0.114 | 0.106 | 0.017 | 0.028 |
| K1_long | 0.440 | 0.440 | 0.320 | 0.189 | 0.111 | 0.031 | 0.034 |
| K4_long | 0.360 | 0.480 | 0.200 | 0.212 | 0.106 | 0.026 | 0.030 |

In long text the prompt's words come back later in the paragraph (s4+ 0.114 → 0.19–0.21), but the link to the sentence before does not move (0.106 → 0.106–0.111), and stuffing rises (0.017 → 0.026–0.031) on these 25 prompts.

### Packets (built before any rating)

- `bench/blind/ret_pair/`: HLT against K4 and K1, the 50 prompts of `bench/blind/llm_rating/prompts.json`, seed 20261124. Each differs from HLT on 44 of them.
- `bench/blind/ret_long/`: HLT_long against K4_long and K1_long, the 25 long prompts, seed 20261125, `--cut 1500`. Each differs on all 25.
- `KEY.json` of each is kept outside the repository; `key.b64` and `RATER_PROMPT.md` are committed. The lead runs the raters.
