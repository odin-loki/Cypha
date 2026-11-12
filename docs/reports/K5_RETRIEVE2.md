# G4: build on the retrieval win

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/k5-retrieve2` (from main a37cd45e) · GENERATION_PLAN §13.66, §13.67

(The method, the arms, the screen, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- §13.66 made retrieval the default (K1: `--k5-retrieve k=1,rel=0.5`). Blind raters preferred it to HLT 0.860 at 460 characters and 0.880 at 1,500. K4 (k 4) read 0.820 and 0.940: better on long text.
- G2 left four known limits ([K5_RETRIEVE](K5_RETRIEVE.md)):
  - the planner's fill of fact sentences does not read the overlay, so the fact sentences keep K5's general vocabulary;
  - one fixed k for all lengths;
  - in long text the link to the sentence before does not move (coh_measures 0.106 → 0.111), though the prompt's words come back;
  - grammar is no better, and the subject's words sometimes come as a patchwork.
- This job tests two ideas: the fill reads the overlay (R-fill), and the retrieval changes as the text grows (R-grow).

## The method

All new code is behind new fields of `--k5-retrieve`, off by default. With them off the GEN file is byte-identical (checked below). Code: `lm/retrieve.hpp`, `lm/word_kn.hpp`, `lm/planner/writer.hpp`, `lm/planner/discourse.hpp`.

**R-fill: `fill=1`.** The planner's fill (`planner/writer.hpp`, `Filler::fill` and `fill_dba`) builds its scratch state from the context only. With `fill=1` the scratch state also takes the continuation's overlay. So a fact sentence's class words are drawn from K5 leaning on the retrieved document, as the free sentences already are. The fact's names come from the store as before. The honest fill rules stay on: a number, a month, a nationality or a country may still come only from the fact (`--k5-plan-honest life`, the default). The fill's beam score reads the same overlay, so the chosen skeleton may change too.

**R-grow: `grow=B[,k2=N][,qtext=1]`.** Each time the text passes another B bytes, at the next sentence boundary, the overlay is rebuilt from the k2 best documents:
- without `qtext`, the query is the prompt, as at the start. Only the first later retrieval runs (the prompt gives the same documents after). So `k=1,grow=500,k2=4` is K1 for about the first 500 bytes, then K4's four documents.
- with `qtext=1`, the query is the prompt and the text so far (a moving query), the same BM25 rule. It is run again at each B bytes. The overlay then follows what the paragraph has turned to.
- The check runs before each word (at a sentence end) and once at the top of each sentence of the discourse writer and of K5's gate, so the K drafts of a sentence share one retrieval.
- Every later retrieval is logged (`CYPHA_K5_RET_LOG`, a fourth field `grow=R`). Each distinct (prompt, round, documents) is logged once.

**Guards (unchanged).** Only `train_all.txt` is searched (the `--k5-more` text: enwik9 past enwik8, GEN-A and SCRIBE overlap removed). K5's own 7 MB text is never retrieved. A moving query reads K5's own text so far, which comes from the same counts, so it cannot reach GEN-A's pages either. The 12-word copy cap holds against every retrieved document: they are K5's training text, so the cap's windows hold them.

**`leak_check.py copy`** now takes all of a prompt's logged documents (several lines with grow). With one line per prompt it gives what it gave before.

## The baseline

**K1**, today's default: `scripts/gen_eval.sh dev K1 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `cyphalm_trie` and `scribe` built by `g++ -O2` as `scripts/build_1tb_tools.sh` builds them (`libsais.o` linked into `scribe`), `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`. Its dev numbers (G2): SCRIBE v2 0.704, copy32 0.275, fact precision 0.918 on dev (0.924 on test).

**The long baseline: K1_long**, the same with `--bytes 1800 --k5-max-words 600` (G2's `bench/k5_retrieve/runs/K1_long.gen.txt`): SCRIBE v2 0.575, copy32 0.270.

**Checks (run before any arm):**
- main's build (a37cd45e): the GEN file equals `bench/k5_retrieve/runs/K1.gen.txt` under `cmp`, MODEL line aside;
- my build with the new fields off: the same; and its long run equals `K1_long.gen.txt`;
- selftest: the new fields parse and are off by default.

**Done before this was written:**
- main's build (a37cd45e): **equal** to K1's GEN file under `cmp` (MODEL line aside), and its retrieval log has the same lines (5:36 wall, 11.3 GB; SCRIBE v2 0.704, copy32 0.275).
- selftest with my build: **passes** (the new fields parse, are off by default, and `fill=2` is refused).
- My build with the fields off, short and long, runs next, before any arm.

**Before this was written** (stated so the reader can weigh it): to check that the code runs, I ran all the new fields at once (`k=1,rel=0.5,fill=1,grow=400,k2=4,qtext=1`, `--bytes 1800 --k5-max-words 600`) with `cyphalm_trie sample` on dev prompts 0–7, no SCRIBE (4:32 wall with the model's load, 11.4 GB). The retrieval log shows the later retrievals firing (rounds 1–4 per prompt). With the moving query the first document stays first for some prompts and drops out for others. The leak check on that log: 0 same-title documents, 0 in GEN-A's source bytes (148 documents, 63 distinct). I did not read the samples' text. No code changed after this run.

## The arms

Dev, seed 0, 250 prompts; the baseline's command plus:

| Arm | Flags | What it tests |
|---|---|---|
| RF | `--k5-retrieve k=1,rel=0.5,fill=1` | R-fill: the fact sentences read the overlay |
| RG | `--k5-retrieve k=1,rel=0.5,grow=500,k2=4` | R-grow by schedule: K1 for the first ~500 bytes, then K4's breadth |
| RQ | `--k5-retrieve k=1,rel=0.5,grow=500,k2=4,qtext=1` | R-grow by a moving query: every 500 bytes, four documents for the prompt and the text so far |
| RFG | `--k5-retrieve k=1,rel=0.5,fill=1,grow=500,k2=4` | RF and RG together |

Each arm is also run long (`--bytes 1800 --k5-max-words 600`, labels `X_long`), because RG, RQ and RFG change only text past about 500 bytes.

Why these and not others:
- **rel tuned.** G2's K4L (rel 0.2) read as HLT, and rel above 0.5 raised copy32 fast (K4H 0.501). A K1 at rel 0.7 would likely sit near the 0.30 cap. Not run.
- **Near-duplicate documents.** G2's logs give 983 distinct documents of 1,000 for K4, but that is across prompts. Near-duplicates within one prompt's four documents were not measured, and k=1 (the default) cannot have them. Not run.
- **Why 500 bytes.** At 460 characters K1 reads best (0.860 against K4's 0.820); at 1,500, K4 (0.940 against 0.880). The switch is put just past the short reading's cut.

**Machine limits:** at most 2 arms at once, at most 6 threads each, no start while `free -g` shows under 14 GB available.

## Measures

Dev, against K1:
- SCRIBE v2 with its 95% interval, v3.1 beside it, copy32, verbatim13, wall time, peak RAM, samples unlike K1 (all 250, the 50 rated at 460 characters after the cut, and the 25 long ones).
- **Fact precision:** `python3 -I scripts/gen_fact_check.py run --v2`, with its 95% interval.
- **J2's measures** (`bench/para_plan/hook_screen.py`, 460 characters, each arm minus K1 with a 95% paired bootstrap): `on_prev`, stuff3, **mean surprise** (`scripts/surprise_profile.py`).
- **The prompt's words by sentence** (`bench/k5_coherence/coh_measures.py`): 460 characters on the 250, and 1,500 characters on the long runs.
- **The leak check** (`bench/k5_retrieve/leak_check.py check`) and the copy against each prompt's own documents (`leak_check.py copy`), short and long.

## The screen

**Reading.** I read the 12 unrated dev prompts G2 read (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), each arm beside K1: RF at 460 characters, and every arm's long run at 1,500 characters. I ask: are the fact sentences still plain and true (RF), does the text keep to the prompt's subject and do its sentences follow from each other (long), and does it stuff keywords. **Keyword patchwork is the known failure:** an arm that stays "on topic" by stringing the documents' names and phrases together, or by repeating the prompt's words, fails.

**An arm passes if all hold:**
- verbatim13 is 0, short and long;
- copy32 at most 0.30, short and long;
- SCRIBE v2 at least K1's minus 0.03 (0.674), and long at least K1_long's minus 0.03 (0.545);
- fact precision at least 0.874 (K1 − 0.05, K1 taken as 0.924, its test figure, the stricter of its two);
- the leak check gives 0 same-title documents and 0 in GEN-A's source bytes, short and long;
- by my strict reading it reads at least as well as K1, with no more keyword patchwork.

`on_prev` and mean surprise are reported, not gated.

**Candidates:** at most 2 that pass. At most one per idea: the R-fill arm (RF) and the best R-grow arm (RG, RQ, RFG). The R-grow arms are ranked by my long reading, then the long link to the sentence before (`coh_measures.py` at 1,500), then long SCRIBE v2. If two R-grow arms pass and RF fails, both may go. If none pass, nothing goes to the raters and I say so.

## The deciding judge

- **460 characters:** `python3 scripts/blind_rate.py make --out bench/blind/ret2_pair --ids bench/blind/llm_rating/prompts.json --arm K1=bench/k5_retrieve/runs/K1.gen.txt --arm X=... --mode pair --baseline K1 --seed 20261127`: the 50 prompts, four fresh blind Claude raters.
- **1,500 characters:** `--out bench/blind/ret2_long --ids bench/blind/q3_long/prompts.json --arm K1=bench/k5_retrieve/runs/K1_long.gen.txt --arm X=..._long.gen.txt --mode pair --baseline K1 --seed 20261128 --cut 1500`: the 25 prompts, four more fresh raters.
- A candidate whose 460-character text equals K1's on all 50 rated prompts (expected for RG and RQ) is left out of the short packet. Its short reading is then 0.5 by construction (the same text), and the long reading decides.
- Packets, `key.b64`, `RATER_PROMPT.md` and the GEN files are committed and pushed before the raters run. `KEY.json` stays out of the repository. The lead runs the raters.
- **The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Adopted** if it reads better by the rule on either reading and its win share on the other reading is at least 0.5. **The best** such arm is the one with the higher of the weaker of its two lower bounds (§13.66's reading of the rule; a reading equal by construction counts 0.5 as its lower bound), then the lower copy32 (dev). Odin can veto. Then GEN-A test is read once, for it only.

## Predictions

| # | Prediction |
|---|---|
| P1 | main's build and my build with the fields off equal K1 (`cmp`); my long run equals K1_long |
| P2 | The leak check: 0 same-title documents and 0 in GEN-A's source bytes, every arm, short and long |
| P3 | RF changes the fact sentences of most planned prompts: unlike K1 in 235–250 of 250 (K1 differs from HLT in 214) |
| P4 | RF's fact precision within −0.04 to +0.01 of K1 (the class words lean to the subject's own words; the honest rules stop invented numbers and countries, not names) |
| P5 | RF: SCRIBE v2 +0.00 to +0.03; copy32 +0.00 to +0.02; `on_prev` (hook_screen) +0.00 to +0.03 |
| P6 | RG and RQ: equal to K1 at 460 characters on all 50 rated prompts |
| P7 | Long: RG and RQ copy32 0.27–0.30; RQ's copy against its own documents above RG's |
| P8 | Long link to the sentence before (`coh_measures`, 1,500): K1_long 0.111; RG 0.11–0.13; RQ 0.12–0.16 |
| P9 | Long stuff3 up at most 0.005 for RG; RQ up 0.000–0.010 (a moving query can feed the paragraph's own words back) |
| P10 | RQ's wall time 10–60% above K1's (larger queries); RF's and RG's within 15% |
| P11 | By my reading RF passes; at least one R-grow arm passes; at most one fails for patchwork |
| P12 | Raters: RF against K1 at 460 characters 0.50–0.60 (not better by the rule); the best R-grow arm 0.55–0.65 at 1,500, adopted only if its interval clears 0.5 |

## Results

(Added after the runs, under this line.)
