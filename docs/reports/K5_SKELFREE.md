# D3: real sentence shapes for the free sentences

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/k5-skelfree` (from main 4629f68c) · GENERATION_PLAN §13.37, §13.38, §13.48, §13.63, §13.66, §13.69

(The method, the bank, the arms, the screen, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- Today's writer is K1 (§13.66). Blind readers rate it 2.28 out of 10; human text 8.54 (§13.69). Readers mark down broken clauses first.
- The planner's fact sentences are built plan → skeleton → fill: a real sentence's shape in K5's 256 exchange classes, filled by K5 (§13.37, [K5_PLANNER](K5_PLANNER.md)). Those sentences end where a real sentence ends.
- The free sentences, most of the text, are K5's left-to-right drafts (16 per sentence), chosen by the gate. K5 sees four words back and has no idea where its sentence is going, so a clause often breaks after four words.
- **Idea:** write the free sentences skeleton-first too. Take the shape of a whole real sentence, then let K5 fill it on the subject (the retrieval overlay), with the copy cap and the repeat ban.

## The method

New flag, off by default, byte-identical when off: `--k5-skelfree bank=FILE[,n=N][,k5=M][,pool=P][,fills=F]`. Code: `native/include/cypha/lm/planner/free_skel.hpp` (new), a branch in `planner/discourse.hpp`, the flag in `native/tools/lm/cyphalm_trie.cpp`, the miner `native/tools/lm/k5_skelfree.cpp` (new). K5's vocabulary, its classes and every word ID stay as today (§13.48 W1b: they must not move).

**1. The bank** (`k5_skelfree mine`; rule 10: mined, nothing hand-written).
- Text: `/mnt/d/cypha_llm/k5scale/clean/train_all.txt` bytes [0, 100,000,000), the range the fact skeletons were mined from. It is clean enwik9 past enwik8 with GEN-A's and SCRIBE's overlap removed.
- Each document (line) is cut into sentences as K5's gate cuts a draft (`structure::split`).
- A sentence becomes a skeleton, the sequence of its tokens' K5 classes, if all hold:
  - it starts a sentence: it is its document's first, or the one before it ended; and its first token begins with a capital A–Z or a digit;
  - it ends with `.`, `!` or `?`;
  - every token is a K5 word (no `<unk>`, no newline);
  - it has 5 to 32 tokens.
- Skeletons are counted. All are kept.
- **Bank** (`bench/k5_skelfree/bank_e9.txt`, sha256 `05899f17…cfab3e45`, 10.6 MB): 807,712 sentences read in 30,359 documents; **166,441 kept, 164,983 distinct skeletons** (758 seen twice or more). Left out: 85,810 starting inside a sentence, 46,631 not ended, 114,838 outside 5–32 tokens, 393,992 with a word K5 does not know. Kept lengths run from 5 to 32 tokens, most at 12–18.

**2. The choice** (per free sentence).
- `pool` skeletons (256) are drawn from the bank in proportion to their counts: a random real sentence's shape.
- Each is scored as the planner scores fact skeletons (§13.37 step 2): the mean log2 probability of its classes under the structure statistics' class 5-gram after the text's last four classes, plus a quarter of the rhythm expert's length model after the sentence before.
- The best `n` are kept. A skeleton offered once is not offered again in that continuation.

**3. The fill.** Each kept skeleton is filled left to right by K5 on word IDs:
- at each class code, K5's next-word distribution (with the retrieval overlay and every online part, as K5's own sampler reads it) is cut to the words of that class and renormalised;
- the 12-word copy cap and the repeat ban apply as in K5's sampler (the repeat ban is lifted if it leaves no word);
- K5's nucleus (0.8) over the class's kept words, then one draw;
- with `fills=F`, the skeleton is filled F times (own streams) and the fill with the highest mean log2 P under K5 is kept;
- a fill is dropped if the cap leaves no word at a position.

**4. The gate.** The skeleton drafts go to the gate that chooses today, unchanged:
- on prompts without a plan (about two thirds), K5's z-scored gate (the prompt association, the LDA topic term, the markup share), re-implemented beside K5 in `free_skel.hpp` (`generate_free_plain`);
- after a plan's fact sentences, the discourse writer's paragraph gate (prompt association, topic, entity grid, markup).
- With `k5=M`, the first M of K5's own 16 drafts (the same streams as today) join the skeleton drafts. With `k5=0`, skeleton drafts replace K5's.
- Skeleton drafts are made only at a sentence start (the text ends with `.`, `!`, `?` or a newline). Otherwise, or if no skeleton could be filled, the gate gets K5's 16 drafts as today.

**Not changed:** the fact sentences (plan, skeleton, honest fill, gate), the retrieval, the 337 MB counts, the LDA, the copy cap, the 600-byte cut.

## The baseline

**K1**, today's default: `scripts/gen_eval.sh dev K1 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `cyphalm_trie` and `scribe` built by `g++ -O2` as `scripts/build_1tb_tools.sh` builds them (`libsais.o` linked into `scribe`), `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`. Its dev numbers (§13.66): SCRIBE v2 0.704 [0.673, 0.735], copy32 0.275, verbatim13 0, fact precision 0.918.

**Checks (before any arm):**
- my build with the flag off: the GEN file equals `bench/k5_retrieve/runs/K1.gen.txt` under `cmp`, MODEL line aside;
- my build with `--k5-skelfree bank=...,n=0,k5=16` (no skeleton drafts, all 16 of K5's): the same text. This checks that the re-implemented gate and the new paragraph-gate branch choose as today's code does;
- selftest passes (the flag's parser; bad specs refused).

**Done before this was written** (stated so the reader can weigh it):
- selftest with my build: **passes**.
- To check that the code runs and to time it, I ran `cyphalm_trie sample` (no SCRIBE) on 8 unrated dev prompts (200–207), three ways: the default; `n=0,k5=16`; and SA (`n=16,k5=0`).
  - `n=0,k5=16` **equals** the default under `cmp` (MODEL line aside), so the re-implemented gate and the new branch choose as today's code.
  - SA ran: 1,264 skeleton drafts over 79 sentence starts, the gate chose a skeleton draft for all 79 (it has no other). Wall 6:40 against the default's 8:20 (one at a time, the model's load included); peak RAM 11.3 GB both. Each sample is 614–623 bytes long.
  - I did not read the samples' text.
- The mining rule's two start conditions (a sentence after one that ended; a capital or a digit first) were added after I looked at 24 skeletons of a first bank, shown as their classes' most probable words: some began mid-sentence ("members . The most …", "\" . In …"). That was the only look at any skeleton before this was written.
- My build with the flag off on all 250 dev prompts (the `cmp` against K1) runs next, before any arm.

## The arms

Dev, seed 0, 250 prompts; the baseline's command plus `--k5-skelfree bank=bench/k5_skelfree/bank_e9.txt,` and:

| Arm | Fields | What it tests |
|---|---|---|
| SA | `n=16,k5=0` | all 16 drafts skeleton-first |
| SH | `n=8,k5=8` | half skeleton drafts, half K5's own; the gate picks |
| SF | `n=16,k5=0,fills=2` | SA with the more fluent of two fills per skeleton |
| SHF | `n=8,k5=8,fills=2` | SH with the more fluent of two fills |

**Machine:** one arm at a time, at most 6 threads, no start while `free -g` shows under 16 GB available.

## Measures (dev)

- SCRIBE v2 with its 95% interval, v3.1 beside it, copy32, verbatim13, wall time, peak RAM; samples unlike K1 (all 250 and the 50 rated).
- **Fact precision:** `python3 -I scripts/gen_fact_check.py run --v2` with its 95% interval.
- The gate's choice: skeleton drafts made, and how often the gate chose one (stderr).
- **Ended sentences** (unregistered helper, reported only): the share of sentences in the first 460 characters that end with `.`, `!` or `?` before the cut.

## The screen

**An arm passes if all hold:**
- verbatim13 is 0;
- copy32 at most 0.30;
- fact precision (`gen_fact_check.py run --v2`) at least 0.874;
- SCRIBE v2 at least K1's minus 0.03 (0.704, so 0.674);
- **my strict reading of grammar.** I read the 12 unrated dev prompts read before (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), first 460 characters, each arm beside K1. For each free sentence I ask one question: is it a whole clause a reader can parse (a subject, a finite verb, its parts in order), whatever it says? I count the whole sentences and the broken ones per arm. An arm passes if its share of whole sentences is above K1's on the 12 prompts, it stays on the subject at least as K1 does, and it shows no keyword stuffing or patchwork of the retrieved document's phrases. A frame filled with words that do not fit it ("The museum is the most river of the battle") counts as broken.

**Candidates:** at most 2 that pass, chosen by my reading first (the share of whole sentences), then lower copy32. If none pass, nothing goes to the raters and I say so.

## The deciding judge

- `python3 scripts/blind_rate.py make --out bench/blind/skf_pair --ids bench/blind/llm_rating/prompts.json --arm K1=bench/k5_retrieve/runs/K1.gen.txt --arm X=... [--arm Y=...] --mode pair --baseline K1 --seed 20261202`: the 50 prompts, 460 characters, four fresh blind Claude raters.
- Packets, `key.b64`, `RATER_PROMPT.md`, the GEN files, summaries and fact checks are committed and pushed before the raters run. `KEY.json` stays out of the repository. The lead runs the raters.
- **The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Adopted** if it reads better by the rule; the better of two (higher lower bound, then lower copy32) becomes the default; Odin can veto. Then GEN-A test is read once, for it only, with the fact check.

## Predictions

| # | Prediction |
|---|---|
| P1 | My build with the flag off equals K1 (`cmp`); with `n=0,k5=16` too |
| P2 | verbatim13 0 in every arm; copy32 0.20–0.30 (the class cut breaks some long runs; the overlay keeps others) |
| P3 | Fact precision within ±0.02 of K1 (the fact sentences do not change) |
| P4 | SCRIBE v2: SA and SF 0.02–0.08 below K1; SH and SHF within ±0.03 |
| P5 | In SH and SHF the gate chooses a skeleton draft for 40–60% of free sentences |
| P6 | By my reading, the skeleton arms have more whole sentences than K1, but more misfit words in them (a class's word that does not fit the meaning). SF and SHF read best of the four |
| P7 | At least one arm passes the screen; SA fails SCRIBE or the reading |
| P8 | Raters: the best candidate's win share against K1 about 0.55–0.65; adopted only if its lower bound clears 0.5 |
| P9 | Wall time per arm within ±30% of my K1 run's (fills=2 arms at the top); peak RAM 11.3–11.6 GB |

## Results

(Added after the runs, under this line.)
