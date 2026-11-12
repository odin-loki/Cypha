# D1: a bidirectional repair pass for K5's sentences

**Odin Loch** · registered 2026-11-12, before any arm runs · branch `wip/k5-repair` (from main 4629f68c) · GENERATION_PLAN §13.66, §13.69

(The method, the arms, the screen, the deciding rule and the predictions below were written and committed before any arm ran. Results are added under them.)

## Why

- Today's writer is K1 (§13.66). Blind readers rate it 2.28 of 10; human text 8.54 (§13.69).
- Readers mark it down first for sentences that break mid-clause: "In many as a student at a great field."
- K5 writes left to right from the last four words. Once a clause has gone wrong, it cannot see it, and it cannot go back.
- **Hypothesis:** a right-to-left K5 sees each word from the other side. A word that both directions find unlikely is where a clause broke. Resampling only those words, with context from both sides, mends the clause and leaves the rest of the sentence alone.

## How this differs from K5_MCMC

[K5_MCMC](K5_MCMC.md) (§13.32) edited whole paragraphs by Metropolis–Hastings under K5's left-to-right probability. Its failures, and what this pass does about each:
- **Raising K5's own probability made the text bland and more copied** (R1, R4: both judges fell; copy32 +0.02). Here only flagged words move (expected well under a tenth of the words), and the default draw is from the two-sided conditional at T 2, not a sharpened target.
- **Insert and delete moves padded the text and cut it at odd places.** Here there are no inserts or deletes. The sentence keeps its length and its end.
- **Local edits judged by a 5-gram model joined sentences in new places** (D_S rose in every chain). Here a word is judged by both directions over a window inside its own sentence; the sentence's last word and the sentence ends are never changed.
- **The chain worked on whole paragraphs after the fact,** so the next sentence was not written from the repaired one. Here each sentence is repaired as soon as K5 ends it, so K5 writes the next sentence after the repaired text.
- **The planner's fact sentences** did not exist then. Here they are never touched, so fact precision should hold.

## The method

New flag, off by default, byte-identical when off: `--k5-repair on|off|tau=B,t=T,kf=N,kb=N,sweeps=N`. Code: `native/include/cypha/lm/k5_repair.hpp` (new), a small hook in `lm/word_kn.hpp` (listed below), and the flag in `native/tools/lm/cyphalm_trie.cpp`.

**The backward K5.**
- The same 337 MB text K5 counts from (`fuzzy/train.txt` then the more-text), each document read backwards by word.
- The same vocabulary and word ids, the same order (5), the same Kneser–Ney estimate and the same kept-context rule. Its counts come from one more suffix array over the reversed ids (`lm/count_index.hpp`).
- Its floor uses K5's own 256 exchange classes: a class trigram counted on the reversed class stream, and K5's P(word | class).
- Nothing else moves: the forward K5, its classes, the planner, the gate and retrieval are as they are.
- With retrieval on (today's default, K1), the backward model reads the same retrieved document, backwards, as an overlay with the same weight.

**When it runs.** Each time K5's sampler ends a sentence (a word `.` `!` `?` or a newline, or the byte or word limit), the pass looks at the words the sampler wrote since the last repair.
- Only those words may change. The prompt and the planner's fact sentences are never changed. The sentence's last word is never changed.
- It runs on every draft of the sentence gate, before the gate chooses. So the gate chooses among repaired drafts, and K5 writes the next sentence after the repaired one.
- The gate's terms that are summed while a draft is drawn (prompt association, topic) see the draft before its repair. The draft's text, and the state the next sentence starts from, are the repaired ones.

**Flag.** For each word x_i of the sentence:
- sf(i) = −log2 Pf(x_i | the 4 words before), forward K5 as it writes (retrieval overlay and class floor);
- sb(i) = −log2 Pb(x_i | up to 4 words after, inside the sentence), backward K5.
- A word is flagged when both directions find it unlikely: min(sf(i), sb(i)) ≥ τ.
- A join between two words is broken when forward K5 does not expect the right word and backward K5 does not expect the left one: min(sf(i+1), sb(i)) ≥ τ. Both its words are flagged (those that may change).

**Resample** (Gibbs-style, left to right over the flagged words):
- Candidates at i: the kf most probable words of forward K5 at i, the kb most probable of backward K5 at i, and the current word. Never `<unk>`, `<s>` or a sentence end.
- A candidate is dropped if it would make a 13-word copy of the training text (Odin's 12-word cap) or break the repeat ban.
- Score over the window, with w at i: F(w) is the sum of log2 Pf over positions i … i+4 (inside the sentence); B(w) is the sum of log2 Pb over positions i−4 … i.
- w is drawn with probability proportional to 2^((F(w) + B(w)) / T).
  - T 2 is the geometric mean of the two directions. Each direction alone already scores x_i given both sides, so their plain product counts the evidence twice.
  - T 1 is their plain product, as the task first put it.
  - T 0 takes the best candidate.
- Up to `sweeps` sweeps. The flags are found again before each sweep; a sweep with no flag ends the repair.
- The random stream is seeded from the sentence's word ids, so the text does not depend on the number of threads.

**Byte-identical when off.** With no `--k5-repair` (or `off`), no new field is set and no hook runs. `WordKN::prob` gains an optional overlay argument; without it the arithmetic is as before.

**The hook in `word_kn.hpp`** (all marked `K5_REPAIR.md`; 39 lines added, 5 changed):
- a forward declaration `namespace repair { class Repairer; }` after `struct GenStats;`;
- `WordKN::prob(…, const Overlay* ov = nullptr)`: one word's probability with an overlay, in `dist`'s arithmetic and order (the selftest checks it against `dist` on every word);
- `Options::repair`, `Options::repair_desc`, and one line in `Options::describe`;
- `K5::load`: one line, builds the backward model;
- `K5::State`: two fields (`rp`, `rp_seg`);
- `K5::start`: one line; `K5::step_word`: one line before `return w;`;
- three hook declarations after `K5::train_ids`, defined in `k5_repair.hpp`;
- `#include "cypha/lm/k5_repair.hpp"` at the end of the file.

**Selftest** (`cyphalm_trie selftest`): the flag parses and is off by default; the forward probability with an overlay equals `dist` on every word, with and without an overlay; the backward model's distribution sums to one and matches its one-word probability; a broken sentence after a prompt is repaired with the prompt and the end kept, the text equal to its ids re-rendered, and the repeat ban rebuilt. **Passes.**

## The baseline

**K1**, today's default: `scripts/gen_eval.sh dev K1 OUT --prompts 250 --k5 /mnt/d/cypha_llm/fuzzy/train.txt`, with `cyphalm_trie` and `scribe` built by `g++ -O2` as `scripts/build_1tb_tools.sh` builds them (`libsais.o` linked into `scribe`), `WIKI=ENWIK8=/mnt/d/cypha_llm/enwik8`, `SCRIBE_VEC=/mnt/d/cypha_llm/scribe/v32`, `K5_THREADS=6`.

**Checks first:**
- main's build (4629f68c): the GEN file equals `bench/k5_retrieve/runs/K1.gen.txt` under `cmp`, MODEL line included. **Done before this was written: equal** (10:16 wall on a loaded PC, 11.3 GB). SCRIBE v2 0.704, as K1.
- my build with the flag off: the same. **Done before this was written: equal**, MODEL line included (8:44 wall, 11.3 GB).
- selftest: **passes.**

**Before this was written** (stated so the reader can weigh it): to check that the code runs and to time it, I ran R8's flags with `cyphalm_trie sample` on dev prompts 0–7 (no SCRIBE, no measure). I did not read the samples' text. The backward model built in 125 s (772 MB; K5's forward model 767 MB); peak RAM 12.1 GB. The pass saw 754 sentences (every gate draft), 11,879 of their words; 1,640 flags; it changed 1,026 words (8.6%) in 410 sentences; it took 14 s of 141 s sampling (all threads). Nothing was changed after this run.

## The arms

Dev, seed 0, 250 prompts, the baseline's command plus:

| Arm | Flags | What it tests |
|---|---|---|
| R8 | `--k5-repair tau=8,t=2,kf=16,kb=16,sweeps=3` | the pass as designed: words both directions give under 1/256 |
| R6 | `--k5-repair tau=6,t=2,kf=16,kb=16,sweeps=3` | a looser flag (1/64): more words move |
| P8 | `--k5-repair tau=8,t=1,kf=16,kb=16,sweeps=3` | the plain forward × backward product (sharper) |
| F8 | `--k5-repair tau=8,t=1,kf=32,kb=0,sweeps=3` | the task's first form: proposals from forward K5 only, accepted by the product |

**Machine limits:** one arm at a time, 6 threads, no start while `free -g` shows under 16 GB available.

## Measures (dev)

- SCRIBE v2 with its 95% interval, v3.1 beside it, copy32, verbatim13, wall time, peak RAM.
- **Fact precision:** `python3 -I scripts/gen_fact_check.py run --v2` (GEN_FACTS amendment A1), with its 95% interval.
- **WRITING_SIGNAL's mean surprise** and J2's measures (`bench/para_plan/hook_screen.py --surprise scripts/surprise_profile.py`, 460 characters, each arm minus K1 with a 95% paired bootstrap). Lower surprise reads better between variants of one writer (§13.57).
- The pass's own counts: sentences seen, flags, words changed, sentences changed, time in the pass.
- Samples unlike K1 (all 250 and the 50 rated).

## The screen

**Reading.** I read the 12 dev prompts K5_COHERENCE and K5_RETRIEVE read (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13), each arm beside K1. I ask, strictly: is each sentence a sentence (subject, verb, a clause that closes), and is each change a mend or a new break? Patchwork of stock phrases, blander text with the same breaks, and new breaks all count against an arm. I count, per arm and for K1, the sentences in those 12 samples (first 460 characters) that break mid-clause.

**An arm passes if all hold:**
- verbatim13 is 0;
- copy32 at most 0.30;
- fact precision at least 0.874;
- SCRIBE v2 at least K1's minus 0.03 (K1 dev 0.704, so 0.674);
- mean surprise not above K1's by more than 0.1 bits a word (the upper end of its 95% interval);
- by my strict reading, fewer broken sentences than K1 on the 12 prompts, and no new kind of fault.

**Candidates:** at most 2 that pass, chosen by my reading (fewest broken sentences) first, then lower mean surprise, then SCRIBE v2. If none pass, nothing goes to the raters and I say so.

## The deciding judge

- `python scripts/blind_rate.py make --out bench/blind/rep_pair --ids bench/blind/llm_rating/prompts.json --arm K1=bench/k5_retrieve/runs/K1.gen.txt --arm X=… [--arm Y=…] --mode pair --baseline K1 --seed 20261130`: the 50 prompts, 460 characters, four fresh blind Claude raters.
- Packets, `key.b64`, `RATER_PROMPT.md`, the GEN files, summaries and fact checks are committed and pushed before the raters run. `KEY.json` stays out of the repository. The lead runs the raters.
- **The §13.38 rule:** an arm reads better if its win share minus 0.5 is above 0, its 95% bootstrap interval excludes 0, and both rater pairs agree on the sign.
- **Adopted** if it reads better by the rule. The best such arm (higher lower bound first, then lower copy32) can become the default; Odin can veto. Then GEN-A test is read once, for it only.

## Predictions

| # | Prediction |
|---|---|
| P1 | My build with the flag off equals K1 (`cmp`) |
| P2 | R8 changes 6–12% of the words K5's sampler drafts (all gate drafts counted); R6 1.5–2.5 times as many as R8 |
| P3 | verbatim13 0 in every arm; copy32 within ±0.02 of K1 (0.275) in every arm |
| P4 | Fact precision within ±0.02 of K1 (0.918): the fact sentences are not touched |
| P5 | SCRIBE v2 within ±0.03 of K1 in every arm |
| P6 | Mean surprise falls in every arm, by 0.02–0.15 bits a word (R6 and P8 the most) |
| P7 | By my reading, R8 and F8 mend some breaks (about 1 in 5 broken sentences) and P8 reads blander; R6 makes some new breaks |
| P8 | At least one arm passes the screen and goes to the raters |
| P9 | Raters: the best candidate's win share against K1 is about 0.55, with the interval touching 0.5 (not adopted); only a few words per sample change |
| P10 | The backward model takes 100–160 s to build and adds 0.5–1.5 GB of peak RAM (K1: 11.3 GB); the pass itself is under 5% of the sampling time; wall time rises by 2–4 minutes (the build) |

## Results

(Added after the runs, under this line.)
