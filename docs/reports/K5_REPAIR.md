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

### Amendment 1 (2026-11-12, a fault fix; recorded before the arms are rerun)

- **What happened.** All four arms stopped in the sampler (`std::out_of_range` in `basic_string::substr`) after about 5 minutes. No SCRIBE ran; no text was read; nothing was scored.
- **The fault.** The gate also ends a draft at 240 bytes without a sentence end. The pass's segment then ran on into the next draft, so a repair could rewrite text from before that draft began. The gate's own code reads the draft from its start offset, which the shorter rewrite had moved.
- **The fix.** The segment also ends where the gate cuts a draft (`gate_max` bytes since the segment began). A repair then never reaches before the draft it is in. The method, arms, screen and predictions are unchanged.
- The 8-prompt smoke run did not meet the fault. The selftest passes. The flag-off check stands (the change is inside the pass, which does not run when the flag is off).

### Amendment 2 (2026-11-12, the same fault, fixed fully; recorded before the arms are rerun)

- **What happened.** The rerun after amendment 1 stopped in the same way in all four arms. Again no SCRIBE ran, no text was read, nothing was scored.
- **Why amendment 1 was not enough.** When a sentence ends before the gate's 24-byte minimum, the gate does not end the draft, but the pass had started a new segment there. Its 240-byte cut was then counted from the wrong place, and a segment could still run into the next draft.
- **The fix.** The pass now follows the gate's own rule for where a draft ends: at a sentence end once the draft holds 24 bytes, at 240 bytes, or at the limits, tested on the text after the repair. It keeps the draft's start in the state, and a segment never begins before it. The method, arms, screen and predictions are unchanged.
- **Checked before the rerun:** the selftest passes, and two 250-prompt stress runs on K5's own 7 MB text with a very loose flag (`tau=3`), one with and one without the planner, ran to the end.

### Checks (2026-11-12)

- main's build (4629f68c) and my build with the flag off: both equal `bench/k5_retrieve/runs/K1.gen.txt` under `cmp`, MODEL line included. **Hit.** The selftest passes.
- The first two runs of the arms stopped on a fault (amendments 1 and 2). The third ran to the end; the numbers below are from it.

### The screen (dev, seed 0, 250 prompts; files in `bench/k5_repair/runs/`, measures in `bench/k5_repair/screen/`)

One arm at a time, 6 threads, on a PC shared with other jobs (wall times vary by load).

| Arm | SCRIBE v2 [95%] | v3.1 | copy32 | verbatim13 | fact precision [95%] | mean surprise, minus K1 [95%] | words changed | wall | peak RAM | screen |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | 0.704 [0.673, 0.735] | 0.665 | 0.275 | 0 | 0.918 [0.867, 0.962] | (7.806 bits) | — | 8:44 | 11.3 GB | baseline |
| R8 | 0.696 [0.665, 0.728] | 0.641 | 0.288 | 0 | 0.929 [0.881, 0.970] | −0.205 [−0.271, −0.140] | 9.0% | 9:13 | 12.1 GB | numbers pass; **fails by reading** |
| R6 | 0.674 [0.642, 0.707] | 0.609 | **0.310** | 0 | 0.929 [0.881, 0.970] | −0.396 [−0.477, −0.316] | 19.4% | 13:22 | 12.1 GB | **fails (copy32)** |
| P8 | 0.706 [0.673, 0.736] | 0.644 | 0.293 | 0 | 0.924 [0.875, 0.966] | −0.230 [−0.298, −0.166] | 8.1% | 10:58 | 12.1 GB | numbers pass; **fails by reading** |
| F8 | 0.701 [0.670, 0.734] | 0.650 | 0.288 | 0 | 0.924 [0.877, 0.966] | −0.204 [−0.271, −0.143] | 8.1% | 10:37 | 12.1 GB | numbers pass; **fails by reading** |

- "Words changed" is the share of the words K5's sampler drafted (every gate draft) that the pass changed. R8: 31,124 sentences, 78,301 flags, 48,594 words changed in 18,698 sentences.
- Each arm differs from K1 in 210–213 of 250 samples. The other 37–40 are planned prompts written wholly by the planner, which the pass does not touch.
- The pass costs more than the smoke run showed: 437–492 s of thread time in R8, P8 and F8 (1,389 s in R6), against about 1,500 s of sampling. The backward model built in 88–163 s, 772 MB.
- J2's other measures do not move: `on_prev` +0.005 to +0.013 (all intervals hold 0), orphans, DEF+BIRTH and the subject kept by sentence within noise. stuff3 falls a little (R6 −0.0026).
- Sentences of 4 words or fewer (first 460 characters, all 250 samples): K1 0.121, R8 0.113, R6 0.105, P8 0.110, F8 0.118. The pass does not add fragments overall.

**Reading** (dev 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13; K1, R8, P8 and F8 side by side, R6 on five of them):
- **Most breaks stay.** The flagged words are mostly replaced one by one, and the clause around them stays broken. "As it turned out, in his memory." becomes "As it turned out, in his opinion." (dev 12). "This is not the most secure near 90 degrees." becomes "This is not the most easily rotated 90 degrees." (R8) or "the most skilled at 90 degrees" (P8) (dev 2).
- **Some real mends, few.** Dev 5: "only states may be obtained by the king" becomes "may be appointed by the king". Dev 11, R8: "used to determine the carrier is Esperanto. Alfred was not as affluent countries, whose economies." becomes "used to determine the sequence of Esperanto. They can be used to describe the way to India." Dev 2, R8: "Taliban, perhaps" becomes "which, perhaps".
- **The changes reach the next sentences.** Each draft is repaired before the gate chooses, so the gate picks other drafts, and the paragraph after the first change is new text. It breaks about as often as K1's. In dev 1 and dev 7, R8's new text is a run of fragments ("The character of sulfuric acid. An alternative method. … The nerve conduction."; "The University. He also the world."). The count above shows this is not a trend over all 250 samples.
- **Topic words go.** Lower surprise here partly means common words replace the subject's rare ones: dev 12 "time zones of Dune" becomes "of investment" (R8) or "of thought" (P8, F8), and the Alia thread is lost; "He died in Edinburgh" becomes "Paris" or "London"; dev 0 and 7 lose "region" for "process" or "species". This is the K5_MCMC failure again, on a smaller scale.
- **My count** of sentences that break mid-clause, in the first 460 characters of the 12 prompts: K1 about 26, R8 about 31, P8 and F8 about the same as K1. R6 is worse (many long run-on clauses: dev 2 "which, if any. But where is it in a time slot in each of the different types of active galaxy, …").
- By the registered rule (fewer broken sentences than K1, and no new kind of fault), **no arm passes**. R8, P8 and F8 pass every number; R6 fails copy32.

**Candidates: none.** By the registered rule, nothing goes to the raters, and no packets were made.

### Predictions

- P1 flag off equals K1: **hit**.
- P2 R8 changes 6–12% of drafted words: **hit** (9.0%); R6 1.5–2.5 times as many: **hit** (2.2 times).
- P3 verbatim13 0 in every arm: **hit**; copy32 within ±0.02 of K1: **hit** for R8 (+0.013), P8 (+0.018), F8 (+0.013), **miss** for R6 (+0.035).
- P4 fact precision within ±0.02 of K1: **hit** (0.924–0.929).
- P5 SCRIBE v2 within ±0.03: **hit** in every arm (R6 −0.030, at the edge).
- P6 mean surprise falls in every arm: **hit**; by 0.02–0.15 bits: **miss** (0.20–0.40); R6 and P8 the most: **hit**.
- P7 R8 and F8 mend about 1 in 5 broken sentences: **miss** (a few mends, about as many new breaks); P8 blander: not seen beyond R8; R6 makes new breaks: **hit**.
- P8 at least one arm passes the screen: **miss**.
- P9 raters: not run.
- P10 build 100–160 s: **hit** for three arms, **miss** for R8 (163 s; 88 s for P8 is also below); +0.5–1.5 GB peak RAM: **hit** (+0.8 GB); the pass under 5% of sampling: **miss** (about a third); wall +2–4 minutes: **hit** apart from R6 (+4:38).

### What it says

- A right-to-left K5 finds the words a left-to-right K5 drew badly: it flags about 15% of drafted words at 8 bits, and resampling them lowers mean surprise by 0.2 bits a word with copy and fact precision held.
- But the breaks readers see are not single bad words. A clause goes wrong over several words, and a 5-gram in either direction, read 4 words at a time, finds each word plausible next to its neighbours. Swapping words one at a time keeps the broken frame; where it lowers surprise most, it does so by putting common words in place of the subject's own.
- Lower surprise is a screen between variants of one writer (§13.57), not proof. Here the reading says the arms are not better, so they were not rated. If the lead wants a second opinion, R8 (or F8, the task's first form) is the arm to rate; that would need an amendment.
- Next, if this line goes on: repair at the level of the clause (redraw the span between two flagged joins with both directions, so the frame can change), and keep the retrieved document's words from being replaced (a word the overlay raises stays).
