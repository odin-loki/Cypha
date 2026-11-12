# X1: where the writing breaks

**Odin Loch** · 2026-11-13 · branch `wip/where-it-breaks` (from main deda93c3) · GENERATION_PLAN §13.48–13.74 · [K5_DECODE](K5_DECODE.md) · [WRITING_SIGNAL](WRITING_SIGNAL.md) · D4: `docs/reports/K5_MIXDEC.md` and `docs/DECODER.md` on `wip/k5-mixdec`

(The prompts, the texts, the measures and the way broken spots are marked were written and committed before the probe ran on them. Results are added below them.)

## Why

- **Odin:** "a logistic map of the output and a best effort of the most probable output. Somethings fucking up somewhere."
- Today's default, T8 (§13.71), reads **2.50 of 10** to blind raters; human text 8.64 (§13.74). Sentences break mid-clause and drift.
- I read Odin's request as two things:
  1. a **log-probability map**: each word of the output, and of the human text, scored by each part of the model;
  2. the model's **most probable output** (best-effort MAP, by beam search), to tell a model problem from a sampling problem. If the most probable text is fluent, sampling breaks it. If it is broken too, the model is wrong.

## The writer under test

T8, as `cyphalm_trie sample --k5 /mnt/d/cypha_llm/fuzzy/train.txt` builds it today:
- K5, a word Kneser–Ney model of order 5. Vocabulary (32,000 words) and 256 exchange classes from `fuzzy/train.txt` (7 MB); counts from it plus `train_all.txt` (337 MB in all). The class model is the backoff floor.
- K1's retrieval overlay: the one document of the 337 MB nearest the prompt (BM25) adds its n-grams to the counts (`k=1,rel=0.5`).
- The 16-draft sentence gate. Its terms: the prompt association (weight 1), the LDA topic term (weight 1), markup (−5). Its fluency weight is 0.
- The planner's fact sentences (the discourse writer), where the planner has a plan.
- Temperature 0.8, nucleus 0.8, the 11-byte repeat ban, the 12-word copy cap.
- D4's decoder (MR, `wip/k5-mixdec`) adds a byte learner: the 1 TB trie, byte KN, an online n-gram, a byte LSTM and a geometric mixer, primed on the retrieved article and the prompt, then frozen. MR multiplies K5's nucleus by the learner's context ratio (r 0.3).

## The tool (no change to the writer)

- `native/tools/lm/k5_probe.cpp` (new). It builds K5 exactly as `sample` does and reads it; it writes nothing. With `wip/k5-mixdec`'s `decoder/learner.hpp` on the include path (that branch is unmerged), it also builds the learner as MR does (copy expert off).
- `scripts/logprob_map.py` (new): the probe's input, the tables and the example maps.
- Modes:
  - `map`: per word of a text after its prompt (measures A1–A8 below);
  - `gate`: T8's own drafts, redrawn with the writer's seeds (measure A9);
  - `beam`: the most probable output (measures B1–B4).

## The prompts (registered)

GEN-A dev, `bench/eval/gen_prompts_dev.tsv`. The rule: the first 30 dev prompts, by index, that are not among the 50 rated prompts (`bench/blind/llm_rating/prompts.json`).

**Prompt ids:** 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 35, 36.

**Texts per prompt**, each its first 600 bytes (T8's own length; the human text cut at the last space before 600 bytes):
- **human**: the real continuation;
- **T8**: `bench/k5_decode/runs/T8.gen.txt` (today's default, seed 20261101 + id);
- **MR**: `bench/k5_mixdec/runs/MR.gen.txt` on `wip/k5-mixdec` (D4's context ratio; it was run at T 1.0, on K1).

The probe's input file is `bench/where_it_breaks/items.tsv`.

## The measures (registered)

**A. The log-probability map.** For every word of every text, given the prompt and the text before it:
- **A1** log2 P under K5 with the overlay: the distribution T8 samples from, before temperature (<unk> and <s> removed; an unknown word gets <unk>'s mass);
- **A2** log2 P under K5 without the overlay (the class floor kept): K5 order 5 alone;
- **A3 which order answered:** the highest order whose context holds the word (5, 4, 3, 2), else the class floor; <unk> marked apart;
- **A4 how far back the context matched:** the highest order whose context K5 has seen at all (4 words back for order 5);
- **A5 the overlay's share:** the highest order of the retrieved document's counts that holds the word in this context, and A1 − A2 in bits;
- **A6** the word's rank in A1's distribution, and K5's most probable word there;
- **A7 the nucleus cut:** was the word inside T8's nucleus (T 0.8, candidates to 0.995 of the mass, then the repeat ban and the copy cap, then nucleus 0.8), the nucleus's size, and the mass above the word;
- **A8 the byte learner** (where built): log2 P of the word's bytes under the learner primed on the retrieved article and the prompt, frozen, reading the text; and the same bytes' context-free prior (the base learner), as MR's ratio uses them.
- **A9 the gate** (T8 only): at each sentence, the 16 drafts the writer drew, redrawn with `K5::draft`, the writer's seed and its stream numbers (the plain gate's sentence numbers, or the discourse writer's free-sentence numbers from 1000). The draft equal to T8's text is the one kept. Reported: the kept draft's rank by fluency (mean log2 P a word) among the 16, the fluency of the kept, the mean and the most fluent draft, and the plain gate's choice recomputed from its terms (a check that the redraw is exact). Sentences no draft reproduces are counted as the planner's fact sentences.

**Marked by reading, before any map was computed.** In each of the 30 T8 texts I marked the first word at which a sentence stops working, one per broken sentence, up to three per text: **grammar** (the clause is abandoned or cannot be parsed) or **sense** (grammatical, but the word makes the sentence false or absurd, e.g. "Auckland is a town located *throughout* the animal kingdom"). The marks are committed with this registration: `bench/where_it_breaks/t8_breaks.tsv` (90 spots).

**Model failures in the human text:** human words with P < 2^-15 under A1, or <unk>. Counted by kind (vocabulary hole, class floor, order 2 only, order 3 or more).

**Tabulated at T8's broken spots, against all T8 words:** the order that answered, the context depth, the overlay, the rank and P. **Sampling into the tail** is a broken word of rank above 10 or P < 0.02. **A high-probability wrong word** is a broken word of rank 1–3 with P ≥ 0.1.

**B. The most probable output**, for the same 30 prompts, no sampling, the same constraints (the repeat ban, the copy cap; <unk> and <s> never written):
- **B1** greedy (width 1) under K5 with the overlay;
- **B2, B3** word-level beam search, widths 8 and 16, ranked by the sum of log2 P; each hypothesis grows one word a step, and the search stops when the best hypothesis passes 460 bytes (the raters' cut);
- **B4** (if the learner is built) width 8 under K5 × the learner's context ratio, MR's form: log2 P_K5 + 0.3 (log2 P_mix(w | context) − log2 P_mix(w)), over each hypothesis's 16 most probable words;
- read for grammar, repetition (beam degeneration) and subject; their mean bits a word under A1 compared with T8's, MR's and the human text's; repeated word 3-grams.

**C. The diagnosis:** the causes ranked by how many of the marked breaks each explains, with counts and examples, and what each points to fixing. Five annotated example maps, human against T8 on the same prompt (prompts 0, 7, 12, 16, 21: chosen now, by id, before the maps).

**Expectations, written before the run** (not a rule; nothing is adopted from this report):
- most of T8's broken words are answered by order 2 or 3, with no order-5 context;
- more than half of the broken words are of rank above 10: sampling into the tail of a wide nucleus;
- the human text has more order-1/floor words and more <unk> than T8 (the 7 MB vocabulary);
- the kept draft's fluency rank is no better than chance (about 8.5), since the gate's fluency weight is 0;
- the beam output is more grammatical than T8 but repeats (beam degeneration), and leaves the subject.

**Cost.** One process: the K5 build on 337 MB (about 11 GB, 3 minutes), the learner's trie (1 GB), 6 threads. Started only with at least 16 GB available. A smoke run on dev prompts 248 and 249 (outside the list) checked the tool before this was written; it is not reported.
