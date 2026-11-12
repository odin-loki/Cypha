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

## Results (2026-11-13)

**The run.** `k5_probe` on the 90 registered texts, 6 threads: 5:43 wall, 2.1 GB peak (K5 built in 74 s; the learner's base primed at 1.94 bits a byte, as D4's). A second map-only run read the beam outputs (`items_beam.tsv`). Files: `bench/where_it_breaks/runs/` (map, gate, beam, and `beam_map.tsv`, the beam outputs' map), `tables.md` (every table, made by `logprob_map.py report`), `examples.md` (the five full maps), `numbers.json`.

**Checks.**
- Replay: the probe's state rewrites every human and T8 text byte for byte (`x1.info.tsv`, replay_ok 1). Inside K5's own sentences, 0.6% of T8's words fall outside the recomputed nucleus (the hash order of tied words, and the lifted ban). Every other sampled word is inside it.
- The gate redraw is exact. 207 of T8's sentences are one of the 16 redrawn drafts, byte for byte. For the 145 plain-gate sentences, the gate's choice recomputed from its terms is the kept draft **145 of 145**.
- The other 51 sentences match no draft. They are the planner's fact sentences. Three prompts (0, 21, 36) are planner sentences from start to cut. In the nine mixed prompts, the free sentences after the facts match again, so the redraw stays exact past them.

### A. What the maps show

**1. T8 lives where K5 is sure; human text does not** (Table 1, all words, first 600 bytes):

| | human | T8 | MR | beam 16 |
|---|---|---|---|---|
| mean bits a word under K5 + overlay | 7.50 | 3.38 | 4.10 | 1.35 |
| answered by order 5 | 0.116 | 0.508 | 0.374 | 0.813 |
| answered by order 2 or the class floor | 0.303 | 0.053 | 0.084 | 0.007 |
| the word is K5's most probable | 0.234 | 0.518 | 0.404 | 0.843 |
| inside T8's nucleus (T 0.8, nucleus 0.8) | **0.481** | 0.916 | 0.791 | — |
| <unk> | 0.077 | 0.012 | 0.012 | 0 |
| learner's bits a word (primed, frozen) | 10.47 | 7.89 | 7.55 | 7.72 |

- K5 gives the real continuation 7.5 bits a word. **Half of the real words fall outside the nucleus T8 samples from.**
- T8's text costs 3.4 bits a word, less than half the human text's. The byte learner agrees (7.9 against 10.5 bits a word). Both models prefer the broken text to the real one.
- The overlay (the retrieved document) helps T8 by 0.87 bits a word and the human text by 0.04. It pulls the writer to the article's phrases, not toward what the human writes.

**2. Model failures on human text** (Table 3): 562 of 2,999 human words (0.187) are <unk> or below 2^-15.
- <unk>: 231 (0.077). Almost all are names: Holden, Vauxhall, Isuzu, Opel, Chevrolet, Schwarzenegger, Coleman, Indira, "Göteborgs". The 7 MB vocabulary cannot write the subject's own names.
- The class floor (no context held the word): 148 (0.049). Order 2 only: 120 (0.040). Orders 3+, still below 2^-15: 63.
- These are true words that K5 calls nearly impossible. They do not cause T8's breaks (T8 never writes <unk> in its own sentences). They show that K5 has no way to write the real continuation.

**3. Where T8 breaks** (Table 4; 90 spots marked by reading before the maps, at most 3 a text):
- **Source.** 63 spots are in K5's own sentences (2,975 words: 54 spots in the plain gate's drafts, 9 in discourse free sentences). **27 are in the planner's fact sentences** (773 words: 21% of T8's words, 30% of the breaks).
- **Order and context at the spot.** 23% of the broken words were answered by order 5 (all T8 words: 51%). 70% had a 4-word context K5 had seen (all T8 words: 76%). So the context was there, but the word had never followed it. In K5's own sentences, **48% of the broken words are "splices"** (the 4-word context was seen, the word never after it), against 25% of all their words. (Human text has 25% splices too, so a splice alone is not an error.)
- **Sampling into the tail, or a wrong word K5 believes** (K5's own sentences, 63 spots):
  - **high-probability wrong word** (rank 1–3, P ≥ 0.1): **24** (38%);
  - middle (rank 2–10, P 0.02–0.1: one of a handful of near-equal words): **21** (33%);
  - **tail draw** (rank above 10, or P < 0.02): **18** (29%).
  - In the planner's fact sentences the tail dominates: 16 of 27 (59%), mean 7.9 bits at the spot.
- **The nucleus is wide where it breaks:** a median of 12 words at the broken spots, against 6 at all T8 words.
- **Not causes:** sentence starts (4 of 90 spots are in a sentence's first two words; Table 2), and <unk> in K5's own sentences (0 of 63).
- Examples from the writer's own sentences:
  - believed: "The process uses the nitric acid ( **2** nd century BC )": "2" has P 0.53, rank 1, order 5, from the overlay. "the indigenous people of the central **nervous** system": P 0.08, rank 2, order 4.
  - spliced: "the history of the **best** pitchers in the game" (order 3); "Divisions of Bangladesh **is** a city" (order 4).
  - tail: "began **his** party membership fees" (rank 12); in a planner sentence, "For almost a **hero**" (P 0.0004, rank 96).

**4. The gate chooses topic over form** (Table 5, 207 redrawn sentences):
- The kept draft's fluency rank among 16 is 6.99 on average (chance 8.5). It is the most fluent draft in 19% of sentences and in the less fluent half in 39%.
- Mean log2 P a word: kept −2.88, the 16 drafts' mean −3.12, the most fluent −2.03. **The gate gives up 0.85 bits a word of fluency** for a mean z of +3.79 on prompt association plus topic. Its fluency weight is 0 by default.
- Example (prompt 2, sentence 0). Kept: "Ground and pound is also a reference to the ground and the sets and a strong local art or using effective striking." (−2.93). Passed over: "Ground and pound is also used as a tactic to win their fights." (−1.34).

**5. The byte learner sees the breaks only weakly.** At the 90 spots its log2 P is −9.89, against −7.89 for all T8 words. But its context ratio (MR's term) is below its median at only 57% of the spots (chance 50%). It is a local byte model; it does not know the clause is wrong either.

### B. The most probable output

| text (30 prompts) | bits a word | order 5 | in the retrieved document | repeated word 3-grams | sentence ends per 100 words |
|---|---|---|---|---|---|
| greedy | 2.10 | 0.763 | 0.515 | 0.031 | 6.3 |
| beam 8 | 1.44 | — | — | 0.009 | 4.7 |
| beam 16 | 1.35 | 0.813 | 0.743 | 0.008 | 4.5 |
| beam 8 × learner ratio (MR's form) | 1.79 | 0.755 | 0.687 | 0.033 | 10.8 |
| T8 (sampled) | 3.38 | 0.508 | 0.445 | 0.016 | 6.3 |
| human | 7.50 | 0.116 | 0.190 | 0.007 | 4.2 |

- **No beam degeneration.** The repeat ban stops loops: beam 16 repeats no more than human text. Greedy drifts to stock phrases ("The first of these was the …", "The first............") and repeats four times as much.
- **The beam stays on the subject.** Three words in four lie in the retrieved document's own n-grams. It writes the article back, 12 words at most (the cap), then jumps.
- **It is not coherent.** It is training text spliced where a few words overlap:
  - "Ground and pound or ground' n' pound) is a stand - up comedian and actor. He was born in Paris, France."
  - "In other words businesses have liabilities. … The main categories of assets are usually listed in the Court of Appeal of England and Wales."
  - "Alia was born in year 10, 191 after the foundation of the University of Oxford in the United Kingdom."
- **Read sentence by sentence** (prompts 1, 3, 4, 5, 9, 10, 11, 13, 17, 24; whole = grammatical and sensible): beam 16 has **10 of 40** sentences whole (0.25), T8 **9 of 61** (0.15). The most probable text is only a little better than the sampled one.
- With the learner's ratio, the beam writes more and shorter fragments (10.8 sentence ends per 100 words): "Verus, Marcus Aurelius, Hadrian's nephew. As an imperial prince, Verus, Hadrian."

**So this is a model problem first and a sampling problem second.** The text K5 likes best breaks the same way T8 does: at the joins between remembered spans. Sampling adds the tail draws (18 of the 63 breaks in K5's sentences). The planner adds its own (27 of 90).

### C. The causes, ranked

| # | Cause | Breaks it explains (of 90) | Evidence | What it points to fixing |
|---|---|---|---|---|
| 1 | **K5 sees 4 words.** It cannot tell a sound clause from two remembered spans joined at a shared word. | **45**: words in K5's own sentences that K5 ranks in its top 10 (24 high-probability, 21 middle); 30 of the 63 are splices | the beam output breaks the same way (0.25 whole sentences); human text costs 7.5 bits a word and half of it lies outside the nucleus | A judge of the whole clause, not the next word: score each draft with a longer context (an ∞-gram longest match over the 337 MB, or a clause-level check) and reject drafts with a weak join. No sampling setting fixes this. |
| 2 | **The planner's fact sentences** | **27** (30% of the breaks from 21% of the words) | 38% of their words lie outside K5's nucleus, 29% rank above 10, 4.8% are <unk>; wrong entities ("Barbara Anderson" after Holdens; prompts 0, 21, 36 are all planner) and class fills ("a town located throughout the animal kingdom") | Gate fact sentences by K5's fluency (refuse a fill whose words fall outside the nucleus) and by the entity's link to the prompt. Read `--no-k5-plan` against T8 with raters. |
| 3 | **The nucleus is too wide where K5 is unsure** | **18** tail draws in K5's sentences | 12 words in the nucleus at the breaks, 6 overall | A relative cut (min-p). At 0.1 of the top word's P it removes 14 of the 63 breaks and 9.5% of K5's sampled words. Small and cheap: worth one arm. |
| 4 | **The gate ignores fluency** | adds to 1 and 3 (not separable) | the kept draft is the most fluent in 19%; 0.85 bits a word given up | A fluency floor, not a weight: drop drafts that hold a word below a P threshold or a weak join, then choose by topic. (A fluency weight alone, F1 and F2 in §13.71, did not win.) |
| 5 | **The 7 MB vocabulary** | 0 of T8's own breaks | 231 of 2,999 human words are <unk>, mostly the subject's names | A vocabulary from the 337 MB, or a channel for names, so the writer can stay on the subject's names. |
| — | Sentence-boundary resets | 4 | breaks are not at sentence starts | none |

**Five example maps** (human against T8 on the same prompt). Each word: log2 P under K5 + overlay / the order that answered (F the class floor, U <unk>) / N if outside T8's nucleus. The full maps are in `bench/where_it_breaks/examples.md`.

- **Prompt 0 (Holdens in New Zealand).** Human: "While[-7/4/N] the[-3/4] Holden[-18/2/N] name[-14/F/N] … GM[-16/2/N] products … Vauxhall[U/N] , Bedford[-12/3/N] , Isuzu[U/N]". T8 (all planner): "the biography[-9/4] of New[-11/2/N] Zealand Barbara[-24/F/N] Anderson[-10/2/N] was injured[-8/3] as the surrounding[-12/3/N] dock[-12/3/N]". The human text is full of names K5 cannot write. The planner fills its skeleton with an unrelated entity at −24 bits.
- **Prompt 7 (Gothenburg University).** Human: "founded as G[-17/2/N] ö teborgs[U/N] h[-17/2/N] ö gskola[U/N] ( Gothenburg[U/N] University College ) in 1891[-12/3/N]". T8 (planner): "He was born in Chalmers[U/N] University[-17/3/N] of Technology in Sweden[-11/2/N] … in Gothenburg[U/N] , New[-6/4/N] York". Every T8 word here comes from a skeleton; "New York" is a fill K5 itself ranks low.
- **Prompt 12 (Dune).** Human: "Unable[-13/3/N] to resist[-9/4/N] … immediately[-12/3/N] see[-16/F/N] her resemblance[-16/2/N] to Jessica". T8 (plain gate): "His[-6/3] successor[-7/4] , the[-3/5] sister[-4/3] of[0/4] the[-2/5] same[-6/4] time[-8/4] she also was a desert planet". No word is below −8 bits and none is outside the nucleus. Every step is likely; the clause is spliced at "of the same time" and "was a desert planet". This is cause 1.
- **Prompt 16 (East Pakistan, 1971).** Human: "This move had further[-13/2/N] alienated[-10/2/N] India[-17/F/N] … accelerated[-16/2/N] a previously[-13/2/N] cautious[-21/F/N] new direction". T8 (planner): "Dhaka[U] is a town[-7/5] and municipality[-2/5] in[0/5] the[0/5] county[0/5] of[0/5] East[-6/5/N] Pakistan , India". The skeleton is a whole remembered 5-gram path ("a town and municipality in the county of"): fluent, and false.
- **Prompt 21 (the California recall).** Human: "After Arnold[-19/F/N] Schwarzenegger[U/N] announced[-10/2/N] his candidacy[U/N] , Coleman[U/N] stated[-12/4/N] …". T8 (planner): "He is also a term[-9/4] exceeding[-10/3/N] California[-12/F/N] … the California Department of Defense began[-9/4/N] his[-7/2/N] party[-11/2/N] membership". The human text's subject words are <unk>. T8's breaks sit at order 2–3 backoffs.

**Expectations, scored:**
- most broken words answered by order 2 or 3, with no order-5 context: **miss**. 70% had the 4-word context; orders 3–4 answered 61%. The break is a splice after a seen context, not a missing context.
- more than half of the broken words of rank above 10: **miss** (26% overall; 29% in K5's own sentences).
- human text has more floor and <unk> words than T8: **hit** (0.061 and 0.077, against 0.004 and 0.012).
- the kept draft's fluency rank no better than chance: **miss** (6.99 against 8.5: topic and fluency correlate a little).
- the beam more grammatical but repetitive, leaving the subject: **partly**. It is a little more often whole (0.25 against 0.15), not repetitive (the repeat ban), on the subject (it copies the article), and spliced.

**In one line:** the writer breaks because K5 judges each word by the four before it, so it cannot see that a clause was joined from two remembered spans. Sampling and the planner add breaks on top, but the most probable text breaks too.
