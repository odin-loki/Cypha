# K5 structure experts, meta-mixed by a learned sentence ranker

**Odin Loch** · 2026-11-05 · GENERATION_PLAN §13.33

**Odin:** "K5 does sentence structure basically. Why don't we work on an algorithm/mixer/expert for each type of structure or relation and metamix them all together. Let's do this." And: "We have more than enough data to try this."

The experts, the arms, the rules and the predictions below were written and committed before any run of them. Results are added under them.

## Why sentence choice

- §13.31: four blind raters give the gated K5 2.09 of 10, human text 8.50. K5 writes sentence-shaped text that does not connect.
- §13.15–13.24: every model blended into K5's next-word probabilities made the text blander.
- So the experts here never touch K5's word probabilities. K5 writes candidate sentences. The experts score each whole sentence. A learned mixer joins the scores and picks one.

## Setup (Odin's PC, WSL)

- Branch `wip/k5-structure` from main 39b0d84. 8 threads at most (three other sessions share the PC).
- **K5:** the current default (nucleus 0.8, 256 exchange classes as the floor, the 12-word copy cap), trained on `/mnt/d/cypha_llm/fuzzy/train.txt` (7 MB) only. K5 at 330 MB did not help writing (K5_SCALE.md).
- **Other text:** `/mnt/d/cypha_llm/k5scale/clean/train_all.txt`: 329.6 MB, 100,794 Wikipedia documents of enwik9 past enwik8, one per line. Overlap with GEN-A and SCRIBE's ranges is already removed. K5 never saw it. It is cut by documents, by byte position:
  - **M**, meta-training: the documents that start in the first 60 MB;
  - **H**, held-out: the documents that start in [60, 65) MB;
  - **S**, the experts' statistics: the documents from 65 MB to the end (about 265 MB).
  - No document is in two parts. The experts' statistics never see M or H, so an expert cannot know a held-out document's own names or numbers.
- **Data size:** the statistics are built three times: from the first 10 MB of S, the first 50 MB, and all of S (265 MB). Sentence bits are reported for each.
- **Amendment (2026-11-05, before any run of an arm).** A timing test on 1,000 targets of H found that only about 16% of real sentences pass the target filter below: 70% hold a word outside K5's vocabulary, and the repeat ban takes 40% of the rest. M at 60 MB would give only about 55,000 boundaries. So the cut moves: **M is [0, 110) MB, H is [110, 120) MB, and S is [120 MB, the end)** (about 210 MB). The statistics sizes become 10 / 50 / 210 MB. Nothing else changes. Drafting costs about 0.6 CPU seconds a boundary (8 drafts), so M takes 2–3 hours on 6–8 threads.

## Sentences

- A sentence is cut as K5's gate cuts a candidate: K5's tokens, ended by ".", "!" or "?" once it holds at least 24 bytes, or at 240 bytes.
- Every sentence, real or written, is held as K5's tokens and rendered with K5's own spacing (`piece`). So real and written sentences differ in their words, never in their whitespace.

## The training signal

- **Windows.** Each document of M (and H) is cut into windows of 7 sentences: a prompt of 1 sentence, then up to 6 more. The next window starts after the last.
- **Boundaries.** In a window, sentence j (j = 0..5 after the prompt) is a target. Its context is the prompt and sentences 0..j−1.
- **Candidates.** K5 writes N = 8 drafts from that context, with the gate's rule for streams and stops. The real sentence joins them: 9 candidates. Choosing at random costs log₂ 9 = 3.17 bits.
- **No unknown words in the target.** A target is kept only if every word of it is in K5's vocabulary. K5 never writes `<unk>`, so otherwise "has a rare word" would tell the real sentence apart for free. The context may hold unknown words.
- **The target passes K5's own bans** (amendment, 2026-11-05, before any run). K5's drafts can never repeat a 12-byte window of the text before them (the repeat ban), and never hold 13 consecutive words of K5's training text (the cap). A real sentence can. So "repeats the context" would mark the real sentence for free, and at writing time no draft can do it. A target is kept only if K5 could have written it: in vocabulary, no banned repeat, within the cap.
- The order of the boundaries is shuffled once (seed 20261105) before online training.
- **A second signal (P, for "pseudo"):** the drafts only, without the real sentence. The target is the draft closest to the real sentence (the F1 of their content words, ties to the lower k). This is exactly the choice the writer faces: all its candidates are drafts. Its metric is "draft bits": −log₂ P(target) among 8 (log₂ 8 = 3 bits at random).

## The experts

Each expert scores one candidate sentence given the prompt and the paragraph so far. Each gives one or a few numbers. They live in `native/include/cypha/lm/structure/`.

| Expert | Structure | What it computes |
|---|---|---|
| **k5** | syntax | the mean log₂ probability per word under K5 (with `<unk>` and `<s>` removed, as the gate's fluency) |
| **pmi** | the current gate | the gate's mean prompt association (positive PMI of co-occurrence in K5's text). Today's writer picks by this alone. |
| **class** | grammar skeleton | the mean log₂ probability per token of the sentence's exchange classes under a class 5-gram (K5's 256 classes; absolute discounting, interpolated), trained on S, continuing from the previous sentence's last 4 classes |
| **topic** | what the paragraph is about | LDA, from `wip/k5-discourse`, when it lands |
| **grid** | who each sentence is about | the entity grid, from `wip/k5-discourse`, when it lands |
| **relations** | typed slots | (a) the share of the sentence's names that already occur in the context; (b) the mean, over its names, of the best document-level PPMI with a context name, counted in S; (c) slot fit: the mean over typed slots of log₂ P(type ∣ the two words before) − log₂ P(type), counted in S. Types: person, place, year, number, other name. A name is typed from S: place if it follows "in", "at", "from", "near" or "of" often; person if its document's next pronoun is often he or she. |
| **reference** | pronouns | (a) the mean log₂ P(pronoun ∣ the paragraph's main entity) over the sentence's subject pronouns (he, she, it, they and their cases); the main entity is the most frequent name of the context; P is counted in S as the first subject pronoun after a name; (b) a pronoun with no name or noun before it in the context (a flag) |
| **time** | years and numbers | (a) the mean log₂ P(bucket of the gap to the context's nearest year), counted in S between a sentence's years and the years before it in its document; a separate bucket for "the context has no year"; (b) the mean log₂ P(number shape) (digits, commas, decimal point) |
| **transitions** | how sentences open | log₂ P(opening type ∣ the two previous sentences' types), a Markov chain counted in S. Types: connective, pronoun subject, "The …", date or time phrase, named subject, other. |
| **rhythm** | form | log₂ P(length bucket ∣ the previous sentence's length bucket) (ECDF buckets from S); brackets and quotes balanced (a flag); the gate's markup share |
| **novelty** | no copying, no echo | the longest run of tokens shared with K5's training text (up to 13), and with the context |

Notes fixed before the runs (2026-11-05):
- **topic** and **grid** are `wip/k5-discourse`'s `lda.hpp` and `entity_grid.hpp`, merged in. topic: LDA's mean log₂ p_topic(w ∣ the context's topics) / p(w) over the candidate's content words. grid: the grid's log₂ probability change when the candidate joins (full), and its continuation score (kept entities only). The grid is counted on S. LDA (100 topics, 200 sweeps, AD-LDA on 2 threads) is trained on the first 10 MB of S for S10 and on the first 50 MB of S for S50 and S210 (LDA on 210 MB would take hours).
- The k5 feature is clipped at −25 bits per word on average: a few real words get almost no mass from K5.

Experts are added only if the data says so: an expert stays in the mix if removing it costs held-out sentence bits.

**A scoring interface** (for `wip/k5-mcmc`): `score(prompt, paragraph so far, candidate)` returns every expert's numbers and the mixed score of the one-layer ranker (a log-linear energy). The candidate set is not needed for that score.

## The meta-mixer

- **The ranker.** Each candidate i has features f_ij. They are scaled by each feature's standard deviation over M's candidates (z_ij). Its score is s_i = Σ_j w_j z_ij, and P(i) = softmax(s).
- **Learning.** Online, one pass over M's boundaries in the shuffled order, by exponentiated gradient on log loss with signed weights (EG±: w = w⁺ − w⁻, Σ (w⁺ + w⁻) = U). The gradient is g_j = Σ_i P(i) z_ij − z_real,j; w⁺_j ← w⁺_j e^(−η g_j), w⁻_j ← w⁻_j e^(η g_j), then the total is put back to U. Then the weights are frozen and read on H.
- **Arms:**
  - **M1:** one weight set. U and η are chosen on M's own prequential loss (the loss of each boundary before learning from it), from U ∈ {2, 4, 8, 16} × η ∈ {0.002, 0.005, 0.01, 0.02, 0.05}.
  - **M2:** 12 weight sets, chosen by sentence index after the prompt (0, 1, 2, 3+) × the prompt's length (≤ 130, ≤ 200, > 200 characters), as in PAQ.
  - **M3:** a second layer. Three first-layer rankers, with sets chosen by (none), (index × prompt length), (the previous sentence's opening type). Their candidate distributions are mixed linearly, the mixture's weights learned by exponentiated gradient on log loss.
  - **M4:** M3 with a mixture over learning rates (MetaGrad style): each first-layer ranker at η ∈ {0.002, 0.005, 0.01, 0.02, 0.05}, all 15 joined by the second layer.
  - **L0** (a reference, not a candidate for the writer): the same softmax ranker with unconstrained weights fit by full-batch gradient descent with a small L2, on M. It shows how much a linear ranker can get.
- **Baselines:**
  - **each expert alone:** M1 with only that expert's features;
  - **uniform:** every feature at the same weight, signed by its natural direction (higher is more real-like), with one shared scale fit on M;
  - **pmi alone** is today's gate.
- The mix arm for the writer is chosen by M's prequential loss, not by H.

## The metric: sentence bits

- On H's boundaries: −log₂ P(the real sentence) among the 9 candidates, averaged over boundaries. Also top-1: the share of boundaries where the real sentence gets the highest score.
- For each expert alone, for uniform weights, for M1–M4 and L0, at each statistics size (10 / 50 / 265 MB).
- **Draft bits** for the P signal, the same way among the 8 drafts.
- **Which experts the mixer leans on:** the weights of M1 (in z units), and the ablation of each expert (M1 trained again without it; the rise in H's sentence bits).
- 95% intervals by bootstrap over H's boundaries (1,000 resamples).

## The writer

- **Flag:** `sample --k5 … --k5-struct FILE` (the experts' statistics and the frozen mixer). `--k5-struct-temp T`: 0 (default) picks the argmax; T > 0 samples from softmax(s / T). The name is not `--k5-meta`, which `wip/k5-discourse` uses for its MBR pool.
- K5 writes its gate's 8 candidates per sentence (the same streams as today's gate). The mixer picks one, instead of the pmi score.
- The 12-word copy cap holds inside every candidate, as today.
- With `--k5-struct` off, `sample`'s output stays byte-identical to main's (checked with `cmp` on all 250 dev GEN lines).

**Writing arms** (GEN-A dev, 250 prompts, seed offset 0):
- **W0:** today's K5 (the baseline);
- **W1:** the chosen mix, trained with R (real among drafts), argmax;
- **W2:** W1 sampled at T 0.5;
- **W3:** the chosen mix trained with P (drafts only), argmax;
- **W4:** W1 with 16 candidates per sentence.
- **W5** (added 2026-11-05, before any writing run or held-out read): the chosen mix trained with R, but **without the k5 expert** (`run --drop k5`). Why: a smoke test on 2,112 boundaries from inside M (scored on themselves, so not a result) showed K5's own probability alone picks the real sentence 92% of the time, because drafts come from K5's nucleus and real sentences do not. The mix gave k5 a large negative weight. As a writer, that mix would pick K5's least probable draft. W3 (the P signal) is one guard against this; W5 is the other.
- **W6** (added 2026-11-05, before any writing run or held-out read): the chosen mix trained with R, **without the topic expert** (`run --drop topic`). Why: §13.34's blind raters marked LDA topic rescaling down (−0.22 to −0.42) while SCRIBE v2 and v3.1 rose with it. Topic words can fool SCRIBE. If W1 beats W6 on SCRIBE but not with the raters, topic is acting as a "more topic words" knob. The topic weight and its ablation are reported in any case.

Every arm's dev GEN file is saved under `bench/k5_structure/` and pushed, for the four blind Claude raters in the cloud (50 dev prompts).

## Rules

1. **The mix:** the chosen mix arm must beat both the best single expert and uniform weights on H's sentence bits (at the largest statistics size), each by a 95% bootstrap interval that excludes 0.
2. **The writer** becomes a recommended K5 option only if all of these hold:
   - the best writing arm at seed 0 and W0 are run at 4 fresh seed offsets (1000, 2000, 3000, 4000);
   - its 4-seed mean of SCRIBE v3.1 is above W0's, and the intervals do not overlap (the interval of a 4-seed mean is the mean of the 4 per-seed 95% bootstrap bounds);
   - its 4-seed mean of SCRIBE v2 does not fall by more than 0.02;
   - the copy cap holds (verbatim13 0 at every seed) and copy32 rises by at most 0.02;
   - the cloud raters confirm it (their rule is theirs).
3. **GEN-A test is read once,** only for an adopted arm.
4. Tuning uses M and GEN-A dev only. H is read for the metric, after every choice on M is made.

## Predictions

- **P1:** K5 alone reaches about 2.0–2.5 sentence bits, with a **negative** weight: the real sentence is less probable under K5 than drafts drawn from K5's nucleus.
- **P2:** novelty, class and rhythm are each better than 3.0 bits alone; pmi alone is about 2.9–3.1 (it barely tells real from drafted).
- **P3:** the learned mix reaches 1.2–1.8 bits, top-1 50–70%, and beats the best single expert by 0.2–0.6 bits. Rule 1 holds (about 85%).
- **P4:** M2–M4 gain less than 0.05 bits over M1. L0 is within 0.05 bits of the best EG arm.
- **P5:** the mixer leans most on k5 (negative), novelty, class and rhythm. Relations, reference and time each gain less than 0.03 bits in the ablation; they fire on few sentences.
- **P6:** the statistics from 265 MB beat those from 10 MB by 0.02–0.10 sentence bits.
- **P7:** the R-trained writer (W1) picks less probable drafts. SCRIBE v3.1 moves by −0.01 to +0.03; v2 falls by 0 to 0.04 (v2 likes predictable text); copy32 falls. The P-trained writer (W3) is safer for v2. The writing rule holds for some arm (about 30%).
- **P8:** the raters see a gain under 0.5 points over today's K5 (2.09). The text still does not connect: sentence choice among 8 K5 drafts cannot add what no draft has.

## Results (2026-11-05)

### The data

- **H:** 10,572 boundaries (17,090 targets in vocabulary, 6,501 of them then repeat-banned, 17 over the cap). **M:** 119,850 boundaries (71,248 repeat-banned, 118 over the cap). 8 drafts each, so 9 candidates.
- Drafting M took 2.5 hours on 6 threads (9,140 s, 1.1 GB). The statistics took 42 s (10 MB), 62 s (50 MB) and a few minutes (210 MB). The grid took 16 s, LDA 100 s (10 MB) and 614 s (50 MB) on 2 threads. Features take about 1 minute per 120,000 boundaries on 8 threads. A `run` takes 70 s.
- A bug was found and fixed before any reported number: `features` shared one line buffer across threads. The first features files were rebuilt; no result used them.

### Sentence bits (signal R: the real sentence among 9; chance 3.17 bits)

Held out on H, 95% bootstrap intervals. Statistics from 210 MB of S, except the size columns.

| Arm | 10 MB | 50 MB | **210 MB** [95%] | top-1 (210 MB) |
|---|---:|---:|---:|---:|
| k5 alone | 0.415 | 0.415 | 0.415 [0.390, 0.440] | 0.927 |
| pmi alone (today's gate) | 2.887 | 2.887 | 2.887 [2.867, 2.907] | 0.321 |
| class alone | | | 3.062 [3.050, 3.074] | 0.269 |
| relations alone | 2.557 | 2.435 | 2.312 [2.286, 2.340] | 0.535 |
| reference alone | | | 3.117 [3.111, 3.123] | 0.497 |
| time alone | | | 3.165 [3.163, 3.168] | 0.426 |
| transitions alone | | | 3.171 [3.170, 3.171] | 0.287 |
| rhythm alone | | | 2.883 [2.867, 2.896] | 0.282 |
| novelty alone | 2.177 | 2.177 | 2.177 [2.146, 2.208] | 0.609 |
| topic alone | 1.772 | 1.375 | 1.375 [1.341, 1.409] | 0.710 |
| grid alone | | | 2.397 [2.373, 2.421] | 0.444 |
| uniform | 2.986 | 2.837 | 2.724 [2.700, 2.747] | 0.331 |
| M1 one set | 0.170 | 0.132 | 0.116 [0.101, 0.129] | 0.977 |
| M2 12 sets | | | 0.112 [0.099, 0.125] | 0.978 |
| M3 second layer | | | 0.111 [0.098, 0.123] | 0.978 |
| **M4 learning-rate mixture (chosen)** | 0.166 | 0.128 | **0.113 [0.101, 0.124]** | 0.977 |
| L0 batch fit | 0.173 | 0.138 | 0.120 [0.108, 0.130] | 0.977 |

(k5, pmi and novelty read only K5's text, so they do not change with S. topic's LDA is the same at 50 and 210 MB.)

- **Rule 1 holds** at every size. M4 is chosen by M's prequential loss (0.1074). Against k5 alone: −0.301 [−0.324, −0.280]. Against uniform: −2.611 [−2.636, −2.585].
- **But the task is nearly solved by one artefact.** K5's own probability alone picks the real sentence 92.7% of the time: the real sentence averages −8.26 bits a word under K5, the drafts −5.09. Drafts come from K5's nucleus; real sentences do not. So "real among drafts" mostly teaches "pick the improbable one".
- **Data size pays, a little:** 0.166 → 0.128 → 0.113 bits from 10 to 210 MB. The gain comes from relations (2.56 → 2.31 alone) and topic (1.77 → 1.38).

**M1's weights** (z units; U 16, η 0.01), and the **ablation** of each expert (M1 trained again without it):

| Expert | Weights | Held-out bits without it (change vs M1 0.116) |
|---|---|---:|
| k5 | k5 −3.65 | 0.720 (+0.604) |
| pmi | +0.27 | +0.007 |
| class | +1.68 | +0.050 |
| relations | ent_old +0.30, ent_new −0.16, ent_pmi +0.43, slot_fit +0.07 | +0.006 |
| reference | pron_agree +0.15, pron_orphan −0.11 | +0.001 |
| time | year_fit +0.08, n_years +0.01 | +0.001 |
| transitions | open +0.13 | +0.001 |
| rhythm | len +0.36, balanced +0.03, markup −0.01 | +0.002 |
| novelty | match_train +0.11, match_ctx +0.10 | 0.000 |
| **topic** | **+1.23** | **+0.044** |
| grid | grid_full +1.25, grid_cont −0.23 | +0.012 |

**Without k5** (`--drop k5`, W5's mix): M4 0.707 bits [0.681, 0.732], top-1 0.844. It fails rule 1 (k5 alone is better). Its weights move to topic +1.69, match_train −1.52 (real sentences copy less of K5's text), len +0.86, balanced +0.66, ent_pmi +0.44, ent_new −0.32. Its ablations: topic +0.473, novelty +0.245, rhythm +0.126, relations +0.058, grid +0.022, reference +0.006, pmi +0.004, time +0.004, transitions +0.002, class +0.001.

**Without topic** (`--drop topic`, W6's mix): M4 0.151 bits [0.137, 0.166]; rule 1 still holds.

### Draft bits (signal P: the draft closest to the real sentence among 8; chance 3.000 bits)

- Every expert alone: 2.995–3.012 bits. The chosen mix (M3): **2.9915** [2.9883, 2.9947], top-1 0.149 (chance 0.125). Rule 1 holds formally (−0.0035 against grid alone, −0.017 against uniform), but the gain is 0.008 bits.
- The experts cannot tell which of K5's drafts is nearest the real sentence. (Top-1 above 0.125 for some single experts, up to 0.56 for time, comes from ties: a feature equal on all drafts picks the first, and the pseudo target ties to the first.)

### Writing (GEN-A dev, 250 prompts)

Seed offset 0:

| Arm | SCRIBE v2 [95%] | SCRIBE v3.1 [95%] | copy32 | verbatim13 | time / 250 prompts (8 threads) | RAM |
|---|---:|---:|---:|---:|---:|---:|
| **W0** today's K5 | **0.498** [0.469, 0.527] | **0.483** [0.466, 0.500] | 0.0380 | 0 | 6.4–8.1 min (2 threads) | 0.5 GB |
| W1 R mix (M4), argmax | 0.411 [0.393, 0.432] | 0.467 [0.456, 0.479] | 0.0188 | 0 | 2.6 min | 1.9 GB |
| W2 R mix, T 0.5 | 0.408 [0.386, 0.430] | 0.468 [0.457, 0.479] | 0.0179 | 0 | 3.1 min | 1.9 GB |
| W3 P mix (M3) | 0.463 [0.428, 0.502] | 0.480 [0.455, 0.506] | 0.0598 | 0 | 1.7 min | 1.9 GB |
| W4 R mix, 16 candidates | 0.414 [0.395, 0.434] | 0.480 [0.471, 0.489] | 0.0192 | 0 | 5.0 min | 1.9 GB |
| W5 R mix without k5 | 0.453 [0.431, 0.474] | 0.483 [0.471, 0.495] | 0.0144 | 0 | 2.4 min | 1.9 GB |
| W6 R mix without topic | 0.394 [0.376, 0.412] | 0.459 [0.448, 0.471] | 0.0158 | 0 | 4.6 min | 1.9 GB |

(Times are wall clock for the whole `gen_eval.sh` run: sampling and both SCRIBE versions, on a PC loaded by three other sessions. W0 ran on 2 threads, the others on 8. RAM is peak RSS: the structure model's statistics at 210 MB take about 1.4 GB.)

**Confirmation** of the best seed-0 arm (W5, which ties W0 on v3.1) against W0, seed offsets 1000–4000:

| Seed offset | W0 v2 | W5 v2 | W0 v3.1 | W5 v3.1 | W0 copy32 | W5 copy32 |
|---|---:|---:|---:|---:|---:|---:|
| 1000 | 0.496 | 0.441 | 0.470 [0.451, 0.488] | 0.482 [0.472, 0.493] | 0.0409 | 0.0184 |
| 2000 | 0.501 | 0.430 | 0.474 [0.458, 0.492] | 0.473 [0.462, 0.484] | 0.0442 | 0.0183 |
| 3000 | 0.505 | 0.437 | 0.476 [0.459, 0.493] | 0.479 [0.468, 0.490] | 0.0496 | 0.0190 |
| 4000 | 0.493 | 0.443 | 0.480 [0.464, 0.497] | 0.485 [0.476, 0.495] | 0.0428 | 0.0188 |
| mean | 0.499 | **0.438** | 0.475 [0.458, 0.493] | 0.480 [0.470, 0.491] | 0.044 | 0.019 |

- **Rule 2 fails.** v3.1 rises by 0.005, but the intervals overlap. v2 falls by 0.061 (the bound is 0.02). The cap holds (verbatim13 0 at every seed) and copy32 halves.
- **Nothing is adopted. GEN-A test was not read.**
- The dev GEN files are in `bench/k5_structure/` (W0–W6 at seed 0; W0 and W5 at the 4 confirmation seeds), for the cloud's blind raters, who are the deciding judge.

### Predictions

- P1 (k5 alone 2.0–2.5 bits, negative weight): sign hit, size missed. k5 alone is far stronger, 0.41 bits.
- P2 (novelty, class, rhythm better than 3.0 alone; pmi 2.9–3.1): hit for novelty (2.18) and rhythm (2.88), missed for class (3.06); pmi hit (2.89).
- P3 (mix 1.2–1.8 bits, top-1 50–70%, beats the best single expert by 0.2–0.6; rule 1 holds): the gain over the best expert hit (0.30) and rule 1 held. The level missed: 0.11 bits and 98%, because k5 alone is so strong.
- P4 (M2–M4 gain less than 0.05 over M1; L0 within 0.05): hit (−0.003 to −0.005; L0 +0.004).
- P5 (leans on k5, novelty, class, rhythm; relations, reference, time each under 0.03): half. It leans on k5, class, topic and grid. Novelty and rhythm matter only once k5 is gone. Reference, time and transitions are each under 0.003 (hit). Relations is under 0.03 with k5 in the mix (hit) and 0.058 without it.
- P6 (210 MB beats 10 MB by 0.02–0.10): hit (0.053).
- P7 (W1 picks less probable drafts; v3.1 −0.01 to +0.03; v2 falls 0–0.04; copy32 falls; W3 safer for v2; rule holds for some arm about 30%): direction hit, size missed. v3.1 −0.016, v2 −0.087 (more than predicted), copy32 halved. W3 is safer for v2 (0.463), but it copies more (0.060). No arm passed.
- P8 (raters under +0.5): for the cloud.

### What it means

1. **The training signal is the weak point, not the mixer.** The mixer learns (rule 1 holds, prequential and held-out agree, the learning-rate mixture and the sets add a little). But "the real sentence among K5's drafts" is a game won by noticing that the real one is improbable to K5, has rare names that co-occur with the prompt's, and stays on the prompt's topic. At writing time every candidate is a K5 draft, so the first cue says "pick the strangest draft". That lowers v2 (W1 −0.087).
2. **The drafts-only signal finds nothing.** Among 8 K5 drafts, none of these experts tells which is nearest the real sentence (0.008 bits out of 3). K5's 8 drafts per sentence are all about equally far from what a person would write. Choosing among them cannot add what no draft has (P8's reasoning).
3. **Topic carries the most real signal** once k5 is gone (+0.47 bits), and W6 (no topic) writes worse than W1 on both SCRIBE versions. §13.34 showed that topic words can fool SCRIBE, so only the raters can say whether W1 or W5 reads better. With k5 in the mix, the topic weight is +1.23 z.
4. **What would help next:** candidates that differ more than K5's 8 nucleus draws (other K5 variants, a content plan, retrieval), so that a ranker has something to choose. Or a signal of human preference among drafts (the raters' scores) instead of real-vs-drafted.

### For the merge

- Branch `wip/k5-structure`. Code: `native/include/cypha/lm/structure/` (experts.hpp, mixer.hpp, writer.hpp), `native/tools/lm/k5_structure.cpp` (stats, discourse, drafts, features, run, selftest; `k5_structure selftest` passes), a few lines in `word_kn.hpp` (the `StructureWriter` hook, the public `draft()`, `cut_bytes()`, `train_ids()`) and in `cyphalm_trie.cpp` (`--k5-struct`, `--k5-struct-temp`).
- `wip/k5-discourse` is merged in (lda.hpp, entity_grid.hpp, mbr.hpp, stopwords.hpp, its word_kn.hpp changes).
- **Default output is unchanged:** `sample --k5` on all 250 dev prompts is byte-identical to main 39b0d84 (`cmp`), both before and after the merge.
- **For wip/k5-mcmc:** `structure::Scorer(model, k5).score(prompt, paragraph, candidate)` returns every feature and the mixed score (a log-linear energy). The R-trained model carries K5's "improbable is real" weight, so an MCMC energy should use the `--drop k5` model (`runs/S210.Rnok5.chosen.model`) or add K5's own log probability with a positive sign.
- Mixer models: `bench/k5_structure/runs/*.model` (their `stats` line points to `~/k5s_data/S210.stats` on Odin's PC). Statistics, drafts and features: `~/k5s_data/` on Odin's PC (not pushed: 0.2–1 GB each).
