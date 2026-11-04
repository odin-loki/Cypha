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

## Results

(None yet. Added below after the runs.)
