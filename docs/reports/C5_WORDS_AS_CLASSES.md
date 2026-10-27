# C5 / S-1: words as classes in Core Cypha

Odin Loch, 2026-10-27

Tool: `native/tools/cypha_word_classes.cpp` (subcommand `s1`), linked against the built core library.
Design: [SEMANTICS.md](../SEMANTICS.md) §2–3, experiments S-1 to S-3 in §7. Plan: Track C, row C5.
Baseline: [C1_PPMI_SVD_VECTORS.md](C1_PPMI_SVD_VECTORS.md).

## Summary

Each of the top 10,000 enwik8 words became a Cypha class. Each of its 9.87 million occurrences was one observation: a random-indexing vector of its ±5 context. Cypha's own training step was run online for one pass.

**Verdict.**
- **Cypha learns a little meaning here, and no more than the counts it is fed.** Its best word vectors score WordSim-353 0.240, SimLex-999 0.134 and MEN 0.372. Plain averages of the same observations, whitened the same way, score 0.256, 0.101 and 0.358, which is level within noise. PPMI-SVD on the same pairs scores 0.723, 0.265 and 0.732.
- **The library defaults do worse than the counts.** With them, WordSim is 0.017–0.068. Two changes recover count level:
  - a 1/(n+1) step, so each word's differential becomes a running mean instead of a 20-observation moving average;
  - a much slower WorldPrior.
- **Cypha's other parts add nothing that carries over:**
  - The competitor push adds +0.02 to +0.04 rho.
  - The contrastive encoder update lowers similarity.
  - The variances do not give hypernym direction: 47–54% against a frequency baseline of 60.6%.
  - Relation maps on Cypha vectors find 0–12% of capitals in the top 10. PPMI-SVD finds 81%.

**Predictions:** P1 miss, P2 miss, P3 miss. **Gates:** S-1, S-2 and S-3 all fail. PPMI-SVD stays the vector source, and the C1 tool stays.

## 1. What was run

### 1.1 Data

| item | value |
|---|---|
| Corpus | `/root/cdata/enwik8`, cleaned and tokenised by the code copied verbatim from `native/tools/ppmi_svd.cpp`: 11,226,546 tokens, 265,565 types |
| Stream | Top 50,000 words kept, others deleted before windowing (as C1). A line ends a window |
| Classes | Top 10,000 words; 9,866,716 class tokens (87.9% of all tokens) |
| Observation | `h_i = Σ_{0<|j−i|≤5} r(w_j)/|j−i|`, where `r(w)` is a fixed random index vector with d = 256 and 8 entries of ±1 (seed 1). The PPMI-SVD vectors are not used as input |
| Training | One online pass in corpus order, single thread |

### 1.2 Cypha parts used

| Cypha part | Role here |
|---|---|
| `create_fresh_model_root`, `save_cypha_file`, `Cypha::load` | A fresh d = 256 model. `world_lr` and `delta_lr` come from `FreshModelParams` |
| `CyphaDifMemoryState::memory_train` | The whole class update. It scores the slots under the shared v0. It moves the true class differential toward `h − θ0` (`Δ += lr·(h − θ0 − Δ)`, the natural-gradient step under a shared diagonal variance). It pushes the competitors away (softmax weight, capped at 0.5). It applies MDL shrinkage to every slot (λ = 0.001, cold start 8) and updates the WorldPrior θ0, v0 |
| `batch_encode`, `contrastive_update_encoder_w` | With the encoder on: `h = W x`, and the Fisher–Rao contrastive update of W after every miss (true class against the predicted one) |
| `class_mean_and_variance` | Class Gaussians for the encoder update |
| `regression::ridge_fit_bias` | A cross-check of the ridge map written here. They agree to 1e-10 |
| `Cypha::update` (public API) | Cost probe only |

### 1.3 What had to be written here

- **Sampled competitors.** The public `Cypha::update` scores every class. At 10,000 classes and 10M tokens that would take 70–121 hours (§5). Instead, each token swaps the true class and 5 competitors into a 6-slot scratch memory, and `memory_train` runs on those slots. The competitors come from a 4M-entry unigram^0.75 table, as in word2vec. Rows are copied back afterwards. This is SEMANTICS §3's sampled update, built on the unmodified core.
- **A per-class diagonal variance.** The core keeps one shared v0 for every class, so KL between two classes is symmetric and says nothing about direction. Here each class keeps an exponential moving average of `(h − μ_pre)²`, initialised at v0, with the class's own step size.
- **The `--sched inv` step.** The differential's step is `clamp(1/(n_k+1), 0.002, 0.1)`, passed to `memory_train` as its `delta_lr`. This is a scalar form of the `K_a = Σ_a/(n_a+κ0)` step in EMBEDDING_GEOMETRY §4.2.
- **Evaluation:**
  - KL and Fisher–Rao distances between diagonal Gaussians;
  - a vector-output Bayesian ridge map in dual form, with leave-one-out λ and predictive variance (Cypha's regression module has only scalar ridge and MKE experts);
  - the WordNet reader;
  - the count baselines, built from exactly the same observations.

## 2. Word similarity

Spearman rho is computed on the pairs where both words are among the 10,000 classes and have a PPMI-SVD row. That leaves 259 of 353 WordSim pairs, 651 of 999 SimLex pairs and 1,316 of 3,000 MEN pairs. Approximate standard errors are 0.06, 0.04 and 0.03.

The table uses three final runs, all built from the current source:
- **Defaults:** the library defaults (`delta_lr` 0.05 fixed, `world_lr` 0.008) with the encoder frozen.
- **Tuned:** the 1/(n+1) step and `world_lr` 5e-5, encoder frozen.
- **Contrastive:** the library defaults with the encoder on (`enc_lr` 0.002).

| method | WordSim-353 | SimLex-999 | MEN |
|---|---|---|---|
| Cypha defaults: cos(μ), μ = θ0 + Δ | 0.017 | 0.116 | 0.137 |
| Cypha defaults: −symmetric KL | 0.068 | 0.140 | 0.156 |
| Cypha defaults: −Fisher–Rao distance | 0.067 | 0.145 | 0.174 |
| Cypha tuned: cos(μ) | 0.141 | −0.048 | 0.136 |
| Cypha tuned: cos(Δ) | 0.137 | 0.105 | 0.241 |
| **Cypha tuned: cos(Δ) whitened by v0** (the Fisher–Rao metric of the shared v0) | **0.240** | **0.134** | **0.372** |
| Cypha tuned: −symmetric KL | 0.231 | 0.066 | 0.171 |
| Cypha tuned: −Fisher–Rao distance | 0.231 | 0.062 | 0.174 |
| Cypha contrastive: cos(μ) | 0.093 | −0.004 | 0.103 |
| Cypha contrastive: whitened cos(Δ) | 0.125 | 0.054 | 0.114 |
| Cypha contrastive: −symmetric KL | −0.050 | 0.018 | 0.085 |
| (a) random indexing, plain sum | 0.147 | 0.084 | 0.177 |
| (a′) random indexing, mean-centred | 0.208 | 0.066 | 0.223 |
| (a″) random indexing, f^−1/2 context weights, centred | 0.402 | 0.112 | 0.358 |
| **(a‴) random indexing, centred, whitened by the tuned run's v0** (the matched count baseline) | **0.256** | **0.101** | **0.358** |
| Count Gaussian (exact per-word mean and variance): −symmetric KL | 0.235 | 0.072 | 0.179 |
| **(b) PPMI-SVD, 300 d** | **0.723** | **0.265** | **0.732** |

The full-vocabulary PPMI-SVD scores (C1) are 0.672, 0.255 and 0.637. These pairs are the more frequent subset, so they score a little higher.

**Reading the table:**
- Cypha's KL (0.231) equals the exact count Gaussian's KL (0.235). Its whitened differential equals the whitened count mean (0.240 against 0.256), within one standard error on every set.
- With a 1/(n+1) step and a still prior, Cypha's class update computes a running mean of its observations. Its similarity is therefore the similarity of the counts.
- cos(μ) is weak in every run because the shared θ0 dominates every mean.

**Nearest neighbours** (cosine, top 8):

| word | Cypha tuned (Δ) | random indexing, centred | PPMI-SVD |
|---|---|---|---|
| france | poland norway florence switzerland iceland denmark algeria hungary | spain japan italy afghanistan england poland catalonia iraq | belgium paris switzerland luxembourg french spain italy netherlands |
| king | creation collapse emergence restoration suppression establishment aftermath development | duke ministry aragon archbishop kings duchy brandenburg darkness | son kings ii iii reign throne iv prince |
| red | crescent international movement federation icrc oriental committee wolf | white inside between within across tigers yellow monster | blue white yellow black green orange purple colored |
| computer | troll pistol proprietary few dozen lengthy highly nervous | multimedia file database simple dynamic video whereas plus | computers hardware machines computing graphics systems pc ibm |

Cypha groups some words by the position they fill ("the X of": creation, collapse, emergence) or by phrase ("red cross / crescent"), not by meaning.

### 2.1 Ablations (what moves the Cypha numbers)

The extra runs (R3 to R10) used earlier builds of the same file. Their similarity and hypernymy code is identical: the final runs reproduce R1 and R10 exactly. Their relation numbers used an older pair list and are not reported. "n/a" means that build had no (a‴) row.

| run | change from the defaults (encoder frozen unless stated) | cos(μ) WS / SL / MEN | whitened cos(Δ) | −sym KL | matched count (a‴) | hypernym KL, all / matched | var, all / matched | smaller \|Δ\|, all / matched | train acc (6 slots) |
|---|---|---|---|---|---|---|---|---|---|
| F1 | none (library defaults) | 0.017 / 0.116 / 0.137 | 0.038 / 0.092 / 0.103 | 0.068 / 0.140 / 0.156 | 0.276 / 0.136 / 0.340 | 46.9% / 47.5% | 50.5% / 50.1% | 48.7% / 47.5% | 0.439 |
| R5 | fixed step 0.01 | 0.131 / 0.055 / 0.196 | −0.005 / 0.143 / 0.105 | 0.176 / 0.153 / 0.199 | n/a | 44.8% / 46.2% | 49.5% / 51.2% | 46.3% / 47.2% | 0.413 |
| R3 | 1/(n+1) step | 0.126 / −0.058 / 0.129 | 0.035 / 0.165 / 0.125 | 0.238 / 0.063 / 0.178 | n/a | 49.5% / 45.9% | 52.5% / 51.9% | 55.6% / 49.6% | 0.404 |
| R7 | 1/(n+1), world_lr 5e-4 | 0.130 / −0.051 / 0.132 | 0.202 / 0.133 / 0.329 | 0.233 / 0.065 / 0.173 | 0.258 / 0.103 / 0.364 | 50.1% / 45.9% | 51.6% / 51.3% | 56.0% / 49.8% | 0.422 |
| F10 | 1/(n+1), world_lr 5e-5 (the tuned run) | 0.141 / −0.048 / 0.136 | 0.240 / 0.134 / 0.372 | 0.231 / 0.066 / 0.171 | 0.256 / 0.101 / 0.358 | 50.2% / 45.6% | 51.5% / 51.0% | 56.1% / 50.2% | 0.432 |
| R8 | as R7, no competitors (pull only) | 0.142 / 0.052 / 0.162 | 0.185 / 0.113 / 0.286 | 0.232 / 0.068 / 0.174 | 0.258 / 0.103 / 0.364 | 49.5% / 45.7% | 49.6% / 50.8% | 53.7% / 49.0% | (1 slot) |
| R9 | as R7, f^−1/2-weighted observations | 0.070 / 0.066 / 0.126 | 0.180 / 0.060 / 0.263 | 0.161 / 0.006 / 0.126 | (a″) 0.402 / 0.112 / 0.358 | 50.6% / 46.8% | 44.0% / 45.5% | 62.5% / 56.7% | 0.295 |
| R6 | as R3, f^−1/2-weighted observations | 0.052 / 0.057 / 0.074 | −0.023 / 0.055 / 0.027 | 0.163 / 0.010 / 0.128 | (a″) 0.402 / 0.112 / 0.358 | 49.6% / 46.3% | 43.9% / 45.3% | 61.9% / 56.1% | 0.285 |
| F2 | contrastive encoder on (enc_lr 0.002) | 0.093 / −0.004 / 0.103 | 0.125 / 0.054 / 0.114 | −0.050 / 0.018 / 0.085 | 0.213 / 0.067 / 0.231 | 53.8% / 55.1% | 53.8% / 55.2% | 49.2% / 48.8% | 0.418 |

**What the ablations show:**
1. **The fixed step forgets.** With `delta_lr` 0.05, Δk is an exponential average over about the last 20 contexts, so it is noise for every word seen more than a few dozen times. The 1/(n+1) step turns it into the running mean: −sym KL on WordSim goes from 0.068 to 0.238 (F1 to R3).
2. **The moving WorldPrior corrupts old differentials.** Δk is stored relative to θ0 as it was when the class was last updated, so a moving θ0 leaves old differentials stale. Slowing `world_lr` from 0.008 to 5e-5 raises whitened WordSim from 0.035 to 0.240 (R3 to F10).
3. **The competitor push helps a little.** Adding 5 sampled competitors to pull-only training raises whitened rho by +0.017, +0.020 and +0.043 (R8 to R7). It is the one gain from Cypha's machinery over plain averaging. It is not enough to beat the averaged counts.
4. **The contrastive encoder update hurts.** It moved W far from the identity (‖W − I‖_F = 18.0). It lowered every similarity form and the training accuracy (0.418 against 0.439), and it costs 10 times the time. This matches the earlier result recorded in PLAN, where learning `W_enc` online on semantic vectors scored 49.3% against 62.5% for a frozen encoder.
5. **Weighting the observations helps the counts, not Cypha.** f^−1/2 context weights lift the count baseline to 0.402 but lower Cypha (R9, R6). The likely cause: rare context words now carry up to about 340 times the weight of "the", so the observations are heavy-tailed and the Gaussian update follows the outliers. The prior's v0 grew 40-fold (mean 3.3–4.2 against 0.08), and the training accuracy fell to 0.29.

## 3. Hypernymy direction (S-2)

**Pairs.** From Open English WordNet: `globalwordnet/english-wordnet`, `src/yaml/noun.*.yaml`, 26 files fetched from GitHub on 2026-10-27, 71,872 synsets.
- Every direct noun hypernym link gives (hyponym lemma, hypernym lemma) pairs: 107,279 single-word pairs.
- 13,908 distinct pairs have both words among the classes.
- 186 pairs whose reverse also occurs (different senses) were dropped.
- **13,722 pairs scored.** 4,627 of them are frequency-matched (the two words within a factor of 2 in frequency).

Accuracy is the fraction of pairs where the rule picks the right direction. The 95% half-width is ±0.8 points on all pairs and ±1.4 on matched pairs (normal approximation).

| direction rule | defaults (F1), all / matched | tuned (F10), all / matched | contrastive (F2), all / matched |
|---|---|---|---|
| **frequency baseline: more frequent word = hypernym** | **60.6% / 53.1%** | same | same |
| Cypha: KL(hypo ‖ hyper) < KL(hyper ‖ hypo), per-class variance | 46.9% / 47.5% | 50.2% / 45.6% | 53.8% / 55.1% |
| Cypha: total variance, var(hyper) > var(hypo) | 50.5% / 50.1% | 51.5% / 51.0% | 53.8% / 55.2% |
| Cypha: smaller differential norm ‖Δ‖ = hypernym (not registered) | 48.7% / 47.5% | 56.1% / 50.2% | 49.2% / 48.8% |
| Cypha core: epistemic variance v0/(n+1), larger = hypernym | 39.4% / 46.9% | 39.4% / 46.9% | 39.4% / 46.9% |
| Cypha core: KL under the shared v0 | 50.0% (symmetric by construction) | same | same |
| Count Gaussian: KL direction, exact per-word moments | 50.0% / 45.4% | same | same |
| Count Gaussian: total variance | 50.8% / 50.2% | same | same |

**Reading the table:**
- No KL or variance rule reaches the frequency baseline on all pairs.
- The exact count Gaussian fails as well (50.0% and 50.8%). So the spread of a word's random-indexing contexts does not encode how general the word is. The failure is in the observation and the moment estimate, not only in Cypha.
- Vilnis and McCallum learned their variances through a ranking loss; they did not take them from context moments.
- The one reading above a baseline is the contrastive-encoder run on matched pairs: 55.1–55.2% against 53.1%. That is about 2 points, roughly two standard errors, from one run with no bootstrap.
- The core's epistemic variance v0/(n+1) is the frequency rule reversed (39.4% = 100% − 60.6%).
- "Smaller ‖Δ‖ = hypernym" reaches 53.7–62.5% on all pairs in the 1/(n+1) runs, but ‖Δ‖ tracks frequency there (Spearman −0.75 to −0.80 with log frequency). On matched pairs it is at 49.0–56.7%, and it was not registered.

## 4. Relations as maps (S-3)

**Pairs.** Built-in single-word lists, keeping pairs where both words are classes. Pairs whose subject and object are the same word (panama, kuwait, luxembourg) or whose object is only part of a name (haiti → port) were removed. That leaves capital-of with 26 pairs and country → language with 55.

**Map.** `object ≈ A·subject + b`, fitted by Bayesian ridge in dual form on unit-length vectors.
- Cypha's regression module has no vector-output map, so this was written here and checked against `regression::ridge_fit_bias` on one output dimension (difference about 1e-10).
- λ is chosen per fold by closed-form inner leave-one-out. It came out at or near the bottom of the grid: mean λ is 3e-9 to 9e-9, and the grid starts at 1e-8 times a kernel scale below 1. So the map is effectively the minimum-norm interpolant.
- The comparison is the mean-offset method: `x + mean(y − x)`.
- Scoring is leave-one-out, ranking the true object among all 10,000 classes by cosine, with the subject excluded.

| relation | representation | ridge @1 | ridge @10 | offset @1 | offset @10 | median rank (ridge / offset) |
|---|---|---|---|---|---|---|
| capital-of (26) | Cypha defaults (Δ) | 0.0% | 0.0% | 0.0% | 0.0% | 879 / 1539 |
| | Cypha tuned (Δ) | 0.0% | 11.5% | 0.0% | 3.8% | 261 / 345 |
| | Cypha contrastive (Δ) | 0.0% | 0.0% | 7.7% | 11.5% | 1917 / 1998 |
| | random indexing, centred | 0.0% | 7.7% | 7.7% | 19.2% | 237 / 109 |
| | **PPMI-SVD** | 0.0% | **80.8%** | **73.1%** | **96.2%** | 5 / 1 |
| country → language (55) | Cypha defaults (Δ) | 18.2% | 27.3% | 0.0% | 1.8% | 1022 / 1185 |
| | Cypha tuned (Δ) | 16.4% | 40.0% | 1.8% | 25.5% | 18 / 72 |
| | Cypha contrastive (Δ) | 21.8% | 29.1% | 5.5% | 20.0% | 504 / 198 |
| | random indexing, centred | 16.4% | 40.0% | 10.9% | 32.7% | 22 / 28 |
| | **PPMI-SVD** | **36.4%** | **90.9%** | **67.3%** | **96.4%** | 2 / 1 |

The ridge map's @1 is 0% for capitals in every representation, while PPMI-SVD's @10 is 81%. With λ at the floor, the prediction is a blend of the training capitals, and the nearest word is most likely one of those other capitals.

**Abstention.** Answer only the fraction with the lowest ridge predictive variance; the table gives @10 at each coverage. For comparison, the last column uses the cosine of the top candidate as the confidence.

| relation | representation | 100% | 75% | 50% | 25% | AUROC: −variance predicts a hit | AUROC: cosine predicts a hit |
|---|---|---|---|---|---|---|---|
| capital-of | Cypha tuned | 12% | 15% | 8% | 14% | 0.52 | 0.26 |
| | random indexing | 8% | 10% | 8% | 14% | 0.69 | 0.52 |
| | PPMI-SVD | 81% | 85% | **92%** | 86% | **0.71** | 0.73 |
| country → language | Cypha tuned | 40% | 41% | 39% | 36% | 0.49 | 0.62 |
| | random indexing | 40% | 41% | 43% | 50% | 0.56 | 0.72 |
| | PPMI-SVD | 91% | 95% | **100%** | **100%** | **0.85** | 0.90 |

- **PPMI-SVD:** the predictive variance helps. At 50% coverage the miss rate falls from 19% to 8% for capitals and from 9% to 0% for languages.
- **Cypha vectors:** the variance carries no information (AUROC 0.49–0.52), and there is almost nothing correct to keep.

## 5. Cost

| item | value |
|---|---|
| Cleaning, tokenising, vocabulary | 2.7 s |
| Cypha training, encoder frozen (9.87M steps, 6 slots, one thread) | 82.2 s, 8.3 µs per step (244 s when the CPU was shared with two other jobs) |
| Cypha training, contrastive encoder on | 805.6 s, 81.6 µs per step |
| Whole run, encoder frozen | 100 s |
| Peak RAM | 343 MB (includes the corpus, the 50,000 × 300 PPMI-SVD matrix and the count baselines) |
| Model | 20.5 MB as float32 (10,000 × 256 differentials plus per-class variances); θ0, v0 and the 256 × 256 encoder add 0.53 MB |
| Public `Cypha::update`, scoring every class | 354 / 1,003 / 6,948 µs per update at 100 / 400 / 1,600 classes, which extrapolates to 70–121 hours for 10,000 classes × 10M tokens |
| For comparison, PPMI-SVD (C1) | 152 s, 640 MB |

## 6. Predictions (registered in SEMANTICS §7 before the run)

| | prediction | result | score |
|---|---|---|---|
| **P1** | Cypha means reach random-indexing level (WordSim 0.35–0.55) and stay below PPMI-SVD on the same pairs, unless the contrastive update adds discrimination | Best Cypha form: WordSim 0.240 (whitened Δ, tuned run); cos(μ) at most 0.142. Level with the matched count baseline (0.256) but below the band and below the best random-indexing variant (0.402). Below PPMI-SVD (0.723): yes. The contrastive update removed discrimination rather than adding it (0.125) | **Miss** (the ordering held; the level was too optimistic) |
| **P2** | Hypernym direction from KL or variance is 55–70%, at least the frequency baseline | All pairs: 46.9–53.8% against a 60.6% baseline. Only the contrastive-encoder run on frequency-matched pairs reaches 55.1–55.2% (baseline 53.1% on those pairs, about two standard errors) | **Miss** |
| **P3** | The capital-of map gets leave-one-out @10 of 20–60% | Cypha 0%, 11.5% and 0% across the three runs (random indexing 7.7%; PPMI-SVD 80.8%, above the band) | **Miss** |

**Gates (SEMANTICS §7):**
- **S-1** (Cypha ≥ PPMI-SVD on WordSim and SimLex): **fail**. Cypha reaches 0.240 and 0.134 against 0.723 and 0.265.
- **S-2** (above the frequency baseline): **fail** on all pairs. The frequency-matched reading with the encoder on is the only one above its baseline, and it has no bootstrap interval.
- **S-3** (ridge beats the mean offset, and abstention lowers the wrong rate):
  - On Cypha's tuned vectors the ridge does beat the offset (11.5% against 3.8%, and 40.0% against 25.5%), but abstention does not lower the wrong rate. **Fail.**
  - On PPMI-SVD it is the other way round: abstention works, but the offset beats the ridge.

## 7. Verdict

**Does Cypha's machinery learn meaning here?** Weakly.
- Its vectors correlate with human similarity judgements: about 0.24 on WordSim and 0.37 on MEN.
- Some neighbourhoods are right: France's neighbours are countries.
- Everything it learns is what a running mean of the observations already holds. The tuned Cypha matches the matched count baseline on every similarity set.

**Does it beat counts?** No.
- **Against matched counts:** it equals the plain average of the same random-indexing contexts.
- **Against better counts:** it loses badly to the weighted count baseline (0.402) and to PPMI-SVD (0.723).

**Where Cypha differs from averaging, it mostly loses:**
- With the default fixed step, it forgets.
- With a moving WorldPrior, its old differentials go stale.
- The contrastive encoder bends the space the wrong way.
- Its per-class variance (written here) and its epistemic variance carry no is-a direction.

The one Cypha-specific gain is small: the sampled-competitor push adds about +0.02 to +0.04 rho.

**The underlying reason.** A random-indexing observation is a fixed linear projection of the word's co-occurrence row. The mean of fixed features cannot hold more than that projected row does. Beating counts needs one of two things:
- a non-linear reweighting, which is what the logarithm and SVD in PPMI-SVD provide;
- a learned context side, which is what word2vec has.

Here Cypha's Gaussians only average. The encoder update, the one part that could learn the context side, fires only on misses, through a 256 × 256 linear W, and makes things worse.

## 8. Next steps

1. **Keep PPMI-SVD as the vector source.** Do not cut the C1 tool (TOOL_PROFILE_PLAN). Feed SEMANTICS S-4 from PPMI-SVD vectors for now.
2. **Fix the class step in the library, not per tool.** Use a confidence-scaled step `1/(n_k + κ0)`, the `K_a` of EMBEDDING_GEOMETRY §4.2. Store Δ so that a moving θ0 does not make it stale, either by re-centring when θ0 moves or by freezing θ0 after warm-up. These two changes alone were worth +0.2 rho here, and the §4.2 rule needs both.
3. **Test the §4.2 pull–push rule where it can win: teacher-initialised (S-5).** Start the classes at PPMI-SVD vectors (or at a teacher's vectors), then check whether Cypha's online pull–push keeps them (LCI ≥ 0.95) or improves them on held-out text. From scratch on fixed random features, the rule can only match counts.
4. **Learn the context side.** Before another from-scratch S-1 attempt:
   - use a two-sided update (the context items also move, as in §4.2's mirror-image updates for `μ_b` and `μ_c`, and as in SGNS);
   - or use non-linear observations, such as PPMI-weighted sparse contexts.

   Leave the current `contrastive_update_encoder_w` off for semantics.
5. **Hypernymy.** Moments of context vectors do not give breadth, even computed exactly. If is-a matters, learn the variance through a direction-ranking loss, as Vilnis and McCallum did. Report a bootstrap interval and frequency-matched pairs as the main reading.
6. **Relations.**
   - Choose λ by evidence (type-II maximum likelihood). Leave-one-out pins it at the grid floor, which makes the map an interpolant and puts its @1 at 0%.
   - Report the mean offset alongside, since it wins on PPMI-SVD.
   - The predictive variance is already a useful abstention signal on PPMI-SVD (AUROC 0.71–0.85).
7. **Tool.** `cypha_word_classes` now has a subcommand table with `s1` (also the default). It grows into the single semantics tool with `train`, `distill`, `eval` and `export` (EMBEDDING_GEOMETRY §9.3). Its evaluation (similarity, WordNet direction, relation maps with abstention) can become `eval`, run on any exported vectors.

## Reproduction

The build line is in the tool's header. The three final runs:

```
ARGS="/root/cdata/enwik8 EN-WS-353-ALL.txt EN-SIMLEX-999.txt EN-MEN-TR-3k.txt enwik8.vectors.f32 enwik8.vocab.tsv <english-wordnet>/src/yaml"
cypha_word_classes s1 $ARGS --enc-lr 0                                            # defaults, encoder frozen (with the API cost probe)
cypha_word_classes s1 $ARGS --enc-lr 0 --sched inv --world-lr 0.00005 --no-bench  # tuned
cypha_word_classes s1 $ARGS --no-bench                                            # contrastive encoder on
```

The PPMI-SVD inputs come from `native/tools/ppmi_svd.cpp` (C1). The results are deterministic: the final runs reproduced the earlier builds' similarity and hypernymy numbers to the last digit.
