# S-1b: learned contexts for words as classes

**Odin Loch** · 2026-10-28

Tool: `native/tools/cypha_word_classes.cpp`, now with subcommands `s1`, `train`, `distill`, `eval` and `export`, linked against the core library.
Design: [SEMANTICS.md](../SEMANTICS.md) §7 ("Why, in hindsight"), [EMBEDDING_GEOMETRY.md](../EMBEDDING_GEOMETRY.md) §4.2, §5.1, §9.3. Library: `native/include/cypha/semantics/geometry.hpp` ([QE_GEOMETRY_LIBRARY.md](QE_GEOMETRY_LIBRARY.md)).
Baselines: [C1_PPMI_SVD_VECTORS.md](C1_PPMI_SVD_VECTORS.md), [C5_WORDS_AS_CLASSES.md](C5_WORDS_AS_CLASSES.md).

## Summary

S-1 failed because Cypha's class mean only averaged fixed random context vectors. SEMANTICS §7 named three fixes. Each was run here as a variant, with the same pairs, the same corpus and the same baselines as C5.

**Verdict.**
- **Learned contexts work. Two tables of Cypha Gaussians, trained by a logistic objective with natural-gradient steps (variant b), reach PPMI-SVD level.**
  - C5 pairs: WordSim **0.750** [0.692, 0.797] and SimLex **0.270** [0.192, 0.347], against PPMI-SVD's 0.723 and 0.265. C5's best Cypha row was 0.240 and 0.134.
  - C1 pairs: WordSim 0.683 and SimLex 0.246, against 0.672 and 0.255.
- **The gate fails, narrowly.** It asks for P(Δ > 0) ≥ 0.95 against PPMI-SVD on both test sets. (b) has P = 0.941 on WordSim and 0.604 on SimLex. The point estimates are at or above PPMI-SVD's. They are not significantly above it.
- **PMI-weighted inputs (variant a) do not help.** Cypha's update still only averages a 256-dimensional random projection.
  - With per-event PPMI weights, WordSim is 0.215.
  - Even when the class mean is exactly the projection of the word's PPMI row (a2), the count mean reaches only 0.351, and Cypha's update 0.256.
- **The §4.2 pull–push rule (variant c) learns nothing from a random start in the time available.** MEN dev is 0.073 after one epoch, and the vectors are still random. Started from (b)'s tables, it keeps their structure but does not improve it: MEN 0.711, WordSim 0.723, SimLex 0.218, against (b)'s 0.730, 0.736 and 0.234 for the same word table.
- **Relations come back.** On (b)'s vectors, capital-of ridge @10 is 69% (PPMI-SVD 81%) and offset @10 is 100% (PPMI-SVD 96%). Country → language ridge @10 is 95% (PPMI-SVD 91%).
- **Is-a does not come back.** (b)'s Fisher variance tracks frequency. "Larger variance = hypernym" scores 39.1%, against the frequency baseline's 60.6%.
- **Export for QE5 works.** `gguf_mine.py compare` reads the Cypha and PPMI-SVD vectors. On words, pair Spearman is 0.714 and kNN@10 Jaccard 0.166; on SIF sentence vectors, 0.931 and 0.489. k-means word classes (k = 256 and 1,024) are written for the class-context expert.
- **Distillation works from a good start, not from a random one.** With PPMI-SVD as a stand-in teacher, `distill` reaches pair Spearman 0.935 starting from (b), but 0.770 from random vectors.

**Predictions:** R0 hit, PA miss, PA2 miss, PB miss (WordSim 0.750 is 0.010 above the band), PC miss, PC2 hit, PG hit, PD miss, PE hit, PF miss (offset above the band), PX miss (the vectors agree more and are more isotropic than predicted). 4 hits, 7 misses.

**Gate (S-1b replaces PPMI-SVD as the vector source):** fails on the significance bar. PPMI-SVD stays the source. (b) is a level, Cypha-learned alternative, and Odin decides whether level is enough (§9).

## Predictions (registered before the runs)

Protocol fixed before the runs:
- **Pairs.** The C5 pairs: both words among the top 10,000 enwik8 words (WordSim 259, SimLex 651, MEN 1,316). The C1 pairs (50,000 words) are reported as a second view.
- **Dev and test.** MEN is the dev set: every choice of a setting is made on MEN alone. WordSim-353 and SimLex-999 are the test sets, read once per variant, after its setting is fixed.
- **Intervals.** Paired bootstrap over pairs, 2,000 resamples, 95% percentile intervals, and P(Δ > 0) for each variant against PPMI-SVD on the same pairs.
- **Gate.** S-1b replaces PPMI-SVD as the vector source only if its best variant is ≥ PPMI-SVD on both WordSim and SimLex on the C5 pairs, with P(Δ > 0) ≥ 0.95 on each.

Dev grids, fixed before the runs (MEN rho on the C5 pairs decides; ties go to the cheaper setting):
- (b): prior variance Σ0 ∈ {0.01, 0.025, 0.1} at 1 epoch; then the best Σ0 at 5 epochs; word means W against W + C read from every run.
- (c): (τ, η) ∈ {(0.05, 10), (0.1, 10), (0.1, 30), (0.2, 30)} at 1 epoch; then the best at more epochs if time allows.
- Fixed everywhere: d = 256, the 50,000-word stream of C1, dynamic window 5, 5 negatives, subsampling t = 1e-4, seed 1.

| ID | Prediction |
|---|---|
| R0 | The unchanged tool reproduces C5's tuned row exactly (whitened cos(Δ) 0.240 / 0.134 / 0.372; PPMI-SVD 0.723 / 0.265 / 0.732), and `ppmi_svd` reproduces C1's headline (0.672 / 0.255 / 0.637) |
| PA | (a) PPMI-weighted observations, C5's tuned Cypha update: whitened cos(Δ) rises above 0.240 to WordSim 0.40–0.60 and SimLex 0.15–0.25. It stays level (±0.03) with the plain mean of the same observations, and below PPMI-SVD |
| PB | (b) Two tables under a logistic (SGNS) objective, natural-gradient steps with a diagonal Fisher precision: WordSim 0.62–0.74, SimLex 0.22–0.32 on the C5 pairs, within ±0.06 of PPMI-SVD on both |
| PC | (c) The §4.2 pull–push rule (softmax at τ, in-batch and hard negatives) on two tables: WordSim 0.50–0.70, SimLex 0.18–0.30; not above (b) |
| PG | The gate fails: no variant has P(Δ > 0) ≥ 0.95 against PPMI-SVD on both WordSim and SimLex |
| PD | `distill` with the PPMI-SVD vectors as a stand-in teacher (top 10,000 words, from a random start): pair Spearman with the teacher ≥ 0.80, and WordSim and SimLex within 0.05 of the teacher's own |
| PE | Is-a on the best variant: variance or KL direction stays below the frequency baseline (60.6%) on all pairs; the variance rule lands at 35–50%, because a Fisher-precision variance tracks frequency |
| PF | Capital-of on the best variant: ridge @10 40–85% and offset @10 60–95% (PPMI-SVD: 81% and 96%) |
| PA2 | (Registered after (a) missed, before its run.) (a2) weight each context by PPMI(w, c) · n_w / C(w, c), so that the plain mean of the observations is exactly the random projection of the word's PPMI row. The matched count mean reaches WordSim 0.35–0.55. Cypha's whitened Δ falls more than 0.10 below it, because the weights are heavy-tailed (C5's R9 lesson) |
| PC2 | (Registered after (c) from scratch scored MEN 0.073, before its run.) (c) started from (b)'s two tables, τ = 0.05, η = 100, κ0 = 1000, one epoch: the word means' MEN stays within ±0.02 of (b)'s W (0.730); WordSim and SimLex stay below (b)'s W + C |
| PX | Export: `compare` of PPMI-SVD against the best Cypha word vectors gives pair Spearman 0.4–0.7 and kNN@10 Jaccard 0.10–0.30; on SIF sentence vectors, pair Spearman 0.6–0.85. `geometry` on the Cypha word vectors gives a random-pair cosine floor of 0.02–0.20 |

## 1. What was run

### 1.1 Data and protocol

| item | value |
|---|---|
| Corpus | `/root/cdata/enwik8`, cleaned and tokenised by C1's code (copied verbatim): 11,226,546 tokens, 265,565 types |
| Stream | Top 50,000 words kept (96.56% of tokens); the rest deleted before windowing. A line ends a window |
| Word pairs | WordSim-353, SimLex-999 and MEN, from `mfaruqui/eval-word-vectors` (sha256 in Reproduce) |
| C5 pairs | Both words among the top 10,000: WordSim 259, SimLex 651, MEN 1,316 |
| C1 pairs | Both words among the top 50,000: WordSim 350, SimLex 982, MEN 2,800 |
| WordNet | Open English WordNet `src/yaml/noun.*.yaml`, 26 files, 71,872 synsets (fetched 2026-10-27, the same count as C5) |
| Machine | 4 cores shared with other jobs, 15 GB. Every run single-threaded under `nice -n 5`, one heavy job at a time. Timings are wall-clock on a loaded machine |

### 1.2 The variants

| variant | what learns | where the update is |
|---|---|---|
| C5 tuned (the S-1 baseline) | 10,000 Cypha classes. Observation: a fixed random-indexing context vector, d = 256. 1/(n+1) step, `world_lr` 5e-5 | Core Cypha `CyphaDifMemoryState::memory_train` (libcypha_core) |
| (a) PPMI-weighted inputs | As C5 tuned; each context's index vector is weighted by PPMI(w, c) (C1's PPMI: window 5, 1/distance, α = 0.75) | The same core update |
| (a2) projected PPMI rows | As (a), with weight PPMI(w, c) · n_w / C(w, c). The plain mean of a word's observations is then exactly Σ_c PPMI(w, c) r(c) | The same core update |
| (b) learned contexts, logistic | Two tables of diagonal Gaussians N(μ, Σ): 50,000 word classes and 50,000 contexts. Skip-gram negative-sampling objective (Levy & Goldberg 2014: its optimum is shifted PMI). Each event and each negative is one logistic observation. Update: the online diagonal Laplace step, precision += p(1 − p) x² (the Fisher information), μ += Σ · gradient (the natural-gradient step). Σ0 is the prior and so the first step size | `train_sgns` in the tool; storage in `geometry.hpp` `ItemStore` |
| (c) §4.2 pull–push | The same two tables in one `ItemStore`. Anchor = a word, positive = one context from its window. Candidates: 5 unigram^0.75 contexts, 4 in-batch contexts from 61–64 examples back, and the 2 most similar of 20 sampled contexts (hard negatives). Softmax at τ, the δ = 0.1 mask, K = Σ/(n + κ0), mirror updates, online variance | `geometry.hpp` `pull_push_step`, unchanged |

Subsampling (t = 1e-4) and the dynamic window (b ~ U{1..5}) follow word2vec. (b) uses every pair in the window. (c) uses one context per token, because the library step costs about 110 µs (§6). With every pair, one epoch would take about an hour.

### 1.3 Reproducing C1 and C5 first

Both baselines were rebuilt before any change. Both reproduce to the last printed digit (prediction R0 hit).

| run | WordSim | SimLex | MEN | C1 / C5 report |
|---|---|---|---|---|
| `ppmi_svd`, min_count 1, p = 0.5 (C1 pairs 350 / 982 / 2,800) | 0.6723 | 0.2547 | 0.6371 | 0.672 / 0.255 / 0.637 |
| `ppmi_svd`, p = 0 | 0.6810 | 0.2624 | 0.6335 | 0.681 / 0.262 / 0.633 |
| `cypha_word_classes s1`, tuned: whitened cos(Δ) (C5 pairs 259 / 651 / 1,316) | 0.240 | 0.134 | 0.372 | 0.240 / 0.134 / 0.372 |
| same run: PPMI-SVD on the C5 pairs | 0.723 | 0.265 | 0.732 | 0.723 / 0.265 / 0.732 |
| same run: matched count baseline (a‴) | 0.256 | 0.101 | 0.358 | 0.256 / 0.101 / 0.358 |

The hypernymy table (13,722 pairs, frequency baseline 60.6%) and the relation tables (capital-of: Cypha 11.5%, PPMI-SVD 80.8%) also reproduce exactly.

**Which PPMI-SVD row C5 used.** It is C1's headline vectors: all of enwik8, min_count 1, p = 0.5, 300 dimensions, seed 12345. C5 scores them only on the pairs whose words are both among the top 10,000. Those pairs are more frequent words, so they score higher (0.723 against 0.672 on WordSim).

The refactored tool was checked against the unchanged one on the tuned run. The output is byte-identical apart from timings and one row label.

## 2. Word similarity (the gate)

### 2.1 C5 pairs

Spearman rho with a 95% paired-bootstrap interval. Δ is against PPMI-SVD on the same pairs and the same resamples. MEN is the dev set: every setting below was chosen on it.

| vectors | WordSim-353 (259) | Δ WordSim, P(Δ > 0) | SimLex-999 (651) | Δ SimLex, P(Δ > 0) | MEN, dev (1,316) |
|---|---|---|---|---|---|
| **(b) W + C**, Σ0 0.3, 5 epochs | **0.750** [0.692, 0.797] | +0.027 [−0.008, +0.061], **0.941** | **0.270** [0.192, 0.347] | +0.005 [−0.029, +0.037], **0.604** | 0.756 [0.736, 0.776] |
| (b) W (the word means alone) | 0.736 [0.669, 0.788] | +0.013 [−0.031, +0.054], 0.720 | 0.234 [0.152, 0.311] | −0.031 [−0.072, +0.004], 0.045 | 0.730 [0.706, 0.752] |
| (c) from scratch, 1 epoch | not read (dev failed) | | not read | | 0.073 [0.021, 0.126] |
| (c) from (b)'s tables, 1 epoch | 0.723 [0.654, 0.776] | −0.001 [−0.052, +0.047], 0.488 | 0.218 [0.137, 0.297] | −0.047 [−0.093, −0.006], 0.013 | 0.711 [0.686, 0.736] |
| (a) PPMI-weighted observations | 0.215 [0.092, 0.326] | −0.509 | 0.110 [0.028, 0.193] | −0.155 | 0.308 [0.258, 0.355] |
| (a2) projected PPMI rows | 0.256 [0.146, 0.362] | −0.467 | 0.082 [0.000, 0.158] | −0.183 | 0.241 [0.189, 0.294] |
| C5 tuned (S-1's best: whitened cos(Δ)) | 0.240 [0.119, 0.350] | −0.484 | 0.134 [0.052, 0.214] | −0.131 | 0.372 [0.324, 0.416] |
| C5 matched count mean (a‴) | 0.256 [0.131, 0.371] | −0.468 | 0.101 [0.022, 0.178] | −0.163 | 0.358 [0.309, 0.403] |
| **PPMI-SVD 300 d** | **0.723** [0.657, 0.777] | | **0.265** [0.189, 0.341] | | 0.732 [0.710, 0.753] |

Reading the table:
- **(b) is level with PPMI-SVD.** It is above on every point estimate. Only MEN, the dev set it was tuned on, is significantly above (+0.024 [+0.010, +0.037]).
- **W + C beats W.** Adding the context table helps most on SimLex (+0.036). This is the "w + c" of Levy, Goldberg and Dagan (2015).
- **(a) and (a2) are far below.** They sit at C5's level.

### 2.2 C1 pairs (top 50,000 words)

| vectors | WordSim-353 (350) | Δ, P(Δ > 0) | SimLex-999 (982) | Δ, P(Δ > 0) | MEN, dev (2,800) |
|---|---|---|---|---|---|
| (b) W + C | 0.683 [0.615, 0.741] | +0.011 [−0.026, +0.047], 0.724 | 0.246 [0.180, 0.307] | −0.009 [−0.041, +0.021], 0.279 | 0.669 [0.649, 0.689] |
| (b) W | 0.627 [0.550, 0.694] | −0.046, 0.028 | 0.202 [0.134, 0.264] | −0.053, 0.002 | 0.613 [0.589, 0.636] |
| PPMI-SVD | 0.672 [0.607, 0.732] | | 0.255 [0.190, 0.316] | | 0.637 [0.614, 0.658] |

On the full vocabulary, (b) is level again. Rare words are where W alone is weakest.

### 2.3 Dev grids (MEN, C5 pairs)

**(b).** One epoch each. The registered grid's best (Σ0 = 0.1) sat at its edge, so the grid was extended upward on dev.

| Σ0 | epochs | W | W + C |
|---|---|---|---|
| 0.01 | 1 | 0.304 | 0.451 |
| 0.025 | 1 | 0.535 | 0.597 |
| 0.1 | 1 | 0.694 | 0.712 |
| 0.3 | 1 | 0.676 | 0.736 |
| 1.0 | 1 | diverged (NaN in 49,996 of 50,000 rows) | diverged |
| **0.3** | **5** | 0.730 | **0.756** |

Σ0 is the prior variance, so it sets the first step (word2vec's 0.025 is the second row). The precision then grows with each word's own Fisher information. That is why a larger start works: the step shrinks by itself as evidence comes in.

**(c), from a random start.** The first registered point (τ 0.05, η 10, κ0 10) scored MEN 0.073 after one epoch (6,407,796 events, 702 s).
- The vectors had not left their random start. The random-pair cosine was 0.002 with sd 0.063, which is the spread of random unit vectors in 256 dimensions. The neighbours of "france" were "venue invasion bc unless …".
- The step K = Σ/(n + κ0) scales with Σ ≈ 2/d. So in 256 dimensions an η of 10–30 is about 15 times weaker than the library test's η = 20 in 32 dimensions. The other three registered points used η ≤ 30, so the grid was stopped after the second point had started. Steps were checked on a 5 MB prefix of enwik8 instead (MEN, 558 pairs among that prefix's top 5,000 words; (b) on the same prefix: W 0.224, W + C 0.351):

| (τ, η, κ0) | 5 MB MEN | | (τ, η, κ0) | 5 MB MEN |
|---|---|---|---|---|
| (0.05, 10, 10) | 0.035 | | (0.05, 160, 1000) | 0.002 |
| (0.05, 100, 10) | −0.043 | | (0.05, 1000, 1000) | 0.064 |
| (0.05, 100, 100) | 0.055 | | (0.05, 4000, 1000) | 0.076 |
| (0.05, 300, 100) | 0.111 | | (0.2, 4000, 1000) | 0.068 |
| (0.1, 300, 100) | 0.099 | | (0.05, 1600, 10000) | 0.002 |

No setting learns. The reason, as far as these runs show:
- From a random start, a word's pull is the mean of its contexts' random directions. That mean is about 1/√m long for m distinct contexts, so the steady drift is small against each step's noise.
- K = Σ/(n + κ0) counts every event as one unit of evidence, even while the partners are still random. So the step decays before any structure forms.
- (b) escapes both. Its contexts start at zero, and its precision grows only as fast as the partners' squared size p(1 − p) x². Early events are cheap in evidence.

**(c), started from (b).** On the 5 MB prefix, from (b)'s W at 0.224: (0.05, 10, 1000) gave 0.226, (0.05, 100, 1000) 0.245 and (0.05, 100, 10) 0.007. A gentle rate keeps or slightly improves the start, and a fast one destroys it. On the full corpus, (0.05, 100, 1000) for one epoch (6,407,796 events, 764 s) took W's MEN from 0.730 to 0.711. The rule keeps the structure it is given but does not add to it.

### 2.4 Why (a) fails

From the two `s1` runs (C5 pairs; WordSim / SimLex / MEN):

| form | (a) PPMI(w, c) weights | (a2) PPMI(w, c) · n_w / C(w, c) weights | C5 tuned (plain) |
|---|---|---|---|
| Cypha whitened cos(Δ) | 0.215 / 0.110 / 0.308 | 0.256 / 0.082 / 0.241 | 0.240 / 0.134 / 0.372 |
| Cypha −symmetric KL | 0.155 / 0.141 / 0.182 | 0.018 / 0.006 / 0.039 | 0.231 / 0.066 / 0.171 |
| plain mean of the same observations, centred (a″) | 0.209 / 0.067 / 0.246 | **0.377 / 0.128 / 0.388** | 0.208 / 0.066 / 0.223 |
| the same, whitened by the trained v0 (a‴, matched) | 0.238 / 0.095 / 0.344 | 0.351 / 0.126 / 0.376 | 0.256 / 0.101 / 0.358 |
| sampled-slot training accuracy (6 slots) | 0.586 | 0.260 | 0.432 |

- **(a).** Per-event PPMI weights barely change the mean, because frequent contexts still dominate it: a context's share is P(c | w) · PPMI(w, c). The whitened count mean is 0.238, level with C5.
- **(a2).** Here the count mean is exactly the random projection of the PPMI row. It reaches only 0.377. So 256 random dimensions with 8 nonzeros keep about half of what the SVD keeps (0.723).
- **Cypha's update loses a further 0.095–0.135** on (a2), against the matched count mean. The weights are heavy-tailed (up to n_w · 5 · PMI), and the competitor push follows the outliers. This is C5's lesson 5 again. Training accuracy falls to 0.260.
- **What this shows.** A fixed projection, however it is weighted, caps what averaging can learn. The context side has to be learned, which is what (b) does.

### 2.5 Nearest neighbours (cosine, top 8, (b) W + C against PPMI-SVD, top 10,000 words)

| word | (b) W + C | PPMI-SVD |
|---|---|---|
| france | spain belgium italy paris french netherlands germany switzerland | belgium paris switzerland luxembourg french spain italy netherlands |
| king | kings throne son queen vii iii iv reigned | son kings ii iii reign throne iv prince |
| red | white blue yellow cross green sox black crescent | blue white yellow black green orange purple colored |
| computer | computers hardware computing software ibm machines graphics console | computers hardware machines computing graphics systems pc ibm |
| water | dioxide salt soil nutrients liquid surface oxygen vapor | liquid dioxide vapor mixture nitrogen oxygen waters salt |
| run | runs running home ran programs operating batter inning | runs running ran operating home hit inning hitting |
| good | bad that's evil luck thing hope you things | bad that's better excellent you it's doing your |
| bank | banks banking monetary fund gaza imf west central | banks fund monetary funds debt investment financial loans |

C5's Cypha neighbours of "king" were "creation collapse emergence …". Here they are the king's own words, and "bank" shows both its senses (money and the West Bank).

## 3. Relations and is-a on (b)

### 3.1 Relations as maps (S-3; C5's leave-one-out ridge map and mean offset)

Candidates are the top 10,000 words. The map and its abstention are C5's code, unchanged.

| relation | vectors | ridge @1 | ridge @10 | offset @1 | offset @10 | median rank (ridge / offset) | AUROC, −variance → hit@10 |
|---|---|---|---|---|---|---|---|
| capital-of (26) | (b) W + C | 0.0% | 69.2% | 73.1% | **100%** | 5 / 1 | 0.722 |
| | (b) W | 0.0% | 57.7% | 38.5% | 100% | 10 / 2 | 0.491 |
| | (a) | 0.0% | 0.0% | 0.0% | 3.8% | 537 / 651 | — |
| | PPMI-SVD | 0.0% | 80.8% | 73.1% | 96.2% | 5 / 1 | 0.714 |
| country → language (55) | (b) W + C | 40.0% | **94.5%** | 72.7% | **100%** | 3 / 1 | 0.590 |
| | (b) W | 36.4% | 89.1% | 74.5% | 96.4% | 3 / 1 | 0.755 |
| | (a) | 27.3% | 45.5% | 3.6% | 12.7% | 13 / 75 | 0.487 |
| | PPMI-SVD | 36.4% | 90.9% | 67.3% | 96.4% | 2 / 1 | 0.848 |

- On the C1 view (candidates among 50,000 words), capital-of has 76 pairs. (b) W + C gets ridge @10 34.2% and offset @10 84.2%, against PPMI-SVD's 38.2% and 55.3%. Country → language (65 pairs): ridge @10 84.6% against 86.2%, offset @10 98.5% against 89.2%.
- **The ridge @1 is still 0% for capitals everywhere.** This is C5's finding: leave-one-out puts λ at the grid floor, so the map interpolates the training capitals.
- **Abstention.** On capital-of, the ridge variance is as useful for (b) W + C as for PPMI-SVD (AUROC 0.72 and 0.71). At 75% coverage, @10 rises from 69% to 85%. On country → language it is weaker (0.59 against 0.85).

### 3.2 Hypernymy direction (S-2)

13,722 Open English WordNet pairs, 4,627 of them frequency-matched (C5's pairs). The variance is (b)'s diagonal Σ of the word table, that is 1/precision.

| rule | all pairs (±95%) | frequency-matched (±95%) |
|---|---|---|
| frequency baseline: more frequent word = hypernym | 60.6% (±0.8) | 53.1% (±1.4) |
| (b): KL(hypo ‖ hyper) < KL(hyper ‖ hypo) | 39.1% (±0.8) | 46.0% (±1.4) |
| (b): total variance, var(hyper) > var(hypo) | 39.1% (±0.8) | 46.0% (±1.4) |
| (b): smaller mean norm = hypernym (**not registered**) | 61.9% (±0.8) | 56.5% (±1.4) |

- The Fisher variance is close to the frequency rule reversed (100% − 60.6% = 39.4%): frequent words gather precision, so they have small Σ. The core's epistemic variance did the same in C5 (39.4%).
- (c) started from (b) has the library's online variance instead (the spread of an item's partners about it). It scores 60.6% / 53.3% by KL and 61.1% / 54.7% by total variance. That is at most 1.6 points above the frequency baseline, and it was not registered either.
- **One unregistered reading.** "Smaller ‖μ‖ = hypernym" is 3.4 points above the frequency baseline on matched pairs (56.5% against 53.1%), and 1.3 points above it on all pairs. On the C1 view (39,462 pairs, 8,604 matched) it is 67.1% against 66.5%, and 55.9% against 53.3%. It was not registered, and the intervals are normal approximations. It needs a registered retest with a paired bootstrap before it counts.

## 4. Distillation (`distill`, §5.1) with PPMI-SVD as a stand-in teacher

The teacher is C1's PPMI-SVD (300 d), on its 10,000 most frequent words. The student is a 256-d `ItemStore`, trained by `geometry.hpp` `distill_step` (§5.1, α = 1) with the library test's settings: η = 20, κ0 = 10, τ_T = τ_S = 0.05. Each anchor and epoch has two softmaxes: a local set (teacher top 10, 10 from ranks 10–59, 4 random) and a global set (24 random). There were 60 epochs. Agreement is the §7.1 suite in `geometry.hpp`, on 183,553 stratified pairs (5 near, 5 middle and 10 random per item). Force KL is at τ 0.05 over each tenth anchor's top 60 plus 24 random.

| student start | pair Spearman | kNN@10 Jaccard | CKA | force KL | WordSim (C5 pairs) | SimLex | MEN | time | peak RAM |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.770 | 0.374 | 0.319 | 0.171 | 0.586 [0.506, 0.660] | 0.196 [0.120, 0.271] | 0.584 | 352 s | 879 MB |
| (b) W + C | **0.935** | **0.694** | **0.752** | **0.060** | 0.742 [0.679, 0.792] | 0.265 [0.189, 0.341] | 0.738 | 409 s | 933 MB |
| (teacher, PPMI-SVD) | 1 | 1 | 1 | 0 | 0.723 | 0.265 | 0.732 | | |

- **From random, 60 epochs are not enough.** The mean KL was still falling: 0.991 at epoch 10, 0.608 at epoch 60. The student is 0.137 below the teacher on WordSim.
- **From (b), the student moves to the teacher and keeps (b)'s level.** Its SimLex equals the teacher's (Δ −0.000 [−0.017, +0.015]).
- **What this means for QE5.** Start Cypha's items at their own (b) vectors and then distil Qwen's geometry. Do not start from random (SEMANTICS S-5).
- `distill` reads any teacher `.npy` plus index, matched on the `text` column. The Qwen vectors from `gguf_mine.py embedtexts` drop in unchanged.

## 5. Export for QE5

`export` writes, for any vector set:
- `<name>_vectors.npy`: float32 (`--fp16` for float16), one unit-length row per word, in frequency order. The dimensions are rotated into variance order by `geometry.hpp` `mrl_order` (a rotation, so no cosine changes), so a prefix works as an MRL prefix.
- `<name>_index.tsv`: header `row id side n_tokens norm text count`. `text` is the word and `norm` its length before normalising. These are the columns of `gguf_mine.py embedtexts`, plus `count`.
- With `--texts FILE`: `<name>_sent_vectors.npy` and `<name>_sent_index.tsv`, one row per input line. SIF (Arora, Liang & Ma 2017): word weights a/(a + p(w)), a = 1e-3, p(w) from the corpus counts; the mean of the unit word vectors; then the first principal component of the sentence matrix removed; unit length. The sentences are the first 2,000 Dolly instructions. None of them lacks a known word.
- With `--kmeans`: spherical k-means, seeded by k-means++, as `word<TAB>class`.

**Files** (all in `/root/work/sem/`):

| file | what |
|---|---|
| `export/cypha_vectors.npy`, `export/cypha_index.tsv` | (b) W + C, 50,000 × 256, float32, unit rows |
| `export/cypha_sent_vectors.npy`, `export/cypha_sent_index.tsv` | 2,000 SIF sentence vectors from them |
| `export/ppmi_vectors.npy`, `export/ppmi_index.tsv` | C1's PPMI-SVD, 50,000 × 300, the same format |
| `export/ppmi_sent_vectors.npy`, `export/ppmi_sent_index.tsv` | the same 2,000 sentences from PPMI-SVD |
| `classes_k256.tsv`, `classes_k1024.tsv` | k-means word classes over the 50,000 (b) vectors |
| `out/sgns_v0p3_e5{,_var,_wc}.npy` with `_index.tsv` | (b)'s raw tables: word means, their variance, words + contexts |

**`gguf_mine.py compare`** (PPMI-SVD as `--a`, Cypha as `--b`, `--key text`, 5,000 sampled items for words, single-threaded BLAS):

| | pair Spearman (near / mid / random) | kNN@10 Jaccard | top-10 in top-50 | linear CKA | Procrustes r² | force KL at τ 0.05 (fitted τ_S, KL) |
|---|---|---|---|---|---|---|
| words (50,000 matched) | 0.714 (0.650 / 0.235 / 0.327) | 0.166 | 0.495 | 0.413 | −0.005 | 1.426 (0.0566, 1.395) |
| SIF sentences (2,000) | 0.931 (0.859 / 0.648 / 0.723) | 0.489 | 0.938 | 0.794 | 0.581 | 0.465 (0.0539, 0.453) |

**`gguf_mine.py geometry`**:

| | random-pair cosine, mean (IQR) | mean-vector norm | TwoNN | hubness skew (CSLS) | PCA dims for 90 / 95 / 99% | MRL agreement at 32 / 64 / 128 dims |
|---|---|---|---|---|---|---|
| Cypha words | 0.004 (−0.049, 0.053) | 0.057 | 38.1 | 2.45 (0.79) | 196 / 220 / 244 | 0.881 / 0.929 / 0.969 |
| PPMI-SVD words | 0.030 (−0.010, 0.063) | 0.175 | 28.7 | 1.18 (0.48) | 247 / 272 / 293 | 0.756 / 0.834 / 0.919 |
| Cypha SIF sentences | 0.008 (−0.078, 0.079) | 0.096 | 14.5 | 1.59 (0.53) | 149 / 187 / 234 | 0.934 / 0.968 / 0.991 |

- **The two word spaces agree in rank order (0.714), but not in neighbourhoods (Jaccard 0.166) or linearly (CKA 0.41, Procrustes r² ≈ 0).** They are two different factorisations of the same counts. Averaged into sentences, they nearly coincide (0.931).
- **Cypha's words are nearly isotropic** (floor 0.004 against PPMI-SVD's 0.030). They are more hub-prone (skew 2.45), and CSLS fixes most of it (0.79).
- **The MRL rotation matters.** Before it, Cypha's 32-dimension prefix kept 0.619 agreement with the full vector; after it, 0.881.

**k-means word classes** (for the class-context expert, TOOL_PROFILE P1):

| k | iterations | mean cosine to centre | class sizes min / median / max | time |
|---|---|---|---|---|
| 256 | 40 | 0.410 | 30 / 190 / 525 | 27 s |
| 1,024 | 17 | 0.457 | 4 / 42 / 171 | 55 s |

Examples at k = 1,024: "france" is with paris lyon bordeaux toulouse gaulle commune département …; "red" with black white blue color green dark brown yellow …; "january" with the other months; "computer" with designed machine machines computing portable calculator pdp colossus …. At k = 256, "france" shares a class with French function words (de, et, des, le, du), the same leak C1 saw in PPMI-SVD's neighbours of "paris".

## 6. Cost

| item | value |
|---|---|
| Tool lines | 1,812 → 2,638 (+826) for `train`, `distill`, `eval`, `export`, the `.npy` reader and writer, the PPMI table, the bootstrap and k-means. The relation, hypernymy and corpus code was moved out of `s1` into functions that `eval` shares, not copied. No new tool, and `geometry.hpp` is unchanged |
| Build | Clean under `-Wall -Wextra` (g++ 13.3), against `libcypha_core.a` built from `native/CMakeLists.txt` target `cypha_core` |
| PPMI-SVD (C1 tool) | 136 s for the whole run including the min_count 2 ablation; headline vectors at 87 s; peak RAM 607 MB |
| (a), (a2) | 216 s and 228 s per run (the 9.87M-step Cypha pass is 189–207 s); peak RAM 1,198 MB, of which the sort of 97M co-occurrence keys is most |
| (b), one epoch | 33.5M events, 305–338 s (about 9–10 µs per event, six logistic terms in 256 dimensions); peak RAM 303 MB |
| (b), 5 epochs | 1,542 s; peak RAM 316 MB |
| (b) model | Two tables of 50,000 × 256 means and variances in fp32: 205 MB in memory; the exported word means are 51 MB (fp32) or 26 MB (fp16) |
| (c), one epoch | 6.4M events, 702 s: about 110 µs per `pull_push_step`, 12 times (b)'s cost per event. Most of it is likely heap allocation inside the library step (a `std::vector` per touched item, per call); this was not profiled |
| `distill`, 10,000 items, 60 epochs | 352–409 s; peak RAM 879–933 MB (two 10,000 × 10,000 double cosine matrices for the teacher kNN and the stratified pairs) |
| `export` + k-means 256 and 1,024 | 92 s; peak RAM 207 MB |
| `eval` | 14 s for five vector sets with 2,000 resamples, relations and hypernymy |

## 7. Predictions scored

| | prediction | result | score |
|---|---|---|---|
| R0 | C1 and C5 reproduce exactly | Both, to the last printed digit | **Hit** |
| PA | (a) whitened cos(Δ) WordSim 0.40–0.60, SimLex 0.15–0.25; level (±0.03) with its count mean; below PPMI-SVD | 0.215 and 0.110; count mean 0.238 (Δ −0.023, level); below PPMI-SVD | **Miss** (the level held; the rise did not happen) |
| PA2 | (a2) count mean WordSim 0.35–0.55; Cypha more than 0.10 below it | Count mean 0.351 (whitened) and 0.377 (centred). Cypha 0.256: 0.095 below on WordSim, 0.044 on SimLex, 0.135 on MEN | **Miss** (the band held; the drop was not above 0.10 on every set) |
| PB | (b) WordSim 0.62–0.74, SimLex 0.22–0.32, within ±0.06 of PPMI-SVD | 0.750 and 0.270; Δ +0.027 and +0.005 | **Miss** (WordSim 0.010 above the band; the rest held) |
| PC | (c) WordSim 0.50–0.70, SimLex 0.18–0.30, not above (b) | From scratch it never passed dev (MEN 0.073), so the test sets were not read | **Miss** (far below the band) |
| PC2 | (c) from (b): W's MEN within ±0.02 of 0.730; WordSim and SimLex below (b) W + C | MEN 0.711 (Δ −0.019); WordSim 0.723 < 0.750; SimLex 0.218 < 0.270 | **Hit** |
| PG | The gate fails | (b): P = 0.941 on WordSim, 0.604 on SimLex | **Hit** |
| PD | Distilled pair Spearman ≥ 0.80; WordSim and SimLex within 0.05 of the teacher (from a random start) | From random: 0.770; WordSim 0.586 (−0.137), SimLex 0.196 (−0.068). From (b): 0.935, WordSim +0.018, SimLex −0.000 | **Miss** (the random start was registered) |
| PE | Is-a below the frequency baseline; variance rule 35–50% | 39.1% (KL and variance) against 60.6% | **Hit** |
| PF | Capital-of ridge @10 40–85%, offset @10 60–95% | 69.2% and 100% | **Miss** (the offset was above the band) |
| PX | Export: words pair Spearman 0.4–0.7, Jaccard 0.10–0.30; sentences 0.6–0.85; floor 0.02–0.20 | 0.714; 0.166; 0.931; 0.004 | **Miss** (only the Jaccard was in its band) |

## 8. Verdict

**Does learning the context side fix S-1?** Yes. With two tables and a logistic objective, Cypha's natural-gradient Gaussians go from WordSim 0.240 to 0.750 on the same pairs, and from SimLex 0.134 to 0.270. That is PPMI-SVD's level.

**What exactly does the work.**
- **The second table.** A fixed projection caps averaging at about 0.38 even when it projects the PPMI row exactly (a2). Learned contexts lift that cap.
- **The logistic objective.** Its optimum is the shifted PMI matrix, so it supplies the log-odds weighting that averaging cannot.
- **Cypha's step.** The diagonal Laplace update sets every item's step to its own posterior variance. The variance falls with the Fisher information the item has gathered. Rare items move a lot and frequent ones little, as EMBEDDING_GEOMETRY §4.2 asks. It needs no learning-rate schedule. After one epoch, Σ0 = 0.3 beat word2vec's step size (Σ0 = 0.025) by 0.139 on MEN (W + C: 0.736 against 0.597).

**What does not work.**
- **The §4.2 rule from scratch.** In 256 dimensions, from random unit vectors, nothing forms within an epoch at any step size tried. The evidence count n grows with every event, informative or not. The library step also costs 110 µs.
- **The variance as is-a.** A Fisher variance measures how much evidence an item has, not how broad its meaning is.

**Gate.** (b) is level with PPMI-SVD, not significantly above it. P = 0.941 on WordSim misses the 0.95 bar by 0.009, and SimLex is at 0.604. By the registered rule, PPMI-SVD stays the vector source.

## 9. Decisions for Odin and next steps

1. **The gate's reading.** The registered gate asks for "significantly above". (b) is level and Cypha-learned: online-capable, with a variance per item, and from the same counts. Its lower 2.5% bounds are −0.008 (WordSim) and −0.029 (SimLex), so it would pass a non-inferiority gate with a margin of 0.03. That margin would be chosen after the fact. If Odin wants a non-inferiority gate, he should set the margin first, and the test should be a fresh read: new pairs or a new seed. Until then, PPMI-SVD stays the source, and keeping both costs nothing.
2. **If (b) is accepted, the C1 tool becomes a baseline** (TOOL_PROFILE_PLAN), and `export` already writes (b)'s vectors in the form QE5 reads.
3. **`geometry.hpp` (not changed here; it belongs to the QE library).**
   - `pull_push_step` allocates per call. A caller-owned scratch buffer would cut most of its 110 µs.
   - K = Σ/(n + κ0) should count information, not events. A Fisher-weighted count (n += p(1 − p)|x|², as (b) does) would let the §4.2 rule start from random vectors. Equivalently, Σ should scale as 1/d, so that η does not have to.
4. **Is-a.** Learn breadth through a direction-ranking loss (C5 §8, point 5). Then retest the unregistered norm reading with a registered prediction.
5. **Distillation from Qwen.** `distill` takes a teacher `.npy` plus index as `gguf_mine.py embedtexts` writes it. Embed the 50,000-word list with the teacher, then run `distill --init` from (b) (S-5).

## 9. A non-inferiority read on fresh word sets (registered 2026-10-28, before the run)

The gate above asked for superiority, and (b) came out level. Odin's instruction is to decide from the evidence, so the question for adoption is changed to the one that matters for a vector source: is Cypha's own learner (b) **no worse** than PPMI-SVD? That needs a margin fixed in advance and a read on data never used here.

- **Data, never read before:** MTurk-771, RW (Stanford rare words), SimVerb-3500, RG-65, YP-130, MC-30, from the same `mfaruqui/eval-word-vectors` folder.
- **Pairs:** both words in the top 50,000 (the C1 protocol); paired bootstrap, 2,000 resamples, as `eval`.
- **Rule:** (b) `sgns_v0p3_e5_wc` replaces PPMI-SVD as the vector source if, on each of the three large sets (MTurk-771, RW, SimVerb-3500), the lower 95% bound of ρ(b) − ρ(PPMI-SVD) is above −0.03. The three small sets are reported only.
- **Prediction:** (b) passes on all three large sets, with point differences between −0.01 and +0.04.

## Reproduce

Build the core library and the tools (from the repo root):

```
cmake -S native -B /root/work/sem/cb_core -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /root/work/sem/cb_core --target cypha_core -j1
g++ -O3 -DNDEBUG -std=c++20 -msse4.1 -fopenmp -Wall -Wextra -DCYPHA_ACCEL_GPU_MIN_BATCH_ROWS=1 -DCYPHA_HP_GATE24=1 \
    -DCYPHA_HP_SLOT_MAX=24 -DCYPHA_LLM_ALGORITHM_HP=1 -DHP_SLOT_MAX=24 -DHP_XSIMD=1 \
    -Inative/third_party/hp/include -Inative/third_party/hp/third_party/xsimd/include -Inative/include -Inative/third_party \
    native/tools/cypha_word_classes.cpp /root/work/sem/cb_core/libcypha_core.a -lpthread -o cwc
g++ -O2 -std=c++20 native/tools/ppmi_svd.cpp -o ppmi_svd
```

Inputs, from `raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/`, with sha256:
- `EN-WS-353-ALL.txt` `ce12919b240af45bdf46d243601eaca2b6212b159b5e7e9b435ee7ee40c5619c`
- `EN-SIMLEX-999.txt` `405394ffb7d25f6d0e3c041f87c9f27d29dc6b047506769953e85c4523b11685`
- `EN-MEN-TR-3k.txt` `888b13580faa362c7b11da0a670258cf1b1eb4bf0ecbcd418918729a3bd3f43a`

WordNet: Open English WordNet `src/yaml/noun.*.yaml` (26 files) from `globalwordnet/english-wordnet` (main) into `wn/`.

```
E=/root/cdata/enwik8; P="ev/EN-WS-353-ALL.txt ev/EN-SIMLEX-999.txt ev/EN-MEN-TR-3k.txt"
./ppmi_svd $E $P out/enwik8 --no-ri                                    # C1 headline, out/enwik8.vectors.f32
./cwc export out/enwik8.vectors.f32 --out export --name ppmi \
      --texts texts/dolly_instructions_2000.txt                         # PPMI-SVD as .npy + index (+ SIF sentences);
                                                                        # eval and distill read the .f32 directly
S1="./cwc s1 $E $P out/enwik8.vectors.f32 out/enwik8.vocab.tsv wn --enc-lr 0 --sched inv --world-lr 0.00005 --no-bench"
$S1 --save out/c5_tuned                                                 # C5 tuned (R0)
$S1 --obs ppmi --save out/s1b_a                                         # (a)
$S1 --obs ppmi-row --save out/s1b_a2                                    # (a2)
./cwc train $E --method sgns --epochs 5 --var0 0.3 --out out/sgns       # (b): out/sgns.npy, _var.npy, _wc.npy
./cwc train $E --method pullpush --epochs 1 --tau 0.05 --eta 10 --kappa0 10 --out out/pp          # (c) from scratch
./cwc train $E --method pullpush --init out/sgns --epochs 1 --tau 0.05 --eta 100 --kappa0 1000 --out out/pp_init
./cwc eval $P wn out/sgns_wc.npy out/sgns.npy out/pp_init.npy out/s1b_a.npy out/s1b_a2.npy out/c5_tuned.npy out/enwik8.vectors.f32   # C5 pairs
./cwc eval $P wn out/sgns_wc.npy out/sgns.npy out/enwik8.vectors.f32 --top 50000                     # C1 pairs
./cwc eval $P wn out/X.npy out/enwik8.vectors.f32 --dev                                              # dev (MEN only)
./cwc distill out/enwik8.vectors.f32 --out out/distill_rand --items 10000 --epochs 60
./cwc distill out/enwik8.vectors.f32 --init out/sgns_wc.npy --out out/distill_init --items 10000 --epochs 60
./cwc export out/sgns_wc.npy --out export --name cypha --texts texts/dolly_instructions_2000.txt --kmeans 256,1024 --classes-dir .
python3 scripts/gguf_mine.py compare --a export/ppmi_vectors.npy --b export/cypha_vectors.npy \
        --index-a export/ppmi_index.tsv --index-b export/cypha_index.tsv --key text --out cmp_words
python3 scripts/gguf_mine.py geometry --vectors export/cypha_vectors.npy --index export/cypha_index.tsv --out geo_words
```

`texts/dolly_instructions_2000.txt` is the `instruction` field of the first 2,000 records of `/root/cdata/dolly.jsonl`, whitespace-collapsed, one per line. Every run is deterministic for its seed (seed 1). State hashes (`ItemStore::state_hash`): (b) at 5 epochs `d775ddc8993c2acb`, (c) from scratch `31fd777203bf096d`, (c) from (b) `39a5ceb4f1dbc7b4`. Run the Python checks with `OPENBLAS_NUM_THREADS=1`.
