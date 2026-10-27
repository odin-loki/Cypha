# Cypha semantics: Core Cypha as the statistical meaning layer

**Odin Loch** · 2026-10-27 · Track C of [PLAN.md](PLAN.md)

**In one line.** Meaning in a transformer comes from vectors plus statistical association. Core Cypha already learns statistics over vectors. If its classes are words and concepts and its observations are contexts, it can learn meaning itself, online, with uncertainty. CyphaLM then uses that to generalise, and still does the task (PLAN rule 10).

---

## 1. Why this is the missing piece

**What CyphaLM lacks.** It predicts from exact byte contexts. The 8K count test found the typical longest match is about 10 bytes, and an unseen phrasing just backs off ([report](reports/CYPHALM_QA_AND_CONTEXT_TEST.md)). That is why it scored 0/8 on facts it had read.

**What a transformer has.**
- **Similar words get similar vectors.** This is the distributional idea.
- **Contexts are vectors too**, so an unseen phrasing lands near seen ones and the prediction transfers.
- **Facts and relations are stored as associations.** Feed-forward layers act as key-value memories (Geva et al. 2021), and relations are close to linear maps on subject vectors (Hernandez et al. 2024).

**Vectors and association are one thing, not two.** Word2vec's vectors are an implicit factorisation of a word–context PMI matrix (Levy & Goldberg 2014): the dot product of two vectors approximates their association. A vector is compressed association statistics. The learning part is choosing what to associate with what.

---

## 2. Reinterpreting Core Cypha

Core Cypha keeps, per class, a diagonal Gaussian over a latent vector:
- a shared `WorldPrior θ₀` plus a per-class `ClassDifferential Δk`;
- natural-gradient updates in Fisher–Rao geometry;
- MDL shrinkage;
- a contrastive update;
- epistemic variance, NIG/GH gating and an OOD flag.

So far its classes were labels and its vectors came from outside. Change what a class and an observation are, and the same machinery becomes a meaning learner:

| Cypha part | As a meaning learner |
|---|---|
| A class `k` | A word, entity, sense or concept |
| The observation `h` | A vector for the context of one occurrence: neighbouring words, syntax, position |
| `Δk` (mean offset from the world prior) | The word's embedding, learned online from its contexts |
| The class variance | How broad or vague the word is ("animal" wide, "beagle" narrow) |
| MDL shrinkage on `‖Δk‖` | Rare words stay near the prior instead of getting noisy vectors |
| Epistemic variance, NIG/GH gate, OOD flag | "This word or context is unfamiliar": not knowing, at the vector level, which feeds the don't-know tag (DISCOVERY §3.5) |
| Fisher–Rao distance, KL divergence | Similarity of meaning. KL is **directional**: `KL(dog ‖ animal)` is small and `KL(animal ‖ dog)` is large, which is the is-a relation (Gaussian embeddings, Vilnis & McCallum 2015) |
| Contrastive update (pull to the true class, push from a competitor) | Learning by prediction: which word fits this context. With sampled competitors this is word2vec's negative sampling, done in Cypha's geometry |
| `DIFRegressor` / ridge / RLS posterior | A relation as a map, `object ≈ A_r · subject + b_r`, with predictive variance, so it can say "I don't know the capital of X" |
| `CausalField` (recurrent context state) | The running meaning of the sentence so far |
| Multi-modal LLR fusion | Several context views (left words, right words, syntax) combined, a statistical cousin of attention heads |
| Replay buffer, online learning | Meaning that keeps updating as text streams in, with no retraining |

---

## 3. The learning: statistical association

For each word occurrence at position i:

1. **Observation.** `h_i = Σ_{j ≠ i, |j − i| ≤ 5} (1/|j − i|) · r(w_j)`, where `r(w)` is a fixed sparse random index vector for word w (random indexing, d ≈ 256). Later versions add syntax (the tagger's tags) and position as further views.
2. **Update.**
   - Cypha's natural-gradient step pulls `w_i`'s Gaussian toward `h_i`.
   - The contrastive step pushes it away from k sampled competitor classes, drawn by unigram^0.75 as in word2vec.
   - MDL decay shrinks rarely seen classes toward the world prior.
3. **What accumulates.** Each word's mean is where its contexts sit; its variance is how widely they spread; its evidence count gives its epistemic uncertainty.

**Senses.** If a word's contexts form two clusters ("bank" by a river or money), its single Gaussian fits badly: high variance and frequent OOD-flagged contexts. That is the signal to split the class into sense classes. This is DISCOVERY G9 concept formation, promoted through LCI.

**Cost.** Scoring 10,000+ classes for every token is about 6 million operations per token. The sampled-competitor update scores only the true class and k ≈ 5 competitors, which makes the pass linear in corpus size.

---

## 4. Operations on meaning

| Operation | How |
|---|---|
| Similar words | Cosine of means, or Fisher–Rao / symmetric KL between Gaussians |
| Is-a, broader-narrower | KL asymmetry and variance ordering |
| Relations | A learned map per relation with predictive variance. Abstain when the variance is high |
| Phrase and sentence meaning | The `CausalField` state, or SIF-weighted sums of word means |
| New senses and concepts | Split classes with poor fit (above); cluster classes into concepts (G9) |
| Unfamiliar input | Epistemic variance and the OOD flag become the don't-know tag with reason `UNKNOWN_ENTITY` or `OUT_OF_SCOPE` |

---

## 5. Into CyphaLM (rule 10: CyphaLM does the task)

The semantic layer gives CyphaLM's mixture two new experts. CyphaLM still writes every answer byte.

1. **A class-context expert.** It predicts the next byte from the context's meaning (the class distribution of recent words) instead of exact bytes, so "Who authored Atlas Shrugged?" reuses what was learned from "Who wrote …". This is PLAN B4.
2. **A similar-context expert.** It retrieves stored contexts that are close in vector space, not identical, and mixes in what followed them. This is the count-model form of a kNN-LM (Khandelwal et al. 2020) and of attention's content-based lookup.

Both are judged by the tool-profile plan ([TOOL_PROFILE_PLAN.md](TOOL_PROFILE_PLAN.md)): Q/A coverage at a 2% wrong rate, and BPC.

---

## 6. Extracting it from a trained LLM

All of this enters as low-trust priors and hypotheses (ECS Beta(1, 3)). After learning, it is Cypha's own parameters.

| From the teacher | Into Cypha | Status |
|---|---|---|
| Token embeddings | Starting means for word classes, mapped into Cypha's space by an orthogonal Procrustes alignment on shared words | `gguf_mine.py embed` exports them |
| Contextual vectors | Better starting means; sense splits | `gguf_mine.py vectors --senses` |
| Feed-forward key-value memories (neuron → tokens it writes) | Candidate associations and concept groups, as hypotheses | `gguf_mine.py neurons` |
| Relation maps (affine fit from subject states to object states) | Priors for Cypha's relation regressors | Needs hidden-layer access (GGUF_MINING §9) |
| Next-token distributions on a corpus | Soft targets for CyphaLM's mixture and class expert: knowledge distillation (Hinton, Vinyals & Dean 2015). The student learns association strengths, not just top answers | Plan item |

Teacher-initialised and from-scratch versions are always compared, so the teacher's contribution is measured, not assumed.

**Copying a whole embedding space.** [EMBEDDING_GEOMETRY.md](EMBEDDING_GEOMETRY.md) (2026-10-27) plans how Cypha's vectors are made to attract and repel like Qwen3-Embedding-8B's: a full dissection of that model, the same pull–push forces in Cypha's Fisher geometry, relational distillation, and Qwen3.5-9B as the miner of pairs and hard negatives (work items QE0–QE9, predictions Q1–Q10).

---

## 7. Experiments and gates

| ID | Experiment | Gate |
|---|---|---|
| S-1 | **Words as classes**: 10,000 word classes on enwik8 contexts. Word similarity against random indexing and PPMI-SVD on the same pairs. **Done 2026-10-28, gate failed** ([report](reports/C5_WORDS_AS_CLASSES.md)): best WordSim / SimLex / MEN 0.240 / 0.134 / 0.372, level with a matched count baseline (0.256 / 0.101 / 0.358), far below PPMI-SVD (0.723 / 0.265 / 0.732) | Replaces PPMI-SVD as the vector source if ≥ PPMI-SVD on WordSim and SimLex (same pairs) |
| S-2 | Hypernymy from KL asymmetry and variance (BLESS, HyperLex or WordNet pairs) | Direction accuracy above the frequency baseline, with a bootstrap interval |
| S-3 | Relations as maps with abstention (capital-of and one more) | Leave-one-out accuracy above the mean-offset method; abstention lowers the wrong rate |
| S-4 | The class-context and similar-context experts inside CyphaLM | Tool-profile gates: Q/A coverage at 2% wrong, BPC |
| S-5 | Teacher-initialised against from-scratch | LCI ≥ 0.95 for keeping the teacher priors |
| S-6 | Sense splitting by fit and OOD | Split senses improve S-1 and S-4 on held-out data |

**Predictions for S-1 to S-3**, registered before the run (PLAN rule 3):
- **P1.** Cypha means reach random-indexing level (WordSim 0.35–0.55) and stay below PPMI-SVD on the same pairs, unless the contrastive update adds discrimination.
- **P2.** Hypernym direction from KL or variance is 55–70%, at least the frequency baseline.
- **P3.** The capital-of map gets leave-one-out @10 of 20–60%.

If S-1 matches PPMI-SVD, the C1 tool becomes a baseline, not a dependency, and the tool profile can cut it.

**Result (2026-10-28).** P1, P2 and P3 all missed; S-1 to S-3 fail their gates, and PPMI-SVD stays the vector source ([report](reports/C5_WORDS_AS_CLASSES.md)).
- Hypernym direction from KL or variance was 47–54%, against a frequency baseline of 60.6%.
- The capital-of map found 0–12% in the top 10, against 81% on PPMI-SVD vectors.
- The contrastive encoder update lowered similarity.

**Why, in hindsight** (Odin Loch's reading):
- The observations were fixed random context vectors, and Cypha's class update is a running mean of them. So a class mean is a linear projection of raw co-occurrence counts. It cannot express the log-odds (PMI) weighting that makes PPMI-SVD and word2vec work.
- Word2vec gets PMI because it learns a second table of context vectors under a logistic objective (Levy & Goldberg 2014).
- **What has to change:**
  - The contexts must be learned too (two tables), or PMI-weighted before they reach Cypha.
  - The pull–push must be the softmax-weighted, temperature-scaled rule of [EMBEDDING_GEOMETRY.md](EMBEDDING_GEOMETRY.md) §4.2, not a mean with a capped push.
  - The targets can come from a teacher's geometry (§5 there).

S-1's tool stays as the home for that work.

**S-1b (2026-10-28,** [report](reports/S1B_LEARNED_CONTEXTS.md)**).** Learned contexts with a logistic objective, in Cypha's natural-gradient Gaussians, reach PPMI-SVD's level on the same pairs (0.750 / 0.270 against 0.723 / 0.265). PMI-weighted inputs stay at count level; the §4.2 pull–push rule works only from a warm start. The gate (superiority) failed narrowly, and a non-inferiority read on fresh sets failed on rare words, so PPMI-SVD stays the vector source.

---

## 8. References

- Levy & Goldberg (2014). Neural word embedding as implicit matrix factorization. *NeurIPS*.
- Mikolov et al. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS*. (Negative sampling.)
- Sahlgren (2005). An introduction to random indexing.
- Vilnis & McCallum (2015). Word representations via Gaussian embedding. *ICLR*.
- Amari (1998). Natural gradient works efficiently in learning. *Neural Computation*.
- Geva, Schuster, Berant & Levy (2021). Transformer feed-forward layers are key-value memories. *EMNLP*.
- Geva, Bastings, Filippova & Globerson (2023). Dissecting recall of factual associations in auto-regressive language models. *EMNLP*.
- Hernandez et al. (2024). Linearity of relation decoding in transformer language models. *ICLR*.
- Khandelwal et al. (2020). Generalization through memorization: nearest neighbor language models. *ICLR*.
- Hinton, Vinyals & Dean (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
- Schönemann (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika*.
