# QE: the Cypha-side geometry library

Odin Loch, 2026-10-28

Library: `native/include/cypha/semantics/geometry.hpp` (header-only, namespace `cypha::semantics`).
Test: `native/tools/geometry_test.cpp`, CTest `native_geometry_test`. Synthetic data only; about 9 s on one core.
Design: [EMBEDDING_GEOMETRY.md](../EMBEDDING_GEOMETRY.md) §4.2, §4.5, §5.1, §7.1, §9.3; [SEMANTICS.md](../SEMANTICS.md) §2–3.

## What was built

- **Item store.** Each item has a mean μ, a diagonal variance Σ, an evidence count n and a world prior θ₀.
  - θ₀ is shared by default; an item can have its own when it is set.
  - Storage is fp32. μ, Σ and θ₀ export to and import from fp16; n stays fp32.
  - An FNV-1a hash covers the whole state.
- **§4.2 pull–push rule (`pull_push_step`).** The step has these parts:
  - a softmax at τ over {b} ∪ C;
  - the false-negative mask (the δ margin, or a known paraphrase admitted at ECS ≥ Moderate through `eval::grade_at_least`);
  - tangent cosine gradients;
  - the natural-gradient step K = Σ/(n + κ₀);
  - mirror updates for b and every candidate;
  - optional query–query and document–document uniformity terms;
  - MDL shrinkage toward θ₀;
  - an online variance update.
  - Helpers: a unigram^0.75 sampler.
- **§5.1 distillation (`distill_step`).** The force on each pair is (p_T − p_S)/τ_S, applied through the same step. The combined loss α·KL + (1 − α)·InfoNCE is supported.
- **MRL.**
  - `mrl_order` is a PCA rotation by cyclic Jacobi. It changes no cosine and carries Σ as diag(VᵀΣV).
  - The nested loss uses the prefixes {32, 64, 128, 256} capped at d (`mrl_prefixes`).
- **Post-processing.** Centring, PCA and ZCA whitening, and CSLS kNN.
- **§7.1 metrics.**
  - Spearman with average ranks for ties, and stratified pair sampling (near, middle, random).
  - kNN@k Jaccard, and recall of the teacher's top 10 within the student's top 50.
  - Linear CKA.
  - k-occurrence skewness (hubness).
  - Force-match KL.
  - Per-item disagreement and the confidence check.

**Reused rather than written again:** `eval/ecs.hpp` (`Grade`, `grade_at_least`), and the fp16 rounding routine of `src/cyphalm/component_dump.cpp`.

**Written here instead of reused:** a Jacobi solver, because the existing ones (`src/preprocessor_fit.cpp`, `src/intelligence/measurers.cpp`, `tools/ppmi_svd.cpp`) are internal to `.cpp` files. `thought::Welford` is a scalar in double precision, so the per-dimension variance is written here as the same recurrence in fp32. Nothing in `third_party` applied.

## Test numbers

**(b) Planted teacher.** 400 items in 64 dimensions, 20 clusters with an 8-dimensional local subspace each, 40 paraphrase pairs and 20 reflected hard negatives. The student has 32 dimensions and starts from random. Each run is 60 epochs; pair Spearman is over stratified pairs (5 near, 5 middle and 10 random per item).

| Run | Pair Spearman | kNN@10 Jaccard | Top-10 in top-50 | CKA | Force KL |
|---|---|---|---|---|---|
| Start (random) | −0.002 | 0.012 | 0.127 | 0.067 | 6.58 |
| InfoNCE only (§4.2) | 0.367 | 0.429 | 0.768 | 0.286 | 2.43 |
| Relational distillation | **0.943** | **0.889** | 1.000 | 0.916 | 0.013 |
| 0.7·KL + 0.3·InfoNCE | 0.942 | 0.866 | 1.000 | 0.911 | 0.060 |

The asserted thresholds were set from these runs:
- InfoNCE is more than 0.3 above the start.
- Distillation reaches Spearman ≥ 0.85 and Jaccard ≥ 0.5.
- Distillation beats InfoNCE on Spearman, Jaccard and force KL.
- The combined loss reaches Spearman ≥ 0.8.

With teacher seeds 7 and 31337 the results were 0.40, 0.95 and 0.94 for InfoNCE, distillation and the combined loss. InfoNCE learns only the near stratum (Spearman 0.74 within it, about 0 in the middle and random strata): hard labels carry no graded far-field information.

After distillation, paraphrase pairs have a mean cosine of 0.977 and the planted hard negatives 0.625.

**(c) Mask.** The student starts from a random projection of the teacher, so the paraphrase pairs start at cosine 0.954. After InfoNCE that samples each duplicate as a negative, the mean cosine is **0.750 with the mask and 0.640 without it**.

**(d) Confidence-scaled steps.** On the same gradient, an item with n = 1 turns 0.0152 rad and an item with n = 1000 turns 0.00017 rad. MDL at λ = 0.5 raises the cosine to θ₀ from 0.014 to 0.328 for an item seen once, and leaves an item with n = 1000 unchanged (0.2616 to 0.2620).

**(e) CSLS.** With 600 points and 12 planted hubs, the k-occurrence skewness at k = 10 falls from 3.18 under cosine to 0.32 under CSLS. The hubs' mean N₁₀ falls from 42.7 to 10.5.

**(f) MRL.**

| Prefix | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| Agreement before ordering | 0.867 | 0.954 | 0.983 | 1 |
| Agreement after ordering | 0.993 | 0.998 | 0.9997 | 1 |

Agreement is with the full 256-dimensional vector, on a decaying spectrum in a random basis. After ordering it rises strictly with prefix length.

A 64-dimensional student trained with the nested loss scores 0.918 at prefix 32, against 0.826 for a flat one. At the full 64 dimensions the two are equal (0.917 against 0.917).

**(g) Confidence check.** A quarter of the items get noisier training. The Spearman column is between variance and disagreement.

| Route | Variance, noisy items | Variance, clean items | Spearman |
|---|---|---|---|
| Distillation (teacher similarities + N(0, 0.3²)) | 0.0306 | 0.0242 | **0.283** |
| InfoNCE (half the positives random) | 0.0408 | 0.0284 | 0.142 |

With other seeds, distillation gave 0.45 and 0.43, and InfoNCE gave 0.10 and 0.03. The InfoNCE value is weak because that route leaves every item's far field wrong.

**(h) Determinism.** The same seed gives a bit-identical hash (`6fc6b79174ae5241`); a different seed gives a different one.

## Spec choices

1. **The force carries 1/τ, as in §3.2.** §4.2 omits it. The test uses η = 20, κ₀ = 10 and τ = τ_T = τ_S = 0.05; η = 60 diverges.
2. **Means stay on the unit sphere.** After the preconditioned step, the step is projected back onto the tangent, because a diagonal K is not isotropic, and the mean is then renormalised. As a result, MDL toward θ₀ = 0 does nothing. θ₀ should be a direction, for example `set_world_prior_to_mean`.
3. **Evidence and variance.** The anchor and the positive each gain one observation. The residual for the anchor is the direction of its partner (in distillation, the p_T-weighted spread of the candidates). The recurrence is Σ ← Σ + (r² − Σ)/(n + κ₀), which is Σ₀ at prior weight κ₀, the same form as S-1's rate. Negatives are moved but not counted.
4. **MDL rate.** λ_eff = λ/(n + 1).
5. **The paraphrase mask covers both sides.** It applies when c is a known paraphrase of b, as §4.2 says, and also when c is a paraphrase of a. The pair is admitted only at ECS ≥ Moderate.
6. **The nested loss is the mean over prefixes**, not the sum, so η does not depend on how many prefixes there are.
7. **Distillation candidate sets.** One mixed set at τ_T = 0.05 gave pair Spearman 0.51, because the far field gets no weight. Each anchor therefore gets two softmaxes: a local set (top 10, 10 from ranks 10–59 and 4 random) and a global set (24 random). This is the caller's sampling, not a change to the library.
8. **Confidence check.** Disagreement is 1 − Spearman of the item's teacher and student similarity rows, over every other item, or over 1000 seeded partners above that.
9. **fp16 limits.** Variances below 6.1 × 10⁻⁵ become fp16 subnormals and lose precision. A log-variance export is the fix if that matters.

## What is left

- Wiring into the S-1 semantics tool (`cypha_word_classes.cpp`, with `train`, `distill`, `eval` and `export`) and into Core Cypha's DIF classes. This includes the confusion source (i) of §4.2 from the DIF classifier, and Σ and n from the class store.
- Running on the real teacher vectors from `gguf_mine.py embedtexts` (QE5–QE8), on the hashed probe set of §7.1, split into dev and test.
- Not done: the calibration match by relation class (§7.1, which needs B2), and Slerp checkpoint merging.
