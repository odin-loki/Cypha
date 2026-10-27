# S-1b: learned contexts for words as classes

**Odin Loch** · 2026-10-28

(Draft. Predictions registered before any run; results below them are filled in after.)

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
