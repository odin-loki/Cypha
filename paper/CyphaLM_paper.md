# CyphaLM: Explicit Compression, Uncertainty, and Self-Organisation for Online Language Modelling

**Author:** Odin Loch  
**Status:** Draft — results populated by `experiments/` scripts

---

## Abstract

<!-- TARGET: ~250 words. Auto-fill after experiments complete. -->

Conventional neural language models entangle lossy distribution compression, lossless residual encoding, statistical density estimation, and self-organised feature discovery in a single optimisation process. We present **CyphaLM**, which makes each mechanism explicit: Izaac GF(2^n) embeddings (zero embedding parameters), CellAI multi-scale SSM (O(n) sequence processing), CyphaDIF NIG expert fields (exact epistemic/aleatoric uncertainty), and GRIA alpha-projection (edge-of-chaos alpha ~ 0.5).

**Placeholder metrics:**
- Embedding kNN accuracy (Izaac vs learned): `{{EXP01_KNN}}`
- SSM long-range accuracy vs transformer: `{{EXP02_LONG_ACC}}`
- Expert self-organisation purity: `{{EXP03_PURITY}}`
- Edge-of-chaos fraction at convergence: `{{EXP04_EDGE_FRAC}}`
- Toy LM perplexity: `{{EXP05_PPL}}`
- Code corpus bits/char: `{{EXP06_BPC}}`
- Uncertainty calibration ECE: `{{EXP07_ECE}}`
- Online adaptation improvement: `{{EXP08_PPL_DELTA}}`
- Catastrophic forgetting retention: `{{EXP09_RETENTION}}`
- Parameter efficiency vs transformer: `{{EXP10_BPC_RATIO}}`

---

## 1. Introduction

### 1.1 Four mechanisms in neural networks

Lossy compression (GRIA alpha -> 1), lossless residual channel (epistemic uncertainty), statistical P(Y|X) approximation, and self-organisation at the edge of chaos (alpha ~ 0.5).

### 1.2 Limitations of entangled mechanisms

Opacity, parameter inefficiency, no calibrated uncertainty, catastrophic forgetting.

### 1.3 CyphaLM thesis

Explicit mechanisms are more efficient and interpretable.

---

## 2. Background

- 2.1 Compression-based learning (MDL, GRIA-NMP)
- 2.2 Normal-Inverse-Gamma posteriors
- 2.3 State space models (O(n) advantage)
- 2.4 Grand Unified Law: alpha = 1 - H(f(X)) / H(X)

---

## 3. CyphaLM Architecture

### 3.1 Izaac GF(2^n) embeddings

See Figure 1 (`paper/figures/fig1_embedding_benchmark.png`).

**Result placeholder:** Izaac collision rate = `{{EXP01_COLLISION}}`; GF-rank Spearman = `{{EXP01_SPEARMAN}}`.

### 3.2 CellAI SSM temporal layer

See Figure 2 (`paper/figures/fig2_ssm_capacity.png`).

**Result placeholder:** CellAI O(n) runtime advantage at n > 100: `{{EXP02_RUNTIME_RATIO}}`.

### 3.3 CyphaDIF expert field

See Figure 3 (`paper/figures/fig3_expert_self_organisation.png`).

**Result placeholder:** Expert purity = `{{EXP03_PURITY}}` (target > 0.85).

### 3.4 GRIA alpha-projection

See Figure 4 (`paper/figures/fig4_alpha_spectrum.png`).

**Result placeholder:** Mean alpha at 20k steps = `{{EXP04_MEAN_ALPHA}}`.

### 3.5 End-to-end pipeline

Token -> Izaac -> CellAI -> CyphaDIF -> GRIA -> P(next token).

---

## 4. Theoretical Properties

- 4.1 Epistemic variance bounds lossless residual mass
- 4.2 O(n) inference complexity
- 4.3 Bounded expert creation
- 4.4 alpha ~ 0.5 maximises bits-per-parameter

---

## 5. Experiments

### 5.1 Embedding benchmark (Fig 1)

| Method | kNN accuracy | Collision rate |
|--------|-------------|----------------|
| Izaac | `{{EXP01_KNN_IZAAC}}` | `{{EXP01_COLLISION_IZAAC}}` |
| Random learned | `{{EXP01_KNN_RANDOM}}` | `{{EXP01_COLLISION_RANDOM}}` |
| Freq-init learned | `{{EXP01_KNN_FREQ}}` | `{{EXP01_COLLISION_FREQ}}` |

### 5.2 SSM sequence capacity (Fig 2)

Lag-1 and lag-50 accuracy vs sequence length; runtime scaling.

### 5.3 Expert self-organisation (Fig 3)

Two-topic synthetic corpus; purity score `{{EXP03_PURITY}}`.

### 5.4 Alpha spectrum emergence (Fig 4)

Alpha histograms at checkpoints; edge-of-chaos fraction `{{EXP04_EDGE_FRAC}}`.

### 5.5 Language modelling (Figs 5–6)

**Toy vocabulary (Fig 5):** perplexity `{{EXP05_PPL}}`, syntax validity `{{EXP05_SYNTAX}}`.

**Code corpus (Fig 6):** bits/char `{{EXP06_BPC}}`, expert count `{{EXP06_EXPERTS}}`.

### 5.6 Uncertainty calibration (Fig 7)

ECE (NIG) = `{{EXP07_ECE}}` vs dropout baseline `{{EXP07_ECE_DROPOUT}}`.

### 5.7 Online adaptation (Fig 8)

JSON perplexity after Python pretrain; improvement `{{EXP08_PPL_DELTA}}`.

### 5.8 Catastrophic forgetting (Fig 9)

Task A retention ratio `{{EXP09_RETENTION}}` (target <= 1.10).

### 5.9 Parameter efficiency (Fig 10)

Bits-per-token vs parameter count; CyphaLM vs transformer at matched params.

---

## 6. Discussion

- 6.1 Epistemic residual as anti-hallucination signal
- 6.2 Comparison to SSM literature (Mamba, S4)
- 6.3 Comparison to MoE literature
- 6.4 Limitations

---

## 7. Conclusion

CyphaLM makes the lossless residual a first-class measurable quantity.

---

## References

<!-- Placeholder bibliography -->
- Gu & Dao (2023) — Mamba
- Amini et al. (2020) — Deep Evidential Regression
- Grunwald (2007) — MDL
- Shazeer et al. (2017) — Mixture of Experts
- Langton (1990); Poole et al. (2016) — Edge of chaos
- Tishby et al. (2000) — Information Bottleneck

---

*Generated skeleton. Run `python scripts/run_cypha_lm_report.py` to populate metrics in `cypha_lm/REPORT.md`.*
