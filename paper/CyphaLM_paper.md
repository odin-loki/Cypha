# CyphaLM: Explicit Compression, Uncertainty, and Self-Organisation for Online Language Modelling

**Author:** Odin Loch  
**Status:** Draft — results populated by `experiments/` scripts

---

## Abstract

Conventional neural language models entangle lossy distribution compression, lossless residual encoding, statistical density estimation, and self-organised feature discovery in a single optimisation process. We present **CyphaLM**, which makes each mechanism explicit: Izaac GF(2^n) embeddings (zero embedding parameters, kNN accuracy 0.0125, collision rate 0.0), CellAI multi-scale SSM (O(1) per-token inference), CyphaDIF NIG expert fields (exact epistemic / aleatoric uncertainty decomposition, expert purity 1.00 on synthetic routing tasks), and GRIA alpha-projection. On a character-level language modelling task (D17, Gutenberg corpus, 40k training tokens), CyphaLM achieves 4.50 bits per character held-out versus a bigram baseline of 3.69 bpc. Online adaptation reduces OOD perplexity by 0.25 bpc on a new text domain. Toy vocabulary perplexity reaches 12.90 at convergence. The architecture is fully NumPy-based, supports online training, and requires no gradient computation.

**Key results (from `cypha_lm/REPORT.md`):**

| Experiment | Metric | Value |
|------------|--------|-------|
| Embedding (E01) | Izaac kNN acc | 0.0125 |
| Embedding (E01) | GF-rank Spearman | 0.1434 |
| Expert routing (E03) | Purity (2-topic) | 1.0000 |
| Toy LM (E05) | Final perplexity | 12.9032 |
| Code LM (E06) | Bits per char | 0.6146 |
| D17 held-out BPC | Moby Dick char-LM | 4.50 |
| D17 online adapt | BPC improvement | 0.25 |

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

**Result:** Izaac collision rate = `0.0000`; GF-rank Spearman = `0.1434`.

### 3.2 CellAI SSM temporal layer

See Figure 2 (`paper/figures/fig2_ssm_capacity.png`).

**Result:** CellAI O(n) runtime advantage at n > 100: `N/A`.

### 3.3 CyphaDIF expert field

See Figure 3 (`paper/figures/fig3_expert_self_organisation.png`).

**Result:** Expert purity = `1.0000` (target > 0.85).

### 3.4 GRIA alpha-projection

See Figure 4 (`paper/figures/fig4_alpha_spectrum.png`).

**Result:** Mean alpha at 20k steps = `N/A`.

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
| Izaac | `0.0125` | `0.0000` |
| Random learned | `0.0025` | `0.0000` |
| Freq-init learned | `0.0100` | `0.0000` |

### 5.2 SSM sequence capacity (Fig 2)

Lag-1 and lag-50 accuracy vs sequence length; runtime scaling.

### 5.3 Expert self-organisation (Fig 3)

Two-topic synthetic corpus; purity score `1.0000`.

### 5.4 Alpha spectrum emergence (Fig 4)

Alpha histograms at checkpoints; edge-of-chaos fraction `N/A`.

### 5.5 Language modelling (Figs 5–6)

**Toy vocabulary (Fig 5):** perplexity `12.9032`, syntax validity `N/A`.

**Code corpus (Fig 6):** bits/char `0.6146`, expert count `1`.

### 5.6 Uncertainty calibration (Fig 7)

ECE (NIG) = `N/A` vs dropout baseline `N/A`.

### 5.7 Online adaptation (Fig 8)

JSON perplexity after Python pretrain; improvement `N/A`.

### 5.8 Catastrophic forgetting (Fig 9)

Task A retention ratio `N/A` (target <= 1.10).

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
