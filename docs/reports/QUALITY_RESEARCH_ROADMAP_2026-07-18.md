> **Note (2026-07-18):** Draft research/audit note. Living status is [CONTINUUM_CLOSEOUT_2026-07-18.md](CONTINUUM_CLOSEOUT_2026-07-18.md) + [CYPHA_BILL_OF_WORK.md](../../CYPHA_BILL_OF_WORK.md). Release is **v2.3.24** / CI is **windows_msvc** (not MinGW). Treat numbers below as proposals unless cross-checked against those sources.
# Cypha — Benchmark & Model-Quality Research Roadmap

**Prepared for:** Odin Loch
**Scope:** Ideas for improving *AI performance* — BPC, accuracy, calibration, OOD — on the bench suite and external benchmarks. Architecture and algorithm changes included. Throughput is out of scope here (separate roadmap) except where quality work depends on it.

**Grounding — the current quality state as your own reports record it:**

| Surface | Number | Constraint I found in the code |
|---|---|---|
| D17 CyphaLM (WikiText-2 char, hybrid) | **BPC 2.873** pin; 25/25 cell hypotheses failed to beat it (best H19 +48 mBPC) | Single 128-hidden LSTM layer, **BPTT-1**, **vanilla SGD** (no momentum/Adam), no gradient clipping, no LayerNorm, no dropout/weight-decay, **300k of ~11M available train chars**, single online pass |
| S3 XOR (no kernel) | **~51%** (chance); kernel LLR path ~76% | `W_enc` is a *linear* projection trained only by contrastive Fisher–Rao residual — a linear encoder cannot represent XOR, full stop |
| Phase-3 per-class GMM | NO-GO (50.5% ON vs 51.2% OFF) | Online soft-EM never places components on the XOR lobes; your own report names component placement as the failure |
| D10 ECG5000 | 90.11% (from 85.96% via feature enrichment + 44 passes) | Hand-built stats+FFT features; SOTA on ECG5000 is ~94–95% |
| SOM/GNG/discriminative upgrades | All reverted (−3.5% to +1.0%, high variance) | Evaluated at 3 seeds on one synthetic task |

**Central thesis:** the D17 cell-hypothesis sweeps kept perturbing the *cell* while the *training recipe* was the binding constraint. BPTT-1 + raw SGD + 3% of the corpus is a regime where almost no architectural change can show its value — which is exactly why 25 hypotheses in a row lost to baseline. Fix the recipe first (§1), then re-run the architecture ideas (§2–3) on top; several will likely flip from no-go to win. Same pattern on the DIF side: the linear encoder is the wall (§4), and every opt-in that failed was evaluated behind that wall.

---

## 1. CyphaLM training recipe (largest expected BPC movement, mostly non-architectural)

Ordered by expected effect. These compound; together they plausibly move 2.873 → **sub-2.0**, and each one changes the verdict landscape for everything you've already built.

**1.1 — BPTT window: 1 → 32–128.** The header says it plainly: "Stateful backward (BPTT-1)". One-step truncation means the recurrent weights receive essentially no long-range credit assignment; the model is being trained as barely more than a gated bigram over embeddings. Truncated BPTT with T=64 is the standard char-LM setting and is, on its own, likely the largest single BPC lever in the entire codebase — plausibly several hundred mBPC. (Note the interplay: `bptt_ssm_update` exists for the SSM branch at 18.2% of step time; the LSTM branch deserves the same treatment.) Sweep T ∈ {8, 16, 32, 64, 128} vs BPC and wall-time; this is also the prerequisite for the parallelization work in the perf roadmap.

**1.2 — Optimizer: SGD → Adam/AdamW with schedule.** `apply_grads` is `w -= lr·g`, nothing else. Adam (β=0.9/0.999) with linear warmup → cosine decay is worth large, well-documented gains on char LMs, especially with the heterogeneous parameter scales you have (embeddings vs recurrent vs output). Add **global-norm gradient clipping** (1.0) — this also structurally addresses the class of NaN incidents your H15 fix report documents, rather than patching them per-hypothesis. Momentum-SGD is the cheap fallback if Adam's memory bothers you (it won't at this size).

**1.3 — Use the corpus.** 300k of ~11M WikiText-2 train chars is <3%. Train on all of it, multiple epochs with LR decay. Nothing about the online-learning story prevents an offline pre-train + online fine-tune split — that framing (*pre-trained prior θ₀, online Δ adaptation*) is actually more aligned with the Cypha world-prior architecture than training from scratch on a sliver. Expect this + 1.1 + 1.2 to dominate every architecture delta you've measured to date.

**1.4 — Normalization and regularization.** LayerNorm (or RMSNorm) on the pre-activation gates — LN-LSTM is a solved recipe and stabilizes exactly the gate-scale drift your custom eml/axiom activations are exposed to. Then, once data volume goes up (1.3): dropout on non-recurrent connections, variational/zoneout on recurrent state, weight decay (AdamW), label smoothing 0.05–0.1. At the current 300k-char scale regularization is premature; at 11M chars it becomes load-bearing.

**1.5 — Initialization.** Orthogonal init for `Wh`, forget-gate bias init to +1 (classic, free ~50–100 mBPC on LSTMs), scaled uniform for `Wy`. Tie this to your α-spectrum work: your D04 measurement already shows mean α ≈ 0.498 — the research-grade version is *criticality-targeted init* (choose gains so the network starts at α ≈ 0.5 rather than measuring that it drifted there). That's a paper section, not just a tweak (§6.1).

**1.6 — Mini-batch training.** Batch B=32–64 sequence streams per step (persistent hidden states per stream, the standard char-LM batching scheme). Better gradient estimates, and it's the enabler for the GPU path. `CyphaLMBatch` exists — the research task is making batched training the *default* D17 recipe rather than a side path, and re-pinning the baseline there.

**1.7 — Re-run the 25 cell hypotheses on the new recipe.** This is cheap once 1.1–1.6 land and scientifically important: the sweep's uniform failure is evidence about the *recipe*, not about the hypotheses. Expect a different ranking; H19 (+48 mBPC behind baseline in the starved regime) is exactly the kind of candidate that can flip.

---

## 2. CyphaLM capacity & architecture

Only meaningful after §1 — in the current regime these will measure as noise.

**2.1 — Scale and depth.** hidden 128 → 512–1024, 2–3 stacked layers with residual connections between them. The AWD-LSTM recipe (Merity et al.) is the reference point: a properly regularized 3-layer LSTM is still competitive on WikiText-scale data and would give you a *citable* comparison line. Sweep the quality/size frontier — it also produces the model-size ladder a product needs.

**2.2 — Tokenization.** `bpe_tokenizer.cpp` exists but D17 is char-level. Move the flagship benchmark to byte-level BPE (vocab 2–8k), report bits-per-byte alongside BPC so numbers stay comparable. Subword tokens multiply the effective context window (T=64 BPTT over BPE ≈ several hundred chars of context) and are worth a lot on WikiText specifically. Keep a char-level track for the enwik8/text8 external benchmark (§5.3).

**2.3 — xLSTM-style cell upgrades.** Your custom-gate program (eml/axiom/sr-laws) is philosophically adjacent to xLSTM (2024): **exponential gating with normalizer state (sLSTM)** and **matrix memory (mLSTM)**. These are the two published cell modifications with real evidence of closing the LSTM↔transformer gap, and they slot into your existing gate-dispatch framework as new modes. This is the most promising *cell* research direction available to you — far more so than another round of local perturbations — and mLSTM's matrix memory has a natural GRIA reading (graded write/erase as an α-controlled operation) that could make it *yours* rather than a port.

**2.4 — Modern SSM parameterization.** You have `selective_ssm.cpp`, `cellai_ssm`, `hierarchical_ssm`, `reversible_ssm_cell`. Audit them against the components that make Mamba-class SSMs work: input-dependent discretization Δ(x) (selectivity), HiPPO/S4D initialization of the state matrix, and (for training speed) associative-scan formulation. If your SelectiveSSM lacks input-dependent Δ or principled A-init, those are the two upgrades with published evidence; everything else in that family is decoration.

**2.5 — A single attention layer over your memory structures.** You already maintain `context_bank` and `compressive_memory`. The cheapest large BPC win in the hybrid literature is one sliding-window (or bank-keyed) attention layer on top of a recurrent stack. Concretely: h_t attends over the compressive-memory slots / context bank entries, output added residually before `Wy`. This keeps the online/recurrent identity of CyphaLM while removing the fixed-size-state bottleneck that caps every pure-RNN/SSM on long-range benchmarks (your D22 long-range suite is where it will show).

**2.6 — Retrieval-augmented decoding (kNN-LM) — the most Cypha-native idea on this list.** You already have a similarity index and kernel memory. kNN-LM (Khandelwal et al.): at each step, look up the k nearest stored hidden states, form a next-token distribution from their recorded continuations, and interpolate with the model's softmax: p = λ·p_kNN + (1−λ)·p_model. Published gains are large (≈15–20% perplexity reduction) *with no retraining*, it strengthens with corpus size, and it is a perfect story fit — "Tier-3 infinite context" made literal. Research extensions: context-dependent λ via your NIG confidence gate (interpolate more when the world prior says the input is far from θ₀), and GRIA-α as the trust signal.

**2.7 — Dynamic n-gram fusion.** `ngram_fusion` exists — make the interpolation weight context-dependent (pointer-sentinel style: a small learned gate on h_t decides n-gram vs neural mass) instead of fixed. Cheap, reliable mBPC, and composes with 2.6.

**2.8 — Distillation from a stronger teacher.** Train CyphaLM on soft targets (full next-token distributions) from a small transformer teacher trained on the same corpus. Soft targets are a strictly richer signal than one-hot chars and consistently lift small students. Given your NNGP/NTK distillation research thread, "distill a transformer into the θ₀⊕Δ hybrid" is both a benchmark tactic and a paper.

---

## 3. Mixture, gating and monitor machinery (LM level)

**3.1 — NIG-expert MoE with EM routing.** You built the EM keystone and fixed the DIF MoE with it; `cyphalm_nig_expert` suggests the LM analogue is partially there. A 2–4 expert LM with responsibility-based routing and a load-balancing term is worth testing *after* §1 — expert methods are meaningless in the starved regime.

**3.2 — Criticality-guided training control.** You measure α per expert (D04: fraction_edge_of_chaos 0.996). Close the loop: use α as a *control* signal — per-layer gain/LR modulation to hold α ≈ 0.5 during training, and cell-level intervention when the monitor detects drift toward order (α→1, vanishing) or chaos (α→0, exploding). This converts the intelligence monitor from telemetry into a training algorithm, is unique to your framework, and is testable with a simple ablation: α-controlled vs fixed schedule at matched budget.

**3.3 — IB encoder with β-annealing.** Phase 6 shipped fixed-β opt-in. The literature result is that annealed β (deterministic warmup: β 0→target) is what makes VIB-style objectives win; fixed β mostly loses. Re-evaluate under annealing before judging the idea.

---

## 4. CyphaDIF classification quality

**4.1 — Break the linear-encoder wall (the XOR fix, properly).** Three escalating options:

a. **Make kernel features the default.** The kernel LLR path already scores ~76% on XOR; RFF/Nyström/SORF shipped in Phase 5. Default-on with auto γ selection (your `auto_rff_gamma_cv` config exists) and D=512; re-baseline. This is the pragmatic fix and costs one flag flip plus golden regen.

b. **A 2-layer MLP encoder trained by backprop of the actual classification loss.** The contrastive Fisher–Rao residual currently trains a *linear* map; a linear map cannot solve XOR regardless of the training rule — the Phase-3 GMM was asked to compensate for a representational deficit upstream of it. Backprop the cross-entropy of the DIF's own class posterior through one hidden nonlinear layer into `W_enc` (the Fisher–Rao residual is exactly the gradient signal at the top; extend it one layer down). This preserves the online character and directly targets the stated ≥75% XOR gate.

c. **Deep encoder + DIF as the head.** Longer-term: any pretrained/self-supervised feature stack in front, DIF as the probabilistic classification head. This is also the honest scaling story for image/audio domains in the bench suite.

**4.2 — GMM component placement (your own promotion path, plus alternatives).** The replay buffer is the missing ingredient: **k-means++ / short batch-EM warm-start from replayed latents per class** before switching to online EM — components start on the lobes instead of hoping online updates find them. Alternatives worth one experiment each: hard-EM splits on the largest-residual component (your report's suggestion), split/merge EM, and rival-penalized competitive learning (RPCL, which automatically kills spurious components — nice fit for online). Gate: XOR ≥75% *with* 4.1b's nonlinear encoder OR standalone with kernel features off, per your promotion criteria.

**4.3 — Covariance structure: diagonal → low-rank + diagonal.** Per-class Δk currently lives on a diagonal-Gaussian manifold; correlated features (most real data) are invisible to it. Factor-analysis structure (Σ = D + VVᵀ, rank r=2–4) keeps the natural-gradient update tractable via Woodbury and materially improves Gaussian classifiers on correlated tabular/time-series data. This is squarely inside your information-geometry framing — the FA manifold has a known Fisher metric — so it's a paper-compatible upgrade, not a hack.

**4.4 — Generalize the D10 lesson into the framework.** D10 went 85.96 → 90.11 purely via per-series z-scoring, differencing, full-window FFT, feature standardization, and **more online passes**. Three generalizations: (i) feature standardization and multi-pass defaults across all tabular/TS domains, not just ECG; (ii) **MiniRocket features** as a built-in time-series encoder — random-convolution features are embarrassingly cheap, fit your random-features story exactly (they're the convolutional cousin of RFF), and MiniRocket + linear head is at/near SOTA on most of the UCR archive — this alone likely takes D10 past 94%; (iii) latent-space augmentation during online training (mixup between replayed latents of the same class, jitter/window-warp for TS inputs).

**4.5 — Calibration as a first-class metric.** DIF's pitch is *probabilistic* classification, but the bench reports accuracy only. Add ECE/Brier to every domain, and post-hoc temperature scaling fit on a held-out slice (one scalar; preserves argmax, fixes ECE). The BMA opt-in (Phase 4) should be re-judged on ECE/Brier/NLL, not accuracy — analytic BMA's expected benefit is calibration and OOD, and it may already be a default-on win under the right metric.

**4.6 — Seed ensembling.** Cheap accuracy + calibration: 3–5 seed models, average posteriors. Doubles as the variance estimate the SOM report showed you need.

---

## 5. Evaluation hygiene & benchmark strategy (what makes the numbers *sellable*)

**5.1 — Statistical discipline.** The SOM report's verdicts hinged on 3 seeds with ±0.01 noise around deltas of similar size. Standardize: ≥5 seeds, mean ± 95% CI, and a fixed rule (promote only if CI excludes zero). Split tuning from testing — a dev split for sweeps, test touched once per milestone. Your baseline-lock machinery is the right skeleton; extend it with CIs.

**5.2 — Stronger baselines in-suite.** Current comparisons are mostly vs logistic regression. Add XGBoost/LightGBM (tabular), MiniRocket + ridge (TS), a small trained transformer (LM). Beating LR is not a defensible claim to a defence customer or to Quinn's due-diligence reviewers; beating XGBoost on tabular OOD/continual settings *is* — and those settings are where DIF's design should genuinely win.

**5.3 — External, citable benchmarks.** Map bench domains onto public suites so results transfer outside the repo: full **UCR/UEA archive** (128 datasets) instead of ECG5000 alone; **OpenML-CC18** for tabular; **enwik8/text8** and **WikiText-2 (full, standard eval)** for the LM; report standard metrics in standard protocols. One overnight per suite once §1 lands.

**5.4 — Play to the architecture's actual strengths: OOD, drift, continual.** DIF has native anomaly scores, drift signals, EWC, and replay — none of which standard classifiers have — yet the headline numbers are IID accuracy, where DIF fights uphill. Build three benchmark tracks where the design should *dominate*: (i) OOD detection AUROC (in-distribution class task + held-out novel classes; compare vs softmax-confidence and Mahalanobis baselines); (ii) label-drift / covariate-shift streams (online accuracy over a drifting stream vs periodically-retrained XGBoost); (iii) continual learning (Split/Permuted protocols, report average accuracy *and* forgetting — connects to your Cypha forgetting-bounds addendum). These are also precisely the metrics defence evaluators care about.

**5.5 — One honest headline table.** For the M&A packaging: a single table — task class, Cypha, best classical baseline, best deep baseline, and the *setting* (IID / OOD / streaming / continual) — with CIs. The credibility of the whole pitch rests on the strong-baseline columns being real.

---

## 6. Research-grade bets (paper-adjacent, higher risk)

**6.1 — Criticality-targeted initialization and control (GRIA-native).** §1.5 + §3.2 as a unified study: initialize at α≈0.5 (connect to dynamical-isometry/mean-field init literature — your α is a cousin of their Jacobian spectral criteria), hold it there by feedback during training, and show a BPC/accuracy delta at matched compute. If it works it's the first *operational* payoff of the α ≈ 0.5 program and the strongest possible section for the Cypha paper series.

**6.2 — mLSTM matrix memory as a graded (α-controlled) operator.** Take xLSTM's matrix memory (2.3) and derive its write/erase dynamics inside GRIA — the write is an irreversible contraction whose grade you can set. Benchmark vs stock mLSTM; any measurable difference in either direction is publishable within your framework.

**6.3 — NIG-gated retrieval interpolation.** §2.6's λ driven by the GH/NIG confidence gate — trust retrieval exactly when the world prior flags the input as far from θ₀. Unifies the DIF OOD machinery with the LM, which no published kNN-LM variant does.

**6.4 — Natural-gradient VI on the low-rank manifold.** §4.3's FA-structured classes with proper natural-gradient updates (the Fisher metric for factor-analysis models is known). Extends the "Cramér–Rao efficient" claim beyond the diagonal case — closes an obvious reviewer objection to the current papers.

**6.5 — Teacher-distilled θ₀.** §2.8 framed as theory: the world prior as a distilled compression of a larger model, Δk as task adaptation. Connects the NNGP/NTK distillation thread, the compression papers, and the product in one arc.

---

## 7. Priority order

| # | Work | Cost | Expected effect |
|---|------|------|-----------------|
| 1 | §1.1–1.3 BPTT window + Adam/clip + full corpus | days | Largest BPC move available; likely 2.873 → low-2s alone |
| 2 | §1.4–1.6 LN, init, batching; re-pin baseline | ~1 wk | Stability + further BPC; kills the NaN class |
| 3 | §4.1a + §4.4 kernel-default + MiniRocket + standardize/passes defaults | days | XOR ≥75% closed pragmatically; D10 → ~94%; broad tabular/TS lift |
| 4 | §5.1–5.2 seeds/CIs + strong baselines | days | Makes every later verdict trustworthy and sellable |
| 5 | §1.7 re-run cell hypotheses; §2.2 BPE track | ~1 wk | Recovers value from work already done |
| 6 | §4.1b nonlinear encoder + §4.2 warm-started GMM | 1–2 wk | The *principled* XOR close; Phase-3 promotion path |
| 7 | §2.6 kNN-LM + §2.7 dynamic fusion | ~1 wk | Big BPC for near-zero training cost; flagship demo |
| 8 | §2.1/§2.3 scale + xLSTM cells; §2.5 attention-over-memory | wks | Competitive external LM numbers |
| 9 | §5.4 OOD/drift/continual tracks + §4.5 calibration | ~1 wk | The benchmarks where Cypha should *win outright* |
| 10 | §6 research bets | open | Papers + differentiation |

The through-line: items 1–2 unblock everything; items 3–4 are the fastest visible scoreboard changes; item 9 is where the architecture stops competing on other people's terms and starts competing on its own.

