# Cypha Event-Forecasting Framework — Design & Upgrade Plan

**Status:** Shipped v1 (2026-08-08) — Phases 1–9 implemented under `native/include/cypha/forecast/`; see [`docs/research/forecasting/README.md`](docs/research/forecasting/README.md).

| Phase | Status |
|-------|--------|
| 1 ORF/SORF encoder | Shipped; SORF in `everyday_profile.json`; `orf_encoder_bench` gate |
| 2 Token vocabulary | Shipped (`event_vocabulary.hpp`) |
| 3 Node estimators | Shipped (`node_estimator.hpp`) |
| 4 Sequence training | Shipped (GDELT tokens in `forecast_pipeline`) |
| 5 Rollout tree | Shipped (`rollout_tree.hpp`) |
| 6 Interpretability | Shipped (`interpretability.hpp`) |
| 7 Forgetting mitigations | Shipped (per-theater + EWC) |
| 8 VIEWS validation | Shipped (CRPS + leaderboard baselines) |
| 9 Live monitoring | Shipped (`GdeltCsvTail`, REST `/forecast/ingest`) |

**Author:** Odin Loch
**Scope:** (1) fix Cypha's documented nonlinear-boundary weakness, (2) repurpose Cypha as the node-level estimator inside a generative scenario-tree forecasting framework, (3) pick real datasets to train and validate it against.

---

## 0. Executive Summary

Cypha (native C++, GRIA algebra, NIG-Bayesian classifier/regressor/sequence model) was evaluated as a base for a complex-event / geopolitical-escalation forecasting framework. Verdict: **good fit for the estimator, wrong fit as a static graph.**

- Its MDL cold-start protection + NIG calibrated uncertainty are genuinely built for scarce-data regimes — which matches the true shape of great-power-war data (thousands of minor-dispute examples, a handful of major-war examples).
- Its documented weakness is **nonlinear decision boundaries** (XOR-class problems), not multi-class differentiation — fixable with a concrete, literature-backed encoder swap (Section 2).
- Rather than hand-building an external Bayesian network, the scenario tree should be **generated** by `CyphaLM`'s own ancestral-sampling machinery, guided toward rare/tail branches using its existing `active_query_score()` primitive (Section 3).
- Training and validation data should be layered: dense historical dispute data for node-level estimators, a live academic benchmark (VIEWS) for validating calibrated output against real published models (Section 4).

This plan does **not** claim to solve everything. Two problems stay open regardless of the changes below: shared-model catastrophic forgetting (D16B) if the system trains continuously online, and the irreducible scarcity of true great-power-war examples. Both are called out explicitly rather than papered over.

---

## 1. Current State Assessment (from the Cypha repo, as of this clone)

| Component | What it actually is | Verdict for this use case |
|---|---|---|
| `CyphaDIF` classifier | Online Bayesian generative classifier: shared `WorldPrior` + per-class `Δk` differential, LLR-argmax decision, natural-gradient updates | Good — this is the node-level escalation-probability estimator |
| NIG posteriors | Normal-Inverse-Gamma per-field uncertainty tracking | Good — gives calibrated epistemic uncertainty per estimate, not just a point probability |
| MDL cold-start protection | Regularizes hard toward simplicity for a class's first ~8 observations | Good — matches scarce-positive-example reality of major-war data |
| `RFFEncoder` | Random Fourier Features, i.i.d. Gaussian sampling, auto-applied for dim ≤ 30 | Weak point — see Section 2 |
| `CyphaLM` sequence spine | Hybrid GRIA+LSTM, `predict_next`, predictive arithmetic coding, 8 generation modes (ancestral, adversarial, OOD, MDL-ball, latent-boundary interpolation, etc.) | Strong — this is the generative engine for the scenario tree, see Section 3 |
| `active_query_score()` | Entropy × boundary-proximity, built for active learning | Repurposed here as the tree-expansion exploration signal |
| `drift_score()` / `anomaly_score()` | Online distribution-shift and outlier detection | Repurposed here as "we've left the training distribution" live alarm |
| Shared-model continual learning (D16B) | Documented, acknowledged, open problem — forgetting when training sequentially across many tasks/classes; EWC helps only modestly | Real risk if this system trains continuously on live data — mitigation plan in Section 5, not a solved problem |
| Multi-class capacity | Digits (10-class) 0.922, Go 99.5%, Poker 93.3% — all close to ceiling | Not actually a documented weakness — corrects an earlier assumption in this project's discussion |

---

## 2. Upgrade 1 — Fixing the Nonlinear-Boundary Ceiling

**Problem:** `RFFEncoder` samples its random projection matrix i.i.d. Gaussian. On XOR-class problems (S3 benchmark), Cypha scores 0.482 (near chance) vs. sklearn RBF-SVM's 0.825 — a ~2.7pp gap remains even with the current kernel-LLR mitigation.

**Fix (validated on Cypha 2026-08):** replace the i.i.d. Gaussian projection with an **Orthogonal Random Features (ORF)** matrix — same interface (raw features → D-dim vector → existing LLR/NIG pipeline), different sampling procedure.

- ORF: orthogonalize the Gaussian matrix (e.g. via QR decomposition), rescale rows by their chi-distributed norms to preserve the marginal distribution. Proven to reduce kernel approximation error vs. plain RFF (Yu et al., NeurIPS 2016; Choromanski et al. follow-ups).
- SORF (Structured ORF): uses Hadamard-structured matrices instead of dense orthogonal ones. Matches ORF's accuracy at O(d log d) instead of O(d²) — relevant given Cypha's native/HPC C++ focus and existing SGEMV-based kernels.
- Empirical ordering from the literature: **SORF ≈ ORF < QMC-RFF < plain RFF** on kernel approximation error (lower is better).

**Implementation sketch:**
1. Add `OrthogonalRFFEncoder` alongside `RFFEncoder` (same interface, drop-in).
2. Generate the projection matrix via block-wise QR-orthogonalized Gaussian blocks (standard ORF construction) for the dense version; Hadamard-Rademacher-Gaussian chain for SORF.
3. Re-run the existing pinned benchmarks (`S3` XOR, `D01` linear/blobs) unchanged — this is exactly what the bench harness (`bench/BASELINE_REPORT.md`) is for. Compare new XOR accuracy against the current 0.482 pin.
4. Only promote to default if it beats the pin without regressing D01/R1-R4. Keep old `RFFEncoder` available for regression comparison.

**Honest limit:** this addresses the nonlinear-boundary weakness specifically. It does not touch forgetting (D16B) or data scarcity. Treat it as one fix among several, not a general-purpose upgrade.

---

## 3. The Forecasting Framework — Generated Tree, Not Hand-Built Graph

**Why not a static Bayesian network / DAG:** a hand-authored graph (nodes = events, edges = conditional dependencies you write down yourself) is transparent but frozen — you have to anticipate every dependency in advance, and it can't update its own structure as new data arrives. It also tends toward false independence between branches unless you painstakingly wire every cross-term by hand (this is what the earlier flat Monte Carlo model in this project did, and its main weakness).

**Proposed alternative: guided ancestral rollout tree, generated by `CyphaLM` itself.**

### 3.1 Event tokenization
- Define a finite vocabulary of discrete world-state tokens per theater (e.g. `TWN_BLOCKADE_DECLARED`, `RUS_NATO_ARTICLE5_INVOKED`, `IRN_CEASEFIRE_HOLDS`, `PRK_ARTILLERY_EXCHANGE`). Factor tokens by theater + escalation level rather than one giant flat vocabulary, so the sequence model can learn theater-specific transition dynamics while still sharing structure across theaters via the shared `WorldPrior`.
- This mirrors how GDELT/CAMEO event codes already work (Section 4) — the tokenization scheme should map directly onto an existing coded-event taxonomy rather than inventing a new one from scratch, so training data doesn't need hand-relabeling.

### 3.2 Tree generation
- Seed `CyphaLM` with the current real-world context (recent tokens = actual recent events).
- Use **ancestral sampling** (`k ~ context, h ~ p(h|k)`) repeatedly: each rollout is one path through the tree, sampled from the model's own learned `predict_next` distribution — not a hand-set transition probability.
- Run many rollouts to build the tree; edge weight = empirical frequency of that transition across rollouts, annotated with the model's own NIG-based confidence at that step.

### 3.3 Guided exploration (the actual upgrade over naive Monte Carlo)
Naive rollout sampling wastes almost all compute on the boring, non-escalatory branches — the tail events (cascade to great-power war) are exactly the ones you care about and exactly the ones a flat Monte Carlo process will under-sample.

- Repurpose `active_query_score()` (entropy × boundary-proximity, already built for active learning) as an **MCTS-style expansion bonus**: at each node, bias further rollouts toward children with high active-query score — i.e. toward branches the model is genuinely uncertain about, not branches it's already confident are calm.
- Use the regression head / NIG variance as the **leaf value estimate** (standard MCTS backup), and `drift_score()` as a flag on any node whose real-world analog has left the training distribution — this is your "the model is now guessing outside its experience" alarm, and for great-power-war-scale outcomes it will likely fire often. That's a feature, not a bug: it tells you exactly where the forecast is least trustworthy.

### 3.4 What stays as a hand-authored layer
Keep a **thin, explicit graph on top** for interpretability: after generating the rollout tree, extract the highest-probability and highest-uncertainty paths and let a human (you) annotate *why* — same role a hand-drawn graph would play, but built from what the model actually learned rather than imposed on it beforehand. This is the auditability trade-off called out in the earlier discussion, addressed rather than ignored.

---

## 4. Data Strategy

### 4.1 Node-level / dispute-escalation training data
| Dataset | What it gives you | Fit |
|---|---|---|
| **Correlates of War MID v5 / Gibler-Miller-Little (GML) MID v2** | Dyad-year and incident-level militarized disputes, 1816–2014, hostility level 1 (no dispute) through 5 (war), ~2,000+ dispute-years | Primary source for "does this dispute escalate" node estimators — dense enough at low escalation levels, sparse (honestly) at war level, which is the correct reflection of reality, not a data-quality problem to paper over |
| **UCDP/PRIO Armed Conflict Dataset** | State-based armed conflict onset, duration, and intensity, 1946–present | Complements MID with a longer-running, more consistently updated modern series |
| **International Crisis Behavior (ICB) dataset** | Crisis-level data explicitly coding major/great-power involvement and perceptions | The most direct source for training the specific "does this cascade to involve a great power" transition — the step the whole framework cares about most |
| **GDELT (CAMEO-coded event stream)** | Near-real-time, high-frequency coded events (thousands/day), from which the tokenization scheme in Section 3.1 should be built | Source of the discrete token vocabulary for `CyphaLM` sequence training, and the live feed for `drift_score()` monitoring once deployed |

### 4.2 Validation / benchmark dataset — recommended: **VIEWS (Violence & Impacts Early-Warning System)**

This is the single best fit for testing the finished framework, for a specific reason: VIEWS's own 2023/24 Prediction Challenge already reframed conflict forecasting as **predicting a full probability distribution over battle-related fatalities, with uncertainty** — not a point estimate. That's exactly the output shape a NIG-calibrated Cypha estimator naturally produces, so no awkward reformatting is needed to compare against it.

- Public, live, country-month and subnational (PRIO-grid-month) forecasting target.
- Existing published leaderboard of competing models (Conflictology benchmark, Observed Markov Models, Bayesian Negative Binomial GLMM, "Forests of UncertainT(r)ees") to score the Cypha-based framework against directly — a real external check, not a self-graded backtest.
- Actively maintained: 2026 VIEWS outputs already flag Ukraine, Palestine/Israel, Sudan, Pakistan, and Nigeria as the highest-projected battle-fatality countries for the year, so there's live signal to validate against as the year progresses, not just historical backtesting.

**Recommended validation protocol:** train node-level estimators and the tokenized sequence model on MID/UCDP/ICB/GDELT (Section 4.1), then generate the framework's own country-month fatality-distribution forecasts and score them against VIEWS's held-out true-future window using the same scoring rules VIEWS uses (CRPS / ignorance score) — apples-to-apples against a real published benchmark, rather than an invented success metric.

---

## 5. Mitigating Shared-Model Forgetting (D16B)

Not solved by anything above — called out on its own because it's the thing most likely to quietly break a system meant to keep learning from live data.

- **Per-theater isolated models** (`D16F` shows zero forgetting by architecture when models are isolated) as the default, rather than one shared model across all theaters — costs some cross-theater transfer, buys guaranteed no-forgetting.
- Where cross-theater sharing is worth the risk (e.g. the shared `WorldPrior`), keep **EWC on**, even though it's only "modest help" per the existing D16B scoping doc — modest is still better than nothing for a system meant to run for years.
- Maintain a **priority replay buffer** of rare/high-value historical events (the actual great-power-war-adjacent cases) so retraining passes don't let the rarest, most important examples get diluted out by the much larger volume of routine dispute data.

---

## 6. Phased Implementation Plan

| Phase | Deliverable | Depends on |
|---|---|---|
| **1. Encoder upgrade** | `OrthogonalRFFEncoder` implemented, benchmarked against existing S3/D01 pins, promoted only if it beats them | Section 2 |
| **2. Tokenization scheme** | Finite per-theater event vocabulary defined, mapped onto CAMEO/GDELT codes | Section 3.1, 4.1 |
| **3. Node-estimator training** | `CyphaDIF` classifiers/regressors trained on MID/UCDP/ICB dispute→escalation transitions | Section 4.1 |
| **4. Sequence training** | `CyphaLM` trained on tokenized GDELT event sequences | Section 3.1, 4.1 |
| **5. Rollout tree engine** | Ancestral sampling + `active_query_score`-guided expansion + NIG-based leaf values implemented as the tree generator | Section 3.2, 3.3 |
| **6. Interpretability layer** | Extraction of top-probability / top-uncertainty paths into a human-annotated summary graph | Section 3.4 |
| **7. Forgetting mitigations** | Per-theater isolation + EWC + priority replay wired in before any continuous live-data training starts | Section 5 |
| **8. Validation run** | Framework's country-month fatality distributions scored against VIEWS's true-future window using CRPS/ignorance score | Section 4.2 |
| **9. Live monitoring** | `drift_score()`/`anomaly_score()` wired to a live GDELT feed as an ongoing "outside training distribution" alarm | Section 3.3, 4.1 |

Phases 1–2 can run in parallel. Phase 8 is the real test of everything before it — no phase should be treated as "done" until it clears validation, not just unit-level benchmarks.

---

## 7. Success Metrics

- **Calibration, not just accuracy**: CRPS or ignorance score against VIEWS's held-out true-future window (matches how VIEWS itself scores contestants).
- **Ranking against the VIEWS leaderboard baselines** (Conflictology, Observed Markov Models, Bayesian Negative Binomial GLMM) — being competitive with, not necessarily beating, established published models is a reasonable first bar.
- **Encoder regression check**: `OrthogonalRFFEncoder` must not regress any existing pinned benchmark (D01, R1–R4) while improving S3 (XOR).
- **Drift-alarm sanity check**: `drift_score()` should fire noticeably around known real out-of-distribution shocks (e.g. the Feb 2026 US/Israel–Iran war onset) when backtested — if it doesn't, the alarm isn't trustworthy for live use.

---

## 8. Open Problems — Not Fixed By This Plan

Stated plainly so nothing here gets oversold later:

1. **Great-power-war examples are still scarce.** No encoder change or tokenization scheme manufactures data that doesn't exist. The framework's tail-event forecasts should always be read with wider uncertainty than its well-populated (minor-dispute) forecasts.
2. **Shared-model forgetting is mitigated, not solved.** EWC is acknowledged as only modest help in Cypha's own docs.
3. **Learned tree structure is less auditable than a hand-drawn graph**, even with the Section 3.4 interpretability layer — a generated edge is a statistical association, not a documented causal argument.
4. **The Orthogonal Random Features upgrade is a literature-backed hypothesis, not a validated Cypha result**, until Phase 1's benchmark run confirms it.

---

## Appendix: Key Sources

- Yu, Suresh, et al. — *Orthogonal Random Features* (NeurIPS 2016) — ORF/SORF construction and accuracy gains over plain RFF.
- Hegre, Vesco, et al. — *The 2023/24 VIEWS Prediction Challenge: Predicting the Number of Fatalities in Armed Conflict, with Uncertainty* — validation benchmark and scoring methodology.
- Correlates of War Project — MID v5 / GML MID v2 — dispute-escalation training data.
- Cypha repo: `docs/RESEARCH_STATUS.md`, `docs/archive/reports/one_cypha/PGM_CELL_INTEGRATION_2026-07-18.md`, `bench/BASELINE_REPORT.md` — current architecture and benchmark pins referenced throughout this plan.
