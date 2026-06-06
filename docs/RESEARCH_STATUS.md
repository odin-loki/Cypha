# CyphaDIF — Research Status

**Last updated:** 2026-05-31 (CyphaLM upgrade + multi-view plan docs) | **Report:** `cypha_bench/BASELINE_REPORT.md` (2026-05-31; re-run after upgrade)

This is the canonical research journal for CyphaDIF and the Cypha stack. It records what we have tried, what the numbers show, what is confirmed, what is broken, and where we are going next. Intended audience: future developers and researchers picking up this project.

---

## Quick state summary

| System | Status | Verdict |
|--------|--------|---------|
| **CyphaDIF classifier** | Working, benchmarked | Competitive on linear/tabular; hard limit on nonlinear boundaries |
| **CyphaDIF regressor (DIFRegressor)** | Working | Comparable to Ridge on smooth domains; poor on nonlinear equations |
| **C++ / CUDA / Qt port** | M1–M6 complete | Parity with Python on all ported ops; deliberation and kernel LLR Python-only |
| **cypha_accel (GPU fused kernels)** | Working | CuPy GPU path used automatically; NumPy fallback |
| **cypha_lm (CyphaLM)** | Beats trigram at 40k | D17 **4.154** / D04 **4.122** BPC; +0.19–0.24 vs bigram; char-LSTM still best; **multi-view training planned** → [`MULTI_VIEW_TRAINING_PLAN.md`](MULTI_VIEW_TRAINING_PLAN.md) |
| **cypha_som (SOM upgrades)** | Benchmarked, reverted | All upgrades worse than baseline; U3/U5/U6 structurally safe |
| **cypha_bench (eval harness)** | 17 domains complete | D04 full LLM suite; D17 extended integration; `adapters/cyphalm_bench.py` |
| **cypha_studio (PySide6 + FastAPI)** | Working | GUI + REST + registry; **CyphaLM `/generate` SSE** (FastAPI-only) |

---

## Benchmark results — all 17 domains

Run on 2026-05-31 using `cypha_bench/config/everyday_profile.json` (deliberation off, `delta_lr=0.03`). Full raw report: [`cypha_bench/BASELINE_REPORT.md`](../cypha_bench/BASELINE_REPORT.md).

### Classification domains

| Domain | Task | Cypha acc. | Baseline (best) | Gap | Verdict |
|--------|------|-----------|----------------|-----|---------|
| D01 — Stat baselines | 4-Gaussian blobs | **99.75%** | SGD 100% | −0.25pp | ✅ Strong |
| D01 — Stat baselines | Linear 2-class | 78.25% | SGD 79.25% | −1.0pp | ✅ Strong |
| D01 — Stat baselines | High-dim noisy | 78.5% | SGD 83.25% | −4.75pp | ✅ Reasonable |
| D06 — Go | Move polarity | **99.5%** | LR/GB 100% | −0.5pp | ✅ Strong |
| D07 — Poker | Hand rank | 93.33% | GB 99.83% | −6.5pp | ✅ Good |
| D08 — MNIST (raw) | Digit class | 72.0% | kNN 91%, LR 94.8% | −22pp | ⚠ Weak (raw pixels) |
| D08 — MNIST (HOG) | Digit class | **89.6%** | LR 94.8% | −5.2pp | ✅ Good |
| D09 — Documents | 20 Newsgroups (16cl) | 33.12% | LR 33.12% | 0pp | ✅ Tied (hard task) |
| D09 — Documents | Gutenberg book cls. | **64.58%** | SGD 33.33% | +31pp | ✅ Beats SGD |
| D11C — RL | Trajectory preference | **92.0%** | — | — | ✅ Strong |
| D12C — Intrusion | Online attack detect | **99.5%** | — | 5-step latency | ✅ Strong |
| D15A — Robustness | Gaussian noise σ=0 | 84.67% | — | — | ✅ Baseline |
| D15C — Robustness | FGSM adversarial | **86.67%** | natural 85.78% | +0.9pp | ✅ Robust |
| D16A — Continual | Task discovery (ARI) | **1.000** | — | perfect routing | ✅ Strong |

### Regression domains

| Domain | Task | Cypha R² | Baseline (best) | Verdict |
|--------|------|---------|----------------|---------|
| D01 — Linear reg. | Linear target | 0.756 | SGD ≈1.000 | ⚠ Gap (LLR linearity) |
| D01 — Nonlinear reg. | Sinusoidal | −0.037 | SGD −0.005 | ⚠ Both poor |
| D05 — Chess | Rating prediction | **0.647** | Ridge 0.658 | ✅ Near-tied |
| D11A — RL | CartPole value fn. | 0.012 | Ridge −0.046 | ⚠ Both poor (sparse RL) |
| D14A — Physics | Feynman equations | −0.010 | — | ❌ Nonlinear equations — hard limit |

### OOD / uncertainty domains

| Domain | Task | Metric | Value | Verdict |
|--------|------|--------|-------|---------|
| D12A — Intrusion | Binary OOD detect | AUROC | **0.889** | ✅ Good |
| D14B — Physics | Extrapolation AUROC | AUROC | 1.000 / 0.000 | ⚠ Classifier yes, regressor no |
| Cross-domain | Mean OOD AUROC | AUROC | **0.844** | ✅ Good |

### Continual learning

| Domain | Task | Metric | Value | Verdict |
|--------|------|--------|-------|---------|
| D16B | Block forgetting (shared model) | Forgetting score | 0.813 | ❌ 81% forgetting — NOT zero-forgetting |
| D16E | Save / restore fidelity | Retention ratio | 1.000 | ✅ Save/restore is perfect |
| D16F | Per-task isolated models | Forgetting score | 0.000 | ✅ Zero forgetting by architecture |

**Key finding on forgetting:** CyphaDIF does NOT have zero forgetting in a shared-model multi-task scenario. `task_a_accuracy_before=0.842` → `task_a_accuracy_after=0.158` after block training on task B. Zero forgetting holds **only** for per-task isolated model files (save + reload). Documentation has been updated to reflect this.

### Language model (D04 + D17)

**Pre-upgrade pin (40k train, `context_mode=full`):**

| Domain | CyphaLM | Bigram | Trigram |
|--------|---------|--------|---------|
| D04 Gutenberg | 5.001 | 3.841 | 4.980 |
| D17 WikiText-2 valid | 4.658 | 3.914 | 4.398 |

**Post-upgrade (2026-05-31, `gria_ngram`, 2 epochs, BPTT-64, Laplace prior):**

| Domain | Train | CyphaLM | Bigram | Trigram | 4-gram | 5-gram | Char-LSTM | Δ vs bi | Δ vs tri |
|--------|-------|---------|--------|---------|--------|--------|-----------|---------|----------|
| D17 | 3k fast | **4.565** | 5.037 | 6.223 | — | — | — | **−0.47** | **−1.66** |
| D04 | 3k fast | **5.233** | 6.201 | 6.675 | — | — | — | **−0.97** | **−1.44** |
| D17 | 40k full | **4.154** | 3.914 | **4.398** | 5.286 | 5.579 | 3.589 | +0.24 | **−0.24** |
| D04 | 40k full | **4.122** | 3.931 | **4.522** | 5.592 | 5.879 | 3.505 | +0.19 | **−0.40** |
| D17 | full corpus (`CYPHA_BENCH_FULL_CORPUS=1`) | *run bench* | *run bench* | *run bench* | *run bench* | *run bench* | *run bench* | — | — |

**Ablation (40k, D17):** `gria_ngram` **4.154** &lt; `full` **4.725** &lt; `ssm_only` **4.451**; `ablation_no_ssm` **4.165** ≈ gria_ngram — explicit n-gram embeds carry most of the gain.

**40k verdict:** CyphaLM **beats trigram** on WikiText valid (−0.24 BPC); still **+0.24 BPC vs bigram**. NumPy char-LSTM (**3.589**) remains strongest baseline — target for next upgrade pass.

| Domain | Task | Metric | Value | Verdict |
|--------|------|--------|-------|---------|
| D17B | Alpha spectrum | mean_alpha | 0.095 (post-upgrade) | ⚠ Still low alpha; 1 active expert |
| D17D | Online adaptation BPC gain | ΔBPC | −0.288 (WikiText) | ✅ Adapts online |

**CyphaLM upgrade (2026 Q2):** config adds `context_mode`, `ngram_context`, `train_epochs`, `bptt_steps`, `laplace_smoothing`, `gria_lr_decay`. Bench adds 4-gram / 5-gram / NumPy char-LSTM baselines and `run_lm_ablations()` (`full`, `gria_ngram`, `ssm_only`, `ablation_no_dif`, `ablation_no_ssm`). Profiles: `cyphalm_d04_gutenberg.json`, `cyphalm_d17_wikitext.json`. Details: [`cypha_lm/README.md`](../cypha_lm/README.md), [`cypha_bench/README.md`](../cypha_bench/README.md).

Config (legacy pin): `cypha_bench/config/cyphalm_profile.json` and per-domain profiles under `config/profiles/`.

D04 runs the full **CyphaLM** stack: learning curve, n-gram + LSTM baselines, context-length BPC, expert routing, save/restore, sampling comparison, ablation summary.

D17 uses **WikiText-2 official train/valid/test** splits (not random 80/20). Requires `cypha_bench/data/wikitext2/` — CI fetches via Hugging Face; bench fails loudly on synthetic fallback unless `CYPHA_BENCH_FAST=1`. Set `CYPHA_BENCH_FULL_CORPUS=1` to train on the entire `wiki.train.tokens` file for beat-bigram runs.

### Known weak domains

| Domain | Task | Cypha result | Root cause |
|--------|------|-------------|-----------|
| D10A | ECG classification (5-class) | 20% (chance) | CellAI SSM not tuned for temporal ECG |
| D10B | ECG sliding window | 17.5% | Same as above |
| D10D | Financial return sign | 49.9% | Efficient market — near-chance is expected |

---

## Confirmed architectural properties

| Property | Evidence | Status |
|----------|----------|--------|
| No catastrophic forgetting (per-task isolation) | D16F: forgetting_score=0.0 | ✅ Confirmed |
| Shared-model forgetting is real | D16B: 81.25% forgetting score | ⚠ Known limit |
| Uncertainty increases with noise | D15: epistemic_var rises with σ | ✅ Confirmed |
| Adversarial robustness | D15C: FGSM minimal drop | ✅ Confirmed |
| OOD detection (AUROC >0.80) | Cross-domain mean 0.844 | ✅ Confirmed |
| Task routing / discovery | D16A: ARI=1.0 | ✅ Confirmed |
| Online adaptation (BPC) | D17D: −0.250 bpc gain | ✅ Confirmed |
| Save/restore fidelity | D16E: retention=1.0 | ✅ Confirmed |
| Linear separability ceiling | D01/XOR: 48.2% vs kernel SVM 80.5% | ✅ Confirmed hard limit |

---

## Confirmed hard limits

| Limit | Evidence | Proposed fix |
|-------|----------|-------------|
| **Nonlinear decision boundaries (XOR etc.)** | 48.2% vs 80.5% kernel SVM — 32.3pp gap | Kernel LLR (Nyström) — top priority |
| **Linear regression gap** | D01 R²=0.756 vs SGD R²≈1.0 for linear targets | Kernel LLR for LLR score + auto-gamma RFF |
| **Feynman equations** | Mean R²=−0.010 on nonlinear physics | Same — Kernel LLR |
| **ECG / temporal** | D10 20% accuracy (chance) | CellAI SSM tuning; temporal-aware features |
| **CyphaLM BPC** | D17 beats trigram at 40k (−0.24 BPC); +0.24 vs bigram; char-LSTM 3.589 | Full WikiText train; close bigram gap; match char-LSTM |
| **MNIST raw** | 72% vs 95% (LR+HOG) | Feature engineering (HOG) bridges most of the gap |

---

## Research history

### Phase 0 — Foundation (2025)

- **CyphaDIF architecture designed:** DIF classifier combining AIXI/Solomonoff, information geometry, Free Energy Principle, and Information Bottleneck.
- **Python reference implementation:** `Cypha.py` (~7.1k lines). Encoders: Vector, RFF, NIG. Expert system: normal-inverse-gamma Bayesian field. Training: online, replay buffer, continual.
- **Core algorithms proven:** GH–NIG world gate, field-conditioned inference, temperature calibration, DIFRegressor, MKERegressor.

### Phase 1 — Studio and REST (early 2026)

- **CyphaStudio built:** PySide6 desktop app with dataset, trainer, experiment, registry, and inference tabs.
- **FastAPI REST server:** `/predict`, `/update`, `/load`, `/register`, `/adapt_temperature` and supporting routes.
- **Binary format v3:** `.cypha` little-endian keyed format; stable across Python and C++.
- **Result:** full Python reference product (studio + API + model).

### Phase 2 — Native C++ port (Q1 2026)

- **Milestones M1–M6 completed:** `cypha_parity`, `cypha_rest`, `cypha_qt_stub` built.
- **Parity tests:** 20+ CTests and subprocess pytest cases. All pass within float64 tolerance.
- **GPU acceleration:** `cypha_accel` fused LLR pipeline (CuPy/NumPy).
- **MinGW cross-build:** Windows PE from WSL; CI job `mingw_cross`.
- **Verified:** binary format round-trip, registry, experiment DB (SQLite), model card, preprocessor, regression head, MKE regressor, two-stage pipeline.
- **Known gap:** deliberation (`deliberation_lo/hi`) and Kernel LLR are Python-only; not in C++.

### Phase 3 — Comprehensive benchmarking (Q2 2026)

- **cypha_bench launched:** 17-domain evaluation harness covering statistical baselines, regression, classification, tabular, images, text, RL, intrusion, information geometry, physics, robustness, continual learning, and language modelling.
- **Tuning run:** `cypha_bench/config/everyday_profile.json` (post-diagnostic). Key change: deliberation disabled, `delta_lr=0.03`.
- **D04 rewritten for CyphaLM (2026-05-31):** prior D04 used `CyphaDIF + CharNgramEncoder` with a probability-indexing bug (33.2 bpc floor). Domain now runs the full CyphaLM stack; held-out **5.202 bpc** vs bigram **4.151** on Gutenberg.
- **Key result:** D17 is the integration benchmark: **~4.50 bpc** on real corpora (Moby Dick / WikiText when installed); online adaptation gain **−0.295 bpc** (D17D).

### Phase 4 — SOM upgrade evaluation (Q2 2026)

- **6 upgrades designed:** U1 GNG, U2 SOM encoder, U3 GRIA controller, U4 discriminative feedback, U5 Hebbian topology, U6 temporal SOM.
- **Benchmark result:** All upgrades degraded accuracy on 7/9 standard domains. U2 SOM encoder was worst. U3/U5/U6 safe but CellAI-specific (no effect on CyphaDIF classification).
- **Decision:** Reverted all. Flags remain in `cypha_som/` (all OFF by default) for CellAI research.
- **Report:** [`docs/reports/SOM_UPGRADE_REPORT.md`](reports/SOM_UPGRADE_REPORT.md)

### Phase 5 — Diagnostics and architecture audit (Q2 2026)

- **Diagnostic suite run:** 9-domain diagnostic; confirmed XOR ceiling, noise robustness, OOD behaviour.
- **XOR confirmed:** 48.2% vs kernel SVM 80.5% — 32.3pp gap; this is the hard LLR-linearity ceiling.
- **Key prescription:** Kernel LLR (Nyström RBF blend) is the evidence-ranked top priority.
- **NIG field investigation:** heavy-tailed inputs handled correctly; `field_a_eff` reduces numerical issues.
- **Report:** [`docs/reports/DIAGNOSTIC_REPORT.md`](reports/DIAGNOSTIC_REPORT.md)

### Phase 6 — CyphaLM (2026)

- **CyphaLM implemented:** Izaac GF(2^n) embeddings + CellAI SSM + CyphaDIF expert routing + GRIA projection.
- **Real evaluation (D17):** 4.50 bpc held-out vs bigram 3.69 (40k train). LM stack is functional but above bigram.
- **Online adaptation works:** D17D shows −0.250 bpc improvement after online adaptation on OOD text.
- **Upgrade track (2026-05):** `context_mode` ablations, Laplace GRIA prior, multi-epoch + BPTT, extended n-gram and char-LSTM baselines, `CYPHA_BENCH_FULL_CORPUS` for full WikiText train.
- **Post-upgrade (40k):** D17 **4.154** BPC (beats trigram **4.398** by −0.24; +0.24 vs bigram). D04 **4.122** BPC (beats trigram **4.522** by −0.40; +0.19 vs bigram). `gria_ngram` ablation wins over `full` on both domains.
- **Paper figures generated** (`paper/figures/`); paper draft `paper/CyphaLM_paper.md` still has `{{EXP0N_*}}` placeholders.
- **Report:** `cypha_lm/REPORT.md` (generated by `scripts/run_cypha_lm_report.py`)

---

## Hypothesis ledger

Each hypothesis we have investigated with the result:

| Hypothesis | Tested | Result |
|-----------|--------|--------|
| GNG prototypes improve routing | Yes (U1) | Degraded 7/9 domains |
| SOM encoder improves features | Yes (U2) | Degraded 7/9 domains |
| GRIA entropy control helps GNG | Structural (U3) | Safe; no measurable classification effect |
| Discriminative feedback improves encoder | Yes (U4) | Degraded structured domains |
| Hebbian diffusion aids CellAI | Structural (U5) | Safe; CellAI-only; not tested on classification |
| Temporal SOM improves SSM decay | Structural (U6) | Safe; CellAI-only |
| Combining all SOM upgrades (U1–U6) | Yes | Worst overall; upgrades interact adversely |
| LLR ceiling explains XOR gap | Yes | Confirmed at 32.3pp — highest-priority fix |
| CupyAccel matches NumPy float64 | Yes | Confirmed in parity tests |
| Save/restore is lossless | Yes | D16E retention_ratio=1.0 |
| Shared-model multi-task = no forgetting | No | D16B: 81.25% forgetting — **refuted** |
| Adversarial robustness is good | Yes | D15C: FGSM minimal drop |
| OOD detection works | Yes | Cross-domain mean AUROC=0.844 |
| CyphaLM can learn language structure | Yes (partial) | D17 **beats trigram** at 40k (4.154 vs 4.398); +0.24 vs bigram; char-LSTM still best |
| Multi-view online training helps LM/DIF | Planned | Spec: [`MULTI_VIEW_TRAINING_PLAN.md`](MULTI_VIEW_TRAINING_PLAN.md) — not yet implemented |
| `gria_ngram` beats `full` on char-LM | In progress | Ablation runner — *run bench to refresh* |
| D04 "33.2 bpc" proves CyphaLM failure | No | **Old benchmark bug — refuted; D04 now CyphaLM** |

---

## Current priorities (ranked by evidence)

### Priority 1 — Kernel LLR (Nyström RBF)

**Evidence:** 32.3pp XOR gap. This is a hard LLR-linearity ceiling that affects all nonlinear domains: XOR, Feynman equations (R²=-0.01), and sinusoidal regression.

**What to do:**
1. Implement `KernelMemory` reservoir + Nyström sketch in `Cypha.py` (prototype exists — `use_kernel_llr=True` flag added 2026-05-30).
2. Benchmark on XOR suite and Feynman D14.
3. If +5pp on XOR, port to C++.

**Prototype state:** Python-only, `use_kernel_llr=True` in `CyphaDIF(...)`. Not in C++. Not in parity fixtures.

### Priority 2 — Auto-gamma RFF

**Evidence:** Manual gamma tuning is the largest single source of improvement in the tuning report. Auto-gamma via cross-validation (`RFFEncoder.auto_gamma_cv`) exists but is not default.

**What to do:**
1. Make `auto_gamma_cv` the default encoder path for `RFFEncoder`.
2. Add to the parity fixture generators.
3. Re-run D08 (MNIST) and D14 (Feynman) with auto-gamma.

### Priority 3 — CyphaLM: beat-bigram roadmap

**Status:** Partial success — **beats trigram at 40k** on D17; bigram and char-LSTM still ahead.

**Evidence (40k train, post-upgrade `gria_ngram`):** D17 **4.154** vs bigram **3.914** (Δ +0.24); vs trigram **4.398** (Δ **−0.24** ✅). Char-LSTM **3.589**. Pre-upgrade D04 **5.001** vs bigram **3.841**; full D04 re-bench pending.

**Root causes (unchanged):**
- Most learning in GRIA; SSM/DIF under-trained at default single-pass online loop.
- Warm-started experts under-reported in D17B (`mean_alpha` low).
- Char-level LM: bigram/trigram are strong; 4/5-gram set an upper practical bound.

**Beat-bigram roadmap (ordered):**

| Step | Action | Success criterion |
|------|--------|-------------------|
| 1 | Profile `context_mode=gria_ngram`, `ngram_context=2`, `train_epochs=2`, `laplace_smoothing=1` | D17 ablation: `gria_ngram` **4.154** &lt; `full` **4.725** ✅ |
| 2 | D17: `bptt_steps=64`, tune `gria_lr` / `tau_slow` via `cyphalm_sweep.py` | Held-out BPC &lt; bigram on 40k — **+0.24 remaining** |
| 3 | `CYPHA_BENCH_FULL_CORPUS=1` WikiText train | BPC &lt; bigram on official valid — *next* |
| 4 | Compare vs **4-gram, 5-gram, char-LSTM** (new baselines) | **&lt; trigram** ✅; char-LSTM **3.589** still best |
| 5 | Fix CyphaDIF warm-start `active_experts` reporting; re-run D17B | `n_experts` matches profile when warm-started |
| 6 | Regenerate `BASELINE_REPORT.md`, paper `fig04_*`, update this doc | All placeholder cells filled |
| 7 | **Multi-view online training** (Phase 1) | See [`MULTI_VIEW_TRAINING_PLAN.md`](MULTI_VIEW_TRAINING_PLAN.md): block shuffle + view_id; BPC &lt; bigram or ≥0.05 ↓ vs 4.154 |

**Commands:**

```powershell
pip install -e cypha_lm/
python cypha_bench/run_all.py --domain 17
$env:CYPHA_BENCH_FULL_CORPUS="1"; python cypha_bench/run_all.py --domain 17
python cypha_bench/tuning/cyphalm_sweep.py --corpus both --n-train 8000 --write-profile
```

See [`cypha_bench/README.md`](../cypha_bench/README.md) (ablations, env vars, baseline table) and [`cypha_lm/README.md`](../cypha_lm/README.md) (config fields).

### Priority 4 — Multi-view online training (CyphaLM → CyphaDIF)

**Status:** Planned — full spec in [`MULTI_VIEW_TRAINING_PLAN.md`](MULTI_VIEW_TRAINING_PLAN.md).

**Idea:** Structure-preserving reorderings (block shuffle, rotated start, bidirectional passes, task-block permutations) each macro-epoch, with explicit `view_id` and memory policy (reset fast / carry slow). Exploits online routing, replay, and expert growth instead of single static stream training.

**Phase 1 (LM):** `cypha_views/` module → D17 **17E_multi_view** → beat bigram or ≥0.05 BPC improvement.  
**Convergence (250k sweep):** `same_order_e2` **peaks @ 40k** then overtrains; `schedule_b` still improving at 250k (best **3.936** BPC). Full write-up: [`FINDINGS_CYPHALM_TRAINING.md`](FINDINGS_CYPHALM_TRAINING.md). Artifacts: `cyphalm_convergence_limit.json`, `cyphalm_beat_bigram_sweep.json` (in progress).

**Next (beat-bigram):** (1b) `cyphalm_beat_bigram_sweep.py` — schedule_b × 70k–150k × hyperparams; (1c) full WikiText + schedule_b; fast axes: laplace, ngram_context, schedule_c, gria_lr_decay.

**Component ablation study:** systematic isolation + combinatorics — [`CYPHALM_ALGORITHM_STUDY.md`](CYPHALM_ALGORITHM_STUDY.md). **Profile updated (2026-06):** `schedule_b`, `alpha_learnable=false`, `gria_lr_decay=0.3`, `ngram_context=3`, `view_id_dim=8`. Phase 1c full-corpus D17 run in progress.

### Priority 5 — Shared-model continual learning

**Evidence:** D16B forgetting_score=0.813. This is a significant architectural gap if the claim is "no forgetting". The current architecture only achieves zero forgetting with per-task isolated model files (D16F).

**What to do:**
1. Investigate elastic weight consolidation (EWC) as a post-hoc overlay on the NIG field.
2. Alternatively, redesign the expert routing so task-specific experts are not overwritten.
3. Update the marketing claim: "no forgetting **per isolated model file**; shared-model continual learning is an open problem."

### Priority 6 — CellAI / ECG / temporal

**Evidence:** D10 ECG: 20% accuracy (5-class chance). The SSM integration is not tuned for temporal signals.

**What to do:**
1. Instrument `CellAISSM` to verify that hidden state norms do not collapse on ECG sequences.
2. Try temporal feature engineering (sliding window FFT, wavelet) before the CyphaDIF classifier.
3. Tune τ_fast/τ_slow decay parameters for ECG sampling rates.

---

## Forward research map

```
2026 Q3 — Priority 1: Kernel LLR prototype → benchmark → port decision
2026 Q3 — Priority 2: Auto-gamma RFF default → D08/D14 re-benchmark
2026 Q4 — Priority 3: CyphaLM beat-bigram (gria_ngram + full WikiText) → D17 re-eval → paper draft
2026 Q4 — Priority 4: Multi-view online training — Phase 1 LM (`MULTI_VIEW_TRAINING_PLAN.md`) → Phase 2 D16/DIF
2026 Q4 — Priority 5: Continual learning investigation → EWC overlay
2027 Q1 — Priority 6: CellAI SSM temporal tuning → D10 re-eval
2027 Q1 — Paper: fill {{EXP0N_*}} placeholders → narrative reconciliation → submit
```

---

## Benchmark report index

| Report | Location | Generated | Contents |
|--------|----------|-----------|---------|
| Post-diagnostic tuned results (main) | [`cypha_bench/BASELINE_REPORT.md`](../cypha_bench/BASELINE_REPORT.md) | 2026-05-30 | 17 domains, full metric tables |
| Tuning history | [`cypha_bench/TUNING_REPORT.md`](../cypha_bench/TUNING_REPORT.md) → [`docs/reports/BENCH_TUNING_REPORT.md`](reports/BENCH_TUNING_REPORT.md) | 2026-05 | Before/after tuning deltas |
| Arch tuning | [`cypha_bench/ARCH_TUNING_REPORT.md`](../cypha_bench/ARCH_TUNING_REPORT.md) → [`docs/reports/BENCH_ARCH_TUNING_REPORT.md`](reports/BENCH_ARCH_TUNING_REPORT.md) | 2026-05 | Architecture grid |
| Arch rescore | [`cypha_bench/ARCH_RESCORE_REPORT.md`](../cypha_bench/ARCH_RESCORE_REPORT.md) → [`docs/reports/BENCH_ARCH_RESCORE_REPORT.md`](reports/BENCH_ARCH_RESCORE_REPORT.md) | 2026-05 | Post-architecture rescore |
| Upgrade evaluation | [`cypha_bench/UPGRADE_REPORT.md`](../cypha_bench/UPGRADE_REPORT.md) → [`docs/reports/BENCH_UPGRADE_REPORT.md`](reports/BENCH_UPGRADE_REPORT.md) | 2026-05 | SOM upgrade + deliberation effects |
| SOM upgrade deep dive | [`docs/reports/SOM_UPGRADE_REPORT.md`](reports/SOM_UPGRADE_REPORT.md) | 2026-05 | U1–U6 detailed results |
| Diagnostic session | [`docs/reports/DIAGNOSTIC_REPORT.md`](reports/DIAGNOSTIC_REPORT.md) | 2026-05 | XOR, noise, uncertainty investigation |
| CyphaLM experiments | `cypha_lm/REPORT.md` | 2026-05-23 | D01–D10 LM experiments, figures |

---

## How to reproduce

```bash
# Full 17-domain benchmark
python benchmark_baseline.py

# Individual domain
python cypha_bench/run_bench.py --domain d17

# CyphaLM domains only
python cypha_bench/run_all.py --domain 4
python cypha_bench/run_all.py --domain 17

# Full WikiText train (beat-bigram)
# PowerShell: $env:CYPHA_BENCH_FULL_CORPUS="1"
python cypha_bench/run_all.py --domain 17

# Legacy perplexity script
python benchmarks/perplexity_eval.py

# Regenerate CyphaLM report + figures
python scripts/run_cypha_lm_report.py

# SOM upgrade evaluation
python scripts/run_som_upgrade_eval.py
```

Fixture regeneration (after intentional `Cypha.py` changes):
```bash
python scripts/generate_parity_fixtures.py
pytest tests/test_parity_fixtures.py -v
```

---

*Maintainer note: update the "Quick state summary" and "Current priorities" tables whenever a phase milestone completes.*
