# CyphaDIF — Research Status

**Last updated:** 2026-05-31 | **Report:** `cypha_bench/BASELINE_REPORT.md` (2026-05-31)

This is the canonical research journal for CyphaDIF and the Cypha stack. It records what we have tried, what the numbers show, what is confirmed, what is broken, and where we are going next. Intended audience: future developers and researchers picking up this project.

---

## Quick state summary

| System | Status | Verdict |
|--------|--------|---------|
| **CyphaDIF classifier** | Working, benchmarked | Competitive on linear/tabular; hard limit on nonlinear boundaries |
| **CyphaDIF regressor (DIFRegressor)** | Working | Comparable to Ridge on smooth domains; poor on nonlinear equations |
| **C++ / CUDA / Qt port** | M1–M6 complete | Parity with Python on all ported ops; deliberation and kernel LLR Python-only |
| **cypha_accel (GPU fused kernels)** | Working | CuPy GPU path used automatically; NumPy fallback |
| **cypha_lm (CyphaLM)** | Research prototype | D17: 4.50 bpc (bigram: 3.69); above bigram but not competitive |
| **cypha_som (SOM upgrades)** | Benchmarked, reverted | All upgrades worse than baseline; U3/U5/U6 structurally safe |
| **cypha_bench (eval harness)** | 17 domains complete | Comprehensive; D04 re-designed to use proper 80/20 held-out eval |
| **cypha_studio (PySide6 + FastAPI)** | Working | GUI + REST + registry; native `cypha_rest` also complete |

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

### Language model (D17)

| Domain | Task | Metric | CyphaLM | Bigram | Verdict |
|--------|------|--------|---------|--------|---------|
| D17 — CyphaLM | Held-out BPC (Gutenberg) | bits/char | 4.497 | 3.691 | ⚠ Above bigram |
| D17B | Alpha spectrum | mean_alpha | 0.1875 | — | ⚠ Low alpha (1 expert) |
| D17D | Online adaptation BPC gain | ΔBPC | −0.250 | — | ✅ Adapts online |

### Language model (D04 char-level)

| Domain | Task | Cypha BPC | SGD BPC | Verdict |
|--------|------|-----------|---------|---------|
| D04 | Char LM — held-out 20% (100-char vocab) | **32.61** | 6.64 (random) | ❌ CyphaDIF concentrates probability mass; unseen chars get floor prob |

The `probs[char_id]` indexing bug has been fixed (2026-05-31). The new evaluation uses an 80/20 train/test split and reports mean BPC over the held-out suffix. The result (32.61 bpc, worse than random 6.64 = log₂(100)) is the **correct honest result**: CyphaDIF is a discriminative classifier optimised for accuracy, not probability calibration — it concentrates probability mass on the top predicted class and gives near-zero probability to rare characters. **This is a fundamental limitation of CyphaDIF for any LM task.** D17/CyphaLM (4.50 bpc) is the dedicated language-model component.

### Known weak domains

| Domain | Task | Cypha result | Root cause |
|--------|------|-------------|-----------|
| D04 | Char LM BPC | 32.61 bpc (worse than random 6.64) | CyphaDIF concentrates probability; not a LM. Use D17/CyphaLM instead. |
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
| **CyphaLM BPC** | 4.50 bpc vs bigram 3.69 | Longer training; SSM decay tuning; multi-expert |
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
- **D04 bug discovered:** `d04_generation_language.py` runs `CyphaDIF + CharNgramEncoder` not `CyphaLM`; probability indexing bug yields 33.2 bpc floor. All references to "33.2 bpc CyphaLM failure" in prior docs were incorrect — fixed.
- **Key result:** D17 is the real CyphaLM benchmark: **4.50 bpc** (Moby Dick, held-out), bigram baseline 3.69.

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
- **Real evaluation (D17):** 4.50 bpc held-out vs bigram 3.69. LM stack is functional but above bigram.
- **Online adaptation works:** D17D shows −0.250 bpc improvement after online adaptation on OOD text.
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
| CyphaLM can learn language structure | Partial | D17: 4.50 bpc; adapts online; above bigram |
| D04 "33.2 bpc" proves CyphaLM failure | No | **Benchmark bug — refuted** |

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

### Priority 3 — CyphaLM BPC improvement

**Evidence:** D17 BPC=4.50 vs bigram 3.69. The gap is likely from:
- Short context window (CellAI SSM τ values not tuned for text).
- Single expert (`n_experts=1` in D17B; mean_alpha=0.1875 — very low, near-zero experts active).
- Gutenberg tokenization is character-level — bigram is a strong baseline here.

**What to do:**
1. Try `n_experts > 1` in D17 config (current D17B shows only 1 active expert — routing is collapsed).
2. Tune SSM decay constants (τ_fast, τ_slow) for character-level LM.
3. Fix D04 probability indexing bug in `cypha_bench/domains/d04_generation_language.py`.
4. Fill paper placeholders from `cypha_lm/REPORT.md` and `paper/figures/`.

### Priority 4 — Shared-model continual learning

**Evidence:** D16B forgetting_score=0.813. This is a significant architectural gap if the claim is "no forgetting". The current architecture only achieves zero forgetting with per-task isolated model files (D16F).

**What to do:**
1. Investigate elastic weight consolidation (EWC) as a post-hoc overlay on the NIG field.
2. Alternatively, redesign the expert routing so task-specific experts are not overwritten.
3. Update the marketing claim: "no forgetting **per isolated model file**; shared-model continual learning is an open problem."

### Priority 5 — CellAI / ECG / temporal

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
2026 Q4 — Priority 3: CyphaLM n_experts tuning → D17 re-eval → paper draft
2026 Q4 — Priority 4: Continual learning investigation → EWC overlay
2027 Q1 — Priority 5: CellAI SSM temporal tuning → D10 re-eval
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

# CyphaLM evaluation only
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
