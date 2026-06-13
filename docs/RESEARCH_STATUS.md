# CyphaDIF — Research Status

**Last updated:** 2026-06-13 (P7 native-only stack; hybrid GRIA+LSTM @ 300k) | **Report:** `bench/BASELINE_REPORT.md` (D17 hybrid refresh) | **Runtime:** native C++ only — `cypha_rest`, `cypha_bench_run`, **53 CTests**

This is the canonical research journal for CyphaDIF and the Cypha stack. It records what we have tried, what the numbers show, what is confirmed, what is broken, and where we are going next. Intended audience: future developers and researchers picking up this project.

---

## Quick state summary

| System | Status | Verdict |
|--------|--------|---------|
| **CyphaDIF classifier** | Working, benchmarked | Competitive on linear/tabular; hard limit on nonlinear boundaries |
| **CyphaDIF regressor (DIFRegressor)** | Working | Comparable to Ridge on smooth domains; poor on nonlinear equations |
| **Native C++ / CUDA / Qt (M1–M6 + P7)** | Shipped | Sole production runtime; Kernel LLR in `native/src/kernel_memory.cpp`; **53 CTests** gate CI |
| **cypha::accel (GPU fused kernels)** | Working | Native CUDA when `-DCYPHA_ENABLE_CUDA=ON`; ISO C++ thread fallback |
| **CyphaLM (native)** | Best @ 300k: **2.873 BPC** (`hybrid_gria_lstm`) | **Beats bigram (−0.61)** and char-LSTM bench (−0.11); GRIA-only stack **3.838**; via `cyphalm_bench_native` / REST `/generate` — long-range + V2 sweeps → [`CYPHALM_LONG_RANGE_TESTS.md`](CYPHALM_LONG_RANGE_TESTS.md), [`CYPHALM_MODEL_CLASS_RESEARCH.md`](CYPHALM_MODEL_CLASS_RESEARCH.md) |
| **cypha_som (SOM upgrades)** | Removed (archived) | Failed experiment — see [`docs/archive/failed_experiments/cypha_som/README.md`](archive/failed_experiments/cypha_som/README.md) |
| **Bench harness (`cypha_bench_run`)** | 17 domains complete | D04 full LLM suite; D17 extended integration; configs + reports under `bench/` |
| **cypha_qt_shell / cypha_rest (native)** | Working | Qt Studio + cpp-httplib REST + registry; **CyphaLM `/generate` + `/generate/stream` (SSE)** on native `cypha_rest` |

---

## Benchmark results — all 17 domains

Run on 2026-05-31 using `bench/config/everyday_profile.json` (deliberation off, `delta_lr=0.03`). Full raw report: [`bench/BASELINE_REPORT.md`](../bench/BASELINE_REPORT.md).

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
| D09 — Branch A (frozen MiniLM, 2k) | 20 Newsgroups | **62.5%** | LogReg 60.3% | +2.2pp | ✅ CyphaDIF on ST embeddings |
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
| D04 Gutenberg | 40k full | **4.122** | 3.931 | **4.522** | 5.592 | 5.879 | 3.505 | +0.19 | **−0.40** |
| D04 Gutenberg | 300k (`hybrid_gria_lstm`, Moby Dick) | **2.993** | 3.633 | 3.424 | 4.040 | 4.941 | 3.047 | **−0.64** | **−0.43** |
| D17 | 300k (`hybrid_gria_lstm`, Phase 1c) | **2.873** | 3.478 | 4.398 | — | — | 2.979 | **−0.61** | **−1.53** |
| D17 | 300k (`gria_ngram` stack) | **3.838** | 3.478 | 4.398 | — | — | 2.979 | +0.36 | **−0.56** |

**Ablation (40k, D17):** `gria_ngram` **4.154** &lt; `full` **4.725** &lt; `ssm_only` **4.451**; `ablation_no_ssm` **4.165** ≈ gria_ngram — explicit n-gram embeds carry most of the gain.

**300k verdict (2026-06-07):** **Hybrid GRIA+LSTM** (`hybrid_gria_lstm`) **beats bigram, trigram, and char-LSTM bench** on WikiText valid. Blend learns ~**99.6% LSTM**. Default D17 profile updated; Phase 1c cap **300k** (`CYPHA_BENCH_FULL_N_TRAIN`).

| Domain | Task | Metric | Value | Verdict |
|--------|------|--------|-------|---------|
| D17B | Alpha spectrum | mean_alpha | 0.095 (post-upgrade) | ⚠ Still low alpha; 1 active expert |
| D17D | Online adaptation BPC gain | ΔBPC | −0.288 (WikiText) | ✅ Adapts online |

**CyphaLM upgrade (2026 Q2):** config adds `context_mode`, `ngram_context`, `train_epochs`, `bptt_steps`, `laplace_smoothing`, `gria_lr_decay`. Native bench adds 4-gram / 5-gram / char-LSTM baselines and LM ablations (`full`, `gria_ngram`, `ssm_only`, `ablation_no_dif`, `ablation_no_ssm`). Profiles: `bench/config/profiles/cyphalm_d04_gutenberg.json`, `cyphalm_d17_wikitext.json`. Details: [`docs/native/CYPHALM_NATIVE_BUILD.md`](native/CYPHALM_NATIVE_BUILD.md), [`docs/port/PORT_CONTRACT.md`](port/PORT_CONTRACT.md) §6.

Config (legacy pin): `bench/config/cyphalm_profile.json` and per-domain profiles under `config/profiles/`.

D04 runs the full **CyphaLM** stack: learning curve, n-gram + LSTM baselines, context-length BPC, expert routing, save/restore, sampling comparison, ablation summary.

D17 uses **WikiText-2 official train/valid/test** splits (not random 80/20). Requires `bench/data/wikitext2/` — CI fetches via Hugging Face; bench fails loudly on synthetic fallback unless `CYPHA_BENCH_FAST=1`. Set `CYPHA_BENCH_FULL_CORPUS=1` to train on the entire `wiki.train.tokens` file for beat-bigram runs.

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
| **CyphaLM BPC (GRIA-only)** | GRIA stack @ 300k **3.838** (+0.36 vs bigram); hybrid **2.873** resolves gap | Hybrid is default; GRIA-only path for ablation / long-range SSM probes |
| **MNIST raw** | 72% vs 95% (LR+HOG) | Feature engineering (HOG) bridges most of the gap |

---

## Research history

### Phase 0 — Foundation (2025)

- **CyphaDIF architecture designed:** DIF classifier combining AIXI/Solomonoff, information geometry, Free Energy Principle, and Information Bottleneck.
- **Python reference implementation:** `cypha_core` (~7.1k lines). Encoders: Vector, RFF, NIG. Expert system: normal-inverse-gamma Bayesian field. Training: online, replay buffer, continual.
- **Core algorithms proven:** GH–NIG world gate, field-conditioned inference, temperature calibration, DIFRegressor, MKERegressor.

### Phase 1 — Studio and REST (early 2026)

- **CyphaStudio built (Python reference, removed P7):** PySide6 desktop app with dataset, trainer, experiment, registry, and inference tabs — superseded by **`cypha_qt_shell`**.
- **REST server:** `/predict`, `/update`, `/load`, `/register`, `/adapt_temperature` and supporting routes — now **`cypha_rest`** (cpp-httplib); Python FastAPI removed P7.
- **Binary format v3:** `.cypha` little-endian keyed format; authoritative in native C++.
- **Result:** native Qt shell + REST + model runtime (P7 decommission).

### Phase 2 — Native C++ port (Q1 2026)

- **Milestones M1–M6 completed:** `cypha_parity`, `cypha_rest`, `cypha_qt_shell` built.
- **Parity tests:** **53 CTests** (`ctest -R native_`) and subprocess cases. All pass within float64 tolerance.
- **GPU acceleration:** `cypha::accel` fused LLR pipeline (CUDA / parallel CPU).
- **MinGW cross-build:** Windows PE from WSL; CI job `mingw_cross`.
- **Verified:** binary format round-trip, registry, experiment DB (SQLite), model card, preprocessor, regression head, MKE regressor, two-stage pipeline.
- **Kernel LLR:** Nyström whitening in C++ (`native/src/kernel_memory.cpp`, CTest `native_kernel_llr`); deliberation native (`native_gh_infer_deliberation`).

### Phase 3 — Comprehensive benchmarking (Q2 2026)

- **`bench/` launched:** 17-domain evaluation harness covering statistical baselines, regression, classification, tabular, images, text, RL, intrusion, information geometry, physics, robustness, continual learning, and language modelling.
- **Tuning run:** `bench/config/everyday_profile.json` (post-diagnostic). Key change: deliberation disabled, `delta_lr=0.03`.
- **D04 rewritten for CyphaLM (2026-05-31):** prior D04 used `CyphaDIF + CharNgramEncoder` with a probability-indexing bug (33.2 bpc floor). Domain now runs the full CyphaLM stack; held-out **5.202 bpc** vs bigram **4.151** on Gutenberg.
- **Key result:** D17 is the integration benchmark: **~4.50 bpc** on real corpora (Moby Dick / WikiText when installed); online adaptation gain **−0.295 bpc** (D17D).

### Phase 4 — SOM upgrade evaluation (Q2 2026)

- **6 upgrades designed:** U1 GNG, U2 SOM encoder, U3 GRIA controller, U4 discriminative feedback, U5 Hebbian topology, U6 temporal SOM.
- **Benchmark result:** All upgrades degraded accuracy on 7/9 standard domains. U2 SOM encoder was worst. U3/U5/U6 safe but CellAI-specific (no effect on CyphaDIF classification).
- **Decision:** Reverted all. Python package removed; native SOM smoke tests remain under `native/src/som/`.
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
- **Report:** `bench/BASELINE_REPORT.md` (regenerated via `cypha_bench_run --report-only`)

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
| Native CUDA accel matches CPU float64 | Yes | Confirmed in `native_score_batch` / `native_cuda_smoke` |
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
1. ~~Implement `KernelMemory` reservoir + Nyström sketch~~ — **done** (native C++, median-γ whitening).
2. Benchmark on XOR suite and Feynman D14 — `cypha_bench_run --domain-tag d03_xor`; `xor_kernel_bench` CTest smoke.
3. ~~Wire native kernel train in `memory_train.cpp`~~ — **done**; online XOR bench via `xor_kernel_bench`.

**Current state:** Nyström whitening native-only; M=256; XOR **+10.6 pp** (61.1% kernel vs 50.5% linear, 5 seeds); sklearn RBF ceiling **~79%** on same splits — **~18 pp** gap remains.

### Priority 2 — Auto-gamma RFF

**Evidence:** Manual gamma tuning is the largest single source of improvement in the tuning report. Auto-gamma via cross-validation (`RFFEncoder.auto_gamma_cv`) exists but is not default.

**What to do:**
1. Make `auto_gamma_cv` the default encoder path for `RFFEncoder`.
2. Add to the parity fixture generators.
3. Re-run D08 (MNIST) and D14 (Feynman) with auto-gamma.

### Priority 3 — CyphaLM: beat-bigram roadmap

**Status:** ✅ **Achieved @ 300k** via **hybrid GRIA+LSTM** — D17 **2.873 BPC** (−0.61 vs bigram, −0.11 vs char-LSTM bench). GRIA-only stack peaked @ **3.838** (+0.36 vs bigram).

**Evidence (300k, `hybrid_gria_lstm`):** D17 Phase 1c **2.873**; D04 Moby Dick bench **2.993** (learning-curve run; sweep **2.859**); bigram D17 **3.478** / D04 **3.633**. Cypha Tests **1A pass @ char shuffle** (+4.54 BPC @ 300k hybrid).

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
cmake --build native/build --target cyphalm_bench_native
cypha_bench_run --domain 17
$env:CYPHA_BENCH_FULL_CORPUS="1"; cypha_bench_run --domain 17
cypha_tune_run --config bench/config/cyphalm_sweep.json --corpus both --n-train 8000 --write-profile
```

See [`docs/port/PORT_CONTRACT.md`](port/PORT_CONTRACT.md) §6 (bench env vars, domain tags) and [`docs/native/NATIVE_QUICKSTART.md`](native/NATIVE_QUICKSTART.md) (bench/tune commands).

### Priority 4 — Multi-view online training (CyphaLM → CyphaDIF)

**Status:** Planned — full spec in [`MULTI_VIEW_TRAINING_PLAN.md`](MULTI_VIEW_TRAINING_PLAN.md).

**Idea:** Structure-preserving reorderings (block shuffle, rotated start, bidirectional passes, task-block permutations) each macro-epoch, with explicit `view_id` and memory policy (reset fast / carry slow). Exploits online routing, replay, and expert growth instead of single static stream training.

**Phase 1 (LM):** Multi-view + convergence complete. **Hybrid @ 300k: 2.873 BPC (D17), 2.993 (D04).**  
**Long-range context (Cypha Tests 1C):** SSM warm-up + reset probes pass @ 300k. **1A @ char shuffle: +4.54 BPC** (block shuffle flat). See [`CYPHALM_LONG_RANGE_TESTS.md`](CYPHALM_LONG_RANGE_TESTS.md).  
**Upgrade V2:** Learnable views neutral; gated fusion worse — keep fixed views + sum fusion.  
**Model-class C2 hybrid:** **Default profile.** See [`CYPHALM_MODEL_CLASS_RESEARCH.md`](CYPHALM_MODEL_CLASS_RESEARCH.md).

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
2026 Q4 — Priority 3: CyphaLM — Phase 1c full corpus; **Upgrade V2** (learnable views + fusion) + **model-class** (char-LSTM hybrid) in parallel
2026 Q4 — Priority 4: Multi-view online training — Phase 1 LM (`MULTI_VIEW_TRAINING_PLAN.md`) → Phase 2 D16/DIF
2026 Q4 — Priority 5: Continual learning investigation → EWC overlay
2027 Q1 — Priority 6: CellAI SSM temporal tuning → D10 re-eval
2027 Q1 — Paper: fill {{EXP0N_*}} placeholders → narrative reconciliation → submit
```

---

## Benchmark report index

| Report | Location | Generated | Contents |
|--------|----------|-----------|---------|
| Post-diagnostic tuned results (main) | [`bench/BASELINE_REPORT.md`](../bench/BASELINE_REPORT.md) | 2026-05-30 | 17 domains, full metric tables |
| Tuning history | [`bench/TUNING_REPORT.md`](../bench/TUNING_REPORT.md) → [`docs/reports/BENCH_TUNING_REPORT.md`](reports/BENCH_TUNING_REPORT.md) | 2026-05 | Before/after tuning deltas |
| Arch tuning | [`bench/ARCH_TUNING_REPORT.md`](../bench/ARCH_TUNING_REPORT.md) → [`docs/reports/BENCH_ARCH_TUNING_REPORT.md`](reports/BENCH_ARCH_TUNING_REPORT.md) | 2026-05 | Architecture grid |
| Arch rescore | [`bench/ARCH_RESCORE_REPORT.md`](../bench/ARCH_RESCORE_REPORT.md) → [`docs/reports/BENCH_ARCH_RESCORE_REPORT.md`](reports/BENCH_ARCH_RESCORE_REPORT.md) | 2026-05 | Post-architecture rescore |
| Upgrade evaluation | [`bench/UPGRADE_REPORT.md`](../bench/UPGRADE_REPORT.md) → [`docs/reports/BENCH_UPGRADE_REPORT.md`](reports/BENCH_UPGRADE_REPORT.md) | 2026-05 | SOM upgrade + deliberation effects |
| SOM upgrade deep dive | [`docs/reports/SOM_UPGRADE_REPORT.md`](reports/SOM_UPGRADE_REPORT.md) | 2026-05 | U1–U6 detailed results |
| Diagnostic session | [`docs/reports/DIAGNOSTIC_REPORT.md`](reports/DIAGNOSTIC_REPORT.md) | 2026-05 | XOR, noise, uncertainty investigation |
| CyphaLM experiments | `bench/BASELINE_REPORT.md` (D04/D17 sections) | 2026-05-31 | LM BPC, context curves, ablations |

---

## How to reproduce

```bash
# Full 17-domain benchmark
cypha_bench_run

# Individual domain
cypha_bench_run --domain-tag d17

# CyphaLM domains only
cypha_bench_run --domain 4
cypha_bench_run --domain 17

# Full WikiText train (beat-bigram)
# PowerShell: $env:CYPHA_BENCH_FULL_CORPUS="1"
cypha_bench_run --domain 17

# Legacy perplexity script
cyphalm_bench_native --profile d17

# Regenerate CyphaLM report + figures
cypha_bench_run --report-only
```

Full native validation gate:
```powershell
powershell -File scripts\cypha_native_validate_all.ps1
```

Parity fixture updates (after intentional contract changes): see [`docs/verify/MAINTENANCE.md`](verify/MAINTENANCE.md) — update `fixtures/` sidecars, then `ctest -R native_<fixture>`.

---

*Maintainer note: update the "Quick state summary" and "Current priorities" tables whenever a phase milestone completes.*
