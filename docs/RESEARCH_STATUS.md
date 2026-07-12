# CyphaDIF — Research Status

**Last updated:** 2026-07-12  
**Runtime:** native C++ only — `cypha_rest`, `cypha_bench_run`, **160 CTests** *(see `scripts/cypha_native_validate_all.ps1` for the current authoritative count)*

This is the canonical research journal for CyphaDIF and the Cypha stack. It records what we have tried, what the numbers show, what is confirmed, what is broken, and where we are going next. Intended audience: future developers and researchers picking up this project.

---

## Quick state summary

| System | Status | Verdict |
|--------|--------|---------|
| **CyphaDIF classifier** | Working, benchmarked | Competitive on linear/tabular; hard limit on nonlinear boundaries |
| **CyphaDIF regressor (DIFRegressor)** | Working | Comparable to Ridge on smooth domains; poor on nonlinear equations |
| **Native C++ / CUDA / Qt (M1–M6 + P7)** | Shipped | Sole production runtime; Kernel LLR in `native/src/kernel_memory.cpp`; **160 CTests** gate CI *(see `scripts/cypha_native_validate_all.ps1` for the current authoritative count)* |
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
| D14A — Physics | Feynman equations | 0.444 *(re-run 2026-07-11; was −0.010 on 2026-05-31)* | Ridge (per-eq., beaten on all 20) | ✅ Beats Ridge per-equation today — see Priority 1 update |

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

D17 uses **WikiText-2 official train/valid/test** splits (not random 80/20). Requires `bench/data/wikitext2/` — download via **`scripts/download_wikitext2.ps1`** or **`scripts/download_wikitext2.sh`** (see **`bench/data/wikitext2/README.md`**); CI may fetch via Hugging Face. When WikiText is absent, **`load_bench_corpus`** falls back to **`bench/data/gutenberg/*.txt`** with source tag **`gutenberg_fallback`** (Moby Dick preferred). Bench fails loudly on missing corpus unless `CYPHA_BENCH_FAST=1` (synthetic fallback in overnight/bench smokes). Set `CYPHA_BENCH_FULL_CORPUS=1` to train on the entire `wiki.train.tokens` file with **`wiki.valid.tokens`** held out (see `bench/config/d17_wikitext_full_profile.json`). **Overnight 300k run:** `bench/config/d17_wikitext_overnight_profile.json`, `cyphalm_bench_native --overnight`, or `CYPHA_BENCH_OVERNIGHT=1`; **`-Fast`** on overnight scripts sets **`CYPHA_BENCH_FAST=1`** so runs succeed without WikiText. CTests **`native_d17_wikitext_smoke`** (512 train), **`native_d17_wikitext_overnight_smoke`** (500 train), **`native_overnight_mini_smoke`** (800 train), **`native_corpus_smoke`**, and **`native_d25_corpus_smoke`** (d25 corpus readiness) use FAST synthetic or gutenberg fallback. Baseline lock: [`bench/BASELINE_LOCK.json`](../bench/BASELINE_LOCK.json).

### Known weak domains

| Domain | Task | Cypha result | Root cause |
|--------|------|-------------|-----------|
| D10A | ECG classification (5-class) | **60.67%** (~3× chance; stale 20%/17.5% figures retired 2026-07-11) | Not an SSM issue — 10A–10D route through the `cypha_core` DIF expert-routing classifier, not `CellAISSM`; see [`D10_ECG_SSM_DIAGNOSIS_2026-07-11.md`](reports/D10_ECG_SSM_DIAGNOSIS_2026-07-11.md) |
| D10B | ECG sliding window | 29.46% | Same path as above; modestly above chance |
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
| **Nonlinear decision boundaries (XOR etc.)** | 48.2% vs 80.5% kernel SVM — 32.3pp gap | Kernel LLR (Nyström) — **partially shipped**; tuning continues ([`upgrades/NONLINEAR_BOUNDARY.md`](research/upgrades/NONLINEAR_BOUNDARY.md)) |
| **Linear regression gap** | D01 R²=0.756 vs SGD R²≈1.0 for linear targets | Kernel LLR for LLR score + auto-gamma RFF |
| **Feynman equations** | Mean R²=−0.010 on nonlinear physics *(2026-05-31; re-run 2026-07-11 now shows R²=0.444, beats Ridge on all 20 equations — see Priority 1 update; kernel LLR not applicable, D14 uses a separate linear expert-mixture regressor, not `KernelMemory`)* | Re-run shows this is no longer a hard limit on current HEAD; kernel LLR wiring there deferred (different subsystem) |
| **ECG / temporal** | *(retired 2026-07-11 — no longer a hard limit)* D10A now scores **60.67%** (~3× chance) via the DIF classifier; the SSM was never on this path — see [`D10_ECG_SSM_DIAGNOSIS_2026-07-11.md`](reports/D10_ECG_SSM_DIAGNOSIS_2026-07-11.md) | N/A — resolved incidentally; no SSM defect existed |
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
- **Parity tests:** **106 CTests** (`ctest -R native_`) and subprocess cases (suite grew from 64 at Phase 2). All pass within float64 tolerance.
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

### Phase 7 — Intelligence Stats baseline lock (2026-06-14)

- **`bench/BASELINE_LOCK.json`:** D17 hybrid **2.873 BPC** @ 300k reference pin; **`scripts/publish_release.ps1`** for local `gh release create`.
- **Overnight wiring:** CTest **`native_overnight_mini_smoke`** (800-train `--overnight` check).
- **Cell / RPSM / EWC:** H16 SR gate laws; bench **d21** RPSM end-to-end; CyphaLM EWC embed + lm_head Fisher; optional federated TLS smoke.
- **CI gate:** **93 CTests** (`ctest -R native_`); optional **`federated_tls`** job (`continue-on-error`).

### Phase 8 — Hybrid EWC + cross-profile intelligence (2026-06-14)

- **Hybrid EWC:** `HybridEwcRegularizer` snapshots SSM multiscale **α** + GRIA per-token **α** with diagonal Fisher penalty; B0 **`ngram_count_table`** online prior when `ngram_context > 0`; H01 **`use_alpha_forget_gate`** scales char-LSTM forget gate by mean GRIA α. CTest **`native_ewc_hybrid_smoke`**.
- **Baseline lock tooling:** **`cypha_baseline_lock`** merges D17 / d21 / cell-sweep BPC into **`bench/BASELINE_LOCK.json`**; **`scripts/update_baseline_lock.ps1`** wrapper; CTest **`native_baseline_lock_smoke`**.
- **Cross-domain bench d22:** `d22_intelligence_cross_profile.json` — d18 intelligence report + d16 EWC probe + d20 cell sweep smoke; CTest **`native_d22_cross_smoke`**.
- **Checkpoint fix:** CyphaLM save/load persists **`ngram_count_table`** in checkpoint JSON (B0 count path).
- **CI:** optional **`federated_tls`** job (**`-DCYPHA_ENABLE_OPENSSL=ON`**, **`native_federated_tls_smoke`**); local mirror **`scripts/ci_federated_tls_linux.sh`**. **96 CTests** in blocking gate.

### Phase 9 — hybrid EWC weights + overnight lock validation (v2.3.9, shipped 2026-06-14)

- **Hybrid EWC weight Fisher:** extend `HybridEwcRegularizer` beyond Phase 8 α Fisher — diagonal Fisher on GRIA low-rank **U**/**V** (`gria_lowrank.hpp`) and SSM per-layer **W_fast** (`cellai_ssm.hpp`); char-LSTM embed/head Fisher retained. CTest **`native_ewc_weights_smoke`**.
- **EWC checkpoint persistence:** CyphaLM `checkpoint.json` save/load of EWC anchor + running Fisher blocks so overnight / continual runs resume without re-snapshot. Validated in weight-Fisher smoke round-trip.
- **Unified overnight runner:** **`scripts/run_overnight_all.ps1`** — orchestrates D17 WikiText 300k (`run_d17_overnight.ps1`), d21 RPSM overnight (`run_rpsm_overnight.ps1`), 28-variant cell sweep, then **`update_baseline_lock.ps1`** to refresh **`bench/BASELINE_LOCK.json`**.
- **Bench d23:** overnight lock validation — FAST smoke wiring for `BASELINE_LOCK.json` schema, mini overnight token budget, and cross-check vs D17 hybrid **2.873 BPC** pin; profile **`bench/config/d23_overnight_lock_profile.json`**. CTest **`native_d23_overnight_lock_smoke`**.
- **CI:** **98 CTests** shipped (Phase 9).

### Phase 10 — EWC bias/W_slow + production lock (v2.3.10) — shipped

- **Hybrid EWC bias + W_slow Fisher:** diagonal Fisher on GRIA **`bias`** and SSM **`W_slow`** layer-0; extends Phase 9 U/V/W_fast Fisher. CTest **`native_ewc_weights_smoke`**.
- **Baseline lock `--run all`:** **`cypha_baseline_lock --run all`** chains d17 → d21 → cell-sweep; cell-sweep merges into **`cell_sweep_results`**.
- **Bench d24:** production lock validation via **`native_d24_production_lock_smoke`** (~7s fast).
- **Federated TLS Windows CI mirror:** **`scripts/ci_federated_tls_windows.ps1`**.
- **CI:** **99 CTests** (+1 d24 smoke; extended weight Fisher smoke).

### Phase 11 — corpus readiness + overnight -Fast (v2.3.11) — shipped

- **WikiText-2 download tooling:** **`scripts/download_wikitext2.ps1`** (PowerShell 5+) and **`scripts/download_wikitext2.sh`** (Linux/CI) fetch Salesforce WikiText-2 raw into **`bench/data/wikitext2/wikitext-2/`**; layout documented in **`bench/data/wikitext2/README.md`**.
- **Gutenberg fallback:** when WikiText is absent, **`load_bench_corpus("d17"|"d21", ...)`** in **`cyphalm_corpus.cpp`** uses **`bench/data/gutenberg/*.txt`** (Moby Dick preferred) with source tag **`gutenberg_fallback`** instead of throwing.
- **Corpus smoke:** **`corpus_smoke`** CLI probes d17 + d21 corpus load; CTest **`native_corpus_smoke`**.
- **Bench d25:** corpus readiness validation — **`run_d25_corpus_readiness`** checks WikiText or gutenberg fallback, optionally invokes **`corpus_smoke`**, writes **`bench/report/tables/d25_corpus_readiness.json`**; profile **`bench/config/d25_corpus_readiness_profile.json`**. CTest **`native_d25_corpus_smoke`**.
- **Overnight `-Fast` fix:** **`run_d17_overnight.ps1`**, **`run_rpsm_overnight.ps1`**, **`run_overnight_all.ps1`**, and **`update_baseline_lock.ps1`** propagate **`-Fast`** and set **`CYPHA_BENCH_FAST=1`** so overnight/baseline-lock smokes run without WikiText installed.
- **CI:** **101 CTests** (+1 **`native_d25_corpus_smoke`**; also **`native_corpus_smoke`**).

### Phase 12 — medium overnight tier + baseline lock validator (v2.3.12) — shipped everywhere

- **Medium overnight tier:** **`-Medium`** on **`run_d17_overnight.ps1`**, **`run_rpsm_overnight.ps1`**, **`run_overnight_all.ps1`**, and **`update_baseline_lock.ps1`** — 5k train / 256 eval, real WikiText or gutenberg fallback (no **`CYPHA_BENCH_FAST`**). **`cypha_baseline_lock --medium`** writes **`status=medium_smoke`** to **`overnight_results`**.
- **Bench d26:** medium overnight lock validation — **`run_d26_medium_overnight_validation`** runs **`cypha_baseline_lock --run d17 --medium`**, checks finite BPC and **`medium_smoke`** status; profile **`bench/config/d26_medium_overnight_profile.json`**. CTest **`native_d26_medium_overnight_smoke`**.
- **Baseline lock validator:** **`scripts/validate_baseline_lock.ps1`** (`-LockFile`, `-Strict`) and **`baseline_lock_validate`** CLI — schema_version, d17 hybrid **2.873 BPC** pin, overnight/rpsm/cell-sweep sections. CTest **`native_baseline_lock_validate_smoke`**.
- **Release preview:** **`scripts/publish_release.ps1 -DryRun`** / **`-NotesOnly`** — generate release notes without calling **`gh`**.
- **CI:** optional **`corpus_and_d25`** job (WikiText fetch + **`native_corpus_smoke`** / **`native_d25_corpus_smoke`**, `continue-on-error`). Blocking gate **103 CTests** at Phase 12 (+2 d26 + baseline-lock validate smokes).

### Phase 13 — production overnight tier (v2.3.13) — shipped

- **Production overnight tier:** **`-Production`** on **`run_d17_overnight.ps1`**, **`run_rpsm_overnight.ps1`**, **`run_overnight_all.ps1`**, and **`update_baseline_lock.ps1`** — 300k train / 2000 eval, real WikiText or gutenberg (mutually exclusive with **`-Fast`** / **`-Medium`**). **`cypha_baseline_lock --production`** sets **`CYPHA_BENCH_FULL_CORPUS=1`**, **`CYPHA_BENCH_OVERNIGHT=1`**, **`CYPHA_BENCH_FULL_N_TRAIN=300000`**, writes **`status=production`** to **`overnight_results`**.
- **Dedicated production runner:** **`scripts/run_production_overnight.ps1`** — chains **`run_overnight_all.ps1 -Production`**, logs to **`bench/results/production_overnight_<timestamp>.log`**. Maintainer-only; not run in CI.
- **Bench d27:** production overnight lock validation — **`run_d27_production_lock_validation`** validates **`bench/BASELINE_LOCK.json`** for production tier; if **`overnight_results.n_train < 300000`**, reports **`status=pending_production`** (smoke pass); if **≥ 300k**, validates BPC within **0.05** of d17 hybrid **2.873** pin; profile **`bench/config/d27_production_lock_profile.json`**. CTest **`native_d27_production_lock_smoke`**.
- **Production validator:** **`scripts/validate_baseline_lock.ps1 -Production`** and **`baseline_lock_validate --production`** — when **`overnight_results.n_train >= 300000`**, require **`status=production`** or **`completed`** and BPC within **0.05** of pin.
- **CI:** blocking gate **104 CTests** (+1 d27 smoke). Full 300k production overnight remains maintainer workflow via **`run_production_overnight.ps1`**.

### Phase 14 — overnight completion gate (v2.3.14) — shipped

- **Status validator fix:** **`scripts/validate_baseline_lock.ps1`** and **`baseline_lock_validate`** accept **`medium_smoke`** and **`production`** (fixes lock validation after medium/production overnight runs).
- **Cell sweep artifact path:** default overnight output **`bench/results/cell_sweep`** via **`bench_paths::results_dir()`**; wired through **`cypha_baseline_lock --output-dir`**, **`update_baseline_lock.ps1`**, and **`run_overnight_all.ps1`**.
- **Bench d28:** unified overnight completion validation — **`run_d28_overnight_complete_validation`** checks **`overnight_results`**, **`rpsm_results`**, and **`cell_sweep_results`** share **`n_train`** / **`n_eval`**; **`pending_overnight_complete`** when **< 300k** (smoke pass); full gate when **≥ 300k**; profile **`bench/config/d28_overnight_complete_profile.json`**. CTest **`native_d28_overnight_complete_smoke`**.
- **Post-overnight finalize:** **`scripts/finalize_production_overnight.ps1`** — **`validate_baseline_lock.ps1 -Production`**, d27 + d28 bench domains, lock section summary; chained from **`run_production_overnight.ps1`** on success.
- **CI:** blocking gate **106 CTests** (+2 Phase 14 smokes). Full 300k production overnight remains maintainer workflow.

### Phase 15 — release readiness gate (v2.3.15) — shipped

- **Bench d29:** release readiness validation — schema + production tier (d27) + overnight-complete (d28) + release script presence (`scripts/finalize_production_overnight.ps1`, `scripts/run_production_overnight.ps1`, `bench/results/.gitkeep`); optional **`baseline_lock_validate --production`**; profile **`bench/config/d29_release_readiness_profile.json`**. CTest **`native_d29_release_readiness_smoke`**.
- **Local validate env vars:** **`CYPHA_VALIDATE_OVERNIGHT_COMPLETE=1`** on **`cypha_native_validate_all.ps1`** runs d28 after baseline lock validate; **`CYPHA_VALIDATE_RELEASE_READINESS=1`** runs d29; **`CYPHA_VALIDATE_PRODUCTION=1`** runs production lock validate; **`CYPHA_STRICT_TEST_COUNT=1`** fails when native_ count ≠ **107**.
- **Lock commit helper:** **`scripts/commit_production_lock.ps1`** — chains **`finalize_production_overnight.ps1`**, then stage/commit updated **`bench/BASELINE_LOCK.json`** after 300k overnight (`-DryRun` / `-Force`; maintainer-only; never pushes).
- **Production overnight watcher:** **`scripts/watch_production_overnight.ps1`** — log byte growth, last line, process PIDs, lock section summary; stall warn after 30m without log growth.
- **CI:** blocking gate **107 CTests** (+1 d29 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

### Phase 16 — artifact path hygiene gate (v2.3.16) — shipped

- **Bench d30:** artifact path hygiene validation — legacy repo-root **`results/`** path detection in **`cell_sweep_results.artifact_path`**, verifies **`bench/results/.gitkeep`**; profile **`bench/config/d30_artifact_hygiene_profile.json`**; report **`bench/report/tables/d30_artifact_hygiene_validation.json`**. CTest **`native_d30_artifact_hygiene_smoke`**.
- **Legacy migration:** **`scripts/migrate_legacy_results.ps1`** — merge repo-root **`results/`** cell-sweep artifacts into **`bench/results/cell_sweep/`** (`-DryRun`, `-RemoveLegacy`).
- **Overnight progress logging:** stderr **`[cyphalm]`** / **`[cell_sweep]`** (full sweep only); **`run_d17_overnight.ps1`** tees to **`bench/results/overnight_d17_<timestamp>.log`**.
- **Local validate env var:** **`CYPHA_VALIDATE_ARTIFACT_HYGIENE=1`** on **`cypha_native_validate_all.ps1`** runs d30 when profile exists.
- **CI:** blocking gate **108 CTests** (+1 d30 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

### Phase 17 — post-overnight pipeline gate (v2.3.17) — shipped

- **Bench d31:** post-overnight pipeline validation — d27→d30 chain + pipeline script presence (`poll_and_finalize_overnight.ps1`, `finalize_production_overnight.ps1`, `commit_production_lock.ps1`, `migrate_legacy_results.ps1`); profile **`bench/config/d31_post_overnight_pipeline_profile.json`**; report **`bench/report/tables/d31_post_overnight_pipeline_validation.json`**. CTest **`native_d31_post_overnight_pipeline_smoke`**.
- **Poll + finalize:** **`scripts/poll_and_finalize_overnight.ps1`** — poll until overnight processes exit, then **`finalize_production_overnight.ps1`** + **`commit_production_lock.ps1`** (`-DryRun` preview by default; **`-Force`** to git commit; never pushes); **`watch_production_overnight.ps1`** hints this script when processes disappear.
- **Legacy cleanup:** **`scripts/cleanup_legacy_results.ps1`** — one-shot **`migrate_legacy_results.ps1`** + **`RemoveLegacy`**; **`migrate_legacy_results.ps1 -ArchiveLegacy`** archives repo-root **`results/`** to **`bench/results/legacy_archive_<timestamp>/`**.
- **Local validate env var:** **`CYPHA_VALIDATE_POST_OVERNIGHT_PIPELINE=1`** on **`cypha_native_validate_all.ps1`** runs d31 when profile exists.
- **CI:** blocking gate **109 CTests** (+1 d31 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

### Phase 18 — production complete gate (v2.3.18) — shipped

- **Bench d32:** production complete validation — full production tier gate when **`overnight_results.n_train >= 300000`**; profile **`bench/config/d32_production_complete_profile.json`**; report **`bench/report/tables/d32_production_complete_validation.json`**. CTest **`native_d32_production_complete_smoke`**.
- **Unified validator:** **`scripts/validate_production_complete.ps1`** — chains **`validate_baseline_lock.ps1 -Production`**, **`finalize_production_overnight.ps1`**, **`cypha_bench_run --domain-tag d31`** + d30; **`-AllowPending`** for smoke when lock below 300k.
- **Background poll:** **`scripts/start_poll_finalize_background.ps1`** — detached **`poll_and_finalize_overnight.ps1`** after manual production overnight start.
- **Cell sweep sidecar:** **`overnight_progress.log`** written beside cell sweep output during full overnight sweep.
- **Release publish:** **`scripts/publish_release.ps1`** — **`gh auth`** preflight before **`gh release create`**.
- **Local validate env var:** **`CYPHA_VALIDATE_PRODUCTION_COMPLETE=1`** on **`cypha_native_validate_all.ps1`** runs d32 when profile exists.
- **CI:** blocking gate **110 CTests** (+1 d32 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

### Phase 19 — release publish gate (v2.3.19) — shipped

- **Bench d33:** release publish validation — publish script presence (`publish_release.ps1`, `create_release_notes.ps1`, `validate_production_complete.ps1`, `commit_production_lock.ps1`) + production/overnight-complete tiers + **`gh auth`** preflight metadata; profile **`bench/config/d33_release_publish_profile.json`**; report **`bench/report/tables/d33_release_publish_validation.json`**. CTest **`native_d33_release_publish_smoke`**.
- **Release publish smoke:** **`scripts/verify_release_publish.ps1`** — chains **`validate_production_complete.ps1`**, **`cypha_bench_run --domain-tag d33`**, **`publish_release.ps1 -DryRun`** (no `gh` call).
- **Poll BuildDir auto-detect:** **`poll_and_finalize_overnight.ps1`** and **`start_poll_finalize_background.ps1`** detect BuildDir from running **`run_production_overnight.ps1`** when default **`native/build`**.
- **Local validate env var:** **`CYPHA_VALIDATE_RELEASE_PUBLISH=1`** on **`cypha_native_validate_all.ps1`** runs d33 when profile exists.
- **CI:** blocking gate **111 CTests** (+1 d33 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

### Phase 20 — repo smoke hygiene gate (v2.3.20) — shipped

- **Bench d34:** repo smoke hygiene validation — repo-root **`d*_smoke.json`** leak detection + **`.gitignore`** patterns; profile **`bench/config/d34_repo_smoke_hygiene_profile.json`**; report **`bench/report/tables/d34_repo_smoke_hygiene_validation.json`**. CTest **`native_d34_repo_smoke_hygiene_smoke`**.
- **Repo smoke cleanup:** **`scripts/cleanup_repo_smoke_artifacts.ps1`** — remove repo-root smoke JSON spill files (`-DryRun` / `-Force`).
- **Poll heartbeat:** **`poll_and_finalize_overnight.ps1`** — per-cycle **HEARTBEAT** line (timestamp, process count, lock `n_train`).
- **Local validate env var:** **`CYPHA_VALIDATE_REPO_SMOKE_HYGIENE=1`** on **`cypha_native_validate_all.ps1`** runs d34 when profile exists.
- **CI:** blocking gate **112 CTests** (+1 d34 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

### Phase 21 — lock commit pipeline gate (v2.3.21) — shipped

- **Bench d35:** lock commit pipeline validation — post-overnight commit toolchain script presence (`commit_production_lock.ps1`, `finalize_production_overnight.ps1`, `poll_and_finalize_overnight.ps1`, `validate_production_complete.ps1`); profile **`bench/config/d35_lock_commit_pipeline_profile.json`**; report **`bench/report/tables/d35_lock_commit_pipeline_validation.json`**. CTest **`native_d35_lock_commit_pipeline_smoke`**.
- **Production pipeline smoke:** **`scripts/verify_production_pipeline.ps1`** — chains production complete + release publish + repo smoke cleanup preview + optional d35 (`-AllowPending` for smoke when lock below 300k).
- **Overnight watch + poll:** **`watch_production_overnight.ps1`** — **`done/28`** cell sweep variant progress; poll dedupe + heartbeat log fix in **`poll_and_finalize_overnight.ps1`** / **`start_poll_finalize_background.ps1`**.
- **Local validate env var:** **`CYPHA_VALIDATE_LOCK_COMMIT_PIPELINE=1`** on **`cypha_native_validate_all.ps1`** runs d35 when profile exists.
- **CI:** blocking gate **113 CTests** (+1 d35 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

### Phase 22 — production pipeline E2E gate (v2.3.22) — shipped

- **Bench d36:** production pipeline E2E validation — full maintainer overnight→publish toolchain script presence + **d27–d35** bench profiles; profile **`bench/config/d36_pipeline_e2e_profile.json`**; report **`bench/report/tables/d36_pipeline_e2e_validation.json`**. CTest **`native_d36_pipeline_e2e_smoke`**.
- **Post-overnight wrapper:** **`scripts/run_post_overnight.ps1`** — poll/finalize/commit + **`verify_production_pipeline.ps1`** (`-SkipPoll`, `-AllowPending`).
- **Local validate env var:** **`CYPHA_VALIDATE_PIPELINE_E2E=1`** on **`cypha_native_validate_all.ps1`** runs d36 when profile exists.
- **CI:** blocking gate **114 CTests** (+1 d36 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

### Phase 23 — overnight lock refresh gate (v2.3.23) — shipped

- **Bench d37:** overnight lock refresh validation — post-overnight baseline lock update toolchain (`update_baseline_lock.ps1`, migrate scripts, `finalize_production_overnight.ps1`); profile **`bench/config/d37_lock_refresh_profile.json`**; report **`bench/report/tables/d37_lock_refresh_validation.json`**. CTest **`native_d37_lock_refresh_smoke`**.
- **In-flight migrate:** **`scripts/migrate_inflight_overnight_artifacts.ps1`** — merge repo-root **`results/`** spill into **`bench/results/cell_sweep/`**; chained from **`run_post_overnight.ps1`** (`-DryRun` preview unless **`-SkipMigrate`**).
- **Local validate env var:** **`CYPHA_VALIDATE_LOCK_REFRESH=1`** on **`cypha_native_validate_all.ps1`** runs d37 when profile exists.
- **Offline release notes:** **`publish_release.ps1 -NotesPath`** for offline **`gh release create`** workflow.
- **CI:** blocking gate **115 CTests** (+1 d37 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

### Phase 24 — overnight completion certificate gate (v2.3.24) — prep

- **Bench d38:** production overnight completion certificate — full 300k cross-section + 28-variant cell sweep + production/overnight-complete gates; profile **`bench/config/d38_overnight_certificate_profile.json`**; report **`bench/report/tables/d38_overnight_certificate_validation.json`**. CTest **`native_d38_overnight_certificate_smoke`** *(when merged)*.
- **Poll auto-commit:** **`scripts/poll_and_finalize_overnight.ps1 -AutoCommit`** — post-finalize **`commit_production_lock.ps1 -Force`** when **`overnight_results.n_train >= 300000`**; **`start_poll_finalize_background.ps1 -AutoCommit`** passthrough.
- **Variant stall detector:** **`scripts/watch_production_overnight.ps1`** — **`-StallMinutes`**, **`-LogFile`** append for **STALL_WARNING** when variant count unchanged while overnight running.
- **Local validate env var:** **`CYPHA_VALIDATE_OVERNIGHT_CERTIFICATE=1`** on **`cypha_native_validate_all.ps1`** runs d38 when profile exists.
- **CI:** blocking gate **115 CTests** today; **116** when d38 merges (+1 smoke). Full 300k production overnight **in progress** — maintainer workflow via **`run_production_overnight.ps1`**; **`gh auth login`** still required for GitHub Release publish.

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

## Possible upgrades

Planned engineering directions distilled from research specs. Full index: [`docs/research/upgrades/README.md`](research/upgrades/README.md). Roadmap: [`docs/FUTURE.md`](FUTURE.md) §10.

| Upgrade | Doc | Status | Success criterion |
|---------|-----|--------|-------------------|
| CyphaDIF matrix refactor (RPSM Option A) | [`RPSM_COMBINED_SPEC.md`](research/upgrades/RPSM_COMBINED_SPEC.md) | **Planned** | Parity green; batched LLR; faster infer |
| RPSM sequence layer (Option B) | [`RPSM_COMBINED_SPEC.md`](research/upgrades/RPSM_COMBINED_SPEC.md) + [`RPSM_IMPLEMENTATION.md`](research/upgrades/RPSM_IMPLEMENTATION.md) | **Planned** | D17 BPC < **2.873** (hybrid baseline) |
| Nyström / nonlinear boundary fixes | [`NONLINEAR_BOUNDARY.md`](research/upgrades/NONLINEAR_BOUNDARY.md) | **Partially shipped** | Native kernel LLR live; close ~18 pp sklearn XOR gap |
| Cell hypothesis testbench (28 variants) | [`CELL_HYPOTHESIS_TESTBENCH.md`](research/upgrades/CELL_HYPOTHESIS_TESTBENCH.md) | **Planned** | Beat char-LSTM / hybrid on D17 @ 300k |
| RPSM core fixes (spectral α, norm η, orthogonal init) | [`RPSM_IMPLEMENTATION.md`](research/upgrades/RPSM_IMPLEMENTATION.md) | **Planned** | Forgetting ratio < 0.01; α ∈ [0.3, 0.6] |

**Execution order (RPSM track):** Option A → kernel LLR into A (tuning) → Option B → global memory → D17 benchmark.

---

## Current priorities (ranked by evidence)

### Priority 1 — Kernel LLR (Nyström RBF)

**Evidence:** 32.3pp XOR gap. This is a hard LLR-linearity ceiling that affects all nonlinear domains: XOR, Feynman equations (R²=-0.01), and sinusoidal regression.

**What to do:**
1. ~~Implement `KernelMemory` reservoir + Nyström sketch~~ — **done** (native C++, median-γ whitening).
2. Benchmark on XOR suite and Feynman D14 — `cypha_bench_run --domain-tag d03_xor`; `xor_kernel_bench` CTest smoke.
3. ~~Wire native kernel train in `memory_train.cpp`~~ — **done**; online XOR bench via `xor_kernel_bench`.
4. **Tuning:** diverse landmarks, M=512 profile, close sklearn RBF gap — see [`NONLINEAR_BOUNDARY.md`](research/upgrades/NONLINEAR_BOUNDARY.md).

**Current state:** Nyström whitening native-only; M=256 default; XOR **+9–10 pp** vs linear; sklearn RBF ceiling **~79%** — **~18 pp** gap remains.

**Update (2026-07-11) — reproduced baseline, M-sweep, RFF auto-gamma basis added:**

- **Reproduced baseline** (`xor_kernel_bench --kernel-feature-mode latent --kernel-m 256 --gamma-scale 1.0`, 3 seeds × 8 passes, generalizable latent-only mode — not the XOR-specific `xor_pair` hack): linear **51.2%** → Nyström kernel **62.07%** (**+10.83 pp**, matches documented +9–10pp) — gap to sklearn ceiling (~79%) **≈16.9 pp**, consistent with the documented ~18pp within seed noise.
- **`kernel_m` sweep (Nyström, same latent setup):** M=256 → 62.07%; M=384 → 62.53% (+0.47pp over M=256). **M=512 aborted** — `recompute_nystrom()` runs the whitening Cholesky/eigh solve on *every* training step (not just landmark fill-up), so per-step cost is `O(M^3)`; M=512 did not finish in a reasonable wall-clock budget (>15 min) and M=768 was not attempted. **Conclusion: Nyström landmark count is already near its practical (not just statistical) ceiling at M=256–384** — diminishing accuracy returns *and* superlinear cost blowup both push against raising M further on this code path.
- **Implemented Fix 2 (RFF kernel LLR) with auto-gamma default**, closing the priority-2 item below at the same time: added `KernelMemory::make_rff()` (fixed random-Fourier-Features basis, `phi(x) = sqrt(2/M)*cos(Wx+b)`, no landmark reservoir/no recompute — `O(M·d)` per step instead of `O(M^3)`) and `KernelMemory::auto_gamma_median_heuristic()` (median pairwise-distance heuristic, the same rule the Nyström path already uses for its landmarks, applied here to a calibration batch up front). Wired into `xor_kernel_bench` via `--kernel-basis rff [--rff-dim N] [--rff-gamma-scale S] [--rff-fixed-gamma G]`; auto-gamma is the default, `--rff-fixed-gamma` is opt-in for comparison. No changes to existing Nyström call sites, defaults, or the shipped `xor_pair`/`d03_xor` bench domain config.
- **RFF auto-gamma sweep (latent mode, same task/split, 3 seeds × 8 passes):**

  | `rff_dim` | kernel acc | Δ vs linear | gap to sklearn (~79%) |
  |---|---|---|---|
  | 128 | 61.7% | +10.5pp | ~17.3pp |
  | 256 | 65.0% | +13.8pp | ~14.0pp |
  | 512 | 68.0% | +16.8pp | ~11.0pp |
  | 768 | 70.3% | +19.0pp | ~8.7pp |
  | 1024 | 71.1% | +19.9pp | ~7.9pp |
  | 2048 | 74.2% | +23.0pp | ~4.8pp |
  | 3072 | 75.8% | +24.5pp | ~3.2pp |
  | **4096** | **76.3%** | **+25.1pp** | **~2.7pp** (best found) |

  Returns diminish sharply past ~3072–4096 (+0.56pp for the last +1024 dims). RFF is also markedly cheaper than Nyström at comparable/larger effective dimension (no cubic recompute), so pushing `rff_dim` well past `kernel_m`'s practical ceiling is tractable.
- **Auto-gamma validated against fixed-gamma** at `rff_dim=512`: fixed γ=1.0 → 49.4% (worse than the 51.2% linear baseline — a badly-chosen fixed γ can make the kernel path actively harmful); γ=0.2 → 55.6%; γ=0.01 → 64.3%; **auto-gamma (γ≈0.052, computed) → 68.0%** — beats every fixed γ tried, confirming the median-heuristic default is the right call for this task.
- **Best configuration found:** RFF, latent features, auto-gamma, `rff_dim=4096` → **76.3% accuracy, ~2.7pp gap to the sklearn RBF SVM ceiling** — versus the ~18pp gap at the previous M=256 Nyström default. CTests: all 10 kernel/xor-related CTests (`native_kernel_llr`, `native_xor_kernel_bench_smoke`, `native_kernel_snapshot_roundtrip`, `native_kernel_cypha_roundtrip`, `native_d44_kernel_nystrom_cyphalm_smoke`, `native_d59/d67/d68/d71_*_grid_joint_smoke`, `native_kernel_llm_h04_smoke`) pass unchanged — purely additive (`KernelMemory::make_rff`/`auto_gamma_median_heuristic` + new opt-in `xor_kernel_bench` CLI flags), no change to shipped Nyström defaults or the `d03_xor` bench domain.
- **Not yet done:** wiring `--kernel-basis rff` into the `d03_xor` bench domain / `bench_domains.cpp` as a tracked experiment (currently CLI-only in `xor_kernel_bench`); re-running Feynman (D14) and sinusoidal regression with the RFF basis to see if the same gap-closing generalizes beyond XOR; promoting `rff_dim=2048–4096` auto-gamma to a new default profile once validated on those other nonlinear domains.

**Update (2026-07-11, continued) — RFF wired into the real `d03_xor` bench domain; Feynman (D14) generalization checked and deferred:**

- **`d03_xor` bench domain wiring (done):** `run_d03_xor()` in `native/src/bench/bench_domains.cpp` already shells out to `xor_kernel_bench`; added a D03-only opt-in env-gate — `CYPHA_D03_KERNEL_BASIS=rff` (+ `CYPHA_D03_KERNEL_FEATURE_MODE`, `CYPHA_D03_RFF_DIM`, `CYPHA_D03_RFF_GAMMA_SCALE`) — that appends `--kernel-basis rff [...]` to that same subprocess call, following the identical env-var opt-in convention `CYPHA_D03_VIEW_SCHEDULE` already established for D03 experiments (not a second config system). Default (flag unset) is byte-identical to the pre-existing Nyström/`xor_pair` call — verified via rerun (`native_d03_xor_kernel_basis_default_smoke` CTest + manual rerun, see below). New CTests: `native_d03_xor_kernel_basis_default_smoke`, `native_d03_xor_kernel_basis_rff_smoke`.
- **Before/after in the real bench domain** (`cypha_bench_run --domain-tag d03_xor`, table at `bench/report/tables/d03_xor_kernel.json`):

  | Config | Mode | Linear acc | Kernel acc | Δpp | Wall time |
  |---|---|---|---|---|---|
  | Default (unchanged) | `xor_pair` + Nyström M=512 | 51.4% | **98.3%** | +46.9pp | ~114s (1 seed × 2 passes, FAST) |
  | Opt-in, generalizable | `latent` + Nyström M=512 | 51.4% | 54.7% | +3.3pp | ~117s (1 seed × 2 passes, FAST — full 3×8 run impractical, same `O(M^3)`/step ceiling as Priority 1) |
  | Opt-in, generalizable | `latent` + **RFF `rff_dim=4096`** | 51.2% | **76.3%** | +25.1pp | **27s total (3 seeds × 8 passes, full — no FAST)** |

  The `latent`+RFF row **exactly reproduces the standalone tool's validated 76.3% / ~2.7pp-sklearn-gap figure**, live, inside the real wired bench domain — confirming the wiring is correct, not just plumbing that compiles. The shipped default (`xor_pair` hand-engineered features) already exceeds the sklearn ceiling regardless of kernel basis, so the RFF-vs-Nyström gap-closing story only shows up in the generalizable `latent` mode, which is opt-in (not the production default).
- **Feynman (D14) generalization — deferred, documented reasoning (not forced):** grepped `native/src` + `docs/` for "feynman"/"sinusoid" — D14 (`run_d14`, `14A_feynman_all_equations`, per-equation Ridge `ridge_rmse` comparison) is the only live nonlinear-regression domain with a baseline-comparison framing; "sinusoidal regression" exists only in doc prose (this file's regression table), not in any current native bench domain. D14's regressor (`OnlineRegressor`/`online_reg_train_step`/`online_reg_predict`) routes through a plain linear `batch_llr_from_x` discriminant over discrete expert clusters, then a per-expert mean/variance mixture head — it never touches `KernelMemory`/`CyphaInferOptions.kernel_mem`, unlike the classification kernel-LLR path Fix 1/2 target. Wiring RFF there needs a new kernelized expert-routing discriminant on **both** the train step (`TrainStepExtras.kernel_mem`, currently `nullptr` from `online_reg_train_step`) and prediction (`online_reg_predict`'s direct `batch_llr_from_x` call has no kernel option at all) — a two-sided subsystem change, correctly out of scope for this pass. **Re-ran the existing D14 baseline instead** (full mode, `n_train=1600`, current HEAD): mean **R²=0.444** (per-equation range 0.03–0.79), **CyphaDIF beats the per-equation Ridge RMSE baseline on all 20 Feynman equations** — materially better than the stale `mean_r2=-0.010` row below (dated 2026-05-31, pre-dating many phases of general model fixes since). Regression table row updated to reflect this re-run.

**Update (2026-07-11, dedicated pass) — D14 kernelized expert-routing discriminant implemented and measured; result is a clean negative.**

Followed through on the deferred item above: built the two-sided kernelized expert-routing discriminant for D14 (`OnlineRegressor`'s `pick_dif_regressor_expert` train-time routing decision + `online_reg_predict`'s mixture-softmax weights, both in `native/src/bench/bench_domains.cpp`), env-gated exactly like `CYPHA_D03_KERNEL_BASIS`/`CYPHA_D03_RFF_DIM`/`CYPHA_D03_RFF_GAMMA_SCALE`.

- **What's kernelized vs. what isn't:** D14's final scalar prediction is (and remains) a linear per-expert running mean/variance mixture (`regression::expert_target_ema_step` + `regression::predict_mixture_scalar`) — that head was never the target. Only the *discriminant that decides which expert a sample routes to* (both at train time and at predict time, for the softmax mixture weights) is now optionally kernelized via a new `KernelMemory::make_rff` + `auto_gamma_median_heuristic` instance attached to `OnlineRegressor`, calibrated on up to 256 latent (`h`) samples encoded before training starts (same recipe as D03's `latent` mode). `CYPHA_D14_KERNEL_BASIS=rff` (+ `CYPHA_D14_RFF_DIM`, `CYPHA_D14_RFF_GAMMA_SCALE`, `CYPHA_D14_KERNEL_BLEND`, `CYPHA_D14_KERNEL_LR_SCALE`) turns it on for the 14A equations loop only — 14B extrapolation and 14C noise-curve stay linear-only regardless of the env gate, since they don't carry the Ridge-comparison framing this pass targets. Unset reproduces pre-existing 14A/14B/14C output **byte-for-byte** — verified via rerun before/after with no env vars set (identical `mean_r2`/`mean_rmse`/per-equation `rmse`/`r2`/`mae`/`mean_epistemic_var` to full double precision; only the report's own `timestamp` field differs).
- **Implementation gotcha worth recording:** `score_matrix_use_field`'s own `kernel_mem`/`use_kernel_llr` parameters are silently inert whenever `CYPHA_USE_RPSM_LLR` is unset (the documented default) — the function early-returns via `rpsm_score_matrix_batched()` *before* reaching its kernel-blend branch. `batch_llr_from_x` (what D14's routing/prediction previously called) goes through that same function, so a first wiring attempt (blending inside `score_matrix_use_field`'s own args) silently no-opped and reproduced the exact linear baseline even with the kernel path "enabled" and `kernel_blend=1.0` (pure kernel, no linear at all). Fixed by blending manually in a small D14-local helper (`kernel_blend_llr`) that calls `score_matrix_use_field` for the linear/RPSM score and combines it with `KernelMemory::score_all()` itself, independent of the RPSM env toggle. Left `score_matrix_use_field`'s own kernel args untouched (out of scope here, used by other domains/tests) — recorded in case a future pass revisits that function's kernel-blend branch.
- **Result — RFF kernelized routing makes D14 worse, monotonically with more kernel influence, at every setting tried** (`n_train=1600`, `kBenchSeed`, same split as the 0.444 baseline):

  | `rff_dim` | `kernel_blend` | mean R² | Δ vs baseline (0.444) |
  |---|---|---|---|
  | — (baseline, linear-only) | — | **0.4444** | — |
  | 128 | 0.1 | 0.4330 | −0.011 |
  | 512 | 0.1 | 0.3997 | −0.045 |
  | 512 | 0.25 | 0.3405 | −0.104 |
  | 128 | 0.5 | 0.2850 | −0.159 |
  | 512 | 0.5 | 0.2633 | −0.181 |
  | 4096 | 0.5 | 0.1314 | −0.313 |
  | 2048 | 0.5 | 0.1263 | −0.318 |

  Degradation is monotonic in both knobs — smaller `kernel_blend` and smaller `rff_dim` both move results back toward the linear baseline, as expected at the `blend→0` limit — but **no configuration tried matches, let alone beats, the 0.444 linear-only baseline**, and it's uniform: every one of the 20 equations individually loses R² under the kernel path (e.g. `wave_speed` 0.79→0.29, `kinetic_energy`/`hooke`/`capacitor_energy` 0.76→0.17 at `rff_dim=4096, blend=0.5`), not just the mean.
- **Why this is a plausible, not surprising, negative result:** unlike D03/XOR, D14's "classes" being routed between are `K≈10` arbitrary expert-cluster IDs with no inherent nonlinear boundary structure to exploit — they're an implementation detail of the mixture-of-experts regressor, not a real nonlinear decision problem. Calibrating a high-dimensional RFF basis via a single median-heuristic gamma pass over 256 latent samples from a freshly-initialized (effectively random, pre-contrastive-training) encoder, then using it to override/blend a discriminant that only needs to pick among ~10 mostly-arbitrary buckets, plausibly just injects routing noise rather than resolving any real nonlinearity — consistent with the observed monotonic-with-kernel-influence degradation (more kernel signal = more noise = worse) rather than a saturating improvement curve like D03's RFF-dim sweep showed.
- **Tests:** `ctest --test-dir native/build_kernel3 -R "d14|kernel|xor"` — 14/14 pass, including two new smoke tests (`native_d14_kernel_basis_default_smoke`, `native_d14_kernel_basis_rff_smoke`) added since D14 had no prior dedicated CTest coverage. Full native suite (171 tests) reruns clean too (169 passed, 2 skipped by design — CUDA/federated-TLS unavailable in this environment) — confirms no regressions in the shared `OnlineRegressor`/`make_online_regressor`/`pick_dif_regressor_expert`/`online_reg_predict` helpers used by D01/D05/D06/D11A/D14 alike.
- **Recommendation: do NOT turn this on by default.** Every configuration swept underperforms the existing linear-routing baseline, with no sign of a sweet spot — this isn't "needs more tuning", it's "the mechanism doesn't fit this domain". Leave `CYPHA_D14_KERNEL_BASIS` opt-in/off-by-default indefinitely unless a future pass finds a fundamentally different way to calibrate/apply the kernel (e.g. calibrating gamma after some encoder warm-up instead of on a fresh random encoder, or kernelizing something more structurally meaningful than the arbitrary expert-ID routing). Recorded here as an honest negative result, same spirit as the RPSM BPTT finding earlier this session — the D03/XOR kernel-LLR gap-closing effect does **not** generalize to D14's architecturally different expert-routing regressor.

### Priority 2 — Auto-gamma RFF

**Evidence:** Manual gamma tuning is the largest single source of improvement in the tuning report. Auto-gamma via cross-validation (`RFFEncoder.auto_gamma_cv`) exists but is not default.

**What to do:**
1. Make `auto_gamma_cv` the default encoder path for `RFFEncoder`.
2. Add to the parity fixture generators.
3. Re-run D08 (MNIST) and D14 (Feynman) with auto-gamma.

**Update (2026-07-11):** For the kernel-LLR/XOR track specifically, auto-gamma is now shipped and default for the new RFF kernel basis — see the Priority 1 update above (`KernelMemory::auto_gamma_median_heuristic`, median pairwise-distance heuristic, no CV grid search needed since XOR's calibration batch is cheap and the heuristic already beat every fixed γ tried). The separate `RFFEncoder.auto_gamma_cv` preprocessor path (D08/D14, item 1–3 above) is untouched — still opt-in, not covered by this pass.

### Priority 3 — CyphaLM: beat-bigram roadmap

**Status:** ✅ **Achieved @ 300k** via **hybrid GRIA+LSTM** — D17 **2.873 BPC** (−0.61 vs bigram, −0.11 vs char-LSTM bench). GRIA-only stack peaked @ **3.838** (+0.36 vs bigram).

**Evidence (300k, `hybrid_gria_lstm`):** D17 Phase 1c **2.873**; D04 Moby Dick bench **2.993** (learning-curve run; sweep **2.859**); bigram D17 **3.478** / D04 **3.633**. Cypha Tests **1A pass @ char shuffle** (+4.54 BPC @ 300k hybrid).

**Root causes (unchanged):**
- Most learning in GRIA; SSM/DIF under-trained at default single-pass online loop.
- Warm-started experts under-reported in D17B (`mean_alpha` low).
- Char-level LM: bigram/trigram are strong; 4/5-gram set an upper practical bound.

**Performance (2026-07-12):** D17 production training throughput is **252.2 chars/sec** single-threaded on MSVC (`windows-vs2026-release`), up from an initial ~96–126 chars/sec baseline, via allocator-churn fixes (thread-local scratch buffers, persistent gradient members) and a cache-friendly transpose-matvec loop interchange in `lstm_backward`/`bptt_ssm_update` — zero arithmetic change, D17 BPC pin confirmed bit-identical before/after. See [`PERFORMANCE_PROFILE_2026-07-12.md`](reports/PERFORMANCE_PROFILE_2026-07-12.md) for the full profiling breakdown.

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

**Status (2026-07-11): resolved as stale, no fix needed.** The 20%/17.5% chance-level figures no longer reproduce on current HEAD — D10A now measures **60.67% accuracy** (~3× chance). The root finding: D10's scored ECG classification (10A–10D) never touches `CellAISSM` at all — it routes through the `cypha_core` DIF expert-routing classifier + hand-engineered `TimeSeriesEncoder` features. The SSM is only touched by an optional, non-scored `10E_ssm_diagnose` probe. Full writeup, including the supplementary SSM diagnostic run against the doc's three original hypotheses (state-norm collapse, τ_fast/τ_slow suitability, routing-head connectivity — all ruled out or N/A): [`D10_ECG_SSM_DIAGNOSIS_2026-07-11.md`](reports/D10_ECG_SSM_DIAGNOSIS_2026-07-11.md).

**Remaining (optional, longer-horizon) future item — not a bug fix:** if D10 is ever pushed toward real-ECG-classification-quality accuracy (>90%), that needs real UCR ECG5000 data (`bench/data/ecg5000/`) and/or a richer feature front-end or purpose-built sequence model — a dedicated future pass, not an SSM tuning task.

---

## Forward research map

```
2026 Q3 — Priority 1: Kernel LLR tuning → close sklearn XOR gap
2026 Q3 — Priority 2: Auto-gamma RFF default → D08/D14 re-benchmark
2026 Q4 — RPSM Option A (matrix refactor) → Option B sequence layer — see upgrades/
2026 Q4 — Cell hypothesis testbench Tier 1–2
2026 Q4 — Multi-view online training Phase 2 D16/DIF
2026 Q4 — Priority 5: Continual learning investigation → EWC overlay
2026 Q3 — D10 re-eval: done (2026-07-11) — 60.67% accuracy, not an SSM issue; see [`D10_ECG_SSM_DIAGNOSIS_2026-07-11.md`](reports/D10_ECG_SSM_DIAGNOSIS_2026-07-11.md)
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
