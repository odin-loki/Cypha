# Cypha — custom AI architecture: Differential Information Field Classifier (CyphaDIF)

[![CI](https://github.com/odin-loki/Cypha/actions/workflows/ci.yml/badge.svg)](https://github.com/odin-loki/Cypha/actions/workflows/ci.yml) · **[Releases](https://github.com/odin-loki/Cypha/releases)** (native Linux/Windows installers — latest **v2.2.8**)

> **A custom AI architecture invented from first principles by unifying four otherwise-disconnected threads of theory: AIXI / Solomonoff (minimum-description-length priors over class complexity, `‖Δk‖_F ≤ C`), information geometry (natural gradients on the Gaussian manifold, Cramér–Rao efficient), the Free Energy Principle / active inference (a shared world prior `θ₀` plus differential class offsets `Δk`), and the Information Bottleneck (contrastive encoder feedback via Fisher–Rao residuals). The core invariant is that every class `k` is represented as `θ₀ ⊕ Δk` — a shared world prior plus a small class-specific offset in natural parameter space — and classification is `y* = argmax_k [log p(h | θ₀ ⊕ Δk) + log p(k | context)]`. The architecture is domain-agnostic (any input is reduced to a latent vector by a pluggable `Encoder`), handles heavy-tailed inputs through a Generalised-Hyperbolic / NIG gate, detects out-of-distribution inputs natively, supports online corrections, and reports calibrated regression uncertainty. The engineering layer — CMake-built C++ native core, REST server, Qt desktop Studio, optional CUDA acceleration, SQLite-backed persistent state — is validated by **53 CTest** cases across named fixtures (`cypha_parity`, `memory_train_parity`, `quantile_dif_train_parity`, `mke_train_step_parity`, `regression_m4_parity`, …).**

---

## What this folder is

Cypha is a *fully custom* AI architecture — not a wrapper around a transformer, not a fork of an existing framework, not a deployment of a published paper. The object at the centre is **CyphaDIF**, the Differential Information Field Classifier, which derives a single learning rule from the intersection of four formal programmes:

- **AIXI / Solomonoff** contributes a minimum-description-length prior on class complexity. The class-specific offset `Δk` is regularised by `‖Δk‖_F ≤ C` so that simpler classes are preferred when the evidence is weak, with cold-start protection for the first `_MDL_COLD_START = 8` observations.
- **Information geometry** contributes the choice of update rule. Updates are natural gradients on the diagonal-Gaussian manifold, which is Cramér–Rao efficient — informally, the cheapest update that does not waste information.
- **Active inference / Free Energy Principle** contributes the structural decomposition. There is one shared **`WorldPrior`** `θ₀` (a diagonal Gaussian fitted online by Welford / EMA — Tier-3 "infinite" context that never forgets) and one **`ClassDifferential`** `Δk` per class, attracted toward observations of that class with MDL decay.
- **Information Bottleneck** contributes the encoder objective. The trainable projection `W_enc : raw_features → latent h` is updated by **contrastive Fisher–Rao residuals** — pull the latent toward the correct class's natural-parameter manifold, push it away from competitors, capped at Frobenius norm `8.0`.

These four threads collapse into one operational rule: every class has the same world prior plus a small offset, classification is an `argmax` over those offsets, and learning is a natural-gradient step that respects an MDL constraint and a Fisher–Rao encoder loss simultaneously.

The model is **domain-agnostic** — anything that can be turned into a latent vector by a pluggable `Encoder` works (numbers, text, spectrograms, behavioural telemetry, …). Out of the box it ships with `VectorEncoder` (passthrough), `RFFEncoder` (Random Fourier Features over an RBF kernel, `D = 256` features default), and `ConcatEncoder` (concatenate multiple encoders' outputs). The `MultiModalCyphaDIF` extends this with **per-encoder LLR fusion** so each modality contributes its own log-likelihood-ratio and they are summed at the decision step.

Five enhancement phases sit on top of the v1 foundation:

- **Phase 1 — Tiered context.** A 3-tier `TieredContextBuffer` (short / mid / long), an `NIGField` group at confidence threshold `τ = 0.99`, and field-confidence-weighted blending into the context prior.
- **Phase 2 — Generation overhaul.** Eight named generation modes — temperature-scaled, field-conditioned, latent-boundary interpolation (`α`-blended), adversarial (entropy-maximising), OOD sampling, MDL-ball constrained (Fisher–Rao radius), ancestral (`k ~ context`, `h ~ p(h | k)`), and Gaussian KDE sampled from the priority replay buffer.
- **Phase 3 — Active learning & anomaly detection.** `anomaly_score(x)` (gate value, high = anomalous), `active_query_score(x)` (entropy × `(1 − max p)` — boundary proximity), `drift_score()` (concept-drift signal from world-prior drift), and `infer_full(x)` returning a complete probabilistic breakdown.
- **Phase 4 — Priority replay.** Recency-and-surprise weighted buffer of capacity `10 000`, replay rate `0.30`, with KDE generation from stored latents.
- **Phase 5 — Sequence & multi-modal.** `predict_next(label)` for next-label distributions, `ConcatEncoder` for feature concatenation, `MultiModalCyphaDIF` for per-modality LLR fusion.

The folder is the **native C++ implementation** of this AI architecture. The harmonic-spectrum theoretical backbone (the `σ_k ∝ 1/k` and `α ≈ 0.85` SGD-narrative claims) lives separately at [`../Compression Algorithms/NMP_neural_compression_research_paper.md`](../Compression%20Algorithms/NMP_neural_compression_research_paper.md). Cypha is the *implementation leg*: one authoritative runtime in `native/`, validated by CTest parity fixtures.

---

## Quick start (native)

Build outside OneDrive on Windows (cloud sync locks object files). Full guide: [`docs/native/NATIVE_QUICKSTART.md`](docs/native/NATIVE_QUICKSTART.md).

```powershell
# Windows — configure + build
cmake -S native -B C:\Temp\cypha_build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build C:\Temp\cypha_build --parallel

# Validate (108 CTests)
ctest --test-dir C:\Temp\cypha_build -R native_ --output-on-failure

# Full production gate (rebuild + CTest + bench smoke + tune dry-run)
powershell -File scripts\cypha_native_validate_all.ps1
```

```bash
# Linux / WSL
cmake -S native -B /tmp/cypha_build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build /tmp/cypha_build --parallel
ctest --test-dir /tmp/cypha_build -R native_ --output-on-failure
bash scripts/ci_native_linux.sh
```

**Run services** (after build or release install):

```bash
# REST API (CyphaDIF + CyphaLM + Branch A)
cypha_rest --listen 127.0.0.1:8099 --cypha fixtures/reference.cypha

# Qt Studio shell (build with -DCYPHA_BUILD_QT=ON)
cypha_qt_shell

# Benchmarks (d01–d17)
cypha_bench_run --list-domains
cypha_bench_run --from-domain 1
```

Prebuilt bundles: **[GitHub Releases](https://github.com/odin-loki/Cypha/releases)** → `packaging/install_release_windows.ps1` or `packaging/install_release_linux.sh`.

---

## 📑 Source documents

| File | Role |
|---|---|
| [`README.md`](README.md) | This file. |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history — milestones, bug fixes, benchmark deltas. |
| [`bessel_ratios.npz`](bessel_ratios.npz) | Pre-computed `K_n` Bessel ratios (`16 384` uniform points, `x ∈ [10⁻⁶, 120]`, max rel-err `< 5 × 10⁻³`) — replaces per-call `scipy.special.kv` in the GH-posterior hot path. |
| [`native/README.md`](native/README.md) | Native C++ core build & test guide — CTest harness, parity test inventory, SQLite amalgamation, CUDA smoke test. |
| [`docs/README.md`](docs/README.md) | Documentation hub — all sub-documents indexed by purpose. |
| [`docs/native/NATIVE_QUICKSTART.md`](docs/native/NATIVE_QUICKSTART.md) | One-page native install → validate → bench → tune → REST. |
| [`docs/port/PORT_CONTRACT.md`](docs/port/PORT_CONTRACT.md) | The parity contract — `.cypha` v3, REST shapes, bench §6. |
| [`docs/verify/VERIFICATION_STATUS.md`](docs/verify/VERIFICATION_STATUS.md) | Current CTest parity results across all fixtures. |
| [`docs/reports/DIAGNOSTIC_REPORT.md`](docs/reports/DIAGNOSTIC_REPORT.md) | 2026-05-30 full diagnostic: three root-cause bugs found, +23.5 pp on linearly-separable 2-class. |
| [`docs/reports/SOM_UPGRADE_REPORT.md`](docs/reports/SOM_UPGRADE_REPORT.md) | SOM/GNG/GRIA/Hebbian upgrade evaluation: all six upgrades benchmarked; default flags remain OFF. |
| [`native/`](native/) | C++ native core. CMake build. Milestones M1–M6 complete. |
| [`fixtures/`](fixtures/) | Committed parity assets — input vectors and expected outputs for CTest validation. |
| [`bench/`](bench/) | Native benchmark tree: config, data, report, artifacts. |
| [`packaging/`](packaging/) | Release install scripts; bundles built via `scripts/package_release_*.sh`. |

> **Note on file paths.** The repository's HRNA / NMP research paper is *not* inside `Cypha/`; it lives at [`../Compression Algorithms/NMP_neural_compression_research_paper.md`](../Compression%20Algorithms/NMP_neural_compression_research_paper.md). Cypha is the engineering implementation; that paper is the theoretical home.

---

## 🧠 The architecture

```
                        x  (raw input — vector, text, spectrum, …)
                        │
                        ▼
        ┌──────────────────────────────────┐
        │   Encoder (pluggable)            │   VectorEncoder │ RFFEncoder │ ConcatEncoder
        │     raw → feature vector f       │
        └──────────────────────────────────┘
                        │ f
                        ▼
        ┌──────────────────────────────────┐
        │   EncoderProjection W_enc        │   Fisher–Rao contrastive update,
        │     f → latent h                 │   ‖W‖_F capped at 8.0
        └──────────────────────────────────┘
                        │ h
                        ▼
   ┌────────────────────┴────────────────────┐
   │                                          │
   ▼                                          ▼
WorldPrior θ₀                       ClassDifferential Δk   (per-class)
diagonal Gaussian                   natural-parameter offset
Welford/EMA, Tier-3                 attracted toward h, MDL-decayed
"infinite context"                  ‖Δk‖_F ≤ C   (Solomonoff prior)
   │                                          │
   └────────────────────┬─────────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │   DIFMemory                      │   LLR matrix per class:
        │     score k = log p(h|θ₀ ⊕ Δk)   │     log p(h | N(μ_k, diag v_k))
        │     + GH gate (heavy-tailed)     │   Generalised-Hyperbolic /
        └──────────────────────────────────┘     NIG posterior, Bessel-ratio LUT
                        │ LLR_k
                        ▼
        ┌──────────────────────────────────┐
        │   TieredContextBuffer            │   short (window 32) / mid (EMA 0.98) / long
        │   + NIGField (τ = 0.99)          │   field-confidence-weighted prior
        │     → log p(k | context)         │
        └──────────────────────────────────┘
                        │
                        ▼
              y* = argmax_k [ LLR_k + log p(k|context) ]
              + confidence, anomaly_score, r_eff, OOD flag
```

Training is the same machinery in reverse:

1. The `WorldPrior` updates by EMA toward the new observation's latent `h`.
2. The matched class's `ClassDifferential` is attracted toward `h` (with MDL decay subtracted).
3. The `EncoderProjection` is updated by the Fisher–Rao contrastive gradient — pull `h` toward `(μ_k, v_k)`, push it away from the runner-up `(μ_j, v_j)`.
4. The replay buffer stores `(x, h, label)` weighted by surprise (high LLR-residual = high priority); a fraction `replay_ratio = 0.30` of subsequent steps come from the buffer.
5. Every `_ALIGN_EVERY = 500` steps, the encoder is realigned to the dominant `Δk` directions, ensuring the latent space remains expressive.

A separate **`CausalField`** runs alongside as a recurrent SGEMV update for sequential context, and the **regressor variants** (`DIFRegressor`, `RFFRegressor`, `TwoStageDIFRegressor`, `MKERegressor`) replace the LLR-argmax with a ridge-regression / RLS posterior so the same architecture handles regression with calibrated uncertainty.

---

## ⚙️ Reference defaults

These defaults come from a profiled medium-grid tuning programme, not from guesses.

### `CyphaDIF`

| Parameter | Default | Origin |
|---|---|---|
| `feat_dim` | `128` | profiled on OpenML 1464 + tuning grid |
| `field_dim` | `128` | matched to `feat_dim` for the no-injection fast path |
| `rff_D` (RFFEncoder) | `256` | RFF kernel approximation budget |
| `n_experts` (MKE) | `8` | mixture-of-experts head |
| `temperature` | `1.15` | classification optimum (regressor overrides to `1.05`) |
| `context_win` | `32` | profiled medium grid |
| `replay_ratio` | `0.30` | priority replay rate |
| `replay_capacity` | `10 000` | Phase-4 buffer (5× v1) |
| LR — world | `0.008` | classification-optimal |
| LR — delta | `0.05` | profiled medium grid |
| LR — encoder | `0.002` | Fisher–Rao stability |
| `mdl_lambda` | `0.001` | Solomonoff prior strength |
| `mdl_cold_start` | `8` | observations before MDL kicks in |
| `OOD_THRESHOLD` | `3.0` | anomaly_score gate |
| `OOD_SIGMA` | `15.0` | OOD distribution width |
| `align_every` | `500` | encoder-realignment cadence |

---

## 🧪 Parity test inventory (selected, from `native/README.md`)

| Test | What it verifies |
|---|---|
| `cypha_parity` | Top-level infer vs `fixtures/` |
| `memory_train_parity` | `DIFMemory` training step |
| `quantile_dif_train_parity` | Quantile-DIF training step |
| `mke_train_step_parity` | Mixture-of-experts training step |
| `regression_m4_parity` | M4 regression |
| `cuda_smoke` | CUDA path smoke test |

(Full inventory in [`native/README.md`](native/README.md). CI gate: **two blocking jobs** — **`build_and_test`** (Linux native + CTest) and **`mingw_cross`** (MinGW Windows PE cross-compile); Windows local gate via `scripts/cypha_native_validate_all.ps1`.)

---

## 📊 Empirical benchmark results (2026-05-30)

Full diagnostic run documented in [`docs/reports/DIAGNOSTIC_REPORT.md`](docs/reports/DIAGNOSTIC_REPORT.md). Three root-cause bugs were found and fixed; results below are **post-fix**:

| Task | CyphaDIF | SGD (online) | SVM ceiling | Notes |
|------|----------|--------------|-------------|-------|
| S1 — linearly-separable 2-class | **0.783** | 0.644 | 0.898 | RFF + 4 passes + deliberation disabled |
| S3 — XOR (nonlinear) | 0.482 | 0.498 | 0.825 | **Hard LLR-linearity limit** — kernel LLR required |
| R1 — Iris | 0.900 | 0.821 | 0.968 | Auto-RFF for dim≤30 |
| R2 — Wine | 0.969 | 0.964 | 0.987 | Near-saturated |
| R3 — Digits (10-class) | **0.922** | 0.900 | 0.982 | delta_lr=0.03 fix |
| R4 — Breast cancer | 0.957 | 0.950 | 0.983 | | |

**Key findings:**
- Catastrophic forgetting ratio: **0.000** (perfect retention; sufficient-statistics design).
- Label-noise robustness at 30% noise: **79.1%** accuracy (well above chance for 5-class).
- Convergence to 100% on well-separated 5-class Gaussian clusters: **step 50** (matches SGD online).
- XOR / nonlinear boundaries: linear LLR ~50%; **Nyström kernel LLR ~61%** (+10.6 pp, M=256) on S3 XOR. Diagnostic ceiling (~83% kernel SVM) still open — see [`docs/FUTURE.md`](docs/FUTURE.md) §0a.
- **D04 / D17:** CyphaLM benchmarks (Izaac → CellAI SSM → CyphaDIF → GRIA) via **`cypha_bench_run`** — held-out BPC, context-length curve, expert routing, save/restore parity, and sampling benchmarks.
- **CyphaLM REST:** native `cypha_rest` — `POST /generate` and `/generate/stream` (SSE) with per-token CyphaDIF routing.

---

## 🚧 Honest framing

- **The AI is bespoke.** CyphaDIF is not a fork, not a wrapper, not a tuning of an existing model. It is a from-first-principles architecture whose learning rule is derived from the intersection of four formal programmes (AIXI / Solomonoff, information geometry, FEP, IB).
- **The proof surface is parity correctness, not leaderboard ML accuracy.** No "we beat X on benchmark Y" claim. Instead: "the native runtime matches committed fixture goldens across this CTest matrix." Benchmark numbers (§ above) are honest measurements on standard sklearn datasets, not cherry-picked.
- **Nonlinear decision boundaries:** Linear LLR caps XOR near ~50%. **Nyström kernel LLR** (M=256 landmarks) reaches **~61%** (+10.6 pp) — implemented in C++; diagnostic ~83% kernel-SVM ceiling remains. See [`docs/FUTURE.md`](docs/FUTURE.md) §0a.
- **Theoretical backbone lives elsewhere.** The harmonic-spectrum / `σ_k ∝ 1/k` / `α ≈ 0.85` claims belong to [`../Compression Algorithms/NMP_neural_compression_research_paper.md`](../Compression%20Algorithms/NMP_neural_compression_research_paper.md), not to Cypha itself. Cypha is the implementation leg.
- **Optional CUDA.** The native core works without GPU; CUDA is a local build flag (`-DCYPHA_ENABLE_CUDA=ON`). CI does not compile CUDA — validate with **`native_cuda_smoke`** / **`native_score_batch`** locally when changing accel code (see [`docs/native/ACCEL_CUDA.md`](docs/native/ACCEL_CUDA.md)).
- **Future waves.** RFF auto-gamma, Qt UX polish, packaged binaries, multi-model REST, ONNX export — see [`docs/FUTURE.md`](docs/FUTURE.md) and [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md).

---

## 🎯 What this displaces

| Standard | Limitation | What Cypha offers |
|---|---|---|
| Python-only research repo | Slow at deploy time | Native C++ core with CTest-validated parity fixtures |
| C++-only production runtime | Hard to iterate on | Qt shell + REST for train/infer without a second stack |
| Off-the-shelf classifier (sklearn, XGBoost, …) | Black-box training rule | First-principles architecture; every constant is derivable |
| Transformer + softmax classifier | Calibration is an afterthought | GH gate gives natively-calibrated heavy-tail handling and OOD flag |
| Notebook + Flask script | No persistence | SQLite-backed state (amalgamated `3.47.2`) |
| Custom REST + Python | No GUI | `cypha_qt_shell` + `cypha_rest` |
| Standalone classifier | No regression path | `DIFRegressor` / `TwoStageDIFRegressor` reuse the same machinery |

---

[← Back to main README](../README.md)
