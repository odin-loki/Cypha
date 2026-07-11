# Nonlinear boundary fix — CyphaDIF discriminant

**Author:** Odin Loch  
**Problem:** Linear LLR in latent space hard-ceils XOR at ~48.2% vs kernel SVM ~83.5% (**32.3 pp gap**). Same limit affects nonlinear regression (Feynman R²≈0) and weakens CyphaLM token routing.

**Status:** **Fix 1 (Nyström kernel LLR) SHIPPED for XOR** — native C++ in `native/src/kernel_memory.cpp`; xor_pair features (`build_xor_pair_features`) wired in train + infer (`kernel_features` / `kernel_x`). CTests `native_kernel_llr`, `native_xor_kernel_bench_smoke`. Latent-only mode ~59–71%; xor_pair default ~97% (exceeds sklearn RBF ~79%). See [`docs/FUTURE.md`](../../FUTURE.md) §0a.

---

## Root cause

Pipeline: `x → W_enc (linear) → h → LLR (linear in h) → y*`

Diagonal Gaussian LLR with tied covariances is linear in h. No nonlinear transform in the discriminant path — structural, not tuning.

---

## Fix taxonomy (ordered by cost / fit)

| # | Fix | Status | Notes |
|---|-----|--------|-------|
| **1** | **Nyström kernel LLR** | **Partially shipped** | Whitened landmarks, median-γ RBF; `use_kernel_llr` in score path |
| 2 | RFF kernel LLR | Partial (RFF encoder exists) | Extend to LLR path; preferred for streaming LM |
| 3 | Learned nonlinear encoder (2-layer MLP) | Planned | Diagnostic — if XOR closes, bottleneck is encoder only |
| 4 | GRIA-α kernel | Planned | Theoretically coherent; build on Fix 1 Nyström |
| 5 | Spectral mixture kernel | Long-term | σ_k ∝ 1/k harmonic structure |

---

## Fix 1 — Nyström kernel LLR (shipped baseline)

Nyström features:

```
φ(h) = K(h, Z) · K(Z,Z)^{−1/2}     Z = m landmarks
```

LLR in RKHS: `log p(φ(h) | N(μ_k^φ, diag v_k^φ))`

**Native (shipped):** Reservoir landmarks (M=256 default), Cholesky whitening, online softmax gradient on φ(h), blended into `score_matrix` when kernel enabled.

**Measured (2026-06-14):** Native linear **51.2%** → kernel **97.8%** with xor_pair features (+46.5 pp, 3 seeds, 8 passes, M=512, γ_scale=2, lr_scale=2). Latent-only kernel **~59–71%** (+9–20 pp); sklearn RBF ceiling ~79% on same splits — **xor_pair exceeds ceiling**.

**Hyperparameters (profile `bench/config/kernel_llr_profile.json`):** M=512, γ_scale=2.0, kernel_lr_scale=2.0, blend=1.0, feature mode `xor_pair`.

**Remaining work:** Auto-select xor_pair for XOR-like tasks; wire kernel_x into batched `score_matrix_use_field`; optional GRIA-α kernel (Fix 4).

---

## Fix 2 — RFF kernel LLR

Random Fourier Features approximate RBF without landmark eigendecomp. Faster init; lower accuracy per feature. **Use for CyphaLM streaming** (no runtime landmark refit). Auto-gamma RFF shipped in preprocessor ([`FUTURE.md`](../../FUTURE.md) §0b).

**Update (2026-07-11):** RFF kernel LLR basis (with auto-gamma via median heuristic as the default) now also implemented directly in `KernelMemory` (`make_rff` / `auto_gamma_median_heuristic` in `native/src/kernel_memory.cpp`) and exposed via `xor_kernel_bench --kernel-basis rff`, as a drop-in alternative to the Nyström landmark sketch for the XOR kernel-LLR benchmark — `O(M·d)` per step instead of `O(M^3)`, which let landmark/feature count scale well past the Nyström M=256–384 practical ceiling. Best found: `rff_dim=4096`, latent features, auto-gamma → 76.3% accuracy, ~2.7pp gap to the sklearn RBF ceiling (vs ~18pp at the Nyström M=256 default). Full sweep and fixed-vs-auto-gamma comparison in [`RESEARCH_STATUS.md`](../../RESEARCH_STATUS.md) Priority 1. Not yet wired into the `d03_xor` bench domain or re-validated on Feynman/sinusoidal regression.

**Update (2026-07-11, continued) — D03 bench-domain wiring + Feynman generalization check:**

- **RFF now genuinely wired into the real `d03_xor` bench domain**, not just the standalone `xor_kernel_bench` tool. `run_d03_xor()` in `native/src/bench/bench_domains.cpp` already shelled out to `xor_kernel_bench` as a subprocess (unchanged mechanism); added an opt-in env-var config (`CYPHA_D03_KERNEL_BASIS=rff` + `CYPHA_D03_KERNEL_FEATURE_MODE`, `CYPHA_D03_RFF_DIM`, `CYPHA_D03_RFF_GAMMA_SCALE`) that appends `--kernel-basis rff [...]` to that same subprocess call — the same D03-only env-gate opt-in convention already used by `CYPHA_D03_VIEW_SCHEDULE` (bench: D03 multi-view pilot) a few commits prior, rather than inventing a second config mechanism. Default (flag unset) reproduces the pre-existing Nyström/`xor_pair` call byte-for-byte — confirmed via rerun (see below).
- **Before/after, in the real bench domain** (`cypha_bench_run --domain-tag d03_xor`, `CYPHA_BENCH_FAST=1`, table written to `bench/report/tables/d03_xor_kernel.json`):
  - **Default (unchanged) path** — `xor_pair` features + Nyström M=512: linear **51.4%** → kernel **98.3%** (`+46.9pp`), byte-identical JSON shape to pre-existing behavior. `xor_pair` is XOR-specific hand-engineered features (`[x0,x1,x0·x1,x0²,x1²]`), so it already exceeds the sklearn RBF ceiling regardless of kernel basis — this path is not where the RFF gap-closing story applies.
  - **Latent (generalizable) mode, opt-in** — same real bench domain, `CYPHA_D03_KERNEL_FEATURE_MODE=latent`: Nyström M=512 → **54.7%** (FAST, 1 seed × 2 passes; full 3×8 run not completed — see practical-ceiling note in Priority 1 above, same `O(M^3)`-per-step cost applies here since it's basis-dependent, not feature-dependent). `CYPHA_D03_KERNEL_BASIS=rff CYPHA_D03_RFF_DIM=4096`, full 3 seeds × 8 passes (no `CYPHA_BENCH_FAST`): linear **51.2%** → RFF kernel **76.3%** (`0.763/0.743/0.784` per seed) — this **exactly reproduces the standalone tool's validated 76.3%/~2.7pp-gap figure**, live, inside the real wired bench domain, in **27 seconds total** (vs the Nyström latent path not finishing a full run in a reasonable time at M=512 — the RFF speedup is what makes the "live" comparison practical at all here).
  - **Conclusion:** the wiring is correct and reproduces the standalone-tool study faithfully; the gap-closing effect is real in the real bench domain, but only visible when using the generalizable `latent` feature mode — the shipped default (`xor_pair`) already saturates near/above the sklearn ceiling via feature engineering, so RFF vs Nyström is moot there.
- **Feynman (D14) generalization check — deferred, not a quick-pass fit.** Grepped `native/src` + `docs/` for "feynman"/"sinusoid": D14 (`run_d14()` in `bench_domains.cpp`, `14A_feynman_all_equations`) is the only real, currently-implemented nonlinear-regression domain with a baseline-comparison framing (per-equation Ridge `ridge_rmse`); "sinusoidal regression" only appears in doc prose (`RESEARCH_STATUS.md`'s regression table), not in any current native bench domain — it appears to be a stale/removed benchmark row, not a live domain to re-test. D14's regression path (`OnlineRegressor` / `online_reg_train_step` / `online_reg_predict`) does **not** use `KernelMemory`/`CyphaInferOptions.kernel_mem` at all — it routes to discrete "expert" clusters via a plain linear `batch_llr_from_x` discriminant, then predicts a continuous scalar via `regression::predict_mixture_scalar` over each expert's running mean/variance. This is architecturally disjoint from the classification kernel-LLR path (`infer_at_h` + `CyphaInferOptions.kernel_mem`) that Fix 1/2 target: wiring RFF here would mean building a *new* kernelized expert-routing discriminant for both the training step (`dif_train_step_vector`'s `TrainStepExtras.kernel_mem`, currently passed as `nullptr` from `online_reg_train_step`) **and** the prediction path (`online_reg_predict` calls `batch_llr_from_x` directly, no kernel option exists there at all) — a two-sided change to a different subsystem, not a cheap flag-add, and out of scope for this pass. **Deferred to its own dedicated pass.**
  - **D14 baseline re-run (current HEAD, `native/build_kernel2`)** for the record: full mode (`n_train=1600`), mean **R²=0.444**, mean RMSE dominated by a couple of large-constant equations (`coulombs_law`, `relativistic_KE`); per-equation R² ranges 0.03–0.79 and **CyphaDIF regression RMSE beats the per-equation Ridge baseline on all 20 equations** (e.g. `wave_speed` R²=0.79, `kinetic_energy`/`hooke`/`capacitor_energy` R²=0.76). This is a materially different (better) result than the stale `mean_r2=-0.010` figure in `RESEARCH_STATUS.md`'s regression table (dated 2026-05-31, pre-dating many intervening phases of general model fixes) — updated below. Confirms D14 is alive and beating its Ridge baseline today, but via the unrelated linear-expert-mixture path, not kernel LLR.

**Update (2026-07-11, dedicated pass) — D14 kernelized expert-routing discriminant built and measured: negative result, not recommended for default-on.**

Picked up the deferral above and built the two-sided kernelized expert-routing discriminant (`native/src/bench/bench_domains.cpp`: `OnlineRegressor` gains an optional `KernelMemory` used by both `pick_dif_regressor_expert` at train time and `online_reg_predict`'s mixture-softmax weights), env-gated the same way as `CYPHA_D03_KERNEL_BASIS` — `CYPHA_D14_KERNEL_BASIS=rff` (+ `CYPHA_D14_RFF_DIM`, `CYPHA_D14_RFF_GAMMA_SCALE`, `CYPHA_D14_KERNEL_BLEND`, `CYPHA_D14_KERNEL_LR_SCALE`), scoped to the 14A equations loop only. D14's final scalar head (linear per-expert mean/variance mixture) is unchanged — only the *routing* discriminant between the ~10 arbitrary expert clusters is kernelized, exactly as scoped in the deferral. Default (env unset) reproduces pre-existing 14A/14B/14C output byte-for-byte (verified via rerun; only the report's `timestamp` differs).

One implementation snag worth recording for future kernel-LLR work: `score_matrix_use_field` (what `batch_llr_from_x` — and therefore D14's original routing/prediction calls — goes through) early-returns via `rpsm_score_matrix_batched()` whenever `CYPHA_USE_RPSM_LLR` is unset (the documented default), *before* reaching its own `kernel_mem`/`use_kernel_llr` blend branch. A first wiring attempt that relied on that function's built-in kernel args silently no-opped — reproduced the exact linear baseline even with the kernel path "enabled" and `kernel_blend=1.0`. Fixed by blending manually against `KernelMemory::score_all()` in a small D14-local helper instead of relying on `score_matrix_use_field`'s own (RPSM-shadowed) kernel-blend branch.

**Result: the D03/XOR gap-closing effect does not generalize to D14.** Swept `rff_dim ∈ {128, 512, 2048, 4096}` × `kernel_blend ∈ {0.1, 0.25, 0.5}` (same seed/split as the 0.444 baseline) — every single configuration underperforms the linear-only baseline, monotonically worse with more kernel influence (higher blend, higher `rff_dim`):

| `rff_dim` | `kernel_blend` | mean R² |
|---|---|---|
| — (linear baseline) | — | **0.4444** |
| 128 | 0.1 | 0.4330 |
| 512 | 0.1 | 0.3997 |
| 512 | 0.25 | 0.3405 |
| 128 | 0.5 | 0.2850 |
| 512 | 0.5 | 0.2633 |
| 4096 | 0.5 | 0.1314 |
| 2048 | 0.5 | 0.1263 |

Every one of the 20 equations individually loses R² under the kernel path at the swept settings (e.g. `wave_speed` 0.79→0.29, `kinetic_energy`/`hooke`/`capacitor_energy` 0.76→0.17 at `rff_dim=4096, blend=0.5`) — this is a uniform degradation, not a mixed bag. Plausible explanation: unlike XOR's genuinely nonlinear 2-class boundary, D14's "classes" are `K≈10` arbitrary expert-cluster IDs — an implementation detail of the mixture-of-experts regressor with no real nonlinear structure for a kernel to exploit — and calibrating a high-dimensional RFF projection from a single median-heuristic pass over a freshly-initialized (pre-training) encoder's latent space, then using it to override a discriminant that only needs to pick among ~10 mostly-arbitrary buckets, looks like it just adds routing noise rather than resolving anything.

**Recommendation:** do not turn `CYPHA_D14_KERNEL_BASIS=rff` on by default — leave it opt-in indefinitely. This is a clean, honest negative result (same spirit as this session's RPSM BPTT finding): the D03/XOR RFF kernel-LLR mechanism does not transfer to D14's architecturally different expert-routing regressor. Tests: `ctest --test-dir native/build_kernel3 -R "d14|kernel|xor"` 14/14 pass (2 new smoke tests added, D14 had none before); full native suite 171 tests also clean (169 pass, 2 skipped by design).

---

## Diagnostic protocol

| Step | Test | Expected |
|------|------|----------|
| D1 | XOR baseline | ~48% CyphaDIF, ~83% kernel SVM |
| D2 | Nonlinear encoder alone | >80% if encoder is bottleneck |
| D3 | Nyström LLR + xor_pair | >80% (native: **~97%** xor_pair; ~60–71% latent-only) |
| D4 | Encoder + Nyström | Match/exceed SVM |
| D5 | Full diagnostic suite | All tasks ≥ SVM ceiling |

Domains: S1 linear, S3 XOR, R1 Iris, R3 digits — see [`DIAGNOSTIC_REPORT.md`](../../reports/DIAGNOSTIC_REPORT.md).

---

## Parity requirements

| CTest | Status |
|-------|--------|
| `native_kernel_llr` | **Shipped** |
| `native_kernel_snapshot_roundtrip` | **Shipped** |
| `native_kernel_cypha_roundtrip` | **Shipped** |
| `rff_kernel_parity` | Planned |
| `nonlinear_enc_parity` | Planned |
| `alpha_kernel_parity` | Planned |

Extend [`PORT_CONTRACT.md`](../../port/PORT_CONTRACT.md) when new fixes land in native.

---

## Connection to CyphaLM / RPSM

- Nyström in CyphaDIF router → nonlinear token boundaries  
- RFF variant for autoregressive streaming  
- Feeds **Option A** in [RPSM_COMBINED_SPEC.md](RPSM_COMBINED_SPEC.md) Step 2 before Option B sequence layer  

Zero catastrophic forgetting (per isolated model) must be verified after each fix — forgetting ratio probe on D16F-style saves.
