# Hidden-dimension scale-up plan (P1) — 2026-07-11

**Status:** Planning only. No implementation in this document's scope.
**Priority:** P1 in `docs/reports/DEV_PLAN_2026-07-11.md:126` ("Hidden-dim scale experiment (512 → 1024) — Paper IV: κ ≥ 0.91 achievable with no new theory, purely a scale-up").
**Relationship to the parallel P0 work:** independent. This plan does not touch `scripts/run_production_overnight.ps1`, `scripts/run_d17_overnight.ps1`, `native/tools/cypha_cell_hypothesis_sweep.cpp`, or `bench/BASELINE_LOCK.json`. All verification commands below use `native/build_scale` (a separate scratch build dir) or read-only inspection, to avoid colliding with the live 300k overnight run in `native/build_math`.

---

## 1. The claim being scoped

Paper IV states the residual κ gap between the current measured system (~0.86 κ at 300k with math-integration, per `docs/reports/DEV_PLAN_2026-07-11.md:84`) and human-median κ (0.91–0.92) is concentrated entirely on `D_eff` (representational scale), not on any of the other six profile axes, and that this closes by increasing hidden dimension to 1024, "no new theoretical breakthroughs required":

> "The remaining gap is purely representational scale (D_eff), not intelligence structure. Scaling the hidden dimension closes it." — `docs/research/intelligence_stats/cypha_self_correcting_paper4.md:681`
>
> "Estimated κ at 1024 hidden dim: 0.91–0.92 ... achievable with the current architecture, no new theoretical breakthroughs required." — `cypha_self_correcting_paper4.md:693-695`

Paper IV's own §7 roadmap allocates this to **Phase 5** (weeks 7–8): "Increase hidden dimension to 512, 1024. Measure D_eff as function of hidden dim. Verify κ increases toward 0.91 with scale. Target: confirm D_eff → 0.50 at 1024 hidden dim, κ ≥ 0.90." (`cypha_self_correcting_paper4.md:765-769`). Paper IV Phases 1–4 (τ-aware forget gate, D_eff/σ_branch regularizers, NIG output layer, epistemic feedback, profile monitoring) are the prerequisite architecture for Phase 5 and are largely already implemented under the "math-integration" preset — see §3 below for exactly what maps to what.

Paper V (`docs/research/intelligence_stats/soft_world_paper5.md`) gives a second, independent data point for the same lever: its "Soft World Cypha" system is estimated at κ 0.93 @ 256-dim, rising to **κ 0.95 with D_eff 0.58→0.62 at 1024-dim** (`soft_world_paper5.md:517-526`), i.e. Paper V also treats hidden-dim scaling as a positive but *secondary* lever on top of a richer data-acquisition/simulation layer that does not exist in the codebase yet (§6). Paper V is out of scope for implementation here (P3 in the roadmap) but its numbers corroborate that scaling hidden dim is directionally sound.

**This document's job:** ground that claim in the actual codebase — current value, what actually scales with it, what's hard-coded, what it will cost, what's measurable today vs. not, and a concrete phased plan — before committing to an expensive 1024-dim @ 300k production run.

---

## 2. Current hidden-dimension value (confirmed)

The "hidden dimension" in this codebase is `CyphaLMConfig::lstm_hidden` — the width of the char-level LSTM head that does the heavy lifting in the `hybrid_gria_lstm` cell (the D17 production context mode).

| Location | Value | Note |
|---|---|---|
| `native/include/cypha/cyphalm/cyphalm_config.hpp:78` | `int lstm_hidden = 128;` | struct default |
| `bench/config/profiles/cyphalm_d17_wikitext.json:25` | `"lstm_hidden": 128` | D17 production profile (loaded by `apply_bench_profile("d17", cfg)`, `native/src/cyphalm/cyphalm_config.cpp:242`) |
| `bench/config/profiles/cyphalm_d17_hybrid.json:25` | `"lstm_hidden": 128` | alias of the above (`:43` `"alias_of": "cyphalm_d17_wikitext.json"`) |

**Confirmed: the D17 300k production benchmark trains at hidden dim = 128**, not 256 as Paper IV's own "Full self-correcting Cypha, 256-dim" reference point assumes (`cypha_self_correcting_paper4.md:654,662`, cross-referenced in `DEV_PLAN_2026-07-11.md:92`). This is itself worth flagging: **the codebase is currently one scale step behind even Paper IV's stated starting point.** The practical scale-up path is therefore 128 → 256 → 512 → 1024, not 256 → 512 → 1024 as the paper implicitly assumes — see the phase table in §7.

`bench/config/profiles/cyphalm_llm.json` and `bench/config/profiles/cyphalm_d04_gutenberg.json` were also checked; d04 uses a different, smaller `lstm_hidden` (Gutenberg profile, not production-relevant here).

---

## 3. What Paper IV Phase 1–4 architecture already exists (context for Phase 5)

Before scaling, it's worth confirming how much of Paper IV's *own* prerequisite stack (Phases 1–4) is already shipped, since Phase 5 (hidden-dim scale) assumes it as a starting point. Mapping Paper IV §3.2/§7 mechanisms to this codebase's math-integration preset (`native/src/cyphalm/cyphalm_math_integration.cpp:101-139`, `apply_math_integration_preset`):

| Paper IV mechanism | Codebase equivalent | Status |
|---|---|---|
| τ-aware forget gate (`cypha_self_correcting_paper4.md:255-280`) | `cfg.use_tau_forget_gate`, `cfg.use_reu_forget_gate` (`cyphalm_math_integration.cpp:128-130`) | shipped |
| D_eff participation-ratio regularizer (`:208-226`) | `cfg.use_lstm_d_eff_hidden_nudge` (`:124`), `lstm_hidden_d_eff()` (`native/src/cyphalm/cyphalm_model.cpp:642-659`) | shipped (variance-proxy method; see §5 for the eigenvalue-method gap) |
| σ_branch spectral-norm regularizer (`:235-244`) | not found as a standalone regularizer; `sigma_branch` is measured (`measurers.hpp:38`) but no spectral-norm-toward-1.0 backprop term located in `cyphalm_model.cpp` | **partially shipped** — measured, not directly regularized as a Wh spectral norm; flagged for confirmation, not blocking Phase 5 |
| NIG output layer / r_eu (`:284-317`) | `use_reu_forget_gate`, DIF `NIGExpert` (`cyphalm_nig_expert.hpp`), `compute_epistemic_ratio` (`measurers.hpp:24`) | shipped |
| Lipschitz regularizer (`:319-344`) | `compute_lipschitz_sensitivity` measured (`measurers.hpp:34`); no explicit backprop regularizer term found | measured only |
| Online temperature calibration (`:348-372`) | `compute_calibration` (`measurers.hpp:21`); navigation-loss calibration lambda exists via `use_full_navigation_loss` | shipped as a loss term, not as a literal `OnlineCalibration` module |
| Epistemic feedback loop / `SelfCorrectingCypha` wrapper (§2, §4.2) | `native/include/cypha/intelligence/self_correcting_infer.hpp`, `epistemic_threshold.hpp` | present as headers — **not verified wired into the D17 train/eval loop in this pass; needs a follow-up read of `self_correcting_infer.hpp` + its .cpp before Phase 5 sign-off** |

None of these are blockers for Phase 5 specifically (hidden-dim scale is orthogonal to whether σ_branch/Lipschitz have literal regularizer terms vs. measured-only), but the "self-correcting" wrapper gap is worth a follow-up ticket since Paper IV's κ=0.89 estimate assumes it's active.

---

## 4. What actually scales with `lstm_hidden` in this implementation

Audited every `native/src/cyphalm/*.cpp` and `native/include/cypha/cyphalm/*.hpp` reference to `lstm_hidden`/`hidden` (41 source files, 48 headers). **Finding: the implementation is fully dynamically sized (`std::vector`-based) — there is no fixed-size array anywhere in `native/src/cyphalm` or `native/include/cypha/cyphalm` keyed to a specific hidden width** (checked for `std::array<double, 64|128|160|256|512>` and C-array literals `[64]/[128]/[160]/[256]`; zero matches). This is good news: there is no hard *compile-time* blocker to raising `lstm_hidden` to 512 or 1024.

### 4.1 Components that scale with hidden dim (quadratically)

`CharLSTMHead` (`native/include/cypha/cyphalm/char_lstm.hpp:49-109`, `native/src/cyphalm/char_lstm.cpp`) is the only component whose cost is a function of `hidden`:

| Tensor | Shape | Role |
|---|---|---|
| `Wx` | `(4·hidden) × hidden` | input→gates |
| `Wh` | `(4·hidden) × hidden` | recurrent→gates |
| `E` | `vocab × hidden` | embedding |
| `Wy` | `vocab × hidden` | output projection |

(`char_lstm.cpp:60-64`, sizes allocated in the constructor). `Wx`/`Wh` are the quadratic-in-`hidden` terms; `E`/`Wy` are linear-in-`hidden` (fixed `vocab_size=256` for D17). Confirmed via direct read of `forward_step`/`backward_step` (`char_lstm.cpp:83-204`, `206-350`): the hot loops are `matvec_rowmajor` (rows×cols multiply-adds) called on `Wx`/`Wh` with `rows=4·hidden, cols=hidden`, and `outer_rowmajor` of the same shape for the `dWx`/`dWh` gradients, plus two more `O(hidden²)` loops for `dx`/`dh_prev` (`char_lstm.cpp:326-347`). This is the actual matvec/gate implementation the task asked to confirm against — it is a **plain dense LSTM gate matrix**, not a low-rank or structured one, so the classic `O(hidden²)` recurrent-weight cost applies directly and exactly (see §5 for the quantified ratio).

`axiom_grammar_from_seed(seed, hidden)` (`axiom_activation.hpp:33-51`, used by H15) and `SrGateLaws` per-dimension gate laws (H16, `char_lstm.cpp:100-114`) are `O(hidden)` — linear, not a concern.

### 4.2 Components that do NOT scale with hidden dim (decoupled — confirmed)

Everything the task asked to check as a potential blocker turned out to be an **independent** config dimension, not derived from `lstm_hidden`:

| Component | Sizing param | Independent? |
|---|---|---|
| GRIA low-rank field | `field_dim=160`, `gria_rank=32` (`cyphalm_config.hpp:48,55`) | yes — `GRIALowRank` constructed with `field_dim`, not `lstm_hidden` (`cyphalm_batch.cpp:66`) |
| DIF experts (`CyphaDIF`) | `field_dim_`, `n_experts` (`cyphalm_dif.hpp:57-58`) | yes — `Expert` I/O dims come from `field_dim`, never `lstm_hidden` (`cyphalm_dif.hpp:52`) |
| RPSM | `rpsm_state_dim=128`, `rpsm_feat_dim=64` (`cyphalm_config.hpp:149-150`) | yes — separate config fields, wired at `cyphalm_model.cpp:331-332` |
| Kernel LLR (`kernel_m`) | `kernel_m=256` (`cyphalm_config.hpp:209`) | yes — Nyström basis count over the DIF field, not the LSTM hidden state |
| `ContextBank` slots | `context_bank_slots=64`, own `embed_dim` (`context_bank.hpp:11`) | yes — constructed with `d_embed`, not `lstm_hidden` |
| Cell-hypothesis `n_experts` (H06=4, H14=8) | `cypha_cell_hypothesis.cpp:101,133` | yes — DIF expert counts, orthogonal to `lstm_hidden`; these variants will run unmodified at any `lstm_hidden` value |

The hybrid blend that combines the GRIA/DIF path and the LSTM path happens at the **vocab-size logit level** (`blend_log_probs`, `char_lstm.cpp:412-418`, operating on `log_g`/`log_l` which are both `vocab_size`-length), not at the hidden-state level — so there is no dimensional-mismatch risk between the LSTM hidden state and the rest of the hybrid stack when `lstm_hidden` changes. **Conclusion: scaling `lstm_hidden` alone, with everything else held fixed, is architecturally clean.** No cell-hypothesis variant needs a code change to remain runnable at hidden=512/1024.

### 4.3 The one genuine blocker found: D_eff eigenvalue measurement caps at 256 dims

`participation_ratio_covariance_eigenvalue` (`native/src/intelligence/measurers.cpp:137-197`) has a hard guard:

```137:141:native/src/intelligence/measurers.cpp
double participation_ratio_covariance_eigenvalue(const double* activations, int n_samples,
                                                 int n_dims) {
  if (n_samples < 2 || n_dims <= 0 || n_dims > 256) {
    return participation_ratio_variance_proxy(activations, n_samples, n_dims);
  }
```

Above 256 dims it **silently falls back** to the cheaper `VarianceProxy` method instead of throwing or warning. The reason is cost, not correctness: it builds a full `n_dims × n_dims` covariance matrix (`measurers.cpp:150-165`) and diagonalizes it via 64 iterations of cyclic Jacobi rotation (`jacobi_symmetric_eigenvalues`, `measurers.cpp:80-135`), which is `O(n_dims³)` per call — already ~1B FLOPs at `n_dims=256`, and would be ~69B FLOPs at `n_dims=1024`. This directly conflicts with Phase 5's own goal ("measure D_eff as function of hidden dim" — the paper explicitly wants the higher-fidelity eigenvalue method, marked "(Paper IV fidelity)" in `measurers.hpp:9-10`) — **at hidden=512/1024 the "fidelity" method the paper wants is exactly the one that's disabled.**

This is a real gap but a cheap fix, not a redesign: participation ratio is `PR = (Σλ)² / Σλ²` where `λ` are covariance eigenvalues. For a symmetric PSD matrix `Σλ = trace(C)` and `Σλ² = trace(C²) = ||C||_F²` (Frobenius norm squared of the covariance matrix) — **both are computable directly from the covariance matrix without ever diagonalizing it.** Recommendation (Phase 0 of §7 below): add a third `ParticipationRatioMethod::TraceFrobenius` that computes `trace(C)²/‖C‖_F²` in `O(n_dims²·n_samples)` (the covariance-build cost already paid) instead of `O(n_dims³)`, drop the `n_dims > 256` guard for that path, and make it the default fidelity method for hidden ≥ 256. This removes both the cost wall and the silent-fallback correctness gap in one change, and is a prerequisite for trusting any D_eff number reported at 512/1024.

By default this doesn't bite today: `apply_math_integration_preset` sets `use_lstm_d_eff_hidden_nudge = true` but `use_eigenvalue_d_eff = false` (`cyphalm_math_integration.cpp:124-125`), so production runs use the cheap variance-proxy path per-step already. The eigenvalue method is opt-in via `--use-eigenvalue-d-eff` (`cyphalm_bench_native.cpp:109,218`) and is what any serious D_eff-vs-hidden-dim validation run should use — which is exactly why it needs fixing before Phase 5's sanity experiment, not after.

`lstm_hidden_d_eff()` (`cyphalm_model.cpp:642-659`) draws its sample matrix from `lstm_h_history_rows_`, a ring buffer capped at `kLstmHiddenHistoryMax = 48` rows (`cyphalm_model.hpp:217`, `cyphalm_model.cpp:636-639`) — so `n_samples` is always ≤48 regardless of hidden dim; only `n_dims` (=`hidden`) grows, which is exactly the axis the `>256` guard is checking.

---

## 5. Cost estimate: memory and wall-clock at 512 and 1024

### 5.1 Memory (not a real constraint)

`CharLSTMHead` parameter count = `E + Wx + Wh + b + Wy + by = vocab·hidden + 4·hidden² + 4·hidden² + 4·hidden + vocab·hidden + vocab` (`char_lstm.cpp:60-65`), with `vocab=256` fixed for D17:

| hidden | params | doubles (8B) | vs. hidden=128 |
|---|---|---|---|
| 128 | ~197k | ~1.6 MB | 1× |
| 256 | ~657k | ~5.3 MB | ~3.3× |
| 512 | ~2.36M | ~18.9 MB | ~12× |
| 1024 | ~8.92M | ~71.4 MB | ~45× |

Even at 1024, total weight memory is tens of MB — **memory is not a practical blocker** on any machine capable of running this repo today. (Transient per-step gradient buffers in `CharLSTMGrad` mirror the weight sizes and are stack/heap-allocated per call, not accumulated — same order of magnitude, still trivial.)

### 5.2 Wall-clock (the real constraint)

Per-token step cost is dominated by the `O(hidden²)` terms in `char_lstm.cpp` forward/backward (§4.1): two `4·hidden²` matvecs forward (`Wx`, `Wh`), two `4·hidden²` outer-product gradient builds backward (`dWx`, `dWh`), and two more `4·hidden²` loops for `dx`/`dh_prev` — six `4·hidden²` terms, i.e. ≈`24·hidden²` mult-adds, plus `≈3·vocab·hidden` linear terms (Wy forward/backward + `dh_new`). At `vocab=256`:

| hidden | quadratic term (24h²) | linear term (3·256·h) | total unit-ops | ratio vs. hidden=128 |
|---|---|---|---|---|
| 128 | 393,216 | 98,304 | 491,520 | 1.0× |
| 256 | 1,572,864 | 196,608 | 1,769,472 | 3.6× |
| 512 | 6,291,456 | 393,216 | 6,684,672 | ~13.6× |
| 1024 | 25,165,824 | 786,432 | 25,952,256 | ~52.8× |

This is close to, but slightly below, pure `hidden²` scaling (16× and 64× respectively) because the linear `vocab·hidden` terms (Wy/embedding) don't shrink proportionally — they're a bigger fraction of total cost at small `hidden` and a smaller fraction at large `hidden`. **Actual measured ratio will land between the linear-adjusted estimate (~13.6×/~52.8×) and the naive hidden² estimate (16×/64×)** — either way, the recurrent weight matrices dominate, confirming the task's `hidden_dim²` prior against this codebase's real matvec/gate implementation.

**Wall-clock baseline (measured, read-only from existing logs — not re-run to avoid colliding with the live overnight):** the June 28 D17 production run (hidden=128, n_train=300000, `--math-integration`, single-threaded) started at transcript time `17:07:01` and its `overnight_d17_*.log` was last written at `17:52:31` — **~45.5 minutes wall-clock** (`bench/results/overnight_d17_20260628_170701.log` file timestamps; run parameters from `bench/results/production_overnight_20260628_170701.log:19-22`). This is the empirical anchor, not a synthetic estimate.

Applying the compute ratios above, assuming the LSTM matvec/gradient path is roughly 70–90% of that wall-clock (the rest being GRIA n-gram fusion, corpus I/O, and math-integration profiler overhead, none of which scale with `hidden`):

| hidden | est. wall-clock (single-thread, 300k steps, math-integration) | assumption |
|---|---|---|
| 128 (current) | 45.5 min (measured) | — |
| 256 | ~1.4–2.0 hours | 3.6× compute ratio |
| 512 | ~5–9.5 hours | 13.6× compute ratio |
| 1024 | ~18–36 hours | 52.8× compute ratio |

**This is a wide range on purpose — it is an estimate, not a measurement, and the whole point of §7 Phase 2 is to replace it with a real number at a cheap tier before committing to a 300k production run at 512 or 1024.** A naive 1024-dim @ 300k run could plausibly take over a day single-threaded; that is the central practical risk this plan flags for the user before any expensive run is scheduled. `cyphalm_parallel.hpp`/`set_thread_count` exists but D17 production explicitly runs `--threads 1` (`bench/config/d17_wikitext_full_profile.json:14`) for determinism/pinning reasons — multi-threading is a possible mitigation but changes the BPC pin and is out of scope for this plan.

---

## 6. What's measurable today vs. not (D_eff tooling)

Confirmed: `native/include/cypha/intelligence/measurers.hpp` is exactly the "intelligence-stats" tool the task expected, and it does have D_eff:

```5:18:native/include/cypha/intelligence/measurers.hpp
/// Participation ratio computation method (Paper IV D_eff).
enum class ParticipationRatioMethod {
  /// Column-variance proxy (fast; legacy default).
  VarianceProxy = 0,
  /// Covariance eigenvalue PR: ``(Σλ)² / Σλ²`` (Paper IV fidelity).
  CovarianceEigenvalue = 1,
};

/// Participation ratio ``(Σλ)² / Σλ²`` from column variances, divided by ``n_dims``.
double compute_participation_ratio(const double* activations, int n_samples, int n_dims);

/// Same as above with explicit method (Phase 35 eigenvalue PR).
double compute_participation_ratio(const double* activations, int n_samples, int n_dims,
                                   ParticipationRatioMethod method);
```

Wired into the model via `CyphaLMModel::lstm_hidden_d_eff()` (`cyphalm_model.cpp:642-659`), which is fed to the profile-guided-loss hidden-state nudge (`:1008-1017`) and reported through `IntelligenceProfiler`/`LmIntelligenceMonitor` into the same 7-stat profile JSON that every bench domain already emits (`profile_completeness`, `statistics` — see the `bpc`/`kappa` JSON block visible in every `bench/results/*.log`, e.g. `production_overnight_20260628_170701.log:23-224`).

**So:** the tooling to measure D_eff at any hidden size exists and needs zero new code to *run* — every existing `cyphalm_bench_native --intelligence-profile` invocation already reports a `d_eff` observation (via the variance-proxy method) alongside `bpc`/`kappa` for whatever `lstm_hidden` was configured. The only real gap is the eigenvalue-fidelity method's `n_dims > 256` ceiling (§4.3), which matters only if the sanity experiment (§7 Phase 2) wants the higher-fidelity number rather than the variance proxy the production preset already uses. **Recommendation: run the Phase 2 sanity experiment with the variance-proxy method first** (zero new code, already default), and only invest in the trace/Frobenius fix (§4.3) if the sanity experiment's D_eff trend looks ambiguous enough to need the higher-fidelity method to trust it.

---

## 7. Proposed phase plan

Renumbered relative to Paper IV's own §7 (which assumes Phases 1–4 are done and starts hidden-dim scaling at Phase 5, from a 256-dim baseline). Because this codebase's production baseline is actually 128-dim (§2) and has one measurement gap (§4.3), the practical plan needs two phases *before* the paper's "Phase 5" and treats the paper's single Phase 5 as three phases (sanity → mid-scale → production) to de-risk the 300k commitment the task asked to avoid rushing into.

| Phase | Goal | Scope | Depends on |
|---|---|---|---|
| **0** | Fix D_eff eigenvalue-method scaling ceiling | Add `ParticipationRatioMethod::TraceFrobenius` (`trace(C)²/‖C‖_F²`, `O(n_dims²)` not `O(n_dims³)`) to `measurers.cpp`; drop/raise the `n_dims > 256` guard for the new method; unit test at `n_dims` ∈ {128, 256, 512, 1024} against the existing Jacobi method for `n_dims ≤ 256` (should match to float precision) | — |
| **1** | Add `--lstm-hidden N` CLI plumbing | `cyphalm_bench_native.cpp`: add `Args::lstm_hidden` (default -1 = unset), parse `--lstm-hidden`, apply unconditionally after `apply_bench_profile`/`apply_bench_mode`/`apply_cell_variant` (i.e. **not** gated behind `--math-integration` like the other grid-search overrides at `cyphalm_bench_native.cpp:267-352` — hidden-dim needs to be settable in vanilla hybrid mode too) | none — independent of Phase 0 |
| **2** | Fast/medium sanity experiment: does D_eff move the right way at all? | New bench domain (see below) sweeping `lstm_hidden ∈ {128, 256, 512}` at a **medium tier** (`n_train=5000`, matching the `kD41ScaleNTrain` convention at `bench_domains.cpp:4835`), `--intelligence-profile` on, measuring `bpc`, `kappa`, `d_eff` per size. This is the "measure before you commit" step the task asked for. | Phase 1 (needs the CLI flag); Phase 0 only if variance-proxy trend is ambiguous |
| **3** | Confirm at hidden=512, n_train=300000 (single seed, single-threaded, scratch build) | Run `d17`/hybrid @ hidden=512 @ full 300k in `native/build_scale` (not `native/build_math`), record actual wall-clock against the §5.2 estimate, record `bpc`/`kappa`/`d_eff` delta vs. the locked hidden=128 baseline (`bench/BASELINE_LOCK.json:39-50`) | Phase 2 showing a positive D_eff trend |
| **4** | Confirm at hidden=1024, n_train=300000 | Same as Phase 3 at hidden=1024. Given the §5.2 estimate (18–36h single-threaded), schedule as its own overnight-scale run, independent of and non-colliding with the P0 crash-fix overnight run | Phase 3 showing continued positive trend and BPC not regressing |
| **5** | Lock + compare to Paper IV target | If Phase 4 confirms κ trending toward 0.90–0.92 and `d_eff → 0.50`, propose a `bench/BASELINE_LOCK.json` entry for the new hidden-dim-scaled D17 config (as a **new, separate lock section** — e.g. `d17_hidden_scale_results` — not overwriting the existing `d17_hybrid_baseline`/`overnight_results` pin, since those are explicitly the 128-dim reference other domains validate against, e.g. `kD17HybridPinBpc` in `bench_domains.cpp:3285`) | Phase 4 |

**New bench domain (Phase 2+):** following the `d41_math_integration_scale_validation` pattern (`bench_domains.cpp:5073-5177`) — subprocess-drive `cyphalm_bench_native` at each `lstm_hidden` value via `run_math_integration_bench_subprocess`-style helper, diff `bpc`/`kappa`/`d_eff` across sizes, `finalize_domain(...)`, write a results table under `cypha_bench.domains.tables_dir()`. The next free domain slot is **`d77`** (highest currently registered is `d76`, `bench_domains.cpp:9344`) — propose `d77_hidden_dim_scale_validation`, registered alongside the others at `bench_domains.cpp:9225-9344`.

Do **not** add a new `bench/config/profiles/cyphalm_d17_*.json` variant profile for this — `apply_bench_profile` only grants full-WikiText-corpus treatment to the literal string `"d17"` (`bench_domains.cpp:1165`, `cyphalm_config.cpp:241-242`), so a differently-named profile would silently lose the production corpus. The clean approach is `--profile d17 --lstm-hidden N` (Phase 1's flag applied as a post-profile-load override on top of the existing, unmodified `d17` profile), which keeps the full-corpus/vocab/pin logic intact and only changes the one dimension under test.

---

## 8. Summary of blockers and risks

1. **Wall-clock, not memory, is the binding constraint.** Memory scales to ~71 MB even at 1024-dim (trivial). Compute scales ~13.6× at 512-dim and ~53× at 1024-dim relative to the current 45.5-minute 300k baseline — plausibly 18–36 hours single-threaded at 1024-dim. This must be measured at a cheap tier (Phase 2) before scheduling an expensive overnight run, exactly as the task requested.
2. **The D_eff "fidelity" (eigenvalue) measurement method silently degrades above 256 dims** (`measurers.cpp:139`) — it is not wired to fail loudly, so a naive scale-up run would report a D_eff number at 512/1024 that's actually the cheaper variance-proxy metric, undermining the very validation Phase 5 exists to do. Cheap fix identified (trace/Frobenius reformulation avoids `O(n³)` entirely) but not yet implemented (Phase 0).
3. **No `--lstm-hidden` CLI override exists yet** — trivial to add (Phase 1), follows the exact pattern of `--n-experts`/`--kernel-m` already in `cyphalm_bench_native.cpp`, but must be wired *outside* the `--math-integration`-gated block so it works in plain hybrid mode too.
4. **The current production baseline is 128-dim, not 256-dim** as Paper IV's own reference point assumes — the practical scale path has one more step (128→256) than the paper implicitly plans for, and the existing `2.873`/`2.864` BPC pins in `BASELINE_LOCK.json` are locked at 128-dim and must not be overwritten by a scaled run (any scaled result needs a new, separate lock section).
5. **No architectural blocker was found for the scale-up itself** — the codebase is fully dynamically sized (`std::vector`), GRIA/DIF/RPSM/kernel-LLR/context-bank are already dimensionally decoupled from `lstm_hidden`, and no cell-hypothesis variant (H01–H22) has a hidden-size-specific hardcoded assumption. This is the single most load-bearing finding for de-risking Phase 5: the paper's "no new theoretical breakthroughs" claim holds at the *architecture* level; the actual risk is entirely in *measured cost* and *measurement fidelity*, both addressed by Phases 0–2 below.
6. **Paper IV's own prerequisite stack has one unverified piece**: the `SelfCorrectingCypha`/epistemic-feedback wrapper (`self_correcting_infer.hpp`) was not confirmed wired into the live D17 train/eval loop in this pass — worth a follow-up read before treating Paper IV's κ=0.89 pre-scale estimate as fully "shipped," independent of the hidden-dim question.

---

## Appendix: commands reference

All commands use a scratch build directory (`native/build_scale`) to avoid touching `native/build_math`, which the parallel P0 subagent's live overnight run depends on. Read-only inspection of `bench/BASELINE_LOCK.json` and existing logs was used for all baseline numbers in this document — nothing was re-run against the production build during this scoping pass.

```powershell
# One-time: configure a separate scratch build (does not touch native/build_math)
cmake -S native -B native/build_scale -DCMAKE_BUILD_TYPE=Release
cmake --build native/build_scale --target cyphalm_bench_native --config Release

# Phase 2 sanity sweep (medium tier, n_train=5000) — after Phase 1 CLI flag lands
native/build_scale/cyphalm_bench_native --profile d17 --mode hybrid --n-train 5000 --n-eval 256 `
    --threads 1 --intelligence-profile --lstm-hidden 128
native/build_scale/cyphalm_bench_native --profile d17 --mode hybrid --n-train 5000 --n-eval 256 `
    --threads 1 --intelligence-profile --lstm-hidden 256
native/build_scale/cyphalm_bench_native --profile d17 --mode hybrid --n-train 5000 --n-eval 256 `
    --threads 1 --intelligence-profile --lstm-hidden 512

# Phase 3/4 production-scale confirmation (run only after Phase 2 shows a positive D_eff trend;
# expect multi-hour to >1-day single-threaded wall-clock per §5.2 — schedule accordingly)
native/build_scale/cyphalm_bench_native --profile d17 --mode hybrid --n-train 300000 --n-eval 2000 `
    --threads 1 --intelligence-profile --lstm-hidden 512
native/build_scale/cyphalm_bench_native --profile d17 --mode hybrid --n-train 300000 --n-eval 2000 `
    --threads 1 --intelligence-profile --lstm-hidden 1024

# Read-only baseline inspection used to write this document (no writes):
Get-Content bench/BASELINE_LOCK.json
Get-Item bench/results/overnight_d17_20260628_170701.log | Select-Object LastWriteTime
```

### Files read/cited in this scoping pass

- `docs/reports/DEV_PLAN_2026-07-11.md` (§2–3)
- `docs/research/intelligence_stats/cypha_self_correcting_paper4.md` (full read)
- `docs/research/intelligence_stats/soft_world_paper5.md` (§5, D_eff/κ figures)
- `native/include/cypha/cyphalm/cyphalm_config.hpp`
- `native/src/cyphalm/cyphalm_config.cpp`
- `native/include/cypha/cyphalm/char_lstm.hpp`, `native/src/cyphalm/char_lstm.cpp`
- `native/include/cypha/cyphalm/axiom_activation.hpp`
- `native/include/cypha/cyphalm/cyphalm_dif.hpp`
- `native/include/cypha/cyphalm/context_bank.hpp`
- `native/src/cyphalm/cyphalm_gria.cpp`
- `native/src/cyphalm/cypha_cell_hypothesis.cpp` (read-only; variant table)
- `native/src/cyphalm/cyphalm_model.cpp` (`lstm_hidden_d_eff`, history buffer, train loop)
- `native/include/cypha/intelligence/measurers.hpp`, `native/src/intelligence/measurers.cpp`
- `native/src/cyphalm/cyphalm_math_integration.cpp` (`apply_math_integration_preset`)
- `native/tools/cyphalm_bench_native.cpp` (CLI `Args`/`parse_args`/`main`)
- `native/src/bench/bench_domains.cpp` (domain registration table, `d41` pattern, `run_cyphalm_domain`)
- `bench/config/profiles/cyphalm_d17_wikitext.json`, `cyphalm_d17_hybrid.json`
- `bench/config/d17_wikitext_full_profile.json`
- `bench/BASELINE_LOCK.json` (read-only)
- `bench/results/production_overnight_20260628_170701.log`, `bench/results/overnight_d17_20260628_170701.log` (read-only, timing evidence)

---

## Phase 0-1 results (2026-07-11)

**Status:** Implemented and run. Executed by a follow-up subagent against `native/build_scale` (scratch build, does not touch `native/build_math`). All commands single-threaded, `n_train ≤ 5000`, per the containing task's constraints — no 300k-scale or 1024-dim run was performed.

### Phase 0 — D_eff eigenvalue-method scaling ceiling: fixed

Added `ParticipationRatioMethod::TraceFrobenius` to `measurers.hpp`/`measurers.cpp`, matching the reformulation this document proposed in §4.3: for a symmetric covariance matrix `C`, `trace(C) == Σλ` and `trace(C²) == ||C||_F² == Σλ²` exactly (not an approximation — `trace(C²) = Σᵢ Σⱼ Cᵢⱼ·Cⱼᵢ = Σᵢ Σⱼ Cᵢⱼ²` since `C` is symmetric), so the participation ratio `(Σλ)²/Σλ²` is computable directly from the already-built covariance matrix without ever diagonalizing it — `O(n_dims²·n_samples)` instead of `O(n_dims³)`.

Changes:
- `participation_ratio_covariance_eigenvalue` (the function backing `ParticipationRatioMethod::CovarianceEigenvalue`, i.e. what `cfg.use_eigenvalue_d_eff = true` selects) no longer silently falls back to the cheaper, *numerically different* `VarianceProxy` metric above 256 dims. It now delegates to the new trace/Frobenius path — same `(Σλ)²/Σλ²` quantity, computed without Jacobi diagonalization, valid at any `n_dims`.
- Below 256 dims the original Jacobi-diagonalization path is unchanged (kept for the existing exact-match unit test against `VarianceProxy`), since it was already affordable there.
- Added `test_trace_frobenius_participation_ratio` to `native/tools/intelligence_profiler_papers.cpp` (registered CTest `native_intelligence_profiler_papers`): verifies `TraceFrobenius` agrees with the Jacobi `CovarianceEigenvalue` result to `1e-6` at small `n_dims` (algebraic-identity sanity check), then exercises the new >256-dim path directly at `n_dims=300` (above the old hard guard) with a deterministic, non-degenerate synthetic covariance structure, asserting `CovarianceEigenvalue` and `TraceFrobenius` now agree to `1e-9` there too (i.e. the >256-dim path no longer silently returns the different `VarianceProxy` number).
- `ctest --test-dir native/build_scale -R "intelligence_profiler|measurer|d_eff|deff"` → both `native_intelligence_profiler_smoke` and `native_intelligence_profiler_papers` pass; `native_cyphalm_bench_intelligence_profile` (the CLI-level intelligence-profile smoke test) also still passes.

### Phase 1 — `--lstm-hidden` CLI override: added

Added `--lstm-hidden N` to `cyphalm_bench_native.cpp`, applied **unconditionally** (not gated behind `--math-integration`, per this document's own note in §7 Phase 1) right after the profile/mode/cell-variant setup and before the math-integration-gated override block. Default is unset (`-1`), which leaves whatever the profile already configured (128 for `d17`). The resolved value is echoed back in the output JSON as `"lstm_hidden"` for traceability. Usage:

```powershell
native/build_scale/cyphalm_bench_native.exe --profile d17 --mode hybrid --n-train 5000 --n-eval 256 --threads 1 --intelligence-profile --lstm-hidden 256 --bench-seed 42
```

### Phase 2 sanity sweep — results (medium tier, `n_train=5000`, `n_eval=256`, `--bench-seed 42`, single-threaded, `native/build_scale`)

| hidden | bpc | bpc (LSTM-only) | kappa (criticality_score) | d_eff | alpha | calibration | tau | r_eu | sigma_branch | lipschitz |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 4.0396 | 4.0311 | 0.8312 | 0.586536 | 0.3333 | 0.7443 | 0.4906 | 0.1682 | 0.500 | 0.3886 |
| 256 | 3.9300 | 3.9214 | 0.8348 | 0.586536 | 0.3335 | 0.7694 | 0.4906 | 0.1682 | 0.500 | 0.3886 |
| 512 | 3.8486 | 3.8401 | 0.8333 | 0.586536 | 0.3349 | 0.7576 | 0.4906 | 0.1682 | 0.500 | 0.3886 |

Raw JSON captured at `bench/results/hidden_dim_scale/sweep_hidden{128,256,512}.json` (not committed, per this task's constraints; regenerate with the command above at each hidden value).

### Finding: D_eff shows **no trend** — but for a wiring reason, not a noise reason

BPC improves monotonically and by a real margin as hidden dim increases (4.040 → 3.930 → 3.849, roughly 2.7% then 2.1% relative improvement per doubling) — consistent with the extra LSTM capacity actually being used, and with what the §5.2 cost model assumed.

**`d_eff` is bit-for-bit identical (`0.5865363087488848`) across all three hidden sizes.** So are `sigma_branch`, `tau`, `r_eu`, and `lipschitz`. This is not sampling noise — five different statistics agreeing to 16 significant figures across a 4× change in the swept parameter is a wiring signature, not a measurement-precision one. Tracing it down: the `d_eff` (and `sigma_branch`/`tau`/`r_eu`/`lipschitz`) values that `--intelligence-profile` reports come from `LmIntelligenceMonitor`, which is fed `field_x_` — the **GRIA low-rank field** (`field_dim=160`, `cyphalm_config.hpp:48`) — on every call to `observe_token` (`cyphalm_model.cpp:904`, `:1676`), not the LSTM hidden state. This document's own §4.2 already established that the GRIA field is dimensionally decoupled from `lstm_hidden` ("yes — `GRIALowRank` constructed with `field_dim`, not `lstm_hidden`"); what wasn't caught at planning time is that **this decoupling also applies to the D_eff *measurement path*, not just the architecture** — so sweeping `--lstm-hidden` changes the model's capacity and BPC (confirmed above) but is structurally invisible to the `d_eff` statistic the `--intelligence-profile` flag reports, regardless of which `ParticipationRatioMethod` is selected. `lstm_hidden_d_eff()` (`cyphalm_model.cpp:642-659`) *does* correctly key off `lstm_hidden` — it's the function this document's §4.3 already identified as hitting the eigenvalue guard — but it is currently wired only into the profile-guided-loss hidden-state gradient nudge (`use_lstm_d_eff_hidden_nudge`), and is never surfaced into `IntelligenceProfiler`'s exported observation/statistics. There is currently no code path by which `cyphalm_bench_native --intelligence-profile`'s JSON output can report an `lstm_hidden`-sensitive D_eff number at all, with or without `--math-integration` or `--use-eigenvalue-d-eff`.

`kappa` (`criticality_score`) moves within a ~0.4pp band (0.8312 → 0.8348 → 0.8333) with no monotonic direction — consistent with it being driven almost entirely by small `alpha`/`calibration` wobble (from the blended output distribution shifting slightly with more LSTM capacity) rather than by any real `d_eff` movement, since `d_eff` itself never moves.

### Go/no-go recommendation: **no-go on Phase 3+ (512/1024 @ 300k) until the D_eff measurement path is fixed**

This sanity sweep did exactly what it was supposed to do — surface a blocking problem cheaply (a few CPU-minutes at `n_train=5000`) before committing to hours-to-days of `n_train=300000` runs. The problem it surfaced is more fundamental than "is the signal noisy": **the currently-exported `d_eff` statistic cannot detect a hidden-dim effect at all**, independent of scale, because it is measuring the wrong tensor (the 160-dim GRIA field, not the `lstm_hidden`-width LSTM state). Running Phase 3/4 today would produce BPC deltas that look exactly like this sweep's (real, monotonic, capacity-driven) but a `d_eff`/`kappa` number that would again be flat by construction — which would either be misread as "the paper's claim doesn't hold" (wrong conclusion) or as "the paper's claim holds because BPC improved" (wrong evidence for that specific claim, which is about `D_eff` specifically, not BPC).

**Before any Phase 3/4 scale-up run:**
1. Wire `lstm_hidden_d_eff()` (or an equivalent measurement over the actual LSTM hidden-state history, `lstm_h_history_rows_`) into the exported bench JSON as its own statistic (e.g. `intelligence_profile.lstm_hidden_d_eff`), distinct from the existing field-based `d_eff`. This is a small, targeted addition — the measurement function and its `>256`-dim-safe `TraceFrobenius` path already exist after this pass; it only needs a call site and an export field, not new math.
2. Re-run this exact Phase 2 sweep (`hidden ∈ {128, 256, 512}`, `n_train=5000`) once that statistic is exported, and confirm it actually moves with `lstm_hidden` (a basic sanity check on the fix itself) before trusting any subsequent Phase 3/4 result.
3. Only then is a 512-dim @ 300k confirmation run (Phase 3) evidence for or against Paper IV's specific `D_eff`-driven κ claim.

The architectural finding from §4 (no blocker to scaling `lstm_hidden` itself) and the cost model from §5 (13.6×/53× compute ratios) both still stand and are not affected by this measurement-wiring gap — they concern the *model*, not the *instrumentation*. Phase 0 and Phase 1 as scoped are complete and independently correct (verified by CTest and by `lstm_hidden` visibly changing BPC in the sweep above); the blocker identified here is a new, distinct gap in the *reporting* layer discovered by actually running Phase 2, exactly as intended.

### Note on run stability under system load

Both `native/build_scale` sweep runs at `hidden=512` were executed while several other single-threaded native benchmark processes (`native/build_rpsm`, `native/build_math`) were concurrently running on the same machine (parallel agent work, per this task's constraints). The first `hidden=512` attempt exited with code `-1` and empty stdout/stderr after ~22 minutes of wall-clock (no Windows Error Reporting crash entry was logged for it, and no lingering process was found afterward) — most likely an external termination under heavy multi-process CPU contention rather than an application bug. A clean re-run under the same contention completed normally (exit code 0) in ~15 minutes and produced the `hidden=512` row above. Anyone re-running this sweep on a busy machine should treat a `-1`/empty-output exit as "retry," not as a code defect, unless it reproduces consistently in isolation.
