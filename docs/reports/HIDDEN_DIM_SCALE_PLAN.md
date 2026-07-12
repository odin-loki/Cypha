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

---

## Phase 2b results (2026-07-11): `lstm_hidden_d_eff` export fix and re-run

**Status:** Implemented, built, tested, and re-run. Executed by a follow-up subagent against `native/build_scale` (reused from the Phase 0/1 subagent, rebuilt after this change; `native/build_math` and `native/build_rpsm` were not touched). All commands single-threaded, `n_train ≤ 5000`.

### What `lstm_hidden_d_eff()` was already doing correctly

Confirmed by reading `CyphaLMModel::lstm_hidden_d_eff()` (`native/src/cyphalm/cyphalm_model.cpp:646-665`) end to end before touching anything: it is **not** a measurement that needed fixing. It:

- Draws its sample matrix from `lstm_h_history_rows_` (`native/include/cypha/cyphalm/cyphalm_model.hpp:218`), a ring buffer of up to `kLstmHiddenHistoryMax = 48` real LSTM hidden-state vectors, appended by `append_lstm_hidden_history(lstm_h_)` every hybrid `predict_next` step (`cyphalm_model.cpp:802`) — i.e. it is fed the actual post-`forward_step` LSTM hidden state (`lstm_h_`), width `lstm_->hidden`, not the GRIA field.
- Sizes its flattened sample matrix as `rows × hidden` using `lstm_->hidden` directly (`cyphalm_model.cpp:650-659`), so it automatically tracks whatever `--lstm-hidden` is set to, with no hard-coded dimension anywhere in the function.
- Delegates to `cypha::intelligence::compute_participation_ratio(...)` with `ParticipationRatioMethod::CovarianceEigenvalue` or `VarianceProxy` depending on `cfg_.use_eigenvalue_d_eff` (`cyphalm_model.cpp:660-664`) — the exact same, already-validated participation-ratio machinery the GRIA-field `d_eff` statistic uses, just pointed at a different tensor. Since Phase 0 (this doc, above) fixed `CovarianceEigenvalue` to delegate to the exact `TraceFrobenius` identity above 256 dims instead of silently falling back to `VarianceProxy`, this function is now also safe to call with `--use-eigenvalue-d-eff` at hidden ≥ 256 (not exercised in this pass — see below — but no longer a latent correctness trap if a future run enables it).
- Returns `-1.0` only when there's no LSTM head or fewer than 4 history rows have been observed yet — a legitimate "not enough data" guard, not a bug.

Its only real gap, exactly as the plan doc identified: it was called from exactly one place, `native/src/cyphalm/cyphalm_model.cpp:1013`, purely to compute a training-time gradient nudge target (`use_lstm_d_eff_hidden_nudge`) — never surfaced anywhere in the bench JSON. No changes were made to its math or its existing call site.

### Where/how it was exported

Two small, additive changes, no existing behavior touched:

1. `native/include/cypha/cyphalm/cyphalm_model.hpp`: added a public inline accessor `double lstm_hidden_d_eff_report() const { return lstm_hidden_d_eff(); }` next to the other public read-only accessors (`train_step_count()`, etc.), since the existing `lstm_hidden_d_eff()` is private. Zero behavior change — pure visibility wrapper.
2. `native/tools/cyphalm_bench_native.cpp`: immediately after `out["intelligence_profile"] = export_intelligence_monitor_report(profiler)` (the point where the existing 7-stat GRIA-field profile, including the existing `d_eff`, is serialized), added two new **nested** keys under the same `intelligence_profile` object:
   - `intelligence_profile.lstm_hidden_d_eff` — the new statistic (`model.lstm_hidden_d_eff_report()`), `null` when the model reports `< 0` (not enough history / no LSTM head).
   - `intelligence_profile.lstm_hidden_d_eff_method` — `"variance_proxy"` or `"covariance_eigenvalue"`, echoing `cfg.use_eigenvalue_d_eff`, so any consumer of the JSON knows which participation-ratio method produced the number without cross-referencing the CLI args.

   This only runs when `--intelligence-profile` is passed (same gate as the rest of the block) and is called *after* `model.accumulate_intelligence_profile(...)` has already run the eval corpus through the model, so the 48-row hidden-state history reflects the eval pass — methodologically consistent with how the existing GRIA `d_eff` is measured (also over the eval pass, via the same `IntelligenceProfiler`).

   The existing `intelligence_profile.statistics[].d_eff` (GRIA-field, fixed `field_dim=160`) is **untouched** — same call, same value, same position in the JSON. This was verified empirically in the re-run below: `d_eff` is bit-identical (`0.5865363087488848`) across all three hidden sizes, exactly as before Phase 2b, confirming the change is purely additive.

### CTest results

`ctest --test-dir native/build_scale -R "intelligence|d_eff|deff|measurer"` (6 matching tests, after rebuilding `cyphalm_bench_native`, `cypha_intelligence_bench`, `intelligence_lm_monitor_smoke`, `cypha_bench_run`):

```
100% tests passed, 0 tests failed out of 6
  native_intelligence_profiler_smoke .......... Passed
  native_intelligence_profiler_papers ......... Passed
  native_intelligence_bench_smoke ............. Passed
  native_cyphalm_bench_intelligence_profile ... Passed
  native_d39_intelligence_monitor_smoke ....... Passed
  native_intelligence_lm_monitor_smoke ........ Passed
```

No regressions. `native_cyphalm_bench_intelligence_profile` in particular exercises the exact CLI/JSON path that was modified and still passes, confirming `profile_completeness` (which checks the original 7-stat matrix, unaffected by the new nested keys) still reports `all_complete: true`.

### Corrected Phase 2 sweep — results (medium tier, `n_train=5000`, `n_eval=256`, `--bench-seed 42`, single-threaded, `native/build_scale`, variance-proxy method for both `d_eff` and `lstm_hidden_d_eff`)

| hidden | bpc | bpc (LSTM-only) | kappa (criticality_score) | GRIA `d_eff` (unchanged) | **`lstm_hidden_d_eff` (new)** |
|---|---|---|---|---|---|
| 128 | 4.039556 | 4.031056 | 0.831220 | 0.586536 | **0.270968** |
| 256 | 3.929983 | 3.921449 | 0.834821 | 0.586536 | **0.485134** |
| 512 | 3.848554 | 3.840080 | 0.833334 | 0.586536 | **0.576461** |

Raw JSON captured at `bench/results/hidden_dim_scale/sweep_hidden{128,256,512}_v2.json` (not committed, per this task's constraints; regenerate with the Phase 2 command above at each hidden value).

### Finding: `lstm_hidden_d_eff` now shows the expected upward trend — and it's a large one

`lstm_hidden_d_eff` **more than doubles from hidden=128 to hidden=256** (0.271 → 0.485, +79% relative) and continues climbing to hidden=512 (→ 0.576, +19% relative over 256), while the diminishing-returns shape tracks the diminishing-returns shape of the BPC improvement (4.040→3.930 is a bigger absolute drop than 3.930→3.849). This is exactly the qualitative signature Paper IV predicts: representational capacity (D_eff of the actual hidden state) increasing with hidden width, with the rate of increase slowing as the statistic approaches its ceiling of 1.0 (full participation). Unlike the flat GRIA `d_eff` (bit-identical `0.586536` at all three sizes, confirming the fix is additive and did not disturb the pre-existing statistic), `lstm_hidden_d_eff` is clearly and monotonically sensitive to `--lstm-hidden`, resolving the exact blocker this task set out to fix.

`kappa` (`criticality_score`) still moves only within a ~0.35pp band (0.8312 → 0.8348 → 0.8333), non-monotonically — this is expected and *not* a new problem: `criticality_score` is computed from the 7-stat profile matrix (`IntelligenceProfiler::criticality_score_for`), which is still built exclusively from the GRIA-field observations (`d_eff`, `alpha`, `sigma_branch`, `tau`, `r_eu`, `lipschitz`, `calibration`) via `LmIntelligenceMonitor::observe_token` — `lstm_hidden_d_eff` was deliberately exported as a new, separate, informational statistic (per this task's scope: "additive... do NOT replace/overwrite the existing GRIA-field d_eff"), not wired into the `kappa` formula itself. Whether `kappa` *should* eventually incorporate `lstm_hidden_d_eff` (e.g. as an additional profile axis, or blended into the existing `d_eff` axis) is a separate design decision outside this task's scope — flagged here as a natural follow-up, not a defect in this fix.

### Go/no-go recommendation: **go on Phase 3 (512-dim @ 300k), conditional as below — hold on Phase 4 (1024-dim) until Phase 3 lands**

The specific blocker this task was created to resolve is fixed and empirically confirmed: `lstm_hidden_d_eff` is now exported, distinct from and non-disruptive to the existing GRIA `d_eff`, passes all measurement CTests, and — critically — actually trends upward with `--lstm-hidden` in the same cheap sweep that previously showed it flat. This is no longer "BPC improved, trust that D_eff must be behind it" (the ungrounded inference the previous no-go was correctly worried about); it is now a direct, first-party measurement of the LSTM hidden state's effective dimensionality, and it moves the way Paper IV's theory says it should.

**Recommend proceeding to Phase 3 (hidden=512 @ n_train=300000, single seed, single-threaded, `native/build_scale`)**, with three conditions carried over from the original plan and unchanged by this fix:

1. Schedule it only once the machine is no longer contended by the live `native/build_math` D17→D21→cell-sweep overnight and the `native/build_rpsm` verification — Phase 3's own wall-clock estimate (~5–9.5h per §5.2) is itself sensitive to CPU contention, as directly observed in this pass (the `hidden=512` sweep at `n_train=5000` alone took ~15–22 minutes under contention against an ~ instant expectation at that scale in isolation).
2. Capture both `d_eff` (GRIA-field, for continuity with the existing `BASELINE_LOCK.json` pins) and the new `lstm_hidden_d_eff` in the Phase 3 output, and confirm `lstm_hidden_d_eff` continues its upward trend at production scale/seed — the medium-tier (`n_train=5000`) trend confirmed here is a sanity signal, not a guarantee it survives 60× more training steps, though there is no specific mechanism identified that would reverse it.
3. Do not lock a new `BASELINE_LOCK.json` section (per §7 Phase 5 of the original plan) until Phase 3's `kappa`/BPC numbers are in hand — this fix changes what's *measurable*, not what a full-scale run will actually *produce*.

**Hold on Phase 4 (hidden=1024)** until Phase 3 completes and its `lstm_hidden_d_eff` trend is confirmed at production scale — per the original plan's own phase ordering and cost model (§5.2's 18–36h single-threaded estimate for 1024-dim is the largest single commitment in this plan and should not be scheduled on an unconfirmed extrapolation from a 5,000-step sweep).

---

## Phase 3 results (2026-07-11): production-scale (`n_train=300000`) confirmation

**Status:** Executed by a follow-up subagent against a freshly rebuilt `native/build_scale` (HEAD at `da1ad6d`, confirmed via `git log`/`git show c1844db --stat` — includes `ff26a57`, `c1844db`, `da1ad6d`, the Phase 0/1/2b commits above). `native/build_math`, `bench/BASELINE_LOCK.json`, and the overnight orchestration scripts were not touched (verified via `git status --short bench/BASELINE_LOCK.json` at the end of this pass — clean). Both runs below are same-binary, same-day, same-seed (`--bench-seed 42`), same-flags apples-to-apples comparisons from this pass — **not** a comparison against the June 28 `BASELINE_LOCK.json` pins (those used `--math-integration`, which this pass deliberately did not pass, per the task's exact command spec).

Command (identical for both, only `--lstm-hidden` differs):

```powershell
native/build_scale/cyphalm_bench_native.exe --profile d17 --n-train 300000 --n-eval 2000 `
    --lstm-hidden {128|512} --intelligence-profile --threads 1 --bench-seed 42
```

Note: `--overnight`/`CYPHA_BENCH_FULL_CORPUS` was **not** set (not part of the task's specified flags), so both runs used the capped-at-10M-characters WikiText-2 train split (`full_corpus: false` in both output JSONs) rather than the uncapped full 10.8MB file — the cap barely binds (WikiText-2 train is ~10.8MB) and, critically, is **identical for both runs**, so the hidden=128-vs-512 comparison itself is unaffected; only an absolute comparison against a hypothetical full-corpus number would need to account for it.

### Result table

| hidden | bpc | bpc (LSTM-only) | kappa (`criticality_score`) | GRIA `d_eff` (fixed 160-dim field) | **`lstm_hidden_d_eff`** | wall-clock (successful attempt) |
|---|---|---|---|---|---|---|
| 128 | 3.044487 | 3.044341 | 0.835314 | 0.577841 | **0.599997** | ~52.3 min |
| 512 | 2.945026 | 2.944884 | 0.840743 | 0.577841 (bit-identical) | **0.331333** | ~16h 39min |

Full JSON captured at `bench/results/hidden_dim_scale/hidden128_300k.json` and `bench/results/hidden_dim_scale/hidden512_300k.json` (both `.log`/`.json` are identical copies; not committed to the lock, per this task's constraints). `bpc_lstm_only ≈ bpc` and `hybrid_gria_weight ≈ 0.000117` in both, confirming the hybrid blend is running in near-pure-LSTM mode symmetrically in both runs (a fair comparison of the LSTM head specifically, which is what `--lstm-hidden` controls).

### Run stability note (contention, not correctness)

Both runs hit the exact `-1`/empty-output external-termination failure mode already documented above ("Note on run stability under system load") **repeatedly** during this pass — the machine had upward of 15–18 concurrent `cyphalm_bench_native.exe`/other native-benchmark processes at points during this window, from several other sibling agents building/testing in `native/build_rpsm`, `native/build_kernel`, `native/build_multiview`, `native/build_softworld`, etc. (confirmed via `Get-Process` snapshots and cross-referencing other agents' terminal logs — no evidence of a deliberate kill script; all repo orchestration scripts that call `Stop-Process` were checked and either target a specific tracked PID or explicitly comment "never kills processes"/"do not kill other cyphalm processes here"). Both the hidden=128 and hidden=512 attempt-1 runs died with `exit_code=-1`, empty stdout, at effectively the same wall-clock instant each time — a signature of an external/contention-driven termination rather than independent application crashes.

**Mitigation used:** a self-healing watchdog loop (`Start-Process -Wait` in a retry loop, up to 40 attempts, 10s backoff, run detached from the Shell tool's own job tracking) was used for both, per the existing plan doc's own guidance to "treat -1/empty-output exit as retry, not a code defect." Both succeeded on **attempt 2**:

- hidden=128: attempt 1 failed after ~8.4 min (contention-killed); attempt 2 succeeded cleanly in **52.3 minutes** — close to the §5.2 anchor (45.5 min, math-integration, June 28) despite this run not using `--math-integration` and running under a busier machine.
- hidden=512: attempt 1 failed after ~8.4 min; attempt 2 succeeded, but took **16 hours 39 minutes** — well above the §5.2 estimate range of 5–9.5h. Given the sustained high process contention observed throughout this specific run's lifetime (confirmed via periodic `Get-Process` CPU-accumulation checks showing the expected steady climb, so the process was never stalled, just slower than an uncontended run would be), **this number should be read as "measured wall-clock under heavy shared-machine contention," not as a clean single-tenant estimate** — but it is the real, honest number for this specific run, exactly as the task asked to record "for future cost-estimation accuracy." The actual-to-baseline ratio observed (16.65h / 52.3min ≈ 19.1×) is higher than the §5.2 compute-only model's 13.6× estimate, consistent with added contention overhead on top of the pure `O(hidden²)` scaling.

### Finding 1: BPC and kappa both improve at hidden=512, consistent with expectations

BPC improves from 3.0445 → 2.9450 (−3.3% relative) and `kappa` (`criticality_score`) improves marginally from 0.8353 → 0.8407 — both directionally consistent with Paper IV's claim and with the medium-tier (`n_train=5000`) Phase 2b sweep's BPC trend (4.040→3.930→3.849 at 128/256/512). The GRIA-field `d_eff` is bit-identical (`0.577841`) between the two runs, exactly as expected (§4.2/Phase 2b: it measures the fixed-160-dim GRIA field, structurally independent of `lstm_hidden`) — this is a useful internal consistency check confirming nothing else about the harness changed between the two runs besides the swept parameter.

### Finding 2: `lstm_hidden_d_eff` trend **reverses** at production scale — does not match the 5k-scale sweep's direction

This is the central, unexpected result of this phase. The Phase 2b medium-tier sweep (`n_train=5000`) found `lstm_hidden_d_eff` climbing monotonically: 0.271 (h=128) → 0.485 (h=256) → 0.576 (h=512). At production scale (`n_train=300000`, 60× more training steps, same hidden values, same seed), the picture is different:

| hidden | `lstm_hidden_d_eff` @ n_train=5000 (Phase 2b) | `lstm_hidden_d_eff` @ n_train=300000 (Phase 3, this pass) |
|---|---|---|
| 128 | 0.270968 | **0.599997** |
| 512 | 0.576461 | **0.331333** |

Not only does the hidden=128→512 *direction* reverse between the two scales (up at 5k, down at 300k), the *within-hidden-size* values also move a lot with more training: hidden=128's `lstm_hidden_d_eff` nearly doubles (0.271→0.600) with 60× more steps, while hidden=512's *drops* by nearly half (0.576→0.331). **This directly contradicts this document's own conditional go-ahead for Phase 3** ("Recommend proceeding to Phase 3... condition 2: confirm `lstm_hidden_d_eff` continues its upward trend at production scale") — the trend did not continue; it inverted.

**Root-cause analysis (not fixed in this pass — flagged for a dedicated follow-up):** `lstm_hidden_d_eff()` draws its sample matrix from `lstm_h_history_rows_`, a ring buffer hard-capped at `kLstmHiddenHistoryMax = 48` rows (`native/include/cypha/cyphalm/cyphalm_model.hpp:218`, unchanged by this pass) — **regardless of `lstm_hidden`**. This means the sample-to-dimension ratio for the participation-ratio estimate is:

| hidden | n_samples | n_dims | samples/dims ratio |
|---|---|---|---|
| 128 | 48 | 128 | 0.375 |
| 512 | 48 | 512 | 0.094 |

At hidden=512 the statistic is being estimated from **4× fewer samples per dimension** than at hidden=128, i.e. a much more severely underdetermined regime (48 samples can span at most a 47-dimensional subspace of a 512-dimensional state — by construction, most of the raw covariance structure above rank ~47 is unobservable no matter what the true representational spread is). `use_eigenvalue_d_eff` was not passed in either run (both report `"lstm_hidden_d_eff_method": "variance_proxy"`), so this is the cheap per-dimension-variance proxy, not the Phase-0-fixed `TraceFrobenius`/eigenvalue path — though switching methods would not fix the underlying sample-count problem, since both methods consume the same 48-row history.

A second, non-exclusive explanation worth flagging: this may not be purely a sampling artifact. A fully-trained (300k-step) 512-dim LSTM has had much more opportunity than a 5,000-step one to specialize its representation — if the network learns to concentrate its *useful* variance onto a relatively fixed-size subset of directions regardless of how wide the hidden state is (i.e., the task's intrinsic complexity, not the available width, sets the effective dimensionality), then the *width-normalized* participation ratio (`compute_participation_ratio` is explicitly "divided by `n_dims`" per its own doc comment in `measurers.hpp:171`) would mechanically decrease as `hidden` grows even while raw representational usage stays flat or grows sublinearly — an authentic finding about *this specific normalized metric*, not necessarily evidence against Paper IV's broader capacity claim (which BPC/kappa, the two axes that do **not** depend on this narrow 48-row/`n_dims`-normalized statistic, both still support in this same run).

Either way — undersampling artifact, genuine representational compression, or some mix — **the current `lstm_hidden_d_eff` statistic cannot be trusted as production-scale evidence for or against Paper IV's `D_eff`-driven κ claim until the 48-row history cap is addressed** (e.g., scaling `kLstmHiddenHistoryMax` with `hidden`, or reporting an unnormalized effective-dimension count alongside the normalized ratio, or explicitly flagging low sample/dim ratios in the exported JSON). This is a new, distinct gap from the Phase 0 (eigenvalue-ceiling) and Phase 2b (wrong-tensor) gaps already fixed — those fixes were necessary but not sufficient; this one is about statistical power, not wiring or algorithmic ceiling, and was only visible once a real production-scale (`n_train=300000`) run was performed, exactly as Phase 3 was designed to surface.

### Go/no-go recommendation: **no-go on Phase 4 (1024-dim) — hold until the `lstm_hidden_d_eff` sampling gap is resolved and re-measured**

Phase 3 did exactly what a de-risking phase is supposed to do: it caught a real problem before an even more expensive commitment. Recommending against Phase 4 for three independent, compounding reasons:

1. **The specific metric this entire plan is built around (`lstm_hidden_d_eff`) inverted direction at production scale.** Proceeding to 1024-dim on the strength of a trend that just broke at the previous scale step would repeat exactly the mistake Phase 2/2b's own no-go/go logic was designed to avoid — extrapolating from an unconfirmed (here, actively contradicted) smaller-scale signal.
2. **Wall-clock cost came in far above estimate even before contention is accounted for.** The 512-dim run took 16h39min against a 5–9.5h estimate (≈1.75–3.3× over, even granting heavy observed machine contention as a mitigating factor). A 1024-dim run, per the original §5.2 model (~53× vs. hidden=128's baseline, vs. 512-dim's ~13.6–19×), could plausibly run 30–70+ hours under similar contention — a very large commitment to make while the measurement this run would supposedly validate is currently unreliable.
3. **BPC and kappa, the two metrics that did move in the expected direction, are not `D_eff`-specific evidence.** They support "more hidden-dim capacity helps the model" in general (a weaker, less novel claim than Paper IV's specific `D_eff`-driven κ mechanism), which does not by itself justify the Phase 4 cost under this plan's own stated goal of validating the `D_eff` mechanism specifically.

**Recommended next step before any Phase 4 scheduling:** a small, targeted follow-up — scale `kLstmHiddenHistoryMax` (or add a `--lstm-hidden-history-rows` override) so the sample count tracks `hidden` (e.g., ≥2× `hidden` rows, or report both the raw and width-normalized participation ratio side by side), then re-run the cheap `n_train=5000` sweep and this Phase 3 pair to confirm which of the two explanations in Finding 2 is correct before spending another multi-hour-to-multi-day budget on 1024-dim. The **bonus 1024-dim run offered as optional in this task was not started**, per this recommendation — Phase 3's own result is the reason not to, not a time-budget constraint (there was time remaining in this pass).

---

## Epistemic feedback loop verification (2026-07-11)

**Status:** Investigated end-to-end, wired in behind an opt-in flag, measured, tested, no regressions. This resolves the §3 open question ("present as headers — not verified wired into the D17 train/eval loop... the self-correcting wrapper gap is worth a follow-up ticket since Paper IV's κ=0.89 estimate assumes it's active"). Executed against a fresh `native/build_selfcorrect` scratch build (does not touch `native/build_math`, `bench/BASELINE_LOCK.json`, or the overnight orchestration scripts — verified clean via `git status --short` on those paths throughout). `native/build_scale`'s independent hidden-dim-scale confirmation run and `native/src/intelligence/measurers.*`/`cyphalm_bench_native.cpp`'s `--lstm-hidden` parsing were not touched.

### 1. What `self_correcting_infer.hpp`/`.cpp` and `epistemic_threshold.hpp`/`.cpp` actually do

Read end to end. The mechanism is exactly as advertised: `self_correcting_infer_at_h_impl` (`native/src/intelligence/self_correcting_infer.cpp:16-52`) runs one inference pass, computes `r_eu = 1 − confidence`, and while `EpistemicThreshold::should_correct(r_eu)` (`r_eu > nig_.mean()`, `epistemic_threshold.cpp:8-10`) is true, re-infers up to `max_passes` times with widened "deliberation" bounds (`deliberation_lo *= 0.82` floor `0.05`, `deliberation_hi += 0.05` ceiling `1.0`, `self_correcting_infer.cpp:36-37`), keeping the highest-confidence pass, then calls `threshold.update(r_eu, corrected)` to adapt the learned threshold (lower it after a helpful correction, raise it after a false-positive-shaped one, `epistemic_threshold.cpp:12-18`). This part of the doc's original suspicion — "is the mechanism itself real, or a stub" — is resolved: it is a real, working, correctly-implemented mechanism, not a stub.

### 2. Call-site trace — correcting an inaccuracy in the task's own premise

The task's brief (and, transitively, a prior grep this task inherited) asserted `self_correcting_infer`/`SelfCorrectingResult`/`EpistemicThreshold` are referenced in `cyphalm_model.cpp`, `intelligence_profiler.cpp`, `cyphalm_generation.cpp`, and `branch_a_router.cpp`. Re-grepping precisely for the actual symbols (not just the substring `"epistemic"`) shows **this was only true for one of the four**:

| File | Claimed to reference the wrapper | Actually does? |
|---|---|---|
| `cyphalm_generation.cpp` | yes | **yes** — but see below, it reimplements the LM-native analog inline, it doesn't call `self_correcting_infer()` |
| `intelligence_profiler.cpp` | yes | **no** — `intelligence_profiler.cpp:48`'s `LandscapeSystemClass::SelfCorrectingCypha` is an unrelated static reference-landscape enum case (a hardcoded Paper-IV comparison point for `kappa`), not a call into `self_correcting_infer.hpp`. Its other `epistemic_var`/`r_eu` references are the DIF-measured uncertainty, not this wrapper. |
| `branch_a_router.cpp` | yes | **no** — its `epistemic_threshold_` is a plain `double` router-config field feeding a simple `shannon_entropy(probs) > threshold` abstention check (`branch_a_router.cpp:270-382`); it does not use the `EpistemicThreshold` class or `self_correcting_infer` at all. (The real caller in this family is `native/apps/branch_a_rest_routes.cpp`, not `branch_a_router.cpp`.) |
| `cyphalm_model.cpp` (pre-this-change) | yes | **no** — zero references to `self_correct`, `SelfCorrectingResult`, or `EpistemicThreshold` existed before this pass; grep confirmed only unrelated `epistemic_var`/`compute_epistemic_ratio` usage (the DIF measurement feeding `use_reu_forget_gate`, a different, already-shipped Paper IV mechanism). |

The **real** call sites of the literal `self_correcting_infer()`/`self_correcting_infer_at_h()` functions are: `native/apps/cypha_rest.cpp:864` (a REST inference endpoint, opt-in via request-body `self_correct`), and two test/smoke tools (`intelligence_profiler_papers.cpp:137`, and `lm_self_correct_smoke.cpp`, which despite its name only exercises `EpistemicThreshold` directly). **None of these are on the D17 train/eval/bench path.** `cyphalm_generation.cpp` (`generate_decode`/`stream_generate`, the text-generation/decode path — not train/eval) has its own separate, LM-native reimplementation of the same algorithm (`self_correct_predict`, `cyphalm_generation.cpp:236-278`, using the same `EpistemicThreshold` class but operating on `CyphaLMModel`/`PredictNextOutput` directly instead of calling `self_correcting_infer_at_h`), gated by `DecodeParams::self_correct` which **defaults to `false`** (`cyphalm_generation.hpp:35`) and is never set by any D17 bench/train/eval call site.

**Architectural finding, not just a wiring gap:** `self_correcting_infer()`/`self_correcting_infer_at_h()` take a `cypha::CyphaInferModel&` (`infer_cpu.hpp:66`) — a classification model (labels + confidence, `CyphaDIF` + `VectorEncoder`) — not a `cypha::cyphalm::CyphaLMModel&` (the D17 char-LSTM LM, which returns per-token `log_probs`, not a label). These are different, incompatible types. This is why `cyphalm_generation.cpp` had to write its *own* parallel implementation rather than calling the header's functions directly — the literal wrapper cannot be called on the D17 model at all without an adapter. This matters for scoping the fix (§4 below): the "small, low-risk fix" is reusing the existing LM-native pattern in the D17 eval loop, not making `self_correcting_infer()` itself reach D17.

### 3. Direct evidence: `eval_bpc`/`accumulate_intelligence_profile` (pre-this-change)

The two functions that back every `cyphalm_bench_native --profile d17` BPC/κ/r_eu number call `predict_next(tok)` once per token and use the result as-is — `native/src/cyphalm/cyphalm_model.cpp` (pre-change) `eval_bpc` and `accumulate_intelligence_profile`, both a simple `for` loop with no `EpistemicThreshold`/self-correction of any kind. `cyphalm_bench_native.cpp`'s `main()` calls exactly these two functions (plus `train_sequence`, which also has zero self-correction references). **Conclusion: for the current default D17/math-integration bench profile, Paper IV's κ=0.89 estimate's "self-correcting wrapper active" assumption does NOT hold** — the wrapper exists in the codebase, is correctly implemented, and is active on the REST-serving and (opt-in, off-by-default) text-generation paths, but was completely unreached by the D17 train/eval/bench loop before this pass. This confirms the doc's §3 suspicion.

### 4. Fix implemented: opt-in `--use-self-correcting-loop`

Given the CyphaInferModel/CyphaLMModel type mismatch above, the correct low-risk fix is not "call `self_correcting_infer()` from `eval_bpc`" (wrong type, would need a non-trivial adapter) but to bring the *already-proven* LM-native pattern (`cyphalm_generation.cpp`'s `self_correct_predict`, which has been exercising this exact algorithm on `CyphaLMModel` since it was written) into the eval loop, self-contained and independent of `cyphalm_generation.cpp` (zero changes to that file, zero regression risk there):

- `cyphalm_config.hpp`: new `bool use_self_correcting_loop = false;` (default off — does not change the locked D17 baseline unless explicitly requested).
- `cyphalm_model.hpp`/`.cpp`: new private `CyphaLMModel::self_correct_if_needed(initial, threshold)` — computes live `r_eu` via the same `compute_epistemic_ratio` already used elsewhere in this file; if `r_eu` exceeds the threshold and `context_mode == Hybrid` (D17's mode), re-blends the cached hybrid GRIA/LSTM logits via the existing `repredict_hybrid_blend` (no extra forward pass) at a progressively LSTM-shifted blend for up to 3 passes, keeping the highest-confidence result — algorithmically identical to `self_correct_predict`, written independently so `cyphalm_generation.cpp` is untouched.
- `eval_bpc`/`accumulate_intelligence_profile`: when `cfg_.use_self_correcting_loop`, route each `predict_next(tok)` result through `self_correct_if_needed` before scoring/observing, using one `EpistemicThreshold(0.5, 5.0)` instance per eval pass (mirrors the prior `(0.5, 5.0)` default used at every other call site: `cypha_rest.cpp`, `shell_main.cpp`, `intelligence_profiler_papers.cpp`).
- `cyphalm_bench_native.cpp`: new `--use-self-correcting-loop` CLI flag, applied unconditionally (same rationale as `--lstm-hidden`: a verification flag, should work without also requiring `--math-integration`), echoed into the output JSON as `use_self_correcting_loop` for traceability.

### 5. Measured effect at `--profile d17 --n-train 5000 --n-eval 256 --intelligence-profile --bench-seed 42`

| | bpc | `criticality_score` (κ) | r_eu | `lstm_hidden_d_eff` |
|---|---|---|---|---|
| flag off (baseline) | 4.039556 | 0.855927 | 0.168189 | 0.270968 |
| flag on | 4.039556 | 0.855927 | 0.168189 | 0.270968 |

**Bit-for-bit identical** — confirmed with a full-file diff of the two output JSONs (only the echoed `use_self_correcting_loop` field itself differs). This is a real, verified null result, not a wiring failure: `EpistemicThreshold`'s prior mean is `0.5` (`epistemic_threshold.cpp:5`, `prior_mu=0.5` default used at every existing call site including this new one), and this profile's measured `r_eu` is chronically low — the profile JSON's own `failure_modes.low_r_eu: true` and the `r_eu` critical target gap (`point: 0.168` vs. `critical_target: 0.7`) show the model's live epistemic ratio never gets close to `0.5` at any point in this 256-token eval pass, so `threshold.should_correct(r_eu)` never returns true and the correction loop never fires a single time (confirmed directly with temporary `std::cerr` instrumentation during development, since removed — every one of 256 tokens logged `trigger=0`). Since the threshold never fires, it also never adapts away from its `0.5` prior, so this null result is stable, not a transient cold-start artifact — it would not resolve itself with a longer eval window at this training scale.

### 6. Test results

`ctest --test-dir native/build_selfcorrect -R "self_correct|epistemic|intelligence"` (7 matching tests, after building `intelligence_profiler_smoke`, `intelligence_profiler_papers`, `cypha_intelligence_bench`, `cypha_bench_run`, `intelligence_lm_monitor_smoke`, `lm_self_correct_smoke`, `cyphalm_bench_native`):

```
100% tests passed, 0 tests failed out of 7
  native_intelligence_profiler_smoke .......... Passed
  native_intelligence_profiler_papers .......... Passed
  native_intelligence_bench_smoke .............. Passed
  native_cyphalm_bench_intelligence_profile .... Passed
  native_d39_intelligence_monitor_smoke ........ Passed
  native_intelligence_lm_monitor_smoke ......... Passed
  native_lm_self_correct_smoke ................. Passed
```

No regressions. The flag-off path was also directly confirmed bit-identical to this document's own recorded Phase 2b numbers (§ above: `bpc=4.039556`, `criticality_score=0.855927`, `lstm_hidden_d_eff=0.270968` at hidden=128) both before and after this change, so the locked D17/math-integration baseline (`bench/BASELINE_REPORT.md`, `bench/BASELINE_LOCK.json` — neither touched by this pass) is unaffected by construction, not just by flag default.

### 7. Answer to the open question

**Is the epistemic feedback loop active for the current default D17/math-integration profile, and does Paper IV's κ=0.89 assumption hold?**

- **Before this pass: no.** The wrapper existed, was correctly implemented, and was active on the REST-serving path and the (default-off) text-generation path, but was never reached by D17 train/eval/bench — Paper IV's "self-correcting wrapper active" assumption did not hold for the profile its own κ=0.89 estimate is meant to describe.
- **After this pass: wired in, opt-in, verified inert at default settings.** `--use-self-correcting-loop` now makes the D17 eval/intelligence-profile path exercise the same algorithm as the generation path, with zero effect on the existing default behavior (bit-identical output when the flag is absent, confirmed above and by the unchanged 7-test suite). Turning the flag *on* at the current default `n_train=5000` medium-tier profile, current `lstm_hidden=128`, and math-integration's default r_eu regime produces **no measurable change** in `bpc`/κ/r_eu, because live `r_eu` never crosses even the lenient `0.5` trigger threshold at this scale — the model is comfortably within its "confident enough" regime throughout this eval window (consistent with `failure_modes.low_r_eu: true` already being flagged in every prior profile run in this document).
- **Practical implication for Paper IV's κ=0.89 estimate:** the estimate's assumption is now *technically satisfiable* (the mechanism can be activated), but activating it changes nothing at the scale/profile measured so far, so it cannot be credited with (or blamed for) any part of the existing κ numbers in this document or in `BASELINE_LOCK.json`. Whether it would engage at other scales (e.g. earlier in training when the model is less confident, or at higher `lstm_hidden` where the Phase 2b/Phase 3 results above show `d_eff`/representational behavior is itself still not fully understood) is open — this pass answered "is it wired and safe," not "does it ever fire in some other regime," which would need a dedicated sweep over training step / `r_eu` trajectory and is a natural next follow-up but out of scope here.

---

## §Phase 3 follow-up resolved (2026-07-12): history-buffer sampling fix

**Status:** Root cause fixed and verified against the cheap control sweep, in a fresh `native/build_deff` scratch build (HEAD at/after `8833611`; `native/build_math`, `bench/BASELINE_LOCK.json`, `bench/BASELINE_REPORT.md`, and the overnight orchestration scripts were not touched — confirmed via `git status --short` throughout, and this fix's own commit only includes `native/` source + this doc). This is the exact follow-up Finding 2 (above) called for: "scale `kLstmHiddenHistoryMax` ... then re-run the cheap `n_train=5000` sweep and this Phase 3 pair to confirm which of the two explanations in Finding 2 is correct."

### 1. Mechanism

Read `lstm_h_history_rows_`/`kLstmHiddenHistoryMax`/`lstm_hidden_d_eff()` in `native/include/cypha/cyphalm/cyphalm_model.hpp` and `native/src/cyphalm/cyphalm_model.cpp`, and `compute_participation_ratio` in `native/src/intelligence/measurers.cpp` end to end to confirm Finding 2's diagnosis exactly:

- `append_lstm_hidden_history()` is called once per training step (`CyphaLMModel::train_step`, hybrid-mode LSTM path), unconditionally recording the current LSTM hidden state `h` into a ring buffer, evicting the oldest row once the cap is reached. There is no subsampling or step-interval gating — every single training step's hidden state either enters the buffer or evicts the oldest one, so the buffer always holds exactly the *most recent* `min(step_count, cap)` hidden states.
- `lstm_hidden_d_eff()` flattens the buffer into an `n_samples x n_dims` matrix (`n_samples` = buffer row count, `n_dims` = `lstm_->hidden`) and calls `compute_participation_ratio(..., VarianceProxy or CovarianceEigenvalue)`, which computes `(Σλ)² / Σλ²` from the sample covariance/variance and **divides by `n_dims`** to produce a `[0,1]`-normalized ratio. The pre-fix bug was structural, not a math error in that formula: the cap feeding `n_samples` was a compile-time constant (`kLstmHiddenHistoryMax = 48`) completely decoupled from `n_dims` (`lstm_hidden`), so the *statistical power* of the estimate (how well `n_samples` rows can resolve an `n_dims`-dimensional covariance structure) silently degraded as `lstm_hidden` grew — exactly Finding 2's "explanation 1" (undersampling artifact), confirmed as the operative mechanism by the control-sweep result in §2 below (the fix alone, with no change to training dynamics, restores the expected monotonic trend).
- `compute_participation_ratio` (Phase 0 fix, commit `ff26a57`) already had three methods (`VarianceProxy`, `CovarianceEigenvalue` with a `TraceFrobenius` fallback above 256 dims) but all three shared the same undersampling exposure, since all three consumed the same 48-row history and all three discarded the *unnormalized* `(Σλ)²/Σλ²` effective-dimension count by dividing by `n_dims` before returning — there was no way for a caller to tell a well-powered measurement from an underpowered one without independently re-deriving `n_samples`/`n_dims` from source.

### 2. Fix implemented

- **History cap now scales with `lstm_hidden`:** `CyphaLMModel`'s constructor sets a new per-instance member `lstm_h_history_max_ = std::max(48, 2 * cfg_.lstm_hidden)` (replacing the old `static constexpr kLstmHiddenHistoryMax = 48`), targeting this document's own recommended "≥2× `hidden` rows" floor uniformly across all `lstm_hidden` values, with `48` retained as a floor for tiny/degenerate configs. This is a **per-instance**, not compile-time, constant because it depends on `cfg_`, which is only available at construction. Concretely: hidden=128 → 256 rows (sample_ratio 2.0, changed from 48/0.375 pre-fix — intentional, documented, not preserved bit-for-bit, since 48 was never derived from anything hidden=128-specific to begin with); hidden=256 → 512 rows; hidden=512 → 1024 rows (sample_ratio 2.0, up from 48/0.094 pre-fix — the specific regime Finding 2 flagged as severely underdetermined).
- **Ring buffer switched from `std::vector` to `std::deque`:** eviction (`pop_front()`) is now O(1) instead of O(current row count). This matters because the row count itself now scales with `lstm_hidden` (up to ~1000+ rows at hidden=512) and eviction happens once per training step across potentially 300k+ steps — an O(rows)-per-step cost would have compounded into a real, avoidable slowdown at exactly the scale this fix targets.
- **Memory footprint, checked not assumed:** at hidden=512, cap=1024 rows × 512 `double`s/row × 8 bytes = **4,194,304 bytes (4 MiB)** of raw sample data, plus ~1024 `std::vector` object headers (~24–32 bytes each on this 64-bit MinGW target) ≈ 32 KiB of bookkeeping overhead — trivial relative to the model's other buffers (embedding table, GRIA low-rank factors, etc.), confirming the "should be trivial" expectation rather than assuming it.
- **Raw effective-dimension count exposed alongside the normalized ratio** (the doc's suggested secondary safeguard): `measurers.{hpp,cpp}` gained `compute_participation_ratio_raw(...)`, returning the unnormalized `(Σλ)²/Σλ²` in `[0, n_dims]` that the existing `compute_participation_ratio` was computing internally and then discarding via the `/ n_dims` division. `CyphaLMModel` gained `LstmHiddenDEffReport lstm_hidden_d_eff_detail()` (fields: `normalized`, `raw`, `sample_ratio`, `n_samples`, `n_dims`); the existing scalar `lstm_hidden_d_eff_report()` is unchanged (still returns just `normalized`, for backward compatibility with any existing consumers). `cyphalm_bench_native`'s JSON output now includes `lstm_hidden_d_eff_raw`, `lstm_hidden_d_eff_sample_ratio`, and `lstm_hidden_d_eff_n_samples` alongside the pre-existing `lstm_hidden_d_eff` and `lstm_hidden_d_eff_method`, so a future reader can tell at a glance whether a given `lstm_hidden_d_eff` value was well-powered without re-deriving it from `lstm_hidden` and a hardcoded constant.

### 3. Control-sweep result (`n_train=5000`, same command as Phase 2b, fresh `native/build_deff`)

```powershell
native/build_deff/cyphalm_bench_native.exe --profile d17 --n-train 5000 --n-eval 256 `
    --lstm-hidden {128|256|512} --intelligence-profile --bench-seed 42
```

| hidden | `lstm_hidden_d_eff` (post-fix) | `lstm_hidden_d_eff` (Phase 2b, pre-fix) | `lstm_hidden_d_eff_raw` | `sample_ratio` | `n_samples` | bpc |
|---|---|---|---|---|---|---|
| 128 | **0.278387** | 0.270968 | 35.634 | 2.0 | 256 | 4.039556 |
| 256 | **0.503581** | ~0.485* | 128.917 | 2.0 | 512 | 3.929983 |
| 512 | **0.595738** | 0.576461 | 305.018 | 2.0 | 1024 | 3.848554 |

(*Phase 2b's hidden=256 value is quoted from this document's Finding 2 narrative ("0.271 (h=128) → 0.485 (h=256) → 0.576 (h=512)"), which did not carry the same full-precision figure as the 128/512 endpoints elsewhere in this document; not independently re-verified bit-for-bit here.)

**Still climbs monotonically 128→256→512 post-fix** (0.278 → 0.504 → 0.596), same direction as Phase 2b, confirming the fix does not disturb the already-correct medium-tier (`n_train=5000`) trend — this was the control: at this scale, 48 rows was already a reasonable (if not ideal) sample count relative to `n_dims`, so no dramatic change was expected, and none occurred (the individual values shift somewhat, expected since the sample count itself changed from 48 to 256/512/1024, pulling in a longer, more representative slice of training history — the *direction* and *monotonicity* are what matter for this check, and both hold). `sample_ratio=2.0` at every hidden size confirms the fix's intended uniform statistical-power floor is actually being achieved in practice, not just in the constructor formula.

### 4. Production-scale (`n_train=300000`) re-confirmation: **hidden=128 complete; hidden=512 still in progress**

Both Phase 3 production-scale points were re-launched under the fix, same exact command/seed as the original Phase 3 pass, in the same fresh `native/build_deff`, with a retry watchdog (per this document's own "-1/empty-output → retry" guidance):

```powershell
native/build_deff/cyphalm_bench_native.exe --profile d17 --n-train 300000 --n-eval 2000 `
    --lstm-hidden {128|512} --intelligence-profile --threads 1 --bench-seed 42
```

Both were started at 2026-07-12T14:01 local time under the same directly-observed heavy machine contention noted throughout this document (sibling `cyphalm_bench_native.exe`/other native-benchmark processes competing for the 64 cores).

**hidden=128 finished on its first attempt** (no `-1`/empty-output retries needed) at 2026-07-12T14:58, i.e. ≈57 min wall-clock — close to the ≈52 min clean estimate, so contention on this pass was comparatively mild for this point:

| hidden | `lstm_hidden_d_eff` (post-fix, 300k) | `lstm_hidden_d_eff` (Phase 3, pre-fix, 300k) | `lstm_hidden_d_eff_raw` | `sample_ratio` (post-fix) | `sample_ratio` (pre-fix) | `n_samples` (post-fix) | bpc |
|---|---|---|---|---|---|---|---|
| 128 | **0.693926** | 0.600 | 88.823 | 2.0 | 0.375 | 256 | 3.044487 |
| 512 | *pending* | 0.331 | *pending* | 2.0 (target) | 0.094 | 1024 (target) | *pending* |

hidden=128's post-fix value (0.694) is higher than its pre-fix value (0.600), consistent with removing an undersampling-driven downward bias even at the point that was already the *better-sampled* of the two pre-fix (`sample_ratio` 0.375, vs. 0.094 at hidden=512) — i.e. the fix moves the metric in the expected direction at both ends, not just at the severely-underdetermined one, which is itself a useful sanity check on the mechanism (§1–2), independent of what hidden=512 turns out to show.

**hidden=512 is still on its first attempt as of this writing.** Note on process supervision, for anyone checking this later: an earlier pass of this run was found to have a live benchmark process but a *dead* supervisory watchdog (the outer PowerShell retry-loop had exited while its child `cyphalm_bench_native.exe` kept running as an orphan) — meaning if that attempt had died to contention, nothing would have retried it. This was caught and corrected: the unsupervised run was stopped and relaunched as a **genuinely detached** process using `Start-Process -FilePath powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File', '<watchdog script path>')` (no `-Wait` on the outer call), with the retry-loop logic itself living in a standalone script file at `bench/results/hidden_dim_scale_deff/watchdog_300k_hidden512.ps1` rather than inline. This was verified independent of the launching session by confirming the watchdog process (and its benchmark child) both remained alive and `Responding: True` via `Get-Process` *after* the immediate launcher process had already exited — i.e. it survives its own parent, which is the actual test for true OS-level detachment on Windows (child/orphan processes are reparented, not killed, when an ancestor exits).

**How to check progress later:**
- Process alive: `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "watchdog_300k_hidden512|build_deff.*lstm-hidden.*512" }` — should show the `powershell.exe` watchdog host plus its `cyphalm_bench_native.exe` child.
- Watchdog/attempt history: `bench/results/hidden_dim_scale_deff/300k_hidden512_watchdog.log` — one `attempt N starting ...` / `attempt N exit=... outlen=...` line pair per attempt, `FINISHED ok=True ...` on success.
- Final result: `bench/results/hidden_dim_scale_deff/300k_hidden512.json` — non-empty only once an attempt succeeds (`exit=0` and `outlen>0` in the log).

Given this document's own prior observation that this exact point took 16h39min under comparable contention pre-fix (vs. a 5–9.5h clean estimate), it should be expected to run for several more hours at minimum. **This document is being committed with the hidden=512 leg of the production re-confirmation explicitly incomplete, per this task's own instruction not to guess or extrapolate a result not actually measured.** The mechanism fix itself (§1–2), the control-sweep confirmation (§3), and now the hidden=128 production point above are complete and not contingent on hidden=512 finishing. What remains strictly open is whether `lstm_hidden_d_eff` at hidden=512/300k-steps, once measured without the severe undersampling artifact (`sample_ratio` 0.094→2.0, the single largest statistical-power change of the two points), lands **above** hidden=128's post-fix value of 0.693926 (supporting Finding 2's "was purely a sampling artifact" explanation and resolving the inversion) or still lands **at or below** it (which would instead support Finding 2's second, non-exclusive explanation — genuine representational compression at scale, an authentic finding about training dynamics rather than a measurement bug). **Do not treat either outcome as already known.** Check the JSON file above for the actual completed number before drawing that conclusion; if this document has not been updated with that value filled into the table above, treat the hidden=512 re-confirmation as still-pending follow-up work.

### 5. Updated go/no-go recommendation on Phase 4 (1024-dim)

**Still no-go, pending the hidden=512 leg of the production re-confirmation in §4 — upgraded from "no-go, sampling gap unaddressed" to "no-go, sampling gap addressed, one of two production points re-confirmed, awaiting the second (higher-value) measurement."** The original Phase 3 no-go had three legs (§ above): (1) the `lstm_hidden_d_eff` inversion itself, (2) wall-clock cost far above estimate, (3) BPC/kappa alone being weaker, non-`D_eff`-specific evidence. This pass resolves the *measurement* half of leg (1) for both points (the statistic is no longer known to be underpowered at either hidden size, `sample_ratio=2.0` uniformly) and resolves the *empirical* half for the hidden=128 point specifically (0.600→0.694, moved in the expected direction). It does **not yet** resolve the empirical half for hidden=512, which is the point that actually determines whether the inversion (0.600→0.331 pre-fix) is resolved, since that requires comparing hidden=512's post-fix value against hidden=128's post-fix value (0.693926) once it exists. Legs (2) and (3) are unchanged by this pass. **Do not schedule Phase 4 until hidden=512's post-fix value is recorded in the §4 table above** — if it comes in above 0.693926, that resolves the specific concern this plan is built around and makes a Phase 4 go-decision worth revisiting on its own (unchanged) cost/contention merits; if it does not, that is itself a significant, novel finding (genuine width-independent representational compression at scale) that would argue against Phase 4 even more strongly than the original inversion did, since it would no longer be attributable to a measurement bug.
