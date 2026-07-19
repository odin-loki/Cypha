# Stub audit — `ReversibleSSMCell::backward_stub` and `EwcRegularizer` diagonal Fisher (2026-07-11)

**Scope:** Follow-up on the two "known stub functions (not urgent, just noted)" from
`DEV_PLAN_2026-07-11.md` §2.4: `reversible_ssm_cell.hpp:7-16` (`backward_stub`) and
`ewc_regularizer.hpp:11` (diagonal Fisher stub). Minor cleanup pass, not a feature project.

**HEAD at start:** `e951f73`. Fresh build dir: `native/build_stubs` (Ninja, `-DCMAKE_BUILD_TYPE=Release`,
MinGW/Strawberry `c++.exe`). Did not touch `native/build_math`, `native/build_scale`,
`bench/BASELINE_LOCK.json`, overnight orchestration scripts, `native/src/intelligence/measurers.*`,
`--lstm-hidden` parsing, or `self_correcting_infer.*`/`epistemic_threshold.*` (a sibling agent was
concurrently editing `epistemic_threshold`-adjacent code in `cyphalm_config.hpp`/`cyphalm_model.hpp`/
`cyphalm_bench_native.cpp` during this session — confirmed via `git diff` and left untouched).

**Bottom line:**
- **Part A (`ReversibleSSMCell`):** the "stub" name was stale, not the math. `backward_stub()` is an
  *exact* analytic inverse of `forward()`, already correctly implemented. Renamed to `reconstruct()`,
  fixed the docstrings, added a regression test. No behavior change.
- **Part B (`EwcRegularizer`):** the crude `F_i ≈ anchor_i²` proxy is **live** — it's wired into both
  the D16 bench domain (`16B_forgetting_resistance`, `16H_ewc_overlay_smoke`) *and* the REST API's
  `POST /train` (`ewc_snapshot`/`ewc_lambda`). Implemented a real diagonal-Fisher estimate
  (`EwcRegularizer::snapshot_calibrated`) using this codebase's existing closed-form update math (no
  new autodiff), gated behind `CYPHA_D16_REAL_FISHER=1` and wired only into the D16 bench domain's
  forgetting probe. The REST endpoint and every other existing caller are untouched — default
  behavior is unchanged everywhere unless that env var is set.

---

## Part A — `ReversibleSSMCell` / H11

### 1. Is the math actually correct?

Forward (`native/src/cyphalm/reversible_ssm_cell.cpp`):

```
y_i = x_i + tanh(delta_i)
```

`delta` is cached **verbatim** into `last_delta_` (not derived from anything else), alongside `x` and
`y`. The (renamed) reconstruct step:

```
x_hat_i = y_i - tanh(delta_i)
        = (x_i + tanh(delta_i)) - tanh(delta_i)
        = x_i
```

This is an **exact** algebraic identity — not an approximation, and not "numerically incomplete."
There is no missing piece: `delta` never needs to be reconstructed from `y` or anything else (which
*would* be underdetermined/lossy, since `tanh` isn't invertible from `y` alone without also knowing
`x`) — it's simply cached from the same `forward()` call and read back. The only source of error is
ordinary floating-point rounding in the two `+`/`-` operations (not from re-evaluating `tanh`, which
happens exactly once, in `forward()`), confirmed empirically to be within `1e-9` absolute across
normal, saturated (`delta = ±100`), zero-padded (`delta` shorter than `x`), and sequential-call cases
— see `native/tools/reversible_ssm_cell_smoke.cpp`.

**Verdict: the "stub" name/comment was stale/misleading. There is no genuine subtlety being missed.**
Renamed `backward_stub()` → `reconstruct()`; updated the class docstring and the H11 sweep description
(`cypha_cell_hypothesis.cpp`) to say "exact analytic backward reconstruct" instead of "backward
reconstruct stub."

### 2. Call sites — is the reconstruction actually used?

Grepped `native/src/**`, `native/tools/**`, `native/tests/**` for `ReversibleSSMCell`, `backward_stub`,
`H11`, and the config flag it's actually gated by (`use_reversible_cell`, not the class name directly —
`cyphalm_generation.cpp` and `branch_a_router.cpp` do **not** reference this class at all, contrary to
what a prior grep apparently suggested; the only file that constructs/calls it is
`native/src/cyphalm/cyphalm_model.cpp`):

- **Construction:** `cyphalm_model.cpp:318`, gated by `cfg_.use_reversible_cell`.
- **`forward()`:** `cyphalm_model.cpp:547-550`, inside the SSM context pipeline — this *is* on the live
  numerics path when H11 is selected (it replaces `ctx` with `x + tanh(delta)`).
- **`reconstruct()` (was `backward_stub()`):** `cyphalm_model.cpp:1132-1134`:
  ```cpp
  if (cfg_.use_reversible_cell && reversible_cell_ && reversible_cell_->has_pair()) {
      (void)reversible_cell_->reconstruct();
  }
  ```
  **The call happens, but its return value is immediately discarded.** `reconstruct()` is a `const`
  method that only reads the cell's own private cached vectors and has no side effects, so this line
  is provably a no-op beyond the (cheap) computation itself — there is no gradient-checkpointing /
  memory-savings recompute-on-backward consumer anywhere downstream. In effect: **called, but dead —
  the reconstruction path exists and is exercised, but nothing uses its output.** Added a code comment
  at the call site explaining this so a future reader doesn't assume it's load-bearing.

### 3. Is H11 in the standard cell sweep, and does it work?

- `cypha_cell_hypothesis.cpp`'s variant table includes `H11` (tier 2, `runnable=true`), and the sweep's
  `should_run()` includes every runnable variant when `--overnight-sweep` is passed with no
  `--cell-variant` filter — **H11 is in the standard 25-variant sweep**, not excluded.
- `bench/BASELINE_LOCK.json`'s `cell_sweep_results` only records aggregate numbers (`variant_count: 25`,
  best BPC `2.864`) from the 2026-06-17 production run — it does not break results down per-variant, so
  H11's specific historical BPC isn't recoverable from the lock. No `H11` mentions exist in any
  `bench/results/*.log` currently in the repo.
- **Ran H11 standalone** (`cypha_cell_hypothesis_sweep.exe --cell-variant H11 --n-train N`) in the fresh
  `native/build_stubs` to check current behavior:

  | `n_train` | `bpc` | Notes |
  |---|---|---|
  | 200 | 43.17 (finite) | undertrained, as expected |
  | 2000 | `null` (NaN) | diverges |

  **H11 has a pre-existing numerical-divergence issue independent of this rename** — it diverges
  somewhere between 200 and 2000 training steps, similar in flavor to H15's documented divergence
  (`DEV_PLAN_2026-07-11.md` §1.6) but a different variant/mechanism. This is **not** caused by the
  `backward_stub`→`reconstruct` rename: `reconstruct()`'s result is discarded at its only call site and
  it mutates no shared state, so it cannot influence training numerics regardless of what it's named or
  how it's implemented. Untangling *why* H11 diverges at this scale is a separate numerical-stability
  question (candidate causes: the `x + tanh(delta)` coupling saturating and starving gradient signal
  back into the SSM path it wraps) and is out of scope for this cleanup pass — flagging it here as a
  finding worth a future targeted look, analogous to H15.

### 4. Changes made

- `native/include/cypha/cyphalm/reversible_ssm_cell.hpp` / `.cpp`: renamed `backward_stub()` →
  `reconstruct()`, rewrote the docstrings/comments to state the exact-inverse derivation instead of
  calling it a stub.
- `native/src/cyphalm/cyphalm_model.cpp`: updated the call site + added a comment explaining the
  discarded-return-value/dead-compute situation.
- `native/src/cyphalm/cypha_cell_hypothesis.cpp`: updated H11's `notes` string.
- New regression test: `native/tools/reversible_ssm_cell_smoke.cpp` (registered as ctest
  `native_reversible_ssm_cell_smoke`) — covers the basic round trip, saturated deltas, zero-padded
  short `delta`, `reset()`, and sequential `forward()` calls, asserting `reconstruct()` recovers the
  original `x` within `1e-9`.

**No functional/behavioral change** — this is a pure rename + doc fix + test addition.

---

## Part B — `EwcRegularizer` diagonal Fisher

### 1. What the stub actually computes

`native/src/ewc_regularizer.cpp`'s `build_diagonal_fisher()` set `F_i = anchor_i² + ε`, i.e. the
squared **parameter value** at snapshot time, not `F_i = E[(∂ log p/∂θ_i)²]` (squared **gradient**,
the standard diagonal-Fisher EWC formula). These are unrelated quantities: a parameter with a large
value but a near-zero gradient (i.e. not contributing to the loss around the anchor) would be
*wrongly* treated as highly important under the stub, while a small-valued but sensitive parameter
would be under-protected. Confirmed this isn't merely "simplified" but structurally the wrong formula.

### 2. Where it's used

Grepped `native/src/**` / `native/tools/**` / `native/apps/**` for `EwcRegularizer` (careful to
distinguish it from the unrelated, separately-implemented `CyphaLMEwcRegularizer` /
`HybridEwcRegularizer` in `cyphalm_ewc_regularizer.hpp/.cpp`, which is a different class entirely and
out of scope here). `cypha::EwcRegularizer` (the audited class) is used in:

- **`native/src/bench/bench_domains.cpp`** — D16 domain, two live experiments:
  - `16B_forgetting_resistance` (`run_d16_ewc_probe`): trains iris → snapshot → trains wine/digits →
    re-evaluates iris, comparing `forgetting_score` baseline-vs-EWC, reporting
    `ewc_reduces_forgetting` / `forgetting_delta`. **This is exactly the kind of catastrophic-forgetting
    comparison metric the task asked about**, and it already existed.
  - `16H_ewc_overlay_smoke`: snapshots *before* any training (mem is empty), so `anchor_D_` ends up
    size-0 while `mem.D` later grows — `penalty()`/`apply_pull()` both guard on size match and
    no-op when sizes differ, meaning this particular smoke test only ever exercises the `enc_w` pull,
    never the `D` pull. Pre-existing behavior, unrelated to the Fisher formula itself; left as-is
    (not a Part B target — real-Fisher calibration requires post-training data to calibrate against,
    which doesn't exist before training starts).
- **`native/apps/cypha_rest.cpp`** (`g_ewc`) — the REST server's `POST /train` endpoint: a client can
  set `"ewc_snapshot": true` and `"ewc_lambda": <λ>` in the request body to snapshot/activate EWC live,
  in production request-serving code. **This is a genuinely live, user-facing path**, not just a bench
  domain.
- **`native/tools/ewc_smoke.cpp`, `native/tools/ewc_d16b_smoke.cpp`** — existing regression tests
  exercising `snapshot()`/`penalty()`/`apply_pull()` (unchanged, still pass).

**Verdict: EWC is live, in two places.** Per the task's decision tree this requires an actual fix, not
just a documentation correction.

### 3. What a real diagonal Fisher looks like here, and why it doesn't need new autodiff

`EwcRegularizer` overlays two parameter groups: `D` (per-class deltas, `CyphaDifMemoryState`) and
`enc_w` (encoder projection). Neither is trained via a generic autodiff/backprop pass — both use
hand-derived closed-form update rules already in this codebase, and in both cases the update rule
*is* (up to a learning-rate factor) the gradient of a loss this codebase already computes explicitly:

1. **`D`:** `CyphaDifMemoryState::memory_train` (`native/src/memory_train.cpp`) returns the scalar loss
   `-log_norm + 0.5·r·h_mu0 - cross_k + 0.5·d_sq_k`, where (for the true-label row `k`)
   `cross_k = Σ_j D[k,j]·r[j]` and `d_sq_k = Σ_j D[k,j]²·world_inv_v[j]`, `r[j] = (h[j]-mu0[j])·world_inv_v[j]`.
   Only row `k` appears in the loss, so `∂loss/∂D[k,j] = D[k,j]·world_inv_v[j] - r[j]`, and `0` for
   every other row — matching the codebase's own per-step loss-attribution convention.
   `mu0 = world_mu` exactly for every model `EwcRegularizer` is used against, because `f_field` is
   all-zero at creation (`create_model.cpp`) and is **never mutated** anywhere in `native/src/*.cpp`
   (grepped for assignments — only ever read) — so the field-correction term is provably zero here, not
   just ignored as an approximation.
2. **`enc_w`:** `contrastive_update_encoder_w` (`native/src/encoder_contrastive.cpp`) is already
   *documented* as a "Fisher-Rao-style encoder update": `W[i,j] += lr·weight·diff[i]·f[j]` where
   `diff = r_pred - r_true` are Fisher-Rao residuals `(h-mu)/max(v,ε)`. The per-step contribution
   before the `lr` scale — `weight·diff[i]·f[j]` — *is* `∂loss/∂W[i,j]`, and it's zero for
   correctly-classified samples (no update fires), which is the correct behavior for a gradient, not
   an approximation of one.

`EwcRegularizer::snapshot_calibrated(mem, infer, calib_x, calib_labels)` (new method, additive — the
existing `snapshot()` is completely untouched and still the default for every existing caller) reuses
exactly these two formulas: for each calibration sample it recomputes `r`/`cross`/`d_sq` read-only
(mirroring `memory_train`'s forward math without mutating any state), derives `∂loss/∂D[k,j]` and,
when the sample would trigger an encoder update (`pred != true`, the same condition
`dif_train_step_vector` uses), `∂loss/∂W[i,j]`, squares and accumulates both across the batch, and
divides by the sample count. Falls back to the old anchor-squared behavior if the calibration batch is
empty (documented, tested contract). No new automatic differentiation was introduced — every
derivative is the closed form already implied by hand-written update rules elsewhere in this file.

### 4. Wiring — opt-in, default-off

Wired into `run_d16_ewc_probe`'s `16B_forgetting_resistance` probe only, gated by
`CYPHA_D16_REAL_FISHER=1` (checked via `d16_real_fisher_enabled()`, same `std::getenv` convention as
`CYPHA_D14_KERNEL_BASIS` / `CYPHA_D03_VIEW_SCHEDULE` etc. elsewhere in `bench_domains.cpp`). When set,
the probe calibrates on the just-trained iris task's own training set (the theoretically correct
choice — Fisher is meant to be estimated on the anchor task's data, at the anchor point) instead of the
squared-anchor proxy, and reports `ewc_real_fisher: true/false` in its JSON output. **Off by default —
every existing D16 result and the REST `/train` endpoint are byte-for-byte unaffected** unless a caller
explicitly opts in.

### 5. Measured effect (small scale)

Ran the new `native/tools/ewc_real_fisher_smoke.cpp` (mirrors `ewc_d16b_smoke.cpp`'s synthetic 3-task
setup) comparing `snapshot()` vs. `snapshot_calibrated()` on identical post-training state:

```
penalty_anchor=0.0220813   penalty_real_fisher=1.7707e-09   (differ=true)
```

The real-Fisher penalty came out **~7 orders of magnitude smaller** than the anchor-squared one on
this data. This is expected, not a bug: right after fully training on a task, the model is near a local
optimum for it, so `∂loss/∂θ` is near zero there — which is exactly what real Fisher information
should show (low curvature/importance once converged), whereas the anchor-squared proxy has no
relationship to convergence at all and is arbitrary. **Practical implication for anyone enabling
`CYPHA_D16_REAL_FISHER`:** `ewc_lambda` was tuned (0.25–0.5 in existing call sites) against the old
proxy's much larger magnitude; a real-Fisher deployment would likely need its own `ewc_lambda`
re-tuning to get a comparable pull strength, rather than assuming the same λ carries over. This is
noted here rather than auto-tuned, since re-tuning λ against a production forgetting metric is beyond
this cleanup pass's scope.

### 6. Changes made

- `native/include/cypha/ewc_regularizer.hpp` / `native/src/ewc_regularizer.cpp`: added
  `EwcRegularizer::snapshot_calibrated(mem, infer, calib_x, calib_labels)` (real diagonal Fisher via
  calibration-batch squared gradients, derived as above); corrected the class-level docstring to
  clearly distinguish the legacy anchor-squared proxy (still `snapshot()`'s behavior, kept for the REST
  endpoint and every other existing caller) from the new calibrated estimate.
- `native/src/bench/bench_domains.cpp`: added `d16_real_fisher_enabled()` env-gate helper; wired
  `snapshot_calibrated` into `run_d16_ewc_probe`'s EWC arm behind `CYPHA_D16_REAL_FISHER=1`; added
  `ewc_real_fisher` to that experiment's JSON output.
- New regression test: `native/tools/ewc_real_fisher_smoke.cpp` (ctest `native_ewc_real_fisher_smoke`)
  — asserts both `snapshot()` and `snapshot_calibrated()` produce zero penalty immediately after
  snapshot, that the empty-calibration-batch fallback exactly reproduces legacy behavior, and that the
  real-Fisher and anchor-squared penalties are genuinely different (not a no-op rename) after further
  training.

---

## Test results

`ctest --test-dir native/build_stubs -R "reversible|ewc|ssm_cell"` (after building the affected
targets): **7/7 passed** —

```
native_ewc_smoke .................... Passed
native_ewc_real_fisher_smoke ........ Passed
native_ewc_d16b_smoke ................ Passed
native_ewc_cyphalm_smoke ............. Passed
native_ewc_hybrid_smoke .............. Passed
native_ewc_weights_smoke ............. Passed
native_reversible_ssm_cell_smoke ..... Passed
```

Also re-ran the cell-hypothesis smoke suite (touched `cypha_cell_hypothesis.cpp`'s H11 description
string): `native_cell_hypothesis_sweep_smoke`, `native_cell_hypothesis_tier2_smoke`,
`native_cell_hypothesis_tier3_smoke` — **3/3 passed**.

**Note on side effects avoided:** an initial exploratory run of `cypha_bench_run.exe --domain-tag d16`
regenerated the shared `bench/BASELINE_REPORT.md` (merges whatever per-domain result caches exist on
disk, not just the domain just run) — reverted immediately via `git checkout -- bench/BASELINE_REPORT.md`
and switched to the dedicated smoke-test binaries above for all further verification, to avoid
interfering with the live production-overnight pipeline's artifacts.
