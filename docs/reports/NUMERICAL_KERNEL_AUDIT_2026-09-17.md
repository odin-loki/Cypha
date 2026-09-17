# Numerical audit — inference gate, GIG/Bessel kernels, field, CUDA

**Date:** 2026-09-17 · **Status:** **R1 fixed**; **R3/R4 pinned by a CTest**; the rest recorded
**Scope:** `native/src/infer_cpu.cpp`, `nig_gig_math.cpp`, `nig_gig_score_match.cpp`,
`bessel_table*`, `nig_field.cpp`, `accel_cuda.cu`

This began while tracing the GIG world gate back through the Python archive during the
[historical documentation](../history/README.md) work, which turned up four wrong limits
(**N1–N4** below). Those were narrow and unreachable, so the surrounding numerical kernels were
audited in full to see whether the list was complete. **It was not.** Six module audits produced
25 candidate findings; each was then put to an independent adversarial check that assumed it was
wrong. **21 survived and 2 were refuted.** Two of the 25 checks were lost when the machine
restarted; both of those findings (R3 and L13) were instead verified by hand, so nothing here
rests on a check that did not finish.

Four of the survivors are reachable through shipped defaults, and three of those affect numbers
a user actually sees.

**Of those four, exactly one was ours to fix.** R1 lives in the kernel-LLR path, which has no
Python ancestor at all — `kernel_llr`, `KernelMemory` and `kernel_blend` occur **zero** times in
both `root-monolith/Cypha.py` and `cypha-v8/Cypha.py` — and `docs/port/PORT_CONTRACT.md:65` states
outright that *"Fixtures assume `use_kernel_llr=False`"*. Nothing documented constrains it, so it
was fixed and a regression test now guards it.

R3 and R4 are **faithful ports of Python defects** — the originals are at `Cypha.py:137` and
`:3129` — and changing them would break the parity the port contract asserts. **R2 is different
again:** its reference is the FastAPI `InferenceEngine` (`infer_cpu.hpp:226`), which is not in the
archive at all, and `PORT_CONTRACT.md:75` pins the `R_eff`-vs-`_mahal_ema` pairing explicitly with
CTest `native_gh_infer_deliberation`. Its Python counterpart computes something else entirely —
`anomaly_score = 1.0 − ood_gate`, bounded in `[0,1]` (`root-monolith/Cypha.py:1362`), and `r_eff`
occurs **zero** times in the archive — so there is no reference to check a change against.

All three are left as-is deliberately. Because none of them had *any* test coverage,
**R3 and R4 are now pinned** by CTest `native_ported_defects_pinned`, which asserts the defective
behaviour on purpose so that changing it is a deliberate contract decision rather than an
accident. Diverging from the reference remains the owner's call, not this report's.

---

## Summary

### Reachable in a shipped configuration

| # | Where | Defect |
|---|---|---|
| ~~**R1**~~ | `infer_cpu.cpp` | `world_gate` applied **twice** on the kernel-LLR path — **FIXED**, guarded by CTest `native_kernel_gate_invariant` |
| **R2** | `infer_cpu.cpp:1271-1280` | anomaly score divides a latent-variance-scaled `r_eff` by a dimensionless `mahal_ema` — **`use_gh` defaults to true**, so this is the default `/predict` path |
| **R3** | `infer_cpu.cpp:865-885` | ECE binning drops any sample at confidence exactly `1.0`, and scores an **all-NaN evaluation as a perfect 0.0** |
| **R4** | `infer_cpu.cpp:903-950` | `adapt_temperature_ece` never scores the incumbent temperature, so it can replace a better one with a worse one and report success |

### Latent — wrong, but not reachable today

| # | Where | Defect |
|---|---|---|
| **N1** | `nig_gig_math.cpp:105-107` | large-`x` limit returns `psi/chi`; correct is `sqrt(psi/chi)` |
| **N2** | `nig_gig_math.cpp:102-104` | small-`x` limit returns `psi/chi`; correct is `2/chi` |
| **N3** | `nig_gig_math.cpp:97-99` | the `chi/psi < kEps` guard carries N2's wrong form |
| **N4** | `nig_gig_score_match.cpp:62` | large-`x` series coefficient `6.75`; correct is `3/8` |
| **L5** | `nig_gig_score_match.cpp:70` | `K₀/K₁` computed as `k2k1(x) − 2/x`; the fit's constant `a0 = 1.99945961 ≠ 2` leaves a residual that survives as `−5.4e-4/x`, so the result is **negative on 44.1%** of `[1e-6, 120]` — for a quantity provably in `(0,1)`. Zero-crossing at `x = 0.0036342844` |
| **L6** | `nig_gig_math.cpp:85` | `active_k0k1` small-`x` returns `1/x` → `1e8`, where `K₀/K₁ → 0` (true value `2.08e-8` at `x=1e-9`). The limit is **inverted** |
| **L7** | `nig_gig_math.cpp:119` | `gig_e_v_lam_neg1` small-`x` returns `chi/psi`: at `χ=1, ψ=1e-12` that is `1e12` against a true `E[V] = 13.93` — **11 orders of magnitude** |
| **L8** | `nig_gig_math.cpp:61,65` | the `x → ∞` limit is clamped to the last table node `1.01252583`, never approaching 1: at `x = 1e4` the true value is `1.00015` — a **1.25%** floor that never decays |
| **L9** | `bessel_table.hpp:7` | the grid is **uniform** (spacing `7.3247e-3`) over `[1e-6, 120]`, so the first cell spans `[1e-6, 7.33e-3]` where `K₂/K₁` falls from `2e6` to `273`. Linear interpolation at its midpoint gives `1.00e6` against a true `545.97` — a **183,085% error** |
| ~~L10~~ | `nig_gig_score_match.cpp:89` | **NOT CONFIRMED.** The claim was that `+0.5·psi·(x1−x0)` has the wrong sign. The natural operational check — that the predictive log-likelihood decreases as the observation grows more anomalous — **passes** (0 non-monotone steps over `mp ∈ [0, 200]`). The term is dimensionally odd and can dominate (`+74.1` against a `−5.0` penalty at `ψ=16`), but I could not show it is wrong. Recorded as unresolved |
| **L11** | `accel_cuda.cu:157,178,41,54` | **NaN asymmetry verified by execution**: `fmax(NaN, 0.0) = 0.0` on the CUDA path while the CPU's `std::max(NaN, 0.0) = NaN` (`infer_cpu.cpp:126,1199`), so a NaN Mahalanobis silently becomes a maximally-confident gate on GPU. `pool_ensure` frees and nulls `g_pool` before `cudaMalloc` but leaves `g_pool_doubles` at the old capacity on failure, so a later call returns `cudaSuccess` with a null pool. No `GigNormalisationMode` dispatch on GPU |
| **L12** | `nig_field.cpp:20` | `field_diag_a` calls `a_out.assign(static_cast<size_t>(fd), 0.0)` **before** its `fd <= 0` guard, so a negative `fd` throws `std::length_error` rather than returning. (The companion float32 power-iteration overflow at `:188` needs `W_T` entries above ~1e38 and is not reachable.) |
| **L13** | `infer_cpu.cpp:1252` | `infer_at_h` hardcodes `gh_chi = gh_psi = 1.0`, silently ignoring the `CyphaInferOptions` fields that callers set |

**Refuted** on adversarial check: `accel_cuda.cu:128` (device interpolator geometry — the
hardcoded literals equal the real ones) and `nig_field.cpp:100` (inject early-return).

### What I verified personally

**Everything in both tables has now been checked by hand.** R1–R4, N1–N4 and L13 were read in
the source and reproduced; the workings are below and the commands are in
[Reproducing](#reproducing). L5–L12 were subsequently re-derived and re-measured the same way,
and the table entries above carry those measurements rather than the audit's.

That pass changed two entries:

- **L10 did not survive.** It was recorded as a wrong-sign term. The operational check — a
  predictive log-likelihood must fall as the observation gets more anomalous — passes cleanly
  over `mp ∈ [0, 200]`. It is now marked unresolved rather than confirmed.
- **L9 was understated.** "Cannot represent the pole" turns out to mean a 183,085% interpolation
  error in the first grid cell.

No finding in this report now rests on someone else's measurement.

L13, verified by hand: `infer_at_h` calls
`classify_at_h(m, h, h_field, m.temperature, mahal_ema_opt, m.mahal_std_ema, 1.0, 1.0, …)` at
`infer_cpu.cpp:1252` — the two literals are `gh_chi` and `gh_psi`. Meanwhile
`CyphaInferOptions::gh_chi/gh_psi/gh_alpha` exist at `infer_cpu.hpp:33-35` and *are* written by
callers, at `cypha.cpp:471` and `cypha_rest.cpp:1039`. So on the `infer_at_h` path those settings
are silently discarded. The GH path is unaffected — `gh_infer_at_h` takes them as explicit
arguments (`cypha_rest.cpp:1042`) — and since `use_gh` defaults to true, the default path honours
them. The failure is confined to the non-GH path, where a caller's configuration is accepted and
ignored.

---

## R1 — the world gate is applied twice on the kernel-LLR path

```cpp
if (use_kernel_llr && kernel_mem != nullptr && kernel_mem->n_basis() >= 4 && K > 0 && disc_lin > kEps) {
  const double conf_lin = disc_lin * out.world_gate;
  const double disc_new = probs[static_cast<std::size_t>(best_i)];
  out.disc = disc_new * (conf_lin / disc_lin);
}
```
— `native/src/infer_cpu.cpp:1165-1169`

`conf_lin / disc_lin` is `(disc_lin · world_gate) / disc_lin`, so `disc_lin` cancels exactly and
the block reduces to **`out.disc *= world_gate`**. Nineteen lines later:

```cpp
out.confidence = out.disc * out.world_gate;
```
— `:1184`

so the kernel path returns `softmax_max · world_gate²` where every other path returns
`softmax_max · world_gate`. `out.disc` is corrupted too: it no longer holds the discriminative
probability the rest of the code assumes.

At a gate of 0.6 that is a **40% understatement** of confidence. The adversarial check measured
it on the shipped fixture through the production GH path: row_2 `0.536132 → 0.324529` (39.5%),
row_3 `0.352374 → 0.198134` (43.8%), with the ratio equal to the gate to every printed digit.

The invariant it breaks is sharp: at `kernel_blend = 0.0` the blended LLR vector is bit-identical
to the linear one and the label is unchanged — yet the confidence is still scaled by the gate.

**Reachability.** `use_kernel_llr` is false by default (`infer_cpu.hpp:38`, `cypha.cpp:223`), but
the REST server reads it **straight from the request body** (`cypha_rest.cpp:285-286`, forwarded
at `:273`, consumed at `:1042`/`:1073`, emitted at `:1171`), so no saved flag is needed once a
trained kernel is present. It fires whenever `world_gate < 1`, i.e. `mahal_per_dim/r_base >
2.5075` — the out-of-distribution regime. And because `cls.confidence` feeds `apply_deliberation`
(`:992-997`), a configured abstain band can turn the doubled gate into a changed **label**, not
just a changed number.

No test can catch it: `kernel_llr_golden.cpp` never calls `classify_at_h`, and
`xor_kernel_bench.cpp` compares `res.label` only, never `res.confidence`.

### Fixed

The block was deleted; `out.disc` already holds the blended softmax maximum, which is what the
tail of the function expects. The linear softmax above it (`linear_best`, `z_lin`, `p_lin`,
`disc_lin`) fed only this block and went with it, removing a redundant softmax per call on the
kernel path. `linear_llrs` is kept — the blend still uses it.

**Proven against the built binary, not on paper.** `native/tools/kernel_gate_invariant.cpp`
exercises the `kernel_blend = 0.0` invariant, where the blended LLRs are bit-identical to the
linear ones so every downstream quantity must match. Before the fix:

```
world_gate = 0.541920330374
  ok    label       1.000000000000 == 1.000000000000
  ok    world_gate  0.541920330374 == 0.541920330374
  FAIL  disc        0.997472568864 != 0.540550664058   (diff 4.569e-01)
  FAIL  confidence  0.540550664058 != 0.292935394450   (diff 2.476e-01)
  FAIL  kernel confidence equals disc*gate^2 (0.292935394450) - the gate is applied twice
```

A **45.8% understatement** of confidence on an identical label. After the fix all four match
exactly. Registered as CTest **`native_kernel_gate_invariant`**.

The test asserts `world_gate < 1` before checking anything, so it fails loudly rather than
passing vacuously if a future change stops the input exercising the gated path.

---

## R2 — the anomaly score divides two different quantities

```cpp
const double r_base = (mahal_ema_fallback > 0.0 && std::isfinite(mahal_ema_fallback)) ? mahal_ema_fallback : 1.0;
return std::max(0.0, (r_eff - r_base) / r_base);
```
— `native/src/infer_cpu.cpp:1275-1279`

The `r_eff` it receives was formed by `gh_infer_at_h` against a *different* baseline:

```cpp
const double inv_mean = mean_inv_v(m);
const double r_base = 1.0 / (inv_mean + kEps);
out.r_eff = nig_r_eff_scalar(std::max(mahal_sq, 0.0), r_base, chi, psi);
```
— `:1209-1211`

That `r_base` is a latent variance, in `h²` units. `mahal_ema` is the EMA of the per-dimension
Mahalanobis — dimensionless, and ≈ 1 for a fitted world prior. The subtraction mixes units. The
only dimensionless inflation `r_eff` carries is `r_eff/r = 1/E[1/V]`, so the score should be built
against `gh_infer_at_h`'s own baseline, which that function already forms two lines later as
`gh_scale`.

**This is the default path.** `use_gh` is `true` by default at `cypha.hpp:32` and `:57`, and the
REST handlers read `body.value("use_gh", true)`. The output is consumed widely: `anomaly_score`
and `is_ood` go into the response JSON and the `explanation` block, `is_ood` raises the
`epistemic_var` floor to 0.5, both accumulate into the session summary, and the Qt shell renders
an OOD banner and bar from them.

On the shipped fixture (`r_base = 0.0875` against `mahal_ema = 1.0011`, 11.4× apart) all four GH
cases score exactly **0.0** instead of 0.178 / 0.297 / 1.160 / 1.437, and `is_ood` would need
`mahal/dim > 195` — about 14σ per dimension. In the other direction, a model whose mean latent
variance exceeds ~10 flags `is_ood` on perfectly in-distribution input.

It is a **ported upstream design flaw**, documented as a pairing at `infer_cpu.hpp:226` and in
`docs/port/PORT_CONTRACT.md:75`, so a fix has to update the port contract rather than being filed
as a silent bug.

---

## R3 — ECE binning drops the top of its own range, and rewards NaN

```cpp
for (int i = 0; i < n; ++i) {
  if (confs[i] >= lo && confs[i] < hi) { ... }
}
...
if (sum_w > 0.0) {
  ece += sum_w * std::abs(sum_c / sum_w - sum_corr / sum_w) / static_cast<double>(n);
}
```
— `native/src/infer_cpu.cpp:874-883`

The last bin is `[0.9, 1.0)`, strictly excluding 1.0. A sample at confidence exactly 1.0 falls
into **no bin at all** — it is dropped from the numerator while `n` still counts it in the
denominator. NaN confidences are dropped the same way, because every comparison against NaN is
false.

Reproduced here, in the shipped arithmetic:

| input | reported ECE | true ECE |
|---|---:|---:|
| 10 samples at conf `1.0`, all **wrong** | **0.000000** | 1.0 |
| the same at conf `0.999999` | 0.999999 | ≈1.0 |
| 10 samples, all conf `NaN` | **0.000000** | undefined |
| 5 at `1.0` (dropped) + 5 at `0.05`, all wrong | 0.025000 | 0.525 |

A perfectly confident, perfectly wrong classifier scores a **perfect calibration error of zero**,
and moving the confidence down by one part in a million restores the correct answer. Because
`adapt_temperature_ece` *minimises* this quantity, a temperature that drives confidences to 1.0
or to NaN does not merely evade detection — it **wins the search**.

Confidence exactly 1.0 is not exotic: it is `disc · world_gate`, and a saturated softmax rounds
to 1.0 in double precision while `world_gate` is exactly 1.0 whenever `r_eff ≤ r_base`.

### Not fixed: it is a faithful port

The Python reference has the identical predicate:

```python
mask = (confs >= lo) & (confs < hi)
```
— `docs/history/archive/root-monolith/Cypha.py:137` (`_compute_ece`)

so the C++ reproduces its reference exactly. Correcting the bin edge would change results against
the parity the port contract asserts, which makes it a deliberate divergence rather than a bug
fix. **Left as-is.**

---

## R4 — the temperature search has no non-regression guarantee

```cpp
double best_ece = std::numeric_limits<double>::infinity();
double best_T = infer.temperature;
```
— `native/src/infer_cpu.cpp:905-906`, with `infer.temperature = best_T;` at `:950`

`best_ece` starts at infinity, so the **incumbent temperature is never scored**. The first grid
point beats infinity unconditionally, and the winner is stored whether or not it is an
improvement. The caller gets a bare `double` and the REST route returns only
`{"temperature","n_used"}`, so a regression is undetectable from outside.

The adversarial check measured a **1.44× ECE regression** with an in-bounds incumbent (optimum
7.5672 at ECE 0.006231; the 20-point log-spaced grid returned 5.662237 at 0.008970). Only an
off-grid incumbent is needed — the grid points are ~19% apart.

Separately, `n_grid = 1` takes the branch at `:932-935`, which ignores `T_max` and stores `T_min`
with **no evaluation at all**; the REST handler clamps `n_grid < 1` up to 1 rather than rejecting
it, so `POST /adapt_temperature` with `n_grid` of 0 or 1 silently sets `temperature = 0.3`
regardless of the data.

### Not fixed: it is a faithful port

The Python reference seeds the search the same way:

```python
best_ece, best_T = float('inf'), self.temperature
```
— `docs/history/archive/root-monolith/Cypha.py:3129`

so the incumbent goes unscored there too. The parity is stated as a contract at
`infer_cpu.hpp:198`. Seeding `best_ece` from the incumbent would diverge from the reference, so it
is a deliberate decision to make, not a silent fix. **Left as-is.** Returning the chosen ECE
alongside `T` would preserve parity exactly while letting the caller reject a regression itself —
that is the change worth considering.

---

## N1 — the large-`x` asymptote is the wrong function

*(The four findings below are the original GIG/Bessel limits; they remain latent.)*

```cpp
if (x > 120.0) {
  return psi / chi_g;
}
const double ratio = active_k2k1(x);
return std::sqrt(psi / chi_g) * ratio;
```
— `native/src/nig_gig_math.cpp:105-110`

For `V ~ GIG(λ = −1, χ, ψ)` the first inverse moment is

```
E[1/V] = sqrt(psi/chi) · K₂(x)/K₁(x),    x = sqrt(chi·psi)
```

As `x → ∞`, `K₂/K₁ → 1`, so **`E[1/V] → sqrt(psi/chi)`**. The branch returns `psi/chi`, which
agrees only when `chi == psi`.

The neighbouring function takes the analogous limit **correctly**:

```cpp
if (x > 120.0) {
  return std::sqrt(chi_g / std::max(psi, kEps));
}
```
— `native/src/nig_gig_math.cpp:121-123` (`gig_e_v_lam_neg1`)

so the two siblings disagree about how to take the same kind of limit.

**Measured** (`scipy.special.kve`, ratio is scaling-invariant):

| `chi` | `psi` | `x` | `K₂/K₁` | correct `E[1/V]` | branch returns | error |
|---:|---:|---:|---:|---:|---:|---:|
| 14400 | 1 | 120 | 1.0125258270 | 8.437715e-3 | 6.944444e-5 | **121.5×** |
| 10000 | 4 | 200 | 1.0075093284 | 2.015019e-2 | 4.000000e-4 | **50.4×** |

`K₂/K₁` → 1 is itself confirmed: 1.0125258 at `x=120`, 1.0075093 at 200, 1.0001500 at 10⁴.

### What it would do

The gate is `world_gate = r_base / max(r_eff, r_base)` with
`r_eff = r_base / max(E[1/V], 1e-8)` (`infer_cpu.cpp:1162-1163`), i.e. `gate = min(E[1/V], 1)`.
At the default `gh_chi = gh_psi = 1` the branch engages when
`chi_post = 1 + mahal_per_dim/r_base` exceeds 14400 — so the gate would fall **discontinuously
from 8.4377e-3 to 6.9444e-5** as `mahal_per_dim/r_base` crosses 14399.

### Why it does not fire

`r_base` is `mahal_ema` when present, which is the running mean of `mahal_per_dim` itself. All
20 model fixtures under `fixtures/` carry `mahal_ema` between **0.9517 and 1.0011**, so
`r_base ≈ 1` and the trigger needs `mahal_per_dim > ~14400` — roughly **120 standard deviations
per latent dimension** from the world mean.

Two further dampers: both sides of the jump are already deep in the rejecting regime, so no
plausible decision threshold sits between 6.9e-5 and 8.4e-3; and in `gh_infer_at_h` the result
feeds `t_adj = temperature / max(gh_scale, 0.01)` (`infer_cpu.cpp:1211-1212`), where the
discontinuity at 8.44e-3 is already below the 0.01 clamp — so the deliberation-temperature path
is bit-for-bit unaffected. The paths that would see the full step are
`out.confidence = disc * world_gate` (`:1184`) and `gh_train_lr_scale`
(`train_step_vector.cpp:270`).

### It is mirrored on the GPU, and parity cannot catch it

`native/src/accel_cuda.cu:141-160` carries the same branches. And the parity harness
`native/tools/cuda_smoke.cpp:78-93` compares against `cypha::nig_r_eff_scalar` — the CPU
production function itself — so it is a self-comparison, not an independent oracle. A CPU/GPU
parity test cannot detect a defect both sides share. (CUDA is an opt-in build:
`CMakeLists.txt:19` declares `CYPHA_ENABLE_CUDA ... OFF`.)

---

## N2 / N3 — the small-`x` limit is also wrong

```cpp
if (chi0 < kEps || psi < kEps) {
  return psi / std::max(chi0, kEps);      // N3
}
...
if (x < 1e-6) {
  return psi / chi_g;                     // N2
}
```
— `native/src/nig_gig_math.cpp:97-99, 102-104`

As `x → 0`, `K₂/K₁ → 2/x`, so

```
E[1/V] → sqrt(psi/chi) · 2/sqrt(chi·psi) = 2/chi
```

Both branches return `psi/chi`, which is correct only when `psi == 2`.

**Measured:**

| `chi` | `psi` | `x` | correct `E[1/V]` = `2/chi` | branch returns | error |
|---:|---:|---:|---:|---:|---:|
| 1 | 1e-10 | 1.0e-05 | 2 | 1e-10 | 2×10¹⁰ |
| 2 | 1e-12 | 1.4e-06 | 1 | 5e-13 | 2×10¹² |
| 1e-4 | 1e-10 | 1.0e-07 | 20000 | 1e-06 | 2×10¹⁰ |

The closed form agrees with `2/chi` to ten significant figures in this regime, confirming the
limit rather than assuming it.

N2 is **unreachable in the shipping gate**: `chi_post ≥ gh_chi = 1.0` is hard-coded at
`infer_cpu.cpp:1224` and `:1252`, so `x ≥ 1` on that path. N3 is the more reachable of the two
and needs a caller-supplied `psi < 1e-8`.

---

## N4 — the score-match series has the wrong second-order coefficient

```cpp
if (x > 120.0) {
  return 1.0 + 1.5 / x + 6.75 / (x * x);
}
```
— `native/src/nig_gig_score_match.cpp:62`

Deriving the ratio from the standard asymptotic series
`K_ν(x) ~ sqrt(π/2x)·e^{−x}·[1 + (4ν²−1)/(8x) + (4ν²−1)(4ν²−9)/(2!(8x)²) + …]`:

```
K₂ : 1 + 15/8·x⁻¹ + 105/128·x⁻²
K₁ : 1 +  3/8·x⁻¹ −  15/128·x⁻²
K₂/K₁ = 1 + 3/2·x⁻¹ + 3/8·x⁻² + O(x⁻³)
```

The second-order coefficient is **3/8 = 0.375**, not 6.75 — off by exactly **18×**. Checked
numerically:

| `x` | true `K₂/K₁` | shipped `1 + 1.5/x + 6.75/x²` | correct `1 + 1.5/x + 0.375/x²` |
|---:|---:|---:|---:|
| 150 | 1.010016556517 | 1.010300000000 | 1.010016666667 |
| 500 | 1.003001497008 | 1.003027000000 | 1.003001500000 |

The linear term `1.5/x` is right, so this looks like a transcription slip in one coefficient.

**Unreachable through its caller.** `active_k2k1` is invoked only at `nig_gig_math.cpp:108`,
*after* the `x > 120` early return of N1 — so this large-`x` branch cannot be reached through
`gig_e_inv_v_lam_neg1` at all. It is reachable only via `nig_gate_predictive_loglik`, which is
called only from `native/tools/gate_score_match_p7_smoke.cpp`. ScoreMatch mode is also off by
default (`Lut` unless `CYPHA_GIG_SCORE_MATCH` is set, `nig_gig_score_match.cpp:38-51`).

N1 and N4 therefore mask each other: the dead branch is dead *because* of the other defect.

---

## Test coverage

No test or fixture exercises the affected range.

- `native/tests/` contains no test for either function.
- The only non-product caller, `native/tools/gate_score_match_p7_smoke.cpp`, draws
  `chi, psi ∈ [0.1, 10]`, `mp ~ Gamma(2, 0.5)`, `r_base ∈ [0.05, 2]`. Simulating that design
  2,000 times, the largest `x` reached is **28.6**, against a cut at 120.
- The `fixtures/gh_infer_deliberation` golden cases run at `x = 2.09, 2.24, 3.23` (back-solved
  from their recorded `r_eff` values).

---

## Reproducing

```bash
python3 - <<'PY'
import numpy as np
from scipy.special import kve
# N1: large-x limit
for chi, psi in [(14400.0, 1.0), (10000.0, 4.0)]:
    x = np.sqrt(chi*psi)
    print(chi, psi, "correct", np.sqrt(psi/chi)*kve(2,x)/kve(1,x), "code", psi/chi)
# N2: small-x limit
for chi, psi in [(1.0, 1e-10), (2.0, 1e-12)]:
    x = np.sqrt(chi*psi)
    print(chi, psi, "correct", 2/chi, "closed", np.sqrt(psi/chi)*kve(2,x)/kve(1,x), "code", psi/chi)
# N4: series coefficient
from fractions import Fraction as F
t = lambda nu: (F(4*nu*nu-1, 8), F((4*nu*nu-1)*(4*nu*nu-9), 128))
(a1,a2),(c1,c2) = t(2), t(1)
print("K2/K1 second-order coefficient =", a2-c2-a1*c1+c1*c1)   # -> 3/8
PY
```

---

## What was checked and found sound

- The Bessel lookup table itself: `native/src/bessel_table_data.cpp` satisfies the exact
  recurrence `K₂/K₁ − K₀/K₁ = 2/x` at every grid point, with `K₂/K₁ → 2/x` as `x → 0` and both
  ratios → 1 as `x → ∞`. Its last tabulated entry, 1.01252582699306592, matches
  `scipy.special` `K₂(120)/K₁(120)` to all printed digits.
- The GIG index: `gig_e_inv_v_lam_neg1` / `gig_e_v_lam_neg1` evaluate GIG(λ = −1) moments, which
  is **correct** — λ = −½ is the *prior* index and the callers form the conjugate update
  `chi + innovation²/R` before evaluating, taking λ to −1. Reproduced against the posterior of a
  λ = −½ prior to six decimals. An earlier reading of this as a mislabelling was wrong.
- `gig_e_v_lam_neg1`'s own large-`x` branch, which is correct.
