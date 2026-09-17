# Numerical audit — inference gate, GIG/Bessel kernels, field, CUDA

**Date:** 2026-09-17 · **Status:** **R1–R4 and N1–N4, L5–L9 all fixed and CTest-guarded**; L11–L13 recorded
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

All three are left as-is. Because none of them had *any* test coverage, **R3 and R4 are pinned**
by CTest `native_ported_defects_pinned`, which asserts the defective behaviour on purpose so that
changing it is deliberate rather than accidental.

> ### The parity argument above is weaker than it looks
>
> **Python is gone.** `CHANGELOG.md:55` records the runtime decommissioned at P7 (v2.4.0,
> 2026-08-16): `Cypha.py`, `cypha_core/`, `cypha_studio/`, `cypha_lm/` and the rest removed from
> the product path, with native C++ the sole runtime. Outside `docs/history/archive/` the
> repository holds **five** `.py` files, all utility scripts — no runtime, and nothing that can
> execute the reference these defects are "faithful" to.
>
> So "it matches Python" is a historical explanation for how each defect got here, **not a live
> reason to keep it**. There is no second implementation to stay in step with, and no parity
> test exercises any of R2/R3/R4. The port contract is now a record of how the port was done,
> not a constraint on what the code may become.
>
> That makes these ordinary engineering calls, on their merits:
>
> | | Worth fixing? | Why |
> |---|---|---|
> | **R3** | **Yes** | An all-NaN evaluation scores a perfect `0.0`, and `adapt_temperature_ece` *minimises* this — so a temperature that produces NaN confidences does not merely evade detection, it **wins the search**. Plus a perfectly-confident, perfectly-wrong model reports ECE `0.000000`. Live on `POST /adapt_temperature`, no test coverage. |
> | **R4** | **Probably** | Can install a worse temperature than the incumbent and report success (measured 1.44× ECE regression); `n_grid=1` sets `T_min` with no search at all. Small blast radius, one endpoint. Returning the chosen ECE alongside `T` fixes it without touching the search. |
> | **R2** | **Judgement call** | The unit mismatch is real and it is the default `/predict` path, feeding the OOD flag users see. But its reference is unavailable, so "correct" has to be decided rather than looked up — the substantive question is what the anomaly score is *for*. |
> | **L5–L13, N1–N4** | **Not now** | Wrong, sometimes spectacularly (L9's 183,085% interpolation error, L5 negative across 44% of its domain), but in paths nothing reaches. Fix opportunistically if that code is touched. |
>
> **R3 and R4 were subsequently fixed on exactly this reasoning.** See below. R2 remains open:
> its reference is unavailable, so "correct" has to be decided rather than looked up, and its
> `anomaly > 3.0` threshold was tuned against the current scale — changing the formula without
> retuning the threshold would move OOD rates on the default path.

---

## R3 and R4 — fixed

Both are corrected in `native/src/infer_cpu.cpp` and guarded by CTest
`native_ported_defects_pinned`, which asserts the **correct** behaviour. (That test previously
asserted the defective behaviour, to stop an accidental change while the parity question was
open; the question is settled, so it was rewritten rather than worked around.)

**R3, `compute_ece_bins`:** the top bin is now closed at 1.0; non-finite confidences are
excluded from both numerator and denominator, so they neither score nor dilute; and an
evaluation with no finite sample returns `+infinity`, so it can never win a minimisation.

**R4, `adapt_temperature_ece`:** the incumbent temperature is scored before the grid, so the
search can only improve on it, and the `n_grid <= 1` path evaluates its single point instead of
storing it unscored.

### Verified red-then-green against the shipped function

`compute_ece_bins` was moved out of its anonymous namespace and declared in `infer_cpu.hpp`, so
the test drives the real function rather than a copy of it. With the old ECE body restored in
place — signature kept, so it still compiles and links — three assertions fail:

```
FAIL  conf==1.0, all wrong -> ECE is 1.0        got 0.000000000, want 1.000000000
FAIL  all-NaN evaluation -> +inf                got 0.000000000, want inf
FAIL  5 NaN + 5 confident-wrong -> ECE is 1.0   got 0.000000000, want 1.000000000
```

and with the old `+infinity` seeding, the end-to-end R4 check — which finds the grid's own
optimum, adopts it as the incumbent, then re-runs with a single grid point pinned elsewhere —
returns `0.300000` where the incumbent `8.000000` was better. All pass after the fix.

An earlier attempt at this red test was **invalid** and is recorded as such: stashing the whole
source file left the header declaring a symbol the old file did not define, the build failed,
and the runner silently executed a stale binary that reported PASS. Reverting only the function
body is what produced the result above.

---

## Summary

### Reachable in a shipped configuration

| # | Where | Defect |
|---|---|---|
| ~~**R1**~~ | `infer_cpu.cpp` | `world_gate` applied **twice** on the kernel-LLR path — **FIXED**, guarded by CTest `native_kernel_gate_invariant` |
| ~~**R2**~~ | `infer_cpu.cpp` | anomaly score divided a latent-variance-scaled `r_eff` by a dimensionless `mahal_ema` — **FIXED**, now `max(0, r_eff/r_base − 1)`, guarded by CTest `native_ported_defects_pinned` |
| ~~**R3**~~ | `infer_cpu.cpp` | ECE binning dropped confidence exactly `1.0` and scored an all-NaN evaluation as a perfect `0.0` — **FIXED**, guarded by CTest `native_ported_defects_pinned` |
| ~~**R4**~~ | `infer_cpu.cpp` | `adapt_temperature_ece` never scored the incumbent temperature — **FIXED**, guarded by the same CTest |

### Latent — wrong, but not reachable today (N1–N4 and L5–L9 have since been fixed)

| # | Where | Defect |
|---|---|---|
| ~~**N1**~~ | `nig_gig_math.cpp` | large-`x` limit returned `psi/chi`; correct is `sqrt(psi/chi)` — **FIXED** |
| ~~**N2**~~ | `nig_gig_math.cpp` | small-`x` limit returned `psi/chi`; correct is `2/chi` — **FIXED** |
| ~~**N3**~~ | `nig_gig_math.cpp` | the `chi/psi < kEps` guard carried N2's wrong form — **FIXED** |
| ~~**N4**~~ | `nig_gig_score_match.cpp` | large-`x` series coefficient was `6.75`; correct is `3/8` — **FIXED** |
| ~~**L5**~~ | `nig_gig_score_match.cpp:70` | `K₀/K₁` computed as `k2k1(x) − 2/x`; the fit's constant `a0 = 1.99945961 ≠ 2` leaves a residual that survives as `−5.4e-4/x`, so the result is **negative on 44.1%** of `[1e-6, 120]` — for a quantity provably in `(0,1)`. Zero-crossing at `x = 0.0036342844` — **FIXED** |
| ~~**L6**~~ | `nig_gig_math.cpp:85` | `active_k0k1` small-`x` returns `1/x` → `1e8`, where `K₀/K₁ → 0` (true value `2.08e-8` at `x=1e-9`). The limit is **inverted** — **FIXED** |
| ~~**L7**~~ | `nig_gig_math.cpp:119` | `gig_e_v_lam_neg1` small-`x` returns `chi/psi`: at `χ=1, ψ=1e-12` that is `1e12` against a true `E[V] = 13.93` — **11 orders of magnitude** — **FIXED** |
| ~~**L8**~~ | `nig_gig_math.cpp:61,65` | the `x → ∞` limit is clamped to the last table node `1.01252583`, never approaching 1: at `x = 1e4` the true value is `1.00015` — a **1.25%** floor that never decays — **FIXED** |
| ~~**L9**~~ | `bessel_table.hpp:7` | the grid is **uniform** (spacing `7.3247e-3`) over `[1e-6, 120]`, so the first cell spans `[1e-6, 7.33e-3]` where `K₂/K₁` falls from `2e6` to `273`. Linear interpolation at its midpoint gives `1.00e6` against a true `545.97` — a **183,085% error** — **FIXED** |
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

---

## N1–N4, L5–L9 — fixed together, because they were one defect

These nine findings were filed separately, against four functions and two backends. Working
through them made it clear they are all the same mistake seen from different angles: **`K₂/K₁`
was being treated as the primitive quantity, and `K₀/K₁` derived from it.**

That is backwards. `K₂/K₁` carries a `2/x` pole — it runs to `2e6` at `x = 1e-6` — while `K₀/K₁`
is confined to `[0, 1)`, increases monotonically, and is smooth everywhere. The two are related
by an exact recurrence, with no approximation anywhere in it:

```
K₂(x) − K₀(x) = (2/x)·K₁(x)     ⟹     K₂/K₁ = 2/x + K₀/K₁
```

Approximating the bounded function and adding the pole back analytically is well conditioned.
Doing it the other way round is not, and that produced every one of these findings:

| Backend | What it did | What went wrong |
|---|---|---|
| LUT (default) | interpolated a `K₂/K₁` table | linear interpolation of a pole: **183,084%** worst-cell error (L9); clamped at both ends (L6, L8) |
| ScoreMatch (opt-in) | fitted `z = x·K₂/K₁`, then `K₀/K₁ = z/x − 2/x` | catastrophic cancellation: **negative on 44.1%** of the domain (L5); wrong series coefficient (N4) |
| GIG moments | hand-written limit branches to paper over both | every branch returned the wrong limit (N1, N2, N3, L7) |

### The fix

`K₀/K₁` is now the primitive in both backends, and `K₂/K₁ = 2/x + K₀/K₁` in both:

- **`x ≤ 0.35`** — closed form. With `L = −ln(x/2) − γ`, the analytic expansion is
  `x·L + x³·(L²/2 + L/2 + 1/4) + O(x⁵·poly(L))`; the `x⁵` coefficients are a least-squares fit of
  the residual against `scipy.special.kv`.
- **`0.35 < x < 0.65`** — a smoothstep blend into the mid-range approximant, so the two meet with
  matching value and slope. Without it the seam left a `3.6e-5` step that made `K₀/K₁` **decrease**
  at one point, which it must never do.
- **`0.65 ≤ x ≤ 120`** — the shipped `K₀/K₁` table (LUT), or a degree-4/degree-4 rational fit of
  `K₀/K₁` itself (ScoreMatch). Every denominator coefficient of that fit is positive, so it is
  pole-free for all `x > 0` by construction rather than merely on the fitted interval.
- **`x > 120`** — `1 − 1/(2x) + 3/(8x²)`, from the standard asymptotic series for `K_ν`. Adding
  `2/x` recovers `1 + (3/2)/x + (3/8)/x²`, which is N4's correct coefficient — it now falls out of
  the recurrence instead of being written down separately and got wrong.

The table is still used, but only where interpolating it is actually accurate. Its first cell
spans `[1e-6, 7.3e-3]` — four decades — which is where L9 lived.

With `K₂/K₁` reproducing the pole analytically, **all four GIG moment special cases were deleted.**
`gig_e_inv_v_lam_neg1` and `gig_e_v_lam_neg1` are now four lines each: clamp the parameters away
from zero, evaluate the closed form. Every limit N1, N2, N3 and L7 asserted by hand now falls out
of the algebra:

| Regime | True limit | What the old branch returned |
|---|---|---|
| `psi → 0` | `E[1/V] → 2/chi` (the GIG degenerates to `InvGamma(1, chi/2)`) | `psi/chi`, i.e. ≈ 0 |
| `psi → 0` | `E[V]` diverges logarithmically — `InvGamma(1, ·)` has **no mean** | `chi/psi`: `1e12` against a true `13.93` |
| `chi → 0` | `E[V] → 0` | `chi/psi` |
| `x → ∞` | `E[1/V] → sqrt(psi/chi)` | `psi/chi` — at `chi=1e4, psi=2e4` that is `2.0` against a true `1.414`, a **41%** overstatement |

Note the `psi → 0` row for `E[V]`: an intermediate version of this fix returned `0.0` there, which
is the right limit for `chi → 0` and the wrong one for `psi → 0`. The two degenerate directions
are not interchangeable, and the guard had been testing `chi0 < kEps || psi < kEps` as though they
were. Clamping `psi` and evaluating the series reports the divergence at the correct rate.

### Measured result

Relative error against `scipy.special.kv`, over `x ∈ [1e-9, 1e5]`:

| | before | after |
|---|---|---|
| `K₂/K₁`, worst | **183,084%** (LUT first cell) | `1.9e-6` |
| `K₀/K₁`, worst | `7415%` (LUT, below the first node) | `2.5e-5` |
| `K₀/K₁`, sign | **negative on 44.1%** of the domain (ScoreMatch) | in `[0, 1)` everywhere, monotone |
| `K₂/K₁` as `x → ∞` | floored at `1.0125` | `1.00015` at `x = 1e4` (true `1.00015`) |

The ScoreMatch backend also got **three orders of magnitude more accurate** as a side effect: its
`K₂/K₁` worst error fell from `1.3e-3` to `1.9e-6`, because it no longer approximates the pole at
all. The two backends now agree with each other to `7.9e-7`.

### The Phase 7 acceptance gate was inverted

Fixing the kernels broke `native_gate_score_match_p7_smoke`, and the reason is worth recording,
because the test had been passing for the wrong reason since it was written.

Its only assertion was `loglik_score_match >= loglik_lut − 1e-6`. There was **no reference value**
anywhere in it. That does not measure accuracy; it measures which backend reports the *higher*
log-likelihood, and rewards whichever one overstates it. Evaluated against `scipy` on the test's
own 512-sample holdout:

| | held-out loglik | distance from exact |
|---|---|---|
| exact (`scipy.special.kv`) | −25.3006064719 | — |
| **old** LUT | −25.3172787408 | `1.67e-2` |
| **old** ScoreMatch | −25.1582268234 | **`1.42e-1`** |
| **new** LUT | −25.3005467919 | `5.97e-5` |
| **new** ScoreMatch | −25.3005712187 | `3.53e-5` |

So the gate passed the score-match backend with a margin of **+0.159** at a moment when that
backend was **8.5× further from the truth** than the LUT it was being compared against. After the
fix the two agree to `2.4e-5` and score-match is the *closer* of the two — and the one-sided
assertion started failing, precisely because score-match had stopped overstating.

The test now pins the exact value and asserts that both backends track it, and that score-match is
no further from it than the LUT. That is what "optimality acceptance" was meant to mean. The new
bounds are ~270× and ~2800× tighter than what the old code delivered.

### The CUDA device path carried the same defects

`accel_cuda.cu` uploaded `kBesselK2K1` and re-implemented the same interpolation and the same three
wrong limit branches. Left alone, the fix would have made the CPU correct and the GPU 183,084%
wrong — a divergence that **CPU/GPU parity tests cannot catch**, because they compare the two paths
against each other rather than against a reference.

It now mirrors the host exactly: it uploads `kBesselK0K1`, rebuilds `K₂/K₁` from the recurrence,
and shares the same constants and branch structure. There is no CUDA toolchain in this
environment, so the device code is **not compile-tested here**; the arithmetic was verified by
transcribing the device functions into host C++ (stripping `__device__`, mapping `fmax`/`fmin`/
`log`/`sqrt` to `<cmath>`) and checking them against the host path — **bit-identical at every test
point**, including both degenerate directions.

`native_bessel_gig_limits` now also checks, as plain text, that the shared constants appear in both
files and that the device references `kBesselK0K1` and not `kBesselK2K1`. That guard needs no GPU
and was confirmed to fail when a single device constant is perturbed. `pool_ensure`'s stale-capacity
bug (part of L11) was fixed in the same pass: it now clears `g_pool_doubles` when it frees the
buffer, so a failed `cudaMalloc` can no longer leave a non-zero capacity that makes the next call
return `cudaSuccess` with a null pool.

### Adversarial review of the fix

The change was put to an independent adversarial reviewer instructed to find reasons it was wrong
or dangerous, plus two call-site and pinned-test audits. Nothing was found that blocks it; the
reviewer specifically could not substantiate any cancellation at large `x`, any consumer relying on
the old small-`x` behaviour, any overflow or NaN reachable from the new `2/chi` clamp (checked over
the cross product of `chi, psi ∈ {0, 1e-300, 1e300}` — every result finite), or any compatibility
break for serialised models. Its measured blast radius on the world gate across the reachable
parameter band is at most **1.1%**, and in that case the new value is the correct one.

Three of its findings were accepted and fixed:

- `detail::k0k1_small_x` clamped its *result* to `[0,1]`, which turned the series' divergence above
  `x ≈ 1.7` into an exact `0.0` — in range, silent, and the worst possible answer for a quantity
  that tends to 1. Both helpers now saturate their *argument* to the documented domain instead, so
  an out-of-range call returns the boundary value. All current callers guard correctly, so this is
  defence in depth rather than a live bug.
- The `Lut` enum was still documented as "unchanged numerics", which this change falsifies.
- The `psi → 0` branch of `E[V]` returns a `kEps` artifact. The comment claimed the clamp "reports
  that divergence at the right rate" — the rate is right, but the magnitude is set by a constant
  that exists to guard a division, and the comment now says so.

One was **refuted**: the reviewer measured the small-`x` series as *less* accurate than the table
over `x ∈ [0.22, 0.5]` and put the crossover at `0.22`, concluding the blend window sits in the
wrong place. Re-measuring on a **worst-in-cell** basis — linear interpolation is exact at the nodes
and worst mid-cell, so comparing at arbitrary sample points flatters it — the series wins up to
`x ≈ 0.44` and the table above. The window `[0.35, 0.65]` straddles that crossover. The loose
comment it was reacting to has been replaced with the measured figure.

Its remaining points were documentation and provenance debt, now closed by
`scripts/gen_gig_k0k1_fit.py`.

### Provenance

`scripts/gen_gig_k0k1_fit.py` derives, validates and regenerates everything this change introduced:

- `--fit` re-derives the series and rational coefficients from `scipy` and prints them C++-ready.
- `--validate` (default) checks the shipped constants and the shipped table, and asserts range,
  monotonicity, accuracy and cross-backend agreement. This is the same set of properties
  `native_bessel_gig_limits` enforces in CI.
- `--table` regenerates `native/src/bessel_table_data.cpp`, which had been unreproducible: its
  header names `scripts/gen_native_bessel_table.py`, and **that script is not in the repository**.
  The regenerated columns agree with the shipped ones to `2.6e-15` relative, and the grid is exact,
  so the shipped file was left in place rather than churned for last-bit noise.
- `--p7-reference` recomputes the exact held-out log-likelihood pinned by the Phase 7 test.


### Guarded by

CTest **`native_bessel_gig_limits`** (`native/tools/bessel_gig_limits.cpp`) drives **both**
backends and checks three things separately, because the defects fell into three groups:
accuracy against pinned `scipy` values across every seam; range and shape (`K₀/K₁ ∈ [0,1)` and
non-decreasing, `K₂/K₁ > 1` and non-increasing, over 4,000 log-spaced points); and each limit
above. It also pins `K₂/K₁` at `x = 0.00365` — L9's worst point — and checks the two backends
against each other.

### One golden fixture was regenerated

`fixtures/gh_infer_deliberation/sidecar.json` pinned `expected_confidence`, `expected_r_eff` and
`expected_chi_new` to `1e-10` absolute. Its own description read *"parity vs Cypha.py"*: the
values were captured from the Python reference, which carried the same interpolation error the
C++ inherited. Holding the fixed code to them would mean preserving bug-compatibility with a
runtime that was decommissioned at P7 (`CHANGELOG.md`).

It was regenerated with the repo's own `cypha_fixture_gen` — the sanctioned native path, which
the header comment describes as replacing the removed Python generators — and its description now
says it is a native golden rather than a parity fixture. `reference.cypha` and `f_field.json` are
**byte-identical**; only the expectations moved, by `1e-7` to `1.6e-6`.

Before accepting them, each was checked against `scipy` at the operating points the fixture
actually exercises. The new values are **8–15× closer to truth**:

| `x` | true `K₂/K₁` | old (interp `K₂/K₁`) | new (`2/x` + interp `K₀/K₁`) |
|---|---|---|---|
| 1.00 | 2.699483935594 | 2.699509054277 (`9.3e-6`) | 2.699482285144 (`6.1e-7`) |
| 1.68 | 1.979140909120 | 1.979145628525 (`2.4e-6`) | 1.979140409177 (`2.5e-7`) |
| 2.00 | 1.814307758764 | 1.814308323651 (`3.1e-7`) | 1.814307690258 (`3.8e-8`) |

### Why this was safe to change

The audit's own [What was checked and found sound](#what-was-checked-and-found-sound) section
records that `bessel_table_data.cpp` satisfies `K₂/K₁ − K₀/K₁ = 2/x` at **every grid point**. The
two tabulated columns are therefore already mutually consistent, so building `K₂/K₁` from the
`K₀/K₁` column changes nothing at the nodes and only improves the values between them. No table
data was edited.


## Test coverage

*(As found during the audit. `native_bessel_gig_limits` has since been added and now covers all
of this — see [the fix](#n1n4-l5l9--fixed-together-because-they-were-one-defect) above.)*

No test or fixture exercised the affected range.

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
