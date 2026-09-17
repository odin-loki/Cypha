# Numerical kernel audit — GIG/Bessel gate

**Date:** 2026-09-17 · **Scope:** `native/src/nig_gig_math.cpp`, `native/src/nig_gig_score_match.cpp`,
`native/src/accel_cuda.cu` · **Status:** findings recorded, **no code changed**

These defects surfaced during the [historical documentation](../history/README.md) work, while
tracing the GIG world gate back through the Python archive. They are recorded here rather than
fixed, at the owner's direction.

Every claim below was derived from first principles and reproduced numerically in this
repository. Nothing is carried over from a summary.

---

## Summary

| # | Where | Defect | Severity |
|---|---|---|---|
| **N1** | `nig_gig_math.cpp:105-107` | large-`x` limit returns `psi/chi`; correct is `sqrt(psi/chi)` | latent — needs ~120σ/dim |
| **N2** | `nig_gig_math.cpp:102-104` | small-`x` limit returns `psi/chi`; correct is `2/chi` | unreachable in the shipping gate |
| **N3** | `nig_gig_math.cpp:97-99` | the `chi/psi < kEps` guard carries N2's wrong form | latent |
| **N4** | `nig_gig_score_match.cpp:62` | large-`x` series coefficient `6.75`; correct is `0.375` | unreachable through its caller |

None is known to fire in any shipped configuration, fixture or test. They are recorded because
each is wrong independently of reachability, and because reachability rests on
`mahal_ema ≈ 1`, which is a property of the current fixtures rather than a guarantee.

---

## N1 — the large-`x` asymptote is the wrong function

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
