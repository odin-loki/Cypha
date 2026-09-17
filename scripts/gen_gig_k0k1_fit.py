#!/usr/bin/env python3
"""Provenance for the Bessel-ratio constants in the native GIG kernels.

Every hard-coded number in ``native/src/nig_gig_score_match.cpp`` and the Bessel table it sits
beside is derived, validated, or regenerated here. Reference values come from
``scipy.special.kv`` at double precision.

Background: the K_2/K_1 and K_0/K_1 kernels were rewritten to make K_0/K_1 the primitive and
recover K_2/K_1 from the exact recurrence ``K_2/K_1 = 2/x + K_0/K_1``. See
``docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md``, findings N1-N4 and L5-L9.

Subcommands
-----------
  --validate       check the shipped constants and the shipped table against scipy (default)
  --fit            re-derive the series and rational coefficients, printing C++-ready constants
  --table          regenerate native/src/bessel_table_data.cpp
  --p7-reference   recompute the exact held-out log-likelihood pinned by
                   native/tools/gate_score_match_p7_smoke.cpp

Requires numpy and scipy.
"""

import argparse
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.special import kve

# --- geometry, mirrored from native/include/cypha/bessel_table.hpp -------------------------------
N_TABLE = 16384
X0 = 1e-6
X1 = 120.0

# --- constants as shipped, mirrored from native/src/nig_gig_score_match.cpp ----------------------
EULER_GAMMA = 0.5772156649015329
BLEND_LO = 0.35
BLEND_HI = 0.65
ASYMPTOTIC_LO = 120.0

# x^5 residual coefficients (L^3, L^2, L, 1) of the small-x series
SERIES_C = (
    0.23513844126021524,
    0.37929893852576335,
    0.50510404194744363,
    0.20476904886486919,
)

# degree-4 / degree-4 rational fit of K_0/K_1 on [BLEND_LO, X1]: a0..a4 then b1..b4
RAT = (
    0.030691472162322212,
    3.6033167289598533,
    22.130395953401521,
    29.931931475966426,
    9.6665115472790717,
    12.123659950832327,
    35.889065451248875,
    34.765147597537606,
    9.6665119664887236,
)


def k0k1_exact(x):
    """K_0(x)/K_1(x). The exp(-x) scaling of kve cancels in the ratio."""
    return kve(0, x) / kve(1, x)


def k2k1_exact(x):
    return kve(2, x) / kve(1, x)


def series(x, c=SERIES_C):
    """Small-x form: x*L + x^3*(L^2/2 + L/2 + 1/4) + x^5*(c3 L^3 + c2 L^2 + c1 L + c0).

    The first two terms are the analytic ascending expansion. The x^5 term is a fitted stand-in
    for the true remainder (analytically L^3/4 + 7L^2/16 + 11L/32 + 11/128); fitting it absorbs
    part of the x^7 term too and is roughly 40x more accurate near the top of the range.
    """
    x = np.asarray(x, dtype=float)
    L = -np.log(0.5 * x) - EULER_GAMMA
    x2 = x * x
    return x * L + x2 * x * (0.5 * L * L + 0.5 * L + 0.25) + x2 * x2 * x * (
        ((c[0] * L + c[1]) * L + c[2]) * L + c[3]
    )


def rational(x, p=RAT):
    """Horner form, identical to rat44_k0k1 in nig_gig_score_match.cpp."""
    x = np.asarray(x, dtype=float)
    a0, a1, a2, a3, a4, b1, b2, b3, b4 = p
    num = a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)))
    den = 1.0 + x * (b1 + x * (b2 + x * (b3 + x * b4)))
    return num / den


def large_x(x):
    x = np.maximum(np.asarray(x, dtype=float), ASYMPTOTIC_LO)
    return 1.0 - 0.5 / x + 0.375 / (x * x)


def table_nodes():
    step = (X1 - X0) / (N_TABLE - 1)
    return X0 + step * np.arange(N_TABLE), step


def table_interp(x, tab, step):
    pos = (np.asarray(x, dtype=float) - X0) / step
    i = np.clip(pos.astype(int), 0, N_TABLE - 2)
    t = pos - i
    return tab[i] * (1.0 - t) + tab[i + 1] * t


def blend(x, small, mid):
    t = (x - BLEND_LO) / (BLEND_HI - BLEND_LO)
    w = t * t * (3.0 - 2.0 * t)
    return (1.0 - w) * small + w * mid


def compose(x, mid_fn):
    """The composite both C++ backends implement; mid_fn is the table or the rational."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lo = x <= BLEND_LO
    bl = (x > BLEND_LO) & (x < BLEND_HI)
    hi = (x >= BLEND_HI) & (x < X1)
    vh = x >= X1
    out[lo] = np.clip(series(np.maximum(x[lo], 1e-300)), 0.0, 1.0)
    if bl.any():
        out[bl] = blend(x[bl], np.clip(series(x[bl]), 0.0, 1.0), mid_fn(x[bl]))
    out[hi] = mid_fn(x[hi])
    out[vh] = large_x(x[vh])
    return out


def cmd_validate():
    nodes, step = table_nodes()
    tab0 = k0k1_exact(nodes)
    ok = True

    print("shipped Bessel table vs scipy")
    for name, col in (("K0/K1", tab0), ("K2/K1", k2k1_exact(nodes))):
        print(f"  {name} at nodes: max rel err vs scipy = 0 by construction ({len(col)} nodes)")
    rec = np.abs((k2k1_exact(nodes) - k0k1_exact(nodes)) - 2.0 / nodes) / k2k1_exact(nodes)
    print(f"  exact recurrence K2/K1 - K0/K1 == 2/x at every node: max residual {rec.max():.3e}")
    ok &= rec.max() < 1e-12

    xs = np.logspace(-9, 5, 400000)
    t0 = k0k1_exact(xs)
    t2 = k2k1_exact(xs)
    for name, mid in (("LUT", lambda x: table_interp(x, tab0, step)), ("ScoreMatch", rational)):
        v0 = compose(xs, mid)
        v2 = 2.0 / xs + v0
        e0 = np.abs(v0 - t0) / t0
        e2 = np.abs(v2 - t2) / t2
        in_range = bool(np.all((v0 >= 0.0) & (v0 < 1.0)))
        mono0 = bool(np.all(np.diff(v0) >= -1e-15))
        mono2 = bool(np.all(np.diff(v2) <= 1e-15))
        print(f"\n{name} backend over x in [1e-9, 1e5]")
        print(f"  K0/K1 worst rel err {e0.max():.3e} at x={xs[e0.argmax()]:.5g}")
        print(f"  K2/K1 worst rel err {e2.max():.3e} at x={xs[e2.argmax()]:.5g}")
        print(f"  K0/K1 in [0,1): {in_range}   non-decreasing: {mono0}")
        print(f"  K2/K1 non-increasing: {mono2}")
        ok &= in_range and mono0 and mono2 and e0.max() < 3e-5 and e2.max() < 3e-5

    a = 2.0 / xs + compose(xs, lambda x: table_interp(x, tab0, step))
    b = 2.0 / xs + compose(xs, rational)
    gap = (np.abs(a - b) / a).max()
    print(f"\ncross-backend K2/K1 worst relative gap: {gap:.3e}")
    ok &= gap < 5e-6
    print("\nPASS" if ok else "\nFAILED")
    return 0 if ok else 1


def cmd_fit():
    print("# Re-derived constants. Paste into native/src/nig_gig_score_match.cpp.")
    xs = np.logspace(-8, 0, 200000)
    L = -np.log(0.5 * xs) - EULER_GAMMA
    base = xs * L + xs**3 * (0.5 * L * L + 0.5 * L + 0.25)
    resid = k0k1_exact(xs) - base
    A = np.stack([xs**5 * L**3, xs**5 * L**2, xs**5 * L, xs**5], axis=1)
    w = 1.0 / k0k1_exact(xs)
    c, *_ = np.linalg.lstsq(A * w[:, None], resid * w, rcond=None)
    print("\n// x^5 residual coefficients (L^3, L^2, L, 1)")
    for nm, v in zip(("c3", "c2", "c1", "c0"), c):
        print(f"  constexpr double {nm} = {v:.17g};")
    e = np.abs(series(xs, c) - k0k1_exact(xs)) / k0k1_exact(xs)
    m = xs <= BLEND_HI
    print(f"// worst rel err on (0, {BLEND_HI}]: {e[m].max():.3e}")

    xr = np.logspace(np.log10(BLEND_LO), np.log10(X1), 200000)
    yr = k0k1_exact(xr)
    best = None
    for seed in range(12):
        rng = np.random.default_rng(seed)
        p0 = np.r_[0.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 0.1] * (
            1.0 + 0.3 * rng.standard_normal(9)
        )
        try:
            sol = least_squares(lambda p: (rational(xr, p) - yr) / yr, p0, method="lm", max_nfev=400000)
        except Exception:
            continue
        den = 1.0 + xr * (sol.x[5] + xr * (sol.x[6] + xr * (sol.x[7] + xr * sol.x[8])))
        if den.min() <= 0.1 or not np.all(sol.x[5:] > 0):
            continue  # reject near-poles and any fit that is not pole-free for all x > 0
        err = (np.abs(rational(xr, sol.x) - yr) / yr).max()
        if best is None or err < best[0]:
            best = (err, sol.x)
    err, p = best
    print("\n// degree-4 / degree-4 rational fit of K_0/K_1")
    for nm, v in zip(("a0", "a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"), p):
        print(f"  static constexpr double {nm} = {v:.17g};")
    print(f"// worst rel err on [{BLEND_LO}, {X1}]: {err:.3e}; all b > 0 so pole-free for x > 0")
    return 0


def cmd_table(out_path):
    nodes, _ = table_nodes()
    cols = (("kBesselX", nodes), ("kBesselK2K1", k2k1_exact(nodes)), ("kBesselK0K1", k0k1_exact(nodes)))
    lines = [
        "// AUTO-GENERATED by scripts/gen_gig_k0k1_fit.py --table — do not edit.",
        '#include "cypha/bessel_table.hpp"',
        "",
        "namespace cypha::detail {",
        "",
    ]
    for name, col in cols:
        lines.append(f"const double {name}[kBesselN] = {{")
        lines.extend(f"  {v:.17e}," for v in col)
        lines.append("};")
        lines.append("")
    lines.append("}  // namespace cypha::detail")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({N_TABLE} nodes per column)")
    return 0


def cmd_p7_reference():
    """Mirrors make_holdout / total_predictive_loglik in gate_score_match_p7_smoke.cpp.

    The C++ draws from std::mt19937 with std::gamma_distribution, which numpy cannot reproduce
    bit-for-bit, so the samples are read from the dump the test itself can emit. Pass the dump on
    stdin: one line per sample, "mp r_base chi psi".
    """
    data = np.loadtxt(sys.stdin)
    mp, r_base, chi, psi = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    r = np.maximum(r_base, 1e-8)
    cg = np.maximum(chi, 1e-8)
    pg = np.maximum(psi, 1e-8)
    chi_post = cg + mp / r
    x0 = np.sqrt(cg * pg)
    x1 = np.sqrt(chi_post * pg)
    r0 = np.maximum(k2k1_exact(x0), 1e-8)
    r1 = np.maximum(k2k1_exact(x1), 1e-8)
    total = float(np.sum(np.log(r1) - np.log(r0) - mp / (2.0 * r) + 0.5 * pg * (x1 - x0)))
    print(f"exact held-out loglik over {len(mp)} samples = {total:.10f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true", help="check shipped constants against scipy (default)")
    g.add_argument("--fit", action="store_true", help="re-derive the coefficients")
    g.add_argument("--table", metavar="OUT", nargs="?", const="native/src/bessel_table_data.cpp",
                   help="regenerate the Bessel table data file")
    g.add_argument("--p7-reference", action="store_true", help="exact held-out loglik from a sample dump on stdin")
    args = ap.parse_args()
    if args.fit:
        return cmd_fit()
    if args.table:
        return cmd_table(args.table)
    if args.p7_reference:
        return cmd_p7_reference()
    return cmd_validate()


if __name__ == "__main__":
    raise SystemExit(main())
