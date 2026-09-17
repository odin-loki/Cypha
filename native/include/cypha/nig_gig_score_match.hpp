#pragma once

namespace cypha {

/// GH/NIG GIG normalisation backend for ``gig_e_*`` helpers (Phase 7).
enum class GigNormalisationMode {
  /// Composite ``K_0/K_1`` backend: closed-form series, then the shipped Bessel LUT, then the
  /// large-``x`` asymptotic; ``K_2/K_1`` follows by the exact recurrence. Default.
  ///
  /// This used to interpolate a ``K_2/K_1`` table directly and was documented as "unchanged
  /// numerics". Both changed when the Bessel kernels were fixed — see
  /// ``docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md``, findings N1–N4 and L5–L9.
  Lut = 0,
  /// Hyvärinen score-matching rational fit — no partition function / Bessel LUT on hot path.
  ScoreMatch = 1,
};

/// Active mode: default ``Lut``; set ``CYPHA_GIG_SCORE_MATCH=1`` for ``ScoreMatch``.
GigNormalisationMode gig_normalisation_mode();

/// Override mode (tests); pass ``nullopt`` to revert to env/default.
void set_gig_normalisation_mode_override(GigNormalisationMode mode, bool active);

namespace detail {

/// Blend window joining the small-``x`` series to the mid-range approximant. Both backends use the
/// same window so they agree below it, and the blend removes the step (and the resulting
/// non-monotonicity in ``K_0/K_1``) that a hard switch leaves behind.
inline constexpr double kK0K1BlendLo = 0.35;
inline constexpr double kK0K1BlendHi = 0.65;

/// Above this the large-``x`` asymptotic replaces the mid-range approximant. It coincides with the
/// Bessel table's last node (``detail::kBesselX1``), which is where the table runs out.
inline constexpr double kK0K1AsymptoticLo = 120.0;

/// ``K_0(x)/K_1(x)`` for small ``x``. With ``L = −ln(x/2) − γ`` the analytic expansion is
/// ``x·L + x³·(L²/2 + L/2 + 1/4) + O(x⁵·poly(L))``; the ``x⁵`` coefficients are a least-squares fit
/// of the residual against ``scipy.special.kv``. Relative error ``≤ 2.6×10⁻⁵`` on ``(0, 0.65]``.
///
/// **Valid only on ``(0, kK0K1BlendHi]``.** The argument is saturated to that interval, so an
/// out-of-domain call returns the boundary value rather than the series' divergence — it is still
/// wrong, but it stays finite, inside ``[0,1)`` and monotone.
double k0k1_small_x(double x);

/// ``K_0(x)/K_1(x)`` for large ``x``: ``1 − 1/(2x) + 3/(8x²)``, from the standard asymptotic series
/// ``K_ν(x) ~ √(π/2x)·e⁻ˣ·[1 + (4ν²−1)/(8x) + …]``. Relative error ``≤ 2.2×10⁻⁷`` for ``x ≥ 120``.
///
/// **Valid only for ``x ≥ kK0K1AsymptoticLo``.** The argument is saturated there, because the
/// series exceeds 1 below ``x ≈ 1.2`` and is unbounded as ``x → 0``.
double k0k1_large_x(double x);

/// Smoothstep partition of unity between ``k0k1_small_x`` and a mid-range value on the blend window.
double k0k1_blend(double x, double mid_value);

}  // namespace detail

/// ``K_2(x)/K_1(x)`` for the score-match backend, as ``2/x + K_0/K_1`` (exact Bessel recurrence).
double gig_k2k1_score_match(double x);

/// ``K_0(x)/K_1(x)`` for the score-match backend: small-``x`` series, then a rational fit of
/// ``K_0/K_1`` itself, then the large-``x`` asymptotic. No Bessel table on the hot path.
///
/// This used to be evaluated as ``K_2/K_1 − 2/x``. Because the fitted ``K_2/K_1`` only approximates
/// the ``2/x`` pole, that difference cancelled catastrophically and returned a NEGATIVE value on
/// 44.1% of the domain, for a quantity that is mathematically confined to ``[0, 1)``. See
/// ``docs/reports/NUMERICAL_KERNEL_AUDIT_2026-09-17.md``, finding L5.
double gig_k0k1_score_match(double x);

/// Held-out NIG gate predictive log-likelihood proxy (Tier-1 acceptance metric for Phase 7).
/// Uses ``k2k1_fn`` for the Bessel-ratio increment between ``chi`` and ``chi + mp/r``.
double nig_gate_predictive_loglik(double mp, double r_base, double chi, double psi,
                                  double (*k2k1_fn)(double));

}  // namespace cypha
