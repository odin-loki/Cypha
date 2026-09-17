#include "cypha/nig_gig_score_match.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <optional>

#include "cypha/env.hpp"

namespace cypha {

namespace {

constexpr double kEps = 1e-8;

/// Rational fit of ``K_0(x)/K_1(x)`` itself on ``x∈[0.35,120]`` (degree 4 over degree 4, least
/// squares against ``scipy.special.kv`` on a log-spaced grid). Max rel-err ``≈7.1×10⁻⁸``.
///
/// The previous backend fitted ``z(x)=x·K_2/K_1`` and recovered ``K_0/K_1`` as ``z/x − 2/x``. That
/// difference cancels catastrophically as ``x→0`` — the fit's ``z(0)=1.99945961`` rather than the
/// true 2, so the recovered ratio ran to ``−∞`` — and the result was negative on 44.1% of the
/// domain (finding L5). Fitting the bounded, smooth ``K_0/K_1`` directly and recovering ``K_2/K_1``
/// by the exact recurrence ``K_2/K_1 = 2/x + K_0/K_1`` removes the cancellation entirely and is
/// also three orders of magnitude more accurate.
struct ScoreMatchK0K1Coeffs {
  static constexpr double a0 = 0.030691472162322212;
  static constexpr double a1 = 3.6033167289598533;
  static constexpr double a2 = 22.130395953401521;
  static constexpr double a3 = 29.931931475966426;
  static constexpr double a4 = 9.6665115472790717;
  static constexpr double b1 = 12.123659950832327;
  static constexpr double b2 = 35.889065451248875;
  static constexpr double b3 = 34.765147597537606;
  static constexpr double b4 = 9.6665119664887236;
};

std::atomic<int> g_mode_override{-1};  // -1 = use env; else static_cast<int>(GigNormalisationMode)

/// Every denominator coefficient is positive, so the denominator exceeds 1 for all x > 0 and the
/// fit is pole-free by construction, not merely on the fitted interval.
double rat44_k0k1(double x) {
  using C = ScoreMatchK0K1Coeffs;
  const double num = C::a0 + x * (C::a1 + x * (C::a2 + x * (C::a3 + x * C::a4)));
  const double den = 1.0 + x * (C::b1 + x * (C::b2 + x * (C::b3 + x * C::b4)));
  return num / std::max(den, kEps);
}

}  // namespace

GigNormalisationMode gig_normalisation_mode() {
  const int ov = g_mode_override.load(std::memory_order_relaxed);
  if (ov >= 0) {
    return static_cast<GigNormalisationMode>(ov);
  }
  const std::optional<std::string> env = env_get("CYPHA_GIG_SCORE_MATCH");
  if (env.has_value()) {
    const std::string& v = *env;
    if (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON") {
      return GigNormalisationMode::ScoreMatch;
    }
  }
  return GigNormalisationMode::Lut;
}

void set_gig_normalisation_mode_override(GigNormalisationMode mode, bool active) {
  g_mode_override.store(active ? static_cast<int>(mode) : -1, std::memory_order_relaxed);
}

namespace detail {

double k0k1_small_x(double x) {
  // K_0/K_1 = x·L + x³·(L²/2 + L/2 + 1/4) + x⁵·(c₃L³ + c₂L² + c₁L + c₀),  L = −ln(x/2) − γ.
  // The first two terms are the analytic expansion; the x⁵ coefficients are fitted against
  // scipy.special.kv. Relative error stays below 2.6e-5 for every x in (0, 0.65].
  constexpr double kEulerGamma = 0.5772156649015329;
  constexpr double c3 = 0.23513844126021524;
  constexpr double c2 = 0.37929893852576335;
  constexpr double c1 = 0.50510404194744363;
  constexpr double c0 = 0.20476904886486919;
  // Saturate the ARGUMENT to the valid interval, not the result. Clamping the result instead
  // turned the series' blow-up above x ~ 1.7 into an exact 0.0 -- in range, silent, and the worst
  // possible answer for a quantity that tends to 1.
  const double xs = std::clamp(x, 1e-300, kK0K1BlendHi);
  const double L = -std::log(0.5 * xs) - kEulerGamma;
  const double x2 = xs * xs;
  const double x3 = x2 * xs;
  const double x5 = x3 * x2;
  const double v = xs * L + x3 * (0.5 * L * L + 0.5 * L + 0.25) + x5 * (((c3 * L + c2) * L + c1) * L + c0);
  // K_0/K_1 is confined to [0,1).
  return std::clamp(v, 0.0, 1.0);
}

double k0k1_large_x(double x) {
  // K_ν(x) ~ √(π/2x)·e⁻ˣ·[1 + (4ν²−1)/(8x) + (4ν²−1)(4ν²−9)/(2!(8x)²) + …] gives
  // K_0/K_1 = 1 − 1/(2x) + 3/(8x²) + O(x⁻³). Saturated below its valid range: the series exceeds
  // 1 for x < ~1.2 and diverges as x → 0, so an out-of-domain call must not see it raw.
  const double xs = std::max(x, kK0K1AsymptoticLo);
  return 1.0 - 0.5 / xs + 0.375 / (xs * xs);
}

double k0k1_blend(double x, double mid_value) {
  if (x <= kK0K1BlendLo) {
    return k0k1_small_x(x);
  }
  if (x >= kK0K1BlendHi) {
    return mid_value;
  }
  const double t = (x - kK0K1BlendLo) / (kK0K1BlendHi - kK0K1BlendLo);
  const double w = t * t * (3.0 - 2.0 * t);  // smoothstep: C¹ at both ends
  return (1.0 - w) * k0k1_small_x(x) + w * mid_value;
}

}  // namespace detail

double gig_k0k1_score_match(double x) {
  if (x <= 0.0) {
    return 0.0;
  }
  if (x >= detail::kK0K1AsymptoticLo) {
    return detail::k0k1_large_x(x);
  }
  if (x <= detail::kK0K1BlendLo) {
    return detail::k0k1_small_x(x);
  }
  return detail::k0k1_blend(x, rat44_k0k1(x));
}

double gig_k2k1_score_match(double x) {
  // Exact Bessel recurrence K_2 = K_0 + (2/x)·K_1, so K_2/K_1 = 2/x + K_0/K_1. Adding a positive
  // quantity to the pole cannot cancel, and it keeps the two ratios mutually consistent.
  const double xs = std::max(x, kEps);
  return 2.0 / xs + gig_k0k1_score_match(xs);
}

double nig_gate_predictive_loglik(double mp, double r_base, double chi, double psi,
                                  double (*k2k1_fn)(double)) {
  if (k2k1_fn == nullptr || mp < 0.0) {
    return -1e300;
  }
  const double r = std::max(r_base, kEps);
  const double chi_g = std::max(chi, kEps);
  const double psi_g = std::max(psi, kEps);
  const double chi_post = chi_g + mp / r;
  const double x0 = std::sqrt(chi_g * psi_g);
  const double x1 = std::sqrt(chi_post * psi_g);
  if (x0 <= 1e-8) {
    return -mp / (2.0 * r);
  }
  const double r0 = std::max(k2k1_fn(x0), kEps);
  const double r1 = std::max(k2k1_fn(x1), kEps);
  return std::log(r1) - std::log(r0) - mp / (2.0 * r) + 0.5 * psi_g * (x1 - x0);
}

}  // namespace cypha
