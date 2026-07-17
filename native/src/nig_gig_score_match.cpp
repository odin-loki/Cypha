#include "cypha/nig_gig_score_match.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <optional>

#include "cypha/env.hpp"

namespace cypha {

namespace {

constexpr double kEps = 1e-8;

/// Coefficients from offline Hyvärinen score-matching fit of ``z(x)=x·K_2/K_1`` on ``x∈[10⁻⁶,120]``
/// (mean rel-err ``≈4.8×10⁻⁴`` vs reference ``scipy.special.kv``; max rel-err ``≈1.3×10⁻³``).
struct ScoreMatchK2K1Coeffs {
  static constexpr double a0 = 1.99945961;
  static constexpr double a1 = 3.84404161;
  static constexpr double a2 = 1.84564964;
  static constexpr double b1 = 1.85102869;
  static constexpr double b2 = -5.84011815e-05;
};

std::atomic<int> g_mode_override{-1};  // -1 = use env; else static_cast<int>(GigNormalisationMode)

double rat21_z(double x) {
  const double x2 = x * x;
  const double num = ScoreMatchK2K1Coeffs::a0 + ScoreMatchK2K1Coeffs::a1 * x +
                     ScoreMatchK2K1Coeffs::a2 * x2;
  const double den = 1.0 + ScoreMatchK2K1Coeffs::b1 * x + ScoreMatchK2K1Coeffs::b2 * x2;
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

double gig_k2k1_score_match(double x) {
  if (x <= 1e-8) {
    return 2.0 / std::max(x, 1e-8);
  }
  if (x > 120.0) {
    return 1.0 + 1.5 / x + 6.75 / (x * x);
  }
  const double z = rat21_z(x);
  return z / x;
}

double gig_k0k1_score_match(double x) {
  // Exact recurrence: K_2 - K_0 = (2/x) K_1  =>  K_0/K_1 = K_2/K_1 - 2/x.
  return gig_k2k1_score_match(x) - 2.0 / std::max(x, kEps);
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
