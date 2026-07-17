// Phase 7 optimality acceptance: score-matching GH/NIG gate vs Bessel LUT on held-out log-likelihood.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cypha/nig_gig_math.hpp"
#include "cypha/nig_gig_score_match.hpp"

namespace {

struct HoldoutSample {
  double mp;
  double r_base;
  double chi;
  double psi;
};

std::vector<HoldoutSample> make_holdout(std::uint64_t seed, int n) {
  std::mt19937 rng(static_cast<std::uint32_t>(seed & 0xffffffffu));
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::gamma_distribution<double> gamma_mp(2.0, 0.5);
  std::vector<HoldoutSample> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    HoldoutSample s;
    s.chi = std::exp(u01(rng) * (std::log(10.0) - std::log(0.1)) + std::log(0.1));
    s.psi = std::exp(u01(rng) * (std::log(10.0) - std::log(0.1)) + std::log(0.1));
    s.mp = std::max(gamma_mp(rng), 0.0);
    s.r_base = 0.05 + u01(rng) * 1.95;
    out.push_back(s);
  }
  return out;
}

double total_predictive_loglik(const std::vector<HoldoutSample>& samples, double (*k2k1_fn)(double)) {
  double sum = 0.0;
  for (const HoldoutSample& s : samples) {
    sum += cypha::nig_gate_predictive_loglik(s.mp, s.r_base, s.chi, s.psi, k2k1_fn);
  }
  return sum;
}

double max_abs_gate_delta(const std::vector<HoldoutSample>& samples) {
  cypha::set_gig_normalisation_mode_override(cypha::GigNormalisationMode::Lut, true);
  double max_delta = 0.0;
  for (const HoldoutSample& s : samples) {
    const double chi_post = s.chi + s.mp / std::max(s.r_base, 1e-8);
    const double e_lut = cypha::gig_e_inv_v_lam_neg1(chi_post, s.psi);
    cypha::set_gig_normalisation_mode_override(cypha::GigNormalisationMode::ScoreMatch, true);
    const double e_sm = cypha::gig_e_inv_v_lam_neg1(chi_post, s.psi);
    cypha::set_gig_normalisation_mode_override(cypha::GigNormalisationMode::Lut, true);
    const double r_eff_lut = s.r_base / std::max(e_lut, 1e-8);
    const double r_eff_sm = s.r_base / std::max(e_sm, 1e-8);
    const double gate_lut = s.r_base / std::max(r_eff_lut, s.r_base);
    const double gate_sm = s.r_base / std::max(r_eff_sm, s.r_base);
    max_delta = std::max(max_delta, std::abs(gate_lut - gate_sm));
  }
  cypha::set_gig_normalisation_mode_override(cypha::GigNormalisationMode::Lut, false);
  return max_delta;
}

}  // namespace

int main() {
  constexpr int kHoldout = 512;
  constexpr std::uint64_t kSeed = 42424242u;

  const std::vector<HoldoutSample> holdout = make_holdout(kSeed, kHoldout);

  const double ll_lut = total_predictive_loglik(holdout, cypha::gig_k2k1_lut);
  const double ll_sm = total_predictive_loglik(holdout, cypha::gig_k2k1_score_match);
  const double delta = ll_sm - ll_lut;
  const double max_gate_delta = max_abs_gate_delta(holdout);

  std::cout << "gate_score_match_p7_smoke:\n"
            << "  holdout_n=" << kHoldout << "  seed=" << kSeed << "\n"
            << "  loglik_lut=" << ll_lut << "  loglik_sm=" << ll_sm << "  delta=" << delta << "\n"
            << "  max_abs_gate_delta=" << max_gate_delta << "\n";

  constexpr double kLoglikTol = 1e-6;
  const bool loglik_ok = ll_sm >= ll_lut - kLoglikTol;
  if (!loglik_ok) {
    std::cerr << "FAIL: held-out score-match loglik " << ll_sm << " < LUT " << ll_lut << " (tol " << kLoglikTol
              << ")\n";
    return 1;
  }

  std::cout << "gate_score_match_p7_smoke: PASS (score-match loglik >= LUT; LUT retained; opt-in CYPHA_GIG_SCORE_MATCH=1)\n";
  return 0;
}
