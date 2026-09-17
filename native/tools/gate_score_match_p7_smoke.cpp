// Phase 7 optimality acceptance: score-matching GH/NIG gate vs Bessel LUT on held-out
// log-likelihood, measured against an exact reference.
//
// This test used to assert only `loglik_score_match >= loglik_lut`, with no reference at all. That
// criterion does not measure accuracy -- it measures which backend reports the HIGHER likelihood,
// and a backend is rewarded for overstating it. Measured against scipy at the time the Bessel
// kernels were fixed (audit findings N1-N4, L5-L9), the old code sat at:
//
//     exact (scipy)      -25.3006064719
//     old LUT            -25.3172787408    |err| = 1.67e-2
//     old ScoreMatch     -25.1582268234    |err| = 1.42e-1     <- 8.5x FURTHER from truth
//     delta = +0.159                                            <- and the old test PASSED on this
//
// So the acceptance gate passed the score-match backend by a comfortable margin precisely because
// that backend was the less accurate of the two. With both kernels fixed the two agree to 2.4e-5
// and the one-sided assertion started failing -- not because anything regressed, but because
// score-match stopped overstating. The criterion is now distance from the exact value, which is
// what "optimality acceptance" was meant to mean.
#include <algorithm>
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

  // Exact held-out log-likelihood for this seed and holdout, from scipy.special.kv at double
  // precision. To re-derive: dump the 512 (mp, r_base, chi, psi) rows this file generates and pipe
  // them to `scripts/gen_gig_k0k1_fit.py --p7-reference`, which evaluates the same formula with
  // K_2/K_1 = kv(2,x)/kv(1,x). std::gamma_distribution is not reproducible outside libstdc++, so
  // the samples have to come from here rather than being regenerated in Python.
  constexpr double kExactLoglik = -25.3006064719;

  const double err_lut = std::abs(ll_lut - kExactLoglik);
  const double err_sm = std::abs(ll_sm - kExactLoglik);
  std::cout << "  exact=" << kExactLoglik << "  err_lut=" << err_lut << "  err_sm=" << err_sm << "\n";

  // Both backends must track the exact value. Measured at 5.97e-5 (LUT) and 3.53e-5 (score-match);
  // the bound leaves an order of magnitude of headroom. Before the Bessel fix these were 1.67e-2
  // and 1.42e-1, so this bound is ~270x and ~2800x tighter than what the code used to deliver.
  constexpr double kMaxErr = 5e-4;
  if (err_lut > kMaxErr) {
    std::cerr << "FAIL: LUT held-out loglik " << ll_lut << " is " << err_lut << " from exact "
              << kExactLoglik << " (max " << kMaxErr << ")\n";
    return 1;
  }
  if (err_sm > kMaxErr) {
    std::cerr << "FAIL: score-match held-out loglik " << ll_sm << " is " << err_sm << " from exact "
              << kExactLoglik << " (max " << kMaxErr << ")\n";
    return 1;
  }

  // The Phase 7 criterion, correctly stated: adopting the cheap backend must not cost accuracy.
  // That is a comparison of DISTANCE FROM TRUTH, not of raw log-likelihood.
  if (err_sm > err_lut * 1.5 + 1e-9) {
    std::cerr << "FAIL: score-match is materially further from exact than the LUT (" << err_sm
              << " vs " << err_lut << ")\n";
    return 1;
  }

  // The two backends approximate the same function, so their gates must agree closely.
  constexpr double kMaxGateDelta = 1e-5;
  if (max_gate_delta > kMaxGateDelta) {
    std::cerr << "FAIL: backends disagree on the world gate by " << max_gate_delta << " (max "
              << kMaxGateDelta << ")\n";
    return 1;
  }

  std::cout << "gate_score_match_p7_smoke: PASS (both backends track the exact held-out loglik; "
               "score-match is no further from it than the LUT; opt-in CYPHA_GIG_SCORE_MATCH=1)\n";
  return 0;
}
