// rpsm_spectral_alpha_smoke — Phase -1 Fix 1 regression guard for RPSM_UPGRADE_PLAN.md /
// RESEARCH_STATUS.md:393 ("RPSM core fixes: spectral alpha ... alpha in [0.3, 0.6]").
//
// RPSM_IMPLEMENTATION.md's own "verify first" order (item 1) is: "gria_alpha_spectral in
// [0.3, 0.6] on random Psi". This test checks exactly that, at every configured tier
// (Tiny/Small/Medium/Large, RPSM_IMPLEMENTATION.md:97-102), on Psi ~ N(0,1) ("edge-of-chaos
// init: randn(L,D)*1.0", RPSM_IMPLEMENTATION.md:104). It also checks the function is a pure,
// deterministic function of Psi (same input -> same output) and degrades gracefully on
// degenerate input (all-zero Psi).
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "cypha/rpsm/rpsm_sequence_layer.hpp"

namespace {

bool check_tier(int n_levels, int state_dim, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> psi(static_cast<std::size_t>(n_levels) * static_cast<std::size_t>(state_dim));
  for (auto& v : psi) {
    v = nd(rng);
  }

  const double alpha = cypha::rpsm::gria_alpha_spectral(psi.data(), n_levels, state_dim);
  if (!std::isfinite(alpha)) {
    std::cerr << "rpsm_spectral_alpha_smoke: non-finite alpha at L=" << n_levels << " D=" << state_dim
              << "\n";
    return false;
  }
  if (!(alpha >= 0.3 && alpha <= 0.6)) {
    std::cerr << "rpsm_spectral_alpha_smoke: alpha=" << alpha << " outside [0.3, 0.6] at L="
              << n_levels << " D=" << state_dim << "\n";
    return false;
  }

  // Determinism: identical Psi must give an identical alpha.
  const double alpha2 = cypha::rpsm::gria_alpha_spectral(psi.data(), n_levels, state_dim);
  if (alpha != alpha2) {
    std::cerr << "rpsm_spectral_alpha_smoke: non-deterministic alpha (" << alpha << " vs " << alpha2
              << ")\n";
    return false;
  }

  return true;
}

bool test_tiers_edge_of_chaos() {
  // Tiny/Small/Medium/Large, RPSM_IMPLEMENTATION.md:97-102 (constant L/D=1/32 across tiers).
  const struct { int levels, dim; } tiers[] = {{4, 128}, {8, 256}, {16, 512}, {32, 1024}};
  for (const auto& t : tiers) {
    if (!check_tier(t.levels, t.dim, 1234)) {
      return false;
    }
  }
  // A second, independent seed to make sure the range isn't a lucky draw.
  for (const auto& t : tiers) {
    if (!check_tier(t.levels, t.dim, 999999)) {
      return false;
    }
  }
  return true;
}

bool test_degenerate_zero_psi() {
  const int levels = 4;
  const int dim = 128;
  std::vector<double> psi(static_cast<std::size_t>(levels) * static_cast<std::size_t>(dim), 0.0);
  const double alpha = cypha::rpsm::gria_alpha_spectral(psi.data(), levels, dim);
  if (!std::isfinite(alpha)) {
    std::cerr << "rpsm_spectral_alpha_smoke: non-finite alpha on all-zero Psi\n";
    return false;
  }
  // sigma_max=0 -> normalized=0 -> centered=-1 -> alpha should sit near the lower band edge.
  if (!(alpha >= 0.3 && alpha <= 0.6)) {
    std::cerr << "rpsm_spectral_alpha_smoke: degenerate alpha=" << alpha << " outside [0.3, 0.6]\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  try {
    if (!test_tiers_edge_of_chaos()) {
      return 1;
    }
    if (!test_degenerate_zero_psi()) {
      return 1;
    }
    std::cout << "rpsm_spectral_alpha_smoke OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_spectral_alpha_smoke: " << e.what() << "\n";
    return 1;
  }
}
