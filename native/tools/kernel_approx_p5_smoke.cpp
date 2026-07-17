// Phase 5 optimality acceptance: ‖K̂−K‖_F vs iid/uniform baselines at rff_D=256.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cypha/kernel_memory.hpp"
#include "cypha/rff_features.hpp"

namespace {

std::vector<double> make_synthetic_data(int n, int d, std::uint64_t seed) {
  std::mt19937 rng(static_cast<std::uint32_t>(seed & 0xffffffffu));
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> X(static_cast<std::size_t>(n) * static_cast<std::size_t>(d));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      X[static_cast<std::size_t>(i * d + j)] = nd(rng);
    }
  }
  return X;
}

std::vector<double> uniform_landmarks(const std::vector<double>& X, int n, int d, int m) {
  std::vector<double> landmarks(static_cast<std::size_t>(m) * static_cast<std::size_t>(d));
  m = std::min(m, n);
  for (int i = 0; i < m; ++i) {
    const double* src = X.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    double* dst = landmarks.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    for (int j = 0; j < d; ++j) {
      dst[j] = src[j];
    }
  }
  return landmarks;
}

std::vector<double> leverage_landmarks(const std::vector<double>& X, int n, int d, int m, double gamma,
                                       std::uint64_t seed) {
  std::vector<int> idx;
  cypha::select_leverage_landmark_indices(X.data(), n, d, m, gamma, 1e-6, seed, idx);
  std::vector<double> landmarks(static_cast<std::size_t>(idx.size()) * static_cast<std::size_t>(d));
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
    const double* src = X.data() + static_cast<std::size_t>(idx[static_cast<std::size_t>(i)]) *
                                      static_cast<std::size_t>(d);
    double* dst = landmarks.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    for (int j = 0; j < d; ++j) {
      dst[j] = src[j];
    }
  }
  return landmarks;
}

}  // namespace

int main() {
  constexpr int kN = 48;
  constexpr int kD = 8;
  constexpr int kRffDim = 256;
  constexpr int kLandmarks = 32;
  constexpr int kRffSeeds = 8;
  constexpr std::uint64_t kSeed = 42424242u;

  const std::vector<double> X = make_synthetic_data(kN, kD, kSeed);
  const double gamma = cypha::KernelMemory::auto_gamma_median_heuristic(X.data(), kN, kD, 1.0, kN, kSeed);

  double err_iid_sum = 0.0;
  double err_sorf_sum = 0.0;
  for (int s = 0; s < kRffSeeds; ++s) {
    const std::uint64_t seed = kSeed + static_cast<std::uint64_t>(s) * 9973u;
    std::vector<double> w_iid;
    std::vector<double> b_iid;
    std::vector<double> w_sorf;
    std::vector<double> b_sorf;
    std::mt19937 rng_iid(static_cast<std::uint32_t>(seed & 0xffffffffu));
    std::mt19937 rng_sorf(static_cast<std::uint32_t>(seed & 0xffffffffu));
    cypha::init_rff_weights(cypha::RffProjectionKind::IidGaussian, rng_iid, gamma, kRffDim, kD, w_iid, b_iid, true);
    cypha::init_rff_weights(cypha::RffProjectionKind::Sorf, rng_sorf, gamma, kRffDim, kD, w_sorf, b_sorf, true);
    err_iid_sum += cypha::rff_kernel_frobenius_error(X.data(), kN, kD, w_iid.data(), b_iid.data(), kRffDim, gamma);
    err_sorf_sum += cypha::rff_kernel_frobenius_error(X.data(), kN, kD, w_sorf.data(), b_sorf.data(), kRffDim, gamma);
  }
  const double err_iid = err_iid_sum / static_cast<double>(kRffSeeds);
  const double err_sorf = err_sorf_sum / static_cast<double>(kRffSeeds);

  const std::vector<double> lm_uniform = uniform_landmarks(X, kN, kD, kLandmarks);
  const std::vector<double> lm_leverage = leverage_landmarks(X, kN, kD, kLandmarks, gamma, kSeed);
  const double err_nys_uniform =
      cypha::nystrom_kernel_frobenius_error(X.data(), kN, kD, lm_uniform.data(), kLandmarks, gamma);
  const double err_nys_leverage =
      cypha::nystrom_kernel_frobenius_error(X.data(), kN, kD, lm_leverage.data(), kLandmarks, gamma);

  std::cout << "kernel_approx_p5_smoke:\n"
            << "  gamma=" << gamma << "\n"
            << "  rff_D=" << kRffDim << "  ||K-Khat||_F iid=" << err_iid << " sorf=" << err_sorf << "\n"
            << "  nystrom_m=" << kLandmarks << "  ||K-Khat||_F uniform=" << err_nys_uniform
            << " leverage=" << err_nys_leverage << "\n";

  bool pass = true;
  const bool sorf_ok = err_sorf < err_iid;
  if (!sorf_ok) {
    std::cerr << "WARN: SORF mean Frobenius error not lower than iid at D=256 (opt-in path; not promoted)\n";
  }
  if (!(err_nys_leverage < err_nys_uniform)) {
    std::cerr << "FAIL: leverage-score Nyström Frobenius error not lower than uniform landmarks\n";
    pass = false;
  }

  if (!pass) {
    return 1;
  }
  std::cout << "kernel_approx_p5_smoke: PASS"
            << (sorf_ok ? " (SORF+leverage)" : " (leverage Tier-1; SORF opt-in pending promotion)") << "\n";
  return 0;
}
