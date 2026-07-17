// score_matrix_parallel_parity — parallel vs serial RPSM batched LLR within 1e-12.
// Product fixture (small d,K) stays under the work gate; synthetic d=128 K=16 exercises OpenMP.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cypha/parallel_rows.hpp"
#include "cypha/rpsm/psi_matrices.hpp"

namespace {

constexpr double kTol = 1e-12;

bool near_eq(double a, double b) { return std::abs(a - b) <= kTol; }

void set_env_parallel_rows(const char* value) {
#if defined(_WIN32)
  std::string assign = std::string("CYPHA_SCORE_PARALLEL_ROWS=") + value;
  _putenv(assign.c_str());
#else
  setenv("CYPHA_SCORE_PARALLEL_ROWS", value, 1);
#endif
}

cypha::rpsm::PsiMatrices make_synth_psi(int d, int K) {
  cypha::rpsm::PsiMatrices psi;
  psi.feat_dim = d;
  psi.n_classes = K;
  psi.mu.assign(static_cast<std::size_t>((1 + K) * d), 0.0);
  psi.inv_var.assign(static_cast<std::size_t>(d), 1.0);
  psi.counts.assign(static_cast<std::size_t>(K), 10.0);
  psi.v_mean = 1.0;
  for (int j = 0; j < d; ++j) {
    psi.mu[static_cast<std::size_t>(j)] = 0.01 * static_cast<double>(j);
  }
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < d; ++j) {
      psi.mu[static_cast<std::size_t>((1 + k) * d + j)] =
          0.1 * static_cast<double>(k + 1) + 0.001 * static_cast<double>(j);
    }
  }
  return psi;
}

void check_parity_at_n(int n, int d, int K) {
  auto psi = make_synth_psi(d, K);
  std::vector<double> ctx(static_cast<std::size_t>(K), 0.0);
  for (int k = 0; k < K; ++k) {
    ctx[static_cast<std::size_t>(k)] = 0.01 * static_cast<double>(k);
  }
  std::vector<double> H(static_cast<std::size_t>(n * d));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      H[static_cast<std::size_t>(i * d + j)] = 0.02 * static_cast<double>(i) + 0.003 * static_cast<double>(j);
    }
  }

  std::vector<double> llr_par(static_cast<std::size_t>(n * K));
  std::vector<double> llr_ser(static_cast<std::size_t>(n * K));

  set_env_parallel_rows("1");
  if (!cypha::should_parallel_score_rows(n, d, K)) {
    throw std::runtime_error("expected work gate to allow parallel at n=" + std::to_string(n) +
                             " d=" + std::to_string(d) + " K=" + std::to_string(K));
  }
  cypha::rpsm::batched_llr_gemm(H.data(), n, psi, ctx.data(), llr_par.data());

  set_env_parallel_rows("0");
  cypha::rpsm::batched_llr_gemm(H.data(), n, psi, ctx.data(), llr_ser.data());

  for (std::size_t i = 0; i < llr_par.size(); ++i) {
    if (!near_eq(llr_par[i], llr_ser[i])) {
      throw std::runtime_error("parallel vs serial mismatch at n=" + std::to_string(n) + " idx=" +
                               std::to_string(i) + " par=" + std::to_string(llr_par[i]) +
                               " ser=" + std::to_string(llr_ser[i]));
    }
  }
}

}  // namespace

int main() {
  try {
    constexpr int kD = 256;
    constexpr int kK = 32;
    // n=32 work=262k (below 1e6 gate); n=128/256 exercise OpenMP.
    check_parity_at_n(128, kD, kK);
    check_parity_at_n(256, kD, kK);

    set_env_parallel_rows("1");
    std::cout << "score_matrix_parallel_parity OK (synth d=" << kD << " K=" << kK
              << ", n=128/256, tol=" << kTol << ")\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "score_matrix_parallel_parity: " << ex.what() << "\n";
    return 1;
  }
}
