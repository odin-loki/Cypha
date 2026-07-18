/// Perf §3.1: CPU score_matrix inv_v fold — compare folded accel path to a serial
/// reference with association (H−μ0)*(inv_v*D), and check finite outputs.
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/accel_backend.hpp"

namespace {

void score_matrix_ref(const double* H, int n, int d, int K, const double* mu0, const double* inv_v,
                      const double* D, const double* D_sq, const double* u_k, const double* ctx,
                      double* llr) {
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < K; ++k) {
      double cross = 0.0;
      for (int j = 0; j < d; ++j) {
        const double d_scaled = inv_v[j] * D[k * d + j];
        cross += (H[i * d + j] - mu0[j]) * d_scaled;
      }
      llr[i * K + k] = cross - 0.5 * D_sq[k] - u_k[k] + ctx[k];
    }
  }
}

bool nearly_equal(double a, double b, double atol, double rtol) {
  const double diff = std::abs(a - b);
  return diff <= atol + rtol * std::max(std::abs(a), std::abs(b));
}

}  // namespace

int main() {
  cypha::accel::init();

  constexpr int n = 8;
  constexpr int d = 16;
  constexpr int K = 5;
  constexpr double atol = 1e-12;
  constexpr double rtol = 1e-12;

  std::vector<double> H(static_cast<std::size_t>(n) * d);
  std::vector<double> mu0(static_cast<std::size_t>(d));
  std::vector<double> inv_v(static_cast<std::size_t>(d));
  std::vector<double> D(static_cast<std::size_t>(K) * d);
  std::vector<double> D_sq(static_cast<std::size_t>(K));
  std::vector<double> u_k(static_cast<std::size_t>(K));
  std::vector<double> ctx(static_cast<std::size_t>(K));

  for (int j = 0; j < d; ++j) {
    mu0[static_cast<std::size_t>(j)] = 0.01 * static_cast<double>(j);
    inv_v[static_cast<std::size_t>(j)] = 0.5 + 0.03 * static_cast<double>(j);
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      H[static_cast<std::size_t>(i * d + j)] =
          0.1 * static_cast<double>(i) - 0.02 * static_cast<double>(j);
    }
  }
  for (int k = 0; k < K; ++k) {
    D_sq[static_cast<std::size_t>(k)] = 0.05 * static_cast<double>(k + 1);
    u_k[static_cast<std::size_t>(k)] = 0.001 * static_cast<double>(k);
    ctx[static_cast<std::size_t>(k)] = -0.002 * static_cast<double>(k);
    for (int j = 0; j < d; ++j) {
      D[static_cast<std::size_t>(k * d + j)] =
          0.03 * static_cast<double>(k + 1) - 0.01 * static_cast<double>(j);
    }
  }

  std::vector<double> llr_ref(static_cast<std::size_t>(n) * K, 0.0);
  std::vector<double> llr(static_cast<std::size_t>(n) * K, 0.0);

  score_matrix_ref(H.data(), n, d, K, mu0.data(), inv_v.data(), D.data(), D_sq.data(), u_k.data(),
                   ctx.data(), llr_ref.data());
  cypha::accel::score_matrix(H.data(), n, d, K, mu0.data(), inv_v.data(), D.data(), D_sq.data(),
                             u_k.data(), ctx.data(), llr.data());

  int mismatches = 0;
  for (std::size_t i = 0; i < llr.size(); ++i) {
    if (!std::isfinite(llr[i]) || !std::isfinite(llr_ref[i]) ||
        !nearly_equal(llr[i], llr_ref[i], atol, rtol)) {
      ++mismatches;
      if (mismatches <= 4) {
        std::fprintf(stderr, "mismatch[%zu]: got=%.17g ref=%.17g\n", i, llr[i], llr_ref[i]);
      }
    }
  }
  if (mismatches != 0) {
    std::fprintf(stderr, "score_matrix_inv_v_fold_smoke: FAIL mismatches=%d / %zu device=%s\n",
                 mismatches, llr.size(), cypha::accel::device_info().c_str());
    cypha::accel::shutdown();
    return 1;
  }

  // Uniform inv_v=1: fold is a no-op scale; still must be finite and match ref.
  std::fill(inv_v.begin(), inv_v.end(), 1.0);
  score_matrix_ref(H.data(), n, d, K, mu0.data(), inv_v.data(), D.data(), D_sq.data(), u_k.data(),
                   ctx.data(), llr_ref.data());
  cypha::accel::score_matrix(H.data(), n, d, K, mu0.data(), inv_v.data(), D.data(), D_sq.data(),
                             u_k.data(), ctx.data(), llr.data());
  for (std::size_t i = 0; i < llr.size(); ++i) {
    if (!std::isfinite(llr[i]) || !nearly_equal(llr[i], llr_ref[i], atol, rtol)) {
      std::fprintf(stderr, "score_matrix_inv_v_fold_smoke: FAIL inv_v=1 at %zu\n", i);
      cypha::accel::shutdown();
      return 1;
    }
  }

  std::printf("score_matrix_inv_v_fold_smoke: PASS n=%d d=%d K=%d device=%s\n", n, d, K,
              cypha::accel::device_info().c_str());
  cypha::accel::shutdown();
  return 0;
}
