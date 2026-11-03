/// Throughput-lock scaffold: measure score_matrix µs/row locally (no CI hard-fail).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/accel_backend.hpp"

int main() {
  cypha::accel::init();

  constexpr int n = 32;
  constexpr int d = 64;
  constexpr int K = 16;
  constexpr int warm = 8;
  constexpr int iters = 64;

  std::vector<double> H(static_cast<std::size_t>(n) * d, 0.01);
  std::vector<double> mu0(static_cast<std::size_t>(d), 0.0);
  std::vector<double> inv_v(static_cast<std::size_t>(d), 1.0);
  std::vector<double> D(static_cast<std::size_t>(K) * d, 0.02);
  std::vector<double> D_sq(static_cast<std::size_t>(K), 0.1);
  std::vector<double> u_k(static_cast<std::size_t>(K), 0.0);
  std::vector<double> ctx(static_cast<std::size_t>(K), 0.0);
  std::vector<double> llr(static_cast<std::size_t>(n) * K, 0.0);

  for (int i = 0; i < warm; ++i) {
    cypha::accel::score_matrix(H.data(), n, d, K, mu0.data(), inv_v.data(), D.data(), D_sq.data(),
                               u_k.data(), ctx.data(), llr.data());
  }

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    cypha::accel::score_matrix(H.data(), n, d, K, mu0.data(), inv_v.data(), D.data(), D_sq.data(),
                               u_k.data(), ctx.data(), llr.data());
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(iters);
  const double us_per_row = us / static_cast<double>(n);

  if (!std::isfinite(us_per_row) || us_per_row <= 0.0) {
    std::puts("throughput_lock_smoke: FAIL (non-finite timing)");
    return 1;
  }
  for (double v : llr) {
    if (!std::isfinite(v)) {
      std::puts("throughput_lock_smoke: FAIL (non-finite score_matrix output)");
      return 1;
    }
  }

  std::printf(
      "throughput_lock_smoke: PASS score_matrix_us_per_row=%.6f n=%d d=%d K=%d device=%s\n",
      us_per_row, n, d, K, cypha::accel::device_info().c_str());
  cypha::accel::shutdown();
  return 0;
}
