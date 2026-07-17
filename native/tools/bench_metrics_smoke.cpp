/// Smoke test for MC2 (ECE), MS1 (generalization gap), MR1 (CRPS), MR2 (90% interval coverage).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "cypha/bench/bench_metrics.hpp"

namespace {

bool finite_in_unit(double x) {
  return std::isfinite(x) && x >= 0.0 && x <= 1.0;
}

}  // namespace

int main() {
  const std::vector<double> confs = {0.1, 0.2, 0.8, 0.9, 0.85, 0.15};
  const std::vector<double> corr = {0.0, 0.0, 1.0, 1.0, 1.0, 0.0};
  const double ece = cypha::bench::expected_calibration_error(confs, corr, 10);
  assert(finite_in_unit(ece));

  const double gap =
      cypha::bench::accuracy({"a", "b", "c", "d"}, {"a", "b", "c", "d"}) -
      cypha::bench::accuracy({"a", "b", "c", "d"}, {"a", "x", "x", "d"});
  assert(std::isfinite(gap));

  const std::vector<double> y_true = {0.0, 1.0, 2.0, 3.0};
  const std::vector<double> mu = {0.1, 1.0, 2.0, 2.9};
  const std::vector<double> sigma = {0.5, 0.5, 0.5, 0.5};
  const double crps = cypha::bench::crps_gaussian_mean(y_true, mu, sigma);
  const double cov90 = cypha::bench::predictive_interval_coverage(y_true, mu, sigma, 1.645);
  assert(std::isfinite(crps) && crps >= 0.0);
  assert(finite_in_unit(cov90));

  std::printf("bench_metrics_smoke: ece=%.4f gap=%.4f crps=%.4f cov90=%.4f PASS\n", ece, gap, crps, cov90);
  return 0;
}
