/// Smoke test for MC2 (ECE) and MS1 (generalization gap helpers).
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

  std::printf("bench_metrics_smoke: ece=%.4f gap=%.4f PASS\n", ece, gap);
  return 0;
}
