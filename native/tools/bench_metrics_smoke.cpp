/// Smoke test for MC1 (macro-F1, balanced accuracy), MC2 (ECE), MS1 (generalization gap),
/// MR1 (CRPS), MR2 (90% interval coverage).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "cypha/bench/bench_metrics.hpp"

namespace {

bool finite_in_unit(double x) {
  return std::isfinite(x) && x >= 0.0 && x <= 1.0;
}

}  // namespace

int main() {
  const std::vector<std::string> y_cls_true = {"a", "a", "b", "b"};
  const std::vector<std::string> y_cls_pred = {"a", "b", "b", "b"};
  const double macro_f1 = cypha::bench::f1_macro(y_cls_true, y_cls_pred);
  const double bal_acc = cypha::bench::balanced_accuracy(y_cls_true, y_cls_pred);
  // a: P=1 R=0.5 F1=2/3; b: P=2/3 R=1 F1=0.8 → macro = 11/15
  if (!finite_in_unit(macro_f1) || !finite_in_unit(bal_acc) ||
      std::abs(macro_f1 - 11.0 / 15.0) > 1e-12 || std::abs(bal_acc - 0.75) > 1e-12) {
    std::fprintf(stderr, "MC1 fixture failed: macro_f1=%.12f bal_acc=%.12f\n", macro_f1, bal_acc);
    return 1;
  }

  const std::vector<double> confs = {0.1, 0.2, 0.8, 0.9, 0.85, 0.15};
  const std::vector<double> corr = {0.0, 0.0, 1.0, 1.0, 1.0, 0.0};
  const double ece = cypha::bench::expected_calibration_error(confs, corr, 10);
  assert(finite_in_unit(ece));

  const double gap =
      cypha::bench::accuracy({"a", "b", "c", "d"}, {"a", "b", "c", "d"}) -
      cypha::bench::accuracy({"a", "b", "c", "d"}, {"a", "x", "x", "d"});
  assert(std::isfinite(gap));

  const std::vector<double> y_reg = {0.0, 1.0, 2.0, 3.0};
  const std::vector<double> mu = {0.1, 1.0, 2.0, 2.9};
  const std::vector<double> sigma = {0.5, 0.5, 0.5, 0.5};
  const double crps = cypha::bench::crps_gaussian_mean(y_reg, mu, sigma);
  const double cov90 = cypha::bench::predictive_interval_coverage(y_reg, mu, sigma, 1.645);
  assert(std::isfinite(crps) && crps >= 0.0);
  assert(finite_in_unit(cov90));

  std::printf(
      "bench_metrics_smoke: macro_f1=%.4f bal_acc=%.4f ece=%.4f gap=%.4f crps=%.4f cov90=%.4f PASS\n", macro_f1,
      bal_acc, ece, gap, crps, cov90);
  return 0;
}
