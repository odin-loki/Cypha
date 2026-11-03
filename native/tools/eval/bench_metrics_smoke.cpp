/// Smoke test for MC1 (macro-F1, balanced accuracy), MC2 (ECE), MC4 (margin distribution),
/// MS1 (generalization gap), MR1 (CRPS), MR2 (90% interval coverage), MR3 (residual diagnostics).
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

  // MC4: margins 2, 2, 1, 10 → mean=3.75, p50=2, p10=1.3 (linear interp)
  const double row0[] = {3.0, 1.0};
  const double row1[] = {1.0, 3.0};
  const double row2[] = {5.0, 4.0};
  const double row3[] = {10.0, 0.0};
  const std::vector<double> mc4_margins = {
      cypha::bench::logit_margin_top2(row0, 2), cypha::bench::logit_margin_top2(row1, 2),
      cypha::bench::logit_margin_top2(row2, 2), cypha::bench::logit_margin_top2(row3, 2),
  };
  const cypha::bench::MarginDistribution md = cypha::bench::margin_distribution(mc4_margins);
  if (!std::isfinite(md.mean) || !std::isfinite(md.p50) || !std::isfinite(md.p10) || md.mean < 0.0 ||
      md.p50 < 0.0 || md.p10 < 0.0 || std::abs(md.mean - 3.75) > 1e-12 || std::abs(md.p50 - 2.0) > 1e-12 ||
      std::abs(md.p10 - 1.3) > 1e-12) {
    std::fprintf(stderr, "MC4 fixture failed: mean=%.12f p50=%.12f p10=%.12f\n", md.mean, md.p50, md.p10);
    return 1;
  }

  const std::vector<double> y_ac = {2.0, 0.0, 2.0, 0.0};
  const std::vector<double> p_ac = {1.0, 1.0, 1.0, 1.0};
  const double resid_ac = cypha::bench::residual_autocorr_lag1(y_ac, p_ac);
  const double resid_sf = cypha::bench::residual_spectral_flatness(y_ac, p_ac);
  if (!std::isfinite(resid_ac) || !std::isfinite(resid_sf) || std::abs(resid_ac + 0.75) > 1e-12 ||
      resid_sf < 0.0 || resid_sf > 1.0) {
    std::fprintf(stderr, "MR3 fixture failed: resid_ac=%.12f resid_sf=%.12f\n", resid_ac, resid_sf);
    return 1;
  }

  std::printf(
      "bench_metrics_smoke: macro_f1=%.4f bal_acc=%.4f ece=%.4f gap=%.4f crps=%.4f cov90=%.4f "
      "margin_mean=%.4f margin_p50=%.4f margin_p10=%.4f resid_ac=%.4f resid_sf=%.4f PASS\n",
      macro_f1, bal_acc, ece, gap, crps, cov90, md.mean, md.p50, md.p10, resid_ac, resid_sf);
  return 0;
}
