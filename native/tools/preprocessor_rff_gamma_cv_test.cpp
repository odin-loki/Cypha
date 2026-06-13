// Smoke test for estimate_rff_gamma_cv / PreprocessorState::auto_rff_gamma_cv.
#include <cmath>
#include <iostream>
#include <vector>

#include "cypha/numpy_default_rng.hpp"
#include "cypha/preprocessor.hpp"

namespace {

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

bool gamma_in_grid(double g) {
  for (double cand : cypha::default_rff_gamma_cv_grid()) {
    if (near(g, cand, 1e-12)) {
      return true;
    }
  }
  return false;
}

int test_estimate_with_targets() {
  cypha::NumpyDefaultRng rng(0);
  constexpr int n = 120;
  constexpr int d = 4;
  constexpr int rff_d = 32;
  std::vector<double> x(static_cast<std::size_t>(n * d));
  std::vector<double> y(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    double acc = 0.0;
    for (int j = 0; j < d; ++j) {
      const double v = rng.normal(0.0, 1.0);
      x[static_cast<std::size_t>(i * d + j)] = v;
      acc += (j + 1) * v;
    }
    y[static_cast<std::size_t>(i)] = acc;
  }
  const double g =
      cypha::estimate_rff_gamma_cv(x, n, d, rff_d, 42, &y, 1);
  if (!std::isfinite(g) || g <= 0.0 || !gamma_in_grid(g)) {
    std::cerr << "estimate_rff_gamma_cv(y): bad gamma " << g << "\n";
    return 1;
  }
  return 0;
}

int test_fit_auto_cv_without_targets() {
  cypha::NumpyDefaultRng rng(1);
  constexpr int n = 80;
  constexpr int d = 3;
  std::vector<double> x(static_cast<std::size_t>(n * d));
  for (int i = 0; i < n * d; ++i) {
    x[static_cast<std::size_t>(i)] = rng.normal(0.0, 1.0);
  }
  cypha::PreprocessorState pre;
  pre.scale = true;
  pre.pca_dim = -1;
  pre.rff_dim = 24;
  pre.auto_rff_gamma_cv = true;
  pre.auto_rff_gamma = true;
  pre.seed = 7;
  pre.fit_from_design_matrix(x, n, d);
  if (!pre.fitted || pre.rff_w.empty()) {
    std::cerr << "fit auto_rff_gamma_cv: not fitted\n";
    return 1;
  }
  if (!gamma_in_grid(pre.rff_gamma)) {
    std::cerr << "fit auto_rff_gamma_cv: gamma not in grid " << pre.rff_gamma << "\n";
    return 1;
  }
  return 0;
}

int test_cv_differs_from_median() {
  cypha::NumpyDefaultRng rng(2);
  constexpr int n = 100;
  constexpr int d = 5;
  std::vector<double> x(static_cast<std::size_t>(n * d));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      x[static_cast<std::size_t>(i * d + j)] = rng.normal(0.0, 0.1 + 0.3 * static_cast<double>(j));
    }
  }
  cypha::PreprocessorState cv_pre;
  cv_pre.scale = false;
  cv_pre.rff_dim = 40;
  cv_pre.auto_rff_gamma_cv = true;
  cv_pre.seed = 11;
  cv_pre.fit_from_design_matrix(x, n, d);

  cypha::PreprocessorState med_pre;
  med_pre.scale = false;
  med_pre.rff_dim = 40;
  med_pre.auto_rff_gamma = true;
  med_pre.seed = 11;
  med_pre.fit_from_design_matrix(x, n, d);

  if (near(cv_pre.rff_gamma, med_pre.rff_gamma, 1e-12)) {
    std::cerr << "cv vs median: expected different gamma (cv=" << cv_pre.rff_gamma
              << " median=" << med_pre.rff_gamma << ")\n";
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  try {
    if (test_estimate_with_targets() != 0) {
      return 1;
    }
    if (test_fit_auto_cv_without_targets() != 0) {
      return 1;
    }
    if (test_cv_differs_from_median() != 0) {
      return 1;
    }
    std::cout << "preprocessor_rff_gamma_cv_test OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "preprocessor_rff_gamma_cv_test: " << e.what() << "\n";
    return 1;
  }
}
