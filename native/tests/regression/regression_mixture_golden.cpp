// CTest: scalar mixture combine vs fixed reference (matches `DIFRegressor.predict` for d=1).
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "cypha/regression_stub.hpp"

int main() {
  constexpr std::size_t k = 3;
  const double probs[] = {0.2, 0.5, 0.3};
  const double mu[] = {1.0, 2.0, 3.0};
  const double var[] = {0.1, 0.4, 0.2};

  double y = 0.0;
  double u = 0.0;
  cypha::regression::predict_mixture_scalar(probs, mu, var, k, y, u);

  const double y_exp = 2.1;
  const double v_exp = 0.28;
  const double u_exp = std::sqrt(v_exp);

  if (std::abs(y - y_exp) > 1e-15 || std::abs(u - u_exp) > 1e-15) {
    std::cerr << "regression_mixture_golden: want y=" << y_exp << " u=" << u_exp << " got y=" << y
              << " u=" << u << "\n";
    return 1;
  }
  if (cypha::regression::native_regression_milestone() < 2) {
    std::cerr << "native_regression_milestone expected >= 2\n";
    return 1;
  }
  return 0;
}
