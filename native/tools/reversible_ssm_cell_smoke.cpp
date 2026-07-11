/// Regression test for `ReversibleSSMCell` (H11): `reconstruct()` must be an exact algebraic
/// inverse of `forward()` — `x_hat = y - tanh(delta) == x` — since `delta` is cached verbatim
/// rather than re-derived. See docs/reports/STUB_AUDIT_2026-07-11.md.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/reversible_ssm_cell.hpp"

namespace {

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

}  // namespace

int main() {
  using cypha::cyphalm::ReversibleSSMCell;

  // Fresh cell: no pair yet, reconstruct() returns empty.
  {
    ReversibleSSMCell cell;
    assert(!cell.has_pair());
    assert(cell.reconstruct().empty());
  }

  // Basic forward/reconstruct round trip.
  {
    ReversibleSSMCell cell;
    const std::vector<double> x = {0.5, -1.25, 3.0, 0.0, -0.001};
    const std::vector<double> delta = {0.1, -2.0, 0.75, 5.0, -0.3};
    const std::vector<double> y = cell.forward(x, delta);
    assert(cell.has_pair());
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double expected_y = x[i] + std::tanh(delta[i]);
      assert(std::abs(y[i] - expected_y) < 1e-12);
    }
    const std::vector<double> x_hat = cell.reconstruct();
    assert(x_hat.size() == x.size());
    const double diff = max_abs_diff(x_hat, x);
    if (diff >= 1e-9) {
      std::printf("reversible_ssm_cell_smoke: FAIL round-trip diff=%.3e\n", diff);
      return 1;
    }
  }

  // Large/extreme deltas (tanh saturation) still invert exactly.
  {
    ReversibleSSMCell cell;
    const std::vector<double> x = {10.0, -10.0, 0.0, 42.0};
    const std::vector<double> delta = {100.0, -100.0, 0.0, 1e-6};
    cell.forward(x, delta);
    const std::vector<double> x_hat = cell.reconstruct();
    const double diff = max_abs_diff(x_hat, x);
    if (diff >= 1e-9) {
      std::printf("reversible_ssm_cell_smoke: FAIL saturated-delta diff=%.3e\n", diff);
      return 1;
    }
  }

  // delta shorter than x: forward zero-pads; reconstruct must respect the same padding.
  {
    ReversibleSSMCell cell;
    const std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> delta = {0.4};
    const std::vector<double> y = cell.forward(x, delta);
    assert(std::abs(y[1] - x[1]) < 1e-12);  // tanh(0) == 0 for the zero-padded tail
    assert(std::abs(y[2] - x[2]) < 1e-12);
    const std::vector<double> x_hat = cell.reconstruct();
    const double diff = max_abs_diff(x_hat, x);
    if (diff >= 1e-9) {
      std::printf("reversible_ssm_cell_smoke: FAIL short-delta diff=%.3e\n", diff);
      return 1;
    }
  }

  // reset() clears the stored pair.
  {
    ReversibleSSMCell cell;
    cell.forward({1.0, 2.0}, {0.1, 0.2});
    assert(cell.has_pair());
    cell.reset();
    assert(!cell.has_pair());
    assert(cell.reconstruct().empty());
  }

  // Sequential forward calls: reconstruct() always reflects the *latest* forward pair only
  // (this class caches a single (x, delta, y) triple, not a stack — matches its call site in
  // cyphalm_model.cpp, which reconstructs once per step against that step's own forward pair).
  {
    ReversibleSSMCell cell;
    cell.forward({1.0, 1.0}, {0.2, 0.2});
    const std::vector<double> x2 = {5.0, -5.0};
    const std::vector<double> delta2 = {0.9, -0.9};
    cell.forward(x2, delta2);
    const std::vector<double> x_hat = cell.reconstruct();
    const double diff = max_abs_diff(x_hat, x2);
    if (diff >= 1e-9) {
      std::printf("reversible_ssm_cell_smoke: FAIL sequential-forward diff=%.3e\n", diff);
      return 1;
    }
  }

  std::puts("reversible_ssm_cell_smoke: PASS");
  return 0;
}
