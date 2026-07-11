#pragma once

#include <vector>

namespace cypha::cyphalm {

/// H11: RevNet-style additive coupling on SSM context (forward + exact analytic backward reconstruct).
///
/// Forward: ``y = x + tanh(delta)``. Because ``delta`` is cached verbatim (not recomputed) from the
/// forward call, ``reconstruct()`` is an *exact* algebraic inverse — ``y - tanh(delta) == x`` to
/// floating-point precision — not an approximation. See `reversible_ssm_cell.cpp` for the derivation.
class ReversibleSSMCell {
 public:
  void reset();

  /// ``delta`` is the SSM-produced context increment; returns y = x + f(delta).
  std::vector<double> forward(const std::vector<double>& x, const std::vector<double>& delta);

  /// Reconstruct prior x from the stored pair (y, delta): ``x_hat = y - tanh(delta)``.
  /// Exact inverse of `forward()` (see class docstring), not a numerical approximation.
  std::vector<double> reconstruct() const;

  bool has_pair() const { return has_pair_; }

 private:
  std::vector<double> last_x_;
  std::vector<double> last_delta_;
  std::vector<double> last_y_;
  bool has_pair_{false};
};

}  // namespace cypha::cyphalm
