#pragma once

#include <vector>

namespace cypha::cyphalm {

/// H11: RevNet-style additive coupling stub on SSM context (forward + backward reconstruct).
class ReversibleSSMCell {
 public:
  void reset();

  /// ``delta`` is the SSM-produced context increment; returns y = x + f(delta).
  std::vector<double> forward(const std::vector<double>& x, const std::vector<double>& delta);

  /// Backward stub: reconstruct prior x from stored pair (y, delta).
  std::vector<double> backward_stub() const;

  bool has_pair() const { return has_pair_; }

 private:
  std::vector<double> last_x_;
  std::vector<double> last_delta_;
  std::vector<double> last_y_;
  bool has_pair_{false};
};

}  // namespace cypha::cyphalm
