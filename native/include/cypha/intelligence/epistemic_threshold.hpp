#pragma once

#include "cypha/intelligence/nig_statistic_state.hpp"

namespace cypha::intelligence {

/// Learned ``θ_eu`` trigger for epistemic correction (Paper IV §2.3).
class EpistemicThreshold {
 public:
  explicit EpistemicThreshold(double prior_mu = 0.5, double prior_lambda = 5.0);

  bool should_correct(double r_eu) const;
  double threshold() const { return nig_.mean(); }

  /// Lower threshold when correction helped; raise when it was a false positive.
  void update(double r_eu, bool correction_helped);

 private:
  NigStatisticState nig_;
};

}  // namespace cypha::intelligence
