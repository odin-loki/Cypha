#include "cypha/intelligence/epistemic_threshold.hpp"

namespace cypha::intelligence {

EpistemicThreshold::EpistemicThreshold(double prior_mu, double prior_lambda)
    : nig_(prior_mu, prior_lambda, 3.0, 0.1) {}

bool EpistemicThreshold::should_correct(double r_eu) const {
  return r_eu > nig_.mean();
}

void EpistemicThreshold::update(double r_eu, bool correction_helped) {
  if (correction_helped) {
    nig_.update(r_eu * 0.9);
  } else {
    nig_.update(r_eu * 1.1);
  }
}

}  // namespace cypha::intelligence
