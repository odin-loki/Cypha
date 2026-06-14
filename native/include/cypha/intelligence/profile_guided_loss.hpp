#pragma once

#include "cypha/intelligence/intelligence_profiler.hpp"

namespace cypha::intelligence {

/// Paper IV profile-guided regularizers on ``r_eu`` and ``τ`` toward critical targets.
struct ProfileGuidedLossConfig {
  double lambda_r_eu = 0.1;
  double lambda_tau = 0.1;
  double target_r_eu = 0.5;
  double target_tau = 0.5;
};

struct ProfileGuidedLossTerms {
  double r_eu_penalty = 0.0;
  double tau_penalty = 0.0;
  double total = 0.0;
};

ProfileGuidedLossTerms compute_profile_guided_loss(const ProfileObservation& obs,
                                                   const ProfileGuidedLossConfig& cfg = {});

ProfileGuidedLossTerms compute_profile_guided_loss_from_profiler(
    const IntelligenceProfiler& profiler, const ProfileGuidedLossConfig& cfg = {});

}  // namespace cypha::intelligence
