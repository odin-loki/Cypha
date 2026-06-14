#include "cypha/intelligence/profile_guided_loss.hpp"

#include <cmath>

namespace cypha::intelligence {

namespace {

ProfileObservation mean_observation(const IntelligenceProfiler& profiler) {
  const auto matrix = profiler.get_profile_matrix();
  ProfileObservation obs;
  obs.alpha = matrix[0][0];
  obs.d_eff = matrix[1][0];
  obs.sigma_branch = matrix[2][0];
  obs.tau = matrix[3][0];
  obs.r_eu = matrix[4][0];
  obs.lipschitz = matrix[5][0];
  obs.calibration = matrix[6][0];
  return obs;
}

}  // namespace

ProfileGuidedLossTerms compute_profile_guided_loss(const ProfileObservation& obs,
                                                   const ProfileGuidedLossConfig& cfg) {
  ProfileGuidedLossTerms out;
  const double dr = obs.r_eu - cfg.target_r_eu;
  const double dt = obs.tau - cfg.target_tau;
  out.r_eu_penalty = cfg.lambda_r_eu * dr * dr;
  out.tau_penalty = cfg.lambda_tau * dt * dt;
  out.total = out.r_eu_penalty + out.tau_penalty;
  return out;
}

ProfileGuidedLossTerms compute_profile_guided_loss_from_profiler(
    const IntelligenceProfiler& profiler, const ProfileGuidedLossConfig& cfg) {
  return compute_profile_guided_loss(mean_observation(profiler), cfg);
}

ProfileGuidedLossGrad compute_profile_guided_loss_grad(const ProfileObservation& obs,
                                                       const ProfileGuidedLossConfig& cfg) {
  ProfileGuidedLossGrad out;
  const double dr = obs.r_eu - cfg.target_r_eu;
  const double dt = obs.tau - cfg.target_tau;
  out.d_alpha_uniform = 2.0 * cfg.lambda_r_eu * dr * 0.15 + 2.0 * cfg.lambda_tau * dt * 0.08;
  return out;
}

ProfileGuidedLossGrad compute_profile_guided_loss_grad_from_profiler(
    const IntelligenceProfiler& profiler, const ProfileGuidedLossConfig& cfg) {
  return compute_profile_guided_loss_grad(mean_observation(profiler), cfg);
}

}  // namespace cypha::intelligence
