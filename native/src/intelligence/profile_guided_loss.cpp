#include "cypha/intelligence/profile_guided_loss.hpp"

#include <algorithm>
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

double weighted_sq_penalty(double lambda, double delta) {
  return lambda * delta * delta;
}

}  // namespace

ProfileGuidedLossConfig default_profile_guided_loss_config() {
  ProfileGuidedLossConfig cfg;
  const auto targets = IntelligenceProfiler::critical_targets();
  cfg.target_alpha = targets[0];
  cfg.target_d_eff = targets[1];
  cfg.target_sigma_branch = targets[2];
  cfg.target_tau = targets[3];
  cfg.target_r_eu = targets[4];
  cfg.target_lipschitz = targets[5];
  cfg.target_calibration = targets[6];
  return cfg;
}

namespace {

void scale_all_lambdas(ProfileGuidedLossConfig& cfg, double factor) {
  cfg.lambda_alpha *= factor;
  cfg.lambda_d_eff *= factor;
  cfg.lambda_sigma_branch *= factor;
  cfg.lambda_tau *= factor;
  cfg.lambda_r_eu *= factor;
  cfg.lambda_lipschitz *= factor;
  cfg.lambda_calibration *= factor;
}

}  // namespace

ProfileGuidedLossConfig scale_profile_guided_loss_config(ProfileGuidedLossConfig cfg, double kappa,
                                                         double target_kappa) {
  scale_all_lambdas(cfg, std::clamp(1.0 - kappa / target_kappa, 0.1, 1.0));
  return cfg;
}

ProfileGuidedLossConfig scale_profile_guided_loss_from_trajectory(
    ProfileGuidedLossConfig cfg, double kappa_now, KappaTrajectoryState& state,
    double target_kappa, int ema_window, bool suppress_rise_above_target) {
  state.prev_ema_kappa = state.ema_kappa;
  const double ema_alpha =
      ema_window > 0 ? 2.0 / (static_cast<double>(ema_window) + 1.0) : 1.0;
  if (state.sample_count == 0) {
    state.ema_kappa = kappa_now;
  } else {
    state.ema_kappa = ema_alpha * kappa_now + (1.0 - ema_alpha) * state.ema_kappa;
  }
  ++state.sample_count;

  cfg = scale_profile_guided_loss_config(cfg, state.ema_kappa, target_kappa);
  if (state.sample_count > 1) {
    const double dkappa = state.ema_kappa - state.prev_ema_kappa;
    double traj_scale = 1.0 - 4.0 * dkappa;
    if (suppress_rise_above_target && state.ema_kappa > target_kappa && dkappa > 0.0) {
      traj_scale = 1.0 - 8.0 * dkappa;
    }
    scale_all_lambdas(cfg, std::clamp(traj_scale, 0.45, 1.25));
  }
  return cfg;
}

ProfileGuidedLossConfig scale_profile_guided_loss_by_stat_deviation(ProfileGuidedLossConfig cfg,
                                                                   const ProfileObservation& obs,
                                                                   double deviation_span) {
  const double span = std::max(deviation_span, 1e-6);
  const auto scale_one = [&](double delta) {
    return std::clamp(0.5 + std::abs(delta) / span, 0.5, 2.0);
  };
  cfg.lambda_alpha *= scale_one(obs.alpha - cfg.target_alpha);
  cfg.lambda_d_eff *= scale_one(obs.d_eff - cfg.target_d_eff);
  cfg.lambda_sigma_branch *= scale_one(obs.sigma_branch - cfg.target_sigma_branch);
  cfg.lambda_tau *= scale_one(obs.tau - cfg.target_tau);
  cfg.lambda_r_eu *= scale_one(obs.r_eu - cfg.target_r_eu);
  cfg.lambda_lipschitz *= scale_one(obs.lipschitz - cfg.target_lipschitz);
  cfg.lambda_calibration *= scale_one(obs.calibration - cfg.target_calibration);
  return cfg;
}

ProfileGuidedLossConfig resolve_adaptive_profile_guided_config(
    ProfileGuidedLossConfig cfg, const ProfileObservation& obs,
    const AdaptiveNavigationOptions& opts, KappaTrajectoryState* trajectory_state) {
  if (opts.use_adaptive_lambdas) {
    const double kappa = IntelligenceProfiler::criticality_score_for(obs);
    if (opts.use_trajectory_lambdas && trajectory_state != nullptr) {
      cfg = scale_profile_guided_loss_from_trajectory(cfg, kappa, *trajectory_state,
                                                      opts.target_kappa, opts.trajectory_window,
                                                      opts.use_kappa_trajectory_ceiling);
    } else {
      cfg = scale_profile_guided_loss_config(cfg, kappa, opts.target_kappa);
    }
  }
  if (opts.use_per_stat_deviation_lambdas) {
    cfg = scale_profile_guided_loss_by_stat_deviation(cfg, obs, opts.deviation_span);
  }
  if (opts.use_kappa_ceiling_lambdas) {
    const double kappa = IntelligenceProfiler::criticality_score_for(obs);
    if (kappa > opts.target_kappa) {
      const double excess =
          (kappa - opts.target_kappa) / std::max(1e-6, 1.0 - opts.target_kappa);
      const double strength = std::max(opts.kappa_ceiling_strength, 0.0);
      const double min_scale = std::clamp(opts.kappa_ceiling_min_scale, 0.05, 1.0);
      scale_all_lambdas(cfg,
                        std::clamp(1.0 - strength * excess, min_scale, 1.0));
    }
  }
  return cfg;
}

ProfileGuidedLossGrad kappa_excess_grad_nudge(const ProfileObservation& obs, double kappa,
                                              double target_kappa, double strength,
                                              double margin) {
  ProfileGuidedLossGrad out;
  const double m = std::max(margin, 0.0);
  if (kappa <= target_kappa + m || strength <= 0.0) {
    return out;
  }
  const double excess =
      (kappa - target_kappa - m) / std::max(1e-6, 1.0 - target_kappa - m);
  const double s = strength * std::clamp(excess, 0.0, 1.0);
  const auto targets = IntelligenceProfiler::critical_targets();
  const double d_alpha = obs.alpha - targets[0];
  const double d_tau = obs.tau - targets[3];
  const double d_d_eff = obs.d_eff - targets[1];
  out.d_alpha_uniform = -0.10 * s - 0.04 * d_alpha * s;
  out.d_logit_uniform = 0.08 * s * d_tau + 0.06 * s * d_d_eff;
  out.d_h_hidden_uniform = 0.05 * s * d_d_eff;
  return out;
}

double scale_kernel_blend_from_kappa(double base_blend, double kappa, double target_kappa,
                                     double floor) {
  const double b = std::max(0.0, base_blend);
  const double f = std::max(0.0, floor);
  if (kappa <= target_kappa) {
    return b;
  }
  const double excess = (kappa - target_kappa) / std::max(1e-6, 1.0 - target_kappa);
  const double scale = 1.0 - std::clamp(excess, 0.0, 1.0);
  return std::max(f, b * scale);
}

double scale_navigation_warmup_from_kappa(double base_warmup, double kappa, double target_kappa,
                                          double strength, double floor) {
  const double w = std::clamp(base_warmup, 0.0, 1.0);
  const double f = std::clamp(floor, 0.0, 1.0);
  if (kappa <= target_kappa || strength <= 0.0) {
    return w;
  }
  const double excess = (kappa - target_kappa) / std::max(1e-6, 1.0 - target_kappa);
  const double scale = 1.0 - strength * std::clamp(excess, 0.0, 1.0);
  return std::max(f, w * scale);
}

ProfileStatDeltas profile_stat_deltas(const ProfileObservation& obs,
                                      const ProfileGuidedLossConfig& cfg) {
  ProfileStatDeltas d;
  d.alpha = obs.alpha - cfg.target_alpha;
  d.d_eff = obs.d_eff - cfg.target_d_eff;
  d.sigma_branch = obs.sigma_branch - cfg.target_sigma_branch;
  d.tau = obs.tau - cfg.target_tau;
  d.r_eu = obs.r_eu - cfg.target_r_eu;
  d.lipschitz = obs.lipschitz - cfg.target_lipschitz;
  d.calibration = obs.calibration - cfg.target_calibration;
  return d;
}

ProfileGuidedLossTerms compute_profile_guided_loss(const ProfileObservation& obs,
                                                   const ProfileGuidedLossConfig& cfg) {
  ProfileGuidedLossTerms out;
  const double d_alpha = obs.alpha - cfg.target_alpha;
  const double d_d_eff = obs.d_eff - cfg.target_d_eff;
  const double d_sigma = obs.sigma_branch - cfg.target_sigma_branch;
  const double d_tau = obs.tau - cfg.target_tau;
  const double d_r_eu = obs.r_eu - cfg.target_r_eu;
  const double d_lipschitz = obs.lipschitz - cfg.target_lipschitz;
  const double d_calibration = obs.calibration - cfg.target_calibration;

  out.alpha_penalty = weighted_sq_penalty(cfg.lambda_alpha, d_alpha);
  out.d_eff_penalty = weighted_sq_penalty(cfg.lambda_d_eff, d_d_eff);
  out.sigma_branch_penalty = weighted_sq_penalty(cfg.lambda_sigma_branch, d_sigma);
  out.tau_penalty = weighted_sq_penalty(cfg.lambda_tau, d_tau);
  out.r_eu_penalty = weighted_sq_penalty(cfg.lambda_r_eu, d_r_eu);
  out.lipschitz_penalty = weighted_sq_penalty(cfg.lambda_lipschitz, d_lipschitz);
  out.calibration_penalty = weighted_sq_penalty(cfg.lambda_calibration, d_calibration);

  out.navigation_loss_total = d_alpha * d_alpha + d_d_eff * d_d_eff + d_sigma * d_sigma +
                              d_tau * d_tau + d_r_eu * d_r_eu + d_lipschitz * d_lipschitz +
                              d_calibration * d_calibration;
  out.total = out.alpha_penalty + out.d_eff_penalty + out.sigma_branch_penalty + out.tau_penalty +
              out.r_eu_penalty + out.lipschitz_penalty + out.calibration_penalty;
  return out;
}

ProfileGuidedLossTerms compute_profile_guided_loss_from_profiler(
    const IntelligenceProfiler& profiler, const ProfileGuidedLossConfig& cfg) {
  return compute_profile_guided_loss(mean_observation(profiler), cfg);
}

ProfileGuidedLossGrad compute_profile_guided_loss_grad(const ProfileObservation& obs,
                                                       const ProfileGuidedLossConfig& cfg,
                                                       int gria_field_dim) {
  ProfileGuidedLossGrad out;
  const double d_alpha = obs.alpha - cfg.target_alpha;
  const double d_r_eu = obs.r_eu - cfg.target_r_eu;
  const double d_sigma = obs.sigma_branch - cfg.target_sigma_branch;
  const double d_d_eff = obs.d_eff - cfg.target_d_eff;
  const double d_tau = obs.tau - cfg.target_tau;
  const double d_calibration = obs.calibration - cfg.target_calibration;
  out.d_alpha_uniform = 2.0 * cfg.lambda_alpha * d_alpha * 0.10 +
                        2.0 * cfg.lambda_r_eu * d_r_eu * 0.15 +
                        2.0 * cfg.lambda_sigma_branch * d_sigma * 0.12;
  out.d_logit_uniform = 2.0 * cfg.lambda_d_eff * d_d_eff * 0.08 +
                        2.0 * cfg.lambda_tau * d_tau * 0.10 +
                        2.0 * cfg.lambda_calibration * d_calibration * 0.08;
  out.d_h_hidden_uniform = 2.0 * cfg.lambda_d_eff * d_d_eff * 0.08;

  const double d_lipschitz = obs.lipschitz - cfg.target_lipschitz;
  if (gria_field_dim > 0 && std::abs(d_lipschitz) > 1e-12) {
    const double nudge = cfg.lambda_lipschitz * d_lipschitz;
    out.d_gria_input.assign(static_cast<std::size_t>(gria_field_dim), nudge);
  }
  return out;
}

ProfileGuidedLossGrad compute_profile_guided_loss_grad_from_profiler(
    const IntelligenceProfiler& profiler, const ProfileGuidedLossConfig& cfg, int gria_field_dim) {
  return compute_profile_guided_loss_grad(mean_observation(profiler), cfg, gria_field_dim);
}

}  // namespace cypha::intelligence
