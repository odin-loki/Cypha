#pragma once

#include <vector>

#include "cypha/intelligence/intelligence_profiler.hpp"

namespace cypha::intelligence {

/// Paper IV profile-guided navigation loss: weighted squared distance of all seven
/// statistics toward critical targets (defaults from ``IntelligenceProfiler::critical_targets()``).
struct ProfileGuidedLossConfig {
  double lambda_alpha = 0.05;
  double lambda_d_eff = 0.05;
  double lambda_sigma_branch = 0.05;
  double lambda_tau = 0.05;
  double lambda_r_eu = 0.05;
  double lambda_lipschitz = 0.05;
  double lambda_calibration = 0.05;
  double target_alpha = 0.50;
  double target_d_eff = 0.45;
  double target_sigma_branch = 0.50;
  double target_tau = 0.65;
  double target_r_eu = 0.70;
  double target_lipschitz = 0.50;
  double target_calibration = 0.82;
};

/// Default lambdas (0.05 each) and targets from ``IntelligenceProfiler::critical_targets()``.
ProfileGuidedLossConfig default_profile_guided_loss_config();

/// Scale all navigation lambdas by ``clamp(1 - kappa/target_kappa, 0.1, 1.0)`` (low κ → stronger nudge).
ProfileGuidedLossConfig scale_profile_guided_loss_config(ProfileGuidedLossConfig cfg, double kappa,
                                                         double target_kappa = 0.89);

/// Rolling κ state for trajectory-aware λ scheduling (Phase 31).
struct KappaTrajectoryState {
  double ema_kappa = 0.0;
  double prev_ema_kappa = 0.0;
  int sample_count = 0;
};

/// EMA κ scaling plus ``dκ/dt`` boost: falling κ → stronger nudge (``clamp(1 - 4·dκ, 0.75, 1.25)``).
ProfileGuidedLossConfig scale_profile_guided_loss_from_trajectory(
    ProfileGuidedLossConfig cfg, double kappa_now, KappaTrajectoryState& state,
    double target_kappa = 0.89, int ema_window = 16, bool suppress_rise_above_target = false);

/// Scale each stat λ by ``clamp(0.5 + |obs_i - target_i| / span, 0.5, 2.0)`` (Phase 32).
ProfileGuidedLossConfig scale_profile_guided_loss_by_stat_deviation(
    ProfileGuidedLossConfig cfg, const ProfileObservation& obs, double deviation_span = 0.5);

struct AdaptiveNavigationOptions {
  bool use_adaptive_lambdas = false;
  bool use_trajectory_lambdas = false;
  bool use_per_stat_deviation_lambdas = false;
  double target_kappa = 0.89;
  int trajectory_window = 16;
  double deviation_span = 0.5;
  /// When κ exceeds target, weaken navigation λ (Phase 34 joint κ–BPC tuning).
  bool use_kappa_ceiling_lambdas = false;
  /// Multiplier on κ excess when applying ceiling (Phase 35; default 2.5).
  double kappa_ceiling_strength = 2.5;
  /// Minimum λ scale when κ ceiling active (Phase 35; default 0.35).
  double kappa_ceiling_min_scale = 0.35;
  /// When EMA κ exceeds target and is rising, damp trajectory λ boost (Phase 36).
  bool use_kappa_trajectory_ceiling = false;
};

/// Unified κ / trajectory / per-stat deviation λ resolver (train + export).
ProfileGuidedLossConfig resolve_adaptive_profile_guided_config(
    ProfileGuidedLossConfig cfg, const ProfileObservation& obs,
    const AdaptiveNavigationOptions& opts, KappaTrajectoryState* trajectory_state = nullptr);

/// Per-stat observation minus target (for export telemetry).
struct ProfileStatDeltas {
  double alpha = 0.0;
  double d_eff = 0.0;
  double sigma_branch = 0.0;
  double tau = 0.0;
  double r_eu = 0.0;
  double lipschitz = 0.0;
  double calibration = 0.0;
};

ProfileStatDeltas profile_stat_deltas(const ProfileObservation& obs,
                                      const ProfileGuidedLossConfig& cfg);

struct ProfileGuidedLossTerms {
  double alpha_penalty = 0.0;
  double d_eff_penalty = 0.0;
  double sigma_branch_penalty = 0.0;
  double tau_penalty = 0.0;
  double r_eu_penalty = 0.0;
  double lipschitz_penalty = 0.0;
  double calibration_penalty = 0.0;
  /// Unweighted ``||P - P*||²`` (same as ``IntelligenceProfiler::navigation_loss`` w.r.t. cfg targets).
  double navigation_loss_total = 0.0;
  double total = 0.0;
};

struct ProfileGuidedLossGrad {
  /// Uniform nudge added to every GRIA ``d_alpha`` entry (α / r_eu / σ coupling).
  double d_alpha_uniform = 0.0;
  /// Uniform nudge added to every LSTM output logit (D_eff / τ / C coupling).
  double d_logit_uniform = 0.0;
  /// Uniform nudge on LSTM ``h_new`` (Paper IV hidden-state D_eff regularizer).
  double d_h_hidden_uniform = 0.0;
  /// Optional input-direction nudge on GRIA field input (length = field dim when used).
  std::vector<double> d_gria_input;
};

/// Extra uniform nudges when κ exceeds target (Phase 36 joint κ–BPC tuning).
ProfileGuidedLossGrad kappa_excess_grad_nudge(const ProfileObservation& obs, double kappa,
                                              double target_kappa, double strength = 1.0,
                                              double margin = 0.02);

/// Scale kernel LLR blend down when κ exceeds target (Phase 38 criticality routing).
double scale_kernel_blend_from_kappa(double base_blend, double kappa, double target_kappa,
                                     double floor = 0.08);

/// Damp navigation warmup ramp when κ exceeds target (Phase 40 joint tuning).
double scale_navigation_warmup_from_kappa(double base_warmup, double kappa, double target_kappa,
                                          double strength = 0.35, double floor = 0.65);

ProfileGuidedLossTerms compute_profile_guided_loss(
    const ProfileObservation& obs,
    const ProfileGuidedLossConfig& cfg = default_profile_guided_loss_config());

ProfileGuidedLossTerms compute_profile_guided_loss_from_profiler(
    const IntelligenceProfiler& profiler,
    const ProfileGuidedLossConfig& cfg = default_profile_guided_loss_config());

/// Gradients of ``compute_profile_guided_loss`` w.r.t. profiler means (for train-step backprop).
ProfileGuidedLossGrad compute_profile_guided_loss_grad(
    const ProfileObservation& obs, const ProfileGuidedLossConfig& cfg = default_profile_guided_loss_config(),
    int gria_field_dim = 0);

ProfileGuidedLossGrad compute_profile_guided_loss_grad_from_profiler(
    const IntelligenceProfiler& profiler,
    const ProfileGuidedLossConfig& cfg = default_profile_guided_loss_config(), int gria_field_dim = 0);

}  // namespace cypha::intelligence
