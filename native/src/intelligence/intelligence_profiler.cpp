#include "cypha/intelligence/intelligence_profiler.hpp"

#include "cypha/intelligence/measurers.hpp"

#include <cmath>

namespace cypha::intelligence {

namespace {

constexpr double kHealthEps = 1e-8;
constexpr double kMaxProfileDistance = std::sqrt(7.0);

ProfileObservation make_observation(double alpha, double d_eff, double sigma_branch, double tau,
                                    double r_eu, double lipschitz, double calibration) {
  ProfileObservation obs;
  obs.alpha = alpha;
  obs.d_eff = d_eff;
  obs.sigma_branch = sigma_branch;
  obs.tau = tau;
  obs.r_eu = r_eu;
  obs.lipschitz = lipschitz;
  obs.calibration = calibration;
  return obs;
}

}  // namespace

IntelligenceProfiler::IntelligenceProfiler() = default;

std::array<double, kProfileStatisticCount> IntelligenceProfiler::critical_targets() {
  return {0.50, 0.45, 0.50, 0.65, 0.70, 0.50, 0.82};
}

double IntelligenceProfiler::axis_distance(double value, double target) {
  return std::abs(value - target);
}

ProfileObservation IntelligenceProfiler::landscape_reference(LandscapeSystemClass system) {
  switch (system) {
    case LandscapeSystemClass::SimpleFfn:
      return make_observation(0.75, 0.15, 0.35, 0.05, 0.10, 0.70, 0.45);
    case LandscapeSystemClass::LargeTransformer:
      return make_observation(0.52, 0.35, 0.50, 0.55, 0.20, 0.45, 0.60);
    case LandscapeSystemClass::CyphaAugmented:
      return make_observation(0.52, 0.35, 0.50, 0.55, 0.75, 0.45, 0.85);
    case LandscapeSystemClass::SelfCorrectingCypha:
      return make_observation(0.50, 0.45, 0.50, 0.65, 0.70, 0.50, 0.82);
    case LandscapeSystemClass::SoftWorldCypha:
      return make_observation(0.50, 0.58, 0.50, 0.87, 0.65, 0.50, 0.88);
    case LandscapeSystemClass::HumanMedian:
      return make_observation(0.50, 0.50, 0.50, 0.65, 0.50, 0.50, 0.70);
  }
  return make_observation(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
}

double IntelligenceProfiler::profile_distance(const ProfileObservation& a, const ProfileObservation& b) {
  const double d_alpha = a.alpha - b.alpha;
  const double d_d_eff = a.d_eff - b.d_eff;
  const double d_sigma = a.sigma_branch - b.sigma_branch;
  const double d_tau = a.tau - b.tau;
  const double d_r_eu = a.r_eu - b.r_eu;
  const double d_l = a.lipschitz - b.lipschitz;
  const double d_c = a.calibration - b.calibration;
  return std::sqrt(d_alpha * d_alpha + d_d_eff * d_d_eff + d_sigma * d_sigma + d_tau * d_tau +
                   d_r_eu * d_r_eu + d_l * d_l + d_c * d_c);
}

double IntelligenceProfiler::profile_distance_normalized(const ProfileObservation& a,
                                                         const ProfileObservation& b) {
  return profile_distance(a, b) / kMaxProfileDistance;
}

double IntelligenceProfiler::navigation_loss(const ProfileObservation& obs) {
  const auto targets = critical_targets();
  const double d_alpha = obs.alpha - targets[0];
  const double d_d_eff = obs.d_eff - targets[1];
  const double d_sigma = obs.sigma_branch - targets[2];
  const double d_tau = obs.tau - targets[3];
  const double d_r_eu = obs.r_eu - targets[4];
  const double d_l = obs.lipschitz - targets[5];
  const double d_c = obs.calibration - targets[6];
  return d_alpha * d_alpha + d_d_eff * d_d_eff + d_sigma * d_sigma + d_tau * d_tau +
         d_r_eu * d_r_eu + d_l * d_l + d_c * d_c;
}

FailureModeFlags IntelligenceProfiler::predict_failure_modes(const ProfileObservation& obs) {
  FailureModeFlags flags;
  flags.low_calibration = obs.calibration < 0.40;
  flags.high_lipschitz = obs.lipschitz > 0.65;
  flags.low_tau = obs.tau < 0.15;
  flags.explosive_branching = obs.sigma_branch > 0.70;
  flags.damped_branching = obs.sigma_branch < 0.30;
  flags.low_d_eff = obs.d_eff < 0.20;
  flags.low_r_eu = obs.r_eu < 0.20;
  flags.extreme_alpha = obs.alpha < 0.30 || obs.alpha > 0.80;
  return flags;
}

bool IntelligenceProfiler::dominates(const ProfileObservation& a, const ProfileObservation& b) {
  const auto targets = critical_targets();
  const double values_a[kProfileStatisticCount] = {a.alpha, a.d_eff, a.sigma_branch, a.tau, a.r_eu,
                                                   a.lipschitz, a.calibration};
  const double values_b[kProfileStatisticCount] = {b.alpha, b.d_eff, b.sigma_branch, b.tau, b.r_eu,
                                                   b.lipschitz, b.calibration};
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    if (axis_distance(values_a[i], targets[i]) >= axis_distance(values_b[i], targets[i])) {
      return false;
    }
  }
  return true;
}

double IntelligenceProfiler::criticality_score_for(const ProfileObservation& obs) {
  const auto targets = critical_targets();
  double deviation_sum = 0.0;
  deviation_sum += axis_distance(obs.alpha, targets[0]);
  deviation_sum += axis_distance(obs.d_eff, targets[1]);
  deviation_sum += axis_distance(obs.sigma_branch, targets[2]);
  deviation_sum += axis_distance(obs.tau, targets[3]);
  deviation_sum += axis_distance(obs.r_eu, targets[4]);
  deviation_sum += axis_distance(obs.lipschitz, targets[5]);
  deviation_sum += axis_distance(obs.calibration, targets[6]);
  return 1.0 - deviation_sum / static_cast<double>(kProfileStatisticCount);
}

void IntelligenceProfiler::update_statistic(ProfileStatistic stat, double value) {
  nig_states_[static_cast<std::size_t>(stat)].update(value);
}

void IntelligenceProfiler::update(const ProfileObservation& observation) {
  update_statistic(ProfileStatistic::Alpha, observation.alpha);
  update_statistic(ProfileStatistic::DEff, observation.d_eff);
  update_statistic(ProfileStatistic::SigmaBranch, observation.sigma_branch);
  update_statistic(ProfileStatistic::Tau, observation.tau);
  update_statistic(ProfileStatistic::REu, observation.r_eu);
  update_statistic(ProfileStatistic::Lipschitz, observation.lipschitz);
  update_statistic(ProfileStatistic::Calibration, observation.calibration);
}

void IntelligenceProfiler::update_from_batch(const ProfileBatch& batch) {
  ProfileObservation obs;
  if (batch.input != nullptr && batch.output != nullptr && batch.n_samples > 0 && batch.n_dims > 0) {
    obs.alpha = compute_alpha_gria(batch.input, batch.output, batch.n_samples, batch.n_dims);
    obs.d_eff = compute_participation_ratio(batch.output, batch.n_samples, batch.n_dims);

    if (batch.perturbed_input != nullptr && batch.perturbed_output != nullptr) {
      obs.sigma_branch =
          compute_branching_ratio_sensitivity(batch.output, batch.perturbed_output, batch.perturbed_input,
                                            batch.n_samples, batch.n_dims);
      obs.lipschitz =
          compute_lipschitz_sensitivity(batch.output, batch.perturbed_output, batch.n_samples, batch.n_dims);
    }
  }

  if (batch.sequence != nullptr && batch.n_timesteps > 1 && batch.n_dims > 0) {
    obs.tau = compute_memory_depth_normalized(batch.sequence, batch.n_timesteps, batch.n_dims,
                                              batch.tau_max_lag, batch.tau_max_steps);
  }

  if (batch.confidences != nullptr && batch.correct != nullptr && batch.n_labels > 0) {
    obs.calibration = compute_calibration(batch.confidences, batch.correct, batch.n_labels);
  }
  if (batch.epistemic_var.has_value() && batch.aleatoric_var.has_value()) {
    obs.r_eu = compute_epistemic_ratio(*batch.epistemic_var, *batch.aleatoric_var);
  }

  if (batch.sigma_branch.has_value()) {
    obs.sigma_branch = batch.sigma_branch.value();
  }
  if (batch.tau.has_value()) {
    obs.tau = batch.tau.value();
  }
  if (batch.lipschitz.has_value()) {
    obs.lipschitz = batch.lipschitz.value();
  }

  update(obs);
}

std::array<int, kProfileStatisticCount> IntelligenceProfiler::get_statistic_update_counts() const {
  std::array<int, kProfileStatisticCount> counts{};
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    counts[i] = nig_states_[i].n_updates();
  }
  return counts;
}

std::array<std::array<double, IntelligenceProfiler::kMatrixCols>, kProfileStatisticCount>
IntelligenceProfiler::get_profile_matrix() const {
  std::array<std::array<double, kMatrixCols>, kProfileStatisticCount> matrix{};
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    const NigStatisticState& state = nig_states_[i];
    matrix[i][0] = state.mean();
    matrix[i][1] = state.epistemic_var();
    matrix[i][2] = state.aleatoric_var();
  }
  return matrix;
}

double IntelligenceProfiler::criticality_score() const {
  const auto targets = critical_targets();
  double deviation_sum = 0.0;
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    deviation_sum += std::abs(nig_states_[i].mean() - targets[i]);
  }
  return 1.0 - deviation_sum / static_cast<double>(kProfileStatisticCount);
}

double IntelligenceProfiler::health_signal() const {
  double mahal = 0.0;
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    const NigStatisticState& state = nig_states_[i];
    const double delta = state.mean() - state.running_mean();
    mahal += (delta * delta) / (state.observation_variance() + kHealthEps);
  }
  return mahal;
}

std::array<int, kProfileStatisticCount> IntelligenceProfiler::statistic_update_counts() const {
  std::array<int, kProfileStatisticCount> counts{};
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    counts[i] = nig_states_[i].n_updates();
  }
  return counts;
}

}  // namespace cypha::intelligence
