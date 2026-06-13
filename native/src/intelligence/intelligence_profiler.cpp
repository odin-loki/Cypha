#include "cypha/intelligence/intelligence_profiler.hpp"

#include "cypha/intelligence/measurers.hpp"

#include <cmath>

namespace cypha::intelligence {

namespace {

constexpr double kHealthEps = 1e-8;

}  // namespace

IntelligenceProfiler::IntelligenceProfiler() = default;

std::array<double, kProfileStatisticCount> IntelligenceProfiler::critical_targets() {
  return {0.50, 0.45, 0.50, 0.65, 0.70, 0.50, 0.82};
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
  }
  if (batch.confidences != nullptr && batch.correct != nullptr && batch.n_labels > 0) {
    obs.calibration = compute_calibration(batch.confidences, batch.correct, batch.n_labels);
  }
  if (batch.epistemic_var.has_value() && batch.aleatoric_var.has_value()) {
    obs.r_eu = compute_epistemic_ratio(*batch.epistemic_var, *batch.aleatoric_var);
  }
  obs.sigma_branch = batch.sigma_branch.value_or(0.5);
  obs.tau = batch.tau.value_or(0.5);
  obs.lipschitz = batch.lipschitz.value_or(0.5);
  update(obs);
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

}  // namespace cypha::intelligence
