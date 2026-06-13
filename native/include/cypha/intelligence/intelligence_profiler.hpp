#pragma once

#include <array>
#include <optional>

#include "cypha/intelligence/nig_statistic_state.hpp"
#include "cypha/intelligence/profile_statistic.hpp"

namespace cypha::intelligence {

/// One row of the seven-statistic profile vector ``P = (α, D_eff, σ, τ, r_eu, L, C)``.
struct ProfileObservation {
  double alpha = 0.5;
  double d_eff = 0.5;
  double sigma_branch = 0.5;
  double tau = 0.5;
  double r_eu = 0.5;
  double lipschitz = 0.5;
  double calibration = 0.5;
};

/// Optional raw batch for measurers; unset dynamics default to ``0.5``.
struct ProfileBatch {
  const double* input = nullptr;
  const double* output = nullptr;
  int n_samples = 0;
  int n_dims = 0;
  const double* confidences = nullptr;
  const int* correct = nullptr;
  int n_labels = 0;
  std::optional<double> epistemic_var;
  std::optional<double> aleatoric_var;
  std::optional<double> sigma_branch;
  std::optional<double> tau;
  std::optional<double> lipschitz;
};

/// Online 7×3 intelligence profile matrix (point, epistemic, aleatoric per statistic).
class IntelligenceProfiler {
 public:
  static constexpr std::size_t kMatrixCols = 3;

  IntelligenceProfiler();

  void update(const ProfileObservation& observation);
  void update_from_batch(const ProfileBatch& batch);

  /// Row order matches ``ProfileStatistic``; columns: point, epistemic var, aleatoric var.
  std::array<std::array<double, kMatrixCols>, kProfileStatisticCount> get_profile_matrix() const;

  /// ``κ = 1 - (1/7) Σ |P_i - P*_i|`` vs near-critical targets.
  double criticality_score() const;

  /// Diagonal Mahalanobis distance of current means from observation baselines.
  double health_signal() const;

  static std::array<double, kProfileStatisticCount> critical_targets();

 private:
  void update_statistic(ProfileStatistic stat, double value);

  std::array<NigStatisticState, kProfileStatisticCount> nig_states_;
};

}  // namespace cypha::intelligence
