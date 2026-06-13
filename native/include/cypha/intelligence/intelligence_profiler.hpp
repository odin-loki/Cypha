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
  const double* perturbed_input = nullptr;
  const double* perturbed_output = nullptr;
  const double* sequence = nullptr;
  int n_timesteps = 0;
  int tau_max_lag = 32;
  int tau_max_steps = 512;
  const double* confidences = nullptr;
  const int* correct = nullptr;
  int n_labels = 0;
  std::optional<double> epistemic_var;
  std::optional<double> aleatoric_var;
  std::optional<double> sigma_branch;
  std::optional<double> tau;
  std::optional<double> lipschitz;
};

/// Paper III landscape reference classes (estimated profiles from Paper III §3).
enum class LandscapeSystemClass {
  SimpleFfn,
  LargeTransformer,
  CyphaAugmented,
  SelfCorrectingCypha,
  SoftWorldCypha,
  HumanMedian,
};

/// Paper II failure-mode flags from P-space signature (Paper II §4.1).
struct FailureModeFlags {
  bool low_calibration = false;
  bool high_lipschitz = false;
  bool low_tau = false;
  bool explosive_branching = false;
  bool damped_branching = false;
  bool low_d_eff = false;
  bool low_r_eu = false;
  bool extreme_alpha = false;
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

  /// Paper III estimated reference profile for a system class.
  static ProfileObservation landscape_reference(LandscapeSystemClass system);

  /// Paper II: unweighted Euclidean distance in P-space.
  static double profile_distance(const ProfileObservation& a, const ProfileObservation& b);

  /// Paper III: distance normalised by ``sqrt(7)``.
  static double profile_distance_normalized(const ProfileObservation& a, const ProfileObservation& b);

  /// Paper II: ``||P - P*||²`` navigation loss toward critical targets.
  static double navigation_loss(const ProfileObservation& obs);

  /// Paper II: predict failure modes from marginal thresholds.
  static FailureModeFlags predict_failure_modes(const ProfileObservation& obs);

  /// Paper III: ``A ≻ B`` if A is closer to critical on every axis.
  static bool dominates(const ProfileObservation& a, const ProfileObservation& b);

  /// Criticality score for an arbitrary observation vs targets.
  static double criticality_score_for(const ProfileObservation& obs);

 private:
  void update_statistic(ProfileStatistic stat, double value);

  static double axis_distance(double value, double target);

  std::array<NigStatisticState, kProfileStatisticCount> nig_states_;
};

}  // namespace cypha::intelligence
