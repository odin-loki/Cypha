#pragma once

#include <array>
#include <optional>

#include "cypha/intelligence/causal_graph.hpp"
#include "cypha/intelligence/nig_statistic_state.hpp"
#include "cypha/intelligence/profile_observation.hpp"
#include "cypha/intelligence/profile_statistic.hpp"

namespace cypha::intelligence {

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

  /// Raw NIG update counts per statistic (diagnostics / completeness checks).
  std::array<int, kProfileStatisticCount> get_statistic_update_counts() const;

  /// ``κ = 1 - (1/7) Σ |P_i - P*_i|`` vs near-critical targets.
  double criticality_score() const;

  /// Diagonal Mahalanobis distance of current means from observation baselines.
  double health_signal() const;

  /// Per-statistic NIG update counts (for profile completeness validation).
  std::array<int, kProfileStatisticCount> statistic_update_counts() const;

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

  /// Default weight for ``apply_causal_fidelity``: bounds the maximum causal-fidelity-driven
  /// kappa boost to 5% when fidelity is at its ceiling. Chosen to be small relative to kappa's
  /// own [0, 1] range and to Paper V's ~0.80 (current, per
  /// docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md §5.4) -> 0.90-0.93 (target) gap, so the
  /// causal-graph signal can nudge kappa but cannot on its own move it anywhere near that gap,
  /// let alone destabilise the existing 7-axis calibration.
  static constexpr double kDefaultCausalFidelityWeight = 0.05;

  /// Fold a causal-graph fidelity signal (``CausalGraphMonitor::causal_fidelity()``, in
  /// ``[0, 1)``) into a criticality score as a bounded multiplicative adjustment:
  /// ``kappa * (1 + weight * fidelity)``, clamped back to ``[0, 1]`` afterward so the result
  /// stays in kappa's normal range. Multiplicative (not additive) so the size of the boost
  /// scales with kappa's own value -- a well-grounded causal graph makes an already-good
  /// profile look modestly better, rather than adding the same flat bonus regardless of how
  /// far the profile actually is from critical (see
  /// docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md §9 for the full design rationale).
  ///
  /// Exactly a no-op (returns ``kappa`` unchanged, bit-identical) whenever ``causal_fidelity``
  /// is ``<= 0`` -- i.e. whenever the causal graph has insufficient data (both estimated edges
  /// have ``n < 2``) -- so kappa's existing behaviour is completely unaffected until the graph
  /// has genuinely accumulated enough history to say something about how the profile axes move
  /// together.
  static double apply_causal_fidelity(double kappa, double causal_fidelity,
                                      double weight = kDefaultCausalFidelityWeight);

  /// Bench-run-scoped ``CausalGraphMonitor`` that lives for as long as this
  /// ``IntelligenceProfiler`` does (i.e. one instance per bench run / training run, not one
  /// per report call). See docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md §9.7: prior to this,
  /// ``intelligence_profile_report_json`` constructed a *fresh* ``CausalGraphMonitor`` on every
  /// call, so its online-correlation edges never saw more than one observation
  /// (``causal_fidelity()`` was therefore always exactly ``0.0``). Callers that want the causal
  /// graph to accumulate real cross-step history (e.g. ``LmIntelligenceMonitor::flush_to_profiler``,
  /// which has access to this profiler's own per-token history) should feed observations into
  /// this monitor directly rather than constructing their own. Declared ``const`` (backed by a
  /// ``mutable`` member) so it is reachable both from mutating call sites (e.g.
  /// ``flush_to_profiler``, which takes a non-const ``IntelligenceProfiler&``) and from the
  /// read-only report path (``intelligence_profile_report_json`` takes a
  /// ``const IntelligenceProfiler&`` since most callers only ever read a profiler when
  /// generating a report) -- the graph's own accumulation state is a live diagnostic, not part
  /// of this profiler's logical (NIG-state) value, so ``const``-mutability here mirrors how the
  /// rest of this class already treats diagnostics vs. core state.
  CausalGraphMonitor& causal_graph() const { return causal_graph_; }

 private:
  void update_statistic(ProfileStatistic stat, double value);

  static double axis_distance(double value, double target);

  std::array<NigStatisticState, kProfileStatisticCount> nig_states_;
  mutable CausalGraphMonitor causal_graph_;
};

}  // namespace cypha::intelligence
