#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/soft_world_monitor.hpp"

namespace cypha::intelligence {

/// Paper V causal graph: fixed topology of P-space edges tracking soft-world maturation
/// signals. Edge *structure* (which nodes are connected) is still hand-specified, not
/// discovered (Paper V §3.2's PC-algorithm-style skeleton discovery is out of scope here;
/// see docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md). The alpha->calibration and tau->r_eu
/// edge *weights* are estimated online via ``OnlineCorrelation`` from real accumulated
/// profiler observations rather than a fixed per-observation formula; the remaining edges
/// (query->r_eu, simulation->world_model, world_model->maturation, maturation->tau) report
/// directly measured deltas/NIG means from ``SoftWorldMonitor``, which is already real online
/// inference, not a placeholder. ``causal_fidelity()`` aggregates the two estimated edges into
/// a single confidence-weighted signal that is fed back into kappa (see
/// ``IntelligenceProfiler::apply_causal_fidelity``, docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md
/// §9) -- this is the graph's first and only feedback path into kappa/criticality_score.
struct CausalEdge {
  std::string from;
  std::string to;
  double weight = 0.0;
};

/// Online (Welford-style) Pearson correlation estimator for two paired scalar streams.
/// Used so causal-edge weights reflect an actual statistical relationship between the
/// named variables across observed history, instead of a fixed formula evaluated on the
/// current observation alone.
class OnlineCorrelation {
 public:
  void update(double x, double y);

  int n() const { return n_; }

  /// Pearson correlation coefficient in [-1, 1]; ``0.0`` until at least 2 observations
  /// have been accumulated (i.e. there is genuinely no correlation to estimate yet).
  double correlation() const;

 private:
  int n_ = 0;
  double mean_x_ = 0.0;
  double mean_y_ = 0.0;
  double cov_ = 0.0;
  double var_x_ = 0.0;
  double var_y_ = 0.0;
};

/// One acquisition → simulation → maturation cycle (Paper V trajectory entry).
struct SimulationStepEvent {
  double r_eu_before = 0.0;
  double r_eu_after = 0.0;
  double resolution = 0.0;
  double maturation_level = 0.0;
};

class CausalGraphMonitor {
 public:
  CausalGraphMonitor();

  /// Record a directed causal influence (weight in [0, 1]).
  void record_edge(std::string from, std::string to, double weight);

  /// Ingest profiler observation and update default Paper V edges + soft-world monitor.
  void observe_profile(const ProfileObservation& obs);

  /// Record acquisition / simulation events into the embedded soft-world monitor.
  void record_acquisition(double r_eu_before, double r_eu_after);
  void record_simulation(double resolution);

  /// Full Paper V cycle: acquisition, simulation resolution, maturation snapshot.
  void simulation_step(double r_eu_before, double r_eu_after, double resolution);

  /// Multi-step Paper V loop: profile ingest + ``n_steps`` acquisition/simulation cycles.
  void run_simulation_trajectory(int n_steps, const ProfileObservation& obs, double resolution_scale = 0.3);

  const SoftWorldMonitor& soft_world() const { return soft_world_; }
  const std::vector<CausalEdge>& edges() const { return edges_; }
  const std::vector<SimulationStepEvent>& trajectory() const { return trajectory_; }

  /// Current online-estimated alpha->calibration edge correlation (raw, signed; see
  /// ``OnlineCorrelation``). Exposed for testing/diagnostics of the real vs. fixed-formula gap.
  double alpha_calibration_correlation() const { return alpha_calibration_corr_.correlation(); }
  int alpha_calibration_n() const { return alpha_calibration_corr_.n(); }

  /// Current online-estimated tau->r_eu edge correlation (raw, signed).
  double tau_r_eu_correlation() const { return tau_r_eu_corr_.correlation(); }
  int tau_r_eu_n() const { return tau_r_eu_corr_.n(); }

  /// Aggregate causal-graph fidelity in ``[0, 1)``: a confidence-weighted mean of
  /// ``|correlation|`` across the online-correlation-estimated edges (alpha<->calibration,
  /// tau<->r_eu). "Confidence-weighted" means each edge's contribution is scaled by how much
  /// history backs it (``1 - 1/n``, i.e. 0 below ``n=2`` and rising toward 1 with more
  /// observations), so a single lucky-looking correlation from a small sample can't report
  /// high fidelity the way a long, consistently-correlated history can.
  ///
  /// Exactly ``0.0`` whenever *neither* edge has accumulated ``n >= 2`` observations yet --
  /// the same "not enough data" convention ``OnlineCorrelation::correlation()`` itself uses --
  /// so a degenerate/insufficient-data graph reports a well-defined neutral value instead of a
  /// fabricated one. See ``IntelligenceProfiler::apply_causal_fidelity`` for how this plugs
  /// into kappa.
  double causal_fidelity() const;

  nlohmann::json to_json() const;
  nlohmann::json trajectory_json() const;

 private:
  SoftWorldMonitor soft_world_;
  std::vector<CausalEdge> edges_;
  std::vector<SimulationStepEvent> trajectory_;
  ProfileObservation last_obs_{};
  bool has_last_obs_{false};
  OnlineCorrelation alpha_calibration_corr_;
  OnlineCorrelation tau_r_eu_corr_;
};

}  // namespace cypha::intelligence
