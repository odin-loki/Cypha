#pragma once

#include <cstdint>
#include <vector>

#include "cypha/intelligence/causal_graph.hpp"
#include "cypha/intelligence/criticality_vector.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"

namespace cypha::cyphalm {

/// Rolling token-stream monitor for the full seven-stat intelligence profile.
class LmIntelligenceMonitor {
 public:
  static constexpr int kMaxFieldHistory = 64;
  static constexpr double kPerturbationEps = 0.01;
  static constexpr int kTauMaxLag = 32;

  void observe_token(const std::vector<double>& input_embed,
                     const std::vector<double>& field_hidden,
                     const std::vector<double>& log_probs, double epistemic_var,
                     double aleatoric_var, std::int64_t next_token_id, int vocab_size);

  void set_use_eigenvalue_d_eff(bool enabled) { use_eigenvalue_d_eff_ = enabled; }

  void flush_to_profiler(cypha::intelligence::IntelligenceProfiler& profiler);

  cypha::intelligence::ProfileObservation snapshot_observation() const;

  /// Hot criticality gauges from accumulated token history (read-only).
  cypha::intelligence::CriticalityHotInput hot_criticality_input() const;

  void reset();

 private:
  cypha::intelligence::ProfileObservation compute_observation() const;
  void trim_field_history();
  void append_perturbation_pair(const double* base, const double* perturbed);

  /// Feed the profiler's persistent causal graph with several genuinely time-varying
  /// (alpha, calibration)/(tau, r_eu) checkpoints reconstructed from this monitor's own
  /// per-token history (growing-prefix snapshots), rather than only the one or two aggregate
  /// observations `flush_to_profiler` already feeds the profiler's NIG state. See
  /// docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md §9.7 for why this is necessary: those two
  /// aggregate observations are both derived from the same summed per-token statistics, so at
  /// least one axis of each edge is identical across them (zero variance -> zero correlation)
  /// even though nominally n=2. No-op (feeds nothing) when this monitor's per-token history is
  /// too short to reconstruct even one meaningful checkpoint.
  void feed_causal_checkpoints(cypha::intelligence::CausalGraphMonitor& causal) const;

  int field_dim_ = 0;
  int embed_dim_ = 0;
  int field_count_ = 0;

  std::vector<double> field_history_;
  std::vector<double> embed_history_;
  std::vector<double> sequence_trace_;
  std::vector<double> confidences_;
  std::vector<int> correct_;
  std::vector<double> step_alphas_;

  std::vector<double> base_fields_;
  std::vector<double> perturbed_fields_;
  std::vector<double> perturbations_;
  int pair_count_ = 0;

  double epistemic_sum_ = 0.0;
  double aleatoric_sum_ = 0.0;
  int variance_steps_ = 0;
  std::vector<double> epistemic_history_;
  std::vector<double> aleatoric_history_;

  std::vector<double> prev_field_;
  bool use_eigenvalue_d_eff_ = false;
};

}  // namespace cypha::cyphalm
