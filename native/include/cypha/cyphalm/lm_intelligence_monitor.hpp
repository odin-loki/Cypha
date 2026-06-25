#pragma once

#include <cstdint>
#include <vector>

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

  void reset();

 private:
  cypha::intelligence::ProfileObservation compute_observation() const;
  void trim_field_history();
  void append_perturbation_pair(const double* base, const double* perturbed);

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

  std::vector<double> prev_field_;
  bool use_eigenvalue_d_eff_ = false;
};

}  // namespace cypha::cyphalm
