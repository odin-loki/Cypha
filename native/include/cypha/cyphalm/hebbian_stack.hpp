#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cypha/cyphalm/hebbian_encoder.hpp"
#include "cypha/cyphalm/hebbian_graph.hpp"
#include "cypha/cyphalm/hebbian_ssm.hpp"

namespace cypha::cyphalm {

/// Composition layer wiring Tier-4 hooks (native stand-in for `CellAISSM` Hebbian paths).
struct HebbianStackConfig {
  bool use_sparse_hebbian{true};
  bool use_hebb_graph{false};
  double ssm_hebb_lr{1e-4};
  int d_state{0};
  int n_layers{0};
  HebbianGraphConfig graph{};
};

struct HebbianStack {
  HebbianStackConfig cfg;
  HebbianEncoder encoder;
  HebbianSSMState ssm;
  std::unique_ptr<HebbianGraph> graph;

  void configure(const HebbianStackConfig& config_in);
  void on_ssm_layer_context(std::vector<double>& ctx, int layer, const double* fast_state,
                            const double* slow_state);
  void encoder_train_step(const double* f, const double* h, const std::string& true_label,
                          const std::string& pred_label, double lr, double weight = 1.0);
};

}  // namespace cypha::cyphalm
