#pragma once

#include <cstddef>
#include <vector>

namespace cypha::cyphalm {

/// Per-layer sparse Hebbian lateral weights (Cypha Tests 2C).
struct HebbianSSMState {
  int d_state{0};
  int n_layers{0};
  std::vector<std::vector<double>> w;

  void resize(int d_state_in, int n_layers_in);
};

/// Outer-product sparse update on top-|pre|/8 activations (matches `CellAISSM.sparse_hebbian_update`).
void sparse_hebbian_update(HebbianSSMState& state, const double* fast_state, const double* slow_state, double lr,
                           int layer);

}  // namespace cypha::cyphalm
