#include "cypha/cyphalm/hebbian_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cypha::cyphalm {

void HebbianSSMState::resize(int d_state_in, int n_layers_in) {
  d_state = d_state_in;
  n_layers = n_layers_in;
  w.assign(static_cast<std::size_t>(n_layers_in),
           std::vector<double>(static_cast<std::size_t>(d_state_in * d_state_in), 0.0));
}

void sparse_hebbian_update(HebbianSSMState& state, const double* fast_state, const double* slow_state, double lr,
                           int layer) {
  const int d = state.d_state;
  if (d <= 0 || layer < 0 || layer >= state.n_layers || fast_state == nullptr || slow_state == nullptr) {
    return;
  }
  auto& w_layer = state.w[static_cast<std::size_t>(layer)];
  if (static_cast<int>(w_layer.size()) != d * d) {
    return;
  }

  const int k = std::max(1, d / 8);
  std::vector<int> idx(static_cast<std::size_t>(d));
  for (int i = 0; i < d; ++i) {
    idx[static_cast<std::size_t>(i)] = i;
  }
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
    return std::abs(fast_state[static_cast<std::size_t>(a)]) >
           std::abs(fast_state[static_cast<std::size_t>(b)]);
  });

  for (int i = 0; i < d; ++i) {
    const double post_i = slow_state[static_cast<std::size_t>(i)];
    for (int kk = 0; kk < k; ++kk) {
      const int j = idx[static_cast<std::size_t>(kk)];
      const double pre_j = fast_state[static_cast<std::size_t>(j)];
      w_layer[static_cast<std::size_t>(i * d + j)] += lr * post_i * pre_j;
    }
  }
}

}  // namespace cypha::cyphalm
