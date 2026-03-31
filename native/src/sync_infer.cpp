#include "cypha/sync_infer.hpp"

#include <algorithm>

#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"

namespace cypha {

void sync_infer_model_from_memory(CyphaInferModel& m, const CyphaDifMemoryState& s) {
  constexpr double kMinVar = 1e-4;
  m.labels = s.labels;
  m.D = s.D;
  m.n_obs = s.n_obs_buf;
  m.d_latent = s.d_latent;
  m.mu_world = s.world_mu;
  m.inv_v.resize(static_cast<std::size_t>(s.d_latent));
  double sum_v = 0.0;
  for (int j = 0; j < s.d_latent; ++j) {
    double vj = s.world_v[static_cast<std::size_t>(j)];
    sum_v += vj;
    m.inv_v[static_cast<std::size_t>(j)] = 1.0 / std::max(vj, kMinVar);
  }
  m.v_mean = sum_v / static_cast<double>(s.d_latent);
}

}  // namespace cypha
