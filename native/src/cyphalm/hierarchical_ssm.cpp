#include "cypha/cyphalm/hierarchical_ssm.hpp"

#include <numeric>
#include <stdexcept>

namespace cypha::cyphalm {

HierarchicalSSM::HierarchicalSSM(CellAISSMConfig fast_cfg, int compress_every)
    : compress_every_(compress_every), fast_(fast_cfg), slow_(fast_cfg) {
  if (compress_every_ < 1) {
    throw std::invalid_argument("HierarchicalSSM: compress_every must be >= 1");
  }
  pool_accum_.assign(static_cast<std::size_t>(fast_cfg.d_state), 0.0);
  reset();
}

void HierarchicalSSM::reset() {
  fast_.reset();
  slow_.reset();
  token_count_ = 0;
  pool_count_ = 0;
  std::fill(pool_accum_.begin(), pool_accum_.end(), 0.0);
}

std::vector<double> HierarchicalSSM::step(const std::vector<double>& e_t) {
  auto ctx = fast_.step(e_t);

  const auto mean_h = fast_.mean_fast_state();
  for (int i = 0; i < static_cast<int>(pool_accum_.size()); ++i) {
    pool_accum_[static_cast<std::size_t>(i)] += mean_h[static_cast<std::size_t>(i)];
  }
  ++pool_count_;
  ++token_count_;

  if (pool_count_ >= compress_every_) {
    std::vector<double> pooled(pool_accum_.size());
    const double inv = 1.0 / static_cast<double>(pool_count_);
    for (std::size_t i = 0; i < pooled.size(); ++i) {
      pooled[i] = pool_accum_[i] * inv;
    }
    slow_.step(pooled);
    pool_count_ = 0;
    std::fill(pool_accum_.begin(), pool_accum_.end(), 0.0);
  }

  return ctx;
}

}  // namespace cypha::cyphalm
