#pragma once

#include "cypha/cyphalm/cellai_ssm.hpp"

#include <vector>

namespace cypha::cyphalm {

/// Fast CellAI tier + slow tier stepped every compress_every tokens (Tier 1 A5).
class HierarchicalSSM {
 public:
  HierarchicalSSM(CellAISSMConfig fast_cfg, int compress_every = 64);

  void reset();
  std::vector<double> step(const std::vector<double>& e_t);

  int compress_every() const { return compress_every_; }
  int token_count() const { return token_count_; }
  const CellAISSM& fast_tier() const { return fast_; }
  const CellAISSM& slow_tier() const { return slow_; }

 private:
  int compress_every_;
  int token_count_{0};
  std::vector<double> pool_accum_;
  int pool_count_{0};

  CellAISSM fast_;
  CellAISSM slow_;
};

}  // namespace cypha::cyphalm
