#pragma once

#include "cypha/cyphalm/cellai_ssm.hpp"

namespace cypha::cyphalm {

/// View-block memory carry policy (Tier 1 / multi-view training).
enum class MemoryPolicy {
  RESET_ALL,
  CARRY_SLOW,
  RESET_FAST_ONLY,
};

/// Apply memory policy at a view block boundary.
void apply_memory_policy(MemoryPolicy policy, CellAISSM& ssm);

}  // namespace cypha::cyphalm
