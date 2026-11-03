#include "cypha/cyphalm/memory_policy.hpp"

namespace cypha::cyphalm {

void apply_memory_policy(MemoryPolicy policy, CellAISSM& ssm) {
  switch (policy) {
    case MemoryPolicy::RESET_ALL:
      ssm.reset();
      break;
    case MemoryPolicy::CARRY_SLOW:
      ssm.reset_fast_only();
      break;
    case MemoryPolicy::RESET_FAST_ONLY:
      ssm.reset_fast_only();
      break;
  }
}

}  // namespace cypha::cyphalm
