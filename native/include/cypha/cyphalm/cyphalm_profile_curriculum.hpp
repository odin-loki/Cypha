#pragma once

#include <functional>
#include <vector>

namespace cypha::cyphalm {

/// Hardest-first step order from per-index difficulty (``1 - confidence`` proxy).
std::vector<int> profile_curriculum_order(const std::vector<int>& token_ids, int n_steps,
                                          const std::function<double(int step_idx)>& difficulty_fn);

}  // namespace cypha::cyphalm
