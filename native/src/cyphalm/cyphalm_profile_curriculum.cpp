#include "cypha/cyphalm/cyphalm_profile_curriculum.hpp"

#include <algorithm>

#include "cypha/curriculum.hpp"

namespace cypha::cyphalm {

std::vector<int> profile_curriculum_order(const std::vector<int>& token_ids, int n_steps,
                                          const std::function<double(int step_idx)>& difficulty_fn) {
  if (token_ids.size() < 2 || n_steps <= 0 || !difficulty_fn) {
    return {};
  }
  const int steps = std::min(n_steps, static_cast<int>(token_ids.size()) - 1);
  std::vector<double> confidences;
  confidences.reserve(static_cast<std::size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    const double difficulty = std::clamp(difficulty_fn(i), 0.0, 1.0);
    confidences.push_back(1.0 - difficulty);
  }
  return cypha::curriculum_order_ascending_confidence(confidences, steps);
}

}  // namespace cypha::cyphalm
