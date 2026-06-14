/// Smoke test for hardest-first curriculum row ordering.
#include <cassert>
#include <cstdio>
#include <vector>

#include "cypha/curriculum.hpp"

int main() {
  const std::vector<double> confidences = {0.9, 0.2, 0.5, 0.1};
  const std::vector<int> order = cypha::curriculum_order_ascending_confidence(confidences, 4);
  assert(order.size() == 4);
  assert(order[0] == 3);
  assert(order[1] == 1);
  assert(order[2] == 2);
  assert(order[3] == 0);

  const double max_conf = cypha::row_max_softmax_confidence(confidences.data(), 4);
  assert(max_conf == 0.9);

  std::puts("curriculum_smoke: PASS");
  return 0;
}
