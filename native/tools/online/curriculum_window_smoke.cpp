/// Smoke test for windowed (hardest-first + within-window random) curriculum ordering.
/// docs/FUTURE.md §6: "hardest first, then randomise within a window."
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "cypha/curriculum.hpp"

int main() {
  // Descending confidence by index: row i has confidence (0.9 - 0.1*i), so the hardest-first
  // (ascending confidence) base order is exactly the reverse index sequence.
  const std::vector<double> confidences = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2};
  const int n = 8;

  // window <= 1 must reproduce curriculum_order_ascending_confidence exactly (no shuffling, no
  // rng draws) -- this is the "opt-in / default-off, byte-identical" contract.
  {
    std::mt19937 rng(123);
    const std::vector<int> base = cypha::curriculum_order_ascending_confidence(confidences, n);
    const std::vector<int> windowed0 = cypha::curriculum_order_windowed(confidences, n, 0, rng);
    const std::vector<int> windowed1 = cypha::curriculum_order_windowed(confidences, n, 1, rng);
    assert(base == windowed0);
    assert(base == windowed1);
    assert(base[0] == 7 && base[7] == 0);  // hardest (lowest confidence) first
  }

  // window == 4: the first 4 positions must be some permutation of the 4 hardest indices
  // {7,6,5,4}, and the last 4 positions a permutation of the 4 easiest {3,2,1,0}. The windowed
  // order must still differ from the strict (unshuffled) hardest-first order for at least one seed
  // (otherwise the "randomise within window" behaviour would be silently absent), and must be
  // fully deterministic given a fixed seed.
  {
    std::mt19937 rng_a(42);
    const std::vector<int> order_a = cypha::curriculum_order_windowed(confidences, n, 4, rng_a);
    std::mt19937 rng_b(42);
    const std::vector<int> order_b = cypha::curriculum_order_windowed(confidences, n, 4, rng_b);
    assert(order_a == order_b);  // deterministic given the same seed

    const std::set<int> first_window(order_a.begin(), order_a.begin() + 4);
    const std::set<int> second_window(order_a.begin() + 4, order_a.end());
    assert(first_window == std::set<int>({7, 6, 5, 4}));
    assert(second_window == std::set<int>({3, 2, 1, 0}));

    const std::vector<int> strict = cypha::curriculum_order_ascending_confidence(confidences, n);
    bool any_seed_reorders = (order_a != strict);
    if (!any_seed_reorders) {
      // Extremely unlikely (1-in-576 chance across two independent 4-element shuffles), but try a
      // few more seeds before concluding the windowing has no effect.
      for (std::uint32_t seed = 1; seed < 32 && !any_seed_reorders; ++seed) {
        std::mt19937 rng_c(seed);
        const std::vector<int> order_c = cypha::curriculum_order_windowed(confidences, n, 4, rng_c);
        any_seed_reorders = (order_c != strict);
      }
    }
    assert(any_seed_reorders);
  }

  // window >= n_rows shuffles the whole sequence at once; the multiset of indices is preserved.
  {
    std::mt19937 rng(7);
    const std::vector<int> order = cypha::curriculum_order_windowed(confidences, n, 100, rng);
    std::vector<int> sorted_order = order;
    std::sort(sorted_order.begin(), sorted_order.end());
    for (int i = 0; i < n; ++i) {
      assert(sorted_order[static_cast<std::size_t>(i)] == i);
    }
  }

  std::puts("curriculum_window_smoke: PASS");
  return 0;
}
