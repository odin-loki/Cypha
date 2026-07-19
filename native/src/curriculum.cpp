#include "cypha/curriculum.hpp"

#include <algorithm>

#include "cypha/portable_shuffle.hpp"

namespace cypha {

double row_max_softmax_confidence(const double* probs, int k) {
  if (k <= 0 || probs == nullptr) {
    return 0.0;
  }
  double best = probs[0];
  for (int j = 1; j < k; ++j) {
    if (probs[static_cast<std::size_t>(j)] > best) {
      best = probs[static_cast<std::size_t>(j)];
    }
  }
  return best;
}

std::vector<int> curriculum_order_ascending_confidence(const std::vector<double>& max_confidences, int n_rows) {
  struct RankRow {
    int index;
    double confidence;
  };
  std::vector<RankRow> ranked;
  ranked.reserve(static_cast<std::size_t>(n_rows));
  for (int i = 0; i < n_rows; ++i) {
    const double conf =
        i < static_cast<int>(max_confidences.size()) ? max_confidences[static_cast<std::size_t>(i)] : 0.0;
    ranked.push_back({i, conf});
  }
  std::sort(ranked.begin(), ranked.end(), [](const RankRow& a, const RankRow& b) {
    if (a.confidence != b.confidence) {
      return a.confidence < b.confidence;
    }
    return a.index < b.index;
  });
  std::vector<int> order(static_cast<std::size_t>(n_rows));
  for (int i = 0; i < n_rows; ++i) {
    order[static_cast<std::size_t>(i)] = ranked[static_cast<std::size_t>(i)].index;
  }
  return order;
}

std::vector<int> curriculum_order_windowed(const std::vector<double>& max_confidences, int n_rows, int window,
                                            std::mt19937& rng) {
  std::vector<int> order = curriculum_order_ascending_confidence(max_confidences, n_rows);
  if (window <= 1 || n_rows <= 1) {
    return order;
  }
  for (std::size_t start = 0; start < order.size(); start += static_cast<std::size_t>(window)) {
    const std::size_t end = std::min(order.size(), start + static_cast<std::size_t>(window));
    portable_shuffle(order.begin() + static_cast<std::ptrdiff_t>(start),
                     order.begin() + static_cast<std::ptrdiff_t>(end), rng);
  }
  return order;
}

}  // namespace cypha
