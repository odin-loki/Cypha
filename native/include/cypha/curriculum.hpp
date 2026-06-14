#pragma once

#include <vector>

namespace cypha {

/// Max softmax probability for one row (confidence proxy).
double row_max_softmax_confidence(const double* probs, int k);

/// Reorder ``0..n_rows-1`` by ascending ``max_confidences`` (hardest-first curriculum).
std::vector<int> curriculum_order_ascending_confidence(const std::vector<double>& max_confidences, int n_rows);

}  // namespace cypha
