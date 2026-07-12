#pragma once

#include <random>
#include <vector>

namespace cypha {

/// Max softmax probability for one row (confidence proxy).
double row_max_softmax_confidence(const double* probs, int k);

/// Reorder ``0..n_rows-1`` by ascending ``max_confidences`` (hardest-first curriculum).
std::vector<int> curriculum_order_ascending_confidence(const std::vector<double>& max_confidences, int n_rows);

/// Same hardest-first order as ``curriculum_order_ascending_confidence``, then locally randomised:
/// the resulting index sequence is cut into contiguous chunks of ``window`` positions (the last
/// chunk may be shorter) and each chunk is independently ``std::shuffle``d with ``rng``. This keeps
/// examples roughly ordered by difficulty (hardest-first) while breaking the strict total order the
/// model could otherwise overfit to. ``window <= 1`` performs no shuffling and returns exactly
/// ``curriculum_order_ascending_confidence``'s output (deterministic, no ``rng`` draws).
std::vector<int> curriculum_order_windowed(const std::vector<double>& max_confidences, int n_rows, int window,
                                            std::mt19937& rng);

}  // namespace cypha
