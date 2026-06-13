#include "cypha/intelligence/measurers.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cypha::intelligence {

namespace {

constexpr double kEps = 1e-12;

double histogram_entropy(const double* values, int n, int n_bins) {
  if (n <= 0) {
    return 0.0;
  }
  const int bins = std::max(1, n_bins);
  double lo = values[0];
  double hi = values[0];
  for (int i = 1; i < n; ++i) {
    lo = std::min(lo, values[i]);
    hi = std::max(hi, values[i]);
  }
  if (hi <= lo) {
    hi = lo + 1.0;
  }
  std::vector<double> hist(static_cast<std::size_t>(bins), 0.0);
  const double width = (hi - lo) / static_cast<double>(bins);
  for (int i = 0; i < n; ++i) {
    int idx = static_cast<int>((values[i] - lo) / width);
    if (idx >= bins) {
      idx = bins - 1;
    }
    if (idx < 0) {
      idx = 0;
    }
    hist[static_cast<std::size_t>(idx)] += 1.0;
  }
  double ent = 0.0;
  const double norm = static_cast<double>(n) + kEps * static_cast<double>(bins);
  for (double c : hist) {
    const double p = (c + kEps) / norm;
    ent -= p * std::log(p);
  }
  return ent;
}

double column_variance(const double* activations, int n_samples, int n_dims, int col) {
  if (n_samples <= 0) {
    return 0.0;
  }
  double mean = 0.0;
  for (int r = 0; r < n_samples; ++r) {
    mean += activations[static_cast<std::size_t>(r * n_dims + col)];
  }
  mean /= static_cast<double>(n_samples);
  double var = 0.0;
  for (int r = 0; r < n_samples; ++r) {
    const double d = activations[static_cast<std::size_t>(r * n_dims + col)] - mean;
    var += d * d;
  }
  return var / static_cast<double>(n_samples);
}

}  // namespace

double compute_participation_ratio(const double* activations, int n_samples, int n_dims) {
  if (activations == nullptr || n_samples <= 0 || n_dims <= 0) {
    return 0.0;
  }
  double sum = 0.0;
  double sum_sq = 0.0;
  for (int d = 0; d < n_dims; ++d) {
    const double v = column_variance(activations, n_samples, n_dims, d);
    sum += v;
    sum_sq += v * v;
  }
  if (sum_sq <= kEps) {
    return 0.0;
  }
  const double participation = (sum * sum) / sum_sq;
  return std::clamp(participation / static_cast<double>(n_dims), 0.0, 1.0);
}

double compute_calibration(const double* confidences, const int* correct, int n, int n_bins) {
  if (confidences == nullptr || correct == nullptr || n <= 0) {
    return 0.0;
  }
  const int bins = std::max(1, n_bins);
  std::vector<double> bin_conf(static_cast<std::size_t>(bins), 0.0);
  std::vector<double> bin_acc(static_cast<std::size_t>(bins), 0.0);
  std::vector<int> bin_count(static_cast<std::size_t>(bins), 0);

  for (int i = 0; i < n; ++i) {
    const double c = std::clamp(confidences[i], 0.0, 1.0);
    int idx = static_cast<int>(c * static_cast<double>(bins));
    if (idx >= bins) {
      idx = bins - 1;
    }
    bin_conf[static_cast<std::size_t>(idx)] += c;
    bin_acc[static_cast<std::size_t>(idx)] += static_cast<double>(correct[i]);
    bin_count[static_cast<std::size_t>(idx)] += 1;
  }

  double ece = 0.0;
  for (int b = 0; b < bins; ++b) {
    const int count = bin_count[static_cast<std::size_t>(b)];
    if (count <= 0) {
      continue;
    }
    const double avg_conf = bin_conf[static_cast<std::size_t>(b)] / static_cast<double>(count);
    const double avg_acc = bin_acc[static_cast<std::size_t>(b)] / static_cast<double>(count);
    ece += std::abs(avg_conf - avg_acc) * (static_cast<double>(count) / static_cast<double>(n));
  }
  return std::clamp(1.0 - ece, 0.0, 1.0);
}

double compute_epistemic_ratio(double epistemic_var, double aleatoric_var) {
  const double total = epistemic_var + aleatoric_var + kEps;
  return std::clamp(epistemic_var / total, 0.0, 1.0);
}

double compute_alpha_gria(const double* input, const double* output, int n_samples, int n_dims,
                          int n_bins) {
  if (input == nullptr || output == nullptr || n_samples <= 0 || n_dims <= 0) {
    return 0.5;
  }
  const int flat = n_samples * n_dims;
  const double h_x = histogram_entropy(input, flat, n_bins);
  const double h_f = histogram_entropy(output, flat, n_bins);
  const double alpha = 1.0 - h_f / (h_x + kEps);
  return std::clamp(alpha, 0.0, 1.0);
}

}  // namespace cypha::intelligence
