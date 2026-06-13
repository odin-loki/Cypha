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

double normalize_branching_ratio(double sigma_raw) {
  if (sigma_raw < 0.0) {
    return 0.0;
  }
  return std::clamp(sigma_raw / (1.0 + sigma_raw), 0.0, 1.0);
}

double compute_lipschitz_sensitivity(const double* base, const double* perturbed, int n_samples,
                                     int n_dims) {
  if (base == nullptr || perturbed == nullptr || n_samples <= 0 || n_dims <= 0) {
    return 0.5;
  }
  double ratio_sum = 0.0;
  int count = 0;
  for (int r = 0; r < n_samples; ++r) {
    double out_norm = 0.0;
    double in_norm = 0.0;
    for (int d = 0; d < n_dims; ++d) {
      const std::size_t idx = static_cast<std::size_t>(r * n_dims + d);
      const double delta = perturbed[idx] - base[idx];
      out_norm += delta * delta;
      in_norm += base[idx] * base[idx];
    }
    if (in_norm > kEps) {
      ratio_sum += std::sqrt(out_norm) / std::sqrt(in_norm);
      ++count;
    }
  }
  if (count == 0) {
    return 0.5;
  }
  return normalize_branching_ratio(ratio_sum / static_cast<double>(count));
}

double compute_branching_ratio_sensitivity(const double* base_output, const double* perturbed_output,
                                         const double* perturbation, int n_samples, int n_dims) {
  if (base_output == nullptr || perturbed_output == nullptr || perturbation == nullptr ||
      n_samples <= 0 || n_dims <= 0) {
    return 0.5;
  }
  double ratio_sum = 0.0;
  int count = 0;
  for (int r = 0; r < n_samples; ++r) {
    double out_norm = 0.0;
    double delta_norm = 0.0;
    for (int d = 0; d < n_dims; ++d) {
      const std::size_t idx = static_cast<std::size_t>(r * n_dims + d);
      const double out_delta = perturbed_output[idx] - base_output[idx];
      out_norm += out_delta * out_delta;
      delta_norm += perturbation[idx] * perturbation[idx];
    }
    if (delta_norm > kEps) {
      ratio_sum += std::sqrt(out_norm) / std::sqrt(delta_norm);
      ++count;
    }
  }
  if (count == 0) {
    return 0.5;
  }
  return normalize_branching_ratio(ratio_sum / static_cast<double>(count));
}

double normalize_memory_depth(int tau_steps, int tau_max) {
  if (tau_steps <= 0 || tau_max <= 0) {
    return 0.0;
  }
  const int capped = std::min(tau_steps, tau_max);
  const double denom = std::log(static_cast<double>(tau_max) + 1.0);
  if (denom <= kEps) {
    return 0.0;
  }
  return std::clamp(std::log(static_cast<double>(capped) + 1.0) / denom, 0.0, 1.0);
}

double compute_memory_depth_normalized(const double* sequence, int n_timesteps, int n_dims,
                                       int max_lag, int tau_max) {
  if (sequence == nullptr || n_timesteps < 2 || n_dims <= 0 || max_lag < 1) {
    return 0.0;
  }
  const int lag_limit = std::min(max_lag, n_timesteps - 1);
  constexpr double kMiFloor = 0.05;
  int tau_steps = 0;

  for (int lag = 1; lag <= lag_limit; ++lag) {
    double sum_xy = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    int count = 0;
    for (int t = lag; t < n_timesteps; ++t) {
      for (int d = 0; d < n_dims; ++d) {
        const std::size_t prev_idx = static_cast<std::size_t>((t - lag) * n_dims + d);
        const std::size_t cur_idx = static_cast<std::size_t>(t * n_dims + d);
        const double x = sequence[prev_idx];
        const double y = sequence[cur_idx];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        ++count;
      }
    }
    if (count < 2) {
      break;
    }
    const double inv_n = 1.0 / static_cast<double>(count);
    const double cov = sum_xy * inv_n - (sum_x * inv_n) * (sum_y * inv_n);
    const double var_x = sum_x2 * inv_n - (sum_x * inv_n) * (sum_x * inv_n);
    const double var_y = sum_y2 * inv_n - (sum_y * inv_n) * (sum_y * inv_n);
    const double denom = std::sqrt(std::max(var_x, 0.0) * std::max(var_y, 0.0)) + kEps;
    const double corr = std::abs(cov / denom);
    if (corr > kMiFloor) {
      tau_steps = lag;
    } else {
      break;
    }
  }

  return normalize_memory_depth(tau_steps, tau_max);
}

}  // namespace cypha::intelligence
