#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cypha {
namespace cyphalm {

inline double sigmoid(double x) {
  x = std::max(-40.0, std::min(40.0, x));
  return 1.0 / (1.0 + std::exp(-x));
}

/// Convex blend in probability space: alpha*GRIA + (1-alpha)*LSTM (log domain).
inline void blend_log_probs(const double* log_g, const double* log_l, int vocab_size, double blend_logit,
                            double* log_out) {
  const double alpha = sigmoid(blend_logit);
  for (int k = 0; k < vocab_size; ++k) {
    const double p_g = std::exp(log_g[k]);
    const double p_l = std::exp(log_l[k]);
    const double p = alpha * p_g + (1.0 - alpha) * p_l;
    log_out[k] = std::log(p + 1e-12);
  }
}

inline std::vector<double> blend_log_probs(const std::vector<double>& log_g, const std::vector<double>& log_l,
                                           double blend_logit) {
  const int n = static_cast<int>(log_g.size());
  std::vector<double> out(static_cast<std::size_t>(n));
  blend_log_probs(log_g.data(), log_l.data(), n, blend_logit, out.data());
  return out;
}

/// d/d(blend_logit) of CE loss on blended distribution for ``target_id``.
inline double blend_logit_grad(const double* log_g, const double* log_l, int vocab_size, double blend_logit,
                               int target_id) {
  if (target_id < 0 || target_id >= vocab_size) {
    return 0.0;
  }
  const double alpha = sigmoid(blend_logit);
  const double p_g = std::exp(log_g[target_id]);
  const double p_l = std::exp(log_l[target_id]);
  const double p_t = alpha * p_g + (1.0 - alpha) * p_l;
  if (p_t <= 0.0) {
    return 0.0;
  }
  const double d_loss_d_alpha = -(p_g - p_l) / p_t;
  const double d_alpha_d_logit = alpha * (1.0 - alpha);
  return d_loss_d_alpha * d_alpha_d_logit;
}

}  // namespace cyphalm
}  // namespace cypha
