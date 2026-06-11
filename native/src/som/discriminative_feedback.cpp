#include "cypha/som/discriminative_feedback.hpp"

#include <cmath>
#include <stdexcept>

namespace cypha::som {

namespace {

constexpr double kEps = 1e-9;

}  // namespace

DiscriminativeFeedback::DiscriminativeFeedback(DiscriminativeFeedbackConfig cfg) : beta_(cfg.beta) {}

std::vector<double> DiscriminativeFeedback::compute_d(const std::vector<double>& delta_mu, int K,
                                                      int d,
                                                      const std::vector<double>& inv_v) const {
  if (K <= 0 || d <= 0) {
    return {};
  }
  if (static_cast<int>(delta_mu.size()) < K * d) {
    throw std::invalid_argument("DiscriminativeFeedback::compute_d: delta_mu too small");
  }
  if (static_cast<int>(inv_v.size()) < d) {
    throw std::invalid_argument("DiscriminativeFeedback::compute_d: inv_v too small");
  }
  std::vector<double> importance(static_cast<std::size_t>(d), 0.0);
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < d; ++j) {
      const double dm = delta_mu[static_cast<std::size_t>(k * d + j)];
      importance[static_cast<std::size_t>(j)] +=
          std::abs(dm) * inv_v[static_cast<std::size_t>(j)];
    }
  }
  double s = 0.0;
  for (double v : importance) {
    s += v;
  }
  s += kEps;
  for (double& v : importance) {
    v /= s;
  }
  return importance;
}

std::vector<double> DiscriminativeFeedback::modulate(const std::vector<double>& dW, int rows,
                                                     int cols,
                                                     const std::vector<double>& d) const {
  if (rows <= 0 || cols <= 0) {
    return dW;
  }
  const int n = rows * cols;
  if (static_cast<int>(dW.size()) < n) {
    return dW;
  }
  if (rows > 1 && static_cast<int>(d.size()) == cols) {
    std::vector<double> out = dW;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const std::size_t idx = static_cast<std::size_t>(r * cols + c);
        out[idx] = dW[idx] + beta_ * (d[static_cast<std::size_t>(c)] * dW[idx]);
      }
    }
    return out;
  }
  if (rows == 1 && static_cast<int>(d.size()) == cols) {
    return modulate(dW, 1, cols, d);
  }
  if (static_cast<int>(d.size()) == n) {
    std::vector<double> out = dW;
    for (int i = 0; i < n; ++i) {
      out[static_cast<std::size_t>(i)] =
          dW[static_cast<std::size_t>(i)] +
          beta_ * (d[static_cast<std::size_t>(i)] * dW[static_cast<std::size_t>(i)]);
    }
    return out;
  }
  return dW;
}

void DiscriminativeFeedback::modulate_inplace(std::vector<double>& dW,
                                              const std::vector<double>& d) const {
  if (dW.empty() || d.size() != dW.size()) {
    return;
  }
  for (std::size_t i = 0; i < dW.size(); ++i) {
    dW[i] += beta_ * (d[i] * dW[i]);
  }
}

}  // namespace cypha::som
