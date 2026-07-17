#include "cypha/em_step.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cypha {

namespace {

void log_sum_exp_normalize(const double* log_w, int k, double eps, double* out) {
  if (k <= 0) {
    return;
  }
  double mx = log_w[0];
  for (int i = 1; i < k; ++i) {
    mx = std::max(mx, log_w[static_cast<std::size_t>(i)]);
  }
  double sum = 0.0;
  for (int i = 0; i < k; ++i) {
    out[static_cast<std::size_t>(i)] = std::exp(log_w[static_cast<std::size_t>(i)] - mx);
    sum += out[static_cast<std::size_t>(i)];
  }
  if (k <= 8) {
    sum += eps;
  } else {
    sum = std::max(sum, eps);
  }
  for (int i = 0; i < k; ++i) {
    out[static_cast<std::size_t>(i)] /= sum;
  }
}

}  // namespace

void diagonal_gaussian_loglik_row(const double* x, int d, const double* mu_k, const double* v_k, int k,
                                  double min_var, double* loglik_out) {
  for (int i = 0; i < k; ++i) {
    const double* mu = mu_k + static_cast<std::ptrdiff_t>(i) * d;
    const double* v = v_k + static_cast<std::ptrdiff_t>(i) * d;
    loglik_out[static_cast<std::size_t>(i)] = diagonal_gaussian_logpdf(x, mu, v, d, min_var);
  }
}

void responsibilities(const double* loglik, const double* prior, int k, double temperature, double eps,
                      double* r_out) {
  if (k <= 0) {
    return;
  }
  const double inv_t = 1.0 / (temperature + eps);
  std::vector<double> log_w(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    log_w[static_cast<std::size_t>(i)] =
        std::log(prior[static_cast<std::size_t>(i)] + eps) + loglik[static_cast<std::size_t>(i)] * inv_t;
  }
  log_sum_exp_normalize(log_w.data(), k, eps, r_out);
}

void weighted_moment_finalize(double weight_sum, const double* mean_acc, const double* m2_acc, int d,
                              double* mean_out, double* var_out, double min_var) {
  const double denom = std::max(weight_sum, min_var);
  for (int j = 0; j < d; ++j) {
    const double mean = mean_acc[static_cast<std::size_t>(j)] / denom;
    const double ex2 = m2_acc[static_cast<std::size_t>(j)] / denom;
    mean_out[static_cast<std::size_t>(j)] = mean;
    var_out[static_cast<std::size_t>(j)] = std::max(ex2 - mean * mean, min_var);
  }
}

double mixture_log_likelihood(const double* x_row, int n, int d, const double* mu_k, const double* v_k,
                              const double* prior, int k, double min_var) {
  if (n <= 0 || k <= 0 || d <= 0) {
    return 0.0;
  }
  std::vector<double> loglik(static_cast<std::size_t>(k));
  std::vector<double> log_w(static_cast<std::size_t>(k));
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const double* x = x_row + static_cast<std::ptrdiff_t>(i) * d;
    diagonal_gaussian_loglik_row(x, d, mu_k, v_k, k, min_var, loglik.data());
    for (int j = 0; j < k; ++j) {
      log_w[static_cast<std::size_t>(j)] =
          std::log(prior[static_cast<std::size_t>(j)] + kEmEps) + loglik[static_cast<std::size_t>(j)];
    }
    double mx = log_w[0];
    for (int j = 1; j < k; ++j) {
      mx = std::max(mx, log_w[static_cast<std::size_t>(j)]);
    }
    double sum = 0.0;
    for (int j = 0; j < k; ++j) {
      sum += std::exp(log_w[static_cast<std::size_t>(j)] - mx);
    }
    total += mx + std::log(std::max(sum, kEmEps));
  }
  return total;
}

}  // namespace cypha
