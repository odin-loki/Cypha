#pragma once

/// Reusable EM step primitives for diagonal-Gaussian mixtures.
///
/// E-step: ``responsibilities()`` computes ``r_i ∝ prior_i · exp(loglik_i / T)`` with log-sum-exp
/// normalisation, temperature scaling, and an epsilon floor. M-step: ``weighted_moment_update()``
/// / ``weighted_moment_finalize()`` accumulate weighted mean and diagonal variance.
///
/// Diagonal-Gaussian log-likelihood follows the ``generation.hpp`` ``batch_logpdf`` convention
/// (same Mahalanobis weighting as ``score_matrix_use_field`` via ``inv_v``).

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace cypha {

constexpr double kEmMinVar = 1e-4;
constexpr double kEmEps = 1e-8;

/// ``log p(x | N(mu, diag(v)))`` up to an additive ``d·log(2π)/2`` constant omitted for EM ratios.
inline double diagonal_gaussian_logpdf(const double* x, const double* mu, const double* v, int d,
                                       double min_var = kEmMinVar) {
  double log_norm = 0.0;
  double maha = 0.0;
  for (int j = 0; j < d; ++j) {
    const double vs = std::max(v[static_cast<std::size_t>(j)], min_var);
    log_norm += 0.5 * std::log(vs);
    const double diff = x[static_cast<std::size_t>(j)] - mu[static_cast<std::size_t>(j)];
    maha += 0.5 * diff * diff / vs;
  }
  return -log_norm - maha;
}

/// ``loglik[k] = log p(x | component k)`` with ``mu_k`` / ``v_k`` row-major ``K×d``.
void diagonal_gaussian_loglik_row(const double* x, int d, const double* mu_k, const double* v_k, int k,
                                  double min_var, double* loglik_out);

/// Normalised responsibilities ``r[k]`` from per-component log-likelihoods and mixing priors.
void responsibilities(const double* loglik, const double* prior, int k, double temperature, double eps,
                      double* r_out);

/// Accumulate weighted first and second moments for one observation (M-step building block).
inline void weighted_moment_update(const double* x, int d, double weight, double& weight_sum, double* mean_acc,
                                   double* m2_acc) {
  weight_sum += weight;
  for (int j = 0; j < d; ++j) {
    const double xj = x[static_cast<std::size_t>(j)];
    mean_acc[static_cast<std::size_t>(j)] += weight * xj;
    m2_acc[static_cast<std::size_t>(j)] += weight * xj * xj;
  }
}

/// Divide accumulators by ``weight_sum``; write mean and diagonal variance (floored at ``min_var``).
void weighted_moment_finalize(double weight_sum, const double* mean_acc, const double* m2_acc, int d,
                              double* mean_out, double* var_out, double min_var = kEmMinVar);

/// Mixture log-likelihood ``Σ_n log Σ_k prior_k · exp(loglik_{nk})`` (log-sum-exp per row).
double mixture_log_likelihood(const double* x_row, int n, int d, const double* mu_k, const double* v_k,
                              const double* prior, int k, double min_var = kEmMinVar);

}  // namespace cypha
