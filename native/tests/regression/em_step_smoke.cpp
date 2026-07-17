/// EM keystone smoke: recover two separated 1-D Gaussians; mixture log-likelihood non-decreasing.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cypha/em_step.hpp"

namespace {

constexpr double kTrueMu0 = -5.0;
constexpr double kTrueMu1 = 5.0;
constexpr double kTrueVar = 1.0;
constexpr int kN = 400;
constexpr int kMaxIter = 50;
constexpr double kMeanTol = 0.5;
constexpr double kLlTol = 1e-9;

double sample_1d(std::mt19937& rng, double mu, double var) {
  std::normal_distribution<double> dist(mu, std::sqrt(var));
  return dist(rng);
}

}  // namespace

int main() {
  std::mt19937 rng(42);
  std::vector<double> x(static_cast<std::size_t>(kN));
  for (int i = 0; i < kN / 2; ++i) {
    x[static_cast<std::size_t>(i)] = sample_1d(rng, kTrueMu0, kTrueVar);
  }
  for (int i = kN / 2; i < kN; ++i) {
    x[static_cast<std::size_t>(i)] = sample_1d(rng, kTrueMu1, kTrueVar);
  }

  constexpr int k = 2;
  constexpr int d = 1;
  std::vector<double> mu_k = {-2.0, 2.0};
  std::vector<double> v_k = {1.0, 1.0};
  std::vector<double> prior = {0.5, 0.5};

  std::vector<double> ll_hist;
  ll_hist.reserve(static_cast<std::size_t>(kMaxIter + 1));
  ll_hist.push_back(cypha::mixture_log_likelihood(x.data(), kN, d, mu_k.data(), v_k.data(), prior.data(), k));

  std::vector<double> loglik(static_cast<std::size_t>(k));
  std::vector<double> r(static_cast<std::size_t>(k));
  std::vector<double> mean_acc(static_cast<std::size_t>(d * k), 0.0);
  std::vector<double> m2_acc(static_cast<std::size_t>(d * k), 0.0);
  std::vector<double> nk(static_cast<std::size_t>(k), 0.0);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    std::fill(mean_acc.begin(), mean_acc.end(), 0.0);
    std::fill(m2_acc.begin(), m2_acc.end(), 0.0);
    std::fill(nk.begin(), nk.end(), 0.0);

    for (int i = 0; i < kN; ++i) {
      const double xi = x[static_cast<std::size_t>(i)];
      cypha::diagonal_gaussian_loglik_row(&xi, d, mu_k.data(), v_k.data(), k, cypha::kEmMinVar, loglik.data());
      cypha::responsibilities(loglik.data(), prior.data(), k, 1.0, cypha::kEmEps, r.data());
      for (int j = 0; j < k; ++j) {
        cypha::weighted_moment_update(&xi, d, r[static_cast<std::size_t>(j)], nk[static_cast<std::size_t>(j)],
                                      mean_acc.data() + static_cast<std::ptrdiff_t>(j) * d,
                                      m2_acc.data() + static_cast<std::ptrdiff_t>(j) * d);
      }
    }

    for (int j = 0; j < k; ++j) {
      cypha::weighted_moment_finalize(nk[static_cast<std::size_t>(j)],
                                      mean_acc.data() + static_cast<std::ptrdiff_t>(j) * d,
                                      m2_acc.data() + static_cast<std::ptrdiff_t>(j) * d, d,
                                      mu_k.data() + static_cast<std::ptrdiff_t>(j) * d,
                                      v_k.data() + static_cast<std::ptrdiff_t>(j) * d);
    }
    for (int j = 0; j < k; ++j) {
      prior[static_cast<std::size_t>(j)] = nk[static_cast<std::size_t>(j)] / static_cast<double>(kN);
    }

    ll_hist.push_back(cypha::mixture_log_likelihood(x.data(), kN, d, mu_k.data(), v_k.data(), prior.data(), k));
  }

  for (std::size_t t = 1; t < ll_hist.size(); ++t) {
    if (ll_hist[t] + kLlTol < ll_hist[t - 1]) {
      std::fprintf(stderr, "em_step_smoke: log-likelihood decreased at step %zu (%.6f -> %.6f)\n", t,
                   ll_hist[t - 1], ll_hist[t]);
      return 1;
    }
  }

  const double mu_a = mu_k[0];
  const double mu_b = mu_k[1];
  const double est_lo = std::min(mu_a, mu_b);
  const double est_hi = std::max(mu_a, mu_b);
  if (std::abs(est_lo - kTrueMu0) > kMeanTol || std::abs(est_hi - kTrueMu1) > kMeanTol) {
    std::fprintf(stderr, "em_step_smoke: mean recovery failed (%.3f, %.3f)\n", est_lo, est_hi);
    return 1;
  }

  const int left = (mu_a <= mu_b) ? 0 : 1;
  const int right = 1 - left;
  for (int i = 0; i < kN; ++i) {
    const double xi = x[static_cast<std::size_t>(i)];
    cypha::diagonal_gaussian_loglik_row(&xi, d, mu_k.data(), v_k.data(), k, cypha::kEmMinVar, loglik.data());
    cypha::responsibilities(loglik.data(), prior.data(), k, 1.0, cypha::kEmEps, r.data());
    const int prefer = (xi < 0.0) ? left : right;
    const int other = 1 - prefer;
    if (!(r[static_cast<std::size_t>(prefer)] > r[static_cast<std::size_t>(other)])) {
      std::fprintf(stderr, "em_step_smoke: bimodal responsibility failed at x=%.3f\n", xi);
      return 1;
    }
  }

  std::puts("em_step_smoke: PASS");
  return 0;
}
