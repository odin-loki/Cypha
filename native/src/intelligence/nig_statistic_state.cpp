#include "cypha/intelligence/nig_statistic_state.hpp"

#include <cmath>
#include <stdexcept>

namespace cypha::intelligence {

namespace {

constexpr double kMinDenominator = 1e-12;

}  // namespace

NigStatisticState::NigStatisticState(double mu0, double lambda0, double alpha0, double beta0)
    : mu_(mu0), lambda_(lambda0), alpha_(alpha0), beta_(beta0), running_mean_(mu0) {
  if (lambda0 <= 0.0 || alpha0 <= 1.0 || beta0 <= 0.0) {
    throw std::invalid_argument("NigStatisticState: require lambda > 0, alpha > 1, beta > 0");
  }
}

void NigStatisticState::update(double observation) {
  ++n_observations_;
  const double delta = observation - running_mean_;
  running_mean_ += delta / static_cast<double>(n_observations_);
  const double delta2 = observation - running_mean_;
  running_m2_ += delta * delta2;

  const double mu_prior = mu_;
  const double lambda_prior = lambda_;
  lambda_ = lambda_prior + 1.0;
  mu_ = (lambda_prior * mu_prior + observation) / lambda_;
  alpha_ = alpha_ + 0.5;
  beta_ = beta_ + lambda_prior * (observation - mu_prior) * (observation - mu_prior) / (2.0 * lambda_);
  ++n_updates_;
}

double NigStatisticState::epistemic_var() const {
  const double denom = lambda_ * (alpha_ - 1.0);
  return beta_ / std::max(denom, kMinDenominator);
}

double NigStatisticState::aleatoric_var() const {
  return beta_ / std::max(alpha_ - 1.0, kMinDenominator);
}

double NigStatisticState::observation_variance() const {
  if (n_observations_ < 2) {
    return 1e-8;
  }
  return running_m2_ / static_cast<double>(n_observations_ - 1);
}

}  // namespace cypha::intelligence
