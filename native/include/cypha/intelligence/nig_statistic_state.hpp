#pragma once

namespace cypha::intelligence {

/// Scalar Normal-Inverse-Gamma posterior for one intelligence statistic (Paper I §4.2).
class NigStatisticState {
 public:
  NigStatisticState(double mu0 = 0.5, double lambda0 = 1.0, double alpha0 = 2.0, double beta0 = 0.1);

  void update(double observation);

  double mean() const { return mu_; }
  double lambda() const { return lambda_; }
  double alpha() const { return alpha_; }
  double beta() const { return beta_; }

  /// Epistemic variance of the posterior mean (NIG marginal).
  double epistemic_var() const;
  /// Aleatoric / irreducible variance component.
  double aleatoric_var() const;

  /// Welford mean of raw observations (health baseline).
  double running_mean() const { return running_mean_; }
  /// Sample variance of raw observations (diagonal Mahalanobis weight).
  double observation_variance() const;

  int n_updates() const { return n_updates_; }

 private:
  double mu_{};
  double lambda_{};
  double alpha_{};
  double beta_{};
  double running_mean_{};
  double running_m2_{};
  int n_updates_{0};
  int n_observations_{0};
};

}  // namespace cypha::intelligence
