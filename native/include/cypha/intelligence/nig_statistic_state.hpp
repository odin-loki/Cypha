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

  /// Every field, for ``Cypha::save`` / ``Cypha::load``.
  struct Raw {
    double mu{};
    double lambda{};
    double alpha{};
    double beta{};
    double running_mean{};
    double running_m2{};
    int n_updates{0};
    int n_observations{0};
  };
  Raw raw() const {
    return Raw{mu_, lambda_, alpha_, beta_, running_mean_, running_m2_, n_updates_, n_observations_};
  }
  void set_raw(const Raw& r) {
    mu_ = r.mu;
    lambda_ = r.lambda;
    alpha_ = r.alpha;
    beta_ = r.beta;
    running_mean_ = r.running_mean;
    running_m2_ = r.running_m2;
    n_updates_ = r.n_updates;
    n_observations_ = r.n_observations;
  }

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
