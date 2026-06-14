#pragma once

#include <vector>

#include "cypha/cyphalm/cyphalm_nig_expert.hpp"

namespace cypha::cyphalm {

/// H06: recurrent cell whose hidden state is a diagonal NIG expert (one-step Bayesian update).
class NigStateCell {
 public:
  NigStateCell(double kappa0, double alpha0, double beta0, int dim);

  void reset();
  /// Observe ``x``, update NIG sufficient statistics, return predictive mean (cell state).
  std::vector<double> step(const double* x, int dim);
  const NIGExpert& state() const { return state_; }
  double epistemic_variance_mean() const { return state_.epistemic_variance_mean(); }

 private:
  double kappa0_;
  double alpha0_;
  double beta0_;
  int dim_;
  NIGExpert state_;
};

}  // namespace cypha::cyphalm
