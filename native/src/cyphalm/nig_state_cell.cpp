#include "cypha/cyphalm/nig_state_cell.hpp"

#include <stdexcept>

namespace cypha::cyphalm {

NigStateCell::NigStateCell(double kappa0, double alpha0, double beta0, int dim)
    : kappa0_(kappa0), alpha0_(alpha0), beta0_(beta0), dim_(dim), state_(kappa0, alpha0, beta0, dim) {}

void NigStateCell::reset() {
  state_ = NIGExpert(kappa0_, alpha0_, beta0_, dim_);
}

std::vector<double> NigStateCell::step(const double* x, int dim) {
  if (dim != state_.dim()) {
    throw std::invalid_argument("NigStateCell::step: dimension mismatch");
  }
  state_.update(x);
  std::vector<double> mean;
  state_.predictive_mean(mean);
  return mean;
}

}  // namespace cypha::cyphalm
