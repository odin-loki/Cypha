// Golden χ_new values from reference parity fixture `_nig_adapt(chi, psi, innovation_sq, R, alpha=0.98)` (same α as gh_train_step).
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "cypha/infer_cpu.hpp"

namespace {

struct Row {
  double chi;
  double psi;
  double innov;
  double R;
  double alpha;
  double chi_new_expected;
};

int run() {
  const Row rows[] = {
      {1.0, 1.0, 0.5, 0.1, 0.98, 2.0193943711834006},
      {1.0, 1.0, 2.0, 0.05, 0.98, 5.834957899296454},
      {2.5, 1.0, 0.01, 0.2, 0.98, 1.2218599849092555},
  };
  for (const Row& r : rows) {
    auto out = cypha::nig_adapt_session_chi(r.chi, r.psi, r.innov, r.R, r.alpha);
    if (std::abs(out.second - r.psi) > 1e-12) {
      std::cerr << "psi drift\n";
      return 2;
    }
    if (std::abs(out.first - r.chi_new_expected) > 1e-9) {
      std::cerr << "chi_new mismatch: got " << out.first << " want " << r.chi_new_expected << "\n";
      return 1;
    }
  }
  return 0;
}

}  // namespace

int main() {
  return run();
}
