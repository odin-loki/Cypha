#pragma once

namespace cypha {

/// GIG / NIG scalars shared by ``infer_cpu``, ``accel``, and CUDA host paths (reference parity fixture).

double gig_e_inv_v_lam_neg1(double chi0, double psi);
double gig_e_v_lam_neg1(double chi0, double psi);
double nig_adapt_chi_impl(double chi, double psi, double innovation_sq, double R, double alpha);
double nig_r_eff_scalar(double mp, double r_base, double chi, double psi);

}  // namespace cypha
