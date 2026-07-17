#pragma once

namespace cypha {

/// GIG / NIG scalars shared by ``infer_cpu``, ``accel``, and CUDA host paths (reference parity fixture).

double gig_e_inv_v_lam_neg1(double chi0, double psi);
double gig_e_v_lam_neg1(double chi0, double psi);
double nig_adapt_chi_impl(double chi, double psi, double innovation_sq, double R, double alpha);
double nig_r_eff_scalar(double mp, double r_base, double chi, double psi);

/// Conjugate NIG posterior scale τ_k = v_mean / (n_obs + 1) (same as MDL ``u_k``).
double nig_delta_posterior_scale(double n_obs, double v_mean);

/// ``Σ_jj = τ / inv_v_j`` diagonal posterior variance of class delta Δ_k.
double nig_delta_posterior_var_j(double n_obs, double v_mean, double inv_v_j);

/// Analytic BMA correction subtracted from MAP class LLR:
/// ``0.5 · τ · (d + Σ_j r_j² / inv_v_j)`` with ``r = inv_v ⊙ (h − μ₀)``.
double nig_delta_bma_llr_correction(int d, double n_obs, double v_mean, const double* inv_v,
                                    const double* r);

/// Epistemic variance ``rᵀ Σ r`` for credible-interval confidence (same τ as above).
double nig_delta_bma_epistemic_var(double n_obs, double v_mean, const double* inv_v, const double* r,
                                   int d);

/// Lower credible bound on ``[0,1]`` probability from point estimate and epistemic std.
double nig_delta_credible_lower(double prob, double epistemic_std, double z, double temperature);

}  // namespace cypha
