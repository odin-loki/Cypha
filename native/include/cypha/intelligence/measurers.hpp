#pragma once

namespace cypha::intelligence {

/// Participation ratio ``(Σλ)² / Σλ²`` from column variances, divided by ``n_dims``.
double compute_participation_ratio(const double* activations, int n_samples, int n_dims);

/// Expected calibration error; returns ``C = 1 - ECE``.
double compute_calibration(const double* confidences, const int* correct, int n, int n_bins = 10);

/// ``r_eu = σ²_e / (σ²_e + σ²_a)``.
double compute_epistemic_ratio(double epistemic_var, double aleatoric_var);

/// GRIA grade ``α = 1 - H(f(X)) / H(X)`` from histogram entropies.
double compute_alpha_gria(const double* input, const double* output, int n_samples, int n_dims,
                          int n_bins = 16);

/// Map raw branching ratio ``σ ∈ [0, ∞)`` to ``σ/(1+σ)``.
double normalize_branching_ratio(double sigma_raw);

/// Empirical Lipschitz ``E[||f(x+δ)-f(x)|| / ||δ||]`` from paired activations.
double compute_lipschitz_sensitivity(const double* base, const double* perturbed, int n_samples,
                                     int n_dims);

/// Branching ratio from output sensitivity to input perturbation.
double compute_branching_ratio_sensitivity(const double* base_output, const double* perturbed_output,
                                         const double* perturbation, int n_samples, int n_dims);

/// Log-normalised memory depth ``log(τ+1)/log(τ_max+1)``.
double normalize_memory_depth(int tau_steps, int tau_max = 512);

/// Estimate ``τ`` from lagged sequence correlation; returns normalised depth in ``[0,1]``.
double compute_memory_depth_normalized(const double* sequence, int n_timesteps, int n_dims,
                                       int max_lag = 32, int tau_max = 512);

}  // namespace cypha::intelligence
