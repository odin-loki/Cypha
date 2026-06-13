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

}  // namespace cypha::intelligence
