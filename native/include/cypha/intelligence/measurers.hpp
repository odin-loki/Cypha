#pragma once

namespace cypha::intelligence {

/// Participation ratio computation method (Paper IV D_eff).
enum class ParticipationRatioMethod {
  /// Column-variance proxy (fast; legacy default).
  VarianceProxy = 0,
  /// Covariance eigenvalue PR: ``(Σλ)² / Σλ²`` via Jacobi diagonalization (Paper IV
  /// fidelity). ``O(n_dims^3)`` per call; for ``n_dims > 256`` this transparently
  /// delegates to the ``TraceFrobenius`` method below instead of diagonalizing (see
  /// its docs for why that is exact, not an approximation).
  CovarianceEigenvalue = 1,
  /// Covariance eigenvalue PR computed as ``trace(C)^2 / trace(C^2)`` directly from the
  /// covariance matrix, without ever diagonalizing it. For a symmetric PSD matrix,
  /// ``trace(C) == Σλ`` and ``trace(C^2) == Σλ^2 == ||C||_F^2`` exactly, so this is
  /// algebraically identical to ``CovarianceEigenvalue`` (not a proxy/approximation) but
  /// costs ``O(n_dims^2 * n_samples)`` instead of ``O(n_dims^3)`` since the ``O(n_dims^3)``
  /// Jacobi diagonalization is skipped entirely. Safe and recommended for any ``n_dims``,
  /// including well above 256 (Phase 0 fix for the D_eff hidden-dim scale-up plan).
  TraceFrobenius = 2,
};

/// Participation ratio ``(Σλ)² / Σλ²`` from column variances, divided by ``n_dims``.
double compute_participation_ratio(const double* activations, int n_samples, int n_dims);

/// Same as above with explicit method (Phase 35 eigenvalue PR; Phase 0 hidden-dim-scale
/// trace/Frobenius reformulation for ``n_dims`` above the Jacobi-affordable range).
double compute_participation_ratio(const double* activations, int n_samples, int n_dims,
                                   ParticipationRatioMethod method);

/// Raw (unnormalized) participation ratio ``(Σλ)² / Σλ²`` -- an *effective-dimension
/// count* in ``[0, n_dims]`` (bounded above by ``min(n_samples, n_dims)`` in practice,
/// since a covariance estimated from ``n_samples`` rows has rank at most
/// ``n_samples - 1``), *before* the ``/ n_dims`` width-normalization that
/// ``compute_participation_ratio`` applies. ``compute_participation_ratio(...) ==
/// compute_participation_ratio_raw(...) / n_dims`` exactly (Phase 3 follow-up,
/// docs/reports/HIDDEN_DIM_SCALE_PLAN.md: the width-normalized ratio alone cannot
/// distinguish "genuinely lower-dimensional representation" from "under-sampled
/// relative to width," since both push the normalized ratio down the same way -- the
/// raw count plus the caller-known ``n_samples``/``n_dims`` disambiguates).
double compute_participation_ratio_raw(const double* activations, int n_samples, int n_dims,
                                       ParticipationRatioMethod method = ParticipationRatioMethod::VarianceProxy);

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
