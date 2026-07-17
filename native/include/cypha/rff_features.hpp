#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace cypha {

/// Random Fourier feature projection kind (Phase 5 optimality).
enum class RffProjectionKind {
  /// Legacy iid Gaussian rows (Python ``RFFEncoder`` default).
  IidGaussian,
  /// Structured orthogonal random features (SORF / Fastfood blocks; Yu et al.).
  Sorf,
};

/// In-place Walsh–Hadamard transform; ``n`` must be a power of two.
void fwht_inplace(double* v, int n);

/// Next power of two ≥ ``n`` (minimum 1).
int next_pow2(int n);

/// Fill ``w_flat`` (**D×d_in** row-major) and ``b`` (length ``D``) with iid Gaussian RFF weights.
/// ``gamma`` is the RBF bandwidth (row scale ``N(0, sqrt(2·gamma))`` in ``KernelMemory::make_rff``;
/// ``N(0, gamma)`` in preprocessor — pass the value your caller already uses).
void init_rff_weights_iid(std::mt19937& rng, double gamma, int D, int d_in, std::vector<double>& w_flat,
                          std::vector<double>& b, bool kernel_memory_scale = false);

/// SORF / Fastfood structured orthogonal RFF weights (same layout as ``init_rff_weights_iid``).
void init_rff_weights_sorf(std::mt19937& rng, double gamma, int D, int d_in, std::vector<double>& w_flat,
                           std::vector<double>& b, bool kernel_memory_scale = false);

/// Dispatch by ``kind``; default ``IidGaussian`` preserves legacy numerics.
void init_rff_weights(RffProjectionKind kind, std::mt19937& rng, double gamma, int D, int d_in,
                      std::vector<double>& w_flat, std::vector<double>& b,
                      bool kernel_memory_scale = false);

/// Exact RBF kernel value.
double rbf_kernel_value(const double* a, const double* b, int d, double gamma);

/// ``‖K − K̂_rff‖_F`` on ``n`` points: ``K̂_ij = φ(x_i)·φ(x_j)`` with cosine RFF features.
/// ``gamma`` is the exact RBF bandwidth used for ``K``.
double rff_kernel_frobenius_error(const double* X_rowmajor, int n, int d, const double* W, const double* b,
                                  int D, double gamma);

/// ``‖K − K̂_nystrom‖_F`` for whitened Nyström with ``m`` landmarks (row-major ``m×d``).
double nystrom_kernel_frobenius_error(const double* X_rowmajor, int n, int d, const double* landmarks_rowmajor,
                                      int m, double gamma, double ridge = 1e-6);

/// Select ``m`` landmark indices from ``n`` points via ridge-leverage-score sampling (Alaoui–Mahoney).
/// Writes chosen row indices into ``out_indices`` (length ``m``). Uses ``rng_seed`` for reproducibility.
void select_leverage_landmark_indices(const double* X_rowmajor, int n, int d, int m, double gamma,
                                      double ridge, std::uint64_t rng_seed, std::vector<int>& out_indices);

}  // namespace cypha
