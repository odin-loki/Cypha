#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace cypha {

/// Default variational-IB trade-off ``β`` (Alemi et al.); meaningful when ``use_variational_ib_encoder``.
constexpr double kVariationalIbBetaDefault = 1.0;
/// Isotropic prior std for ``KL(q(t|x) || N(0, σ²I))`` compression term.
constexpr double kVariationalIbPriorSigma = 1.0;

/// Fisher–Rao-style encoder update: `W += lr * weight * outer(r_j - r_k, f)` (row-major `W`, `d×d`).
void contrastive_update_encoder_w(std::vector<double>& w_row_major, int d, const double* f, const double* h,
                                  const double* mu_k, const double* v_k, const double* mu_j, const double* v_j,
                                  double weight, double lr, int& update_count_for_fro_cap);

/// Variational IB encoder update (Alemi bound, deterministic ``h=Wf`` limit):
/// ``grad_h = h/σ² − β·(r_k − r_j)`` then ``W += lr·weight·outer(grad_h, f)``. ``‖W‖_F ≤ 8`` unchanged.
void variational_ib_update_encoder_w(std::vector<double>& w_row_major, int d, const double* f, const double* h,
                                     const double* mu_k, const double* v_k, const double* mu_j, const double* v_j,
                                     double weight, double lr, double beta, int& update_count_for_fro_cap);

/// Latent MI proxy in ``[0,1]``: between-class centroid separation / (within + between).
/// Higher ⇒ latent carries more label information at fixed ``‖h‖``.
double latent_class_mi_proxy(const std::vector<std::vector<double>>& h_samples,
                             const std::vector<std::string>& labels);

/// Python `EncoderProjection.align_to_offsets` (VectorEncoder path).
void encoder_align_to_offsets(std::vector<double>& w_row_major, int d,
                              const std::vector<std::vector<double>>& delta_mus);

/// Python ``EncoderProjection.__init__``: QR(normal) * 0.5 row-major ``W`` (``d×d``).
void init_encoder_projection_w(int d, std::uint64_t seed, std::vector<double>& w_row_major);

}  // namespace cypha
