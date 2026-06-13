#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace cypha {

/// Fisher–Rao-style encoder update: `W += lr * weight * outer(r_j - r_k, f)` (row-major `W`, `d×d`).
void contrastive_update_encoder_w(std::vector<double>& w_row_major, int d, const double* f, const double* h,
                                  const double* mu_k, const double* v_k, const double* mu_j, const double* v_j,
                                  double weight, double lr, int& update_count_for_fro_cap);

/// Python `EncoderProjection.align_to_offsets` (VectorEncoder path).
void encoder_align_to_offsets(std::vector<double>& w_row_major, int d,
                              const std::vector<std::vector<double>>& delta_mus);

/// Python ``EncoderProjection.__init__``: QR(normal) * 0.5 row-major ``W`` (``d×d``).
void init_encoder_projection_w(int d, std::uint64_t seed, std::vector<double>& w_row_major);

}  // namespace cypha
