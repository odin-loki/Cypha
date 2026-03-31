#pragma once

#include <vector>

namespace cypha {

struct CNode;

/// Diagonal timescales τ (same grouping as Python `NIGField`).
void field_diag_a(int fd, std::vector<double>& a_out);

/// `A_eff = diag(a) + W_T` in row-major float32 (per-element float64 sum then float32 round — matches Python).
void recompute_field_a_eff(int fd, const std::vector<double>& w_t_row_major,
                           std::vector<float>& a_eff_out);

/// If ``root`` has valid ``field_W_T`` (square tensor), set or replace top-level ``field_a_eff``
/// (float64 row-major, shape ``fd×fd``) from ``recompute_field_a_eff`` (matches Python ``_A_eff`` layout).
void patch_field_a_eff_into_root(CNode& root);

/// `field_h += strength * (‖field_h‖/‖signal‖) * signal` with L2 cap (Python `NIGField.inject`).
void nig_field_inject(std::vector<double>& field_h, const double* signal, int fd, double strength);

/// `h_new = A_eff @ h` in fp32, cap ‖h‖ ≤ 50, write `field_h`.
void nig_field_evolve(const std::vector<float>& a_eff, int fd, const double* h_in, std::vector<double>& field_h);

/// Causal `W_T` SGD step + spectral-radius trim + refresh `a_eff` (Python `update_causal`).
void nig_field_update_causal(std::vector<double>& w_t_row_major, int fd, const double* h_t, const double* h_target,
                             double lr, std::vector<float>& sr_vec, std::vector<float>& a_eff_out);

/// Map latent `h` (dim `d`) to field injection vector (dim `fd`): normalize `h`, then optional `w_inject @ h_norm`.
bool latent_to_field_signal(const double* h, int d, const std::vector<double>& w_inject_row_major, int fd,
                            std::vector<double>& signal_out);

}  // namespace cypha
