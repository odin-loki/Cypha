#pragma once
/// Accelerated compute: **CUDA** when built with ``-DCYPHA_ENABLE_CUDA=ON`` and a
/// GPU is present; otherwise **ISO C++** parallel CPU (``std::thread``) over rows.
///
/// Thread safety: ``init()`` / ``shutdown()`` are not thread-safe.  Compute entry
/// points use one global mutex and are safe for one caller at a time (or wrap
/// externally for concurrent use).

#include <string>

namespace cypha {
namespace accel {

bool init();
bool is_available();  ///< true only when CUDA GPU path is active
std::string device_info();
void shutdown();

void batch_encode(const double* x_row, int n, int d, const double* w_row, double* h_out);

void score_matrix(const double* h_row, int n, int d, int K, const double* mu0,
                  const double* inv_v, const double* D_row, const double* D_sq,
                  const double* u_k, const double* ctx, double* llr_out);

void softmax_rows(const double* logits, int n, int K, double temperature, double* probs_out);

void world_gate_batch(const double* h_row, int n, int d, const double* psi_vec, double chi,
                      double* gates_out);

/// GH–NIG world gate (``world_gate_vector`` / ``world_gate_vector_use_field``): Mahalanobis vs ``mu0``,
/// then ``r_base / max(R_eff, r_base)`` with ``R_eff`` from ``nig_r_eff_scalar``. ``mu0`` must already
/// include field offset when ``use_field`` applies.
void world_gate_nig_field_batch(const double* h_row, int n, int d, const double* mu0,
                                const double* inv_v, double r_base, double gh_chi, double gh_psi,
                                double* gates_out);

}  // namespace accel
}  // namespace cypha
