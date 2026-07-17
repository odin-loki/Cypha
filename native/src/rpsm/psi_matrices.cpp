#include "cypha/rpsm/psi_matrices.hpp"

#include <cmath>
#include <vector>

#include "cypha/infer_cpu.hpp"

namespace cypha {
namespace rpsm {

PsiMatrices build_psi_from_model(const CyphaInferModel& m) {
  PsiMatrices psi;
  build_psi_from_model_into(psi, m);
  return psi;
}

void build_psi_from_model_into(PsiMatrices& psi, const CyphaInferModel& m) {
  const int d = m.d_latent;
  const int K = static_cast<int>(m.labels.size());
  psi.feat_dim = d;
  psi.n_classes = K;
  psi.mu.assign(static_cast<std::size_t>((1 + K) * d), 0.0);
  psi.inv_var = m.inv_v;
  psi.counts = m.n_obs;
  psi.v_mean = m.v_mean;

  for (int j = 0; j < d; ++j) {
    psi.mu[static_cast<std::size_t>(j)] = m.mu_world[static_cast<std::size_t>(j)];
  }
  double h_sq = 0.0;
  for (double v : m.field_h) {
    h_sq += v * v;
  }
  if (std::isfinite(h_sq) && h_sq <= 1e8) {
    for (int j = 0; j < d; ++j) {
      double acc = 0.0;
      for (int t = 0; t < m.field_dim; ++t) {
        acc += m.f_field[static_cast<std::size_t>(j * m.field_dim + t)] * m.field_h[static_cast<std::size_t>(t)];
      }
      psi.mu[static_cast<std::size_t>(j)] += acc;
    }
  }

  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < d; ++j) {
      psi.mu[static_cast<std::size_t>((1 + k) * d + j)] = m.D[static_cast<std::size_t>(k * d + j)];
    }
  }
}

void batched_llr_gemm(const double* h_row_major, int n, const PsiMatrices& psi, const double* ctx,
                      double* llr_out) {
  const int d = psi.feat_dim;
  const int K = psi.n_classes;
  if (n <= 0 || K <= 0 || d <= 0) {
    return;
  }

  // Perf (2026-07-17): thread_local b_row/bias — bias/b_row depend only on Ψ, not batch row count;
  // reusing avoids two heap allocs per score_matrix / rpsm_score_matrix call (REST /predict batch
  // paths and uncertainty_rank). Safe: infer_parallel_rows shards rows; each thread owns scratch.
  thread_local std::vector<double> b_row;
  thread_local std::vector<double> bias;
  b_row.resize(static_cast<std::size_t>(K * d));
  bias.resize(static_cast<std::size_t>(K));

  for (int k = 0; k < K; ++k) {
    const double* delta = psi.mu.data() + static_cast<std::size_t>((1 + k) * d);
    double d_sq = 0.0;
    double cross_mu = 0.0;
    for (int j = 0; j < d; ++j) {
      const double bkj = psi.inv_var[static_cast<std::size_t>(j)] * delta[j];
      b_row[static_cast<std::size_t>(k * d + j)] = bkj;
      d_sq += delta[j] * bkj;
      cross_mu += psi.mu[static_cast<std::size_t>(j)] * bkj;
    }
    const double nk = psi.counts[static_cast<std::size_t>(k)];
    const double u_k = psi.v_mean / (nk + 1.0);
    const double ctx_k = ctx != nullptr ? ctx[k] : 0.0;
    bias[static_cast<std::size_t>(k)] = -cross_mu - 0.5 * d_sq - u_k + ctx_k;
  }

  for (int i = 0; i < n; ++i) {
    const double* hi = h_row_major + static_cast<std::size_t>(i * d);
    double* llr_row = llr_out + static_cast<std::size_t>(i * K);
    for (int k = 0; k < K; ++k) {
      double acc = bias[static_cast<std::size_t>(k)];
      const double* bk = b_row.data() + static_cast<std::size_t>(k * d);
      for (int j = 0; j < d; ++j) {
        acc += hi[j] * bk[j];
      }
      llr_row[k] = acc;
    }
  }
}

}  // namespace rpsm
}  // namespace cypha
