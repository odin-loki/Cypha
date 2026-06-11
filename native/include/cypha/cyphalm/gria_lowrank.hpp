#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cypha {
namespace cyphalm {

struct GRIALowRankGrad {
  std::vector<double> dU;       // field_dim x rank
  std::vector<double> dV;       // rank x vocab_size
  std::vector<double> d_alpha;  // vocab_size
  std::vector<double> d_bias;   // vocab_size
  std::vector<double> dv;       // field_dim
};

/// Low-rank GRIA: W^T = U @ V with U (field_dim, rank), V (rank, vocab).
/// Logits: alpha_k * z_k + (1-alpha_k) * bias_k where z = V^T @ (U^T @ v).
class GRIALowRank {
 public:
  int field_dim{160};
  int vocab_size{256};
  int rank{32};
  bool alpha_learnable{true};

  std::vector<double> U;      // field_dim x rank row-major
  std::vector<double> V;      // rank x vocab_size row-major
  std::vector<double> alpha;  // vocab_size
  std::vector<double> bias;   // vocab_size

  GRIALowRank() = default;
  GRIALowRank(int field_dim_in, int vocab_size_in, int rank_in = 32, double alpha_init = 0.5,
              bool alpha_learnable_in = true, std::uint64_t seed = 42);

  void logits(const double* v, double* z_out) const;
  void forward(const double* v, double* log_probs_out) const;

  GRIALowRankGrad cross_entropy_gradients(const double* v, int target_id) const;

  void update_weights(const GRIALowRankGrad& grad, double lr);
  void update_alpha(const GRIALowRankGrad& grad, double lr);
  void update_bias(const GRIALowRankGrad& grad, double lr);

  /// CE loss + gradient descent on U, V, alpha, bias.
  double train_step(const double* v, int target_id, double lr);

  void set_laplace_prior(const double* token_counts, int n, double smoothing);
  std::vector<double> grad_v_cross_entropy(const double* v, int target_id) const;

  void load_state(const std::vector<double>& u, const std::vector<double>& v,
                  const std::vector<double>& alpha_in, const std::vector<double>& bias_in,
                  bool alpha_learnable_in);

  /// Approximate Python full-rank ``W`` (vocab × d_input row-major flat) as low-rank U/V.
  void load_from_full_w(const std::vector<double>& w_vocab_x_d, int d_input, int target_rank);

  std::map<std::string, double> alpha_spectrum() const;
};

}  // namespace cyphalm
}  // namespace cypha
