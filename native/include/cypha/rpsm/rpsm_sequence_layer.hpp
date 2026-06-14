#pragma once

#include <cstdint>
#include <vector>

#include "cypha/rpsm/psi_matrices.hpp"

namespace cypha::rpsm {

/// Tiny/Small RPSM sequence layer config (Option B scaffold).
struct RpsmSequenceConfig {
  int n_levels = 4;
  int state_dim = 128;
  int feat_dim = 64;
  int n_classes = 128;
  double alpha_carry = 0.5;
  std::uint64_t seed = 42;
};

/// Minimal RPSM sequence layer: level-0 batched LLR via ``PsiMatrices`` and hidden carry ``h_t``.
class RpsmSequenceLayer {
 public:
  explicit RpsmSequenceLayer(RpsmSequenceConfig cfg);

  void reset();

  /// One timestep. ``input`` length ``input_dim``; ``log_probs_out`` length ``n_classes`` (log space).
  /// Returns Frobenius norm of carried hidden state (diagnostics).
  double step(const double* input, int input_dim, double* log_probs_out);

  const std::vector<double>& hidden() const { return h_; }
  const PsiMatrices& psi() const { return psi_; }
  int n_classes() const { return cfg_.n_classes; }

 private:
  RpsmSequenceConfig cfg_;
  PsiMatrices psi_;
  std::vector<double> h_;
  std::vector<double> psi_rows_;
  std::vector<double> w_enc_;
  std::vector<double> w_carry_;
  std::vector<double> feat_buf_;
  std::vector<double> llr_buf_;

  static void log_softmax_row(const double* logits, int k, double* log_out);
};

}  // namespace cypha::rpsm
