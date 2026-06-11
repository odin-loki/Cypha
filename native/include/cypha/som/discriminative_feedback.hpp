#pragma once

/// Encoder update modulation from class separation + inverse variance (mirrors
/// cypha_som.discriminative_feedback.DiscriminativeFeedback, upgrade U4).

#include <vector>

namespace cypha::som {

struct DiscriminativeFeedbackConfig {
  double beta{0.1};
};

class DiscriminativeFeedback {
 public:
  explicit DiscriminativeFeedback(DiscriminativeFeedbackConfig cfg = {});

  /// ``delta_mu``: K×d row-major; ``inv_v``: length d. Returns length-d importance weights.
  std::vector<double> compute_d(const std::vector<double>& delta_mu, int K, int d,
                                const std::vector<double>& inv_v) const;

  /// 2-D row-major (rows×cols) or 1-D; ``d`` must match column count or vector length.
  std::vector<double> modulate(const std::vector<double>& dW, int rows, int cols,
                               const std::vector<double>& d) const;

  /// In-place 1-D modulation; no-op when sizes mismatch.
  void modulate_inplace(std::vector<double>& dW, const std::vector<double>& d) const;

  double beta() const { return beta_; }

 private:
  double beta_;
};

}  // namespace cypha::som
