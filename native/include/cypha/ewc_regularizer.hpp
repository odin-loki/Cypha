#pragma once

#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"

#include <vector>

namespace cypha {

/// Elastic weight consolidation overlay on class deltas ``D`` and encoder ``enc_w``.
/// Diagonal Fisher stub: ``F_i ≈ anchor_i²`` at snapshot time.
class EwcRegularizer {
 public:
  void snapshot(const CyphaDifMemoryState& mem, const CyphaInferModel& infer);

  /// ``λ/2 Σ F_i (θ_i − θ*_i)²`` over ``D`` and ``enc_w``.
  double penalty(const CyphaDifMemoryState& mem, const CyphaInferModel& infer) const;

  /// Pull parameters toward anchor with strength ``ewc_lambda * lr``.
  void apply_pull(CyphaDifMemoryState& mem, CyphaInferModel& infer, double ewc_lambda, double lr) const;

  bool has_snapshot() const { return !anchor_D_.empty(); }

 private:
  std::vector<double> anchor_D_;
  std::vector<double> anchor_enc_w_;
  std::vector<double> fisher_D_;
  std::vector<double> fisher_enc_w_;
};

}  // namespace cypha
