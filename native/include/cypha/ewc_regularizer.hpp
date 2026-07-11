#pragma once

#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"

#include <string>
#include <vector>

namespace cypha {

/// Elastic weight consolidation overlay on class deltas ``D`` and encoder ``enc_w``.
///
/// ``snapshot()`` uses a crude diagonal-Fisher *proxy*: ``F_i ≈ anchor_i²`` (squared anchor
/// *parameter value*, not a gradient-based quantity) at snapshot time. This is NOT the standard
/// EWC diagonal Fisher (``F_i = E[(∂loss/∂θ_i)²]``, an expectation of squared *gradients*) — it is
/// a cheap always-available fallback that requires no calibration data and is kept for backward
/// compatibility with existing callers (`bench/BASELINE_LOCK.json`-adjacent bench domains and the
/// REST `/train` `ewc_snapshot` endpoint both currently rely on this exact behavior).
///
/// ``snapshot_calibrated()`` computes the theoretically-correct diagonal Fisher estimate —
/// ``F_i ≈ E_x[(∂loss/∂θ_i)²]`` — from a calibration batch, by reusing this codebase's existing
/// closed-form training-update math (see `ewc_regularizer.cpp` for the per-parameter gradient
/// derivations, one for `D` from `CyphaDifMemoryState::memory_train`'s loss and one for `enc_w`
/// from `contrastive_update_encoder_w`'s Fisher-Rao residual) rather than the anchor-squared proxy.
/// See docs/reports/STUB_AUDIT_2026-07-11.md.
class EwcRegularizer {
 public:
  void snapshot(const CyphaDifMemoryState& mem, const CyphaInferModel& infer);

  /// Real diagonal-Fisher variant of `snapshot()`: ``F_i`` is estimated from ``E[(∂loss/∂θ_i)²]``
  /// over ``(calib_x, calib_labels)`` (raw preprocessed inputs + true labels, e.g. the just-trained
  /// task's own training set) instead of the squared-anchor proxy. Falls back to `snapshot()`'s
  /// anchor-squared behavior if the calibration batch is empty. Anchors (`anchor_D_`/`anchor_enc_w_`)
  /// are captured identically either way.
  void snapshot_calibrated(const CyphaDifMemoryState& mem, const CyphaInferModel& infer,
                           const std::vector<std::vector<double>>& calib_x, const std::vector<std::string>& calib_labels);

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
