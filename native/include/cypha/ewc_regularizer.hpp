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
///
/// ``D`` (class-delta score matrix) is a *growable* per-class table: `CyphaDifMemoryState::D`
/// resizes (append-only, existing rows preserved in place — see `memory_train.cpp`
/// `get_or_create`) every time a brand-new class label is trained, which is exactly what happens
/// in a shared-model multi-task continual-learning scenario (e.g. bench D16B: task A's classes
/// are anchored, then task B introduces *new* classes). `penalty()`/`apply_pull()` therefore
/// compare/pull only the ``anchor_D_``-sized *prefix* of the current ``D`` (the rows that existed
/// at snapshot time) rather than requiring an exact size match — new-task rows are untouched
/// (there is no Fisher information for a class that didn't exist at snapshot time, so nothing to
/// protect there). Before 2026-07-12 these functions required an exact size match, which made the
/// ``D`` term of the penalty/pull silently degrade to a no-op the moment any new class was trained
/// after `snapshot()` — see docs/reports/EWC_D16B_SCOPING_2026-07-12.md for the diagnosis. This
/// only affects ``D``; ``enc_w`` is always a fixed ``d_latent × d_latent`` matrix and was never
/// affected by this size-mismatch issue.
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

  /// Opt-in (default off): also anchor the NIG world field mean (``mem.world_mu``), using its own
  /// snapshot-time inverse variance (``mem.world_inv_v``) as the diagonal Fisher/precision — this
  /// is exactly the Fisher information of a Gaussian mean under known variance, so it requires no
  /// separate calibration pass. ``world_mu``/``world_v`` are fixed-size (``d_latent``, do not grow
  /// with new classes) shared statistics updated by every training step regardless of task
  /// (`memory_train.cpp` `world_update`) — they are the actual "NIG field" referenced by
  /// `docs/RESEARCH_STATUS.md` Priority 5 ("EWC as a post-hoc overlay on the NIG field"), distinct
  /// from the class-delta / encoder weights this class already anchored. Must be called before
  /// `snapshot()` / `snapshot_calibrated()` to take effect for that snapshot. Off by default so
  /// existing callers (D16B/D16H bench probes, REST `/train`, all pre-existing `EwcRegularizer`
  /// tools) are byte-identical unless they opt in.
  void set_protect_world_field(bool enabled) { protect_world_field_ = enabled; }
  bool protect_world_field() const { return protect_world_field_; }

  /// ``λ/2 Σ F_i (θ_i − θ*_i)²`` over ``D``, ``enc_w``, and (if `set_protect_world_field(true)`) ``world_mu``.
  double penalty(const CyphaDifMemoryState& mem, const CyphaInferModel& infer) const;

  /// Pull parameters toward anchor with strength ``ewc_lambda * lr``.
  void apply_pull(CyphaDifMemoryState& mem, CyphaInferModel& infer, double ewc_lambda, double lr) const;

  bool has_snapshot() const { return !anchor_D_.empty() || !anchor_world_mu_.empty(); }

 private:
  bool protect_world_field_{false};
  std::vector<double> anchor_D_;
  std::vector<double> anchor_enc_w_;
  std::vector<double> anchor_world_mu_;
  std::vector<double> fisher_D_;
  std::vector<double> fisher_enc_w_;
  std::vector<double> fisher_world_mu_;
};

}  // namespace cypha
