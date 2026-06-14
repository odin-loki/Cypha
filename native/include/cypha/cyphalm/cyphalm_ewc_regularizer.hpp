#pragma once

#include "cypha/cyphalm/char_lstm.hpp"

#include <vector>

namespace cypha::cyphalm {

/// EWC overlay on char-LSTM weights: embed ``E``, recurrent ``Wx``/``Wh``, lm_head ``Wy``/``by``.
/// Diagonal Fisher stub: running average of grad² after snapshot, else anchor² at snapshot.
class CyphaLMEwcRegularizer {
 public:
  void snapshot(const CharLSTMHead& lstm);

  /// Accumulate ``F_i ≈ E[g_i²]`` from a BPTT-1 step (``dE``, ``dWx``, ``dWh``, ``dWy``, ``dby``).
  void observe_grads(const CharLSTMGrad& grads);

  /// ``λ/2 Σ F_i (θ_i − θ*_i)²`` over embed, recurrent, and lm_head weights.
  double penalty(const CharLSTMHead& lstm) const;

  /// Pull snapshotted weights toward anchor with strength ``ewc_lambda * lr``.
  void apply_pull(CharLSTMHead& lstm, double ewc_lambda, double lr) const;

  bool has_snapshot() const { return !anchor_Wx_.empty(); }

  /// True when embed table and lm_head anchors were captured (full Fisher overlay).
  bool covers_embed_and_head() const {
    return !anchor_E_.empty() && !anchor_Wy_.empty() && !anchor_by_.empty();
  }

 private:
  std::vector<double> anchor_E_;
  std::vector<double> anchor_Wx_;
  std::vector<double> anchor_Wh_;
  std::vector<double> anchor_Wy_;
  std::vector<double> anchor_by_;
  std::vector<double> fisher_E_;
  std::vector<double> fisher_Wx_;
  std::vector<double> fisher_Wh_;
  std::vector<double> fisher_Wy_;
  std::vector<double> fisher_by_;
  std::size_t grad_observations_{0};
};

}  // namespace cypha::cyphalm
