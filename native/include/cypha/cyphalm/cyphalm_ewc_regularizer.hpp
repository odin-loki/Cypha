#pragma once

#include "cypha/cyphalm/char_lstm.hpp"

#include <vector>

namespace cypha::cyphalm {

/// EWC overlay on char-LSTM recurrent weights (``Wx`` = W_ih, ``Wh`` = W_hh).
/// Diagonal Fisher stub: running average of grad² after snapshot, else anchor² at snapshot.
class CyphaLMEwcRegularizer {
 public:
  void snapshot(const CharLSTMHead& lstm);

  /// Accumulate ``F_i ≈ E[g_i²]`` from a BPTT-1 step (``dWx``, ``dWh`` only).
  void observe_grads(const CharLSTMGrad& grads);

  /// ``λ/2 Σ F_i (θ_i − θ*_i)²`` over ``Wx`` and ``Wh``.
  double penalty(const CharLSTMHead& lstm) const;

  /// Pull ``Wx``/``Wh`` toward anchor with strength ``ewc_lambda * lr``.
  void apply_pull(CharLSTMHead& lstm, double ewc_lambda, double lr) const;

  bool has_snapshot() const { return !anchor_Wx_.empty(); }

 private:
  std::vector<double> anchor_Wx_;
  std::vector<double> anchor_Wh_;
  std::vector<double> fisher_Wx_;
  std::vector<double> fisher_Wh_;
  std::size_t grad_observations_{0};
};

}  // namespace cypha::cyphalm
