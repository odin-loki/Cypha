#pragma once

#include "cypha/cyphalm/cellai_ssm.hpp"
#include "cypha/cyphalm/char_lstm.hpp"
#include "cypha/cyphalm/gria_lowrank.hpp"

#include <vector>

#include <nlohmann/json.hpp>

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

  nlohmann::json get_state() const;
  void set_state(const nlohmann::json& state);

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

struct HybridEwcGradStub {
  CharLSTMGrad lstm;
  bool has_lstm{false};
  std::vector<double> d_gria_alpha;
  std::vector<double> d_gria_U;
  std::vector<double> d_gria_V;
  std::vector<double> d_gria_bias;
  std::vector<double> d_ssm_alpha;
  std::vector<double> d_ssm_w_fast;
  std::vector<double> d_ssm_w_slow;
};

/// Hybrid EWC: char-LSTM + SSM multiscale ``alpha`` + GRIA per-token ``alpha``.
class HybridEwcRegularizer {
 public:
  void snapshot(const CharLSTMHead* lstm, const CellAISSM* ssm, const GRIALowRank* gria);

  void observe_grads(const HybridEwcGradStub& grads);

  double penalty(const CharLSTMHead* lstm, const CellAISSM* ssm, const GRIALowRank* gria) const;

  void apply_pull(CharLSTMHead* lstm, CellAISSM* ssm, GRIALowRank* gria, double ewc_lambda,
                  double lstm_lr, double gria_lr, double ssm_lr) const;

  bool has_snapshot() const {
    return lstm_.has_snapshot() || !anchor_gria_alpha_.empty() || !anchor_gria_U_.empty() ||
           !anchor_gria_bias_.empty() || !anchor_ssm_w_fast_.empty() ||
           !anchor_ssm_w_slow_.empty();
  }

  bool covers_embed_and_head() const { return lstm_.covers_embed_and_head(); }

  bool covers_ssm_alpha() const { return !anchor_ssm_alpha_.empty(); }

  bool covers_gria_alpha() const { return !anchor_gria_alpha_.empty(); }

  bool covers_gria_weights() const { return !anchor_gria_U_.empty() && !anchor_gria_V_.empty(); }

  bool covers_ssm_w_fast() const { return !anchor_ssm_w_fast_.empty(); }

  bool covers_gria_bias() const { return !anchor_gria_bias_.empty(); }

  bool covers_ssm_w_slow() const { return !anchor_ssm_w_slow_.empty(); }

  CyphaLMEwcRegularizer& lstm_part() { return lstm_; }
  const CyphaLMEwcRegularizer& lstm_part() const { return lstm_; }

  nlohmann::json get_state() const;
  void set_state(const nlohmann::json& state);

 private:
  CyphaLMEwcRegularizer lstm_;
  std::vector<double> anchor_ssm_alpha_;
  std::vector<double> anchor_gria_alpha_;
  std::vector<double> anchor_gria_U_;
  std::vector<double> anchor_gria_V_;
  std::vector<double> anchor_gria_bias_;
  std::vector<double> anchor_ssm_w_fast_;
  std::vector<double> anchor_ssm_w_slow_;
  std::vector<double> fisher_ssm_alpha_;
  std::vector<double> fisher_gria_alpha_;
  std::vector<double> fisher_gria_U_;
  std::vector<double> fisher_gria_V_;
  std::vector<double> fisher_gria_bias_;
  std::vector<double> fisher_ssm_w_fast_;
  std::vector<double> fisher_ssm_w_slow_;
  std::size_t ssm_grad_observations_{0};
  std::size_t gria_grad_observations_{0};
  std::size_t gria_uv_grad_observations_{0};
  std::size_t gria_bias_grad_observations_{0};
  std::size_t ssm_w_fast_grad_observations_{0};
  std::size_t ssm_w_slow_grad_observations_{0};
};

}  // namespace cypha::cyphalm
