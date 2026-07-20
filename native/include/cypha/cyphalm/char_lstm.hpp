#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "cypha/cyphalm/axiom_activation.hpp"
#include "cypha/cyphalm/sr_gate_laws.hpp"

namespace cypha {
namespace cyphalm {

enum class LSTMActivationMode {
    Standard,
    Eml,
    Axiom,
};

/// Wave-1 Quality §1.2 — optimizer for ``apply_grads`` (default SGD preserves D17 pin).
enum class LSTMOptim {
    Sgd,
    Adam,
};

/// Wave-1 Quality §1.5 — weight init (default N(0,0.02) preserves goldens).
enum class LSTMInitMode {
    Default,
    Classic,
};

struct CharLSTMGrad {
  std::vector<double> dE;
  std::vector<double> dWx;
  std::vector<double> dWh;
  std::vector<double> db;
  std::vector<double> dWy;
  std::vector<double> dby;
  std::vector<double> dh_prev;
  std::vector<double> dc_prev;
  /// Extra-layer grads for layers 1..L-1 (empty when ``n_layers==1``).
  std::vector<std::vector<double>> dWx_l;
  std::vector<std::vector<double>> dWh_l;
  std::vector<std::vector<double>> db_l;
};

/// Per-layer activations for stacked residual LSTM (``n_layers > 1``).
struct CharLSTMLayerCache {
  std::vector<double> x;       // layer input (token embed for L0; prev residual h for L>0)
  std::vector<double> h;       // prev recurrent h
  std::vector<double> c;
  std::vector<double> i;
  std::vector<double> f;
  std::vector<double> g;
  std::vector<double> o;
  std::vector<double> c_new;
  std::vector<double> h_lstm;  // pre-residual LSTM output
  std::vector<double> h_new;   // post-residual (L0: = h_lstm; L>0: = h_lstm + x)
  std::vector<double> tanh_c_new;
  std::vector<double> gates;
};

struct CharLSTMCache {
  int token_id{0};
  std::vector<double> x;
  std::vector<double> h;
  std::vector<double> c;
  std::vector<double> i;
  std::vector<double> f;
  std::vector<double> g;
  std::vector<double> o;
  std::vector<double> c_new;
  std::vector<double> h_new;
  // Perf (2026-07-12, part 2, docs/reports/PERFORMANCE_PROFILE_2026-07-12.md "Follow-up"):
  // tanh(c_new[j]) as computed by forward_step's h_out = o_gate * tanh(c_out). Cached here so
  // backward_step's non-eml/non-axiom-eml path doesn't call std::tanh a second time on the exact
  // same value it already produced during forward -- pure caching, zero arithmetic change (same
  // double value, bit-for-bit, since it's the identical std::tanh call on the identical input).
  std::vector<double> tanh_c_new;
  std::vector<double> gates;
  std::vector<double> logits;
  std::vector<double> probs;
  bool used_eml{false};
  bool used_axiom{false};
  bool used_sr_gates{false};
  /// Filled when ``n_layers > 1`` (one entry per layer). Empty on the single-layer pin path.
  std::vector<CharLSTMLayerCache> layers;
};

/// Char LSTM head with optional stacked residual depth.
///
/// ``n_layers == 1`` (default): historic single-layer path — D17 pin bit-identical.
/// ``n_layers >= 2``: Layer 0 is embed→LSTM; layers 1..L-1 take previous residual hidden as
/// input with ``h_out = h_lstm + h_in``. Readout ``Wy`` is applied **only to the top-layer
/// residual hidden** (not a sum of residuals).
///
/// Opt-in truncated BPTT / Adam / classic init via setters (Quality Wave 1; default OFF).
class CharLSTMHead {
 public:
  int vocab_size{256};
  int hidden{128};
  /// Stack depth (default 1 preserves D17 pin). Config: ``lstm_layers`` / ``--lstm-layers``.
  int n_layers{1};

  std::vector<double> E;    // vocab_size x hidden
  std::vector<double> Wx;   // layer-0 (4*hidden) x hidden
  std::vector<double> Wh;   // layer-0 (4*hidden) x hidden
  std::vector<double> b;    // layer-0 4*hidden
  std::vector<double> Wy;   // vocab_size x hidden  (top-layer readout)
  std::vector<double> by;   // vocab_size
  /// Layers 1..L-1 weights (size ``n_layers-1``; empty when ``n_layers==1``).
  std::vector<std::vector<double>> Wx_l;
  std::vector<std::vector<double>> Wh_l;
  std::vector<std::vector<double>> b_l;

  CharLSTMHead() = default;
  CharLSTMHead(int vocab_size_in, int hidden_in, std::uint64_t seed = 42,
               LSTMInitMode init_mode = LSTMInitMode::Default, int n_layers_in = 1);

  void set_activation_mode(LSTMActivationMode mode) { activation_mode_ = mode; }
  LSTMActivationMode activation_mode() const { return activation_mode_; }
  void set_axiom_grammar(const AxiomGateGrammar& grammar) { axiom_grammar_ = grammar; }

  /// H16: optional symbolic-regression gate pre-activation override.
  void set_use_sr_gates(bool enabled) { use_sr_gates_ = enabled; }
  bool use_sr_gates() const { return use_sr_gates_; }
  void set_sr_gate_laws(const SrGateLaws& laws) { sr_laws_ = laws; }
  const SrGateLaws& sr_gate_laws() const { return sr_laws_; }

  /// Truncated BPTT window (1 = historic BPTT-1). Env/CLI: ``CYPHA_LSTM_BPTT`` / ``--bptt-lstm``.
  void set_bptt_window(int steps);
  int bptt_window() const { return bptt_window_; }

  void set_optim(LSTMOptim optim);
  LSTMOptim optim() const { return optim_; }
  /// Global L2 grad clip; 0 = disabled (default).
  void set_grad_clip(double clip);
  double grad_clip() const { return grad_clip_; }
  /// AdamW decoupled weight decay; 0 = off (default). Applied only under Adam to E/Wx/Wh/Wy.
  void set_weight_decay(double wd);
  double weight_decay() const { return weight_decay_; }

  /// Reset internal h/c (stateful online API). Flushes any pending BPTT window without apply.
  void reset_state();

  /// Zero Adam moments / step so encode and decode start from identical optimizer state
  /// (checkpoints do not persist Adam; warm post-train moments break online_adapt roundtrips).
  void reset_optim_state();

  /// Stateful forward — updates internal h/c; returns log_probs.
  std::vector<double> forward(int token_id);

  /// Stateful backward with weight update (respects BPTT window / Adam / clip).
  /// Optional ``grads_out`` for EWC overlays (filled on flush).
  void backward(int target_id, double lr, CharLSTMGrad* grads_out = nullptr,
                double logit_nudge = 0.0, double hidden_nudge = 0.0);

  /// Push one cached step into the BPTT window; may flush (apply grads) when full.
  /// Returns true when a weight update occurred. Used by hybrid train path.
  bool push_bptt_step(const CharLSTMCache& cache, int target_id, double lr,
                      CharLSTMGrad* grads_out = nullptr, double logit_nudge = 0.0,
                      double hidden_nudge = 0.0);

  /// Force-flush a partial BPTT window (e.g. end of sequence). No-op if empty.
  void flush_bptt(double lr, CharLSTMGrad* grads_out = nullptr, double logit_nudge = 0.0,
                  double hidden_nudge = 0.0);

  void load_state(const std::vector<double>& E_in, const std::vector<double>& Wx_in,
                  const std::vector<double>& Wh_in, const std::vector<double>& b_in,
                  const std::vector<double>& Wy_in, const std::vector<double>& by_in);

  /// Load extra-layer weights (layers 1..L-1). No-op / ignored when ``n_layers==1``.
  void load_extra_layers(const std::vector<std::vector<double>>& Wx_l_in,
                         const std::vector<std::vector<double>>& Wh_l_in,
                         const std::vector<std::vector<double>>& b_l_in);

  /// External-state forward step (batch / parity).
  /// Layer-0 recurrent state is ``h``/``c``; layers 1..L-1 use internal upper-layer state.
  void forward_step(int token_id, const double* h, const double* c, double* log_probs, std::vector<double>& h_out,
                    std::vector<double>& c_out, CharLSTMCache* cache_out = nullptr,
                    double forget_gate_scale = 1.0) const;

  CharLSTMGrad backward_step(const CharLSTMCache& cache, int target_id,
                             double logit_nudge = 0.0, double hidden_nudge = 0.0) const;

  /// Out-param overload: fills `out` in place instead of returning by value. Callers that own a
  /// persistent CharLSTMGrad scratch buffer (e.g. the online per-step training hot path) can
  /// reuse it across steps and avoid re-allocating the (vocab_size*hidden + 2*(4*hidden)*hidden)
  /// gradient buffers on every call. Numerically identical to the value-returning overload, which
  /// now delegates to this one.
  ///
  /// Optional ``dh_next`` / ``dc_next`` (length ``hidden``) inject truncated-BPTT temporal grads
  /// from the future step into **layer 0** only; pass nullptr for BPTT-1 (default).
  void backward_step(const CharLSTMCache& cache, int target_id, CharLSTMGrad& out,
                     double logit_nudge = 0.0, double hidden_nudge = 0.0,
                     const double* dh_next = nullptr, const double* dc_next = nullptr) const;

  void apply_grads(const CharLSTMGrad& grads, double lr);

  double train_step(int token_id, int target_id, std::vector<double>& h, std::vector<double>& c, double lr);

 private:
  void init_weights(std::uint64_t seed, LSTMInitMode init_mode);
  void ensure_upper_state() const;
  void ensure_adam_state();
  void ensure_grad_scratch(CharLSTMGrad& g) const;
  void accumulate_grads(CharLSTMGrad& acc, const CharLSTMGrad& step) const;
  void clip_grads_inplace(CharLSTMGrad& grads) const;
  void flush_bptt_window(double lr, CharLSTMGrad* grads_out, double logit_nudge, double hidden_nudge);
  void forward_step_stacked(int token_id, const double* h, const double* c, double* log_probs,
                            std::vector<double>& h_out, std::vector<double>& c_out,
                            CharLSTMCache* cache_out, double forget_gate_scale) const;
  void backward_step_stacked(const CharLSTMCache& cache, int target_id, CharLSTMGrad& out,
                             double logit_nudge, double hidden_nudge, const double* dh_next,
                             const double* dc_next) const;

  std::vector<double> h_;
  std::vector<double> c_;
  /// Upper-layer recurrent state (layers 1..L-1). Mutable so const ``forward_step`` can advance it.
  mutable std::vector<std::vector<double>> h_upper_;
  mutable std::vector<std::vector<double>> c_upper_;
  CharLSTMCache cache_;
  bool has_cache_{false};
  LSTMActivationMode activation_mode_{LSTMActivationMode::Standard};
  AxiomGateGrammar axiom_grammar_;
  bool use_sr_gates_{false};
  SrGateLaws sr_laws_;

  int bptt_window_{1};
  LSTMOptim optim_{LSTMOptim::Sgd};
  double grad_clip_{0.0};
  double weight_decay_{0.0};
  std::deque<CharLSTMCache> bptt_caches_;
  std::deque<int> bptt_targets_;

  // Adam moments (allocated lazily when optim_ == Adam).
  std::vector<double> m_E_, v_E_;
  std::vector<double> m_Wx_, v_Wx_;
  std::vector<double> m_Wh_, v_Wh_;
  std::vector<double> m_b_, v_b_;
  std::vector<double> m_Wy_, v_Wy_;
  std::vector<double> m_by_, v_by_;
  std::vector<std::vector<double>> m_Wx_l_, v_Wx_l_;
  std::vector<std::vector<double>> m_Wh_l_, v_Wh_l_;
  std::vector<std::vector<double>> m_b_l_, v_b_l_;
  std::int64_t adam_t_{0};
};

using CharLSTM = CharLSTMHead;

LSTMOptim parse_lstm_optim(const std::string& s);
LSTMInitMode parse_lstm_init_mode(const std::string& s);
std::string lstm_optim_name(LSTMOptim o);
std::string lstm_init_mode_name(LSTMInitMode m);

/// Vector blend helper for ``CyphaLMNative`` (returns sigmoid(blend_logit)).
double blend_log_probs(const std::vector<double>& log_g, const std::vector<double>& log_l, double blend_logit,
                       std::vector<double>& out);

}  // namespace cyphalm
}  // namespace cypha
