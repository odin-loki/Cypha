#pragma once

#include <cstdint>
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

struct CharLSTMGrad {
  std::vector<double> dE;
  std::vector<double> dWx;
  std::vector<double> dWh;
  std::vector<double> db;
  std::vector<double> dWy;
  std::vector<double> dby;
  std::vector<double> dh_prev;
  std::vector<double> dc_prev;
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
  std::vector<double> gates;
  std::vector<double> logits;
  std::vector<double> probs;
  bool used_eml{false};
  bool used_axiom{false};
  bool used_sr_gates{false};
};

/// Single-layer char LSTM head (online BPTT-1). Weight layout matches Python ``CharLSTMHead``.
class CharLSTMHead {
 public:
  int vocab_size{256};
  int hidden{128};

  std::vector<double> E;    // vocab_size x hidden
  std::vector<double> Wx;   // (4*hidden) x hidden
  std::vector<double> Wh;   // (4*hidden) x hidden
  std::vector<double> b;    // 4*hidden
  std::vector<double> Wy;   // vocab_size x hidden
  std::vector<double> by;   // vocab_size

  CharLSTMHead() = default;
  CharLSTMHead(int vocab_size_in, int hidden_in, std::uint64_t seed = 42);

  void set_activation_mode(LSTMActivationMode mode) { activation_mode_ = mode; }
  LSTMActivationMode activation_mode() const { return activation_mode_; }
  void set_axiom_grammar(const AxiomGateGrammar& grammar) { axiom_grammar_ = grammar; }

  /// H16: optional symbolic-regression gate pre-activation override.
  void set_use_sr_gates(bool enabled) { use_sr_gates_ = enabled; }
  bool use_sr_gates() const { return use_sr_gates_; }
  void set_sr_gate_laws(const SrGateLaws& laws) { sr_laws_ = laws; }
  const SrGateLaws& sr_gate_laws() const { return sr_laws_; }

  /// Reset internal h/c (stateful online API).
  void reset_state();

  /// Stateful forward — updates internal h/c; returns log_probs.
  std::vector<double> forward(int token_id);

  /// Stateful backward (BPTT-1) with weight update. Optional ``grads_out`` for EWC overlays.
  void backward(int target_id, double lr, CharLSTMGrad* grads_out = nullptr,
                double logit_nudge = 0.0, double hidden_nudge = 0.0);

  void load_state(const std::vector<double>& E_in, const std::vector<double>& Wx_in,
                  const std::vector<double>& Wh_in, const std::vector<double>& b_in,
                  const std::vector<double>& Wy_in, const std::vector<double>& by_in);

  /// External-state forward step (batch / parity).
  void forward_step(int token_id, const double* h, const double* c, double* log_probs, std::vector<double>& h_out,
                    std::vector<double>& c_out, CharLSTMCache* cache_out = nullptr,
                    double forget_gate_scale = 1.0) const;

  CharLSTMGrad backward_step(const CharLSTMCache& cache, int target_id,
                             double logit_nudge = 0.0, double hidden_nudge = 0.0) const;

  void apply_grads(const CharLSTMGrad& grads, double lr);

  double train_step(int token_id, int target_id, std::vector<double>& h, std::vector<double>& c, double lr);

 private:
  std::vector<double> h_;
  std::vector<double> c_;
  CharLSTMCache cache_;
  bool has_cache_{false};
  LSTMActivationMode activation_mode_{LSTMActivationMode::Standard};
  AxiomGateGrammar axiom_grammar_;
  bool use_sr_gates_{false};
  SrGateLaws sr_laws_;
};

using CharLSTM = CharLSTMHead;

/// Vector blend helper for ``CyphaLMNative`` (returns sigmoid(blend_logit)).
double blend_log_probs(const std::vector<double>& log_g, const std::vector<double>& log_l, double blend_logit,
                       std::vector<double>& out);

}  // namespace cyphalm
}  // namespace cypha
