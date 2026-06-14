#include "cypha/cyphalm/char_lstm.hpp"

#include "cypha/cyphalm/axiom_activation.hpp"
#include "cypha/cyphalm/eml_activation.hpp"
#include "cypha/cyphalm/hybrid_blend.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace cypha {
namespace cyphalm {

namespace {

constexpr double kLogEps = 1e-12;

void matvec_rowmajor(const double* M, int rows, int cols, const double* x, double* y) {
  for (int r = 0; r < rows; ++r) {
    double s = 0.0;
    const double* row = M + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    for (int c = 0; c < cols; ++c) {
      s += row[c] * x[c];
    }
    y[r] = s;
  }
}

void outer_rowmajor(const double* a, int na, const double* b, int nb, double* out) {
  for (int i = 0; i < na; ++i) {
    for (int j = 0; j < nb; ++j) {
      out[static_cast<std::size_t>(i) * static_cast<std::size_t>(nb) + static_cast<std::size_t>(j)] =
          a[i] * b[j];
    }
  }
}

void softmax_logits(const double* logits, int n, double* probs) {
  double mx = logits[0];
  for (int i = 1; i < n; ++i) {
    mx = std::max(mx, logits[i]);
  }
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    probs[i] = std::exp(logits[i] - mx);
    sum += probs[i];
  }
  for (int i = 0; i < n; ++i) {
    probs[i] /= sum;
  }
}

}  // namespace

CharLSTMHead::CharLSTMHead(int vocab_size_in, int hidden_in, std::uint64_t seed) {
  vocab_size = vocab_size_in;
  hidden = hidden_in;
  const int four_h = 4 * hidden;
  E.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  Wx.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  Wh.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  b.assign(static_cast<std::size_t>(four_h), 0.0);
  Wy.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  by.assign(static_cast<std::size_t>(vocab_size), 0.0);

  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  constexpr double kScale = 0.02;
  for (auto& v : E) v = nd(rng) * kScale;
  for (auto& v : Wx) v = nd(rng) * kScale;
  for (auto& v : Wh) v = nd(rng) * kScale;
  for (auto& v : Wy) v = nd(rng) * kScale;
  reset_state();
}

void CharLSTMHead::reset_state() {
  h_.assign(static_cast<std::size_t>(hidden), 0.0);
  c_.assign(static_cast<std::size_t>(hidden), 0.0);
  has_cache_ = false;
}

void CharLSTMHead::forward_step(int token_id, const double* h, const double* c, double* log_probs,
                                std::vector<double>& h_out, std::vector<double>& c_out,
                                CharLSTMCache* cache_out, double forget_gate_scale) const {
  if (token_id < 0 || token_id >= vocab_size) {
    throw std::runtime_error("char_lstm: token_id out of range");
  }
  const int four_h = 4 * hidden;
  const double* x = E.data() + static_cast<std::size_t>(token_id) * static_cast<std::size_t>(hidden);

  std::vector<double> gates(static_cast<std::size_t>(four_h));
  matvec_rowmajor(Wx.data(), four_h, hidden, x, gates.data());
  std::vector<double> wh(static_cast<std::size_t>(four_h));
  matvec_rowmajor(Wh.data(), four_h, hidden, h, wh.data());
  for (int i = 0; i < four_h; ++i) {
    gates[static_cast<std::size_t>(i)] += wh[static_cast<std::size_t>(i)] + b[static_cast<std::size_t>(i)];
  }

  const bool use_sr =
      use_sr_gates_ && sr_laws_.fitted && sr_laws_.hidden == hidden &&
      static_cast<int>(sr_laws_.f_gate.size()) == hidden;
  if (use_sr) {
    for (int j = 0; j < hidden; ++j) {
      gates[static_cast<std::size_t>(j)] =
          sr_laws_.i_gate[static_cast<std::size_t>(j)].predict(h[j], x[j], h[j]);
      gates[static_cast<std::size_t>(hidden + j)] =
          sr_laws_.f_gate[static_cast<std::size_t>(j)].predict(h[j], x[j], c[j]);
      gates[static_cast<std::size_t>(2 * hidden + j)] =
          sr_laws_.g_gate[static_cast<std::size_t>(j)].predict(h[j], x[j], h[j]);
      gates[static_cast<std::size_t>(3 * hidden + j)] =
          sr_laws_.o_gate[static_cast<std::size_t>(j)].predict(h[j], x[j], c[j]);
    }
  }

  std::vector<double> i_gate(static_cast<std::size_t>(hidden));
  std::vector<double> f_gate(static_cast<std::size_t>(hidden));
  std::vector<double> g_gate(static_cast<std::size_t>(hidden));
  std::vector<double> o_gate(static_cast<std::size_t>(hidden));
  const bool use_eml = activation_mode_ == LSTMActivationMode::Eml;
  const bool use_axiom = activation_mode_ == LSTMActivationMode::Axiom;
  for (int j = 0; j < hidden; ++j) {
    if (use_axiom && static_cast<int>(axiom_grammar_.i_gate.size()) == hidden) {
      i_gate[static_cast<std::size_t>(j)] =
          apply_axiom_gate(axiom_grammar_.i_gate[static_cast<std::size_t>(j)],
                           gates[static_cast<std::size_t>(j)], h[j]);
      f_gate[static_cast<std::size_t>(j)] =
          apply_axiom_gate(axiom_grammar_.f_gate[static_cast<std::size_t>(j)],
                           gates[static_cast<std::size_t>(hidden + j)], c[j]);
      g_gate[static_cast<std::size_t>(j)] =
          apply_axiom_gate(axiom_grammar_.g_gate[static_cast<std::size_t>(j)],
                           gates[static_cast<std::size_t>(2 * hidden + j)], h[j]);
      o_gate[static_cast<std::size_t>(j)] =
          apply_axiom_gate(axiom_grammar_.o_gate[static_cast<std::size_t>(j)],
                           gates[static_cast<std::size_t>(3 * hidden + j)], c[j]);
    } else if (use_eml) {
      i_gate[static_cast<std::size_t>(j)] =
          eml_nand(gates[static_cast<std::size_t>(j)], h[j]);
      f_gate[static_cast<std::size_t>(j)] =
          eml_nand(gates[static_cast<std::size_t>(hidden + j)], c[j]);
      g_gate[static_cast<std::size_t>(j)] =
          eml_nand(gates[static_cast<std::size_t>(2 * hidden + j)], h[j]);
      o_gate[static_cast<std::size_t>(j)] =
          eml_nand(gates[static_cast<std::size_t>(3 * hidden + j)], c[j]);
    } else {
      i_gate[static_cast<std::size_t>(j)] = sigmoid(gates[static_cast<std::size_t>(j)]);
      f_gate[static_cast<std::size_t>(j)] = sigmoid(gates[static_cast<std::size_t>(hidden + j)]);
      g_gate[static_cast<std::size_t>(j)] = std::tanh(gates[static_cast<std::size_t>(2 * hidden + j)]);
      o_gate[static_cast<std::size_t>(j)] = sigmoid(gates[static_cast<std::size_t>(3 * hidden + j)]);
    }
  }
  if (forget_gate_scale != 1.0) {
    for (int j = 0; j < hidden; ++j) {
      f_gate[static_cast<std::size_t>(j)] *= forget_gate_scale;
    }
  }

  c_out.assign(static_cast<std::size_t>(hidden), 0.0);
  h_out.assign(static_cast<std::size_t>(hidden), 0.0);
  for (int j = 0; j < hidden; ++j) {
    c_out[static_cast<std::size_t>(j)] =
        f_gate[static_cast<std::size_t>(j)] * c[j] + i_gate[static_cast<std::size_t>(j)] * g_gate[static_cast<std::size_t>(j)];
    if (use_eml) {
      h_out[static_cast<std::size_t>(j)] =
          o_gate[static_cast<std::size_t>(j)] * eml_nand(c_out[static_cast<std::size_t>(j)], 1.0);
    } else if (use_axiom) {
      h_out[static_cast<std::size_t>(j)] =
          o_gate[static_cast<std::size_t>(j)] * std::tanh(c_out[static_cast<std::size_t>(j)]);
    } else {
      h_out[static_cast<std::size_t>(j)] = o_gate[static_cast<std::size_t>(j)] * std::tanh(c_out[static_cast<std::size_t>(j)]);
    }
  }

  std::vector<double> logits(static_cast<std::size_t>(vocab_size));
  matvec_rowmajor(Wy.data(), vocab_size, hidden, h_out.data(), logits.data());
  for (int k = 0; k < vocab_size; ++k) {
    logits[static_cast<std::size_t>(k)] += by[static_cast<std::size_t>(k)];
  }

  std::vector<double> probs(static_cast<std::size_t>(vocab_size));
  softmax_logits(logits.data(), vocab_size, probs.data());
  for (int k = 0; k < vocab_size; ++k) {
    log_probs[k] = std::log(probs[static_cast<std::size_t>(k)] + kLogEps);
  }

  if (cache_out != nullptr) {
    cache_out->token_id = token_id;
    cache_out->x.assign(x, x + hidden);
    cache_out->h.assign(h, h + hidden);
    cache_out->c.assign(c, c + hidden);
    cache_out->i = i_gate;
    cache_out->f = f_gate;
    cache_out->g = g_gate;
    cache_out->o = o_gate;
    cache_out->c_new = c_out;
    cache_out->h_new = h_out;
    cache_out->gates = gates;
    cache_out->logits = logits;
    cache_out->probs = probs;
    cache_out->used_eml = use_eml;
    cache_out->used_axiom = use_axiom;
    cache_out->used_sr_gates = use_sr;
  }
}

CharLSTMGrad CharLSTMHead::backward_step(const CharLSTMCache& cache, int target_id) const {
  if (target_id < 0 || target_id >= vocab_size) {
    throw std::runtime_error("char_lstm: target_id out of range");
  }
  const int four_h = 4 * hidden;
  CharLSTMGrad out;
  out.dE.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  out.dWx.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  out.dWh.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  out.db.assign(static_cast<std::size_t>(four_h), 0.0);
  out.dWy.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  out.dby.assign(static_cast<std::size_t>(vocab_size), 0.0);
  out.dh_prev.assign(static_cast<std::size_t>(hidden), 0.0);
  out.dc_prev.assign(static_cast<std::size_t>(hidden), 0.0);

  std::vector<double> d_logits = cache.probs;
  d_logits[static_cast<std::size_t>(target_id)] -= 1.0;

  outer_rowmajor(d_logits.data(), vocab_size, cache.h_new.data(), hidden, out.dWy.data());
  out.dby = d_logits;

  std::vector<double> dh_new(static_cast<std::size_t>(hidden));
  for (int j = 0; j < hidden; ++j) {
    double s = 0.0;
    for (int k = 0; k < vocab_size; ++k) {
      s += Wy[static_cast<std::size_t>(k) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(j)] *
           d_logits[static_cast<std::size_t>(k)];
    }
    dh_new[static_cast<std::size_t>(j)] = s;
  }

  std::vector<double> do_gate(static_cast<std::size_t>(hidden));
  std::vector<double> dc_new(static_cast<std::size_t>(hidden));
  if (cache.used_eml) {
    for (int j = 0; j < hidden; ++j) {
      const double h_act = eml_nand(cache.c_new[static_cast<std::size_t>(j)], 1.0);
      do_gate[static_cast<std::size_t>(j)] = dh_new[static_cast<std::size_t>(j)] * h_act;
      double deml_dc = 0.0;
      double deml_const = 0.0;
      eml_nand_grad(cache.c_new[static_cast<std::size_t>(j)], 1.0,
                    dh_new[static_cast<std::size_t>(j)] * cache.o[static_cast<std::size_t>(j)],
                    deml_dc, deml_const);
      dc_new[static_cast<std::size_t>(j)] = deml_dc;
    }
  } else {
    for (int j = 0; j < hidden; ++j) {
      const double tanh_c = std::tanh(cache.c_new[static_cast<std::size_t>(j)]);
      do_gate[static_cast<std::size_t>(j)] = dh_new[static_cast<std::size_t>(j)] * tanh_c;
      dc_new[static_cast<std::size_t>(j)] =
          dh_new[static_cast<std::size_t>(j)] * cache.o[static_cast<std::size_t>(j)] * (1.0 - tanh_c * tanh_c);
    }
  }

  std::vector<double> df_gate(static_cast<std::size_t>(hidden));
  std::vector<double> di_gate(static_cast<std::size_t>(hidden));
  std::vector<double> dg_gate(static_cast<std::size_t>(hidden));
  for (int j = 0; j < hidden; ++j) {
    df_gate[static_cast<std::size_t>(j)] = dc_new[static_cast<std::size_t>(j)] * cache.c[static_cast<std::size_t>(j)];
    di_gate[static_cast<std::size_t>(j)] = dc_new[static_cast<std::size_t>(j)] * cache.g[static_cast<std::size_t>(j)];
    dg_gate[static_cast<std::size_t>(j)] = dc_new[static_cast<std::size_t>(j)] * cache.i[static_cast<std::size_t>(j)];
    out.dc_prev[static_cast<std::size_t>(j)] = dc_new[static_cast<std::size_t>(j)] * cache.f[static_cast<std::size_t>(j)];
  }

  std::vector<double> dgates(static_cast<std::size_t>(four_h));
  for (int j = 0; j < hidden; ++j) {
    if (cache.used_eml) {
      double di_gx = 0.0;
      double di_hy = 0.0;
      eml_nand_grad(cache.gates[static_cast<std::size_t>(j)], cache.h[static_cast<std::size_t>(j)],
                    di_gate[static_cast<std::size_t>(j)], di_gx, di_hy);
      dgates[static_cast<std::size_t>(j)] = di_gx;

      double df_gx = 0.0;
      double df_cy = 0.0;
      eml_nand_grad(cache.gates[static_cast<std::size_t>(hidden + j)], cache.c[static_cast<std::size_t>(j)],
                    df_gate[static_cast<std::size_t>(j)], df_gx, df_cy);
      dgates[static_cast<std::size_t>(hidden + j)] = df_gx;
      out.dc_prev[static_cast<std::size_t>(j)] += df_cy;

      double dg_gx = 0.0;
      double dg_hy = 0.0;
      eml_nand_grad(cache.gates[static_cast<std::size_t>(2 * hidden + j)], cache.h[static_cast<std::size_t>(j)],
                    dg_gate[static_cast<std::size_t>(j)], dg_gx, dg_hy);
      dgates[static_cast<std::size_t>(2 * hidden + j)] = dg_gx;

      double do_gx = 0.0;
      double do_cy = 0.0;
      eml_nand_grad(cache.gates[static_cast<std::size_t>(3 * hidden + j)], cache.c[static_cast<std::size_t>(j)],
                    do_gate[static_cast<std::size_t>(j)], do_gx, do_cy);
      dgates[static_cast<std::size_t>(3 * hidden + j)] = do_gx;
      out.dc_prev[static_cast<std::size_t>(j)] += do_cy;
    } else {
      dgates[static_cast<std::size_t>(j)] = di_gate[static_cast<std::size_t>(j)] * cache.i[static_cast<std::size_t>(j)] *
                                            (1.0 - cache.i[static_cast<std::size_t>(j)]);
      dgates[static_cast<std::size_t>(hidden + j)] =
          df_gate[static_cast<std::size_t>(j)] * cache.f[static_cast<std::size_t>(j)] *
          (1.0 - cache.f[static_cast<std::size_t>(j)]);
      dgates[static_cast<std::size_t>(2 * hidden + j)] =
          dg_gate[static_cast<std::size_t>(j)] * (1.0 - cache.g[static_cast<std::size_t>(j)] * cache.g[static_cast<std::size_t>(j)]);
      dgates[static_cast<std::size_t>(3 * hidden + j)] =
          do_gate[static_cast<std::size_t>(j)] * cache.o[static_cast<std::size_t>(j)] *
          (1.0 - cache.o[static_cast<std::size_t>(j)]);
    }
  }

  outer_rowmajor(dgates.data(), four_h, cache.x.data(), hidden, out.dWx.data());
  outer_rowmajor(dgates.data(), four_h, cache.h.data(), hidden, out.dWh.data());
  out.db = dgates;

  std::vector<double> dx(static_cast<std::size_t>(hidden));
  for (int j = 0; j < hidden; ++j) {
    double s = 0.0;
    for (int r = 0; r < four_h; ++r) {
      s += Wx[static_cast<std::size_t>(r) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(j)] *
           dgates[static_cast<std::size_t>(r)];
    }
    dx[static_cast<std::size_t>(j)] = s;
  }
  for (int j = 0; j < hidden; ++j) {
    out.dE[static_cast<std::size_t>(cache.token_id) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(j)] =
        dx[static_cast<std::size_t>(j)];
  }

  for (int j = 0; j < hidden; ++j) {
    double s = 0.0;
    for (int r = 0; r < four_h; ++r) {
      s += Wh[static_cast<std::size_t>(r) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(j)] *
           dgates[static_cast<std::size_t>(r)];
    }
    out.dh_prev[static_cast<std::size_t>(j)] = s;
  }

  return out;
}

void CharLSTMHead::apply_grads(const CharLSTMGrad& grads, double lr) {
  for (std::size_t i = 0; i < E.size(); ++i) E[i] -= lr * grads.dE[i];
  for (std::size_t i = 0; i < Wx.size(); ++i) Wx[i] -= lr * grads.dWx[i];
  for (std::size_t i = 0; i < Wh.size(); ++i) Wh[i] -= lr * grads.dWh[i];
  for (std::size_t i = 0; i < b.size(); ++i) b[i] -= lr * grads.db[i];
  for (std::size_t i = 0; i < Wy.size(); ++i) Wy[i] -= lr * grads.dWy[i];
  for (std::size_t i = 0; i < by.size(); ++i) by[i] -= lr * grads.dby[i];
}

double CharLSTMHead::train_step(int token_id, int target_id, std::vector<double>& h, std::vector<double>& c,
                                double lr) {
  std::vector<double> log_probs(static_cast<std::size_t>(vocab_size));
  std::vector<double> h_new;
  std::vector<double> c_new;
  CharLSTMCache cache;
  forward_step(token_id, h.data(), c.data(), log_probs.data(), h_new, c_new, &cache);
  const double loss = -log_probs[static_cast<std::size_t>(target_id)];
  CharLSTMGrad grads = backward_step(cache, target_id);
  apply_grads(grads, lr);
  h = std::move(h_new);
  c = std::move(c_new);
  return loss;
}

std::vector<double> CharLSTMHead::forward(int token_id) {
  std::vector<double> log_probs(static_cast<std::size_t>(vocab_size));
  std::vector<double> h_new;
  std::vector<double> c_new;
  forward_step(token_id, h_.data(), c_.data(), log_probs.data(), h_new, c_new, &cache_);
  h_ = std::move(h_new);
  c_ = std::move(c_new);
  has_cache_ = true;
  return log_probs;
}

void CharLSTMHead::backward(int target_id, double lr, CharLSTMGrad* grads_out) {
  if (!has_cache_) {
    return;
  }
  CharLSTMGrad grads = backward_step(cache_, target_id);
  if (grads_out != nullptr) {
    *grads_out = grads;
  }
  apply_grads(grads, lr);
  has_cache_ = false;
}

void CharLSTMHead::load_state(const std::vector<double>& E_in, const std::vector<double>& Wx_in,
                              const std::vector<double>& Wh_in, const std::vector<double>& b_in,
                              const std::vector<double>& Wy_in, const std::vector<double>& by_in) {
  E = E_in;
  Wx = Wx_in;
  Wh = Wh_in;
  b = b_in;
  Wy = Wy_in;
  by = by_in;
  reset_state();
}

double blend_log_probs(const std::vector<double>& log_g, const std::vector<double>& log_l, double blend_logit,
                       std::vector<double>& out) {
  const int n = static_cast<int>(log_g.size());
  out.resize(static_cast<std::size_t>(n));
  blend_log_probs(log_g.data(), log_l.data(), n, blend_logit, out.data());
  return sigmoid(blend_logit);
}

}  // namespace cyphalm
}  // namespace cypha
