#include "cypha/cyphalm/char_lstm.hpp"

#include "cypha/cyphalm/axiom_activation.hpp"
#include "cypha/cyphalm/eml_activation.hpp"
#include "cypha/cyphalm/hybrid_blend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>

namespace cypha {
namespace cyphalm {

namespace {

constexpr double kLogEps = 1e-12;

// CYPHA_PERF_TRACE follow-up (2026-07-12, part 2, docs/reports/PERFORMANCE_PROFILE_2026-07-12.md
// "Follow-up" section): fine-grained sub-phase breakdown *inside* backward_step specifically,
// since the top-level trace in cyphalm_model.cpp already showed `lstm_backward` at 45.3% of
// train_step time without saying which part of it dominates. Same env var (`CYPHA_PERF_TRACE`)
// as the top-level trace so one env var enables both dumps; purely diagnostic (stderr summary at
// process exit), never touches training math/state, so it cannot affect the D17 BPC pin whether
// enabled or not.
//
// Guarded by a mutex (not thread_local-merge-on-destruction) because, unlike the top-level
// train_step trace, `backward_step` is also called concurrently on a *shared* CharLSTMHead
// instance from CyphaLMBatch's parallel_batch (multiple worker threads, see the thread_local
// buffer comments above) -- a plain global accumulator would race under that path. The mutex is
// only ever touched when CYPHA_PERF_TRACE is set (diagnostic opt-in, zero cost on the default
// path), and even when enabled it's a single uncontended lock per sub-phase in the intended
// `--threads 1` profiling scenario, negligible next to the microsecond-scale regions being timed.
struct BackwardSubPhaseTrace {
  static constexpr std::size_t kCount = 6;
  static constexpr std::array<const char*, kCount> kNames = {
      "1. output-layer backward (dWy/dby/dh_new)",
      "2. activation gradient (do_gate/dc_new + df/di/dg/dc_prev base)",
      "3. gate derivative dispatch (dgates: per-gate eml/axiom/sigmoid loop)",
      "4. weight-gradient outer products (dWx/dWh/db)",
      "5. input-gradient backprop (dx/dE)",
      "6. hidden-gradient backprop (dh_prev)",
  };
  bool enabled = false;
  std::mutex mu;
  long long calls = 0;
  std::array<double, kCount> totals{};

  BackwardSubPhaseTrace() { enabled = std::getenv("CYPHA_PERF_TRACE") != nullptr; }
  ~BackwardSubPhaseTrace() {
    if (!enabled || calls == 0) return;
    double total = 0.0;
    for (double t : totals) total += t;
    std::cerr << "=== CYPHA_PERF_TRACE: lstm_backward (CharLSTMHead::backward_step) sub-phase"
                 " breakdown over " << calls << " calls (" << total << "s instrumented) ===\n";
    for (std::size_t i = 0; i < kCount; ++i) {
      const double pct = total > 0.0 ? (100.0 * totals[i] / total) : 0.0;
      std::cerr << "  " << kNames[i] << ": " << totals[i] << "s (" << pct << "%)\n";
    }
  }
  void add(std::size_t idx, double seconds) {
    std::lock_guard<std::mutex> lock(mu);
    totals[idx] += seconds;
  }
  void note_call() {
    std::lock_guard<std::mutex> lock(mu);
    ++calls;
  }
};
BackwardSubPhaseTrace g_bwd_trace;

class BwdScopeTimer {
 public:
  explicit BwdScopeTimer(std::size_t idx) : idx_(idx), enabled_(g_bwd_trace.enabled) {
    if (enabled_) t0_ = std::chrono::steady_clock::now();
  }
  ~BwdScopeTimer() {
    if (enabled_) {
      g_bwd_trace.add(idx_,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count());
    }
  }

 private:
  std::size_t idx_;
  bool enabled_;
  std::chrono::steady_clock::time_point t0_;
};

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

// Perf (2026-07-12, part 2, docs/reports/PERFORMANCE_PROFILE_2026-07-12.md "Follow-up"):
// computes `y = M^T * x` for row-major `M` (`rows x cols`) by iterating `r` outer / `c` inner
// instead of the natural-looking `c` outer / `r` inner. The naive form reads `M[r*cols + c]`
// with stride `cols` doubles as `r` varies in the inner loop -- a cache-hostile strided scan
// across a `4*hidden x hidden` (or `vocab_size x hidden`) matrix on every single backward step.
// This form reads each row of `M` sequentially instead. It is a pure loop interchange, not a
// summation-order change: for each fixed `c`, the additions into `y[c]` still occur in strictly
// increasing `r` order -- identical to `for (c) { s=0; for (r) s += M[r*cols+c]*x[r]; y[c]=s; }`,
// just visiting `r` before `c`. Confirmed via CYPHA_PERF_TRACE sub-phase timing (this exact
// pattern against Wy/Wx/Wh accounted for ~18%+22%+29%+29% = the majority of
// CharLSTMHead::backward_step's own time before this fix) that this was the single largest
// remaining `lstm_backward` sub-phase cost, once the dafd677 allocation fix removed the
// allocator overhead that had previously been dominating and masking it.
void matvec_transpose_rowmajor(const double* M, int rows, int cols, const double* x, double* y) {
  for (int c = 0; c < cols; ++c) y[c] = 0.0;
  for (int r = 0; r < rows; ++r) {
    const double xr = x[r];
    const double* row = M + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    for (int c = 0; c < cols; ++c) {
      y[c] += row[c] * xr;
    }
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

namespace {

void fill_orthogonal_block(std::vector<double>& Wh, int hidden, int block, std::mt19937_64& rng) {
  // QR of Gaussian h×h → orthogonal Q written into Wh block rows [block*h, (block+1)*h).
  const int h = hidden;
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> A(static_cast<std::size_t>(h) * static_cast<std::size_t>(h));
  for (auto& v : A) v = nd(rng);
  // Modified Gram-Schmidt on columns.
  std::vector<double> Q(A.size(), 0.0);
  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < h; ++i) {
      Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(j)] =
          A[static_cast<std::size_t>(i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(j)];
    }
    for (int k = 0; k < j; ++k) {
      double dot = 0.0;
      for (int i = 0; i < h; ++i) {
        dot += Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(j)] *
               Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(k)];
      }
      for (int i = 0; i < h; ++i) {
        Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(j)] -=
            dot * Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(k)];
      }
    }
    double norm = 0.0;
    for (int i = 0; i < h; ++i) {
      const double v =
          Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(j)];
      norm += v * v;
    }
    norm = std::sqrt(std::max(norm, 1e-30));
    for (int i = 0; i < h; ++i) {
      Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(j)] /= norm;
    }
  }
  const int row0 = block * h;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < h; ++j) {
      Wh[static_cast<std::size_t>(row0 + i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(j)] =
          Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(h) + static_cast<std::size_t>(j)];
    }
  }
}

void adam_update(std::vector<double>& w, std::vector<double>& m, std::vector<double>& v,
                 const std::vector<double>& g, double lr, std::int64_t t, double beta1, double beta2,
                 double eps) {
  const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(t));
  const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(t));
  for (std::size_t i = 0; i < w.size(); ++i) {
    m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
    v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
    const double mhat = m[i] / bc1;
    const double vhat = v[i] / bc2;
    w[i] -= lr * mhat / (std::sqrt(vhat) + eps);
  }
}

}  // namespace

CharLSTMHead::CharLSTMHead(int vocab_size_in, int hidden_in, std::uint64_t seed, LSTMInitMode init_mode) {
  vocab_size = vocab_size_in;
  hidden = hidden_in;
  init_weights(seed, init_mode);
  reset_state();
}

void CharLSTMHead::init_weights(std::uint64_t seed, LSTMInitMode init_mode) {
  const int four_h = 4 * hidden;
  E.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  Wx.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  Wh.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  b.assign(static_cast<std::size_t>(four_h), 0.0);
  Wy.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  by.assign(static_cast<std::size_t>(vocab_size), 0.0);

  std::mt19937_64 rng(seed);
  if (init_mode == LSTMInitMode::Classic) {
    std::normal_distribution<double> nd(0.0, 1.0);
    constexpr double kScale = 0.02;
    for (auto& v : E) v = nd(rng) * kScale;
    for (auto& v : Wx) v = nd(rng) * kScale;
    for (int block = 0; block < 4; ++block) {
      fill_orthogonal_block(Wh, hidden, block, rng);
    }
    // Forget-gate bias +1 (f gate occupies [hidden, 2*hidden)).
    for (int j = hidden; j < 2 * hidden; ++j) {
      b[static_cast<std::size_t>(j)] = 1.0;
    }
    const double wy_bound =
        std::sqrt(6.0 / static_cast<double>(std::max(1, hidden + vocab_size)));
    std::uniform_real_distribution<double> ud(-wy_bound, wy_bound);
    for (auto& v : Wy) v = ud(rng);
  } else {
    std::normal_distribution<double> nd(0.0, 1.0);
    constexpr double kScale = 0.02;
    for (auto& v : E) v = nd(rng) * kScale;
    for (auto& v : Wx) v = nd(rng) * kScale;
    for (auto& v : Wh) v = nd(rng) * kScale;
    for (auto& v : Wy) v = nd(rng) * kScale;
  }
}

void CharLSTMHead::set_bptt_window(int steps) {
  bptt_window_ = std::max(1, steps);
}

void CharLSTMHead::set_optim(LSTMOptim optim) {
  optim_ = optim;
  if (optim_ == LSTMOptim::Adam) {
    ensure_adam_state();
  }
}

void CharLSTMHead::set_grad_clip(double clip) {
  grad_clip_ = std::max(0.0, clip);
}

void CharLSTMHead::set_weight_decay(double wd) {
  weight_decay_ = std::max(0.0, wd);
}

void CharLSTMHead::ensure_adam_state() {
  if (m_E_.size() == E.size()) return;
  m_E_.assign(E.size(), 0.0);
  v_E_.assign(E.size(), 0.0);
  m_Wx_.assign(Wx.size(), 0.0);
  v_Wx_.assign(Wx.size(), 0.0);
  m_Wh_.assign(Wh.size(), 0.0);
  v_Wh_.assign(Wh.size(), 0.0);
  m_b_.assign(b.size(), 0.0);
  v_b_.assign(b.size(), 0.0);
  m_Wy_.assign(Wy.size(), 0.0);
  v_Wy_.assign(Wy.size(), 0.0);
  m_by_.assign(by.size(), 0.0);
  v_by_.assign(by.size(), 0.0);
  adam_t_ = 0;
}

void CharLSTMHead::ensure_grad_scratch(CharLSTMGrad& g) const {
  const int four_h = 4 * hidden;
  g.dE.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  g.dWx.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  g.dWh.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  g.db.assign(static_cast<std::size_t>(four_h), 0.0);
  g.dWy.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  g.dby.assign(static_cast<std::size_t>(vocab_size), 0.0);
  g.dh_prev.assign(static_cast<std::size_t>(hidden), 0.0);
  g.dc_prev.assign(static_cast<std::size_t>(hidden), 0.0);
}

void CharLSTMHead::accumulate_grads(CharLSTMGrad& acc, const CharLSTMGrad& step) const {
  for (std::size_t i = 0; i < acc.dE.size(); ++i) acc.dE[i] += step.dE[i];
  for (std::size_t i = 0; i < acc.dWx.size(); ++i) acc.dWx[i] += step.dWx[i];
  for (std::size_t i = 0; i < acc.dWh.size(); ++i) acc.dWh[i] += step.dWh[i];
  for (std::size_t i = 0; i < acc.db.size(); ++i) acc.db[i] += step.db[i];
  for (std::size_t i = 0; i < acc.dWy.size(); ++i) acc.dWy[i] += step.dWy[i];
  for (std::size_t i = 0; i < acc.dby.size(); ++i) acc.dby[i] += step.dby[i];
}

void CharLSTMHead::clip_grads_inplace(CharLSTMGrad& grads) const {
  if (grad_clip_ <= 0.0) return;
  double sq = 0.0;
  auto add = [&](const std::vector<double>& v) {
    for (double x : v) sq += x * x;
  };
  add(grads.dE);
  add(grads.dWx);
  add(grads.dWh);
  add(grads.db);
  add(grads.dWy);
  add(grads.dby);
  const double norm = std::sqrt(sq);
  if (norm <= grad_clip_ || norm <= 0.0) return;
  const double scale = grad_clip_ / norm;
  auto scale_v = [&](std::vector<double>& v) {
    for (double& x : v) x *= scale;
  };
  scale_v(grads.dE);
  scale_v(grads.dWx);
  scale_v(grads.dWh);
  scale_v(grads.db);
  scale_v(grads.dWy);
  scale_v(grads.dby);
}

void CharLSTMHead::reset_state() {
  h_.assign(static_cast<std::size_t>(hidden), 0.0);
  c_.assign(static_cast<std::size_t>(hidden), 0.0);
  has_cache_ = false;
  bptt_caches_.clear();
  bptt_targets_.clear();
}

void CharLSTMHead::forward_step(int token_id, const double* h, const double* c, double* log_probs,
                                std::vector<double>& h_out, std::vector<double>& c_out,
                                CharLSTMCache* cache_out, double forget_gate_scale) const {
  if (token_id < 0 || token_id >= vocab_size) {
    throw std::runtime_error("char_lstm: token_id out of range");
  }
  const int four_h = 4 * hidden;
  const double* x = E.data() + static_cast<std::size_t>(token_id) * static_cast<std::size_t>(hidden);

  // Perf: forward_step runs once per training/inference step (the dominant hot path at D17
  // production scale). These were previously fresh-allocated std::vector<double> locals on
  // every single call; thread_local statics let the (fixed-size, per-object `hidden`/`vocab_size`)
  // backing storage persist across calls so steady-state operation does zero heap allocation
  // here, without changing any arithmetic (every element is fully overwritten before use) or
  // affecting correctness under CyphaLMBatch's per-thread parallel_batch (each thread gets its
  // own thread_local instance, so there is no cross-thread aliasing).
  thread_local std::vector<double> gates;
  thread_local std::vector<double> wh;
  if (gates.size() != static_cast<std::size_t>(four_h)) gates.resize(static_cast<std::size_t>(four_h));
  if (wh.size() != static_cast<std::size_t>(four_h)) wh.resize(static_cast<std::size_t>(four_h));
  matvec_rowmajor(Wx.data(), four_h, hidden, x, gates.data());
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

  thread_local std::vector<double> i_gate;
  thread_local std::vector<double> f_gate;
  thread_local std::vector<double> g_gate;
  thread_local std::vector<double> o_gate;
  if (i_gate.size() != static_cast<std::size_t>(hidden)) i_gate.resize(static_cast<std::size_t>(hidden));
  if (f_gate.size() != static_cast<std::size_t>(hidden)) f_gate.resize(static_cast<std::size_t>(hidden));
  if (g_gate.size() != static_cast<std::size_t>(hidden)) g_gate.resize(static_cast<std::size_t>(hidden));
  if (o_gate.size() != static_cast<std::size_t>(hidden)) o_gate.resize(static_cast<std::size_t>(hidden));
  const bool use_eml = activation_mode_ == LSTMActivationMode::Eml;
  const bool use_axiom = activation_mode_ == LSTMActivationMode::Axiom;
  for (int j = 0; j < hidden; ++j) {
    if (use_axiom && static_cast<int>(axiom_grammar_.i_gate.size()) == hidden) {
      i_gate[static_cast<std::size_t>(j)] =
          apply_axiom_gate(axiom_grammar_.i_gate[static_cast<std::size_t>(j)],
                           gates[static_cast<std::size_t>(j)], h[j], false);
      f_gate[static_cast<std::size_t>(j)] =
          apply_axiom_gate(axiom_grammar_.f_gate[static_cast<std::size_t>(j)],
                           gates[static_cast<std::size_t>(hidden + j)], c[j], false);
      g_gate[static_cast<std::size_t>(j)] =
          apply_axiom_gate(axiom_grammar_.g_gate[static_cast<std::size_t>(j)],
                           gates[static_cast<std::size_t>(2 * hidden + j)], h[j], true);
      o_gate[static_cast<std::size_t>(j)] =
          apply_axiom_gate(axiom_grammar_.o_gate[static_cast<std::size_t>(j)],
                           gates[static_cast<std::size_t>(3 * hidden + j)], c[j], false);
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

  // Perf (2026-07-12, part 2): tanh_c records tanh(c_out[j]) for the axiom/plain branches (both
  // use the identical o_gate*tanh(c_out) formula) so backward_step can reuse it directly instead
  // of recomputing std::tanh on the same value -- see CharLSTMCache::tanh_c_new. Not meaningful
  // for the eml branch (which doesn't use tanh at all), so left at whatever the thread_local
  // buffer previously held there; backward_step's `cache.used_eml` branch never reads it.
  thread_local std::vector<double> tanh_c;
  if (tanh_c.size() != static_cast<std::size_t>(hidden)) tanh_c.resize(static_cast<std::size_t>(hidden));

  c_out.assign(static_cast<std::size_t>(hidden), 0.0);
  h_out.assign(static_cast<std::size_t>(hidden), 0.0);
  for (int j = 0; j < hidden; ++j) {
    c_out[static_cast<std::size_t>(j)] =
        f_gate[static_cast<std::size_t>(j)] * c[j] + i_gate[static_cast<std::size_t>(j)] * g_gate[static_cast<std::size_t>(j)];
    if (use_eml) {
      h_out[static_cast<std::size_t>(j)] =
          o_gate[static_cast<std::size_t>(j)] * eml_nand(c_out[static_cast<std::size_t>(j)], 1.0);
    } else if (use_axiom) {
      tanh_c[static_cast<std::size_t>(j)] = std::tanh(c_out[static_cast<std::size_t>(j)]);
      h_out[static_cast<std::size_t>(j)] = o_gate[static_cast<std::size_t>(j)] * tanh_c[static_cast<std::size_t>(j)];
    } else {
      tanh_c[static_cast<std::size_t>(j)] = std::tanh(c_out[static_cast<std::size_t>(j)]);
      h_out[static_cast<std::size_t>(j)] = o_gate[static_cast<std::size_t>(j)] * tanh_c[static_cast<std::size_t>(j)];
    }
  }

  thread_local std::vector<double> logits;
  thread_local std::vector<double> probs;
  if (logits.size() != static_cast<std::size_t>(vocab_size)) logits.resize(static_cast<std::size_t>(vocab_size));
  if (probs.size() != static_cast<std::size_t>(vocab_size)) probs.resize(static_cast<std::size_t>(vocab_size));
  matvec_rowmajor(Wy.data(), vocab_size, hidden, h_out.data(), logits.data());
  for (int k = 0; k < vocab_size; ++k) {
    logits[static_cast<std::size_t>(k)] += by[static_cast<std::size_t>(k)];
  }

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
    cache_out->tanh_c_new = tanh_c;
    cache_out->gates = gates;
    cache_out->logits = logits;
    cache_out->probs = probs;
    cache_out->used_eml = use_eml;
    cache_out->used_axiom = use_axiom;
    cache_out->used_sr_gates = use_sr;
  }
}

CharLSTMGrad CharLSTMHead::backward_step(const CharLSTMCache& cache, int target_id,
                                         double logit_nudge, double hidden_nudge) const {
  CharLSTMGrad out;
  backward_step(cache, target_id, out, logit_nudge, hidden_nudge, nullptr, nullptr);
  return out;
}

void CharLSTMHead::backward_step(const CharLSTMCache& cache, int target_id, CharLSTMGrad& out,
                                 double logit_nudge, double hidden_nudge, const double* dh_next,
                                 const double* dc_next) const {
  if (target_id < 0 || target_id >= vocab_size) {
    throw std::runtime_error("char_lstm: target_id out of range");
  }
  const int four_h = 4 * hidden;
  // Perf: `out` is caller-owned (see header doc). `.assign(n, 0.0)` on an already-correctly-sized
  // vector reuses its existing heap buffer (no realloc), so callers that pass the same `out`
  // instance across repeated calls (the online training hot path does, via a persistent member)
  // pay for this zero-fill once and then never re-allocate these buffers again.
  out.dE.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  out.dWx.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  out.dWh.assign(static_cast<std::size_t>(four_h) * static_cast<std::size_t>(hidden), 0.0);
  out.db.assign(static_cast<std::size_t>(four_h), 0.0);
  out.dWy.assign(static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(hidden), 0.0);
  out.dby.assign(static_cast<std::size_t>(vocab_size), 0.0);
  out.dh_prev.assign(static_cast<std::size_t>(hidden), 0.0);
  out.dc_prev.assign(static_cast<std::size_t>(hidden), 0.0);

  // Internal scratch temporaries: thread_local (not member/mutable) so this stays safe under
  // CyphaLMBatch's parallel_batch, which calls backward_step from multiple threads on the same
  // shared CharLSTMHead instance -- each thread gets its own copy, no aliasing/races, and within
  // a thread the backing storage is reused call-to-call once warmed up (same reasoning as
  // forward_step above).
  thread_local std::vector<double> d_logits;
  thread_local std::vector<double> dh_new;
  thread_local std::vector<double> do_gate;
  thread_local std::vector<double> dc_new;
  thread_local std::vector<double> df_gate;
  thread_local std::vector<double> di_gate;
  thread_local std::vector<double> dg_gate;
  thread_local std::vector<double> dgates;
  thread_local std::vector<double> dx;
  if (dh_new.size() != static_cast<std::size_t>(hidden)) dh_new.resize(static_cast<std::size_t>(hidden));
  if (do_gate.size() != static_cast<std::size_t>(hidden)) do_gate.resize(static_cast<std::size_t>(hidden));
  if (dc_new.size() != static_cast<std::size_t>(hidden)) dc_new.resize(static_cast<std::size_t>(hidden));
  if (df_gate.size() != static_cast<std::size_t>(hidden)) df_gate.resize(static_cast<std::size_t>(hidden));
  if (di_gate.size() != static_cast<std::size_t>(hidden)) di_gate.resize(static_cast<std::size_t>(hidden));
  if (dg_gate.size() != static_cast<std::size_t>(hidden)) dg_gate.resize(static_cast<std::size_t>(hidden));
  if (dx.size() != static_cast<std::size_t>(hidden)) dx.resize(static_cast<std::size_t>(hidden));
  if (dgates.size() != static_cast<std::size_t>(four_h)) dgates.resize(static_cast<std::size_t>(four_h));

  if (g_bwd_trace.enabled) g_bwd_trace.note_call();

  {
    BwdScopeTimer __t(0);  // 1. output-layer backward (dWy/dby/dh_new)
    d_logits = cache.probs;
    d_logits[static_cast<std::size_t>(target_id)] -= 1.0;
    if (logit_nudge != 0.0) {
      for (int k = 0; k < vocab_size; ++k) {
        d_logits[static_cast<std::size_t>(k)] += logit_nudge;
      }
    }

    outer_rowmajor(d_logits.data(), vocab_size, cache.h_new.data(), hidden, out.dWy.data());
    out.dby = d_logits;

    matvec_transpose_rowmajor(Wy.data(), vocab_size, hidden, d_logits.data(), dh_new.data());
    if (hidden_nudge != 0.0) {
      for (int j = 0; j < hidden; ++j) {
        dh_new[static_cast<std::size_t>(j)] += hidden_nudge;
      }
    }
    if (dh_next != nullptr) {
      for (int j = 0; j < hidden; ++j) {
        dh_new[static_cast<std::size_t>(j)] += dh_next[static_cast<std::size_t>(j)];
      }
    }
  }

  {
    BwdScopeTimer __t(1);  // 2. activation gradient (do_gate/dc_new + df/di/dg/dc_prev base)
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
        // Perf (2026-07-12, part 2): tanh(c_new[j]) was previously recomputed here even though
        // forward_step already computed it once to produce h_out[j] = o_gate[j] * tanh(c_out[j]).
        // cache.tanh_c_new caches that exact value (see forward_step), so this is now a plain
        // lookup instead of a second std::tanh call per hidden dim per backward step -- zero
        // arithmetic change (same value, just not recomputed).
        const double tanh_c = cache.tanh_c_new[static_cast<std::size_t>(j)];
        do_gate[static_cast<std::size_t>(j)] = dh_new[static_cast<std::size_t>(j)] * tanh_c;
        dc_new[static_cast<std::size_t>(j)] =
            dh_new[static_cast<std::size_t>(j)] * cache.o[static_cast<std::size_t>(j)] * (1.0 - tanh_c * tanh_c);
      }
    }
    if (dc_next != nullptr) {
      for (int j = 0; j < hidden; ++j) {
        dc_new[static_cast<std::size_t>(j)] += dc_next[static_cast<std::size_t>(j)];
      }
    }

    for (int j = 0; j < hidden; ++j) {
      df_gate[static_cast<std::size_t>(j)] = dc_new[static_cast<std::size_t>(j)] * cache.c[static_cast<std::size_t>(j)];
      di_gate[static_cast<std::size_t>(j)] = dc_new[static_cast<std::size_t>(j)] * cache.g[static_cast<std::size_t>(j)];
      dg_gate[static_cast<std::size_t>(j)] = dc_new[static_cast<std::size_t>(j)] * cache.i[static_cast<std::size_t>(j)];
      out.dc_prev[static_cast<std::size_t>(j)] = dc_new[static_cast<std::size_t>(j)] * cache.f[static_cast<std::size_t>(j)];
    }
  }

  // Per-dimension derivative dispatch matching apply_axiom_gate's forward selection
  // (H15: axiom_grammar_ mixes Sigmoid/Tanh/Eml per hidden dimension). Mirrors the
  // eml_nand_grad convention used below for eml-selected dims: the second (state_ref)
  // argument's gradient is only propagated into dc_prev (f/o use c[j] as state_ref);
  // i/g use h[j] as state_ref, and — consistent with the pure-eml branch above — that
  // term is not separately accumulated since out.dh_prev is fully recomputed from
  // dgates via the Wh backprop below.
  // Matches apply_axiom_gate(..., candidate): control maps tanh→[0,1]; candidate maps sig/eml→[-1,1].
  auto axiom_gate_grad = [](AxiomGateFn fn, double pre, double state_ref, double gate_val,
                            double d_gate_val, bool candidate, double& d_pre, double& d_state_extra) {
    d_state_extra = 0.0;
    if (candidate) {
      switch (fn) {
        case AxiomGateFn::Sigmoid: {
          // gate = 2*σ-1 ⇒ dgate/dpre = 0.5*(1-gate²)
          d_pre = d_gate_val * 0.5 * (1.0 - gate_val * gate_val);
          break;
        }
        case AxiomGateFn::Tanh:
          d_pre = d_gate_val * (1.0 - gate_val * gate_val);
          break;
        case AxiomGateFn::Eml: {
          double d_eml = 0.0;
          eml_nand_grad(pre, state_ref, 2.0 * d_gate_val, d_eml, d_state_extra);
          d_pre = d_eml;
          break;
        }
      }
    } else {
      switch (fn) {
        case AxiomGateFn::Sigmoid:
          d_pre = d_gate_val * gate_val * (1.0 - gate_val);
          break;
        case AxiomGateFn::Tanh:
          // gate = 0.5*(tanh+1) ⇒ dgate/dpre = 2*gate*(1-gate)
          d_pre = d_gate_val * 2.0 * gate_val * (1.0 - gate_val);
          break;
        case AxiomGateFn::Eml:
          eml_nand_grad(pre, state_ref, d_gate_val, d_pre, d_state_extra);
          break;
      }
    }
  };
  const bool use_axiom_grad =
      cache.used_axiom && static_cast<int>(axiom_grammar_.i_gate.size()) == hidden;

  {
  BwdScopeTimer __t2(2);  // 3. gate derivative dispatch (dgates: per-gate eml/axiom/sigmoid loop)
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
    } else if (use_axiom_grad) {
      double d_pre = 0.0;
      double d_extra = 0.0;

      axiom_gate_grad(axiom_grammar_.i_gate[static_cast<std::size_t>(j)],
                      cache.gates[static_cast<std::size_t>(j)], cache.h[static_cast<std::size_t>(j)],
                      cache.i[static_cast<std::size_t>(j)], di_gate[static_cast<std::size_t>(j)], false, d_pre,
                      d_extra);
      dgates[static_cast<std::size_t>(j)] = d_pre;

      axiom_gate_grad(axiom_grammar_.f_gate[static_cast<std::size_t>(j)],
                      cache.gates[static_cast<std::size_t>(hidden + j)], cache.c[static_cast<std::size_t>(j)],
                      cache.f[static_cast<std::size_t>(j)], df_gate[static_cast<std::size_t>(j)], false, d_pre,
                      d_extra);
      dgates[static_cast<std::size_t>(hidden + j)] = d_pre;
      out.dc_prev[static_cast<std::size_t>(j)] += d_extra;

      axiom_gate_grad(axiom_grammar_.g_gate[static_cast<std::size_t>(j)],
                      cache.gates[static_cast<std::size_t>(2 * hidden + j)], cache.h[static_cast<std::size_t>(j)],
                      cache.g[static_cast<std::size_t>(j)], dg_gate[static_cast<std::size_t>(j)], true, d_pre,
                      d_extra);
      dgates[static_cast<std::size_t>(2 * hidden + j)] = d_pre;

      axiom_gate_grad(axiom_grammar_.o_gate[static_cast<std::size_t>(j)],
                      cache.gates[static_cast<std::size_t>(3 * hidden + j)], cache.c[static_cast<std::size_t>(j)],
                      cache.o[static_cast<std::size_t>(j)], do_gate[static_cast<std::size_t>(j)], false, d_pre,
                      d_extra);
      dgates[static_cast<std::size_t>(3 * hidden + j)] = d_pre;
      out.dc_prev[static_cast<std::size_t>(j)] += d_extra;
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
  }

  {
    BwdScopeTimer __t3(3);  // 4. weight-gradient outer products (dWx/dWh/db)
    outer_rowmajor(dgates.data(), four_h, cache.x.data(), hidden, out.dWx.data());
    outer_rowmajor(dgates.data(), four_h, cache.h.data(), hidden, out.dWh.data());
    out.db = dgates;
  }

  {
    BwdScopeTimer __t4(4);  // 5. input-gradient backprop (dx/dE)
    matvec_transpose_rowmajor(Wx.data(), four_h, hidden, dgates.data(), dx.data());
    for (int j = 0; j < hidden; ++j) {
      out.dE[static_cast<std::size_t>(cache.token_id) * static_cast<std::size_t>(hidden) + static_cast<std::size_t>(j)] =
          dx[static_cast<std::size_t>(j)];
    }
  }

  {
    BwdScopeTimer __t5(5);  // 6. hidden-gradient backprop (dh_prev)
    matvec_transpose_rowmajor(Wh.data(), four_h, hidden, dgates.data(), out.dh_prev.data());
  }
}

void CharLSTMHead::apply_grads(const CharLSTMGrad& grads_in, double lr) {
  const CharLSTMGrad* grads_ptr = &grads_in;
  CharLSTMGrad clipped;
  if (grad_clip_ > 0.0) {
    clipped = grads_in;
    clip_grads_inplace(clipped);
    grads_ptr = &clipped;
  }
  const CharLSTMGrad& grads = *grads_ptr;
  if (optim_ == LSTMOptim::Adam) {
    ensure_adam_state();
    ++adam_t_;
    constexpr double kBeta1 = 0.9;
    constexpr double kBeta2 = 0.999;
    constexpr double kEps = 1e-8;
    adam_update(E, m_E_, v_E_, grads.dE, lr, adam_t_, kBeta1, kBeta2, kEps);
    adam_update(Wx, m_Wx_, v_Wx_, grads.dWx, lr, adam_t_, kBeta1, kBeta2, kEps);
    adam_update(Wh, m_Wh_, v_Wh_, grads.dWh, lr, adam_t_, kBeta1, kBeta2, kEps);
    adam_update(b, m_b_, v_b_, grads.db, lr, adam_t_, kBeta1, kBeta2, kEps);
    adam_update(Wy, m_Wy_, v_Wy_, grads.dWy, lr, adam_t_, kBeta1, kBeta2, kEps);
    adam_update(by, m_by_, v_by_, grads.dby, lr, adam_t_, kBeta1, kBeta2, kEps);
    // AdamW: decoupled weight decay after Adam step; skip biases (b, by).
    if (weight_decay_ > 0.0) {
      const double scale = lr * weight_decay_;
      for (double& w : E) w -= scale * w;
      for (double& w : Wx) w -= scale * w;
      for (double& w : Wh) w -= scale * w;
      for (double& w : Wy) w -= scale * w;
    }
    return;
  }
  for (std::size_t i = 0; i < E.size(); ++i) E[i] -= lr * grads.dE[i];
  for (std::size_t i = 0; i < Wx.size(); ++i) Wx[i] -= lr * grads.dWx[i];
  for (std::size_t i = 0; i < Wh.size(); ++i) Wh[i] -= lr * grads.dWh[i];
  for (std::size_t i = 0; i < b.size(); ++i) b[i] -= lr * grads.db[i];
  for (std::size_t i = 0; i < Wy.size(); ++i) Wy[i] -= lr * grads.dWy[i];
  for (std::size_t i = 0; i < by.size(); ++i) by[i] -= lr * grads.dby[i];
}

void CharLSTMHead::flush_bptt_window(double lr, CharLSTMGrad* grads_out, double logit_nudge,
                                     double hidden_nudge) {
  if (bptt_caches_.empty()) return;
  CharLSTMGrad acc;
  ensure_grad_scratch(acc);
  CharLSTMGrad step;
  std::vector<double> dh(static_cast<std::size_t>(hidden), 0.0);
  std::vector<double> dc(static_cast<std::size_t>(hidden), 0.0);
  for (std::size_t ti = bptt_caches_.size(); ti > 0; --ti) {
    const std::size_t t = ti - 1;
    // Only the last (newest) step in the window gets navigation nudges — matches BPTT-1.
    const double ln = (t + 1 == bptt_caches_.size()) ? logit_nudge : 0.0;
    const double hn = (t + 1 == bptt_caches_.size()) ? hidden_nudge : 0.0;
    backward_step(bptt_caches_[t], bptt_targets_[t], step, ln, hn, dh.data(), dc.data());
    accumulate_grads(acc, step);
    dh = step.dh_prev;
    dc = step.dc_prev;
  }
  if (grads_out != nullptr) {
    *grads_out = acc;
  }
  apply_grads(acc, lr);
  bptt_caches_.clear();
  bptt_targets_.clear();
}

bool CharLSTMHead::push_bptt_step(const CharLSTMCache& cache, int target_id, double lr,
                                  CharLSTMGrad* grads_out, double logit_nudge, double hidden_nudge) {
  if (bptt_window_ <= 1) {
    // Prefer caller-owned scratch (hybrid hot path) to avoid re-allocating ~1.5MB grads each step.
    CharLSTMGrad local;
    CharLSTMGrad& grads = (grads_out != nullptr) ? *grads_out : local;
    backward_step(cache, target_id, grads, logit_nudge, hidden_nudge, nullptr, nullptr);
    apply_grads(grads, lr);
    return true;
  }
  bptt_caches_.push_back(cache);
  bptt_targets_.push_back(target_id);
  if (static_cast<int>(bptt_caches_.size()) < bptt_window_) {
    return false;
  }
  flush_bptt_window(lr, grads_out, logit_nudge, hidden_nudge);
  return true;
}

void CharLSTMHead::flush_bptt(double lr, CharLSTMGrad* grads_out, double logit_nudge,
                              double hidden_nudge) {
  flush_bptt_window(lr, grads_out, logit_nudge, hidden_nudge);
}

double CharLSTMHead::train_step(int token_id, int target_id, std::vector<double>& h, std::vector<double>& c,
                                double lr) {
  std::vector<double> log_probs(static_cast<std::size_t>(vocab_size));
  std::vector<double> h_new;
  std::vector<double> c_new;
  CharLSTMCache cache;
  forward_step(token_id, h.data(), c.data(), log_probs.data(), h_new, c_new, &cache);
  const double loss = -log_probs[static_cast<std::size_t>(target_id)];
  (void)push_bptt_step(cache, target_id, lr, nullptr, 0.0, 0.0);
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

void CharLSTMHead::backward(int target_id, double lr, CharLSTMGrad* grads_out, double logit_nudge,
                            double hidden_nudge) {
  if (!has_cache_) {
    return;
  }
  (void)push_bptt_step(cache_, target_id, lr, grads_out, logit_nudge, hidden_nudge);
  has_cache_ = false;
}

LSTMOptim parse_lstm_optim(const std::string& s) {
  if (s == "adam" || s == "Adam" || s == "ADAM") return LSTMOptim::Adam;
  return LSTMOptim::Sgd;
}

LSTMInitMode parse_lstm_init_mode(const std::string& s) {
  if (s == "classic" || s == "Classic" || s == "CLASSIC") return LSTMInitMode::Classic;
  return LSTMInitMode::Default;
}

std::string lstm_optim_name(LSTMOptim o) {
  return o == LSTMOptim::Adam ? "adam" : "sgd";
}

std::string lstm_init_mode_name(LSTMInitMode m) {
  return m == LSTMInitMode::Classic ? "classic" : "default";
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
