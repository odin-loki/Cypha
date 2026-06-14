#include "cypha/cyphalm/cellai_ssm.hpp"

#include "cypha/cyphalm/hebbian_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cypha::cyphalm {

namespace {

inline void hebbian_sparse_update_impl(HebbianSSMState& state, const double* fast_state,
                                         const double* slow_state, double lr, int layer) {
  sparse_hebbian_update(state, fast_state, slow_state, lr, layer);
}

}  // namespace

namespace {

struct Rng {
  std::uint32_t state;
  explicit Rng(std::uint32_t seed) : state(seed ? seed : 1u) {}
  double normal() {
    double u1 = static_cast<double>(next()) / static_cast<double>(0x7FFFFFFFu);
    double u2 = static_cast<double>(next()) / static_cast<double>(0x7FFFFFFFu);
    if (u1 < 1e-12) {
      u1 = 1e-12;
    }
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
  }
  std::uint32_t next() {
    state = static_cast<std::uint32_t>((static_cast<std::uint64_t>(state) * 48271u) % 2147483647u);
    return state;
  }
};

}  // namespace

double CellAISSM::clip(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

std::vector<double> CellAISSM::matvec(const std::vector<double>& mat, int rows, int cols,
                                      const std::vector<double>& x) {
  std::vector<double> out(static_cast<std::size_t>(rows), 0.0);
  for (int r = 0; r < rows; ++r) {
    double acc = 0.0;
    const std::size_t row_off = static_cast<std::size_t>(r * cols);
    for (int c = 0; c < cols; ++c) {
      acc += mat[row_off + static_cast<std::size_t>(c)] * x[static_cast<std::size_t>(c)];
    }
    out[static_cast<std::size_t>(r)] = acc;
  }
  return out;
}

CellAISSM::CellAISSM(CellAISSMConfig cfg) : cfg_(cfg) {
  if (cfg_.d_input < 1 || cfg_.d_state < 1 || cfg_.n_layers < 1) {
    throw std::invalid_argument("CellAISSM: invalid dims");
  }
  if (cfg_.tau_fast <= 0.0 || cfg_.tau_slow <= 0.0) {
    throw std::invalid_argument("CellAISSM: tau must be positive");
  }

  lambda_fast_ = std::exp(-1.0 / cfg_.tau_fast);
  lambda_slow_ = std::exp(-1.0 / cfg_.tau_slow);

  layer_input_dims_.assign(static_cast<std::size_t>(cfg_.n_layers), 0);
  layer_input_dims_[0] = cfg_.d_input;
  for (int i = 1; i < cfg_.n_layers; ++i) {
    layer_input_dims_[static_cast<std::size_t>(i)] = 2 * cfg_.d_state;
  }

  Rng rng(static_cast<std::uint32_t>(cfg_.seed + 1));
  W_fast_.resize(static_cast<std::size_t>(cfg_.n_layers));
  W_slow_.resize(static_cast<std::size_t>(cfg_.n_layers));
  W_hebb_.resize(static_cast<std::size_t>(cfg_.n_layers));
  a_kernel_fast_.resize(static_cast<std::size_t>(cfg_.n_layers));
  a_kernel_slow_.resize(static_cast<std::size_t>(cfg_.n_layers));

  for (int layer = 0; layer < cfg_.n_layers; ++layer) {
    const int in_dim = layer_input_dims_[static_cast<std::size_t>(layer)];
    W_fast_[static_cast<std::size_t>(layer)].assign(static_cast<std::size_t>(cfg_.d_state * in_dim), 0.0);
    W_slow_[static_cast<std::size_t>(layer)].assign(static_cast<std::size_t>(cfg_.d_state * in_dim), 0.0);
    for (auto& v : W_fast_[static_cast<std::size_t>(layer)]) {
      v = rng.normal() * 0.05;
    }
    for (auto& v : W_slow_[static_cast<std::size_t>(layer)]) {
      v = rng.normal() * 0.05;
    }
    W_hebb_[static_cast<std::size_t>(layer)].assign(
        static_cast<std::size_t>(cfg_.d_state * cfg_.d_state), 0.0);

    a_kernel_fast_[static_cast<std::size_t>(layer)].assign(static_cast<std::size_t>(cfg_.d_state), 0.0);
    a_kernel_slow_[static_cast<std::size_t>(layer)].assign(static_cast<std::size_t>(cfg_.d_state), 0.0);
    a_kernel_fast_[static_cast<std::size_t>(layer)][0] = lambda_fast_;
    a_kernel_slow_[static_cast<std::size_t>(layer)][0] = lambda_slow_;
  }

  alpha_.assign(static_cast<std::size_t>(cfg_.n_layers), 0.5);
  reset();
}

void CellAISSM::reset() {
  h_.assign(static_cast<std::size_t>(cfg_.n_layers),
            std::vector<double>(static_cast<std::size_t>(cfg_.d_state), 0.0));
  s_.assign(static_cast<std::size_t>(cfg_.n_layers),
            std::vector<double>(static_cast<std::size_t>(cfg_.d_state), 0.0));
}

void CellAISSM::reset_fast_only() {
  for (auto& row : h_) {
    std::fill(row.begin(), row.end(), 0.0);
  }
}

void CellAISSM::reset_slow_only() {
  for (auto& row : s_) {
    std::fill(row.begin(), row.end(), 0.0);
  }
}

void CellAISSM::enable_hebb_graph(const HebbianGraphConfig& cfg) {
  HebbianGraphConfig gc = cfg;
  if (gc.n <= 0) {
    gc.n = 2 * cfg_.d_state;
  }
  hebb_graph_ = std::make_unique<HebbianGraph>(gc);
}

void CellAISSM::enable_temporal_som(cypha::som::TemporalSOMConfig cfg) {
  temporal_som_ = std::make_unique<cypha::som::TemporalSOM>(cfg);
  lam_fast_scale_ = 1.0;
  lam_slow_scale_ = 1.0;
}

void CellAISSM::set_projection_weights(int layer, const std::vector<double>& w_fast,
                                       const std::vector<double>& w_slow) {
  if (layer < 0 || layer >= cfg_.n_layers) {
    throw std::out_of_range("CellAISSM::set_projection_weights: layer out of range");
  }
  const int in_dim = layer_input_dims_[static_cast<std::size_t>(layer)];
  const std::size_t expected = static_cast<std::size_t>(cfg_.d_state * in_dim);
  if (w_fast.size() != expected || w_slow.size() != expected) {
    throw std::invalid_argument("CellAISSM::set_projection_weights: size mismatch");
  }
  W_fast_[static_cast<std::size_t>(layer)] = w_fast;
  W_slow_[static_cast<std::size_t>(layer)] = w_slow;
}

void CellAISSM::sparse_hebbian_update(const std::vector<double>& pre, const std::vector<double>& post,
                                      double lr, int layer) {
  if (layer < 0 || layer >= cfg_.n_layers) {
    return;
  }
  HebbianSSMState state;
  state.d_state = cfg_.d_state;
  state.n_layers = cfg_.n_layers;
  state.w = W_hebb_;
  hebbian_sparse_update_impl(state, pre.data(), post.data(), lr, layer);
  W_hebb_ = std::move(state.w);
}

std::vector<double> CellAISSM::mean_fast_state() const {
  std::vector<double> out(static_cast<std::size_t>(cfg_.d_state), 0.0);
  if (h_.empty()) {
    return out;
  }
  for (const auto& row : h_) {
    for (int i = 0; i < cfg_.d_state; ++i) {
      out[static_cast<std::size_t>(i)] += row[static_cast<std::size_t>(i)];
    }
  }
  const double inv = 1.0 / static_cast<double>(h_.size());
  for (double& v : out) {
    v *= inv;
  }
  return out;
}

std::vector<double> CellAISSM::step(const std::vector<double>& e_t) {
  if (static_cast<int>(e_t.size()) != layer_input_dims_[0]) {
    throw std::invalid_argument("CellAISSM: input dim mismatch");
  }

  std::vector<double> layer_input = e_t;
  if (temporal_som_) {
    const auto [bmu, lf, ls] = temporal_som_->step(e_t, true);
    (void)bmu;
    lam_fast_scale_ = lf;
    lam_slow_scale_ = ls;
  }

  std::vector<double> contexts;
  contexts.reserve(static_cast<std::size_t>(context_dim()));

  for (int layer = 0; layer < cfg_.n_layers; ++layer) {
    auto& h = h_[static_cast<std::size_t>(layer)];
    auto& s = s_[static_cast<std::size_t>(layer)];
    const int in_dim = layer_input_dims_[static_cast<std::size_t>(layer)];

    const auto wh = matvec(W_fast_[static_cast<std::size_t>(layer)], cfg_.d_state, in_dim, layer_input);
    const auto ws = matvec(W_slow_[static_cast<std::size_t>(layer)], cfg_.d_state, in_dim, layer_input);

    const double lf = clip(lambda_fast_ * lam_fast_scale_, 0.01, 0.999);
    const double ls = clip(lambda_slow_ * lam_slow_scale_, 0.01, 0.999);

    if (cfg_.use_spectral_pde) {
      h = spectral_step(h, a_kernel_fast_[static_cast<std::size_t>(layer)]);
      s = spectral_step(s, a_kernel_slow_[static_cast<std::size_t>(layer)]);
      for (int i = 0; i < cfg_.d_state; ++i) {
        h[static_cast<std::size_t>(i)] += (1.0 - lf) * wh[static_cast<std::size_t>(i)];
        s[static_cast<std::size_t>(i)] += (1.0 - ls) * ws[static_cast<std::size_t>(i)];
      }
    } else {
      for (int i = 0; i < cfg_.d_state; ++i) {
        h[static_cast<std::size_t>(i)] = lf * h[static_cast<std::size_t>(i)] + (1.0 - lf) * wh[static_cast<std::size_t>(i)];
        s[static_cast<std::size_t>(i)] = ls * s[static_cast<std::size_t>(i)] + (1.0 - ls) * ws[static_cast<std::size_t>(i)];
      }
    }

    std::vector<double> ctx;
    ctx.reserve(static_cast<std::size_t>(2 * cfg_.d_state));
    if (cfg_.use_multiscale) {
      const double alpha = clip(alpha_[static_cast<std::size_t>(layer)], 0.0, 1.0);
      for (int i = 0; i < cfg_.d_state; ++i) {
        ctx.push_back(alpha * h[static_cast<std::size_t>(i)] + (1.0 - alpha) * s[static_cast<std::size_t>(i)]);
      }
      ctx.insert(ctx.end(), s.begin(), s.end());
    } else {
      ctx.insert(ctx.end(), h.begin(), h.end());
      ctx.insert(ctx.end(), s.begin(), s.end());
    }

    if (hebb_graph_) {
      ctx = hebb_graph_->diffuse(ctx);
      hebb_graph_->update(ctx.data());
    }

    if (cfg_.use_sparse_hebbian) {
      sparse_hebbian_update(h, s, 1e-4, layer);
    }

    contexts.insert(contexts.end(), ctx.begin(), ctx.end());
    layer_input = std::move(ctx);
  }

  return contexts;
}

std::vector<std::vector<double>> CellAISSM::process_sequence(
    const std::vector<std::vector<double>>& embeddings) {
  std::vector<std::vector<double>> out;
  out.reserve(embeddings.size());
  for (const auto& row : embeddings) {
    out.push_back(step(row));
  }
  return out;
}

void CellAISSM::apply_bptt_fast_layer0(const std::vector<double>& grad_h, const std::vector<double>& e,
                                       double ssm_lr, double scale) {
  if (cfg_.n_layers < 1) return;
  if (static_cast<int>(grad_h.size()) != cfg_.d_state ||
      static_cast<int>(e.size()) != layer_input_dims_[0]) {
    return;
  }
  const double lf = clip(lambda_fast_, 0.01, 0.999);
  const int cols = layer_input_dims_[0];
  auto& w = W_fast_[0];
  for (int r = 0; r < cfg_.d_state; ++r) {
    for (int c = 0; c < cols; ++c) {
      w[static_cast<std::size_t>(r * cols + c)] -=
          ssm_lr * scale * (1.0 - lf) * grad_h[static_cast<std::size_t>(r)] * e[static_cast<std::size_t>(c)];
    }
  }
}

void CellAISSM::apply_bptt_delta_avg(const std::vector<double>& avg_delta, double ssm_lr, double scale) {
  if (cfg_.n_layers < 1 || avg_delta.empty()) return;
  auto& w = W_fast_[0];
  const std::size_t n = std::min(w.size(), avg_delta.size());
  for (std::size_t i = 0; i < n; ++i) {
    w[i] -= ssm_lr * scale * avg_delta[i];
  }
}

const std::vector<double>& CellAISSM::w_fast_layer0() const {
  static const std::vector<double> kEmpty;
  if (cfg_.n_layers < 1 || W_fast_.empty()) {
    return kEmpty;
  }
  return W_fast_[0];
}

std::vector<double>& CellAISSM::w_fast_layer0_mut() {
  return W_fast_[0];
}

const std::vector<double>& CellAISSM::w_slow_layer0() const {
  static const std::vector<double> kEmpty;
  if (cfg_.n_layers < 1 || W_slow_.empty()) {
    return kEmpty;
  }
  return W_slow_[0];
}

std::vector<double>& CellAISSM::w_slow_layer0_mut() {
  return W_slow_[0];
}

nlohmann::json CellAISSM::get_state() const {
  nlohmann::json j;
  j["h"] = h_;
  j["s"] = s_;
  j["alpha"] = alpha_;
  j["W_fast"] = W_fast_;
  j["W_slow"] = W_slow_;
  j["W_hebb"] = W_hebb_;
  j["lam_fast_scale"] = lam_fast_scale_;
  j["lam_slow_scale"] = lam_slow_scale_;
  j["d_input"] = cfg_.d_input;
  j["d_state"] = cfg_.d_state;
  j["tau_fast"] = cfg_.tau_fast;
  j["tau_slow"] = cfg_.tau_slow;
  j["n_layers"] = cfg_.n_layers;
  j["seed"] = cfg_.seed;
  return j;
}

void CellAISSM::set_state(const nlohmann::json& state) {
  if (state.contains("h")) {
    h_ = state.at("h").get<std::vector<std::vector<double>>>();
  }
  if (state.contains("s")) {
    s_ = state.at("s").get<std::vector<std::vector<double>>>();
  }
  if (state.contains("alpha")) {
    alpha_ = state.at("alpha").get<std::vector<double>>();
  }
  if (state.contains("W_fast")) {
    W_fast_ = state.at("W_fast").get<std::vector<std::vector<double>>>();
  }
  if (state.contains("W_slow")) {
    W_slow_ = state.at("W_slow").get<std::vector<std::vector<double>>>();
  }
  if (state.contains("W_hebb")) {
    W_hebb_ = state.at("W_hebb").get<std::vector<std::vector<double>>>();
  }
  if (state.contains("lam_fast_scale")) {
    lam_fast_scale_ = state.at("lam_fast_scale").get<double>();
  }
  if (state.contains("lam_slow_scale")) {
    lam_slow_scale_ = state.at("lam_slow_scale").get<double>();
  }
  if (static_cast<int>(h_.size()) != cfg_.n_layers || static_cast<int>(s_.size()) != cfg_.n_layers) {
    throw std::invalid_argument("CellAISSM::set_state: layer count mismatch");
  }
}

}  // namespace cypha::cyphalm
