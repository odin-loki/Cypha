// rpsm_embed_grad_finite_diff — finite-difference regression guard for
// RPSM_UPGRADE_PLAN.md Finding #2 (`CyphaLMModel::rpsm_embed_backprop`, formerly
// `rpsm_embed_backprop_stub`).
//
// The old stub pasted RpsmSequenceLayer's field-space input gradient directly onto the
// leading embed dims, as if field-space and embed-space shared a basis -- they don't
// (field_x_ = proj_ssm_ * ctx, ctx = ssm.step(e)). This test independently re-derives what
// the *correct* embedding gradient should be for that exact forward chain (embed -> SSM
// layer-0 leaky-integrator state transition -> linear field projection -> RpsmSequenceLayer)
// using only forward evaluations of the real production classes (CellAISSM, RpsmSequenceLayer)
// -- a central finite-difference numerical gradient -- and checks it against the same
// transpose-chain-rule formula now implemented in `rpsm_embed_backprop`
// (native/src/cyphalm/cyphalm_model.cpp). A dimensional/basis bug like the old stub's would
// make analytic and numerical gradients disagree by O(1), not by finite-difference noise.
//
// Deliberately uses a single SSM layer (n_layers=1) so the analytic chain below is *exact*,
// with no truncation: CellAISSM::step's public API only exposes W_fast/W_slow for layer 0
// (`w_fast_layer0()`/`w_slow_layer0()`), so `rpsm_embed_backprop` itself only backprops
// through layer 0 even when n_layers>1 in production (documented scope decision, matching
// the same layer-0-only truncation already used by the neighboring `bptt_ssm_update`) --
// n_layers=1 here means that scope decision introduces zero approximation error, giving a
// tight numerical-vs-analytic tolerance.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cypha/cyphalm/cellai_ssm.hpp"
#include "cypha/rpsm/rpsm_sequence_layer.hpp"

namespace {

using cypha::cyphalm::CellAISSM;
using cypha::cyphalm::CellAISSMConfig;
using cypha::rpsm::RpsmSequenceConfig;
using cypha::rpsm::RpsmSequenceLayer;

constexpr int kDEmbed = 6;
constexpr int kDState = 5;    // SSM layer-0 state dim (fast/slow tracks).
constexpr int kFieldDim = 8;  // > RpsmStateDim, exercises the zero-pad path.
constexpr int kRpsmStateDim = 5;
constexpr int kRpsmFeatDim = 4;
constexpr int kNClasses = 6;
constexpr int kTarget = 2;
constexpr std::uint64_t kSeed = 1234;

CellAISSMConfig ssm_config() {
  CellAISSMConfig c;
  c.d_input = kDEmbed;
  c.d_state = kDState;
  c.tau_fast = 1.0;
  c.tau_slow = 20.0;
  c.n_layers = 1;
  c.seed = static_cast<int>(kSeed + 1);
  c.use_spectral_pde = false;  // exercise the leaky-integrator branch this fix assumes.
  c.use_multiscale = true;     // matches bench/config/profiles/cyphalm_d21_rpsm.json.
  c.use_sparse_hebbian = false;
  return c;
}

RpsmSequenceConfig rpsm_config() {
  RpsmSequenceConfig c;
  c.n_levels = 2;
  c.state_dim = kRpsmStateDim;
  c.feat_dim = kRpsmFeatDim;
  c.n_classes = kNClasses;
  c.n_memory_slots = 4;
  c.seed = kSeed + 29;
  return c;
}

// Fixed pseudo-random [kFieldDim x ctx_dim] projection (mirrors proj_ssm_'s role in
// CyphaLMModel::project_field: field = proj * ctx). Independent of CyphaLMModel's own RNG --
// this test only needs *some* fixed linear map, not byte-parity with production init.
std::vector<double> make_projection(int rows, int cols) {
  std::mt19937_64 rng(kSeed + 7);
  std::normal_distribution<double> nd(0.0, 0.1);
  std::vector<double> proj(static_cast<std::size_t>(rows * cols));
  for (auto& v : proj) v = nd(rng);
  return proj;
}

std::vector<double> matvec(const std::vector<double>& m, int rows, int cols,
                            const std::vector<double>& x) {
  std::vector<double> y(static_cast<std::size_t>(rows), 0.0);
  for (int r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) acc += m[static_cast<std::size_t>(r * cols + c)] * x[static_cast<std::size_t>(c)];
    y[static_cast<std::size_t>(r)] = acc;
  }
  return y;
}

std::vector<double> matvec_transpose(const std::vector<double>& m, int rows, int cols,
                                     const std::vector<double>& x) {
  std::vector<double> y(static_cast<std::size_t>(cols), 0.0);
  for (int c = 0; c < cols; ++c) {
    double acc = 0.0;
    for (int r = 0; r < rows; ++r) acc += m[static_cast<std::size_t>(r * cols + c)] * x[static_cast<std::size_t>(r)];
    y[static_cast<std::size_t>(c)] = acc;
  }
  return y;
}

// Forward pass: e -> ssm.step(e) -> ctx -> field = proj*ctx -> RpsmSequenceLayer -> NLL(target).
// Fresh CellAISSM/RpsmSequenceLayer each call (same seeds) => deterministic, zero-state-leak
// single-step evaluation purely as a function of `e`.
double forward_nll(const std::vector<double>& proj, int ctx_dim, const std::vector<double>& e) {
  CellAISSM ssm(ssm_config());
  const auto ctx = ssm.step(e);
  const auto field = matvec(proj, kFieldDim, ctx_dim, ctx);

  RpsmSequenceLayer layer(rpsm_config());
  std::vector<double> log_probs(static_cast<std::size_t>(kNClasses));
  layer.step(field.data(), kFieldDim, log_probs.data());
  return -log_probs[static_cast<std::size_t>(kTarget)];
}

// Analytic d(loss)/d(e) via the same transpose-chain-rule now implemented in
// `CyphaLMModel::rpsm_embed_backprop` (native/src/cyphalm/cyphalm_model.cpp): field_grad
// (RpsmSequenceLayer::input_grad(), zero-padded to field_dim) -> proj^T -> grad_ctx ->
// CellAISSM layer-0 leaky-integrator transpose (accounting for the multiscale alpha blend)
// -> grad_e. Keep this in sync with `rpsm_embed_backprop` if that function's math changes.
std::vector<double> analytic_grad_e(const std::vector<double>& proj, int ctx_dim,
                                    const std::vector<double>& e) {
  CellAISSM ssm(ssm_config());
  const auto ctx = ssm.step(e);
  const auto field = matvec(proj, kFieldDim, ctx_dim, ctx);

  RpsmSequenceLayer layer(rpsm_config());
  // lr is irrelevant to input_grad_'s value (see rpsm_sequence_layer.cpp train_step); the
  // resulting SGD-mutated layer is discarded immediately after reading input_grad().
  layer.train_step(field.data(), kFieldDim, kTarget, 0.01);
  const auto& field_grad = layer.input_grad();

  std::vector<double> grad_field(static_cast<std::size_t>(kFieldDim), 0.0);
  for (std::size_t i = 0; i < field_grad.size() && i < grad_field.size(); ++i) {
    grad_field[i] = field_grad[i];
  }

  const auto grad_ctx = matvec_transpose(proj, kFieldDim, ctx_dim, grad_field);

  const int sd = kDState;
  std::vector<double> grad_h(static_cast<std::size_t>(sd));
  std::vector<double> grad_s(static_cast<std::size_t>(sd));
  const double alpha = std::clamp(ssm.multiscale_alpha().empty() ? 0.5 : ssm.multiscale_alpha()[0], 0.0, 1.0);
  for (int i = 0; i < sd; ++i) {
    const double g_blend = grad_ctx[static_cast<std::size_t>(i)];
    const double g_s_direct = grad_ctx[static_cast<std::size_t>(sd + i)];
    grad_h[static_cast<std::size_t>(i)] = alpha * g_blend;
    grad_s[static_cast<std::size_t>(i)] = (1.0 - alpha) * g_blend + g_s_direct;
  }

  const auto& w_fast = ssm.w_fast_layer0();
  const auto& w_slow = ssm.w_slow_layer0();
  const double lf = ssm.lambda_fast();
  const double ls = ssm.lambda_slow();
  std::vector<double> grad_e(kDEmbed, 0.0);
  for (int i = 0; i < sd; ++i) {
    const double gh = (1.0 - lf) * grad_h[static_cast<std::size_t>(i)];
    const double gs = (1.0 - ls) * grad_s[static_cast<std::size_t>(i)];
    const double* wf_row = w_fast.data() + static_cast<std::size_t>(i) * kDEmbed;
    const double* ws_row = w_slow.data() + static_cast<std::size_t>(i) * kDEmbed;
    for (int j = 0; j < kDEmbed; ++j) {
      grad_e[static_cast<std::size_t>(j)] += gh * wf_row[j] + gs * ws_row[j];
    }
  }
  return grad_e;
}

bool test_finite_difference_matches_analytic() {
  const int ctx_dim = ssm_config().n_layers * 2 * kDState;
  const auto proj = make_projection(kFieldDim, ctx_dim);

  std::vector<double> e(static_cast<std::size_t>(kDEmbed));
  {
    std::mt19937_64 rng(kSeed + 99);
    std::normal_distribution<double> nd(0.0, 0.3);
    for (auto& v : e) v = nd(rng);
  }

  const auto analytic = analytic_grad_e(proj, ctx_dim, e);

  constexpr double kEps = 1e-5;
  std::vector<double> numeric(static_cast<std::size_t>(kDEmbed), 0.0);
  for (int j = 0; j < kDEmbed; ++j) {
    auto e_plus = e;
    auto e_minus = e;
    e_plus[static_cast<std::size_t>(j)] += kEps;
    e_minus[static_cast<std::size_t>(j)] -= kEps;
    const double loss_plus = forward_nll(proj, ctx_dim, e_plus);
    const double loss_minus = forward_nll(proj, ctx_dim, e_minus);
    numeric[static_cast<std::size_t>(j)] = (loss_plus - loss_minus) / (2.0 * kEps);
  }

  double max_abs_analytic = 0.0;
  for (double v : analytic) max_abs_analytic = std::max(max_abs_analytic, std::fabs(v));
  if (!(max_abs_analytic > 1e-6)) {
    std::cerr << "rpsm_embed_grad_finite_diff: analytic gradient is degenerately small ("
              << max_abs_analytic << "); test would be vacuous\n";
    return false;
  }

  bool ok = true;
  for (int j = 0; j < kDEmbed; ++j) {
    const double a = analytic[static_cast<std::size_t>(j)];
    const double n = numeric[static_cast<std::size_t>(j)];
    const double diff = std::fabs(a - n);
    const double scale = std::max({1.0, std::fabs(a), std::fabs(n)});
    const double rel = diff / scale;
    if (!std::isfinite(a) || !std::isfinite(n) || rel > 1e-4) {
      std::cerr << "rpsm_embed_grad_finite_diff: dim " << j << " analytic=" << a
                << " numeric=" << n << " rel_err=" << rel << "\n";
      ok = false;
    }
  }
  return ok;
}

}  // namespace

int main() {
  try {
    if (!test_finite_difference_matches_analytic()) {
      return 1;
    }
    std::cout << "rpsm_embed_grad_finite_diff OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_embed_grad_finite_diff: " << e.what() << "\n";
    return 1;
  }
}
