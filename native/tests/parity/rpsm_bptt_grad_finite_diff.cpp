// rpsm_bptt_grad_finite_diff — finite-difference regression guard for RPSM_UPGRADE_PLAN.md
// §14 (BPTT through RpsmSequenceLayer's own hierarchy recurrence).
//
// Unlike native_rpsm_embed_grad_finite_diff (which checks a *single-step-local* gradient),
// this test verifies genuine *multi-step* backprop-through-time: it runs a fixed
// `bptt_window`-length sequence of (input, target) pairs through a fresh RpsmSequenceLayer,
// reads how much `bptt_backward_and_apply()` moved `w_up_`/`w_enc_`/`w_carry_` at the window
// flush, and independently re-derives the same gradient via central finite-difference on the
// *sum of the window's per-step losses* (with weights held fixed across the window, matching
// this mechanism's own truncated-BPTT semantics -- see rpsm_sequence_layer.cpp). A window-boundary
// bug (off-by-one in which step's target feeds which gradient, gradient scaling errors, double
// counting/dropping terms) would show up as an O(1) relative mismatch here, not finite-difference
// noise, because most of what this test is exercising (`w_up_`'s reconstruction-error path and
// the injection/encode paths threading through *every* cached step, not just the most recent
// one) literally could not produce a nonzero numeric-vs-analytic match by accident.
//
// `beta_memory` is deliberately 0 here: `bptt_backward_and_apply()`'s own comments document a
// stop-gradient on M_slots' read path (a bounded, measured simplification, not a silent
// truncation of the mechanism this test verifies) -- zeroing the memory blend's *forward*
// contribution too means that documented omission cannot introduce a spurious mismatch in
// exactly the terms this test is designed to catch.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cypha/rpsm/rpsm_sequence_layer.hpp"

namespace {

using cypha::rpsm::RpsmSequenceConfig;
using cypha::rpsm::RpsmSequenceLayer;

constexpr int kStateDim = 6;
constexpr int kFeatDim = 4;
constexpr int kNLevels = 2;
constexpr int kNClasses = 5;
constexpr int kNMemorySlots = 3;
constexpr int kWindow = 4;
constexpr std::uint64_t kSeed = 4242;
// Matches the production `rpsm_lr` default (bench/config/profiles/cyphalm_d21_rpsm.json).
// Deliberately *not* pushed higher: `bptt_backward_and_apply()`'s cached `enc_grad_t`/`c.enc_pre`
// snapshot the classifier gradient as it stood forward-time (i.e. treats psi_.mu's own per-step
// SGD trajectory as given data, not something to differentiate `w_up_`/`w_enc_`/`w_carry_`
// through) -- correct to first order, but at a large enough lr the finite-difference probe's
// weight perturbation measurably changes psi_.mu's *own* trajectory too (since psi_.mu is
// SGD-updated from feat_buf_, which the perturbed weights change), which is a real but
// higher-order (and, by design, out-of-scope for this mechanism -- BPTT-through-an-optimizer's-
// own-trajectory is a different, much larger problem than BPTT through the hidden-state
// recurrence this test verifies) effect. That coupling shrinks faster than the direct gradient
// as lr shrinks, so kLr=0.01 keeps the probe in the regime where it is negligible relative to
// the 5e-4 tolerance below, while still exercising a realistic-scale update.
constexpr double kLr = 0.01;

RpsmSequenceConfig make_config() {
  RpsmSequenceConfig c;
  c.n_levels = kNLevels;
  c.state_dim = kStateDim;
  c.feat_dim = kFeatDim;
  c.n_classes = kNClasses;
  c.n_memory_slots = kNMemorySlots;
  c.seed = kSeed;
  c.bptt_window = kWindow;
  // Isolate the mechanism under test: no spectral alpha / normalised eta (keeps `alpha`/`lr`
  // constant and known across the window, matching the derivation in
  // bptt_backward_and_apply()'s comments), and beta_memory=0 (see file header comment above).
  c.use_spectral_alpha = false;
  c.use_normalized_eta = false;
  c.beta_memory = 0.0;
  return c;
}

struct StepSample {
  std::vector<double> input;
  int target = 0;
};

std::vector<StepSample> make_sequence() {
  std::mt19937_64 rng(kSeed + 5);
  std::normal_distribution<double> nd(0.0, 0.4);
  std::uniform_int_distribution<int> tgt_dist(0, kNClasses - 1);
  std::vector<StepSample> seq(static_cast<std::size_t>(kWindow));
  for (auto& s : seq) {
    s.input.assign(static_cast<std::size_t>(kStateDim), 0.0);
    for (auto& v : s.input) v = nd(rng);
    s.target = tgt_dist(rng);
  }
  return seq;
}

// Runs exactly kWindow train_step calls (one full BPTT window) from a fresh layer built with
// the given weight overrides, returning the sum of the window's per-step `metrics.loss`.
double window_loss_sum(const std::vector<StepSample>& seq, const std::vector<double>* w_up_override,
                       const std::vector<double>* w_enc_override,
                       const std::vector<double>* w_carry_override) {
  RpsmSequenceLayer layer(make_config());
  if (w_up_override != nullptr) layer.w_up_mut() = *w_up_override;
  if (w_enc_override != nullptr) layer.w_enc_mut() = *w_enc_override;
  if (w_carry_override != nullptr) layer.w_carry_mut() = *w_carry_override;

  double total = 0.0;
  for (int t = 0; t < kWindow; ++t) {
    const auto& s = seq[static_cast<std::size_t>(t)];
    const auto m = layer.train_step(s.input.data(), static_cast<int>(s.input.size()), s.target, kLr);
    total += m.loss;
  }
  return total;
}

// Central finite-difference gradient of `window_loss_sum` w.r.t. one entry of the weight matrix
// selected by `which` (0=w_up_, 1=w_enc_, 2=w_carry_), holding every other weight at `base_*`.
double numeric_grad(const std::vector<StepSample>& seq, int which, std::size_t idx, double eps,
                    const std::vector<double>& base_up, const std::vector<double>& base_enc,
                    const std::vector<double>& base_carry) {
  std::vector<double> up_plus = base_up, up_minus = base_up;
  std::vector<double> enc_plus = base_enc, enc_minus = base_enc;
  std::vector<double> carry_plus = base_carry, carry_minus = base_carry;
  if (which == 0) {
    up_plus[idx] += eps;
    up_minus[idx] -= eps;
  } else if (which == 1) {
    enc_plus[idx] += eps;
    enc_minus[idx] -= eps;
  } else {
    carry_plus[idx] += eps;
    carry_minus[idx] -= eps;
  }
  const double loss_plus = window_loss_sum(seq, &up_plus, &enc_plus, &carry_plus);
  const double loss_minus = window_loss_sum(seq, &up_minus, &enc_minus, &carry_minus);
  return (loss_plus - loss_minus) / (2.0 * eps);
}

bool check_weight_group(const char* name, int which, const std::vector<double>& base_up,
                        const std::vector<double>& base_enc, const std::vector<double>& base_carry,
                        const std::vector<double>& before, const std::vector<double>& after,
                        const std::vector<StepSample>& seq) {
  // bptt_backward_and_apply() applies `w -= mean_lr * grad * (1/window)`; with normalised-eta
  // off, every cached step's effective_lr is exactly `kLr`, so mean_lr == kLr and the analytic
  // per-entry gradient recovers exactly via algebraic inversion of that update rule.
  std::vector<double> analytic(before.size());
  for (std::size_t i = 0; i < before.size(); ++i) {
    analytic[i] = (before[i] - after[i]) * static_cast<double>(kWindow) / kLr;
  }

  double max_abs_analytic = 0.0;
  for (double v : analytic) max_abs_analytic = std::max(max_abs_analytic, std::fabs(v));
  if (!(max_abs_analytic > 1e-8)) {
    std::cerr << "rpsm_bptt_grad_finite_diff: " << name
              << " analytic gradient is degenerately small (" << max_abs_analytic
              << "); test would be vacuous\n";
    return false;
  }

  constexpr double kEps = 1e-6;
  bool ok = true;
  for (std::size_t idx = 0; idx < before.size(); ++idx) {
    const double n = numeric_grad(seq, which, idx, kEps, base_up, base_enc, base_carry);
    const double a = analytic[idx];
    const double diff = std::fabs(a - n);
    const double scale = std::max({1.0, std::fabs(a), std::fabs(n)});
    const double rel = diff / scale;
    if (!std::isfinite(a) || !std::isfinite(n) || rel > 5e-4) {
      std::cerr << "rpsm_bptt_grad_finite_diff: " << name << " idx=" << idx << " analytic=" << a
                << " numeric=" << n << " rel_err=" << rel << "\n";
      ok = false;
    }
  }
  return ok;
}

bool test_bptt_gradients_match_finite_difference() {
  const auto seq = make_sequence();

  RpsmSequenceLayer layer(make_config());
  const std::vector<double> base_up = layer.w_up();
  const std::vector<double> base_enc = layer.w_enc();
  const std::vector<double> base_carry = layer.w_carry();

  bool any_flushed = false;
  for (int t = 0; t < kWindow; ++t) {
    const auto& s = seq[static_cast<std::size_t>(t)];
    layer.train_step(s.input.data(), static_cast<int>(s.input.size()), s.target, kLr);
    if (layer.bptt_window_flushed()) any_flushed = true;
  }
  if (!any_flushed) {
    std::cerr << "rpsm_bptt_grad_finite_diff: window never flushed after " << kWindow
              << " steps -- bptt_window wiring is broken\n";
    return false;
  }

  const auto& after_up = layer.w_up();
  const auto& after_enc = layer.w_enc();
  const auto& after_carry = layer.w_carry();

  bool ok = true;
  ok &= check_weight_group("w_up_", 0, base_up, base_enc, base_carry, base_up, after_up, seq);
  ok &= check_weight_group("w_enc_", 1, base_up, base_enc, base_carry, base_enc, after_enc, seq);
  ok &= check_weight_group("w_carry_", 2, base_up, base_enc, base_carry, base_carry, after_carry, seq);
  return ok;
}

}  // namespace

int main() {
  try {
    if (!test_bptt_gradients_match_finite_difference()) {
      return 1;
    }
    std::cout << "rpsm_bptt_grad_finite_diff OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_bptt_grad_finite_diff: " << e.what() << "\n";
    return 1;
  }
}
