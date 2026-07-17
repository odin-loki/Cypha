// rpsm_normalized_eta_smoke — Phase -1 Fix 2 regression guard for RPSM_UPGRADE_PLAN.md /
// RESEARCH_STATUS.md:393 ("RPSM core fixes: ... norm eta ...").
//
// RPSM_IMPLEMENTATION.md's Fix 2 describes normalising a *state-update* coefficient
// (`Psi_new = Psi + eta*(E_gated @ W_update)`, RPSM_IMPLEMENTATION.md:51-52) applied to a
// hierarchy-state blend that does not exist as a separate learned matrix in this codebase's
// RpsmSequenceLayer (W_up already serves that role and is trained via the ordinary SGD loop).
// Per RPSM_UPGRADE_PLAN.md Phase -1's own scoping ("scale `lr` in `train_step`"), this
// implementation normalises the *SGD learning rate* used for every RPSM-trained parameter by
// the Frobenius norm of the current step's multi-level hierarchy error, ||E||_F, mirroring the
// spec's eta = eta_base / (||E_gated||_F + eps) formula but applied to the actual training loop
// rather than a separate state-update term. This test checks the two properties that matter for
// *this* mapping of the fix:
//   1. Stability: many steps with use_normalized_eta=true stay finite (no NaN/blowup), including
//      the risk case this implementation is most exposed to -- the hierarchy error shrinking
//      toward 0 late in training, which could blow up 1/(||E||_F+eps) without the eta_norm_max_scale
//      clamp.
//   2. Correctness is preserved: the already-validated Phase 0 property (multiclass training
//      beats a fresh, untrained layer) still holds with both Phase -1 fixes enabled together,
//      i.e. the fixes do not regress the frozen-classifier fix they're layered on top of.
#include <cmath>
#include <iostream>
#include <vector>

#include "cypha/rpsm/rpsm_sequence_layer.hpp"

namespace {

bool test_long_run_stability() {
  cypha::rpsm::RpsmSequenceConfig cfg;
  cfg.n_levels = 4;
  cfg.state_dim = 32;
  cfg.feat_dim = 16;
  cfg.n_classes = 8;
  cfg.n_memory_slots = 8;
  cfg.use_izaac_init = true;
  cfg.use_normalized_eta = true;
  cfg.use_spectral_alpha = true;
  cfg.seed = 11;
  cfg.hierarchy_loss_weight = 0.05;
  cfg.surprise_threshold = 0.01;

  cypha::rpsm::RpsmSequenceLayer layer(cfg);
  std::vector<double> input(static_cast<std::size_t>(cfg.feat_dim), 0.0);

  constexpr int kSteps = 1500;  // long enough for the hierarchy error to plausibly approach 0.
  constexpr double kLr = 0.05;
  constexpr int kNumTargets = 4;

  for (int t = 0; t < kSteps; ++t) {
    // Slowly-decaying-amplitude input: designed to let the hierarchy error shrink over time,
    // which is exactly the regime that would blow up an unclamped 1/(||E||_F+eps) scale.
    const double amp = 1.0 / (1.0 + 0.01 * static_cast<double>(t));
    for (int j = 0; j < cfg.feat_dim; ++j) {
      input[static_cast<std::size_t>(j)] =
          amp * (0.1 * std::sin(0.3 * static_cast<double>(t + j)) +
                 0.05 * static_cast<double>(t % kNumTargets));
    }
    const int target = t % kNumTargets;
    const auto m = layer.train_step(input.data(), cfg.feat_dim, target, kLr);
    if (!std::isfinite(m.loss) || !std::isfinite(m.nll) || !std::isfinite(m.hierarchy_loss) ||
        !std::isfinite(m.surprise)) {
      std::cerr << "rpsm_normalized_eta_smoke: non-finite metrics at t=" << t << " loss=" << m.loss
                << " nll=" << m.nll << " hierarchy_loss=" << m.hierarchy_loss
                << " surprise=" << m.surprise << "\n";
      return false;
    }
    // Bounded-parameter sanity: hidden state passes through tanh (bounded [-1,1]) but the
    // learned matrices (w_up_/w_enc_/w_carry_/Psi_mu) do not -- a runaway effective_lr would
    // show up here as unbounded growth well before any NaN appears.
    double norm_sq = 0.0;
    for (double v : layer.w_up()) {
      norm_sq += v * v;
    }
    if (!(norm_sq < 1.0e8)) {
      std::cerr << "rpsm_normalized_eta_smoke: w_up_ blew up at t=" << t << " ||w_up||^2=" << norm_sq
                << "\n";
      return false;
    }
  }
  return true;
}

bool test_phase0_property_preserved() {
  cypha::rpsm::RpsmSequenceConfig cfg;
  cfg.n_levels = 3;
  cfg.state_dim = 32;
  cfg.feat_dim = 16;
  cfg.n_classes = 8;
  cfg.n_memory_slots = 8;
  cfg.use_izaac_init = true;
  cfg.use_normalized_eta = true;
  cfg.use_spectral_alpha = true;
  cfg.seed = 7;
  cfg.hierarchy_loss_weight = 0.05;
  cfg.surprise_threshold = 0.01;

  cypha::rpsm::RpsmSequenceLayer layer(cfg);
  std::vector<double> input(static_cast<std::size_t>(cfg.feat_dim), 0.0);

  constexpr int kSteps = 400;
  constexpr double kLr = 0.05;
  constexpr int kNumTargets = 4;

  for (int t = 0; t < kSteps; ++t) {
    for (int j = 0; j < cfg.feat_dim; ++j) {
      input[static_cast<std::size_t>(j)] = 0.1 * std::sin(0.3 * static_cast<double>(t + j)) +
                                            0.05 * static_cast<double>(t % kNumTargets);
    }
    const int target = t % kNumTargets;
    const auto m = layer.train_step(input.data(), cfg.feat_dim, target, kLr);
    if (!std::isfinite(m.loss)) {
      std::cerr << "rpsm_normalized_eta_smoke: non-finite loss at t=" << t << "\n";
      return false;
    }
  }

  cypha::rpsm::RpsmSequenceLayer fresh(cfg);
  std::vector<double> log_probs(static_cast<std::size_t>(cfg.n_classes));

  double trained_nll_sum = 0.0;
  double fresh_nll_sum = 0.0;
  constexpr int kEvalSteps = kNumTargets;
  for (int t = 0; t < kEvalSteps; ++t) {
    for (int j = 0; j < cfg.feat_dim; ++j) {
      input[static_cast<std::size_t>(j)] = 0.1 * std::sin(0.3 * static_cast<double>(t + j)) +
                                            0.05 * static_cast<double>(t % kNumTargets);
    }
    const int target = t % kNumTargets;

    layer.step(input.data(), cfg.feat_dim, log_probs.data());
    trained_nll_sum += -log_probs[static_cast<std::size_t>(target)];

    fresh.step(input.data(), cfg.feat_dim, log_probs.data());
    fresh_nll_sum += -log_probs[static_cast<std::size_t>(target)];
  }

  if (!(trained_nll_sum < fresh_nll_sum)) {
    std::cerr << "rpsm_normalized_eta_smoke: Phase 0 property regressed with Phase -1 fixes "
                 "enabled; trained="
              << trained_nll_sum << " fresh=" << fresh_nll_sum << "\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  try {
    if (!test_long_run_stability()) {
      return 1;
    }
    if (!test_phase0_property_preserved()) {
      return 1;
    }
    std::cout << "rpsm_normalized_eta_smoke OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_normalized_eta_smoke: " << e.what() << "\n";
    return 1;
  }
}
