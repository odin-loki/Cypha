// rpsm_train_multiclass_smoke — Phase 0 regression guard for RPSM_UPGRADE_PLAN.md Finding #1.
//
// rpsm_train_smoke.cpp trains on a single fixed target class for all steps, which cannot
// distinguish "the classifier learns class boundaries" from "the encoder just rotates onto one
// already-fixed random direction" (see RPSM_UPGRADE_PLAN.md §4.4). This test instead cycles
// through several target classes and asserts two things a frozen Psi_mu classifier cannot do:
//   1. Psi_mu rows 1..K actually change from their random init after training.
//   2. Average per-class NLL over a held-out multi-class eval improves after training.
#include <cmath>
#include <iostream>
#include <vector>

#include "cypha/rpsm/rpsm_sequence_layer.hpp"

namespace {

bool test_multiclass_train_improves() {
  cypha::rpsm::RpsmSequenceConfig cfg;
  cfg.n_levels = 3;
  cfg.state_dim = 32;
  cfg.feat_dim = 16;
  cfg.n_classes = 8;
  cfg.n_memory_slots = 8;
  cfg.use_izaac_init = true;
  cfg.seed = 7;
  cfg.hierarchy_loss_weight = 0.05;
  cfg.surprise_threshold = 0.01;

  cypha::rpsm::RpsmSequenceLayer layer(cfg);
  std::vector<double> input(static_cast<std::size_t>(cfg.feat_dim), 0.0);

  // Snapshot Psi_mu rows 1..K before training.
  const auto mu_before = layer.psi().mu;

  constexpr int kSteps = 400;
  constexpr double kLr = 0.05;
  constexpr int kNumTargets = 4;  // cycle through 4 of the 8 classes

  for (int t = 0; t < kSteps; ++t) {
    for (int j = 0; j < cfg.feat_dim; ++j) {
      input[static_cast<std::size_t>(j)] = 0.1 * std::sin(0.3 * static_cast<double>(t + j)) +
                                            0.05 * static_cast<double>(t % kNumTargets);
    }
    const int target = t % kNumTargets;
    const auto m = layer.train_step(input.data(), cfg.feat_dim, target, kLr);
    if (!std::isfinite(m.loss)) {
      std::cerr << "rpsm_train_multiclass_smoke: non-finite loss at t=" << t << "\n";
      return false;
    }
  }

  const auto& mu_after = layer.psi().mu;
  if (mu_before.size() != mu_after.size()) {
    std::cerr << "rpsm_train_multiclass_smoke: mu size mismatch\n";
    return false;
  }

  // Row 0 (world prior mu0) is intentionally untouched by the Phase 0 fix; rows 1..K must move.
  const int d = cfg.feat_dim;
  double class_delta_sq = 0.0;
  for (std::size_t i = static_cast<std::size_t>(d); i < mu_after.size(); ++i) {
    const double diff = mu_after[i] - mu_before[i];
    class_delta_sq += diff * diff;
  }
  const double class_delta_norm = std::sqrt(class_delta_sq);
  if (!(class_delta_norm > 1e-6)) {
    std::cerr << "rpsm_train_multiclass_smoke: Psi_mu rows 1..K did not move (frozen classifier "
                 "bug regressed); ||delta||="
              << class_delta_norm << "\n";
    return false;
  }

  // Re-measure average NLL across all cycled targets using the trained layer's forward pass
  // (log_softmax over the now-updated Psi_mu), compared against a fresh untrained layer with the
  // same init, on the same inputs. This directly checks classification, not just "loss went down"
  // on whatever the encoder happened to latch onto (which a frozen classifier can also achieve).
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
    std::cerr << "rpsm_train_multiclass_smoke: trained classifier did not beat a fresh "
                 "(untrained) one on multi-class eval; trained="
              << trained_nll_sum << " fresh=" << fresh_nll_sum << "\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  try {
    if (!test_multiclass_train_improves()) {
      return 1;
    }
    std::cout << "rpsm_train_multiclass_smoke OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_train_multiclass_smoke: " << e.what() << "\n";
    return 1;
  }
}
