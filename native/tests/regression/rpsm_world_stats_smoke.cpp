// rpsm_world_stats_smoke — §15 (RPSM_UPGRADE_PLAN.md) regression guard for the online mu0/inv_var
// world-stats update ported from `CyphaDifMemoryState::memory_train`'s `world_update()`
// (native/src/memory_train.cpp:41-121).
//
// Three checks:
//   1. Parity: an independent, from-scratch reimplementation of the Welford (n<=20) / EMA (n>20)
//      recurrence, driven by the exact `feat()` vectors `RpsmSequenceLayer::train_step` actually
//      produced (captured after each call), must match `psi().mu`'s row 0 (`mu0`) and
//      `psi().inv_var` to near machine precision at several checkpoints spanning both regimes.
//      This is the "gold standard" cross-check this document's own §10.3/§14.2 sections use for
//      new numeric code paths, adapted for a plain recurrence rather than a gradient.
//   2. Stability: many steps with the flag enabled stay finite (no NaN/blowup), including the
//      EMA regime's own risk case (inv_var = 1/v could blow up if v collapsed toward 0; the
//      kWorldMinVar floor, ported from memory_train.cpp, is what prevents that).
//   3. Correctness preserved: the already-validated Phase 0 property (multiclass training beats
//      a fresh, untrained layer) still holds with the new flag enabled, i.e. it does not regress
//      the frozen-classifier fix it's layered on top of.
#include <cmath>
#include <iostream>
#include <vector>

#include "cypha/rpsm/rpsm_sequence_layer.hpp"

namespace {

constexpr double kWorldMinVarRef = 1e-4;
constexpr int kWorldWelfordStepsRef = 20;

bool test_matches_independent_reference() {
  cypha::rpsm::RpsmSequenceConfig cfg;
  cfg.n_levels = 3;
  cfg.state_dim = 24;
  cfg.feat_dim = 12;
  cfg.n_classes = 6;
  cfg.n_memory_slots = 8;
  cfg.use_izaac_init = true;
  cfg.use_online_world_stats = true;
  cfg.world_stats_lr = 0.008;
  cfg.seed = 13;
  cfg.hierarchy_loss_weight = 0.05;
  cfg.surprise_threshold = 0.01;

  cypha::rpsm::RpsmSequenceLayer layer(cfg);
  std::vector<double> input(static_cast<std::size_t>(cfg.feat_dim), 0.0);

  const int d = cfg.feat_dim;
  // Independent from-scratch reference, ported directly from memory_train.cpp:41-121 (not by
  // calling into the production code at all -- a fresh reimplementation of the same formula).
  std::vector<double> ref_mu0(static_cast<std::size_t>(d), 0.0);
  std::vector<double> ref_m2(static_cast<std::size_t>(d), 1.0);
  std::vector<double> ref_v(static_cast<std::size_t>(d), 1.0);
  std::int64_t ref_n = 0;

  // mu0 starts at the layer's own random init (N(0, 0.05)), not at 0 -- snapshot it so the
  // reference recurrence starts from the same point the production code does.
  {
    const auto& mu_init = layer.psi().mu;
    for (int j = 0; j < d; ++j) {
      ref_mu0[static_cast<std::size_t>(j)] = mu_init[static_cast<std::size_t>(j)];
    }
  }

  constexpr int kSteps = 30;  // spans both the Welford (<=20) and EMA (>20) regimes.
  constexpr double kLr = 0.05;
  constexpr int kNumTargets = 4;

  auto check_step = [&](int t) -> bool {
    ++ref_n;
    const auto& feat = layer.feat();
    if (static_cast<int>(feat.size()) != d) {
      std::cerr << "rpsm_world_stats_smoke: feat() size mismatch at t=" << t << "\n";
      return false;
    }
    if (ref_n <= kWorldWelfordStepsRef) {
      for (int j = 0; j < d; ++j) {
        const double delta0 = feat[static_cast<std::size_t>(j)] - ref_mu0[static_cast<std::size_t>(j)];
        ref_mu0[static_cast<std::size_t>(j)] += delta0 / static_cast<double>(ref_n);
        ref_m2[static_cast<std::size_t>(j)] +=
            delta0 * (feat[static_cast<std::size_t>(j)] - ref_mu0[static_cast<std::size_t>(j)]);
      }
      if (ref_n > 1) {
        for (int j = 0; j < d; ++j) {
          ref_v[static_cast<std::size_t>(j)] = std::max(
              ref_m2[static_cast<std::size_t>(j)] / static_cast<double>(ref_n - 1), kWorldMinVarRef);
        }
      }
    } else {
      const double lr = cfg.world_stats_lr;
      for (int j = 0; j < d; ++j) {
        const double delta = feat[static_cast<std::size_t>(j)] - ref_mu0[static_cast<std::size_t>(j)];
        ref_mu0[static_cast<std::size_t>(j)] += lr * delta;
        double vj = (1.0 - lr) * ref_v[static_cast<std::size_t>(j)] + lr * delta * delta;
        ref_v[static_cast<std::size_t>(j)] = std::max(vj, kWorldMinVarRef);
      }
    }

    const auto& psi = layer.psi();
    double max_mu_err = 0.0;
    double max_inv_var_err = 0.0;
    for (int j = 0; j < d; ++j) {
      max_mu_err = std::max(max_mu_err,
                             std::fabs(psi.mu[static_cast<std::size_t>(j)] - ref_mu0[static_cast<std::size_t>(j)]));
      const double ref_inv_var = std::min(1.0 / ref_v[static_cast<std::size_t>(j)], cfg.inv_var_max_scale);
      max_inv_var_err =
          std::max(max_inv_var_err, std::fabs(psi.inv_var[static_cast<std::size_t>(j)] - ref_inv_var));
    }
    if (!(max_mu_err < 1e-9)) {
      std::cerr << "rpsm_world_stats_smoke: mu0 mismatch at t=" << t << " max_err=" << max_mu_err << "\n";
      return false;
    }
    if (!(max_inv_var_err < 1e-9)) {
      std::cerr << "rpsm_world_stats_smoke: inv_var mismatch at t=" << t
                << " max_err=" << max_inv_var_err << "\n";
      return false;
    }
    return true;
  };

  for (int t = 0; t < kSteps; ++t) {
    for (int j = 0; j < cfg.feat_dim; ++j) {
      input[static_cast<std::size_t>(j)] = 0.1 * std::sin(0.3 * static_cast<double>(t + j)) +
                                            0.05 * static_cast<double>(t % kNumTargets);
    }
    const int target = t % kNumTargets;
    const auto m = layer.train_step(input.data(), cfg.feat_dim, target, kLr);
    if (!std::isfinite(m.loss)) {
      std::cerr << "rpsm_world_stats_smoke: non-finite loss at t=" << t << "\n";
      return false;
    }
    if (!check_step(t)) {
      return false;
    }
  }
  return true;
}

bool test_long_run_stability() {
  cypha::rpsm::RpsmSequenceConfig cfg;
  cfg.n_levels = 4;
  cfg.state_dim = 32;
  cfg.feat_dim = 16;
  cfg.n_classes = 8;
  cfg.n_memory_slots = 8;
  cfg.use_izaac_init = true;
  cfg.use_online_world_stats = true;
  cfg.use_normalized_eta = true;
  cfg.use_spectral_alpha = true;
  cfg.seed = 11;
  cfg.hierarchy_loss_weight = 0.05;
  cfg.surprise_threshold = 0.01;

  cypha::rpsm::RpsmSequenceLayer layer(cfg);
  std::vector<double> input(static_cast<std::size_t>(cfg.feat_dim), 0.0);

  constexpr int kSteps = 1500;
  constexpr double kLr = 0.05;
  constexpr int kNumTargets = 4;

  for (int t = 0; t < kSteps; ++t) {
    const double amp = 1.0 / (1.0 + 0.01 * static_cast<double>(t));
    for (int j = 0; j < cfg.feat_dim; ++j) {
      input[static_cast<std::size_t>(j)] =
          amp * (0.1 * std::sin(0.3 * static_cast<double>(t + j)) +
                 0.05 * static_cast<double>(t % kNumTargets));
    }
    const int target = t % kNumTargets;
    const auto m = layer.train_step(input.data(), cfg.feat_dim, target, kLr);
    if (!std::isfinite(m.loss) || !std::isfinite(m.nll)) {
      std::cerr << "rpsm_world_stats_smoke: non-finite metrics at t=" << t << " loss=" << m.loss
                << " nll=" << m.nll << "\n";
      return false;
    }
    for (double v : layer.psi().inv_var) {
      if (!std::isfinite(v) || v > 1.0e8) {
        std::cerr << "rpsm_world_stats_smoke: inv_var blew up at t=" << t << " v=" << v << "\n";
        return false;
      }
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
  cfg.use_online_world_stats = true;
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
      std::cerr << "rpsm_world_stats_smoke: non-finite loss at t=" << t << "\n";
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
    std::cerr << "rpsm_world_stats_smoke: Phase 0 property regressed with §15 enabled; trained="
              << trained_nll_sum << " fresh=" << fresh_nll_sum << "\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  try {
    if (!test_matches_independent_reference()) {
      return 1;
    }
    if (!test_long_run_stability()) {
      return 1;
    }
    if (!test_phase0_property_preserved()) {
      return 1;
    }
    std::cout << "rpsm_world_stats_smoke OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_world_stats_smoke: " << e.what() << "\n";
    return 1;
  }
}
