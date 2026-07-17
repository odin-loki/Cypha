// rpsm_train_smoke — Option B train loop: 20 SGD steps, loss decreases.
#include <cmath>
#include <iostream>
#include <vector>

#include "cypha/rpsm/rpsm_sequence_layer.hpp"

namespace {

bool test_train_loss_decreases() {
  cypha::rpsm::RpsmSequenceConfig cfg;
  cfg.n_levels = 3;
  cfg.state_dim = 32;
  cfg.feat_dim = 16;
  cfg.n_classes = 16;
  cfg.n_memory_slots = 8;
  cfg.use_izaac_init = true;
  cfg.seed = 23;
  cfg.hierarchy_loss_weight = 0.05;
  cfg.surprise_threshold = 0.01;

  cypha::rpsm::RpsmSequenceLayer layer(cfg);
  std::vector<double> input(static_cast<std::size_t>(cfg.feat_dim), 0.0);

  constexpr int kSteps = 20;
  constexpr double kLr = 0.05;
  constexpr int kTarget = 3;

  double first_loss = 0.0;
  double last_loss = 0.0;
  for (int t = 0; t < kSteps; ++t) {
    for (int j = 0; j < cfg.feat_dim; ++j) {
      input[static_cast<std::size_t>(j)] = 0.1 * std::sin(0.3 * static_cast<double>(t + j));
    }
    const auto m = layer.train_step(input.data(), cfg.feat_dim, kTarget, kLr);
    if (!std::isfinite(m.loss)) {
      std::cerr << "rpsm_train_smoke: non-finite loss at t=" << t << "\n";
      return false;
    }
    if (t == 0) {
      first_loss = m.loss;
    }
    last_loss = m.loss;
  }

  if (!(last_loss < first_loss)) {
    std::cerr << "rpsm_train_smoke: loss did not decrease first=" << first_loss << " last=" << last_loss
              << "\n";
    return false;
  }

  if (layer.activation_mix() == cypha::rpsm::IzaacActivationMix::TanhOnly && cfg.seed == 23) {
    // seed 23 % 5 == 3 -> ReluTanh
    std::cerr << "rpsm_train_smoke: unexpected activation mix for seed\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  try {
    if (!test_train_loss_decreases()) {
      return 1;
    }
    std::cout << "rpsm_train_smoke OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_train_smoke: " << e.what() << "\n";
    return 1;
  }
}
