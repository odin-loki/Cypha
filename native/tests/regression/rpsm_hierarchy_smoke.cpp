// rpsm_hierarchy_smoke — Option B hierarchy: W_up/W_down, multi-level carry, M_slots, Izaac init.
#include <cmath>
#include <iostream>
#include <vector>

#include "cypha/rpsm/rpsm_sequence_layer.hpp"

namespace {

constexpr double kEps = 1e-12;

bool finite_vec(const double* v, int n) {
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) {
      return false;
    }
  }
  return true;
}

bool w_up_nonzero(const std::vector<double>& w_up) {
  double norm_sq = 0.0;
  for (double v : w_up) {
    norm_sq += v * v;
  }
  return norm_sq > 1e-9;
}

bool test_hierarchy_smoke() {
  cypha::rpsm::RpsmSequenceConfig cfg;
  cfg.n_levels = 4;
  cfg.state_dim = 32;
  cfg.feat_dim = 16;
  cfg.n_classes = 32;
  cfg.n_memory_slots = 8;
  cfg.use_izaac_init = true;
  cfg.seed = 19;

  cypha::rpsm::RpsmSequenceLayer layer(cfg);
  if (!w_up_nonzero(layer.w_up())) {
    std::cerr << "rpsm_hierarchy_smoke: W_up not initialized\n";
    return false;
  }

  std::vector<double> input(static_cast<std::size_t>(cfg.feat_dim), 0.01);
  std::vector<double> log_probs(static_cast<std::size_t>(cfg.n_classes));
  for (int t = 0; t < 64; ++t) {
    input[static_cast<std::size_t>(t % cfg.feat_dim)] = 0.01 * static_cast<double>(t + 1);
    const double h_norm = layer.step(input.data(), cfg.feat_dim, log_probs.data());
    if (!std::isfinite(h_norm) || !finite_vec(log_probs.data(), cfg.n_classes)) {
      std::cerr << "rpsm_hierarchy_smoke: non-finite output at t=" << t << "\n";
      return false;
    }
  }

  if (layer.all_levels().size() != static_cast<std::size_t>(cfg.n_levels)) {
    std::cerr << "rpsm_hierarchy_smoke: level count mismatch\n";
    return false;
  }

  double l0_norm = 0.0;
  double l3_norm = 0.0;
  for (double v : layer.level_hidden(0)) {
    l0_norm += v * v;
  }
  for (double v : layer.level_hidden(3)) {
    l3_norm += v * v;
  }
  if (l0_norm <= kEps || l3_norm <= kEps) {
    std::cerr << "rpsm_hierarchy_smoke: level norms too small\n";
    return false;
  }

  bool slots_written = false;
  for (double v : layer.global_memory().slots()) {
    if (std::abs(v) > 1e-9) {
      slots_written = true;
      break;
    }
  }
  if (!slots_written) {
    std::cerr << "rpsm_hierarchy_smoke: global memory slots never written\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  try {
    if (!test_hierarchy_smoke()) {
      return 1;
    }
    std::cout << "rpsm_hierarchy_smoke OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_hierarchy_smoke: " << e.what() << "\n";
    return 1;
  }
}
