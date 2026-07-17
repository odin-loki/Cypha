// rpsm_sequence_smoke — Option B scaffold: standalone layer + CyphaLM rpsm mode train smoke.
#include <cmath>
#include <iostream>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/rpsm/rpsm_sequence_layer.hpp"

namespace {

bool finite_vec(const double* v, int n) {
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) {
      return false;
    }
  }
  return true;
}

bool test_layer_smoke() {
  cypha::rpsm::RpsmSequenceConfig cfg;
  cfg.n_levels = 4;
  cfg.state_dim = 128;
  cfg.feat_dim = 64;
  cfg.n_classes = 64;
  cfg.seed = 7;
  cypha::rpsm::RpsmSequenceLayer layer(cfg);
  std::vector<double> input(static_cast<std::size_t>(cfg.feat_dim), 0.01);
  std::vector<double> log_probs(static_cast<std::size_t>(cfg.n_classes));
  for (int t = 0; t < 128; ++t) {
    input[static_cast<std::size_t>(t % cfg.feat_dim)] = 0.01 * static_cast<double>(t + 1);
    const double h_norm = layer.step(input.data(), cfg.feat_dim, log_probs.data());
    if (!std::isfinite(h_norm) || !finite_vec(log_probs.data(), cfg.n_classes)) {
      std::cerr << "rpsm_sequence_smoke: non-finite layer output at t=" << t << "\n";
      return false;
    }
  }
  return true;
}

bool test_cyphalm_rpsm_smoke() {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 64;
  cfg.d_embed = 32;
  cfg.d_state = 64;
  cfg.field_dim = 64;
  cfg.context_mode = cypha::cyphalm::ContextMode::Rpsm;
  cfg.use_rpsm_layer = true;
  cfg.rpsm_n_levels = 4;
  cfg.rpsm_state_dim = 128;
  cfg.rpsm_feat_dim = 64;
  cfg.train_epochs = 1;
  cfg.seed = 11;

  const auto ids = cypha::cyphalm::synthetic_corpus(120, cfg.vocab_size, cfg.seed);
  cypha::cyphalm::CyphaLMModel model(cfg);
  model.train_sequence(ids, 100, 1);
  const double bpc = model.eval_bpc(ids, 16);
  if (!std::isfinite(bpc) || bpc < 0.0) {
    std::cerr << "rpsm_sequence_smoke: invalid CyphaLM bpc=" << bpc << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  try {
    if (!test_layer_smoke()) {
      return 1;
    }
    if (!test_cyphalm_rpsm_smoke()) {
      return 1;
    }
    std::cout << "rpsm_sequence_smoke OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_sequence_smoke: " << e.what() << "\n";
    return 1;
  }
}
