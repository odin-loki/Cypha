/// Quality Wave 1 smoke: opt-in LSTM BPTT>1 + Adam + grad clip + classic init yields finite BPC.
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/cyphalm_model.hpp"

int main() {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 64;
  cfg.field_dim = 32;
  cfg.d_embed = 32;
  cfg.lstm_hidden = 32;
  cfg.context_mode = cypha::cyphalm::ContextMode::CharLstm;
  cfg.seed = 11;
  cfg.lstm_bptt_steps = 8;
  cfg.lstm_optim = "adam";
  cfg.lstm_grad_clip = 1.0;
  cfg.lstm_init = "classic";
  cfg.lstm_lr = 0.01;

  cypha::cyphalm::CyphaLMModel model(cfg);
  std::vector<int> ids;
  ids.reserve(512);
  for (int i = 0; i < 512; ++i) {
    ids.push_back((i * 7 + 3) % cfg.vocab_size);
  }

  model.train_sequence(ids, 256, 1, nullptr);
  const double bpc = model.eval_bpc(ids, 64, nullptr);
  if (!std::isfinite(bpc) || bpc <= 0.0 || bpc > 20.0) {
    std::printf("quality_recipe_wave1_smoke: FAIL (bpc=%g)\n", bpc);
    return 1;
  }

  // Default path (BPTT-1 / SGD / default init) still constructs and trains.
  cypha::cyphalm::CyphaLMConfig cfg_default;
  cfg_default.vocab_size = 64;
  cfg_default.field_dim = 32;
  cfg_default.d_embed = 32;
  cfg_default.lstm_hidden = 32;
  cfg_default.context_mode = cypha::cyphalm::ContextMode::CharLstm;
  cfg_default.seed = 11;
  cypha::cyphalm::CyphaLMModel model_default(cfg_default);
  model_default.train_sequence(ids, 64, 1, nullptr);
  const double bpc_default = model_default.eval_bpc(ids, 32, nullptr);
  if (!std::isfinite(bpc_default)) {
    std::printf("quality_recipe_wave1_smoke: FAIL (default path bpc=%g)\n", bpc_default);
    return 1;
  }

  std::printf("quality_recipe_wave1_smoke: PASS (recipe_bpc=%.4f default_bpc=%.4f)\n", bpc,
              bpc_default);
  return 0;
}
