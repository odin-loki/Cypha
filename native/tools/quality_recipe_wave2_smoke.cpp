/// Quality Wave 2 smoke: AdamW + LR warmup/cosine + Wave-1 recipe yields finite BPC.
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

int main() {
  // Identity when schedule knobs are off.
  {
    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.lstm_lr = 0.05;
    if (std::abs(cypha::cyphalm::lstm_lr_at_step(cfg, 0) - 0.05) > 1e-15 ||
        std::abs(cypha::cyphalm::lstm_lr_at_step(cfg, 100) - 0.05) > 1e-15) {
      std::printf("quality_recipe_wave2_smoke: FAIL (identity schedule)\n");
      return 1;
    }
  }

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
  cfg.lstm_weight_decay = 1e-4;
  cfg.lstm_lr_warmup_steps = 16;
  cfg.lstm_lr_cosine_steps = 64;

  // Warmup start near-zero, end of warmup at peak, cosine floor at 0.1*lr.
  const double lr0 = cypha::cyphalm::lstm_lr_at_step(cfg, 0);
  const double lr_warm = cypha::cyphalm::lstm_lr_at_step(cfg, 15);
  const double lr_cos0 = cypha::cyphalm::lstm_lr_at_step(cfg, 16);
  const double lr_floor = cypha::cyphalm::lstm_lr_at_step(cfg, 16 + 64);
  if (!(lr0 > 0.0 && lr0 < cfg.lstm_lr && std::abs(lr_warm - cfg.lstm_lr) < 1e-12 &&
        std::abs(lr_cos0 - cfg.lstm_lr) < 1e-12 &&
        std::abs(lr_floor - 0.1 * cfg.lstm_lr) < 1e-12)) {
    std::printf("quality_recipe_wave2_smoke: FAIL (schedule lr0=%g warm=%g cos0=%g floor=%g)\n",
                lr0, lr_warm, lr_cos0, lr_floor);
    return 1;
  }

  cypha::cyphalm::CyphaLMModel model(cfg);
  std::vector<int> ids;
  ids.reserve(512);
  for (int i = 0; i < 512; ++i) {
    ids.push_back((i * 7 + 3) % cfg.vocab_size);
  }

  model.train_sequence(ids, 256, 1, nullptr);
  const double bpc = model.eval_bpc(ids, 64, nullptr);
  if (!std::isfinite(bpc) || bpc <= 0.0 || bpc > 20.0) {
    std::printf("quality_recipe_wave2_smoke: FAIL (bpc=%g)\n", bpc);
    return 1;
  }

  std::printf("quality_recipe_wave2_smoke: PASS (bpc=%.4f lr0=%.6f floor=%.6f)\n", bpc, lr0,
              lr_floor);
  return 0;
}
