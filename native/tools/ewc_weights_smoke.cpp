/// Smoke test for hybrid EWC weight Fisher on GRIA U/V and SSM W_fast layer-0.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

void train_sequence(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids, int steps) {
  model.reset_context();
  const int n = std::min(steps, static_cast<int>(ids.size()) - 1);
  for (int i = 0; i < n; ++i) {
    (void)model.train_step(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]),
                           static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i + 1)]));
  }
}

cypha::cyphalm::CyphaLMConfig make_cfg(double ewc_lambda) {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 32;
  cfg.field_dim = 32;
  cfg.d_embed = 16;
  cfg.d_state = 16;
  cfg.ssm_layers = 1;
  cfg.gria_rank = 8;
  cfg.bptt_steps = 1;
  cfg.context_mode = cypha::cyphalm::ContextMode::Hybrid;
  cfg.ewc_lambda = ewc_lambda;
  cfg.gria_lr = 0.06;
  cfg.ssm_lr = 0.25;
  cfg.train_ssm = true;
  cfg.seed = 29;
  return cfg;
}

}  // namespace

int main() {
  const std::vector<int> task_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<int> task_b = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  // Coverage: snapshot after task A must anchor GRIA weights + SSM W_fast/W_slow.
  {
    cypha::cyphalm::CyphaLMModel model(make_cfg(1.0));
    train_sequence(model, task_a, 80);
    model.ewc_snapshot();
    assert(model.hybrid_ewc_regularizer().covers_gria_weights());
    assert(model.hybrid_ewc_regularizer().covers_gria_bias());
    assert(model.hybrid_ewc_regularizer().covers_ssm_w_fast());
    assert(model.hybrid_ewc_regularizer().covers_ssm_w_slow());
    train_sequence(model, task_b, 40);
    const auto m = model.train_step(7, 8);
    if (!(m.ewc_penalty > 0.0)) {
      std::fprintf(stderr, "ewc_weights_smoke: expected ewc_penalty > 0 after snapshot\n");
      return 1;
    }
    std::printf("ewc_weights_smoke: covers_* ok ewc_penalty=%.6e PASS\n", m.ewc_penalty);
  }

  // Second probe: BPTT-2 step still reports a positive penalty.
  {
    cypha::cyphalm::CyphaLMConfig cfg = make_cfg(1.0);
    cfg.bptt_steps = 2;
    cfg.seed = 31;
    cypha::cyphalm::CyphaLMModel model(cfg);
    train_sequence(model, {1, 2, 3, 4, 5, 6}, 6);
    model.ewc_snapshot();
    assert(model.hybrid_ewc_regularizer().covers_gria_weights());
    assert(model.hybrid_ewc_regularizer().covers_gria_bias());
    assert(model.hybrid_ewc_regularizer().covers_ssm_w_fast());
    assert(model.hybrid_ewc_regularizer().covers_ssm_w_slow());
    const auto m = model.train_step(7, 8);
    assert(m.ewc_penalty > 0.0);
  }

  return 0;
}
