/// Smoke: CharLSTM path receives profile-guided navigation loss under --math-integration preset.
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/cyphalm_math_integration.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/intelligence/profile_completeness.hpp"

int main() {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 64;
  cfg.field_dim = 32;
  cfg.d_embed = 32;
  cfg.context_mode = cypha::cyphalm::ContextMode::CharLstm;
  cfg.seed = 7;
  cypha::cyphalm::apply_math_integration_preset(cfg);

  cypha::cyphalm::CyphaLMModel model(cfg);
  std::vector<int> ids;
  for (int i = 0; i < 96; ++i) {
    ids.push_back((i * 5 + 1) % static_cast<int>(cfg.vocab_size));
  }

  cypha::intelligence::IntelligenceProfiler profiler;
  model.train_sequence(ids, 48, 1, &profiler);

  const double bpc = model.eval_bpc(ids, 32, nullptr);
  model.accumulate_intelligence_profile(ids, 32, profiler);
  const auto completeness = cypha::intelligence::validate_profile_completeness(profiler);

  if (!std::isfinite(bpc)) {
    std::puts("navigation_loss_char_lstm_smoke: FAIL (bpc not finite)");
    return 1;
  }
  if (!completeness.all_complete) {
    std::puts("navigation_loss_char_lstm_smoke: FAIL (incomplete profile)");
    return 1;
  }

  std::puts("navigation_loss_char_lstm_smoke: PASS");
  return 0;
}
