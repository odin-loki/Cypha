/// Smoke: hybrid LSTM receives direct logit navigation grads under math-integration preset.
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
  cfg.context_mode = cypha::cyphalm::ContextMode::Hybrid;
  cfg.seed = 11;
  cypha::cyphalm::apply_math_integration_preset(cfg);

  cypha::cyphalm::CyphaLMModel model(cfg);
  const double blend_before = model.hybrid_blend_logit();

  std::vector<int> ids;
  for (int i = 0; i < 96; ++i) {
    ids.push_back((i * 7 + 2) % static_cast<int>(cfg.vocab_size));
  }

  cypha::intelligence::IntelligenceProfiler profiler;
  model.train_sequence(ids, 48, 1, &profiler);

  const double blend_after = model.hybrid_blend_logit();
  const double bpc = model.eval_bpc(ids, 32, nullptr);
  model.accumulate_intelligence_profile(ids, 32, profiler);
  const auto completeness = cypha::intelligence::validate_profile_completeness(profiler);

  if (!std::isfinite(bpc)) {
    std::puts("navigation_loss_hybrid_smoke: FAIL (bpc not finite)");
    return 1;
  }
  if (!completeness.all_complete) {
    std::puts("navigation_loss_hybrid_smoke: FAIL (incomplete profile)");
    return 1;
  }
  if (std::abs(blend_after - blend_before) < 1e-12) {
    std::puts("navigation_loss_hybrid_smoke: FAIL (hybrid_blend_logit unchanged)");
    return 1;
  }

  std::puts("navigation_loss_hybrid_smoke: PASS");
  return 0;
}
