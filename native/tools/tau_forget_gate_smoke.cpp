/// Smoke: Paper IV τ/r_eu forget-gate scaling under math-integration preset.
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/cyphalm_math_integration.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

int main() {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 64;
  cfg.field_dim = 32;
  cfg.d_embed = 32;
  cfg.context_mode = cypha::cyphalm::ContextMode::Hybrid;
  cfg.seed = 11;
  cypha::cyphalm::apply_math_integration_preset(cfg);

  if (!cfg.use_tau_forget_gate) {
    std::puts("tau_forget_gate_smoke: FAIL (use_tau_forget_gate not set in math preset)");
    return 1;
  }
  if (!cfg.use_kernel_llr) {
    std::puts("tau_forget_gate_smoke: FAIL (use_kernel_llr not set in math preset)");
    return 1;
  }
  if (std::abs(cfg.per_stat_deviation_span - 1.0) > 1e-9) {
    std::puts("tau_forget_gate_smoke: FAIL (per_stat_deviation_span != 1.0)");
    return 1;
  }

  cypha::cyphalm::CyphaLMModel model(cfg);
  std::vector<int> ids;
  for (int i = 0; i < 96; ++i) {
    ids.push_back((i * 5 + 1) % static_cast<int>(cfg.vocab_size));
  }
  model.train_sequence(ids, 48, 1, nullptr);
  const double bpc = model.eval_bpc(ids, 32, nullptr);
  if (!std::isfinite(bpc)) {
    std::puts("tau_forget_gate_smoke: FAIL (bpc not finite)");
    return 1;
  }

  std::puts("tau_forget_gate_smoke: PASS");
  return 0;
}
