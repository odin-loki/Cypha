/// Smoke: LM generate self-correct path (Paper IV stub) with epistemic_halt auto-enabled.
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/intelligence/epistemic_threshold.hpp"

int main() {
  cypha::cyphalm::CyphaLMConfig cfg;
  cfg.vocab_size = 64;
  cfg.field_dim = 32;
  cfg.d_embed = 32;
  cfg.context_mode = cypha::cyphalm::ContextMode::Hybrid;
  cfg.seed = 42;
  cypha::cyphalm::CyphaLMModel model(cfg);

  std::vector<int> ids;
  ids.reserve(128);
  for (int i = 0; i < 128; ++i) {
    ids.push_back((i * 7 + 3) % static_cast<int>(cfg.vocab_size));
  }
  model.train_sequence(ids, 64, 1, nullptr);

  cypha::intelligence::EpistemicThreshold threshold(0.35, 5.0);
  cypha::cyphalm::DecodeParams params;
  params.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
  params.self_correct = true;
  if (params.self_correct) {
    params.epistemic_halt = true;
  }

  const cypha::cyphalm::GenerateOutput gen =
      cypha::cyphalm::generate_decode(model, {ids[0]}, 8, params, &threshold);

  if (!gen.generated_ids.empty() || gen.halted_on_epistemic || gen.halted_on_uncertainty ||
      gen.self_corrected) {
    std::puts("lm_self_correct_smoke: PASS");
    return 0;
  }

  std::puts("lm_self_correct_smoke: FAIL (empty generate with no halt)");
  return 1;
}
