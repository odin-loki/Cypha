/// Smoke: CyphaLM hp backend — finite BPC, config, compression profile.
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.vocab_size = 64;
    cfg.hp_table_bits = 18;
    cypha::cyphalm::apply_hp_production_recipe(cfg);

    if (cfg.context_mode != cypha::cyphalm::ContextMode::Hp) {
        std::puts("hp_llm_smoke: FAIL (context_mode != Hp)");
        return 1;
    }

    cypha::cyphalm::CyphaLMModel model(cfg);
    std::vector<int> ids;
    for (int i = 0; i < 96; ++i) {
        ids.push_back((i * 7 + 3) % static_cast<int>(cfg.vocab_size));
    }
    model.train_sequence(ids, 48, 1, nullptr);
    const double bpc = model.eval_bpc(ids, 32, nullptr);
    if (!std::isfinite(bpc)) {
        std::puts("hp_llm_smoke: FAIL (bpc not finite)");
        return 1;
    }

    const auto profile = model.compression_profile();
    if (!profile.contains("algorithm") || profile.at("algorithm") != "hp") {
        std::puts("hp_llm_smoke: FAIL (compression_profile algorithm != hp)");
        return 1;
    }

    std::printf("hp_llm_smoke OK bpc=%.4f table_bits=%d\n", bpc, cfg.hp_table_bits);
    return 0;
}
