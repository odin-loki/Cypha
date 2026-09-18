/// Smoke: CyphaLM hp backend — finite BPC, production RAM-speed profile, RSS sample.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

long read_vmhwm_kb() {
#if defined(__linux__)
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            long kb = 0;
            if (std::sscanf(line.c_str(), "VmHWM: %ld kB", &kb) == 1) {
                return kb;
            }
        }
    }
#endif
    return -1;
}

}  // namespace

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.vocab_size = 16;
    cfg.hp_table_bits = 16;
    cypha::cyphalm::apply_hp_production_recipe(cfg);

    if (cfg.context_mode != cypha::cyphalm::ContextMode::Hp) {
        std::puts("hp_llm_smoke: FAIL (context_mode != Hp)");
        return 1;
    }
    if (cfg.hp_slot_max != 24) {
        std::printf("hp_llm_smoke: FAIL (hp_slot_max=%d expected 24)\n", cfg.hp_slot_max);
        return 1;
    }
    if (cfg.hp_table_bits != 16) {
        std::printf("hp_llm_smoke: FAIL (hp_table_bits=%d expected 16 for smoke)\n", cfg.hp_table_bits);
        return 1;
    }

    const int compile_slot = cypha::cyphalm::hp_compile_slot_max();
    if (compile_slot != 24 && compile_slot != 35) {
        std::printf("hp_llm_smoke: FAIL (unexpected compile HP_SLOT_MAX=%d)\n", compile_slot);
        return 1;
    }

    cypha::cyphalm::CyphaLMModel model(cfg);
    std::vector<int> ids;
    for (int i = 0; i < 32; ++i) {
        ids.push_back((i * 7 + 3) % static_cast<int>(cfg.vocab_size));
    }
    model.train_sequence(ids, 16, 1, nullptr);
    const double bpc = model.eval_bpc(ids, 8, nullptr);
    if (!std::isfinite(bpc)) {
        std::puts("hp_llm_smoke: FAIL (bpc not finite)");
        return 1;
    }

    const auto profile = model.compression_profile();
    if (!profile.contains("algorithm") || profile.at("algorithm") != "hp") {
        std::puts("hp_llm_smoke: FAIL (compression_profile algorithm != hp)");
        return 1;
    }
    if (profile.at("hp_profile") != "production") {
        std::puts("hp_llm_smoke: FAIL (hp_profile != production)");
        return 1;
    }

    const long rss_kb = read_vmhwm_kb();
    std::printf(
        "hp_llm_smoke OK bpc=%.4f table_bits=%d slot_max=%d compile_slot=%d "
        "rss_kb=%ld lab_ref_slot24_mem22_kb=1600000 lab_ref_slot35_mem22_kb=15000000\n",
        bpc, cfg.hp_table_bits, cfg.hp_slot_max, compile_slot, rss_kb);
    return 0;
}
