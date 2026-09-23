// hp_checkpoint_roundtrip_smoke — binary HPCP v1 predictor checkpoint round-trip.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace {

using cypha::cyphalm::CyphaLMConfig;
using cypha::cyphalm::CyphaLMModel;
using cypha::cyphalm::apply_hp_production_recipe;

std::vector<int> synthetic_ids(int n, int vocab, int seed) {
    std::vector<int> out(static_cast<std::size_t>(n));
    std::uint64_t s = static_cast<std::uint64_t>(seed);
    for (int i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        out[static_cast<std::size_t>(i)] = 1 + static_cast<int>(s % static_cast<std::uint64_t>(vocab - 1));
    }
    return out;
}

bool near(double a, double b, double atol) { return std::abs(a - b) <= atol; }

}  // namespace

int main() {
    CyphaLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.d_embed = 8;
    cfg.d_state = 16;
    cfg.field_dim = 16;
    cfg.hp_table_bits = 16;
    apply_hp_production_recipe(cfg);

    const auto train_ids = synthetic_ids(180, cfg.vocab_size, 42);
    const auto eval_ids = synthetic_ids(64, cfg.vocab_size, 99);

    CyphaLMModel model(cfg);
    model.train_sequence(train_ids, static_cast<int>(train_ids.size()) - 1, 1);
    const double bpc_before = model.eval_bpc(eval_ids, static_cast<int>(eval_ids.size()));

    const std::string base = "/tmp/cyphalm_hp_ckpt_roundtrip";
    std::error_code ec;
    std::filesystem::remove(base + ".json", ec);
    std::filesystem::remove(base + ".hpbin", ec);

    cypha::cyphalm::save_cyphalm_model(model, base);
    CyphaLMModel loaded = cypha::cyphalm::load_cyphalm_model(base + ".json");
    const double bpc_after = loaded.eval_bpc(eval_ids, static_cast<int>(eval_ids.size()));

    if (!near(bpc_before, bpc_after, 1e-9)) {
        std::cerr << "hp_checkpoint_roundtrip_smoke FAIL bpc before=" << bpc_before
                  << " after=" << bpc_after << "\n";
        return 1;
    }

    if (!std::filesystem::exists(base + ".hpbin")) {
        std::cerr << "hp_checkpoint_roundtrip_smoke FAIL missing .hpbin\n";
        return 1;
    }

    // Lossy knobs must survive save/load, and the trained tables must come back
    // into a predictor of the same shape. eval_bpc() resets the model, so compare
    // the live predictors by continuing the stream instead.
    {
        CyphaLMConfig lc = cfg;
        lc.hp_cm_drop = (1ull << 19) | (1ull << 25);  // nestmod, o6b
        lc.hp_cm_bits_cap = 17;
        lc.hp_match_bits_cap = 17;
        lc.hp_pool_slots = 8;
        lc.hp_pool_bits_cap = 16;
        lc.hp_lossy_mem = 17;
        CyphaLMModel lm(lc);
        lm.train_sequence(train_ids, static_cast<int>(train_ids.size()) - 1, 1);
        const std::string lbase = "/tmp/cyphalm_hp_ckpt_roundtrip_lossy";
        std::filesystem::remove(lbase + ".json", ec);
        std::filesystem::remove(lbase + ".hpbin", ec);
        cypha::cyphalm::save_cyphalm_model(lm, lbase);
        CyphaLMModel ll = cypha::cyphalm::load_cyphalm_model(lbase + ".json");
        const auto& c2 = ll.config();
        if (c2.hp_cm_drop != lc.hp_cm_drop || c2.hp_cm_bits_cap != 17 ||
            c2.hp_match_bits_cap != 17 || c2.hp_pool_slots != 8 || c2.hp_pool_bits_cap != 16 ||
            c2.hp_lossy_mem != 17) {
            std::cerr << "hp_checkpoint_roundtrip_smoke FAIL lossy knobs not restored\n";
            return 1;
        }
        std::vector<std::uint8_t> tail(eval_ids.begin(), eval_ids.end());
        const double a = lm.hp_backend().observe_stream_bits(tail.data(), tail.size());
        const double b = ll.hp_backend().observe_stream_bits(tail.data(), tail.size());
        CyphaLMModel fresh(lc);  // control: untrained tables must score differently
        const double f = fresh.hp_backend().observe_stream_bits(tail.data(), tail.size());
        if (near(a, f, 1e-9)) {
            std::cerr << "hp_checkpoint_roundtrip_smoke FAIL control: trained == fresh\n";
            return 1;
        }
        if (!near(a, b, 1e-9)) {
            std::cerr << "hp_checkpoint_roundtrip_smoke FAIL lossy live bits " << a << " vs " << b
                      << "\n";
            return 1;
        }
    }

    std::cout << "hp_checkpoint_roundtrip_smoke OK bpc=" << bpc_before << "\n";
    return 0;
}
