// cyphalm_checkpoint_golden — hp config checkpoint save/load roundtrip.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace fs = std::filesystem;

namespace {

using cypha::cyphalm::ContextMode;
using cypha::cyphalm::CyphaLMConfig;
using cypha::cyphalm::CyphaLMModel;

std::vector<int> synthetic_ids(int n, int vocab, int seed) {
    std::vector<int> out(static_cast<std::size_t>(n));
    std::uint64_t s = static_cast<std::uint64_t>(seed);
    for (int i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        out[static_cast<std::size_t>(i)] = 1 + static_cast<int>(s % static_cast<std::uint64_t>(vocab - 1));
    }
    return out;
}

double eval_bpc(CyphaLMModel& model, const std::vector<int>& ids, int n_eval) {
    return model.eval_bpc(ids, n_eval);
}

bool near(double a, double b, double atol) { return std::abs(a - b) <= atol; }

int test_hp_roundtrip(const char* label) {
    CyphaLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.d_embed = 8;
    cfg.d_state = 16;
    cfg.field_dim = 16;
    cfg.context_mode = ContextMode::Hp;
    cfg.hp_table_bits = 16;
    cfg.hp_mixer_lr = 2;
    cfg.seed = 42;

    const auto train_ids = synthetic_ids(220, cfg.vocab_size, 42);
    const auto eval_ids = synthetic_ids(80, cfg.vocab_size, 99);

    CyphaLMModel model(cfg);
    model.train_sequence(train_ids, static_cast<int>(train_ids.size()) - 1, 1);
    const double bpc_before = eval_bpc(model, eval_ids, static_cast<int>(eval_ids.size()) - 1);

    const std::string base = std::string("C:/Temp/cyphalm_ckpt_golden_") + label;
    model.save(base);
    CyphaLMModel loaded = cypha::cyphalm::load_cyphalm_model(base + ".json");
    const double bpc_after = eval_bpc(loaded, eval_ids, static_cast<int>(eval_ids.size()) - 1);

    if (!near(bpc_before, bpc_after, 1e-9)) {
        std::cerr << "FAIL " << label << " roundtrip bpc before=" << bpc_before << " after=" << bpc_after
                  << "\n";
        return 1;
    }
    std::cout << "OK " << label << " roundtrip bpc=" << bpc_before << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    int failures = 0;
    failures += test_hp_roundtrip("hp") != 0 ? 1 : 0;

    if (failures == 0) {
        std::cout << "All cyphalm_checkpoint_golden checks PASSED.\n";
        return 0;
    }
    std::cerr << failures << " cyphalm_checkpoint_golden check(s) FAILED.\n";
    return 1;
}
