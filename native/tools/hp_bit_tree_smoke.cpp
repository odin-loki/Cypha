/// Parity: MSB bit-tree joint log P(byte) vs legacy per-byte clone vs log_prob_byte.
/// On gate24 builds, production scoring uses legacy 256-clone only (DFS pool OOM at v78 scale).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

constexpr double kLog2 = 0.6931471805599453;

}  // namespace

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = 16;
    cypha::cyphalm::CyphaLMModel model(cfg);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::vector<int> warm;
    for (int i = 0; i < 128; ++i) {
        warm.push_back(byte_dist(rng));
    }
    model.reset_context();
    for (int b : warm) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(b));
    }

    auto& hp = model.hp_backend();

#if defined(CYPHA_HP_GATE24)
    const auto legacy = hp.next_byte_log_probs_legacy(256);
    const int truth = byte_dist(rng);
    const double single = hp.log_prob_byte(static_cast<std::uint8_t>(truth));
    const double legacy_t = legacy[static_cast<std::size_t>(truth)];
    const double single_delta = std::abs(single - legacy_t);
    if (single_delta > 1e-6) {
        std::printf("hp_bit_tree_smoke FAIL gate24 legacy single_delta=%.9g\n", single_delta);
        return 1;
    }
    std::puts("hp_bit_tree_smoke OK gate24 (legacy scoring path; bit-tree DFS skipped at v78 scale)");
    return 0;
#else
    const auto tree = hp.next_byte_log_probs_bit_tree(256);
    const auto legacy = hp.next_byte_log_probs_legacy(256);

    double max_delta = 0.0;
    for (int b = 0; b < 256; ++b) {
        const double d = std::abs(tree[static_cast<std::size_t>(b)] -
                                  legacy[static_cast<std::size_t>(b)]);
        max_delta = std::max(max_delta, d);
    }

    const int truth = byte_dist(rng);
    const double single = hp.log_prob_byte(static_cast<std::uint8_t>(truth));
    const double tree_t = tree[static_cast<std::size_t>(truth)];
    const double single_delta = std::abs(single - tree_t);

    if (max_delta > 1e-6 || single_delta > 1e-6) {
        std::printf("hp_bit_tree_smoke FAIL max_tree_legacy_delta=%.9g single_delta=%.9g\n",
                    max_delta, single_delta);
        return 1;
    }

    std::printf("hp_bit_tree_smoke OK max_tree_legacy_delta=%.9g single_delta=%.9g\n", max_delta,
                single_delta);
    return 0;
#endif
}
