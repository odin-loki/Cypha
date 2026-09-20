/// Parity: MSB bit-tree joint log P(byte) vs legacy per-byte clone.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "hp/predictor.hpp"

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

    (void)model.eval_bpc_compress_equivalent(warm, static_cast<int>(warm.size()));

    model.reset_context();
    for (int b : warm) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(b));
    }

    auto& hp = model.hp_backend();
    hp::Predictor& pred = hp.predictor();
    const int p0 = pred.predict();
    const int c0 = pred.c0();
    hp::UndoFrame frame;
    pred.update_tracked(1, frame);
    pred.undo_checkpoint(frame);
    for (int trial = 0; trial < 32; ++trial) {
        const int bit = byte_dist(rng) & 1;
        pred.update_tracked(bit, frame);
        pred.undo_checkpoint(frame);
    }
    if (pred.predict() != p0 || pred.c0() != c0) {
        std::printf("hp_bit_tree_smoke FAIL round_trip drift\n");
        return 1;
    }

    const auto tree = hp.next_byte_log_probs_bit_tree(256);
    const auto legacy = hp.next_byte_log_probs_legacy(256);

    double max_delta = 0.0;
    for (int b = 0; b < 256; ++b) {
        max_delta = std::max(max_delta, std::abs(tree[static_cast<std::size_t>(b)] -
                                                   legacy[static_cast<std::size_t>(b)]));
    }

    if (max_delta > 1e-6) {
        std::printf("hp_bit_tree_smoke FAIL max_tree_legacy_delta=%.9g\n", max_delta);
        return 1;
    }

    std::printf("hp_bit_tree_smoke OK max_tree_legacy_delta=%.9g\n", max_delta);
    return 0;
}
