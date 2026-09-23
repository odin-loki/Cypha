/// Undo round-trip + bit-tree parity vs legacy clone path.
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
    cfg.hp_frozen_scoring = false;  // this test checks exact (compression-semantics) parity
    cypha::cyphalm::CyphaLMModel model(cfg);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::vector<int> warm;
    warm.reserve(128);
    for (int i = 0; i < 128; ++i) {
        warm.push_back(byte_dist(rng));
    }

    const double bpc_ref =
        model.eval_bpc_compress_equivalent(warm, static_cast<int>(warm.size()));

    model.reset_context();
    for (int b : warm) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(b));
    }

    auto& hp = model.hp_backend();
    hp::Predictor& pred = hp.predictor();
    const int p_before = pred.predict();
    const int c0_before = pred.c0();

    hp::UndoFrame frame;
    pred.update_tracked(1, frame);
    pred.undo_checkpoint(frame);

    if (pred.predict() != p_before || pred.c0() != c0_before) {
        std::printf("hp_undo_smoke FAIL round_trip predict/c0 drift\n");
        return 1;
    }

    for (int trial = 0; trial < 32; ++trial) {
        const int bit = byte_dist(rng) & 1;
        pred.update_tracked(bit, frame);
        pred.undo_checkpoint(frame);
        if (pred.predict() != p_before || pred.c0() != c0_before) {
            std::printf("hp_undo_smoke FAIL drift after trial %d\n", trial);
            return 1;
        }
    }

    const double bpc_after =
        model.eval_bpc_compress_equivalent(warm, static_cast<int>(warm.size()));
    if (std::abs(bpc_ref - bpc_after) > 1e-9) {
        std::printf("hp_undo_smoke FAIL eval_bpc drift ref=%.9f after=%.9f\n", bpc_ref, bpc_after);
        return 1;
    }

    model.reset_context();
    for (int b : warm) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(b));
    }

    {
        hp::Predictor& pred2 = hp.predictor();
        const int p0 = pred2.predict();
        const int c0 = pred2.c0();
        hp::UndoFrame frame2;
        pred2.update_tracked(1, frame2);
        pred2.undo_checkpoint(frame2);
        for (int trial = 0; trial < 32; ++trial) {
            const int bit = byte_dist(rng) & 1;
            pred2.update_tracked(bit, frame2);
            pred2.undo_checkpoint(frame2);
        }
        if (pred2.predict() != p0 || pred2.c0() != c0) {
            std::printf("hp_undo_smoke FAIL tree preamble round_trip drift\n");
            return 1;
        }
    }

    const auto tree = hp.next_byte_log_probs_bit_tree(256);
    const auto legacy = hp.next_byte_log_probs_legacy(256);
    double max_delta = 0.0;
    int worst_byte = -1;
    for (int b = 0; b < 256; ++b) {
        const double d = std::abs(tree[static_cast<std::size_t>(b)] -
                                  legacy[static_cast<std::size_t>(b)]);
        if (d > max_delta) {
            max_delta = d;
            worst_byte = b;
        }
    }

    if (max_delta > 1e-6) {
        std::printf("hp_undo_smoke FAIL max_tree_legacy_delta=%.9g byte=%d\n", max_delta,
                    worst_byte);
        return 1;
    }

    std::printf("hp_undo_smoke OK round_trip max_tree_legacy_delta=%.9g eval_bpc=%.6f\n", max_delta,
                bpc_ref);
    return 0;
}
