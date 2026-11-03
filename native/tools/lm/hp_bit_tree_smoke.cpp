/// Parity: undo round-trip, MSB bit-tree vs legacy fork, assign reuse vs legacy fork,
/// and no leak: scoring the next-byte distribution must not change the live model.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
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
    // gate24 grows some tables +9 bits (capped at 2^24), so a mem-16 predictor
    // is still ~400 MB and the fresh-copy reference paths below copy it 256x.
    // Cap every table at 2^16: same code paths, a fraction of the bytes.
    cfg.hp_cm_bits_cap = 16;
    cfg.hp_match_bits_cap = 16;
    cfg.hp_pool_bits_cap = 16;
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
    const auto reuse = hp.next_byte_log_probs_assign_reuse(256);

    double max_tree_legacy_delta = 0.0;
    double max_assign_legacy_delta = 0.0;
    for (int b = 0; b < 256; ++b) {
        max_tree_legacy_delta =
            std::max(max_tree_legacy_delta,
                     std::abs(tree[static_cast<std::size_t>(b)] -
                              legacy[static_cast<std::size_t>(b)]));
        max_assign_legacy_delta =
            std::max(max_assign_legacy_delta,
                     std::abs(reuse[static_cast<std::size_t>(b)] -
                              legacy[static_cast<std::size_t>(b)]));
    }

    const int truth = byte_dist(rng);
    const double single = hp.log_prob_byte(static_cast<std::uint8_t>(truth));
    const double single_delta =
        std::abs(single - legacy[static_cast<std::size_t>(truth)]);

    if (max_tree_legacy_delta > 1e-6 || max_assign_legacy_delta > 1e-6 ||
        single_delta > 1e-6) {
        std::printf(
            "hp_bit_tree_smoke FAIL max_tree_legacy_delta=%.9g "
            "max_assign_legacy_delta=%.9g single_delta=%.9g\n",
            max_tree_legacy_delta, max_assign_legacy_delta, single_delta);
        return 1;
    }

    // Leak check on text. Word matches and DMC node splits only happen on
    // structured input; before the undo fix a speculative branch could reset
    // the live word match or leave a DMC split behind.
    {
        std::string text;
        const char* words[] = {"the ", "cat ", "sat ", "on ", "a ", "mat ", "and ", "the ",
                               "dog ", "ran ", "[[link]] ", "{{cite}} ", "\n"};
        std::mt19937 wr(7);
        while (text.size() < 6000) text += words[wr() % 13];
        const hp::Config hcfg = cypha::cyphalm::hp_config_from_cyphalm(cfg);
        cypha::cyphalm::HpSequenceBackend served(hcfg), twin(hcfg);
        const std::size_t warm_n = 4000;
        for (std::size_t i = 0; i < warm_n; ++i) {
            served.consume_byte(static_cast<std::uint8_t>(text[i]));
            twin.consume_byte(static_cast<std::uint8_t>(text[i]));
        }
        double max_leak = 0.0;
        double max_parity = 0.0;
        for (std::size_t i = warm_n; i < warm_n + 48; ++i) {
            const auto dist = served.next_byte_log_probs(256);
            if (i == warm_n) {
                const auto ref = served.next_byte_log_probs_assign_reuse(256);
                for (int b = 0; b < 256; ++b) {
                    max_parity = std::max(max_parity, std::abs(dist[static_cast<std::size_t>(b)] -
                                                               ref[static_cast<std::size_t>(b)]));
                }
            }
            const auto nb = static_cast<std::uint8_t>(text[i]);
            max_leak = std::max(max_leak, std::abs(served.observe_next_byte(nb) -
                                                   twin.observe_next_byte(nb)));
        }
        if (max_leak > 0.0 || max_parity > 1e-9) {
            std::printf("hp_bit_tree_smoke FAIL text max_leak=%.9g max_parity=%.9g\n", max_leak,
                        max_parity);
            return 1;
        }
        std::printf("hp_bit_tree_smoke text max_leak=0 max_parity=%.3g\n", max_parity);
    }

    std::printf(
        "hp_bit_tree_smoke OK max_tree_legacy_delta=%.9g max_assign_legacy_delta=%.9g "
        "single_delta=%.9g\n",
        max_tree_legacy_delta, max_assign_legacy_delta, single_delta);
    return 0;
}
