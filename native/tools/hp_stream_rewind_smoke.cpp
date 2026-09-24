/// hp::StreamRewind: after frozen advances (with bit-tree scoring nested inside),
/// rewind() must return the predictor bit-for-bit to its saved state. Checked by
/// comparing full checkpoints, then by continuing both it and an untouched copy.
/// Run for gate24 and again with every optional upstream context model on
/// (hp_extra_cms = 127: their tables and wiki trackers must rewind too).
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "hp/checkpoint.hpp"

namespace {

std::string serialize(hp::Predictor& p) {
    (void)p.predict();  // per-bit scratch is recomputed by predict; compare after it
    std::ostringstream os;
    p.write_checkpoint(os);
    return os.str();
}

int run(std::uint32_t extra_cms) {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.hp_extra_cms = extra_cms;
    cfg.hp_table_bits = 16;
    cfg.hp_cm_bits_cap = 16;
    cfg.hp_match_bits_cap = 16;
    cfg.hp_pool_bits_cap = 16;
    cypha::cyphalm::HpSequenceBackend hp(cypha::cyphalm::hp_config_from_cyphalm(cfg));

    std::string text;
    const char* words[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "Then ", "the ",
                           "dog ", "ran ", "[[link]] ", "{{cite}} ", "\n", "1984 ", "'''x''' ",
                           "''y'' ", "<ref name=\"a\">", "<ref>", "<ref group=n>", "</ref> ",
                           "<ref name=b/> ", "<page>", "</page>\n"};
    const int nwords = extra_cms != 0 ? 23 : 15;
    std::mt19937 rng(7);
    while (text.size() < 12000) text += words[rng() % nwords];
    for (std::size_t i = 0; i < 8000; ++i) hp.consume_byte(static_cast<std::uint8_t>(text[i]));

    hp.set_learning(false);
    hp::Predictor& pred = hp.predictor();
    const std::string before = serialize(pred);
    auto twin = hp.predictor_snapshot();

    {
        hp::StreamRewind rw(pred);
        for (int trial = 0; trial < 40; ++trial) {
            const int n = 1 + static_cast<int>(rng() % 40);
            for (int k = 0; k < n; ++k) {
                if (k % 3 == 0) (void)cypha::cyphalm::HpSequenceBackend::byte_log_probs_bit_tree(pred, 256);
                const std::uint8_t b = (rng() % 4 == 0) ? static_cast<std::uint8_t>(rng() % 256)
                                                         : static_cast<std::uint8_t>(text[8000 + rng() % 4000]);
                cypha::cyphalm::HpSequenceBackend::consume_byte_on(pred, b);
            }
            rw.rewind();
            if (serialize(pred) != before) {
                std::printf("hp_stream_rewind_smoke FAIL state differs after rewind (trial %d, %d bytes)\n",
                            trial, n);
                return 1;
            }
        }
    }

    // Continue both on the same text: distributions must match exactly.
    for (std::size_t i = 8000; i < 9000; ++i) {
        const auto a = cypha::cyphalm::HpSequenceBackend::byte_log_probs_bit_tree(pred, 256);
        const auto b = cypha::cyphalm::HpSequenceBackend::byte_log_probs_bit_tree(*twin, 256);
        if (a != b) {
            std::printf("hp_stream_rewind_smoke FAIL distributions differ at byte %zu\n", i);
            return 1;
        }
        cypha::cyphalm::HpSequenceBackend::consume_byte_on(pred, static_cast<std::uint8_t>(text[i]));
        cypha::cyphalm::HpSequenceBackend::consume_byte_on(*twin, static_cast<std::uint8_t>(text[i]));
    }
    // Serve path: frozen scoring (the generation default) inside the rewind.
    hp.set_frozen_scoring(true);
    const std::string before_serve = serialize(pred);
    {
        hp::StreamRewind rw(pred);
        for (int trial = 0; trial < 20; ++trial) {
            const int n = 1 + static_cast<int>(rng() % 24);
            for (int k = 0; k < n; ++k) {
                (void)hp.serve_next_byte_log_probs(256);
                hp.serve_advance_byte(static_cast<std::uint8_t>(text[9000 + rng() % 3000]));
            }
            rw.rewind();
            if (serialize(pred) != before_serve) {
                std::printf("hp_stream_rewind_smoke FAIL serve-path state differs after rewind (trial %d)\n", trial);
                return 1;
            }
        }
    }
    std::printf("hp_stream_rewind_smoke OK extra_cms=%u: 60 rewinds exact, 1000 bytes identical after\n",
                extra_cms);
    return 0;
}

}  // namespace

int main() {
    if (run(0) != 0) return 1;
    return run(127);
}
