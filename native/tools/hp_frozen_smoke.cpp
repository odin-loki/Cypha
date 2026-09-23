/// Frozen serve: with learning off, consuming bytes must not change anything the
/// model has learned (hp::Predictor::learned_digest), only its context.
#include <cstdio>
#include <random>
#include <string>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.hp_table_bits = 16;
    cfg.hp_cm_bits_cap = 16;
    cfg.hp_match_bits_cap = 16;
    cfg.hp_pool_bits_cap = 16;
    cypha::cyphalm::HpSequenceBackend hp(cypha::cyphalm::hp_config_from_cyphalm(cfg));

    std::string text;
    const char* words[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "Then ", "the ",
                           "dog ", "ran ", "[[link]] ", "{{cite}} ", "\n"};
    std::mt19937 rng(11);
    while (text.size() < 9000) text += words[rng() % 13];
    auto consume = [&](std::size_t a, std::size_t b) {
        for (std::size_t i = a; i < b; ++i) hp.consume_byte(static_cast<std::uint8_t>(text[i]));
    };

    consume(0, 5000);
    const std::uint64_t trained = hp.predictor().learned_digest();

    hp.set_learning(false);
    consume(5000, 7000);
    (void)hp.next_byte_log_probs(256);  // scoring while frozen
    const std::uint64_t after_frozen = hp.predictor().learned_digest();
    hp.set_learning(true);
    consume(7000, 7200);
    const std::uint64_t after_learning = hp.predictor().learned_digest();

    if (after_frozen != trained) {
        std::printf("hp_frozen_smoke FAIL learned state changed while frozen\n");
        return 1;
    }
    if (after_learning == trained) {
        std::printf("hp_frozen_smoke FAIL control: learning on changed nothing\n");
        return 1;
    }
    std::printf("hp_frozen_smoke OK digest %016llx unchanged over 2000 frozen bytes\n",
                static_cast<unsigned long long>(trained));
    return 0;
}
