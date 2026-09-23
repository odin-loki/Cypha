/// Frozen serve: with learning off, consuming bytes must not change anything the
/// model has learned (hp::Predictor::learned_digest), only its context.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
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
    // Frozen scoring: normalised distribution, and scoring changes nothing.
    hp.set_frozen_scoring(true);
    const std::uint64_t before_scoring = hp.predictor().learned_digest();
    const auto lp = hp.next_byte_log_probs(256);
    (void)hp.serve_greedy_next_byte();
    (void)hp.log_prob_byte('t');
    double z = 0.0;
    for (double v : lp) z += std::exp(v);
    if (std::abs(z - 1.0) > 1e-9 || hp.predictor().learned_digest() != before_scoring ||
        !hp.predictor().learning()) {
        std::printf("hp_frozen_smoke FAIL frozen scoring: sum=%.12f digest %s learning %d\n", z,
                    hp.predictor().learned_digest() == before_scoring ? "same" : "changed",
                    hp.predictor().learning() ? 1 : 0);
        return 1;
    }
    hp.set_frozen_scoring(false);

    // Generation must continue from the trained model. prime_serve_context used
    // to call reset_context(), which replaced the predictor with an untrained one.
    {
        cypha::cyphalm::CyphaLMModel model(cfg);
        std::vector<int> ids(text.begin(), text.begin() + 5000);
        for (int b : ids) model.hp_backend().consume_byte(static_cast<std::uint8_t>(b));
        const std::uint64_t trained_model = model.hp_backend().predictor().learned_digest();
        cypha::cyphalm::DecodeParams p;
        p.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
        p.learn_from_output = false;
        std::vector<int> prompt = {'t'};  // one byte: the prompt's last byte is learned
        (void)cypha::cyphalm::generate_decode(model, prompt, 16, p);
        // Only the single prompt byte may have been learned; an untrained predictor
        // would have a completely different digest.
        cypha::cyphalm::CyphaLMModel twin(cfg);
        for (int b : ids) twin.hp_backend().consume_byte(static_cast<std::uint8_t>(b));
        twin.reset_stream();
        twin.serve_advance('t');
        if (model.hp_backend().predictor().learned_digest() !=
                twin.hp_backend().predictor().learned_digest() ||
            trained_model == cypha::cyphalm::CyphaLMModel(cfg).hp_backend().predictor().learned_digest()) {
            std::printf("hp_frozen_smoke FAIL generation did not keep the trained model\n");
            return 1;
        }
    }

    std::printf("hp_frozen_smoke OK digest %016llx unchanged over 2000 frozen bytes\n",
                static_cast<unsigned long long>(trained));
    return 0;
}
