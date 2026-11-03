/// Smoke: CyphaLM serve vs train split — generation without train_step_count growth.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = 16;

    cypha::cyphalm::CyphaLMModel model(cfg);
    const std::vector<int> prompt = {72, 101, 108, 108, 111};  // "Hello"

    const std::uint32_t steps_before = model.train_step_count();
    cypha::cyphalm::DecodeParams greedy;
    greedy.strategy = cypha::cyphalm::DecodeStrategy::Greedy;
    greedy.temperature = 0.0;
    const auto gen = cypha::cyphalm::generate_decode(model, prompt, 8, greedy);
    const std::uint32_t steps_after_gen = model.train_step_count();

    if (steps_after_gen != steps_before) {
        std::printf("cyphalm_serve_smoke FAIL serve generation bumped train_step_count %u -> %u\n",
                    steps_before, steps_after_gen);
        return 1;
    }
    if (gen.generated_ids.empty()) {
        std::puts("cyphalm_serve_smoke FAIL empty generation");
        return 1;
    }

    model.reset_context();
    (void)model.train_step(static_cast<std::uint32_t>(prompt[0]),
                           static_cast<std::uint32_t>(prompt[1]), nullptr);
    if (model.train_step_count() != 1) {
        std::printf("cyphalm_serve_smoke FAIL train_step_count=%u expected 1\n",
                    model.train_step_count());
        return 1;
    }

    model.reset_context();
    for (std::size_t i = 0; i + 1 < prompt.size(); ++i) {
        model.serve_advance(static_cast<std::uint32_t>(prompt[i]));
    }
    const auto pred = model.serve_predict_next(static_cast<std::uint32_t>(prompt.back()));
    if (pred.log_probs.size() != static_cast<std::size_t>(cfg.vocab_size)) {
        std::puts("cyphalm_serve_smoke FAIL serve_predict_next log_probs size");
        return 1;
    }
    double sum = 0.0;
    for (double lp : pred.log_probs) {
        sum += std::exp(lp);
    }
    if (!std::isfinite(sum) || sum < 0.5 || sum > 1.5) {
        std::printf("cyphalm_serve_smoke FAIL prob mass sum=%.6f\n", sum);
        return 1;
    }

    model.reset_context();
    const std::uint32_t g1 = model.serve_greedy_next(static_cast<std::uint32_t>(prompt[0]));
    model.reset_context();
    model.serve_advance(static_cast<std::uint32_t>(prompt[0]));
    const std::uint32_t g2 =
        static_cast<std::uint32_t>(model.hp_backend().serve_greedy_next_byte());
    if (g1 != g2) {
        std::printf("cyphalm_serve_smoke FAIL greedy mismatch %u vs %u\n", g1, g2);
        return 1;
    }

    std::printf("cyphalm_serve_smoke OK generated=%zu greedy=%u train_steps=%u\n",
                gen.generated_ids.size(), g1, model.train_step_count());
    return 0;
}
