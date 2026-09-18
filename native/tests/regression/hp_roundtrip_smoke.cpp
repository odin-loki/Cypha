// hp_roundtrip_smoke — Cypha hp backend determinism + byte log-prob smoke.
#include <cmath>
#include <iostream>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

using cypha::cyphalm::CyphaLMConfig;
using cypha::cyphalm::CyphaLMModel;
using cypha::cyphalm::apply_hp_production_recipe;

bool finite_log_probs(const std::vector<double>& lp) {
    for (double v : lp) {
        if (!std::isfinite(v)) return false;
    }
    return !lp.empty();
}

}  // namespace

int main() {
    CyphaLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.hp_table_bits = 16;
    apply_hp_production_recipe(cfg);

    CyphaLMModel model(cfg);
    model.reset_context();

    const std::vector<std::uint32_t> tokens = {3, 7, 7, 4, 11, 2, 9, 14, 5, 3};
    double loss_sum = 0.0;
    for (std::size_t t = 0; t + 1 < tokens.size(); ++t) {
        const auto pred = model.predict_next(tokens[t]);
        if (!finite_log_probs(pred.log_probs)) {
            std::cerr << "hp_roundtrip_smoke: non-finite log_probs at step " << t << "\n";
            return 1;
        }
        const auto step = model.train_step(tokens[t], tokens[t + 1]);
        loss_sum += step.loss;
    }

    CyphaLMModel model2(cfg);
    model2.reset_context();
    double loss2 = 0.0;
    for (std::size_t t = 0; t + 1 < tokens.size(); ++t) {
        (void)model2.predict_next(tokens[t]);
        loss2 += model2.train_step(tokens[t], tokens[t + 1]).loss;
    }
    if (std::abs(loss_sum - loss2) > 1e-9) {
        std::cerr << "hp_roundtrip_smoke: determinism fail loss1=" << loss_sum << " loss2=" << loss2
                  << "\n";
        return 1;
    }

    const double bpc = model.eval_bpc(
        std::vector<int>(tokens.begin(), tokens.end()), static_cast<int>(tokens.size()) - 1);
    if (!std::isfinite(bpc)) {
        std::cerr << "hp_roundtrip_smoke: bpc not finite\n";
        return 1;
    }

    std::cout << "hp_roundtrip_smoke OK loss=" << loss_sum << " bpc=" << bpc << "\n";
    return 0;
}
