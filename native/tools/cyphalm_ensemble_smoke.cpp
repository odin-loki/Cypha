/// Serve-time ensemble (CyphaLMModel::add_ensemble_member): the mixed
/// distribution is the normalised weighted geometric mean of the members',
/// members advance in step, and word-lookahead generation (exact rewinds over
/// every model) changes nothing learned.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

cypha::cyphalm::CyphaLMConfig small_cfg() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.hp_table_bits = 16;
    cfg.hp_cm_bits_cap = 16;
    cfg.hp_match_bits_cap = 16;
    cfg.hp_pool_bits_cap = 16;
    return cfg;
}

std::string corpus(unsigned seed, int variant) {
    const char* a[] = {"the ", "cat ", "sat ", "on ", "a ", "mat. ", "Then ", "dog ", "ran ", "\n"};
    const char* b[] = {"insects ", "have ", "six ", "legs ", "and ", "wings. ", "The ", "body ", "\n"};
    std::mt19937 rng(seed);
    std::string t;
    while (t.size() < 6000) t += variant == 0 ? a[rng() % 10] : b[rng() % 9];
    return t;
}

void train(cypha::cyphalm::CyphaLMModel& m, const std::string& t) {
    for (char c : t) m.hp_backend().consume_byte(static_cast<std::uint8_t>(c));
}

}  // namespace

int main() {
    const auto cfg = small_cfg();
    const std::string ta = corpus(1, 0), tb = corpus(2, 1);
    const double w = 0.4;

    cypha::cyphalm::CyphaLMModel a(cfg), b(cfg), ens(cfg), mem(cfg);
    train(a, ta);
    train(b, tb);
    train(ens, ta);
    train(mem, tb);
    ens.add_ensemble_member(std::move(mem), w);

    // Mixed distribution == normalised a^(1-w) b^w, over 300 shared bytes.
    const std::string probe = "the insects sat on a mat. The body has six legs and the dog ran.\n";
    for (int step = 0; step < 300; ++step) {
        const auto la = a.hp_backend().serve_next_byte_log_probs(256);
        const auto lb = b.hp_backend().serve_next_byte_log_probs(256);
        const auto le = ens.hp_backend().serve_next_byte_log_probs(256);
        double z = 0.0, sum_e = 0.0, max_err = 0.0;
        for (int i = 0; i < 256; ++i) z += std::exp((1 - w) * la[i] + w * lb[i]);
        for (int i = 0; i < 256; ++i) {
            const double want = (1 - w) * la[i] + w * lb[i] - std::log(z);
            max_err = std::max(max_err, std::abs(want - le[i]));
            sum_e += std::exp(le[i]);
        }
        if (max_err > 1e-9 || std::abs(sum_e - 1.0) > 1e-9) {
            std::printf("cyphalm_ensemble_smoke FAIL mix at step %d: err %.3g sum %.12f\n", step, max_err, sum_e);
            return 1;
        }
        const auto c = static_cast<std::uint8_t>(probe[static_cast<std::size_t>(step) % probe.size()]);
        a.hp_backend().serve_advance_byte(c);
        b.hp_backend().serve_advance_byte(c);
        ens.hp_backend().serve_advance_byte(c);
    }

    // Word lookahead on the ensemble: every model's learned state unchanged
    // except by the one learned prompt byte, i.e. same as its twins' digests.
    const auto preds = ens.hp_backend().all_predictors();
    if (preds.size() != 2) {
        std::printf("cyphalm_ensemble_smoke FAIL expected 2 predictors, got %zu\n", preds.size());
        return 1;
    }
    cypha::cyphalm::DecodeParams p;
    p.word_candidates = 4;
    const auto g = cypha::cyphalm::generate_decode(ens, {'t'}, 60, p);
    for (auto* twin : {&a, &b}) {
        twin->reset_stream(/*keep_history=*/true);
        twin->set_serve_mode(true);
        twin->serve_advance('t');
    }
    if (g.generated_ids.size() != 60 || preds[0]->learned_digest() != a.hp_backend().predictor().learned_digest() ||
        preds[1]->learned_digest() != b.hp_backend().predictor().learned_digest()) {
        std::printf("cyphalm_ensemble_smoke FAIL word lookahead: %zu bytes or learned state changed\n",
                    g.generated_ids.size());
        return 1;
    }
    std::printf("cyphalm_ensemble_smoke OK mix exact over 300 bytes; lookahead kept both models\n");
    return 0;
}
