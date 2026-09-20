// hp_predictor_assign_parity_smoke — copy_state_from / assign_from_ must match clone_from
// after mutating a reusable loop-local predictor (parity regression).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "hp/predictor.hpp"

namespace {

constexpr double kLogEps = 1e-300;
constexpr double kMaxDeltaNats = 1e-6;

double bit_log_prob(hp::Predictor& snap, int byte) {
    double log_p = 0.0;
    for (int i = 7; i >= 0; --i) {
        const int p12 = snap.predict();
        const int bit = (byte >> i) & 1;
        const double p1 = static_cast<double>(p12) / 4096.0;
        const double p = bit ? p1 : (1.0 - p1);
        log_p += std::log(std::max(p, kLogEps));
        snap.update(bit);
    }
    return log_p;
}

double clone_byte_log_prob(const hp::Predictor& pred, const hp::Config& cfg, int byte) {
    hp::Predictor snap(cfg);
    hp::Predictor::clone_from(pred, snap);
    return bit_log_prob(snap, byte);
}

double assign_byte_log_prob(hp::Predictor& scratch, const hp::Predictor& pred, int byte) {
    scratch.copy_state_from(pred);
    return bit_log_prob(scratch, byte);
}

}  // namespace

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = 16;
    cypha::cyphalm::CyphaLMModel model(cfg);

    std::mt19937 rng(4242);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::vector<int> warm;
    for (int i = 0; i < 256; ++i) {
        warm.push_back(byte_dist(rng));
    }
    model.reset_context();
    for (int b : warm) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(b));
    }

    const hp::Config hp_cfg = cypha::cyphalm::hp_config_from_cyphalm(
        cfg.hp_table_bits, cfg.hp_mixer_lr, cfg.hp_gria);
    const hp::Predictor& pred = model.hp_backend().predictor();
    hp::Predictor scratch(hp_cfg);

    double max_single_delta = 0.0;
    int worst_byte = -1;
    for (int b = 0; b < 256; ++b) {
        const double clone_lp = clone_byte_log_prob(pred, hp_cfg, b);
        const double assign_lp = assign_byte_log_prob(scratch, pred, b);
        const double delta = std::abs(clone_lp - assign_lp);
        if (delta > max_single_delta) {
            max_single_delta = delta;
            worst_byte = b;
        }
    }

    auto& hp = model.hp_backend();
    const auto legacy = hp.next_byte_log_probs_legacy(256);
    const auto assign_path = hp.next_byte_log_probs_assign_reuse(256);
    double max_vocab_delta = 0.0;
    for (int b = 0; b < 256; ++b) {
        const double delta =
            std::abs(legacy[static_cast<std::size_t>(b)] -
                     assign_path[static_cast<std::size_t>(b)]);
        max_vocab_delta = std::max(max_vocab_delta, delta);
    }

    const double log_prob_clone = clone_byte_log_prob(pred, hp_cfg, 173);
    const double log_prob_assign = hp.log_prob_byte(173);
    const double log_prob_delta = std::abs(log_prob_clone - log_prob_assign);

    if (max_single_delta > kMaxDeltaNats || max_vocab_delta > kMaxDeltaNats ||
        log_prob_delta > kMaxDeltaNats) {
        std::printf(
            "hp_predictor_assign_parity_smoke FAIL max_single_delta=%.9g (byte=%d) "
            "max_vocab_delta=%.9g log_prob_delta=%.9g\n",
            max_single_delta, worst_byte, max_vocab_delta, log_prob_delta);
        return 1;
    }

    std::printf(
        "hp_predictor_assign_parity_smoke OK max_single_delta=%.9g max_vocab_delta=%.9g "
        "log_prob_delta=%.9g\n",
        max_single_delta, max_vocab_delta, log_prob_delta);
    return 0;
}
