// hp_gate24_ci_gate — compress-faithful BPC + latency sanity (measured thresholds).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kFixtureBpc = 6.53989;
constexpr double kFixtureBpcTol = 0.05;
// Measured on gate24 table_bits=16 CI fixture (2026-09-20, post #7/#12/#13):
//   Linux KVM 4 vCPU: log_prob_byte ~3.5 ms; next_byte_log_probs(32) ~20 ms
//   GitHub Actions macOS (run 35488004757): log_prob_byte ~10.1 ms
// Limits = measured CI ceiling + ~25% slack (not invented).
constexpr double kLogProbByteMaxUs = 13000.0;
constexpr double kNextByteLogProbs32MaxMs = 30.0;

std::vector<int> fixture_tokens() {
    return {3, 7, 7, 4, 11, 2, 9, 14, 5, 3, 19, 22, 8, 1, 30, 12};
}

double elapsed_us(Clock::time_point t0, int iters) {
    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    return (sec / static_cast<double>(iters)) * 1e6;
}

double elapsed_ms(Clock::time_point t0, int iters) {
    return elapsed_us(t0, iters) / 1000.0;
}

double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const std::size_t mid = v.size() / 2;
    if (v.size() % 2 == 1) {
        return v[mid];
    }
    return 0.5 * (v[mid - 1] + v[mid]);
}

}  // namespace

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.vocab_size = 32;
    cfg.hp_table_bits = 16;
    cypha::cyphalm::apply_hp_production_recipe(cfg);

    cypha::cyphalm::CyphaLMModel model(cfg);
    const auto tokens = fixture_tokens();
    model.train_sequence(tokens, static_cast<int>(tokens.size()) - 1, 1);

    const double bpc = model.eval_bpc(tokens, static_cast<int>(tokens.size()) - 1);
    if (!std::isfinite(bpc)) {
        std::cerr << "hp_gate24_ci_gate FAIL bpc not finite\n";
        return 1;
    }
    if (std::abs(bpc - kFixtureBpc) > kFixtureBpcTol) {
        std::cerr << "hp_gate24_ci_gate FAIL bpc=" << bpc << " expected~=" << kFixtureBpc
                  << " tol=" << kFixtureBpcTol << "\n";
        return 1;
    }

    auto& hp = model.hp_backend();
    for (int w = 0; w < 3; ++w) {
        (void)hp.log_prob_byte(5);
        (void)hp.next_byte_log_probs(32);
    }

    std::vector<double> log_prob_samples;
    for (int i = 0; i < 5; ++i) {
        const auto t0 = Clock::now();
        (void)hp.log_prob_byte(static_cast<std::uint8_t>(5 + (i % 3)));
        log_prob_samples.push_back(elapsed_us(t0, 1));
    }
    const double log_prob_us = median(log_prob_samples);

    std::vector<double> next_samples;
    for (int i = 0; i < 3; ++i) {
        const auto t1 = Clock::now();
        (void)hp.next_byte_log_probs(32);
        next_samples.push_back(elapsed_ms(t1, 1));
    }
    const double next_ms = median(next_samples);

    if (log_prob_us > kLogProbByteMaxUs) {
        std::cerr << "hp_gate24_ci_gate FAIL log_prob_byte_us=" << log_prob_us
                  << " max=" << kLogProbByteMaxUs << "\n";
        return 1;
    }
    if (next_ms > kNextByteLogProbs32MaxMs) {
        std::cerr << "hp_gate24_ci_gate FAIL next_byte_log_probs32_ms=" << next_ms
                  << " max=" << kNextByteLogProbs32MaxMs << "\n";
        return 1;
    }

    std::printf(
        "hp_gate24_ci_gate OK bpc=%.5f log_prob_byte_us=%.0f next_byte_log_probs32_ms=%.0f\n", bpc,
        log_prob_us, next_ms);
    return 0;
}
