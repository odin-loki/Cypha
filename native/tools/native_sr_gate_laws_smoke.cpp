/// Smoke test for H16 symbolic-regression gate law fitting on LSTM traces.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cypha/cyphalm/char_lstm.hpp"
#include "cypha/cyphalm/sr_gate_laws.hpp"

namespace {

cypha::cyphalm::SrGateTrace synthetic_linear_trace(int hidden, int steps) {
    cypha::cyphalm::SrGateTrace trace;
    for (int t = 0; t < steps; ++t) {
        cypha::cyphalm::SrGateTraceStep step;
        step.h.assign(static_cast<std::size_t>(hidden), 0.0);
        step.x.assign(static_cast<std::size_t>(hidden), 0.0);
        step.c.assign(static_cast<std::size_t>(hidden), 0.0);
        step.gates_pre.assign(static_cast<std::size_t>(4 * hidden), 0.0);
        for (int j = 0; j < hidden; ++j) {
            const double h = std::sin(static_cast<double>(t) + 0.3 * static_cast<double>(j));
            const double x = std::cos(0.2 * static_cast<double>(t) + 0.5 * static_cast<double>(j));
            const double c = 0.15 * h;
            step.h[static_cast<std::size_t>(j)] = h;
            step.x[static_cast<std::size_t>(j)] = x;
            step.c[static_cast<std::size_t>(j)] = c;
            step.gates_pre[static_cast<std::size_t>(j)] = 0.6 * h + 0.3 * x + 0.1 * h + 0.02;
            step.gates_pre[static_cast<std::size_t>(hidden + j)] = 0.7 * h + 0.4 * x + 0.2 * c + 0.05;
            step.gates_pre[static_cast<std::size_t>(2 * hidden + j)] = -0.5 * h + 0.2 * x + 0.1 * h;
            step.gates_pre[static_cast<std::size_t>(3 * hidden + j)] = 0.3 * h + 0.1 * x + 0.15 * c - 0.1;
        }
        trace.steps.push_back(std::move(step));
    }
    return trace;
}

}  // namespace

int main() {
    const int hidden = 8;
    const int steps = 24;
    const auto trace = synthetic_linear_trace(hidden, steps);
    const cypha::cyphalm::SrGateLaws laws = cypha::cyphalm::fit_sr_gate_laws(trace, hidden);
    assert(laws.fitted);
    assert(static_cast<int>(laws.f_gate.size()) == hidden);

    double min_f_r2 = laws.f_gate[0].r2;
    for (const auto& coeff : laws.f_gate) {
        min_f_r2 = std::min(min_f_r2, coeff.r2);
    }
    assert(min_f_r2 > 0.5);

    const auto json = cypha::cyphalm::sr_gate_laws_to_json(laws);
    assert(json.contains("f_gate"));
    assert(json.contains("mean_r2"));
    const auto restored = cypha::cyphalm::sr_gate_laws_from_json(json);
    assert(restored.fitted);
    assert(restored.f_gate.size() == laws.f_gate.size());

    cypha::cyphalm::CharLSTMHead lstm(32, hidden, 99);
    std::vector<int> tokens;
    for (int i = 0; i < 16; ++i) {
        tokens.push_back(i % 32);
    }
    const auto lstm_trace = cypha::cyphalm::collect_lstm_gate_trace(lstm, tokens, 16);
    assert(!lstm_trace.steps.empty());
    const cypha::cyphalm::SrGateLaws lstm_laws =
        cypha::cyphalm::fit_sr_gate_laws(lstm_trace, hidden);
    assert(lstm_laws.fitted);
    const double mean_r2 = cypha::cyphalm::sr_gate_laws_mean_r2(lstm_laws);

    lstm.set_use_sr_gates(true);
    lstm.set_sr_gate_laws(lstm_laws);
    std::vector<double> log_probs(static_cast<std::size_t>(lstm.vocab_size));
    std::vector<double> h(static_cast<std::size_t>(hidden), 0.0);
    std::vector<double> c(static_cast<std::size_t>(hidden), 0.0);
    std::vector<double> h_out;
    std::vector<double> c_out;
    cypha::cyphalm::CharLSTMCache cache;
    lstm.forward_step(3, h.data(), c.data(), log_probs.data(), h_out, c_out, &cache);
    assert(cache.used_sr_gates);

    std::printf("native_sr_gate_laws_smoke: synthetic_min_f_r2=%.4f lstm_mean_r2=%.4f PASS\n", min_f_r2,
                mean_r2);
    return 0;
}
