#pragma once

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha::cyphalm {

class CharLSTMHead;

/// Per-dimension linear law: gate_pre ≈ w_h·h + w_x·x + w_c·c + bias.
struct SrGateLawCoeff {
    double w_h = 0.0;
    double w_x = 0.0;
    double w_c = 0.0;
    double bias = 0.0;
    double r2 = 0.0;

    double predict(double h, double x, double c) const { return w_h * h + w_x * x + w_c * c + bias; }
};

/// Fitted closed-form laws for i/f/g/o gate pre-activations (H16 scaffold).
struct SrGateLaws {
    int hidden = 0;
    bool fitted = false;
    std::vector<SrGateLawCoeff> i_gate;
    std::vector<SrGateLawCoeff> f_gate;
    std::vector<SrGateLawCoeff> g_gate;
    std::vector<SrGateLawCoeff> o_gate;
};

struct SrGateTraceStep {
    std::vector<double> x;
    std::vector<double> h;
    std::vector<double> c;
    std::vector<double> gates_pre;
};

struct SrGateTrace {
    std::vector<SrGateTraceStep> steps;
};

/// Record gate pre-activations from a short char-LSTM forward trace.
SrGateTrace collect_lstm_gate_trace(const CharLSTMHead& lstm, const std::vector<int>& token_ids, int max_steps);

/// Fit per-dimension linear laws on collected trace (forget gate uses h, x, c).
SrGateLaws fit_sr_gate_laws(const SrGateTrace& trace, int hidden);

nlohmann::json sr_gate_laws_to_json(const SrGateLaws& laws);
SrGateLaws sr_gate_laws_from_json(const nlohmann::json& j);

/// Mean R² across all fitted gate dimensions (0 if not fitted).
double sr_gate_laws_mean_r2(const SrGateLaws& laws);

}  // namespace cypha::cyphalm
