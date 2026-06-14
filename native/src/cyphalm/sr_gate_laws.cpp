#include "cypha/cyphalm/sr_gate_laws.hpp"

#include "cypha/cyphalm/char_lstm.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cypha::cyphalm {

namespace {

constexpr int kFeat = 4;

bool solve_linear_system(double a[4][4], double b[4], double out[4]) {
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        double pivot_val = std::abs(a[col][col]);
        for (int row = col + 1; row < 4; ++row) {
            const double v = std::abs(a[row][col]);
            if (v > pivot_val) {
                pivot_val = v;
                pivot = row;
            }
        }
        if (pivot_val < 1e-14) {
            return false;
        }
        if (pivot != col) {
            for (int k = 0; k < 4; ++k) {
                std::swap(a[col][k], a[pivot][k]);
            }
            std::swap(b[col], b[pivot]);
        }
        const double inv = 1.0 / a[col][col];
        for (int k = 0; k < 4; ++k) {
            a[col][k] *= inv;
        }
        b[col] *= inv;
        for (int row = 0; row < 4; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = a[row][col];
            if (factor == 0.0) {
                continue;
            }
            for (int k = 0; k < 4; ++k) {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = b[i];
    }
    return true;
}

bool fit_linear_law(const std::vector<double>& y, const std::vector<double>& h,
                    const std::vector<double>& x, const std::vector<double>& c, SrGateLawCoeff& coeff) {
    const int n = static_cast<int>(y.size());
    if (n < 2) {
        return false;
    }
    double xt_x[4][4] = {};
    double xt_y[4] = {};
    for (int i = 0; i < n; ++i) {
        const double feats[4] = {1.0, h[static_cast<std::size_t>(i)], x[static_cast<std::size_t>(i)],
                                 c[static_cast<std::size_t>(i)]};
        for (int a = 0; a < kFeat; ++a) {
            xt_y[a] += feats[a] * y[static_cast<std::size_t>(i)];
            for (int b = 0; b < kFeat; ++b) {
                xt_x[a][b] += feats[a] * feats[b];
            }
        }
    }
    double beta[4] = {};
    if (!solve_linear_system(xt_x, xt_y, beta)) {
        return false;
    }
    coeff.bias = beta[0];
    coeff.w_h = beta[1];
    coeff.w_x = beta[2];
    coeff.w_c = beta[3];

    double mean_y = 0.0;
    for (double v : y) {
        mean_y += v;
    }
    mean_y /= static_cast<double>(n);
    double ss_tot = 0.0;
    double ss_res = 0.0;
    for (int i = 0; i < n; ++i) {
        const double pred =
            coeff.predict(h[static_cast<std::size_t>(i)], x[static_cast<std::size_t>(i)],
                          c[static_cast<std::size_t>(i)]);
        const double diff = y[static_cast<std::size_t>(i)] - mean_y;
        ss_tot += diff * diff;
        const double err = y[static_cast<std::size_t>(i)] - pred;
        ss_res += err * err;
    }
    coeff.r2 = ss_tot > 1e-14 ? 1.0 - ss_res / ss_tot : 1.0;
    return true;
}

void fit_gate_block(const SrGateTrace& trace, int hidden, int gate_offset, bool use_c_ref,
                    std::vector<SrGateLawCoeff>& out) {
    out.assign(static_cast<std::size_t>(hidden), SrGateLawCoeff{});
    if (trace.steps.empty()) {
        return;
    }
    for (int j = 0; j < hidden; ++j) {
        std::vector<double> ys;
        std::vector<double> hs;
        std::vector<double> xs;
        std::vector<double> cs;
        for (const auto& step : trace.steps) {
            if (static_cast<int>(step.gates_pre.size()) < 4 * hidden) {
                continue;
            }
            ys.push_back(step.gates_pre[static_cast<std::size_t>(gate_offset + j)]);
            hs.push_back(step.h[static_cast<std::size_t>(j)]);
            xs.push_back(step.x[static_cast<std::size_t>(j)]);
            cs.push_back(use_c_ref ? step.c[static_cast<std::size_t>(j)] : step.h[static_cast<std::size_t>(j)]);
        }
        fit_linear_law(ys, hs, xs, cs, out[static_cast<std::size_t>(j)]);
    }
}

nlohmann::json coeff_to_json(const SrGateLawCoeff& c) {
    return {{"w_h", c.w_h}, {"w_x", c.w_x}, {"w_c", c.w_c}, {"bias", c.bias}, {"r2", c.r2}};
}

SrGateLawCoeff coeff_from_json(const nlohmann::json& j) {
    SrGateLawCoeff c;
    if (j.contains("w_h")) c.w_h = j.at("w_h").get<double>();
    if (j.contains("w_x")) c.w_x = j.at("w_x").get<double>();
    if (j.contains("w_c")) c.w_c = j.at("w_c").get<double>();
    if (j.contains("bias")) c.bias = j.at("bias").get<double>();
    if (j.contains("r2")) c.r2 = j.at("r2").get<double>();
    return c;
}

std::vector<SrGateLawCoeff> coeffs_from_json_array(const nlohmann::json& arr) {
    std::vector<SrGateLawCoeff> out;
    if (!arr.is_array()) {
        return out;
    }
    for (const auto& item : arr) {
        out.push_back(coeff_from_json(item));
    }
    return out;
}

}  // namespace

SrGateTrace collect_lstm_gate_trace(const CharLSTMHead& lstm, const std::vector<int>& token_ids,
                                    int max_steps) {
    SrGateTrace trace;
    if (lstm.hidden <= 0 || token_ids.empty() || max_steps <= 0) {
        return trace;
    }
    std::vector<double> h(static_cast<std::size_t>(lstm.hidden), 0.0);
    std::vector<double> c(static_cast<std::size_t>(lstm.hidden), 0.0);
    std::vector<double> log_probs(static_cast<std::size_t>(lstm.vocab_size));
    const int n = std::min(max_steps, static_cast<int>(token_ids.size()));
    for (int t = 0; t < n; ++t) {
        const int tid = token_ids[static_cast<std::size_t>(t)];
        if (tid < 0 || tid >= lstm.vocab_size) {
            continue;
        }
        CharLSTMCache cache;
        std::vector<double> h_out;
        std::vector<double> c_out;
        lstm.forward_step(tid, h.data(), c.data(), log_probs.data(), h_out, c_out, &cache);
        SrGateTraceStep step;
        step.x = cache.x;
        step.h = cache.h;
        step.c = cache.c;
        step.gates_pre = cache.gates;
        trace.steps.push_back(std::move(step));
        h = std::move(h_out);
        c = std::move(c_out);
    }
    return trace;
}

SrGateLaws fit_sr_gate_laws(const SrGateTrace& trace, int hidden) {
    SrGateLaws laws;
    laws.hidden = hidden;
    if (hidden <= 0 || trace.steps.empty()) {
        return laws;
    }
    fit_gate_block(trace, hidden, 0, false, laws.i_gate);
    fit_gate_block(trace, hidden, hidden, true, laws.f_gate);
    fit_gate_block(trace, hidden, 2 * hidden, false, laws.g_gate);
    fit_gate_block(trace, hidden, 3 * hidden, true, laws.o_gate);
    laws.fitted = !laws.f_gate.empty();
    return laws;
}

nlohmann::json sr_gate_laws_to_json(const SrGateLaws& laws) {
    nlohmann::json j;
    j["hidden"] = laws.hidden;
    j["fitted"] = laws.fitted;
    auto pack = [&](const std::vector<SrGateLawCoeff>& coeffs) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& c : coeffs) {
            arr.push_back(coeff_to_json(c));
        }
        return arr;
    };
    j["i_gate"] = pack(laws.i_gate);
    j["f_gate"] = pack(laws.f_gate);
    j["g_gate"] = pack(laws.g_gate);
    j["o_gate"] = pack(laws.o_gate);
    j["mean_r2"] = sr_gate_laws_mean_r2(laws);
    return j;
}

SrGateLaws sr_gate_laws_from_json(const nlohmann::json& j) {
    SrGateLaws laws;
    if (j.contains("hidden")) laws.hidden = j.at("hidden").get<int>();
    if (j.contains("fitted")) laws.fitted = j.at("fitted").get<bool>();
    if (j.contains("i_gate")) laws.i_gate = coeffs_from_json_array(j.at("i_gate"));
    if (j.contains("f_gate")) laws.f_gate = coeffs_from_json_array(j.at("f_gate"));
    if (j.contains("g_gate")) laws.g_gate = coeffs_from_json_array(j.at("g_gate"));
    if (j.contains("o_gate")) laws.o_gate = coeffs_from_json_array(j.at("o_gate"));
    return laws;
}

double sr_gate_laws_mean_r2(const SrGateLaws& laws) {
    if (!laws.fitted) {
        return 0.0;
    }
    double sum = 0.0;
    int count = 0;
    auto accum = [&](const std::vector<SrGateLawCoeff>& coeffs) {
        for (const auto& c : coeffs) {
            sum += c.r2;
            ++count;
        }
    };
    accum(laws.i_gate);
    accum(laws.f_gate);
    accum(laws.g_gate);
    accum(laws.o_gate);
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

}  // namespace cypha::cyphalm
