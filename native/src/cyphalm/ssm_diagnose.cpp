#include "cypha/cyphalm/ssm_diagnose.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "cypha/cyphalm/cyphalm_model.hpp"

namespace cypha::cyphalm {

namespace {

constexpr double kCollapseNorm = 1e-6;
constexpr double kExplodeNorm = 1e6;

bool is_finite(double v) { return std::isfinite(v); }

nlohmann::json summarize_norm_track(const std::vector<double>& track) {
    nlohmann::json out;
    if (track.empty()) {
        out["min"] = nullptr;
        out["max"] = nullptr;
        out["final"] = nullptr;
        out["collapsed"] = false;
        out["exploded"] = false;
        return out;
    }
    const auto [mn_it, mx_it] = std::minmax_element(track.begin(), track.end());
    const double mn = *mn_it;
    const double mx = *mx_it;
    const double fin = track.back();
    bool collapsed = fin < kCollapseNorm;
    bool exploded = false;
    for (double v : track) {
        if (!is_finite(v) || v > kExplodeNorm) {
            exploded = true;
            break;
        }
    }
    if (mx > 0.0 && mn > 0.0 && mx / mn > 1e6) {
        collapsed = true;
    }
    out["min"] = mn;
    out["max"] = mx;
    out["final"] = fin;
    out["collapsed"] = collapsed;
    out["exploded"] = exploded;
    return out;
}

nlohmann::json decay_json(const CellAISSM& ssm) {
    const auto st = ssm.get_state();
    const double tau_fast = st.value("tau_fast", 10.0);
    const double tau_slow = st.value("tau_slow", 100.0);
    const double lambda_slow = std::exp(-1.0 / std::max(tau_slow, 1e-6));
    return nlohmann::json{
        {"tau_fast", tau_fast},
        {"tau_slow", tau_slow},
        {"lambda_fast", ssm.lambda_fast()},
        {"lambda_slow", lambda_slow},
    };
}

}  // namespace

double vector_l2_norm(const std::vector<double>& v) {
    double acc = 0.0;
    for (double x : v) acc += x * x;
    return std::sqrt(acc);
}

double layer_stack_mean_norm(const std::vector<std::vector<double>>& states) {
    if (states.empty()) return 0.0;
    double acc = 0.0;
    for (const auto& row : states) acc += vector_l2_norm(row);
    return acc / static_cast<double>(states.size());
}

std::vector<double> fit_input_dim(const std::vector<double>& row, int d_input) {
    std::vector<double> out(static_cast<std::size_t>(std::max(0, d_input)), 0.0);
    const std::size_t n = std::min(out.size(), row.size());
    for (std::size_t i = 0; i < n; ++i) out[i] = row[i];
    return out;
}

nlohmann::json diagnose_cellai_sequence(CellAISSM& ssm,
                                        const std::vector<std::vector<double>>& inputs,
                                        int sample_stride, const char* domain_tag) {
    const int stride = std::max(1, sample_stride);
    ssm.reset();

    std::vector<double> fast_track;
    std::vector<double> slow_track;
    std::vector<double> ctx_track;
    nlohmann::json ctx_samples = nlohmann::json::array();

    for (std::size_t t = 0; t < inputs.size(); ++t) {
        const auto& row = inputs[t];
        if (static_cast<int>(row.size()) != ssm.d_input()) {
            throw std::invalid_argument("ssm_diagnose: input dim mismatch");
        }
        const auto ctx = ssm.step(row);
        fast_track.push_back(layer_stack_mean_norm(ssm.h_states()));
        slow_track.push_back(layer_stack_mean_norm(ssm.s_states()));
        const double ctx_n = vector_l2_norm(ctx);
        ctx_track.push_back(ctx_n);
        if (static_cast<int>(t) % stride == 0 || t + 1 == inputs.size()) {
            ctx_samples.push_back(
                nlohmann::json{{"step", static_cast<int>(t)}, {"context_norm", ctx_n}});
        }
    }

    const auto fast_summary = summarize_norm_track(fast_track);
    const auto slow_summary = summarize_norm_track(slow_track);
    const auto ctx_summary = summarize_norm_track(ctx_track);
    const bool collapsed =
        fast_summary.value("collapsed", false) || slow_summary.value("collapsed", false);
    const bool exploded =
        fast_summary.value("exploded", false) || slow_summary.value("exploded", false);

    nlohmann::json decay = decay_json(ssm);
    return nlohmann::json{
        {"domain", domain_tag},
        {"steps", static_cast<int>(inputs.size())},
        {"sample_stride", stride},
        {"tau_fast", decay["tau_fast"]},
        {"tau_slow", decay["tau_slow"]},
        {"lambda_fast", decay["lambda_fast"]},
        {"lambda_slow", decay["lambda_slow"]},
        {"decay_rates",
         nlohmann::json{{"fast", decay["lambda_fast"]}, {"slow", decay["lambda_slow"]}}},
        {"state_norms",
         nlohmann::json{{"fast_track", fast_track},
                        {"slow_track", slow_track},
                        {"context_track", ctx_track},
                        {"context_samples", ctx_samples}}},
        {"summary",
         nlohmann::json{{"fast", fast_summary},
                        {"slow", slow_summary},
                        {"context", ctx_summary},
                        {"collapsed", collapsed},
                        {"exploded", exploded}}},
        {"checks_passed", !collapsed && !exploded},
        {"tool", "cyphalm_ssm_diagnose"},
    };
}

nlohmann::json diagnose_model_tokens(CyphaLMModel& model, const std::vector<int>& token_ids,
                                     int max_steps, const char* domain_tag) {
    CellAISSM* active = model.active_ssm();
    if (active == nullptr) {
        return nlohmann::json{{"domain", domain_tag},
                              {"error", "no_active_ssm"},
                              {"checks_passed", false}};
    }

    const int n = std::min(max_steps, static_cast<int>(token_ids.size()));
    if (n <= 0) {
        return nlohmann::json{{"domain", domain_tag},
                              {"error", "empty_token_stream"},
                              {"checks_passed", false}};
    }

    std::vector<std::vector<double>> embeddings;
    embeddings.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        embeddings.push_back(
            model.embed_vector(static_cast<std::uint32_t>(token_ids[static_cast<std::size_t>(i)])));
    }

    auto report = diagnose_cellai_sequence(*active, embeddings, std::max(1, n / 16), domain_tag);

    model.reset_context();
    std::vector<double> field_rms;
    field_rms.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        (void)model.predict_next(static_cast<std::uint32_t>(token_ids[static_cast<std::size_t>(i)]));
        field_rms.push_back(vector_l2_norm(model.field_vector()));
    }

    const auto& cfg = model.config();
    const double proj_rms = model.ssm_projection_rms();
    const bool routing_connected = proj_rms > 1e-8 && model.has_gria_routing();
    report["projection"] = nlohmann::json{
        {"field_dim", cfg.field_dim},
        {"context_dim", active->context_dim()},
        {"proj_weight_rms", proj_rms},
        {"field_output_mean_rms",
         field_rms.empty()
             ? 0.0
             : std::accumulate(field_rms.begin(), field_rms.end(), 0.0) /
                   static_cast<double>(field_rms.size())},
        {"connected_to_routing", routing_connected},
    };
    if (!routing_connected) {
        report["checks_passed"] = false;
    }
    return report;
}

}  // namespace cypha::cyphalm
