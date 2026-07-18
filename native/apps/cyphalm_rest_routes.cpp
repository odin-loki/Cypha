#include "cypha/cyphalm/cyphalm_rest.hpp"

#include <chrono>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/cypha.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/intelligence/epistemic_threshold.hpp"

namespace cypha::cyphalm {

namespace {

std::mutex* g_mu{nullptr};
cypha::Cypha* g_cypha{nullptr};
std::string g_lm_source;
int g_lm_generations = 0;
std::chrono::steady_clock::time_point g_lm_loaded = std::chrono::steady_clock::now();
cypha::intelligence::EpistemicThreshold g_lm_epistemic_threshold(0.5, 5.0);

DecodeParams decode_params_from_json(const nlohmann::json& body) {
    DecodeParams p;
    p.strategy = decode_strategy_from_string(body.value("strategy", std::string("temperature")));
    p.temperature = body.value("temperature", 0.9);
    p.top_k = body.value("top_k", 40);
    p.top_p = body.value("top_p", 0.9);
    p.seed = body.value("seed", static_cast<std::uint64_t>(42));
    if (body.contains("uncertainty_threshold") && !body["uncertainty_threshold"].is_null()) {
        p.uncertainty_threshold = body["uncertainty_threshold"].get<double>();
    }
    p.epistemic_halt = body.value("epistemic_halt", false);
    p.self_correct = body.value("self_correct", false);
    if (p.self_correct) {
        p.epistemic_halt = true;
    }
    return p;
}

std::string body_strategy_name(DecodeStrategy s) {
    switch (s) {
        case DecodeStrategy::Greedy:
            return "greedy";
        case DecodeStrategy::TopK:
            return "top_k";
        case DecodeStrategy::TopP:
            return "top_p";
        case DecodeStrategy::UncertaintyGated:
            return "uncertainty_gated";
        default:
            return "temperature";
    }
}

nlohmann::json generate_response_json(const GenerateOutput& gen) {
    nlohmann::json out;
    out["generated_ids"] = gen.generated_ids;
    out["halted_on_uncertainty"] = gen.halted_on_uncertainty;
    out["halted_on_epistemic"] = gen.halted_on_epistemic;
    out["r_eu_proxy"] = gen.r_eu_proxy;
    out["self_corrected"] = gen.self_corrected;
    out["self_correct_passes"] = gen.self_correct_passes;
    out["strategy"] = body_strategy_name(gen.strategy);
    nlohmann::json steps = nlohmann::json::array();
    for (const auto& s : gen.per_step) {
        nlohmann::json row;
        if (s.halted) {
            row["token_id"] = nullptr;
            row["loss"] = nullptr;
        } else {
            row["token_id"] = s.token_id;
            row["loss"] = s.loss;
        }
        row["epistemic_var"] = s.epistemic_var;
        row["aleatoric_var"] = s.aleatoric_var;
        row["active_experts"] = s.active_experts;
        if (s.halted) row["halted"] = true;
        steps.push_back(row);
    }
    out["per_step"] = steps;
    out["n_tokens"] = static_cast<int>(gen.generated_ids.size());
    return out;
}

void handle_generate(const nlohmann::json& body, httplib::Response& res, bool force_stream) {
    std::lock_guard<std::mutex> lk(*g_mu);
    CyphaLMModel* lm = g_cypha ? g_cypha->sequence() : nullptr;
    if (!lm) {
        res.status = 503;
        res.set_content(R"({"detail":"No sequence model loaded"})", "application/json");
        return;
    }
    std::vector<int> prompt = body.value("prompt_ids", std::vector<int>{});
    const int max_tokens = body.value("max_tokens", 64);
    const bool stream = force_stream || body.value("stream", false);
    const DecodeParams params = decode_params_from_json(body);

    if (stream) {
        std::ostringstream sse;
        stream_generate(
            *lm, prompt, max_tokens, params,
            [&sse](const nlohmann::json& chunk) {
                sse << "data: " << chunk.dump() << "\n\n";
                return true;
            },
            &g_lm_epistemic_threshold);
        sse << "data: {\"done\": true}\n\n";
        ++g_lm_generations;
        res.set_content(sse.str(), "text/event-stream");
        return;
    }

    const GenerateOutput gen = generate_decode(*lm, prompt, max_tokens, params, &g_lm_epistemic_threshold);
    ++g_lm_generations;
    res.set_content(generate_response_json(gen).dump(), "application/json");
}

}  // namespace

void cyphalm_rest_configure(std::mutex* mu, cypha::Cypha* cypha) {
    g_mu = mu;
    g_cypha = cypha;
}

bool cyphalm_rest_lm_loaded() {
    std::lock_guard<std::mutex> lk(*g_mu);
    return g_cypha && g_cypha->sequence_loaded();
}

void cyphalm_rest_lm_load(const std::string& checkpoint_path) {
    std::lock_guard<std::mutex> lk(*g_mu);
    if (!g_cypha) {
        throw std::runtime_error("cyphalm REST not configured");
    }
    if (!g_cypha->load_sequence(checkpoint_path)) {
        throw std::runtime_error("failed to load sequence checkpoint");
    }
    g_lm_source = checkpoint_path;
    g_lm_generations = 0;
    g_lm_loaded = std::chrono::steady_clock::now();
}

nlohmann::json cyphalm_rest_generate_json(const std::vector<int>& prompt_ids, int max_tokens,
                                          const DecodeParams& params) {
    std::lock_guard<std::mutex> lk(*g_mu);
    CyphaLMModel* lm = g_cypha ? g_cypha->sequence() : nullptr;
    if (!lm) {
        throw std::runtime_error("No sequence model loaded");
    }
    const GenerateOutput gen = generate_decode(*lm, prompt_ids, max_tokens, params, &g_lm_epistemic_threshold);
    ++g_lm_generations;
    return generate_response_json(gen);
}

void register_cyphalm_rest_routes(httplib::Server& svr) {
    svr.Post("/sequence/load", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            if (!body.contains("checkpoint_path")) {
                res.status = 400;
                res.set_content(R"({"detail":"checkpoint_path required"})", "application/json");
                return;
            }
            const std::string path = body.at("checkpoint_path").get<std::string>();
            // Load under one lock (do not call cyphalm_rest_lm_load — it also locks g_mu).
            nlohmann::json out;
            {
                std::lock_guard<std::mutex> lk(*g_mu);
                if (!g_cypha) {
                    throw std::runtime_error("cyphalm REST not configured");
                }
                if (!g_cypha->load_sequence(path)) {
                    throw std::runtime_error("failed to load sequence checkpoint");
                }
                g_lm_source = path;
                g_lm_generations = 0;
                g_lm_loaded = std::chrono::steady_clock::now();
                CyphaLMModel* lm = g_cypha->sequence();
                out["loaded"] = true;
                out["summary"] = lm ? lm_summary_json(*lm, g_lm_source, g_lm_generations) : nullptr;
            }
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 400;
            nlohmann::json err;
            err["detail"] = ex.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Get("/sequence/metrics", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(*g_mu);
        CyphaLMModel* lm = g_cypha ? g_cypha->sequence() : nullptr;
        if (!lm) {
            res.status = 503;
            res.set_content(R"({"detail":"No sequence model loaded"})", "application/json");
            return;
        }
        auto j = lm_summary_json(*lm, g_lm_source, g_lm_generations);
        j["uptime_s"] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - g_lm_loaded).count();
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/predict_next", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            if (!body.contains("token_id")) {
                res.status = 400;
                res.set_content(R"({"detail":"token_id required"})", "application/json");
                return;
            }
            std::lock_guard<std::mutex> lk(*g_mu);
            CyphaLMModel* lm = g_cypha ? g_cypha->sequence() : nullptr;
            if (!lm) {
                res.status = 503;
                res.set_content(R"({"detail":"No sequence model loaded"})", "application/json");
                return;
            }
            const int tid = body.at("token_id").get<int>();
            res.set_content(predict_next_json(*lm, tid).dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 400;
            nlohmann::json err;
            err["detail"] = ex.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/generate", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            handle_generate(body, res, false);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"detail":"bad json"})", "application/json");
        }
    });

    svr.Post("/generate/stream", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            handle_generate(body, res, true);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"detail":"bad json"})", "application/json");
        }
    });
}

}  // namespace cypha::cyphalm
