#include "cypha/branch_a_rest.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/branch_a_router.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_rest.hpp"

namespace cypha {
namespace {

std::mutex g_router_mu;
BranchARouter g_router;
std::string g_router_json_path;

std::string normalize_branch_a_json_path(const std::string& path) {
    if (path.empty()) {
        return path;
    }
    namespace fs = std::filesystem;
    const fs::path p(path);
    if (fs::is_regular_file(p)) {
        return path;
    }
    const fs::path with_json = p.parent_path() / (p.filename().string() + ".json");
    if (fs::is_regular_file(with_json)) {
        return with_json.string();
    }
    if (p.extension() != ".json") {
        const fs::path alt = p;
        const fs::path alt_json = alt.parent_path() / (alt.stem().string() + ".json");
        if (fs::is_regular_file(alt_json)) {
            return alt_json.string();
        }
    }
    return path;
}

std::string env_or(const char* key, const std::string& fallback) {
    if (const char* v = std::getenv(key)) {
        return std::string(v);
    }
    return fallback;
}

std::string ollama_base_url() {
    return env_or("CYPHA_OLLAMA_URL", "http://127.0.0.1:11434");
}

std::string ollama_model() {
    const std::string m = env_or("CYPHA_OLLAMA_MODEL", "mistral");
    return m.empty() ? "mistral" : m;
}

bool ollama_reachable() {
    try {
        httplib::Client cli(ollama_base_url());
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(3, 0);
        if (auto res = cli.Get("/api/tags")) {
            return res->status == 200;
        }
    } catch (...) {
    }
    return false;
}

nlohmann::json ollama_generate(const std::string& prompt, const std::string& model,
                               const std::string& system) {
    const auto t0 = std::chrono::steady_clock::now();
    httplib::Client cli(ollama_base_url());
    cli.set_connection_timeout(120, 0);
    cli.set_read_timeout(120, 0);
    nlohmann::json payload;
    payload["model"] = model;
    payload["prompt"] = prompt;
    payload["stream"] = false;
    if (!system.empty()) {
        payload["system"] = system;
    }
    auto res = cli.Post("/api/generate", payload.dump(), "application/json");
    if (!res || res->status != 200) {
        throw std::runtime_error("Ollama generate failed");
    }
    const auto body = nlohmann::json::parse(res->body);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    nlohmann::json out;
    out["provider"] = "ollama";
    out["text"] = body.value("response", std::string(""));
    out["model"] = model;
    out["latency_ms"] = ms;
    out["done"] = body.value("done", true);
    return out;
}

nlohmann::json route_to_json(const BranchARouteResult& r) {
    return nlohmann::json{
        {"label", r.label},
        {"confidence", r.confidence},
        {"epistemic_var", r.epistemic_var},
        {"abstain", r.abstain},
        {"embedding_backend", r.embedding_backend},
        {"action", r.action},
    };
}

}  // namespace

void branch_a_rest_configure(const std::string& checkpoint_json_path) {
    std::lock_guard<std::mutex> lk(g_router_mu);
    g_router_json_path = normalize_branch_a_json_path(checkpoint_json_path);
    g_router.set_checkpoint_base(g_router_json_path);
    if (!g_router_json_path.empty()) {
        g_router.try_load_checkpoint(g_router_json_path);
    }
}

bool branch_a_rest_router_trained() {
    std::lock_guard<std::mutex> lk(g_router_mu);
    return g_router.is_trained();
}

nlohmann::json branch_a_rest_summary_json() {
    std::lock_guard<std::mutex> lk(g_router_mu);
    return g_router.summary();
}

void register_branch_a_rest_routes(httplib::Server& svr) {
    svr.Get("/route/health", [](const httplib::Request&, httplib::Response& res) {
        nlohmann::json j;
        {
            std::lock_guard<std::mutex> lk(g_router_mu);
            j["router_trained"] = g_router.is_trained();
            j["router_summary"] = g_router.is_trained() ? g_router.summary() : nullptr;
        }
        j["ollama_url"] = ollama_base_url();
        j["ollama_model"] = ollama_model();
        j["ollama_reachable"] = ollama_reachable();
        j["lm_loaded"] = cyphalm::cyphalm_rest_lm_loaded();
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/route/text", [](const httplib::Request& req, httplib::Response& res) {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            const auto body = nlohmann::json::parse(req.body);
            const std::string text = body.value("text", std::string(""));
            if (text.find_first_not_of(" \t\n\r") == std::string::npos) {
                res.status = 400;
                res.set_content(R"({"detail":"text must be non-empty"})", "application/json");
                return;
            }
            std::optional<double> threshold;
            if (body.contains("epistemic_threshold") && !body["epistemic_threshold"].is_null()) {
                threshold = body["epistemic_threshold"].get<double>();
            }
            BranchARouteResult route;
            {
                std::lock_guard<std::mutex> lk(g_router_mu);
                route = g_router.route(text, threshold);
            }
            auto j = route_to_json(route);
            j["latency_ms"] =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            res.set_content(j.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 500;
            nlohmann::json err;
            err["detail"] = std::string("Branch A route failed: ") + ex.what();
            res.set_content(err.dump(), "application/json");
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"detail":"bad json"})", "application/json");
        }
    });

    svr.Post("/route/generate", [](const httplib::Request& req, httplib::Response& res) {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            const auto body = nlohmann::json::parse(req.body);
            const std::string text = body.value("text", std::string(""));
            if (text.find_first_not_of(" \t\n\r") == std::string::npos) {
                res.status = 400;
                res.set_content(R"({"detail":"text must be non-empty"})", "application/json");
                return;
            }

            BranchARouter::DispatchGenerateOptions opt;
            if (body.contains("epistemic_threshold") && !body["epistemic_threshold"].is_null()) {
                opt.epistemic_threshold = body["epistemic_threshold"].get<double>();
            }
            opt.max_tokens = body.value("max_tokens", 128);
            if (body.contains("ollama_model") && body["ollama_model"].is_string()) {
                opt.ollama_model = body["ollama_model"].get<std::string>();
            }
            if (body.contains("ollama_system") && body["ollama_system"].is_string()) {
                opt.ollama_system = body["ollama_system"].get<std::string>();
            }
            opt.cypha_lm_strategy = body.value("cypha_lm_strategy", std::string("top_p"));
            opt.cypha_lm_temperature = body.value("cypha_lm_temperature", 0.9);

            BranchARouteResult route;
            {
                std::lock_guard<std::mutex> lk(g_router_mu);
                route = g_router.route(text, opt.epistemic_threshold);
            }

            nlohmann::json out;
            out["route"] = route_to_json(route);
            out["generation"] = nullptr;

            if (route.abstain) {
                try {
                    const std::string model = opt.ollama_model.value_or(ollama_model());
                    const std::string system = opt.ollama_system.value_or(
                        "You are a helpful assistant. The user's query was flagged as "
                        "out-of-domain for the Cypha router; answer directly and concisely.");
                    out["generation"] = ollama_generate(text, model, system);
                } catch (const std::exception& ex) {
                    nlohmann::json gen;
                    gen["provider"] = "ollama";
                    gen["error"] = ex.what();
                    gen["text"] = "";
                    out["generation"] = gen;
                }
            } else if (!cyphalm::cyphalm_rest_lm_loaded()) {
                nlohmann::json gen;
                gen["provider"] = "none";
                gen["text"] = "";
                gen["reason"] = "CyphaLM not loaded; routing only (in-domain).";
                out["generation"] = gen;
            } else {
                const int vocab = 128;
                const auto prompt_ids = encode_prompt_chars(text, vocab);
                cyphalm::DecodeParams params;
                params.strategy = cyphalm::decode_strategy_from_string(opt.cypha_lm_strategy);
                params.temperature = opt.cypha_lm_temperature;
                params.top_p = 0.92;
                const auto gen_t0 = std::chrono::steady_clock::now();
                const auto gen_json =
                    cyphalm::cyphalm_rest_generate_json(prompt_ids, opt.max_tokens, params);
                const auto generated = gen_json.value("generated_ids", std::vector<int>{});
                nlohmann::json gen;
                gen["provider"] = "cypha_lm";
                gen["text"] = decode_generated_ids(generated, text, vocab);
                gen["generated_ids"] = generated;
                gen["latency_ms"] =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              gen_t0)
                        .count();
                gen["n_tokens"] = gen_json.value("n_tokens", static_cast<int>(generated.size()));
                out["generation"] = gen;
            }

            out["latency_ms"] =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 500;
            nlohmann::json err;
            err["detail"] = std::string("Branch A dispatch failed: ") + ex.what();
            res.set_content(err.dump(), "application/json");
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"detail":"bad json"})", "application/json");
        }
    });

    svr.Post("/route/save", [](const httplib::Request&, httplib::Response& res) {
        try {
            std::string path;
            nlohmann::json summary;
            {
                std::lock_guard<std::mutex> lk(g_router_mu);
                if (!g_router.is_trained()) {
                    res.status = 400;
                    res.set_content(
                        R"({"detail":"Router not trained — call /route/text or /route/generate first"})",
                        "application/json");
                    return;
                }
                path = g_router.save_checkpoint();
                summary = g_router.summary();
            }
            nlohmann::json out;
            out["saved"] = true;
            out["checkpoint"] = path;
            out["summary"] = summary;
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 500;
            nlohmann::json err;
            err["detail"] = std::string("Failed to save checkpoint: ") + ex.what();
            res.set_content(err.dump(), "application/json");
        }
    });
}

}  // namespace cypha
