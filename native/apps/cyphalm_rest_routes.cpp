#include "cypha/cyphalm/cyphalm_rest.hpp"

#include <chrono>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/cypha.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/predictive_codec.hpp"
#include "cypha/forecast/forecast_pipeline.hpp"
#include "cypha/forecast/gdelt_monitor.hpp"
#include "cypha/forecast/views_baselines.hpp"
#include "cypha/forecast/dispute_data.hpp"

namespace cypha::cyphalm {

namespace {

std::mutex* g_mu{nullptr};
cypha::Cypha* g_cypha{nullptr};
std::string g_lm_source;
int g_lm_generations = 0;
std::chrono::steady_clock::time_point g_lm_loaded = std::chrono::steady_clock::now();
cypha::intelligence::EpistemicThreshold g_lm_epistemic_threshold(0.5, 5.0);
cypha::forecast::GdeltMonitor g_forecast_monitor;

std::string bytes_to_hex(const std::vector<std::uint8_t>& bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.resize(bytes.size() * 2);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        out[2 * i] = kHex[bytes[i] >> 4];
        out[2 * i + 1] = kHex[bytes[i] & 0x0f];
    }
    return out;
}

std::vector<std::uint8_t> hex_to_bytes(const std::string& hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("bytes_hex length must be even");
    }
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::runtime_error("invalid hex digit");
    };
    std::vector<std::uint8_t> out(hex.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<std::uint8_t>((nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]));
    }
    return out;
}

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

    // Predictive arithmetic coding (LLMZip-style): tokens <-> bitstream under model P(next|prefix).
    svr.Post("/sequence/compress", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            if (!body.contains("token_ids") || !body["token_ids"].is_array()) {
                res.status = 400;
                res.set_content(R"({"detail":"token_ids array required"})", "application/json");
                return;
            }
            std::vector<std::uint32_t> tokens;
            tokens.reserve(body["token_ids"].size());
            for (const auto& t : body["token_ids"]) {
                tokens.push_back(t.get<std::uint32_t>());
            }
            std::lock_guard<std::mutex> lk(*g_mu);
            if (!g_cypha || !g_cypha->sequence()) {
                res.status = 503;
                res.set_content(R"({"detail":"No sequence model loaded"})", "application/json");
                return;
            }
            auto result = g_cypha->compress_tokens(tokens);
            if (!result.detail.empty()) {
                res.status = 400;
                nlohmann::json err;
                err["detail"] = result.detail;
                res.set_content(err.dump(), "application/json");
                return;
            }
            nlohmann::json out;
            out["bytes_hex"] = bytes_to_hex(result.bytes);
            out["n_bytes"] = result.bytes.size();
            out["n_coded"] = result.n_coded;
            out["n_tokens"] = result.n_tokens;
            out["model_bpc"] = result.model_bpc;
            out["coded_bpc"] = result.coded_bpc;
            out["seed"] = tokens.empty() ? 0 : tokens.front();
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 400;
            nlohmann::json err;
            err["detail"] = ex.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/sequence/decompress", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            if (!body.contains("bytes_hex") || !body.contains("seed") || !body.contains("n_tokens")) {
                res.status = 400;
                res.set_content(R"({"detail":"bytes_hex, seed, n_tokens required"})", "application/json");
                return;
            }
            const auto bytes = hex_to_bytes(body.at("bytes_hex").get<std::string>());
            const auto seed = body.at("seed").get<std::uint32_t>();
            const auto n_tokens = body.at("n_tokens").get<std::size_t>();
            std::lock_guard<std::mutex> lk(*g_mu);
            if (!g_cypha || !g_cypha->sequence()) {
                res.status = 503;
                res.set_content(R"({"detail":"No sequence model loaded"})", "application/json");
                return;
            }
            std::string detail;
            auto tokens = g_cypha->decompress_tokens(bytes, seed, n_tokens, &detail);
            if (!detail.empty() || tokens.size() != n_tokens) {
                res.status = 400;
                nlohmann::json err;
                err["detail"] = detail.empty() ? "decompress failed" : detail;
                res.set_content(err.dump(), "application/json");
                return;
            }
            nlohmann::json out;
            out["token_ids"] = tokens;
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 400;
            nlohmann::json err;
            err["detail"] = ex.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/forecast/run", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
            std::filesystem::path data_dir = body.value("data_dir", std::string("bench/data/forecast"));
            if (!std::filesystem::exists(data_dir)) {
                const char* candidates[] = {"bench/data/forecast", "../bench/data/forecast"};
                for (const char* c : candidates) {
                    if (std::filesystem::exists(c)) {
                        data_dir = c;
                        break;
                    }
                }
            }
            cypha::forecast::ForecastPipeline pipe;
            const auto result = pipe.run(data_dir);
            nlohmann::json out;
            out["node_eval_acc"] = result.node_result.eval_accuracy;
            out["sequence_eval_bpc"] = result.sequence_eval_bpc;
            out["tree_nodes"] = result.scenario_tree.nodes.size();
            out["paths"] = result.paths.size();
            out["views_crps"] = result.views_validation.mean_crps;
            out["views_ignorance"] = result.views_validation.mean_ignorance;
            out["drift_alarms"] = result.drift_alarms;
            const auto views_path = cypha::forecast::resolve_views_csv_path(data_dir);
            if (std::filesystem::exists(views_path)) {
                const auto train = cypha::forecast::load_views_csv(views_path, "train");
                const auto holdout = cypha::forecast::load_views_csv(views_path, "holdout");
                if (!train.empty() && !holdout.empty()) {
                    const auto board = cypha::forecast::run_views_leaderboard(train, holdout);
                    out["views_leaderboard"] = {
                        {"cypha_crps", board.cypha.result.mean_crps},
                        {"conflictology_crps", board.conflictology.result.mean_crps},
                        {"markov_crps", board.observed_markov.result.mean_crps},
                        {"negbin_crps", board.negbin_glmm.result.mean_crps},
                    };
                }
            }
            if (!result.detail.empty()) {
                out["detail"] = result.detail;
            }
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 400;
            nlohmann::json err;
            err["detail"] = ex.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/forecast/ingest", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
            cypha::forecast::GdeltEvent ev;
            ev.year = body.value("year", 0);
            ev.month = body.value("month", 0);
            ev.day = body.value("day", 0);
            ev.actor1 = body.value("actor1", std::string());
            ev.actor2 = body.value("actor2", std::string());
            ev.cameo_code = body.value("cameo_code", 0);
            ev.goldstein = body.value("goldstein", 0);
            ev.theater = body.value("theater", std::string("GLB"));
            const double drift = body.value("drift", 0.0);
            const double anomaly = body.value("anomaly", 0.0);
            std::lock_guard<std::mutex> lock(*g_mu);
            const auto alarm = g_forecast_monitor.ingest(ev, drift, anomaly);
            nlohmann::json out;
            out["fired"] = alarm.fired;
            out["drift_score"] = alarm.drift_score;
            out["anomaly_score"] = alarm.anomaly_score;
            out["reason"] = alarm.reason;
            res.set_content(out.dump(), "application/json");
        } catch (const std::exception& ex) {
            res.status = 400;
            nlohmann::json err;
            err["detail"] = ex.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/forecast/monitor/reset", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(*g_mu);
        g_forecast_monitor = cypha::forecast::GdeltMonitor{};
        nlohmann::json out;
        out["status"] = "reset";
        res.set_content(out.dump(), "application/json");
    });
}

}  // namespace cypha::cyphalm
