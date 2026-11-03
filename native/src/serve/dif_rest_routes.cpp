#include "cypha/dif_rest.hpp"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/cypha.hpp"

namespace cypha {
namespace {

std::mutex* g_mu{nullptr};
Cypha* g_cypha{nullptr};

constexpr int kDefaultNSamples = 10;
constexpr int kDefaultNSteps = 30;
constexpr int kDefaultKNeighbors = 5;
constexpr double kDefaultTemperature = 1.0;
constexpr double kLangevinStepSize = 0.05;

std::vector<double> parse_input(const nlohmann::json& body) {
    std::vector<double> x;
    for (const auto& v : body.at("input")) {
        x.push_back(v.get<double>());
    }
    return x;
}

CyphaInferOptions infer_options_from_body(const nlohmann::json& body) {
    CyphaInferOptions opt{};
    if (g_cypha && g_cypha->infer()) {
        const CyphaInferModel& m = *g_cypha->infer();
        opt.deliberation_lo = body.value("deliberation_lo", m.deliberation_lo);
        opt.deliberation_hi = body.value("deliberation_hi", m.deliberation_hi);
    }
    opt.use_field = true;
    return opt;
}

SampleMode sample_mode_from_string(const std::string& mode) {
    if (mode == "langevin") {
        return SampleMode::Langevin;
    }
    if (mode == "from_observation") {
        return SampleMode::FromObservation;
    }
    return SampleMode::RetrievalAugmented;
}

nlohmann::json samples_to_json(const std::vector<std::vector<double>>& samples) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& row : samples) {
        nlohmann::json r = nlohmann::json::array();
        for (double v : row) {
            r.push_back(v);
        }
        arr.push_back(std::move(r));
    }
    return arr;
}

nlohmann::json hits_to_json(const std::vector<RetrieveHit>& hits) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& hit : hits) {
        nlohmann::json row;
        row["index"] = hit.index;
        row["log_likelihood"] = hit.log_likelihood;
        row["predicted_label"] = hit.predicted_label;
        arr.push_back(std::move(row));
    }
    return arr;
}

std::vector<double> flatten_database(const nlohmann::json& database, int input_dim, int& n_db_out) {
    n_db_out = static_cast<int>(database.size());
    std::vector<double> flat(static_cast<std::size_t>(n_db_out * input_dim));
    for (int i = 0; i < n_db_out; ++i) {
        const auto& row = database.at(static_cast<std::size_t>(i));
        if (!row.is_array() || static_cast<int>(row.size()) != input_dim) {
            throw std::runtime_error("database row dim mismatch");
        }
        for (int j = 0; j < input_dim; ++j) {
            flat[static_cast<std::size_t>(i * input_dim + j)] = row.at(static_cast<std::size_t>(j)).get<double>();
        }
    }
    return flat;
}

void maybe_reseed_rng(const nlohmann::json& body) {
    if (body.contains("seed") && body["seed"].is_number_integer() && g_cypha) {
        g_cypha->rng().seed(static_cast<std::uint32_t>(body["seed"].get<int>()));
    }
}

void respond_detail(httplib::Response& res, int status, const std::string& detail) {
    res.status = status;
    nlohmann::json err;
    err["detail"] = detail;
    res.set_content(err.dump(), "application/json");
}

}  // namespace

void dif_rest_configure(std::mutex* mu, Cypha* cypha) {
    g_mu = mu;
    g_cypha = cypha;
}

void register_dif_rest_routes(httplib::Server& svr) {
    svr.Post("/sample", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            std::lock_guard<std::mutex> lk(*g_mu);
            if (!g_cypha || !g_cypha->loaded()) {
                respond_detail(res, 503, "No model loaded");
                return;
            }

            const std::string mode = body.value("mode", std::string(""));
            if (mode != "langevin" && mode != "from_observation" && mode != "retrieval_augmented") {
                respond_detail(res, 400, "mode must be langevin, from_observation, or retrieval_augmented");
                return;
            }

            std::vector<double> x = parse_input(body);
            maybe_reseed_rng(body);

            const int n_samples = body.value("n_samples", kDefaultNSamples);
            const int n_steps = body.value("n_steps", kDefaultNSteps);
            const double temperature = body.value("temperature", kDefaultTemperature);

            SampleOpts opts{};
            opts.mode = sample_mode_from_string(mode);
            opts.n_samples = n_samples;
            opts.n_steps = n_steps;
            opts.temperature = temperature;
            opts.step_size = kLangevinStepSize;
            opts.k_neighbors = body.value("k_neighbors", kDefaultKNeighbors);
            opts.infer_opt = infer_options_from_body(body);
            opts.x = x.data();
            opts.x_dim = static_cast<int>(x.size());
            if (body.contains("label") && body["label"].is_string()) {
                opts.label = body["label"].get<std::string>();
            }

            std::vector<double> db_flat;
            if (opts.mode == SampleMode::RetrievalAugmented) {
                if (!body.contains("database") || !body["database"].is_array()) {
                    respond_detail(res, 400, "database required for retrieval_augmented");
                    return;
                }
                int n_db = 0;
                const int input_dim = static_cast<int>(x.size());
                db_flat = flatten_database(body["database"], input_dim, n_db);
                opts.database_x = db_flat.data();
                opts.n_db = n_db;
                opts.database_input_dim = input_dim;
            }

            const SampleOut out = g_cypha->sample(opts);
            if (!out.detail.empty()) {
                const int status = (out.detail == "No model loaded") ? 503 : 400;
                respond_detail(res, status, out.detail);
                return;
            }

            nlohmann::json response;
            response["mode"] = mode;
            response["label"] = out.label;
            response["n_samples"] = n_samples;
            response["space"] = "latent";
            response["samples"] = samples_to_json(out.h);
            res.set_content(response.dump(), "application/json");
        } catch (const nlohmann::json::parse_error&) {
            respond_detail(res, 400, "bad json");
        } catch (const std::exception& ex) {
            respond_detail(res, 400, ex.what());
        }
    });

    svr.Post("/retrieve", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            std::lock_guard<std::mutex> lk(*g_mu);
            if (!g_cypha || !g_cypha->loaded()) {
                respond_detail(res, 503, "No model loaded");
                return;
            }

            if (!body.contains("database") || !body["database"].is_array()) {
                respond_detail(res, 400, "database required");
                return;
            }

            std::vector<double> x = parse_input(body);
            const int top_k = body.value("top_k", kDefaultKNeighbors);

            RetrieveOpts ropts{};
            ropts.top_k = top_k;
            ropts.infer_opt = infer_options_from_body(body);
            if (body.contains("label") && body["label"].is_string()) {
                ropts.label = body["label"].get<std::string>();
            }

            int n_db = 0;
            const int input_dim = static_cast<int>(x.size());
            const std::vector<double> db_flat = flatten_database(body["database"], input_dim, n_db);

            const RetrieveOut out =
                g_cypha->retrieve(x.data(), static_cast<int>(x.size()), db_flat.data(), n_db, input_dim, ropts);
            if (!out.detail.empty()) {
                const int status = (out.detail == "No model loaded") ? 503 : 400;
                respond_detail(res, status, out.detail);
                return;
            }

            nlohmann::json response;
            response["hits"] = hits_to_json(out.hits);
            response["top_k"] = top_k;
            res.set_content(response.dump(), "application/json");
        } catch (const nlohmann::json::parse_error&) {
            respond_detail(res, 400, "bad json");
        } catch (const std::exception& ex) {
            respond_detail(res, 400, ex.what());
        }
    });
}

}  // namespace cypha
