#include "cypha/dif_rest.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "cypha/generation.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/preprocessor.hpp"

namespace cypha {
namespace {

std::mutex* g_mu{nullptr};
std::unique_ptr<CyphaInferModel>* g_model{nullptr};
std::unique_ptr<PreprocessorState>* g_pre{nullptr};
std::mt19937* g_rng{nullptr};

constexpr int kDefaultNSamples = 10;
constexpr int kDefaultNSteps = 30;
constexpr int kDefaultKNeighbors = 5;
constexpr double kDefaultTemperature = 1.0;
constexpr double kLangevinStepSize = 0.05;

std::vector<double> transform_input(const nlohmann::json& body) {
    std::vector<double> x;
    for (const auto& v : body.at("input")) {
        x.push_back(v.get<double>());
    }
    if (g_pre && *g_pre) {
        x = (*g_pre)->transform_one(x);
    }
    return x;
}

bool input_dim_ok(const std::vector<double>& x) {
    return g_model && *g_model && static_cast<int>(x.size()) == (*g_model)->d_latent;
}

CyphaInferOptions infer_options_from_body(const nlohmann::json& body) {
    CyphaInferOptions opt{};
    if (g_model && *g_model) {
        opt.deliberation_lo = body.value("deliberation_lo", (*g_model)->deliberation_lo);
        opt.deliberation_hi = body.value("deliberation_hi", (*g_model)->deliberation_hi);
    }
    opt.use_field = true;
    return opt;
}

std::string infer_label_for_h(const CyphaInferModel& m, const double* h, const CyphaInferOptions& opt) {
    const InferAtHResult inf = infer_at_h(m, h, opt);
    return inf.label;
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
    if (body.contains("seed") && body["seed"].is_number_integer()) {
        g_rng->seed(static_cast<std::uint32_t>(body["seed"].get<int>()));
    }
}

}  // namespace

void dif_rest_configure(std::mutex* mu, std::unique_ptr<CyphaInferModel>* model,
                        std::unique_ptr<PreprocessorState>* pre, std::mt19937* rng) {
    g_mu = mu;
    g_model = model;
    g_pre = pre;
    g_rng = rng;
}

void register_dif_rest_routes(httplib::Server& svr) {
    svr.Post("/dif/generate", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            std::lock_guard<std::mutex> lk(*g_mu);
            if (!g_model || !*g_model) {
                res.status = 503;
                res.set_content(R"({"detail":"No model loaded"})", "application/json");
                return;
            }
            CyphaInferModel& m = **g_model;

            const std::string mode = body.value("mode", std::string(""));
            if (mode != "langevin" && mode != "from_observation" && mode != "retrieval_augmented") {
                res.status = 400;
                res.set_content(
                    R"({"detail":"mode must be langevin, from_observation, or retrieval_augmented"})",
                    "application/json");
                return;
            }

            std::vector<double> x = transform_input(body);
            if (!input_dim_ok(x)) {
                res.status = 400;
                res.set_content(R"({"detail":"input dim mismatch after preprocessor"})", "application/json");
                return;
            }

            maybe_reseed_rng(body);

            const int n_samples = body.value("n_samples", kDefaultNSamples);
            const int n_steps = body.value("n_steps", kDefaultNSteps);
            const double temperature = body.value("temperature", kDefaultTemperature);
            const CyphaInferOptions opt = infer_options_from_body(body);

            std::vector<double> h_query;
            batch_encode(m, x.data(), 1, h_query);

            std::string label;
            if (body.contains("label") && body["label"].is_string()) {
                label = body["label"].get<std::string>();
            }

            std::vector<std::vector<double>> samples;
            if (mode == "langevin") {
                if (label.empty()) {
                    label = infer_label_for_h(m, h_query.data(), opt);
                }
                samples = generate_langevin(m, label, n_samples, n_steps, kLangevinStepSize, temperature,
                                            g_rng, nullptr, nullptr);
            } else if (mode == "from_observation") {
                if (label.empty()) {
                    label = infer_label_for_h(m, h_query.data(), opt);
                }
                samples = generate_from_observation(m, h_query.data(), label, n_samples, temperature, n_steps,
                                                    g_rng, nullptr);
            } else {
                if (!body.contains("database") || !body["database"].is_array()) {
                    res.status = 400;
                    res.set_content(R"({"detail":"database required for retrieval_augmented"})",
                                    "application/json");
                    return;
                }
                const int k_neighbors = body.value("k_neighbors", kDefaultKNeighbors);
                int n_db = 0;
                const int input_dim = static_cast<int>(x.size());
                std::vector<double> db_flat = flatten_database(body["database"], input_dim, n_db);
                if (label.empty()) {
                    const std::vector<RetrieveHit> hits =
                        retrieve_from_x(m, x.data(), db_flat.data(), n_db, input_dim, k_neighbors, opt);
                    if (!hits.empty()) {
                        std::vector<double> h_db;
                        batch_encode(m, db_flat.data(), n_db, h_db);
                        int nearest_j = 0;
                        double best_d = std::numeric_limits<double>::infinity();
                        for (int j = 0; j < static_cast<int>(hits.size()); ++j) {
                            const int idx = hits[static_cast<std::size_t>(j)].index;
                            const double* h_i = h_db.data() + static_cast<std::ptrdiff_t>(idx) * m.d_latent;
                            double s = 0.0;
                            for (int t = 0; t < m.d_latent; ++t) {
                                const double diff = h_i[t] - h_query[static_cast<std::size_t>(t)];
                                s += diff * diff;
                            }
                            if (s < best_d) {
                                best_d = s;
                                nearest_j = j;
                            }
                        }
                        label = hits[static_cast<std::size_t>(nearest_j)].predicted_label;
                    } else {
                        label = infer_label_for_h(m, h_query.data(), opt);
                    }
                }
                samples = generate_retrieval_augmented(m, x.data(), db_flat.data(), n_db, input_dim, k_neighbors,
                                                       n_samples, temperature, n_steps, opt, g_rng, nullptr,
                                                       nullptr, nullptr);
            }

            nlohmann::json out;
            out["mode"] = mode;
            out["label"] = label;
            out["n_samples"] = n_samples;
            out["space"] = "latent";
            out["samples"] = samples_to_json(samples);
            res.set_content(out.dump(), "application/json");
        } catch (const nlohmann::json::parse_error&) {
            res.status = 400;
            res.set_content(R"({"detail":"bad json"})", "application/json");
        } catch (const std::exception& ex) {
            res.status = 400;
            nlohmann::json err;
            err["detail"] = ex.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/dif/retrieve", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            std::lock_guard<std::mutex> lk(*g_mu);
            if (!g_model || !*g_model) {
                res.status = 503;
                res.set_content(R"({"detail":"No model loaded"})", "application/json");
                return;
            }
            CyphaInferModel& m = **g_model;

            if (!body.contains("database") || !body["database"].is_array()) {
                res.status = 400;
                res.set_content(R"({"detail":"database required"})", "application/json");
                return;
            }

            std::vector<double> x = transform_input(body);
            if (!input_dim_ok(x)) {
                res.status = 400;
                res.set_content(R"({"detail":"input dim mismatch after preprocessor"})", "application/json");
                return;
            }

            const int top_k = body.value("top_k", kDefaultKNeighbors);
            std::optional<std::string> label_filter;
            if (body.contains("label") && body["label"].is_string()) {
                label_filter = body["label"].get<std::string>();
            }

            const CyphaInferOptions opt = infer_options_from_body(body);
            int n_db = 0;
            const int input_dim = static_cast<int>(x.size());
            std::vector<double> db_flat = flatten_database(body["database"], input_dim, n_db);

            std::vector<RetrieveHit> hits =
                retrieve_from_x(m, x.data(), db_flat.data(), n_db, input_dim, top_k, opt, label_filter);

            nlohmann::json out;
            out["hits"] = hits_to_json(hits);
            out["top_k"] = top_k;
            res.set_content(out.dump(), "application/json");
        } catch (const nlohmann::json::parse_error&) {
            res.status = 400;
            res.set_content(R"({"detail":"bad json"})", "application/json");
        } catch (const std::exception& ex) {
            res.status = 400;
            nlohmann::json err;
            err["detail"] = ex.what();
            res.set_content(err.dump(), "application/json");
        }
    });
}

}  // namespace cypha
