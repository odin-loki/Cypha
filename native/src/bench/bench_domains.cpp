// bench_domains — native bench domain runners (d01–d25) for cypha_bench_run.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <zlib.h>

#ifdef _WIN32
#include <io.h>
#ifndef popen
#define popen _popen
#define pclose _pclose
#endif
#endif

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_encoder_chess.hpp"
#include "cypha/bench/bench_encoder_document.hpp"
#include "cypha/bench/bench_encoder_go.hpp"
#include "cypha/bench/bench_encoder_image.hpp"
#include "cypha/bench/bench_encoder_poker.hpp"
#include "cypha/bench/bench_encoder_text.hpp"
#include "cypha/bench/bench_encoder_timeseries.hpp"
#include "cypha/bench/bench_baselines.hpp"
#include "cypha/bench/bench_figures.hpp"
#include "cypha/bench/bench_metrics.hpp"
#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_profile.hpp"
#include "cypha/bench/bench_report_json.hpp"
#include "cypha/bench/bench_cross_domain.hpp"
#include "cypha/bench/bench_report.hpp"
#include "cypha/bench/bench_tune.hpp"
#include "cypha/create_model.hpp"
#include "cypha/csv_ingest.hpp"
#include "cypha/ewc_regularizer.hpp"
#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/cyphalm_parallel.hpp"
#include "cypha/cyphalm/ssm_diagnose.hpp"
#include "cypha/intelligence/profile_from_model.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/sync_infer.hpp"
#include "cypha/train_step_vector.hpp"

#include "cypha/bench/bench_domains.hpp"

namespace cypha::bench {

namespace {


using Json = nlohmann::json;

namespace fs = std::filesystem;

int domain_number_impl(const std::string& tag) {
    int n = 0;
    for (std::size_t i = 1; i < tag.size() && std::isdigit(static_cast<unsigned char>(tag[i])); ++i) {
        n = n * 10 + (tag[i] - '0');
    }
    return n;
}

fs::path g_tool_dir;
bool g_ssm_diagnose = false;

}  // namespace

void set_tool_dir(const fs::path& dir) { g_tool_dir = dir; }

void set_ssm_diagnose(bool enabled) { g_ssm_diagnose = enabled; }

int domain_number(const std::string& tag) { return domain_number_impl(tag); }

namespace {

bool bench_fast_mode() {
    const char* v = std::getenv("CYPHA_BENCH_FAST");
    if (v == nullptr) {
        return false;
    }
    const std::string s(v);
    return s == "1" || s == "true" || s == "True" || s == "yes";
}

std::string capture_process_output(const std::string& cmd) {
    FILE* fp = popen(cmd.c_str(), "r");
    if (fp == nullptr) {
        throw std::runtime_error("popen failed");
    }
    std::string out;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), fp) != nullptr) {
        out += buf;
    }
    pclose(fp);
    return out;
}

std::mt19937 make_rng(std::uint64_t seed) {
    return std::mt19937(static_cast<std::mt19937::result_type>(seed));
}

Json make_synthetic_classification(int n, int d, std::uint64_t seed) {
    std::mt19937 rng = make_rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<std::vector<double>> xs(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(d)));
    std::vector<std::string> ys(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < d; ++j) {
            const double v = gauss(rng);
            xs[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = v;
            sum += v;
        }
        ys[static_cast<std::size_t>(i)] = (sum > 0.0) ? "1" : "0";
    }
    return Json{{"xs", xs}, {"ys", ys}, {"input_dim", d}, {"n", n}};
}

struct TabularDataset {
    std::string name;
    std::string source;
    std::vector<std::vector<double>> x;
    std::vector<std::string> y;
};

TabularDataset make_synthetic_tabular(const std::string& name, int n, int d, int n_classes, std::uint64_t seed) {
    std::mt19937 rng = make_rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    TabularDataset ds;
    ds.name = name;
    ds.source = "synthetic";
    ds.x.resize(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(d)));
    ds.y.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int cls = i % std::max(1, n_classes);
        for (int j = 0; j < d; ++j) {
            ds.x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                gauss(rng) + static_cast<double>(cls) * 1.5;
        }
        ds.y[static_cast<std::size_t>(i)] = std::to_string(cls);
    }
    return ds;
}

TabularDataset load_tabular_dataset(const std::string& name, int fallback_n, int fallback_d, int fallback_classes,
                                    std::uint64_t seed) {
    const std::filesystem::path csv = cypha::bench::data_dir() / (name + ".csv");
    if (std::filesystem::is_regular_file(csv)) {
        cypha::CsvDenseSpec spec;
        spec.has_header = true;
        spec.target_col_name = "target";
        const cypha::CsvDenseResult loaded = cypha::load_csv_dense(csv, spec);
        if (loaded.n_rows > 0 && loaded.n_features > 0) {
            TabularDataset ds;
            ds.name = name;
            ds.source = "csv";
            ds.x.resize(static_cast<std::size_t>(loaded.n_rows),
                        std::vector<double>(static_cast<std::size_t>(loaded.n_features)));
            for (int r = 0; r < loaded.n_rows; ++r) {
                const double* row = loaded.x_rowmajor.data() + static_cast<std::size_t>(r * loaded.n_features);
                std::copy(row, row + loaded.n_features, ds.x[static_cast<std::size_t>(r)].begin());
            }
            ds.y = loaded.y_class;
            return ds;
        }
    }
    return make_synthetic_tabular(name, fallback_n, fallback_d, fallback_classes, seed);
}

void standardize_train_test(std::vector<std::vector<double>>& train_x, std::vector<std::vector<double>>& test_x) {
    if (train_x.empty()) return;
    const int d = static_cast<int>(train_x.front().size());
    std::vector<double> mean(static_cast<std::size_t>(d), 0.0);
    std::vector<double> stdv(static_cast<std::size_t>(d), 1.0);
    for (const auto& row : train_x) {
        for (int j = 0; j < d; ++j) mean[static_cast<std::size_t>(j)] += row[static_cast<std::size_t>(j)];
    }
    for (double& m : mean) m /= static_cast<double>(train_x.size());
    for (const auto& row : train_x) {
        for (int j = 0; j < d; ++j) {
            const double diff = row[static_cast<std::size_t>(j)] - mean[static_cast<std::size_t>(j)];
            stdv[static_cast<std::size_t>(j)] += diff * diff;
        }
    }
    for (double& s : stdv) {
        s = std::sqrt(s / static_cast<double>(train_x.size()));
        if (s < 1e-12) s = 1.0;
    }
    auto apply = [&](std::vector<std::vector<double>>& rows) {
        for (auto& row : rows) {
            for (int j = 0; j < d; ++j) {
                row[static_cast<std::size_t>(j)] =
                    (row[static_cast<std::size_t>(j)] - mean[static_cast<std::size_t>(j)]) / stdv[static_cast<std::size_t>(j)];
            }
        }
    };
    apply(train_x);
    apply(test_x);
}

void stratified_split(const TabularDataset& ds, double test_frac, std::uint64_t seed,
                      std::vector<std::vector<double>>& train_x, std::vector<std::string>& train_y,
                      std::vector<std::vector<double>>& test_x, std::vector<std::string>& test_y) {
    std::unordered_map<std::string, std::vector<int>> by_class;
    for (int i = 0; i < static_cast<int>(ds.y.size()); ++i) by_class[ds.y[static_cast<std::size_t>(i)]].push_back(i);
    std::mt19937 rng = make_rng(seed);
    for (auto& kv : by_class) {
        auto& idx = kv.second;
        std::shuffle(idx.begin(), idx.end(), rng);
        const int n_test = std::max(1, static_cast<int>(std::round(static_cast<double>(idx.size()) * test_frac)));
        for (int k = 0; k < static_cast<int>(idx.size()); ++k) {
            const int i = idx[static_cast<std::size_t>(k)];
            if (k < n_test) {
                test_x.push_back(ds.x[static_cast<std::size_t>(i)]);
                test_y.push_back(ds.y[static_cast<std::size_t>(i)]);
            } else {
                train_x.push_back(ds.x[static_cast<std::size_t>(i)]);
                train_y.push_back(ds.y[static_cast<std::size_t>(i)]);
            }
        }
    }
}

std::vector<double> flatten_rowmajor(const std::vector<std::vector<double>>& rows) {
    if (rows.empty()) {
        return {};
    }
    const int d = static_cast<int>(rows.front().size());
    std::vector<double> out;
    out.reserve(rows.size() * static_cast<std::size_t>(d));
    for (const auto& row : rows) {
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

void fit_apply_preprocessor(std::vector<std::vector<double>>& train_x, std::vector<std::vector<double>>& test_x,
                            const Json& pre_cfg, Json* meta_out) {
    if (train_x.empty()) {
        return;
    }
    cypha::PreprocessorState pre;
    pre.scale = pre_cfg.value("scale", true);
    pre.pca_dim = pre_cfg.value("pca_dim", -1);
    pre.rff_dim = pre_cfg.value("rff_dim", -1);
    pre.auto_rff_gamma = pre_cfg.value("auto_rff_gamma", false);
    pre.auto_rff_gamma_cv = pre_cfg.value("auto_rff_gamma_cv", false);
    pre.seed = pre_cfg.value("seed", 42);
    const int n = static_cast<int>(train_x.size());
    const int d = static_cast<int>(train_x.front().size());
    const std::vector<double> flat = flatten_rowmajor(train_x);
    pre.fit_from_design_matrix(flat, n, d);
    for (auto& row : train_x) {
        row = pre.transform_one(row);
    }
    for (auto& row : test_x) {
        row = pre.transform_one(row);
    }
    if (meta_out != nullptr) {
        *meta_out = Json{{"scale", pre.scale},
                         {"pca_dim", pre.pca_dim},
                         {"rff_dim", pre.rff_dim},
                         {"auto_rff_gamma", pre.auto_rff_gamma},
                         {"auto_rff_gamma_cv", pre.auto_rff_gamma_cv},
                         {"rff_gamma", pre.rff_gamma},
                         {"input_dim", pre.input_dim},
                         {"output_dim", pre.output_dim}};
    }
}

Json train_eval_vectors(const std::vector<std::vector<double>>& train_x, const std::vector<std::string>& train_y,
                        const std::vector<std::vector<double>>& test_x, const std::vector<std::string>& test_y,
                        const cypha::bench::ProfileJson& regime, const std::string& dataset_name) {
    if (train_x.empty() || test_x.empty()) {
        return Json{{"dataset", dataset_name}, {"accuracy", 0.0}, {"n_train", 0}, {"n_test", 0}, {"expert_count", 0}};
    }
    std::vector<std::vector<double>> tr = train_x;
    std::vector<std::vector<double>> te = test_x;
    Json preprocessor_meta;
    if (regime.contains("preprocessor") && regime["preprocessor"].is_object()) {
        fit_apply_preprocessor(tr, te, regime["preprocessor"], &preprocessor_meta);
    }
    const int d = static_cast<int>(tr.front().size());

    cypha::FreshModelParams fp;
    fp.input_dim = d;
    fp.field_dim = regime.value("field_dim", 128);
    fp.world_lr = regime.value("world_lr", 0.008);
    fp.delta_lr = regime.value("delta_lr", 0.03);
    fp.temperature = regime.value("temperature", 1.0);

    const cypha::CNode root = cypha::create_fresh_model_root(fp);
    cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim);
    cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
    cypha::ReplayBuffer replay(10000);
    cypha::TrainStepParams tsp;
    tsp.enc_lr = regime.value("enc_lr", 0.002);
    tsp.replay_ratio = regime.value("replay_ratio", 0.0);
    tsp.replay_cap = 10000;

    const double world_lr = fp.world_lr;
    const double delta_lr = fp.delta_lr;
    const double ood_sigma = regime.value("ood_sigma", 12.0);
    std::mt19937 rng = make_rng(42);
    int enc_updates = 0;

    const int train_n = static_cast<int>(tr.size());
    const int passes = std::max(1, regime.value("n_epochs", 1));
    for (int p = 0; p < passes; ++p) {
        std::vector<int> order(static_cast<std::size_t>(train_n));
        for (int i = 0; i < train_n; ++i) order[static_cast<std::size_t>(i)] = i;
        std::shuffle(order.begin(), order.end(), make_rng(42 + static_cast<std::uint64_t>(p)));
        for (int idx : order) {
            cypha::dif_train_step_vector(infer, mem, replay, tr[static_cast<std::size_t>(idx)].data(), d,
                                         train_y[static_cast<std::size_t>(idx)], world_lr, delta_lr, world_lr,
                                         delta_lr, ood_sigma, tsp, rng, enc_updates, nullptr, nullptr);
        }
    }
    cypha::sync_infer_model_from_memory(infer, mem);

    std::vector<std::string> y_true;
    std::vector<std::string> y_pred;
    y_true.reserve(te.size());
    y_pred.reserve(te.size());
    for (std::size_t i = 0; i < te.size(); ++i) {
        y_true.push_back(test_y[i]);
        std::vector<double> llr;
        cypha::batch_llr_from_x(infer, te[i].data(), 1, llr);
        int best = 0;
        for (int k = 1; k < static_cast<int>(infer.labels.size()); ++k) {
            if (llr[static_cast<std::size_t>(k)] > llr[static_cast<std::size_t>(best)]) best = k;
        }
        y_pred.push_back(infer.labels.empty() ? "0" : infer.labels[static_cast<std::size_t>(best)]);
    }

    Json result = Json{
        {"dataset", dataset_name},
        {"cypha_scores", Json{{"accuracy", cypha::bench::accuracy(y_true, y_pred)}}},
        {"baselines", cypha::bench::offline_classification_baselines_json(tr, train_y, te, test_y)},
        {"n_train", train_n},
        {"n_test", static_cast<int>(te.size())},
        {"expert_count", static_cast<int>(infer.labels.size())},
        {"backend", "cypha_core"},
    };
    if (!preprocessor_meta.empty()) {
        result["preprocessor"] = preprocessor_meta;
    }
    return result;
}

Json run_tabular_dataset(const TabularDataset& ds, const cypha::bench::ProfileJson& regime) {
    std::vector<std::vector<double>> train_x;
    std::vector<std::string> train_y;
    std::vector<std::vector<double>> test_x;
    std::vector<std::string> test_y;
    stratified_split(ds, 0.2, 42, train_x, train_y, test_x, test_y);
    standardize_train_test(train_x, test_x);
    Json result = train_eval_vectors(train_x, train_y, test_x, test_y, regime, ds.name);
    result["data_source"] = ds.source;
    return result;
}

Json train_eval_classifier(const Json& data, const cypha::bench::ProfileJson& regime, const std::string& task) {
    const int n = data.at("n").get<int>();
    const auto& xs = data.at("xs");
    const auto& ys = data.at("ys");

    const int train_n = static_cast<int>(static_cast<double>(n) * 0.8);
    std::vector<std::vector<double>> train_x;
    std::vector<std::string> train_y;
    std::vector<std::vector<double>> test_x;
    std::vector<std::string> test_y;
    train_x.reserve(static_cast<std::size_t>(train_n));
    train_y.reserve(static_cast<std::size_t>(train_n));
    for (int i = 0; i < train_n; ++i) {
        train_x.push_back(xs[i].get<std::vector<double>>());
        train_y.push_back(ys[i].get<std::string>());
    }
    for (int i = train_n; i < n; ++i) {
        test_x.push_back(xs[i].get<std::vector<double>>());
        test_y.push_back(ys[i].get<std::string>());
    }
    Json out = train_eval_vectors(train_x, train_y, test_x, test_y, regime, task);
    out["task"] = task;
    if (out.contains("cypha_scores")) {
        out["accuracy"] = out["cypha_scores"]["accuracy"];
    }
    return out;
}

constexpr std::uint64_t kBenchSeed = 42;

double field_confidence_proxy(const cypha::CyphaInferModel& m) {
    return std::min(m.mid_n / 200.0, 1.0);
}

struct OnlineClassifier {
    cypha::CyphaInferModel infer;
    cypha::CyphaDifMemoryState mem;
    cypha::ReplayBuffer replay;
    cypha::TrainStepParams tsp;
    std::mt19937 rng;
    int enc_updates{0};
    double world_lr{0.008};
    double delta_lr{0.03};
    double ood_sigma{12.0};
};

OnlineClassifier make_online_classifier(int input_dim, std::uint64_t seed, double enc_lr,
                                        const cypha::bench::ProfileJson& regime) {
    cypha::FreshModelParams fp;
    fp.input_dim = input_dim;
    fp.field_dim = regime.value("field_dim", 128);
    fp.world_lr = regime.value("world_lr", 0.008);
    fp.delta_lr = regime.value("delta_lr", 0.03);
    fp.temperature = regime.value("temperature", 1.0);

    const cypha::CNode root = cypha::create_fresh_model_root(fp);
    OnlineClassifier c{
        cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim),
        cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim),
        cypha::ReplayBuffer(10000),
        cypha::TrainStepParams{},
        make_rng(seed),
        0,
        fp.world_lr,
        fp.delta_lr,
        regime.value("ood_sigma", 12.0),
    };
    c.tsp.enc_lr = enc_lr;
    c.tsp.replay_ratio = regime.value("replay_ratio", 0.0);
    c.tsp.replay_cap = 10000;
    return c;
}

double online_clf_train_step(OnlineClassifier& c, const std::vector<double>& x, const std::string& label) {
    const int d = static_cast<int>(x.size());
    return cypha::dif_train_step_vector(c.infer, c.mem, c.replay, x.data(), d, label, c.world_lr, c.delta_lr,
                                         c.world_lr, c.delta_lr, c.ood_sigma, c.tsp, c.rng, c.enc_updates, nullptr,
                                         nullptr);
}

std::string online_clf_predict(const cypha::CyphaInferModel& m, const std::vector<double>& x) {
    std::vector<double> llr;
    cypha::batch_llr_from_x(m, x.data(), 1, llr);
    if (m.labels.empty()) return "0";
    int best = 0;
    for (int k = 1; k < static_cast<int>(m.labels.size()); ++k) {
        if (llr[static_cast<std::size_t>(k)] > llr[static_cast<std::size_t>(best)]) best = k;
    }
    return m.labels[static_cast<std::size_t>(best)];
}

double online_clf_epistemic(const cypha::CyphaInferModel& m, const std::vector<double>& x) {
    std::vector<double> llr;
    cypha::batch_llr_from_x(m, x.data(), 1, llr);
    const int k = static_cast<int>(m.labels.size());
    if (k <= 1) return 0.0;
    std::vector<double> probs;
    cypha::softmax_batch_reference(llr.data(), 1, k, 1e-12, probs);
    return cypha::regression::mke_routing_entropy(probs.data(), k, 1e-12);
}

Json clf_metrics_native(const cypha::CyphaInferModel& m, const std::vector<std::vector<double>>& xs,
                        const std::vector<std::string>& ys) {
    std::vector<std::string> y_true;
    std::vector<std::string> y_pred;
    std::vector<double> epistemic;
    y_true.reserve(xs.size());
    y_pred.reserve(xs.size());
    epistemic.reserve(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        y_true.push_back(ys[i]);
        y_pred.push_back(online_clf_predict(m, xs[i]));
        epistemic.push_back(online_clf_epistemic(m, xs[i]));
    }
    std::vector<double> errors;
    errors.reserve(xs.size());
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        errors.push_back(y_true[i] == y_pred[i] ? 0.0 : 1.0);
    }
    double mean_epi = 0.0;
    if (!epistemic.empty()) {
        for (double v : epistemic) mean_epi += v;
        mean_epi /= static_cast<double>(epistemic.size());
    }
    return Json{
        {"accuracy", cypha::bench::accuracy(y_true, y_pred)},
        {"mean_epistemic_var", mean_epi},
        {"expert_count", static_cast<int>(m.labels.size())},
    };
}

void train_classifier_online(OnlineClassifier& c, const std::vector<std::vector<double>>& xs,
                             const std::vector<std::string>& ys, int passes, std::uint64_t seed) {
    const int n = static_cast<int>(xs.size());
    for (int p = 0; p < passes; ++p) {
        std::vector<int> order(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
        std::shuffle(order.begin(), order.end(), make_rng(seed + static_cast<std::uint64_t>(p)));
        for (int idx : order) {
            (void)online_clf_train_step(c, xs[static_cast<std::size_t>(idx)], ys[static_cast<std::size_t>(idx)]);
        }
    }
    cypha::sync_infer_model_from_memory(c.infer, c.mem);
}

cypha::bench::GrayImage downsample_nearest(const cypha::bench::GrayImage& src, int out_rows, int out_cols) {
    cypha::bench::GrayImage out;
    out.rows = out_rows;
    out.cols = out_cols;
    out.pixels.assign(static_cast<std::size_t>(out_rows * out_cols), 0);
    for (int y = 0; y < out_rows; ++y) {
        for (int x = 0; x < out_cols; ++x) {
            const int sy = std::min(src.rows - 1, y * src.rows / out_rows);
            const int sx = std::min(src.cols - 1, x * src.cols / out_cols);
            out.pixels[static_cast<std::size_t>(y * out_cols + x)] =
                src.pixels[static_cast<std::size_t>(sy * src.cols + sx)];
        }
    }
    return out;
}

std::vector<double> raw_digit_features(const cypha::bench::GrayImage& img8) {
    std::vector<double> out(64);
    for (int i = 0; i < 64; ++i) out[static_cast<std::size_t>(i)] = static_cast<double>(img8.pixels[static_cast<std::size_t>(i)]) / 16.0;
    return out;
}

struct DigitDataset {
    std::vector<std::vector<double>> train_x;
    std::vector<std::string> train_y;
    std::vector<std::vector<double>> test_x;
    std::vector<std::string> test_y;
};

DigitDataset load_digits_raw_dataset() {
    cypha::bench::VisionDataset vis = cypha::bench::load_vision_dataset();
    DigitDataset ds;
    auto encode = [](const cypha::bench::GrayImage& img) {
        return raw_digit_features(downsample_nearest(img, 8, 8));
    };
    for (std::size_t i = 0; i < vis.train_images.size(); ++i) {
        ds.train_x.push_back(encode(vis.train_images[i]));
        ds.train_y.push_back(vis.train_labels[i]);
    }
    for (std::size_t i = 0; i < vis.test_images.size(); ++i) {
        ds.test_x.push_back(encode(vis.test_images[i]));
        ds.test_y.push_back(vis.test_labels[i]);
    }
    standardize_train_test(ds.train_x, ds.test_x);
    return ds;
}

DigitDataset load_digits_hog_dataset() {
    cypha::bench::VisionDataset vis = cypha::bench::load_vision_dataset();
    cypha::bench::ImageEncoder enc;
    DigitDataset ds;
    auto encode = [&](const cypha::bench::GrayImage& img) {
        const auto small = downsample_nearest(img, 8, 8);
        const auto hog = enc.hog_features(small, 4, 9);
        std::vector<double> row(hog.size());
        for (std::size_t i = 0; i < hog.size(); ++i) row[i] = static_cast<double>(hog[i]);
        return row;
    };
    for (std::size_t i = 0; i < vis.train_images.size(); ++i) {
        ds.train_x.push_back(encode(vis.train_images[i]));
        ds.train_y.push_back(vis.train_labels[i]);
    }
    for (std::size_t i = 0; i < vis.test_images.size(); ++i) {
        ds.test_x.push_back(encode(vis.test_images[i]));
        ds.test_y.push_back(vis.test_labels[i]);
    }
    standardize_train_test(ds.train_x, ds.test_x);
    return ds;
}

double gzip_compression_ratio_bytes(const std::vector<std::uint8_t>& original) {
    if (original.empty()) return 1.0;
    z_stream strm{};
    if (deflateInit2(&strm, 9, Z_DEFLATED, 15 + 16, 9, Z_DEFAULT_STRATEGY) != Z_OK) return 1.0;
    std::vector<std::uint8_t> out(deflateBound(&strm, static_cast<uLong>(original.size())));
    strm.next_in = const_cast<Bytef*>(original.data());
    strm.avail_in = static_cast<uInt>(original.size());
    strm.next_out = out.data();
    strm.avail_out = static_cast<uInt>(out.size());
    (void)deflate(&strm, Z_FINISH);
    const uLong compressed = strm.total_out;
    deflateEnd(&strm);
    return static_cast<double>(original.size()) / static_cast<double>(std::max<uLong>(compressed, 1));
}

double gzip_compression_ratio_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return 1.0;
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return gzip_compression_ratio_bytes(bytes);
}

std::unordered_map<std::string, std::filesystem::path> iter_canterbury_files() {
    static const char* kNames[] = {"alice29.txt",  "asyoulik.txt", "cp.html",      "fields.c",
                                   "grammar.lsp",  "kennedy.xls",  "lcet10.txt",   "plrabn12.txt",
                                   "ptt5",         "sum",          "xargs.1"};
    std::unordered_map<std::string, std::filesystem::path> found;
    const std::filesystem::path base = cypha::bench::data_dir() / "canterbury";
    if (std::filesystem::is_directory(base)) {
        for (const char* name : kNames) {
            const std::filesystem::path p = base / name;
            if (std::filesystem::is_regular_file(p)) found[name] = p;
        }
    }
    if (!found.empty()) return found;

    const std::filesystem::path synth_dir = cypha::bench::data_dir() / "canterbury_synthetic";
    std::filesystem::create_directories(synth_dir);
    std::mt19937 rng = make_rng(kBenchSeed);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    const std::string unit = "the quick brown fox jumps over the lazy dog";
    std::string prose;
    for (int i = 0; i < 200; ++i) {
        if (i > 0) prose.push_back(' ');
        prose += unit;
    }
    std::ostringstream code;
    for (int i = 0; i < 100; ++i) code << "int func_" << i << "() { return " << i << "; }\n";
    std::string binary;
    binary.reserve(4096);
    for (int i = 0; i < 4096; ++i) binary.push_back(static_cast<char>(byte_dist(rng)));
    struct SynthFile {
        const char* name;
        std::string text;
    };
    const SynthFile synth[] = {
        {"synthetic_prose.txt", prose},
        {"synthetic_code.c", code.str()},
        {"synthetic_binary.bin", binary},
    };
    for (const auto& sf : synth) {
        const std::filesystem::path p = synth_dir / sf.name;
        std::ofstream out(p, std::ios::binary);
        out.write(sf.text.data(), static_cast<std::streamsize>(sf.text.size()));
        found[sf.name] = p;
    }
    return found;
}

std::string read_text_lossy(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();
    for (char& ch : text) {
        if (static_cast<unsigned char>(ch) > 127) ch = '?';
    }
    return text;
}

Json train_stream_on_text(const std::string& text, int max_steps, const cypha::bench::ProfileJson& regime) {
    cypha::bench::CharNgramEncoder enc(5, 128);
    const std::size_t vocab_len = std::min<std::size_t>(text.size(), 50000);
    enc.build_vocab(text.substr(0, vocab_len));
    const int dim = std::max(enc.dim(), 1);
    OnlineClassifier clf = make_online_classifier(dim, kBenchSeed, regime.value("enc_lr", 0.002), regime);

    std::vector<double> losses;
    std::vector<double> alphas;
    const int n = static_cast<int>(text.size());
    const int limit = std::min(max_steps, std::max(0, n - enc.dim()));
    for (int step = 0; step < limit; ++step) {
        const std::size_t end = std::min(text.size(), static_cast<std::size_t>(step + enc.dim() + 20));
        const std::string window = text.substr(static_cast<std::size_t>(step), end - static_cast<std::size_t>(step));
        const auto xf = enc.encode(window);
        double sum = 0.0;
        for (float v : xf) sum += v;
        if (sum == 0.0) continue;
        std::vector<double> x(xf.size());
        for (std::size_t i = 0; i < xf.size(); ++i) x[i] = static_cast<double>(xf[i]);
        const char next_char = (static_cast<std::size_t>(step + enc.dim()) < text.size())
                                   ? text[static_cast<std::size_t>(step + enc.dim())]
                                   : text.back();
        const std::string label = "c" + std::to_string(static_cast<unsigned char>(next_char) % 64);
        const double loss = online_clf_train_step(clf, x, label);
        losses.push_back(loss);
        alphas.push_back(field_confidence_proxy(clf.infer));
    }
    cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
    double mean_loss = std::numeric_limits<double>::quiet_NaN();
    double mean_alpha = 0.5;
    if (!losses.empty()) {
        mean_loss = 0.0;
        for (double v : losses) mean_loss += v;
        mean_loss /= static_cast<double>(losses.size());
    }
    if (!alphas.empty()) {
        mean_alpha = 0.0;
        for (double v : alphas) mean_alpha += v;
        mean_alpha /= static_cast<double>(alphas.size());
    }
    return Json{
        {"mean_loss", mean_loss},
        {"mean_alpha_proxy", mean_alpha},
        {"expert_count", static_cast<int>(clf.infer.labels.size())},
        {"steps", static_cast<int>(losses.size())},
    };
}

struct RegExpertStat {
    std::vector<double> mu;
    double var_ema{0.0};
    int n_updates{0};
};

std::string pick_dif_regressor_expert(int step, int n_existing, int k_target, cypha::CyphaInferModel& infer,
                                      const double* x, int d) {
    if (n_existing < k_target && step <= k_target * 20) {
        return "_e" + std::to_string(step % k_target);
    }
    if (n_existing == 0) return "_e0";
    std::vector<double> llr;
    cypha::batch_llr_from_x(infer, x, 1, llr);
    int bi = 0;
    for (int k = 1; k < static_cast<int>(infer.labels.size()); ++k) {
        if (llr[static_cast<std::size_t>(k)] > llr[static_cast<std::size_t>(bi)]) bi = k;
    }
    return infer.labels[static_cast<std::size_t>(bi)];
}

struct OnlineRegressor {
    cypha::CyphaInferModel infer;
    cypha::CyphaDifMemoryState mem;
    cypha::ReplayBuffer replay;
    cypha::TrainStepParams tsp;
    std::mt19937 rng;
    int enc_updates{0};
    double world_lr{0.015};
    double delta_lr{0.05};
    double ood_sigma{12.0};
    double target_lr{0.08};
    int n_experts_cap{8};
    std::unordered_map<std::string, RegExpertStat> experts;
    int total_steps{0};
};

OnlineRegressor make_online_regressor(int input_dim, std::uint64_t seed, const cypha::bench::ProfileJson& regime) {
    cypha::FreshModelParams fp;
    fp.input_dim = input_dim;
    fp.field_dim = regime.value("field_dim", 128);
    fp.world_lr = regime.value("world_lr", 0.015);
    fp.delta_lr = regime.value("delta_lr", 0.05);
    fp.temperature = regime.value("temperature", 1.0);

    const cypha::CNode root = cypha::create_fresh_model_root(fp);
    OnlineRegressor r{
        cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim),
        cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim),
        cypha::ReplayBuffer(10000),
        cypha::TrainStepParams{},
        make_rng(seed),
        0,
        fp.world_lr,
        fp.delta_lr,
        regime.value("ood_sigma", 12.0),
        regime.value("target_lr", 0.08),
        std::max(2, regime.value("n_experts", 4)),
        {},
        0,
    };
    r.tsp.replay_ratio = regime.value("replay_ratio", 0.0);
    r.tsp.replay_cap = 10000;
    return r;
}

void online_reg_train_step(OnlineRegressor& r, const std::vector<double>& x, double y) {
    ++r.total_steps;
    const int d = static_cast<int>(x.size());
    const int k_target = std::max(r.n_experts_cap, 4);
    const int n_existing = static_cast<int>(r.mem.labels.size());
    const std::string expert = pick_dif_regressor_expert(r.total_steps, n_existing, k_target, r.infer, x.data(), d);
    (void)cypha::dif_train_step_vector(r.infer, r.mem, r.replay, x.data(), d, expert, r.world_lr, r.delta_lr,
                                       r.world_lr, r.delta_lr, r.ood_sigma, r.tsp, r.rng, r.enc_updates, nullptr,
                                       nullptr);
    auto& st = r.experts[expert];
    cypha::regression::expert_target_ema_step(st.mu, st.var_ema, st.n_updates, &y, 1, r.target_lr);
}

void online_reg_predict(const OnlineRegressor& r, const std::vector<double>& x, double& y_hat, double& unc) {
    std::vector<double> llr;
    cypha::batch_llr_from_x(r.infer, x.data(), 1, llr);
    const int k = static_cast<int>(r.infer.labels.size());
    if (k == 0) {
        y_hat = 0.0;
        unc = 0.0;
        return;
    }
    std::vector<double> probs;
    cypha::softmax_batch_reference(llr.data(), 1, k, 1e-12, probs);
    std::vector<double> mu;
    std::vector<double> var;
    mu.reserve(static_cast<std::size_t>(k));
    var.reserve(static_cast<std::size_t>(k));
    for (const auto& lbl : r.infer.labels) {
        const auto it = r.experts.find(lbl);
        if (it == r.experts.end()) {
            mu.push_back(0.0);
            var.push_back(0.0);
        } else {
            mu.push_back(it->second.mu.empty() ? 0.0 : it->second.mu[0]);
            var.push_back(it->second.var_ema);
        }
    }
    cypha::regression::predict_mixture_scalar(probs.data(), mu.data(), var.data(), static_cast<std::size_t>(k), y_hat,
                                              unc);
}

Json reg_metrics_native(const OnlineRegressor& r, const std::vector<std::vector<double>>& xs,
                        const std::vector<double>& ys) {
    std::vector<double> preds;
    std::vector<double> unc;
    preds.reserve(xs.size());
    unc.reserve(xs.size());
    for (const auto& x : xs) {
        double y_hat = 0.0;
        double u = 0.0;
        online_reg_predict(r, x, y_hat, u);
        preds.push_back(y_hat);
        unc.push_back(u);
    }
    double mean_unc_sq = 0.0;
    if (!unc.empty()) {
        for (double v : unc) mean_unc_sq += v * v;
        mean_unc_sq /= static_cast<double>(unc.size());
    }
    return Json{
        {"rmse", cypha::bench::rmse(ys, preds)},
        {"mae", cypha::bench::mae(ys, preds)},
        {"r2", cypha::bench::r2(ys, preds)},
        {"mean_epistemic_var", mean_unc_sq},
        {"expert_count", static_cast<int>(r.experts.size())},
    };
}

struct MultitaskBundle {
    std::string name;
    std::vector<std::vector<double>> train_x;
    std::vector<std::string> train_y;
    std::vector<std::vector<double>> test_x;
    std::vector<std::string> test_y;
};

std::vector<MultitaskBundle> load_multitask_datasets() {
    std::vector<MultitaskBundle> out;
    out.push_back({});
    out.back().name = "iris";
    {
        TabularDataset iris = load_tabular_dataset("iris", 150, 4, 3, 1);
        stratified_split(iris, 0.25, kBenchSeed, out.back().train_x, out.back().train_y, out.back().test_x,
                         out.back().test_y);
        standardize_train_test(out.back().train_x, out.back().test_x);
    }
    out.push_back({});
    out.back().name = "wine";
    {
        TabularDataset wine = load_tabular_dataset("wine", 178, 13, 3, 2);
        stratified_split(wine, 0.25, kBenchSeed, out.back().train_x, out.back().train_y, out.back().test_x,
                         out.back().test_y);
        standardize_train_test(out.back().train_x, out.back().test_x);
    }
    out.push_back({});
    out.back().name = "digits";
    {
        DigitDataset digits = load_digits_raw_dataset();
        out.back().train_x = std::move(digits.train_x);
        out.back().train_y = std::move(digits.train_y);
        out.back().test_x = std::move(digits.test_x);
        out.back().test_y = std::move(digits.test_y);
    }
    return out;
}

std::vector<double> pad_to_max(const std::vector<double>& x, int max_dim) {
    if (static_cast<int>(x.size()) >= max_dim) return x;
    std::vector<double> out(static_cast<std::size_t>(max_dim), 0.0);
    std::copy(x.begin(), x.end(), out.begin());
    return out;
}

Json run_d01() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    Json tasks = Json::array();
    tasks.push_back(train_eval_classifier(make_synthetic_classification(400, 10, 42), regime, "linearly_separable_2class"));
    tasks.push_back(train_eval_classifier(make_synthetic_classification(400, 8, 7), regime, "4_gaussian_blobs"));
    const Json experiments{{"tasks", tasks}, {"backend", "cypha_core"}};
    cypha::bench::finalize_domain("d01", experiments);
    return experiments;
}

struct LocalExpertStat {
    std::vector<double> mu;
    double var_ema{0.0};
    int n_updates{0};
};

Json run_d02() {
    const cypha::bench::ProfileJson regime = cypha::bench::regression_params();
    const int d = 12;
    const int n = 300;
    std::mt19937 rng = make_rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    cypha::FreshModelParams fp;
    fp.input_dim = d;
    fp.field_dim = regime.value("field_dim", 128);
    fp.world_lr = regime.value("world_lr", 0.015);
    fp.delta_lr = regime.value("delta_lr", 0.05);
    const cypha::CNode root = cypha::create_fresh_model_root(fp);
    cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root, nullptr, fp.field_dim);
    cypha::CyphaDifMemoryState mem = cypha::CyphaDifMemoryState::from_cypha_root(root, nullptr, fp.field_dim);
    cypha::ReplayBuffer replay(10000);
    cypha::TrainStepParams tsp;
    tsp.replay_ratio = 0.0;
    std::mt19937 trng = make_rng(42);
    int enc_updates = 0;

    const int k_target = std::max(2, regime.value("n_experts", 4));
    std::unordered_map<std::string, LocalExpertStat> experts;
    std::vector<double> xs(static_cast<std::size_t>(d));
    std::vector<double> y_train;
    std::vector<double> y_test;
    std::vector<std::vector<double>> x_train;
    std::vector<std::vector<double>> x_test;

    for (int i = 0; i < n; ++i) {
        double y = 0.0;
        for (int j = 0; j < d; ++j) {
            xs[static_cast<std::size_t>(j)] = gauss(rng);
            y += xs[static_cast<std::size_t>(j)] * (0.2 + 0.05 * j);
        }
        y += 0.1 * gauss(rng);
        const bool is_test = i >= static_cast<int>(static_cast<double>(n) * 0.8);
        if (is_test) {
            y_test.push_back(y);
            x_test.push_back(xs);
            continue;
        }
        y_train.push_back(y);
        x_train.push_back(xs);
        const int step = static_cast<int>(y_train.size());
        const int n_existing = static_cast<int>(experts.size());
        std::string expert;
        if (n_existing < k_target && step <= k_target * 20) {
            expert = "_e" + std::to_string(step % k_target);
        } else if (n_existing == 0) {
            expert = "_e0";
        } else {
            std::vector<double> llr;
            cypha::batch_llr_from_x(infer, xs.data(), 1, llr);
            int bi = 0;
            for (int k = 1; k < static_cast<int>(infer.labels.size()); ++k) {
                if (llr[static_cast<std::size_t>(k)] > llr[static_cast<std::size_t>(bi)]) bi = k;
            }
            expert = infer.labels[static_cast<std::size_t>(bi)];
        }
        cypha::dif_train_step_vector(infer, mem, replay, xs.data(), d, expert, fp.world_lr, fp.delta_lr,
                                     fp.world_lr, fp.delta_lr, regime.value("ood_sigma", 12.0), tsp, trng,
                                     enc_updates, nullptr, nullptr);
        auto& st = experts[expert];
        cypha::regression::expert_target_ema_step(st.mu, st.var_ema, st.n_updates, &y, 1,
                                                  regime.value("target_lr", 0.08));
    }
    cypha::sync_infer_model_from_memory(infer, mem);

    std::vector<double> preds;
    preds.reserve(y_test.size());
    for (const auto& xrow : x_test) {
        std::vector<double> llr;
        cypha::batch_llr_from_x(infer, xrow.data(), 1, llr);
        std::vector<double> probs;
        cypha::softmax_batch_reference(llr.data(), 1, static_cast<int>(infer.labels.size()), 1e-12, probs);
        std::vector<double> mu;
        std::vector<double> var;
        mu.reserve(infer.labels.size());
        var.reserve(infer.labels.size());
        for (const auto& lbl : infer.labels) {
            const auto& st = experts[lbl];
            mu.push_back(st.mu.empty() ? 0.0 : st.mu[0]);
            var.push_back(st.var_ema);
        }
        double y_hat = 0.0;
        double unc = 0.0;
        cypha::regression::predict_mixture_scalar(probs.data(), mu.data(), var.data(), mu.size(), y_hat, unc);
        preds.push_back(y_hat);
    }

    const Json experiments{
        {"dataset", "synthetic_regression"},
        {"cypha_scores",
         Json{{"rmse", cypha::bench::rmse(y_test, preds)},
              {"mae", cypha::bench::mae(y_test, preds)},
              {"r2", cypha::bench::r2(y_test, preds)}}},
        {"baselines", cypha::bench::offline_regression_baselines_json(x_train, y_train, x_test, y_test)},
        {"n_experts", static_cast<int>(experts.size())},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d02", experiments);
    return experiments;
}

Json run_d03() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const std::string tabular = "tabular";
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile, &tabular);

    Json datasets = Json::array();
    datasets.push_back(run_tabular_dataset(load_tabular_dataset("iris", 150, 4, 3, 1), regime));
    datasets.push_back(run_tabular_dataset(load_tabular_dataset("wine", 178, 13, 3, 2), regime));

    const Json experiments{
        {"domain", "d03_classification"},
        {"datasets", datasets},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d03", experiments);
    return experiments;
}

std::vector<std::vector<double>> floats_to_doubles(const std::vector<std::vector<float>>& in) {
    std::vector<std::vector<double>> out;
    out.reserve(in.size());
    for (const auto& row : in) {
        std::vector<double> d(row.size());
        for (std::size_t i = 0; i < row.size(); ++i) d[i] = static_cast<double>(row[i]);
        out.push_back(std::move(d));
    }
    return out;
}

void subsample_vision(cypha::bench::VisionDataset& ds, int max_train, int max_test) {
    auto pick = [](std::vector<cypha::bench::GrayImage>& imgs, std::vector<std::string>& labels, int max_n, std::uint64_t seed) {
        if (static_cast<int>(imgs.size()) <= max_n) return;
        std::vector<int> idx(static_cast<std::size_t>(imgs.size()));
        for (int i = 0; i < static_cast<int>(idx.size()); ++i) idx[static_cast<std::size_t>(i)] = i;
        std::shuffle(idx.begin(), idx.end(), make_rng(seed));
        std::vector<cypha::bench::GrayImage> new_imgs;
        std::vector<std::string> new_labels;
        new_imgs.reserve(static_cast<std::size_t>(max_n));
        new_labels.reserve(static_cast<std::size_t>(max_n));
        for (int k = 0; k < max_n; ++k) {
            const int i = idx[static_cast<std::size_t>(k)];
            new_imgs.push_back(std::move(imgs[static_cast<std::size_t>(i)]));
            new_labels.push_back(labels[static_cast<std::size_t>(i)]);
        }
        imgs = std::move(new_imgs);
        labels = std::move(new_labels);
    };
    pick(ds.train_images, ds.train_labels, max_train, 42);
    pick(ds.test_images, ds.test_labels, max_test, 7);
}

Json run_vision_encoding(const cypha::bench::VisionDataset& ds, const std::string& mode,
                         const cypha::bench::ProfileJson& regime) {
    const cypha::bench::ImageEncoder enc;
    const auto train_f = enc.encode_batch(ds.train_images, mode);
    const auto test_f = enc.encode_batch(ds.test_images, mode);
    auto train_x = floats_to_doubles(train_f);
    auto test_x = floats_to_doubles(test_f);
    standardize_train_test(train_x, test_x);
    Json result = train_eval_vectors(train_x, ds.train_labels, test_x, ds.test_labels, regime, mode);
    result["encoding"] = mode;
    return result;
}

Json run_d08() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const std::string vision = "vision";
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile, &vision);

    cypha::bench::VisionDataset ds = cypha::bench::load_vision_dataset();
    subsample_vision(ds, cypha::bench::bench_scale(10000, 2000), cypha::bench::bench_scale(2000, 500));

    Json experiments = Json::array();
    experiments.push_back(run_vision_encoding(ds, "raw", regime));
    experiments.push_back(run_vision_encoding(ds, "hog", regime));

    const Json payload{
        {"domain", "d08_computer_vision"},
        {"data_source", ds.source},
        {"experiments", experiments},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d08", payload);
    return payload;
}

Json run_d10_ssm_diagnose() {
    cypha::bench::TimeSeriesEncoder enc(32, 16);
    const auto ecg = cypha::bench::load_ecg5000(kBenchSeed);
    cypha::cyphalm::CellAISSMConfig cfg;
    cfg.d_input = enc.feature_dim();
    cfg.d_state = 128;
    cfg.tau_fast = 10.0;
    cfg.tau_slow = 100.0;
    cfg.n_layers = 2;
    cfg.seed = static_cast<int>(kBenchSeed + 1);
    cfg.use_spectral_pde = true;
    cfg.use_multiscale = true;
    cfg.use_sparse_hebbian = true;
    cypha::cyphalm::CellAISSM ssm(cfg);

    const int steps = cypha::bench::bench_scale(512, 128);
    std::vector<std::vector<double>> inputs;
    inputs.reserve(static_cast<std::size_t>(steps));
    for (const auto& series : ecg.x_train) {
        const auto feat = enc.encode_series(series);
        inputs.push_back(cypha::cyphalm::fit_input_dim(
            std::vector<double>(feat.begin(), feat.end()), ssm.d_input()));
        if (static_cast<int>(inputs.size()) >= steps) break;
    }
    while (static_cast<int>(inputs.size()) < steps && !ecg.x_train.empty()) {
        const auto& series = ecg.x_train[inputs.size() % ecg.x_train.size()];
        const auto feat = enc.encode_series(series);
        inputs.push_back(cypha::cyphalm::fit_input_dim(
            std::vector<double>(feat.begin(), feat.end()), ssm.d_input()));
    }

    auto report = cypha::cyphalm::diagnose_cellai_sequence(ssm, inputs, std::max(1, steps / 16), "d10");
    report["data_source"] = ecg.source;
    report["encoder"] = Json{{"window", 32}, {"n_fft", 16}, {"feature_dim", enc.feature_dim()}};
    return report;
}

Json run_cyphalm_domain(const std::string& domain_id, const std::string& profile) {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_bench_profile(profile, cfg);
    cypha::cyphalm::apply_bench_mode(cypha::cyphalm::BenchMode::Hybrid, cfg);
    if (profile == "d17" && cfg.vocab_size < 256) cfg.vocab_size = 256;
    if (profile == "d04" && cfg.vocab_size < 128) cfg.vocab_size = 128;

    const bool full_corpus = (profile == "d17") &&
                             (cypha::cyphalm::bench_full_corpus_enabled() || cypha::bench::bench_overnight_enabled());
    const int full_n_train = cypha::bench::bench_full_n_train();
    const int default_n_train = (profile == "d04") ? 8000 : (full_corpus ? full_n_train : 4000);
    const int default_n_eval = (profile == "d04") ? 1000 : (full_corpus ? 2000 : 500);
    const int n_train = cypha::bench::bench_scale(default_n_train, full_corpus ? 512 : 800);
    const int n_eval = cypha::bench::bench_scale(default_n_eval, full_corpus ? 64 : 100);

    cypha::cyphalm::LMCorpus corpus;
    bool synthetic = false;
    try {
        const int max_chars = full_corpus ? 0 : 2'000'000;
        corpus = cypha::cyphalm::load_bench_corpus(profile, max_chars, cfg.vocab_size,
                                                   cfg.bpe_merges_path, cfg.bpe_vocab_path);
    } catch (const std::exception&) {
        if (!bench_fast_mode()) {
            throw;
        }
        synthetic = true;
        corpus.profile = profile;
        corpus.source = "synthetic";
        corpus.vocab_size = cfg.vocab_size;
        corpus.train_ids = cypha::cyphalm::synthetic_corpus(n_train + n_eval + 64, cfg.vocab_size, cfg.seed);
        corpus.eval_ids.assign(corpus.train_ids.begin() + n_train, corpus.train_ids.end());
        corpus.train_ids.resize(static_cast<std::size_t>(n_train));
    }

    cfg.vocab_size = corpus.vocab_size;

    cypha::cyphalm::CyphaLMModel model(cfg);
    model.train_sequence(corpus.train_ids, n_train, cfg.train_epochs);
    const double bpc = model.eval_bpc(corpus.eval_ids, n_eval);
    const auto alpha_profile = model.compression_profile();

    const Json experiments{
        {"profile", profile},
        {"mode", "hybrid"},
        {"corpus", corpus.source},
        {"synthetic", synthetic},
        {"full_corpus", full_corpus},
        {"n_train", n_train},
        {"n_eval", n_eval},
        {"bpc", std::isnan(bpc) ? Json(nullptr) : Json(bpc)},
        {"vocab_size", cfg.vocab_size},
        {"17B_alpha_spectrum",
         Json{{"mean_alpha", alpha_profile.value("mean_alpha", 0.0)},
              {"fraction_edge_of_chaos", alpha_profile.value("fraction_near_edge_of_chaos", 0.0)},
              {"n_experts", alpha_profile.value("n_experts", 0)}}},
        {"backend", "cypha_lm_native"},
    };
    Json payload = experiments;
    if (g_ssm_diagnose) {
        const int diag_steps = cypha::bench::bench_scale(512, 128);
        payload["ssm_diagnose"] = model.ssm_diagnostic_report(corpus.eval_ids, diag_steps);
    }
    cypha::bench::finalize_domain(domain_id, payload);
    return payload;
}

Json run_d04() { return run_cyphalm_domain("d04", "d04"); }

Json run_d17() { return run_cyphalm_domain("d17", "d17"); }

Json run_d21_rpsm_overnight_smoke() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_bench_profile("d21", cfg);
    cypha::cyphalm::apply_bench_mode(cypha::cyphalm::BenchMode::Rpsm, cfg);
    if (cfg.vocab_size < 256) cfg.vocab_size = 256;
    cfg.view_schedule = "same_order";

    const bool full_corpus = cypha::cyphalm::bench_full_corpus_enabled() ||
                             cypha::bench::bench_overnight_enabled();
    const int full_n_train = cypha::bench::bench_full_n_train();
    const int default_n_train = full_corpus ? full_n_train : 500;
    const int default_n_eval = full_corpus ? 2000 : 500;
    const int n_train = cypha::bench::bench_scale(default_n_train, 500);
    const int n_eval = cypha::bench::bench_scale(default_n_eval, 64);

    cypha::cyphalm::LMCorpus corpus;
    bool synthetic = false;
    try {
        const int max_chars = full_corpus ? 0 : 2'000'000;
        corpus = cypha::cyphalm::load_bench_corpus("d21", max_chars, cfg.vocab_size,
                                                   cfg.bpe_merges_path, cfg.bpe_vocab_path);
    } catch (const std::exception&) {
        if (!bench_fast_mode()) {
            throw;
        }
        synthetic = true;
        corpus.profile = "d21";
        corpus.source = "synthetic";
        corpus.vocab_size = cfg.vocab_size;
        corpus.train_ids = cypha::cyphalm::synthetic_corpus(n_train + n_eval + 64, cfg.vocab_size, cfg.seed);
        corpus.eval_ids.assign(corpus.train_ids.begin() + n_train, corpus.train_ids.end());
        corpus.train_ids.resize(static_cast<std::size_t>(n_train));
    }

    cfg.vocab_size = corpus.vocab_size;

    cypha::cyphalm::CyphaLMModel model(cfg);
    model.train_sequence(corpus.train_ids, n_train, cfg.train_epochs);
    const double bpc = model.eval_bpc(corpus.eval_ids, n_eval);
    const auto alpha_profile = model.compression_profile();

    const Json experiments{
        {"profile", "d21"},
        {"mode", "rpsm"},
        {"corpus", corpus.source},
        {"synthetic", synthetic},
        {"full_corpus", full_corpus},
        {"n_train", n_train},
        {"n_eval", n_eval},
        {"bpc", std::isnan(bpc) ? Json(nullptr) : Json(bpc)},
        {"vocab_size", cfg.vocab_size},
        {"rpsm_n_levels", cfg.rpsm_n_levels},
        {"rpsm_state_dim", cfg.rpsm_state_dim},
        {"rpsm_feat_dim", cfg.rpsm_feat_dim},
        {"17B_alpha_spectrum",
         Json{{"mean_alpha", alpha_profile.value("mean_alpha", 0.0)},
              {"fraction_edge_of_chaos", alpha_profile.value("fraction_near_edge_of_chaos", 0.0)},
              {"n_experts", alpha_profile.value("n_experts", 0)}}},
        {"backend", "cypha_lm_native"},
    };
    cypha::bench::finalize_domain("d21_rpsm_overnight", experiments);
    return experiments;
}

Json run_d13() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);

    const auto files = iter_canterbury_files();
    std::vector<double> ratios;
    std::vector<double> alphas;
    Json names = Json::array();
    for (const auto& kv : files) {
        names.push_back(kv.first);
        ratios.push_back(gzip_compression_ratio_file(kv.second));
        const std::string text = read_text_lossy(kv.second);
        const Json stats = train_stream_on_text(text, cypha::bench::bench_scale(1500, 400), regime);
        alphas.push_back(stats.value("mean_alpha_proxy", 0.5));
    }

    Json exp13a{
        {"spearman_alpha_vs_gzip", cypha::bench::safe_spearman(alphas, ratios)},
        {"files", names},
        {"gzip_ratios", ratios},
        {"mean_alpha_proxy", alphas},
    };

    std::vector<double> text_alphas;
    std::vector<double> bin_alphas;
    for (const auto& kv : files) {
        const std::string text = read_text_lossy(kv.second);
        const Json stats = train_stream_on_text(text, cypha::bench::bench_scale(800, 200), regime);
        const double alpha = stats.value("mean_alpha_proxy", 0.5);
        const std::string& name = kv.first;
        const bool is_binary = (name.find(".xls") != std::string::npos || name == "ptt5" || name == "sum" ||
                                name.find(".bin") != std::string::npos || name.find("binary") != std::string::npos);
        if (is_binary) bin_alphas.push_back(alpha);
        else text_alphas.push_back(alpha);
    }
    auto mean_vec = [](const std::vector<double>& v) -> Json {
        if (v.empty()) return nullptr;
        double s = 0.0;
        for (double x : v) s += x;
        return s / static_cast<double>(v.size());
    };

    const Json experiments{
        {"13A_alpha_vs_compression", exp13a},
        {"13B_binary_vs_text_alpha",
         Json{{"mean_alpha_text", mean_vec(text_alphas)}, {"mean_alpha_binary", mean_vec(bin_alphas)}}},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d13", experiments);
    return experiments;
}

using FeynmanFn = std::function<double(const std::vector<double>&)>;

struct FeynmanSpec {
    std::string name;
    int n_inputs;
    FeynmanFn fn;
};

std::vector<FeynmanSpec> feynman_equations() {
    return {
        {"newton_second_law", 2, [](const std::vector<double>& a) { return a[0] / a[1]; }},
        {"kinetic_energy", 2, [](const std::vector<double>& a) { return 0.5 * a[0] * a[1] * a[1]; }},
        {"gravitational_pe", 3, [](const std::vector<double>& a) { return a[0] * a[1] * a[2]; }},
        {"ohms_law", 2, [](const std::vector<double>& a) { return a[0] / a[1]; }},
        {"ideal_gas", 4, [](const std::vector<double>& a) { return (a[0] * a[1] * a[2]) / a[3]; }},
        {"coulombs_law", 3, [](const std::vector<double>& a) { return (8.99e9 * a[0] * a[1]) / (a[2] * a[2]); }},
        {"wave_speed", 2, [](const std::vector<double>& a) { return a[0] * a[1]; }},
        {"relativistic_KE", 3, [](const std::vector<double>& a) {
             const double ratio = a[1] / a[2];
             return a[0] * a[2] * a[2] * (1.0 / std::sqrt(std::max(1.0 - ratio * ratio, 1e-9)) - 1.0);
         }},
        {"lens_equation", 2, [](const std::vector<double>& a) { return 1.0 / (1.0 / a[0] + 1.0 / a[1]); }},
        {"bernoulli", 3, [](const std::vector<double>& a) { return a[2] + 0.5 * a[0] * a[1] * a[1]; }},
        {"hooke", 2, [](const std::vector<double>& a) { return 0.5 * a[0] * a[1] * a[1]; }},
        {"snell", 3, [](const std::vector<double>& a) { return a[0] * std::sin(a[1]) / a[2]; }},
        {"Stefan_Boltzmann", 2, [](const std::vector<double>& a) { return a[0] * std::pow(a[1], 4); }},
        {"thermal_expansion", 3, [](const std::vector<double>& a) { return a[0] * a[1] * a[2]; }},
        {"capacitor_energy", 2, [](const std::vector<double>& a) { return 0.5 * a[0] * a[1] * a[1]; }},
        {"log_decay", 3, [](const std::vector<double>& a) { return a[0] * std::exp(-a[1] * a[2]); }},
        {"centripetal", 3, [](const std::vector<double>& a) { return a[0] * a[1] * a[1] / a[2]; }},
        {"diffraction", 2, [](const std::vector<double>& a) {
             return std::asin(std::clamp(a[0] / a[1], -1.0, 1.0));
         }},
        {"entropy_ideal_gas", 3, [](const std::vector<double>& a) {
             return a[0] * a[1] * std::log(std::max(a[2], 1e-6));
         }},
        {"drag_force", 4, [](const std::vector<double>& a) { return 0.5 * a[0] * a[1] * a[2] * a[3] * a[3]; }},
    };
}

void generate_feynman_dataset(const FeynmanFn& fn, int n_inputs, int n_samples, double noise_std, std::uint64_t seed,
                              std::vector<std::vector<double>>& xs, std::vector<double>& ys) {
    std::mt19937 rng = make_rng(seed);
    std::uniform_real_distribution<double> uniform(0.1, 5.0);
    std::normal_distribution<double> gauss(0.0, 1.0);
    xs.clear();
    ys.clear();
    xs.reserve(static_cast<std::size_t>(n_samples));
    ys.reserve(static_cast<std::size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> row(static_cast<std::size_t>(n_inputs));
        for (int j = 0; j < n_inputs; ++j) row[static_cast<std::size_t>(j)] = uniform(rng);
        double y = fn(row);
        if (!std::isfinite(y)) continue;
        xs.push_back(std::move(row));
        ys.push_back(y);
    }
    if (ys.empty()) return;
    double mean_abs = 0.0;
    for (double y : ys) mean_abs += std::abs(y);
    mean_abs /= static_cast<double>(ys.size());
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ys[i] += gauss(rng) * noise_std * (mean_abs + 1e-8);
        if (!std::isfinite(ys[i])) {
            xs.erase(xs.begin() + static_cast<std::ptrdiff_t>(i));
            ys.erase(ys.begin() + static_cast<std::ptrdiff_t>(i));
            --i;
        }
    }
}

Json run_d14() {
    const cypha::bench::ProfileJson regime = cypha::bench::regression_params();
    const int n_train = cypha::bench::bench_scale(1600, 400);
    const int n_test = cypha::bench::bench_scale(400, 100);

    Json per_equation = Json::object();
    double mean_rmse = 0.0;
    double mean_r2 = 0.0;
    int eq_count = 0;
    for (const auto& eq : feynman_equations()) {
        std::vector<std::vector<double>> xs;
        std::vector<double> ys;
        generate_feynman_dataset(eq.fn, eq.n_inputs, n_train + n_test, 0.01, kBenchSeed, xs, ys);
        if (static_cast<int>(xs.size()) < n_train + n_test) continue;
        std::vector<std::vector<double>> train_x(xs.begin(), xs.begin() + n_train);
        std::vector<std::vector<double>> test_x(xs.begin() + n_train, xs.begin() + n_train + n_test);
        std::vector<double> train_y(ys.begin(), ys.begin() + n_train);
        std::vector<double> test_y(ys.begin() + n_train, ys.begin() + n_train + n_test);

        OnlineRegressor reg = make_online_regressor(eq.n_inputs, kBenchSeed, regime);
        for (std::size_t i = 0; i < train_x.size(); ++i) {
            online_reg_train_step(reg, train_x[i], train_y[i]);
        }
        cypha::sync_infer_model_from_memory(reg.infer, reg.mem);
        Json m = reg_metrics_native(reg, test_x, test_y);
        m["ridge_rmse"] = cypha::bench::ridge_baseline(train_x, train_y, test_x, test_y).rmse;
        per_equation[eq.name] = m;
        mean_rmse += m["rmse"].get<double>();
        mean_r2 += m["r2"].get<double>();
        ++eq_count;
    }
    if (eq_count > 0) {
        mean_rmse /= static_cast<double>(eq_count);
        mean_r2 /= static_cast<double>(eq_count);
    }

    // 14B extrapolation
    const auto eqs = feynman_equations();
    const FeynmanSpec& ke = eqs[1];  // kinetic_energy
    std::mt19937 rng_in = make_rng(kBenchSeed + 1);
    std::uniform_real_distribution<double> in_dist(0.1, 5.0);
    std::uniform_real_distribution<double> out_dist(5.1, 10.0);
    std::vector<std::vector<double>> x_in;
    std::vector<double> y_in;
    std::vector<std::vector<double>> x_out;
    std::vector<double> y_out;
    for (int i = 0; i < cypha::bench::bench_scale(1600, 400); ++i) {
        std::vector<double> row(2);
        row[0] = in_dist(rng_in);
        row[1] = in_dist(rng_in);
        x_in.push_back(row);
        y_in.push_back(ke.fn(row));
    }
    for (int i = 0; i < cypha::bench::bench_scale(400, 100); ++i) {
        std::vector<double> row(2);
        row[0] = out_dist(rng_in);
        row[1] = out_dist(rng_in);
        x_out.push_back(row);
        y_out.push_back(ke.fn(row));
    }
    OnlineRegressor reg_b = make_online_regressor(2, kBenchSeed + 1, regime);
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<int> order(static_cast<int>(x_in.size()));
        for (int i = 0; i < static_cast<int>(order.size()); ++i) order[static_cast<std::size_t>(i)] = i;
        std::shuffle(order.begin(), order.end(), make_rng(kBenchSeed + static_cast<std::uint64_t>(pass)));
        for (int idx : order) online_reg_train_step(reg_b, x_in[static_cast<std::size_t>(idx)], y_in[static_cast<std::size_t>(idx)]);
    }
    cypha::sync_infer_model_from_memory(reg_b.infer, reg_b.mem);

    std::vector<double> train_mu(2, 0.0);
    std::vector<double> train_std(2, 1.0);
    for (int j = 0; j < 2; ++j) {
        for (const auto& row : x_in) train_mu[static_cast<std::size_t>(j)] += row[static_cast<std::size_t>(j)];
        train_mu[static_cast<std::size_t>(j)] /= static_cast<double>(x_in.size());
    }
    for (int j = 0; j < 2; ++j) {
        for (const auto& row : x_in) {
            const double d = row[static_cast<std::size_t>(j)] - train_mu[static_cast<std::size_t>(j)];
            train_std[static_cast<std::size_t>(j)] += d * d;
        }
        train_std[static_cast<std::size_t>(j)] =
            std::sqrt(train_std[static_cast<std::size_t>(j)] / static_cast<double>(x_in.size())) + 1e-6;
    }
    auto ood_score = [&](const std::vector<double>& x) {
        double s = 0.0;
        for (int j = 0; j < 2; ++j) {
            const double z = (x[static_cast<std::size_t>(j)] - train_mu[static_cast<std::size_t>(j)]) /
                             train_std[static_cast<std::size_t>(j)];
            s += z * z;
        }
        return std::sqrt(s);
    };
    std::vector<int> labels;
    std::vector<double> scores;
    const int n_in_eval = std::min(200, static_cast<int>(x_in.size()));
    for (int i = 0; i < n_in_eval; ++i) {
        labels.push_back(0);
        scores.push_back(ood_score(x_in[static_cast<std::size_t>(i)]));
    }
    for (const auto& row : x_out) {
        labels.push_back(1);
        scores.push_back(ood_score(row));
    }
    std::vector<double> reg_u_in;
    std::vector<double> reg_u_out;
    for (int i = 0; i < n_in_eval; ++i) {
        double y_hat = 0.0;
        double u = 0.0;
        online_reg_predict(reg_b, x_in[static_cast<std::size_t>(i)], y_hat, u);
        reg_u_in.push_back(u);
    }
    for (const auto& row : x_out) {
        double y_hat = 0.0;
        double u = 0.0;
        online_reg_predict(reg_b, row, y_hat, u);
        reg_u_out.push_back(u);
    }
    std::vector<int> reg_labels = labels;
    std::vector<double> reg_scores;
    reg_scores.insert(reg_scores.end(), reg_u_in.begin(), reg_u_in.end());
    reg_scores.insert(reg_scores.end(), reg_u_out.begin(), reg_u_out.end());

    // 14C noise curve
    const FeynmanSpec& ohms = eqs[3];
    const double noise_levels[] = {0.0, 0.05, 0.1, 0.2, 0.5};
    Json noise_curve = Json::object();
    for (double nl : noise_levels) {
        std::vector<std::vector<double>> xs;
        std::vector<double> ys;
        generate_feynman_dataset(ohms.fn, ohms.n_inputs, cypha::bench::bench_scale(2000, 500), nl, kBenchSeed, xs, ys);
        const int split = static_cast<int>(static_cast<double>(xs.size()) * 0.8);
        if (split <= 0 || split >= static_cast<int>(xs.size())) continue;
        std::vector<std::vector<double>> tr(xs.begin(), xs.begin() + split);
        std::vector<std::vector<double>> te(xs.begin() + split, xs.end());
        std::vector<double> ytr(ys.begin(), ys.begin() + split);
        std::vector<double> yte(ys.begin() + split, ys.end());
        OnlineRegressor reg_c = make_online_regressor(ohms.n_inputs, kBenchSeed, regime);
        for (std::size_t i = 0; i < tr.size(); ++i) online_reg_train_step(reg_c, tr[i], ytr[i]);
        cypha::sync_infer_model_from_memory(reg_c.infer, reg_c.mem);
        Json m = reg_metrics_native(reg_c, te, yte);
        noise_curve[std::to_string(nl)] =
            Json{{"rmse", m["rmse"]}, {"mean_epistemic_var", m["mean_epistemic_var"]}};
    }

    const Json experiments{
        {"14A_feynman_all_equations",
         Json{{"per_equation", per_equation}, {"mean_rmse", mean_rmse}, {"mean_r2", mean_r2}}},
        {"14B_extrapolation_uncertainty",
         Json{{"extrapolation_auroc", cypha::bench::safe_auroc(labels, scores)},
              {"regressor_uncertainty_auroc", cypha::bench::safe_auroc(reg_labels, reg_scores)}}},
        {"14C_noise_vs_aleatoric", noise_curve},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d14", experiments);
    return experiments;
}

Json run_d15() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const DigitDataset digits = load_digits_hog_dataset();

    auto experiment_15a = [&]() {
        OnlineClassifier clf = make_online_classifier(static_cast<int>(digits.train_x.front().size()), kBenchSeed,
                                                      regime.value("enc_lr", 0.002), regime);
        train_classifier_online(clf, digits.train_x, digits.train_y, 4, kBenchSeed);
        std::mt19937 rng = make_rng(kBenchSeed);
        std::normal_distribution<double> gauss(0.0, 1.0);
        const double levels[] = {0.0, 0.1, 0.2, 0.5, 1.0};
        Json curve = Json::object();
        for (double std : levels) {
            std::vector<std::vector<double>> noisy = digits.test_x;
            for (auto& row : noisy) {
                for (double& v : row) v += gauss(rng) * std;
            }
            curve[std::to_string(std)] = clf_metrics_native(clf.infer, noisy, digits.test_y);
        }
        return curve;
    };

    auto experiment_15b = [&]() {
        OnlineClassifier clf = make_online_classifier(static_cast<int>(digits.train_x.front().size()), kBenchSeed + 1,
                                                      regime.value("enc_lr", 0.002), regime);
        train_classifier_online(clf, digits.train_x, digits.train_y, 4, kBenchSeed + 1);
        std::mt19937 rng = make_rng(kBenchSeed + 1);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        const double rates[] = {0.1, 0.25, 0.5, 0.75};
        Json curve = Json::object();
        for (double rate : rates) {
            std::vector<std::vector<double>> dropped = digits.test_x;
            for (auto& row : dropped) {
                for (double& v : row) {
                    if (uni(rng) < rate) v = 0.0;
                }
            }
            curve[std::to_string(rate)] = clf_metrics_native(clf.infer, dropped, digits.test_y);
        }
        return curve;
    };

    auto experiment_15c = [&]() {
        OnlineClassifier clf = make_online_classifier(static_cast<int>(digits.train_x.front().size()), kBenchSeed + 2,
                                                      regime.value("enc_lr", 0.002), regime);
        train_classifier_online(clf, digits.train_x, digits.train_y, 4, kBenchSeed + 2);
        const int n = std::min(500, static_cast<int>(digits.test_x.size()));
        int acc_nat = 0;
        int acc_adv = 0;
        double epi_nat = 0.0;
        double epi_adv = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto& x = digits.test_x[static_cast<std::size_t>(i)];
            const std::string& y = digits.test_y[static_cast<std::size_t>(i)];
            const std::string pred = online_clf_predict(clf.infer, x);
            if (pred == y) ++acc_nat;
            epi_nat += online_clf_epistemic(clf.infer, x);

            std::vector<double> x_adv = x;
            for (std::size_t j = 0; j < x_adv.size(); ++j) {
                std::vector<double> x_plus = x;
                std::vector<double> x_minus = x;
                x_plus[j] += 1e-4;
                x_minus[j] -= 1e-4;
                const std::string p_plus = online_clf_predict(clf.infer, x_plus);
                const std::string p_minus = online_clf_predict(clf.infer, x_minus);
                if (p_plus != y) x_adv[j] += 0.1;
                else if (p_minus != y) x_adv[j] -= 0.1;
                if (x_adv[j] < 0.0) x_adv[j] = 0.0;
            }
            const std::string pred_a = online_clf_predict(clf.infer, x_adv);
            if (pred_a == y) ++acc_adv;
            epi_adv += online_clf_epistemic(clf.infer, x_adv);
        }
        return Json{
            {"accuracy_natural", static_cast<double>(acc_nat) / static_cast<double>(n)},
            {"accuracy_adversarial", static_cast<double>(acc_adv) / static_cast<double>(n)},
            {"mean_epistemic_natural", epi_nat / static_cast<double>(n)},
            {"mean_epistemic_adversarial", epi_adv / static_cast<double>(n)},
        };
    };

    const Json experiments{
        {"15A_gaussian_noise", experiment_15a()},
        {"15B_feature_dropout", experiment_15b()},
        {"15C_adversarial_fgsm_proxy", experiment_15c()},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d15", experiments);
    return experiments;
}

Json run_d16_ewc_probe() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto tasks = load_multitask_datasets();
    int max_dim = 0;
    for (const auto& t : tasks) {
        max_dim = std::max(max_dim, static_cast<int>(t.train_x.empty() ? 0 : t.train_x.front().size()));
    }

    auto make_multitask_clf = [&](std::uint64_t seed) {
        return make_online_classifier(max_dim, seed, 0.0, regime);
    };

    auto eval_task = [&](const OnlineClassifier& c, const MultitaskBundle& task) {
        std::vector<std::vector<double>> xp;
        std::vector<std::string> labels;
        xp.reserve(task.test_x.size());
        labels.reserve(task.test_y.size());
        for (std::size_t i = 0; i < task.test_x.size(); ++i) {
            xp.push_back(pad_to_max(task.test_x[i], max_dim));
            labels.push_back(task.name + "_" + task.test_y[i]);
        }
        return clf_metrics_native(c.infer, xp, labels)["accuracy"].get<double>();
    };

    const int steps = cypha::bench::bench_scale(3000, 800);
    auto train_task = [&](OnlineClassifier& clf, const MultitaskBundle& task, cypha::TrainStepExtras* extras) {
        std::vector<int> order(static_cast<int>(task.train_x.size()));
        for (int i = 0; i < static_cast<int>(order.size()); ++i) {
            order[static_cast<std::size_t>(i)] = i;
        }
        std::shuffle(order.begin(), order.end(), make_rng(kBenchSeed + 2));
        const int limit = std::min(steps, static_cast<int>(order.size()));
        for (int k = 0; k < limit; ++k) {
            const int i = order[static_cast<std::size_t>(k)];
            const auto x = pad_to_max(task.train_x[static_cast<std::size_t>(i)], max_dim);
            const std::string label = task.name + "_" + task.train_y[static_cast<std::size_t>(i)];
            const int d = static_cast<int>(x.size());
            (void)cypha::dif_train_step_vector(clf.infer, clf.mem, clf.replay, x.data(), d, label, clf.world_lr,
                                               clf.delta_lr, clf.world_lr, clf.delta_lr, clf.ood_sigma, clf.tsp,
                                               clf.rng, clf.enc_updates, nullptr, extras);
        }
    };
    const MultitaskBundle* iris = nullptr;
    const MultitaskBundle* wine = nullptr;
    const MultitaskBundle* digits = nullptr;
    for (const auto& t : tasks) {
        if (t.name == "iris") iris = &t;
        if (t.name == "wine") wine = &t;
        if (t.name == "digits") digits = &t;
    }
    auto forgetting_probe = [&](bool use_ewc, std::uint64_t seed) {
        OnlineClassifier clf = make_multitask_clf(seed);
        cypha::EwcRegularizer ewc;
        cypha::TrainStepExtras extras{};
        if (use_ewc) {
            extras.ewc = &ewc;
            extras.ewc_lambda = 0.5;
        }
        train_task(clf, *iris, use_ewc ? &extras : nullptr);
        if (use_ewc) {
            ewc.snapshot(clf.mem, clf.infer);
        }
        const double acc_before = eval_task(clf, *iris);
        train_task(clf, *wine, use_ewc ? &extras : nullptr);
        train_task(clf, *digits, use_ewc ? &extras : nullptr);
        cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
        const double acc_after = eval_task(clf, *iris);
        const double forgetting = (acc_before - acc_after) / std::max(acc_before, 1e-6);
        Json row{
            {"task_a_accuracy_before", acc_before},
            {"task_a_accuracy_after", acc_after},
            {"forgetting_score", forgetting},
        };
        if (use_ewc) {
            row["ewc_lambda"] = extras.ewc_lambda;
            row["ewc_penalty_final"] = ewc.penalty(clf.mem, clf.infer);
        }
        return row;
    };
    const Json baseline = forgetting_probe(false, kBenchSeed + 1);
    const Json ewc_row = forgetting_probe(true, kBenchSeed + 51);
    const double baseline_forgetting = baseline["forgetting_score"].get<double>();
    const double ewc_forgetting = ewc_row["forgetting_score"].get<double>();
    return Json{
        {"baseline", baseline},
        {"ewc", ewc_row},
        {"ewc_reduces_forgetting", ewc_forgetting <= baseline_forgetting},
        {"forgetting_delta", baseline_forgetting - ewc_forgetting},
    };
}

Json run_d16() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto tasks = load_multitask_datasets();
    int max_dim = 0;
    for (const auto& t : tasks) {
        max_dim = std::max(max_dim, static_cast<int>(t.train_x.empty() ? 0 : t.train_x.front().size()));
    }

    auto make_multitask_clf = [&](std::uint64_t seed) {
        return make_online_classifier(max_dim, seed, 0.0, regime);
    };

    auto eval_task = [&](const OnlineClassifier& c, const MultitaskBundle& task) {
        std::vector<std::vector<double>> xp;
        std::vector<std::string> labels;
        xp.reserve(task.test_x.size());
        labels.reserve(task.test_y.size());
        for (std::size_t i = 0; i < task.test_x.size(); ++i) {
            xp.push_back(pad_to_max(task.test_x[i], max_dim));
            labels.push_back(task.name + "_" + task.test_y[i]);
        }
        return clf_metrics_native(c.infer, xp, labels)["accuracy"].get<double>();
    };

    // 16A task discovery
    Json exp16a;
    {
        OnlineClassifier clf = make_multitask_clf(kBenchSeed);
        const int max_steps = cypha::bench::bench_scale(12000, 3000);
        std::unordered_map<std::string, std::size_t> cursors;
        std::vector<std::string> task_order;
        for (const auto& t : tasks) task_order.push_back(t.name);
        int step = 0;
        while (step < max_steps) {
            for (const auto& t : tasks) {
                if (step >= max_steps) break;
                const std::size_t idx = cursors[t.name]++;
                if (idx >= t.train_x.size()) cursors[t.name] = 0;
                const std::size_t i = idx % t.train_x.size();
                const auto x = pad_to_max(t.train_x[i], max_dim);
                const std::string label = t.name + "_" + t.train_y[i];
                (void)online_clf_train_step(clf, x, label);
                ++step;
            }
        }
        cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
        Json per_task = Json::object();
        for (const auto& t : tasks) per_task[t.name] = eval_task(clf, t);
        exp16a = Json{
            {"routing_ari", 0.0},
            {"per_task_accuracy", per_task},
            {"expert_count", static_cast<int>(clf.infer.labels.size())},
        };
    }

    // 16B forgetting (baseline vs EWC zero-forgetting probe)
    const Json exp16b = run_d16_ewc_probe();

    // 16D interleaving
    Json exp16d = Json::object();
    {
        const char* strategies[] = {"round_robin", "random", "block"};
        for (const char* strategy : strategies) {
            OnlineClassifier clf = make_multitask_clf(kBenchSeed + 3);
            const int max_steps = cypha::bench::bench_scale(9000, 2000);
            std::unordered_map<std::string, std::size_t> cursors;
            std::mt19937 rng = make_rng(kBenchSeed + 3);
            int step = 0;
            while (step < max_steps) {
                std::vector<std::string> order;
                if (std::string(strategy) == "random") {
                    order.push_back(tasks[static_cast<std::size_t>(rng() % tasks.size())].name);
                } else {
                    for (const auto& t : tasks) order.push_back(t.name);
                }
                for (const std::string& tid : order) {
                    if (step >= max_steps) break;
                    const MultitaskBundle* task = nullptr;
                    for (const auto& t : tasks) {
                        if (t.name == tid) task = &t;
                    }
                    if (!task) continue;
                    if (std::string(strategy) == "block") {
                        for (int b = 0; b < std::min(1000, max_steps - step); ++b) {
                            const std::size_t i = cursors[tid]++ % task->train_x.size();
                            const auto x = pad_to_max(task->train_x[i], max_dim);
                            (void)online_clf_train_step(clf, x, tid + "_" + task->train_y[i]);
                            ++step;
                        }
                    } else {
                        const std::size_t i = cursors[tid]++ % task->train_x.size();
                        const auto x = pad_to_max(task->train_x[i], max_dim);
                        (void)online_clf_train_step(clf, x, tid + "_" + task->train_y[i]);
                        ++step;
                    }
                }
            }
            cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
            Json accs = Json::object();
            for (const auto& t : tasks) accs[t.name] = eval_task(clf, t);
            exp16d[strategy] = accs;
        }
    }

    // 16E save/restore via snapshot copy
    Json exp16e;
    {
        OnlineClassifier clf = make_multitask_clf(kBenchSeed + 4);
        const int steps = cypha::bench::bench_scale(3000, 800);
        auto train_named = [&](const MultitaskBundle& task) {
            std::vector<int> order(static_cast<int>(task.train_x.size()));
            for (int i = 0; i < static_cast<int>(order.size()); ++i) order[static_cast<std::size_t>(i)] = i;
            std::shuffle(order.begin(), order.end(), make_rng(kBenchSeed + 5));
            const int limit = std::min(steps, static_cast<int>(order.size()));
            for (int k = 0; k < limit; ++k) {
                const int i = order[static_cast<std::size_t>(k)];
                const auto x = pad_to_max(task.train_x[static_cast<std::size_t>(i)], max_dim);
                (void)online_clf_train_step(clf, x, task.name + "_" + task.train_y[static_cast<std::size_t>(i)]);
            }
        };
        const MultitaskBundle* iris = nullptr;
        const MultitaskBundle* wine = nullptr;
        const MultitaskBundle* digits = nullptr;
        for (const auto& t : tasks) {
            if (t.name == "iris") iris = &t;
            if (t.name == "wine") wine = &t;
            if (t.name == "digits") digits = &t;
        }
        train_named(*iris);
        const double acc_before = eval_task(clf, *iris);
        const OnlineClassifier snapshot = clf;
        train_named(*wine);
        train_named(*digits);
        const double acc_corrupted = eval_task(clf, *iris);
        clf = snapshot;
        const double acc_restored = eval_task(clf, *iris);
        exp16e = Json{
            {"task_a_before", acc_before},
            {"task_a_corrupted", acc_corrupted},
            {"task_a_restored", acc_restored},
            {"retention_ratio", acc_restored / std::max(acc_before, 1e-6)},
        };
    }

    // 16F per-task models
    Json exp16f;
    {
        Json per_task = Json::object();
        for (const auto& t : tasks) {
            OnlineClassifier clf = make_multitask_clf(kBenchSeed + 10);
            for (int pass = 0; pass < 3; ++pass) {
                std::vector<int> order(static_cast<int>(t.train_x.size()));
                for (int i = 0; i < static_cast<int>(order.size()); ++i) order[static_cast<std::size_t>(i)] = i;
                std::shuffle(order.begin(), order.end(), make_rng(kBenchSeed + 11 + static_cast<std::uint64_t>(pass)));
                for (int idx : order) {
                    const auto x = pad_to_max(t.train_x[static_cast<std::size_t>(idx)], max_dim);
                    (void)online_clf_train_step(clf, x, t.name + "_" + t.train_y[static_cast<std::size_t>(idx)]);
                }
            }
            cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
            per_task[t.name] = eval_task(clf, t);
        }
        exp16f = Json{
            {"per_task_accuracy", per_task},
            {"forgetting_score", 0.0},
            {"note", "per-task isolated models — zero forgetting by architecture"},
        };
    }

    // 16G view streams (simplified round_robin vs block shuffle forgetting)
    Json exp16g;
    {
        const int max_steps = cypha::bench::bench_scale(3000, 1500);
        const int warm_steps = std::max(1, max_steps / 6);
        auto forgetting_with_stream = [&](bool block_shuffle, std::uint64_t seed) {
            OnlineClassifier clf = make_multitask_clf(seed);
            const MultitaskBundle* iris = nullptr;
            for (const auto& t : tasks) {
                if (t.name == "iris") iris = &t;
            }
            std::vector<int> warm_order(static_cast<int>(iris->train_x.size()));
            for (int i = 0; i < static_cast<int>(warm_order.size()); ++i) warm_order[static_cast<std::size_t>(i)] = i;
            std::shuffle(warm_order.begin(), warm_order.end(), make_rng(seed + 1));
            for (int k = 0; k < std::min(warm_steps, static_cast<int>(warm_order.size())); ++k) {
                const int i = warm_order[static_cast<std::size_t>(k)];
                const auto x = pad_to_max(iris->train_x[static_cast<std::size_t>(i)], max_dim);
                (void)online_clf_train_step(clf, x, "iris_" + iris->train_y[static_cast<std::size_t>(i)]);
            }
            const double acc_before = eval_task(clf, *iris);
            int step = 0;
            std::mt19937 rng = make_rng(seed + 2);
            std::unordered_map<std::string, std::size_t> cursors;
            while (step < max_steps - warm_steps) {
                std::vector<std::string> order;
                if (block_shuffle) {
                    order = {"iris", "wine", "digits"};
                    std::shuffle(order.begin(), order.end(), rng);
                } else {
                    for (const auto& t : tasks) order.push_back(t.name);
                }
                for (const std::string& tid : order) {
                    if (step >= max_steps - warm_steps) break;
                    const MultitaskBundle* task = nullptr;
                    for (const auto& t : tasks) {
                        if (t.name == tid) task = &t;
                    }
                    const std::size_t i = cursors[tid]++ % task->train_x.size();
                    const auto x = pad_to_max(task->train_x[i], max_dim);
                    (void)online_clf_train_step(clf, x, tid + "_" + task->train_y[i]);
                    ++step;
                }
            }
            cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
            const double acc_after = eval_task(clf, *iris);
            Json per_task = Json::object();
            for (const auto& t : tasks) per_task[t.name] = eval_task(clf, t);
            return Json{
                {"task_a_accuracy_before", acc_before},
                {"task_a_accuracy_after", acc_after},
                {"forgetting_score", (acc_before - acc_after) / std::max(acc_before, 1e-6)},
                {"per_task_accuracy", per_task},
            };
        };
        const Json round_robin = forgetting_with_stream(false, kBenchSeed + 22);
        const Json task_block = forgetting_with_stream(true, kBenchSeed + 23);
        exp16g = Json{
            {"max_steps", max_steps},
            {"macro_epochs", 2},
            {"round_robin", round_robin},
            {"task_block_shuffle", task_block},
            {"forgetting_delta", task_block["forgetting_score"].get<double>() -
                                 round_robin["forgetting_score"].get<double>()},
        };
    }

    // 16H EWC overlay smoke (anchor snapshot + optional penalty during 16B-style iris train)
    Json exp16h;
    {
        const MultitaskBundle* iris = nullptr;
        for (const auto& t : tasks) {
            if (t.name == "iris") {
                iris = &t;
            }
        }
        if (iris != nullptr && !iris->train_x.empty()) {
            OnlineClassifier clf = make_multitask_clf(kBenchSeed + 40);
            cypha::EwcRegularizer ewc;
            ewc.snapshot(clf.mem, clf.infer);
            cypha::TrainStepExtras extras{};
            extras.ewc = &ewc;
            extras.ewc_lambda = 0.25;
            const int steps = std::min(cypha::bench::bench_scale(400, 100),
                                       static_cast<int>(iris->train_x.size()));
            for (int k = 0; k < steps; ++k) {
                const auto x = pad_to_max(iris->train_x[static_cast<std::size_t>(k)], max_dim);
                const std::string label = iris->name + "_" + iris->train_y[static_cast<std::size_t>(k)];
                const int d = static_cast<int>(x.size());
                (void)cypha::dif_train_step_vector(clf.infer, clf.mem, clf.replay, x.data(), d, label, clf.world_lr,
                                                   clf.delta_lr, clf.world_lr, clf.delta_lr, clf.ood_sigma, clf.tsp,
                                                   clf.rng, clf.enc_updates, nullptr, &extras);
            }
            cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
            exp16h = Json{
                {"ewc_lambda", 0.25},
                {"ewc_penalty_after_train", ewc.penalty(clf.mem, clf.infer)},
                {"iris_accuracy", eval_task(clf, *iris)},
                {"ewc_active", true},
            };
        } else {
            exp16h = Json{{"ewc_active", false}, {"detail", "iris task missing"}};
        }
    }

    const Json experiments{
        {"16A_task_discovery", exp16a},
        {"16B_forgetting_resistance", exp16b},
        {"16D_interleaving_comparison", exp16d},
        {"16E_save_restore", exp16e},
        {"16F_per_task_models", exp16f},
        {"16G_view_streams", exp16g},
        {"16H_ewc_overlay", exp16h},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d16", experiments);
    return experiments;
}

void holdout_split_indices(int n, double test_frac, std::uint64_t seed, std::vector<int>& train_idx,
                           std::vector<int>& test_idx) {
    std::vector<int> order(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
    std::shuffle(order.begin(), order.end(), make_rng(seed));
    const int n_test = std::max(1, static_cast<int>(std::round(static_cast<double>(n) * test_frac)));
    test_idx.assign(order.begin(), order.begin() + n_test);
    train_idx.assign(order.begin() + n_test, order.end());
}

struct NslKddData {
    std::string source;
    std::vector<std::vector<double>> x_train;
    std::vector<int> y_train;
    std::vector<std::vector<double>> x_test;
    std::vector<int> y_test;
};

NslKddData synthetic_nsl_kdd(std::uint64_t seed, int n_train, int n_test, int n_features) {
    std::mt19937 rng = make_rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    NslKddData ds;
    ds.source = "synthetic";
    ds.x_train.resize(static_cast<std::size_t>(n_train), std::vector<double>(static_cast<std::size_t>(n_features)));
    ds.y_train.resize(static_cast<std::size_t>(n_train));
    for (int i = 0; i < n_train; ++i) {
        for (int j = 0; j < n_features; ++j) {
            ds.x_train[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = gauss(rng);
        }
        ds.y_train[static_cast<std::size_t>(i)] = uni(rng) < 0.2 ? 1 : 0;
    }
    ds.x_test.resize(static_cast<std::size_t>(n_test), std::vector<double>(static_cast<std::size_t>(n_features)));
    ds.y_test.resize(static_cast<std::size_t>(n_test));
    std::vector<double> attack_shift(static_cast<std::size_t>(n_features));
    for (int j = 0; j < n_features; ++j) attack_shift[static_cast<std::size_t>(j)] = gauss(rng) * 0.5 + 1.5;
    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_features; ++j) {
            ds.x_test[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = gauss(rng);
        }
        if (uni(rng) < 0.3) {
            for (int j = 0; j < n_features; ++j) {
                ds.x_test[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] +=
                    attack_shift[static_cast<std::size_t>(j)];
            }
            ds.y_test[static_cast<std::size_t>(i)] = 1;
        } else {
            ds.y_test[static_cast<std::size_t>(i)] = 0;
        }
    }
    return ds;
}

NslKddData load_nsl_kdd() {
    const std::filesystem::path train_p = cypha::bench::data_dir() / "nsl_kdd" / "KDDTrain+.txt";
    const std::filesystem::path test_p = cypha::bench::data_dir() / "nsl_kdd" / "KDDTest+.txt";
    if (!std::filesystem::is_regular_file(train_p) || !std::filesystem::is_regular_file(test_p)) {
        return synthetic_nsl_kdd(kBenchSeed, 5000, 2000, 41);
    }
    std::unordered_map<std::string, int> proto_map;
    std::unordered_map<std::string, int> service_map;
    std::unordered_map<std::string, int> flag_map;
    auto load_file = [&](const std::filesystem::path& path, std::vector<std::vector<double>>& xs,
                         std::vector<int>& ys) {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::vector<std::string> cells;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) cells.push_back(cell);
            if (cells.size() < 42) continue;
            std::vector<double> row;
            row.reserve(41);
            for (int i = 0; i < 41; ++i) {
                if (i == 1 || i == 2 || i == 3) {
                    const std::string col = (i == 1) ? "protocol_type" : (i == 2 ? "service" : "flag");
                    const std::string cat_key = col + ":" + cells[static_cast<std::size_t>(i)];
                    std::unordered_map<std::string, int>* cmap = nullptr;
                    if (col == "protocol_type") cmap = &proto_map;
                    else if (col == "service") cmap = &service_map;
                    else cmap = &flag_map;
                    if (!cmap->count(cat_key)) (*cmap)[cat_key] = static_cast<int>(cmap->size());
                    row.push_back(static_cast<double>((*cmap)[cat_key]));
                } else {
                    row.push_back(std::stod(cells[static_cast<std::size_t>(i)]));
                }
            }
            ys.push_back(cells[41] == "normal" ? 0 : 1);
            xs.push_back(std::move(row));
        }
    };
    NslKddData ds;
    ds.source = "nsl_kdd";
    load_file(train_p, ds.x_train, ds.y_train);
    load_file(test_p, ds.x_test, ds.y_test);
    if (ds.x_train.empty()) return synthetic_nsl_kdd(kBenchSeed, 5000, 2000, 41);
    standardize_train_test(ds.x_train, ds.x_test);
    return ds;
}

Json run_d05() {
    const cypha::bench::ProfileJson regime = cypha::bench::regression_params();
    const auto ds = cypha::bench::load_chess_dataset(cypha::bench::bench_scale(4000, 1000), kBenchSeed);
    std::vector<int> train_idx;
    std::vector<int> test_idx;
    holdout_split_indices(static_cast<int>(ds.x.size()), 0.2, kBenchSeed, train_idx, test_idx);
    std::vector<std::vector<double>> train_x;
    std::vector<double> train_y;
    std::vector<std::vector<double>> test_x;
    std::vector<double> test_y;
    for (int i : train_idx) {
        train_x.push_back(ds.x[static_cast<std::size_t>(i)]);
        train_y.push_back(ds.y[static_cast<std::size_t>(i)]);
    }
    for (int i : test_idx) {
        test_x.push_back(ds.x[static_cast<std::size_t>(i)]);
        test_y.push_back(ds.y[static_cast<std::size_t>(i)]);
    }
    standardize_train_test(train_x, test_x);
    OnlineRegressor reg = make_online_regressor(static_cast<int>(train_x.front().size()), kBenchSeed, regime);
    for (int p = 0; p < 4; ++p) {
        std::vector<int> order(static_cast<std::size_t>(train_x.size()));
        for (int i = 0; i < static_cast<int>(order.size()); ++i) order[static_cast<std::size_t>(i)] = i;
        std::shuffle(order.begin(), order.end(), make_rng(kBenchSeed + static_cast<std::uint64_t>(p)));
        for (int idx : order) {
            online_reg_train_step(reg, train_x[static_cast<std::size_t>(idx)], train_y[static_cast<std::size_t>(idx)]);
        }
    }
    cypha::sync_infer_model_from_memory(reg.infer, reg.mem);
    Json scores = reg_metrics_native(reg, test_x, test_y);
    const Json baselines = cypha::bench::offline_regression_baselines_json(train_x, train_y, test_x, test_y);
    scores["ridge_rmse"] = baselines["ridge"]["rmse"];
    const Json experiments{
        {"domain", "d05_chess"},
        {"data_source", ds.source},
        {"n_samples", static_cast<int>(ds.x.size())},
        {"cypha_scores", scores},
        {"baselines", baselines},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d05", experiments);
    return experiments;
}

Json run_d06() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson clf_regime = cypha::bench::classification_params(&profile);
    const cypha::bench::ProfileJson reg_regime = cypha::bench::regression_params(&profile);
    const auto ds = cypha::bench::generate_go_dataset(cypha::bench::bench_scale(8000, 2000), kBenchSeed);
    std::vector<int> tr_idx;
    std::vector<int> te_idx;
    holdout_split_indices(static_cast<int>(ds.x.size()), 0.2, kBenchSeed, tr_idx, te_idx);
    std::vector<std::vector<double>> train_x;
    std::vector<std::string> train_y_cls;
    std::vector<double> train_y_reg;
    std::vector<std::vector<double>> test_x;
    std::vector<std::string> test_y_cls;
    std::vector<double> test_y_reg;
    for (int i : tr_idx) {
        train_x.push_back(ds.x[static_cast<std::size_t>(i)]);
        train_y_cls.push_back(ds.y_cls[static_cast<std::size_t>(i)]);
        train_y_reg.push_back(ds.y_reg[static_cast<std::size_t>(i)]);
    }
    for (int i : te_idx) {
        test_x.push_back(ds.x[static_cast<std::size_t>(i)]);
        test_y_cls.push_back(ds.y_cls[static_cast<std::size_t>(i)]);
        test_y_reg.push_back(ds.y_reg[static_cast<std::size_t>(i)]);
    }
    standardize_train_test(train_x, test_x);
    OnlineRegressor reg = make_online_regressor(static_cast<int>(train_x.front().size()), kBenchSeed, reg_regime);
    for (int p = 0; p < 4; ++p) {
        std::vector<int> order(static_cast<std::size_t>(train_x.size()));
        for (int i = 0; i < static_cast<int>(order.size()); ++i) order[static_cast<std::size_t>(i)] = i;
        std::shuffle(order.begin(), order.end(), make_rng(kBenchSeed + static_cast<std::uint64_t>(p)));
        for (int idx : order) {
            online_reg_train_step(reg, train_x[static_cast<std::size_t>(idx)], train_y_reg[static_cast<std::size_t>(idx)]);
        }
    }
    cypha::sync_infer_model_from_memory(reg.infer, reg.mem);
    OnlineClassifier clf = make_online_classifier(static_cast<int>(train_x.front().size()), kBenchSeed,
                                                  clf_regime.value("enc_lr", 0.002), clf_regime);
    train_classifier_online(clf, train_x, train_y_cls, std::max(1, clf_regime.value("n_epochs", 1)), kBenchSeed);
    const Json experiments{
        {"domain", "d06_go"},
        {"n_samples", static_cast<int>(ds.x.size())},
        {"regression",
         Json{{"cypha_scores", reg_metrics_native(reg, test_x, test_y_reg)},
              {"ridge_rmse", cypha::bench::ridge_baseline(train_x, train_y_reg, test_x, test_y_reg).rmse}}},
        {"classification", Json{{"cypha_scores", clf_metrics_native(clf.infer, test_x, test_y_cls)}}},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d06", experiments);
    return experiments;
}

Json run_d07() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto ds = cypha::bench::generate_poker_dataset(cypha::bench::bench_scale(12000, 3000), kBenchSeed);
    TabularDataset tab;
    tab.x = ds.x;
    tab.y = ds.y;
    std::vector<std::vector<double>> train_x;
    std::vector<std::string> train_y;
    std::vector<std::vector<double>> test_x;
    std::vector<std::string> test_y;
    stratified_split(tab, 0.2, kBenchSeed, train_x, train_y, test_x, test_y);
    standardize_train_test(train_x, test_x);
    OnlineClassifier clf = make_online_classifier(static_cast<int>(train_x.front().size()), kBenchSeed,
                                                  regime.value("enc_lr", 0.002), regime);
    train_classifier_online(clf, train_x, train_y, std::max(1, regime.value("n_epochs", 1)), kBenchSeed);
    Json scores = clf_metrics_native(clf.infer, test_x, test_y);
    std::vector<double> epistemic;
    std::vector<double> boundary_dist;
    for (const auto& xrow : test_x) {
        epistemic.push_back(online_clf_epistemic(clf.infer, xrow));
        boundary_dist.push_back(std::abs(xrow[0] - 0.35));
    }
    scores["boundary_uncertainty_spearman"] = cypha::bench::safe_spearman(boundary_dist, epistemic);
    const Json experiments{
        {"domain", "d07_poker"},
        {"n_hands", static_cast<int>(ds.x.size())},
        {"cypha_scores", scores},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d07", experiments);
    return experiments;
}

Json run_d09() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto news = cypha::bench::load_news_documents(cypha::bench::bench_scale(2000, 800), kBenchSeed);
    TabularDataset tab;
    tab.x = news.x;
    tab.y.resize(news.y.size());
    for (std::size_t i = 0; i < news.y.size(); ++i) tab.y[i] = std::to_string(news.y[i]);
    std::vector<std::vector<double>> train_x;
    std::vector<std::string> train_y;
    std::vector<std::vector<double>> test_x;
    std::vector<std::string> test_y;
    stratified_split(tab, 0.2, kBenchSeed, train_x, train_y, test_x, test_y);
    standardize_train_test(train_x, test_x);
    OnlineClassifier model = make_online_classifier(static_cast<int>(train_x.front().size()), kBenchSeed,
                                                  regime.value("enc_lr", 0.002), regime);
    train_classifier_online(model, train_x, train_y, std::max(1, regime.value("n_epochs", 1)), kBenchSeed);
    Json news_scores = clf_metrics_native(model.infer, test_x, test_y);

    const auto gutenberg = cypha::bench::load_gutenberg_segments(cypha::bench::bench_scale(400, 80), kBenchSeed);
    cypha::bench::DocumentEncoder doc_enc(2000, 1, 2);
    std::vector<std::string> fit_docs = news.texts;
    fit_docs.insert(fit_docs.end(), gutenberg.segments.begin(), gutenberg.segments.end());
    doc_enc.fit(fit_docs);
    auto gutenberg_raw = doc_enc.encode_batch(gutenberg.segments);
    std::vector<std::vector<double>> gutenberg_x;
    gutenberg_x.reserve(gutenberg_raw.size());
    for (const auto& row : gutenberg_raw) {
        std::vector<double> d(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) d[j] = static_cast<double>(row[j]);
        gutenberg_x.push_back(std::move(d));
    }
    gutenberg_x = cypha::bench::reduce_features(gutenberg_x, static_cast<int>(train_x.front().size()));
    std::vector<double> mean(train_x.front().size(), 0.0);
    std::vector<double> stdv(train_x.front().size(), 1.0);
    for (const auto& row : train_x) {
        for (std::size_t j = 0; j < mean.size(); ++j) mean[j] += row[j];
    }
    for (double& m : mean) m /= static_cast<double>(train_x.size());
    for (const auto& row : train_x) {
        for (std::size_t j = 0; j < stdv.size(); ++j) {
            const double diff = row[j] - mean[j];
            stdv[j] += diff * diff;
        }
    }
    for (double& s : stdv) {
        s = std::sqrt(s / static_cast<double>(train_x.size()));
        if (s < 1e-12) s = 1.0;
    }
    for (auto& row : gutenberg_x) {
        for (std::size_t j = 0; j < row.size(); ++j) row[j] = (row[j] - mean[j]) / stdv[j];
    }
    std::vector<double> ep_in;
    std::vector<double> ep_ood;
    const int in_n = std::min(200, static_cast<int>(test_x.size()));
    for (int i = 0; i < in_n; ++i) ep_in.push_back(online_clf_epistemic(model.infer, test_x[static_cast<std::size_t>(i)]));
    for (const auto& row : gutenberg_x) ep_ood.push_back(online_clf_epistemic(model.infer, row));
    double mean_in = 0.0;
    double mean_ood = 0.0;
    for (double v : ep_in) mean_in += v;
    for (double v : ep_ood) mean_ood += v;
    if (!ep_in.empty()) mean_in /= static_cast<double>(ep_in.size());
    if (!ep_ood.empty()) mean_ood /= static_cast<double>(ep_ood.size());

    TabularDataset book_tab;
    book_tab.x = gutenberg_x;
    book_tab.y = gutenberg.labels;
    std::vector<std::vector<double>> book_train_x;
    std::vector<std::string> book_train_y;
    std::vector<std::vector<double>> book_test_x;
    std::vector<std::string> book_test_y;
    stratified_split(book_tab, 0.2, kBenchSeed + 1, book_train_x, book_train_y, book_test_x, book_test_y);
    OnlineClassifier book_model = make_online_classifier(static_cast<int>(book_train_x.front().size()), kBenchSeed + 1,
                                                         regime.value("enc_lr", 0.002), regime);
    train_classifier_online(book_model, book_train_x, book_train_y, 3, kBenchSeed + 1);

    const Json experiments{
        {"domain", "d09_documents"},
        {"20news",
         Json{{"n_samples", static_cast<int>(news.x.size())},
              {"data_source", news.source},
              {"cypha_scores", news_scores}}},
        {"gutenberg_ood",
         Json{{"mean_epistemic_in", mean_in},
              {"mean_epistemic_ood", mean_ood},
              {"n_ood", static_cast<int>(ep_ood.size())}}},
        {"gutenberg_book_classification",
         Json{{"n_segments", static_cast<int>(gutenberg_x.size())},
              {"cypha_scores", clf_metrics_native(book_model.infer, book_test_x, book_test_y)}}},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d09", experiments);
    return experiments;
}

Json run_d10_experiment_a() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto ecg = cypha::bench::load_ecg5000(kBenchSeed);
    const int win = std::min(32, static_cast<int>(ecg.x_train.front().size()));
    cypha::bench::TimeSeriesEncoder enc(win, 16);
    std::vector<std::vector<double>> xtr;
    std::vector<std::string> ytr;
    std::vector<std::vector<double>> xte;
    std::vector<std::string> yte;
    for (std::size_t i = 0; i < ecg.x_train.size(); ++i) {
        const auto feat = enc.encode_series(ecg.x_train[i]);
        std::vector<double> row(feat.size());
        for (std::size_t j = 0; j < feat.size(); ++j) row[j] = static_cast<double>(feat[j]);
        xtr.push_back(std::move(row));
        ytr.push_back(std::to_string(ecg.y_train[i]));
    }
    for (std::size_t i = 0; i < ecg.x_test.size(); ++i) {
        const auto feat = enc.encode_series(ecg.x_test[i]);
        std::vector<double> row(feat.size());
        for (std::size_t j = 0; j < feat.size(); ++j) row[j] = static_cast<double>(feat[j]);
        xte.push_back(std::move(row));
        yte.push_back(std::to_string(ecg.y_test[i]));
    }
    OnlineClassifier clf = make_online_classifier(static_cast<int>(xtr.front().size()), kBenchSeed,
                                                  regime.value("enc_lr", 0.002), regime);
    train_classifier_online(clf, xtr, ytr, 8, kBenchSeed);
    Json m = clf_metrics_native(clf.infer, xte, yte);
    m["data_source"] = ecg.source;
    return m;
}

Json run_d10_experiment_b() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto ecg = cypha::bench::load_ecg5000(kBenchSeed);
    cypha::bench::TimeSeriesEncoder enc(10, 6);
    std::vector<std::vector<double>> x;
    std::vector<std::string> y;
    for (std::size_t si = 0; si < ecg.x_train.size(); ++si) {
        const auto [windows, _] = enc.sliding_windows(ecg.x_train[si], 5);
        for (const auto& row : windows) {
            x.push_back(row);
            y.push_back(std::to_string(ecg.y_train[si]));
        }
    }
    const int split = static_cast<int>(static_cast<double>(x.size()) * 0.8);
    std::vector<std::vector<double>> train_x(x.begin(), x.begin() + split);
    std::vector<std::string> train_y(y.begin(), y.begin() + split);
    std::vector<std::vector<double>> test_x(x.begin() + split, x.end());
    std::vector<std::string> test_y(y.begin() + split, y.end());
    OnlineClassifier clf = make_online_classifier(static_cast<int>(train_x.front().size()), kBenchSeed + 1,
                                                  regime.value("enc_lr", 0.002), regime);
    train_classifier_online(clf, train_x, train_y, 4, kBenchSeed + 1);
    return clf_metrics_native(clf.infer, test_x, test_y);
}

Json run_d10_experiment_c() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto ecg = cypha::bench::load_ecg5000(kBenchSeed);
    int normal_cls = ecg.y_train.empty() ? 1 : ecg.y_train[0];
    for (int v : ecg.y_train) normal_cls = std::min(normal_cls, v);
    cypha::bench::TimeSeriesEncoder enc(std::min(50, static_cast<int>(ecg.x_train.front().size())));
    std::vector<std::vector<double>> xtr;
    std::vector<std::string> ytr;
    for (std::size_t i = 0; i < ecg.x_train.size(); ++i) {
        if (ecg.y_train[i] != normal_cls) continue;
        const auto feat = enc.encode_series(ecg.x_train[i]);
        std::vector<double> row(feat.size());
        for (std::size_t j = 0; j < feat.size(); ++j) row[j] = static_cast<double>(feat[j]);
        xtr.push_back(std::move(row));
        ytr.push_back(std::to_string(normal_cls));
    }
    std::vector<std::vector<double>> xte;
    std::vector<int> yte_bin;
    for (std::size_t i = 0; i < ecg.x_test.size(); ++i) {
        const auto feat = enc.encode_series(ecg.x_test[i]);
        std::vector<double> row(feat.size());
        for (std::size_t j = 0; j < feat.size(); ++j) row[j] = static_cast<double>(feat[j]);
        xte.push_back(std::move(row));
        yte_bin.push_back(ecg.y_test[i] == normal_cls ? 0 : 1);
    }
    OnlineClassifier clf = make_online_classifier(static_cast<int>(xtr.front().size()), kBenchSeed + 2,
                                                  regime.value("enc_lr", 0.002), regime);
    train_classifier_online(clf, xtr, ytr, 3, kBenchSeed + 2);
    std::vector<double> scores;
    for (const auto& row : xte) scores.push_back(online_clf_epistemic(clf.infer, row));
    return Json{{"ood_auroc", cypha::bench::safe_auroc(yte_bin, scores)}};
}

Json run_d10_experiment_d() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto fin = cypha::bench::load_financial_returns(kBenchSeed);
    const int split = static_cast<int>(static_cast<double>(fin.x.size()) * 0.8);
    std::vector<std::vector<double>> train_x(fin.x.begin(), fin.x.begin() + split);
    std::vector<std::string> train_y;
    train_y.reserve(static_cast<std::size_t>(split));
    for (int i = 0; i < split; ++i) train_y.push_back(std::to_string(fin.y[static_cast<std::size_t>(i)]));
    std::vector<std::vector<double>> test_x(fin.x.begin() + split, fin.x.end());
    std::vector<std::string> test_y;
    for (std::size_t i = static_cast<std::size_t>(split); i < fin.y.size(); ++i) {
        test_y.push_back(std::to_string(fin.y[i]));
    }
    OnlineClassifier clf = make_online_classifier(static_cast<int>(train_x.front().size()), kBenchSeed + 4,
                                                  regime.value("enc_lr", 0.002), regime);
    train_classifier_online(clf, train_x, train_y, 3, kBenchSeed + 4);
    Json m = clf_metrics_native(clf.infer, test_x, test_y);
    m["note"] = "near_chance_expected";
    return m;
}

Json run_d10() {
    Json experiments{
        {"10A_ecg_classification", run_d10_experiment_a()},
        {"10B_ecg_sliding_window", run_d10_experiment_b()},
        {"10C_ecg_ood_detection", run_d10_experiment_c()},
        {"10D_financial_return_sign", run_d10_experiment_d()},
        {"backend", "cypha_core"},
    };
    if (g_ssm_diagnose) {
        experiments["10E_ssm_diagnose"] = run_d10_ssm_diagnose();
    }
    cypha::bench::finalize_domain("d10", experiments);
    return experiments;
}

class CartPoleEnv {
  public:
    explicit CartPoleEnv(std::uint64_t seed) : rng_(static_cast<std::mt19937::result_type>(seed)) { reset(); }

    std::array<double, 4> reset() {
        std::uniform_real_distribution<double> uni(-0.05, 0.05);
        for (double& v : state_) v = uni(rng_);
        return state_;
    }

    std::pair<std::array<double, 4>, double> step(int action) {
        const double force = (action == 1) ? force_mag_ : -force_mag_;
        const double costheta = std::cos(state_[2]);
        const double sintheta = std::sin(state_[2]);
        const double temp = (force + mp_ * l_ * state_[3] * state_[3] * sintheta) / (mc_ + mp_);
        const double theta_acc =
            (g_ * sintheta - costheta * temp) / (l_ * (4.0 / 3.0 - mp_ * costheta * costheta / (mc_ + mp_)));
        const double x_acc = temp - mp_ * l_ * theta_acc * costheta / (mc_ + mp_);
        state_[0] += dt_ * state_[1];
        state_[1] += dt_ * x_acc;
        state_[2] += dt_ * state_[3];
        state_[3] += dt_ * theta_acc;
        const bool done = std::abs(state_[2]) > 0.2094 || std::abs(state_[0]) > 2.4;
        return {state_, done ? 0.0 : 1.0};
    }

    std::mt19937& rng() { return rng_; }

  private:
    std::mt19937 rng_;
    std::array<double, 4> state_{};
    double g_{9.8};
    double mc_{1.0};
    double mp_{0.1};
    double l_{0.5};
    double dt_{0.02};
    double force_mag_{10.0};
};

std::pair<std::vector<std::vector<double>>, std::vector<double>> collect_cartpole_returns(int n_episodes,
                                                                                          std::uint64_t seed) {
    CartPoleEnv env(seed);
    constexpr double gamma = 0.99;
    std::vector<std::vector<double>> states;
    std::vector<double> returns;
    for (int ep = 0; ep < n_episodes; ++ep) {
        auto s = env.reset();
        std::vector<std::pair<std::array<double, 4>, double>> traj;
        for (int t = 0; t < 200; ++t) {
            const int action = static_cast<int>(env.rng()() % 2);
            const auto [s_next, reward] = env.step(action);
            traj.emplace_back(s, reward);
            s = s_next;
            if (reward == 0.0) break;
        }
        double g = 0.0;
        for (auto it = traj.rbegin(); it != traj.rend(); ++it) g = it->second + gamma * g;
        for (const auto& [st, _] : traj) {
            states.push_back({st[0], st[1], st[2], st[3]});
            returns.push_back(g);
        }
    }
    return {states, returns};
}

Json run_d11_experiment_a() {
    const cypha::bench::ProfileJson regime = cypha::bench::regression_params();
    const auto [xs, ys] = collect_cartpole_returns(cypha::bench::bench_scale(1000, 300), kBenchSeed);
    const int split = static_cast<int>(static_cast<double>(xs.size()) * 0.8);
    std::vector<std::vector<double>> train_x(xs.begin(), xs.begin() + split);
    std::vector<double> train_y(ys.begin(), ys.begin() + split);
    std::vector<std::vector<double>> test_x(xs.begin() + split, xs.end());
    std::vector<double> test_y(ys.begin() + split, ys.end());
    OnlineRegressor reg = make_online_regressor(4, kBenchSeed, regime);
    for (int p = 0; p < 4; ++p) {
        std::vector<int> order(static_cast<std::size_t>(train_x.size()));
        for (int i = 0; i < static_cast<int>(order.size()); ++i) order[static_cast<std::size_t>(i)] = i;
        std::shuffle(order.begin(), order.end(), make_rng(kBenchSeed + static_cast<std::uint64_t>(p)));
        for (int idx : order) {
            online_reg_train_step(reg, train_x[static_cast<std::size_t>(idx)], train_y[static_cast<std::size_t>(idx)]);
        }
    }
    cypha::sync_infer_model_from_memory(reg.infer, reg.mem);
    Json m = reg_metrics_native(reg, test_x, test_y);
    m["ridge_rmse"] = cypha::bench::ridge_baseline(train_x, train_y, test_x, test_y).rmse;
    return m;
}

Json run_d11_experiment_b() {
    const cypha::bench::ProfileJson regime = cypha::bench::regression_params();
    constexpr int size = 4;
    constexpr int n_states = size * size;
    std::vector<std::vector<double>> q_true(static_cast<std::size_t>(n_states), std::vector<double>(4, 0.0));
    for (int iter = 0; iter < 1000; ++iter) {
        auto q_new = q_true;
        for (int s = 0; s < n_states; ++s) {
            if (s == n_states - 1) continue;
            const int r = s / size;
            const int c = s % size;
            const int moves[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (int a = 0; a < 4; ++a) {
                const int nr = std::max(0, std::min(size - 1, r + moves[a][0]));
                const int nc = std::max(0, std::min(size - 1, c + moves[a][1]));
                const int ns = nr * size + nc;
                const double reward = (ns == n_states - 1) ? 1.0 : 0.0;
                q_new[static_cast<std::size_t>(s)][static_cast<std::size_t>(a)] =
                    reward + 0.9 * *std::max_element(q_true[static_cast<std::size_t>(ns)].begin(),
                                                      q_true[static_cast<std::size_t>(ns)].end());
            }
        }
        q_true = std::move(q_new);
    }
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    for (int s = 0; s < n_states; ++s) {
        for (int a = 0; a < 4; ++a) {
            std::vector<double> row(static_cast<std::size_t>(n_states + 4), 0.0);
            row[static_cast<std::size_t>(s)] = 1.0;
            row[static_cast<std::size_t>(n_states + a)] = 1.0;
            x.push_back(std::move(row));
            y.push_back(q_true[static_cast<std::size_t>(s)][static_cast<std::size_t>(a)]);
        }
    }
    const int split = static_cast<int>(static_cast<double>(x.size()) * 0.8);
    OnlineRegressor reg = make_online_regressor(static_cast<int>(x.front().size()), kBenchSeed + 1, regime);
    for (int p = 0; p < 5; ++p) {
        std::vector<int> order(static_cast<std::size_t>(split));
        for (int i = 0; i < split; ++i) order[static_cast<std::size_t>(i)] = i;
        std::shuffle(order.begin(), order.end(), make_rng(kBenchSeed + static_cast<std::uint64_t>(p)));
        for (int idx : order) {
            online_reg_train_step(reg, x[static_cast<std::size_t>(idx)], y[static_cast<std::size_t>(idx)]);
        }
    }
    cypha::sync_infer_model_from_memory(reg.infer, reg.mem);
    std::vector<double> preds;
    for (int i = split; i < static_cast<int>(x.size()); ++i) {
        double y_hat = 0.0;
        double unc = 0.0;
        online_reg_predict(reg, x[static_cast<std::size_t>(i)], y_hat, unc);
        preds.push_back(y_hat);
    }
    std::vector<double> y_test(y.begin() + split, y.end());
    return Json{{"q_value_mae", cypha::bench::mae(y_test, preds)}, {"n_pairs", static_cast<int>(preds.size())}};
}

Json run_d11_experiment_c() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    CartPoleEnv env(kBenchSeed + 2);
    std::vector<std::vector<double>> pairs_x;
    std::vector<std::string> pairs_y;
    for (int n = 0; n < 500; ++n) {
        const auto s0 = env.reset();
        std::vector<std::array<double, 4>> traj_a;
        std::vector<std::array<double, 4>> traj_b;
        double ret_a = 0.0;
        double ret_b = 0.0;
        auto s = s0;
        for (int t = 0; t < 50; ++t) {
            const int action = static_cast<int>(env.rng()() % 2);
            const auto [s_next, reward] = env.step(action);
            traj_a.push_back(s);
            ret_a += reward;
            s = s_next;
            if (reward == 0.0) break;
        }
        s = s0;
        for (int t = 0; t < 50; ++t) {
            const int action = static_cast<int>(env.rng()() % 2);
            const auto [s_next, reward] = env.step(action);
            traj_b.push_back(s);
            ret_b += reward;
            s = s_next;
            if (reward == 0.0) break;
        }
        if (traj_a.empty() || traj_b.empty()) continue;
        std::array<double, 4> mean_a{};
        std::array<double, 4> mean_b{};
        for (const auto& st : traj_a) {
            for (int i = 0; i < 4; ++i) mean_a[static_cast<std::size_t>(i)] += st[static_cast<std::size_t>(i)];
        }
        for (const auto& st : traj_b) {
            for (int i = 0; i < 4; ++i) mean_b[static_cast<std::size_t>(i)] += st[static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < 4; ++i) {
            mean_a[static_cast<std::size_t>(i)] /= static_cast<double>(traj_a.size());
            mean_b[static_cast<std::size_t>(i)] /= static_cast<double>(traj_b.size());
        }
        std::vector<double> feat;
        feat.insert(feat.end(), mean_a.begin(), mean_a.end());
        feat.insert(feat.end(), mean_b.begin(), mean_b.end());
        feat.push_back(static_cast<double>(traj_a.size()));
        feat.push_back(static_cast<double>(traj_b.size()));
        pairs_x.push_back(std::move(feat));
        pairs_y.push_back(ret_a > ret_b ? "1" : "0");
    }
    const int split = static_cast<int>(static_cast<double>(pairs_x.size()) * 0.8);
    std::vector<std::vector<double>> train_x(pairs_x.begin(), pairs_x.begin() + split);
    std::vector<std::string> train_y(pairs_y.begin(), pairs_y.begin() + split);
    std::vector<std::vector<double>> test_x(pairs_x.begin() + split, pairs_x.end());
    std::vector<std::string> test_y(pairs_y.begin() + split, pairs_y.end());
    OnlineClassifier clf = make_online_classifier(static_cast<int>(train_x.front().size()), kBenchSeed + 2,
                                                  regime.value("enc_lr", 0.002), regime);
    train_classifier_online(clf, train_x, train_y, 4, kBenchSeed + 2);
    return clf_metrics_native(clf.infer, test_x, test_y);
}

Json run_d11() {
    const Json experiments{
        {"11A_cartpole_value_regression", run_d11_experiment_a()},
        {"11B_gridworld_q_estimation", run_d11_experiment_b()},
        {"11C_trajectory_preference", run_d11_experiment_c()},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d11", experiments);
    return experiments;
}

Json run_d12_experiment_a() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto data = load_nsl_kdd();
    std::vector<std::vector<double>> x_norm;
    for (std::size_t i = 0; i < data.x_train.size(); ++i) {
        if (data.y_train[i] == 0) x_norm.push_back(data.x_train[i]);
    }
    OnlineClassifier clf = make_online_classifier(static_cast<int>(x_norm.front().size()), kBenchSeed,
                                                  regime.value("enc_lr", 0.002), regime);
    for (int p = 0; p < 3; ++p) {
        std::vector<int> order(static_cast<std::size_t>(x_norm.size()));
        for (int i = 0; i < static_cast<int>(order.size()); ++i) order[static_cast<std::size_t>(i)] = i;
        std::shuffle(order.begin(), order.end(), make_rng(kBenchSeed + static_cast<std::uint64_t>(p)));
        for (int idx : order) {
            (void)online_clf_train_step(clf, x_norm[static_cast<std::size_t>(idx)], "normal");
        }
    }
    cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
    std::vector<double> scores;
    for (const auto& row : data.x_test) scores.push_back(online_clf_epistemic(clf.infer, row));
    return Json{{"cypha_ood_auroc", cypha::bench::safe_auroc(data.y_test, scores)},
                {"data_source", data.source}};
}

Json run_d12_experiment_b() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto data = load_nsl_kdd();
    std::vector<std::vector<double>> tr_x;
    std::vector<std::string> tr_y;
    for (std::size_t i = 0; i < data.x_train.size(); ++i) {
        tr_x.push_back(data.x_train[i]);
        tr_y.push_back(data.y_train[i] == 0 ? "normal" : "attack");
    }
    OnlineClassifier clf = make_online_classifier(static_cast<int>(tr_x.front().size()), kBenchSeed + 1,
                                                  regime.value("enc_lr", 0.002), regime);
    train_classifier_online(clf, tr_x, tr_y, 3, kBenchSeed + 1);
    const int n = std::min(2000, static_cast<int>(data.x_test.size()));
    std::vector<double> scores;
    for (int i = 0; i < n; ++i) scores.push_back(online_clf_epistemic(clf.infer, data.x_test[static_cast<std::size_t>(i)]));
    double mean_epi = 0.0;
    for (double v : scores) mean_epi += v;
    if (!scores.empty()) mean_epi /= static_cast<double>(scores.size());
    return Json{{"mean_epistemic_attack", mean_epi}, {"n_test", n}};
}

Json run_d12_experiment_c() {
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const auto data = load_nsl_kdd();
    OnlineClassifier clf = make_online_classifier(static_cast<int>(data.x_test.front().size()), kBenchSeed + 2,
                                                  regime.value("enc_lr", 0.002), regime);
    int normal_count = 0;
    for (std::size_t i = 0; i < data.x_train.size() && normal_count < 500; ++i) {
        if (data.y_train[i] != 0) continue;
        (void)online_clf_train_step(clf, data.x_train[i], "normal");
        ++normal_count;
    }
    cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
    std::vector<int> attack_idx;
    for (std::size_t i = 0; i < data.y_test.size() && static_cast<int>(attack_idx.size()) < 200; ++i) {
        if (data.y_test[i] == 1) attack_idx.push_back(static_cast<int>(i));
    }
    if (attack_idx.empty()) return Json{{"detection_latency_steps", nullptr}};
    int correct = 0;
    int latency = -1;
    for (int step = 1; step <= static_cast<int>(attack_idx.size()); ++step) {
        const int idx = attack_idx[static_cast<std::size_t>(step - 1)];
        (void)online_clf_train_step(clf, data.x_test[static_cast<std::size_t>(idx)], "attack");
        if (online_clf_predict(clf.infer, data.x_test[static_cast<std::size_t>(idx)]) == "attack") ++correct;
        const double acc = static_cast<double>(correct) / static_cast<double>(step);
        if (acc >= 0.8 && latency < 0) latency = step;
    }
    return Json{{"detection_latency_steps", latency < 0 ? Json(nullptr) : Json(latency)},
                {"final_attack_acc", static_cast<double>(correct) / static_cast<double>(attack_idx.size())}};
}

Json run_d12() {
    const Json experiments{
        {"12A_binary_intrusion", run_d12_experiment_a()},
        {"12B_attack_types", run_d12_experiment_b()},
        {"12C_online_detection", run_d12_experiment_c()},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d12", experiments);
    return experiments;
}

Json run_d03_xor() {
    const int seeds = bench_fast_mode() ? 1 : 3;
    const int passes = bench_fast_mode() ? 2 : 8;
#if defined(_WIN32)
    const fs::path bench_exe = g_tool_dir / "xor_kernel_bench.exe";
#else
    const fs::path bench_exe = g_tool_dir / "xor_kernel_bench";
#endif
    if (!fs::is_regular_file(bench_exe)) {
        throw std::runtime_error("xor_kernel_bench not found beside cypha_bench_run: " + bench_exe.string());
    }
    std::ostringstream cmd;
    cmd << "\"" << bench_exe.string() << "\""
        << " --seeds " << seeds << " --passes " << passes << " --kernel-blend 1.0"
        << " --kernel-m 512 --gamma-scale 2.0 --kernel-lr-scale 2.0 --kernel-xor-features";
    const Json j = Json::parse(capture_process_output(cmd.str()));
    const Json experiments{
        {"S3_xor_linear",
         Json{{"accuracy", j.at("linear_mean_acc")}, {"seeds", seeds}, {"passes", passes}}},
        {"S3_xor_kernel_llr",
         Json{{"accuracy", j.at("kernel_mean_acc")},
              {"delta_pp", j.at("delta_pp")},
              {"kernel_m", j.value("kernel_m", 512)},
              {"kernel_feature_mode", j.value("kernel_feature_mode", "xor_pair")},
              {"backend", "xor_kernel_bench_native"}}},
    };
    cypha::bench::finalize_domain("d03_xor_kernel", experiments);
    return experiments;
}

Json run_d18_intelligence_profile() {
    const fs::path root = cypha::bench::repo_root();
    const auto profiler = cypha::intelligence::profile_from_reference_fixture(root);
    nlohmann::json report = cypha::intelligence::intelligence_profile_report_json(profiler);
    const Json experiments = {
        {"P1_reference_fixture_profile",
         Json{{"criticality_score", report.at("criticality_score")},
              {"health_signal", report.at("health_signal")},
              {"navigation_loss", report.at("navigation_loss")},
              {"landscape_kappa", report.at("landscape_kappa")}}},
    };
    cypha::bench::finalize_domain("d18_intelligence_profile", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d18_intelligence_profile.json";
    std::ofstream out(table_path);
    if (out) {
        out << report.dump(2);
    }
    return experiments;
}

Json run_d19_cell_hypothesis_smoke() {
    struct Tier1Spec {
        const char* id;
        const char* bench_mode;
    };
    const Tier1Spec tier1[] = {
        {"H01", "hybrid"},
        {"H03", "ssm"},
        {"H04", "ssm_gria"},
        {"H05", "hybrid"},
    };

    const int n_train = cypha::bench::bench_scale(400, 120);
    const int n_eval = cypha::bench::bench_scale(80, 40);
    Json rows = Json::array();
    Json scaffold = Json::array();
    for (const auto& spec : tier1) {
        cypha::cyphalm::CyphaLMConfig cfg;
        cypha::cyphalm::apply_bench_profile("d17", cfg);
        cypha::cyphalm::apply_bench_mode(cypha::cyphalm::parse_bench_mode(spec.bench_mode), cfg);
        if (cfg.vocab_size < 256) cfg.vocab_size = 256;

        cypha::cyphalm::LMCorpus corpus;
        corpus.profile = "d17";
        corpus.source = "synthetic";
        corpus.vocab_size = cfg.vocab_size;
        corpus.train_ids =
            cypha::cyphalm::synthetic_corpus(n_train + n_eval + 32, cfg.vocab_size, cfg.seed);
        const std::size_t split = static_cast<std::size_t>(n_train);
        corpus.eval_ids.assign(corpus.train_ids.begin() + static_cast<std::ptrdiff_t>(split),
                               corpus.train_ids.end());
        corpus.train_ids.resize(split);
        cfg.vocab_size = corpus.vocab_size;

        cypha::cyphalm::CyphaLMModel model(cfg);
        model.train_sequence(corpus.train_ids, n_train, cfg.train_epochs);
        const double bpc = model.eval_bpc(corpus.eval_ids, n_eval);
        rows.push_back(Json{{"id", spec.id},
                            {"bench_mode", spec.bench_mode},
                            {"bpc", std::isnan(bpc) ? Json(nullptr) : Json(bpc)}});
    }
    scaffold.push_back(Json{{"status", "scaffold"}, {"remaining_variants", 24}});

    const Json experiments{
        {"tier1_smoke", rows},
        {"scaffold", scaffold},
        {"backend", "cypha_cell_hypothesis_sweep"},
    };
    cypha::bench::finalize_domain("d19_cell_hypothesis", experiments);
    return experiments;
}

Json run_d20_cell_hypothesis_overnight_smoke() {
    struct OvernightSpec {
        const char* id;
        const char* bench_mode;
    };
    const OvernightSpec variants[] = {
        {"B2", "hybrid"},
        {"H06", "hybrid"},
        {"H14", "hybrid"},
    };

    const int n_train = cypha::bench::bench_scale(200, 200);
    const int n_eval = cypha::bench::bench_scale(40, 40);
    Json rows = Json::array();
    for (const auto& spec : variants) {
        cypha::cyphalm::CyphaLMConfig cfg;
        cypha::cyphalm::apply_bench_profile("d17", cfg);
        cypha::cyphalm::apply_cell_variant(spec.id, cfg);
        cypha::cyphalm::apply_bench_mode(cypha::cyphalm::parse_bench_mode(spec.bench_mode), cfg);
        if (cfg.vocab_size < 256) cfg.vocab_size = 256;

        cypha::cyphalm::LMCorpus corpus;
        corpus.profile = "d17";
        corpus.source = "synthetic";
        corpus.vocab_size = cfg.vocab_size;
        corpus.train_ids =
            cypha::cyphalm::synthetic_corpus(n_train + n_eval + 32, cfg.vocab_size, cfg.seed);
        const std::size_t split = static_cast<std::size_t>(n_train);
        corpus.eval_ids.assign(corpus.train_ids.begin() + static_cast<std::ptrdiff_t>(split),
                               corpus.train_ids.end());
        corpus.train_ids.resize(split);
        cfg.vocab_size = corpus.vocab_size;

        cypha::cyphalm::CyphaLMModel model(cfg);
        model.train_sequence(corpus.train_ids, n_train, cfg.train_epochs);
        const double bpc = model.eval_bpc(corpus.eval_ids, n_eval);
        rows.push_back(Json{{"id", spec.id},
                            {"bench_mode", spec.bench_mode},
                            {"n_train", n_train},
                            {"bpc", std::isnan(bpc) ? Json(nullptr) : Json(bpc)}});
    }

    const Json experiments{
        {"overnight_sweep_smoke", rows},
        {"variant_count", 3},
        {"backend", "cypha_cell_hypothesis_sweep --overnight-sweep-smoke"},
    };
    cypha::bench::finalize_domain("d20_cell_hypothesis_overnight", experiments);
    return experiments;
}

Json run_d22_intelligence_cross_profile() {
    const Json d18 = run_d18_intelligence_profile();
    const Json d16_ewc = run_d16_ewc_probe();
    const Json d20 = run_d20_cell_hypothesis_overnight_smoke();

    const Json experiments{
        {"d18_intelligence_profile", d18},
        {"d16_ewc_probe", d16_ewc},
        {"d20_cell_sweep_smoke", d20},
        {"sub_domains", Json::array({"d18", "d16", "d20"})},
        {"backend", "cypha_bench_cross_intelligence"},
    };
    cypha::bench::finalize_domain("d22_intelligence_cross_profile", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d22_intelligence_cross_profile.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

fs::path resolve_native_exe_dir() {
    if (const char* raw = std::getenv("CYPHA_NATIVE_EXE_DIR")) {
        if (*raw != '\0') {
            return fs::absolute(raw);
        }
    }
    return fs::current_path();
}

Json load_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open JSON: " + path.string());
    }
    Json j;
    in >> j;
    return j;
}

void validate_overnight_lock_section(const Json& section, const char* section_name) {
    if (!section.is_object()) {
        throw std::runtime_error(std::string(section_name) + " is not an object");
    }
    if (!section.contains("status") || !section["status"].is_string()) {
        throw std::runtime_error(std::string(section_name) + " missing status");
    }
    const std::string status = section["status"].get<std::string>();
    if (status == "pending") {
        throw std::runtime_error(std::string(section_name) + " still pending");
    }
    if (!section.contains("bpc") || section["bpc"].is_null()) {
        throw std::runtime_error(std::string(section_name) + " missing bpc");
    }
    if (!section.contains("run_at") || section["run_at"].is_null()) {
        throw std::runtime_error(std::string(section_name) + " missing run_at");
    }
}

Json run_baseline_lock_subprocess(const fs::path& exe_dir, const fs::path& lock_path,
                                  const char* run_kind, int n_train, int n_eval,
                                  bool fast = true, bool medium = false) {
    const fs::path baseline_lock_exe = cypha::bench::resolve_runner_exe("cypha_baseline_lock", exe_dir);
    if (!fs::is_regular_file(baseline_lock_exe)) {
        throw std::runtime_error("missing cypha_baseline_lock: " + baseline_lock_exe.string());
    }
    std::vector<std::string> args = {
        "--run", run_kind, "--n-train", std::to_string(n_train), "--n-eval",
        std::to_string(n_eval), "--lock-file", fs::absolute(lock_path).string(), "--exe-dir",
        fs::absolute(exe_dir).string()};
    if (fast) {
        args.insert(args.begin() + 2, "--fast");
    } else if (medium) {
        args.insert(args.begin() + 2, "--medium");
    }
    const cypha::bench::RunProcessResult proc = cypha::bench::run_executable_capture(baseline_lock_exe, args);
    if (proc.exit_code != 0) {
        throw std::runtime_error(std::string("cypha_baseline_lock --run ") + run_kind + " exit=" +
                                 std::to_string(proc.exit_code));
    }
    if (proc.stdout_text.empty()) {
        throw std::runtime_error(std::string("cypha_baseline_lock --run ") + run_kind + " produced no stdout");
    }
    return Json::parse(proc.stdout_text);
}

Json run_d23_overnight_lock_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path lock_path = fs::current_path() / "d23_overnight_lock_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const int n_train = cypha::bench::bench_scale(200, 200);
    const int n_eval = cypha::bench::bench_scale(64, 64);

    const Json d17_report = run_baseline_lock_subprocess(exe_dir, lock_path, "d17", n_train, n_eval);
    const Json d21_report = run_baseline_lock_subprocess(exe_dir, lock_path, "d21", n_train, n_eval);

    const Json lock = load_json_file(lock_path);
    if (!lock.contains("overnight_results")) {
        throw std::runtime_error("lock JSON missing overnight_results");
    }
    validate_overnight_lock_section(lock["overnight_results"], "overnight_results");
    if (!lock.contains("rpsm_results")) {
        throw std::runtime_error("lock JSON missing rpsm_results");
    }
    validate_overnight_lock_section(lock["rpsm_results"], "rpsm_results");

    const Json experiments{
        {"d17_baseline_lock", d17_report},
        {"d21_baseline_lock", d21_report},
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"rpsm_results", lock["rpsm_results"]},
        {"n_train", n_train},
        {"n_eval", n_eval},
        {"backend", "cypha_baseline_lock"},
    };
    cypha::bench::finalize_domain("d23_overnight_lock_validation", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d23_overnight_lock_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

void validate_cell_sweep_lock_section(const Json& lock) {
    if (lock.contains("cell_sweep_results")) {
        validate_overnight_lock_section(lock["cell_sweep_results"], "cell_sweep_results");
        return;
    }
    if (!lock.contains("overnight_results")) {
        throw std::runtime_error("lock JSON missing cell_sweep_results and overnight_results");
    }
    const Json& section = lock["overnight_results"];
    if (!section.is_object() || section.value("mode", "") != "cell-sweep") {
        throw std::runtime_error("lock JSON missing populated cell_sweep_results");
    }
    validate_overnight_lock_section(section, "overnight_results (cell-sweep)");
}

constexpr double kD17HybridPinBpc = 2.873;
constexpr double kD17HybridPinTolerance = 0.02;
constexpr double kD17ProductionPinTolerance = 0.05;
constexpr int kProductionNTrainMin = 300000;

void require_lock_key(const Json& obj, const char* key, const char* ctx) {
    if (!obj.contains(key) || obj[key].is_null()) {
        throw std::runtime_error(std::string(ctx) + " missing '" + key + "'");
    }
}

void validate_baseline_lock_schema(const Json& lock) {
    if (!lock.contains("schema_version") || !lock["schema_version"].is_number_integer() ||
        lock["schema_version"].get<int>() != 1) {
        throw std::runtime_error("schema_version must be 1");
    }

    require_lock_key(lock, "d17_hybrid_baseline", "lock");
    const Json& d17 = lock["d17_hybrid_baseline"];
    if (!d17.is_object()) {
        throw std::runtime_error("d17_hybrid_baseline is not an object");
    }
    for (const char* key : {"bpc", "profile", "mode", "n_train", "n_eval"}) {
        require_lock_key(d17, key, "d17_hybrid_baseline");
    }
    if (d17["profile"].get<std::string>() != "d17") {
        throw std::runtime_error("d17_hybrid_baseline profile must be d17");
    }
    if (d17["mode"].get<std::string>() != "hybrid") {
        throw std::runtime_error("d17_hybrid_baseline mode must be hybrid");
    }
    if (!d17["bpc"].is_number()) {
        throw std::runtime_error("d17_hybrid_baseline bpc must be numeric");
    }
    const double pin_delta = std::abs(d17["bpc"].get<double>() - kD17HybridPinBpc);
    if (pin_delta > kD17HybridPinTolerance) {
        throw std::runtime_error("d17_hybrid_baseline bpc pin out of reference tolerance");
    }

    require_lock_key(lock, "overnight_results", "lock");
    require_lock_key(lock, "rpsm_results", "lock");
    validate_overnight_lock_section(lock["overnight_results"], "overnight_results");
    validate_overnight_lock_section(lock["rpsm_results"], "rpsm_results");

    if (lock.contains("cell_sweep_results") && !lock["cell_sweep_results"].is_null()) {
        validate_overnight_lock_section(lock["cell_sweep_results"], "cell_sweep_results");
    }
}

std::string validate_production_tier_lock(const Json& lock) {
    if (!lock.contains("overnight_results") || !lock["overnight_results"].is_object()) {
        throw std::runtime_error("overnight_results is not an object");
    }
    const Json& overnight = lock["overnight_results"];
    require_lock_key(overnight, "n_train", "overnight_results");
    if (!overnight["n_train"].is_number_integer()) {
        throw std::runtime_error("overnight_results n_train must be an integer");
    }
    const int n_train = overnight["n_train"].get<int>();
    if (n_train < kProductionNTrainMin) {
        return "pending_production";
    }

    require_lock_key(overnight, "status", "overnight_results");
    const std::string status = overnight["status"].get<std::string>();
    if (status != "production" && status != "completed") {
        throw std::runtime_error("overnight_results status '" + status +
                                 "' invalid for production tier (n_train=" + std::to_string(n_train) +
                                 "; expected production or completed)");
    }
    require_lock_key(overnight, "bpc", "overnight_results");
    if (!overnight["bpc"].is_number()) {
        throw std::runtime_error("overnight_results bpc must be numeric");
    }
    const double bpc = overnight["bpc"].get<double>();
    const double prod_delta = std::abs(bpc - kD17HybridPinBpc);
    if (prod_delta > kD17ProductionPinTolerance) {
        throw std::runtime_error("overnight_results bpc out of production pin tolerance (delta " +
                                 std::to_string(prod_delta) + ")");
    }
    return "production_validated";
}

std::string validate_overnight_complete_lock(const Json& lock) {
    static constexpr const char* kSections[] = {"overnight_results", "rpsm_results",
                                                "cell_sweep_results"};
    for (const char* name : kSections) {
        if (!lock.contains(name) || !lock[name].is_object()) {
            throw std::runtime_error(std::string("lock JSON missing ") + name);
        }
        validate_overnight_lock_section(lock[name], name);
        require_lock_key(lock[name], "n_train", name);
        if (!lock[name]["n_train"].is_number_integer()) {
            throw std::runtime_error(std::string(name) + " n_train must be an integer");
        }
    }

    const int n_train = lock["overnight_results"]["n_train"].get<int>();
    for (const char* name : kSections) {
        if (lock[name]["n_train"].get<int>() != n_train) {
            throw std::runtime_error("overnight lock sections have mismatched n_train");
        }
    }

    if (n_train < kProductionNTrainMin) {
        return "pending_overnight_complete";
    }

    const int n_eval = lock["overnight_results"]["n_eval"].get<int>();
    for (const char* name : kSections) {
        require_lock_key(lock[name], "n_eval", name);
        if (!lock[name]["n_eval"].is_number_integer()) {
            throw std::runtime_error(std::string(name) + " n_eval must be an integer");
        }
        if (lock[name]["n_eval"].get<int>() != n_eval) {
            throw std::runtime_error("overnight lock sections have mismatched n_eval");
        }
        require_lock_key(lock[name], "status", name);
        const std::string status = lock[name]["status"].get<std::string>();
        if (status != "production" && status != "completed") {
            throw std::runtime_error(std::string(name) + " status '" + status +
                                     "' invalid for overnight complete tier (n_train=" +
                                     std::to_string(n_train) + "; expected production or completed)");
        }
    }

    return "overnight_complete_validated";
}

Json probe_bench_corpus_profile(const std::string& profile) {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_bench_profile(profile, cfg);
    if (profile == "d17" && cfg.vocab_size < 256) cfg.vocab_size = 256;

    Json report;
    report["profile"] = profile;
    try {
        const int max_chars = cypha::bench::bench_scale(500'000, 100'000);
        const cypha::cyphalm::LMCorpus corpus = cypha::cyphalm::load_bench_corpus(
            profile, max_chars, cfg.vocab_size, cfg.bpe_merges_path, cfg.bpe_vocab_path);
        report["ok"] = true;
        report["source"] = corpus.source;
        report["train_tokens"] = corpus.train_ids.size();
        report["eval_tokens"] = corpus.eval_ids.size();
        report["vocab_size"] = corpus.vocab_size;
    } catch (const std::exception& ex) {
        report["ok"] = false;
        report["error"] = ex.what();
    }
    return report;
}

Json run_d25_corpus_readiness() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path data_root = cypha::bench::data_dir();

    const Json d17_report = probe_bench_corpus_profile("d17");
    const Json d21_report = probe_bench_corpus_profile("d21");
    if (!d17_report.value("ok", false)) {
        throw std::runtime_error("d17 corpus readiness failed: " +
                                 d17_report.value("error", std::string("unknown")));
    }
    if (!d21_report.value("ok", false)) {
        throw std::runtime_error("d21 corpus readiness failed: " +
                                 d21_report.value("error", std::string("unknown")));
    }

    Json corpus_smoke_report{{"invoked", false}, {"reason", "corpus_smoke not built"}};
    const fs::path corpus_smoke_exe = cypha::bench::resolve_runner_exe("corpus_smoke", exe_dir);
    if (fs::is_regular_file(corpus_smoke_exe)) {
        const cypha::bench::RunProcessResult proc =
            cypha::bench::run_executable_capture(corpus_smoke_exe, {});
        corpus_smoke_report = Json{{"invoked", true},
                                   {"exit_code", proc.exit_code},
                                   {"stdout", proc.stdout_text},
                                   {"stderr", proc.stderr_text}};
        if (proc.exit_code != 0) {
            throw std::runtime_error("corpus_smoke exit=" + std::to_string(proc.exit_code));
        }
    }

    const fs::path wt_train = data_root / "wikitext2" / "wikitext-2" / "wiki.train.tokens";
    bool gutenberg_present = false;
    for (const char* name : {"moby_dick.txt", "alice.txt", "sherlock_holmes.txt"}) {
        if (fs::is_regular_file(data_root / "gutenberg" / name)) {
            gutenberg_present = true;
            break;
        }
    }

    const Json experiments{
        {"d17", d17_report},
        {"d21", d21_report},
        {"corpus_smoke", corpus_smoke_report},
        {"wikitext2_present", fs::is_regular_file(wt_train)},
        {"gutenberg_present", gutenberg_present},
        {"backend", "load_bench_corpus"},
    };
    cypha::bench::finalize_domain("d25_corpus_readiness", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d25_corpus_readiness.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d24_production_lock_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path lock_path = fs::current_path() / "d24_production_lock_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const int n_train = cypha::bench::bench_scale(200, 200);
    const int n_eval = cypha::bench::bench_scale(64, 64);

    const Json all_report =
        run_baseline_lock_subprocess(exe_dir, lock_path, "all", n_train, n_eval);

    const Json lock = load_json_file(lock_path);
    if (!lock.contains("overnight_results")) {
        throw std::runtime_error("lock JSON missing overnight_results");
    }
    validate_overnight_lock_section(lock["overnight_results"], "overnight_results");
    if (!lock.contains("rpsm_results")) {
        throw std::runtime_error("lock JSON missing rpsm_results");
    }
    validate_overnight_lock_section(lock["rpsm_results"], "rpsm_results");
    validate_cell_sweep_lock_section(lock);

    const Json experiments{
        {"all_baseline_lock", all_report},
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"rpsm_results", lock["rpsm_results"]},
        {"cell_sweep_results",
         lock.contains("cell_sweep_results") ? lock["cell_sweep_results"] : lock["overnight_results"]},
        {"n_train", n_train},
        {"n_eval", n_eval},
        {"backend", "cypha_baseline_lock"},
    };
    cypha::bench::finalize_domain("d24_production_lock_validation", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d24_production_lock_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d26_medium_overnight_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path lock_path = fs::current_path() / "d26_medium_overnight_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const int n_train = cypha::bench::bench_scale(5000, 5000);
    const int n_eval = cypha::bench::bench_scale(256, 256);

    const Json d17_report =
        run_baseline_lock_subprocess(exe_dir, lock_path, "d17", n_train, n_eval, false, true);

    const Json lock = load_json_file(lock_path);
    if (!lock.contains("overnight_results")) {
        throw std::runtime_error("lock JSON missing overnight_results");
    }
    validate_overnight_lock_section(lock["overnight_results"], "overnight_results");
    const std::string status = lock["overnight_results"]["status"].get<std::string>();
    if (status != "medium_smoke") {
        throw std::runtime_error("overnight_results status expected medium_smoke, got " + status);
    }
    const double bpc = lock["overnight_results"]["bpc"].get<double>();
    if (!std::isfinite(bpc)) {
        throw std::runtime_error("overnight_results bpc is not finite");
    }

    const Json experiments{
        {"d17_baseline_lock", d17_report},
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"n_train", n_train},
        {"n_eval", n_eval},
        {"bpc", bpc},
        {"status", status},
        {"backend", "cypha_baseline_lock"},
    };
    cypha::bench::finalize_domain("d26_medium_overnight_validation", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d26_medium_overnight_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d27_production_lock_validation() {
    const fs::path lock_path = fs::current_path() / "d27_production_lock_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);
    const std::string validation_status = validate_production_tier_lock(lock);

    const int n_train = lock["overnight_results"]["n_train"].get<int>();
    const Json experiments{
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"d17_hybrid_baseline", lock["d17_hybrid_baseline"]},
        {"n_train", n_train},
        {"validation_status", validation_status},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_pin_bpc", kD17HybridPinBpc},
        {"production_pin_tolerance", kD17ProductionPinTolerance},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d27_production_lock_validation", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d27_production_lock_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d28_overnight_complete_validation() {
    const fs::path lock_path = fs::current_path() / "d28_overnight_complete_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);
    const std::string production_status = validate_production_tier_lock(lock);
    const std::string validation_status = validate_overnight_complete_lock(lock);

    const int n_train = lock["overnight_results"]["n_train"].get<int>();
    const Json experiments{
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"rpsm_results", lock["rpsm_results"]},
        {"cell_sweep_results", lock["cell_sweep_results"]},
        {"d17_hybrid_baseline", lock["d17_hybrid_baseline"]},
        {"n_train", n_train},
        {"validation_status", validation_status},
        {"production_status", production_status},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_pin_bpc", kD17HybridPinBpc},
        {"production_pin_tolerance", kD17ProductionPinTolerance},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d28_overnight_complete_validation", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d28_overnight_complete_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

std::vector<DomainSpec> build_all_domains() {
    return {
        {"d01", "cypha_bench.domains.d01_statistical_baselines", run_d01},
        {"d02", "cypha_bench.domains.d02_regression", run_d02},
        {"d03", "cypha_bench.domains.d03_classification", run_d03},
        {"d03_xor", "cypha_bench.domains.d03_xor_kernel", run_d03_xor},
        {"d04", "cypha_bench.domains.d04_generation_language", run_d04},
        {"d05", "cypha_bench.domains.d05_chess", run_d05},
        {"d06", "cypha_bench.domains.d06_go", run_d06},
        {"d07", "cypha_bench.domains.d07_poker", run_d07},
        {"d08", "cypha_bench.domains.d08_computer_vision", run_d08},
        {"d09", "cypha_bench.domains.d09_documents", run_d09},
        {"d10", "cypha_bench.domains.d10_time_series", run_d10},
        {"d11", "cypha_bench.domains.d11_reinforcement_learning", run_d11},
        {"d12", "cypha_bench.domains.d12_anomaly_detection", run_d12},
        {"d13", "cypha_bench.domains.d13_compression", run_d13},
        {"d14", "cypha_bench.domains.d14_symbolic_regression", run_d14},
        {"d15", "cypha_bench.domains.d15_adversarial_robustness", run_d15},
        {"d16", "cypha_bench.domains.d16_multitask", run_d16},
        {"d17", "cypha_bench.domains.d17_cyphalm_integration", run_d17},
        {"d18", "cypha_bench.domains.d18_intelligence_profile", run_d18_intelligence_profile},
        {"d19", "cypha_bench.domains.d19_cell_hypothesis", run_d19_cell_hypothesis_smoke},
        {"d20", "cypha_bench.domains.d20_cell_hypothesis_overnight", run_d20_cell_hypothesis_overnight_smoke},
        {"d21", "cypha_bench.domains.d21_rpsm_overnight", run_d21_rpsm_overnight_smoke},
        {"d22", "cypha_bench.domains.d22_intelligence_cross_profile", run_d22_intelligence_cross_profile},
        {"d23", "cypha_bench.domains.d23_overnight_lock_validation", run_d23_overnight_lock_validation},
        {"d24", "cypha_bench.domains.d24_production_lock_validation", run_d24_production_lock_validation},
        {"d25", "cypha_bench.domains.d25_corpus_readiness", run_d25_corpus_readiness},
        {"d26", "cypha_bench.domains.d26_medium_overnight_validation", run_d26_medium_overnight_validation},
        {"d27", "cypha_bench.domains.d27_production_lock_validation", run_d27_production_lock_validation},
        {"d28", "cypha_bench.domains.d28_overnight_complete_validation", run_d28_overnight_complete_validation},
    };
}

}  // namespace

std::vector<DomainSpec> all_domains() { return build_all_domains(); }

}  // namespace cypha::bench
