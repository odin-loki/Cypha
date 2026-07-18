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
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <zlib.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
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
#include "cypha/curriculum.hpp"
#include "cypha/env.hpp"
#include "cypha/ewc_regularizer.hpp"
#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/cyphalm_parallel.hpp"
#include "cypha/cyphalm/cyphalm_views.hpp"
#include "cypha/cyphalm/ssm_diagnose.hpp"
#include "cypha/intelligence/profile_from_model.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/memory_train.hpp"
#include "cypha/preprocessor.hpp"
#include "cypha/regression_stub.hpp"
#include "cypha/replay_buffer.hpp"
#include "cypha/rff_features.hpp"
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
    const std::optional<std::string> v = cypha::env_get("CYPHA_BENCH_FAST");
    if (!v.has_value()) {
        return false;
    }
    const std::string& s = *v;
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
    pre.rff_gamma = pre_cfg.value("rff_gamma", 1.0);
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

// Phase 2 (multi-view online training for CyphaDIF, see docs/reports/MULTI_VIEW_DIF_PHASE2_PLAN.md):
// D03-only opt-in pilot. Reuses the exact same pure, format-agnostic transforms Phase 1 built for
// CyphaLM (`cypha/cyphalm/cyphalm_views.hpp`) -- they operate on plain `vector<int>`, so feeding a
// sample-index vector instead of a token-id vector reorders (x, y) pairs instead of tokens, with no
// changes to those functions. Off (empty/"same_order") reproduces the pre-existing per-pass
// `std::shuffle(..., make_rng(42 + p))` order exactly.
std::vector<int> dif_view_order(int n, const std::string& view_schedule, int pass_idx) {
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
    std::shuffle(idx.begin(), idx.end(), make_rng(42 + static_cast<std::uint64_t>(pass_idx)));
    if (view_schedule.empty() || view_schedule == "same_order") {
        return idx;
    }
    const auto view_names = cypha::cyphalm::resolve_view_schedule(view_schedule, 1);
    const std::string& view = view_names[static_cast<std::size_t>(pass_idx) % view_names.size()];
    if (view == "reverse" || view == "backward") {
        return cypha::cyphalm::view_reverse(idx);
    }
    if (view == "rotated") {
        return cypha::cyphalm::view_rotate_start(idx, std::max(1, n / 4));
    }
    if (view == "block_shuffle") {
        const int block = std::clamp(n / 4, 4, 64);
        const auto blocks = cypha::cyphalm::block_shuffle_blocks(
            idx, block, 42 + static_cast<std::uint64_t>(pass_idx) + 1000);
        std::vector<int> flat;
        flat.reserve(idx.size());
        for (const auto& b : blocks) flat.insert(flat.end(), b.begin(), b.end());
        return flat;
    }
    // "forward" (and any unrecognized name) falls back to the shuffled IID base order.
    return idx;
}

std::string d03_view_schedule_from_env() {
    const std::optional<std::string> v = cypha::env_get("CYPHA_D03_VIEW_SCHEDULE");
    if (!v.has_value() || v->empty()) return "";
    return *v;
}

// docs/FUTURE.md §6 curriculum ordering: hardest-first (by current-model confidence), randomised
// within a window. Opt-in / default-off (unset or 0 == pre-existing shuffled-per-pass order,
// byte-identical), same env-gate convention as `CYPHA_D03_VIEW_SCHEDULE` above. Applies to any
// caller of `train_eval_vectors` (D03 tabular + D08 vision both go through it); not limited to D03
// because curriculum ordering (unlike the view-schedule pilot) has no per-domain assumptions.
int curriculum_window_from_env() {
    const std::optional<std::string> v = cypha::env_get("CYPHA_CURRICULUM_WINDOW");
    if (!v.has_value() || v->empty()) return 0;
    try {
        const int w = std::stoi(*v);
        return w > 0 ? w : 0;
    } catch (...) {
        return 0;
    }
}

bool d03_class_block_from_env() {
    const std::optional<std::string> v = cypha::env_get("CYPHA_D03_CLASS_BLOCK");
    if (!v.has_value() || v->empty()) return false;
    return *v == "1" || *v == "true" || *v == "TRUE" || *v == "yes";
}

/// DIF-V1: group by label, shuffle block order (not arbitrary index windows).
std::vector<int> class_block_order(const std::vector<std::string>& labels, int pass_idx) {
    std::unordered_map<std::string, std::vector<int>> by_label;
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
        by_label[labels[static_cast<std::size_t>(i)]].push_back(i);
    }
    std::vector<std::string> keys;
    keys.reserve(by_label.size());
    for (const auto& kv : by_label) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    std::mt19937 rng = make_rng(42 + static_cast<std::uint64_t>(pass_idx) + 9000);
    std::shuffle(keys.begin(), keys.end(), rng);
    std::vector<int> out;
    out.reserve(labels.size());
    for (const std::string& k : keys) {
        auto& idxs = by_label[k];
        std::shuffle(idxs.begin(), idxs.end(), rng);
        out.insert(out.end(), idxs.begin(), idxs.end());
    }
    return out;
}

// Hardest-first-then-windowed-random order over `tr`/`train_y` using the model's *current*
// confidence (max softmax prob) on each row. Falls back to the pre-existing `dif_view_order`
// shuffle when the model has not yet seen any labels (k==0, i.e. before the first train step of an
// online run) since there is no meaningful confidence signal yet.
std::vector<int> curriculum_view_order(const cypha::CyphaInferModel& infer,
                                       const std::vector<std::vector<double>>& tr,
                                       const std::string& view_schedule, int pass_idx, int window,
                                       std::mt19937& curriculum_rng) {
    const int train_n = static_cast<int>(tr.size());
    const int k = static_cast<int>(infer.labels.size());
    if (k <= 0) {
        return dif_view_order(train_n, view_schedule, pass_idx);
    }
    const std::vector<double> flat = flatten_rowmajor(tr);
    std::vector<double> llr;
    cypha::batch_llr_from_x(infer, flat.data(), train_n, llr);
    std::vector<double> probs;
    cypha::softmax_batch_reference(llr.data(), train_n, k, 1e-8, probs);
    std::vector<double> confidences(static_cast<std::size_t>(train_n));
    for (int i = 0; i < train_n; ++i) {
        confidences[static_cast<std::size_t>(i)] =
            cypha::row_max_softmax_confidence(probs.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(k), k);
    }
    return cypha::curriculum_order_windowed(confidences, train_n, window, curriculum_rng);
}

Json clf_metrics_native(const cypha::CyphaInferModel& m, const std::vector<std::vector<double>>& xs,
                        const std::vector<std::string>& ys, std::vector<double>* epistemic_out = nullptr);

Json train_eval_vectors(const std::vector<std::vector<double>>& train_x, const std::vector<std::string>& train_y,
                        const std::vector<std::vector<double>>& test_x, const std::vector<std::string>& test_y,
                        const cypha::bench::ProfileJson& regime, const std::string& dataset_name,
                        const std::string& view_schedule = "") {
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
    const int curriculum_window = curriculum_window_from_env();
    const bool class_block = d03_class_block_from_env();
    std::mt19937 curriculum_rng = make_rng(4242);
    for (int p = 0; p < passes; ++p) {
        std::vector<int> order;
        if (class_block) {
            order = class_block_order(train_y, p);
        } else if (curriculum_window > 0) {
            order = curriculum_view_order(infer, tr, view_schedule, p, curriculum_window, curriculum_rng);
        } else {
            order = dif_view_order(train_n, view_schedule, p);
        }
        for (int idx : order) {
            cypha::dif_train_step_vector(infer, mem, replay, tr[static_cast<std::size_t>(idx)].data(), d,
                                         train_y[static_cast<std::size_t>(idx)], world_lr, delta_lr, world_lr,
                                         delta_lr, ood_sigma, tsp, rng, enc_updates, nullptr, nullptr);
        }
    }
    cypha::sync_infer_model_from_memory(infer, mem);

    Json test_scores = clf_metrics_native(infer, te, test_y);
    Json train_scores = clf_metrics_native(infer, tr, train_y);
    if (train_scores.contains("accuracy") && test_scores.contains("accuracy")) {
        test_scores["train_accuracy"] = train_scores["accuracy"];
        test_scores["generalization_gap"] =
            train_scores["accuracy"].get<double>() - test_scores["accuracy"].get<double>();
    }

    Json result = Json{
        {"dataset", dataset_name},
        {"cypha_scores", test_scores},
        {"baselines", cypha::bench::offline_classification_baselines_json(tr, train_y, te, test_y)},
        {"n_train", train_n},
        {"n_test", static_cast<int>(te.size())},
        {"expert_count", static_cast<int>(infer.labels.size())},
        {"backend", "cypha_core"},
    };
    if (!preprocessor_meta.empty()) {
        result["preprocessor"] = preprocessor_meta;
    }
    if (curriculum_window > 0) {
        result["curriculum_window"] = curriculum_window;
    }
    if (class_block) {
        result["class_block"] = true;
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
    // D03-only Phase 2 multi-view pilot, opt-in via env (default unset == pre-existing behavior).
    // See docs/reports/MULTI_VIEW_DIF_PHASE2_PLAN.md; not wired into D08/D09/D16.
    const std::string view_schedule = d03_view_schedule_from_env();
    Json result = train_eval_vectors(train_x, train_y, test_x, test_y, regime, ds.name, view_schedule);
    result["data_source"] = ds.source;
    if (!view_schedule.empty()) {
        result["view_schedule"] = view_schedule;
    }
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
                        const std::vector<std::string>& ys, std::vector<double>* epistemic_out) {
    const int n = static_cast<int>(xs.size());
    const int k = static_cast<int>(m.labels.size());
    std::vector<std::string> y_true;
    std::vector<std::string> y_pred;
    std::vector<double> epistemic;
    std::vector<double> confidences;
    std::vector<double> correct;
    std::vector<double> margins;
    y_true.reserve(xs.size());
    y_pred.reserve(xs.size());
    epistemic.reserve(xs.size());
    if (n > 0 && k > 0) {
        const std::vector<double> flat = flatten_rowmajor(xs);
        std::vector<double> llr;
        cypha::batch_llr_from_x(m, flat.data(), n, llr);
        std::vector<double> probs;
        cypha::softmax_batch_reference(llr.data(), n, k, 1e-12, probs);
        confidences.reserve(static_cast<std::size_t>(n));
        correct.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            y_true.push_back(ys[static_cast<std::size_t>(i)]);
            int best = 0;
            for (int j = 1; j < k; ++j) {
                if (llr[static_cast<std::size_t>(i * k + j)] > llr[static_cast<std::size_t>(i * k + best)]) {
                    best = j;
                }
            }
            const std::string pred = m.labels.empty() ? "0" : m.labels[static_cast<std::size_t>(best)];
            y_pred.push_back(pred);
            const double conf = cypha::row_max_softmax_confidence(
                probs.data() + static_cast<std::size_t>(i * k), k);
            confidences.push_back(conf);
            correct.push_back(pred == ys[static_cast<std::size_t>(i)] ? 1.0 : 0.0);
            epistemic.push_back(
                cypha::regression::mke_routing_entropy(probs.data() + static_cast<std::size_t>(i * k), k, 1e-12));
            margins.push_back(cypha::bench::logit_margin_top2(
                llr.data() + static_cast<std::size_t>(i * k), k));
        }
    }
    if (epistemic_out != nullptr) {
        *epistemic_out = epistemic;
    }
    double mean_epi = 0.0;
    double mean_conf = 0.0;
    if (!epistemic.empty()) {
        for (double v : epistemic) mean_epi += v;
        mean_epi /= static_cast<double>(epistemic.size());
    }
    if (!confidences.empty()) {
        for (double v : confidences) mean_conf += v;
        mean_conf /= static_cast<double>(confidences.size());
    }
    Json out{
        {"accuracy", cypha::bench::accuracy(y_true, y_pred)},
        {"macro_f1", cypha::bench::f1_macro(y_true, y_pred)},
        {"balanced_accuracy", cypha::bench::balanced_accuracy(y_true, y_pred)},
        {"mean_epistemic_var", mean_epi},
        {"expert_count", static_cast<int>(m.labels.size())},
    };
    if (!confidences.empty()) {
        out["mean_confidence"] = mean_conf;
        out["ece"] = cypha::bench::expected_calibration_error(confidences, correct);
    }
    if (!margins.empty()) {
        const cypha::bench::MarginDistribution md = cypha::bench::margin_distribution(margins);
        out["margin_mean"] = md.mean;
        out["margin_p50"] = md.p50;
        out["margin_p10"] = md.p10;
    }
    return out;
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
        const auto downsampled = downsample_nearest(img, 8, 8);
        const auto hog = enc.hog_features(downsampled, 4, 9);
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

// ``score_matrix_use_field``'s own ``kernel_mem``/``use_kernel_llr`` args are a no-op by default:
// it early-returns via ``rpsm_score_matrix_batched`` whenever ``CYPHA_USE_RPSM_LLR`` is unset (the
// documented default), *before* reaching its kernel-blend branch. Blend manually here instead --
// same ``(1-blend)*lin + blend*ker`` formula as ``classify_at_h``/``score_matrix_use_field``, applied
// on top of whatever linear (RPSM or legacy) scores the model actually produces -- so the D14 opt-in
// kernel path works regardless of the RPSM env toggle.
void kernel_blend_llr_batched(const cypha::CyphaInferModel& infer, const double* h, int n,
                              const cypha::KernelMemory* kernel_mem, double kernel_blend,
                              std::vector<double>& llr) {
    cypha::score_matrix_use_field(infer, h, n, llr);
    const int K = static_cast<int>(infer.labels.size());
    const int d = infer.d_latent;
    if (kernel_mem != nullptr && kernel_mem->n_basis() >= 4 && K > 0 && n > 0 && d > 0) {
        std::vector<double> kernel_scores(static_cast<std::size_t>(K));
        for (int i = 0; i < n; ++i) {
            kernel_mem->score_all(h + static_cast<std::size_t>(i) * static_cast<std::size_t>(d), infer.labels,
                                    kernel_scores);
            for (int k = 0; k < K; ++k) {
                const double lin = llr[static_cast<std::size_t>(i * K + k)];
                const double ker = kernel_scores[static_cast<std::size_t>(k)];
                llr[static_cast<std::size_t>(i * K + k)] = (1.0 - kernel_blend) * lin + kernel_blend * ker;
            }
        }
    }
}

void kernel_blend_llr(const cypha::CyphaInferModel& infer, const double* h,
                      const cypha::KernelMemory* kernel_mem, double kernel_blend, std::vector<double>& llr) {
    kernel_blend_llr_batched(infer, h, 1, kernel_mem, kernel_blend, llr);
}

// ``kernel_mem``/``use_kernel_llr``/``kernel_blend`` mirror the D03 kernel-LLR blend convention;
// default args (nullptr/false) reproduce the pre-existing linear-only routing decision exactly.
std::string pick_dif_regressor_expert(int step, int n_existing, int k_target, cypha::CyphaInferModel& infer,
                                      const double* x, [[maybe_unused]] int d, const cypha::KernelMemory* kernel_mem = nullptr,
                                      bool use_kernel_llr = false, double kernel_blend = 0.5) {
    if (n_existing < k_target && step <= k_target * 20) {
        return "_e" + std::to_string(step % k_target);
    }
    if (n_existing == 0) return "_e0";
    std::vector<double> llr;
    if (use_kernel_llr && kernel_mem != nullptr) {
        std::vector<double> h;
        cypha::batch_encode(infer, x, 1, h);
        kernel_blend_llr(infer, h.data(), kernel_mem, kernel_blend, llr);
    } else {
        cypha::batch_llr_from_x(infer, x, 1, llr);
    }
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
    // Opt-in kernelized (RFF, auto-gamma) expert-routing discriminant (D14 dedicated pass, mirrors
    // the D03 xor_kernel_bench/KernelMemory pattern). Null/false by default -- reproduces the
    // pre-existing plain linear `batch_llr_from_x` routing exactly when unset.
    std::unique_ptr<cypha::KernelMemory> kernel_mem;
    bool use_kernel_llr{false};
    double kernel_blend{0.5};
    double kernel_lr_scale{1.0};
    int kernel_calib_warmup_steps{0};
    int rff_dim{4096};
    double rff_gamma_scale{1.0};
    std::uint64_t kernel_seed{0};
    std::vector<std::vector<double>> kernel_recalib_x;
    bool kernel_recalib_done{false};
    // Stage-2 residual RFF head (Upgrade wave 2 fork 1b): online ridge on cos-RFF(x) → residual.
    bool use_residual_rff{false};
    int residual_rff_dim{256};
    double residual_rff_lr{0.02};
    std::vector<double> residual_rff_w;  // D×d row-major
    std::vector<double> residual_rff_b;  // D
    std::vector<double> residual_coef;   // D
    double residual_bias{0.0};
};

// Opt-in RFF auto-gamma kernel-LLR basis for the D14 Feynman-equations regression domain's
// expert-routing discriminant (deferred dedicated pass from the D03 kernel-LLR track --
// docs/research/upgrades/NONLINEAR_BOUNDARY.md Fix 2 / docs/RESEARCH_STATUS.md Priority 1). D14's
// regressor does not use `KernelMemory` at all on current HEAD: expert selection
// (`pick_dif_regressor_expert`) and the mixture-softmax weights (`online_reg_predict`) both route
// through a plain linear `batch_llr_from_x`/`score_matrix_use_field` discriminant over the discrete
// expert clusters; the final scalar prediction itself stays a linear per-expert running mean/var
// mixture (`regression::predict_mixture_scalar`) either way -- only the *routing* discriminant is
// kernelized here, exactly as deferred. Env-gated the same way as
// `CYPHA_D03_KERNEL_BASIS`/`CYPHA_D03_RFF_DIM`/`CYPHA_D03_RFF_GAMMA_SCALE`; unset reproduces the
// pre-existing D14 output byte-for-byte.
struct D14KernelConfig {
    bool enabled = false;
    int rff_dim = 4096;
    double rff_gamma_scale = 1.0;
    double kernel_blend = 0.5;
    double kernel_lr_scale = 1.0;
    int calib_warmup_steps = 0;
    bool residual_rff = false;
    int residual_rff_dim = 256;
};

D14KernelConfig d14_kernel_config_from_env() {
    D14KernelConfig cfg;
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D14_KERNEL_BASIS"); v.has_value() && *v == "rff") {
        cfg.enabled = true;
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D14_RFF_DIM"); v.has_value() && !v->empty()) {
        cfg.rff_dim = std::atoi(v->c_str());
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D14_RFF_GAMMA_SCALE"); v.has_value() && !v->empty()) {
        cfg.rff_gamma_scale = std::atof(v->c_str());
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D14_KERNEL_BLEND"); v.has_value() && !v->empty()) {
        cfg.kernel_blend = std::atof(v->c_str());
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D14_KERNEL_LR_SCALE"); v.has_value() && !v->empty()) {
        cfg.kernel_lr_scale = std::atof(v->c_str());
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D14_KERNEL_CALIB_WARMUP_STEPS"); v.has_value() &&
        !v->empty()) {
        cfg.calib_warmup_steps = std::atoi(v->c_str());
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D14_RESIDUAL_RFF"); v.has_value() &&
        (*v == "1" || *v == "true" || *v == "TRUE")) {
        cfg.residual_rff = true;
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D14_RESIDUAL_RFF_DIM"); v.has_value() &&
        !v->empty()) {
        cfg.residual_rff_dim = std::max(16, std::atoi(v->c_str()));
    }
    return cfg;
}

OnlineRegressor make_online_regressor(int input_dim, std::uint64_t seed, const cypha::bench::ProfileJson& regime,
                                      const D14KernelConfig* kernel_cfg = nullptr,
                                      const std::vector<std::vector<double>>* calib_x = nullptr) {
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

    if (kernel_cfg != nullptr && kernel_cfg->enabled) {
        const int kdim = r.infer.d_latent;
        std::vector<double> calib_rowmajor;
        if (calib_x != nullptr && kdim > 0) {
            const std::size_t n_calib = std::min<std::size_t>(calib_x->size(), 256);
            if (n_calib > 0) {
                std::vector<std::vector<double>> calib_slice;
                calib_slice.reserve(n_calib);
                for (std::size_t i = 0; i < n_calib; ++i) {
                    calib_slice.push_back((*calib_x)[i]);
                }
                const std::vector<double> calib_flat = flatten_rowmajor(calib_slice);
                cypha::batch_encode(r.infer, calib_flat.data(), static_cast<int>(n_calib), calib_rowmajor);
            }
        }
        const int n_calib_rows =
            kdim > 0 ? static_cast<int>(calib_rowmajor.size() / static_cast<std::size_t>(kdim)) : 0;
        const double gamma = cypha::KernelMemory::auto_gamma_median_heuristic(
            calib_rowmajor.data(), n_calib_rows, kdim, kernel_cfg->rff_gamma_scale, 256, seed);
        r.kernel_mem = std::make_unique<cypha::KernelMemory>(
            cypha::KernelMemory::make_rff(kdim, kernel_cfg->rff_dim, gamma, seed));
        r.use_kernel_llr = true;
        r.kernel_blend = kernel_cfg->kernel_blend;
        r.kernel_lr_scale = kernel_cfg->kernel_lr_scale;
        r.kernel_calib_warmup_steps = kernel_cfg->calib_warmup_steps;
        r.rff_dim = kernel_cfg->rff_dim;
        r.rff_gamma_scale = kernel_cfg->rff_gamma_scale;
        r.kernel_seed = seed;
    }
    if (kernel_cfg != nullptr && kernel_cfg->residual_rff) {
        const int d_in = input_dim;
        const int D = kernel_cfg->residual_rff_dim;
        r.use_residual_rff = true;
        r.residual_rff_dim = D;
        std::mt19937 rng(static_cast<std::uint32_t>(seed ^ 0x51FF1u));
        constexpr double kGamma = 1.0;
        cypha::init_rff_weights(cypha::RffProjectionKind::IidGaussian, rng, kGamma, D, d_in,
                                r.residual_rff_w, r.residual_rff_b, /*kernel_memory_scale=*/true);
        r.residual_coef.assign(static_cast<std::size_t>(D), 0.0);
        r.residual_bias = 0.0;
    }
    return r;
}

void residual_rff_features(const OnlineRegressor& r, const std::vector<double>& x, std::vector<double>& phi) {
    const int D = r.residual_rff_dim;
    const int d = static_cast<int>(x.size());
    phi.assign(static_cast<std::size_t>(D), 0.0);
    if (D <= 0 || r.residual_rff_w.empty() || r.residual_rff_b.size() != static_cast<std::size_t>(D)) {
        return;
    }
    const double scale = std::sqrt(2.0 / static_cast<double>(D));
    for (int i = 0; i < D; ++i) {
        double dot = r.residual_rff_b[static_cast<std::size_t>(i)];
        const double* row = r.residual_rff_w.data() + static_cast<std::size_t>(i * d);
        for (int j = 0; j < d; ++j) {
            dot += row[j] * x[static_cast<std::size_t>(j)];
        }
        phi[static_cast<std::size_t>(i)] = scale * std::cos(dot);
    }
}

double residual_rff_predict(const OnlineRegressor& r, const std::vector<double>& x) {
    if (!r.use_residual_rff || r.residual_coef.empty()) {
        return 0.0;
    }
    std::vector<double> phi;
    residual_rff_features(r, x, phi);
    double y = r.residual_bias;
    for (std::size_t i = 0; i < phi.size() && i < r.residual_coef.size(); ++i) {
        y += r.residual_coef[i] * phi[i];
    }
    return y;
}

void residual_rff_train(OnlineRegressor& r, const std::vector<double>& x, double residual) {
    if (!r.use_residual_rff || r.residual_coef.empty()) {
        return;
    }
    std::vector<double> phi;
    residual_rff_features(r, x, phi);
    double pred = r.residual_bias;
    for (std::size_t i = 0; i < phi.size(); ++i) {
        pred += r.residual_coef[i] * phi[i];
    }
    const double err = residual - pred;
    const double lr = r.residual_rff_lr;
    r.residual_bias += lr * err;
    for (std::size_t i = 0; i < phi.size(); ++i) {
        r.residual_coef[i] += lr * err * phi[i];
    }
}

void online_reg_predict_mixture(const OnlineRegressor& r, const std::vector<double>& x, double& y_hat,
                                double& unc);

void online_reg_train_step(OnlineRegressor& r, const std::vector<double>& x, double y) {
    if (r.use_kernel_llr && r.kernel_mem != nullptr && r.kernel_calib_warmup_steps > 0 &&
        !r.kernel_recalib_done && r.kernel_recalib_x.size() < 256) {
        r.kernel_recalib_x.push_back(x);
    }
    ++r.total_steps;
    const int d = static_cast<int>(x.size());
    const int k_target = std::max(r.n_experts_cap, 4);
    const int n_existing = static_cast<int>(r.mem.labels.size());
    const std::string expert = pick_dif_regressor_expert(r.total_steps, n_existing, k_target, r.infer, x.data(), d,
                                                          r.kernel_mem.get(), r.use_kernel_llr, r.kernel_blend);
    if (r.use_kernel_llr && r.kernel_mem != nullptr) {
        cypha::TrainStepExtras extras;
        extras.kernel_mem = r.kernel_mem.get();
        extras.use_kernel_llr = true;
        extras.kernel_blend = r.kernel_blend;
        extras.kernel_lr_scale = r.kernel_lr_scale;
        (void)cypha::dif_train_step_vector(r.infer, r.mem, r.replay, x.data(), d, expert, r.world_lr, r.delta_lr,
                                           r.world_lr, r.delta_lr, r.ood_sigma, r.tsp, r.rng, r.enc_updates, nullptr,
                                           &extras);
    } else {
        (void)cypha::dif_train_step_vector(r.infer, r.mem, r.replay, x.data(), d, expert, r.world_lr, r.delta_lr,
                                           r.world_lr, r.delta_lr, r.ood_sigma, r.tsp, r.rng, r.enc_updates, nullptr,
                                           nullptr);
    }
    auto& st = r.experts[expert];
    cypha::regression::expert_target_ema_step(st.mu, st.var_ema, st.n_updates, &y, 1, r.target_lr);

    if (r.use_kernel_llr && r.kernel_mem != nullptr && !r.kernel_recalib_done && r.kernel_calib_warmup_steps > 0 &&
        r.total_steps == r.kernel_calib_warmup_steps) {
        const int kdim = r.infer.d_latent;
        if (kdim > 0 && !r.kernel_recalib_x.empty()) {
            const std::size_t n_calib = std::min<std::size_t>(r.kernel_recalib_x.size(), 256);
            const std::vector<double> calib_flat = flatten_rowmajor(
                std::vector<std::vector<double>>(r.kernel_recalib_x.begin(),
                                                 r.kernel_recalib_x.begin() + static_cast<std::ptrdiff_t>(n_calib)));
            std::vector<double> calib_rowmajor;
            cypha::batch_encode(r.infer, calib_flat.data(), static_cast<int>(n_calib), calib_rowmajor);
            const int n_calib_rows =
                static_cast<int>(calib_rowmajor.size() / static_cast<std::size_t>(kdim));
            const double gamma = cypha::KernelMemory::auto_gamma_median_heuristic(
                calib_rowmajor.data(), n_calib_rows, kdim, r.rff_gamma_scale, 256, r.kernel_seed);
            r.kernel_mem = std::make_unique<cypha::KernelMemory>(
                cypha::KernelMemory::make_rff(kdim, r.rff_dim, gamma, r.kernel_seed));
        }
        r.kernel_recalib_done = true;
    }
    if (r.use_residual_rff) {
        double y_hat = 0.0;
        double unc = 0.0;
        online_reg_predict_mixture(r, x, y_hat, unc);
        residual_rff_train(r, x, y - y_hat);
    }
}

void online_reg_predict_mixture(const OnlineRegressor& r, const std::vector<double>& x, double& y_hat, double& unc) {
    std::vector<double> llr;
    if (r.use_kernel_llr && r.kernel_mem != nullptr) {
        std::vector<double> h;
        cypha::batch_encode(r.infer, x.data(), 1, h);
        kernel_blend_llr(r.infer, h.data(), r.kernel_mem.get(), r.kernel_blend, llr);
    } else {
        cypha::batch_llr_from_x(r.infer, x.data(), 1, llr);
    }
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

void online_reg_predict(const OnlineRegressor& r, const std::vector<double>& x, double& y_hat, double& unc) {
    online_reg_predict_mixture(r, x, y_hat, unc);
    if (r.use_residual_rff) {
        y_hat += residual_rff_predict(r, x);
    }
}

Json reg_metrics_native(const OnlineRegressor& r, const std::vector<std::vector<double>>& xs,
                        const std::vector<double>& ys) {
    const int n = static_cast<int>(xs.size());
    const int k = static_cast<int>(r.infer.labels.size());
    std::vector<double> preds;
    std::vector<double> unc;
    preds.reserve(xs.size());
    unc.reserve(xs.size());
    if (n > 0 && k > 0) {
        const std::vector<double> flat = flatten_rowmajor(xs);
        std::vector<double> llr;
        if (r.use_kernel_llr && r.kernel_mem != nullptr) {
            std::vector<double> h;
            cypha::batch_encode(r.infer, flat.data(), n, h);
            kernel_blend_llr_batched(r.infer, h.data(), n, r.kernel_mem.get(), r.kernel_blend, llr);
        } else {
            cypha::batch_llr_from_x(r.infer, flat.data(), n, llr);
        }
        std::vector<double> probs;
        cypha::softmax_batch_reference(llr.data(), n, k, 1e-12, probs);
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
        for (int i = 0; i < n; ++i) {
            double y_hat = 0.0;
            double u = 0.0;
            cypha::regression::predict_mixture_scalar(probs.data() + static_cast<std::size_t>(i * k), mu.data(),
                                                      var.data(), static_cast<std::size_t>(k), y_hat, u);
            if (r.use_residual_rff) {
                y_hat += residual_rff_predict(r, xs[static_cast<std::size_t>(i)]);
            }
            preds.push_back(y_hat);
            unc.push_back(u);
        }
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
        {"crps", cypha::bench::crps_gaussian_mean(ys, preds, unc)},
        {"interval_coverage_90", cypha::bench::predictive_interval_coverage(ys, preds, unc, 1.645)},
        {"residual_autocorr_lag1", cypha::bench::residual_autocorr_lag1(ys, preds)},
        {"residual_spectral_flatness", cypha::bench::residual_spectral_flatness(ys, preds)},
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
    const int test_n = static_cast<int>(x_test.size());
    const int k_experts = static_cast<int>(infer.labels.size());
    if (test_n > 0 && k_experts > 0) {
        const std::vector<double> flat = flatten_rowmajor(x_test);
        std::vector<double> llr;
        cypha::batch_llr_from_x(infer, flat.data(), test_n, llr);
        std::vector<double> probs;
        cypha::softmax_batch_reference(llr.data(), test_n, k_experts, 1e-12, probs);
        std::vector<double> mu;
        std::vector<double> var;
        mu.reserve(infer.labels.size());
        var.reserve(infer.labels.size());
        for (const auto& lbl : infer.labels) {
            const auto& st = experts[lbl];
            mu.push_back(st.mu.empty() ? 0.0 : st.mu[0]);
            var.push_back(st.var_ema);
        }
        for (int i = 0; i < test_n; ++i) {
            double y_hat = 0.0;
            double unc = 0.0;
            cypha::regression::predict_mixture_scalar(probs.data() + static_cast<std::size_t>(i * k_experts),
                                                      mu.data(), var.data(), mu.size(), y_hat, unc);
            preds.push_back(y_hat);
        }
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
              {"mean_expert_alpha", alpha_profile.value("mean_expert_alpha", 0.0)},
              {"fraction_edge_of_chaos", alpha_profile.value("fraction_near_edge_of_chaos", 0.0)},
              {"cfg_n_experts", alpha_profile.value("cfg_n_experts", 0)},
              {"max_experts", alpha_profile.value("max_experts", 0)},
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
              {"mean_expert_alpha", alpha_profile.value("mean_expert_alpha", 0.0)},
              {"fraction_edge_of_chaos", alpha_profile.value("fraction_near_edge_of_chaos", 0.0)},
              {"cfg_n_experts", alpha_profile.value("cfg_n_experts", 0)},
              {"max_experts", alpha_profile.value("max_experts", 0)},
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
    // Opt-in kernelized (RFF, auto-gamma) expert-routing discriminant for 14A only -- see
    // D14KernelConfig above. Unset (default) reproduces pre-existing 14A/14B/14C output exactly;
    // 14B/14C intentionally keep calling the 3-arg `make_online_regressor` overload (kernel path
    // stays off there regardless of the env gate) since only 14A carries the Ridge-comparison
    // framing this pass targets.
    const D14KernelConfig d14_kernel_cfg = d14_kernel_config_from_env();

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

        OnlineRegressor reg = make_online_regressor(eq.n_inputs, kBenchSeed, regime, &d14_kernel_cfg, &train_x);
        for (std::size_t i = 0; i < train_x.size(); ++i) {
            online_reg_train_step(reg, train_x[i], train_y[i]);
        }
        cypha::sync_infer_model_from_memory(reg.infer, reg.mem);
        Json m = reg_metrics_native(reg, test_x, test_y);
        m["ridge_rmse"] = cypha::bench::ridge_baseline(train_x, train_y, test_x, test_y).rmse;
        if (d14_kernel_cfg.enabled) {
            m["kernel_basis"] = "rff";
            m["rff_dim"] = d14_kernel_cfg.rff_dim;
        }
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

    Json exp14a{{"per_equation", per_equation}, {"mean_rmse", mean_rmse}, {"mean_r2", mean_r2}};
    exp14a["kernel_basis"] = d14_kernel_cfg.enabled ? "rff" : "linear";
    if (d14_kernel_cfg.enabled) {
        exp14a["rff_dim"] = d14_kernel_cfg.rff_dim;
        exp14a["rff_gamma_scale"] = d14_kernel_cfg.rff_gamma_scale;
        exp14a["kernel_blend"] = d14_kernel_cfg.kernel_blend;
    }
    exp14a["residual_rff"] = d14_kernel_cfg.residual_rff;
    if (d14_kernel_cfg.residual_rff) {
        exp14a["residual_rff_dim"] = d14_kernel_cfg.residual_rff_dim;
    }
    const Json experiments{
        {"14A_feynman_all_equations", exp14a},
        {"14B_extrapolation_uncertainty",
         Json{{"extrapolation_auroc", cypha::bench::safe_auroc(labels, scores)},
              {"regressor_uncertainty_auroc", cypha::bench::safe_auroc(reg_labels, reg_scores)}}},
        {"14C_noise_vs_aleatoric", noise_curve},
        {"backend", "cypha_core"},
    };
    cypha::bench::finalize_domain("d14", experiments);
    return experiments;
}

Json run_d15_fgsm_robustness_curve_impl(const std::vector<double>& epsilons, int max_eval,
                                        std::uint64_t seed) {
    if (epsilons.size() < 3) {
        throw std::runtime_error("FGSM robustness curve requires >=3 epsilon points");
    }
    const cypha::bench::ProfileJson profile = cypha::bench::load_profile();
    const cypha::bench::ProfileJson regime = cypha::bench::classification_params(&profile);
    const DigitDataset digits = load_digits_hog_dataset();
    if (digits.test_x.empty()) {
        throw std::runtime_error("digits HOG test split is empty");
    }

    OnlineClassifier clf =
        make_online_classifier(static_cast<int>(digits.train_x.front().size()), seed, regime.value("enc_lr", 0.002),
                               regime);
    train_classifier_online(clf, digits.train_x, digits.train_y, 4, seed);

    const int n = max_eval >= 0
                      ? std::min(max_eval, static_cast<int>(digits.test_x.size()))
                      : std::min(500, static_cast<int>(digits.test_x.size()));

    struct AdvSample {
        std::vector<double> x;
        std::string y;
        std::vector<int> sign;
    };
    std::vector<AdvSample> adv_samples;
    adv_samples.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto& x = digits.test_x[static_cast<std::size_t>(i)];
        const std::string& y = digits.test_y[static_cast<std::size_t>(i)];
        AdvSample sample{x, y, std::vector<int>(x.size(), 0)};
        for (std::size_t j = 0; j < x.size(); ++j) {
            std::vector<double> x_plus = x;
            std::vector<double> x_minus = x;
            x_plus[j] += 1e-4;
            x_minus[j] -= 1e-4;
            const std::string p_plus = online_clf_predict(clf.infer, x_plus);
            const std::string p_minus = online_clf_predict(clf.infer, x_minus);
            if (p_plus != y) {
                sample.sign[j] = 1;
            } else if (p_minus != y) {
                sample.sign[j] = -1;
            }
        }
        adv_samples.push_back(std::move(sample));
    }

    auto apply_fgsm = [](const AdvSample& sample, double epsilon) {
        std::vector<double> x_adv = sample.x;
        if (epsilon <= 0.0) {
            return x_adv;
        }
        for (std::size_t j = 0; j < x_adv.size(); ++j) {
            if (sample.sign[j] != 0) {
                x_adv[j] += static_cast<double>(sample.sign[j]) * epsilon;
                if (x_adv[j] < 0.0) {
                    x_adv[j] = 0.0;
                }
            }
        }
        return x_adv;
    };

    Json points = Json::array();
    for (double epsilon : epsilons) {
        int correct = 0;
        double epi_sum = 0.0;
        for (const AdvSample& sample : adv_samples) {
            const std::vector<double> x_adv = apply_fgsm(sample, epsilon);
            const std::string pred = online_clf_predict(clf.infer, x_adv);
            if (pred == sample.y) {
                ++correct;
            }
            epi_sum += online_clf_epistemic(clf.infer, x_adv);
        }
        points.push_back(Json{
            {"epsilon", epsilon},
            {"accuracy", static_cast<double>(correct) / static_cast<double>(n)},
            {"mean_epistemic", epi_sum / static_cast<double>(n)},
            {"n_eval", n},
        });
    }

    return Json{
        {"curve_id", "adversarial_robustness"},
        {"metric", "accuracy"},
        {"perturbation", "fgsm_proxy"},
        {"dataset", "digits_hog"},
        {"bench_seed", seed},
        {"n_eval", n},
        {"epsilons_requested", epsilons},
        {"points", points},
    };
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
        const std::vector<double> epsilons = {0.0, 0.05, 0.1, 0.2, 0.5};
        Json curve = run_d15_fgsm_robustness_curve_impl(epsilons, cypha::bench::bench_scale(500, 120),
                                                        kBenchSeed + 2);
        curve["accuracy_natural"] = curve["points"].front()["accuracy"];
        curve["mean_epistemic_natural"] = curve["points"].front()["mean_epistemic"];
        for (const auto& row : curve["points"]) {
            if (std::abs(row["epsilon"].get<double>() - 0.1) < 1e-9) {
                curve["accuracy_adversarial"] = row["accuracy"];
                curve["mean_epistemic_adversarial"] = row["mean_epistemic"];
                break;
            }
        }
        return curve;
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

// Opt-in: use EwcRegularizer::snapshot_calibrated (real diagonal Fisher from calibration-batch
// squared gradients) instead of the default snapshot() squared-anchor proxy. Off by default so
// existing D16B/D16H results are unchanged unless explicitly requested. See
// docs/reports/STUB_AUDIT_2026-07-11.md.
bool d16_real_fisher_enabled() {
    const std::optional<std::string> v = cypha::env_get("CYPHA_D16_REAL_FISHER");
    if (!v.has_value()) {
        return false;
    }
    const std::string& s = *v;
    return s == "1" || s == "true" || s == "True" || s == "yes";
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
    const bool task_sticky = []() {
        const std::optional<std::string> v = cypha::env_get("CYPHA_D16_TASK_STICKY");
        if (!v.has_value()) return false;
        return *v == "1" || *v == "true" || *v == "TRUE" || *v == "yes";
    }();
    auto train_task = [&](OnlineClassifier& clf, const MultitaskBundle& task, cypha::TrainStepExtras* extras) {
        if (task_sticky) {
            clf.mem.task_prefix_protect = task.name;
        } else {
            clf.mem.task_prefix_protect.clear();
        }
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
        clf.mem.task_prefix_protect.clear();
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
            if (d16_real_fisher_enabled()) {
                std::vector<std::vector<double>> calib_x;
                std::vector<std::string> calib_y;
                calib_x.reserve(iris->train_x.size());
                calib_y.reserve(iris->train_y.size());
                for (std::size_t i = 0; i < iris->train_x.size(); ++i) {
                    calib_x.push_back(pad_to_max(iris->train_x[i], max_dim));
                    calib_y.push_back(iris->name + "_" + iris->train_y[i]);
                }
                ewc.snapshot_calibrated(clf.mem, clf.infer, calib_x, calib_y);
            } else {
                ewc.snapshot(clf.mem, clf.infer);
            }
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
            row["ewc_real_fisher"] = d16_real_fisher_enabled();
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
        {"task_sticky", task_sticky},
    };
}

// D16B EWC strength sweep (docs/reports/EWC_D16B_SCOPING_2026-07-12.md): extends
// `run_d16_ewc_probe`'s baseline-vs-single-lambda probe into a real trade-off curve — several
// `ewc_lambda` settings, each with/without the NIG world-field (`world_mu`) protection, and now
// also reports task B's own post-training accuracy (wine, digits) alongside task A's
// before/after — not just a single forgetting_score number. Not part of `all_domains()` / the
// default bench report: called only from the standalone `ewc_d16b_sweep` tool
// (native/tools/ewc_d16b_sweep.cpp) so it never touches `bench/report/tables/` or
// `bench/BASELINE_REPORT.md`.
Json run_d16_ewc_sweep_impl() {
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

    auto run_setting = [&](double ewc_lambda, bool protect_world_field, std::uint64_t seed) {
        OnlineClassifier clf = make_multitask_clf(seed);
        cypha::EwcRegularizer ewc;
        cypha::TrainStepExtras extras{};
        const bool use_ewc = ewc_lambda > 0.0;
        if (use_ewc) {
            ewc.set_protect_world_field(protect_world_field);
            extras.ewc = &ewc;
            extras.ewc_lambda = ewc_lambda;
        }
        train_task(clf, *iris, use_ewc ? &extras : nullptr);
        if (use_ewc) {
            ewc.snapshot(clf.mem, clf.infer);
        }
        const double task_a_before = eval_task(clf, *iris);
        train_task(clf, *wine, use_ewc ? &extras : nullptr);
        const double task_b_wine_after = eval_task(clf, *wine);
        train_task(clf, *digits, use_ewc ? &extras : nullptr);
        cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
        const double task_a_after = eval_task(clf, *iris);
        const double task_b_digits_after = eval_task(clf, *digits);
        const double forgetting = (task_a_before - task_a_after) / std::max(task_a_before, 1e-6);
        Json row{
            {"ewc_lambda", ewc_lambda},
            {"protect_world_field", protect_world_field},
            {"task_a_accuracy_before", task_a_before},
            {"task_a_accuracy_after", task_a_after},
            {"forgetting_score", forgetting},
            {"task_b_wine_accuracy_after", task_b_wine_after},
            {"task_b_digits_accuracy_after", task_b_digits_after},
        };
        if (use_ewc) {
            row["ewc_penalty_final"] = ewc.penalty(clf.mem, clf.infer);
        }
        return row;
    };

    Json rows = Json::array();
    std::uint64_t seed = kBenchSeed + 100;
    rows.push_back(run_setting(0.0, false, seed++));  // baseline: EWC off entirely.
    const std::array<double, 3> lambdas{0.1, 0.5, 2.0};  // low / medium / high Fisher penalty weight.
    for (double lambda : lambdas) {
        rows.push_back(run_setting(lambda, false, seed++));  // classic EWC: D (class-delta prefix) + enc_w only.
        rows.push_back(run_setting(lambda, true, seed++));   // + NIG world-field (world_mu) protection.
    }
    return Json{
        {"rows", rows},
        {"n_train_steps_per_task", steps},
        {"note",
         "Standalone sweep, not part of the default bench report -- see native/tools/ewc_d16b_sweep.cpp and "
         "docs/reports/EWC_D16B_SCOPING_2026-07-12.md"},
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

    // 16G view streams (round_robin vs task_block_shuffle forgetting probe).
    // task_block_shuffle is a negative control (FAST mean acc ~0.58 vs RR ~0.81; wine/digits collapse).
    // Do not enable in production profiles — see docs/reports/D16_MULTIVIEW_POLICY_2026-07-17.md §2.
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

    // 16I DIF-V3 replay-interleave: round-robin stream with elevated replay_ratio.
    // Index-reorder (16G) is a closed negative; this probes the existing ReplayBuffer path.
    Json exp16i;
    {
        const int max_steps = cypha::bench::bench_scale(3000, 1500);
        const int warm_steps = std::max(1, max_steps / 6);
        auto forgetting_with_replay = [&](double replay_ratio, std::uint64_t seed) {
            OnlineClassifier clf = make_multitask_clf(seed);
            clf.tsp.replay_ratio = replay_ratio;
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
            std::unordered_map<std::string, std::size_t> cursors;
            while (step < max_steps - warm_steps) {
                for (const auto& t : tasks) {
                    if (step >= max_steps - warm_steps) break;
                    const std::size_t i = cursors[t.name]++ % t.train_x.size();
                    const auto x = pad_to_max(t.train_x[i], max_dim);
                    (void)online_clf_train_step(clf, x, t.name + "_" + t.train_y[i]);
                    ++step;
                }
            }
            cypha::sync_infer_model_from_memory(clf.infer, clf.mem);
            const double acc_after = eval_task(clf, *iris);
            Json per_task = Json::object();
            double mean_acc = 0.0;
            for (const auto& t : tasks) {
                const double a = eval_task(clf, t);
                per_task[t.name] = a;
                mean_acc += a;
            }
            mean_acc /= static_cast<double>(std::max<std::size_t>(1, tasks.size()));
            return Json{
                {"replay_ratio", replay_ratio},
                {"task_a_accuracy_before", acc_before},
                {"task_a_accuracy_after", acc_after},
                {"forgetting_score", (acc_before - acc_after) / std::max(acc_before, 1e-6)},
                {"mean_task_accuracy", mean_acc},
                {"per_task_accuracy", per_task},
            };
        };
        const Json rr0 = forgetting_with_replay(0.0, kBenchSeed + 50);
        const Json rr22 = forgetting_with_replay(0.22, kBenchSeed + 51);
        const Json rr50 = forgetting_with_replay(0.5, kBenchSeed + 52);
        exp16i = Json{
            {"max_steps", max_steps},
            {"replay_ratio_0", rr0},
            {"replay_ratio_0_22", rr22},
            {"replay_ratio_0_50", rr50},
            {"forgetting_delta_0_22",
             rr22["forgetting_score"].get<double>() - rr0["forgetting_score"].get<double>()},
            {"forgetting_delta_0_50",
             rr50["forgetting_score"].get<double>() - rr0["forgetting_score"].get<double>()},
            {"note", "DIF-V3 replay-interleave; negative delta = reduced forgetting vs RR"},
        };
    }

    const Json experiments{
        {"16A_task_discovery", exp16a},
        {"16B_forgetting_resistance", exp16b},
        {"16D_interleaving_comparison", exp16d},
        {"16E_save_restore", exp16e},
        {"16F_per_task_models", exp16f},
        {"16G_view_streams", exp16g},
        {"16H_ewc_overlay", exp16h},
        {"16I_replay_interleave", exp16i},
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
    std::vector<double> epistemic;
    Json scores = clf_metrics_native(clf.infer, test_x, test_y, &epistemic);
    std::vector<double> boundary_dist;
    boundary_dist.reserve(test_x.size());
    for (const auto& xrow : test_x) {
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

// Opt-in RFF auto-gamma kernel-LLR basis for the real D03 XOR bench domain (Fix 2 in
// docs/research/upgrades/NONLINEAR_BOUNDARY.md; sweep in docs/RESEARCH_STATUS.md Priority 1). Follows
// the same D03 env-var opt-in convention as `d03_view_schedule_from_env()`/`CYPHA_D03_VIEW_SCHEDULE`
// above: unset/"nystrom" reproduces the pre-existing `--kernel-xor-features` Nystrom call byte-for-byte
// (confirmed via rerun); "rff" swaps in `xor_kernel_bench --kernel-basis rff` (KernelMemory::make_rff +
// auto_gamma_median_heuristic) on the same subprocess call instead of a second, parallel config system.
struct D03KernelExperimentConfig {
    std::string basis = "nystrom";
    std::string feature_mode = "xor_pair";
    int rff_dim = 4096;
    double rff_gamma_scale = 1.0;
    std::string nystrom_landmark_sampling = "uniform";
    std::string rff_projection = "iid";
};

D03KernelExperimentConfig d03_kernel_experiment_from_env() {
    D03KernelExperimentConfig cfg;
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D03_KERNEL_BASIS"); v.has_value() && !v->empty()) {
        cfg.basis = *v;
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D03_KERNEL_FEATURE_MODE"); v.has_value() && !v->empty()) {
        cfg.feature_mode = *v;
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D03_RFF_DIM"); v.has_value() && !v->empty()) {
        cfg.rff_dim = std::atoi(v->c_str());
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D03_RFF_GAMMA_SCALE"); v.has_value() && !v->empty()) {
        cfg.rff_gamma_scale = std::atof(v->c_str());
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D03_NYSTROM_LANDMARK_SAMPLING");
        v.has_value() && !v->empty()) {
        cfg.nystrom_landmark_sampling = *v;
    }
    if (const std::optional<std::string> v = cypha::env_get("CYPHA_D03_RFF_PROJECTION"); v.has_value() && !v->empty()) {
        cfg.rff_projection = *v;
    }
    return cfg;
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
    const D03KernelExperimentConfig kernel_cfg = d03_kernel_experiment_from_env();
    std::ostringstream cmd;
    cmd << "\"" << bench_exe.string() << "\""
        << " --seeds " << seeds << " --passes " << passes << " --kernel-blend 1.0"
        << " --kernel-m 512 --gamma-scale 2.0 --kernel-lr-scale 2.0"
        << " --kernel-feature-mode " << kernel_cfg.feature_mode;
    if (kernel_cfg.basis == "rff") {
        cmd << " --kernel-basis rff --rff-dim " << kernel_cfg.rff_dim
            << " --rff-gamma-scale " << kernel_cfg.rff_gamma_scale;
        if (kernel_cfg.rff_projection == "sorf") {
            cmd << " --rff-projection sorf";
        }
    } else if (kernel_cfg.nystrom_landmark_sampling == "leverage") {
        cmd << " --nystrom-landmark-sampling leverage";
    }
    const Json j = Json::parse(capture_process_output(cmd.str()));
    Json kernel_result{
        {"accuracy", j.at("kernel_mean_acc")},
        {"delta_pp", j.at("delta_pp")},
        {"kernel_basis", j.value("kernel_basis", "nystrom")},
        {"kernel_m", j.value("kernel_m", 512)},
        {"kernel_feature_mode", j.value("kernel_feature_mode", "xor_pair")},
        {"backend", "xor_kernel_bench_native"},
    };
    if (kernel_cfg.basis == "rff") {
        kernel_result["rff_dim"] = j.value("rff_dim", kernel_cfg.rff_dim);
        kernel_result["rff_gamma_scale"] = j.value("rff_gamma_scale", kernel_cfg.rff_gamma_scale);
        kernel_result["rff_projection"] = j.value("rff_projection", kernel_cfg.rff_projection);
    } else {
        kernel_result["nystrom_landmark_sampling"] =
            j.value("nystrom_landmark_sampling", kernel_cfg.nystrom_landmark_sampling);
    }
    Json experiments{
        {"S3_xor_linear",
         Json{{"accuracy", j.at("linear_mean_acc")}, {"seeds", seeds}, {"passes", passes}}},
        {"S3_xor_kernel_llr", kernel_result},
    };
    if (kernel_cfg.basis == "rff") {
        experiments["kernel_basis"] = "rff";
    }
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

Json parse_subprocess_json_stdout(const std::string& text);
fs::path resolve_native_exe_dir();

std::optional<double> variant_row_kappa(const Json& row) {
    if (!row.contains("kappa") || row["kappa"].is_null() || !row["kappa"].is_number()) {
        return std::nullopt;
    }
    const double value = row["kappa"].get<double>();
    if (!std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

std::optional<double> variant_row_bpc(const Json& row) {
    if (!row.contains("bpc") || row["bpc"].is_null() || !row["bpc"].is_number()) {
        return std::nullopt;
    }
    const double value = row["bpc"].get<double>();
    if (!std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

Json build_pareto_ranked_variants(const Json& rows, double w = 0.1) {
    struct RankedEntry {
        std::string id;
        double kappa;
        double bpc;
        double normalized_bpc;
        double pareto_score;
        bool nondominated;
    };
    std::vector<RankedEntry> ranked;
    ranked.reserve(rows.size());
    double min_bpc = std::numeric_limits<double>::infinity();
    double max_bpc = -std::numeric_limits<double>::infinity();
    for (const auto& row : rows) {
        if (!row.is_object()) {
            continue;
        }
        const auto kappa = variant_row_kappa(row);
        const auto bpc = variant_row_bpc(row);
        if (!kappa.has_value() || !bpc.has_value()) {
            continue;
        }
        min_bpc = std::min(min_bpc, *bpc);
        max_bpc = std::max(max_bpc, *bpc);
        ranked.push_back({row.value("id", ""), *kappa, *bpc, 0.0, 0.0, false});
    }
    const double bpc_span = max_bpc - min_bpc;
    for (auto& entry : ranked) {
        entry.normalized_bpc =
            bpc_span > 0.0 ? (entry.bpc - min_bpc) / bpc_span : 0.0;
        entry.pareto_score = entry.kappa - w * entry.normalized_bpc;
    }
    for (auto& entry : ranked) {
        bool dominated = false;
        for (const auto& other : ranked) {
            if (other.id == entry.id) {
                continue;
            }
            if (other.kappa >= entry.kappa && other.bpc <= entry.bpc &&
                (other.kappa > entry.kappa || other.bpc < entry.bpc)) {
                dominated = true;
                break;
            }
        }
        entry.nondominated = !dominated;
    }
    std::sort(ranked.begin(), ranked.end(), [](const RankedEntry& a, const RankedEntry& b) {
        if (a.nondominated != b.nondominated) {
            return a.nondominated > b.nondominated;
        }
        return a.pareto_score > b.pareto_score;
    });
    Json out = Json::array();
    for (const auto& entry : ranked) {
        out.push_back(Json{{"id", entry.id},
                           {"kappa", entry.kappa},
                           {"bpc", entry.bpc},
                           {"normalized_bpc", entry.normalized_bpc},
                           {"pareto_score", entry.pareto_score},
                           {"nondominated", entry.nondominated}});
    }
    return out;
}

Json build_kappa_ranked_variants(const Json& rows) {
    struct RankedEntry {
        std::string id;
        double kappa;
    };
    std::vector<RankedEntry> ranked;
    ranked.reserve(rows.size());
    for (const auto& row : rows) {
        if (!row.is_object()) {
            continue;
        }
        const auto kappa = variant_row_kappa(row);
        if (!kappa.has_value()) {
            continue;
        }
        ranked.push_back({row.value("id", ""), *kappa});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const RankedEntry& a, const RankedEntry& b) { return a.kappa > b.kappa; });
    Json out = Json::array();
    for (const auto& entry : ranked) {
        out.push_back(Json{{"id", entry.id}, {"kappa", entry.kappa}});
    }
    return out;
}

Json run_d20_cell_hypothesis_overnight_smoke() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path sweep_exe =
        cypha::bench::resolve_runner_exe("cypha_cell_hypothesis_sweep", exe_dir);
    if (!fs::is_regular_file(sweep_exe)) {
        throw std::runtime_error("missing cypha_cell_hypothesis_sweep: " + sweep_exe.string());
    }

    const cypha::bench::RunProcessResult proc = cypha::bench::run_executable_capture(
        sweep_exe, {"--overnight-sweep-smoke", "--intelligence-profile"});
    if (proc.exit_code != 0) {
        throw std::runtime_error("cypha_cell_hypothesis_sweep --overnight-sweep-smoke exit=" +
                                 std::to_string(proc.exit_code));
    }

    const Json sweep_out = parse_subprocess_json_stdout(proc.stdout_text);
    if (sweep_out.empty()) {
        throw std::runtime_error("cypha_cell_hypothesis_sweep produced no JSON stdout");
    }

    Json rows = Json::array();
    if (sweep_out.contains("results") && sweep_out["results"].is_array()) {
        for (const auto& row : sweep_out["results"]) {
            Json slim = Json{{"id", row.value("id", "")},
                             {"bench_mode", row.value("bench_mode", "")},
                             {"n_train", row.value("n_train", 0)},
                             {"bpc", row.contains("bpc") ? row["bpc"] : Json(nullptr)}};
            const auto kappa = variant_row_kappa(row);
            if (kappa.has_value()) {
                slim["kappa"] = *kappa;
            } else {
                slim["kappa"] = nullptr;
            }
            rows.push_back(std::move(slim));
        }
    }

    const Json kappa_ranked = build_kappa_ranked_variants(rows);
    const Json pareto_ranked = build_pareto_ranked_variants(rows);
    const Json experiments{
        {"overnight_sweep_smoke", rows},
        {"kappa_ranked_variants", kappa_ranked},
        {"pareto_ranked_variants", pareto_ranked},
        {"variant_count", rows.size()},
        {"n_train", sweep_out.value("n_train", 0)},
        {"n_eval", sweep_out.value("n_eval", 0)},
        {"intelligence_profile", sweep_out.value("intelligence_profile", true)},
        {"backend", "cypha_cell_hypothesis_sweep --overnight-sweep-smoke --intelligence-profile"},
    };
    cypha::bench::finalize_domain("d20_cell_hypothesis_overnight", experiments);
    return experiments;
}

Json run_d22_intelligence_cross_profile() {
    const Json d18 = run_d18_intelligence_profile();
    const Json d16_ewc = run_d16_ewc_probe();
    const Json d20 = run_d20_cell_hypothesis_overnight_smoke();

    Json kappa_ranked = Json::array();
    if (d20.contains("kappa_ranked_variants") && d20["kappa_ranked_variants"].is_array()) {
        kappa_ranked = d20["kappa_ranked_variants"];
    }
    Json pareto_ranked = Json::array();
    if (d20.contains("pareto_ranked_variants") && d20["pareto_ranked_variants"].is_array()) {
        pareto_ranked = d20["pareto_ranked_variants"];
    }
    Json best_pareto_variant = nullptr;
    if (!pareto_ranked.empty()) {
        best_pareto_variant = pareto_ranked.front();
    }

    const Json experiments{
        {"d18_intelligence_profile", d18},
        {"d16_ewc_probe", d16_ewc},
        {"d20_cell_sweep_smoke", d20},
        {"kappa_ranked_variants", kappa_ranked},
        {"pareto_ranked_variants", pareto_ranked},
        {"best_pareto_variant", best_pareto_variant},
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
    if (const std::optional<std::string> raw = cypha::env_get("CYPHA_NATIVE_EXE_DIR"); raw.has_value() && !raw->empty()) {
        const fs::path env_path = fs::absolute(*raw);
        if (fs::is_regular_file(env_path / "cyphalm_bench_native.exe") ||
            fs::is_regular_file(env_path / "cypha_bench_run.exe")) {
            return env_path;
        }
    }
#if defined(_WIN32)
    char module_path[MAX_PATH];
    if (GetModuleFileNameA(nullptr, module_path, MAX_PATH) != 0) {
        const fs::path self_dir = fs::path(module_path).parent_path();
        if (fs::is_regular_file(self_dir / "cyphalm_bench_native.exe") ||
            fs::is_regular_file(self_dir / "cypha_bench_run.exe")) {
            return fs::absolute(self_dir);
        }
    }
#endif
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
constexpr int kExpectedCellSweepVariants = 25;

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

std::string normalize_path_slashes(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    return path;
}

bool is_legacy_repo_root_results_artifact_path(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    const std::string norm = normalize_path_slashes(path);
    if (norm.find("bench/results") != std::string::npos) {
        return false;
    }
    return norm.size() >= 8 && norm.compare(norm.size() - 8, 8, "/results") == 0;
}

bool lock_references_legacy_summary_csv(const Json& lock) {
    const std::string dump = normalize_path_slashes(lock.dump());
    const std::string needle = "/results/summary.csv";
    std::size_t pos = 0;
    while ((pos = dump.find(needle, pos)) != std::string::npos) {
        if (pos < 6 || dump.compare(pos - 6, 6, "bench/") != 0) {
            return true;
        }
        pos += needle.size();
    }
    return false;
}

bool is_repo_root_smoke_leak_filename(const std::string& name) {
    if (name.size() < 14 || name[0] != 'd' || name[3] != '_') {
        return false;
    }
    if (name[1] < '0' || name[1] > '9' || name[2] < '0' || name[2] > '9') {
        return false;
    }
    if (name.compare(name.size() - 5, 5, ".json") != 0) {
        return false;
    }
    const std::string suffix_smoke = "_smoke.json";
    if (name.size() > 4 + suffix_smoke.size() &&
        name.compare(name.size() - suffix_smoke.size(), suffix_smoke.size(), suffix_smoke) == 0) {
        return true;
    }
    const std::string suffix_smoke_alt = "smoke.json";
    return name.size() > 4 + suffix_smoke_alt.size() &&
           name.compare(name.size() - suffix_smoke_alt.size(), suffix_smoke_alt.size(),
                        suffix_smoke_alt) == 0;
}

std::vector<std::string> scan_legacy_repo_root_cell_sweep_artifacts(const fs::path& results_dir) {
    std::vector<std::string> artifacts;
    if (!fs::is_directory(results_dir)) {
        return artifacts;
    }
    if (fs::is_regular_file(results_dir / "summary.csv")) {
        artifacts.push_back("results/summary.csv");
    }
    if (fs::is_regular_file(results_dir / "manifest.json")) {
        artifacts.push_back("results/manifest.json");
    }
    for (const auto& entry : fs::directory_iterator(results_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.size() >= 13 && name.compare(0, 8, "variant_") == 0 &&
            name.compare(name.size() - 5, 5, ".json") == 0) {
            artifacts.push_back("results/" + name);
        }
    }
    std::sort(artifacts.begin(), artifacts.end());
    return artifacts;
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

Json run_d29_release_readiness_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path lock_path = fs::current_path() / "d29_release_readiness_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);
    const std::string production_status = validate_production_tier_lock(lock);
    const std::string overnight_complete_status = validate_overnight_complete_lock(lock);

    const fs::path repo = cypha::bench::bench_root().parent_path();
    const std::array<fs::path, 3> required_files{
        repo / "scripts" / "finalize_production_overnight.ps1",
        repo / "scripts" / "run_production_overnight.ps1",
        repo / "bench" / "results" / ".gitkeep",
    };
    Json release_files = Json::object();
    for (const fs::path& path : required_files) {
        const bool present = fs::is_regular_file(path);
        release_files[path.lexically_relative(repo).generic_string()] = present;
        if (!present) {
            throw std::runtime_error("release readiness file missing: " + path.string());
        }
    }

    Json baseline_lock_validate_report{{"invoked", false},
                                       {"reason", "baseline_lock_validate not built"}};
    const fs::path validate_exe = cypha::bench::resolve_runner_exe("baseline_lock_validate", exe_dir);
    if (fs::is_regular_file(validate_exe)) {
        const cypha::bench::RunProcessResult proc = cypha::bench::run_executable_capture(
            validate_exe,
            {"--lock-file", fs::absolute(lock_path).string(), "--production"});
        baseline_lock_validate_report = Json{{"invoked", true},
                                             {"exit_code", proc.exit_code},
                                             {"stdout", proc.stdout_text},
                                             {"stderr", proc.stderr_text}};
        if (proc.exit_code != 0) {
            throw std::runtime_error("baseline_lock_validate exit=" + std::to_string(proc.exit_code));
        }
    }

    const bool release_ready = production_status == "production_validated" &&
                               overnight_complete_status == "overnight_complete_validated";
    const std::string validation_status = release_ready ? "release_ready" : "pending_release";

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
        {"overnight_complete_status", overnight_complete_status},
        {"release_files", release_files},
        {"baseline_lock_validate", baseline_lock_validate_report},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_pin_bpc", kD17HybridPinBpc},
        {"production_pin_tolerance", kD17ProductionPinTolerance},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d29_release_readiness_validation", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d29_release_readiness_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d30_artifact_hygiene_validation() {
    const fs::path lock_path = fs::current_path() / "d30_artifact_hygiene_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);

    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path gitkeep_path = repo / "bench" / "results" / ".gitkeep";
    const bool gitkeep_present = fs::is_regular_file(gitkeep_path);
    if (!gitkeep_present) {
        throw std::runtime_error("bench/results/.gitkeep missing");
    }

    std::vector<std::string> warnings;
    std::string cell_sweep_artifact_path;
    bool legacy_artifact = false;

    if (lock.contains("cell_sweep_results") && lock["cell_sweep_results"].is_object()) {
        const Json& cell_sweep = lock["cell_sweep_results"];
        if (cell_sweep.contains("artifact_path") && !cell_sweep["artifact_path"].is_null()) {
            cell_sweep_artifact_path = cell_sweep["artifact_path"].get<std::string>();
            if (!cell_sweep_artifact_path.empty()) {
                if (is_legacy_repo_root_results_artifact_path(cell_sweep_artifact_path)) {
                    legacy_artifact = true;
                    warnings.push_back(
                        "cell_sweep_results.artifact_path uses repo-root /results; prefer "
                        "bench/results or null/empty");
                }
            }
        }
    }

    const bool legacy_summary_csv = lock_references_legacy_summary_csv(lock);
    if (legacy_summary_csv) {
        warnings.push_back(
            "lock references repo-root results/summary.csv (informational; migrate to "
            "bench/results)");
    }

    const std::string validation_status =
        legacy_artifact ? "legacy_artifact_path" : "hygiene_ok";

    const Json experiments{
        {"lock_file", lock_path.string()},
        {"cell_sweep_results", lock.contains("cell_sweep_results") ? lock["cell_sweep_results"] : Json{}},
        {"cell_sweep_artifact_path",
         cell_sweep_artifact_path.empty() ? Json(nullptr) : Json(cell_sweep_artifact_path)},
        {"bench_results_gitkeep", gitkeep_present},
        {"legacy_summary_csv_referenced", legacy_summary_csv},
        {"validation_status", validation_status},
        {"warnings", warnings},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d30_artifact_hygiene_validation", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d30_artifact_hygiene_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d31_post_overnight_pipeline_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();

    const std::array<const char*, 4> required_scripts{
        "scripts/poll_and_finalize_overnight.ps1",
        "scripts/finalize_production_overnight.ps1",
        "scripts/commit_production_lock.ps1",
        "scripts/migrate_legacy_results.ps1",
    };
    Json pipeline_scripts = Json::object();
    for (const char* rel : required_scripts) {
        const fs::path path = repo / rel;
        const bool present = fs::is_regular_file(path);
        pipeline_scripts[rel] = present;
        if (!present) {
            throw std::runtime_error("pipeline script missing: " + path.string());
        }
    }

    Json validation_chain = Json::object();
    validation_chain["d27"] = run_d27_production_lock_validation();
    validation_chain["d28"] = run_d28_overnight_complete_validation();
    validation_chain["d29"] = run_d29_release_readiness_validation();
    validation_chain["d30"] = run_d30_artifact_hygiene_validation();

    const Json experiments{
        {"pipeline_scripts", pipeline_scripts},
        {"validation_chain", validation_chain},
        {"validation_status", "pipeline_ok"},
        {"backend", "post_overnight_pipeline"},
    };
    cypha::bench::finalize_domain("d31_post_overnight_pipeline_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d31_post_overnight_pipeline_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d32_production_complete_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path script_path = repo / "scripts" / "validate_production_complete.ps1";
    const bool script_present = fs::is_regular_file(script_path);
    if (!script_present) {
        throw std::runtime_error("pipeline script missing: " + script_path.string());
    }

    const fs::path lock_path = fs::current_path() / "d32_production_complete_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);

    require_lock_key(lock["overnight_results"], "n_train", "overnight_results");
    if (!lock["overnight_results"]["n_train"].is_number_integer()) {
        throw std::runtime_error("overnight_results n_train must be an integer");
    }
    const int n_train = lock["overnight_results"]["n_train"].get<int>();

    std::string production_status;
    std::string overnight_complete_status;
    std::string validation_status;

    if (n_train < kProductionNTrainMin) {
        validation_status = "pending_production_complete";
        production_status = "pending_production";
        overnight_complete_status = "pending_overnight_complete";
    } else {
        production_status = validate_production_tier_lock(lock);
        overnight_complete_status = validate_overnight_complete_lock(lock);
        validation_status = (production_status == "production_validated" &&
                             overnight_complete_status == "overnight_complete_validated")
                                ? "production_complete_validated"
                                : "pending_production_complete";
    }

    const Json experiments{
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"rpsm_results", lock["rpsm_results"]},
        {"cell_sweep_results", lock.contains("cell_sweep_results") ? lock["cell_sweep_results"] : Json{}},
        {"d17_hybrid_baseline", lock["d17_hybrid_baseline"]},
        {"n_train", n_train},
        {"validation_status", validation_status},
        {"production_status", production_status},
        {"overnight_complete_status", overnight_complete_status},
        {"validate_production_complete_script", script_path.lexically_relative(repo).generic_string()},
        {"validate_production_complete_script_present", script_present},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_pin_bpc", kD17HybridPinBpc},
        {"production_pin_tolerance", kD17ProductionPinTolerance},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d32_production_complete_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d32_production_complete_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

bool gh_cli_present() {
#ifdef _WIN32
    return std::system("gh --version >NUL 2>&1") == 0;
#else
    return std::system("gh --version >/dev/null 2>&1") == 0;
#endif
}

bool script_text_contains(const fs::path& script_path, const std::string& needle) {
    if (!fs::is_regular_file(script_path)) {
        return false;
    }
    std::ifstream in(script_path);
    if (!in) {
        return false;
    }
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return content.find(needle) != std::string::npos;
}

bool publish_script_has_gh_auth_preflight(const fs::path& publish_script) {
    return script_text_contains(publish_script, "gh auth");
}

Json commit_script_has_dryrun_and_force(const fs::path& commit_script) {
    Json flags{{"DryRun", false}, {"Force", false}};
    if (!fs::is_regular_file(commit_script)) {
        return flags;
    }
    std::ifstream in(commit_script);
    if (!in) {
        return flags;
    }
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    flags["DryRun"] = content.find("-DryRun") != std::string::npos;
    flags["Force"] = content.find("-Force") != std::string::npos;
    return flags;
}

Json run_d33_release_publish_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();

    const std::array<const char*, 4> required_scripts{
        "scripts/publish_release.ps1",
        "scripts/create_release_notes.ps1",
        "scripts/validate_production_complete.ps1",
        "scripts/commit_production_lock.ps1",
    };
    Json publish_scripts = Json::object();
    for (const char* rel : required_scripts) {
        const fs::path path = repo / rel;
        const bool present = fs::is_regular_file(path);
        publish_scripts[rel] = present;
        if (!present) {
            throw std::runtime_error("publish script missing: " + path.string());
        }
    }

    const fs::path publish_script_path = repo / "scripts" / "publish_release.ps1";
    const bool gh_publish_script_present = fs::is_regular_file(publish_script_path);
    const bool gh_auth_required =
        gh_publish_script_present && publish_script_has_gh_auth_preflight(publish_script_path);

    const fs::path lock_path = fs::current_path() / "d33_release_publish_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);

    require_lock_key(lock["overnight_results"], "n_train", "overnight_results");
    if (!lock["overnight_results"]["n_train"].is_number_integer()) {
        throw std::runtime_error("overnight_results n_train must be an integer");
    }
    const int n_train = lock["overnight_results"]["n_train"].get<int>();

    std::string production_status;
    std::string overnight_complete_status;
    std::string validation_status;

    if (n_train < kProductionNTrainMin) {
        validation_status = "pending_release_publish";
        production_status = "pending_production";
        overnight_complete_status = "pending_overnight_complete";
    } else {
        production_status = validate_production_tier_lock(lock);
        overnight_complete_status = validate_overnight_complete_lock(lock);
        validation_status = (production_status == "production_validated" &&
                             overnight_complete_status == "overnight_complete_validated")
                                ? "release_publish_ready"
                                : "pending_release_publish";
    }

    const Json experiments{
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"rpsm_results", lock["rpsm_results"]},
        {"cell_sweep_results", lock.contains("cell_sweep_results") ? lock["cell_sweep_results"] : Json{}},
        {"d17_hybrid_baseline", lock["d17_hybrid_baseline"]},
        {"n_train", n_train},
        {"validation_status", validation_status},
        {"production_status", production_status},
        {"overnight_complete_status", overnight_complete_status},
        {"publish_scripts", publish_scripts},
        {"gh_cli_present", gh_cli_present()},
        {"gh_publish_script_present", gh_publish_script_present},
        {"gh_auth_required", gh_auth_required},
        {"gh_auth_note",
         "scripts/publish_release.ps1 runs gh auth status preflight before gh release create"},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_pin_bpc", kD17HybridPinBpc},
        {"production_pin_tolerance", kD17ProductionPinTolerance},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d33_release_publish_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d33_release_publish_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d34_repo_smoke_hygiene_validation() {
    const fs::path lock_path = fs::current_path() / "d34_repo_smoke_hygiene_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const fs::path repo = cypha::bench::bench_root().parent_path();
    std::vector<std::string> leaked_smoke_files;
    if (fs::is_directory(repo)) {
        for (const auto& entry : fs::directory_iterator(repo)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (is_repo_root_smoke_leak_filename(name)) {
                leaked_smoke_files.push_back(name);
            }
        }
    }
    std::sort(leaked_smoke_files.begin(), leaked_smoke_files.end());

    const fs::path legacy_results_dir = repo / "results";
    const std::vector<std::string> legacy_cell_sweep_artifacts =
        scan_legacy_repo_root_cell_sweep_artifacts(legacy_results_dir);

    std::vector<std::string> warnings;
    if (!leaked_smoke_files.empty()) {
        warnings.push_back(
            "repo root contains leaked dNN smoke JSON files (cypha_bench_run copied lock to cwd); "
            "remove or run scripts/cleanup_repo_smoke_artifacts.ps1");
    }
    if (!legacy_cell_sweep_artifacts.empty()) {
        warnings.push_back(
            "repo-root results/ contains legacy cell-sweep artifacts; migrate via "
            "scripts/migrate_legacy_results.ps1");
    }

    const bool has_leaks =
        !leaked_smoke_files.empty() || !legacy_cell_sweep_artifacts.empty();
    const std::string validation_status =
        has_leaks ? "repo_root_smoke_leak" : "repo_root_smoke_ok";

    const fs::path cleanup_script = repo / "scripts" / "cleanup_repo_smoke_artifacts.ps1";
    const bool script_present = fs::is_regular_file(cleanup_script);

    const Json experiments{
        {"lock_file", lock_path.string()},
        {"repo_root", repo.string()},
        {"leaked_smoke_files", leaked_smoke_files},
        {"legacy_results_dir_present", fs::is_directory(legacy_results_dir)},
        {"legacy_cell_sweep_artifacts", legacy_cell_sweep_artifacts},
        {"cleanup_repo_smoke_artifacts_script",
         cleanup_script.lexically_relative(repo).generic_string()},
        {"cleanup_repo_smoke_artifacts_script_present", script_present},
        {"validation_status", validation_status},
        {"warnings", warnings},
        {"backend", "repo_root_scan"},
    };
    cypha::bench::finalize_domain("d34_repo_smoke_hygiene_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d34_repo_smoke_hygiene_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d35_lock_commit_pipeline_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();

    const std::array<const char*, 4> required_scripts{
        "scripts/commit_production_lock.ps1",
        "scripts/finalize_production_overnight.ps1",
        "scripts/poll_and_finalize_overnight.ps1",
        "scripts/validate_production_complete.ps1",
    };
    Json commit_scripts = Json::object();
    for (const char* rel : required_scripts) {
        const fs::path path = repo / rel;
        const bool present = fs::is_regular_file(path);
        commit_scripts[rel] = present;
        if (!present) {
            throw std::runtime_error("commit pipeline script missing: " + path.string());
        }
    }

    const fs::path commit_script_path = repo / "scripts" / "commit_production_lock.ps1";
    const Json dryrun_force_flags = commit_script_has_dryrun_and_force(commit_script_path);

    const fs::path lock_path = fs::current_path() / "d35_lock_commit_pipeline_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);

    require_lock_key(lock["overnight_results"], "n_train", "overnight_results");
    if (!lock["overnight_results"]["n_train"].is_number_integer()) {
        throw std::runtime_error("overnight_results n_train must be an integer");
    }
    const int n_train = lock["overnight_results"]["n_train"].get<int>();

    std::string production_status;
    std::string overnight_complete_status;
    std::string validation_status;

    if (n_train < kProductionNTrainMin) {
        validation_status = "pending_lock_commit";
        production_status = "pending_production";
        overnight_complete_status = "pending_overnight_complete";
    } else {
        production_status = validate_production_tier_lock(lock);
        overnight_complete_status = validate_overnight_complete_lock(lock);
        validation_status = (production_status == "production_validated" &&
                             overnight_complete_status == "overnight_complete_validated")
                                ? "lock_commit_ready"
                                : "pending_lock_commit";
    }

    const Json experiments{
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"rpsm_results", lock["rpsm_results"]},
        {"cell_sweep_results", lock.contains("cell_sweep_results") ? lock["cell_sweep_results"] : Json{}},
        {"d17_hybrid_baseline", lock["d17_hybrid_baseline"]},
        {"n_train", n_train},
        {"validation_status", validation_status},
        {"production_status", production_status},
        {"overnight_complete_status", overnight_complete_status},
        {"commit_scripts", commit_scripts},
        {"commit_script_has_dryrun_and_force", dryrun_force_flags},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_pin_bpc", kD17HybridPinBpc},
        {"production_pin_tolerance", kD17ProductionPinTolerance},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d35_lock_commit_pipeline_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d35_lock_commit_pipeline_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d36_pipeline_e2e_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path config = cypha::bench::config_dir();

    const std::array<const char*, 9> required_scripts{
        "scripts/verify_production_pipeline.ps1",
        "scripts/run_production_overnight.ps1",
        "scripts/update_baseline_lock.ps1",
        "scripts/poll_and_finalize_overnight.ps1",
        "scripts/finalize_production_overnight.ps1",
        "scripts/commit_production_lock.ps1",
        "scripts/validate_production_complete.ps1",
        "scripts/verify_release_publish.ps1",
        "scripts/cleanup_repo_smoke_artifacts.ps1",
    };
    Json pipeline_scripts = Json::object();
    for (const char* rel : required_scripts) {
        const fs::path path = repo / rel;
        const bool present = fs::is_regular_file(path);
        pipeline_scripts[rel] = present;
        if (!present) {
            throw std::runtime_error("pipeline E2E script missing: " + path.string());
        }
    }

    const std::array<const char*, 9> required_domain_profiles{
        "d27_production_lock_profile.json",
        "d28_overnight_complete_profile.json",
        "d29_release_readiness_profile.json",
        "d30_artifact_hygiene_profile.json",
        "d31_post_overnight_pipeline_profile.json",
        "d32_production_complete_profile.json",
        "d33_release_publish_profile.json",
        "d34_repo_smoke_hygiene_profile.json",
        "d35_lock_commit_pipeline_profile.json",
    };
    Json bench_profiles = Json::object();
    for (const char* rel : required_domain_profiles) {
        const fs::path path = config / rel;
        const bool present = fs::is_regular_file(path);
        bench_profiles[rel] = present;
        if (!present) {
            throw std::runtime_error("bench profile missing: " + path.string());
        }
    }

    const fs::path lock_path = fs::current_path() / "d36_pipeline_e2e_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);

    require_lock_key(lock["overnight_results"], "n_train", "overnight_results");
    if (!lock["overnight_results"]["n_train"].is_number_integer()) {
        throw std::runtime_error("overnight_results n_train must be an integer");
    }
    const int n_train = lock["overnight_results"]["n_train"].get<int>();

    std::string production_status;
    std::string overnight_complete_status;
    std::string validation_status;

    if (n_train < kProductionNTrainMin) {
        validation_status = "pending_pipeline_e2e";
        production_status = "pending_production";
        overnight_complete_status = "pending_overnight_complete";
    } else {
        production_status = validate_production_tier_lock(lock);
        overnight_complete_status = validate_overnight_complete_lock(lock);
        validation_status = (production_status == "production_validated" &&
                             overnight_complete_status == "overnight_complete_validated")
                                ? "pipeline_e2e_ready"
                                : "pending_pipeline_e2e";
    }

    const Json experiments{
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"rpsm_results", lock["rpsm_results"]},
        {"cell_sweep_results", lock.contains("cell_sweep_results") ? lock["cell_sweep_results"] : Json{}},
        {"d17_hybrid_baseline", lock["d17_hybrid_baseline"]},
        {"n_train", n_train},
        {"validation_status", validation_status},
        {"production_status", production_status},
        {"overnight_complete_status", overnight_complete_status},
        {"pipeline_scripts", pipeline_scripts},
        {"bench_profiles", bench_profiles},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_pin_bpc", kD17HybridPinBpc},
        {"production_pin_tolerance", kD17ProductionPinTolerance},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d36_pipeline_e2e_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d36_pipeline_e2e_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d37_lock_refresh_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();

    const std::array<const char*, 3> required_scripts{
        "scripts/update_baseline_lock.ps1",
        "scripts/migrate_legacy_results.ps1",
        "scripts/finalize_production_overnight.ps1",
    };
    Json lock_refresh_scripts = Json::object();
    for (const char* rel : required_scripts) {
        const fs::path path = repo / rel;
        const bool present = fs::is_regular_file(path);
        lock_refresh_scripts[rel] = present;
        if (!present) {
            throw std::runtime_error("lock refresh script missing: " + path.string());
        }
    }

    const fs::path inflight_path = repo / "scripts" / "migrate_inflight_overnight_artifacts.ps1";
    lock_refresh_scripts["scripts/migrate_inflight_overnight_artifacts.ps1"] =
        fs::is_regular_file(inflight_path);

    const fs::path update_lock_path = repo / "scripts" / "update_baseline_lock.ps1";
    const bool update_lock_has_production_switch =
        script_text_contains(update_lock_path, "-Production");
    if (!update_lock_has_production_switch) {
        throw std::runtime_error("update_baseline_lock.ps1 missing -Production switch");
    }

    const fs::path lock_path = fs::current_path() / "d37_lock_refresh_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);

    require_lock_key(lock["overnight_results"], "n_train", "overnight_results");
    if (!lock["overnight_results"]["n_train"].is_number_integer()) {
        throw std::runtime_error("overnight_results n_train must be an integer");
    }
    const int n_train = lock["overnight_results"]["n_train"].get<int>();

    std::vector<std::string> warnings;
    std::string cell_sweep_artifact_path;
    if (lock.contains("cell_sweep_results") && lock["cell_sweep_results"].is_object()) {
        const Json& cell_sweep = lock["cell_sweep_results"];
        if (cell_sweep.contains("artifact_path") && !cell_sweep["artifact_path"].is_null()) {
            cell_sweep_artifact_path = cell_sweep["artifact_path"].get<std::string>();
            if (!cell_sweep_artifact_path.empty() &&
                is_legacy_repo_root_results_artifact_path(cell_sweep_artifact_path)) {
                warnings.push_back(
                    "cell_sweep_results.artifact_path uses repo-root /results; prefer "
                    "bench/results or null/empty");
            }
        }
    }

    std::string production_status;
    std::string overnight_complete_status;
    std::string validation_status;

    if (n_train < kProductionNTrainMin) {
        validation_status = "pending_lock_refresh";
        production_status = "pending_production";
        overnight_complete_status = "pending_overnight_complete";
    } else {
        production_status = validate_production_tier_lock(lock);
        overnight_complete_status = validate_overnight_complete_lock(lock);
        validation_status = (production_status == "production_validated" &&
                             overnight_complete_status == "overnight_complete_validated")
                                ? "lock_refresh_ready"
                                : "pending_lock_refresh";
    }

    const Json experiments{
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"rpsm_results", lock["rpsm_results"]},
        {"cell_sweep_results", lock.contains("cell_sweep_results") ? lock["cell_sweep_results"] : Json{}},
        {"d17_hybrid_baseline", lock["d17_hybrid_baseline"]},
        {"n_train", n_train},
        {"validation_status", validation_status},
        {"production_status", production_status},
        {"overnight_complete_status", overnight_complete_status},
        {"lock_refresh_scripts", lock_refresh_scripts},
        {"update_baseline_lock_has_production_switch", update_lock_has_production_switch},
        {"cell_sweep_artifact_path",
         cell_sweep_artifact_path.empty() ? Json(nullptr) : Json(cell_sweep_artifact_path)},
        {"warnings", warnings},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_pin_bpc", kD17HybridPinBpc},
        {"production_pin_tolerance", kD17ProductionPinTolerance},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d37_lock_refresh_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d37_lock_refresh_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d38_overnight_certificate_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();

    const std::array<const char*, 2> required_scripts{
        "scripts/run_post_overnight.ps1",
        "scripts/validate_production_complete.ps1",
    };
    Json certificate_scripts = Json::object();
    for (const char* rel : required_scripts) {
        const fs::path path = repo / rel;
        const bool present = fs::is_regular_file(path);
        certificate_scripts[rel] = present;
        if (!present) {
            throw std::runtime_error("overnight certificate script missing: " + path.string());
        }
    }

    const fs::path lock_path = fs::current_path() / "d38_overnight_certificate_smoke.json";
    if (!fs::exists(lock_path)) {
        fs::copy_file(cypha::bench::bench_root() / "BASELINE_LOCK.json", lock_path,
                      fs::copy_options::overwrite_existing);
    }

    const Json lock = load_json_file(lock_path);
    validate_baseline_lock_schema(lock);

    require_lock_key(lock["overnight_results"], "n_train", "overnight_results");
    if (!lock["overnight_results"]["n_train"].is_number_integer()) {
        throw std::runtime_error("overnight_results n_train must be an integer");
    }
    const int n_train = lock["overnight_results"]["n_train"].get<int>();

    std::vector<std::string> warnings;
    int variant_count = -1;
    bool variant_count_sufficient = false;
    if (lock.contains("cell_sweep_results") && lock["cell_sweep_results"].is_object()) {
        const Json& cell_sweep = lock["cell_sweep_results"];
        if (cell_sweep.contains("variant_count") && cell_sweep["variant_count"].is_number_integer()) {
            variant_count = cell_sweep["variant_count"].get<int>();
            variant_count_sufficient = variant_count >= kExpectedCellSweepVariants;
        } else {
            warnings.push_back("cell_sweep_results missing variant_count key");
        }
    } else {
        warnings.push_back("cell_sweep_results missing variant_count key");
    }

    std::string production_status;
    std::string overnight_complete_status;
    std::string validation_status;

    if (n_train < kProductionNTrainMin) {
        validation_status = "pending_overnight_certificate";
        production_status = "pending_production";
        overnight_complete_status = "pending_overnight_complete";
    } else {
        static constexpr const char* kSections[] = {"overnight_results", "rpsm_results",
                                                    "cell_sweep_results"};
        int ref_n_eval = -1;
        for (const char* name : kSections) {
            if (!lock.contains(name) || !lock[name].is_object()) {
                throw std::runtime_error(std::string("lock JSON missing ") + name);
            }
            const Json& section = lock[name];
            require_lock_key(section, "n_train", name);
            if (!section["n_train"].is_number_integer()) {
                throw std::runtime_error(std::string(name) + " n_train must be an integer");
            }
            if (section["n_train"].get<int>() < kProductionNTrainMin) {
                throw std::runtime_error(std::string(name) + " n_train below production minimum");
            }
            require_lock_key(section, "n_eval", name);
            if (!section["n_eval"].is_number_integer()) {
                throw std::runtime_error(std::string(name) + " n_eval must be an integer");
            }
            const int section_n_eval = section["n_eval"].get<int>();
            if (ref_n_eval < 0) {
                ref_n_eval = section_n_eval;
            } else if (section_n_eval != ref_n_eval) {
                throw std::runtime_error("overnight lock sections have mismatched n_eval");
            }
            if (section["n_train"].get<int>() != n_train) {
                throw std::runtime_error("overnight lock sections have mismatched n_train");
            }
        }

        production_status = validate_production_tier_lock(lock);
        overnight_complete_status = validate_overnight_complete_lock(lock);
        const bool tiers_ready = production_status == "production_validated" &&
                                 overnight_complete_status == "overnight_complete_validated";
        validation_status = (tiers_ready && variant_count_sufficient)
                                ? "overnight_certificate_ready"
                                : "pending_overnight_certificate";
    }

    const Json experiments{
        {"lock_file", lock_path.string()},
        {"overnight_results", lock["overnight_results"]},
        {"rpsm_results", lock["rpsm_results"]},
        {"cell_sweep_results", lock.contains("cell_sweep_results") ? lock["cell_sweep_results"] : Json{}},
        {"d17_hybrid_baseline", lock["d17_hybrid_baseline"]},
        {"n_train", n_train},
        {"validation_status", validation_status},
        {"production_status", production_status},
        {"overnight_complete_status", overnight_complete_status},
        {"certificate_scripts", certificate_scripts},
        {"expected_cell_sweep_variants", kExpectedCellSweepVariants},
        {"variant_count", variant_count >= 0 ? Json(variant_count) : Json(nullptr)},
        {"variant_count_sufficient", variant_count_sufficient},
        {"warnings", warnings},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_pin_bpc", kD17HybridPinBpc},
        {"production_pin_tolerance", kD17ProductionPinTolerance},
        {"backend", "baseline_lock_validate"},
    };
    cypha::bench::finalize_domain("d38_overnight_certificate_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d38_overnight_certificate_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

constexpr int kD39SmokeNTrain = 80;
constexpr int kD39SmokeNEval = 32;
constexpr int kExpectedProfileStatistics = 7;

bool json_point_in_unit_interval(const Json& v) {
    if (!v.is_number()) {
        return false;
    }
    const double x = v.get<double>();
    return std::isfinite(x) && x >= 0.0 && x <= 1.0;
}

bool intelligence_profile_statistics_complete(const Json& intelligence_profile) {
    if (!intelligence_profile.contains("statistics") ||
        !intelligence_profile["statistics"].is_array()) {
        return false;
    }
    const Json& stats = intelligence_profile["statistics"];
    if (stats.size() != kExpectedProfileStatistics) {
        return false;
    }
    for (const auto& stat : stats) {
        if (!stat.contains("point") || !json_point_in_unit_interval(stat["point"])) {
            return false;
        }
    }
    return true;
}

bool profile_completeness_json_complete(const Json& completeness) {
    return completeness.is_object() && completeness.value("all_complete", false);
}

bool parse_bench_intelligence_profile_complete(const Json& bench_output) {
    if (bench_output.contains("profile_completeness") &&
        profile_completeness_json_complete(bench_output["profile_completeness"])) {
        return true;
    }
    if (!bench_output.contains("intelligence_profile") ||
        !bench_output["intelligence_profile"].is_object()) {
        return false;
    }
    const Json& intelligence_profile = bench_output["intelligence_profile"];
    if (intelligence_profile.contains("profile_completeness") &&
        profile_completeness_json_complete(intelligence_profile["profile_completeness"])) {
        return true;
    }
    return intelligence_profile_statistics_complete(intelligence_profile);
}

Json parse_subprocess_json_stdout(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return Json::object();
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    const std::string trimmed = text.substr(begin, end - begin + 1);
    try {
        return Json::parse(trimmed);
    } catch (const std::exception&) {
        return Json::object();
    }
}

Json run_d39_intelligence_monitor_profile_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";

    const std::array<const char*, 4> required_sources{
        "include/cypha/cyphalm/lm_intelligence_monitor.hpp",
        "src/cyphalm/lm_intelligence_monitor.cpp",
        "include/cypha/intelligence/profile_completeness.hpp",
        "src/intelligence/profile_completeness.cpp",
    };
    Json monitor_sources = Json::object();
    for (const char* rel : required_sources) {
        const fs::path path = native_root / rel;
        const bool present = fs::is_regular_file(path);
        monitor_sources[std::string("native/") + rel] = present;
        if (!present) {
            throw std::runtime_error("intelligence monitor source missing: " + path.string());
        }
    }

    const fs::path hook_hpp = native_root / "include/cypha/cyphalm/cyphalm_intelligence_hook.hpp";
    const fs::path hook_cpp = native_root / "src/cyphalm/cyphalm_intelligence_hook.cpp";
    const bool hook_extended =
        fs::is_regular_file(hook_hpp) && fs::is_regular_file(hook_cpp) &&
        script_text_contains(hook_hpp, "export_intelligence_monitor_report") &&
        script_text_contains(hook_cpp, "lm_intelligence_monitor.hpp");
    monitor_sources["native/cyphalm_intelligence_hook_extended"] = hook_extended;

    std::vector<std::string> warnings;
    const bool lm_monitor_ctest_referenced =
        script_text_contains(native_root / "CMakeLists.txt", "native_intelligence_lm_monitor_smoke");
    if (!lm_monitor_ctest_referenced) {
        warnings.push_back(
            "native/CMakeLists.txt missing native_intelligence_lm_monitor_smoke CTest reference");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const cypha::bench::RunProcessResult proc = cypha::bench::run_executable_capture(
        bench_native_exe,
        {"--profile", "d17", "--mode", "hybrid", "--n-train", std::to_string(kD39SmokeNTrain),
         "--n-eval", std::to_string(kD39SmokeNEval), "--intelligence-profile"});

    Json bench_output = Json::object();
    bool stdout_parsed = false;
    if (!proc.stdout_text.empty()) {
        bench_output = parse_subprocess_json_stdout(proc.stdout_text);
        stdout_parsed = !bench_output.empty();
    }
    if (proc.exit_code == 0 && !stdout_parsed) {
        throw std::runtime_error("cyphalm_bench_native --intelligence-profile produced no JSON stdout");
    }
    if (proc.exit_code != 0) {
        throw std::runtime_error("cyphalm_bench_native --intelligence-profile exit=" +
                                 std::to_string(proc.exit_code));
    }

    const bool profile_complete = stdout_parsed && parse_bench_intelligence_profile_complete(bench_output);
    const std::string validation_status =
        profile_complete ? "profile_monitor_ready" : "pending_profile_monitor";

    const Json experiments{
        {"monitor_sources", monitor_sources},
        {"lm_monitor_ctest_referenced", lm_monitor_ctest_referenced},
        {"cyphalm_bench_native",
         Json{{"exit_code", proc.exit_code},
              {"stdout_parsed", stdout_parsed},
              {"profile_complete", profile_complete}}},
        {"n_train", kD39SmokeNTrain},
        {"n_eval", kD39SmokeNEval},
        {"expected_profile_statistics", kExpectedProfileStatistics},
        {"validation_status", validation_status},
        {"warnings", warnings},
        {"backend", "cyphalm_bench_native --intelligence-profile"},
    };
    cypha::bench::finalize_domain("d39_intelligence_monitor_profile_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d39_intelligence_monitor_profile_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

constexpr int kD40SmokeNTrain = 100;
constexpr int kD40SmokeNEval = 40;

bool json_bpc_finite(const Json& bench_output) {
    if (!bench_output.contains("bpc") || bench_output["bpc"].is_null()) {
        return false;
    }
    if (!bench_output["bpc"].is_number()) {
        return false;
    }
    return std::isfinite(bench_output["bpc"].get<double>());
}

bool profile_guided_loss_math_ready(const fs::path& pgl_hpp) {
    if (!fs::is_regular_file(pgl_hpp)) {
        return false;
    }
    const bool has_navigation =
        script_text_contains(pgl_hpp, "navigation_loss_total");
    const bool has_seven_stat_lambdas =
        script_text_contains(pgl_hpp, "lambda_alpha") &&
        script_text_contains(pgl_hpp, "lambda_calibration");
    return has_navigation || has_seven_stat_lambdas;
}

Json run_d40_math_integration_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";

    const std::array<const char*, 2> required_sources{
        "include/cypha/cyphalm/cyphalm_math_integration.hpp",
        "src/cyphalm/cyphalm_math_integration.cpp",
    };
    Json math_sources = Json::object();
    for (const char* rel : required_sources) {
        const fs::path path = native_root / rel;
        const bool present = fs::is_regular_file(path);
        math_sources[std::string("native/") + rel] = present;
        if (!present) {
            throw std::runtime_error("math integration source missing: " + path.string());
        }
    }

    const fs::path pgl_hpp = native_root / "include/cypha/intelligence/profile_guided_loss.hpp";
    const fs::path pgl_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    const bool pgl_present = fs::is_regular_file(pgl_hpp) && fs::is_regular_file(pgl_cpp);
    const bool pgl_math_ready = pgl_present && profile_guided_loss_math_ready(pgl_hpp);
    math_sources["native/profile_guided_loss_present"] = pgl_present;
    math_sources["native/profile_guided_loss_math_ready"] = pgl_math_ready;
    if (!pgl_present) {
        throw std::runtime_error("profile_guided_loss sources missing under native/");
    }
    if (!pgl_math_ready) {
        throw std::runtime_error(
            "profile_guided_loss.hpp missing navigation_loss_total or 7-stat lambdas");
    }

    std::vector<std::string> warnings;

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const cypha::bench::RunProcessResult proc = cypha::bench::run_executable_capture(
        bench_native_exe,
        {"--profile", "d17", "--mode", "hybrid", "--math-integration", "--intelligence-profile",
         "--n-train", std::to_string(kD40SmokeNTrain), "--n-eval", std::to_string(kD40SmokeNEval)});

    Json bench_output = Json::object();
    bool stdout_parsed = false;
    if (!proc.stdout_text.empty()) {
        bench_output = parse_subprocess_json_stdout(proc.stdout_text);
        stdout_parsed = !bench_output.empty();
    }
    if (proc.exit_code == 0 && !stdout_parsed) {
        throw std::runtime_error(
            "cyphalm_bench_native --math-integration produced no JSON stdout");
    }
    if (proc.exit_code != 0) {
        throw std::runtime_error("cyphalm_bench_native --math-integration exit=" +
                                 std::to_string(proc.exit_code));
    }

    const bool profile_complete = stdout_parsed && parse_bench_intelligence_profile_complete(bench_output);
    const bool bpc_finite = stdout_parsed && json_bpc_finite(bench_output);
    const bool math_ready = profile_complete && bpc_finite;
    const std::string validation_status =
        math_ready ? "math_integration_ready" : "pending_math_integration";

    const Json experiments{
        {"math_sources", math_sources},
        {"cyphalm_bench_native",
         Json{{"exit_code", proc.exit_code},
              {"stdout_parsed", stdout_parsed},
              {"profile_complete", profile_complete},
              {"bpc_finite", bpc_finite}}},
        {"n_train", kD40SmokeNTrain},
        {"n_eval", kD40SmokeNEval},
        {"validation_status", validation_status},
        {"warnings", warnings},
        {"backend", "cyphalm_bench_native --math-integration --intelligence-profile"},
    };
    cypha::bench::finalize_domain("d40_math_integration_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d40_math_integration_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

constexpr int kD41ScaleNTrain = 5000;
constexpr int kD41ScaleNEval = 256;
constexpr int kMathIntegrationBenchSeed = 42;
constexpr double kD50LockBpcTolerance = 0.025;
constexpr double kD50LockKappaTolerance = 0.03;

void apply_math_integration_subprocess_env(int n_train) {
#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
    _putenv_s("CYPHA_BENCH_FULL_CORPUS", "1");
    _putenv_s("CYPHA_BENCH_OVERNIGHT", "1");
    _putenv_s("CYPHA_BENCH_FULL_N_TRAIN", std::to_string(n_train).c_str());
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
    setenv("CYPHA_BENCH_FULL_CORPUS", "1", 1);
    setenv("CYPHA_BENCH_OVERNIGHT", "1", 1);
    setenv("CYPHA_BENCH_FULL_N_TRAIN", std::to_string(n_train).c_str(), 1);
#endif
}

std::optional<double> json_bpc_value(const Json& bench_output) {
    if (!bench_output.contains("bpc") || bench_output["bpc"].is_null() ||
        !bench_output["bpc"].is_number()) {
        return std::nullopt;
    }
    const double value = bench_output["bpc"].get<double>();
    if (!std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

std::optional<double> json_kappa_value(const Json& bench_output) {
    if (!bench_output.contains("intelligence_profile") ||
        !bench_output["intelligence_profile"].is_object()) {
        return std::nullopt;
    }
    const Json& intelligence_profile = bench_output["intelligence_profile"];
    if (!intelligence_profile.contains("criticality_score") ||
        !intelligence_profile["criticality_score"].is_number()) {
        return std::nullopt;
    }
    const double value = intelligence_profile["criticality_score"].get<double>();
    if (!std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

Json run_math_integration_bench_subprocess(const fs::path& bench_native_exe, int n_train,
                                           int n_eval, bool math_integration,
                                           const char* label, double per_stat_span = -1.0,
                                           double kappa_target = -1.0,
                                           double kappa_ceiling_strength = -1.0,
                                           double kappa_ceiling_min_scale = -1.0,
                                           std::int64_t bench_seed = -1,
                                           bool use_eigenvalue_d_eff = false,
                                           bool use_reu_forget_gate = false,
                                           double kappa_nav_warmup_strength = -1.0,
                                           double kappa_nav_warmup_floor = -1.0,
                                           bool disable_kappa_nav_warmup = false,
                                           double kappa_kernel_blend_floor = -1.0,
                                           bool disable_kappa_kernel_blend_scale = false,
                                           double kappa_excess_grad_margin = -1.0,
                                           double kappa_excess_grad_scale = -1.0,
                                           bool disable_kappa_excess_grad_nudge = false,
                                           double reu_forget_gate_blend = -1.0,
                                           int kappa_trajectory_window = -1,
                                           int navigation_loss_warmup_steps = -1,
                                           double free_energy_beta = -1.0,
                                           double kernel_blend = -1.0,
                                           int kernel_m = -1,
                                           bool hybrid_blend_logit_set = false,
                                           double hybrid_blend_logit = 0.0,
                                           double mdl_forget_max_norm = -1.0,
                                           double kernel_lr_scale = -1.0,
                                           double alpha_init = -1.0,
                                           double hybrid_blend_lr = -1.0,
                                           int n_experts = -1,
                                           int max_memory_slots = -1,
                                           int compress_interval = -1) {
    apply_math_integration_subprocess_env(n_train);
    std::vector<std::string> args{"--profile", "d17", "--mode", "hybrid", "--intelligence-profile",
                                  "--n-train", std::to_string(n_train), "--n-eval",
                                  std::to_string(n_eval)};
    if (bench_seed >= 0) {
        args.push_back("--bench-seed");
        args.push_back(std::to_string(bench_seed));
    }
    if (math_integration) {
        args.push_back("--math-integration");
        if (per_stat_span > 0.0) {
            args.push_back("--per-stat-deviation-span");
            args.push_back(std::to_string(per_stat_span));
        }
        if (kappa_target > 0.0) {
            args.push_back("--kappa-lambda-target");
            args.push_back(std::to_string(kappa_target));
        }
        if (kappa_ceiling_strength > 0.0) {
            args.push_back("--kappa-ceiling-strength");
            args.push_back(std::to_string(kappa_ceiling_strength));
        }
        if (kappa_ceiling_min_scale > 0.0) {
            args.push_back("--kappa-ceiling-min-scale");
            args.push_back(std::to_string(kappa_ceiling_min_scale));
        }
        if (use_eigenvalue_d_eff) {
            args.push_back("--use-eigenvalue-d-eff");
        }
        if (use_reu_forget_gate) {
            args.push_back("--use-reu-forget-gate");
        }
        if (disable_kappa_nav_warmup) {
            args.push_back("--disable-kappa-navigation-warmup");
        }
        if (kappa_nav_warmup_strength > 0.0) {
            args.push_back("--kappa-navigation-warmup-strength");
            args.push_back(std::to_string(kappa_nav_warmup_strength));
        }
        if (kappa_nav_warmup_floor > 0.0) {
            args.push_back("--kappa-navigation-warmup-floor");
            args.push_back(std::to_string(kappa_nav_warmup_floor));
        }
        if (disable_kappa_kernel_blend_scale) {
            args.push_back("--disable-kappa-kernel-blend-scale");
        }
        if (kappa_kernel_blend_floor > 0.0) {
            args.push_back("--kappa-kernel-blend-floor");
            args.push_back(std::to_string(kappa_kernel_blend_floor));
        }
        if (disable_kappa_excess_grad_nudge) {
            args.push_back("--disable-kappa-excess-grad-nudge");
        }
        if (kappa_excess_grad_scale > 0.0) {
            args.push_back("--kappa-excess-grad-scale");
            args.push_back(std::to_string(kappa_excess_grad_scale));
        }
        if (kappa_excess_grad_margin >= 0.0) {
            args.push_back("--kappa-excess-grad-margin");
            args.push_back(std::to_string(kappa_excess_grad_margin));
        }
        if (reu_forget_gate_blend >= 0.0) {
            args.push_back("--reu-forget-gate-blend");
            args.push_back(std::to_string(reu_forget_gate_blend));
        }
        if (kappa_trajectory_window > 0) {
            args.push_back("--kappa-trajectory-window");
            args.push_back(std::to_string(kappa_trajectory_window));
        }
        if (navigation_loss_warmup_steps >= 0) {
            args.push_back("--navigation-loss-warmup-steps");
            args.push_back(std::to_string(navigation_loss_warmup_steps));
        }
        if (free_energy_beta > 0.0) {
            args.push_back("--free-energy-beta");
            args.push_back(std::to_string(free_energy_beta));
        }
        if (kernel_blend > 0.0) {
            args.push_back("--kernel-blend");
            args.push_back(std::to_string(kernel_blend));
        }
        if (kernel_m > 0) {
            args.push_back("--kernel-m");
            args.push_back(std::to_string(kernel_m));
        }
        if (hybrid_blend_logit_set) {
            args.push_back("--hybrid-blend-logit");
            args.push_back(std::to_string(hybrid_blend_logit));
        }
        if (mdl_forget_max_norm > 0.0) {
            args.push_back("--mdl-forget-max-norm");
            args.push_back(std::to_string(mdl_forget_max_norm));
        }
        if (kernel_lr_scale > 0.0) {
            args.push_back("--kernel-lr-scale");
            args.push_back(std::to_string(kernel_lr_scale));
        }
        if (alpha_init > 0.0) {
            args.push_back("--alpha-init");
            args.push_back(std::to_string(alpha_init));
        }
        if (hybrid_blend_lr > 0.0) {
            args.push_back("--hybrid-blend-lr");
            args.push_back(std::to_string(hybrid_blend_lr));
        }
        if (n_experts > 0) {
            args.push_back("--n-experts");
            args.push_back(std::to_string(n_experts));
        }
        if (max_memory_slots > 0) {
            args.push_back("--max-memory-slots");
            args.push_back(std::to_string(max_memory_slots));
        }
        if (compress_interval > 0) {
            args.push_back("--compress-interval");
            args.push_back(std::to_string(compress_interval));
        }
    }

    const cypha::bench::RunProcessResult proc =
        cypha::bench::run_executable_capture(bench_native_exe, args);
    Json bench_output = Json::object();
    bool stdout_parsed = false;
    if (!proc.stdout_text.empty()) {
        bench_output = parse_subprocess_json_stdout(proc.stdout_text);
        stdout_parsed = !bench_output.empty();
    }
    if (proc.exit_code != 0) {
        throw std::runtime_error(std::string(label) + " exit=" + std::to_string(proc.exit_code));
    }
    if (!stdout_parsed) {
        throw std::runtime_error(std::string(label) + " produced no JSON stdout");
    }

    const bool profile_complete = parse_bench_intelligence_profile_complete(bench_output);
    const bool bpc_finite = json_bpc_finite(bench_output);
    const auto bpc = json_bpc_value(bench_output);
    const auto kappa = json_kappa_value(bench_output);

    Json row = Json{{"exit_code", proc.exit_code},
                    {"stdout_parsed", stdout_parsed},
                    {"profile_complete", profile_complete},
                    {"bpc_finite", bpc_finite},
                    {"math_integration", math_integration}};
    if (bpc.has_value()) {
        row["bpc"] = *bpc;
    } else {
        row["bpc"] = nullptr;
    }
    if (kappa.has_value()) {
        row["kappa"] = *kappa;
    } else {
        row["kappa"] = nullptr;
    }
    return row;
}

Json run_d41_math_integration_scale_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";

    const std::array<const char*, 2> required_sources{
        "include/cypha/cyphalm/cyphalm_math_integration.hpp",
        "src/cyphalm/cyphalm_math_integration.cpp",
    };
    Json math_sources = Json::object();
    for (const char* rel : required_sources) {
        const fs::path path = native_root / rel;
        const bool present = fs::is_regular_file(path);
        math_sources[std::string("native/") + rel] = present;
        if (!present) {
            throw std::runtime_error("math integration source missing: " + path.string());
        }
    }

    const fs::path pgl_hpp = native_root / "include/cypha/intelligence/profile_guided_loss.hpp";
    const fs::path pgl_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    const bool pgl_present = fs::is_regular_file(pgl_hpp) && fs::is_regular_file(pgl_cpp);
    const bool pgl_math_ready = pgl_present && profile_guided_loss_math_ready(pgl_hpp);
    math_sources["native/profile_guided_loss_present"] = pgl_present;
    math_sources["native/profile_guided_loss_math_ready"] = pgl_math_ready;
    if (!pgl_present) {
        throw std::runtime_error("profile_guided_loss sources missing under native/");
    }
    if (!pgl_math_ready) {
        throw std::runtime_error(
            "profile_guided_loss.hpp missing navigation_loss_total or 7-stat lambdas");
    }

    std::vector<std::string> warnings;

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const Json baseline =
        run_math_integration_bench_subprocess(bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval,
                                              false, "cyphalm_bench_native baseline");
    const Json math_integration =
        run_math_integration_bench_subprocess(bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval,
                                              true, "cyphalm_bench_native --math-integration");

    const bool baseline_ready =
        baseline.value("profile_complete", false) && baseline.value("bpc_finite", false);
    const bool math_ready =
        math_integration.value("profile_complete", false) &&
        math_integration.value("bpc_finite", false);
    const std::string validation_status =
        baseline_ready && math_ready ? "math_integration_scale_ready"
                                     : "pending_math_integration_scale";

    Json delta = Json::object();
    const bool baseline_bpc_ok = baseline.contains("bpc") && !baseline["bpc"].is_null();
    const bool math_bpc_ok = math_integration.contains("bpc") && !math_integration["bpc"].is_null();
    const bool baseline_kappa_ok =
        baseline.contains("kappa") && !baseline["kappa"].is_null();
    const bool math_kappa_ok =
        math_integration.contains("kappa") && !math_integration["kappa"].is_null();

    if (baseline_bpc_ok && math_bpc_ok) {
        delta["delta_bpc"] =
            math_integration["bpc"].get<double>() - baseline["bpc"].get<double>();
    } else {
        delta["delta_bpc"] = nullptr;
        warnings.push_back("delta_bpc unavailable (missing finite bpc on baseline or math run)");
    }

    if (baseline_kappa_ok && math_kappa_ok) {
        delta["delta_kappa"] =
            math_integration["kappa"].get<double>() - baseline["kappa"].get<double>();
    } else {
        delta["delta_kappa"] = nullptr;
        warnings.push_back("delta_kappa unavailable (missing finite kappa on baseline or math run)");
    }

    const Json experiments{
        {"math_sources", math_sources},
        {"baseline", baseline},
        {"math_integration", math_integration},
        {"delta_bpc", delta["delta_bpc"]},
        {"delta_kappa", delta["delta_kappa"]},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"warnings", warnings},
        {"backend",
         "cyphalm_bench_native baseline + --math-integration --intelligence-profile @ medium scale"},
    };
    cypha::bench::finalize_domain("d41_math_integration_scale_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d41_math_integration_scale_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

bool lock_section_bpc_finite(const Json& section) {
    if (!section.is_object() || !section.contains("bpc") || section["bpc"].is_null() ||
        !section["bpc"].is_number()) {
        return false;
    }
    return std::isfinite(section["bpc"].get<double>());
}

bool lock_math_integration_results_usable(const Json& mir) {
    if (!mir.is_object()) {
        return false;
    }
    if (!mir.contains("baseline") || !mir.contains("math_integration")) {
        return false;
    }
    if (!mir["baseline"].is_object() || !mir["math_integration"].is_object()) {
        return false;
    }
    return lock_section_bpc_finite(mir["baseline"]) &&
           lock_section_bpc_finite(mir["math_integration"]);
}

Json lock_section_to_bench_row(const Json& section, bool math_integration) {
    Json row = Json{{"exit_code", 0},
                    {"stdout_parsed", true},
                    {"profile_complete", section.value("profile_complete", true)},
                    {"bpc_finite", lock_section_bpc_finite(section)},
                    {"math_integration", math_integration},
                    {"source", "BASELINE_LOCK.json"}};
    if (section.contains("bpc") && !section["bpc"].is_null()) {
        row["bpc"] = section["bpc"];
    } else {
        row["bpc"] = nullptr;
    }
    if (section.contains("kappa") && !section["kappa"].is_null()) {
        row["kappa"] = section["kappa"];
    } else {
        row["kappa"] = nullptr;
    }
    return row;
}

Json run_d42_math_integration_production_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";

    const std::array<const char*, 2> required_sources{
        "include/cypha/cyphalm/cyphalm_math_integration.hpp",
        "src/cyphalm/cyphalm_math_integration.cpp",
    };
    Json math_sources = Json::object();
    for (const char* rel : required_sources) {
        const fs::path path = native_root / rel;
        const bool present = fs::is_regular_file(path);
        math_sources[std::string("native/") + rel] = present;
        if (!present) {
            throw std::runtime_error("math integration source missing: " + path.string());
        }
    }

    const fs::path pgl_hpp = native_root / "include/cypha/intelligence/profile_guided_loss.hpp";
    const fs::path pgl_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    const bool pgl_present = fs::is_regular_file(pgl_hpp) && fs::is_regular_file(pgl_cpp);
    const bool pgl_math_ready = pgl_present && profile_guided_loss_math_ready(pgl_hpp);
    math_sources["native/profile_guided_loss_present"] = pgl_present;
    math_sources["native/profile_guided_loss_math_ready"] = pgl_math_ready;
    if (!pgl_present) {
        throw std::runtime_error("profile_guided_loss sources missing under native/");
    }
    if (!pgl_math_ready) {
        throw std::runtime_error(
            "profile_guided_loss.hpp missing navigation_loss_total or 7-stat lambdas");
    }

    const fs::path overnight_script = repo / "scripts" / "run_d17_overnight.ps1";
    if (!fs::is_regular_file(overnight_script)) {
        throw std::runtime_error("missing scripts/run_d17_overnight.ps1");
    }
    const bool overnight_has_math_switch =
        script_text_contains(overnight_script, "MathIntegration");
    const bool overnight_has_math_env =
        script_text_contains(overnight_script, "CYPHA_OVERNIGHT_MATH_INTEGRATION");
    if (!overnight_has_math_switch && !overnight_has_math_env) {
        throw std::runtime_error(
            "run_d17_overnight.ps1 missing MathIntegration param and "
            "CYPHA_OVERNIGHT_MATH_INTEGRATION");
    }

    const fs::path lock_path = cypha::bench::bench_root() / "BASELINE_LOCK.json";
    const Json lock = load_json_file(lock_path);
    if (!lock.contains("math_integration_results")) {
        throw std::runtime_error("BASELINE_LOCK.json missing math_integration_results key");
    }
    const Json& math_integration_results = lock["math_integration_results"];

    std::vector<std::string> warnings;
    Json overnight_wiring{
        {"script", "scripts/run_d17_overnight.ps1"},
        {"MathIntegration_switch", overnight_has_math_switch},
        {"CYPHA_OVERNIGHT_MATH_INTEGRATION", overnight_has_math_env},
    };

    Json baseline;
    Json math_integration;
    std::string bench_backend;
    int n_train = kD41ScaleNTrain;
    int n_eval = kD41ScaleNEval;
    const bool lock_usable = lock_math_integration_results_usable(math_integration_results);

    if (lock_usable) {
        baseline = lock_section_to_bench_row(math_integration_results["baseline"], false);
        math_integration =
            lock_section_to_bench_row(math_integration_results["math_integration"], true);
        if (math_integration_results.contains("n_train") &&
            math_integration_results["n_train"].is_number_integer()) {
            n_train = math_integration_results["n_train"].get<int>();
        }
        if (math_integration_results.contains("n_eval") &&
            math_integration_results["n_eval"].is_number_integer()) {
            n_eval = math_integration_results["n_eval"].get<int>();
        }
        bench_backend = "BASELINE_LOCK.json math_integration_results";
    } else {
        const fs::path exe_dir = resolve_native_exe_dir();
        const fs::path bench_native_exe =
            cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
        if (!fs::is_regular_file(bench_native_exe)) {
            throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
        }

#if defined(_WIN32)
        _putenv_s("CYPHA_BENCH_FAST", "1");
#else
        setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

        baseline = run_math_integration_bench_subprocess(bench_native_exe, kD41ScaleNTrain,
                                                         kD41ScaleNEval, false,
                                                         "cyphalm_bench_native baseline");
        math_integration = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native --math-integration");
        bench_backend =
            "cyphalm_bench_native baseline + --math-integration --intelligence-profile @ medium "
            "scale (lock stub)";
        if (!math_integration_results.is_null()) {
            warnings.push_back(
                "math_integration_results present but incomplete; used subprocess smoke");
        }
    }

    const bool baseline_ready =
        baseline.value("profile_complete", false) && baseline.value("bpc_finite", false);
    const bool math_ready =
        math_integration.value("profile_complete", false) &&
        math_integration.value("bpc_finite", false);
    const bool production_tier =
        n_train >= kProductionNTrainMin &&
        (!math_integration_results.contains("status") ||
         math_integration_results["status"].is_null() ||
         (math_integration_results["status"].is_string() &&
          math_integration_results["status"].get<std::string>() != "pending"));
    const std::string validation_status =
        baseline_ready && math_ready && production_tier ? "math_integration_production_ready"
                                                        : "pending_math_integration_production";

    Json delta = Json::object();
    const bool baseline_bpc_ok = baseline.contains("bpc") && !baseline["bpc"].is_null();
    const bool math_bpc_ok = math_integration.contains("bpc") && !math_integration["bpc"].is_null();
    const bool baseline_kappa_ok = baseline.contains("kappa") && !baseline["kappa"].is_null();
    const bool math_kappa_ok =
        math_integration.contains("kappa") && !math_integration["kappa"].is_null();

    if (baseline_bpc_ok && math_bpc_ok) {
        delta["delta_bpc"] =
            math_integration["bpc"].get<double>() - baseline["bpc"].get<double>();
    } else {
        delta["delta_bpc"] = nullptr;
        warnings.push_back("delta_bpc unavailable (missing finite bpc on baseline or math run)");
    }

    if (baseline_kappa_ok && math_kappa_ok) {
        delta["delta_kappa"] =
            math_integration["kappa"].get<double>() - baseline["kappa"].get<double>();
    } else {
        delta["delta_kappa"] = nullptr;
        warnings.push_back("delta_kappa unavailable (missing finite kappa on baseline or math run)");
    }

    const Json experiments{
        {"math_sources", math_sources},
        {"overnight_wiring", overnight_wiring},
        {"math_integration_results", math_integration_results},
        {"lock_file", lock_path.string()},
        {"lock_results_used", lock_usable},
        {"baseline", baseline},
        {"math_integration", math_integration},
        {"delta_bpc", delta["delta_bpc"]},
        {"delta_kappa", delta["delta_kappa"]},
        {"n_train", n_train},
        {"n_eval", n_eval},
        {"production_n_train_min", kProductionNTrainMin},
        {"production_tier", production_tier},
        {"validation_status", validation_status},
        {"warnings", warnings},
        {"backend", bench_backend},
    };
    cypha::bench::finalize_domain("d42_math_integration_production_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d42_math_integration_production_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

bool lock_has_best_pareto_variant(const Json& lock) {
    if (!lock.contains("cell_sweep_results") || !lock["cell_sweep_results"].is_object()) {
        return false;
    }
    const Json& cell_sweep = lock["cell_sweep_results"];
    if (!cell_sweep.contains("best_pareto_variant")) {
        return false;
    }
    const Json& best = cell_sweep["best_pareto_variant"];
    if (best.is_string()) {
        return !best.get<std::string>().empty();
    }
    if (best.is_object()) {
        return best.contains("id") && best["id"].is_string() &&
               !best["id"].get<std::string>().empty();
    }
    return false;
}

bool d22_has_pareto_ranked_variants(const Json& d22) {
    return d22.contains("pareto_ranked_variants") && d22["pareto_ranked_variants"].is_array() &&
           !d22["pareto_ranked_variants"].empty();
}

Json run_d43_math_integration_lock_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";

    const fs::path baseline_lock_cpp = native_root / "tools" / "cypha_baseline_lock.cpp";
    if (!fs::is_regular_file(baseline_lock_cpp)) {
        throw std::runtime_error("missing native/tools/cypha_baseline_lock.cpp");
    }
    const bool baseline_lock_source_has_d17_math =
        script_text_contains(baseline_lock_cpp, "d17-math");
    if (!baseline_lock_source_has_d17_math) {
        throw std::runtime_error("cypha_baseline_lock.cpp missing d17-math run kind");
    }

    const fs::path update_lock_path = repo / "scripts" / "update_baseline_lock.ps1";
    if (!fs::is_regular_file(update_lock_path)) {
        throw std::runtime_error("missing scripts/update_baseline_lock.ps1");
    }
    const bool update_lock_has_d17_math = script_text_contains(update_lock_path, "d17-math");
    if (!update_lock_has_d17_math) {
        throw std::runtime_error("update_baseline_lock.ps1 missing d17-math run option");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path baseline_lock_exe =
        cypha::bench::resolve_runner_exe("cypha_baseline_lock", exe_dir);
    const bool baseline_lock_exe_present = fs::is_regular_file(baseline_lock_exe);
    if (!baseline_lock_exe_present) {
        throw std::runtime_error("missing cypha_baseline_lock: " + baseline_lock_exe.string());
    }

    const fs::path lock_path = cypha::bench::bench_root() / "BASELINE_LOCK.json";
    const Json lock = load_json_file(lock_path);
    if (!lock.contains("math_integration_results")) {
        throw std::runtime_error("BASELINE_LOCK.json missing math_integration_results key");
    }
    const Json& math_integration_results = lock["math_integration_results"];

    int n_train = 0;
    if (math_integration_results.contains("n_train") &&
        math_integration_results["n_train"].is_number_integer()) {
        n_train = math_integration_results["n_train"].get<int>();
    }

    std::string math_status;
    if (math_integration_results.contains("status") &&
        math_integration_results["status"].is_string()) {
        math_status = math_integration_results["status"].get<std::string>();
    }

    std::vector<std::string> warnings;
    const bool medium_tier = n_train >= kD41ScaleNTrain;
    if (medium_tier && math_status == "pending") {
        throw std::runtime_error(
            "math_integration_results status=pending at n_train>=" +
            std::to_string(kD41ScaleNTrain));
    }
    const bool math_lock_status_ok =
        !medium_tier || (!math_status.empty() && math_status != "pending");

    Json pareto_source = Json::object();
    Json pareto_ranked_variants = Json::array();
    Json best_pareto_variant = nullptr;
    bool pareto_available = lock_has_best_pareto_variant(lock);

    if (pareto_available) {
        const Json& cell_sweep = lock["cell_sweep_results"];
        pareto_source = Json{{"kind", "cell_sweep_results"},
                             {"best_pareto_variant", cell_sweep["best_pareto_variant"]}};
        if (cell_sweep["best_pareto_variant"].is_object()) {
            best_pareto_variant = cell_sweep["best_pareto_variant"];
        }
    } else {
        const Json d22 = run_d22_intelligence_cross_profile();
        pareto_available = d22_has_pareto_ranked_variants(d22);
        if (d22.contains("pareto_ranked_variants") && d22["pareto_ranked_variants"].is_array()) {
            pareto_ranked_variants = d22["pareto_ranked_variants"];
        }
        if (d22.contains("best_pareto_variant") && !d22["best_pareto_variant"].is_null()) {
            best_pareto_variant = d22["best_pareto_variant"];
        }
        pareto_source = Json{{"kind", "d22_intelligence_cross_profile"},
                             {"pareto_ranked_variant_count", pareto_ranked_variants.size()}};
        if (!pareto_available) {
            warnings.push_back(
                "d22 cross profile produced empty pareto_ranked_variants and lock lacks "
                "cell_sweep_results.best_pareto_variant");
        }
    }

    const bool d17_math_supported =
        baseline_lock_source_has_d17_math && update_lock_has_d17_math && baseline_lock_exe_present;
    const std::string validation_status =
        math_lock_status_ok && pareto_available && d17_math_supported
            ? "math_integration_lock_ready"
            : "pending_math_integration_lock";

    const Json experiments{
        {"lock_file", lock_path.string()},
        {"math_integration_results", math_integration_results},
        {"n_train", n_train},
        {"medium_tier_n_train_min", kD41ScaleNTrain},
        {"medium_tier", medium_tier},
        {"math_lock_status", math_status.empty() ? Json(nullptr) : Json(math_status)},
        {"math_lock_status_ok", math_lock_status_ok},
        {"pareto_available", pareto_available},
        {"pareto_source", pareto_source},
        {"pareto_ranked_variants", pareto_ranked_variants},
        {"best_pareto_variant", best_pareto_variant},
        {"d17_math_supported", d17_math_supported},
        {"baseline_lock_exe_present", baseline_lock_exe_present},
        {"baseline_lock_source_has_d17_math", baseline_lock_source_has_d17_math},
        {"update_baseline_lock_has_d17_math", update_lock_has_d17_math},
        {"validation_status", validation_status},
        {"warnings", warnings},
        {"backend", "baseline_lock_math_integration_validate"},
    };
    cypha::bench::finalize_domain("d43_math_integration_lock_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d43_math_integration_lock_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d44_kernel_nystrom_cyphalm_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";

    const fs::path dif_cpp = native_root / "src/cyphalm/cyphalm_dif.cpp";
    const fs::path dif_hpp = native_root / "include/cypha/cyphalm/cyphalm_dif.hpp";
    const fs::path traj_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    const bool kernel_wired = fs::is_regular_file(dif_cpp) && fs::is_regular_file(dif_hpp) &&
                              script_text_contains(dif_cpp, "kernel_mem_") &&
                              script_text_contains(dif_cpp, "KernelMemory") &&
                              script_text_contains(dif_hpp, "kernel_memory.hpp");
    if (!kernel_wired) {
        throw std::runtime_error("CyphaDIF missing KernelMemory Nyström wiring");
    }
    const bool trajectory_wired = fs::is_regular_file(traj_cpp) &&
                                  script_text_contains(traj_cpp,
                                                       "scale_profile_guided_loss_from_trajectory");
    if (!trajectory_wired) {
        throw std::runtime_error("profile_guided_loss missing kappa trajectory scheduler");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path smoke_exe = cypha::bench::resolve_runner_exe("kernel_llm_h04_smoke", exe_dir);
    if (!fs::is_regular_file(smoke_exe)) {
        throw std::runtime_error("missing kernel_llm_h04_smoke: " + smoke_exe.string());
    }

    const cypha::bench::RunProcessResult proc =
        cypha::bench::run_executable_capture(smoke_exe, {});
    if (proc.exit_code != 0) {
        throw std::runtime_error("kernel_llm_h04_smoke exit=" + std::to_string(proc.exit_code));
    }

    const bool smoke_pass =
        proc.stdout_text.find("kernel_llm_h04_smoke: PASS") != std::string::npos;
    if (!smoke_pass) {
        throw std::runtime_error("kernel_llm_h04_smoke did not report PASS");
    }

    const std::string validation_status =
        kernel_wired && trajectory_wired && smoke_pass ? "kernel_nystrom_cyphalm_ready"
                                                       : "pending_kernel_nystrom_cyphalm";

    const Json experiments{
        {"kernel_wired", kernel_wired},
        {"trajectory_wired", trajectory_wired},
        {"kernel_llm_h04_smoke",
         Json{{"exit_code", proc.exit_code}, {"pass_reported", smoke_pass}}},
        {"validation_status", validation_status},
        {"backend", "kernel_nystrom_cyphalm_validate"},
    };
    cypha::bench::finalize_domain("d44_kernel_nystrom_cyphalm_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d44_kernel_nystrom_cyphalm_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d45_per_stat_navigation_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const bool per_stat_wired = fs::is_regular_file(pg_cpp) &&
                                script_text_contains(pg_cpp,
                                                     "scale_profile_guided_loss_by_stat_deviation") &&
                                script_text_contains(pg_cpp, "resolve_adaptive_profile_guided_config");
    if (!per_stat_wired) {
        throw std::runtime_error("profile_guided_loss missing per-stat deviation scheduler");
    }
    const bool export_wired = fs::is_regular_file(math_cpp) &&
                              script_text_contains(math_cpp, "stat_deltas") &&
                              script_text_contains(math_cpp, "use_per_stat_deviation_lambdas");
    if (!export_wired) {
        throw std::runtime_error("cyphalm_math_integration missing enriched export telemetry");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const cypha::bench::RunProcessResult proc = cypha::bench::run_executable_capture(
        bench_native_exe,
        {"--profile", "d17", "--mode", "hybrid", "--math-integration", "--intelligence-profile",
         "--n-train", std::to_string(kD40SmokeNTrain), "--n-eval", std::to_string(kD40SmokeNEval)});

    Json bench_output = Json::object();
    if (!proc.stdout_text.empty()) {
        bench_output = parse_subprocess_json_stdout(proc.stdout_text);
    }
    if (proc.exit_code != 0) {
        throw std::runtime_error("cyphalm_bench_native --math-integration exit=" +
                                 std::to_string(proc.exit_code));
    }
    if (bench_output.empty()) {
        throw std::runtime_error("cyphalm_bench_native --math-integration produced no JSON stdout");
    }
    if (!bench_output.contains("math_integration") ||
        !bench_output["math_integration"].is_object()) {
        throw std::runtime_error("bench JSON missing math_integration report");
    }
    const Json& math = bench_output["math_integration"];
    const bool has_stat_deltas =
        math.contains("stat_deltas") && math["stat_deltas"].is_object();
    const bool has_nav_config = math.contains("navigation_config") &&
                                math["navigation_config"].is_object() &&
                                math["navigation_config"].value("use_per_stat_deviation_lambdas",
                                                                 false);
    const bool has_effective =
        math.contains("effective_lambdas") && math["effective_lambdas"].is_object();
    const bool profile_complete = parse_bench_intelligence_profile_complete(bench_output);
    const bool bpc_finite = json_bpc_finite(bench_output);
    const bool math_ready =
        per_stat_wired && export_wired && has_stat_deltas && has_nav_config && has_effective &&
        profile_complete && bpc_finite;
    const std::string validation_status =
        math_ready ? "per_stat_navigation_ready" : "pending_per_stat_navigation";

    const Json experiments{
        {"per_stat_wired", per_stat_wired},
        {"export_wired", export_wired},
        {"has_stat_deltas", has_stat_deltas},
        {"has_nav_config", has_nav_config},
        {"has_effective_lambdas", has_effective},
        {"profile_complete", profile_complete},
        {"bpc_finite", bpc_finite},
        {"cyphalm_bench_native", Json{{"exit_code", proc.exit_code}}},
        {"n_train", kD40SmokeNTrain},
        {"n_eval", kD40SmokeNEval},
        {"validation_status", validation_status},
        {"backend", "cyphalm_bench_native --math-integration per-stat navigation"},
    };
    cypha::bench::finalize_domain("d45_per_stat_navigation_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d45_per_stat_navigation_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d46_math_stack_upgrade_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path model_cpp = native_root / "src/cyphalm/cyphalm_model.cpp";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const bool tau_gate_wired = fs::is_regular_file(model_cpp) &&
                                script_text_contains(model_cpp, "use_tau_forget_gate") &&
                                script_text_contains(model_cpp, "hybrid_forget_gate_scale");
    const bool math_stack_wired = fs::is_regular_file(math_cpp) &&
                                  script_text_contains(math_cpp, "use_tau_forget_gate") &&
                                  script_text_contains(math_cpp, "use_kernel_llr") &&
                                  script_text_contains(math_cpp, "per_stat_deviation_span = 1.0");
    if (!tau_gate_wired || !math_stack_wired) {
        throw std::runtime_error("Phase 33 math stack wiring incomplete");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path tau_smoke = cypha::bench::resolve_runner_exe("tau_forget_gate_smoke", exe_dir);
    if (!fs::is_regular_file(tau_smoke)) {
        throw std::runtime_error("missing tau_forget_gate_smoke: " + tau_smoke.string());
    }
    const cypha::bench::RunProcessResult tau_proc =
        cypha::bench::run_executable_capture(tau_smoke, {});
    if (tau_proc.exit_code != 0) {
        throw std::runtime_error("tau_forget_gate_smoke exit=" + std::to_string(tau_proc.exit_code));
    }

    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }
#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif
    const cypha::bench::RunProcessResult bench_proc = cypha::bench::run_executable_capture(
        bench_native_exe,
        {"--profile", "d17", "--mode", "hybrid", "--math-integration", "--intelligence-profile",
         "--n-train", std::to_string(kD40SmokeNTrain), "--n-eval", std::to_string(kD40SmokeNEval)});
    Json bench_output = Json::object();
    if (!bench_proc.stdout_text.empty()) {
        bench_output = parse_subprocess_json_stdout(bench_proc.stdout_text);
    }
    if (bench_proc.exit_code != 0) {
        throw std::runtime_error("cyphalm_bench_native math stack exit=" +
                                 std::to_string(bench_proc.exit_code));
    }
    if (!bench_output.contains("math_integration") ||
        !bench_output["math_integration"].is_object()) {
        throw std::runtime_error("bench JSON missing math_integration");
    }
    const Json& math = bench_output["math_integration"];
    const Json& nav = math.contains("navigation_config") && math["navigation_config"].is_object()
                          ? math["navigation_config"]
                          : Json::object();
    const bool tau_in_export = nav.value("use_tau_forget_gate", false);
    const bool kernel_in_export = nav.value("use_kernel_llr", false);
    const double span = nav.value("per_stat_deviation_span", 0.0);
    const bool span_ok = std::abs(span - 1.0) < 1e-6;
    const bool profile_complete = parse_bench_intelligence_profile_complete(bench_output);
    const bool bpc_finite = json_bpc_finite(bench_output);
    const bool math_ready = tau_in_export && kernel_in_export && span_ok && profile_complete &&
                              bpc_finite && tau_proc.stdout_text.find("PASS") != std::string::npos;
    const std::string validation_status =
        math_ready ? "math_stack_upgrade_ready" : "pending_math_stack_upgrade";

    const Json experiments{
        {"tau_gate_wired", tau_gate_wired},
        {"math_stack_wired", math_stack_wired},
        {"tau_forget_gate_smoke", Json{{"exit_code", tau_proc.exit_code}}},
        {"tau_in_export", tau_in_export},
        {"kernel_in_export", kernel_in_export},
        {"per_stat_deviation_span", span},
        {"profile_complete", profile_complete},
        {"bpc_finite", bpc_finite},
        {"validation_status", validation_status},
        {"backend", "Phase33_tau_kernel_span_math_stack"},
    };
    cypha::bench::finalize_domain("d46_math_stack_upgrade_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d46_math_stack_upgrade_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

constexpr int kD47AblationNTrain = 500;
constexpr int kD47AblationNEval = 80;

Json run_d47_span_ablation_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "use_kappa_ceiling_lambdas") ||
        !script_text_contains(math_cpp, "use_lstm_d_eff_hidden_nudge")) {
        throw std::runtime_error("math integration preset missing Phase 34 flags");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--per-stat-deviation-span")) {
        throw std::runtime_error("cyphalm_bench_native missing --per-stat-deviation-span");
    }
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "use_kappa_ceiling_lambdas")) {
        throw std::runtime_error("profile_guided_loss missing kappa ceiling scheduler");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> spans{0.5, 1.0, 1.5};
    Json span_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;
    for (double span : spans) {
        const Json baseline = run_math_integration_bench_subprocess(
            bench_native_exe, kD47AblationNTrain, kD47AblationNEval, false,
            "cyphalm_bench_native baseline", span);
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD47AblationNTrain, kD47AblationNEval, true,
            "cyphalm_bench_native --math-integration", span);
        if (!baseline.contains("bpc") || !math.contains("bpc") || baseline["bpc"].is_null() ||
            math["bpc"].is_null()) {
            throw std::runtime_error("span ablation missing bpc");
        }
        const double base_bpc = baseline["bpc"].get<double>();
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double score = kappa - 0.1 * delta_bpc;
        Json row{{"per_stat_deviation_span", span},
                 {"baseline_bpc", base_bpc},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"pareto_score", score}};
        span_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const std::string validation_status =
        best_row != nullptr ? "span_ablation_ready" : "pending_span_ablation";

    const Json experiments{
        {"span_ablation_rows", span_rows},
        {"best_span", best_row},
        {"n_train", kD47AblationNTrain},
        {"n_eval", kD47AblationNEval},
        {"validation_status", validation_status},
        {"backend", "span_ablation_math_integration"},
    };
    cypha::bench::finalize_domain("d47_span_ablation_validation", experiments);
    const fs::path table_path = cypha::bench::tables_dir() / "d47_span_ablation_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d48_kappa_ceiling_ablation_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "kappa_ceiling_strength") ||
        !script_text_contains(math_cpp, "use_eigenvalue_d_eff")) {
        throw std::runtime_error("math integration preset missing Phase 35 flags");
    }
    const fs::path meas_cpp = native_root / "src/intelligence/measurers.cpp";
    if (!fs::is_regular_file(meas_cpp) ||
        !script_text_contains(meas_cpp, "participation_ratio_covariance_eigenvalue")) {
        throw std::runtime_error("measurers missing eigenvalue D_eff PR");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--kappa-lambda-target")) {
        throw std::runtime_error("cyphalm_bench_native missing --kappa-lambda-target");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> targets{0.80, 0.83, 0.86};
    Json target_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;
    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k");
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("kappa ceiling ablation missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double target : targets) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native --math-integration kappa ceiling", -1.0, target);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("kappa ceiling ablation missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.05 * std::abs(delta_kappa);
        Json row{{"kappa_lambda_target", target},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        target_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "kappa_ceiling_joint_ready" : "kappa_ceiling_ablation_ready";

    const Json experiments{
        {"kappa_ceiling_rows", target_rows},
        {"best_target", best_row},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "kappa_ceiling_ablation_math_integration"},
    };
    cypha::bench::finalize_domain("d48_kappa_ceiling_ablation_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d48_kappa_ceiling_ablation_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d49_ceiling_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "kappa_excess_grad_nudge") ||
        !script_text_contains(pg_cpp, "use_kappa_trajectory_ceiling")) {
        throw std::runtime_error("profile_guided_loss missing Phase 36 joint tuning");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--kappa-ceiling-min-scale")) {
        throw std::runtime_error("cyphalm_bench_native missing --kappa-ceiling-min-scale");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 2> strengths{1.5, 2.5};
    const std::array<double, 2> min_scales{0.40, 0.55};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("ceiling grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double strength : strengths) {
        for (double min_scale : min_scales) {
            const Json math = run_math_integration_bench_subprocess(
                bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
                "cyphalm_bench_native ceiling grid", -1.0, -1.0, strength, min_scale,
                kMathIntegrationBenchSeed);
            if (!math.contains("bpc") || math["bpc"].is_null()) {
                throw std::runtime_error("ceiling grid missing math bpc");
            }
            const double math_bpc = math["bpc"].get<double>();
            const double delta_bpc = math_bpc - base_bpc;
            double kappa = 0.0;
            if (math.contains("kappa") && !math["kappa"].is_null()) {
                kappa = math["kappa"].get<double>();
            }
            const double delta_kappa = kappa - base_kappa;
            const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
            Json row{{"kappa_ceiling_strength", strength},
                     {"kappa_ceiling_min_scale", min_scale},
                     {"baseline_bpc", base_bpc},
                     {"baseline_kappa", base_kappa},
                     {"math_bpc", math_bpc},
                     {"delta_bpc", delta_bpc},
                     {"kappa", kappa},
                     {"delta_kappa", delta_kappa},
                     {"joint_score", score}};
            grid_rows.push_back(row);
            if (score > best_score) {
                best_score = score;
                best_row = row;
            }
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "ceiling_grid_joint_ready" : "ceiling_grid_ablation_ready";

    const Json experiments{
        {"ceiling_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "ceiling_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d49_ceiling_grid_joint_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d49_ceiling_grid_joint_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d50_math_joint_lock_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const fs::path model_cpp = native_root / "src/cyphalm/cyphalm_model.cpp";
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "kappa_excess_grad_nudge") ||
        !fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "kappa_excess_grad_margin") ||
        !fs::is_regular_file(model_cpp) ||
        !script_text_contains(model_cpp, "use_kappa_excess_grad_nudge")) {
        throw std::runtime_error("Phase 37 joint lock tuning sources missing");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--bench-seed")) {
        throw std::runtime_error("cyphalm_bench_native missing --bench-seed");
    }

    const fs::path lock_path = cypha::bench::bench_root() / "BASELINE_LOCK.json";
    const Json lock = load_json_file(lock_path);
    if (!lock.contains("math_integration_results")) {
        throw std::runtime_error("BASELINE_LOCK.json missing math_integration_results key");
    }
    const Json& lock_math = lock["math_integration_results"];
    if (!lock_math.contains("baseline") || !lock_math.contains("math_integration")) {
        throw std::runtime_error("math_integration_results missing baseline/math_integration");
    }
    const double lock_base_bpc = lock_math["baseline"]["bpc"].get<double>();
    const double lock_math_bpc = lock_math["math_integration"]["bpc"].get<double>();
    const double lock_base_kappa = lock_math["baseline"]["kappa"].get<double>();
    const double lock_math_kappa = lock_math["math_integration"]["kappa"].get<double>();
    const double lock_delta_bpc = lock_math_bpc - lock_base_bpc;
    const double lock_delta_kappa = lock_math_kappa - lock_base_kappa;

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k pinned seed", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    const Json math = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
        "cyphalm_bench_native math @ 5k pinned seed", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null() || !math.contains("bpc") ||
        math["bpc"].is_null()) {
        throw std::runtime_error("d50 joint lock missing bpc");
    }

    const double base_bpc = baseline["bpc"].get<double>();
    const double math_bpc = math["bpc"].get<double>();
    const double delta_bpc = math_bpc - base_bpc;
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;
    const double math_kappa =
        math.contains("kappa") && !math["kappa"].is_null() ? math["kappa"].get<double>() : 0.0;
    const double delta_kappa = math_kappa - base_kappa;

    const bool joint_ok = delta_bpc < 0.0 && std::abs(delta_kappa) <= 0.05;
    const bool lock_bpc_repro =
        std::abs(base_bpc - lock_base_bpc) <= kD50LockBpcTolerance &&
        std::abs(math_bpc - lock_math_bpc) <= kD50LockBpcTolerance;
    const bool lock_kappa_repro =
        std::abs(base_kappa - lock_base_kappa) <= kD50LockKappaTolerance &&
        std::abs(math_kappa - lock_math_kappa) <= kD50LockKappaTolerance;
    const bool lock_repro_ok = lock_bpc_repro && lock_kappa_repro;

    const std::string validation_status =
        joint_ok && lock_repro_ok
            ? "joint_lock_ready"
            : joint_ok ? "joint_ready"
                       : lock_repro_ok ? "lock_repro_ready" : "joint_lock_ablation_ready";

    const Json experiments{
        {"bench_seed", kMathIntegrationBenchSeed},
        {"baseline", baseline},
        {"math_integration", math},
        {"delta_bpc", delta_bpc},
        {"delta_kappa", delta_kappa},
        {"lock_baseline_bpc", lock_base_bpc},
        {"lock_math_bpc", lock_math_bpc},
        {"lock_delta_bpc", lock_delta_bpc},
        {"lock_delta_kappa", lock_delta_kappa},
        {"lock_bpc_tolerance", kD50LockBpcTolerance},
        {"lock_kappa_tolerance", kD50LockKappaTolerance},
        {"joint_ok", joint_ok},
        {"lock_repro_ok", lock_repro_ok},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "joint_lock_math_integration"},
    };
    cypha::bench::finalize_domain("d50_math_joint_lock_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d50_math_joint_lock_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d51_opt_in_lever_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "kappa_ceiling_min_scale = 0.40") ||
        !script_text_contains(math_cpp, "use_kappa_kernel_blend_scale") ||
        !script_text_contains(math_cpp, "use_reu_forget_gate = true")) {
        throw std::runtime_error("math integration preset missing Phase 39 tuning");
    }
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "scale_kernel_blend_from_kappa")) {
        throw std::runtime_error("profile_guided_loss missing Phase 38 kernel blend scaling");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--use-eigenvalue-d-eff")) {
        throw std::runtime_error("cyphalm_bench_native missing opt-in lever flags");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k lever ablation", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("d51 lever ablation missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    struct LeverRow {
        const char* lever_id;
        bool eigen;
        bool reu;
    };
    const std::array<LeverRow, 4> levers{{{ "preset", false, false },
                                          { "eigenvalue_d_eff", true, false },
                                          { "reu_forget_gate", false, true },
                                          { "eigen_reu", true, true }}};

    Json lever_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    for (const LeverRow& lever : levers) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native opt-in lever", -1.0, -1.0, -1.0, -1.0, kMathIntegrationBenchSeed,
            lever.eigen, lever.reu);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("d51 lever ablation missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"lever_id", lever.lever_id},
                 {"use_eigenvalue_d_eff", lever.eigen},
                 {"use_reu_forget_gate", lever.reu},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        lever_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "opt_in_lever_joint_ready" : "opt_in_lever_ablation_ready";

    const Json experiments{
        {"lever_rows", lever_rows},
        {"best_lever", best_row},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "opt_in_lever_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d51_opt_in_lever_joint_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d51_opt_in_lever_joint_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d52_preset_ship_lock_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "use_reu_forget_gate = true") ||
        !script_text_contains(math_cpp, "kappa_lambda_target = 0.83") ||
        !script_text_contains(math_cpp, "kappa_ceiling_min_scale = 0.40")) {
        throw std::runtime_error("math integration preset missing Phase 39 ship lock");
    }

    const fs::path lock_path = cypha::bench::bench_root() / "BASELINE_LOCK.json";
    const Json lock = load_json_file(lock_path);
    if (!lock.contains("math_integration_results")) {
        throw std::runtime_error("BASELINE_LOCK.json missing math_integration_results key");
    }
    const Json& lock_math = lock["math_integration_results"];
    const double lock_base_bpc = lock_math["baseline"]["bpc"].get<double>();
    const double lock_math_bpc = lock_math["math_integration"]["bpc"].get<double>();
    const double lock_base_kappa = lock_math["baseline"]["kappa"].get<double>();
    const double lock_math_kappa = lock_math["math_integration"]["kappa"].get<double>();

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k preset ship", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    const Json math = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
        "cyphalm_bench_native math @ 5k preset ship", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null() || !math.contains("bpc") ||
        math["bpc"].is_null()) {
        throw std::runtime_error("d52 preset ship lock missing bpc");
    }

    const double base_bpc = baseline["bpc"].get<double>();
    const double math_bpc = math["bpc"].get<double>();
    const double delta_bpc = math_bpc - base_bpc;
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;
    const double math_kappa =
        math.contains("kappa") && !math["kappa"].is_null() ? math["kappa"].get<double>() : 0.0;
    const double delta_kappa = math_kappa - base_kappa;

    const bool joint_ok = delta_bpc < 0.0 && std::abs(delta_kappa) <= 0.05;
    const bool lock_repro_ok =
        std::abs(base_bpc - lock_base_bpc) <= kD50LockBpcTolerance &&
        std::abs(math_bpc - lock_math_bpc) <= kD50LockBpcTolerance &&
        std::abs(base_kappa - lock_base_kappa) <= kD50LockKappaTolerance &&
        std::abs(math_kappa - lock_math_kappa) <= kD50LockKappaTolerance;

    const std::string validation_status =
        joint_ok && lock_repro_ok ? "preset_ship_lock_ready"
                                  : joint_ok ? "preset_ship_joint_ready"
                                               : "preset_ship_ablation_ready";

    const Json experiments{
        {"bench_seed", kMathIntegrationBenchSeed},
        {"baseline", baseline},
        {"math_integration", math},
        {"delta_bpc", delta_bpc},
        {"delta_kappa", delta_kappa},
        {"lock_baseline_bpc", lock_base_bpc},
        {"lock_math_bpc", lock_math_bpc},
        {"joint_ok", joint_ok},
        {"lock_repro_ok", lock_repro_ok},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "preset_ship_lock_math_integration"},
        {"preset_flags",
         Json{{"use_reu_forget_gate", true},
              {"kappa_lambda_target", 0.83},
              {"kappa_ceiling_min_scale", 0.40}}},
    };
    cypha::bench::finalize_domain("d52_preset_ship_lock_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d52_preset_ship_lock_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d53_production_preset_ship_lock_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "use_reu_forget_gate = true") ||
        !script_text_contains(math_cpp, "use_kappa_navigation_warmup_scale = true")) {
        throw std::runtime_error("math integration preset missing Phase 40 production ship lock");
    }
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "scale_navigation_warmup_from_kappa")) {
        throw std::runtime_error("profile_guided_loss missing Phase 40 navigation warmup scaling");
    }
    const fs::path prod_script = repo / "scripts" / "run_production_overnight.ps1";
    if (!fs::is_regular_file(prod_script) ||
        !script_text_contains(prod_script, "MathIntegration")) {
        throw std::runtime_error("run_production_overnight.ps1 missing -MathIntegration");
    }

    const fs::path lock_path = cypha::bench::bench_root() / "BASELINE_LOCK.json";
    const Json lock = load_json_file(lock_path);
    if (!lock.contains("math_integration_results")) {
        throw std::runtime_error("BASELINE_LOCK.json missing math_integration_results key");
    }
    const Json& lock_math = lock["math_integration_results"];
    const bool lock_usable = lock_math_integration_results_usable(lock_math);

    int n_train = kD41ScaleNTrain;
    int n_eval = kD41ScaleNEval;
    if (lock_math.contains("n_train") && lock_math["n_train"].is_number_integer()) {
        n_train = lock_math["n_train"].get<int>();
    }
    if (lock_math.contains("n_eval") && lock_math["n_eval"].is_number_integer()) {
        n_eval = lock_math["n_eval"].get<int>();
    }

    std::string math_status;
    if (lock_math.contains("status") && lock_math["status"].is_string()) {
        math_status = lock_math["status"].get<std::string>();
    }

    const bool production_tier = n_train >= kProductionNTrainMin;
    bool overnight_aligned = true;
    int overnight_n_train = 0;
    if (lock.contains("overnight_results") && lock["overnight_results"].is_object()) {
        const Json& overnight = lock["overnight_results"];
        if (overnight.contains("n_train") && overnight["n_train"].is_number_integer()) {
            overnight_n_train = overnight["n_train"].get<int>();
            if (production_tier && overnight_n_train != n_train) {
                overnight_aligned = false;
            }
        }
    }

    bool lock_joint_ok = false;
    double lock_delta_bpc = 0.0;
    double lock_delta_kappa = 0.0;
    if (lock_usable) {
        const double lock_base_bpc = lock_math["baseline"]["bpc"].get<double>();
        const double lock_math_bpc = lock_math["math_integration"]["bpc"].get<double>();
        lock_delta_bpc = lock_math_bpc - lock_base_bpc;
        if (lock_math["baseline"].contains("kappa") && lock_math["math_integration"].contains("kappa")) {
            lock_delta_kappa =
                lock_math["math_integration"]["kappa"].get<double>() -
                lock_math["baseline"]["kappa"].get<double>();
        }
        lock_joint_ok = lock_delta_bpc < 0.0 && std::abs(lock_delta_kappa) <= 0.05;
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k production ship", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    const Json math = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
        "cyphalm_bench_native math @ 5k production ship", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null() || !math.contains("bpc") ||
        math["bpc"].is_null()) {
        throw std::runtime_error("d53 production ship lock missing bpc");
    }

    const double subprocess_delta_bpc = math["bpc"].get<double>() - baseline["bpc"].get<double>();
    double subprocess_delta_kappa = 0.0;
    if (baseline.contains("kappa") && !baseline["kappa"].is_null() && math.contains("kappa") &&
        !math["kappa"].is_null()) {
        subprocess_delta_kappa =
            math["kappa"].get<double>() - baseline["kappa"].get<double>();
    }
    const bool subprocess_joint_ok =
        subprocess_delta_bpc < 0.0 && std::abs(subprocess_delta_kappa) <= 0.05;

    const bool production_lock_ready =
        production_tier && (math_status == "production" || math_status == "completed");

    const std::string validation_status =
        production_lock_ready && lock_joint_ok && overnight_aligned
            ? "production_preset_ship_lock_ready"
            : production_tier && lock_joint_ok
                  ? "production_preset_joint_ready"
                  : !production_tier && subprocess_joint_ok
                        ? "pending_production_preset_ship_lock"
                        : subprocess_joint_ok ? "preset_ship_production_wiring_ready"
                                              : "production_preset_ship_ablation_ready";

    const Json experiments{
        {"bench_seed", kMathIntegrationBenchSeed},
        {"baseline", baseline},
        {"math_integration", math},
        {"subprocess_delta_bpc", subprocess_delta_bpc},
        {"subprocess_delta_kappa", subprocess_delta_kappa},
        {"subprocess_joint_ok", subprocess_joint_ok},
        {"lock_delta_bpc", lock_delta_bpc},
        {"lock_delta_kappa", lock_delta_kappa},
        {"lock_joint_ok", lock_joint_ok},
        {"lock_usable", lock_usable},
        {"math_integration_results", lock_math},
        {"n_train", n_train},
        {"n_eval", n_eval},
        {"overnight_n_train", overnight_n_train},
        {"overnight_aligned", overnight_aligned},
        {"production_tier", production_tier},
        {"production_lock_ready", production_lock_ready},
        {"production_n_train_min", kProductionNTrainMin},
        {"math_lock_status", math_status.empty() ? Json(nullptr) : Json(math_status)},
        {"validation_status", validation_status},
        {"backend", "production_preset_ship_lock_math_integration"},
    };
    cypha::bench::finalize_domain("d53_production_preset_ship_lock_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d53_production_preset_ship_lock_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d54_production_math_certificate_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "use_reu_forget_gate = true") ||
        !script_text_contains(math_cpp, "use_kappa_navigation_warmup_scale = true")) {
        throw std::runtime_error("math integration preset missing Phase 40 production certificate");
    }
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "scale_navigation_warmup_from_kappa")) {
        throw std::runtime_error("profile_guided_loss missing navigation warmup scaling");
    }
    const fs::path prod_script = repo / "scripts" / "run_production_overnight.ps1";
    const fs::path overnight_script = repo / "scripts" / "run_d17_overnight.ps1";
    if (!fs::is_regular_file(prod_script) ||
        !script_text_contains(prod_script, "MathIntegration")) {
        throw std::runtime_error("run_production_overnight.ps1 missing -MathIntegration");
    }
    if (!fs::is_regular_file(overnight_script) ||
        (!script_text_contains(overnight_script, "MathIntegration") &&
         !script_text_contains(overnight_script, "CYPHA_OVERNIGHT_MATH_INTEGRATION"))) {
        throw std::runtime_error("run_d17_overnight.ps1 missing MathIntegration wiring");
    }

    const fs::path lock_path = cypha::bench::bench_root() / "BASELINE_LOCK.json";
    const Json lock = load_json_file(lock_path);
    if (!lock.contains("math_integration_results")) {
        throw std::runtime_error("BASELINE_LOCK.json missing math_integration_results key");
    }
    const Json& lock_math = lock["math_integration_results"];
    const bool lock_usable = lock_math_integration_results_usable(lock_math);

    int n_train = kD41ScaleNTrain;
    int n_eval = kD41ScaleNEval;
    if (lock_math.contains("n_train") && lock_math["n_train"].is_number_integer()) {
        n_train = lock_math["n_train"].get<int>();
    }
    if (lock_math.contains("n_eval") && lock_math["n_eval"].is_number_integer()) {
        n_eval = lock_math["n_eval"].get<int>();
    }

    std::string math_status;
    if (lock_math.contains("status") && lock_math["status"].is_string()) {
        math_status = lock_math["status"].get<std::string>();
    }

    const bool production_tier = n_train >= kProductionNTrainMin;
    bool overnight_aligned = true;
    int overnight_n_train = 0;
    double hybrid_bpc = 0.0;
    bool hybrid_bpc_ok = false;
    if (lock.contains("overnight_results") && lock["overnight_results"].is_object()) {
        const Json& overnight = lock["overnight_results"];
        if (overnight.contains("n_train") && overnight["n_train"].is_number_integer()) {
            overnight_n_train = overnight["n_train"].get<int>();
            if (production_tier && overnight_n_train != n_train) {
                overnight_aligned = false;
            }
        }
        if (overnight.contains("bpc") && overnight["bpc"].is_number()) {
            hybrid_bpc = overnight["bpc"].get<double>();
            hybrid_bpc_ok = std::isfinite(hybrid_bpc);
        }
    }

    bool lock_joint_ok = false;
    double lock_delta_bpc = 0.0;
    double lock_delta_kappa = 0.0;
    double lock_math_bpc = 0.0;
    if (lock_usable) {
        const double lock_base_bpc = lock_math["baseline"]["bpc"].get<double>();
        lock_math_bpc = lock_math["math_integration"]["bpc"].get<double>();
        lock_delta_bpc = lock_math_bpc - lock_base_bpc;
        if (lock_math["baseline"].contains("kappa") && lock_math["math_integration"].contains("kappa")) {
            lock_delta_kappa =
                lock_math["math_integration"]["kappa"].get<double>() -
                lock_math["baseline"]["kappa"].get<double>();
        }
        lock_joint_ok = lock_delta_bpc < 0.0 && std::abs(lock_delta_kappa) <= 0.05;
    }

    constexpr double kHybridBpcTolerance = 0.05;
    double delta_bpc_vs_hybrid = 0.0;
    bool hybrid_bpc_gate_ok = false;
    bool delta_bpc_vs_hybrid_ok = false;
    if (lock_usable && hybrid_bpc_ok && production_tier) {
        delta_bpc_vs_hybrid = lock_math_bpc - hybrid_bpc;
        delta_bpc_vs_hybrid_ok = true;
        hybrid_bpc_gate_ok = lock_math_bpc <= hybrid_bpc + kHybridBpcTolerance;
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k certificate", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    const Json math = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
        "cyphalm_bench_native math @ 5k certificate", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null() || !math.contains("bpc") ||
        math["bpc"].is_null()) {
        throw std::runtime_error("d54 production certificate missing bpc");
    }

    const double subprocess_delta_bpc = math["bpc"].get<double>() - baseline["bpc"].get<double>();
    double subprocess_delta_kappa = 0.0;
    if (baseline.contains("kappa") && !baseline["kappa"].is_null() && math.contains("kappa") &&
        !math["kappa"].is_null()) {
        subprocess_delta_kappa =
            math["kappa"].get<double>() - baseline["kappa"].get<double>();
    }
    const bool subprocess_joint_ok =
        subprocess_delta_bpc < 0.0 && std::abs(subprocess_delta_kappa) <= 0.05;

    const bool production_lock_ready =
        production_tier && (math_status == "production" || math_status == "completed");

    const std::string validation_status =
        production_lock_ready && lock_joint_ok && overnight_aligned && hybrid_bpc_gate_ok
            ? "production_math_certificate_ready"
            : production_tier && lock_joint_ok && overnight_aligned
                  ? "production_math_joint_ready"
                  : !production_tier && subprocess_joint_ok
                        ? "pending_production_math_certificate"
                        : subprocess_joint_ok ? "production_math_certificate_wiring_ready"
                                              : "production_math_certificate_ablation_ready";

    const Json experiments{
        {"bench_seed", kMathIntegrationBenchSeed},
        {"baseline", baseline},
        {"math_integration", math},
        {"subprocess_delta_bpc", subprocess_delta_bpc},
        {"subprocess_delta_kappa", subprocess_delta_kappa},
        {"subprocess_joint_ok", subprocess_joint_ok},
        {"lock_delta_bpc", lock_delta_bpc},
        {"lock_delta_kappa", lock_delta_kappa},
        {"lock_joint_ok", lock_joint_ok},
        {"lock_usable", lock_usable},
        {"delta_bpc_vs_hybrid_baseline",
         delta_bpc_vs_hybrid_ok ? Json(delta_bpc_vs_hybrid) : Json(nullptr)},
        {"hybrid_bpc", hybrid_bpc_ok ? Json(hybrid_bpc) : Json(nullptr)},
        {"hybrid_bpc_gate_ok", hybrid_bpc_gate_ok},
        {"hybrid_bpc_tolerance", kHybridBpcTolerance},
        {"math_integration_results", lock_math},
        {"n_train", n_train},
        {"n_eval", n_eval},
        {"overnight_n_train", overnight_n_train},
        {"overnight_aligned", overnight_aligned},
        {"production_tier", production_tier},
        {"production_lock_ready", production_lock_ready},
        {"production_n_train_min", kProductionNTrainMin},
        {"math_lock_status", math_status.empty() ? Json(nullptr) : Json(math_status)},
        {"validation_status", validation_status},
        {"backend", "production_math_certificate"},
    };
    cypha::bench::finalize_domain("d54_production_math_certificate_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d54_production_math_certificate_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d55_nav_warmup_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "scale_navigation_warmup_from_kappa")) {
        throw std::runtime_error("profile_guided_loss missing navigation warmup scaling");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--kappa-navigation-warmup-strength")) {
        throw std::runtime_error("cyphalm_bench_native missing --kappa-navigation-warmup-strength");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 2> strengths{0.25, 0.35};
    const std::array<double, 2> floors{0.60, 0.65};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k nav warmup grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("nav warmup grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double strength : strengths) {
        for (double floor : floors) {
            const Json math = run_math_integration_bench_subprocess(
                bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
                "cyphalm_bench_native nav warmup grid", -1.0, -1.0, -1.0, -1.0,
                kMathIntegrationBenchSeed, false, false, strength, floor);
            if (!math.contains("bpc") || math["bpc"].is_null()) {
                throw std::runtime_error("nav warmup grid missing math bpc");
            }
            const double math_bpc = math["bpc"].get<double>();
            const double delta_bpc = math_bpc - base_bpc;
            double kappa = 0.0;
            if (math.contains("kappa") && !math["kappa"].is_null()) {
                kappa = math["kappa"].get<double>();
            }
            const double delta_kappa = kappa - base_kappa;
            const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
            Json row{{"kappa_navigation_warmup_strength", strength},
                     {"kappa_navigation_warmup_floor", floor},
                     {"baseline_bpc", base_bpc},
                     {"baseline_kappa", base_kappa},
                     {"math_bpc", math_bpc},
                     {"delta_bpc", delta_bpc},
                     {"kappa", kappa},
                     {"delta_kappa", delta_kappa},
                     {"joint_score", score}};
            grid_rows.push_back(row);
            if (score > best_score) {
                best_score = score;
                best_row = row;
            }
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "nav_warmup_grid_joint_ready" : "nav_warmup_grid_ablation_ready";

    const Json experiments{
        {"nav_warmup_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell",
         Json{{"kappa_navigation_warmup_strength", 0.35},
              {"kappa_navigation_warmup_floor", 0.65}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "nav_warmup_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d55_nav_warmup_grid_joint_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d55_nav_warmup_grid_joint_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json find_cell_sweep_variant_row(const Json& sweep_out, const char* variant_id) {
    if (!sweep_out.contains("results") || !sweep_out["results"].is_array()) {
        return nullptr;
    }
    for (const auto& row : sweep_out["results"]) {
        if (row.is_object() && row.value("id", "") == variant_id) {
            return row;
        }
    }
    return nullptr;
}

Json run_cell_sweep_bench_subprocess(const fs::path& sweep_exe, bool math_integration,
                                     const char* label, std::int64_t bench_seed = -1) {
#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif
    if (bench_seed >= 0) {
#if defined(_WIN32)
        _putenv_s("CYPHA_BENCH_SEED", std::to_string(bench_seed).c_str());
#else
        setenv("CYPHA_BENCH_SEED", std::to_string(bench_seed).c_str(), 1);
#endif
    }
    std::vector<std::string> args{"--overnight-sweep-smoke", "--intelligence-profile"};
    if (math_integration) {
        args.push_back("--math-integration");
    }
    if (bench_seed >= 0) {
        args.push_back("--bench-seed");
        args.push_back(std::to_string(bench_seed));
    }
    const cypha::bench::RunProcessResult proc =
        cypha::bench::run_executable_capture(sweep_exe, args);
    Json sweep_out = Json::object();
    bool stdout_parsed = false;
    if (!proc.stdout_text.empty()) {
        sweep_out = parse_subprocess_json_stdout(proc.stdout_text);
        stdout_parsed = !sweep_out.empty();
    }
    if (proc.exit_code != 0) {
        throw std::runtime_error(std::string(label) + " exit=" + std::to_string(proc.exit_code));
    }
    if (!stdout_parsed) {
        throw std::runtime_error(std::string(label) + " produced no JSON stdout");
    }
    sweep_out["exit_code"] = proc.exit_code;
    sweep_out["stdout_parsed"] = stdout_parsed;
    return sweep_out;
}

Json run_d56_cell_sweep_math_integration_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path sweep_cpp = native_root / "tools/cypha_cell_hypothesis_sweep.cpp";
    if (!fs::is_regular_file(sweep_cpp) ||
        !script_text_contains(sweep_cpp, "--math-integration") ||
        !script_text_contains(sweep_cpp, "apply_math_integration_preset")) {
        throw std::runtime_error("cell sweep missing Phase 42 math integration wiring");
    }
    const fs::path overnight_script = repo / "scripts" / "run_d17_overnight.ps1";
    if (!fs::is_regular_file(overnight_script) ||
        !script_text_contains(overnight_script, "--math-integration")) {
        throw std::runtime_error("run_d17_overnight.ps1 missing cell sweep --math-integration");
    }
    const fs::path lock_cpp = native_root / "tools/cypha_baseline_lock.cpp";
    if (!fs::is_regular_file(lock_cpp) ||
        !script_text_contains(lock_cpp, "math_integration")) {
        throw std::runtime_error("cypha_baseline_lock missing cell sweep math integration");
    }

    const fs::path lock_path = cypha::bench::bench_root() / "BASELINE_LOCK.json";
    const Json lock = load_json_file(lock_path);
    bool lock_math_enabled = false;
    int lock_n_train = 0;
    if (lock.contains("cell_sweep_results") && lock["cell_sweep_results"].is_object()) {
        const Json& cell_sweep = lock["cell_sweep_results"];
        if (cell_sweep.contains("math_integration_enabled") &&
            cell_sweep["math_integration_enabled"].is_boolean()) {
            lock_math_enabled = cell_sweep["math_integration_enabled"].get<bool>();
        }
        if (cell_sweep.contains("n_train") && cell_sweep["n_train"].is_number_integer()) {
            lock_n_train = cell_sweep["n_train"].get<int>();
        }
    }
    const bool production_tier = lock_n_train >= kProductionNTrainMin;

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path sweep_exe =
        cypha::bench::resolve_runner_exe("cypha_cell_hypothesis_sweep", exe_dir);
    if (!fs::is_regular_file(sweep_exe)) {
        throw std::runtime_error("missing cypha_cell_hypothesis_sweep: " + sweep_exe.string());
    }

    const Json baseline_sweep =
        run_cell_sweep_bench_subprocess(sweep_exe, false, "cell sweep baseline smoke",
                                        kMathIntegrationBenchSeed);
    const Json math_sweep =
        run_cell_sweep_bench_subprocess(sweep_exe, true, "cell sweep math smoke",
                                        kMathIntegrationBenchSeed);

    const Json baseline_b2 = find_cell_sweep_variant_row(baseline_sweep, "B2");
    const Json math_b2 = find_cell_sweep_variant_row(math_sweep, "B2");
    if (baseline_b2.is_null() || math_b2.is_null()) {
        throw std::runtime_error("d56 cell sweep missing B2 row");
    }
    if (!baseline_b2.contains("bpc") || baseline_b2["bpc"].is_null() || !math_b2.contains("bpc") ||
        math_b2["bpc"].is_null()) {
        throw std::runtime_error("d56 cell sweep B2 missing bpc");
    }

    const double baseline_b2_bpc = baseline_b2["bpc"].get<double>();
    const double math_b2_bpc = math_b2["bpc"].get<double>();
    const double delta_bpc = math_b2_bpc - baseline_b2_bpc;

    double baseline_kappa = 0.0;
    double math_kappa = 0.0;
    const auto baseline_kappa_opt = variant_row_kappa(baseline_b2);
    const auto math_kappa_opt = variant_row_kappa(math_b2);
    if (baseline_kappa_opt.has_value()) {
        baseline_kappa = *baseline_kappa_opt;
    }
    if (math_kappa_opt.has_value()) {
        math_kappa = *math_kappa_opt;
    }
    const double delta_kappa = math_kappa - baseline_kappa;

    const bool joint_ok = delta_bpc < 0.0 && std::abs(delta_kappa) <= 0.05;

    Json baseline_pareto = nullptr;
    Json math_pareto = nullptr;
    if (baseline_sweep.contains("pareto_ranked_variants") &&
        baseline_sweep["pareto_ranked_variants"].is_array() &&
        !baseline_sweep["pareto_ranked_variants"].empty()) {
        baseline_pareto = baseline_sweep["pareto_ranked_variants"][0];
    }
    if (math_sweep.contains("pareto_ranked_variants") &&
        math_sweep["pareto_ranked_variants"].is_array() &&
        !math_sweep["pareto_ranked_variants"].empty()) {
        math_pareto = math_sweep["pareto_ranked_variants"][0];
    }

    const bool production_lock_ready =
        production_tier && lock_math_enabled &&
        (lock.contains("cell_sweep_results") &&
         lock["cell_sweep_results"].contains("status") &&
         lock["cell_sweep_results"]["status"].is_string() &&
         (lock["cell_sweep_results"]["status"].get<std::string>() == "production" ||
          lock["cell_sweep_results"]["status"].get<std::string>() == "completed"));

    const std::string validation_status =
        production_lock_ready && joint_ok ? "production_cell_sweep_math_ready"
        : production_tier && lock_math_enabled && joint_ok
              ? "cell_sweep_math_joint_ready"
              : production_tier && !lock_math_enabled && joint_ok
                    ? "pending_cell_sweep_math_integration"
                    : !production_tier && joint_ok ? "pending_cell_sweep_math_integration"
                    : joint_ok ? "cell_sweep_math_wiring_ready"
                               : "cell_sweep_math_ablation_ready";

    const Json experiments{
        {"bench_seed", kMathIntegrationBenchSeed},
        {"baseline_sweep", baseline_sweep},
        {"math_sweep", math_sweep},
        {"baseline_b2", baseline_b2},
        {"math_b2", math_b2},
        {"delta_bpc", delta_bpc},
        {"delta_kappa", delta_kappa},
        {"joint_ok", joint_ok},
        {"baseline_best_pareto", baseline_pareto},
        {"math_best_pareto", math_pareto},
        {"lock_math_enabled", lock_math_enabled},
        {"lock_n_train", lock_n_train},
        {"production_tier", production_tier},
        {"production_lock_ready", production_lock_ready},
        {"production_n_train_min", kProductionNTrainMin},
        {"validation_status", validation_status},
        {"backend", "cell_sweep_math_integration_joint"},
    };
    cypha::bench::finalize_domain("d56_cell_sweep_math_integration_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d56_cell_sweep_math_integration_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d57_production_cell_sweep_math_certificate_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path sweep_cpp = native_root / "tools/cypha_cell_hypothesis_sweep.cpp";
    if (!fs::is_regular_file(sweep_cpp) ||
        !script_text_contains(sweep_cpp, "--math-integration") ||
        !script_text_contains(sweep_cpp, "apply_math_integration_preset")) {
        throw std::runtime_error("cell sweep missing Phase 42 math integration wiring");
    }
    const fs::path prod_script = repo / "scripts" / "run_production_overnight.ps1";
    const fs::path finalize_script = repo / "scripts" / "finalize_production_overnight.ps1";
    if (!fs::is_regular_file(prod_script) ||
        !script_text_contains(prod_script, "MathIntegration")) {
        throw std::runtime_error("run_production_overnight.ps1 missing -MathIntegration");
    }
    if (!fs::is_regular_file(finalize_script) ||
        !script_text_contains(finalize_script, "domain-tag d56")) {
        throw std::runtime_error("finalize_production_overnight.ps1 missing d56 gate");
    }

    const fs::path lock_path = cypha::bench::bench_root() / "BASELINE_LOCK.json";
    const Json lock = load_json_file(lock_path);
    if (!lock.contains("cell_sweep_results")) {
        throw std::runtime_error("BASELINE_LOCK.json missing cell_sweep_results key");
    }
    const Json& cell_sweep = lock["cell_sweep_results"];

    bool lock_math_enabled = false;
    int n_train = 0;
    int n_eval = 0;
    std::string cell_status;
    double lock_b2_bpc = 0.0;
    bool lock_b2_bpc_ok = false;
    if (cell_sweep.contains("math_integration_enabled") &&
        cell_sweep["math_integration_enabled"].is_boolean()) {
        lock_math_enabled = cell_sweep["math_integration_enabled"].get<bool>();
    }
    if (cell_sweep.contains("n_train") && cell_sweep["n_train"].is_number_integer()) {
        n_train = cell_sweep["n_train"].get<int>();
    }
    if (cell_sweep.contains("n_eval") && cell_sweep["n_eval"].is_number_integer()) {
        n_eval = cell_sweep["n_eval"].get<int>();
    }
    if (cell_sweep.contains("status") && cell_sweep["status"].is_string()) {
        cell_status = cell_sweep["status"].get<std::string>();
    }
    if (cell_sweep.contains("b2_bpc") && cell_sweep["b2_bpc"].is_number()) {
        lock_b2_bpc = cell_sweep["b2_bpc"].get<double>();
        lock_b2_bpc_ok = std::isfinite(lock_b2_bpc);
    } else if (cell_sweep.contains("bpc") && cell_sweep["bpc"].is_number()) {
        lock_b2_bpc = cell_sweep["bpc"].get<double>();
        lock_b2_bpc_ok = std::isfinite(lock_b2_bpc);
    }

    const bool production_tier = n_train >= kProductionNTrainMin;
    bool overnight_aligned = true;
    int overnight_n_train = 0;
    double hybrid_bpc = 0.0;
    bool hybrid_bpc_ok = false;
    if (lock.contains("overnight_results") && lock["overnight_results"].is_object()) {
        const Json& overnight = lock["overnight_results"];
        if (overnight.contains("n_train") && overnight["n_train"].is_number_integer()) {
            overnight_n_train = overnight["n_train"].get<int>();
            if (production_tier && overnight_n_train != n_train) {
                overnight_aligned = false;
            }
        }
        if (overnight.contains("bpc") && overnight["bpc"].is_number()) {
            hybrid_bpc = overnight["bpc"].get<double>();
            hybrid_bpc_ok = std::isfinite(hybrid_bpc);
        }
    }

    constexpr double kHybridBpcTolerance = 0.05;
    double delta_bpc_vs_hybrid = 0.0;
    bool hybrid_bpc_gate_ok = false;
    bool delta_bpc_vs_hybrid_ok = false;
    if (lock_math_enabled && lock_b2_bpc_ok && hybrid_bpc_ok && production_tier) {
        delta_bpc_vs_hybrid = lock_b2_bpc - hybrid_bpc;
        delta_bpc_vs_hybrid_ok = true;
        hybrid_bpc_gate_ok = lock_b2_bpc <= hybrid_bpc + kHybridBpcTolerance;
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path sweep_exe =
        cypha::bench::resolve_runner_exe("cypha_cell_hypothesis_sweep", exe_dir);
    if (!fs::is_regular_file(sweep_exe)) {
        throw std::runtime_error("missing cypha_cell_hypothesis_sweep: " + sweep_exe.string());
    }

    const Json baseline_sweep =
        run_cell_sweep_bench_subprocess(sweep_exe, false, "cell sweep baseline certificate",
                                        kMathIntegrationBenchSeed);
    const Json math_sweep =
        run_cell_sweep_bench_subprocess(sweep_exe, true, "cell sweep math certificate",
                                        kMathIntegrationBenchSeed);

    const Json baseline_b2 = find_cell_sweep_variant_row(baseline_sweep, "B2");
    const Json math_b2 = find_cell_sweep_variant_row(math_sweep, "B2");
    if (baseline_b2.is_null() || math_b2.is_null()) {
        throw std::runtime_error("d57 cell sweep missing B2 row");
    }
    if (!baseline_b2.contains("bpc") || baseline_b2["bpc"].is_null() || !math_b2.contains("bpc") ||
        math_b2["bpc"].is_null()) {
        throw std::runtime_error("d57 cell sweep B2 missing bpc");
    }

    const double subprocess_delta_bpc =
        math_b2["bpc"].get<double>() - baseline_b2["bpc"].get<double>();
    double subprocess_delta_kappa = 0.0;
    const auto base_kappa_opt = variant_row_kappa(baseline_b2);
    const auto math_kappa_opt = variant_row_kappa(math_b2);
    if (base_kappa_opt.has_value() && math_kappa_opt.has_value()) {
        subprocess_delta_kappa = *math_kappa_opt - *base_kappa_opt;
    }
    const bool subprocess_joint_ok =
        subprocess_delta_bpc < 0.0 && std::abs(subprocess_delta_kappa) <= 0.05;

    const bool production_lock_ready =
        production_tier && lock_math_enabled &&
        (cell_status == "production" || cell_status == "completed");

    const std::string validation_status =
        production_lock_ready && subprocess_joint_ok && overnight_aligned && hybrid_bpc_gate_ok
            ? "production_cell_sweep_math_certificate_ready"
            : production_tier && subprocess_joint_ok && overnight_aligned
                  ? "production_cell_sweep_math_joint_ready"
                  : !production_tier && subprocess_joint_ok
                        ? "pending_production_cell_sweep_math_certificate"
                        : subprocess_joint_ok
                              ? "production_cell_sweep_math_certificate_wiring_ready"
                              : "production_cell_sweep_math_certificate_ablation_ready";

    const Json experiments{
        {"bench_seed", kMathIntegrationBenchSeed},
        {"baseline_sweep", baseline_sweep},
        {"math_sweep", math_sweep},
        {"baseline_b2", baseline_b2},
        {"math_b2", math_b2},
        {"subprocess_delta_bpc", subprocess_delta_bpc},
        {"subprocess_delta_kappa", subprocess_delta_kappa},
        {"subprocess_joint_ok", subprocess_joint_ok},
        {"delta_bpc_vs_hybrid_baseline",
         delta_bpc_vs_hybrid_ok ? Json(delta_bpc_vs_hybrid) : Json(nullptr)},
        {"hybrid_bpc", hybrid_bpc_ok ? Json(hybrid_bpc) : Json(nullptr)},
        {"hybrid_bpc_gate_ok", hybrid_bpc_gate_ok},
        {"hybrid_bpc_tolerance", kHybridBpcTolerance},
        {"lock_b2_bpc", lock_b2_bpc_ok ? Json(lock_b2_bpc) : Json(nullptr)},
        {"cell_sweep_results", cell_sweep},
        {"lock_math_enabled", lock_math_enabled},
        {"n_train", n_train},
        {"n_eval", n_eval},
        {"overnight_n_train", overnight_n_train},
        {"overnight_aligned", overnight_aligned},
        {"production_tier", production_tier},
        {"production_lock_ready", production_lock_ready},
        {"production_n_train_min", kProductionNTrainMin},
        {"cell_lock_status", cell_status.empty() ? Json(nullptr) : Json(cell_status)},
        {"validation_status", validation_status},
        {"backend", "production_cell_sweep_math_certificate"},
    };
    cypha::bench::finalize_domain("d57_production_cell_sweep_math_certificate_validation",
                                  experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() /
        "d57_production_cell_sweep_math_certificate_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d58_production_overnight_math_complete_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path prod_script = repo / "scripts" / "run_production_overnight.ps1";
    const fs::path lock_script = repo / "scripts" / "update_baseline_lock.ps1";
    const fs::path finalize_script = repo / "scripts" / "finalize_production_overnight.ps1";
    if (!fs::is_regular_file(prod_script) ||
        !script_text_contains(prod_script, "MathIntegration")) {
        throw std::runtime_error("run_production_overnight.ps1 missing -MathIntegration");
    }
    if (!fs::is_regular_file(lock_script) ||
        !script_text_contains(lock_script, "MathIntegration")) {
        throw std::runtime_error("update_baseline_lock.ps1 missing -MathIntegration");
    }
    if (!fs::is_regular_file(finalize_script) ||
        !script_text_contains(finalize_script, "domain-tag d57")) {
        throw std::runtime_error("finalize_production_overnight.ps1 missing d57 gate");
    }

    const fs::path lock_path = cypha::bench::bench_root() / "BASELINE_LOCK.json";
    const Json lock = load_json_file(lock_path);
    if (!lock.contains("math_integration_results")) {
        throw std::runtime_error("BASELINE_LOCK.json missing math_integration_results key");
    }
    if (!lock.contains("overnight_results") || !lock.contains("cell_sweep_results")) {
        throw std::runtime_error("BASELINE_LOCK.json missing overnight or cell_sweep sections");
    }

    const Json& lock_math = lock["math_integration_results"];
    const Json& overnight = lock["overnight_results"];
    const Json& cell_sweep = lock["cell_sweep_results"];
    const bool lock_math_usable = lock_math_integration_results_usable(lock_math);

    int math_n_train = 0;
    int overnight_n_train = overnight.contains("n_train") && overnight["n_train"].is_number_integer()
                               ? overnight["n_train"].get<int>()
                               : 0;
    int cell_n_train = cell_sweep.contains("n_train") && cell_sweep["n_train"].is_number_integer()
                           ? cell_sweep["n_train"].get<int>()
                           : 0;
    if (lock_math.contains("n_train") && lock_math["n_train"].is_number_integer()) {
        math_n_train = lock_math["n_train"].get<int>();
    }

    std::string math_status;
    if (lock_math.contains("status") && lock_math["status"].is_string()) {
        math_status = lock_math["status"].get<std::string>();
    }
    std::string cell_status;
    if (cell_sweep.contains("status") && cell_sweep["status"].is_string()) {
        cell_status = cell_sweep["status"].get<std::string>();
    }

    const bool production_tier = math_n_train >= kProductionNTrainMin;
    const bool math_production_ready =
        production_tier && (math_status == "production" || math_status == "completed");
    const bool cell_math_enabled =
        cell_sweep.contains("math_integration_enabled") &&
        cell_sweep["math_integration_enabled"].is_boolean() &&
        cell_sweep["math_integration_enabled"].get<bool>();
    const bool cell_production_ready =
        cell_n_train >= kProductionNTrainMin &&
        (cell_status == "production" || cell_status == "completed");
    const bool tier_aligned =
        production_tier && math_n_train == overnight_n_train && math_n_train == cell_n_train;
    const bool best_pareto_present =
        cell_sweep.contains("best_pareto_variant") &&
        (cell_sweep["best_pareto_variant"].is_object() ||
         cell_sweep["best_pareto_variant"].is_string());

    bool lock_joint_ok = false;
    double lock_delta_bpc = 0.0;
    double lock_delta_kappa = 0.0;
    double lock_math_bpc = 0.0;
    if (lock_math_usable) {
        const double lock_base_bpc = lock_math["baseline"]["bpc"].get<double>();
        lock_math_bpc = lock_math["math_integration"]["bpc"].get<double>();
        lock_delta_bpc = lock_math_bpc - lock_base_bpc;
        if (lock_math["baseline"].contains("kappa") && lock_math["math_integration"].contains("kappa")) {
            lock_delta_kappa =
                lock_math["math_integration"]["kappa"].get<double>() -
                lock_math["baseline"]["kappa"].get<double>();
        }
        lock_joint_ok = lock_delta_bpc < 0.0 && std::abs(lock_delta_kappa) <= 0.05;
    }

    constexpr double kHybridBpcTolerance = 0.05;
    double hybrid_bpc = 0.0;
    bool hybrid_bpc_ok = false;
    if (overnight.contains("bpc") && overnight["bpc"].is_number()) {
        hybrid_bpc = overnight["bpc"].get<double>();
        hybrid_bpc_ok = std::isfinite(hybrid_bpc);
    }
    bool hybrid_math_bpc_gate_ok = false;
    bool hybrid_cell_bpc_gate_ok = false;
    if (lock_math_usable && hybrid_bpc_ok && production_tier) {
        hybrid_math_bpc_gate_ok = lock_math_bpc <= hybrid_bpc + kHybridBpcTolerance;
    }
    double lock_b2_bpc = 0.0;
    bool lock_b2_bpc_ok = false;
    if (cell_sweep.contains("b2_bpc") && cell_sweep["b2_bpc"].is_number()) {
        lock_b2_bpc = cell_sweep["b2_bpc"].get<double>();
        lock_b2_bpc_ok = std::isfinite(lock_b2_bpc);
    } else if (cell_sweep.contains("bpc") && cell_sweep["bpc"].is_number()) {
        lock_b2_bpc = cell_sweep["bpc"].get<double>();
        lock_b2_bpc_ok = std::isfinite(lock_b2_bpc);
    }
    if (cell_math_enabled && lock_b2_bpc_ok && hybrid_bpc_ok && production_tier) {
        hybrid_cell_bpc_gate_ok = lock_b2_bpc <= hybrid_bpc + kHybridBpcTolerance;
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k math complete", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    const Json math = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
        "cyphalm_bench_native math @ 5k math complete", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null() || !math.contains("bpc") ||
        math["bpc"].is_null()) {
        throw std::runtime_error("d58 production math complete missing bpc");
    }

    const double subprocess_delta_bpc = math["bpc"].get<double>() - baseline["bpc"].get<double>();
    double subprocess_delta_kappa = 0.0;
    if (baseline.contains("kappa") && !baseline["kappa"].is_null() && math.contains("kappa") &&
        !math["kappa"].is_null()) {
        subprocess_delta_kappa =
            math["kappa"].get<double>() - baseline["kappa"].get<double>();
    }
    const bool subprocess_joint_ok =
        subprocess_delta_bpc < 0.0 && std::abs(subprocess_delta_kappa) <= 0.05;

    const bool production_lock_ready =
        math_production_ready && cell_production_ready && cell_math_enabled && tier_aligned &&
        best_pareto_present && lock_joint_ok && hybrid_math_bpc_gate_ok &&
        hybrid_cell_bpc_gate_ok;

    const std::string validation_status =
        production_lock_ready && subprocess_joint_ok
            ? "production_overnight_math_complete_ready"
            : production_tier && subprocess_joint_ok && tier_aligned && lock_joint_ok
                  ? "production_overnight_math_joint_ready"
                  : !production_tier && subprocess_joint_ok
                        ? "pending_production_overnight_math_complete"
                        : subprocess_joint_ok ? "production_overnight_math_wiring_ready"
                                              : "production_overnight_math_ablation_ready";

    const Json experiments{
        {"bench_seed", kMathIntegrationBenchSeed},
        {"baseline", baseline},
        {"math_integration", math},
        {"subprocess_delta_bpc", subprocess_delta_bpc},
        {"subprocess_delta_kappa", subprocess_delta_kappa},
        {"subprocess_joint_ok", subprocess_joint_ok},
        {"lock_delta_bpc", lock_delta_bpc},
        {"lock_delta_kappa", lock_delta_kappa},
        {"lock_joint_ok", lock_joint_ok},
        {"lock_usable", lock_math_usable},
        {"math_integration_results", lock_math},
        {"overnight_results", overnight},
        {"cell_sweep_results", cell_sweep},
        {"math_n_train", math_n_train},
        {"overnight_n_train", overnight_n_train},
        {"cell_n_train", cell_n_train},
        {"tier_aligned", tier_aligned},
        {"cell_math_enabled", cell_math_enabled},
        {"best_pareto_present", best_pareto_present},
        {"hybrid_bpc", hybrid_bpc_ok ? Json(hybrid_bpc) : Json(nullptr)},
        {"hybrid_math_bpc_gate_ok", hybrid_math_bpc_gate_ok},
        {"hybrid_cell_bpc_gate_ok", hybrid_cell_bpc_gate_ok},
        {"hybrid_bpc_tolerance", kHybridBpcTolerance},
        {"production_tier", production_tier},
        {"production_lock_ready", production_lock_ready},
        {"production_n_train_min", kProductionNTrainMin},
        {"math_lock_status", math_status.empty() ? Json(nullptr) : Json(math_status)},
        {"cell_lock_status", cell_status.empty() ? Json(nullptr) : Json(cell_status)},
        {"validation_status", validation_status},
        {"backend", "production_overnight_math_complete"},
    };
    cypha::bench::finalize_domain("d58_production_overnight_math_complete_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d58_production_overnight_math_complete_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d59_kernel_blend_floor_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "scale_kernel_blend_from_kappa")) {
        throw std::runtime_error("profile_guided_loss missing kernel blend scaling");
    }
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "use_kappa_kernel_blend_scale") ||
        !script_text_contains(math_cpp, "kappa_kernel_blend_floor = 0.08")) {
        throw std::runtime_error("math integration preset missing kernel blend floor");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--kappa-kernel-blend-floor")) {
        throw std::runtime_error("cyphalm_bench_native missing --kappa-kernel-blend-floor");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> floors{0.05, 0.08, 0.12};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k kernel blend grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("kernel blend floor grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double floor : floors) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native kernel blend floor grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, floor);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("kernel blend floor grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"kappa_kernel_blend_floor", floor},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "kernel_blend_floor_grid_joint_ready" : "kernel_blend_floor_grid_ablation_ready";

    const Json experiments{
        {"kernel_blend_floor_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"kappa_kernel_blend_floor", 0.08}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "kernel_blend_floor_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d59_kernel_blend_floor_grid_joint_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d59_kernel_blend_floor_grid_joint_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d60_excess_grad_margin_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "kappa_excess_grad_nudge")) {
        throw std::runtime_error("profile_guided_loss missing kappa excess grad nudge");
    }
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "kappa_excess_grad_margin = 0.02")) {
        throw std::runtime_error("math integration preset missing excess grad margin");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--kappa-excess-grad-margin")) {
        throw std::runtime_error("cyphalm_bench_native missing --kappa-excess-grad-margin");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> margins{0.01, 0.02, 0.04};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k excess grad margin grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("excess grad margin grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double margin : margins) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native excess grad margin grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, margin);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("excess grad margin grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"kappa_excess_grad_margin", margin},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "excess_grad_margin_grid_joint_ready" : "excess_grad_margin_grid_ablation_ready";

    const Json experiments{
        {"excess_grad_margin_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"kappa_excess_grad_margin", 0.02}, {"kappa_excess_grad_scale", 0.35}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "excess_grad_margin_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d60_excess_grad_margin_grid_joint_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d60_excess_grad_margin_grid_joint_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d61_excess_grad_scale_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "kappa_excess_grad_scale = 0.35")) {
        throw std::runtime_error("math integration preset missing excess grad scale");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--kappa-excess-grad-scale")) {
        throw std::runtime_error("cyphalm_bench_native missing --kappa-excess-grad-scale");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> scales{0.25, 0.35, 0.50};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k excess grad scale grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("excess grad scale grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double scale : scales) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native excess grad scale grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, scale);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("excess grad scale grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"kappa_excess_grad_scale", scale},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "excess_grad_scale_grid_joint_ready" : "excess_grad_scale_grid_ablation_ready";

    const Json experiments{
        {"excess_grad_scale_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"kappa_excess_grad_margin", 0.02}, {"kappa_excess_grad_scale", 0.35}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "excess_grad_scale_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d61_excess_grad_scale_grid_joint_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d61_excess_grad_scale_grid_joint_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

bool math_ablation_table_joint_ready(const Json& table) {
    if (table.is_null() || !table.is_object() || !table.contains("validation_status") ||
        !table["validation_status"].is_string()) {
        return false;
    }
    const std::string status = table["validation_status"].get<std::string>();
    return status.find("joint_ready") != std::string::npos ||
           status == "joint_lock_ready" || status == "preset_ship_lock_ready" ||
           status == "span_ablation_ready";
}

Json try_load_ablation_table(const fs::path& filename) {
    const fs::path path = cypha::bench::tables_dir() / filename;
    if (!fs::is_regular_file(path)) {
        return Json(nullptr);
    }
    std::ifstream in(path);
    if (!in) {
        return Json(nullptr);
    }
    Json j;
    in >> j;
    return j;
}

Json run_d62_math_ablation_stack_complete_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "apply_math_integration_preset") ||
        !script_text_contains(math_cpp, "use_kappa_kernel_blend_scale") ||
        !script_text_contains(math_cpp, "use_kappa_navigation_warmup_scale") ||
        !script_text_contains(math_cpp, "use_kappa_excess_grad_nudge")) {
        throw std::runtime_error("math integration preset incomplete for stack complete");
    }
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "scale_kernel_blend_from_kappa") ||
        !script_text_contains(pg_cpp, "scale_navigation_warmup_from_kappa") ||
        !script_text_contains(pg_cpp, "kappa_excess_grad_nudge")) {
        throw std::runtime_error("profile_guided_loss missing navigation math stack");
    }

    const fs::path finalize_script = repo / "scripts" / "finalize_production_overnight.ps1";
    if (!fs::is_regular_file(finalize_script) ||
        !script_text_contains(finalize_script, "domain-tag d58")) {
        throw std::runtime_error("finalize_production_overnight.ps1 missing d58 gate");
    }

    const std::array<const char*, 24> stack_tables{
        "d47_span_ablation_validation.json",
        "d48_kappa_ceiling_ablation_validation.json",
        "d49_ceiling_grid_joint_validation.json",
        "d50_math_joint_lock_validation.json",
        "d51_opt_in_lever_joint_validation.json",
        "d52_preset_ship_lock_validation.json",
        "d55_nav_warmup_grid_joint_validation.json",
        "d59_kernel_blend_floor_grid_joint_validation.json",
        "d60_excess_grad_margin_grid_joint_validation.json",
        "d61_excess_grad_scale_grid_joint_validation.json",
        "d63_reu_forget_blend_grid_joint_validation.json",
        "d64_kappa_trajectory_window_grid_joint_validation.json",
        "d65_navigation_loss_warmup_grid_joint_validation.json",
        "d66_free_energy_beta_grid_joint_validation.json",
        "d67_kernel_blend_grid_joint_validation.json",
        "d68_kernel_m_grid_joint_validation.json",
        "d69_hybrid_blend_logit_grid_joint_validation.json",
        "d70_mdl_forget_max_norm_grid_joint_validation.json",
        "d71_kernel_lr_scale_grid_joint_validation.json",
        "d72_alpha_init_grid_joint_validation.json",
        "d73_hybrid_blend_lr_grid_joint_validation.json",
        "d74_n_experts_grid_joint_validation.json",
        "d75_max_memory_slots_grid_joint_validation.json",
        "d76_compress_interval_grid_joint_validation.json",
    };

    Json table_audit = Json::array();
    int tables_present = 0;
    int tables_joint_ready = 0;
    for (const char* name : stack_tables) {
        const Json table = try_load_ablation_table(name);
        const bool present = !table.is_null();
        const bool joint_ready = math_ablation_table_joint_ready(table);
        if (present) {
            ++tables_present;
        }
        if (joint_ready) {
            ++tables_joint_ready;
        }
        table_audit.push_back(Json{{"table", name},
                                   {"present", present},
                                   {"joint_ready", joint_ready},
                                   {"validation_status",
                                    present && table.contains("validation_status")
                                        ? table["validation_status"]
                                        : Json(nullptr)}});
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k stack complete", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    const Json math = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
        "cyphalm_bench_native math @ 5k stack complete", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null() || !math.contains("bpc") ||
        math["bpc"].is_null()) {
        throw std::runtime_error("stack complete missing subprocess bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double math_bpc = math["bpc"].get<double>();
    const double delta_bpc = math_bpc - base_bpc;
    double delta_kappa = 0.0;
    if (baseline.contains("kappa") && math.contains("kappa") && !baseline["kappa"].is_null() &&
        !math["kappa"].is_null()) {
        delta_kappa = math["kappa"].get<double>() - baseline["kappa"].get<double>();
    }
    const bool subprocess_joint_ok = delta_bpc < 0.0 && std::abs(delta_kappa) <= 0.05;
    const bool stack_tables_complete =
        tables_present == static_cast<int>(stack_tables.size()) &&
        tables_joint_ready == static_cast<int>(stack_tables.size());

    std::string validation_status = "math_ablation_stack_ablation_ready";
    if (subprocess_joint_ok && stack_tables_complete) {
        validation_status = "math_ablation_stack_complete_ready";
    } else if (subprocess_joint_ok) {
        validation_status = "math_ablation_stack_joint_ready";
    }

    const Json experiments{
        {"table_audit", table_audit},
        {"tables_present", tables_present},
        {"tables_joint_ready", tables_joint_ready},
        {"tables_expected", static_cast<int>(stack_tables.size())},
        {"subprocess_baseline_bpc", base_bpc},
        {"subprocess_math_bpc", math_bpc},
        {"subprocess_delta_bpc", delta_bpc},
        {"subprocess_delta_kappa", delta_kappa},
        {"subprocess_joint_ok", subprocess_joint_ok},
        {"stack_tables_complete", stack_tables_complete},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "math_ablation_stack_complete"},
    };
    cypha::bench::finalize_domain("d62_math_ablation_stack_complete_validation", experiments);
    const fs::path table_path =
        cypha::bench::tables_dir() / "d62_math_ablation_stack_complete_validation.json";
    std::ofstream out(table_path);
    if (out) {
        out << experiments.dump(2);
    }
    return experiments;
}

Json run_d63_reu_forget_blend_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const fs::path model_cpp = native_root / "src/cyphalm/cyphalm_model.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "reu_forget_gate_blend = 0.25")) {
        throw std::runtime_error("math integration preset missing reu forget blend");
    }
    if (!fs::is_regular_file(model_cpp) ||
        !script_text_contains(model_cpp, "reu_forget_gate_blend")) {
        throw std::runtime_error("cyphalm_model missing reu forget blend scaling");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--reu-forget-gate-blend")) {
        throw std::runtime_error("cyphalm_bench_native missing --reu-forget-gate-blend");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> blends{0.0, 0.25, 0.50};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k reu forget blend grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("reu forget blend grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double blend : blends) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native reu forget blend grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, blend);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("reu forget blend grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"reu_forget_gate_blend", blend},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "reu_forget_blend_grid_joint_ready" : "reu_forget_blend_grid_ablation_ready";

    const Json experiments{
        {"reu_forget_blend_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"reu_forget_gate_blend", 0.25}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "reu_forget_blend_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d63_reu_forget_blend_grid_joint_validation", experiments);
    const fs::path d63_table_path =
        cypha::bench::tables_dir() / "d63_reu_forget_blend_grid_joint_validation.json";
    std::ofstream d63_out(d63_table_path);
    if (d63_out) {
        d63_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d64_kappa_trajectory_window_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const fs::path pg_cpp = native_root / "src/intelligence/profile_guided_loss.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "kappa_trajectory_window = 16")) {
        throw std::runtime_error("math integration preset missing kappa trajectory window");
    }
    if (!fs::is_regular_file(pg_cpp) ||
        !script_text_contains(pg_cpp, "scale_profile_guided_loss_from_trajectory")) {
        throw std::runtime_error("profile_guided_loss missing trajectory scaling");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--kappa-trajectory-window")) {
        throw std::runtime_error("cyphalm_bench_native missing --kappa-trajectory-window");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<int, 3> windows{8, 16, 32};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k trajectory window grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("trajectory window grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (int window : windows) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native trajectory window grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, window);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("trajectory window grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"kappa_trajectory_window", window},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status = joint_ok ? "kappa_trajectory_window_grid_joint_ready"
                                                   : "kappa_trajectory_window_grid_ablation_ready";

    const Json experiments{
        {"kappa_trajectory_window_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"kappa_trajectory_window", 16}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "kappa_trajectory_window_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d64_kappa_trajectory_window_grid_joint_validation", experiments);
    const fs::path d64_table_path =
        cypha::bench::tables_dir() / "d64_kappa_trajectory_window_grid_joint_validation.json";
    std::ofstream d64_out(d64_table_path);
    if (d64_out) {
        d64_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d65_navigation_loss_warmup_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const fs::path model_cpp = native_root / "src/cyphalm/cyphalm_model.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "navigation_loss_warmup_steps = 200")) {
        throw std::runtime_error("math integration preset missing navigation loss warmup steps");
    }
    if (!fs::is_regular_file(model_cpp) ||
        !script_text_contains(model_cpp, "navigation_loss_warmup_steps")) {
        throw std::runtime_error("cyphalm_model missing navigation loss warmup ramp");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--navigation-loss-warmup-steps")) {
        throw std::runtime_error("cyphalm_bench_native missing --navigation-loss-warmup-steps");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<int, 3> warmup_steps{100, 200, 400};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k nav loss warmup grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("nav loss warmup grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (int steps : warmup_steps) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native nav loss warmup grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, static_cast<int>(-1.0), steps);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("nav loss warmup grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"navigation_loss_warmup_steps", steps},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "navigation_loss_warmup_grid_joint_ready"
                 : "navigation_loss_warmup_grid_ablation_ready";

    const Json experiments{
        {"navigation_loss_warmup_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"navigation_loss_warmup_steps", 200}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "navigation_loss_warmup_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d65_navigation_loss_warmup_grid_joint_validation", experiments);
    const fs::path d65_table_path =
        cypha::bench::tables_dir() / "d65_navigation_loss_warmup_grid_joint_validation.json";
    std::ofstream d65_out(d65_table_path);
    if (d65_out) {
        d65_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d66_free_energy_beta_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const fs::path model_cpp = native_root / "src/cyphalm/cyphalm_model.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "free_energy_beta = 0.01") ||
        !script_text_contains(math_cpp, "use_free_energy_loss = true")) {
        throw std::runtime_error("math integration preset missing free energy loss");
    }
    if (!fs::is_regular_file(model_cpp) ||
        !script_text_contains(model_cpp, "free_energy_beta")) {
        throw std::runtime_error("cyphalm_model missing free energy penalty");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) ||
        !script_text_contains(bench_cpp, "--free-energy-beta")) {
        throw std::runtime_error("cyphalm_bench_native missing --free-energy-beta");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> betas{0.005, 0.01, 0.02};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k free energy beta grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("free energy beta grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double beta : betas) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native free energy beta grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, static_cast<int>(-1.0), -1, beta);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("free energy beta grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"free_energy_beta", beta},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "free_energy_beta_grid_joint_ready" : "free_energy_beta_grid_ablation_ready";

    const Json experiments{
        {"free_energy_beta_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"free_energy_beta", 0.01}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "free_energy_beta_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d66_free_energy_beta_grid_joint_validation", experiments);
    const fs::path d66_table_path =
        cypha::bench::tables_dir() / "d66_free_energy_beta_grid_joint_validation.json";
    std::ofstream d66_out(d66_table_path);
    if (d66_out) {
        d66_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d67_kernel_blend_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const fs::path model_cpp = native_root / "src/cyphalm/cyphalm_model.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "kernel_blend = 0.25") ||
        !script_text_contains(math_cpp, "use_kernel_llr = true")) {
        throw std::runtime_error("math integration preset missing kernel blend");
    }
    if (!fs::is_regular_file(model_cpp) ||
        !script_text_contains(model_cpp, "set_runtime_kernel_blend")) {
        throw std::runtime_error("cyphalm_model missing runtime kernel blend");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) || !script_text_contains(bench_cpp, "--kernel-blend")) {
        throw std::runtime_error("cyphalm_bench_native missing --kernel-blend");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> blends{0.15, 0.25, 0.40};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k kernel blend grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("kernel blend grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double blend : blends) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native kernel blend grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, static_cast<int>(-1.0), -1, -1.0, blend);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("kernel blend grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"kernel_blend", blend},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "kernel_blend_grid_joint_ready" : "kernel_blend_grid_ablation_ready";

    const Json experiments{
        {"kernel_blend_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"kernel_blend", 0.25}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "kernel_blend_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d67_kernel_blend_grid_joint_validation", experiments);
    const fs::path d67_table_path =
        cypha::bench::tables_dir() / "d67_kernel_blend_grid_joint_validation.json";
    std::ofstream d67_out(d67_table_path);
    if (d67_out) {
        d67_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d68_kernel_m_grid_joint_validation() {
    const fs::path repo = cypha::bench::bench_root().parent_path();
    const fs::path native_root = repo / "native";
    const fs::path math_cpp = native_root / "src/cyphalm/cyphalm_math_integration.cpp";
    const fs::path dif_cpp = native_root / "src/cyphalm/cyphalm_dif.cpp";
    if (!fs::is_regular_file(math_cpp) ||
        !script_text_contains(math_cpp, "kernel_m = 64")) {
        throw std::runtime_error("math integration preset missing kernel_m");
    }
    if (!fs::is_regular_file(dif_cpp) || !script_text_contains(dif_cpp, "kernel_m")) {
        throw std::runtime_error("cyphalm_dif missing kernel_m wiring");
    }
    const fs::path bench_cpp = native_root / "tools/cyphalm_bench_native.cpp";
    if (!fs::is_regular_file(bench_cpp) || !script_text_contains(bench_cpp, "--kernel-m")) {
        throw std::runtime_error("cyphalm_bench_native missing --kernel-m");
    }

    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<int, 3> kernel_ms{32, 64, 128};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k kernel m grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("kernel m grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (int km : kernel_ms) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native kernel m grid", -1.0, -1.0, -1.0, -1.0, kMathIntegrationBenchSeed,
            false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0, false, -1.0, static_cast<int>(-1.0), -1,
            -1.0, -1.0, km);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("kernel m grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"kernel_m", km},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "kernel_m_grid_joint_ready" : "kernel_m_grid_ablation_ready";

    const Json experiments{
        {"kernel_m_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"kernel_m", 64}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "kernel_m_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d68_kernel_m_grid_joint_validation", experiments);
    const fs::path d68_table_path =
        cypha::bench::tables_dir() / "d68_kernel_m_grid_joint_validation.json";
    std::ofstream d68_out(d68_table_path);
    if (d68_out) {
        d68_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d69_hybrid_blend_logit_grid_joint_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> blend_logits{0.0, 0.5, 1.0};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k hybrid blend logit grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("hybrid blend logit grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double logit : blend_logits) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native hybrid blend logit grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, static_cast<int>(-1.0), -1, -1.0, -1.0, -1, true, logit);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("hybrid blend logit grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"hybrid_blend_logit", logit},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "hybrid_blend_logit_grid_joint_ready" : "hybrid_blend_logit_grid_ablation_ready";

    const Json experiments{
        {"hybrid_blend_logit_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"hybrid_blend_logit", 0.5}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "hybrid_blend_logit_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d69_hybrid_blend_logit_grid_joint_validation", experiments);
    const fs::path d69_table_path =
        cypha::bench::tables_dir() / "d69_hybrid_blend_logit_grid_joint_validation.json";
    std::ofstream d69_out(d69_table_path);
    if (d69_out) {
        d69_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d70_mdl_forget_max_norm_grid_joint_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> max_norms{2.0, 4.0, 8.0};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k mdl forget max norm grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("mdl forget max norm grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double max_norm : max_norms) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native mdl forget max norm grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, static_cast<int>(-1.0), -1, -1.0, -1.0, -1, false, 0.0, max_norm);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("mdl forget max norm grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"mdl_forget_max_norm", max_norm},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "mdl_forget_max_norm_grid_joint_ready" : "mdl_forget_max_norm_grid_ablation_ready";

    const Json experiments{
        {"mdl_forget_max_norm_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"mdl_forget_max_norm", 4.0}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "mdl_forget_max_norm_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d70_mdl_forget_max_norm_grid_joint_validation", experiments);
    const fs::path d70_table_path =
        cypha::bench::tables_dir() / "d70_mdl_forget_max_norm_grid_joint_validation.json";
    std::ofstream d70_out(d70_table_path);
    if (d70_out) {
        d70_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d71_kernel_lr_scale_grid_joint_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> lr_scales{0.5, 1.0, 2.0};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k kernel lr scale grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("kernel lr scale grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double scale : lr_scales) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native kernel lr scale grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, static_cast<int>(-1.0), -1, -1.0, -1.0, -1, false, 0.0, -1.0, scale);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("kernel lr scale grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"kernel_lr_scale", scale},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "kernel_lr_scale_grid_joint_ready" : "kernel_lr_scale_grid_ablation_ready";

    const Json experiments{
        {"kernel_lr_scale_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"kernel_lr_scale", 1.0}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "kernel_lr_scale_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d71_kernel_lr_scale_grid_joint_validation", experiments);
    const fs::path d71_table_path =
        cypha::bench::tables_dir() / "d71_kernel_lr_scale_grid_joint_validation.json";
    std::ofstream d71_out(d71_table_path);
    if (d71_out) {
        d71_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d72_alpha_init_grid_joint_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> alpha_inits{0.3, 0.5, 0.7};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k alpha init grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("alpha init grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double alpha : alpha_inits) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native alpha init grid", -1.0, -1.0, -1.0, -1.0, kMathIntegrationBenchSeed,
            false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0, false, -1.0, static_cast<int>(-1.0), -1, -1.0,
            -1.0, -1, false, 0.0, -1.0, -1.0, alpha);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("alpha init grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"alpha_init", alpha},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "alpha_init_grid_joint_ready" : "alpha_init_grid_ablation_ready";

    const Json experiments{
        {"alpha_init_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"alpha_init", 0.5}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "alpha_init_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d72_alpha_init_grid_joint_validation", experiments);
    const fs::path d72_table_path =
        cypha::bench::tables_dir() / "d72_alpha_init_grid_joint_validation.json";
    std::ofstream d72_out(d72_table_path);
    if (d72_out) {
        d72_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d73_hybrid_blend_lr_grid_joint_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<double, 3> blend_lrs{0.005, 0.01, 0.02};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k hybrid blend lr grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("hybrid blend lr grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (double blend_lr : blend_lrs) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native hybrid blend lr grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, static_cast<int>(-1.0), -1, -1.0, -1.0, -1, false, 0.0, -1.0, -1.0, -1.0, blend_lr);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("hybrid blend lr grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"hybrid_blend_lr", blend_lr},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "hybrid_blend_lr_grid_joint_ready" : "hybrid_blend_lr_grid_ablation_ready";

    const Json experiments{
        {"hybrid_blend_lr_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"hybrid_blend_lr", 0.01}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "hybrid_blend_lr_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d73_hybrid_blend_lr_grid_joint_validation", experiments);
    const fs::path d73_table_path =
        cypha::bench::tables_dir() / "d73_hybrid_blend_lr_grid_joint_validation.json";
    std::ofstream d73_out(d73_table_path);
    if (d73_out) {
        d73_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d74_n_experts_grid_joint_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<int, 3> expert_counts{4, 8, 12};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k n experts grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("n experts grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (int experts : expert_counts) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native n experts grid", -1.0, -1.0, -1.0, -1.0, kMathIntegrationBenchSeed,
            false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0, false, -1.0, static_cast<int>(-1.0), -1, -1.0,
            -1.0, -1, false, 0.0, -1.0, -1.0, -1.0, -1.0, experts);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("n experts grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"n_experts", experts},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "n_experts_grid_joint_ready" : "n_experts_grid_ablation_ready";

    const Json experiments{
        {"n_experts_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"n_experts", 8}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "n_experts_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d74_n_experts_grid_joint_validation", experiments);
    const fs::path d74_table_path =
        cypha::bench::tables_dir() / "d74_n_experts_grid_joint_validation.json";
    std::ofstream d74_out(d74_table_path);
    if (d74_out) {
        d74_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d75_max_memory_slots_grid_joint_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<int, 3> slot_counts{128, 256, 512};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k max memory slots grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("max memory slots grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (int slots : slot_counts) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native max memory slots grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, static_cast<int>(-1.0), -1, -1.0, -1.0, -1, false, 0.0, -1.0, -1.0, -1.0, -1.0, -1, slots);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("max memory slots grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"max_memory_slots", slots},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "max_memory_slots_grid_joint_ready" : "max_memory_slots_grid_ablation_ready";

    const Json experiments{
        {"max_memory_slots_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"max_memory_slots", 256}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "max_memory_slots_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d75_max_memory_slots_grid_joint_validation", experiments);
    const fs::path d75_table_path =
        cypha::bench::tables_dir() / "d75_max_memory_slots_grid_joint_validation.json";
    std::ofstream d75_out(d75_table_path);
    if (d75_out) {
        d75_out << experiments.dump(2);
    }
    return experiments;
}

Json run_d76_compress_interval_grid_joint_validation() {
    const fs::path exe_dir = resolve_native_exe_dir();
    const fs::path bench_native_exe =
        cypha::bench::resolve_runner_exe("cyphalm_bench_native", exe_dir);
    if (!fs::is_regular_file(bench_native_exe)) {
        throw std::runtime_error("missing cyphalm_bench_native: " + bench_native_exe.string());
    }

#if defined(_WIN32)
    _putenv_s("CYPHA_BENCH_FAST", "1");
#else
    setenv("CYPHA_BENCH_FAST", "1", 1);
#endif

    const std::array<int, 3> compress_intervals{8, 16, 32};
    Json grid_rows = Json::array();
    Json best_row = nullptr;
    double best_score = -1e18;

    const Json baseline = run_math_integration_bench_subprocess(
        bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, false,
        "cyphalm_bench_native baseline @ 5k compress interval grid", -1.0, -1.0, -1.0, -1.0,
        kMathIntegrationBenchSeed);
    if (!baseline.contains("bpc") || baseline["bpc"].is_null()) {
        throw std::runtime_error("compress interval grid missing baseline bpc");
    }
    const double base_bpc = baseline["bpc"].get<double>();
    const double base_kappa =
        baseline.contains("kappa") && !baseline["kappa"].is_null() ? baseline["kappa"].get<double>()
                                                                   : 0.0;

    for (int interval : compress_intervals) {
        const Json math = run_math_integration_bench_subprocess(
            bench_native_exe, kD41ScaleNTrain, kD41ScaleNEval, true,
            "cyphalm_bench_native compress interval grid", -1.0, -1.0, -1.0, -1.0,
            kMathIntegrationBenchSeed, false, false, -1.0, -1.0, false, -1.0, false, -1.0, -1.0,
            false, -1.0, static_cast<int>(-1.0), -1, -1.0, -1.0, -1, false, 0.0, -1.0, -1.0, -1.0, -1.0, -1, -1,
            interval);
        if (!math.contains("bpc") || math["bpc"].is_null()) {
            throw std::runtime_error("compress interval grid missing math bpc");
        }
        const double math_bpc = math["bpc"].get<double>();
        const double delta_bpc = math_bpc - base_bpc;
        double kappa = 0.0;
        if (math.contains("kappa") && !math["kappa"].is_null()) {
            kappa = math["kappa"].get<double>();
        }
        const double delta_kappa = kappa - base_kappa;
        const double score = -delta_bpc - 0.08 * std::abs(delta_kappa);
        Json row{{"compress_interval", interval},
                 {"baseline_bpc", base_bpc},
                 {"baseline_kappa", base_kappa},
                 {"math_bpc", math_bpc},
                 {"delta_bpc", delta_bpc},
                 {"kappa", kappa},
                 {"delta_kappa", delta_kappa},
                 {"joint_score", score}};
        grid_rows.push_back(row);
        if (score > best_score) {
            best_score = score;
            best_row = row;
        }
    }

    const bool joint_ok = best_row != nullptr && best_row["delta_bpc"].get<double>() < 0.0 &&
                          std::abs(best_row["delta_kappa"].get<double>()) <= 0.05;
    const std::string validation_status =
        joint_ok ? "compress_interval_grid_joint_ready" : "compress_interval_grid_ablation_ready";

    const Json experiments{
        {"compress_interval_grid_rows", grid_rows},
        {"best_cell", best_row},
        {"preset_cell", Json{{"compress_interval", 16}}},
        {"bench_seed", kMathIntegrationBenchSeed},
        {"n_train", kD41ScaleNTrain},
        {"n_eval", kD41ScaleNEval},
        {"validation_status", validation_status},
        {"backend", "compress_interval_grid_joint_math_integration"},
    };
    cypha::bench::finalize_domain("d76_compress_interval_grid_joint_validation", experiments);
    const fs::path d76_table_path =
        cypha::bench::tables_dir() / "d76_compress_interval_grid_joint_validation.json";
    std::ofstream d76_out(d76_table_path);
    if (d76_out) {
        d76_out << experiments.dump(2);
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
        {"d29", "cypha_bench.domains.d29_release_readiness_validation", run_d29_release_readiness_validation},
        {"d30", "cypha_bench.domains.d30_artifact_hygiene_validation", run_d30_artifact_hygiene_validation},
        {"d31", "cypha_bench.domains.d31_post_overnight_pipeline_validation",
         run_d31_post_overnight_pipeline_validation},
        {"d32", "cypha_bench.domains.d32_production_complete_validation",
         run_d32_production_complete_validation},
        {"d33", "cypha_bench.domains.d33_release_publish_validation",
         run_d33_release_publish_validation},
        {"d34", "cypha_bench.domains.d34_repo_smoke_hygiene_validation",
         run_d34_repo_smoke_hygiene_validation},
        {"d35", "cypha_bench.domains.d35_lock_commit_pipeline_validation",
         run_d35_lock_commit_pipeline_validation},
        {"d36", "cypha_bench.domains.d36_pipeline_e2e_validation", run_d36_pipeline_e2e_validation},
        {"d37", "cypha_bench.domains.d37_lock_refresh_validation", run_d37_lock_refresh_validation},
        {"d38", "cypha_bench.domains.d38_overnight_certificate_validation",
         run_d38_overnight_certificate_validation},
        {"d39", "cypha_bench.domains.d39_intelligence_monitor_profile_validation",
         run_d39_intelligence_monitor_profile_validation},
        {"d40", "cypha_bench.domains.d40_math_integration_validation",
         run_d40_math_integration_validation},
        {"d41", "cypha_bench.domains.d41_math_integration_scale_validation",
         run_d41_math_integration_scale_validation},
        {"d42", "cypha_bench.domains.d42_math_integration_production_validation",
         run_d42_math_integration_production_validation},
        {"d43", "cypha_bench.domains.d43_math_integration_lock_validation",
         run_d43_math_integration_lock_validation},
        {"d44", "cypha_bench.domains.d44_kernel_nystrom_cyphalm_validation",
         run_d44_kernel_nystrom_cyphalm_validation},
        {"d45", "cypha_bench.domains.d45_per_stat_navigation_validation",
         run_d45_per_stat_navigation_validation},
        {"d46", "cypha_bench.domains.d46_math_stack_upgrade_validation",
         run_d46_math_stack_upgrade_validation},
        {"d47", "cypha_bench.domains.d47_span_ablation_validation",
         run_d47_span_ablation_validation},
        {"d48", "cypha_bench.domains.d48_kappa_ceiling_ablation_validation",
         run_d48_kappa_ceiling_ablation_validation},
        {"d49", "cypha_bench.domains.d49_ceiling_grid_joint_validation",
         run_d49_ceiling_grid_joint_validation},
        {"d50", "cypha_bench.domains.d50_math_joint_lock_validation",
         run_d50_math_joint_lock_validation},
        {"d51", "cypha_bench.domains.d51_opt_in_lever_joint_validation",
         run_d51_opt_in_lever_joint_validation},
        {"d52", "cypha_bench.domains.d52_preset_ship_lock_validation",
         run_d52_preset_ship_lock_validation},
        {"d53", "cypha_bench.domains.d53_production_preset_ship_lock_validation",
         run_d53_production_preset_ship_lock_validation},
        {"d54", "cypha_bench.domains.d54_production_math_certificate_validation",
         run_d54_production_math_certificate_validation},
        {"d55", "cypha_bench.domains.d55_nav_warmup_grid_joint_validation",
         run_d55_nav_warmup_grid_joint_validation},
        {"d56", "cypha_bench.domains.d56_cell_sweep_math_integration_validation",
         run_d56_cell_sweep_math_integration_validation},
        {"d57", "cypha_bench.domains.d57_production_cell_sweep_math_certificate_validation",
         run_d57_production_cell_sweep_math_certificate_validation},
        {"d58", "cypha_bench.domains.d58_production_overnight_math_complete_validation",
         run_d58_production_overnight_math_complete_validation},
        {"d59", "cypha_bench.domains.d59_kernel_blend_floor_grid_joint_validation",
         run_d59_kernel_blend_floor_grid_joint_validation},
        {"d60", "cypha_bench.domains.d60_excess_grad_margin_grid_joint_validation",
         run_d60_excess_grad_margin_grid_joint_validation},
        {"d61", "cypha_bench.domains.d61_excess_grad_scale_grid_joint_validation",
         run_d61_excess_grad_scale_grid_joint_validation},
        {"d62", "cypha_bench.domains.d62_math_ablation_stack_complete_validation",
         run_d62_math_ablation_stack_complete_validation},
        {"d63", "cypha_bench.domains.d63_reu_forget_blend_grid_joint_validation",
         run_d63_reu_forget_blend_grid_joint_validation},
        {"d64", "cypha_bench.domains.d64_kappa_trajectory_window_grid_joint_validation",
         run_d64_kappa_trajectory_window_grid_joint_validation},
        {"d65", "cypha_bench.domains.d65_navigation_loss_warmup_grid_joint_validation",
         run_d65_navigation_loss_warmup_grid_joint_validation},
        {"d66", "cypha_bench.domains.d66_free_energy_beta_grid_joint_validation",
         run_d66_free_energy_beta_grid_joint_validation},
        {"d67", "cypha_bench.domains.d67_kernel_blend_grid_joint_validation",
         run_d67_kernel_blend_grid_joint_validation},
        {"d68", "cypha_bench.domains.d68_kernel_m_grid_joint_validation",
         run_d68_kernel_m_grid_joint_validation},
        {"d69", "cypha_bench.domains.d69_hybrid_blend_logit_grid_joint_validation",
         run_d69_hybrid_blend_logit_grid_joint_validation},
        {"d70", "cypha_bench.domains.d70_mdl_forget_max_norm_grid_joint_validation",
         run_d70_mdl_forget_max_norm_grid_joint_validation},
        {"d71", "cypha_bench.domains.d71_kernel_lr_scale_grid_joint_validation",
         run_d71_kernel_lr_scale_grid_joint_validation},
        {"d72", "cypha_bench.domains.d72_alpha_init_grid_joint_validation",
         run_d72_alpha_init_grid_joint_validation},
        {"d73", "cypha_bench.domains.d73_hybrid_blend_lr_grid_joint_validation",
         run_d73_hybrid_blend_lr_grid_joint_validation},
        {"d74", "cypha_bench.domains.d74_n_experts_grid_joint_validation",
         run_d74_n_experts_grid_joint_validation},
        {"d75", "cypha_bench.domains.d75_max_memory_slots_grid_joint_validation",
         run_d75_max_memory_slots_grid_joint_validation},
        {"d76", "cypha_bench.domains.d76_compress_interval_grid_joint_validation",
         run_d76_compress_interval_grid_joint_validation},
    };
}

}  // namespace

std::vector<DomainSpec> all_domains() { return build_all_domains(); }

DomainJson run_d16_ewc_sweep() { return run_d16_ewc_sweep_impl(); }

DomainJson run_d15_fgsm_robustness_curve(const std::vector<double>& epsilons, int max_eval,
                                          std::uint64_t seed) {
    return run_d15_fgsm_robustness_curve_impl(epsilons, max_eval, seed);
}

}  // namespace cypha::bench
