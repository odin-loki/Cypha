#include "cypha/bench/bench_cross_domain.hpp"

#include "cypha/bench/bench_report_json.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cypha::bench {

namespace {

bool has_any_key(const ProfileJson& metrics, const std::vector<const char*>& keys) {
    for (const char* k : keys) {
        if (metrics.contains(k)) return true;
    }
    return false;
}

std::string lower_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool experiments_is_dict(const ProfileJson& payload) {
    const auto it = payload.find("experiments");
    return it != payload.end() && it->is_object();
}

ProfileJson collect_calibration_rows(const std::unordered_map<std::string, ProfileJson>& tables) {
    static const std::vector<const char*> kClassKeys = {
        "accuracy", "f1_macro", "mean_epistemic_var", "mean_confidence"};
    static const std::vector<const char*> kRegKeys = {"rmse", "r2", "mean_epistemic_var"};
    static const std::vector<const char*> kOodKeys = {
        "ood_auroc", "cypha_ood_auroc", "extrapolation_auroc"};

    ProfileJson rows = ProfileJson::array();
    std::vector<std::string> domain_ids;
    domain_ids.reserve(tables.size());
    for (const auto& kv : tables) domain_ids.push_back(kv.first);
    std::sort(domain_ids.begin(), domain_ids.end());

    for (const std::string& domain_id : domain_ids) {
        if (domain_id.rfind("cross_", 0) == 0) continue;
        const ProfileJson& payload = tables.at(domain_id);
        if (!experiments_is_dict(payload)) continue;
        const ProfileJson& experiments = payload["experiments"];
        for (auto it = experiments.begin(); it != experiments.end(); ++it) {
            const std::string exp_name = it.key();
            if (exp_name.empty() || exp_name[0] == '_') continue;
            if (!it->is_object()) continue;
            const ProfileJson& metrics = *it;

            ProfileJson row = ProfileJson{
                {"domain", domain_id},
                {"experiment", exp_name},
                {"task_type", "unknown"},
            };
            if (has_any_key(metrics, kClassKeys)) {
                row["task_type"] = "classification";
                if (metrics.contains("mean_confidence") && metrics["mean_confidence"].is_number()) {
                    row["ece_proxy"] = metrics["mean_confidence"];
                    if (metrics.contains("accuracy") && metrics["accuracy"].is_number()) {
                        row["ece_proxy"] = std::abs(metrics["mean_confidence"].get<double>() -
                                                    metrics["accuracy"].get<double>());
                    }
                }
            } else if (has_any_key(metrics, kRegKeys)) {
                row["task_type"] = "regression";
            }
            for (const char* k : kOodKeys) {
                if (metrics.contains(k)) row["ood_auroc"] = metrics[k];
            }
            if (metrics.contains("mean_epistemic_var")) {
                row["mean_epistemic_var"] = metrics["mean_epistemic_var"];
            }
            if (metrics.contains("cypha_mean_epistemic_var")) {
                row["mean_epistemic_var"] = metrics["cypha_mean_epistemic_var"];
            }
            if (metrics.contains("uncertainty_rank_correlation")) {
                row["uncertainty_rank_correlation"] = metrics["uncertainty_rank_correlation"];
            }
            rows.push_back(json_safe(row));
        }
    }
    return rows;
}

std::optional<ProfileJson> extract_adaptation(const std::string& domain_id, const std::string& exp_name,
                                              const ProfileJson& metrics) {
    static const char* kHints[] = {
        "adaptation", "drift", "online", "cross_asset", "10D", "17D", "detection_latency"};
    const std::string name_l = lower_copy(exp_name);
    bool matched = false;
    for (const char* h : kHints) {
        if (name_l.find(h) != std::string::npos) {
            matched = true;
            break;
        }
    }
    if (!matched) {
        if (domain_id == "d17" && metrics.contains("bpc_improvement")) {
            matched = true;
        } else if (!metrics.contains("detection_latency_steps")) {
            return std::nullopt;
        }
    }

    ProfileJson row = ProfileJson{{"domain", domain_id}, {"experiment", exp_name}};
    if (metrics.contains("detection_latency_steps")) {
        row["T_adapt_steps"] = metrics["detection_latency_steps"];
    }
    if (metrics.contains("bpc_improvement")) {
        row["T_adapt_steps"] = metrics.value("bpc_improvement", ProfileJson(nullptr));
        row["metric"] = "bpc_improvement";
    }
    if (metrics.contains("final_attack_acc")) row["recovery_accuracy"] = metrics["final_attack_acc"];
    if (metrics.contains("bpc_ood_before_adapt") && metrics.contains("bpc_ood_after_adapt")) {
        row["pre_drift_metric"] = metrics["bpc_ood_before_adapt"];
        row["post_adapt_metric"] = metrics["bpc_ood_after_adapt"];
    }
    if (row.size() <= 2) return std::nullopt;
    return json_safe(row);
}

std::optional<ProfileJson> scan_forgetting(const std::string& domain_id, const std::string& exp_name,
                                           const ProfileJson& metrics) {
    if (metrics.contains("forgetting_score")) {
        return json_safe(ProfileJson{
            {"domain", domain_id},
            {"experiment", exp_name},
            {"forgetting_score", metrics["forgetting_score"]},
            {"accuracy_before", metrics.value("task_a_accuracy_before", ProfileJson(nullptr))},
            {"accuracy_after", metrics.value("task_a_accuracy_after", ProfileJson(nullptr))},
        });
    }
    if (metrics.contains("task_a_accuracy_before") && metrics.contains("task_a_accuracy_after") &&
        metrics["task_a_accuracy_before"].is_number() && metrics["task_a_accuracy_after"].is_number()) {
        const double before = metrics["task_a_accuracy_before"].get<double>();
        const double after = metrics["task_a_accuracy_after"].get<double>();
        const double score = (before - after) / std::max(before, 1e-6);
        return json_safe(ProfileJson{
            {"domain", domain_id},
            {"experiment", exp_name},
            {"forgetting_score", score},
            {"accuracy_before", before},
            {"accuracy_after", after},
        });
    }
    return std::nullopt;
}

void walk_alpha_metrics(const std::string& domain_id, const std::string& exp_name, const ProfileJson& metrics,
                        const std::string& prefix, ProfileJson& rows) {
    if (metrics.contains("mean_alpha") || metrics.contains("mean_alpha_proxy")) {
        ProfileJson alpha = metrics.contains("mean_alpha") ? metrics["mean_alpha"] : metrics["mean_alpha_proxy"];
        ProfileJson row = ProfileJson{
            {"domain", domain_id},
            {"experiment", prefix + exp_name},
            {"mean_alpha", alpha},
        };
        if (metrics.contains("fraction_edge_of_chaos")) {
            row["fraction_edge_of_chaos"] = metrics["fraction_edge_of_chaos"];
        }
        // Trim leading underscore from experiment name when prefix ends with one.
        if (!prefix.empty() && row["experiment"].is_string()) {
            std::string ex = row["experiment"].get<std::string>();
            while (!ex.empty() && ex.front() == '_') ex.erase(ex.begin());
            row["experiment"] = ex;
        }
        rows.push_back(json_safe(row));
    }
    if (metrics.contains("per_equation") && metrics["per_equation"].is_object()) {
        for (auto it = metrics["per_equation"].begin(); it != metrics["per_equation"].end(); ++it) {
            if (it->is_object()) walk_alpha_metrics(domain_id, it.key(), *it, exp_name + "_", rows);
        }
    }
    if (metrics.contains("gzip_ratios") && metrics.contains("mean_alpha_proxy") &&
        metrics["gzip_ratios"].is_array()) {
        const ProfileJson& ratios = metrics["gzip_ratios"];
        ProfileJson alphas = metrics["mean_alpha_proxy"];
        if (!alphas.is_array()) alphas = ProfileJson::array({alphas});
        const std::size_t n = ratios.size();
        for (std::size_t i = 0; i < n; ++i) {
            ProfileJson alpha = i < alphas.size() ? alphas[i] : ProfileJson(nullptr);
            rows.push_back(json_safe(ProfileJson{
                {"domain", domain_id},
                {"experiment", exp_name + "_file_" + std::to_string(i)},
                {"mean_alpha", alpha},
                {"gzip_ratio", ratios[i]},
            }));
        }
    }
}

double mean_of(const std::vector<double>& vals) {
    if (vals.empty()) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.0;
    for (double v : vals) sum += v;
    return sum / static_cast<double>(vals.size());
}

double std_of(const std::vector<double>& vals) {
    if (vals.empty()) return std::numeric_limits<double>::quiet_NaN();
    const double m = mean_of(vals);
    double acc = 0.0;
    for (double v : vals) {
        const double d = v - m;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(vals.size()));
}

void foreach_experiment(const std::unordered_map<std::string, ProfileJson>& tables,
                        const std::function<void(const std::string&, const std::string&, const ProfileJson&)>& fn) {
    std::vector<std::string> domain_ids;
    domain_ids.reserve(tables.size());
    for (const auto& kv : tables) domain_ids.push_back(kv.first);
    std::sort(domain_ids.begin(), domain_ids.end());
    for (const std::string& domain_id : domain_ids) {
        if (domain_id.rfind("cross_", 0) == 0) continue;
        const ProfileJson& payload = tables.at(domain_id);
        if (!experiments_is_dict(payload)) continue;
        const ProfileJson& experiments = payload["experiments"];
        for (auto it = experiments.begin(); it != experiments.end(); ++it) {
            if (!it->is_object()) continue;
            fn(domain_id, it.key(), *it);
        }
    }
}

}  // namespace

ProfileJson run_uncertainty_calibration() {
    const auto tables = load_all_domain_tables();
    const ProfileJson rows = collect_calibration_rows(tables);
    std::vector<double> ood_vals;
    for (const auto& row : rows) {
        if (row.contains("ood_auroc") && row["ood_auroc"].is_number()) {
            ood_vals.push_back(row["ood_auroc"].get<double>());
        }
    }
    const ProfileJson summary = ProfileJson{
        {"n_experiments", rows.size()},
        {"mean_ood_auroc", mean_of(ood_vals)},
    };
    const ProfileJson result = ProfileJson{{"calibration_rows", rows}, {"summary", summary}};
    finalize_domain("cross_uncertainty_calibration", result);
    return result;
}

ProfileJson run_online_adaptation() {
    const auto tables = load_all_domain_tables();
    ProfileJson rows = ProfileJson::array();
    foreach_experiment(tables, [&](const std::string& domain_id, const std::string& exp_name,
                                   const ProfileJson& metrics) {
        if (auto row = extract_adaptation(domain_id, exp_name, metrics)) rows.push_back(*row);
    });
    const ProfileJson result =
        ProfileJson{{"adaptation_rows", rows}, {"n_domains_with_adaptation_signal", rows.size()}};
    finalize_domain("cross_online_adaptation", result);
    return result;
}

ProfileJson run_forgetting_resistance() {
    const auto tables = load_all_domain_tables();
    ProfileJson rows = ProfileJson::array();
    foreach_experiment(tables, [&](const std::string& domain_id, const std::string& exp_name,
                                   const ProfileJson& metrics) {
        if (auto row = scan_forgetting(domain_id, exp_name, metrics)) rows.push_back(*row);
    });
    std::vector<double> scores;
    for (const auto& row : rows) {
        if (row.contains("forgetting_score") && row["forgetting_score"].is_number()) {
            scores.push_back(row["forgetting_score"].get<double>());
        }
    }
    const ProfileJson result =
        ProfileJson{{"forgetting_rows", rows}, {"mean_forgetting_score", mean_of(scores)}};
    finalize_domain("cross_forgetting_resistance", result);
    return result;
}

ProfileJson run_alpha_spectrum_global() {
    const auto tables = load_all_domain_tables();
    ProfileJson rows = ProfileJson::array();
    foreach_experiment(tables, [&](const std::string& domain_id, const std::string& exp_name,
                                   const ProfileJson& metrics) {
        walk_alpha_metrics(domain_id, exp_name, metrics, "", rows);
    });

    std::vector<double> alphas;
    for (const auto& row : rows) {
        if (!row.contains("mean_alpha")) continue;
        const ProfileJson& val = row["mean_alpha"];
        if (val.is_number()) {
            alphas.push_back(val.get<double>());
        } else if (val.is_array()) {
            for (const auto& item : val) {
                if (item.is_number()) alphas.push_back(item.get<double>());
            }
        }
    }

    double within_band = std::numeric_limits<double>::quiet_NaN();
    if (!alphas.empty()) {
        int count = 0;
        for (double a : alphas) {
            if (std::abs(a - 0.5) < 0.15) ++count;
        }
        within_band = static_cast<double>(count) / static_cast<double>(alphas.size());
    }

    const ProfileJson summary = ProfileJson{
        {"n_measurements", rows.size()},
        {"global_mean_alpha", mean_of(alphas)},
        {"global_std_alpha", std_of(alphas)},
        {"within_gul_band_fraction", within_band},
    };
    const ProfileJson result = ProfileJson{{"alpha_rows", rows}, {"summary", summary}};
    finalize_domain("cross_alpha_spectrum_global", result);
    return result;
}

void run_all_cross_domain() {
    (void)run_uncertainty_calibration();
    (void)run_online_adaptation();
    (void)run_forgetting_resistance();
    (void)run_alpha_spectrum_global();
}

}  // namespace cypha::bench
