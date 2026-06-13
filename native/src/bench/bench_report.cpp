#include "cypha/bench/bench_report.hpp"

#include "cypha/bench/bench_figures.hpp"
#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_report_json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace cypha::bench {

namespace fs = std::filesystem;

namespace {

std::string utc_now_report() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto t = clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M UTC");
    return oss.str();
}

bool is_domain_id(const std::string& id) {
    if (id.size() < 2 || id[0] != 'd') return false;
    for (std::size_t i = 1; i < id.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(id[i]))) return false;
    }
    return true;
}

void flatten_metrics(const ProfileJson& obj, const std::string& prefix, std::vector<std::pair<std::string, ProfileJson>>& rows) {
    if (!obj.is_object()) return;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const std::string key = prefix.empty() ? it.key() : prefix + "." + it.key();
        const ProfileJson& v = it.value();
        if (v.is_object()) {
            bool all_scalar = true;
            for (auto jt = v.begin(); jt != v.end(); ++jt) {
                if (jt->is_object() || jt->is_array()) {
                    all_scalar = false;
                    break;
                }
            }
            if (all_scalar) {
                for (auto jt = v.begin(); jt != v.end(); ++jt) {
                    rows.emplace_back(key + "." + jt.key(), jt.value());
                }
            } else {
                flatten_metrics(v, key, rows);
            }
        } else if (v.is_array() && !v.empty() && !v[0].is_object() && !v[0].is_array()) {
            rows.emplace_back(key, v);
        } else if (!v.is_object() && !v.is_array()) {
            rows.emplace_back(key, v);
        }
    }
}

std::string format_value(const ProfileJson& v) {
    if (v.is_null()) return "—";
    if (v.is_number_float() || v.is_number_integer()) {
        const double d = v.get<double>();
        if (!std::isfinite(d)) return "—";
        if (std::abs(d) < 0.001 && d != 0.0) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(2) << d;
            return oss.str();
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << d;
        return oss.str();
    }
    if (v.is_array() && v.size() > 6) {
        return "[" + std::to_string(v.size()) + " items]";
    }
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    return v.dump();
}

ProfileJson experiments_from_payload(const ProfileJson& payload) {
    ProfileJson experiments = payload.value("experiments", ProfileJson::object());
    if (experiments.is_array()) {
        ProfileJson out = ProfileJson::object();
        int i = 0;
        for (const auto& item : experiments) {
            if (!item.is_object()) continue;
            std::string name = "run_" + std::to_string(i);
            if (item.contains("encoding")) name = item["encoding"].get<std::string>();
            else if (item.contains("task")) name = item["task"].get<std::string>();
            out[name] = item;
            ++i;
        }
        return out;
    }
    if (experiments.is_object() && !experiments.empty()) return experiments;
    if (payload.contains("tasks") && payload["tasks"].is_array()) {
        ProfileJson out = ProfileJson::object();
        int i = 0;
        for (const auto& t : payload["tasks"]) {
            if (!t.is_object()) continue;
            const std::string name = t.value("task", "task_" + std::to_string(i));
            ProfileJson merged = ProfileJson::object();
            if (t.contains("scores") && t["scores"].is_object()) {
                for (auto it = t["scores"].begin(); it != t["scores"].end(); ++it) merged[it.key()] = it.value();
            }
            if (t.contains("cypha_metrics") && t["cypha_metrics"].is_object()) {
                for (auto it = t["cypha_metrics"].begin(); it != t["cypha_metrics"].end(); ++it) {
                    merged[it.key()] = it.value();
                }
            }
            out[name] = merged;
            ++i;
        }
        return out;
    }
    return ProfileJson::object();
}

void append_metric_table(std::vector<std::string>& lines, const ProfileJson& metrics, std::size_t limit) {
    std::vector<std::pair<std::string, ProfileJson>> flat;
    flatten_metrics(metrics, "", flat);
    std::sort(flat.begin(), flat.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    if (flat.empty()) {
        lines.push_back("_Empty metrics._");
        return;
    }
    lines.push_back("| Metric | Value |");
    lines.push_back("| --- | --- |");
    const std::size_t n = std::min(limit, flat.size());
    for (std::size_t i = 0; i < n; ++i) {
        lines.push_back("| `" + flat[i].first + "` | " + format_value(flat[i].second) + " |");
    }
}

}  // namespace

std::string build_markdown(const std::unordered_map<std::string, ProfileJson>* tables_ptr) {
    std::unordered_map<std::string, ProfileJson> loaded;
    if (!tables_ptr) {
        loaded = load_all_domain_tables();
        tables_ptr = &loaded;
    }
    const auto& tables = *tables_ptr;

    std::vector<std::string> lines;
    lines.push_back("# Cypha Bench Baseline Report");
    lines.push_back("");
    lines.push_back("Generated: " + utc_now_report());
    lines.push_back("");
    lines.push_back("Default parameters only — no hyperparameter tuning.");
    lines.push_back("");

    std::vector<std::string> domain_ids;
    std::vector<std::string> cross_ids;
    for (const auto& kv : tables) {
        if (is_domain_id(kv.first)) domain_ids.push_back(kv.first);
        else if (kv.first.rfind("cross_", 0) == 0) cross_ids.push_back(kv.first);
    }
    std::sort(domain_ids.begin(), domain_ids.end());
    std::sort(cross_ids.begin(), cross_ids.end());

    lines.push_back("## Executive Summary");
    lines.push_back("");
    lines.push_back("- Domains run: **" + std::to_string(domain_ids.size()) + "**");
    lines.push_back("- Cross-domain analyses: **" + std::to_string(cross_ids.size()) + "**");
    lines.push_back("");

    for (const std::string& domain_id : domain_ids) {
        const ProfileJson& payload = tables.at(domain_id);
        std::string heading = domain_id;
        std::transform(heading.begin(), heading.end(), heading.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        lines.push_back("## " + heading);
        lines.push_back("");
        if (payload.contains("timestamp") && payload["timestamp"].is_string()) {
            lines.push_back("*Timestamp:* " + payload["timestamp"].get<std::string>());
            lines.push_back("");
        }
        const ProfileJson experiments = experiments_from_payload(payload);
        if (experiments.empty()) {
            lines.push_back("_No experiments recorded._");
            lines.push_back("");
            continue;
        }
        for (auto it = experiments.begin(); it != experiments.end(); ++it) {
            lines.push_back("### " + it.key());
            lines.push_back("");
            if (!it->is_object()) {
                lines.push_back("- result: " + format_value(*it));
                lines.push_back("");
                continue;
            }
            if (it->value("skipped", false)) {
                lines.push_back("- **skipped:** " + it->value("reason", "unknown"));
                lines.push_back("");
                continue;
            }
            append_metric_table(lines, *it, 30);
            lines.push_back("");
        }
    }

    if (!cross_ids.empty()) {
        lines.push_back("## Cross-Domain Analyses");
        lines.push_back("");
        for (const std::string& cid : cross_ids) {
            const ProfileJson& payload = tables.at(cid);
            lines.push_back("### " + cid);
            lines.push_back("");
            ProfileJson summary = payload.value("summary", ProfileJson(nullptr));
            if (summary.is_null()) summary = payload.value("experiments", ProfileJson::object());
            if (summary.is_null()) summary = payload;
            append_metric_table(lines, summary, 40);
            lines.push_back("");
        }
    }

    lines.push_back("> **Note:** Report figures (JSON + native PNG) live in `bench/report/figures/` (`generate_figure_data`).");
    lines.push_back("");

    std::ostringstream out;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        out << lines[i];
        if (i + 1 < lines.size()) out << '\n';
    }
    return out.str();
}

fs::path build_report(const fs::path& output_path) {
    const fs::path out = output_path.empty() ? bench_root() / "BASELINE_REPORT.md" : output_path;
    fs::create_directories(out.parent_path());
    std::ofstream file(out);
    file << build_markdown();
    return out;
}

fs::path build_report_summary(const fs::path& output_path) {
    const fs::path out = output_path.empty() ? bench_root() / "report" / "summary.json" : output_path;
    fs::create_directories(out.parent_path());

    const auto tables = load_all_domain_tables();
    int domain_count = 0;
    int cross_count = 0;
    for (const auto& kv : tables) {
        if (is_domain_id(kv.first)) ++domain_count;
        else if (kv.first.rfind("cross_", 0) == 0) ++cross_count;
    }

    const ProfileJson summary = ProfileJson{
        {"generated_utc", utc_now_report()},
        {"domains_run", domain_count},
        {"cross_domain_analyses", cross_count},
        {"tables_dir", tables_dir().string()},
        {"figures_dir", figures_dir().string()},
        {"figures_note", "Native figure JSON and PNG under bench/report/figures/ (see figures_manifest.json)."},
        {"report_markdown", (bench_root() / "BASELINE_REPORT.md").string()},
    };

    std::ofstream file(out);
    file << summary.dump(2);
    return out;
}

}  // namespace cypha::bench
