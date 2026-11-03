#include "cypha/bench/bench_report_json.hpp"

#include "cypha/bench/bench_paths.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace cypha::bench {

namespace fs = std::filesystem;

namespace {

std::string utc_timestamp_iso8601() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const auto t = clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count()
        << "+00:00";
    return oss.str();
}

}  // namespace

fs::path save_domain_table(const std::string& domain_id, const ProfileJson& metrics) {
    const fs::path out_dir = tables_dir();
    fs::create_directories(out_dir);
    ProfileJson payload = ProfileJson::object({
        {"domain", domain_id},
        {"timestamp", utc_timestamp_iso8601()},
    });
    for (auto it = metrics.begin(); it != metrics.end(); ++it) payload[it.key()] = it.value();

    const fs::path out = out_dir / (domain_id + ".json");
    std::ofstream file(out);
    file << payload.dump(2);
    return out;
}

std::optional<ProfileJson> load_domain_table(const std::string& domain_id) {
    const fs::path path = tables_dir() / (domain_id + ".json");
    if (!fs::is_regular_file(path)) return std::nullopt;
    std::ifstream in(path);
    ProfileJson j;
    in >> j;
    return j;
}

ProfileJson finalize_domain(const std::string& domain_id, const ProfileJson& experiments) {
    const ProfileJson result = ProfileJson{{"experiments", experiments}};
    save_domain_table(domain_id, result);
    return result;
}

ProfileJson json_safe(const ProfileJson& obj) {
    if (obj.is_null() || obj.is_boolean() || obj.is_string()) return obj;
    if (obj.is_number()) {
        if (obj.is_number_float()) {
            const double v = obj.get<double>();
            if (!std::isfinite(v)) return ProfileJson(nullptr);
        }
        return obj;
    }
    if (obj.is_array()) {
        ProfileJson out = ProfileJson::array();
        for (const auto& item : obj) out.push_back(json_safe(item));
        return out;
    }
    if (obj.is_object()) {
        ProfileJson out = ProfileJson::object();
        for (auto it = obj.begin(); it != obj.end(); ++it) out[it.key()] = json_safe(it.value());
        return out;
    }
    return ProfileJson(obj.dump());
}

std::unordered_map<std::string, ProfileJson> load_all_domain_tables() {
    std::unordered_map<std::string, ProfileJson> tables;
    const fs::path dir = tables_dir();
    if (!fs::is_directory(dir)) return tables;

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
        paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());

    for (const fs::path& path : paths) {
        const std::string stem = path.stem().string();
        if (!(stem.rfind("d", 0) == 0 || stem.rfind("cross_", 0) == 0)) continue;

        std::string key = stem;
        if (stem.rfind("d", 0) == 0) {
            const auto us = stem.find('_');
            if (us != std::string::npos && us > 1) key = stem.substr(0, us);
        }
        if (tables.count(key) != 0) continue;

        try {
            std::ifstream in(path);
            ProfileJson j;
            in >> j;
            tables[key] = std::move(j);
        } catch (const std::exception&) {
            continue;
        }
    }
    return tables;
}

}  // namespace cypha::bench
