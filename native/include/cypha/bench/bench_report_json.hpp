#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace cypha::bench {

using ProfileJson = nlohmann::json;

/// Write ``bench/report/tables/<domain_id>.json`` (mirrors ``save_domain_table``).
std::filesystem::path save_domain_table(const std::string& domain_id, const ProfileJson& metrics);

std::optional<ProfileJson> load_domain_table(const std::string& domain_id);

ProfileJson finalize_domain(const std::string& domain_id, const ProfileJson& experiments);

/// Recursively sanitize JSON for output (NaN/Inf → null), mirroring ``bench_common.json_safe``.
ProfileJson json_safe(const ProfileJson& obj);

/// Load all ``d*.json`` and ``cross_*.json`` tables from ``tables_dir()``.
std::unordered_map<std::string, ProfileJson> load_all_domain_tables();

}  // namespace cypha::bench
