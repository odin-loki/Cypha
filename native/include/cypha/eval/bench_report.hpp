#pragma once

#include "cypha/bench/bench_report_json.hpp"

#include <filesystem>
#include <string>

namespace cypha::bench {

/// Build markdown report text from saved domain tables.
std::string build_markdown(const std::unordered_map<std::string, ProfileJson>* tables = nullptr);

/// Write ``BASELINE_REPORT.md`` (default under ``bench_root()``).
std::filesystem::path build_report(const std::filesystem::path& output_path = {});

/// Write ``report/summary.json`` with executive summary metadata.
std::filesystem::path build_report_summary(const std::filesystem::path& output_path = {});

}  // namespace cypha::bench
