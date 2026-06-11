#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha::bench {

using TuneJson = nlohmann::json;

struct TuneCell {
    std::string id;
    TuneJson args = TuneJson::object();
};

struct TuneSweepSpec {
    std::string sweep_id;
    std::string description;
    std::string runner;
    TuneJson defaults = TuneJson::object();
    TuneJson grid = TuneJson::object();
    std::vector<TuneCell> cells;
};

struct RunProcessResult {
    int exit_code = -1;
    std::string stdout_text;
    std::string stderr_text;
};

struct TuneRunResult {
    std::string cell_id;
    TuneJson args = TuneJson::object();
    int exit_code = -1;
    double seconds = 0.0;
    TuneJson output = TuneJson::object();
    bool ok = false;
    std::string error;
};

struct TuneSweepResult {
    std::string sweep_id;
    std::string runner;
    TuneJson defaults = TuneJson::object();
    std::vector<TuneRunResult> runs;
    double elapsed_s = 0.0;
};

/// Load sweep JSON (``sweep_id``, ``runner``, ``defaults``, ``grid`` and/or ``cells``).
TuneSweepSpec load_tune_sweep(const std::filesystem::path& config_path);

/// Expand explicit ``cells`` or cartesian product of ``grid`` keys.
std::vector<TuneCell> expand_tune_cells(const TuneSweepSpec& spec);

/// Merge defaults + cell args into CLI tokens (``--key value``).
std::vector<std::string> cell_to_cli_args(const TuneJson& defaults, const TuneJson& cell_args);

std::string normalize_cli_flag(const std::string& key);

std::string exe_name_for_runner(const std::string& runner);

std::filesystem::path resolve_runner_exe(const std::string& runner, const std::filesystem::path& exe_dir);

RunProcessResult run_executable_capture(const std::filesystem::path& exe, const std::vector<std::string>& args);

TuneSweepResult run_tune_sweep(const TuneSweepSpec& spec, const std::filesystem::path& exe_dir, int max_cells = -1);

TuneJson tune_sweep_result_to_json(const TuneSweepResult& result);

}  // namespace cypha::bench
