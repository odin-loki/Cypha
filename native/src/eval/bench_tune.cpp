#include "cypha/bench/bench_tune.hpp"

#include "cypha/env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifndef _WIN32
#include <cstdio>
#include <sys/wait.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace cypha::bench {

namespace fs = std::filesystem;

namespace {

std::string quote_arg(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    out += '"';
    return out;
}

std::string json_scalar_to_string(const TuneJson& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_number_integer()) return std::to_string(v.get<long long>());
    if (v.is_number_unsigned()) return std::to_string(v.get<unsigned long long>());
    if (v.is_number_float()) return std::to_string(v.get<double>());
    throw std::runtime_error("tune arg value must be scalar, got: " + v.dump());
}

std::string env_path_for_runner(const std::string& runner) {
    if (runner == "cyphalm_bench_native") return "CYPHALM_BENCH_NATIVE_BIN";
    if (runner == "cypha_bench_run") return "CYPHA_BENCH_RUN_BIN";
    if (runner == "cypha_tune_run") return "CYPHA_TUNE_RUN_BIN";
    return {};
}

void product_recurse(const TuneJson& grid, const std::vector<std::string>& keys, std::size_t depth,
                     TuneJson& current, std::vector<TuneCell>& out) {
    if (depth >= keys.size()) {
        TuneCell cell;
        std::ostringstream id;
        bool first = true;
        for (const auto& k : keys) {
            const std::string val = json_scalar_to_string(current[k]);
            if (!first) id << '_';
            first = false;
            id << k << '=' << val;
        }
        cell.id = id.str();
        cell.args = current;
        out.push_back(std::move(cell));
        return;
    }
    const std::string& key = keys[depth];
    const TuneJson& values = grid.at(key);
    if (!values.is_array()) {
        throw std::runtime_error("tune grid value must be array for key: " + key);
    }
    for (const auto& v : values) {
        current[key] = v;
        product_recurse(grid, keys, depth + 1, current, out);
    }
}

TuneJson parse_runner_stdout(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return TuneJson::object();
    const auto end = text.find_last_not_of(" \t\r\n");
    const std::string trimmed = text.substr(begin, end - begin + 1);
    try {
        return TuneJson::parse(trimmed);
    } catch (const std::exception&) {
        return TuneJson{{"raw_stdout", trimmed}};
    }
}

}  // namespace

std::string normalize_cli_flag(const std::string& key) {
    std::string out;
    out.reserve(key.size());
    for (char c : key) {
        out.push_back(c == '_' ? '-' : c);
    }
    return out;
}

TuneSweepSpec load_tune_sweep(const fs::path& config_path) {
    std::ifstream in(config_path);
    if (!in) {
        throw std::runtime_error("cannot open tune config: " + config_path.string());
    }
    TuneJson j;
    in >> j;
    TuneSweepSpec spec;
    spec.sweep_id = j.value("sweep_id", config_path.stem().string());
    spec.description = j.value("description", "");
    spec.runner = j.value("runner", "cyphalm_bench_native");
    if (j.contains("defaults") && j["defaults"].is_object()) {
        spec.defaults = j["defaults"];
    }
    if (j.contains("grid") && j["grid"].is_object()) {
        spec.grid = j["grid"];
    }
    if (j.contains("cells") && j["cells"].is_array()) {
        for (const auto& c : j["cells"]) {
            TuneCell cell;
            cell.id = c.value("id", "");
            if (c.contains("args") && c["args"].is_object()) {
                cell.args = c["args"];
            } else if (c.is_object()) {
                cell.args = c;
                cell.args.erase("id");
            }
            if (cell.id.empty()) {
                std::ostringstream id;
                bool first = true;
                for (auto it = cell.args.begin(); it != cell.args.end(); ++it) {
                    if (!first) id << '_';
                    first = false;
                    id << it.key() << '=' << json_scalar_to_string(it.value());
                }
                cell.id = id.str();
            }
            spec.cells.push_back(std::move(cell));
        }
    }
    if (spec.cells.empty() && spec.grid.empty()) {
        throw std::runtime_error("tune config must define cells and/or grid");
    }
    return spec;
}

std::vector<TuneCell> expand_tune_cells(const TuneSweepSpec& spec) {
    if (!spec.cells.empty()) return spec.cells;
    std::vector<std::string> keys;
    keys.reserve(spec.grid.size());
    for (auto it = spec.grid.begin(); it != spec.grid.end(); ++it) {
        keys.push_back(it.key());
    }
    std::sort(keys.begin(), keys.end());
    std::vector<TuneCell> out;
    TuneJson current = TuneJson::object();
    product_recurse(spec.grid, keys, 0, current, out);
    return out;
}

std::vector<std::string> cell_to_cli_args(const TuneJson& defaults, const TuneJson& cell_args) {
    TuneJson merged = defaults;
    for (auto it = cell_args.begin(); it != cell_args.end(); ++it) {
        merged[it.key()] = it.value();
    }
    std::vector<std::string> keys;
    keys.reserve(merged.size());
    for (auto it = merged.begin(); it != merged.end(); ++it) {
        keys.push_back(it.key());
    }
    std::sort(keys.begin(), keys.end());
    std::vector<std::string> cli;
    cli.reserve(keys.size() * 2);
    for (const auto& key : keys) {
        const std::string flag = normalize_cli_flag(key);
        const TuneJson& val = merged[key];
        if (val.is_boolean()) {
            if (val.get<bool>()) cli.push_back("--" + flag);
            continue;
        }
        cli.push_back("--" + flag);
        cli.push_back(json_scalar_to_string(val));
    }
    return cli;
}

std::string exe_name_for_runner(const std::string& runner) {
#ifdef _WIN32
    return runner + ".exe";
#else
    return runner;
#endif
}

fs::path resolve_runner_exe(const std::string& runner, const fs::path& exe_dir) {
    const std::string env_key = env_path_for_runner(runner);
    if (!env_key.empty()) {
        if (const std::optional<std::string> raw = cypha::env_get(env_key.c_str()); raw.has_value() && !raw->empty()) {
            const fs::path p(*raw);
            if (fs::is_regular_file(p)) return fs::absolute(p);
        }
    }
    if (!exe_dir.empty()) {
        const fs::path local = exe_dir / exe_name_for_runner(runner);
        if (fs::is_regular_file(local)) return fs::absolute(local);
    }
    return fs::absolute(exe_dir.empty() ? fs::path(exe_name_for_runner(runner)) : exe_dir / exe_name_for_runner(runner));
}

RunProcessResult run_executable_capture(const fs::path& exe, const std::vector<std::string>& args) {
    RunProcessResult result;
#ifdef _WIN32
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
        result.exit_code = 127;
        result.stderr_text = "CreatePipe failed";
        return result;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    std::string cmd = quote_arg(exe.string());
    for (const auto& a : args) {
        cmd += ' ';
        cmd += quote_arg(a);
    }
    std::vector<char> buf(cmd.begin(), cmd.end());
    buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdOutput = write_pipe;
    si.hStdError = write_pipe;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};

    const BOOL ok = CreateProcessA(nullptr, buf.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
    CloseHandle(write_pipe);
    if (!ok) {
        CloseHandle(read_pipe);
        result.exit_code = 127;
        result.stderr_text = "CreateProcess failed";
        return result;
    }

    char chunk[4096];
    DWORD nread = 0;
    while (ReadFile(read_pipe, chunk, sizeof(chunk), &nread, nullptr) && nread > 0) {
        result.stdout_text.append(chunk, chunk + nread);
    }
    CloseHandle(read_pipe);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    result.exit_code = static_cast<int>(code);
#else
    std::ostringstream cmd;
    cmd << quote_arg(exe.string());
    for (const auto& a : args) {
        cmd << ' ' << quote_arg(a);
    }
    cmd << " 2>&1";
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        result.exit_code = 127;
        result.stderr_text = "popen failed";
        return result;
    }
    char chunk[4096];
    while (fgets(chunk, sizeof(chunk), pipe) != nullptr) {
        result.stdout_text += chunk;
    }
    const int status = pclose(pipe);
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else {
        result.exit_code = 127;
    }
#endif
    return result;
}

TuneSweepResult run_tune_sweep(const TuneSweepSpec& spec, const fs::path& exe_dir, int max_cells) {
    const auto cells = expand_tune_cells(spec);
    const fs::path runner_exe = resolve_runner_exe(spec.runner, exe_dir);
    if (!fs::is_regular_file(runner_exe)) {
        throw std::runtime_error("runner executable not found: " + runner_exe.string());
    }

    TuneSweepResult out;
    out.sweep_id = spec.sweep_id;
    out.runner = spec.runner;
    out.defaults = spec.defaults;
    const auto t0 = std::chrono::steady_clock::now();

    int ran = 0;
    for (const auto& cell : cells) {
        if (max_cells >= 0 && ran >= max_cells) break;
        ++ran;

        TuneRunResult row;
        row.cell_id = cell.id;
        row.args = cell.args;
        const auto cli = cell_to_cli_args(spec.defaults, cell.args);
        const auto t_cell = std::chrono::steady_clock::now();
        const auto proc = run_executable_capture(runner_exe, cli);
        row.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cell).count();
        row.exit_code = proc.exit_code;
        row.ok = (proc.exit_code == 0);
        if (!proc.stdout_text.empty()) {
            row.output = parse_runner_stdout(proc.stdout_text);
        }
        if (!row.ok) {
            row.error = proc.stderr_text.empty() ? ("exit_code=" + std::to_string(proc.exit_code)) : proc.stderr_text;
        }
        out.runs.push_back(std::move(row));
    }

    out.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

TuneJson tune_sweep_result_to_json(const TuneSweepResult& result) {
    TuneJson runs = TuneJson::array();
    for (const auto& r : result.runs) {
        TuneJson row = {
            {"cell_id", r.cell_id},
            {"args", r.args},
            {"exit_code", r.exit_code},
            {"seconds", r.seconds},
            {"ok", r.ok},
            {"output", r.output},
        };
        if (!r.error.empty()) row["error"] = r.error;
        runs.push_back(std::move(row));
    }
    return TuneJson{
        {"sweep_id", result.sweep_id},
        {"runner", result.runner},
        {"defaults", result.defaults},
        {"elapsed_s", result.elapsed_s},
        {"runs", runs},
    };
}

}  // namespace cypha::bench
