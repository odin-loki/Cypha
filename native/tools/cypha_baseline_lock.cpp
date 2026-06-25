// cypha_baseline_lock — run bench subprocesses and merge BPC into bench/BASELINE_LOCK.json.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/wait.h>
#endif

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

enum class RunKind { D17, D17Math, D21, CellSweep, All };

struct Args {
    RunKind run = RunKind::D17;
    int n_train = 300000;
    int n_eval = 2000;
    int threads = 1;
    bool fast = false;
    bool medium = false;
    bool production = false;
    bool math_integration = false;
    bool n_train_explicit = false;
    bool n_eval_explicit = false;
    fs::path lock_file;
    fs::path exe_dir;
    fs::path output_dir;
};

void usage() {
    std::cerr
        << "usage: cypha_baseline_lock --run {d17,d17-math,d21,cell-sweep,all}\n"
        << "       [--n-train N] [--n-eval M] [--threads T] [--fast] [--medium] [--production]\n"
        << "       [--lock-file PATH] [--exe-dir DIR] [--output-dir DIR]\n";
}

RunKind parse_run_kind(const std::string& s) {
    if (s == "d17") return RunKind::D17;
    if (s == "d17-math") return RunKind::D17Math;
    if (s == "d21") return RunKind::D21;
    if (s == "cell-sweep") return RunKind::CellSweep;
    if (s == "all") return RunKind::All;
    throw std::runtime_error("unknown --run: " + s +
                             " (expected d17, d17-math, d21, cell-sweep, or all)");
}

Args parse_args(int argc, char** argv) {
    Args a;
    bool run_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (k == "--run") {
            a.run = parse_run_kind(need("--run"));
            run_set = true;
        } else if (k == "--n-train") {
            a.n_train = std::stoi(need("--n-train"));
            a.n_train_explicit = true;
        } else if (k == "--n-eval") {
            a.n_eval = std::stoi(need("--n-eval"));
            a.n_eval_explicit = true;
        } else if (k == "--threads") {
            a.threads = std::stoi(need("--threads"));
        } else if (k == "--fast") {
            a.fast = true;
        } else if (k == "--medium") {
            a.medium = true;
        } else if (k == "--production") {
            a.production = true;
        } else if (k == "--lock-file") {
            a.lock_file = need("--lock-file");
        } else if (k == "--exe-dir") {
            a.exe_dir = need("--exe-dir");
        } else if (k == "--output-dir") {
            a.output_dir = need("--output-dir");
        } else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    if (!run_set) {
        throw std::runtime_error("missing required --run");
    }
    if ((a.fast && a.medium) || (a.fast && a.production) || (a.medium && a.production)) {
        throw std::runtime_error("cannot combine --fast, --medium, and --production");
    }
    if (a.fast) {
        if (!a.n_train_explicit) a.n_train = 200;
        if (!a.n_eval_explicit) a.n_eval = 64;
    } else if (a.medium) {
        if (!a.n_train_explicit) a.n_train = 5000;
        if (!a.n_eval_explicit) a.n_eval = 256;
    } else     if (a.production) {
        if (!a.n_train_explicit) a.n_train = 300000;
        if (!a.n_eval_explicit) a.n_eval = 2000;
    }
    if (a.output_dir.empty() && (a.run == RunKind::CellSweep || a.run == RunKind::All)) {
        a.output_dir = cypha::bench::results_dir() / "cell_sweep";
    }
    if (a.run == RunKind::D17Math) {
        a.math_integration = true;
    }
    if (cypha::bench::bench_env_truthy("CYPHA_OVERNIGHT_MATH_INTEGRATION")) {
        a.math_integration = true;
    }
    return a;
}

std::string run_status_label(const Args& args) {
    if (args.fast) return "fast_smoke";
    if (args.medium) return "medium_smoke";
    if (args.production) return "production";
    return "completed";
}

void set_env_var(const char* key, const char* value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

std::string quote_shell(const std::string& s) {
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

std::string exe_name(const std::string& base) {
#ifdef _WIN32
    return base + ".exe";
#else
    return base;
#endif
}

struct ProcessResult {
    int exit_code{127};
    std::string stdout_text;
};

ProcessResult run_capture(const fs::path& exe, const std::vector<std::string>& args,
                          const fs::path& work_dir) {
    ProcessResult r;
#ifdef _WIN32
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
        throw std::runtime_error("CreatePipe failed");
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    std::string cmd = quote_shell(exe.string());
    for (const auto& a : args) {
        cmd += ' ';
        cmd += quote_shell(a);
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

    const std::string work = work_dir.empty() ? exe.parent_path().string() : work_dir.string();
    const BOOL ok =
        CreateProcessA(nullptr, buf.data(), nullptr, nullptr, TRUE, 0, nullptr,
                       work.empty() ? nullptr : work.c_str(), &si, &pi);
    CloseHandle(write_pipe);
    if (!ok) {
        CloseHandle(read_pipe);
        throw std::runtime_error("CreateProcess failed for " + exe.string());
    }

    char chunk[4096];
    DWORD nread = 0;
    while (ReadFile(read_pipe, chunk, sizeof(chunk), &nread, nullptr) && nread > 0) {
        r.stdout_text.append(chunk, chunk + nread);
    }
    CloseHandle(read_pipe);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    r.exit_code = static_cast<int>(code);
#else
    std::ostringstream cmd;
    if (!work_dir.empty()) {
        cmd << "cd " << quote_shell(fs::absolute(work_dir).string()) << " && ";
    }
    cmd << quote_shell(fs::absolute(exe).string());
    for (const auto& a : args) {
        cmd << ' ' << quote_shell(a);
    }
    cmd << " 2>&1";
    FILE* fp = popen(cmd.str().c_str(), "r");
    if (fp == nullptr) {
        throw std::runtime_error("popen failed for " + exe.string());
    }
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), fp) != nullptr) {
        r.stdout_text += buf;
    }
    const int status = pclose(fp);
    if (WIFEXITED(status)) {
        r.exit_code = WEXITSTATUS(status);
    }
#endif
    return r;
}

fs::path resolve_exe_dir(const Args& a, char** argv) {
    if (!a.exe_dir.empty()) {
        return fs::absolute(a.exe_dir);
    }
    if (argv && argv[0]) {
        const fs::path self = fs::absolute(argv[0]);
        if (self.has_parent_path()) {
            return self.parent_path();
        }
    }
    return fs::current_path();
}

std::string iso_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

Json load_lock_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read lock file: " + path.string());
    }
    Json j;
    in >> j;
    return j;
}

void write_lock_file(const fs::path& path, const Json& j) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write lock file: " + path.string());
    }
    out << j.dump(2) << '\n';
}

std::string extract_json_blob(const std::string& mixed) {
    const std::size_t last_brace = mixed.rfind("\n{");
    if (last_brace != std::string::npos) {
        return mixed.substr(last_brace + 1);
    }
    const std::size_t start = mixed.find('{');
    const std::size_t end = mixed.rfind('}');
    if (start != std::string::npos && end != std::string::npos && end >= start) {
        return mixed.substr(start, end - start + 1);
    }
    return mixed;
}

double extract_bpc_bench(const Json& j) {
    if (!j.contains("bpc") || j["bpc"].is_null()) {
        throw std::runtime_error("bench JSON missing bpc");
    }
    const double bpc = j["bpc"].get<double>();
    if (std::isnan(bpc)) {
        throw std::runtime_error("bench JSON bpc is NaN");
    }
    return bpc;
}

double extract_kappa(const Json& j) {
    if (j.contains("intelligence_profile") && j["intelligence_profile"].is_object()) {
        const Json& ip = j["intelligence_profile"];
        if (ip.contains("criticality_score") && ip["criticality_score"].is_number()) {
            const double k = ip["criticality_score"].get<double>();
            if (std::isfinite(k)) {
                return k;
            }
        }
        if (ip.contains("profile_completeness") && ip["profile_completeness"].is_object()) {
            const Json& pc = ip["profile_completeness"];
            if (pc.contains("kappa") && pc["kappa"].is_number()) {
                const double k = pc["kappa"].get<double>();
                if (std::isfinite(k)) {
                    return k;
                }
            }
        }
    }
    if (j.contains("profile_completeness") && j["profile_completeness"].is_object()) {
        const Json& pc = j["profile_completeness"];
        if (pc.contains("kappa") && pc["kappa"].is_number()) {
            const double k = pc["kappa"].get<double>();
            if (std::isfinite(k)) {
                return k;
            }
        }
    }
    throw std::runtime_error("bench JSON missing kappa");
}

struct CellSweepSummary {
    double bpc;
    double b2_bpc{std::numeric_limits<double>::quiet_NaN()};
    int variant_count{0};
    std::string output_dir;
    std::string best_pareto_id;
    Json best_pareto = Json::object();
};

CellSweepSummary extract_cell_sweep_summary(const Json& j) {
    CellSweepSummary s;
    s.variant_count = j.value("variant_count", 0);
    if (j.contains("output_dir") && j["output_dir"].is_string()) {
        s.output_dir = j["output_dir"].get<std::string>();
    }
    double pareto_bpc = std::numeric_limits<double>::quiet_NaN();
    if (j.contains("best_pareto_variant") && j["best_pareto_variant"].is_object()) {
        const Json& bp = j["best_pareto_variant"];
        s.best_pareto = bp;
        s.best_pareto_id = bp.value("id", "");
        if (bp.contains("bpc") && !bp["bpc"].is_null() && bp["bpc"].is_number()) {
            pareto_bpc = bp["bpc"].get<double>();
        }
    } else if (j.contains("pareto_ranked_variants") && j["pareto_ranked_variants"].is_array() &&
               !j["pareto_ranked_variants"].empty()) {
        const Json& first = j["pareto_ranked_variants"].front();
        if (first.is_object()) {
            s.best_pareto = first;
            s.best_pareto_id = first.value("id", "");
            if (first.contains("bpc") && !first["bpc"].is_null() && first["bpc"].is_number()) {
                pareto_bpc = first["bpc"].get<double>();
            }
        }
    }
    if (!j.contains("results") || !j["results"].is_array()) {
        throw std::runtime_error("cell sweep JSON missing results array");
    }
    bool found_b2 = false;
    for (const auto& row : j["results"]) {
        if (row.value("id", "") == "B2" && row.contains("bpc") && !row["bpc"].is_null()) {
            s.b2_bpc = row["bpc"].get<double>();
            found_b2 = true;
            break;
        }
    }
    if (std::isfinite(pareto_bpc)) {
        s.bpc = pareto_bpc;
    } else if (found_b2) {
        s.bpc = s.b2_bpc;
    } else {
        for (const auto& row : j["results"]) {
            if (row.contains("bpc") && !row["bpc"].is_null()) {
                s.bpc = row["bpc"].get<double>();
                break;
            }
        }
    }
    if (std::isnan(s.bpc)) {
        throw std::runtime_error("cell sweep JSON has no usable bpc");
    }
    return s;
}

void apply_bench_env(const Args& args) {
    if (args.fast) {
        set_env_var("CYPHA_BENCH_FAST", "1");
    }
    set_env_var("CYPHA_BENCH_FULL_CORPUS", "1");
    set_env_var("CYPHA_BENCH_OVERNIGHT", "1");
    set_env_var("CYPHA_BENCH_FULL_N_TRAIN", std::to_string(args.n_train).c_str());
}

Json make_env_snapshot(const Args& args) {
    Json env = {{"CYPHA_BENCH_FULL_CORPUS", "1"}, {"CYPHA_BENCH_OVERNIGHT", "1"}};
    if (args.fast) {
        env["CYPHA_BENCH_FAST"] = "1";
    }
    env["CYPHA_BENCH_FULL_N_TRAIN"] = std::to_string(args.n_train);
    return env;
}

ProcessResult run_d17_bench(const fs::path& exe_dir, const Args& args) {
    apply_bench_env(args);
    const fs::path exe = exe_dir / exe_name("cyphalm_bench_native");
    if (!fs::is_regular_file(exe)) {
        throw std::runtime_error("missing executable: " + exe.string());
    }
    return run_capture(exe, {"--profile", "d17", "--mode", "hybrid", "--overnight", "--n-train",
                             std::to_string(args.n_train), "--n-eval", std::to_string(args.n_eval),
                             "--threads", std::to_string(args.threads)},
                     exe_dir);
}

ProcessResult run_d17_math_bench(const fs::path& exe_dir, const Args& args, bool math_integration) {
    apply_bench_env(args);
    const fs::path exe = exe_dir / exe_name("cyphalm_bench_native");
    if (!fs::is_regular_file(exe)) {
        throw std::runtime_error("missing executable: " + exe.string());
    }
    std::vector<std::string> bench_args{"--profile",       "d17",
                                        "--mode",          "hybrid",
                                        "--intelligence-profile",
                                        "--n-train",       std::to_string(args.n_train),
                                        "--n-eval",        std::to_string(args.n_eval),
                                        "--threads",       std::to_string(args.threads)};
    if (math_integration) {
        bench_args.push_back("--math-integration");
    }
    return run_capture(exe, bench_args, exe_dir);
}

ProcessResult run_d21_bench(const fs::path& exe_dir, const Args& args) {
    apply_bench_env(args);
    const fs::path exe = exe_dir / exe_name("cyphalm_bench_native");
    if (!fs::is_regular_file(exe)) {
        throw std::runtime_error("missing executable: " + exe.string());
    }
    return run_capture(exe, {"--profile", "d21", "--mode", "rpsm", "--overnight", "--n-train",
                             std::to_string(args.n_train), "--n-eval", std::to_string(args.n_eval),
                             "--threads", std::to_string(args.threads)},
                     exe_dir);
}

ProcessResult run_cell_sweep(const fs::path& exe_dir, const Args& args) {
    apply_bench_env(args);
    const fs::path exe = exe_dir / exe_name("cypha_cell_hypothesis_sweep");
    if (!fs::is_regular_file(exe)) {
        throw std::runtime_error("missing executable: " + exe.string());
    }
    std::vector<std::string> sweep_args;
    if (args.fast) {
        sweep_args.push_back("--overnight-sweep-smoke");
    } else {
        sweep_args.push_back("--overnight-sweep");
    }
    sweep_args.push_back("--profile");
    sweep_args.push_back("d17");
    sweep_args.push_back("--n-train");
    sweep_args.push_back(std::to_string(args.n_train));
    sweep_args.push_back("--n-eval");
    sweep_args.push_back(std::to_string(args.n_eval));
    sweep_args.push_back("--threads");
    sweep_args.push_back(std::to_string(args.threads));
    sweep_args.push_back("--intelligence-profile");
    if (args.math_integration) {
        sweep_args.push_back("--math-integration");
    }
    if (!args.output_dir.empty()) {
        sweep_args.push_back("--output-dir");
        sweep_args.push_back(fs::absolute(args.output_dir).string());
    }
    return run_capture(exe, sweep_args, exe_dir);
}

void merge_overnight_results(Json& lock, const Args& args, double bpc, const Json& bench_json) {
    Json& section = lock["overnight_results"];
    section["status"] = run_status_label(args);
    section["n_train"] = args.n_train;
    section["n_eval"] = args.n_eval;
    section["bpc"] = bpc;
    section["run_at"] = iso_timestamp_now();
    section["env"] = make_env_snapshot(args);
    section["profile"] = "d17";
    section["mode"] = "hybrid";
    section["runner"] = "cyphalm_bench_native --overnight";
    if (bench_json.contains("context_mode")) {
        section["context_mode"] = bench_json["context_mode"];
    }
    if (bench_json.contains("corpus")) {
        section["corpus"] = bench_json["corpus"];
    }
    if (bench_json.contains("synthetic")) {
        section["synthetic"] = bench_json["synthetic"];
    }
}

void merge_cell_sweep_results(Json& lock, const Args& args, double bpc,
                              const CellSweepSummary& cell_summary) {
    if (!lock.contains("cell_sweep_results")) {
        lock["cell_sweep_results"] = Json::object();
    }
    Json& section = lock["cell_sweep_results"];
    section["status"] = run_status_label(args);
    section["profile"] = "d17";
    section["mode"] = "cell-sweep";
    section["n_train"] = args.n_train;
    section["n_eval"] = args.n_eval;
    section["bpc"] = bpc;
    section["run_at"] = iso_timestamp_now();
    section["runner"] = args.fast
                            ? (args.math_integration
                                   ? "cypha_cell_hypothesis_sweep --overnight-sweep-smoke "
                                     "--intelligence-profile --math-integration"
                                   : "cypha_cell_hypothesis_sweep --overnight-sweep-smoke "
                                     "--intelligence-profile")
                            : (args.math_integration
                                   ? "cypha_cell_hypothesis_sweep --overnight-sweep "
                                     "--intelligence-profile --math-integration"
                                   : "cypha_cell_hypothesis_sweep --overnight-sweep "
                                     "--intelligence-profile");
    if (args.math_integration) {
        section["math_integration_enabled"] = true;
    }
    section["env"] = make_env_snapshot(args);
    section["variant_count"] = cell_summary.variant_count;
    if (!std::isnan(cell_summary.b2_bpc)) {
        section["b2_bpc"] = cell_summary.b2_bpc;
    }
    if (!cell_summary.best_pareto.is_null() && cell_summary.best_pareto.is_object()) {
        section["best_pareto_variant"] = cell_summary.best_pareto;
    } else if (!cell_summary.best_pareto_id.empty()) {
        section["best_pareto_variant"] = cell_summary.best_pareto_id;
    }
    if (!cell_summary.output_dir.empty()) {
        section["artifact_path"] = cell_summary.output_dir;
    }
}

struct MathArmResult {
    double bpc{0.0};
    double kappa{0.0};
    Json bench_json;
};

MathArmResult parse_math_arm(const Json& bench_json) {
    MathArmResult arm;
    arm.bench_json = bench_json;
    arm.bpc = extract_bpc_bench(bench_json);
    arm.kappa = extract_kappa(bench_json);
    return arm;
}

ProcessResult run_math_bench_arm(const fs::path& exe_dir, const Args& args, bool math_integration,
                                 const char* label) {
    ProcessResult proc = run_d17_math_bench(exe_dir, args, math_integration);
    if (proc.exit_code != 0) {
        std::cerr << "cypha_baseline_lock: subprocess exit=" << proc.exit_code << " (" << label
                  << ")\n";
        if (!proc.stdout_text.empty()) {
            std::cerr << proc.stdout_text << "\n";
        }
        throw std::runtime_error(std::string(label) + " exit=" + std::to_string(proc.exit_code));
    }
    if (proc.stdout_text.empty()) {
        throw std::runtime_error(std::string(label) + " produced no stdout");
    }
    return proc;
}

Json run_d17_math(const fs::path& exe_dir, const Args& args, MathArmResult& baseline_out,
                  MathArmResult& math_out) {
    const ProcessResult baseline_proc =
        run_math_bench_arm(exe_dir, args, false, "cyphalm_bench_native baseline");
    baseline_out = parse_math_arm(Json::parse(extract_json_blob(baseline_proc.stdout_text)));

    const ProcessResult math_proc =
        run_math_bench_arm(exe_dir, args, true, "cyphalm_bench_native --math-integration");
    math_out = parse_math_arm(Json::parse(extract_json_blob(math_proc.stdout_text)));

    return Json{{"run", "d17-math"},
                {"n_train", args.n_train},
                {"n_eval", args.n_eval},
                {"fast", args.fast},
                {"medium", args.medium},
                {"production", args.production},
                {"status", run_status_label(args)},
                {"baseline", Json{{"bpc", baseline_out.bpc}, {"kappa", baseline_out.kappa}}},
                {"math_integration",
                 Json{{"bpc", math_out.bpc}, {"kappa", math_out.kappa}}}};
}

void merge_math_integration_results(Json& lock, const Args& args, const MathArmResult& baseline,
                                    const MathArmResult& math_arm) {
    if (!lock.contains("math_integration_results")) {
        lock["math_integration_results"] = Json::object();
    }
    Json& section = lock["math_integration_results"];
    section["baseline"] = Json{{"bpc", baseline.bpc},
                               {"kappa", baseline.kappa},
                               {"profile_complete", true}};
    section["math_integration"] = Json{{"bpc", math_arm.bpc},
                                         {"kappa", math_arm.kappa},
                                         {"profile_complete", true}};
    section["n_train"] = args.n_train;
    section["n_eval"] = args.n_eval;
    section["status"] = run_status_label(args);
    section["run_at"] = iso_timestamp_now();
}

void merge_rpsm_results(Json& lock, const Args& args, double bpc, const Json& bench_json) {
    if (!lock.contains("rpsm_results")) {
        lock["rpsm_results"] = Json::object();
    }
    Json& section = lock["rpsm_results"];
    section["status"] = run_status_label(args);
    section["profile"] = "d21";
    section["mode"] = "rpsm";
    section["n_train"] = args.n_train;
    section["n_eval"] = args.n_eval;
    section["bpc"] = bpc;
    section["run_at"] = iso_timestamp_now();
    section["runner"] = "cyphalm_bench_native --overnight --mode rpsm";
    section["env"] = make_env_snapshot(args);
    if (bench_json.contains("corpus")) {
        section["corpus"] = bench_json["corpus"];
    }
    if (bench_json.contains("synthetic")) {
        section["synthetic"] = bench_json["synthetic"];
    }
}

std::string run_kind_name(RunKind kind) {
    switch (kind) {
        case RunKind::D17:
            return "d17";
        case RunKind::D17Math:
            return "d17-math";
        case RunKind::D21:
            return "d21";
        case RunKind::CellSweep:
            return "cell-sweep";
        case RunKind::All:
            return "all";
    }
    return "unknown";
}

struct RunOutcome {
    Json report;
    Json bench_json;
    double bpc{0.0};
    CellSweepSummary cell_summary;
    bool has_cell_summary{false};
};

RunOutcome execute_run_kind(const Args& args, RunKind kind, const fs::path& exe_dir) {
    Args step = args;
    step.run = kind;

    ProcessResult proc;
    if (kind == RunKind::D17) {
        proc = run_d17_bench(exe_dir, step);
    } else if (kind == RunKind::D21) {
        proc = run_d21_bench(exe_dir, step);
    } else {
        proc = run_cell_sweep(exe_dir, step);
    }

    if (proc.exit_code != 0) {
        std::cerr << "cypha_baseline_lock: subprocess exit=" << proc.exit_code
                  << " (run=" << run_kind_name(kind) << ")\n";
        if (!proc.stdout_text.empty()) {
            std::cerr << proc.stdout_text << "\n";
        }
        throw std::runtime_error("subprocess failed for run=" + run_kind_name(kind) + " exit=" +
                                 std::to_string(proc.exit_code));
    }
    if (proc.stdout_text.empty()) {
        throw std::runtime_error("subprocess produced no stdout for run=" + run_kind_name(kind));
    }

    RunOutcome outcome;
    outcome.bench_json = Json::parse(extract_json_blob(proc.stdout_text));
    outcome.report = {{"run", run_kind_name(kind)},
                        {"n_train", step.n_train},
                        {"n_eval", step.n_eval},
                        {"fast", step.fast},
                        {"medium", step.medium},
                        {"production", step.production},
                        {"status", run_status_label(step)}};

    if (kind == RunKind::CellSweep) {
        outcome.cell_summary = extract_cell_sweep_summary(outcome.bench_json);
        outcome.has_cell_summary = true;
        outcome.bpc = outcome.cell_summary.bpc;
        outcome.report["bpc"] = outcome.bpc;
        outcome.report["b2_bpc"] = outcome.cell_summary.b2_bpc;
        outcome.report["variant_count"] = outcome.cell_summary.variant_count;
    } else {
        outcome.bpc = extract_bpc_bench(outcome.bench_json);
        outcome.report["bpc"] = outcome.bpc;
    }
    return outcome;
}

void merge_run_result(Json& lock, const Args& args, RunKind kind, const RunOutcome& outcome) {
    if (kind == RunKind::D21) {
        merge_rpsm_results(lock, args, outcome.bpc, outcome.bench_json);
        return;
    }
    if (kind == RunKind::CellSweep) {
        merge_cell_sweep_results(lock, args, outcome.bpc, outcome.cell_summary);
        return;
    }
    merge_overnight_results(lock, args, outcome.bpc, outcome.bench_json);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        const fs::path exe_dir = resolve_exe_dir(args, argv);
        const fs::path lock_path =
            args.lock_file.empty() ? cypha::bench::bench_root() / "BASELINE_LOCK.json"
                                   : fs::absolute(args.lock_file);

        Json lock = load_lock_file(lock_path);
        Json report;

        if (args.run == RunKind::D17Math) {
            MathArmResult baseline;
            MathArmResult math_arm;
            report = run_d17_math(exe_dir, args, baseline, math_arm);
            report["lock_file"] = lock_path.string();
            merge_math_integration_results(lock, args, baseline, math_arm);
            write_lock_file(lock_path, lock);
        } else {
            Json reports = Json::array();
            const std::vector<RunKind> sequence =
                args.run == RunKind::All
                    ? std::vector<RunKind>{RunKind::D17, RunKind::D21, RunKind::CellSweep}
                    : std::vector<RunKind>{args.run};

            for (RunKind kind : sequence) {
                const RunOutcome outcome = execute_run_kind(args, kind, exe_dir);
                merge_run_result(lock, args, kind, outcome);
                reports.push_back(outcome.report);
            }
            write_lock_file(lock_path, lock);

            report = {{"run", run_kind_name(args.run)},
                      {"lock_file", lock_path.string()},
                      {"n_train", args.n_train},
                      {"n_eval", args.n_eval},
                      {"fast", args.fast},
                      {"medium", args.medium},
                      {"production", args.production},
                      {"status", run_status_label(args)},
                      {"runs", reports}};
            if (reports.size() == 1 && reports[0].contains("bpc")) {
                report["bpc"] = reports[0]["bpc"];
            }
        }
        std::cout << report.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cypha_baseline_lock: " << e.what() << "\n";
        usage();
        return 1;
    }
}
