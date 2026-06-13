// cypha_diagnostics_run — native orchestrator for cypha_diagnostics phases 1–4.
// Runs parity exes + inline cypha_core checks; reports pass/fail (no sklearn).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <nlohmann/json.hpp>

#include "cypha/som/som_encoder.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

struct Args {
    fs::path fixtures = "fixtures";
    fs::path out = "cypha_diagnostics/results";
    fs::path exe_dir;
    std::vector<int> phases{1, 2, 3, 4};
    bool list_only = false;
    bool skip_subprocess = false;
};

struct CheckResult {
    std::string name;
    std::string phase;
    std::string kind;  // "parity_exe" | "inline"
    bool pass{false};
    int exit_code{-1};
    double seconds{0.0};
    std::string detail;
};

void usage() {
    std::cerr
        << "usage: cypha_diagnostics_run [--fixtures DIR] [--out DIR] [--exe-dir DIR]\n"
        << "       [--phases 1,2,3,4] [--list] [--inline-only]\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (k == "--fixtures") {
            a.fixtures = need("--fixtures");
        } else if (k == "--out") {
            a.out = need("--out");
        } else if (k == "--exe-dir") {
            a.exe_dir = need("--exe-dir");
        } else if (k == "--phases") {
            a.phases.clear();
            const std::string v = need("--phases");
            std::stringstream ss(v);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) {
                    a.phases.push_back(std::stoi(tok));
                }
            }
        } else if (k == "--list") {
            a.list_only = true;
        } else if (k == "--inline-only") {
            a.skip_subprocess = true;
        } else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

bool phase_enabled(const Args& a, int p) {
    for (int x : a.phases) {
        if (x == p) {
            return true;
        }
    }
    return false;
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

std::string exe_name(const std::string& base) {
#ifdef _WIN32
    return base + ".exe";
#else
    return base;
#endif
}

std::string quote_arg(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

int run_process(const fs::path& exe, const std::vector<std::string>& args) {
#ifdef _WIN32
    std::string cmd = quote_arg(exe.string());
    for (const auto& a : args) {
        cmd += ' ';
        cmd += quote_arg(a);
    }
    std::vector<char> buf(cmd.begin(), cmd.end());
    buf.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        return 127;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return static_cast<int>(code);
#else
    std::ostringstream cmd;
    cmd << quote_arg(exe.string());
    for (const auto& a : args) {
        cmd << ' ' << quote_arg(a);
    }
    return std::system(cmd.str().c_str());
#endif
}

CheckResult run_parity_exe(const fs::path& exe_dir, const std::string& phase, const std::string& base,
                           const std::vector<std::string>& args) {
    CheckResult r;
    r.phase = phase;
    r.kind = "parity_exe";
    r.name = base;
    const fs::path exe = exe_dir / exe_name(base);
    if (!fs::is_regular_file(exe)) {
        r.pass = false;
        r.exit_code = 127;
        r.detail = "missing executable: " + exe.string();
        return r;
    }

    const auto t0 = std::chrono::steady_clock::now();
    r.exit_code = run_process(exe, args);
    const auto t1 = std::chrono::steady_clock::now();
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.pass = (r.exit_code == 0);
    if (!r.pass) {
        r.detail = "exit_code=" + std::to_string(r.exit_code);
    }
    return r;
}

CheckResult run_som_encoder_smoke() {
    CheckResult r;
    r.phase = "2";
    r.kind = "inline";
    r.name = "som_encoder_smoke";
    try {
        cypha::som::OnlineSOMConfig cfg;
        cfg.k = 4;
        cfg.T = 100;
        cfg.seed = 42;
        cypha::som::OnlineSOMEncoder som(8, cfg);
        std::vector<double> z(8, 0.0);
        for (int i = 0; i < 8; ++i) {
            z[static_cast<std::size_t>(i)] = static_cast<double>(i) * 0.1;
        }
        const auto h1 = som.encode(z, true);
        const auto h2 = som.encode(z, false);
        if (static_cast<int>(h1.size()) != 8 || static_cast<int>(h2.size()) != 8) {
            r.detail = "encoded dim mismatch";
            return r;
        }
        if (som.step_count() != 1) {
            r.detail = "expected one training step";
            return r;
        }
        r.pass = true;
        r.exit_code = 0;
    } catch (const std::exception& ex) {
        r.detail = ex.what();
    }
    return r;
}

struct ParityJob {
    std::string phase;
    std::string exe;
    std::vector<std::string> args;
};

std::vector<ParityJob> build_jobs(const fs::path& fix) {
    return {
        // Phase 1 — baseline establishment (CyphaDIF classify / model hotpaths)
        {"1", "cypha_parity", {fix.string() + "/reference.cypha", fix.string() + "/native_parity.bin"}},
        {"1", "create_model_smoke", {}},
        {"1", "preprocess_train_classify_parity", {fix.string() + "/studio_trainer_preprocess_classify_hotpath"}},
        {"1", "preprocess_train_classify_parity", {fix.string() + "/csv_preprocess_classify_hotpath"}},

        // Phase 2 — encoder quality (preprocessor + RFF)
        {"2", "preprocessor_parity", {fix.string() + "/preprocessor"}},
        {"2", "preprocessor_fit_parity",
         {fix.string() + "/preprocessor_fit", fix.string() + "/preprocessor_fit_no_scale",
          fix.string() + "/preprocessor_fit_rff"}},
        {"2", "regression_rff_parity", {fix.string() + "/rff_regression/sidecar.json"}},

        // Phase 3 — NIG calibration & deliberation
        {"3", "nig_adapt_parity", {}},
        {"3", "gh_infer_deliberation_parity", {fix.string() + "/gh_infer_deliberation"}},

        // Phase 4 — online learning dynamics
        {"4", "train_step_vector_parity", {fix.string() + "/train_step_vector"}},
        {"4", "memory_train_parity", {fix.string() + "/memory_train"}},
        {"4", "memory_train_roundtrip", {fix.string() + "/memory_train"}},
        {"4", "mke_train_step_parity", {fix.string() + "/mke_train_step"}},
        {"4", "kernel_llr_parity", {fix.string() + "/kernel_llr/sidecar.json"}},
        {"4", "batch_llr_parity", {fix.string() + "/batch_llr/sidecar.json"}},
    };
}

void write_json_report(const fs::path& out_dir, const std::vector<CheckResult>& results) {
    fs::create_directories(out_dir);
    Json by_phase = Json::object();
    int passed = 0;
    int failed = 0;
    for (const auto& r : results) {
        Json item = {{"name", r.name},
                     {"kind", r.kind},
                     {"pass", r.pass},
                     {"exit_code", r.exit_code},
                     {"seconds", r.seconds}};
        if (!r.detail.empty()) {
            item["detail"] = r.detail;
        }
        by_phase[r.phase].push_back(item);
        if (r.pass) {
            ++passed;
        } else {
            ++failed;
        }
    }
    const Json summary = {{"passed", passed},
                          {"failed", failed},
                          {"total", passed + failed},
                          {"all_pass", failed == 0},
                          {"orchestrator", "cypha_diagnostics_run"},
                          {"note", "sklearn baselines omitted; native parity + inline checks only"}};
    const Json root = {{"summary", summary}, {"phases", by_phase}};
    const fs::path out_path = out_dir / "native_diagnostics_summary.json";
    std::ofstream f(out_path);
    if (!f) {
        throw std::runtime_error("cannot write " + out_path.string());
    }
    f << root.dump(2) << '\n';
    std::cout << "[DIAG] Wrote " << out_path << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const fs::path fix = fs::absolute(args.fixtures);
        const fs::path exe_dir = resolve_exe_dir(args, argv);

        std::cout << "[DIAG] Native diagnostics (phases ";
        for (std::size_t i = 0; i < args.phases.size(); ++i) {
            if (i) {
                std::cout << ',';
            }
            std::cout << args.phases[i];
        }
        std::cout << ")\n";
        std::cout << "[DIAG] fixtures=" << fix << " exe_dir=" << exe_dir << '\n';

        const std::vector<ParityJob> jobs = build_jobs(fix);
        if (args.list_only) {
            for (const auto& j : jobs) {
                if (!phase_enabled(args, std::stoi(j.phase))) {
                    continue;
                }
                std::cout << "phase " << j.phase << "  " << j.exe;
                for (const auto& a : j.args) {
                    std::cout << ' ' << a;
                }
                std::cout << '\n';
            }
            std::cout << "phase 2  som_encoder_smoke (inline)\n";
            return 0;
        }

        std::vector<CheckResult> results;
        for (const auto& j : jobs) {
            if (!phase_enabled(args, std::stoi(j.phase))) {
                continue;
            }
            if (args.skip_subprocess) {
                continue;
            }
            std::cout << "\n== phase " << j.phase << " :: " << j.exe << " ==\n";
            CheckResult r = run_parity_exe(exe_dir, j.phase, j.exe, j.args);
            std::cout << (r.pass ? "[PASS] " : "[FAIL] ") << r.name;
            if (!r.detail.empty()) {
                std::cout << " — " << r.detail;
            }
            std::cout << " (" << r.seconds << "s)\n";
            results.push_back(std::move(r));
        }

        if (phase_enabled(args, 2)) {
            std::cout << "\n== phase 2 :: som_encoder_smoke (inline) ==\n";
            CheckResult r = run_som_encoder_smoke();
            std::cout << (r.pass ? "[PASS] " : "[FAIL] ") << r.name;
            if (!r.detail.empty()) {
                std::cout << " — " << r.detail;
            }
            std::cout << '\n';
            results.push_back(std::move(r));
        }

        write_json_report(fs::absolute(args.out), results);

        int n_fail = 0;
        for (const auto& r : results) {
            if (!r.pass) {
                ++n_fail;
            }
        }
        if (n_fail == 0) {
            std::cout << "\n[DIAGNOSTICS COMPLETE] all " << results.size() << " checks passed\n";
            return 0;
        }
        std::cout << "\n[DIAGNOSTICS COMPLETE] " << n_fail << " of " << results.size() << " checks failed\n";
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "cypha_diagnostics_run error: " << ex.what() << '\n';
        return 2;
    }
}
