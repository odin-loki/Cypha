/// CTest smoke for MG4 memorization canary (finite recall_rate, >=1 canary recalled).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_tune.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

fs::path exe_dir_from_argv(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--exe-dir") {
            return fs::absolute(argv[i + 1]);
        }
    }
    return fs::current_path();
}

Json load_json(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read " + path.string());
    }
    Json j;
    in >> j;
    return j;
}

void validate_report(const Json& report) {
    if (!report.contains("recall_rate") || !report["recall_rate"].is_number()) {
        throw std::runtime_error("report missing recall_rate");
    }
    const double recall = report["recall_rate"].get<double>();
    if (!std::isfinite(recall) || recall < 0.0 || recall > 1.0) {
        throw std::runtime_error("recall_rate out of range");
    }
    if (!report.contains("n_recalled") || !report["n_recalled"].is_number_integer()) {
        throw std::runtime_error("report missing n_recalled");
    }
    if (!report.contains("canaries") || !report["canaries"].is_array()) {
        throw std::runtime_error("report missing canaries array");
    }
    if (report["canaries"].size() < 1) {
        throw std::runtime_error("canaries array empty");
    }
    const int n_recalled = report["n_recalled"].get<int>();
    if (n_recalled < 1) {
        throw std::runtime_error("smoke requires >=1 canary recalled");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const fs::path exe_dir = exe_dir_from_argv(argc, argv);
        const fs::path canary_exe =
            cypha::bench::resolve_runner_exe("cyphalm_memorization_canary", exe_dir);
        if (!fs::is_regular_file(canary_exe)) {
            std::fprintf(stderr, "missing cyphalm_memorization_canary: %s\n",
                         canary_exe.string().c_str());
            return 1;
        }

        const fs::path out =
            fs::temp_directory_path() / "cypha_memorization_canary_smoke.json";
        const std::vector<std::string> cli = {
            "--out", out.string(),
            "--seed", "42",
        };
        const cypha::bench::RunProcessResult proc =
            cypha::bench::run_executable_capture(canary_exe, cli);
        if (proc.exit_code != 0) {
            std::fprintf(stderr, "canary runner exit=%d stderr=%s\n", proc.exit_code,
                         proc.stderr_text.c_str());
            return 1;
        }

        const Json report = load_json(out);
        validate_report(report);

        std::printf("memorization_canary_smoke: recall_rate=%.4f (%d/%d) PASS\n",
                    report["recall_rate"].get<double>(), report["n_recalled"].get<int>(),
                    report["n_canaries"].get<int>());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "memorization_canary_smoke: %s\n", e.what());
        return 1;
    }
}
