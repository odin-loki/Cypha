/// CTest smoke for MG3 needle-in-haystack (finite recall_rate, >=1 depth recalled).
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
    if (!report.contains("depths") || !report["depths"].is_array()) {
        throw std::runtime_error("report missing depths array");
    }
    if (report["depths"].size() < 1) {
        throw std::runtime_error("depths array empty");
    }
    bool any_token_recall = false;
    for (const auto& row : report["depths"]) {
        if (!row.contains("bpc_answer") || row["bpc_answer"].is_null()) {
            throw std::runtime_error("depth missing bpc_answer");
        }
        const double bpc = row["bpc_answer"].get<double>();
        if (!std::isfinite(bpc)) {
            throw std::runtime_error("non-finite bpc_answer");
        }
        if (!row.contains("token_recall") || !row["token_recall"].is_number()) {
            throw std::runtime_error("depth missing token_recall");
        }
        if (row["token_recall"].get<double>() > 0.0) {
            any_token_recall = true;
        }
    }
    if (!any_token_recall) {
        throw std::runtime_error("smoke requires >0 token_recall on at least one depth");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const fs::path exe_dir = exe_dir_from_argv(argc, argv);
        const fs::path runner_exe =
            cypha::bench::resolve_runner_exe("cyphalm_needle_haystack", exe_dir);
        if (!fs::is_regular_file(runner_exe)) {
            std::fprintf(stderr, "missing cyphalm_needle_haystack: %s\n", runner_exe.string().c_str());
            return 1;
        }

        const fs::path out = fs::temp_directory_path() / "cypha_needle_haystack_smoke.json";
        // MG3 warm-up A/B (2026-07-17): --context-warmup lifted recall_rate 0.05→0.35 (≥0.1);
        // keep default-on for smoke.
        const std::vector<std::string> cli = {
            "--out", out.string(),
            "--seed", "42",
            "--context-warmup",
        };
        const cypha::bench::RunProcessResult proc =
            cypha::bench::run_executable_capture(runner_exe, cli);
        if (proc.exit_code != 0) {
            std::fprintf(stderr, "needle runner exit=%d stderr=%s\n", proc.exit_code,
                         proc.stderr_text.c_str());
            return 1;
        }

        const Json report = load_json(out);
        validate_report(report);

        std::printf("needle_haystack_smoke: recall_rate=%.4f (%d/%d) PASS\n",
                    report["recall_rate"].get<double>(), report["n_recalled"].get<int>(),
                    report["n_depths"].get<int>());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "needle_haystack_smoke: %s\n", e.what());
        return 1;
    }
}
