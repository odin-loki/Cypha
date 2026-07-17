/// CTest smoke for MS2 algebraic fingerprint JSON (finite vector + spectrum_position).
#include <cmath>
#include <cstdio>
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

void validate_fingerprint_block(const Json& block, const char* label) {
    if (!block.contains("vector") || !block["vector"].is_array()) {
        throw std::runtime_error(std::string(label) + " missing vector");
    }
    if (block["vector"].size() < 4) {
        throw std::runtime_error(std::string(label) + " vector too short");
    }
    for (const auto& v : block["vector"]) {
        if (!v.is_number()) {
            throw std::runtime_error(std::string(label) + " non-numeric vector entry");
        }
        const double x = v.get<double>();
        if (!std::isfinite(x)) {
            throw std::runtime_error(std::string(label) + " non-finite vector entry");
        }
    }
    if (!block.contains("spectrum_position") || !block["spectrum_position"].is_number()) {
        throw std::runtime_error(std::string(label) + " missing spectrum_position");
    }
    const double sp = block["spectrum_position"].get<double>();
    if (!std::isfinite(sp)) {
        throw std::runtime_error(std::string(label) + " non-finite spectrum_position");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const fs::path exe_dir = exe_dir_from_argv(argc, argv);
        const fs::path runner_exe =
            cypha::bench::resolve_runner_exe("cyphalm_algebraic_fingerprint", exe_dir);
        if (!fs::is_regular_file(runner_exe)) {
            std::fprintf(stderr, "missing cyphalm_algebraic_fingerprint: %s\n",
                         runner_exe.string().c_str());
            return 1;
        }

        const fs::path out = fs::temp_directory_path() / "cypha_algebraic_fingerprint_smoke.json";
        const std::vector<std::string> cli = {
            "--out", out.string(),
            "--seed", "42",
        };
        const cypha::bench::RunProcessResult proc =
            cypha::bench::run_executable_capture(runner_exe, cli);
        if (proc.exit_code != 0) {
            std::fprintf(stderr, "runner exit=%d stderr=%s\n", proc.exit_code,
                         proc.stderr_text.c_str());
            return 1;
        }

        const Json report = load_json(out);
        validate_fingerprint_block(report.at("generated"), "generated");
        validate_fingerprint_block(report.at("scored_train"), "scored_train");

        std::printf("algebraic_fingerprint_smoke: generated spectrum_position=%.4f PASS\n",
                    report["generated"]["spectrum_position"].get<double>());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "algebraic_fingerprint_smoke: %s\n", e.what());
        return 1;
    }
}
