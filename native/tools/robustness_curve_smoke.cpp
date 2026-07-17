/// CTest smoke for MC3 FGSM robustness curve JSON (>=3 epsilon points, finite accuracy).
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

void validate_curve(const Json& curve) {
    if (!curve.contains("points") || !curve["points"].is_array()) {
        throw std::runtime_error("curve missing points array");
    }
    const auto& points = curve["points"];
    if (points.size() < 3) {
        throw std::runtime_error("curve needs >=3 epsilon points");
    }
    for (const auto& row : points) {
        if (!row.contains("epsilon") || !row["epsilon"].is_number()) {
            throw std::runtime_error("point missing epsilon");
        }
        if (!row.contains("accuracy") || row["accuracy"].is_null()) {
            throw std::runtime_error("point missing accuracy");
        }
        const double acc = row["accuracy"].get<double>();
        if (!std::isfinite(acc) || acc < 0.0 || acc > 1.0) {
            throw std::runtime_error("accuracy out of range");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const fs::path exe_dir = exe_dir_from_argv(argc, argv);
        const fs::path curve_exe =
            cypha::bench::resolve_runner_exe("cyphalm_robustness_curve", exe_dir);
        if (!fs::is_regular_file(curve_exe)) {
            std::fprintf(stderr, "missing cyphalm_robustness_curve: %s\n", curve_exe.string().c_str());
            return 1;
        }

        const fs::path out = fs::temp_directory_path() / "cypha_robustness_curve_smoke.json";
        const std::vector<std::string> cli = {
            "--out", out.string(),
            "--epsilons", "0,0.05,0.1",
            "--max-eval", "40",
            "--bench-seed", "42",
        };
        const cypha::bench::RunProcessResult proc =
            cypha::bench::run_executable_capture(curve_exe, cli);
        if (proc.exit_code != 0) {
            std::fprintf(stderr, "curve runner exit=%d stderr=%s\n", proc.exit_code,
                         proc.stderr_text.c_str());
            return 1;
        }

        const Json curve = load_json(out);
        validate_curve(curve);

        std::printf("robustness_curve_smoke: %zu eps points PASS\n", curve["points"].size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "robustness_curve_smoke: %s\n", e.what());
        return 1;
    }
}
