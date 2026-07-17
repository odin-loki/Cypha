// cyphalm_sample_efficiency_curve — BPC sample-efficiency curve at n_train tiers (MC5/MG5).
// Invokes cyphalm_bench_native once per tier; aggregates JSON curve table.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_report_json.hpp"
#include "cypha/bench/bench_tune.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

struct Args {
    fs::path exe_dir;
    fs::path out_path;
    std::string profile = "d17";
    std::string mode = "hybrid";
    std::vector<int> tiers;
    int n_eval = -1;
    std::int64_t bench_seed = 42;
    bool write_table = false;
};

void usage() {
    std::cerr
        << "usage: cyphalm_sample_efficiency_curve [--exe-dir DIR] [--out PATH]\n"
        << "       [--profile d17|d04] [--mode hybrid|char_lstm|...]\n"
        << "       [--tiers N,N,...] [--n-eval M] [--bench-seed S] [--write-table]\n"
        << "Default tiers: 500,2000,5000 (FAST: 80,160,240 via CYPHA_BENCH_FAST=1).\n";
}

std::vector<int> default_tiers() {
    if (cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST")) {
        return {80, 160, 240};
    }
    return {500, 2000, 5000};
}

std::vector<int> parse_tiers(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        out.push_back(std::stoi(part));
    }
    if (out.size() < 2) {
        throw std::runtime_error("--tiers requires at least two comma-separated values");
    }
    return out;
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

Json parse_bench_stdout(const std::string& text) {
    const std::string blob = extract_json_blob(text);
    try {
        return Json::parse(blob);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("failed to parse bench JSON: ") + e.what());
    }
}

Args parse_args(int argc, char** argv) {
    Args a;
    a.exe_dir = fs::current_path();
    a.out_path = cypha::bench::results_dir() / "sample_efficiency_curve.json";
    a.tiers = default_tiers();
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--exe-dir") a.exe_dir = fs::absolute(need("--exe-dir"));
        else if (k == "--out") a.out_path = fs::absolute(need("--out"));
        else if (k == "--profile") a.profile = need("--profile");
        else if (k == "--mode") a.mode = need("--mode");
        else if (k == "--tiers") a.tiers = parse_tiers(need("--tiers"));
        else if (k == "--n-eval") a.n_eval = std::stoi(need("--n-eval"));
        else if (k == "--bench-seed") a.bench_seed = std::stoll(need("--bench-seed"));
        else if (k == "--write-table") a.write_table = true;
        else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + k);
        }
    }
    return a;
}

double require_finite_bpc(const Json& bench_out, int n_train) {
    if (!bench_out.contains("bpc") || bench_out["bpc"].is_null()) {
        throw std::runtime_error("tier n_train=" + std::to_string(n_train) + " missing bpc");
    }
    const double bpc = bench_out["bpc"].get<double>();
    if (!std::isfinite(bpc)) {
        throw std::runtime_error("tier n_train=" + std::to_string(n_train) + " non-finite bpc");
    }
    return bpc;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const fs::path bench_exe =
            cypha::bench::resolve_runner_exe("cyphalm_bench_native", args.exe_dir);
        if (!fs::is_regular_file(bench_exe)) {
            throw std::runtime_error("missing cyphalm_bench_native: " + bench_exe.string());
        }

        const int n_eval =
            args.n_eval >= 0 ? args.n_eval : cypha::bench::bench_scale(256, 64);
        const auto t0 = std::chrono::steady_clock::now();

        Json points = Json::array();
        std::string corpus;
        bool synthetic = false;
        for (const int n_train : args.tiers) {
            const std::vector<std::string> cli = {
                "--profile", args.profile,
                "--mode", args.mode,
                "--n-train", std::to_string(n_train),
                "--n-eval", std::to_string(n_eval),
                "--threads", "1",
                "--bench-seed", std::to_string(args.bench_seed),
            };
            const cypha::bench::RunProcessResult proc =
                cypha::bench::run_executable_capture(bench_exe, cli);
            if (proc.exit_code != 0) {
                throw std::runtime_error("cyphalm_bench_native n_train=" + std::to_string(n_train) +
                                         " exit=" + std::to_string(proc.exit_code) + " stderr=" +
                                         proc.stderr_text);
            }
            const Json bench_out = parse_bench_stdout(proc.stdout_text);
            const double bpc = require_finite_bpc(bench_out, n_train);
            if (corpus.empty() && bench_out.contains("corpus")) {
                corpus = bench_out.value("corpus", "");
                synthetic = bench_out.value("synthetic", false);
            }
            Json row = {
                {"n_train", n_train},
                {"n_eval", n_eval},
                {"bpc", bpc},
                {"accuracy", nullptr},
            };
            if (bench_out.contains("hybrid_gria_weight") && !bench_out["hybrid_gria_weight"].is_null()) {
                row["hybrid_gria_weight"] = bench_out["hybrid_gria_weight"];
            }
            points.push_back(std::move(row));
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_s =
            std::chrono::duration<double>(t1 - t0).count();

        Json curve = {
            {"curve_id", "sample_efficiency"},
            {"metric", "bpc"},
            {"runner", "cyphalm_bench_native"},
            {"profile", args.profile},
            {"mode", args.mode},
            {"bench_seed", args.bench_seed},
            {"fast", cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST")},
            {"corpus", corpus.empty() ? Json(nullptr) : Json(corpus)},
            {"synthetic", synthetic},
            {"n_eval", n_eval},
            {"tiers_requested", args.tiers},
            {"elapsed_s", elapsed_s},
            {"points", points},
        };

        fs::create_directories(args.out_path.parent_path());
        {
            std::ofstream out(args.out_path);
            if (!out) {
                throw std::runtime_error("cannot write " + args.out_path.string());
            }
            out << curve.dump(2) << '\n';
        }

        if (args.write_table) {
            cypha::bench::save_domain_table("sample_efficiency_curve", curve);
        }

        std::cout << curve.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cyphalm_sample_efficiency_curve: " << e.what() << '\n';
        usage();
        return 1;
    }
}
