// cyphalm_robustness_curve — D15C FGSM-proxy accuracy vs epsilon curve (MC3 Addendum 2).
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

#include "cypha/bench/bench_domains.hpp"
#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_report_json.hpp"
#include "cypha/bench/bench_tune.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

struct Args {
    fs::path out_path;
    std::vector<double> epsilons;
    int max_eval = -1;
    std::uint64_t bench_seed = 42;
    bool write_table = false;
};

void usage() {
    std::cerr
        << "usage: cyphalm_robustness_curve [--out PATH]\n"
        << "       [--epsilons E,E,...] [--max-eval N] [--bench-seed S] [--write-table]\n"
        << "Default epsilons: 0,0.05,0.1,0.2,0.5 (FAST: 0,0.05,0.1 via CYPHA_BENCH_FAST=1).\n";
}

std::vector<double> default_epsilons() {
    if (cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST")) {
        return {0.0, 0.05, 0.1};
    }
    return {0.0, 0.05, 0.1, 0.2, 0.5};
}

std::vector<double> parse_epsilons(const std::string& csv) {
    std::vector<double> out;
    std::stringstream ss(csv);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) {
            continue;
        }
        out.push_back(std::stod(part));
    }
    if (out.size() < 3) {
        throw std::runtime_error("--epsilons requires at least three comma-separated values");
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    a.out_path = cypha::bench::results_dir() / "robustness_curve.json";
    a.epsilons = default_epsilons();
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (k == "--out") {
            a.out_path = fs::absolute(need("--out"));
        } else if (k == "--epsilons") {
            a.epsilons = parse_epsilons(need("--epsilons"));
        } else if (k == "--max-eval") {
            a.max_eval = std::stoi(need("--max-eval"));
        } else if (k == "--bench-seed") {
            a.bench_seed = static_cast<std::uint64_t>(std::stoll(need("--bench-seed")));
        } else if (k == "--write-table") {
            a.write_table = true;
        } else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + k);
        }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const int max_eval =
            args.max_eval >= 0 ? args.max_eval : cypha::bench::bench_scale(500, 120);
        const auto t0 = std::chrono::steady_clock::now();

        Json curve =
            cypha::bench::run_d15_fgsm_robustness_curve(args.epsilons, max_eval, args.bench_seed);
        curve["runner"] = "cyphalm_robustness_curve";
        curve["profile"] = "d15";
        curve["fast"] = cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST");

        const auto t1 = std::chrono::steady_clock::now();
        curve["elapsed_s"] = std::chrono::duration<double>(t1 - t0).count();

        fs::create_directories(args.out_path.parent_path());
        {
            std::ofstream out(args.out_path);
            if (!out) {
                throw std::runtime_error("cannot write " + args.out_path.string());
            }
            out << curve.dump(2) << '\n';
        }

        if (args.write_table) {
            cypha::bench::save_domain_table("robustness_curve", curve);
        }

        const auto& points = curve.at("points");
        const double acc0 = points.front().at("accuracy").get<double>();
        const double acc_last = points.back().at("accuracy").get<double>();
        std::printf("robustness_curve: eps=%zu acc_nat=%.4f acc_max_eps=%.4f -> %s\n", points.size(),
                    acc0, acc_last, args.out_path.string().c_str());
        std::cout << curve.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cyphalm_robustness_curve: " << e.what() << '\n';
        usage();
        return 1;
    }
}
