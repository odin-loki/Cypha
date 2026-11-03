// cypha_tune_run — native tuning/sweep orchestrator mirroring bench/tuning/*.py.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_tune.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

struct Args {
    fs::path config;
    fs::path out;
    fs::path exe_dir;
    int max_cells = -1;
    bool dry_run = false;
    bool write = false;
};

void usage() {
    std::cerr
        << "usage: cypha_tune_run --config PATH [--out PATH] [--exe-dir DIR]\n"
        << "       [--max-cells N] [--write] [--dry-run]\n"
        << "\n"
        << "Loads a sweep JSON (runner + defaults + grid/cells), invokes\n"
        << "cyphalm_bench_native or cypha_bench_run per cell, emits results JSON.\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--config") {
            a.config = need("--config");
        } else if (k == "--out") {
            a.out = need("--out");
        } else if (k == "--exe-dir") {
            a.exe_dir = need("--exe-dir");
        } else if (k == "--max-cells") {
            a.max_cells = std::stoi(need("--max-cells"));
        } else if (k == "--write") {
            a.write = true;
        } else if (k == "--dry-run") {
            a.dry_run = true;
        } else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    if (a.config.empty()) {
        throw std::runtime_error("--config is required");
    }
    return a;
}

fs::path resolve_exe_dir(const Args& a, char** argv) {
    if (!a.exe_dir.empty()) return fs::absolute(a.exe_dir);
    if (argv && argv[0]) {
        const fs::path self = fs::absolute(argv[0]);
        if (self.has_parent_path()) return self.parent_path();
    }
    return fs::current_path();
}

fs::path default_out_path(const cypha::bench::TuneSweepSpec& spec) {
    return cypha::bench::config_dir() / (spec.sweep_id + "_results.json");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const fs::path config_path = fs::absolute(args.config);
        const cypha::bench::TuneSweepSpec spec = cypha::bench::load_tune_sweep(config_path);
        const fs::path exe_dir = resolve_exe_dir(args, argv);

        if (args.dry_run) {
            const auto cells = cypha::bench::expand_tune_cells(spec);
            Json preview = {
                {"sweep_id", spec.sweep_id},
                {"runner", spec.runner},
                {"exe_dir", exe_dir.string()},
                {"cell_count", cells.size()},
                {"cells", Json::array()},
            };
            int n = 0;
            for (const auto& cell : cells) {
                if (args.max_cells >= 0 && n >= args.max_cells) break;
                ++n;
                preview["cells"].push_back({
                    {"id", cell.id},
                    {"cli", cypha::bench::cell_to_cli_args(spec.defaults, cell.args)},
                });
            }
            std::cout << preview.dump(2) << std::endl;
            return 0;
        }

        const auto result = cypha::bench::run_tune_sweep(spec, exe_dir, args.max_cells);
        const Json out = cypha::bench::tune_sweep_result_to_json(result);
        std::cout << out.dump(2) << std::endl;

        if (args.write) {
            const fs::path out_path = args.out.empty() ? default_out_path(spec) : fs::absolute(args.out);
            fs::create_directories(out_path.parent_path());
            std::ofstream file(out_path);
            file << out.dump(2) << '\n';
            std::cerr << "Wrote " << out_path.string() << '\n';
        }
        const bool all_ok = std::all_of(result.runs.begin(), result.runs.end(),
                                        [](const cypha::bench::TuneRunResult& r) { return r.ok; });
        return all_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "cypha_tune_run: " << e.what() << '\n';
        usage();
        return 1;
    }
}
