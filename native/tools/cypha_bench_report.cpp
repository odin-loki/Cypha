// cypha_bench_report — native report generator mirroring bench/report/generate_report.py
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cypha/bench/bench_cross_domain.hpp"
#include "cypha/bench/bench_figures.hpp"
#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_report.hpp"

namespace {

void usage() {
    std::cerr << "usage: cypha_bench_report [--output PATH] [--skip-cross-domain]\n";
}

struct Args {
    std::string output;
    bool skip_cross_domain = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--output") a.output = need("--output");
        else if (k == "--skip-cross-domain") a.skip_cross_domain = true;
        else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (!args.skip_cross_domain) cypha::bench::run_all_cross_domain();

        const auto md_path = args.output.empty()
                                 ? cypha::bench::build_report()
                                 : cypha::bench::build_report(std::filesystem::path(args.output));
        const auto summary_path = cypha::bench::build_report_summary();
        const auto figure_paths = cypha::bench::generate_figure_data();

        std::cout << "Report written to " << md_path.string() << "\n";
        std::cout << "Summary JSON: " << summary_path.string() << "\n";
        std::cout << "Figures: " << cypha::bench::figures_dir().string() << " (" << figure_paths.size()
                  << " files, JSON+PNG)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cypha_bench_report: " << e.what() << "\n";
        usage();
        return 1;
    }
}
