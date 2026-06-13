// cypha_bench_run — native master runner mirroring bench/run_all.py.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cypha/bench/bench_cross_domain.hpp"
#include "cypha/bench/bench_domains.hpp"
#include "cypha/bench/bench_figures.hpp"
#include "cypha/bench/bench_paths.hpp"
#include "cypha/bench/bench_report.hpp"

namespace {

namespace fs = std::filesystem;
using cypha::bench::DomainSpec;

struct Args {
    int domain = 0;
    int from_domain = 1;
    std::string domain_tag;
    bool report_only = false;
    bool list_domains = false;
    bool ssm_diagnose = false;
};

void usage() {
    std::cerr << "usage: cypha_bench_run [--domain N] [--domain-tag TAG] [--from-domain N] "
                 "[--report-only] [--list-domains] [--ssm-diagnose]\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--domain") a.domain = std::stoi(need("--domain"));
        else if (k == "--domain-tag") a.domain_tag = need("--domain-tag");
        else if (k == "--from-domain") a.from_domain = std::stoi(need("--from-domain"));
        else if (k == "--report-only") a.report_only = true;
        else if (k == "--list-domains") a.list_domains = true;
        else if (k == "--ssm-diagnose") a.ssm_diagnose = true;
        else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

bool ssm_diagnose_enabled(const Args& args) {
    if (args.ssm_diagnose) {
        return true;
    }
    const char* v = std::getenv("CYPHA_SSM_DIAGNOSE");
    if (v == nullptr) {
        return false;
    }
    const std::string s(v);
    return s == "1" || s == "true" || s == "True" || s == "yes";
}

}  // namespace


int main(int argc, char** argv) {
    try {
        if (argc >= 1 && argv[0] != nullptr) {
            std::error_code ec;
            const fs::path self = fs::absolute(fs::path(argv[0]), ec);
            if (!ec) {
                cypha::bench::set_tool_dir(self.parent_path());
            }
        }
        const Args args = parse_args(argc, argv);
        cypha::bench::set_ssm_diagnose(ssm_diagnose_enabled(args));
        const auto domains = cypha::bench::all_domains();

        if (args.list_domains) {
            for (const auto& d : domains) {
                std::cout << d.tag << "\t" << d.module_path << "\n";
            }
            return 0;
        }

        if (args.report_only) {
            std::cout << "Running cross-domain analyses...\n";
            cypha::bench::run_all_cross_domain();
            const auto report_path = cypha::bench::build_report();
            const auto summary_path = cypha::bench::build_report_summary();
            const auto figure_paths = cypha::bench::generate_figure_data();
            std::cout << "Report: " << report_path.string() << "\n";
            std::cout << "Summary JSON: " << summary_path.string() << "\n";
            std::cout << "Figures: " << cypha::bench::figures_dir().string() << " (" << figure_paths.size()
                      << " files, JSON+PNG)\n";
            std::cout << "Tables: " << cypha::bench::tables_dir().string() << "\n";
            return 0;
        }

        std::vector<DomainSpec> selected = domains;
        if (!args.domain_tag.empty()) {
            selected.clear();
            for (const auto& d : domains) {
                if (d.tag == args.domain_tag) {
                    selected.push_back(d);
                }
            }
            if (selected.empty()) {
                throw std::runtime_error("unknown domain tag: " + args.domain_tag);
            }
        } else if (args.domain > 0) {
            selected.clear();
            char buf[16];
            std::snprintf(buf, sizeof(buf), "d%02d", args.domain);
            const std::string want = buf;
            for (const auto& d : domains) {
                if (d.tag == want) {
                    selected.push_back(d);
                }
            }
            if (selected.empty()) {
                throw std::runtime_error("unknown domain: " + std::to_string(args.domain));
            }
        } else if (args.from_domain > 1) {
            selected.clear();
            for (const auto& d : domains) {
                if (cypha::bench::domain_number(d.tag) >= args.from_domain) selected.push_back(d);
            }
        }

        std::vector<std::string> failures;
        for (const auto& d : selected) {
            std::cout << "\n============================================================\n";
            std::cout << "Running " << d.tag << ": " << d.module_path << "\n";
            std::cout << "============================================================\n";
            const auto t0 = std::chrono::steady_clock::now();
            try {
                (void)d.run();
                const double sec =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::cout << "  Completed in " << sec << "s\n";
            } catch (const std::exception& e) {
                failures.push_back(d.tag + ": " + e.what());
                std::cerr << "  FAILED: " << e.what() << "\n";
            }
        }

        std::cout << "\nRunning cross-domain analyses...\n";
        try {
            cypha::bench::run_all_cross_domain();
        } catch (const std::exception& e) {
            failures.push_back(std::string("cross_domain: ") + e.what());
            std::cerr << "  cross-domain FAILED: " << e.what() << "\n";
        }

        try {
            const auto report_path = cypha::bench::build_report();
            const auto summary_path = cypha::bench::build_report_summary();
            const auto figure_paths = cypha::bench::generate_figure_data();
            std::cout << "\nDone. Report: " << report_path.string() << "\n";
            std::cout << "Summary JSON: " << summary_path.string() << "\n";
            std::cout << "Figures: " << cypha::bench::figures_dir().string() << " (" << figure_paths.size()
                      << " files, JSON+PNG)\n";
        } catch (const std::exception& e) {
            failures.push_back(std::string("report: ") + e.what());
            std::cerr << "  report FAILED: " << e.what() << "\n";
        }

        std::cout << "Tables: " << cypha::bench::tables_dir().string() << "\n";
        if (!failures.empty()) {
            std::cout << failures.size() << " failure(s):\n";
            for (const auto& f : failures) std::cout << "  - " << f << "\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cypha_bench_run: " << e.what() << "\n";
        usage();
        return 1;
    }
}
