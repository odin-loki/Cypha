// cypha_cell_hypothesis_sweep — smoke runner for cell hypothesis variants (Tier 1+2).
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/cyphalm_parallel.hpp"

namespace {

struct Args {
    std::string profile = "d17";
    int n_train = 4000;
    int n_eval = 500;
    int threads = 0;
    bool smoke = false;
    bool tier1_only = false;
    bool tier2_only = false;
    bool tier2_smoke = false;
    bool list_variants = false;
    std::string cell_variant;
};

void usage() {
    std::cerr << "usage: cypha_cell_hypothesis_sweep [--smoke] [--tier1-only] [--tier2-only]\n"
              << "       [--tier2-smoke] [--list-variants] [--cell-variant H06]\n"
              << "       [--profile d17] [--n-train N] [--n-eval M] [--threads T]\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--profile") a.profile = need("--profile");
        else if (k == "--n-train") a.n_train = std::stoi(need("--n-train"));
        else if (k == "--n-eval") a.n_eval = std::stoi(need("--n-eval"));
        else if (k == "--threads") a.threads = std::stoi(need("--threads"));
        else if (k == "--smoke") a.smoke = true;
        else if (k == "--tier1-only") a.tier1_only = true;
        else if (k == "--tier2-only") a.tier2_only = true;
        else if (k == "--tier2-smoke") a.tier2_smoke = true;
        else if (k == "--list-variants") a.list_variants = true;
        else if (k == "--cell-variant") a.cell_variant = need("--cell-variant");
        else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    if (a.tier2_smoke) {
        a.n_train = 200;
        a.n_eval = 40;
    }
    if (a.smoke) {
        a.n_train = 400;
        a.n_eval = 80;
        a.tier1_only = true;
    }
    return a;
}

bool should_run(const cypha::cyphalm::CellVariantSpec& v, const Args& args) {
    if (!v.runnable) {
        return false;
    }
    if (!args.cell_variant.empty()) {
        return v.id == args.cell_variant;
    }
    if (args.tier2_smoke) {
        return v.id == "H06" || v.id == "H08" || v.id == "H14";
    }
    if (args.tier2_only) {
        return v.tier == 2;
    }
    if (args.tier1_only) {
        return v.tier == 1;
    }
    return v.tier <= 2 || v.id[0] == 'B';
}

nlohmann::json run_variant(const cypha::cyphalm::CellVariantSpec& spec, const Args& args) {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_bench_profile(args.profile, cfg);
    cypha::cyphalm::apply_cell_variant(spec.id, cfg);
    if (args.profile == "d17" && cfg.vocab_size < 256) cfg.vocab_size = 256;

    cypha::cyphalm::LMCorpus corpus;
    bool synthetic = false;
    try {
        corpus = cypha::cyphalm::load_bench_corpus(args.profile, 1'000'000, cfg.vocab_size,
                                                   cfg.bpe_merges_path, cfg.bpe_vocab_path);
    } catch (const std::exception&) {
        synthetic = true;
        corpus.profile = args.profile;
        corpus.source = "synthetic";
        corpus.vocab_size = cfg.vocab_size;
        corpus.train_ids = cypha::cyphalm::synthetic_corpus(args.n_train + args.n_eval + 64,
                                                            cfg.vocab_size, cfg.seed);
        const std::size_t split = static_cast<std::size_t>(args.n_train);
        corpus.eval_ids.assign(corpus.train_ids.begin() + static_cast<std::ptrdiff_t>(split),
                               corpus.train_ids.end());
        corpus.train_ids.resize(split);
    }
    cfg.vocab_size = corpus.vocab_size;

    cypha::cyphalm::CyphaLMModel model(cfg);
    model.train_sequence(corpus.train_ids, args.n_train, cfg.train_epochs);
    const double bpc = model.eval_bpc(corpus.eval_ids, args.n_eval);
    const auto alpha_profile = model.compression_profile();

    nlohmann::json row = {
        {"id", spec.id},
        {"name", spec.name},
        {"tier", spec.tier},
        {"bench_mode", spec.bench_mode},
        {"notes", spec.notes},
        {"bpc", std::isnan(bpc) ? nullptr : nlohmann::json(bpc)},
        {"corpus", corpus.source},
        {"synthetic", synthetic},
        {"mean_alpha", alpha_profile.value("mean_alpha", 0.0)},
    };
    if (spec.id == "B2" && !std::isnan(bpc)) {
        row["delta_vs_b2"] = 0.0;
    }
    return row;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        cypha::cyphalm::set_thread_count(args.threads);

        if (args.list_variants) {
            nlohmann::json listed = nlohmann::json::array();
            for (const auto& v : cypha::cyphalm::all_cell_variants()) {
                listed.push_back({{"id", v.id},
                                  {"name", v.name},
                                  {"tier", v.tier},
                                  {"runnable", v.runnable},
                                  {"bench_mode", v.bench_mode},
                                  {"notes", v.notes}});
            }
            std::cout << listed.dump(2) << std::endl;
            return 0;
        }

        nlohmann::json results = nlohmann::json::array();
        nlohmann::json skipped = nlohmann::json::array();
        double b2_bpc = std::numeric_limits<double>::quiet_NaN();

        for (const auto& v : cypha::cyphalm::all_cell_variants()) {
            if (!should_run(v, args)) {
                skipped.push_back({{"id", v.id}, {"name", v.name}, {"tier", v.tier}});
                continue;
            }
            auto row = run_variant(v, args);
            if (v.id == "B2") {
                if (!row["bpc"].is_null()) {
                    b2_bpc = row["bpc"].get<double>();
                }
            }
            results.push_back(std::move(row));
        }

        for (auto& row : results) {
            if (!row["bpc"].is_null() && !std::isnan(b2_bpc) && row.value("id", "") != "B2") {
                row["delta_vs_b2"] = row["bpc"].get<double>() - b2_bpc;
            }
        }

        nlohmann::json out = {
            {"profile", args.profile},
            {"n_train", args.n_train},
            {"n_eval", args.n_eval},
            {"smoke", args.smoke || args.tier1_only || args.tier2_only || args.tier2_smoke},
            {"cell_variant", args.cell_variant.empty() ? nullptr : nlohmann::json(args.cell_variant)},
            {"results", results},
            {"skipped", skipped},
            {"variant_count", cypha::cyphalm::all_cell_variants().size()},
        };
        std::cout << out.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cypha_cell_hypothesis_sweep: " << e.what() << "\n";
        usage();
        return 1;
    }
}
