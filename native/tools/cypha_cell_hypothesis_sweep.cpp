// cypha_cell_hypothesis_sweep — smoke runner for cell hypothesis variants (Tier 1 subset + scaffold).
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

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
    bool list_variants = false;
};

struct CellVariantSpec {
    const char* id;
    const char* name;
    int tier;
    bool runnable;
    const char* bench_mode;
    const char* notes;
};

const std::vector<CellVariantSpec>& all_variants() {
    static const std::vector<CellVariantSpec> kVariants = {
        {"B0", "4-gram", 0, false, "", "scaffold — not a CyphaLM cell"},
        {"B1", "char-LSTM", 0, true, "char_lstm", "locked baseline"},
        {"B2", "hybrid_gria_lstm", 0, true, "hybrid", "locked baseline"},
        {"H01", "alpha-gate cell", 1, true, "hybrid", "GRIA alpha as forget gate proxy"},
        {"H02", "EML activation cell", 1, false, "", "scaffold — Sheffer eml() cell TBD"},
        {"H03", "CausalField cell", 1, true, "ssm", "SSM/SGEMV recurrence primitive"},
        {"H04", "Pure CyphaDIF LM", 1, true, "ssm_gria", "DIF + GRIA without LSTM"},
        {"H05", "alpha-fitness aux loss", 1, true, "hybrid", "hybrid + profile-guided loss tag"},
        {"H06", "NIG-state cell", 2, false, "", "scaffold"},
        {"H07", "Differential gate", 2, false, "", "scaffold"},
        {"H08", "TieredContext cell", 2, false, "", "scaffold"},
        {"H09", "GRIA-gated mixture", 2, false, "", "scaffold"},
        {"H10", "NMP regularised", 2, false, "", "scaffold"},
        {"H11", "Reversible cell", 2, false, "", "scaffold"},
        {"H12", "MDL forget", 2, false, "", "scaffold"},
        {"H13", "Priority replay recurrence", 2, false, "", "scaffold"},
        {"H14", "OOD-branching cell", 2, false, "", "scaffold"},
        {"H15", "AXIOM-evolved cell", 3, false, "", "scaffold"},
        {"H16", "SR on trained LSTM gates", 3, false, "", "scaffold"},
        {"H17", "Sheffer-only cell", 3, false, "", "scaffold"},
        {"H18", "CA state cell", 3, false, "", "scaffold"},
        {"H19", "Izaac-seeded init", 3, false, "", "scaffold"},
        {"H20", "Spectral state cell", 3, false, "spectral", "spectral scaffold"},
        {"H21", "Free Energy cell", 3, false, "", "scaffold"},
        {"H22", "Algebraic fingerprint cell", 3, false, "", "scaffold"},
    };
    return kVariants;
}

void usage() {
    std::cerr << "usage: cypha_cell_hypothesis_sweep [--smoke] [--tier1-only] [--list-variants]\n"
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
        else if (k == "--list-variants") a.list_variants = true;
        else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    if (a.smoke) {
        a.n_train = 400;
        a.n_eval = 80;
        a.tier1_only = true;
    }
    return a;
}

bool should_run(const CellVariantSpec& v, bool tier1_only) {
    if (!v.runnable) {
        return false;
    }
    if (tier1_only) {
        return v.tier == 1;
    }
    return v.tier <= 2 || v.id[0] == 'B';
}

nlohmann::json run_variant(const CellVariantSpec& spec, const Args& args) {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_bench_profile(args.profile, cfg);
    const auto bench_mode = cypha::cyphalm::parse_bench_mode(spec.bench_mode);
    cypha::cyphalm::apply_bench_mode(bench_mode, cfg);
    if (args.profile == "d17" && cfg.vocab_size < 256) cfg.vocab_size = 256;
    if (spec.id == std::string("H01")) {
        cfg.alpha_init = 0.5;
        cfg.alpha_learnable = true;
    }
    if (spec.id == std::string("H05")) {
        cfg.alpha_learnable = true;
    }

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
    if (spec.id == std::string("B2") && !std::isnan(bpc)) {
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
            for (const auto& v : all_variants()) {
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
        nlohmann::json scaffold = nlohmann::json::array();
        double b2_bpc = std::numeric_limits<double>::quiet_NaN();

        for (const auto& v : all_variants()) {
            if (!should_run(v, args.tier1_only)) {
                scaffold.push_back(
                    {{"id", v.id}, {"name", v.name}, {"tier", v.tier}, {"status", "scaffold"}});
                continue;
            }
            auto row = run_variant(v, args);
            if (std::string(v.id) == "B2") {
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
            {"smoke", args.smoke || args.tier1_only},
            {"results", results},
            {"scaffold", scaffold},
            {"variant_count", all_variants().size()},
        };
        std::cout << out.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cypha_cell_hypothesis_sweep: " << e.what() << "\n";
        usage();
        return 1;
    }
}
