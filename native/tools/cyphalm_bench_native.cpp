// cyphalm_bench_native — char-LM BPC benchmark CLI for native CyphaLM tiers.
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_alpha_spectrum.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/cyphalm_parallel.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_from_model.hpp"

namespace {

struct Args {
    std::string mode = "hybrid";
    std::string profile = "d17";
    int n_train = 40000;
    int n_eval = 2000;
    int threads = 0;
    bool analysis = false;
    int analysis_steps = 256;
    bool intelligence_profile = false;
};

void usage() {
    std::cerr
        << "usage: cyphalm_bench_native --mode {char_lstm,ssm,hybrid,ssm_gria,context_bank,spectral}\n"
        << "       --profile {d17,d04} --n-train N --n-eval M --threads T\n"
        << "       --analysis [--analysis-steps N]\n"
        << "       --intelligence-profile\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--mode") a.mode = need("--mode");
        else if (k == "--profile") a.profile = need("--profile");
        else if (k == "--n-train") a.n_train = std::stoi(need("--n-train"));
        else if (k == "--n-eval") a.n_eval = std::stoi(need("--n-eval"));
        else if (k == "--threads") a.threads = std::stoi(need("--threads"));
        else if (k == "--analysis") a.analysis = true;
        else if (k == "--analysis-steps") a.analysis_steps = std::stoi(need("--analysis-steps"));
        else if (k == "--intelligence-profile") a.intelligence_profile = true;
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
        cypha::cyphalm::set_thread_count(args.threads);

        cypha::cyphalm::CyphaLMConfig cfg;
        cypha::cyphalm::apply_bench_profile(args.profile, cfg);
        const auto bench_mode = cypha::cyphalm::parse_bench_mode(args.mode);
        cypha::cyphalm::apply_bench_mode(bench_mode, cfg);
        if (args.profile == "d17" && cfg.vocab_size < 256) cfg.vocab_size = 256;
        if (args.profile == "d04" && cfg.vocab_size < 128) cfg.vocab_size = 128;

        cypha::cyphalm::LMCorpus corpus;
        bool synthetic = false;
        try {
            corpus = cypha::cyphalm::load_bench_corpus(args.profile, 10'000'000, cfg.vocab_size,
                                                       cfg.bpe_merges_path, cfg.bpe_vocab_path);
        } catch (const std::exception&) {
            synthetic = true;
            corpus.profile = args.profile;
            corpus.source = "synthetic";
            corpus.vocab_size = cfg.vocab_size;
            corpus.train_ids =
                cypha::cyphalm::synthetic_corpus(args.n_train + args.n_eval + 64, cfg.vocab_size,
                                                 cfg.seed);
            const std::size_t split = static_cast<std::size_t>(args.n_train);
            corpus.eval_ids.assign(corpus.train_ids.begin() + static_cast<std::ptrdiff_t>(split),
                                   corpus.train_ids.end());
            corpus.train_ids.resize(split);
        }

        cfg.vocab_size = corpus.vocab_size;
        if (!cfg.bpe_merges_path.empty() && !cfg.bpe_vocab_path.empty()) {
            corpus.source += "+bpe";
        }

        cypha::cyphalm::CyphaLMModel model(cfg);
        model.train_sequence(corpus.train_ids, args.n_train, cfg.train_epochs);
        cypha::intelligence::IntelligenceProfiler profiler;
        const double bpc = model.eval_bpc(corpus.eval_ids, args.n_eval,
                                          args.intelligence_profile ? &profiler : nullptr);

        nlohmann::json out = {
            {"mode", args.mode},
            {"profile", args.profile},
            {"context_mode", cypha::cyphalm::context_mode_string(cfg.context_mode)},
            {"n_train", args.n_train},
            {"n_eval", args.n_eval},
            {"train_epochs", cfg.train_epochs},
            {"threads", cypha::cyphalm::effective_thread_count()},
            {"corpus", corpus.source},
            {"synthetic", synthetic},
            {"bpc", bpc},
            {"vocab_size", cfg.vocab_size},
        };
        if (std::isnan(bpc)) out["bpc"] = nullptr;
        if (args.intelligence_profile) {
            out["intelligence_profile"] = cypha::intelligence::intelligence_profile_report_json(profiler);
        }
        if (args.analysis) {
            const auto profile = model.compression_profile();
            out["alpha_spectrum"] = {
                {"mean_alpha", profile.value("mean_alpha", 0.0)},
                {"fraction_edge_of_chaos", profile.value("fraction_near_edge_of_chaos", 0.0)},
                {"n_experts", profile.value("n_experts", 0)},
            };
            const auto track =
                cypha::cyphalm::alpha_spectrum_track(model, args.analysis_steps, corpus.train_ids);
            out["alpha_track_steps"] = track.size();
            if (!track.empty()) out["alpha_track_last"] = track.back();
        }
        if (cfg.context_mode == cypha::cyphalm::ContextMode::Hybrid) {
            out["hybrid_blend_logit"] = model.hybrid_blend_logit();
            out["hybrid_gria_weight"] = model.hybrid_gria_weight();
            const double saved_blend = model.hybrid_blend_logit();
            model.set_hybrid_blend_logit(-40.0);
            out["bpc_lstm_only"] = model.eval_bpc(corpus.eval_ids, args.n_eval);
            model.set_hybrid_blend_logit(saved_blend);
        }
        std::cout << out.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cyphalm_bench_native: " << e.what() << "\n";
        usage();
        return 1;
    }
}
