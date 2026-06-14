// cyphalm_bench_native — char-LM BPC benchmark CLI for native CyphaLM tiers.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cypha_cell_hypothesis.hpp"
#include "cypha/cyphalm/cyphalm_alpha_spectrum.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/cyphalm_parallel.hpp"
#include "cypha/bench/bench_paths.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_from_model.hpp"

namespace {

struct Args {
    std::string mode = "hybrid";
    std::string profile = "d17";
    std::string cell_variant;
    int n_train = 40000;
    int n_eval = 2000;
    int threads = 0;
    bool analysis = false;
    int analysis_steps = 256;
    bool intelligence_profile = false;
    bool overnight = false;
    bool n_train_explicit = false;
    bool n_eval_explicit = false;
};

void usage() {
    std::cerr
        << "usage: cyphalm_bench_native --mode {char_lstm,ssm,hybrid,ssm_gria,context_bank,spectral,rpsm}\n"
        << "       --cell-variant {B0..H22}  (overrides --mode)\n"
        << "       --profile {d17,d21,d04} --n-train N --n-eval M --threads T\n"
        << "       --overnight  (D17/D21: full WikiText + 300k train budget; or CYPHA_BENCH_OVERNIGHT=1)\n"
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
        else if (k == "--cell-variant") a.cell_variant = need("--cell-variant");
        else if (k == "--profile") a.profile = need("--profile");
        else if (k == "--n-train") {
            a.n_train = std::stoi(need("--n-train"));
            a.n_train_explicit = true;
        } else if (k == "--n-eval") {
            a.n_eval = std::stoi(need("--n-eval"));
            a.n_eval_explicit = true;
        } else if (k == "--threads") a.threads = std::stoi(need("--threads"));
        else if (k == "--overnight") a.overnight = true;
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
        Args args = parse_args(argc, argv);
        const bool overnight = args.overnight || cypha::bench::bench_overnight_enabled();
        if (overnight) {
            if (!args.n_train_explicit) args.n_train = cypha::bench::bench_full_n_train();
            if (!args.n_eval_explicit) args.n_eval = 2000;
#if defined(_WIN32)
            _putenv_s("CYPHA_BENCH_OVERNIGHT", "1");
            _putenv_s("CYPHA_BENCH_FULL_CORPUS", "1");
#else
            setenv("CYPHA_BENCH_OVERNIGHT", "1", 1);
            setenv("CYPHA_BENCH_FULL_CORPUS", "1", 1);
#endif
        }
        cypha::cyphalm::set_thread_count(args.threads);

        cypha::cyphalm::CyphaLMConfig cfg;
        cypha::cyphalm::apply_bench_profile(args.profile, cfg);
        std::string mode_label = args.mode;
        if (!args.cell_variant.empty()) {
            cypha::cyphalm::apply_cell_variant(args.cell_variant, cfg);
            if (const auto* spec = cypha::cyphalm::find_cell_variant(args.cell_variant)) {
                mode_label = spec->bench_mode;
            }
        } else {
            const auto bench_mode = cypha::cyphalm::parse_bench_mode(args.mode);
            cypha::cyphalm::apply_bench_mode(bench_mode, cfg);
        }
        if (args.profile == "d17" && cfg.vocab_size < 256) cfg.vocab_size = 256;
        if (args.profile == "d21" && cfg.vocab_size < 256) cfg.vocab_size = 256;
        if (args.profile == "d04" && cfg.vocab_size < 128) cfg.vocab_size = 128;

        cypha::cyphalm::LMCorpus corpus;
        bool synthetic = false;
        const bool full_corpus = (args.profile == "d17" || args.profile == "d21") &&
                                 (cypha::cyphalm::bench_full_corpus_enabled() || overnight);
        try {
            const int max_chars = full_corpus ? 0 : 10'000'000;
            corpus = cypha::cyphalm::load_bench_corpus(args.profile, max_chars, cfg.vocab_size,
                                                       cfg.bpe_merges_path, cfg.bpe_vocab_path);
        } catch (const std::exception&) {
            if (!cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST")) {
                throw;
            }
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
            {"mode", mode_label},
            {"cell_variant", args.cell_variant.empty() ? nullptr : nlohmann::json(args.cell_variant)},
            {"profile", args.profile},
            {"context_mode", cypha::cyphalm::context_mode_string(cfg.context_mode)},
            {"n_train", args.n_train},
            {"n_eval", args.n_eval},
            {"train_epochs", cfg.train_epochs},
            {"threads", cypha::cyphalm::effective_thread_count()},
            {"corpus", corpus.source},
            {"synthetic", synthetic},
            {"full_corpus", full_corpus},
            {"overnight", overnight},
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
