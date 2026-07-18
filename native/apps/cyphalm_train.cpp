// cyphalm_train — train CyphaLM from corpus text and save a Python-compatible checkpoint.
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_checkpoint.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_intelligence_hook.hpp"
#include "cypha/cyphalm/cyphalm_math_integration.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/cyphalm_parallel.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_completeness.hpp"
#include "cypha/intelligence/profile_guided_loss.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string profile = "d17";
    std::string corpus;
    int synthetic_tokens = 0;
    int epochs = 1;
    std::string out_dir;
    int max_chars = 0;
    int max_train_steps = 0;
    int threads = 0;
    bool profile_guided_loss = false;
    bool intelligence_monitor = false;
    bool math_integration = false;
    int bptt_lstm = -1;
    std::string lstm_optim;
    double grad_clip = -1.0;
    std::string lstm_init;
};

void usage() {
    std::cerr
        << "usage: cyphalm_train --profile {d17,d04} --epochs N --out checkpoint_dir/\n"
        << "       (--corpus bench/data/... | --synthetic-tokens N)\n"
        << "       [--max-chars M] [--max-train-steps S] [--threads T]\n"
        << "       [--profile-guided-loss] [--intelligence-monitor] [--math-integration]\n"
        << "       [--bptt-lstm N] [--optim sgd|adam] [--grad-clip C] [--lstm-init default|classic]\n";
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
        else if (k == "--corpus") a.corpus = need("--corpus");
        else if (k == "--synthetic-tokens") a.synthetic_tokens = std::stoi(need("--synthetic-tokens"));
        else if (k == "--epochs") a.epochs = std::stoi(need("--epochs"));
        else if (k == "--out") a.out_dir = need("--out");
        else if (k == "--max-chars") a.max_chars = std::stoi(need("--max-chars"));
        else if (k == "--max-train-steps") a.max_train_steps = std::stoi(need("--max-train-steps"));
        else if (k == "--threads") a.threads = std::stoi(need("--threads"));
        else if (k == "--profile-guided-loss") a.profile_guided_loss = true;
        else if (k == "--intelligence-monitor") a.intelligence_monitor = true;
        else if (k == "--math-integration") a.math_integration = true;
        else if (k == "--bptt-lstm") a.bptt_lstm = std::stoi(need("--bptt-lstm"));
        else if (k == "--optim") a.lstm_optim = need("--optim");
        else if (k == "--grad-clip") a.grad_clip = std::stod(need("--grad-clip"));
        else if (k == "--lstm-init") a.lstm_init = need("--lstm-init");
        else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    if (a.profile != "d17" && a.profile != "d04") {
        throw std::runtime_error("--profile must be d17 or d04");
    }
    if (a.out_dir.empty()) throw std::runtime_error("missing --out checkpoint_dir/");
    if (a.corpus.empty() && a.synthetic_tokens <= 0) {
        throw std::runtime_error("provide --corpus or --synthetic-tokens");
    }
    if (!a.corpus.empty() && a.synthetic_tokens > 0) {
        throw std::runtime_error("use either --corpus or --synthetic-tokens, not both");
    }
    if (a.epochs < 1) throw std::runtime_error("--epochs must be >= 1");
    return a;
}

fs::path checkpoint_base_path(const std::string& out) {
    fs::path p(out);
    if (p.extension() == ".json") {
        return p.replace_extension();
    }
    return p / "checkpoint";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        cypha::cyphalm::set_thread_count(args.threads);

        cypha::cyphalm::CyphaLMConfig cfg;
        cypha::cyphalm::apply_bench_profile(args.profile, cfg);
        bool profile_guided_loss = args.profile_guided_loss;
        bool intelligence_monitor = args.intelligence_monitor;
        if (args.math_integration) {
            cypha::cyphalm::apply_math_integration_preset(cfg);
            profile_guided_loss = true;
            intelligence_monitor = true;
        }
        if (args.profile == "d17" && cfg.vocab_size < 256) cfg.vocab_size = 256;
        if (args.profile == "d04" && cfg.vocab_size < 128) cfg.vocab_size = 128;
        cfg.train_epochs = args.epochs;
        if (args.bptt_lstm > 0) cfg.lstm_bptt_steps = args.bptt_lstm;
        if (!args.lstm_optim.empty()) cfg.lstm_optim = args.lstm_optim;
        if (args.grad_clip >= 0.0) cfg.lstm_grad_clip = args.grad_clip;
        if (!args.lstm_init.empty()) cfg.lstm_init = args.lstm_init;

        cypha::cyphalm::LMCorpus corpus;
        bool synthetic = false;
        if (args.synthetic_tokens > 0) {
            synthetic = true;
            corpus.profile = args.profile;
            corpus.source = "synthetic";
            corpus.vocab_size = cfg.vocab_size;
            const int n_tokens = std::max(64, args.synthetic_tokens);
            corpus.train_ids =
                cypha::cyphalm::synthetic_corpus(n_tokens + 64, cfg.vocab_size, cfg.seed);
            const std::size_t split =
                static_cast<std::size_t>(std::min(n_tokens, static_cast<int>(corpus.train_ids.size() * 8 / 10)));
            corpus.eval_ids.assign(corpus.train_ids.begin() + static_cast<std::ptrdiff_t>(split),
                                   corpus.train_ids.end());
            corpus.train_ids.resize(split);
        } else {
            corpus = cypha::cyphalm::load_corpus_file(args.corpus, args.profile, args.max_chars,
                                                      cfg.vocab_size, cfg.bpe_merges_path,
                                                      cfg.bpe_vocab_path);
        }

        cfg.vocab_size = corpus.vocab_size;
        const int n_train = static_cast<int>(corpus.train_ids.size()) - 1;
        if (n_train < 1) throw std::runtime_error("corpus too short for training");
        const int train_steps =
            args.max_train_steps > 0 ? std::min(args.max_train_steps, n_train) : n_train;

        cypha::cyphalm::CyphaLMModel model(cfg);
        cypha::intelligence::IntelligenceProfiler train_profiler;
        cypha::intelligence::IntelligenceProfiler* train_profiler_ptr =
            (intelligence_monitor || profile_guided_loss) ? &train_profiler : nullptr;
        model.train_sequence(corpus.train_ids, train_steps, args.epochs, train_profiler_ptr);
        cypha::intelligence::IntelligenceProfiler eval_profiler;
        cypha::intelligence::IntelligenceProfiler* eval_profiler_ptr =
            (profile_guided_loss || intelligence_monitor) ? &eval_profiler : train_profiler_ptr;
        const double bpc = model.eval_bpc(corpus.eval_ids, static_cast<int>(corpus.eval_ids.size()) - 1,
                                          eval_profiler_ptr);

        const fs::path out_dir =
            (fs::path(args.out_dir).extension() == ".json") ? fs::path(args.out_dir).parent_path()
                                                            : fs::path(args.out_dir);
        if (!out_dir.empty()) {
            fs::create_directories(out_dir);
        }
        const fs::path ckpt_base = checkpoint_base_path(args.out_dir);
        cypha::cyphalm::save_cyphalm_model(model, ckpt_base.string());

        nlohmann::json out = {
            {"profile", args.profile},
            {"context_mode", cypha::cyphalm::context_mode_string(cfg.context_mode)},
            {"epochs", args.epochs},
            {"train_steps", train_steps},
            {"train_tokens", static_cast<int>(corpus.train_ids.size())},
            {"eval_tokens", static_cast<int>(corpus.eval_ids.size())},
            {"corpus", corpus.source},
            {"synthetic", synthetic},
            {"bpc", bpc},
            {"vocab_size", cfg.vocab_size},
            {"checkpoint_json", (ckpt_base.string() + ".json")},
            {"checkpoint_npz", (ckpt_base.string() + ".npz")},
            {"threads", cypha::cyphalm::effective_thread_count()},
        };
        if (std::isnan(bpc)) out["bpc"] = nullptr;
        if (profile_guided_loss) {
            const auto pg_loss =
                cypha::intelligence::compute_profile_guided_loss_from_profiler(eval_profiler);
            out["profile_guided_loss"] = {
                {"r_eu_penalty", pg_loss.r_eu_penalty},
                {"tau_penalty", pg_loss.tau_penalty},
                {"total", pg_loss.total},
                {"integration", "post_train_eval_bpc_profiler"},
            };
        }
        if (intelligence_monitor) {
            const auto completeness =
                cypha::intelligence::validate_profile_completeness(train_profiler);
            out["intelligence_monitor"] =
                cypha::cyphalm::export_intelligence_monitor_report(train_profiler);
            out["profile_completeness"] = cypha::intelligence::profile_completeness_to_json(completeness);
            std::cerr << "intelligence monitor completeness: "
                      << (completeness.all_complete ? "complete" : "incomplete");
            if (!completeness.missing_stats.empty()) {
                std::cerr << " (missing:";
                for (const auto& name : completeness.missing_stats) {
                    std::cerr << ' ' << name;
                }
                std::cerr << ')';
            }
            std::cerr << '\n';
        }
        std::cout << out.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cyphalm_train: " << e.what() << "\n";
        usage();
        return 1;
    }
}
