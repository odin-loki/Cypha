// cyphalm_ssm_diagnose — Phase-5 CellAI SSM probe for D10/D17 (FUTURE.md §0c).
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_encoder_timeseries.hpp"
#include "cypha/cyphalm/cellai_ssm.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_corpus.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/ssm_diagnose.hpp"

namespace {

struct Args {
    std::string domain = "synthetic";
    std::string profile = "d17";
    int steps = 512;
    std::uint64_t seed = 42;
};

void usage() {
    std::cerr << "usage: cyphalm_ssm_diagnose --domain {d10,d17,synthetic}\n"
              << "       [--profile d17|d04] [--steps N] [--seed S]\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (k == "--domain") a.domain = need("--domain");
        else if (k == "--profile") a.profile = need("--profile");
        else if (k == "--steps") a.steps = std::stoi(need("--steps"));
        else if (k == "--seed") a.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
        else if (k == "--help" || k == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

cypha::cyphalm::CellAISSM make_d10_ssm(int d_input, std::uint64_t seed) {
    cypha::cyphalm::CellAISSMConfig cfg;
    cfg.d_input = d_input;
    cfg.d_state = 128;
    cfg.tau_fast = 10.0;
    cfg.tau_slow = 100.0;
    cfg.n_layers = 2;
    cfg.seed = static_cast<int>(seed + 1);
    cfg.use_spectral_pde = true;
    cfg.use_multiscale = true;
    cfg.use_sparse_hebbian = true;
    return cypha::cyphalm::CellAISSM(cfg);
}

nlohmann::json run_d10_diagnose(const Args& args) {
    cypha::bench::TimeSeriesEncoder enc(32, 16);
    const cypha::bench::EcgSplit ecg = cypha::bench::load_ecg5000(args.seed);

    cypha::cyphalm::CellAISSM ssm = make_d10_ssm(enc.feature_dim(), args.seed);
    std::vector<std::vector<double>> inputs;
    inputs.reserve(static_cast<std::size_t>(args.steps));
    for (const auto& series : ecg.x_train) {
        const auto feat = enc.encode_series(series);
        inputs.push_back(cypha::cyphalm::fit_input_dim(
            std::vector<double>(feat.begin(), feat.end()), ssm.d_input()));
        if (static_cast<int>(inputs.size()) >= args.steps) break;
    }
    while (static_cast<int>(inputs.size()) < args.steps) {
        const auto& series = ecg.x_train[static_cast<std::size_t>(inputs.size() % ecg.x_train.size())];
        const auto feat = enc.encode_series(series);
        inputs.push_back(cypha::cyphalm::fit_input_dim(
            std::vector<double>(feat.begin(), feat.end()), ssm.d_input()));
    }

    auto report = cypha::cyphalm::diagnose_cellai_sequence(ssm, inputs, std::max(1, args.steps / 16),
                                                           "d10");
    report["data_source"] = ecg.source;
    report["encoder"] = nlohmann::json{{"window", 32}, {"n_fft", 16}, {"feature_dim", enc.feature_dim()}};
    return report;
}

nlohmann::json run_d17_diagnose(const Args& args) {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_bench_profile(args.profile, cfg);
    cypha::cyphalm::apply_bench_mode(cypha::cyphalm::BenchMode::Hybrid, cfg);
    if (args.profile == "d17" && cfg.vocab_size < 256) cfg.vocab_size = 256;
    if (args.profile == "d04" && cfg.vocab_size < 128) cfg.vocab_size = 128;

    cypha::cyphalm::LMCorpus corpus;
    bool synthetic = false;
    const int n_train = (args.profile == "d04") ? 8000 : 4000;
    try {
        corpus = cypha::cyphalm::load_bench_corpus(args.profile, 500000, cfg.vocab_size,
                                                   cfg.bpe_merges_path, cfg.bpe_vocab_path);
    } catch (const std::exception&) {
        synthetic = true;
        corpus.profile = args.profile;
        corpus.source = "synthetic";
        corpus.vocab_size = cfg.vocab_size;
        corpus.train_ids =
            cypha::cyphalm::synthetic_corpus(n_train + args.steps + 64, cfg.vocab_size, cfg.seed);
        corpus.eval_ids.assign(corpus.train_ids.begin() + static_cast<std::ptrdiff_t>(n_train),
                               corpus.train_ids.end());
        corpus.train_ids.resize(static_cast<std::size_t>(n_train));
    }
    cfg.vocab_size = corpus.vocab_size;

    cypha::cyphalm::CyphaLMModel model(cfg);
    model.train_sequence(corpus.train_ids, n_train, cfg.train_epochs);
    auto report = model.ssm_diagnostic_report(corpus.eval_ids, args.steps);
    report["profile"] = args.profile;
    report["corpus"] = corpus.source;
    report["synthetic"] = synthetic;
    report["domain"] = args.profile == "d04" ? "d04" : "d17";
    return report;
}

nlohmann::json run_synthetic_diagnose(const Args& args) {
    cypha::cyphalm::CellAISSMConfig cfg;
    cfg.d_input = 64;
    cfg.d_state = 128;
    cfg.tau_fast = 1.0;
    cfg.tau_slow = 20.0;
    cfg.n_layers = 2;
    cfg.seed = static_cast<int>(args.seed);
    cfg.use_spectral_pde = false;
    cfg.use_multiscale = true;
    cypha::cyphalm::CellAISSM ssm(cfg);

    std::vector<std::vector<double>> inputs(static_cast<std::size_t>(args.steps));
    for (int t = 0; t < args.steps; ++t) {
        std::vector<double> row(static_cast<std::size_t>(cfg.d_input), 0.0);
        for (int d = 0; d < cfg.d_input; ++d) {
            row[static_cast<std::size_t>(d)] =
                std::sin(0.07 * static_cast<double>(t) + 0.13 * static_cast<double>(d));
        }
        inputs[static_cast<std::size_t>(t)] = std::move(row);
    }
    return cypha::cyphalm::diagnose_cellai_sequence(ssm, inputs, std::max(1, args.steps / 16),
                                                    "synthetic");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        nlohmann::json out;
        if (args.domain == "d10") {
            out = run_d10_diagnose(args);
        } else if (args.domain == "d17" || args.domain == "d04") {
            Args profile_args = args;
            if (args.domain == "d04") profile_args.profile = "d04";
            out = run_d17_diagnose(profile_args);
        } else if (args.domain == "synthetic") {
            out = run_synthetic_diagnose(args);
        } else {
            throw std::runtime_error("unknown --domain: " + args.domain);
        }
        std::cout << out.dump(2) << std::endl;
        return out.value("checks_passed", true) ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << "cyphalm_ssm_diagnose: " << ex.what() << "\n";
        usage();
        return 1;
    }
}
