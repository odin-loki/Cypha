// cyphalm_algebraic_fingerprint — MS2: fingerprint vector on generated + scored CyphaLM text.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/text_algebraic_fingerprint.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

struct Args {
    fs::path out_path;
    int train_epochs = 30;
    int max_gen_tokens = 24;
    std::uint64_t seed = 42;
    bool write_table = false;
};

struct CharCodec {
    std::unordered_map<char, int> c2i;
    std::vector<char> i2c;
    int vocab_size = 128;
};

void usage() {
    std::cerr << "usage: cyphalm_algebraic_fingerprint [--out PATH] [--train-epochs E]\n"
              << "       [--max-gen-tokens N] [--seed S] [--write-table]\n"
              << "FAST (CYPHA_BENCH_FAST=1): train_epochs=20, max_gen_tokens=16.\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };
        if (arg == "--out") {
            a.out_path = need("--out");
        } else if (arg == "--train-epochs") {
            a.train_epochs = std::stoi(need("--train-epochs"));
        } else if (arg == "--max-gen-tokens") {
            a.max_gen_tokens = std::stoi(need("--max-gen-tokens"));
        } else if (arg == "--seed") {
            a.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
        } else if (arg == "--write-table") {
            a.write_table = true;
        } else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + arg);
        }
    }
    if (cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST")) {
        a.train_epochs = 20;
        a.max_gen_tokens = 16;
    }
    return a;
}

void build_vocab(const std::string& text, int vocab_size, CharCodec& codec) {
    std::vector<char> uniq;
    for (unsigned char ch : text) {
        const char c = static_cast<char>(ch);
        if (std::find(uniq.begin(), uniq.end(), c) == uniq.end()) {
            uniq.push_back(c);
        }
    }
    std::sort(uniq.begin(), uniq.end());
    const int limit = std::max(1, vocab_size - 1);
    if (static_cast<int>(uniq.size()) > limit) {
        uniq.resize(static_cast<std::size_t>(limit));
    }
    codec.c2i.clear();
    codec.i2c.assign(static_cast<std::size_t>(limit + 1), '?');
    codec.c2i['?'] = 0;
    codec.i2c[0] = '?';
    for (std::size_t i = 0; i < uniq.size(); ++i) {
        const int id = static_cast<int>(i + 1);
        codec.c2i[uniq[i]] = id;
        codec.i2c[static_cast<std::size_t>(id)] = uniq[i];
    }
    codec.vocab_size = limit + 1;
}

std::vector<int> encode_text(const std::string& text, const CharCodec& codec) {
    std::vector<int> out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        const auto it = codec.c2i.find(static_cast<char>(ch));
        out.push_back(it == codec.c2i.end() ? 0 : it->second);
    }
    return out;
}

std::string decode_ids(const std::vector<int>& ids, const CharCodec& codec) {
    std::string out;
    out.reserve(ids.size());
    for (int id : ids) {
        if (id >= 0 && id < static_cast<int>(codec.i2c.size())) {
            out.push_back(codec.i2c[static_cast<std::size_t>(id)]);
        } else {
            out.push_back('?');
        }
    }
    return out;
}

Json sample_block(const cypha::cyphalm::TextAlgebraicFingerprint& fp, const std::string& text) {
    Json block = cypha::cyphalm::fingerprint_to_json(fp, true);
    block["text_len"] = static_cast<int>(text.size());
    block["text_preview"] = text.substr(0, std::min<std::size_t>(48, text.size()));
    return block;
}

Json run_ms2_experiment(const Args& args) {
    static const std::string kTrain =
        "the quick brown fox jumps over the lazy dog. "
        "algebraic fingerprints track structure in generated language model output. "
        "MS2 linear complexity spectral flatness run-length entropy ngram entropy. ";

    CharCodec codec;
    build_vocab(kTrain, 64, codec);

    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.vocab_size = codec.vocab_size;
    cfg.lstm_hidden = 96;
    cfg.context_mode = cypha::cyphalm::ContextMode::CharLstm;
    cfg.lstm_lr = 0.35;
    cfg.train_epochs = args.train_epochs;
    cfg.seed = static_cast<std::int64_t>(args.seed);
    cfg.ngram_context = 0;
    cfg.d_embed = 48;
    cfg.field_dim = 48;

    const std::vector<int> train_ids = encode_text(kTrain, codec);
    cypha::cyphalm::CyphaLMModel model(cfg);
    model.train_sequence(train_ids, static_cast<int>(train_ids.size()) - 1, args.train_epochs, nullptr);

    const std::string prompt = "algebraic ";
    const std::vector<int> prompt_ids = encode_text(prompt, codec);
    const auto gen = cypha::cyphalm::generate_greedy(model, prompt_ids, args.max_gen_tokens);
    const std::string generated = prompt + decode_ids(gen.generated_ids, codec);

    const auto scored_fp = cypha::cyphalm::compute_text_algebraic_fingerprint(kTrain);
    const auto generated_fp = cypha::cyphalm::compute_text_algebraic_fingerprint(generated);

    Json out;
    out["metric_id"] = "MS2";
    out["runner"] = "cyphalm_algebraic_fingerprint";
    out["context_mode"] = "char_lstm";
    out["train_epochs"] = args.train_epochs;
    out["max_gen_tokens"] = args.max_gen_tokens;
    out["seed"] = args.seed;
    out["fast"] = cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST");
    out["scored_train"] = sample_block(scored_fp, kTrain);
    out["generated"] = sample_block(generated_fp, generated);
    return out;
}

fs::path default_results_path() {
    const fs::path native = fs::path(__FILE__).parent_path().parent_path();
    return native.parent_path() / "bench" / "results" / "algebraic_fingerprint_ms2.json";
}

void write_json_file(const fs::path& path, const Json& j) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write " + path.string());
    }
    out << j.dump(2) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const Json report = run_ms2_experiment(args);

        const fs::path out = args.out_path.empty() ? default_results_path() : args.out_path;
        write_json_file(out, report);

        if (args.write_table) {
            const fs::path table = fs::path(__FILE__).parent_path().parent_path().parent_path() /
                                   "bench" / "report" / "tables" / "algebraic_fingerprint_ms2.json";
            write_json_file(table, report);
        }

        const double sp = report["generated"]["spectrum_position"].get<double>();
        std::printf("algebraic_fingerprint_ms2: generated spectrum_position=%.4f -> %s\n", sp,
                    out.string().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cyphalm_algebraic_fingerprint: %s\n", e.what());
        return 1;
    }
}
