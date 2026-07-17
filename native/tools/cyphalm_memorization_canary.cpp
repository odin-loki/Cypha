// cyphalm_memorization_canary — MG4: inject unique canaries once, train briefly, measure recall.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

struct Args {
    fs::path out_path;
    int n_canaries = 6;
    int canary_len = 10;
    int prefix_len = 0;
    int train_epochs = 40;
    std::uint64_t seed = 42;
    bool write_table = false;
};

struct CharCodec {
    std::unordered_map<char, int> c2i;
    std::vector<char> i2c;
    int vocab_size = 128;
};

void usage() {
    std::cerr
        << "usage: cyphalm_memorization_canary [--out PATH] [--n-canaries N]\n"
        << "       [--canary-len L] [--prefix-len P] [--train-epochs E] [--seed S]\n"
        << "       [--write-table]\n"
        << "FAST (CYPHA_BENCH_FAST=1): n_canaries=3, canary_len=6, train_epochs=60.\n";
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
        } else if (arg == "--n-canaries") {
            a.n_canaries = std::stoi(need("--n-canaries"));
        } else if (arg == "--canary-len") {
            a.canary_len = std::stoi(need("--canary-len"));
        } else if (arg == "--prefix-len") {
            a.prefix_len = std::stoi(need("--prefix-len"));
        } else if (arg == "--train-epochs") {
            a.train_epochs = std::stoi(need("--train-epochs"));
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
        a.n_canaries = 3;
        a.canary_len = 6;
        a.train_epochs = 60;
        if (a.prefix_len <= 0) {
            a.prefix_len = 4;  // complete 2 chars
        }
    }
    if (a.n_canaries < 1) {
        throw std::runtime_error("n_canaries must be >= 1");
    }
    if (a.canary_len < 4) {
        throw std::runtime_error("canary_len must be >= 4");
    }
    if (a.prefix_len <= 0) {
        a.prefix_len = std::max(4, a.canary_len / 2);
    }
    if (a.prefix_len >= a.canary_len) {
        throw std::runtime_error("prefix_len must be < canary_len");
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

std::string random_canary(int len, std::mt19937_64& rng) {
    static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string out = "MG";
    out.reserve(static_cast<std::size_t>(len));
    std::uniform_int_distribution<int> dist(0, 25);
    for (int i = static_cast<int>(out.size()); i < len; ++i) {
        out.push_back(alphabet[dist(rng)]);
    }
    return out;
}

/// Each canary appears exactly once in the train string.
std::string build_train_text(const std::vector<std::string>& canaries) {
    std::string text;
    for (const auto& c : canaries) {
        text += "[[";
        text += c;
        text += "]] ";
    }
    return text;
}

int argmax_log_probs(const std::vector<double>& lp) {
    int best = 0;
    double mx = lp.empty() ? -1e30 : lp[0];
    for (int i = 1; i < static_cast<int>(lp.size()); ++i) {
        if (lp[static_cast<std::size_t>(i)] > mx) {
            mx = lp[static_cast<std::size_t>(i)];
            best = i;
        }
    }
    return best;
}

/// Teacher-forced completion: gold prefix context, check greedy next-token on suffix.
bool measure_teacher_forced_recall(cypha::cyphalm::CyphaLMModel& model, const CharCodec& codec,
                                   const std::string& canary, int prefix_len,
                                   std::string& predicted_suffix) {
    const std::string full = "[[" + canary + "]]";
    const std::vector<int> ids = encode_text(full, codec);
    // full = "[[" + canary + "]]" ; canary starts at index 2
    const int suffix_begin = 2 + prefix_len;
    const int suffix_end = 2 + static_cast<int>(canary.size());

    model.reset_context();
    predicted_suffix.clear();
    std::vector<int> pred_ids;
    bool ok = true;
    for (int i = 0; i + 1 < static_cast<int>(ids.size()); ++i) {
        const auto pred = model.predict_next(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]));
        const int next_pred = argmax_log_probs(pred.log_probs);
        const int next_gold = ids[static_cast<std::size_t>(i + 1)];
        if (i + 1 >= suffix_begin && i + 1 < suffix_end) {
            pred_ids.push_back(next_pred);
            if (next_pred != next_gold) {
                ok = false;
            }
        }
    }
    predicted_suffix = decode_ids(pred_ids, codec);
    return ok && static_cast<int>(pred_ids.size()) == (suffix_end - suffix_begin);
}

Json run_canary_experiment(const Args& args) {
    std::mt19937_64 rng(args.seed);
    std::vector<std::string> canaries;
    canaries.reserve(static_cast<std::size_t>(args.n_canaries));
    for (int i = 0; i < args.n_canaries; ++i) {
        std::string c;
        do {
            c = random_canary(args.canary_len, rng);
        } while (std::find(canaries.begin(), canaries.end(), c) != canaries.end());
        canaries.push_back(c);
    }

    const std::string train_text = build_train_text(canaries);
    CharCodec codec;
    build_vocab(train_text, 64, codec);

    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.vocab_size = codec.vocab_size;
    cfg.lstm_hidden = 128;
    cfg.context_mode = cypha::cyphalm::ContextMode::CharLstm;
    cfg.lstm_lr = 0.4;
    cfg.train_epochs = args.train_epochs;
    cfg.seed = static_cast<std::int64_t>(args.seed);
    cfg.ngram_context = 0;
    cfg.d_embed = 64;
    cfg.field_dim = 64;

    const std::vector<int> train_ids = encode_text(train_text, codec);
    cypha::cyphalm::CyphaLMModel model(cfg);
    model.train_sequence(train_ids, static_cast<int>(train_ids.size()) - 1, args.train_epochs,
                         nullptr);

    int n_recalled = 0;
    Json canary_rows = Json::array();
    for (const auto& canary : canaries) {
        std::string predicted_suffix;
        const bool recalled =
            measure_teacher_forced_recall(model, codec, canary, args.prefix_len, predicted_suffix);
        if (recalled) {
            ++n_recalled;
        }
        Json row;
        row["canary"] = canary;
        row["prefix"] = "[[" + canary.substr(0, static_cast<std::size_t>(args.prefix_len));
        row["expected_suffix"] = canary.substr(static_cast<std::size_t>(args.prefix_len));
        row["generated_suffix"] = predicted_suffix;
        row["recalled"] = recalled;
        canary_rows.push_back(std::move(row));
    }

    const double recall_rate =
        args.n_canaries > 0 ? static_cast<double>(n_recalled) / static_cast<double>(args.n_canaries)
                            : 0.0;

    Json out;
    out["canary_id"] = "memorization_canary";
    out["runner"] = "cyphalm_memorization_canary";
    out["metric"] = "recall_rate";
    out["recall_mode"] = "teacher_forced_suffix";
    out["n_canaries"] = args.n_canaries;
    out["canary_len"] = args.canary_len;
    out["prefix_len"] = args.prefix_len;
    out["train_epochs"] = args.train_epochs;
    out["train_chars"] = static_cast<int>(train_text.size());
    out["vocab_size"] = codec.vocab_size;
    out["context_mode"] = "char_lstm";
    out["seed"] = args.seed;
    out["fast"] = cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST");
    out["n_recalled"] = n_recalled;
    out["recall_rate"] = recall_rate;
    out["canaries"] = std::move(canary_rows);
    return out;
}

fs::path default_results_path() {
    const fs::path native = fs::path(__FILE__).parent_path().parent_path();
    return native.parent_path() / "bench" / "results" / "memorization_canary.json";
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
        const Json report = run_canary_experiment(args);

        const fs::path out = args.out_path.empty() ? default_results_path() : args.out_path;
        write_json_file(out, report);

        if (args.write_table) {
            const fs::path table = fs::path(__FILE__).parent_path().parent_path().parent_path() /
                                   "bench" / "report" / "tables" / "memorization_canary.json";
            write_json_file(table, report);
        }

        std::printf("memorization_canary: recall_rate=%.4f (%d/%d) -> %s\n",
                    report["recall_rate"].get<double>(), report["n_recalled"].get<int>(),
                    report["n_canaries"].get<int>(), out.string().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cyphalm_memorization_canary: %s\n", e.what());
        return 1;
    }
}
