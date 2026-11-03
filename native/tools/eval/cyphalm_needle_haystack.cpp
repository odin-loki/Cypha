// cyphalm_needle_haystack — MC3/MG3: long-range recall (needle in haystack).
// Plants a unique fact early, pads with filler, prompts at end; scores char-LSTM recall + BPC.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_generation.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

constexpr double kLog2 = 0.6931471805599453;

struct Args {
    fs::path out_path;
    int needle_len = 10;
    int train_epochs = 5;
    std::vector<int> haystack_chars;
    std::uint64_t seed = 42;
    bool write_table = false;
    /// Opt-in: recap fact before QUESTION + extra teacher-forced prefix passes (SSM/context warm-up).
    bool context_warmup = false;
    int warmup_passes = 2;
};

struct CharCodec {
    std::unordered_map<char, int> c2i;
    std::vector<char> i2c;
    int vocab_size = 128;
};

void usage() {
    std::cerr
        << "usage: cyphalm_needle_haystack [--out PATH] [--needle-len L]\n"
        << "       [--haystack-chars N,N,...] [--train-epochs E] [--seed S] [--write-table]\n"
        << "       [--context-warmup] [--warmup-passes N]\n"
        << "FAST (CYPHA_BENCH_FAST=1): haystack 32,64; train_epochs=15.\n"
        << "  --context-warmup: recap fact before QUESTION + N prefix warm-up passes at eval.\n";
}

std::vector<int> default_haystack_chars() {
    if (cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST")) {
        return {32, 64};
    }
    return {256, 512, 1024};
}

std::vector<int> parse_int_csv(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        out.push_back(std::stoi(part));
    }
    if (out.size() < 1) {
        throw std::runtime_error("haystack-chars requires at least one value");
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    a.haystack_chars = default_haystack_chars();
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
        } else if (arg == "--needle-len") {
            a.needle_len = std::stoi(need("--needle-len"));
        } else if (arg == "--haystack-chars") {
            a.haystack_chars = parse_int_csv(need("--haystack-chars"));
        } else if (arg == "--train-epochs") {
            a.train_epochs = std::stoi(need("--train-epochs"));
        } else if (arg == "--seed") {
            a.seed = static_cast<std::uint64_t>(std::stoull(need("--seed")));
        } else if (arg == "--write-table") {
            a.write_table = true;
        } else if (arg == "--context-warmup") {
            a.context_warmup = true;
        } else if (arg == "--warmup-passes") {
            a.warmup_passes = std::stoi(need("--warmup-passes"));
        } else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + arg);
        }
    }
    if (cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST")) {
        a.train_epochs = 15;
    }
    if (a.needle_len < 4) {
        throw std::runtime_error("needle_len must be >= 4");
    }
    if (a.warmup_passes < 1) {
        throw std::runtime_error("warmup_passes must be >= 1");
    }
    return a;
}

void build_vocab(const std::string& text, int vocab_size, CharCodec& codec) {
    std::vector<char> uniq;
    uniq.reserve(256);
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

std::string random_needle(int len, std::uint64_t seed) {
    static const char* alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
    std::string out = "MG3";
    out.reserve(static_cast<std::size_t>(len));
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, 35);
    for (int i = static_cast<int>(out.size()); i < len; ++i) {
        out.push_back(alphabet[dist(rng) % 36]);
    }
    return out;
}

std::string haystack_filler(int target_chars, std::uint64_t seed) {
    static const char* unit =
        " the quick brown fox jumps over the lazy dog 0123456789 "
        "abcdefghijklmnopqrstuvwxyz ";
    std::string out;
    out.reserve(static_cast<std::size_t>(target_chars));
    std::mt19937_64 rng(seed + 991);
    while (static_cast<int>(out.size()) < target_chars) {
        out += unit;
        if (static_cast<int>(out.size()) > target_chars) {
            out.resize(static_cast<std::size_t>(target_chars));
        }
        if ((rng() & 3u) == 0u) {
            out.push_back(static_cast<char>('a' + static_cast<int>(rng() % 26)));
        }
    }
    return out;
}

struct NeedleSequence {
    std::string text;
    std::string needle;
    std::string prompt_prefix;
    int answer_start = 0;
};

NeedleSequence build_sequence(const std::string& needle, int haystack_chars, std::uint64_t seed,
                              bool context_warmup) {
    NeedleSequence seq;
    seq.needle = needle;
    const std::string fact = "FACT: The secret code is [[" + needle + "]].\n";
    const std::string haystack = haystack_filler(haystack_chars, seed);
    const std::string recap =
        context_warmup ? ("RECAP: The secret code is [[" + needle + "]].\n") : std::string{};
    seq.prompt_prefix = "QUESTION: What is the secret code? ANSWER: ";
    seq.text = fact + haystack + recap + seq.prompt_prefix + needle;
    seq.answer_start =
        static_cast<int>(fact.size() + haystack.size() + recap.size() + seq.prompt_prefix.size());
    return seq;
}

void warmup_prefix(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids, int answer_start,
                   int warmup_passes) {
    model.reset_context();
    const int passes = std::max(1, warmup_passes);
    for (int pass = 0; pass < passes; ++pass) {
        for (int i = 0; i < answer_start - 1; ++i) {
            model.predict_next(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]));
        }
    }
}

double eval_answer_bpc(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids,
                       int answer_start, int warmup_passes) {
    const int vocab = static_cast<int>(model.config().vocab_size);
    if (answer_start < 1 || answer_start >= static_cast<int>(ids.size())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    warmup_prefix(model, ids, answer_start, warmup_passes);
    double bits = 0.0;
    int scored = 0;
    for (int i = answer_start - 1; i < static_cast<int>(ids.size()) - 1; ++i) {
        const auto pred =
            model.predict_next(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i)]));
        const int nxt = ids[static_cast<std::size_t>(i + 1)];
        if (nxt < 0 || nxt >= vocab ||
            static_cast<std::size_t>(nxt) >= pred.log_probs.size()) {
            continue;
        }
        bits += -pred.log_probs[static_cast<std::size_t>(nxt)] / kLog2;
        ++scored;
    }
    if (scored <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return bits / static_cast<double>(scored);
}

int argmax_token(const std::vector<double>& log_probs) {
    if (log_probs.empty()) return 0;
    int best = 0;
    double best_lp = log_probs[0];
    for (int i = 1; i < static_cast<int>(log_probs.size()); ++i) {
        if (log_probs[static_cast<std::size_t>(i)] > best_lp) {
            best_lp = log_probs[static_cast<std::size_t>(i)];
            best = i;
        }
    }
    return best;
}

double eval_token_recall(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids,
                         int answer_start, int warmup_passes) {
    const int vocab = static_cast<int>(model.config().vocab_size);
    if (answer_start < 1 || answer_start >= static_cast<int>(ids.size())) {
        return 0.0;
    }
    warmup_prefix(model, ids, answer_start, warmup_passes);
    int correct = 0;
    int total = 0;
    for (int i = answer_start; i < static_cast<int>(ids.size()); ++i) {
        const auto pred =
            model.predict_next(static_cast<std::uint32_t>(ids[static_cast<std::size_t>(i - 1)]));
        const int expected = ids[static_cast<std::size_t>(i)];
        if (expected < 0 || expected >= vocab ||
            static_cast<std::size_t>(expected) >= pred.log_probs.size()) {
            continue;
        }
        if (argmax_token(pred.log_probs) == expected) {
            ++correct;
        }
        ++total;
    }
    if (total <= 0) {
        return 0.0;
    }
    return static_cast<double>(correct) / static_cast<double>(total);
}

Json run_depth_tier(const Args& args, const std::string& needle, int haystack_chars,
                    const CharCodec& codec) {
    const NeedleSequence seq =
        build_sequence(needle, haystack_chars, args.seed + haystack_chars, args.context_warmup);
    const int warmup_passes = args.context_warmup ? args.warmup_passes : 1;
    const std::vector<int> ids = encode_text(seq.text, codec);
    const int answer_start = seq.answer_start;
    if (answer_start <= 0 || answer_start >= static_cast<int>(seq.text.size())) {
        throw std::runtime_error("invalid answer_start for haystack=" + std::to_string(haystack_chars));
    }

    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.vocab_size = codec.vocab_size;
    cfg.lstm_hidden = cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST") ? 64 : 96;
    cfg.context_mode = cypha::cyphalm::ContextMode::CharLstm;
    cfg.lstm_lr = cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST") ? 0.25 : 0.12;
    cfg.train_epochs = args.train_epochs;
    cfg.seed = static_cast<std::int64_t>(args.seed);
    cfg.ngram_context = 0;

    cypha::cyphalm::CyphaLMModel model(cfg);
    model.train_sequence(ids, static_cast<int>(ids.size()) - 1, args.train_epochs, nullptr);

    const std::vector<int> prompt_ids =
        encode_text(seq.text.substr(0, static_cast<std::size_t>(answer_start)), codec);
    const int suffix_len = static_cast<int>(needle.size());
    const auto gen = cypha::cyphalm::generate_greedy(model, prompt_ids, suffix_len);
    const std::string generated = decode_ids(gen.generated_ids, codec);
    const bool exact_recall =
        static_cast<int>(gen.generated_ids.size()) >= suffix_len &&
        generated.substr(0, static_cast<std::size_t>(suffix_len)) == needle;

    const double bpc_answer = eval_answer_bpc(model, ids, answer_start, warmup_passes);
    const double token_recall = eval_token_recall(model, ids, answer_start, warmup_passes);

    Json row;
    row["haystack_chars"] = haystack_chars;
    row["context_chars"] = answer_start;
    row["needle"] = needle;
    row["generated"] = generated.substr(0, static_cast<std::size_t>(suffix_len));
    row["exact_recall"] = exact_recall;
    row["token_recall"] = token_recall;
    row["recalled"] = token_recall >= 1.0;
    row["bpc_answer"] = std::isnan(bpc_answer) ? Json(nullptr) : Json(bpc_answer);
    row["warmup_passes"] = warmup_passes;
    return row;
}

Json run_experiment(const Args& args) {
    const std::string needle = random_needle(args.needle_len, args.seed);

    std::string vocab_text;
    for (int depth : args.haystack_chars) {
        vocab_text += build_sequence(needle, depth, args.seed + depth, args.context_warmup).text;
    }

    CharCodec codec;
    build_vocab(vocab_text, 128, codec);

    Json depths = Json::array();
    int n_recalled = 0;
    double token_recall_sum = 0.0;
    for (int depth : args.haystack_chars) {
        Json row = run_depth_tier(args, needle, depth, codec);
        if (row.value("recalled", false)) {
            ++n_recalled;
        }
        token_recall_sum += row.value("token_recall", 0.0);
        depths.push_back(std::move(row));
    }

    const double recall_rate =
        args.haystack_chars.empty()
            ? 0.0
            : token_recall_sum / static_cast<double>(args.haystack_chars.size());

    Json out;
    out["haystack_id"] = "needle_haystack";
    out["runner"] = "cyphalm_needle_haystack";
    out["metric"] = "token_recall";
    out["needle_len"] = args.needle_len;
    out["needle"] = needle;
    out["train_epochs"] = args.train_epochs;
    out["context_mode"] = "char_lstm";
    out["seed"] = args.seed;
    out["fast"] = cypha::bench::bench_env_truthy("CYPHA_BENCH_FAST");
    out["context_warmup"] = args.context_warmup;
    out["warmup_passes"] = args.context_warmup ? args.warmup_passes : 1;
    out["haystack_chars_requested"] = args.haystack_chars;
    out["n_depths"] = static_cast<int>(args.haystack_chars.size());
    out["n_recalled"] = n_recalled;
    out["recall_rate"] = recall_rate;
    out["depths"] = std::move(depths);
    return out;
}

fs::path default_results_path() {
    return cypha::bench::results_dir() / "needle_haystack.json";
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
        const Json report = run_experiment(args);

        const fs::path out = args.out_path.empty() ? default_results_path() : args.out_path;
        write_json_file(out, report);

        if (args.write_table) {
            const fs::path table =
                cypha::bench::repo_root() / "bench" / "report" / "tables" / "needle_haystack.json";
            write_json_file(table, report);
        }

        std::printf("needle_haystack: recall_rate=%.4f (%d/%d) -> %s\n",
                    report["recall_rate"].get<double>(), report["n_recalled"].get<int>(),
                    report["n_depths"].get<int>(), out.string().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cyphalm_needle_haystack: %s\n", e.what());
        return 1;
    }
}
