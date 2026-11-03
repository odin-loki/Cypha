/// Lower-level shard merge spike: hp::Predictor directly (see hp/shard_merge.hpp).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "hp/shard_merge.hpp"

namespace {

std::vector<int> synthetic_bytes(int n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<int> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = dist(rng);
    }
    return out;
}

std::string resolve_corpus_path(const std::string& path) {
    if (std::filesystem::exists(path)) {
        return path;
    }
    const std::string up = std::string("../") + path;
    if (std::filesystem::exists(up)) {
        return up;
    }
    const std::string up2 = std::string("../../") + path;
    if (std::filesystem::exists(up2)) {
        return up2;
    }
    return path;
}

std::vector<int> load_bytes(const std::string& path, int max_n) {
    const std::string resolved = resolve_corpus_path(path);
    std::ifstream in(resolved, std::ios::binary);
    std::vector<int> out;
    int ch = 0;
    while (in && (max_n <= 0 || static_cast<int>(out.size()) < max_n)) {
        ch = in.get();
        if (ch == std::char_traits<char>::eof()) {
            break;
        }
        out.push_back(ch & 0xff);
    }
    return out;
}

std::pair<std::vector<int>, std::vector<int>> split_train_holdout(const std::vector<int>& ids,
                                                                  double holdout_frac) {
    if (holdout_frac <= 0.0 || ids.size() < 4) {
        return {ids, {}};
    }
    const double frac = std::min(0.9, std::max(0.0, holdout_frac));
    const std::size_t holdout_n =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(ids.size() * frac)));
    if (holdout_n >= ids.size()) {
        return {ids, {}};
    }
    const std::size_t train_n = ids.size() - holdout_n;
    return {std::vector<int>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(train_n)),
            std::vector<int>(ids.begin() + static_cast<std::ptrdiff_t>(train_n), ids.end())};
}

double eval_slice(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids, int begin,
                  int end) {
    if (begin >= end) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<int> slice(ids.begin() + begin, ids.begin() + end);
    return model.eval_bpc_compress_equivalent(slice, static_cast<int>(slice.size()));
}

void train_slice(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids, int begin,
                 int end) {
    model.reset_context();
    for (int i = begin; i < end; ++i) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(ids[static_cast<std::size_t>(i)]));
    }
}

void consume_range(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids,
                   std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end && i < ids.size(); ++i) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(ids[i]));
    }
}

double observe_full_keep_tables(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids) {
    if (ids.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<std::uint8_t> bytes(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>(ids[i]);
    }
    const double bits = model.hp_backend().observe_stream_bits(bytes.data(), bytes.size());
    return bits / static_cast<double>(ids.size());
}

cypha::cyphalm::CyphaLMModel make_gate24_model(int table_bits) {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = table_bits;
    cypha::cyphalm::normalize_hp_table_bits(cfg);
    return cypha::cyphalm::CyphaLMModel(cfg);
}

const char* merge_status_name(hp::MergeStatus s) {
    switch (s) {
        case hp::MergeStatus::Ok:
            return "ok";
        case hp::MergeStatus::ConfigMismatch:
            return "config_mismatch";
        case hp::MergeStatus::EmptyInput:
            return "empty_input";
    }
    return "unknown";
}

void boundary_replay_train_tail(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& train_ids,
                               std::size_t replay_bytes) {
    if (replay_bytes == 0 || train_ids.empty()) {
        return;
    }
    const std::size_t boundary = train_ids.size();
    const std::size_t begin = hp::boundary_replay_begin(boundary, replay_bytes);
    consume_range(model, train_ids, begin, boundary);
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--corpus PATH] [--bytes N] [--table-bits B] "
                 "[--holdout-frac F] [--boundary-replay-bytes W]\n"
                 "  Splits corpus, trains independent Predictors on train shards,\n"
                 "  merges tables (hp/shard_merge.hpp), reports merged + holdout BPC.\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus = "bench/data/canterbury/alice29.txt";
    int nbytes = 65536;
    int table_bits = 16;
    double holdout_frac = 0.2;
    int boundary_replay_bytes = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--corpus" && i + 1 < argc) {
            corpus = argv[++i];
        } else if (arg == "--bytes" && i + 1 < argc) {
            nbytes = std::stoi(argv[++i]);
        } else if (arg == "--table-bits" && i + 1 < argc) {
            table_bits = std::stoi(argv[++i]);
        } else if (arg == "--holdout-frac" && i + 1 < argc) {
            holdout_frac = std::stod(argv[++i]);
        } else if (arg == "--boundary-replay-bytes" && i + 1 < argc) {
            boundary_replay_bytes = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    auto ids = load_bytes(corpus, nbytes);
    const std::string resolved = resolve_corpus_path(corpus);
    bool synthetic = false;
    if (ids.size() < 2) {
        ids = synthetic_bytes(nbytes, 42);
        synthetic = true;
    }

    const auto [train_ids, holdout_ids] = split_train_holdout(ids, holdout_frac);
    const bool holdout_enabled = !holdout_ids.empty();
    const int n = static_cast<int>(ids.size());
    const int train_n = static_cast<int>(train_ids.size());
    const int mid = train_n / 2;

    std::size_t replay_bytes = 0;
    if (holdout_enabled) {
        if (boundary_replay_bytes > 0) {
            replay_bytes = static_cast<std::size_t>(boundary_replay_bytes);
        } else if (train_ids.size() >= 512) {
            replay_bytes = std::min<std::size_t>(4096, std::max<std::size_t>(256, train_ids.size() / 10));
        }
    }

    auto single = make_gate24_model(table_bits);
    const double single_pass_bpc = eval_slice(single, ids, 0, n);

    auto shard_a = make_gate24_model(table_bits);
    train_slice(shard_a, train_ids, 0, mid);
    auto shard_b = make_gate24_model(table_bits);
    train_slice(shard_b, train_ids, mid, train_n);

    auto merged_tables = make_gate24_model(table_bits);
    const hp::MergeStatus merge_status = hp::merge_predictor_tables(
        merged_tables.hp_backend().predictor(), shard_a.hp_backend().predictor(),
        static_cast<std::uint64_t>(mid), shard_b.hp_backend().predictor(),
        static_cast<std::uint64_t>(train_n - mid));

    auto eval_model = make_gate24_model(table_bits);
    eval_model.hp_backend().predictor().transfer_tables_from(merged_tables.hp_backend().predictor());
    const double merged_in_sample_bpc = observe_full_keep_tables(eval_model, ids);

    double merged_holdout_bpc = std::numeric_limits<double>::quiet_NaN();
    double single_holdout_bpc = std::numeric_limits<double>::quiet_NaN();
    if (holdout_enabled) {
        auto merged_holdout_model = make_gate24_model(table_bits);
        merged_holdout_model.hp_backend().predictor().transfer_tables_from(
            merged_tables.hp_backend().predictor());
        if (replay_bytes > 0) {
            const std::size_t join = static_cast<std::size_t>(mid);
            consume_range(merged_holdout_model, train_ids,
                          hp::boundary_replay_begin(join, replay_bytes), join);
            boundary_replay_train_tail(merged_holdout_model, train_ids, replay_bytes);
        }
        merged_holdout_bpc = observe_full_keep_tables(merged_holdout_model, holdout_ids);

        auto single_train = make_gate24_model(table_bits);
        train_slice(single_train, train_ids, 0, train_n);
        auto single_eval = make_gate24_model(table_bits);
        single_eval.hp_backend().predictor().transfer_tables_from(
            single_train.hp_backend().predictor());
        if (replay_bytes > 0) {
            boundary_replay_train_tail(single_eval, train_ids, replay_bytes);
        }
        single_holdout_bpc = observe_full_keep_tables(single_eval, holdout_ids);
    }

    std::printf("hp_shard_merge_spike OK corpus=%s bytes=%d train=%d holdout=%zu table_bits=%d%s\n",
                resolved.c_str(), n, train_n, holdout_ids.size(), table_bits,
                synthetic ? " synthetic_fallback=1" : "");
    std::printf("  single_pass_bpc=%.6f\n", single_pass_bpc);
    std::printf("  merged_in_sample_bpc=%.6f merge_status=%s\n", merged_in_sample_bpc,
                merge_status_name(merge_status));
    if (holdout_enabled) {
        std::printf("  boundary_replay_bytes=%zu\n", replay_bytes);
        std::printf("  merged_holdout_bpc=%.6f\n", merged_holdout_bpc);
        std::printf("  single_stream_holdout_bpc=%.6f\n", single_holdout_bpc);
        std::printf("  merged_holdout_delta=%.6f\n", merged_holdout_bpc - single_holdout_bpc);
    }
    std::printf("  docs=docs/reports/CYPHALM_TRAIN_SCALE.md\n");
    return 0;
}
