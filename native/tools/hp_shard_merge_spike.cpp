/// Lower-level shard merge spike: hp::Predictor directly (see hp/shard_merge.hpp).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
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

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--corpus PATH] [--bytes N] [--table-bits B]\n"
                 "  Splits corpus into two shards, trains independent Predictors,\n"
                 "  merges tables (hp/shard_merge.hpp) and reports merged_bpc.\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus = "bench/data/canterbury/alice29.txt";
    int nbytes = 65536;
    int table_bits = 16;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--corpus" && i + 1 < argc) {
            corpus = argv[++i];
        } else if (arg == "--bytes" && i + 1 < argc) {
            nbytes = std::stoi(argv[++i]);
        } else if (arg == "--table-bits" && i + 1 < argc) {
            table_bits = std::stoi(argv[++i]);
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

    const int n = static_cast<int>(ids.size());
    const int mid = n / 2;

    auto single = make_gate24_model(table_bits);
    const double single_pass_bpc = eval_slice(single, ids, 0, n);

    auto shard_a = make_gate24_model(table_bits);
    train_slice(shard_a, ids, 0, mid);
    const double shard_a_bpc = eval_slice(shard_a, ids, 0, mid);

    auto shard_b = make_gate24_model(table_bits);
    train_slice(shard_b, ids, mid, n);
    const double shard_b_bpc = eval_slice(shard_b, ids, mid, n);

    auto merged_tables = make_gate24_model(table_bits);
    const hp::MergeStatus merge_status = hp::merge_predictor_tables(
        merged_tables.hp_backend().predictor(), shard_a.hp_backend().predictor(),
        static_cast<std::uint64_t>(mid), shard_b.hp_backend().predictor(),
        static_cast<std::uint64_t>(n - mid));

    auto eval_model = make_gate24_model(table_bits);
    eval_model.hp_backend().predictor().transfer_tables_from(merged_tables.hp_backend().predictor());
    const double merged_bpc = observe_full_keep_tables(eval_model, ids);

    std::printf("hp_shard_merge_spike OK corpus=%s bytes=%d table_bits=%d%s\n", resolved.c_str(),
                n, table_bits, synthetic ? " synthetic_fallback=1" : "");
    std::printf("  single_pass_bpc=%.6f\n", single_pass_bpc);
    std::printf("  shard_a_bpc[%d..%d)=%.6f\n", 0, mid, shard_a_bpc);
    std::printf("  shard_b_bpc[%d..%d)=%.6f\n", mid, n, shard_b_bpc);
    std::printf("  merged_bpc=%.6f merge_status=%s delta_vs_single=%.6f\n", merged_bpc,
                merge_status_name(merge_status), merged_bpc - single_pass_bpc);
    std::printf("  docs=docs/reports/CYPHALM_TRAIN_SCALE.md\n");
    return 0;
}
