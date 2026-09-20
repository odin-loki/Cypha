/// Minimal shard + adapt spike for gate24 hp observe BPC.
/// Splits a byte corpus, adapts independent predictors per shard, stubs merge.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"
#include "hp/shard_merge.hpp"

namespace {

std::vector<int> load_bytes(const std::string& path, int max_bytes) {
    std::ifstream in(path, std::ios::binary);
    std::vector<int> out;
    int ch = 0;
    while (in && (max_bytes <= 0 || static_cast<int>(out.size()) < max_bytes)) {
        ch = in.get();
        if (ch == std::char_traits<char>::eof()) {
            break;
        }
        out.push_back(ch & 0xff);
    }
    return out;
}

cypha::cyphalm::CyphaLMConfig spike_cfg(int table_bits_override) {
    cypha::cyphalm::CyphaLMConfig cfg;
    cfg.vocab_size = 256;
    if (table_bits_override > 0) {
        cfg.hp_table_bits = table_bits_override;
    } else {
        cfg.hp_table_bits = 16;
    }
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    if (table_bits_override > 0) {
        cfg.hp_table_bits = table_bits_override;
        cypha::cyphalm::normalize_hp_table_bits(cfg);
    }
    return cfg;
}

double observe_bpc_ids(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids) {
    if (ids.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    model.reset_context();
    return model.eval_bpc(ids, static_cast<int>(ids.size()), nullptr);
}

void adapt_consume_ids(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids) {
    model.reset_context();
    for (int b : ids) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(b));
    }
}

std::vector<std::vector<int>> split_shards(const std::vector<int>& data, int n_shards) {
    n_shards = std::max(1, n_shards);
    std::vector<std::vector<int>> shards(static_cast<std::size_t>(n_shards));
    const std::size_t base = data.size() / static_cast<std::size_t>(n_shards);
    const std::size_t rem = data.size() % static_cast<std::size_t>(n_shards);
    std::size_t off = 0;
    for (int s = 0; s < n_shards; ++s) {
        const std::size_t len = base + (static_cast<std::size_t>(s) < rem ? 1 : 0);
        shards[static_cast<std::size_t>(s)].assign(
            data.begin() + static_cast<std::ptrdiff_t>(off),
            data.begin() + static_cast<std::ptrdiff_t>(off + len));
        off += len;
    }
    return shards;
}

bool write_shard_files(const std::vector<std::vector<int>>& shards, const std::string& out_dir) {
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        return false;
    }
    for (std::size_t i = 0; i < shards.size(); ++i) {
        const std::string path = out_dir + "/shard" + std::to_string(i) + ".bin";
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            return false;
        }
        for (int b : shards[i]) {
            out.put(static_cast<char>(b & 0xff));
        }
    }
    return true;
}

struct MergeAttempt {
    bool ok = false;
    std::string status = "stub";
    std::string detail;
    double merged_bpc = std::numeric_limits<double>::quiet_NaN();
};

double observe_bpc_keep_tables(cypha::cyphalm::CyphaLMModel& model, const std::vector<int>& ids) {
    if (ids.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Caller must provide a fresh predictor (or call reset_stream_state only after
    // transfer_tables_from onto a newly constructed model).
    std::vector<std::uint8_t> bytes(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>(ids[i]);
    }
    const double bits = model.hp_backend().observe_stream_bits(bytes.data(), bytes.size());
    return bits / static_cast<double>(ids.size());
}

MergeAttempt merge_workers_and_measure(
    const cypha::cyphalm::CyphaLMConfig& cfg, const std::vector<int>& full_ids,
    const std::vector<std::vector<int>>& shards,
    const std::vector<std::unique_ptr<cypha::cyphalm::CyphaLMModel>>& workers) {
    MergeAttempt r;
    if (workers.empty() || shards.empty() || workers.size() != shards.size()) {
        r.status = "error";
        r.detail = "worker/shard count mismatch";
        return r;
    }

    cypha::cyphalm::CyphaLMModel merged_tables(cfg);
    std::uint64_t merged_bytes = 0;
    for (std::size_t i = 0; i < workers.size(); ++i) {
        const std::uint64_t shard_bytes = static_cast<std::uint64_t>(shards[i].size());
        const hp::MergeStatus st = hp::merge_predictor_tables(
            merged_tables.hp_backend().predictor(), merged_bytes, workers[i]->hp_backend().predictor(),
            shard_bytes);
        if (st == hp::MergeStatus::EmptyInput) {
            r.status = "error";
            r.detail = "empty shard byte weight";
            return r;
        }
        merged_bytes += shard_bytes;
    }

    cypha::cyphalm::CyphaLMModel eval_model(cfg);
    eval_model.hp_backend().predictor().transfer_tables_from(merged_tables.hp_backend().predictor());
    r.merged_bpc = observe_bpc_keep_tables(eval_model, full_ids);
    r.ok = true;
    r.status = "weighted_table_merge";
    r.detail =
        "StateMap/Counter/mixer/APM weighted merge; context hash slots keep richer "
        "bit-history; eval uses transfer_tables_from onto a fresh predictor then "
        "full-corpus observe. In-sample (shard train bytes == eval corpus); not "
        "compress-equivalent to single_stream_bpc. Boundary replay not applied.";
    return r;
}

nlohmann::json shard_bpc_json(const std::string& label, double bpc, std::size_t nbytes) {
    return nlohmann::json{{"label", label},
                          {"bytes", nbytes},
                          {"metric", "observe_bit_serial_bpc"},
                          {"bpc", bpc}};
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus;
    std::string shard_dir;
    int max_bytes = 0;
    int n_shards = 2;
    int table_bits = 0;
    bool write_shards = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--corpus" && i + 1 < argc) {
            corpus = argv[++i];
        } else if (arg == "--max-bytes" && i + 1 < argc) {
            max_bytes = std::stoi(argv[++i]);
        } else if (arg == "--shards" && i + 1 < argc) {
            n_shards = std::stoi(argv[++i]);
        } else if (arg == "--table-bits" && i + 1 < argc) {
            table_bits = std::stoi(argv[++i]);
        } else if (arg == "--write-shards" && i + 1 < argc) {
            write_shards = true;
            shard_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::fprintf(stderr,
                         "usage: cyphalm_hp_shard_spike --corpus <path> "
                         "[--max-bytes N] [--shards 2] [--table-bits 22] [--write-shards <dir>]\n");
            return 0;
        }
    }

    if (corpus.empty()) {
        std::fprintf(stderr, "--corpus required\n");
        return 1;
    }

    const auto ids = load_bytes(corpus, max_bytes);
    if (ids.empty()) {
        std::fprintf(stderr, "empty corpus: %s\n", corpus.c_str());
        return 1;
    }

    const auto shards = split_shards(ids, n_shards);
    if (write_shards) {
        if (shard_dir.empty()) {
            shard_dir = "/tmp/cyphalm_hp_shards";
        }
        if (!write_shard_files(shards, shard_dir)) {
            std::fprintf(stderr, "failed to write shards to %s\n", shard_dir.c_str());
            return 1;
        }
    }

    const auto cfg = spike_cfg(table_bits);
    nlohmann::json out;
    out["corpus"] = corpus;
    out["hp_profile"] = "gate24";
    out["hp_table_bits"] = cfg.hp_table_bits;
    out["total_bytes"] = ids.size();
    out["n_shards"] = n_shards;
    if (write_shards) {
        out["shard_dir"] = shard_dir;
    }

    nlohmann::json shard_sizes = nlohmann::json::array();
    for (const auto& s : shards) {
        shard_sizes.push_back(s.size());
    }
    out["shard_byte_counts"] = shard_sizes;

    // (a)+(d) single-stream baseline
    {
        cypha::cyphalm::CyphaLMModel model(cfg);
        const double bpc = observe_bpc_ids(model, ids);
        out["single_stream_bpc"] = shard_bpc_json("single_stream", bpc, ids.size());
    }

    // Sequential shard pass — must match single-stream (validates split + determinism)
    {
        cypha::cyphalm::CyphaLMModel model(cfg);
        model.reset_context();
        double bits = 0.0;
        constexpr double kLog2 = 0.6931471805599453;
        for (const auto& shard : shards) {
            std::vector<std::uint8_t> bytes(shard.size());
            for (std::size_t i = 0; i < shard.size(); ++i) {
                bytes[i] = static_cast<std::uint8_t>(shard[i]);
            }
            bits += model.hp_backend().observe_stream_bits(bytes.data(), bytes.size());
        }
        const double bpc = bits / static_cast<double>(ids.size());
        out["sequential_shards_bpc"] = shard_bpc_json("sequential_shards", bpc, ids.size());
    }

    // (b) per-shard isolated adapt BPC (fresh predictor per shard)
    nlohmann::json isolated = nlohmann::json::array();
    for (std::size_t i = 0; i < shards.size(); ++i) {
        cypha::cyphalm::CyphaLMModel model(cfg);
        const double bpc = observe_bpc_ids(model, shards[i]);
        isolated.push_back(
            shard_bpc_json("shard" + std::to_string(i) + "_isolated_observe", bpc, shards[i].size()));
    }
    out["parallel_shard_isolated_bpc"] = isolated;

    // Train one predictor per shard (state retained for merge stub)
    std::vector<std::unique_ptr<cypha::cyphalm::CyphaLMModel>> workers;
    workers.reserve(shards.size());
    for (std::size_t i = 0; i < shards.size(); ++i) {
        auto model = std::make_unique<cypha::cyphalm::CyphaLMModel>(cfg);
        adapt_consume_ids(*model, shards[i]);
        workers.push_back(std::move(model));
    }

    // (c) merge trained shard tables + measure full-corpus BPC
    if (!workers.empty()) {
        const MergeAttempt m = merge_workers_and_measure(cfg, ids, shards, workers);
        out["merge_status"] = m.status;
        out["merge_ok"] = m.ok;
        out["merge_detail"] = m.detail;
        if (m.ok && std::isfinite(m.merged_bpc)) {
            out["merged_full_corpus_bpc"] =
                shard_bpc_json("merged_full_corpus", m.merged_bpc, ids.size());
            out["merged_bpc"] = m.merged_bpc;
        } else {
            out["merged_full_corpus_bpc"] = nullptr;
            out["merged_bpc"] = nullptr;
        }
    }

    const double single_bpc = out["single_stream_bpc"]["bpc"].get<double>();
    const double seq_bpc = out["sequential_shards_bpc"]["bpc"].get<double>();
    out["sequential_matches_single"] = std::abs(single_bpc - seq_bpc) < 1e-9;
    out["single_pass_bpc"] = out["single_stream_bpc"];
    if (out.contains("merged_bpc") && !out["merged_bpc"].is_null()) {
        const double merged_bpc = out["merged_bpc"].get<double>();
        out["merged_vs_single_delta_bpc"] = merged_bpc - single_bpc;
    }
    out["note"] =
        "merged_full_corpus_bpc uses weighted table merge (approximate); "
        "single_stream_bpc is exact single-pass observe. "
        "enwik gate24 ref 1.611729 @ mem 22 via scripts/measure_enwik_gate24.sh.";

    std::cout << out.dump(2) << '\n';
    return 0;
}
