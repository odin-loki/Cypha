/// Minimal shard + adapt spike for gate24 hp observe BPC.
/// Splits a byte corpus, adapts independent predictors per shard, merges tables.
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

struct MergeProfile {
    std::string name;
    hp::ShardMergeOptions merge_opts;
    hp::BoundaryReplayConfig replay_cfg;
};

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
    std::vector<std::uint8_t> bytes(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>(ids[i]);
    }
    const double bits = model.hp_backend().observe_stream_bits(bytes.data(), bytes.size());
    return bits / static_cast<double>(ids.size());
}

hp::Predictor& predictor(cypha::cyphalm::CyphaLMModel& model) {
    return model.hp_backend().predictor();
}

const hp::Predictor& predictor(const cypha::cyphalm::CyphaLMModel& model) {
    return model.hp_backend().predictor();
}

std::vector<std::uint8_t> to_bytes(const std::vector<int>& ids) {
    std::vector<std::uint8_t> bytes(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>(ids[i]);
    }
    return bytes;
}

std::vector<std::size_t> shard_boundaries_from_sizes(
    const std::vector<std::vector<int>>& shards) {
    std::vector<std::size_t> bounds;
    bounds.reserve(shards.size() + 1);
    std::size_t off = 0;
    for (const auto& shard : shards) {
        off += shard.size();
        bounds.push_back(off);
    }
    return bounds;
}

cypha::cyphalm::CyphaLMModel make_eval_from_merged_tables(
    const cypha::cyphalm::CyphaLMConfig& cfg, cypha::cyphalm::CyphaLMModel& merged_tables,
    const std::vector<int>& train_ids, const std::vector<std::vector<int>>& train_shards,
    const hp::BoundaryReplayConfig& replay_cfg) {
    cypha::cyphalm::CyphaLMModel eval_model(cfg);
    predictor(eval_model).transfer_tables_from(predictor(merged_tables));
    if (replay_cfg.replay_bytes > 0 && !train_ids.empty()) {
        const auto train_bytes = to_bytes(train_ids);
        const auto bounds = shard_boundaries_from_sizes(train_shards);
        hp::prepare_merged_predictor_for_holdout(predictor(eval_model), train_bytes.data(),
                                                 train_bytes.size(), bounds, replay_cfg);
    }
    return eval_model;
}

MergeProfile baseline_merge_profile(std::size_t replay_bytes) {
    MergeProfile p;
    p.name = "baseline";
    p.replay_cfg.replay_bytes = replay_bytes;
    p.replay_cfg.passes = 1;
    p.replay_cfg.distance_weighted = false;
    p.replay_cfg.bridge_finetune_bytes = 0;
    return p;
}

MergeProfile gated_only_profile(std::size_t replay_bytes) {
    MergeProfile p = baseline_merge_profile(replay_bytes);
    p.name = "gated_only";
    p.merge_opts = hp::holdout_merge_options();
    return p;
}

MergeProfile replay_distance_profile(std::size_t replay_bytes) {
    MergeProfile p = baseline_merge_profile(replay_bytes);
    p.name = "replay_distance";
    p.replay_cfg.distance_weighted = true;
    p.replay_cfg.max_passes_per_byte = 4;
    return p;
}

MergeProfile bridge_finetune_profile(std::size_t replay_bytes, std::size_t bridge_bytes) {
    MergeProfile p = baseline_merge_profile(replay_bytes);
    p.name = "bridge_finetune";
    p.replay_cfg.bridge_finetune_bytes = bridge_bytes;
    return p;
}

MergeProfile full_train_bridge_profile(std::size_t replay_bytes, std::size_t train_bytes) {
    MergeProfile p = baseline_merge_profile(replay_bytes);
    p.name = "full_train_bridge";
    p.replay_cfg.bridge_finetune_bytes = train_bytes;
    return p;
}

MergeProfile gated_full_train_bridge_profile(std::size_t replay_bytes, std::size_t train_bytes) {
    MergeProfile p = full_train_bridge_profile(replay_bytes, train_bytes);
    p.name = "gated_full_train_bridge";
    p.merge_opts = hp::holdout_merge_options();
    return p;
}

MergeProfile holdout_merge_profile(std::size_t replay_bytes, std::size_t train_bytes) {
    MergeProfile p = gated_full_train_bridge_profile(replay_bytes, train_bytes);
    p.name = "holdout";
    return p;
}

std::vector<MergeProfile> holdout_ablation_profiles(std::size_t replay_bytes,
                                                    std::size_t train_bytes) {
    return {baseline_merge_profile(replay_bytes), gated_only_profile(replay_bytes),
            replay_distance_profile(replay_bytes),
            bridge_finetune_profile(replay_bytes, std::min(replay_bytes, std::size_t{4096})),
            full_train_bridge_profile(replay_bytes, train_bytes),
            gated_full_train_bridge_profile(replay_bytes, train_bytes),
            holdout_merge_profile(replay_bytes, train_bytes)};
}

MergeAttempt merge_workers_and_measure(
    const cypha::cyphalm::CyphaLMConfig& cfg, const std::vector<int>& eval_ids,
    const std::vector<std::vector<int>>& shards,
    const std::vector<std::unique_ptr<cypha::cyphalm::CyphaLMModel>>& workers,
    const MergeProfile& profile, const std::vector<int>& train_ids_for_replay,
    const std::vector<std::vector<int>>& train_shards_for_replay, bool in_sample_eval) {
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
            predictor(merged_tables), merged_bytes, predictor(*workers[i]), shard_bytes,
            profile.merge_opts);
        if (st == hp::MergeStatus::EmptyInput) {
            r.status = "error";
            r.detail = "empty shard byte weight";
            return r;
        }
        merged_bytes += shard_bytes;
    }

    auto eval_model = make_eval_from_merged_tables(cfg, merged_tables, train_ids_for_replay,
                                                   train_shards_for_replay, profile.replay_cfg);
    r.merged_bpc = observe_bpc_keep_tables(eval_model, eval_ids);
    r.ok = std::isfinite(r.merged_bpc);
    r.status = profile.name + ":weighted_table_merge";
    if (profile.replay_cfg.replay_bytes > 0) {
        r.status += "+boundary_replay";
        if (profile.replay_cfg.distance_weighted) {
            r.status += "_distance_weighted";
        }
        if (profile.replay_cfg.bridge_finetune_bytes > 0) {
            r.status += "+bridge_finetune";
        }
    }
    if (profile.merge_opts.min_counter_n > 0 || profile.merge_opts.min_statemap_count > 0) {
        r.status += "+confidence_gated";
    }
    r.detail = in_sample_eval
                   ? "In-sample eval (shard train bytes overlap eval corpus). "
                   : "Holdout eval (eval bytes not in shard training). ";
    r.detail += "profile=" + profile.name;
    return r;
}

nlohmann::json merge_profile_json(const MergeProfile& profile) {
    return nlohmann::json{{"name", profile.name},
                          {"min_counter_n", profile.merge_opts.min_counter_n},
                          {"min_statemap_count", profile.merge_opts.min_statemap_count},
                          {"replay_bytes", profile.replay_cfg.replay_bytes},
                          {"replay_passes", profile.replay_cfg.passes},
                          {"distance_weighted", profile.replay_cfg.distance_weighted},
                          {"max_passes_per_byte", profile.replay_cfg.max_passes_per_byte},
                          {"bridge_finetune_bytes", profile.replay_cfg.bridge_finetune_bytes}};
}

nlohmann::json shard_bpc_json(const std::string& label, double bpc, std::size_t nbytes) {
    return nlohmann::json{{"label", label},
                          {"bytes", nbytes},
                          {"metric", "observe_bit_serial_bpc"},
                          {"bpc", bpc}};
}

int default_boundary_replay_bytes(std::size_t train_bytes, int cli_override) {
    return static_cast<int>(hp::default_boundary_replay_bytes(train_bytes, cli_override));
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus;
    std::string shard_dir;
    int max_bytes = 0;
    int n_shards = 2;
    int table_bits = 0;
    bool write_shards = false;
    double holdout_frac = 0.2;
    int boundary_replay_bytes = -1;  // -1 => auto when holdout enabled
    std::string merge_profile_mode = "both";  // baseline | holdout | both | ablation

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
        } else if (arg == "--holdout-frac" && i + 1 < argc) {
            holdout_frac = std::stod(argv[++i]);
        } else if (arg == "--boundary-replay-bytes" && i + 1 < argc) {
            boundary_replay_bytes = std::stoi(argv[++i]);
        } else if (arg == "--merge-profile" && i + 1 < argc) {
            merge_profile_mode = argv[++i];
        } else if (arg == "--write-shards" && i + 1 < argc) {
            write_shards = true;
            shard_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::fprintf(stderr,
                         "usage: cyphalm_hp_shard_spike --corpus <path> "
                         "[--max-bytes N] [--shards 2] [--table-bits 22] "
                         "[--holdout-frac 0.2] [--boundary-replay-bytes W] "
                         "[--merge-profile baseline|holdout|both|ablation] "
                         "[--write-shards <dir>]\n");
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

    const auto [train_ids, holdout_ids] = split_train_holdout(ids, holdout_frac);
    const bool holdout_enabled = !holdout_ids.empty();
    const int replay_bytes =
        holdout_enabled ? default_boundary_replay_bytes(train_ids.size(), boundary_replay_bytes) : 0;

    const auto train_shards = split_shards(train_ids, n_shards);
    const auto full_shards = split_shards(ids, n_shards);

    if (write_shards) {
        if (shard_dir.empty()) {
            shard_dir = "/tmp/cyphalm_hp_shards";
        }
        if (!write_shard_files(full_shards, shard_dir)) {
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
    out["holdout_frac"] = holdout_frac;
    out["train_bytes"] = train_ids.size();
    out["holdout_bytes"] = holdout_ids.size();
    out["boundary_replay_bytes"] = replay_bytes;
    out["merge_profile_mode"] = merge_profile_mode;
    if (write_shards) {
        out["shard_dir"] = shard_dir;
    }

    nlohmann::json shard_sizes = nlohmann::json::array();
    for (const auto& s : full_shards) {
        shard_sizes.push_back(s.size());
    }
    out["shard_byte_counts"] = shard_sizes;

    nlohmann::json train_shard_sizes = nlohmann::json::array();
    for (const auto& s : train_shards) {
        train_shard_sizes.push_back(s.size());
    }
    out["train_shard_byte_counts"] = train_shard_sizes;

    // (a)+(d) single-stream baseline (full corpus)
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
        for (const auto& shard : full_shards) {
            std::vector<std::uint8_t> bytes(shard.size());
            for (std::size_t i = 0; i < shard.size(); ++i) {
                bytes[i] = static_cast<std::uint8_t>(shard[i]);
            }
            bits += model.hp_backend().observe_stream_bits(bytes.data(), bytes.size());
        }
        const double bpc = bits / static_cast<double>(ids.size());
        out["sequential_shards_bpc"] = shard_bpc_json("sequential_shards", bpc, ids.size());
    }

    // (b) per-shard isolated adapt BPC (fresh predictor per shard, full corpus shards)
    nlohmann::json isolated = nlohmann::json::array();
    for (std::size_t i = 0; i < full_shards.size(); ++i) {
        cypha::cyphalm::CyphaLMModel model(cfg);
        const double bpc = observe_bpc_ids(model, full_shards[i]);
        isolated.push_back(
            shard_bpc_json("shard" + std::to_string(i) + "_isolated_observe", bpc, full_shards[i].size()));
    }
    out["parallel_shard_isolated_bpc"] = isolated;

    // Train one predictor per train shard (holdout bytes excluded from training)
    std::vector<std::unique_ptr<cypha::cyphalm::CyphaLMModel>> workers;
    workers.reserve(train_shards.size());
    for (std::size_t i = 0; i < train_shards.size(); ++i) {
        auto model = std::make_unique<cypha::cyphalm::CyphaLMModel>(cfg);
        adapt_consume_ids(*model, train_shards[i]);
        workers.push_back(std::move(model));
    }

    // (c) in-sample merge on full corpus (legacy metric; train includes eval bytes)
    std::vector<std::unique_ptr<cypha::cyphalm::CyphaLMModel>> full_workers;
    full_workers.reserve(full_shards.size());
    for (std::size_t i = 0; i < full_shards.size(); ++i) {
        auto model = std::make_unique<cypha::cyphalm::CyphaLMModel>(cfg);
        adapt_consume_ids(*model, full_shards[i]);
        full_workers.push_back(std::move(model));
    }

    if (!full_workers.empty()) {
        MergeProfile in_sample_profile;
        in_sample_profile.name = "in_sample";
        const MergeAttempt m = merge_workers_and_measure(cfg, ids, full_shards, full_workers,
                                                         in_sample_profile, ids, full_shards,
                                                         true);
        out["merge_status"] = m.status;
        out["merge_ok"] = m.ok;
        out["merge_detail"] = m.detail;
        if (m.ok) {
            out["merged_full_corpus_bpc"] =
                shard_bpc_json("merged_full_corpus_in_sample", m.merged_bpc, ids.size());
            out["merged_bpc"] = m.merged_bpc;
        } else {
            out["merged_full_corpus_bpc"] = nullptr;
            out["merged_bpc"] = nullptr;
        }
    }

    // (d) fair holdout eval: train shards on train prefix only, eval on holdout tail
    if (holdout_enabled && !workers.empty()) {
        const std::size_t replay_window = static_cast<std::size_t>(replay_bytes);
        const bool run_ablation = merge_profile_mode == "ablation";
        const bool run_baseline =
            merge_profile_mode == "baseline" || merge_profile_mode == "both" || run_ablation;
        const bool run_holdout =
            merge_profile_mode == "holdout" || merge_profile_mode == "both";

        cypha::cyphalm::CyphaLMModel single_train(cfg);
        adapt_consume_ids(single_train, train_ids);
        hp::BoundaryReplayConfig single_replay;
        single_replay.replay_bytes = replay_window;
        auto single_eval = make_eval_from_merged_tables(cfg, single_train, train_ids, train_shards,
                                                        single_replay);
        const double single_holdout_bpc = observe_bpc_keep_tables(single_eval, holdout_ids);
        out["single_stream_holdout_bpc"] =
            shard_bpc_json("single_stream_holdout", single_holdout_bpc, holdout_ids.size());

        auto record_holdout_profile = [&](const MergeProfile& profile, const std::string& key) {
            const MergeAttempt attempt = merge_workers_and_measure(
                cfg, holdout_ids, train_shards, workers, profile, train_ids, train_shards,
                false);
            nlohmann::json row;
            row["profile"] = merge_profile_json(profile);
            row["merged_holdout_bpc"] = attempt.merged_bpc;
            row["status"] = attempt.status;
            if (std::isfinite(attempt.merged_bpc) && std::isfinite(single_holdout_bpc)) {
                row["vs_single_delta_bpc"] = attempt.merged_bpc - single_holdout_bpc;
            }
            out[key] = row;
            return attempt;
        };

        if (run_baseline) {
            const MergeProfile baseline = baseline_merge_profile(replay_window);
            const MergeAttempt merged_holdout = record_holdout_profile(baseline, "merged_holdout");
            out["merged_holdout_bpc"] =
                shard_bpc_json("merged_holdout_baseline", merged_holdout.merged_bpc,
                               holdout_ids.size());
            out["merged_holdout_ok"] = merged_holdout.ok;
            out["merged_holdout_status"] = merged_holdout.status;
            out["merged_holdout_profile"] = merge_profile_json(baseline);
            if (std::isfinite(merged_holdout.merged_bpc) && std::isfinite(single_holdout_bpc)) {
                out["merged_holdout_vs_single_delta_bpc"] =
                    merged_holdout.merged_bpc - single_holdout_bpc;
            }
        }

        if (run_holdout) {
            const MergeProfile improved = holdout_merge_profile(replay_window, train_ids.size());
            const MergeAttempt merged_improved =
                record_holdout_profile(improved, "merged_holdout_improved");
            out["merged_holdout_improved_bpc"] =
                shard_bpc_json("merged_holdout_improved", merged_improved.merged_bpc,
                               holdout_ids.size());
            out["merged_holdout_improved_ok"] = merged_improved.ok;
            out["merged_holdout_improved_status"] = merged_improved.status;
            out["merged_holdout_improved_profile"] = merge_profile_json(improved);
            if (std::isfinite(merged_improved.merged_bpc) && std::isfinite(single_holdout_bpc)) {
                out["merged_holdout_improved_vs_single_delta_bpc"] =
                    merged_improved.merged_bpc - single_holdout_bpc;
            }
            if (run_baseline && out.contains("merged_holdout_vs_single_delta_bpc")) {
                const double baseline_delta =
                    out["merged_holdout_vs_single_delta_bpc"].get<double>();
                const double improved_delta =
                    out["merged_holdout_improved_vs_single_delta_bpc"].get<double>();
                out["holdout_improved_vs_baseline_delta_bpc"] = improved_delta - baseline_delta;
            }
        }

        if (run_ablation) {
            nlohmann::json ablation = nlohmann::json::array();
            double best_delta = std::numeric_limits<double>::infinity();
            std::string best_name;
            for (const MergeProfile& profile :
                 holdout_ablation_profiles(replay_window, train_ids.size())) {
                const MergeAttempt attempt = merge_workers_and_measure(
                    cfg, holdout_ids, train_shards, workers, profile, train_ids, train_shards,
                    false);
                nlohmann::json row;
                row["name"] = profile.name;
                row["profile"] = merge_profile_json(profile);
                row["merged_holdout_bpc"] = attempt.merged_bpc;
                if (std::isfinite(attempt.merged_bpc) && std::isfinite(single_holdout_bpc)) {
                    const double delta = attempt.merged_bpc - single_holdout_bpc;
                    row["vs_single_delta_bpc"] = delta;
                    if (delta < best_delta) {
                        best_delta = delta;
                        best_name = profile.name;
                    }
                }
                ablation.push_back(row);
            }
            out["holdout_ablation"] = ablation;
            out["holdout_ablation_best_profile"] = best_name;
            out["holdout_ablation_best_vs_single_delta_bpc"] = best_delta;
        }

        {
            cypha::cyphalm::CyphaLMModel cold(cfg);
            const double cold_bpc = observe_bpc_ids(cold, holdout_ids);
            out["cold_holdout_bpc"] = shard_bpc_json("cold_holdout", cold_bpc, holdout_ids.size());
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
        "merged_full_corpus_bpc is in-sample (optimistic). Prefer merged_holdout_bpc vs "
        "single_stream_holdout_bpc for fair merge comparison. "
        "enwik gate24 ref 1.611729 @ mem 22 via scripts/measure_enwik_gate24.sh.";

    std::cout << out.dump(2) << '\n';
    return 0;
}
