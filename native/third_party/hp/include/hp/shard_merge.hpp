#pragma once
//
// hp/shard_merge.hpp — gate24 train-scale shard table merge.
//
// Merges additive / pooled statistics (Counter, StateMap, mixer weights, APM)
// from independently trained Predictors. Path-dependent state (byte rings,
// match positions, wiki/wordstream cursors) is NOT merged; use reset_stream_state
// or transfer_tables_from onto a fresh Predictor before full-corpus BPC eval.

#include "hp/models.hpp"
#include "hp/predictor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace hp {

enum class MergeStatus {
    Ok,
    ConfigMismatch,
    EmptyInput,
};

/// Strategy C boundary replay + optional bridge fine-tune before holdout eval.
struct BoundaryReplayConfig {
    std::size_t replay_bytes = 0;
    int passes = 1;
    bool distance_weighted = false;
    int max_passes_per_byte = 4;
    /// Sequential adapt on the last ``bridge_finetune_bytes`` of train after merge.
    std::size_t bridge_finetune_bytes = 0;
};

inline void merge_counter_arrays(Counter* dst, const Counter* src, std::size_t n,
                                 std::uint64_t src_weight, std::uint64_t dst_weight,
                                 std::uint16_t min_counter_n = 0) {
    for (std::size_t i = 0; i < n; ++i) {
        merge_counter_cell(dst[i], src[i], src_weight, dst_weight, min_counter_n);
    }
}

inline void copy_counter_arrays(Counter* dst, const Counter* src, std::size_t n) {
    std::memcpy(dst, src, n * sizeof(Counter));
}

inline void Predictor::merge_shard_tables(const Predictor& src, std::uint64_t src_bytes,
                                          std::uint64_t dst_bytes,
                                          const ShardMergeOptions& opts) {
    if (src_bytes == 0) {
        return;
    }
    if (dst_bytes == 0) {
        transfer_tables_from(src);
        return;
    }

    merge_counter_arrays(bias_.data(), src.bias_.data(), bias_.size(), src_bytes, dst_bytes,
                         opts.min_counter_n);

    for (int i = 0; i < n_ctx_chain_; ++i) {
        ctx_chain_[i]->merge_tables_from(*src.ctx_chain_[i], src_bytes, dst_bytes,
                                         opts.min_statemap_count);
    }

    for (int i = 0; i < kMatchModels; ++i) {
        match_[i].merge_counters_from(src.match_[i], src_bytes, dst_bytes, opts.min_counter_n);
    }
#if HP_SPARSE_UTF8
    smatch_.merge_counters_from(src.smatch_, src_bytes, dst_bytes, opts.min_counter_n);
#endif
#if HP_SKIPK_MOD
    skipk_.merge_counters_from(src.skipk_, src_bytes, dst_bytes, opts.min_counter_n);
#endif
#if HP_SKIP3_MOD
    skip3_.merge_counters_from(src.skip3_, src_bytes, dst_bytes, opts.min_counter_n);
#endif
#if HP_SKIP4_MOD
    skip4_.merge_counters_from(src.skip4_, src_bytes, dst_bytes, opts.min_counter_n);
#endif
#if HP_SKIP5_MOD
    skip5_.merge_counters_from(src.skip5_, src_bytes, dst_bytes, opts.min_counter_n);
#endif

    hebb_.merge_tables_from(src.hebb_, src_bytes, dst_bytes, opts.min_statemap_count);
    pool_.merge_tables_from(src.pool_, src_bytes, dst_bytes, opts.min_statemap_count);
    mixer_.merge_from(src.mixer_, src_bytes, dst_bytes);
    apm_c0_.merge_from(src.apm_c0_, src_bytes, dst_bytes);
    apm_lex_.merge_from(src.apm_lex_, src_bytes, dst_bytes);
    apm_gria_.merge_from(src.apm_gria_, src_bytes, dst_bytes);
    hedge_.merge_from(src.hedge_, src_bytes, dst_bytes);
}

inline MergeStatus merge_predictor_tables(Predictor& dst, std::uint64_t dst_bytes,
                                          const Predictor& src, std::uint64_t src_bytes,
                                          const ShardMergeOptions& opts = {}) {
    if (src_bytes == 0) {
        return MergeStatus::EmptyInput;
    }
    if (dst_bytes == 0) {
        dst.transfer_tables_from(src);
        return MergeStatus::Ok;
    }
    dst.merge_shard_tables(src, src_bytes, dst_bytes, opts);
    return MergeStatus::Ok;
}

inline MergeStatus merge_predictor_tables(Predictor& dst, const Predictor& a,
                                          std::uint64_t a_bytes, const Predictor& b,
                                          std::uint64_t b_bytes,
                                          const ShardMergeOptions& opts = {}) {
    if (a_bytes == 0 && b_bytes == 0) {
        return MergeStatus::EmptyInput;
    }
    if (a_bytes == 0) {
        dst.transfer_tables_from(b);
        return MergeStatus::Ok;
    }
    dst.transfer_tables_from(a);
    if (b_bytes == 0) {
        return MergeStatus::Ok;
    }
    return merge_predictor_tables(dst, a_bytes, b, b_bytes, opts);
}

inline void Predictor::merge_shard_tables(const Predictor& src, std::uint64_t src_bytes,
                                          std::uint64_t dst_bytes) {
    merge_shard_tables(src, src_bytes, dst_bytes, {});
}

inline void Predictor::transfer_tables_from(const Predictor& src) {
    copy_counter_arrays(bias_.data(), src.bias_.data(), bias_.size());

    for (int i = 0; i < n_ctx_chain_; ++i) {
        ctx_chain_[i]->copy_tables_from(*src.ctx_chain_[i]);
    }

    for (int i = 0; i < kMatchModels; ++i) {
        match_[i].copy_counters_from(src.match_[i]);
    }
#if HP_SPARSE_UTF8
    smatch_.copy_counters_from(src.smatch_);
#endif
#if HP_SKIPK_MOD
    skipk_.copy_counters_from(src.skipk_);
#endif
#if HP_SKIP3_MOD
    skip3_.copy_counters_from(src.skip3_);
#endif
#if HP_SKIP4_MOD
    skip4_.copy_counters_from(src.skip4_);
#endif
#if HP_SKIP5_MOD
    skip5_.copy_counters_from(src.skip5_);
#endif

    hebb_.copy_tables_from(src.hebb_);
    pool_.copy_tables_from(src.pool_);
    mixer_.copy_from(src.mixer_);
    apm_c0_.copy_from(src.apm_c0_);
    apm_lex_.copy_from(src.apm_lex_);
    apm_gria_.copy_from(src.apm_gria_);
    hedge_.copy_from(src.hedge_);
}

/// Consume one byte on ``pred`` (MSB-first, train path).
inline void consume_byte_on(Predictor& pred, std::uint8_t byte) {
    for (int i = 7; i >= 0; --i) {
        const int bit = (static_cast<int>(byte) >> i) & 1;
        (void)pred.predict();
        pred.update(bit);
    }
}

/// Strategy C: byte offset to start replaying ``replay_bytes`` before shard boundary.
inline std::size_t boundary_replay_begin(std::size_t boundary_byte, std::size_t replay_bytes) {
    if (replay_bytes == 0 || boundary_byte == 0) {
        return boundary_byte;
    }
    return boundary_byte > replay_bytes ? boundary_byte - replay_bytes : 0;
}

inline int boundary_replay_passes_for_byte(std::size_t byte_index, std::size_t begin,
                                           std::size_t boundary, int max_passes_per_byte) {
    if (boundary <= begin + 1) {
        return 1;
    }
    const std::size_t window = boundary - begin;
    const std::size_t dist_to_boundary = boundary - byte_index - 1;
    const int max_p = std::max(1, max_passes_per_byte);
    return 1 + (max_p - 1) * static_cast<int>((window - dist_to_boundary) / window);
}

/// Replay ``[begin, boundary)`` with optional multi-pass / distance weighting.
inline void boundary_replay_range(Predictor& pred, const std::uint8_t* bytes, std::size_t nbytes,
                                  std::size_t begin, std::size_t boundary,
                                  const BoundaryReplayConfig& cfg) {
    if (cfg.replay_bytes == 0 || boundary == 0 || bytes == nullptr) {
        return;
    }
    const std::size_t replay_begin = boundary_replay_begin(boundary, cfg.replay_bytes);
    begin = std::max(begin, replay_begin);
    if (begin >= boundary || boundary > nbytes) {
        return;
    }
    if (cfg.distance_weighted) {
        for (std::size_t i = begin; i < boundary; ++i) {
            const int n = boundary_replay_passes_for_byte(i, begin, boundary,
                                                          cfg.max_passes_per_byte);
            for (int p = 0; p < n; ++p) {
                consume_byte_on(pred, bytes[i]);
            }
        }
        return;
    }
    const int base_passes = std::max(1, cfg.passes);
    for (int pass = 0; pass < base_passes; ++pass) {
        for (std::size_t i = begin; i < boundary; ++i) {
            consume_byte_on(pred, bytes[i]);
        }
    }
}

/// Replay windows ending at each shard join in ``shard_boundaries``.
inline void boundary_replay_shard_joins(Predictor& pred, const std::uint8_t* train_bytes,
                                        std::size_t train_nbytes,
                                        const std::vector<std::size_t>& shard_boundaries,
                                        const BoundaryReplayConfig& cfg) {
    if (cfg.replay_bytes == 0 || train_bytes == nullptr || shard_boundaries.size() < 2) {
        return;
    }
    for (std::size_t i = 0; i + 1 < shard_boundaries.size(); ++i) {
        const std::size_t boundary = shard_boundaries[i];
        if (boundary == 0 || boundary > train_nbytes) {
            continue;
        }
        boundary_replay_range(pred, train_bytes, train_nbytes, 0, boundary, cfg);
    }
}

/// Replay train tail + optional bridge fine-tune before holdout observe.
inline void prepare_merged_predictor_for_holdout(Predictor& pred, const std::uint8_t* train_bytes,
                                                 std::size_t train_nbytes,
                                                 const std::vector<std::size_t>& shard_boundaries,
                                                 const BoundaryReplayConfig& cfg) {
    if (train_bytes == nullptr || train_nbytes == 0) {
        return;
    }
    boundary_replay_shard_joins(pred, train_bytes, train_nbytes, shard_boundaries, cfg);
    boundary_replay_range(pred, train_bytes, train_nbytes, 0, train_nbytes, cfg);
    if (cfg.bridge_finetune_bytes > 0) {
        const std::size_t begin =
            boundary_replay_begin(train_nbytes, cfg.bridge_finetune_bytes);
        for (std::size_t i = begin; i < train_nbytes; ++i) {
            consume_byte_on(pred, train_bytes[i]);
        }
    }
}

/// Default auto replay window for holdout eval (min 256, max 8192, ~10% of train).
inline std::size_t default_boundary_replay_bytes(std::size_t train_bytes, int cli_override) {
    if (cli_override > 0) {
        return static_cast<std::size_t>(cli_override);
    }
    if (train_bytes < 512) {
        return 0;
    }
    const std::size_t tenth = train_bytes / 10;
    return std::min<std::size_t>(8192, std::max<std::size_t>(256, tenth));
}

/// Recommended holdout profile from PR #15 follow-up experiments.
inline ShardMergeOptions holdout_merge_options() {
    ShardMergeOptions o;
    o.min_counter_n = 8;
    o.min_statemap_count = 4;
    return o;
}

/// Recommended post-merge prep: boundary replay + sequential adapt on full train prefix.
inline BoundaryReplayConfig holdout_boundary_replay_config(std::size_t replay_bytes,
                                                           std::size_t train_bytes) {
    BoundaryReplayConfig cfg;
    cfg.replay_bytes = replay_bytes;
    cfg.passes = 1;
    cfg.distance_weighted = false;
    cfg.bridge_finetune_bytes = train_bytes;
    return cfg;
}

/// Merge shard tables in shard order. Boundary replay (consume prefix windows) is
/// performed by the caller between merges — see ``boundary_replay_begin``.
inline MergeStatus merge_predictor_tables_sequential(
    Predictor& dst, const std::vector<const Predictor*>& shards,
    const std::vector<std::uint64_t>& shard_bytes, const ShardMergeOptions& opts = {}) {
    if (shards.size() != shard_bytes.size() || shards.empty()) {
        return MergeStatus::EmptyInput;
    }
    std::uint64_t merged_bytes = 0;
    for (std::size_t i = 0; i < shards.size(); ++i) {
        if (shards[i] == nullptr || shard_bytes[i] == 0) {
            return MergeStatus::EmptyInput;
        }
        const MergeStatus st =
            merge_predictor_tables(dst, merged_bytes, *shards[i], shard_bytes[i], opts);
        if (st != MergeStatus::Ok) {
            return st;
        }
        merged_bytes += shard_bytes[i];
    }
    return MergeStatus::Ok;
}

inline void Predictor::reset_stream_state() {
    byte_ring_.reset();
    hist_ = 0;
    word_hash_ = 0;
    letter_hash_ = 0;
    hist2_ = 0;
    std::memset(line_buf_, 0, sizeof(line_buf_));
    cur_line_idx_ = 0;
    col_pos_ = 0;
    tag_depth_ = 0;
    in_tag_ = 0;
    tag_name_ = 0;
    prev_word_ = 0;
    word_hash_prev_ = 0;
    std::memset(word_ring_, 0, sizeof(word_ring_));
    c0_ = 1;
    bitpos_ = 0;
    pr_final_ = 2048;
    std::memset(exp_p_, 0, sizeof(exp_p_));
    n_exp_ = 0;
    mixed_p_ = 2048;
    last_mlen_ = 0;
    sparse_ = 0;
#if HP_GATE_BREAK
    break_age_ = 0;
#endif
#if HP_PRONOUN_MOD
    std::memset(pw_, 0, sizeof(pw_));
    pw_n_ = 0;
    pronoun_ = 0;
#endif
#if HP_UTF8_IDLE || HP_GATE_UTF8
    utf8left_ = 0;
#endif
    rebind_internal_pointers_();
    set_byte_contexts();
}

}  // namespace hp
