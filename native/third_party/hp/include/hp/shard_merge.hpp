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

namespace hp {

enum class MergeStatus {
    Ok,
    ConfigMismatch,
    EmptyInput,
};

inline void merge_counter_arrays(Counter* dst, const Counter* src, std::size_t n,
                                 std::uint64_t src_weight, std::uint64_t dst_weight) {
    for (std::size_t i = 0; i < n; ++i) {
        Counter& d = dst[i];
        const Counter& s = src[i];
        if (s.n == 0) {
            continue;
        }
        if (d.n == 0) {
            d = s;
            continue;
        }
        const std::uint64_t total = dst_weight + src_weight;
        if (total == 0) {
            continue;
        }
        const int merged_p = static_cast<int>(
            (static_cast<std::uint64_t>(d.p) * dst_weight +
             static_cast<std::uint64_t>(s.p) * src_weight) /
            total);
        const int merged_n = static_cast<int>(
            std::min<std::uint64_t>(65535u, (static_cast<std::uint64_t>(d.n) * dst_weight +
                                             static_cast<std::uint64_t>(s.n) * src_weight) /
                                                total));
        d.p = static_cast<std::uint16_t>(merged_p);
        d.n = static_cast<std::uint16_t>(merged_n);
    }
}

inline void copy_counter_arrays(Counter* dst, const Counter* src, std::size_t n) {
    std::memcpy(dst, src, n * sizeof(Counter));
}

inline MergeStatus merge_predictor_tables(Predictor& dst, std::uint64_t dst_bytes,
                                          const Predictor& src, std::uint64_t src_bytes) {
    if (src_bytes == 0) {
        return MergeStatus::EmptyInput;
    }
    if (dst_bytes == 0) {
        dst.transfer_tables_from(src);
        return MergeStatus::Ok;
    }
    dst.merge_shard_tables(src, src_bytes, dst_bytes);
    return MergeStatus::Ok;
}

inline MergeStatus merge_predictor_tables(Predictor& dst, const Predictor& a,
                                          std::uint64_t a_bytes, const Predictor& b,
                                          std::uint64_t b_bytes) {
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
    return merge_predictor_tables(dst, a_bytes, b, b_bytes);
}

inline void Predictor::merge_shard_tables(const Predictor& src, std::uint64_t src_bytes,
                                          std::uint64_t dst_bytes) {
    if (src_bytes == 0) {
        return;
    }
    if (dst_bytes == 0) {
        transfer_tables_from(src);
        return;
    }

    merge_counter_arrays(bias_.data(), src.bias_.data(), bias_.size(), src_bytes, dst_bytes);

    for (int i = 0; i < n_ctx_chain_; ++i) {
        ctx_chain_[i]->merge_tables_from(*src.ctx_chain_[i], src_bytes, dst_bytes);
    }

    for (int i = 0; i < kMatchModels; ++i) {
        match_[i].merge_counters_from(src.match_[i], src_bytes, dst_bytes);
    }
#if HP_SPARSE_UTF8
    smatch_.merge_counters_from(src.smatch_, src_bytes, dst_bytes);
#endif
#if HP_SKIPK_MOD
    skipk_.merge_counters_from(src.skipk_, src_bytes, dst_bytes);
#endif
#if HP_SKIP3_MOD
    skip3_.merge_counters_from(src.skip3_, src_bytes, dst_bytes);
#endif
#if HP_SKIP4_MOD
    skip4_.merge_counters_from(src.skip4_, src_bytes, dst_bytes);
#endif
#if HP_SKIP5_MOD
    skip5_.merge_counters_from(src.skip5_, src_bytes, dst_bytes);
#endif

    hebb_.merge_tables_from(src.hebb_, src_bytes, dst_bytes);
    pool_.merge_tables_from(src.pool_, src_bytes, dst_bytes);
    mixer_.merge_from(src.mixer_, src_bytes, dst_bytes);
    apm_c0_.merge_from(src.apm_c0_, src_bytes, dst_bytes);
    apm_lex_.merge_from(src.apm_lex_, src_bytes, dst_bytes);
    apm_gria_.merge_from(src.apm_gria_, src_bytes, dst_bytes);
    hedge_.merge_from(src.hedge_, src_bytes, dst_bytes);
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

/// Strategy C: byte offset to start replaying ``replay_bytes`` before shard boundary.
inline std::size_t boundary_replay_begin(std::size_t boundary_byte, std::size_t replay_bytes) {
    if (replay_bytes == 0 || boundary_byte == 0) {
        return boundary_byte;
    }
    return boundary_byte > replay_bytes ? boundary_byte - replay_bytes : 0;
}

/// Merge shard tables in shard order. Boundary replay (consume prefix windows) is
/// performed by the caller between merges — see ``boundary_replay_begin``.
inline MergeStatus merge_predictor_tables_sequential(
    Predictor& dst, const std::vector<const Predictor*>& shards,
    const std::vector<std::uint64_t>& shard_bytes) {
    if (shards.size() != shard_bytes.size() || shards.empty()) {
        return MergeStatus::EmptyInput;
    }
    std::uint64_t merged_bytes = 0;
    for (std::size_t i = 0; i < shards.size(); ++i) {
        if (shards[i] == nullptr || shard_bytes[i] == 0) {
            return MergeStatus::EmptyInput;
        }
        const MergeStatus st =
            merge_predictor_tables(dst, merged_bytes, *shards[i], shard_bytes[i]);
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
