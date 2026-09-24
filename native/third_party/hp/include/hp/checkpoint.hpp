#pragma once
// hp/checkpoint.hpp — binary Predictor checkpoint (Cypha train-once / serve-many).

#include <cstdio>
#include <cstdlib>
#include <istream>
#include <ostream>
#include <type_traits>

#include "hp/blob_io.hpp"

#define HP_CKPT_WRITE_CM(os, cm) (cm).checkpoint_write(os)
#define HP_CKPT_READ_CM(is, cm) (cm).checkpoint_read(is)

template <typename T>
inline void checkpoint_write_trivial(std::ostream& os, const T& o) {
    static_assert(std::is_trivially_copyable_v<T>);
    blob::write_trivial_object(os, o);
}

template <typename T>
inline void checkpoint_read_trivial(std::istream& is, T& o) {
    static_assert(std::is_trivially_copyable_v<T>);
    blob::read_trivial_object(is, o);
}

inline void Predictor::write_checkpoint(std::ostream& os) const {
    const char magic[4] = {'H', 'P', 'C', 'P'};
    os.write(magic, 4);
    // v2 appends the sentence memory (stream state v1 left out); v3 packs
    // context-model slots to 16 bits (state + 6-bit checksum); v4 adds the
    // mixer's layer-1 scale and skip; v5 appends the optional upstream context
    // models (Config::extra_cms != 0 only: extra_cms = 0 still writes v4).
    const std::uint32_t ver = xcms_ ? 5 : 4;
    blob::write_pod(os, ver);
    // HP_CKPT_SIZES=1: report each component's size on stderr.
    static const bool sizes = std::getenv("HP_CKPT_SIZES") != nullptr;
    std::streamoff mark_at = os.tellp();
    auto mark = [&](const char* name) {
        if (!sizes) return;
        const std::streamoff now = os.tellp();
        std::fprintf(stderr, "ckpt %-14s %12lld\n", name, static_cast<long long>(now - mark_at));
        mark_at = now;
    };
    byte_ring_.checkpoint_write(os);
    mark("byte_ring_");
    HP_CKPT_WRITE_CM(os, o1_);
    mark("o1_");
    HP_CKPT_WRITE_CM(os, o2_);
    mark("o2_");
    HP_CKPT_WRITE_CM(os, o3_);
    mark("o3_");
    HP_CKPT_WRITE_CM(os, o4_);
    mark("o4_");
    HP_CKPT_WRITE_CM(os, o6_);
    mark("o6_");
    HP_CKPT_WRITE_CM(os, o6b_);
    mark("o6b_");
    HP_CKPT_WRITE_CM(os, word_);
    mark("word_");
    HP_CKPT_WRITE_CM(os, col_);
    mark("col_");
    HP_CKPT_WRITE_CM(os, tag_);
    mark("tag_");
    HP_CKPT_WRITE_CM(os, wbi_);
    mark("wbi_");
    HP_CKPT_WRITE_CM(os, sp13_);
    mark("sp13_");
    HP_CKPT_WRITE_CM(os, sp24_);
    mark("sp24_");
    HP_CKPT_WRITE_CM(os, wstr_sp_);
    mark("wstr_sp_");
    HP_CKPT_WRITE_CM(os, brk_);
    mark("brk_");
    HP_CKPT_WRITE_CM(os, link_);
    mark("link_");
    HP_CKPT_WRITE_CM(os, num_);
    mark("num_");
    HP_CKPT_WRITE_CM(os, sen_);
    mark("sen_");
    HP_CKPT_WRITE_CM(os, sentst_);
    mark("sentst_");
    HP_CKPT_WRITE_CM(os, sentmem_cm_);
    mark("sentmem_cm_");
    HP_CKPT_WRITE_CM(os, sengrp_);
    mark("sengrp_");
    HP_CKPT_WRITE_CM(os, nestmod_);
    mark("nestmod_");
    HP_CKPT_WRITE_CM(os, paramod_);
    mark("paramod_");
    HP_CKPT_WRITE_CM(os, linemod_);
    mark("linemod_");
    HP_CKPT_WRITE_CM(os, statemod_);
    mark("statemod_");
    HP_CKPT_WRITE_CM(os, tplmod_);
    mark("tplmod_");
    HP_CKPT_WRITE_CM(os, infokeymod_);
    mark("infokeymod_");
    HP_CKPT_WRITE_CM(os, linkpipemod_);
    mark("linkpipemod_");
    HP_CKPT_WRITE_CM(os, catmod_);
    mark("catmod_");
    HP_CKPT_WRITE_CM(os, headingmod_);
    mark("headingmod_");
    HP_CKPT_WRITE_CM(os, titlemod_);
    mark("titlemod_");
    HP_CKPT_WRITE_CM(os, sectitlemod_);
    mark("sectitlemod_");
    HP_CKPT_WRITE_CM(os, wikistackmod_);
    mark("wikistackmod_");
    HP_CKPT_WRITE_CM(os, capmaskmod_);
    mark("capmaskmod_");
    HP_CKPT_WRITE_CM(os, uppergapmod_);
    mark("uppergapmod_");
    HP_CKPT_WRITE_CM(os, wordlenmod_);
    mark("wordlenmod_");
    for (int i = 0; i < kMatchModels; ++i) match_[i].checkpoint_write(os);
    mark("match_[]");
    smatch_.checkpoint_write(os);
    mark("smatch_");
    skipk_.checkpoint_write(os);
    mark("skipk_");
    skip3_.checkpoint_write(os);
    mark("skip3_");
    skip4_.checkpoint_write(os);
    mark("skip4_");
    lzp_.checkpoint_write(os);
    mark("lzp_");
    dmc_.checkpoint_write(os);
    mark("dmc_");
    for (int i = 0; i < static_cast<int>(sizeof(wmatch_) / sizeof(wmatch_[0])); ++i) {
        wmatch_[i].checkpoint_write(os);
    mark("match_[]");
    }
    hebb_.checkpoint_write(os);
    mark("hebb_");
    pool_.checkpoint_write(os);
    mark("pool_");
    mixer_.checkpoint_write(os);
    mark("mixer_");
    apm_c0_.checkpoint_write(os);
    mark("apm_c0_");
    apm_lex_.checkpoint_write(os);
    mark("apm_lex_");
    apm_gria_.checkpoint_write(os);
    mark("apm_gria_");
    hedge_.checkpoint_write(os);
    mark("hedge_");
    blob::write_array(os, bias_);
    gria_.checkpoint_write(os);
    mark("gria_");
    checkpoint_write_trivial(os, wiki_);
    checkpoint_write_trivial(os, streams_);
    checkpoint_write_trivial(os, brackets_);
    checkpoint_write_trivial(os, cache_);
    checkpoint_write_trivial(os, numbers_);
    blob::write_pod(os, hist_);
    blob::write_pod(os, word_hash_);
    blob::write_pod(os, letter_hash_);
    blob::write_pod(os, hist2_);
    os.write(reinterpret_cast<const char*>(line_buf_), sizeof(line_buf_));
    blob::write_pod(os, cur_line_idx_);
    blob::write_pod(os, col_pos_);
    blob::write_pod(os, tag_depth_);
    blob::write_pod(os, in_tag_);
    blob::write_pod(os, tag_name_);
    blob::write_pod(os, prev_word_);
    blob::write_pod(os, word_hash_prev_);
    os.write(reinterpret_cast<const char*>(word_ring_), sizeof(word_ring_));
    blob::write_pod(os, c0_);
    blob::write_pod(os, bitpos_);
    blob::write_pod(os, pr_final_);
    os.write(reinterpret_cast<const char*>(exp_p_), static_cast<std::streamsize>(kExpPBase * sizeof(exp_p_[0])));
    blob::write_pod(os, n_exp_);
    blob::write_pod(os, mixed_p_);
    blob::write_pod(os, last_mlen_);
    blob::write_pod(os, sparse_);
    checkpoint_write_trivial(os, sentmem_);
    mark("rest");
    if (ver >= 5) {
        blob::write_pod(os, xcms_);
        for (int i = kCtxModels; i < n_ctx_chain_; ++i) HP_CKPT_WRITE_CM(os, *ctx_chain_[i]);
        os.write(reinterpret_cast<const char*>(exp_p_ + kExpPBase),
                 static_cast<std::streamsize>(sizeof(exp_p_) - kExpPBase * sizeof(exp_p_[0])));
        checkpoint_write_trivial(os, wx_);
        mark("extra_cms");
    }
}

inline void Predictor::read_checkpoint(std::istream& is) {
    char magic[4] = {};
    is.read(magic, 4);
    if (magic[0] != 'H' || magic[1] != 'P' || magic[2] != 'C' || magic[3] != 'P') {
        return;
    }
    std::uint32_t ver = 0;
    blob::read_pod(is, ver);
    if (ver < 1 || ver > 5) return;
    // The optional context models shape the mixer and the chain: the file
    // must have been written with this predictor's Config::extra_cms.
    if ((ver >= 5) != (xcms_ != 0)) {
        is.setstate(std::ios::failbit);
        return;
    }
    g_hp_ckpt_read_version = static_cast<int>(ver);
    byte_ring_.checkpoint_read(is);
    HP_CKPT_READ_CM(is, o1_);
    HP_CKPT_READ_CM(is, o2_);
    HP_CKPT_READ_CM(is, o3_);
    HP_CKPT_READ_CM(is, o4_);
    HP_CKPT_READ_CM(is, o6_);
    HP_CKPT_READ_CM(is, o6b_);
    HP_CKPT_READ_CM(is, word_);
    HP_CKPT_READ_CM(is, col_);
    HP_CKPT_READ_CM(is, tag_);
    HP_CKPT_READ_CM(is, wbi_);
    HP_CKPT_READ_CM(is, sp13_);
    HP_CKPT_READ_CM(is, sp24_);
    HP_CKPT_READ_CM(is, wstr_sp_);
    HP_CKPT_READ_CM(is, brk_);
    HP_CKPT_READ_CM(is, link_);
    HP_CKPT_READ_CM(is, num_);
    HP_CKPT_READ_CM(is, sen_);
    HP_CKPT_READ_CM(is, sentst_);
    HP_CKPT_READ_CM(is, sentmem_cm_);
    HP_CKPT_READ_CM(is, sengrp_);
    HP_CKPT_READ_CM(is, nestmod_);
    HP_CKPT_READ_CM(is, paramod_);
    HP_CKPT_READ_CM(is, linemod_);
    HP_CKPT_READ_CM(is, statemod_);
    HP_CKPT_READ_CM(is, tplmod_);
    HP_CKPT_READ_CM(is, infokeymod_);
    HP_CKPT_READ_CM(is, linkpipemod_);
    HP_CKPT_READ_CM(is, catmod_);
    HP_CKPT_READ_CM(is, headingmod_);
    HP_CKPT_READ_CM(is, titlemod_);
    HP_CKPT_READ_CM(is, sectitlemod_);
    HP_CKPT_READ_CM(is, wikistackmod_);
    HP_CKPT_READ_CM(is, capmaskmod_);
    HP_CKPT_READ_CM(is, uppergapmod_);
    HP_CKPT_READ_CM(is, wordlenmod_);
    for (int i = 0; i < kMatchModels; ++i) match_[i].checkpoint_read(is);
    smatch_.checkpoint_read(is);
    skipk_.checkpoint_read(is);
    skip3_.checkpoint_read(is);
    skip4_.checkpoint_read(is);
    lzp_.checkpoint_read(is);
    dmc_.checkpoint_read(is);
    for (int i = 0; i < static_cast<int>(sizeof(wmatch_) / sizeof(wmatch_[0])); ++i) {
        wmatch_[i].checkpoint_read(is);
    }
    hebb_.checkpoint_read(is);
    pool_.checkpoint_read(is);
    mixer_.checkpoint_read(is);
    apm_c0_.checkpoint_read(is);
    apm_lex_.checkpoint_read(is);
    apm_gria_.checkpoint_read(is);
    hedge_.checkpoint_read(is);
    blob::read_array(is, bias_);
    gria_.checkpoint_read(is);
    checkpoint_read_trivial(is, wiki_);
    checkpoint_read_trivial(is, streams_);
    checkpoint_read_trivial(is, brackets_);
    checkpoint_read_trivial(is, cache_);
    checkpoint_read_trivial(is, numbers_);
    blob::read_pod(is, hist_);
    blob::read_pod(is, word_hash_);
    blob::read_pod(is, letter_hash_);
    blob::read_pod(is, hist2_);
    is.read(reinterpret_cast<char*>(line_buf_), sizeof(line_buf_));
    blob::read_pod(is, cur_line_idx_);
    blob::read_pod(is, col_pos_);
    blob::read_pod(is, tag_depth_);
    blob::read_pod(is, in_tag_);
    blob::read_pod(is, tag_name_);
    blob::read_pod(is, prev_word_);
    blob::read_pod(is, word_hash_prev_);
    is.read(reinterpret_cast<char*>(word_ring_), sizeof(word_ring_));
    blob::read_pod(is, c0_);
    blob::read_pod(is, bitpos_);
    blob::read_pod(is, pr_final_);
    is.read(reinterpret_cast<char*>(exp_p_), static_cast<std::streamsize>(kExpPBase * sizeof(exp_p_[0])));
    blob::read_pod(is, n_exp_);
    blob::read_pod(is, mixed_p_);
    blob::read_pod(is, last_mlen_);
    blob::read_pod(is, sparse_);
    if (ver >= 2) checkpoint_read_trivial(is, sentmem_);  // v1: sentence memory starts empty
    if (ver >= 5) {
        std::uint32_t x = 0;
        blob::read_pod(is, x);
        if (x != xcms_) {
            is.setstate(std::ios::failbit);
            g_hp_ckpt_read_version = 5;
            return;
        }
        for (int i = kCtxModels; i < n_ctx_chain_; ++i) HP_CKPT_READ_CM(is, *ctx_chain_[i]);
        is.read(reinterpret_cast<char*>(exp_p_ + kExpPBase),
                static_cast<std::streamsize>(sizeof(exp_p_) - kExpPBase * sizeof(exp_p_[0])));
        checkpoint_read_trivial(is, wx_);
    }
    g_hp_ckpt_read_version = 5;
    rebind_internal_pointers_();
}
