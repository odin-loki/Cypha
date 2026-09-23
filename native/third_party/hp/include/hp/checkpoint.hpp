#pragma once
// hp/checkpoint.hpp — binary Predictor checkpoint (Cypha train-once / serve-many).

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
    const std::uint32_t ver = 1;
    blob::write_pod(os, ver);
    byte_ring_.checkpoint_write(os);
    HP_CKPT_WRITE_CM(os, o1_);
    HP_CKPT_WRITE_CM(os, o2_);
    HP_CKPT_WRITE_CM(os, o3_);
    HP_CKPT_WRITE_CM(os, o4_);
    HP_CKPT_WRITE_CM(os, o6_);
    HP_CKPT_WRITE_CM(os, o6b_);
    HP_CKPT_WRITE_CM(os, word_);
    HP_CKPT_WRITE_CM(os, col_);
    HP_CKPT_WRITE_CM(os, tag_);
    HP_CKPT_WRITE_CM(os, wbi_);
    HP_CKPT_WRITE_CM(os, sp13_);
    HP_CKPT_WRITE_CM(os, sp24_);
    HP_CKPT_WRITE_CM(os, wstr_sp_);
    HP_CKPT_WRITE_CM(os, brk_);
    HP_CKPT_WRITE_CM(os, link_);
    HP_CKPT_WRITE_CM(os, num_);
    HP_CKPT_WRITE_CM(os, sen_);
    HP_CKPT_WRITE_CM(os, sentst_);
    HP_CKPT_WRITE_CM(os, sentmem_cm_);
    HP_CKPT_WRITE_CM(os, sengrp_);
    HP_CKPT_WRITE_CM(os, nestmod_);
    HP_CKPT_WRITE_CM(os, paramod_);
    HP_CKPT_WRITE_CM(os, linemod_);
    HP_CKPT_WRITE_CM(os, statemod_);
    HP_CKPT_WRITE_CM(os, tplmod_);
    HP_CKPT_WRITE_CM(os, infokeymod_);
    HP_CKPT_WRITE_CM(os, linkpipemod_);
    HP_CKPT_WRITE_CM(os, catmod_);
    HP_CKPT_WRITE_CM(os, headingmod_);
    HP_CKPT_WRITE_CM(os, titlemod_);
    HP_CKPT_WRITE_CM(os, sectitlemod_);
    HP_CKPT_WRITE_CM(os, wikistackmod_);
    HP_CKPT_WRITE_CM(os, capmaskmod_);
    HP_CKPT_WRITE_CM(os, uppergapmod_);
    HP_CKPT_WRITE_CM(os, wordlenmod_);
    for (int i = 0; i < kMatchModels; ++i) match_[i].checkpoint_write(os);
    smatch_.checkpoint_write(os);
    skipk_.checkpoint_write(os);
    skip3_.checkpoint_write(os);
    skip4_.checkpoint_write(os);
    lzp_.checkpoint_write(os);
    dmc_.checkpoint_write(os);
    for (int i = 0; i < static_cast<int>(sizeof(wmatch_) / sizeof(wmatch_[0])); ++i) {
        wmatch_[i].checkpoint_write(os);
    }
    hebb_.checkpoint_write(os);
    pool_.checkpoint_write(os);
    mixer_.checkpoint_write(os);
    apm_c0_.checkpoint_write(os);
    apm_lex_.checkpoint_write(os);
    apm_gria_.checkpoint_write(os);
    hedge_.checkpoint_write(os);
    blob::write_array(os, bias_);
    gria_.checkpoint_write(os);
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
    os.write(reinterpret_cast<const char*>(exp_p_), sizeof(exp_p_));
    blob::write_pod(os, n_exp_);
    blob::write_pod(os, mixed_p_);
    blob::write_pod(os, last_mlen_);
    blob::write_pod(os, sparse_);
}

inline void Predictor::read_checkpoint(std::istream& is) {
    char magic[4] = {};
    is.read(magic, 4);
    if (magic[0] != 'H' || magic[1] != 'P' || magic[2] != 'C' || magic[3] != 'P') {
        return;
    }
    std::uint32_t ver = 0;
    blob::read_pod(is, ver);
    if (ver != 1) return;
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
    is.read(reinterpret_cast<char*>(exp_p_), sizeof(exp_p_));
    blob::read_pod(is, n_exp_);
    blob::read_pod(is, mixed_p_);
    blob::read_pod(is, last_mlen_);
    blob::read_pod(is, sparse_);
    rebind_internal_pointers_();
}
