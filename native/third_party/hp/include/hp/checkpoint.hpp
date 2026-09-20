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
#if HP_HASH2_O6
    HP_CKPT_WRITE_CM(os, o6b_);
#endif
    HP_CKPT_WRITE_CM(os, word_);
    HP_CKPT_WRITE_CM(os, col_);
    HP_CKPT_WRITE_CM(os, tag_);
    HP_CKPT_WRITE_CM(os, wbi_);
    HP_CKPT_WRITE_CM(os, sp13_);
    HP_CKPT_WRITE_CM(os, sp24_);
#if HP_WORD_STREAMS
    HP_CKPT_WRITE_CM(os, wstr_sp_);
#endif
#if HP_BRACKET
    HP_CKPT_WRITE_CM(os, brk_);
#endif
#if HP_LINKWORD
    HP_CKPT_WRITE_CM(os, link_);
#endif
#if HP_NUMERIC
    HP_CKPT_WRITE_CM(os, num_);
#endif
#if HP_PAT_MODEL
    HP_CKPT_WRITE_CM(os, pat_);
#endif
#if HP_PPMD
    HP_CKPT_WRITE_CM(os, ppm_);
#endif
#if HP_STEMMER
    HP_CKPT_WRITE_CM(os, stem0_);
#endif
#if HP_STEMMER
#if HP_STEMMER_N >= 2
    HP_CKPT_WRITE_CM(os, stem1_);
#endif
#endif
#if HP_SENWORD
    HP_CKPT_WRITE_CM(os, sen_);
#endif
#if HP_SENT_STREAM
    HP_CKPT_WRITE_CM(os, sentst_);
#endif
#if HP_SENT_MEM
    HP_CKPT_WRITE_CM(os, sentmem_cm_);
#endif
#if HP_SENGRP_MOD
    HP_CKPT_WRITE_CM(os, sengrp_);
#endif
#if HP_NEST_MOD
    HP_CKPT_WRITE_CM(os, nestmod_);
#endif
#if HP_PARA_MOD
    HP_CKPT_WRITE_CM(os, paramod_);
#endif
#if HP_LINE_MOD
    HP_CKPT_WRITE_CM(os, linemod_);
#endif
#if HP_STATE_MOD
    HP_CKPT_WRITE_CM(os, statemod_);
#endif
#if HP_DOM_MOD
    HP_CKPT_WRITE_CM(os, dommod_);
#endif
#if HP_HDR_MOD
    HP_CKPT_WRITE_CM(os, hdrmod_);
#endif
#if HP_DEPTH_MOD
    HP_CKPT_WRITE_CM(os, depthmod_);
#endif
#if HP_FCCXT_MOD
    HP_CKPT_WRITE_CM(os, fccxtmod_);
#endif
#if HP_TPLNAME_MOD
    HP_CKPT_WRITE_CM(os, tplmod_);
#endif
#if HP_INFOKEY_MOD
    HP_CKPT_WRITE_CM(os, infokeymod_);
#endif
#if HP_BARIDX_MOD
    HP_CKPT_WRITE_CM(os, baridxmod_);
#endif
#if HP_PERIOD_MOD
    HP_CKPT_WRITE_CM(os, periodmod_);
#endif
#if HP_PRONOUN_MOD
    HP_CKPT_WRITE_CM(os, pronounmod_);
#endif
#if HP_LINKPIPE_MOD
    HP_CKPT_WRITE_CM(os, linkpipemod_);
#endif
#if HP_CITE_MOD
    HP_CKPT_WRITE_CM(os, citemod_);
#endif
#if HP_CAT_MOD
    HP_CKPT_WRITE_CM(os, catmod_);
#endif
#if HP_REDIR_MOD
    HP_CKPT_WRITE_CM(os, redirmod_);
#endif
#if HP_HEADING_MOD
    HP_CKPT_WRITE_CM(os, headingmod_);
#endif
#if HP_EXTLINK_MOD
    HP_CKPT_WRITE_CM(os, extlinkmod_);
#endif
#if HP_REFNAME_MOD
    HP_CKPT_WRITE_CM(os, refnamemod_);
#endif
#if HP_QOCXT_MOD
    HP_CKPT_WRITE_CM(os, qocxtmod_);
#endif
#if HP_ENTITY_MOD
    HP_CKPT_WRITE_CM(os, entitymod_);
#endif
#if HP_INDENT_MOD
    HP_CKPT_WRITE_CM(os, indentmod_);
#endif
#if HP_LISTLEVEL_MOD
    HP_CKPT_WRITE_CM(os, listlevelmod_);
#endif
#if HP_ISSE_MOD
    HP_CKPT_WRITE_CM(os, issemod_);
#endif
#if HP_MAGIC_MOD
    HP_CKPT_WRITE_CM(os, magicmod_);
#endif
#if HP_NOWIKI_MOD
    HP_CKPT_WRITE_CM(os, nowikimod_);
#endif
#if HP_TITLE_MOD
    HP_CKPT_WRITE_CM(os, titlemod_);
#endif
#if HP_PAGEID_MOD
    HP_CKPT_WRITE_CM(os, pageidmod_);
#endif
#if HP_USER_MOD
    HP_CKPT_WRITE_CM(os, usermod_);
#endif
#if HP_TEXT_MOD
    HP_CKPT_WRITE_CM(os, textmod_);
#endif
#if HP_NS_MOD
    HP_CKPT_WRITE_CM(os, nsmod_);
#endif
#if HP_DUMPREDIR_MOD
    HP_CKPT_WRITE_CM(os, dumpredirmod_);
#endif
#if HP_IP_MOD
    HP_CKPT_WRITE_CM(os, ipmod_);
#endif
#if HP_REVCOMMENT_MOD
    HP_CKPT_WRITE_CM(os, revcommentmod_);
#endif
#if HP_MINOR_MOD
    HP_CKPT_WRITE_CM(os, minormod_);
#endif
#if HP_WIKIMODEL_MOD
    HP_CKPT_WRITE_CM(os, wikimodelmod_);
#endif
#if HP_SECTITLE_MOD
    HP_CKPT_WRITE_CM(os, sectitlemod_);
#endif
#if HP_PARSERFN_MOD
    HP_CKPT_WRITE_CM(os, parserfnmod_);
#endif
#if HP_TABLECLASS_MOD
    HP_CKPT_WRITE_CM(os, tableclassmod_);
#endif
#if HP_ANCHOR_MOD
    HP_CKPT_WRITE_CM(os, anchormod_);
#endif
#if HP_PUBID_MOD
    HP_CKPT_WRITE_CM(os, pubidmod_);
#endif
#if HP_TEMPPOS_MOD
    HP_CKPT_WRITE_CM(os, tempposmod_);
#endif
#if HP_WIKISTACK_MOD
    HP_CKPT_WRITE_CM(os, wikistackmod_);
#endif
#if HP_LANG_MOD
    HP_CKPT_WRITE_CM(os, langmod_);
#endif
#if HP_CATSORT_MOD
    HP_CKPT_WRITE_CM(os, catsortmod_);
#endif
#if HP_TBLROW_MOD
    HP_CKPT_WRITE_CM(os, tblrowmod_);
#endif
#if HP_FILEOPT_MOD
    HP_CKPT_WRITE_CM(os, fileoptmod_);
#endif
#if HP_DEFAULTSORT_MOD
    HP_CKPT_WRITE_CM(os, defaultsortmod_);
#endif
#if HP_REDIRTARGET_MOD
    HP_CKPT_WRITE_CM(os, redirtargetmod_);
#endif
#if HP_DAB_MOD
    HP_CKPT_WRITE_CM(os, dabmod_);
#endif
#if HP_HATNOTE_MOD
    HP_CKPT_WRITE_CM(os, hatnotemod_);
#endif
#if HP_LASTLINK_MOD
    HP_CKPT_WRITE_CM(os, lastlinkmod_);
#endif
#if HP_FWORD_MOD
    HP_CKPT_WRITE_CM(os, fwordmod_);
#endif
#if HP_YEAR_MOD
    HP_CKPT_WRITE_CM(os, yearmod_);
#endif
#if HP_CAPMASK_MOD
    HP_CKPT_WRITE_CM(os, capmaskmod_);
#endif
#if HP_CELLTXT_MOD
    HP_CKPT_WRITE_CM(os, celltxtmod_);
#endif
#if HP_HTTPHOST_MOD
    HP_CKPT_WRITE_CM(os, httphostmod_);
#endif
#if HP_PAREN_MOD
    HP_CKPT_WRITE_CM(os, parenmod_);
#endif
#if HP_LISTPOS_MOD
    HP_CKPT_WRITE_CM(os, listposmod_);
#endif
#if HP_SHAPE_MOD
    HP_CKPT_WRITE_CM(os, shapemod_);
#endif
#if HP_SUFFIX_MOD
    HP_CKPT_WRITE_CM(os, suffixmod_);
#endif
#if HP_PREFIX_MOD
    HP_CKPT_WRITE_CM(os, prefixmod_);
#endif
#if HP_CHARCLS_MOD
    HP_CKPT_WRITE_CM(os, charclsmod_);
#endif
#if HP_VOWEL_MOD
    HP_CKPT_WRITE_CM(os, vowelmod_);
#endif
#if HP_CONTR_MOD
    HP_CKPT_WRITE_CM(os, contrmod_);
#endif
#if HP_HYPHEN_MOD
    HP_CKPT_WRITE_CM(os, hyphenmod_);
#endif
#if HP_TOKENCLS_MOD
    HP_CKPT_WRITE_CM(os, tokenclsmod_);
#endif
#if HP_RUNLEN_MOD
    HP_CKPT_WRITE_CM(os, runlenmod_);
#endif
#if HP_WPOS_MOD
    HP_CKPT_WRITE_CM(os, wposmod_);
#endif
#if HP_BLANK_MOD
    HP_CKPT_WRITE_CM(os, blankmod_);
#endif
#if HP_SPRUN_MOD
    HP_CKPT_WRITE_CM(os, sprunmod_);
#endif
#if HP_LINELEN_MOD
    HP_CKPT_WRITE_CM(os, linelenmod_);
#endif
#if HP_TAGDIST_MOD
    HP_CKPT_WRITE_CM(os, tagdistmod_);
#endif
#if HP_MARKDIST_MOD
    HP_CKPT_WRITE_CM(os, markdistmod_);
#endif
#if HP_UPPERGAP_MOD
    HP_CKPT_WRITE_CM(os, uppergapmod_);
#endif
#if HP_MONTH_MOD
    HP_CKPT_WRITE_CM(os, monthmod_);
#endif
#if HP_GALLERY_MOD
    HP_CKPT_WRITE_CM(os, gallerymod_);
#endif
#if HP_SECKIND_MOD
    HP_CKPT_WRITE_CM(os, seckindmod_);
#endif
#if HP_CITEKIND_MOD
    HP_CKPT_WRITE_CM(os, citekindmod_);
#endif
#if HP_TAGNAME_MOD
    HP_CKPT_WRITE_CM(os, tagnamemod_);
#endif
#if HP_COLSPAN_MOD
    HP_CKPT_WRITE_CM(os, colspanmod_);
#endif
#if HP_STYLE_MOD
    HP_CKPT_WRITE_CM(os, stylemod_);
#endif
#if HP_COORD_MOD
    HP_CKPT_WRITE_CM(os, coordmod_);
#endif
#if HP_DIGITGAP_MOD
    HP_CKPT_WRITE_CM(os, digitgapmod_);
#endif
#if HP_DOTGAP_MOD
    HP_CKPT_WRITE_CM(os, dotgapmod_);
#endif
#if HP_COMMAGAP_MOD
    HP_CKPT_WRITE_CM(os, commagapmod_);
#endif
#if HP_WORDLEN_MOD
    HP_CKPT_WRITE_CM(os, wordlenmod_);
#endif
#if HP_SENTLEN_MOD
    HP_CKPT_WRITE_CM(os, sentlenmod_);
#endif
#if HP_LOWERGAP_MOD
    HP_CKPT_WRITE_CM(os, lowergapmod_);
#endif
#if HP_DIGITPOS_MOD
    HP_CKPT_WRITE_CM(os, digitposmod_);
#endif
#if HP_SLASHGAP_MOD
    HP_CKPT_WRITE_CM(os, slashgapmod_);
#endif
#if HP_DIGLEN_MOD
    HP_CKPT_WRITE_CM(os, diglenmod_);
#endif
#if HP_PREVLINE_MOD
    HP_CKPT_WRITE_CM(os, prevlinemod_);
#endif
#if HP_PREVSENT_MOD
    HP_CKPT_WRITE_CM(os, prevsentmod_);
#endif
#if HP_LINKLEN_MOD
    HP_CKPT_WRITE_CM(os, linklenmod_);
#endif
#if HP_TPLLEN_MOD
    HP_CKPT_WRITE_CM(os, tpllenmod_);
#endif
#if HP_PARALEN_MOD
    HP_CKPT_WRITE_CM(os, paralenmod_);
#endif
#if HP_ALNUMLEN_MOD
    HP_CKPT_WRITE_CM(os, alnumlenmod_);
#endif
#if HP_SPLEN_MOD
    HP_CKPT_WRITE_CM(os, splenmod_);
#endif
#if HP_TITLEWORD_MOD
    HP_CKPT_WRITE_CM(os, titlewordmod_);
#endif
#if HP_HEADWORD_MOD
    HP_CKPT_WRITE_CM(os, headwordmod_);
#endif
#if HP_INIT_MOD
    HP_CKPT_WRITE_CM(os, initmod_);
#endif
#if HP_ORDINAL_MOD
    HP_CKPT_WRITE_CM(os, ordinalmod_);
#endif
#if HP_UNIT_MOD
    HP_CKPT_WRITE_CM(os, unitmod_);
#endif
#if HP_DECIMAL_MOD
    HP_CKPT_WRITE_CM(os, decimalmod_);
#endif
#if HP_REPEAT_MOD
    HP_CKPT_WRITE_CM(os, repeatmod_);
#endif
#if HP_CASEFLIP_MOD
    HP_CKPT_WRITE_CM(os, caseflipmod_);
#endif
#if HP_LEAD_MOD
    HP_CKPT_WRITE_CM(os, leadmod_);
#endif
#if HP_INFOVAL_MOD
    HP_CKPT_WRITE_CM(os, infovalmod_);
#endif
#if HP_LINKTRAIL_MOD
    HP_CKPT_WRITE_CM(os, linktrailmod_);
#endif
#if HP_CELLKIND_MOD
    HP_CKPT_WRITE_CM(os, cellkindmod_);
#endif
#if HP_TBLCOL_MOD
    HP_CKPT_WRITE_CM(os, tblcolmod_);
#endif
#if HP_HEADIDX_MOD
    HP_CKPT_WRITE_CM(os, headidxmod_);
#endif
#if HP_HTMLFMT_MOD
    HP_CKPT_WRITE_CM(os, htmlfmtmod_);
#endif
#if HP_INFOBOX_MOD
    HP_CKPT_WRITE_CM(os, infoboxmod_);
#endif
#if HP_SECLEVEL_MOD
    HP_CKPT_WRITE_CM(os, seclevelmod_);
#endif
#if HP_BRACE3_MOD
    HP_CKPT_WRITE_CM(os, brace3mod_);
#endif
#if HP_NAMEDARG_MOD
    HP_CKPT_WRITE_CM(os, namedargmod_);
#endif
#if HP_INCLUDE_MOD
    HP_CKPT_WRITE_CM(os, includemod_);
#endif
#if HP_SIG_MOD
    HP_CKPT_WRITE_CM(os, sigmod_);
#endif
#if HP_WIKIBOLD_MOD
    HP_CKPT_WRITE_CM(os, wikiboldmod_);
#endif
#if HP_URLPART_MOD
    HP_CKPT_WRITE_CM(os, urlpartmod_);
#endif
#if HP_REFIDX_MOD
    HP_CKPT_WRITE_CM(os, refidxmod_);
#endif
    for (int i = 0; i < kMatchModels; ++i) match_[i].checkpoint_write(os);
#if HP_SPARSE_UTF8
    smatch_.checkpoint_write(os);
#endif
#if HP_SKIPK_MOD
    skipk_.checkpoint_write(os);
#endif
#if HP_SKIP3_MOD
    skip3_.checkpoint_write(os);
#endif
#if HP_SKIP4_MOD
    skip4_.checkpoint_write(os);
#endif
#if HP_SKIP5_MOD
    skip5_.checkpoint_write(os);
#endif
#if HP_LZP_MOD
    lzp_.checkpoint_write(os);
#endif
#if HP_DMC_MOD
    dmc_.checkpoint_write(os);
#endif
#if HP_WORD_MATCH
    for (int i = 0; i < static_cast<int>(sizeof(wmatch_) / sizeof(wmatch_[0])); ++i) {
        wmatch_[i].checkpoint_write(os);
    }
#endif
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
#if HP_STEMMER || HP_STEM_FOLD || HP_POS_GATE || HP_WT3_CTX
    checkpoint_write_trivial(os, stems_);
#endif
    checkpoint_write_trivial(os, brackets_);
    checkpoint_write_trivial(os, cache_);
#if HP_NUMERIC
    checkpoint_write_trivial(os, numbers_);
#endif
#if HP_GATE_BRANCH
    checkpoint_write_trivial(os, branch3_);
#endif
#if HP_GATE_BREAK
    blob::write_pod(os, break_age_);
#endif
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
#if HP_PRONOUN_MOD
    os.write(reinterpret_cast<const char*>(pw_), sizeof(pw_));
    blob::write_pod(os, pw_n_);
    blob::write_pod(os, pronoun_);
#endif
#if HP_UTF8_IDLE || HP_GATE_UTF8
    blob::write_pod(os, utf8left_);
#endif
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
#if HP_HASH2_O6
    HP_CKPT_READ_CM(is, o6b_);
#endif
    HP_CKPT_READ_CM(is, word_);
    HP_CKPT_READ_CM(is, col_);
    HP_CKPT_READ_CM(is, tag_);
    HP_CKPT_READ_CM(is, wbi_);
    HP_CKPT_READ_CM(is, sp13_);
    HP_CKPT_READ_CM(is, sp24_);
#if HP_WORD_STREAMS
    HP_CKPT_READ_CM(is, wstr_sp_);
#endif
#if HP_BRACKET
    HP_CKPT_READ_CM(is, brk_);
#endif
#if HP_LINKWORD
    HP_CKPT_READ_CM(is, link_);
#endif
#if HP_NUMERIC
    HP_CKPT_READ_CM(is, num_);
#endif
#if HP_PAT_MODEL
    HP_CKPT_READ_CM(is, pat_);
#endif
#if HP_PPMD
    HP_CKPT_READ_CM(is, ppm_);
#endif
#if HP_STEMMER
    HP_CKPT_READ_CM(is, stem0_);
#endif
#if HP_STEMMER
#if HP_STEMMER_N >= 2
    HP_CKPT_READ_CM(is, stem1_);
#endif
#endif
#if HP_SENWORD
    HP_CKPT_READ_CM(is, sen_);
#endif
#if HP_SENT_STREAM
    HP_CKPT_READ_CM(is, sentst_);
#endif
#if HP_SENT_MEM
    HP_CKPT_READ_CM(is, sentmem_cm_);
#endif
#if HP_SENGRP_MOD
    HP_CKPT_READ_CM(is, sengrp_);
#endif
#if HP_NEST_MOD
    HP_CKPT_READ_CM(is, nestmod_);
#endif
#if HP_PARA_MOD
    HP_CKPT_READ_CM(is, paramod_);
#endif
#if HP_LINE_MOD
    HP_CKPT_READ_CM(is, linemod_);
#endif
#if HP_STATE_MOD
    HP_CKPT_READ_CM(is, statemod_);
#endif
#if HP_DOM_MOD
    HP_CKPT_READ_CM(is, dommod_);
#endif
#if HP_HDR_MOD
    HP_CKPT_READ_CM(is, hdrmod_);
#endif
#if HP_DEPTH_MOD
    HP_CKPT_READ_CM(is, depthmod_);
#endif
#if HP_FCCXT_MOD
    HP_CKPT_READ_CM(is, fccxtmod_);
#endif
#if HP_TPLNAME_MOD
    HP_CKPT_READ_CM(is, tplmod_);
#endif
#if HP_INFOKEY_MOD
    HP_CKPT_READ_CM(is, infokeymod_);
#endif
#if HP_BARIDX_MOD
    HP_CKPT_READ_CM(is, baridxmod_);
#endif
#if HP_PERIOD_MOD
    HP_CKPT_READ_CM(is, periodmod_);
#endif
#if HP_PRONOUN_MOD
    HP_CKPT_READ_CM(is, pronounmod_);
#endif
#if HP_LINKPIPE_MOD
    HP_CKPT_READ_CM(is, linkpipemod_);
#endif
#if HP_CITE_MOD
    HP_CKPT_READ_CM(is, citemod_);
#endif
#if HP_CAT_MOD
    HP_CKPT_READ_CM(is, catmod_);
#endif
#if HP_REDIR_MOD
    HP_CKPT_READ_CM(is, redirmod_);
#endif
#if HP_HEADING_MOD
    HP_CKPT_READ_CM(is, headingmod_);
#endif
#if HP_EXTLINK_MOD
    HP_CKPT_READ_CM(is, extlinkmod_);
#endif
#if HP_REFNAME_MOD
    HP_CKPT_READ_CM(is, refnamemod_);
#endif
#if HP_QOCXT_MOD
    HP_CKPT_READ_CM(is, qocxtmod_);
#endif
#if HP_ENTITY_MOD
    HP_CKPT_READ_CM(is, entitymod_);
#endif
#if HP_INDENT_MOD
    HP_CKPT_READ_CM(is, indentmod_);
#endif
#if HP_LISTLEVEL_MOD
    HP_CKPT_READ_CM(is, listlevelmod_);
#endif
#if HP_ISSE_MOD
    HP_CKPT_READ_CM(is, issemod_);
#endif
#if HP_MAGIC_MOD
    HP_CKPT_READ_CM(is, magicmod_);
#endif
#if HP_NOWIKI_MOD
    HP_CKPT_READ_CM(is, nowikimod_);
#endif
#if HP_TITLE_MOD
    HP_CKPT_READ_CM(is, titlemod_);
#endif
#if HP_PAGEID_MOD
    HP_CKPT_READ_CM(is, pageidmod_);
#endif
#if HP_USER_MOD
    HP_CKPT_READ_CM(is, usermod_);
#endif
#if HP_TEXT_MOD
    HP_CKPT_READ_CM(is, textmod_);
#endif
#if HP_NS_MOD
    HP_CKPT_READ_CM(is, nsmod_);
#endif
#if HP_DUMPREDIR_MOD
    HP_CKPT_READ_CM(is, dumpredirmod_);
#endif
#if HP_IP_MOD
    HP_CKPT_READ_CM(is, ipmod_);
#endif
#if HP_REVCOMMENT_MOD
    HP_CKPT_READ_CM(is, revcommentmod_);
#endif
#if HP_MINOR_MOD
    HP_CKPT_READ_CM(is, minormod_);
#endif
#if HP_WIKIMODEL_MOD
    HP_CKPT_READ_CM(is, wikimodelmod_);
#endif
#if HP_SECTITLE_MOD
    HP_CKPT_READ_CM(is, sectitlemod_);
#endif
#if HP_PARSERFN_MOD
    HP_CKPT_READ_CM(is, parserfnmod_);
#endif
#if HP_TABLECLASS_MOD
    HP_CKPT_READ_CM(is, tableclassmod_);
#endif
#if HP_ANCHOR_MOD
    HP_CKPT_READ_CM(is, anchormod_);
#endif
#if HP_PUBID_MOD
    HP_CKPT_READ_CM(is, pubidmod_);
#endif
#if HP_TEMPPOS_MOD
    HP_CKPT_READ_CM(is, tempposmod_);
#endif
#if HP_WIKISTACK_MOD
    HP_CKPT_READ_CM(is, wikistackmod_);
#endif
#if HP_LANG_MOD
    HP_CKPT_READ_CM(is, langmod_);
#endif
#if HP_CATSORT_MOD
    HP_CKPT_READ_CM(is, catsortmod_);
#endif
#if HP_TBLROW_MOD
    HP_CKPT_READ_CM(is, tblrowmod_);
#endif
#if HP_FILEOPT_MOD
    HP_CKPT_READ_CM(is, fileoptmod_);
#endif
#if HP_DEFAULTSORT_MOD
    HP_CKPT_READ_CM(is, defaultsortmod_);
#endif
#if HP_REDIRTARGET_MOD
    HP_CKPT_READ_CM(is, redirtargetmod_);
#endif
#if HP_DAB_MOD
    HP_CKPT_READ_CM(is, dabmod_);
#endif
#if HP_HATNOTE_MOD
    HP_CKPT_READ_CM(is, hatnotemod_);
#endif
#if HP_LASTLINK_MOD
    HP_CKPT_READ_CM(is, lastlinkmod_);
#endif
#if HP_FWORD_MOD
    HP_CKPT_READ_CM(is, fwordmod_);
#endif
#if HP_YEAR_MOD
    HP_CKPT_READ_CM(is, yearmod_);
#endif
#if HP_CAPMASK_MOD
    HP_CKPT_READ_CM(is, capmaskmod_);
#endif
#if HP_CELLTXT_MOD
    HP_CKPT_READ_CM(is, celltxtmod_);
#endif
#if HP_HTTPHOST_MOD
    HP_CKPT_READ_CM(is, httphostmod_);
#endif
#if HP_PAREN_MOD
    HP_CKPT_READ_CM(is, parenmod_);
#endif
#if HP_LISTPOS_MOD
    HP_CKPT_READ_CM(is, listposmod_);
#endif
#if HP_SHAPE_MOD
    HP_CKPT_READ_CM(is, shapemod_);
#endif
#if HP_SUFFIX_MOD
    HP_CKPT_READ_CM(is, suffixmod_);
#endif
#if HP_PREFIX_MOD
    HP_CKPT_READ_CM(is, prefixmod_);
#endif
#if HP_CHARCLS_MOD
    HP_CKPT_READ_CM(is, charclsmod_);
#endif
#if HP_VOWEL_MOD
    HP_CKPT_READ_CM(is, vowelmod_);
#endif
#if HP_CONTR_MOD
    HP_CKPT_READ_CM(is, contrmod_);
#endif
#if HP_HYPHEN_MOD
    HP_CKPT_READ_CM(is, hyphenmod_);
#endif
#if HP_TOKENCLS_MOD
    HP_CKPT_READ_CM(is, tokenclsmod_);
#endif
#if HP_RUNLEN_MOD
    HP_CKPT_READ_CM(is, runlenmod_);
#endif
#if HP_WPOS_MOD
    HP_CKPT_READ_CM(is, wposmod_);
#endif
#if HP_BLANK_MOD
    HP_CKPT_READ_CM(is, blankmod_);
#endif
#if HP_SPRUN_MOD
    HP_CKPT_READ_CM(is, sprunmod_);
#endif
#if HP_LINELEN_MOD
    HP_CKPT_READ_CM(is, linelenmod_);
#endif
#if HP_TAGDIST_MOD
    HP_CKPT_READ_CM(is, tagdistmod_);
#endif
#if HP_MARKDIST_MOD
    HP_CKPT_READ_CM(is, markdistmod_);
#endif
#if HP_UPPERGAP_MOD
    HP_CKPT_READ_CM(is, uppergapmod_);
#endif
#if HP_MONTH_MOD
    HP_CKPT_READ_CM(is, monthmod_);
#endif
#if HP_GALLERY_MOD
    HP_CKPT_READ_CM(is, gallerymod_);
#endif
#if HP_SECKIND_MOD
    HP_CKPT_READ_CM(is, seckindmod_);
#endif
#if HP_CITEKIND_MOD
    HP_CKPT_READ_CM(is, citekindmod_);
#endif
#if HP_TAGNAME_MOD
    HP_CKPT_READ_CM(is, tagnamemod_);
#endif
#if HP_COLSPAN_MOD
    HP_CKPT_READ_CM(is, colspanmod_);
#endif
#if HP_STYLE_MOD
    HP_CKPT_READ_CM(is, stylemod_);
#endif
#if HP_COORD_MOD
    HP_CKPT_READ_CM(is, coordmod_);
#endif
#if HP_DIGITGAP_MOD
    HP_CKPT_READ_CM(is, digitgapmod_);
#endif
#if HP_DOTGAP_MOD
    HP_CKPT_READ_CM(is, dotgapmod_);
#endif
#if HP_COMMAGAP_MOD
    HP_CKPT_READ_CM(is, commagapmod_);
#endif
#if HP_WORDLEN_MOD
    HP_CKPT_READ_CM(is, wordlenmod_);
#endif
#if HP_SENTLEN_MOD
    HP_CKPT_READ_CM(is, sentlenmod_);
#endif
#if HP_LOWERGAP_MOD
    HP_CKPT_READ_CM(is, lowergapmod_);
#endif
#if HP_DIGITPOS_MOD
    HP_CKPT_READ_CM(is, digitposmod_);
#endif
#if HP_SLASHGAP_MOD
    HP_CKPT_READ_CM(is, slashgapmod_);
#endif
#if HP_DIGLEN_MOD
    HP_CKPT_READ_CM(is, diglenmod_);
#endif
#if HP_PREVLINE_MOD
    HP_CKPT_READ_CM(is, prevlinemod_);
#endif
#if HP_PREVSENT_MOD
    HP_CKPT_READ_CM(is, prevsentmod_);
#endif
#if HP_LINKLEN_MOD
    HP_CKPT_READ_CM(is, linklenmod_);
#endif
#if HP_TPLLEN_MOD
    HP_CKPT_READ_CM(is, tpllenmod_);
#endif
#if HP_PARALEN_MOD
    HP_CKPT_READ_CM(is, paralenmod_);
#endif
#if HP_ALNUMLEN_MOD
    HP_CKPT_READ_CM(is, alnumlenmod_);
#endif
#if HP_SPLEN_MOD
    HP_CKPT_READ_CM(is, splenmod_);
#endif
#if HP_TITLEWORD_MOD
    HP_CKPT_READ_CM(is, titlewordmod_);
#endif
#if HP_HEADWORD_MOD
    HP_CKPT_READ_CM(is, headwordmod_);
#endif
#if HP_INIT_MOD
    HP_CKPT_READ_CM(is, initmod_);
#endif
#if HP_ORDINAL_MOD
    HP_CKPT_READ_CM(is, ordinalmod_);
#endif
#if HP_UNIT_MOD
    HP_CKPT_READ_CM(is, unitmod_);
#endif
#if HP_DECIMAL_MOD
    HP_CKPT_READ_CM(is, decimalmod_);
#endif
#if HP_REPEAT_MOD
    HP_CKPT_READ_CM(is, repeatmod_);
#endif
#if HP_CASEFLIP_MOD
    HP_CKPT_READ_CM(is, caseflipmod_);
#endif
#if HP_LEAD_MOD
    HP_CKPT_READ_CM(is, leadmod_);
#endif
#if HP_INFOVAL_MOD
    HP_CKPT_READ_CM(is, infovalmod_);
#endif
#if HP_LINKTRAIL_MOD
    HP_CKPT_READ_CM(is, linktrailmod_);
#endif
#if HP_CELLKIND_MOD
    HP_CKPT_READ_CM(is, cellkindmod_);
#endif
#if HP_TBLCOL_MOD
    HP_CKPT_READ_CM(is, tblcolmod_);
#endif
#if HP_HEADIDX_MOD
    HP_CKPT_READ_CM(is, headidxmod_);
#endif
#if HP_HTMLFMT_MOD
    HP_CKPT_READ_CM(is, htmlfmtmod_);
#endif
#if HP_INFOBOX_MOD
    HP_CKPT_READ_CM(is, infoboxmod_);
#endif
#if HP_SECLEVEL_MOD
    HP_CKPT_READ_CM(is, seclevelmod_);
#endif
#if HP_BRACE3_MOD
    HP_CKPT_READ_CM(is, brace3mod_);
#endif
#if HP_NAMEDARG_MOD
    HP_CKPT_READ_CM(is, namedargmod_);
#endif
#if HP_INCLUDE_MOD
    HP_CKPT_READ_CM(is, includemod_);
#endif
#if HP_SIG_MOD
    HP_CKPT_READ_CM(is, sigmod_);
#endif
#if HP_WIKIBOLD_MOD
    HP_CKPT_READ_CM(is, wikiboldmod_);
#endif
#if HP_URLPART_MOD
    HP_CKPT_READ_CM(is, urlpartmod_);
#endif
#if HP_REFIDX_MOD
    HP_CKPT_READ_CM(is, refidxmod_);
#endif
    for (int i = 0; i < kMatchModels; ++i) match_[i].checkpoint_read(is);
#if HP_SPARSE_UTF8
    smatch_.checkpoint_read(is);
#endif
#if HP_SKIPK_MOD
    skipk_.checkpoint_read(is);
#endif
#if HP_SKIP3_MOD
    skip3_.checkpoint_read(is);
#endif
#if HP_SKIP4_MOD
    skip4_.checkpoint_read(is);
#endif
#if HP_SKIP5_MOD
    skip5_.checkpoint_read(is);
#endif
#if HP_LZP_MOD
    lzp_.checkpoint_read(is);
#endif
#if HP_DMC_MOD
    dmc_.checkpoint_read(is);
#endif
#if HP_WORD_MATCH
    for (int i = 0; i < static_cast<int>(sizeof(wmatch_) / sizeof(wmatch_[0])); ++i) {
        wmatch_[i].checkpoint_read(is);
    }
#endif
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
#if HP_STEMMER || HP_STEM_FOLD || HP_POS_GATE || HP_WT3_CTX
    checkpoint_read_trivial(is, stems_);
#endif
    checkpoint_read_trivial(is, brackets_);
    checkpoint_read_trivial(is, cache_);
#if HP_NUMERIC
    checkpoint_read_trivial(is, numbers_);
#endif
#if HP_GATE_BRANCH
    checkpoint_read_trivial(is, branch3_);
#endif
#if HP_GATE_BREAK
    blob::read_pod(is, break_age_);
#endif
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
#if HP_PRONOUN_MOD
    is.read(reinterpret_cast<char*>(pw_), sizeof(pw_));
    blob::read_pod(is, pw_n_);
    blob::read_pod(is, pronoun_);
#endif
#if HP_UTF8_IDLE || HP_GATE_UTF8
    blob::read_pod(is, utf8left_);
#endif
    rebind_internal_pointers_();
}
