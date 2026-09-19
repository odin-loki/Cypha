#pragma once
//
// hp/predictor.hpp — the model stack.
//
// Per bit:
//   1. every expert emits a stretched opinion
//   2. the mixer combines them, gated by (GRIA alpha bucket, partial byte, …)
//   3. two/three APM stages recalibrate
//   4. the coder codes the bit with the final probability
//   5. everything updates on the observed bit
//
// Steps 1-3 and 5 are IDENTICAL in the compressor and the decompressor, and
// depend only on data both sides already have.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hp/bracket.hpp"
#include "hp/discover.hpp"
#include "hp/english.hpp"
#include "hp/features.hpp"
#include "hp/gria.hpp"
#include "hp/hedge.hpp"
#include "hp/int_math.hpp"
#include "hp/mixer.hpp"
#include "hp/models.hpp"
#include "hp/numeric.hpp"
#include "hp/pattern_cache.hpp"
#include "hp/stat_gates.hpp"
#include "hp/statemap.hpp"
#include "hp/wiki.hpp"
#include "hp/stemmer.hpp"
#include "hp/wordmatch.hpp"
#include "hp/wordstream.hpp"
#include "hp/sentmem.hpp"

#ifndef HP_W0
#define HP_W0 0
#endif
#ifndef HP_WA
#define HP_WA 1
#endif
#ifndef HP_WB
#define HP_WB 5
#endif
#ifndef HP_WG
#define HP_WG 2
#endif
#ifndef HP_APM_CTX
#define HP_APM_CTX 0
#endif
#ifndef HP_APM_NCTX
#define HP_APM_NCTX 256
#endif
#ifndef HP_USE_ENT_GATE
#define HP_USE_ENT_GATE 1
#endif

namespace hp {

struct Config {
    int table_bits = 22;
    int buf_bits = 26;
    int match_bits = 22;
    int mixer_lr = 2;
    bool gria = true;

    // Encoder and decoder must agree. match/buf sizes are a function of
    // table_bits (the only size the archive header carries).
    // buf_bits = table_bits + HP_BUF_DELTA (default 3 -> 25 at mem 22).
    void normalize() {
        if (table_bits < 16) table_bits = 16;
        if (table_bits > 28) table_bits = 28;
        match_bits = table_bits;
        buf_bits = table_bits + HP_BUF_DELTA;
        if (buf_bits > 28) buf_bits = 28;
    }
};

class Predictor {
 public:
    static constexpr int kExtraCtx =
        (HP_WORD_STREAMS ? 1 : 0) +
        (HP_BRACKET ? 1 : 0) +
        (HP_LINKWORD ? 1 : 0) +
        (HP_NUMERIC ? 1 : 0) +
        (HP_PAT_MODEL ? 1 : 0) +
        (HP_PPMD ? 1 : 0) +
        (HP_STEMMER ? HP_STEMMER_N : 0) +
        (HP_SENWORD ? 1 : 0) +
        (HP_SENT_STREAM ? 1 : 0) +
        (HP_SENT_MEM ? 1 : 0) +
        (HP_SENGRP_MOD ? 1 : 0) +
        (HP_NEST_MOD ? 1 : 0) +
        (HP_PARA_MOD ? 1 : 0) +
        (HP_LINE_MOD ? 1 : 0) +
        (HP_STATE_MOD ? 1 : 0) +
        (HP_DOM_MOD ? 1 : 0) +
        (HP_HDR_MOD ? 1 : 0) +
        (HP_DEPTH_MOD ? 1 : 0) +
        (HP_FCCXT_MOD ? 1 : 0) +
        (HP_TPLNAME_MOD ? 1 : 0) +
        (HP_INFOKEY_MOD ? 1 : 0) +
        (HP_BARIDX_MOD ? 1 : 0) +
        (HP_PERIOD_MOD ? 1 : 0) +
        (HP_PRONOUN_MOD ? 1 : 0) +
        (HP_HASH2_O6 ? 1 : 0) +
        (HP_LINKPIPE_MOD ? 1 : 0) +
        (HP_CITE_MOD ? 1 : 0) +
        (HP_CAT_MOD ? 1 : 0) +
        (HP_REDIR_MOD ? 1 : 0) +
        (HP_HEADING_MOD ? 1 : 0) +
        (HP_EXTLINK_MOD ? 1 : 0) +
        (HP_REFNAME_MOD ? 1 : 0) +
        (HP_QOCXT_MOD ? 1 : 0) +
        (HP_ENTITY_MOD ? 1 : 0) +
        (HP_INDENT_MOD ? 1 : 0) +
        (HP_LISTLEVEL_MOD ? 1 : 0) +
        (HP_ISSE_MOD ? 1 : 0) +
        (HP_MAGIC_MOD ? 1 : 0) +
        (HP_NOWIKI_MOD ? 1 : 0) +
        (HP_TITLE_MOD ? 1 : 0) +
        (HP_PAGEID_MOD ? 1 : 0) +
        (HP_USER_MOD ? 1 : 0) +
        (HP_TEXT_MOD ? 1 : 0) +
        (HP_NS_MOD ? 1 : 0) +
        (HP_DUMPREDIR_MOD ? 1 : 0) +
        (HP_IP_MOD ? 1 : 0) +
        (HP_REVCOMMENT_MOD ? 1 : 0) +
        (HP_MINOR_MOD ? 1 : 0) +
        (HP_WIKIMODEL_MOD ? 1 : 0) +
        (HP_SECTITLE_MOD ? 1 : 0) +
        (HP_PARSERFN_MOD ? 1 : 0) +
        (HP_TABLECLASS_MOD ? 1 : 0) +
        (HP_ANCHOR_MOD ? 1 : 0) +
        (HP_PUBID_MOD ? 1 : 0) +
        (HP_TEMPPOS_MOD ? 1 : 0) +
        (HP_WIKISTACK_MOD ? 1 : 0) +
        (HP_LANG_MOD ? 1 : 0) +
        (HP_CATSORT_MOD ? 1 : 0) +
        (HP_TBLROW_MOD ? 1 : 0) +
        (HP_FILEOPT_MOD ? 1 : 0) +
        (HP_DEFAULTSORT_MOD ? 1 : 0) +
        (HP_REDIRTARGET_MOD ? 1 : 0) +
        (HP_DAB_MOD ? 1 : 0) +
        (HP_HATNOTE_MOD ? 1 : 0) +
        (HP_LASTLINK_MOD ? 1 : 0) +
        (HP_FWORD_MOD ? 1 : 0) +
        (HP_YEAR_MOD ? 1 : 0) +
        (HP_CAPMASK_MOD ? 1 : 0) +
        (HP_CELLTXT_MOD ? 1 : 0) +
        (HP_HTTPHOST_MOD ? 1 : 0) +
        (HP_PAREN_MOD ? 1 : 0) +
        (HP_LISTPOS_MOD ? 1 : 0) +
        (HP_SHAPE_MOD ? 1 : 0) +
        (HP_SUFFIX_MOD ? 1 : 0) +
        (HP_PREFIX_MOD ? 1 : 0) +
        (HP_CHARCLS_MOD ? 1 : 0) +
        (HP_VOWEL_MOD ? 1 : 0) +
        (HP_CONTR_MOD ? 1 : 0) +
        (HP_HYPHEN_MOD ? 1 : 0) +
        (HP_TOKENCLS_MOD ? 1 : 0) +
        (HP_RUNLEN_MOD ? 1 : 0) +
        (HP_WPOS_MOD ? 1 : 0) +
        (HP_BLANK_MOD ? 1 : 0) +
        (HP_SPRUN_MOD ? 1 : 0) +
        (HP_LINELEN_MOD ? 1 : 0) +
        (HP_TAGDIST_MOD ? 1 : 0) +
        (HP_MARKDIST_MOD ? 1 : 0) +
        (HP_UPPERGAP_MOD ? 1 : 0) +
        (HP_MONTH_MOD ? 1 : 0) +
        (HP_GALLERY_MOD ? 1 : 0) +
        (HP_SECKIND_MOD ? 1 : 0) +
        (HP_CITEKIND_MOD ? 1 : 0) +
        (HP_TAGNAME_MOD ? 1 : 0) +
        (HP_COLSPAN_MOD ? 1 : 0) +
        (HP_STYLE_MOD ? 1 : 0) +
        (HP_COORD_MOD ? 1 : 0) +
        (HP_DIGITGAP_MOD ? 1 : 0) +
        (HP_DOTGAP_MOD ? 1 : 0) +
        (HP_COMMAGAP_MOD ? 1 : 0) +
        (HP_WORDLEN_MOD ? 1 : 0) +
        (HP_SENTLEN_MOD ? 1 : 0) +
        (HP_LOWERGAP_MOD ? 1 : 0) +
        (HP_DIGITPOS_MOD ? 1 : 0) +
        (HP_SLASHGAP_MOD ? 1 : 0) +
        (HP_DIGLEN_MOD ? 1 : 0) +
        (HP_PREVLINE_MOD ? 1 : 0) +
        (HP_PREVSENT_MOD ? 1 : 0) +
        (HP_LINKLEN_MOD ? 1 : 0) +
        (HP_TPLLEN_MOD ? 1 : 0) +
        (HP_PARALEN_MOD ? 1 : 0) +
        (HP_ALNUMLEN_MOD ? 1 : 0) +
        (HP_SPLEN_MOD ? 1 : 0) +
        (HP_TITLEWORD_MOD ? 1 : 0) +
        (HP_HEADWORD_MOD ? 1 : 0) +
        (HP_INIT_MOD ? 1 : 0) +
        (HP_ORDINAL_MOD ? 1 : 0) +
        (HP_UNIT_MOD ? 1 : 0) +
        (HP_DECIMAL_MOD ? 1 : 0) +
        (HP_REPEAT_MOD ? 1 : 0) +
        (HP_CASEFLIP_MOD ? 1 : 0) +
        (HP_LEAD_MOD ? 1 : 0) +
        (HP_INFOVAL_MOD ? 1 : 0) +
        (HP_LINKTRAIL_MOD ? 1 : 0) +
        (HP_CELLKIND_MOD ? 1 : 0) +
        (HP_TBLCOL_MOD ? 1 : 0) +
        (HP_HEADIDX_MOD ? 1 : 0) +
        (HP_HTMLFMT_MOD ? 1 : 0) +
        (HP_INFOBOX_MOD ? 1 : 0) +
        (HP_SECLEVEL_MOD ? 1 : 0) +
        (HP_BRACE3_MOD ? 1 : 0) +
        (HP_NAMEDARG_MOD ? 1 : 0) +
        (HP_INCLUDE_MOD ? 1 : 0) +
        (HP_SIG_MOD ? 1 : 0) +
        (HP_WIKIBOLD_MOD ? 1 : 0) +
        (HP_URLPART_MOD ? 1 : 0) +
        (HP_REFIDX_MOD ? 1 : 0);
    static constexpr int kCtxModels = 11 + kExtraCtx;
    static constexpr int kMatchModels = 5 + (HP_MATCH_18 ? 1 : 0)
        + (HP_MATCH_13 ? 1 : 0) + (HP_MATCH_01 ? 1 : 0)
        + (HP_MATCH_02 ? 1 : 0) + (HP_MATCH_05 ? 1 : 0)
        + (HP_MATCH_07 ? 1 : 0) + (HP_MATCH_09 ? 1 : 0)
        + (HP_MATCH_12 ? 1 : 0) + (HP_MATCH_20 ? 1 : 0);
    static constexpr int kWordMatch = HP_WORD_MATCH
        ? (HP_WORD_MATCH_N + (HP_WMATCH_4 ? 1 : 0) + (HP_WMATCH_5 ? 1 : 0)) : 0;
    static constexpr int kCtw = HP_CTW ? 1 : 0;
    static constexpr int kDiscovered =
        DiscoveryPool::kSlots * ContextModel::kOutputs;
    static constexpr int kBaseExperts =
        kCtxModels * ContextModel::kOutputs + 1 /*bias*/ + kMatchModels
        + kWordMatch + (HP_SPARSE_UTF8 ? 1 : 0)
        + 1 /*hebb*/ + kCtw + kDiscovered
        + (HP_DMC_MOD ? 1 : 0) + (HP_LZP_MOD ? 1 : 0)
        + (HP_SR_MOD ? 1 : 0) + (HP_SKIPK_MOD ? 1 : 0)
        + (HP_SKIP3_MOD ? 1 : 0) + (HP_SKIP4_MOD ? 1 : 0)
        + (HP_SKIP5_MOD ? 1 : 0);
#if HP_HEDGE_L1
    static constexpr int kHedgeInputs = 0;
#else
    static constexpr int kHedgeInputs = 2;
#endif
    static constexpr int kNumExperts = kBaseExperts + kHedgeInputs;

    enum Gate {
        kGateC0 = 0, kGateAlpha, kGatePrev, kGateMatch, kGateEntropy,
        kGateHebb,
#if HP_EXTRA_GATES
        kGateWiki, kGatePattern,
#endif
#if HP_POS_GATE
        kGatePos,
#endif
#if HP_GATE_SHAPE
        kGateShape,
#endif
#if HP_GATE_BRANCH
        kGateBranch,
#endif
#if HP_GATE_DISP
        kGateDisp,
#endif
#if HP_GATE_MLEN2
        kGateMlen2,
#endif
#if HP_GATE_ARGMAX
        kGateArgmax,
#endif
#if HP_SEN_GROUP
        kGateSenGroup,
#endif
#if HP_GATE_BREAK
        kGateBreak,
#endif
#if HP_GATE_WORDPOS
        kGateWordPos,
#endif
#if HP_GATE_HEDGE
        kGateHedge,
#endif
#if HP_GATE_FWORD
        kGateFword,
#endif
#if HP_GATE_UTF8
        kGateUtf8,
#endif
#if HP_GATE_NEST
        kGateNest,
#endif
#if HP_GATE_AGREE
        kGateAgree,
#endif
#if HP_GATE_FCLASS
        kGateFclass,
#endif
#if HP_GATE_WMLEN
        kGateWmLen,
#endif
        kNumGates
    };

    static int slot_bits(int base, int delta) {
#if HP_SLOT_GROW
        if (delta < 0) delta = 0;
        int b = base + delta;
#if HP_SLOT_GROW_EXTRA
        if (delta >= 2) b += 1;
#endif
#if HP_SLOT_WORD2
        if (delta == 1) b += 1;
#endif
#if HP_SLOT_WORD3
        if (delta == 1) b += 1;
#endif
#if HP_SLOT_WORD4
        if (delta == 1) b += 1;
#endif
#if HP_SLOT_WORD5
        if (delta == 1) b += 1;
#endif
#if HP_SLOT_WORD6
        if (delta == 1) b += 1;
#endif
#if HP_SLOT_WORD7
        if (delta == 1) b += 1;
#endif
#if HP_SLOT_WORD8
        if (delta == 1) b += 1;
#endif
#if HP_SLOT_WORD9
        if (delta == 1) b += 1;
#endif
#if HP_SLOT_WORD10
        if (delta == 1) b += 1;
#endif
#elif HP_SLOT_SIZES
        int b = base + delta;
#else
        int b = base;
        (void)delta;
#endif
        if (b < 16) b = 16;
        if (b > HP_SLOT_MAX) b = HP_SLOT_MAX;
        return b;
    }

    static int match_bits(int b) {
        b += (HP_MATCH_GROW ? 1 : 0) + (HP_MATCH_GROW2 ? 1 : 0);
        if (b < 16) b = 16;
        if (b > 28) b = 28;
        return b;
    }

    static int add_bits(int b, int extra) {
        b += extra;
        if (b < 16) b = 16;
        if (b > HP_SLOT_MAX) b = HP_SLOT_MAX;
        return b;
    }

    explicit Predictor(const Config& cfg)
        : byte_ring_(cfg.buf_bits),
          o1_(add_bits(slot_bits(cfg.table_bits, -2), HP_SLOT_O12 ? 1 : 0), 1023),
          o2_(add_bits(slot_bits(cfg.table_bits, -2), HP_SLOT_O12 ? 1 : 0), 1023),
          o3_(add_bits(slot_bits(cfg.table_bits, 0),
                       (HP_SLOT_O34 ? 1 : 0) + (HP_SLOT_O34B ? 1 : 0)
                       + (HP_SLOT_O34C ? 1 : 0) + (HP_SLOT_O34D ? 1 : 0)
                       + (HP_SLOT_O34E ? 1 : 0) + (HP_SLOT_O34F ? 1 : 0)
                       + (HP_SLOT_O34G ? 1 : 0)), 511),
          o4_(add_bits(slot_bits(cfg.table_bits, 0),
                       (HP_SLOT_O34 ? 1 : 0) + (HP_SLOT_O34B ? 1 : 0)
                       + (HP_SLOT_O34C ? 1 : 0) + (HP_SLOT_O34D ? 1 : 0)
                       + (HP_SLOT_O34E ? 1 : 0) + (HP_SLOT_O34F ? 1 : 0)
                       + (HP_SLOT_O34G ? 1 : 0)), 255),
          o6_(add_bits(slot_bits(cfg.table_bits, 0),
                       (HP_SLOT_O6 ? 1 : 0) + (HP_SLOT_O6B ? 1 : 0)
                       + (HP_SLOT_O6C ? 1 : 0) + (HP_SLOT_O6D ? 1 : 0)
                       + (HP_SLOT_O6E ? 1 : 0) + (HP_SLOT_O6F ? 1 : 0)
                       + (HP_SLOT_O6G ? 1 : 0) + (HP_SLOT_O6H ? 1 : 0)
                       + (HP_SLOT_O6I ? 1 : 0) + (HP_SLOT_O6J ? 1 : 0)), 127),
#if HP_HASH2_O6
          o6b_(add_bits(slot_bits(cfg.table_bits, 0),
                       (HP_SLOT_O6 ? 1 : 0) + (HP_SLOT_O6B ? 1 : 0)
                       + (HP_SLOT_O6C ? 1 : 0) + (HP_SLOT_O6D ? 1 : 0)
                       + (HP_SLOT_O6E ? 1 : 0) + (HP_SLOT_O6F ? 1 : 0)
                       + (HP_SLOT_O6G ? 1 : 0) + (HP_SLOT_O6H ? 1 : 0)
                       + (HP_SLOT_O6I ? 1 : 0) + (HP_SLOT_O6J ? 1 : 0)), 127),
#endif
          word_(slot_bits(cfg.table_bits, 1), 255),
          col_(add_bits(slot_bits(cfg.table_bits, HP_SLOT_COL2 ? 2 : 0),
                       HP_SLOT_COL3 ? 1 : 0), 255),
          tag_(slot_bits(cfg.table_bits, 2), 255),
          wbi_(slot_bits(cfg.table_bits, 1), 255),
          sp13_(add_bits(slot_bits(cfg.table_bits, 0), HP_SLOT_SP ? 1 : 0), 255),
          sp24_(add_bits(slot_bits(cfg.table_bits, 0), HP_SLOT_SP ? 1 : 0), 255),
#if HP_WORD_STREAMS
          wstr_sp_(slot_bits(cfg.table_bits, HP_SLOT_WSTR2 ? 2 : 0), 255),
#endif
#if HP_BRACKET
          brk_(slot_bits(cfg.table_bits, HP_SLOT_BRK2 ? 2 : 0), 255),
#endif
#if HP_LINKWORD
          link_(slot_bits(cfg.table_bits, 2), 255),
#endif
#if HP_NUMERIC
          num_(slot_bits(cfg.table_bits, HP_SLOT_NUM2 ? 2 : 0), 255),
#endif
#if HP_PAT_MODEL
          pat_(cfg.table_bits, 255),
#endif
#if HP_PPMD
          ppm_(cfg.table_bits, 127),
#endif
#if HP_STEMMER
          stem0_(cfg.table_bits, 255),
#if HP_STEMMER_N >= 2
          stem1_(cfg.table_bits, 255),
#endif
#endif
#if HP_SENWORD
          sen_(slot_bits(cfg.table_bits, 2), 255),
#endif
#if HP_SENT_STREAM
          sentst_(slot_bits(cfg.table_bits, HP_SLOT_S3 ? (HP_SLOT_S4 ? 3 : 2) : 0), 255),
#endif
#if HP_SENT_MEM
          sentmem_cm_(slot_bits(cfg.table_bits, HP_SLOT_SMEM ? 2 : 0), 255),
#endif
#if HP_SENGRP_MOD
          sengrp_(slot_bits(cfg.table_bits, HP_SLOT_SGRP ? 2 : 0), 255),
#endif
#if HP_NEST_MOD
          nestmod_(cfg.table_bits, 255),
#endif
#if HP_PARA_MOD
          paramod_(cfg.table_bits, 255),
#endif
#if HP_LINE_MOD
          linemod_(cfg.table_bits, 255),
#endif
#if HP_STATE_MOD
          statemod_(cfg.table_bits, 255),
#endif
#if HP_DOM_MOD
          dommod_(cfg.table_bits, 255),
#endif
#if HP_HDR_MOD
          hdrmod_(cfg.table_bits, 255),
#endif
#if HP_DEPTH_MOD
          depthmod_(cfg.table_bits, 255),
#endif
#if HP_FCCXT_MOD
          fccxtmod_(cfg.table_bits, 255),
#endif
#if HP_TPLNAME_MOD
          tplmod_(cfg.table_bits, 255),
#endif
#if HP_INFOKEY_MOD
          infokeymod_(cfg.table_bits, 255),
#endif
#if HP_BARIDX_MOD
          baridxmod_(cfg.table_bits, 255),
#endif
#if HP_PERIOD_MOD
          periodmod_(cfg.table_bits, 255),
#endif
#if HP_PRONOUN_MOD
          pronounmod_(cfg.table_bits, 255),
#endif
#if HP_LINKPIPE_MOD
          linkpipemod_(cfg.table_bits, 255),
#endif
#if HP_CITE_MOD
          citemod_(cfg.table_bits, 255),
#endif
#if HP_CAT_MOD
          catmod_(cfg.table_bits, 255),
#endif
#if HP_REDIR_MOD
          redirmod_(cfg.table_bits, 255),
#endif
#if HP_HEADING_MOD
          headingmod_(cfg.table_bits, 255),
#endif
#if HP_EXTLINK_MOD
          extlinkmod_(cfg.table_bits, 255),
#endif
#if HP_REFNAME_MOD
          refnamemod_(cfg.table_bits, 255),
#endif
#if HP_QOCXT_MOD
          qocxtmod_(cfg.table_bits, 255),
#endif
#if HP_ENTITY_MOD
          entitymod_(cfg.table_bits, 255),
#endif
#if HP_INDENT_MOD
          indentmod_(cfg.table_bits, 255),
#endif
#if HP_LISTLEVEL_MOD
          listlevelmod_(cfg.table_bits, 255),
#endif
#if HP_ISSE_MOD
          issemod_(cfg.table_bits, 255),
#endif
#if HP_MAGIC_MOD
          magicmod_(cfg.table_bits, 255),
#endif
#if HP_NOWIKI_MOD
          nowikimod_(cfg.table_bits, 255),
#endif
#if HP_TITLE_MOD
          titlemod_(cfg.table_bits, 255),
#endif
#if HP_PAGEID_MOD
          pageidmod_(cfg.table_bits, 255),
#endif
#if HP_USER_MOD
          usermod_(cfg.table_bits, 255),
#endif
#if HP_TEXT_MOD
          textmod_(cfg.table_bits, 255),
#endif
#if HP_NS_MOD
          nsmod_(cfg.table_bits, 255),
#endif
#if HP_DUMPREDIR_MOD
          dumpredirmod_(cfg.table_bits, 255),
#endif
#if HP_IP_MOD
          ipmod_(cfg.table_bits, 255),
#endif
#if HP_REVCOMMENT_MOD
          revcommentmod_(cfg.table_bits, 255),
#endif
#if HP_MINOR_MOD
          minormod_(cfg.table_bits, 255),
#endif
#if HP_WIKIMODEL_MOD
          wikimodelmod_(cfg.table_bits, 255),
#endif
#if HP_SECTITLE_MOD
          sectitlemod_(cfg.table_bits, 255),
#endif
#if HP_PARSERFN_MOD
          parserfnmod_(cfg.table_bits, 255),
#endif
#if HP_TABLECLASS_MOD
          tableclassmod_(cfg.table_bits, 255),
#endif
#if HP_ANCHOR_MOD
          anchormod_(cfg.table_bits, 255),
#endif
#if HP_PUBID_MOD
          pubidmod_(cfg.table_bits, 255),
#endif
#if HP_TEMPPOS_MOD
          tempposmod_(cfg.table_bits, 255),
#endif
#if HP_WIKISTACK_MOD
          wikistackmod_(cfg.table_bits, 255),
#endif
#if HP_LANG_MOD
          langmod_(cfg.table_bits, 255),
#endif
#if HP_CATSORT_MOD
          catsortmod_(cfg.table_bits, 255),
#endif
#if HP_TBLROW_MOD
          tblrowmod_(cfg.table_bits, 255),
#endif
#if HP_FILEOPT_MOD
          fileoptmod_(cfg.table_bits, 255),
#endif
#if HP_DEFAULTSORT_MOD
          defaultsortmod_(cfg.table_bits, 255),
#endif
#if HP_REDIRTARGET_MOD
          redirtargetmod_(cfg.table_bits, 255),
#endif
#if HP_DAB_MOD
          dabmod_(cfg.table_bits, 255),
#endif
#if HP_HATNOTE_MOD
          hatnotemod_(cfg.table_bits, 255),
#endif
#if HP_LASTLINK_MOD
          lastlinkmod_(cfg.table_bits, 255),
#endif
#if HP_FWORD_MOD
          fwordmod_(cfg.table_bits, 255),
#endif
#if HP_YEAR_MOD
          yearmod_(cfg.table_bits, 255),
#endif
#if HP_CAPMASK_MOD
          capmaskmod_(cfg.table_bits, 255),
#endif
#if HP_CELLTXT_MOD
          celltxtmod_(cfg.table_bits, 255),
#endif
#if HP_HTTPHOST_MOD
          httphostmod_(cfg.table_bits, 255),
#endif
#if HP_PAREN_MOD
          parenmod_(cfg.table_bits, 255),
#endif
#if HP_LISTPOS_MOD
          listposmod_(cfg.table_bits, 255),
#endif
#if HP_SHAPE_MOD
          shapemod_(cfg.table_bits, 255),
#endif
#if HP_SUFFIX_MOD
          suffixmod_(cfg.table_bits, 255),
#endif
#if HP_PREFIX_MOD
          prefixmod_(cfg.table_bits, 255),
#endif
#if HP_CHARCLS_MOD
          charclsmod_(cfg.table_bits, 255),
#endif
#if HP_VOWEL_MOD
          vowelmod_(cfg.table_bits, 255),
#endif
#if HP_CONTR_MOD
          contrmod_(cfg.table_bits, 255),
#endif
#if HP_HYPHEN_MOD
          hyphenmod_(cfg.table_bits, 255),
#endif
#if HP_TOKENCLS_MOD
          tokenclsmod_(cfg.table_bits, 255),
#endif
#if HP_RUNLEN_MOD
          runlenmod_(cfg.table_bits, 255),
#endif
#if HP_WPOS_MOD
          wposmod_(cfg.table_bits, 255),
#endif
#if HP_BLANK_MOD
          blankmod_(cfg.table_bits, 255),
#endif
#if HP_SPRUN_MOD
          sprunmod_(cfg.table_bits, 255),
#endif
#if HP_LINELEN_MOD
          linelenmod_(cfg.table_bits, 255),
#endif
#if HP_TAGDIST_MOD
          tagdistmod_(cfg.table_bits, 255),
#endif
#if HP_MARKDIST_MOD
          markdistmod_(cfg.table_bits, 255),
#endif
#if HP_UPPERGAP_MOD
          uppergapmod_(cfg.table_bits, 255),
#endif
#if HP_MONTH_MOD
          monthmod_(cfg.table_bits, 255),
#endif
#if HP_GALLERY_MOD
          gallerymod_(cfg.table_bits, 255),
#endif
#if HP_SECKIND_MOD
          seckindmod_(cfg.table_bits, 255),
#endif
#if HP_CITEKIND_MOD
          citekindmod_(cfg.table_bits, 255),
#endif
#if HP_TAGNAME_MOD
          tagnamemod_(cfg.table_bits, 255),
#endif
#if HP_COLSPAN_MOD
          colspanmod_(cfg.table_bits, 255),
#endif
#if HP_STYLE_MOD
          stylemod_(cfg.table_bits, 255),
#endif
#if HP_COORD_MOD
          coordmod_(cfg.table_bits, 255),
#endif
#if HP_DIGITGAP_MOD
          digitgapmod_(cfg.table_bits, 255),
#endif
#if HP_DOTGAP_MOD
          dotgapmod_(cfg.table_bits, 255),
#endif
#if HP_COMMAGAP_MOD
          commagapmod_(cfg.table_bits, 255),
#endif
#if HP_WORDLEN_MOD
          wordlenmod_(cfg.table_bits, 255),
#endif
#if HP_SENTLEN_MOD
          sentlenmod_(cfg.table_bits, 255),
#endif
#if HP_LOWERGAP_MOD
          lowergapmod_(cfg.table_bits, 255),
#endif
#if HP_DIGITPOS_MOD
          digitposmod_(cfg.table_bits, 255),
#endif
#if HP_SLASHGAP_MOD
          slashgapmod_(cfg.table_bits, 255),
#endif
#if HP_DIGLEN_MOD
          diglenmod_(cfg.table_bits, 255),
#endif
#if HP_PREVLINE_MOD
          prevlinemod_(cfg.table_bits, 255),
#endif
#if HP_PREVSENT_MOD
          prevsentmod_(cfg.table_bits, 255),
#endif
#if HP_LINKLEN_MOD
          linklenmod_(cfg.table_bits, 255),
#endif
#if HP_TPLLEN_MOD
          tpllenmod_(cfg.table_bits, 255),
#endif
#if HP_PARALEN_MOD
          paralenmod_(cfg.table_bits, 255),
#endif
#if HP_ALNUMLEN_MOD
          alnumlenmod_(cfg.table_bits, 255),
#endif
#if HP_SPLEN_MOD
          splenmod_(cfg.table_bits, 255),
#endif
#if HP_TITLEWORD_MOD
          titlewordmod_(cfg.table_bits, 255),
#endif
#if HP_HEADWORD_MOD
          headwordmod_(cfg.table_bits, 255),
#endif
#if HP_INIT_MOD
          initmod_(cfg.table_bits, 255),
#endif
#if HP_ORDINAL_MOD
          ordinalmod_(cfg.table_bits, 255),
#endif
#if HP_UNIT_MOD
          unitmod_(cfg.table_bits, 255),
#endif
#if HP_DECIMAL_MOD
          decimalmod_(cfg.table_bits, 255),
#endif
#if HP_REPEAT_MOD
          repeatmod_(cfg.table_bits, 255),
#endif
#if HP_CASEFLIP_MOD
          caseflipmod_(cfg.table_bits, 255),
#endif
#if HP_LEAD_MOD
          leadmod_(cfg.table_bits, 255),
#endif
#if HP_INFOVAL_MOD
          infovalmod_(cfg.table_bits, 255),
#endif
#if HP_LINKTRAIL_MOD
          linktrailmod_(cfg.table_bits, 255),
#endif
#if HP_CELLKIND_MOD
          cellkindmod_(cfg.table_bits, 255),
#endif
#if HP_TBLCOL_MOD
          tblcolmod_(cfg.table_bits, 255),
#endif
#if HP_HEADIDX_MOD
          headidxmod_(cfg.table_bits, 255),
#endif
#if HP_HTMLFMT_MOD
          htmlfmtmod_(cfg.table_bits, 255),
#endif
#if HP_INFOBOX_MOD
          infoboxmod_(cfg.table_bits, 255),
#endif
#if HP_SECLEVEL_MOD
          seclevelmod_(cfg.table_bits, 255),
#endif
#if HP_BRACE3_MOD
          brace3mod_(cfg.table_bits, 255),
#endif
#if HP_NAMEDARG_MOD
          namedargmod_(cfg.table_bits, 255),
#endif
#if HP_INCLUDE_MOD
          includemod_(cfg.table_bits, 255),
#endif
#if HP_SIG_MOD
          sigmod_(cfg.table_bits, 255),
#endif
#if HP_WIKIBOLD_MOD
          wikiboldmod_(cfg.table_bits, 255),
#endif
#if HP_URLPART_MOD
          urlpartmod_(cfg.table_bits, 255),
#endif
#if HP_REFIDX_MOD
          refidxmod_(cfg.table_bits, 255),
#endif
          match_{ {&byte_ring_, match_bits(cfg.match_bits), 3},
                  {&byte_ring_, match_bits(cfg.match_bits), 4},
                  {&byte_ring_, match_bits(cfg.match_bits), 6},
                  {&byte_ring_, match_bits(cfg.match_bits), 10},
                  {&byte_ring_, match_bits(cfg.match_bits), 16}
#if HP_MATCH_18
                  , {&byte_ring_, match_bits(cfg.match_bits), 8}
#endif
#if HP_MATCH_13
                  , {&byte_ring_, match_bits(cfg.match_bits), 13}
#endif
#if HP_MATCH_01
                  , {&byte_ring_, match_bits(cfg.match_bits), 1}
#endif
#if HP_MATCH_02
                  , {&byte_ring_, match_bits(cfg.match_bits), 2}
#endif
#if HP_MATCH_05
                  , {&byte_ring_, match_bits(cfg.match_bits), 5}
#endif
#if HP_MATCH_07
                  , {&byte_ring_, match_bits(cfg.match_bits), 7}
#endif
#if HP_MATCH_09
                  , {&byte_ring_, match_bits(cfg.match_bits), 9}
#endif
#if HP_MATCH_12
                  , {&byte_ring_, match_bits(cfg.match_bits), 12}
#endif
#if HP_MATCH_20
                  , {&byte_ring_, match_bits(cfg.match_bits), 20}
#endif
          },
#if HP_SPARSE_UTF8
          smatch_(&byte_ring_, match_bits(cfg.match_bits), 4),
#endif
#if HP_SKIPK_MOD
          skipk_(&byte_ring_, match_bits(cfg.match_bits), 3, 2),
#endif
#if HP_SKIP3_MOD
          skip3_(&byte_ring_, match_bits(cfg.match_bits), 3, 3),
#endif
#if HP_SKIP4_MOD
          skip4_(&byte_ring_, match_bits(cfg.match_bits), 3, 4),
#endif
#if HP_SKIP5_MOD
          skip5_(&byte_ring_, match_bits(cfg.match_bits), 3, 5),
#endif
#if HP_LZP_MOD
          lzp_(match_bits(cfg.match_bits) > 2 ? match_bits(cfg.match_bits) - 2
                                              : match_bits(cfg.match_bits)),
#endif
#if HP_DMC_MOD
          dmc_(HP_DMC_GROW ? 20 : 18),
#endif
#if HP_WORD_MATCH
          wmatch_{
              WordMatchModel(&byte_ring_,
                             cfg.match_bits > 2 ? cfg.match_bits - 2 : cfg.match_bits, 1,
                             cfg.buf_bits > 2 ? cfg.buf_bits - 2 : cfg.buf_bits),
              WordMatchModel(&byte_ring_,
                             cfg.match_bits > 2 ? cfg.match_bits - 2 : cfg.match_bits, 2,
                             cfg.buf_bits > 2 ? cfg.buf_bits - 2 : cfg.buf_bits),
              WordMatchModel(&byte_ring_,
                             cfg.match_bits > 2 ? cfg.match_bits - 2 : cfg.match_bits, 3,
                             cfg.buf_bits > 2 ? cfg.buf_bits - 2 : cfg.buf_bits)
#if HP_WMATCH_4
              , WordMatchModel(&byte_ring_,
                               cfg.match_bits > 2 ? cfg.match_bits - 2 : cfg.match_bits, 3,
                               cfg.buf_bits > 2 ? cfg.buf_bits - 2 : cfg.buf_bits)
#endif
#if HP_WMATCH_5
              , WordMatchModel(&byte_ring_,
                               cfg.match_bits > 2 ? cfg.match_bits - 2 : cfg.match_bits, 1,
                               cfg.buf_bits > 2 ? cfg.buf_bits - 2 : cfg.buf_bits)
#endif
          },
#endif
          hebb_(cfg.table_bits, 255),
          pool_(cfg.table_bits, 0xC0FFEEull),
          mixer_(kNumExperts, gate_sizes(), 256, cfg.mixer_lr, gate_rates(cfg.mixer_lr)),
          apm_c0_(256),
          apm_lex_(256 * 256),
          apm_gria_(GriaGate::kBuckets * 256),
#if HP_HEDGE_L1
          hedge_(kNumGates),
#else
          hedge_(kBaseExperts),
#endif
          bias_() {
        counter_init(bias_.data(), bias_.size());
#if HP_ENGLISH_PRIOR
        for (int i = 0; i < 256; ++i) {
            bias_[static_cast<std::size_t>(i)].p =
                static_cast<std::uint16_t>(english_bit_prior16(i));
        }
#endif
        gria_.set_enabled(cfg.gria);
        init_ctx_chain_();
        set_byte_contexts();
    }

    /// Deep copy with pointer rebind (``ctx_chain_``, match rings). Requires ``cfg`` used at
    /// construction because ``Predictor`` has no default constructor.
    static Predictor clone_from(const Predictor& o, const Config& cfg) {
        Predictor p(cfg);
        p.assign_from_(o);
        return p;
    }

    Predictor& operator=(const Predictor& o) {
        if (this == &o) return *this;
        assign_from_(o);
        return *this;
    }

    Predictor(Predictor&&) noexcept = default;
    Predictor& operator=(Predictor&&) noexcept = default;

    int predict() {
        mixer_.reset_inputs();
        const int bias_p = counter_predict_p(bias_[c0_]);
        mixer_.add(stretch(bias_p));
#if HP_TRACK_EXP_P
        n_exp_ = 0;
#endif

        int out[ContextModel::kOutputs];
        int backoff = bias_p;
        int mlen = 0;
        int wml = 0;
#if HP_GATE_MLEN2
        int l0 = 0, l1 = 0;
#endif
        for (int i = 0; i < n_ctx_chain_; ++i) {
            ctx_chain_[i]->predict(c0_, backoff, out);
            for (int j = 0; j < ContextModel::kOutputs; ++j) {
                mixer_.add(out[j]);
#if HP_TRACK_EXP_P
                exp_p_[n_exp_++] = squash(out[j]);
#endif
            }
            if (i < 4) backoff = ctx_chain_[i]->last_p();
            else if (i == 4) backoff = o1_.last_p();
        }
        for (int i = 0; i < kMatchModels; ++i) {
            const int ms = match_[i].predict(c0_, bitpos_);
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
            const int l = match_[i].match_len();
            if (l > mlen) mlen = l;
#if HP_GATE_MLEN2
            if (l > l0) { l1 = l0; l0 = l; }
            else if (l > l1) l1 = l;
#endif
        }
#if HP_SPARSE_UTF8
        {
            const int ms = smatch_.predict(c0_, bitpos_);
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
        }
#endif
#if HP_SKIPK_MOD
        {
            const int ms = skipk_.predict(c0_, bitpos_);
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
        }
#endif
#if HP_SKIP3_MOD
        {
            const int ms = skip3_.predict(c0_, bitpos_);
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
        }
#endif
#if HP_SKIP4_MOD
        {
            const int ms = skip4_.predict(c0_, bitpos_);
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
        }
#endif
#if HP_SKIP5_MOD
        {
            const int ms = skip5_.predict(c0_, bitpos_);
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
        }
#endif
#if HP_LZP_MOD
        {
            const int ms = lzp_.predict(c0_, bitpos_);
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
        }
#endif
#if HP_SR_MOD
        {
            const int ms = sr_.predict(c0_);
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
        }
#endif
#if HP_DMC_MOD
        {
            const int ms = dmc_.predict();
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
        }
#endif
#if HP_WORD_MATCH
        for (int i = 0; i < kWordMatch; ++i) {
            const int ms = wmatch_[i].predict(c0_, bitpos_);
            mixer_.add(ms);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(ms);
#endif
            const int l = wmatch_[i].match_len();
            if (l > mlen) mlen = l;
            if (l > wml) wml = l;
        }
#endif
        {
            const int hs = hebb_.predict(c0_);
            mixer_.add(hs);
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = squash(hs);
#endif
        }
#if HP_CTW
        {
            // Recursive KT weighting over the contiguous order chain.
            // P_o = (KT_o + P_{o-1}) / 2, which is CTW with β = 1/2.
            int p = backoff_kt_(bias_[c0_]);
            ContextModel* ord[5] = {&o1_, &o2_, &o3_, &o4_, &o6_};
            for (int i = 0; i < 5; ++i) {
                const int kt = py_estimate(ord[i]->n0(), ord[i]->n1(), p);
                p = (kt + p) >> 1;
            }
            mixer_.add(stretch(p));
#if HP_TRACK_EXP_P
            exp_p_[n_exp_++] = p;
#endif
        }
#endif
        {
            int dout[kDiscovered];
            pool_.predict(c0_, o1_.last_p(), dout);
            for (int i = 0; i < kDiscovered; ++i) {
                mixer_.add(dout[i]);
#if HP_TRACK_EXP_P
                exp_p_[n_exp_++] = squash(dout[i]);
#endif
            }
        }
        sparse_ = o6_.sparsity();

#if !HP_HEDGE_L1
        for (int e = 0; e < n_exp_ && e < kBaseExperts; ++e) hedge_.set(e, exp_p_[e]);
        const int ph = hedge_.mix();
        mixer_.add(stretch(ph));
        int wmax = 0;
        for (int e = 0; e < kBaseExperts; ++e) wmax = std::max(wmax, hedge_.weight(e));
        mixer_.add(clamp_int((wmax * kBaseExperts - 65536) >> 5, -2047, 2047));
#endif

        mixer_.set_ctx(kGateC0, c0_);
        mixer_.set_ctx(kGateAlpha, gria_.bucket());
        mixer_.set_ctx(kGatePrev, static_cast<int>(hist_ & 0xff));
        mixer_.set_ctx(kGateMatch, mlen > 31 ? 31 : mlen);
        last_mlen_ = mlen;
        mixer_.set_ctx(kGateHebb, hebb_.strength() > 15 ? 15 : hebb_.strength());
        mixer_.set_ctx(kGateEntropy, gria_.entropy_bucket());
#if HP_EXTRA_GATES
        mixer_.set_ctx(kGateWiki, wiki_.state() + (wiki_.is_paragraph() << 4));
        mixer_.set_ctx(kGatePattern, cache_.cls());
#endif
#if HP_POS_GATE
        mixer_.set_ctx(kGatePos, static_cast<int>(stems_.type() & 31u));
#endif
#if HP_GATE_SHAPE
        mixer_.set_ctx(kGateShape, shape6_bin(o6_.n0(), o6_.n1(), o6_.last_p()));
#endif
#if HP_GATE_DISP
        mixer_.set_ctx(kGateDisp, disp_var_bin(exp_p_, n_exp_));
#endif
#if HP_GATE_MLEN2
        mixer_.set_ctx(kGateMlen2, mlen2_bin(l0, l1));
#endif
#if HP_GATE_ARGMAX
        mixer_.set_ctx(kGateArgmax, argmax_bin(exp_p_, n_exp_));
#endif
#if HP_GATE_BRANCH
        mixer_.set_ctx(kGateBranch, branch3_.bin());
#endif
#if HP_SEN_GROUP
        mixer_.set_ctx(kGateSenGroup, wiki_.sen_group());
#endif
#if HP_GATE_BREAK
        mixer_.set_ctx(kGateBreak, qlog_u32(static_cast<std::uint32_t>(break_age_ + 1), 16));
#endif
#if HP_GATE_WORDPOS
        mixer_.set_ctx(kGateWordPos, streams_.word_len() > 15 ? 15 : streams_.word_len());
#endif
#if HP_GATE_HEDGE
        {
            int wmax = 0;
            for (int d = 0; d < hedge_.size(); ++d)
                wmax = std::max(wmax, hedge_.weight(d));
            mixer_.set_ctx(kGateHedge, wmax >> 12 > 15 ? 15 : (wmax >> 12));
        }
#endif
#if HP_GATE_FWORD
        mixer_.set_ctx(kGateFword, streams_.first_word_bin());
#endif
#if HP_GATE_UTF8
        mixer_.set_ctx(kGateUtf8, utf8left_ > 3 ? 3 : utf8left_);
#endif
#if HP_GATE_NEST
        mixer_.set_ctx(kGateNest, wiki_.nest_markup() ? 1 : 0);
#endif
#if HP_GATE_AGREE
        mixer_.set_ctx(kGateAgree, agree_bin(exp_p_, n_exp_));
#endif
#if HP_GATE_FCLASS
        mixer_.set_ctx(kGateFclass, streams_.first_class());
#endif
#if HP_GATE_WMLEN
        mixer_.set_ctx(kGateWmLen, wml > 15 ? 15 : wml);
#endif
        mixer_.set_ctx2(c0_);
        int pr = mixer_.mix();

#if HP_HEDGE_L1
        for (int j = 0; j < mixer_.num_layer1(); ++j)
            hedge_.set(j, mixer_.layer1_p(j));
        const int ph = hedge_.mix();
        pr = (pr + ph) >> 1;
#endif
        mixed_p_ = pr;

        const int a = apm_c0_.refine(pr, c0_);
        const int b = apm_lex_.refine(pr, static_cast<int>(hist_ & 0xff) * 256 + c0_);
        const int g = apm_gria_.refine(pr, gria_.bucket() * 256 + c0_);
        pr_final_ = clamp_int((HP_W0 * pr + HP_WA * a + HP_WB * b + HP_WG * g) >> 3, 1, 4094);
        return pr_final_;
    }

    void update(int y) {
        gria_.account_bit(y ? pr_final_ : 4096 - pr_final_);

        mixer_.update(y);
        hedge_.update(y, gria_.switch_rate_q16());
        apm_c0_.update(y);
        apm_lex_.update(y);
        apm_gria_.update(y);

        counter_update(bias_[c0_], y, 1023);
        const int ens = pr_final_;
        o1_.update(y, ens); o2_.update(y, ens); o3_.update(y, ens);
        o4_.update(y, ens); o6_.update(y, ens); word_.update(y, ens);
        sp13_.update(y, ens); sp24_.update(y, ens);
        col_.update(y, ens); tag_.update(y, ens); wbi_.update(y, ens);
#if HP_WORD_STREAMS
        wstr_sp_.update(y, ens);
#endif
#if HP_BRACKET
        brk_.update(y, ens);
#endif
#if HP_LINKWORD
        link_.update(y, ens);
#endif
#if HP_NUMERIC
        num_.update(y, ens);
#endif
#if HP_PAT_MODEL
        pat_.update(y, ens);
#endif
#if HP_PPMD
        ppm_.update(y, ens);
#endif
#if HP_STEMMER
        stem0_.update(y, ens);
#if HP_STEMMER_N >= 2
        stem1_.update(y, ens);
#endif
#endif
#if HP_SENWORD
        sen_.update(y, ens);
#endif
#if HP_SENT_STREAM
        sentst_.update(y, ens);
#endif
#if HP_SENT_MEM
        sentmem_cm_.update(y, ens);
#endif
#if HP_SENGRP_MOD
        sengrp_.update(y, ens);
#endif
#if HP_NEST_MOD
        nestmod_.update(y, ens);
#endif
#if HP_PARA_MOD
        paramod_.update(y, ens);
#endif
#if HP_LINE_MOD
        linemod_.update(y, ens);
#endif
#if HP_STATE_MOD
        statemod_.update(y, ens);
#endif
#if HP_DOM_MOD
        dommod_.update(y, ens);
#endif
#if HP_HDR_MOD
        hdrmod_.update(y, ens);
#endif
#if HP_DEPTH_MOD
        depthmod_.update(y, ens);
#endif
#if HP_FCCXT_MOD
        fccxtmod_.update(y, ens);
#endif
#if HP_TPLNAME_MOD
        tplmod_.update(y, ens);
#endif
#if HP_INFOKEY_MOD
        infokeymod_.update(y, ens);
#endif
#if HP_BARIDX_MOD
        baridxmod_.update(y, ens);
#endif
#if HP_PERIOD_MOD
        periodmod_.update(y, ens);
#endif
#if HP_PRONOUN_MOD
        pronounmod_.update(y, ens);
#endif
#if HP_HASH2_O6
        o6b_.update(y, ens);
#endif
#if HP_LINKPIPE_MOD
        linkpipemod_.update(y, ens);
#endif
#if HP_CITE_MOD
        citemod_.update(y, ens);
#endif
#if HP_CAT_MOD
        catmod_.update(y, ens);
#endif
#if HP_REDIR_MOD
        redirmod_.update(y, ens);
#endif
#if HP_HEADING_MOD
        headingmod_.update(y, ens);
#endif
#if HP_EXTLINK_MOD
        extlinkmod_.update(y, ens);
#endif
#if HP_REFNAME_MOD
        refnamemod_.update(y, ens);
#endif
#if HP_QOCXT_MOD
        qocxtmod_.update(y, ens);
#endif
#if HP_ENTITY_MOD
        entitymod_.update(y, ens);
#endif
#if HP_INDENT_MOD
        indentmod_.update(y, ens);
#endif
#if HP_LISTLEVEL_MOD
        listlevelmod_.update(y, ens);
#endif
#if HP_ISSE_MOD
        issemod_.update(y, ens);
#endif
#if HP_MAGIC_MOD
        magicmod_.update(y, ens);
#endif
#if HP_NOWIKI_MOD
        nowikimod_.update(y, ens);
#endif
#if HP_TITLE_MOD
        titlemod_.update(y, ens);
#endif
#if HP_PAGEID_MOD
        pageidmod_.update(y, ens);
#endif
#if HP_USER_MOD
        usermod_.update(y, ens);
#endif
#if HP_TEXT_MOD
        textmod_.update(y, ens);
#endif
#if HP_NS_MOD
        nsmod_.update(y, ens);
#endif
#if HP_DUMPREDIR_MOD
        dumpredirmod_.update(y, ens);
#endif
#if HP_IP_MOD
        ipmod_.update(y, ens);
#endif
#if HP_REVCOMMENT_MOD
        revcommentmod_.update(y, ens);
#endif
#if HP_MINOR_MOD
        minormod_.update(y, ens);
#endif
#if HP_WIKIMODEL_MOD
        wikimodelmod_.update(y, ens);
#endif
#if HP_SECTITLE_MOD
        sectitlemod_.update(y, ens);
#endif
#if HP_PARSERFN_MOD
        parserfnmod_.update(y, ens);
#endif
#if HP_TABLECLASS_MOD
        tableclassmod_.update(y, ens);
#endif
#if HP_ANCHOR_MOD
        anchormod_.update(y, ens);
#endif
#if HP_PUBID_MOD
        pubidmod_.update(y, ens);
#endif
#if HP_TEMPPOS_MOD
        tempposmod_.update(y, ens);
#endif
#if HP_WIKISTACK_MOD
        wikistackmod_.update(y, ens);
#endif
#if HP_LANG_MOD
        langmod_.update(y, ens);
#endif
#if HP_CATSORT_MOD
        catsortmod_.update(y, ens);
#endif
#if HP_TBLROW_MOD
        tblrowmod_.update(y, ens);
#endif
#if HP_FILEOPT_MOD
        fileoptmod_.update(y, ens);
#endif
#if HP_DEFAULTSORT_MOD
        defaultsortmod_.update(y, ens);
#endif
#if HP_REDIRTARGET_MOD
        redirtargetmod_.update(y, ens);
#endif
#if HP_DAB_MOD
        dabmod_.update(y, ens);
#endif
#if HP_HATNOTE_MOD
        hatnotemod_.update(y, ens);
#endif
#if HP_LASTLINK_MOD
        lastlinkmod_.update(y, ens);
#endif
#if HP_FWORD_MOD
        fwordmod_.update(y, ens);
#endif
#if HP_YEAR_MOD
        yearmod_.update(y, ens);
#endif
#if HP_CAPMASK_MOD
        capmaskmod_.update(y, ens);
#endif
#if HP_CELLTXT_MOD
        celltxtmod_.update(y, ens);
#endif
#if HP_HTTPHOST_MOD
        httphostmod_.update(y, ens);
#endif
#if HP_PAREN_MOD
        parenmod_.update(y, ens);
#endif
#if HP_LISTPOS_MOD
        listposmod_.update(y, ens);
#endif
#if HP_SHAPE_MOD
        shapemod_.update(y, ens);
#endif
#if HP_SUFFIX_MOD
        suffixmod_.update(y, ens);
#endif
#if HP_PREFIX_MOD
        prefixmod_.update(y, ens);
#endif
#if HP_CHARCLS_MOD
        charclsmod_.update(y, ens);
#endif
#if HP_VOWEL_MOD
        vowelmod_.update(y, ens);
#endif
#if HP_CONTR_MOD
        contrmod_.update(y, ens);
#endif
#if HP_HYPHEN_MOD
        hyphenmod_.update(y, ens);
#endif
#if HP_TOKENCLS_MOD
        tokenclsmod_.update(y, ens);
#endif
#if HP_RUNLEN_MOD
        runlenmod_.update(y, ens);
#endif
#if HP_WPOS_MOD
        wposmod_.update(y, ens);
#endif
#if HP_BLANK_MOD
        blankmod_.update(y, ens);
#endif
#if HP_SPRUN_MOD
        sprunmod_.update(y, ens);
#endif
#if HP_LINELEN_MOD
        linelenmod_.update(y, ens);
#endif
#if HP_TAGDIST_MOD
        tagdistmod_.update(y, ens);
#endif
#if HP_MARKDIST_MOD
        markdistmod_.update(y, ens);
#endif
#if HP_UPPERGAP_MOD
        uppergapmod_.update(y, ens);
#endif
#if HP_MONTH_MOD
        monthmod_.update(y, ens);
#endif
#if HP_GALLERY_MOD
        gallerymod_.update(y, ens);
#endif
#if HP_SECKIND_MOD
        seckindmod_.update(y, ens);
#endif
#if HP_CITEKIND_MOD
        citekindmod_.update(y, ens);
#endif
#if HP_TAGNAME_MOD
        tagnamemod_.update(y, ens);
#endif
#if HP_COLSPAN_MOD
        colspanmod_.update(y, ens);
#endif
#if HP_STYLE_MOD
        stylemod_.update(y, ens);
#endif
#if HP_COORD_MOD
        coordmod_.update(y, ens);
#endif
#if HP_DIGITGAP_MOD
        digitgapmod_.update(y, ens);
#endif
#if HP_DOTGAP_MOD
        dotgapmod_.update(y, ens);
#endif
#if HP_COMMAGAP_MOD
        commagapmod_.update(y, ens);
#endif
#if HP_WORDLEN_MOD
        wordlenmod_.update(y, ens);
#endif
#if HP_SENTLEN_MOD
        sentlenmod_.update(y, ens);
#endif
#if HP_LOWERGAP_MOD
        lowergapmod_.update(y, ens);
#endif
#if HP_DIGITPOS_MOD
        digitposmod_.update(y, ens);
#endif
#if HP_SLASHGAP_MOD
        slashgapmod_.update(y, ens);
#endif
#if HP_DIGLEN_MOD
        diglenmod_.update(y, ens);
#endif
#if HP_PREVLINE_MOD
        prevlinemod_.update(y, ens);
#endif
#if HP_PREVSENT_MOD
        prevsentmod_.update(y, ens);
#endif
#if HP_LINKLEN_MOD
        linklenmod_.update(y, ens);
#endif
#if HP_TPLLEN_MOD
        tpllenmod_.update(y, ens);
#endif
#if HP_PARALEN_MOD
        paralenmod_.update(y, ens);
#endif
#if HP_ALNUMLEN_MOD
        alnumlenmod_.update(y, ens);
#endif
#if HP_SPLEN_MOD
        splenmod_.update(y, ens);
#endif
#if HP_TITLEWORD_MOD
        titlewordmod_.update(y, ens);
#endif
#if HP_HEADWORD_MOD
        headwordmod_.update(y, ens);
#endif
#if HP_INIT_MOD
        initmod_.update(y, ens);
#endif
#if HP_ORDINAL_MOD
        ordinalmod_.update(y, ens);
#endif
#if HP_UNIT_MOD
        unitmod_.update(y, ens);
#endif
#if HP_DECIMAL_MOD
        decimalmod_.update(y, ens);
#endif
#if HP_REPEAT_MOD
        repeatmod_.update(y, ens);
#endif
#if HP_CASEFLIP_MOD
        caseflipmod_.update(y, ens);
#endif
#if HP_LEAD_MOD
        leadmod_.update(y, ens);
#endif
#if HP_INFOVAL_MOD
        infovalmod_.update(y, ens);
#endif
#if HP_LINKTRAIL_MOD
        linktrailmod_.update(y, ens);
#endif
#if HP_CELLKIND_MOD
        cellkindmod_.update(y, ens);
#endif
#if HP_TBLCOL_MOD
        tblcolmod_.update(y, ens);
#endif
#if HP_HEADIDX_MOD
        headidxmod_.update(y, ens);
#endif
#if HP_HTMLFMT_MOD
        htmlfmtmod_.update(y, ens);
#endif
#if HP_INFOBOX_MOD
        infoboxmod_.update(y, ens);
#endif
#if HP_SECLEVEL_MOD
        seclevelmod_.update(y, ens);
#endif
#if HP_BRACE3_MOD
        brace3mod_.update(y, ens);
#endif
#if HP_NAMEDARG_MOD
        namedargmod_.update(y, ens);
#endif
#if HP_INCLUDE_MOD
        includemod_.update(y, ens);
#endif
#if HP_SIG_MOD
        sigmod_.update(y, ens);
#endif
#if HP_WIKIBOLD_MOD
        wikiboldmod_.update(y, ens);
#endif
#if HP_URLPART_MOD
        urlpartmod_.update(y, ens);
#endif
#if HP_REFIDX_MOD
        refidxmod_.update(y, ens);
#endif
        for (int i = 0; i < kMatchModels; ++i) match_[i].update(y);
#if HP_SPARSE_UTF8
        smatch_.update(y);
#endif
#if HP_SKIPK_MOD
        skipk_.update(y);
#endif
#if HP_SKIP3_MOD
        skip3_.update(y);
#endif
#if HP_SKIP4_MOD
        skip4_.update(y);
#endif
#if HP_SKIP5_MOD
        skip5_.update(y);
#endif
#if HP_LZP_MOD
        lzp_.update(y);
#endif
#if HP_SR_MOD
        sr_.update(y);
#endif
#if HP_DMC_MOD
        dmc_.update(y);
#endif
#if HP_WORD_MATCH
        for (int i = 0; i < kWordMatch; ++i) wmatch_[i].update(y);
#endif
        hebb_.update(y);
        pool_.update(y, y ? (4096 - pr_final_) >> 4 : pr_final_ >> 4);

        c0_ = (c0_ << 1) | y;
        ++bitpos_;
        if (bitpos_ == 8) {
            const int byte = c0_ & 0xff;
            c0_ = 1;
            bitpos_ = 0;
            end_of_byte(byte);
        }
    }

    const GriaGate& gria() const { return gria_; }
    int discovery_replacements() const { return pool_.replaced(); }
    int cache_hits() const {
#if HP_PATTERN_CACHE_STATS
        return cache_.hits();
#else
        return 0;
#endif
    }
    int cache_lookups() const {
#if HP_PATTERN_CACHE_STATS
        return cache_.lookups();
#else
        return 0;
#endif
    }
    const PatternCache& patterns() const { return cache_; }

    int expert_count() const { return n_exp_; }
    int expert_p(int i) const { return exp_p_[i]; }
    int mixed_p() const { return mixed_p_; }
    int sparse_fraction() const { return sparse_; }
    int mixer_n() const { return mixer_.num_layer1(); }
    int mixer_dot(int j) const { return mixer_.layer1_dot(j); }
    int mixer_p(int j) const { return mixer_.layer1_p(j); }
    int c0() const { return c0_; }
    int wiki_state() const { return wiki_.state(); }
    int entropy_bucket() const { return gria_.entropy_bucket(); }
    int last_match_len() const { return last_mlen_; }

 private:
    static const std::vector<int>& gate_sizes() {
        static const std::vector<int> s = [] {
        std::vector<int> out = {256, GriaGate::kBuckets, 256, 32,
                              GriaGate::kEntBuckets, 16};
#if HP_EXTRA_GATES
        out.push_back(32);   // wiki state × isParagraph
        out.push_back(PatternCache::kNClass);
#endif
#if HP_POS_GATE
        out.push_back(32);
#endif
#if HP_GATE_SHAPE
        out.push_back(16);
#endif
#if HP_GATE_BRANCH
        out.push_back(16);
#endif
#if HP_GATE_DISP
        out.push_back(16);
#endif
#if HP_GATE_MLEN2
        out.push_back(16);
#endif
#if HP_GATE_ARGMAX
        out.push_back(16);
#endif
#if HP_SEN_GROUP
        out.push_back(4);
#endif
#if HP_GATE_BREAK
        out.push_back(16);
#endif
#if HP_GATE_WORDPOS
        out.push_back(16);
#endif
#if HP_GATE_HEDGE
        out.push_back(16);
#endif
#if HP_GATE_FWORD
        out.push_back(16);
#endif
#if HP_GATE_UTF8
        out.push_back(4);
#endif
#if HP_GATE_NEST
        out.push_back(2);
#endif
#if HP_GATE_AGREE
        out.push_back(16);
#endif
#if HP_GATE_FCLASS
        out.push_back(4);
#endif
#if HP_GATE_WMLEN
        out.push_back(16);
#endif
        return out;
        }();
        return s;
    }

    static const std::vector<int>& gate_rates(int base) {
#if HP_PER_MIXER_LR
        (void)base;
        static const std::vector<int> r = [] {
        std::vector<int> out = {2, 3, 2, 4, 3, 4};
#if HP_EXTRA_GATES
        out.push_back(3);
        out.push_back(3);
#endif
#if HP_POS_GATE
        out.push_back(3);
#endif
#if HP_GATE_SHAPE
        out.push_back(3);
#endif
#if HP_GATE_BRANCH
        out.push_back(3);
#endif
#if HP_GATE_DISP
        out.push_back(3);
#endif
#if HP_GATE_MLEN2
        out.push_back(3);
#endif
#if HP_GATE_ARGMAX
        out.push_back(3);
#endif
#if HP_SEN_GROUP
        out.push_back(3);
#endif
#if HP_GATE_BREAK
        out.push_back(3);
#endif
#if HP_GATE_WORDPOS
        out.push_back(3);
#endif
#if HP_GATE_HEDGE
        out.push_back(3);
#endif
#if HP_GATE_FWORD
        out.push_back(3);
#endif
#if HP_GATE_UTF8
        out.push_back(3);
#endif
#if HP_GATE_NEST
        out.push_back(3);
#endif
#if HP_GATE_AGREE
        out.push_back(3);
#endif
#if HP_GATE_FCLASS
        out.push_back(3);
#endif
#if HP_GATE_WMLEN
        out.push_back(3);
#endif
        #if HP_LR1_SCALE != 100
        for (std::size_t q = 0; q < out.size(); ++q) {
            int v = (out[q] * HP_LR1_SCALE + 50) / 100;
            out[q] = v < 1 ? 1 : v;
        }
#endif
        return out;
        }();
        return r;
#else
        static const std::vector<int> r(static_cast<std::size_t>(kNumGates), base);
        return r;
#endif
    }

    static int backoff_kt_(const Counter& c) {
        return counter_predict_p(c);
    }

#if HP_PRED_GATE
    bool pred_gate_mute() const {
        return wiki_.nest_markup() ||
               brackets_.square_depth() > 0 ||
               brackets_.curly_depth() > 0;
    }
#endif

    std::uint32_t h2(std::uint64_t salt, std::uint64_t key) {
#if HP_PATTERN_CACHE
        return cache_.hash_memo(salt, key);
#else
        return hash2(salt, key);
#endif
    }

    void end_of_byte(int byte) {
        hist_ = (hist_ << 8) | static_cast<std::uint64_t>(byte);

        const bool alnum = (byte >= 'a' && byte <= 'z') ||
                           (byte >= 'A' && byte <= 'Z') ||
                           (byte >= '0' && byte <= '9');
        const int at_boundary = (!alnum && word_hash_ != 0) ? 1 : 0;
        const bool letter = (byte >= 'a' && byte <= 'z') ||
                            (byte >= 'A' && byte <= 'Z');
        if (alnum) {
            word_hash_ = mix64(word_hash_ * 0x100000001B3ull +
                               static_cast<std::uint64_t>(byte | 0x20));
#if HP_PRONOUN_MOD
            if (pw_n_ < 11) pw_[pw_n_++] = static_cast<std::uint8_t>(byte | 32);
#endif
        } else {
            word_hash_ = 0;
#if HP_PRONOUN_MOD
            if (pw_n_ > 0) {
                pronoun_ = pronoun_word(pw_, pw_n_);
                pw_n_ = 0;
            }
#endif
        }
        if (letter) {
            letter_hash_ = mix64(letter_hash_ * 0x100000001B3ull +
                                 static_cast<std::uint64_t>(byte | 0x20));
        } else {
            letter_hash_ = 0;
        }

        if (byte == '\n') {
            if (col_pos_ < kLineMax)
                std::memset(line_buf_[cur_line_idx_] + col_pos_, 0,
                            static_cast<std::size_t>(kLineMax - col_pos_));
            cur_line_idx_ ^= 1;
            col_pos_ = 0;
        } else {
            if (col_pos_ < kLineMax)
                line_buf_[cur_line_idx_][col_pos_] = static_cast<std::uint8_t>(byte);
            if (col_pos_ < kLineMax - 1) ++col_pos_;
        }

#if HP_WIKI_STATES
        wiki_.push(byte);
#else
        if (byte == '<') { in_tag_ = 1; tag_name_ = 0; if (tag_depth_ < 15) ++tag_depth_; }
        else if (byte == '>') { in_tag_ = 0; }
        else if (byte == '/' && in_tag_) { if (tag_depth_ > 0) --tag_depth_; }
        else if (in_tag_) tag_name_ = mix64(tag_name_ * 31 + byte);
#endif

#if HP_BRACKET
        brackets_.push(byte);
#endif
        streams_.push(byte, alnum);
#if HP_SENT_MEM
#if HP_SENT_DOM
        sentmem_.set_domain(wiki_.sen_group());
#endif
        if (at_boundary) sentmem_.push_word(word_hash_prev_);
        if (byte == '.' || byte == '!' || byte == '?' || byte == '\n')
            sentmem_.end_sentence();
#endif
#if HP_STEMMER || HP_STEM_FOLD || HP_POS_GATE || HP_WT3_CTX
        {
            const int nest = wiki_.nest_markup() ||
                             brackets_.square_depth() > 0 ||
                             brackets_.curly_depth() > 0;
            stems_.push(byte, nest);
        }
#endif
#if HP_NUMERIC
        numbers_.push(byte);
#endif

        if (!alnum && word_hash_prev_ != 0) {
            hebb_.potentiate(prev_word_, word_hash_prev_);
            prev_word_ = word_hash_prev_;
            word_ring_[3] = word_ring_[2];
            word_ring_[2] = word_ring_[1];
            word_ring_[1] = word_ring_[0];
            word_ring_[0] = prev_word_;
        }
        word_hash_prev_ = word_hash_;
#if HP_HEBB_GRP
        hebb_.set_context(prev_word_ +
                          static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull);
#else
        hebb_.set_context(prev_word_);
#endif
#if HP_GATE_BRANCH
        branch3_.push_byte(byte, hist_);
#endif
        hist2_ = (hist2_ << 8) | ((hist_ >> 56) & 0xffull);
        pool_.end_byte();
        pool_.set_contexts(hist_, hist2_);

        byte_ring_.push(static_cast<std::uint8_t>(byte));
        for (int i = 0; i < kMatchModels; ++i) match_[i].push_byte(byte, hist_);
#if HP_GATE_BREAK
        {
            int ml = 0;
            for (int i = 0; i < kMatchModels; ++i)
                if (match_[i].match_len() > ml) ml = match_[i].match_len();
            break_age_ = ml > 0 ? (break_age_ < 255 ? break_age_ + 1 : 255) : 0;
        }
#endif
#if HP_SPARSE_UTF8
        {
            std::uint64_t sh = 0;
            for (int i = 0; i < 4; ++i)
                sh = (sh << 8) | ((hist_ >> (16 * i)) & 0xffull);
            smatch_.push_byte(byte, sh);
        }
#endif
#if HP_SKIPK_MOD
        skipk_.push_byte(byte, hist_);
#endif
#if HP_SKIP3_MOD
        skip3_.push_byte(byte, hist_);
#endif
#if HP_SKIP4_MOD
        skip4_.push_byte(byte, hist_);
#endif
#if HP_SKIP5_MOD
        skip5_.push_byte(byte, hist_);
#endif
#if HP_LZP_MOD
        lzp_.push_byte(byte, hist_);
#endif
#if HP_SR_MOD
        sr_.push_byte(byte);
#endif
#if HP_WORD_MATCH
        // fx2 keys: {0} current, {1,3} current-alt + word-before-prev, {7,2} letters + last
        const std::uint64_t w0 = word_hash_ ? word_hash_ : word_ring_[0];
        const std::uint64_t w13 = mix64(word_hash_ * 263ull + word_ring_[1]);
        const std::uint64_t w72 = mix64(letter_hash_ * 997ull + word_ring_[0]);
#if HP_WMATCH_4
        const std::uint64_t w123 = mix64(word_hash_ * 31ull + word_ring_[0] * 17ull +
                                         word_ring_[1]);
#endif
        std::uint64_t whist[5] = {w0, w13, w72, 0, 0};
        int nw = 3;
#if HP_WMATCH_4
        whist[nw++] = w123;
#endif
#if HP_WMATCH_5
        whist[nw++] = streams_.stream(1);
#endif
        (void)nw;
        for (int i = 0; i < kWordMatch; ++i)
            wmatch_[i].push_byte(byte, whist[i], at_boundary);
#endif

#if HP_UTF8_IDLE || HP_GATE_UTF8
        if (utf8left_ > 0) {
            if ((byte & 0xC0) == 0x80) --utf8left_;
            else utf8left_ = 0;
        }
        if (byte >= 0xC0 && byte < 0xE0) utf8left_ = 1;
        else if (byte >= 0xE0 && byte < 0xF0) utf8left_ = 2;
        else if (byte >= 0xF0 && byte < 0xF8) utf8left_ = 3;
#endif
        cache_.observe_byte(byte, wiki_.state(), wiki_.in_table(),
                            streams_.first_class());

        gria_.account_byte(byte);
        set_byte_contexts();
    }

    void set_byte_contexts() {
        const int col = (col_pos_ < kLineMax) ? col_pos_ : kLineMax - 1;
        {
            std::uint64_t ck;
#if HP_TABLE_ABOVE
            if (wiki_.in_table())
                ck = (static_cast<std::uint64_t>(wiki_.above_cell()) << 16) |
                     static_cast<std::uint64_t>(col & 63);
            else
#endif
                ck = (static_cast<std::uint64_t>(line_buf_[cur_line_idx_ ^ 1][col]) << 16) |
                     static_cast<std::uint64_t>(col & 63);
#if HP_COL_GRP
            ck += static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull;
#endif
            col_.set_context(h2(21, ck));
        }
#if HP_WIKI_STATES
        tag_.set_context(h2(22, wiki_.context_key()
#if HP_TAG_GRP
            + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
        ));
#else
        tag_.set_context(h2(22, (static_cast<std::uint64_t>(tag_depth_ & 15) << 40) |
                                    (static_cast<std::uint64_t>(in_tag_) << 39) |
                                    (tag_name_ & 0x7FFFFFFFFFull)
#if HP_TAG_GRP
                                    + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
                                    ));
#endif
#if HP_WBI_SENTPOS
        wbi_.set_context(h2(23, prev_word_ * 0x9E3779B97F4A7C15ull + word_hash_ +
                               static_cast<std::uint64_t>(streams_.sent_pos()) * 17ull));
#elif HP_WBI_GRP
        wbi_.set_context(h2(23, prev_word_ * 0x9E3779B97F4A7C15ull + word_hash_ +
                               static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull));
#else
        wbi_.set_context(h2(23, prev_word_ * 0x9E3779B97F4A7C15ull + word_hash_));
#endif

        o1_.set_context(h2(1, hist_ & 0xffull));
        o2_.set_context(h2(2, hist_ & 0xffffull));
        o3_.set_context(h2(3, hist_ & 0xffffffull));
        o4_.set_context(h2(4, hist_ & 0xffffffffull));
        o6_.set_context(h2(6, hist_ & 0xffffffffffffull));
#if HP_SECTION_MUTE
        if (wiki_.mute_words())
            word_.set_idle();
        else
#endif
#if HP_PRED_GATE
        if (pred_gate_mute())
            word_.set_idle();
        else
#endif
#if HP_UTF8_IDLE
        if (utf8left_ > 0)
            word_.set_idle();
        else
#endif
#if HP_STEM_FOLD
        word_.set_context(h2(7, word_hash_ ^ (stems_.sentence() * 0x9E3779B97F4A7C15ull)));
#elif HP_SENT_RECENCY
        word_.set_context(h2(7, word_hash_ * 1471ull + streams_.recency_at() +
                               (hist_ & 0xffull)));
#elif HP_WORD_GRP
        word_.set_context(h2(7, word_hash_ +
                               static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull));
#else
        word_.set_context(h2(7, word_hash_));
#endif
        sp13_.set_context(h2(8, (((hist_ >> 0) & 0xffull) |
                                   (((hist_ >> 16) & 0xffull) << 8))
#if HP_SP_GRP
                                   + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
                                   ));
        sp24_.set_context(h2(9, (((hist_ >> 8) & 0xffull) |
                                   (((hist_ >> 24) & 0xffull) << 8))
#if HP_SP_GRP
                                   + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
                                   ));
#if HP_WORD_STREAMS
#if HP_SECTION_MUTE
        if (wiki_.mute_words())
            wstr_sp_.set_idle();
        else
#endif
#if HP_PRED_GATE
        if (pred_gate_mute())
            wstr_sp_.set_idle();
        else
#endif
#if HP_UTF8_IDLE
        if (utf8left_ > 0)
            wstr_sp_.set_idle();
        else
#endif
#if HP_STEM_FOLD
        wstr_sp_.set_context(h2(25, streams_.prev0() * 131ull + streams_.stream(2) +
                                   stems_.typed() * 17ull + stems_.paragraph()));
#elif HP_FIRST_WORD
        wstr_sp_.set_context(h2(25, streams_.prev0() * 131ull + streams_.stream(2) +
                                   streams_.first_word() * 89ull));
#elif HP_WT3_CTX
        wstr_sp_.set_context(h2(25, streams_.prev0() * 131ull + streams_.stream(2) +
                                   stems_.wt3() * 17ull));
#else
#if HP_WSTR_GRP
        wstr_sp_.set_context(h2(25, streams_.prev0() * 131ull + streams_.stream(2) +
                               static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull));
#else
        wstr_sp_.set_context(h2(25, streams_.prev0() * 131ull + streams_.stream(2)));
#endif
#endif
#endif
#if HP_BRACKET
        {
            std::uint64_t bk = brackets_.context_key();
#if HP_BRK_CLOSE
            bk += static_cast<std::uint64_t>(brackets_.closer()) << 16;
#endif
#if HP_QUOTE_STACK
            bk += static_cast<std::uint64_t>(brackets_.quote()) << 24;
#endif
#if HP_BRK_GRP
            bk += static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull;
#endif
            brk_.set_context(h2(26, bk));
        }
#endif
#if HP_LINKWORD
#if HP_UTF8_IDLE
        if (utf8left_ > 0) {
            link_.set_idle();
#if HP_SENWORD
            sen_.set_idle();
#endif
        } else
#endif
        {
            const std::uint64_t lw = wiki_.linkword();
#if HP_SENWORD
#if HP_LINK_NUM
            link_.set_context(h2(29, (lw ? lw : word_hash_) * 3301ull +
                                   numbers_.previous() * 3191ull
#if HP_LINK_GRP
                                   + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
                                   ));
#else
            link_.set_context(h2(29, lw
#if HP_LINK_GRP
                + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
            ));
#endif
            {
                const std::uint64_t sw = wiki_.senword();
                sen_.set_context(h2(31, (sw ? sw * 1471ull + (hist_ & 0xffull) : 0)
#if HP_SENWORD_GRP
                    + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
                ));
            }
#else
            const std::uint64_t sw = wiki_.senword();
            const std::uint64_t key = lw ? lw : (sw ? sw * 1471ull + (hist_ & 0xffull) : 0);
#if HP_LINK_NUM
            link_.set_context(h2(29, key * 3301ull + numbers_.previous() * 3191ull
#if HP_LINK_GRP
                + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
            ));
#else
            link_.set_context(h2(29, key
#if HP_LINK_GRP
                + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
            ));
#endif
#endif
        }
#endif
#if HP_STEMMER
#if HP_PRED_GATE
        if (pred_gate_mute()) {
            stem0_.set_idle();
#if HP_STEMMER_N >= 2
            stem1_.set_idle();
#endif
        } else
#endif
        {
            stem0_.set_context(h2(32, stems_.ctx0() + stems_.ctx1()));
#if HP_STEMMER_N >= 2
            stem1_.set_context(h2(33, stems_.ctx1()));
#endif
        }
#endif
#if HP_SENT_STREAM
#if HP_SENT_GRP_CTX
        sentst_.set_context(h2(34, streams_.stream(3) * 83ull + (hist_ & 0xffull) +
                               static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull));
#else
        sentst_.set_context(h2(34, streams_.stream(3) * 83ull + (hist_ & 0xffull)));
#endif
#endif
#if HP_SENT_MEM
#if HP_SENT_CUR
        sentmem_cm_.set_context(h2(35, sentmem_.match_hash() * 53ull +
                                       sentmem_.cur_hash() * 17ull +
                                       static_cast<std::uint64_t>(sentmem_.word_pos()) +
                                       (hist_ & 0xffull)));
#elif HP_SENT_ALIGN
        sentmem_cm_.set_context(h2(35, sentmem_.aligned_word() * 53ull +
                                       static_cast<std::uint64_t>(sentmem_.word_pos()) +
                                       (hist_ & 0xffull)));
#elif HP_SMEM_GRP
        sentmem_cm_.set_context(h2(35, sentmem_.match_hash() * 53ull +
                                       static_cast<std::uint64_t>(sentmem_.word_pos()) +
                                       (hist_ & 0xffull) +
                                       static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull));
#else
        sentmem_cm_.set_context(h2(35, sentmem_.match_hash() * 53ull +
                                       static_cast<std::uint64_t>(sentmem_.word_pos()) +
                                       (hist_ & 0xffull)));
#endif
#endif
#if HP_NUMERIC
        num_.set_context(h2(30, numbers_.context_key()
#if HP_NUM_GRP
            + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
#endif
        ));
#endif
#if HP_PAT_MODEL
        pat_.set_context(h2(27, (static_cast<std::uint64_t>(cache_.cls()) << 16) |
                                   (hist_ & 0xffull)));
#endif
#if HP_PPMD
        ppm_.set_context(h2(28, hist_));
#endif
#if HP_SENGRP_MOD
#if HP_SENGRP_WORD
        sengrp_.set_context(h2(36, static_cast<std::uint64_t>(wiki_.sen_group()) +
                                   ((hist_ & 0xffffffull) << 8) + word_hash_ * 17ull));
#elif HP_SENGRP_POS
        sengrp_.set_context(h2(36, static_cast<std::uint64_t>(wiki_.sen_group()) +
                                   ((hist_ & 0xffffffull) << 8) +
                                   static_cast<std::uint64_t>(streams_.sent_pos()) * 17ull));
#elif HP_SENGRP_C0
        sengrp_.set_context(h2(36, static_cast<std::uint64_t>(wiki_.sen_group()) +
                                   ((hist_ & 0xffffffull) << 8) +
                                   static_cast<std::uint64_t>(c0_) * 17ull));
#else
        sengrp_.set_context(h2(36, static_cast<std::uint64_t>(wiki_.sen_group()) +
                                   ((hist_ & 0xffffffull) << 8)));
#endif
#endif
#if HP_NEST_MOD
        nestmod_.set_context(h2(37, static_cast<std::uint64_t>(wiki_.nest_markup()) +
                                    ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PARA_MOD
        paramod_.set_context(h2(38, static_cast<std::uint64_t>(wiki_.is_paragraph()) +
                                    ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LINE_MOD
        linemod_.set_context(h2(39, static_cast<std::uint64_t>(wiki_.line_kind() & 255) +
                                    ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_STATE_MOD
        statemod_.set_context(h2(40, static_cast<std::uint64_t>(wiki_.state()) +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DOM_MOD
        dommod_.set_context(h2(41, static_cast<std::uint64_t>(wiki_.sent_domain()) +
                                   ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_HDR_MOD
        hdrmod_.set_context(h2(42, static_cast<std::uint64_t>(wiki_.wiki_header()) +
                                   ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DEPTH_MOD
        depthmod_.set_context(h2(43, static_cast<std::uint64_t>(wiki_.depth()) +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_FCCXT_MOD
        fccxtmod_.set_context(h2(44, static_cast<std::uint64_t>(wiki_.cell_first()) +
                                     (static_cast<std::uint64_t>(wiki_.above_cell()) << 8) +
                                     (static_cast<std::uint64_t>(wiki_.tbl_cell() & 31) << 16) +
                                     ((hist_ & 0xffull) << 24)));
#endif
#if HP_TPLNAME_MOD
        tplmod_.set_context(h2(45, wiki_.tpl_name() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_INFOKEY_MOD
        infokeymod_.set_context(h2(46, wiki_.infokey() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_BARIDX_MOD
        baridxmod_.set_context(h2(47, static_cast<std::uint64_t>(wiki_.bar_idx()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PERIOD_MOD
        {
            int period = 0;
            const int n = col_pos_;
            if (n >= 8) {
                for (int p = 2; p <= 32 && p * 2 <= n; ++p) {
                    int ok = 1;
                    for (int k = 0; k < 8; ++k) {
                        if (line_buf_[cur_line_idx_][n - 1 - k] !=
                            line_buf_[cur_line_idx_][n - 1 - k - p]) {
                            ok = 0;
                            break;
                        }
                    }
                    if (ok) { period = p; break; }
                }
            }
            periodmod_.set_context(h2(48, static_cast<std::uint64_t>(period) +
                                          ((hist_ & 0xffffffull) << 8)));
        }
#endif
#if HP_PRONOUN_MOD
        pronounmod_.set_context(h2(49, static_cast<std::uint64_t>(pronoun_) +
                                       ((hist_ & 0xffffffull) << 8) +
                                       (word_hash_ * 17ull)));
#endif
#if HP_HASH2_O6
        o6b_.set_context(h2(106, hist_ & 0xffffffffffffull));
#endif
#if HP_LINKPIPE_MOD
        linkpipemod_.set_context(h2(51, wiki_.link_disp() +
                                        ((hist_ & 0xffffffull) << 8) +
                                        static_cast<std::uint64_t>(wiki_.after_pipe()) * 131ull));
#endif
#if HP_CITE_MOD
        citemod_.set_context(h2(52, static_cast<std::uint64_t>(wiki_.in_ref()) +
                                    ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_CAT_MOD
        catmod_.set_context(h2(53, wiki_.cat_ns() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_REDIR_MOD
        redirmod_.set_context(h2(54, static_cast<std::uint64_t>(wiki_.in_redir()) +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_HEADING_MOD
        headingmod_.set_context(h2(55, static_cast<std::uint64_t>(wiki_.heading_level()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_EXTLINK_MOD
        extlinkmod_.set_context(h2(56, static_cast<std::uint64_t>(wiki_.in_ext()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_REFNAME_MOD
        refnamemod_.set_context(h2(57, wiki_.refname() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_QOCXT_MOD
        qocxtmod_.set_context(h2(58, static_cast<std::uint64_t>(brackets_.quote()) +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_ENTITY_MOD
        entitymod_.set_context(h2(59, wiki_.entity() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_INDENT_MOD
        indentmod_.set_context(h2(60, static_cast<std::uint64_t>(wiki_.indent_level()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LISTLEVEL_MOD
        listlevelmod_.set_context(h2(61, static_cast<std::uint64_t>(wiki_.list_level()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_ISSE_MOD
        issemod_.set_context(h2(62, (hist_ & 0xffull) +
                                    (static_cast<std::uint64_t>(o6_.last_p() >> 6) * 17ull)));
#endif
#if HP_MAGIC_MOD
        magicmod_.set_context(h2(63, wiki_.magic() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_NOWIKI_MOD
        nowikimod_.set_context(h2(64, static_cast<std::uint64_t>(wiki_.in_nowiki()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TITLE_MOD
        titlemod_.set_context(h2(65, wiki_.page_title() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PAGEID_MOD
        pageidmod_.set_context(h2(66, wiki_.page_id() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_USER_MOD
        usermod_.set_context(h2(67, wiki_.username() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TEXT_MOD
        textmod_.set_context(h2(68, static_cast<std::uint64_t>(wiki_.in_text()) +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_NS_MOD
        nsmod_.set_context(h2(69, wiki_.ns_id() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DUMPREDIR_MOD
        dumpredirmod_.set_context(h2(70, static_cast<std::uint64_t>(wiki_.dump_redir()) +
                                          ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_IP_MOD
        ipmod_.set_context(h2(71, wiki_.ip_hash() + ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_REVCOMMENT_MOD
        revcommentmod_.set_context(h2(72, wiki_.rev_comment() +
                                          ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_MINOR_MOD
        minormod_.set_context(h2(73, static_cast<std::uint64_t>(wiki_.minor_edit()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_WIKIMODEL_MOD
        wikimodelmod_.set_context(h2(74, wiki_.wiki_model() +
                                          ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SECTITLE_MOD
        sectitlemod_.set_context(h2(75, wiki_.sectitle() +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PARSERFN_MOD
        parserfnmod_.set_context(h2(76, wiki_.parser_fn() +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TABLECLASS_MOD
        tableclassmod_.set_context(h2(77, wiki_.table_class() +
                                          ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_ANCHOR_MOD
        anchormod_.set_context(h2(78, wiki_.anchor() +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PUBID_MOD
        pubidmod_.set_context(h2(79, wiki_.pub_id() +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TEMPPOS_MOD
        tempposmod_.set_context(h2(80, wiki_.temp_pos() +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_WIKISTACK_MOD
        wikistackmod_.set_context(h2(81, wiki_.wiki_stack() +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LANG_MOD
        langmod_.set_context(h2(82, wiki_.lang_prefix() +
                                    ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_CATSORT_MOD
        catsortmod_.set_context(h2(83, wiki_.cat_sort() +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TBLROW_MOD
        tblrowmod_.set_context(h2(84, static_cast<std::uint64_t>(wiki_.tbl_row()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_FILEOPT_MOD
        fileoptmod_.set_context(h2(85, wiki_.file_opt() +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DEFAULTSORT_MOD
        defaultsortmod_.set_context(h2(86, wiki_.default_sort() +
                                           ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_REDIRTARGET_MOD
        redirtargetmod_.set_context(h2(87, wiki_.redir_target() +
                                           ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DAB_MOD
        dabmod_.set_context(h2(88, wiki_.dab() +
                                   ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_HATNOTE_MOD
        hatnotemod_.set_context(h2(89, wiki_.hatnote() +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LASTLINK_MOD
        lastlinkmod_.set_context(h2(90, wiki_.last_link() +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_FWORD_MOD
        fwordmod_.set_context(h2(91, wiki_.first_word() +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_YEAR_MOD
        yearmod_.set_context(h2(92, wiki_.year() +
                                    ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_CAPMASK_MOD
        capmaskmod_.set_context(h2(93, static_cast<std::uint64_t>(wiki_.cap_mask()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_CELLTXT_MOD
        celltxtmod_.set_context(h2(94, wiki_.cell_text() +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_HTTPHOST_MOD
        httphostmod_.set_context(h2(95, wiki_.http_host() +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PAREN_MOD
        parenmod_.set_context(h2(96, wiki_.last_paren() +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LISTPOS_MOD
        listposmod_.set_context(h2(97, static_cast<std::uint64_t>(wiki_.list_pos()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SHAPE_MOD
        shapemod_.set_context(h2(98, wiki_.word_shape() +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SUFFIX_MOD
        suffixmod_.set_context(h2(99, wiki_.word_suffix() +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PREFIX_MOD
        prefixmod_.set_context(h2(100, wiki_.word_prefix() +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_CHARCLS_MOD
        charclsmod_.set_context(h2(101, static_cast<std::uint64_t>(wiki_.char_cls()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_VOWEL_MOD
        vowelmod_.set_context(h2(102, static_cast<std::uint64_t>(wiki_.vowel_mask()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_CONTR_MOD
        contrmod_.set_context(h2(103, wiki_.contraction() +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_HYPHEN_MOD
        hyphenmod_.set_context(h2(104, wiki_.hyphen_word() +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TOKENCLS_MOD
        tokenclsmod_.set_context(h2(105, static_cast<std::uint64_t>(wiki_.token_cls()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_RUNLEN_MOD
        runlenmod_.set_context(h2(106, static_cast<std::uint64_t>(wiki_.run_len()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_WPOS_MOD
        wposmod_.set_context(h2(107, static_cast<std::uint64_t>(wiki_.word_pos()) +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_BLANK_MOD
        blankmod_.set_context(h2(108, static_cast<std::uint64_t>(wiki_.blank_n()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SPRUN_MOD
        sprunmod_.set_context(h2(109, static_cast<std::uint64_t>(wiki_.space_run()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LINELEN_MOD
        linelenmod_.set_context(h2(110, static_cast<std::uint64_t>(wiki_.line_len()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TAGDIST_MOD
        tagdistmod_.set_context(h2(111, static_cast<std::uint64_t>(wiki_.tag_dist()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_MARKDIST_MOD
        markdistmod_.set_context(h2(112, static_cast<std::uint64_t>(wiki_.mark_dist()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_UPPERGAP_MOD
        uppergapmod_.set_context(h2(113, static_cast<std::uint64_t>(wiki_.upper_gap()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_MONTH_MOD
        monthmod_.set_context(h2(114, static_cast<std::uint64_t>(wiki_.month()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_GALLERY_MOD
        gallerymod_.set_context(h2(115, static_cast<std::uint64_t>(wiki_.in_gallery()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SECKIND_MOD
        seckindmod_.set_context(h2(116, static_cast<std::uint64_t>(wiki_.sec_kind()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_CITEKIND_MOD
        citekindmod_.set_context(h2(117, static_cast<std::uint64_t>(wiki_.cite_kind()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TAGNAME_MOD
        tagnamemod_.set_context(h2(118, wiki_.tag_name_cm() +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_COLSPAN_MOD
        colspanmod_.set_context(h2(119, static_cast<std::uint64_t>(wiki_.col_span()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_STYLE_MOD
        stylemod_.set_context(h2(120, wiki_.style_hash() +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_COORD_MOD
        coordmod_.set_context(h2(121, static_cast<std::uint64_t>(wiki_.in_coord()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DIGITGAP_MOD
        digitgapmod_.set_context(h2(122, static_cast<std::uint64_t>(wiki_.digit_gap()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DOTGAP_MOD
        dotgapmod_.set_context(h2(123, static_cast<std::uint64_t>(wiki_.dot_gap()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_COMMAGAP_MOD
        commagapmod_.set_context(h2(124, static_cast<std::uint64_t>(wiki_.comma_gap()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_WORDLEN_MOD
        wordlenmod_.set_context(h2(125, static_cast<std::uint64_t>(wiki_.word_len()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SENTLEN_MOD
        sentlenmod_.set_context(h2(126, static_cast<std::uint64_t>(wiki_.sent_len()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LOWERGAP_MOD
        lowergapmod_.set_context(h2(127, static_cast<std::uint64_t>(wiki_.lower_gap()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DIGITPOS_MOD
        digitposmod_.set_context(h2(128, static_cast<std::uint64_t>(wiki_.digit_pos()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SLASHGAP_MOD
        slashgapmod_.set_context(h2(129, static_cast<std::uint64_t>(wiki_.slash_gap()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DIGLEN_MOD
        diglenmod_.set_context(h2(130, static_cast<std::uint64_t>(wiki_.dig_len()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PREVLINE_MOD
        prevlinemod_.set_context(h2(131, static_cast<std::uint64_t>(wiki_.prev_line()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PREVSENT_MOD
        prevsentmod_.set_context(h2(132, static_cast<std::uint64_t>(wiki_.prev_sent()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LINKLEN_MOD
        linklenmod_.set_context(h2(133, static_cast<std::uint64_t>(wiki_.link_len()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TPLLEN_MOD
        tpllenmod_.set_context(h2(134, static_cast<std::uint64_t>(wiki_.tpl_len()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_PARALEN_MOD
        paralenmod_.set_context(h2(135, static_cast<std::uint64_t>(wiki_.para_len()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_ALNUMLEN_MOD
        alnumlenmod_.set_context(h2(136, static_cast<std::uint64_t>(wiki_.alnum_len()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SPLEN_MOD
        splenmod_.set_context(h2(137, static_cast<std::uint64_t>(wiki_.sp_len()) +
                                      ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TITLEWORD_MOD
        titlewordmod_.set_context(h2(138, static_cast<std::uint64_t>(wiki_.title_word()) +
                                          ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_HEADWORD_MOD
        headwordmod_.set_context(h2(139, static_cast<std::uint64_t>(wiki_.head_word()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_INIT_MOD
        initmod_.set_context(h2(140, static_cast<std::uint64_t>(wiki_.in_init()) +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_ORDINAL_MOD
        ordinalmod_.set_context(h2(141, static_cast<std::uint64_t>(wiki_.ordinal()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_UNIT_MOD
        unitmod_.set_context(h2(142, static_cast<std::uint64_t>(wiki_.unit()) +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_DECIMAL_MOD
        decimalmod_.set_context(h2(143, static_cast<std::uint64_t>(wiki_.in_decimal()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_REPEAT_MOD
        repeatmod_.set_context(h2(144, static_cast<std::uint64_t>(wiki_.word_repeat()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_CASEFLIP_MOD
        caseflipmod_.set_context(h2(145, static_cast<std::uint64_t>(wiki_.case_flip()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LEAD_MOD
        leadmod_.set_context(h2(146, static_cast<std::uint64_t>(wiki_.in_lead()) +
                                     ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_INFOVAL_MOD
        infovalmod_.set_context(h2(147, static_cast<std::uint64_t>(wiki_.info_val()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_LINKTRAIL_MOD
        linktrailmod_.set_context(h2(148, static_cast<std::uint64_t>(wiki_.link_trail()) +
                                          ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_CELLKIND_MOD
        cellkindmod_.set_context(h2(149, static_cast<std::uint64_t>(wiki_.cell_kind()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_TBLCOL_MOD
        tblcolmod_.set_context(h2(150, static_cast<std::uint64_t>(wiki_.tbl_col()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_HEADIDX_MOD
        headidxmod_.set_context(h2(151, static_cast<std::uint64_t>(wiki_.head_idx()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_HTMLFMT_MOD
        htmlfmtmod_.set_context(h2(152, static_cast<std::uint64_t>(wiki_.html_fmt()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_INFOBOX_MOD
        infoboxmod_.set_context(h2(153, static_cast<std::uint64_t>(wiki_.in_infobox()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SECLEVEL_MOD
        seclevelmod_.set_context(h2(154, static_cast<std::uint64_t>(wiki_.sec_level()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_BRACE3_MOD
        brace3mod_.set_context(h2(155, static_cast<std::uint64_t>(wiki_.brace3()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_NAMEDARG_MOD
        namedargmod_.set_context(h2(156, static_cast<std::uint64_t>(wiki_.named_arg()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_INCLUDE_MOD
        includemod_.set_context(h2(157, static_cast<std::uint64_t>(wiki_.include_bits()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_SIG_MOD
        sigmod_.set_context(h2(158, static_cast<std::uint64_t>(wiki_.sig_run()) +
                                    ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_WIKIBOLD_MOD
        wikiboldmod_.set_context(h2(159, static_cast<std::uint64_t>(wiki_.wiki_bold()) +
                                         ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_URLPART_MOD
        urlpartmod_.set_context(h2(160, static_cast<std::uint64_t>(wiki_.url_part()) +
                                        ((hist_ & 0xffffffull) << 8)));
#endif
#if HP_REFIDX_MOD
        refidxmod_.set_context(h2(161, static_cast<std::uint64_t>(wiki_.ref_idx()) +
                                       ((hist_ & 0xffffffull) << 8)));
#endif
    }

    void rebind_internal_pointers_() {
        init_ctx_chain_();
        for (int i = 0; i < kMatchModels; ++i) match_[i].set_ring(&byte_ring_);
#if HP_SPARSE_UTF8
        smatch_.set_ring(&byte_ring_);
#endif
#if HP_SKIPK_MOD
        skipk_.set_ring(&byte_ring_);
#endif
#if HP_SKIP3_MOD
        skip3_.set_ring(&byte_ring_);
#endif
#if HP_SKIP4_MOD
        skip4_.set_ring(&byte_ring_);
#endif
#if HP_SKIP5_MOD
        skip5_.set_ring(&byte_ring_);
#endif
#if HP_WORD_MATCH
        for (int i = 0; i < static_cast<int>(sizeof(wmatch_) / sizeof(wmatch_[0])); ++i) {
            wmatch_[i].set_ring(&byte_ring_);
        }
#endif
        set_byte_contexts();
    }

    void assign_from_(const Predictor& o) {
        byte_ring_ = o.byte_ring_;
        o1_ = o.o1_;
        o2_ = o.o2_;
        o3_ = o.o3_;
        o4_ = o.o4_;
        o6_ = o.o6_;
#if HP_HASH2_O6
        o6b_ = o.o6b_;
#endif
        word_ = o.word_;
        col_ = o.col_;
        tag_ = o.tag_;
        wbi_ = o.wbi_;
        sp13_ = o.sp13_;
        sp24_ = o.sp24_;
#if HP_WORD_STREAMS
        wstr_sp_ = o.wstr_sp_;
#endif
#if HP_BRACKET
        brk_ = o.brk_;
#endif
#if HP_LINKWORD
        link_ = o.link_;
#endif
#if HP_NUMERIC
        num_ = o.num_;
#endif
#if HP_PAT_MODEL
        pat_ = o.pat_;
#endif
#if HP_PPMD
        ppm_ = o.ppm_;
#endif
#if HP_STEMMER
        stem0_ = o.stem0_;
#if HP_STEMMER_N >= 2
        stem1_ = o.stem1_;
#endif
#endif
#if HP_SENWORD
        sen_ = o.sen_;
#endif
#if HP_SENT_STREAM
        sentst_ = o.sentst_;
#endif
#if HP_SENT_MEM
        sentmem_ = o.sentmem_;
        sentmem_cm_ = o.sentmem_cm_;
#endif
#if HP_SENGRP_MOD
        sengrp_ = o.sengrp_;
#endif
        for (int i = 0; i < kMatchModels; ++i) match_[i] = o.match_[i];
#if HP_SPARSE_UTF8
        smatch_ = o.smatch_;
#endif
#if HP_SKIPK_MOD
        skipk_ = o.skipk_;
#endif
#if HP_SKIP3_MOD
        skip3_ = o.skip3_;
#endif
#if HP_SKIP4_MOD
        skip4_ = o.skip4_;
#endif
#if HP_SKIP5_MOD
        skip5_ = o.skip5_;
#endif
#if HP_LZP_MOD
        lzp_ = o.lzp_;
#endif
#if HP_DMC_MOD
        dmc_ = o.dmc_;
#endif
#if HP_WORD_MATCH
        for (int i = 0; i < static_cast<int>(sizeof(wmatch_) / sizeof(wmatch_[0])); ++i) {
            wmatch_[i] = o.wmatch_[i];
        }
#endif
        hebb_ = o.hebb_;
        pool_ = o.pool_;
        mixer_ = o.mixer_;
        apm_c0_ = o.apm_c0_;
        apm_lex_ = o.apm_lex_;
        apm_gria_ = o.apm_gria_;
        hedge_ = o.hedge_;
        bias_ = o.bias_;
        gria_ = o.gria_;
        wiki_ = o.wiki_;
        streams_ = o.streams_;
#if HP_STEMMER || HP_STEM_FOLD || HP_POS_GATE || HP_WT3_CTX
        stems_ = o.stems_;
#endif
        brackets_ = o.brackets_;
        cache_ = o.cache_;
#if HP_NUMERIC
        numbers_ = o.numbers_;
#endif
#if HP_GATE_BRANCH
        branch3_ = o.branch3_;
#endif
#if HP_GATE_BREAK
        break_age_ = o.break_age_;
#endif
        hist_ = o.hist_;
        word_hash_ = o.word_hash_;
        letter_hash_ = o.letter_hash_;
        hist2_ = o.hist2_;
        std::memcpy(line_buf_, o.line_buf_, sizeof(line_buf_));
        cur_line_idx_ = o.cur_line_idx_;
        col_pos_ = o.col_pos_;
        tag_depth_ = o.tag_depth_;
        in_tag_ = o.in_tag_;
        tag_name_ = o.tag_name_;
        prev_word_ = o.prev_word_;
        word_hash_prev_ = o.word_hash_prev_;
        std::memcpy(word_ring_, o.word_ring_, sizeof(word_ring_));
        c0_ = o.c0_;
        bitpos_ = o.bitpos_;
        pr_final_ = o.pr_final_;
        std::memcpy(exp_p_, o.exp_p_, sizeof(exp_p_));
        n_exp_ = o.n_exp_;
        mixed_p_ = o.mixed_p_;
        last_mlen_ = o.last_mlen_;
        sparse_ = o.sparse_;
#if HP_PRONOUN_MOD
        std::memcpy(pw_, o.pw_, sizeof(pw_));
        pw_n_ = o.pw_n_;
        pronoun_ = o.pronoun_;
#endif
#if HP_UTF8_IDLE || HP_GATE_UTF8
        utf8left_ = o.utf8left_;
#endif
        rebind_internal_pointers_();
    }

    void init_ctx_chain_() {
        n_ctx_chain_ = 0;
        ctx_chain_[n_ctx_chain_++] = &o1_;
        ctx_chain_[n_ctx_chain_++] = &o2_;
        ctx_chain_[n_ctx_chain_++] = &o3_;
        ctx_chain_[n_ctx_chain_++] = &o4_;
        ctx_chain_[n_ctx_chain_++] = &o6_;
        ctx_chain_[n_ctx_chain_++] = &word_;
        ctx_chain_[n_ctx_chain_++] = &sp13_;
        ctx_chain_[n_ctx_chain_++] = &sp24_;
        ctx_chain_[n_ctx_chain_++] = &col_;
        ctx_chain_[n_ctx_chain_++] = &tag_;
        ctx_chain_[n_ctx_chain_++] = &wbi_;
#if HP_WORD_STREAMS
        ctx_chain_[n_ctx_chain_++] = &wstr_sp_;
#endif
#if HP_BRACKET
        ctx_chain_[n_ctx_chain_++] = &brk_;
#endif
#if HP_LINKWORD
        ctx_chain_[n_ctx_chain_++] = &link_;
#endif
#if HP_NUMERIC
        ctx_chain_[n_ctx_chain_++] = &num_;
#endif
#if HP_PAT_MODEL
        ctx_chain_[n_ctx_chain_++] = &pat_;
#endif
#if HP_PPMD
        ctx_chain_[n_ctx_chain_++] = &ppm_;
#endif
#if HP_STEMMER
        ctx_chain_[n_ctx_chain_++] = &stem0_;
#if HP_STEMMER_N >= 2
        ctx_chain_[n_ctx_chain_++] = &stem1_;
#endif
#endif
#if HP_SENWORD
        ctx_chain_[n_ctx_chain_++] = &sen_;
#endif
#if HP_SENT_STREAM
        ctx_chain_[n_ctx_chain_++] = &sentst_;
#endif
#if HP_SENT_MEM
        ctx_chain_[n_ctx_chain_++] = &sentmem_cm_;
#endif
#if HP_SENGRP_MOD
        ctx_chain_[n_ctx_chain_++] = &sengrp_;
#endif
#if HP_NEST_MOD
        ctx_chain_[n_ctx_chain_++] = &nestmod_;
#endif
#if HP_PARA_MOD
        ctx_chain_[n_ctx_chain_++] = &paramod_;
#endif
#if HP_LINE_MOD
        ctx_chain_[n_ctx_chain_++] = &linemod_;
#endif
#if HP_STATE_MOD
        ctx_chain_[n_ctx_chain_++] = &statemod_;
#endif
#if HP_DOM_MOD
        ctx_chain_[n_ctx_chain_++] = &dommod_;
#endif
#if HP_HDR_MOD
        ctx_chain_[n_ctx_chain_++] = &hdrmod_;
#endif
#if HP_DEPTH_MOD
        ctx_chain_[n_ctx_chain_++] = &depthmod_;
#endif
#if HP_FCCXT_MOD
        ctx_chain_[n_ctx_chain_++] = &fccxtmod_;
#endif
#if HP_TPLNAME_MOD
        ctx_chain_[n_ctx_chain_++] = &tplmod_;
#endif
#if HP_INFOKEY_MOD
        ctx_chain_[n_ctx_chain_++] = &infokeymod_;
#endif
#if HP_BARIDX_MOD
        ctx_chain_[n_ctx_chain_++] = &baridxmod_;
#endif
#if HP_PERIOD_MOD
        ctx_chain_[n_ctx_chain_++] = &periodmod_;
#endif
#if HP_PRONOUN_MOD
        ctx_chain_[n_ctx_chain_++] = &pronounmod_;
#endif
#if HP_HASH2_O6
        ctx_chain_[n_ctx_chain_++] = &o6b_;
#endif
#if HP_LINKPIPE_MOD
        ctx_chain_[n_ctx_chain_++] = &linkpipemod_;
#endif
#if HP_CITE_MOD
        ctx_chain_[n_ctx_chain_++] = &citemod_;
#endif
#if HP_CAT_MOD
        ctx_chain_[n_ctx_chain_++] = &catmod_;
#endif
#if HP_REDIR_MOD
        ctx_chain_[n_ctx_chain_++] = &redirmod_;
#endif
#if HP_HEADING_MOD
        ctx_chain_[n_ctx_chain_++] = &headingmod_;
#endif
#if HP_EXTLINK_MOD
        ctx_chain_[n_ctx_chain_++] = &extlinkmod_;
#endif
#if HP_REFNAME_MOD
        ctx_chain_[n_ctx_chain_++] = &refnamemod_;
#endif
#if HP_QOCXT_MOD
        ctx_chain_[n_ctx_chain_++] = &qocxtmod_;
#endif
#if HP_ENTITY_MOD
        ctx_chain_[n_ctx_chain_++] = &entitymod_;
#endif
#if HP_INDENT_MOD
        ctx_chain_[n_ctx_chain_++] = &indentmod_;
#endif
#if HP_LISTLEVEL_MOD
        ctx_chain_[n_ctx_chain_++] = &listlevelmod_;
#endif
#if HP_ISSE_MOD
        ctx_chain_[n_ctx_chain_++] = &issemod_;
#endif
#if HP_MAGIC_MOD
        ctx_chain_[n_ctx_chain_++] = &magicmod_;
#endif
#if HP_NOWIKI_MOD
        ctx_chain_[n_ctx_chain_++] = &nowikimod_;
#endif
#if HP_TITLE_MOD
        ctx_chain_[n_ctx_chain_++] = &titlemod_;
#endif
#if HP_PAGEID_MOD
        ctx_chain_[n_ctx_chain_++] = &pageidmod_;
#endif
#if HP_USER_MOD
        ctx_chain_[n_ctx_chain_++] = &usermod_;
#endif
#if HP_TEXT_MOD
        ctx_chain_[n_ctx_chain_++] = &textmod_;
#endif
#if HP_NS_MOD
        ctx_chain_[n_ctx_chain_++] = &nsmod_;
#endif
#if HP_DUMPREDIR_MOD
        ctx_chain_[n_ctx_chain_++] = &dumpredirmod_;
#endif
#if HP_IP_MOD
        ctx_chain_[n_ctx_chain_++] = &ipmod_;
#endif
#if HP_REVCOMMENT_MOD
        ctx_chain_[n_ctx_chain_++] = &revcommentmod_;
#endif
#if HP_MINOR_MOD
        ctx_chain_[n_ctx_chain_++] = &minormod_;
#endif
#if HP_WIKIMODEL_MOD
        ctx_chain_[n_ctx_chain_++] = &wikimodelmod_;
#endif
#if HP_SECTITLE_MOD
        ctx_chain_[n_ctx_chain_++] = &sectitlemod_;
#endif
#if HP_PARSERFN_MOD
        ctx_chain_[n_ctx_chain_++] = &parserfnmod_;
#endif
#if HP_TABLECLASS_MOD
        ctx_chain_[n_ctx_chain_++] = &tableclassmod_;
#endif
#if HP_ANCHOR_MOD
        ctx_chain_[n_ctx_chain_++] = &anchormod_;
#endif
#if HP_PUBID_MOD
        ctx_chain_[n_ctx_chain_++] = &pubidmod_;
#endif
#if HP_TEMPPOS_MOD
        ctx_chain_[n_ctx_chain_++] = &tempposmod_;
#endif
#if HP_WIKISTACK_MOD
        ctx_chain_[n_ctx_chain_++] = &wikistackmod_;
#endif
#if HP_LANG_MOD
        ctx_chain_[n_ctx_chain_++] = &langmod_;
#endif
#if HP_CATSORT_MOD
        ctx_chain_[n_ctx_chain_++] = &catsortmod_;
#endif
#if HP_TBLROW_MOD
        ctx_chain_[n_ctx_chain_++] = &tblrowmod_;
#endif
#if HP_FILEOPT_MOD
        ctx_chain_[n_ctx_chain_++] = &fileoptmod_;
#endif
#if HP_DEFAULTSORT_MOD
        ctx_chain_[n_ctx_chain_++] = &defaultsortmod_;
#endif
#if HP_REDIRTARGET_MOD
        ctx_chain_[n_ctx_chain_++] = &redirtargetmod_;
#endif
#if HP_DAB_MOD
        ctx_chain_[n_ctx_chain_++] = &dabmod_;
#endif
#if HP_HATNOTE_MOD
        ctx_chain_[n_ctx_chain_++] = &hatnotemod_;
#endif
#if HP_LASTLINK_MOD
        ctx_chain_[n_ctx_chain_++] = &lastlinkmod_;
#endif
#if HP_FWORD_MOD
        ctx_chain_[n_ctx_chain_++] = &fwordmod_;
#endif
#if HP_YEAR_MOD
        ctx_chain_[n_ctx_chain_++] = &yearmod_;
#endif
#if HP_CAPMASK_MOD
        ctx_chain_[n_ctx_chain_++] = &capmaskmod_;
#endif
#if HP_CELLTXT_MOD
        ctx_chain_[n_ctx_chain_++] = &celltxtmod_;
#endif
#if HP_HTTPHOST_MOD
        ctx_chain_[n_ctx_chain_++] = &httphostmod_;
#endif
#if HP_PAREN_MOD
        ctx_chain_[n_ctx_chain_++] = &parenmod_;
#endif
#if HP_LISTPOS_MOD
        ctx_chain_[n_ctx_chain_++] = &listposmod_;
#endif
#if HP_SHAPE_MOD
        ctx_chain_[n_ctx_chain_++] = &shapemod_;
#endif
#if HP_SUFFIX_MOD
        ctx_chain_[n_ctx_chain_++] = &suffixmod_;
#endif
#if HP_PREFIX_MOD
        ctx_chain_[n_ctx_chain_++] = &prefixmod_;
#endif
#if HP_CHARCLS_MOD
        ctx_chain_[n_ctx_chain_++] = &charclsmod_;
#endif
#if HP_VOWEL_MOD
        ctx_chain_[n_ctx_chain_++] = &vowelmod_;
#endif
#if HP_CONTR_MOD
        ctx_chain_[n_ctx_chain_++] = &contrmod_;
#endif
#if HP_HYPHEN_MOD
        ctx_chain_[n_ctx_chain_++] = &hyphenmod_;
#endif
#if HP_TOKENCLS_MOD
        ctx_chain_[n_ctx_chain_++] = &tokenclsmod_;
#endif
#if HP_RUNLEN_MOD
        ctx_chain_[n_ctx_chain_++] = &runlenmod_;
#endif
#if HP_WPOS_MOD
        ctx_chain_[n_ctx_chain_++] = &wposmod_;
#endif
#if HP_BLANK_MOD
        ctx_chain_[n_ctx_chain_++] = &blankmod_;
#endif
#if HP_SPRUN_MOD
        ctx_chain_[n_ctx_chain_++] = &sprunmod_;
#endif
#if HP_LINELEN_MOD
        ctx_chain_[n_ctx_chain_++] = &linelenmod_;
#endif
#if HP_TAGDIST_MOD
        ctx_chain_[n_ctx_chain_++] = &tagdistmod_;
#endif
#if HP_MARKDIST_MOD
        ctx_chain_[n_ctx_chain_++] = &markdistmod_;
#endif
#if HP_UPPERGAP_MOD
        ctx_chain_[n_ctx_chain_++] = &uppergapmod_;
#endif
#if HP_MONTH_MOD
        ctx_chain_[n_ctx_chain_++] = &monthmod_;
#endif
#if HP_GALLERY_MOD
        ctx_chain_[n_ctx_chain_++] = &gallerymod_;
#endif
#if HP_SECKIND_MOD
        ctx_chain_[n_ctx_chain_++] = &seckindmod_;
#endif
#if HP_CITEKIND_MOD
        ctx_chain_[n_ctx_chain_++] = &citekindmod_;
#endif
#if HP_TAGNAME_MOD
        ctx_chain_[n_ctx_chain_++] = &tagnamemod_;
#endif
#if HP_COLSPAN_MOD
        ctx_chain_[n_ctx_chain_++] = &colspanmod_;
#endif
#if HP_STYLE_MOD
        ctx_chain_[n_ctx_chain_++] = &stylemod_;
#endif
#if HP_COORD_MOD
        ctx_chain_[n_ctx_chain_++] = &coordmod_;
#endif
#if HP_DIGITGAP_MOD
        ctx_chain_[n_ctx_chain_++] = &digitgapmod_;
#endif
#if HP_DOTGAP_MOD
        ctx_chain_[n_ctx_chain_++] = &dotgapmod_;
#endif
#if HP_COMMAGAP_MOD
        ctx_chain_[n_ctx_chain_++] = &commagapmod_;
#endif
#if HP_WORDLEN_MOD
        ctx_chain_[n_ctx_chain_++] = &wordlenmod_;
#endif
#if HP_SENTLEN_MOD
        ctx_chain_[n_ctx_chain_++] = &sentlenmod_;
#endif
#if HP_LOWERGAP_MOD
        ctx_chain_[n_ctx_chain_++] = &lowergapmod_;
#endif
#if HP_DIGITPOS_MOD
        ctx_chain_[n_ctx_chain_++] = &digitposmod_;
#endif
#if HP_SLASHGAP_MOD
        ctx_chain_[n_ctx_chain_++] = &slashgapmod_;
#endif
#if HP_DIGLEN_MOD
        ctx_chain_[n_ctx_chain_++] = &diglenmod_;
#endif
#if HP_PREVLINE_MOD
        ctx_chain_[n_ctx_chain_++] = &prevlinemod_;
#endif
#if HP_PREVSENT_MOD
        ctx_chain_[n_ctx_chain_++] = &prevsentmod_;
#endif
#if HP_LINKLEN_MOD
        ctx_chain_[n_ctx_chain_++] = &linklenmod_;
#endif
#if HP_TPLLEN_MOD
        ctx_chain_[n_ctx_chain_++] = &tpllenmod_;
#endif
#if HP_PARALEN_MOD
        ctx_chain_[n_ctx_chain_++] = &paralenmod_;
#endif
#if HP_ALNUMLEN_MOD
        ctx_chain_[n_ctx_chain_++] = &alnumlenmod_;
#endif
#if HP_SPLEN_MOD
        ctx_chain_[n_ctx_chain_++] = &splenmod_;
#endif
#if HP_TITLEWORD_MOD
        ctx_chain_[n_ctx_chain_++] = &titlewordmod_;
#endif
#if HP_HEADWORD_MOD
        ctx_chain_[n_ctx_chain_++] = &headwordmod_;
#endif
#if HP_INIT_MOD
        ctx_chain_[n_ctx_chain_++] = &initmod_;
#endif
#if HP_ORDINAL_MOD
        ctx_chain_[n_ctx_chain_++] = &ordinalmod_;
#endif
#if HP_UNIT_MOD
        ctx_chain_[n_ctx_chain_++] = &unitmod_;
#endif
#if HP_DECIMAL_MOD
        ctx_chain_[n_ctx_chain_++] = &decimalmod_;
#endif
#if HP_REPEAT_MOD
        ctx_chain_[n_ctx_chain_++] = &repeatmod_;
#endif
#if HP_CASEFLIP_MOD
        ctx_chain_[n_ctx_chain_++] = &caseflipmod_;
#endif
#if HP_LEAD_MOD
        ctx_chain_[n_ctx_chain_++] = &leadmod_;
#endif
#if HP_INFOVAL_MOD
        ctx_chain_[n_ctx_chain_++] = &infovalmod_;
#endif
#if HP_LINKTRAIL_MOD
        ctx_chain_[n_ctx_chain_++] = &linktrailmod_;
#endif
#if HP_CELLKIND_MOD
        ctx_chain_[n_ctx_chain_++] = &cellkindmod_;
#endif
#if HP_TBLCOL_MOD
        ctx_chain_[n_ctx_chain_++] = &tblcolmod_;
#endif
#if HP_HEADIDX_MOD
        ctx_chain_[n_ctx_chain_++] = &headidxmod_;
#endif
#if HP_HTMLFMT_MOD
        ctx_chain_[n_ctx_chain_++] = &htmlfmtmod_;
#endif
#if HP_INFOBOX_MOD
        ctx_chain_[n_ctx_chain_++] = &infoboxmod_;
#endif
#if HP_SECLEVEL_MOD
        ctx_chain_[n_ctx_chain_++] = &seclevelmod_;
#endif
#if HP_BRACE3_MOD
        ctx_chain_[n_ctx_chain_++] = &brace3mod_;
#endif
#if HP_NAMEDARG_MOD
        ctx_chain_[n_ctx_chain_++] = &namedargmod_;
#endif
#if HP_INCLUDE_MOD
        ctx_chain_[n_ctx_chain_++] = &includemod_;
#endif
#if HP_SIG_MOD
        ctx_chain_[n_ctx_chain_++] = &sigmod_;
#endif
#if HP_WIKIBOLD_MOD
        ctx_chain_[n_ctx_chain_++] = &wikiboldmod_;
#endif
#if HP_URLPART_MOD
        ctx_chain_[n_ctx_chain_++] = &urlpartmod_;
#endif
#if HP_REFIDX_MOD
        ctx_chain_[n_ctx_chain_++] = &refidxmod_;
#endif
    }

    static int pronoun_word(const std::uint8_t* w, int n) {
        auto eq = [&](const char* s) {
            int m = 0;
            while (s[m]) ++m;
            if (m != n) return 0;
            for (int i = 0; i < m; ++i)
                if (w[i] != static_cast<std::uint8_t>(s[i])) return 0;
            return 1;
        };
        return eq("he") || eq("she") || eq("it") || eq("his") || eq("her") ||
               eq("him") || eq("they") || eq("them") || eq("this") ||
               eq("that") || eq("who") || eq("its") || eq("we") || eq("our") ||
               eq("you") || eq("their");
    }

    ByteRing byte_ring_;
    ContextModel o1_, o2_, o3_, o4_, o6_;
#if HP_HASH2_O6
    ContextModel o6b_;
#endif
    ContextModel word_;
    ContextModel col_, tag_, wbi_;
    ContextModel sp13_, sp24_;
#if HP_WORD_STREAMS
    ContextModel wstr_sp_;
#endif
#if HP_BRACKET
    ContextModel brk_;
#endif
#if HP_LINKWORD
    ContextModel link_;
#endif
#if HP_NUMERIC
    ContextModel num_;
#endif
#if HP_PAT_MODEL
    ContextModel pat_;
#endif
#if HP_PPMD
    ContextModel ppm_;
#endif
#if HP_STEMMER
    ContextModel stem0_;
#if HP_STEMMER_N >= 2
    ContextModel stem1_;
#endif
#endif
#if HP_SENWORD
    ContextModel sen_;
#endif
#if HP_SENT_STREAM
    ContextModel sentst_;
#endif
#if HP_SENT_MEM
    SentenceMemory sentmem_;
    ContextModel sentmem_cm_;
#endif
#if HP_SENGRP_MOD
    ContextModel sengrp_;
#endif
#if HP_NEST_MOD
    ContextModel nestmod_;
#endif
#if HP_PARA_MOD
    ContextModel paramod_;
#endif
#if HP_LINE_MOD
    ContextModel linemod_;
#endif
#if HP_STATE_MOD
    ContextModel statemod_;
#endif
#if HP_DOM_MOD
    ContextModel dommod_;
#endif
#if HP_HDR_MOD
    ContextModel hdrmod_;
#endif
#if HP_DEPTH_MOD
    ContextModel depthmod_;
#endif
#if HP_FCCXT_MOD
    ContextModel fccxtmod_;
#endif
#if HP_TPLNAME_MOD
    ContextModel tplmod_;
#endif
#if HP_INFOKEY_MOD
    ContextModel infokeymod_;
#endif
#if HP_BARIDX_MOD
    ContextModel baridxmod_;
#endif
#if HP_PERIOD_MOD
    ContextModel periodmod_;
#endif
#if HP_PRONOUN_MOD
    ContextModel pronounmod_;
#endif
#if HP_LINKPIPE_MOD
    ContextModel linkpipemod_;
#endif
#if HP_CITE_MOD
    ContextModel citemod_;
#endif
#if HP_CAT_MOD
    ContextModel catmod_;
#endif
#if HP_REDIR_MOD
    ContextModel redirmod_;
#endif
#if HP_HEADING_MOD
    ContextModel headingmod_;
#endif
#if HP_EXTLINK_MOD
    ContextModel extlinkmod_;
#endif
#if HP_REFNAME_MOD
    ContextModel refnamemod_;
#endif
#if HP_QOCXT_MOD
    ContextModel qocxtmod_;
#endif
#if HP_ENTITY_MOD
    ContextModel entitymod_;
#endif
#if HP_INDENT_MOD
    ContextModel indentmod_;
#endif
#if HP_LISTLEVEL_MOD
    ContextModel listlevelmod_;
#endif
#if HP_ISSE_MOD
    ContextModel issemod_;
#endif
#if HP_MAGIC_MOD
    ContextModel magicmod_;
#endif
#if HP_NOWIKI_MOD
    ContextModel nowikimod_;
#endif
#if HP_TITLE_MOD
    ContextModel titlemod_;
#endif
#if HP_PAGEID_MOD
    ContextModel pageidmod_;
#endif
#if HP_USER_MOD
    ContextModel usermod_;
#endif
#if HP_TEXT_MOD
    ContextModel textmod_;
#endif
#if HP_NS_MOD
    ContextModel nsmod_;
#endif
#if HP_DUMPREDIR_MOD
    ContextModel dumpredirmod_;
#endif
#if HP_IP_MOD
    ContextModel ipmod_;
#endif
#if HP_REVCOMMENT_MOD
    ContextModel revcommentmod_;
#endif
#if HP_MINOR_MOD
    ContextModel minormod_;
#endif
#if HP_WIKIMODEL_MOD
    ContextModel wikimodelmod_;
#endif
#if HP_SECTITLE_MOD
    ContextModel sectitlemod_;
#endif
#if HP_PARSERFN_MOD
    ContextModel parserfnmod_;
#endif
#if HP_TABLECLASS_MOD
    ContextModel tableclassmod_;
#endif
#if HP_ANCHOR_MOD
    ContextModel anchormod_;
#endif
#if HP_PUBID_MOD
    ContextModel pubidmod_;
#endif
#if HP_TEMPPOS_MOD
    ContextModel tempposmod_;
#endif
#if HP_WIKISTACK_MOD
    ContextModel wikistackmod_;
#endif
#if HP_LANG_MOD
    ContextModel langmod_;
#endif
#if HP_CATSORT_MOD
    ContextModel catsortmod_;
#endif
#if HP_TBLROW_MOD
    ContextModel tblrowmod_;
#endif
#if HP_FILEOPT_MOD
    ContextModel fileoptmod_;
#endif
#if HP_DEFAULTSORT_MOD
    ContextModel defaultsortmod_;
#endif
#if HP_REDIRTARGET_MOD
    ContextModel redirtargetmod_;
#endif
#if HP_DAB_MOD
    ContextModel dabmod_;
#endif
#if HP_HATNOTE_MOD
    ContextModel hatnotemod_;
#endif
#if HP_LASTLINK_MOD
    ContextModel lastlinkmod_;
#endif
#if HP_FWORD_MOD
    ContextModel fwordmod_;
#endif
#if HP_YEAR_MOD
    ContextModel yearmod_;
#endif
#if HP_CAPMASK_MOD
    ContextModel capmaskmod_;
#endif
#if HP_CELLTXT_MOD
    ContextModel celltxtmod_;
#endif
#if HP_HTTPHOST_MOD
    ContextModel httphostmod_;
#endif
#if HP_PAREN_MOD
    ContextModel parenmod_;
#endif
#if HP_LISTPOS_MOD
    ContextModel listposmod_;
#endif
#if HP_SHAPE_MOD
    ContextModel shapemod_;
#endif
#if HP_SUFFIX_MOD
    ContextModel suffixmod_;
#endif
#if HP_PREFIX_MOD
    ContextModel prefixmod_;
#endif
#if HP_CHARCLS_MOD
    ContextModel charclsmod_;
#endif
#if HP_VOWEL_MOD
    ContextModel vowelmod_;
#endif
#if HP_CONTR_MOD
    ContextModel contrmod_;
#endif
#if HP_HYPHEN_MOD
    ContextModel hyphenmod_;
#endif
#if HP_TOKENCLS_MOD
    ContextModel tokenclsmod_;
#endif
#if HP_RUNLEN_MOD
    ContextModel runlenmod_;
#endif
#if HP_WPOS_MOD
    ContextModel wposmod_;
#endif
#if HP_BLANK_MOD
    ContextModel blankmod_;
#endif
#if HP_SPRUN_MOD
    ContextModel sprunmod_;
#endif
#if HP_LINELEN_MOD
    ContextModel linelenmod_;
#endif
#if HP_TAGDIST_MOD
    ContextModel tagdistmod_;
#endif
#if HP_MARKDIST_MOD
    ContextModel markdistmod_;
#endif
#if HP_UPPERGAP_MOD
    ContextModel uppergapmod_;
#endif
#if HP_MONTH_MOD
    ContextModel monthmod_;
#endif
#if HP_GALLERY_MOD
    ContextModel gallerymod_;
#endif
#if HP_SECKIND_MOD
    ContextModel seckindmod_;
#endif
#if HP_CITEKIND_MOD
    ContextModel citekindmod_;
#endif
#if HP_TAGNAME_MOD
    ContextModel tagnamemod_;
#endif
#if HP_COLSPAN_MOD
    ContextModel colspanmod_;
#endif
#if HP_STYLE_MOD
    ContextModel stylemod_;
#endif
#if HP_COORD_MOD
    ContextModel coordmod_;
#endif
#if HP_DIGITGAP_MOD
    ContextModel digitgapmod_;
#endif
#if HP_DOTGAP_MOD
    ContextModel dotgapmod_;
#endif
#if HP_COMMAGAP_MOD
    ContextModel commagapmod_;
#endif
#if HP_WORDLEN_MOD
    ContextModel wordlenmod_;
#endif
#if HP_SENTLEN_MOD
    ContextModel sentlenmod_;
#endif
#if HP_LOWERGAP_MOD
    ContextModel lowergapmod_;
#endif
#if HP_DIGITPOS_MOD
    ContextModel digitposmod_;
#endif
#if HP_SLASHGAP_MOD
    ContextModel slashgapmod_;
#endif
#if HP_DIGLEN_MOD
    ContextModel diglenmod_;
#endif
#if HP_PREVLINE_MOD
    ContextModel prevlinemod_;
#endif
#if HP_PREVSENT_MOD
    ContextModel prevsentmod_;
#endif
#if HP_LINKLEN_MOD
    ContextModel linklenmod_;
#endif
#if HP_TPLLEN_MOD
    ContextModel tpllenmod_;
#endif
#if HP_PARALEN_MOD
    ContextModel paralenmod_;
#endif
#if HP_ALNUMLEN_MOD
    ContextModel alnumlenmod_;
#endif
#if HP_SPLEN_MOD
    ContextModel splenmod_;
#endif
#if HP_TITLEWORD_MOD
    ContextModel titlewordmod_;
#endif
#if HP_HEADWORD_MOD
    ContextModel headwordmod_;
#endif
#if HP_INIT_MOD
    ContextModel initmod_;
#endif
#if HP_ORDINAL_MOD
    ContextModel ordinalmod_;
#endif
#if HP_UNIT_MOD
    ContextModel unitmod_;
#endif
#if HP_DECIMAL_MOD
    ContextModel decimalmod_;
#endif
#if HP_REPEAT_MOD
    ContextModel repeatmod_;
#endif
#if HP_CASEFLIP_MOD
    ContextModel caseflipmod_;
#endif
#if HP_LEAD_MOD
    ContextModel leadmod_;
#endif
#if HP_INFOVAL_MOD
    ContextModel infovalmod_;
#endif
#if HP_LINKTRAIL_MOD
    ContextModel linktrailmod_;
#endif
#if HP_CELLKIND_MOD
    ContextModel cellkindmod_;
#endif
#if HP_TBLCOL_MOD
    ContextModel tblcolmod_;
#endif
#if HP_HEADIDX_MOD
    ContextModel headidxmod_;
#endif
#if HP_HTMLFMT_MOD
    ContextModel htmlfmtmod_;
#endif
#if HP_INFOBOX_MOD
    ContextModel infoboxmod_;
#endif
#if HP_SECLEVEL_MOD
    ContextModel seclevelmod_;
#endif
#if HP_BRACE3_MOD
    ContextModel brace3mod_;
#endif
#if HP_NAMEDARG_MOD
    ContextModel namedargmod_;
#endif
#if HP_INCLUDE_MOD
    ContextModel includemod_;
#endif
#if HP_SIG_MOD
    ContextModel sigmod_;
#endif
#if HP_WIKIBOLD_MOD
    ContextModel wikiboldmod_;
#endif
#if HP_URLPART_MOD
    ContextModel urlpartmod_;
#endif
#if HP_REFIDX_MOD
    ContextModel refidxmod_;
#endif
    MatchModel match_[kMatchModels];
#if HP_SPARSE_UTF8
    MatchModel smatch_;
#endif
#if HP_SKIPK_MOD
    MatchModel skipk_;
#endif
#if HP_SKIP3_MOD
    MatchModel skip3_;
#endif
#if HP_SKIP4_MOD
    MatchModel skip4_;
#endif
#if HP_SKIP5_MOD
    MatchModel skip5_;
#endif
#if HP_LZP_MOD
    LzpModel lzp_;
#endif
#if HP_SR_MOD
    SrModel sr_;
#endif
#if HP_DMC_MOD
    DmcModel dmc_;
#endif
#if HP_WORD_MATCH
    WordMatchModel wmatch_[3 + (HP_WMATCH_4 ? 1 : 0) + (HP_WMATCH_5 ? 1 : 0)];
#endif
    HebbianModel hebb_;
    DiscoveryPool pool_;
    MixerNet mixer_;
    APM apm_c0_, apm_lex_, apm_gria_;
    Hedge hedge_;
    std::array<Counter, 256> bias_{};
    GriaGate gria_;
    WikiMachine wiki_;
    WordStreams streams_;
#if HP_STEMMER || HP_STEM_FOLD || HP_POS_GATE || HP_WT3_CTX
    StemStreams stems_;
#endif
    BracketMachine brackets_;
    PatternCache cache_;
#if HP_NUMERIC
    NumericField numbers_;
#endif
#if HP_GATE_BRANCH
    Branch3 branch3_;
#endif
#if HP_GATE_BREAK
    int break_age_ = 0;
#endif

    std::uint64_t hist_ = 0;
    std::uint64_t word_hash_ = 0;
    std::uint64_t letter_hash_ = 0;
    std::uint64_t hist2_ = 0;
    static constexpr int kLineMax = 256;
    std::uint8_t line_buf_[2][kLineMax] = {{0}};
    int cur_line_idx_ = 0;
    int col_pos_ = 0;
    int tag_depth_ = 0;
    int in_tag_ = 0;
    std::uint64_t tag_name_ = 0;
    std::uint64_t prev_word_ = 0;
    std::uint64_t word_hash_prev_ = 0;
    std::uint64_t word_ring_[4] = {0, 0, 0, 0};
    ContextModel* ctx_chain_[kCtxModels];
    int n_ctx_chain_ = 0;
    int c0_ = 1;
    int bitpos_ = 0;
    int pr_final_ = 2048;
    int exp_p_[kNumExperts + 8] = {0};
    int n_exp_ = 0;
    int mixed_p_ = 2048;
    int last_mlen_ = 0;
    int sparse_ = 0;
#if HP_PRONOUN_MOD
    std::uint8_t pw_[12] = {};
    int pw_n_ = 0;
    int pronoun_ = 0;
#endif
#if HP_UTF8_IDLE || HP_GATE_UTF8
    int utf8left_ = 0;
#endif
};

}  // namespace hp
