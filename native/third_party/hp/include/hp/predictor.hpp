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
#include <optional>

#include "hp/numeric.hpp"
#include "hp/pattern_cache.hpp"
#include "hp/stat_gates.hpp"
#include "hp/statemap.hpp"
#include "hp/wiki.hpp"
#include "hp/wordmatch.hpp"
#include "hp/wordstream.hpp"
#include "hp/sentmem.hpp"
#include "hp/undo.hpp"


namespace hp {

struct Config {
    int table_bits = 22;
    int buf_bits = 26;
    int match_bits = 22;
    int mixer_lr = 2;
    bool gria = true;

    // Lossy knobs (CyphaLM serve tiers). Defaults reproduce gate24 exactly;
    // any non-default value changes predictions, so a model must be read back
    // with the same values it was trained with. Not carried in hp archives.
    std::uint64_t cm_drop = 0;    // bit i: drop context model i (Predictor::CmId); no table, zero input
    int cm_bits_cap = 0;          // >0: cap every context-model table at this many bits
    std::uint32_t gate_drop = 0;  // bit j: drop mixer layer-1 weight set j (Predictor::Gate)
    int mixer_skip = 0;           // >0: skip the mixer update when |err| < this (gate24 = 32)
    int match_bits_cap = 0;       // >0: cap byte-match hash tables (13 models) at this many bits
    int pool_slots = 0;           // 1..11: keep only this many discovered-context slots (gate24 = 12)
    int pool_bits_cap = 0;        // >0: cap discovered-context tables at this many bits
    int hebb_bits_cap = 0;        // >0: cap the Hebbian word-association tables at this many bits
    std::uint32_t match_drop = 0; // bit k: drop byte-match model k (match_[0..8], smatch, skipk, skip3, skip4)

    // Upstream mixer gains (CompressionAlgorithm H33, v91, v93). Defaults
    // reproduce gate24; the trained values are carried in the checkpoint.
    int lr1_scale = 100;          // percent scale on the layer-1 per-set rates (upstream best 40)
    int mixer_scale = 0;          // Q16 multiplier on layer-1 dots, 0 = off (upstream 49152 = 0.75)
    int mixer_skip_l1 = 0;        // skip a layer-1 set's update when its |err| < this (upstream 80)

    // Upstream context models (CompressionAlgorithm hp), off by default.
    // Bit k adds Predictor::ExtraCm k: its table and two mixer inputs. 0 is
    // gate24 exactly (no table, no input, v4 checkpoint). Part of the trained
    // model: a checkpoint loads only into a predictor with the same value.
    std::uint32_t extra_cms = 0;

    // Encoder and decoder must agree. match/buf sizes are a function of
    // table_bits (the only size the archive header carries).
    // buf_bits = table_bits + 3 (25 at mem 22).
    void normalize() {
        if (table_bits < 16) table_bits = 16;
        if (table_bits > 28) table_bits = 28;
        match_bits = table_bits;
        buf_bits = table_bits + 3;
        if (buf_bits > 28) buf_bits = 28;
    }
};

class Predictor {
 public:
    static constexpr int kExtraCtx = 24;  // gate24 wiki / stream context models
    static constexpr int kCtxModels = 11 + kExtraCtx;
    static constexpr int kMatchModels = 9;
    static constexpr int kWordMatch = 4;
    static constexpr int kDiscovered =
        DiscoveryPool::kSlots * ContextModel::kOutputs;
    static constexpr int kBaseExperts =
        kCtxModels * ContextModel::kOutputs + 1 /*bias*/ + kMatchModels
        + kWordMatch + 1 /*sparse utf8*/
        + 1 /*hebb*/ + kDiscovered
        + 5 /*dmc, lzp, skipk, skip3, skip4*/;
    static constexpr int kNumExperts = kBaseExperts;  // gate24 (extra_cms = 0)

    // Optional upstream context models (Config::extra_cms bit k), appended to
    // the context-model chain after the gate24 set, in this order. Upstream
    // 8 MiB enwik8 gains at SLOT_MAX 24 in the comments.
    enum ExtraCm {
        kXWikibold = 0,   // HP_WIKIBOLD_MOD: '' / ''' bold-italic state, salt 159 (-1,703 B)
        kXSentpos,        // HP_SENTPOS_MOD: nth word in the sentence, salt 185 (-461 B)
        kXCappara,        // HP_CAPPARA_MOD: cap_mask x is_paragraph, salt 206 (-256 B)
        kXRefgroup,       // HP_REFGROUP_MOD: <ref> / name= / group= class, salt 211 (-217 B)
        kXStatetrans,     // HP_STATETRANS_MOD: previous x current wiki state, salt 194 (-213 B)
        kXCrossO2Sentmem, // HP_CROSS_STACK cross0: o2 x sentmem_cm live hashes, salt 0x5858
        kXCrossWordBrk,   // HP_CROSS_STACK cross1: word x brk live hashes, salt 0x5859
        kNumExtraCm
    };
    static constexpr std::uint32_t kExtraCmMask = (1u << kNumExtraCm) - 1u;
    static constexpr int kMaxCtxModels = kCtxModels + kNumExtraCm;
    static constexpr int kMaxExperts = kNumExperts + kNumExtraCm * ContextModel::kOutputs;
    /// Extra-CM bits that need the WikiExtra trackers.
    static constexpr std::uint32_t kExtraWikiMask =
        (1u << kXWikibold) | (1u << kXSentpos) | (1u << kXRefgroup) | (1u << kXStatetrans);
    static int num_extra_cms(std::uint32_t extra_cms) {
        int n = 0;
        for (int k = 0; k < kNumExtraCm; ++k) n += (extra_cms >> k) & 1u;
        return n;
    }
    /// Mixer input count for ``cfg``.
    static int num_experts(const Config& cfg) {
        return kNumExperts + num_extra_cms(cfg.extra_cms) * ContextModel::kOutputs;
    }

    enum Gate {
        kGateC0 = 0, kGateAlpha, kGatePrev, kGateMatch, kGateEntropy,
        kGateHebb,
        kGateWiki, kGatePattern,
        kGateArgmax,
        kGateWordPos,
        kNumGates
    };

    // Context models in ctx_chain_ order (bit index for Config::cm_drop).
    enum CmId {
        kCmO1 = 0,
        kCmO2,
        kCmO3,
        kCmO4,
        kCmO6,
        kCmWord,
        kCmSp13,
        kCmSp24,
        kCmCol,
        kCmTag,
        kCmWbi,
        kCmWstrSp,
        kCmBrk,
        kCmLink,
        kCmNum,
        kCmSen,
        kCmSentst,
        kCmSentmemCm,
        kCmSengrp,
        kCmNestMod,
        kCmParaMod,
        kCmLineMod,
        kCmStateMod,
        kCmTplMod,
        kCmInfokeyMod,
        kCmO6b,
        kCmLinkpipeMod,
        kCmCatMod,
        kCmHeadingMod,
        kCmTitleMod,
        kCmSectitleMod,
        kCmWikistackMod,
        kCmCapmaskMod,
        kCmUppergapMod,
        kCmWordlenMod,
        // Config::extra_cms (ExtraCm order); chained only when enabled.
        kCmWikiboldMod,
        kCmSentposMod,
        kCmCapparaMod,
        kCmRefgroupMod,
        kCmStatetransMod,
        kCmCrossO2Sentmem,
        kCmCrossWordBrk,
        kNumCm
    };
    static const char* cm_name(int i) {
        static const char* const k[] = {"o1", "o2", "o3", "o4", "o6", "word", "sp13", "sp24", "col", "tag", "wbi", "wstr_sp", "brk", "link", "num", "sen", "sentst", "sentmem_cm", "sengrp", "nestmod", "paramod", "linemod", "statemod", "tplmod", "infokeymod", "o6b", "linkpipemod", "catmod", "headingmod", "titlemod", "sectitlemod", "wikistackmod", "capmaskmod", "uppergapmod", "wordlenmod", "wikiboldmod", "sentposmod", "capparamod", "refgroupmod", "statetransmod", "cross_o2_sentmem", "cross_word_brk"};
        return (i >= 0 && i < kNumCm) ? k[i] : "?";
    }

    static_assert(kCmWordlenMod + 1 == kCtxModels, "CmId must list every gate24 context model");
    static_assert(kNumCm == kMaxCtxModels, "CmId must list every optional context model");
    static_assert(kNumCm <= 64, "Config::cm_drop is a 64-bit mask");

    // Table bits for optional context model ``k``: 0 (no table) unless enabled.
    static int extra_cm_bits_(const Config& cfg, int k, int bits) {
        if (!((cfg.extra_cms >> k) & 1u)) return 0;
        return cm_bits_(cfg, kCtxModels + k, bits);
    }

    static int slot_bits(int base, int delta) {
        if (delta < 0) delta = 0;
        int b = base + delta;
        if (delta >= 2) b += 1;
        if (delta == 1) b += 8;
        if (b < 16) b = 16;
        if (b > HP_SLOT_MAX) b = HP_SLOT_MAX;
        return b;
    }

    // Table bits for the byte-match models, capped by Config::match_bits_cap.
    static int byte_match_bits_(const Config& cfg, int id) {
        if ((cfg.match_drop >> id) & 1u) return 0;
        const int b = match_bits(cfg.match_bits);
        return (cfg.match_bits_cap > 0 && b > cfg.match_bits_cap) ? cfg.match_bits_cap : b;
    }

    static int match_bits(int b) {
        b += 1;
        if (b < 16) b = 16;
        if (b > 28) b = 28;
        return b;
    }

    // Table bits for context model ``id``: 0 = dropped, else capped by cm_bits_cap.
    static int cm_bits_(const Config& cfg, int id, int bits) {
        if ((cfg.cm_drop >> id) & 1u) return 0;
        if (cfg.cm_bits_cap > 0 && bits > cfg.cm_bits_cap) return cfg.cm_bits_cap;
        return bits;
    }

    static int add_bits(int b, int extra) {
        b += extra;
        if (b < 16) b = 16;
        if (b > HP_SLOT_MAX) b = HP_SLOT_MAX;
        return b;
    }

    explicit Predictor(const Config& cfg)
        : byte_ring_(cfg.buf_bits),
          cfg_(cfg),
          o1_(cm_bits_(cfg, kCmO1, slot_bits(cfg.table_bits, -2)), 1023),
          o2_(cm_bits_(cfg, kCmO2, slot_bits(cfg.table_bits, -2)), 1023),
          o3_(cm_bits_(cfg, kCmO3, add_bits(slot_bits(cfg.table_bits, 0), 6)), 511),
          o4_(cm_bits_(cfg, kCmO4, add_bits(slot_bits(cfg.table_bits, 0), 6)), 255),
          o6_(cm_bits_(cfg, kCmO6, add_bits(slot_bits(cfg.table_bits, 0), 9)), 127),
          o6b_(cm_bits_(cfg, kCmO6b, add_bits(slot_bits(cfg.table_bits, 0), 9)), 127),
          word_(cm_bits_(cfg, kCmWord, slot_bits(cfg.table_bits, 1)), 255),
          col_(cm_bits_(cfg, kCmCol, add_bits(slot_bits(cfg.table_bits, 0), 1)), 255),
          tag_(cm_bits_(cfg, kCmTag, slot_bits(cfg.table_bits, 2)), 255),
          wbi_(cm_bits_(cfg, kCmWbi, slot_bits(cfg.table_bits, 1)), 255),
          sp13_(cm_bits_(cfg, kCmSp13, slot_bits(cfg.table_bits, 0)), 255),
          sp24_(cm_bits_(cfg, kCmSp24, slot_bits(cfg.table_bits, 0)), 255),
          wstr_sp_(cm_bits_(cfg, kCmWstrSp, slot_bits(cfg.table_bits, 2)), 255),
          brk_(cm_bits_(cfg, kCmBrk, slot_bits(cfg.table_bits, 0)), 255),
          link_(cm_bits_(cfg, kCmLink, slot_bits(cfg.table_bits, 2)), 255),
          num_(cm_bits_(cfg, kCmNum, slot_bits(cfg.table_bits, 0)), 255),
          sen_(cm_bits_(cfg, kCmSen, slot_bits(cfg.table_bits, 2)), 255),
          sentst_(cm_bits_(cfg, kCmSentst, slot_bits(cfg.table_bits, 3)), 255),
          sentmem_cm_(cm_bits_(cfg, kCmSentmemCm, slot_bits(cfg.table_bits, 0)), 255),
          sengrp_(cm_bits_(cfg, kCmSengrp, slot_bits(cfg.table_bits, 2)), 255),
          nestmod_(cm_bits_(cfg, kCmNestMod, cfg.table_bits), 255),
          paramod_(cm_bits_(cfg, kCmParaMod, cfg.table_bits), 255),
          linemod_(cm_bits_(cfg, kCmLineMod, cfg.table_bits), 255),
          statemod_(cm_bits_(cfg, kCmStateMod, cfg.table_bits), 255),
          tplmod_(cm_bits_(cfg, kCmTplMod, cfg.table_bits), 255),
          infokeymod_(cm_bits_(cfg, kCmInfokeyMod, cfg.table_bits), 255),
          linkpipemod_(cm_bits_(cfg, kCmLinkpipeMod, cfg.table_bits), 255),
          catmod_(cm_bits_(cfg, kCmCatMod, cfg.table_bits), 255),
          headingmod_(cm_bits_(cfg, kCmHeadingMod, cfg.table_bits), 255),
          titlemod_(cm_bits_(cfg, kCmTitleMod, cfg.table_bits), 255),
          sectitlemod_(cm_bits_(cfg, kCmSectitleMod, cfg.table_bits), 255),
          wikistackmod_(cm_bits_(cfg, kCmWikistackMod, cfg.table_bits), 255),
          capmaskmod_(cm_bits_(cfg, kCmCapmaskMod, cfg.table_bits), 255),
          uppergapmod_(cm_bits_(cfg, kCmUppergapMod, cfg.table_bits), 255),
          wordlenmod_(cm_bits_(cfg, kCmWordlenMod, cfg.table_bits), 255),
          wikiboldmod_(extra_cm_bits_(cfg, kXWikibold, cfg.table_bits), 255),
          sentposmod_(extra_cm_bits_(cfg, kXSentpos, cfg.table_bits), 255),
          capparamod_(extra_cm_bits_(cfg, kXCappara, cfg.table_bits), 255),
          refgroupmod_(extra_cm_bits_(cfg, kXRefgroup, cfg.table_bits), 255),
          statetransmod_(extra_cm_bits_(cfg, kXStatetrans, cfg.table_bits), 255),
          cross0_(extra_cm_bits_(cfg, kXCrossO2Sentmem, cfg.table_bits), 255),
          cross1_(extra_cm_bits_(cfg, kXCrossWordBrk, cfg.table_bits), 255),
          match_{ {&byte_ring_, byte_match_bits_(cfg, 0), 3},
                  {&byte_ring_, byte_match_bits_(cfg, 1), 4},
                  {&byte_ring_, byte_match_bits_(cfg, 2), 6},
                  {&byte_ring_, byte_match_bits_(cfg, 3), 10},
                  {&byte_ring_, byte_match_bits_(cfg, 4), 16}
                  , {&byte_ring_, byte_match_bits_(cfg, 5), 8}
                  , {&byte_ring_, byte_match_bits_(cfg, 6), 1}
                  , {&byte_ring_, byte_match_bits_(cfg, 7), 2}
                  , {&byte_ring_, byte_match_bits_(cfg, 8), 5}
          },
          smatch_(&byte_ring_, byte_match_bits_(cfg, 9), 4),
          skipk_(&byte_ring_, byte_match_bits_(cfg, 10), 3, 2),
          skip3_(&byte_ring_, byte_match_bits_(cfg, 11), 3, 3),
          skip4_(&byte_ring_, byte_match_bits_(cfg, 12), 3, 4),
          lzp_(match_bits(cfg.match_bits) > 2 ? match_bits(cfg.match_bits) - 2
                                              : match_bits(cfg.match_bits)),
          dmc_(18),
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
              , WordMatchModel(&byte_ring_,
                               cfg.match_bits > 2 ? cfg.match_bits - 2 : cfg.match_bits, 3,
                               cfg.buf_bits > 2 ? cfg.buf_bits - 2 : cfg.buf_bits)
          },
          hebb_(cfg.hebb_bits_cap > 0 && cfg.hebb_bits_cap < cfg.table_bits ? cfg.hebb_bits_cap
                                                                            : cfg.table_bits,
                255),
          pool_(cfg.pool_bits_cap > 0 && cfg.pool_bits_cap < cfg.table_bits ? cfg.pool_bits_cap
                                                                            : cfg.table_bits,
                0xC0FFEEull, cfg.pool_slots),
          mixer_(num_experts(cfg), gate_sizes(), 256, cfg.mixer_lr, scaled_gate_rates(cfg.lr1_scale)),
          apm_c0_(256),
          apm_lex_(256 * 256),
          apm_gria_(GriaGate::kBuckets * 256),
          hedge_(kNumGates),
          bias_() {
        counter_init(bias_.data(), bias_.size());
        for (int i = 0; i < 256; ++i) {
            bias_[static_cast<std::size_t>(i)].p =
                static_cast<std::uint16_t>(english_bit_prior16(i));
        }
        gria_.set_enabled(cfg.gria);
        mixer_.set_lossy(cfg.mixer_skip, cfg.gate_drop);
        mixer_.set_upstream(cfg.mixer_scale, cfg.mixer_skip_l1);
        xcms_ = cfg.extra_cms & kExtraCmMask;
        init_ctx_chain_();
        set_byte_contexts();
    }

    /// Deep copy with pointer rebind into ``dst`` (must share ``dst``'s construction config).
    static void clone_from(const Predictor& o, Predictor& dst) { dst.copy_state_from(o); }

    Predictor(const Predictor&) = delete;
    Predictor& operator=(const Predictor& o) {
        if (this == &o) return *this;
        assign_from_(o);
        return *this;
    }

    Predictor(Predictor&&) = delete;
    Predictor& operator=(Predictor&&) = delete;

    /// Deep copy live state from ``o`` with pointer rebind (preferred over rvalue assign).
    void copy_state_from(const Predictor& o) {
        if (this != &o) {
            assign_from_(o);
        }
    }

    /// Record mutations during ``update(y)``; restore with ``undo_checkpoint(frame)``.
    void update_tracked(int y, UndoFrame& frame) {
        frame.push_predictor(*this, cfg_);
        update(y);
    }

    void undo_checkpoint(const UndoFrame& frame) { frame.pop_predictor(*this); }

    const Config& config() const { return cfg_; }

    /// The last up-to-``n`` bytes of history, oldest first (from the byte
    /// ring, so exact rewinds cover it). Returns how many were written.
    std::size_t recent_bytes(std::uint8_t* out, std::size_t n) const {
        const std::uint32_t pos = byte_ring_.pos();
        std::size_t k = n;
        if (k > pos) k = pos;
        if (k > static_cast<std::size_t>(byte_ring_.mask()) + 1) k = static_cast<std::size_t>(byte_ring_.mask()) + 1;
        for (std::size_t i = 0; i < k; ++i) out[i] = byte_ring_.at(pos - static_cast<std::uint32_t>(k - i));
        return k;
    }

    /// Serve-time RAM cut sized per table: fold each context-model and
    /// byte-match table by halves while its projected occupancy (two folded
    /// slots in use -> one: 1 - (1 - occ)^2) stays at or under
    /// ``max_occupancy``. Sparse tables (e.g. a skip model at 1%) shrink a
    /// lot, busy ones not at all. Tables keep >= ``min_bits`` bits. Returns
    /// the bytes freed.
    std::size_t fold_auto(double max_occupancy, int min_bits = 12) {
        std::size_t freed = 0;
        auto projected = [](double o) { return 1.0 - (1.0 - o) * (1.0 - o); };
        for (int i = 0; i < n_ctx_chain_; ++i) {
            ContextModel& m = *ctx_chain_[i];
            int bits = m.table_bits();
            if (bits == 0) continue;
            double occ = m.occupancy();
            int target = bits;
            while (target > min_bits && projected(occ) <= max_occupancy) {
                occ = projected(occ);
                --target;
            }
            if (target < bits) {
                freed += ((std::size_t{1} << bits) - (std::size_t{1} << target)) * sizeof(std::uint16_t);
                m.fold_to(target);
            }
        }
        MatchModel* ms[] = {&match_[0], &match_[1], &match_[2], &match_[3], &match_[4], &match_[5], &match_[6],
                            &match_[7], &match_[8], &smatch_,   &skipk_,    &skip3_,    &skip4_};
        for (MatchModel* mm : ms) {
            int bits = mm->table_bits();
            if (bits == 0) continue;
            double occ = mm->occupancy();
            int target = bits;
            while (target > min_bits && projected(occ) <= max_occupancy) {
                occ = projected(occ);
                --target;
            }
            if (target < bits) {
                freed += ((std::size_t{1} << bits) - (std::size_t{1} << target)) * sizeof(std::uint32_t);
                mm->fold_to(target);
            }
        }
        return freed;
    }

    /// Serve-time RAM cut: shrink trained tables to the caps in ``target``
    /// (cm_bits_cap, match_bits_cap, pool_bits_cap; 0 = keep) by folding, and
    /// adopt those caps, so the result saves and reloads like a model built
    /// with them. Other Config fields must match.
    void fold_tables(const Config& target) {
        for (int i = 0; i < n_ctx_chain_; ++i) {
            ContextModel& m = *ctx_chain_[i];
            if ((target.cm_drop >> ctx_chain_id_[i]) & 1u) {
                m.drop();
                continue;
            }
            if (target.cm_bits_cap > 0 && m.table_bits() > target.cm_bits_cap) m.fold_to(target.cm_bits_cap);
        }
        cfg_.cm_drop |= target.cm_drop;
        {
            MatchModel* ms[] = {&match_[0], &match_[1], &match_[2], &match_[3], &match_[4],
                                &match_[5], &match_[6], &match_[7], &match_[8], &smatch_,
                                &skipk_,    &skip3_,    &skip4_};
            static_assert(kMatchModels == 9, "byte-match model list");
            for (int k = 0; k < 13; ++k) {
                if ((target.match_drop >> k) & 1u) ms[k]->drop();
                else if (target.match_bits_cap > 0) ms[k]->fold_to(target.match_bits_cap);
            }
            cfg_.match_drop |= target.match_drop;
        }
        if (target.pool_bits_cap > 0) pool_.fold_to(target.pool_bits_cap);
        if (target.hebb_bits_cap > 0) hebb_.fold_to(target.hebb_bits_cap);
        cfg_.hebb_bits_cap = target.hebb_bits_cap > 0 ? target.hebb_bits_cap : cfg_.hebb_bits_cap;
        cfg_.cm_bits_cap = target.cm_bits_cap > 0 ? target.cm_bits_cap : cfg_.cm_bits_cap;
        cfg_.match_bits_cap = target.match_bits_cap > 0 ? target.match_bits_cap : cfg_.match_bits_cap;
        cfg_.pool_bits_cap = target.pool_bits_cap > 0 ? target.pool_bits_cap : cfg_.pool_bits_cap;
    }

    int predict() {
        mixer_.reset_inputs();
        const int bias_p = counter_predict_p(bias_[c0_]);
        mixer_.add(stretch(bias_p));
        n_exp_ = 0;

        int out[ContextModel::kOutputs];
        int backoff = bias_p;
        int mlen = 0;
        int wml = 0;
        for (int i = 0; i < n_ctx_chain_; ++i) {
            ctx_chain_[i]->predict(c0_, backoff, out);
            for (int j = 0; j < ContextModel::kOutputs; ++j) {
                mixer_.add(out[j]);
                exp_p_[n_exp_++] = squash(out[j]);
            }
            if (i < 4) backoff = ctx_chain_[i]->last_p();
            else if (i == 4) backoff = o1_.last_p();
        }
        for (int i = 0; i < kMatchModels; ++i) {
            const int ms = match_[i].predict(c0_, bitpos_);
            mixer_.add(ms);
            exp_p_[n_exp_++] = squash(ms);
            const int l = match_[i].match_len();
            if (l > mlen) mlen = l;
        }
        {
            const int ms = smatch_.predict(c0_, bitpos_);
            mixer_.add(ms);
            exp_p_[n_exp_++] = squash(ms);
        }
        {
            const int ms = skipk_.predict(c0_, bitpos_);
            mixer_.add(ms);
            exp_p_[n_exp_++] = squash(ms);
        }
        {
            const int ms = skip3_.predict(c0_, bitpos_);
            mixer_.add(ms);
            exp_p_[n_exp_++] = squash(ms);
        }
        {
            const int ms = skip4_.predict(c0_, bitpos_);
            mixer_.add(ms);
            exp_p_[n_exp_++] = squash(ms);
        }
        {
            const int ms = lzp_.predict(c0_, bitpos_);
            mixer_.add(ms);
            exp_p_[n_exp_++] = squash(ms);
        }
        {
            const int ms = dmc_.predict();
            mixer_.add(ms);
            exp_p_[n_exp_++] = squash(ms);
        }
        for (int i = 0; i < kWordMatch; ++i) {
            const int ms = wmatch_[i].predict(c0_, bitpos_);
            mixer_.add(ms);
            exp_p_[n_exp_++] = squash(ms);
            const int l = wmatch_[i].match_len();
            if (l > mlen) mlen = l;
            if (l > wml) wml = l;
        }
        {
            const int hs = hebb_.predict(c0_);
            mixer_.add(hs);
            exp_p_[n_exp_++] = squash(hs);
        }
        {
            int dout[kDiscovered];
            pool_.predict(c0_, o1_.last_p(), dout);
            for (int i = 0; i < kDiscovered; ++i) {
                mixer_.add(dout[i]);
                exp_p_[n_exp_++] = squash(dout[i]);
            }
        }
        sparse_ = o6_.sparsity();


        mixer_.set_ctx(kGateC0, c0_);
        mixer_.set_ctx(kGateAlpha, gria_.bucket());
        mixer_.set_ctx(kGatePrev, static_cast<int>(hist_ & 0xff));
        mixer_.set_ctx(kGateMatch, mlen > 31 ? 31 : mlen);
        last_mlen_ = mlen;
        mixer_.set_ctx(kGateHebb, hebb_.strength() > 15 ? 15 : hebb_.strength());
        mixer_.set_ctx(kGateEntropy, gria_.entropy_bucket());
        mixer_.set_ctx(kGateWiki, wiki_.state() + (wiki_.is_paragraph() << 4));
        mixer_.set_ctx(kGatePattern, cache_.cls());
        mixer_.set_ctx(kGateArgmax, argmax_bin(exp_p_, n_exp_));
        mixer_.set_ctx(kGateWordPos, streams_.word_len() > 15 ? 15 : streams_.word_len());
        mixer_.set_ctx2(c0_);
        int pr = mixer_.mix();

        for (int j = 0; j < mixer_.num_layer1(); ++j)
            hedge_.set(j, mixer_.layer1_p(j));
        const int ph = hedge_.mix();
        pr = (pr + ph) >> 1;
        mixed_p_ = pr;

        const int a = apm_c0_.refine(pr, c0_);
        const int b = apm_lex_.refine(pr, static_cast<int>(hist_ & 0xff) * 256 + c0_);
        const int g = apm_gria_.refine(pr, gria_.bucket() * 256 + c0_);
        pr_final_ = clamp_int((0 * pr + 1 * a + 5 * b + 2 * g) >> 3, 1, 4094);
        return pr_final_;
    }

    /// Learning on (default) trains every table, counter, mixer and APM on each
    /// bit. Off: bits only advance context (history, hashes, match pointers, DMC
    /// position); nothing learned changes and no hash slot is claimed. CyphaLM
    /// turns it off while consuming its own generated bytes, so a model cannot
    /// reinforce what it just said.
    void set_learning(bool on) {
        learning_ = on;
        for (int i = 0; i < n_ctx_chain_; ++i) ctx_chain_[i]->set_frozen(!on);
        pool_.set_frozen(!on);
    }
    bool learning() const { return learning_; }

    /// Serve-time adaptation speed: mixer learning rates = trained x num/den,
    /// small-error skip threshold = ``skip`` (<0: trained). Idempotent, runtime
    /// only; (1, 1, -1) restores training behaviour.
    void set_serve_adaptation(int num, int den, int skip) { mixer_.set_rate_scale(num, den, skip); }


    /// Hash of everything that learns: context/pool/Hebbian tables and StateMaps,
    /// match/LZP/word-match counters, DMC graph, mixer, APMs, hedge, bias
    /// counters. Excludes context state (history, hashes, match history index,
    /// pointers), so consuming bytes with learning off must leave it unchanged.
    std::uint64_t learned_digest() const {
        std::uint64_t h = 0xcbf29ce484222325ull;
        for (int i = 0; i < n_ctx_chain_; ++i) h = ctx_chain_[i]->learned_digest(h);
        for (int i = 0; i < kMatchModels; ++i) h = match_[i].learned_digest(h);
        h = smatch_.learned_digest(h);
        h = skipk_.learned_digest(h);
        h = skip3_.learned_digest(h);
        h = skip4_.learned_digest(h);
        h = lzp_.learned_digest(h);
        h = dmc_.learned_digest(h);
        for (int i = 0; i < kWordMatch; ++i) h = wmatch_[i].learned_digest(h);
        h = hebb_.learned_digest(h);
        h = pool_.learned_digest(h);
        h = mixer_.learned_digest(h);
        h = apm_c0_.learned_digest(h);
        h = apm_lex_.learned_digest(h);
        h = apm_gria_.learned_digest(h);
        h = hedge_.learned_digest(h);
        return fnv_bytes(h, bias_.data(), bias_.size() * sizeof(bias_[0]));
    }

    void update(int y) {
        gria_.account_bit(y ? pr_final_ : 4096 - pr_final_);
        if (!learning_) {
            dmc_.advance(y);
            advance_bit_(y);
            return;
        }

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
        wstr_sp_.update(y, ens);
        brk_.update(y, ens);
        link_.update(y, ens);
        num_.update(y, ens);
        sen_.update(y, ens);
        sentst_.update(y, ens);
        sentmem_cm_.update(y, ens);
        sengrp_.update(y, ens);
        nestmod_.update(y, ens);
        paramod_.update(y, ens);
        linemod_.update(y, ens);
        statemod_.update(y, ens);
        tplmod_.update(y, ens);
        infokeymod_.update(y, ens);
        o6b_.update(y, ens);
        linkpipemod_.update(y, ens);
        catmod_.update(y, ens);
        headingmod_.update(y, ens);
        titlemod_.update(y, ens);
        sectitlemod_.update(y, ens);
        wikistackmod_.update(y, ens);
        capmaskmod_.update(y, ens);
        uppergapmod_.update(y, ens);
        wordlenmod_.update(y, ens);
        for (int i = kCtxModels; i < n_ctx_chain_; ++i) ctx_chain_[i]->update(y, ens);
        for (int i = 0; i < kMatchModels; ++i) match_[i].update(y);
        smatch_.update(y);
        skipk_.update(y);
        skip3_.update(y);
        skip4_.update(y);
        lzp_.update(y);
        dmc_.update(y);
        for (int i = 0; i < kWordMatch; ++i) wmatch_[i].update(y);
        hebb_.update(y);
        pool_.update(y, y ? (4096 - pr_final_) >> 4 : pr_final_ >> 4);
        advance_bit_(y);
    }

    void advance_bit_(int y) {
        hp_undo_note(c0_);
        hp_undo_note(bitpos_);
        c0_ = (c0_ << 1) | y;
        ++bitpos_;
        if (bitpos_ == 8) {
            const int byte = c0_ & 0xff;
            hp_undo_note(c0_);
            hp_undo_note(bitpos_);
            c0_ = 1;
            bitpos_ = 0;
            if (UndoRecorderScope::active() == nullptr ||
                UndoRecorderScope::active()->records_byte_end()) {
                end_of_byte(byte);
            }
        }
    }

    /// Lossy serve: reset context hash slots with fewer than ``min_total`` bit
    /// observations (n0+n1) across all chained context models and cold bias counters.
    void prune_cold_hash_slots(int min_total) {
        if (min_total <= 0) return;
        for (int i = 0; i < n_ctx_chain_; ++i) {
            ctx_chain_[i]->prune_cold_hash_slots(min_total);
        }
        for (auto& c : bias_) {
            if (c.n < min_total) {
                c.p = 32768;
                c.n = 0;
            }
        }
    }

    const GriaGate& gria() const { return gria_; }
    int discovery_replacements() const { return pool_.replaced(); }
    int cache_hits() const {
        return 0;
    }
    int cache_lookups() const {
        return 0;
    }
    const PatternCache& patterns() const { return cache_; }

    int expert_count() const { return n_exp_; }
    int expert_p(int i) const { return exp_p_[i]; }
    int mixed_p() const { return mixed_p_; }
    int sparse_fraction() const { return sparse_; }

    /// Binary checkpoint (Cypha train-once / serve-many). Format: ``HPCP`` v1.
    void write_checkpoint(std::ostream& os) const;
    void read_checkpoint(std::istream& is);

    int mixer_n() const { return mixer_.num_layer1(); }
    int mixer_dot(int j) const { return mixer_.layer1_dot(j); }
    int mixer_p(int j) const { return mixer_.layer1_p(j); }
    int c0() const { return c0_; }
    int wiki_state() const { return wiki_.state(); }
    int entropy_bucket() const { return gria_.entropy_bucket(); }
    int last_match_len() const { return last_mlen_; }

    /// Weighted merge of additive hp tables from an independently trained shard.
    void merge_shard_tables(const Predictor& src, std::uint64_t src_bytes,
                            std::uint64_t dst_bytes);

    /// Weighted merge with optional confidence gating (see ``hp/shard_merge.hpp``).
    void merge_shard_tables(const Predictor& src, std::uint64_t src_bytes,
                            std::uint64_t dst_bytes, const ShardMergeOptions& opts);

    /// Copy mergeable tables into a fresh predictor (runtime path state unchanged).
    void transfer_tables_from(const Predictor& src);

    /// Clear path-dependent runtime state; learned tables are preserved.
    /// keep_history: keep the byte ring (the text match models copy from).
    void reset_stream_state(bool keep_history = false);

    /// Rebind match/wordstream internal pointers after copy (bit-tree scratch fork).
    void rebind_streams() { rebind_internal_pointers_(); }

 private:
    static const std::vector<int>& gate_sizes() {
        static const std::vector<int> s = [] {
        std::vector<int> out = {256, GriaGate::kBuckets, 256, 32,
                              GriaGate::kEntBuckets, 16};
        out.push_back(32);   // wiki state × isParagraph
        out.push_back(PatternCache::kNClass);
        out.push_back(16);
        out.push_back(16);
        return out;
        }();
        return s;
    }

    static const std::vector<int>& gate_rates(int base) {
        (void)base;
        static const std::vector<int> r = [] {
        std::vector<int> out = {2, 3, 2, 4, 3, 4};
        out.push_back(3);
        out.push_back(3);
        out.push_back(3);
        out.push_back(3);
        return out;
        }();
        return r;
    }

    // Layer-1 rates x lr1_scale / 100, rounded, at least 1 (upstream HP_LR1_SCALE).
    static std::vector<int> scaled_gate_rates(int lr1_scale) {
        std::vector<int> out = gate_rates(0);
        if (lr1_scale != 100) {
            for (int& r : out) {
                const int v = (r * lr1_scale + 50) / 100;
                r = v < 1 ? 1 : v;
            }
        }
        return out;
    }

    static int backoff_kt_(const Counter& c) {
        return counter_predict_p(c);
    }


    std::uint32_t h2(std::uint64_t salt, std::uint64_t key) {
        return cache_.hash_memo(salt, key);
    }

    void end_of_byte(int byte) {
        hp_undo_note(hist_);
        hist_ = (hist_ << 8) | static_cast<std::uint64_t>(byte);

        const bool alnum = (byte >= 'a' && byte <= 'z') ||
                           (byte >= 'A' && byte <= 'Z') ||
                           (byte >= '0' && byte <= '9');
        const int at_boundary = (!alnum && word_hash_ != 0) ? 1 : 0;
        const bool letter = (byte >= 'a' && byte <= 'z') ||
                            (byte >= 'A' && byte <= 'Z');
        if (alnum) {
            hp_undo_note(word_hash_);
            word_hash_ = mix64(word_hash_ * 0x100000001B3ull +
                               static_cast<std::uint64_t>(byte | 0x20));
        } else {
            hp_undo_note(word_hash_);
            word_hash_ = 0;
        }
        if (letter) {
            hp_undo_note(letter_hash_);
            letter_hash_ = mix64(letter_hash_ * 0x100000001B3ull +
                                 static_cast<std::uint64_t>(byte | 0x20));
        } else {
            hp_undo_note(letter_hash_);
            letter_hash_ = 0;
        }

        if (byte == '\n') {
            if (col_pos_ < kLineMax)
                std::memset(line_buf_[cur_line_idx_] + col_pos_, 0,
                            static_cast<std::size_t>(kLineMax - col_pos_));
            hp_undo_note(cur_line_idx_);
            cur_line_idx_ ^= 1;
            hp_undo_note(col_pos_);
            col_pos_ = 0;
        } else {
            if (col_pos_ < kLineMax) {
                hp_undo_note(line_buf_[cur_line_idx_][col_pos_]);
                line_buf_[cur_line_idx_][col_pos_] = static_cast<std::uint8_t>(byte);
            }
            if (col_pos_ < kLineMax - 1) {
                hp_undo_note(col_pos_);
                ++col_pos_;
            }
        }

        wiki_.push(byte, (xcms_ & kExtraWikiMask) ? &wx_ : nullptr);

        brackets_.push(byte);
        streams_.push(byte, alnum);
        if (at_boundary) sentmem_.push_word(word_hash_prev_);
        if (byte == '.' || byte == '!' || byte == '?' || byte == '\n')
            sentmem_.end_sentence();
        numbers_.push(byte);

        if (!alnum && word_hash_prev_ != 0) {
            if (learning_) hebb_.potentiate(prev_word_, word_hash_prev_);
            prev_word_ = word_hash_prev_;
            word_ring_[3] = word_ring_[2];
            word_ring_[2] = word_ring_[1];
            word_ring_[1] = word_ring_[0];
            word_ring_[0] = prev_word_;
        }
        word_hash_prev_ = word_hash_;
        hebb_.set_context(prev_word_);
        hp_undo_note(hist2_);
        hist2_ = (hist2_ << 8) | ((hist_ >> 56) & 0xffull);
        if (learning_) pool_.end_byte();
        pool_.set_contexts(hist_, hist2_);

        byte_ring_.push(static_cast<std::uint8_t>(byte));
        for (int i = 0; i < kMatchModels; ++i) match_[i].push_byte(byte, hist_);
        {
            std::uint64_t sh = 0;
            for (int i = 0; i < 4; ++i)
                sh = (sh << 8) | ((hist_ >> (16 * i)) & 0xffull);
            smatch_.push_byte(byte, sh);
        }
        skipk_.push_byte(byte, hist_);
        skip3_.push_byte(byte, hist_);
        skip4_.push_byte(byte, hist_);
        lzp_.push_byte(byte, hist_);
        // fx2 keys: {0} current, {1,3} current-alt + word-before-prev, {7,2} letters + last
        const std::uint64_t w0 = word_hash_ ? word_hash_ : word_ring_[0];
        const std::uint64_t w13 = mix64(word_hash_ * 263ull + word_ring_[1]);
        const std::uint64_t w72 = mix64(letter_hash_ * 997ull + word_ring_[0]);
        const std::uint64_t w123 = mix64(word_hash_ * 31ull + word_ring_[0] * 17ull +
                                         word_ring_[1]);
        std::uint64_t whist[5] = {w0, w13, w72, 0, 0};
        int nw = 3;
        whist[nw++] = w123;
        (void)nw;
        for (int i = 0; i < kWordMatch; ++i)
            wmatch_[i].push_byte(byte, whist[i], at_boundary);

        cache_.observe_byte(byte, wiki_.state(), wiki_.in_table(),
                            streams_.first_class());

        gria_.account_byte(byte);
        set_byte_contexts();
    }

    void set_byte_contexts() {
        const int col = (col_pos_ < kLineMax) ? col_pos_ : kLineMax - 1;
        {
            std::uint64_t ck;
                ck = (static_cast<std::uint64_t>(line_buf_[cur_line_idx_ ^ 1][col]) << 16) |
                     static_cast<std::uint64_t>(col & 63);
            ck += static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull;
            col_.set_context(h2(21, ck));
        }
        tag_.set_context(h2(22, wiki_.context_key()
            + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
        ));
        wbi_.set_context(h2(23, prev_word_ * 0x9E3779B97F4A7C15ull + word_hash_ +
                               static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull));

        o1_.set_context(h2(1, hist_ & 0xffull));
        o2_.set_context(h2(2, hist_ & 0xffffull));
        o3_.set_context(h2(3, hist_ & 0xffffffull));
        o4_.set_context(h2(4, hist_ & 0xffffffffull));
        o6_.set_context(h2(6, hist_ & 0xffffffffffffull));
        word_.set_context(h2(7, word_hash_));
        sp13_.set_context(h2(8, (((hist_ >> 0) & 0xffull) |
                                   (((hist_ >> 16) & 0xffull) << 8))
                                   + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
                                   ));
        sp24_.set_context(h2(9, (((hist_ >> 8) & 0xffull) |
                                   (((hist_ >> 24) & 0xffull) << 8))
                                   + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
                                   ));
        wstr_sp_.set_context(h2(25, streams_.prev0() * 131ull + streams_.stream(2) +
                               static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull));
        {
            std::uint64_t bk = brackets_.context_key();
            bk += static_cast<std::uint64_t>(brackets_.quote()) << 24;
            bk += static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull;
            brk_.set_context(h2(26, bk));
        }
        {
            const std::uint64_t lw = wiki_.linkword();
            link_.set_context(h2(29, lw
            ));
            {
                const std::uint64_t sw = wiki_.senword();
                sen_.set_context(h2(31, (sw ? sw * 1471ull + (hist_ & 0xffull) : 0)
                    + static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull
                ));
            }
        }
        sentst_.set_context(h2(34, streams_.stream(3) * 83ull + (hist_ & 0xffull) +
                               static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull));
        sentmem_cm_.set_context(h2(35, sentmem_.match_hash() * 53ull +
                                       static_cast<std::uint64_t>(sentmem_.word_pos()) +
                                       (hist_ & 0xffull) +
                                       static_cast<std::uint64_t>(wiki_.sen_group()) * 131ull));
        num_.set_context(h2(30, numbers_.context_key()
        ));
        sengrp_.set_context(h2(36, static_cast<std::uint64_t>(wiki_.sen_group()) +
                                   ((hist_ & 0xffffffull) << 8) + word_hash_ * 17ull));
        nestmod_.set_context(h2(37, static_cast<std::uint64_t>(wiki_.nest_markup()) +
                                    ((hist_ & 0xffffffull) << 8)));
        paramod_.set_context(h2(38, static_cast<std::uint64_t>(wiki_.is_paragraph()) +
                                    ((hist_ & 0xffffffull) << 8)));
        linemod_.set_context(h2(39, static_cast<std::uint64_t>(wiki_.line_kind() & 255) +
                                    ((hist_ & 0xffffffull) << 8)));
        statemod_.set_context(h2(40, static_cast<std::uint64_t>(wiki_.state()) +
                                     ((hist_ & 0xffffffull) << 8)));
        tplmod_.set_context(h2(45, wiki_.tpl_name() + ((hist_ & 0xffffffull) << 8)));
        infokeymod_.set_context(h2(46, wiki_.infokey() + ((hist_ & 0xffffffull) << 8)));
        o6b_.set_context(h2(106, hist_ & 0xffffffffffffull));
        linkpipemod_.set_context(h2(51, wiki_.link_disp() +
                                        ((hist_ & 0xffffffull) << 8) +
                                        static_cast<std::uint64_t>(wiki_.after_pipe()) * 131ull));
        catmod_.set_context(h2(53, wiki_.cat_ns() + ((hist_ & 0xffffffull) << 8)));
        headingmod_.set_context(h2(55, static_cast<std::uint64_t>(wiki_.heading_level()) +
                                       ((hist_ & 0xffffffull) << 8)));
        titlemod_.set_context(h2(65, wiki_.page_title() + ((hist_ & 0xffffffull) << 8)));
        sectitlemod_.set_context(h2(75, wiki_.sectitle() +
                                         ((hist_ & 0xffffffull) << 8)));
        wikistackmod_.set_context(h2(81, wiki_.wiki_stack() +
                                         ((hist_ & 0xffffffull) << 8)));
        capmaskmod_.set_context(h2(93, static_cast<std::uint64_t>(wiki_.cap_mask()) +
                                       ((hist_ & 0xffffffull) << 8)));
        uppergapmod_.set_context(h2(113, static_cast<std::uint64_t>(wiki_.upper_gap()) +
                                         ((hist_ & 0xffffffull) << 8)));
        wordlenmod_.set_context(h2(125, static_cast<std::uint64_t>(wiki_.word_len()) +
                                        ((hist_ & 0xffffffull) << 8)));
        if (xcms_) set_extra_contexts_();
    }

    // Optional upstream context models: hashing as upstream hp (salts, keys).
    // Only enabled ones are hashed, so extra_cms = 0 leaves the hash memo alone.
    void set_extra_contexts_() {
        const std::uint64_t h3 = (hist_ & 0xffffffull) << 8;
        if ((xcms_ >> kXWikibold) & 1u)
            wikiboldmod_.set_context(h2(159, static_cast<std::uint64_t>(wx_.wikibold) + h3));
        if ((xcms_ >> kXSentpos) & 1u)
            sentposmod_.set_context(h2(185, static_cast<std::uint64_t>(wx_.sentpos) + h3));
        if ((xcms_ >> kXStatetrans) & 1u)
            statetransmod_.set_context(
                h2(194, static_cast<std::uint64_t>(wx_.state_trans(wiki_.state())) + h3));
        if ((xcms_ >> kXCappara) & 1u)
            capparamod_.set_context(h2(206, static_cast<std::uint64_t>(wiki_.cap_mask()) +
                                                (static_cast<std::uint64_t>(wiki_.is_paragraph()) << 8) +
                                                ((hist_ & 0xffffffull) << 16)));
        if ((xcms_ >> kXRefgroup) & 1u)
            refgroupmod_.set_context(h2(211, static_cast<std::uint64_t>(wx_.refgroup) + h3));
        // HP_CROSS_STACK (upstream bind_one_cross_): mix the live hashes of two
        // chained models, bound after every other context is set.
        if ((xcms_ >> kXCrossO2Sentmem) & 1u) bind_cross_(cross0_, o2_, sentmem_cm_, 0x5858u);
        if ((xcms_ >> kXCrossWordBrk) & 1u) bind_cross_(cross1_, word_, brk_, 0x5859u);
    }

    void bind_cross_(ContextModel& cm, const ContextModel& a, const ContextModel& b,
                     std::uint32_t salt) {
        const std::uint64_t mix =
            (static_cast<std::uint64_t>(a.last_h()) * 0x9E3779B97F4A7C15ull) ^
            (static_cast<std::uint64_t>(b.last_h()) * 0xBF58476D1CE4E5B9ull);
        cm.set_context(h2(salt, mix));
    }

    void rebind_internal_pointers_() {
        init_ctx_chain_();
        for (int i = 0; i < kMatchModels; ++i) match_[i].set_ring(&byte_ring_);
        smatch_.set_ring(&byte_ring_);
        skipk_.set_ring(&byte_ring_);
        skip3_.set_ring(&byte_ring_);
        skip4_.set_ring(&byte_ring_);
        for (int i = 0; i < static_cast<int>(sizeof(wmatch_) / sizeof(wmatch_[0])); ++i) {
            wmatch_[i].set_ring(&byte_ring_);
        }
        set_byte_contexts();
    }

    void assign_from_(const Predictor& o) {
        learning_ = o.learning_;
        byte_ring_ = o.byte_ring_;
        o1_ = o.o1_;
        o2_ = o.o2_;
        o3_ = o.o3_;
        o4_ = o.o4_;
        o6_ = o.o6_;
        o6b_ = o.o6b_;
        word_ = o.word_;
        col_ = o.col_;
        tag_ = o.tag_;
        wbi_ = o.wbi_;
        sp13_ = o.sp13_;
        sp24_ = o.sp24_;
        wstr_sp_ = o.wstr_sp_;
        brk_ = o.brk_;
        link_ = o.link_;
        num_ = o.num_;
        sen_ = o.sen_;
        sentst_ = o.sentst_;
        sentmem_ = o.sentmem_;
        sentmem_cm_ = o.sentmem_cm_;
        sengrp_ = o.sengrp_;
        nestmod_ = o.nestmod_;
        paramod_ = o.paramod_;
        linemod_ = o.linemod_;
        statemod_ = o.statemod_;
        tplmod_ = o.tplmod_;
        infokeymod_ = o.infokeymod_;
        linkpipemod_ = o.linkpipemod_;
        catmod_ = o.catmod_;
        headingmod_ = o.headingmod_;
        titlemod_ = o.titlemod_;
        sectitlemod_ = o.sectitlemod_;
        wikistackmod_ = o.wikistackmod_;
        capmaskmod_ = o.capmaskmod_;
        uppergapmod_ = o.uppergapmod_;
        wordlenmod_ = o.wordlenmod_;
        wikiboldmod_ = o.wikiboldmod_;
        sentposmod_ = o.sentposmod_;
        capparamod_ = o.capparamod_;
        refgroupmod_ = o.refgroupmod_;
        statetransmod_ = o.statetransmod_;
        cross0_ = o.cross0_;
        cross1_ = o.cross1_;
        xcms_ = o.xcms_;
        wx_ = o.wx_;
        for (int i = 0; i < kMatchModels; ++i) match_[i] = o.match_[i];
        smatch_ = o.smatch_;
        skipk_ = o.skipk_;
        skip3_ = o.skip3_;
        skip4_ = o.skip4_;
        lzp_ = o.lzp_;
        dmc_ = o.dmc_;
        for (int i = 0; i < static_cast<int>(sizeof(wmatch_) / sizeof(wmatch_[0])); ++i) {
            wmatch_[i] = o.wmatch_[i];
        }
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
        brackets_ = o.brackets_;
        cache_ = o.cache_;
        numbers_ = o.numbers_;
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
        rebind_internal_pointers_();
        mixer_.reset_inputs();
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
        ctx_chain_[n_ctx_chain_++] = &wstr_sp_;
        ctx_chain_[n_ctx_chain_++] = &brk_;
        ctx_chain_[n_ctx_chain_++] = &link_;
        ctx_chain_[n_ctx_chain_++] = &num_;
        ctx_chain_[n_ctx_chain_++] = &sen_;
        ctx_chain_[n_ctx_chain_++] = &sentst_;
        ctx_chain_[n_ctx_chain_++] = &sentmem_cm_;
        ctx_chain_[n_ctx_chain_++] = &sengrp_;
        ctx_chain_[n_ctx_chain_++] = &nestmod_;
        ctx_chain_[n_ctx_chain_++] = &paramod_;
        ctx_chain_[n_ctx_chain_++] = &linemod_;
        ctx_chain_[n_ctx_chain_++] = &statemod_;
        ctx_chain_[n_ctx_chain_++] = &tplmod_;
        ctx_chain_[n_ctx_chain_++] = &infokeymod_;
        ctx_chain_[n_ctx_chain_++] = &o6b_;
        ctx_chain_[n_ctx_chain_++] = &linkpipemod_;
        ctx_chain_[n_ctx_chain_++] = &catmod_;
        ctx_chain_[n_ctx_chain_++] = &headingmod_;
        ctx_chain_[n_ctx_chain_++] = &titlemod_;
        ctx_chain_[n_ctx_chain_++] = &sectitlemod_;
        ctx_chain_[n_ctx_chain_++] = &wikistackmod_;
        ctx_chain_[n_ctx_chain_++] = &capmaskmod_;
        ctx_chain_[n_ctx_chain_++] = &uppergapmod_;
        ctx_chain_[n_ctx_chain_++] = &wordlenmod_;
        for (int i = 0; i < n_ctx_chain_; ++i) ctx_chain_id_[i] = static_cast<std::int8_t>(i);
        ContextModel* const extra[kNumExtraCm] = {&wikiboldmod_,   &sentposmod_, &capparamod_,
                                                  &refgroupmod_,   &statetransmod_,
                                                  &cross0_,        &cross1_};
        for (int k = 0; k < kNumExtraCm; ++k) {
            if (!((xcms_ >> k) & 1u)) continue;
            ctx_chain_id_[n_ctx_chain_] = static_cast<std::int8_t>(kCtxModels + k);
            ctx_chain_[n_ctx_chain_++] = extra[k];
        }
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
    Config cfg_;
    ContextModel o1_, o2_, o3_, o4_, o6_;
    ContextModel o6b_;
    ContextModel word_;
    ContextModel col_, tag_, wbi_;
    ContextModel sp13_, sp24_;
    ContextModel wstr_sp_;
    ContextModel brk_;
    ContextModel link_;
    ContextModel num_;
    ContextModel sen_;
    ContextModel sentst_;
    SentenceMemory sentmem_;
    ContextModel sentmem_cm_;
    ContextModel sengrp_;
    ContextModel nestmod_;
    ContextModel paramod_;
    ContextModel linemod_;
    ContextModel statemod_;
    ContextModel tplmod_;
    ContextModel infokeymod_;
    ContextModel linkpipemod_;
    ContextModel catmod_;
    ContextModel headingmod_;
    ContextModel titlemod_;
    ContextModel sectitlemod_;
    ContextModel wikistackmod_;
    ContextModel capmaskmod_;
    ContextModel uppergapmod_;
    ContextModel wordlenmod_;
    // Config::extra_cms (ExtraCm order); tableless unless enabled.
    ContextModel wikiboldmod_;
    ContextModel sentposmod_;
    ContextModel capparamod_;
    ContextModel refgroupmod_;
    ContextModel statetransmod_;
    ContextModel cross0_;
    ContextModel cross1_;
    MatchModel match_[kMatchModels];
    MatchModel smatch_;
    MatchModel skipk_;
    MatchModel skip3_;
    MatchModel skip4_;
    LzpModel lzp_;
    DmcModel dmc_;
    WordMatchModel wmatch_[kWordMatch];
    HebbianModel hebb_;
    DiscoveryPool pool_;
    MixerNet mixer_;
    APM apm_c0_, apm_lex_, apm_gria_;
    Hedge hedge_;
    std::array<Counter, 256> bias_{};
    GriaGate gria_;
    WikiMachine wiki_;
    WordStreams streams_;
    BracketMachine brackets_;
    PatternCache cache_;
    NumericField numbers_;
    std::uint32_t xcms_ = 0;  // Config::extra_cms & kExtraCmMask
    WikiExtra wx_;            // trackers for the optional wiki context models

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
    ContextModel* ctx_chain_[kMaxCtxModels];
    std::int8_t ctx_chain_id_[kMaxCtxModels] = {};  // CmId of each chained model (cm_drop bit)
    bool learning_ = true;
    int n_ctx_chain_ = 0;
    int c0_ = 1;
    int bitpos_ = 0;
    int pr_final_ = 2048;
    // First kExpPBase entries: the v1-v4 checkpoint layout; the tail holds the
    // optional context models' outputs.
    static constexpr int kExpPBase = kNumExperts + 8;
    int exp_p_[kMaxExperts + 8] = {0};
    int n_exp_ = 0;
    int mixed_p_ = 2048;
    int last_mlen_ = 0;
    int sparse_ = 0;
};

/// Exact, cheap rewind of frozen advances (learning off): generate a few
/// bytes on the live predictor, then return it bit-for-bit to where it was.
/// Saves the predictor's inline state (~170 KB) and records every heap write
/// made while advancing (ring, match tables) in an undo frame. Learned tables
/// are not written while learning is off, so nothing else changes. Used for
/// lookahead decoding without copying the model (~1 GB).
/// Relies on no container being resized while frozen.
class StreamRewind {
 public:
    explicit StreamRewind(Predictor& p) : StreamRewind(std::vector<Predictor*>{&p}) {}
    /// Several predictors advanced together (an ensemble): one undo frame.
    explicit StreamRewind(const std::vector<Predictor*>& ps) : ps_(ps) {
        frame_.set_records_byte_end(true);
        for (Predictor* p : ps_) {
            saved_.emplace_back(new unsigned char[sizeof(Predictor)]);
            std::memcpy(saved_.back().get(), static_cast<const void*>(p), sizeof(Predictor));
        }
        scope_.emplace(frame_);
    }
    ~StreamRewind() { scope_.reset(); }

    StreamRewind(const StreamRewind&) = delete;
    StreamRewind& operator=(const StreamRewind&) = delete;

    /// Back to the state at construction. Recording continues.
    void rewind() {
        frame_.restore_patches();
        frame_.clear();
        for (std::size_t i = 0; i < ps_.size(); ++i) {
            std::memcpy(static_cast<void*>(ps_[i]), saved_[i].get(), sizeof(Predictor));
        }
    }

 private:
    std::vector<Predictor*> ps_;
    std::vector<std::unique_ptr<unsigned char[]>> saved_;
    UndoFrame frame_;
    std::optional<UndoRecorderScope> scope_;
};

inline UndoFrame::~UndoFrame() = default;

inline void UndoFrame::clear() {
    patches_.clear();
    sizes_.clear();
    snap_.reset();
    has_snap_ = false;
}

inline void PredictorUndoStack::clear() { frames_.clear(); }

inline UndoFrame& PredictorUndoStack::push_frame() {
    // Frames are referenced by live UndoRecorderScopes while deeper frames are
    // pushed; reserve so emplace_back never relocates them (bit-tree depth <= 17).
    if (frames_.capacity() < kReserve) frames_.reserve(kReserve);
    frames_.emplace_back();
    return frames_.back();
}

inline void PredictorUndoStack::pop_frame(Predictor& pred) {
    if (frames_.empty()) {
        return;
    }
    frames_.back().pop_predictor(pred);
    frames_.pop_back();
}

inline void UndoFrame::push_predictor(const Predictor& p, const Config& cfg) {
    if (!snap_) {
        snap_ = std::make_unique<Predictor>(cfg);
    }
    snap_->copy_state_from(p);
    has_snap_ = true;
}

inline void UndoFrame::pop_predictor(Predictor& p) const {
    if (!patches_.empty()) {
        restore_patches();
        return;
    }
    if (has_snap_ && snap_) {
        p.copy_state_from(*snap_);
    }
}

#include "hp/checkpoint.hpp"

}  // namespace hp
