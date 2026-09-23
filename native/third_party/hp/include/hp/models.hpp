#pragma once
//
// hp/models.hpp — the expert set.
//
// Each model turns "what have I seen before in this context" into a
// probability for the next bit. They are deliberately simple; the point of
// step 0 is a correct, fast, bit-exact skeleton with a real ablation harness,
// not a maximal model zoo. Adding models is the step-2/3 work and it bolts on
// here without touching the coder or the mixer.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <vector>

#include "hp/blob_io.hpp"
#include "hp/hash_table.hpp"
#include "hp/features.hpp"
#include "hp/int_math.hpp"
#include "hp/statemap.hpp"
#include "hp/undo.hpp"

namespace hp {

// ---------------------------------------------------------------------------
// Integer hashing. Splitmix64 finaliser -- fast, well-distributed, exact.
// ---------------------------------------------------------------------------
inline std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline std::uint32_t hash2(std::uint64_t a, std::uint64_t b) {
    return static_cast<std::uint32_t>(mix64(a * 0x100000001B3ull + b));
}

// ---------------------------------------------------------------------------
// Adaptive bit counter
// ---------------------------------------------------------------------------
//
// p is P(bit == 1) in 16-bit. The update is a running mean that decays into a
// fixed-rate EMA once the observation count hits `limit`:
//
//     p += (target - p) / (n + 2)
//
// Integer division truncates toward zero -- deterministic, and the small bias
// it introduces is symmetric and harmless. Low-order contexts get a high
// limit (they are near-stationary and want a long memory); high-order
// contexts get a low limit so they can track local structure.

struct Counter {
    std::uint16_t p;
    std::uint16_t n;
};

/// Confidence gating for shard table merge (Strategy B+).
struct ShardMergeOptions {
    std::uint16_t min_counter_n = 0;
    std::uint16_t min_statemap_count = 0;
};

/// Weighted merge of one counter cell with optional confidence gating.
inline void merge_counter_cell(Counter& dst, const Counter& src, std::uint64_t src_weight,
                               std::uint64_t dst_weight, std::uint16_t min_n = 0) {
    if (src.n == 0) {
        return;
    }
    if (dst.n == 0) {
        dst = src;
        return;
    }
    if (min_n > 0) {
        if (src.n < min_n) {
            return;
        }
        if (dst.n < min_n) {
            dst = src;
            return;
        }
    }
    const std::uint64_t total = dst_weight + src_weight;
    if (total == 0) {
        return;
    }
    const int merged_p = static_cast<int>(
        (static_cast<std::uint64_t>(dst.p) * dst_weight +
         static_cast<std::uint64_t>(src.p) * src_weight) /
        total);
    const int merged_n = static_cast<int>(
        std::min<std::uint64_t>(65535u, (static_cast<std::uint64_t>(dst.n) * dst_weight +
                                         static_cast<std::uint64_t>(src.n) * src_weight) /
                                            total));
    dst.p = static_cast<std::uint16_t>(merged_p);
    dst.n = static_cast<std::uint16_t>(merged_n);
}

inline void hp_undo_note(Counter& cell) {
    if (UndoRecorderScope::active()) {
        UndoRecorderScope::active()->note(cell);
    }
}

inline void counter_init(Counter* c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) { c[i].p = 32768; c[i].n = 0; }
}

inline int counter_predict_p(const Counter& c) { return c.p >> 4; }

inline int counter_predict(const Counter& c) {
    return stretch(c.p >> 4);
}

inline void counter_update(Counter& c, int y, int limit) {
    hp_undo_note(c);
    const int target = y ? 65535 : 0;
    const int d = target - static_cast<int>(c.p);
    c.p = static_cast<std::uint16_t>(static_cast<int>(c.p) + d / (c.n + 2));
    if (c.n < limit) ++c.n;
}

// ---------------------------------------------------------------------------
// A single hashed context model
// ---------------------------------------------------------------------------
/// Checkpoint format version being read (Predictor::read_checkpoint sets it);
/// ContextModel converts pre-v3 tables.
inline thread_local int g_hp_ckpt_read_version = 3;

class ContextModel {
 public:
    // Two outputs per bit now:
    //   [0] indirect  -- StateMap(bit history)   : pooled across contexts
    //   [1] direct PY -- Pitman-Yor discounted   : this context's own counts
    // Both come from the SAME stored state, so cost is 2 bytes per slot instead
    // of the 4 the old {p,n} counter used -- twice the table for the same RAM,
    // which is itself worth a couple of percent.
    static constexpr int kOutputs = 2;

    // table_bits == 0 builds a dropped model: one-slot tables, predicts 0.5
    // (stretch 0) and never learns. Used by Config::cm_drop.
    ContextModel(int table_bits, int limit)
        : off_(table_bits <= 0),
          mask_((1u << (table_bits > 0 ? table_bits : 0)) - 1),
          bits_(table_bits),
          limit_(limit),
          t_(table_bits),
          sm_() {}

    // A slot is 16 bits: bit-history state (882 states) in the low 10 bits and
    // a 6-bit checksum of the context above them (0 = empty slot). Formerly a
    // 16-bit state plus an 8-bit checksum: 3 bytes a slot, now 2.
    static constexpr unsigned kStateBits = 10;
    static constexpr std::uint16_t kStateMask = (1u << kStateBits) - 1;
    static int slot_state(std::uint16_t v) { return v & kStateMask; }
    static unsigned slot_chk(std::uint16_t v) { return v >> kStateBits; }
    static std::uint16_t slot_pack(int state, unsigned chk) {
        return static_cast<std::uint16_t>((chk << kStateBits) | static_cast<unsigned>(state));
    }
    // Checksum from the context bits above the index: 1..63 (0 marks empty).
    static unsigned chk_of(std::uint32_t above_index) {
        const unsigned v = above_index & 63u;
        return v == 0 ? 63u : v;
    }

    void set_context(std::uint32_t h) {
        hp_undo_note(h_);
        hp_undo_note(idle_);
        h_ = h;
        idle_ = false;
    }
    // Frozen models neither learn nor claim hash slots (Predictor::set_learning).
    void set_frozen(bool f) { frozen_ = f; }

    int table_bits() const { return off_ ? 0 : bits_; }

    /// Turn the model off and free its table (serve-time Config::cm_drop):
    /// it then predicts 0.5 like a model built dropped.
    void drop() {
        if (off_) return;
        off_ = true;
        bits_ = 0;
        mask_ = 0;
        t_.resize_bits(0);
        idx_ = 0;
        state_ = 0;
    }

    /// Shrink a trained table to ``bits`` by folding halves together, as if it
    /// had been trained that size: slot i and i + half share an index at one
    /// bit fewer, and the dropped index bit moves into the checksum
    /// (want = ((mixed >> bits) & 255) + 1, so it is exactly recomputable).
    /// On a collision the slot with more observations stays.
    void fold_to(int bits) {
        if (off_ || bits <= 0 || bits >= bits_) return;
        const StateTable& st = state_table();
        while (bits_ > bits) {
            const std::size_t half = static_cast<std::size_t>(1) << (bits_ - 1);
            std::vector<std::uint16_t> nt(half, 0);
            const std::uint16_t* t = t_.data();
            for (std::size_t i = 0; i < half; ++i) {
                int best_pri = -1;
                for (unsigned top = 0; top < 2; ++top) {
                    const std::uint16_t v = t[i + (top ? half : 0)];
                    const unsigned c6 = slot_chk(v);
                    if (c6 == 0) continue;
                    const int pri = st.n0(slot_state(v)) + st.n1(slot_state(v));
                    if (pri <= best_pri) continue;
                    best_pri = pri;
                    // The dropped index bit becomes the checksum's low bit
                    // (63 stands for 0 too, so that case is approximate).
                    nt[i] = slot_pack(slot_state(v), chk_of((c6 << 1) | top));
                }
            }
            --bits_;
            mask_ = (1u << bits_) - 1;
            t_.resize_bits(bits_);
            std::memcpy(t_.data(), nt.data(), half * sizeof(std::uint16_t));
        }
        idx_ &= mask_;
    }

    std::uint64_t learned_digest(std::uint64_t h) const {
        h = fnv_bytes(h, t_.data(), t_.size() * sizeof(std::uint16_t));
        return sm_.learned_digest(h);
    }
    // fx2 sets(): keep a mixer slot but do not pollute the table.
    void set_idle() {
        hp_undo_note(idle_);
        hp_undo_note(h_);
        idle_ = true;
        h_ = 0;
    }

    // Writes kOutputs stretched values into out[]. backoff is the parent
    // order's probability, used by the PY estimate.
    void predict(int c0, int backoff_p12, int* out) {
        if (off_) {
            for (int j = 0; j < kOutputs; ++j) out[j] = 0;
            return;
        }
        const std::uint32_t mixed =
            h_ ^ (static_cast<std::uint32_t>(c0) * 0x9E3779B1u);
        const StateTable& st = state_table();
        const std::uint32_t idx0 = mixed & mask_;
        const unsigned want = chk_of(mixed >> bits_);
        int best = 0;
        int best_pri = 1 << 30;
        int found = -1;
        const int nprobe = 3;
        for (int p = 0; p < nprobe; ++p) {
            const std::uint32_t i = idx0 ^ static_cast<std::uint32_t>(p);
            const std::uint16_t v = t_.get(i);
            if (slot_chk(v) == want) {
                found = static_cast<int>(i);
                break;
            }
            const int stt = slot_state(v);
            const int pri = (slot_chk(v) == 0) ? -1 : (st.n0(stt) + st.n1(stt));
            if (pri < best_pri) {
                best_pri = pri;
                best = static_cast<int>(i);
            }
        }
        if (found >= 0) {
            hp_undo_note(idx_);
            idx_ = static_cast<std::uint32_t>(found);
        } else if (!frozen_) {
            hp_undo_note(idx_);
            idx_ = static_cast<std::uint32_t>(best);
            hp_undo_note(t_.ref(idx_));
            t_.ref(idx_) = slot_pack(0, want);
        }
        hp_undo_note(state_);
        // Frozen: an unseen context reads as the empty state and claims no slot.
        state_ = (found >= 0 || !frozen_) ? slot_state(t_.get(idx_)) : 0;
        p_ind_ = sm_.predict(state_);
        {
            int a0 = st.n0(state_), a1 = st.n1(state_);
            if (a0 > 12) a0 = 12;
            if (a1 > 12) a1 = 12;
            p_py_ = py_estimate(a0, a1, backoff_p12);
        }
        out[0] = stretch(p_ind_);
        out[1] = stretch(p_py_);
    }

    int last_p() const { return p_ind_; }
    int last_py() const { return p_py_; }
    int n0() const { return state_table().n0(state_); }
    int n1() const { return state_table().n1(state_); }

    // 0..255 sparsity signal: high when this context has little history.
    int sparsity() const {
        const StateTable& st = state_table();
        const int n = st.n0(state_) + st.n1(state_);
        return n >= 8 ? 0 : 255 - n * 32;
    }

    /// Merge learned tables from ``src`` (StateMap + hash slot states).
    void merge_tables_from(const ContextModel& src, std::uint64_t src_weight,
                           std::uint64_t dst_weight, std::uint16_t min_statemap_count = 0) {
        sm_.merge_from(src.sm_, src_weight, dst_weight, min_statemap_count);
        const std::size_t n = t_.size();
        const StateTable& st = state_table();
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint16_t dv = t_.data()[i];
            const std::uint16_t sv = src.t_.data()[i];
            const int ds = slot_state(dv), ss = slot_state(sv);
            if (ss == 0) {
                continue;
            }
            if (ds == 0) {
                t_.data()[i] = sv;
                continue;
            }
            const int d_ev = st.n0(ds) + st.n1(ds);
            const int s_ev = st.n0(ss) + st.n1(ss);
            if (s_ev > d_ev) {
                t_.data()[i] = sv;
            }
        }
    }

    void copy_tables_from(const ContextModel& src) {
        sm_.copy_tables_from(src.sm_);
        const std::size_t n = t_.size();
        std::memcpy(t_.data(), src.t_.data(), n * sizeof(std::uint16_t));
    }

    /// Lossy serve: reset hash slots whose bit-history state has fewer than
    /// ``min_total`` observations (n0+n1). Cold entries fall back to uniform.
    void prune_cold_hash_slots(int min_total) {
        if (min_total <= 0) return;
        const StateTable& st = state_table();
        for (std::size_t i = 0; i < t_.size(); ++i) {
            if (slot_chk(t_.data()[i]) == 0) continue;
            const int state = slot_state(t_.data()[i]);
            if (st.n0(state) + st.n1(state) < min_total) {
                t_.data()[i] = 0;
            }
        }
    }

    void update(int y, int ens_p12 = -1) {
        if (idle_ || off_ || frozen_) return;
        std::int32_t ncl = 0;
        if (ens_p12 >= 0) {
            const std::int32_t diff =
                (static_cast<std::int32_t>(p_ind_) -
                 static_cast<std::int32_t>(ens_p12)) << 10;
            ncl = (diff * 4) >> 8;
        }
        sm_.update(y, limit_, ncl);
        const StateTable& st = state_table();
        hp_undo_note(t_.ref(idx_));
        t_.ref(idx_) = slot_pack(st.next(state_, y), slot_chk(t_.get(idx_)));
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, mask_);
        blob::write_pod(os, bits_);
        blob::write_pod(os, limit_);
        t_.checkpoint_write(os);  // packed slots (checkpoint v3)
        sm_.checkpoint_write(os);
        blob::write_pod(os, h_);
        blob::write_pod(os, idle_);
        blob::write_pod(os, idx_);
        blob::write_pod(os, state_);
        blob::write_pod(os, p_ind_);
        blob::write_pod(os, p_py_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, mask_);
        blob::read_pod(is, bits_);
        blob::read_pod(is, limit_);
        t_.checkpoint_read(is);
        if (g_hp_ckpt_read_version < 3) {
            // v1/v2: 16-bit states + separate 8-bit checksums; pack them.
            ZeroBuf<std::uint8_t> chk;
            chk.read(is);
            std::uint16_t* t = t_.data();
            for (std::size_t i = 0; i < t_.size(); ++i) {
                const std::uint8_t c8 = chk[i];
                t[i] = c8 == 0 ? static_cast<std::uint16_t>(0)
                               : slot_pack(t[i] & kStateMask, chk_of(static_cast<std::uint8_t>(c8 - 1u)));
            }
        }
        sm_.checkpoint_read(is);
        blob::read_pod(is, h_);
        blob::read_pod(is, idle_);
        blob::read_pod(is, idx_);
        blob::read_pod(is, state_);
        blob::read_pod(is, p_ind_);
        blob::read_pod(is, p_py_);
    }

 private:
    bool off_;
    bool frozen_ = false;
    std::uint32_t mask_;
    int bits_;
    int limit_;
    HashTable<std::uint16_t> t_;  // packed slots: state (10 bits) | checksum (6 bits)
    StateMap sm_;
    std::uint32_t h_ = 0;
    bool idle_ = false;
    std::uint32_t idx_ = 0;
    int state_ = 0;
    int p_ind_ = 2048;
    int p_py_ = 2048;
};

// ---------------------------------------------------------------------------
// Shared byte ring buffer for match models
// ---------------------------------------------------------------------------
//
// Every byte-keyed MatchModel in a Predictor reads the same byte history.
// Storing that history once (instead of once per model) cuts RAM by
// (N-1) * 2^buf_bits with no effect on compression — the models only
// differ in their hash tables and match state.

class ByteRing {
 public:
    explicit ByteRing(int buf_bits)
        : mask_((1u << buf_bits) - 1),
          buf_(static_cast<std::size_t>(1) << buf_bits, 0) {}

    void push(std::uint8_t byte) {
        hp_undo_note(buf_[pos_ & mask_]);
        hp_undo_note(pos_);
        buf_[pos_ & mask_] = byte;
        ++pos_;
    }

    std::uint32_t pos() const { return pos_; }
    std::uint32_t mask() const { return mask_; }

    std::uint8_t at(std::uint32_t p) const { return buf_[p & mask_]; }

    void reset() {
        std::fill(buf_.begin(), buf_.end(), 0);
        pos_ = 0;
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, mask_);
        blob::write_vec(os, buf_);
        blob::write_pod(os, pos_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, mask_);
        blob::read_vec(is, buf_);
        blob::read_pod(is, pos_);
    }

 private:
    std::uint32_t mask_;
    std::vector<std::uint8_t> buf_;
    std::uint32_t pos_ = 0;
};

// ---------------------------------------------------------------------------
// Match model
// ---------------------------------------------------------------------------
//
// Finds the most recent occurrence of the current MINLEN-byte suffix and
// predicts that the next byte repeats. On text this is the single most
// valuable model after low-order contexts: it captures repeated phrases,
// boilerplate, and -- on enwik -- repeated markup structures, at unbounded
// distance.
//
// Confidence is not hard-coded. A small counter table indexed by
// (quantised match length, predicted bit) learns how much the match model
// deserves to be trusted at each length. That keeps the model honest when it
// is wrong and lets it be very sharp when it is reliably right.

class MatchModel {
 public:
    static constexpr int kMinLen = 6;

    // order = how many past bytes are hashed to find a match. A BANK of
    // these at different orders is the cheapest real diversity available:
    // short orders fire often and loosely, long orders fire rarely and
    // authoritatively, and they decorrelate from each other far more than
    // adjacent context models do.
    MatchModel(ByteRing* ring, int table_bits, int order = 6, int skip = 1)
        : ring_(ring), order_(order), skip_(skip < 1 ? 1 : skip),
          off_(table_bits <= 0),
          tab_mask_((1u << (table_bits > 0 ? table_bits : 0)) - 1),
          tab_(table_bits > 0 ? table_bits : 0) {
        counter_init(st_.data(), st_.size());
    }

    void set_ring(ByteRing* ring) { ring_ = ring; }

    // Called once per byte after the shared ring has been updated.
    // `hist` holds the current kMinLen-byte suffix in its low bytes.
    void push_byte(int byte, std::uint64_t hist) {
        if (off_) return;  // dropped (Config::match_drop): predicts nothing
        const std::uint32_t pos = ring_->pos();
        // 1. Verify the standing prediction before anything else.
        if (len_ > 0) {
            if (ptr_ < pos && ring_->at(ptr_) == static_cast<std::uint8_t>(byte)) {
                if (len_ < 65535) {
                    hp_undo_note(len_);
                    ++len_;
                }
                hp_undo_note(ptr_);
                ++ptr_;
            } else {
                hp_undo_note(len_);
                len_ = 0;
            }
        }

        // 2. Index this suffix; adopt a new match if we have none.
        std::uint64_t key = 0;
        if (skip_ <= 1) {
            const std::uint64_t maskbits =
                (order_ >= 8) ? ~0ull : ((1ull << (order_ * 8)) - 1ull);
            key = hist & maskbits;
        } else {
            std::uint64_t h = hist;
            for (int i = 0; i < order_; ++i) {
                h >>= (8u * static_cast<unsigned>(skip_ - 1));
                key = (key << 8) | (h & 0xffull);
                h >>= 8;
            }
        }
        const std::uint32_t h =
            hash2(0x4D415443ull + static_cast<std::uint64_t>(order_) +
                      static_cast<std::uint64_t>(skip_) * 17ull,
                  key) & tab_mask_;
        if (len_ == 0) {
            const std::uint32_t cand = tab_.get(h);
            if (cand > 0 && cand < pos) {
                hp_undo_note(ptr_);
                hp_undo_note(len_);
                ptr_ = cand;
                len_ = 1;
            }
        }
        hp_undo_note(tab_.ref(h));
        tab_.ref(h) = pos;

        // 3. Drop the match if it has fallen out of the ring buffer.
        if (len_ > 0 && (pos - ptr_) > ring_->mask()) {
            hp_undo_note(len_);
            len_ = 0;
        }
    }

    // Once per bit. bitpos is 0..7, c0 is the partial byte with sentinel.
    int predict(int c0, int bitpos) {
        if (off_) {
            valid_ = false;
            return 0;
        }
        valid_ = false;
        const std::uint32_t pos = ring_->pos();
        if (len_ == 0 || ptr_ >= pos) return 0;

        const int pred_byte = ring_->at(ptr_);
        // The match only stands if the bits decoded so far in the current
        // byte agree with the predicted byte.
        if (bitpos > 0) {
            if (((pred_byte | 0x100) >> (8 - bitpos)) != c0) {
                hp_undo_note(len_);
                len_ = 0;
                return 0;
            }
        }
        expected_ = (pred_byte >> (7 - bitpos)) & 1;
        const int lq = len_ > 31 ? 31 : len_;
        hp_undo_note(sidx_);
        hp_undo_note(valid_);
        sidx_ = lq * 2 + expected_;
        valid_ = true;
        return counter_predict(st_[sidx_]);
    }

    void update(int y) {
        if (valid_) counter_update(st_[sidx_], y, 255);
    }

    int match_len() const { return len_; }
    std::uint64_t learned_digest(std::uint64_t h) const {
        return fnv_bytes(h, st_.data(), sizeof(st_));
    }

    void merge_counters_from(const MatchModel& src, std::uint64_t src_weight,
                             std::uint64_t dst_weight, std::uint16_t min_counter_n = 0) {
        for (std::size_t i = 0; i < st_.size(); ++i) {
            merge_counter_cell(st_[i], src.st_[i], src_weight, dst_weight, min_counter_n);
        }
    }

    void copy_counters_from(const MatchModel& src) { st_ = src.st_; }

    /// Turn the model off and free its table (serve-time Config::match_drop).
    void drop() {
        off_ = true;
        tab_.resize_bits(0);
        tab_mask_ = 0;
        len_ = 0;
    }

    /// Shrink the position table to ``bits``: of two folded entries keep the
    /// more recent position (what a smaller table would hold).
    void fold_to(int bits) {
        int cur = 0;
        while ((1u << cur) - 1 < tab_mask_) ++cur;
        if (bits <= 0 || bits >= cur) return;
        const std::size_t n = static_cast<std::size_t>(1) << bits;
        std::vector<std::uint32_t> nt(n, 0);
        for (std::size_t k = 0; k < tab_.size(); ++k) {
            const std::uint32_t v = tab_.at(k);
            std::uint32_t& d = nt[k & (n - 1)];
            if (v > d) d = v;
        }
        tab_.resize_bits(bits);
        std::memcpy(tab_.data(), nt.data(), n * sizeof(std::uint32_t));
        tab_mask_ = static_cast<std::uint32_t>(n - 1);
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, order_);
        blob::write_pod(os, skip_);
        blob::write_pod(os, tab_mask_);
        tab_.checkpoint_write(os);
        blob::write_array(os, st_);
        blob::write_pod(os, ptr_);
        blob::write_pod(os, len_);
        blob::write_pod(os, expected_);
        blob::write_pod(os, sidx_);
        blob::write_pod(os, valid_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, order_);
        blob::read_pod(is, skip_);
        blob::read_pod(is, tab_mask_);
        tab_.checkpoint_read(is);
        blob::read_array(is, st_);
        blob::read_pod(is, ptr_);
        blob::read_pod(is, len_);
        blob::read_pod(is, expected_);
        blob::read_pod(is, sidx_);
        blob::read_pod(is, valid_);
    }

 private:
    ByteRing* ring_;
    int order_;
    int skip_;
    bool off_ = false;
    std::uint32_t tab_mask_;
    HashTable<std::uint32_t> tab_;
    std::array<Counter, 64> st_{};
    std::uint32_t ptr_ = 0;
    int len_ = 0;
    int expected_ = 0;
    int sidx_ = 0;
    bool valid_ = false;
};


// ---------------------------------------------------------------------------
// Hebbian associative word model
// ---------------------------------------------------------------------------
//
// "Cells that fire together wire together." When word A is followed by word
// B, the association A->B is POTENTIATED. All associations DECAY slowly
// (synaptic scaling), so stale links fade without ever being explicitly
// deleted. The winner for a given A is whatever currently has the strongest
// synapse.
//
// WHY THIS IS THE REVERSE OF THE DICTIONARY THAT FAILED
// ----------------------------------------------------
// The static dictionary built a word table from the input and SHIPPED it:
// the transform won 1,498 B and the 10,121 B of stored table destroyed the
// gain. A Hebbian dictionary costs ZERO bytes to transmit, because encoder
// and decoder potentiate the identical synapses from data both already have.
// It is a dictionary that is learned rather than stored -- which removes the
// exact term that killed the previous attempt.
//
// It also fires on a different axis from every context model: association
// strength is not a suffix statistic, so it should raise effective rank.

class HebbianModel {
 public:
    HebbianModel(int table_bits, int limit)
        : mask_((1u << table_bits) - 1),
          syn_target_(static_cast<std::size_t>(1) << table_bits, 0),
          syn_strength_(static_cast<std::size_t>(1) << table_bits, 0),
          sm_(),
          t_(table_bits),
          limit_(limit) {}

    // Called at each word boundary with the completed word and its
    // predecessor. Potentiates prev -> cur.
    void potentiate(std::uint64_t prev_word, std::uint64_t cur_word) {
        if (prev_word == 0) return;
        const std::uint32_t slot = static_cast<std::uint32_t>(
            mix64(prev_word)) & mask_;
        std::uint64_t& target = syn_target_[slot];
        std::uint8_t& strength = syn_strength_[slot];
        if (target == cur_word) {
            if (strength < 255) {
                hp_undo_note(strength);
                ++strength;          // potentiation
            }
        } else if (strength > 0) {
            hp_undo_note(strength);
            --strength;                               // competition
            if (strength == 0) {
                hp_undo_note(target);
                target = cur_word;     // takeover
            }
        } else {
            hp_undo_note(target);
            hp_undo_note(strength);
            target = cur_word;
            strength = 1;
        }
        // Synaptic scaling: global slow decay keeps strengths bounded and
        // lets the network forget associations that stop being reinforced.
        hp_undo_note(tick_);
        if ((++tick_ & 0x3FF) == 0 && strength > 0) {
            hp_undo_note(strength);
            --strength;
        }
    }

    // Context for the current bit: the strongest association from the
    // previous word, bucketed by synaptic strength.
    void set_context(std::uint64_t prev_word) {
        const std::uint32_t slot = static_cast<std::uint32_t>(
            mix64(prev_word)) & mask_;
        const std::uint64_t target = syn_target_[slot];
        const std::uint8_t strength = syn_strength_[slot];
        strength_ = strength;
        h_ = hash2(0x48454242ull,
                   target * 131ull + static_cast<std::uint64_t>(
                       strength >= 32 ? 3 : (strength >= 8 ? 2 :
                       (strength >= 2 ? 1 : 0))));
    }

    int predict(int c0) {
        idx_ = (h_ ^ (static_cast<std::uint32_t>(c0) * 0x9E3779B1u)) & mask_;
        state_ = t_.get(idx_);
        return stretch(sm_.predict(state_));
    }

    void update(int y) {
        sm_.update(y, limit_);
        const StateTable& st = state_table();
        hp_undo_note(t_.ref(idx_));
        t_.ref(idx_) = static_cast<std::uint16_t>(st.next(state_, y));
    }

    int strength() const { return strength_; }

    int table_bits() const {
        int b = 0;
        while ((1u << b) - 1 < mask_) ++b;
        return b;
    }

    /// Shrink to ``bits`` as if built that size: synapses keep the stronger of
    /// two folded slots, bit-history slots the busier state.
    void fold_to(int bits) {
        const int cur = table_bits();
        if (bits <= 0 || bits >= cur) return;
        const std::size_t n = static_cast<std::size_t>(1) << bits;
        std::vector<std::uint64_t> nt(n, 0);
        std::vector<std::uint8_t> ns(n, 0);
        std::vector<std::uint16_t> nst(n, 0);
        const StateTable& st = state_table();
        for (std::size_t k = 0; k < syn_target_.size(); ++k) {
            const std::size_t j = k & (n - 1);
            if (syn_strength_[k] > ns[j]) {
                ns[j] = syn_strength_[k];
                nt[j] = syn_target_[k];
            }
            const std::uint16_t v = t_.data()[k];
            if (v != 0 && st.n0(v) + st.n1(v) > st.n0(nst[j]) + st.n1(nst[j])) nst[j] = v;
        }
        syn_target_ = std::move(nt);
        syn_strength_ = std::move(ns);
        t_.resize_bits(bits);
        std::memcpy(t_.data(), nst.data(), n * sizeof(std::uint16_t));
        mask_ = static_cast<std::uint32_t>(n - 1);
        idx_ &= mask_;
    }
    std::uint64_t learned_digest(std::uint64_t h) const {
        h = fnv_bytes(h, syn_target_.data(), syn_target_.size() * sizeof(syn_target_[0]));
        h = fnv_bytes(h, syn_strength_.data(), syn_strength_.size());
        h = fnv_bytes(h, t_.data(), t_.size() * sizeof(std::uint16_t));
        return sm_.learned_digest(h);
    }

    void merge_tables_from(const HebbianModel& src, std::uint64_t src_weight,
                           std::uint64_t dst_weight, std::uint16_t min_statemap_count = 0) {
        sm_.merge_from(src.sm_, src_weight, dst_weight, min_statemap_count);
        const std::size_t n = t_.size();
        const StateTable& st = state_table();
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint16_t ds = t_.data()[i];
            const std::uint16_t ss = src.t_.data()[i];
            if (ss == 0) {
                continue;
            }
            if (ds == 0) {
                t_.data()[i] = ss;
                continue;
            }
            const int d_ev = st.n0(ds) + st.n1(ds);
            const int s_ev = st.n0(ss) + st.n1(ss);
            if (s_ev > d_ev) {
                t_.data()[i] = ss;
            }
        }
        for (std::size_t i = 0; i < syn_target_.size(); ++i) {
            if (src.syn_strength_[i] > syn_strength_[i]) {
                syn_target_[i] = src.syn_target_[i];
                syn_strength_[i] = src.syn_strength_[i];
            }
        }
    }

    void copy_tables_from(const HebbianModel& src) {
        sm_.copy_tables_from(src.sm_);
        const std::size_t n = t_.size();
        std::memcpy(t_.data(), src.t_.data(), n * sizeof(std::uint16_t));
        syn_target_ = src.syn_target_;
        syn_strength_ = src.syn_strength_;
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, mask_);
        blob::write_vec(os, syn_target_);
        blob::write_vec(os, syn_strength_);
        sm_.checkpoint_write(os);
        t_.checkpoint_write(os);
        blob::write_pod(os, limit_);
        blob::write_pod(os, h_);
        blob::write_pod(os, idx_);
        blob::write_pod(os, state_);
        blob::write_pod(os, strength_);
        blob::write_pod(os, tick_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, mask_);
        blob::read_vec(is, syn_target_);
        blob::read_vec(is, syn_strength_);
        sm_.checkpoint_read(is);
        t_.checkpoint_read(is);
        blob::read_pod(is, limit_);
        blob::read_pod(is, h_);
        blob::read_pod(is, idx_);
        blob::read_pod(is, state_);
        blob::read_pod(is, strength_);
        blob::read_pod(is, tick_);
    }

 private:
    std::uint32_t mask_;
    std::vector<std::uint64_t> syn_target_;
    std::vector<std::uint8_t> syn_strength_;
    StateMap sm_;
    HashTable<std::uint16_t> t_;
    int limit_;
    std::uint32_t h_ = 0, idx_ = 0;
    int state_ = 0, strength_ = 0;
    std::uint32_t tick_ = 0;
};

// Small DMC graph. Clone-on-threshold, integer counts, one stretched p.
class DmcModel {
 public:
    explicit DmcModel(int cap_bits = 18)
        : cap_(static_cast<std::uint32_t>(1u << (cap_bits > 20 ? 20 : cap_bits))) {
        // Full capacity up front: speculative (undo-recorded) splits patch cells
        // inside nodes_, so the buffer must never move. Untouched pages cost no RSS.
        nodes_.reserve(cap_);
        nodes_.resize(256);
        for (int i = 0; i < 256; ++i) {
            nodes_[static_cast<std::size_t>(i)].n0 = 1;
            nodes_[static_cast<std::size_t>(i)].n1 = 1;
            nodes_[static_cast<std::size_t>(i)].nx[0] =
                static_cast<std::uint32_t>((i << 1) & 255);
            nodes_[static_cast<std::size_t>(i)].nx[1] =
                static_cast<std::uint32_t>(((i << 1) & 255) | 1);
        }
    }

    int predict() {
        const Node& n = nodes_[cur_];
        const int den = static_cast<int>(n.n0) + static_cast<int>(n.n1);
        int p12 = (static_cast<int>(n.n1) << 12) / (den ? den : 1);
        if (p12 < 1) p12 = 1;
        if (p12 > 4094) p12 = 4094;
        return stretch(p12);
    }

    std::uint64_t learned_digest(std::uint64_t h) const {
        const std::uint64_t n = nodes_.size();
        h = fnv_bytes(h, &n, sizeof(n));
        for (const Node& nd : nodes_) {
            h = fnv_bytes(h, &nd.n0, sizeof(nd.n0));
            h = fnv_bytes(h, &nd.n1, sizeof(nd.n1));
            h = fnv_bytes(h, nd.nx, sizeof(nd.nx));
        }
        return h;
    }

    // Follow the y edge without counting or splitting (frozen serve).
    void advance(int y) {
        std::uint32_t nxt = nodes_[cur_].nx[y];
        if (nxt >= nodes_.size()) nxt = 0;
        hp_undo_note(cur_);
        cur_ = nxt;
    }

    void update(int y) {
        Node& n = nodes_[cur_];
        if (y) {
            hp_undo_note(n.n1);
            if (n.n1 < 65535) ++n.n1;
        } else {
            hp_undo_note(n.n0);
            if (n.n0 < 65535) ++n.n0;
        }
        std::uint32_t nxt = n.nx[y];
        if (nxt >= nodes_.size()) nxt = 0;
        const int tot = static_cast<int>(n.n0) + static_cast<int>(n.n1);
        const Node ch = nodes_[nxt];
        const int cht = static_cast<int>(ch.n0) + static_cast<int>(ch.n1);
        if (tot >= 8 && cht > 2 && tot >= cht * 4 &&
            nodes_.size() < cap_) {
            Node nn = ch;
            const int tn = cht ? cht : 1;
            nn.n0 = static_cast<std::uint16_t>(
                1 + (static_cast<int>(ch.n0) * tot) / (tn * 2));
            nn.n1 = static_cast<std::uint16_t>(
                1 + (static_cast<int>(ch.n1) * tot) / (tn * 2));
            if (nn.n0 > ch.n0) nn.n0 = ch.n0;
            if (nn.n1 > ch.n1) nn.n1 = ch.n1;
            const std::uint32_t parent = cur_;
            hp_undo_note(nodes_[nxt].n0);
            hp_undo_note(nodes_[nxt].n1);
            if (ch.n0 > nn.n0)
                nodes_[nxt].n0 = static_cast<std::uint16_t>(ch.n0 - nn.n0 + 1);
            if (ch.n1 > nn.n1)
                nodes_[nxt].n1 = static_cast<std::uint16_t>(ch.n1 - nn.n1 + 1);
            hp_undo_note_size(nodes_);
            nodes_.push_back(nn);
            hp_undo_note(nodes_[parent].nx[y]);
            nodes_[parent].nx[y] = static_cast<std::uint32_t>(nodes_.size() - 1);
            nxt = nodes_[parent].nx[y];
        }
        hp_undo_note(cur_);
        cur_ = nxt;
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, cap_);
        blob::write_pod(os, cur_);
        const std::uint64_t n = nodes_.size();
        blob::write_pod(os, n);
        for (const auto& node : nodes_) {
            blob::write_pod(os, node.n0);
            blob::write_pod(os, node.n1);
            blob::write_pod(os, node.nx[0]);
            blob::write_pod(os, node.nx[1]);
        }
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, cap_);
        blob::read_pod(is, cur_);
        std::uint64_t n = 0;
        blob::read_pod(is, n);
        nodes_.reserve(cap_ > n ? cap_ : n);
        nodes_.resize(n);
        for (auto& node : nodes_) {
            blob::read_pod(is, node.n0);
            blob::read_pod(is, node.n1);
            blob::read_pod(is, node.nx[0]);
            blob::read_pod(is, node.nx[1]);
        }
    }

 private:
    struct Node {
        std::uint16_t n0 = 1, n1 = 1;
        std::uint32_t nx[2] = {0, 0};
    };
    std::vector<Node> nodes_;
    std::uint32_t cap_;
    std::uint32_t cur_ = 0;
};

// Last-occurrence of order-3 context predicts the following byte. No length.
class LzpModel {
 public:
    LzpModel(int table_bits)
        : mask_((1u << table_bits) - 1),
          pred_(static_cast<std::size_t>(1) << table_bits, 0) {
        counter_init(st_.data(), st_.size());
    }

    std::uint64_t learned_digest(std::uint64_t h) const {
        return fnv_bytes(h, st_.data(), sizeof(st_));
    }

    void push_byte(int byte, std::uint64_t hist) {
        const std::uint32_t h =
            hash2(0x4C5A5033ull, hist & 0xffffffull) & mask_;
        expected_ = pred_[h];
        hp_undo_note(pred_[h]);
        pred_[h] = static_cast<std::uint8_t>(byte);
        have_ = 1;
    }

    int predict(int c0, int bitpos) {
        valid_ = false;
        if (!have_) return 0;
        if (bitpos > 0) {
            if (((expected_ | 0x100) >> (8 - bitpos)) != c0) return 0;
        }
        expected_bit_ = (expected_ >> (7 - bitpos)) & 1;
        sidx_ = (bitpos * 2) + expected_bit_;
        valid_ = true;
        return counter_predict(st_[sidx_]);
    }

    void update(int y) {
        if (valid_) counter_update(st_[sidx_], y, 255);
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, mask_);
        blob::write_vec(os, pred_);
        blob::write_array(os, st_);
        blob::write_pod(os, expected_);
        blob::write_pod(os, expected_bit_);
        blob::write_pod(os, sidx_);
        blob::write_pod(os, have_);
        blob::write_pod(os, valid_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, mask_);
        blob::read_vec(is, pred_);
        blob::read_array(is, st_);
        blob::read_pod(is, expected_);
        blob::read_pod(is, expected_bit_);
        blob::read_pod(is, sidx_);
        blob::read_pod(is, have_);
        blob::read_pod(is, valid_);
    }

 private:
    std::uint32_t mask_;
    std::vector<std::uint8_t> pred_;
    std::array<Counter, 64> st_{};
    int expected_ = 0;
    int expected_bit_ = 0;
    int sidx_ = 0;
    int have_ = 0;
    bool valid_ = false;
};

}  // namespace hp
