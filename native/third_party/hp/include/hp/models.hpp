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
#include <vector>

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
class ContextModel {
 public:
    // Two outputs per bit now:
    //   [0] indirect  -- StateMap(bit history)   : pooled across contexts
    //   [1] direct PY -- Pitman-Yor discounted   : this context's own counts
    // Both come from the SAME stored state, so cost is 2 bytes per slot instead
    // of the 4 the old {p,n} counter used -- twice the table for the same RAM,
    // which is itself worth a couple of percent.
    static constexpr int kOutputs = HP_PY_EXPERT ? 2 : 1;

    ContextModel(int table_bits, int limit)
        : mask_((1u << table_bits) - 1),
#if HP_HASH_CHK
          bits_(table_bits),
#endif
          limit_(limit),
          t_(table_bits),
#if HP_HASH_CHK
          chk_(static_cast<std::size_t>(1) << table_bits, 0),
#endif
          sm_() {}

    void set_context(std::uint32_t h) { h_ = h; idle_ = false; }
    // fx2 sets(): keep a mixer slot but do not pollute the table.
    void set_idle() { idle_ = true; h_ = 0; }

    // Writes kOutputs stretched values into out[]. backoff is the parent
    // order's probability, used by the PY estimate.
    void predict(int c0, int backoff_p12, int* out) {
        const std::uint32_t mixed =
            h_ ^ (static_cast<std::uint32_t>(c0) * 0x9E3779B1u);
        const StateTable& st = state_table();
#if HP_HASH_CHK
        const std::uint32_t idx0 = mixed & mask_;
        const std::uint8_t want =
            static_cast<std::uint8_t>(((mixed >> bits_) & 255u) + 1u);
        int best = 0;
        int best_pri = 1 << 30;
        int found = -1;
        const int nprobe = HP_HASH_P5 ? 5 : 3;
        for (int p = 0; p < nprobe; ++p) {
            const std::uint32_t i = idx0 ^ static_cast<std::uint32_t>(p);
            if (chk_[i] == want) {
                found = static_cast<int>(i);
                break;
            }
            const int stt = t_.get(i);
            const int pri = (chk_[i] == 0)
                                ? -1
                                : (st.n0(stt) + st.n1(stt));
            if (pri < best_pri) {
                best_pri = pri;
                best = static_cast<int>(i);
            }
        }
        if (found >= 0) {
            hp_undo_note(idx_);
            idx_ = static_cast<std::uint32_t>(found);
        } else {
            hp_undo_note(idx_);
            idx_ = static_cast<std::uint32_t>(best);
            hp_undo_note(t_.ref(idx_));
            t_.ref(idx_) = 0;
            hp_undo_note(chk_[idx_]);
            chk_[idx_] = want;
        }
#else
        hp_undo_note(idx_);
        idx_ = mixed & mask_;
#endif
        hp_undo_note(state_);
        state_ = t_.get(idx_);
        p_ind_ = sm_.predict(state_);
#if HP_PY_EXPERT
#if HP_STATE_TABLE2
        {
            int a0 = st.n0(state_), a1 = st.n1(state_);
            if (a0 > 12) a0 = 12;
            if (a1 > 12) a1 = 12;
            p_py_ = py_estimate(a0, a1, backoff_p12);
        }
#else
        p_py_ = py_estimate(st.n0(state_), st.n1(state_), backoff_p12);
#endif
        out[0] = stretch(p_ind_);
        out[1] = stretch(p_py_);
#else
        (void)backoff_p12;
        (void)st;
        out[0] = stretch(p_ind_);
#endif
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
                           std::uint64_t dst_weight) {
        sm_.merge_from(src.sm_, src_weight, dst_weight);
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
#if HP_HASH_CHK
                chk_[i] = src.chk_[i];
#endif
                continue;
            }
            const int d_ev = st.n0(ds) + st.n1(ds);
            const int s_ev = st.n0(ss) + st.n1(ss);
            if (s_ev > d_ev) {
                t_.data()[i] = ss;
#if HP_HASH_CHK
                chk_[i] = src.chk_[i];
#endif
            }
        }
    }

    void copy_tables_from(const ContextModel& src) {
        sm_.copy_tables_from(src.sm_);
        const std::size_t n = t_.size();
        std::memcpy(t_.data(), src.t_.data(), n * sizeof(std::uint16_t));
#if HP_HASH_CHK
        chk_ = src.chk_;
#endif
    }

    /// Lossy serve: reset hash slots whose bit-history state has fewer than
    /// ``min_total`` observations (n0+n1). Cold entries fall back to uniform.
    void prune_cold_hash_slots(int min_total) {
        if (min_total <= 0) return;
        const StateTable& st = state_table();
        for (std::size_t i = 0; i < t_.size(); ++i) {
#if HP_HASH_CHK
            if (chk_[i] == 0) continue;
#endif
            const int state = static_cast<int>(t_.data()[i]);
            if (st.n0(state) + st.n1(state) < min_total) {
                t_.data()[i] = 0;
#if HP_HASH_CHK
                chk_[i] = 0;
#endif
            }
        }
    }

    void update(int y, int ens_p12 = -1) {
        if (idle_) return;
        std::int32_t ncl = 0;
#if HP_NCL
        if (ens_p12 >= 0) {
            const std::int32_t diff =
                (static_cast<std::int32_t>(p_ind_) -
                 static_cast<std::int32_t>(ens_p12)) << 10;
            ncl = (diff * HP_NCL_LAMBDA) >> 8;
        }
#else
        (void)ens_p12;
#endif
        sm_.update(y, limit_, ncl);
        const StateTable& st = state_table();
        hp_undo_note(t_.ref(idx_));
        t_.ref(idx_) = static_cast<std::uint16_t>(st.next(state_, y));
    }

 private:
    std::uint32_t mask_;
#if HP_HASH_CHK
    int bits_;
#endif
    int limit_;
    HashTable<std::uint16_t> t_;  // bit-history states (882 states -> 16 bit)
#if HP_HASH_CHK
    std::vector<std::uint8_t> chk_;
#endif
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
          tab_mask_((1u << table_bits) - 1),
          tab_(table_bits) {
        counter_init(st_.data(), st_.size());
    }

    void set_ring(ByteRing* ring) { ring_ = ring; }

    // Called once per byte after the shared ring has been updated.
    // `hist` holds the current kMinLen-byte suffix in its low bytes.
    void push_byte(int byte, std::uint64_t hist) {
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

    void merge_counters_from(const MatchModel& src, std::uint64_t src_weight,
                             std::uint64_t dst_weight) {
        for (std::size_t i = 0; i < st_.size(); ++i) {
            merge_counter(st_[i], src.st_[i], src_weight, dst_weight);
        }
    }

    void copy_counters_from(const MatchModel& src) { st_ = src.st_; }

 private:
    static void merge_counter(Counter& dst, const Counter& src, std::uint64_t src_weight,
                              std::uint64_t dst_weight) {
        if (src.n == 0) {
            return;
        }
        if (dst.n == 0) {
            dst = src;
            return;
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

 private:
    ByteRing* ring_;
    int order_;
    int skip_;
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

    void merge_tables_from(const HebbianModel& src, std::uint64_t src_weight,
                           std::uint64_t dst_weight) {
        sm_.merge_from(src.sm_, src_weight, dst_weight);
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

    void update(int y) {
        Node& n = nodes_[cur_];
        if (y) {
            if (n.n1 < 65535) ++n.n1;
        } else {
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
            if (ch.n0 > nn.n0)
                nodes_[nxt].n0 = static_cast<std::uint16_t>(ch.n0 - nn.n0 + 1);
            if (ch.n1 > nn.n1)
                nodes_[nxt].n1 = static_cast<std::uint16_t>(ch.n1 - nn.n1 + 1);
            nodes_.push_back(nn);
            nodes_[parent].nx[y] = static_cast<std::uint32_t>(nodes_.size() - 1);
            nxt = nodes_[parent].nx[y];
        }
        cur_ = nxt;
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

    void push_byte(int byte, std::uint64_t hist) {
        const std::uint32_t h =
            hash2(0x4C5A5033ull, hist & 0xffffffull) & mask_;
        expected_ = pred_[h];
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

// Move-to-front rank of the previous byte, mixed with c0.
class SrModel {
 public:
    SrModel() {
        for (int i = 0; i < 256; ++i) mtf_[i] = static_cast<std::uint8_t>(i);
        counter_init(st_, 256);
    }

    void push_byte(int byte) {
        int r = 0;
        while (r < 256 && mtf_[r] != static_cast<std::uint8_t>(byte)) ++r;
        last_rank_ = r > 31 ? 31 : r;
        if (r > 0 && r < 256) {
            const std::uint8_t v = mtf_[r];
            for (int i = r; i > 0; --i) mtf_[i] = mtf_[i - 1];
            mtf_[0] = v;
        }
    }

    int predict(int c0) {
        sidx_ = (last_rank_ << 3) | (c0 & 7);
        return counter_predict(st_[sidx_]);
    }

    void update(int y) { counter_update(st_[sidx_], y, 255); }

 private:
    std::uint8_t mtf_[256];
    Counter st_[256];
    int last_rank_ = 0;
    int sidx_ = 0;
};

}  // namespace hp
