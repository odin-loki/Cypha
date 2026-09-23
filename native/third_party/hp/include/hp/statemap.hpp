#pragma once
//
// hp/statemap.hpp — indirect context modelling.
//
// THE MULTIPLICATIVE IDEA
// -----------------------
// Direct modelling is  context -> probability.  Every context learns alone,
// so a context seen 3 times has a terrible estimate and a context seen 30000
// times has a good one, and neither helps the other.
//
// Indirect modelling is  context -> bit history -> probability.  Two hops.
// The first hop stores a compact STATE describing what has happened in this
// context (roughly: how many 0s, how many 1s, and what came last). The second
// hop is a StateMap: a small table, shared across ALL contexts, mapping each
// state to the probability that empirically follows it.
//
// The payoff is statistical pooling. A context seen 3 times lands in the same
// state as thousands of other contexts seen 3 times with the same pattern, and
// inherits their pooled statistics immediately. That is the "default to fall
// back on that gets better as it sees more data" -- it is not bolted on, it
// falls out of the representation.
//
// This is multiplicative, not additive: it upgrades the estimator inside
// EVERY context model at once, rather than adding an eleventh opinion.
//
// STATE ENCODING
// --------------
// A byte encoding (n0, n1, last_bit). Counts are capped and, on the classic
// PAQ observation, seeing a bit partially DISCOUNTS the opposite count --
// nonstationary sources should forget contradicted evidence rather than
// average it forever. Cap 20 per side keeps the table small and adaptive.

#include <algorithm>
#include <array>
#include <cstdint>
#include <istream>
#include <ostream>

#include "hp/blob_io.hpp"
#include "hp/features.hpp"
#include "hp/int_math.hpp"
#include "hp/undo.hpp"

namespace hp {

// ---------------------------------------------------------------------------
// Bit-history state table, built once by exact integer recursion.
// ---------------------------------------------------------------------------
class StateTable {
 public:
    static constexpr int kCap = 20;
    static constexpr int kStates = (kCap + 1) * (kCap + 1) * 2;

    StateTable() {
        for (int n0 = 0; n0 <= kCap; ++n0)
            for (int n1 = 0; n1 <= kCap; ++n1)
                for (int last = 0; last < 2; ++last) {
                    const int s = idx(n0, n1, last);
                    next_[s][0] = static_cast<std::uint16_t>(step(n0, n1, 0));
                    next_[s][1] = static_cast<std::uint16_t>(step(n0, n1, 1));
                    n0_[s] = static_cast<std::uint8_t>(n0);
                    n1_[s] = static_cast<std::uint8_t>(n1);
                }
    }

    int next(int state, int y) const { return next_[state][y]; }
    int n0(int state) const { return n0_[state]; }
    int n1(int state) const { return n1_[state]; }

    static int idx(int n0, int n1, int last) {
        return (n0 * (kCap + 1) + n1) * 2 + last;
    }

 private:
    // Observing y increments its own count and discounts the opposite one.
    // Discount schedule: counts above 2 are halved-toward-2. Contradicted
    // evidence decays fast; the first couple of observations are protected.
    static int step(int n0, int n1, int y) {
        if (y) {
            if (n1 < kCap) ++n1;
            if (n0 > 2) n0 = 2 + (n0 - 2) / 2;
        } else {
            if (n0 < kCap) ++n0;
            if (n1 > 2) n1 = 2 + (n1 - 2) / 2;
        }
        return idx(n0, n1, y);
    }

    std::uint16_t next_[kStates][2];
    std::uint8_t n0_[kStates], n1_[kStates];
};

inline const StateTable& state_table() {
    static const StateTable t;
    return t;
}

// ---------------------------------------------------------------------------
// StateMap: state -> probability, learned online, shared across contexts.
// ---------------------------------------------------------------------------
//
// Each entry packs a 22-bit probability and a 10-bit count. The count drives
// an adaptive rate: fast while the estimate is fresh, slow once settled. This
// is the pooling layer -- every context in the same state trains the same
// entry, so sparse contexts ride on dense ones.

class StateMap {
 public:
    StateMap() {
        for (auto& v : t_) v = (1u << 31) | 0u;  // p = 0.5, count = 0
    }

    int predict(int cx) {
        idx_ = static_cast<std::size_t>(cx);
        return static_cast<int>(t_[idx_] >> 20);  // 12-bit
    }

    // ncl_extra is the Liu & Yao 1999 term lambda*(p_i - p_ens) in 22-bit.
    // Default 0 preserves the original update. Both sides pass the same
    // ensemble p (already computed), so determinism holds.
    void update(int y, int limit = 1023, std::int32_t ncl_extra = 0) {
        hp_undo_note(t_[idx_]);
        const std::uint32_t p = t_[idx_] >> 10;      // 22-bit probability
        std::uint32_t n = t_[idx_] & 1023u;          // count
        if (n < static_cast<std::uint32_t>(limit)) ++n;
        // p += (target - p + ncl_extra) / (n + 2)
        const std::int32_t target = y ? ((1 << 22) - 1) : 0;
        const std::int32_t d = target - static_cast<std::int32_t>(p) + ncl_extra;
        const std::int32_t np = static_cast<std::int32_t>(p) +
                                d / static_cast<std::int32_t>(n + 2);
        t_[idx_] = (static_cast<std::uint32_t>(np) << 10) | n;
    }

    /// Weighted merge of packed (prob<<10)|count entries (train-scale shard merge).
    void merge_from(const StateMap& src, std::uint64_t src_weight, std::uint64_t dst_weight,
                    std::uint16_t min_count = 0) {
        for (std::size_t i = 0; i < t_.size(); ++i) {
            const std::uint32_t sv = src.t_[i];
            const std::uint32_t sn = sv & 1023u;
            if (sn == 0) {
                continue;
            }
            const std::uint32_t dv = t_[i];
            const std::uint32_t dn = dv & 1023u;
            if (dn == 0) {
                t_[i] = sv;
                continue;
            }
            if (min_count > 0) {
                if (sn < min_count) {
                    continue;
                }
                if (dn < min_count) {
                    t_[i] = sv;
                    continue;
                }
            }
            const std::uint64_t total = dst_weight + src_weight;
            if (total == 0) {
                continue;
            }
            const std::uint32_t sp = sv >> 10;
            const std::uint32_t dp = dv >> 10;
            const std::uint32_t mp =
                static_cast<std::uint32_t>((static_cast<std::uint64_t>(dp) * dst_weight +
                                            static_cast<std::uint64_t>(sp) * src_weight) /
                                           total);
            const std::uint32_t mn = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(1023u, (static_cast<std::uint64_t>(dn) * dst_weight +
                                                  static_cast<std::uint64_t>(sn) * src_weight) /
                                                     total));
            t_[i] = (mp << 10) | mn;
        }
    }

    void copy_tables_from(const StateMap& src) { t_ = src.t_; }

    void checkpoint_write(std::ostream& os) const {
        blob::write_array(os, t_);
        blob::write_pod(os, idx_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_array(is, t_);
        blob::read_pod(is, idx_);
    }

 private:
    std::array<std::uint32_t, StateTable::kStates> t_{};
    std::size_t idx_ = 0;
};

// ---------------------------------------------------------------------------
// Pitman-Yor style discounted estimate, integer.
// ---------------------------------------------------------------------------
//
// The PY / Kneser-Ney insight: natural language counts follow a power law, so
// subtracting a fixed discount d from every observed count and redistributing
// that mass to the backoff distribution beats add-one smoothing badly.
//
//   P(1) = (n1 - d)+ / (n0 + n1)  +  (d * types / (n0 + n1)) * P_backoff(1)
//
// Here d = 0.5 (Q8: 128), types = number of distinct symbols seen (0, 1 or 2
// in the binary case). Everything in Q12 so it lands straight in the mixer.

inline int py_estimate(int n0, int n1, int backoff_p12) {
    const int total = n0 + n1;
    if (total == 0) return backoff_p12;
    const int d = 128;                                   // 0.5 in Q8
    const int types = (n0 > 0 ? 1 : 0) + (n1 > 0 ? 1 : 0);
    // numerator in Q8, then to Q12
    int num = (n1 << 8) - (n1 > 0 ? d : 0);
    if (num < 0) num = 0;
    const int mass = (d * types);                        // Q8
    const int denom = total << 8;                        // Q8
    // p = num/denom + (mass/denom) * backoff
    const std::int64_t direct = (static_cast<std::int64_t>(num) << 12) / denom;
    const std::int64_t back =
        (static_cast<std::int64_t>(mass) * backoff_p12) / denom;
    int p = static_cast<int>(direct + back);
    return clamp_int(p, 1, 4094);
}

}  // namespace hp
