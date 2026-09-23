#pragma once
//
// hp/hedge.hpp — exponential-weights Bayesian model averaging with
// fixed-share tracking, integer only.
//
// WHY THIS EXISTS
// ---------------
// The mixer in mixer.hpp combines experts MULTIPLICATIVELY: logit-space
// mixing is a product of experts, p ~ prod p_i^w_i. Products are sharp -- a
// confident expert can veto -- but they are also fragile: correlated experts
// compound each other's confidence, and a product has no regret guarantee at
// all. The redundancy profiler measured exactly that failure.
//
// This file adds the complementary operation: a SUM. Bayesian model averaging
// over the same experts,
//
//     P = sum_d  w_d * p_d ,      w_d  ~  2^(-eta * L_d)
//
// where L_d is expert d's cumulative code length. This is the Hedge /
// exponential-weights algorithm, and unlike the product it carries a proven
// regret bound: cumulative loss exceeds the best single expert's by at most
// O(sqrt(T log N)).
//
// So the two mix in opposite directions, and the layer-2 mixer learns when to
// trust which. That is the actual "algorithms that multiply together":
// a product-of-experts and a sum-of-experts feeding the same combiner, each
// covering the other's failure mode.
//
// FIXED SHARE -- AND WHAT ALPHA IS FOR
// ------------------------------------
// Plain Hedge converges to the best FIXED expert and then stops adapting. On
// enwik9 the best expert is not fixed: markup, prose, tables and link farms
// want different ones. Fixed-share (Herbster & Warmuth) fixes this by leaking
// a fraction sigma of the weight mass back to uniform on every step:
//
//     w_d  <-  (1 - sigma) w_d  +  sigma / N
//
// which converts the bound into a SWITCHING regret bound -- competitive with
// the best *sequence* of experts rather than the best single one.
//
// sigma is the switching rate, and it should be high exactly when the source
// regime is changing. That is what GRIA's alpha measures. So alpha enters
// here as a STATISTIC controlling the switching rate, not as a gate context.
// It is a scalar driving an adaptive rate, which is what an order parameter
// is actually for.

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include "hp/blob_io.hpp"
#include "hp/features.hpp"
#include "hp/int_math.hpp"
#include "hp/undo.hpp"


namespace hp {

class Hedge {
 public:
    explicit Hedge(int n)
        : n_(n),
          w_(static_cast<std::size_t>(n),
             static_cast<std::uint16_t>((1u << 16) / (n ? n : 1))),
          p_(static_cast<std::size_t>(n), 2048) {}

    // Record expert d's 12-bit probability for this bit.
    void set(int d, int p12) { p_[d] = clamp_int(p12, 1, 4094); }

    // Weighted average, 12-bit out.
    int mix() {
        std::uint64_t acc = 0, wsum = 0;
        for (int d = 0; d < n_; ++d) {
            acc += static_cast<std::uint64_t>(w_[d]) * static_cast<std::uint32_t>(p_[d]);
            wsum += w_[d];
        }
        if (wsum == 0) return 2048;
        pr_ = clamp_int(static_cast<int>(acc / wsum), 1, 4094);
        return pr_;
    }

    // sigma_q16: switching rate in Q16, driven by the GRIA alpha trajectory.
    void update(int y, std::uint32_t sigma_q16) {
        // 1. Multiplicative loss update: w_d *= 2^(-eta * L_d).
        //    L_d = -log2(p_d(actual)) in Q16; eta folded into the shift.
        std::uint64_t wsum = 0;
        for (int d = 0; d < n_; ++d) {
            const int pa = y ? p_[d] : 4096 - p_[d];
            const std::uint32_t loss_q16 =
                (12u << 16) - log2_q16(static_cast<std::uint32_t>(pa < 1 ? 1 : pa));
            // Linearised multiplicative update: for small eta*L,
            //   2^(-eta*L) ~= 1 - eta*L*ln2
            // Standard Hedge theory sets eta ~ sqrt(8 ln N / T). With
            // T ~ 2.5e7 bits and N ~ 24 that is ~1e-3 -- three orders of
            // magnitude below the naive choice. Getting this wrong collapses
            // the posterior onto one expert within a few thousand bits and
            // the average degenerates.
            const std::uint32_t decay = ((loss_q16 >> 8) * 96) >> 8;
            const std::uint32_t keepf = 65536u - (decay > 8192u ? 8192u : decay);
            hp_undo_note(w_[d]);
            w_[d] = static_cast<Weight>(
                (static_cast<std::uint64_t>(w_[d]) * keepf) >> 16);
            if (w_[d] < 16) {
                hp_undo_note(w_[d]);
                w_[d] = 16;
            }   // floor: never fully kill an expert
            wsum += w_[d];
        }

        // 2. Renormalise to a fixed total so the weights cannot drift or
        //    underflow. Total is 1<<16.
        if (wsum > 0) {
            for (int d = 0; d < n_; ++d) {
                hp_undo_note(w_[d]);
                w_[d] = static_cast<Weight>(
                    (static_cast<std::uint64_t>(w_[d]) << 16) / wsum);
            }
        }

        // 3. Fixed share: leak sigma of the mass back to uniform. This is the
        //    switching-regret step, and sigma is alpha-driven.
        if (sigma_q16 > 0) {
            const std::uint32_t unif = (1u << 16) / static_cast<std::uint32_t>(n_);
            for (int d = 0; d < n_; ++d) {
                const std::uint64_t keep =
                    (static_cast<std::uint64_t>(w_[d]) * (65536u - sigma_q16)) >> 16;
                const std::uint64_t share =
                    (static_cast<std::uint64_t>(unif) * sigma_q16) >> 16;
                hp_undo_note(w_[d]);
                w_[d] = static_cast<Weight>(keep + share);
            }
        }
    }

    int weight(int d) const { return static_cast<int>(w_[d]); }
    int last() const { return pr_; }
    int size() const { return n_; }

    void merge_from(const Hedge& src, std::uint64_t src_weight, std::uint64_t dst_weight) {
        if (static_cast<int>(src.w_.size()) != n_) {
            return;
        }
        const std::uint64_t total = dst_weight + src_weight;
        if (total == 0) {
            return;
        }
        for (int d = 0; d < n_; ++d) {
            w_[d] = static_cast<Weight>(
                static_cast<std::uint64_t>((static_cast<std::uint64_t>(w_[d]) * dst_weight +
                                            static_cast<std::uint64_t>(src.w_[d]) * src_weight) /
                                           total));
        }
    }

    void copy_from(const Hedge& src) {
        if (static_cast<int>(src.w_.size()) == n_) {
            w_ = src.w_;
        }
    }

    void checkpoint_write(std::ostream& os) const {
        blob::write_pod(os, n_);
        blob::write_vec(os, w_);
        blob::write_vec(os, p_);
        blob::write_pod(os, pr_);
    }

    void checkpoint_read(std::istream& is) {
        blob::read_pod(is, n_);
        blob::read_vec(is, w_);
        blob::read_vec(is, p_);
        blob::read_pod(is, pr_);
    }

 private:
    using Weight = std::uint16_t;
    int n_;
    std::vector<Weight> w_;
    std::vector<int> p_;
    int pr_ = 2048;
};

}  // namespace hp
