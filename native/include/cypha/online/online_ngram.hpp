// An online n-gram expert, PAQ-style (Odin Loch, 2026-11-04; docs/reports/ONLINE_LEARNING_UPGRADES.md). Every byte is
// 8 binary predictions, most significant bit first, with the partial byte c0 as PAQ keeps it. For each order k (the k
// bytes before, then c0) a hashed table holds fixed-point bit counts (x128, 16 bits, halved past a cap), as the bitwise
// GLN's counts (bitwise_gln.hpp). A small gated logistic mixer joins the orders' bit predictions in the stretch domain;
// its weight set is chosen by the bit position and the longest order whose context has been seen. P(byte) is the product
// of the 8 mixed bit probabilities; each order alone gives one too (the product of its own 8).
// It pretrains its counts on CyphaLM's training text (counts only), then predicts and learns (counts and mixer) over the
// training text's last bytes and on, byte by byte: each byte is predicted before it is learned.
// Optimisation (2026-11-05), every number unchanged: a slot's count, count of ones and check are packed in 5 bytes (one
// cache line a lookup instead of three, the same memory); every key a byte needs is computed and prefetched before the
// counts are read (the byte, or every partial byte for dist, is known in advance); pretrain fills the orders' tables on
// a few threads (CYPHA_NGRAM_THREADS, 4 by default), each table in its original order.
// Rule 10 (PLAN): an expert inside CyphaLM. It answers nothing.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "cypha/online/byte_lstm.hpp"  // gln_stretch, gln_squash

namespace cypha::online {

class OnlineNgram {
public:
    struct Params {
        std::vector<int> orders{0, 1, 2, 3, 4, 5, 6};
        int tb = 22;        // hash table bits for the high orders (an order k table holds at most 2^(8k+10) slots)
        int cap = 255;      // counts are halved past cap (in whole counts)
        double lr = 0.02;   // the mixer's step
        bool check = true;  // an 8-bit check per slot: a slot another context holds reads as unseen and is taken over
    };

    explicit OnlineNgram(const Params& p) : p_(p) {
        O_ = static_cast<int>(p.orders.size());
        for (int o : p.orders) {
            const int bits = std::min(p.tb, 8 * o + 10);
            mask_.push_back((std::uint64_t{1} << bits) - 1);
            t_.emplace_back(std::size_t{1} << bits, Slot{});
        }
        K_ = O_ + 1;  // the orders and a bias
        W_.assign(static_cast<std::size_t>(8) * (O_ + 1) * K_, 0.3);
        for (std::size_t q = K_ - 1; q < W_.size(); q += K_) W_[q] = 0.0;
        hist_.assign(64, 32);
    }

    int orders() const { return O_; }

    // Counts only, over text[0, n) (the history then holds text's last bytes). The orders' tables are independent, so
    // they are filled on a few threads, each table in the original order.
    void pretrain(const std::uint8_t* text, std::size_t n) {
        int nt = 4;
        if (const char* e = std::getenv("CYPHA_NGRAM_THREADS")) nt = std::max(1, std::atoi(e));
        nt = std::min(nt, O_);
        auto part = [&](int t) {
            const int q0 = O_ * t / nt, q1 = O_ * (t + 1) / nt;
            std::uint64_t hh[32];
            std::size_t keys[8][32];
            for (std::size_t i = 0; i < n; ++i) {
                for (int q = q0; q < q1; ++q) {
                    std::uint64_t v = 0;
                    const int o = p_.orders[static_cast<std::size_t>(q)];
                    for (int k = 1; k <= o; ++k) v = v * 1099511628211ull + (i >= static_cast<std::size_t>(k) ? text[i - k] : hist_at(i, k, text)) + 1u;
                    hh[q] = v;
                }
                const unsigned y = text[i];
                unsigned c0 = 1;
                for (int j = 0; j < 8; ++j) {
                    for (int q = q0; q < q1; ++q) {
                        keys[j][q] = slot_of(q, hh[q], c0);
                        prefetch(q, keys[j][q]);
                    }
                    c0 = (c0 << 1) | ((y >> (7 - j)) & 1u);
                }
                c0 = 1;
                for (int j = 0; j < 8; ++j) {
                    const unsigned bit = (y >> (7 - j)) & 1u;
                    for (int q = q0; q < q1; ++q) {
                        Slot& sl = t_[static_cast<std::size_t>(q)][keys[j][q]];
                        if (p_.check) claim(sl, check_of(hh[q], c0));
                        count(sl, bit);
                    }
                    c0 = (c0 << 1) | bit;
                }
            }
        };
        if (nt <= 1) {
            part(0);
        } else {
            std::vector<std::thread> th;
            for (int t = 1; t < nt; ++t) th.emplace_back(part, t);
            part(0);
            for (auto& x : th) x.join();
        }
        for (std::size_t i = n > hist_.size() ? n - hist_.size() : 0; i < n; ++i) push(text[i]);  // the last bytes
    }

    // The mixed probability of byte y and each order's (po[q], q < orders()); then learns y (when learn).
    double code(std::uint8_t y, double* po, bool learn = true) {
        ctx();
        double pm = 1.0;
        for (int q = 0; q < O_; ++q) po[q] = 1.0;
        std::size_t keys[8][32];
        {
            unsigned c0 = 1;
            for (int j = 0; j < 8; ++j) {
                for (int q = 0; q < O_; ++q) {
                    keys[j][q] = slot(q, c0);
                    prefetch(q, keys[j][q]);
                }
                c0 = (c0 << 1) | ((y >> (7 - j)) & 1u);
            }
        }
        unsigned c0 = 1;
        double x[33];
        for (int j = 0; j < 8; ++j) {
            const unsigned bit = (y >> (7 - j)) & 1u;
            int seen = 0;
            for (int q = 0; q < O_; ++q) {
                Slot& sl = t_[static_cast<std::size_t>(q)][keys[j][q]];
                if (p_.check) claim(sl, check(q, c0));
                const std::uint16_t c = sl.n;
                const double p = (sl.n1 / 128.0 + 0.4) / (c / 128.0 + 0.8);
                po[q] *= bit ? p : 1 - p;
                x[q] = c ? gln_stretch(p) : 0.0;
                if (c) seen = q + 1;
            }
            x[O_] = 1.0;
            double* w = &W_[(static_cast<std::size_t>(j) * (O_ + 1) + seen) * K_];
            double s = 0;
            for (int k = 0; k < K_; ++k) s += w[k] * x[k];
            const double p = std::clamp(gln_squash(s), 1e-5, 1 - 1e-5);
            pm *= bit ? p : 1 - p;
            if (learn) {
                const double e = bit - p;
                for (int k = 0; k < K_; ++k) w[k] = std::clamp(w[k] + p_.lr * e * x[k], -16.0, 16.0);
                for (int q = 0; q < O_; ++q) count(t_[static_cast<std::size_t>(q)][keys[j][q]], bit);
            }
            c0 = (c0 << 1) | bit;
        }
        push(y);
        return pm;
    }

    // FULL_DISTRIBUTION_MIXING.md (context8k --full-mix): the whole next-byte distribution, mixed (P[256]) and each
    // order's (Po[q * 256 + s]), from the state code() predicts from, learning nothing. A slot that another context
    // holds reads as unseen, as claim() leaves it. code(y) then gives P[y] (unless two nodes of one byte share a slot).
    void dist(double* P, double* Po) {
        ctx();
        double pn[256], pq[32][256];
        double x[33];
        constexpr unsigned kAhead = 8;  // partial bytes prefetched ahead
        for (unsigned c0 = 1; c0 < 256 && c0 <= kAhead; ++c0)
            for (int q = 0; q < O_; ++q) prefetch(q, slot(q, c0));
        for (unsigned c0 = 1; c0 < 256; ++c0) {
            if (c0 + kAhead < 256)
                for (int q = 0; q < O_; ++q) prefetch(q, slot(q, c0 + kAhead));
            int j = 0;
            while ((c0 >> (j + 1)) != 0) ++j;  // the bit position: floor(log2(c0))
            int seen = 0;
            for (int q = 0; q < O_; ++q) {
                const std::size_t key = slot(q, c0);
                const Slot& sl = t_[static_cast<std::size_t>(q)][key];
                const bool mine = !p_.check || sl.ck == check(q, c0);
                const std::uint16_t c = mine ? sl.n : 0, c1 = mine ? sl.n1 : 0;
                const double p = (c1 / 128.0 + 0.4) / (c / 128.0 + 0.8);
                pq[q][c0] = p;
                x[q] = c ? gln_stretch(p) : 0.0;
                if (c) seen = q + 1;
            }
            x[O_] = 1.0;
            const double* w = &W_[(static_cast<std::size_t>(j) * (O_ + 1) + seen) * K_];
            double s = 0;
            for (int k = 0; k < K_; ++k) s += w[k] * x[k];
            pn[c0] = std::clamp(gln_squash(s), 1e-5, 1 - 1e-5);
        }
        for (unsigned y = 0; y < 256; ++y) {
            double pm = 1.0;
            for (int q = 0; q < O_; ++q) Po[q * 256 + y] = 1.0;
            unsigned c0 = 1;
            for (int j = 0; j < 8; ++j) {
                const unsigned bit = (y >> (7 - j)) & 1u;
                pm *= bit ? pn[c0] : 1 - pn[c0];
                for (int q = 0; q < O_; ++q) Po[q * 256 + y] *= bit ? pq[q][c0] : 1 - pq[q][c0];
                c0 = (c0 << 1) | bit;
            }
            P[y] = pm;
        }
    }

    std::size_t bytes() const {
        std::size_t b = W_.size() * sizeof(double);
        for (const auto& t : t_) b += t.size() * (p_.check ? 5 : 4);  // as before: two 16-bit counts and the check
        return b;
    }

private:
    void ctx() {
        hh_.resize(static_cast<std::size_t>(O_));
        for (int q = 0; q < O_; ++q) {
            std::uint64_t v = 0;
            for (int i = 1; i <= p_.orders[static_cast<std::size_t>(q)]; ++i) v = v * 1099511628211ull + hist_[hist_.size() - i] + 1u;
            hh_[static_cast<std::size_t>(q)] = v;
        }
    }
#pragma pack(push, 1)
    struct Slot {  // one hashed context: counts x128 (ones, all) and the 8-bit check
        std::uint16_t n1 = 0, n = 0;
        std::uint8_t ck = 0;
    };
#pragma pack(pop)
    static_assert(sizeof(Slot) == 5, "a slot is 5 bytes");

    std::size_t slot_of(int q, std::uint64_t h, unsigned c0) const {
        const std::uint64_t k = static_cast<std::uint64_t>(p_.orders[static_cast<std::size_t>(q)]);
        return static_cast<std::size_t>(((h * 0x9E3779B97F4A7C15ull) ^ (static_cast<std::uint64_t>(c0) * 0x100000001B3ull) ^
                                         (k * 0xC2B2AE3D27D4EB4Full)) >> 7 & mask_[static_cast<std::size_t>(q)]);
    }
    std::size_t slot(int q, unsigned c0) const { return slot_of(q, hh_[static_cast<std::size_t>(q)], c0); }
    // The slot's check (from the hash's high bits, never 0); a slot held by another context is cleared and taken over.
    static std::uint8_t check_of(std::uint64_t h, unsigned c0) {
        const std::uint64_t v = h * 0xD6E8FEB86659FD93ull + c0 * 0x9E3779B97F4A7C15ull;
        return static_cast<std::uint8_t>((v >> 56) | 1u);
    }
    std::uint8_t check(int q, unsigned c0) const { return check_of(hh_[static_cast<std::size_t>(q)], c0); }
    static void claim(Slot& sl, std::uint8_t c) {
        if (sl.ck != c) {
            sl.ck = c;
            sl.n1 = sl.n = 0;
        }
    }
    void count(Slot& sl, unsigned bit) const {
        std::uint16_t c1 = sl.n1, c = sl.n;
        c1 = static_cast<std::uint16_t>(c1 + 128u * bit);
        c = static_cast<std::uint16_t>(c + 128u);
        if (c > static_cast<unsigned>(p_.cap) * 128u) {
            c1 = static_cast<std::uint16_t>(c1 / 2);
            c = static_cast<std::uint16_t>(c / 2);
        }
        sl.n1 = c1;
        sl.n = c;
    }
    void prefetch(int q, std::size_t key) const {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&t_[static_cast<std::size_t>(q)][key], 1);
#else
        (void)q;
        (void)key;
#endif
    }
    // The byte k back from position i of the text pretrain reads, before the text starts: the history before it.
    std::uint8_t hist_at(std::size_t i, int k, const std::uint8_t*) const {
        return hist_[hist_.size() - static_cast<std::size_t>(k) + i];
    }
    void push(std::uint8_t y) {
        std::rotate(hist_.begin(), hist_.begin() + 1, hist_.end());
        hist_.back() = y;
    }

    Params p_;
    int O_ = 0, K_ = 0;
    std::vector<std::uint64_t> mask_, hh_;
    std::vector<std::vector<Slot>> t_;
    std::vector<double> W_;
    std::vector<std::uint8_t> hist_;
};

}  // namespace cypha::online
