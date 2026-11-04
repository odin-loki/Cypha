// An online n-gram expert, PAQ-style (Odin Loch, 2026-11-04; docs/reports/ONLINE_LEARNING_UPGRADES.md). Every byte is
// 8 binary predictions, most significant bit first, with the partial byte c0 as PAQ keeps it. For each order k (the k
// bytes before, then c0) a hashed table holds fixed-point bit counts (x128, 16 bits, halved past a cap), as the bitwise
// GLN's counts (bitwise_gln.hpp). A small gated logistic mixer joins the orders' bit predictions in the stretch domain;
// its weight set is chosen by the bit position and the longest order whose context has been seen. P(byte) is the product
// of the 8 mixed bit probabilities; each order alone gives one too (the product of its own 8).
// It pretrains its counts on CyphaLM's training text (counts only), then predicts and learns (counts and mixer) over the
// training text's last bytes and on, byte by byte: each byte is predicted before it is learned.
// Rule 10 (PLAN): an expert inside CyphaLM. It answers nothing.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
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
            n1_.emplace_back(std::size_t{1} << bits, 0);
            n_.emplace_back(std::size_t{1} << bits, 0);
            if (p.check) ck_.emplace_back(std::size_t{1} << bits, 0);
        }
        K_ = O_ + 1;  // the orders and a bias
        W_.assign(static_cast<std::size_t>(8) * (O_ + 1) * K_, 0.3);
        for (std::size_t q = K_ - 1; q < W_.size(); q += K_) W_[q] = 0.0;
        hist_.assign(64, 32);
    }

    int orders() const { return O_; }

    // Counts only, over text[0, n) (the history then holds text's last bytes).
    void pretrain(const std::uint8_t* text, std::size_t n) {
        for (std::size_t t = 0; t < n; ++t) {
            ctx();
            unsigned c0 = 1;
            for (int j = 0; j < 8; ++j) {
                const unsigned bit = (text[t] >> (7 - j)) & 1u;
                for (int q = 0; q < O_; ++q) {
                    const std::size_t key = slot(q, c0);
                    if (p_.check) claim(q, key, c0);
                    count(q, key, bit);
                }
                c0 = (c0 << 1) | bit;
            }
            push(text[t]);
        }
    }

    // The mixed probability of byte y and each order's (po[q], q < orders()); then learns y (when learn).
    double code(std::uint8_t y, double* po, bool learn = true) {
        ctx();
        double pm = 1.0;
        for (int q = 0; q < O_; ++q) po[q] = 1.0;
        unsigned c0 = 1;
        std::size_t keys[32];
        double x[33];
        for (int j = 0; j < 8; ++j) {
            const unsigned bit = (y >> (7 - j)) & 1u;
            int seen = 0;
            for (int q = 0; q < O_; ++q) {
                keys[q] = slot(q, c0);
                if (p_.check) claim(q, keys[q], c0);
                const std::uint16_t c = n_[q][keys[q]];
                const double p = (n1_[q][keys[q]] / 128.0 + 0.4) / (c / 128.0 + 0.8);
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
                for (int q = 0; q < O_; ++q) count(q, keys[q], bit);
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
        for (unsigned c0 = 1; c0 < 256; ++c0) {
            const int j = 31 - __builtin_clz(c0);
            int seen = 0;
            for (int q = 0; q < O_; ++q) {
                const std::size_t key = slot(q, c0);
                const bool mine = !p_.check || ck_[static_cast<std::size_t>(q)][key] == check(q, c0);
                const std::uint16_t c = mine ? n_[q][key] : 0, c1 = mine ? n1_[q][key] : 0;
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
        for (const auto& t : n_) b += 2 * t.size() * sizeof(std::uint16_t);
        for (const auto& t : ck_) b += t.size();
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
    std::size_t slot(int q, unsigned c0) const {
        const std::uint64_t h = hh_[static_cast<std::size_t>(q)];
        const std::uint64_t k = static_cast<std::uint64_t>(p_.orders[static_cast<std::size_t>(q)]);
        return static_cast<std::size_t>(((h * 0x9E3779B97F4A7C15ull) ^ (static_cast<std::uint64_t>(c0) * 0x100000001B3ull) ^
                                         (k * 0xC2B2AE3D27D4EB4Full)) >> 7 & mask_[static_cast<std::size_t>(q)]);
    }
    // The slot's check (from the hash's high bits, never 0); a slot held by another context is cleared and taken over.
    std::uint8_t check(int q, unsigned c0) const {
        const std::uint64_t h = hh_[static_cast<std::size_t>(q)] * 0xD6E8FEB86659FD93ull + c0 * 0x9E3779B97F4A7C15ull;
        return static_cast<std::uint8_t>((h >> 56) | 1u);
    }
    void claim(int q, std::size_t key, unsigned c0) {
        const std::uint8_t c = check(q, c0);
        std::uint8_t& s = ck_[static_cast<std::size_t>(q)][key];
        if (s != c) {
            s = c;
            n1_[static_cast<std::size_t>(q)][key] = n_[static_cast<std::size_t>(q)][key] = 0;
        }
    }
    void count(int q, std::size_t key, unsigned bit) {
        std::uint16_t& c1 = n1_[static_cast<std::size_t>(q)][key];
        std::uint16_t& c = n_[static_cast<std::size_t>(q)][key];
        c1 = static_cast<std::uint16_t>(c1 + 128u * bit);
        c = static_cast<std::uint16_t>(c + 128u);
        if (c > static_cast<unsigned>(p_.cap) * 128u) {
            c1 = static_cast<std::uint16_t>(c1 / 2);
            c = static_cast<std::uint16_t>(c / 2);
        }
    }
    void push(std::uint8_t y) {
        std::rotate(hist_.begin(), hist_.begin() + 1, hist_.end());
        hist_.back() = y;
    }

    Params p_;
    int O_ = 0, K_ = 0;
    std::vector<std::uint64_t> mask_, hh_;
    std::vector<std::vector<std::uint16_t>> n1_, n_;
    std::vector<std::vector<std::uint8_t>> ck_;
    std::vector<double> W_;
    std::vector<std::uint8_t> hist_;
};

}  // namespace cypha::online
