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
// OPT_SIMD (2026-11-05, GENERATION_PLAN §13.41 O5), every number unchanged again, the same hashing and replacement: the
// tables are asked for 2 MB pages (Linux: madvise MADV_HUGEPAGE), so a random lookup misses the TLB less when the OS has
// them; dist() computes each key once (it used to hash twice, to prefetch and to read). And less arithmetic, each value
// by the same expressions: a slot's probability and stretch for counts that were never halved come from a table made at
// the start; dist() forms the 256 byte products down the bit tree (shared prefixes once, each product in the old order);
// the parts of the slot hash that do not depend on the partial byte are multiplied once a byte.
// Rule 10 (PLAN): an expert inside CyphaLM. It answers nothing.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "cypha/online/byte_lstm.hpp"  // gln_stretch, gln_squash

namespace cypha::online {

// OPT_SIMD: a table of n zeroed bytes in whole 2 MB pages, 2 MB aligned. On Linux it is mapped anonymously and its pages
// are asked to be huge; elsewhere it is 64-byte aligned memory, zeroed.
class NgramPages {
public:
    static constexpr std::size_t kHuge = std::size_t{1} << 21;
    NgramPages() = default;
    explicit NgramPages(std::size_t n) { alloc(n); }
    NgramPages(const NgramPages& o) {
        alloc(o.n_);
        if (n_) std::memcpy(p_, o.p_, n_);
    }
    NgramPages& operator=(const NgramPages& o) {
        if (this != &o) {
            release();
            alloc(o.n_);
            if (n_) std::memcpy(p_, o.p_, n_);
        }
        return *this;
    }
    NgramPages(NgramPages&& o) noexcept : p_(o.p_), n_(o.n_), cap_(o.cap_), mapped_(o.mapped_) { o.p_ = nullptr, o.n_ = o.cap_ = 0; }
    NgramPages& operator=(NgramPages&& o) noexcept {
        if (this != &o) {
            release();
            p_ = o.p_, n_ = o.n_, cap_ = o.cap_, mapped_ = o.mapped_;
            o.p_ = nullptr, o.n_ = o.cap_ = 0;
        }
        return *this;
    }
    ~NgramPages() { release(); }
    unsigned char* data() { return p_; }
    const unsigned char* data() const { return p_; }

private:
    void alloc(std::size_t n) {
        n_ = n;
        cap_ = (n + kHuge - 1) / kHuge * kHuge;
        if (!cap_) return;
#if defined(__linux__)
        void* m = ::mmap(nullptr, cap_ + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m != MAP_FAILED) {  // keep the 2 MB aligned part
            const auto a = reinterpret_cast<std::uintptr_t>(m), b = (a + kHuge - 1) & ~(kHuge - 1), e = a + cap_ + kHuge;
            if (b > a) ::munmap(m, b - a);
            if (e > b + cap_) ::munmap(reinterpret_cast<void*>(b + cap_), e - (b + cap_));
            p_ = reinterpret_cast<unsigned char*>(b);
            mapped_ = true;
#if defined(MADV_HUGEPAGE)
            ::madvise(p_, cap_, MADV_HUGEPAGE);
#endif
            // Touch every page now, in order, as the old zero-filled vector did: faulting them in later, at random
            // from the pretrain threads, measured slower.
            std::memset(p_, 0, cap_);
            return;
        }
#endif
        p_ = static_cast<unsigned char*>(::operator new(cap_, std::align_val_t{64}));
        std::memset(p_, 0, cap_);
        mapped_ = false;
    }
    void release() {
        if (!p_) return;
#if defined(__linux__)
        if (mapped_) {
            ::munmap(p_, cap_);
            p_ = nullptr;
            return;
        }
#endif
        ::operator delete(p_, std::align_val_t{64});
        p_ = nullptr;
    }
    unsigned char* p_ = nullptr;
    std::size_t n_ = 0, cap_ = 0;
    bool mapped_ = false;
};

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
        t_.reserve(p.orders.size());
        for (int o : p.orders) {
            const int bits = std::min(p.tb, 8 * o + 10);
            mask_.push_back((std::uint64_t{1} << bits) - 1);
            t_.emplace_back(std::size_t{1} << bits);
        }
        K_ = O_ + 1;  // the orders and a bias
        W_.assign(static_cast<std::size_t>(8) * (O_ + 1) * K_, 0.3);
        for (std::size_t q = K_ - 1; q < W_.size(); q += K_) W_[q] = 0.0;
        hist_.assign(64, 32);
        // OPT_SIMD: a slot's probability and stretch depend on its two counts only. For counts that are whole multiples
        // of 128 (a slot never halved), up to 63 x 128, they are made here by the same expressions and read back.
        tabL_ = std::clamp(p_.cap + 1, 0, 63);
        const int L = tabL_ + 1;
        ptab_.resize(static_cast<std::size_t>(L) * L);
        xtab_.resize(static_cast<std::size_t>(L) * L);
        for (int a = 0; a < L; ++a)
            for (int b = 0; b < L; ++b) {
                const std::uint16_t c = static_cast<std::uint16_t>(a * 128), c1 = static_cast<std::uint16_t>(b * 128);
                const double p = (c1 / 128.0 + 0.4) / (c / 128.0 + 0.8);
                ptab_[static_cast<std::size_t>(a) * L + b] = p;
                xtab_[static_cast<std::size_t>(a) * L + b] = c ? gln_stretch(p) : 0.0;
            }
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
            Key kq[32];
            std::size_t keys[8][32];
            for (std::size_t i = 0; i < n; ++i) {
                for (int q = q0; q < q1; ++q) {
                    std::uint64_t v = 0;
                    const int o = p_.orders[static_cast<std::size_t>(q)];
                    for (int k = 1; k <= o; ++k) v = v * 1099511628211ull + (i >= static_cast<std::size_t>(k) ? text[i - k] : hist_at(i, k, text)) + 1u;
                    kq[q] = key_of(q, v);
                }
                const unsigned y = text[i];
                unsigned c0 = 1;
                for (int j = 0; j < 8; ++j) {
                    for (int q = q0; q < q1; ++q) {
                        keys[j][q] = kslot(kq[q], c0);
                        prefetch(q, keys[j][q]);
                    }
                    c0 = (c0 << 1) | ((y >> (7 - j)) & 1u);
                }
                c0 = 1;
                for (int j = 0; j < 8; ++j) {
                    const unsigned bit = (y >> (7 - j)) & 1u;
                    for (int q = q0; q < q1; ++q) {
                        Slot& sl = t_[static_cast<std::size_t>(q)][keys[j][q]];
                        if (p_.check) claim(sl, kcheck(kq[q], c0));
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
                double p, xq;
                pstretch(sl.n1, c, p, xq);
                po[q] *= bit ? p : 1 - p;
                x[q] = xq;
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
        std::size_t ring[16][32];       // OPT_SIMD: the keys of the partial bytes prefetched ahead, computed once
        constexpr unsigned kAhead = 8;  // partial bytes prefetched ahead
        for (unsigned c0 = 1; c0 < 256 && c0 <= kAhead; ++c0)
            for (int q = 0; q < O_; ++q) prefetch(q, ring[c0 & 15][q] = slot(q, c0));
        for (unsigned c0 = 1; c0 < 256; ++c0) {
            if (c0 + kAhead < 256)
                for (int q = 0; q < O_; ++q) prefetch(q, ring[(c0 + kAhead) & 15][q] = slot(q, c0 + kAhead));
            const int j = static_cast<int>(std::bit_width(c0)) - 1;  // the bit position: floor(log2(c0))
            int seen = 0;
            for (int q = 0; q < O_; ++q) {
                const Slot& sl = t_[static_cast<std::size_t>(q)][ring[c0 & 15][q]];
                const bool mine = !p_.check || sl.ck == check(q, c0);
                const std::uint16_t c = mine ? sl.n : 0, c1 = mine ? sl.n1 : 0;
                double p, xq;
                pstretch(c1, c, p, xq);
                pq[q][c0] = p;
                x[q] = xq;
                if (c) seen = q + 1;
            }
            x[O_] = 1.0;
            const double* w = &W_[(static_cast<std::size_t>(j) * (O_ + 1) + seen) * K_];
            double s = 0;
            for (int k = 0; k < K_; ++k) s += w[k] * x[k];
            pn[c0] = std::clamp(gln_squash(s), 1e-5, 1 - 1e-5);
        }
        tree(pn, P);
        for (int q = 0; q < O_; ++q) tree(pq[q], Po + q * 256);
    }

    std::size_t bytes() const {
        std::size_t b = W_.size() * sizeof(double);
        for (const auto& t : t_) b += t.size() * (p_.check ? 5 : 4);  // as before: two 16-bit counts and the check
        return b;
    }

private:
    // An order's context hash h, with the parts of the slot hash and the check that do not depend on the partial byte
    // multiplied out (OPT_SIMD): slot = ((h A) ^ (c0 B) ^ (k C)) >> 7 & mask, check = ((h D + c0 E) >> 56) | 1, as before.
    struct Key {
        std::uint64_t hk = 0, hc = 0, mask = 0;  // (h A) ^ (k C); h D; the table's mask
    };
    Key key_of(int q, std::uint64_t h) const {
        const std::uint64_t k = static_cast<std::uint64_t>(p_.orders[static_cast<std::size_t>(q)]);
        return {(h * 0x9E3779B97F4A7C15ull) ^ (k * 0xC2B2AE3D27D4EB4Full), h * 0xD6E8FEB86659FD93ull,
                mask_[static_cast<std::size_t>(q)]};
    }
    static std::size_t kslot(const Key& k, unsigned c0) {
        return static_cast<std::size_t>((k.hk ^ (static_cast<std::uint64_t>(c0) * 0x100000001B3ull)) >> 7 & k.mask);
    }
    // The slot's check (from the hash's high bits, never 0); a slot held by another context is cleared and taken over.
    static std::uint8_t kcheck(const Key& k, unsigned c0) {
        const std::uint64_t v = k.hc + c0 * 0x9E3779B97F4A7C15ull;
        return static_cast<std::uint8_t>((v >> 56) | 1u);
    }
    // The orders' context hashes from the history (kept until the history moves).
    void ctx() {
        if (ctx_ok_) return;
        kq_.resize(static_cast<std::size_t>(O_));
        for (int q = 0; q < O_; ++q) {
            std::uint64_t v = 0;
            for (int i = 1; i <= p_.orders[static_cast<std::size_t>(q)]; ++i) v = v * 1099511628211ull + hist_[hist_.size() - i] + 1u;
            kq_[static_cast<std::size_t>(q)] = key_of(q, v);
        }
        ctx_ok_ = true;
    }
    // A slot's probability of a one and its stretch (0 when unseen), from its counts: (c1 / 128 + 0.4) / (c / 128 + 0.8)
    // and gln_stretch of it, as before. Counts that are multiples of 128 read the table made by the same expressions;
    // both counts 0 is that table's first entry.
    void pstretch(std::uint16_t c1, std::uint16_t c, double& p, double& x) const {
        if (((c | c1) & 127u) == 0 && (c >> 7) <= tabL_ && (c1 >> 7) <= tabL_) {
            const std::size_t i = static_cast<std::size_t>(c >> 7) * static_cast<std::size_t>(tabL_ + 1) + (c1 >> 7);
            p = ptab_[i];
            x = xtab_[i];
            return;
        }
        p = (c1 / 128.0 + 0.4) / (c / 128.0 + 0.8);
        x = c ? gln_stretch(p) : 0.0;
    }
    // P(byte) for all 256 bytes from the bit tree's probabilities of a one pr[c0] (OPT_SIMD): a node's product is its
    // parent's times its edge's probability, so each byte's product is formed in the old order (1, times bit 0's, times
    // bit 1's, ...), and the shared prefixes once.
    static void tree(const double* pr, double* out) {
        double prod[256];
        prod[1] = 1.0;
        for (unsigned c0 = 1; c0 < 128; ++c0) {
            const double a = prod[c0], p = pr[c0];
            prod[2 * c0] = a * (1 - p);
            prod[2 * c0 + 1] = a * p;
        }
        for (unsigned c0 = 128; c0 < 256; ++c0) {
            const double a = prod[c0], p = pr[c0];
            out[2 * c0 - 256] = a * (1 - p);
            out[2 * c0 - 255] = a * p;
        }
    }
#pragma pack(push, 1)
    struct Slot {  // one hashed context: counts x128 (ones, all) and the 8-bit check
        std::uint16_t n1 = 0, n = 0;
        std::uint8_t ck = 0;
    };
#pragma pack(pop)
    static_assert(sizeof(Slot) == 5, "a slot is 5 bytes");
    // OPT_SIMD: an order's table of zeroed slots, 5 bytes each, on huge pages where the OS gives them. (Twelve slots to a
    // 64-byte line, so that none crosses two lines, measured slower: the index arithmetic cost more than the 1 in 16
    // slots that cross.)
    struct Table {
        NgramPages pg;
        std::size_t n = 0;
        explicit Table(std::size_t slots) : pg(slots * sizeof(Slot)), n(slots) {}
        Slot& operator[](std::size_t i) { return reinterpret_cast<Slot*>(pg.data())[i]; }
        const Slot& operator[](std::size_t i) const { return reinterpret_cast<const Slot*>(pg.data())[i]; }
        std::size_t size() const { return n; }
        const void* addr(std::size_t i) const { return pg.data() + i * sizeof(Slot); }
    };

    std::size_t slot(int q, unsigned c0) const { return kslot(kq_[static_cast<std::size_t>(q)], c0); }
    std::uint8_t check(int q, unsigned c0) const { return kcheck(kq_[static_cast<std::size_t>(q)], c0); }
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
        __builtin_prefetch(t_[static_cast<std::size_t>(q)].addr(key), 1);
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
        ctx_ok_ = false;
    }

    Params p_;
    int O_ = 0, K_ = 0;
    bool ctx_ok_ = false;  // kq_ is the current history's
    int tabL_ = 0;         // ptab_ and xtab_ hold counts up to tabL_ x 128
    std::vector<std::uint64_t> mask_;
    std::vector<Key> kq_;  // each order's context key
    std::vector<Table> t_;
    std::vector<double> W_;
    std::vector<double> ptab_, xtab_;
    std::vector<std::uint8_t> hist_;
};

}  // namespace cypha::online
