// One count index and one Kneser-Ney core for CyphaLM (docs/GENERATION_PLAN.md §13.40, merge M1;
// docs/reports/MERGE_COUNTS.md).
//
// CountIndex: one suffix array over a sequence of symbols (word ids for K5, lm/word_kn.hpp; bytes for byte
// Kneser-Ney, lm/byte_kn.hpp, and SCRIBE's MATCH, eval/scribe.hpp). From it come
//   - the count of any string, of any length (infini-gram, Liu et al. 2024: a suffix array gives counts of every order);
//   - the symbols that follow a string, in ascending order, each with its count (one sub-interval per symbol);
//   - continuation counts, the distinct symbols before a string (Kneser-Ney's N1+(. c w)); Shareghi, Petri, Haffari and
//     Cohn (TACL 2016) compute Kneser-Ney of every order from one suffix structure this way;
//   - the longest prefix of a pattern that occurs (matching statistics: K5's copy cap, SCRIBE's MATCH).
// The suffix array is libsais's (induced sorting, O(n)): a shorter suffix sorts before a longer one it begins, so the
// suffixes that start with a string form one interval, sorted by the symbol after it.
//
// kn: the Kneser-Ney core that byte KN and word KN share (Chen and Goodman's interpolated modified Kneser-Ney; byte KN
// and word KN are the same maths over different symbols, Teh 2006). The discounts of one order from its counts of
// counts, and one context's values from its entries' counts. Both keep the floating-point order of the two estimators
// they replace, so their tables are the same bit for bit.
//
// Rule 10 (PLAN): CyphaLM's own counts. No neural network; it answers nothing.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

// third_party/libsais (libsais.h): a target that builds an index compiles third_party/libsais/libsais.c. Declared here
// so that the header needs no include path where no index is built.
extern "C" {
std::int32_t libsais(const std::uint8_t* T, std::int32_t* SA, std::int32_t n, std::int32_t fs, std::int32_t* freq);
std::int32_t libsais_int(std::int32_t* T, std::int32_t* SA, std::int32_t n, std::int32_t k, std::int32_t fs);
}

namespace cypha::cyphalm {

namespace cidx {

// S: std::uint8_t (bytes) or std::uint32_t (ids below sigma <= 2^31).
template <class S>
class CountIndex {
    static_assert(std::is_same_v<S, std::uint8_t> || std::is_same_v<S, std::uint32_t>, "CountIndex: bytes or 32-bit ids");

 public:
    using Pos = std::uint32_t;
    struct Range {
        Pos lo = 0, hi = 0;  // suffix-array ranks [lo, hi)
        Pos size() const { return hi - lo; }
        bool empty() const { return hi <= lo; }
    };

    // The index keeps text (moved in). sigma: every symbol is below it (256 for bytes).
    void build(std::vector<S> text, std::uint32_t sigma) {
        if (text.size() >= (std::size_t{1} << 31)) throw std::runtime_error("CountIndex: text over 2^31 symbols");
        t_ = std::move(text);
        sigma_ = sigma;
        const auto n = static_cast<std::int32_t>(t_.size());
        sa_.assign(t_.size(), 0);
        c_.assign(static_cast<std::size_t>(sigma) + 1, 0);
        for (const S s : t_) {
            if (static_cast<std::uint32_t>(s) >= sigma) throw std::runtime_error("CountIndex: a symbol is not below sigma");
            ++c_[static_cast<std::size_t>(s) + 1];
        }
        for (std::size_t s = 0; s < sigma; ++s) c_[s + 1] += c_[s];
        if (n > 0) {
            std::int32_t r = 0;
            if constexpr (std::is_same_v<S, std::uint8_t>) {
                r = libsais(t_.data(), reinterpret_cast<std::int32_t*>(sa_.data()), n, 0, nullptr);
            } else {
                // libsais_int changes T while it sorts and restores it at the end.
                r = libsais_int(reinterpret_cast<std::int32_t*>(t_.data()), reinterpret_cast<std::int32_t*>(sa_.data()), n,
                                static_cast<std::int32_t>(sigma), 0);
            }
            if (r != 0) throw std::runtime_error("CountIndex: libsais failed");
        }
    }
    void clear() {
        std::vector<S>().swap(t_);
        std::vector<Pos>().swap(sa_);
        std::vector<Pos>().swap(c_);
    }
    bool built() const { return !c_.empty(); }
    std::size_t size() const { return t_.size(); }
    std::uint32_t sigma() const { return sigma_; }
    const std::vector<S>& text() const { return t_; }
    std::size_t bytes() const { return t_.capacity() * sizeof(S) + sa_.capacity() * 4 + c_.capacity() * 4; }

    Pos sa(std::size_t i) const { return sa_[i]; }
    // Does the suffix of rank i have a symbol at offset d, and which.
    bool has(std::size_t i, std::size_t d) const { return sa_[i] + d < t_.size(); }
    S at(std::size_t i, std::size_t d) const { return t_[sa_[i] + d]; }
    Range all() const { return {0, static_cast<Pos>(t_.size())}; }
    // The suffixes that start with symbol s.
    Range first(S s) const { return {c_[s], c_[static_cast<std::size_t>(s) + 1]}; }

    // Inside r, whose suffixes share their first d symbols: those whose symbol d is s. The suffix that ends after d
    // symbols (at most one) sorts first and is never kept.
    Range narrow(Range r, std::size_t d, S s) const {
        if (static_cast<std::uint32_t>(s) >= sigma_) return {r.lo, r.lo};
        if (d == 0 && r.lo == 0 && r.hi == t_.size()) return first(s);
        const std::size_t n = t_.size();
        const auto key = [&](Pos i) -> std::int64_t {
            const std::size_t p = std::size_t{sa_[i]} + d;
            return p < n ? static_cast<std::int64_t>(t_[p]) : -1;
        };
        Pos a = r.lo, b = r.hi;
        while (a < b) {  // the first rank with key >= s
            const Pos m = a + (b - a) / 2;
            if (key(m) < static_cast<std::int64_t>(s)) a = m + 1;
            else b = m;
        }
        Pos c = a, e = r.hi;
        while (c < e) {  // the first rank with key > s
            const Pos m = c + (e - c) / 2;
            if (key(m) <= static_cast<std::int64_t>(s)) c = m + 1;
            else e = m;
        }
        return {a, c};
    }
    // The suffixes that start with p[0, m) (every suffix when m == 0): two binary searches inside the interval of the
    // first symbol, comparing from the second.
    Range find(const S* p, std::size_t m) const {
        if (m == 0) return all();
        if (static_cast<std::uint32_t>(p[0]) >= sigma_) return {};
        const Range r = first(p[0]);
        if (m == 1 || r.empty()) return r;
        Pos a = r.lo, b = r.hi;
        while (a < b) {  // the first suffix not below p
            const Pos x = a + (b - a) / 2;
            if (cmp(sa_[x] + 1, p + 1, m - 1) < 0) a = x + 1;
            else b = x;
        }
        Pos c = a, e = r.hi;
        while (c < e) {  // the first suffix that does not start with p
            const Pos x = c + (e - c) / 2;
            if (cmp(sa_[x] + 1, p + 1, m - 1) == 0) c = x + 1;
            else e = x;
        }
        return {a, c};
    }
    std::size_t count(const S* p, std::size_t m) const { return find(p, m).size(); }
    // The longest prefix of p[0, m) that occurs in the text: its common prefix with one of the two suffixes beside
    // where p would sort.
    std::size_t match(const S* p, std::size_t m) const {
        if (m == 0 || static_cast<std::uint32_t>(p[0]) >= sigma_) return 0;
        const Range r = first(p[0]);
        if (r.empty()) return 0;
        if (m == 1) return 1;
        Pos a = r.lo, b = r.hi;
        while (a < b) {
            const Pos x = a + (b - a) / 2;
            if (cmp(sa_[x] + 1, p + 1, m - 1) < 0) a = x + 1;
            else b = x;
        }
        std::size_t best = 0;
        if (a < r.hi) best = std::max(best, common(sa_[a] + 1, p + 1, m - 1));
        if (a > r.lo) best = std::max(best, common(sa_[a - 1] + 1, p + 1, m - 1));
        return 1 + best;
    }
    // Inside r (suffixes sharing their first d symbols): the ranks whose suffix goes on past d symbols. Only the first
    // rank can end there.
    Range longer(Range r, std::size_t d) const {
        if (!r.empty() && !has(r.lo, d)) ++r.lo;
        return r;
    }
    // The common prefix of the suffixes at ranks i - 1 and i, at most cap symbols.
    std::size_t lcp(std::size_t i, std::size_t cap) const {
        const std::size_t n = t_.size(), a = sa_[i - 1], b = sa_[i];
        const std::size_t lim = std::min(cap, std::min(n - a, n - b));
        std::size_t k = 0;
        while (k < lim && t_[a + k] == t_[b + k]) ++k;
        return k;
    }

 private:
    // The text from position i against q[0, m): < 0 when it sorts before every string that starts with q (a text that
    // ends first sorts first), 0 when it starts with q, > 0 after.
    int cmp(std::size_t i, const S* q, std::size_t m) const {
        const std::size_t n = t_.size();
        const std::size_t li = i < n ? std::min(m, n - i) : 0;
        if constexpr (std::is_same_v<S, std::uint8_t>) {
            const int r = li ? std::memcmp(t_.data() + i, q, li) : 0;
            if (r != 0) return r;
        } else {
            for (std::size_t k = 0; k < li; ++k)
                if (t_[i + k] != q[k]) return t_[i + k] < q[k] ? -1 : 1;
        }
        return li < m ? -1 : 0;
    }
    // The common prefix of the text from position i and q[0, m).
    std::size_t common(std::size_t i, const S* q, std::size_t m) const {
        const std::size_t n = t_.size();
        const std::size_t li = i < n ? std::min(m, n - i) : 0;
        std::size_t k = 0;
        while (k < li && t_[i + k] == q[k]) ++k;
        return k;
    }
    std::vector<S> t_;
    std::vector<Pos> sa_;
    std::vector<Pos> c_;  // c_[s]: the first rank of the suffixes that start with s (c_[sigma]: the length)
    std::uint32_t sigma_ = 0;
};

// Distinct symbols before positions, counted with an epoch stamp per symbol (no clearing between counts).
class Distinct {
 public:
    explicit Distinct(std::size_t sigma = 0) : stamp_(sigma, 0) {}
    void resize(std::size_t sigma) {
        if (stamp_.size() < sigma) stamp_.resize(sigma, 0);
    }
    void next() {
        if (++epoch_ == 0) {
            std::fill(stamp_.begin(), stamp_.end(), 0);
            epoch_ = 1;
        }
    }
    bool add(std::size_t s) {  // true when s is new since the last next()
        if (stamp_[s] == epoch_) return false;
        stamp_[s] = epoch_;
        return true;
    }

 private:
    std::vector<std::uint32_t> stamp_;
    std::uint32_t epoch_ = 1;
};

}  // namespace cidx

namespace kn {

// What a zero count of counts gives (the two estimators differ only there):
//   k5:  each n_j is taken as at least 1 and each discount as at least 0 (fuzzy_bake.py's KN, lm/word_kn.hpp);
//   bkn: D1 0.5, D2 1, D3+ 1.5 when n1, n2, n3 is zero, Y 0.5 when n1 + 2 n2 is zero, each discount clamped to
//        [0, j] (token_model.hpp's KneserNey, lm/byte_kn.hpp).
enum class Zero { k5, bkn };

// Chen and Goodman's modified Kneser-Ney discounts of one order from its counts of counts n[1..4]:
//   Y = n1 / (n1 + 2 n2), D1 = 1 - 2Y n2/n1, D2 = 2 - 3Y n3/n2, D3+ = 3 - 4Y n4/n3; D[0] is 0.
inline void discounts(const double n_in[5], double D[4], Zero z) {
    if (z == Zero::k5) {
        double nj[5];
        for (int j = 1; j <= 4; ++j) nj[j] = std::max(1.0, n_in[j]);
        const double Y = nj[1] / (nj[1] + 2 * nj[2]);
        D[0] = 0;
        D[1] = std::max(0.0, 1 - 2 * Y * nj[2] / nj[1]);
        D[2] = std::max(0.0, 2 - 3 * Y * nj[3] / nj[2]);
        D[3] = std::max(0.0, 3 - 4 * Y * nj[4] / nj[3]);
        return;
    }
    const double n1 = n_in[1], n2 = n_in[2], n3 = n_in[3], n4 = n_in[4];
    const double Y = (n1 + 2 * n2) > 0 ? n1 / (n1 + 2 * n2) : 0.5;
    D[1] = n1 > 0 ? 1 - 2 * Y * n2 / n1 : 0.5;
    D[2] = n2 > 0 ? 2 - 3 * Y * n3 / n2 : 1.0;
    D[3] = n3 > 0 ? 3 - 4 * Y * n4 / n3 : 1.5;
    for (int j = 1; j <= 3; ++j) D[j] = std::clamp(D[j], 0.0, static_cast<double>(j));
}

// One context: its entries' counts c[0, m), in ascending symbol order. n(c) = sum c; each entry's discount
// d = D[min(c, 3)] goes to gamma, its value max(c - d, 0) / n(c) goes to emit(j, value). With drop_singletons, an entry
// seen once is not emitted and its value goes to gamma too (byte KN's pruning), so the context still sums to one.
// Returns gamma = (sum of what went to it) / n(c); *tot gets n(c).
//   P(w | c) = value(w) + gamma P(w | c').
template <class Count, class Emit>
inline double context(const Count* c, std::size_t m, const double D[4], bool drop_singletons, Emit&& emit, double* tot = nullptr) {
    double nn = 0;
    for (std::size_t j = 0; j < m; ++j) nn += static_cast<double>(c[j]);
    double g = 0;
    for (std::size_t j = 0; j < m; ++j) {
        const double d = D[std::min<std::uint64_t>(static_cast<std::uint64_t>(c[j]), 3)];
        g += d;
        const double v = std::max(static_cast<double>(c[j]) - d, 0.0);
        if (drop_singletons && c[j] == 1) {
            g += v;
            continue;
        }
        emit(j, v / nn);
    }
    if (tot) *tot = nn;
    return g / nn;
}

}  // namespace kn

}  // namespace cypha::cyphalm
