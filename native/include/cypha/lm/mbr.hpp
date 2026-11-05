// Minimum Bayes risk selection for K5 (docs/reports/K5_DISCOURSE.md, GENERATION_PLAN §13.32).
//
// Decision theory: among N drafts, keep the one with the least expected loss when the other drafts stand in for the
// distribution of texts the model would write. The utility is chrF (Popović, 2015): the F-score of character n-gram
// overlap, n = 1..6, whitespace removed, beta 2 (recall weighs twice). It uses no word vectors and no co-occurrence
// counts, only the drafts' own characters.
//
//   mbr_utility(h, others) = mean over r in others of chrF(h, r)
//   mbr_select(drafts)     = argmax_i mean_{j != i} chrF(d_i, d_j)   (ties: the lower index)
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm::mbr {

inline constexpr int kChrN = 6;
inline constexpr double kChrBeta = 2.0;

// A text's character n-gram counts, n = 1..kChrN, whitespace removed (sacreBLEU's chrF). Keys hash the n bytes and n.
struct NgramBag {
    std::unordered_map<std::uint64_t, std::uint32_t> c[kChrN];
    std::size_t total[kChrN] = {};
};

inline NgramBag bag(std::string_view s) {
    std::string t;
    t.reserve(s.size());
    for (const char ch : s)
        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') t.push_back(ch);
    NgramBag b;
    for (int n = 1; n <= kChrN; ++n) {
        if (t.size() < static_cast<std::size_t>(n)) break;
        auto& m = b.c[n - 1];
        m.reserve(t.size());
        for (std::size_t i = 0; i + static_cast<std::size_t>(n) <= t.size(); ++i) {
            std::uint64_t h = 0xcbf29ce484222325ull ^ static_cast<std::uint64_t>(n);
            for (int k = 0; k < n; ++k) h = (h ^ static_cast<unsigned char>(t[i + static_cast<std::size_t>(k)])) * 0x100000001b3ull;
            ++m[h];
        }
        b.total[n - 1] = t.size() + 1 - static_cast<std::size_t>(n);
    }
    return b;
}

// chrF of hypothesis h against reference r: the mean over n of precision and recall (orders with no n-grams on
// either side are left out of the means), then F_beta. In [0, 1]; 1 for identical non-empty texts.
inline double chrf(const NgramBag& h, const NgramBag& r, double beta = kChrBeta) {
    double sp = 0, sr = 0;
    int np = 0, nr = 0;
    for (int n = 0; n < kChrN; ++n) {
        if (h.total[n] == 0 && r.total[n] == 0) continue;
        std::size_t m = 0;
        const auto& small = h.c[n].size() <= r.c[n].size() ? h.c[n] : r.c[n];
        const auto& big = h.c[n].size() <= r.c[n].size() ? r.c[n] : h.c[n];
        for (const auto& [k, x] : small) {
            const auto it = big.find(k);
            if (it != big.end()) m += std::min(x, it->second);
        }
        if (h.total[n]) sp += static_cast<double>(m) / static_cast<double>(h.total[n]), ++np;
        if (r.total[n]) sr += static_cast<double>(m) / static_cast<double>(r.total[n]), ++nr;
        if (!h.total[n]) ++np;  // an empty side has precision (or recall) 0 at this order
        if (!r.total[n]) ++nr;
    }
    const double p = np ? sp / np : 0, rc = nr ? sr / nr : 0;
    if (p + rc <= 0) return 0;
    const double b2 = beta * beta;
    return (1 + b2) * p * rc / (b2 * p + rc);
}

inline double chrf(std::string_view h, std::string_view r, double beta = kChrBeta) { return chrf(bag(h), bag(r), beta); }

// Each draft's expected utility against the others: u_i = mean_{j != i} chrF(d_i, d_j). One draft: u = {0}.
inline std::vector<double> utilities(const std::vector<std::string>& drafts, double beta = kChrBeta) {
    const std::size_t n = drafts.size();
    std::vector<NgramBag> b;
    b.reserve(n);
    for (const auto& d : drafts) b.push_back(bag(d));
    std::vector<double> u(n, 0.0);
    if (n < 2) return u;
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0;
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) s += chrf(b[i], b[j], beta);
        u[i] = s / static_cast<double>(n - 1);
    }
    return u;
}

// The MBR choice: the index of the draft with the highest expected utility (ties: the lower index).
inline std::size_t select(const std::vector<std::string>& drafts, double beta = kChrBeta) {
    const auto u = utilities(drafts, beta);
    std::size_t best = 0;
    for (std::size_t i = 1; i < u.size(); ++i)
        if (u[i] > u[best]) best = i;
    return best;
}

// Selftest: identity, symmetry of the F1 form, bounds, and the choice of the central draft.
inline bool self_test(std::string& err) {
    if (std::abs(chrf("the cat sat", "the cat sat") - 1.0) > 1e-12) return err = "mbr: chrF(x, x) != 1", false;
    if (chrf("abc", "xyz") != 0.0) return err = "mbr: disjoint chrF != 0", false;
    const double a = chrf("the cat sat on the mat", "a cat sat on a hat", 1.0), b = chrf("a cat sat on a hat", "the cat sat on the mat", 1.0);
    if (std::abs(a - b) > 1e-12 || a <= 0 || a >= 1) return err = "mbr: chrF1 not symmetric or out of (0, 1)", false;
    const std::vector<std::string> d = {"zzzz qqqq", "the cat sat on the mat", "the cat sat on a mat", "a cat sat on the mat"};
    if (select(d) == 0) return err = "mbr: chose the outlier", false;
    return true;
}

}  // namespace cypha::cyphalm::mbr
