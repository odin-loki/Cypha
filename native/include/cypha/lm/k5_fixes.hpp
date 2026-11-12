// docs/reports/K5_FIXES.md (F1, GENERATION_PLAN §13.76): fixes for where X1 found the writing breaks
// (docs/reports/WHERE_IT_BREAKS.md). Every flag is off by default; with all off, nothing here runs and K5 writes as
// before, byte for byte.
//
//   --k5-minp P  (MP) the min-p cut. In each draw, after the repeat ban and the copy cap, a candidate whose probability
//                under K5 (before temperature, the distribution X1's map reads) is below P times the most probable
//                candidate's is dropped before the nucleus. The most probable word left is never dropped. At
//                temperature T the cut is P^(1/T) times the top word's probability after temperature: the same words.
//   --k5-ff X    (FF) the gate's fluency floor. Among a sentence's drafts, those whose mean log2 P a word (the gate's
//                fluency) is more than X bits below the most fluent draft's are dropped; the gate then chooses among
//                the rest by its own terms, as before.
//   --k5-cj J    (CJ) the clause judge. For each word of a draft, L = the length in words of the longest span of the
//                text (prompt, text so far, the draft) ending at that word that occurs in K5's own id stream (its
//                count index over the 337 MB: the suffix array, count_index.hpp; capped at 16; never across <unk>).
//                A join is a word where that match restarts (L no longer than at the word before). A weak join is a
//                join carried by at most J words of context (L - 1 <= J): the span before it and the span after it
//                share at most J words of training evidence. A draft's weak joins per word is its join score. The
//                gate keeps the drafts whose score is at most the lower median of the drafts left (at least half,
//                ties kept), then chooses among them as before. With FF too, the FF floor goes first.
//
// The 12-word copy cap is untouched: no draft can complete 13 training words, so L stays at 12 or less inside K5's own
// sentences, and the judge cannot ask for longer copies.
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cypha::cyphalm::wkn::fix {

struct Params {
    double minp = 0;  // --k5-minp P (0: off)
    double ff = 0;    // --k5-ff X bits (0: off)
    int cj = -1;      // --k5-cj J (-1: off)
    bool gate_on() const { return ff > 0 || cj >= 0; }
    std::string describe() const {  // for the MODEL line; empty when off
        std::string s;
        char b[64];
        if (minp > 0) std::snprintf(b, sizeof b, "\tminp=%g", minp), s += b;
        if (ff > 0) std::snprintf(b, sizeof b, "\tff=%g", ff), s += b;
        if (cj >= 0) std::snprintf(b, sizeof b, "\tcj=%d", cj), s += b;
        return s;
    }
};

inline bool is_flag(const std::string& name) { return name == "--k5-minp" || name == "--k5-ff" || name == "--k5-cj"; }

inline bool parse(const std::string& name, const std::string& val, Params& p, std::string& err) {
    char* e = nullptr;
    const double x = std::strtod(val.c_str(), &e);
    if (val.empty() || *e) return err = name + ": bad value " + val, false;
    if (name == "--k5-minp") {
        if (!(x >= 0 && x < 1)) return err = name + ": P in [0, 1)", false;
        p.minp = x;
    } else if (name == "--k5-ff") {
        if (!(x >= 0 && x <= 100)) return err = name + ": X in [0, 100] bits", false;
        p.ff = x;
    } else if (name == "--k5-cj") {
        if (!(x >= -1 && x <= 15 && x == static_cast<int>(x))) return err = name + ": J in -1 (off) .. 15", false;
        p.cj = static_cast<int>(x);
    } else {
        return err = "unknown flag " + name, false;
    }
    return true;
}

inline constexpr std::size_t kJoinCap = 16;

// CJ: the weak joins per word of the last n ids of t[0, len) (the draft; the ids before it are the prompt and the text
// so far). Index: cidx::CountIndex<std::uint32_t> (count(p, m) > 0 when p[0, m) occurs). unk: the <unk> id.
template <class Index>
double weak_joins(const Index& idx, const std::uint32_t* t, std::size_t len, std::size_t n, int J, std::uint32_t unk) {
    if (n == 0 || n > len) return 0;
    // The longest span ending at i (at most bound words, never across <unk>), stepping down from bound: a span occurs
    // only if every shorter span ending at i does.
    const auto longest = [&](std::size_t i, std::size_t bound) -> std::size_t {
        if (t[i] == unk) return 0;
        std::size_t m = std::min({bound, kJoinCap, i + 1});
        for (std::size_t k = 1; k < m; ++k)
            if (t[i - k] == unk) {
                m = k;
                break;
            }
        while (m > 0 && idx.count(t + i + 1 - m, m) == 0) --m;
        return m;
    };
    const std::size_t a = len - n;
    std::size_t prev = a > 0 ? longest(a - 1, kJoinCap) : 0;
    std::size_t weak = 0;
    for (std::size_t i = a; i < len; ++i) {
        const std::size_t L = longest(i, prev + 1);
        if (L <= prev && (L == 0 || L - 1 <= static_cast<std::size_t>(J))) ++weak;
        prev = L;
    }
    return static_cast<double>(weak) / static_cast<double>(n);
}

// The drafts the gate may choose (1) after FF and CJ. flu: each draft's mean log2 P a word; wj: each draft's weak joins
// per word (read only with CJ). Never empty when there is a draft.
inline std::vector<char> allowed(const Params& p, const std::vector<double>& flu, const std::vector<double>& wj) {
    const std::size_t K = flu.size();
    std::vector<char> ok(K, 1);
    if (K == 0) return ok;
    if (p.ff > 0) {
        const double best = *std::max_element(flu.begin(), flu.end());
        for (std::size_t k = 0; k < K; ++k) ok[k] = flu[k] >= best - p.ff;
    }
    if (p.cj >= 0 && wj.size() == K) {
        std::vector<double> r;
        for (std::size_t k = 0; k < K; ++k)
            if (ok[k]) r.push_back(wj[k]);
        std::sort(r.begin(), r.end());
        const double med = r[(r.size() + 1) / 2 - 1];  // the lower median: at least half are kept
        for (std::size_t k = 0; k < K; ++k) ok[k] = ok[k] && wj[k] <= med;
    }
    return ok;
}

inline double mean(const std::vector<double>& v) {
    double s = 0;
    for (const double x : v) s += x;
    return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}

// Selftest: the parser, the floor, the median and the joins on a toy index.
template <class Index>
bool self_test(std::string& err) {
    Params p;
    if (p.gate_on() || !p.describe().empty() || !parse("--k5-minp", "0.1", p, err) || p.minp != 0.1 || !parse("--k5-ff", "1", p, err) ||
        !parse("--k5-cj", "2", p, err) || p.cj != 2 || parse("--k5-minp", "1.5", p, err) || parse("--k5-cj", "x", p, err) ||
        p.describe() != "\tminp=0.1\tff=1\tcj=2")
        return err = "fix: parse", false;
    Params f;
    f.ff = 1;
    auto ok = allowed(f, {-2.0, -3.5, -2.9, -1.5}, {});
    if (ok != std::vector<char>{1, 0, 0, 1}) return err = "fix: ff", false;
    Params c;
    c.cj = 2;
    ok = allowed(c, {0, 0, 0, 0}, {0.3, 0.1, 0.2, 0.1});
    if (ok != std::vector<char>{0, 1, 0, 1}) return err = "fix: cj median", false;
    f.cj = 2;
    ok = allowed(f, {-2.0, -3.5, -2.9, -1.5}, {0.0, 0.0, 0.0, 0.2});
    if (ok != std::vector<char>{1, 0, 0, 0}) return err = "fix: ff then cj", false;
    // Text 2 3 4 5 6 3 4 7 1: "3 4 5" and "3 4 7" occur, "3 5" does not.
    Index idx;
    idx.build({2, 3, 4, 5, 6, 3, 4, 7, 1}, 9);
    const std::uint32_t d1[] = {2, 3, 4, 5};  // one span: no join
    const std::uint32_t d2[] = {2, 3, 4, 7};  // "2 3 4 7" never, "3 4 7" does (L 3 after L 3): a join carried by 2 words
    const std::uint32_t d3[] = {6, 3, 5};     // "3 5" never: L 1 after L 2, a join carried by 0 words
    if (weak_joins(idx, d1, 4, 3, 2, 0) != 0 || weak_joins(idx, d2, 4, 1, 2, 0) != 1 || weak_joins(idx, d2, 4, 1, 1, 0) != 0 ||
        weak_joins(idx, d3, 3, 1, 0, 0) != 1)
        return err = "fix: joins", false;
    return true;
}

}  // namespace cypha::cyphalm::wkn::fix
