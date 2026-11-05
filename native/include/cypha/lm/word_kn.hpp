// K5 in CyphaLM: word-level interpolated Kneser-Ney of order 5 over clean prose, and the sampler that writes with it
// (docs/GENERATION_PLAN.md §13.11 "Generation: build K5's design into CyphaLM", §13.19). A port of the arm K5 of
// bench/fuzzy/fuzzy_bake.py (the reference): the tokeniser TOKRE, the vocabulary, the estimator KN and generate_one.
//
// Training text: one document per line (SCRIBE's cleaned prose, e.g. fuzzy_bake.py's train.txt). Each document is
// BOS followed by its tokens, as fuzzy_bake.py's prep() encodes it.
//
// Tokens (TOKRE = [A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+(?:[.,][0-9]+)*|\n|[^\sA-Za-z0-9] over code points, \s as Python's):
// words with inner apostrophes, numbers with inner . and ,, a newline, and any other single character that is not
// white space. Vocabulary: <unk> (0), <s> (1), then the tokens seen at least twice, most frequent first (ties by first
// occurrence), up to 32,000 entries in all. Anything else is <unk>.
//
// The estimate (Chen and Goodman's interpolated modified Kneser-Ney, as fuzzy_bake.py's KN builds it):
//   - order 5 counts every 5-gram whose target lies at position 4 or later of the id stream (documents are not cut);
//   - orders 4, 3, 2 count distinct left extensions of the order above's distinct entries (continuation counts);
//   - order 1 is p1(w) = (N1+(. w) + 0.01) / sum, over the whole vocabulary;
//   - per order, from its counts of counts n1..n4 (each at least 1): Y = n1 / (n1 + 2 n2), D1 = 1 - 2Y n2/n1,
//     D2 = 2 - 3Y n3/n2, D3+ = 3 - 4Y n4/n3, each at least 0;
//   - P_k(w | c) = gamma(c) P_{k-1}(w | c') + (n(c w) - D(n(c w))) / n(c), gamma(c) = sum_w D(n(c w)) / n(c);
//   - the walk goes up from order 2 and stops at the first context that was not seen. Every P sums to one.
//
// The sampler (generate_one), per step, at most 300 words:
//   1. the KN distribution over the last four ids, <unk> and <s> set to 0, renormalised (T != 1 raises it to 1/T first);
//   2. candidates: the 3,000 most probable words, cut at the first whose cumulative mass reaches 0.995;
//   3. bans: the word-level repeat ban (a word is banned when any of its bytes, written in turn, would follow an 11-byte
//      window with the byte that followed the same window earlier, prompt included) and the copy cap (a word may not
//      complete a run of cap + 1 tokens that occurs in the training ids; the prompt counts); if the repeat ban leaves
//      nothing, it is lifted; if the cap leaves nothing, the most probable word anywhere that the cap allows is taken;
//   4. nucleus 0.9 over the kept words, renormalised, and one draw;
//   5. the word's surface form (piece: no space after ( [ { $ or a newline or at the start, none before . , ; : ! ? ) ] } %
//      or an apostrophe), until the text holds at least 600 bytes; it is then cut to 600 bytes at a character boundary.
// The draws use std::mt19937_64 seeded with 20261101 + the prompt's index (fuzzy_bake.py: numpy's generator, same
// seed), so the texts are not fuzzy_bake.py's, but they are drawn from the same distributions. Words of equal
// probability are ordered by a hash salted per step, as numpy's argpartition and argsort leave them in no particular
// order. Ordering them by id (by frequency) let the most frequent words of a tied tail block into the nucleus, which
// the 0.9 cut splits at about half the steps, and cost about 0.03 SCRIBE on GEN-A dev (§13.19).
// Checked against fuzzy_bake.py on its own train.txt: the same vocabulary and id stream, the same entries per order,
// probabilities equal to about 13 digits, and its continuation bits per byte on GEN-A dev (1.549997).
//
// Words outside the vocabulary: in the prompt they are <unk> in the context, as in fuzzy_bake.py; <unk> is never
// written. Nothing spells new words (the byte model does not join in; §13.19).
//
// Rule 10 (PLAN): CyphaLM's own model. No neural network; it answers nothing.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cypha/lm/count_index.hpp"  // GENERATION_PLAN §13.40 M1: K5's counts and copy cap
#include "cypha/lm/entity_grid.hpp"  // K5_DISCOURSE.md
#include "cypha/lm/lda.hpp"          // K5_DISCOURSE.md
#include "cypha/lm/mbr.hpp"          // K5_DISCOURSE.md

namespace cypha::cyphalm::wkn {

inline constexpr std::uint32_t kUnk = 0, kBos = 1;
inline constexpr std::size_t kVmax = 32000;     // fuzzy_bake.py VMAX
inline constexpr std::uint64_t kSeed = 20261101;  // fuzzy_bake.py SEED
inline constexpr int kOrder = 5;      // K5's order (the default)
inline constexpr int kMaxOrder = 7;   // --k5-order N: up to 7 (context keys are 128 bits: six 16-bit ids)
using Key = unsigned __int128;        // a context: w-1 in bits 0-15, w-2 in bits 16-31, ...
inline std::size_t key_hash(Key k) {
    const auto lo = static_cast<std::uint64_t>(k), hi = static_cast<std::uint64_t>(k >> 64);
    std::uint64_t x = lo ^ (hi * 0x9E3779B97F4A7C15ull) ^ 0x2545F4914F6CDD1Dull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return static_cast<std::size_t>(x ^ (x >> 31));
}
struct KeyHasher {
    std::size_t operator()(Key k) const { return key_hash(k); }
};

// ---------------------------------------------------------------- tokens

// The code point at s[i] and its length in bytes; a byte that does not start a valid sequence is one character.
inline std::size_t utf8_at(std::string_view s, std::size_t i, char32_t& cp) {
    const auto c = static_cast<unsigned char>(s[i]);
    const std::size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) {
        cp = 0xFFFD;
        return 1;
    }
    cp = n == 1 ? c : n == 2 ? (c & 0x1Fu) : n == 3 ? (c & 0x0Fu) : (c & 0x07u);
    for (std::size_t k = 1; k < n; ++k) {
        const auto d = static_cast<unsigned char>(s[i + k]);
        if ((d & 0xC0) != 0x80) {
            cp = 0xFFFD;
            return 1;
        }
        cp = (cp << 6) | (d & 0x3Fu);
    }
    return n;
}

// Python's re \s on str (Py_UNICODE_ISSPACE): checked against every code point.
inline bool py_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
inline bool az(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
inline bool d09(char c) { return c >= '0' && c <= '9'; }

// TOKRE.findall: calls f(token) for each token, a view into s.
template <class F>
void tokenize(std::string_view s, F&& f) {
    const std::size_t n = s.size();
    for (std::size_t i = 0; i < n;) {
        const char c = s[i];
        std::size_t j = i;
        if (az(c)) {
            while (j < n && az(s[j])) ++j;
            while (j + 1 < n && s[j] == '\'' && az(s[j + 1])) {
                ++j;
                while (j < n && az(s[j])) ++j;
            }
        } else if (d09(c)) {
            while (j < n && d09(s[j])) ++j;
            while (j + 1 < n && (s[j] == '.' || s[j] == ',') && d09(s[j + 1])) {
                ++j;
                while (j < n && d09(s[j])) ++j;
            }
        } else {
            char32_t cp = 0;
            const std::size_t len = utf8_at(s, i, cp);
            if (cp != '\n' && py_space(cp)) {
                i += len;
                continue;
            }
            j = i + len;
        }
        f(s.substr(i, j - i));
        i = j;
    }
}
inline std::vector<std::string> tokens(std::string_view s) {
    std::vector<std::string> v;
    tokenize(s, [&](std::string_view t) { v.emplace_back(t); });
    return v;
}

// fuzzy_bake.py piece(tok, prev): the token's surface form after prev (has_prev false: the start of the text).
inline std::string piece(std::string_view tok, std::string_view prev, bool has_prev = true) {
    if (tok == "\n") return "\n";
    const bool after = !has_prev || prev == "\n" || prev == "(" || prev == "[" || prev == "{" || prev == "$";
    const bool before = tok == "'s" || (tok.size() == 1 && std::string_view(".,;:!?)]}%'").find(tok[0]) != std::string_view::npos);
    if (after || before) return std::string(tok);
    std::string o;
    o.reserve(tok.size() + 1);
    o += ' ';
    o += tok;
    return o;
}

// FNV-1a 64 over ids or bytes: fuzzy_bake.py's ngram_hashes / window_hashes (and cyphalm_trie's Spans32).
inline constexpr std::uint64_t kFnv0 = 0xcbf29ce484222325ull, kFnvP = 0x100000001b3ull;
inline std::uint64_t fnv_step(std::uint64_t h, std::uint64_t x) { return (h ^ x) * kFnvP; }
inline std::uint64_t mix64(std::uint64_t x) {  // splitmix64's finaliser
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// ---------------------------------------------------------------- the estimate

// §K5 upgrades, step 2b: dynamic counts. The prompt's n-grams and those of the text written so far, for one
// continuation only (the shared model never changes). In a context, a dynamic count d is discounted by Dd and the
// discounted mass goes to the lower order, so every distribution still sums to one:
//   P_k(w | c) = [(n(cw) - D) + L (d(cw) - Dd)] / (n(c) + L d(c)) + [sum D + L Dd types(c)] / (n(c) + L d(c)) P_{k-1}(w | c').
// Where the shared model has not seen a context and the overlay has, the overlay alone fills that order.
class Overlay {
 public:
    struct Ctx {
        std::vector<std::pair<std::uint16_t, double>> e;  // (word, count)
        double tot = 0;
    };
    double L = 1.0, Dd = 0.5;
    int order = kOrder;
    void push(std::uint32_t w) {
        hist_.push_back(w);
        const std::size_t t = hist_.size() - 1;
        for (int k = 2; k <= order; ++k) {
            const auto n = static_cast<std::size_t>(k - 1);
            if (t < n) break;
            Key key = 0;
            for (std::size_t i = 0; i < n; ++i) key |= Key{hist_[t - 1 - i]} << (16 * i);
            Ctx& c = m_[k][key];
            bool found = false;
            for (auto& [x, d] : c.e)
                if (x == w) d += 1, found = true;
            if (!found) c.e.emplace_back(static_cast<std::uint16_t>(w), 1.0);
            c.tot += 1;
        }
    }
    const Ctx* find(int k, Key key) const {
        const auto it = m_[k].find(key);
        return it == m_[k].end() ? nullptr : &it->second;
    }

 private:
    std::vector<std::uint32_t> hist_;
    std::unordered_map<Key, Ctx, KeyHasher> m_[kMaxOrder + 1];
};

// The counts come from one suffix array over the id stream (lm/count_index.hpp; GENERATION_PLAN §13.40, M1): the
// entries of order k are the k-grams that end at position order - 1 or later, as the old tables (which counted each
// order's distinct left extensions of the order above's entries) had them:
//   - top order: n(c w), the occurrences of c w;
//   - order k below it: the distinct words x before c w over its occurrences that start at order - k or later;
//   - order 1: the distinct words before w over its occurrences at order - 1 or later.
// Contexts seen at least `keep` times are kept as tables (as before: values, gamma, n(c)); the others are counted from
// the suffix array when asked, with the same arithmetic in the same order (kn::context), so every probability is the
// same bit for bit. keep 1 keeps every context (and the suffix array can then go).
class WordKN {
 public:
    using Index = cidx::CountIndex<std::uint32_t>;
    static constexpr std::uint32_t kKeep = 64;
    // ids: the training stream (documents as BOS + token ids), every id below V <= 65,536. order: 2 to kMaxOrder.
    // keep: contexts with at least this many occurrences are kept as tables. keep_index false: the index is dropped
    // after the build (only with keep 1, when nothing is counted later).
    void build(std::vector<std::uint32_t> ids, std::size_t V, int order = kOrder, std::uint32_t keep = kKeep, bool keep_index = true) {
        V_ = V;
        order_ = std::clamp(order, 2, kMaxOrder);
        keep_ = std::max<std::uint32_t>(keep, 1);
        idx_.build(std::move(ids), static_cast<std::uint32_t>(std::max<std::size_t>(V, 1)));
        const Index& x = idx_;
        const std::size_t N = x.size();
        // Common prefixes of neighbours in the suffix array, up to the top order's length.
        std::vector<std::uint8_t> lcp(N, 0);
        for (std::size_t i = 1; i < N; ++i) lcp[i] = static_cast<std::uint8_t>(x.lcp(i, static_cast<std::size_t>(order_)));
        cidx::Distinct seen(V);
        std::vector<std::uint32_t> cnt;
        for (int k = 2; k <= order_; ++k) {
            const auto L = static_cast<std::size_t>(k - 1);
            const std::size_t minpos = k == order_ ? 0 : static_cast<std::size_t>(order_ - k);
            Level& lv = lv_[k];
            lv = Level{};
            cnt.clear();
            double nj[5] = {0, 0, 0, 0, 0};
            for (std::size_t i = 0; i < N;) {
                std::size_t j = i + 1;
                while (j < N && lcp[j] >= L) ++j;
                if (!x.has(i, L - 1)) {  // shorter than a context
                    i = j;
                    continue;
                }
                const Index::Range r = x.longer({static_cast<Index::Pos>(i), static_cast<Index::Pos>(j)}, L);
                const bool kept = r.size() >= keep_;
                const std::size_t e0 = lv.ids.size();
                std::size_t m = 0;
                for (std::size_t a = r.lo; a < r.hi;) {
                    std::size_t b = a + 1;
                    while (b < r.hi && lcp[b] > L) ++b;
                    const std::uint32_t c = count(a, b, k == order_, minpos, seen);
                    if (c > 0) {
                        ++m;
                        if (c <= 4) nj[c] += 1;
                        if (kept) lv.ids.push_back(static_cast<std::uint16_t>(x.at(a, L))), cnt.push_back(c);
                    }
                    a = b;
                }
                if (m > 0) {
                    ++lv.contexts, lv.entries += m;
                    if (kept) {
                        Key key = 0;
                        for (std::size_t q = 0; q < L; ++q) key |= Key{x.at(i, L - 1 - q)} << (16 * q);
                        lv.keys.push_back(key);
                        lv.off.push_back(static_cast<std::uint32_t>(e0));
                    }
                }
                i = j;
            }
            lv.off.push_back(static_cast<std::uint32_t>(lv.ids.size()));
            kn::discounts(nj, lv.D, kn::Zero::k5);
            const std::size_t C = lv.keys.size();
            lv.gam.resize(C), lv.tot.resize(C), lv.disc.resize(lv.ids.size());
            for (std::size_t c = 0; c < C; ++c) {
                double* d = lv.disc.data() + lv.off[c];
                lv.gam[c] = kn::context(cnt.data() + lv.off[c], lv.off[c + 1] - lv.off[c], lv.D, false, [&](std::size_t e, double v) { d[e] = v; }, &lv.tot[c]);
            }
            lv.keys.shrink_to_fit(), lv.off.shrink_to_fit(), lv.ids.shrink_to_fit();
        }
        // Order 1: the distinct words before w over its occurrences at order - 1 or later.
        std::vector<double> cc(V, 0.0);
        for (std::uint32_t w = 0; w < V; ++w) {
            const Index::Range r = x.first(w);
            cc[w] = count(r.lo, r.hi, false, static_cast<std::size_t>(order_ - 1), seen);
        }
        double z = 0;
        for (double& v : cc) z += (v += 0.01);
        p1_.resize(V);
        for (std::size_t w = 0; w < V; ++w) p1_[w] = cc[w] / z;
        by_p1_.resize(V);
        for (std::uint32_t w = 0; w < V; ++w) by_p1_[w] = w;
        std::stable_sort(by_p1_.begin(), by_p1_.end(), [&](std::uint32_t a, std::uint32_t b) { return p1_[a] > p1_[b]; });
        if (!keep_index && keep_ <= 1) idx_.clear();
    }
    // The suffix array over the training ids (the copy cap and the prompt association read it too).
    const Index& index() const { return idx_; }
    // The vocabulary by its order-1 probability, most probable first: every word outside the stored contexts' entries
    // keeps this order in any distribution, which makes a tight threshold for the most probable words (K5::generate).
    // (With a class floor, base != p1, the threshold is not tight; K5::generate then sorts all words.)
    const std::vector<std::uint32_t>& by_p1() const { return by_p1_; }
    const std::vector<double>& p1() const { return p1_; }
    int order() const { return order_; }

    // P(. | ctx) into p[0, V): ctx[0, n) is the history, its last word last. ov: the continuation's dynamic counts;
    // base: the floor distribution in place of order 1 (§K5 upgrades step 3, class backoff). Without either, the walk
    // is exactly K5's.
    void dist(const std::uint32_t* ctx, std::size_t n, double* p, const Overlay* ov = nullptr, const double* base = nullptr) const {
        if (base) std::copy(base, base + V_, p);
        else std::copy(p1_.begin(), p1_.end(), p);
        bool main_ok = true;
        View v;
        for (int k = 2; k <= order_; ++k) {
            const std::size_t L = static_cast<std::size_t>(k - 1);
            if (n < L) break;
            Key key = 0;
            for (std::size_t i = 0; i < L; ++i) key |= Key{ctx[n - 1 - i]} << (16 * i);
            bool c = false;
            if (main_ok) {
                c = context(k, ctx, n, key, v);
                main_ok = c;
            }
            const Overlay::Ctx* oc = ov && k <= ov->order ? ov->find(k, key) : nullptr;
            if (!oc) {
                if (!c) break;
                const double g = v.gam;
                for (std::size_t w = 0; w < V_; ++w) p[w] *= g;
                for (std::size_t e = 0; e < v.m; ++e) p[v.ids[e]] += v.disc[e];
                continue;
            }
            const double T = c ? v.tot : 0.0, G = c ? v.gam * T : 0.0;
            const double den = T + ov->L * oc->tot;
            const double g = (G + ov->L * ov->Dd * static_cast<double>(oc->e.size())) / den;
            for (std::size_t w = 0; w < V_; ++w) p[w] *= g;
            if (c)
                for (std::size_t e = 0; e < v.m; ++e) p[v.ids[e]] += v.disc[e] * T / den;
            for (const auto& [w, d] : oc->e) p[w] += ov->L * (d - ov->Dd) / den;
        }
    }
    // docs/reports/K5_MCMC.md: P(w | ctx) alone, as dist would give it without an overlay; base_w is the floor's
    // probability of w (p1()[w], or the class floor's).
    double prob(const std::uint32_t* ctx, std::size_t n, std::uint32_t w, double base_w) const {
        double p = base_w;
        View v;
        for (int k = 2; k <= order_; ++k) {
            const std::size_t L = static_cast<std::size_t>(k - 1);
            if (n < L) break;
            Key key = 0;
            for (std::size_t i = 0; i < L; ++i) key |= Key{ctx[n - 1 - i]} << (16 * i);
            if (!context(k, ctx, n, key, v)) break;
            p *= v.gam;
            const std::uint16_t* f = std::lower_bound(v.ids, v.ids + v.m, static_cast<std::uint16_t>(w));
            if (f != v.ids + v.m && *f == w) p += v.disc[f - v.ids];
        }
        return p;
    }
    // docs/reports/OPT_NUCLEUS.md: dist's walk without an overlay. f(gamma, ids, disc, m) gets each context it applies,
    // order 2 up, in turn (the views last only until the next call). Returns the number of orders applied.
    template <class F>
    int walk(const std::uint32_t* ctx, std::size_t n, F&& f) const {
        View v;
        int found = 0;
        for (int k = 2; k <= order_; ++k) {
            const std::size_t L = static_cast<std::size_t>(k - 1);
            if (n < L) break;
            Key key = 0;
            for (std::size_t i = 0; i < L; ++i) key |= Key{ctx[n - 1 - i]} << (16 * i);
            if (!context(k, ctx, n, key, v)) break;
            f(v.gam, v.ids, v.disc, v.m);
            ++found;
        }
        return found;
    }
    std::size_t vocab_size() const { return V_; }
    std::size_t contexts(int k) const { return lv_[k].contexts; }
    std::size_t entries(int k) const { return k <= order_ ? lv_[k].entries : 0; }
    std::size_t kept_contexts(int k) const { return lv_[k].keys.size(); }
    const double* discounts(int k) const { return lv_[k].D; }
    std::size_t bytes() const {
        std::size_t b = p1_.size() * 8 + by_p1_.size() * 4 + idx_.bytes();
        for (int k = 2; k <= order_; ++k)
            b += lv_[k].keys.size() * 40 + lv_[k].off.size() * 4 + lv_[k].ids.size() * 10;
        return b;
    }

 private:
    struct Level {  // the kept contexts of one order
        std::vector<Key> keys;            // contexts, ascending: w-1 in bits 0-15, w-2 in bits 16-31, ...
        std::vector<std::uint32_t> off;   // entries of context c: [off[c], off[c + 1])
        std::vector<std::uint16_t> ids;   // the next words, ascending within a context
        std::vector<double> disc;         // (n(c w) - D) / n(c)
        std::vector<double> gam;          // sum D / n(c)
        std::vector<double> tot;          // n(c), for the dynamic counts
        double D[4] = {0, 0, 0, 0};
        std::size_t contexts = 0, entries = 0;  // every context and entry of the order, kept or not
    };
    struct View {  // one context's entries, gamma and n(c): in a Level, or counted into a thread's scratch
        const std::uint16_t* ids = nullptr;
        const double* disc = nullptr;
        std::size_t m = 0;
        double gam = 0, tot = 0;
    };
    // An entry's count over suffix-array ranks [a, b) (the occurrences of c w): the occurrences at the top order, else
    // the distinct words before those that start at minpos or later.
    std::uint32_t count(std::size_t a, std::size_t b, bool top, std::size_t minpos, cidx::Distinct& seen) const {
        if (top) return static_cast<std::uint32_t>(b - a);
        seen.next();
        const std::vector<std::uint32_t>& t = idx_.text();
        std::uint32_t c = 0;
        for (std::size_t q = a; q < b; ++q) {
            const std::size_t p = idx_.sa(q);
            if (p >= minpos && seen.add(t[p - 1])) ++c;
        }
        return c;
    }
    // The context of order k made of the last k - 1 ids of ctx[0, n) (key: them packed): kept, or counted from the
    // suffix array with the kept contexts' arithmetic. False when the order has no such context.
    bool context(int k, const std::uint32_t* ctx, std::size_t n, Key key, View& v) const {
        const Level& lv = lv_[k];
        const auto it = std::lower_bound(lv.keys.begin(), lv.keys.end(), key);
        if (it != lv.keys.end() && *it == key) {
            const auto c = static_cast<std::size_t>(it - lv.keys.begin());
            v.ids = lv.ids.data() + lv.off[c];
            v.disc = lv.disc.data() + lv.off[c];
            v.m = lv.off[c + 1] - lv.off[c];
            v.gam = lv.gam[c], v.tot = lv.tot[c];
            return true;
        }
        if (keep_ <= 1 || !idx_.built()) return false;
        const auto L = static_cast<std::size_t>(k - 1);
        const Index::Range r = idx_.longer(idx_.find(ctx + n - L, L), L);
        if (r.empty()) return false;
        struct Scratch {
            std::vector<std::uint16_t> ids;
            std::vector<std::uint32_t> cnt;
            std::vector<double> disc;
            cidx::Distinct seen;
        };
        thread_local Scratch s;
        s.ids.clear(), s.cnt.clear();
        s.seen.resize(V_);
        const std::size_t minpos = k == order_ ? 0 : static_cast<std::size_t>(order_ - k);
        for (std::size_t a = r.lo; a < r.hi;) {
            const std::uint32_t w = idx_.at(a, L);
            std::size_t b = a + 1;
            while (b < r.hi && idx_.at(b, L) == w) ++b;
            const std::uint32_t c = count(a, b, k == order_, minpos, s.seen);
            if (c > 0) s.ids.push_back(static_cast<std::uint16_t>(w)), s.cnt.push_back(c);
            a = b;
        }
        if (s.ids.empty()) return false;
        s.disc.resize(s.ids.size());
        v.gam = kn::context(s.cnt.data(), s.cnt.size(), lv.D, false, [&](std::size_t e, double x) { s.disc[e] = x; }, &v.tot);
        v.ids = s.ids.data(), v.disc = s.disc.data(), v.m = s.ids.size();
        return true;
    }
    std::size_t V_ = 0;
    int order_ = kOrder;
    std::uint32_t keep_ = kKeep;
    Index idx_;
    std::vector<double> p1_;
    std::vector<std::uint32_t> by_p1_;
    Level lv_[kMaxOrder + 1];
};

// §K5 upgrades, step 3: Martin, Liermann and Ney's exchange algorithm (bench/fuzzy/so_bench.py exchange): each word
// seen at least min_count times, most frequent first, moves to the class that most raises the class-bigram
// likelihood. Ids are by frequency (K5's vocabulary), so the start deals them to the K classes in turn. <unk> and <s>
// keep classes K and K + 1. Returns each word's class; loglik gets the likelihood after each pass.
inline std::vector<std::uint32_t> exchange_classes(const std::vector<std::uint32_t>& ids, std::size_t V, std::size_t K,
                                                   int passes, std::size_t min_count, std::vector<double>* loglik = nullptr) {
    const std::size_t KK = K + 2;
    std::vector<std::uint32_t> cls(V);
    for (std::size_t w = 0; w < V; ++w) cls[w] = w == kUnk ? static_cast<std::uint32_t>(K) : w == kBos ? static_cast<std::uint32_t>(K + 1) : static_cast<std::uint32_t>((w - 2) % K);
    std::vector<std::uint64_t> bg;
    bg.reserve(ids.size());
    for (std::size_t t = 1; t < ids.size(); ++t) bg.push_back(std::uint64_t{ids[t - 1]} * V + ids[t]);
    std::sort(bg.begin(), bg.end());
    struct E {
        std::uint32_t x;
        double c;
    };
    std::vector<std::vector<E>> out(V), in(V);
    std::vector<double> freq(V, 0.0);
    for (const std::uint32_t w : ids) freq[w] += 1;
    for (std::size_t a = 0; a < bg.size();) {
        std::size_t b = a;
        while (b < bg.size() && bg[b] == bg[a]) ++b;
        const auto x = static_cast<std::uint32_t>(bg[a] / V), y = static_cast<std::uint32_t>(bg[a] % V);
        out[x].push_back({y, static_cast<double>(b - a)});
        in[y].push_back({x, static_cast<double>(b - a)});
        a = b;
    }
    std::vector<double> M(KK * KK, 0.0), Lr(KK, 0.0), Rc(KK, 0.0);
    for (std::size_t x = 0; x < V; ++x)
        for (const E& e : out[x]) M[cls[x] * KK + cls[e.x]] += e.c, Lr[cls[x]] += e.c, Rc[cls[e.x]] += e.c;
    const auto xl = [](double v) { return v > 0 ? v * std::log(v) : 0.0; };
    const auto ll = [&] {
        double s = 0;
        for (const double v : M) s += xl(v);
        for (std::size_t k = 0; k < KK; ++k) s -= xl(Lr[k]) + xl(Rc[k]);
        return s;
    };
    if (loglik) loglik->assign(1, ll());
    std::vector<std::uint32_t> words;
    for (std::uint32_t w = 2; w < V; ++w)
        if (freq[w] >= static_cast<double>(min_count)) words.push_back(w);  // ids are already most frequent first
    std::vector<double> rw(KK, 0.0), cw(KK, 0.0), gain(K, 0.0);
    std::vector<std::uint32_t> nzr, nzc;
    for (int ps = 0; ps < passes; ++ps) {
        for (const std::uint32_t w : words) {
            nzr.clear(), nzc.clear();
            double selfc = 0, nl = 0, nr = 0;
            for (const E& e : out[w]) {
                nl += e.c;
                if (e.x == w) {
                    selfc += e.c;
                    continue;
                }
                if (rw[cls[e.x]] == 0) nzr.push_back(cls[e.x]);
                rw[cls[e.x]] += e.c;
            }
            for (const E& e : in[w]) {
                nr += e.c;
                if (e.x == w) continue;
                if (cw[cls[e.x]] == 0) nzc.push_back(cls[e.x]);
                cw[cls[e.x]] += e.c;
            }
            const std::uint32_t a0 = cls[w];
            for (const std::uint32_t j : nzr) M[a0 * KK + j] -= rw[j];
            for (const std::uint32_t i : nzc) M[i * KK + a0] -= cw[i];
            M[a0 * KK + a0] -= selfc;
            Lr[a0] -= nl, Rc[a0] -= nr;
            std::size_t best = a0;
            double bg2 = -1e300;
            for (std::size_t k = 0; k < K; ++k) {
                double g = 0;
                for (const std::uint32_t j : nzr) g += xl(M[k * KK + j] + rw[j]) - xl(M[k * KK + j]);
                for (const std::uint32_t i : nzc) g += xl(M[i * KK + k] + cw[i]) - xl(M[i * KK + k]);
                g -= (xl(Lr[k] + nl) - xl(Lr[k])) + (xl(Rc[k] + nr) - xl(Rc[k]));
                if (g > bg2) bg2 = g, best = k;
            }
            for (const std::uint32_t j : nzr) M[best * KK + j] += rw[j];
            for (const std::uint32_t i : nzc) M[i * KK + best] += cw[i];
            M[best * KK + best] += selfc;
            Lr[best] += nl, Rc[best] += nr;
            cls[w] = static_cast<std::uint32_t>(best);
            for (const std::uint32_t j : nzr) rw[j] = 0;
            for (const std::uint32_t i : nzc) cw[i] = 0;
        }
        if (loglik) loglik->push_back(ll());
    }
    return cls;
}

// The class model: P(w | h) = P(c(w) | c(h-2), c(h-1)) P(w | c(w)), Kneser-Ney of order 3 over class ids (so_bench.py
// ClassTri), words within a class by count.
class ClassModel {
 public:
    void build(const std::vector<std::uint32_t>& ids, std::size_t V, std::size_t K, int passes = 6) {
        V_ = V;
        cls_ = exchange_classes(ids, V, K, passes, 3, &loglik_);
        std::vector<std::uint32_t> cids(ids.size());
        for (std::size_t t = 0; t < ids.size(); ++t) cids[t] = cls_[ids[t]];
        kn_.build(std::move(cids), K + 2, 3, 1, false);  // every class context kept; the suffix array goes after the build
        std::vector<double> cnt(V, 0.01), ctot(K + 2, 0.0);
        for (const std::uint32_t w : ids) cnt[w] += 1;
        for (std::size_t w = 0; w < V; ++w) ctot[cls_[w]] += cnt[w];
        pwc_.resize(V);
        for (std::size_t w = 0; w < V; ++w) pwc_[w] = cnt[w] / ctot[cls_[w]];
        K_ = K;
        cmass_.assign(K + 2, 0.0);  // K5_MCMC.md: each class's word mass, for prob
        for (std::size_t w = 0; w < V; ++w) cmass_[cls_[w]] += pwc_[w];
        static std::atomic<std::uint64_t> builds{0};  // OPT_NUCLEUS.md
        uid_ = ++builds;
    }
    // K5_MCMC.md: dist's p[w] alone.
    double prob(const std::uint32_t* ctx, std::size_t n, std::uint32_t w) const {
        std::uint32_t cc[2];
        const std::size_t m = std::min<std::size_t>(n, 2);
        for (std::size_t i = 0; i < m; ++i) cc[i] = cls_[ctx[n - m + i]];
        double pc[4096 + 2];
        kn_.dist(cc, m, pc);
        double z = 0;
        for (std::size_t c = 0; c < K_ + 2; ++c) z += pc[c] * cmass_[c];
        return pc[cls_[w]] * pwc_[w] / z;
    }
    bool on() const { return V_ > 0; }
    void dist(const std::uint32_t* ctx, std::size_t n, double* p) const {
        std::uint32_t cc[2];
        const std::size_t m = std::min<std::size_t>(n, 2);
        for (std::size_t i = 0; i < m; ++i) cc[i] = cls_[ctx[n - m + i]];
        std::vector<double> pc(K_ + 2);
        kn_.dist(cc, m, pc.data());
        double z = 0;
        for (std::size_t w = 0; w < V_; ++w) z += (p[w] = pc[cls_[w]] * pwc_[w]);
        for (std::size_t w = 0; w < V_; ++w) p[w] /= z;
    }
    const std::vector<double>& loglik() const { return loglik_; }
    const std::vector<std::uint32_t>& classes() const { return cls_; }
    // docs/reports/OPT_NUCLEUS.md: dist's parts. pc gets the class distribution after ctx (K + 2 entries); the return
    // is dist's normaliser z = sum over w, in id order, of pc[c(w)] P(w | c(w)), so dist's p[w] is
    // (pc[c(w)] * pwc()[w]) / z. pc and z depend only on the last two classes, so each thread keeps the z it has summed.
    double floor_parts(const std::uint32_t* ctx, std::size_t n, double* pc) const {
        std::uint32_t cc[2] = {0, 0};
        const std::size_t m = std::min<std::size_t>(n, 2);
        for (std::size_t i = 0; i < m; ++i) cc[i] = cls_[ctx[n - m + i]];
        kn_.dist(cc, m, pc);
        struct Cache {
            std::uint64_t uid = 0;
            std::unordered_map<std::uint64_t, double> z;
        };
        thread_local Cache cache;
        if (cache.uid != uid_) cache.uid = uid_, cache.z.clear();
        const std::uint64_t key = (std::uint64_t{m} << 48) | (std::uint64_t{cc[0]} << 24) | cc[1];
        const auto [it, fresh] = cache.z.try_emplace(key, 0.0);
        if (fresh) {
            double z = 0;
            for (std::size_t w = 0; w < V_; ++w) z += pc[cls_[w]] * pwc_[w];
            it->second = z;
        }
        return it->second;
    }
    const std::vector<double>& pwc() const { return pwc_; }
    std::size_t num_classes() const { return K_ + 2; }

 private:
    std::size_t V_ = 0, K_ = 0;
    std::vector<std::uint32_t> cls_;
    std::vector<double> pwc_, loglik_, cmass_;
    WordKN kn_;
    std::uint64_t uid_ = 0;  // OPT_NUCLEUS.md: this build's number (floor_parts' cache)
};

// ---------------------------------------------------------------- bans

// fuzzy_bake.py RepeatBan: the byte sampler's 11-byte rule lifted to words.
class RepeatBan {
 public:
    static constexpr std::size_t W = 11;
    RepeatBan() = default;
    explicit RepeatBan(std::string_view text) { add(text); }
    void add(std::string_view b) {
        const std::size_t old = text_.size();
        text_.append(b);
        for (std::size_t s = old > W ? old - W : 0; s + W < text_.size(); ++s) {
            const Key k = key(text_.data() + s);
            const std::size_t h = KeyHash{}(k);
            seen_[(h >> 6) & (kSeen - 1)] |= std::uint64_t{1} << (h & 63);
            nx_[k].set(static_cast<unsigned char>(text_[s + W]));
        }
    }
    // Would writing pb repeat an earlier (11 bytes, next byte) pair, earlier in the text or inside pb itself?
    bool banned(std::string_view pb) const {
        const std::size_t h = std::min(text_.size(), W);
        buf_.assign(text_, text_.size() - h, h);
        buf_.append(pb);
        local_.clear();
        for (std::size_t i = h; i < buf_.size(); ++i) {
            if (i < W) continue;
            const Key k = key(buf_.data() + i - W);
            const auto c = static_cast<unsigned char>(buf_[i]);
            const std::size_t hk = KeyHash{}(k);
            if ((seen_[(hk >> 6) & (kSeen - 1)] >> (hk & 63)) & 1) {  // a window of the text, perhaps
                const auto it = nx_.find(k);
                if (it != nx_.end() && it->second.has(c)) return true;
            }
            for (const auto& [lk, lc] : local_)
                if (lc == c && lk == k) return true;
            local_.emplace_back(k, c);
        }
        return false;
    }
    // docs/reports/OPT_NUCLEUS.md: banned(pb) for many pieces after the same text. prepare reads the text once;
    // banned(b, pb) then gives banned(pb)'s answer. The checks are the same three:
    //   - a 12-byte window that ends inside pb at pb[j], j < 11, is the text's last 11 - j bytes and pb[0, j]. It is a
    //     window of the text when the text holds those 11 - j bytes ending at some e, followed by pb[0, j]. With mu(e)
    //     the longest common suffix (at most 11) of text[0, e) and the text, the least such j at e is 11 - mu(e), and
    //     a larger j needs a longer match. So pb is banned there exactly when it starts with text[e, e + 12 - mu(e))
    //     for some e (e + 12 - mu(e) within the text): the forbidden prefixes, kept by length;
    //   - windows wholly inside pb (j >= 11), looked up as banned() looks them up;
    //   - a window that repeats inside the new bytes (banned()'s local list), compared byte by byte. Two equal windows
    //     ending at pb[j1] and pb[j2], d = j2 - j1, need tail[x] = tail[x + d] for x in [j1, 11 - d): so j1 >= A(d), the
    //     start of the tail's longest d-periodic end, and j2 < |pb|. A piece shorter than min over d of A(d) + d + 1
    //     (12 for a tail with no period) has no such pair.
    // A text shorter than 11 bytes goes to banned(pb).
    struct Batch {
        bool on = false;
        std::uint32_t lens = 0;                       // bit l: some forbidden prefix has l bytes
        std::vector<unsigned __int128> pre[W + 1];    // the forbidden prefixes of l bytes (byte i in bits 8i), sorted
        char tail[W] = {};                            // the text's last 11 bytes
        std::size_t lmin = 0;                         // the shortest forbidden prefix (W + 1: none)
        std::size_t local = 0;                        // the shortest piece that can repeat a window inside itself
        // A filter: f2[h(first two bytes)] has bit l when some forbidden prefix of l >= 2 bytes starts with two bytes
        // of that hash; one: the forbidden prefixes of one byte. Only the lengths it names are searched.
        std::vector<std::uint16_t> f2 = std::vector<std::uint16_t>(kF2, 0);
        std::vector<std::uint16_t> f2set;             // the entries to clear at the next prepare
        std::uint64_t one[4] = {0, 0, 0, 0};
        static constexpr std::size_t kF2 = 4096;
        static std::size_t h2(unsigned char a, unsigned char c) {
            const std::size_t v = a | std::size_t{c} << 8;
            return (v ^ (v >> 12) ^ (v >> 5)) & (kF2 - 1);
        }
    };
    void prepare(Batch& b) const {
        const std::size_t n = text_.size();
        b.on = n >= W;
        b.lens = 0;
        for (auto& v : b.pre) v.clear();
        for (const std::uint16_t i : b.f2set) b.f2[i] = 0;
        b.f2set.clear();
        std::fill(b.one, b.one + 4, 0);
        if (!b.on) return;
        const char* t = text_.data();
        std::memcpy(b.tail, t + n - W, W);
        b.local = W + 1;
        for (std::size_t d = 1; d < W; ++d) {
            std::size_t a = W - d;  // the d-periodic end of the tail starts at a
            while (a > 0 && b.tail[a - 1] == b.tail[a - 1 + d]) --a;
            b.local = std::min(b.local, a + d + 1);
        }
        for (std::size_t e = 1; e < n; ++e) {
            std::size_t mu = 0;
            while (mu < W && mu < e && t[e - 1 - mu] == t[n - 1 - mu]) ++mu;
            if (mu == 0) continue;
            const std::size_t l = W + 1 - mu;
            if (e + l > n) continue;
            unsigned __int128 k = 0;
            for (std::size_t i = 0; i < l; ++i) k |= static_cast<unsigned __int128>(static_cast<unsigned char>(t[e + i])) << (8 * i);
            b.pre[l].push_back(k);
            b.lens |= 1u << l;
            const auto c0 = static_cast<unsigned char>(t[e]);
            if (l == 1) {
                b.one[c0 >> 6] |= std::uint64_t{1} << (c0 & 63);
            } else {
                const std::size_t h = Batch::h2(c0, static_cast<unsigned char>(t[e + 1]));
                if (b.f2[h] == 0) b.f2set.push_back(static_cast<std::uint16_t>(h));
                b.f2[h] = static_cast<std::uint16_t>(b.f2[h] | 1u << l);
            }
        }
        for (auto& v : b.pre) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
        b.lmin = W + 1;
        for (std::size_t l = 1; l <= W; ++l)
            if ((b.lens >> l) & 1) {
                b.lmin = l;
                break;
            }
    }
    // False when no piece of L bytes starting c0 c1 can be banned after b's text (banned(b, pb) is then false): the
    // checks banned(b, pb) makes before it reads more of the piece.
    static bool may_ban(const Batch& b, std::size_t L, unsigned char c0, unsigned char c1) {  // without branches
        const bool other = !b.on | (L >= b.local) | (L > W);
        const bool one = (b.one[c0 >> 6] >> (c0 & 63)) & 1;
        const bool two = (L >= 2) & ((b.f2[Batch::h2(c0, c1)] & ((2u << std::min(L, W)) - 1)) != 0);
        return other | ((L >= b.lmin) & (one | two));
    }
    bool banned(const Batch& b, std::string_view pb) const {
        if (!b.on) return banned(pb);
        const std::size_t L = pb.size();
        if (L < b.lmin && L < b.local && L <= W) return false;  // no forbidden prefix fits, no window inside pb, no repeat
        if (L >= b.lmin) {
            const auto c0 = static_cast<unsigned char>(pb[0]);
            if ((b.one[c0 >> 6] >> (c0 & 63)) & 1) return true;  // a forbidden prefix of one byte
            if (L >= 2) {
                const std::size_t lmax = std::min(L, W);
                std::uint32_t mask = b.f2[Batch::h2(c0, static_cast<unsigned char>(pb[1]))] & ((2u << lmax) - 1);
                if (mask) {
                    unsigned __int128 k = 0;
                    for (std::size_t l = 1; mask; ++l) {
                        k |= static_cast<unsigned __int128>(static_cast<unsigned char>(pb[l - 1])) << (8 * (l - 1));
                        if ((mask >> l) & 1) {
                            if (std::binary_search(b.pre[l].begin(), b.pre[l].end(), k)) return true;
                            mask &= ~(1u << l);
                        }
                    }
                }
            }
        }
        for (std::size_t j = W; j < L; ++j) {
            const Key kk = key(pb.data() + j - W);
            const auto c = static_cast<unsigned char>(pb[j]);
            const std::size_t hk = KeyHash{}(kk);
            if ((seen_[(hk >> 6) & (kSeen - 1)] >> (hk & 63)) & 1) {
                const auto it = nx_.find(kk);
                if (it != nx_.end() && it->second.has(c)) return true;
            }
        }
        if (L < b.local) return false;
        // The window ending at pb[j] is (tail + pb)[j, j + 12).
        const auto at = [&](std::size_t i) { return i < W ? b.tail[i] : pb[i - W]; };
        for (std::size_t j2 = 1; j2 < L; ++j2)
            for (std::size_t j1 = 0; j1 < j2; ++j1) {
                if (pb[j1] != pb[j2]) continue;
                std::size_t q = 0;
                while (q < W && at(j1 + q) == at(j2 + q)) ++q;
                if (q == W) return true;
            }
        return false;
    }

 private:
    struct Key {
        std::uint64_t a = 0;
        std::uint32_t b = 0;
        bool operator==(const Key& o) const { return a == o.a && b == o.b; }
    };
    struct KeyHash {
        std::size_t operator()(const Key& k) const {
            std::uint64_t x = k.a ^ (std::uint64_t{k.b} * 0x9E3779B97F4A7C15ull);
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
            return static_cast<std::size_t>(x ^ (x >> 31));
        }
    };
    struct Next {
        std::uint64_t m[4] = {0, 0, 0, 0};
        void set(unsigned c) { m[c >> 6] |= std::uint64_t{1} << (c & 63); }
        bool has(unsigned c) const { return (m[c >> 6] >> (c & 63)) & 1; }
    };
    static Key key(const char* p) {
        Key k;
        std::memcpy(&k.a, p, 8);
        std::memcpy(&k.b, p + 8, 3);
        return k;
    }
    static constexpr std::size_t kSeen = 1024;  // a 65,536-bit filter of the windows in nx_
    std::string text_;
    std::unordered_map<Key, Next, KeyHash> nx_;
    std::vector<std::uint64_t> seen_ = std::vector<std::uint64_t>(kSeen, 0);
    mutable std::string buf_;
    mutable std::vector<std::pair<Key, unsigned char>> local_;
};

// The copy cap (Odin: at most 12 consecutive words copied): the n-token windows of the training ids, read from K5's
// suffix array (GENERATION_PLAN §13.40 M1; before, a sorted set of the windows' 64-bit hashes). A view: K5::cap_set()
// makes one; the index and the hash store belong to the K5 model.
//   prefix(ctx, r) / completes(r, w): does ctx's last n - 1 ids followed by w occur (exact).
//   prefix(ctx, h) / has(h): the old hash calls, for code that hashes windows itself (lm/planner/writer.hpp,
//   lm/k5_mcmc.hpp); their windows' hashes are made on first use, once per model, exactly as before.
class NgramSet {
 public:
    using Index = cidx::CountIndex<std::uint32_t>;
    struct Hashes {  // the windows' hashes and their first n - 1 ids' hashes, sorted, made on first use
        std::once_flag once;
        std::vector<std::uint64_t> h, hp;
    };
    NgramSet() = default;
    NgramSet(const Index* idx, int n, Hashes* hs) : idx_(idx), n_(n), hs_(hs) {}
    int n() const { return n_; }
    bool on() const { return n_ > 0 && idx_ && idx_->size() >= static_cast<std::size_t>(n_); }
    // The suffixes that start with ctx's last n - 1 ids and go on; false when ctx is shorter or none does (then no
    // candidate can complete a window).
    bool prefix(const std::vector<std::uint32_t>& ctx, Index::Range& r) const {
        const auto m = static_cast<std::size_t>(n_ - 1);
        if (!on() || ctx.size() < m) return false;
        r = idx_->longer(idx_->find(ctx.data() + ctx.size() - m, m), m);
        return !r.empty();
    }
    // Would w complete a window after the prefix r?
    bool completes(Index::Range r, std::uint32_t w) const { return !idx_->narrow(r, static_cast<std::size_t>(n_ - 1), w).empty(); }
    // Does ids hold an n-token window of the training ids?
    bool any(const std::vector<std::uint32_t>& ids) const {
        if (!on()) return false;
        for (std::size_t s = 0; s + static_cast<std::size_t>(n_) <= ids.size(); ++s)
            if (idx_->count(ids.data() + s, static_cast<std::size_t>(n_)) > 0) return true;
        return false;
    }
    // The distinct windows.
    std::size_t size() const {
        if (!on()) return 0;
        const auto n = static_cast<std::size_t>(n_);
        std::size_t d = 0;
        for (std::size_t i = 0; i < idx_->size(); ++i) d += idx_->has(i, n - 1) && (i == 0 || idx_->lcp(i, n) < n);
        return d;
    }
    // The hash calls (see above).
    bool has(std::uint64_t h) const {
        if (!on()) return false;
        const Hashes& s = hashes();
        return std::binary_search(s.h.begin(), s.h.end(), h);
    }
    bool prefix(const std::vector<std::uint32_t>& ctx, std::uint64_t& h) const {
        const auto m = static_cast<std::size_t>(n_ - 1);
        if (!on() || ctx.size() < m) return false;
        h = kFnv0;
        for (std::size_t j = ctx.size() - m; j < ctx.size(); ++j) h = fnv_step(h, ctx[j]);
        const Hashes& s = hashes();
        return std::binary_search(s.hp.begin(), s.hp.end(), h);
    }

 private:
    const Hashes& hashes() const {
        std::call_once(hs_->once, [&] {
            const std::vector<std::uint32_t>& ids = idx_->text();
            const auto n = static_cast<std::size_t>(n_);
            if (ids.size() < n) return;
            hs_->h.reserve(ids.size() - n + 1), hs_->hp.reserve(ids.size() - n + 1);
            for (std::size_t s = 0; s + n <= ids.size(); ++s) {
                std::uint64_t h = kFnv0;
                for (std::size_t j = 0; j + 1 < n; ++j) h = fnv_step(h, ids[s + j]);
                hs_->hp.push_back(h);
                hs_->h.push_back(fnv_step(h, ids[s + n - 1]));
            }
            for (auto* v : {&hs_->h, &hs_->hp}) {
                std::sort(v->begin(), v->end());
                v->erase(std::unique(v->begin(), v->end()), v->end());
            }
        });
        return *hs_;
    }
    const Index* idx_ = nullptr;
    int n_ = 0;
    Hashes* hs_ = nullptr;
};

// ---------------------------------------------------------------- the model

// §13.24: a byte model that reranks K5's nucleus. begin reads the prompt; score gives log2 of the probability of a
// word's written bytes (its piece, spacing included) after the text so far, and leaves the state as it was; advance reads
// the piece that was written. One per continuation (per thread).
struct ByteScorer {
    virtual ~ByteScorer() = default;
    virtual void begin(std::string_view prompt) = 0;
    virtual double score(std::string_view piece) = 0;
    virtual void advance(std::string_view piece) = 0;
    // §13.24 pmi: log2 of the piece's probability with no context (the model's own prior on it), so the rerank can keep
    // only what the context adds. A byte model's prior favours short, common words, which K5 already weighs.
    virtual double prior(std::string_view) { return 0; }
};

class K5;
struct Options;
struct GenStats;
// docs/reports/K5_STRUCTURE.md (GENERATION_PLAN §13.33): a writer that replaces generate's sentence choice (the structure
// experts and their meta-mixer, cypha/lm/structure/writer.hpp). Shared by every thread, so generate is const.
struct StructureWriter {
    virtual ~StructureWriter() = default;
    virtual std::string generate(const K5& m, std::string_view prompt, std::uint64_t seed, const Options& o, GenStats* st) const = 0;
};

struct Options {
    // docs/reports/K5_MCMC.md: rewrites each continuation after generate (k5_mcmc.hpp's paragraph sampler; seed is the
    // continuation's), and post_desc joins the MODEL line. Empty: K5's own text, unchanged.
    std::function<std::string(const K5&, std::string_view prompt, std::string gen, std::uint64_t seed, const Options&)> post;
    std::string post_desc;
    // §13.24: the byte scorer (made once per continuation) and how it reranks the words inside the nucleus: 1 a product
    // of experts, p_k5^(1 - w) * p_bytes^w; 2 linear, (1 - w) p_k5 + w p_bytes, each normalised over the nucleus; 3 pmi,
    // p_k5 * (p_bytes / p_bytes_prior)^w (DExperts' form: the byte model adds only what the context tells it). K5
    // still chooses the candidates and the nucleus (the plausibility gate of §13.16).
    std::function<std::unique_ptr<ByteScorer>()> scorer;
    int scorer_mode = 1;
    double scorer_w = 0.25;
    bool cap = true;           // apply the model's copy cap (K5::load's cap; a model built with cap 0 has none)
    double temperature = 1.0;  // fuzzy_bake.py: 1.0
    double cand_mass = 0.995;  // --k5-cand-mass P
    double nucleus = 0.8;      // --k5-nucleus P: 0.8, adopted by K5_UPGRADES.md step 1 (§13.19's K5: 0.9)
    std::size_t cand_max = 3000;              // --k5-cand-max N
    int max_words = 300;
    std::size_t bytes = 600;
    // docs/reports/K5_UPGRADES.md. The defaults are the adopted arms (nucleus 0.8, 256 exchange classes as the backoff
    // floor, the 8-candidate sentence gate by prompt association); --k5-base k5 restores §13.19's K5 (k5_v1 below).
    std::uint64_t seed_off = 0;                        // --k5-seed N: seeds 20261101 + N + the prompt's index
    double cache_w = 0, cache_w2 = 0, cache_r = 0.99;  // step 2a: --k5-cache W, --k5-cache2 W2, --k5-cache-decay R
    double dyn = 0;                                    // step 2b: --k5-dyn L (the dynamic counts' weight)
    int order = kOrder;                                // step 3: --k5-order N (build)
    std::size_t classes = 256;                         // step 3: --k5-classes K (build; exchange classes; 0: none)
    int class_mode = 1;                                // --k5-class-mode backoff (1) | mix (2)
    double class_w = 0.1;                              // --k5-class-w W (mix)
    int gate_k = 8;                                    // step 4: --k5-gate K[,wf=..][,wp=..][,wm=..][,min=..][,max=..] (1: off)
    double gate_wf = 0, gate_wp = 1, gate_wm = 5;
    std::size_t gate_min = 24, gate_max = 240;
    // K5_STRUCTURE.md: --k5-struct FILE (cyphalm_trie loads it); null: K5's own gate. desc goes on the MODEL line.
    std::shared_ptr<const StructureWriter> structure;
    std::string structure_desc;
    // docs/reports/K5_DISCOURSE.md (GENERATION_PLAN §13.32): K5's discourse helpers, all off by default. With any of
    // mbr_sent_w, eg_w or lda_w set, the gate z-scores each of its terms over the K candidates before weighing them.
    double mbr_sent_w = 0;      // --k5-mbr-sent W: the gate's MBR term (a candidate's mean chrF to the other candidates)
    int mbr_doc = 0;            // --k5-mbr-doc N: N whole drafts, the one with the highest mean chrF to the others kept
    double eg_w = 0;            // --k5-eg W: the gate's entity-grid term
    int eg_mode = 1;            // --k5-eg-mode full (1: the grid's log-probability change) | cont (2: kept entities only)
    std::size_t lda_k = 0;      // --k5-lda K: an LDA topic model with K topics (build; 0: none)
    int lda_iters = 200;        // --k5-lda-iters N: its Gibbs sweeps
    double lda_beta = 0;        // --k5-lda-beta B: p <- p * (p_topic(w | prompt) / p(w))^B on content words, renormalised
    double lda_w = 0;           // --k5-lda-w W: the gate's topic term (mean log2 p_topic / p over the candidate's words)
    std::string meta;           // --k5-meta SPEC: MBR over the pooled drafts of variants, e.g. base:2,gate:2,eg:2,topic:2
    std::string disc_text;      // --k5-disc-text PATH: the text the grid and LDA learn from (default: K5's own)
    std::string disc_cache;     // --k5-disc-cache DIR: load the grid and LDA from DIR, or train and save them there
    unsigned disc_threads = 8;  // --k5-disc-threads N: the grid's counting threads (exact on any number)
    unsigned lda_threads = 1;   // --k5-lda-threads N: LDA's Gibbs threads (1: exact; more: AD-LDA, which fit worse here)
    bool zgate() const { return mbr_sent_w != 0 || eg_w != 0 || lda_w != 0; }
    // The settings that differ from K5's, for the MODEL line (empty for K5 itself).
    std::string describe() const {
        std::string s;
        const auto add = [&](const char* k, double v) {
            char b[64];
            std::snprintf(b, sizeof b, "\t%s=%g", k, v);
            s += b;
        };
        const Options d;
        if (nucleus != d.nucleus) add("nucleus", nucleus);
        if (cand_mass != d.cand_mass) add("cand_mass", cand_mass);
        if (cand_max != d.cand_max) add("cand_max", static_cast<double>(cand_max));
        if (seed_off) add("seed_off", static_cast<double>(seed_off));
        if (cache_w > 0 || cache_w2 > 0) add("cache", cache_w), add("cache2", cache_w2), add("cache_decay", cache_r);
        if (dyn > 0) add("dyn", dyn);
        if (order != kOrder) add("order", order);
        if (classes != d.classes || class_mode != d.class_mode || class_w != d.class_w)
            add("classes", static_cast<double>(classes)), add(class_mode == 2 ? "class_mix" : "class_backoff", class_mode == 2 ? class_w : 1);
        if (gate_k != d.gate_k || gate_wf != d.gate_wf || gate_wp != d.gate_wp || gate_wm != d.gate_wm || gate_min != d.gate_min || gate_max != d.gate_max)
            add("gate", gate_k > 1 ? gate_k : 0), add("wf", gate_wf), add("wp", gate_wp), add("wm", gate_wm);
        if (structure) s += "\tstruct=" + structure_desc;  // K5_STRUCTURE.md
        // K5_DISCOURSE.md
        if (mbr_sent_w != 0) add("mbr_sent", mbr_sent_w);
        if (mbr_doc > 1) add("mbr_doc", mbr_doc);
        if (eg_w != 0) add("eg", eg_w), add("eg_mode", eg_mode);
        if (lda_k > 0) add("lda", static_cast<double>(lda_k)), add("lda_iters", lda_iters), add("lda_beta", lda_beta), add("lda_w", lda_w);
        if (!meta.empty()) s += "\tmeta=" + meta;
        if (!disc_text.empty()) s += "\tdisc_text=" + disc_text;
        if (disc_threads != d.disc_threads) add("disc_threads", disc_threads);
        if (lda_threads != d.lda_threads) add("lda_threads", lda_threads);
        return s;
    }
};

// §13.19's K5, before K5_UPGRADES.md: nucleus 0.9, no classes, no gate (--k5-base k5).
inline Options k5_v1() {
    Options o;
    o.nucleus = 0.9;
    o.classes = 0;
    o.gate_k = 0, o.gate_wf = 1, o.gate_wp = 0;
    return o;
}

// Reads one --k5-... flag of K5_UPGRADES.md into o. Returns false and err for a bad value or an unknown name.
inline bool apply_flag(const std::string& name, const std::string& val, Options& o, std::string& err) {
    const auto num = [&](double& x, double lo, double hi) {
        char* e = nullptr;
        x = std::strtod(val.c_str(), &e);
        if (val.empty() || *e || !(x >= lo && x <= hi)) {
            err = name + ": bad value " + val;
            return false;
        }
        return true;
    };
    double x = 0;
    if (name == "--k5-base") {  // reset every K5 setting: k5 (§13.19's K5) or k5up (the defaults); later flags apply on top
        if (val != "k5" && val != "k5up") return err = name + ": k5 or k5up", false;
        const auto keep_scorer = o.scorer;
        const auto keep_post = o.post;
        const auto keep_desc = o.post_desc;
        o = val == "k5" ? k5_v1() : Options{};
        o.scorer = keep_scorer;
        o.post = keep_post, o.post_desc = keep_desc;
        return true;
    }
    if (name == "--k5-seed") return num(x, 0, 1e15) && ((o.seed_off = static_cast<std::uint64_t>(x)), true);
    if (name == "--k5-nucleus") return num(o.nucleus, 1e-6, 1);
    if (name == "--k5-cand-mass") return num(o.cand_mass, 1e-6, 1);
    if (name == "--k5-cand-max") return num(x, 1, 1e6) && ((o.cand_max = static_cast<std::size_t>(x)), true);
    if (name == "--k5-cache") return num(o.cache_w, 0, 0.9);
    if (name == "--k5-cache2") return num(o.cache_w2, 0, 0.9);
    if (name == "--k5-cache-decay") return num(o.cache_r, 0.5, 1);
    if (name == "--k5-dyn") return num(o.dyn, 0, 1e6);
    if (name == "--k5-order") return num(x, 2, kMaxOrder) && ((o.order = static_cast<int>(x)), true);
    if (name == "--k5-classes") return num(x, 0, 4096) && ((o.classes = static_cast<std::size_t>(x)), true);
    if (name == "--k5-class-w") return num(o.class_w, 0, 1);
    if (name == "--k5-class-mode") {
        if (val != "backoff" && val != "mix") return err = name + ": backoff or mix", false;
        o.class_mode = val == "mix" ? 2 : 1;
        return true;
    }
    // K5_DISCOURSE.md
    if (name == "--k5-mbr-sent") return num(o.mbr_sent_w, -100, 100);
    if (name == "--k5-mbr-doc") return num(x, 0, 64) && ((o.mbr_doc = static_cast<int>(x)), true);
    if (name == "--k5-eg") return num(o.eg_w, -100, 100);
    if (name == "--k5-eg-mode") {
        if (val != "full" && val != "cont") return err = name + ": full or cont", false;
        o.eg_mode = val == "cont" ? 2 : 1;
        return true;
    }
    if (name == "--k5-lda") return num(x, 0, 2000) && ((o.lda_k = static_cast<std::size_t>(x)), true);
    if (name == "--k5-lda-iters") return num(x, 1, 100000) && ((o.lda_iters = static_cast<int>(x)), true);
    if (name == "--k5-lda-beta") return num(o.lda_beta, 0, 10);
    if (name == "--k5-lda-w") return num(o.lda_w, -100, 100);
    if (name == "--k5-disc-text") return val.empty() ? (err = name + ": empty", false) : ((o.disc_text = val), true);
    if (name == "--k5-disc-cache") return val.empty() ? (err = name + ": empty", false) : ((o.disc_cache = val), true);
    if (name == "--k5-disc-threads") return num(x, 1, 64) && ((o.disc_threads = static_cast<unsigned>(x)), true);
    if (name == "--k5-lda-threads") return num(x, 1, 64) && ((o.lda_threads = static_cast<unsigned>(x)), true);
    if (name == "--k5-meta") {
        if (val.empty()) return err = name + ": empty", false;
        for (std::size_t a = 0; a <= val.size();) {
            std::size_t b = val.find(',', a);
            if (b == std::string::npos) b = val.size();
            const std::string f = val.substr(a, b - a);
            const std::size_t q = f.find(':');
            const std::string v = f.substr(0, q);
            if (q == std::string::npos || (v != "base" && v != "gate" && v != "eg" && v != "topic" && v != "all") || std::atoi(f.c_str() + q + 1) < 1)
                return err = name + ": fields are VARIANT:COUNT, VARIANT one of base, gate, eg, topic, all", false;
            a = b + 1;
        }
        o.meta = val;
        return true;
    }
    if (name == "--k5-gate") {
        std::size_t a = 0;
        bool first = true;
        while (a <= val.size()) {
            std::size_t b = val.find(',', a);
            if (b == std::string::npos) b = val.size();
            const std::string f = val.substr(a, b - a);
            char* e = nullptr;
            if (first) {
                o.gate_k = static_cast<int>(std::strtol(f.c_str(), &e, 10));
                if (f.empty() || *e || o.gate_k < 1 || o.gate_k > 64) return err = name + ": K in [1, 64]", false;
            } else {
                const std::size_t q = f.find('=');
                if (q == std::string::npos) return err = name + ": bad field " + f, false;
                const std::string k = f.substr(0, q);
                const double v = std::strtod(f.c_str() + q + 1, &e);
                if (*e) return err = name + ": bad field " + f, false;
                if (k == "wf") o.gate_wf = v;
                else if (k == "wp") o.gate_wp = v;
                else if (k == "wm") o.gate_wm = v;
                else if (k == "min") o.gate_min = static_cast<std::size_t>(v);
                else if (k == "max") o.gate_max = static_cast<std::size_t>(v);
                else return err = name + ": unknown field " + k, false;
            }
            first = false;
            a = b + 1;
        }
        return true;
    }
    err = "unknown flag " + name;
    return false;
}

// §K5 upgrades, step 2a: the word cache of one continuation. Each word of the prompt and of the text so far weighs
// R^(its age in words); unigram and (previous word, word) pairs. <unk> and <s> are never cached.
class WordCache {
 public:
    void init(std::size_t V, double r) {
        c1_.assign(V, 0.0);
        r_ = r;
    }
    void push(std::uint32_t w) {
        scale_ /= r_;
        if (scale_ > 1e200) rescale();
        if (w >= 2) {
            if (c1_[w] == 0) present_.push_back(w);
            c1_[w] += scale_;
            z1_ += scale_;
            if (prev_ >= 2) {
                Big& b = c2_[prev_];
                bool found = false;
                for (auto& [x, c] : b.e)
                    if (x == w) c += scale_, found = true;
                if (!found) b.e.emplace_back(w, scale_);
                b.tot += scale_;
            }
        }
        prev_ = w;
    }
    // p <- (1 - a1 - a2) p + a1 cache1 + a2 cache2(. | previous word); a2 is 0 when the previous word has no cached pair.
    void apply(double* p, std::size_t V, double w1, double w2) const {
        const Big* b = nullptr;
        if (w2 > 0 && prev_ >= 2) {
            const auto it = c2_.find(prev_);
            if (it != c2_.end()) b = &it->second;
        }
        const double a1 = z1_ > 0 ? w1 : 0.0, a2 = b ? w2 : 0.0;
        if (a1 == 0 && a2 == 0) return;
        const double f = 1 - a1 - a2;
        for (std::size_t w = 0; w < V; ++w) p[w] *= f;
        if (a1 > 0)
            for (const std::uint32_t x : present_) p[x] += a1 * c1_[x] / z1_;
        if (a2 > 0)
            for (const auto& [x, c] : b->e) p[x] += a2 * c / b->tot;
    }

 private:
    struct Big {
        std::vector<std::pair<std::uint32_t, double>> e;
        double tot = 0;
    };
    void rescale() {
        const double f = 1 / scale_;
        for (const std::uint32_t x : present_) c1_[x] *= f;
        z1_ *= f;
        for (auto& [k, b] : c2_) {
            for (auto& [x, c] : b.e) c *= f;
            b.tot *= f;
        }
        scale_ = 1;
    }
    std::vector<double> c1_;
    std::vector<std::uint32_t> present_;
    std::unordered_map<std::uint32_t, Big> c2_;
    double z1_ = 0, scale_ = 1, r_ = 0.99;
    std::uint32_t prev_ = kBos;
};

struct GenStats {
    std::size_t steps = 0;
    std::size_t rep_banned = 0;      // candidates the repeat ban removed
    std::size_t cap_steps = 0;       // steps at which the cap removed a candidate
    std::size_t cap_banned = 0;      // candidates the cap removed
    std::size_t cap_mass_steps = 0;  // ... of which the cap removed mass the draw could reach (inside the 0.9 nucleus)
    std::size_t rep_lifted = 0;      // steps at which the repeat ban left nothing and was lifted
    std::size_t cap_outside = 0;     // steps at which the cap left no candidate (a word from outside them was taken)
    std::size_t gate_sents = 0, gate_cands = 0;  // §K5 upgrades step 4: sentences gated, candidates written
    std::unordered_map<std::string, std::size_t> pool_chosen;  // K5_DISCOURSE.md: MBR choices per pool variant
    void add(const GenStats& o) {
        steps += o.steps, rep_banned += o.rep_banned, cap_steps += o.cap_steps, cap_banned += o.cap_banned;
        cap_mass_steps += o.cap_mass_steps, rep_lifted += o.rep_lifted, cap_outside += o.cap_outside;
        gate_sents += o.gate_sents, gate_cands += o.gate_cands;
        for (const auto& [k, v] : o.pool_chosen) pool_chosen[k] += v;
    }
};

class K5 {
 public:
    // Training text: one document per line. Returns false (and err) if it cannot be read or holds no tokens.
    // bo: the build options (order, classes); null for K5's. vmax: the vocabulary cap (K5_SCALE: --k5-vmax).
    bool load(const std::string& path, int cap, std::string& err, const Options* bo = nullptr, std::size_t vmax = kVmax) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            err = "cannot read " + path;
            return false;
        }
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        build(text, cap, bo ? bo->order : kOrder, bo ? bo->classes : 0, vmax);
        if (vocab_.size() <= 2) {
            err = path + " holds no tokens seen twice";
            return false;
        }
        if (bo) build_discourse(text, *bo);
        return true;
    }

    // K5_DISCOURSE.md: the entity grid (when o asks for it) and the LDA topic model (o.lda_k > 0), trained on the same
    // text, one document per line, K5's tokens.
    void build_discourse(std::string_view k5_text, const Options& o) {
        const bool want_eg = o.eg_w != 0 || o.meta.find("eg") != std::string::npos;
        if (!want_eg && o.lda_k == 0) return;
        const auto t0 = std::chrono::steady_clock::now();
        // The helpers' text: --k5-disc-text PATH (e.g. 330 MB of clean enwik9), else K5's own.
        std::string own;
        std::string_view text = k5_text;
        std::string tag = "k5text";
        if (!o.disc_text.empty()) {
            tag = o.disc_text.substr(o.disc_text.find_last_of('/') + 1);
            std::ifstream in(o.disc_text, std::ios::binary);
            own.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            text = own;
        }
        const auto tok = [](std::string_view line, auto&& emit) { tokenize(line, emit); };
        const auto cache = [&](const std::string& name) { return o.disc_cache.empty() ? std::string{} : o.disc_cache + "/" + name; };
        if (want_eg) {
            const std::string cp = cache("eg_" + tag + ".bin");
            const bool hit = !cp.empty() && eg_.load(cp);
            if (!hit) {
                eg_.train_lines(text, tok, o.disc_threads);
                if (!cp.empty()) eg_.save(cp);
            }
            std::fprintf(stderr, "k5: entity grid on %s (%zu bytes%s): %zu docs, %zu sentences, %zu entity columns; P(S|--) %.4f P(-|SS) %.4f "
                         "P(S|-S) %.4f (%.1f s)\n", tag.c_str(), text.size(), hit ? ", cached" : "", eg_.docs(), eg_.sentences(), eg_.columns(),
                         std::exp2(eg_.logp(eg::kAbsent, eg::kAbsent, eg::kS)), std::exp2(eg_.logp(eg::kS, eg::kS, eg::kAbsent)),
                         std::exp2(eg_.logp(eg::kAbsent, eg::kS, eg::kS)), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        if (o.lda_k > 0) {
            const auto t1 = std::chrono::steady_clock::now();
            lda::Params lp;
            lp.K = o.lda_k;
            lp.iters = o.lda_iters;
            lp.threads = o.lda_threads;
            const std::string cp = cache("lda_" + tag + "_K" + std::to_string(lp.K) + "_it" + std::to_string(lp.iters) + "_t" + std::to_string(lp.threads) + ".bin");
            const bool hit = !cp.empty() && lda_.load(cp);
            if (!hit) {
                lda_.train_lines(text, tok, lp);
                if (!cp.empty()) lda_.save(cp);
            }
            k2l_.assign(vocab_.size(), -1);
            for (std::size_t w = 2; w < vocab_.size(); ++w) k2l_[w] = lda_.id(vocab_[w]);
            std::fprintf(stderr, "k5: LDA on %s (%zu bytes%s): %zu topics, %zu docs, %zu content words, %zu tokens, %d sweeps, %u threads; log P(w|z)",
                         tag.c_str(), text.size(), hit ? ", cached" : "", lda_.topics(), lda_.corpus_docs(), lda_.vocab_size(), lda_.corpus_tokens(),
                         lp.iters, lp.threads);
            for (const double x : lda_.loglik()) std::fprintf(stderr, " %.0f", x);
            std::fprintf(stderr, " (%.1f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
        }
    }
    const eg::EntityGrid& entity_grid() const { return eg_; }
    const lda::LDA& topic_model() const { return lda_; }

    // vmax: the vocabulary's size cap (K5_SCALE: --k5-vmax), at most 65,536 (ids are 16 bits).
    void build(std::string_view text, int cap, int order = kOrder, std::size_t classes = 0, std::size_t vmax = kVmax) {
        vmax = std::min<std::size_t>(vmax, 65536);
        const auto t0 = std::chrono::steady_clock::now();
        // Types in first-occurrence order and every token's type (~0u: a document's start).
        std::unordered_map<std::string_view, std::uint32_t> type;
        std::vector<std::string_view> tview;
        std::vector<std::uint64_t> tcount;
        std::vector<std::uint32_t> seq;
        docs_ = 0;
        for (std::size_t a = 0; a < text.size();) {
            std::size_t b = text.find('\n', a);
            if (b == std::string_view::npos) b = text.size();
            if (b > a) {
                ++docs_;
                seq.push_back(~0u);
                tokenize(text.substr(a, b - a), [&](std::string_view t) {
                    const auto [it, fresh] = type.try_emplace(t, static_cast<std::uint32_t>(tview.size()));
                    if (fresh) {
                        tview.push_back(t);
                        tcount.push_back(0);
                    }
                    ++tcount[it->second];
                    seq.push_back(it->second);
                });
            }
            a = b + 1;
        }
        std::vector<std::uint32_t> order_t(tview.size());
        for (std::uint32_t i = 0; i < order_t.size(); ++i) order_t[i] = i;
        std::stable_sort(order_t.begin(), order_t.end(), [&](std::uint32_t x, std::uint32_t y) { return tcount[x] > tcount[y]; });
        vocab_ = {"<unk>", "<s>"};
        std::vector<std::uint32_t> id_of(tview.size(), kUnk);
        for (const std::uint32_t t : order_t) {
            if (tcount[t] < 2 || vocab_.size() >= vmax) break;
            id_of[t] = static_cast<std::uint32_t>(vocab_.size());
            vocab_.emplace_back(tview[t]);
        }
        tid_.clear();
        tid_.reserve(vocab_.size() * 2);
        for (std::uint32_t i = 0; i < vocab_.size(); ++i) tid_.emplace(vocab_[i], i);
        {
            std::vector<std::uint32_t> ids;
            ids.reserve(seq.size());
            unk_ = 0;
            for (const std::uint32_t s : seq) {
                ids.push_back(s == ~0u ? kBos : id_of[s]);
                unk_ += s != ~0u && id_of[s] == kUnk;
            }
            std::vector<std::uint32_t>().swap(seq);
            tokens_ = ids.size();
            // One suffix array over the ids gives the counts, the copy cap's windows, fuzzy_bake.py has_run's 13-token
            // runs and each word's positions (GENERATION_PLAN §13.40 M1).
            std::uint32_t keep = WordKN::kKeep;
            if (const char* e = std::getenv("CYPHA_K5_KEEP")) keep = static_cast<std::uint32_t>(std::max(1, std::atoi(e)));
            kn_.build(std::move(ids), vocab_.size(), order, keep);
        }
        const std::vector<std::uint32_t>& ids = kn_.index().text();
        hs13_ = std::make_unique<NgramSet::Hashes>();
        hscap_ = std::make_unique<NgramSet::Hashes>();
        cap_ = cap;
        // Each word's count (the sentence gate's prompt association; its positions are its suffix-array interval).
        const std::size_t V = vocab_.size();
        cnt_.assign(V, 0.0);
        for (std::uint32_t w = 0; w < V; ++w) cnt_[w] = static_cast<double>(kn_.index().first(w).size());
        if (classes > 0) cm_.build(ids, V, classes);
        else cm_ = ClassModel{};
        fast_ = std::make_unique<FastTables>();  // OPT_NUCLEUS.md: made on the first draw
        secs_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    std::uint32_t id(std::string_view t) const {
        const auto it = tid_.find(std::string(t));
        return it == tid_.end() ? kUnk : it->second;
    }
    const std::vector<std::string>& vocab() const { return vocab_; }
    const WordKN& kn() const { return kn_; }
    const ClassModel& classes() const { return cm_; }
    // The copy cap's windows (cap + 1 tokens; none with cap 0), a view on the suffix array.
    NgramSet cap_set() const { return cap_ + 1 == 13 ? run13() : NgramSet(&kn_.index(), cap_ > 0 ? cap_ + 1 : 0, hscap_.get()); }
    int cap() const { return cap_; }
    std::size_t docs() const { return docs_; }
    std::size_t tokens() const { return tokens_; }
    std::size_t unk() const { return unk_; }
    double build_seconds() const { return secs_; }
    std::size_t bytes() const {
        std::size_t b = kn_.bytes() + vocab_.size() * 64;
        for (const auto* h : {hs13_.get(), hscap_.get()})
            if (h) b += (h->h.capacity() + h->hp.capacity()) * 8;
        return b;
    }

    // fuzzy_bake.py has_run(g): does g, tokenised and mapped to ids, hold 13 consecutive ids of the training stream?
    bool has_run13(std::string_view g) const {
        std::vector<std::uint32_t> ids;
        tokenize(g, [&](std::string_view t) { ids.push_back(id(t)); });
        return run13().any(ids);
    }

    // One continuation's state: the context, the repeat ban, the text, and the online parts (step 2).
    struct State {
        std::vector<std::uint32_t> ctx;
        std::string prev;
        bool has_prev = false;
        RepeatBan rep;
        std::string out;
        Overlay ov;
        WordCache cache;
        int step = 0;
        std::shared_ptr<const std::vector<double>> topic_mult;  // K5_DISCOURSE.md: --k5-lda-beta's factor per word
    };
    State start(std::string_view prompt, const Options& o) const {
        State s;
        s.ctx = {kBos};
        tokenize(prompt, [&](std::string_view t) {
            s.ctx.push_back(id(t));
            s.prev.assign(t);
            s.has_prev = true;
        });
        s.rep = RepeatBan(prompt);
        if (o.dyn > 0) {
            s.ov.L = o.dyn;
            s.ov.order = kn_.order();
            for (const std::uint32_t w : s.ctx) s.ov.push(w);
        }
        if (o.cache_w > 0 || o.cache_w2 > 0) {
            s.cache.init(vocab_.size(), o.cache_r);
            for (const std::uint32_t w : s.ctx) s.cache.push(w);
        }
        if (o.lda_beta > 0 && lda_.trained()) {  // K5_DISCOURSE.md: the prompt's topic mixture rescales content words
            const auto lr = topic_lr(prompt);
            auto m = std::make_shared<std::vector<double>>(vocab_.size(), 1.0);
            for (std::size_t w = 0; w < lr.size(); ++w)
                if (lr[w] != 0) (*m)[w] = std::exp2(o.lda_beta * lr[w]);
            s.topic_mult = std::move(m);
        }
        return s;
    }
    // K5_DISCOURSE.md: for every K5 word, log2 (p_topic(w | prompt) / p(w)) clipped to [-4, 4] for content words, 0
    // for the others.
    std::vector<double> topic_lr(std::string_view prompt) const {
        std::vector<double> lr(vocab_.size(), 0.0);
        if (!lda_.trained()) return lr;
        const auto theta = lda_.infer(wkn::tokens(prompt));
        const auto q = lda_.word_dist(theta);
        for (std::size_t w = 0; w < vocab_.size(); ++w)
            if (k2l_[w] >= 0) {
                const auto x = static_cast<std::size_t>(k2l_[w]);
                lr[w] = std::clamp(std::log2(q[x] / lda_.unigram(x)), -4.0, 4.0);
            }
        return lr;
    }
    // The state reads word w (the online parts learn it).
    static void learn(State& s, std::uint32_t w, const Options& o) {
        s.ctx.push_back(w);
        if (o.dyn > 0) s.ov.push(w);
        if (o.cache_w > 0 || o.cache_w2 > 0) s.cache.push(w);
    }
    // P(. | state) into p: K5 (with the dynamic counts and the class floor), then the class mix, then the cache.
    void model_dist(const State& s, const Options& o, double* p, std::vector<double>& tmp) const {
        const std::size_t V = vocab_.size();
        const double* base = nullptr;
        if (cm_.on()) {
            tmp.resize(V);
            cm_.dist(s.ctx.data(), s.ctx.size(), tmp.data());
            if (o.class_mode == 1) base = tmp.data();
        }
        kn_.dist(s.ctx.data(), s.ctx.size(), p, o.dyn > 0 ? &s.ov : nullptr, base);
        if (cm_.on() && o.class_mode == 2)
            for (std::size_t w = 0; w < V; ++w) p[w] = (1 - o.class_w) * p[w] + o.class_w * tmp[w];
        if (o.cache_w > 0 || o.cache_w2 > 0) s.cache.apply(p, V, o.cache_w, o.cache_w2);
        if (s.topic_mult) {  // K5_DISCOURSE.md: topic rescaling, renormalised
            const double* m = s.topic_mult->data();
            double z = 0;
            for (std::size_t w = 0; w < V; ++w) p[w] *= m[w], z += p[w];
            for (std::size_t w = 0; w < V; ++w) p[w] /= z;
        }
    }

    // docs/reports/K5_MCMC.md: model_dist's p[w] alone, after ctx[0, n) (its last word last). Without the online parts
    // and the class mix (o.dyn 0, no cache, class_mode backoff), which word_prob_ok checks.
    static bool word_prob_ok(const Options& o) { return o.dyn <= 0 && o.cache_w <= 0 && o.cache_w2 <= 0 && o.class_mode == 1; }
    double word_prob(const std::uint32_t* ctx, std::size_t n, std::uint32_t w) const {
        return kn_.prob(ctx, n, w, cm_.on() ? cm_.prob(ctx, n, w) : kn_.p1()[w]);
    }

    // fuzzy_bake.py cont_bits: bits of the continuation's tokens under KN (no zeroing; an unknown word gets <unk>'s
    // mass) and the bytes of their pieces. With o, the online parts read the prompt and each continuation word in turn.
    std::pair<double, double> cont_bits(std::string_view prompt, std::string_view cont, const Options& o = Options{}) const {
        State s = start(prompt, o);
        std::vector<double> p(vocab_.size()), tmp;
        double bits = 0, nb = 0;
        tokenize(cont, [&](std::string_view t) {
            const std::uint32_t y = id(t);
            model_dist(s, o, p.data(), tmp);
            bits -= std::log2(std::max(p[y], 1e-12));
            nb += static_cast<double>(piece(t, s.prev, s.has_prev).size());
            learn(s, y, o);
            s.prev.assign(t);
            s.has_prev = true;
        });
        return {bits, nb};
    }

    // §K5 upgrades step 4: the association of every word with the prompt's words, the mean over the prompt's words v
    // outside the 100 most frequent of PPMI(v, w) = max(0, log2(c(v, w) N / (2 W c(v) c(w)))), c(v, w) counted within W
    // words in a document of the training text, kept only when c(v, w) >= 2. K5's own counts; nothing of SCRIBE.
    std::vector<double> prompt_assoc(const std::vector<std::uint32_t>& ctx, std::size_t W = 10, std::size_t skip_top = 100) const {
        const std::size_t V = vocab_.size();
        std::vector<double> a(V, 0.0), c(V, 0.0);
        std::vector<std::uint32_t> vs, touched;
        for (const std::uint32_t v : ctx)
            if (v >= 2 + skip_top && std::find(vs.begin(), vs.end(), v) == vs.end()) vs.push_back(v);
        if (vs.empty()) return a;
        const std::vector<std::uint32_t>& ids = kn_.index().text();
        const double N = static_cast<double>(ids.size());
        for (const std::uint32_t v : vs) {
            touched.clear();
            // v's positions: its suffix-array interval (in suffix order; the counts below do not depend on the order).
            const NgramSet::Index::Range r = kn_.index().first(v);
            for (std::uint32_t e = r.lo; e < r.hi; ++e) {
                const std::size_t t = kn_.index().sa(e);
                for (std::size_t j = t; j-- > 0 && t - j <= W;) {
                    if (ids[j] == kBos) break;
                    if (c[ids[j]] == 0) touched.push_back(ids[j]);
                    c[ids[j]] += 1;
                }
                for (std::size_t j = t + 1; j < ids.size() && j - t <= W; ++j) {
                    if (ids[j] == kBos) break;
                    if (c[ids[j]] == 0) touched.push_back(ids[j]);
                    c[ids[j]] += 1;
                }
            }
            for (const std::uint32_t w : touched) {
                if (c[w] >= 2 && w >= 2) {
                    const double pmi = std::log2(c[w] * N / (2.0 * static_cast<double>(W) * cnt_[v] * cnt_[w]));
                    if (pmi > 0) a[w] += pmi;
                }
                c[w] = 0;
            }
        }
        for (double& x : a) x /= static_cast<double>(vs.size());
        return a;
    }

    // fuzzy_bake.py generate_one: the continuation of prompt, seeded with seed. With o.gate_k > 1, the sentence gate.
    std::string generate(std::string_view prompt, std::uint64_t seed, const Options& o, GenStats* st = nullptr) const {
        if (o.structure) return o.structure->generate(*this, prompt, seed, o, st);  // K5_STRUCTURE.md
        if (o.mbr_doc > 1 || !o.meta.empty()) return generate_pool(prompt, seed, o, st);  // K5_DISCOURSE.md
        GenStats g;
        const std::size_t V = vocab_.size();
        if (V <= 2) return {};  // no word but <unk> and <s>
        State s = start(prompt, o);
        Scratch sc;
        sc.p.resize(V);
        if (o.gate_k <= 1) {
            std::mt19937_64 rng(seed);
            std::unique_ptr<ByteScorer> bs = o.scorer ? o.scorer() : nullptr;  // §13.24
            if (bs) bs->begin(prompt);
            while (s.step < o.max_words) {
                const std::uint64_t salt = mix64(seed ^ mix64(static_cast<std::uint64_t>(s.step) + 0x5EED));
                step_word(s, rng, salt, o, g, bs.get(), sc, nullptr);
                if (s.out.size() >= o.bytes) break;
            }
        } else if (o.zgate()) {
            gate_z(prompt, seed, o, s, sc, g);  // K5_DISCOURSE.md
        } else {
            // The sentence gate: K candidates from the same state, each with its own stream; the best by CyphaLM's
            // own score is kept.
            const std::vector<double> assoc = o.gate_wp != 0 ? prompt_assoc(s.ctx) : std::vector<double>{};
            for (std::uint64_t sent = 0; s.step < o.max_words && s.out.size() < o.bytes; ++sent) {
                State best;
                GenStats bg;
                double bscore = 0;
                for (int k = 0; k < o.gate_k; ++k) {
                    State c = s;
                    GenStats gk;
                    const std::uint64_t cs = mix64(seed ^ mix64(0x6A7E000000ull + sent * 256 + static_cast<std::uint64_t>(k)));
                    std::mt19937_64 rng(cs);
                    const std::size_t b0 = c.out.size();
                    double slp = 0, spmi = 0;
                    std::size_t nw = 0;
                    while (c.step < o.max_words && c.out.size() < o.bytes) {
                        const std::uint64_t salt = mix64(cs ^ mix64(static_cast<std::uint64_t>(c.step) + 0x5EED));
                        double lp = 0;
                        const std::uint32_t w = step_word(c, rng, salt, o, gk, nullptr, sc, &lp);
                        slp += lp, spmi += assoc.empty() ? 0.0 : assoc[w], ++nw;
                        const std::size_t nb = c.out.size() - b0;
                        const std::string& t = vocab_[w];
                        if (nb >= o.gate_max) break;
                        if (nb >= o.gate_min && (t == "." || t == "!" || t == "?" || t == "\n")) break;
                    }
                    std::size_t mk = 0;
                    for (std::size_t i = b0; i < c.out.size(); ++i) mk += std::string_view("[]{}|=<>&*#").find(c.out[i]) != std::string_view::npos;
                    const double nb = static_cast<double>(std::max<std::size_t>(c.out.size() - b0, 1)), n = static_cast<double>(std::max<std::size_t>(nw, 1));
                    const double score = o.gate_wf * slp / n + o.gate_wp * spmi / n - o.gate_wm * static_cast<double>(mk) / nb;
                    ++g.gate_cands;
                    if (k == 0 || score > bscore) bscore = score, best = std::move(c), bg = gk;
                }
                s = std::move(best);
                g.add(bg);
                ++g.gate_sents;
            }
        }
        std::string out = std::move(s.out);
        cut_bytes(out, o.bytes);
        if (st) st->add(g);
        return out;
    }

    // Cut to the byte limit; a character the cut splits is dropped (decode "ignore").
    static void cut_bytes(std::string& out, std::size_t bytes) {
        if (out.size() > bytes) {
            out.resize(bytes);
            std::size_t i = out.size();
            std::size_t back = 0;
            while (i > 0 && back < 4 && (static_cast<unsigned char>(out[i - 1]) & 0xC0) == 0x80) --i, ++back;
            if (i > 0) {
                const auto c = static_cast<unsigned char>(out[i - 1]);
                const std::size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
                if (need > back + 1) out.resize(i - 1);
            }
        }
    }

    // K5_STRUCTURE.md: one candidate sentence of the gate, drawn from s (left as it was) with the stream of sentence
    // sent and candidate k of seed, as generate's gate draws it. c gets the state after it; words its words; lp their
    // log2 probabilities (the gate's fluency).
    void draft(const State& s, std::uint64_t seed, std::uint64_t sent, int k, const Options& o, State& c,
               std::vector<std::uint32_t>& words, std::vector<double>& lp, GenStats& g) const {
        c = s;
        words.clear(), lp.clear();
        thread_local Scratch sc;
        sc.p.resize(vocab_.size());
        const std::uint64_t cs = mix64(seed ^ mix64(0x6A7E000000ull + sent * 256 + static_cast<std::uint64_t>(k)));
        std::mt19937_64 rng(cs);
        const std::size_t b0 = c.out.size();
        while (c.step < o.max_words && c.out.size() < o.bytes) {
            const std::uint64_t salt = mix64(cs ^ mix64(static_cast<std::uint64_t>(c.step) + 0x5EED));
            double l = 0;
            const std::uint32_t w = step_word(c, rng, salt, o, g, nullptr, sc, &l);
            words.push_back(w), lp.push_back(l);
            const std::size_t nb = c.out.size() - b0;
            const std::string& t = vocab_[w];
            if (nb >= o.gate_max) break;
            if (nb >= o.gate_min && (t == "." || t == "!" || t == "?" || t == "\n")) break;
        }
        ++g.gate_cands;
    }
    // K5_STRUCTURE.md: the training text's ids (documents start with <s>).
    const std::vector<std::uint32_t>& train_ids() const { return kn_.index().text(); }
    // K5_DISCOURSE.md: the drafts of --k5-mbr-doc N (N drafts of o) or --k5-meta SPEC (the pooled drafts of variants of
    // o), and the MBR choice among them (the highest mean chrF to the other drafts). Draft i draws from its own seed.
    std::vector<std::pair<std::string, Options>> pool_variants(const Options& o) const {
        std::vector<std::pair<std::string, Options>> v;
        Options b = o;
        b.mbr_doc = 0, b.meta.clear();
        if (o.meta.empty()) {
            for (int i = 0; i < o.mbr_doc; ++i) v.emplace_back("draft", b);
            return v;
        }
        for (std::size_t a = 0; a <= o.meta.size();) {
            std::size_t e = o.meta.find(',', a);
            if (e == std::string::npos) e = o.meta.size();
            const std::string f = o.meta.substr(a, e - a);
            const std::size_t q = f.find(':');
            const std::string name = f.substr(0, q);
            const int n = std::atoi(f.c_str() + q + 1);
            Options x = b;
            if (name == "base") x.gate_k = 1, x.lda_beta = 0, x.eg_w = 0, x.lda_w = 0, x.mbr_sent_w = 0;
            else if (name == "gate") x.lda_beta = 0, x.eg_w = 0, x.lda_w = 0, x.mbr_sent_w = 0;
            else if (name == "eg") x.lda_beta = 0, x.lda_w = 0, x.mbr_sent_w = 0;
            else if (name == "topic") x.eg_w = 0, x.mbr_sent_w = 0;
            for (int i = 0; i < n; ++i) v.emplace_back(name, x);
            a = e + 1;
        }
        return v;
    }
    std::string generate_pool(std::string_view prompt, std::uint64_t seed, const Options& o, GenStats* st, std::size_t* chosen = nullptr) const {
        const auto vs = pool_variants(o);
        std::vector<std::string> drafts(vs.size());
        std::vector<GenStats> gs(vs.size());
        for (std::size_t i = 0; i < vs.size(); ++i)
            drafts[i] = generate(prompt, mix64(seed ^ mix64(0xD0C5000000ull + i)), vs[i].second, &gs[i]);
        const std::size_t b = mbr::select(drafts);
        if (st) st->add(gs[b]), ++st->pool_chosen[vs[b].first];
        if (chosen) *chosen = b;
        return drafts[b];
    }

 private:
    struct Scratch {
        std::vector<double> p, pre, tmp, rw;
        std::vector<std::uint32_t> top, cand, kept;
        std::vector<std::string> pieces;
        std::vector<char> rb, cb;
        // OPT_NUCLEUS.md: fast_pick's own (check mode runs both picks on one scratch).
        struct Item {
            std::uint64_t pb;  // p's bits (p > 0, so they order as p)
            std::uint32_t x;
        };
        std::vector<double> q, qs, gs, pc, fp, frw, hm;
        std::vector<std::uint32_t> stamp, pos, seen, ptr, fcand, fkept, hc, boff;
        std::vector<Item> items, sorted;
        std::uint32_t cur = 0;
        std::vector<char> frb, fcb;
        RepeatBan::Batch rbatch;
        std::size_t k = 0, fk = 0;  // the nucleus' size, each pick's
    };
    // OPT_NUCLEUS.md: each word's two pieces in one buffer (bare: at the start or after ( [ { $ or a newline; spaced:
    // after any other word, as piece() writes them).
    struct FastTables {
        std::once_flag once;
        bool ok = true;  // false if a word is too long for meta (then fast_pick is not used)
        std::string bytes;
        // word w: its spaced piece at bytes[off, off + len + sp), its bare piece at bytes[off + sp, off + len + sp), and
        // the bare piece's first two bytes c0, c1 (0 past its end); packed as off | len << 32 | sp << 46 | c0 << 47 | c1 << 55.
        std::vector<std::uint64_t> meta;
        static std::size_t length(std::uint64_t m, bool bare) { return ((m >> 32) & 0x3FFF) + (bare ? 0 : (m >> 46) & 1); }
        // The piece's first two bytes.
        static unsigned char byte0(std::uint64_t m, bool bare) { return bare || !((m >> 46) & 1) ? static_cast<unsigned char>(m >> 47) : ' '; }
        static unsigned char byte1(std::uint64_t m, bool bare) {
            return bare || !((m >> 46) & 1) ? static_cast<unsigned char>(m >> 55) : static_cast<unsigned char>(m >> 47);
        }
        std::string_view piece(std::uint32_t w, bool bare) const {
            const std::uint64_t m = meta[w];
            const std::size_t a = static_cast<std::uint32_t>(m) + (bare ? (m >> 46) & 1 : 0);
            return std::string_view(bytes.data() + a, length(m, bare));
        }
    };
    const FastTables& fast_tables() const {
        std::call_once(fast_->once, [&] {
            FastTables& f = *fast_;
            const std::size_t V = vocab_.size();
            f.meta.resize(V);
            for (std::size_t w = 0; w < V; ++w) {
                const std::string& t = vocab_[w];
                const std::string sp = piece(t, "x", true);  // " " + the word, or the word
                const std::uint64_t n = t.size(), s = sp.size() - n;
                const std::uint64_t c0 = n > 0 ? static_cast<unsigned char>(t[0]) : 0, c1 = n > 1 ? static_cast<unsigned char>(t[1]) : 0;
                if (n >= 0x4000 || f.bytes.size() + sp.size() >= (std::uint64_t{1} << 32)) f.ok = false;
                f.meta[w] = f.bytes.size() | n << 32 | s << 46 | c0 << 47 | c1 << 55;
                f.bytes += sp;
            }
        });
        return *fast_;
    }
    // OPT_NUCLEUS.md: CYPHA_K5_NUCLEUS=full draws every word by full_pick; =check draws by both and stops at the
    // first difference. Read once.
    static int nucleus_mode() {
        static const int m = [] {
            const char* e = std::getenv("CYPHA_K5_NUCLEUS");
            const std::string v = e ? e : "";
            return v == "full" ? 1 : v == "check" ? 2 : 0;
        }();
        return m;
    }
    // K5_DISCOURSE.md: the sentence gate with z-scored terms. K candidates per sentence from the same state and the same
    // streams as the default gate; each term is z-scored over the K candidates, then
    //   score = wf z(fluency) + wp z(pmi) + mbr_sent_w z(mbr) + eg_w z(entity grid) + lda_w z(topic) - wm markup.
    void gate_z(std::string_view prompt, std::uint64_t seed, const Options& o, State& s, Scratch& sc, GenStats& g) const {
        const std::vector<double> assoc = o.gate_wp != 0 ? prompt_assoc(s.ctx) : std::vector<double>{};
        const std::vector<double> tlr = o.lda_w != 0 && lda_.trained() ? topic_lr(prompt) : std::vector<double>{};
        const bool use_eg = o.eg_w != 0 && eg_.trained();
        eg::EntityGrid::History hist;
        if (use_eg)
            for (const auto& sent : eg::split_sentences(wkn::tokens(prompt))) eg_.append(hist, eg_.roles(sent));
        const auto K = static_cast<std::size_t>(o.gate_k);
        std::vector<State> cand(K);
        std::vector<GenStats> gk(K);
        std::vector<std::string> text(K);
        std::vector<eg::SentRoles> roles(K);
        std::vector<double> tf(K), tp(K), tm(K), te(K), tt(K), mk(K);
        const auto z = [](std::vector<double>& x) {
            double m = 0, v = 0;
            for (const double a : x) m += a;
            m /= static_cast<double>(x.size());
            for (const double a : x) v += (a - m) * (a - m);
            const double sd = std::sqrt(v / static_cast<double>(x.size()));
            for (double& a : x) a = sd > 1e-12 ? (a - m) / sd : 0.0;
        };
        for (std::uint64_t sent = 0; s.step < o.max_words && s.out.size() < o.bytes; ++sent) {
            for (std::size_t k = 0; k < K; ++k) {
                State& c = cand[k];
                c = s;
                gk[k] = GenStats{};
                const std::uint64_t cs = mix64(seed ^ mix64(0x6A7E000000ull + sent * 256 + static_cast<std::uint64_t>(k)));
                std::mt19937_64 rng(cs);
                const std::size_t b0 = c.out.size();
                double slp = 0, spmi = 0, stop = 0;
                std::size_t nw = 0;
                std::vector<std::string> toks;
                while (c.step < o.max_words && c.out.size() < o.bytes) {
                    const std::uint64_t salt = mix64(cs ^ mix64(static_cast<std::uint64_t>(c.step) + 0x5EED));
                    double lp = 0;
                    const std::uint32_t w = step_word(c, rng, salt, o, gk[k], nullptr, sc, &lp);
                    slp += lp, spmi += assoc.empty() ? 0.0 : assoc[w], stop += tlr.empty() ? 0.0 : tlr[w], ++nw;
                    const std::string& t = vocab_[w];
                    if (use_eg) toks.push_back(t);
                    const std::size_t nb = c.out.size() - b0;
                    if (nb >= o.gate_max) break;
                    if (nb >= o.gate_min && (t == "." || t == "!" || t == "?" || t == "\n")) break;
                }
                std::size_t m = 0;
                for (std::size_t i = b0; i < c.out.size(); ++i) m += std::string_view("[]{}|=<>&*#").find(c.out[i]) != std::string_view::npos;
                const double nb = static_cast<double>(std::max<std::size_t>(c.out.size() - b0, 1)), n = static_cast<double>(std::max<std::size_t>(nw, 1));
                tf[k] = slp / n, tp[k] = spmi / n, tt[k] = stop / n, mk[k] = static_cast<double>(m) / nb;
                text[k] = c.out.substr(b0);
                if (use_eg) {
                    roles[k] = eg_.roles(toks);
                    te[k] = o.eg_mode == 2 ? eg_.continuation_score(hist, roles[k]) : eg_.append_logp(hist, roles[k]);
                }
                ++g.gate_cands;
            }
            if (o.mbr_sent_w != 0) tm = mbr::utilities(text);
            z(tf), z(tp), z(tm), z(te), z(tt);
            std::size_t best = 0;
            double bscore = 0;
            for (std::size_t k = 0; k < K; ++k) {
                const double score = o.gate_wf * tf[k] + o.gate_wp * tp[k] + o.mbr_sent_w * tm[k] + o.eg_w * te[k] + o.lda_w * tt[k] - o.gate_wm * mk[k];
                if (k == 0 || score > bscore) bscore = score, best = k;
            }
            if (use_eg) eg_.append(hist, roles[best]);
            s = std::move(cand[best]);
            g.add(gk[best]);
            ++g.gate_sents;
        }
    }

    // One step of generate_one on s: draws a word, writes its piece and learns it. lp (if not null): log2 of the
    // word's probability under the model before temperature, <unk> and <s> removed (the gate's fluency).
    std::uint32_t step_word(State& s, std::mt19937_64& rng, std::uint64_t salt, const Options& o, GenStats& g, ByteScorer* bs,
                            Scratch& sc, double* lp) const {
        ++g.steps;
        ++s.step;
        std::uint32_t w = 0;
        std::string str;
        const int mode = nucleus_mode();
        if (mode == 2 && fast_applies(s, o)) {
            check_pick(s, rng, salt, o, g, bs, sc, lp, w, str);
        } else if (mode == 1 || !fast_applies(s, o) || !fast_pick(s, rng, salt, o, g, bs, sc, lp, w, str)) {
            full_pick(s, rng, salt, o, g, bs, sc, lp, w, str);
        }
        s.out += str;
        s.rep.add(str);
        if (bs) bs->advance(str);
        learn(s, w, o);
        s.prev = vocab_[w];
        s.has_prev = true;
        return w;
    }
    // step_word's choice, as generate_one makes it: the whole distribution, sorted. w gets the word, str its piece.
    void full_pick(const State& s, std::mt19937_64& rng, std::uint64_t salt, const Options& o, GenStats& g, ByteScorer* bs,
                   Scratch& sc, double* lp, std::uint32_t& w, std::string& str) const {
        const std::size_t V = vocab_.size();
        std::vector<double>& p = sc.p;
        std::vector<std::uint32_t>&top = sc.top, &cand = sc.cand, &kept = sc.kept;
        p.resize(V);
        model_dist(s, o, p.data(), sc.tmp);
        if (lp && o.temperature != 1.0) sc.pre = p;
        if (o.temperature != 1.0)
            for (double& x : p) x = std::pow(x, 1.0 / o.temperature);
        p[kUnk] = p[kBos] = 0;
        double z = 0;
        for (const double x : p) z += x;
        for (double& x : p) x /= z;
        // Words of equal probability (common in the tail: the same continuation count, no stored context) are ordered
        // by a hash salted per step, not by id: ids follow frequency, and the most frequent of a tied tail block are words
        // that follow one word only ("Angeles", "Shrugged", and sentence openers such as "Instead" and "Moreover", which
        // only follow a full stop), which an id order would let in whenever a cut splits the block. numpy's argpartition
        // and argsort leave ties in no particular order.
        const auto above = [&](std::uint32_t x, std::uint32_t y) {
            if (p[x] != p[y]) return p[x] > p[y];
            const std::uint64_t hx = mix64(x ^ salt), hy = mix64(y ^ salt);
            return hx != hy ? hx < hy : x < y;
        };
        // Candidates: the cand_max most probable (ties by the salted hash), cut where their cumulative mass reaches
        // cand_mass.
        // tau, the least p among m words of positive p (the first m by p1), is at most the m-th largest p, so the
        // words with p >= tau hold the m most probable (for any distribution; tight when only stored contexts differ
        // from the order-1 ranking).
        const std::size_t m = std::min(o.cand_max, V - 2);
        double tau = 1;
        for (std::size_t i = 0, got = 0; got < m; ++i) {
            const std::uint32_t x = kn_.by_p1()[i];
            if (p[x] > 0) tau = std::min(tau, p[x]), ++got;
            if (i + 1 == V) {
                tau = 0;
                break;
            }
        }
        top.clear();
        for (std::uint32_t x = 0; x < V; ++x)
            if (p[x] >= tau) top.push_back(x);
        const std::size_t mm = std::min(m, top.size());
        std::partial_sort(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(mm), top.end(), above);
        cand.clear();
        double cum = 0;
        for (std::size_t i = 0; i < mm; ++i) {
            cand.push_back(top[i]);
            cum += p[top[i]];
            if (cum >= o.cand_mass) break;
        }
        // Bans.
        const std::size_t nc = cand.size();
        sc.pieces.resize(nc);
        sc.rb.assign(nc, 0);
        sc.cb.assign(nc, 0);
        std::vector<std::string>& pieces = sc.pieces;
        std::vector<char>&rb = sc.rb, &cb = sc.cb;
        const NgramSet cs = cap_set();
        NgramSet::Index::Range hp;
        const bool capped = o.cap && cs.prefix(s.ctx, hp);
        std::size_t ncap = 0;
        bool any_kept = false, any_cap_ok = false;
        for (std::size_t i = 0; i < nc; ++i) {
            pieces[i] = piece(vocab_[cand[i]], s.prev, s.has_prev);
            cb[i] = capped && cs.completes(hp, cand[i]);
            rb[i] = s.rep.banned(pieces[i]);
            ncap += cb[i];
            g.rep_banned += rb[i] && !cb[i];
            any_cap_ok = any_cap_ok || !cb[i];
            any_kept = any_kept || (!rb[i] && !cb[i]);
        }
        g.cap_banned += ncap;
        g.cap_steps += ncap > 0;
        w = 0;
        str.clear();
        sc.k = 0;
        if (!any_cap_ok) {  // the cap removed every candidate: the most probable word anywhere that it allows
            ++g.cap_outside;
            top.resize(V);
            for (std::uint32_t x = 0; x < V; ++x) top[x] = x;
            std::sort(top.begin(), top.end(), above);
            for (const std::uint32_t x : top)
                if (p[x] > 0 && !cs.completes(hp, x)) {
                    w = x;
                    break;
                }
            str = piece(vocab_[w], s.prev, s.has_prev);
        } else {
            if (!any_kept) {
                ++g.rep_lifted;
                std::fill(rb.begin(), rb.end(), 0);
            }
            kept.clear();
            double zk = 0;
            for (std::size_t i = 0; i < nc; ++i)
                if (!rb[i] && !cb[i]) kept.push_back(static_cast<std::uint32_t>(i)), zk += p[cand[i]];
            // Nucleus over the kept words, then one draw (numpy's choice: the first cdf entry above u).
            std::size_t k = 0;
            double c2 = 0;
            while (k < kept.size()) {
                c2 += p[cand[kept[k]]] / zk;
                ++k;
                if (c2 >= o.nucleus) break;
            }
            if (ncap > 0) {  // was a word the cap removed inside the nucleus the draw would have had without it?
                double zu = 0, cu = 0;
                for (std::size_t i = 0; i < nc; ++i)
                    if (!rb[i]) zu += p[cand[i]];
                for (std::size_t i = 0; i < nc; ++i) {
                    if (rb[i]) continue;
                    if (cb[i]) {
                        ++g.cap_mass_steps;
                        break;
                    }
                    cu += p[cand[i]] / zu;
                    if (cu >= o.nucleus) break;
                }
            }
            double zn = 0;
            for (std::size_t i = 0; i < k; ++i) zn += p[cand[kept[i]]];
            // §13.24: the byte scorer reranks the words inside the nucleus.
            std::vector<double>& rw = sc.rw;
            rw.resize(k);
            for (std::size_t i = 0; i < k; ++i) rw[i] = p[cand[kept[i]]] / zn;
            if (bs && k > 1) {
                std::vector<double> lb(k);
                double mx = -1e300;
                for (std::size_t i = 0; i < k; ++i) {
                    lb[i] = bs->score(pieces[kept[i]]);
                    if (o.scorer_mode == 3) lb[i] -= bs->prior(pieces[kept[i]]);
                    mx = std::max(mx, lb[i]);
                }
                double zb = 0, zr = 0;
                for (std::size_t i = 0; i < k; ++i) zb += std::exp2(lb[i] - mx);
                for (std::size_t i = 0; i < k; ++i) {
                    const double pb = std::exp2(lb[i] - mx) / zb;
                    rw[i] = o.scorer_mode == 3 ? std::exp2(std::log2(std::max(rw[i], 1e-300)) + o.scorer_w * (lb[i] - mx))
                          : o.scorer_mode == 2 ? (1 - o.scorer_w) * rw[i] + o.scorer_w * pb
                                               : std::exp2((1 - o.scorer_w) * std::log2(std::max(rw[i], 1e-300)) +
                                                           o.scorer_w * std::log2(std::max(pb, 1e-300)));
                    zr += rw[i];
                }
                for (double& x : rw) x /= zr;
            }
            const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;
            std::size_t j = 0;
            double cdf = 0;
            for (; j + 1 < k; ++j) {
                cdf += rw[j];
                if (cdf > u) break;
            }
            w = cand[kept[j]];
            str = std::move(pieces[kept[j]]);
            sc.k = k;
        }
        if (lp) {
            if (o.temperature != 1.0) {
                const double zr = 1 - sc.pre[kUnk] - sc.pre[kBos];
                *lp = std::log2(std::max(sc.pre[w] / zr, 1e-300));
            } else {
                *lp = std::log2(std::max(p[w], 1e-300));
            }
        }
    }

    // docs/reports/OPT_NUCLEUS.md (GENERATION_PLAN §13.41, O6): the same choice as full_pick, without sorting the
    // vocabulary. It applies to K5's own walk (no dynamic counts, no cache, no class mix, no topic factor) at
    // temperature 1.
    bool fast_applies(const State& s, const Options& o) const {
        return fast_ && o.temperature == 1.0 && o.dyn <= 0 && o.cache_w <= 0 && o.cache_w2 <= 0 && !(cm_.on() && o.class_mode != 1) && !s.topic_mult;
    }
    // q[x] for x in [2, V): the floor times the walk's gammas g[0..ng), rounded after each product as dist rounds it.
    template <bool On, int NG>
    static void fill_unseen_n(double* q, std::size_t V, const double* pc, const std::uint32_t* cls, const double* pwc, double zc,
                              const double* p1, const double* g) {
        double gg[NG + 1] = {};
        for (int k = 0; k < NG; ++k) gg[k] = g[k];
        for (std::size_t x = 2; x < V; ++x) {
            double v;
            if constexpr (On) v = (pc[cls[x]] * pwc[x]) / zc;
            else v = p1[x];
            for (int k = 0; k < NG; ++k) v *= gg[k];
            q[x] = v;
        }
    }
    template <bool On>
    static void fill_unseen(double* q, std::size_t V, const double* pc, const std::uint32_t* cls, const double* pwc, double zc,
                            const double* p1, const double* g, std::size_t ng) {
        switch (ng) {
            case 0: return fill_unseen_n<On, 0>(q, V, pc, cls, pwc, zc, p1, g);
            case 1: return fill_unseen_n<On, 1>(q, V, pc, cls, pwc, zc, p1, g);
            case 2: return fill_unseen_n<On, 2>(q, V, pc, cls, pwc, zc, p1, g);
            case 3: return fill_unseen_n<On, 3>(q, V, pc, cls, pwc, zc, p1, g);
            case 4: return fill_unseen_n<On, 4>(q, V, pc, cls, pwc, zc, p1, g);
            case 5: return fill_unseen_n<On, 5>(q, V, pc, cls, pwc, zc, p1, g);
            case 6: return fill_unseen_n<On, 6>(q, V, pc, cls, pwc, zc, p1, g);
            default:
                for (std::size_t x = 2; x < V; ++x) {
                    double v;
                    if constexpr (On) v = (pc[cls[x]] * pwc[x]) / zc;
                    else v = p1[x];
                    for (std::size_t k = 0; k < ng; ++k) v *= g[k];
                    q[x] = v;
                }
        }
    }
    // The maths. After the walk's contexts (orders 2 to k*, gammas g2..gk*), a word w that none of them holds has
    // q(w) = (((b(w) g2) g3) ...) gk*, b(w) the floor (the class floor (pc[c(w)] P(w | c(w))) / zc, or order 1), each
    // product rounded as dist rounds it. The words a context holds (the seen words) get the walk's sums, word by word,
    // in dist's order. Then:
    //   1. z, the sum of q over the vocabulary in id order (<unk> and <s> as 0), as step_word sums it; p(w) = q(w) / z.
    //      This sum needs every word once, in order: it is the one pass over the vocabulary that stays. Beside it, the
    //      mass and number of words in each band of q (its top 14 bits: exponent and 2 bits of mantissa).
    //   2. the candidates are a prefix of the vocabulary in full_pick's order (p descending, then the salted hash; the
    //      hash is a bijection, so it never ties). From the top band down, the first band b where the mass reaches
    //      cand_mass z (with a margin of 1e-9 for the order of the sums) or the count reaches cand_max gives a bound
    //      t = the least q in b. Every word with q >= t (1 - 2^-40) is taken; a word below that has p < fl(t / z), and
    //      every word with p >= fl(t / z) is taken. Those are sorted (by bands of p, then exactly), and the cut is made
    //      among them as full_pick makes it: cum += p, word by word, until cum >= cand_mass or cand_max words;
    //   3. the bans, the nucleus and the draw exactly as full_pick, on the same candidates in the same order.
    // Returns false, having changed nothing, when the cut is not reached above fl(t / z), the bound would take a word
    // of q 0, or the cap leaves no candidate; full_pick then makes the choice.
    bool fast_pick(const State& s, std::mt19937_64& rng, std::uint64_t salt, const Options& o, GenStats& g, ByteScorer* bs,
                   Scratch& sc, double* lp, std::uint32_t& w, std::string& str) const {
        const std::size_t V = vocab_.size();
        const FastTables& ft = fast_tables();
        if (!ft.ok) return false;
        // The floor.
        const bool on = cm_.on();
        double zc = 1;
        if (on) {
            sc.pc.resize(cm_.num_classes());
            zc = cm_.floor_parts(s.ctx.data(), s.ctx.size(), sc.pc.data());
        }
        const double* pc = sc.pc.data();
        const std::uint32_t* cls = on ? cm_.classes().data() : nullptr;
        const double* pwc = on ? cm_.pwc().data() : nullptr;
        const double* p1 = kn_.p1().data();
        const auto base = [&](std::uint32_t x) { return on ? (pc[cls[x]] * pwc[x]) / zc : p1[x]; };
        // The walk: the seen words' values, order by order, as dist applies them.
        std::vector<double>&q = sc.q, &qs = sc.qs, &gs = sc.gs;
        std::vector<std::uint32_t>&stamp = sc.stamp, &pos = sc.pos, &seen = sc.seen;
        q.resize(V);
        stamp.resize(V, 0);
        pos.resize(V);
        if (++sc.cur == 0) std::fill(stamp.begin(), stamp.end(), 0), sc.cur = 1;
        const std::uint32_t cur = sc.cur;
        seen.clear(), qs.clear(), gs.clear();
        kn_.walk(s.ctx.data(), s.ctx.size(), [&](double gam, const std::uint16_t* ids, const double* disc, std::size_t m) {
            if (gs.empty()) {  // the first order: every entry is new (a context's entries are distinct)
                gs.push_back(gam);
                seen.resize(m), qs.resize(m);
                for (std::size_t e = 0; e < m; ++e) {
                    const std::uint32_t x = ids[e];
                    stamp[x] = cur;
                    pos[x] = static_cast<std::uint32_t>(e);
                    seen[e] = x;
                    qs[e] = base(x) * gam;
                }
                for (std::size_t e = 0; e < m; ++e) qs[e] += disc[e];  // a second loop, as dist adds (no fused multiply-add)
                return;
            }
            for (double& v : qs) v *= gam;
            gs.push_back(gam);
            for (std::size_t e = 0; e < m; ++e) {
                const std::uint32_t x = ids[e];
                if (stamp[x] != cur) {
                    stamp[x] = cur;
                    pos[x] = static_cast<std::uint32_t>(seen.size());
                    seen.push_back(x);
                    double v = base(x);
                    for (const double gk : gs) v *= gk;
                    qs.push_back(v);
                }
                qs[pos[x]] += disc[e];
            }
        });
        // 1. q over the vocabulary and z, in id order: every word's chain (vectorised; each lane rounds as dist
        // does), the seen words' values over them, then the sum, word by word.
        double* qq = q.data();
        if (on) fill_unseen<true>(qq, V, pc, cls, pwc, zc, p1, gs.data(), gs.size());
        else fill_unseen<false>(qq, V, pc, cls, pwc, zc, p1, gs.data(), gs.size());
        for (std::size_t i = 0; i < seen.size(); ++i) qq[seen[i]] = qs[i];
        qq[kUnk] = qq[kBos] = 0;
        constexpr int kBandShift = 50;  // a band: the top 14 bits of a q >= 0
        constexpr std::size_t kBands = std::size_t{1} << (63 - kBandShift);
        std::vector<double>& hm = sc.hm;
        std::vector<std::uint32_t>& hc = sc.hc;
        hm.assign(kBands, 0.0);
        hc.assign(kBands, 0);
        double z = 0;
        for (std::size_t x = 2; x < V; ++x) {
            const double v = qq[x];
            z += v;
            const std::size_t b = static_cast<std::size_t>(std::bit_cast<std::uint64_t>(v) >> kBandShift);
            hm[b] += v;
            ++hc[b];
        }
        // 2. The candidates: the bound, the words above it, sorted, and the cut.
        const std::size_t mc = std::min(o.cand_max, V - 2);
        const double need = o.cand_mass * z * (1 + 1e-9);
        std::size_t b = kBands;
        {
            double acc = 0;
            std::size_t cnt = 0;
            while (b-- > 0) {
                acc += hm[b], cnt += hc[b];
                if (acc >= need || cnt >= mc) break;
            }
        }
        if (b == 0 || b >= kBands) return false;  // the bound would take words of q 0 (or every word): full_pick
        const double t = std::bit_cast<double>(static_cast<std::uint64_t>(b) << kBandShift);
        const double tb = t * (1 - 0x1p-40), plim = t / z;
        std::vector<std::uint32_t>& tk = sc.ptr;  // the words taken
        tk.resize(V);
        std::size_t nt = 0;
        for (std::uint32_t x = 2; x < V; ++x) {
            tk[nt] = x;
            nt += qq[x] >= tb;
        }
        // Sort: by bands of p (its top 20 bits), descending, then within a band by p descending and the hash.
        using Item = Scratch::Item;
        std::vector<Item>& it = sc.items;
        it.clear();
        for (std::size_t i = 0; i < nt; ++i) {
            const std::uint32_t x = tk[i];
            const double p = qq[x] / z;
            if (p >= plim) it.push_back({std::bit_cast<std::uint64_t>(p), x});
        }
        constexpr int kSortShift = 44;
        const std::uint64_t top = std::bit_cast<std::uint64_t>(1.0) >> kSortShift, lo = std::bit_cast<std::uint64_t>(plim) >> kSortShift;
        const auto before = [salt](const Item& a, const Item& c) {
            return a.pb != c.pb ? a.pb > c.pb : mix64(a.x ^ salt) < mix64(c.x ^ salt);
        };
        std::vector<Item>& srt = sc.sorted;
        if (top >= lo && top - lo < 16384) {
            const std::size_t nbk = static_cast<std::size_t>(top - lo) + 1;
            std::vector<std::uint32_t>& off = sc.boff;
            off.assign(nbk + 1, 0);
            const auto bk = [&](const Item& a) { return static_cast<std::size_t>(top - std::min(top, a.pb >> kSortShift)); };
            for (const Item& a : it) ++off[bk(a) + 1];
            for (std::size_t i = 0; i < nbk; ++i) off[i + 1] += off[i];
            srt.resize(it.size());
            for (const Item& a : it) srt[off[bk(a)]++] = a;
            for (std::size_t i = 0, a = 0; i < nbk; ++i) {  // off[i] is now band i's end
                const std::size_t e = off[i];
                if (e - a > 32) {
                    std::sort(srt.begin() + static_cast<std::ptrdiff_t>(a), srt.begin() + static_cast<std::ptrdiff_t>(e), before);
                } else {
                    for (std::size_t j = a + 1; j < e; ++j)  // insertion sort
                        for (std::size_t k = j; k > a && before(srt[k], srt[k - 1]); --k) std::swap(srt[k], srt[k - 1]);
                }
                a = e;
            }
        } else {
            srt = it;
            std::sort(srt.begin(), srt.end(), before);
        }
        std::vector<std::uint32_t>& cand = sc.fcand;
        std::vector<double>& cp = sc.fp;
        cand.clear(), cp.clear();
        double cum = 0;
        bool cut = false;
        for (const Item& a : srt) {
            const double p = std::bit_cast<double>(a.pb);
            cand.push_back(a.x);
            cp.push_back(p);
            cum += p;
            if (cum >= o.cand_mass || cand.size() == mc) {
                cut = true;
                break;
            }
        }
        if (!cut) return false;  // the bound was too high (not seen in practice): full_pick
        // 3. Bans, as full_pick.
        const std::size_t nc = cand.size();
        std::vector<char>&rb = sc.frb, &cb = sc.fcb;
        rb.assign(nc, 0);
        cb.assign(nc, 0);
        const std::string& pv = s.prev;
        const bool after = !s.has_prev || pv == "\n" || pv == "(" || pv == "[" || pv == "{" || pv == "$";  // bare, as piece()
        const auto pieces = [&](std::size_t i) { return ft.piece(cand[i], after); };
        s.rep.prepare(sc.rbatch);
        const NgramSet cs = cap_set();
        NgramSet::Index::Range hp;
        const bool capped = o.cap && cs.prefix(s.ctx, hp);
        std::size_t ncap = 0, nrep = 0;
        bool any_kept = false, any_cap_ok = false;
        const std::uint64_t* meta = ft.meta.data();
        const auto rep_banned = [&](std::size_t i) {
            const std::uint64_t md = meta[cand[i]];
            return RepeatBan::may_ban(sc.rbatch, FastTables::length(md, after), FastTables::byte0(md, after), FastTables::byte1(md, after)) &&
                   s.rep.banned(sc.rbatch, pieces(i));
        };
        if (!capped) {  // the usual case: no word completes a copied run
            for (std::size_t i = 0; i < nc; ++i) {
                if (i + 16 < nc) __builtin_prefetch(meta + cand[i + 16]);
                rb[i] = rep_banned(i);
                nrep += static_cast<unsigned char>(rb[i]);
            }
            any_cap_ok = nc > 0;
            any_kept = nrep < nc;
        } else {
            for (std::size_t i = 0; i < nc; ++i) {
                cb[i] = cs.completes(hp, cand[i]);
                rb[i] = rep_banned(i);
                ncap += cb[i];
                nrep += rb[i] && !cb[i];
                any_cap_ok = any_cap_ok || !cb[i];
                any_kept = any_kept || (!rb[i] && !cb[i]);
            }
        }
        if (!any_cap_ok) return false;  // the cap left no candidate: full_pick looks outside them
        g.rep_banned += nrep;
        g.cap_banned += ncap;
        g.cap_steps += ncap > 0;
        if (!any_kept) {
            ++g.rep_lifted;
            std::fill(rb.begin(), rb.end(), 0);
        }
        std::vector<std::uint32_t>& kept = sc.fkept;
        kept.clear();
        double zk = 0;
        for (std::size_t i = 0; i < nc; ++i)
            if (!rb[i] && !cb[i]) kept.push_back(static_cast<std::uint32_t>(i)), zk += cp[i];
        std::size_t k = 0;
        double c2 = 0;
        while (k < kept.size()) {
            c2 += cp[kept[k]] / zk;
            ++k;
            if (c2 >= o.nucleus) break;
        }
        if (ncap > 0) {
            double zu = 0, cu = 0;
            for (std::size_t i = 0; i < nc; ++i)
                if (!rb[i]) zu += cp[i];
            for (std::size_t i = 0; i < nc; ++i) {
                if (rb[i]) continue;
                if (cb[i]) {
                    ++g.cap_mass_steps;
                    break;
                }
                cu += cp[i] / zu;
                if (cu >= o.nucleus) break;
            }
        }
        double zn = 0;
        for (std::size_t i = 0; i < k; ++i) zn += cp[kept[i]];
        std::vector<double>& rw = sc.frw;
        rw.resize(k);
        for (std::size_t i = 0; i < k; ++i) rw[i] = cp[kept[i]] / zn;
        if (bs && k > 1) {
            std::vector<double> lb(k);
            double mx = -1e300;
            for (std::size_t i = 0; i < k; ++i) {
                lb[i] = bs->score(pieces(kept[i]));
                if (o.scorer_mode == 3) lb[i] -= bs->prior(pieces(kept[i]));
                mx = std::max(mx, lb[i]);
            }
            double zb = 0, zr = 0;
            for (std::size_t i = 0; i < k; ++i) zb += std::exp2(lb[i] - mx);
            for (std::size_t i = 0; i < k; ++i) {
                const double pb = std::exp2(lb[i] - mx) / zb;
                rw[i] = o.scorer_mode == 3 ? std::exp2(std::log2(std::max(rw[i], 1e-300)) + o.scorer_w * (lb[i] - mx))
                      : o.scorer_mode == 2 ? (1 - o.scorer_w) * rw[i] + o.scorer_w * pb
                                           : std::exp2((1 - o.scorer_w) * std::log2(std::max(rw[i], 1e-300)) +
                                                       o.scorer_w * std::log2(std::max(pb, 1e-300)));
                zr += rw[i];
            }
            for (double& x : rw) x /= zr;
        }
        const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;
        std::size_t j = 0;
        double cdf = 0;
        for (; j + 1 < k; ++j) {
            cdf += rw[j];
            if (cdf > u) break;
        }
        w = cand[kept[j]];
        str.assign(pieces(kept[j]));
        sc.fk = k;
        if (lp) *lp = std::log2(std::max(cp[kept[j]], 1e-300));
        return true;
    }
    // CYPHA_K5_NUCLEUS=check: both picks from the same state and the same stream; any difference in the candidates
    // (words, order, p bit for bit), the nucleus, the word, its piece, lp or the counts stops the run.
    void check_pick(const State& s, std::mt19937_64& rng, std::uint64_t salt, const Options& o, GenStats& g, ByteScorer* bs,
                    Scratch& sc, double* lp, std::uint32_t& w, std::string& str) const {
        bool fast = false;
        if (!both_picks(s, rng, salt, o, g, bs, sc, lp, w, str, fast)) {
            std::fprintf(stderr, "k5 nucleus check: the picks differ at step %d (word %u, nucleus %zu / %zu, candidates %zu / %zu)\n",
                         s.step, w, sc.k, sc.fk, sc.cand.size(), sc.fcand.size());
            std::abort();
        }
        thread_local std::size_t n_fast = 0, n_all = 0;
        ++n_all, n_fast += fast;
        if ((n_all & 0xFFFF) == 0) std::fprintf(stderr, "k5 nucleus check: %zu steps the same (%zu by both picks)\n", n_all, n_fast);
    }
    // Both picks from s with the same stream; w, str, lp, g and rng as full_pick leaves them. fast: fast_pick made its
    // choice. False when it differs from full_pick's.
    bool both_picks(const State& s, std::mt19937_64& rng, std::uint64_t salt, const Options& o, GenStats& g, ByteScorer* bs,
                    Scratch& sc, double* lp, std::uint32_t& w, std::string& str, bool& fast) const {
        std::mt19937_64 r2 = rng;
        GenStats g1, g2;
        double l1 = 0, l2 = 0;
        std::uint32_t w2 = 0;
        std::string s2;
        full_pick(s, rng, salt, o, g1, bs, sc, lp ? &l1 : nullptr, w, str);
        fast = fast_pick(s, r2, salt, o, g2, bs, sc, lp ? &l2 : nullptr, w2, s2);
        bool same = true;
        if (fast) {
            same = w == w2 && str == s2 && rng == r2 && sc.k == sc.fk && sc.cand.size() == sc.fcand.size() &&
                   std::memcmp(&l1, &l2, sizeof l1) == 0 && g1.rep_banned == g2.rep_banned && g1.cap_banned == g2.cap_banned &&
                   g1.cap_steps == g2.cap_steps && g1.cap_mass_steps == g2.cap_mass_steps && g1.rep_lifted == g2.rep_lifted &&
                   g1.cap_outside == g2.cap_outside;
            for (std::size_t i = 0; same && i < sc.cand.size(); ++i)
                same = sc.cand[i] == sc.fcand[i] && std::memcmp(&sc.p[sc.cand[i]], &sc.fp[i], sizeof(double)) == 0 &&
                       sc.rb[i] == sc.frb[i] && sc.cb[i] == sc.fcb[i];
        }
        g.add(g1);
        if (lp) *lp = l1;
        return same;
    }

 public:
    // OPT_NUCLEUS.md: writes up to steps words after prompt, each chosen by full_pick and by fast_pick from the same
    // state and stream (the text follows full_pick). False, with err, at the first difference; both counts the steps at
    // which fast_pick made a choice, all the steps compared.
    bool check_nucleus(std::string_view prompt, std::uint64_t seed, const Options& o, int steps, std::size_t& both, std::size_t& all,
                       std::string& err) const {
        if (vocab_.size() <= 2) return true;
        State s = start(prompt, o);
        if (!fast_applies(s, o)) return true;
        Scratch sc;
        sc.p.resize(vocab_.size());
        std::mt19937_64 rng(seed);
        GenStats g;
        for (int t = 0; t < steps; ++t) {
            const std::uint64_t salt = mix64(seed ^ mix64(static_cast<std::uint64_t>(s.step) + 0x5EED));
            ++s.step;
            std::uint32_t w = 0;
            std::string str;
            double lp = 0;
            bool fast = false;
            if (!both_picks(s, rng, salt, o, g, nullptr, sc, &lp, w, str, fast)) {
                err = "nucleus: fast_pick differs from full_pick after \"" + std::string(prompt) + "\", step " + std::to_string(t);
                return false;
            }
            both += fast, ++all;
            s.out += str;
            s.rep.add(str);
            learn(s, w, o);
            s.prev = vocab_[w];
            s.has_prev = true;
        }
        return true;
    }

 private:

    std::vector<std::string> vocab_;
    std::unordered_map<std::string, std::uint32_t> tid_;
    WordKN kn_;
    ClassModel cm_;
    eg::EntityGrid eg_;        // K5_DISCOURSE.md
    lda::LDA lda_;             // K5_DISCOURSE.md
    std::vector<long> k2l_;    // K5 id -> LDA content-word id (-1: none)
    // fuzzy_bake.py has_run's 13-token windows (the verbatim measure), a view on the suffix array.
    NgramSet run13() const { return NgramSet(&kn_.index(), 13, hs13_.get()); }
    std::unique_ptr<NgramSet::Hashes> hs13_, hscap_;  // the hash calls' windows, made on first use
    std::unique_ptr<FastTables> fast_;                // OPT_NUCLEUS.md: fast_pick's tables, made on first use
    std::vector<double> cnt_;
    int cap_ = 12;
    std::size_t docs_ = 0, tokens_ = 0, unk_ = 0;
    double secs_ = 0;
};

// ---------------------------------------------------------------- K5_DISCOURSE.md: the helpers' stable scoring calls

// log2 score of candidate (text) given context (prompt + paragraph so far, text) under the LDA topic model: the sum
// over the candidate's content words of log2 (p_topic(w | context's topics) / p(w)). See lda.hpp.
inline double lda_score(const lda::LDA& m, std::string_view context, std::string_view candidate) {
    return m.score(tokens(context), tokens(candidate));
}
// log2 score of candidate given context under the entity grid: the change in the grid's log2 probability when the
// candidate's sentences follow the context's (cont: kept entities only). See entity_grid.hpp.
inline double eg_score(const eg::EntityGrid& g, std::string_view context, std::string_view candidate, bool cont = false) {
    return g.score(tokens(context), tokens(candidate), cont);
}

// ---------------------------------------------------------------- cyphalm_trie sample --k5

// The share of the texts' bytes that lie in a 32-byte span of ref (fuzzy_bake.py copy_share; cyphalm_trie's Spans32).
inline double copy32_share(std::string_view ref, const std::vector<std::string>& texts) {
    constexpr std::size_t K = 32;
    const auto hash = [](const char* p) {
        std::uint64_t h = kFnv0;
        for (std::size_t i = 0; i < K; ++i) h = fnv_step(h, static_cast<unsigned char>(p[i]));
        return h;
    };
    std::vector<std::uint64_t> h;
    if (ref.size() >= K) h.reserve(ref.size() - K + 1);
    for (std::size_t i = 0; i + K <= ref.size(); ++i) h.push_back(hash(ref.data() + i));
    std::sort(h.begin(), h.end());
    h.erase(std::unique(h.begin(), h.end()), h.end());
    std::size_t cov = 0, all = 0;
    for (const auto& t : texts) {
        std::vector<char> m(t.size(), 0);
        for (std::size_t i = 0; i + K <= t.size(); ++i)
            if (std::binary_search(h.begin(), h.end(), hash(t.data() + i))) std::fill(m.begin() + static_cast<std::ptrdiff_t>(i), m.begin() + static_cast<std::ptrdiff_t>(i + K), 1);
        cov += static_cast<std::size_t>(std::count(m.begin(), m.end(), 1));
        all += t.size();
    }
    return all ? static_cast<double>(cov) / static_cast<double>(all) : 0.0;
}

// cyphalm_trie sample --k5 TEXT [--k5-cap N]: K5 writes every GEN line (GENERATION_PLAN §13.19). The prompts are
// sample's (--prompts-file, or the holdout's); prompt id = f * per_file + k, dealt to the out files in order. Each file
// holds a MODEL line, then PROMPT and GEN lines (GEN\ttrie_T1.0\t..., so scribe ladder --gen-trie reads it); esc is
// sample's printable(). Prints the speed and the cap's work to stderr, and GEN_EVAL lines to stdout: k5_cont_bpb
// (fuzzy_bake.py's cont bits per byte, when conts are given), copy32 against the training text, and verbatim13 (the
// share of samples holding 13 consecutive training tokens; fuzzy_bake.py's 13-word runs).
inline int run_sample(const std::string& text_path, int cap, const std::vector<std::string>& prompts,
                      const std::vector<std::string>& conts, const std::vector<std::string>& outs, int per_file,
                      std::size_t bytes, const std::vector<double>& temps,
                      const std::function<std::string(const std::string&)>& esc, const Options* base = nullptr,
                      std::uint64_t seed = kSeed, std::size_t vmax = kVmax) {  // K5_SCALE: --k5-seed, --k5-vmax
    std::string err;
    K5 m;
    const Options bo = base ? *base : Options{};
    if (bo.gate_k > 1 && bo.scorer) {
        std::cerr << "--k5-gate does not take --k5-lstm\n";
        return 2;
    }
    if (!m.load(text_path, cap, err, &bo, vmax)) {
        std::cerr << "--k5: " << err << "\n";
        return 1;
    }
    if (m.classes().on()) {
        std::fprintf(stderr, "k5: %zu exchange classes (%s); class-bigram loglik", bo.classes, bo.class_mode == 2 ? "mix" : "backoff");
        for (const double x : m.classes().loglik()) std::fprintf(stderr, " %.0f", x);
        std::fprintf(stderr, "\n");
    }
    if (!bo.describe().empty()) std::fprintf(stderr, "k5: options%s\n", bo.describe().c_str());
    std::fprintf(stderr, "k5: %zu docs, %zu tokens (unk %.4f), vocab %zu; entries o2 %zu o3 %zu o4 %zu o5 %zu; cap %d (%zu windows); "
                 "%.1f MB; built in %.1f s\n", m.docs(), m.tokens(), static_cast<double>(m.unk()) / static_cast<double>(std::max<std::size_t>(m.tokens(), 1)),
                 m.vocab().size(), m.kn().entries(2), m.kn().entries(3), m.kn().entries(4), m.kn().entries(5), cap,
                 m.cap_set().size(), static_cast<double>(m.bytes()) / 1e6, m.build_seconds());
    const std::size_t nt = temps.size(), nf = outs.size(), np = static_cast<std::size_t>(per_file) * nf;
    std::vector<std::string> gens(np * nt);
    std::vector<GenStats> gst(np * nt);
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<std::size_t> next{0};
    const auto work = [&]() {
        for (std::size_t j; (j = next++) < np * nt;) {
            const std::size_t id = j / nt, ti = j % nt;
            Options o = base ? *base : Options{};
            o.temperature = temps[ti];
            o.bytes = bytes;
            gens[j] = m.generate(prompts[id], seed + o.seed_off + id + 7919 * ti, o, &gst[j]);
            if (o.post) gens[j] = o.post(m, prompts[id], std::move(gens[j]), seed + o.seed_off + id + 7919 * ti, o);  // K5_MCMC.md
        }
    };
    unsigned nth = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    if (const char* e = std::getenv("K5_THREADS")) nth = std::max(1, std::atoi(e));  // K5_UPGRADES.md: the gate's runs
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < nth; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto label = [](double t) {
        char b[32];
        std::snprintf(b, sizeof b, std::fabs(t * 10.0 - std::round(t * 10.0)) < 1e-9 ? "%.1f" : "%.2f", t);
        return std::string(b);
    };
    for (std::size_t f = 0; f < nf; ++f) {
        std::ofstream out(outs[f]);
        if (!out) {
            std::cerr << "cannot write " << outs[f] << "\n";
            return 1;
        }
        out << "MODEL\tk5\tclean=" << text_path << "\tcap=" << cap << "\tvocab=" << m.vocab().size() << bo.describe() << bo.post_desc << "\n";
        for (int k = 0; k < per_file; ++k) {
            const std::size_t id = f * static_cast<std::size_t>(per_file) + static_cast<std::size_t>(k);
            out << "PROMPT\t" << id << "\t" << esc(prompts[id]) << "\n";
            for (std::size_t ti = 0; ti < nt; ++ti) out << "GEN\ttrie_T" << label(temps[ti]) << "\t" << esc(gens[id * nt + ti]) << "\n";
        }
    }
    GenStats all;
    std::size_t nbytes = 0;
    for (std::size_t j = 0; j < gens.size(); ++j) all.add(gst[j]), nbytes += gens[j].size();
    std::fprintf(stderr, "k5: generated %zu bytes in %.1f s on %u threads (%.1f us per byte, %.2f ms per word)\n", nbytes, secs, nth,
                 1e6 * secs / static_cast<double>(std::max<std::size_t>(nbytes, 1)), 1e3 * secs * nth / static_cast<double>(std::max<std::size_t>(all.steps, 1)));
    std::fprintf(stderr, "k5: %zu words; repeat ban removed %zu candidates, lifted at %zu steps; cap removed %zu candidates at %zu steps "
                 "(%zu inside the draw's nucleus), left none at %zu\n", all.steps, all.rep_banned, all.rep_lifted, all.cap_banned,
                 all.cap_steps, all.cap_mass_steps, all.cap_outside);
    if (all.gate_sents) std::fprintf(stderr, "k5: the gate kept %zu sentences of %zu candidates\n", all.gate_sents, all.gate_cands);
    if (!all.pool_chosen.empty()) {  // K5_DISCOURSE.md
        std::fprintf(stderr, "k5: MBR kept drafts of");
        for (const auto& [k, v] : all.pool_chosen) std::fprintf(stderr, " %s %zu", k.c_str(), v);
        std::fprintf(stderr, "\n");
    }
    if (!conts.empty()) {
        double b = 0, n = 0;
        for (std::size_t k = 0; k < np && k < conts.size(); ++k) {
            const auto [x, y] = m.cont_bits(prompts[k], conts[k], bo);
            b += x, n += y;
        }
        std::printf("GEN_EVAL k5_cont_bpb %.6f (word-level; an unknown word costs <unk>'s bits)\n", n > 0 ? b / n : 0.0);
    }
    std::ifstream in(text_path, std::ios::binary);
    const std::string ref((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::printf("GEN_EVAL copy32");
    for (std::size_t ti = 0; ti < nt; ++ti) {
        std::vector<std::string> g;
        std::size_t runs = 0;
        for (std::size_t id = 0; id < np; ++id) g.push_back(gens[id * nt + ti]), runs += m.has_run13(gens[id * nt + ti]);
        std::printf(" T%s %.4f verbatim13 %.4f (%zu of %zu)", label(temps[ti]).c_str(), copy32_share(ref, g),
                    static_cast<double>(runs) / static_cast<double>(std::max<std::size_t>(np, 1)), runs, np);
    }
    std::printf("\n");
    return 0;
}

// ---------------------------------------------------------------- selftest

// Tokens, pieces, the repeat ban, the cap, and KN: every distribution sums to one, and a few probabilities equal a
// direct count-based computation of the same estimate.
inline bool self_test(std::string& err) {
    const auto toks = [](std::string_view s) {
        std::string j;
        for (const auto& t : tokens(s)) j += "[" + t + "]";
        return j;
    };
    const std::pair<const char*, const char*> tk[] = {
        {"Don't stop\xE2\x80\x94it's 3.14, ok?\nNo.", "[Don't][stop][\xE2\x80\x94][it's][3.14][,][ok][?][\n][No][.]"},
        {"rock'n'roll a''b x' 1,000.5 2. .5", "[rock'n'roll][a]['][']" "[b][x]['][1,000.5][2][.][.][5]"},
        {"caf\xC3\xA9\xC2\xA0x\ty\r\x0Bz", "[caf][\xC3\xA9][x][y][z]"},
        {"\xE3\x80\x80(a)$5 _b", "[(][a][)][$][5][_][b]"},
    };
    for (const auto& [in, want] : tk)
        if (toks(in) != want) {
            err = "tokens of " + std::string(in) + ": " + toks(in);
            return false;
        }
    if (piece(".", "x") != "." || piece("x", "(") != "x" || piece("x", "y") != " x" || piece("\n", "y") != "\n" ||
        piece("x", "", false) != "x" || piece("x", "\n") != "x" || piece("'", "x") != "'" || piece("s", "'") != " s" ||
        piece("%", "5") != "%" || piece("5", "$") != "5") {
        err = "piece";
        return false;
    }
    {
        RepeatBan r("the quick brown fox. the quick brow");
        if (!r.banned("n") || r.banned("x") || r.banned(" ")) {
            err = "repeat ban";
            return false;
        }
        RepeatBan s("short");
        if (s.banned(" abcdefghijk abcdefghijkl") != true || s.banned(" words") != false) {
            err = "repeat ban inside a word";
            return false;
        }
    }
    // A toy corpus: sentences from a small grammar, a document per line, with one long sentence that repeats.
    std::string text;
    {
        std::mt19937_64 r(7);
        const char* subj[] = {"The cat", "A dog", "The old man", "Her brother", "The king of France", "It"};
        const char* verb[] = {"saw", "liked", "found", "wrote about", "did not see", "lost"};
        const char* obj[] = {"the river", "a small house", "three apples", "the 1990s", "Paris", "his own book"};
        const char* tail[] = {".", ", and left.", " (twice).", "; it rained.", "!", " in 1,200 days."};
        for (int d = 0; d < 60; ++d) {
            for (int s = 0; s < 8; ++s) {
                text += subj[r() % 6], text += " ", text += verb[r() % 6], text += " ", text += obj[r() % 6], text += tail[r() % 6];
                text += " ";
            }
            if (d % 7 == 0) text += "One two three four five six seven eight nine ten eleven twelve thirteen fourteen. ";
            text += std::string("Zq") + static_cast<char>('a' + d % 26) + static_cast<char>('a' + d / 26) + ".\n";  // seen once: <unk>
        }
    }
    K5 m;
    m.build(text, 12);
    const std::size_t V = m.vocab().size();
    if (m.docs() != 60 || V < 30 || m.id("Zqda") != kUnk || m.id("cat") == kUnk || m.vocab()[m.id("cat")] != "cat") {
        err = "toy vocabulary";
        return false;
    }
    std::vector<double> p(V);
    const char* ctxs[] = {"", "The", "The cat saw", "The cat saw the river", "Her brother lost Zqda", "never seen words here",
                          "One two three four five six seven eight nine ten eleven twelve"};
    for (const char* c : ctxs) {
        std::vector<std::uint32_t> ctx = {kBos};
        for (const auto& t : tokens(c)) ctx.push_back(m.id(t));
        m.kn().dist(ctx.data(), ctx.size(), p.data());
        double z = 0, lo = 1;
        for (const double x : p) z += x, lo = std::min(lo, x);
        if (std::fabs(z - 1) > 1e-12 || lo < 0) {  // a discount clipped to 0 can leave a word at 0, as in fuzzy_bake.py
            err = "KN does not sum to one after \"" + std::string(c) + "\"";
            return false;
        }
    }
    {  // Direct counts: P(saw | The cat) at order 3, from raw trigram-context continuation counts and the bigram level.
        std::vector<std::uint32_t> ids;
        for (std::size_t a = 0; a < text.size();) {
            std::size_t b = text.find('\n', a);
            ids.push_back(kBos);
            for (const auto& t : tokens(std::string_view(text).substr(a, b - a))) ids.push_back(m.id(t));
            a = b + 1;
        }
        // ctx = (cat) at order 2 and (The cat) at order 3; the order-3 context's continuation counts come from the
        // distinct 5-grams' (w-3 w-2 w-1 w) 4-gram types' left extensions, i.e. distinct w-3 before "The cat w".
        const std::uint32_t The = m.id("The"), cat = m.id("cat"), saw = m.id("saw");
        std::vector<std::array<std::uint32_t, 5>> five;
        for (std::size_t t = 4; t < ids.size(); ++t) five.push_back({ids[t - 4], ids[t - 3], ids[t - 2], ids[t - 1], ids[t]});
        std::sort(five.begin(), five.end());
        five.erase(std::unique(five.begin(), five.end()), five.end());
        std::vector<std::array<std::uint32_t, 4>> four;  // distinct (w-3 w-2 w-1 w)
        for (const auto& f : five) four.push_back({f[1], f[2], f[3], f[4]});
        std::sort(four.begin(), four.end());
        four.erase(std::unique(four.begin(), four.end()), four.end());
        std::vector<std::array<std::uint32_t, 3>> three;  // distinct (w-2 w-1 w)
        for (const auto& f : four) three.push_back({f[1], f[2], f[3]});
        std::sort(three.begin(), three.end());
        three.erase(std::unique(three.begin(), three.end()), three.end());
        // Order 3 (context "The cat"): N(w) = distinct w-3 with (w-3, The, cat, w) in four.
        std::unordered_map<std::uint32_t, double> n3;
        for (const auto& f : four)
            if (f[1] == The && f[2] == cat) n3[f[3]] += 1;
        // Order 2 (context "cat"): N(w) = distinct w-2 with (w-2, cat, w) in three.
        std::unordered_map<std::uint32_t, double> n2;
        for (const auto& f : three)
            if (f[1] == cat) n2[f[2]] += 1;
        // Order 1: distinct w-1 before w, over the distinct (w-1 w) pairs of the order-2 level.
        std::vector<std::array<std::uint32_t, 2>> two;
        for (const auto& f : three) two.push_back({f[1], f[2]});
        std::sort(two.begin(), two.end());
        two.erase(std::unique(two.begin(), two.end()), two.end());
        std::vector<double> cc(V, 0.01);
        for (const auto& f : two) cc[f[1]] += 1;
        double zc = 0;
        for (const double x : cc) zc += x;
        const auto level = [](const std::unordered_map<std::uint32_t, double>& n, const double* D, std::uint32_t w, double lower) {
            double tot = 0, g = 0;
            for (const auto& [x, c] : n) tot += c, g += D[std::min(3, static_cast<int>(c))];
            const auto it = n.find(w);
            const double c = it == n.end() ? 0 : it->second;
            return (c > 0 ? (c - D[std::min(3, static_cast<int>(c))]) / tot : 0.0) + g / tot * lower;
        };
        const double q = level(n3, m.kn().discounts(3), saw, level(n2, m.kn().discounts(2), saw, cc[saw] / zc));
        // The 4-gram context "BOS The cat"-style contexts would go further; ask with a history of exactly two words.
        const std::uint32_t h[2] = {The, cat};
        m.kn().dist(h, 2, p.data());
        if (n3.empty() || std::fabs(p[saw] - q) > 1e-12 * q) {
            err = "KN differs from the direct count (" + std::to_string(p[saw]) + " vs " + std::to_string(q) + ")";
            return false;
        }
    }
    {  // GENERATION_PLAN §13.40 M1: the count index against a direct scan; the same distributions whatever is kept as
       // tables (every context, the default, none); the exact cap against the hash calls.
        const std::vector<std::uint32_t>& ids = m.train_ids();
        std::mt19937_64 r(11);
        for (int t = 0; t < 300; ++t) {
            const std::size_t len = 1 + r() % 14, s = r() % (ids.size() - len);
            std::vector<std::uint32_t> q(ids.begin() + static_cast<std::ptrdiff_t>(s), ids.begin() + static_cast<std::ptrdiff_t>(s + len));
            if (t % 3 == 0) q[r() % len] = static_cast<std::uint32_t>(r() % V);
            std::size_t direct = 0, longest = 0;
            for (std::size_t a = 0; a < ids.size(); ++a) {
                std::size_t k = 0;
                while (k < len && a + k < ids.size() && ids[a + k] == q[k]) ++k;
                direct += k == len, longest = std::max(longest, k);
            }
            if (m.kn().index().count(q.data(), len) != direct || m.kn().index().match(q.data(), len) != longest) {
                err = "count index: count or match differs from a direct scan";
                return false;
            }
        }
        WordKN all, none;
        all.build(ids, V, kOrder, 1);
        none.build(ids, V, kOrder, 1u << 30);
        std::vector<double> pa(V), pn(V);
        for (int t = 0; t < 400; ++t) {
            const std::size_t pos = 1 + r() % (ids.size() - 1), len = std::min<std::size_t>(pos, 1 + r() % 6);
            std::vector<std::uint32_t> c(ids.begin() + static_cast<std::ptrdiff_t>(pos - len), ids.begin() + static_cast<std::ptrdiff_t>(pos));
            if (t % 4 == 1) c.back() = static_cast<std::uint32_t>(r() % V);
            m.kn().dist(c.data(), c.size(), p.data());
            all.dist(c.data(), c.size(), pa.data());
            none.dist(c.data(), c.size(), pn.data());
            const std::uint32_t w = pos < ids.size() ? ids[pos] : kBos;
            if (std::memcmp(p.data(), pa.data(), V * 8) != 0 || std::memcmp(p.data(), pn.data(), V * 8) != 0 ||
                m.kn().prob(c.data(), c.size(), w, m.kn().p1()[w]) != none.prob(c.data(), c.size(), w, none.p1()[w])) {
                err = "KN differs between contexts kept as tables and counted from the suffix array";
                return false;
            }
        }
        if (all.entries(4) != m.kn().entries(4) || all.kept_contexts(4) != all.contexts(4) || none.kept_contexts(4) != 0) {
            err = "count index: kept contexts";
            return false;
        }
        const NgramSet cs = m.cap_set();
        for (std::size_t pos = 12; pos < ids.size(); pos += 7) {
            std::vector<std::uint32_t> c(ids.begin() + static_cast<std::ptrdiff_t>(pos - 12), ids.begin() + static_cast<std::ptrdiff_t>(pos));
            NgramSet::Index::Range rr;
            std::uint64_t h = 0;
            const bool a = cs.prefix(c, rr), b = cs.prefix(c, h);
            const std::uint32_t w1 = ids[pos], w2 = static_cast<std::uint32_t>(r() % V);
            if (a != b || (a && (cs.completes(rr, w1) != cs.has(fnv_step(h, w1)) || cs.completes(rr, w2) != cs.has(fnv_step(h, w2)) || !cs.completes(rr, w1)))) {
                err = "copy cap: the suffix array and the window hashes differ";
                return false;
            }
        }
    }
    // The cap: after twelve words of the repeated sentence the thirteenth is never written; with --k5-cap 0 it is.
    {
        const std::string pr = "One two three four five six seven eight nine ten eleven twelve";
        Options o = k5_v1();
        o.bytes = 120;
        int with = 0, without = 0;
        GenStats st;
        for (std::uint64_t s = 0; s < 40; ++s) {
            with += m.generate(pr, s, o, &st).rfind(" thirteen", 0) == 0;
            Options o0 = o;
            o0.cap = false;
            without += m.generate(pr, s, o0).rfind(" thirteen", 0) == 0;
        }
        if (with != 0 || without == 0 || st.cap_banned == 0) {
            err = "copy cap (with " + std::to_string(with) + ", without " + std::to_string(without) + ")";
            return false;
        }
        const std::string g1 = m.generate("The cat saw", 5, o), g2 = m.generate("The cat saw", 5, o);
        if (g1 != g2 || g1.empty() || g1.size() > o.bytes || g1.find("<unk>") != std::string::npos || m.has_run13(g1)) {
            err = "generate: " + g1;
            return false;
        }
    }
    // K5_UPGRADES.md: the flags, the online parts, orders 6 and 7, the classes and the gate.
    {
        Options o;
        std::string e;
        Options v1;
        if (!apply_flag("--k5-base", "k5", v1, e) || v1.nucleus != 0.9 || v1.classes != 0 || v1.gate_k != 0 ||
            !apply_flag("--k5-gate", "4,wf=1,wp=0.5,wm=5", o, e) || o.gate_k != 4 || o.gate_wp != 0.5 || !apply_flag("--k5-cache", "0.1", o, e) ||
            apply_flag("--k5-nucleus", "1.5", o, e) || apply_flag("--k5-bogus", "1", o, e) || !apply_flag("--k5-class-mode", "mix", o, e) ||
            o.class_mode != 2 || Options{}.describe() != "") {
            err = "apply_flag";
            return false;
        }
    }
    const auto sums = [&](const K5& mm, const Options& o, const char* what) {
        for (const char* c : ctxs) {
            K5::State s = mm.start(c, o);
            for (const auto& t : tokens("the river . It saw"))  // the online parts also learn a few words
                K5::learn(s, mm.id(t), o);
            std::vector<double> q(mm.vocab().size()), tmp;
            mm.model_dist(s, o, q.data(), tmp);
            double z = 0, lo = 1;
            for (const double x : q) z += x, lo = std::min(lo, x);
            if (std::fabs(z - 1) > 1e-9 || lo < 0) {
                err = std::string(what) + " does not sum to one after \"" + c + "\" (" + std::to_string(z) + ")";
                return false;
            }
        }
        return true;
    };
    {
        Options o = k5_v1();
        o.dyn = 3;
        if (!sums(m, o, "dynamic counts")) return false;
        o = k5_v1();
        o.cache_w = 0.2, o.cache_w2 = 0.1, o.cache_r = 0.9;
        if (!sums(m, o, "the cache")) return false;
        // The dynamic counts raise a word the prompt put after its context; the shared model is unchanged.
        Options d = k5_v1();
        d.dyn = 5;
        const std::string pr = "The cat lost Paris. The cat lost";
        const auto prob = [&](const Options& oo) {
            K5::State s = m.start(pr, oo);
            std::vector<double> q(V), tmp;
            m.model_dist(s, oo, q.data(), tmp);
            return q[m.id("Paris")];
        };
        const double p0 = prob(k5_v1()), p1 = prob(d), p2 = prob(k5_v1());
        if (!(p1 > p0) || p2 != p0) {
            err = "dynamic counts do not raise the prompt's continuation";
            return false;
        }
        // K5's own path: cont_bits with default options is the plain walk.
        if (m.cont_bits("The cat", "saw the river.").first != m.cont_bits("The cat", "saw the river.", k5_v1()).first) {
            err = "cont_bits";
            return false;
        }
    }
    for (int ord : {6, 7}) {
        K5 m6;
        m6.build(text, 12, ord);
        if (m6.kn().order() != ord || m6.kn().entries(ord) == 0 || !sums(m6, k5_v1(), "order 6/7")) {
            if (err.empty()) err = "order " + std::to_string(ord);
            return false;
        }
        Options o = k5_v1();
        o.dyn = 2;
        if (!sums(m6, o, "order 6/7 with dynamic counts")) return false;
    }
    {
        K5 mc;
        mc.build(text, 12, kOrder, 8);
        const auto& ll = mc.classes().loglik();
        if (!mc.classes().on() || ll.size() != 7 || ll.back() < ll.front()) {
            err = "exchange classes";
            return false;
        }
        Options o = k5_v1();
        if (!sums(mc, o, "class backoff")) return false;
        o.class_mode = 2, o.class_w = 0.3;
        if (!sums(mc, o, "class mix")) return false;
    }
    {  // The gate: deterministic, within the budget, no copy run; K 1 is K5 itself.
        Options o = k5_v1();
        o.bytes = 200;
        const std::string a0 = m.generate("The cat saw", 9, o);
        Options g1 = o;
        g1.gate_k = 1;
        Options g4 = o;
        g4.gate_k = 4, g4.gate_wp = 1;
        GenStats st;
        const std::string a1 = m.generate("The cat saw", 9, g1), b1 = m.generate("The old man found Paris", 9, g4, &st),
                          b2 = m.generate("The old man found Paris", 9, g4);
        if (a0 != a1 || b1 != b2 || b1.empty() || b1.size() > o.bytes || m.has_run13(b1) || st.gate_sents == 0 ||
            st.gate_cands != st.gate_sents * 4) {
            err = "gate: " + b1;
            return false;
        }
        const auto as = m.prompt_assoc({kBos, m.id("Paris"), m.id("river")}, 10, 0);
        double mx = 0;
        for (const double x : as) mx = std::max(mx, x);
        if (!(mx > 0)) {
            err = "prompt association";
            return false;
        }
    }
    {  // docs/reports/OPT_NUCLEUS.md: the repeat ban's batch check gives banned()'s answers; fast_pick makes full_pick's choice.
        std::mt19937_64 r(23);
        const char al[] = "ab e.";  // a small alphabet, so windows repeat often
        std::size_t nb = 0, nall = 0;
        for (int t = 0; t < 4000; ++t) {
            std::string tx;
            const std::size_t n = r() % 90;
            for (std::size_t i = 0; i < n; ++i) tx += al[r() % (t % 2 ? 5 : 3)];
            RepeatBan rb(tx.substr(0, n / 2));
            rb.add(tx.substr(n / 2));
            RepeatBan::Batch b;
            rb.prepare(b);
            for (int k = 0; k < 24; ++k) {
                std::string pc;
                const std::size_t L = 1 + r() % 30;
                if (k % 3 == 0 && n > 0) {  // a piece of the text, perhaps changed
                    const std::size_t a = r() % n;
                    pc = tx.substr(a, L);
                    if (k % 2) pc[r() % pc.size()] = al[r() % 5];
                } else {
                    for (std::size_t i = 0; i < L; ++i) pc += al[r() % (k % 2 ? 2 : 5)];
                }
                const bool want = rb.banned(pc);
                nb += want, ++nall;
                if (rb.banned(b, pc) != want) {
                    err = "repeat ban: the batch check differs from banned() (text \"" + tx + "\", piece \"" + pc + "\")";
                    return false;
                }
            }
        }
        if (nb == 0 || nb == nall) {
            err = "repeat ban: the batch test bans every piece or none";
            return false;
        }
        K5 mc;
        mc.build(text, 12, kOrder, 8);
        std::vector<Options> os(5, k5_v1());
        os[1] = Options{};
        os[2].cand_max = 5;
        os[3].cand_mass = 1.0, os[3].nucleus = 0.5;
        os[4].nucleus = 1.0, os[4].cap = false;
        std::size_t both = 0, all = 0;
        for (const K5* mm : {&m, &mc})
            for (const Options& oo : os)
                for (const char* c : ctxs)
                    for (std::uint64_t sd = 0; sd < 3; ++sd)
                        if (!mm->check_nucleus(c, 100 + sd, oo, 120, both, all, err)) return false;
        // fast_pick declines where z is 0 (the toy text puts all of some contexts' mass on <s>, which is zeroed).
        if (both * 4 < all * 3) {
            err = "nucleus: fast_pick chose at only " + std::to_string(both) + " of " + std::to_string(all) + " steps";
            return false;
        }
    }
    {  // K5_DISCOURSE.md: the helpers' own tests, then their hooks.
        if (!mbr::self_test(err) || !eg::self_test(err) || !lda::self_test(err)) return false;
        Options o = k5_v1();
        o.bytes = 200;
        o.gate_k = 4, o.gate_wf = 0, o.gate_wp = 1, o.gate_wm = 0;
        Options oz = o;
        oz.eg_w = 1e-12;  // no grid built: the term is 0, so the z-scored gate must choose as the default gate
        if (!oz.zgate() || m.generate("The old man found Paris", 9, o) != m.generate("The old man found Paris", 9, oz)) {
            err = "discourse: the z-scored gate differs from the gate with a null term";
            return false;
        }
        Options e = k5_v1();
        if (!apply_flag("--k5-lda", "3", e, err) || !apply_flag("--k5-lda-iters", "20", e, err) || !apply_flag("--k5-lda-beta", "0.5", e, err) ||
            !apply_flag("--k5-eg", "1", e, err) || !apply_flag("--k5-eg-mode", "cont", e, err) || !apply_flag("--k5-mbr-sent", "1", e, err) ||
            !apply_flag("--k5-mbr-doc", "3", e, err) || !apply_flag("--k5-meta", "base:1,gate:1,eg:1,topic:1", e, err) ||
            apply_flag("--k5-meta", "foo:1", e, err) || apply_flag("--k5-eg-mode", "x", e, err) || e.eg_mode != 2 || e.lda_k != 3) {
            err = "discourse flags: " + err;
            return false;
        }
        K5 md;
        md.build(text, 12);
        md.build_discourse(text, e);
        if (!md.entity_grid().trained() || !md.topic_model().trained()) return err = "discourse: models not built", false;
        if (!sums(md, e, "topic rescaling")) return false;
        e.gate_k = 4, e.lda_w = 1, e.bytes = 200;
        Options e1 = e;
        e1.mbr_doc = 0, e1.meta.clear();
        const std::string g1 = md.generate("The old man found Paris", 9, e1), g2 = md.generate("The old man found Paris", 9, e1);
        GenStats st;
        const std::string p1 = md.generate("The old man found Paris", 9, e, &st), p2 = md.generate("The old man found Paris", 9, e);
        Options e3 = e;
        e3.meta.clear();
        const std::string d1 = md.generate("The cat saw", 5, e3);
        if (g1 != g2 || g1.empty() || g1.size() > e.bytes || md.has_run13(g1) || p1 != p2 || p1.empty() || st.pool_chosen.size() != 1 ||
            d1.empty() || d1.size() > e.bytes) {
            err = "discourse: generation (" + g1 + " | " + p1 + ")";
            return false;
        }
    }
    return true;
}

}  // namespace cypha::cyphalm::wkn
