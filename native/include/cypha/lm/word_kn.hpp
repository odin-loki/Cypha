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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm::wkn {

inline constexpr std::uint32_t kUnk = 0, kBos = 1;
inline constexpr std::size_t kVmax = 32000;     // fuzzy_bake.py VMAX
inline constexpr std::uint64_t kSeed = 20261101;  // fuzzy_bake.py SEED
inline constexpr int kOrder = 5;

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

class WordKN {
 public:
    // ids: the training stream (documents as BOS + token ids), every id below V <= 65,536.
    void build(const std::vector<std::uint32_t>& ids, std::size_t V) {
        V_ = V;
        constexpr int L = kOrder - 1;
        std::vector<Pair> pr;
        if (ids.size() > static_cast<std::size_t>(L)) {
            pr.reserve(ids.size() - L);
            for (std::size_t t = L; t < ids.size(); ++t) {
                std::uint64_t k = 0;
                for (int i = 0; i < L; ++i) k |= std::uint64_t{ids[t - 1 - static_cast<std::size_t>(i)]} << (16 * i);
                pr.push_back({k, static_cast<std::uint16_t>(ids[t])});
            }
        }
        for (int k = kOrder; k >= 2; --k) {
            std::sort(pr.begin(), pr.end());
            Level& lv = lv_[k];
            lv = Level{};
            std::vector<std::uint32_t> cnt;
            for (std::size_t a = 0; a < pr.size();) {
                std::size_t b = a + 1;
                while (b < pr.size() && pr[b].key == pr[a].key && pr[b].w == pr[a].w) ++b;
                if (lv.keys.empty() || lv.keys.back() != pr[a].key) {
                    lv.keys.push_back(pr[a].key);
                    lv.off.push_back(static_cast<std::uint32_t>(lv.ids.size()));
                }
                lv.ids.push_back(pr[a].w);
                cnt.push_back(static_cast<std::uint32_t>(b - a));
                a = b;
            }
            lv.off.push_back(static_cast<std::uint32_t>(lv.ids.size()));
            finish(lv, cnt);
            if (k > 2) {  // the next order's pairs: this order's distinct entries, the farthest context word dropped
                const std::uint64_t mask = (std::uint64_t{1} << (16 * (k - 2))) - 1;
                pr.clear();
                for (std::size_t c = 0; c < lv.keys.size(); ++c)
                    for (std::uint32_t e = lv.off[c]; e < lv.off[c + 1]; ++e) pr.push_back({lv.keys[c] & mask, lv.ids[e]});
            }
        }
        std::vector<double> cc(V, 0.0);
        for (const std::uint16_t w : lv_[2].ids) cc[w] += 1.0;
        double z = 0;
        for (double& x : cc) z += (x += 0.01);
        p1_.resize(V);
        for (std::size_t w = 0; w < V; ++w) p1_[w] = cc[w] / z;
        by_p1_.resize(V);
        for (std::uint32_t w = 0; w < V; ++w) by_p1_[w] = w;
        std::stable_sort(by_p1_.begin(), by_p1_.end(), [&](std::uint32_t x, std::uint32_t y) { return p1_[x] > p1_[y]; });
    }
    // The vocabulary by its order-1 probability, most probable first: every word outside the stored contexts' entries
    // keeps this order in any distribution, which makes a tight threshold for the most probable words (K5::generate).
    const std::vector<std::uint32_t>& by_p1() const { return by_p1_; }

    // P(. | ctx) into p[0, V): ctx[0, n) is the history, its last word last.
    void dist(const std::uint32_t* ctx, std::size_t n, double* p) const {
        std::copy(p1_.begin(), p1_.end(), p);
        for (int k = 2; k <= kOrder; ++k) {
            const std::size_t L = static_cast<std::size_t>(k - 1);
            if (n < L) break;
            std::uint64_t key = 0;
            for (std::size_t i = 0; i < L; ++i) key |= std::uint64_t{ctx[n - 1 - i]} << (16 * i);
            const Level& lv = lv_[k];
            const auto it = std::lower_bound(lv.keys.begin(), lv.keys.end(), key);
            if (it == lv.keys.end() || *it != key) break;
            const auto c = static_cast<std::size_t>(it - lv.keys.begin());
            const double g = lv.gam[c];
            for (std::size_t w = 0; w < V_; ++w) p[w] *= g;
            for (std::uint32_t e = lv.off[c]; e < lv.off[c + 1]; ++e) p[lv.ids[e]] += lv.disc[e];
        }
    }
    std::size_t vocab_size() const { return V_; }
    std::size_t contexts(int k) const { return lv_[k].keys.size(); }
    std::size_t entries(int k) const { return lv_[k].ids.size(); }
    const double* discounts(int k) const { return lv_[k].D; }
    std::size_t bytes() const {
        std::size_t b = p1_.size() * 8;
        for (int k = 2; k <= kOrder; ++k)
            b += lv_[k].keys.size() * 16 + lv_[k].off.size() * 4 + lv_[k].ids.size() * 10;
        return b;
    }

 private:
    struct Pair {
        std::uint64_t key;
        std::uint16_t w;
        bool operator<(const Pair& o) const { return key != o.key ? key < o.key : w < o.w; }
    };
    struct Level {
        std::vector<std::uint64_t> keys;  // contexts, ascending: w-1 in bits 0-15, w-2 in bits 16-31, ...
        std::vector<std::uint32_t> off;   // entries of context c: [off[c], off[c + 1])
        std::vector<std::uint16_t> ids;   // the next words, ascending within a context
        std::vector<double> disc;         // (n(c w) - D) / n(c)
        std::vector<double> gam;          // sum D / n(c)
        double D[4] = {0, 0, 0, 0};
    };
    static void finish(Level& lv, const std::vector<std::uint32_t>& cnt) {
        double nj[5] = {0, 0, 0, 0, 0};
        for (const std::uint32_t c : cnt)
            if (c <= 4) nj[c] += 1;
        for (int j = 1; j <= 4; ++j) nj[j] = std::max(1.0, nj[j]);
        const double Y = nj[1] / (nj[1] + 2 * nj[2]);
        lv.D[0] = 0;
        lv.D[1] = std::max(0.0, 1 - 2 * Y * nj[2] / nj[1]);
        lv.D[2] = std::max(0.0, 2 - 3 * Y * nj[3] / nj[2]);
        lv.D[3] = std::max(0.0, 3 - 4 * Y * nj[4] / nj[3]);
        const std::size_t C = lv.keys.size();
        lv.gam.assign(C, 0.0);
        lv.disc.resize(cnt.size());
        for (std::size_t c = 0; c < C; ++c) {
            double tot = 0, g = 0;
            for (std::uint32_t e = lv.off[c]; e < lv.off[c + 1]; ++e) tot += cnt[e];
            for (std::uint32_t e = lv.off[c]; e < lv.off[c + 1]; ++e) {
                const double dc = lv.D[std::min<std::uint32_t>(cnt[e], 3)];
                lv.disc[e] = (cnt[e] - dc) / tot;
                g += dc;
            }
            lv.gam[c] = g / tot;
        }
    }
    std::size_t V_ = 0;
    std::vector<double> p1_;
    std::vector<std::uint32_t> by_p1_;
    Level lv_[kOrder + 1];
};

// ---------------------------------------------------------------- bans

// fuzzy_bake.py RepeatBan: the byte sampler's 11-byte rule lifted to words.
class RepeatBan {
 public:
    static constexpr std::size_t W = 11;
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

// The copy cap (Odin: at most 12 consecutive words copied): the hashes of every n-token window of the training ids.
class NgramSet {
 public:
    void build(const std::vector<std::uint32_t>& ids, int n) {
        n_ = n;
        h_.clear();
        hp_.clear();
        if (n <= 0 || ids.size() < static_cast<std::size_t>(n)) return;
        h_.reserve(ids.size() - static_cast<std::size_t>(n) + 1);
        hp_.reserve(h_.capacity());
        for (std::size_t s = 0; s + static_cast<std::size_t>(n) <= ids.size(); ++s) {
            std::uint64_t h = kFnv0;
            for (int j = 0; j + 1 < n; ++j) h = fnv_step(h, ids[s + static_cast<std::size_t>(j)]);
            hp_.push_back(h);
            h_.push_back(fnv_step(h, ids[s + static_cast<std::size_t>(n - 1)]));
        }
        for (auto* v : {&h_, &hp_}) {
            std::sort(v->begin(), v->end());
            v->erase(std::unique(v->begin(), v->end()), v->end());
        }
    }
    int n() const { return n_; }
    bool on() const { return n_ > 0 && !h_.empty(); }
    bool has(std::uint64_t h) const { return std::binary_search(h_.begin(), h_.end(), h); }
    // The hash of ctx's last n - 1 ids, to be finished with a candidate (fnv_step); false when ctx is shorter or its
    // last n - 1 ids begin no window (then no candidate can complete one).
    bool prefix(const std::vector<std::uint32_t>& ctx, std::uint64_t& h) const {
        const auto m = static_cast<std::size_t>(n_ - 1);
        if (!on() || ctx.size() < m) return false;
        h = kFnv0;
        for (std::size_t j = ctx.size() - m; j < ctx.size(); ++j) h = fnv_step(h, ctx[j]);
        return std::binary_search(hp_.begin(), hp_.end(), h);
    }
    // Does ids hold an n-token window of the training ids?
    bool any(const std::vector<std::uint32_t>& ids) const {
        if (!on()) return false;
        for (std::size_t s = 0; s + static_cast<std::size_t>(n_) <= ids.size(); ++s) {
            std::uint64_t h = kFnv0;
            for (int j = 0; j < n_; ++j) h = fnv_step(h, ids[s + static_cast<std::size_t>(j)]);
            if (has(h)) return true;
        }
        return false;
    }
    std::size_t size() const { return h_.size(); }
    std::size_t bytes() const { return (h_.size() + hp_.size()) * 8; }

 private:
    int n_ = 0;
    std::vector<std::uint64_t> h_, hp_;  // the windows' hashes and their first n - 1 ids' hashes
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

struct Options {
    // §13.24: the byte scorer (made once per continuation) and how it reranks the words inside the nucleus: 1 a product
    // of experts, p_k5^(1 - w) * p_bytes^w; 2 linear, (1 - w) p_k5 + w p_bytes, each normalised over the nucleus; 3 pmi,
    // p_k5 * (p_bytes / p_bytes_prior)^w (DExperts' form: the byte model adds only what the context tells it). K5
    // still chooses the candidates and the nucleus (the plausibility gate of §13.16).
    std::function<std::unique_ptr<ByteScorer>()> scorer;
    int scorer_mode = 1;
    double scorer_w = 0.25;
    bool cap = true;           // apply the model's copy cap (K5::load's cap; a model built with cap 0 has none)
    double temperature = 1.0;  // fuzzy_bake.py: 1.0
    double cand_mass = 0.995, nucleus = 0.9;
    std::size_t cand_max = 3000;
    int max_words = 300;
    std::size_t bytes = 600;
};

struct GenStats {
    std::size_t steps = 0;
    std::size_t rep_banned = 0;      // candidates the repeat ban removed
    std::size_t cap_steps = 0;       // steps at which the cap removed a candidate
    std::size_t cap_banned = 0;      // candidates the cap removed
    std::size_t cap_mass_steps = 0;  // ... of which the cap removed mass the draw could reach (inside the 0.9 nucleus)
    std::size_t rep_lifted = 0;      // steps at which the repeat ban left nothing and was lifted
    std::size_t cap_outside = 0;     // steps at which the cap left no candidate (a word from outside them was taken)
    void add(const GenStats& o) {
        steps += o.steps, rep_banned += o.rep_banned, cap_steps += o.cap_steps, cap_banned += o.cap_banned;
        cap_mass_steps += o.cap_mass_steps, rep_lifted += o.rep_lifted, cap_outside += o.cap_outside;
    }
};

class K5 {
 public:
    // Training text: one document per line. Returns false (and err) if it cannot be read or holds no tokens.
    bool load(const std::string& path, int cap, std::string& err, std::size_t vmax = kVmax) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            err = "cannot read " + path;
            return false;
        }
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        build(text, cap, vmax);
        if (vocab_.size() <= 2) {
            err = path + " holds no tokens seen twice";
            return false;
        }
        return true;
    }

    // vmax: the vocabulary's size cap (K5_SCALE: --k5-vmax), at most 65,536 (ids are 16 bits).
    void build(std::string_view text, int cap, std::size_t vmax = kVmax) {
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
        std::vector<std::uint32_t> order(tview.size());
        for (std::uint32_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::uint32_t x, std::uint32_t y) { return tcount[x] > tcount[y]; });
        vocab_ = {"<unk>", "<s>"};
        std::vector<std::uint32_t> id_of(tview.size(), kUnk);
        for (const std::uint32_t t : order) {
            if (tcount[t] < 2 || vocab_.size() >= vmax) break;
            id_of[t] = static_cast<std::uint32_t>(vocab_.size());
            vocab_.emplace_back(tview[t]);
        }
        tid_.clear();
        tid_.reserve(vocab_.size() * 2);
        for (std::uint32_t i = 0; i < vocab_.size(); ++i) tid_.emplace(vocab_[i], i);
        std::vector<std::uint32_t> ids;
        ids.reserve(seq.size());
        unk_ = 0;
        for (const std::uint32_t s : seq) {
            ids.push_back(s == ~0u ? kBos : id_of[s]);
            unk_ += s != ~0u && id_of[s] == kUnk;
        }
        tokens_ = ids.size();
        kn_.build(ids, vocab_.size());
        run13_.build(ids, 13);  // fuzzy_bake.py has_run: the measure of verbatim runs of more than 12 tokens
        if (cap > 0 && cap + 1 != 13) capset_.build(ids, cap + 1);
        cap_ = cap;
        secs_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    std::uint32_t id(std::string_view t) const {
        const auto it = tid_.find(std::string(t));
        return it == tid_.end() ? kUnk : it->second;
    }
    const std::vector<std::string>& vocab() const { return vocab_; }
    const WordKN& kn() const { return kn_; }
    const NgramSet& cap_set() const { return cap_ + 1 == 13 ? run13_ : capset_; }
    int cap() const { return cap_; }
    std::size_t docs() const { return docs_; }
    std::size_t tokens() const { return tokens_; }
    std::size_t unk() const { return unk_; }
    double build_seconds() const { return secs_; }
    std::size_t bytes() const { return kn_.bytes() + run13_.bytes() + capset_.bytes() + vocab_.size() * 64; }

    // fuzzy_bake.py has_run(g): does g, tokenised and mapped to ids, hold 13 consecutive ids of the training stream?
    bool has_run13(std::string_view g) const {
        std::vector<std::uint32_t> ids;
        tokenize(g, [&](std::string_view t) { ids.push_back(id(t)); });
        return run13_.any(ids);
    }

    // fuzzy_bake.py cont_bits: bits of the continuation's tokens under KN (no zeroing; an unknown word gets <unk>'s
    // mass) and the bytes of their pieces.
    std::pair<double, double> cont_bits(std::string_view prompt, std::string_view cont) const {
        std::vector<std::uint32_t> ctx = {kBos};
        std::string prev;
        bool has_prev = false;
        tokenize(prompt, [&](std::string_view t) {
            ctx.push_back(id(t));
            prev.assign(t);
            has_prev = true;
        });
        std::vector<double> p(vocab_.size());
        double bits = 0, nb = 0;
        tokenize(cont, [&](std::string_view t) {
            const std::uint32_t y = id(t);
            kn_.dist(ctx.data(), ctx.size(), p.data());
            bits -= std::log2(std::max(p[y], 1e-12));
            nb += static_cast<double>(piece(t, prev, has_prev).size());
            ctx.push_back(y);
            prev.assign(t);
            has_prev = true;
        });
        return {bits, nb};
    }

    // fuzzy_bake.py generate_one: the continuation of prompt, seeded with seed.
    std::string generate(std::string_view prompt, std::uint64_t seed, const Options& o, GenStats* st = nullptr) const {
        std::mt19937_64 rng(seed);
        GenStats g;
        std::vector<std::uint32_t> ctx = {kBos};
        std::string prev;
        bool has_prev = false;
        tokenize(prompt, [&](std::string_view t) {
            ctx.push_back(id(t));
            prev.assign(t);
            has_prev = true;
        });
        RepeatBan rep(prompt);
        const NgramSet& cs = cap_set();
        std::unique_ptr<ByteScorer> bs = o.scorer ? o.scorer() : nullptr;  // §13.24
        if (bs) bs->begin(prompt);
        std::vector<double> rw;
        const std::size_t V = vocab_.size();
        if (V <= 2) return {};  // no word but <unk> and <s>
        std::vector<double> p(V);
        std::vector<std::uint32_t> top, cand, kept;
        top.reserve(V);
        std::vector<std::string> pieces;
        std::vector<char> rb, cb;
        std::string out;
        // Words of equal probability (common in the tail: the same continuation count, no stored context) are ordered
        // by a hash salted per step, not by id: ids follow frequency, and the most frequent of a tied tail block are words
        // that follow one word only ("Angeles", "Shrugged", and sentence openers such as "Instead" and "Moreover", which
        // only follow a full stop), which an id order would let in whenever a cut splits the block. numpy's argpartition
        // and argsort leave ties in no particular order.
        std::uint64_t salt = 0;
        const auto above = [&](std::uint32_t x, std::uint32_t y) {
            if (p[x] != p[y]) return p[x] > p[y];
            const std::uint64_t hx = mix64(x ^ salt), hy = mix64(y ^ salt);
            return hx != hy ? hx < hy : x < y;
        };
        for (int step = 0; step < o.max_words; ++step) {
            salt = mix64(seed ^ mix64(static_cast<std::uint64_t>(step) + 0x5EED));
            ++g.steps;
            kn_.dist(ctx.data(), ctx.size(), p.data());
            if (o.temperature != 1.0)
                for (double& x : p) x = std::pow(x, 1.0 / o.temperature);
            p[kUnk] = p[kBos] = 0;
            double z = 0;
            for (const double x : p) z += x;
            for (double& x : p) x /= z;
            // Candidates: the cand_max most probable (ties by the salted hash), cut where their cumulative mass reaches
            // cand_mass.
            // tau, the least p among m words of positive p (the first m by p1), is at most the m-th largest p, so the
            // words with p >= tau hold the m most probable.
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
            std::partial_sort(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(m), top.end(), above);
            cand.clear();
            double cum = 0;
            for (std::size_t i = 0; i < m; ++i) {
                cand.push_back(top[i]);
                cum += p[top[i]];
                if (cum >= o.cand_mass) break;
            }
            // Bans.
            const std::size_t nc = cand.size();
            pieces.resize(nc);
            rb.assign(nc, 0);
            cb.assign(nc, 0);
            std::uint64_t hp = 0;
            const bool capped = o.cap && cs.prefix(ctx, hp);
            std::size_t ncap = 0;
            bool any_kept = false, any_cap_ok = false;
            for (std::size_t i = 0; i < nc; ++i) {
                pieces[i] = piece(vocab_[cand[i]], prev, has_prev);
                cb[i] = capped && cs.has(fnv_step(hp, cand[i]));
                rb[i] = rep.banned(pieces[i]);
                ncap += cb[i];
                g.rep_banned += rb[i] && !cb[i];
                any_cap_ok = any_cap_ok || !cb[i];
                any_kept = any_kept || (!rb[i] && !cb[i]);
            }
            g.cap_banned += ncap;
            g.cap_steps += ncap > 0;
            std::uint32_t w = 0;
            std::string s;
            if (!any_cap_ok) {  // the cap removed every candidate: the most probable word anywhere that it allows
                ++g.cap_outside;
                top.resize(V);
                for (std::uint32_t x = 0; x < V; ++x) top[x] = x;
                std::sort(top.begin(), top.end(), above);
                for (const std::uint32_t x : top)
                    if (p[x] > 0 && !cs.has(fnv_step(hp, x))) {
                        w = x;
                        break;
                    }
                s = piece(vocab_[w], prev, has_prev);
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
                s = std::move(pieces[kept[j]]);
            }
            out += s;
            rep.add(s);
            if (bs) bs->advance(s);
            ctx.push_back(w);
            prev = vocab_[w];
            has_prev = true;
            if (out.size() >= o.bytes) break;
        }
        if (out.size() > o.bytes) {  // cut to the byte limit; a character the cut splits is dropped (decode "ignore")
            out.resize(o.bytes);
            std::size_t i = out.size();
            std::size_t back = 0;
            while (i > 0 && back < 4 && (static_cast<unsigned char>(out[i - 1]) & 0xC0) == 0x80) --i, ++back;
            if (i > 0) {
                const auto c = static_cast<unsigned char>(out[i - 1]);
                const std::size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
                if (need > back + 1) out.resize(i - 1);
            }
        }
        if (st) st->add(g);
        return out;
    }

 private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, std::uint32_t> tid_;
    WordKN kn_;
    NgramSet run13_, capset_;
    int cap_ = 12;
    std::size_t docs_ = 0, tokens_ = 0, unk_ = 0;
    double secs_ = 0;
};

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
    if (!m.load(text_path, cap, err, vmax)) {
        std::cerr << "--k5: " << err << "\n";
        return 1;
    }
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
            gens[j] = m.generate(prompts[id], seed + id + 7919 * ti, o, &gst[j]);
        }
    };
    const unsigned nth = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
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
        out << "MODEL\tk5\tclean=" << text_path << "\tcap=" << cap << "\tvocab=" << m.vocab().size() << "\n";
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
    if (!conts.empty()) {
        double b = 0, n = 0;
        for (std::size_t k = 0; k < np && k < conts.size(); ++k) {
            const auto [x, y] = m.cont_bits(prompts[k], conts[k]);
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
    // The cap: after twelve words of the repeated sentence the thirteenth is never written; with --k5-cap 0 it is.
    {
        const std::string pr = "One two three four five six seven eight nine ten eleven twelve";
        Options o;
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
    return true;
}

}  // namespace cypha::cyphalm::wkn
