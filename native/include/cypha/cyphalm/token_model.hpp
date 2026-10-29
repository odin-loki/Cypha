// CyphaLM count models over token IDs (docs/TOKENS_PLAN.md T2; docs/reports/TOKENS_T0_T2.md).
//
//   KneserNey    orders 1..K, interpolated modified Kneser-Ney (Chen & Goodman 1998): raw counts at the top order,
//                continuation counts below it, three discounts per order from the count of counts, a uniform base.
//   TokenInfini  an ∞-gram over token IDs (Liu et al. 2024's rule): the longest suffix of the history that occurs in
//                the training tokens with a token after it; P(next) = that token's count over the suffix's count.
//
// Contexts are 64-bit hashes of token spans (a 64-bit collision over ~10^7 spans has probability ~10^-5). Counts,
// discounts and lookups are exact otherwise. The suffix array comes from libsais (libsais_int).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "libsais.h"

namespace cypha::cyphalm::tok {

inline std::uint64_t splitmix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
inline constexpr std::uint64_t kEmptyCtx = 0x6a09e667f3bcc909ULL;
// The key of a span grows to the left: key(t_s..t_e) = step(key(t_{s+1}..t_e), t_s), key(empty) = kEmptyCtx.
inline std::uint64_t step(std::uint64_t h, std::uint32_t t) { return splitmix(h ^ (std::uint64_t{t} * 0xD6E8FEB86659FD93ULL + 1)); }

class KneserNey {
 public:
    struct Entry {
        std::uint64_t ctx;
        std::uint32_t tok, c;
    };
    struct Ctx {
        std::uint64_t ctx;
        double sum;
        std::uint32_t n1, n2, n3;
    };
    struct Order {
        std::vector<Entry> e;   // sorted by (ctx, tok)
        std::vector<Ctx> cx;    // sorted by ctx
        double D[4] = {0, 0, 0, 0};  // D[1], D[2], D[3] (3 = three or more)
        std::uint64_t nc[5] = {0, 0, 0, 0, 0};  // count of counts 1..4
    };

    KneserNey() = default;
    // Counts the tokens t[0..n) with vocabulary size V (every id < V).
    KneserNey(const std::uint32_t* t, std::size_t n, std::uint32_t V, int K) : V_(V), K_(K), ord_(static_cast<std::size_t>(K) + 1) {
        if (K < 1) throw std::runtime_error("order must be at least 1");
        std::vector<std::uint64_t> hp(n, kEmptyCtx), hc(n);  // H_{k-1}(i), H_k(i)
        struct Tri {
            std::uint64_t a, b;
            std::uint32_t tok;
        };
        for (int k = 1; k <= K; ++k) {
            // H_k(i) = key(t_{i-k+1}..t_i), valid for i >= k - 1.
            for (std::size_t i = 0; i < n; ++i) hc[i] = i + 1 >= static_cast<std::size_t>(k) ? step(hp[i], t[i + 1 - static_cast<std::size_t>(k)]) : 0;
            std::vector<Entry> raw;
            if (k == K) {
                // Raw counts of (key(t_{i-k+1}..t_{i-1}), t_i).
                raw.reserve(n);
                for (std::size_t i = static_cast<std::size_t>(k) - 1; i < n; ++i)
                    raw.push_back({k == 1 ? kEmptyCtx : hp[i - 1], t[i], 1});
                if (k == 1)
                    for (auto& r : raw) r.ctx = kEmptyCtx;
                sort_count(raw);
            } else {
                // Continuation counts: distinct left extensions (key(t_{i-k}..t_{i-1}), t_i) of (key(t_{i-k+1}..t_{i-1}), t_i).
                std::vector<Tri> tri;
                tri.reserve(n);
                for (std::size_t i = static_cast<std::size_t>(k); i < n; ++i)
                    tri.push_back({hc[i - 1], k == 1 ? kEmptyCtx : hp[i - 1], t[i]});
                std::sort(tri.begin(), tri.end(), [](const Tri& x, const Tri& y) { return x.a != y.a ? x.a < y.a : x.tok < y.tok; });
                tri.erase(std::unique(tri.begin(), tri.end(), [](const Tri& x, const Tri& y) { return x.a == y.a && x.tok == y.tok; }), tri.end());
                raw.reserve(tri.size());
                for (const auto& x : tri) raw.push_back({x.b, x.tok, 1});
                sort_count(raw);
            }
            Order& o = ord_[static_cast<std::size_t>(k)];
            o.e = std::move(raw);
            for (const auto& x : o.e)
                if (x.c <= 4) ++o.nc[x.c];
            set_discounts(o);
            for (std::size_t i = 0; i < o.e.size();) {
                std::size_t j = i;
                Ctx c{o.e[i].ctx, 0.0, 0, 0, 0};
                while (j < o.e.size() && o.e[j].ctx == c.ctx) {
                    c.sum += o.e[j].c;
                    if (o.e[j].c == 1) ++c.n1;
                    else if (o.e[j].c == 2) ++c.n2;
                    else ++c.n3;
                    ++j;
                }
                o.cx.push_back(c);
                i = j;
            }
            hp.swap(hc);
        }
    }

    std::uint32_t vocab() const { return V_; }
    int order() const { return K_; }
    const Order& order_table(int k) const { return ord_.at(static_cast<std::size_t>(k)); }
    Order& order_table_mut(int k) { return ord_.at(static_cast<std::size_t>(k)); }

    // P(w | the history h[0..len)), the last K - 1 tokens used.
    double prob(const std::uint32_t* h, std::size_t len, std::uint32_t w) const {
        double p = 1.0 / static_cast<double>(V_);
        std::uint64_t ctx = kEmptyCtx;
        for (int k = 1; k <= K_; ++k) {
            if (k > 1) {
                if (len < static_cast<std::size_t>(k - 1)) break;
                ctx = step(ctx, h[len - static_cast<std::size_t>(k - 1)]);
            }
            const Order& o = ord_[static_cast<std::size_t>(k)];
            const Ctx* c = find_ctx(o, ctx);
            if (!c) break;
            const std::uint32_t cnt = count(o, ctx, w);
            const double d = cnt == 0 ? 0.0 : o.D[std::min<std::uint32_t>(cnt, 3)];
            const double gamma = (o.D[1] * c->n1 + o.D[2] * c->n2 + o.D[3] * c->n3) / c->sum;
            p = std::max(static_cast<double>(cnt) - d, 0.0) / c->sum + gamma * p;
        }
        return p;
    }

    // The whole distribution over the vocabulary (for sampling and for checks). O(K V).
    void dist(const std::uint32_t* h, std::size_t len, std::vector<double>& P) const {
        P.assign(V_, 1.0 / static_cast<double>(V_));
        std::uint64_t ctx = kEmptyCtx;
        for (int k = 1; k <= K_; ++k) {
            if (k > 1) {
                if (len < static_cast<std::size_t>(k - 1)) break;
                ctx = step(ctx, h[len - static_cast<std::size_t>(k - 1)]);
            }
            const Order& o = ord_[static_cast<std::size_t>(k)];
            const Ctx* c = find_ctx(o, ctx);
            if (!c) break;
            const double gamma = (o.D[1] * c->n1 + o.D[2] * c->n2 + o.D[3] * c->n3) / c->sum;
            for (double& x : P) x *= gamma;
            auto lo = std::lower_bound(o.e.begin(), o.e.end(), ctx, [](const Entry& e, std::uint64_t v) { return e.ctx < v; });
            for (; lo != o.e.end() && lo->ctx == ctx; ++lo)
                P[lo->tok] += std::max(static_cast<double>(lo->c) - o.D[std::min<std::uint32_t>(lo->c, 3)], 0.0) / c->sum;
        }
    }

    std::size_t bytes() const {
        std::size_t b = 0;
        for (const auto& o : ord_) b += o.e.capacity() * sizeof(Entry) + o.cx.capacity() * sizeof(Ctx);
        return b;
    }

    // Chen & Goodman's estimates from the count of counts n1..n4, each clamped to [0, j].
    static void set_discounts(Order& o) {
        const double n1 = static_cast<double>(o.nc[1]), n2 = static_cast<double>(o.nc[2]), n3 = static_cast<double>(o.nc[3]),
                     n4 = static_cast<double>(o.nc[4]);
        const double Y = (n1 + 2 * n2) > 0 ? n1 / (n1 + 2 * n2) : 0.5;
        o.D[1] = n1 > 0 ? 1 - 2 * Y * n2 / n1 : 0.5;
        o.D[2] = n2 > 0 ? 2 - 3 * Y * n3 / n2 : 1.0;
        o.D[3] = n3 > 0 ? 3 - 4 * Y * n4 / n3 : 1.5;
        for (int j = 1; j <= 3; ++j) o.D[j] = std::clamp(o.D[j], 0.0, static_cast<double>(j));
    }

 private:
    static void sort_count(std::vector<Entry>& v) {
        std::sort(v.begin(), v.end(), [](const Entry& x, const Entry& y) { return x.ctx != y.ctx ? x.ctx < y.ctx : x.tok < y.tok; });
        std::size_t o = 0;
        for (std::size_t i = 0; i < v.size();) {
            std::size_t j = i;
            std::uint32_t c = 0;
            while (j < v.size() && v[j].ctx == v[i].ctx && v[j].tok == v[i].tok) c += v[j++].c;
            v[o++] = {v[i].ctx, v[i].tok, c};
            i = j;
        }
        v.resize(o);
        v.shrink_to_fit();
    }
    static const Ctx* find_ctx(const Order& o, std::uint64_t ctx) {
        auto it = std::lower_bound(o.cx.begin(), o.cx.end(), ctx, [](const Ctx& c, std::uint64_t v) { return c.ctx < v; });
        return it != o.cx.end() && it->ctx == ctx ? &*it : nullptr;
    }
    static std::uint32_t count(const Order& o, std::uint64_t ctx, std::uint32_t w) {
        auto it = std::lower_bound(o.e.begin(), o.e.end(), std::make_pair(ctx, w), [](const Entry& e, const std::pair<std::uint64_t, std::uint32_t>& v) {
            return e.ctx != v.first ? e.ctx < v.first : e.tok < v.second;
        });
        return it != o.e.end() && it->ctx == ctx && it->tok == w ? it->c : 0;
    }

    std::uint32_t V_ = 0;
    int K_ = 0;
    std::vector<Order> ord_;
};

// ∞-gram over token IDs: a suffix array of the training tokens.
class TokenInfini {
 public:
    struct Range {
        std::size_t lo = 0, hi = 0;
    };
    struct Result {
        int n = 0;                 // length of the longest matching suffix with a follower (0: none)
        std::uint64_t total = 0;   // its occurrences with a token after them
        std::uint64_t count = 0;   // of those, followed by the queried token
    };

    TokenInfini() = default;
    TokenInfini(const std::uint32_t* t, std::size_t n, std::uint32_t V) : T_(t, t + n) {
        if (n >= static_cast<std::size_t>(INT32_MAX)) throw std::runtime_error("too many tokens for the suffix array");
        std::vector<std::int32_t> tmp(T_.begin(), T_.end());
        SA_.resize(n);
        if (libsais_int(tmp.data(), SA_.data(), static_cast<std::int32_t>(n), static_cast<std::int32_t>(V), 0) != 0)
            throw std::runtime_error("libsais_int failed");
    }

    // Rows of the suffixes starting with p[0..m) inside r.
    Range find(const std::uint32_t* p, std::size_t m, Range r) const {
        const std::size_t N = T_.size();
        auto cmp = [&](std::int32_t pos) -> int {  // suffix at pos against p: -1 less, 0 starts with p, 1 greater
            for (std::size_t j = 0; j < m; ++j) {
                const std::size_t q = static_cast<std::size_t>(pos) + j;
                if (q >= N) return -1;
                if (T_[q] != p[j]) return T_[q] < p[j] ? -1 : 1;
            }
            return 0;
        };
        std::size_t lo = r.lo, hi = r.hi;
        while (lo < hi) {
            const std::size_t mid = (lo + hi) / 2;
            if (cmp(SA_[mid]) < 0) lo = mid + 1;
            else hi = mid;
        }
        std::size_t a = lo;
        hi = r.hi;
        while (lo < hi) {
            const std::size_t mid = (lo + hi) / 2;
            if (cmp(SA_[mid]) <= 0) lo = mid + 1;
            else hi = mid;
        }
        return {a, lo};
    }

    // The longest suffix of h[0..len) (at most cap tokens, at most hint when hint >= 0) occurring with a follower,
    // and how often w follows it.
    Result query(const std::uint32_t* h, std::size_t len, std::uint32_t w, std::size_t cap, int hint = -1) const {
        Result res;
        std::size_t m = std::min(len, cap);
        if (hint >= 0) m = std::min<std::size_t>(m, static_cast<std::size_t>(hint));
        const std::size_t N = T_.size();
        for (; m > 0; --m) {
            Range r = find(h + len - m, m, Range{0, SA_.size()});
            if (r.lo < r.hi && static_cast<std::size_t>(SA_[r.lo]) + m == N) ++r.lo;  // the suffix at the very end has no follower
            if (r.lo >= r.hi) continue;
            res.n = static_cast<int>(m);
            res.total = r.hi - r.lo;
            std::vector<std::uint32_t> pw(h + len - m, h + len);
            pw.push_back(w);
            const Range rw = find(pw.data(), pw.size(), r);
            res.count = rw.hi - rw.lo;
            return res;
        }
        return res;
    }

    // For sampling: the longest suffix of h[0..len) (as query) and the tokens that follow it with their counts.
    // Returns n = 0 and no followers when nothing matches, or when the suffix has more than max_scan occurrences
    // (a short, frequent context, where the ∞-gram adds little to Kneser-Ney).
    int followers(const std::uint32_t* h, std::size_t len, std::size_t cap, int hint, std::size_t max_scan,
                  std::vector<std::pair<std::uint32_t, std::uint64_t>>& out) const {
        out.clear();
        std::size_t m = std::min(len, cap);
        if (hint >= 0) m = std::min<std::size_t>(m, static_cast<std::size_t>(hint));
        const std::size_t N = T_.size();
        for (; m > 0; --m) {
            Range r = find(h + len - m, m, Range{0, SA_.size()});
            if (r.lo < r.hi && static_cast<std::size_t>(SA_[r.lo]) + m == N) ++r.lo;
            if (r.lo >= r.hi) continue;
            if (r.hi - r.lo > max_scan) return static_cast<int>(m);
            std::vector<std::uint32_t> next;
            next.reserve(r.hi - r.lo);
            for (std::size_t x = r.lo; x < r.hi; ++x) next.push_back(T_[static_cast<std::size_t>(SA_[x]) + m]);
            std::sort(next.begin(), next.end());
            for (std::size_t i = 0; i < next.size();) {
                std::size_t j = i;
                while (j < next.size() && next[j] == next[i]) ++j;
                out.emplace_back(next[i], j - i);
                i = j;
            }
            return static_cast<int>(m);
        }
        return 0;
    }

    std::size_t size() const { return T_.size(); }
    std::size_t bytes() const { return T_.capacity() * sizeof(std::uint32_t) + SA_.capacity() * sizeof(std::int32_t); }

 private:
    std::vector<std::uint32_t> T_;
    std::vector<std::int32_t> SA_;
};

// The byte model's InfiniGram buckets, in tokens: match length x count.
inline int tok_len_bucket(int n) { return n <= 0 ? 0 : n == 1 ? 1 : n == 2 ? 2 : n <= 4 ? 3 : n <= 8 ? 4 : n <= 16 ? 5 : 6; }
inline int tok_count_bucket(std::uint64_t t) { return t <= 1 ? 0 : t <= 3 ? 1 : t <= 15 ? 2 : 3; }
inline int tok_bucket(int n, std::uint64_t t) { return tok_len_bucket(n) * 4 + tok_count_bucket(t); }
inline constexpr int kTokBuckets = 28;

// cyphalm_trie's fit_lambdas: per bucket, the λ on a 0.005 grid that maximises the log likelihood of
// λ pe + (1 - λ) pb over the first `upto` items.
inline void fit_lambdas(const std::vector<double>& pe, const std::vector<double>& pb, const std::vector<int>& bk,
                        std::size_t upto, int nb, double* lam) {
    for (int b = 0; b < nb; ++b) {
        double best = -1e300;
        lam[b] = 0;
        for (int g = 0; g < 200; ++g) {
            const double l = g * 0.005;
            double ll = 0;
            bool any = false;
            for (std::size_t k = 0; k < upto; ++k)
                if (bk[k] == b) {
                    ll += std::log(l * pe[k] + (1 - l) * pb[k]);
                    any = true;
                }
            if (!any) break;
            if (ll > best) {
                best = ll;
                lam[b] = l;
            }
        }
    }
}

}  // namespace cypha::cyphalm::tok
