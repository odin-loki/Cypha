// CyphaLM byte-level Kneser-Ney, packed (docs/reports/BYTE_KN.md; HANDOFF 5.22).
//
// The estimate is token_model.hpp's KneserNey with V = 256 (Chen & Goodman 1998): orders 1..K, raw counts at order
// K, continuation counts (distinct left extensions) below it, three discounts per order from the count of counts, a
// uniform base of 1/256:
//     P_k(w | c) = max(n(c w) - D, 0) / n(c) + gamma(c) P_{k-1}(w | c'),
//     gamma(c)   = (D1 n1(c) + D2 n2(c) + D3 n3+(c)) / n(c).
// A context that is not stored ends the recursion at the shorter one.
//
// Packed form, per order k (contexts of k - 1 bytes):
//   - contexts grouped by their last byte (256 groups; order 1 has the one empty context), sorted by a 64-bit key
//     inside each group (the key of token_model.hpp: a hash grown to the left, one step per byte);
//   - per context: the key, the offset of its children, gamma as a float, a count band (n < 4, < 64, >= 64): 17 bytes;
//   - per child: its byte and its value max(n - D, 0) / n(c) as a float: 5 bytes.
// Pruning (from order m on): contexts with n(c) below a floor are dropped (the walk backs off), and children seen once
// are dropped with their value added to gamma, so every stored context still sums to one.
//
// The build goes order by order and, inside an order, group by group, so it holds one group's pairs at a time.
// Rule 10 (PLAN): an expert inside CyphaLM. It answers nothing.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace cypha::cyphalm::bkn {

inline std::uint64_t splitmix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
inline constexpr std::uint64_t kEmptyCtx = 0x6a09e667f3bcc909ULL;
// key(t_s..t_e) = step(key(t_{s+1}..t_e), t_s): the same keys as token_model.hpp over byte ids.
inline std::uint64_t step(std::uint64_t h, std::uint32_t t) { return splitmix(h ^ (std::uint64_t{t} * 0xD6E8FEB86659FD93ULL + 1)); }

struct Prune {
    int from_order = 1 << 30;      // pruning applies at orders >= from_order (default: none)
    std::uint32_t min_ctx = 1;     // drop contexts with n(c) below this
    bool drop_singletons = false;  // drop children seen once (their value goes to gamma)
};

class ByteKN {
 public:
    struct Order {
        std::uint32_t group[257] = {};  // contexts of group b are [group[b], group[b + 1])
        std::vector<std::uint64_t> key;
        std::vector<std::uint32_t> off;  // size contexts + 1
        std::vector<float> gamma;
        std::vector<std::uint8_t> band;
        std::vector<std::uint8_t> sym;
        std::vector<float> val;
        double D[4] = {0, 0, 0, 0};
        std::uint64_t nc[5] = {0, 0, 0, 0, 0};  // count of counts 1..4, before pruning
        std::uint64_t ctx_before = 0, child_before = 0;  // before pruning
    };

    ByteKN() = default;
    ByteKN(const std::uint8_t* t, std::size_t n, int K, const Prune& pr = Prune{}) { build(t, n, K, pr); }

    int order() const { return K_; }
    const Order& table(int k) const { return ord_.at(static_cast<std::size_t>(k)); }

    void build(const std::uint8_t* t, std::size_t n, int K, const Prune& pr = Prune{}) {
        if (K < 1 || K > 32) throw std::runtime_error("ByteKN: order must be in 1..32");
        if (n > 0xffffffffULL) throw std::runtime_error("ByteKN: text over 4 GiB");
        K_ = K;
        ord_.assign(static_cast<std::size_t>(K) + 1, Order{});
        // Positions grouped by the byte before them, gathered a chunk of groups at a time (at most kChunk positions,
        // or one group), so the build never holds a position for every byte: pos[gs[b] - base .. gs[b + 1] - base)
        // hold every i >= 1 with t[i - 1] == b, for the groups of the current chunk.
        std::vector<std::uint32_t> gs(257, 0), pos;
        for (std::size_t i = 1; i < n; ++i) ++gs[t[i - 1] + 1u];
        for (int b = 0; b < 256; ++b) gs[b + 1] += gs[b];
        std::uint32_t base = 0;
        int chunk_lo = -1, chunk_hi = -1;  // groups [chunk_lo, chunk_hi) are in pos
        auto load = [&](int g) {  // make sure group g is in pos
            if (g >= chunk_lo && g < chunk_hi) return;
            chunk_lo = g;
            chunk_hi = g + 1;
            while (chunk_hi < 256 && gs[static_cast<std::size_t>(chunk_hi) + 1] - gs[static_cast<std::size_t>(g)] <= kChunk) ++chunk_hi;
            base = gs[static_cast<std::size_t>(g)];
            pos.assign(gs[static_cast<std::size_t>(chunk_hi)] - base, 0);
            std::vector<std::uint32_t> fill(gs.begin() + g, gs.begin() + chunk_hi);
            for (std::size_t i = 1; i < n; ++i) {
                const int b = t[i - 1];
                if (b >= chunk_lo && b < chunk_hi) pos[fill[static_cast<std::size_t>(b - chunk_lo)]++ - base] = static_cast<std::uint32_t>(i);
            }
        };
        for (int k = 1; k <= K; ++k) {
            Order& o = ord_[static_cast<std::size_t>(k)];
            const int groups = k == 1 ? 1 : 256;
            // Pass 1: the count of counts (for the discounts) and the sizes before pruning.
            for (int g = 0; g < groups; ++g) {
                if (k > 1) load(g);
                const std::vector<Pair> v = group_pairs(t, n, gs, pos, base, k, K, g);
                for (std::size_t a = 0; a < v.size(); ++a) {
                    if (v[a].c <= 4) ++o.nc[v[a].c];
                    if (a == 0 || v[a].ctx != v[a - 1].ctx) ++o.ctx_before;
                }
                o.child_before += v.size();
            }
            set_discounts(o);
            // Pass 2: values, gamma and pruning, group by group, straight into the packed arrays.
            const Prune use = k >= pr.from_order ? pr : Prune{};
            for (int g = 0; g < groups; ++g) {
                o.group[g] = static_cast<std::uint32_t>(o.key.size());
                if (k > 1) load(g);
                const std::vector<Pair> v = group_pairs(t, n, gs, pos, base, k, K, g);
                for (std::size_t a = 0; a < v.size();) {
                    std::size_t b = a;
                    std::uint64_t s = 0;
                    while (b < v.size() && v[b].ctx == v[a].ctx) s += v[b++].c;
                    if (s >= use.min_ctx) {
                        const double nn = static_cast<double>(s);
                        double gm = 0;
                        const std::uint32_t first = static_cast<std::uint32_t>(o.sym.size());
                        for (std::size_t j = a; j < b; ++j) {
                            const std::uint32_t x = v[j].c;
                            const double D = o.D[std::min<std::uint32_t>(x, 3)];
                            gm += D;
                            const double val = std::max(static_cast<double>(x) - D, 0.0);
                            if (use.drop_singletons && x == 1) {
                                gm += val;  // its value moves into gamma
                                continue;
                            }
                            o.sym.push_back(static_cast<std::uint8_t>(v[j].sym));
                            o.val.push_back(static_cast<float>(val / nn));
                        }
                        o.key.push_back(v[a].ctx);
                        o.off.push_back(first);
                        o.gamma.push_back(static_cast<float>(gm / nn));
                        o.band.push_back(static_cast<std::uint8_t>(s < 4 ? 0 : s < 64 ? 1 : 2));
                    }
                    a = b;
                }
            }
            for (int g = groups; g <= 256; ++g) o.group[g] = static_cast<std::uint32_t>(o.key.size());
            o.off.push_back(static_cast<std::uint32_t>(o.sym.size()));
            o.key.shrink_to_fit(), o.off.shrink_to_fit(), o.gamma.shrink_to_fit(), o.band.shrink_to_fit();
            o.sym.shrink_to_fit(), o.val.shrink_to_fit();
        }
    }

    // The stored file (qa --save / --model): "CYPHABKN 1", K, then per order its discounts, groups and arrays, then an
    // FNV-1a checksum of everything before it. load() refuses a damaged file.
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("ByteKN: cannot write " + path);
        std::uint64_t h = 14695981039346656037ull;
        auto put = [&](const void* p, std::size_t n) {
            const auto* c = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
            f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
        };
        auto vec = [&](const auto& v) {
            const std::uint64_t n = v.size();
            put(&n, 8);
            if (n) put(v.data(), n * sizeof(v[0]));
        };
        put("CYPHABKN1", 9);
        const std::int32_t K = K_;
        put(&K, 4);
        for (int k = 1; k <= K_; ++k) {
            const Order& o = ord_[static_cast<std::size_t>(k)];
            put(o.D, sizeof o.D);
            put(o.group, sizeof o.group);
            vec(o.key), vec(o.off), vec(o.gamma), vec(o.band), vec(o.sym), vec(o.val);
        }
        f.write(reinterpret_cast<const char*>(&h), 8);
        if (!f) throw std::runtime_error("ByteKN: cannot write " + path);
    }
    static ByteKN load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("ByteKN: cannot open " + path);
        std::uint64_t h = 14695981039346656037ull;
        auto get = [&](void* p, std::size_t n) {
            if (!f.read(static_cast<char*>(p), static_cast<std::streamsize>(n))) throw std::runtime_error("ByteKN: " + path + " is short");
            const auto* c = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
        };
        auto vec = [&](auto& v) {
            std::uint64_t n = 0;
            get(&n, 8);
            if (n > (std::uint64_t{1} << 34)) throw std::runtime_error("ByteKN: " + path + " is damaged");
            v.resize(static_cast<std::size_t>(n));
            if (n) get(v.data(), static_cast<std::size_t>(n) * sizeof(v[0]));
        };
        char magic[9];
        get(magic, 9);
        if (std::memcmp(magic, "CYPHABKN1", 9) != 0) throw std::runtime_error("ByteKN: " + path + " is not a stored model");
        std::int32_t K = 0;
        get(&K, 4);
        if (K < 1 || K > 32) throw std::runtime_error("ByteKN: " + path + " is damaged");
        ByteKN m;
        m.K_ = K;
        m.ord_.assign(static_cast<std::size_t>(K) + 1, Order{});
        for (int k = 1; k <= K; ++k) {
            Order& o = m.ord_[static_cast<std::size_t>(k)];
            get(o.D, sizeof o.D);
            get(o.group, sizeof o.group);
            vec(o.key), vec(o.off), vec(o.gamma), vec(o.band), vec(o.sym), vec(o.val);
            if (o.off.size() != o.key.size() + 1 || o.gamma.size() != o.key.size() || o.band.size() != o.key.size() ||
                o.val.size() != o.sym.size() || o.group[256] != o.key.size() || o.off.back() != o.sym.size())
                throw std::runtime_error("ByteKN: " + path + " is damaged");
        }
        const std::uint64_t want = h;
        std::uint64_t got = 0;
        if (!f.read(reinterpret_cast<char*>(&got), 8) || got != want) throw std::runtime_error("ByteKN: " + path + " fails its checksum");
        return m;
    }

    // P(w | h[0..len)), the last K - 1 bytes used.
    double prob(const std::uint8_t* h, std::size_t len, unsigned w) const {
        double p = 1.0 / 256.0;
        std::uint64_t key = kEmptyCtx;
        for (int k = 1; k <= K_; ++k) {
            if (k > 1) {
                if (len < static_cast<std::size_t>(k - 1)) break;
                key = step(key, h[len - static_cast<std::size_t>(k - 1)]);
            }
            const Order& o = ord_[static_cast<std::size_t>(k)];
            const std::uint32_t c = find(o, k == 1 ? 0 : h[len - 1], key);
            if (c == kNone) break;
            p = value(o, c, w) + static_cast<double>(o.gamma[c]) * p;
        }
        return p;
    }

    // The whole distribution; returns the deepest order found (0: none) and its count band in *band.
    int dist(const std::uint8_t* h, std::size_t len, double* P, int* band = nullptr) const {
        for (int s = 0; s < 256; ++s) P[s] = 1.0 / 256.0;
        std::uint64_t key = kEmptyCtx;
        int deep = 0, bd = 0;
        for (int k = 1; k <= K_; ++k) {
            if (k > 1) {
                if (len < static_cast<std::size_t>(k - 1)) break;
                key = step(key, h[len - static_cast<std::size_t>(k - 1)]);
            }
            const Order& o = ord_[static_cast<std::size_t>(k)];
            const std::uint32_t c = find(o, k == 1 ? 0 : h[len - 1], key);
            if (c == kNone) break;
            const double gm = o.gamma[c];
            for (int s = 0; s < 256; ++s) P[s] *= gm;
            for (std::uint32_t j = o.off[c]; j < o.off[c + 1]; ++j) P[o.sym[j]] += static_cast<double>(o.val[j]);
            deep = k;
            bd = o.band[c];
        }
        if (band) *band = bd;
        return deep;
    }

    // P(w) and the bucket (deepest order found - 1) * 3 + band, for way (b)'s lambdas; buckets() of them.
    double prob_bucket(const std::uint8_t* h, std::size_t len, unsigned w, int& bucket) const {
        double p = 1.0 / 256.0;
        std::uint64_t key = kEmptyCtx;
        int deep = 0, bd = 0;
        for (int k = 1; k <= K_; ++k) {
            if (k > 1) {
                if (len < static_cast<std::size_t>(k - 1)) break;
                key = step(key, h[len - static_cast<std::size_t>(k - 1)]);
            }
            const Order& o = ord_[static_cast<std::size_t>(k)];
            const std::uint32_t c = find(o, k == 1 ? 0 : h[len - 1], key);
            if (c == kNone) break;
            p = value(o, c, w) + static_cast<double>(o.gamma[c]) * p;
            deep = k;
            bd = o.band[c];
        }
        bucket = deep == 0 ? 0 : (deep - 1) * 3 + bd;
        return p;
    }
    int buckets() const { return std::max(1, K_ * 3); }

    std::size_t bytes() const {
        std::size_t b = 0;
        for (const auto& o : ord_)
            b += o.key.capacity() * 8 + o.off.capacity() * 4 + o.gamma.capacity() * 4 + o.band.capacity() + o.sym.capacity() +
                 o.val.capacity() * 4 + sizeof(Order);
        return b;
    }
    std::size_t contexts() const {
        std::size_t c = 0;
        for (const auto& o : ord_) c += o.key.size();
        return c;
    }
    std::size_t children() const {
        std::size_t c = 0;
        for (const auto& o : ord_) c += o.sym.size();
        return c;
    }

    // Chen & Goodman's discounts from the count of counts, each clamped to [0, j] (as token_model.hpp).
    static void set_discounts(Order& o) {
        const double n1 = static_cast<double>(o.nc[1]), n2 = static_cast<double>(o.nc[2]), n3 = static_cast<double>(o.nc[3]),
                     n4 = static_cast<double>(o.nc[4]);
        const double Y = (n1 + 2 * n2) > 0 ? n1 / (n1 + 2 * n2) : 0.5;
        o.D[1] = n1 > 0 ? 1 - 2 * Y * n2 / n1 : 0.5;
        o.D[2] = n2 > 0 ? 2 - 3 * Y * n3 / n2 : 1.0;
        o.D[3] = n3 > 0 ? 3 - 4 * Y * n4 / n3 : 1.5;
        for (int j = 1; j <= 3; ++j) o.D[j] = std::clamp(o.D[j], 0.0, static_cast<double>(j));
    }

    static constexpr std::uint32_t kNone = 0xffffffffu;
    static constexpr std::uint32_t kChunk = 4u << 20;  // positions gathered at a time (16 MB)

 private:
    static double value(const Order& o, std::uint32_t c, unsigned w) {
        const std::uint8_t* s = o.sym.data();
        std::uint32_t lo = o.off[c], hi = o.off[c + 1];
        while (lo < hi) {  // children sorted by byte
            const std::uint32_t mid = (lo + hi) >> 1;
            if (s[mid] < w) lo = mid + 1;
            else hi = mid;
        }
        return lo < o.off[c + 1] && s[lo] == w ? static_cast<double>(o.val[lo]) : 0.0;
    }
    static std::uint32_t find(const Order& o, unsigned g, std::uint64_t key) {
        const std::uint64_t* a = o.key.data() + o.group[g];
        const std::uint64_t* b = o.key.data() + o.group[g + 1];
        const std::uint64_t* it = std::lower_bound(a, b, key);
        return it != b && *it == key ? static_cast<std::uint32_t>(it - o.key.data()) : kNone;
    }
    struct Pair {
        std::uint64_t ctx;
        std::uint32_t sym, c;
    };
    // The (context, byte, count) triples of order k, group g, sorted by (context, byte): raw counts at the top order,
    // continuation counts (distinct left extensions) below it.
    static std::vector<Pair> group_pairs(const std::uint8_t* t, std::size_t n, const std::vector<std::uint32_t>& gs,
                                         const std::vector<std::uint32_t>& pos, std::uint32_t base, int k, int K, int g) {
        const std::size_t L = static_cast<std::size_t>(k - 1);
        auto key_of = [&](std::size_t i) {  // key(t_{i-L} .. t_{i-1})
            std::uint64_t h = kEmptyCtx;
            for (std::size_t d = 1; d <= L; ++d) h = step(h, t[i - d]);
            return h;
        };
        std::vector<Pair> v;
        if (k == K) {
            if (k == 1) {
                std::uint32_t c1[256] = {};
                for (std::size_t i = 0; i < n; ++i) ++c1[t[i]];
                for (int s = 0; s < 256; ++s)
                    if (c1[s]) v.push_back({kEmptyCtx, static_cast<std::uint32_t>(s), c1[s]});
                return v;
            }
            v.reserve(gs[static_cast<std::size_t>(g) + 1] - gs[static_cast<std::size_t>(g)]);
            for (std::uint32_t q = gs[static_cast<std::size_t>(g)]; q < gs[static_cast<std::size_t>(g) + 1]; ++q) {
                const std::size_t i = pos[q - base];
                if (i >= L) v.push_back({key_of(i), t[i], 1});
            }
            std::sort(v.begin(), v.end(), [](const Pair& x, const Pair& y) { return x.ctx != y.ctx ? x.ctx < y.ctx : x.sym < y.sym; });
            std::size_t w = 0;
            for (std::size_t a = 0; a < v.size();) {
                std::size_t b = a;
                std::uint32_t c = 0;
                while (b < v.size() && v[b].ctx == v[a].ctx && v[b].sym == v[a].sym) c += v[b++].c;
                v[w++] = {v[a].ctx, v[a].sym, c};
                a = b;
            }
            v.resize(w);
            return v;
        }
        struct Tri {
            std::uint64_t ctx, ext;
            std::uint32_t sym;
        };
        std::vector<Tri> tri;
        if (k == 1) {  // N1+(. w): the distinct bytes seen before w
            std::vector<std::uint8_t> seen(256 * 256, 0);
            for (std::size_t i = 1; i < n; ++i) seen[t[i - 1] * 256u + t[i]] = 1;
            for (int w = 0; w < 256; ++w) {
                std::uint32_t d = 0;
                for (int x = 0; x < 256; ++x) d += seen[static_cast<std::size_t>(x) * 256 + static_cast<std::size_t>(w)];
                if (d) v.push_back({kEmptyCtx, static_cast<std::uint32_t>(w), d});
            }
            return v;
        } else {
            tri.reserve(gs[static_cast<std::size_t>(g) + 1] - gs[static_cast<std::size_t>(g)]);
            for (std::uint32_t q = gs[static_cast<std::size_t>(g)]; q < gs[static_cast<std::size_t>(g) + 1]; ++q) {
                const std::size_t i = pos[q - base];
                if (i < L + 1) continue;
                const std::uint64_t c = key_of(i);
                tri.push_back({c, step(c, t[i - L - 1]), t[i]});
            }
        }
        std::sort(tri.begin(), tri.end(), [](const Tri& x, const Tri& y) {
            return x.ctx != y.ctx ? x.ctx < y.ctx : x.sym != y.sym ? x.sym < y.sym : x.ext < y.ext;
        });
        for (std::size_t a = 0; a < tri.size();) {
            std::size_t b = a;
            std::uint32_t d = 0;
            while (b < tri.size() && tri[b].ctx == tri[a].ctx && tri[b].sym == tri[a].sym) {
                if (b == a || tri[b].ext != tri[b - 1].ext) ++d;
                ++b;
            }
            v.push_back({tri[a].ctx, tri[a].sym, d});
            a = b;
        }
        return v;
    }

    int K_ = 0;
    std::vector<Order> ord_;
};

}  // namespace cypha::cyphalm::bkn
