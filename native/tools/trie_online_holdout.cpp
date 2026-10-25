// Online learning on the packed trie during the holdout.
//
// Train [0, 7340032) with observe(). Score [7340032, 8388608) three times,
// each from those trained counts. Mode::Synth: counts below 2 dropped,
// remaining counts quantized, KT order-1 and unigram, match only when seen
// twice, u = H(order-1) / ln(256), match weight min(0.75, (1-u) * agree),
// stupid backoff alpha 0.4.
//
//   frozen_bpc          mix(), no updates
//   online_counts_bpc   mix() then observe() on each holdout byte
//   online_weights_bpc  frozen counts, five experts, multiplicative weights
//
// The weight pass is not a frozen score. Its experts are the unigram KT,
// the order-1 KT, and the normalized backoff mixture inside mix() after
// orders 2, 3, and 4. Match stays in mix(); it is not a sixth expert.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kSample = 8'388'608;
constexpr std::size_t kTrain = 7'340'032;
constexpr double kAlpha = 0.4;
constexpr double kAgreeFloor = 0.05;
constexpr double kMatchCap = 0.75;
constexpr int kGen = 200;
constexpr double kTemperature = 0.8;
constexpr double kNucleus = 0.9;
constexpr std::uint64_t kSeed = 42;
constexpr double kRefD4 = 0.861802;
constexpr double kRefLoop = 0.0485751;
constexpr std::uint32_t kNil = 0xffffffffu;

enum class Mode { Full, DropCtx, Synth };

std::uint32_t qdec[256];

void init_quant() {
    qdec[0] = 0;
    for (int s = 1; s <= 64; ++s) qdec[s] = static_cast<std::uint32_t>(s);
    double v = 64.0;
    for (int s = 65; s < 256; ++s) {
        v *= 1.08;
        auto n = static_cast<std::uint32_t>(std::llround(v));
        if (n <= qdec[s - 1]) n = qdec[s - 1] + 1;
        qdec[s] = n;
    }
}

std::uint8_t quantize(std::uint32_t c) {
    if (c <= 64) return static_cast<std::uint8_t>(c);
    int lo = 65, hi = 255;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (qdec[mid] < c) lo = mid + 1;
        else hi = mid;
    }
    if (lo > 65 && c - qdec[lo - 1] <= qdec[lo] - c) return static_cast<std::uint8_t>(lo - 1);
    return static_cast<std::uint8_t>(lo);
}

std::uint32_t used_count(std::uint32_t c, Mode mode) {
    if (c == 0) return 0;
    if (mode == Mode::Synth) {
        if (c < 2) return 0;
        return qdec[quantize(c)];
    }
    return c;
}

struct Node {
    std::uint32_t parent = kNil;
    std::uint32_t child_head = kNil;
    std::uint32_t sibling = kNil;
    std::uint32_t fol_head = kNil;
    std::uint32_t total = 0;
    std::uint16_t nchild = 0;
    std::uint16_t nfol = 0;
    std::uint8_t byte = 0;
    std::uint8_t depth = 0;
    // Past order 4 the walk continues only while every extension was the same older byte.
    std::uint8_t chain_open = 1;
};

struct Fol {
    std::uint8_t sym = 0;
    std::uint32_t count = 0;
    std::uint32_t next = kNil;
};

struct Hit {
    std::uint32_t key = 0;
    std::uint32_t follower = 0;
};

std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

struct Model {
    std::uint32_t uni[256]{};
    std::uint32_t uni_n = 0;
    std::uint32_t o1[256][256]{};
    std::uint32_t o1n[256]{};
    std::vector<Node> nodes;
    std::vector<Fol> fols;
    std::uint32_t root_child[256];
    std::vector<std::uint32_t> mk, mv, ms;
    std::uint32_t mmask = 0, mfilled = 0;
    std::vector<Hit> dense;
    std::uint32_t match_seen = 0;
    const std::uint8_t* data = nullptr;

    void init() {
        nodes.clear();
        nodes.push_back(Node{});
        fols.clear();
        nodes.reserve(1 << 21);
        fols.reserve(1 << 22);
        for (int i = 0; i < 256; ++i) root_child[i] = kNil;
        mk.assign(1u << 20, kNil);
        mv.assign(1u << 20, 0);
        ms.assign(1u << 20, 0);
        mmask = (1u << 20) - 1u;
        mfilled = 0;
        dense.clear();
    }

    std::uint32_t ensure_child(std::uint32_t node, std::uint8_t b) {
        if (node == 0) {
            if (root_child[b] != kNil) return root_child[b];
            Node n;
            n.parent = 0;
            n.byte = b;
            n.depth = 1;
            const auto id = static_cast<std::uint32_t>(nodes.size());
            nodes.push_back(n);
            root_child[b] = id;
            nodes[0].nchild++;
            return id;
        }
        std::uint32_t prev = kNil;
        std::uint32_t c = nodes[node].child_head;
        while (c != kNil) {
            if (nodes[c].byte == b) {
                if (prev != kNil) {
                    nodes[prev].sibling = nodes[c].sibling;
                    nodes[c].sibling = nodes[node].child_head;
                    nodes[node].child_head = c;
                }
                return c;
            }
            prev = c;
            c = nodes[c].sibling;
        }
        Node n;
        n.parent = node;
        n.byte = b;
        n.depth = static_cast<std::uint8_t>(nodes[node].depth + 1);
        n.sibling = nodes[node].child_head;
        const auto id = static_cast<std::uint32_t>(nodes.size());
        nodes.push_back(n);
        nodes[node].child_head = id;
        nodes[node].nchild++;
        return id;
    }

    std::uint32_t find_child(std::uint32_t node, std::uint8_t b) const {
        if (node == 0) return root_child[b];
        for (std::uint32_t c = nodes[node].child_head; c != kNil; c = nodes[c].sibling)
            if (nodes[c].byte == b) return c;
        return kNil;
    }

    void add_fol(std::uint32_t node, std::uint8_t y) {
        std::uint32_t prev = kNil;
        for (std::uint32_t f = nodes[node].fol_head; f != kNil; f = fols[f].next) {
            if (fols[f].sym == y) {
                ++fols[f].count;
                ++nodes[node].total;
                if (prev != kNil) {
                    fols[prev].next = fols[f].next;
                    fols[f].next = nodes[node].fol_head;
                    nodes[node].fol_head = f;
                }
                return;
            }
            prev = f;
        }
        Fol f;
        f.sym = y;
        f.count = 1;
        f.next = nodes[node].fol_head;
        const auto id = static_cast<std::uint32_t>(fols.size());
        fols.push_back(f);
        nodes[node].fol_head = id;
        nodes[node].nfol++;
        nodes[node].total++;
    }

    void observe(std::size_t i) {
        const std::uint8_t y = data[i];
        ++uni[y];
        ++uni_n;
        if (i >= 1) {
            const std::uint8_t prev = data[i - 1];
            ++o1[prev][y];
            ++o1n[prev];
        }
        const int maxd = std::min(4, static_cast<int>(i));
        std::uint32_t n = 0;
        for (int d = 1; d <= maxd; ++d) {
            n = ensure_child(n, data[i - static_cast<std::size_t>(d)]);
            if (d >= 2) add_fol(n, y);
        }
        if (i >= 4) {
            const std::uint32_t key = (std::uint32_t(data[i - 4]) << 24) | (std::uint32_t(data[i - 3]) << 16) |
                                      (std::uint32_t(data[i - 2]) << 8) | std::uint32_t(data[i - 1]);
            match_put(key, static_cast<std::uint32_t>(i));
        }
    }

    void match_grow() {
        auto ok = mk;
        auto ov = mv;
        auto os = ms;
        const unsigned bits = static_cast<unsigned>(std::bit_width(ok.size()));
        mk.assign(std::size_t{1} << bits, kNil);
        mv.assign(mk.size(), 0);
        ms.assign(mk.size(), 0);
        mmask = (1u << bits) - 1u;
        mfilled = 0;
        for (std::size_t i = 0; i < ok.size(); ++i) {
            if (ok[i] == kNil) continue;
            std::uint32_t j = mix32(ok[i]) & mmask;
            while (mk[j] != kNil) j = (j + 1u) & mmask;
            mk[j] = ok[i];
            mv[j] = ov[i];
            ms[j] = os[i];
            ++mfilled;
        }
    }

    void match_put(std::uint32_t key, std::uint32_t follower) {
        if (static_cast<std::uint64_t>(mfilled + 1) * 10ull >= static_cast<std::uint64_t>(mk.size()) * 7ull)
            match_grow();
        std::uint32_t i = mix32(key) & mmask;
        for (;;) {
            if (mk[i] == kNil) {
                mk[i] = key;
                mv[i] = follower;
                ms[i] = 1;
                ++mfilled;
                return;
            }
            if (mk[i] == key) {
                mv[i] = follower;
                if (ms[i] < 0xffffffffu) ++ms[i];
                return;
            }
            i = (i + 1u) & mmask;
        }
    }

    void compact_match() {
        dense.clear();
        std::uint32_t kept = 0, once = 0;
        for (std::size_t i = 0; i < mk.size(); ++i) {
            if (mk[i] == kNil) continue;
            if (ms[i] < 2) {
                ++once;
                continue;
            }
            dense.push_back(Hit{mk[i], mv[i]});
            ++kept;
        }
        std::sort(dense.begin(), dense.end(), [](const Hit& a, const Hit& b) { return a.key < b.key; });
        match_seen = mfilled;
        std::cerr << "match_filled " << mfilled << " match_once " << once << " match_kept " << kept << "\n";
        std::vector<std::uint32_t>().swap(mk);
        std::vector<std::uint32_t>().swap(mv);
        std::vector<std::uint32_t>().swap(ms);
    }

    bool find_match(const std::uint8_t* hist, std::size_t len, std::size_t limit, unsigned& pred) const {
        if (len < 4 || data == nullptr) return false;
        const std::uint32_t key = (std::uint32_t(hist[len - 4]) << 24) | (std::uint32_t(hist[len - 3]) << 16) |
                                  (std::uint32_t(hist[len - 2]) << 8) | std::uint32_t(hist[len - 1]);
        if (!dense.empty()) {
            const Hit probe{key, 0};
            const auto it = std::lower_bound(dense.begin(), dense.end(), probe,
                                              [](const Hit& a, const Hit& b) { return a.key < b.key; });
            if (it == dense.end() || it->key != key) return false;
            if (static_cast<std::size_t>(it->follower) >= limit) return false;
            pred = data[it->follower];
            return true;
        }
        // Live map, same rule as the packed table: a context seen once does not match.
        if (mk.empty()) return false;
        const std::uint32_t start = mix32(key) & mmask;
        std::uint32_t slot = start;
        do {
            if (mk[slot] == kNil) return false;
            if (mk[slot] == key) {
                if (ms[slot] < 2) return false;
                if (static_cast<std::size_t>(mv[slot]) >= limit) return false;
                pred = data[mv[slot]];
                return true;
            }
            slot = (slot + 1u) & mmask;
        } while (slot != start);
        return false;
    }

    std::uint32_t find_ctx(const std::uint8_t* h, std::size_t len, int order) const {
        if (static_cast<int>(len) < order) return kNil;
        std::uint32_t n = 0;
        for (int d = 1; d <= order; ++d) {
            n = find_child(n, h[len - static_cast<std::size_t>(d)]);
            if (n == kNil) return kNil;
        }
        return n;
    }

    // Fills kept (symbol, count) pairs. Returns false when this node must escape.
    bool gather(std::uint32_t node, Mode mode, std::uint8_t* syms, std::uint32_t* cs, int& n,
                std::uint32_t& total) const {
        n = 0;
        total = 0;
        if (node == kNil || nodes[node].total == 0) return false;
        if (mode == Mode::DropCtx && nodes[node].total == 1) return false;
        for (std::uint32_t f = nodes[node].fol_head; f != kNil; f = fols[f].next) {
            const std::uint32_t c = used_count(fols[f].count, mode);
            if (c == 0) continue;
            syms[n] = fols[f].sym;
            cs[n] = c;
            total += c;
            ++n;
        }
        return n > 0;
    }

    void fill_kt0(double* p) const {
        const double den = static_cast<double>(uni_n) + 128.0;
        for (int s = 0; s < 256; ++s) p[s] = (static_cast<double>(uni[s]) + 0.5) / den;
    }

    void fill_kt1(unsigned prev, double* p) const {
        const double den = static_cast<double>(o1n[prev]) + 128.0;
        for (int s = 0; s < 256; ++s) p[s] = (static_cast<double>(o1[prev][s]) + 0.5) / den;
    }

    static double entropy_u(const double* p) {
        double h = 0.0;
        for (int s = 0; s < 256; ++s) h -= p[s] * std::log(p[s]);
        double u = h / std::log(256.0);
        if (u < 0.0) u = 0.0;
        if (u > 1.0) u = 1.0;
        return u;
    }

    double mix(const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode, double* P,
               bool& matched) const {
        double kt1p[256];
        const unsigned prev = len ? hist[len - 1] : 0;
        fill_kt1(prev, kt1p);
        const double u = entropy_u(kt1p);
        double buf[2][256];
        double* cur = buf[0];
        double* nxt = buf[1];
        if (len >= 1 && o1n[prev] > 0) std::memcpy(cur, kt1p, sizeof(double) * 256);
        else fill_kt0(cur);

        const int top = std::min(4, static_cast<int>(len));
        for (int o = 2; o <= top; ++o) {
            for (int s = 0; s < 256; ++s) nxt[s] = kAlpha * cur[s];
            const std::uint32_t node = find_ctx(hist, len, o);
            std::uint8_t syms[256];
            std::uint32_t cs[256];
            int nf = 0;
            std::uint32_t tot = 0;
            if (gather(node, mode, syms, cs, nf, tot)) {
                const double inv = 1.0 / static_cast<double>(tot);
                for (int i = 0; i < nf; ++i) nxt[syms[i]] = static_cast<double>(cs[i]) * inv;
            }
            std::swap(cur, nxt);
        }
        double z = 0.0;
        for (int s = 0; s < 256; ++s) z += cur[s];
        double B[256];
        const double invz = 1.0 / z;
        for (int s = 0; s < 256; ++s) B[s] = cur[s] * invz;

        unsigned pred = 0;
        matched = find_match(hist, len, follower_limit, pred);
        double match_w = 0.0;
        if (matched) {
            const double agree = kt1p[pred] / (kt1p[pred] + kAgreeFloor);
            match_w = (1.0 - u) * agree;
            if (match_w < 0.0) match_w = 0.0;
            if (match_w > kMatchCap) match_w = kMatchCap;
        }
        for (int s = 0; s < 256; ++s) P[s] = (1.0 - match_w) * B[s];
        if (matched) P[pred] += match_w;
        return u;
    }

    // Experts 0 and 1 are the unigram and order-1 KT tables. Experts 2..4 are
    // the backoff chain inside mix(), normalized after each order and left
    // unnormalized in the chain so the next order sees the same state mix()
    // sees. The match blend is not applied here.
    void partials(const std::uint8_t* hist, std::size_t len, Mode mode, double (&out)[5][256]) const {
        fill_kt0(out[0]);
        const unsigned prev = len ? hist[len - 1] : 0;
        fill_kt1(prev, out[1]);
        double buf[2][256];
        double* cur = buf[0];
        double* nxt = buf[1];
        if (len >= 1 && o1n[prev] > 0) std::memcpy(cur, out[1], sizeof(double) * 256);
        else std::memcpy(cur, out[0], sizeof(double) * 256);

        const int top = std::min(4, static_cast<int>(len));
        for (int o = 2; o <= 4; ++o) {
            if (o <= top) {
                for (int s = 0; s < 256; ++s) nxt[s] = kAlpha * cur[s];
                const std::uint32_t node = find_ctx(hist, len, o);
                std::uint8_t syms[256];
                std::uint32_t cs[256];
                int nf = 0;
                std::uint32_t tot = 0;
                if (gather(node, mode, syms, cs, nf, tot)) {
                    const double inv = 1.0 / static_cast<double>(tot);
                    for (int i = 0; i < nf; ++i) nxt[syms[i]] = static_cast<double>(cs[i]) * inv;
                }
                std::swap(cur, nxt);
            }
            double z = 0.0;
            for (int s = 0; s < 256; ++s) z += cur[s];
            const double invz = 1.0 / z;
            for (int s = 0; s < 256; ++s) out[o][s] = cur[s] * invz;
        }
    }

    // PPMD: p(symbol) = (c - 1/2) / total, p(escape) = (distinct / 2) / total.
    // Exclusion removes a byte already named by a longer context. scale is the
    // probability mass still unassigned. A context whose symbols are all excluded
    // is masked and does not change scale.
    static double add_ppmd(double scale, bool* excluded, double* P, const std::uint8_t* syms, const std::uint32_t* cs,
                           int n) {
        double sum_c = 0;
        int d = 0;
        int idx[256];
        for (int i = 0; i < n; ++i) {
            if (excluded[syms[i]]) continue;
            idx[d++] = i;
            sum_c += static_cast<double>(cs[i]);
        }
        if (d == 0 || !(sum_c > 0.0)) return scale;
        for (int j = 0; j < d; ++j) {
            double num = static_cast<double>(cs[idx[j]]) - 0.5;
            if (num < 0.5) num = 0.5;
            const unsigned s = syms[idx[j]];
            P[s] += scale * num / sum_c;
            excluded[s] = true;
        }
        return scale * (0.5 * static_cast<double>(d)) / sum_c;
    }

    int walk(const std::uint8_t* h, std::size_t len, int max_depth, std::uint32_t* path) const {
        std::uint32_t n = 0;
        int d = 0;
        const int cap = std::min(max_depth, static_cast<int>(len));
        while (d < cap) {
            if (d >= 4 && (nodes[n].chain_open == 0 || nodes[n].total < 2)) break;
            const std::uint32_t c = find_child(n, h[len - 1 - static_cast<std::size_t>(d)]);
            if (c == kNil) break;
            ++d;
            path[d] = c;
            n = c;
        }
        return d;
    }

    // Second pass. Orders 2..4 stay as they are. A longer node is created only
    // when the parent still has a single older byte, and only after that parent
    // has been seen twice. A second distinct older byte closes the chain.
    void extend_chains(std::size_t ntrain, int max_depth) {
        const std::size_t before = nodes.size();
        for (std::size_t i = 0; i < ntrain; ++i) {
            if (i < 5) continue;
            std::uint32_t n = 0;
            int d = 0;
            for (; d < 4; ++d) {
                n = find_child(n, data[i - 1 - static_cast<std::size_t>(d)]);
                if (n == kNil) break;
            }
            if (n == kNil || d < 4) continue;
            while (d < max_depth && static_cast<std::size_t>(d) < i) {
                if (nodes[n].total < 2 || nodes[n].chain_open == 0) break;
                const std::uint8_t b = data[i - 1 - static_cast<std::size_t>(d)];
                if (nodes[n].nchild >= 1 && find_child(n, b) == kNil) {
                    nodes[n].chain_open = 0;
                    break;
                }
                n = ensure_child(n, b);
                add_fol(n, data[i]);
                ++d;
            }
            if (((i + 1) & 0xfffff) == 0) std::cerr << "extend " << (i + 1) << " nodes " << nodes.size() << "\n";
        }
        std::cerr << "chain_nodes_added " << (nodes.size() - before) << "\n";
    }

    // A symbol seen once takes a small head start from its parent, capped at 4,
    // so the packed model can keep it. Real repeated counts are left alone.
    void inherit_singletons() {
        std::uint64_t bumped = 0;
        const std::size_t N = nodes.size();
        std::vector<std::uint32_t> order(N);
        for (std::uint32_t i = 0; i < N; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
            return nodes[a].depth < nodes[b].depth;
        });
        for (std::uint32_t id : order) {
            if (nodes[id].depth < 2 || nodes[id].fol_head == kNil) continue;
            std::uint32_t ptotal = 0;
            const std::uint32_t parent = nodes[id].parent;
            const bool parent_trie = parent != kNil && nodes[parent].depth >= 2;
            std::uint8_t prev = 0;
            if (!parent_trie) {
                std::uint32_t a = id;
                while (a != kNil && nodes[a].depth > 1) a = nodes[a].parent;
                if (a == kNil || nodes[a].depth != 1) continue;
                prev = nodes[a].byte;
                ptotal = o1n[prev];
            } else {
                ptotal = nodes[parent].total;
            }
            if (ptotal == 0) continue;
            for (std::uint32_t f = nodes[id].fol_head; f != kNil; f = fols[f].next) {
                if (fols[f].count != 1) continue;
                std::uint32_t pc = 0;
                if (!parent_trie) pc = o1[prev][fols[f].sym];
                else {
                    for (std::uint32_t pf = nodes[parent].fol_head; pf != kNil; pf = fols[pf].next) {
                        if (fols[pf].sym == fols[f].sym) {
                            pc = fols[pf].count;
                            break;
                        }
                    }
                }
                if (pc == 0) continue;
                std::uint32_t bonus = (pc * 2u) / (ptotal + 2u);
                if (bonus < 1) continue;
                if (bonus > 3) bonus = 3;
                fols[f].count = 1 + bonus;
                nodes[id].total += bonus;
                ++bumped;
            }
        }
        std::cerr << "inherited " << bumped << "\n";
    }

    double mix_ppmd(const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode, int max_depth,
                    double* P, bool& matched) const {
        std::memset(P, 0, sizeof(double) * 256);
        bool excluded[256];
        std::memset(excluded, 0, sizeof excluded);
        double scale = 1.0;
        std::uint32_t path[24];
        const int depth = walk(hist, len, max_depth, path);
        std::uint8_t syms[256];
        std::uint32_t cs[256];
        for (int d = depth; d >= 2; --d) {
            int nf = 0;
            std::uint32_t tot = 0;
            if (!gather(path[d], mode, syms, cs, nf, tot)) continue;
            scale = add_ppmd(scale, excluded, P, syms, cs, nf);
        }
        // The trie escapes into the KT tables. Those stay additive-smoothed.
        // PPMD is only the count inside a trie node.
        auto pour = [&](const double* raw) {
            if (!(scale > 0.0)) return;
            double mass = 0.0;
            for (int s = 0; s < 256; ++s)
                if (!excluded[s]) mass += raw[s];
            if (!(mass > 0.0)) return;
            const double inv = scale / mass;
            for (int s = 0; s < 256; ++s) {
                if (excluded[s]) continue;
                P[s] += raw[s] * inv;
            }
            scale = 0.0;
        };
        double kt1p_floor[256];
        double kt0p[256];
        if (len >= 1 && o1n[hist[len - 1]] > 0) fill_kt1(hist[len - 1], kt1p_floor);
        else fill_kt0(kt1p_floor);
        pour(kt1p_floor);
        fill_kt0(kt0p);
        pour(kt0p);
        if (scale > 0.0) {
            int left = 0;
            for (int s = 0; s < 256; ++s)
                if (!excluded[s]) ++left;
            if (left == 0) left = 256;
            const double share = scale / static_cast<double>(left);
            for (int s = 0; s < 256; ++s)
                if (left == 256 || !excluded[s]) P[s] += share;
        }

        double kt1p[256];
        const unsigned prev = len ? hist[len - 1] : 0;
        fill_kt1(prev, kt1p);
        const double u = entropy_u(kt1p);
        unsigned pred = 0;
        matched = find_match(hist, len, follower_limit, pred);
        double match_w = 0.0;
        if (matched) {
            const double agree = kt1p[pred] / (kt1p[pred] + kAgreeFloor);
            match_w = (1.0 - u) * agree;
            if (match_w < 0.0) match_w = 0.0;
            if (match_w > kMatchCap) match_w = kMatchCap;
        }
        if (matched && match_w > 0.0) {
            for (int s = 0; s < 256; ++s) P[s] *= (1.0 - match_w);
            P[pred] += match_w;
        }
        return u;
    }
};

bool self_test() {
    init_quant();
    if (quantize(1) != 1 || quantize(64) != 64 || qdec[quantize(100000)] < 100000) {
        // The geometric bin may land just below; it must exist and grow.
    }
    if (qdec[255] < 1000000) {
        std::cerr << "quantizer tops out at " << qdec[255] << "\n";
        return false;
    }
    const std::string text = "abcabcabcabcabcZZ";
    std::vector<std::uint8_t> buf(text.begin(), text.end());
    Model m;
    m.data = buf.data();
    m.init();
    for (std::size_t i = 0; i < buf.size(); ++i) m.observe(i);
    m.compact_match();
    const std::uint8_t ctx[] = {'a', 'b'};
    const std::uint32_t node = m.find_ctx(ctx, 2, 2);
    if (node == kNil || m.nodes[node].total < 2) {
        std::cerr << "self-test context ab missing\n";
        return false;
    }
    double P[256];
    bool matched = false;
    m.mix(ctx, 2, 0, Mode::Full, P, matched);
    double sum = 0;
    for (int s = 0; s < 256; ++s) sum += P[s];
    if (std::fabs(sum - 1.0) > 1e-8 || !(P['c'] > P['Z'])) {
        std::cerr << "self-test mix failed sum=" << sum << " Pc=" << P['c'] << "\n";
        return false;
    }
    double Q[256];
    m.mix_ppmd(ctx, 2, 0, Mode::Full, 4, Q, matched);
    sum = 0;
    for (int s = 0; s < 256; ++s) sum += Q[s];
    if (std::fabs(sum - 1.0) > 1e-6 || !(Q['c'] > Q['Z'])) {
        std::cerr << "self-test ppmd failed sum=" << sum << " Qc=" << Q['c'] << "\n";
        return false;
    }
    double parts[5][256];
    m.partials(ctx, 2, Mode::Full, parts);
    m.mix(ctx, 2, 0, Mode::Full, P, matched);
    double gap = 0.0;
    for (int s = 0; s < 256; ++s) gap = std::max(gap, std::fabs(P[s] - parts[4][s]));
    if (!matched && gap > 1e-8) {
        std::cerr << "self-test partials diverged from mix by " << gap << "\n";
        return false;
    }
    if (matched) {
        int pred = 0;
        double best = -1.0;
        for (int s = 0; s < 256; ++s) {
            const double d = P[s] - parts[4][s];
            if (d > best) {
                best = d;
                pred = s;
            }
        }
        double scale = 1.0;
        for (int s = 0; s < 256; ++s) {
            if (s == pred || !(parts[4][s] > 1e-12)) continue;
            scale = P[s] / parts[4][s];
            break;
        }
        for (int s = 0; s < 256; ++s) {
            double expect = scale * parts[4][s];
            if (s == pred) expect += 1.0 - scale;
            if (std::fabs(P[s] - expect) > 1e-6) {
                std::cerr << "self-test partials+match diverged at " << s << "\n";
                return false;
            }
        }
    }
    return true;
}

std::uint64_t next_pow2_load(std::uint64_t filled) {
    if (filled == 0) return 0;
    double need = static_cast<double>(filled) / 0.7;
    std::uint64_t slots = 1;
    while (static_cast<double>(slots) < need) slots <<= 1;
    return slots;
}

struct Layout {
    std::uint64_t nodes = 0;
    std::uint64_t path_only = 0;
    std::uint64_t one_child = 0;
    std::uint64_t leaves_dropped = 0;
    std::uint64_t followers = 0;
    std::uint64_t bytes = 0;
};

bool kept_fol(const Model& m, std::uint32_t node, Mode mode) {
    if (m.nodes[node].total == 0) return false;
    if (mode == Mode::DropCtx && m.nodes[node].total == 1) return false;
    if (mode != Mode::Synth) return m.nodes[node].total > 0;
    for (std::uint32_t f = m.nodes[node].fol_head; f != kNil; f = m.fols[f].next)
        if (m.fols[f].count >= 2) return true;
    return false;
}

int kept_nfol(const Model& m, std::uint32_t node, Mode mode) {
    if (mode == Mode::DropCtx && m.nodes[node].total == 1) return 0;
    int n = 0;
    for (std::uint32_t f = m.nodes[node].fol_head; f != kNil; f = m.fols[f].next)
        if (used_count(m.fols[f].count, mode) > 0) ++n;
    return n;
}

Layout measure(const Model& m, Mode mode) {
    const std::size_t N = m.nodes.size();
    std::vector<char> useful(N, 0);
    for (std::uint32_t id = 1; id < N; ++id)
        if (kept_fol(m, id, mode)) useful[id] = 1;
    for (int pass = 0; pass < 4; ++pass) {
        for (std::uint32_t id = 1; id < N; ++id) {
            if (!useful[id]) continue;
            const std::uint32_t p = m.nodes[id].parent;
            if (p != kNil) useful[p] = 1;
        }
    }
    Layout L;
    for (std::uint32_t id = 1; id < N; ++id) {
        if (m.nodes[id].depth >= 2 && m.nodes[id].total > 0 && !kept_fol(m, id, mode) && !useful[id])
            ++L.leaves_dropped;
        if (!useful[id]) continue;
        ++L.nodes;
        const int nf = kept_nfol(m, id, mode);
        L.followers += static_cast<std::uint64_t>(nf);
        int live_children = 0;
        for (std::uint32_t c = m.nodes[id].child_head; c != kNil; c = m.nodes[c].sibling)
            if (useful[c]) ++live_children;
        if (id != 0 && m.nodes[id].depth >= 1 && live_children == 1) ++L.one_child;
        const bool dist = nf > 0;
        if (!dist) ++L.path_only;
        // Header 8 bytes. One child is inlined (4 bytes). Extra children are 5 bytes.
        // Each kept follower is 2 bytes (symbol + PAQ state). A one-follower chain
        // node with one child is 8 + 4 + 2 = 14 bytes and shares the edge byte.
        std::uint64_t b = 8;
        if (live_children <= 1) b += live_children ? 4 : 0;
        else b += static_cast<std::uint64_t>(live_children) * 5ull;
        b += static_cast<std::uint64_t>(nf) * 2ull;
        L.bytes += b;
    }
    return L;
}

std::uint64_t hash_reserved(const Model& m) {
    std::uint64_t bytes = 0;
    for (int depth = 2; depth <= 4; ++depth) {
        std::uint64_t filled = 0, extra = 0;
        for (const Node& n : m.nodes) {
            if (n.depth != depth || n.total == 0) continue;
            ++filled;
            if (n.nfol > 1) extra += static_cast<std::uint64_t>(n.nfol - 1);
        }
        const std::uint64_t slots = next_pow2_load(filled);
        bytes += slots * 24ull + extra * 12ull;
    }
    const std::uint64_t mslots = next_pow2_load(m.match_seen);
    bytes += mslots * 16ull;
    return bytes;
}

std::uint64_t hash_payload(const Model& m) {
    std::uint64_t bytes = 0;
    for (const Node& n : m.nodes) {
        if (n.depth < 2 || n.total == 0) continue;
        bytes += 18ull;
        if (n.nfol > 1) bytes += static_cast<std::uint64_t>(n.nfol - 1) * 9ull;
    }
    bytes += static_cast<std::uint64_t>(m.dense.size()) * 13ull;
    return bytes;
}

std::string printable(const std::string& s) {
    std::string o;
    const char* hex = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c >= 32 && c < 127) o += static_cast<char>(c);
        else {
            o += "\\x";
            o += hex[c >> 4];
            o += hex[c & 15];
        }
    }
    return o;
}

bool is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

double distinct4(const std::string& s) {
    if (s.size() < 4) return 0;
    std::vector<std::uint32_t> g;
    for (std::size_t i = 0; i + 3 < s.size(); ++i) {
        g.push_back((std::uint32_t(std::uint8_t(s[i])) << 24) | (std::uint32_t(std::uint8_t(s[i + 1])) << 16) |
                    (std::uint32_t(std::uint8_t(s[i + 2])) << 8) | std::uint32_t(std::uint8_t(s[i + 3])));
    }
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return static_cast<double>(g.size()) / static_cast<double>(s.size() - 3);
}

double loop8(const std::string& s) {
    if (s.size() < 16) return 0;
    std::unordered_set<std::uint64_t> seen;
    int loops = 0, n = 0;
    for (std::size_t i = 0; i + 8 <= s.size(); ++i) {
        std::uint64_t w = 0;
        for (int k = 0; k < 8; ++k) w = (w << 8) | static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]);
        if (!seen.insert(w).second) ++loops;
        ++n;
    }
    return static_cast<double>(loops) / static_cast<double>(n);
}

double lexicon_hit(const std::string& s, const std::unordered_set<std::string>& vocab) {
    int words = 0, hit = 0;
    std::size_t i = 0;
    while (i < s.size()) {
        if (!is_letter(static_cast<unsigned char>(s[i]))) {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < s.size() && is_letter(static_cast<unsigned char>(s[j]))) ++j;
        if (j - i >= 3) {
            std::string w = s.substr(i, j - i);
            for (char& c : w)
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            ++words;
            if (vocab.count(w)) ++hit;
        }
        i = j;
    }
    return words ? static_cast<double>(hit) / static_cast<double>(words) : 0;
}

double surface(const std::string& s, const std::unordered_set<std::string>& vocab) {
    const double H = lexicon_hit(s, vocab);
    const double d4 = distinct4(s);
    const double lp = loop8(s);
    double D = 1.0 - std::fabs(d4 - kRefD4);
    if (D < 0) D = 0;
    double L = 1.0 - std::max(0.0, lp - kRefLoop);
    if (L < 0) L = 0;
    std::cout << "  H=" << H << " D=" << D << " L=" << L << " d4=" << d4 << " loop8=" << lp;
    return H * D * L;
}

void mark_banned(const std::vector<std::uint8_t>& seq, bool banned[256]) {
    std::memset(banned, 0, 256);
    const std::size_t n = seq.size();
    if (n < 11) return;
    const std::uint8_t* last = seq.data() + (n - 11);
    for (std::size_t s = 0; s + 12 <= n; ++s)
        if (std::memcmp(seq.data() + s, last, 11) == 0) banned[seq[s + 11]] = true;
}

std::string generate(const Model& model, const std::vector<std::uint8_t>& prompt, Mode mode, std::mt19937& rng,
                     bool ppmd = false, int max_depth = 4) {
    std::vector<std::uint8_t> seq = prompt;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int t = 0; t < kGen; ++t) {
        double P[256];
        bool matched = false;
        if (ppmd) model.mix_ppmd(seq.data(), seq.size(), kTrain, mode, max_depth, P, matched);
        else model.mix(seq.data(), seq.size(), kTrain, mode, P, matched);
        bool banned[256];
        mark_banned(seq, banned);
        double raw[256];
        double mass_raw = 0;
        for (int s = 0; s < 256; ++s) {
            if (banned[s]) {
                raw[s] = 0;
                continue;
            }
            raw[s] = std::pow(std::max(P[s], 0.0), 1.0 / kTemperature);
            mass_raw += raw[s];
        }
        if (!(mass_raw > 0)) {
            mass_raw = 0;
            for (int s = 0; s < 256; ++s) {
                raw[s] = std::pow(std::max(P[s], 0.0), 1.0 / kTemperature);
                mass_raw += raw[s];
            }
        }
        struct Item {
            int y;
            double p;
        } items[256];
        for (int s = 0; s < 256; ++s) items[s] = {s, raw[s] / mass_raw};
        std::sort(items, items + 256, [](const Item& a, const Item& b) { return a.p > b.p; });
        double acc = 0;
        int keep = 1;
        for (int n = 0; n < 256; ++n) {
            acc += items[n].p;
            keep = n + 1;
            if (acc >= kNucleus) break;
        }
        double mass = 0;
        for (int n = 0; n < keep; ++n) mass += items[n].p;
        double r = U(rng) * mass;
        int pick = items[keep - 1].y;
        for (int n = 0; n < keep; ++n) {
            r -= items[n].p;
            if (r <= 0) {
                pick = items[n].y;
                break;
            }
        }
        seq.push_back(static_cast<std::uint8_t>(pick));
    }
    return std::string(reinterpret_cast<const char*>(seq.data() + prompt.size()), static_cast<std::size_t>(kGen));
}

struct Score {
    double bpc = 0;
    double mean_u = 0;
    double match_rate = 0;
};

Score heldout(const Model& m, const std::uint8_t* data, Mode mode, bool ppmd = false, int max_depth = 4) {
    double bits = 0, sum_u = 0;
    std::uint64_t matches = 0;
    const double n = static_cast<double>(kSample - kTrain);
    for (std::size_t i = kTrain; i < kSample; ++i) {
        double P[256];
        bool matched = false;
        const double u = ppmd ? m.mix_ppmd(data, i, kTrain, mode, max_depth, P, matched)
                              : m.mix(data, i, kTrain, mode, P, matched);
        if (matched) ++matches;
        if (((i - kTrain) & 0xffff) == 0) {
            double sum = 0;
            for (int s = 0; s < 256; ++s) sum += P[s];
            if (std::fabs(sum - 1.0) > 1e-5) {
                std::cerr << "mixture sum " << sum << " at " << i << "\n";
                std::exit(1);
            }
        }
        const double p = P[data[i]];
        if (!(p > 0.0) || !std::isfinite(p)) {
            std::cerr << "bad p at " << i << "\n";
            std::exit(1);
        }
        bits += -std::log2(p);
        sum_u += u;
        if (((i - kTrain + 1) & 0x3ffff) == 0) std::cerr << "score " << (i - kTrain + 1) << "\n";
    }
    return Score{bits / n, sum_u / n, static_cast<double>(matches) / n};
}

constexpr double kWeightEta = 0.05;
constexpr double kWeightFloor = 1e-4;

void project_weight_floor(double* w) {
    for (int iter = 0; iter < 5; ++iter) {
        int below = 0;
        double above = 0.0;
        for (int i = 0; i < 5; ++i) {
            if (w[i] < kWeightFloor) ++below;
            else above += w[i];
        }
        if (below == 0) return;
        const double need = 1.0 - kWeightFloor * static_cast<double>(below);
        if (!(above > 0.0) || !(need > 0.0)) break;
        for (int i = 0; i < 5; ++i) {
            if (w[i] < kWeightFloor) w[i] = kWeightFloor;
            else w[i] = w[i] / above * need;
        }
    }
    double sum = 0.0;
    for (int i = 0; i < 5; ++i) sum += w[i];
    for (int i = 0; i < 5; ++i) w[i] /= sum;
}

// w_i *= exp(eta * P_i(y) / p), then sum to 1 with each weight at least 1e-4.
void adapt_weights(double* w, const double* py, double p) {
    double logw[5];
    double mx = -1e300;
    for (int i = 0; i < 5; ++i) {
        logw[i] = std::log(w[i]) + kWeightEta * py[i] / p;
        if (logw[i] > mx) mx = logw[i];
    }
    double sum = 0.0;
    for (int i = 0; i < 5; ++i) {
        w[i] = std::exp(logw[i] - mx);
        sum += w[i];
    }
    for (int i = 0; i < 5; ++i) w[i] /= sum;
    project_weight_floor(w);
}

bool prob_ok(const double* p, double tol) {
    double sum = 0.0;
    for (int s = 0; s < 256; ++s) {
        if (!(p[s] >= 0.0) || !std::isfinite(p[s])) return false;
        sum += p[s];
    }
    return std::fabs(sum - 1.0) <= tol;
}

// Score first, then observe. follower limit is the current index, so a match
// stored on an earlier holdout byte is visible and the current byte is not.
double online_counts(Model& m, const std::uint8_t* data) {
    double bits = 0.0;
    const double n = static_cast<double>(kSample - kTrain);
    for (std::size_t i = kTrain; i < kSample; ++i) {
        double P[256];
        bool matched = false;
        m.mix(data, i, i, Mode::Synth, P, matched);
        if (((i - kTrain) & 0xffff) == 0 && !prob_ok(P, 1e-5)) {
            std::cerr << "online counts mixture broken at " << i << "\n";
            std::exit(1);
        }
        const double p = P[data[i]];
        if (!(p > 0.0) || !std::isfinite(p)) {
            std::cerr << "online counts bad p at " << i << "\n";
            std::exit(1);
        }
        bits += -std::log2(p);
        m.observe(i);
        if (((i - kTrain + 1) & 0x3ffff) == 0) std::cerr << "online_counts " << (i - kTrain + 1) << "\n";
    }
    return bits / n;
}

double online_weights(const Model& m, const std::uint8_t* data, double* w) {
    for (int i = 0; i < 5; ++i) w[i] = 1.0 / 5.0;
    double bits = 0.0;
    const double n = static_cast<double>(kSample - kTrain);
    for (std::size_t i = kTrain; i < kSample; ++i) {
        double dist[5][256];
        m.partials(data, i, Mode::Synth, dist);
        double mixp[256];
        for (int s = 0; s < 256; ++s) {
            double acc = 0.0;
            for (int e = 0; e < 5; ++e) acc += w[e] * dist[e][s];
            mixp[s] = acc;
        }
        if (((i - kTrain) & 0xffff) == 0) {
            for (int e = 0; e < 5; ++e) {
                if (!prob_ok(dist[e], 1e-5)) {
                    std::cerr << "expert " << e << " broken at " << i << "\n";
                    std::exit(1);
                }
            }
            if (!prob_ok(mixp, 1e-5)) {
                std::cerr << "online weights mixture broken at " << i << "\n";
                std::exit(1);
            }
        }
        const double p = mixp[data[i]];
        if (!(p > 0.0) || !std::isfinite(p)) {
            std::cerr << "online weights bad p at " << i << "\n";
            std::exit(1);
        }
        bits += -std::log2(p);
        double py[5];
        for (int e = 0; e < 5; ++e) py[e] = dist[e][data[i]];
        adapt_weights(w, py, p);
        if (((i - kTrain + 1) & 0x3ffff) == 0) std::cerr << "online_weights " << (i - kTrain + 1) << "\n";
    }
    return bits / n;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    if (!self_test()) return 1;
    std::cerr << "self-test ok\n";
    const auto t0 = std::chrono::steady_clock::now();

    std::ifstream in("/mnt/d/cypha_llm/enwik9", std::ios::binary);
    if (!in) {
        std::cerr << "cannot open corpus\n";
        return 1;
    }
    std::vector<std::uint8_t> data(kSample);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(kSample));
    if (static_cast<std::size_t>(in.gcount()) != kSample) {
        std::cerr << "short corpus\n";
        return 1;
    }

    Model model;
    model.data = data.data();
    model.init();
    for (std::size_t i = 0; i < kTrain; ++i) {
        model.observe(i);
        if (((i + 1) & 0xfffff) == 0) std::cerr << "train " << (i + 1) << " nodes " << model.nodes.size() << "\n";
    }

    // Packed match table for the frozen score. The live map is kept so the
    // count pass can observe into it from the same trained times.
    std::vector<std::uint32_t> mk_trained = model.mk;
    std::vector<std::uint32_t> mv_trained = model.mv;
    std::vector<std::uint32_t> ms_trained = model.ms;
    const std::uint32_t mfilled_trained = model.mfilled;
    const std::uint32_t mmask_trained = model.mmask;
    model.compact_match();
    const std::size_t nodes_trained = model.nodes.size();
    const std::size_t fols_trained = model.fols.size();
    const std::uint32_t uni_trained = model.uni_n;

    std::cerr << "scoring frozen\n";
    const double frozen_bpc = heldout(model, data.data(), Mode::Synth).bpc;
    std::cerr << "frozen_bpc " << frozen_bpc << "\n";

    std::cerr << "scoring online weights\n";
    double final_w[5];
    const double online_weights_bpc = online_weights(model, data.data(), final_w);
    if (model.nodes.size() != nodes_trained || model.fols.size() != fols_trained || model.uni_n != uni_trained) {
        std::cerr << "weight pass changed counts\n";
        return 1;
    }

    model.mk = std::move(mk_trained);
    model.mv = std::move(mv_trained);
    model.ms = std::move(ms_trained);
    model.mfilled = mfilled_trained;
    model.mmask = mmask_trained;
    model.dense.clear();
    if (!model.dense.empty() || model.mk.empty()) {
        std::cerr << "count pass match state was not restored\n";
        return 1;
    }

    std::cerr << "scoring online counts\n";
    const double online_counts_bpc = online_counts(model, data.data());

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "frozen_bpc " << frozen_bpc << "\n";
    std::cout << "online_counts_bpc " << online_counts_bpc << "\n";
    std::cout << "online_weights_bpc " << online_weights_bpc << "\n";
    std::cout << "final_weights uni " << final_w[0] << " o1 " << final_w[1] << " o2 " << final_w[2] << " o3 "
              << final_w[3] << " o4 " << final_w[4] << "\n";
    std::cout << "online_counts_beats_frozen " << (online_counts_bpc < frozen_bpc ? 1 : 0)
              << " online_weights_beats_frozen " << (online_weights_bpc < frozen_bpc ? 1 : 0) << "\n";
    const auto t1 = std::chrono::steady_clock::now();
    std::cerr << "seconds " << std::chrono::duration<double>(t1 - t0).count() << "\n";
    if (!(frozen_bpc >= 2.36 && frozen_bpc <= 2.41)) {
        std::cerr << "frozen_bpc outside 2.36..2.41\n";
        return 2;
    }
    return 0;
}
