// CyphaLM packed byte trie: the one shared copy of the model (PLAN Track B1).
//
// Backoff orders 2..4 stored as a byte trie, plus a 4-byte match. The parent of
// a node is the shorter context, so a miss is the backoff. Three stored forms
// of one trained model:
//   Full      every observed count, 32-bit
//   DropCtx   contexts seen once are not used (escape to the parent)
//   Synth     symbol counts of 1 dropped, remaining counts packed into one
//             byte (quantiser below), one-child edges inlined, match kept only
//             when seen twice
//
// Stupid backoff alpha = 0.4. Order-1 and the unigram stay dense KT tables
// (0.25 MB). u = H(order-1) / ln(256).
// match_w = min(0.75, (1-u) * agree), agree = p/(p+0.05), times >= 2.
// Train [0, train), freeze, score [train, sample). Defaults: 7,340,032 and
// 8,388,608 (the first 8 MiB of enwik9; enwik8's first 8 MiB are the same bytes).
//
// A linear mixture learns on the training stream only. Each byte is scored from
// the counts so far, the weights move (eg_update), then the byte is observed.
// The holdout uses the frozen counts and the frozen weights. Experts: KT
// unigram, KT order-1, stupid backoff stopped after orders 2, 3 and 4
// (mix()'s cur buffer, normalised), and a one-hot 4-byte match when that match
// fires. An order the walk cannot reach reuses the shorter expert.
//
// What is here:        Quantiser, Node/Fol/Hit, Model (observe, fill_experts,
//                      mix, mix_dist, compact_match), eg_update, layout
//                      accounting (measure, hash_*), the Dolly loader, text
//                      helpers (printable), self_test, and sampling with the
//                      no-repeat rule.
// Negative results kept reproducible (CURRENT.md "PPMD, longer chains, or
// inheritance ~2.42"): Model::mix_ppmd, extend_chains, inherit_singletons.
// Rule 10 (PLAN): CyphaLM answers every task. There is no string-search
// answering here, and none may be added.
//
// Constants that used to be file-level (kSample, kTrain, kAlpha, ...) are fields
// of Config. Defaults equal the values the tools used before the consolidation.

#pragma once

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cypha::cyphalm::trie {

inline constexpr std::uint32_t kNil = 0xffffffffu;

struct Config {
    std::size_t sample = 8'388'608;  // bytes read from the corpus
    std::size_t train = 7'340'032;   // [0, train) trains; [train, sample) is the frozen holdout
    double alpha = 0.4;              // stupid-backoff factor
    double agree_floor = 0.05;       // match agreement p/(p+floor)
    double match_cap = 0.75;         // cap on the match weight in mix()
    double eta = 0.05;               // exponentiated-gradient step
    double w_floor = 1e-4;           // lower bound on a mixture weight
    double w_match0 = 0.15;          // initial weight of the match expert (E0..E4 share the rest)
    std::size_t reserve_nodes = std::size_t{1} << 21;
    std::size_t reserve_fols = std::size_t{1} << 22;
    unsigned match_bits = 20;        // initial match hash slots = 2^match_bits
};

enum class Mode { Full, DropCtx, Synth };

// ---- Quantiser: counts above 64 are stored in one byte on a 1.08 geometric ladder.

inline std::array<std::uint32_t, 256> make_qdec() {
    std::array<std::uint32_t, 256> q{};
    q[0] = 0;
    for (int s = 1; s <= 64; ++s) q[s] = static_cast<std::uint32_t>(s);
    double v = 64.0;
    for (int s = 65; s < 256; ++s) {
        v *= 1.08;
        auto n = static_cast<std::uint32_t>(std::llround(v));
        if (n <= q[s - 1]) n = q[s - 1] + 1;
        q[s] = n;
    }
    return q;
}

inline const std::array<std::uint32_t, 256> qdec = make_qdec();

inline std::uint8_t quantize(std::uint32_t c) {
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

inline std::uint32_t used_count(std::uint32_t c, Mode mode) {
    if (c == 0) return 0;
    if (mode == Mode::Synth) {
        if (c < 2) return 0;
        return qdec[quantize(c)];
    }
    return c;
}

// ---- Records

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

inline std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline std::uint32_t key4(const std::uint8_t* h, std::size_t len) {
    return (std::uint32_t(h[len - 4]) << 24) | (std::uint32_t(h[len - 3]) << 16) | (std::uint32_t(h[len - 2]) << 8) |
           std::uint32_t(h[len - 1]);
}

// ---- Model

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
    double alpha = 0.4, agree_floor = 0.05, match_cap = 0.75;
    bool verbose = true;  // compact_match / extend_chains / inherit_singletons log to stderr

    void init(const Config& c = Config{}) {
        alpha = c.alpha;
        agree_floor = c.agree_floor;
        match_cap = c.match_cap;
        nodes.clear();
        nodes.push_back(Node{});
        fols.clear();
        nodes.reserve(c.reserve_nodes);
        fols.reserve(c.reserve_fols);
        for (int i = 0; i < 256; ++i) root_child[i] = kNil;
        const std::size_t slots = std::size_t{1} << c.match_bits;
        mk.assign(slots, kNil);
        mv.assign(slots, 0);
        ms.assign(slots, 0);
        mmask = static_cast<std::uint32_t>(slots - 1u);
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
        if (i >= 4) match_put(key4(data, i), static_cast<std::uint32_t>(i));
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

    // Keep only matches seen at least twice, sorted by key; free the training hash.
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
        if (verbose) std::cerr << "match_filled " << mfilled << " match_once " << once << " match_kept " << kept << "\n";
        std::vector<std::uint32_t>().swap(mk);
        std::vector<std::uint32_t>().swap(mv);
        std::vector<std::uint32_t>().swap(ms);
    }

    bool find_match(const std::uint8_t* hist, std::size_t len, std::size_t limit, unsigned& pred) const {
        if (len < 4 || data == nullptr) return false;
        const std::uint32_t key = key4(hist, len);
        const Hit probe{key, 0};
        const auto it = std::lower_bound(dense.begin(), dense.end(), probe,
                                          [](const Hit& a, const Hit& b) { return a.key < b.key; });
        if (it == dense.end() || it->key != key) return false;
        if (static_cast<std::size_t>(it->follower) >= limit) return false;
        pred = data[it->follower];
        return true;
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

    // 4-byte match against contexts already observed. Requires times >= 2.
    bool match_online(const std::uint8_t* hist, std::size_t len, unsigned& pred) const {
        if (len < 4 || data == nullptr || mk.empty()) return false;
        const std::uint32_t key = key4(hist, len);
        std::uint32_t i = mix32(key) & mmask;
        for (;;) {
            if (mk[i] == kNil) return false;
            if (mk[i] == key) {
                if (ms[i] < 2) return false;
                const std::uint32_t follower = mv[i];
                if (static_cast<std::size_t>(follower) >= len) return false;
                pred = data[follower];
                return true;
            }
            i = (i + 1u) & mmask;
        }
    }

    // E0..E4 as distributions. E2..E4 are mix()'s unnormalised cur after that
    // order, divided by its mass. A shorter expert is copied when this order
    // is past the available context. online_match reads the training hash;
    // otherwise the compacted table, with followers strictly before follower_limit.
    void fill_experts(const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode,
                      double e[5][256], bool& matched, unsigned& pred, bool online_match) const {
        fill_kt0(e[0]);
        const unsigned prev = len ? hist[len - 1] : 0;
        fill_kt1(prev, e[1]);

        double buf[2][256];
        double* cur = buf[0];
        double* nxt = buf[1];
        if (len >= 1 && o1n[prev] > 0) std::memcpy(cur, e[1], sizeof(double) * 256);
        else std::memcpy(cur, e[0], sizeof(double) * 256);

        const int top = std::min(4, static_cast<int>(len));
        std::uint32_t ctx_node[5];
        ctx_node[0] = 0;
        int depth = 0;
        for (int d = 1; d <= top; ++d) {
            const std::uint32_t c = find_child(ctx_node[d - 1], hist[len - static_cast<std::size_t>(d)]);
            if (c == kNil) break;
            ctx_node[d] = c;
            depth = d;
        }

        int reached = 1;
        for (int o = 2; o <= top; ++o) {
            for (int s = 0; s < 256; ++s) nxt[s] = alpha * cur[s];
            const std::uint32_t node = (o <= depth) ? ctx_node[o] : kNil;
            std::uint8_t syms[256];
            std::uint32_t cs[256];
            int nf = 0;
            std::uint32_t tot = 0;
            if (gather(node, mode, syms, cs, nf, tot)) {
                const double inv = 1.0 / static_cast<double>(tot);
                for (int i = 0; i < nf; ++i) nxt[syms[i]] = static_cast<double>(cs[i]) * inv;
            }
            std::swap(cur, nxt);
            double z = 0.0;
            for (int s = 0; s < 256; ++s) z += cur[s];
            if (!(z > 0.0)) z = 1.0;
            const double invz = 1.0 / z;
            for (int s = 0; s < 256; ++s) e[o][s] = cur[s] * invz;
            reached = o;
        }
        for (int o = reached + 1; o <= 4; ++o) std::memcpy(e[o], e[o - 1], sizeof(double) * 256);

        if (online_match) matched = match_online(hist, len, pred);
        else matched = find_match(hist, len, follower_limit, pred);
    }

    // The fixed backoff distribution with the match folded in at weight match_w.
    // Returns u = H(order-1) / ln(256).
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
            for (int s = 0; s < 256; ++s) nxt[s] = alpha * cur[s];
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
            const double agree = kt1p[pred] / (kt1p[pred] + agree_floor);
            match_w = (1.0 - u) * agree;
            if (match_w < 0.0) match_w = 0.0;
            if (match_w > match_cap) match_w = match_cap;
        }
        for (int s = 0; s < 256; ++s) P[s] = (1.0 - match_w) * B[s];
        if (matched) P[pred] += match_w;
        return u;
    }

    // ---- Negative results (CURRENT.md "PPMD, longer chains, or inheritance ~2.42").

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
            if (verbose && ((i + 1) & 0xfffff) == 0)
                std::cerr << "extend " << (i + 1) << " nodes " << nodes.size() << "\n";
        }
        if (verbose) std::cerr << "chain_nodes_added " << (nodes.size() - before) << "\n";
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
        if (verbose) std::cerr << "inherited " << bumped << "\n";
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
            const double agree = kt1p[pred] / (kt1p[pred] + agree_floor);
            match_w = (1.0 - u) * agree;
            if (match_w < 0.0) match_w = 0.0;
            if (match_w > match_cap) match_w = match_cap;
        }
        if (matched && match_w > 0.0) {
            for (int s = 0; s < 256; ++s) P[s] *= (1.0 - match_w);
            P[pred] += match_w;
        }
        return u;
    }
};

// ---- Mixture weights: six experts (E0..E4 and the match).

// E0..E4 share (1 - w_match0) equally; the match expert starts at w_match0.
inline void init_weights(double w[6], const Config& c = Config{}) {
    w[5] = c.w_match0;
    const double share = (1.0 - w[5]) / 5.0;
    for (int i = 0; i < 5; ++i) w[i] = share;
    double s = 0.0;
    for (int i = 0; i < 6; ++i) s += w[i];
    for (int i = 0; i < 6; ++i) w[i] /= s;
}

// p = sum of stored weights times P_i(y) over experts present this step.
// An absent match has effective weight 0 and is left out of the product.
// The exponent cap only stops a vanished weight from overflowing double.
inline void eg_update(double w[6], const double py[6], bool match_on, const Config& c = Config{}) {
    const bool active[6] = {true, true, true, true, true, match_on};
    double p = 0.0;
    for (int i = 0; i < 6; ++i)
        if (active[i]) p += w[i] * py[i];
    if (!(p > 1e-12)) p = 1e-12;
    for (int i = 0; i < 6; ++i) {
        if (!active[i]) continue;
        double ex = c.eta * (py[i] / p);
        if (ex > 40.0) ex = 40.0;
        w[i] *= std::exp(ex);
    }
    double sum = 0.0;
    for (int i = 0; i < 6; ++i) {
        if (!(w[i] > c.w_floor)) w[i] = c.w_floor;
        sum += w[i];
    }
    const double inv = 1.0 / sum;
    for (int i = 0; i < 6; ++i) w[i] *= inv;
}

// Scale the experts that are present so the mixture is a distribution.
inline void mix_dist(const double w[6], const double e[5][256], bool matched, unsigned pred, double* M) {
    double z = 0.0;
    for (int i = 0; i < 5; ++i) z += w[i];
    if (matched) z += w[5];
    const double inv = 1.0 / z;
    std::memset(M, 0, sizeof(double) * 256);
    for (int i = 0; i < 5; ++i) {
        const double a = w[i] * inv;
        for (int s = 0; s < 256; ++s) M[s] += a * e[i][s];
    }
    if (matched) M[pred] += w[5] * inv;
}

// ---- Training

struct TrainLog {
    bool progress = false;           // "train N nodes X w4 .. wmatch .." every 2^20 bytes
    std::uint64_t matches = 0;       // bytes on which the online match fired
};

// Score each byte [a, b) from the counts so far, move the weights, then observe
// the byte. model.data must be data.data(). Synth mode, online (training-hash) match.
inline void train_span(Model& model, const std::vector<std::uint8_t>& data, double w[6], std::size_t a,
                       std::size_t b, const Config& cfg = Config{}, TrainLog* log = nullptr) {
    for (std::size_t i = a; i < b; ++i) {
        double e[5][256];
        bool matched = false;
        unsigned pred = 0;
        model.fill_experts(data.data(), i, i, Mode::Synth, e, matched, pred, true);
        double py[6];
        const unsigned y = data[i];
        for (int k = 0; k < 5; ++k) py[k] = e[k][y];
        py[5] = (matched && pred == y) ? 1.0 : 0.0;
        if (log && matched) ++log->matches;
        eg_update(w, py, matched, cfg);
        model.observe(i);
        if (log && log->progress && ((i + 1) & 0xfffff) == 0) {
            std::cerr << "train " << (i + 1) << " nodes " << model.nodes.size() << " w4 " << w[4] << " wmatch " << w[5]
                      << "\n";
        }
    }
}

// Reads exactly n bytes of the file. Returns "" on success, else the reason.
inline std::string read_prefix(const std::string& path, std::size_t n, std::vector<std::uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "cannot open corpus " + path;
    out.assign(n, 0);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n));
    if (static_cast<std::size_t>(in.gcount()) != n) return "short corpus " + path;
    return {};
}

// ---- Dolly (chat) loader

struct Turn {
    std::string prompt;
    std::string answer;
};

// "User: <instruction>\n[<context>\n]Assistant: " then the response, "\n\n" between turns.
// Returns the concatenated chat text; turns gets one entry per usable record.
inline std::string load_dolly(const std::string& path, std::vector<Turn>& turns) {
    std::ifstream in(path);
    if (!in) return {};
    std::string line, out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto j = nlohmann::json::parse(line);
        const std::string instr = j.value("instruction", "");
        const std::string ctx = j.value("context", "");
        const std::string resp = j.value("response", "");
        if (instr.empty() || resp.empty()) continue;
        std::string prompt = "User: " + instr + "\n";
        if (!ctx.empty()) prompt += ctx + "\n";
        prompt += "Assistant: ";
        turns.push_back(Turn{prompt, resp});
        out += prompt;
        out += resp;
        out += "\n\n";
    }
    return out;
}

// A document boundary, not a one-byte newline. Order 4 must not learn the last
// bytes of Wikipedia as the left context of "User:".
inline constexpr const char* kDocBoundary = "\n\n<|doc|>\n\n";

// Appends boundary + chat to data and repoints model.data. Returns where the chat starts.
inline std::size_t append_chat(Model& model, std::vector<std::uint8_t>& data, const std::string& chat) {
    const std::size_t chat_at = data.size();
    data.insert(data.end(), kDocBoundary, kDocBoundary + std::strlen(kDocBoundary));
    data.insert(data.end(), reinterpret_cast<const std::uint8_t*>(chat.data()),
                reinterpret_cast<const std::uint8_t*>(chat.data()) + chat.size());
    model.data = data.data();
    return chat_at;
}

// ---- Scoring

struct Holdout {
    double backoff_bpc = 0;  // the fixed backoff (Model::mix)
    double mix_bpc = 0;      // the learned mixture, weights frozen
    bool ok = false;         // every probability finite, every mixture a distribution, experts agree with mix()
};

// Scores [cfg.train, cfg.sample) with the frozen counts (Synth, compacted match, followers
// before cfg.train) both ways: the fixed backoff and the mixture with weights w.
// With checks on, each 2^16 bytes it verifies sum(M) = 1 and that the mixture's E4 expert
// reproduces mix() to 1e-6. Failures set ok = false and write the reason to stderr.
inline Holdout score_holdout(const Model& model, const std::uint8_t* data, const double w[6],
                             const Config& cfg = Config{}, bool progress = true) {
    Holdout r;
    double bits_b = 0.0, bits_m = 0.0;
    for (std::size_t i = cfg.train; i < cfg.sample; ++i) {
        double P[256];
        bool matched = false;
        model.mix(data, i, cfg.train, Mode::Synth, P, matched);
        const double pb = P[data[i]];
        if (!(pb > 0.0) || !std::isfinite(pb)) {
            std::cerr << "bad backoff p at " << i << "\n";
            return r;
        }
        bits_b += -std::log2(pb);

        double e[5][256];
        bool em = false;
        unsigned pred = 0;
        model.fill_experts(data, i, cfg.train, Mode::Synth, e, em, pred, false);
        double M[256];
        mix_dist(w, e, em, pred, M);
        const double pm = M[data[i]];
        if (!(pm > 0.0) || !std::isfinite(pm)) {
            std::cerr << "bad mix p at " << i << "\n";
            return r;
        }
        bits_m += -std::log2(pm);

        if (i == cfg.train || ((i - cfg.train) & 0xffff) == 0) {
            double sum = 0.0;
            for (int s = 0; s < 256; ++s) sum += M[s];
            if (std::fabs(sum - 1.0) > 1e-5) {
                std::cerr << "mixture sum " << sum << " at " << i << "\n";
                return r;
            }
            double kt1p[256];
            model.fill_kt1(data[i - 1], kt1p);
            const double u = Model::entropy_u(kt1p);
            double mw = 0.0;
            if (em) {
                const double agree = kt1p[pred] / (kt1p[pred] + model.agree_floor);
                mw = (1.0 - u) * agree;
                if (mw < 0.0) mw = 0.0;
                if (mw > model.match_cap) mw = model.match_cap;
            }
            double maxdiff = 0.0;
            for (int s = 0; s < 256; ++s) {
                double rr = (1.0 - mw) * e[4][s];
                if (em && static_cast<unsigned>(s) == pred) rr += mw;
                maxdiff = std::max(maxdiff, std::fabs(rr - P[s]));
            }
            if (em != matched || maxdiff > 1e-6) {
                std::cerr << "expert backoff diverges from mix at " << i << " diff " << maxdiff << "\n";
                return r;
            }
        }
        if (progress && ((i - cfg.train + 1) & 0x3ffff) == 0) std::cerr << "score " << (i - cfg.train + 1) << "\n";
    }
    const double nhold = static_cast<double>(cfg.sample - cfg.train);
    r.backoff_bpc = bits_b / nhold;
    r.mix_bpc = bits_m / nhold;
    r.ok = true;
    return r;
}

struct Score {
    double bpc = 0;
    double mean_u = 0;
    double match_rate = 0;
};

// The fixed backoff (or PPMD, max_depth) alone, as fold_lm printed it: BPC, mean u, match rate.
// Exits the process on a bad probability, as the tool did.
inline Score heldout(const Model& m, const std::uint8_t* data, Mode mode, const Config& cfg = Config{},
                     bool ppmd = false, int max_depth = 4) {
    double bits = 0, sum_u = 0;
    std::uint64_t matches = 0;
    const double n = static_cast<double>(cfg.sample - cfg.train);
    for (std::size_t i = cfg.train; i < cfg.sample; ++i) {
        double P[256];
        bool matched = false;
        const double u = ppmd ? m.mix_ppmd(data, i, cfg.train, mode, max_depth, P, matched)
                              : m.mix(data, i, cfg.train, mode, P, matched);
        if (matched) ++matches;
        if (((i - cfg.train) & 0xffff) == 0) {
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
        if (((i - cfg.train + 1) & 0x3ffff) == 0) std::cerr << "score " << (i - cfg.train + 1) << "\n";
    }
    return Score{bits / n, sum_u / n, static_cast<double>(matches) / n};
}

// ---- Packed-layout accounting

inline std::uint64_t next_pow2_load(std::uint64_t filled) {
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

inline bool kept_fol(const Model& m, std::uint32_t node, Mode mode) {
    if (m.nodes[node].total == 0) return false;
    if (mode == Mode::DropCtx && m.nodes[node].total == 1) return false;
    if (mode != Mode::Synth) return m.nodes[node].total > 0;
    for (std::uint32_t f = m.nodes[node].fol_head; f != kNil; f = m.fols[f].next)
        if (m.fols[f].count >= 2) return true;
    return false;
}

inline int kept_nfol(const Model& m, std::uint32_t node, Mode mode) {
    if (mode == Mode::DropCtx && m.nodes[node].total == 1) return 0;
    int n = 0;
    for (std::uint32_t f = m.nodes[node].fol_head; f != kNil; f = m.fols[f].next)
        if (used_count(m.fols[f].count, mode) > 0) ++n;
    return n;
}

inline Layout measure(const Model& m, Mode mode) {
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

inline std::uint64_t hash_reserved(const Model& m) {
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

inline std::uint64_t hash_payload(const Model& m) {
    std::uint64_t bytes = 0;
    for (const Node& n : m.nodes) {
        if (n.depth < 2 || n.total == 0) continue;
        bytes += 18ull;
        if (n.nfol > 1) bytes += static_cast<std::uint64_t>(n.nfol - 1) * 9ull;
    }
    bytes += static_cast<std::uint64_t>(m.dense.size()) * 13ull;
    return bytes;
}

// Fixed-size parts of a stored model: unigram + order-1 tables, the compacted match, the weights.
inline std::uint64_t order1_bytes(const Model& m) {
    return sizeof(m.uni) + sizeof(m.uni_n) + sizeof(m.o1) + sizeof(m.o1n);
}
inline std::uint64_t match_bytes(const Model& m) { return m.dense.size() * sizeof(Hit); }

// Counts of contexts per order and the order-2 fan-out (the "what the measurements settled" rows).
struct TrieStats {
    std::uint64_t by_depth[5] = {};
    std::uint64_t singletons = 0, ctx = 0, one_child = 0, sum_child = 0, max_child = 0;
};

inline TrieStats trie_stats(const Model& m) {
    TrieStats t;
    for (const Node& n : m.nodes) {
        if (n.depth < 2 || n.depth > 4 || n.total == 0) continue;
        ++t.by_depth[n.depth];
        ++t.ctx;
        if (n.total == 1) ++t.singletons;
    }
    for (const Node& n : m.nodes) {
        if (n.depth != 2) continue;
        t.sum_child += n.nchild;
        if (n.nchild > t.max_child) t.max_child = n.nchild;
        if (n.nchild == 1) ++t.one_child;
    }
    return t;
}

// ---- Text helpers

inline std::string printable(const std::string& s) {
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

// ---- Sampling

// The no-repeat rule: bans the byte that would complete a second copy of the last 11 bytes.
inline void mark_banned(const std::vector<std::uint8_t>& seq, bool banned[256]) {
    std::memset(banned, 0, 256);
    const std::size_t n = seq.size();
    if (n < 11) return;
    const std::uint8_t* last = seq.data() + (n - 11);
    for (std::size_t s = 0; s + 12 <= n; ++s)
        if (std::memcmp(seq.data() + s, last, 11) == 0) banned[seq[s + 11]] = true;
}

struct SampleOptions {
    double temperature = 0.8;
    double nucleus = 0.9;
    bool greedy = false;      // argmax of the mixture (lowest byte wins a tie); the rng is not used
    bool no_repeat = false;   // apply mark_banned (unless that would ban every byte)
};

// n bytes after prompt from the learned mixture (E0..E4 and the match, weights w), Synth mode,
// compacted match with followers strictly before follower_limit. Temperature, then the
// nucleus, then one uniform draw per byte from mt19937(seed).
inline std::string sample_continuation(const Model& model, const double w[6], const std::string& prompt,
                                       std::size_t follower_limit, int n, std::uint32_t seed,
                                       const SampleOptions& opt = SampleOptions{}) {
    std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int t = 0; t < n; ++t) {
        double e[5][256];
        bool matched = false;
        unsigned pred = 0;
        model.fill_experts(seq.data(), seq.size(), follower_limit, Mode::Synth, e, matched, pred, false);
        double M[256];
        mix_dist(w, e, matched, pred, M);
        bool banned[256];
        if (opt.no_repeat) mark_banned(seq, banned);
        else std::memset(banned, 0, sizeof banned);
        int pick = 0;
        if (opt.greedy) {
            bool all_banned = true;
            for (int s = 0; s < 256; ++s)
                if (!banned[s]) all_banned = false;
            bool have = false;
            for (int s = 0; s < 256; ++s) {
                if (banned[s] && !all_banned) continue;
                if (!have || M[s] > M[pick]) {
                    pick = s;
                    have = true;
                }
            }
        } else {
            struct Item {
                int y;
                double p;
            } items[256];
            double mass = 0;
            for (int pass = 0; pass < 2 && !(mass > 0); ++pass) {
                mass = 0;
                for (int s = 0; s < 256; ++s) {
                    const double base =
                        (pass == 0 && banned[s]) ? 0.0 : std::pow(std::max(M[s], 0.0), 1.0 / opt.temperature);
                    items[s] = {s, base};
                    mass += base;
                }
            }
            std::sort(items, items + 256, [](const Item& a, const Item& b) { return a.p > b.p; });
            double acc = 0;
            int keep = 1;
            for (int k = 0; k < 256; ++k) {
                acc += items[k].p / mass;
                keep = k + 1;
                if (acc >= opt.nucleus) break;
            }
            double km = 0;
            for (int k = 0; k < keep; ++k) km += items[k].p;
            double r = U(rng) * km;
            pick = items[keep - 1].y;
            for (int k = 0; k < keep; ++k) {
                r -= items[k].p;
                if (r <= 0) {
                    pick = items[k].y;
                    break;
                }
            }
        }
        seq.push_back(static_cast<std::uint8_t>(pick));
    }
    return std::string(seq.begin() + static_cast<std::ptrdiff_t>(prompt.size()), seq.end());
}

// ---- Self-test

inline bool self_test() {
    if (qdec[255] < 1000000) {
        std::cerr << "quantizer tops out at " << qdec[255] << "\n";
        return false;
    }
    const std::string text = "abcabcabcabcabcZZ";
    std::vector<std::uint8_t> buf(text.begin(), text.end());
    Model m;
    m.verbose = false;
    m.data = buf.data();
    m.init();
    for (std::size_t i = 0; i < buf.size(); ++i) m.observe(i);
    unsigned online_pred = 0;
    (void)m.match_online(buf.data(), buf.size(), online_pred);
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
    double experts[5][256];
    bool em = false;
    unsigned epred = 0;
    m.fill_experts(ctx, 2, 0, Mode::Full, experts, em, epred, false);
    for (int s = 0; s < 256; ++s) {
        if (std::fabs(P[s] - experts[4][s]) > 1e-8) {
            std::cerr << "self-test expert order2 mismatch\n";
            return false;
        }
    }
    m.fill_experts(buf.data(), buf.size(), 0, Mode::Full, experts, em, epred, false);
    m.mix(buf.data(), buf.size(), 0, Mode::Full, P, matched);
    sum = 0.0;
    for (int s = 0; s < 256; ++s) {
        sum += experts[4][s];
        if (std::fabs(P[s] - experts[4][s]) > 1e-8) {
            std::cerr << "self-test expert order4 mismatch\n";
            return false;
        }
    }
    if (em || std::fabs(sum - 1.0) > 1e-8) {
        std::cerr << "self-test expert sum " << sum << " matched " << em << "\n";
        return false;
    }
    double mw[6] = {0.17, 0.17, 0.17, 0.17, 0.17, 0.15};
    const double mpy[6] = {0.01, 0.02, 0.04, 0.1, 0.5, 1.0};
    eg_update(mw, mpy, true);
    double ws = 0.0;
    for (int i = 0; i < 6; ++i) {
        if (!(mw[i] >= 1e-4) || !std::isfinite(mw[i])) {
            std::cerr << "self-test mixer weight not finite\n";
            return false;
        }
        ws += mw[i];
    }
    if (std::fabs(ws - 1.0) > 1e-9 || !(mw[4] > mw[0] && mw[5] > mw[0])) {
        std::cerr << "self-test mixer update failed\n";
        return false;
    }
    eg_update(mw, mpy, false);
    if (!(mw[5] >= 1e-4)) {
        std::cerr << "self-test absent match was updated\n";
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
    return true;
}

}  // namespace cypha::cyphalm::trie
