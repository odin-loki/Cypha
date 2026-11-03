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
// PackedModel (pack()) is the Synth form built as a read-only structure for
// serving; its distributions equal the training Model's in Synth bit for bit
// (docs/reports/RAM_PROFILE.md, M4).
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
//                      mix, mix_dist, compact_match), PackedModel and pack()
//                      (the serving form; save / load: the stored file),
//                      eg_update, layout
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
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

// KT estimate (c + 1/2) / (n + 128) over 256 counts: the unigram and order-1 experts of both model forms.
inline void fill_kt(const std::uint32_t* c, std::uint32_t n, double* p) {
    const double den = static_cast<double>(n) + 128.0;
    for (int s = 0; s < 256; ++s) p[s] = (static_cast<double>(c[s]) + 0.5) / den;
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

    // Frees the trie, the followers and the match once pack() holds the serving form. The count
    // tables stay, so a released model answers from orders 0 and 1 only.
    void release() {
        std::vector<Node>(1).swap(nodes);
        std::vector<Fol>().swap(fols);
        std::vector<Hit>().swap(dense);
        for (auto* v : {&mk, &mv, &ms}) std::vector<std::uint32_t>().swap(*v);
        for (auto& c : root_child) c = kNil;
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

    void fill_kt0(double* p) const { fill_kt(uni, uni_n, p); }
    void fill_kt1(unsigned prev, double* p) const { fill_kt(o1[prev], o1n[prev], p); }

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
    // (Body: detail::fill_experts below, shared with PackedModel.)
    void fill_experts(const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode,
                      double e[5][256], bool& matched, unsigned& pred, bool online_match) const;

    // The fixed backoff distribution with the match folded in at weight match_w.
    // Returns u = H(order-1) / ln(256). (Body: detail::mix below, shared with PackedModel.)
    double mix(const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode, double* P,
               bool& matched) const;

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

// ---- fill_experts and mix: one body for both model forms
//
// Model and PackedModel run the same arithmetic in the same order, so they give the same doubles.
// M provides fill_kt0, fill_kt1, o1n, alpha, agree_floor, match_cap, find_child, find_ctx, gather,
// find_match and match_online, with Model's meanings.
namespace detail {

template <class M>
void fill_experts(const M& m, const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode,
                  double e[5][256], bool& matched, unsigned& pred, bool online_match) {
    m.fill_kt0(e[0]);
    const unsigned prev = len ? hist[len - 1] : 0;
    m.fill_kt1(prev, e[1]);

    double buf[2][256];
    double* cur = buf[0];
    double* nxt = buf[1];
    if (len >= 1 && m.o1n[prev] > 0) std::memcpy(cur, e[1], sizeof(double) * 256);
    else std::memcpy(cur, e[0], sizeof(double) * 256);

    const int top = std::min(4, static_cast<int>(len));
    std::uint32_t ctx_node[5];
    ctx_node[0] = 0;
    int depth = 0;
    for (int d = 1; d <= top; ++d) {
        const std::uint32_t c = m.find_child(ctx_node[d - 1], hist[len - static_cast<std::size_t>(d)]);
        if (c == kNil) break;
        ctx_node[d] = c;
        depth = d;
    }

    int reached = 1;
    for (int o = 2; o <= top; ++o) {
        for (int s = 0; s < 256; ++s) nxt[s] = m.alpha * cur[s];
        const std::uint32_t node = (o <= depth) ? ctx_node[o] : kNil;
        std::uint8_t syms[256];
        std::uint32_t cs[256];
        int nf = 0;
        std::uint32_t tot = 0;
        if (m.gather(node, mode, syms, cs, nf, tot)) {
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

    if (online_match) matched = m.match_online(hist, len, pred);
    else matched = m.find_match(hist, len, follower_limit, pred);
}

template <class M>
double mix(const M& m, const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode, double* P,
           bool& matched) {
    double kt1p[256];
    const unsigned prev = len ? hist[len - 1] : 0;
    m.fill_kt1(prev, kt1p);
    const double u = Model::entropy_u(kt1p);
    double buf[2][256];
    double* cur = buf[0];
    double* nxt = buf[1];
    if (len >= 1 && m.o1n[prev] > 0) std::memcpy(cur, kt1p, sizeof(double) * 256);
    else m.fill_kt0(cur);

    const int top = std::min(4, static_cast<int>(len));
    for (int o = 2; o <= top; ++o) {
        for (int s = 0; s < 256; ++s) nxt[s] = m.alpha * cur[s];
        const std::uint32_t node = m.find_ctx(hist, len, o);
        std::uint8_t syms[256];
        std::uint32_t cs[256];
        int nf = 0;
        std::uint32_t tot = 0;
        if (m.gather(node, mode, syms, cs, nf, tot)) {
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
    matched = m.find_match(hist, len, follower_limit, pred);
    double match_w = 0.0;
    if (matched) {
        const double agree = kt1p[pred] / (kt1p[pred] + m.agree_floor);
        match_w = (1.0 - u) * agree;
        if (match_w < 0.0) match_w = 0.0;
        if (match_w > m.match_cap) match_w = m.match_cap;
    }
    for (int s = 0; s < 256; ++s) P[s] = (1.0 - match_w) * B[s];
    if (matched) P[pred] += match_w;
    return u;
}

}  // namespace detail

inline void Model::fill_experts(const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode,
                                double e[5][256], bool& matched, unsigned& pred, bool online_match) const {
    detail::fill_experts(*this, hist, len, follower_limit, mode, e, matched, pred, online_match);
}

inline double Model::mix(const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode, double* P,
                         bool& matched) const {
    return detail::mix(*this, hist, len, follower_limit, mode, P, matched);
}

// ---- PackedModel: the read-only serving form (docs/reports/RAM_PROFILE.md, M4)
//
// pack(model) keeps what Mode::Synth reads with the compacted match, and nothing else:
//   - the unigram and order-1 count tables, as trained (the KT experts use full counts);
//   - the trie nodes Synth can reach: a node with a follower seen at least twice, or on the path to
//     one (measure()'s "useful" nodes). Ids are breadth first: 0 is the root, 1 + b the order-1 node
//     of byte b (all 256), then orders 2, 3 and 4. A node's children are consecutive ids sorted by
//     their edge byte, ids kid[n] .. kid[n + 1] - 1, so there are no child or sibling links;
//   - the followers seen at least twice: a symbol byte and the count quantised to one byte (qdec),
//     node n of order >= 2 owning fol[n - kFirst2] .. fol[n - kFirst2 + 1] - 1;
//   - the compacted match (contexts seen at least twice), bucketed by the key's high 16 bits: the
//     low 16 bits, the follower's offset (for follower_limit) and the predicted byte itself, so the
//     serving form does not read the training text.
// fill_experts and mix are Model's (detail:: above), so every distribution is bit-identical to the
// training Model's in Mode::Synth with the compacted match (online_match = false). The packed form
// holds Mode::Synth only, and online_match finds nothing, as in a Model after compact_match.
struct PackedModel {
    static constexpr std::uint32_t kFirst2 = 257;  // the first order-2 id
    std::uint32_t uni[256]{};
    std::uint32_t uni_n = 0;
    std::vector<std::uint32_t> o1;     // 256 x 256 order-1 counts, row = the previous byte (on the heap)
    std::uint32_t o1n[256]{};
    std::vector<std::uint32_t> kid;    // ids 0 .. kend: children [kid[n], kid[n + 1])
    std::vector<std::uint8_t> edge;    // every id: the byte on the edge from its parent
    std::vector<std::uint32_t> fol;    // ids kFirst2 ..: followers [fol[n - kFirst2], fol[n - kFirst2 + 1])
    std::vector<std::uint8_t> fsym, fq;  // follower symbol, quantised count
    std::vector<std::uint32_t> mstart;   // 65,537 bucket starts (empty: no match table)
    std::vector<std::uint16_t> mlo;      // key & 0xffff, sorted within the bucket
    std::vector<std::uint32_t> mfol;     // follower offset in the training text
    std::vector<std::uint8_t> mnext;     // the byte at that offset
    std::uint32_t kend = 0;              // the first order-4 id (order-4 nodes have no children)
    double alpha = 0.4, agree_floor = 0.05, match_cap = 0.75;

    void fill_kt0(double* p) const { fill_kt(uni, uni_n, p); }
    void fill_kt1(unsigned prev, double* p) const { fill_kt(o1.data() + std::size_t{prev} * 256u, o1n[prev], p); }

    std::uint32_t find_child(std::uint32_t n, std::uint8_t b) const {
        if (n == 0) return 1u + b;
        if (n >= kend) return kNil;
        const std::uint8_t* lo = edge.data() + kid[n];
        const std::uint8_t* hi = edge.data() + kid[n + 1];
        const std::uint8_t* it = std::lower_bound(lo, hi, b);
        return (it != hi && *it == b) ? static_cast<std::uint32_t>(it - edge.data()) : kNil;
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

    // Model::gather in Mode::Synth (the only mode stored).
    bool gather(std::uint32_t node, Mode, std::uint8_t* syms, std::uint32_t* cs, int& n, std::uint32_t& total) const {
        n = 0;
        total = 0;
        if (node == kNil || node < kFirst2) return false;
        for (std::uint32_t f = fol[node - kFirst2]; f < fol[node - kFirst2 + 1]; ++f) {
            syms[n] = fsym[f];
            cs[n] = qdec[fq[f]];
            total += cs[n];
            ++n;
        }
        return n > 0;
    }

    bool find_match(const std::uint8_t* hist, std::size_t len, std::size_t limit, unsigned& pred) const {
        if (len < 4 || mstart.empty()) return false;
        const std::uint32_t key = key4(hist, len);
        const std::uint16_t low = static_cast<std::uint16_t>(key & 0xffffu);
        const std::uint16_t* lo = mlo.data() + mstart[key >> 16];
        const std::uint16_t* hi = mlo.data() + mstart[(key >> 16) + 1];
        const std::uint16_t* it = std::lower_bound(lo, hi, low);
        if (it == hi || *it != low) return false;
        const auto j = static_cast<std::size_t>(it - mlo.data());
        if (static_cast<std::size_t>(mfol[j]) >= limit) return false;
        pred = mnext[j];
        return true;
    }

    // The training hash is not kept: as in a Model after compact_match, the online match finds nothing.
    bool match_online(const std::uint8_t*, std::size_t, unsigned&) const { return false; }

    static void require_synth(Mode mode) {
        if (mode == Mode::Synth) return;
        std::cerr << "PackedModel holds Mode::Synth only\n";
        std::abort();
    }

    // Same signatures and meanings as Model's; mode must be Mode::Synth.
    void fill_experts(const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode,
                      double e[5][256], bool& matched, unsigned& pred, bool online_match) const {
        require_synth(mode);
        detail::fill_experts(*this, hist, len, follower_limit, mode, e, matched, pred, online_match);
    }

    double mix(const std::uint8_t* hist, std::size_t len, std::size_t follower_limit, Mode mode, double* P,
               bool& matched) const {
        require_synth(mode);
        return detail::mix(*this, hist, len, follower_limit, mode, P, matched);
    }

    std::uint64_t nodes() const { return edge.size() - 1; }  // without the root; all 256 order-1 nodes
    std::uint64_t followers() const { return fsym.size(); }
    std::uint64_t matches() const { return mlo.size(); }
    // Bytes held: the tables in the struct plus every array (pack() allocates each array once, at its size).
    std::uint64_t bytes() const {
        return sizeof(*this) + o1.size() * 4ull + kid.size() * 4ull + edge.size() + fol.size() * 4ull + fsym.size() +
               fq.size() + mstart.size() * 4ull + mlo.size() * 2ull + mfol.size() * 4ull + mnext.size();
    }

    bool operator==(const PackedModel&) const = default;  // every table and array

    // The arrays fit together, so every lookup stays inside them. load() checks a stored file with it.
    bool consistent() const {
        const std::size_t n = edge.size();
        auto rising = [](const std::vector<std::uint32_t>& v, std::size_t top) {
            for (std::size_t i = 1; i < v.size(); ++i)
                if (v[i] < v[i - 1]) return false;
            return v.empty() || v.back() <= top;
        };
        const bool trie = o1.size() == 65536 && n >= kFirst2 && kend >= kFirst2 && kend <= n &&
                          kid.size() == std::size_t{kend} + 1 && kid.back() == n && rising(kid, n) &&
                          fol.size() == n - kFirst2 + 1 && fol.back() == fsym.size() && rising(fol, fsym.size()) &&
                          fq.size() == fsym.size();
        const bool match = mstart.empty() ? mlo.empty() && mfol.empty() && mnext.empty()
                                          : mstart.size() == 65537 && mstart.back() == mlo.size() &&
                                                rising(mstart, mlo.size()) && mfol.size() == mlo.size() &&
                                                mnext.size() == mlo.size();
        return trie && match;
    }

    // A stored packed file (docs/reports/RAM_PROFILE.md, M7). It also holds the mixture weights w and the
    // scoring split, so a served model is self-contained. Little-endian, no padding: "CYPHAPK" and a zero
    // byte, u32 version 1, u32 0, u64 cfg.train and cfg.sample, f64 alpha, agree_floor, match_cap and w[0..5],
    // u32 uni_n and kend, u32 uni[256] and o1n[256], then o1, kid, edge, fol, fsym, fq, mstart, mlo, mfol and
    // mnext, each a u64 count and its elements, then the FNV-1a 64-bit hash of every byte before it.
    // Errors throw std::runtime_error. load() sets cfg.train, sample, alpha, agree_floor and match_cap and
    // leaves the training-only fields of cfg as they are.
    void save(std::ostream& os, const double w[6], const Config& cfg) const;
    void save(const std::string& path, const double w[6], const Config& cfg) const;
    static PackedModel load(std::istream& is, double w[6], Config& cfg);
    static PackedModel load(const std::string& path, double w[6], Config& cfg);
};

namespace detail {

inline constexpr char kPackMagic[8] = {'C', 'Y', 'P', 'H', 'A', 'P', 'K', '\0'};
inline constexpr std::uint32_t kPackVersion = 1;

struct Fnv64 {
    std::uint64_t h = 14695981039346656037ull;
    void add(const void* p, std::size_t n) {
        const auto* b = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    }
};

// Little-endian writer and reader of the stored file. On a little-endian machine an array is one block.
struct PackOut {
    std::ostream& os;
    Fnv64 sum{};
    void raw(const void* p, std::size_t n) {
        sum.add(p, n);
        os.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    }
    template <class T>
    void num(T v) {  // an unsigned integer
        unsigned char b[sizeof(T)];
        for (std::size_t i = 0; i < sizeof(T); ++i) b[i] = static_cast<unsigned char>(std::uint64_t{v} >> (8 * i));
        raw(b, sizeof b);
    }
    void real(double v) { num(std::bit_cast<std::uint64_t>(v)); }
    template <class T>
    void array(const T* p, std::size_t n) {
        if constexpr (std::endian::native == std::endian::little) raw(p, n * sizeof(T));
        else
            for (std::size_t i = 0; i < n; ++i) num(p[i]);
    }
    template <class T>
    void vec(const std::vector<T>& v) {
        num(std::uint64_t{v.size()});
        array(v.data(), v.size());
    }
};

struct PackIn {
    std::istream& is;
    std::uint64_t left;  // bytes the stream still holds
    Fnv64 sum{};
    void raw(void* p, std::size_t n) {
        if (n > left || !is.read(static_cast<char*>(p), static_cast<std::streamsize>(n)))
            throw std::runtime_error("PackedModel: stored file cut short");
        left -= n;
        sum.add(p, n);
    }
    template <class T>
    T num() {
        unsigned char b[sizeof(T)];
        raw(b, sizeof b);
        std::uint64_t v = 0;
        for (std::size_t i = 0; i < sizeof(T); ++i) v |= std::uint64_t{b[i]} << (8 * i);
        return static_cast<T>(v);
    }
    double real() { return std::bit_cast<double>(num<std::uint64_t>()); }
    template <class T>
    void array(T* p, std::size_t n) {
        if constexpr (std::endian::native == std::endian::little) raw(p, n * sizeof(T));
        else
            for (std::size_t i = 0; i < n; ++i) p[i] = num<T>();
    }
    template <class T>
    void vec(std::vector<T>& v) {
        const auto n = num<std::uint64_t>();
        if (n > left / sizeof(T)) throw std::runtime_error("PackedModel: stored file damaged");
        v.resize(static_cast<std::size_t>(n));
        array(v.data(), v.size());
    }
};

}  // namespace detail

inline void PackedModel::save(std::ostream& os, const double w[6], const Config& cfg) const {
    detail::PackOut o{os};
    o.raw(detail::kPackMagic, sizeof detail::kPackMagic);
    o.num(detail::kPackVersion);
    o.num(std::uint32_t{0});
    o.num(std::uint64_t{cfg.train});
    o.num(std::uint64_t{cfg.sample});
    for (const double v : {alpha, agree_floor, match_cap}) o.real(v);
    for (int i = 0; i < 6; ++i) o.real(w[i]);
    o.num(uni_n);
    o.num(kend);
    o.array(uni, 256);
    o.array(o1n, 256);
    o.vec(o1);
    o.vec(kid);
    o.vec(edge);
    o.vec(fol);
    o.vec(fsym);
    o.vec(fq);
    o.vec(mstart);
    o.vec(mlo);
    o.vec(mfol);
    o.vec(mnext);
    o.num(o.sum.h);
    if (!os) throw std::runtime_error("PackedModel: cannot write the stored file");
}

inline void PackedModel::save(const std::string& path, const double w[6], const Config& cfg) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("PackedModel: cannot write " + path);
    save(os, w, cfg);
    os.close();
    if (!os) throw std::runtime_error("PackedModel: cannot write " + path);
}

inline PackedModel PackedModel::load(std::istream& is, double w[6], Config& cfg) {
    std::uint64_t left = ~std::uint64_t{0};
    if (const auto here = is.tellg(); here >= 0) {
        is.seekg(0, std::ios::end);
        const auto end = is.tellg();
        is.seekg(here);
        if (end >= here) left = static_cast<std::uint64_t>(end - here);
    }
    detail::PackIn in{is, left};
    char magic[sizeof detail::kPackMagic];
    in.raw(magic, sizeof magic);
    if (std::memcmp(magic, detail::kPackMagic, sizeof magic) != 0 || in.num<std::uint32_t>() != detail::kPackVersion)
        throw std::runtime_error("PackedModel: not a stored packed trie (or another version)");
    (void)in.num<std::uint32_t>();
    PackedModel p;
    const auto train = in.num<std::uint64_t>(), sample = in.num<std::uint64_t>();
    p.alpha = in.real();
    p.agree_floor = in.real();
    p.match_cap = in.real();
    double ws[6];
    for (double& v : ws) v = in.real();
    p.uni_n = in.num<std::uint32_t>();
    p.kend = in.num<std::uint32_t>();
    in.array(p.uni, 256);
    in.array(p.o1n, 256);
    in.vec(p.o1);
    in.vec(p.kid);
    in.vec(p.edge);
    in.vec(p.fol);
    in.vec(p.fsym);
    in.vec(p.fq);
    in.vec(p.mstart);
    in.vec(p.mlo);
    in.vec(p.mfol);
    in.vec(p.mnext);
    const std::uint64_t h = in.sum.h;
    if (in.num<std::uint64_t>() != h) throw std::runtime_error("PackedModel: stored file damaged (checksum)");
    if (!p.consistent()) throw std::runtime_error("PackedModel: stored file damaged");
    std::copy(ws, ws + 6, w);
    cfg.train = static_cast<std::size_t>(train);
    cfg.sample = static_cast<std::size_t>(sample);
    cfg.alpha = p.alpha;
    cfg.agree_floor = p.agree_floor;
    cfg.match_cap = p.match_cap;
    return p;
}

inline PackedModel PackedModel::load(const std::string& path, double w[6], Config& cfg) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("PackedModel: cannot open " + path);
    return load(is, w, cfg);
}

// The packed form of a trained model (after compact_match; before it, the match table is empty, as
// Model::find_match then finds nothing). The Model is not changed; m.release() then frees it.
inline PackedModel pack(const Model& m) {
    PackedModel p;
    std::memcpy(p.uni, m.uni, sizeof p.uni);
    p.uni_n = m.uni_n;
    p.o1.assign(&m.o1[0][0], &m.o1[0][0] + 256 * 256);
    std::memcpy(p.o1n, m.o1n, sizeof p.o1n);
    p.alpha = m.alpha;
    p.agree_floor = m.agree_floor;
    p.match_cap = m.match_cap;

    // Kept: a follower seen twice, or a kept descendant. Counted first, so every array is allocated once.
    const std::size_t N = m.nodes.size();
    std::vector<char> keep(N, 0);
    std::size_t nfol = 0, ndeep = 0, n4 = 0;
    for (std::uint32_t id = 1; id < N; ++id) {
        std::size_t k = 0;
        for (std::uint32_t f = m.nodes[id].fol_head; f != kNil; f = m.fols[f].next) k += m.fols[f].count >= 2;
        if (m.nodes[id].depth <= 4) nfol += k;
        for (std::uint32_t a = id; k && a != 0 && a != kNil && !keep[a]; a = m.nodes[a].parent) keep[a] = 1;
    }
    for (std::uint32_t id = 1; id < N; ++id) {
        if (!keep[id] || m.nodes[id].depth < 2 || m.nodes[id].depth > 4) continue;
        ++ndeep;
        n4 += m.nodes[id].depth == 4;
    }
    const std::size_t total = PackedModel::kFirst2 + ndeep;
    p.edge.reserve(total);
    p.kid.reserve(total - n4 + 1);
    p.fol.reserve(ndeep + 1);
    p.fsym.reserve(nfol);
    p.fq.reserve(nfol);

    // Breadth first. src: the training id behind each packed id (kNil: an order-1 byte with nothing kept).
    std::vector<std::uint32_t> src;
    src.reserve(total);
    src.push_back(0);
    p.edge.push_back(0);
    std::vector<std::pair<std::uint8_t, std::uint32_t>> kids;
    std::uint32_t lo = 0, hi = 1;
    for (int d = 0; d < 4; ++d) {
        for (std::uint32_t id = lo; id < hi; ++id) {
            p.kid.push_back(static_cast<std::uint32_t>(src.size()));
            kids.clear();
            if (d == 0) {
                for (int b = 0; b < 256; ++b) {
                    const std::uint32_t c = m.root_child[b];
                    kids.push_back({static_cast<std::uint8_t>(b), c != kNil && keep[c] ? c : kNil});
                }
            } else if (src[id] != kNil) {
                for (std::uint32_t c = m.nodes[src[id]].child_head; c != kNil; c = m.nodes[c].sibling)
                    if (keep[c]) kids.push_back({m.nodes[c].byte, c});
                std::sort(kids.begin(), kids.end());
            }
            for (const auto& [b, c] : kids) {
                p.edge.push_back(b);
                src.push_back(c);
            }
        }
        lo = hi;
        hi = static_cast<std::uint32_t>(src.size());
    }
    p.kend = lo;
    p.kid.push_back(hi);

    std::vector<std::pair<std::uint8_t, std::uint8_t>> fs;
    for (std::uint32_t id = PackedModel::kFirst2; id < src.size(); ++id) {
        p.fol.push_back(static_cast<std::uint32_t>(p.fsym.size()));
        fs.clear();
        for (std::uint32_t f = m.nodes[src[id]].fol_head; f != kNil; f = m.fols[f].next)
            if (m.fols[f].count >= 2) fs.push_back({m.fols[f].sym, quantize(m.fols[f].count)});
        std::sort(fs.begin(), fs.end());
        for (const auto& [s, q] : fs) {
            p.fsym.push_back(s);
            p.fq.push_back(q);
        }
    }
    p.fol.push_back(static_cast<std::uint32_t>(p.fsym.size()));

    // compact_match leaves dense sorted by key, so the buckets fill in order.
    if (m.data != nullptr) {
        p.mstart.assign(65537, 0);
        p.mlo.reserve(m.dense.size());
        p.mfol.reserve(m.dense.size());
        p.mnext.reserve(m.dense.size());
        for (const Hit& h : m.dense) {
            ++p.mstart[(h.key >> 16) + 1];
            p.mlo.push_back(static_cast<std::uint16_t>(h.key & 0xffffu));
            p.mfol.push_back(h.follower);
            p.mnext.push_back(m.data[h.follower]);
        }
        for (std::size_t k = 1; k < p.mstart.size(); ++k) p.mstart[k] += p.mstart[k - 1];
    }
    return p;
}

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
// ModelT is Model or PackedModel.
template <class ModelT>
inline Holdout score_holdout(const ModelT& model, const std::uint8_t* data, const double w[6],
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
// Exits the process on a bad probability, as the tool did. ModelT is Model or PackedModel (no PPMD).
template <class ModelT>
inline Score heldout(const ModelT& m, const std::uint8_t* data, Mode mode, const Config& cfg = Config{},
                     bool ppmd = false, int max_depth = 4) {
    if (!std::is_same_v<ModelT, Model> && ppmd) {
        std::cerr << "PPMD needs the training Model\n";
        std::exit(1);
    }
    double bits = 0, sum_u = 0;
    std::uint64_t matches = 0;
    const double n = static_cast<double>(cfg.sample - cfg.train);
    for (std::size_t i = cfg.train; i < cfg.sample; ++i) {
        double P[256];
        bool matched = false;
        double u = 0.0;
        if constexpr (std::is_same_v<ModelT, Model>)
            u = ppmd ? m.mix_ppmd(data, i, cfg.train, mode, max_depth, P, matched)
                     : m.mix(data, i, cfg.train, mode, P, matched);
        else
            u = m.mix(data, i, cfg.train, mode, P, matched);
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
// nucleus, then one uniform draw per byte from mt19937(seed). ModelT is Model or PackedModel.
template <class ModelT>
inline std::string sample_continuation(const ModelT& model, const double w[6], const std::string& prompt,
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

    // M4: the packed form against the training Model (Mode::Synth, compacted match) on a fixed text with
    // counts above 64 (quantised), singletons (dropped) and matches on both sides of a follower limit. At
    // every prefix of the text plus unseen bytes (the limit alternating between all followers and the first
    // half), E0..E4, the match and mix() must be bit-identical. Sizes and match hits are golden.
    std::string ttext;
    for (int r = 0; r < 66; ++r) ttext += "cat sat. ";
    ttext += "a dog ran far, zq xj vk. the cat ran. a dog sat. a dog sat. ";
    std::vector<std::uint8_t> tb(ttext.begin(), ttext.end());
    Model tm;
    tm.verbose = false;
    tm.data = tb.data();
    Config tc;
    tc.reserve_nodes = 1024;
    tc.reserve_fols = 2048;
    tc.match_bits = 8;
    tm.init(tc);
    for (std::size_t i = 0; i < tb.size(); ++i) tm.observe(i);
    tm.compact_match();
    const PackedModel pk = pack(tm);
    std::string probe = ttext + "the dog sat on a mat. zq xj vk";
    const auto* ph = reinterpret_cast<const std::uint8_t*>(probe.data());
    int differ = 0, hits = 0;
    for (std::size_t len = 0; len <= probe.size(); ++len) {
        const std::size_t limit = len % 2 ? tb.size() : tb.size() / 2;
        double e1[5][256], e2[5][256], P1[256], P2[256];
        bool m1 = false, m2 = false, x1 = false, x2 = false;
        unsigned p1 = 0, p2 = 0;
        tm.fill_experts(ph, len, limit, Mode::Synth, e1, m1, p1, false);
        pk.fill_experts(ph, len, limit, Mode::Synth, e2, m2, p2, false);
        const double u1 = tm.mix(ph, len, limit, Mode::Synth, P1, x1);
        const double u2 = pk.mix(ph, len, limit, Mode::Synth, P2, x2);
        hits += m1;
        if (std::memcmp(e1, e2, sizeof e1) != 0 || std::memcmp(P1, P2, sizeof P1) != 0 ||
            std::memcmp(&u1, &u2, sizeof u1) != 0 || m1 != m2 || p1 != p2 || x1 != x2)
            ++differ;
    }
    // online_match after compact_match (qa's wiki-stage reads): neither form has the training hash, so neither
    // matches, and the experts are the same.
    double eo1[5][256], eo2[5][256];
    bool mo1 = true, mo2 = true;
    unsigned po1 = 0, po2 = 0;
    tm.fill_experts(ph, probe.size(), tb.size(), Mode::Synth, eo1, mo1, po1, true);
    pk.fill_experts(ph, probe.size(), tb.size(), Mode::Synth, eo2, mo2, po2, true);
    const bool online_ok = !mo1 && !mo2 && std::memcmp(eo1, eo2, sizeof eo1) == 0;
    std::uint8_t qmax = 0;
    for (const std::uint8_t q : pk.fq) qmax = std::max(qmax, q);
    if (differ != 0 || hits != 313 || pk.nodes() != 306 || pk.followers() != 54 || pk.matches() != 18 ||
        pk.followers() != measure(tm, Mode::Synth).followers || qmax != 65 || !online_ok) {
        std::cerr << "self-test packed model: differ " << differ << " hits " << hits << " nodes " << pk.nodes()
                  << " followers " << pk.followers() << " matches " << pk.matches() << " qmax " << int(qmax) << "\n";
        return false;
    }

    // M7: the stored file, in memory. The round trip gives the same model (every array), weights and split;
    // a copy with one bit flipped, or cut short, is refused.
    std::size_t image = 0;
    int refused = 0;
    bool same = false;
    try {
        const double tw[6] = {0.1, 0.2, 0.05, 0.15, 0.3, 0.2};
        Config sc;
        sc.train = 600;
        sc.sample = tb.size();
        std::stringstream ss;
        pk.save(ss, tw, sc);
        const std::string img = ss.str();
        image = img.size();
        double bw[6] = {};
        Config bc;
        const PackedModel back = PackedModel::load(ss, bw, bc);
        same = back == pk && std::memcmp(bw, tw, sizeof tw) == 0 && bc.train == 600 && bc.sample == tb.size() &&
               bc.alpha == pk.alpha && bc.match_cap == pk.match_cap;
        std::string flipped = img;
        flipped[img.size() / 2] = static_cast<char>(flipped[img.size() / 2] ^ 1);
        for (const std::string& bad : {flipped, img.substr(0, 60)}) {
            std::istringstream bs(bad);
            try {
                (void)PackedModel::load(bs, bw, bc);
            } catch (const std::runtime_error&) {
                ++refused;
            }
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "self-test stored file: " << e.what() << "\n";
    }
    if (!same || refused != 2 || image != 528449) {
        std::cerr << "self-test stored file: same " << same << " refused " << refused << " bytes " << image << "\n";
        return false;
    }
    return true;
}

}  // namespace cypha::cyphalm::trie
