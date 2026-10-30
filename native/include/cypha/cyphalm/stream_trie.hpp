// Streaming byte-trie model for the 1 TB runs (PLAN Track A; docs/TRAIN_1TB_RUNBOOK.md).
//
// The same model as the live trainer native/tools/trie_stream_train.cpp (orders 0..4 as a byte trie, order-1 and
// unigram tables, a 4-byte match that stores the next byte), with one change: a count about to pass `cap` halves its
// whole table (the node's followers, the order-1 row or the unigram) instead of wrapping past 2^32. The live trainer's
// 32-bit counts wrap on a large corpus (a frequent byte passes 4.29e9 within a few tens of GB), which this header's
// verifier detects in its checkpoints.
//
// Checkpoint files: "TRI1" is the live trainer's old 32-bit layout, read as is. "TRI2" is the same layout followed by
// a trailer (cap and the number of halvings), written by native/tools/trie_stream_train_v2.cpp. "TR64" is the live
// trainer's layout since 2026-10-28: the same fields with 64-bit counts (unigram, order 1, node totals, follower
// counts), 32-byte nodes and 16-byte followers, no trailer.
//
// to_packed() loads a checkpoint into cyphalm::trie::Model (packed_trie.hpp) so today's fixed backoff (Model::mix)
// scores it: the trie, order-1 and unigram tables are byte-compatible; the match becomes Model's compacted table
// (keys seen at least twice) over a side array holding each key's stored next byte. A TR64 checkpoint is scaled on the
// way in: each node's followers are shifted right until the node's total is at most qdec[255] (the largest count the
// packed form's one-byte quantiser keeps apart), followers that reach 0 are dropped, and each order-1 row and the
// unigram are shifted until their totals fit 32 bits. A shift keeps the ratios that the backoff reads.

#pragma once

#include "cypha/cyphalm/packed_trie.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace cypha::cyphalm::stream {

namespace fs = std::filesystem;

constexpr std::uint32_t kNil = 0xffffffffu;
constexpr char kMagic1[4] = {'T', 'R', 'I', '1'};
constexpr char kMagic2[4] = {'T', 'R', 'I', '2'};
constexpr char kMagic64[4] = {'T', 'R', '6', '4'};

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
};

struct Fol {
    std::uint8_t sym = 0;
    std::uint32_t count = 0;
    std::uint32_t next = kNil;
};

static_assert(sizeof(Node) == 28 && sizeof(Fol) == 12, "checkpoint layout");
static_assert(sizeof(trie::Node) == sizeof(Node) && offsetof(trie::Node, depth) == offsetof(Node, depth) &&
                  offsetof(trie::Node, total) == offsetof(Node, total),
              "packed_trie::Node must share the stream layout");
static_assert(sizeof(trie::Fol) == sizeof(Fol) && offsetof(trie::Fol, count) == offsetof(Fol, count) &&
                  offsetof(trie::Fol, next) == offsetof(Fol, next),
              "packed_trie::Fol must share the stream layout");

inline std::uint32_t mix32(std::uint32_t x) {
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
    std::uint32_t root_child[256]{};
    std::vector<std::uint32_t> mk, mv;
    std::vector<std::uint8_t> mb;
    std::vector<std::uint16_t> ms;
    std::uint32_t mmask = 0;
    std::uint32_t mfilled = 0;
    std::uint64_t bytes_seen = 0;
    int hist_n = 0;
    std::uint8_t hist[4]{};
    // TRI2 only: a count reaching `cap` halves its table (cap = 2^32 - 1 by default; small values for tests).
    std::uint32_t cap = 0xffffffffu;
    std::uint64_t halve_uni = 0, halve_o1 = 0, halve_node = 0;

    void init(unsigned match_bits = 20) {
        nodes.clear();
        nodes.push_back(Node{});
        fols.clear();
        for (int i = 0; i < 256; ++i) root_child[i] = kNil;
        mk.assign(std::size_t{1} << match_bits, kNil);
        mv.assign(mk.size(), 0);
        mb.assign(mk.size(), 0);
        ms.assign(mk.size(), 0);
        mmask = static_cast<std::uint32_t>(mk.size() - 1);
        mfilled = 0;
        bytes_seen = 0;
        hist_n = 0;
    }

    void reset_hist() { hist_n = 0; }

    // Half a count, rounded up, so a count of 1 stays 1 (no overflow at 2^32 - 1).
    static std::uint32_t half(std::uint32_t c) { return c / 2u + (c & 1u); }

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

    // Halves every follower count of a node (a count stays at least 1) and recomputes its total.
    void halve_node_counts(std::uint32_t node) {
        std::uint32_t t = 0;
        for (std::uint32_t f = nodes[node].fol_head; f != kNil; f = fols[f].next) {
            fols[f].count = half(fols[f].count);
            t += fols[f].count;
        }
        nodes[node].total = t;
        ++halve_node;
    }

    // Adds n to follower y (merging shards). The node halves until n fits under cap; a node already at one count per
    // follower cannot halve further, so n is clipped instead.
    void add_fol_n(std::uint32_t node, std::uint8_t y, std::uint32_t n) {
        if (n == 0) return;
        while (std::uint64_t{nodes[node].total} + n > cap && nodes[node].total > nodes[node].nfol) halve_node_counts(node);
        if (std::uint64_t{nodes[node].total} + n > cap) {
            if (nodes[node].total >= cap) return;
            n = cap - nodes[node].total;
        }
        std::uint32_t prev = kNil;
        for (std::uint32_t f = nodes[node].fol_head; f != kNil; f = fols[f].next) {
            if (fols[f].sym == y) {
                fols[f].count += n;
                nodes[node].total += n;
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
        f.count = n;
        f.next = nodes[node].fol_head;
        const auto id = static_cast<std::uint32_t>(fols.size());
        fols.push_back(f);
        nodes[node].fol_head = id;
        nodes[node].nfol++;
        nodes[node].total += n;
    }

    void add_fol(std::uint32_t node, std::uint8_t y) {
        if (nodes[node].total >= cap) halve_node_counts(node);
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

    // A shard's match entry: `times` more sightings (saturating), `seen` more for the 16-bit count, y the later follower.
    void match_add(std::uint32_t key, std::uint8_t y, std::uint32_t times, std::uint16_t seen) {
        match_note(key, y);  // inserts or counts once, growing the table if needed
        std::uint32_t i = mix32(key) & mmask;
        while (mk[i] != key) i = (i + 1u) & mmask;
        const std::uint64_t t = std::uint64_t{mv[i]} + times - 1;
        mv[i] = t > 0xffffffffu ? 0xffffffffu : static_cast<std::uint32_t>(t);
        const std::uint32_t s = std::uint32_t{ms[i]} + seen - 1;
        ms[i] = static_cast<std::uint16_t>(std::min<std::uint32_t>(s, 65535u));
    }

    void match_note(std::uint32_t key, std::uint8_t y) {
        if ((static_cast<std::uint64_t>(mfilled) + 1) * 10 >= mk.size() * 7) {
            auto ok = mk, ov = mv;
            auto ob = mb;
            auto os = ms;
            mk.assign(ok.size() * 2, kNil);
            mv.assign(mk.size(), 0);
            mb.assign(mk.size(), 0);
            ms.assign(mk.size(), 0);
            mmask = static_cast<std::uint32_t>(mk.size() - 1);
            mfilled = 0;
            for (std::size_t i = 0; i < ok.size(); ++i) {
                if (ok[i] == kNil) continue;
                std::uint32_t j = mix32(ok[i]) & mmask;
                while (mk[j] != kNil) j = (j + 1u) & mmask;
                mk[j] = ok[i];
                mv[j] = ov[i];
                mb[j] = ob[i];
                ms[j] = os[i];
                ++mfilled;
            }
        }
        std::uint32_t i = mix32(key) & mmask;
        for (;;) {
            if (mk[i] == kNil) {
                mk[i] = key;
                mv[i] = 1;
                mb[i] = y;
                ms[i] = 1;
                ++mfilled;
                return;
            }
            if (mk[i] == key) {
                if (mv[i] < 0xffffffffu) ++mv[i];
                mb[i] = y;
                if (ms[i] < 65535) ++ms[i];
                return;
            }
            i = (i + 1u) & mmask;
        }
    }

    void observe_byte(std::uint8_t y) {
        if (uni_n >= cap) {  // halve the unigram table
            std::uint32_t t = 0;
            for (auto& c : uni) t += (c = half(c));
            uni_n = t;
            ++halve_uni;
        }
        ++uni[y];
        ++uni_n;
        ++bytes_seen;
        if (hist_n >= 1) {
            const std::uint8_t p = hist[0];
            if (o1n[p] >= cap) {
                std::uint32_t t = 0;
                for (auto& c : o1[p]) t += (c = half(c));
                o1n[p] = t;
                ++halve_o1;
            }
            ++o1[p][y];
            ++o1n[p];
        }
        const int maxd = std::min(4, hist_n);
        std::uint32_t n = 0;
        for (int d = 1; d <= maxd; ++d) {
            n = ensure_child(n, hist[d - 1]);
            if (d >= 2) add_fol(n, y);
        }
        if (hist_n >= 4) {
            const std::uint32_t key = (std::uint32_t(hist[3]) << 24) | (std::uint32_t(hist[2]) << 16) |
                                      (std::uint32_t(hist[1]) << 8) | std::uint32_t(hist[0]);
            match_note(key, y);
        }
        if (hist_n < 4) {
            for (int i = hist_n; i > 0; --i) hist[i] = hist[i - 1];
            hist[0] = y;
            ++hist_n;
        } else {
            hist[3] = hist[2];
            hist[2] = hist[1];
            hist[1] = hist[0];
            hist[0] = y;
        }
    }
};

// ---- TR64: the live trainer's 64-bit model (native/tools/trie_stream_train.cpp)

struct Node64 {
    std::uint32_t parent = kNil;
    std::uint32_t child_head = kNil;
    std::uint32_t sibling = kNil;
    std::uint32_t fol_head = kNil;
    std::uint64_t total = 0;
    std::uint16_t nchild = 0;
    std::uint16_t nfol = 0;
    std::uint8_t byte = 0;
    std::uint8_t depth = 0;
};

struct Fol64 {
    std::uint64_t count = 0;
    std::uint32_t next = kNil;
    std::uint8_t sym = 0;
};

static_assert(sizeof(Node64) == 32 && sizeof(Fol64) == 16, "TR64 checkpoint layout");

struct Model64 {
    std::uint64_t uni[256]{};
    std::uint64_t uni_n = 0;
    std::vector<std::uint64_t> o1 = std::vector<std::uint64_t>(256 * 256);  // row = the previous byte
    std::uint64_t o1n[256]{};
    std::vector<Node64> nodes;
    std::vector<Fol64> fols;
    std::uint32_t root_child[256]{};
    std::vector<std::uint32_t> mk, mv;
    std::vector<std::uint8_t> mb;
    std::vector<std::uint16_t> ms;
    std::uint32_t mfilled = 0;
    std::uint64_t bytes_seen = 0;
};

// ---- Checkpoint files

struct Header {
    bool tri2 = false;
    bool tr64 = false;
    std::uint32_t nn = 0, nf = 0, slots = 0, mfilled = 0;
    std::uint64_t bytes_seen = 0;
    std::uint64_t file_bytes = 0, expected_bytes = 0;
    std::uint32_t cap = 0xffffffffu;
    std::uint64_t halve_uni = 0, halve_o1 = 0, halve_node = 0;
    // TR64 through to_packed: nodes whose followers were shifted, followers dropped by the shift, the largest node
    // shift, the unigram's shift and the order-1 rows shifted.
    std::uint64_t scaled_nodes = 0, dropped_fols = 0, scaled_o1_rows = 0;
    unsigned max_shift = 0, uni_shift = 0;
};

inline const char* format_name(const Header& h) { return h.tr64 ? "TR64" : h.tri2 ? "TRI2" : "TRI1"; }

constexpr std::uint64_t kTrailerBytes = 4 + 8 * 3;

// Bytes before the node array: magic, nn, nf, bytes_seen, unigram, uni_n, order 1, its row totals, root children.
inline std::uint64_t fixed_bytes(const Header& h) {
    const std::uint64_t c = h.tr64 ? 8 : 4;
    return 4 + 4 + 4 + 8 + c * 256 + c + c * 256 * 256 + c * 256 + 4 * 256;
}

inline std::uint64_t expected_size(const Header& h) {
    std::uint64_t s = fixed_bytes(h);
    s += static_cast<std::uint64_t>(h.nn) * (h.tr64 ? sizeof(Node64) : sizeof(Node)) +
         static_cast<std::uint64_t>(h.nf) * (h.tr64 ? sizeof(Fol64) : sizeof(Fol));
    s += 4 + 4 + static_cast<std::uint64_t>(h.slots) * (4 + 4 + 1 + 2);
    if (h.tri2) s += kTrailerBytes;
    return s;
}

// Writes `m` as TRI2 (tri2 = true, with the trailer) or TRI1 (the live trainer's layout).
inline bool save(const fs::path& path, const Model& m, bool tri2 = true) {
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out) return false;
        out.write(tri2 ? kMagic2 : kMagic1, 4);
        const auto nn = static_cast<std::uint32_t>(m.nodes.size());
        const auto nf = static_cast<std::uint32_t>(m.fols.size());
        out.write(reinterpret_cast<const char*>(&nn), 4);
        out.write(reinterpret_cast<const char*>(&nf), 4);
        out.write(reinterpret_cast<const char*>(&m.bytes_seen), 8);
        out.write(reinterpret_cast<const char*>(m.uni), sizeof m.uni);
        out.write(reinterpret_cast<const char*>(&m.uni_n), 4);
        out.write(reinterpret_cast<const char*>(m.o1), sizeof m.o1);
        out.write(reinterpret_cast<const char*>(m.o1n), sizeof m.o1n);
        out.write(reinterpret_cast<const char*>(m.root_child), sizeof m.root_child);
        out.write(reinterpret_cast<const char*>(m.nodes.data()), static_cast<std::streamsize>(nn) * sizeof(Node));
        out.write(reinterpret_cast<const char*>(m.fols.data()), static_cast<std::streamsize>(nf) * sizeof(Fol));
        const auto slots = static_cast<std::uint32_t>(m.mk.size());
        out.write(reinterpret_cast<const char*>(&slots), 4);
        out.write(reinterpret_cast<const char*>(&m.mfilled), 4);
        out.write(reinterpret_cast<const char*>(m.mk.data()), static_cast<std::streamsize>(slots) * 4);
        out.write(reinterpret_cast<const char*>(m.mv.data()), static_cast<std::streamsize>(slots) * 4);
        out.write(reinterpret_cast<const char*>(m.mb.data()), static_cast<std::streamsize>(slots));
        out.write(reinterpret_cast<const char*>(m.ms.data()), static_cast<std::streamsize>(slots) * 2);
        if (tri2) {
            out.write(reinterpret_cast<const char*>(&m.cap), 4);
            out.write(reinterpret_cast<const char*>(&m.halve_uni), 8);
            out.write(reinterpret_cast<const char*>(&m.halve_o1), 8);
            out.write(reinterpret_cast<const char*>(&m.halve_node), 8);
        }
        out.flush();
        if (!out) return false;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(path, ec);
        fs::rename(tmp, path, ec);
    }
    return !ec;
}

// Reads the header fields only (and the trailer for TRI2) without loading the arrays.
inline bool read_header(const fs::path& path, Header& h, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return err = "cannot open " + path.string(), false;
    std::error_code ec;
    h.file_bytes = fs::file_size(path, ec);
    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, kMagic1, 4) == 0) h.tri2 = false;
    else if (std::memcmp(magic, kMagic2, 4) == 0) h.tri2 = true;
    else if (std::memcmp(magic, kMagic64, 4) == 0) h.tr64 = true;
    else return err = "bad magic", false;
    in.read(reinterpret_cast<char*>(&h.nn), 4);
    in.read(reinterpret_cast<char*>(&h.nf), 4);
    in.read(reinterpret_cast<char*>(&h.bytes_seen), 8);
    const std::uint64_t at_slots = fixed_bytes(h) +
                                   static_cast<std::uint64_t>(h.nn) * (h.tr64 ? sizeof(Node64) : sizeof(Node)) +
                                   static_cast<std::uint64_t>(h.nf) * (h.tr64 ? sizeof(Fol64) : sizeof(Fol));
    in.seekg(static_cast<std::streamoff>(at_slots));
    in.read(reinterpret_cast<char*>(&h.slots), 4);
    in.read(reinterpret_cast<char*>(&h.mfilled), 4);
    if (!in) return err = "truncated before the match table", false;
    h.expected_bytes = expected_size(h);
    if (h.tri2 && h.file_bytes == h.expected_bytes) {
        in.seekg(static_cast<std::streamoff>(h.expected_bytes - kTrailerBytes));
        in.read(reinterpret_cast<char*>(&h.cap), 4);
        in.read(reinterpret_cast<char*>(&h.halve_uni), 8);
        in.read(reinterpret_cast<char*>(&h.halve_o1), 8);
        in.read(reinterpret_cast<char*>(&h.halve_node), 8);
    }
    return true;
}

inline bool load(const fs::path& path, Model& m, Header& h, std::string& err) {
    if (!read_header(path, h, err)) return false;
    if (h.tr64) return err = "TR64 holds 64-bit counts: load it with load64", false;
    if (h.file_bytes != h.expected_bytes) {
        err = "size " + std::to_string(h.file_bytes) + " != expected " + std::to_string(h.expected_bytes);
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    in.seekg(4 + 4 + 4 + 8);
    m.bytes_seen = h.bytes_seen;
    in.read(reinterpret_cast<char*>(m.uni), sizeof m.uni);
    in.read(reinterpret_cast<char*>(&m.uni_n), 4);
    in.read(reinterpret_cast<char*>(m.o1), sizeof m.o1);
    in.read(reinterpret_cast<char*>(m.o1n), sizeof m.o1n);
    in.read(reinterpret_cast<char*>(m.root_child), sizeof m.root_child);
    m.nodes.resize(h.nn);
    m.fols.resize(h.nf);
    in.read(reinterpret_cast<char*>(m.nodes.data()), static_cast<std::streamsize>(h.nn) * sizeof(Node));
    in.read(reinterpret_cast<char*>(m.fols.data()), static_cast<std::streamsize>(h.nf) * sizeof(Fol));
    std::uint32_t slots = 0;
    in.read(reinterpret_cast<char*>(&slots), 4);
    in.read(reinterpret_cast<char*>(&m.mfilled), 4);
    m.mk.resize(slots);
    m.mv.resize(slots);
    m.mb.resize(slots);
    m.ms.resize(slots);
    in.read(reinterpret_cast<char*>(m.mk.data()), static_cast<std::streamsize>(slots) * 4);
    in.read(reinterpret_cast<char*>(m.mv.data()), static_cast<std::streamsize>(slots) * 4);
    in.read(reinterpret_cast<char*>(m.mb.data()), static_cast<std::streamsize>(slots));
    in.read(reinterpret_cast<char*>(m.ms.data()), static_cast<std::streamsize>(slots) * 2);
    m.mmask = slots ? slots - 1 : 0;
    m.cap = h.cap;
    m.halve_uni = h.halve_uni;
    m.halve_o1 = h.halve_o1;
    m.halve_node = h.halve_node;
    if (!in) return err = "read failed", false;
    return true;
}

// Reads a TR64 checkpoint in full.
inline bool load64(const fs::path& path, Model64& m, Header& h, std::string& err) {
    if (!read_header(path, h, err)) return false;
    if (!h.tr64) return err = "not a TR64 checkpoint", false;
    if (h.file_bytes != h.expected_bytes) {
        err = "size " + std::to_string(h.file_bytes) + " != expected " + std::to_string(h.expected_bytes);
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    in.seekg(4 + 4 + 4 + 8);
    m.bytes_seen = h.bytes_seen;
    in.read(reinterpret_cast<char*>(m.uni), sizeof m.uni);
    in.read(reinterpret_cast<char*>(&m.uni_n), 8);
    in.read(reinterpret_cast<char*>(m.o1.data()), static_cast<std::streamsize>(m.o1.size() * 8));
    in.read(reinterpret_cast<char*>(m.o1n), sizeof m.o1n);
    in.read(reinterpret_cast<char*>(m.root_child), sizeof m.root_child);
    m.nodes.resize(h.nn);
    m.fols.resize(h.nf);
    in.read(reinterpret_cast<char*>(m.nodes.data()), static_cast<std::streamsize>(h.nn) * sizeof(Node64));
    in.read(reinterpret_cast<char*>(m.fols.data()), static_cast<std::streamsize>(h.nf) * sizeof(Fol64));
    std::uint32_t slots = 0;
    in.read(reinterpret_cast<char*>(&slots), 4);
    in.read(reinterpret_cast<char*>(&m.mfilled), 4);
    m.mk.resize(slots);
    m.mv.resize(slots);
    m.mb.resize(slots);
    m.ms.resize(slots);
    in.read(reinterpret_cast<char*>(m.mk.data()), static_cast<std::streamsize>(slots) * 4);
    in.read(reinterpret_cast<char*>(m.mv.data()), static_cast<std::streamsize>(slots) * 4);
    in.read(reinterpret_cast<char*>(m.mb.data()), static_cast<std::streamsize>(slots));
    in.read(reinterpret_cast<char*>(m.ms.data()), static_cast<std::streamsize>(slots) * 2);
    if (!in) return err = "read failed", false;
    return true;
}

// Writes `m` as TR64, the live trainer's layout (for tests; the trainer writes its own).
inline bool save64(const fs::path& path, const Model64& m) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(kMagic64, 4);
    const auto nn = static_cast<std::uint32_t>(m.nodes.size());
    const auto nf = static_cast<std::uint32_t>(m.fols.size());
    out.write(reinterpret_cast<const char*>(&nn), 4);
    out.write(reinterpret_cast<const char*>(&nf), 4);
    out.write(reinterpret_cast<const char*>(&m.bytes_seen), 8);
    out.write(reinterpret_cast<const char*>(m.uni), sizeof m.uni);
    out.write(reinterpret_cast<const char*>(&m.uni_n), 8);
    out.write(reinterpret_cast<const char*>(m.o1.data()), static_cast<std::streamsize>(m.o1.size() * 8));
    out.write(reinterpret_cast<const char*>(m.o1n), sizeof m.o1n);
    out.write(reinterpret_cast<const char*>(m.root_child), sizeof m.root_child);
    out.write(reinterpret_cast<const char*>(m.nodes.data()), static_cast<std::streamsize>(nn) * sizeof(Node64));
    out.write(reinterpret_cast<const char*>(m.fols.data()), static_cast<std::streamsize>(nf) * sizeof(Fol64));
    const auto slots = static_cast<std::uint32_t>(m.mk.size());
    out.write(reinterpret_cast<const char*>(&slots), 4);
    out.write(reinterpret_cast<const char*>(&m.mfilled), 4);
    out.write(reinterpret_cast<const char*>(m.mk.data()), static_cast<std::streamsize>(slots) * 4);
    out.write(reinterpret_cast<const char*>(m.mv.data()), static_cast<std::streamsize>(slots) * 4);
    out.write(reinterpret_cast<const char*>(m.mb.data()), static_cast<std::streamsize>(slots));
    out.write(reinterpret_cast<const char*>(m.ms.data()), static_cast<std::streamsize>(slots) * 2);
    return static_cast<bool>(out);
}

// A 32-bit model widened to TR64, every count multiplied by `scale` (tests: scale 1 is the same model, a large scale
// is one that to_packed must shift back).
inline Model64 widen(const Model& m, std::uint64_t scale = 1) {
    Model64 w;
    for (int s = 0; s < 256; ++s) w.uni[s] = std::uint64_t{m.uni[s]} * scale;
    w.uni_n = std::uint64_t{m.uni_n} * scale;
    for (int p = 0; p < 256; ++p) {
        for (int y = 0; y < 256; ++y) w.o1[static_cast<std::size_t>(p) * 256 + static_cast<std::size_t>(y)] =
            std::uint64_t{m.o1[p][y]} * scale;
        w.o1n[p] = std::uint64_t{m.o1n[p]} * scale;
    }
    std::memcpy(w.root_child, m.root_child, sizeof w.root_child);
    w.nodes.resize(m.nodes.size());
    for (std::size_t i = 0; i < m.nodes.size(); ++i) {
        const Node& a = m.nodes[i];
        Node64& b = w.nodes[i];
        b.parent = a.parent;
        b.child_head = a.child_head;
        b.sibling = a.sibling;
        b.fol_head = a.fol_head;
        b.total = std::uint64_t{a.total} * scale;
        b.nchild = a.nchild;
        b.nfol = a.nfol;
        b.byte = a.byte;
        b.depth = a.depth;
    }
    w.fols.resize(m.fols.size());
    for (std::size_t i = 0; i < m.fols.size(); ++i) {
        w.fols[i].count = std::uint64_t{m.fols[i].count} * scale;
        w.fols[i].next = m.fols[i].next;
        w.fols[i].sym = m.fols[i].sym;
    }
    w.mk = m.mk;
    w.mv = m.mv;
    w.mb = m.mb;
    w.ms = m.ms;
    w.mfilled = m.mfilled;
    w.bytes_seen = m.bytes_seen * scale;
    return w;
}

// The smallest right shift that brings `total` to at most `top`.
inline unsigned shift_to(std::uint64_t total, std::uint64_t top) {
    unsigned k = 0;
    while ((total >> k) > top) ++k;
    return k;
}

// TR64 into packed_trie's Model, scaled as the header comment says. Statistics go to `h`.
inline void scale64_into(Model64& m, trie::Model& pm, Header& h) {
    const std::uint64_t top = trie::qdec[255];
    h.uni_shift = shift_to(m.uni_n, 0xffffffffu);
    std::uint64_t un = 0;
    for (int s = 0; s < 256; ++s) un += pm.uni[s] = static_cast<std::uint32_t>(m.uni[s] >> h.uni_shift);
    pm.uni_n = static_cast<std::uint32_t>(un);
    for (int p = 0; p < 256; ++p) {
        const unsigned k = shift_to(m.o1n[p], 0xffffffffu);
        h.scaled_o1_rows += k > 0;
        std::uint64_t t = 0;
        for (int y = 0; y < 256; ++y)
            t += pm.o1[p][y] =
                static_cast<std::uint32_t>(m.o1[static_cast<std::size_t>(p) * 256 + static_cast<std::size_t>(y)] >> k);
        pm.o1n[p] = static_cast<std::uint32_t>(t);
    }
    std::memcpy(pm.root_child, m.root_child, sizeof pm.root_child);
    std::vector<std::uint64_t>().swap(m.o1);

    const std::size_t nn = m.nodes.size();
    std::size_t kept = 0;
    for (std::size_t i = 0; i < nn; ++i) {
        const unsigned k = shift_to(m.nodes[i].total, top);
        for (auto f = m.nodes[i].fol_head; f != kNil; f = m.fols[f].next) kept += (m.fols[f].count >> k) > 0;
    }
    pm.nodes.resize(nn);
    pm.fols.clear();
    pm.fols.reserve(kept);
    for (std::size_t i = 0; i < nn; ++i) {
        const Node64& a = m.nodes[i];
        trie::Node& b = pm.nodes[i];
        b.parent = a.parent;
        b.child_head = a.child_head;
        b.sibling = a.sibling;
        b.nchild = a.nchild;
        b.byte = a.byte;
        b.depth = a.depth;
        b.chain_open = 1;
        b.fol_head = kNil;
        b.total = 0;
        b.nfol = 0;
        const unsigned k = shift_to(a.total, top);
        if (k > 0) ++h.scaled_nodes;
        h.max_shift = std::max(h.max_shift, k);
        std::uint32_t last = kNil;
        std::uint64_t t = 0;
        for (auto f = a.fol_head; f != kNil; f = m.fols[f].next) {
            const std::uint64_t c = m.fols[f].count >> k;
            if (c == 0) {
                ++h.dropped_fols;
                continue;
            }
            const auto id = static_cast<std::uint32_t>(pm.fols.size());
            trie::Fol nf;
            nf.sym = m.fols[f].sym;
            nf.count = static_cast<std::uint32_t>(c);
            pm.fols.push_back(nf);
            if (last == kNil) b.fol_head = id;
            else pm.fols[last].next = id;
            last = id;
            t += c;
            ++b.nfol;
        }
        b.total = static_cast<std::uint32_t>(t);
    }
    std::vector<Node64>().swap(m.nodes);
    std::vector<Fol64>().swap(m.fols);
}

// Loads a checkpoint into packed_trie's training Model for scoring with Model::mix (Mode::Full or Synth).
// `side` receives one stored next byte per kept match key; pm.data points into it. Peak memory is about the
// checkpoint's size for TRI1 / TRI2 (the trie arrays are read in place, the match is compacted while it is read) and
// about the checkpoint plus the 32-bit model for TR64 (scaled into it, then freed).
inline bool to_packed(const fs::path& path, trie::Model& pm, std::vector<std::uint8_t>& side, Header& h,
                      std::string& err) {
    if (!read_header(path, h, err)) return false;
    if (h.file_bytes != h.expected_bytes) {
        err = "size " + std::to_string(h.file_bytes) + " != expected " + std::to_string(h.expected_bytes);
        return false;
    }
    if (h.tr64) {
        Model64 m;
        if (!load64(path, m, h, err)) return false;
        trie::Config cfg;
        cfg.reserve_nodes = 1;
        cfg.reserve_fols = 1;
        cfg.match_bits = 1;
        pm.verbose = false;
        pm.init(cfg);
        scale64_into(m, pm, h);
        pm.dense.clear();
        side.clear();
        for (std::size_t i = 0; i < m.mk.size(); ++i) {
            if (m.mk[i] == kNil || m.mv[i] < 2) continue;
            pm.dense.push_back(trie::Hit{m.mk[i], static_cast<std::uint32_t>(side.size())});
            side.push_back(m.mb[i]);
        }
        std::sort(pm.dense.begin(), pm.dense.end(),
                  [](const trie::Hit& a, const trie::Hit& b) { return a.key < b.key; });
        pm.match_seen = m.mfilled;
        std::vector<std::uint32_t>().swap(pm.mk);
        std::vector<std::uint32_t>().swap(pm.mv);
        std::vector<std::uint32_t>().swap(pm.ms);
        static const std::uint8_t kNone = 0;
        pm.data = side.empty() ? &kNone : side.data();
        return true;
    }
    trie::Config cfg;
    cfg.reserve_nodes = 1;
    cfg.reserve_fols = 1;
    cfg.match_bits = 1;
    pm.verbose = false;
    pm.init(cfg);
    std::ifstream in(path, std::ios::binary);
    in.seekg(4 + 4 + 4 + 8);
    in.read(reinterpret_cast<char*>(pm.uni), sizeof pm.uni);
    in.read(reinterpret_cast<char*>(&pm.uni_n), 4);
    in.read(reinterpret_cast<char*>(pm.o1), sizeof pm.o1);
    in.read(reinterpret_cast<char*>(pm.o1n), sizeof pm.o1n);
    in.read(reinterpret_cast<char*>(pm.root_child), sizeof pm.root_child);
    pm.nodes.resize(h.nn);
    pm.fols.resize(h.nf);
    in.read(reinterpret_cast<char*>(pm.nodes.data()), static_cast<std::streamsize>(h.nn) * sizeof(Node));
    in.read(reinterpret_cast<char*>(pm.fols.data()), static_cast<std::streamsize>(h.nf) * sizeof(Fol));
    for (auto& n : pm.nodes) n.chain_open = 1;  // the stream trainers leave this byte as padding
    std::uint32_t slots = 0, mfilled = 0;
    in.read(reinterpret_cast<char*>(&slots), 4);
    in.read(reinterpret_cast<char*>(&mfilled), 4);
    std::vector<std::uint32_t> mk(slots), mv(slots);
    std::vector<std::uint8_t> mb(slots);
    in.read(reinterpret_cast<char*>(mk.data()), static_cast<std::streamsize>(slots) * 4);
    in.read(reinterpret_cast<char*>(mv.data()), static_cast<std::streamsize>(slots) * 4);
    in.read(reinterpret_cast<char*>(mb.data()), static_cast<std::streamsize>(slots));
    if (!in) return err = "read failed", false;
    pm.dense.clear();
    side.clear();
    for (std::size_t i = 0; i < slots; ++i) {
        if (mk[i] == kNil || mv[i] < 2) continue;  // packed_trie keeps matches seen at least twice
        pm.dense.push_back(trie::Hit{mk[i], static_cast<std::uint32_t>(side.size())});
        side.push_back(mb[i]);
    }
    std::sort(pm.dense.begin(), pm.dense.end(), [](const trie::Hit& a, const trie::Hit& b) { return a.key < b.key; });
    pm.match_seen = mfilled;
    std::vector<std::uint32_t>().swap(pm.mk);
    std::vector<std::uint32_t>().swap(pm.mv);
    std::vector<std::uint32_t>().swap(pm.ms);
    pm.data = side.empty() ? nullptr : side.data();
    if (side.empty()) {
        static const std::uint8_t kNone = 0;
        pm.data = &kNone;  // find_match needs a data pointer; with no kept keys it never reads it
    }
    return true;
}

// ---- Corpus files, as the live trainer reads them

// Paths holding "/trie/" (the trainers' output), "/labelled/" or "/sha256/" (scripts/verify_corpus_sha.py's lists) are
// skipped, and so are unfinished *.part files.
inline bool trainable(const fs::path& p) {
    const std::string s = p.generic_string();
    if (s.find("/labelled/") != std::string::npos || s.find("/trie/") != std::string::npos ||
        s.find("/sha256/") != std::string::npos)
        return false;
    const std::string name = p.filename().string();
    if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".part") == 0) return false;
    const auto ext = p.extension().string();
    return ext == ".txt" || ext == ".jsonl";
}

inline std::string json_string(const std::string& line, std::size_t& i) {
    std::string out;
    if (i >= line.size() || line[i] != '"') return out;
    ++i;
    while (i < line.size()) {
        char c = line[i++];
        if (c == '"') break;
        if (c == '\\' && i < line.size()) {
            char e = line[i++];
            if (e == 'n') out.push_back('\n');
            else if (e == 't') out.push_back('\t');
            else if (e == 'u' && i + 4 <= line.size()) i += 4;
            else out.push_back(e);
        } else {
            out.push_back(c);
        }
    }
    return out;
}

inline bool extract_text(const std::string& line, std::string& text) {
    const auto key = line.find("\"text\"");
    if (key == std::string::npos) return false;
    auto i = line.find(':', key + 6);
    if (i == std::string::npos) return false;
    ++i;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i >= line.size() || line[i] != '"') return false;
    text = json_string(line, i);
    return !text.empty();
}

// The bytes the trainers observe from one file (text of each .jsonl record, or the whole .txt). With `m` null it
// only counts them (the verifier's --deep check).
inline bool train_file(Model* m, const fs::path& path, std::uint64_t& n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    n = 0;
    if (path.extension().string() == ".jsonl") {
        std::string line, text;
        while (std::getline(in, line)) {
            if (!extract_text(line, text)) continue;
            if (m) {
                m->reset_hist();
                for (unsigned char c : text) m->observe_byte(c);
            }
            n += text.size();
        }
    } else {
        static thread_local std::vector<char> chunk(1 << 16);
        if (m) m->reset_hist();
        while (in) {
            in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            const auto got = in.gcount();
            if (m)
                for (std::streamsize i = 0; i < got; ++i) m->observe_byte(static_cast<std::uint8_t>(chunk[i]));
            n += static_cast<std::uint64_t>(got);
        }
    }
    return true;
}

// ---- Parallel training (trie_stream_train_v2 --threads)
//
// A file is cut into one byte range per thread (a .jsonl cut moves to the next line start, so every record is read
// whole by one thread); each thread trains its own shard model; the shards merge pairwise in parallel, later data into
// earlier, and the result merges into the main model. On a .jsonl file the counts, totals and match entries equal the
// sequential trainer's (no count reaches the cap); a .txt file differs only by the history reset at each cut.

// Adds src's counts to dst. Tables and nodes that would pass dst.cap halve first, as observe_byte does.
inline void merge(Model& dst, const Model& src) {
    auto fold = [&](std::uint32_t* c, std::uint32_t& total, const std::uint32_t* add, std::uint64_t& halvings) {
        std::uint64_t w[256], t = 0;
        for (int s = 0; s < 256; ++s) t += w[s] = std::uint64_t{c[s]} + add[s];
        while (t > dst.cap) {
            std::uint64_t t2 = 0;
            for (auto& x : w) t2 += x = x / 2 + (x & 1);
            ++halvings;
            if (t2 == t) break;
            t = t2;
        }
        for (int s = 0; s < 256; ++s) c[s] = static_cast<std::uint32_t>(std::min<std::uint64_t>(w[s], 0xffffffffu));
        total = static_cast<std::uint32_t>(std::min<std::uint64_t>(t, 0xffffffffu));
    };
    fold(dst.uni, dst.uni_n, src.uni, dst.halve_uni);
    for (int p = 0; p < 256; ++p) fold(dst.o1[p], dst.o1n[p], src.o1[p], dst.halve_o1);
    const auto walk = [&](auto&& self, std::uint32_t dnode, std::uint32_t snode) -> void {
        for (std::uint32_t f = src.nodes[snode].fol_head; f != kNil; f = src.fols[f].next)
            dst.add_fol_n(dnode, src.fols[f].sym, src.fols[f].count);
        if (snode == 0) {
            for (int b = 0; b < 256; ++b)
                if (src.root_child[b] != kNil)
                    self(self, dst.ensure_child(0, static_cast<std::uint8_t>(b)), src.root_child[b]);
            return;
        }
        for (std::uint32_t c = src.nodes[snode].child_head; c != kNil; c = src.nodes[c].sibling)
            self(self, dst.ensure_child(dnode, src.nodes[c].byte), c);
    };
    if (!src.nodes.empty()) walk(walk, 0, 0);
    for (std::size_t i = 0; i < src.mk.size(); ++i)
        if (src.mk[i] != kNil && src.mv[i] > 0) dst.match_add(src.mk[i], src.mb[i], src.mv[i], src.ms[i]);
    dst.bytes_seen += src.bytes_seen;
    dst.halve_uni += src.halve_uni;
    dst.halve_o1 += src.halve_o1;
    dst.halve_node += src.halve_node;
}

// train_file over the bytes [begin, end) of a file: whole .jsonl records starting in it, or the raw .txt bytes.
inline bool train_range(Model* m, const fs::path& path, std::uint64_t begin, std::uint64_t end, std::uint64_t& n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    n = 0;
    in.seekg(static_cast<std::streamoff>(begin));
    std::uint64_t pos = begin;
    if (path.extension().string() == ".jsonl") {
        std::string line, text;
        while (pos < end && std::getline(in, line)) {
            pos += line.size() + 1;
            if (!extract_text(line, text)) continue;
            if (m) {
                m->reset_hist();
                for (unsigned char c : text) m->observe_byte(c);
            }
            n += text.size();
        }
    } else {
        std::vector<char> chunk(1 << 16);
        if (m) m->reset_hist();
        while (pos < end) {
            const auto want = std::min<std::uint64_t>(chunk.size(), end - pos);
            in.read(chunk.data(), static_cast<std::streamsize>(want));
            const auto got = static_cast<std::uint64_t>(in.gcount());
            if (got == 0) break;
            if (m)
                for (std::uint64_t i = 0; i < got; ++i) m->observe_byte(static_cast<std::uint8_t>(chunk[i]));
            pos += got;
            n += got;
        }
    }
    return true;
}

// Cut points for `parts` ranges; a .jsonl cut moves past the line it lands in.
inline std::vector<std::uint64_t> cut_file(const fs::path& path, unsigned parts) {
    std::error_code ec;
    const std::uint64_t size = fs::file_size(path, ec);
    std::vector<std::uint64_t> cuts(parts + 1, size);
    cuts[0] = 0;
    std::ifstream in(path, std::ios::binary);
    const bool jsonl = path.extension().string() == ".jsonl";
    for (unsigned i = 1; i < parts; ++i) {
        std::uint64_t at = size * i / parts;
        if (jsonl && in) {
            in.clear();
            in.seekg(static_cast<std::streamoff>(at));
            std::string skip;
            std::getline(in, skip);
            const auto t = in.tellg();
            at = in && t >= 0 ? static_cast<std::uint64_t>(t) : size;
        }
        cuts[i] = std::max(at, cuts[i - 1]);
    }
    return cuts;
}

// Trains one file on `threads` threads into `main` (threads <= 1: train_file). `match_bits` sizes each shard's
// starting match table.
inline bool train_file_parallel(Model& main, const fs::path& path, unsigned threads, std::uint64_t& n,
                                unsigned match_bits = 16) {
    if (threads <= 1) return train_file(&main, path, n);
    const std::vector<std::uint64_t> cuts = cut_file(path, threads);
    std::vector<Model> shards(threads);
    std::vector<std::uint64_t> got(threads, 0);
    std::vector<char> ok(threads, 1);
    {
        std::vector<std::thread> ws;
        for (unsigned i = 0; i < threads; ++i)
            ws.emplace_back([&, i] {
                shards[i].init(match_bits);
                shards[i].cap = main.cap;
                if (cuts[i] < cuts[i + 1]) ok[i] = train_range(&shards[i], path, cuts[i], cuts[i + 1], got[i]);
            });
        for (auto& w : ws) w.join();
    }
    if (std::find(ok.begin(), ok.end(), 0) != ok.end()) return false;
    for (unsigned step = 1; step < threads; step *= 2) {
        std::vector<std::thread> ws;
        for (unsigned i = 0; i + step < threads; i += 2 * step)
            ws.emplace_back([&, i, step] {
                merge(shards[i], shards[i + step]);
                shards[i + step] = Model{};
            });
        for (auto& w : ws) w.join();
    }
    merge(main, shards[0]);
    n = 0;
    for (auto g : got) n += g;
    return true;
}

}  // namespace cypha::cyphalm::stream
