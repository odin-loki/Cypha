// Streaming byte-trie model for the 1 TB runs (PLAN Track A; docs/TRAIN_1TB_RUNBOOK.md).
//
// The same model as the live trainer native/tools/trie_stream_train.cpp (orders 0..4 as a byte trie, order-1 and
// unigram tables, a 4-byte match that stores the next byte), with one change: a count about to pass `cap` halves its
// whole table (the node's followers, the order-1 row or the unigram) instead of wrapping past 2^32. The live trainer's
// 32-bit counts wrap on a large corpus (a frequent byte passes 4.29e9 within a few tens of GB), which this header's
// verifier detects in its checkpoints.
//
// Checkpoint files: "TRI1" is the live trainer's layout, read as is. "TRI2" is the same layout followed by a trailer
// (cap and the number of halvings), written by native/tools/trie_stream_train_v2.cpp.
// The live trainer and scripts/run_train.sh are not changed by this header.
//
// to_packed() loads a checkpoint into cyphalm::trie::Model (packed_trie.hpp) so today's fixed backoff (Model::mix)
// scores it: the trie, order-1 and unigram tables are byte-compatible; the match becomes Model's compacted table
// (keys seen at least twice) over a side array holding each key's stored next byte.

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
#include <vector>

namespace cypha::cyphalm::stream {

namespace fs = std::filesystem;

constexpr std::uint32_t kNil = 0xffffffffu;
constexpr char kMagic1[4] = {'T', 'R', 'I', '1'};
constexpr char kMagic2[4] = {'T', 'R', 'I', '2'};

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

// ---- Checkpoint files

struct Header {
    bool tri2 = false;
    std::uint32_t nn = 0, nf = 0, slots = 0, mfilled = 0;
    std::uint64_t bytes_seen = 0;
    std::uint64_t file_bytes = 0, expected_bytes = 0;
    std::uint32_t cap = 0xffffffffu;
    std::uint64_t halve_uni = 0, halve_o1 = 0, halve_node = 0;
};

constexpr std::uint64_t kTrailerBytes = 4 + 8 * 3;

inline std::uint64_t expected_size(const Header& h) {
    std::uint64_t s = 4 + 4 + 4 + 8;
    s += sizeof(std::uint32_t) * 256 + 4 + sizeof(std::uint32_t) * 256 * 256 + sizeof(std::uint32_t) * 256 +
         sizeof(std::uint32_t) * 256;
    s += static_cast<std::uint64_t>(h.nn) * sizeof(Node) + static_cast<std::uint64_t>(h.nf) * sizeof(Fol);
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
    else return err = "bad magic", false;
    in.read(reinterpret_cast<char*>(&h.nn), 4);
    in.read(reinterpret_cast<char*>(&h.nf), 4);
    in.read(reinterpret_cast<char*>(&h.bytes_seen), 8);
    const std::uint64_t fixed = 4 + 4 + 4 + 8 + 4 * 256 + 4 + 4 * 256 * 256 + 4 * 256 + 4 * 256;
    const std::uint64_t at_slots = fixed + static_cast<std::uint64_t>(h.nn) * sizeof(Node) +
                                   static_cast<std::uint64_t>(h.nf) * sizeof(Fol);
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

// Loads a checkpoint into packed_trie's training Model for scoring with Model::mix (Mode::Full or Synth).
// `side` receives one stored next byte per kept match key; pm.data points into it. Peak memory is about the
// checkpoint's size: the trie arrays are read in place, the match is compacted while it is read.
inline bool to_packed(const fs::path& path, trie::Model& pm, std::vector<std::uint8_t>& side, Header& h,
                      std::string& err) {
    if (!read_header(path, h, err)) return false;
    if (h.file_bytes != h.expected_bytes) {
        err = "size " + std::to_string(h.file_bytes) + " != expected " + std::to_string(h.expected_bytes);
        return false;
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

// Paths holding "/trie/" (the trainers' output) or "/labelled/" are skipped, and so are unfinished *.part files.
inline bool trainable(const fs::path& p) {
    const std::string s = p.generic_string();
    if (s.find("/labelled/") != std::string::npos || s.find("/trie/") != std::string::npos) return false;
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

}  // namespace cypha::cyphalm::stream
