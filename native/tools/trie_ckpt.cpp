// trie_ckpt: verify, score and compare the 1 TB runs' checkpoints (docs/TRAIN_1TB_RUNBOOK.md).
//
//   trie_ckpt info CKPT
//   trie_ckpt verify CKPT [--corpus ROOT] [--trained LIST] [--deep]
//   trie_ckpt bpc CKPT|MODEL.pk FILE[:FROM:LEN] ...
//   trie_ckpt compare CKPT_A CKPT_B FILE[:FROM:LEN] ...
//   trie_ckpt pack CKPT OUT.pk [--fit FILE[:FROM:LEN]]
//   trie_ckpt selftest
//
// Reads "TRI1" (the live trainer's old 32-bit layout), "TRI2" (trie_stream_train_v2) and "TR64" (the live trainer
// native/tools/trie_stream_train.cpp since 2026-10-28, 64-bit counts) checkpoints through
// cypha/cyphalm/stream_trie.hpp. verify checks the file's size and structure, whether 32-bit counts wrapped (TRI1),
// that every total is its counts' sum (TR64), and with --corpus whether every finished corpus file was trained. bpc
// scores held-out bytes with today's fixed backoff (packed_trie.hpp Model::mix, Mode::Full and Mode::Synth, the 4-byte
// match folded in); given a stored packed model (pack's output, "CYPHAPK") it prints the fixed backoff in Synth and the
// learned mixture with the model's own weights. compare scores two checkpoints on the same bytes, paired over 4 KiB
// blocks. pack writes the checkpoint's packed serving form (packed_trie.hpp PackedModel, the file cyphalm_trie
// --model serves) without the 4-byte match, whose stored follower is an index into its own side array rather than the
// training text; the six mixture weights are fit by exponentiated gradient over --fit's bytes (default: the
// experts' starting weights), with the counts frozen. Scoring only: it answers nothing (PLAN rule 10).

#include "cypha/cyphalm/stream_trie.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace st = cypha::cyphalm::stream;
namespace tr = cypha::cyphalm::trie;

namespace {

constexpr std::uint64_t k2p32 = std::uint64_t{1} << 32;

// ---- verify

struct Verdict {
    std::vector<std::string> problems, notes;
    bool ok() const { return problems.empty(); }
};

template <class M>
void check_structure(const M& m, Verdict& v) {
    const std::size_t nn = m.nodes.size(), nf = m.fols.size();
    if (nn == 0 || m.nodes[0].depth != 0) return v.problems.push_back("no root node"), void();
    std::uint64_t bad_link = 0, bad_depth = 0, bad_nchild = 0, bad_nfol = 0, fol_at_d1 = 0;
    for (int b = 0; b < 256; ++b) {
        const auto c = m.root_child[b];
        if (c == st::kNil) continue;
        if (c >= nn || m.nodes[c].depth != 1 || m.nodes[c].byte != b || m.nodes[c].parent != 0) ++bad_link;
    }
    for (std::size_t i = 1; i < nn; ++i) {
        const auto& n = m.nodes[i];
        if (n.parent >= nn) {
            ++bad_link;
            continue;
        }
        if (n.depth != m.nodes[n.parent].depth + 1 || n.depth > 4) ++bad_depth;
        if ((n.child_head != st::kNil && n.child_head >= nn) || (n.sibling != st::kNil && n.sibling >= nn) ||
            (n.fol_head != st::kNil && n.fol_head >= nf))
            ++bad_link;
        std::uint64_t kids = 0;
        for (auto c = n.child_head; c != st::kNil && c < nn && kids <= 256; c = m.nodes[c].sibling) ++kids;
        if (kids != n.nchild) ++bad_nchild;
        std::uint64_t fl = 0;
        for (auto f = n.fol_head; f != st::kNil && f < nf && fl <= 256; f = m.fols[f].next) ++fl;
        if (fl != n.nfol) ++bad_nfol;
        if (n.depth == 1 && n.fol_head != st::kNil) ++fol_at_d1;
    }
    auto report = [&](std::uint64_t x, const char* what) {
        if (x) v.problems.push_back(std::to_string(x) + " " + what);
    };
    report(bad_link, "nodes with a link out of range");
    report(bad_depth, "nodes with a wrong depth");
    report(bad_nchild, "nodes whose child count does not match their child list");
    report(bad_nfol, "nodes whose follower count does not match their follower list");
    report(fol_at_d1, "order-1 nodes holding followers");
}

void check_counts(const st::Model& m, const st::Header& h, Verdict& v) {
    // Per node: the followers' counts against the node's total.
    std::uint64_t wrapped_total = 0, inconsistent = 0;
    std::uint64_t depth_sum[5] = {0, 0, 0, 0, 0};
    for (std::size_t i = 1; i < m.nodes.size(); ++i) {
        const auto& n = m.nodes[i];
        if (n.fol_head == st::kNil) continue;
        std::uint64_t s = 0;
        for (auto f = n.fol_head; f != st::kNil && f < m.fols.size(); f = m.fols[f].next) s += m.fols[f].count;
        if (n.depth <= 4) depth_sum[n.depth] += n.total;
        if (s == n.total) continue;
        if ((s & 0xffffffffu) == n.total) ++wrapped_total;
        else ++inconsistent;
    }
    if (inconsistent) v.problems.push_back(std::to_string(inconsistent) + " nodes whose total is not their followers' sum");
    if (wrapped_total)
        v.problems.push_back(std::to_string(wrapped_total) + " node totals wrapped past 2^32 (followers' sum is larger)");
    std::uint64_t uni_sum = 0;
    for (auto c : m.uni) uni_sum += c;
    std::uint64_t o1_rows_wrapped = 0;
    for (int p = 0; p < 256; ++p) {
        std::uint64_t s = 0;
        for (auto c : m.o1[p]) s += c;
        if (s != m.o1n[p]) ++o1_rows_wrapped;
    }
    if (h.tri2) {
        if (uni_sum != m.uni_n) v.problems.push_back("unigram total is not its counts' sum");
        if (o1_rows_wrapped) v.problems.push_back(std::to_string(o1_rows_wrapped) + " order-1 rows whose total is off");
        v.notes.push_back("halvings: unigram " + std::to_string(h.halve_uni) + ", order-1 rows " +
                          std::to_string(h.halve_o1) + ", nodes " + std::to_string(h.halve_node) + " (cap " +
                          std::to_string(h.cap) + ")");
        return;
    }
    // TRI1: bytes_seen is 64-bit and never wraps; the unigram counts add to it unless a count wrapped.
    if (uni_sum != m.bytes_seen) {
        const std::uint64_t d = m.bytes_seen - uni_sum;
        v.problems.push_back("unigram counts wrapped: their sum is " + std::to_string(uni_sum) + ", bytes seen " +
                             std::to_string(m.bytes_seen) + " (" + std::to_string(d / k2p32) + " wraps of 2^32)");
    }
    if (m.bytes_seen >= k2p32) v.notes.push_back("uni_n wrapped (bytes seen >= 2^32); the scorer reads the counts");
    if (o1_rows_wrapped) v.problems.push_back(std::to_string(o1_rows_wrapped) + " order-1 row totals wrapped");
    // Each observation at order d also lands in one order-d node (less a few bytes at each file or record start), so
    // the order-d totals fall short of the bytes seen by about one 2^32 per wrap.
    for (int d = 2; d <= 4; ++d) {
        const std::uint64_t short_by = m.bytes_seen > depth_sum[d] ? m.bytes_seen - depth_sum[d] : 0;
        const std::uint64_t wraps = (short_by + k2p32 / 2) / k2p32;
        if (wraps)
            v.problems.push_back("order-" + std::to_string(d) + " totals fall short of the bytes seen by " +
                                 std::to_string(short_by) + " (about " + std::to_string(wraps) + " wraps of 2^32)");
    }
}

// TR64 counts are 64-bit and never wrap, so every total must equal its counts' sum exactly.
void check_counts64(const st::Model64& m, Verdict& v) {
    std::uint64_t inconsistent = 0;
    std::uint64_t depth_sum[5] = {0, 0, 0, 0, 0};
    for (std::size_t i = 1; i < m.nodes.size(); ++i) {
        const auto& n = m.nodes[i];
        if (n.depth <= 4) depth_sum[n.depth] += n.total;
        if (n.fol_head == st::kNil && n.total == 0) continue;
        std::uint64_t s = 0;
        for (auto f = n.fol_head; f != st::kNil && f < m.fols.size(); f = m.fols[f].next) s += m.fols[f].count;
        if (s != n.total) ++inconsistent;
    }
    if (inconsistent) v.problems.push_back(std::to_string(inconsistent) + " nodes whose total is not their followers' sum");
    std::uint64_t uni_sum = 0;
    for (auto c : m.uni) uni_sum += c;
    if (uni_sum != m.uni_n) v.problems.push_back("unigram total is not its counts' sum");
    if (m.uni_n != m.bytes_seen)
        v.problems.push_back("unigram total " + std::to_string(m.uni_n) + " != bytes seen " + std::to_string(m.bytes_seen));
    std::uint64_t bad_rows = 0, o1_sum = 0;
    for (int p = 0; p < 256; ++p) {
        std::uint64_t s = 0;
        for (int y = 0; y < 256; ++y) s += m.o1[static_cast<std::size_t>(p) * 256 + static_cast<std::size_t>(y)];
        bad_rows += s != m.o1n[p];
        o1_sum += m.o1n[p];
    }
    if (bad_rows) v.problems.push_back(std::to_string(bad_rows) + " order-1 rows whose total is off");
    if (o1_sum > m.bytes_seen) v.problems.push_back("order-1 totals exceed the bytes seen");
    // Each record or file start leaves out its first d bytes at order d, so order d falls a little short.
    for (int d = 2; d <= 4; ++d) {
        if (depth_sum[d] > m.bytes_seen) v.problems.push_back("order-" + std::to_string(d) + " totals exceed the bytes seen");
        v.notes.push_back("order-" + std::to_string(d) + " totals " + std::to_string(depth_sum[d]) + " (" +
                          std::to_string(m.bytes_seen - std::min(m.bytes_seen, depth_sum[d])) +
                          " short of the bytes seen: record and file starts)");
    }
}

fs::path rel_to(const fs::path& p, const fs::path& root) {
    const std::string s = p.generic_string(), r = root.generic_string();
    if (s.size() > r.size() && s.compare(0, r.size(), r) == 0) return fs::path(s.substr(r.size() + (s[r.size()] == '/')));
    return p;
}

template <class M>
void check_corpus(const M& m, const fs::path& root, const fs::path& trained_list, bool deep, Verdict& v) {
    std::set<std::string> trained;
    {
        std::ifstream in(trained_list);
        if (!in) return v.problems.push_back("cannot read " + trained_list.string()), void();
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) trained.insert(rel_to(fs::path(line), root).generic_string());
        }
    }
    std::set<std::string> present;
    std::uint64_t parts = 0, txt_bytes = 0, deep_bytes = 0, jsonl = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(root, ec); it != fs::recursive_directory_iterator();
         it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file()) continue;
        const fs::path p = it->path();
        const std::string name = p.filename().string();
        if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".part") == 0) ++parts;
        if (!st::trainable(p)) continue;
        present.insert(rel_to(p, root).generic_string());
        if (p.extension() == ".txt") txt_bytes += it->file_size(ec);
        else ++jsonl;
        if (deep) {
            std::uint64_t n = 0;
            if (st::train_file(nullptr, p, n)) deep_bytes += n;
        }
    }
    std::uint64_t missing = 0, gone = 0;
    for (const auto& f : present)
        if (!trained.count(f)) ++missing;
    for (const auto& f : trained)
        if (!present.count(f)) ++gone;
    v.notes.push_back("corpus: " + std::to_string(present.size()) + " finished files (" + std::to_string(jsonl) +
                      " .jsonl), " + std::to_string(trained.size()) + " listed as trained");
    if (missing) v.problems.push_back(std::to_string(missing) + " finished corpus files not trained yet");
    if (parts) v.problems.push_back(std::to_string(parts) + " unfinished *.part files in the corpus (downloads)");
    if (gone) v.notes.push_back(std::to_string(gone) + " trained files no longer in the corpus");
    if (deep) {
        if (deep_bytes != m.bytes_seen && !missing)
            v.problems.push_back("bytes seen " + std::to_string(m.bytes_seen) + " != the corpus's text bytes " +
                                 std::to_string(deep_bytes));
        else
            v.notes.push_back("corpus text bytes " + std::to_string(deep_bytes) + " (bytes seen " +
                              std::to_string(m.bytes_seen) + ")");
    } else if (jsonl == 0 && !missing && txt_bytes != m.bytes_seen) {
        v.problems.push_back("bytes seen " + std::to_string(m.bytes_seen) + " != the .txt files' size " +
                             std::to_string(txt_bytes));
    }
}

int cmd_verify(const std::vector<std::string>& args) {
    if (args.empty()) return std::cerr << "usage: trie_ckpt verify CKPT [--corpus ROOT] [--trained LIST] [--deep]\n", 2;
    const fs::path ckpt = args[0];
    fs::path corpus, trained = ckpt.parent_path() / "trained.txt";
    bool deep = false;
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--corpus" && i + 1 < args.size()) corpus = args[++i];
        else if (args[i] == "--trained" && i + 1 < args.size()) trained = args[++i];
        else if (args[i] == "--deep") deep = true;
        else return std::cerr << "unknown option " << args[i] << "\n", 2;
    }
    st::Header h;
    std::string err;
    Verdict v;
    if (!st::read_header(ckpt, h, err)) {
        std::cout << "verify " << ckpt.string() << ": NOT VERIFIED\n  - " << err << "\n";
        return 1;
    }
    auto run = [&](const auto& m) {
        std::cout << "verify " << ckpt.string() << " (" << st::format_name(h) << ")\n  bytes seen " << m.bytes_seen
                  << ", nodes " << m.nodes.size() << ", followers " << m.fols.size() << ", match keys " << m.mfilled
                  << ", file " << h.file_bytes << " bytes\n";
        check_structure(m, v);
        if (!corpus.empty()) check_corpus(m, corpus, trained, deep, v);
        else v.notes.push_back("corpus not checked (give --corpus ROOT to check every finished file was trained)");
    };
    if (h.tr64) {
        st::Model64 m;
        if (!st::load64(ckpt, m, h, err)) {
            std::cout << "verify " << ckpt.string() << ": NOT VERIFIED\n  - " << err << "\n";
            return 1;
        }
        check_counts64(m, v);
        run(m);
    } else {
        st::Model m;
        if (!st::load(ckpt, m, h, err)) {
            std::cout << "verify " << ckpt.string() << ": NOT VERIFIED\n  - " << err << "\n";
            return 1;
        }
        check_counts(m, h, v);
        run(m);
    }
    for (const auto& n : v.notes) std::cout << "  note: " << n << "\n";
    for (const auto& p : v.problems) std::cout << "  - " << p << "\n";
    std::cout << (v.ok() ? "VERIFIED" : "NOT VERIFIED") << "\n";
    return v.ok() ? 0 : 1;
}

int cmd_info(const std::vector<std::string>& args) {
    if (args.empty()) return std::cerr << "usage: trie_ckpt info CKPT\n", 2;
    st::Header h;
    std::string err;
    if (!st::read_header(args[0], h, err)) return std::cerr << err << "\n", 1;
    std::cout << "format " << st::format_name(h) << "\nbytes_seen " << h.bytes_seen << "\nnodes " << h.nn
              << "\nfollowers " << h.nf << "\nmatch_slots " << h.slots << "\nmatch_keys " << h.mfilled
              << "\nfile_bytes " << h.file_bytes << "\nexpected_bytes " << h.expected_bytes << "\n";
    if (h.tri2)
        std::cout << "cap " << h.cap << "\nhalvings " << h.halve_uni << " " << h.halve_o1 << " " << h.halve_node << "\n";
    return h.file_bytes == h.expected_bytes ? 0 : 1;
}

// ---- bpc and compare

struct Slice {
    std::string label;
    std::vector<std::uint8_t> bytes;
};

bool read_slice(const std::string& spec, Slice& s) {
    std::string path = spec;
    std::uint64_t from = 0, len = 0;
    const auto c2 = spec.rfind(':');
    const auto c1 = c2 == std::string::npos || c2 == 0 ? std::string::npos : spec.rfind(':', c2 - 1);
    if (c1 != std::string::npos && c1 > 1) {  // FILE:FROM:LEN (a drive letter's colon is position 1)
        path = spec.substr(0, c1);
        from = std::strtoull(spec.c_str() + c1 + 1, nullptr, 10);
        len = std::strtoull(spec.c_str() + c2 + 1, nullptr, 10);
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::uint64_t size = static_cast<std::uint64_t>(in.tellg());
    if (from > size) return false;
    if (len == 0 || from + len > size) len = size - from;
    s.bytes.resize(len);
    in.seekg(static_cast<std::streamoff>(from));
    in.read(reinterpret_cast<char*>(s.bytes.data()), static_cast<std::streamsize>(len));
    s.label = spec;
    return static_cast<bool>(in);
}

struct Score {
    double full = 0, synth = 0;
    std::uint64_t matched = 0;
    std::vector<double> blocks_full, blocks_synth;  // bits per byte over each 4 KiB block
};

Score score(const tr::Model& m, const std::vector<std::uint8_t>& x) {
    Score s;
    const std::size_t n = x.size(), B = 4096;
    double bf = 0, bs = 0, P[256];
    double blk_f = 0, blk_s = 0;
    std::size_t in_blk = 0;
    for (std::size_t i = 0; i < n; ++i) {
        bool matched = false;
        m.mix(x.data(), i, SIZE_MAX, tr::Mode::Full, P, matched);
        const double lf = -std::log2(std::max(P[x[i]], 1e-300));
        if (matched) ++s.matched;
        m.mix(x.data(), i, SIZE_MAX, tr::Mode::Synth, P, matched);
        const double ls = -std::log2(std::max(P[x[i]], 1e-300));
        bf += lf;
        bs += ls;
        blk_f += lf;
        blk_s += ls;
        if (++in_blk == B || i + 1 == n) {
            s.blocks_full.push_back(blk_f / static_cast<double>(in_blk));
            s.blocks_synth.push_back(blk_s / static_cast<double>(in_blk));
            blk_f = blk_s = 0;
            in_blk = 0;
        }
    }
    s.full = n ? bf / static_cast<double>(n) : 0;
    s.synth = n ? bs / static_cast<double>(n) : 0;
    return s;
}

bool load_scorer(const fs::path& ckpt, tr::Model& pm, std::vector<std::uint8_t>& side) {
    st::Header h;
    std::string err;
    if (!st::to_packed(ckpt, pm, side, h, err)) {
        std::cerr << "cannot load " << ckpt << ": " << err << "\n";
        return false;
    }
    std::cerr << "loaded " << ckpt.string() << " (" << st::format_name(h) << ", bytes seen " << h.bytes_seen
              << ", nodes " << h.nn << ", kept match keys " << pm.dense.size() << ")\n";
    if (h.tr64)
        std::cerr << "scaled: unigram shift " << h.uni_shift << ", order-1 rows shifted " << h.scaled_o1_rows
                  << ", nodes shifted " << h.scaled_nodes << " (largest shift " << h.max_shift << "), followers dropped "
                  << h.dropped_fols << ", followers kept " << pm.fols.size() << "\n";
    return true;
}

bool is_pack(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    char magic[8] = {};
    in.read(magic, 8);
    return in && std::memcmp(magic, "CYPHAPK", 8) == 0;
}

// A stored packed model: the fixed backoff (Synth) and the learned mixture with the model's own weights.
int bpc_pack(const std::vector<std::string>& args) {
    double w[6];
    tr::Config cfg;
    tr::PackedModel pk;
    try {
        pk = tr::PackedModel::load(args[0], w, cfg);
    } catch (const std::exception& e) {
        return std::cerr << e.what() << "\n", 1;
    }
    std::cerr << "loaded " << args[0] << " (packed, " << pk.bytes() << " bytes, nodes " << pk.nodes() << ", followers "
              << pk.followers() << ", match keys " << pk.matches() << ")\n";
    std::printf("weights %.6f %.6f %.6f %.6f %.6f %.6f\n", w[0], w[1], w[2], w[3], w[4], w[5]);
    for (std::size_t i = 1; i < args.size(); ++i) {
        Slice s;
        if (!read_slice(args[i], s)) return std::cerr << "cannot read " << args[i] << "\n", 1;
        double bs = 0, bm = 0, P[256], e[5][256];
        for (std::size_t j = 0; j < s.bytes.size(); ++j) {
            bool matched = false;
            unsigned pred = 0;
            pk.mix(s.bytes.data(), j, SIZE_MAX, tr::Mode::Synth, P, matched);
            bs += -std::log2(std::max(P[s.bytes[j]], 1e-300));
            pk.fill_experts(s.bytes.data(), j, SIZE_MAX, tr::Mode::Synth, e, matched, pred, false);
            tr::mix_dist(w, e, matched, pred, P);
            bm += -std::log2(std::max(P[s.bytes[j]], 1e-300));
        }
        const double n = s.bytes.empty() ? 1.0 : static_cast<double>(s.bytes.size());
        std::printf("bpc %s bytes %zu synth %.6f mix %.6f\n", s.label.c_str(), s.bytes.size(), bs / n, bm / n);
    }
    return 0;
}

int cmd_bpc(const std::vector<std::string>& args) {
    if (args.size() < 2) return std::cerr << "usage: trie_ckpt bpc CKPT|MODEL.pk FILE[:FROM:LEN] ...\n", 2;
    if (is_pack(args[0])) return bpc_pack(args);
    tr::Model pm;
    std::vector<std::uint8_t> side;
    if (!load_scorer(args[0], pm, side)) return 1;
    for (std::size_t i = 1; i < args.size(); ++i) {
        Slice s;
        if (!read_slice(args[i], s)) return std::cerr << "cannot read " << args[i] << "\n", 1;
        const Score r = score(pm, s.bytes);
        std::printf("bpc %s bytes %zu full %.6f synth %.6f matched %.4f\n", s.label.c_str(), s.bytes.size(), r.full,
                    r.synth, s.bytes.empty() ? 0.0 : static_cast<double>(r.matched) / static_cast<double>(s.bytes.size()));
    }
    return 0;
}

// Mean of b - a over paired blocks, with a 95% bootstrap interval (2,000 resamples, fixed seed).
void paired(const std::vector<double>& a, const std::vector<double>& b, double& mean, double& lo, double& hi) {
    const std::size_t n = std::min(a.size(), b.size());
    std::vector<double> d(n);
    double s = 0;
    for (std::size_t i = 0; i < n; ++i) s += (d[i] = b[i] - a[i]);
    mean = n ? s / static_cast<double>(n) : 0;
    std::mt19937_64 rng(20261029);
    std::uniform_int_distribution<std::size_t> pick(0, n ? n - 1 : 0);
    std::vector<double> boot(2000);
    for (auto& x : boot) {
        double t = 0;
        for (std::size_t i = 0; i < n; ++i) t += d[pick(rng)];
        x = n ? t / static_cast<double>(n) : 0;
    }
    std::sort(boot.begin(), boot.end());
    lo = boot[49];
    hi = boot[1949];
}

int cmd_compare(const std::vector<std::string>& args) {
    if (args.size() < 3) return std::cerr << "usage: trie_ckpt compare CKPT_A CKPT_B FILE[:FROM:LEN] ...\n", 2;
    std::vector<Slice> slices;
    for (std::size_t i = 2; i < args.size(); ++i) {
        Slice s;
        if (!read_slice(args[i], s)) return std::cerr << "cannot read " << args[i] << "\n", 1;
        slices.push_back(std::move(s));
    }
    std::vector<Score> sa, sb;
    for (int k = 0; k < 2; ++k) {
        tr::Model pm;
        std::vector<std::uint8_t> side;
        if (!load_scorer(args[static_cast<std::size_t>(k)], pm, side)) return 1;
        for (const auto& s : slices) (k == 0 ? sa : sb).push_back(score(pm, s.bytes));
    }  // each model is freed before the next loads
    std::printf("compare A=%s B=%s\n", args[0].c_str(), args[1].c_str());
    std::printf("%-40s %10s %10s %10s %24s %10s %10s\n", "holdout", "bytes", "A_full", "B_full", "B-A full [95%]", "A_synth",
                "B_synth");
    for (std::size_t i = 0; i < slices.size(); ++i) {
        double m = 0, lo = 0, hi = 0;
        paired(sa[i].blocks_full, sb[i].blocks_full, m, lo, hi);
        std::printf("%-40s %10zu %10.6f %10.6f %+8.4f [%+.4f, %+.4f] %10.6f %10.6f\n", slices[i].label.c_str(),
                    slices[i].bytes.size(), sa[i].full, sb[i].full, m, lo, hi, sa[i].synth, sb[i].synth);
    }
    return 0;
}

// ---- pack

// Fits the six weights by exponentiated gradient over `x` (counts frozen), as cyphalm_trie does on its training stream.
void fit_weights(const tr::PackedModel& pk, const std::vector<std::uint8_t>& x, const tr::Config& cfg, double w[6]) {
    tr::init_weights(w, cfg);
    double e[5][256], py[6];
    for (std::size_t i = 0; i < x.size(); ++i) {
        bool matched = false;
        unsigned pred = 0;
        pk.fill_experts(x.data(), i, SIZE_MAX, tr::Mode::Synth, e, matched, pred, false);
        for (int k = 0; k < 5; ++k) py[k] = e[k][x[i]];
        py[5] = matched && pred == x[i] ? 1.0 : 0.0;
        tr::eg_update(w, py, matched, cfg);
    }
}

int cmd_pack(const std::vector<std::string>& args) {
    if (args.size() < 2) return std::cerr << "usage: trie_ckpt pack CKPT OUT.pk [--fit FILE[:FROM:LEN]]\n", 2;
    std::string fit;
    for (std::size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--fit" && i + 1 < args.size()) fit = args[++i];
        else return std::cerr << "unknown option " << args[i] << "\n", 2;
    }
    Slice fs_;
    if (!fit.empty() && !read_slice(fit, fs_)) return std::cerr << "cannot read " << fit << "\n", 1;
    tr::PackedModel pk;
    {
        tr::Model pm;
        std::vector<std::uint8_t> side;
        if (!load_scorer(args[0], pm, side)) return 1;
        pm.dense.clear();  // no match: its follower indexes side, not a training text
        pm.data = nullptr;
        pk = tr::pack(pm);
    }
    const tr::Config cfg;
    double w[6];
    if (fit.empty()) tr::init_weights(w, cfg);
    else fit_weights(pk, fs_.bytes, cfg, w);
    try {
        pk.save(args[1], w, cfg);
    } catch (const std::exception& e) {
        return std::cerr << e.what() << "\n", 1;
    }
    std::error_code ec;
    std::printf("packed %s nodes %llu followers %llu match_keys %llu bytes %llu file %llu\n", args[1].c_str(),
                static_cast<unsigned long long>(pk.nodes()), static_cast<unsigned long long>(pk.followers()),
                static_cast<unsigned long long>(pk.matches()), static_cast<unsigned long long>(pk.bytes()),
                static_cast<unsigned long long>(fs::file_size(args[1], ec)));
    std::printf("weights %.6f %.6f %.6f %.6f %.6f %.6f%s%s\n", w[0], w[1], w[2], w[3], w[4], w[5],
                fit.empty() ? " (initial)" : " fit on ", fit.c_str());
    return 0;
}

// ---- selftest

int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                                \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(c)) {                                                             \
            ++g_failed;                                                         \
            std::fprintf(stderr, "check failed line %d: %s\n", __LINE__, #c);   \
        }                                                                       \
    } while (0)

std::vector<std::uint8_t> make_text(std::uint64_t seed, std::size_t n) {
    static const char* words[] = {"the ", "cat ", "sat ", "on ", "a ", "mat ", "and ", "dog ", "ran ", "far ",
                                  "with ", "red ", "hat. ", "It ", "was ", "late ", "when ", "they ", "met ", "\n"};
    std::vector<std::uint8_t> t;
    std::uint64_t x = seed;
    while (t.size() < n) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        const char* w = words[(x >> 33) % 20];
        for (const char* c = w; *c && t.size() < n; ++c) t.push_back(static_cast<std::uint8_t>(*c));
    }
    return t;
}

int cmd_selftest() {
    const fs::path tmp = fs::temp_directory_path() / ("trie_ckpt_selftest_" + std::to_string(std::random_device{}()));
    fs::create_directories(tmp);
    const auto T = make_text(1, 200000), H = make_text(2, 20000);

    // 1. The stream model and packed_trie's Model, trained on the same bytes, score the holdout identically.
    st::Model sm;
    sm.init(10);
    sm.reset_hist();
    for (auto b : T) sm.observe_byte(b);
    const fs::path c2 = tmp / "a.bin";
    CHECK(st::save(c2, sm, true));
    tr::Config cfg;
    cfg.match_bits = 10;
    tr::Model ref;
    ref.verbose = false;
    ref.init(cfg);
    ref.data = T.data();
    for (std::size_t i = 0; i < T.size(); ++i) ref.observe(i);
    ref.compact_match();
    tr::Model pm;
    std::vector<std::uint8_t> side;
    st::Header h;
    std::string err;
    CHECK(st::to_packed(c2, pm, side, h, err));
    CHECK(h.tri2 && h.bytes_seen == T.size());
    CHECK(pm.nodes.size() == ref.nodes.size() && pm.fols.size() == ref.fols.size() && pm.dense.size() == ref.dense.size());
    double maxd = 0;
    for (auto mode : {tr::Mode::Full, tr::Mode::Synth})
        for (std::size_t i = 0; i < H.size(); i += 7) {
            double P[256], Q[256];
            bool m1 = false, m2 = false;
            pm.mix(H.data(), i, SIZE_MAX, mode, P, m1);
            ref.mix(H.data(), i, SIZE_MAX, mode, Q, m2);
            CHECK(m1 == m2);
            for (int s = 0; s < 256; ++s) maxd = std::max(maxd, std::fabs(P[s] - Q[s]));
        }
    CHECK(maxd < 1e-12);
    std::fprintf(stderr, "packed equivalence: max |dP| %.3g\n", maxd);

    // 2. Round trip: TRI2 and TRI1 files load back to the same model.
    for (bool tri2 : {true, false}) {
        const fs::path c = tmp / (tri2 ? "rt2.bin" : "rt1.bin");
        CHECK(st::save(c, sm, tri2));
        st::Model back;
        st::Header hb;
        CHECK(st::load(c, back, hb, err) && hb.tri2 == tri2);
        CHECK(back.bytes_seen == sm.bytes_seen && back.nodes.size() == sm.nodes.size() &&
              back.fols.size() == sm.fols.size() && back.mfilled == sm.mfilled &&
              std::equal(std::begin(back.uni), std::end(back.uni), std::begin(sm.uni)));
        Verdict v;
        check_structure(back, v);
        check_counts(back, hb, v);
        CHECK(v.ok());
    }

    // 2b. TR64: the same model widened to 64 bits scores identically and verifies; a damaged total is caught. With
    // every count times 2^33, to_packed shifts each node back, so order 2-4 keep their exact ratios (Full mode differs
    // only through the KT tables' +1/2), and the packed form fit on held-out bytes round-trips through its file.
    {
        const fs::path c64 = tmp / "w64.bin";
        CHECK(st::save64(c64, st::widen(sm)));
        tr::Model p64;
        std::vector<std::uint8_t> s64;
        st::Header h64;
        CHECK(st::to_packed(c64, p64, s64, h64, err) && h64.tr64 && h64.scaled_nodes == 0 && h64.dropped_fols == 0);
        CHECK(p64.nodes.size() == pm.nodes.size() && p64.fols.size() == pm.fols.size() &&
              p64.dense.size() == pm.dense.size());
        double d64 = 0;
        for (auto mode : {tr::Mode::Full, tr::Mode::Synth})
            for (std::size_t i = 0; i < H.size(); i += 7) {
                double P[256], Q[256];
                bool m1 = false, m2 = false;
                p64.mix(H.data(), i, SIZE_MAX, mode, P, m1);
                pm.mix(H.data(), i, SIZE_MAX, mode, Q, m2);
                CHECK(m1 == m2);
                for (int s = 0; s < 256; ++s) d64 = std::max(d64, std::fabs(P[s] - Q[s]));
            }
        CHECK(d64 < 1e-12);
        st::Model64 back;
        st::Header hb;
        CHECK(st::load64(c64, back, hb, err));
        Verdict v;
        check_structure(back, v);
        check_counts64(back, v);
        CHECK(v.ok());
        for (auto& n : back.nodes)
            if (n.depth == 4 && n.total > 2) {
                n.total += 1;
                break;
            }
        Verdict vd;
        check_counts64(back, vd);
        CHECK(!vd.ok());

        const fs::path cbig = tmp / "w64big.bin";
        CHECK(st::save64(cbig, st::widen(sm, std::uint64_t{1} << 33)));
        tr::Model pb;
        std::vector<std::uint8_t> sb;
        st::Header hbig;
        CHECK(st::to_packed(cbig, pb, sb, hbig, err) && hbig.scaled_nodes > 0 && hbig.dropped_fols == 0 &&
              hbig.uni_shift > 0);
        bool ratios = pb.nodes.size() == pm.nodes.size();
        for (std::size_t i = 1; ratios && i < pm.nodes.size(); ++i) {
            const auto ta = pm.nodes[i].total, tb = pb.nodes[i].total;
            if (ta == 0) continue;
            for (auto f = pm.nodes[i].fol_head, g = pb.nodes[i].fol_head; f != st::kNil;
                 f = pm.fols[f].next, g = pb.fols[g].next)
                ratios = ratios && g != st::kNil && pb.fols[g].sym == pm.fols[f].sym &&
                         std::uint64_t{pb.fols[g].count} * ta == std::uint64_t{pm.fols[f].count} * tb;
        }
        CHECK(ratios);
        // The reference with pb's unigram and order-1 tables: the shifted nodes alone must not move Full mode.
        tr::Model ref2;
        std::vector<std::uint8_t> side2;
        st::Header h2;
        CHECK(st::to_packed(c2, ref2, side2, h2, err));
        std::memcpy(ref2.uni, pb.uni, sizeof ref2.uni);
        ref2.uni_n = pb.uni_n;
        std::memcpy(ref2.o1, pb.o1, sizeof ref2.o1);
        std::memcpy(ref2.o1n, pb.o1n, sizeof ref2.o1n);
        double dbig = 0;
        for (std::size_t i = 0; i < H.size(); i += 7) {
            double P[256], Q[256];
            bool m1 = false, m2 = false;
            pb.mix(H.data(), i, SIZE_MAX, tr::Mode::Full, P, m1);
            ref2.mix(H.data(), i, SIZE_MAX, tr::Mode::Full, Q, m2);
            CHECK(m1 == m2);
            for (int s = 0; s < 256; ++s) dbig = std::max(dbig, std::fabs(P[s] - Q[s]));
        }
        CHECK(dbig < 1e-12);
        std::fprintf(stderr, "TR64: max |dP| %.3g unscaled, %.3g at 2^33 (Full)\n", d64, dbig);

        pb.dense.clear();
        pb.data = nullptr;
        const tr::PackedModel pk = tr::pack(pb);
        const tr::Config pc;
        double wf[6];
        fit_weights(pk, H, pc, wf);
        double ws = 0;
        for (double x : wf) ws += x;
        CHECK(std::fabs(ws - 1.0) < 1e-9 && wf[4] > wf[0]);
        const fs::path pkf = tmp / "m.pk";
        pk.save(pkf.string(), wf, pc);
        CHECK(is_pack(pkf) && !is_pack(c64));
        double wb[6];
        tr::Config cb;
        const tr::PackedModel back_pk = tr::PackedModel::load(pkf.string(), wb, cb);
        CHECK(back_pk == pk && std::equal(wf, wf + 6, wb) && pk.matches() == 0);
    }

    // 3. Halving: with a small cap no count passes it, totals stay their followers' sums, and verify passes.
    st::Model cm;
    cm.init(10);
    cm.cap = 64;
    cm.reset_hist();
    for (auto b : T) cm.observe_byte(b);
    std::uint64_t over = 0;
    for (std::size_t i = 1; i < cm.nodes.size(); ++i) over += cm.nodes[i].total > 64;
    for (auto c : cm.uni) over += c > 64;
    CHECK(over == 0 && cm.uni_n <= 64 && cm.halve_uni > 0 && cm.halve_node > 0);
    {
        Verdict v;
        st::Header hc;
        hc.tri2 = true;
        check_structure(cm, v);
        check_counts(cm, hc, v);
        CHECK(v.ok());
    }

    // 4. Wraps and damage are caught in TRI1 files.
    {
        st::Model w = sm;
        w.bytes_seen += k2p32;  // a unigram count wrapped once: the counts fall 2^32 short
        st::Header hw;
        Verdict v;
        check_counts(w, hw, v);
        CHECK(!v.ok());
        st::Model d = sm;
        for (auto& n : d.nodes)
            if (n.depth == 3 && n.total > 2) {
                n.total -= 1;  // a total that is not its followers' sum
                break;
            }
        Verdict v2;
        check_counts(d, hw, v2);
        CHECK(!v2.ok());
        st::Model s2 = sm;
        s2.nodes[5].nchild += 1;  // a child count that does not match the list
        Verdict v3;
        check_structure(s2, v3);
        CHECK(!v3.ok());
    }

    // 5. Corpus completeness: a finished file not yet trained and a *.part file are both reported.
    {
        const fs::path root = tmp / "corpus";
        fs::create_directories(root / "wiki");
        std::ofstream(root / "wiki" / "a.txt", std::ios::binary).write(reinterpret_cast<const char*>(T.data()), 5000);
        std::ofstream(root / "wiki" / "b.txt", std::ios::binary).write(reinterpret_cast<const char*>(H.data()), 3000);
        st::Model cm2;
        cm2.init(10);
        std::uint64_t n = 0;
        CHECK(st::train_file(&cm2, root / "wiki" / "a.txt", n) && n == 5000);
        const fs::path list = tmp / "trained.txt";
        std::ofstream(list) << (root / "wiki" / "a.txt").generic_string() << "\n";
        Verdict v;
        check_corpus(cm2, root, list, true, v);
        CHECK(!v.ok());  // b.txt is not trained
        CHECK(st::train_file(&cm2, root / "wiki" / "b.txt", n));
        std::ofstream(list, std::ios::app) << (root / "wiki" / "b.txt").generic_string() << "\n";
        Verdict v2;
        check_corpus(cm2, root, list, true, v2);
        CHECK(v2.ok());
        std::ofstream(root / "wiki" / "c.txt.part") << "partial";
        Verdict v3;
        check_corpus(cm2, root, list, false, v3);
        CHECK(!v3.ok());
    }

    // 5b. Parallel training (trie_stream_train_v2 --threads): on a .jsonl file 4 threads give the one-thread model's
    // counts, structure, match entries and distributions; with a small cap every total stays its followers' sum.
    {
        const fs::path jl = tmp / "p.jsonl";
        {
            std::ofstream o(jl, std::ios::binary);
            for (std::size_t r = 0; r < 400; ++r) {
                const auto t = make_text(100 + r, 300 + (r * 37) % 700);
                o << "{\"text\": \"" << std::string(t.begin(), t.end()) << "\", \"id\": " << r << "}\n";
            }
        }
        st::Model a, b;
        a.init(10);
        b.init(10);
        std::uint64_t na = 0, nb = 0;
        CHECK(st::train_file(&a, jl, na));
        CHECK(st::train_file_parallel(b, jl, 4, nb, 8));
        CHECK(na == nb && a.bytes_seen == b.bytes_seen && a.nodes.size() == b.nodes.size() &&
              a.fols.size() == b.fols.size() && a.mfilled == b.mfilled && a.uni_n == b.uni_n &&
              std::equal(&a.o1[0][0], &a.o1[0][0] + 65536, &b.o1[0][0]));
        const auto cuts = st::cut_file(jl, 4);
        CHECK(cuts.size() == 5 && cuts[0] == 0 && cuts[4] == fs::file_size(jl) && cuts[1] > 0 && cuts[1] < cuts[2]);
        const fs::path fa = tmp / "pa.bin", fb = tmp / "pb.bin";
        CHECK(st::save(fa, a, true) && st::save(fb, b, true));
        tr::Model ma, mb2;
        std::vector<std::uint8_t> sa, sb2;
        st::Header ha, hb2;
        CHECK(st::to_packed(fa, ma, sa, ha, err) && st::to_packed(fb, mb2, sb2, hb2, err));
        CHECK(ma.dense.size() == mb2.dense.size());
        double dp = 0;
        for (auto mode : {tr::Mode::Full, tr::Mode::Synth})
            for (std::size_t i = 0; i < H.size(); i += 5) {
                double P[256], Q[256];
                bool m1 = false, m2 = false;
                ma.mix(H.data(), i, SIZE_MAX, mode, P, m1);
                mb2.mix(H.data(), i, SIZE_MAX, mode, Q, m2);
                CHECK(m1 == m2);
                for (int s = 0; s < 256; ++s) dp = std::max(dp, std::fabs(P[s] - Q[s]));
            }
        CHECK(dp < 1e-12);
        std::fprintf(stderr, "parallel training: max |dP| %.3g against one thread\n", dp);

        st::Model c;
        c.init(10);
        c.cap = 64;
        const fs::path tx = tmp / "p.txt";
        std::ofstream(tx, std::ios::binary).write(reinterpret_cast<const char*>(T.data()), 60000);
        std::uint64_t nc = 0;
        CHECK(st::train_file_parallel(c, tx, 4, nc, 8) && nc == 60000 && c.bytes_seen == 60000);
        Verdict v;
        st::Header hc;
        hc.tri2 = true;
        check_structure(c, v);
        check_counts(c, hc, v);
        CHECK(v.ok() && c.uni_n <= 64 && c.halve_uni > 0 && c.halve_node > 0);
    }

    // 6. compare's pairing: identical blocks give a zero difference.
    {
        double m = 1, lo = 1, hi = 1;
        paired({1, 2, 3}, {1, 2, 3}, m, lo, hi);
        CHECK(m == 0 && lo == 0 && hi == 0);
    }

    std::error_code ec;
    fs::remove_all(tmp, ec);
    std::printf("trie_ckpt selftest: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: trie_ckpt info|verify|bpc|compare|pack|selftest ...\n";
        return 2;
    }
    const std::string cmd = argv[1];
    const std::vector<std::string> args(argv + 2, argv + argc);
    if (cmd == "info") return cmd_info(args);
    if (cmd == "verify") return cmd_verify(args);
    if (cmd == "bpc") return cmd_bpc(args);
    if (cmd == "compare") return cmd_compare(args);
    if (cmd == "pack") return cmd_pack(args);
    if (cmd == "selftest") return cmd_selftest();
    std::cerr << "unknown command " << cmd << "\n";
    return 2;
}
