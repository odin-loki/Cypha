// K5 on more Wikipedia (docs/reports/K5_SCALE.md): the overlap check on 32-byte spans.
//
//   k5_overlap clean DOCS OUT REPORT FORBID... [--info INFO...]
//       DOCS: one document per line. FORBID, INFO: one text per line.
//       Every 32-byte span of every FORBID text is hashed (64-bit FNV-1a). For each document:
//         - the hits are its 32-byte windows whose hash is a forbidden span's;
//         - if hits cover more than 10% of its bytes, the whole document is dropped (a near-copy);
//         - otherwise every sentence that overlaps a hit is cut (sentences end at . ! ? followed by a space),
//           and the document is checked again until it holds no hit.
//       OUT gets the kept documents, one per line. REPORT gets counts per FORBID file and per INFO file (INFO files
//       are only counted: documents of OUT that hold one of their spans).
//   k5_overlap check DOCS FORBID...
//       Prints the number of documents (lines) of DOCS that hold a forbidden span; exit 1 if any.
//
// Build: g++ -std=c++20 -O2 -o k5_overlap scripts/k5_overlap.cpp
// Author: Odin Loch
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {
constexpr std::size_t K = 32;

std::uint64_t fnv(const char* p) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < K; ++i) h = (h ^ static_cast<unsigned char>(p[i])) * 1099511628211ull;
    return h;
}

std::vector<std::string> lines(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot read " << path << "\n";
        std::exit(2);
    }
    std::vector<std::string> v;
    for (std::string s; std::getline(in, s);) v.push_back(std::move(s));
    return v;
}

std::vector<std::uint64_t> span_set(const std::vector<std::string>& files) {
    std::vector<std::uint64_t> h;
    for (const auto& f : files)
        for (const auto& t : lines(f))
            for (std::size_t i = 0; i + K <= t.size(); ++i) h.push_back(fnv(t.data() + i));
    std::sort(h.begin(), h.end());
    h.erase(std::unique(h.begin(), h.end()), h.end());
    return h;
}

// Start positions of the windows of s whose hash is in set.
std::vector<std::size_t> hit_pos(std::string_view s, const std::vector<std::uint64_t>& set) {
    std::vector<std::size_t> p;
    for (std::size_t i = 0; i + K <= s.size(); ++i)
        if (std::binary_search(set.begin(), set.end(), fnv(s.data() + i))) p.push_back(i);
    return p;
}

std::size_t covered(const std::vector<std::size_t>& p) {
    std::size_t c = 0, end = 0;
    for (const std::size_t a : p) {
        const std::size_t b = a + K;
        c += b - std::max(a, end);
        end = b;
    }
    return c;
}

// Sentence ranges [a, b): b is one past the space after . ! ? (or the end).
std::vector<std::pair<std::size_t, std::size_t>> sentences(std::string_view s) {
    std::vector<std::pair<std::size_t, std::size_t>> r;
    std::size_t a = 0;
    for (std::size_t i = 0; i + 1 < s.size(); ++i)
        if ((s[i] == '.' || s[i] == '!' || s[i] == '?') && s[i + 1] == ' ') {
            r.emplace_back(a, i + 2);
            a = i + 2;
        }
    if (a < s.size()) r.emplace_back(a, s.size());
    return r;
}

std::string cut(std::string_view s, const std::vector<std::size_t>& p, std::size_t& cut_sent) {
    std::string out;
    std::size_t j = 0;
    for (const auto& [a, b] : sentences(s)) {
        while (j < p.size() && p[j] + K <= a) ++j;
        const bool hit = j < p.size() && p[j] < b;
        if (hit) {
            ++cut_sent;
            continue;
        }
        out.append(s.substr(a, b - a));
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: k5_overlap clean DOCS OUT REPORT FORBID... [--info INFO...] | k5_overlap check DOCS FORBID...\n";
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "check") {
        const auto docs = lines(argv[2]);
        const auto set = span_set(std::vector<std::string>(argv + 3, argv + argc));
        std::size_t bad = 0;
        for (const auto& d : docs) bad += !hit_pos(d, set).empty();
        std::printf("check %s: %zu documents, %zu hold a forbidden span (%zu spans in the set)\n", argv[2], docs.size(), bad, set.size());
        return bad ? 1 : 0;
    }
    if (mode != "clean" || argc < 6) return 2;
    const std::string docs_path = argv[2], out_path = argv[3], rep_path = argv[4];
    std::vector<std::string> forbid, info;
    bool in_info = false;
    for (int i = 5; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--info") in_info = true;
        else (in_info ? info : forbid).push_back(s);
    }
    auto docs = lines(docs_path);
    std::ofstream rep(rep_path);
    std::vector<std::vector<std::uint64_t>> per;
    for (const auto& f : forbid) per.push_back(span_set({f}));
    const auto all = span_set(forbid);
    std::size_t in_bytes = 0;
    for (const auto& d : docs) in_bytes += d.size() + 1;
    rep << "docs_in " << docs.size() << "\nbytes_in " << in_bytes << "\n";
    for (std::size_t f = 0; f < forbid.size(); ++f) {
        std::size_t nd = 0, ns = 0;
        for (const auto& d : docs) {
            const auto p = hit_pos(d, per[f]);
            nd += !p.empty(), ns += p.size();
        }
        rep << "forbid " << forbid[f] << " spans_in_set " << per[f].size() << " docs_hit " << nd << " windows_hit " << ns << "\n";
        rep.flush();
    }
    std::size_t dropped = 0, dropped_bytes = 0, touched = 0, cut_sent = 0, cut_bytes = 0, rounds_max = 0, emptied = 0;
    std::ofstream out(out_path, std::ios::binary);
    std::size_t kept = 0, out_bytes = 0;
    for (auto& d : docs) {
        auto p = hit_pos(d, all);
        if (!p.empty()) {
            if (covered(p) * 10 > d.size()) {
                ++dropped, dropped_bytes += d.size() + 1;
                continue;
            }
            ++touched;
            const std::size_t before = d.size();
            std::size_t r = 0;
            while (!p.empty()) {
                d = cut(d, p, cut_sent);
                p = hit_pos(d, all);
                ++r;
            }
            rounds_max = std::max(rounds_max, r);
            cut_bytes += before - d.size();
            if (d.empty()) {
                ++emptied;
                continue;
            }
        }
        out << d << "\n";
        ++kept, out_bytes += d.size() + 1;
    }
    out.close();
    rep << "docs_dropped_over_10pct " << dropped << "\nbytes_dropped " << dropped_bytes << "\ndocs_with_sentences_cut " << touched
        << "\nsentences_cut " << cut_sent << "\nbytes_cut " << cut_bytes << "\ndocs_emptied_by_cuts " << emptied
        << "\nmax_rounds " << rounds_max << "\ndocs_out " << kept << "\nbytes_out " << out_bytes << "\n";
    // INFO files: counted on the kept documents.
    const auto outd = lines(out_path);
    for (const auto& f : info) {
        const auto set = span_set({f});
        std::size_t nd = 0, ns = 0;
        for (const auto& d : outd) {
            const auto p = hit_pos(d, set);
            nd += !p.empty(), ns += p.size();
        }
        rep << "info " << f << " spans_in_set " << set.size() << " kept_docs_hit " << nd << " windows_hit " << ns << "\n";
    }
    std::size_t bad = 0;
    for (const auto& d : outd) bad += !hit_pos(d, all).empty();
    rep << "recheck_kept_docs_with_forbidden_span " << bad << "\n";
    std::printf("clean: %zu docs in, %zu out; recheck %zu\n", docs.size(), kept, bad);
    return bad ? 1 : 0;
}
