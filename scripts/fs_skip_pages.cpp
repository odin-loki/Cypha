// The overlap guard of the fact store over enwik9 (docs/reports/FACT_STORE_SCALE.md, GENERATION_PLAN §13.38 A).
//
//   fs_skip_pages DUMP LO OUT.tsv FORBID...
//
// Every complete <page> of DUMP whose <page> tag starts at byte LO or later is cleaned to prose exactly as SCRIBE's
// `docs` does (clean_wiki_body, redirects skipped). A page is a hit if any 32-byte window of its prose has the 64-bit
// FNV-1a hash of a 32-byte span of a FORBID text (one text per line), the test of scripts/k5_overlap.cpp (K5_SCALE's
// overlap check). OUT gets one line per hit page: its text start offset (pattern_learn's PageRef.tb, the key of
// --skip-pages), a bit mask of the FORBID files it hits, its window count per file, and its title.
// On stderr: pages read, docs, and hit pages per FORBID file (to compare with overlap_report.txt).
// Build: g++ -std=c++20 -O2 -pthread -I native/include scripts/fs_skip_pages.cpp -o fs_skip_pages
// Author: Odin Loch
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "cypha/eval/scribe.hpp"

namespace {
constexpr std::size_t K = 32;
std::uint64_t fnv(const char* p) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < K; ++i) h = (h ^ static_cast<unsigned char>(p[i])) * 1099511628211ull;
    return h;
}
std::vector<std::uint64_t> span_set(const std::string& f) {
    std::ifstream in(f, std::ios::binary);
    if (!in) {
        std::cerr << "cannot read " << f << "\n";
        std::exit(2);
    }
    std::vector<std::uint64_t> h;
    for (std::string t; std::getline(in, t);)
        for (std::size_t i = 0; i + K <= t.size(); ++i) h.push_back(fnv(t.data() + i));
    std::sort(h.begin(), h.end());
    h.erase(std::unique(h.begin(), h.end()), h.end());
    return h;
}
struct Page {
    std::size_t tb = 0, te = 0;
    std::string title;
};
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: fs_skip_pages DUMP LO OUT.tsv FORBID...\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    const std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::size_t lo = std::strtoull(argv[2], nullptr, 10);
    std::vector<std::vector<std::uint64_t>> sets;
    for (int i = 4; i < argc; ++i) sets.push_back(span_set(argv[i]));
    std::vector<Page> pages;
    std::size_t pos = lo;
    const std::string_view sv(data);
    while (true) {  // clean_wiki_documents' walk, keeping each page's offsets and title
        const std::size_t p = sv.find("<page>", pos);
        if (p == std::string_view::npos) break;
        const std::size_t e = sv.find("</page>", p);
        if (e == std::string_view::npos) break;
        pos = e + 7;
        const std::size_t t0 = sv.find("<text", p);
        if (t0 == std::string_view::npos || t0 > e) continue;
        const std::size_t gt = sv.find('>', t0);
        if (gt == std::string_view::npos || gt > e || sv[gt - 1] == '/') continue;
        const std::size_t t1 = sv.find("</text>", gt);
        if (t1 == std::string_view::npos || t1 > e) continue;
        Page pg;
        pg.tb = gt + 1, pg.te = t1;
        const std::size_t a = sv.find("<title>", p), b = sv.find("</title>", p);
        if (a < e && b < e) pg.title = std::string(sv.substr(a + 7, b - a - 7));
        pages.push_back(std::move(pg));
    }
    const std::size_t nf = sets.size();
    std::vector<std::vector<std::size_t>> hits(pages.size(), std::vector<std::size_t>(nf, 0));
    std::vector<char> is_doc(pages.size(), 0);
    const unsigned T = 8;
    std::vector<std::thread> th;
    for (unsigned t = 0; t < T; ++t)
        th.emplace_back([&, t]() {
            for (std::size_t i = t; i < pages.size(); i += T) {
                const std::string body = data.substr(pages[i].tb, pages[i].te - pages[i].tb);
                std::size_t b = 0;
                while (b < body.size() && cypha::eval::scribe_detail::is_space_a(static_cast<unsigned char>(body[b]))) ++b;
                if (cypha::eval::scribe_detail::lower_ascii(body.substr(b, 9)) == "#redirect") continue;
                const std::string d = cypha::eval::clean_wiki_body(body);
                if (d.empty()) continue;
                is_doc[i] = 1;
                for (std::size_t w = 0; w + K <= d.size(); ++w) {
                    const std::uint64_t h = fnv(d.data() + w);
                    for (std::size_t f = 0; f < nf; ++f)
                        if (std::binary_search(sets[f].begin(), sets[f].end(), h)) ++hits[i][f];
                }
            }
        });
    for (auto& x : th) x.join();
    std::ofstream out(argv[3]);
    out << "text_offset\tmask\twindows\ttitle\n";
    std::size_t docs = 0, any = 0;
    std::vector<std::size_t> per(nf, 0);
    for (std::size_t i = 0; i < pages.size(); ++i) {
        docs += is_doc[i];
        unsigned mask = 0;
        std::string ws;
        for (std::size_t f = 0; f < nf; ++f) {
            if (hits[i][f]) mask |= 1u << f, ++per[f];
            ws += (f ? "," : "") + std::to_string(hits[i][f]);
        }
        if (!mask) continue;
        ++any;
        out << pages[i].tb << '\t' << mask << '\t' << ws << '\t' << pages[i].title << '\n';
    }
    std::fprintf(stderr, "pages %zu docs %zu hit_pages %zu", pages.size(), docs, any);
    for (std::size_t f = 0; f < nf; ++f) std::fprintf(stderr, " forbid%zu_pages %zu", f, per[f]);
    std::fprintf(stderr, "\n");
    return 0;
}
