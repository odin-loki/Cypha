// docs/reports/K5_RETRIEVE.md: the page title of each cleaned document, for the retrieval's leak check.
//
//   k5_ret_titles ENWIK LO_MIB HI_MIB OUT.tsv
//
// The same pages, in the same order, as `scribe docs enwik:ENWIK:LO:HI` (eval/scribe.hpp clean_wiki_documents: a page
// cut by either end of the range, a redirect, or a page with no prose line gives no document). One row per document:
// its index, the page's byte offset in ENWIK, its title, the document's bytes and FNV-1a 64 hash. It reads and writes
// nothing else and changes no model.
//
// Build: g++ -std=c++23 -O2 -I native/include -o k5_ret_titles native/tools/lm/k5_ret_titles.cpp
//
// Author: Odin Loch
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "cypha/eval/scribe.hpp"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: k5_ret_titles ENWIK LO_MIB HI_MIB OUT.tsv\n");
        return 2;
    }
    constexpr std::size_t MiB = 1 << 20;
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) return std::fprintf(stderr, "cannot open %s\n", argv[1]), 1;
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    const std::size_t b = std::min(size, static_cast<std::size_t>(std::strtod(argv[2], nullptr) * MiB));
    const std::size_t e = std::min(size, static_cast<std::size_t>(std::strtod(argv[3], nullptr) * MiB));
    std::string raw(e > b ? e - b : 0, '\0');
    in.seekg(static_cast<std::streamoff>(b));
    in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    std::FILE* out = std::fopen(argv[4], "w");
    if (!out) return std::fprintf(stderr, "cannot write %s\n", argv[4]), 1;
    std::fprintf(out, "index\toffset\ttitle\tbytes\tfnv64\n");
    // clean_wiki_documents' loop, with the title kept.
    const std::string_view sv(raw);
    std::size_t pos = 0, n = 0;
    while (true) {
        const std::size_t p = sv.find("<page>", pos);
        if (p == std::string_view::npos) break;
        const std::size_t q = sv.find("</page>", p);
        if (q == std::string_view::npos) break;
        pos = q + 7;
        const std::string_view page = sv.substr(p, q - p);
        const std::size_t t0 = page.find("<text");
        if (t0 == std::string_view::npos) continue;
        const std::size_t gt = page.find('>', t0);
        if (gt == std::string_view::npos || page[gt - 1] == '/') continue;
        const std::size_t t1 = page.find("</text>", gt);
        if (t1 == std::string_view::npos) continue;
        const std::string body(page.substr(gt + 1, t1 - gt - 1));
        std::size_t s = 0;
        while (s < body.size() && cypha::eval::scribe_detail::is_space_a(static_cast<unsigned char>(body[s]))) ++s;
        if (cypha::eval::scribe_detail::lower_ascii(body.substr(s, 9)) == "#redirect") continue;
        const std::string d = cypha::eval::clean_wiki_body(body);
        if (d.empty()) continue;
        std::string title;
        const std::size_t a0 = page.find("<title>");
        const std::size_t a1 = a0 == std::string_view::npos ? a0 : page.find("</title>", a0);
        if (a1 != std::string_view::npos) title = std::string(page.substr(a0 + 7, a1 - a0 - 7));
        for (char& c : title)
            if (c == '\t' || c == '\n') c = ' ';
        std::uint64_t h = 1469598103934665603ull;
        for (const unsigned char c : d) h = (h ^ c) * 1099511628211ull;
        std::fprintf(out, "%zu\t%zu\t%s\t%zu\t%016llx\n", n++, b + p, title.c_str(), d.size(), static_cast<unsigned long long>(h));
    }
    std::fclose(out);
    std::fprintf(stderr, "%zu documents\n", n);
    return 0;
}
