// Other names of a page, learned from the dump's own markup (docs/reports/PLAN_FACTS.md, GENERATION_PLAN §13.42 Q1).
//
//   fs_aliases DUMP OUT.tsv [--skip FILE] [--drop LO:HI]... [--min N] [--share F]
//
// Two sources, both CyphaLM's reading of the text:
//   redirect  a redirect page's title names its target ("#REDIRECT [[Gunpowder]]" on page "Black powder");
//   anchor    a link's display text names its target ([[Robert Walpole|Walpole]]), kept when the display was used for
//             that target at least N times (default 2) and for at least a share F of all links with that display
//             (default 0.5; unpiped links count for their own title).
// Targets are followed through redirects (up to 3 hops). Only main-namespace pages are read. Leakage guards: a page
// whose text starts in a --drop range (the prompts' own bytes, the holdout) or on a --skip line (the overlap guard's
// pages, scripts/fs_skip_pages.cpp) gives nothing.
// OUT: alias (lowercased) \t target \t kind \t count \t display_total \t byte_offset (first evidence).
// Build: g++ -std=c++20 -O2 scripts/fs_aliases.cpp -o fs_aliases
// Author: Odin Loch
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
std::string xml_decode(std::string_view s) {
    std::string o;
    for (std::size_t i = 0; i < s.size();) {
        if (s[i] == '&') {
            if (s.compare(i, 5, "&amp;") == 0) { o += '&'; i += 5; continue; }
            if (s.compare(i, 4, "&lt;") == 0) { o += '<'; i += 4; continue; }
            if (s.compare(i, 4, "&gt;") == 0) { o += '>'; i += 4; continue; }
            if (s.compare(i, 6, "&quot;") == 0) { o += '"'; i += 6; continue; }
        }
        o += s[i++];
    }
    return o;
}
std::string trim(std::string s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}
std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
bool is_ns(const std::string& t) {  // a namespace prefix ("Image:", "Category:", "fr:")
    const std::size_t c = t.find(':');
    if (c == std::string::npos || c == 0 || c + 1 >= t.size() || t[c + 1] == ' ') return false;
    const std::string p = lower(t.substr(0, c));
    static const std::unordered_set<std::string> N = {"media", "special", "talk", "user", "user talk", "wikipedia", "wikipedia talk", "image",
                                                      "image talk", "mediawiki", "template", "template talk", "help", "category", "portal",
                                                      "category talk", "wiktionary", "wikt", "w", "meta", "commons"};
    if (N.count(p)) return true;
    return p.size() <= 3 && p.find(' ') == std::string::npos;  // interwiki ("fr", "de", "zh-min-nan" is rare)
}
// A link target as a page title: '_' as ' ', no "#section", first letter upper case.
std::string norm_target(std::string t) {
    for (char& c : t) if (c == '_') c = ' ';
    const std::size_t h = t.find('#');
    if (h != std::string::npos) t = t.substr(0, h);
    t = trim(t);
    if (!t.empty() && t[0] == ':') t = trim(t.substr(1));
    if (!t.empty() && std::islower(static_cast<unsigned char>(t[0]))) t[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(t[0])));
    return t;
}
bool range(const std::string& s, std::size_t& lo, std::size_t& hi) {
    const std::size_t c = s.find(':');
    if (c == std::string::npos) return false;
    lo = std::strtoull(s.substr(0, c).c_str(), nullptr, 10), hi = std::strtoull(s.substr(c + 1).c_str(), nullptr, 10);
    return true;
}
struct Pair {
    std::uint32_t n = 0;
    std::size_t off = 0;
};
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: fs_aliases DUMP OUT.tsv [--skip FILE] [--drop LO:HI]... [--min N] [--share F]\n";
        return 2;
    }
    std::vector<std::pair<std::size_t, std::size_t>> drops;
    std::unordered_set<std::size_t> skip;
    unsigned min_n = 2;
    double share = 0.5;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        std::size_t lo, hi;
        if (a == "--drop" && range(v, lo, hi)) drops.push_back({lo, hi});
        else if (a == "--skip") {
            std::ifstream in(v);
            if (!in) return std::cerr << "cannot read " << v << "\n", 1;
            for (std::string l; std::getline(in, l);)
                if (!l.empty() && std::isdigit(static_cast<unsigned char>(l[0]))) skip.insert(std::strtoull(l.c_str(), nullptr, 10));
        } else if (a == "--min") min_n = static_cast<unsigned>(std::atoi(v.c_str()));
        else if (a == "--share") share = std::atof(v.c_str());
        else return std::cerr << "bad option " << a << "\n", 2;
    }
    std::string data;
    {
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) return std::cerr << "cannot read " << argv[1] << "\n", 1;
        data.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const std::string_view sv(data);
    std::unordered_map<std::string, std::pair<std::string, std::size_t>> redir;  // title -> (target, offset)
    std::unordered_map<std::string, std::uint32_t> total;                        // lowercased display -> links
    std::unordered_map<std::string, Pair> pairs;                                 // lowercased display \x1f target
    std::unordered_map<std::string, std::string> shown;                          // lowercased display -> a surface
    std::size_t pages = 0, kept = 0, dropped = 0, skipped = 0, nredir = 0, links = 0;
    for (std::size_t pos = 0;;) {
        const std::size_t p = sv.find("<page>", pos);
        if (p == std::string_view::npos) break;
        const std::size_t e = sv.find("</page>", p);
        if (e == std::string_view::npos) break;
        pos = e + 7;
        ++pages;
        const std::size_t a = sv.find("<title>", p), b = sv.find("</title>", p);
        if (a > e || b > e) continue;
        const std::string title = xml_decode(sv.substr(a + 7, b - a - 7));
        if (is_ns(title)) continue;
        const std::size_t t0 = sv.find("<text", p);
        if (t0 == std::string_view::npos || t0 > e) continue;
        const std::size_t gt = sv.find('>', t0);
        if (gt == std::string_view::npos || gt > e || sv[gt - 1] == '/') continue;
        const std::size_t t1 = sv.find("</text>", gt);
        if (t1 == std::string_view::npos || t1 > e) continue;
        const std::size_t tb = gt + 1;
        bool drop = false;
        for (const auto& [lo, hi] : drops) drop = drop || (tb >= lo && tb < hi);
        if (drop) { ++dropped; continue; }
        if (skip.count(tb)) { ++skipped; continue; }
        ++kept;
        const std::string_view body = sv.substr(tb, t1 - tb);
        std::size_t s = 0;
        while (s < body.size() && std::isspace(static_cast<unsigned char>(body[s]))) ++s;
        if (lower(std::string(body.substr(s, 9))) == "#redirect") {
            const std::size_t l0 = body.find("[[", s), l1 = l0 == std::string_view::npos ? l0 : body.find("]]", l0);
            if (l1 == std::string_view::npos) continue;
            std::string tg = xml_decode(body.substr(l0 + 2, l1 - l0 - 2));
            const std::size_t bar = tg.find('|');
            if (bar != std::string::npos) tg = tg.substr(0, bar);
            tg = norm_target(tg);
            if (!tg.empty() && !is_ns(tg) && tg != title) redir[title] = {tg, tb}, ++nredir;
            continue;
        }
        for (std::size_t i = body.find("[["); i != std::string_view::npos; i = body.find("[[", i + 2)) {
            const std::size_t j = body.find("]]", i + 2);
            if (j == std::string_view::npos) break;
            const std::string_view inner = body.substr(i + 2, j - i - 2);
            if (inner.empty() || inner.size() > 160 || inner.find("[[") != std::string_view::npos || inner.find('\n') != std::string_view::npos) continue;
            const std::size_t bar = inner.find('|');
            std::string tg = norm_target(xml_decode(inner.substr(0, bar)));
            if (tg.empty() || is_ns(tg)) continue;
            std::string disp = bar == std::string_view::npos ? tg : trim(xml_decode(inner.substr(bar + 1)));
            if (disp.empty() || disp.size() > 60 || disp.find_first_of("[]{}|<>") != std::string::npos) continue;
            ++links;
            const std::string ld = lower(disp);
            ++total[ld];
            if (ld == lower(tg)) continue;
            auto& pr = pairs[ld + '\x1f' + tg];
            if (pr.n++ == 0) pr.off = tb + i;
            shown.try_emplace(ld, disp);
        }
    }
    const auto follow = [&](std::string t) {
        for (int k = 0; k < 3; ++k) {
            const auto it = redir.find(t);
            if (it == redir.end()) break;
            t = it->second.first;
        }
        return t;
    };
    std::ofstream out(argv[2]);
    out << "alias\ttarget\tkind\tcount\tdisplay_total\tbyte_offset\n";
    std::size_t nr = 0, na = 0;
    for (const auto& [t, v] : redir) {
        const std::string tg = follow(v.first);
        if (lower(t) == lower(tg)) continue;
        out << lower(t) << '\t' << tg << "\tredirect\t1\t1\t" << v.second << '\n';
        ++nr;
    }
    for (const auto& [k, v] : pairs) {
        const std::size_t c = k.find('\x1f');
        const std::string ld = k.substr(0, c);
        const std::uint32_t tot = total[ld];
        if (v.n < min_n || static_cast<double>(v.n) < share * static_cast<double>(tot)) continue;
        bool digits = true;
        for (const char ch : ld) digits = digits && (std::isdigit(static_cast<unsigned char>(ch)) || ch == ' ' || ch == '-');
        if (digits) continue;
        const std::string tg = follow(k.substr(c + 1));
        if (ld == lower(tg)) continue;
        out << ld << '\t' << tg << "\tanchor\t" << v.n << '\t' << tot << '\t' << v.off << '\n';
        ++na;
    }
    std::fprintf(stderr, "pages %zu, main-namespace read %zu, dropped %zu, skipped %zu; redirects %zu, links %zu, piped pairs %zu; written: redirect %zu, anchor %zu\n",
                 pages, kept, dropped, skipped, nredir, links, pairs.size(), nr, na);
    return 0;
}
