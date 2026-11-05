// The fact store's coverage of GEN-A prompts (docs/reports/FACT_STORE_SCALE.md, GENERATION_PLAN §13.38 A).
//
//   fs_coverage STOREDIR[,RAWFRAMES] PROMPTS.tsv [--plan-drop LO:HI] [--holdout LO:HI] [--list OUT.tsv]
//
// Loads the store the way `cyphalm_trie sample --plan` does (ContentPlanStore::load, unchanged; the holdout range
// defaults to sample's default [7340032, 8388608)), builds the frames the same way (load_frames and render_frame are
// copies of cyphalm_trie.cpp's), and runs the content plan's own entity finder on every prompt (the prompt column of
// a GEN-A prompts file). Prints two counts with Wilson 95% intervals:
//   planned: the plan is not empty (sample's "plan coverage", GEN-E's coverage);
//   entity:  some candidate the finder lists has at least one fact in the store (frames or not).
// Build: g++ -std=c++20 -O2 -I native/include scripts/fs_coverage.cpp -o fs_coverage
// Author: Odin Loch
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>

#include "cypha/lm/content_plan.hpp"

using cypha::cyphalm::ContentPlanStore;

namespace {
std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> f;
    std::size_t a = 0;
    for (;;) {
        const std::size_t b = s.find(d, a);
        f.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) return f;
        a = b + 1;
    }
}
std::string unescape_tsv(const std::string& s) {
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            o += c == 't' ? '\t' : c == 'n' ? '\n' : c;
        } else o += s[i];
    }
    return o;
}
std::string render_frame(const std::string& pat, const std::string& subj, const std::string& obj) {
    std::string out, prev;
    for (std::string t : split(pat, ' ')) {
        if (t.empty() || t == "^") continue;
        const bool glue = (t.size() == 1 && std::strchr(",.;:)'", t[0])) || prev == "(" || (prev == "'" && t == "s");
        prev = t;
        if (t == "[S]" || t == "[P]") t = subj;
        else if (t == "[O]") t = obj;
        if (!out.empty() && !glue) out += ' ';
        out += t;
    }
    return out;
}
std::map<std::string, std::vector<std::string>> load_frames(const std::string& path) {
    std::map<std::string, std::vector<std::tuple<int, int, std::string>>> by;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto f = split(line, '\t');
        if (f.size() < 11 || f[0] == "relation") continue;
        const bool raw = f.size() >= 12 && !f[11].empty();
        const std::string& p = raw ? f[11] : f[1];
        const std::size_t subj = std::min(p.find("[S]"), p.find("[P]")), obj = p.find("[O]");
        if (obj == std::string::npos || subj == std::string::npos || obj < subj || p.find("<M>") != std::string::npos ||
            p.find('#') != std::string::npos)
            continue;
        std::string r = p;
        for (const char* t : {"[S]", "[P]", "[O]"})
            for (std::size_t q; (q = r.find(t)) != std::string::npos;) r.replace(q, 3, t[1] == 'O' ? "\x02" : "\x01");
        by[f[0]].push_back({-std::atoi(f[10].c_str()), -std::atoi(f[2].c_str()), raw ? r : render_frame(p, "\x01", "\x02")});
    }
    std::map<std::string, std::vector<std::string>> out;
    for (auto& [rel, v] : by) {
        std::sort(v.begin(), v.end());
        auto& o = out[rel];
        for (const auto& t : v)
            if (o.size() < 3 && std::find(o.begin(), o.end(), std::get<2>(t)) == o.end()) o.push_back(std::get<2>(t));
    }
    return out;
}
void wilson(double k, double n, double& lo, double& hi) {
    const double z = 1.959963984540054, p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n);
    const double r = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    lo = (c - r) / d, hi = (c + r) / d;
}
bool range(const std::string& s, std::size_t& lo, std::size_t& hi) {
    const std::size_t c = s.find(':');
    if (c == std::string::npos) return false;
    lo = std::strtoull(s.substr(0, c).c_str(), nullptr, 10), hi = std::strtoull(s.substr(c + 1).c_str(), nullptr, 10);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: fs_coverage STOREDIR[,RAWFRAMES] PROMPTS.tsv [--plan-drop LO:HI] [--holdout LO:HI] [--list OUT]\n";
        return 2;
    }
    std::size_t lo = 7'340'032, hi = 8'388'608, lo2 = 0, hi2 = 0;
    std::string list;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--plan-drop" && range(argv[i + 1], lo2, hi2)) continue;
        if (a == "--holdout" && range(argv[i + 1], lo, hi)) continue;
        if (a == "--list") { list = argv[i + 1]; continue; }
        std::cerr << "bad option " << a << "\n";
        return 2;
    }
    const std::string spec = argv[1];
    const std::size_t comma = spec.find(',');
    const std::string dir = spec.substr(0, comma);
    ContentPlanStore st;
    std::string err;
    if (!st.load(dir, lo, hi, err, lo2, hi2)) {
        std::cerr << err << "\n";
        return 1;
    }
    auto frames = load_frames(dir + "/patterns.tsv");
    if (comma != std::string::npos)
        for (auto& [r, v] : load_frames(spec.substr(comma + 1))) {
            auto& o = frames[r];
            for (auto& f : v)
                if (std::find(o.begin(), o.end(), f) == o.end()) o.insert(o.begin(), f);
        }
    // "entity" uses the finder with every relation allowed: frames for all relations the store holds.
    std::map<std::string, std::vector<std::string>> any = frames;
    for (const char* r : {"capital", "largest_city", "country", "location", "located_in", "birth_place", "death_place",
                          "birth_year", "death_year", "founded", "author", "director", "composer", "founder", "creator",
                          "spouse", "predecessor", "successor", "developer", "publisher", "owner", "genre", "language",
                          "currency", "occupation", "nationality", "party", "family", "order", "class", "inventor"})
        if (any[r].empty()) any[r].push_back("\x01 \x02");
    std::ifstream in(argv[2]);
    std::string line;
    std::getline(in, line);
    std::ofstream lo_out;
    if (!list.empty()) lo_out.open(list);
    int n = 0, planned = 0, entity = 0;
    while (std::getline(in, line)) {
        const auto f = split(line, '\t');
        if (f.size() < 4) continue;
        const std::string p = unescape_tsv(f[2]);
        const auto pl = st.plan(p, frames);
        const auto pe = st.plan(p, any);
        ++n, planned += !pl.facts.empty(), entity += !pe.facts.empty();
        if (lo_out) {
            lo_out << f[0] << '\t' << pl.entity << '\t';
            for (std::size_t q = 0; q < pl.facts.size(); ++q) lo_out << (q ? "|" : "") << pl.facts[q].rel << '=' << pl.facts[q].value;
            lo_out << '\t' << pe.entity << '\n';
        }
    }
    double a, b;
    std::printf("store %s subjects %zu relations_with_frames %zu prompts %d\n", spec.c_str(), st.subjects(), frames.size(), n);
    wilson(planned, n, a, b);
    std::printf("planned %d of %d = %.3f [%.3f, %.3f]\n", planned, n, double(planned) / n, a, b);
    wilson(entity, n, a, b);
    std::printf("entity  %d of %d = %.3f [%.3f, %.3f]\n", entity, n, double(entity) / n, a, b);
    return 0;
}
