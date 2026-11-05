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
//
// PLAN_FACTS (docs/reports/PLAN_FACTS.md, GENERATION_PLAN §13.42 Q1): the planner's own paths, without K5.
//   fs_coverage STOREDIR PROMPTS.tsv --planner SKELBANK [--find 1|2] [--filter PARAMS|on] [--stats S10.stats]
//               [--plan-drop LO:HI] [--holdout LO:HI] [--list OUT.tsv]
// loads the store as `sample --k5-plan` does (planner::FactStore, with the same flags) and runs make_plan's main-entity
// rule (writer.hpp: the candidates in order, the first that is a subject with a fact of a bank relation with at most
// 2 values; up to 4 facts). With --stats it also runs the relations-pattern fallback (fallback.hpp) and the
// co-occurring one on the rest. Prints each path's count with Wilson 95% intervals and the facts per main plan.
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
#include "cypha/lm/planner/fallback.hpp"
#include "cypha/lm/planner/facts.hpp"
#include "cypha/lm/planner/skeleton.hpp"

using cypha::cyphalm::ContentPlanStore;
namespace pln = cypha::cyphalm::planner;

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

// PLAN_FACTS: the planner's paths (see the header).
int planner_paths(const std::string& dir, const std::string& prompts, const std::string& bank_path, const std::string& stats_path,
                  const std::vector<std::pair<std::size_t, std::size_t>>& drops, const std::string& list) {
    pln::FactStore fs;
    pln::SkelBank bank;
    std::string err;
    if (!fs.load(dir, drops, err) || !bank.load(bank_path, err)) {
        std::cerr << err << "\n";
        return 1;
    }
    cypha::cyphalm::structure::Stats stats;
    const bool fb = !stats_path.empty();
    if (fb && !stats.load(stats_path, err)) {
        std::cerr << err << "\n";
        return 1;
    }
    std::ifstream in(prompts);
    std::string line;
    std::getline(in, line);
    std::ofstream lo_out;
    if (!list.empty()) lo_out.open(list);
    int n = 0, n_main = 0, n_pat = 0, n_cooc = 0;
    std::size_t facts_main = 0, facts_pat = 0;
    std::map<std::string, int> rel_main;
    while (std::getline(in, line)) {
        const auto f = split(line, '\t');
        if (f.size() < 4) continue;
        const std::string p = unescape_tsv(f[2]);
        ++n;
        const auto cands = fs.candidates(p, fb ? &stats : nullptr);
        std::string path = "k5", ent = "-", facts;
        for (const auto& c : cands) {
            const std::uint32_t e = fs.find_main(c);
            if (e == ~0u) continue;
            std::map<std::uint32_t, std::vector<std::uint32_t>> by;
            for (const auto fi : fs.of_subj(e))
                if (bank.of(fs.rel(fs.fact(fi).rel))) by[fs.fact(fi).rel].push_back(fi);
            std::vector<std::uint32_t> pick;
            for (const auto& [r, v] : by) {
                std::set<std::uint32_t> vals;
                for (const auto fi : v) vals.insert(fs.fact(fi).val);
                if (vals.size() <= 2) pick.push_back(v[0]);
            }
            if (pick.empty()) continue;
            path = "main", ent = fs.ent(e) + (c == fs.ent(e) ? "" : " <- " + c);
            facts_main += std::min<std::size_t>(pick.size(), 4);
            for (const auto fi : pick) {
                ++rel_main[fs.rel(fs.fact(fi).rel)];
                facts += (facts.empty() ? "" : "|") + fs.rel(fs.fact(fi).rel) + "=" + fs.ent(fs.fact(fi).val);
            }
            break;
        }
        if (path == "k5" && fb && pln::fact_options().fallback == "pattern") {
            const auto pk = pln::typed_fallback(fs, bank, stats, cands, p);
            if (!pk.facts.empty()) {
                path = "pattern" + std::to_string(pk.tier), ent = fs.ent(pk.ent), facts_pat += pk.facts.size(), ++n_pat;
                for (const auto fi : pk.facts)
                    facts += (facts.empty() ? "" : "|") + fs.ent(fs.fact(fi).subj) + ":" + fs.rel(fs.fact(fi).rel) + "=" + fs.ent(fs.fact(fi).val);
            }
        }
        if (path == "k5") {  // make_plan's co-occurring fallback
            for (const auto& c : cands) {
                const std::uint32_t e = fs.find(c);
                if (e == ~0u) continue;
                for (const auto fi : fs.of_val(e))
                    if (bank.of(fs.rel(fs.fact(fi).rel))) path = "cooc";
                if (path == "cooc") break;
            }
            n_cooc += path == "cooc";
        }
        n_main += path == "main";
        if (lo_out) lo_out << f[0] << '\t' << path << '\t' << ent << '\t' << facts << '\n';
    }
    double a, b;
    std::printf("planner store %s: %zu facts (%zu dropped, %zu filtered), %zu subjects, %zu aliases; find %d, filter %s\n", dir.c_str(), fs.size(),
                fs.dropped(), fs.filtered(), fs.subjects(), fs.aliases(), fs.options().find, fs.options().filter.empty() ? "off" : fs.options().filter.c_str());
    wilson(n_main, n, a, b);
    std::printf("main    %d of %d = %.3f [%.3f, %.3f]; facts per main plan %.2f\n", n_main, n, double(n_main) / n, a, b,
                n_main ? double(facts_main) / n_main : 0.0);
    if (fb) {
        wilson(n_pat, n, a, b);
        std::printf("pattern %d of %d = %.3f [%.3f, %.3f]; facts per plan %.2f\n", n_pat, n, double(n_pat) / n, a, b, n_pat ? double(facts_pat) / n_pat : 0.0);
    }
    wilson(n_cooc, n, a, b);
    std::printf("cooc    %d of %d = %.3f [%.3f, %.3f]\n", n_cooc, n, double(n_cooc) / n, a, b);
    std::printf("k5      %d of %d\n", n - n_main - n_pat - n_cooc, n);
    std::printf("main-plan relations:");
    for (const auto& [r, k] : rel_main) std::printf(" %s %d", r.c_str(), k);
    std::printf("\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: fs_coverage STOREDIR[,RAWFRAMES] PROMPTS.tsv [--plan-drop LO:HI] [--holdout LO:HI] [--list OUT]\n"
                     "       fs_coverage STOREDIR PROMPTS.tsv --planner SKELBANK [--find 1|2] [--filter PARAMS|on] [--stats FILE] [...]\n";
        return 2;
    }
    std::size_t lo = 7'340'032, hi = 8'388'608, lo2 = 0, hi2 = 0;
    std::string list, bank, stats;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--plan-drop" && range(argv[i + 1], lo2, hi2)) continue;
        if (a == "--holdout" && range(argv[i + 1], lo, hi)) continue;
        if (a == "--list") { list = argv[i + 1]; continue; }
        if (a == "--planner") { bank = argv[i + 1]; continue; }
        if (a == "--stats") { stats = argv[i + 1]; continue; }
        if (a == "--find") { pln::fact_options().find = std::atoi(argv[i + 1]); continue; }
        if (a == "--filter") { pln::fact_options().filter = argv[i + 1]; continue; }
        if (a == "--fallback") { pln::fact_options().fallback = argv[i + 1]; continue; }
        std::cerr << "bad option " << a << "\n";
        return 2;
    }
    if (!bank.empty()) {
        std::vector<std::pair<std::size_t, std::size_t>> drops = {{lo, hi}};
        if (hi2 > lo2) drops.push_back({lo2, hi2});
        return planner_paths(argv[1], argv[2], bank, stats, drops, list);
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
