// The gibberish filter's sample, threshold choice and measure (docs/reports/PLAN_FACTS.md, GENERATION_PLAN §13.42 Q1).
//
//   fs_filter sample  STOREDIR OUTDIR [--n 300] [--seed S] [--plan-drop LO:HI]
//       N facts drawn at random (seeded) from the store as the planner loads it (facts.hpp read_rows; the holdout
//       and --plan-drop, default the GEN-A prompts' bytes, dropped): OUTDIR/labels_packet.txt (numbered "subject — relation — object" lines, nothing
//       else) and OUTDIR/key.tsv (where each fact came from, and its half: the first N/2 of a second seeded order
//       choose the thresholds, the rest measure them).
//   fs_filter choose  STOREDIR KEY OUTPARAMS LABELS... --stats S10.stats   thresholds from the "choose" half (rule below)
//   fs_filter measure STOREDIR KEY PARAMS LABELS... --stats S10.stats      precision and recall on the "measure" half (Wilson 95%)
//   fs_filter stats   STOREDIR PARAMS --stats S10.stats [--plan-drop LO:HI] [--list OUT.tsv]  facts the filter leaves
//       out, by rule and relation (--list: each left-out fact with its reasons)
// --stats: the structure statistics the planner's model reads (its "stats" line); the place rule uses their name types,
// so choose, measure and stats need the same file as the planner.
// LABELS: one file per rater, lines "N<TAB>sensible|gibberish" (also "N. label" or "N label"). A fact's label is the
// raters' majority; a tie is left out of the choice and of precision and recall (and counted). The filter's positive
// class is gibberish: precision = flagged facts that are gibberish / flagged; recall = gibberish facts flagged / all
// gibberish facts; "sensible kept" = sensible facts not flagged / all sensible facts.
// Choice rule (fixed before any label was read): each rule family (fact_filter.hpp) alone, at each of its thresholds,
// flags some choose-half facts; a family is turned on, at the threshold with the largest (gibberish flagged - sensible
// flagged), only if that number is above 0 (its flags are mostly gibberish). Families: shape, self, year, place
// (place_support 1, 2, 3, 5), person (1, 2), name, lex (lex_support 1, 2, 3, 5, 10), taxon, agree, xrel, spread
// (spread_min 2, 3, 5, 10), p (min_p 0.6, 0.8, 0.9).
// Build: g++ -std=c++20 -O2 -I native/include scripts/fs_filter.cpp -o fs_filter
// Author: Odin Loch
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <tuple>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "cypha/lm/planner/facts.hpp"

namespace pl = cypha::cyphalm::planner;

namespace {
void wilson(double k, double n, double& lo, double& hi) {
    if (n <= 0) { lo = hi = 0; return; }
    const double z = 1.959963984540054, p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n);
    const double r = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    lo = (c - r) / d, hi = (c + r) / d;
}
std::vector<std::pair<std::size_t, std::size_t>> drops_of(const std::string& v) {
    std::vector<std::pair<std::size_t, std::size_t>> d = {{7'340'032, 8'388'608}};
    if (!v.empty()) {
        const auto c = v.find(':');
        d.push_back({std::strtoull(v.substr(0, c).c_str(), nullptr, 10), std::strtoull(v.substr(c + 1).c_str(), nullptr, 10)});
    }
    return d;
}
std::string rel_text(const std::string& r) {
    if (r == "is_a") return "is a";
    std::string o = r;
    for (char& c : o) if (c == '_') c = ' ';
    return o;
}
struct Key {
    int n = 0;
    std::string subj, rel, val, half;
    std::size_t row = 0;
};
// key.tsv: n, subject, relation, value, source, template, pattern, P, n_evidence, n_rows, byte_offset, half
std::vector<Key> read_key(const std::string& path, const std::vector<pl::FactRow>& rows) {
    std::map<std::tuple<std::string, std::string, std::string>, std::size_t> at;
    for (std::size_t i = 0; i < rows.size(); ++i) at[{rows[i].subj, rows[i].rel, rows[i].val}] = i;
    std::vector<Key> out;
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        const auto f = pl::split_tab(line);
        if (f.size() < 12) continue;
        Key k;
        k.n = std::atoi(f[0].c_str()), k.subj = f[1], k.rel = f[2], k.val = f[3], k.half = f[11];
        const auto it = at.find({k.subj, k.rel, k.val});
        if (it == at.end()) {
            std::cerr << "key fact " << k.n << " is not in the store\n";
            std::exit(1);
        }
        k.row = it->second;
        out.push_back(k);
    }
    return out;
}
// n -> gibberish votes, sensible votes
std::map<int, std::pair<int, int>> read_labels(const std::vector<std::string>& files) {
    std::map<int, std::pair<int, int>> v;
    for (const auto& p : files) {
        std::ifstream in(p);
        if (!in) {
            std::cerr << "cannot read " << p << "\n";
            std::exit(1);
        }
        std::string line;
        int got = 0;
        while (std::getline(in, line)) {
            std::size_t i = 0;
            while (i < line.size() && !std::isdigit(static_cast<unsigned char>(line[i]))) ++i;
            if (i >= line.size()) continue;
            const int n = std::atoi(line.c_str() + i);
            std::string rest = pl::lower_s(line.substr(i));
            const bool g = rest.find("gibberish") != std::string::npos, s = rest.find("sensible") != std::string::npos;
            if (g == s) continue;
            (g ? v[n].first : v[n].second)++;
            ++got;
        }
        std::fprintf(stderr, "%s: %d labels\n", p.c_str(), got);
    }
    return v;
}
// label: 1 gibberish, 0 sensible, -1 tie or none
int label_of(const std::map<int, std::pair<int, int>>& v, int n) {
    const auto it = v.find(n);
    if (it == v.end()) return -1;
    return it->second.first > it->second.second ? 1 : it->second.first < it->second.second ? 0 : -1;
}
struct Loaded {
    std::vector<pl::FactRow> rows;
    std::size_t dropped = 0;
};
Loaded load_rows(const std::string& dir, const std::string& drop) {
    Loaded L;
    std::string err;
    if (!pl::FactStore::read_rows(dir, drops_of(drop), L.rows, L.dropped, err)) {
        std::cerr << err << "\n";
        std::exit(1);
    }
    return L;
}
struct Counts {
    int tp = 0, fp = 0, fn = 0, tn = 0, ties = 0, ties_flagged = 0;
};
Counts count(const std::vector<Key>& keys, const std::string& half, const std::map<int, std::pair<int, int>>& lab, const pl::FactFilter& ff) {
    Counts c;
    for (const auto& k : keys) {
        if (half != "all" && k.half != half) continue;
        const bool flag = ff.flags(k.row) != 0;
        const int y = label_of(lab, k.n);
        if (y < 0) {
            ++c.ties, c.ties_flagged += flag;
            continue;
        }
        if (y == 1) (flag ? c.tp : c.fn)++;
        else (flag ? c.fp : c.tn)++;
    }
    return c;
}
void report(const char* what, const Counts& c) {
    double a, b;
    const int g = c.tp + c.fn, s = c.fp + c.tn, fl = c.tp + c.fp;
    std::printf("%s: %d facts with a majority label (%d gibberish, %d sensible), %d ties (%d of them flagged)\n", what, g + s, g, s, c.ties, c.ties_flagged);
    wilson(c.tp, fl, a, b);
    std::printf("  precision (flagged that are gibberish) %d / %d = %.3f [%.3f, %.3f]\n", c.tp, fl, fl ? double(c.tp) / fl : 0.0, a, b);
    wilson(c.tp, g, a, b);
    std::printf("  recall (gibberish flagged)             %d / %d = %.3f [%.3f, %.3f]\n", c.tp, g, g ? double(c.tp) / g : 0.0, a, b);
    wilson(c.tn, s, a, b);
    std::printf("  sensible kept                          %d / %d = %.3f [%.3f, %.3f]\n", c.tn, s, s ? double(c.tn) / s : 0.0, a, b);
    wilson(g, g + s, a, b);
    std::printf("  gibberish share of the labelled facts  %d / %d = %.3f [%.3f, %.3f]\n", g, g + s, (g + s) ? double(g) / (g + s) : 0.0, a, b);
}
}  // namespace

int main(int argc0, char** argv0) {
    // --stats FILE may stand anywhere; it is taken out before the positional arguments are read
    std::vector<char*> av;
    std::string stats_path;
    for (int i = 0; i < argc0; ++i) {
        if (std::string(argv0[i]) == "--stats" && i + 1 < argc0) {
            stats_path = argv0[++i];
            continue;
        }
        av.push_back(argv0[i]);
    }
    const int argc = static_cast<int>(av.size());
    char** argv = av.data();
    if (argc < 3) {
        std::cerr << "usage: fs_filter sample|choose|measure|stats ... (see the header)\n";
        return 2;
    }
    const std::string mode = argv[1], dir = argv[2];
    cypha::cyphalm::structure::Stats stats;
    if (mode != "sample") {
        std::string err;
        if (stats_path.empty()) return std::cerr << "fs_filter " << mode << " needs --stats FILE (the planner model's statistics)\n", 2;
        if (!stats.load(stats_path, err)) return std::cerr << err << "\n", 1;
    }
    const cypha::cyphalm::structure::Stats* st = mode == "sample" ? nullptr : &stats;
    if (mode == "sample") {
        if (argc < 4) return 2;
        const std::string outdir = argv[3];
        int n = 300;
        std::uint64_t seed = 20261105;
        std::string drop = "75497472:100000000";  // the GEN-A prompts' bytes, as every gen_eval run drops them
        for (int i = 4; i + 1 < argc; i += 2) {
            const std::string a = argv[i];
            if (a == "--n") n = std::atoi(argv[i + 1]);
            else if (a == "--seed") seed = std::strtoull(argv[i + 1], nullptr, 10);
            else if (a == "--plan-drop") drop = argv[i + 1];
        }
        const auto L = load_rows(dir, drop);
        std::vector<std::size_t> idx(L.rows.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::mt19937_64 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);
        idx.resize(static_cast<std::size_t>(n));
        std::vector<int> half(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) half[static_cast<std::size_t>(i)] = i;
        std::mt19937_64 rng2(seed + 1);
        std::shuffle(half.begin(), half.end(), rng2);
        std::vector<std::string> hname(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) hname[static_cast<std::size_t>(half[static_cast<std::size_t>(i)])] = i < n / 2 ? "choose" : "measure";
        std::ofstream pk(outdir + "/labels_packet.txt"), ky(outdir + "/key.tsv");
        ky << "n\tsubject\trelation\tvalue\tsource\ttemplate\tpattern\tP\tn_evidence\tn_rows\tbyte_offset\thalf\n";
        for (int i = 0; i < n; ++i) {
            const auto& r = L.rows[idx[static_cast<std::size_t>(i)]];
            pk << (i + 1) << ". " << r.subj << " \xe2\x80\x94 " << rel_text(r.rel) << " \xe2\x80\x94 " << r.val << "\n";
            ky << (i + 1) << '\t' << r.subj << '\t' << r.rel << '\t' << r.val << '\t' << r.src << '\t' << r.tpl << '\t' << r.pattern << '\t' << r.p << '\t'
               << r.n_ev << '\t' << r.n_rows << '\t' << r.off << '\t' << hname[static_cast<std::size_t>(i)] << '\n';
        }
        std::printf("store %s: %zu facts (%zu dropped); sampled %d (seed %llu)\n", dir.c_str(), L.rows.size(), L.dropped, n,
                    static_cast<unsigned long long>(seed));
        return 0;
    }
    if (mode == "stats") {
        if (argc < 4) return 2;
        std::string drop, list;
        for (int i = 4; i + 1 < argc; i += 2) {
            if (std::string(argv[i]) == "--plan-drop") drop = argv[i + 1];
            if (std::string(argv[i]) == "--list") list = argv[i + 1];
        }
        const auto L = load_rows(dir, drop);
        std::ofstream lo;
        if (!list.empty()) lo.open(list), lo << "subject\trelation\tvalue\tsource\treasons\n";
        pl::FilterParams fp;
        std::string err;
        if (!fp.read(argv[3], err)) return std::cerr << err << "\n", 1;
        pl::FactFilter ff(fp, st);
        ff.prepare(L.rows);
        std::map<std::string, std::array<std::size_t, 2>> by_rel;
        std::map<std::string, std::size_t> by_reason;
        std::size_t out = 0;
        for (std::size_t i = 0; i < L.rows.size(); ++i) {
            const auto f = ff.flags(i);
            auto& c = by_rel[L.rows[i].rel];
            ++c[0];
            if (!f) continue;
            ++c[1], ++out;
            std::string why;
            for (int b = 0; b < pl::FactFilter::kNReason; ++b)
                if (f & (1u << b)) ++by_reason[pl::FactFilter::reason_name(b)], why += (why.empty() ? "" : ",") + std::string(pl::FactFilter::reason_name(b));
            if (lo) lo << L.rows[i].subj << '\t' << L.rows[i].rel << '\t' << L.rows[i].val << '\t' << L.rows[i].src << '\t' << why << '\n';
        }
        std::printf("store %s: %zu facts, filter leaves out %zu (%.3f)\n", dir.c_str(), L.rows.size(), out, double(out) / double(L.rows.size()));
        for (const auto& [r, n] : by_reason) std::printf("  rule %-7s %zu\n", r.c_str(), n);
        for (const auto& [r, c] : by_rel) std::printf("  %-13s %7zu facts, %6zu left out (%.3f)\n", r.c_str(), c[0], c[1], c[0] ? double(c[1]) / double(c[0]) : 0.0);
        return 0;
    }
    if (mode == "choose" || mode == "measure") {
        if (argc < 6) return 2;
        const auto L = load_rows(dir, "75497472:100000000");
        const auto keys = read_key(argv[3], L.rows);
        std::vector<std::string> files(argv + 5, argv + argc);
        const auto lab = read_labels(files);
        if (mode == "measure") {
            pl::FilterParams fp;
            std::string err;
            if (!fp.read(argv[4], err)) return std::cerr << err << "\n", 1;
            pl::FactFilter ff(fp, st);
            ff.prepare(L.rows);
            report("measure half", count(keys, "measure", lab, ff));
            report("choose half (seen)", count(keys, "choose", lab, ff));
            report("all 300", count(keys, "all", lab, ff));
            // per relation and per rule, all halves (for the report)
            std::map<std::string, std::array<int, 4>> rel;  // rel -> gib, sens, gib flagged, sens flagged
            for (const auto& k : keys) {
                const int y = label_of(lab, k.n);
                if (y < 0) continue;
                const bool fl = ff.flags(k.row) != 0;
                auto& a = rel[k.rel];
                ++a[y == 1 ? 0 : 1];
                if (fl) ++a[y == 1 ? 2 : 3];
            }
            std::printf("by relation (all labelled): gibberish, sensible, gibberish flagged, sensible flagged\n");
            for (const auto& [r, a] : rel) std::printf("  %-13s %3d %3d %3d %3d\n", r.c_str(), a[0], a[1], a[2], a[3]);
            return 0;
        }
        // choose: each family alone at each threshold
        struct Opt {
            const char* family;
            pl::FilterParams p;
        };
        pl::FilterParams none;
        none.shape = none.self = none.year = none.place = none.person = none.name = none.lex = none.taxon = none.agree = none.xrel = none.spread = 0;
        none.min_p = 0;
        std::vector<std::vector<Opt>> fams;
        auto one = [&](const char* fam, auto set) {
            pl::FilterParams p = none;
            set(p);
            return Opt{fam, p};
        };
        fams.push_back({one("shape", [](pl::FilterParams& p) { p.shape = 1; })});
        fams.push_back({one("self", [](pl::FilterParams& p) { p.self = 1; })});
        fams.push_back({one("year", [](pl::FilterParams& p) { p.year = 1; })});
        {
            std::vector<Opt> v;
            for (const int s : {1, 2, 3, 5}) v.push_back(one("place", [s](pl::FilterParams& p) { p.place = 1, p.place_support = s; }));
            fams.push_back(v);
        }
        fams.push_back({one("person", [](pl::FilterParams& p) { p.person = 1; }), one("person", [](pl::FilterParams& p) { p.person = 2; })});
        fams.push_back({one("name", [](pl::FilterParams& p) { p.name = 1; })});
        {
            std::vector<Opt> v;
            for (const int s : {1, 2, 3, 5, 10}) v.push_back(one("lex", [s](pl::FilterParams& p) { p.lex = 1, p.lex_support = s; }));
            fams.push_back(v);
        }
        fams.push_back({one("taxon", [](pl::FilterParams& p) { p.taxon = 1; })});
        fams.push_back({one("agree", [](pl::FilterParams& p) { p.agree = 1; })});
        fams.push_back({one("xrel", [](pl::FilterParams& p) { p.xrel = 1; })});
        {
            std::vector<Opt> v;
            for (const int s : {2, 3, 5, 10}) v.push_back(one("spread", [s](pl::FilterParams& p) { p.spread = 1, p.spread_min = s; }));
            fams.push_back(v);
        }
        {
            std::vector<Opt> v;
            for (const double m : {0.6, 0.8, 0.9}) v.push_back(one("p", [m](pl::FilterParams& p) { p.min_p = m; }));
            fams.push_back(v);
        }
        pl::FilterParams chosen = none;
        for (const auto& fam : fams) {
            int best_net = 0;
            const Opt* best = nullptr;
            for (const auto& o : fam) {
                pl::FactFilter ff(o.p, st);
                ff.prepare(L.rows);
                const Counts c = count(keys, "choose", lab, ff);
                const int net = c.tp - c.fp;
                const auto& q = o.p;
                const double th = q.place ? q.place_support : q.lex ? q.lex_support : q.spread ? q.spread_min : q.person ? q.person : q.min_p;
                std::printf("  %-6s %-4g -> gibberish flagged %d, sensible flagged %d, net %d\n", o.family, th, c.tp, c.fp, net);
                if (net > best_net) best_net = net, best = &o;
            }
            if (!best) continue;
            const auto& q = best->p;
            if (q.shape) chosen.shape = 1;
            if (q.self) chosen.self = 1;
            if (q.year) chosen.year = 1;
            if (q.place) chosen.place = 1, chosen.place_support = q.place_support;
            if (q.person) chosen.person = q.person;
            if (q.name) chosen.name = 1;
            if (q.lex) chosen.lex = 1, chosen.lex_support = q.lex_support;
            if (q.taxon) chosen.taxon = 1;
            if (q.agree) chosen.agree = 1;
            if (q.xrel) chosen.xrel = 1;
            if (q.spread) chosen.spread = 1, chosen.spread_min = q.spread_min;
            if (q.min_p > 0) chosen.min_p = q.min_p;
        }
        std::ofstream(argv[4]) << chosen.str();
        std::printf("chosen:\n%s", chosen.str().c_str());
        pl::FactFilter ff(chosen, st);
        ff.prepare(L.rows);
        report("choose half, the chosen filter", count(keys, "choose", lab, ff));
        return 0;
    }
    std::cerr << "unknown mode " << mode << "\n";
    return 2;
}
