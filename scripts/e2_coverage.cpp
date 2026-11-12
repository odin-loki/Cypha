// E2 against the planner's entity finder on GEN-A prompts (docs/reports/E2_ENTITIES.md). Offline: the planner is
// not changed; its headers are read only.
//
//   e2_coverage STOREDIR PROMPTS.tsv SKELBANK E2SPANS.tsv [--plan-drop LO:HI] [--stats S10.stats] [--list OUT.tsv]
//
// Loads the store as `sample --k5-plan` does (planner::FactStore; the holdout [7340032, 8388608) and --plan-drop are
// dropped) and, for every prompt, runs make_plan's main-entity
// rule (writer.hpp, as fs_coverage --planner: the candidates in order, the first whose store subject has a fact of a
// bank relation with at most 2 values) over several candidate lists:
//   find1     the planner's default finder (ContentPlanStore::candidates, the content plan's capitalised runs)
//   find2     Q1's finder (--k5-plan-find 2; needs --stats)
//   e2        the E2 spans (cypha_ner --prompt-out) of type PERSON, PLACE, ORG and MISC, in text order
//   e2ppo     the same without MISC
//   e2+find1  the E2 spans first, then find1's candidates not already listed (the registered hook)
//   ngram     every token run of the prompt that is a store subject, longest first, left to right: a ceiling for
//             any finder that maps prompt words to subjects (it is a lookup, not a finder; for scale only)
// For each list it prints, with Wilson 95% intervals: "any" (some candidate is a store subject with at least one
// fact) and "main" (a main-entity plan). The pairs against find1 are listed with a paired bootstrap interval.
// --list writes one row per prompt and list: id, list, first subject with a fact, main-plan subject.
//
// Build: g++ -std=c++20 -O2 -I native/include scripts/e2_coverage.cpp -o e2_coverage
// Author: Odin Loch
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "cypha/lm/content_plan.hpp"
#include "cypha/lm/planner/facts.hpp"
#include "cypha/lm/planner/skeleton.hpp"

namespace pln = cypha::cyphalm::planner;
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
void wilson(double c, double n, double& lo, double& hi) {
    const double z = 1.959964, p = c / n, d = 1 + z * z / n;
    const double r = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    lo = (p + z * z / (2 * n) - r) / d, hi = (p + z * z / (2 * n) + r) / d;
}

struct Result {
    std::string any = "-", main = "-";
};

// make_plan's main-entity rule over a candidate list (writer.hpp; fs_coverage --planner).
Result plan(const pln::FactStore& fs, const pln::SkelBank& bank, const std::vector<std::string>& cands) {
    Result r;
    for (const auto& c : cands) {
        const std::uint32_t e = fs.find_main(c);
        if (e == ~0u || fs.of_subj(e).empty()) continue;
        if (r.any == "-") r.any = fs.ent(e) + (c == fs.ent(e) ? "" : " <- " + c);
        std::map<std::uint32_t, std::vector<std::uint32_t>> by;
        for (const auto fi : fs.of_subj(e))
            if (bank.of(fs.rel(fs.fact(fi).rel))) by[fs.fact(fi).rel].push_back(fi);
        bool ok = false;
        for (const auto& [rel, v] : by) {
            std::set<std::uint32_t> vals;
            for (const auto fi : v) vals.insert(fs.fact(fi).val);
            ok = ok || vals.size() <= 2;
        }
        if (!ok) continue;
        r.main = fs.ent(e) + (c == fs.ent(e) ? "" : " <- " + c);
        break;
    }
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: e2_coverage STOREDIR PROMPTS.tsv SKELBANK E2SPANS.tsv [--plan-drop LO:HI] [--stats FILE] [--list OUT]\n";
        return 2;
    }
    std::vector<std::pair<std::size_t, std::size_t>> drops = {{7340032, 8388608}};  // the holdout, as sample --k5-plan
    std::string stats_path, list;
    for (int i = 5; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--plan-drop") {
            const auto c = v.find(':');
            drops.push_back({std::strtoull(v.substr(0, c).c_str(), nullptr, 10), std::strtoull(v.substr(c + 1).c_str(), nullptr, 10)});
        } else if (a == "--stats") stats_path = v;
        else if (a == "--list") list = v;
    }
    std::string err;
    pln::SkelBank bank;
    if (!bank.load(argv[3], err)) return std::cerr << err << "\n", 1;
    pln::fact_options().find = 1;
    pln::FactStore fs1;
    if (!fs1.load(argv[1], drops, err)) return std::cerr << err << "\n", 1;
    cypha::cyphalm::structure::Stats stats;
    pln::FactStore fs2;
    const bool f2 = !stats_path.empty();
    if (f2) {
        if (!stats.load(stats_path, err)) return std::cerr << err << "\n", 1;
        pln::fact_options().find = 2;
        if (!fs2.load(argv[1], drops, err)) return std::cerr << err << "\n", 1;
        pln::fact_options().find = 1;
    }
    // E2 spans by prompt id, in text order
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> e2;  // id -> (type, surface)
    {
        std::ifstream in(argv[4]);
        std::string line;
        std::getline(in, line);
        while (std::getline(in, line)) {
            const auto f = split(line, '\t');
            if (f.size() >= 5) e2[f[0]].push_back({f[3], f[4]});
        }
    }
    const std::vector<std::string> names = {"find1", "find2", "e2", "e2ppo", "e2+find1", "ngram"};
    std::map<std::string, std::vector<Result>> res;
    std::vector<std::string> ids;
    std::ifstream in(argv[2]);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        const auto f = split(line, '\t');
        if (f.size() < 4) continue;
        const std::string p = unescape_tsv(f[2]);
        ids.push_back(f[0]);
        const auto c1 = ContentPlanStore::candidates(p);
        std::vector<std::string> ce, cp;
        for (const auto& [ty, s] : e2[f[0]]) {
            if (ty == "DATE" || ty == "NUMBER") continue;
            if (std::find(ce.begin(), ce.end(), s) == ce.end()) ce.push_back(s);
            if (ty != "MISC" && std::find(cp.begin(), cp.end(), s) == cp.end()) cp.push_back(s);
        }
        std::vector<std::string> cu = ce;
        for (const auto& c : c1)
            if (std::find(cu.begin(), cu.end(), c) == cu.end()) cu.push_back(c);
        res["find1"].push_back(plan(fs1, bank, c1));
        res["find2"].push_back(f2 ? plan(fs2, bank, fs2.candidates(p, &stats)) : Result{});
        res["e2"].push_back(plan(fs1, bank, ce));
        res["e2ppo"].push_back(plan(fs1, bank, cp));
        res["e2+find1"].push_back(plan(fs1, bank, cu));
        std::vector<std::string> cn;
        {
            const auto t = cypha::cyphalm::wkn::tokens(ContentPlanStore::strip(p));
            std::vector<std::pair<std::uint32_t, std::size_t>> m;
            for (std::size_t a = 0; a < t.size(); ++a) {
                fs1.match_at(t, a, m);
                for (const auto& [e, len] : m)
                    if (!fs1.of_subj(e).empty() && std::find(cn.begin(), cn.end(), fs1.ent(e)) == cn.end()) cn.push_back(fs1.ent(e));
            }
        }
        res["ngram"].push_back(plan(fs1, bank, cn));
    }
    const double n = static_cast<double>(ids.size());
    std::printf("store %s: %zu facts (%zu dropped), %zu subjects; %zu prompts; E2 spans for %zu prompts\n", argv[1], fs1.size(), fs1.dropped(), fs1.subjects(),
                ids.size(), e2.size());
    for (const auto& nm : names) {
        if (nm == "find2" && !f2) continue;
        int a = 0, m = 0;
        for (const auto& r : res[nm]) a += r.any != "-", m += r.main != "-";
        double al, ah, ml, mh;
        wilson(a, n, al, ah);
        wilson(m, n, ml, mh);
        std::printf("%-9s any %3d = %.3f [%.3f, %.3f]   main %3d = %.3f [%.3f, %.3f]\n", nm.c_str(), a, a / n, al, ah, m, m / n, ml, mh);
    }
    // paired against find1: discordant counts and a bootstrap interval of the difference in main-plan prompts
    std::mt19937_64 rng(20261112);
    std::uniform_int_distribution<std::size_t> U(0, ids.size() - 1);
    for (const auto& nm : names) {
        if (nm == "find1" || (nm == "find2" && !f2)) continue;
        for (const char* what : {"any", "main"}) {
            const bool is_any = what[0] == 'a';
            int only_new = 0, only_old = 0, both_diff = 0;
            std::vector<int> d(ids.size());
            for (std::size_t i = 0; i < ids.size(); ++i) {
                const auto& x = res[nm][i];
                const auto& y = res["find1"][i];
                const bool hx = (is_any ? x.any : x.main) != "-", hy = (is_any ? y.any : y.main) != "-";
                only_new += hx && !hy, only_old += hy && !hx;
                both_diff += hx && hy && (is_any ? x.any : x.main) != (is_any ? y.any : y.main);
                d[i] = static_cast<int>(hx) - static_cast<int>(hy);
            }
            std::vector<double> bs;
            for (int b = 0; b < 4000; ++b) {
                double s = 0;
                for (std::size_t i = 0; i < ids.size(); ++i) s += d[U(rng)];
                bs.push_back(s);
            }
            std::sort(bs.begin(), bs.end());
            std::printf("%-9s vs find1 (%s): %+d prompts [%+.0f, %+.0f]; only %s %d, only find1 %d, both but a different subject %d\n", nm.c_str(), what,
                        only_new - only_old, bs[100], bs[3899], nm.c_str(), only_new, only_old, both_diff);
        }
    }
    if (!list.empty()) {
        std::ofstream out(list);
        out << "id\tlist\tfirst_with_fact\tmain\n";
        for (std::size_t i = 0; i < ids.size(); ++i)
            for (const auto& nm : names)
                if (nm != "find2" || f2) out << ids[i] << '\t' << nm << '\t' << res[nm][i].any << '\t' << res[nm][i].main << '\n';
    }
    return 0;
}
