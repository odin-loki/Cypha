// k5_dates: offline steps for the planner's life dates (docs/reports/HONEST_DATES2.md, H3).
//
//   k5_dates classes --k5 TEXT
//       K5's classes as the date rule reads them (planner/life_dates.hpp): each class's share of month names and of
//       numbers under P(w | class), its kind, and its most probable words.
//   k5_dates mine --k5 TEXT --store DIR --text CLEAN --lo A --hi B --out SKEL [--drop LO:HI]
//       year-only skeletons by distant supervision, as k5_planner mine finds skeletons, for birth_year and death_year
//       only: a real sentence of CLEAN's documents starting in [A, B) that holds a store subject and its birth or
//       death year gives a skeleton if no word outside the slots is a month name or a number other than a year.
//   k5_dates trim --k5 TEXT --skel SKEL --out SKEL2
//       skeletons derived from the bank SKEL by the trim rule (life_dates.hpp trim_dates), each with its parent's count;
//       variants already in the bank are left out.
//   k5_dates show --k5 TEXT --skel SKEL [--rel R] [--n N]
//       a bank's skeletons, each class code shown as its most probable word.
//   k5_dates selftest
//
// Author: Odin Loch

#include "cypha/lm/planner/life_dates.hpp"
#include "cypha/lm/planner/writer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace pl = cypha::cyphalm::planner;
namespace st = cypha::cyphalm::structure;
namespace wkn = cypha::cyphalm::wkn;

namespace {

struct Args {
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& d = "") const {
        const auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
};

bool load_k5(const std::string& path, wkn::K5& m) {
    std::string err;
    const wkn::Options bo;
    if (!m.load(path, 12, err, &bo)) {
        std::cerr << err << "\n";
        return false;
    }
    std::fprintf(stderr, "k5: vocab %zu, %zu tokens\n", m.vocab().size(), m.tokens());
    return true;
}

int cmd_classes(const Args& a) {
    wkn::K5 m;
    if (!load_k5(a.get("--k5"), m)) return 1;
    const pl::DateClasses dc(m);
    const auto& cls = m.classes().classes();
    const auto& pwc = m.classes().pwc();
    std::map<std::uint32_t, std::vector<std::pair<double, std::uint32_t>>> top;
    for (std::uint32_t w = 2; w < cls.size(); ++w) top[cls[w]].push_back({pwc[w], w});
    std::printf("class\tkind\tmonth_share\tnumber_share\ttop_words\n");
    for (auto& [c, v] : top) {
        std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        const int k = dc.kind(c);
        std::printf("%u\t%s\t%.3f\t%.3f\t", c, k == pl::kMonthClass ? "month" : k == pl::kNumberClass ? "number" : "-", dc.month_mass(c), dc.number_mass(c));
        for (std::size_t i = 0; i < v.size() && i < 10; ++i) std::printf("%s%s", i ? " " : "", m.vocab()[v[i].second].c_str());
        std::printf("\n");
    }
    return 0;
}

std::vector<std::pair<std::size_t, std::size_t>> drops_of(const std::string& s) {
    std::vector<std::pair<std::size_t, std::size_t>> d;
    if (s.empty()) return d;
    const auto c = s.find(':');
    d.push_back({std::strtoull(s.substr(0, c).c_str(), nullptr, 10), std::strtoull(s.substr(c + 1).c_str(), nullptr, 10)});
    return d;
}

std::string read_all(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int cmd_mine(const Args& a) {
    wkn::K5 m;
    if (!load_k5(a.get("--k5"), m)) return 1;
    pl::FactStore fs;
    std::string err;
    if (!fs.load(a.get("--store"), drops_of(a.get("--drop")), err)) return std::cerr << err << "\n", 1;
    std::vector<std::uint32_t> rels;
    for (const char* r : {"birth_year", "death_year"})
        if (fs.rel_id(r) != ~0u) rels.push_back(fs.rel_id(r));
    const std::string text = read_all(a.get("--text"));
    const std::size_t lo = std::strtoull(a.get("--lo", "0").c_str(), nullptr, 10), hi = std::strtoull(a.get("--hi", "0").c_str(), nullptr, 10);
    pl::SkelBank bank;
    std::size_t docs = 0, sents = 0, matched = 0, shaped = 0, year_only = 0;
    std::map<std::string, std::size_t> per_rel;
    pl::for_docs(text, lo, hi ? hi : text.size(), [&](const std::vector<st::Sent>& ss, std::size_t) {
        ++docs;
        for (const auto& s : ss) {
            ++sents;
            pl::Match mt;
            if (!pl::match_sentence(fs, s.tok, mt, &rels)) continue;
            ++matched;
            std::vector<std::uint16_t> codes;
            if (!pl::shape_of(m, s.tok, mt, codes)) continue;
            ++shaped;
            bool ok = true;
            for (std::size_t i = 0; i < s.tok.size() && ok; ++i) {
                if ((i >= mt.sa && i < mt.sb) || (i >= mt.oa && i < mt.ob)) continue;
                ok = !pl::month_word(s.tok[i]) && !(pl::has_digit(s.tok[i]) && !pl::year_word(s.tok[i]));
            }
            if (!ok) continue;
            ++year_only;
            const std::string& r = fs.rel(fs.fact(mt.fact).rel);
            ++per_rel[r];
            bank.add(r, codes);
        }
    });
    bank.finish();
    if (!bank.save(a.get("--out"))) return std::cerr << "cannot write " << a.get("--out") << "\n", 1;
    std::printf("mine (year-only): %zu docs, %zu sentences, %zu matched a birth or death year, %zu gave a skeleton, %zu year-only; %zu skeletons kept\n",
                docs, sents, matched, shaped, year_only, bank.skeletons());
    for (const auto& [r, n] : per_rel) std::printf("  %s %zu sentences\n", r.c_str(), n);
    return 0;
}

int cmd_trim(const Args& a) {
    wkn::K5 m;
    if (!load_k5(a.get("--k5"), m)) return 1;
    pl::SkelBank in, out;
    std::string err;
    if (!in.load(a.get("--skel"), err)) return std::cerr << err << "\n", 1;
    const pl::DateClasses dc(m);
    std::size_t parents = 0, made = 0;
    std::map<std::string, std::size_t> per_rel;
    for (const auto& [r, v] : in.all()) {
        std::set<std::vector<std::uint16_t>> have;
        for (const auto& s : v) have.insert(s.codes);
        for (const auto& s : v) {
            const auto vs = pl::trim_dates(dc, s.codes, pl::life_rel(r));
            parents += !vs.empty();
            for (const auto& c : vs) {
                if (have.count(c)) continue;  // already in the bank as mined
                for (std::uint32_t k = 0; k < s.count; ++k) out.add(r, c);  // the parent's count
                ++made, ++per_rel[r];
            }
        }
    }
    out.finish();
    if (!out.save(a.get("--out"))) return std::cerr << "cannot write " << a.get("--out") << "\n", 1;
    std::printf("trim: %zu of %zu skeletons have a date to trim; %zu variants not in the bank; %zu distinct kept\n", parents, in.skeletons(), made,
                out.skeletons());
    for (const auto& [r, n] : per_rel) std::printf("  %s %zu\n", r.c_str(), n);
    return 0;
}

int cmd_show(const Args& a) {
    wkn::K5 m;
    if (!load_k5(a.get("--k5"), m)) return 1;
    pl::SkelBank b;
    std::string err;
    if (!b.load(a.get("--skel"), err)) return std::cerr << err << "\n", 1;
    const auto& cls = m.classes().classes();
    const auto& pwc = m.classes().pwc();
    std::map<std::uint32_t, std::pair<double, std::uint32_t>> top;
    for (std::uint32_t w = 2; w < cls.size(); ++w)
        if (pwc[w] > top[cls[w]].first) top[cls[w]] = {pwc[w], w};
    const std::string want = a.get("--rel");
    const std::size_t n = std::strtoull(a.get("--n", "20").c_str(), nullptr, 10);
    for (const auto& [r, v] : b.all()) {
        if (!want.empty() && r != want) continue;
        for (std::size_t i = 0; i < v.size() && i < n; ++i) {
            std::printf("%s %u:", r.c_str(), v[i].count);
            for (const auto c : v[i].codes)
                std::printf(" %s", c == pl::kSlotS ? "[S]" : c == pl::kSlotO ? "[O]" : (std::to_string(c) + "<" + m.vocab()[top[c].second] + ">").c_str());
            std::printf("\n");
        }
    }
    return 0;
}

int cmd_selftest() {
    std::string err;
    if (!pl::life_selftest(err)) return std::cerr << "k5_dates selftest: " << err << "\n", 1;
    std::printf("k5_dates selftest: ok\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: k5_dates classes|mine|trim|show|selftest [--key value ...]\n";
        return 2;
    }
    const std::string cmd = argv[1];
    Args a;
    for (int i = 2; i + 1 < argc; i += 2) a.kv[argv[i]] = argv[i + 1];
    if (cmd == "classes") return cmd_classes(a);
    if (cmd == "mine") return cmd_mine(a);
    if (cmd == "trim") return cmd_trim(a);
    if (cmd == "show") return cmd_show(a);
    if (cmd == "selftest") return cmd_selftest();
    std::cerr << "unknown command " << cmd << "\n";
    return 2;
}
