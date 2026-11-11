// Why do dev prompts get no main-entity facts? (docs/reports/PLAN_FACTS.md, GENERATION_PLAN §13.42 Q1)
//
//   fs_diag STOREDIR SKELBANK PROMPTS.tsv DUMP OUT.tsv [--plan-drop LO:HI]... [--find v1|v2]
//
// For every prompt: the planner's main-entity rule (make_plan's: the content plan's candidates in order, the first
// that is a store subject with a fact of a bank relation and at most 2 values), run on the store loaded with the
// drops (the planner's view) and without them (to see what the drops took). For diagnosis only, the prompt's source
// page is found in DUMP (a 40-byte run of the prompt's plain words, searched in the drop range) and its title is
// checked the same way. Nothing here chooses anything for the planner.
// Reasons for a prompt with no main path:
//   none      no candidate is a store subject, even before the drops
//   dropped   a candidate is a subject only before the drops (its facts lie in the prompts' own bytes)
//   filtered  a candidate has facts after the drops, but none of a bank relation with at most 2 values
// Build: g++ -std=c++20 -O2 -I native/include scripts/fs_diag.cpp -o fs_diag
// Author: Odin Loch
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "cypha/lm/planner/facts.hpp"
#include "cypha/lm/planner/skeleton.hpp"

namespace pl = cypha::cyphalm::planner;
using CPS = cypha::cyphalm::ContentPlanStore;

namespace {
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
struct Info {
    std::size_t all = 0, kept = 0, plannable = 0;
};
Info info(const pl::FactStore& fs, const pl::SkelBank& bank, std::uint32_t e) {
    Info r;
    if (e == ~0u) return r;
    std::map<std::uint32_t, std::set<std::uint32_t>> by;
    for (const auto fi : fs.of_subj(e)) {
        ++r.kept;
        const auto& f = fs.fact(fi);
        if (bank.of(fs.rel(f.rel))) by[f.rel].insert(f.val);
    }
    for (const auto& [rr, v] : by) r.plannable += v.size() <= 2;
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: fs_diag STOREDIR SKELBANK PROMPTS.tsv DUMP OUT.tsv [--plan-drop LO:HI]...\n";
        return 2;
    }
    std::vector<std::pair<std::size_t, std::size_t>> drops = {{7'340'032, 8'388'608}};
    for (int i = 6; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--plan-drop") {
            const auto c = v.find(':');
            drops.push_back({std::strtoull(v.substr(0, c).c_str(), nullptr, 10), std::strtoull(v.substr(c + 1).c_str(), nullptr, 10)});
        }
    }
    std::string err;
    pl::FactStore fs, fs_all;
    if (!fs.load(argv[1], drops, err) || !fs_all.load(argv[1], {}, err)) {
        std::cerr << err << "\n";
        return 1;
    }
    pl::SkelBank bank;
    if (!bank.load(argv[2], err)) {
        std::cerr << err << "\n";
        return 1;
    }
    std::string dump;
    {
        std::ifstream in(argv[4], std::ios::binary);
        dump.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    std::size_t lo = 75'497'472, hi = std::min<std::size_t>(100'000'000, dump.size());
    std::ifstream in(argv[3]);
    std::ofstream out(argv[5]);
    out << "id\ttitle\ttitle_in_prompt\ttitle_all\ttitle_kept\ttitle_plannable\tpath\treason\tentity\tn_cands\tcands_subject_all\tcands_subject_kept\tcands\n";
    std::string line;
    std::getline(in, line);
    std::map<std::string, int> reasons;
    int n = 0, main_n = 0, found_title = 0, t_in = 0, t_all = 0, t_kept = 0, t_plan = 0, t_plan_in = 0;
    while (std::getline(in, line)) {
        const auto f = pl::split_tab(line);
        if (f.size() < 4) continue;
        const std::string p = unescape_tsv(f[2]);
        ++n;
        // the source page (diagnosis only): the longest run of plain words, 40 bytes of it, in the drop range
        std::string title;
        {
            std::size_t best_a = 0, best_n = 0;
            for (std::size_t a = 0; a < p.size();) {
                std::size_t b = a;
                while (b < p.size() && (std::isalnum(static_cast<unsigned char>(p[b])) || p[b] == ' ')) ++b;
                if (b - a > best_n) best_a = a, best_n = b - a;
                a = b + 1;
            }
            if (best_n >= 16) {
                const std::string key = p.substr(best_a, std::min<std::size_t>(best_n, 40));
                const std::size_t at = std::string_view(dump).substr(0, hi).find(key, lo);
                if (at != std::string::npos) {
                    const std::size_t t = dump.rfind("<title>", at);
                    if (t != std::string::npos) title = dump.substr(t + 7, dump.find("</title>", t) - t - 7);
                }
            }
        }
        found_title += !title.empty();
        const auto cands = CPS::candidates(p);
        std::string path = "none", reason, ent = "-", clist;
        int subj_all = 0, subj_kept = 0;
        for (const auto& c : cands) {
            const Info ka = info(fs_all, bank, fs_all.find(c));
            const Info kk = info(fs, bank, fs.find(c));
            subj_all += ka.kept > 0;
            subj_kept += kk.kept > 0;
            clist += (clist.empty() ? "" : "|") + c + "=" + std::to_string(ka.kept) + "/" + std::to_string(kk.kept) + "/" + std::to_string(kk.plannable);
            if (path == "none" && kk.plannable > 0) path = "main", ent = c;
        }
        if (path == "none") {
            int r = 0;  // 0 none, 1 dropped, 2 filtered
            for (const auto& c : cands) {
                const Info ka = info(fs_all, bank, fs_all.find(c));
                const Info kk = info(fs, bank, fs.find(c));
                if (kk.kept > 0) r = std::max(r, 2);
                else if (ka.kept > 0) r = std::max(r, 1);
            }
            reason = r == 2 ? "filtered" : r == 1 ? "dropped" : "none";
            ++reasons[reason];
        } else {
            ++main_n;
        }
        // the title
        std::string tl = pl::lower_s(title), pl_ = pl::lower_s(p);
        std::string tbase = tl.substr(0, tl.find(" ("));
        const bool tin = !title.empty() && pl_.find(tbase) != std::string::npos;
        const Info ta = info(fs_all, bank, title.empty() ? ~0u : fs_all.find(title));
        const Info tk = info(fs, bank, title.empty() ? ~0u : fs.find(title));
        t_in += tin, t_all += ta.kept > 0, t_kept += tk.kept > 0, t_plan += tk.plannable > 0, t_plan_in += tk.plannable > 0 && tin;
        out << f[0] << '\t' << title << '\t' << tin << '\t' << ta.kept << '\t' << tk.kept << '\t' << tk.plannable << '\t' << path << '\t' << reason << '\t'
            << ent << '\t' << cands.size() << '\t' << subj_all << '\t' << subj_kept << '\t' << clist << '\n';
    }
    std::printf("store %s: %zu facts kept (%zu dropped), %zu subjects\n", argv[1], fs.size(), fs.dropped(), fs.subjects());
    std::printf("prompts %d, main path %d\n", n, main_n);
    for (const auto& [r, k] : reasons) std::printf("  no main path: %-9s %d\n", r.c_str(), k);
    std::printf("source title found %d; title named in prompt %d; title a subject before drops %d, after %d, plannable %d (named in prompt and plannable %d)\n",
                found_title, t_in, t_all, t_kept, t_plan, t_plan_in);
    return 0;
}
