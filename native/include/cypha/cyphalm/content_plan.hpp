#pragma once
// GEN-E (docs/reports/GEN_E_CONTENT_PLAN.md): the content-plan expert, the hybrid step of the generation track.
//
// Cypha's fact store picks what to say about the prompt's entity; this expert raises the odds of the bytes that express
// a planned fact, through F2's learned patterns reversed (frames filled with the subject and the value). It is a
// next-byte expert like the copy expert: it never writes text itself (PLAN rule 10), and cyphalm_trie mixes it in
// `sample` only, with lambdas fit on held-out text.
//
// Store: pattern_learn's v2 files (seeds.tsv, facts_f2.tsv). Frames: given by the caller (cyphalm_trie's A3 load_frames),
// relation -> frames with \x01 for the subject and \x02 for the value.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cypha::cyphalm {

class ContentPlanStore {
public:
    struct Fact {
        std::string rel, value;
        bool text = false;  // read from the text (F2, infobox, table, category); else Wikidata (the data pack)
    };
    static constexpr int kMaxFacts = 6;      // facts in one plan
    static constexpr int kMaxValues = 2;     // a relation with more distinct values for the entity is not planned
    static constexpr int kBuckets = 8;       // the copy expert's buckets, by alignment length
    static constexpr int kMinAlign = 3;
    static constexpr int kMaxAlign = 64;

    static std::string lower(const std::string& s) {
        std::string o = s;
        for (char& c : o) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return o;
    }
    // ISO dates and zero-padded years: the year, leading zeros dropped ("" if there is none).
    static std::string year_of(const std::string& v) {
        std::size_t i = 0;
        bool neg = false;
        if (!v.empty() && v[0] == '-') neg = true, i = 1;
        std::size_t e = i;
        while (e < v.size() && std::isdigit(static_cast<unsigned char>(v[e]))) ++e;
        if (e == i || e - i > 4) return {};
        std::string y = v.substr(i, e - i);
        while (y.size() > 1 && y[0] == '0') y.erase(0, 1);
        return neg ? y + " BC" : y;
    }
    // The relation and value a fact is planned under: dates become their year relation.
    static std::pair<std::string, std::string> canon(const std::string& rel, const std::string& value) {
        if (rel == "birth_date" || rel == "death_date") {
            const std::string y = year_of(value);
            return {y.empty() ? std::string() : rel.substr(0, 6) + "year", y};  // birth_date -> birth_year
        }
        if (rel == "birth_year" || rel == "death_year" || rel == "founded") {
            const std::string y = year_of(value);
            return {rel, y.empty() ? value : y};
        }
        return {rel, value};
    }

    // Reads seeds.tsv and facts_f2.tsv under dir. Facts whose evidence offset lies in [lo, hi) or in [lo2, hi2) (the
    // bytes the prompts come from) are dropped. Relation order from facts_f2's offsets inside their pages (pages.tsv);
    // then the other relations by their number of seeds.
    bool load(const std::string& dir, std::size_t lo, std::size_t hi, std::string& err, std::size_t lo2 = 0, std::size_t hi2 = 0) {
        auto split = [](const std::string& s) {
            std::vector<std::string> f;
            std::size_t a = 0;
            for (;;) {
                const std::size_t b = s.find('\t', a);
                f.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
                if (b == std::string::npos) return f;
                a = b + 1;
            }
        };
        std::map<std::string, std::size_t> page_at;
        {
            std::ifstream in(dir + "/pages.tsv");
            std::string line;
            while (std::getline(in, line)) {
                const auto f = split(line);
                if (f.size() >= 2) page_at[f[0]] = static_cast<std::size_t>(std::strtoull(f[1].c_str(), nullptr, 10));
            }
        }
        std::map<std::string, std::vector<double>> pos;
        std::map<std::string, std::size_t> nseed;
        auto add = [&](const std::string& s, const std::string& r, const std::string& o, bool text, std::size_t off) {
            if (s.empty() || r.empty() || o.empty() || (off >= lo && off < hi) || (off >= lo2 && off < hi2)) return;
            const auto [cr, cv] = canon(r, o);
            if (cr.empty() || cv.empty()) return;
            facts_[lower(s)].push_back({cr, cv, text});
            if (title_.find(lower(s)) == title_.end()) title_[lower(s)] = s;
        };
        for (const char* name : {"/facts_f2.tsv", "/seeds.tsv"}) {
            const bool f2 = name[1] == 'f';
            std::ifstream in(dir + name);
            if (!in) {
                err = "content plan: cannot read " + dir + name;
                return false;
            }
            std::string line;
            bool header = true;
            while (std::getline(in, line)) {
                if (header) {
                    header = false;
                    continue;
                }
                const auto f = split(line);
                if (f2 ? f.size() < 9 : f.size() < 6) continue;
                const std::size_t off = static_cast<std::size_t>(std::strtoull((f2 ? f[8] : f[5]).c_str(), nullptr, 10));
                const bool text = f2 || f[3] != "wikidata";
                add(f[0], f[1], f[2], text, off);
                if (f2) {
                    const auto p = page_at.find(f[7]);
                    if (p != page_at.end() && off >= p->second) pos[canon(f[1], f[2]).first].push_back(static_cast<double>(off - p->second));
                } else {
                    ++nseed[canon(f[1], f[2]).first];
                }
            }
        }
        std::vector<std::pair<double, std::string>> a, b;
        for (auto& [r, v] : pos) {
            std::sort(v.begin(), v.end());
            a.push_back({v[v.size() / 2], r});
        }
        for (auto& [r, n] : nseed)
            if (!pos.count(r)) b.push_back({-static_cast<double>(n), r});
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        int k = 0;
        for (const auto* v : {&a, &b})
            for (const auto& x : *v) order_[x.second] = k++;
        return true;
    }

    // The entity's candidates in a prompt, in the registered order: the bold lead, link targets, capitalised runs.
    static std::vector<std::string> candidates(const std::string& p) {
        std::vector<std::string> out;
        if (p.compare(0, 3, "'''") == 0) {
            const std::size_t e = p.find("'''", 3);
            if (e != std::string::npos) out.push_back(strip(p.substr(3, e - 3)));
        }
        for (std::size_t i = p.find("[["); i != std::string::npos; i = p.find("[[", i + 2)) {
            const std::size_t e = p.find_first_of("]|#", i + 2);
            if (e != std::string::npos && e > i + 2) out.push_back(p.substr(i + 2, e - i - 2));
        }
        std::vector<std::string> w;
        {
            const std::string t = strip(p);
            std::string cur;
            for (char c : t) {
                const unsigned char u = static_cast<unsigned char>(c);
                if (std::isalnum(u) || c == '\'' || c == '-' || c == '.' || u >= 0x80) cur += c;
                else {
                    if (!cur.empty()) w.push_back(cur);
                    cur.clear();
                    if (c != ' ') w.push_back(std::string(1, c));
                }
            }
            if (!cur.empty()) w.push_back(cur);
            w.push_back(".");
        }
        std::vector<std::string> run;
        auto flush = [&]() {
            if (!run.empty() && run.back() == "of") run.pop_back();
            if (!run.empty()) {
                std::string s;
                for (const auto& x : run) s += (s.empty() ? "" : " ") + x;
                if (s != "The") out.push_back(s);
                if (run[0] == "The" && run.size() > 1) out.push_back(s.substr(4));
            }
            run.clear();
        };
        for (const auto& x : w) {
            const bool cap = !x.empty() && std::isupper(static_cast<unsigned char>(x[0]));
            if (cap || (x == "of" && !run.empty())) {
                std::string y = x;
                while (y.size() > 1 && y.back() == '.') y.pop_back();
                run.push_back(y);
                if (x.size() > 1 && x.back() == '.') flush();
            } else flush();
        }
        return out;
    }
    // Markup dropped: [[ ]] ''' '' gone, a link's target dropped for its label.
    static std::string strip(const std::string& s) {
        std::string t;
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s.compare(i, 3, "'''") == 0) i += 2;
            else if (s.compare(i, 2, "''") == 0 || s.compare(i, 2, "]]") == 0) i += 1;
            else if (s.compare(i, 2, "[[") == 0) {
                const std::size_t e = s.find("]]", i + 2), bar = s.find('|', i + 2);
                if (e != std::string::npos && bar != std::string::npos && bar < e) i = bar;  // [[target|label]]: the label
                else i += 1;
            } else t += s[i];
        }
        return t;
    }

    struct Plan {
        std::string entity;                    // the store's subject title ("" if none)
        std::vector<Fact> facts;               // in plan order
        std::vector<std::vector<std::string>> real;  // each fact's realisations
    };
    // The prompt's plan: the first candidate with a plannable fact. frames: relation -> frames (\x01 subject, \x02 value).
    Plan plan(const std::string& prompt, const std::map<std::string, std::vector<std::string>>& frames) const {
        Plan pl;
        for (const auto& c : candidates(prompt)) {
            const auto it = facts_.find(lower(c));
            if (it == facts_.end()) continue;
            std::map<std::string, std::vector<const Fact*>> by;
            for (const auto& f : it->second) by[f.rel].push_back(&f);
            std::vector<std::pair<int, const Fact*>> pick;
            for (const auto& [r, v] : by) {
                const auto fr = frames.find(r);
                if (fr == frames.end() || fr->second.empty()) continue;
                std::set<std::string> distinct;
                for (const Fact* f : v) distinct.insert(lower(f->value));
                if (static_cast<int>(distinct.size()) > kMaxValues) continue;
                const Fact* best = v[0];
                for (const Fact* f : v)
                    if (f->text && !best->text) {
                        best = f;
                        break;
                    }
                const auto o = order_.find(r);
                pick.push_back({o == order_.end() ? 1 << 20 : o->second, best});
            }
            if (pick.empty()) continue;
            std::stable_sort(pick.begin(), pick.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            pl.entity = title_.at(lower(c));
            for (const auto& [o, f] : pick) {
                if (static_cast<int>(pl.facts.size()) >= kMaxFacts) break;
                pl.facts.push_back(*f);
                std::vector<std::string> rs;
                for (std::string fr : frames.at(f->rel)) {
                    for (std::size_t q; (q = fr.find('\x01')) != std::string::npos;) fr.replace(q, 1, pl.entity);
                    for (std::size_t q; (q = fr.find('\x02')) != std::string::npos;) fr.replace(q, 1, f->value);
                    if (std::find(rs.begin(), rs.end(), fr) == rs.end()) rs.push_back(fr);
                }
                pl.real.push_back(rs);
            }
            return pl;
        }
        return pl;
    }
    std::size_t subjects() const { return facts_.size(); }
    const std::map<std::string, int>& order() const { return order_; }

    static int bucket(int L) { return L == 0 ? 0 : L < 4 ? 1 : L < 6 ? 2 : L < 8 ? 3 : L < 12 ? 4 : L < 16 ? 5 : L < 32 ? 6 : 7; }

    // Text folded for the "expressed" test: lowercased, [ ] and ' dropped (so "[[Mycenae]]" holds "mycenae").
    static void fold_into(std::string& out, unsigned char c) {
        if (c == '[' || c == ']' || c == '\'') return;
        out += static_cast<char>(std::tolower(c));
    }
    static std::string fold(const std::string& s) {
        std::string o;
        for (unsigned char c : s) fold_into(o, c);
        return o;
    }
    // Whether fact k of the plan is expressed in the generated text, given as fold()ed text.
    static bool expressed(const Plan& pl, std::size_t k, const std::string& gen_folded) {
        const std::string v = fold(pl.facts[k].value);
        return !v.empty() && gen_folded.find(v) != std::string::npos;
    }

    // The expert's prediction after s[0, n) with the facts in `active` (plan indices, plan order). P gets the
    // distribution (zeros when there is no alignment). Returns the alignment length L (0: none).
    static int predict(const Plan& pl, const std::vector<int>& active, const std::uint8_t* s, std::size_t n, double* P) {
        std::fill(P, P + 256, 0.0);
        int bestL = 0, bestk = -1;
        std::vector<int> hits;  // predicted bytes at (bestL, bestk)
        for (int k : active) {
            if (k < 0 || static_cast<std::size_t>(k) >= pl.real.size()) continue;
            for (const auto& r : pl.real[static_cast<std::size_t>(k)]) {
                const auto* rb = reinterpret_cast<const std::uint8_t*>(r.data());
                for (std::size_t j = 1; j < r.size(); ++j) {
                    int L = 0;
                    while (L < kMaxAlign && static_cast<std::size_t>(L) < j && static_cast<std::size_t>(L) < n &&
                           rb[j - 1 - L] == s[n - 1 - L])
                        ++L;
                    if (L < kMinAlign) continue;
                    if (L > bestL || (L == bestL && k == bestk)) {
                        if (L > bestL) hits.clear();
                        bestL = L, bestk = k;
                        hits.push_back(rb[j]);
                    }
                }
            }
            // ties go to the earlier fact: a later fact replaces only with a longer alignment (handled by L > bestL)
        }
        if (bestL == 0) return 0;
        for (int b : hits) P[b] += 1.0 / static_cast<double>(hits.size());
        return bestL;
    }

private:
    std::unordered_map<std::string, std::vector<Fact>> facts_;  // lowercased subject -> facts
    std::unordered_map<std::string, std::string> title_;
    std::map<std::string, int> order_;
};

}  // namespace cypha::cyphalm
