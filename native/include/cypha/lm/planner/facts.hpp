#pragma once

/// The planner's fact store (docs/reports/K5_PLANNER.md, GENERATION_PLAN §13.37-13.38): CyphaLM's own facts from
/// pattern_learn (F2, `seeds.tsv` and `facts_f2.tsv`), held as integer IDs. A fact is (subject, relation, value), each
/// an interned ID; the subject and value also carry their K5 tokens, so every later step searches on IDs only.
///
/// The store directory is a plain argument (verify_f2r_aug today; the enwik9 store can be swapped in). Facts whose
/// evidence offset lies in a dropped range (the prompts' own bytes, `--plan-drop LO:HI`) are not loaded.
///
/// PLAN_FACTS (docs/reports/PLAN_FACTS.md, §13.42 Q1), behind flags (FactOptions): aliases for the main entity, the
/// prompt's lowercase subjects, and the gibberish filter (fact_filter.hpp).
///
/// Author: Odin Loch

#include "cypha/lm/content_plan.hpp"
#include "cypha/lm/planner/fact_filter.hpp"
#include "cypha/lm/structure/experts.hpp"  // read only: the capitalisation statistics (candidates, find 2)
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <tuple>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cypha::cyphalm::planner {

namespace wkn = cypha::cyphalm::wkn;
using CPS = cypha::cyphalm::ContentPlanStore;

inline std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> f;
    std::size_t a = 0;
    for (;;) {
        const std::size_t b = s.find('\t', a);
        f.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) return f;
        a = b + 1;
    }
}
inline std::string lower_s(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}
// A token sequence's key (lowercased tokens), for the subject and value indexes.
inline std::uint64_t seq_key(const std::vector<std::string>& t, std::size_t a, std::size_t n) {
    std::uint64_t h = wkn::kFnv0;
    for (std::size_t i = a; i < a + n; ++i) {
        for (const char c : t[i]) h = wkn::fnv_step(h, static_cast<unsigned char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
        h = wkn::fnv_step(h, 0x1F);
    }
    return wkn::mix64(h ^ n);
}

// PLAN_FACTS (docs/reports/PLAN_FACTS.md, GENERATION_PLAN §13.42 Q1): the store's options, set from cyphalm_trie's flags
// before the store loads (apply_plan_flags builds the store). The defaults keep everything as before.
//   find 2 (--k5-plan-find 2): the main entity is also found through the store's aliases.tsv (redirects and link
//     anchors, scripts/fs_aliases.cpp) and without a possessive or plural ending, and the prompt's lowercase
//     multi-word store subjects follow the content plan's candidates (candidates(), find_main()).
//   filter (--k5-plan-filter PARAMS|on): facts that fact_filter.hpp's rules call gibberish are not loaded.
//   fallback "pattern" (--k5-plan-fallback pattern): fallback.hpp's typed fallback before the co-occurring one.
struct FactOptions {
    int find = 1;
    std::string filter;
    std::string fallback = "cooc";
};
inline FactOptions& fact_options() {
    static FactOptions o;
    return o;
}

class FactStore {
 public:
    struct Fact {
        std::uint32_t subj = 0, rel = 0, val = 0;
        bool text = false;           // read from the text (F2, infobox, table, category); else a Wikidata seed
        std::int8_t wd = -1;         // facts_wd_check: 1 matches Wikidata, 0 differs, -1 unchecked
    };
    static constexpr std::size_t kMaxTok = 6;  // subjects and values longer than this many tokens are not planned

    // PLAN_FACTS: the store's rows as load() holds them, in load()'s order (facts_f2.tsv, then seeds.tsv): after the
    // drops, canonical, markup stripped, at most kMaxTok tokens each side, one row per exact (subject, relation, value)
    // (n_rows: the raw rows behind it). dropped: rows in a drop range.
    static bool read_rows(const std::string& dir, const std::vector<std::pair<std::size_t, std::size_t>>& drops, std::vector<FactRow>& rows,
                          std::size_t& dropped, std::string& err) {
        rows.clear();
        dropped = 0;
        std::map<std::tuple<std::string, std::string, std::string>, std::size_t> at;
        for (const char* name : {"/facts_f2.tsv", "/seeds.tsv"}) {
            const bool f2 = name[1] == 'f';
            std::ifstream in(dir + name);
            if (!in) {
                if (f2) continue;
                err = "planner: cannot read " + dir + name;
                return false;
            }
            std::string line;
            bool header = true;
            while (std::getline(in, line)) {
                if (header) {
                    header = false;
                    continue;
                }
                const auto f = split_tab(line);
                if (f2 ? f.size() < 9 : f.size() < 6) continue;
                const std::size_t off = static_cast<std::size_t>(std::strtoull((f2 ? f[8] : f[5]).c_str(), nullptr, 10));
                bool drop = false;
                for (const auto& [lo, hi] : drops) drop = drop || (off >= lo && off < hi);
                if (drop) {
                    ++dropped;
                    continue;
                }
                const auto [cr, cv0] = CPS::canon(f[1], f[2]);
                const std::string cv = CPS::strip(cv0), subj = CPS::strip(f[0]);
                if (subj.empty() || cr.empty() || cv.empty()) continue;
                const auto st = wkn::tokens(subj), vt = wkn::tokens(cv);
                if (st.empty() || vt.empty() || st.size() > kMaxTok || vt.size() > kMaxTok) continue;
                const auto [it, fresh] = at.try_emplace({subj, cr, cv}, rows.size());
                if (!fresh) {
                    ++rows[it->second].n_rows;
                    continue;
                }
                FactRow r;
                r.subj = subj, r.rel = cr, r.val = cv, r.off = off;
                if (f2) {
                    r.src = "f2", r.p = std::atof(f[3].c_str()), r.n_ev = std::atoi(f[5].c_str()), r.pattern = f[6];
                } else {
                    r.src = f[3], r.tpl = f[4];
                }
                rows.push_back(std::move(r));
            }
        }
        return true;
    }

    // dir: seeds.tsv and facts_f2.tsv (pattern_learn's v2 files); drops: evidence ranges whose facts are left out.
    bool load(const std::string& dir, const std::vector<std::pair<std::size_t, std::size_t>>& drops, std::string& err) {
        opt_ = fact_options();
        if (opt_.find == 2 || !opt_.filter.empty()) return load_v2(dir, drops, err);
        dir_ = dir;
        std::map<std::tuple<std::string, std::string, std::string>, int> wd;
        {
            std::ifstream in(dir + "/facts_wd_check.tsv");
            std::string line;
            bool header = true;
            while (std::getline(in, line)) {
                if (header) {
                    header = false;
                    continue;
                }
                const auto f = split_tab(line);
                if (f.size() < 5) continue;
                const auto [cr, cv] = CPS::canon(f[1], f[2]);
                wd[{lower_s(f[0]), cr, lower_s(cv)}] = f[4] == "right" ? 1 : f[4] == "wrong" ? 0 : -1;
            }
        }
        std::set<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>> seen;
        for (const char* name : {"/facts_f2.tsv", "/seeds.tsv"}) {
            const bool f2 = name[1] == 'f';
            std::ifstream in(dir + name);
            if (!in) {
                if (f2) continue;  // a store may hold seeds only
                err = "planner: cannot read " + dir + name;
                return false;
            }
            std::string line;
            bool header = true;
            while (std::getline(in, line)) {
                if (header) {
                    header = false;
                    continue;
                }
                const auto f = split_tab(line);
                if (f2 ? f.size() < 9 : f.size() < 6) continue;
                const std::size_t off = static_cast<std::size_t>(std::strtoull((f2 ? f[8] : f[5]).c_str(), nullptr, 10));
                bool drop = false;
                for (const auto& [lo, hi] : drops) drop = drop || (off >= lo && off < hi);
                if (drop) {
                    ++dropped_;
                    continue;
                }
                const auto [cr, cv0] = CPS::canon(f[1], f[2]);
                const std::string cv = CPS::strip(cv0), subj = CPS::strip(f[0]);
                if (subj.empty() || cr.empty() || cv.empty()) continue;
                const auto st = wkn::tokens(subj), vt = wkn::tokens(cv);
                if (st.empty() || vt.empty() || st.size() > kMaxTok || vt.size() > kMaxTok) continue;
                Fact x;
                x.subj = intern(ent_, ent_id_, subj, &ent_tok_);
                x.rel = intern(rel_, rel_id_, cr, nullptr);
                x.val = intern(ent_, ent_id_, cv, &ent_tok_);
                if (!seen.insert({x.subj, x.rel, x.val}).second) continue;
                x.text = f2 || f[3] != "wikidata";
                const auto w = wd.find({lower_s(subj), cr, lower_s(cv)});
                if (w != wd.end()) x.wd = static_cast<std::int8_t>(w->second);
                by_subj_[x.subj].push_back(static_cast<std::uint32_t>(facts_.size()));
                by_val_[x.val].push_back(static_cast<std::uint32_t>(facts_.size()));
                facts_.push_back(x);
            }
        }
        for (std::uint32_t e = 0; e < ent_.size(); ++e) {
            const auto& t = ent_tok_[e];
            key_ent_[seq_key(t, 0, t.size())].push_back(e);
        }
        return true;
    }

    // PLAN_FACTS: load() with the options (find 2 or a filter). The rows of read_rows(), the filter's gibberish left out,
    // interned in the same order as load(); without a filter the facts equal load()'s. Then the aliases (find 2).
    bool load_v2(const std::string& dir, const std::vector<std::pair<std::size_t, std::size_t>>& drops, std::string& err) {
        dir_ = dir;
        std::map<std::tuple<std::string, std::string, std::string>, int> wd;
        {
            std::ifstream in(dir + "/facts_wd_check.tsv");
            std::string line;
            bool header = true;
            while (std::getline(in, line)) {
                if (header) {
                    header = false;
                    continue;
                }
                const auto f = split_tab(line);
                if (f.size() < 5) continue;
                const auto [cr, cv] = CPS::canon(f[1], f[2]);
                wd[{lower_s(f[0]), cr, lower_s(cv)}] = f[4] == "right" ? 1 : f[4] == "wrong" ? 0 : -1;
            }
        }
        std::vector<FactRow> rows;
        if (!read_rows(dir, drops, rows, dropped_, err)) return false;
        std::vector<char> keep(rows.size(), 1);
        if (!opt_.filter.empty()) {
            FilterParams fp;
            if (!fp.read(opt_.filter, err)) return false;
            FactFilter ff(fp, stats_);
            ff.prepare(rows);
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const std::uint32_t fl = ff.flags(i);
                if (!fl) continue;
                keep[i] = 0, ++filtered_;
                for (int b = 0; b < FactFilter::kNReason; ++b)
                    if (fl & (1u << b)) ++filter_reason_[FactFilter::reason_name(b)];
            }
        }
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (!keep[i]) continue;
            const auto& r = rows[i];
            Fact x;
            x.subj = intern(ent_, ent_id_, r.subj, &ent_tok_);
            x.rel = intern(rel_, rel_id_, r.rel, nullptr);
            x.val = intern(ent_, ent_id_, r.val, &ent_tok_);
            x.text = r.src != "wikidata";
            const auto w = wd.find({lower_s(r.subj), r.rel, lower_s(r.val)});
            if (w != wd.end()) x.wd = static_cast<std::int8_t>(w->second);
            by_subj_[x.subj].push_back(static_cast<std::uint32_t>(facts_.size()));
            by_val_[x.val].push_back(static_cast<std::uint32_t>(facts_.size()));
            facts_.push_back(x);
        }
        for (std::uint32_t e = 0; e < ent_.size(); ++e) {
            const auto& t = ent_tok_[e];
            key_ent_[seq_key(t, 0, t.size())].push_back(e);
        }
        if (opt_.find == 2) {  // aliases.tsv: alias (lowercased), target, kind, count, display total, byte offset
            std::ifstream in(dir + "/aliases.tsv");
            std::string line;
            std::getline(in, line);
            std::unordered_map<std::uint64_t, std::pair<std::uint32_t, double>> best;  // alias key -> (subject, strength)
            while (std::getline(in, line)) {
                const auto f = split_tab(line);
                if (f.size() < 6) continue;
                const std::size_t off = static_cast<std::size_t>(std::strtoull(f[5].c_str(), nullptr, 10));
                bool drop = false;
                for (const auto& [lo, hi] : drops) drop = drop || (off >= lo && off < hi);
                if (drop) continue;
                const std::uint32_t e = find(f[1]);
                if (e == ~0u || of_subj(e).empty()) continue;  // only names of subjects with facts
                const auto t = wkn::tokens(f[0]);
                if (t.empty() || t.size() > kMaxTok) continue;
                const std::uint64_t k = seq_key(t, 0, t.size());
                const double s = f[2] == "redirect" ? 1e9 : std::atof(f[3].c_str());  // a redirect names its page outright
                auto& b = best[k];
                if (s > b.second) b = {e, s};
            }
            for (const auto& [k, v] : best) alias_[k] = v.first;
        }
        return true;
    }
    // PLAN_FACTS: the main entity for candidate c. find 1: find(c). find 2: c if it is a subject with facts, else the
    // subject an alias of c names, else the same for c without "'s" or a plural "s"; else find(c).
    std::uint32_t find_main(const std::string& c) const {
        const std::uint32_t e = find(c);
        if (opt_.find != 2 || (e != ~0u && !of_subj(e).empty())) return e;
        const auto subj_of = [&](const std::string& s) -> std::uint32_t {
            const std::uint32_t x = find(s);
            if (x != ~0u && !of_subj(x).empty()) return x;
            const auto t = wkn::tokens(s);
            if (t.empty() || t.size() > kMaxTok) return ~0u;
            const auto it = alias_.find(seq_key(t, 0, t.size()));
            return it == alias_.end() ? ~0u : it->second;
        };
        if (const std::uint32_t a = subj_of(c); a != ~0u) return a;
        if (c.size() > 3 && (c.compare(c.size() - 2, 2, "'s") == 0 || c.compare(c.size() - 2, 2, "s'") == 0))
            if (const std::uint32_t a = subj_of(c.substr(0, c.size() - 2)); a != ~0u) return a;
        if (c.size() > 4 && c.back() == 's' && c[c.size() - 2] != 's')
            if (const std::uint32_t a = subj_of(c.substr(0, c.size() - 1)); a != ~0u) return a;
        return e;
    }
    // PLAN_FACTS: the prompt's entity candidates. find 1: the content plan's (CPS::candidates). find 2: those, less
    // month and day names and (with the structure statistics st) a lone capitalised word that stands only at a
    // sentence's start and is mostly lowercase in running text ("Black powder arms ..."); then each lowercase run of 2
    // to kMaxTok tokens, with no function word at either end, that is a store subject with a common-noun title
    // ("Hydrogen peroxide", not "The Other") or an alias of a subject (longest first, left to right).
    std::vector<std::string> candidates(std::string_view prompt, const structure::Stats* st = nullptr) const {
        auto out = CPS::candidates(std::string(prompt));
        if (opt_.find != 2) return out;
        const std::string plain = CPS::strip(std::string(prompt));
        const auto t = wkn::tokens(plain);
        {
            std::vector<std::string> keep;
            for (const auto& c : out) {
                const auto ct = wkn::tokens(c);
                if (ct.size() == 1 && (structure::is_month(ct[0]) || is_day(ct[0]))) continue;
                if (ct.size() == 1 && st) {
                    bool only_first = true;
                    for (std::size_t i = 0; i < t.size(); ++i)
                        if (t[i] == ct[0]) only_first = only_first && (i == 0 || t[i - 1] == "." || t[i - 1] == "!" || t[i - 1] == "?" || t[i - 1] == "\"");
                    if (only_first && !st->cap_name(ct[0], true, false)) continue;
                }
                keep.push_back(c);
            }
            out.swap(keep);
        }
        for (std::size_t a = 0; a < t.size();) {
            std::size_t took = 0;
            for (std::size_t n = std::min(kMaxTok, t.size() - a); n >= 2 && !took; --n) {
                bool word = true, has_lower = false;
                for (std::size_t i = a; i < a + n && word; ++i) {
                    const unsigned char c0 = t[i].empty() ? 0 : static_cast<unsigned char>(t[i][0]);
                    word = std::isalpha(c0) || c0 >= 0x80;
                    has_lower = has_lower || std::islower(c0);
                }
                if (!word || !has_lower || fn_word(t[a]) || fn_word(t[a + n - 1])) continue;
                const std::uint64_t k = seq_key(t, a, n);
                bool hit = alias_.count(k) > 0;
                if (!hit)
                    if (const auto it = key_ent_.find(k); it != key_ent_.end())
                        for (const auto e : it->second) hit = hit || (ent_tok_[e].size() == n && !of_subj(e).empty() && common_title(ent_tok_[e]));
                if (!hit) continue;
                std::string s;
                for (std::size_t i = a; i < a + n; ++i) s += (i > a ? " " : "") + t[i];
                if (std::find(out.begin(), out.end(), s) == out.end()) out.push_back(s);
                took = n;
            }
            a += took ? took : 1;
        }
        return out;
    }
    static bool is_day(const std::string& w) {
        static const std::set<std::string> D = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
        return D.count(w) > 0;
    }
    static bool fn_word(const std::string& w) {
        static const std::set<std::string> F = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "by", "for", "from", "to", "with",
                                                "as", "is", "was", "are", "were", "be", "this", "that", "these", "those", "it", "its",
                                                "his", "her", "their", "he", "she", "they", "we", "which", "who", "but", "not", "other"};
        return F.count(lower_s(w)) > 0;
    }
    // A common-noun title: every word after the first starts lowercase or is not a letter ("Hydrogen peroxide").
    static bool common_title(const std::vector<std::string>& tk) {
        for (std::size_t i = 1; i < tk.size(); ++i)
            if (!tk[i].empty() && tk[i][0] >= 'A' && tk[i][0] <= 'Z') return false;
        return true;
    }
    std::size_t aliases() const { return alias_.size(); }
    std::size_t filtered() const { return filtered_; }
    const std::map<std::string, std::size_t>& filter_reasons() const { return filter_reason_; }
    const FactOptions& options() const { return opt_; }
    // The structure statistics the filter's place rule reads (the planner model's; set before load). Unused without a filter.
    void set_stats(const structure::Stats* st) { stats_ = st; }

    // An entity ID by its surface (case-insensitive tokens), or ~0u.
    std::uint32_t find(const std::string& s) const {
        const auto t = wkn::tokens(s);
        if (t.empty() || t.size() > kMaxTok) return ~0u;
        const auto it = key_ent_.find(seq_key(t, 0, t.size()));
        if (it == key_ent_.end()) return ~0u;
        for (const auto e : it->second)
            if (lower_s(ent_[e]) == lower_s(s)) return e;
        return it->second[0];
    }
    // Entities whose tokens start at t[a] (lengths 1..kMaxTok), longest first.
    void match_at(const std::vector<std::string>& t, std::size_t a, std::vector<std::pair<std::uint32_t, std::size_t>>& out) const {
        out.clear();
        for (std::size_t n = std::min(kMaxTok, t.size() - a); n >= 1; --n) {
            const auto it = key_ent_.find(seq_key(t, a, n));
            if (it == key_ent_.end()) continue;
            for (const auto e : it->second)
                if (ent_tok_[e].size() == n) out.push_back({e, n});
        }
    }
    const std::vector<std::uint32_t>& of_subj(std::uint32_t e) const { return get(by_subj_, e); }
    const std::vector<std::uint32_t>& of_val(std::uint32_t e) const { return get(by_val_, e); }
    const Fact& fact(std::uint32_t i) const { return facts_[i]; }
    std::size_t size() const { return facts_.size(); }
    std::size_t dropped() const { return dropped_; }
    const std::string& ent(std::uint32_t e) const { return ent_[e]; }
    const std::vector<std::string>& ent_tok(std::uint32_t e) const { return ent_tok_[e]; }
    const std::string& rel(std::uint32_t r) const { return rel_[r]; }
    std::uint32_t rel_id(const std::string& r) const {
        const auto it = rel_id_.find(r);
        return it == rel_id_.end() ? ~0u : it->second;
    }
    std::size_t rels() const { return rel_.size(); }
    std::size_t subjects() const { return by_subj_.size(); }
    const std::string& dir() const { return dir_; }

 private:
    static std::uint32_t intern(std::vector<std::string>& v, std::unordered_map<std::string, std::uint32_t>& id, const std::string& s,
                                std::vector<std::vector<std::string>>* tok) {
        const auto [it, fresh] = id.try_emplace(s, static_cast<std::uint32_t>(v.size()));
        if (fresh) {
            v.push_back(s);
            if (tok) tok->push_back(wkn::tokens(s));
        }
        return it->second;
    }
    static const std::vector<std::uint32_t>& get(const std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>& m, std::uint32_t e) {
        static const std::vector<std::uint32_t> none;
        const auto it = m.find(e);
        return it == m.end() ? none : it->second;
    }
    std::string dir_;
    std::vector<Fact> facts_;
    std::vector<std::string> ent_, rel_;
    std::vector<std::vector<std::string>> ent_tok_;
    std::unordered_map<std::string, std::uint32_t> ent_id_, rel_id_;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> by_subj_, by_val_;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> key_ent_;
    std::size_t dropped_ = 0;
    // PLAN_FACTS
    FactOptions opt_;
    const structure::Stats* stats_ = nullptr;
    std::unordered_map<std::uint64_t, std::uint32_t> alias_;  // alias key -> subject
    std::size_t filtered_ = 0;
    std::map<std::string, std::size_t> filter_reason_;
};

}  // namespace cypha::cyphalm::planner
