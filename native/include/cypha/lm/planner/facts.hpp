#pragma once

/// The planner's fact store (docs/reports/K5_PLANNER.md, GENERATION_PLAN §13.37-13.38): CyphaLM's own facts from
/// pattern_learn (F2, `seeds.tsv` and `facts_f2.tsv`), held as integer IDs. A fact is (subject, relation, value), each
/// an interned ID; the subject and value also carry their K5 tokens, so every later step searches on IDs only.
///
/// The store directory is a plain argument (verify_f2r_aug today; the enwik9 store can be swapped in). Facts whose
/// evidence offset lies in a dropped range (the prompts' own bytes, `--plan-drop LO:HI`) are not loaded.
///
/// Author: Odin Loch

#include "cypha/lm/content_plan.hpp"
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <tuple>
#include <string>
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

class FactStore {
 public:
    struct Fact {
        std::uint32_t subj = 0, rel = 0, val = 0;
        bool text = false;           // read from the text (F2, infobox, table, category); else a Wikidata seed
        std::int8_t wd = -1;         // facts_wd_check: 1 matches Wikidata, 0 differs, -1 unchecked
    };
    static constexpr std::size_t kMaxTok = 6;  // subjects and values longer than this many tokens are not planned

    // dir: seeds.tsv and facts_f2.tsv (pattern_learn's v2 files); drops: evidence ranges whose facts are left out.
    bool load(const std::string& dir, const std::vector<std::pair<std::size_t, std::size_t>>& drops, std::string& err) {
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
};

}  // namespace cypha::cyphalm::planner
