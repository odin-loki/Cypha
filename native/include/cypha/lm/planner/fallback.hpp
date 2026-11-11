#pragma once

/// The relations-pattern fallback of the planner (docs/reports/PLAN_FACTS.md, GENERATION_PLAN §13.37, §13.42 Q1):
///   sample --k5 ... --k5-plan STORE --k5-plan-fallback pattern
///
/// When no prompt entity has facts of its own, the relations expert's typed statistics (structure/experts.hpp: a name's
/// type from the pronouns that follow it and the words before it; document co-occurrence of names) choose what kind of
/// fact to write:
///   1. Typed facts about the prompt's names. A store fact whose value is a prompt name is kept only if its relation's
///      value type (the most common type of that relation's values in the store, by the same statistics) is the
///      name's own type: a place name fills a place slot ("X was born in Wellington"), a person a person slot. Facts
///      are ranked by how often the relation opens a document (the skeleton bank's transitions) and by the document
///      PPMI of their subject with the prompt's names; at most 3, one per subject and per relation.
///   2. Else the store subject that co-occurs most with the prompt's names (document PPMI at least kMinPmi), with up
///      to 2 of its facts, as the main path takes them (a bank relation, at most 2 values).
/// Neither: the caller's co-occurring fallback, then plain K5. Search on IDs only.
///
/// Author: Odin Loch

#include "cypha/lm/planner/facts.hpp"
#include "cypha/lm/planner/skeleton.hpp"
#include "cypha/lm/structure/experts.hpp"
#include "cypha/lm/word_kn.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cypha::cyphalm::planner {

struct FallbackPick {
    std::uint32_t ent = ~0u;
    std::vector<std::uint32_t> facts;
    int tier = 0;  // 1 typed facts about prompt names, 2 a co-occurring subject
};

inline constexpr double kMinPmi = 2.0;

// A name's type by the relations expert: 1 person, 2 place, 3 year, 5 other.
inline int name_type(const structure::Stats& stats, const std::string& s) {
    if (structure::year_of(s)) return 3;
    return stats.type_of_name(s);
}

// Each relation's value type (the most common name_type of its values in the store), computed once per store.
inline const std::vector<int>& rel_value_types(const FactStore& fs, const structure::Stats& stats) {
    static std::mutex mu;
    static std::map<const FactStore*, std::vector<int>> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(&fs);
    if (it != cache.end()) return it->second;
    std::vector<std::array<std::size_t, 6>> n(fs.rels());
    for (auto& a : n) a.fill(0);
    for (std::uint32_t i = 0; i < fs.size(); ++i) {
        const auto& f = fs.fact(i);
        ++n[f.rel][static_cast<std::size_t>(name_type(stats, fs.ent(f.val)))];
    }
    std::vector<int> t(fs.rels(), 5);
    for (std::size_t r = 0; r < n.size(); ++r) t[r] = static_cast<int>(std::max_element(n[r].begin(), n[r].end()) - n[r].begin());
    return cache.emplace(&fs, std::move(t)).first->second;
}

inline FallbackPick typed_fallback(const FactStore& fs, const SkelBank& bank, const structure::Stats& stats, const std::vector<std::string>& cands,
                                   std::string_view prompt) {
    FallbackPick out;
    const auto& vt = rel_value_types(fs, stats);
    // the prompt's names (the relations expert's name runs) with their co-occurrence ids
    std::vector<std::pair<std::int32_t, std::uint32_t>> ids;  // (id, docs)
    for (const auto& s : structure::split(wkn::tokens(prompt))) {
        for (const auto& [a, b] : stats.name_runs(s)) {
            const auto* ns = stats.name(structure::Stats::join(s, a, b));
            if (ns && ns->id >= 0) ids.push_back({ns->id, ns->docs});
        }
    }
    const auto pmi = [&](std::uint32_t e) {
        const auto* ns = stats.name(fs.ent(e));
        double best = 0;
        if (!ns || ns->id < 0) return best;
        for (const auto& [id, d] : ids) best = std::max(best, stats.name_ppmi(ns->id, id, ns->docs, d));
        return best;
    };
    // 1. typed facts about the prompt's names
    struct C {
        double score;
        std::uint32_t fact, ent;
    };
    std::vector<C> cs;
    std::set<std::uint32_t> seen_ent;
    for (const auto& c : cands) {
        const std::uint32_t e = fs.find(c);
        if (e == ~0u || !seen_ent.insert(e).second || fs.of_val(e).empty()) continue;
        const int t = name_type(stats, fs.ent(e));
        if (t == 5) continue;  // an untyped name says nothing about the slot
        for (const auto fi : fs.of_val(e)) {
            const auto& f = fs.fact(fi);
            if (vt[f.rel] != t || !bank.of(fs.rel(f.rel))) continue;
            cs.push_back({bank.trans_lp("<s>", fs.rel(f.rel)) + pmi(f.subj), fi, e});
        }
    }
    std::stable_sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.score > b.score; });
    std::set<std::uint32_t> subj, rels;
    for (const auto& c : cs) {
        if (out.facts.size() >= 3) break;
        const auto& f = fs.fact(c.fact);
        if (subj.count(f.subj) || rels.count(f.rel)) continue;
        subj.insert(f.subj), rels.insert(f.rel);
        out.facts.push_back(c.fact);
        if (out.ent == ~0u) out.ent = c.ent;
    }
    if (!out.facts.empty()) {
        out.tier = 1;
        return out;
    }
    // 2. the store subject that co-occurs most with the prompt's names
    if (ids.empty()) return out;
    double best = kMinPmi;
    std::uint32_t be = ~0u;
    std::vector<std::uint32_t> bf;
    for (std::uint32_t i = 0; i < fs.size(); ++i) {
        const std::uint32_t e = fs.fact(i).subj;
        if (fs.of_subj(e).empty() || fs.of_subj(e)[0] != i) continue;  // each subject once (at its first fact)
        const double p = pmi(e);
        if (p <= best) continue;
        std::map<std::uint32_t, std::set<std::uint32_t>> by;
        std::map<std::uint32_t, std::uint32_t> first;
        for (const auto fi : fs.of_subj(e)) {
            const auto& f = fs.fact(fi);
            if (!bank.of(fs.rel(f.rel))) continue;
            by[f.rel].insert(f.val);
            first.try_emplace(f.rel, fi);
        }
        std::vector<std::uint32_t> pick;
        for (const auto& [r, v] : by)
            if (v.size() <= 2) pick.push_back(first[r]);
        if (pick.empty()) continue;
        std::stable_sort(pick.begin(), pick.end(), [&](std::uint32_t a, std::uint32_t b) {
            return bank.trans_lp("<s>", fs.rel(fs.fact(a).rel)) > bank.trans_lp("<s>", fs.rel(fs.fact(b).rel));
        });
        if (pick.size() > 2) pick.resize(2);
        best = p, be = e, bf = pick;
    }
    if (be != ~0u) out.ent = be, out.facts = bf, out.tier = 2;
    return out;
}

}  // namespace cypha::cyphalm::planner
