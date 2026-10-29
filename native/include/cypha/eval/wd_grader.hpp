// S8e: a grader that knows Wikidata's names and places (docs/reports/S8E_GRADER_AND_FRESH_SET.md).
//
// A hypothesis value h (a name, normalised by sources.hpp's norm_name) against one true value item t of its slot:
//   Match    h is one of t's names (English label, any English alias, enwiki title, English demonym, taxon name); or, for
//            birth_place / death_place / location / headquarters, h names a place ancestor of t that is not coarse; or, for
//            located_in (transitive), h names any place ancestor of t; or, for country / nationality, h names a
//            country-like item inside t (England for the United Kingdom).
//   Finer    for the place relations above, h names an item inside t: Wikidata cannot say whether the finer place is right.
//   NoMatch  otherwise.
// Place ancestors are (P131 | P276 | P17)+; country-like and coarse come from P31/P279* (scripts/fresh_slots.py grader).
// Grading only: it judges Cypha's own hypotheses and never answers a question (PLAN rule 10).
// Author: Odin Loch
#pragma once

#include "cypha/eval/sources.hpp"

#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cypha::eval::wdgrade {

enum class Verdict { NoMatch, Match, Finer };

struct Tables {
  std::unordered_map<std::string, std::unordered_set<std::string>> names;  // item -> normalised names
  std::unordered_map<std::string, std::vector<std::string>> by_name;       // normalised name -> items
  std::unordered_map<std::string, std::unordered_set<std::string>> anc;    // item -> place ancestors
  std::unordered_set<std::string> country_like, coarse;

  bool empty() const { return names.empty(); }

  void add_name(const std::string& item, const std::string& raw) {
    const std::string n = sources::norm_name(raw);
    if (item.empty() || n.empty()) return;
    if (names[item].insert(n).second) by_name[n].push_back(item);
  }
  void add_ancestor(const std::string& item, const std::string& a) {
    if (!item.empty() && !a.empty() && a != item) anc[item].insert(a);
  }
  bool named(const std::string& item, const std::string& h) const {
    const auto it = names.find(item);
    return it != names.end() && it->second.count(h) > 0;
  }
  bool inside(const std::string& item, const std::string& outer) const {
    const auto it = anc.find(item);
    return it != anc.end() && it->second.count(outer) > 0;
  }

  /// names.tsv (item, names | separated), anc.tsv (item, ancestors | separated), flags.tsv (item, country_like, coarse).
  bool load(const std::string& dir) {
    auto rows = [](const std::string& path, auto&& fn) {
      std::ifstream f(path);
      if (!f) return false;
      std::string l;
      std::getline(f, l);
      while (std::getline(f, l)) {
        std::vector<std::string> c;
        std::size_t a = 0;
        for (std::size_t b; (b = l.find('\t', a)) != std::string::npos; a = b + 1) c.push_back(l.substr(a, b - a));
        c.push_back(l.substr(a));
        fn(c);
      }
      return true;
    };
    auto parts = [](const std::string& s) {
      std::vector<std::string> o;
      std::size_t a = 0;
      for (std::size_t b; (b = s.find('|', a)) != std::string::npos; a = b + 1) o.push_back(s.substr(a, b - a));
      o.push_back(s.substr(a));
      return o;
    };
    const bool ok1 = rows(dir + "/names.tsv", [&](const std::vector<std::string>& c) {
      if (c.size() >= 2)
        for (const std::string& n : parts(c[1])) add_name(c[0], n);
    });
    const bool ok2 = rows(dir + "/anc.tsv", [&](const std::vector<std::string>& c) {
      if (c.size() >= 2)
        for (const std::string& x : parts(c[1])) add_ancestor(c[0], x);
    });
    const bool ok3 = rows(dir + "/flags.tsv", [&](const std::vector<std::string>& c) {
      if (c.size() < 3) return;
      if (c[1] == "1") country_like.insert(c[0]);
      if (c[2] == "1") coarse.insert(c[0]);
    });
    return ok1 && ok2 && ok3;
  }
};

/// Relations whose places may be named by a containing place (rule 3); located_in takes coarse ones too.
inline bool place_up(const std::string& r) {
  return r == "birth_place" || r == "death_place" || r == "location" || r == "headquarters";
}
inline bool country_relation(const std::string& r) { return r == "country" || r == "nationality"; }

/// The fresh set's relations: Wikidata's lists are taken as complete, so a slot with a true value is closed-world.
inline bool closed_world(const std::string& r) {
  return r == "nationality" || r == "location" || r == "predecessor" || r == "successor" || r == "family" || r == "order" ||
         r == "class";
}

/// The verdict for one true value item t; `rule`, when given, receives the registered rule that decided it (2 a name, 3 a
/// containing place, 4 a finer place, 5 a country-like place inside; 0 no match).
inline Verdict match(const Tables& T, const std::string& r, const std::string& t, const std::string& h, int* rule = nullptr) {
  auto out = [&](Verdict v, int k) {
    if (rule) *rule = k;
    return v;
  };
  if (t.empty() || h.empty()) return out(Verdict::NoMatch, 0);
  if (T.named(t, h)) return out(Verdict::Match, 2);
  const bool up = place_up(r), loc = r == "located_in";
  if (up || loc) {
    if (const auto it = T.anc.find(t); it != T.anc.end())
      for (const std::string& a : it->second)
        if ((loc || !T.coarse.count(a)) && T.named(a, h)) return out(Verdict::Match, 3);
  }
  const auto bn = T.by_name.find(h);
  if (bn == T.by_name.end()) return out(Verdict::NoMatch, 0);
  if (country_relation(r)) {
    for (const std::string& i : bn->second)
      if (T.country_like.count(i) && T.inside(i, t)) return out(Verdict::Match, 5);
  } else if (up || loc) {
    for (const std::string& i : bn->second)
      if (T.inside(i, t)) return out(Verdict::Finer, 4);
  }
  return out(Verdict::NoMatch, 0);
}

/// One hypothesis value v against its slot's rows (value, label 1 true / 0 drawn false), S8c's grader with S8e's rules
/// (docs/reports/S8E_GRADER_AND_FRESH_SET.md, Part 1): +1 true, 0 false, -1 ungraded. Without tables (T null) it is S8c's
/// grader unchanged. `basis` receives what decided it: shown (S15's shown value), rule2 / rule3 / rule5, drawn-false, finer
/// (rule 4), single-valued, closed-world, or open.
inline int grade_slot(const Tables* T, const std::string& r, const std::vector<std::pair<sources::Value, int>>& vals,
                      const sources::Value& v, const char** basis = nullptr) {
  auto out = [&](int g, const char* w) {
    if (basis) *basis = w;
    return g;
  };
  const bool wd = T && !sources::year_relation(r) && !sources::date_relation(r);
  bool any_true = false, match_false = false, finer = false;
  int rule_hit = 0;
  for (const auto& [val, lab] : vals) {
    if (lab == 1) {
      any_true = true;
      if (sources::same_value(r, val, v)) return out(1, "shown");
      if (wd) {
        int k = 0;
        const Verdict m = match(*T, r, val.qid, v.name, &k);
        if (m == Verdict::Match && (rule_hit == 0 || k < rule_hit)) rule_hit = k;
        finer = finer || m == Verdict::Finer;
      }
    } else if (lab == 0 && (sources::same_value(r, val, v) || (wd && T->named(val.qid, v.name)))) {
      match_false = true;
    }
  }
  if (rule_hit) return out(1, rule_hit == 2 ? "rule2" : rule_hit == 3 ? "rule3" : "rule5");
  if (match_false) return out(0, "drawn-false");
  if (finer) return out(-1, "finer");
  if (any_true && sources::single_valued(r)) return out(0, "single-valued");
  if (any_true && wd && closed_world(r)) return out(0, "closed-world");
  return out(-1, "open");
}

}  // namespace cypha::eval::wdgrade
