// Generators G1 (deep deduction), G19 (refinement) and G20 (the wild tier) for Cypha's discovery track
// (docs/DISCOVERY.md section 5; docs/reports/S12A_DEDUCTION_LAWS_REFINEMENT.md).
//
// Header-only. Reads a TripleStore (hypothesis.hpp), G2's rules (gen_rules.hpp) and S9's worlds (worlds.hpp).
//
//   deep_chain     G1: forward chaining of the kept G2 rules inside a world, from every fact of the store, up to a depth.
//                  A fact first derived at depth d keeps its premises and its chain confidence (the product of the
//                  rules' PCA confidences). Depth 1 is G2's own prediction; G1's hypotheses are depth >= 2.
//   refine_rules   G19: one condition atom c(X, v) added to a const, same or inverse rule's subject, chosen among the
//                  constant facts of the rule's supporting subjects (a beam of 3 by refined PCA). Kept at support >= 10,
//                  PCA >= 0.5, PCA at least 0.1 above the parent's, and Benjamini-Hochberg over the refinements.
//   wild_tier      G20: for a slot (s, r), a value drawn from r's objects in another cluster (type and country), kept
//                  when the link predictor scores it at least twice the base rate; a fixed budget.
//
// Nothing here answers a question. Derived facts, refined rules and wild pairs are hypotheses (PLAN rule 10).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cypha/eval/gen_rules.hpp"
#include "cypha/eval/hypothesis.hpp"
#include "cypha/eval/lci.hpp"
#include "cypha/eval/worlds.hpp"

namespace cypha::eval::deep {

// ---------------------------------------------------------------------------
// G1: deep deduction
// ---------------------------------------------------------------------------

/// Every store fact, sorted (a deterministic frontier).
inline std::vector<Triple> store_triples(const TripleStore& st) {
  std::vector<Triple> out;
  out.reserve(st.size());
  for (const auto& [s, fs] : st.subjects())
    for (const auto& f : fs) out.push_back({s, f.r, f.o});
  std::sort(out.begin(), out.end());
  return out;
}

/// G1: chain the kept rules from every store fact up to `depth`. Returns the new facts with their first depth and best
/// chain confidence (worlds.hpp's chain: derived facts join the world, so later steps read them; nothing is contested
/// here, so no conflict checks run).
inline std::vector<Derived> deep_chain(const TripleStore& st, const RuleBook& book, int depth) {
  World w(st);
  ChainConfig cfg;
  cfg.depth = depth;
  cfg.c_min = 2.0;  // no contradiction is ever counted
  ChainResult res = chain(w, book, cfg, ConflictChecks{}, store_triples(st));
  return std::move(res.derived);
}

// ---------------------------------------------------------------------------
// G19: refinement (specialisation by one condition atom)
// ---------------------------------------------------------------------------

struct RefineConfig {
  double min_support = 10.0, min_pca = 0.5, min_gain = 0.1, q = 0.1;
  std::size_t beam = 3;
};

struct Refinement {
  std::size_t parent = 0;  // index into the candidate rules given
  std::string cr, cc;      // the condition atom cr(X, cc) on the rule's subject
  double support = 0, pca_body = 0, hid_support = 0, hid_pca = 0, p_value = 1.0;
  bool kept = false;
  double pca_conf() const { return pca_body > 0 ? support / pca_body : 0.0; }
};

/// The body pairs (x, y) of a const, same or inverse rule; empty for a path rule (not refined).
inline std::vector<std::pair<std::string, std::string>> body_pairs(const TripleStore& st, const Rule& R,
                                                                   const std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>>& by_rel) {
  std::vector<std::pair<std::string, std::string>> out;
  if (R.kind == RuleKind::Const) {
    for (const auto& x : st.subjects_with(R.r1, R.c1)) out.push_back({x, R.head_o});
  } else if (R.kind == RuleKind::Same || R.kind == RuleKind::Inverse) {
    const auto it = by_rel.find(R.r1);
    if (it != by_rel.end())
      for (const auto& [a, b] : it->second) out.push_back(R.kind == RuleKind::Same ? std::make_pair(a, b) : std::make_pair(b, a));
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

/// Facts by relation: r -> (s, o) pairs.
inline std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> facts_by_relation(const TripleStore& st) {
  std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> m;
  for (const auto& [s, fs] : st.subjects())
    for (const auto& f : fs) m[f.r].push_back({s, f.o});
  for (auto& [r, v] : m) std::sort(v.begin(), v.end());
  return m;
}

/// G19: refinements of every const, same or inverse candidate rule with support >= min_support (kept or not).
inline std::vector<Refinement> refine_rules(const TripleStore& st, const std::vector<Rule>& rules, const RefineConfig& cfg = {}) {
  std::vector<Refinement> out;
  const auto by_rel = facts_by_relation(st);
  for (std::size_t ri = 0; ri < rules.size(); ++ri) {
    const Rule& R = rules[ri];
    if (R.kind == RuleKind::Path || R.support < cfg.min_support) continue;
    const auto pairs = body_pairs(st, R, by_rel);
    // condition atoms among the supporting subjects (each subject once), never the head's relation or the body atom
    std::map<std::pair<std::string, std::string>, double> atom_n;
    std::set<std::string> support_x;
    for (const auto& [x, y] : pairs)
      if (st.has(x, R.head, y)) support_x.insert(x);
    for (const auto& x : support_x)
      for (const auto& f : st.facts_of(x)) {
        if (f.r == R.head) continue;
        if (R.kind == RuleKind::Const && f.r == R.r1 && f.o == R.c1) continue;
        atom_n[{f.r, f.o}] += 1;
      }
    std::vector<Refinement> cand;
    for (const auto& [atom, n] : atom_n) {
      if (n < cfg.min_support) continue;
      Refinement F;
      F.parent = ri, F.cr = atom.first, F.cc = atom.second;
      for (const auto& [x, y] : pairs) {
        if (!st.has(x, atom.first, atom.second) || !st.has_relation(x, R.head)) continue;
        const bool hid = gen_hidden_third(x);
        F.pca_body += 1, F.hid_pca += hid;
        if (st.has(x, R.head, y)) F.support += 1, F.hid_support += hid;
      }
      if (F.support < cfg.min_support) continue;
      F.p_value = binomial_upper_tail(F.support, F.pca_body, R.p0);
      cand.push_back(F);
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Refinement& a, const Refinement& b) {
      if (a.pca_conf() != b.pca_conf()) return a.pca_conf() > b.pca_conf();
      if (a.support != b.support) return a.support > b.support;
      return a.cr + a.cc < b.cr + b.cc;
    });
    if (cand.size() > cfg.beam) cand.resize(cfg.beam);
    out.insert(out.end(), cand.begin(), cand.end());
  }
  std::vector<double> p;
  for (const auto& F : out) p.push_back(F.p_value);
  const auto bh = benjamini_hochberg(p, cfg.q);
  for (std::size_t i = 0; i < out.size(); ++i) {
    const Rule& R = rules[out[i].parent];
    out[i].kept = bh.reject[i] && out[i].support >= cfg.min_support && out[i].pca_conf() >= cfg.min_pca &&
                  out[i].pca_conf() >= R.pca_conf() + cfg.min_gain;
  }
  return out;
}

/// What a refinement derives: (x, head, y) for its body pairs whose x holds the condition, not already in the store.
inline std::vector<Triple> refinement_predictions(const TripleStore& st, const Rule& parent, const Refinement& F,
                                                  const std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>>& by_rel) {
  std::vector<Triple> out;
  for (const auto& [x, y] : body_pairs(st, parent, by_rel))
    if (x != y && st.has(x, F.cr, F.cc) && !st.has(x, parent.head, y)) out.push_back({x, parent.head, y});
  return out;
}

// ---------------------------------------------------------------------------
// G20: the wild tier
// ---------------------------------------------------------------------------

/// A node's cluster: its type by relations (person, place, other; a value that is not a subject: value) and its country
/// (`country` or `nationality`, the first in sorted order, else none).
inline std::string cluster_of(const TripleStore& st, const std::string& s) {
  const auto& fs = st.facts_of(s);
  if (fs.empty()) return "value|none";
  std::string type = "other", country;
  for (const auto& f : fs) {
    if (f.r == "birth_year" || f.r == "birth_date" || f.r == "death_year" || f.r == "death_date" || f.r == "occupation") type = "person";
    else if (type != "person" && (f.r == "located_in" || f.r == "country" || f.r == "capital")) type = "place";
    if ((f.r == "country" || f.r == "nationality") && (country.empty() || f.o < country)) country = f.o;
  }
  return type + "|" + (country.empty() ? "none" : country);
}

struct WildConfig {
  double lp_ratio = 2.0;        // kept when P(s r o) >= 2 x the base rate
  std::size_t tries = 20;       // values drawn per slot
};

struct WildPick {
  Triple t;
  double lp = 0, base = 0;
};

/// G20: up to `budget` wild hypotheses for the given slots, drawn with mt19937_64(seed).
inline std::vector<WildPick> wild_tier(const TripleStore& st, const LinkPredictor& lp, std::vector<std::pair<std::string, std::string>> slots,
                                       std::size_t budget, std::uint64_t seed, const WildConfig& cfg = {}) {
  std::vector<WildPick> out;
  std::mt19937_64 rng(seed);
  std::sort(slots.begin(), slots.end());
  std::shuffle(slots.begin(), slots.end(), rng);
  const auto by_rel = facts_by_relation(st);
  for (const auto& [s, r] : slots) {
    if (out.size() >= budget) break;
    const auto it = by_rel.find(r);
    if (it == by_rel.end() || it->second.empty() || st.facts_of(s).empty()) continue;
    std::vector<std::string> objs;
    for (const auto& [a, o] : it->second) objs.push_back(o);
    std::sort(objs.begin(), objs.end());
    objs.erase(std::unique(objs.begin(), objs.end()), objs.end());
    const std::string cs = cluster_of(st, s);
    for (std::size_t k = 0; k < cfg.tries; ++k) {
      const std::string& o = objs[rng() % objs.size()];
      if (o == s || st.has(s, r, o) || cluster_of(st, o) == cs) continue;
      const double b = base_rate(st, r, o, 1.0, false);
      const double p = lp.prob(s, r, o);
      if (p >= cfg.lp_ratio * b) {
        out.push_back({{s, r, o}, p, b});
        break;
      }
    }
  }
  return out;
}

}  // namespace cypha::eval::deep
