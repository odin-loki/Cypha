// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Odin Loch
//
// cypha/eval/gen_concepts.hpp
//
// DISCOVERY 5's G9 (concept formation) and G11 (open relations), S11a (docs/reports/S11A_CONCEPTS_AND_OPEN_RELATIONS.md).
//
// G9: a formal context (entity x attribute, an attribute being a (relation, value) pair the store holds for at least
// `min_support` entities), its closed concepts by LCM (prefix-preserving closure extension, Uno et al. 2004), candidate
// classes (an intent of 2+ attributes whose extent no single attribute already names) and class rules C => b with
// DISCOVERY P5's information-gain test: the class must predict b better than any one of its own attributes does
// (binomial upper tail against that baseline, Benjamini-Hochberg over the batch).
//
// G11: unparsed connecting phrases clustered by their core key or by their connector (the words next to the object),
// and each cluster's majority property with its precision.
//
// These are Cypha's own hypotheses (PLAN rule 10): nothing here answers a question.
//
// Author: Odin Loch

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cypha/eval/ecs.hpp"
#include "cypha/eval/gen_rules.hpp"
#include "cypha/eval/lci.hpp"

namespace cypha::eval::concepts {

// ---------------------------------------------------------------------------
// G9: the formal context
// ---------------------------------------------------------------------------

/// One fact as the context reads it: an entity, a relation and a value already normalised by the caller.
struct Fact3 {
  std::string e, r, v;
};

struct FormalContext {
  std::vector<std::string> entities;          // sorted
  std::vector<std::string> attr_r, attr_v;    // attribute a = (attr_r[a], attr_v[a]), sorted by (r, v)
  std::vector<std::vector<int>> ext;          // ext[a]: sorted entity ids holding a
  std::vector<std::vector<int>> attrs;        // attrs[e]: sorted attribute ids of entity e
  std::vector<std::set<std::string>> rels;    // rels[e]: every relation e holds in the facts, kept attribute or not

  int entity(const std::string& name) const {
    const auto it = std::lower_bound(entities.begin(), entities.end(), name);
    return it != entities.end() && *it == name ? static_cast<int>(it - entities.begin()) : -1;
  }
  int attribute(const std::string& r, const std::string& v) const {
    for (std::size_t a = 0; a < attr_r.size(); ++a)
      if (attr_r[a] == r && attr_v[a] == v) return static_cast<int>(a);
    return -1;
  }
  std::string attr_text(int a) const { return attr_r[static_cast<std::size_t>(a)] + " = " + attr_v[static_cast<std::size_t>(a)]; }
};

/// The context of `facts`: attributes held by at least `min_support` distinct entities; the entities holding one.
/// `all_rels` gives, per entity name, the relations it holds (for PCA); when null, the facts' own relations are used.
inline FormalContext build_context(const std::vector<Fact3>& facts, double min_support,
                                   const std::map<std::string, std::set<std::string>>* all_rels = nullptr) {
  std::map<std::pair<std::string, std::string>, std::set<std::string>> by_attr;
  std::map<std::string, std::set<std::string>> own_rels;
  for (const auto& f : facts) {
    if (f.e.empty() || f.r.empty() || f.v.empty()) continue;
    by_attr[{f.r, f.v}].insert(f.e);
    own_rels[f.e].insert(f.r);
  }
  FormalContext C;
  std::set<std::string> ents;
  for (const auto& [a, es] : by_attr)
    if (static_cast<double>(es.size()) >= min_support) {
      C.attr_r.push_back(a.first), C.attr_v.push_back(a.second);
      ents.insert(es.begin(), es.end());
    }
  C.entities.assign(ents.begin(), ents.end());
  C.ext.resize(C.attr_r.size());
  C.attrs.resize(C.entities.size());
  C.rels.resize(C.entities.size());
  for (std::size_t a = 0; a < C.attr_r.size(); ++a)
    for (const auto& e : by_attr[{C.attr_r[a], C.attr_v[a]}]) {
      const int id = C.entity(e);
      C.ext[a].push_back(id);
      C.attrs[static_cast<std::size_t>(id)].push_back(static_cast<int>(a));
    }
  for (std::size_t e = 0; e < C.entities.size(); ++e) {
    std::sort(C.attrs[e].begin(), C.attrs[e].end());
    const auto& src = all_rels ? *all_rels : own_rels;
    const auto it = src.find(C.entities[e]);
    if (it != src.end()) C.rels[e] = it->second;
  }
  for (auto& x : C.ext) std::sort(x.begin(), x.end());
  return C;
}

// ---------------------------------------------------------------------------
// G9: closed concepts by LCM
// ---------------------------------------------------------------------------

struct Concept {
  std::vector<int> intent;  // sorted attribute ids
  std::vector<int> extent;  // sorted entity ids
};

inline std::vector<int> intersect_sorted(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> out;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

/// The attributes every entity of `ext` holds (the closure of an extent). Empty for an empty extent.
inline std::vector<int> closure_of(const FormalContext& C, const std::vector<int>& ext) {
  if (ext.empty()) return {};
  std::vector<int> I = C.attrs[static_cast<std::size_t>(ext[0])];
  for (std::size_t k = 1; k < ext.size() && !I.empty(); ++k) I = intersect_sorted(I, C.attrs[static_cast<std::size_t>(ext[k])]);
  return I;
}

/// Every closed attribute set with a non-empty intent and an extent of at least `min_support`, by LCM's
/// prefix-preserving closure extension: each closed set is reached exactly once. Stops after `max_concepts`
/// (then *truncated is set). Order: depth-first by attribute id.
inline std::vector<Concept> closed_concepts(const FormalContext& C, double min_support, std::size_t max_concepts = 500000,
                                            bool* truncated = nullptr) {
  std::vector<Concept> out;
  if (truncated) *truncated = false;
  const int m = static_cast<int>(C.attr_r.size());
  std::vector<int> all(C.entities.size());
  for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int>(i);
  const std::vector<int> root = closure_of(C, all);
  bool stop = false;
  std::function<void(const std::vector<int>&, const std::vector<int>&, int)> rec = [&](const std::vector<int>& P, const std::vector<int>& E, int core) {
    for (int i = core + 1; i < m && !stop; ++i) {
      if (std::binary_search(P.begin(), P.end(), i)) continue;
      std::vector<int> E2 = intersect_sorted(E, C.ext[static_cast<std::size_t>(i)]);
      if (static_cast<double>(E2.size()) < min_support) continue;
      std::vector<int> Q = closure_of(C, E2);
      // prefix-preserving: Q and P agree on every attribute below i
      bool ok = true;
      std::vector<int>::const_iterator qi = Q.cbegin(), pi = P.cbegin();
      while (true) {
        const bool qe = qi == Q.end() || *qi >= i, pe = pi == P.end() || *pi >= i;
        if (qe && pe) break;
        if (qe != pe || *qi != *pi) { ok = false; break; }
        ++qi, ++pi;
      }
      if (!ok) continue;
      out.push_back({Q, E2});
      if (out.size() >= max_concepts) {
        stop = true;
        if (truncated) *truncated = true;
        return;
      }
      rec(Q, E2, i);
    }
  };
  if (!root.empty() && static_cast<double>(all.size()) >= min_support) out.push_back({root, all});
  rec(root, all, -1);
  return out;
}

/// |a ∩ b| / |a ∪ b| for sorted id lists, where b's full size is `b_size` (members of b outside the context count in
/// the union, never in the intersection).
inline double jaccard(const std::vector<int>& a, const std::vector<int>& b_present, double b_size) {
  const double in = static_cast<double>(intersect_sorted(a, b_present).size());
  const double un = static_cast<double>(a.size()) + b_size - in;
  return un > 0 ? in / un : 0.0;
}
inline double jaccard(const std::vector<int>& a, const std::vector<int>& b) { return jaccard(a, b, static_cast<double>(b.size())); }

/// Candidate classes (G9's output): an intent of at least 2 attributes, an extent of at least `min_support`, and a
/// Jaccard below `dup` with every single attribute's extent (a class the store already names). Indices into `cs`.
inline std::vector<std::size_t> candidate_classes(const FormalContext& C, const std::vector<Concept>& cs, double min_support = 10.0, double dup = 0.9) {
  std::vector<std::size_t> out;
  for (std::size_t k = 0; k < cs.size(); ++k) {
    const Concept& c = cs[k];
    if (c.intent.size() < 2 || static_cast<double>(c.extent.size()) < min_support) continue;
    bool named = false;
    // only an attribute that shares an entity with the class can reach the bar; the intent's own attributes are supersets
    std::set<int> near;
    for (int e : c.extent)
      for (int a : C.attrs[static_cast<std::size_t>(e)]) near.insert(a);
    for (int a : near)
      if (jaccard(c.extent, C.ext[static_cast<std::size_t>(a)]) >= dup) { named = true; break; }
    if (!named) out.push_back(k);
  }
  return out;
}

// ---------------------------------------------------------------------------
// G9: class rules and DISCOVERY P5's information-gain test
// ---------------------------------------------------------------------------

struct ClassRule {
  std::size_t cls = 0;  // index into the concept list
  int head = -1;            // attribute b
  double support = 0, pca_body = 0, baseline = 0, p_value = 1.0;
  double hid_support = 0, hid_pca = 0;  // T3: among the members in the hidden third
  bool kept = false;
  double pca_conf() const { return pca_body > 0 ? support / pca_body : 0.0; }
};

struct ClassRuleConfig {
  double min_support = 10.0, min_pca = 0.5, q = 0.1;
};

/// PCA confidence of the single-attribute rule i => b over the whole context: |ext i ∩ ext b| over the members of
/// ext i that hold b's relation.
inline double single_pca(const FormalContext& C, int i, int b) {
  const auto& Ei = C.ext[static_cast<std::size_t>(i)];
  const std::string& rb = C.attr_r[static_cast<std::size_t>(b)];
  double body = 0;
  for (int e : Ei) body += C.rels[static_cast<std::size_t>(e)].count(rb) ? 1.0 : 0.0;
  const double sup = static_cast<double>(intersect_sorted(Ei, C.ext[static_cast<std::size_t>(b)]).size());
  return body > 0 ? sup / body : 0.0;
}

/// Every class rule C => b over the candidate classes `classes` with support >= min_support and PCA >= min_pca, its
/// p-value against the best single-attribute rule inside C's intent, and BH at q over the batch. `hidden(e)` marks
/// the entities of T3's hidden third.
inline std::vector<ClassRule> class_rules(const FormalContext& C, const std::vector<Concept>& cs, const std::vector<std::size_t>& classes,
                                          const ClassRuleConfig& cfg = {},
                                          const std::function<bool(const std::string&)>& hidden = gen_hidden_third) {
  std::vector<ClassRule> out;
  std::map<std::pair<int, int>, double> single;  // (i, b) -> PCA confidence
  for (std::size_t k : classes) {
    const Concept& c = cs[k];
    std::map<int, double> cnt;  // b -> members holding it
    for (int e : c.extent)
      for (int a : C.attrs[static_cast<std::size_t>(e)])
        if (!std::binary_search(c.intent.begin(), c.intent.end(), a)) cnt[a] += 1.0;
    for (const auto& [b, sup] : cnt) {
      if (sup < cfg.min_support) continue;
      const std::string& rb = C.attr_r[static_cast<std::size_t>(b)];
      double body = 0, hs = 0, hb = 0;
      const auto& Eb = C.ext[static_cast<std::size_t>(b)];
      for (int e : c.extent) {
        const bool has_r = C.rels[static_cast<std::size_t>(e)].count(rb) > 0;
        body += has_r ? 1.0 : 0.0;
        if (hidden(C.entities[static_cast<std::size_t>(e)])) {
          hb += has_r ? 1.0 : 0.0;
          hs += std::binary_search(Eb.begin(), Eb.end(), e) ? 1.0 : 0.0;
        }
      }
      if (body <= 0 || sup / body < cfg.min_pca) continue;
      double base = 0;
      for (int i : c.intent) {
        auto it = single.find({i, b});
        if (it == single.end()) it = single.emplace(std::make_pair(i, b), single_pca(C, i, b)).first;
        base = std::max(base, it->second);
      }
      ClassRule R;
      R.cls = k, R.head = b, R.support = sup, R.pca_body = body, R.baseline = base;
      R.p_value = binomial_upper_tail(sup, body, base);
      R.hid_support = hs, R.hid_pca = hb;
      out.push_back(R);
    }
  }
  std::vector<double> p;
  for (const auto& R : out) p.push_back(R.p_value);
  const auto bh = benjamini_hochberg(p, cfg.q);
  for (std::size_t i = 0; i < out.size(); ++i) out[i].kept = bh.reject[i];
  return out;
}

/// T3 for a class rule (DISCOVERY 7.1): its precision on the hidden third of its members.
inline TestOutcome class_rule_t3(const ClassRule& R) { return rule_holdout_test(R.hid_support, R.hid_pca); }

// ---------------------------------------------------------------------------
// G11: open relations
// ---------------------------------------------------------------------------

/// The connector of a core key ("[S] was educated at [O]"): the slot order and at most `n` words next to [O], after
/// articles next to [O] are dropped. "[S] was educated at the [O]" -> "S ... educated at O"; "[O] , the capital of [S]"
/// -> "O , capital ... S". A core with no word between its slots gives "".
inline std::string connector_of(const std::string& core, std::size_t n = 2) {
  std::vector<std::string> t;
  for (std::size_t a = 0; a < core.size();) {
    std::size_t b = core.find(' ', a);
    if (b == std::string::npos) b = core.size();
    if (b > a) t.push_back(core.substr(a, b - a));
    a = b + 1;
  }
  if (t.size() < 3) return "";
  const bool o_last = t.back() == "[O]";
  std::vector<std::string> mid(t.begin() + 1, t.end() - 1);
  auto article = [](const std::string& w) { return w == "the" || w == "a" || w == "an"; };
  if (o_last)
    while (!mid.empty() && article(mid.back())) mid.pop_back();
  else
    while (!mid.empty() && article(mid.front())) mid.erase(mid.begin());
  if (mid.empty()) return "";
  std::vector<std::string> w;
  if (o_last) w.assign(mid.end() - static_cast<std::ptrdiff_t>(std::min(n, mid.size())), mid.end());
  else w.assign(mid.begin(), mid.begin() + static_cast<std::ptrdiff_t>(std::min(n, mid.size())));
  std::string s;
  for (const auto& x : w) s += (s.empty() ? "" : " ") + x;
  return o_last ? "S ... " + s + " O" : "O " + s + " ... S";
}

/// Whether a core key's middle holds a word with a letter (G11's phrase filter).
inline bool has_word(const std::string& core) {
  std::size_t a = core.find(']'), b = core.rfind('[');
  if (a == std::string::npos || b == std::string::npos || b <= a) return false;
  for (std::size_t i = a + 1; i < b; ++i) {
    const unsigned char ch = static_cast<unsigned char>(core[i]);
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch >= 0x80) return true;
  }
  return false;
}

struct OpenCluster {
  std::string key;
  std::vector<std::pair<std::string, std::string>> pairs;  // distinct (subject, object key), sorted
};

/// Clusters of (key, subject, object) records with at least `min_pairs` distinct pairs, sorted by size then key.
inline std::vector<OpenCluster> open_clusters(const std::vector<std::array<std::string, 3>>& recs, std::size_t min_pairs = 5) {
  std::map<std::string, std::set<std::pair<std::string, std::string>>> m;
  for (const auto& r : recs)
    if (!r[0].empty()) m[r[0]].insert({r[1], r[2]});
  std::vector<OpenCluster> out;
  for (auto& [k, ps] : m)
    if (ps.size() >= min_pairs) out.push_back({k, {ps.begin(), ps.end()}});
  std::stable_sort(out.begin(), out.end(), [](const OpenCluster& a, const OpenCluster& b) { return a.pairs.size() > b.pairs.size(); });
  return out;
}

struct ClusterGrade {
  std::string prop;          // the property held by the most gradable pairs (ties by name); "" when none holds one
  double gradable = 0, hits = 0;
  double precision() const { return gradable > 0 ? hits / gradable : 0.0; }
};

/// The cluster's majority property among its gradable pairs (`props(s, o)` lists what a pair holds; `gradable(s)` says
/// whether s can be graded), or the precision of a fixed property `fixed` when it is given.
inline ClusterGrade grade_cluster(const OpenCluster& c,
                                  const std::function<std::vector<std::string>(const std::string&, const std::string&)>& props,
                                  const std::function<bool(const std::string&)>& gradable, const std::string& fixed = "") {
  ClusterGrade g;
  std::map<std::string, double> cnt;
  for (const auto& [s, o] : c.pairs) {
    if (!gradable(s)) continue;
    g.gradable += 1;
    std::set<std::string> ps;
    for (const auto& p : props(s, o)) ps.insert(p);
    for (const auto& p : ps) cnt[p] += 1;
  }
  if (!fixed.empty()) {
    g.prop = fixed;
    g.hits = cnt.count(fixed) ? cnt[fixed] : 0.0;
    return g;
  }
  for (const auto& [p, n] : cnt)
    if (n > g.hits) g.hits = n, g.prop = p;  // map order: ties go to the first name
  return g;
}

// ---------------------------------------------------------------------------
// S11d: the rediscovery tally and G11's rule for the date-link skip
// ---------------------------------------------------------------------------

/// One held-out class as read: J_new (best candidate class), J_named (best single attribute), J_kept (best kept class)
/// and the null's J_new.
struct ClassRead {
  double j_new = 0, j_named = 0, j_kept = 0, j_null = 0;
};

/// Counts over held-out classes at the bar (0.5): rediscovered (J_new), redundant (J_named), strict (rediscovered and
/// not redundant), rediscovered by kept classes, and the null.
struct RediscoveryTally {
  double n = 0, red = 0, redun = 0, strict = 0, kept = 0, null = 0;
  void add(const ClassRead& c, double bar = 0.5) {
    n += 1;
    red += c.j_new >= bar;
    redun += c.j_named >= bar;
    strict += c.j_new >= bar && c.j_named < bar;
    kept += c.j_kept >= bar;
    null += c.j_null >= bar;
  }
  /// S11a's gate: the rediscovery rate's Wilson lower bound is above the null rate's Wilson upper bound.
  bool gate_met() const { return n > 0 && wilson_lower(red, n) > wilson_upper(null, n); }
  /// S11d: the same for the strict count.
  bool strict_met() const { return n > 0 && wilson_lower(strict, n) > wilson_upper(null, n); }
};

/// S11d's fresh classes are dev when bit 21 of the caller's FNV-1a hash of r + "\x1f" + v is 0.
inline bool fresh_class_is_dev(std::uint64_t h) { return ((h >> 21) & 1) == 0; }

/// S11d's rule for G11's date-link skip: it becomes the default reader when its one test read gives at least `min_hold`
/// proposals that hold and a pooled precision of at least `min_precision` (S11a's test read without its date proposals:
/// 8 hold, 58 / 107).
inline bool adopt_g11_skip(double holds, double pooled_hits, double pooled_n, double min_hold = 8.0, double min_precision = 58.0 / 107.0) {
  return holds >= min_hold && pooled_n > 0 && pooled_hits / pooled_n >= min_precision;
}

}  // namespace cypha::eval::concepts
