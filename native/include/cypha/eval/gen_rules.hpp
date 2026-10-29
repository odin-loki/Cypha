// Generators G2 (rule induction) and G14 (contradiction resolution) for Cypha's discovery track (docs/DISCOVERY.md
// section 5; docs/reports/S8A_RULES_AND_CONTRADICTIONS.md).
//
// Header-only. Reads a TripleStore (hypothesis.hpp); standard library only.
//
//   G2   AMIE-style closed Horn rules with head r(X, Y): const r1(X, c1) => r(X, c2); same r1(X, Y) => r(X, Y);
//        inverse r1(Y, X) => r(X, Y); path r1(X|Z, Z|X) and r2(Z|Y, Y|Z) => r(X, Y) with the join node Z a subject
//        in at least one atom. Support, head coverage, standard and PCA confidence; a binomial p-value against the
//        head object's base rate (its share of r's facts); kept at support >= 10, PCA >= 0.5 and Benjamini-Hochberg over
//        the candidate batch.
//        A slot (s, r, ?) is ranked by the noisy-or (or the maximum) of the PCA confidences of the rules deriving each
//        value. consequences() gives what the kept rules derive from a hypothesis (test T2).
//   G14  For a slot holding two values that cannot both be literally right: granularity or qualifier (one place
//        contains the other), calendar or precision (two dates at most 13 days apart, or one year at two precisions),
//        time (a relation that changes), or source error (one value is wrong). The explanation with the fewest bits
//        whose precondition holds is chosen. Sense and identity are not modelled (a subject is a page title).
//
// Nothing here answers a question. Rules and their predictions are hypotheses for Cypha's own store (PLAN rule 10).
//
// Author: Odin Loch

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cypha/eval/hypothesis.hpp"
#include "cypha/eval/lci.hpp"

namespace cypha::eval {

/// FNV-1a 64 of a string (the hash pattern_learn splits subjects with).
inline std::uint64_t gen_fnv1a(const std::string& s) {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) h ^= c, h *= 1099511628211ULL;
  return h;
}

/// T3's hidden third of a rule's subjects: (FNV-1a >> 13) mod 3 == 1.
inline bool gen_hidden_third(const std::string& subject) { return ((gen_fnv1a(subject) >> 13) % 3) == 1; }

/// P(Bin(n, p) >= k), by summing log-space terms from k up (exact; terms below 1e-17 of the largest are dropped).
inline double binomial_upper_tail(double k, double n, double p) {
  if (k <= 0) return 1.0;
  if (k > n) return 0.0;
  if (p <= 0.0) return 0.0;
  if (p >= 1.0) return 1.0;
  const double lp = std::log(p), lq = std::log1p(-p), lgn = std::lgamma(n + 1.0);
  auto term = [&](double i) { return lgn - std::lgamma(i + 1.0) - std::lgamma(n - i + 1.0) + i * lp + (n - i) * lq; };
  double mx = -1e300, sum = 0.0;
  std::vector<double> t;
  for (double i = k; i <= n; i += 1.0) {
    const double v = term(i);
    if (v > mx) mx = v;
    t.push_back(v);
    if (v < mx - 40.0 && i > n * p) break;  // past the mode and below e^-40 of the largest
  }
  for (double v : t) sum += std::exp(v - mx);
  return std::min(1.0, std::exp(mx) * sum);
}

// ---------------------------------------------------------------------------
// G2: rules
// ---------------------------------------------------------------------------

enum class RuleKind { Const, Same, Inverse, Path };

inline const char* rule_kind_name(RuleKind k) {
  switch (k) {
    case RuleKind::Const: return "const";
    case RuleKind::Same: return "same";
    case RuleKind::Inverse: return "inverse";
    case RuleKind::Path: return "path";
  }
  return "?";
}

struct Rule {
  RuleKind kind = RuleKind::Const;
  std::string head;    // r of the head r(X, Y)
  std::string head_o;  // const: the head's constant c2
  std::string r1, c1;  // const: the body r1(X, c1); same, inverse, path: the body's (first) relation
  std::string r2;      // path: the second body relation
  bool inv1 = false;   // path: atom 1 is r1(Z, X) when set, else r1(X, Z)
  bool inv2 = false;   // path: atom 2 is r2(Y, Z) when set, else r2(Z, Y)
  double support = 0, body = 0, pca_body = 0, head_n = 0;
  double p0 = 0, p_value = 1.0;
  double hid_support = 0, hid_pca = 0;  // T3: support and PCA body pairs among the hidden third of subjects
  bool kept = false;

  double std_conf() const { return body > 0 ? support / body : 0.0; }
  double pca_conf() const { return pca_body > 0 ? support / pca_body : 0.0; }
  double head_coverage() const { return head_n > 0 ? support / head_n : 0.0; }

  std::string text() const {
    switch (kind) {
      case RuleKind::Const: return r1 + "(X, " + c1 + ") => " + head + "(X, " + head_o + ")";
      case RuleKind::Same: return r1 + "(X, Y) => " + head + "(X, Y)";
      case RuleKind::Inverse: return r1 + "(Y, X) => " + head + "(X, Y)";
      case RuleKind::Path:
        return (inv1 ? r1 + "(Z, X)" : r1 + "(X, Z)") + " & " + (inv2 ? r2 + "(Y, Z)" : r2 + "(Z, Y)") + " => " + head + "(X, Y)";
    }
    return "?";
  }
};

struct RuleMineConfig {
  double min_support = 10.0, min_pca = 0.5, q = 0.1;
  bool const_rules = true, var_rules = true;
  double max_join = 20000.0;  // path: a join node whose body pairs would exceed this is skipped (a hub)
};

/// Indexes over a TripleStore for mining and applying rules. Strings are interned; ids follow first sight.
class RuleIndex {
 public:
  explicit RuleIndex(const TripleStore& st) : st_(st) {
    std::vector<std::string> subj;
    for (const auto& [s, fs] : st.subjects()) subj.push_back(s);
    std::sort(subj.begin(), subj.end());
    for (const auto& s : subj) {
      subjects_.push_back(s);
      for (const auto& f : st.facts_of(s)) {
        vals_[key(s, f.r)].push_back(f.o);
        in_[f.o].push_back({s, f.r});
        rels_[s].insert(f.r);
        relations_.insert(f.r);
      }
    }
  }

  const TripleStore& store() const { return st_; }
  const std::vector<std::string>& subjects() const { return subjects_; }  // sorted
  const std::set<std::string>& relations() const { return relations_; }
  const std::vector<std::string>& values(const std::string& s, const std::string& r) const {
    static const std::vector<std::string> none;
    auto it = vals_.find(key(s, r));
    return it == vals_.end() ? none : it->second;
  }
  /// Facts whose object is o: (subject, relation).
  const std::vector<std::pair<std::string, std::string>>& incoming(const std::string& o) const {
    static const std::vector<std::pair<std::string, std::string>> none;
    auto it = in_.find(o);
    return it == in_.end() ? none : it->second;
  }
  const std::set<std::string>& relations_of(const std::string& s) const {
    static const std::set<std::string> none;
    auto it = rels_.find(s);
    return it == rels_.end() ? none : it->second;
  }
  /// P(o | r): the share of r's facts with object o (the null of a rule's p-value).
  double base_rate(const std::string& r, const std::string& o) const {
    const double n = static_cast<double>(st_.count(r));
    return n > 0 ? static_cast<double>(st_.count(r, o)) / n : 0.0;
  }

 private:
  static std::string key(const std::string& a, const std::string& b) { return a + '\x1f' + b; }
  const TripleStore& st_;
  std::vector<std::string> subjects_;
  std::set<std::string> relations_;
  std::unordered_map<std::string, std::vector<std::string>> vals_;
  std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> in_;
  std::unordered_map<std::string, std::set<std::string>> rels_;
};

namespace gen_detail {

struct Cnt {
  double support = 0, pca = 0, hid_support = 0, hid_pca = 0, p0_sum = 0;
};

inline std::string k2(const std::string& a, const std::string& b) { return a + '\x1f' + b; }
inline std::string k3(const std::string& a, const std::string& b, const std::string& c) { return a + '\x1f' + b + '\x1f' + c; }

// Adds a body pair (x, y) of a variable rule shape `sig` to the counts of every head relation x holds.
inline void count_pair(const RuleIndex& ix, std::unordered_map<std::string, Cnt>& m, const std::string& sig, const std::string& x,
                       const std::string& y) {
  const bool hid = gen_hidden_third(x);
  for (const auto& r : ix.relations_of(x)) {
    Cnt& c = m[k2(sig, r)];
    c.pca += 1, c.p0_sum += ix.base_rate(r, y);
    if (hid) c.hid_pca += 1;
    if (ix.store().has(x, r, y)) {
      c.support += 1;
      if (hid) c.hid_support += 1;
    }
  }
}

}  // namespace gen_detail

/// Mines G2's rules. Returns every candidate with support >= min_support (kept ones flagged), sorted by kind, then PCA
/// confidence (descending), then text.
inline std::vector<Rule> mine_rules(const RuleIndex& ix, const RuleMineConfig& cfg = {}) {
  using gen_detail::Cnt;
  using gen_detail::k2;
  using gen_detail::k3;
  const TripleStore& st = ix.store();
  std::vector<Rule> out;
  if (cfg.const_rules) {
    // r1(X, c1) => r(X, c2): over subjects x holding r, every body fact (r1, c1) of x and every r value c2 of x.
    std::unordered_map<std::string, Cnt> both;        // r1 c1 r c2
    std::unordered_map<std::string, Cnt> pca;         // r1 c1 r
    for (const auto& x : ix.subjects()) {
      const bool hid = gen_hidden_third(x);
      const auto& fs = st.facts_of(x);
      for (const auto& r : ix.relations_of(x)) {
        const auto& vs = ix.values(x, r);
        for (const auto& b : fs) {
          Cnt& p = pca[k3(b.r, b.o, r)];
          p.pca += 1;
          if (hid) p.hid_pca += 1;
          for (const auto& c2 : vs) {
            if (b.r == r && b.o == c2) continue;
            Cnt& c = both[k3(b.r, b.o, r) + '\x1f' + c2];
            c.support += 1;
            if (hid) c.hid_support += 1;
          }
        }
      }
    }
    for (const auto& [k, c] : both) {
      if (c.support < cfg.min_support) continue;
      const std::size_t a = k.find('\x1f'), b = k.find('\x1f', a + 1), d = k.find('\x1f', b + 1);
      Rule R;
      R.kind = RuleKind::Const;
      R.r1 = k.substr(0, a), R.c1 = k.substr(a + 1, b - a - 1), R.head = k.substr(b + 1, d - b - 1), R.head_o = k.substr(d + 1);
      const Cnt& p = pca[k.substr(0, d)];
      R.support = c.support, R.hid_support = c.hid_support, R.pca_body = p.pca, R.hid_pca = p.hid_pca;
      R.body = static_cast<double>(st.subjects_with(R.r1, R.c1).size());
      R.head_n = static_cast<double>(st.count(R.head, R.head_o));  // the head atom r(X, c2)
      R.p0 = ix.base_rate(R.head, R.head_o);
      out.push_back(R);
    }
  }
  if (cfg.var_rules) {
    std::unordered_map<std::string, Cnt> m;  // shape \x1f head
    std::unordered_map<std::string, double> body;
    // same: r1(X, Y) => r(X, Y); inverse: r1(Y, X) => r(X, Y). Body pairs are facts, so they are distinct.
    for (const auto& x : ix.subjects())
      for (const auto& f : st.facts_of(x)) {
        body["S\x1f" + f.r] += 1, body["I\x1f" + f.r] += 1;
        gen_detail::count_pair(ix, m, "S\x1f" + f.r, x, f.o);    // (x, o) for same
        gen_detail::count_pair(ix, m, "I\x1f" + f.r, f.o, x);    // (o, x) for inverse
      }
    // path: for each join node Z, atom 1 from (X, r1, Z) [dir 0] or (Z, r1, X) [dir 1], atom 2 from (Z, r2, Y) [dir 0]
    // or (Y, r2, Z) [dir 1]; combinations (0, 0), (1, 0), (1, 1) (the join is a subject in at least one atom).
    std::unordered_map<std::string, std::set<std::pair<std::string, std::string>>> pairs;
    std::set<std::string> nodes;
    for (const auto& x : ix.subjects()) nodes.insert(x);
    for (const auto& z : nodes) {
      std::vector<std::pair<std::string, std::string>> a0, a1, b0, b1;  // (other end, relation)
      for (const auto& [s, r] : ix.incoming(z)) a0.push_back({s, r}), b1.push_back({s, r});
      for (const auto& f : st.facts_of(z)) a1.push_back({f.o, f.r}), b0.push_back({f.o, f.r});
      auto join = [&](const std::vector<std::pair<std::string, std::string>>& A, const std::vector<std::pair<std::string, std::string>>& B,
                      int d1, int d2) {
        if (static_cast<double>(A.size()) * static_cast<double>(B.size()) > cfg.max_join) return;
        for (const auto& [xa, ra] : A)
          for (const auto& [yb, rb] : B) {
            if (xa == yb || xa == z || yb == z) continue;
            pairs["P\x1f" + ra + '\x1f' + std::to_string(d1) + '\x1f' + rb + '\x1f' + std::to_string(d2)].insert({xa, yb});
          }
      };
      join(a0, b0, 0, 0);
      join(a1, b0, 1, 0);
      join(a1, b1, 1, 1);
    }
    for (const auto& [sig, ps] : pairs) {
      body[sig] = static_cast<double>(ps.size());
      for (const auto& [x, y] : ps) gen_detail::count_pair(ix, m, sig, x, y);
    }
    for (const auto& [k, c] : m) {
      if (c.support < cfg.min_support) continue;
      const std::size_t h = k.rfind('\x1f');
      const std::string sig = k.substr(0, h);
      Rule R;
      R.head = k.substr(h + 1);
      if (sig[0] == 'S' || sig[0] == 'I') {
        R.kind = sig[0] == 'S' ? RuleKind::Same : RuleKind::Inverse;
        R.r1 = sig.substr(2);
        if (R.kind == RuleKind::Same && R.r1 == R.head) continue;
      } else {
        R.kind = RuleKind::Path;
        std::vector<std::string> f;
        for (std::size_t a = 2;;) {
          std::size_t b = sig.find('\x1f', a);
          f.push_back(sig.substr(a, b == std::string::npos ? std::string::npos : b - a));
          if (b == std::string::npos) break;
          a = b + 1;
        }
        R.r1 = f[0], R.inv1 = f[1] == "1", R.r2 = f[2], R.inv2 = f[3] == "1";
      }
      R.support = c.support, R.pca_body = c.pca, R.hid_support = c.hid_support, R.hid_pca = c.hid_pca;
      R.body = body[sig];
      R.head_n = static_cast<double>(st.count(R.head));
      R.p0 = c.pca > 0 ? c.p0_sum / c.pca : 0.0;
      out.push_back(R);
    }
  }
  // p-values, then BH over the batch, then the PCA guard.
  std::vector<double> p;
  for (auto& R : out) {
    R.p_value = binomial_upper_tail(R.support, R.pca_body, R.p0);
    p.push_back(R.p_value);
  }
  const auto bh = benjamini_hochberg(p, cfg.q);
  for (std::size_t i = 0; i < out.size(); ++i) out[i].kept = bh.reject[i] && out[i].pca_conf() >= cfg.min_pca;
  std::sort(out.begin(), out.end(), [](const Rule& a, const Rule& b) {
    if (a.kind != b.kind) return static_cast<int>(a.kind) < static_cast<int>(b.kind);
    if (a.pca_conf() != b.pca_conf()) return a.pca_conf() > b.pca_conf();
    return a.text() < b.text();
  });
  return out;
}

/// The kept rules, indexed for applying them.
class RuleSet {
 public:
  RuleSet(const RuleIndex& ix, const std::vector<Rule>& rules, bool noisy_or = true) : ix_(ix), noisy_or_(noisy_or) {
    for (const auto& R : rules)
      if (R.kept) rules_.push_back(R);
    for (std::size_t i = 0; i < rules_.size(); ++i) {
      const Rule& R = rules_[i];
      if (R.kind == RuleKind::Const) by_body_[R.r1 + '\x1f' + R.c1].push_back(i);
      else by_head_[R.head].push_back(i);
      by_r1_[R.r1].push_back(i);
      if (R.kind == RuleKind::Path) by_r2_[R.r2].push_back(i);
    }
  }
  const std::vector<Rule>& rules() const { return rules_; }

  struct Cand {
    std::string o;
    double score = 0;
    int best = -1;  // the rule with the highest PCA confidence deriving it
    int n_rules = 0;
  };

  /// Values the kept rules derive for (s, r), with their combined scores (noisy-or or maximum of PCA confidences;
  /// each rule counted once per value), sorted by score (descending), then value.
  std::vector<Cand> candidates(const std::string& s, const std::string& r) const {
    std::unordered_map<std::string, std::vector<int>> got;
    const TripleStore& st = ix_.store();
    for (const auto& f : st.facts_of(s)) {
      auto it = by_body_.find(f.r + '\x1f' + f.o);
      if (it == by_body_.end()) continue;
      for (std::size_t i : it->second)
        if (rules_[i].head == r) got[rules_[i].head_o].push_back(static_cast<int>(i));
    }
    auto hit = by_head_.find(r);
    if (hit != by_head_.end())
      for (std::size_t i : hit->second) {
        const Rule& R = rules_[i];
        const int ii = static_cast<int>(i);
        if (R.kind == RuleKind::Same)
          for (const auto& y : ix_.values(s, R.r1)) got[y].push_back(ii);
        else if (R.kind == RuleKind::Inverse) {
          for (const auto& [y, ry] : ix_.incoming(s))
            if (ry == R.r1) got[y].push_back(ii);
        } else {
          for (const auto& z : atom1_from_x(R, s))
            for (const auto& y : atom2_from_z(R, z))
              if (y != s && y != z) got[y].push_back(ii);
        }
      }
    std::vector<Cand> out;
    for (auto& [o, rs] : got) {
      std::sort(rs.begin(), rs.end());
      rs.erase(std::unique(rs.begin(), rs.end()), rs.end());
      Cand c;
      c.o = o, c.n_rules = static_cast<int>(rs.size());
      double keep = 1.0, mx = 0.0;
      for (int i : rs) {
        const double p = rules_[static_cast<std::size_t>(i)].pca_conf();
        keep *= 1.0 - p;
        if (p > mx) mx = p, c.best = i;
      }
      c.score = noisy_or_ ? 1.0 - keep : mx;
      out.push_back(c);
    }
    std::sort(out.begin(), out.end(), [](const Cand& a, const Cand& b) { return a.score != b.score ? a.score > b.score : a.o < b.o; });
    return out;
  }

  struct Consequence {
    std::string s, r, o;
    double pca = 0;
    int rule = -1;
  };

  /// T2: what the kept rules derive if h = (x, r, y) holds, with h filling one body atom (a path rule's other atom from
  /// the store). h itself is left out.
  std::vector<Consequence> consequences(const std::string& x, const std::string& r, const std::string& y) const {
    std::vector<Consequence> out;
    auto add = [&](const std::string& s2, const std::string& r2, const std::string& o2, std::size_t i) {
      if (s2 == x && r2 == r && o2 == y) return;
      if (s2.empty() || o2.empty() || s2 == o2) return;
      out.push_back({s2, r2, o2, rules_[i].pca_conf(), static_cast<int>(i)});
    };
    auto it = by_body_.find(r + '\x1f' + y);
    if (it != by_body_.end())
      for (std::size_t i : it->second) add(x, rules_[i].head, rules_[i].head_o, i);
    auto jt = by_r1_.find(r);
    if (jt != by_r1_.end())
      for (std::size_t i : jt->second) {
        const Rule& R = rules_[i];
        if (R.kind == RuleKind::Same) add(x, R.head, y, i);
        else if (R.kind == RuleKind::Inverse) add(y, R.head, x, i);
        else if (R.kind == RuleKind::Path) {
          // h is atom 1: r1(X, Z) = (x, y) gives X = x, Z = y; r1(Z, X) = (x, y) gives Z = x, X = y.
          const std::string X = R.inv1 ? y : x, Z = R.inv1 ? x : y;
          for (const auto& Y : atom2_from_z(R, Z))
            if (Y != X) add(X, R.head, Y, i);
        }
      }
    auto kt = by_r2_.find(r);
    if (kt != by_r2_.end())
      for (std::size_t i : kt->second) {
        const Rule& R = rules_[i];
        // h is atom 2: r2(Z, Y) = (x, y) gives Z = x, Y = y; r2(Y, Z) = (x, y) gives Y = x, Z = y.
        const std::string Z = R.inv2 ? y : x, Y = R.inv2 ? x : y;
        for (const auto& X : atom1_from_z(R, Z))
          if (X != Y) add(X, R.head, Y, i);
      }
    return out;
  }

 private:
  // Path atoms: Z from X (atom 1), Y from Z (atom 2), X from Z (atom 1, read backwards).
  std::vector<std::string> atom1_from_x(const Rule& R, const std::string& x) const {
    std::vector<std::string> z;
    if (!R.inv1) z = ix_.values(x, R.r1);
    else
      for (const auto& [s, rr] : ix_.incoming(x))
        if (rr == R.r1) z.push_back(s);
    return z;
  }
  std::vector<std::string> atom2_from_z(const Rule& R, const std::string& z) const {
    std::vector<std::string> y;
    if (!R.inv2) y = ix_.values(z, R.r2);
    else
      for (const auto& [s, rr] : ix_.incoming(z))
        if (rr == R.r2) y.push_back(s);
    return y;
  }
  std::vector<std::string> atom1_from_z(const Rule& R, const std::string& z) const {
    std::vector<std::string> x;
    if (R.inv1) x = ix_.values(z, R.r1);
    else
      for (const auto& [s, rr] : ix_.incoming(z))
        if (rr == R.r1) x.push_back(s);
    return x;
  }

  const RuleIndex& ix_;
  bool noisy_or_;
  std::vector<Rule> rules_;
  std::unordered_map<std::string, std::vector<std::size_t>> by_body_, by_head_, by_r1_, by_r2_;
};

/// T3 for a rule (DISCOVERY 7.1): its precision on the hidden third of its subjects, by rule_holdout_test.
inline TestOutcome rule_t3(const Rule& R) { return rule_holdout_test(R.hid_support, R.hid_pca); }

// ---------------------------------------------------------------------------
// G14: contradiction resolution
// ---------------------------------------------------------------------------

enum class HiddenVar { Granularity, Calendar, Time, SourceError };

inline const char* hidden_var_name(HiddenVar v) {
  switch (v) {
    case HiddenVar::Granularity: return "granularity";
    case HiddenVar::Calendar: return "calendar";
    case HiddenVar::Time: return "time";
    case HiddenVar::SourceError: return "source_error";
  }
  return "?";
}

/// A slot holding two values that cannot both be literally right.
struct ContestedSlot {
  std::string s, r;
  std::string a, b;        // the two values, in the store's order
  double sources_a = 1, sources_b = 1;
};

/// What G14 needs to know about the slot, from the caller's relation table and stores.
struct ContestFacts {
  bool is_date = false;          // a date or year relation
  bool time_varying = false;     // the relation can change over time
  bool a_contains_b = false;     // (places) a contains b, through the Known store's located_in, country or capital chain
  bool b_contains_a = false;
  int day_gap = -1;              // dates: days apart when both have a day, else -1
  bool same_year = false;        // dates: the same year at different precision
  int t4_a = 0, t4_b = 0;        // T4 on the subject's page: 1 found, 0 not found, -1 not applicable
};

struct G14Explanation {
  HiddenVar kind = HiddenVar::SourceError;
  int wrong = -1;     // source error: 0 when a is judged wrong, 1 when b is
  double bits = 0.0;  // description length of the chosen explanation
  std::string why;
};

/// DISCOVERY G14: the explanation with the fewest bits whose precondition holds. Granularity and calendar cost 1 bit,
/// time log2(2) + 2, source error log2(2) + 3. For source error the wrong value is the one T4 does not find when it finds
/// the other, else the one with fewer sources, else the later one.
inline G14Explanation explain_contest(const ContestedSlot& c, const ContestFacts& f) {
  G14Explanation e;
  if (!f.is_date && (f.a_contains_b || f.b_contains_a)) {
    e.kind = HiddenVar::Granularity, e.bits = 1.0;
    e.why = f.a_contains_b ? c.a + " contains " + c.b : c.b + " contains " + c.a;
    return e;
  }
  if (f.is_date && (f.same_year || (f.day_gap >= 0 && f.day_gap <= 13))) {
    e.kind = HiddenVar::Calendar, e.bits = 1.0;
    e.why = f.same_year ? "one year at two precisions" : "dates " + std::to_string(f.day_gap) + " days apart";
    return e;
  }
  if (f.time_varying) {
    e.kind = HiddenVar::Time, e.bits = 3.0;
    e.why = "the relation changes over time";
    return e;
  }
  e.kind = HiddenVar::SourceError, e.bits = 4.0;
  if (f.t4_a == 1 && f.t4_b == 0) e.wrong = 1, e.why = "the page names " + c.a + ", not " + c.b;
  else if (f.t4_b == 1 && f.t4_a == 0) e.wrong = 0, e.why = "the page names " + c.b + ", not " + c.a;
  else if (c.sources_a != c.sources_b) e.wrong = c.sources_a > c.sources_b ? 1 : 0, e.why = "fewer sources";
  else e.wrong = 1, e.why = "the later value";
  return e;
}

}  // namespace cypha::eval
