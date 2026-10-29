// Generators G13 (anomaly explanation), G3 (abduction), G4 (structural analogy) and G5 (vector analogy) for Cypha's
// discovery track (docs/DISCOVERY.md section 5; docs/reports/S11C_ANALOGY_ANOMALY_ABDUCTION.md).
//
// Header-only. Reads a TripleStore (hypothesis.hpp) and G2's rules (gen_rules.hpp); standard library only.
//
//   G13  Values far from what their type allows: a death before a birth, a lifespan, spouse or succession gap in the
//        store's own tail, a date and a year that disagree, a capital in another country, a founder born after the
//        founding. Each anomaly gets the cheapest explanation (in bits) whose precondition holds: a lost qualifier,
//        a different entity of the same name, an error in one fact, or a genuinely unusual fact.
//   G3   For a fact no kept rule derives: the sets of missing facts that would let a kept rule derive it, ranked by
//        description length (bits of the added facts under the store, plus the rule's own bits).
//   G4   Analogs of a target that share at least three aligned relations (one with a shared value) carry over the roles
//        their values play: equal to another relation's value, at the end of a two-step path, or (weight lambda) as a
//        constant.
//   G5   Word-vector offsets: unit(t) + mean(unit(b) - unit(a)) over the relation's facts, scored by cosine.
//
// Nothing here answers a question. Everything it makes is a hypothesis for Cypha's own store (PLAN rule 10).
//
// Author: Odin Loch

#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cypha/eval/gen_rules.hpp"
#include "cypha/eval/hypothesis.hpp"

namespace cypha::eval::analogy {

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

struct FactRef {
  std::string s, r, o;
  bool operator==(const FactRef& b) const { return s == b.s && r == b.r && o == b.o; }
  bool operator<(const FactRef& b) const { return s != b.s ? s < b.s : r != b.r ? r < b.r : o < b.o; }
  std::string key() const { return s + '\x1f' + r + '\x1f' + o; }
};

/// One-sided upper tail of an empirical distribution, add-one: P(V >= x) = (#{v >= x} + 1) / (n + 1).
class EmpiricalTail {
 public:
  void add(double v) { v_.push_back(v), sorted_ = false; }
  std::size_t size() const { return v_.size(); }
  double upper_p(double x) const {
    sort();
    const auto it = std::lower_bound(v_.begin(), v_.end(), x);
    const double ge = static_cast<double>(v_.end() - it);
    return (ge + 1.0) / (static_cast<double>(v_.size()) + 1.0);
  }

 private:
  void sort() const {
    if (!sorted_) std::sort(v_.begin(), v_.end()), sorted_ = true;
  }
  mutable std::vector<double> v_;
  mutable bool sorted_ = true;
};

// ---------------------------------------------------------------------------
// G13: anomalies
// ---------------------------------------------------------------------------

enum class AnomalyKind { DeathBeforeBirth, LifespanTail, YearMismatch, CapitalElsewhere, FounderTiming, SpouseGap, SuccessionGap };

inline const char* anomaly_kind_name(AnomalyKind k) {
  switch (k) {
    case AnomalyKind::DeathBeforeBirth: return "A1_death_before_birth";
    case AnomalyKind::LifespanTail: return "A2_lifespan_tail";
    case AnomalyKind::YearMismatch: return "A3_year_mismatch";
    case AnomalyKind::CapitalElsewhere: return "A4_capital_elsewhere";
    case AnomalyKind::FounderTiming: return "A5_founder_timing";
    case AnomalyKind::SpouseGap: return "A6_spouse_gap";
    case AnomalyKind::SuccessionGap: return "A7_succession_gap";
  }
  return "?";
}

/// Impossible kinds have no tail: a real fact cannot make them, so "genuinely unusual" costs infinitely many bits.
inline bool anomaly_impossible(AnomalyKind k) {
  return k == AnomalyKind::DeathBeforeBirth || k == AnomalyKind::YearMismatch || k == AnomalyKind::CapitalElsewhere ||
         k == AnomalyKind::FounderTiming;
}

struct Anomaly {
  AnomalyKind kind = AnomalyKind::DeathBeforeBirth;
  std::vector<FactRef> facts;  // the facts it compares; for A4-A7 facts[0] is the link
  double tail_p = 0;           // the tail kinds' p-value (A2, A6, A7); 0 for the impossible kinds
  int link = -1;               // index of the link fact (A4-A7), else -1
  std::string linked;          // the entity the link points at (A4-A7)
  double value = 0;            // the quantity read (years)
};

struct AnomalyConfig {
  double tail_p = 0.005;       // one-sided tail p at or below which a tail kind fires
  double max_lifespan = 122;   // a lifespan above this fires A2 whatever the tail
  std::size_t min_tail_n = 30; // a tail kind needs at least this many values of its quantity
  double founder_death_slack = 1;  // a founder dead more than this many years before the founding fires A5
};

/// Reads years from values: returns false when the value has no year. The caller supplies its date parser.
using YearOf = std::function<bool(const std::string& value, int& year)>;

namespace detail {
inline bool is_birth(const std::string& r) { return r == "birth_year" || r == "birth_date"; }
inline bool is_death(const std::string& r) { return r == "death_year" || r == "death_date"; }
struct Dated {
  int year = 0;
  FactRef f;
};
inline std::vector<Dated> dated(const TripleStore& st, const std::string& s, bool birth, const YearOf& year_of) {
  std::vector<Dated> out;
  for (const auto& f : st.facts_of(s)) {
    if (birth ? !is_birth(f.r) : !is_death(f.r)) continue;
    int y = 0;
    if (year_of(f.o, y)) out.push_back({y, {s, f.r, f.o}});
  }
  std::sort(out.begin(), out.end(), [](const Dated& a, const Dated& b) { return a.year != b.year ? a.year < b.year : a.f < b.f; });
  return out;
}
inline bool first_birth(const TripleStore& st, const std::string& s, const YearOf& year_of, Dated& d) {
  const auto v = dated(st, s, true, year_of);
  if (v.empty()) return false;
  d = v.front();
  return true;
}
}  // namespace detail

/// Every anomaly of the store (A1-A7), in a fixed order: by subject, then kind.
inline std::vector<Anomaly> find_anomalies(const TripleStore& st, const YearOf& year_of, const AnomalyConfig& cfg = {}) {
  std::vector<std::string> subj;
  for (const auto& [s, fs] : st.subjects()) subj.push_back(s);
  std::sort(subj.begin(), subj.end());
  // the store's own distributions: lifespans, spouse gaps, succession gaps
  EmpiricalTail life, spouse, succ;
  for (const auto& s : subj) {
    const auto B = detail::dated(st, s, true, year_of), D = detail::dated(st, s, false, year_of);
    if (!B.empty() && !D.empty()) life.add(D.back().year - B.front().year);
    detail::Dated b0;
    if (!detail::first_birth(st, s, year_of, b0)) continue;
    for (const auto& f : st.facts_of(s)) {
      if (f.r != "spouse" && f.r != "predecessor" && f.r != "successor") continue;
      detail::Dated b1;
      if (f.o == s || !detail::first_birth(st, f.o, year_of, b1)) continue;
      (f.r == "spouse" ? spouse : succ).add(std::abs(b0.year - b1.year));
    }
  }
  std::vector<Anomaly> out;
  for (const auto& s : subj) {
    const auto B = detail::dated(st, s, true, year_of), D = detail::dated(st, s, false, year_of);
    // A1 and A2: the latest birth against the earliest death; the earliest birth against the latest death
    if (!B.empty() && !D.empty()) {
      if (D.front().year < B.back().year) {
        Anomaly a;
        a.kind = AnomalyKind::DeathBeforeBirth, a.facts = {B.back().f, D.front().f}, a.value = D.front().year - B.back().year;
        out.push_back(a);
      } else {
        const double span = D.back().year - B.front().year;
        const double p = life.upper_p(span);
        if ((life.size() >= cfg.min_tail_n && p <= cfg.tail_p) || span > cfg.max_lifespan) {
          Anomaly a;
          a.kind = AnomalyKind::LifespanTail, a.facts = {B.front().f, D.back().f}, a.tail_p = p, a.value = span;
          out.push_back(a);
        }
      }
    }
    // A3: a date's year against a year relation of the same event
    for (int ev = 0; ev < 2; ++ev) {
      const auto& V = ev == 0 ? B : D;
      const std::string yr = ev == 0 ? "birth_year" : "death_year", dt = ev == 0 ? "birth_date" : "death_date";
      for (const auto& a1 : V) {
        if (a1.f.r != dt) continue;
        bool done = false;
        for (const auto& a2 : V)
          if (a2.f.r == yr && a2.year != a1.year && !done) {
            Anomaly a;
            a.kind = AnomalyKind::YearMismatch, a.facts = {a2.f, a1.f}, a.value = a1.year - a2.year;
            out.push_back(a), done = true;
          }
        if (done) break;
      }
    }
    // A4: capital(C, X) and country(X, Y) with Y != C
    for (const auto& f : st.facts_of(s)) {
      if (f.r != "capital" || f.o == s) continue;
      for (const auto& g : st.facts_of(f.o))
        if (g.r == "country" && g.o != s && g.o != f.o) {
          Anomaly a;
          a.kind = AnomalyKind::CapitalElsewhere, a.facts = {{s, f.r, f.o}, {f.o, g.r, g.o}}, a.link = 0, a.linked = f.o;
          out.push_back(a);
          break;
        }
    }
    // A5: founder(O, P), founded(O) = y; P born after y, or dead more than the slack before y
    {
      int fy = 0;
      FactRef ff;
      bool has = false;
      for (const auto& f : st.facts_of(s))
        if (f.r == "founded" && !has && year_of(f.o, fy)) ff = {s, f.r, f.o}, has = true;
      if (has)
        for (const auto& f : st.facts_of(s)) {
          if (f.r != "founder" || f.o == s) continue;
          const auto PB = detail::dated(st, f.o, true, year_of), PD = detail::dated(st, f.o, false, year_of);
          if (!PB.empty() && PB.front().year > fy) {
            Anomaly a;
            a.kind = AnomalyKind::FounderTiming, a.facts = {{s, f.r, f.o}, ff, PB.front().f}, a.link = 0, a.linked = f.o;
            a.value = PB.front().year - fy;
            out.push_back(a);
          } else if (!PD.empty() && PD.back().year < fy - cfg.founder_death_slack) {
            Anomaly a;
            a.kind = AnomalyKind::FounderTiming, a.facts = {{s, f.r, f.o}, ff, PD.back().f}, a.link = 0, a.linked = f.o;
            a.value = PD.back().year - fy;
            out.push_back(a);
          }
        }
    }
    // A6 and A7: gaps in the store's own tails (each unordered pair once: the subject sorted first)
    detail::Dated b0;
    if (detail::first_birth(st, s, year_of, b0))
      for (const auto& f : st.facts_of(s)) {
        if (f.r != "spouse" && f.r != "predecessor" && f.r != "successor") continue;
        if (f.o == s) continue;
        detail::Dated b1;
        if (!detail::first_birth(st, f.o, year_of, b1)) continue;
        const bool sp = f.r == "spouse";
        const EmpiricalTail& T = sp ? spouse : succ;
        const double gap = std::abs(b0.year - b1.year);
        const double p = T.upper_p(gap);
        if (T.size() < cfg.min_tail_n || p > cfg.tail_p) continue;
        if (st.has(f.o, f.r, s) && f.o < s) continue;  // the pair is read from the other side
        const AnomalyKind k = sp ? AnomalyKind::SpouseGap : AnomalyKind::SuccessionGap;
        Anomaly a;
        a.kind = k, a.facts = {{s, f.r, f.o}, b0.f, b1.f}, a.link = 0, a.linked = f.o, a.tail_p = p, a.value = gap;
        out.push_back(a);
      }
  }
  return out;
}

enum class Verdict { Qualifier, DifferentEntity, Error, Unusual };

inline const char* verdict_name(Verdict v) {
  switch (v) {
    case Verdict::Qualifier: return "qualifier";
    case Verdict::DifferentEntity: return "different_entity";
    case Verdict::Error: return "error";
    case Verdict::Unusual: return "unusual";
  }
  return "?";
}

/// What the explanation of an anomaly may read: the caller's evidence, all optional.
struct AnomalyEvidence {
  YearOf year_of;
  std::function<double(const FactRef&)> sources;              // source rows supporting a fact (0 if unknown)
  std::function<int(const FactRef&)> page;                    // T4: +1 the subject's page states it, -1 states another, 0 unknown
  std::function<bool(const std::string& a, const std::string& b)> contains;  // a contains b (place chains)
  std::function<bool(const std::string& name)> ambiguous;     // a page "name (...)" exists besides it
};

struct AnomalyExplanation {
  Verdict verdict = Verdict::Error;
  int wrong = -1;     // the fact named wrong (error, different entity, an A1-A3 qualifier's unsigned year); -1 otherwise
  double bits = 0;
  double bits_error = 0, bits_unusual = 0;
  std::string why;
};

/// The registered order among the facts of an anomaly when sources and the page do not decide: the year relation over
/// the date (A3), the death over the birth (A1, A2), the link over the dates (A4-A7).
inline int registered_wrong(const Anomaly& a) {
  switch (a.kind) {
    case AnomalyKind::YearMismatch: return 0;           // facts[0] is the year relation
    case AnomalyKind::DeathBeforeBirth:
    case AnomalyKind::LifespanTail: return 1;           // facts[1] is the death
    default: return a.link >= 0 ? a.link : 0;
  }
}

/// Which fact of an anomaly is wrong: fewer source rows; then the one the page does not state when it states another
/// fact of the anomaly; then the registered order.
inline int name_wrong(const Anomaly& a, const AnomalyEvidence& ev, std::string* why = nullptr) {
  const std::size_t n = a.facts.size();
  if (ev.sources) {
    std::vector<double> src(n);
    for (std::size_t i = 0; i < n; ++i) src[i] = ev.sources(a.facts[i]);
    const double mn = *std::min_element(src.begin(), src.end());
    int cnt = 0, at = -1;
    for (std::size_t i = 0; i < n; ++i)
      if (src[i] == mn) ++cnt, at = static_cast<int>(i);
    if (cnt == 1) {
      if (why) *why = "fewer sources";
      return at;
    }
  }
  if (ev.page) {
    std::vector<int> pg(n);
    bool any_stated = false;
    for (std::size_t i = 0; i < n; ++i) pg[i] = ev.page(a.facts[i]), any_stated = any_stated || pg[i] > 0;
    if (any_stated) {
      int cnt = 0, at = -1;
      for (std::size_t i = 0; i < n; ++i)
        if (pg[i] <= 0) ++cnt, at = static_cast<int>(i);
      if (cnt == 1) {
        if (why) *why = "page states the others";
        return at;
      }
    }
  }
  if (why) *why = "registered order";
  return registered_wrong(a);
}

/// The cheapest explanation whose precondition holds (DISCOVERY G13; prices as G14's).
inline AnomalyExplanation explain_anomaly(const Anomaly& a, const AnomalyEvidence& ev) {
  AnomalyExplanation e;
  const double inf = std::numeric_limits<double>::infinity();
  e.bits_error = std::log2(static_cast<double>(std::max<std::size_t>(1, a.facts.size()))) + 3.0;
  e.bits_unusual = anomaly_impossible(a.kind) || a.tail_p <= 0 ? inf : -std::log2(a.tail_p);
  // qualifier: a lost BC sign (A1-A3), or a place inside the other (A4)
  bool qualifier = false;
  int flipped = -1;  // amendment 1 (dev): an A1-A3 qualifier names the fact whose BC sign it restores
  if ((a.kind == AnomalyKind::DeathBeforeBirth || a.kind == AnomalyKind::LifespanTail || a.kind == AnomalyKind::YearMismatch) && ev.year_of &&
      a.facts.size() == 2) {
    int y0 = 0, y1 = 0;
    if (ev.year_of(a.facts[0].o, y0) && ev.year_of(a.facts[1].o, y1))
      for (int flip = 0; flip < 2 && !qualifier; ++flip) {
        int b = y0, d = y1;
        int& y = flip == 0 ? b : d;
        if (std::abs(y) >= 1000 || y == 0) continue;
        y = -y;
        if (a.kind == AnomalyKind::YearMismatch) qualifier = b == d;
        else qualifier = d >= b && d - b <= 122;
        if (qualifier) flipped = flip;
      }
  }
  if (a.kind == AnomalyKind::CapitalElsewhere && ev.contains && a.facts.size() == 2) {
    const std::string& C = a.facts[0].s;
    const std::string& Y = a.facts[1].o;
    qualifier = ev.contains(C, Y) || ev.contains(Y, C);
  }
  const bool other_entity = a.link >= 0 && ev.ambiguous && ev.ambiguous(a.linked);
  struct Opt {
    Verdict v;
    double bits;
    bool ok;
  };
  const Opt opts[4] = {{Verdict::Qualifier, 1.0, qualifier},
                       {Verdict::DifferentEntity, 2.0, other_entity},
                       {Verdict::Error, e.bits_error, true},
                       {Verdict::Unusual, e.bits_unusual, std::isfinite(e.bits_unusual)}};
  double best = inf;
  for (const Opt& o : opts)
    if (o.ok && o.bits < best) best = o.bits, e.verdict = o.v;
  e.bits = best;
  if (e.verdict == Verdict::Error) e.wrong = name_wrong(a, ev, &e.why);
  else if (e.verdict == Verdict::DifferentEntity) e.wrong = a.link, e.why = "ambiguous name";
  else if (e.verdict == Verdict::Qualifier && flipped >= 0) e.wrong = flipped, e.why = "lost BC sign";
  else e.why = e.verdict == Verdict::Qualifier ? "qualifier" : "tail";
  return e;
}

// ---------------------------------------------------------------------------
// G3: abduction through G2's kept rules
// ---------------------------------------------------------------------------

struct Explanation {
  std::vector<FactRef> add;  // the missing facts (one, here)
  int rule = -1;             // index into RuleSet::rules()
  double bits = 0;           // sum of the added facts' bits plus the rule's -log2 PCA
  double support = 0;
};

/// Bits of a fact under the store: -log2 P(fact). The caller supplies it (S6's link predictor).
using FactBits = std::function<double(const FactRef&)>;

namespace detail {
inline std::vector<std::string> vals_or_in(const RuleIndex& ix, const std::string& node, const std::string& r, bool forward) {
  std::vector<std::string> out;
  if (forward) return ix.values(node, r);
  for (const auto& [s, rr] : ix.incoming(node))
    if (rr == r) out.push_back(s);
  return out;
}
}  // namespace detail

/// The explanations of (x, r, y) by the kept rules: each rule with head r and the one fact it still needs. Empty when a
/// rule already derives it with nothing added (an explained observation has no explanation to propose) or none can.
/// max_join bounds the join nodes read per path rule (a hub guard).
inline std::vector<Explanation> abduce(const RuleIndex& ix, const RuleSet& rs, const std::string& x, const std::string& r, const std::string& y,
                                       const FactBits& bits_of, std::size_t max_join = 5000) {
  std::vector<Explanation> out;
  const TripleStore& st = ix.store();
  const auto& R = rs.rules();
  auto propose = [&](std::size_t i, const FactRef& f) {
    if (f.s.empty() || f.o.empty() || f.s == f.o) return;
    if (st.has(f.s, f.r, f.o)) return;
    Explanation e;
    e.add = {f}, e.rule = static_cast<int>(i), e.support = R[i].support;
    e.bits = bits_of(f) - std::log2(std::max(1e-12, R[i].pca_conf()));
    out.push_back(std::move(e));
  };
  for (std::size_t i = 0; i < R.size(); ++i) {
    const Rule& Q = R[i];
    if (Q.head != r) continue;
    switch (Q.kind) {
      case RuleKind::Const:
        if (Q.head_o == y) propose(i, {x, Q.r1, Q.c1});
        break;
      case RuleKind::Same: propose(i, {x, Q.r1, y}); break;
      case RuleKind::Inverse: propose(i, {y, Q.r1, x}); break;
      case RuleKind::Path: {
        // atom 1 links X and Z: r1(X, Z), or r1(Z, X) when inv1; atom 2 links Z and Y: r2(Z, Y), or r2(Y, Z) when inv2
        const auto z1 = detail::vals_or_in(ix, x, Q.r1, !Q.inv1);  // Z with atom 1 in the store
        std::size_t n = 0;
        for (const auto& z : z1) {
          if (z == x || z == y || ++n > max_join) continue;
          propose(i, Q.inv2 ? FactRef{y, Q.r2, z} : FactRef{z, Q.r2, y});
        }
        const auto z2 = detail::vals_or_in(ix, y, Q.r2, Q.inv2);  // Z with atom 2 in the store
        n = 0;
        for (const auto& z : z2) {
          if (z == x || z == y || ++n > max_join) continue;
          propose(i, Q.inv1 ? FactRef{z, Q.r1, x} : FactRef{x, Q.r1, z});
        }
        break;
      }
    }
  }
  std::sort(out.begin(), out.end(), [](const Explanation& a, const Explanation& b) {
    if (a.bits != b.bits) return a.bits < b.bits;
    if (a.support != b.support) return a.support > b.support;
    return a.add.front() < b.add.front();
  });
  return out;
}

/// The distinct added facts in the order of their best explanation.
inline std::vector<std::pair<FactRef, double>> ranked_facts(const std::vector<Explanation>& ex) {
  std::vector<std::pair<FactRef, double>> out;
  std::set<FactRef> seen;
  for (const auto& e : ex)
    for (const auto& f : e.add)
      if (seen.insert(f).second) out.push_back({f, e.bits});
  return out;
}

/// Whether a kept rule derives (x, r, y) from the store as it is.
inline bool derivable(const RuleSet& rs, const std::string& x, const std::string& r, const std::string& y) {
  for (const auto& c : rs.candidates(x, r))
    if (c.o == y) return true;
  return false;
}

// ---------------------------------------------------------------------------
// G4: structural analogy
// ---------------------------------------------------------------------------

struct AnalogConfig {
  int min_aligned = 3;           // DISCOVERY's guard
  int min_identical = 1;         // at least one aligned relation shares a value
  std::size_t max_holders = 500; // a shared (relation, value) held by more subjects is not used to find analogs
  std::size_t keep = 20;         // analogs kept per slot
};

struct Analog {
  std::string s;
  double sys = 0;      // aligned relations plus matched higher-order relations
  double rarity = 0;   // sum over shared values of 1 / log2(1 + holders)
  int aligned = 0, identical = 0, higher = 0;
};

namespace detail {
inline std::map<std::string, std::vector<std::string>> by_rel(const TripleStore& st, const std::string& s) {
  std::map<std::string, std::vector<std::string>> m;
  for (const auto& f : st.facts_of(s)) m[f.r].push_back(f.o);
  for (auto& [r, v] : m) std::sort(v.begin(), v.end());
  return m;
}
inline bool share_relation(const TripleStore& st, const std::string& a, const std::string& b) {
  if (a == b) return false;
  std::set<std::string> ra;
  for (const auto& f : st.facts_of(a)) ra.insert(f.r);
  for (const auto& f : st.facts_of(b))
    if (ra.count(f.r)) return true;
  return false;
}
}  // namespace detail

/// The analogs of target t for relation r (subjects holding r that align with t on at least min_aligned relations
/// other than r, one of them on a shared value), best first.
inline std::vector<Analog> find_analogs(const TripleStore& st, const std::string& t, const std::string& r, const AnalogConfig& cfg = {}) {
  const auto T = detail::by_rel(st, t);
  std::map<std::string, double> rare;  // candidate analog -> rarity of the values it shares with t
  for (const auto& [r2, vs] : T) {
    if (r2 == r) continue;
    for (const auto& v : vs) {
      const auto& holders = st.subjects_with(r2, v);
      if (holders.size() > cfg.max_holders) continue;
      const double w = 1.0 / std::log2(1.0 + static_cast<double>(holders.size()));
      for (const auto& s : holders)
        if (s != t && st.has_relation(s, r)) rare[s] += w;
    }
  }
  std::vector<Analog> out;
  for (const auto& [s, rw] : rare) {
    const auto S = detail::by_rel(st, s);
    Analog a;
    a.s = s, a.rarity = rw;
    for (const auto& [r2, tv] : T) {
      if (r2 == r) continue;
      const auto it = S.find(r2);
      if (it == S.end()) continue;
      const auto& sv = it->second;
      bool same = false;
      for (const auto& v : tv) same = same || std::binary_search(sv.begin(), sv.end(), v);
      ++a.aligned;
      if (same) ++a.identical;
      else if (detail::share_relation(st, sv.front(), tv.front())) ++a.higher;
    }
    if (a.aligned < cfg.min_aligned || a.identical < cfg.min_identical) continue;
    a.sys = a.aligned + a.higher;
    out.push_back(a);
  }
  std::sort(out.begin(), out.end(), [](const Analog& a, const Analog& b) {
    if (a.sys != b.sys) return a.sys > b.sys;
    if (a.rarity != b.rarity) return a.rarity > b.rarity;
    return a.s < b.s;
  });
  if (out.size() > cfg.keep) out.resize(cfg.keep);
  return out;
}

/// Candidate values of (t, r) carried over from the analogs by the roles their values play. lambda weighs a value
/// carried as a constant; equality and path roles weigh 1. Values t already holds for r, and t itself, are left out.
inline std::vector<std::pair<std::string, double>> transfer(const TripleStore& st, const std::vector<Analog>& analogs, const std::string& t,
                                                            const std::string& r, double lambda) {
  std::map<std::string, double> sc;
  const auto T = detail::by_rel(st, t);
  for (const Analog& A : analogs) {
    const auto S = detail::by_rel(st, A.s);
    const auto it = S.find(r);
    if (it == S.end()) continue;
    for (const auto& a : it->second) {
      std::set<std::string> got_struct;
      // equality: a is also s's value of r''
      for (const auto& [r2, sv] : S) {
        if (r2 == r || !std::binary_search(sv.begin(), sv.end(), a)) continue;
        const auto tt = T.find(r2);
        if (tt == T.end()) continue;
        for (const auto& b : tt->second) got_struct.insert(b);
      }
      // path: s r1 z, z r2 a  =>  t r1 z', z' r2 b
      for (const auto& [r1, zs] : S) {
        if (r1 == r) continue;
        const auto tt = T.find(r1);
        if (tt == T.end()) continue;
        for (const auto& z : zs)
          for (const auto& g : st.facts_of(z)) {
            if (g.o != a) continue;
            for (const auto& z2 : tt->second)
              for (const auto& b : st.facts_of(z2))
                if (b.r == g.r) got_struct.insert(b.o);
          }
      }
      for (const auto& b : got_struct) sc[b] += A.sys;
      if (lambda > 0) sc[a] += A.sys * lambda;
    }
  }
  std::vector<std::pair<std::string, double>> out;
  for (const auto& [o, v] : sc) {
    if (o == t || st.has(t, r, o)) continue;
    out.push_back({o, v});
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
  return out;
}

// ---------------------------------------------------------------------------
// G5: vector analogy
// ---------------------------------------------------------------------------

/// Word vectors, unit rows. load_f32: ppmi_svd's `prefix.vectors.f32` (rows x d float32) with `prefix.vocab.tsv`;
/// load_projected: the "CYSIMV1" file (int8 rows with a scale each and an FNV-1a checksum).
class WordVectors {
 public:
  int dim() const { return d_; }
  std::size_t size() const { return id_.size(); }

  bool load_f32(const std::string& f32, const std::string& vocab, std::string& err) {
    std::ifstream vf(vocab);
    std::vector<std::string> words;
    for (std::string l; std::getline(vf, l);) words.push_back(l.substr(0, l.find('\t')));
    std::ifstream in(f32, std::ios::binary | std::ios::ate);
    if (words.empty() || !in) return err = "cannot read " + f32 + " / " + vocab, false;
    const std::size_t bytes = static_cast<std::size_t>(in.tellg());
    if (bytes % (4 * words.size())) return err = "size of " + f32 + " is not rows x d x 4", false;
    d_ = static_cast<int>(bytes / 4 / words.size());
    v_.assign(bytes / 4, 0.0f);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v_.data()), static_cast<std::streamsize>(bytes));
    for (std::size_t r = 0; r < words.size(); ++r) {
      normalise(&v_[r * static_cast<std::size_t>(d_)]);
      id_.emplace(words[r], static_cast<int>(r));
    }
    return true;
  }

  bool load_projected(const std::string& path, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return err = "cannot read " + path, false;
    std::uint64_t h = 0xcbf29ce484222325ull;
    auto get = [&](void* dst, std::size_t n) {
      in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
      const auto* b = static_cast<const unsigned char*>(dst);
      for (std::size_t i = 0; in && i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
      return static_cast<bool>(in);
    };
    auto u32 = [&](std::uint32_t& x) {
      unsigned char b[4];
      if (!get(b, 4)) return false;
      x = std::uint32_t{b[0]} | std::uint32_t{b[1]} << 8 | std::uint32_t{b[2]} << 16 | std::uint32_t{b[3]} << 24;
      return true;
    };
    char magic[8];
    std::uint32_t rows = 0, dim = 0;
    if (!get(magic, 8) || std::memcmp(magic, "CYSIMV1", 8) != 0 || !u32(rows) || !u32(dim) || dim == 0 || dim > 4096 || rows == 0)
      return err = "not a projected vector file", false;
    std::vector<std::string> words(rows);
    for (auto& w : words) {
      unsigned char len = 0;
      w.assign(255, '\0');
      if (!get(&len, 1) || (len && !get(&w[0], len))) return err = "cut short", false;
      w.resize(len);
    }
    std::vector<float> scale(rows);
    for (auto& f : scale) {
      std::uint32_t x = 0;
      if (!u32(x)) return err = "cut short", false;
      std::memcpy(&f, &x, 4);
    }
    std::vector<std::int8_t> q(std::size_t{rows} * dim);
    if (!get(q.data(), q.size())) return err = "cut short", false;
    const std::uint64_t want = h;
    unsigned char b[8];
    if (!in.read(reinterpret_cast<char*>(b), 8)) return err = "cut short", false;
    std::uint64_t got = 0;
    for (int i = 7; i >= 0; --i) got = got << 8 | b[i];
    if (got != want) return err = "checksum mismatch", false;
    d_ = static_cast<int>(dim);
    v_.assign(q.size(), 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t j = 0; j < dim; ++j) v_[r * dim + j] = scale[r] * q[r * dim + j];
      normalise(&v_[r * dim]);
      if (!words[r].empty()) id_.emplace(words[r], static_cast<int>(r));
    }
    return true;
  }

  /// Adds a word with its vector (tests; the vector is normalised).
  void add(const std::string& w, std::vector<float> x) {
    if (d_ == 0) d_ = static_cast<int>(x.size());
    if (static_cast<int>(x.size()) != d_ || id_.count(w)) return;
    normalise(x.data());
    id_.emplace(w, static_cast<int>(v_.size() / static_cast<std::size_t>(d_)));
    v_.insert(v_.end(), x.begin(), x.end());
  }

  /// The words of a name as ppmi_svd reads them: runs of letters (ASCII or two-byte UTF-8), lower-cased.
  static std::vector<std::string> words_of(const std::string& name) {
    std::vector<std::string> out;
    std::string w;
    for (std::size_t i = 0; i < name.size();) {
      const unsigned char c = static_cast<unsigned char>(name[i]);
      if (std::isalpha(c)) {
        w.push_back(static_cast<char>(std::tolower(c))), ++i;
      } else if (c >= 0xC2 && c <= 0xDF && i + 1 < name.size()) {
        unsigned char e = static_cast<unsigned char>(name[i + 1]);
        if (c == 0xC3 && e >= 0x80 && e <= 0x9E) e = static_cast<unsigned char>(e + 0x20);
        w.push_back(static_cast<char>(c)), w.push_back(static_cast<char>(e)), i += 2;
      } else {
        if (!w.empty()) out.push_back(w), w.clear();
        ++i;
      }
    }
    if (!w.empty()) out.push_back(w);
    return out;
  }

  /// The unit mean of a name's words' unit vectors; empty when no word is known.
  std::vector<float> entity(const std::string& name) const {
    std::vector<float> m;
    int n = 0;
    for (const auto& w : words_of(name)) {
      const auto it = id_.find(w);
      if (it == id_.end()) continue;
      if (m.empty()) m.assign(static_cast<std::size_t>(d_), 0.0f);
      const float* x = &v_[static_cast<std::size_t>(it->second) * static_cast<std::size_t>(d_)];
      for (int j = 0; j < d_; ++j) m[static_cast<std::size_t>(j)] += x[j];
      ++n;
    }
    if (n) normalise(m.data());
    return m;
  }

 private:
  void normalise(float* x) const {
    double s = 0;
    for (int j = 0; j < d_; ++j) s += double(x[j]) * x[j];
    if (s <= 0) return;
    const double k = 1.0 / std::sqrt(s);
    for (int j = 0; j < d_; ++j) x[j] = static_cast<float>(x[j] * k);
  }
  int d_ = 0;
  std::vector<float> v_;
  std::unordered_map<std::string, int> id_;
};

inline double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.empty() || a.size() != b.size()) return 0.0;
  double ab = 0, aa = 0, bb = 0;
  for (std::size_t j = 0; j < a.size(); ++j) ab += double(a[j]) * b[j], aa += double(a[j]) * a[j], bb += double(b[j]) * b[j];
  return aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : 0.0;
}

/// G5's relation offsets: v_r = mean of unit(b) - unit(a) over the store's facts (a, r, b) with both vectors.
class VectorAnalogy {
 public:
  VectorAnalogy(const TripleStore& st, const WordVectors& wv) : wv_(wv) {
    std::map<std::string, std::pair<std::vector<double>, double>> acc;
    std::vector<std::string> subj;
    for (const auto& [s, fs] : st.subjects()) subj.push_back(s);
    std::sort(subj.begin(), subj.end());
    for (const auto& s : subj) {
      const auto va = vec(s);
      if (va.empty()) continue;
      for (const auto& f : st.facts_of(s)) {
        const auto vb = vec(f.o);
        if (vb.empty()) continue;
        auto& [sum, n] = acc[f.r];
        if (sum.empty()) sum.assign(va.size(), 0.0);
        for (std::size_t j = 0; j < va.size(); ++j) sum[j] += double(vb[j]) - va[j];
        n += 1;
      }
    }
    for (auto& [r, p] : acc) {
      std::vector<float> v(p.first.size());
      for (std::size_t j = 0; j < v.size(); ++j) v[j] = static_cast<float>(p.first[j] / p.second);
      off_[r] = v, n_[r] = p.second;
    }
  }
  bool has(const std::string& r) const { return off_.count(r) > 0; }
  double facts_used(const std::string& r) const {
    const auto it = n_.find(r);
    return it == n_.end() ? 0.0 : it->second;
  }
  /// The query point unit(t) + v_r; empty when t has no vector or r no offset.
  std::vector<float> query(const std::string& t, const std::string& r) const {
    const auto it = off_.find(r);
    if (it == off_.end()) return {};
    auto q = vec(t);
    if (q.empty()) return {};
    for (std::size_t j = 0; j < q.size(); ++j) q[j] += it->second[j];
    return q;
  }
  /// cosine(query, unit(o)); nullopt-like: returns false when o has no vector.
  bool score(const std::vector<float>& q, const std::string& o, double& out) const {
    const auto vo = vec(o);
    if (q.empty() || vo.empty()) return false;
    out = cosine(q, vo);
    return true;
  }
  std::vector<float> vec(const std::string& name) const {
    const auto it = cache_.find(name);
    if (it != cache_.end()) return it->second;
    return cache_[name] = wv_.entity(name);
  }

 private:
  const WordVectors& wv_;
  std::map<std::string, std::vector<float>> off_;
  std::map<std::string, double> n_;
  mutable std::unordered_map<std::string, std::vector<float>> cache_;
};

}  // namespace cypha::eval::analogy
