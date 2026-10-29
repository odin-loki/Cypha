// Possible worlds, consequence tests and fragility for Cypha's discovery track (docs/DISCOVERY.md 4.4, 7.2 T2, 7.6, 12
// I9; docs/reports/S9_WORLDS_AND_FRAGILITY.md).
//
// Header-only. Reads a TripleStore (hypothesis.hpp) and G2's rules (gen_rules.hpp); standard library only.
//
//   World       a copy-on-write overlay of a store: the store (read only, by const reference) plus assumptions added or
//               retracted, and a layer of facts derived inside the world. A world is a value that is dropped after use; it
//               cannot change its store, and write_back always refuses (I9). export_items gives everything it assumed or
//               derived as tainted knowledge items (S5's taint), never answerable.
//   chain       forward chaining of the kept G2 rules inside a world, up to depth 3, from the assumptions (or from a given
//               frontier). Each derived fact keeps its rule, its premises, its depth and its confidence (the product of
//               the rules' PCA confidences along its chain). A consequence the world already holds is a confirmation. A
//               derived fact at or above the confidence floor that conflicts with the world's held facts (the caller's
//               checks), or two derived facts that cannot both hold, is a contradiction.
//   t2_world    T2 (DISCOVERY 7.1): negative on a contradiction, positive on a confirmation without one, else
//               inconclusive. An assumption's own conflict with the store is T1's and is reported apart.
//   vanished    consequences the retracted facts supported that the retracted world can no longer derive (depth-limited
//               backward search; path joins through facts the world holds) and does not hold.
//   minimal_conflicting_subsets   reductio: the smallest assumption subsets whose world reaches a contradiction.
//   FragilityModel   7.6: facts with their source items, depth-1 rule derivations, rivals and constraints; opinions and
//               states with sources or facts retracted; the top source, the weakest premise, fragility and drops.
//   format_world_card   the hypothetical card of a world (epistemic.hpp's label and closing line), as data.
//
// Nothing here answers a question. Worlds and cards are bookkeeping about Cypha's own store (PLAN rule 10).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cypha/eval/gen_rules.hpp"
#include "cypha/eval/hypothesis.hpp"

namespace cypha::eval {

struct Triple {
  std::string s, r, o;
  bool operator==(const Triple& x) const { return s == x.s && r == x.r && o == x.o; }
  bool operator<(const Triple& x) const { return s != x.s ? s < x.s : r != x.r ? r < x.r : o < x.o; }
  std::string key() const { return s + '\x1f' + r + '\x1f' + o; }
  std::string text() const { return s + " " + r + " " + o; }
};

/// FNV-1a 64 over a store's sorted triples: equal stores give equal fingerprints (I9's check).
inline std::uint64_t store_fingerprint(const TripleStore& st) {
  std::vector<std::string> keys;
  keys.reserve(st.size());
  for (const auto& [s, fs] : st.subjects())
    for (const auto& f : fs) keys.push_back(s + '\x1f' + f.r + '\x1f' + f.o);
  std::sort(keys.begin(), keys.end());
  std::uint64_t h = 1469598103934665603ULL;
  for (const auto& k : keys) {
    for (unsigned char c : k) h = (h ^ c) * 1099511628211ULL;
    h = (h ^ '\n') * 1099511628211ULL;
  }
  return h;
}

// ---------------------------------------------------------------------------
// The world
// ---------------------------------------------------------------------------

class World {
 public:
  explicit World(const TripleStore& base) : base_(&base) {}

  /// An assumption added. A fact the world already holds is not added again (it is still listed as assumed).
  void add(const std::string& s, const std::string& r, const std::string& o) {
    if (s.empty() || r.empty() || o.empty()) return;
    const Triple t{s, r, o};
    assumed_add_.push_back(t);
    retracted_.erase(t.key());
    if (!held(s, r, o)) added_.put(t);
  }
  /// An assumption retracted: a stored fact, an added one or a derived one stops holding in this world.
  void retract(const std::string& s, const std::string& r, const std::string& o) {
    const Triple t{s, r, o};
    assumed_retract_.push_back(t);
    retracted_.insert(t.key());
  }

  /// Held: stored or assumed, and not retracted. Derived facts are not held.
  bool held(const std::string& s, const std::string& r, const std::string& o) const {
    const Triple t{s, r, o};
    if (retracted_.count(t.key())) return false;
    return base_->has(s, r, o) || added_.has(t);
  }
  bool held(const Triple& t) const { return held(t.s, t.r, t.o); }
  bool derived(const Triple& t) const { return !retracted_.count(t.key()) && derived_.has(t); }
  /// Held or derived in this world.
  bool has(const Triple& t) const { return held(t) || derived(t); }
  bool in_store(const Triple& t) const { return !retracted_.count(t.key()) && base_->has(t.s, t.r, t.o); }

  /// Objects of (s, r) over every layer (stored, assumed, derived), retractions applied.
  std::vector<std::string> values(const std::string& s, const std::string& r) const {
    std::vector<std::string> out;
    for (const auto& f : base_->facts_of(s))
      if (f.r == r && !retracted_.count(Triple{s, r, f.o}.key())) out.push_back(f.o);
    added_.values(s, r, retracted_, out);
    derived_.values(s, r, retracted_, out);
    return out;
  }
  /// Subjects holding (r, o) over every layer, retractions applied.
  std::vector<std::string> subjects_with(const std::string& r, const std::string& o) const {
    std::vector<std::string> out;
    for (const auto& s : base_->subjects_with(r, o))
      if (!retracted_.count(Triple{s, r, o}.key())) out.push_back(s);
    added_.subjects(r, o, retracted_, out);
    derived_.subjects(r, o, retracted_, out);
    return out;
  }
  /// The held facts of s (stored and assumed), retractions applied.
  std::vector<TripleStore::Fact> held_facts_of(const std::string& s) const {
    std::vector<TripleStore::Fact> out;
    for (const auto& f : base_->facts_of(s))
      if (!retracted_.count(Triple{s, f.r, f.o}.key())) out.push_back(f);
    auto it = added_.by_s.find(s);
    if (it != added_.by_s.end())
      for (const auto& f : it->second)
        if (!retracted_.count(Triple{s, f.r, f.o}.key())) out.push_back(f);
    return out;
  }

  const std::vector<Triple>& assumed_added() const { return assumed_add_; }
  const std::vector<Triple>& assumed_retracted() const { return assumed_retract_; }
  bool assumed(const Triple& t) const { return std::find(assumed_add_.begin(), assumed_add_.end(), t) != assumed_add_.end(); }
  const TripleStore& store() const { return *base_; }

  /// I9: a world is never written back. Every attempt is refused and leaves the target as it was.
  bool write_back(TripleStore& /*target*/) const { return false; }

  /// Everything the world assumed or derived, as knowledge items carrying the taint flag (S5). A derived item names its
  /// premises: assumptions and other derived facts as world items, stored facts as "store:" ids (absent from any fresh
  /// knowledge store, so the taint check fails closed on them too).
  std::vector<KnowledgeItem> export_items() const {
    std::vector<KnowledgeItem> out;
    for (const Triple& t : assumed_add_) out.push_back(KnowledgeItem{"world:" + t.key(), Opinion{}, std::nullopt, true, {}});
    for (const auto& [k, prem] : derived_premises_) {
      KnowledgeItem it{"world:" + k, Opinion{}, std::nullopt, true, {}};
      for (const Triple& p : prem) it.premises.push_back((in_store(p) ? "store:" : "world:") + p.key());
      out.push_back(std::move(it));
    }
    return out;
  }

  // Used by chain: a fact derived inside the world, with its premises.
  void add_derived(const Triple& t, const std::vector<Triple>& premises) {
    derived_.put(t);
    derived_premises_[t.key()] = premises;
  }

 private:
  struct Layer {
    std::unordered_map<std::string, std::vector<TripleStore::Fact>> by_s;
    std::unordered_map<std::string, std::vector<std::string>> by_ro;
    std::unordered_set<std::string> keys;
    void put(const Triple& t) {
      if (!keys.insert(t.key()).second) return;
      by_s[t.s].push_back({t.r, t.o});
      by_ro[t.r + '\x1f' + t.o].push_back(t.s);
    }
    bool has(const Triple& t) const { return keys.count(t.key()) > 0; }
    void values(const std::string& s, const std::string& r, const std::unordered_set<std::string>& rx, std::vector<std::string>& out) const {
      auto it = by_s.find(s);
      if (it == by_s.end()) return;
      for (const auto& f : it->second)
        if (f.r == r && !rx.count(Triple{s, r, f.o}.key())) out.push_back(f.o);
    }
    void subjects(const std::string& r, const std::string& o, const std::unordered_set<std::string>& rx, std::vector<std::string>& out) const {
      auto it = by_ro.find(r + '\x1f' + o);
      if (it == by_ro.end()) return;
      for (const auto& s : it->second)
        if (!rx.count(Triple{s, r, o}.key())) out.push_back(s);
    }
  };

  const TripleStore* base_;
  Layer added_, derived_;
  std::unordered_set<std::string> retracted_;
  std::vector<Triple> assumed_add_, assumed_retract_;
  std::map<std::string, std::vector<Triple>> derived_premises_;
};

// ---------------------------------------------------------------------------
// Rules inside a world
// ---------------------------------------------------------------------------

/// The kept G2 rules, indexed for chaining in a world (RuleSet reads a fixed store; a world needs its overlay).
class RuleBook {
 public:
  explicit RuleBook(const std::vector<Rule>& rules) {
    for (const Rule& R : rules)
      if (R.kept) rules_.push_back(R);
    for (std::size_t i = 0; i < rules_.size(); ++i) {
      const Rule& R = rules_[i];
      by_head_[R.head].push_back(i);
      if (R.kind == RuleKind::Const) by_body_[R.r1 + '\x1f' + R.c1].push_back(i);
      else by_r1_[R.r1].push_back(i);
      if (R.kind == RuleKind::Path) by_r2_[R.r2].push_back(i);
    }
  }
  const std::vector<Rule>& rules() const { return rules_; }
  const std::vector<std::size_t>& with_head(const std::string& r) const { return find(by_head_, r); }
  const std::vector<std::size_t>& const_with_body(const std::string& r, const std::string& c) const { return find(by_body_, r + '\x1f' + c); }
  const std::vector<std::size_t>& with_r1(const std::string& r) const { return find(by_r1_, r); }
  const std::vector<std::size_t>& with_r2(const std::string& r) const { return find(by_r2_, r); }

 private:
  static const std::vector<std::size_t>& find(const std::unordered_map<std::string, std::vector<std::size_t>>& m, const std::string& k) {
    static const std::vector<std::size_t> none;
    auto it = m.find(k);
    return it == m.end() ? none : it->second;
  }
  std::vector<Rule> rules_;
  std::unordered_map<std::string, std::vector<std::size_t>> by_head_, by_body_, by_r1_, by_r2_;
};

/// One consequence of a fact f in a world: the derived triple, the rule, and the rule's other premise (path rules).
struct WorldConsequence {
  Triple t;
  std::size_t rule = 0;
  std::optional<Triple> other;
};

/// What the kept rules derive from f in world w, f filling one body atom and the other atom (path rules) read from the
/// world. As RuleSet::consequences, over the world's layers.
inline std::vector<WorldConsequence> world_consequences(const World& w, const RuleBook& book, const Triple& f) {
  std::vector<WorldConsequence> out;
  const auto& rs = book.rules();
  auto add = [&](const std::string& s, const std::string& r, const std::string& o, std::size_t i, std::optional<Triple> other) {
    if (s.empty() || o.empty() || s == o) return;
    Triple t{s, r, o};
    if (t == f) return;
    out.push_back({std::move(t), i, std::move(other)});
  };
  for (std::size_t i : book.const_with_body(f.r, f.o)) add(f.s, rs[i].head, rs[i].head_o, i, std::nullopt);
  for (std::size_t i : book.with_r1(f.r)) {
    const Rule& R = rs[i];
    if (R.kind == RuleKind::Same) add(f.s, R.head, f.o, i, std::nullopt);
    else if (R.kind == RuleKind::Inverse) add(f.o, R.head, f.s, i, std::nullopt);
    else if (R.kind == RuleKind::Path) {
      // f is atom 1: r1(X, Z) = f gives X = f.s, Z = f.o; r1(Z, X) = f gives Z = f.s, X = f.o.
      const std::string X = R.inv1 ? f.o : f.s, Z = R.inv1 ? f.s : f.o;
      if (!R.inv2)
        for (const auto& Y : w.values(Z, R.r2)) {
          if (Y != X) add(X, R.head, Y, i, Triple{Z, R.r2, Y});
        }
      else
        for (const auto& Y : w.subjects_with(R.r2, Z))
          if (Y != X) add(X, R.head, Y, i, Triple{Y, R.r2, Z});
    }
  }
  for (std::size_t i : book.with_r2(f.r)) {
    const Rule& R = rs[i];
    // f is atom 2: r2(Z, Y) = f gives Z = f.s, Y = f.o; r2(Y, Z) = f gives Y = f.s, Z = f.o.
    const std::string Z = R.inv2 ? f.o : f.s, Y = R.inv2 ? f.s : f.o;
    if (!R.inv1)
      for (const auto& X : w.subjects_with(R.r1, Z)) {
        if (X != Y) add(X, R.head, Y, i, Triple{X, R.r1, Z});
      }
    else
      for (const auto& X : w.values(Z, R.r1))
        if (X != Y) add(X, R.head, Y, i, Triple{Z, R.r1, X});
  }
  return out;
}

// ---------------------------------------------------------------------------
// Chaining, contradictions, T2
// ---------------------------------------------------------------------------

struct ChainConfig {
  int depth = 3;
  double c_min = 0.8;  // a derived fact below this chain confidence is never counted as a contradiction
};

struct Derived {
  Triple t;
  std::size_t rule = 0;
  std::vector<Triple> premises;
  int depth = 0;
  double conf = 1.0;
  bool by_store = false;  // a confirmation the store itself holds (not only an assumption of the world)
};

struct Clash {
  Triple fact;     // the derived (or assumed) fact
  Triple culprit;  // the held fact (or the other derived fact) it collides with
  std::string kind;
  double conf = 1.0;
  bool among_derived = false;
};

struct ChainResult {
  std::vector<Derived> derived;      // new consequences, in derivation order
  std::vector<Derived> confirmed;    // consequences the world already holds (each once)
  std::vector<Clash> contradictions; // derived facts against held facts, or against each other
  std::vector<Clash> direct;         // an assumption against the other held facts (T1's business, reported apart)
};

/// The caller's domain checks. `held` tests a fact against the world's held facts (stored or assumed, never derived):
/// true and the culprit when they cannot both hold. `pair` tests two derived facts of one slot. `confirmed` says whether
/// the store (w.store(), retractions aside) already holds a consequence under the caller's value equality (default: the
/// exact triple).
struct ConflictChecks {
  std::function<bool(const World&, const Triple&, Triple&, std::string&)> held;
  std::function<bool(const Triple&, const Triple&)> pair;
  std::function<bool(const World&, const Triple&)> confirmed;
};

/// Forward chaining inside w from `frontier` (empty: the world's added assumptions), up to cfg.depth. Derived facts
/// join the world's derived layer, so later steps read them.
inline ChainResult chain(World& w, const RuleBook& book, const ChainConfig& cfg, const ConflictChecks& cc,
                         std::vector<Triple> frontier = {}) {
  ChainResult res;
  if (frontier.empty()) frontier = w.assumed_added();
  for (const Triple& a : w.assumed_added()) {  // direct conflicts of the assumptions
    Triple culprit;
    std::string kind;
    if (cc.held && cc.held(w, a, culprit, kind)) res.direct.push_back({a, culprit, kind, 1.0, false});
  }
  std::unordered_map<std::string, double> conf;  // derived fact -> its chain confidence
  std::unordered_set<std::string> seen_confirmed;
  auto conf_of = [&](const Triple& t) {
    auto it = conf.find(t.key());
    return it == conf.end() ? 1.0 : it->second;
  };
  const auto& rs = book.rules();
  for (int d = 1; d <= cfg.depth && !frontier.empty(); ++d) {
    std::vector<Triple> next;
    for (const Triple& f : frontier) {
      const double cf = conf_of(f);
      for (const WorldConsequence& c : world_consequences(w, book, f)) {
        const double cc_ = rs[c.rule].pca_conf() * cf * (c.other ? conf_of(*c.other) : 1.0);
        std::vector<Triple> prem{f};
        if (c.other) prem.push_back(*c.other);
        const bool by_store = w.in_store(c.t) || (cc.confirmed && cc.confirmed(w, c.t));
        if (by_store || w.held(c.t)) {
          if (seen_confirmed.insert(c.t.key()).second) res.confirmed.push_back({c.t, c.rule, prem, d, cc_, by_store});
          continue;
        }
        if (w.derived(c.t)) {
          auto it = conf.find(c.t.key());
          if (it != conf.end() && cc_ > it->second) it->second = cc_;  // keep the best chain's confidence
          continue;
        }
        w.add_derived(c.t, prem);
        conf[c.t.key()] = cc_;
        res.derived.push_back({c.t, c.rule, prem, d, cc_});
        next.push_back(c.t);
      }
    }
    frontier = std::move(next);
  }
  for (Derived& x : res.derived) x.conf = conf_of(x.t);
  for (const Derived& x : res.derived) {
    if (x.conf < cfg.c_min || !cc.held) continue;
    Triple culprit;
    std::string kind;
    if (cc.held(w, x.t, culprit, kind)) res.contradictions.push_back({x.t, culprit, kind, x.conf, false});
  }
  if (cc.pair) {
    std::map<std::string, std::vector<const Derived*>> slot;
    for (const Derived& x : res.derived)
      if (x.conf >= cfg.c_min) slot[x.t.s + '\x1f' + x.t.r].push_back(&x);
    for (const auto& [k, v] : slot)
      for (std::size_t i = 0; i < v.size(); ++i)
        for (std::size_t j = i + 1; j < v.size(); ++j)
          if (cc.pair(v[i]->t, v[j]->t))
            res.contradictions.push_back({v[i]->t, v[j]->t, "derived pair", std::min(v[i]->conf, v[j]->conf), true});
  }
  return res;
}

/// T2 (DISCOVERY 7.1) through a world: the store plus h, chained. Negative on a contradiction, positive when the store
/// already holds a consequence (and nothing contradicts), inconclusive otherwise. A consequence held only because h is
/// assumed confirms nothing. h's own conflicts (res.direct) do not count: they are T1's.
inline TestOutcome t2_world(const TripleStore& st, const RuleBook& book, const Triple& h, const ChainConfig& cfg,
                            const ConflictChecks& cc, ChainResult* out = nullptr) {
  World w(st);
  w.add(h.s, h.r, h.o);
  ChainResult res = chain(w, book, cfg, cc);
  bool by_store = false;
  for (const Derived& c : res.confirmed) by_store = by_store || c.by_store;
  const TestOutcome o = !res.contradictions.empty() ? TestOutcome::Negative
                        : by_store                   ? TestOutcome::Positive
                                                     : TestOutcome::Inconclusive;
  if (out) *out = std::move(res);
  return o;
}

/// Whether w can derive t within `depth` steps (backward search over the kept rules; a path rule's join node comes from
/// the facts the world holds or has derived). A held fact is derivable at any depth.
inline bool derivable(const World& w, const RuleBook& book, const Triple& t, int depth, std::unordered_set<std::string>* stack = nullptr) {
  if (w.has(t)) return true;
  if (depth <= 0) return false;
  std::unordered_set<std::string> local;
  if (!stack) stack = &local;
  if (!stack->insert(t.key()).second) return false;
  bool ok = false;
  const auto& rs = book.rules();
  for (std::size_t i : book.with_head(t.r)) {
    const Rule& R = rs[i];
    if (R.kind == RuleKind::Const) ok = t.o == R.head_o && derivable(w, book, {t.s, R.r1, R.c1}, depth - 1, stack);
    else if (R.kind == RuleKind::Same) ok = derivable(w, book, {t.s, R.r1, t.o}, depth - 1, stack);
    else if (R.kind == RuleKind::Inverse) ok = derivable(w, book, {t.o, R.r1, t.s}, depth - 1, stack);
    else {
      const std::vector<std::string> zs = R.inv1 ? w.subjects_with(R.r1, t.s) : w.values(t.s, R.r1);
      for (const auto& Z : zs) {
        const Triple a2 = R.inv2 ? Triple{t.o, R.r2, Z} : Triple{Z, R.r2, t.o};
        if (Z != t.o && derivable(w, book, a2, depth - 1, stack)) {
          ok = true;
          break;
        }
      }
    }
    if (ok) break;
  }
  stack->erase(t.key());
  return ok;
}

/// Consequences that disappeared: what the retracted facts derived in the store (chained from them, depth cfg.depth)
/// that the retracted world neither holds nor can derive within cfg.depth.
inline std::vector<Triple> vanished(const World& retracted_world, const RuleBook& book, const ChainConfig& cfg) {
  World before(retracted_world.store());
  const ChainResult r = chain(before, book, cfg, ConflictChecks{}, retracted_world.assumed_retracted());
  std::vector<Triple> out;
  for (const Derived& d : r.derived)
    if (!retracted_world.held(d.t) && !derivable(retracted_world, book, d.t, cfg.depth)) out.push_back(d.t);
  return out;
}

/// Reductio: the minimal subsets of `assumptions` (by index, at most 16) whose world, chained, reaches a contradiction or
/// a direct conflict. Subsets are tried in order of size; a superset of a subset already found is skipped.
inline std::vector<std::vector<std::size_t>> minimal_conflicting_subsets(const TripleStore& st, const RuleBook& book,
                                                                        const std::vector<Triple>& assumptions,
                                                                        const ChainConfig& cfg, const ConflictChecks& cc) {
  std::vector<std::vector<std::size_t>> out;
  const std::size_t n = std::min<std::size_t>(assumptions.size(), 16);
  std::vector<std::uint32_t> found;
  for (std::size_t k = 1; k <= n; ++k)
    for (std::uint32_t m = 1; m < (1u << n); ++m) {
      std::size_t bits = 0;
      for (std::uint32_t x = m; x; x &= x - 1) ++bits;
      if (bits != k) continue;
      bool super = false;
      for (std::uint32_t f : found)
        if ((m & f) == f) super = true;
      if (super) continue;
      World w(st);
      for (std::size_t i = 0; i < n; ++i)
        if (m & (1u << i)) w.add(assumptions[i].s, assumptions[i].r, assumptions[i].o);
      const ChainResult r = chain(w, book, cfg, cc);
      if (r.contradictions.empty() && r.direct.empty()) continue;
      found.push_back(m);
      std::vector<std::size_t> v;
      for (std::size_t i = 0; i < n; ++i)
        if (m & (1u << i)) v.push_back(i);
      out.push_back(std::move(v));
    }
  return out;
}

// ---------------------------------------------------------------------------
// Fragility (7.6)
// ---------------------------------------------------------------------------

struct SourceItem {
  std::size_t source = 0;  // a source instance (the caller's table)
  EvidenceItem item;
};

struct FactDerivation {
  std::size_t rule = 0;
  std::vector<std::size_t> premises;  // other facts of the model
  double pca = 0.0;
};

struct FragFact {
  std::string s, r, o;
  std::vector<SourceItem> sources;
  std::vector<FactDerivation> derivations;
  std::vector<std::size_t> rivals;  // facts competing for the same functional slot
  double phi = 0.0;                 // the slot's functionality (weights the rivals' mass)
  int hard = 0;                     // hard constraint violations
};

/// What a world retracts: source instances and whole facts.
struct Retraction {
  std::unordered_set<std::size_t> sources, facts;
};

/// Held: Known or Believed (the states an answer may use, 2.1).
inline bool held_state(EpistemicState s) { return s == EpistemicState::Known || s == EpistemicState::Believed; }

/// A fact drops when its state class falls: Known to anything else, Believed to anything not held.
inline bool drops(EpistemicState before, EpistemicState after) {
  if (before == EpistemicState::Known) return after != EpistemicState::Known;
  if (before == EpistemicState::Believed) return !held_state(after);
  return false;
}

class FragilityModel {
 public:
  explicit FragilityModel(std::vector<FragFact> facts, FusionConfig cfg = {}) : facts_(std::move(facts)), cfg_(cfg) {
    for (std::size_t i = 0; i < facts_.size(); ++i) {
      for (const SourceItem& si : facts_[i].sources) by_source_[si.source].push_back(i);
      for (const FactDerivation& d : facts_[i].derivations)
        for (std::size_t p : d.premises) dependents_[p].push_back(i);
      for (std::size_t r : facts_[i].rivals) dependents_[r].push_back(i);
    }
    for (auto& [k, v] : by_source_) dedupe(v);
    for (auto& [k, v] : dependents_) dedupe(v);
  }

  const std::vector<FragFact>& facts() const { return facts_; }
  std::size_t size() const { return facts_.size(); }
  const std::vector<std::size_t>& facts_of_source(std::size_t s) const { return find(by_source_, s); }
  const std::vector<std::size_t>& dependents(std::size_t i) const { return find(dependents_, i); }

  /// A fact is alive when it is not retracted and keeps at least one source (a premise or a rival needs this to count).
  bool alive(std::size_t i, const Retraction& R) const {
    if (R.facts.count(i)) return false;
    for (const SourceItem& si : facts_[i].sources)
      if (!R.sources.count(si.source)) return true;
    return false;
  }

  /// The opinion from the fact's own sources, its rivals (as competing mass, weighted by phi) and its constraints.
  Opinion direct(std::size_t i, const Retraction& R) const { return fuse(items(i, R, false), config(i)); }
  /// direct plus the derivation items (mass PCA x the premises' direct P; a derivation with a dead premise is gone).
  Opinion full(std::size_t i, const Retraction& R) const { return fuse(items(i, R, true), config(i)); }

  /// The state from the full opinion and the strongest live rival's direct opinion. A fact whose sources are all
  /// retracted keeps its derivations; with none it is Vacuous (u = 1).
  EpistemicState state(std::size_t i, const Retraction& R, const ClassifyConfig& cc = {}) const {
    if (R.facts.count(i)) return EpistemicState::Vacuous;
    std::optional<Opinion> rival;
    for (std::size_t r : facts_[i].rivals) {
      if (!alive(r, R)) continue;
      const Opinion o = direct(r, R);
      if (!rival || o.P > rival->P) rival = o;
    }
    return classify(full(i, R), rival, cc);
  }

  /// The source instance whose items give the fact the most mass; nullopt when it has none.
  std::optional<std::size_t> top_source(std::size_t i) const {
    std::map<std::size_t, double> m;
    for (const SourceItem& si : facts_[i].sources) m[si.source] += si.item.mass();
    std::optional<std::size_t> best;
    double bm = -1.0;
    for (const auto& [s, v] : m)
      if (v > bm) bm = v, best = s;
    return best;
  }

  /// The premise with the lowest direct P across the fact's derivations; nullopt when it has none.
  std::optional<std::size_t> weakest_premise(std::size_t i) const {
    std::optional<std::size_t> best;
    double bp = 2.0;
    const Retraction none;
    for (const FactDerivation& d : facts_[i].derivations)
      for (std::size_t p : d.premises) {
        const double P = direct(p, none).P;
        if (P < bp || (P == bp && best && p < *best)) bp = P, best = p;
      }
    return best;
  }

  struct Fragility {
    int worlds = 0, drops = 0;
    bool top_dropped = false, premise_dropped = false, premise_world = false;
    double value() const { return worlds ? static_cast<double>(drops) / worlds : 0.0; }
  };

  /// 7.6 for a held fact: its top source retracted, and its weakest premise retracted (when it has one).
  Fragility fragility(std::size_t i, const ClassifyConfig& cc = {}) const {
    Fragility f;
    const Retraction none;
    const EpistemicState s0 = state(i, none, cc);
    if (!held_state(s0)) return f;
    if (const auto top = top_source(i)) {
      Retraction R;
      R.sources.insert(*top);
      ++f.worlds;
      f.top_dropped = drops(s0, state(i, R, cc));
      f.drops += f.top_dropped;
    }
    if (const auto p = weakest_premise(i)) {
      Retraction R;
      R.facts.insert(*p);
      ++f.worlds;
      f.premise_world = true;
      f.premise_dropped = drops(s0, state(i, R, cc));
      f.drops += f.premise_dropped;
    }
    return f;
  }

  /// Held facts that drop when R is applied, checked among the facts R touches, their rivals' holders and one level of
  /// dependents (derivations and rivalry).
  std::vector<std::size_t> dropped_by(const Retraction& R, const ClassifyConfig& cc = {}) const {
    std::set<std::size_t> touched;
    for (std::size_t s : R.sources)
      for (std::size_t i : facts_of_source(s)) touched.insert(i);
    for (std::size_t i : R.facts) touched.insert(i);
    std::set<std::size_t> check(touched.begin(), touched.end());
    for (std::size_t i : touched)
      for (std::size_t d : dependents(i)) check.insert(d);
    const Retraction none;
    std::vector<std::size_t> out;
    for (std::size_t i : check) {
      const EpistemicState s0 = state(i, none, cc);
      if (held_state(s0) && drops(s0, state(i, R, cc))) out.push_back(i);
    }
    return out;
  }

 private:
  static void dedupe(std::vector<std::size_t>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }
  static const std::vector<std::size_t>& find(const std::unordered_map<std::size_t, std::vector<std::size_t>>& m, std::size_t k) {
    static const std::vector<std::size_t> none;
    auto it = m.find(k);
    return it == m.end() ? none : it->second;
  }
  FusionConfig config(std::size_t i) const {
    FusionConfig c = cfg_;
    c.phi = facts_[i].phi;
    c.hard_violations = facts_[i].hard;
    return c;
  }
  std::vector<EvidenceItem> items(std::size_t i, const Retraction& R, bool with_derivations) const {
    std::vector<EvidenceItem> out;
    if (R.facts.count(i)) return out;
    for (const SourceItem& si : facts_[i].sources)
      if (!R.sources.count(si.source)) out.push_back(si.item);
    for (std::size_t r : facts_[i].rivals) {
      if (R.facts.count(r)) continue;
      for (const SourceItem& si : facts_[r].sources) {
        if (R.sources.count(si.source)) continue;
        EvidenceItem e = si.item;
        e.stance = Stance::CompetingValue;
        out.push_back(e);
      }
    }
    if (!with_derivations) return out;
    for (const FactDerivation& d : facts_[i].derivations) {
      double m = d.pca;
      for (std::size_t p : d.premises) m *= alive(p, R) ? direct(p, R).P : 0.0;
      if (m <= 0.0) continue;
      EvidenceItem e;
      e.stance = Stance::Support;
      e.delta = m;
      out.push_back(e);
    }
    return out;
  }

  std::vector<FragFact> facts_;
  FusionConfig cfg_;
  std::unordered_map<std::size_t, std::vector<std::size_t>> by_source_, dependents_;
};

// ---------------------------------------------------------------------------
// The hypothetical card of a world (data, never an answer)
// ---------------------------------------------------------------------------

struct WorldCard {
  std::vector<Triple> added, retracted;
  std::vector<Derived> consequences;
  std::vector<Triple> disappeared;
  std::vector<Clash> contradictions;
  std::vector<std::vector<Triple>> minimal_conflicts;
  std::vector<Triple> held_drops;  // held facts whose state falls in the world (fragility)
  int depth = 3;
  std::size_t examples = 5;
};

/// The card as text: epistemic.hpp's HYPOTHETICAL label and its closing line ("Nothing here is stored."), with the
/// world's consequences and minimal conflicting subsets inserted before the closing line. Always labelled (I8).
inline std::string format_world_card(const WorldCard& c) {
  std::string assumption;
  for (std::size_t i = 0; i < c.added.size() && i < c.examples; ++i) assumption += (assumption.empty() ? "" : "; ") + c.added[i].text();
  if (c.added.size() > c.examples) assumption += "; and " + std::to_string(c.added.size() - c.examples) + " more added";
  for (std::size_t i = 0; i < c.retracted.size() && i < c.examples; ++i)
    assumption += (assumption.empty() ? "not " : "; not ") + c.retracted[i].text();
  if (c.retracted.size() > c.examples) assumption += "; and " + std::to_string(c.retracted.size() - c.examples) + " more retracted";
  HypotheticalCard h;
  h.assumption = assumption.empty() ? "nothing" : assumption;
  h.depth = c.depth;
  h.dropped_count = static_cast<int>(c.disappeared.size());
  for (std::size_t i = 0; i < c.disappeared.size() && i < c.examples; ++i) h.dropped_examples.push_back(c.disappeared[i].text());
  for (const Clash& x : c.contradictions)
    h.contradictions_with_known.push_back(x.fact.text() + " vs " + x.culprit.text() + " (" + x.kind + ")");
  std::string s = format_hypothetical_card(h);
  const std::string close = "Nothing here is stored.\n";
  if (s.size() >= close.size() && s.compare(s.size() - close.size(), close.size(), close) == 0) s.resize(s.size() - close.size());
  if (!c.held_drops.empty()) {
    s += "Held facts that drop: " + std::to_string(c.held_drops.size());
    for (std::size_t i = 0; i < c.held_drops.size() && i < c.examples; ++i) s += (i ? "; " : " (") + c.held_drops[i].text();
    s += c.held_drops.size() > c.examples ? " ...)\n" : ")\n";
  }
  s += "New consequences: " + std::to_string(c.consequences.size());
  for (std::size_t i = 0; i < c.consequences.size() && i < c.examples; ++i)
    s += (i ? "; " : " (") + c.consequences[i].t.text() + " [depth " + std::to_string(c.consequences[i].depth) + "]";
  s += c.consequences.empty() ? "\n" : (c.consequences.size() > c.examples ? " ...)\n" : ")\n");
  if (!c.minimal_conflicts.empty()) {
    s += "Minimal conflicting subsets:";
    for (const auto& v : c.minimal_conflicts) {
      s += " {";
      for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "; " : "") + v[i].text();
      s += "}";
    }
    s += "\n";
  }
  s += "Tainted: every assumption and consequence above is hypothetical.\n";
  s += close;
  return s;
}

}  // namespace cypha::eval
