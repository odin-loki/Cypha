// Hypothesis store and firewall for Cypha's discovery track (docs/DISCOVERY.md sections 1, 4, 6.1, 6.3, 7.2, 7.4, 7.5, 11).
//
// Header-only. Includes epistemic.hpp (Opinion, fuse, classify, HypothesisCard and its label); standard library only.
//
//   1     the firewall: what each mode may read and write; taint carried through derivation; the answer-mode view
//   4.1   the hypothesis record; 4.2 lifecycle states and the legal transitions; 4.3 tiers
//   6.1   P0 = generator reliability x support x constraint; 6.3 EIG and priority (also used by the gap map)
//   6.1, 6.2 (S6) scoring a triple against a store: base rate, the link predictor (pairwise rules), Adamic-Adar, graph
//         distance (bridging), novelty, compression gain, impact, z-scored interestingness
//   7.2   Bayes-factor updating and the decision rule; 7.4 promotion through ECS on independent evidence only;
//   7.5   refutation: kept, counted against the generator and the rule, blocks regeneration while cooling
//   11    invariants I1 (no hypothetical item in answers), I4 (independent evidence), I7 (refuted kept, blocks), I8 (labels)
//
// Nothing here answers a question. The stores are bookkeeping about what Cypha knows and does not (PLAN rules 9 and 10).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cypha/eval/epistemic.hpp"

namespace cypha::eval {

// ---------------------------------------------------------------------------
// 6.3 Value of information
// ---------------------------------------------------------------------------

/// H(p) = -p log2 p - (1-p) log2 (1-p); 0 at p = 0 or 1.
inline double binary_entropy_bits(double p) {
  p = detail::ecs_clamp01(p);
  if (p <= 0.0 || p >= 1.0) return 0.0;
  return -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
}

struct InfoGain {
  double p_pos = 0.0;    // P(pos) = p s + (1 - p)(1 - c)
  double p_plus = 0.0;   // posterior after a positive result
  double p_minus = 0.0;  // posterior after a negative result
  double eig = 0.0;      // H(p) - [P(pos) H(p+) + (1 - P(pos)) H(p-)], bits
};

/// Expected information gain of a test with sensitivity s and specificity c on a binary hypothesis with probability p.
inline InfoGain expected_information_gain(double p, double s, double c) {
  InfoGain g;
  p = detail::ecs_clamp01(p);
  s = detail::ecs_clamp01(s);
  c = detail::ecs_clamp01(c);
  g.p_pos = p * s + (1.0 - p) * (1.0 - c);
  g.p_plus = g.p_pos > 0.0 ? p * s / g.p_pos : p;
  g.p_minus = g.p_pos < 1.0 ? p * (1.0 - s) / (1.0 - g.p_pos) : p;
  g.eig = binary_entropy_bits(p) - (g.p_pos * binary_entropy_bits(g.p_plus) + (1.0 - g.p_pos) * binary_entropy_bits(g.p_minus));
  if (g.eig < 0.0) g.eig = 0.0;  // rounding
  return g;
}

/// Priority = EIG (1 + Impact)(1 + max(0, I)) / Cost. Cost <= 0 is treated as 1.
inline double voi_priority(double eig, double impact, double interestingness = 0.0, double cost = 1.0) {
  return eig * (1.0 + std::max(0.0, impact)) * (1.0 + std::max(0.0, interestingness)) / (cost > 0.0 ? cost : 1.0);
}

/// 6.1: P0 = generator reliability x support term x constraint term (0 hard violation, 0.5 soft, 1 none), each clamped.
inline double plausibility_prior(double generator_reliability, double support, double constraint = 1.0) {
  return detail::ecs_clamp01(generator_reliability) * detail::ecs_clamp01(support) * detail::ecs_clamp01(constraint);
}

// ---------------------------------------------------------------------------
// 6.1, 6.2 Scoring a (subject, relation, object) hypothesis against a store of triples (S6)
// ---------------------------------------------------------------------------

/// Novelty (6.2): surprisal under the store, -log2 P_store(h). P is floored at 1e-12.
inline double novelty_bits(double p_store) { return -std::log2(std::max(1e-12, std::min(1.0, p_store))); }

/// Impact (6.2): gaps the hypothesis would fill, contradictions it would resolve, and the demand on those gaps.
inline double hypothesis_impact(double gaps_filled, double contradictions_resolved, double demand = 0.0,
                                double demand_weight = 1.0) {
  return gaps_filled + contradictions_resolved + demand_weight * demand;
}

/// Mean and standard deviation of a pool, for the z-scores of interestingness.
struct PoolStats {
  double mean = 0.0, sd = 0.0;
};
inline PoolStats pool_stats(const std::vector<double>& x) {
  PoolStats p;
  if (x.empty()) return p;
  for (double v : x) p.mean += v;
  p.mean /= static_cast<double>(x.size());
  for (double v : x) p.sd += (v - p.mean) * (v - p.mean);
  p.sd = std::sqrt(p.sd / static_cast<double>(x.size()));
  return p;
}
inline double zscore(double x, const PoolStats& p) { return p.sd > 0.0 ? (x - p.mean) / p.sd : 0.0; }

/// Interestingness (6.2): I = z(novelty) + z(impact) + z(dL) + z(bridging), each against the current pool.
inline double interestingness(double z_novelty, double z_impact, double z_compression, double z_bridging) {
  return z_novelty + z_impact + z_compression + z_bridging;
}

/// The store the scores read: distinct (s, r, o) triples. A node of the entity graph is any subject or object string;
/// the graph is undirected, one edge per related pair.
class TripleStore {
 public:
  struct Fact {
    std::string r, o;
  };

  /// Adds a triple; a duplicate or an empty field is ignored. Returns whether it was new.
  bool add(const std::string& s, const std::string& r, const std::string& o) {
    if (s.empty() || r.empty() || o.empty() || !triples_.insert(key3(s, r, o)).second) return false;
    facts_[s].push_back({r, o});
    if (ro_[key2(r, o)]++ == 0) ++values_[r];
    ++n_r_[r];
    subjects_[key2(r, o)].push_back(s);
    has_rel_.insert(key2(s, r));
    if (s != o) adj_[s].insert(o), adj_[o].insert(s);
    return true;
  }

  bool has(const std::string& s, const std::string& r, const std::string& o) const { return triples_.count(key3(s, r, o)) > 0; }
  bool has_relation(const std::string& s, const std::string& r) const { return has_rel_.count(key2(s, r)) > 0; }
  std::size_t count(const std::string& r, const std::string& o) const { return lookup(ro_, key2(r, o)); }
  std::size_t count(const std::string& r) const { return lookup(n_r_, r); }
  std::size_t values(const std::string& r) const { return lookup(values_, r); }
  std::size_t size() const { return triples_.size(); }

  const std::vector<Fact>& facts_of(const std::string& s) const {
    static const std::vector<Fact> none;
    auto it = facts_.find(s);
    return it == facts_.end() ? none : it->second;
  }
  const std::vector<std::string>& subjects_with(const std::string& r, const std::string& o) const {
    static const std::vector<std::string> none;
    auto it = subjects_.find(key2(r, o));
    return it == subjects_.end() ? none : it->second;
  }
  const std::unordered_set<std::string>& neighbours(const std::string& node) const {
    static const std::unordered_set<std::string> none;
    auto it = adj_.find(node);
    return it == adj_.end() ? none : it->second;
  }
  const std::unordered_map<std::string, std::vector<Fact>>& subjects() const { return facts_; }

 private:
  static std::string key2(const std::string& a, const std::string& b) { return a + '\x1f' + b; }
  static std::string key3(const std::string& a, const std::string& b, const std::string& c) { return a + '\x1f' + b + '\x1f' + c; }
  static std::size_t lookup(const std::unordered_map<std::string, std::size_t>& m, const std::string& k) {
    auto it = m.find(k);
    return it == m.end() ? 0 : it->second;
  }

  std::unordered_set<std::string> triples_, has_rel_;
  std::unordered_map<std::string, std::vector<Fact>> facts_;
  std::unordered_map<std::string, std::size_t> ro_, n_r_, values_;
  std::unordered_map<std::string, std::vector<std::string>> subjects_;
  std::unordered_map<std::string, std::unordered_set<std::string>> adj_;
};

/// P(o | r) over the relation's values, add-alpha with one unseen value. With `self` set, (s, r, o) is itself in the
/// store and is left out of the counts.
inline double base_rate(const TripleStore& st, const std::string& r, const std::string& o, double alpha = 1.0, bool self = false) {
  const double c = static_cast<double>(st.count(r, o)) - (self ? 1.0 : 0.0);
  const double n = static_cast<double>(st.count(r)) - (self ? 1.0 : 0.0);
  return (std::max(0.0, c) + alpha) / (std::max(0.0, n) + alpha * (static_cast<double>(st.values(r)) + 1.0));
}

/// Adamic-Adar of two nodes (6.1's support, S6b's path term): the sum over common neighbours z of 1 / ln(degree z).
inline double adamic_adar(const TripleStore& st, const std::string& a, const std::string& b) {
  const auto& na = st.neighbours(a);
  const auto& nb = st.neighbours(b);
  const auto& small = na.size() <= nb.size() ? na : nb;
  const auto& large = na.size() <= nb.size() ? nb : na;
  double s = 0.0;
  for (const std::string& z : small)
    if (large.count(z)) s += 1.0 / std::log(static_cast<double>(std::max<std::size_t>(2, st.neighbours(z).size())));
  return s;
}

/// Shortest-path length between two nodes (breadth first), or cap + 1 when there is none within cap steps.
inline int graph_distance(const TripleStore& st, const std::string& a, const std::string& b, int cap = 6) {
  if (a == b) return 0;
  std::unordered_map<std::string, int> seen{{a, 0}};
  std::deque<std::string> q{a};
  while (!q.empty()) {
    const std::string u = q.front();
    q.pop_front();
    const int d = seen[u];
    if (d >= cap) continue;
    for (const std::string& v : st.neighbours(u)) {
      if (v == b) return d + 1;
      if (seen.emplace(v, d + 1).second) q.push_back(v);
    }
  }
  return cap + 1;
}

/// The link predictor (6.1's support term, 6.2's P_store): P(s r o | the other facts of s), from pairwise rules over the
/// store (DISCOVERY G2). For a body fact (r2, o2) of s the rule's confidence is c = (n_both + m b) / (n_body + m), where
/// n_body counts the other subjects with (r2, o2) and some value of r, n_both those with (r, o) too, and b is the base
/// rate. A body fact with the same object adds a copy rule: how often a subject's r and r2 share a value. Rules combine
/// as naive Bayes in log-odds from b. m = 1 was chosen on S6's planted dev set (docs/reports/S6_HYPOTHESIS_SCORING.md).
class LinkPredictor {
 public:
  struct Config {
    double m = 1.0;          // shrinkage toward the base rate, in pseudo-subjects
    bool copy_rules = true;  // rules "r and r2 share a value" (off only for S6's ablation)
  };

  LinkPredictor(const TripleStore& st, Config cfg) : st_(st), cfg_(cfg) {}

  /// `skip`: a fact of s left out of the body (it is the one being coded); `extra`: a fact added to the body.
  double prob(const std::string& s, const std::string& r, const std::string& o, const TripleStore::Fact* skip = nullptr,
              const TripleStore::Fact* extra = nullptr) const {
    const bool self = st_.has(s, r, o);
    const double b = std::min(1.0 - 1e-9, base_rate(st_, r, o, 1.0, self));
    double lo = logit(b);
    auto use = [&](const TripleStore::Fact& f) {
      if (f.r == r && f.o == o) return;
      for (const double c : rule_confidences(s, r, o, f, b)) lo += logit(c) - logit(b);
    };
    for (const auto& f : st_.facts_of(s))
      if (!(skip && f.r == skip->r && f.o == skip->o)) use(f);
    if (extra) use(*extra);
    return 1.0 / (1.0 + std::exp(-lo));
  }

 private:
  static double logit(double p) {
    p = std::min(1.0 - 1e-9, std::max(1e-9, p));
    return std::log(p / (1.0 - p));
  }

  std::vector<double> rule_confidences(const std::string& s, const std::string& r, const std::string& o,
                                       const TripleStore::Fact& f, double b) const {
    std::vector<double> out;
    double body = 0.0, both = 0.0;
    for (const std::string& z : st_.subjects_with(f.r, f.o)) {
      if (z == s || !st_.has_relation(z, r)) continue;
      body += 1.0;
      both += st_.has(z, r, o) ? 1.0 : 0.0;
    }
    if (body > 0.0) out.push_back((both + cfg_.m * b) / (body + cfg_.m));
    if (cfg_.copy_rules && f.o == o && f.r != r) {
      const auto [pair, same] = copy_stats(r, f.r);
      if (pair > 0.0) out.push_back((same + cfg_.m * b) / (pair + cfg_.m));
    }
    return out;
  }

  /// Subjects with both relations, and those whose r and r2 share a value (cached per pair).
  std::pair<double, double> copy_stats(const std::string& r, const std::string& r2) const {
    const std::string k = r + '\x1f' + r2;
    auto it = copy_.find(k);
    if (it != copy_.end()) return it->second;
    double pair = 0.0, same = 0.0;
    for (const auto& [z, fs] : st_.subjects()) {
      if (!st_.has_relation(z, r) || !st_.has_relation(z, r2)) continue;
      pair += 1.0;
      bool shared = false;
      for (const auto& f : fs)
        if (f.r == r && st_.has(z, r2, f.o)) shared = true;
      same += shared ? 1.0 : 0.0;
    }
    return copy_[k] = {pair, same};
  }

  const TripleStore& st_;
  Config cfg_;
  mutable std::unordered_map<std::string, std::pair<double, double>> copy_;
};

/// Compression gain (6.2), bits: coding s's facts with the link predictor as the model, what h saves on s's other facts
/// minus what h itself costs. dL = sum_f [log2 P(f | rest + h) - log2 P(f | rest)] + log2 P(h | facts of s).
inline double compression_gain_bits(const LinkPredictor& lp, const TripleStore& st, const std::string& s,
                                    const std::string& r, const std::string& o) {
  const TripleStore::Fact h{r, o};
  double gain = std::log2(std::max(1e-12, lp.prob(s, r, o)));
  for (const auto& f : st.facts_of(s)) {
    if (f.r == r && f.o == o) continue;
    gain += std::log2(std::max(1e-12, lp.prob(s, f.r, f.o, &f, &h))) - std::log2(std::max(1e-12, lp.prob(s, f.r, f.o, &f)));
  }
  return gain;
}

// ---------------------------------------------------------------------------
// 4.3 Tiers
// ---------------------------------------------------------------------------

enum class Tier { Conjecture, Speculation, Wild };

inline const char* tier_name(Tier t) {
  switch (t) {
    case Tier::Conjecture: return "conjecture";
    case Tier::Speculation: return "speculation";
    case Tier::Wild: return "wild";
  }
  return "?";
}

/// The 4.3 table: P0 >= 0.3 conjecture, 0.05 <= P0 < 0.3 speculation, below that wild. Shares are starting values.
struct TierConfig {
  double conjecture_p0 = 0.3;
  double speculation_p0 = 0.05;
  double share[3] = {0.70, 0.25, 0.05};
};

inline Tier tier_of(double p0, const TierConfig& cfg = {}) {
  if (p0 >= cfg.conjecture_p0) return Tier::Conjecture;
  if (p0 >= cfg.speculation_p0) return Tier::Speculation;
  return Tier::Wild;
}

inline double tier_budget_share(Tier t, const TierConfig& cfg = {}) { return cfg.share[static_cast<int>(t)]; }

// ---------------------------------------------------------------------------
// 4.2 Lifecycle
// ---------------------------------------------------------------------------

enum class HypState { Proposed, Scored, Queued, UnderTest, Supported, Refuted, Parked, Irreducible, Promoted, Retired };

inline const char* hyp_state_name(HypState s) {
  switch (s) {
    case HypState::Proposed: return "proposed";
    case HypState::Scored: return "scored";
    case HypState::Queued: return "queued";
    case HypState::UnderTest: return "under test";
    case HypState::Supported: return "supported";
    case HypState::Refuted: return "refuted";
    case HypState::Parked: return "parked";
    case HypState::Irreducible: return "irreducible";
    case HypState::Promoted: return "promoted";
    case HypState::Retired: return "retired";
  }
  return "?";
}

/// The arrows of the 4.2 diagram, plus the two the text names: "otherwise it waits in the queue" (under test -> queued,
/// and supported -> queued when ECS does not admit it yet) and "parked ... re-scored when the store changes"
/// (parked -> scored). Any open state may be retired. Promoted and retired are final. Nothing is ever deleted.
inline bool legal_transition(HypState from, HypState to) {
  using S = HypState;
  if (from == S::Promoted || from == S::Retired) return false;
  if (to == S::Retired) return from != S::UnderTest;  // a running test finishes first
  switch (from) {
    case S::Proposed: return to == S::Scored;
    case S::Scored: return to == S::Queued;  // a P0 below the floor still queues, in the wild tier
    case S::Queued: return to == S::UnderTest || to == S::Parked;
    case S::UnderTest:
      return to == S::Supported || to == S::Refuted || to == S::Parked || to == S::Irreducible || to == S::Queued;
    case S::Supported: return to == S::Promoted || to == S::Queued;
    case S::Parked: return to == S::Scored;
    default: return false;  // Refuted, Irreducible: only retired
  }
}

// ---------------------------------------------------------------------------
// 4.1 The record
// ---------------------------------------------------------------------------

struct HypScores {
  double plausibility = 0.0, novelty_bits = 0.0, impact = 0.0, testability = 0.0, cost = 1.0, eig = 0.0, voi = 0.0;
};

/// One pre-registered test (7.2 step 1) and, once run, its result.
struct HypTest {
  std::string kind;              // e.g. "T5 Wikidata"
  std::string source;            // where its evidence comes from; evidence from the generator itself is never independent
  std::string predicted;         // the predicted outcome
  std::string refuted_if;        // what would count as refutation
  bool disconfirmation = false;  // the plan searches for evidence against (I6)
  bool independent = false;      // evidence independent of the generator (I4)
  double sensitivity = 0.9;
  double specificity = 0.9;
  double source_trust = 1.0;     // tau of the ECS evidence item it yields on promotion ("at full mass": 1)
  bool done = false;
  bool positive = false;
  double bayes_factor = 1.0;
  double posterior_after = 0.0;
};

struct HypHistory {
  int cycle = 0;
  HypState from = HypState::Proposed;
  HypState to = HypState::Proposed;
  std::string cause;
  double posterior = 0.0;
};

struct Hypothesis {
  std::string id;
  std::string key;          // store form, e.g. "capital|Atlantis|Poseidonia"; regeneration is blocked on it (I7)
  std::string statement;    // the sentence
  std::string generator;    // G1..G20
  int generator_version = 1;
  std::vector<std::string> premises;  // derivation: premise ids with their ECS P
  std::vector<double> premise_p;
  std::string rule;         // rule or analogy used ("" = none)
  std::string params;
  double prior = 0.0;       // P0 (6.1): the generator's support is a prior, never evidence
  double posterior = 0.0;
  HypScores scores;
  std::vector<HypTest> tests;
  HypState state = HypState::Proposed;
  Tier tier = Tier::Wild;
  bool hypothetical = true;  // taint: always set until promotion
  int refuted_cycle = -1;
  std::vector<HypHistory> history;
};

/// I4: evidence counts as independent only when the test says so and does not come from the generator itself.
inline bool independent_evidence(const Hypothesis& h, const HypTest& t) {
  return t.done && t.independent && t.source != h.generator;
}

inline bool has_independent_evidence(const Hypothesis& h) {
  for (const HypTest& t : h.tests)
    if (independent_evidence(h, t)) return true;
  return false;
}

// ---------------------------------------------------------------------------
// 7.2 Updating and deciding
// ---------------------------------------------------------------------------

/// BF = P(result | h) / P(result | not h): s / (1 - c) for a positive result, (1 - s) / c for a negative one.
/// s and c are kept inside [1e-6, 1 - 1e-6] so a factor is never 0 or infinite.
inline double bayes_factor(bool positive, double s, double c) {
  constexpr double e = 1e-6;
  s = std::min(1.0 - e, std::max(e, s));
  c = std::min(1.0 - e, std::max(e, c));
  return positive ? s / (1.0 - c) : (1.0 - s) / c;
}

/// Posterior from a prior and a Bayes factor: O1 = O0 x BF.
inline double update_posterior(double prior, double bf) {
  prior = detail::ecs_clamp01(prior);
  if (prior <= 0.0 || prior >= 1.0) return prior;
  const double o = prior / (1.0 - prior) * bf;
  return o / (1.0 + o);
}

struct HypothesisConfig {
  TierConfig tiers;
  double supported_p = 0.9;       // 7.2: supported above 0.9, with independent evidence
  double refuted_p = 0.05;        // refuted below 0.05
  int irreducible_k = 5;          // K = 5 tests ...
  double irreducible_move = 0.05; // ... that moved the posterior by less than 0.05 in total (sum of |change|)
  int cooling_cycles = 10;        // 7.2 step 6: a refutation blocks regeneration for 10 cycles
  FusionConfig promotion;         // ECS on promotion; its base rate is replaced by the prior (7.4)
};

/// 7.2 step 5 for a hypothesis under test.
inline HypState decide(const Hypothesis& h, const HypothesisConfig& cfg = {}) {
  if (h.posterior > cfg.supported_p && has_independent_evidence(h)) return HypState::Supported;
  if (h.posterior < cfg.refuted_p) return HypState::Refuted;
  int done = 0;
  double moved = 0.0, last = h.prior;
  for (const HypTest& t : h.tests) {
    if (!t.done) continue;
    ++done;
    moved += std::fabs(t.posterior_after - last);
    last = t.posterior_after;
  }
  if (done >= cfg.irreducible_k && moved < cfg.irreducible_move) return HypState::Irreducible;
  return HypState::Queued;
}

// ---------------------------------------------------------------------------
// 1 The firewall
// ---------------------------------------------------------------------------

enum class Mode { Answer, Discovery };
enum class StoreKind { Facts, Hypotheses, GapMap, Journal };

/// Answer mode reads only the fact store (and there only Known and Believed, untainted items). Discovery reads everything.
inline bool may_read(Mode m, StoreKind s) { return m == Mode::Discovery || s == StoreKind::Facts; }

/// Discovery writes only the hypothesis store, the gap map and the journal. Answer mode writes only demand into the gap
/// map (a "don't know" tag, 3.5). Nothing writes facts except promotion through ECS (HypothesisStore::promote).
inline bool may_write(Mode m, StoreKind s) {
  if (m == Mode::Discovery) return s != StoreKind::Facts;
  return s == StoreKind::GapMap;
}

/// An item of the fact store as the firewall sees it: a fact, rule, concept link or sense.
struct KnowledgeItem {
  std::string id;
  Opinion opinion;
  std::optional<Opinion> rival;       // a rival value's opinion, for the Contested check
  bool hypothetical = false;          // the taint flag
  std::vector<std::string> premises;  // ids it was derived from; empty = observed
};

class KnowledgeStore {
 public:
  /// Adds or replaces an item. A derived item inherits the taint of any tainted premise, whatever flag it arrives with.
  void add(KnowledgeItem item) {
    for (const std::string& p : item.premises)
      if (tainted(p)) item.hypothetical = true;
    items_[item.id] = std::move(item);
  }

  const KnowledgeItem* find(const std::string& id) const {
    auto it = items_.find(id);
    return it == items_.end() ? nullptr : &it->second;
  }

  /// Taint through derivation: the item's own flag, or any premise tainted, followed to the observed items. Fails closed:
  /// a missing premise, a cycle, or a chain deeper than the item count counts as tainted.
  bool tainted(const std::string& id) const {
    std::vector<std::string> path;
    return tainted_rec(id, path);
  }

  /// I1 and I10: the item, if answer mode may use it (Known or Believed, not tainted directly or through derivation).
  const KnowledgeItem* answerable(const std::string& id, const ClassifyConfig& cfg = {}) const {
    const KnowledgeItem* it = find(id);
    if (!it || tainted(id)) return nullptr;
    const EpistemicState s = classify(it->opinion, it->rival, cfg);
    return state_use(s, it->opinion.grade).answer ? it : nullptr;
  }

  /// Everything answer mode may use, in id order.
  std::vector<const KnowledgeItem*> answer_view(const ClassifyConfig& cfg = {}) const {
    std::vector<const KnowledgeItem*> out;
    for (const auto& kv : items_)
      if (const KnowledgeItem* it = answerable(kv.first, cfg)) out.push_back(it);
    return out;
  }

  std::size_t size() const { return items_.size(); }

 private:
  bool tainted_rec(const std::string& id, std::vector<std::string>& path) const {
    const KnowledgeItem* it = find(id);
    if (!it || it->hypothetical) return true;
    if (std::find(path.begin(), path.end(), id) != path.end() || path.size() > items_.size()) return true;
    path.push_back(id);
    for (const std::string& p : it->premises)
      if (tainted_rec(p, path)) return true;
    path.pop_back();
    return false;
  }

  std::map<std::string, KnowledgeItem> items_;
};

// ---------------------------------------------------------------------------
// 4, 7.4, 7.5 The hypothesis store
// ---------------------------------------------------------------------------

class HypothesisStore {
 public:
  explicit HypothesisStore(HypothesisConfig cfg = {}) : cfg_(std::move(cfg)) {}

  /// I7: a refuted hypothesis with the same key blocks regeneration for the cooling period, unless new evidence arrived.
  bool blocked(const std::string& key, bool new_evidence = false) const {
    if (new_evidence) return false;
    for (const Hypothesis& h : hyps_)
      if (h.key == key && h.refuted_cycle >= 0 && cycle_ - h.refuted_cycle < cfg_.cooling_cycles) return true;
    return false;
  }

  /// Adds a proposed hypothesis. Refused (false) when blocked or when the id is taken. It is always hypothetical.
  bool propose(Hypothesis h, bool new_evidence = false) {
    if (blocked(h.key, new_evidence) || find(h.id)) return false;
    h.state = HypState::Proposed;
    h.hypothetical = true;
    h.refuted_cycle = -1;
    h.history.push_back({cycle_, HypState::Proposed, HypState::Proposed, new_evidence ? "proposed (new evidence)" : "proposed", h.prior});
    hyps_.push_back(std::move(h));
    return true;
  }

  /// Moves a hypothesis along a legal arrow of 4.2 and logs it. Illegal moves change nothing and return false.
  bool transition(const std::string& id, HypState to, const std::string& cause) {
    Hypothesis* h = find_mut(id);
    if (!h || !legal_transition(h->state, to)) return false;
    h->history.push_back({cycle_, h->state, to, cause, h->posterior});
    h->state = to;
    return true;
  }

  /// proposed or parked -> scored (or a scored one re-scored in place): sets P0, the posterior's start and the tier.
  bool score(const std::string& id, double p0, const HypScores& s = {}) {
    Hypothesis* h = find_mut(id);
    if (!h) return false;
    if (h->state == HypState::Scored) h->history.push_back({cycle_, HypState::Scored, HypState::Scored, "re-scored", p0});
    else if (!transition(id, HypState::Scored, "scored")) return false;
    h->prior = h->posterior = detail::ecs_clamp01(p0);
    h->scores = s;
    h->scores.plausibility = h->prior;
    h->tier = tier_of(h->prior, cfg_.tiers);
    return true;
  }

  /// Records the result of pre-registered test `k` of a hypothesis under test, updates the posterior and applies 7.2
  /// step 5. A refutation is logged, kept, and counted against the generator and the rule (7.5).
  std::optional<HypState> record_result(const std::string& id, std::size_t k, bool positive) {
    Hypothesis* h = find_mut(id);
    if (!h || h->state != HypState::UnderTest || k >= h->tests.size() || h->tests[k].done) return std::nullopt;
    HypTest& t = h->tests[k];
    t.done = true;
    t.positive = positive;
    t.bayes_factor = bayes_factor(positive, t.sensitivity, t.specificity);
    h->posterior = t.posterior_after = update_posterior(h->posterior, t.bayes_factor);
    const HypState next = decide(*h, cfg_);
    transition(id, next, t.kind + (positive ? " positive" : " negative"));
    if (next == HypState::Refuted) {
      h->refuted_cycle = cycle_;
      generators_[h->generator].observe(false);
      if (!h->rule.empty()) rules_[h->rule].observe(false);
    }
    return next;
  }

  /// 7.4: a supported hypothesis enters ECS with its independent test evidence at full mass; the generator's support is
  /// only the base rate a. If ECS admits it (Admit), it is promoted: the taint is cleared, it is written to the fact store,
  /// and the generator gets the credit. Otherwise nothing changes and it stays supported. The opinion is returned either way.
  std::optional<Opinion> promote(const std::string& id, KnowledgeStore& facts) {
    Hypothesis* h = find_mut(id);
    if (!h || h->state != HypState::Supported) return std::nullopt;
    std::vector<EvidenceItem> items;
    for (const HypTest& t : h->tests) {
      if (!independent_evidence(*h, t)) continue;  // I4: dependent evidence never enters ECS
      EvidenceItem e;
      e.stance = t.positive ? Stance::Support : Stance::Oppose;
      e.tau = t.source_trust;
      items.push_back(e);
    }
    FusionConfig fc = cfg_.promotion;
    fc.base_rate = h->prior;
    const Opinion op = fuse(items, fc);
    if (items.empty() || admission_decision(op) != Admission::Admit) return op;
    transition(id, HypState::Promoted, "ECS admitted on independent evidence");
    h->hypothetical = false;
    generators_[h->generator].observe(true);
    if (!h->rule.empty()) rules_[h->rule].observe(true);
    facts.add(KnowledgeItem{h->id, op, std::nullopt, false, {}});
    return op;
  }

  /// Parked hypotheses are re-scored when the store changes (4.2).
  int rescore_parked() {
    int n = 0;
    for (Hypothesis& h : hyps_)
      if (h.state == HypState::Parked && transition(h.id, HypState::Scored, "store changed")) ++n;
    return n;
  }

  void advance_cycle() { ++cycle_; }
  int cycle() const { return cycle_; }

  const Hypothesis* find(const std::string& id) const {
    for (const Hypothesis& h : hyps_)
      if (h.id == id) return &h;
    return nullptr;
  }
  const std::vector<Hypothesis>& all() const { return hyps_; }

  /// Reliability Betas (start at Beta(1, 1)): a promotion is a hit, a refutation a miss (7.4, 7.5).
  BetaPosterior generator_reliability(const std::string& g) const {
    auto it = generators_.find(g);
    return it == generators_.end() ? BetaPosterior{} : it->second;
  }
  BetaPosterior rule_reliability(const std::string& r) const {
    auto it = rules_.find(r);
    return it == rules_.end() ? BetaPosterior{} : it->second;
  }

 private:
  Hypothesis* find_mut(const std::string& id) {
    for (Hypothesis& h : hyps_)
      if (h.id == id) return &h;
    return nullptr;
  }

  HypothesisConfig cfg_;
  std::vector<Hypothesis> hyps_;  // never erased: refuted and retired records stay (I7)
  std::map<std::string, BetaPosterior> generators_;
  std::map<std::string, BetaPosterior> rules_;
  int cycle_ = 0;
};

// ---------------------------------------------------------------------------
// I8 Discovery output is always labelled
// ---------------------------------------------------------------------------

/// True when a text starts with the SPECULATION or the HYPOTHETICAL label of section 10.
inline bool is_labelled(const std::string& text) {
  const std::string spec = hypothesis_label();
  const std::string hyp = std::string("HYPOTHETICAL ") + epi_text::kEmDash + " assuming: ";
  return text.compare(0, spec.size(), spec) == 0 || text.compare(0, hyp.size(), hyp) == 0;
}

/// Any discovery-mode text: kept as it is when labelled, else the SPECULATION label goes first on its own line.
inline std::string label_discovery_text(const std::string& text) {
  return is_labelled(text) ? text : hypothesis_label() + "\n" + text;
}

/// The hypothesis card of a record, in every state and tier.
inline std::string discovery_card(const Hypothesis& h) {
  HypothesisCard c;
  c.statement = h.statement;
  c.from = h.generator + (h.rule.empty() ? "" : ": " + h.rule);
  c.plausibility = h.prior;
  c.novelty_bits = h.scores.novelty_bits;
  c.tests_planned.clear();
  for (std::size_t i = 0; i < h.tests.size(); ++i) c.tests_planned += (i ? ", " : "") + h.tests[i].kind;
  c.status = std::string(hyp_state_name(h.state)) + " (" + tier_name(h.tier) + " tier)";
  return label_discovery_text(format_hypothesis_card(c));
}

}  // namespace cypha::eval
