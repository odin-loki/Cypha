// Generators G1 (deep deduction) and G20 (the wild tier) for Cypha's discovery track
// (docs/DISCOVERY.md section 5; docs/reports/S12A_DEDUCTION_LAWS_REFINEMENT.md).
//
// Header-only. Reads a TripleStore (hypothesis.hpp), G2's rules (gen_rules.hpp) and S9's worlds (worlds.hpp).
//
//   deep_chain     G1: forward chaining of the kept G2 rules inside a world, from every fact of the store, up to a depth.
//                  A fact first derived at depth d keeps its premises and its chain confidence (the product of the
//                  rules' PCA confidences). Depth 1 is G2's own prediction; G1's hypotheses are depth >= 2.
//   g1_prior       G1's prior by depth (S12c): the chain confidence, a cap at depth 3, or calibrated by depth from graded
//                  counts ((true + 1) / (graded + 2)); clamped to [0.02, 0.98].
//   G19 (refinement) was removed by S12c's registered rule (it failed its earn-its-place read and no richer store is
//   scheduled); its code is in git at main's 0ce072f.
//   wild_tier      G20: for a slot (s, r), a value drawn from r's objects in another cluster (type and country), kept
//                  when the link predictor scores it at least twice the base rate; a fixed budget.
//
// Nothing here answers a question. Derived facts and wild pairs are hypotheses (PLAN rule 10).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

/// G1's prior by depth (S12c item 3). Chain: the chain confidence. Cap3: the chain confidence up to depth 3; deeper facts
/// are not hypotheses (NaN). Depth: calibrated from graded counts at the fact's depth, (true + 1) / (graded + 2).
enum class G1Prior { Chain, Cap3, Depth };

struct DepthCalib {
  std::map<int, std::pair<double, double>> tg;  // depth -> (true, graded)
  void add(int depth, bool truth) {
    auto& x = tg[depth];
    x.first += truth, x.second += 1;
  }
  double prior(int depth) const {
    const auto it = tg.find(depth);
    const double t = it == tg.end() ? 0.0 : it->second.first, g = it == tg.end() ? 0.0 : it->second.second;
    return (t + 1.0) / (g + 2.0);
  }
};

inline double clamp_prior(double p) { return std::min(0.98, std::max(0.02, p)); }

inline double g1_prior(G1Prior mode, int depth, double conf, const DepthCalib& cal) {
  switch (mode) {
    case G1Prior::Chain: return clamp_prior(conf);
    case G1Prior::Cap3: return depth <= 3 ? clamp_prior(conf) : std::numeric_limits<double>::quiet_NaN();
    case G1Prior::Depth: return clamp_prior(cal.prior(depth));
  }
  return clamp_prior(conf);
}

/// Facts by relation: r -> (s, o) pairs.
inline std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> facts_by_relation(const TripleStore& st) {
  std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> m;
  for (const auto& [s, fs] : st.subjects())
    for (const auto& f : fs) m[f.r].push_back({s, f.o});
  for (auto& [r, v] : m) std::sort(v.begin(), v.end());
  return m;
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
