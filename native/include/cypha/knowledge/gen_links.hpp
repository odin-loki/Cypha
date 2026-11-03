#pragma once
// Cypha discovery generators G6 and G7 (DISCOVERY section 5; S8b, docs/reports/S8B_LINK_PREDICTION.md): link prediction
// over the store. They propose hypotheses for Cypha's own store and never answer a question for CyphaLM (PLAN rule 10).
//
//   G6  neighbourhood: common neighbours, Jaccard, Adamic-Adar (sum of 1 / ln deg z over shared neighbours z) and
//       resource allocation (sum of 1 / deg z, the degree-normalised form, so hubs count least).
//   G7  path ranking (PRA): random-walk probabilities along typed relation paths of length 1-3 (both directions), the
//       slot's own edge left out, combined by an L2 logistic regression per relation. A hub guard stops a walk at an
//       intermediate node with too many edges of the step's relation; longer paths are ranked down when paths are kept.
//       Each score keeps its strongest path as the explanation.
//   LinkScorer: S6's link predictor (hypothesis.hpp) for every candidate of one slot in one pass, the same numbers as
//       LinkPredictor::prob for a triple not in the store.
//   Logistic: the small L2 logistic regression G7 and S8b's combination use (Newton steps).
//
// Author: Odin Loch

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cypha/eval/hypothesis.hpp"

namespace cypha::eval {

// ---------------------------------------------------------------------------
// G6: neighbourhood scores
// ---------------------------------------------------------------------------

struct NeighbourScore {
  double cn = 0.0, jaccard = 0.0, aa = 0.0, ra = 0.0;
};

/// G6: node a's neighbourhood scores with every node that shares a neighbour with it. AA here equals hypothesis.hpp's
/// adamic_adar(st, a, b) for each such b.
inline std::unordered_map<std::string, NeighbourScore> neighbour_scores(const TripleStore& st, const std::string& a) {
  std::unordered_map<std::string, NeighbourScore> out;
  const auto& na = st.neighbours(a);
  for (const std::string& z : na) {
    const auto& nz = st.neighbours(z);
    const double aa = 1.0 / std::log(static_cast<double>(std::max<std::size_t>(2, nz.size())));
    const double ra = 1.0 / static_cast<double>(std::max<std::size_t>(1, nz.size()));
    for (const std::string& b : nz) {
      if (b == a) continue;
      NeighbourScore& x = out[b];
      x.cn += 1.0, x.aa += aa, x.ra += ra;
    }
  }
  for (auto& [b, x] : out) {
    const double uni = static_cast<double>(na.size() + st.neighbours(b).size()) - x.cn;
    x.jaccard = uni > 0.0 ? x.cn / uni : 0.0;
  }
  return out;
}

// ---------------------------------------------------------------------------
// S6's link predictor, every candidate of a slot in one pass
// ---------------------------------------------------------------------------

/// LinkPredictor::prob(s, r, o) for every o of `cands` at once: one pass over the subjects that share a body fact with s,
/// instead of one per candidate. The arithmetic and its order are LinkPredictor's, so the values are the same.
class LinkScorer {
 public:
  LinkScorer(const TripleStore& st, LinkPredictor::Config cfg) : st_(st), cfg_(cfg) {}

  std::vector<double> probs(const std::string& s, const std::string& r, const std::vector<std::string>& cands) const {
    struct Body {
      const TripleStore::Fact* f = nullptr;
      double body = 0.0;
      std::unordered_map<std::string, double> both;
    };
    std::vector<Body> bodies;
    for (const auto& f : st_.facts_of(s)) {
      Body B;
      B.f = &f;
      for (const std::string& z : st_.subjects_with(f.r, f.o)) {
        if (z == s || !st_.has_relation(z, r)) continue;
        B.body += 1.0;
        for (const auto& g : st_.facts_of(z))
          if (g.r == r) B.both[g.o] += 1.0;
      }
      bodies.push_back(std::move(B));
    }
    std::vector<double> out;
    out.reserve(cands.size());
    for (const std::string& o : cands) {
      const bool self = st_.has(s, r, o);
      const double b = std::min(1.0 - 1e-9, base_rate(st_, r, o, 1.0, self));
      double lo = logit(b);
      for (const Body& B : bodies) {
        const TripleStore::Fact& f = *B.f;
        if (f.r == r && f.o == o) continue;
        if (B.body > 0.0) {
          const auto it = B.both.find(o);
          const double both = it == B.both.end() ? 0.0 : it->second;
          lo += logit((both + cfg_.m * b) / (B.body + cfg_.m)) - logit(b);
        }
        if (cfg_.copy_rules && f.o == o && f.r != r) {
          const auto [pair, same] = copy_stats(r, f.r);
          if (pair > 0.0) lo += logit((same + cfg_.m * b) / (pair + cfg_.m)) - logit(b);
        }
      }
      out.push_back(1.0 / (1.0 + std::exp(-lo)));
    }
    return out;
  }

 private:
  static double logit(double p) {
    p = std::min(1.0 - 1e-9, std::max(1e-9, p));
    return std::log(p / (1.0 - p));
  }
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
  LinkPredictor::Config cfg_;
  mutable std::unordered_map<std::string, std::pair<double, double>> copy_;
};

// ---------------------------------------------------------------------------
// Logistic regression (L2 on the weights, not the intercept; Newton steps)
// ---------------------------------------------------------------------------

struct Logistic {
  std::vector<double> w;
  double b = 0.0;
  bool ok = false;

  double logit(const std::vector<double>& x) const {
    double z = b;
    for (std::size_t i = 0; i < w.size() && i < x.size(); ++i) z += w[i] * x[i];
    return z;
  }
  double prob(const std::vector<double>& x) const { return 1.0 / (1.0 + std::exp(-logit(x))); }

  /// Fits on rows X (all of one length) and labels y. Needs both labels present; otherwise ok stays false.
  void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y, double l2 = 1.0, int iters = 50) {
    ok = false;
    const std::size_t n = std::min(X.size(), y.size());
    if (n == 0) return;
    std::size_t pos = 0;
    for (std::size_t i = 0; i < n; ++i) pos += y[i] ? 1 : 0;
    if (pos == 0 || pos == n) return;
    const std::size_t d = X[0].size(), D = d + 1;  // the intercept is the last coordinate
    std::vector<double> beta(D, 0.0);
    beta[d] = std::log(static_cast<double>(pos) / static_cast<double>(n - pos));
    for (int it = 0; it < iters; ++it) {
      std::vector<double> g(D, 0.0), H(D * D, 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        double z = beta[d];
        for (std::size_t j = 0; j < d; ++j) z += beta[j] * X[i][j];
        const double p = 1.0 / (1.0 + std::exp(-z)), wgt = std::max(1e-10, p * (1.0 - p)), res = (y[i] ? 1.0 : 0.0) - p;
        for (std::size_t j = 0; j < D; ++j) {
          const double xj = j < d ? X[i][j] : 1.0;
          g[j] += res * xj;
          for (std::size_t k = 0; k <= j; ++k) H[j * D + k] += wgt * xj * (k < d ? X[i][k] : 1.0);
        }
      }
      for (std::size_t j = 0; j < D; ++j)
        for (std::size_t k = 0; k < j; ++k) H[k * D + j] = H[j * D + k];
      for (std::size_t j = 0; j < d; ++j) g[j] -= l2 * beta[j], H[j * D + j] += l2;
      std::vector<double> step = solve(H, g, D);
      double norm = 0.0;
      for (std::size_t j = 0; j < D; ++j) beta[j] += step[j], norm += step[j] * step[j];
      if (norm < 1e-16) break;
    }
    w.assign(beta.begin(), beta.begin() + static_cast<std::ptrdiff_t>(d));
    b = beta[d];
    ok = true;
  }

 private:
  /// Gaussian elimination with partial pivoting; a singular pivot leaves that coordinate's step at 0.
  static std::vector<double> solve(std::vector<double> A, std::vector<double> rhs, std::size_t D) {
    for (std::size_t c = 0; c < D; ++c) {
      std::size_t piv = c;
      for (std::size_t r = c + 1; r < D; ++r)
        if (std::fabs(A[r * D + c]) > std::fabs(A[piv * D + c])) piv = r;
      if (std::fabs(A[piv * D + c]) < 1e-12) continue;
      if (piv != c) {
        for (std::size_t k = 0; k < D; ++k) std::swap(A[c * D + k], A[piv * D + k]);
        std::swap(rhs[c], rhs[piv]);
      }
      for (std::size_t r = 0; r < D; ++r) {
        if (r == c) continue;
        const double f = A[r * D + c] / A[c * D + c];
        if (f == 0.0) continue;
        for (std::size_t k = c; k < D; ++k) A[r * D + k] -= f * A[c * D + k];
        rhs[r] -= f * rhs[c];
      }
    }
    std::vector<double> x(D, 0.0);
    for (std::size_t c = 0; c < D; ++c)
      if (std::fabs(A[c * D + c]) >= 1e-12) x[c] = rhs[c] / A[c * D + c];
    return x;
  }
};

// ---------------------------------------------------------------------------
// G7: the typed graph and path ranking
// ---------------------------------------------------------------------------

/// The store as a typed graph: an edge s -r-> o of type 2k (k the relation's index) and its reverse o -r^-1-> s of type
/// 2k + 1. Built from the subjects in sorted order, so ids do not depend on hash order.
class RelGraph {
 public:
  explicit RelGraph(const TripleStore& st) {
    std::vector<std::string> subs;
    subs.reserve(st.subjects().size());
    for (const auto& kv : st.subjects()) subs.push_back(kv.first);
    std::sort(subs.begin(), subs.end());
    std::vector<std::vector<std::pair<int, int>>> adj;
    auto id = [&](const std::string& x) {
      auto [it, fresh] = node_.emplace(x, static_cast<int>(names_.size()));
      if (fresh) names_.push_back(x), adj.emplace_back();
      return it->second;
    };
    for (const std::string& s : subs)
      for (const auto& f : st.facts_of(s)) {
        auto [it, fresh] = rel_.emplace(f.r, static_cast<int>(rels_.size()));
        if (fresh) rels_.push_back(f.r);
        const int a = id(s), b = id(f.o), t = 2 * it->second;
        if (a == b) continue;
        adj[static_cast<std::size_t>(a)].push_back({t, b});
        adj[static_cast<std::size_t>(b)].push_back({t + 1, a});
      }
    types_.resize(adj.size()), targets_.resize(adj.size());
    for (std::size_t n = 0; n < adj.size(); ++n) {
      std::sort(adj[n].begin(), adj[n].end());
      for (const auto& [t, v] : adj[n]) types_[n].push_back(t), targets_[n].push_back(v);
    }
  }

  int node(const std::string& x) const {
    auto it = node_.find(x);
    return it == node_.end() ? -1 : it->second;
  }
  const std::string& name(int n) const { return names_[static_cast<std::size_t>(n)]; }
  std::size_t nodes() const { return names_.size(); }
  /// The forward type of relation r, or -1.
  int forward(const std::string& r) const {
    auto it = rel_.find(r);
    return it == rel_.end() ? -1 : 2 * it->second;
  }
  std::string type_name(int t) const { return rels_[static_cast<std::size_t>(t / 2)] + (t % 2 ? "^-1" : ""); }
  /// The targets of node n's edges of type t, as a [first, last) index range into targets(n).
  std::pair<std::size_t, std::size_t> range(int n, int t) const {
    const auto& ty = types_[static_cast<std::size_t>(n)];
    const auto lo = std::lower_bound(ty.begin(), ty.end(), t), hi = std::upper_bound(lo, ty.end(), t);
    return {static_cast<std::size_t>(lo - ty.begin()), static_cast<std::size_t>(hi - ty.begin())};
  }
  const std::vector<int>& targets(int n) const { return targets_[static_cast<std::size_t>(n)]; }
  const std::vector<int>& types(int n) const { return types_[static_cast<std::size_t>(n)]; }

 private:
  std::unordered_map<std::string, int> node_, rel_;
  std::vector<std::string> names_, rels_;
  std::vector<std::vector<int>> types_, targets_;
};

/// An edge left out of a walk (with its reverse): the slot's own fact.
struct SkipEdge {
  int a = -1, t = -1, b = -1;
  bool hits(int u, int ty, int v) const { return (u == a && ty == t && v == b) || (u == b && ty == (t ^ 1) && v == a); }
};

/// G7, path ranking per relation (PRA).
class PathRanker {
 public:
  struct Config {
    int max_paths = 40;      // paths kept per relation
    int min_support = 3;     // training facts a path must reach
    int train_facts = 300;   // facts of r for path discovery and training
    int negatives = 4;       // drawn objects per training fact
    int hub = 1000;          // an intermediate node with more edges of the step's type is not expanded
    double l2 = 1.0;
    double len_decay = 0.5;  // paths kept by support x len_decay^(length - 1)
    std::uint64_t seed = 20261029;
  };
  struct Model {
    std::vector<std::array<int, 3>> paths;  // types; -1 past the path's end
    Logistic lr;
    bool ok = false;
  };

  PathRanker(const TripleStore& st, const RelGraph& g, Config cfg) : st_(st), g_(g), cfg_(cfg) {
    mass_.assign(g.nodes(), 0.0);
    next_.assign(g.nodes(), 0.0);
  }

  static int path_len(const std::array<int, 3>& p) { return p[2] >= 0 ? 3 : p[1] >= 0 ? 2 : 1; }
  std::string path_name(const std::array<int, 3>& p) const {
    std::string out;
    for (int i = 0; i < path_len(p); ++i) out += (i ? " > " : "") + g_.type_name(p[static_cast<std::size_t>(i)]);
    return out;
  }

  /// Fits relation r's model on the store's facts of r (each with its own edge left out). Cached.
  const Model& fit(const std::string& r) {
    auto it = models_.find(r);
    if (it != models_.end()) return it->second;
    Model& M = models_[r];
    const int R = g_.forward(r);
    if (R < 0) return M;
    // The facts of r, from the subjects in sorted order, shuffled with a seed of their own.
    std::vector<std::pair<std::string, std::string>> facts;
    std::vector<std::string> pool;  // r's objects, one entry per triple (negatives are drawn by triple)
    std::vector<std::string> subs;
    for (const auto& kv : st_.subjects()) subs.push_back(kv.first);
    std::sort(subs.begin(), subs.end());
    for (const std::string& s : subs)
      for (const auto& f : st_.facts_of(s))
        if (f.r == r) facts.push_back({s, f.o}), pool.push_back(f.o);
    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a of r: the same shuffle on every platform
    for (const unsigned char ch : r) h = (h ^ ch) * 1099511628211ULL;
    std::mt19937_64 rng(cfg_.seed ^ h);
    for (std::size_t i = facts.size(); i > 1; --i) std::swap(facts[i - 1], facts[rng() % i]);
    if (facts.size() > static_cast<std::size_t>(cfg_.train_facts)) facts.resize(static_cast<std::size_t>(cfg_.train_facts));
    // Path discovery: the distinct paths that reach o from s for each training fact, by meeting in the middle.
    std::map<std::array<int, 3>, int> support;
    for (const auto& [s, o] : facts) {
      const int a = g_.node(s), b = g_.node(o);
      if (a < 0 || b < 0) continue;
      std::set<std::array<int, 3>> got;
      reaching_paths(a, b, SkipEdge{a, R, b}, got);
      for (const auto& p : got) ++support[p];
    }
    std::vector<std::pair<double, std::array<int, 3>>> ranked;
    for (const auto& [p, n] : support)
      if (n >= cfg_.min_support) ranked.push_back({-static_cast<double>(n) * std::pow(cfg_.len_decay, path_len(p) - 1), p});
    std::sort(ranked.begin(), ranked.end());
    for (std::size_t i = 0; i < ranked.size() && i < static_cast<std::size_t>(cfg_.max_paths); ++i) M.paths.push_back(ranked[i].second);
    if (M.paths.empty()) return M;
    // Training rows: each fact's object against drawn objects, from the same walks (the fact's edge left out).
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    for (const auto& [s, o] : facts) {
      const int a = g_.node(s), b = g_.node(o);
      if (a < 0 || b < 0) continue;
      const auto dist = walks(a, M.paths, SkipEdge{a, R, b});
      X.push_back(features(dist, b)), y.push_back(1);
      for (int k = 0, tries = 0; k < cfg_.negatives && tries < 50; ++tries) {
        const std::string& c = pool[rng() % pool.size()];
        if (c == o || st_.has(s, r, c)) continue;
        const int cn = g_.node(c);
        X.push_back(features(dist, cn)), y.push_back(0);
        ++k;
      }
    }
    M.lr.fit(X, y, cfg_.l2);
    M.ok = M.lr.ok;
    return M;
  }

  /// G7's logit for each candidate of (s, r); `best`, when given, gets each candidate's strongest path index (-1: none).
  std::vector<double> score(const std::string& s, const std::string& r, const std::vector<std::string>& cands,
                            std::vector<int>* best = nullptr) {
    const Model& M = fit(r);
    std::vector<double> out(cands.size(), 0.0);
    if (best) best->assign(cands.size(), -1);
    if (!M.ok) return out;
    const int a = g_.node(s);
    const auto dist = a < 0 ? std::vector<std::unordered_map<int, double>>(M.paths.size()) : walks(a, M.paths, SkipEdge{});
    for (std::size_t i = 0; i < cands.size(); ++i) {
      const auto x = features(dist, g_.node(cands[i]));
      out[i] = M.lr.logit(x);
      if (best) {
        double top = 0.0;
        for (std::size_t k = 0; k < x.size(); ++k)
          if (M.lr.w[k] * x[k] > top) top = M.lr.w[k] * x[k], (*best)[i] = static_cast<int>(k);
      }
    }
    return out;
  }

  const Model* model(const std::string& r) const {
    auto it = models_.find(r);
    return it == models_.end() ? nullptr : &it->second;
  }

  /// The random-walk distribution from node a along each path (a map node -> probability per path).
  std::vector<std::unordered_map<int, double>> walks(int a, const std::vector<std::array<int, 3>>& paths, SkipEdge skip) {
    std::vector<std::unordered_map<int, double>> out(paths.size());
    for (std::size_t k = 0; k < paths.size(); ++k) {
      std::vector<int> cur{a}, nxt;
      mass_[static_cast<std::size_t>(a)] = 1.0;
      const int L = path_len(paths[k]);
      for (int step = 0; step < L; ++step) {
        const int t = paths[k][static_cast<std::size_t>(step)];
        for (const int u : cur) {
          const double p = mass_[static_cast<std::size_t>(u)];
          mass_[static_cast<std::size_t>(u)] = 0.0;
          const auto [lo, hi] = g_.range(u, t);
          const auto& tg = g_.targets(u);
          std::size_t n = hi - lo;
          if (step > 0 && n > static_cast<std::size_t>(cfg_.hub)) continue;  // the hub guard (raw count, as in discovery)
          for (std::size_t e = lo; e < hi; ++e)
            if (skip.hits(u, t, tg[e])) --n;
          if (n == 0) continue;
          const double share = p / static_cast<double>(n);
          for (std::size_t e = lo; e < hi; ++e) {
            const int v = tg[e];
            if (skip.hits(u, t, v)) continue;
            if (next_[static_cast<std::size_t>(v)] == 0.0) nxt.push_back(v);
            next_[static_cast<std::size_t>(v)] += share;
          }
        }
        cur.swap(nxt);
        nxt.clear();
        for (const int v : cur) mass_[static_cast<std::size_t>(v)] = next_[static_cast<std::size_t>(v)], next_[static_cast<std::size_t>(v)] = 0.0;
      }
      for (const int v : cur) {
        if (mass_[static_cast<std::size_t>(v)] > 0.0) out[k][v] = mass_[static_cast<std::size_t>(v)];
        mass_[static_cast<std::size_t>(v)] = 0.0;
      }
    }
    return out;
  }

 private:
  static std::vector<double> features(const std::vector<std::unordered_map<int, double>>& dist, int node) {
    std::vector<double> x(dist.size(), 0.0);
    if (node < 0) return x;
    for (std::size_t k = 0; k < dist.size(); ++k) {
      const auto it = dist[k].find(node);
      if (it != dist[k].end()) x[k] = std::log1p(1000.0 * it->second);
    }
    return x;
  }

  bool expandable(int u, int t, int step) const {
    const auto [lo, hi] = g_.range(u, t);
    return step == 0 || hi - lo <= static_cast<std::size_t>(cfg_.hub);
  }

  /// Every path of length 1-3 from a that reaches b, with the edge `skip` left out and the hub guard on intermediate nodes.
  void reaching_paths(int a, int b, SkipEdge skip, std::set<std::array<int, 3>>& got) const {
    // The last step into b: c -t3-> b for each edge b -t-> c (t3 = t ^ 1).
    std::unordered_map<int, std::vector<int>> into;
    {
      const auto& ty = g_.types(b);
      const auto& tg = g_.targets(b);
      for (std::size_t e = 0; e < ty.size(); ++e) {
        const int c = tg[e], t3 = ty[e] ^ 1;
        if (skip.hits(c, t3, b)) continue;
        into[c].push_back(t3);
      }
    }
    const auto& ty1 = g_.types(a);
    const auto& tg1 = g_.targets(a);
    for (std::size_t e1 = 0; e1 < ty1.size(); ++e1) {
      const int t1 = ty1[e1], m1 = tg1[e1];
      if (skip.hits(a, t1, m1)) continue;
      if (m1 == b) got.insert({t1, -1, -1});
      const auto& ty2 = g_.types(m1);
      const auto& tg2 = g_.targets(m1);
      for (std::size_t e2 = 0; e2 < ty2.size(); ++e2) {
        const int t2 = ty2[e2], m2 = tg2[e2];
        if (skip.hits(m1, t2, m2) || !expandable(m1, t2, 1)) continue;
        if (m2 == b) got.insert({t1, t2, -1});
        const auto it = into.find(m2);
        if (it == into.end()) continue;
        for (const int t3 : it->second)
          if (expandable(m2, t3, 2)) got.insert({t1, t2, t3});
      }
    }
  }

  const TripleStore& st_;
  const RelGraph& g_;
  Config cfg_;
  std::unordered_map<std::string, Model> models_;
  std::vector<double> mass_, next_;
};

}  // namespace cypha::eval
