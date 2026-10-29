// Generator G15 (numeric law discovery) for Cypha's discovery track (docs/DISCOVERY.md section 5 and P9;
// docs/reports/S12A_DEDUCTION_LAWS_REFINEMENT.md).
//
// Header-only. Standard library, lci.hpp (Benjamini-Hochberg), ecs.hpp (Wilson), gen_rules.hpp (binomial tail and the
// hidden third of entities).
//
//   NumTable     one type: entities by numeric attributes. Each attribute has a scale: ratio (positive quantities, so
//                power products are allowed) or interval (years: only differences and orders are allowed). A ratio of
//                two years is never a law (the units check).
//   search_laws  BACON-style search. BACON tries X/Y when X and Y rise together and X*Y when one falls as the other
//                rises, then repeats on the new term; that walk reaches exactly the integer power products, so they are
//                enumerated directly (pairs |e| <= 3, triples |e| <= 2). Interval attributes give differences Y - X and
//                order laws X <= Y. A law is kept when it is nearly constant (robust CV < 0.1 with 80% of entities within
//                10% of the median; for a difference, a robust spread under 0.1 of its attributes' spreads), when it is
//                not chance (chi-square lower tail of the observed log-variance against the variance under independent
//                attributes, Benjamini-Hochberg over every formula of the table), when it is the shortest (a formula in
//                the span of laws already kept is redundant) and when it holds on the held-out third of the entities.
//   plant_*      the planted-law gate's tables: ratio laws (Z = c X / Y, c X Y, c X^1.5, c X) on synthetic entities, and
//                interval laws (Z = birth + c, birth + a gap) on real persons; the null shuffles Z.
//
// Nothing here answers a question. Laws are hypotheses about Cypha's own store (PLAN rule 10).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cypha/eval/ecs.hpp"
#include "cypha/eval/gen_rules.hpp"
#include "cypha/eval/lci.hpp"

namespace cypha::eval::laws {

enum class Scale { Ratio, Interval };

struct NumTable {
  std::string type;
  std::vector<std::string> entities;
  std::vector<std::string> attrs;
  std::vector<Scale> scale;
  std::vector<std::vector<double>> v;  // [attribute][entity]; NaN when missing

  std::size_t add_attr(const std::string& name, Scale sc) {
    attrs.push_back(name);
    scale.push_back(sc);
    v.emplace_back(entities.size(), std::numeric_limits<double>::quiet_NaN());
    return attrs.size() - 1;
  }
  std::size_t add_entity(const std::string& e) {
    entities.push_back(e);
    for (auto& col : v) col.push_back(std::numeric_limits<double>::quiet_NaN());
    return entities.size() - 1;
  }
  bool has(std::size_t a, std::size_t i) const { return !std::isnan(v[a][i]); }
};

// ---------------------------------------------------------------------------
// Robust statistics and the chi-square tail
// ---------------------------------------------------------------------------

inline double median_of(std::vector<double> x) {
  if (x.empty()) return std::numeric_limits<double>::quiet_NaN();
  const std::size_t m = x.size() / 2;
  std::nth_element(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(m), x.end());
  const double hi = x[m];
  if (x.size() % 2 == 1) return hi;
  const double lo = *std::max_element(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(m));
  return 0.5 * (lo + hi);
}

/// 1.4826 * MAD: a robust standard deviation (equal to the SD for normal data).
inline double robust_sd(const std::vector<double>& x) {
  if (x.size() < 2) return 0.0;
  const double m = median_of(x);
  std::vector<double> d(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) d[i] = std::fabs(x[i] - m);
  return 1.4826 * median_of(d);
}

inline double classic_cv(const std::vector<double>& x) {
  if (x.size() < 2) return std::numeric_limits<double>::infinity();
  double s = 0, ss = 0;
  for (double v : x) s += v;
  const double mu = s / static_cast<double>(x.size());
  for (double v : x) ss += (v - mu) * (v - mu);
  const double sd = std::sqrt(ss / static_cast<double>(x.size() - 1));
  return mu != 0 ? std::fabs(sd / mu) : std::numeric_limits<double>::infinity();
}

/// Regularised lower incomplete gamma P(a, x) (series below a + 1, continued fraction above; Numerical Recipes).
inline double gamma_p(double a, double x) {
  if (x <= 0.0 || a <= 0.0) return 0.0;
  const double gln = std::lgamma(a);
  if (x < a + 1.0) {
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 1000; ++n) {
      ap += 1.0;
      del *= x / ap;
      sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-15) break;
    }
    return std::min(1.0, sum * std::exp(-x + a * std::log(x) - gln));
  }
  double b = x + 1.0 - a, c = 1.0 / 1e-300, d = 1.0 / b, h = d;
  for (int i = 1; i < 1000; ++i) {
    const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
    b += 2.0;
    d = an * d + b;
    if (std::fabs(d) < 1e-300) d = 1e-300;
    c = b + an / c;
    if (std::fabs(c) < 1e-300) c = 1e-300;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < 1e-15) break;
  }
  return std::max(0.0, 1.0 - std::exp(-x + a * std::log(x) - gln) * h);
}

/// P(chi-square with k degrees of freedom <= x).
inline double chi2_lower(double x, double k) { return gamma_p(0.5 * k, 0.5 * x); }

// ---------------------------------------------------------------------------
// Laws
// ---------------------------------------------------------------------------

enum class LawKind { Power, Diff, Order };

inline const char* law_kind_name(LawKind k) {
  switch (k) {
    case LawKind::Power: return "power";
    case LawKind::Diff: return "difference";
    case LawKind::Order: return "order";
  }
  return "?";
}

struct Law {
  LawKind kind = LawKind::Power;
  std::vector<std::size_t> attr;  // Power: the attributes; Diff and Order: {X, Y} for Y - X and X <= Y
  std::vector<int> exp;           // Power: the exponents (first positive, greatest common divisor 1)
  std::size_t n = 0;              // entities holding every attribute
  double constant = 0;            // the term's median on the training two thirds (Order: the share with X <= Y)
  double spread = 0;              // Power: robust CV of the term; Diff: robust spread over the attributes' smaller one
  double classic_cv = 0;          // Power only, for the report
  double inlier = 0;              // share of entities within the tolerance (Order: share with X <= Y)
  double p_value = 1.0;
  bool screen = false, bh = false, redundant = false, heldout_ok = false, kept = false;
  std::size_t heldout_n = 0;
  double heldout_share = 0;  // within tolerance (Order: share with X <= Y)
  double length = 0;         // description length: attributes + sum |e|
  double p0 = 0;             // Order: the share of X <= Y over pairs of different entities (the independence null)

  std::string text(const NumTable& t) const {
    if (kind == LawKind::Diff) return t.attrs[attr[1]] + " - " + t.attrs[attr[0]];
    if (kind == LawKind::Order) return t.attrs[attr[0]] + " <= " + t.attrs[attr[1]];
    std::string s;
    for (std::size_t i = 0; i < attr.size(); ++i) {
      if (!s.empty()) s += " * ";
      s += t.attrs[attr[i]];
      if (exp[i] != 1) s += "^" + std::to_string(exp[i]);
    }
    return s;
  }
};

struct LawConfig {
  std::size_t min_n = 30;
  double cv_max = 0.1;          // DISCOVERY: coefficient of variation < 0.1
  double inlier_min = 0.8;      // share of entities within the tolerance
  double ratio_tol = 0.1;       // power products: within 10% of the median
  double diff_tol_sd = 3.0;     // differences: within 3 robust spreads of the median
  int max_exp_pair = 3, max_exp_triple = 2;
  double q = 0.1;               // Benjamini-Hochberg over the table's formulas
  double order_share = 0.95;    // amendment 0
  double order_heldout_viol = 0.10;
  double heldout_min = 0.8;
  std::size_t heldout_min_n = 5;
};

namespace detail {

inline int gcd_int(int a, int b) {
  a = std::abs(a), b = std::abs(b);
  while (b) {
    const int t = a % b;
    a = b, b = t;
  }
  return a;
}

/// Exponent vectors of length k with entries in [-m, m] \ {0}, gcd 1, the first positive.
inline std::vector<std::vector<int>> exponent_vectors(int k, int m) {
  std::vector<std::vector<int>> out;
  std::vector<int> e(static_cast<std::size_t>(k), 0);
  std::function<void(int)> rec = [&](int i) {
    if (i == k) {
      if (e[0] <= 0) return;
      int g = 0;
      for (int x : e) g = gcd_int(g, x);
      if (g == 1) out.push_back(e);
      return;
    }
    for (int x = -m; x <= m; ++x) {
      if (x == 0) continue;
      e[static_cast<std::size_t>(i)] = x;
      rec(i + 1);
    }
  };
  rec(0);
  return out;
}

/// Whether v lies in the span of the rows (Gaussian elimination with partial pivoting).
inline bool in_span(const std::vector<std::vector<double>>& rows, const std::vector<double>& v) {
  if (rows.empty()) return false;
  auto rank = [](std::vector<std::vector<double>> m) {
    std::size_t r = 0;
    const std::size_t cols = m.empty() ? 0 : m[0].size();
    for (std::size_t c = 0; c < cols && r < m.size(); ++c) {
      std::size_t piv = r;
      for (std::size_t i = r; i < m.size(); ++i)
        if (std::fabs(m[i][c]) > std::fabs(m[piv][c])) piv = i;
      if (std::fabs(m[piv][c]) < 1e-9) continue;
      std::swap(m[piv], m[r]);
      for (std::size_t i = 0; i < m.size(); ++i) {
        if (i == r) continue;
        const double f = m[i][c] / m[r][c];
        for (std::size_t j = c; j < cols; ++j) m[i][j] -= f * m[r][j];
      }
      ++r;
    }
    return r;
  };
  auto with = rows;
  with.push_back(v);
  return rank(with) == rank(rows);
}

}  // namespace detail

/// The held-out part of a table's entities: S8a's hidden third (FNV-1a bits 13+ mod 3 == 1).
inline bool heldout_entity(const std::string& e) { return gen_hidden_third(e); }

/// Search one table. Returns every formula searched (kept or not), formulas first by length, then order laws.
inline std::vector<Law> search_laws(const NumTable& t, const LawConfig& cfg = {}) {
  std::vector<Law> out;
  const std::size_t A = t.attrs.size(), E = t.entities.size();
  std::vector<bool> held(E);
  for (std::size_t i = 0; i < E; ++i) held[i] = heldout_entity(t.entities[i]);
  // entities holding every attribute of a set (and positive values for ratio attributes)
  auto rows_of = [&](const std::vector<std::size_t>& as) {
    std::vector<std::size_t> r;
    for (std::size_t i = 0; i < E; ++i) {
      bool ok = true;
      for (std::size_t a : as) ok = ok && t.has(a, i) && (t.scale[a] == Scale::Interval || t.v[a][i] > 0);
      if (ok) r.push_back(i);
    }
    return r;
  };
  // ---- power products over ratio attributes
  std::vector<std::size_t> ratio, interval;
  for (std::size_t a = 0; a < A; ++a) (t.scale[a] == Scale::Ratio ? ratio : interval).push_back(a);
  const auto ex2 = detail::exponent_vectors(2, cfg.max_exp_pair), ex3 = detail::exponent_vectors(3, cfg.max_exp_triple);
  auto eval_power = [&](const std::vector<std::size_t>& as, const std::vector<int>& e, const std::vector<std::size_t>& rows) {
    Law L;
    L.kind = LawKind::Power, L.attr = as, L.exp = e, L.n = rows.size();
    L.length = static_cast<double>(as.size());
    for (int x : e) L.length += std::abs(x);
    std::vector<double> lt, tt, ltrain, lnull(as.size());
    for (std::size_t i : rows) {
      double l = 0;
      for (std::size_t k = 0; k < as.size(); ++k) l += e[k] * std::log(t.v[as[k]][i]);
      lt.push_back(l);
      tt.push_back(std::exp(l));
      if (!held[i]) ltrain.push_back(l);
    }
    const double med = std::exp(median_of(lt));
    L.spread = robust_sd(tt) / std::fabs(median_of(tt));
    L.classic_cv = classic_cv(tt);
    double in = 0;
    for (double x : tt) in += std::fabs(x / med - 1.0) <= cfg.ratio_tol;
    L.inlier = in / static_cast<double>(tt.size());
    L.screen = L.spread < cfg.cv_max && L.inlier >= cfg.inlier_min;
    double vnull = 0;
    for (std::size_t k = 0; k < as.size(); ++k) {
      std::vector<double> lx;
      for (std::size_t i : rows) lx.push_back(std::log(t.v[as[k]][i]));
      const double s = robust_sd(lx);
      vnull += e[k] * e[k] * s * s;
    }
    const double sobs = robust_sd(lt);
    const double nm1 = static_cast<double>(rows.size()) - 1.0;
    L.p_value = vnull > 0 ? chi2_lower(nm1 * sobs * sobs / vnull, nm1) : 1.0;
    L.constant = ltrain.empty() ? med : std::exp(median_of(ltrain));
    double hin = 0;
    for (std::size_t j = 0; j < rows.size(); ++j)
      if (held[rows[j]]) ++L.heldout_n, hin += std::fabs(tt[j] / L.constant - 1.0) <= cfg.ratio_tol;
    L.heldout_share = L.heldout_n ? hin / static_cast<double>(L.heldout_n) : 0.0;
    L.heldout_ok = L.heldout_n < cfg.heldout_min_n || L.heldout_share >= cfg.heldout_min;
    return L;
  };
  for (std::size_t i = 0; i < ratio.size(); ++i)
    for (std::size_t j = i + 1; j < ratio.size(); ++j) {
      const std::vector<std::size_t> as{ratio[i], ratio[j]};
      const auto rows = rows_of(as);
      if (rows.size() < cfg.min_n) continue;
      for (const auto& e : ex2) out.push_back(eval_power(as, e, rows));
    }
  for (std::size_t i = 0; i < ratio.size(); ++i)
    for (std::size_t j = i + 1; j < ratio.size(); ++j)
      for (std::size_t k = j + 1; k < ratio.size(); ++k) {
        const std::vector<std::size_t> as{ratio[i], ratio[j], ratio[k]};
        const auto rows = rows_of(as);
        if (rows.size() < cfg.min_n) continue;
        for (const auto& e : ex3) out.push_back(eval_power(as, e, rows));
      }
  // ---- differences and orders over interval attributes
  for (std::size_t i = 0; i < interval.size(); ++i)
    for (std::size_t j = 0; j < interval.size(); ++j) {
      if (i == j) continue;
      const std::size_t X = interval[i], Y = interval[j];
      const auto rows = rows_of({X, Y});
      if (rows.size() < cfg.min_n) continue;
      std::vector<double> d, dtrain, xs, ys;
      std::size_t le = 0, hle = 0, hn = 0;
      for (std::size_t r : rows) {
        const double x = t.v[X][r], y = t.v[Y][r];
        d.push_back(y - x), xs.push_back(x), ys.push_back(y);
        if (!held[r]) dtrain.push_back(y - x);
        le += x <= y;
        if (held[r]) ++hn, hle += x <= y;
      }
      const double nm1 = static_cast<double>(rows.size()) - 1.0;
      if (i < j) {  // the difference Y - X (once per unordered pair; X earlier in the table)
        Law L;
        L.kind = LawKind::Diff, L.attr = {X, Y}, L.n = rows.size(), L.length = 2 + 2;
        const double sx = robust_sd(xs), sy = robust_sd(ys), sd = robust_sd(d), med = median_of(d);
        const double base = std::min(sx, sy);
        L.spread = base > 0 ? sd / base : std::numeric_limits<double>::infinity();
        const double tol = std::max(cfg.diff_tol_sd * sd, 0.5);
        double in = 0;
        for (double v : d) in += std::fabs(v - med) <= tol;
        L.inlier = in / static_cast<double>(d.size());
        L.screen = L.spread < cfg.cv_max && L.inlier >= cfg.inlier_min;
        const double vnull = sx * sx + sy * sy;
        L.p_value = vnull > 0 ? chi2_lower(nm1 * sd * sd / vnull, nm1) : 1.0;
        L.constant = dtrain.empty() ? med : median_of(dtrain);
        const double tol_train = std::max(cfg.diff_tol_sd * robust_sd(dtrain.empty() ? d : dtrain), 0.5);
        double hin = 0;
        for (std::size_t k = 0; k < rows.size(); ++k)
          if (held[rows[k]]) ++L.heldout_n, hin += std::fabs(d[k] - L.constant) <= tol_train;
        L.heldout_share = L.heldout_n ? hin / static_cast<double>(L.heldout_n) : 0.0;
        L.heldout_ok = L.heldout_n < cfg.heldout_min_n || L.heldout_share >= cfg.heldout_min;
        out.push_back(L);
      }
      Law O;  // the order X <= Y
      O.kind = LawKind::Order, O.attr = {X, Y}, O.n = rows.size(), O.length = 3;
      O.inlier = O.constant = static_cast<double>(le) / static_cast<double>(rows.size());
      O.screen = O.inlier >= cfg.order_share;
      // amendment 1: the null is independence (as for the formulas), not a coin: p0 is the share of X <= Y over pairs
      // of different entities, so an order that only reflects shifted margins is not a law
      double cross = 0;
      for (std::size_t a = 0; a < xs.size(); ++a)
        for (std::size_t c = 0; c < ys.size(); ++c)
          if (a != c) cross += xs[a] <= ys[c];
      const double npairs = static_cast<double>(xs.size()) * static_cast<double>(xs.size() - 1);
      O.p0 = npairs > 0 ? cross / npairs : 1.0;
      O.p_value = O.p0 < 1.0 ? binomial_upper_tail(static_cast<double>(le), static_cast<double>(rows.size()), O.p0) : 1.0;
      O.heldout_n = hn;
      O.heldout_share = hn ? static_cast<double>(hle) / static_cast<double>(hn) : 0.0;
      O.heldout_ok = hn < cfg.heldout_min_n || 1.0 - O.heldout_share <= cfg.order_heldout_viol;
      out.push_back(O);
    }
  // ---- BH over every formula of the table
  std::vector<double> p;
  for (const Law& L : out) p.push_back(L.p_value);
  const auto bh = benjamini_hochberg(p, cfg.q);
  for (std::size_t i = 0; i < out.size(); ++i) out[i].bh = bh.reject[i];
  // ---- shortest first; a formula in the span of the kept ones is redundant
  std::vector<std::size_t> idx(out.size());
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    if (out[a].attr.size() != out[b].attr.size()) return out[a].attr.size() < out[b].attr.size();
    if (out[a].length != out[b].length) return out[a].length < out[b].length;
    return out[a].p_value < out[b].p_value;
  });
  std::vector<std::vector<double>> span_power, span_diff;
  for (std::size_t i : idx) {
    Law& L = out[i];
    if (!(L.screen && L.bh && L.heldout_ok)) continue;
    if (L.kind == LawKind::Order) {
      L.kept = true;
      continue;
    }
    std::vector<double> vec(A, 0.0);
    if (L.kind == LawKind::Power)
      for (std::size_t k = 0; k < L.attr.size(); ++k) vec[L.attr[k]] = L.exp[k];
    else
      vec[L.attr[0]] = -1, vec[L.attr[1]] = 1;
    auto& span = L.kind == LawKind::Power ? span_power : span_diff;
    if (detail::in_span(span, vec)) {
      L.redundant = true;
      continue;
    }
    span.push_back(vec);
    L.kept = true;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The planted-law gate
// ---------------------------------------------------------------------------

enum class PlantKind { Ratio, Product, Power, Scale, Offset, Order };
inline const char* plant_kind_name(PlantKind k) {
  switch (k) {
    case PlantKind::Ratio: return "RATIO";
    case PlantKind::Product: return "PRODUCT";
    case PlantKind::Power: return "POWER";
    case PlantKind::Scale: return "SCALE";
    case PlantKind::Offset: return "OFFSET";
    case PlantKind::Order: return "ORDER";
  }
  return "?";
}
inline bool plant_is_interval(PlantKind k) { return k == PlantKind::Offset || k == PlantKind::Order; }

struct Planted {
  PlantKind kind = PlantKind::Ratio;
  NumTable table, null_table;
  std::size_t z = 0;            // the planted attribute
  std::vector<std::size_t> independent;  // attributes no law may join (synthetic distractors)
  std::size_t base = 0;         // interval: the birth attribute
  double c = 0;                 // the planted constant
  double noise_sd = 0;          // interval: the noise SD in years; ratio: sigma
  std::vector<std::size_t> law_attr;  // ratio: the planted law's attributes
  std::vector<int> law_exp;           // ratio: its exponents (sign-normalised, table order)
};

/// Persons for interval tables: (name, birth year, death year).
struct PersonYears {
  std::string name;
  double birth = 0, death = 0;
};

namespace detail {
inline double lognormal(std::mt19937_64& rng, double mu, double sd) { return std::exp(std::normal_distribution<double>(mu, sd)(rng)); }
}  // namespace detail

/// One planted table and its null (Z shuffled across entities). sigma is the noise level; contamination the share of
/// entities whose Z is drawn independently. persons: the pool for interval kinds (at least n).
inline Planted plant(PlantKind kind, double sigma, std::uint64_t seed, const std::vector<PersonYears>& persons, std::size_t n = 40,
                     double contamination = 0.05) {
  std::mt19937_64 rng(seed);
  Planted P;
  P.kind = kind;
  NumTable& T = P.table;
  T.type = std::string("planted ") + plant_kind_name(kind);
  if (!plant_is_interval(kind)) {
    for (std::size_t i = 0; i < n; ++i) T.add_entity("e" + std::to_string(seed % 100000) + "_" + std::to_string(i));
    const char* names[] = {"X", "Y", "W1", "W2"};
    for (const char* nm : names) {
      const std::size_t a = T.add_attr(nm, Scale::Ratio);
      const double mu = std::normal_distribution<double>(std::log(100.0), 1.0)(rng);
      for (std::size_t i = 0; i < n; ++i) T.v[a][i] = detail::lognormal(rng, mu, 1.0);
    }
    P.independent = {0, 1, 2, 3};
    P.c = std::uniform_real_distribution<double>(0.5, 20.0)(rng);
    P.noise_sd = sigma;
    P.z = T.add_attr("Z", Scale::Ratio);
    for (std::size_t i = 0; i < n; ++i) {
      const double x = T.v[0][i], y = T.v[1][i];
      double z = 0;
      switch (kind) {
        case PlantKind::Ratio: z = P.c * x / y; break;
        case PlantKind::Product: z = P.c * x * y; break;
        case PlantKind::Power: z = P.c * std::pow(x, 1.5); break;
        default: z = P.c * x; break;
      }
      T.v[P.z][i] = z * std::exp(std::normal_distribution<double>(0.0, sigma)(rng));
    }
    switch (kind) {  // the planted law in table order (X=0, Y=1, Z=4), first exponent positive
      case PlantKind::Ratio: P.law_attr = {0, 1, 4}, P.law_exp = {1, -1, -1}; break;   // X / (Y Z) constant: Z = c X / Y
      case PlantKind::Product: P.law_attr = {0, 1, 4}, P.law_exp = {1, 1, -1}; break;  // X Y / Z constant
      case PlantKind::Power: P.law_attr = {0, 4}, P.law_exp = {3, -2}; break;          // X^3 / Z^2 constant
      default: P.law_attr = {0, 4}, P.law_exp = {1, -1}; break;                         // X / Z constant
    }
  } else {
    std::vector<std::size_t> pick(persons.size());
    std::iota(pick.begin(), pick.end(), std::size_t{0});
    std::shuffle(pick.begin(), pick.end(), rng);
    pick.resize(std::min(n, pick.size()));
    for (std::size_t i : pick) T.add_entity(persons[i].name);
    const std::size_t b = T.add_attr("birth", Scale::Interval), d = T.add_attr("death", Scale::Interval);
    std::vector<double> births;
    for (std::size_t k = 0; k < pick.size(); ++k) {
      T.v[b][k] = persons[pick[k]].birth, T.v[d][k] = persons[pick[k]].death;
      births.push_back(persons[pick[k]].birth);
    }
    const double lo = *std::min_element(births.begin(), births.end()), hi = *std::max_element(births.begin(), births.end());
    const std::size_t w = T.add_attr("W", Scale::Interval);
    for (std::size_t k = 0; k < pick.size(); ++k) T.v[w][k] = std::floor(std::uniform_real_distribution<double>(lo, hi + 1.0)(rng));
    P.independent = {w};
    P.base = b;
    const double S = robust_sd(births);
    P.noise_sd = sigma * S;
    P.z = T.add_attr("Z", Scale::Interval);
    P.c = kind == PlantKind::Offset ? std::uniform_real_distribution<double>(10.0, 60.0)(rng) : 0.0;
    for (std::size_t k = 0; k < pick.size(); ++k) {
      double z = T.v[b][k];
      if (kind == PlantKind::Offset) z += P.c;
      else z += std::fabs(std::normal_distribution<double>(0.25, 0.10)(rng)) * S;  // amendment 0
      T.v[P.z][k] = z + (P.noise_sd > 0 ? std::normal_distribution<double>(0.0, P.noise_sd)(rng) : 0.0);
    }
    for (std::size_t k = 0; k < pick.size(); ++k)  // contamination for interval kinds: Z uniform over the birth range
      if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < contamination)
        T.v[P.z][k] = std::floor(std::uniform_real_distribution<double>(lo, hi + 1.0)(rng));
  }
  if (!plant_is_interval(kind))
    for (std::size_t i = 0; i < n; ++i)  // contamination for ratio kinds: Z independent log-normal
      if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < contamination) T.v[P.z][i] = detail::lognormal(rng, std::log(100.0), 1.0);
  // the null: Z shuffled across entities
  P.null_table = T;
  P.null_table.type += " (null)";
  std::vector<double> z = T.v[P.z];
  std::shuffle(z.begin(), z.end(), rng);
  P.null_table.v[P.z] = z;
  return P;
}

/// The planted constant implied by a kept ratio law's term constant K, in the law's planted orientation.
inline double implied_c(PlantKind k, double K) {
  if (K <= 0) return 0;
  return k == PlantKind::Power ? 1.0 / std::sqrt(K) : 1.0 / K;  // X^3 / Z^2 = 1 / c^2; the others 1 / c
}

/// Whether a kept power law is the planted one: the planted attributes, exponents equal up to sign.
inline bool is_planted_power(const Planted& P, const Law& L) {
  if (L.kind != LawKind::Power || L.attr != P.law_attr) return false;
  bool same = true, neg = true;
  for (std::size_t k = 0; k < L.exp.size(); ++k) same = same && L.exp[k] == P.law_exp[k], neg = neg && L.exp[k] == -P.law_exp[k];
  return same || neg;
}

/// Whether the kept laws include the planted one (registered: exponents up to sign and the constant c within 10%; OFFSET
/// within 3 standard errors of the median, or 0.5 year at noise 0; ORDER the order law birth <= Z).
inline bool planted_found(const Planted& P, const std::vector<Law>& laws) {
  const std::size_t n = P.table.entities.size();
  for (const Law& L : laws) {
    if (!L.kept) continue;
    switch (P.kind) {
      case PlantKind::Offset: {
        if (L.kind != LawKind::Diff || L.attr[0] != P.base || L.attr[1] != P.z) break;
        const double tol = P.noise_sd > 0 ? 3.0 * 1.2533 * P.noise_sd / std::sqrt(static_cast<double>(n)) : 0.5;
        if (std::fabs(L.constant - P.c) <= tol) return true;
        break;
      }
      case PlantKind::Order:
        if (L.kind == LawKind::Order && L.attr[0] == P.base && L.attr[1] == P.z) return true;
        break;
      default: {
        if (!is_planted_power(P, L)) break;
        bool same = true;
        for (std::size_t k = 0; k < L.exp.size(); ++k) same = same && L.exp[k] == P.law_exp[k];
        const double K = same ? L.constant : 1.0 / L.constant;
        if (std::fabs(implied_c(P.kind, K) / P.c - 1.0) <= 0.1) return true;
      }
    }
  }
  return false;
}

/// Kept laws that are false in a planted table. Ratio tables: in a null table every kept law; otherwise every kept law
/// but the planted one. Interval tables: any law with the distractor W; in a null table, any law with Z as well (laws
/// among birth, death and an unshuffled Z follow from true structure and are not counted).
inline std::size_t false_laws(const Planted& P, const std::vector<Law>& laws, bool null_table) {
  std::size_t k = 0;
  for (const Law& L : laws) {
    if (!L.kept) continue;
    bool has_z = false, has_indep = false;
    for (std::size_t a : L.attr) {
      has_z = has_z || a == P.z;
      has_indep = has_indep || std::find(P.independent.begin(), P.independent.end(), a) != P.independent.end();
    }
    if (!plant_is_interval(P.kind)) {
      if (null_table || !is_planted_power(P, L)) ++k;
    } else if (has_indep || (null_table && has_z)) {
      ++k;
    }
  }
  return k;
}

}  // namespace cypha::eval::laws
