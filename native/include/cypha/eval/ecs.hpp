// ECS: Evidence Certainty Score (docs/PLAN.md section 7).
//
// Header-only, standard library only. Implements:
//   - Beta posterior helpers (mean, variance, regularized incomplete beta,
//     quantile by bisection, central credible interval) and Wilson bounds
//   - evidence items with mass m = omega * tau_mean * pi * theta * delta (7.1)
//   - Subjective-Logic fusion with functionality, constraint penalties,
//     prior weight W and base rate a (7.2), grades (7.3)
//   - answer confidence and risk-coverage / abstain threshold (7.4)
//   - ECE, Brier, reliability bins, isotonic regression (PAV) (7.5)
//   - admission rules as small functions (7.6)
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cypha::eval {

// ---------------------------------------------------------------------------
// Beta distribution helpers
// ---------------------------------------------------------------------------

inline double beta_mean(double a, double b) { return a / (a + b); }

inline double beta_variance(double a, double b) {
  const double s = a + b;
  return a * b / (s * s * (s + 1.0));
}

namespace detail {

// Continued fraction for the incomplete beta function (modified Lentz).
inline double betacf(double a, double b, double x) {
  constexpr int kMaxIter = 20000;
  constexpr double kEps = 1e-15;
  constexpr double kTiny = 1e-300;
  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::fabs(d) < kTiny) d = kTiny;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= kMaxIter; ++m) {
    const int m2 = 2 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kTiny) d = kTiny;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kTiny) d = kTiny;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < kEps) break;
  }
  return h;
}

inline double ecs_clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

}  // namespace detail

/// Regularized incomplete beta I_x(a, b) = Beta-CDF at x, a,b > 0.
inline double reg_inc_beta(double x, double a, double b) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  const double lbt = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                     a * std::log(x) + b * std::log1p(-x);
  const double bt = std::exp(lbt);
  if (x < (a + 1.0) / (a + b + 2.0)) return bt * detail::betacf(a, b, x) / a;
  return 1.0 - bt * detail::betacf(b, a, 1.0 - x) / b;
}

/// Beta quantile (inverse CDF) by bisection on the monotone CDF.
inline double beta_quantile(double p, double a, double b) {
  if (p <= 0.0) return 0.0;
  if (p >= 1.0) return 1.0;
  double lo = 0.0;
  double hi = 1.0;
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (reg_inc_beta(mid, a, b) < p) {
      lo = mid;
    } else {
      hi = mid;
    }
    if (hi - lo < 1e-15) break;
  }
  return 0.5 * (lo + hi);
}

struct Interval {
  double lo = 0.0;
  double hi = 1.0;
};

/// Central credible interval of Beta(a, b) with total mass `level` (0.9 -> 5%..95%).
inline Interval beta_credible_interval(double a, double b, double level = 0.90) {
  const double tail = 0.5 * (1.0 - level);
  return Interval{beta_quantile(tail, a, b), beta_quantile(1.0 - tail, a, b)};
}

/// Wilson score interval bounds for `successes` out of `n` (default z = 1.96,
/// i.e. 95% two-sided).
inline double wilson_lower(double successes, double n, double z = 1.959963984540054) {
  if (!(n > 0.0)) return 0.0;
  const double p = detail::ecs_clamp01(successes / n);
  const double z2 = z * z;
  const double denom = 1.0 + z2 / n;
  const double centre = (p + z2 / (2.0 * n)) / denom;
  const double half = z * std::sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom;
  return std::max(0.0, centre - half);
}

inline double wilson_upper(double successes, double n, double z = 1.959963984540054) {
  if (!(n > 0.0)) return 1.0;
  const double p = detail::ecs_clamp01(successes / n);
  const double z2 = z * z;
  const double denom = 1.0 + z2 / n;
  const double centre = (p + z2 / (2.0 * n)) / denom;
  const double half = z * std::sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom;
  return std::min(1.0, centre + half);
}

/// A Beta posterior over a probability: source trust tau_s, pattern precision,
/// rule reliability. observe(true) adds 1 to alpha, observe(false) adds 1 to beta.
struct BetaPosterior {
  double alpha = 1.0;
  double beta = 1.0;
  double mean() const { return beta_mean(alpha, beta); }
  double variance() const { return beta_variance(alpha, beta); }
  Interval interval(double level = 0.90) const { return beta_credible_interval(alpha, beta, level); }
  void observe(bool verified_correct, double weight = 1.0) {
    (verified_correct ? alpha : beta) += weight;
  }
};

// Source-trust priors from PLAN 7.1.
inline BetaPosterior prior_unknown_source() { return BetaPosterior{1.0, 1.0}; }
inline BetaPosterior prior_wikidata() { return BetaPosterior{50.0, 1.0}; }
inline BetaPosterior prior_distilled_nn() { return BetaPosterior{1.0, 3.0}; }
inline BetaPosterior prior_infobox() { return BetaPosterior{20.0, 1.0}; }

// ---------------------------------------------------------------------------
// 7.1 Evidence items
// ---------------------------------------------------------------------------

/// Independence weight for an item in a near-duplicate cluster of n items:
/// the cluster carries total weight 1 + ln n, shared equally: (1 + ln n) / n.
inline double independence_weight(std::size_t n) {
  if (n == 0) return 0.0;
  return (1.0 + std::log(static_cast<double>(n))) / static_cast<double>(n);
}

/// Per-item independence weights from a cluster id for each item.
inline std::vector<double> independence_weights(const std::vector<int>& cluster_ids) {
  std::unordered_map<int, std::size_t> count;
  for (int id : cluster_ids) ++count[id];
  std::vector<double> w(cluster_ids.size());
  for (std::size_t i = 0; i < cluster_ids.size(); ++i) w[i] = independence_weight(count[cluster_ids[i]]);
  return w;
}

/// Time factor theta = exp(-dt / h). h <= 0 or non-finite means a stable
/// relation (no decay, theta = 1). dt < 0 is treated as 0.
inline double time_decay(double dt, double h) {
  if (!(h > 0.0) || !std::isfinite(h)) return 1.0;
  if (dt < 0.0) dt = 0.0;
  return std::exp(-dt / h);
}

constexpr int kMaxDerivationDepth = 3;

/// Depth of a derived item: 1 + max depth of its premises (observed facts have depth 0).
inline int derivation_depth(const std::vector<int>& premise_depths) {
  int mx = 0;
  for (int d : premise_depths) mx = std::max(mx, d);
  return mx + 1;
}

/// Derivation factor delta: product of premise ECS values times the rule's
/// reliability (Beta posterior mean). depth <= 0 means directly observed
/// (delta = 1). depth > kMaxDerivationDepth (3) gives delta = 0.
inline double derivation_factor(const std::vector<double>& premise_ecs, double rule_reliability,
                                int depth) {
  if (depth <= 0) return 1.0;
  if (depth > kMaxDerivationDepth) return 0.0;
  double prod = detail::ecs_clamp01(rule_reliability);
  for (double p : premise_ecs) prod *= detail::ecs_clamp01(p);
  return prod;
}

enum class Stance { Support, Oppose, CompetingValue };

/// One observation about a fact. All factors are in [0, 1].
struct EvidenceItem {
  Stance stance = Stance::Support;
  double omega = 1.0;  // independence
  double tau = 1.0;    // source trust (posterior mean tau-bar)
  double pi = 1.0;     // extraction confidence
  double theta = 1.0;  // time
  double delta = 1.0;  // derivation

  /// m_e = omega * tau * pi * theta * delta.
  double mass() const {
    return detail::ecs_clamp01(omega) * detail::ecs_clamp01(tau) * detail::ecs_clamp01(pi) *
           detail::ecs_clamp01(theta) * detail::ecs_clamp01(delta);
  }
};

/// Convenience constructor using a source's Beta posterior mean for tau.
inline EvidenceItem make_evidence(Stance stance, const BetaPosterior& source, double extraction_conf,
                                  double omega = 1.0, double theta = 1.0, double delta = 1.0) {
  EvidenceItem e;
  e.stance = stance;
  e.tau = source.mean();
  e.pi = extraction_conf;
  e.omega = omega;
  e.theta = theta;
  e.delta = delta;
  return e;
}

// ---------------------------------------------------------------------------
// 7.3 Grades (declared before Opinion, which carries one)
// ---------------------------------------------------------------------------

enum class Grade { Contradicted = 0, VeryLow = 1, Low = 2, Moderate = 3, High = 4 };

inline const char* grade_name(Grade g) {
  switch (g) {
    case Grade::High: return "High";
    case Grade::Moderate: return "Moderate";
    case Grade::Low: return "Low";
    case Grade::VeryLow: return "Very low";
    case Grade::Contradicted: return "Contradicted";
  }
  return "?";
}

inline bool grade_at_least(Grade g, Grade minimum) {
  return static_cast<int>(g) >= static_cast<int>(minimum);
}

/// Pure grade function of (P, u, d, lower 90% bound), checked in the order
/// Contradicted, High, Moderate, Low, Very low.
inline Grade grade_from(double P, double u, double d, double lower90) {
  if (d >= 0.5) return Grade::Contradicted;
  if (lower90 >= 0.90) return Grade::High;
  if (P >= 0.75 && u <= 0.3) return Grade::Moderate;
  if (P >= 0.5) return Grade::Low;
  return Grade::VeryLow;  // P < 0.5 and d < 0.5
}

// ---------------------------------------------------------------------------
// 7.2 Fusion
// ---------------------------------------------------------------------------

struct FusionConfig {
  double W = 2.0;             // prior weight
  double base_rate = 0.5;     // a
  double phi = 0.0;           // functionality of the relation (weights competing values)
  int hard_violations = 0;    // hard type violations, +hard_penalty each to s
  int soft_violations = 0;    // soft violations, +soft_penalty each to s
  double hard_penalty = 5.0;  // W_c
  double soft_penalty = 1.0;
  double ci_level = 0.90;
};

struct Opinion {
  double r = 0.0;  // positive evidence
  double s = 0.0;  // negative evidence (total)
  double s_oppose = 0.0;       // part of s: opposing items
  double s_competing = 0.0;    // part of s: phi * competing-value mass
  double s_constraint = 0.0;   // part of s: constraint penalties
  double b = 0.0;
  double d = 0.0;
  double u = 1.0;
  double base_rate = 0.5;
  double P = 0.5;  // b + a*u
  double alpha = 1.0;  // Beta(r + aW, s + (1-a)W)
  double beta = 1.0;
  double lo = 0.0;  // credible interval at FusionConfig::ci_level
  double hi = 1.0;
  Grade grade = Grade::Low;
};

/// Subjective-Logic opinion + Beta interval + grade from positive / negative counts.
inline Opinion opinion_from_counts(double r, double s, const FusionConfig& cfg = {}) {
  Opinion o;
  o.r = std::max(0.0, r);
  o.s = std::max(0.0, s);
  const double a = detail::ecs_clamp01(cfg.base_rate);
  const double W = std::max(0.0, cfg.W);
  o.base_rate = a;
  const double den = o.r + o.s + W;
  if (den > 0.0) {
    o.b = o.r / den;
    o.d = o.s / den;
    o.u = W / den;
  } else {
    o.b = 0.0;
    o.d = 0.0;
    o.u = 1.0;
  }
  o.P = o.b + a * o.u;
  // Guard degenerate Beta parameters (a = 0 or 1 with no evidence).
  o.alpha = std::max(o.r + a * W, 1e-9);
  o.beta = std::max(o.s + (1.0 - a) * W, 1e-9);
  const Interval iv = beta_credible_interval(o.alpha, o.beta, cfg.ci_level);
  o.lo = iv.lo;
  o.hi = iv.hi;
  o.grade = grade_from(o.P, o.u, o.d, o.lo);
  return o;
}

/// Fuse evidence items into an opinion about a fact (PLAN 7.2).
inline Opinion fuse(const std::vector<EvidenceItem>& items, const FusionConfig& cfg = {}) {
  double r = 0.0;
  double s_opp = 0.0;
  double s_comp = 0.0;
  for (const EvidenceItem& e : items) {
    const double m = e.mass();
    switch (e.stance) {
      case Stance::Support: r += m; break;
      case Stance::Oppose: s_opp += m; break;
      case Stance::CompetingValue: s_comp += m; break;
    }
  }
  const double phi = detail::ecs_clamp01(cfg.phi);
  const double s_con = cfg.hard_violations * cfg.hard_penalty + cfg.soft_violations * cfg.soft_penalty;
  const double s = s_opp + phi * s_comp + s_con;
  Opinion o = opinion_from_counts(r, s, cfg);
  o.s_oppose = s_opp;
  o.s_competing = phi * s_comp;
  o.s_constraint = s_con;
  return o;
}

// ---------------------------------------------------------------------------
// 7.4 Answer confidence, risk-coverage, abstain threshold
// ---------------------------------------------------------------------------

/// conf(answer) = ECS(fact) * P(question parse) * prod P(entity link) * P(answer type).
inline double answer_confidence(double ecs_p, double p_parse, const std::vector<double>& p_links,
                                double p_type) {
  double c = ecs_p * p_parse * p_type;
  for (double p : p_links) c *= p;
  return c;
}

struct RiskCoveragePoint {
  double threshold = 0.0;  // answer iff conf >= threshold
  double coverage = 0.0;   // fraction of questions answered
  double accuracy = 0.0;   // accuracy among answered
  std::size_t answered = 0;
};

/// Risk-coverage curve: one point per distinct confidence value, coverage increasing.
inline std::vector<RiskCoveragePoint> risk_coverage_curve(const std::vector<double>& conf,
                                                          const std::vector<int>& correct) {
  std::vector<RiskCoveragePoint> out;
  const std::size_t n = std::min(conf.size(), correct.size());
  if (n == 0) return out;
  std::vector<std::size_t> idx(n);
  for (std::size_t i = 0; i < n; ++i) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(),
                   [&](std::size_t x, std::size_t y) { return conf[x] > conf[y]; });
  std::size_t hits = 0;
  for (std::size_t k = 0; k < n; ++k) {
    hits += correct[idx[k]] ? 1u : 0u;
    const bool group_end = (k + 1 == n) || (conf[idx[k + 1]] != conf[idx[k]]);
    if (group_end) {
      RiskCoveragePoint p;
      p.threshold = conf[idx[k]];
      p.answered = k + 1;
      p.coverage = static_cast<double>(k + 1) / static_cast<double>(n);
      p.accuracy = static_cast<double>(hits) / static_cast<double>(k + 1);
      out.push_back(p);
    }
  }
  return out;
}

struct AbstainChoice {
  bool found = false;
  double threshold = std::numeric_limits<double>::infinity();  // answer iff conf >= threshold
  double coverage = 0.0;
  double accuracy = 0.0;
};

/// Highest-coverage threshold whose answered-set accuracy is >= target_precision.
inline AbstainChoice abstain_threshold(const std::vector<double>& conf, const std::vector<int>& correct,
                                       double target_precision) {
  AbstainChoice best;
  for (const RiskCoveragePoint& p : risk_coverage_curve(conf, correct)) {
    if (p.accuracy >= target_precision) {  // coverage increases along the curve; keep the last
      best.found = true;
      best.threshold = p.threshold;
      best.coverage = p.coverage;
      best.accuracy = p.accuracy;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// 7.5 Calibration: Brier, reliability bins, ECE, isotonic regression
// ---------------------------------------------------------------------------

inline double brier_score(const std::vector<double>& p, const std::vector<int>& y) {
  const std::size_t n = std::min(p.size(), y.size());
  if (n == 0) return 0.0;
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double e = p[i] - (y[i] ? 1.0 : 0.0);
    s += e * e;
  }
  return s / static_cast<double>(n);
}

struct ReliabilityBin {
  double lo = 0.0;
  double hi = 0.0;
  std::size_t count = 0;
  double mean_conf = 0.0;
  double accuracy = 0.0;
};

/// Equal-width reliability bins on [0, 1] (p = 1 goes in the last bin).
inline std::vector<ReliabilityBin> reliability_bins(const std::vector<double>& p,
                                                    const std::vector<int>& y, int nbins = 10) {
  if (nbins < 1) nbins = 1;
  std::vector<ReliabilityBin> bins(static_cast<std::size_t>(nbins));
  for (int k = 0; k < nbins; ++k) {
    bins[static_cast<std::size_t>(k)].lo = static_cast<double>(k) / nbins;
    bins[static_cast<std::size_t>(k)].hi = static_cast<double>(k + 1) / nbins;
  }
  const std::size_t n = std::min(p.size(), y.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double pi = detail::ecs_clamp01(p[i]);
    int k = static_cast<int>(pi * nbins);
    if (k >= nbins) k = nbins - 1;
    ReliabilityBin& b = bins[static_cast<std::size_t>(k)];
    ++b.count;
    b.mean_conf += pi;
    b.accuracy += y[i] ? 1.0 : 0.0;
  }
  for (ReliabilityBin& b : bins) {
    if (b.count > 0) {
      b.mean_conf /= static_cast<double>(b.count);
      b.accuracy /= static_cast<double>(b.count);
    }
  }
  return bins;
}

/// Expected calibration error: sum_b (n_b / N) |acc_b - conf_b|.
inline double expected_calibration_error(const std::vector<double>& p, const std::vector<int>& y,
                                         int nbins = 10) {
  const std::size_t n = std::min(p.size(), y.size());
  if (n == 0) return 0.0;
  double ece = 0.0;
  for (const ReliabilityBin& b : reliability_bins(p, y, nbins)) {
    if (b.count == 0) continue;
    ece += (static_cast<double>(b.count) / static_cast<double>(n)) * std::fabs(b.accuracy - b.mean_conf);
  }
  return ece;
}

/// Pool-adjacent-violators: the non-decreasing least-squares fit to y (optional weights).
inline std::vector<double> isotonic_pav(const std::vector<double>& y,
                                        const std::vector<double>& w = {}) {
  struct Block {
    double sum_w;
    double sum_wy;
    std::size_t len;
  };
  std::vector<Block> st;
  st.reserve(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double wi = (i < w.size()) ? w[i] : 1.0;
    st.push_back(Block{wi, wi * y[i], 1});
    while (st.size() >= 2) {
      Block& b = st[st.size() - 1];
      Block& a = st[st.size() - 2];
      // violation iff mean(a) > mean(b)  <=>  a.sum_wy * b.sum_w > b.sum_wy * a.sum_w
      if (a.sum_wy * b.sum_w > b.sum_wy * a.sum_w) {
        a.sum_w += b.sum_w;
        a.sum_wy += b.sum_wy;
        a.len += b.len;
        st.pop_back();
      } else {
        break;
      }
    }
  }
  std::vector<double> out;
  out.reserve(y.size());
  for (const Block& b : st) {
    const double mean = b.sum_w > 0.0 ? b.sum_wy / b.sum_w : 0.0;
    for (std::size_t k = 0; k < b.len; ++k) out.push_back(mean);
  }
  return out;
}

/// Monotone map from raw P to calibrated P: PAV over (p, outcome) with tied p
/// merged, then piecewise-linear interpolation between the fitted points.
class IsotonicCalibrator {
 public:
  void fit(const std::vector<double>& p, const std::vector<int>& y) {
    xs_.clear();
    ys_.clear();
    const std::size_t n = std::min(p.size(), y.size());
    if (n == 0) return;
    std::vector<std::pair<double, int>> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = {p[i], y[i] ? 1 : 0};
    std::sort(v.begin(), v.end());
    std::vector<double> gx;
    std::vector<double> gy;
    std::vector<double> gw;
    for (std::size_t i = 0; i < n;) {
      std::size_t j = i;
      double sy = 0.0;
      while (j < n && v[j].first == v[i].first) sy += v[j++].second;
      gx.push_back(v[i].first);
      gw.push_back(static_cast<double>(j - i));
      gy.push_back(sy / static_cast<double>(j - i));
      i = j;
    }
    ys_ = isotonic_pav(gy, gw);
    xs_ = std::move(gx);
  }

  bool fitted() const { return !xs_.empty(); }

  double predict(double p) const {
    if (xs_.empty()) return p;
    if (p <= xs_.front()) return ys_.front();
    if (p >= xs_.back()) return ys_.back();
    const auto it = std::upper_bound(xs_.begin(), xs_.end(), p);
    const std::size_t hi = static_cast<std::size_t>(it - xs_.begin());
    const std::size_t lo = hi - 1;
    const double t = (p - xs_[lo]) / (xs_[hi] - xs_[lo]);
    return ys_[lo] + t * (ys_[hi] - ys_[lo]);
  }

 private:
  std::vector<double> xs_;
  std::vector<double> ys_;
};

// ---------------------------------------------------------------------------
// 7.6 What ECS controls: admission rules
// ---------------------------------------------------------------------------

enum class Admission { Admit, Candidate, Reject };

/// Admit: P >= 0.8 and grade >= Moderate. Candidate: 0.5 <= P < 0.8, plus
/// P >= 0.8 with a grade below Moderate (keeps collecting evidence). Reject: P < 0.5.
inline Admission admission_decision(const Opinion& o) {
  if (o.P >= 0.8 && grade_at_least(o.grade, Grade::Moderate)) return Admission::Admit;
  if (o.P >= 0.5) return Admission::Candidate;
  return Admission::Reject;
}

/// Use as an inference premise: Grade High only.
inline bool usable_as_premise(const Opinion& o) { return o.grade == Grade::High; }

/// Use in an answer: conf(answer) at or above the dev-set threshold.
inline bool use_in_answer(double conf, double dev_threshold) { return conf >= dev_threshold; }

/// Pattern admission: Wilson lower bound of precision >= 0.7 over at least 20 checks.
inline bool pattern_admissible(std::size_t correct, std::size_t checks, double min_lower = 0.7,
                               std::size_t min_checks = 20) {
  if (checks < min_checks) return false;
  return wilson_lower(static_cast<double>(correct), static_cast<double>(checks)) >= min_lower;
}

struct AuditOutcome {
  double precision = 0.0;
  double wilson_lower_bound = 0.0;
  bool pause_admission = false;  // lower bound below 0.90: pause, roll back, update posteriors
};

/// Weekly admission audit (PLAN 7.5): Wilson lower bound of precision < 0.90 pauses admission.
inline AuditOutcome weekly_audit(std::size_t correct, std::size_t audited, double min_lower = 0.90) {
  AuditOutcome a;
  a.precision = audited ? static_cast<double>(correct) / static_cast<double>(audited) : 0.0;
  a.wilson_lower_bound = wilson_lower(static_cast<double>(correct), static_cast<double>(audited));
  a.pause_admission = a.wilson_lower_bound < min_lower;
  return a;
}

}  // namespace cypha::eval
