// Standard learners for Cypha's thought processes (docs/THOUGHT.md section 3).
//
// Header-only, standard library plus ../eval/ecs.hpp (Wilson bounds, isotonic PAV).
//   Beta posterior with fractional updates           3.3, 3.4
//   Dirichlet over a branch set                      3.2
//   Welford mean/variance, and a log-space variant   3.3 (costs)
//   Thompson selection over processes, U = p*G - C   3.4, 3.5 (with the epsilon floor)
//   ACT-R incremental utility  U <- U + a (R - U)    3.4
//   TD(lambda) eligibility credit over a trace       3.6
//   VOC stop rule from step gain and cost Welfords   3.7
//   Calibration log: Brier and ECE per process       3.8
//   Wilson bounds                                    (re-exported from eval/ecs.hpp)
//
// Randomness: everything draws from a seeded std::mt19937_64 and from the helpers below
// (uniform01, normal, gamma_sample). std::uniform_real_distribution, std::normal_distribution
// and std::gamma_distribution are implementation-defined, so they are not used: with the same
// seed the streams match across standard libraries. Gamma(k,1) is Marsaglia-Tsang (2000).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "../eval/ecs.hpp"

namespace cypha::thought {

using Rng = std::mt19937_64;

// ---------------------------------------------------------------------------
// Portable random draws
// ---------------------------------------------------------------------------

/// Uniform on the open interval (0, 1) from the top 53 bits of one 64-bit draw.
inline double uniform01(Rng& rng) {
  return (static_cast<double>(rng() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

/// Standard normal by the Marsaglia polar method (stateless: the spare variate is dropped).
inline double normal01(Rng& rng) {
  for (;;) {
    const double u = 2.0 * uniform01(rng) - 1.0;
    const double v = 2.0 * uniform01(rng) - 1.0;
    const double s = u * u + v * v;
    if (s > 0.0 && s < 1.0) return u * std::sqrt(-2.0 * std::log(s) / s);
  }
}

/// Gamma(k, 1) by Marsaglia-Tsang. For k < 1, Gamma(k) = Gamma(k+1) * U^(1/k).
inline double gamma_sample(Rng& rng, double k) {
  if (!(k > 0.0)) return 0.0;
  if (k < 1.0) {
    const double g = gamma_sample(rng, k + 1.0);
    return g * std::pow(uniform01(rng), 1.0 / k);
  }
  const double d = k - 1.0 / 3.0;
  const double c = 1.0 / std::sqrt(9.0 * d);
  for (;;) {
    double x, v;
    do {
      x = normal01(rng);
      v = 1.0 + c * x;
    } while (v <= 0.0);
    v = v * v * v;
    const double u = uniform01(rng);
    const double x2 = x * x;
    if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
    if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
  }
}

// ---------------------------------------------------------------------------
// Wilson bounds (eval/ecs.hpp), re-exported under this namespace
// ---------------------------------------------------------------------------

using eval::wilson_lower;
using eval::wilson_upper;

struct WilsonInterval {
  double lo = 0.0;
  double hi = 1.0;
};

inline WilsonInterval wilson_interval(double successes, double n, double z = 1.959963984540054) {
  return {wilson_lower(successes, n, z), wilson_upper(successes, n, z)};
}

// ---------------------------------------------------------------------------
// Beta posterior with fractional updates (3.3, 3.4, 3.6)
// ---------------------------------------------------------------------------

struct Beta {
  double a = 1.0;  // prior Beta(1, 1)
  double b = 1.0;

  double mean() const { return a / (a + b); }
  double variance() const {
    const double s = a + b;
    return a * b / (s * s * (s + 1.0));
  }
  /// Evidence mass above the prior.
  double n() const { return std::max(0.0, a + b - 2.0); }

  /// Fractional update with credit w >= 0 and reward r in [0, 1]:
  /// a += w*r, b += w*(1-r)   (3.6: a_k <- a_k + e_k R, b_k <- b_k + e_k (1 - R)).
  void update(double r, double w = 1.0) {
    r = std::min(1.0, std::max(0.0, r));
    a += w * r;
    b += w * (1.0 - r);
  }

  double sample(Rng& rng) const {
    const double x = gamma_sample(rng, a);
    const double y = gamma_sample(rng, b);
    const double s = x + y;
    return s > 0.0 ? x / s : mean();
  }

  /// Wilson bounds treating the evidence above the prior as n trials with a - 1 successes.
  WilsonInterval wilson(double z = 1.959963984540054) const {
    return wilson_interval(std::max(0.0, a - 1.0), n(), z);
  }
};

// ---------------------------------------------------------------------------
// Dirichlet over a branch set (3.2)
// ---------------------------------------------------------------------------

struct Dirichlet {
  std::vector<double> alpha;  // prior alpha_j = 1

  Dirichlet() = default;
  explicit Dirichlet(std::size_t k) : alpha(k, 1.0) {}

  std::size_t size() const { return alpha.size(); }
  double total() const {
    double s = 0.0;
    for (double v : alpha) s += v;
    return s;
  }
  double mean(std::size_t j) const { return alpha[j] / total(); }

  /// alpha_j <- alpha_j + w   (3.2 passes w = e_k * R for the branch j taken).
  void update(std::size_t j, double w) {
    if (j < alpha.size() && w > 0.0) alpha[j] += w;
  }

  /// theta ~ Dir(alpha), by normalised Gamma(alpha_j, 1) draws.
  std::vector<double> sample(Rng& rng) const {
    std::vector<double> g(alpha.size());
    double s = 0.0;
    for (std::size_t j = 0; j < alpha.size(); ++j) {
      g[j] = gamma_sample(rng, alpha[j]);
      s += g[j];
    }
    if (!(s > 0.0)) {  // underflow with tiny alphas: fall back to the mean
      const double t = total();
      for (std::size_t j = 0; j < alpha.size(); ++j) g[j] = alpha[j] / t;
      return g;
    }
    for (double& v : g) v /= s;
    return g;
  }

  /// Thompson choice: argmax of a Dirichlet draw (ties to the lowest index).
  std::size_t choose(Rng& rng) const {
    const std::vector<double> th = sample(rng);
    return static_cast<std::size_t>(std::max_element(th.begin(), th.end()) - th.begin());
  }
};

// ---------------------------------------------------------------------------
// Welford running moments (3.3) and the log-space variant for costs
// ---------------------------------------------------------------------------

struct Welford {
  std::size_t count = 0;
  double mu = 0.0;
  double m2 = 0.0;

  void update(double x) {
    ++count;
    const double d = x - mu;
    mu += d / static_cast<double>(count);
    m2 += d * (x - mu);
  }
  double mean() const { return mu; }
  /// Sample variance (n - 1); 0 below two observations.
  double variance() const { return count > 1 ? m2 / static_cast<double>(count - 1) : 0.0; }
  double stddev() const { return std::sqrt(variance()); }
  double stderr_of_mean() const { return count > 0 ? stddev() / std::sqrt(static_cast<double>(count)) : 0.0; }
};

/// Welford of ln(cost): costs are positive and skewed. Costs below kMinCost are clamped so that
/// a free step (cost 0) is a very small cost rather than ln 0.
struct LogWelford {
  static constexpr double kMinCost = 1e-6;
  Welford w;

  void update(double cost) { w.update(std::log(std::max(cost, kMinCost))); }
  std::size_t count() const { return w.count; }
  double log_mean() const { return w.mean(); }
  double log_variance() const { return w.variance(); }
  /// Geometric mean exp(E[ln c]).
  double geometric_mean() const { return std::exp(w.mean()); }
  /// Lognormal estimate of E[c] = exp(mu + s^2/2). `prior` when there is no data.
  double expected(double prior = 0.0) const {
    if (w.count == 0) return prior;
    return std::exp(w.mean() + 0.5 * w.variance());
  }
  /// Draw C~ from the cost posterior: mu* ~ N(mean, s^2/n), C~ = exp(mu* + s^2/2).
  /// One observation gives its value; none gives `prior`.
  double sample(Rng& rng, double prior = 0.0) const {
    if (w.count == 0) return prior;
    if (w.count == 1) return std::exp(w.mean());
    const double mu = w.mean() + w.stderr_of_mean() * normal01(rng);
    return std::exp(mu + 0.5 * w.variance());
  }
};

// ---------------------------------------------------------------------------
// Utility, Thompson selection over processes (3.4, 3.5)
// ---------------------------------------------------------------------------

/// U = p * G - C, with G the value of the goal.
inline double utility(double p, double G, double C) { return p * G - C; }

/// ACT-R utility learning (3.4): U <- U + alpha (R - U). A fast-adapting second estimate.
struct ActRUtility {
  double u = 0.0;
  double alpha = 0.1;
  std::size_t count = 0;
  void update(double r) {
    u += alpha * (r - u);
    ++count;
  }
};

struct ThompsonArm {
  const Beta* success = nullptr;
  const LogWelford* cost = nullptr;
};

struct ThompsonChoice {
  std::size_t index = 0;
  bool floor_pick = false;          // chosen by the epsilon floor, not by the draw
  std::vector<double> drawn;        // U~ for each arm (empty on a floor pick)
};

/// Thompson selection (3.5). With probability min(1, n*eps) the choice is uniform over the n arms,
/// so every arm keeps at least eps of the choices. Otherwise draw p~ ~ Beta, C~ ~ cost posterior,
/// U~ = p~ G - C~, and take the argmax (ties to the lowest index).
inline ThompsonChoice thompson_select(const std::vector<ThompsonArm>& arms, double G, double eps,
                                      Rng& rng, double prior_cost = 0.0) {
  ThompsonChoice out;
  const std::size_t n = arms.size();
  if (n == 0) return out;
  const double u = uniform01(rng);
  const double pf = std::min(1.0, static_cast<double>(n) * eps);
  if (u < pf) {
    out.index = std::min(n - 1, static_cast<std::size_t>(u / eps));
    out.floor_pick = true;
    return out;
  }
  out.drawn.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double p = arms[i].success ? arms[i].success->sample(rng) : uniform01(rng);
    const double c = arms[i].cost ? arms[i].cost->sample(rng, prior_cost) : prior_cost;
    out.drawn[i] = utility(p, G, c);
  }
  out.index = static_cast<std::size_t>(std::max_element(out.drawn.begin(), out.drawn.end()) -
                                       out.drawn.begin());
  return out;
}

// ---------------------------------------------------------------------------
// TD(lambda) credit over a trace (3.6)
// ---------------------------------------------------------------------------

/// Eligibility e_k = lambda^(T - k) for k = 1..T. The last step gets 1. Index 0 is step k = 1.
inline std::vector<double> td_credits(std::size_t T, double lambda = 0.8) {
  std::vector<double> e(T);
  double c = 1.0;
  for (std::size_t i = T; i-- > 0;) {
    e[i] = c;
    c *= lambda;
  }
  return e;
}

// ---------------------------------------------------------------------------
// VOC stop rule (3.7)
// ---------------------------------------------------------------------------

/// VOC = G * E[gain of the next step] - E[cost of the next step] (Russell & Wefald 1991).
inline double voc(const Welford& gain, const LogWelford& cost, double value_per_gain = 1.0,
                  double prior_cost = 0.0) {
  return value_per_gain * gain.mean() - cost.expected(prior_cost);
}

/// Stop when VOC <= 0. With fewer than min_n observations of either, there is no basis to stop.
inline bool voc_stop(const Welford& gain, const LogWelford& cost, double value_per_gain = 1.0,
                     std::size_t min_n = 5) {
  if (gain.count < min_n || cost.count() < min_n) return false;
  return voc(gain, cost, value_per_gain) <= 0.0;
}

// ---------------------------------------------------------------------------
// Calibration log: Brier and ECE per process (3.8)
// ---------------------------------------------------------------------------

class CalibrationLog {
 public:
  static constexpr int kBins = 10;

  struct Stat {
    std::array<double, kBins> n{};   // count per equal-width bin of p-hat
    std::array<double, kBins> sp{};  // sum of p-hat
    std::array<double, kBins> sy{};  // sum of outcomes
    double brier = 0.0;              // sum of (p-hat - R)^2
    double count = 0.0;

    void add(double p, double r) {
      p = std::min(1.0, std::max(0.0, p));
      r = std::min(1.0, std::max(0.0, r));
      int k = static_cast<int>(p * kBins);
      if (k >= kBins) k = kBins - 1;
      const std::size_t i = static_cast<std::size_t>(k);
      n[i] += 1.0;
      sp[i] += p;
      sy[i] += r;
      brier += (p - r) * (p - r);
      count += 1.0;
    }
    double brier_score() const { return count > 0.0 ? brier / count : 0.0; }
    /// sum_b (n_b / N) |acc_b - conf_b|; matches eval::expected_calibration_error for 0/1 outcomes.
    double ece() const {
      if (count <= 0.0) return 0.0;
      double e = 0.0;
      for (std::size_t i = 0; i < static_cast<std::size_t>(kBins); ++i) {
        if (n[i] > 0.0) e += (n[i] / count) * std::fabs(sy[i] / n[i] - sp[i] / n[i]);
      }
      return e;
    }
  };

  /// Log a prediction p-hat made before a run and its reward R after it.
  void add(const std::string& process, double p_hat, double reward) {
    per_[process].add(p_hat, reward);
    all_.add(p_hat, reward);
  }

  std::size_t count(const std::string& process) const {
    const auto it = per_.find(process);
    return it == per_.end() ? 0 : static_cast<std::size_t>(it->second.count);
  }
  double brier(const std::string& process) const {
    const auto it = per_.find(process);
    return it == per_.end() ? 0.0 : it->second.brier_score();
  }
  double ece(const std::string& process) const {
    const auto it = per_.find(process);
    return it == per_.end() ? 0.0 : it->second.ece();
  }
  double brier_all() const { return all_.brier_score(); }
  double ece_all() const { return all_.ece(); }
  std::size_t count_all() const { return static_cast<std::size_t>(all_.count); }
  const std::map<std::string, Stat>& per_process() const { return per_; }

  /// Isotonic correction of a raw p-hat from this process's reliability bins (PAV over bin
  /// accuracy, weighted by bin count, piecewise-linear between bin centres of mass).
  /// Returns p unchanged with fewer than two populated bins.
  double corrected(const std::string& process, double p) const {
    const auto it = per_.find(process);
    if (it == per_.end()) return p;
    std::vector<double> xs, ys, ws;
    for (std::size_t i = 0; i < static_cast<std::size_t>(kBins); ++i) {
      if (it->second.n[i] <= 0.0) continue;
      xs.push_back(it->second.sp[i] / it->second.n[i]);
      ys.push_back(it->second.sy[i] / it->second.n[i]);
      ws.push_back(it->second.n[i]);
    }
    if (xs.size() < 2) return p;
    const std::vector<double> fit = eval::isotonic_pav(ys, ws);
    if (p <= xs.front()) return fit.front();
    if (p >= xs.back()) return fit.back();
    const std::size_t hi = static_cast<std::size_t>(std::upper_bound(xs.begin(), xs.end(), p) - xs.begin());
    const std::size_t lo = hi - 1;
    const double t = (p - xs[lo]) / (xs[hi] - xs[lo]);
    return fit[lo] + t * (fit[hi] - fit[lo]);
  }

 private:
  std::map<std::string, Stat> per_;
  Stat all_;
};

}  // namespace cypha::thought
