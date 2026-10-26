// LCI: Learning Certainty Index (docs/PLAN.md section 8).
//
// Header-only, standard library only. Implements:
//   - Bayesian bootstrap of paired per-item differences (optionally cluster-level)
//   - canary non-inferiority, generalisation gamma, calibration c, stability
//     sigma, cost kappa, and LCI = P_gain * r * gamma * c * sigma * kappa (8.2)
//   - sequential e-value (betting e-process) + geometric alpha spending and
//     Benjamini-Hochberg (8.4)
//   - Thresholdout (Dwork et al. 2015) (8.5)
//   - backward transfer and learning-curve fit (8.6)
//
// Sign convention: every "difference" is (after - before) oriented so that
// larger is better. For lower-is-better metrics (BPC) the caller negates.
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cypha::eval {

namespace detail {

/// Portable uniform in the open interval (0, 1) from raw mt19937_64 output.
/// (std::*_distribution output differs between standard libraries; raw engine
/// output does not, so results are reproducible across toolchains.)
inline double lci_uniform_open(std::mt19937_64& eng) {
  return (static_cast<double>(eng() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

/// Gamma(1, 1) draw, which is exactly Exp(1): -ln U.
inline double lci_gamma1(std::mt19937_64& eng) { return -std::log(lci_uniform_open(eng)); }

/// Laplace(0, scale) draw by inverse CDF.
inline double lci_laplace(std::mt19937_64& eng, double scale) {
  const double u = lci_uniform_open(eng) - 0.5;  // (-0.5, 0.5)
  const double s = (u < 0.0) ? -1.0 : 1.0;
  return -scale * s * std::log1p(-2.0 * std::fabs(u));
}

inline double lci_clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

}  // namespace detail

// ---------------------------------------------------------------------------
// 8.2 P_gain: Bayesian bootstrap
// ---------------------------------------------------------------------------

struct BootstrapOptions {
  std::size_t draws = 4000;
  std::uint64_t seed = 0x43595048414C4349ULL;  // "CYPHALCI"
};

struct BootstrapResult {
  double p_gt = 0.0;   // P(weighted mean difference > threshold)
  double mean = 0.0;   // mean of the draws
  double lo90 = 0.0;   // 5% quantile of the draws
  double hi90 = 0.0;   // 95% quantile of the draws
  std::size_t n_items = 0;
  std::size_t n_eff = 0;  // number of independent units (clusters, or items if no clusters)
  std::size_t draws = 0;
};

/// Bayesian bootstrap (Rubin 1981) of the mean of `diffs`: each draw gives the
/// units Dirichlet(1, ..., 1) weights (normalised Gamma(1,1) draws from a seeded
/// std::mt19937_64) and records the weighted mean. Returns P(mean > threshold).
///
/// If `cluster_ids` is non-empty (one id per item) resampling is at cluster
/// level: each cluster is one unit whose value is the mean of its items, the
/// Dirichlet weights are over clusters, and n_eff = number of clusters.
inline BootstrapResult bayesian_bootstrap(const std::vector<double>& diffs, double threshold,
                                          const std::vector<int>& cluster_ids = {},
                                          const BootstrapOptions& opt = {}) {
  BootstrapResult res;
  res.n_items = diffs.size();
  res.draws = opt.draws;
  std::vector<double> unit;
  if (cluster_ids.empty()) {
    unit = diffs;
  } else {
    if (cluster_ids.size() != diffs.size())
      throw std::invalid_argument("bayesian_bootstrap: cluster_ids size != diffs size");
    std::unordered_map<int, std::size_t> slot;
    std::vector<double> sum;
    std::vector<std::size_t> cnt;
    for (std::size_t i = 0; i < diffs.size(); ++i) {
      auto it = slot.find(cluster_ids[i]);
      if (it == slot.end()) {
        it = slot.emplace(cluster_ids[i], sum.size()).first;
        sum.push_back(0.0);
        cnt.push_back(0);
      }
      sum[it->second] += diffs[i];
      ++cnt[it->second];
    }
    unit.resize(sum.size());
    for (std::size_t k = 0; k < sum.size(); ++k) unit[k] = sum[k] / static_cast<double>(cnt[k]);
  }
  res.n_eff = unit.size();
  if (unit.empty() || opt.draws == 0) return res;

  std::mt19937_64 eng(opt.seed);
  std::vector<double> stats(opt.draws);
  std::size_t above = 0;
  double acc = 0.0;
  for (std::size_t d = 0; d < opt.draws; ++d) {
    double sw = 0.0;
    double swv = 0.0;
    for (double v : unit) {
      const double w = detail::lci_gamma1(eng);
      sw += w;
      swv += w * v;
    }
    const double m = swv / sw;
    stats[d] = m;
    acc += m;
    if (m > threshold) ++above;
  }
  res.p_gt = static_cast<double>(above) / static_cast<double>(opt.draws);
  res.mean = acc / static_cast<double>(opt.draws);
  std::sort(stats.begin(), stats.end());
  res.lo90 = stats[static_cast<std::size_t>(0.05 * static_cast<double>(opt.draws - 1))];
  res.hi90 = stats[static_cast<std::size_t>(0.95 * static_cast<double>(opt.draws - 1))];
  return res;
}

constexpr std::size_t kMinEffectiveN = 200;
constexpr double kLowNEffCap = 0.5;
constexpr double kPromoteThreshold = 0.95;

/// 0.5 if n_eff < min_n_eff (200), else 1: the cap applied to LCI.
inline double n_eff_cap(std::size_t n_eff, std::size_t min_n_eff = kMinEffectiveN) {
  return n_eff < min_n_eff ? kLowNEffCap : 1.0;
}

// ---------------------------------------------------------------------------
// 8.2 Canaries, generalisation, calibration, stability, cost
// ---------------------------------------------------------------------------

struct CanaryInput {
  std::string name;
  std::vector<double> diffs;     // paired per-item differences (after - before)
  std::vector<int> clusters;     // optional cluster ids
  double delta = 0.01;           // delta_k: tolerated drop, same units as diffs (1 point = 0.01)
};

struct CanaryResult {
  std::string name;
  double p_noninferior = 0.0;  // P(Delta_k > -delta_k)
  double delta = 0.0;
  bool pass = false;           // p_noninferior >= required (0.95)
  std::size_t n_eff = 0;
};

/// Non-inferiority on one canary: P(Delta_k > -delta_k) >= required (0.95).
/// A canary with no data cannot show non-inferiority and fails.
inline CanaryResult canary_noninferiority(const CanaryInput& c, double required = 0.95,
                                          const BootstrapOptions& opt = {}) {
  const BootstrapResult b = bayesian_bootstrap(c.diffs, -c.delta, c.clusters, opt);
  CanaryResult r;
  r.name = c.name;
  r.delta = c.delta;
  r.p_noninferior = b.p_gt;
  r.n_eff = b.n_eff;
  r.pass = (b.n_eff > 0) && (b.p_gt >= required);
  return r;
}

/// r = 1 if every canary passes (vacuously true with none), else 0.
inline double canary_r(const std::vector<CanaryResult>& cs) {
  for (const CanaryResult& c : cs)
    if (!c.pass) return 0.0;
  return 1.0;
}

/// gamma = min(1, G / half_target) with G = clip(gain_X / gain_D, 0, 1).
/// If gain_D <= 0 there is no dev gain to compare against and G = 0.
inline double generalisation_gamma(double gain_ext, double gain_dev, double half_target = 0.5) {
  double G = 0.0;
  if (gain_dev > 0.0) G = detail::lci_clamp(gain_ext / gain_dev, 0.0, 1.0);
  return std::min(1.0, G / half_target);
}

/// c = 1 if ECE rises by no more than 0.01 and Brier does not rise, else 0.5.
inline double calibration_factor(double ece_before, double ece_after, double brier_before,
                                 double brier_after, double max_ece_rise = 0.01) {
  const bool ok = (ece_after - ece_before <= max_ece_rise + 1e-12) && (brier_after <= brier_before + 1e-12);
  return ok ? 1.0 : 0.5;
}

/// sigma = 1 if the gain has the same strict sign in every one of at least
/// `min_seeds` (3) seeds / data orders, else 0.5.
inline double stability_sigma(const std::vector<double>& seed_gains, std::size_t min_seeds = 3) {
  if (seed_gains.size() < min_seeds) return 0.5;
  bool all_pos = true;
  bool all_neg = true;
  for (double g : seed_gains) {
    all_pos = all_pos && (g > 0.0);
    all_neg = all_neg && (g < 0.0);
  }
  return (all_pos || all_neg) ? 1.0 : 0.5;
}

/// kappa = 1 if RAM and latency are within budget, else 0.
inline double cost_kappa(double ram, double ram_budget, double latency, double latency_budget) {
  return (ram <= ram_budget && latency <= latency_budget) ? 1.0 : 0.0;
}

struct LciInputs {
  // P_gain on the referee T
  std::vector<double> test_diffs;
  std::vector<int> test_clusters;  // optional; empty = every item independent
  double eps = 0.0;                // smallest gain that matters (0.005 acc., etc.)
  // canaries C
  std::vector<CanaryInput> canaries;
  // generalisation: mean gains on D and X
  double gain_dev = 0.0;
  double gain_ext = 0.0;
  // calibration
  double ece_before = 0.0, ece_after = 0.0;
  double brier_before = 0.0, brier_after = 0.0;
  // stability: gain from each seed / data order
  std::vector<double> seed_gains;
  // cost
  double ram = 0.0, ram_budget = std::numeric_limits<double>::infinity();
  double latency = 0.0, latency_budget = std::numeric_limits<double>::infinity();
  // bootstrap settings
  std::size_t draws = 4000;
  std::uint64_t seed = BootstrapOptions{}.seed;
};

struct LciDecision {
  double p_gain = 0.0;
  double mean_gain = 0.0;
  double gain_lo90 = 0.0;
  double gain_hi90 = 0.0;
  std::size_t n_items = 0;
  std::size_t n_eff = 0;
  bool n_eff_capped = false;  // n_eff < 200: LCI capped at 0.5
  std::vector<CanaryResult> canaries;
  double r = 1.0;
  double gamma = 0.0;
  double calibration = 1.0;
  double sigma = 0.5;
  double kappa = 1.0;
  double product = 0.0;  // P_gain * r * gamma * c * sigma * kappa, before the n_eff cap
  double lci = 0.0;      // after the n_eff cap
  bool meets_threshold = false;  // lci >= 0.95 (the multiplicity check is separate)

  /// Every component as (name, value), for logging in the experiment DB.
  std::vector<std::pair<std::string, double>> components() const {
    std::vector<std::pair<std::string, double>> v = {
        {"P_gain", p_gain}, {"n_eff", static_cast<double>(n_eff)}, {"r", r},
        {"gamma", gamma},   {"c", calibration},                    {"sigma", sigma},
        {"kappa", kappa},   {"product", product},                  {"LCI", lci}};
    for (const CanaryResult& c : canaries) v.emplace_back("canary_p:" + c.name, c.p_noninferior);
    return v;
  }
};

/// LCI = P_gain x r x gamma x c x sigma x kappa, capped at 0.5 when n_eff < 200.
inline LciDecision compute_lci(const LciInputs& in, double promote_threshold = kPromoteThreshold) {
  LciDecision d;
  BootstrapOptions bo;
  bo.draws = in.draws;
  bo.seed = in.seed;
  const BootstrapResult g = bayesian_bootstrap(in.test_diffs, in.eps, in.test_clusters, bo);
  d.p_gain = g.p_gt;
  d.mean_gain = g.mean;
  d.gain_lo90 = g.lo90;
  d.gain_hi90 = g.hi90;
  d.n_items = g.n_items;
  d.n_eff = g.n_eff;

  std::uint64_t k = 1;
  for (const CanaryInput& c : in.canaries) {
    BootstrapOptions co = bo;
    co.seed = in.seed + 0x9E3779B97F4A7C15ULL * k++;  // independent stream per canary
    d.canaries.push_back(canary_noninferiority(c, 0.95, co));
  }
  d.r = canary_r(d.canaries);
  d.gamma = generalisation_gamma(in.gain_ext, in.gain_dev);
  d.calibration = calibration_factor(in.ece_before, in.ece_after, in.brier_before, in.brier_after);
  d.sigma = stability_sigma(in.seed_gains);
  d.kappa = cost_kappa(in.ram, in.ram_budget, in.latency, in.latency_budget);
  d.product = d.p_gain * d.r * d.gamma * d.calibration * d.sigma * d.kappa;
  d.n_eff_capped = d.n_eff < kMinEffectiveN;
  d.lci = d.n_eff_capped ? std::min(d.product, kLowNEffCap) : d.product;
  d.meets_threshold = d.lci >= promote_threshold;
  return d;
}

/// Promote only if LCI >= 0.95 and the multiplicity check (8.4) passed.
inline bool promote(const LciDecision& d, bool multiplicity_ok) {
  return d.meets_threshold && multiplicity_ok;
}

// ---------------------------------------------------------------------------
// 8.4 Multiplicity: alpha spending, e-process, Benjamini-Hochberg
// ---------------------------------------------------------------------------

/// alpha_i = alpha_total * (1 - rho) * rho^(i-1), i = 1, 2, ... (sums to alpha_total).
inline double alpha_spend(std::size_t i, double alpha_total = 0.05, double rho = 0.9) {
  if (i < 1) i = 1;
  return alpha_total * (1.0 - rho) * std::pow(rho, static_cast<double>(i - 1));
}

/// Weekly alpha budget. Every candidate (rejected ones too) takes the next alpha_i.
class AlphaSpender {
 public:
  explicit AlphaSpender(double alpha_total = 0.05, double rho = 0.9)
      : alpha_total_(alpha_total), rho_(rho) {}
  /// alpha for the next candidate (i = count()+1); advances the counter.
  double next() { return alpha_spend(++count_, alpha_total_, rho_); }
  std::size_t count() const { return count_; }
  double spent() const { return alpha_total_ * (1.0 - std::pow(rho_, static_cast<double>(count_))); }
  double remaining() const { return alpha_total_ - spent(); }
  void reset() { count_ = 0; }  // new weekly budget

 private:
  double alpha_total_;
  double rho_;
  std::size_t count_ = 0;
};

/// Anytime-valid e-process for a bounded stream of paired differences d_i in
/// [-B, B] (one-sided null H0: E[d_i | past] <= null_mean).
///
/// Choice: a mixture of fixed-lambda betting martingales. Map d to
/// x = (d + B) / (2B) in [0,1] and m = (null_mean + B) / (2B). For each lambda_j
/// in [0, 1/m) the capital K_j = prod (1 + lambda_j (x_i - m)) is a nonnegative
/// supermartingale under H0 with K_j(0) = 1, so the equal-weight average over j
/// is an e-process (Waudby-Smith & Ramdas 2024; Ramdas et al. 2023). By Ville's
/// inequality P(sup_t E_t >= 1/alpha) <= alpha, so a candidate may be promoted
/// the first time E_t >= 1/alpha_i. The default grid is
/// lambda_j = 0.9 / m * 2^-j, j = 0..7, so no lambda has to be tuned in advance.
///
/// Feed items in random order, or feed one cluster-mean per near-duplicate
/// cluster, so the conditional-mean assumption is plausible.
class EProcess {
 public:
  explicit EProcess(double bound = 1.0, double null_mean = 0.0, std::vector<double> lambdas = {})
      : bound_(bound), null_mean_(null_mean) {
    if (!(bound > 0.0)) throw std::invalid_argument("EProcess: bound must be > 0");
    if (!(null_mean > -bound && null_mean < bound))
      throw std::invalid_argument("EProcess: null_mean must lie strictly inside (-bound, bound)");
    m_ = (null_mean + bound) / (2.0 * bound);
    if (lambdas.empty()) {
      for (int j = 0; j < 8; ++j) lambdas.push_back(0.9 / m_ * std::pow(0.5, j));
    }
    for (double l : lambdas)
      if (!(l >= 0.0 && l < 1.0 / m_)) throw std::invalid_argument("EProcess: lambda outside [0, 1/m)");
    lambdas_ = std::move(lambdas);
    log_k_.assign(lambdas_.size(), 0.0);
  }

  /// Add one observation; throws std::invalid_argument if |d| > bound.
  void update(double d) {
    if (!(d >= -bound_ && d <= bound_)) throw std::invalid_argument("EProcess: observation outside [-B, B]");
    const double x = (d + bound_) / (2.0 * bound_);
    for (std::size_t j = 0; j < lambdas_.size(); ++j) log_k_[j] += std::log1p(lambdas_[j] * (x - m_));
    ++n_;
  }

  void update(const std::vector<double>& ds) {
    for (double d : ds) update(d);
  }

  /// log of the mixture e-value (log-sum-exp, overflow safe).
  double log_e() const {
    double mx = -std::numeric_limits<double>::infinity();
    for (double v : log_k_) mx = std::max(mx, v);
    double s = 0.0;
    for (double v : log_k_) s += std::exp(v - mx);
    return mx + std::log(s / static_cast<double>(log_k_.size()));
  }
  double e_value() const { return std::exp(log_e()); }
  std::size_t count() const { return n_; }

  /// True iff E_t >= 1/alpha.
  bool rejects_at(double alpha) const { return log_e() >= -std::log(alpha); }

 private:
  double bound_;
  double null_mean_;
  double m_ = 0.5;
  std::vector<double> lambdas_;
  std::vector<double> log_k_;
  std::size_t n_ = 0;
};

/// A valid p-value from an e-value (Markov): min(1, 1/e).
inline double e_to_p(double e) { return e > 1.0 ? 1.0 / e : 1.0; }

struct MultiplicityCheck {
  double alpha_i = 0.0;
  double e_value = 1.0;
  double needed = 0.0;  // 1 / alpha_i
  bool pass = false;
};

/// Check candidate number `i` (1-based, rejected candidates count) against its alpha_i.
inline MultiplicityCheck multiplicity_check(const EProcess& ep, std::size_t i, double alpha_total = 0.05,
                                            double rho = 0.9) {
  MultiplicityCheck c;
  c.alpha_i = alpha_spend(i, alpha_total, rho);
  c.needed = 1.0 / c.alpha_i;
  c.e_value = ep.e_value();
  c.pass = ep.log_e() >= std::log(c.needed);
  return c;
}

struct BhResult {
  std::vector<bool> reject;       // per input p-value, original order
  std::size_t n_rejected = 0;
  double p_threshold = 0.0;       // largest rejected p-value (0 if none)
};

/// Benjamini-Hochberg step-up procedure at level q: reject the k smallest
/// p-values where k is the largest index with p_(k) <= k q / m.
inline BhResult benjamini_hochberg(const std::vector<double>& p, double q = 0.05) {
  BhResult out;
  const std::size_t m = p.size();
  out.reject.assign(m, false);
  if (m == 0) return out;
  std::vector<std::size_t> idx(m);
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return p[a] < p[b]; });
  std::size_t k = 0;
  for (std::size_t i = 0; i < m; ++i) {
    if (p[idx[i]] <= static_cast<double>(i + 1) * q / static_cast<double>(m)) k = i + 1;
  }
  for (std::size_t i = 0; i < k; ++i) out.reject[idx[i]] = true;
  out.n_rejected = k;
  out.p_threshold = k ? p[idx[k - 1]] : 0.0;
  return out;
}

/// BH-adjusted p-values (q-values): reject where adjusted <= q.
inline std::vector<double> bh_adjusted(const std::vector<double>& p) {
  const std::size_t m = p.size();
  std::vector<double> adj(m, 1.0);
  if (m == 0) return adj;
  std::vector<std::size_t> idx(m);
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return p[a] < p[b]; });
  double run = 1.0;
  for (std::size_t i = m; i-- > 0;) {
    run = std::min(run, p[idx[i]] * static_cast<double>(m) / static_cast<double>(i + 1));
    adj[idx[i]] = std::min(1.0, run);
  }
  return adj;
}

// ---------------------------------------------------------------------------
// 8.5 Thresholdout (Dwork et al. 2015, Algorithm 1)
// ---------------------------------------------------------------------------

/// Answers queries about the referee T (here `test_value`) through the dev value:
/// returns the dev value unless |dev - test| > T_hat + eta, in which case it
/// returns test + Laplace(sigma) noise and spends one unit of budget. The noisy
/// threshold is T_hat = T + Lap(2 sigma), eta ~ Lap(4 sigma), refreshed whenever
/// the test value is released. When the budget is spent query() refuses; the
/// caller should rotate T (PLAN 8.5).
class Thresholdout {
 public:
  struct Answer {
    double value = std::numeric_limits<double>::quiet_NaN();
    bool used_test = false;         // true if test + noise was returned
    bool budget_exhausted = false;  // true if refused (value is NaN)
  };

  explicit Thresholdout(double threshold = 0.04, double sigma = 0.01, std::size_t budget = 100,
                        std::uint64_t seed = 0x544F5554ULL)
      : threshold_(threshold), sigma_(sigma), budget_(budget), eng_(seed) {
    noisy_threshold_ = threshold_ + detail::lci_laplace(eng_, 2.0 * sigma_);
  }

  Answer query(double dev_value, double test_value) {
    Answer a;
    ++queries_;
    if (budget_ < 1) {
      a.budget_exhausted = true;
      return a;
    }
    const double eta = detail::lci_laplace(eng_, 4.0 * sigma_);
    if (std::fabs(test_value - dev_value) > noisy_threshold_ + eta) {
      const double xi = detail::lci_laplace(eng_, sigma_);
      noisy_threshold_ = threshold_ + detail::lci_laplace(eng_, 2.0 * sigma_);
      --budget_;
      a.value = test_value + xi;
      a.used_test = true;
    } else {
      a.value = dev_value;
    }
    return a;
  }

  std::size_t budget_remaining() const { return budget_; }
  bool exhausted() const { return budget_ < 1; }
  std::size_t queries() const { return queries_; }

  /// New referee after rotation: fresh budget and noisy threshold.
  void rotate(std::size_t budget) {
    budget_ = budget;
    queries_ = 0;
    noisy_threshold_ = threshold_ + detail::lci_laplace(eng_, 2.0 * sigma_);
  }

 private:
  double threshold_;
  double sigma_;
  std::size_t budget_;
  std::size_t queries_ = 0;
  double noisy_threshold_ = 0.0;
  std::mt19937_64 eng_;
};

// ---------------------------------------------------------------------------
// 8.6 Backward transfer and learning curve
// ---------------------------------------------------------------------------

/// Backward transfer (Lopez-Paz & Ranzato 2017). R[i][j] = performance on task j
/// after learning stage i (R[i][i] is "just learned"; the last row is "final").
/// BWT = mean over k = 0..T-2 of (R[T-1][k] - R[k][k]). Negative = forgetting.
inline double backward_transfer(const std::vector<std::vector<double>>& R) {
  const std::size_t T = R.size();
  if (T < 2) return 0.0;
  double s = 0.0;
  for (std::size_t k = 0; k + 1 < T; ++k) s += R[T - 1][k] - R[k][k];
  return s / static_cast<double>(T - 1);
}

/// Same quantity from per-task vectors: mean_k (final_k - just_learned_k).
inline double backward_transfer(const std::vector<double>& final_perf,
                                const std::vector<double>& just_learned) {
  const std::size_t n = std::min(final_perf.size(), just_learned.size());
  if (n == 0) return 0.0;
  double s = 0.0;
  for (std::size_t k = 0; k < n; ++k) s += final_perf[k] - just_learned[k];
  return s / static_cast<double>(n);
}

struct LearningCurveFit {
  bool ok = false;
  double Linf = 0.0;
  double a = 0.0;
  double b = 0.0;
  double sse = 0.0;
  double rmse = 0.0;

  double predict(double n) const { return Linf + a * std::pow(n, -b); }

  /// Data size n at which the fitted curve reaches `target`
  /// (infinity if the target is unreachable).
  double n_for(double target) const {
    if (!ok || a == 0.0) return std::numeric_limits<double>::infinity();
    const double ratio = (target - Linf) / a;
    if (!(ratio > 0.0)) return std::numeric_limits<double>::infinity();
    return std::pow(ratio, -1.0 / b);
  }
};

namespace detail {

/// For fixed b: least squares of L on x = n^-b -> (Linf, a, sse). False if degenerate.
inline bool lci_curve_ls(const std::vector<double>& n, const std::vector<double>& L, double b,
                         double& Linf, double& a, double& sse) {
  const std::size_t N = n.size();
  double sx = 0.0, sy = 0.0;
  std::vector<double> x(N);
  for (std::size_t i = 0; i < N; ++i) {
    x[i] = std::pow(n[i], -b);
    sx += x[i];
    sy += L[i];
  }
  const double mx = sx / static_cast<double>(N);
  const double my = sy / static_cast<double>(N);
  double sxx = 0.0, sxy = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    sxx += (x[i] - mx) * (x[i] - mx);
    sxy += (x[i] - mx) * (L[i] - my);
  }
  if (!(sxx > 1e-300)) return false;
  a = sxy / sxx;
  Linf = my - a * mx;
  sse = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double e = L[i] - (Linf + a * x[i]);
    sse += e * e;
  }
  return true;
}

}  // namespace detail

/// Fit L(n) = Linf + a n^-b: grid over b (coarse step, then golden-section
/// refinement around the best cell); Linf and a by closed-form least squares.
/// Needs at least 3 points with positive, not all equal n.
inline LearningCurveFit fit_learning_curve(const std::vector<double>& n, const std::vector<double>& L,
                                           double b_min = 0.01, double b_max = 4.0,
                                           double b_step = 0.01) {
  LearningCurveFit fit;
  const std::size_t N = std::min(n.size(), L.size());
  if (N < 3) return fit;
  std::vector<double> nn(n.begin(), n.begin() + static_cast<std::ptrdiff_t>(N));
  std::vector<double> LL(L.begin(), L.begin() + static_cast<std::ptrdiff_t>(N));
  for (double v : nn)
    if (!(v > 0.0)) return fit;

  double best_b = 0.0, best_sse = std::numeric_limits<double>::infinity();
  bool any = false;
  for (double b = b_min; b <= b_max + 1e-12; b += b_step) {
    double Linf, a, sse;
    if (!detail::lci_curve_ls(nn, LL, b, Linf, a, sse)) continue;
    if (sse < best_sse) {
      best_sse = sse;
      best_b = b;
      any = true;
    }
  }
  if (!any) return fit;

  // Golden-section refinement on [best_b - step, best_b + step].
  double lo = std::max(b_min, best_b - b_step);
  double hi = std::min(b_max, best_b + b_step);
  const double phi = 0.6180339887498949;
  auto f = [&](double b) {
    double Linf, a, sse;
    return detail::lci_curve_ls(nn, LL, b, Linf, a, sse) ? sse : std::numeric_limits<double>::infinity();
  };
  double c = hi - phi * (hi - lo);
  double d = lo + phi * (hi - lo);
  double fc = f(c), fd = f(d);
  for (int it = 0; it < 80; ++it) {
    if (fc < fd) {
      hi = d;
      d = c;
      fd = fc;
      c = hi - phi * (hi - lo);
      fc = f(c);
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + phi * (hi - lo);
      fd = f(d);
    }
  }
  const double refined = 0.5 * (lo + hi);
  if (f(refined) <= best_sse) best_b = refined;

  double Linf, a, sse;
  if (!detail::lci_curve_ls(nn, LL, best_b, Linf, a, sse)) return fit;
  fit.ok = true;
  fit.Linf = Linf;
  fit.a = a;
  fit.b = best_b;
  fit.sse = sse;
  fit.rmse = std::sqrt(sse / static_cast<double>(N));
  return fit;
}

}  // namespace cypha::eval
