// S10 (docs/reports/S10_CURIOSITY_SCHEDULER.md): DISCOVERY §9's curiosity scheduler.
//
// An arm is a (generator, relation group, tier) of process P2 (conjecture). A pull sends the arm's next hypothesis
// through its tests; it costs the tests it ran and rewards impact x certainty gained when `decide` supports it.
// The scheduler: Thompson sampling over each arm's reward rate per unit cost (a Beta draw of its promotion probability,
// with an empirical-Bayes prior from the pool, times its mean reward per promotion), a learning-progress bonus (bits of
// uncertainty removed per unit cost, recent pulls), demand first within an arm, floors (each generator 2%, the wild
// tier 5% of the cost spent) and retirement (promotion rate's Wilson upper bound below 0.05 after 200 pulls). The
// uniform scheduler, the baseline, picks an arm uniformly at random. `replay` runs either on a recorded pool.
//
// The random numbers are the scheduler's own (splitmix64, Box-Muller, Marsaglia-Tsang), not std::*_distribution, so
// every standard library draws the same.
//
// Rule 10: the scheduler only decides which of Cypha's own hypotheses get tested next. Nothing here answers a question.
#pragma once

#include "cypha/eval/ecs.hpp"
#include "cypha/eval/hypothesis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cypha::eval::curiosity {

// ---------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------

/// splitmix64 with the draws the scheduler needs.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : s_(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (s_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  /// Uniform in [0, 1).
  double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
  /// Uniform in (0, 1).
  double uniform_open() {
    double u = uniform();
    while (u <= 0.0) u = uniform();
    return u;
  }
  /// Uniform integer in [0, n); 0 when n is 0.
  std::size_t below(std::size_t n) {
    if (n == 0) return 0;
    return std::min(n - 1, static_cast<std::size_t>(uniform() * static_cast<double>(n)));
  }
  double normal() {
    const double u1 = uniform_open(), u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
  /// Gamma(a, 1), a > 0 (Marsaglia and Tsang 2000; a < 1 by Gamma(a + 1) U^(1/a)).
  double gamma(double a) {
    if (a <= 0.0) return 0.0;
    if (a < 1.0) return gamma(a + 1.0) * std::pow(uniform_open(), 1.0 / a);
    const double d = a - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
      double x = 0.0, v = 0.0;
      do {
        x = normal();
        v = 1.0 + c * x;
      } while (v <= 0.0);
      v = v * v * v;
      const double u = uniform_open();
      if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
      if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
    }
  }
  double beta(double a, double b) {
    const double x = gamma(a), y = gamma(b);
    return x + y > 0.0 ? x / (x + y) : 0.5;
  }

 private:
  std::uint64_t s_;
};

/// A seed for (draw, seed, stream) that no two triples share in practice.
inline std::uint64_t mix_seed(std::uint64_t a, std::uint64_t b, std::uint64_t c = 0) {
  Rng r(a * 0x9E3779B97F4A7C15ULL ^ (b + 0x632BE59BD9B4E019ULL) * 0xD1B54A32D192ED03ULL ^ (c + 1) * 0x94D049BB133111EBULL);
  return r.next();
}

// ---------------------------------------------------------------------------
// The value of a pull (DISCOVERY §9)
// ---------------------------------------------------------------------------

/// Certainty gained by a promotion: max(0, posterior - P0).
inline double certainty_gained(double prior, double posterior) { return std::max(0.0, posterior - prior); }

/// Bits of uncertainty a pull removed: max(0, H(P0) - H(posterior)).
inline double bits_removed(double prior, double posterior) {
  return std::max(0.0, binary_entropy_bits(prior) - binary_entropy_bits(posterior));
}

/// The reward the scheduler sees: impact x certainty gained when `decide` supports the hypothesis, else 0.
inline double pull_reward(bool supported, double impact, double prior, double posterior) {
  return supported ? std::max(0.0, impact) * certainty_gained(prior, posterior) : 0.0;
}

// ---------------------------------------------------------------------------
// The scheduler
// ---------------------------------------------------------------------------

struct Arm {
  std::string name;
  std::string generator;  // floors are per generator
  Tier tier = Tier::Wild;  // the wild tier has its own floor
};

struct ArmState {
  double pulls = 0.0, cost = 0.0, reward = 0.0, promoted = 0.0;  // reward: only promotions pay
  std::deque<double> recent;  // bits removed per unit cost, the last lp_window pulls
  bool retired = false;
};

enum class Policy { Uniform, Thompson };

struct SchedulerConfig {
  Policy policy = Policy::Thompson;
  double prior_strength = 2.0;       // Beta(m rho, m (1 - rho)), rho = (all promotions + 1) / (all pulls + 2)
  double reward_prior = 0.5;         // the mean reward per promotion before any promotion
  double lp_weight = 0.0;            // lambda: score x (1 + lambda LP)
  std::size_t lp_window = 10;
  bool floors = true;
  double floor_generator = 0.02;     // each generator keeps 2% of the cost spent ...
  double floor_wild = 0.05;          // ... and the wild tier 5%
  double floor_after = 50.0;         // floors apply once this much cost is spent
  bool retire = true;
  double retire_min_pulls = 200.0;
  double retire_upper = 0.05;        // Wilson 95% upper bound of the promotion rate
};

class Scheduler {
 public:
  Scheduler(std::vector<Arm> arms, SchedulerConfig cfg, std::uint64_t seed)
      : arms_(std::move(arms)), cfg_(cfg), st_(arms_.size()), rng_(seed) {
    for (const Arm& a : arms_)
      if (std::find(gens_.begin(), gens_.end(), a.generator) == gens_.end()) gens_.push_back(a.generator);
  }

  std::size_t size() const { return arms_.size(); }
  const Arm& arm(std::size_t a) const { return arms_[a]; }
  const ArmState& state(std::size_t a) const { return st_[a]; }
  double spent() const { return spent_; }
  double pulls() const { return pulls_; }

  /// Mean bits removed per unit cost over the arm's recent pulls (0 before any).
  double learning_progress(std::size_t a) const {
    const auto& r = st_[a].recent;
    if (r.empty()) return 0.0;
    double s = 0.0;
    for (double x : r) s += x;
    return s / static_cast<double>(r.size());
  }

  /// Expected cost of the arm's next pull: (cost spent + c0) / (pulls + 1), c0 the mean cost of all pulls so far.
  double expected_cost(std::size_t a) const {
    const double c0 = pulls_ > 0.0 ? spent_ / pulls_ : 1.0;
    return std::max(1e-9, (st_[a].cost + c0) / (st_[a].pulls + 1.0));
  }

  /// The pool's promotion rate, (all promotions + 1) / (all pulls + 2): the mean of every arm's prior.
  double pooled_rate() const { return (promoted_ + 1.0) / (pulls_ + 2.0); }

  /// The arm's promotion probability posterior, Beta(a, b).
  std::pair<double, double> beta_of(std::size_t a) const {
    const double rho = pooled_rate();
    return {cfg_.prior_strength * rho + st_[a].promoted, cfg_.prior_strength * (1.0 - rho) + st_[a].pulls - st_[a].promoted};
  }

  /// Mean reward per promotion: (the arm's reward + m0) / (its promotions + 1), m0 the mean over all arms so far.
  double reward_size(std::size_t a) const {
    const double m0 = promoted_ > 0.0 ? reward_ / promoted_ : cfg_.reward_prior;
    return (st_[a].reward + m0) / (st_[a].promoted + 1.0);
  }

  /// The Thompson score of a draw q of the arm's promotion probability: q m / expected cost x (1 + lambda LP).
  double score(std::size_t a, double q) const {
    return q * reward_size(a) / expected_cost(a) * (1.0 + cfg_.lp_weight * learning_progress(a));
  }

  /// The next arm among those with hypotheses left (available[a]); -1 when none may be pulled.
  int choose(const std::vector<bool>& available) {
    std::vector<std::size_t> cand;
    for (std::size_t a = 0; a < arms_.size(); ++a)
      if (a < available.size() && available[a] && !st_[a].retired) cand.push_back(a);
    if (cand.empty()) return -1;
    if (cfg_.policy == Policy::Uniform) return static_cast<int>(cand[rng_.below(cand.size())]);
    std::vector<double> sc(arms_.size(), -1.0);
    for (std::size_t a : cand) {
      const auto [ba, bb] = beta_of(a);
      sc[a] = score(a, rng_.beta(ba, bb));
    }
    auto best_of = [&](auto keep) {
      int best = -1;
      for (std::size_t a : cand)
        if (keep(a) && (best < 0 || sc[a] > sc[static_cast<std::size_t>(best)])) best = static_cast<int>(a);
      return best;
    };
    if (cfg_.floors && spent_ >= cfg_.floor_after) {
      // The group furthest under its floor (by share of the cost spent) gets the pull.
      double worst = 0.0;
      int pick = -1;
      for (const std::string& g : gens_) {
        const int b = best_of([&](std::size_t a) { return arms_[a].generator == g; });
        if (b < 0) continue;
        double c = 0.0;
        for (std::size_t a = 0; a < arms_.size(); ++a)
          if (arms_[a].generator == g) c += st_[a].cost;
        const double deficit = cfg_.floor_generator - c / spent_;
        if (deficit > worst) worst = deficit, pick = b;
      }
      const int bw = best_of([&](std::size_t a) { return arms_[a].tier == Tier::Wild; });
      if (bw >= 0) {
        double c = 0.0;
        for (std::size_t a = 0; a < arms_.size(); ++a)
          if (arms_[a].tier == Tier::Wild) c += st_[a].cost;
        const double deficit = cfg_.floor_wild - c / spent_;
        if (deficit > worst) worst = deficit, pick = bw;
      }
      if (pick >= 0) return pick;
    }
    return best_of([](std::size_t) { return true; });
  }

  /// Records a pull of arm a: its cost, its reward, the bits it removed and whether it was promoted.
  void update(std::size_t a, double cost, double reward, double bits, bool promoted) {
    ArmState& s = st_[a];
    cost = std::max(0.0, cost);
    reward = promoted ? std::max(0.0, reward) : 0.0;
    s.pulls += 1.0, s.cost += cost, s.reward += reward, s.promoted += promoted ? 1.0 : 0.0;
    s.recent.push_back(cost > 0.0 ? bits / cost : bits);
    while (s.recent.size() > cfg_.lp_window) s.recent.pop_front();
    spent_ += cost, pulls_ += 1.0, reward_ += reward, promoted_ += promoted ? 1.0 : 0.0;
    if (cfg_.retire && s.pulls >= cfg_.retire_min_pulls && wilson_upper(s.promoted, s.pulls) < cfg_.retire_upper) s.retired = true;
  }

 private:
  std::vector<Arm> arms_;
  SchedulerConfig cfg_;
  std::vector<ArmState> st_;
  std::vector<std::string> gens_;
  Rng rng_;
  double spent_ = 0.0, pulls_ = 0.0, reward_ = 0.0, promoted_ = 0.0;
};

// ---------------------------------------------------------------------------
// Replay on a recorded pool
// ---------------------------------------------------------------------------

/// One recorded hypothesis. The scheduler sees arm, cost, reward, bits and supported; `graded` is for scoring only.
struct PoolItem {
  int arm = 0;
  double cost = 1.0, reward = 0.0, bits = 0.0;
  double prior = 0.0, posterior = 0.0, impact = 1.0, demand = 0.0;
  bool supported = false;
  int graded = -1;  // 1 true, 0 false, -1 ungraded
};

enum class WithinArm { Random, PriorFirst };

/// Each arm's hypotheses in the order it hands them out: demand first, then random, or highest P0 first (ties random).
/// The order never looks at an outcome.
inline std::vector<std::vector<std::size_t>> arm_orders(const std::vector<PoolItem>& pool, std::size_t n_arms, WithinArm order,
                                                         std::uint64_t seed) {
  std::vector<std::vector<std::size_t>> out(n_arms);
  for (std::size_t i = 0; i < pool.size(); ++i)
    if (pool[i].arm >= 0 && static_cast<std::size_t>(pool[i].arm) < n_arms) out[static_cast<std::size_t>(pool[i].arm)].push_back(i);
  Rng rng(seed);
  for (auto& v : out) {
    for (std::size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng.below(i)]);  // Fisher-Yates: random ties
    std::stable_sort(v.begin(), v.end(), [&](std::size_t x, std::size_t y) {
      if (pool[x].demand != pool[y].demand) return pool[x].demand > pool[y].demand;
      return order == WithinArm::PriorFirst && pool[x].prior > pool[y].prior;
    });
  }
  return out;
}

/// The pool with each arm resampled with replacement to its own size (a stratified bootstrap).
inline std::vector<PoolItem> stratified_bootstrap(const std::vector<PoolItem>& pool, std::size_t n_arms, std::uint64_t seed) {
  std::vector<std::vector<std::size_t>> by(n_arms);
  for (std::size_t i = 0; i < pool.size(); ++i)
    if (pool[i].arm >= 0 && static_cast<std::size_t>(pool[i].arm) < n_arms) by[static_cast<std::size_t>(pool[i].arm)].push_back(i);
  Rng rng(seed);
  std::vector<PoolItem> out;
  out.reserve(pool.size());
  for (const auto& v : by)
    for (std::size_t k = 0; k < v.size(); ++k) out.push_back(pool[v[rng.below(v.size())]]);
  return out;
}

struct ReplayResult {
  double cost = 0.0, reward = 0.0, pulls = 0.0, promoted = 0.0;
  double value = 0.0;          // confirmed value: graded true promotions minus graded false ones, each impact x certainty gained
  double graded = 0.0, graded_false = 0.0, brier = 0.0;  // over graded promotions
  std::vector<double> arm_pulls, arm_promoted;
  std::vector<bool> retired;
};

/// Runs a scheduler on the pool until its cost reaches `budget` (the pull that crosses it completes) or nothing is left.
/// Each arm's hypotheses are drawn without replacement in `orders`.
inline ReplayResult replay(const std::vector<PoolItem>& pool, const std::vector<Arm>& arms, const std::vector<std::vector<std::size_t>>& orders,
                           const SchedulerConfig& cfg, double budget, std::uint64_t seed) {
  Scheduler sch(arms, cfg, seed);
  ReplayResult r;
  r.arm_pulls.assign(arms.size(), 0.0), r.arm_promoted.assign(arms.size(), 0.0);
  std::vector<std::size_t> next(arms.size(), 0);
  std::vector<bool> avail(arms.size());
  while (r.cost < budget) {
    for (std::size_t a = 0; a < arms.size(); ++a) avail[a] = a < orders.size() && next[a] < orders[a].size();
    const int a = sch.choose(avail);
    if (a < 0) break;
    const std::size_t ua = static_cast<std::size_t>(a);
    const PoolItem& it = pool[orders[ua][next[ua]++]];
    sch.update(ua, it.cost, it.reward, it.bits, it.supported);
    r.cost += it.cost, r.reward += it.reward, r.pulls += 1.0, r.arm_pulls[ua] += 1.0;
    if (!it.supported) continue;
    r.promoted += 1.0, r.arm_promoted[ua] += 1.0;
    if (it.graded < 0) continue;
    const double v = std::max(0.0, it.impact) * certainty_gained(it.prior, it.posterior);
    r.graded += 1.0;
    if (it.graded == 1) r.value += v;
    else r.value -= v, r.graded_false += 1.0;
    const double e = it.posterior - (it.graded == 1 ? 1.0 : 0.0);
    r.brier += e * e;
  }
  for (std::size_t a = 0; a < arms.size(); ++a) r.retired.push_back(sch.state(a).retired);
  return r;
}

}  // namespace cypha::eval::curiosity
