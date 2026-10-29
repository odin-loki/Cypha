// S4b (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4b"): DISCOVERY §3.4's unknown-unknown signals under one alarm rule.
//
// A unit (a page) has a score (high = unusual) and a stratum (its text type). Its conformal p-value against a reference
// window of its stratum gives an event, p < alpha, so P(event) = alpha while nothing has changed. A Shiryaev–Roberts
// detector on the events raises an alarm when their rate rises; each alarm opens a gap (the caller writes it). The
// reference is running (the last `cap` units of the stratum read) or fixed (a calibration set).
//
// Rule 10: these are Cypha's self-assessment signals. Nothing here answers a question for CyphaLM.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

namespace cypha::eval {

// The registered streams: calibration C after CyphaLM's protocol slice, the test stream S after C (body offsets).
inline constexpr std::size_t kSignalCalibFrom = 9437184;
inline constexpr std::size_t kSignalSplit = 30000000;
inline constexpr std::size_t kSignalTestTo = 50000000;
inline constexpr std::size_t kSignalReaderTo = 7340032;  // V's reader starts with the pages before this (CyphaLM's train)

/// The two page types the store's reader skips ("List of ..." and "... (disambiguation)", pattern_learn's rule): the
/// planted pool L, and never part of C or S.
inline bool signal_store_skips(const std::string& title) {
  std::string t = title;
  for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return t.compare(0, 8, "list of ") == 0 || t.find("(disambiguation)") != std::string::npos;
}

/// A unit's identity: FNV-1a 64 of "set:offset".
inline std::uint64_t signal_unit_id(const std::string& set, std::size_t offset) {
  std::uint64_t h = 1469598103934665603ull;
  for (char c : set + ":" + std::to_string(offset)) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
  return h;
}

/// The tie-break draw V in [0, 1) of a unit: splitmix64 of its id.
inline double signal_uniform(std::uint64_t id) {
  std::uint64_t z = id + 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}

/// The conformal p-value of score s (high = unusual) against reference scores w: (#{w > s} + v (#{w = s} + 1)) / (M + 1).
/// Uniform on (0, 1) when s and the reference are exchangeable; v in [0, 1) randomises ties, so it stays exact for
/// discrete scores.
template <class Ref>
inline double conformal_p(const Ref& w, double s, double v) {
  std::size_t gt = 0, eq = 0;
  for (double x : w) {
    gt += x > s;
    eq += x == s;
  }
  return (static_cast<double>(gt) + v * static_cast<double>(eq + 1)) / static_cast<double>(w.size() + 1);
}

/// The registered rule: events at p < alpha; Shiryaev–Roberts against p1, alarm at R >= A; reference windows.
struct SignalRule {
  double alpha = 0.05;       // the event rate with no shift (p0)
  double p1 = 0.25;          // the registered alternative
  double A = 20000.0;        // P(an alarm within L units) <= L / A with no shift
  std::size_t cap = 500;     // running reference: the last cap units of the stratum
  std::size_t min_ref = 100; // a unit is scored once its stratum's reference holds this many
};

/// Shiryaev–Roberts on Bernoulli events: R_t = (1 + R_{t-1}) L_t, L = p1/p0 for an event, (1-p1)/(1-p0) otherwise.
/// An alarm at R >= A, after which R restarts at 0. With independent events at rate p0, E[L] = 1 and R_t - t is a
/// martingale, so P(an alarm within the first L units) <= L / A.
struct SRDetector {
  double p0 = 0.05, p1 = 0.25, A = 20000.0;
  double R = 0.0;
  std::size_t units = 0, events = 0;          // since the last restart
  double alarm_R = 0.0;                        // at the last alarm
  std::size_t alarm_units = 0, alarm_events = 0;
  SRDetector() = default;
  explicit SRDetector(const SignalRule& r) : p0(r.alpha), p1(r.p1), A(r.A) {}
  bool step(bool event) {
    R = (1.0 + R) * (event ? p1 / p0 : (1.0 - p1) / (1.0 - p0));
    ++units;
    events += event ? 1 : 0;
    if (R < A) return false;
    alarm_R = R, alarm_units = units, alarm_events = events;
    R = 0.0, units = 0, events = 0;
    return true;
  }
};

/// A scored unit: score (high = unusual), stratum, and the identity its tie-break draw comes from.
struct SignalUnit {
  double score = 0.0;
  int stratum = 0;
  std::uint64_t id = 0;
};

/// The reference windows, one per stratum. Running: push() keeps the last cap units. Fixed: filled once, push() ignored.
struct SignalRef {
  std::vector<std::deque<double>> w;
  bool running = true;
  std::size_t cap = 500, min_ref = 100;
  SignalRef(int strata, bool run, const SignalRule& r) : w(static_cast<std::size_t>(strata)), running(run), cap(r.cap), min_ref(r.min_ref) {}
  bool ready(int k) const { return w[static_cast<std::size_t>(k)].size() >= min_ref; }
  double p(const SignalUnit& u) const { return conformal_p(w[static_cast<std::size_t>(u.stratum)], u.score, signal_uniform(u.id)); }
  void push(int k, double s) {
    if (!running) return;
    auto& d = w[static_cast<std::size_t>(k)];
    d.push_back(s);
    if (d.size() > cap) d.pop_front();
  }
  /// The fixed reference: every unit of the calibration set, by stratum.
  void fill(const std::vector<SignalUnit>& cal) {
    for (const SignalUnit& u : cal) w[static_cast<std::size_t>(u.stratum)].push_back(u.score);
  }
};

struct SignalAlarm {
  bool in_test = false;  // S (else C)
  std::size_t pos = 0;   // index in its set
  double R = 0.0;
  std::size_t units = 0, events = 0;  // since the previous restart, the alarm's unit included
};

/// The unshifted run. Running reference: C then S read as one sequence, the windows starting empty. Fixed: the windows
/// are C's units and only S is read. r_before[t] and the reference at S[t] are the state a planted shift at t starts from.
struct SignalRun {
  std::vector<SignalAlarm> alarms;
  std::size_t c_scored = 0, c_events = 0, s_scored = 0, s_events = 0;
  std::vector<double> r_before;        // per S index: R before the unit
  std::vector<signed char> s_event;    // per S index: 1 event, 0 none, -1 not scored
  std::vector<double> s_p;             // per S index: the p-value (-1 not scored)
};

inline SignalRun signal_run(const std::vector<SignalUnit>& C, const std::vector<SignalUnit>& S, int strata, bool running,
                            const SignalRule& rule) {
  SignalRun out;
  SignalRef ref(strata, running, rule);
  if (!running) ref.fill(C);
  SRDetector sr(rule);
  auto read = [&](const SignalUnit& u, bool test, std::size_t pos) {
    double p = -1.0;
    signed char ev = -1;
    if (test) out.r_before.push_back(sr.R);
    if (ref.ready(u.stratum)) {
      p = ref.p(u);
      ev = p < rule.alpha ? 1 : 0;
      (test ? out.s_scored : out.c_scored) += 1;
      (test ? out.s_events : out.c_events) += static_cast<std::size_t>(ev);
      if (sr.step(ev == 1)) out.alarms.push_back({test, pos, sr.alarm_R, sr.alarm_units, sr.alarm_events});
    }
    if (test) out.s_event.push_back(ev), out.s_p.push_back(p);
    ref.push(u.stratum, u.score);
  };
  if (running)
    for (std::size_t i = 0; i < C.size(); ++i) read(C[i], false, i);
  for (std::size_t i = 0; i < S.size(); ++i) read(S[i], true, i);
  return out;
}

/// The reference a planted shift at S[t0] meets: fixed, C's units; running, the last cap units of each stratum read
/// before S[t0] (C then S).
inline SignalRef signal_ref_at(const std::vector<SignalUnit>& C, const std::vector<SignalUnit>& S, int strata, bool running,
                               const SignalRule& rule, std::size_t t0) {
  SignalRef ref(strata, running, rule);
  if (!running) {
    ref.fill(C);
    return ref;
  }
  std::vector<std::vector<double>> back(static_cast<std::size_t>(strata));
  auto take = [&](const SignalUnit& u) {
    auto& b = back[static_cast<std::size_t>(u.stratum)];
    if (b.size() < rule.cap) b.push_back(u.score);
  };
  for (std::size_t i = std::min(t0, S.size()); i-- > 0;) take(S[i]);
  for (std::size_t i = C.size(); i-- > 0;) take(C[i]);
  for (std::size_t k = 0; k < back.size(); ++k) ref.w[k].assign(back[k].rbegin(), back[k].rend());
  return ref;
}

/// One planted shift: the unshifted stream up to S[t0]; from t0 each unit is a plant with probability pi (the pool in a
/// random order, wrapped if it runs out), else S's next unit. delay: units read from t0 to the first alarm, both counted
/// (1 = the first unit read at t0), or 0 when there is none within `window` units. plant: the alarm's unit was a plant;
/// index: its index in the pool or in S.
struct SignalDetection {
  int delay = 0;
  bool plant = false;
  std::size_t index = 0;
  double R = 0.0;
  std::size_t units = 0, events = 0;
};

inline SignalDetection signal_planted(const std::vector<SignalUnit>& C, const std::vector<SignalUnit>& S,
                                      const std::vector<SignalUnit>& pool, int strata, bool running, const SignalRule& rule,
                                      const SignalRun& base, std::size_t t0, double pi, std::size_t window,
                                      std::mt19937_64& rng) {
  SignalDetection d;
  if (pool.empty() || t0 >= S.size()) return d;
  SignalRef ref = signal_ref_at(C, S, strata, running, rule, t0);
  SRDetector sr(rule);
  sr.R = base.r_before[t0];
  std::vector<std::size_t> order(pool.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::shuffle(order.begin(), order.end(), rng);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::size_t next_plant = 0, next_s = t0;
  for (std::size_t j = 0; j < window; ++j) {
    const bool plant = U(rng) < pi || next_s >= S.size();
    std::size_t idx;
    if (plant) {
      if (next_plant == order.size()) std::shuffle(order.begin(), order.end(), rng), next_plant = 0;
      idx = order[next_plant++];
    } else {
      idx = next_s++;
    }
    const SignalUnit& u = plant ? pool[idx] : S[idx];
    if (ref.ready(u.stratum)) {
      const bool ev = ref.p(u) < rule.alpha;
      if (sr.step(ev)) {
        d.delay = static_cast<int>(j + 1);
        d.plant = plant, d.index = idx, d.R = sr.alarm_R, d.units = sr.alarm_units, d.events = sr.alarm_events;
        return d;
      }
    }
    ref.push(u.stratum, u.score);
  }
  return d;
}

/// Delays over replicates (0 = missed within the window).
struct SignalDelayStats {
  std::size_t reps = 0, within50 = 0, within_window = 0;
  double median = 0.0, p90 = 0.0;  // with misses counted as window + 1
};

inline SignalDelayStats signal_delay_stats(const std::vector<int>& delays, std::size_t window) {
  SignalDelayStats s;
  s.reps = delays.size();
  std::vector<double> v;
  for (int d : delays) {
    s.within50 += d > 0 && d <= 50;
    s.within_window += d > 0;
    v.push_back(d > 0 ? d : static_cast<double>(window + 1));
  }
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  auto q = [&](double f) { return v[std::min(v.size() - 1, static_cast<std::size_t>(f * static_cast<double>(v.size() - 1) + 0.5))]; };
  s.median = q(0.5), s.p90 = q(0.9);
  return s;
}

/// The false-alarm check with no dependence between units: the scored events of an unshifted run in `reps` random
/// orders; the mean alarms per 1,000 scored units.
inline double signal_perm_alarms_per_k(const SignalRun& run, const SignalRule& rule, std::size_t reps, std::uint64_t seed) {
  std::vector<char> ev;
  for (signed char e : run.s_event)
    if (e >= 0) ev.push_back(static_cast<char>(e));
  if (ev.empty() || reps == 0) return 0.0;
  std::mt19937_64 rng(seed);
  double alarms = 0;
  for (std::size_t r = 0; r < reps; ++r) {
    std::shuffle(ev.begin(), ev.end(), rng);
    SRDetector sr(rule);
    for (char e : ev) alarms += sr.step(e != 0) ? 1 : 0;
  }
  return 1000.0 * alarms / (static_cast<double>(reps) * static_cast<double>(ev.size()));
}

/// U's score: -ln P(X <= x) for X ~ Binomial(n, q): high when fewer sentences parse than the type's rate predicts.
inline double binom_lower_tail_score(std::size_t n, std::size_t x, double q) {
  if (x >= n || q <= 0.0) return 0.0;
  if (q >= 1.0) return 745.0;  // P(X <= x < n) = 0: the score's cap (-ln of the smallest double)
  const double lq = std::log(q), l1 = std::log1p(-q);
  double lse = -INFINITY, lc = 0.0;  // log C(n, k), from k = 0
  for (std::size_t k = 0; k <= x; ++k) {
    if (k > 0) lc += std::log(static_cast<double>(n - k + 1)) - std::log(static_cast<double>(k));
    const double t = lc + static_cast<double>(k) * lq + static_cast<double>(n - k) * l1;
    lse = t > lse ? t + std::log1p(std::exp(lse - t)) : lse + std::log1p(std::exp(t - lse));
  }
  return std::max(0.0, -lse);
}

}  // namespace cypha::eval
