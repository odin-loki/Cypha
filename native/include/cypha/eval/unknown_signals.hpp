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
#include <unordered_map>
#include <utility>
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

// ---------------------------------------------------------------------------------------------------------------------
// S4c (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4c"): DISCOVERY §3.4's out-of-scope signal, read from a page's prose words
// only. A window of the first kOodWindow prose words of a page is scored against a reference word distribution (the
// reader pages'); three candidates: register (O1), rare content words (O2), topic distance (O3). Rule 10: self-assessment
// only.

inline constexpr std::size_t kOodWindow = 50;            // m: the words of a unit
inline constexpr std::size_t kOodTop = 100;              // F: the reference's most frequent words
inline constexpr double kOodSifA = 0.001;                // O3's weight a / (a + f_w)
inline constexpr std::size_t kSignalDevSplit = 19718592; // dev: C's first half calibrates, its second half is the stream
inline constexpr std::size_t kSignalPlainFrom = 50000000;  // pool P's pages start here (read by no S4b set)

/// S4c's extra list rule, for every page: a line whose first non-blank bytes are '-', a bullet (U+2022) or digits followed
/// by '.' or ')' is a list item. (S3's prose reader already drops '*' and '#' lists.)
inline bool ood_list_line(const std::string& src, std::size_t off) {
  std::size_t i = off;
  while (i < src.size() && (src[i] == ' ' || src[i] == '\t')) ++i;
  if (i >= src.size()) return false;
  const unsigned char c = static_cast<unsigned char>(src[i]);
  if (c == '-') return true;
  if (c == 0xE2 && i + 2 < src.size() && static_cast<unsigned char>(src[i + 1]) == 0x80 && static_cast<unsigned char>(src[i + 2]) == 0xA2) return true;
  std::size_t j = i;
  while (j < src.size() && src[j] >= '0' && src[j] <= '9') ++j;
  return j > i && j < src.size() && (src[j] == '.' || src[j] == ')');
}

/// The reference: word counts over the reader pages' prose words, and F (its kOodTop most frequent words, ties by word).
struct OodRef {
  std::unordered_map<std::string, std::uint64_t> count;
  std::uint64_t N = 0;
  std::vector<std::string> top;                      // F, most frequent first
  std::unordered_map<std::string, std::size_t> bin;  // F's word -> its bin
  std::vector<double> share;                         // p_b: F's shares, then the other words' share
  void add(const std::string& w) { ++count[w], ++N; }
  void finish(std::size_t k = kOodTop) {
    std::vector<std::pair<std::uint64_t, std::string>> v;
    v.reserve(count.size());
    for (const auto& [w, c] : count) v.push_back({c, w});
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    top.clear(), bin.clear(), share.clear();
    double in = 0;
    for (std::size_t i = 0; i < v.size() && i < k; ++i) {
      bin[v[i].second] = top.size();
      top.push_back(v[i].second);
      share.push_back(static_cast<double>(v[i].first) / static_cast<double>(N));
      in += share.back();
    }
    share.push_back(std::max(0.0, 1.0 - in));
  }
  bool in_top(const std::string& w) const { return bin.count(w) > 0; }
  double freq(const std::string& w) const {
    auto it = count.find(w);
    return it == count.end() ? 0.0 : static_cast<double>(it->second) / static_cast<double>(N);
  }
};

/// O1, register: G per word over F's bins and one bin for every other word, (2 / m) sum_b c_b ln(c_b / (m p_b)).
inline double ood_register(const std::vector<std::string>& win, const OodRef& ref) {
  if (win.empty() || ref.share.empty()) return 0.0;
  std::vector<double> c(ref.share.size(), 0.0);
  for (const std::string& w : win) {
    auto it = ref.bin.find(w);
    c[it == ref.bin.end() ? ref.share.size() - 1 : it->second] += 1.0;
  }
  const double m = static_cast<double>(win.size());
  double g = 0;
  for (std::size_t b = 0; b < c.size(); ++b)
    if (c[b] > 0 && ref.share[b] > 0) g += c[b] * std::log(c[b] / (m * ref.share[b]));
  return 2.0 * g / m;
}

/// O2, rare content words: the mean -ln p(w) over the window's words outside F, p(w) = (n_w + 0.5) / (N + 0.5 (V + 1)).
/// 0 when every word is in F.
inline double ood_rare(const std::vector<std::string>& win, const OodRef& ref) {
  const double den = static_cast<double>(ref.N) + 0.5 * static_cast<double>(ref.count.size() + 1);
  double s = 0;
  std::size_t n = 0;
  for (const std::string& w : win) {
    if (ref.in_top(w)) continue;
    auto it = ref.count.find(w);
    const double c = it == ref.count.end() ? 0.0 : static_cast<double>(it->second);
    s += -std::log((c + 0.5) / den), ++n;
  }
  return n ? s / static_cast<double>(n) : 0.0;
}

/// O4's profile (Burrows' Delta): the reference's K most frequent words (ties by the word) and one bin for every other word;
/// each bin's mean share and standard deviation over every whole m-word window of the reference pages' words (windows do
/// not cross pages). sd is at least the binomial sqrt(mu (1 - mu) / m).
inline constexpr std::size_t kOodProfileTop = 500;
struct OodProfile {
  std::unordered_map<std::string, std::size_t> bin;
  std::vector<double> mu, sd;
  std::size_t windows = 0, m = kOodWindow;
  void build(const OodRef& ref, const std::vector<std::vector<std::string>>& pages, std::size_t k = kOodProfileTop, std::size_t m_ = kOodWindow) {
    m = m_;
    std::vector<std::pair<std::uint64_t, std::string>> v;
    v.reserve(ref.count.size());
    for (const auto& [w, c] : ref.count) v.push_back({c, w});
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    bin.clear();
    for (std::size_t i = 0; i < v.size() && i < k; ++i) bin[v[i].second] = i;
    const std::size_t B = bin.size() + 1;
    std::vector<double> sum(B, 0.0), sq(B, 0.0), c(B, 0.0);
    windows = 0;
    for (const auto& words : pages)
      for (std::size_t i = 0; i + m <= words.size(); i += m) {
        std::fill(c.begin(), c.end(), 0.0);
        for (std::size_t j = i; j < i + m; ++j) {
          auto it = bin.find(words[j]);
          c[it == bin.end() ? B - 1 : it->second] += 1.0;
        }
        for (std::size_t b = 0; b < B; ++b) {
          const double x = c[b] / static_cast<double>(m);
          sum[b] += x, sq[b] += x * x;
        }
        ++windows;
      }
    mu.assign(B, 0.0), sd.assign(B, 0.0);
    for (std::size_t b = 0; b < B && windows > 0; ++b) {
      const double W = static_cast<double>(windows);
      mu[b] = sum[b] / W;
      const double floor = std::sqrt(std::max(0.0, mu[b] * (1.0 - mu[b])) / static_cast<double>(m));
      sd[b] = std::max(std::sqrt(std::max(0.0, sq[b] / W - mu[b] * mu[b])), floor);
    }
  }
};

/// O4, register profile: (1 / (K + 1)) sum_b |x_b - mu_b| / sd_b, x_b the bin's share of the window's words.
inline double ood_delta(const std::vector<std::string>& win, const OodProfile& pr) {
  if (win.empty() || pr.mu.empty()) return 0.0;
  const std::size_t B = pr.mu.size();
  std::vector<double> c(B, 0.0);
  for (const std::string& w : win) {
    auto it = pr.bin.find(w);
    c[it == pr.bin.end() ? B - 1 : it->second] += 1.0;
  }
  double d = 0;
  for (std::size_t b = 0; b < B; ++b)
    if (pr.sd[b] > 0) d += std::fabs(c[b] / static_cast<double>(win.size()) - pr.mu[b]) / pr.sd[b];
  return d / static_cast<double>(B);
}

/// O5's bank (topic novelty, the first-story-detection score): TF-IDF vectors of the bank pages over the words outside F,
/// normalised, with an inverted index. idf(w) = ln((N_d + 1) / (df_w + 1)) + 1; a word on no bank page gets ln(N_d + 1) + 1.
struct OodTfidf {
  std::unordered_map<std::string, double> idf;
  double idf_unseen = 1.0;
  std::unordered_map<std::string, std::vector<std::pair<std::uint32_t, double>>> post;  // word -> (page, weight)
  std::size_t pages = 0;
  void build(const OodRef& ref, const std::vector<std::vector<std::string>>& bank) {
    idf.clear(), post.clear();
    pages = bank.size();
    std::unordered_map<std::string, std::uint32_t> df;
    std::vector<std::unordered_map<std::string, double>> tf(bank.size());
    for (std::size_t j = 0; j < bank.size(); ++j) {
      for (const std::string& w : bank[j])
        if (!ref.in_top(w)) tf[j][w] += 1.0;
      for (const auto& [w, c] : tf[j]) ++df[w];
    }
    const double Nd = static_cast<double>(pages);
    for (const auto& [w, d] : df) idf[w] = std::log((Nd + 1.0) / (static_cast<double>(d) + 1.0)) + 1.0;
    idf_unseen = std::log(Nd + 1.0) + 1.0;
    for (std::size_t j = 0; j < bank.size(); ++j) {
      double nn = 0;
      for (auto& [w, c] : tf[j]) c *= idf[w], nn += c * c;
      if (nn <= 0) continue;
      nn = std::sqrt(nn);
      for (const auto& [w, c] : tf[j]) post[w].push_back({static_cast<std::uint32_t>(j), c / nn});
    }
  }
  /// O5: 1 - the largest cosine between the window's TF-IDF vector and a bank page's; 1 when no word is outside F.
  double novelty(const std::vector<std::string>& win, const OodRef& ref) const {
    std::unordered_map<std::string, double> tf;
    for (const std::string& w : win)
      if (!ref.in_top(w)) tf[w] += 1.0;
    double nn = 0;
    for (auto& [w, c] : tf) {
      auto it = idf.find(w);
      c *= it == idf.end() ? idf_unseen : it->second;
      nn += c * c;
    }
    if (nn <= 0 || pages == 0) return 1.0;
    nn = std::sqrt(nn);
    std::vector<double> dot(pages, 0.0);
    for (const auto& [w, c] : tf) {
      auto it = post.find(w);
      if (it == post.end()) continue;
      for (const auto& [j, x] : it->second) dot[j] += c / nn * x;
    }
    return 1.0 - std::max(0.0, *std::max_element(dot.begin(), dot.end()));
  }
};

/// O3's page vector: sum of a / (a + f_w) vec_w over the window's words outside F that have a vector, normalised; empty
/// when there is none. vec(w) returns nullptr for a word with no vector.
template <class VecOf>
inline std::vector<double> ood_page_vector(const std::vector<std::string>& win, const OodRef& ref, std::size_t dim, VecOf&& vec) {
  std::vector<double> v(dim, 0.0);
  bool any = false;
  for (const std::string& w : win) {
    if (ref.in_top(w)) continue;
    const float* x = vec(w);
    if (!x) continue;
    const double a = kOodSifA / (kOodSifA + ref.freq(w));
    for (std::size_t d = 0; d < dim; ++d) v[d] += a * static_cast<double>(x[d]);
    any = true;
  }
  double nn = 0;
  for (double x : v) nn += x * x;
  if (!any || nn <= 0) return {};
  nn = std::sqrt(nn);
  for (double& x : v) x /= nn;
  return v;
}

/// O3, topic distance: 1 - the largest cosine between a page vector and the bank's (unit vectors); 1 for an empty vector.
inline double ood_topic_distance(const std::vector<double>& v, const std::vector<std::vector<double>>& bank) {
  if (v.empty() || bank.empty()) return 1.0;
  double best = -1.0;
  for (const auto& b : bank) {
    double c = 0;
    for (std::size_t d = 0; d < v.size() && d < b.size(); ++d) c += v[d] * b[d];
    best = std::max(best, c);
  }
  return 1.0 - best;
}

// ---------------------------------------------------------------------------------------------------------------------
// S4b-b (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4b-b"): a link whose display tokens [b, e) lie inside one of a
// sentence's date candidates is a linked year or day ("[[1882]]" in "January 18, [[1882]]"). The date candidate reads it,
// so U does not count it as an entity mention.

/// Whether the token span [b, e) lies inside one of the spans (start <= b and e <= end).
inline bool span_within(int b, int e, const std::vector<std::pair<int, int>>& spans) {
  for (const auto& [s, t] : spans)
    if (s <= b && e <= t) return true;
  return false;
}

// ---------------------------------------------------------------------------------------------------------------------
// S4d (docs/reports/S4_UNSEEN_AND_FRONTIER.md, "S4d"): fresh pools and streams, and a second alarm threshold.

/// The article pages from kSignalPlainFrom (not lists, disambiguation pages, namespaces or redirects), by their
/// qualifying index q in dump order: S4c's control P took q % 4 = 0, S4d's control P2 takes q % 4 = 2, and S4d's
/// fresh unshifted stream F reads q % 4 in {1, 3}.
enum class SignalFreshPart { P, F, P2 };
inline SignalFreshPart signal_fresh_part(std::size_t q) {
  return q % 4 == 0 ? SignalFreshPart::P : q % 4 == 2 ? SignalFreshPart::P2 : SignalFreshPart::F;
}

/// S4d's second alarm threshold, read beside SignalRule's A = 20,000 on the same replicates.
inline constexpr double kSignalAHigh = 100000.0;

}  // namespace cypha::eval
