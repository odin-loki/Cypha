// Tests for the thought-process interpreter and standard learners (docs/THOUGHT.md K2, K3).
//
//   thought_test <path to docs/thought/processes.json>
//
// a) known-answer checks of each learner; b) the library loads, every process instantiates, and
// half_split / sprt_decide / tote run through the interpreter on synthetic environments; c) a
// selection benchmark over three processes with different success rates and costs; d) a trace
// replay. Returns nonzero if any check fails. Author: Odin Loch.
#include "cypha/eval/ecs.hpp"
#include "cypha/thought/interpreter.hpp"
#include "cypha/thought/learners.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace cypha::thought;

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    ++g_checks;                                                                  \
    if (!(cond)) {                                                               \
      ++g_failed;                                                                \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                \
    }                                                                            \
  } while (0)

#define CHECK_NEAR(x, y, tol)                                                    \
  do {                                                                           \
    ++g_checks;                                                                  \
    const double _x = (x), _y = (y);                                             \
    if (!(std::fabs(_x - _y) <= (tol))) {                                        \
      ++g_failed;                                                                \
      std::printf("FAIL %s:%d  %s = %.10g, expected %.10g (tol %g)\n", __FILE__, \
                  __LINE__, #x, _x, _y, static_cast<double>(tol));               \
    }                                                                            \
  } while (0)

// ---------------------------------------------------------------------------
// a) learners, known answers
// ---------------------------------------------------------------------------

static void test_random_draws() {
  Rng rng(7);
  // uniform01 is open and has mean 1/2
  double s = 0, lo = 1, hi = 0;
  for (int i = 0; i < 20000; ++i) {
    const double u = uniform01(rng);
    s += u;
    lo = std::min(lo, u);
    hi = std::max(hi, u);
  }
  CHECK(lo > 0.0 && hi < 1.0);
  CHECK_NEAR(s / 20000, 0.5, 0.01);
  // Gamma(k, 1): mean k, variance k (k < 1 uses the boost)
  for (double k : {0.5, 1.0, 3.7}) {
    Welford w;
    for (int i = 0; i < 40000; ++i) w.update(gamma_sample(rng, k));
    CHECK_NEAR(w.mean(), k, 5.0 * std::sqrt(k / 40000.0));
    CHECK_NEAR(w.variance(), k, 0.1 * k + 0.02);
  }
  // same seed, same stream
  Rng r1(99), r2(99);
  CHECK(gamma_sample(r1, 2.3) == gamma_sample(r2, 2.3));
  CHECK(normal01(r1) == normal01(r2));
}

static void test_beta() {
  Beta b;
  CHECK_NEAR(b.mean(), 0.5, 1e-15);
  b.update(1.0);
  b.update(1.0);
  b.update(1.0);
  b.update(0.0);
  CHECK_NEAR(b.a, 4.0, 1e-15);  // Beta(1+3, 1+1)
  CHECK_NEAR(b.b, 2.0, 1e-15);
  CHECK_NEAR(b.mean(), 2.0 / 3.0, 1e-15);
  CHECK_NEAR(b.variance(), 8.0 / (36.0 * 7.0), 1e-15);
  // fractional credit (3.6): a += e R, b += e (1 - R)
  Beta f;
  f.update(1.0, 0.64);
  f.update(0.25, 2.0);
  CHECK_NEAR(f.a, 1.0 + 0.64 + 0.5, 1e-15);
  CHECK_NEAR(f.b, 1.0 + 1.5, 1e-15);
  // sampling matches the moments of Beta(2, 5)
  Rng rng(11);
  Beta p;
  p.a = 2;
  p.b = 5;
  Welford w;
  for (int i = 0; i < 40000; ++i) w.update(p.sample(rng));
  CHECK_NEAR(w.mean(), 2.0 / 7.0, 0.004);
  CHECK_NEAR(w.variance(), 10.0 / (49.0 * 8.0), 0.002);
}

static void test_dirichlet() {
  Dirichlet d(3);
  CHECK_NEAR(d.mean(0), 1.0 / 3.0, 1e-15);
  d.update(0, 2.0);
  d.update(2, 0.5);
  CHECK_NEAR(d.alpha[0], 3.0, 1e-15);
  CHECK_NEAR(d.alpha[1], 1.0, 1e-15);
  CHECK_NEAR(d.alpha[2], 1.5, 1e-15);
  CHECK_NEAR(d.mean(0), 3.0 / 5.5, 1e-15);
  CHECK_NEAR(d.mean(1), 1.0 / 5.5, 1e-15);
  CHECK_NEAR(d.mean(2), 1.5 / 5.5, 1e-15);
  Rng rng(5);
  std::vector<Welford> w(3);
  for (int i = 0; i < 30000; ++i) {
    const std::vector<double> th = d.sample(rng);
    double s = 0;
    for (std::size_t j = 0; j < 3; ++j) {
      w[j].update(th[j]);
      s += th[j];
    }
    if (i < 100) CHECK_NEAR(s, 1.0, 1e-12);
  }
  for (std::size_t j = 0; j < 3; ++j) CHECK_NEAR(w[j].mean(), d.mean(j), 0.006);
  // a heavily favoured branch is the Thompson choice almost always
  Dirichlet e(2);
  e.update(1, 50.0);
  int n1 = 0;
  for (int i = 0; i < 2000; ++i) n1 += e.choose(rng) == 1;
  CHECK(n1 > 1900);
}

static void test_welford() {
  // known answer: the textbook set
  Welford k;
  for (double x : {2, 4, 4, 4, 5, 5, 7, 9}) k.update(x);
  CHECK_NEAR(k.mean(), 5.0, 1e-13);
  CHECK_NEAR(k.variance(), 32.0 / 7.0, 1e-13);
  // against a two-pass computation, on data with a large offset
  Rng rng(3);
  std::vector<double> xs;
  Welford w;
  for (int i = 0; i < 5000; ++i) {
    const double x = 1.0e6 + 25.0 * normal01(rng);
    xs.push_back(x);
    w.update(x);
  }
  long double m = 0;  // long double so the reference is more accurate than the thing it checks
  for (double x : xs) m += x;
  m /= static_cast<long double>(xs.size());
  long double ss = 0;
  for (double x : xs) ss += (x - m) * (x - m);
  const double var2 = static_cast<double>(ss / static_cast<long double>(xs.size() - 1));
  CHECK_NEAR(w.mean(), static_cast<double>(m), 1e-7);  // relative 1e-13 at an offset of 1e6 (one ulp is 1.2e-10)
  CHECK_NEAR(w.variance() / var2, 1.0, 1e-9);
  // log-space variant: ln c = 0, 1, 2
  LogWelford lw;
  lw.update(1.0);
  lw.update(std::exp(1.0));
  lw.update(std::exp(2.0));
  CHECK_NEAR(lw.log_mean(), 1.0, 1e-13);
  CHECK_NEAR(lw.log_variance(), 1.0, 1e-13);
  CHECK_NEAR(lw.geometric_mean(), std::exp(1.0), 1e-12);
  CHECK_NEAR(lw.expected(), std::exp(1.5), 1e-12);
  CHECK_NEAR(LogWelford().expected(3.5), 3.5, 0.0);  // no data: the prior
  // the cost posterior is centred on the mean of the log
  Rng r2(8);
  Welford draws;
  for (int i = 0; i < 20000; ++i) draws.update(std::log(lw.sample(r2)) - 0.5);
  CHECK_NEAR(draws.mean(), 1.0, 0.02);
}

static void test_thompson_actr_td_voc() {
  Rng rng(21);
  // Thompson: a clearly better arm wins; the epsilon floor gives every arm at least eps (3.5)
  Beta good, bad;
  good.a = 50;
  good.b = 5;
  bad.a = 5;
  bad.b = 50;
  LogWelford c;
  c.update(1.0);
  c.update(1.0);
  std::vector<ThompsonArm> arms{{&good, &c}, {&bad, &c}};
  int pick_bad_nofloor = 0, pick_bad_floor = 0;
  for (int i = 0; i < 20000; ++i) {
    pick_bad_nofloor += thompson_select(arms, 10.0, 0.0, rng).index == 1;
    pick_bad_floor += thompson_select(arms, 10.0, 0.02, rng).index == 1;
  }
  CHECK(pick_bad_nofloor == 0);
  CHECK_NEAR(pick_bad_floor / 20000.0, 0.02, 0.004);  // floor: each of 2 arms keeps eps = 2%
  // a cheaper arm of equal success wins on U = p G - C
  Beta same;
  same.a = 80;
  same.b = 20;
  LogWelford cheap, dear;
  for (int i = 0; i < 5; ++i) {
    cheap.update(1.0);
    dear.update(6.0);
  }
  std::vector<ThompsonArm> arms2{{&same, &dear}, {&same, &cheap}};
  int cheap_wins = 0;
  for (int i = 0; i < 2000; ++i) cheap_wins += thompson_select(arms2, 10.0, 0.0, rng).index == 1;
  CHECK(cheap_wins > 1990);
  CHECK_NEAR(utility(0.8, 10.0, 3.0), 5.0, 1e-15);

  // ACT-R: U <- U + a (R - U)
  ActRUtility u;
  u.alpha = 0.1;
  u.update(1.0);
  CHECK_NEAR(u.u, 0.1, 1e-15);
  u.update(1.0);
  CHECK_NEAR(u.u, 0.19, 1e-15);
  u.update(0.0);
  CHECK_NEAR(u.u, 0.171, 1e-15);

  // TD(lambda), 4-step trace, lambda = 0.8: e_k = 0.8^(4-k)
  const std::vector<double> e = td_credits(4, 0.8);
  CHECK(e.size() == 4);
  CHECK_NEAR(e[0], 0.512, 1e-15);
  CHECK_NEAR(e[1], 0.64, 1e-15);
  CHECK_NEAR(e[2], 0.8, 1e-15);
  CHECK_NEAR(e[3], 1.0, 1e-15);
  CHECK(td_credits(0).empty());
  CHECK_NEAR(td_credits(1)[0], 1.0, 1e-15);
  Beta step;  // credit applied to a Beta: R = 1 -> a += e, R = 0 -> b += e
  step.update(1.0, e[0]);
  CHECK_NEAR(step.a, 1.512, 1e-15);
  CHECK_NEAR(step.b, 1.0, 1e-15);

  // VOC sign (3.7): VOC = G * E[gain] - E[cost]
  Welford gain_hi, gain_lo;
  LogWelford cost_lo, cost_hi;
  for (int i = 0; i < 10; ++i) {
    gain_hi.update(0.5);
    gain_lo.update(0.05);
    cost_lo.update(0.1);
    cost_hi.update(0.2);
  }
  CHECK_NEAR(voc(gain_hi, cost_lo), 0.4, 1e-12);
  CHECK(!voc_stop(gain_hi, cost_lo));   // VOC > 0: keep going
  CHECK_NEAR(voc(gain_lo, cost_hi), -0.15, 1e-12);
  CHECK(voc_stop(gain_lo, cost_hi));    // VOC < 0: stop
  CHECK(!voc_stop(gain_lo, cost_hi, 10.0));  // a bigger goal value flips it: 10*0.05 - 0.2 > 0
  Welford g3;
  LogWelford c3;
  for (int i = 0; i < 3; ++i) {
    g3.update(0.0);
    c3.update(1.0);
  }
  CHECK(!voc_stop(g3, c3));             // fewer than min_n observations: no basis to stop
}

static void test_wilson_and_calibration() {
  // Wilson 95%: 8/10 -> (0.490162, 0.943318); 0/10 -> lower bound 0
  const WilsonInterval w = wilson_interval(8, 10);
  CHECK_NEAR(w.lo, 0.49016247153664183, 1e-12);
  CHECK_NEAR(w.hi, 0.9433178485456247, 1e-12);
  CHECK_NEAR(wilson_lower(45, 50), 0.7863976856252035, 1e-12);
  CHECK_NEAR(wilson_lower(0, 10), 0.0, 1e-15);
  CHECK_NEAR(wilson_upper(0, 10), 0.2775327998628892, 1e-12);
  Beta b;
  b.update(1.0, 8.0);
  b.update(0.0, 2.0);
  CHECK_NEAR(b.wilson().lo, w.lo, 1e-12);

  // Brier and ECE
  CalibrationLog cal;
  for (int i = 0; i < 10; ++i) cal.add("calibrated", 0.7, i < 7 ? 1.0 : 0.0);
  CHECK_NEAR(cal.ece("calibrated"), 0.0, 1e-12);
  CHECK_NEAR(cal.brier("calibrated"), 0.7 * 0.09 + 0.3 * 0.49, 1e-12);
  for (int i = 0; i < 10; ++i) cal.add("overconfident", 0.9, i < 5 ? 1.0 : 0.0);
  CHECK_NEAR(cal.ece("overconfident"), 0.4, 1e-12);
  CHECK_NEAR(cal.brier("overconfident"), 0.5 * 0.01 + 0.5 * 0.81, 1e-12);
  CHECK_NEAR(cal.ece_all(), 0.2, 1e-12);  // (10*0 + 10*0.4) / 20
  CHECK(cal.count("overconfident") == 10);
  // the isotonic correction pulls 0.9 down to what actually happened
  for (int i = 0; i < 10; ++i) cal.add("overconfident", 0.2, i < 2 ? 1.0 : 0.0);
  CHECK_NEAR(cal.corrected("overconfident", 0.9), 0.5, 1e-12);
  CHECK_NEAR(cal.corrected("overconfident", 0.2), 0.2, 1e-12);

  // streaming ECE equals eval::expected_calibration_error on 0/1 outcomes
  Rng rng(17);
  std::vector<double> p;
  std::vector<int> y;
  CalibrationLog c2;
  for (int i = 0; i < 3000; ++i) {
    const double ph = uniform01(rng);
    const int yy = uniform01(rng) < 0.5 * ph + 0.25 ? 1 : 0;
    p.push_back(ph);
    y.push_back(yy);
    c2.add("x", ph, yy);
  }
  CHECK_NEAR(c2.ece("x"), cypha::eval::expected_calibration_error(p, y, 10), 1e-12);
  CHECK_NEAR(c2.brier("x"), cypha::eval::brier_score(p, y), 1e-12);
}

// ---------------------------------------------------------------------------
// Test environments
// ---------------------------------------------------------------------------

static Json synth(const std::string& id, const Json& steps, const std::string& start) {
  return Json{{"id", id}, {"name", id}, {"cat", "solve"}, {"ref", "test"}, {"when", "test"}, {"start", start},
              {"steps", steps}, {"stop", "n/a"}, {"math", "n/a"}, {"learn", Json::array()}, {"feedback", "test"}};
}

/// The library plus the synthetic processes used by the tests.
static std::shared_ptr<const Library> make_library(const Library& base) {
  Library lib = base;
  const Json three = {{"work", {{"op", "act"}, {"next", {"check"}}}},
                      {"check", {{"op", "test"}, {"next", {"out"}}}},
                      {"out", {{"op", "return"}}}};
  for (const char* id : {"syn_cheap", "syn_best", "syn_sure"}) lib.add_process(synth(id, three, "work"));
  lib.add_process(synth("syn_branch",
                        {{"pick", {{"op", "decide"}, {"next", {"good", "bad"}}}},
                         {"good", {{"op", "act"}, {"next", {"out"}}}},
                         {"bad", {{"op", "act"}, {"next", {"out"}}}},
                         {"out", {{"op", "return"}}}},
                        "pick"));
  lib.add_process(synth("syn_child", {{"x", {{"op", "act"}, {"next", {"y"}}}}, {"y", {{"op", "return"}}}}, "x"));
  lib.add_process(synth("syn_parent",
                        {{"a", {{"op", "act"}, {"next", {"b"}}}},
                         {"b", {{"op", "call:syn_child"}, {"next", {"c"}}}},
                         {"c", {{"op", "return"}}}},
                        "a"));
  lib.add_process(synth("syn_rec", {{"go", {{"op", "call:syn_rec"}, {"next", {"out"}}}}, {"out", {{"op", "return"}}}}, "go"));
  lib.add_process(synth("syn_unbound", {{"u", {{"op", "classify"}, {"next", {"out"}}}}, {"out", {{"op", "return"}}}}, "u"));
  return std::make_shared<const Library>(std::move(lib));
}

static Primitive ok_stub(double cost = 1.0) {
  return [cost](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    r.cost = cost;
    return r;
  };
}

static std::size_t count_op(const RunTrace& t, const std::string& op) {
  std::size_t n = 0;
  for (const StepRecord& r : t.steps) n += r.op == op;
  return n;
}

// ---------------------------------------------------------------------------
// b) the library, instantiation, half_split, sprt_decide, tote
// ---------------------------------------------------------------------------

static void test_library_and_instantiation(const Library& lib, const std::shared_ptr<const Library>& full) {
  CHECK(lib.processes().size() >= 142);
  CHECK(lib.find("half_split") != nullptr);
  CHECK(lib.find("nonexistent") == nullptr);
  CHECK(lib.validate().empty());
  std::size_t steps = 0, calls = 0;
  for (const Process& p : lib.processes()) {
    CHECK(lib.check(p.id).empty());
    steps += p.steps.size();
    for (const auto& [sid, s] : p.steps) calls += s.is_call();
  }
  std::printf("library: %zu processes, %zu steps, %zu calls\n", lib.processes().size(), steps, calls);
  CHECK(steps >= 840);
  CHECK(calls >= 43);
  const Process& hs = *lib.find("half_split");
  CHECK(hs.start == "chain" && hs.steps.at("narrow").next.size() == 2);

  // Instantiate every process and run it once on a universal stub environment: every primitive is
  // bound, every decision goes through the learned Dirichlet. None may error out or hit an impasse.
  Interpreter it(full, Config{}, 2026);
  for (const std::string& prim : lib.primitives())
    if (prim != "return") it.bind(prim, ok_stub(1.0));
  std::size_t done = 0, budget = 0, bad = 0, total = 0;
  for (const Process& p : lib.processes()) {
    CHECK(it.unbound(p.id).empty());
    WorkingMemory wm;
    RunOptions opt;
    opt.budget = 300;
    const RunResult r = it.run(p.id, "smoke", wm, opt);
    ++total;
    if (r.trace.outcome == Outcome::Done) ++done;
    else if (r.trace.outcome == Outcome::Budget) ++budget;
    else {
      ++bad;
      std::printf("  smoke run of %s ended %s %s\n", p.id.c_str(), outcome_name(r.trace.outcome), r.trace.note.c_str());
    }
    CHECK(max_depth_of(r.trace) <= 5);
  }
  std::printf("instantiate+run every process: %zu done, %zu budget, %zu other of %zu\n", done, budget, bad, total);
  CHECK(bad == 0);
  CHECK(done >= total * 95 / 100);
}

struct HalfSplitEnv {
  int n = 64;
  int fault = 0;
  Rng rng;
  bool random_split = false;
  int tests = 0;
};

static void bind_half_split(Interpreter& it, HalfSplitEnv& env) {
  it.bind("half_split.chain", [&env](const Step&, StepContext&, WorkingMemory& wm) {
    wm.set("lo", 0);
    wm.set("hi", env.n);
    return StepResult{};
  });
  it.bind("half_split.mid", [&env](const Step&, StepContext&, WorkingMemory& wm) {
    const int lo = static_cast<int>(wm.num("lo")), hi = static_cast<int>(wm.num("hi"));
    int m = (lo + hi - 1) / 2;  // splits [lo, hi) into halves
    if (env.random_split) m = lo + static_cast<int>(uniform01(env.rng) * (hi - 1 - lo));  // uniform on [lo, hi-2]
    wm.set("m", m);
    return StepResult{};
  });
  it.bind("half_split.test", [&env](const Step&, StepContext&, WorkingMemory& wm) {
    // a measurement after component m: bad iff the fault is at or upstream of m
    const int lo = static_cast<int>(wm.num("lo")), hi = static_cast<int>(wm.num("hi")), m = static_cast<int>(wm.num("m"));
    const bool bad = env.fault <= m;
    wm.set("hi", bad ? m + 1 : hi);
    wm.set("lo", bad ? lo : m + 1);
    ++env.tests;
    StepResult r;
    r.cost = 1.0;
    r.value = bad;
    return r;
  });
  it.bind("half_split.narrow", [](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    r.chosen_next = (wm.num("hi") - wm.num("lo") == 1) ? "repair" : "mid";
    return r;
  });
  it.bind("half_split.repair", [](const Step&, StepContext&, WorkingMemory& wm) {
    wm.set("repaired", wm.num("lo"));
    StepResult r;
    r.cost = 5.0;
    return r;
  });
  it.bind("half_split.confirm", [&env](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    r.ok = static_cast<int>(wm.num("repaired", -1)) == env.fault;
    r.cost = 1.0;
    return r;
  });
}

static void test_half_split(const std::shared_ptr<const Library>& lib, double& mean_half, double& mean_rand,
                            double& mean_probe) {
  const int seeds = 200;
  double sum_half = 0, sum_rand = 0, sum_probe = 0;
  int max_half = 0;
  bool all_found = true, all_done = true;
  for (int variant = 0; variant < 2; ++variant) {
    for (int seed = 0; seed < seeds; ++seed) {
      HalfSplitEnv env;
      env.rng.seed(1000 + static_cast<std::uint64_t>(seed));
      env.fault = static_cast<int>(uniform01(env.rng) * env.n);
      env.random_split = variant == 1;
      Interpreter it(lib, Config{}, static_cast<std::uint64_t>(seed));
      bind_half_split(it, env);
      WorkingMemory wm;
      const RunResult r = it.run("half_split", "chain64", wm);
      const int tests = static_cast<int>(count_op(r.trace, "test"));
      all_done = all_done && r.trace.outcome == Outcome::Done;
      all_found = all_found && static_cast<int>(wm.num("repaired", -1)) == env.fault && r.reward == 1.0;
      CHECK(tests == env.tests);
      if (variant == 0) {
        sum_half += tests;
        max_half = std::max(max_half, tests);
      } else {
        sum_rand += tests;
      }
    }
  }
  // naive baseline: probe components one at a time in random order until the faulty one is hit
  Rng prng(55);
  for (int seed = 0; seed < seeds; ++seed) {
    std::vector<int> order(64);
    std::iota(order.begin(), order.end(), 0);
    for (int i = 63; i > 0; --i) std::swap(order[static_cast<std::size_t>(i)], order[static_cast<std::size_t>(uniform01(prng) * (i + 1))]);
    const int fault = static_cast<int>(uniform01(prng) * 64);
    int t = 0;
    for (int c : order) {
      ++t;
      if (c == fault) break;
    }
    sum_probe += t;
  }
  mean_half = sum_half / seeds;
  mean_rand = sum_rand / seeds;
  mean_probe = sum_probe / seeds;
  std::printf("half_split n=64, %d seeds: halving %.3f tests (max %d), random split %.3f, random probing %.3f\n", seeds,
              mean_half, max_half, mean_rand, mean_probe);
  CHECK(all_done);
  CHECK(all_found);
  CHECK(mean_half <= 7.0);
  CHECK(mean_half < mean_rand);
  CHECK(mean_half < mean_probe);
}

static void test_sprt(const std::shared_ptr<const Library>& lib, double& err0, double& err1, double& n0, double& n1,
                      double& theory) {
  const double alpha = 0.05, beta = 0.05, p0 = 0.4, p1 = 0.6;
  const double A = std::log((1 - beta) / alpha), B = std::log(beta / (1 - alpha));
  const double up = std::log(p1 / p0), down = std::log((1 - p1) / (1 - p0));
  const int runs = 2000;
  Rng env(424242);
  bool truth_is_h1 = false;
  Interpreter it(lib, Config{}, 5);
  it.bind("sprt_decide.bounds", [&](const Step&, StepContext&, WorkingMemory& wm) {
    wm.set("A", A);
    wm.set("B", B);
    wm.set("L", 0.0);
    return StepResult{};
  });
  it.bind("sprt_decide.obs", [&](const Step&, StepContext&, WorkingMemory& wm) {
    const bool x = uniform01(env) < (truth_is_h1 ? p1 : p0);
    wm.set("x", x);
    StepResult r;
    r.cost = 1.0;
    r.value = x;
    return r;
  });
  it.bind("sprt_decide.llr", [&](const Step&, StepContext&, WorkingMemory& wm) {
    wm.set("L", wm.num("L") + (wm.get("x")->get<bool>() ? up : down));
    return StepResult{};
  });
  it.bind("sprt_decide.check", [&](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    r.chosen_next = wm.num("L") >= wm.num("A") ? "h1" : (wm.num("L") <= wm.num("B") ? "h0" : "obs");
    return r;
  });
  int wrong0 = 0, wrong1 = 0;
  double used0 = 0, used1 = 0;
  for (int h = 0; h < 2; ++h) {
    truth_is_h1 = h == 1;
    for (int i = 0; i < runs; ++i) {
      WorkingMemory wm;
      RunOptions opt;
      opt.budget = 2000;
      opt.reward = [&](const RunTrace& t, WorkingMemory&) {
        const bool said_h1 = t.steps.back().step == "h1";
        return said_h1 == truth_is_h1 ? 1.0 : 0.0;
      };
      const RunResult r = it.run("sprt_decide", "bernoulli", wm, opt);
      CHECK(r.trace.outcome == Outcome::Done);
      const bool said_h1 = r.trace.steps.back().step == "h1";
      (h == 0 ? used0 : used1) += static_cast<double>(count_op(r.trace, "perceive"));
      if (h == 0 && said_h1) ++wrong0;
      if (h == 1 && !said_h1) ++wrong1;
    }
  }
  err0 = static_cast<double>(wrong0) / runs;
  err1 = static_cast<double>(wrong1) / runs;
  n0 = used0 / runs;
  n1 = used1 / runs;
  // Wald's bounds are conservative: with +-delta steps the walk must reach net +-N, N = ceil(A/delta).
  // Exact error: gambler's ruin with r = q/p, start N, range 2N.
  const double delta = up;  // symmetric: down == -up
  const int N = static_cast<int>(std::ceil(A / delta - 1e-12));
  const double r = (1 - p0) / p0;
  theory = (1 - std::pow(r, N)) / (1 - std::pow(r, 2 * N));
  const double se = std::sqrt(alpha * (1 - alpha) / runs);
  std::printf("sprt_decide alpha=beta=0.05 (p0=0.4,p1=0.6), %d runs per hypothesis: type I %.4f, type II %.4f, "
              "exact %.4f; mean evidence %.1f (H0), %.1f (H1)\n", runs, err0, err1, theory, n0, n1);
  CHECK(std::fabs(down + up) < 1e-12);
  CHECK(err0 <= alpha + 3 * se);
  CHECK(err1 <= beta + 3 * se);
  CHECK_NEAR(err0, theory, 4 * std::sqrt(theory * (1 - theory) / runs));
  CHECK_NEAR(err1, theory, 4 * std::sqrt(theory * (1 - theory) / runs));
  CHECK(err0 > 0.5 * alpha && err1 > 0.5 * beta);  // near alpha / beta, not trivially zero
}

static void test_tote_voc_budget(const std::shared_ptr<const Library>& lib) {
  // TOTE closes a gap: compare, operate, compare, ... exit
  {
    Interpreter it(lib, Config{}, 3);
    double state = 0.0;
    const double goal = 10.0, tol = 0.5;
    int ops = 0;
    it.bind("tote.test", [&](const Step&, StepContext&, WorkingMemory&) {
      StepResult r;
      r.chosen_next = (goal - state) <= tol ? "exit" : "operate";
      return r;
    });
    it.bind("tote.operate", [&](const Step&, StepContext& c, WorkingMemory&) {
      state += 1.0 + 2.0 * uniform01(*c.rng);
      ++ops;
      StepResult r;
      r.cost = 1.0;
      return r;
    });
    WorkingMemory wm;
    const RunResult r = it.run("tote", "gap", wm);
    CHECK(r.trace.outcome == Outcome::Done);
    CHECK(goal - state <= tol);
    CHECK(ops >= 4 && ops <= 10);
    CHECK(count_op(r.trace, "act") == static_cast<std::size_t>(ops));
  }
  // stop on budget, then on VOC (3.7): a futile operator (gain 0, cost 1)
  {
    Interpreter it(lib, Config{}, 4);
    it.bind("tote.test", [](const Step&, StepContext&, WorkingMemory&) {
      StepResult r;
      r.chosen_next = "operate";  // the gap never closes
      return r;
    });
    it.bind("tote.operate", [](const Step&, StepContext&, WorkingMemory&) {
      StepResult r;
      r.cost = 1.0;
      r.gain = 0.0;
      return r;
    });
    WorkingMemory wm;
    RunOptions opt;
    opt.budget = 50;
    const RunResult r1 = it.run("tote", "futile", wm, opt);
    CHECK(r1.trace.outcome == Outcome::Budget);
    CHECK_NEAR(r1.trace.spent, 50.0, 1e-12);
    const RunResult r2 = it.run("tote", "futile", wm, opt);
    CHECK(r2.trace.outcome == Outcome::Voc);
    CHECK(r2.trace.spent < r1.trace.spent);
    std::printf("tote on a futile operator: run 1 stops on %s at cost %.0f, run 2 stops on %s at cost %.0f\n",
                outcome_name(r1.trace.outcome), r1.trace.spent, outcome_name(r2.trace.outcome), r2.trace.spent);
    // a worthwhile operator (gain 0.5 per unit cost 0.1) never trips the rule
    Interpreter jt(lib, Config{}, 4);
    int n = 0;
    jt.bind("tote.test", [&](const Step&, StepContext&, WorkingMemory&) {
      StepResult r;
      r.chosen_next = n >= 30 ? "exit" : "operate";
      return r;
    });
    jt.bind("tote.operate", [&](const Step&, StepContext&, WorkingMemory&) {
      ++n;
      StepResult r;
      r.cost = 0.1;
      r.gain = 0.5;
      return r;
    });
    for (int i = 0; i < 3; ++i) {
      n = 0;
      CHECK(jt.run("tote", "useful", wm).trace.outcome == Outcome::Done);
    }
  }
}

static void test_interpreter_mechanics(const std::shared_ptr<const Library>& lib) {
  // Learned branch choice (3.2): a decide step with no guard learns the branch that pays.
  {
    Interpreter it(lib, Config{}, 9);
    it.bind("act", ok_stub(0.0));
    it.bind("decide", ok_stub(0.0));  // no chosen_next: the Dirichlet chooses
    RunOptions opt;
    opt.reward = [](const RunTrace& t, WorkingMemory&) { return t.steps[1].step == "good" ? 1.0 : 0.0; };
    int good_last = 0;
    for (int i = 0; i < 400; ++i) {
      WorkingMemory wm;
      const RunResult r = it.run("syn_branch", "c", wm, opt);
      CHECK(r.trace.steps[0].how == "learned");
      if (i >= 300) good_last += r.trace.steps[1].step == "good";
    }
    const Dirichlet* d = it.branch_stats("syn_branch", "pick", "c");
    CHECK(d != nullptr && d->size() == 2);
    std::printf("learned branch: Dirichlet mean P(good) = %.3f, chose good %d%% of the last 100\n", d->mean(0), good_last);
    CHECK(d->mean(0) > 0.8);
    CHECK(good_last >= 80);
    CHECK_NEAR(d->alpha[1], 1.0, 1e-12);  // R = 0 never adds to the bad branch
  }
  // Sub-process calls pass the parent's credit to the child (3.6); child budget share (3.7)
  {
    Interpreter it(lib, Config{}, 1);
    it.bind("act", ok_stub(2.0));
    RunOptions opt;
    opt.reward = [](const RunTrace&, WorkingMemory&) { return 1.0; };
    WorkingMemory wm;
    const RunResult r = it.run("syn_parent", "c", wm, opt);
    CHECK(r.trace.outcome == Outcome::Done);
    CHECK(r.trace.steps.size() == 3 && r.trace.children.size() == 1);
    CHECK(r.trace.steps[1].child == 0);
    // parent trace a, b, c -> credits 0.64, 0.8, 1.0; the child Beta gets the credit of its call step
    CHECK_NEAR(it.step_stats("syn_parent", "a", "c").success.a, 1.64, 1e-12);
    CHECK_NEAR(it.step_stats("syn_parent", "b", "c").success.a, 1.8, 1e-12);
    CHECK_NEAR(it.step_stats("syn_parent", "c", "c").success.a, 2.0, 1e-12);
    CHECK_NEAR(it.proc_stats("syn_child", "c").success.a, 1.8, 1e-12);
    CHECK_NEAR(it.proc_stats("syn_child", "c").success.b, 1.0, 1e-12);
    CHECK_NEAR(it.step_stats("syn_child", "x", "c").success.a, 1.0 + 0.8 * 0.8, 1e-12);
    CHECK_NEAR(it.step_stats("syn_child", "y", "c").success.a, 1.0 + 0.8, 1e-12);
    CHECK_NEAR(it.proc_stats("syn_parent", "c").success.a, 2.0, 1e-12);
    CHECK_NEAR(r.trace.children[0].budget, 49.0, 1e-12);  // no cost data yet: the cap, 0.5 of the 98 left after step a
    CHECK_NEAR(r.trace.spent, 4.0, 1e-12);                // a (2) + the child's x (2); the return steps are free
    CHECK(r.trace.children[0].depth == 1);
  }
  // depth cap 5 on a self-recursive process; each nested budget is at most half the parent's remainder
  {
    Interpreter it(lib, Config{}, 1);
    WorkingMemory wm;
    const RunResult r = it.run("syn_rec", "c", wm);
    CHECK(r.trace.outcome == Outcome::Done);
    CHECK(max_depth_of(r.trace) == 5);
    const RunTrace* t = &r.trace;
    int depth = 0;
    double parent_budget = t->budget;
    while (!t->children.empty()) {
      t = &t->children[0];
      ++depth;
      CHECK(t->budget <= 0.5 * parent_budget + 1e-12);
      parent_budget = t->budget;
    }
    CHECK(depth == 5);
    CHECK(t->steps[0].note == "depth_cap" && !t->steps[0].ok);
  }
  // impasse: nothing applicable, a process judged inapplicable, a missing primitive
  {
    Interpreter it(lib, Config{}, 1);
    it.bind("act", ok_stub(1.0));
    it.bind("test", ok_stub(1.0));
    WorkingMemory wm;
    Goal none{"c", {}};
    const SolveResult s0 = it.solve(none, wm);
    CHECK(s0.impasse && s0.run.impasse());
    ProcStats& ps = it.proc_stats("syn_cheap", "c");
    ps.applicable.b = 30;  // P(applicable) = 1/32 < theta
    const RunResult rn = it.run("syn_cheap", "c", wm);
    CHECK(rn.impasse() && rn.trace.steps.empty());
    Goal g{"c", {"syn_cheap", "syn_best"}};
    for (int i = 0; i < 50; ++i) CHECK(it.solve(g, wm).chosen == "syn_best");
    Goal only_bad{"c", {"syn_cheap"}};
    CHECK(it.solve(only_bad, wm).impasse);
    const double before = it.proc_stats("syn_unbound", "c").applicable.mean();
    CHECK(it.unbound("syn_unbound") == std::vector<std::string>{"classify"});
    const RunResult ru = it.run("syn_unbound", "c", wm);
    CHECK(ru.impasse() && ru.trace.note == "no_primitive");
    CHECK(it.proc_stats("syn_unbound", "c").applicable.mean() < before);  // a straight-away IMPASSE is a negative label
    // a callback that names a branch outside `next` is an error, not a silent fallback
    it.bind("syn_branch.pick", [](const Step&, StepContext&, WorkingMemory&) {
      StepResult r;
      r.chosen_next = "nowhere";
      return r;
    });
    CHECK(it.run("syn_branch", "c", wm).trace.outcome == Outcome::Error);
  }
}

// ---------------------------------------------------------------------------
// c) selection benchmark; d) replay
// ---------------------------------------------------------------------------

struct BenchArm {
  const char* id;
  double p, cost;
};
static const BenchArm kArms[3] = {{"syn_cheap", 0.60, 1.0}, {"syn_best", 0.85, 2.0}, {"syn_sure", 0.95, 5.0}};
static const double kG = 10.0;

struct BenchResult {
  std::vector<int> chosen;
  std::vector<double> regret;       // expected (pseudo) regret per episode
  std::vector<double> p_hat;
  std::vector<int> y;
  double ece_all = 0, brier_all = 0, ece_best = 0;
  std::string traces;               // exact text of every trace
  std::string digest;               // exact text of the final learner state
};

static BenchResult run_bench(const std::shared_ptr<const Library>& lib, std::uint64_t seed, int episodes,
                             bool keep_text) {
  BenchResult out;
  Rng env(seed ^ 0x9e3779b97f4a7c15ULL);
  auto it = std::make_unique<Interpreter>(lib, Config{}, seed);
  for (int i = 0; i < 3; ++i) {
    const BenchArm a = kArms[i];
    it->bind(std::string(a.id) + ".work", [&env, a](const Step&, StepContext&, WorkingMemory&) {
      StepResult r;
      r.cost = a.cost * std::exp(0.2 * normal01(env) - 0.02);  // lognormal around the true cost
      return r;
    });
    it->bind(std::string(a.id) + ".check", [&env, a](const Step&, StepContext&, WorkingMemory&) {
      StepResult r;
      r.ok = uniform01(env) < a.p;  // the process's own check: success with the true rate
      return r;
    });
  }
  const Goal goal{"synthetic", {"syn_cheap", "syn_best", "syn_sure"}};
  double best_u = -1e9;
  for (const BenchArm& a : kArms) best_u = std::max(best_u, a.p * kG - a.cost);
  RunOptions opt;
  opt.G = kG;
  for (int e = 0; e < episodes; ++e) {
    WorkingMemory wm;
    const SolveResult s = it->solve(goal, wm, opt);
    int idx = 0;
    for (int i = 0; i < 3; ++i)
      if (s.chosen == kArms[i].id) idx = i;
    out.chosen.push_back(idx);
    out.regret.push_back(best_u - (kArms[idx].p * kG - kArms[idx].cost));
    out.p_hat.push_back(s.run.trace.p_hat);
    out.y.push_back(s.run.reward >= 0.5 ? 1 : 0);
    if (keep_text) trace_text(s.run.trace, out.traces);
  }
  const CalibrationLog& cal = it->learners().calibration;
  out.ece_all = cal.ece_all();
  out.brier_all = cal.brier_all();
  out.ece_best = cal.ece("syn_best");
  char buf[64];
  for (const BenchArm& a : kArms) {
    const ProcStats& ps = it->proc_stats(a.id, "synthetic");
    std::snprintf(buf, sizeof buf, "%a %a %a %a|", ps.success.a, ps.success.b, ps.cost.log_mean(), ps.util.u);
    out.digest += buf;
  }
  return out;
}

static double sum_range(const std::vector<double>& v, std::size_t a, std::size_t b) {
  double s = 0;
  for (std::size_t i = a; i < b; ++i) s += v[i];
  return s;
}

static void test_selection_benchmark(const std::shared_ptr<const Library>& lib, BenchResult& R) {
  const int N = 2000;
  R = run_bench(lib, 20261027, N, false);
  const double r1 = sum_range(R.regret, 0, 1000), r2 = sum_range(R.regret, 1000, 2000);
  const double r_first500 = sum_range(R.regret, 0, 500), r_last500 = sum_range(R.regret, 1500, 2000);
  int best_late = 0, best_all = 0;
  int counts[3] = {0, 0, 0};
  for (int e = 0; e < N; ++e) {
    best_all += R.chosen[static_cast<std::size_t>(e)] == 1;
    if (e >= 1000) best_late += R.chosen[static_cast<std::size_t>(e)] == 1;
    ++counts[R.chosen[static_cast<std::size_t>(e)]];
  }
  const double share_late = best_late / 1000.0;
  std::printf("selection (G=10; arms p/cost: cheap 0.60/1, best 0.85/2, sure 0.95/5; U* = 6.5):\n");
  std::printf("  cumulative regret at 250/500/1000/1500/2000: %.1f / %.1f / %.1f / %.1f / %.1f\n",
              sum_range(R.regret, 0, 250), r_first500, r1, r1 + sum_range(R.regret, 1000, 1500), r1 + r2);
  std::printf("  regret first 1000 = %.1f, second 1000 = %.1f; first 500 = %.1f, last 500 = %.1f\n", r1, r2, r_first500, r_last500);
  std::printf("  best chosen %.1f%% overall, %.1f%% in the second 1000 (counts cheap/best/sure = %d/%d/%d)\n",
              100.0 * best_all / N, 100.0 * share_late, counts[0], counts[1], counts[2]);
  std::printf("  calibration of p-hat: ECE %.4f, Brier %.4f (best arm ECE %.4f)\n", R.ece_all, R.brier_all, R.ece_best);
  CHECK(r2 < r1);                       // regret grows sublinearly
  CHECK(r_last500 < 0.8 * r_first500);
  CHECK(share_late > 0.80);
  CHECK(R.ece_all < 0.05);
  CHECK(R.ece_best < 0.05);
  // the streaming ECE agrees with eval/ecs.hpp on the stored (p-hat, outcome) pairs
  CHECK_NEAR(R.ece_all, cypha::eval::expected_calibration_error(R.p_hat, R.y, 10), 1e-9);
  CHECK_NEAR(R.brier_all, cypha::eval::brier_score(R.p_hat, R.y), 1e-9);
  // ECE over just the last 1000 predictions
  const std::vector<double> ph(R.p_hat.begin() + 1000, R.p_hat.end());
  const std::vector<int> yy(R.y.begin() + 1000, R.y.end());
  const double ece_late = cypha::eval::expected_calibration_error(ph, yy, 10);
  std::printf("  ECE over the last 1000 predictions: %.4f\n", ece_late);
  CHECK(ece_late < 0.05);
}

static void test_replay(const std::shared_ptr<const Library>& lib) {
  const BenchResult a = run_bench(lib, 77, 300, true);
  const BenchResult b = run_bench(lib, 77, 300, true);
  const BenchResult c = run_bench(lib, 78, 300, true);
  CHECK(!a.traces.empty());
  CHECK(a.traces == b.traces);   // bit-identical (hex floats)
  CHECK(a.digest == b.digest);   // and the learned state
  CHECK(a.traces != c.traces);   // a different seed differs
  std::printf("replay: seed 77 twice -> %zu bytes of trace text, hash %016llx, identical; seed 78 differs\n",
              a.traces.size(), static_cast<unsigned long long>(fnv1a(a.traces)));

  // a trace with learned branches and a nested call replays identically too
  auto run_mixed = [&](std::uint64_t seed) {
    Interpreter it(lib, Config{}, seed);
    it.bind("act", [](const Step&, StepContext& c, WorkingMemory&) {
      StepResult r;
      r.cost = 1.0 + uniform01(*c.rng);
      return r;
    });
    it.bind("decide", ok_stub(0.0));
    RunOptions opt;
    opt.reward = [](const RunTrace& t, WorkingMemory&) { return t.steps[1].step == "good" ? 1.0 : 0.0; };
    std::string text;
    for (int i = 0; i < 60; ++i) {
      WorkingMemory wm;
      trace_text(it.run("syn_branch", "c", wm, opt).trace, text);
      trace_text(it.run("syn_parent", "c", wm, opt).trace, text);
    }
    return text;
  };
  CHECK(run_mixed(5) == run_mixed(5));
  CHECK(run_mixed(5) != run_mixed(6));

  // the sink receives each learned trace
  Interpreter it(lib, Config{}, 1);
  it.bind("act", ok_stub(1.0));
  int seen = 0;
  it.set_trace_sink([&](const RunTrace& t) { seen += t.reward >= 0.0 && !t.process.empty(); });
  WorkingMemory wm;
  it.run("syn_parent", "c", wm);
  CHECK(seen == 1);
}

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "../docs/thought/processes.json";
  std::shared_ptr<const Library> full;
  Library base;
  try {
    base = Library::load(path);
    full = make_library(base);
  } catch (const std::exception& e) {
    std::printf("FAIL cannot load %s: %s\n", path.c_str(), e.what());
    return 1;
  }

  test_random_draws();
  test_beta();
  test_dirichlet();
  test_welford();
  test_thompson_actr_td_voc();
  test_wilson_and_calibration();
  std::printf("learners: known-answer checks done (%d checks so far)\n", g_checks);

  test_library_and_instantiation(base, full);
  double mh, mr, mp;
  test_half_split(full, mh, mr, mp);
  double e0, e1, n0, n1, th;
  test_sprt(full, e0, e1, n0, n1, th);
  test_tote_voc_budget(full);
  test_interpreter_mechanics(full);
  BenchResult bench;
  test_selection_benchmark(full, bench);
  test_replay(full);

  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed ? 1 : 0;
}
