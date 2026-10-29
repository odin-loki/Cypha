// K5: the first thought processes run end to end on benchmarks with known answers, and the DIF applicability
// comparison (docs/THOUGHT.md sections 3.1, 10, 12; docs/reports/K4_K5_THOUGHT_BINDINGS.md).
//
//   thought_k5 qa  PROCESSES_JSON ROWS_TSV [--read-test]   K5a: confidence_monitor in dual_process on CyphaLM's rows
//   thought_k5 chains PROCESSES_JSON                       K5b: half_split, differential_diagnosis, random probing
//   thought_k5 hyps PROCESSES_JSON                         K5c: ach and sprt_decide
//   thought_k5 monitor PROCESSES_JSON                      K5d: change_detection (CUSUM) and spc_control (Shewhart)
//   thought_k5 applicability PROCESSES_JSON                3.1: Beta-class, Beta-pooled and the Cypha DIF classifier
//
// Generated cases use dev seeds 1-100 for choices and test seeds 1001-1200 (the applicability streams: 1001-1003).
// Rule 10: nothing here answers a question except CyphaLM's own recorded answers in K5a.
//
// Author: Odin Loch
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "cypha/thought/bindings_cypha.hpp"

using namespace cypha::thought;
namespace bd = cypha::thought::bind;
namespace ev = cypha::eval;

static Json synth(const std::string& id, const Json& steps, const std::string& start) {
  return Json{{"id", id}, {"name", id}, {"cat", "diagnose"}, {"ref", "K5 baseline"}, {"when", "a fault along a chain"},
              {"start", start}, {"steps", steps}, {"stop", "fault found"}, {"math", "n/a"},
              {"learn", Json::array()}, {"feedback", "cost to isolate"}};
}

/// The library plus K5's baseline process (random probing, one component at a time).
static std::shared_ptr<const Library> load_library(const std::string& path) {
  Library lib = Library::load(path);
  lib.add_process(synth("random_probe",
                        {{"pick", {{"op", "select"}, {"next", {"probe"}}}},
                         {"probe", {{"op", "test"}, {"next", {"done"}}}},
                         {"done", {{"op", "decide"}, {"next", {"out", "pick"}}}},
                         {"out", {{"op", "return"}}}},
                        "pick"));
  return std::make_shared<const Library>(std::move(lib));
}

static double mean(const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }

static std::pair<double, double> boot_ci(const std::vector<double>& v, std::uint64_t seed = 20261029, int B = 2000) {
  if (v.empty()) return {0, 0};
  Rng r(seed);
  std::vector<double> m(B);
  for (int b = 0; b < B; ++b) {
    double s = 0;
    for (std::size_t i = 0; i < v.size(); ++i) s += v[static_cast<std::size_t>(uniform01(r) * v.size()) % v.size()];
    m[static_cast<std::size_t>(b)] = s / v.size();
  }
  std::sort(m.begin(), m.end());
  return {m[static_cast<std::size_t>(0.025 * B)], m[static_cast<std::size_t>(0.975 * B)]};
}

// ===========================================================================
// Environments
// ===========================================================================

struct Chain {
  int n = 64;
  std::vector<double> prior;  // fault probability per component
  std::vector<double> cost;   // cost of the split test after component j (or of probing component j)
  int fault = 0;
};

static Chain make_chain(std::uint64_t seed, bool skewed) {
  Rng r(seed * 7919 + (skewed ? 17 : 3));
  Chain c;
  c.prior.assign(c.n, 1.0 / c.n);
  c.cost.assign(c.n, 1.0);
  if (skewed) {
    std::vector<int> perm(c.n);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = c.n - 1; i > 0; --i) std::swap(perm[i], perm[static_cast<std::size_t>(uniform01(r) * (i + 1)) % (i + 1)]);
    double z = 0;
    for (int k = 0; k < c.n; ++k) z += std::pow(k + 1.0, -1.2);
    for (int k = 0; k < c.n; ++k) c.prior[perm[k]] = std::pow(k + 1.0, -1.2) / z;
    for (int j = 0; j < c.n; ++j) c.cost[j] = 1.0 + std::floor(uniform01(r) * 5.0);  // 1..5
  }
  double u = uniform01(r), acc = 0;
  c.fault = c.n - 1;
  for (int k = 0; k < c.n; ++k) {
    acc += c.prior[k];
    if (u <= acc) {
      c.fault = k;
      break;
    }
  }
  return c;
}

struct AchCase {
  int k = 4;
  int truth = 0;
  std::vector<std::vector<int>> M;  // M[item][hyp] in {+1 consistent, 0 neutral, -1 inconsistent}
};

static AchCase make_ach(std::uint64_t seed, int informative = 12, int planted = 4) {
  Rng r(seed * 104729 + 5);
  AchCase c;
  c.truth = static_cast<int>(uniform01(r) * c.k) % c.k;
  auto draw = [&](bool is_true) {
    const double u = uniform01(r);
    if (is_true) return u < 0.6 ? 1 : (u < 0.9 ? 0 : -1);
    return u < 0.45 ? 1 : (u < 0.65 ? 0 : -1);
  };
  for (int i = 0; i < informative; ++i) {
    std::vector<int> row(c.k);
    for (int h = 0; h < c.k; ++h) row[h] = draw(h == c.truth);
    c.M.push_back(row);
  }
  for (int i = 0; i < planted; ++i) c.M.push_back(std::vector<int>(c.k, 1));  // consistent with all: no diagnosticity
  for (std::size_t i = c.M.size() - 1; i > 0; --i) std::swap(c.M[i], c.M[static_cast<std::size_t>(uniform01(r) * (i + 1)) % (i + 1)]);
  return c;
}

struct Stream {  // Bernoulli evidence (sprt) or Gaussian monitoring
  bool h1 = false;
  double p0 = 0.4, p1 = 0.6;
  double shift = 0.0;   // mean shift (sigma units) from t = shift_at
  int shift_at = 0;
  std::uint64_t seed = 1;
  Rng rng;
  int t = 0;
  double next_bernoulli() {
    ++t;
    return uniform01(rng) < (h1 ? p1 : p0) ? 1.0 : 0.0;
  }
  double next_gauss() {
    const double x = normal01(rng) + (t >= shift_at ? shift : 0.0);
    ++t;
    return x;
  }
};

// ===========================================================================
// Bindings of K5's process steps (state in a Task the caller points to)
// ===========================================================================

enum class TT { ChainU, ChainS, Binary, Hyp, Monitor, Planning };
static const char* tt_name(TT t) {
  switch (t) {
    case TT::ChainU: return "chain_uniform";
    case TT::ChainS: return "chain_skewed";
    case TT::Binary: return "binary_stream";
    case TT::Hyp: return "hypotheses";
    case TT::Monitor: return "monitor_stream";
    case TT::Planning: return "planning";
  }
  return "?";
}

struct Task {
  TT type = TT::ChainU;
  std::vector<double> x;  // situation features
  Chain chain;
  AchCase ach;
  Stream stream;
  // run state
  int lo = 0, hi = 0, pick = -1, tests = 0, alarm_t = -1;
  double spent = 0.0, llr = 0.0, A = 0.0, B = 0.0;
  std::vector<int> probed;
  int decision = -1;  // sprt: 1 = H1, 0 = H0
  int ach_pick = -1;
  bool found = false;
  // monitoring
  double cusum_h = 5.0, shewhart_L = 2.782;
  int window = 50;
  bool alarm = false;
  void reset_run() {
    lo = 0;
    hi = chain.n - 1;
    pick = -1;
    tests = 0;
    alarm_t = -1;
    spent = 0;
    llr = 0;
    probed.clear();
    decision = -1;
    ach_pick = -1;
    found = false;
    alarm = false;
    stream.rng.seed(stream.seed);
    stream.t = 0;
  }
};

static bool is_chain(const Task& t) { return t.type == TT::ChainU || t.type == TT::ChainS; }

static StepResult impasse_r() {
  StepResult r;
  r.ok = false;
  r.impasse = true;
  r.cost = 1.0;
  return r;
}

static double mass(const Chain& c, int a, int b) {
  double s = 0;
  for (int k = a; k <= b; ++k) s += c.prior[k];
  return s;
}

static double hb(double p) { return (p <= 0 || p >= 1) ? 0.0 : -(p * std::log2(p) + (1 - p) * std::log2(1 - p)); }

/// Bind K5's step callbacks for every process, against the task `cur` points to. Generic ops come from bd::bind_all.
static void bind_k5(Interpreter& it, bd::Toolkit& tk, Task*& cur) {
  bd::bind_all(it, tk);
  // ---- half_split
  it.bind("half_split.chain", [&cur](const Step&, StepContext&, WorkingMemory&) {
    if (!is_chain(*cur)) return impasse_r();
    StepResult r;
    cur->lo = 0;
    cur->hi = cur->chain.n - 1;
    return r;
  });
  it.bind("half_split.mid", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    const Chain& c = cur->chain;
    const double tot = mass(c, cur->lo, cur->hi);
    double best = 1e300, acc = 0;
    int j = cur->lo;
    for (int k = cur->lo; k < cur->hi; ++k) {
      acc += c.prior[k];
      const double d = std::fabs(acc - (tot - acc));
      if (d < best - 1e-15) {
        best = d;
        j = k;
      }
    }
    cur->pick = j;
    return r;
  });
  it.bind("half_split.test", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    const int j = cur->pick;
    r.cost = cur->chain.cost[j];
    cur->spent += r.cost;
    ++cur->tests;
    if (cur->chain.fault <= j) cur->hi = j;
    else cur->lo = j + 1;
    return r;
  });
  it.bind("half_split.narrow", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    r.chosen_next = cur->lo == cur->hi ? std::string("repair") : std::string("mid");
    return r;
  });
  it.bind("half_split.repair", [](const Step&, StepContext&, WorkingMemory&) { return StepResult{}; });
  it.bind("half_split.confirm", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    cur->found = cur->lo == cur->chain.fault;
    r.ok = cur->found;
    return r;
  });
  // ---- differential_diagnosis: the test with the most expected information per unit cost
  it.bind("differential_diagnosis.findings", [&cur](const Step&, StepContext&, WorkingMemory&) {
    if (!is_chain(*cur)) return impasse_r();
    cur->lo = 0;
    cur->hi = cur->chain.n - 1;
    return StepResult{};
  });
  it.bind("differential_diagnosis.list", [](const Step&, StepContext&, WorkingMemory&) { return StepResult{}; });
  it.bind("differential_diagnosis.testpick", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    const Chain& c = cur->chain;
    const double tot = mass(c, cur->lo, cur->hi);
    double best = -1, acc = 0;
    int j = cur->lo;
    for (int k = cur->lo; k < cur->hi; ++k) {
      acc += c.prior[k];
      const double v = hb(acc / tot) / c.cost[k];
      if (v > best + 1e-15) {
        best = v;
        j = k;
      }
    }
    cur->pick = j;
    return r;
  });
  it.bind("differential_diagnosis.result", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    const int j = cur->pick;
    r.cost = cur->chain.cost[j];
    cur->spent += r.cost;
    ++cur->tests;
    if (cur->chain.fault <= j) cur->hi = j;
    else cur->lo = j + 1;
    return r;
  });
  it.bind("differential_diagnosis.threshold", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    r.chosen_next = cur->lo == cur->hi ? std::string("act") : std::string("testpick");
    if (cur->lo == cur->hi) cur->found = cur->lo == cur->chain.fault;
    return r;
  });
  // ---- random_probe: one component at a time, in random order
  it.bind("random_probe.pick", [&cur](const Step&, StepContext& c, WorkingMemory&) {
    if (!is_chain(*cur)) return impasse_r();
    StepResult r;
    std::vector<int> left;
    for (int k = 0; k < cur->chain.n; ++k)
      if (std::find(cur->probed.begin(), cur->probed.end(), k) == cur->probed.end()) left.push_back(k);
    cur->pick = left[static_cast<std::size_t>(uniform01(*c.rng) * left.size()) % left.size()];
    return r;
  });
  it.bind("random_probe.probe", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    r.cost = cur->chain.cost[cur->pick];
    cur->spent += r.cost;
    ++cur->tests;
    cur->probed.push_back(cur->pick);
    cur->found = cur->pick == cur->chain.fault;
    return r;
  });
  it.bind("random_probe.done", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    r.chosen_next = cur->found ? std::string("out") : std::string("pick");
    return r;
  });
  // ---- sprt_decide: Wald's test on a Bernoulli stream
  it.bind("sprt_decide.bounds", [&cur](const Step&, StepContext&, WorkingMemory&) {
    if (cur->type != TT::Binary) return impasse_r();
    StepResult r;
    const double a = 0.05, b = 0.05;
    cur->A = std::log((1 - b) / a);
    cur->B = std::log(b / (1 - a));
    cur->llr = 0;
    return r;
  });
  it.bind("sprt_decide.obs", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    r.cost = 1.0;
    wm.set("obs", cur->stream.next_bernoulli());
    return r;
  });
  it.bind("sprt_decide.llr", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    const double x = wm.num("obs"), p0 = cur->stream.p0, p1 = cur->stream.p1;
    cur->llr += x > 0.5 ? std::log(p1 / p0) : std::log((1 - p1) / (1 - p0));
    return r;
  });
  it.bind("sprt_decide.check", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    if (cur->llr >= cur->A) {
      cur->decision = 1;
      r.chosen_next = std::string("h1");
    } else if (cur->llr <= cur->B) {
      cur->decision = 0;
      r.chosen_next = std::string("h0");
    } else {
      r.chosen_next = std::string("obs");
    }
    return r;
  });
  // ---- ach: analysis of competing hypotheses (Heuer): drop non-diagnostic evidence, least inconsistency wins
  it.bind("ach.hyps", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    if (cur->type != TT::Hyp) return impasse_r();
    wm.set("ach_k", cur->ach.k);
    return StepResult{};
  });
  it.bind("ach.evidence", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    wm.set("ach_items", static_cast<int>(cur->ach.M.size()));
    return r;
  });
  it.bind("ach.matrix", [](const Step&, StepContext&, WorkingMemory&) { return StepResult{}; });
  it.bind("ach.diagnostic", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    Json keep = Json::array();
    for (std::size_t i = 0; i < cur->ach.M.size(); ++i) {
      const auto& row = cur->ach.M[i];
      if (*std::max_element(row.begin(), row.end()) != *std::min_element(row.begin(), row.end())) keep.push_back(i);
    }
    wm.set("ach_keep", keep);
    return r;
  });
  it.bind("ach.refine", [](const Step&, StepContext&, WorkingMemory&) { return StepResult{}; });
  it.bind("ach.score", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    const AchCase& c = cur->ach;
    std::vector<int> inc(c.k, 0), con(c.k, 0);
    for (const Json& i : *wm.get("ach_keep"))
      for (int h = 0; h < c.k; ++h) {
        inc[h] += c.M[i.get<std::size_t>()][h] < 0;
        con[h] += c.M[i.get<std::size_t>()][h] > 0;
      }
    int best = 0;
    for (int h = 1; h < c.k; ++h)
      if (inc[h] < inc[best] || (inc[h] == inc[best] && con[h] > con[best])) best = h;
    cur->ach_pick = best;
    r.value = best;
    return r;
  });
  it.bind("ach.sensitive", [](const Step&, StepContext&, WorkingMemory&) { return StepResult{}; });
  it.bind("ach.report", [&tk, &cur](const Step& s, StepContext& c, WorkingMemory& wm) {
    wm.set("report", Json{{"top", cur->ach_pick}});
    return bd::do_communicate(tk, s, c, wm);
  });
  // ---- change_detection: CUSUM through the monitor binding
  it.bind("change_detection.obs", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    if (cur->type != TT::Monitor) return impasse_r();
    StepResult r;
    r.cost = 1.0;
    wm.set("obs", cur->stream.next_gauss());
    return r;
  });
  it.bind("change_detection.cusum", [&tk, &cur](const Step& s, StepContext& c, WorkingMemory& wm) {
    if (!wm.has("cusum")) wm.set("cusum", Json{{"S", 0.0}, {"mu0", 0.0}, {"sigma", 1.0}, {"k", 0.5}, {"h", cur->cusum_h}});
    return bd::do_monitor(tk, s, c, wm);
  });
  it.bind("change_detection.alarm", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    const bool a = wm.get("alarm")->get<bool>();
    if (a) {
      cur->alarm = true;
      cur->alarm_t = cur->stream.t;
      r.chosen_next = std::string("investigate");
    } else if (cur->stream.t >= cur->window) {  // the watch window is over: no alarm
      r.done = true;
    } else {
      r.chosen_next = std::string("obs");
    }
    return r;
  });
  it.bind("change_detection.reset", [](const Step&, StepContext&, WorkingMemory& wm) {
    wm.erase("cusum");
    return StepResult{};
  });
  // ---- spc_control: a Shewhart individuals rule (one-sided, x > L), over the watch window
  it.bind("spc_control.sample", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    if (cur->type != TT::Monitor) return impasse_r();
    StepResult r;
    // read observations until one breaks the limit or the window ends (the subgroup loop of a running chart)
    double x = 0;
    bool brk = false;
    while (cur->stream.t < cur->window) {
      x = cur->stream.next_gauss();
      r.cost += 1.0;
      if (x > cur->shewhart_L) {
        brk = true;
        break;
      }
    }
    wm.set("spc_break", brk);
    return r;
  });
  it.bind("spc_control.stat", [](const Step&, StepContext&, WorkingMemory&) { return StepResult{}; });
  it.bind("spc_control.rules", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    const bool brk = wm.get("spc_break")->get<bool>();
    if (brk) {
      cur->alarm = true;
      cur->alarm_t = cur->stream.t;
    }
    r.chosen_next = brk ? std::string("investigate") : std::string("continue");
    return r;
  });
}

// ===========================================================================
// K5b: chains
// ===========================================================================

static RunResult run_on(Interpreter& it, const std::string& pid, Task& t, Task*& cur, const std::string& cls,
                        RunOptions opt = {}) {
  t.reset_run();
  cur = &t;
  WorkingMemory wm;
  Json sit = Json::array();
  for (double v : t.x) sit.push_back(v);
  wm.set("situation", sit);
  return it.run(pid, cls, wm, opt);
}

static int cmd_chains(const std::string& lib_path) {
  auto lib = load_library(lib_path);
  bd::Toolkit tk;
  Task* cur = nullptr;
  Config cfg;
  cfg.max_steps = 100000;
  std::printf("## K5b: planted-fault chains of 64 (test seeds 1001-1200)\n\n");
  std::printf("| chains | process | tests | cost | found | cost 95%% CI |\n|---|---|---:|---:|---:|---|\n");
  std::map<std::string, std::vector<double>> cost_skew;
  for (bool skewed : {false, true}) {
    for (const std::string pid : {"half_split", "differential_diagnosis", "random_probe"}) {
      Interpreter it(lib, cfg, 41);
      bind_k5(it, tk, cur);
      std::vector<double> tests, costs, found;
      for (std::uint64_t s = 1001; s <= 1200; ++s) {
        Task t;
        t.type = skewed ? TT::ChainS : TT::ChainU;
        t.chain = make_chain(s, skewed);
        RunOptions opt;
        opt.budget = 1e9;
        const RunResult r = run_on(it, pid, t, cur, tt_name(t.type), opt);
        (void)r;
        tests.push_back(t.tests);
        costs.push_back(t.spent);
        found.push_back(t.found ? 1.0 : 0.0);
      }
      if (skewed) cost_skew[pid] = costs;
      const auto ci = boot_ci(costs);
      std::printf("| %s | %s | %.3f | %.2f | %.3f | [%.2f, %.2f] |\n", skewed ? "skewed" : "uniform", pid.c_str(),
                  mean(tests), mean(costs), mean(found), ci.first, ci.second);
    }
  }
  {
    std::vector<double> d(cost_skew["half_split"].size());
    for (std::size_t i = 0; i < d.size(); ++i)
      d[i] = (cost_skew["half_split"][i] - cost_skew["differential_diagnosis"][i]);
    const auto ci = boot_ci(d);
    std::printf("\nSkewed: half_split minus differential_diagnosis cost %.2f [%.2f, %.2f]; relative saving %.3f\n",
                mean(d), ci.first, ci.second, mean(d) / mean(cost_skew["half_split"]));
  }
  // the standard learners: Thompson selection among the three on skewed chains, 2,000 episodes, G = 100
  {
    Interpreter it(lib, cfg, 77);
    bind_k5(it, tk, cur);
    const Goal g{"chain_skewed", {"half_split", "differential_diagnosis", "random_probe"}};
    std::map<std::string, int> picks_second;
    std::vector<double> regret;
    double cum = 0;
    const int E = 2000;
    for (int e = 0; e < E; ++e) {
      Task t;
      t.type = TT::ChainS;
      t.chain = make_chain(5001 + static_cast<std::uint64_t>(e), true);
      t.reset_run();
      cur = &t;
      WorkingMemory wm;
      RunOptions opt;
      opt.budget = 1e9;
      opt.G = 100;
      opt.reward = [&t](const RunTrace&, WorkingMemory&) { return t.found ? std::max(0.0, 1.0 - t.spent / 100.0) : 0.0; };
      const SolveResult s = it.solve(g, wm, opt);
      if (e >= E / 2) picks_second[s.chosen]++;
      // regret against differential_diagnosis on the same chain
      Task o;
      o.type = TT::ChainS;
      o.chain = t.chain;
      Interpreter oi(lib, cfg, 1);
      bind_k5(oi, tk, cur);
      RunOptions oo;
      oo.budget = 1e9;
      run_on(oi, "differential_diagnosis", o, cur, "x", oo);
      cum += t.spent - o.spent;
      if ((e + 1) % 500 == 0) regret.push_back(cum);
    }
    std::printf("\nThompson on skewed chains (2,000 episodes, G 100): second half picks:");
    for (const auto& [k, v] : picks_second) std::printf(" %s %d (%.3f)", k.c_str(), v, v / 1000.0);
    std::printf("\nCumulative extra cost against differential_diagnosis at 500/1000/1500/2000: %.0f %.0f %.0f %.0f\n",
                regret[0], regret[1], regret[2], regret[3]);
  }
  return 0;
}

// ===========================================================================
// K5c: ach and sprt
// ===========================================================================

static int cmd_hyps(const std::string& lib_path) {
  auto lib = load_library(lib_path);
  bd::Toolkit tk;
  Task* cur = nullptr;
  Config cfg;
  cfg.max_steps = 100000;
  Interpreter it(lib, cfg, 3);
  bind_k5(it, tk, cur);
  std::vector<double> ach_ok, base_ok, used;
  for (std::uint64_t s = 1001; s <= 1200; ++s) {
    Task t;
    t.type = TT::Hyp;
    t.ach = make_ach(s);
    WorkingMemory wm;
    t.reset_run();
    cur = &t;
    const RunResult r = it.run("ach", "hyps", wm);
    ach_ok.push_back(t.ach_pick == t.ach.truth ? 1.0 : 0.0);
    used.push_back(wm.get("ach_keep") ? static_cast<double>(wm.get("ach_keep")->size()) : 0.0);
    (void)r;
    // baseline: the hypothesis with the most consistent items, all evidence counted
    std::vector<int> con(t.ach.k, 0);
    for (const auto& row : t.ach.M)
      for (int h = 0; h < t.ach.k; ++h) con[h] += row[h] > 0;
    const int b = static_cast<int>(std::max_element(con.begin(), con.end()) - con.begin());
    base_ok.push_back(b == t.ach.truth ? 1.0 : 0.0);
  }
  auto w = [](const std::vector<double>& v) {
    const double k = std::accumulate(v.begin(), v.end(), 0.0);
    return std::make_pair(ev::wilson_lower(k, v.size()), ev::wilson_upper(k, v.size()));
  };
  const auto wa = w(ach_ok), wb = w(base_ok);
  std::vector<double> d(ach_ok.size());
  for (std::size_t i = 0; i < d.size(); ++i) d[i] = ach_ok[i] - base_ok[i];
  const auto dc = boot_ci(d);
  std::printf("## K5c: 200 competing-hypothesis cases (4 hypotheses, 12 informative + 4 planted items)\n\n");
  std::printf("ach accuracy %.3f [%.3f, %.3f], evidence used %.2f of 16\n", mean(ach_ok), wa.first, wa.second, mean(used));
  std::printf("counting baseline %.3f [%.3f, %.3f]; ach minus baseline %.3f [%.3f, %.3f]\n", mean(base_ok), wb.first,
              wb.second, mean(d), dc.first, dc.second);
  // sprt_decide, 2,000 runs per hypothesis
  int e0 = 0, e1 = 0;
  double n0 = 0, n1 = 0;
  for (int h = 0; h < 2; ++h)
    for (int i = 0; i < 2000; ++i) {
      Task t;
      t.type = TT::Binary;
      t.stream.h1 = h == 1;
      t.stream.seed = 900000 + static_cast<std::uint64_t>(h) * 100000 + static_cast<std::uint64_t>(i);
      RunOptions opt;
      opt.budget = 1e9;
      run_on(it, "sprt_decide", t, cur, "binary", opt);
      if (h == 0) {
        e0 += t.decision == 1;
        n0 += t.stream.t;
      } else {
        e1 += t.decision == 0;
        n1 += t.stream.t;
      }
    }
  std::printf("sprt_decide (p0 0.4, p1 0.6, alpha = beta = 0.05): type I %.4f, type II %.4f; mean evidence %.1f / %.1f\n",
              e0 / 2000.0, e1 / 2000.0, n0 / 2000.0, n1 / 2000.0);
  return 0;
}

// ===========================================================================
// K5d: monitoring
// ===========================================================================

/// Run length of a one-sided CUSUM (k 0.5) and of a Shewhart individuals rule (x > L) on N(shift, 1) from t = 0.
static double cusum_rl(std::uint64_t seed, double shift, double h, int cap = 100000) {
  Rng r(seed);
  double S = 0;
  for (int t = 1; t <= cap; ++t) {
    S = std::max(0.0, S + normal01(r) + shift - 0.5);
    if (S > h) return t;
  }
  return cap;
}

static int cmd_monitor(const std::string& lib_path) {
  // dev seeds 1-100: choose h so the in-control run length is about 370 (each seed: 20 streams)
  double best_h = 5, best_gap = 1e9;
  for (double h = 3.0; h <= 6.0 + 1e-9; h += 0.05) {
    std::vector<double> rl;
    for (std::uint64_t s = 1; s <= 100; ++s)
      for (int k = 0; k < 20; ++k) rl.push_back(cusum_rl(s * 1000 + static_cast<std::uint64_t>(k), 0.0, h));
    const double g = std::fabs(mean(rl) - 370.0);
    if (g < best_gap) {
      best_gap = g;
      best_h = h;
    }
  }
  const double L = 2.7822;  // P(Z > L) = 1 / 370
  auto lib = load_library(lib_path);
  bd::Toolkit tk;
  Task* cur = nullptr;
  Config cfg;
  cfg.max_steps = 10000000;
  std::printf("## K5d: CUSUM (h %.2f, chosen on dev for an in-control run length of 370) and Shewhart (L %.4f)\n\n",
              best_h, L);
  std::printf("| shift | process | mean run length | 95%% CI | median |\n|---|---|---:|---|---:|\n");
  for (double shift : {0.0, 1.0, 3.0}) {
    for (const std::string pid : {"change_detection", "spc_control"}) {
      Interpreter it(lib, cfg, 5);
      bind_k5(it, tk, cur);
      std::vector<double> rl;
      for (std::uint64_t s = 1001; s <= 1200; ++s) {
        Task t;
        t.type = TT::Monitor;
        t.stream.shift = shift;
        t.stream.shift_at = 0;
        t.stream.seed = s * 31 + 7;
        t.cusum_h = best_h;
        t.shewhart_L = L;
        t.window = 100000;
        RunOptions opt;
        opt.budget = 1e12;
        run_on(it, pid, t, cur, "monitor", opt);
        rl.push_back(t.alarm ? t.alarm_t : t.window);
      }
      std::vector<double> sorted = rl;
      std::sort(sorted.begin(), sorted.end());
      const auto ci = boot_ci(rl);
      std::printf("| %.0f sigma | %s | %.2f | [%.2f, %.2f] | %.0f |\n", shift, pid.c_str(), mean(rl), ci.first, ci.second,
                  sorted[sorted.size() / 2]);
    }
  }
  return 0;
}

// ===========================================================================
// K5a: confidence_monitor on CyphaLM's recorded T-QA answers
// ===========================================================================

struct Row {
  std::string set, form, lm, answer;
  int row = 0, in_slice = 0, grade = 1;  // grade: 0 right, 1 abstained (its own message), 2 wrong
  double f[5] = {0, 0, 0, 0, 0};          // mean_lp, min_lp, -entropy0, ig_n0, question_lp
};

static const char* kFeat[5] = {"mean_lp", "min_lp", "neg_entropy0", "ig_n0", "question_lp"};

static std::vector<Row> read_rows(const std::string& path) {
  std::vector<Row> v;
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::vector<std::string> c;
    std::stringstream ss(line);
    std::string x;
    while (std::getline(ss, x, '\t')) c.push_back(x);
    if (c.size() == 12) c.push_back(std::string());  // an empty answer: getline drops the last empty field
    if (c.size() < 13) continue;
    Row r;
    r.set = c[0];
    r.row = std::stoi(c[1]);
    r.in_slice = std::stoi(c[2]);
    r.form = c[3];
    r.lm = c[4];
    r.grade = std::stoi(c[5]);
    r.f[0] = std::stod(c[6]);
    r.f[1] = std::stod(c[7]);
    r.f[2] = -std::stod(c[8]);
    r.f[3] = std::stod(c[9]);
    r.f[4] = std::stod(c[10]);
    r.answer = c[12];
    v.push_back(r);
  }
  return v;
}

struct Gate {
  int f = -1;  // -1: answer nothing
  double tau = 0;
  ev::IsotonicCalibrator cal;
};

/// A0's rule, through the calibrator: among (feature, tau) with wrong / n <= 2% on dev, the most right; ties to fewer
/// wrong, then fewer answered; no right answer anywhere: answer nothing.
static Gate choose_gate(const std::vector<const Row*>& dev) {
  Gate best;
  int br = 0, bw = 0, ba = 0;
  const double n = static_cast<double>(dev.size());
  for (int f = 0; f < 5; ++f) {
    std::vector<double> p;
    std::vector<int> y;
    for (const Row* r : dev)
      if (r->grade != 1) {
        p.push_back(r->f[f]);
        y.push_back(r->grade == 0);
      }
    if (p.empty()) continue;
    ev::IsotonicCalibrator cal;
    cal.fit(p, y);
    std::vector<double> taus;
    for (double v : p) taus.push_back(cal.predict(v));
    std::sort(taus.begin(), taus.end());
    taus.erase(std::unique(taus.begin(), taus.end()), taus.end());
    for (double tau : taus) {
      int right = 0, wrong = 0, ans = 0;
      for (const Row* r : dev) {
        if (r->grade == 1 || cal.predict(r->f[f]) < tau) continue;
        ++ans;
        right += r->grade == 0;
        wrong += r->grade == 2;
      }
      if (wrong > 0.02 * n) continue;
      if (right > br || (right == br && right > 0 && (wrong < bw || (wrong == bw && ans < ba)))) {
        br = right;
        bw = wrong;
        ba = ans;
        best.f = f;
        best.tau = tau;
        best.cal = cal;
      }
    }
  }
  if (br == 0) best.f = -1;
  return best;
}

static int cmd_qa(const std::string& lib_path, const std::string& rows_path, bool read_test) {
  const std::vector<Row> rows = read_rows(rows_path);
  auto lib = load_library(lib_path);
  bd::Toolkit tk;
  std::printf("## K5a: confidence_monitor in dual_process on CyphaLM's T-QA answers (%zu rows)\n\n", rows.size());
  std::printf("| set | form | lm | n | gate (feature, tau) | right | abstained | wrong | wrong rate (Wilson hi) | IDK cards with a reason |\n");
  std::printf("|---|---|---|---:|---|---:|---:|---:|---|---:|\n");
  std::map<std::string, Gate> gates;
  for (const std::string form : {"chat", "wiki"})
    for (const std::string lm : {"trie", "trie_ig"}) {
      std::vector<const Row*> dev;
      for (const Row& r : rows)
        if (r.set == "dev" && r.form == form && r.lm == lm) dev.push_back(&r);
      gates[form + "|" + lm] = choose_gate(dev);
    }
  for (const std::string set : {"dev", "test"}) {
    if (set == "test" && !read_test) continue;
    for (const std::string form : {"chat", "wiki"})
      for (const std::string lm : {"trie", "trie_ig"}) {
        const Gate& g = gates[form + "|" + lm];
        bd::Toolkit ct;
        ct.calibrator = &g.cal;
        Interpreter it(lib, Config{}, 1);
        bd::bind_all(it, ct);
        const Row* cur = nullptr;
        // dual_process: fast = CyphaLM's own recorded answer; the gate is confidence_monitor; slow = no other
        // CyphaLM route exists, so it abstains.
        it.bind("dual_process.fast", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
          StepResult r;
          if (cur->grade == 1) wm.set("own_idk", true);
          wm.set("answer", Json{{"text", cur->answer.substr(0, 120)}, {"source", "cyphalm"}});
          return r;
        });
        it.bind("dual_process.gate", [&](const Step&, StepContext&, WorkingMemory& wm) {
          WorkingMemory sub;
          sub.set("mode", std::string("answer"));
          sub.set("raw", g.f >= 0 ? cur->f[g.f] : -1e300);
          sub.set("answer", *wm.get("answer"));
          if (wm.has("own_idk")) sub.set("own_idk", true);
          RunResult cm = it.run("confidence_monitor", "qa", sub);
          wm.set("card", *sub.get("card"));
          StepResult r;
          r.chosen_next = std::string("answer");  // the card decides what is sent; slow has no route
          (void)cm;
          return r;
        });
        it.bind("confidence_monitor.produce", [](const Step&, StepContext&, WorkingMemory&) { return StepResult{}; });
        it.bind("confidence_monitor.gate", [&g](const Step&, StepContext&, WorkingMemory& wm) {
          StepResult r;
          const bool own_idk = wm.has("own_idk");
          const bool pass = !own_idk && g.f >= 0 && wm.num("calibrated") >= g.tau;
          if (!pass) {
            wm.set("idk", true);
            wm.set("reason", std::string(own_idk ? "NO_EVIDENCE" : "LOW_CERTAINTY"));
          }
          r.chosen_next = pass ? std::string("answer") : std::string("idk");
          return r;
        });
        int right = 0, abst = 0, wrong = 0, idk_reason = 0, n = 0;
        for (const Row& r : rows) {
          if (r.set != set || r.form != form || r.lm != lm) continue;
          cur = &r;
          ++n;
          WorkingMemory wm;
          RunOptions opt;
          it.run("dual_process", "qa", wm, opt);
          const Json& card = *wm.get("card");
          if (card.at("kind").get<std::string>() == "idk") {
            ++abst;
            idk_reason += card.contains("reason") && !card.at("reason").get<std::string>().empty();
          } else {
            right += r.grade == 0;
            wrong += r.grade == 2;
          }
        }
        const std::string gs = g.f >= 0 ? std::string(kFeat[g.f]) + " " + std::to_string(g.tau) : "answer nothing";
        std::printf("| %s | %s | %s | %d | %s | %d | %d | %d | %.3f (%.3f) | %d |\n", set.c_str(), form.c_str(), lm.c_str(),
                    n, gs.c_str(), right, abst, wrong, n ? wrong / static_cast<double>(n) : 0.0,
                    ev::wilson_upper(wrong, n), idk_reason);
      }
  }
  return 0;
}

// ===========================================================================
// 3.1: applicability
// ===========================================================================

static const std::vector<std::string> kCands = {"half_split", "differential_diagnosis", "random_probe", "sprt_decide",
                                                "ach",        "change_detection",       "spc_control"};

static Task make_task(Rng& r, std::uint64_t seed, bool ood = false) {
  Task t;
  const double u = uniform01(r);
  t.type = ood ? TT::Planning : static_cast<TT>(std::min(4, static_cast<int>(u * 5)));
  auto nz = [&](double v) { return v + 0.3 * normal01(r); };
  double f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  switch (t.type) {
    case TT::ChainU: f[0] = 6; break;
    case TT::ChainS: f[0] = 6; f[1] = 1.2; f[2] = 1; break;
    case TT::Binary: f[0] = 5; f[3] = 1; f[4] = 2; break;
    case TT::Hyp: f[0] = 4; f[4] = 4; break;
    case TT::Monitor: f[0] = 8; f[3] = 1; break;
    case TT::Planning: f[0] = 3; f[4] = 0; f[1] = -3; f[2] = 4; break;
  }
  for (int i = 0; i < 5; ++i) f[i] = nz(f[i]);
  for (int i = 5; i < 8; ++i) f[i] = normal01(r);
  t.x.assign(f, f + 8);
  if (t.type == TT::ChainU || t.type == TT::ChainS) t.chain = make_chain(seed, t.type == TT::ChainS);
  if (t.type == TT::Hyp) t.ach = make_ach(seed);
  if (t.type == TT::Binary) {
    t.stream.h1 = uniform01(r) < 0.5;
    t.stream.seed = seed * 13 + 1;
  }
  if (t.type == TT::Monitor) {
    t.stream.shift = uniform01(r) < 0.5 ? 2.0 : 0.0;
    t.stream.shift_at = 0;
    t.stream.seed = seed * 17 + 3;
    t.cusum_h = 4.0;
    t.shewhart_L = 3.0;
    t.window = 50;
  }
  return t;
}

/// The measured reward of a run on a task (section 8).
static double reward_of(const Task& t, const RunTrace& tr) {
  if (tr.outcome == Outcome::Impasse) return 0.0;
  switch (t.type) {
    case TT::ChainU:
    case TT::ChainS: return t.found ? std::max(0.0, 1.0 - t.spent / 100.0) : 0.0;
    case TT::Binary: return t.decision == (t.stream.h1 ? 1 : 0) ? std::max(0.0, 1.0 - t.stream.t / 200.0) : 0.0;
    case TT::Hyp: return t.ach_pick == t.ach.truth ? 1.0 : 0.0;
    case TT::Monitor: return (t.alarm == (t.stream.shift > 0)) ? 1.0 : 0.0;
    case TT::Planning: return 0.0;
  }
  return 0.0;
}

struct AppStats {
  std::vector<double> p_all, p_chosen, regret;
  std::vector<int> y_all, y_chosen;
  int impasse = 0, tasks = 0;
  std::vector<double> ood_flag_new, ood_flag_known;
};

/// Truth for one (task, process): +1 applies (R >= 0.5), 0 does not (IMPASSE), -1 neither (no label).
static int truth_of(const std::shared_ptr<const Library>& lib, Task t, const std::string& pid, double& R) {
  bd::Toolkit tk;
  Task* cur = nullptr;
  Config cfg;
  cfg.max_steps = 100000;
  Interpreter oi(lib, cfg, 1);
  bind_k5(oi, tk, cur);
  RunOptions opt;
  opt.budget = 1e9;
  opt.reward = [&t](const RunTrace& tr, WorkingMemory&) { return reward_of(t, tr); };
  const RunResult r = run_on(oi, pid, t, cur, "oracle", opt);
  R = r.reward;
  if (r.trace.outcome == Outcome::Impasse) return 0;
  return R >= 0.5 ? 1 : -1;
}

static AppStats run_stream(const std::shared_ptr<const Library>& lib, const std::string& kind, std::uint64_t seed,
                           int train, int heldout, int n_ood) {
  bd::Toolkit tk;
  Task* cur = nullptr;
  Config cfg;
  cfg.max_steps = 100000;
  Interpreter it(lib, cfg, seed);
  bind_k5(it, tk, cur);
  std::shared_ptr<bd::DifApplicability> dif;
  if (kind == "dif") {
    dif = std::make_shared<bd::DifApplicability>(8, 20);
    it.set_applicability(dif);
  } else {
    it.set_applicability(std::make_shared<bd::BetaApplicability>(kind == "beta_class"));
  }
  Rng r(seed);
  AppStats st;
  const int total = train + heldout;
  for (int i = 0; i < total + n_ood; ++i) {
    const bool is_ood = i >= total;
    Task t = make_task(r, seed * 1000003 + static_cast<std::uint64_t>(i), is_ood);
    const std::string cls = kind == "beta_class" ? tt_name(t.type) : "any";
    t.reset_run();
    cur = &t;
    WorkingMemory wm;
    Json sit = Json::array();
    for (double v : t.x) sit.push_back(v);
    wm.set("situation", sit);
    const bool eval = i >= train;
    std::vector<int> truth(kCands.size());
    std::vector<double> Rc(kCands.size(), 0.0), qp(kCands.size(), 0.5);
    if (eval) {
      for (std::size_t c = 0; c < kCands.size(); ++c) {
        truth[c] = truth_of(lib, t, kCands[c], Rc[c]);
        const ApplicabilityQuery q = it.applicability()->query(kCands[c], cls, wm);
        qp[c] = q.p;
        if (is_ood) {
          st.ood_flag_new.push_back(q.ood ? 1.0 : 0.0);
          continue;
        }
        st.ood_flag_known.push_back(q.ood ? 1.0 : 0.0);
        if (truth[c] >= 0) {
          st.p_all.push_back(q.p);
          st.y_all.push_back(truth[c]);
        }
      }
    }
    if (is_ood) continue;
    t.reset_run();
    cur = &t;
    RunOptions opt;
    opt.budget = 1e9;
    opt.G = 1.0;
    opt.reward = [&t](const RunTrace& tr, WorkingMemory&) { return reward_of(t, tr); };
    const SolveResult s = it.solve(Goal{cls, kCands}, wm, opt);
    if (!eval) continue;
    ++st.tasks;
    const double got = s.impasse ? 0.0 : s.run.reward;
    st.impasse += s.impasse || s.run.trace.outcome == Outcome::Impasse;
    st.regret.push_back(*std::max_element(Rc.begin(), Rc.end()) - got);
    if (!s.impasse) {
      const std::size_t c = static_cast<std::size_t>(std::find(kCands.begin(), kCands.end(), s.chosen) - kCands.begin());
      if (truth[c] >= 0) {  // P(applicable) of the chosen process, as predicted before the run
        st.p_chosen.push_back(qp[c]);
        st.y_chosen.push_back(truth[c]);
      }
    }
  }
  return st;
}

static int g_train = 1500, g_heldout = 1000, g_ood = 200;

/// AUROC of p against binary truth (ties count half), report-only.
static double auroc(const std::vector<double>& p, const std::vector<int>& y) {
  std::vector<double> pos, neg;
  for (std::size_t i = 0; i < p.size(); ++i) (y[i] ? pos : neg).push_back(p[i]);
  if (pos.empty() || neg.empty()) return 0.5;
  std::sort(neg.begin(), neg.end());
  double s = 0;
  for (double v : pos) {
    const auto lo = std::lower_bound(neg.begin(), neg.end(), v), hi = std::upper_bound(neg.begin(), neg.end(), v);
    s += static_cast<double>(lo - neg.begin()) + 0.5 * static_cast<double>(hi - lo);
  }
  return s / (static_cast<double>(pos.size()) * static_cast<double>(neg.size()));
}

static int cmd_applicability(const std::string& lib_path) {
  auto lib = load_library(lib_path);
  std::printf("## 3.1: applicability (training 1,500 tasks, held-out 1,000, then 200 of a new type)\n\n");
  std::printf("| model | stream | ECE, chosen (n) | ECE, all pairs | Brier, all | IMPASSE share | mean regret | 95%% CI | OOD flagged: new / known |\n");
  std::printf("|---|---:|---:|---:|---:|---:|---:|---|---|\n");
  std::map<std::string, std::vector<AppStats>> all;
  for (const std::string kind : {"beta_class", "beta_pooled", "dif"}) {
    for (std::uint64_t seed : {1001ULL, 1002ULL, 1003ULL}) {
      AppStats st = run_stream(lib, kind, seed, g_train, g_heldout, g_ood);
      const double ece = ev::expected_calibration_error(st.p_all, st.y_all);
      const double brier = ev::brier_score(st.p_all, st.y_all);
      const auto ci = boot_ci(st.regret);
      const double ece_c = ev::expected_calibration_error(st.p_chosen, st.y_chosen);
      std::printf("| %s | %llu | %.4f (%zu) | %.4f | %.4f | %.3f | %.4f | [%.4f, %.4f] | %.3f / %.3f |\n", kind.c_str(),
                  static_cast<unsigned long long>(seed), ece_c, st.p_chosen.size(), ece, brier,
                  st.impasse / static_cast<double>(st.tasks),
                  mean(st.regret), ci.first, ci.second, mean(st.ood_flag_new), mean(st.ood_flag_known));
      std::fflush(stdout);
      all[kind].push_back(std::move(st));
    }
  }
  std::printf("\nReport-only (post hoc): AUROC of P(applicable) over all (task, process) pairs, and the mean P where the "
              "process applies / does not:\n");
  for (const auto& [kind, v] : all)
    for (std::size_t k = 0; k < v.size(); ++k) {
      double s1 = 0, n1 = 0, s0 = 0, n0 = 0;
      for (std::size_t i = 0; i < v[k].p_all.size(); ++i) (v[k].y_all[i] ? (s1 += v[k].p_all[i], n1 += 1) : (s0 += v[k].p_all[i], n0 += 1));
      std::printf("  %s seed %zu: AUROC %.4f, mean P %.3f / %.3f\n", kind.c_str(), 1001 + k, auroc(v[k].p_all, v[k].y_all),
                  n1 ? s1 / n1 : 0.0, n0 ? s0 / n0 : 0.0);
    }
  // LCI: DIF against Beta-pooled on per-task regret (positive = DIF better), seed 1001 the test, the others stability;
  // the dev gain from a dev stream (seed 1)
  AppStats dev_p = run_stream(lib, "beta_pooled", 1, g_train, g_heldout, 0);
  AppStats dev_d = run_stream(lib, "dif", 1, g_train, g_heldout, 0);
  ev::LciInputs in;
  const AppStats& tp = all["beta_pooled"][0];
  const AppStats& td = all["dif"][0];
  for (std::size_t i = 0; i < std::min(tp.regret.size(), td.regret.size()); ++i) in.test_diffs.push_back(tp.regret[i] - td.regret[i]);
  in.eps = 0.0;
  in.gain_dev = mean(dev_p.regret) - mean(dev_d.regret);
  in.gain_ext = mean(tp.regret) - mean(td.regret);
  for (int k = 0; k < 3; ++k) in.seed_gains.push_back(mean(all["beta_pooled"][k].regret) - mean(all["dif"][k].regret));
  in.ece_before = ev::expected_calibration_error(tp.p_all, tp.y_all);
  in.ece_after = ev::expected_calibration_error(td.p_all, td.y_all);
  in.brier_before = ev::brier_score(tp.p_all, tp.y_all);
  in.brier_after = ev::brier_score(td.p_all, td.y_all);
  const ev::LciDecision d = ev::compute_lci(in);
  std::printf("\nLCI, DIF against Beta-pooled (regret gain per task): ");
  for (const auto& [k, v] : d.components()) std::printf("%s %.4f  ", k.c_str(), v);
  std::printf("\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: thought_k5 qa|chains|hyps|monitor|applicability PROCESSES_JSON [ROWS] [--read-test]\n");
    return 2;
  }
  const std::string cmd = argv[1], lib = argv[2];
  if (cmd == "qa" && argc >= 4) return cmd_qa(lib, argv[3], argc >= 5 && std::string(argv[4]) == "--read-test");
  if (cmd == "chains") return cmd_chains(lib);
  if (cmd == "hyps") return cmd_hyps(lib);
  if (cmd == "monitor") return cmd_monitor(lib);
  if (cmd == "applicability") {
    if (argc >= 6) {  // a smaller stream for a smoke run: TRAIN HELDOUT OOD
      g_train = std::atoi(argv[3]);
      g_heldout = std::atoi(argv[4]);
      g_ood = std::atoi(argv[5]);
    }
    return cmd_applicability(lib);
  }
  std::fprintf(stderr, "unknown command\n");
  return 2;
}
