// Interpreter for Cypha's thought-process library (docs/THOUGHT.md sections 1.3, 3, 5, 12: K2).
//
// Header-only. Needs nlohmann/json (native/third_party) and learners.hpp.
//
//   Library       loads docs/thought/processes.json (written by scripts/thought_check.py --json),
//                 checks that each process can be instantiated (known primitives, resolvable calls,
//                 valid transitions).
//   Interpreter   runs a process: primitive registry (op name -> callback), `call:` ops as
//                 recursive sub-runs (depth cap 5, budget share), guards (a callback's chosen_next)
//                 or the learned Dirichlet for the next step, stops on final steps / budget / VOC,
//                 one trace record per step, and after the run updates every standard learner from
//                 the reward R (3.2-3.8). solve() selects among applicable processes by Thompson
//                 sampling (3.5) and returns IMPASSE when none applies.
//
// Not here yet: the DIF applicability classifier (3.1) - applicability is a Beta per
// (process, task class) for now; bindings of primitives to Cypha's real tools (K4); chunking (K6);
// mining (K7). The TraceSink hook is where traces go to the experiment DB.
//
// Author: Odin Loch
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "learners.hpp"

namespace cypha::thought {

using Json = nlohmann::json;

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

struct Step {
  std::string id;
  std::string op;   // a primitive, or "call:<process id>"
  std::string doc;  // the YAML `do` text
  std::vector<std::string> next;

  bool is_call() const { return op.rfind("call:", 0) == 0; }
  std::string call_target() const { return is_call() ? op.substr(5) : std::string(); }
  bool is_final() const { return next.empty(); }
};

struct Process {
  std::string id, name, cat, ref, when, start, stop, math, feedback;
  std::vector<std::string> learn;
  std::map<std::string, Step> steps;
};

class Library {
 public:
  static Library load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("thought: cannot open " + path);
    return from_json(Json::parse(f));
  }

  static Library from_json(const Json& j) {
    Library lib;
    for (auto it = j.at("primitives").begin(); it != j.at("primitives").end(); ++it) lib.prims_.insert(it.key());
    for (auto it = j.at("categories").begin(); it != j.at("categories").end(); ++it) lib.cats_.insert(it.key());
    for (const Json& pj : j.at("processes")) lib.add_process(pj);
    return lib;
  }

  /// Add (or for mined processes, later, propose) a process. Throws on a duplicate id.
  void add_process(const Json& pj) {
    Process p;
    p.id = pj.at("id").get<std::string>();
    if (index_.count(p.id)) throw std::runtime_error("thought: duplicate process " + p.id);
    auto str = [&](const char* k) { return pj.contains(k) ? pj.at(k).get<std::string>() : std::string(); };
    p.name = str("name");
    p.cat = str("cat");
    p.ref = str("ref");
    p.when = str("when");
    p.start = str("start");
    p.stop = str("stop");
    p.math = str("math");
    p.feedback = str("feedback");
    if (pj.contains("learn")) p.learn = pj.at("learn").get<std::vector<std::string>>();
    for (auto it = pj.at("steps").begin(); it != pj.at("steps").end(); ++it) {
      Step s;
      s.id = it.key();
      s.op = it.value().at("op").get<std::string>();
      if (it.value().contains("do")) s.doc = it.value().at("do").get<std::string>();
      if (it.value().contains("next")) s.next = it.value().at("next").get<std::vector<std::string>>();
      p.steps.emplace(s.id, std::move(s));
    }
    index_[p.id] = procs_.size();
    procs_.push_back(std::move(p));
  }

  const Process* find(const std::string& id) const {
    const auto it = index_.find(id);
    return it == index_.end() ? nullptr : &procs_[it->second];
  }
  const std::vector<Process>& processes() const { return procs_; }
  const std::set<std::string>& primitives() const { return prims_; }
  const std::set<std::string>& categories() const { return cats_; }

  /// Why a process cannot be instantiated; empty when it can. Same rules as thought_check.py.
  std::vector<std::string> check(const std::string& id) const {
    std::vector<std::string> errs;
    const Process* p = find(id);
    if (!p) return {"no process " + id};
    if (!p->steps.count(p->start)) errs.push_back(id + ": start '" + p->start + "' is not a step");
    for (const auto& [sid, s] : p->steps) {
      if (s.is_call()) {
        if (!find(s.call_target())) errs.push_back(id + "." + sid + ": calls unknown process " + s.call_target());
      } else if (!prims_.count(s.op)) {
        errs.push_back(id + "." + sid + ": unknown primitive " + s.op);
      }
      for (const std::string& n : s.next)
        if (!p->steps.count(n)) errs.push_back(id + "." + sid + ": next '" + n + "' is not a step");
      if (s.next.empty() && s.op != "return") errs.push_back(id + "." + sid + ": final step must be 'return'");
      if (!s.next.empty() && s.op == "return") errs.push_back(id + "." + sid + ": 'return' cannot have next");
    }
    return errs;
  }

  std::vector<std::string> validate() const {
    std::vector<std::string> errs;
    for (const Process& p : procs_) {
      std::vector<std::string> e = check(p.id);
      errs.insert(errs.end(), e.begin(), e.end());
    }
    return errs;
  }

 private:
  std::set<std::string> prims_;
  std::set<std::string> cats_;
  std::vector<Process> procs_;
  std::map<std::string, std::size_t> index_;
};

// ---------------------------------------------------------------------------
// Working memory, step context, results
// ---------------------------------------------------------------------------

/// The shared global workspace: goals, partial results and the current focus. Ordered, so a
/// dump is deterministic. The interpreter writes "focus" (process.step) and "last_value".
class WorkingMemory {
 public:
  void set(const std::string& k, Json v) { m_[k] = std::move(v); }
  bool has(const std::string& k) const { return m_.count(k) != 0; }
  const Json* get(const std::string& k) const {
    const auto it = m_.find(k);
    return it == m_.end() ? nullptr : &it->second;
  }
  double num(const std::string& k, double def = 0.0) const {
    const Json* v = get(k);
    return (v && v->is_number()) ? v->get<double>() : def;
  }
  std::string str(const std::string& k, const std::string& def = "") const {
    const Json* v = get(k);
    return (v && v->is_string()) ? v->get<std::string>() : def;
  }
  void erase(const std::string& k) { m_.erase(k); }
  void clear() { m_.clear(); }
  const std::map<std::string, Json>& items() const { return m_; }

 private:
  std::map<std::string, Json> m_;
};

struct StepContext {
  std::string process;     // process being run
  std::string step;        // step id
  std::string task_class;  // context bucket for the learners
  int depth = 0;           // 0 for the top-level process
  std::size_t index = 0;   // position of this step in the process's own trace
  double budget_left = 0;  // budget remaining before this step
  Rng* rng = nullptr;      // the interpreter's seeded stream (optional use)
};

struct StepResult {
  bool ok = true;                           // the step succeeded (a test passed, ...)
  Json value;                               // result payload (null by default)
  double cost = 0.0;                        // time / compute / queries, in the run's budget units
  std::optional<std::string> chosen_next;   // guard: take this branch (must be in step.next)
  bool done = false;                        // the goal is reached; stop now
  std::optional<double> gain;               // measured improvement of the goal measure (feeds VOC)
  bool impasse = false;                     // cannot proceed
};

using Primitive = std::function<StepResult(const Step&, StepContext&, WorkingMemory&)>;

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

enum class Outcome { Done, Budget, Voc, MaxSteps, Impasse, Error };

inline const char* outcome_name(Outcome o) {
  switch (o) {
    case Outcome::Done: return "done";
    case Outcome::Budget: return "budget";
    case Outcome::Voc: return "voc";
    case Outcome::MaxSteps: return "max_steps";
    case Outcome::Impasse: return "impasse";
    case Outcome::Error: return "error";
  }
  return "?";
}

struct StepRecord {
  std::string step, op;
  bool ok = true;
  std::string value;          // Json dump
  double cost = 0.0;
  bool has_gain = false;
  double gain = 0.0;
  std::string next;           // branch taken ("" for a final step or a stop)
  std::string how;            // final | guard | single | learned | stop
  int branches = 0;           // size of the next set
  int branch_index = -1;      // index of the branch taken within step.next
  int child = -1;             // index into RunTrace::children for a call step
  std::string note;           // depth_cap | no_primitive | bad_next | ...
  double credit = 0.0;        // TD(lambda) eligibility e_k assigned at learning time
};

struct RunTrace {
  std::string process, task_class;
  int depth = 0;
  double budget = 0.0, spent = 0.0;
  double p_hat = 0.0;         // predicted success before the run (3.8)
  Outcome outcome = Outcome::Done;
  std::string value;          // last non-null value at exit
  std::string note;
  double reward = 0.0;        // set by learning
  std::vector<StepRecord> steps;
  std::vector<RunTrace> children;
};

inline int max_depth_of(const RunTrace& t) {
  int d = t.depth;
  for (const RunTrace& c : t.children) d = std::max(d, max_depth_of(c));
  return d;
}

inline std::size_t count_steps(const RunTrace& t) {
  std::size_t n = t.steps.size();
  for (const RunTrace& c : t.children) n += count_steps(c);
  return n;
}

/// Exact text form of a trace (hex floats), for replay comparison and hashing.
inline void trace_text(const RunTrace& t, std::string& out) {
  char buf[96];
  auto dbl = [&](double x) {
    std::snprintf(buf, sizeof buf, "%a", x);
    out += buf;
  };
  out += "RUN " + t.process + " " + t.task_class + " d" + std::to_string(t.depth) + " " + outcome_name(t.outcome) + " b=";
  dbl(t.budget);
  out += " s=";
  dbl(t.spent);
  out += " p=";
  dbl(t.p_hat);
  out += " r=";
  dbl(t.reward);
  out += " v=" + t.value + " n=" + t.note + "\n";
  for (const StepRecord& r : t.steps) {
    out += " " + r.step + " " + r.op + (r.ok ? " ok " : " no ") + r.value + " c=";
    dbl(r.cost);
    out += r.has_gain ? " g=" : " g-";
    if (r.has_gain) dbl(r.gain);
    out += " ->" + r.next + " " + r.how + " " + std::to_string(r.branch_index) + "/" + std::to_string(r.branches) +
           " k" + std::to_string(r.child) + " [" + r.note + "] e=";
    dbl(r.credit);
    out += "\n";
  }
  for (const RunTrace& c : t.children) trace_text(c, out);
}

inline std::string trace_text(const RunTrace& t) {
  std::string s;
  trace_text(t, s);
  return s;
}

inline std::uint64_t fnv1a(const std::string& s, std::uint64_t h = 1469598103934665603ULL) {
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

// ---------------------------------------------------------------------------
// Learner store (3.2 - 3.4, 3.8)
// ---------------------------------------------------------------------------

struct StepStats {
  Beta success;      // credited by TD(lambda), plus a direct update from test/verify steps
  LogWelford cost;   // log cost
  Welford gain;      // improvement of the goal measure after the step (for VOC)
};

struct ProcStats {
  Beta success;      // P(success | process, task class)
  Beta applicable;   // P(applicable | process, task class): stands in for the DIF classifier (3.1)
  LogWelford cost;   // log cost of a whole run
  ActRUtility util;  // U <- U + a (R*G - C), a fast second estimate; not used for selection
  std::size_t runs = 0;
};

struct Learners {
  std::map<std::string, ProcStats> proc;      // process | task class
  std::map<std::string, StepStats> step;      // process | step | task class
  std::map<std::string, Dirichlet> branch;    // process | step | task class
  CalibrationLog calibration;                 // per process id
};

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

struct Config {
  int max_depth = 5;                // call nesting cap (THOUGHT 5)
  double theta = 0.1;               // run only if P(applicable) >= theta (1.3)
  double epsilon = 0.02;            // selection floor per applicable process (3.5)
  double lambda = 0.8;              // TD(lambda) (3.6)
  double act_alpha = 0.1;           // ACT-R rate (3.4)
  double prior_cost = 0.0;          // cost used before any run has been seen
  std::size_t voc_min_n = 5;        // no VOC stop with fewer observations of a step
  std::size_t max_steps = 1000;     // safety cap on steps per process run
  double child_cap = 0.5;           // a child gets at most this share of the parent's remaining budget
  double useful_threshold = 0.5;    // R >= this counts as success or a useful partial result (3.1)
};

using RewardFn = std::function<double(const RunTrace&, WorkingMemory&)>;
using TraceSink = std::function<void(const RunTrace&)>;

struct RunOptions {
  double budget = 100.0;
  double G = 1.0;       // value of the goal, in cost units (3.4)
  RewardFn reward;      // measured feedback (section 8); default: pass rate of test/verify steps
};

struct RunResult {
  RunTrace trace;
  double reward = 0.0;
  bool impasse() const { return trace.outcome == Outcome::Impasse; }
};

struct Goal {
  std::string task_class;
  std::vector<std::string> candidates;  // process ids to choose between
};

struct SolveResult {
  bool impasse = false;
  std::string chosen;
  bool floor_pick = false;
  std::vector<std::string> applicable;
  RunResult run;
};

/// Default reward: the pass rate of the test/verify steps anywhere in the run (section 8,
/// "a check inside the process"); with no such step, 1 if the run reached its end, else 0.
inline double default_reward(const RunTrace& t, WorkingMemory&) {
  double pass = 0, total = 0;
  std::vector<const RunTrace*> stack{&t};
  while (!stack.empty()) {
    const RunTrace* r = stack.back();
    stack.pop_back();
    for (const StepRecord& s : r->steps)
      if (s.op == "test" || s.op == "verify") {
        total += 1;
        pass += s.ok ? 1 : 0;
      }
    for (const RunTrace& c : r->children) stack.push_back(&c);
  }
  if (total > 0) return pass / total;
  return t.outcome == Outcome::Done ? 1.0 : 0.0;
}

class Interpreter {
 public:
  Interpreter(std::shared_ptr<const Library> lib, Config cfg = {}, std::uint64_t seed = 1)
      : lib_(std::move(lib)), cfg_(cfg), rng_(seed) {}

  /// Register a callback. `key` is a primitive op name ("test"), or "<process>.<step>" for one
  /// step of one process; the step-specific binding wins.
  void bind(const std::string& key, Primitive fn) { prims_[key] = std::move(fn); }
  void reseed(std::uint64_t seed) { rng_.seed(seed); }
  void set_trace_sink(TraceSink s) { sink_ = std::move(s); }

  const Library& library() const { return *lib_; }
  const Config& config() const { return cfg_; }
  Learners& learners() { return L_; }
  const Learners& learners() const { return L_; }
  Rng& rng() { return rng_; }

  /// Ops used by `id` (and, through call:, its callees) that have no binding. K4 works down this list.
  std::vector<std::string> unbound(const std::string& id) const {
    std::set<std::string> seen, missing;
    collect_unbound(id, seen, missing);
    return {missing.begin(), missing.end()};
  }

  ProcStats& proc_stats(const std::string& pid, const std::string& cls) { return L_.proc[key(pid, cls)]; }
  StepStats& step_stats(const std::string& pid, const std::string& sid, const std::string& cls) {
    return L_.step[key(pid, sid, cls)];
  }
  const Dirichlet* branch_stats(const std::string& pid, const std::string& sid, const std::string& cls) const {
    const auto it = L_.branch.find(key(pid, sid, cls));
    return it == L_.branch.end() ? nullptr : &it->second;
  }

  /// Run one process (1.3): applicability test, prediction, execution, reward, learning.
  RunResult run(const std::string& pid, const std::string& task_class, WorkingMemory& wm, const RunOptions& opt = {}) {
    RunResult res;
    if (!lib_->find(pid)) {
      res.trace = blank(pid, task_class, 0, opt.budget);
      res.trace.outcome = Outcome::Error;
      res.trace.note = "unknown process";
      return res;
    }
    ProcStats& ps = proc_stats(pid, task_class);
    if (ps.applicable.mean() < cfg_.theta) {  // not applicable here: IMPASSE, no run
      res.trace = blank(pid, task_class, 0, opt.budget);
      res.trace.outcome = Outcome::Impasse;
      res.trace.note = "not applicable";
      return res;
    }
    const double p_hat = ps.success.mean();
    res.trace = execute(pid, task_class, 0, opt.budget, p_hat, opt, wm);
    res.reward = opt.reward ? opt.reward(res.trace, wm) : default_reward(res.trace, wm);
    res.reward = std::min(1.0, std::max(0.0, res.reward));
    learn(res.trace, res.reward, 1.0, true, opt);
    if (sink_) sink_(res.trace);
    return res;
  }

  /// Choose among a goal's applicable candidates by Thompson sampling (3.5), run the choice.
  /// IMPASSE when none is applicable.
  SolveResult solve(const Goal& goal, WorkingMemory& wm, const RunOptions& opt = {}) {
    SolveResult out;
    for (const std::string& pid : goal.candidates) {
      if (!lib_->find(pid)) continue;
      if (proc_stats(pid, goal.task_class).applicable.mean() >= cfg_.theta) out.applicable.push_back(pid);
    }
    if (out.applicable.empty()) {
      out.impasse = true;
      out.run.trace = blank("(goal)", goal.task_class, 0, opt.budget);
      out.run.trace.outcome = Outcome::Impasse;
      out.run.trace.note = "no applicable process";
      return out;
    }
    std::vector<ThompsonArm> arms;
    for (const std::string& pid : out.applicable) {
      const ProcStats& ps = proc_stats(pid, goal.task_class);
      arms.push_back({&ps.success, &ps.cost});
    }
    const ThompsonChoice ch = thompson_select(arms, opt.G, cfg_.epsilon, rng_, cfg_.prior_cost);
    out.chosen = out.applicable[ch.index];
    out.floor_pick = ch.floor_pick;
    out.run = run(out.chosen, goal.task_class, wm, opt);
    return out;
  }

 private:
  static std::string key(const std::string& a, const std::string& b) { return a + '\x1f' + b; }
  static std::string key(const std::string& a, const std::string& b, const std::string& c) {
    return a + '\x1f' + b + '\x1f' + c;
  }

  static RunTrace blank(const std::string& pid, const std::string& cls, int depth, double budget) {
    RunTrace t;
    t.process = pid;
    t.task_class = cls;
    t.depth = depth;
    t.budget = budget;
    return t;
  }

  void collect_unbound(const std::string& id, std::set<std::string>& seen, std::set<std::string>& missing) const {
    if (!seen.insert(id).second) return;
    const Process* p = lib_->find(id);
    if (!p) return;
    for (const auto& [sid, s] : p->steps) {
      if (s.is_call()) {
        collect_unbound(s.call_target(), seen, missing);
      } else if (s.op != "return" && !prims_.count(id + "." + sid) && !prims_.count(s.op)) {
        missing.insert(s.op);
      }
    }
  }

  const Primitive* lookup(const std::string& pid, const Step& s) const {
    auto it = prims_.find(pid + "." + s.id);
    if (it == prims_.end()) it = prims_.find(s.op);
    return it == prims_.end() ? nullptr : &it->second;
  }

  /// Budget share for a child: phi = E[child cost] / E[parent remaining cost], capped at
  /// child_cap of what the parent has left (3.7). Without cost data the cap applies.
  double child_budget(const std::string& parent, const std::string& child, const std::string& cls, double spent,
                      double remaining) {
    double phi = cfg_.child_cap;
    const ProcStats& pc = proc_stats(child, cls);
    const ProcStats& pp = proc_stats(parent, cls);
    if (pc.cost.count() > 0 && pp.cost.count() > 0) {
      const double e_child = pc.cost.expected();
      const double e_rem = std::max(pp.cost.expected() - spent, e_child);
      phi = e_rem > 0.0 ? e_child / e_rem : cfg_.child_cap;
    }
    return std::min(phi, cfg_.child_cap) * remaining;
  }

  RunTrace execute(const std::string& pid, const std::string& cls, int depth, double budget, double p_hat,
                   const RunOptions& opt, WorkingMemory& wm) {
    RunTrace t = blank(pid, cls, depth, budget);
    t.p_hat = p_hat;
    const Process* P = lib_->find(pid);
    std::string cur = P->start;
    for (;;) {
      if (t.steps.size() >= cfg_.max_steps) {
        t.outcome = Outcome::MaxSteps;
        break;
      }
      const auto sit = P->steps.find(cur);
      if (sit == P->steps.end()) {
        t.outcome = Outcome::Error;
        t.note = "no step " + cur;
        break;
      }
      const Step& step = sit->second;
      if (!step.is_final()) {
        if (t.spent >= t.budget) {  // budget (3.7)
          t.outcome = Outcome::Budget;
          break;
        }
        if (!t.steps.empty()) {     // value of computation (3.7), before each step after the first
          const auto st = L_.step.find(key(pid, cur, cls));
          if (st != L_.step.end() && voc_stop(st->second.gain, st->second.cost, opt.G, cfg_.voc_min_n)) {
            t.outcome = Outcome::Voc;
            break;
          }
        }
      }

      StepRecord rec;
      rec.step = step.id;
      rec.op = step.op;
      rec.branches = static_cast<int>(step.next.size());
      StepResult r;
      if (step.is_call()) {
        if (depth + 1 > cfg_.max_depth) {  // depth cap
          r.ok = false;
          rec.note = "depth_cap";
        } else {
          const double remaining = t.budget - t.spent;
          const std::string child = step.call_target();
          const double cb = child_budget(pid, child, cls, t.spent, remaining);
          RunTrace ct = execute(child, cls, depth + 1, cb, proc_stats(child, cls).success.mean(), opt, wm);
          r.ok = ct.outcome == Outcome::Done;
          r.cost = ct.spent;
          rec.child = static_cast<int>(t.children.size());
          if (!ct.value.empty()) {
            r.value = Json::parse(ct.value);
            t.value = ct.value;
          }
          t.children.push_back(std::move(ct));
        }
      } else if (const Primitive* fn = lookup(pid, step)) {
        StepContext ctx;
        ctx.process = pid;
        ctx.step = step.id;
        ctx.task_class = cls;
        ctx.depth = depth;
        ctx.index = t.steps.size();
        ctx.budget_left = t.budget - t.spent;
        ctx.rng = &rng_;
        r = (*fn)(step, ctx, wm);
      } else if (step.op == "return") {
        // built-in: a final node costs nothing and returns the last value
      } else {
        r.ok = false;
        r.impasse = true;
        rec.note = "no_primitive";
      }

      rec.ok = r.ok;
      rec.value = r.value.dump();
      rec.cost = r.cost;
      if (r.gain) {
        rec.has_gain = true;
        rec.gain = *r.gain;
      }
      if (!r.value.is_null()) {
        t.value = rec.value;
        wm.set("last_value", r.value);
      }
      wm.set("focus", Json(pid + "." + step.id));
      t.spent += r.cost;

      // next step: guard, single, or the learned Dirichlet (3.2)
      bool stop = false;
      if (step.is_final()) {
        rec.how = "final";
        t.outcome = Outcome::Done;
        stop = true;
      } else if (r.impasse) {
        rec.how = "stop";
        t.outcome = Outcome::Impasse;
        t.note = rec.note;
        stop = true;
      } else if (r.done) {
        rec.how = "stop";
        t.outcome = Outcome::Done;
        stop = true;
      } else {
        int idx = -1;
        if (r.chosen_next) {
          for (std::size_t i = 0; i < step.next.size(); ++i)
            if (step.next[i] == *r.chosen_next) idx = static_cast<int>(i);
          if (idx < 0) {
            rec.note = "bad_next";
            rec.how = "stop";
            t.outcome = Outcome::Error;
            t.note = "bad chosen_next '" + *r.chosen_next + "' at " + pid + "." + step.id;
            stop = true;
          } else {
            rec.how = "guard";
          }
        } else if (step.next.size() == 1) {
          idx = 0;
          rec.how = "single";
        } else {
          Dirichlet& d = L_.branch[key(pid, cur, cls)];
          if (d.size() != step.next.size()) d = Dirichlet(step.next.size());
          idx = static_cast<int>(d.choose(rng_));
          rec.how = "learned";
        }
        if (idx >= 0) {
          rec.branch_index = idx;
          rec.next = step.next[static_cast<std::size_t>(idx)];
          cur = rec.next;
        }
      }
      t.steps.push_back(std::move(rec));
      if (stop) break;
    }
    return t;
  }

  /// Update every standard learner from the reward R (3.2 - 3.8). `scale` is the parent's credit
  /// for a sub-process (1 at the top): a child learns from every parent that uses it.
  void learn(RunTrace& t, double R, double scale, bool top, const RunOptions& opt) {
    t.reward = R;
    const std::vector<double> e = td_credits(t.steps.size(), cfg_.lambda);
    for (std::size_t k = 0; k < t.steps.size(); ++k) {
      StepRecord& rec = t.steps[k];
      const double credit = e[k] * scale;
      rec.credit = credit;
      StepStats& ss = step_stats(t.process, rec.step, t.task_class);
      ss.success.update(R, credit);                         // 3.6
      if (rec.op == "test" || rec.op == "verify")           // a step with its own check
        ss.success.update(rec.ok ? 1.0 : 0.0, 1.0);
      if (rec.note.empty() || rec.note == "bad_next") {     // 3.3: cost and gain, observed directly
        ss.cost.update(rec.cost);
        if (rec.has_gain) ss.gain.update(rec.gain);
      }
      if (rec.branch_index >= 0 && rec.branches >= 2 && rec.how != "stop") {   // 3.2
        Dirichlet& d = L_.branch[key(t.process, rec.step, t.task_class)];
        if (d.size() != static_cast<std::size_t>(rec.branches)) d = Dirichlet(static_cast<std::size_t>(rec.branches));
        d.update(static_cast<std::size_t>(rec.branch_index), credit * R);
      }
      if (rec.child >= 0) learn(t.children[static_cast<std::size_t>(rec.child)], R, credit, false, opt);
    }
    ProcStats& ps = proc_stats(t.process, t.task_class);
    ps.success.update(R, scale);                            // 3.4
    ps.cost.update(t.spent);
    ++ps.runs;
    if (top) {
      ps.util.alpha = cfg_.act_alpha;
      ps.util.update(R * opt.G - t.spent);                  // ACT-R on the same scale as p*G - C
      L_.calibration.add(t.process, t.p_hat, R);            // 3.8
      if (t.outcome == Outcome::Impasse)                    // applicability labels (3.1)
        ps.applicable.update(0.0);
      else if (R >= cfg_.useful_threshold)
        ps.applicable.update(1.0);
    }
  }

  std::shared_ptr<const Library> lib_;
  Config cfg_;
  Rng rng_;
  std::map<std::string, Primitive> prims_;
  Learners L_;
  TraceSink sink_;
};

}  // namespace cypha::thought
