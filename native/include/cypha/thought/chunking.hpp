// K6: chunking (docs/THOUGHT.md section 4; docs/reports/K6_K7_CHUNKING_AND_MINING.md).
//
// Header-only. Turns repeated System 2 results into System 1 rules, as THOUGHT 4 writes it:
//   1. compile   when the last 5 System 2 results for a situation class are identical, a rule
//                "class -> that result" is compiled; it is tainted;
//   2. trial     a tainted rule's prediction is scored against System 2, which still runs and answers;
//   3. keep      the rule joins System 1 (it answers) when the Wilson lower bound of its agreement with
//                System 2 is >= 0.9 over at least 20 checked uses;
//   4. drop      a kept rule is still checked on a share `audit` of its uses; any rule is dropped when its
//                agreement over its last 20 checked uses falls below 0.8. After a drop the class is barred
//                (reentry_n = 0) or may recompile after reentry_n identical results in a row.
//
// bind_dual_chunks() binds the library's `dual_process` steps to a ChunkStore: fast = the class's rule,
// gate = a kept rule answers unless audited, slow + run = System 2 (the caller's callback), compare + monitor
// = the rule's agreement. A rule's answer has source "chunk": in answer mode the rule-10 guard
// (bindings.hpp do_communicate) turns it into an "I don't know" card. Chunks speed up deliberate work; they
// never answer a question (PLAN rule 10).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <utility>

#include "cypha/thought/interpreter.hpp"
#include "cypha/thought/learners.hpp"

namespace cypha::thought {

struct ChunkRules {
  int compile_n = 5;          // identical System 2 results in a row to compile
  int keep_min_checks = 20;   // checked uses before a rule may be kept
  double keep_lb = 0.9;       // Wilson lower bound of agreement to keep
  int drop_window = 20;       // the last checked uses the drop test reads
  double drop_below = 0.8;    // dropped when agreement over the window is below this
  double audit = 0.10;        // share of a kept rule's uses that System 2 still checks
  int reentry_n = 10;         // identical results in a row to recompile after a drop; 0 = barred
};

enum class ChunkStatus { None, Tainted, Kept };

inline const char* chunk_status_name(ChunkStatus s) {
  switch (s) {
    case ChunkStatus::None: return "none";
    case ChunkStatus::Tainted: return "tainted";
    case ChunkStatus::Kept: return "kept";
  }
  return "?";
}

struct Chunk {
  std::string result;
  ChunkStatus status = ChunkStatus::None;
  int checks = 0, agree = 0;  // checked uses since this rule was compiled
  std::deque<int> recent;     // the last drop_window checks: 1 agreed, 0 did not
};

struct ChunkClass {
  std::deque<std::string> last;  // the latest System 2 results for the class
  Chunk rule;
  int dropped = 0;               // times a rule of this class was dropped
  int compiled = 0, kept = 0;    // events, for reports
};

class ChunkStore {
 public:
  explicit ChunkStore(ChunkRules r = {}) : r_(r) {}

  const ChunkRules& rules() const { return r_; }

  /// The class's rule, tainted or kept; nullptr when there is none.
  const Chunk* lookup(const std::string& cls) const {
    const auto it = c_.find(cls);
    if (it == c_.end() || it->second.rule.status == ChunkStatus::None) return nullptr;
    return &it->second.rule;
  }

  /// After System 2 ran for the class: record its result; compile when the rule asks for it.
  void observe_slow(const std::string& cls, const std::string& result) {
    ChunkClass& k = c_[cls];
    const int need = k.dropped > 0 ? r_.reentry_n : r_.compile_n;
    k.last.push_back(result);
    while (static_cast<int>(k.last.size()) > std::max(r_.compile_n, r_.reentry_n)) k.last.pop_front();
    if (k.rule.status != ChunkStatus::None || need <= 0) return;  // a rule exists, or the class is barred
    if (static_cast<int>(k.last.size()) < need) return;
    const std::string& r = k.last.back();
    for (int i = 0; i < need; ++i)
      if (k.last[k.last.size() - 1 - static_cast<std::size_t>(i)] != r) return;
    k.rule = Chunk{};
    k.rule.result = r;
    k.rule.status = ChunkStatus::Tainted;
    ++k.compiled;
  }

  /// A checked use of the class's rule: did it agree with System 2?
  void check(const std::string& cls, bool agree) {
    const auto it = c_.find(cls);
    if (it == c_.end() || it->second.rule.status == ChunkStatus::None) return;
    ChunkClass& k = it->second;
    Chunk& ch = k.rule;
    ++ch.checks;
    ch.agree += agree ? 1 : 0;
    ch.recent.push_back(agree ? 1 : 0);
    while (static_cast<int>(ch.recent.size()) > r_.drop_window) ch.recent.pop_front();
    if (static_cast<int>(ch.recent.size()) == r_.drop_window) {
      int s = 0;
      for (int v : ch.recent) s += v;
      if (static_cast<double>(s) < r_.drop_below * r_.drop_window) {  // drop: the class must recompile
        ch = Chunk{};
        ++k.dropped;
        k.last.clear();
        return;
      }
    }
    if (ch.status == ChunkStatus::Tainted && ch.checks >= r_.keep_min_checks &&
        wilson_interval(ch.agree, ch.checks).lo >= r_.keep_lb) {
      ch.status = ChunkStatus::Kept;
      ++k.kept;
    }
  }

  const std::map<std::string, ChunkClass>& classes() const { return c_; }

 private:
  ChunkRules r_;
  std::map<std::string, ChunkClass> c_;
};

/// The task the dual process is working on: its situation class and System 2 (result, cost).
struct DualTask {
  std::string cls;
  std::function<std::pair<std::string, double>()> system2;
  // filled by the run
  std::string fast_result, slow_result, route;  // route: "fast" (System 1 answered) or "slow"
  bool had_rule = false, audited = false, checked = false, agreed = false;
};

/// Bind the library's `dual_process` steps to a ChunkStore (step-specific bindings win over any toolkit binding).
/// `cur` points at the task being run; `audit_rng` draws the audits (its own stream, so arms share System 2's draws).
inline void bind_dual_chunks(Interpreter& it, ChunkStore& cs, DualTask*& cur, Rng& audit_rng,
                             double fast_cost = 1.0) {
  it.bind("dual_process.fast", [&cs, &cur, fast_cost](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    const Chunk* ch = cur ? cs.lookup(cur->cls) : nullptr;
    if (!ch) {
      r.ok = false;
      return r;
    }
    cur->had_rule = true;
    cur->fast_result = ch->result;
    r.value = Json{{"text", ch->result}, {"status", chunk_status_name(ch->status)}, {"source", "chunk"}};
    wm.set("fast", r.value);
    r.cost = fast_cost;
    return r;
  });
  it.bind("dual_process.gate", [&cs, &cur, &audit_rng](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    const Chunk* ch = cur ? cs.lookup(cur->cls) : nullptr;
    if (ch && ch->status == ChunkStatus::Kept) {
      const bool audit = uniform01(audit_rng) < cs.rules().audit;
      cur->audited = audit;
      if (!audit) {
        cur->route = "fast";
        wm.set("answer", Json{{"text", ch->result}, {"source", "chunk"}, {"confidence", 1.0}});
        r.chosen_next = std::string("answer");
        return r;
      }
    }
    r.chosen_next = std::string("slow");
    return r;
  });
  it.bind("dual_process.slow", [](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    wm.set("system2", Json("deliberate"));
    return r;
  });
  it.bind("dual_process.run", [&cur](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    const auto [res, cost] = cur->system2();
    cur->slow_result = res;
    cur->route = "slow";
    wm.set("answer", Json{{"text", res}, {"source", "system2"}, {"confidence", 1.0}});
    r.value = Json(res);
    r.cost = cost;
    return r;
  });
  it.bind("dual_process.compare", [&cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    if (cur->had_rule) {
      cur->checked = true;
      cur->agreed = cur->fast_result == cur->slow_result;
      r.ok = cur->agreed;
      r.chosen_next = std::string("monitor");
    } else {
      r.chosen_next = std::string("answer");
    }
    return r;
  });
  it.bind("dual_process.monitor", [&cs, &cur](const Step&, StepContext&, WorkingMemory&) {
    StepResult r;
    cs.check(cur->cls, cur->agreed);
    return r;
  });
}

/// Run one task through `dual_process`, then let the chunk store see System 2's result. Returns the answer.
inline std::string run_dual_chunks(Interpreter& it, ChunkStore& cs, DualTask& t, DualTask*& cur, WorkingMemory& wm,
                                   const RunOptions& opt = {}) {
  cur = &t;
  it.run("dual_process", "chunks", wm, opt);
  if (t.route == "slow") cs.observe_slow(t.cls, t.slow_result);
  cur = nullptr;
  return t.route == "fast" ? t.fast_result : t.slow_result;
}

}  // namespace cypha::thought
