// K6 and K7: chunking, process mining and MDL structural edits on their benchmarks (docs/THOUGHT.md sections 4, 6,
// 12; docs/reports/K6_K7_CHUNKING_AND_MINING.md). The calibration map of DIF applicability is `thought_k5 calibration`.
//
//   thought_k6 chunk PROCESSES_JSON [--read-test --audit A --reentry R]   K6: a stream of diagnosis tasks
//   thought_k6 mine PROCESSES_JSON [--read-test --miner M0|M1|M2]        K7a: recovery from synthetic logs; the library
//   thought_k6 edits PROCESSES_JSON [--read-test --w W --variant E0|E1|E2]  K7b: MDL edits, edit precision, LCI promotion
//
// Dev seeds 1-3 (chunk streams) and 1-50 (planted processes) for choices; held-out seeds 1001-1003 and 1001-1100,
// read once with --read-test. Rule 10: nothing here answers a question; chunk answers in answer mode become
// "I don't know" cards.
//
// Author: Odin Loch
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "cypha/thought/bindings.hpp"
#include "cypha/thought/chunking.hpp"
#include "cypha/thought/mining.hpp"

using namespace cypha::thought;
namespace bd = cypha::thought::bind;
namespace mn = cypha::thought::mine;

static double mean(const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }

static std::pair<double, double> boot_ci(const std::vector<double>& v, std::uint64_t seed = 20261029, int B = 2000) {
  if (v.empty()) return {0, 0};
  Rng r(seed);
  std::vector<double> m(static_cast<std::size_t>(B));
  for (int b = 0; b < B; ++b) {
    double s = 0;
    for (std::size_t i = 0; i < v.size(); ++i) s += v[static_cast<std::size_t>(uniform01(r) * v.size()) % v.size()];
    m[static_cast<std::size_t>(b)] = s / v.size();
  }
  std::sort(m.begin(), m.end());
  return {m[static_cast<std::size_t>(0.025 * B)], m[static_cast<std::size_t>(0.975 * B)]};
}

static std::string wilson_text(double k, double n) {
  const WilsonInterval w = wilson_interval(k, n);
  char buf[96];
  std::snprintf(buf, sizeof buf, "%.4f [%.4f, %.4f] (%g / %g)", n > 0 ? k / n : 0.0, w.lo, w.hi, k, n);
  return buf;
}

// ===========================================================================
// K6: a stream of diagnosis tasks with recurring situation classes
// ===========================================================================

enum Kind { kDet = 0, kMostly = 1, kAmbig = 2, kShift = 3 };
static const char* kKindName[4] = {"deterministic", "mostly", "ambiguous", "shifting"};

struct ClassSpec {
  int kind = kDet;
  int r = 0, r2 = 0;
};

struct TaskRec {
  int cls = 0;
  int truth = 0;
  int s2 = 0;  // System 2's result (the truth, or a wrong result with probability 0.02)
};

struct StreamData {
  std::vector<ClassSpec> classes;
  std::vector<TaskRec> tasks;
};

static constexpr int kClasses = 80, kResults = 16, kTasks = 4000;

static StreamData make_stream(std::uint64_t seed) {
  StreamData S;
  Rng rc(seed * 7919 + 1);
  std::vector<int> kinds;
  for (int i = 0; i < 40; ++i) kinds.push_back(kDet);
  for (int i = 0; i < 16; ++i) kinds.push_back(kMostly);
  for (int i = 0; i < 16; ++i) kinds.push_back(kAmbig);
  for (int i = 0; i < 8; ++i) kinds.push_back(kShift);
  for (int i = kClasses - 1; i > 0; --i) std::swap(kinds[static_cast<std::size_t>(i)], kinds[static_cast<std::size_t>(uniform01(rc) * (i + 1))]);
  for (int c = 0; c < kClasses; ++c) {
    ClassSpec cs;
    cs.kind = kinds[static_cast<std::size_t>(c)];
    cs.r = static_cast<int>(uniform01(rc) * kResults);
    cs.r2 = (cs.r + 1 + static_cast<int>(uniform01(rc) * (kResults - 1))) % kResults;
    S.classes.push_back(cs);
  }
  std::vector<int> order(kClasses);
  std::iota(order.begin(), order.end(), 0);
  for (int i = kClasses - 1; i > 0; --i) std::swap(order[static_cast<std::size_t>(i)], order[static_cast<std::size_t>(uniform01(rc) * (i + 1))]);
  std::vector<double> cum;
  double tot = 0;
  for (int i = 0; i < kClasses; ++i) cum.push_back(tot += 1.0 / std::pow(i + 1.0, 1.1));
  Rng rt(seed * 31 + 7), rs(seed * 131 + 11);
  for (int t = 0; t < kTasks; ++t) {
    const double u = uniform01(rt) * tot;
    const int pos = static_cast<int>(std::lower_bound(cum.begin(), cum.end(), u) - cum.begin());
    TaskRec tr;
    tr.cls = order[static_cast<std::size_t>(std::min(pos, kClasses - 1))];
    const ClassSpec& cs = S.classes[static_cast<std::size_t>(tr.cls)];
    const double v = uniform01(rt);
    const int other = (cs.r + 1 + static_cast<int>(uniform01(rt) * (kResults - 1))) % kResults;
    switch (cs.kind) {
      case kDet: tr.truth = cs.r; break;
      case kMostly: tr.truth = v < 0.92 ? cs.r : other; break;
      case kAmbig: tr.truth = v < 0.6 ? cs.r : cs.r2; break;
      case kShift: tr.truth = t < kTasks / 2 ? cs.r : cs.r2; break;
    }
    const bool wrong = uniform01(rs) < 0.02;
    const int alt = (tr.truth + 1 + static_cast<int>(uniform01(rs) * (kResults - 1))) % kResults;
    tr.s2 = wrong ? alt : tr.truth;
    S.tasks.push_back(tr);
  }
  return S;
}

static std::string res_name(int r) { return "f" + std::to_string(r); }

struct ArmStats {
  std::vector<double> wrong;  // per task: 1 when the answer is not the truth
  double s1 = 0, s1_right = 0, cost = 0;
  int kept_any[4] = {0, 0, 0, 0}, compiled[4] = {0, 0, 0, 0}, dropped[4] = {0, 0, 0, 0}, classes[4] = {0, 0, 0, 0};
  std::vector<double> shift_delay;  // per shifting class kept at the switch: its tasks until the drop (-1: never)
  int answer_mode_s1 = 0, answer_mode_cards = 0;
};

/// The `none` arm: System 2 answers every task.
static ArmStats run_none(const StreamData& S) {
  ArmStats a;
  for (const TaskRec& t : S.tasks) {
    a.wrong.push_back(t.s2 != t.truth ? 1.0 : 0.0);
    a.cost += 10.0;
  }
  return a;
}

/// The `naive` ablation: compile after 1 result, never drop, no audits.
static ArmStats run_naive(const StreamData& S) {
  ArmStats a;
  std::map<int, int> rule;
  for (const TaskRec& t : S.tasks) {
    const auto it = rule.find(t.cls);
    int ans;
    if (it != rule.end()) {
      ans = it->second;
      a.s1 += 1;
      a.s1_right += ans == t.truth ? 1 : 0;
      a.cost += 1.0;
    } else {
      ans = t.s2;
      rule[t.cls] = t.s2;
      a.cost += 10.0;
    }
    a.wrong.push_back(ans != t.truth ? 1.0 : 0.0);
  }
  return a;
}

/// The `chunk` arm: THOUGHT 4's rules through the library's dual_process (and, in answer mode, the rule-10 guard).
static ArmStats run_chunk(const std::shared_ptr<const Library>& lib, const StreamData& S, std::uint64_t seed,
                          const ChunkRules& rules, bool answer_mode = false) {
  ArmStats a;
  Config cfg;
  Interpreter it(lib, cfg, seed);
  ChunkStore cs(rules);
  DualTask* cur = nullptr;
  Rng audit_rng(seed * 977 + 5);
  bind_dual_chunks(it, cs, cur, audit_rng);
  bd::Toolkit tk;
  Step comm_step;
  comm_step.id = "say";
  comm_step.op = "communicate";
  std::set<int> kept_at_switch;
  std::map<int, int> since_switch;
  for (std::size_t i = 0; i < S.tasks.size(); ++i) {
    const TaskRec& tr = S.tasks[i];
    if (static_cast<int>(i) == kTasks / 2) {  // the shifting classes' switch
      for (int c = 0; c < kClasses; ++c)
        if (S.classes[static_cast<std::size_t>(c)].kind == kShift) {
          const Chunk* ch = cs.lookup("c" + std::to_string(c));
          if (ch && ch->status == ChunkStatus::Kept) kept_at_switch.insert(c);
        }
    }
    DualTask t;
    t.cls = "c" + std::to_string(tr.cls);  // the situation class key
    const int s2 = tr.s2;
    t.system2 = [s2]() { return std::make_pair(res_name(s2), 10.0); };
    WorkingMemory wm;
    if (answer_mode) wm.set("mode", Json("answer"));
    const std::string ans = run_dual_chunks(it, cs, t, cur, wm);
    const bool right = ans == res_name(tr.truth);
    a.wrong.push_back(right ? 0.0 : 1.0);
    if (t.route == "fast") {
      a.s1 += 1;
      a.s1_right += right ? 1 : 0;
      a.cost += 1.0;
    } else {
      a.cost += 10.0 + (t.had_rule ? 1.0 : 0.0);
    }
    if (answer_mode && t.route == "fast") {
      ++a.answer_mode_s1;
      StepContext ctx;
      const StepResult r = bd::do_communicate(tk, comm_step, ctx, wm);
      if (r.value.value("kind", std::string()) == "idk") ++a.answer_mode_cards;
    }
    if (static_cast<int>(i) >= kTasks / 2 && kept_at_switch.count(tr.cls)) {  // the shifting class's delay to the drop
      int& n = since_switch[tr.cls];
      if (n >= 0) {
        ++n;
        const Chunk* ch = cs.lookup(t.cls);
        if (!ch || ch->result != res_name(S.classes[static_cast<std::size_t>(tr.cls)].r)) {
          a.shift_delay.push_back(n);
          n = -1000000;  // done
        }
      }
    }
  }
  for (int c : kept_at_switch) {
    const auto itn = since_switch.find(c);
    if (itn == since_switch.end() || itn->second >= 0) a.shift_delay.push_back(-1);
  }
  for (int c = 0; c < kClasses; ++c) {
    const int k = S.classes[static_cast<std::size_t>(c)].kind;
    ++a.classes[k];
    const auto itc = cs.classes().find("c" + std::to_string(c));
    if (itc == cs.classes().end()) continue;
    a.compiled[k] += itc->second.compiled > 0;
    a.kept_any[k] += itc->second.kept > 0;
    a.dropped[k] += itc->second.dropped > 0;
  }
  return a;
}

static void print_arm(const char* name, const ArmStats& a, double n) {
  std::printf("| %s | %.4f | %.4f | %s | %.3f |\n", name, a.s1 / n, mean(a.wrong), a.s1 > 0 ? wilson_text(a.s1_right, a.s1).c_str() : "-",
              a.cost / n);
}

static int cmd_chunk(const std::string& lib_path, bool read_test, double audit, int reentry) {
  auto lib = std::make_shared<const Library>(Library::load(lib_path));
  if (!read_test) {
    std::printf("## K6 dev (streams 1-3, %d tasks each): audit share x re-entry after a drop\n\n", kTasks);
    std::printf("| audit | re-entry | System 1 hit rate | wrong rate | wrong - none | chunk accuracy |\n|---|---|---:|---:|---:|---|\n");
    std::vector<double> none_w;
    std::vector<StreamData> streams;
    for (std::uint64_t s : {1ULL, 2ULL, 3ULL}) {
      streams.push_back(make_stream(s));
      const ArmStats n = run_none(streams.back());
      none_w.insert(none_w.end(), n.wrong.begin(), n.wrong.end());
    }
    double best_hit = -1;
    struct Opt { double audit; int reentry; double hit, wrong; };
    std::vector<Opt> feas;
    for (double au : {0.05, 0.10, 0.20})
      for (int re : {0, 10}) {
        ChunkRules R;
        R.audit = au;
        R.reentry_n = re;
        std::vector<double> w;
        double s1 = 0, s1r = 0, n = 0;
        for (std::size_t k = 0; k < streams.size(); ++k) {
          const ArmStats a = run_chunk(lib, streams[k], 1 + k, R);
          w.insert(w.end(), a.wrong.begin(), a.wrong.end());
          s1 += a.s1;
          s1r += a.s1_right;
          n += static_cast<double>(a.wrong.size());
        }
        const double hit = s1 / n, wr = mean(w), dw = wr - mean(none_w);
        std::printf("| %.2f | %s | %.4f | %.4f | %+.4f | %s |\n", au, re ? "10 identical" : "barred", hit, wr, dw, wilson_text(s1r, s1).c_str());
        if (dw <= 0.005) {
          feas.push_back({au, re, hit, wr});
          best_hit = std::max(best_hit, hit);
        }
      }
    const Opt* pick = nullptr;
    for (const Opt& o : feas)
      if (o.hit >= best_hit - 0.01 && (!pick || o.wrong < pick->wrong - 1e-12)) pick = &o;
    std::printf("\nnone: wrong rate %.4f\n", mean(none_w));
    if (pick) std::printf("DEV CHOICE: audit %.2f, re-entry %s\n", pick->audit, pick->reentry ? "10" : "barred (0)");
    else std::printf("DEV CHOICE: none feasible\n");
    return 0;
  }
  ChunkRules R;
  R.audit = audit;
  R.reentry_n = reentry;
  std::printf("## K6 held-out (streams 1001-1003, read once): audit %.2f, re-entry %d\n\n", audit, reentry);
  std::printf("| arm | System 1 hit rate | wrong rate | chunk accuracy (Wilson) | cost per task |\n|---|---:|---:|---|---:|\n");
  ArmStats none_all, chunk_all, naive_all;
  std::vector<double> diff, diff_naive;
  int kept_any[4] = {0, 0, 0, 0}, compiled[4] = {0, 0, 0, 0}, dropped[4] = {0, 0, 0, 0}, classes[4] = {0, 0, 0, 0};
  std::vector<double> delays;
  for (std::uint64_t s : {1001ULL, 1002ULL, 1003ULL}) {
    const StreamData S = make_stream(s);
    const ArmStats n = run_none(S), c = run_chunk(lib, S, s, R), v = run_naive(S);
    for (std::size_t i = 0; i < n.wrong.size(); ++i) {
      diff.push_back(c.wrong[i] - n.wrong[i]);
      diff_naive.push_back(v.wrong[i] - n.wrong[i]);
    }
    auto acc = [](ArmStats& to, const ArmStats& f) {
      to.wrong.insert(to.wrong.end(), f.wrong.begin(), f.wrong.end());
      to.s1 += f.s1;
      to.s1_right += f.s1_right;
      to.cost += f.cost;
    };
    acc(none_all, n);
    acc(chunk_all, c);
    acc(naive_all, v);
    for (int k = 0; k < 4; ++k) {
      kept_any[k] += c.kept_any[k];
      compiled[k] += c.compiled[k];
      dropped[k] += c.dropped[k];
      classes[k] += c.classes[k];
    }
    delays.insert(delays.end(), c.shift_delay.begin(), c.shift_delay.end());
  }
  const double N = static_cast<double>(none_all.wrong.size());
  print_arm("none", none_all, N);
  print_arm("chunk", chunk_all, N);
  print_arm("naive (ablation)", naive_all, N);
  const auto ci = boot_ci(diff), cin = boot_ci(diff_naive);
  std::printf("\nWrong answers, chunk - none (paired per task): %+.4f [%+.4f, %+.4f]\n", mean(diff), ci.first, ci.second);
  std::printf("Wrong answers, naive - none: %+.4f [%+.4f, %+.4f]\n", mean(diff_naive), cin.first, cin.second);
  std::printf("\n| class kind | classes | ever compiled | ever kept | ever dropped |\n|---|---:|---:|---:|---:|\n");
  for (int k = 0; k < 4; ++k) std::printf("| %s | %d | %d | %d | %d |\n", kKindName[k], classes[k], compiled[k], kept_any[k], dropped[k]);
  std::printf("\nShifting classes kept at the switch: %zu; tasks of the class until the drop:", delays.size());
  for (double d : delays) std::printf(" %g", d);
  std::printf("\n");
  const double chunk_acc = chunk_all.s1 > 0 ? chunk_all.s1_right / chunk_all.s1 : 0.0;
  const bool gate = chunk_acc >= 0.9 && chunk_all.s1 > 0 && ci.second <= 0.005;
  std::printf("\nGATE (accuracy >= 0.9, System 1 hit rate above none's 0, wrong difference upper <= +0.005): %s\n", gate ? "MET" : "NOT MET");
  // answer mode: every System 1 answer must become an "I don't know" card
  const StreamData S = make_stream(1001);
  const ArmStats am = run_chunk(lib, S, 1001, R, true);
  std::printf("Answer mode (stream 1001): System 1 answers %d, \"I don't know\" cards %d\n", am.answer_mode_s1, am.answer_mode_cards);
  return 0;
}

// ===========================================================================
// K7a: recovery from synthetic logs, and the library
// ===========================================================================

struct Recov {
  int exact = 0, cases = 0;
  double tp = 0, mined = 0, target = 0, found = 0, spurious = 0;
};

/// Exact recovery: no mined edge outside the planted process, and every planted edge traversed >= 10 times mined.
static bool recovered(const mn::EdgeSet& mined, const mn::EdgeSet& planted, const mn::EdgeSet& target, Recov& R) {
  bool ok = true;
  for (const auto& e : mined) {
    R.mined += 1;
    if (planted.count(e)) R.tp += 1;
    else {
      R.spurious += 1;
      ok = false;
    }
  }
  for (const auto& e : target) {
    R.target += 1;
    if (!mined.count(e)) ok = false;
    else R.found += 1;
  }
  return ok;
}

static mn::EdgeSet label_edges_counted(const mn::Planted& P, const std::map<std::pair<int, int>, int>& wc, int min_count) {
  mn::EdgeSet e;
  for (const auto& [uv, n] : wc)
    if (n >= min_count)
      e.insert({P.model.nodes[static_cast<std::size_t>(uv.first)].labels.back(),
                P.model.nodes[static_cast<std::size_t>(uv.second)].labels.front()});
  return e;
}

static void mine_planted_range(std::uint64_t from, std::uint64_t to, Recov& R, const mn::MinerRules& rules = {}) {
  for (std::uint64_t seed = from; seed <= to; ++seed) {
    Rng r(seed * 104729 + 3);
    const int nl = 6 + static_cast<int>(uniform01(r) * 7);
    const mn::Planted P = mn::make_planted(r, nl, 0);
    std::map<std::pair<int, int>, int> wc;
    const mn::Log log = mn::make_log(P, r, 200, 0.05, &wc);
    const mn::EdgeSet mined = mn::heuristics_miner(log, rules);
    const mn::EdgeSet planted = P.model.label_edges();
    const mn::EdgeSet target = label_edges_counted(P, wc, 10);
    ++R.cases;
    if (recovered(mined, planted, target, R)) ++R.exact;
  }
}

static void print_recov(const char* name, const Recov& R) {
  const double prec = R.mined > 0 ? R.tp / R.mined : 0.0, rec = R.target > 0 ? R.found / R.target : 0.0;
  const double f1 = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
  std::printf("| %s | %s | %.4f | %.4f | %.4f | %g |\n", name, wilson_text(R.exact, R.cases).c_str(), prec, rec, f1, R.spurious);
}

/// A library process as a planted model: one node per step (labelled by its id), uniform branch choice.
static bool library_planted(const Process& p, mn::Planted& P) {
  std::map<std::string, int> id;
  for (const auto& [sid, s] : p.steps) id[sid] = P.model.add_node({sid});
  if (!id.count(p.start)) return false;
  P.model.add_edge(mn::Model::kStartNode, id.at(p.start));
  for (const auto& [sid, s] : p.steps) {
    if (s.next.empty()) P.model.add_edge(id.at(sid), mn::Model::kEndNode);
    for (const std::string& n : s.next)
      if (id.count(n)) P.model.add_edge(id.at(sid), id.at(n));
  }
  for (int u = 0; u < static_cast<int>(P.model.nodes.size()); ++u) {
    const auto& o = P.model.nodes[static_cast<std::size_t>(u)].out;
    for (int v : o) P.prob[{u, v}] = 1.0 / static_cast<double>(o.size());
  }
  return true;
}

static int cmd_mine(const std::string& lib_path, bool read_test, const std::string& miner) {
  const Library lib = Library::load(lib_path);
  std::printf("## K7a: recovery (heuristics miner with loop measures; 200 traces, 5%% noise)\n\n");
  std::printf("| set | exact recovery (Wilson) | edge precision | recall (edges >= 10) | F1 | spurious edges |\n|---|---|---:|---:|---:|---:|\n");
  std::map<std::string, Recov> dev;
  for (const char* m : {"M0", "M1", "M2"}) {  // amendment 2: the dev choice
    mine_planted_range(1, 50, dev[m], mn::miner_variant(m));
    print_recov((std::string("dev planted 1-50, ") + m).c_str(), dev[m]);
  }
  std::string pick = "M0";
  int best = -1;
  for (const char* m : {"M0", "M2", "M1"}) best = std::max(best, dev[m].exact);
  double fewest = 1e18;
  for (const char* m : {"M0", "M2", "M1"})  // within 2 of the best: fewer spurious edges; then M0, M2, M1
    if (dev[m].exact >= best - 2 && dev[m].spurious < fewest - 1e-12) {
      fewest = dev[m].spurious;
      pick = m;
    }
  std::printf("\nDEV CHOICE (amendment 2): %s\n\n", pick.c_str());
  if (read_test) {
    std::printf("| set | exact recovery (Wilson) | edge precision | recall (edges >= 10) | F1 | spurious edges |\n|---|---|---:|---:|---:|---:|\n");
    Recov h0, hm;
    mine_planted_range(1001, 1100, h0, mn::miner_variant("M0"));
    print_recov("held-out planted 1001-1100 (read once), M0 (THOUGHT)", h0);
    if (miner != "M0") {
      mine_planted_range(1001, 1100, hm, mn::miner_variant(miner));
      print_recov(("held-out planted 1001-1100 (read once), " + miner).c_str(), hm);
    } else {
      hm = h0;
    }
    std::printf("\nGATE K7a on %s (exact recovery in >= 80%% of held-out processes): %s\n", miner.c_str(),
                hm.exact >= 0.8 * hm.cases ? "MET" : "NOT MET");
  }
  // the library: step ids as labels, uniform branch choice, no noise
  Recov L;
  std::vector<std::string> missed;
  int idx = 0;
  for (const Process& p : lib.processes()) {
    mn::Planted P;
    if (!library_planted(p, P)) continue;
    Rng r(500000 + static_cast<std::uint64_t>(idx++));
    std::map<std::pair<int, int>, int> wc;
    const mn::Log log = mn::make_log(P, r, 200, 0.0, &wc);
    const mn::EdgeSet mined = mn::heuristics_miner(log, mn::miner_variant(miner.empty() ? "M0" : miner));
    ++L.cases;
    if (recovered(mined, P.model.label_edges(), label_edges_counted(P, wc, 10), L)) ++L.exact;
    else missed.push_back(p.id);
  }
  print_recov(("library (144 processes, report only), " + (miner.empty() ? std::string("M0") : miner)).c_str(), L);
  std::printf("\nLibrary processes not recovered exactly (%zu):", missed.size());
  for (const std::string& m : missed) std::printf(" %s", m.c_str());
  std::printf("\n");
  return 0;
}

// ===========================================================================
// K7b: MDL structural edits, edit precision on held-out traces, LCI promotion
// ===========================================================================

struct EditsStats {
  double accepted = 0, improved = 0;
  std::map<std::string, std::pair<double, double>> by_kind;  // kind -> (accepted, improved)
  double defects = 0, repaired = 0;
  std::map<std::string, std::pair<double, double>> defect_by_kind;
  int cases = 0, edited = 0, promoted = 0, promoted_loss = 0;
  double gap_num = 0, gap_den = 0;  // pooled: sum (H(start) - H(final)) / sum (H(start) - H(planted)), held-out
};

static void run_edit_case(std::uint64_t seed, double w, EditsStats& E, const std::string& variant = "E0", bool verbose = false) {
  Rng r(seed * 15485863 + 9);
  const int nl = 6 + static_cast<int>(uniform01(r) * 7);
  const int reuse = static_cast<int>(std::lround(0.2 * nl));
  const mn::Planted P = mn::make_planted(r, nl, reuse);
  std::map<std::pair<int, int>, int> wc;
  const mn::Log train = mn::make_log(P, r, 200, 0.05, &wc);
  const mn::Log held = mn::make_log(P, r, 300, 0.05);
  mn::Model start = P.model;
  std::vector<mn::Defect> defects;
  std::vector<mn::DefectKind> kinds;
  for (int k = 0; k < 2; ++k) kinds.push_back(static_cast<mn::DefectKind>(static_cast<int>(uniform01(r) * 3)));
  std::stable_sort(kinds.begin(), kinds.end(), [](mn::DefectKind a, mn::DefectKind b) {
    return (a == mn::DefectKind::Collapse) < (b == mn::DefectKind::Collapse);  // a collapse renumbers: last
  });
  for (mn::DefectKind kind : kinds) {
    mn::Defect d;
    if (mn::apply_defect(P.model, start, kind, r, wc, d)) defects.push_back(d);
  }
  mn::MdlParams prm;
  prm.w = w;
  prm.alphabet = mn::alphabet_of(train);
  if (variant == "E1") prm.edge_factor = 1.0;
  std::vector<mn::EditStep> steps;
  if (variant == "E2") {  // search on traces 0-149, validate on 150-199
    const mn::Log search(train.begin(), train.begin() + 150), valid(train.begin() + 150, train.end());
    steps = mn::edit_search(start, search, prm, 30, &valid);
  } else {
    steps = mn::edit_search(start, train, prm);
  }
  ++E.cases;
  mn::Model prev = start;
  auto hsum = [&](const mn::Model& m) {
    const std::vector<double> v = mn::heldout_bits(m, train, held, prm);
    return std::accumulate(v.begin(), v.end(), 0.0);
  };
  double h_prev = hsum(prev);
  const double h_start = h_prev, h_planted = hsum(P.model);
  for (const mn::EditStep& s : steps) {
    const double h = hsum(s.after);
    const bool imp = h < h_prev - 1e-9;
    E.accepted += 1;
    E.improved += imp ? 1 : 0;
    auto& bk = E.by_kind[mn::edit_name(s.edit.kind)];
    bk.first += 1;
    bk.second += imp ? 1 : 0;
    if (verbose) std::printf("    %s: train DL %.1f -> %.1f, held-out bits %.1f -> %.1f\n", s.edit.text.c_str(), s.dl_before, s.dl_after, h_prev, h);
    h_prev = h;
    prev = s.after;
  }
  const mn::Model& fin = steps.empty() ? start : steps.back().after;
  const auto tri_p = mn::trigrams(P.model), tri_f = mn::trigrams(fin);
  for (const mn::Defect& d : defects) {
    const bool ok = mn::repaired(d, tri_p, tri_f);
    E.defects += 1;
    E.repaired += ok ? 1 : 0;
    auto& dk = E.defect_by_kind[mn::defect_name(d.kind)];
    dk.first += 1;
    dk.second += ok ? 1 : 0;
  }
  E.gap_num += h_start - h_prev;
  E.gap_den += h_start - h_planted;
  if (!steps.empty()) {
    ++E.edited;
    const cypha::eval::LciDecision d = mn::promote_lci(start, fin, train, held, prm);
    if (d.lci >= 0.95) {
      ++E.promoted;
      if (h_prev >= h_start - 1e-9) ++E.promoted_loss;
    }
  }
}

static void print_edits(const char* name, const EditsStats& E) {
  std::printf("\n### %s (%d cases)\n\n", name, E.cases);
  std::printf("- Edit precision (accepted edits that improve held-out bits): %s\n", wilson_text(E.improved, E.accepted).c_str());
  for (const auto& [k, v] : E.by_kind) std::printf("  - %s: %s\n", k.c_str(), wilson_text(v.second, v.first).c_str());
  std::printf("- Planted defects repaired: %s\n", wilson_text(E.repaired, E.defects).c_str());
  for (const auto& [k, v] : E.defect_by_kind) std::printf("  - %s: %s\n", k.c_str(), wilson_text(v.second, v.first).c_str());
  std::printf("- Held-out gap closed (start to planted), pooled: %.4f (bits saved %.1f of %.1f)\n", E.gap_den != 0 ? E.gap_num / E.gap_den : 0.0,
              E.gap_num, E.gap_den);
  std::printf("- Cases edited %d; promoted by LCI %d; promoted with a held-out loss %d\n", E.edited, E.promoted, E.promoted_loss);
}

static int cmd_edits(const std::string& lib_path, bool read_test, double w, const std::string& variant) {
  (void)lib_path;
  std::printf("## K7b: MDL edits (2 planted defects per case; 200 training, 300 held-out traces; 5%% noise)\n");
  if (!read_test) {
    double best = -1, best_w = 0.5, best_rep = -1;
    for (double ww : {0.5, 1.0}) {
      EditsStats E;
      for (std::uint64_t s = 1; s <= 50; ++s) run_edit_case(s, ww, E);
      char nm[64];
      std::snprintf(nm, sizeof nm, "dev seeds 1-50, E0, w = %.1f", ww);
      print_edits(nm, E);
      const double prec = E.accepted > 0 ? E.improved / E.accepted : 0.0;
      const double rep = E.defects > 0 ? E.repaired / E.defects : 0.0;
      if (prec > best + 0.01 || (std::abs(prec - best) <= 0.01 && rep > best_rep)) {
        best = prec;
        best_w = ww;
        best_rep = rep;
      }
    }
    std::printf("\nDEV CHOICE (registered): w = %.1f\n", best_w);
    // amendment 3: E0 / E1 / E2 at w = 0.5
    std::map<std::string, std::pair<double, double>> res;  // precision, repaired share
    for (const char* v : {"E1", "E2"}) {
      EditsStats E;
      for (std::uint64_t s = 1; s <= 50; ++s) run_edit_case(s, 0.5, E, v);
      print_edits((std::string("dev seeds 1-50, ") + v + ", w = 0.5").c_str(), E);
      res[v] = {E.accepted > 0 ? E.improved / E.accepted : 0.0, E.defects > 0 ? E.repaired / E.defects : 0.0};
    }
    {
      EditsStats E;
      for (std::uint64_t s = 1; s <= 50; ++s) run_edit_case(s, 0.5, E, "E0");
      res["E0"] = {E.accepted > 0 ? E.improved / E.accepted : 0.0, E.defects > 0 ? E.repaired / E.defects : 0.0};
    }
    double bp = -1;
    for (const auto& [v, pr] : res) bp = std::max(bp, pr.first);
    std::string pick;
    double brep = -1;
    for (const char* v : {"E0", "E1", "E2"})
      if (res[v].first >= bp - 0.02 && res[v].second > brep + 1e-12) {
        brep = res[v].second;
        pick = v;
      }
    std::printf("\nDEV CHOICE (amendment 3): %s (precision E0 %.4f, E1 %.4f, E2 %.4f)\n", pick.c_str(), res["E0"].first,
                res["E1"].first, res["E2"].first);
    return 0;
  }
  EditsStats E;
  for (std::uint64_t s = 1001; s <= 1100; ++s) run_edit_case(s, w, E, variant);
  char nm[96];
  std::snprintf(nm, sizeof nm, "held-out seeds 1001-1100 (read once), %s, w = %.1f", variant.c_str(), w);
  print_edits(nm, E);
  const double prec = E.accepted > 0 ? E.improved / E.accepted : 0.0;
  std::printf("\nGATE K7b (edit precision >= 0.9; no promoted model with a held-out loss): %s\n",
              prec >= 0.9 && E.promoted_loss == 0 ? "MET" : "NOT MET");
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: thought_k6 chunk|mine|edits PROCESSES_JSON [--read-test] [--audit A --reentry R] [--miner M] [--w W --variant V]\n");
    return 2;
  }
  const std::string cmd = argv[1], lib = argv[2];
  bool rt = false;
  double audit = 0.10, w = 0.5;
  int reentry = 10;
  std::string miner = "M0", variant = "E0";
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--read-test") rt = true;
    else if (a == "--audit" && i + 1 < argc) audit = std::atof(argv[++i]);
    else if (a == "--reentry" && i + 1 < argc) reentry = std::atoi(argv[++i]);
    else if (a == "--w" && i + 1 < argc) w = std::atof(argv[++i]);
    else if (a == "--miner" && i + 1 < argc) miner = argv[++i];
    else if (a == "--variant" && i + 1 < argc) variant = argv[++i];
  }
  if (cmd == "chunk") return cmd_chunk(lib, rt, audit, reentry);
  if (cmd == "mine") return cmd_mine(lib, rt, miner);
  if (cmd == "edits") return cmd_edits(lib, rt, w, variant);
  std::fprintf(stderr, "unknown command\n");
  return 2;
}
