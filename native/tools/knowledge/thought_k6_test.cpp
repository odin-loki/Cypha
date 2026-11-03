// CTest native_thought_k6_test: chunking (K6), process mining and MDL edits (K7), and the calibration map of DIF
// applicability (HANDOFF 5.23). docs/reports/K6_K7_CHUNKING_AND_MINING.md section 4.
//
// Usage: thought_k6_test PROCESSES_JSON
//
// Author: Odin Loch
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "cypha/thought/bindings_cypha.hpp"
#include "cypha/thought/chunking.hpp"
#include "cypha/thought/mining.hpp"

using namespace cypha::thought;
namespace bd = cypha::thought::bind;
namespace mn = cypha::thought::mine;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond)                                                              \
  do {                                                                           \
    ++g_checks;                                                                  \
    if (!(cond)) {                                                               \
      ++g_failed;                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
    }                                                                            \
  } while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::abs((a) - (b)) <= (tol))

// ---------------------------------------------------------------------------
// K6: the chunk rules
// ---------------------------------------------------------------------------

static void test_chunk_rules() {
  ChunkStore cs;
  for (int i = 0; i < 4; ++i) cs.observe_slow("k", "r");
  CHECK(cs.lookup("k") == nullptr);  // 4 identical: not yet
  cs.observe_slow("k", "r");
  CHECK(cs.lookup("k") != nullptr && cs.lookup("k")->status == ChunkStatus::Tainted && cs.lookup("k")->result == "r");

  ChunkStore c2;
  for (const char* x : {"a", "a", "a", "a", "b"}) c2.observe_slow("k", x);
  CHECK(c2.lookup("k") == nullptr);  // not identical
  for (int i = 0; i < 4; ++i) c2.observe_slow("k", "b");
  CHECK(c2.lookup("k") != nullptr && c2.lookup("k")->result == "b");

  // keep: the first n with Wilson lower bound(n / n) >= 0.9, and n >= 20
  int nstar = 0;
  for (int n = 1; n < 200; ++n)
    if (wilson_interval(n, n).lo >= 0.9) {
      nstar = n;
      break;
    }
  CHECK(nstar >= 20);
  for (int i = 0; i < nstar - 1; ++i) cs.check("k", true);
  CHECK(cs.lookup("k")->status == ChunkStatus::Tainted);
  cs.check("k", true);
  CHECK(cs.lookup("k")->status == ChunkStatus::Kept);

  // drop: agreement over the last 20 below 0.8 (5 disagreements); then re-entry needs 10 identical
  for (int i = 0; i < 4; ++i) cs.check("k", false);
  CHECK(cs.lookup("k") != nullptr);  // 16 / 20 = 0.8: kept
  cs.check("k", false);
  CHECK(cs.lookup("k") == nullptr);  // 15 / 20 < 0.8: dropped
  for (int i = 0; i < 9; ++i) cs.observe_slow("k", "s");
  CHECK(cs.lookup("k") == nullptr);
  cs.observe_slow("k", "s");
  CHECK(cs.lookup("k") != nullptr && cs.lookup("k")->result == "s");

  // barred re-entry
  ChunkRules br;
  br.reentry_n = 0;
  ChunkStore c3(br);
  for (int i = 0; i < 5; ++i) c3.observe_slow("k", "r");
  for (int i = 0; i < 20; ++i) c3.check("k", false);
  CHECK(c3.lookup("k") == nullptr);
  for (int i = 0; i < 50; ++i) c3.observe_slow("k", "r");
  CHECK(c3.lookup("k") == nullptr);

  // a tainted rule is dropped too
  ChunkStore c4;
  for (int i = 0; i < 5; ++i) c4.observe_slow("k", "r");
  for (int i = 0; i < 20; ++i) c4.check("k", i % 2 == 0);
  CHECK(c4.lookup("k") == nullptr);
}

// ---------------------------------------------------------------------------
// K6: dual_process through the interpreter, audits, rule 10
// ---------------------------------------------------------------------------

static void test_dual_process(const std::shared_ptr<const Library>& lib) {
  for (double audit : {0.0, 1.0}) {
    ChunkRules R;
    R.audit = audit;
    ChunkStore cs(R);
    Interpreter it(lib, Config{}, 7);
    DualTask* cur = nullptr;
    Rng ar(3);
    bind_dual_chunks(it, cs, cur, ar);
    int fast = 0, slow = 0;
    for (int i = 0; i < 80; ++i) {
      DualTask t;
      t.cls = "k";
      t.system2 = []() { return std::make_pair(std::string("r"), 10.0); };
      WorkingMemory wm;
      const std::string ans = run_dual_chunks(it, cs, t, cur, wm);
      CHECK(ans == "r");
      (t.route == "fast" ? fast : slow)++;
    }
    if (audit == 0.0) {
      CHECK(fast > 0 && cs.lookup("k") && cs.lookup("k")->status == ChunkStatus::Kept);
    } else {
      CHECK(fast == 0 && slow == 80);  // every use audited: System 2 always answers
    }
  }
  // rule 10: in answer mode a kept rule's answer becomes an "I don't know" card
  ChunkRules R;
  R.audit = 0.0;
  ChunkStore cs(R);
  Interpreter it(lib, Config{}, 9);
  DualTask* cur = nullptr;
  Rng ar(5);
  bind_dual_chunks(it, cs, cur, ar);
  bd::Toolkit tk;
  Step say;
  say.id = "say";
  say.op = "communicate";
  int s1 = 0, cards = 0;
  for (int i = 0; i < 80; ++i) {
    DualTask t;
    t.cls = "k";
    t.system2 = []() { return std::make_pair(std::string("r"), 10.0); };
    WorkingMemory wm;
    wm.set("mode", Json("answer"));
    run_dual_chunks(it, cs, t, cur, wm);
    if (t.route == "fast") {
      ++s1;
      StepContext ctx;
      const StepResult r = bd::do_communicate(tk, say, ctx, wm);
      if (r.value.value("kind", std::string()) == "idk") ++cards;
    }
  }
  CHECK(s1 > 0 && cards == s1);
}

// ---------------------------------------------------------------------------
// K7: the miner
// ---------------------------------------------------------------------------

static mn::Log repeat(const mn::Trace& t, int n) { return mn::Log(static_cast<std::size_t>(n), t); }

static void test_miner() {
  mn::Counts c;
  c.df[{"a", "b"}] = 10;
  c.df[{"b", "a"}] = 2;
  CHECK_NEAR(mn::dependency(c, "a", "b"), 8.0 / 13.0, 1e-12);
  c.df[{"a", "a"}] = 9;
  CHECK_NEAR(mn::dependency(c, "a", "a"), 0.9, 1e-12);
  // amendment 2's M2: one stray reverse count ignored
  mn::Counts c2;
  c2.df[{"a", "b"}] = 26;
  c2.df[{"b", "a"}] = 1;
  CHECK_NEAR(mn::dependency(c2, "a", "b"), 25.0 / 28.0, 1e-12);
  CHECK_NEAR(mn::dependency(c2, "a", "b", 1), 26.0 / 27.0, 1e-12);
  mn::Log noisy = repeat({"a", "b", "c"}, 26);
  noisy.push_back({"b", "a", "c"});
  CHECK(!mn::heuristics_miner(noisy).count({"a", "b"}));
  CHECK(mn::heuristics_miner(noisy, mn::miner_variant("M2")).count({"a", "b"}));
  CHECK(mn::miner_variant("M1").dep == 0.8 && mn::miner_variant("M0").noise_floor == 0);

  const mn::EdgeSet e = mn::heuristics_miner(repeat({"a", "b", "c"}, 20));
  CHECK(e == (mn::EdgeSet{{"^", "a"}, {"a", "b"}, {"b", "c"}, {"c", "$"}}));
  CHECK(mn::heuristics_miner(repeat({"a", "b", "c"}, 9)).empty());  // counts below 10
  // a length-2 loop: a b a b c
  const mn::EdgeSet l = mn::heuristics_miner(repeat({"a", "b", "a", "b", "c"}, 20));
  CHECK(l.count({"a", "b"}) && l.count({"b", "a"}) && l.count({"b", "c"}));
  // a self-loop
  const mn::EdgeSet s = mn::heuristics_miner(repeat({"a", "a", "a", "b"}, 10));
  CHECK(s.count({"a", "a"}) && s.count({"a", "b"}));
}

// ---------------------------------------------------------------------------
// K7: description length, edits, promotion
// ---------------------------------------------------------------------------

static void test_mdl() {
  mn::Model m;
  const int A = m.add_node({"A"});
  m.add_edge(mn::Model::kStartNode, A);
  m.add_edge(A, mn::Model::kEndNode);
  const mn::Log log = repeat({"A"}, 4);
  mn::MdlParams P;
  P.w = 0.5;
  P.alphabet = mn::alphabet_of(log);
  CHECK_NEAR(P.alphabet, 3.0, 1e-12);
  // trace bits: 8 transitions with p = (4 + 0.5) / (4 + 0.5 * 2) = 0.9
  const double tb = 8 * -std::log2(0.9);
  // model bits: one label log2 3; two edges 2 log2(3 + 2) each; params 0.5 log2(4) for start and A
  const double mb = std::log2(3.0) + 2 * 2 * std::log2(5.0) + 0.5 * 2 + 0.5 * 2;
  CHECK_NEAR(mn::description_length(m, log, P), tb + mb, 1e-9);

  // an escape: trace "A B" with B unknown to the model
  const mn::Log log2 = {{"A", "B"}};
  const mn::Replay r = mn::replay(m, log2);
  CHECK(r.esc.count(A) && r.esc.at(A) == 1.0);

  // remove an unused edge lowers DL; add an edge for a frequent escape lowers DL
  mn::Model m2 = m;
  const int B = m2.add_node({"B"});
  m2.add_edge(A, B);
  m2.add_edge(B, mn::Model::kEndNode);
  mn::MdlParams P2;
  P2.alphabet = 4;
  CHECK(mn::description_length(m, log, P2) < mn::description_length(m2, log, P2));  // B unused: simpler wins
  const mn::Log logab = repeat({"A", "B"}, 30);
  mn::Model m3 = m;
  m3.add_node({"B"});  // B exists but A -> B is missing
  m3.add_edge(3, mn::Model::kEndNode);
  const mn::Replay r3 = mn::replay(m3, logab);
  const auto cand = mn::candidate_edits(m3, r3);
  bool has_add = false;
  for (const mn::Edit& e : cand)
    if (e.kind == mn::EditKind::AddEdge && e.u == A && e.v == 3) has_add = true;
  CHECK(has_add);
  const auto steps = mn::edit_search(m3, logab, P2);
  CHECK(!steps.empty() && steps.front().after.has_edge(A, 3));
}

/// A label used in two contexts: a x b and c x d. The planted model has two x nodes; the collapsed one has one.
static void test_split_and_merge() {
  mn::Model planted;
  const int a = planted.add_node({"a"}), x1 = planted.add_node({"x"}), b = planted.add_node({"b"});
  const int c = planted.add_node({"c"}), x2 = planted.add_node({"x"}), d = planted.add_node({"d"});
  planted.add_edge(0, a);
  planted.add_edge(0, c);
  planted.add_edge(a, x1);
  planted.add_edge(x1, b);
  planted.add_edge(b, 1);
  planted.add_edge(c, x2);
  planted.add_edge(x2, d);
  planted.add_edge(d, 1);
  mn::Log log;
  for (int i = 0; i < 60; ++i) log.push_back(i % 2 ? mn::Trace{"a", "x", "b"} : mn::Trace{"c", "x", "d"});
  const mn::Model col = mn::collapse(planted, x1, x2);
  CHECK(col.nodes.size() == planted.nodes.size() - 1);
  mn::MdlParams P;
  P.alphabet = mn::alphabet_of(log);
  CHECK(mn::description_length(planted, log, P) < mn::description_length(col, log, P));
  const auto steps = mn::edit_search(col, log, P);
  CHECK(!steps.empty());
  const mn::Model& fin = steps.back().after;
  CHECK(mn::trigrams(fin) == mn::trigrams(planted) || std::any_of(steps.begin(), steps.end(), [](const mn::EditStep& s) {
          return s.edit.kind == mn::EditKind::Split;
        }));
  bool split = false;
  for (const auto& s : steps) split = split || s.edit.kind == mn::EditKind::Split;
  CHECK(split);
  // promotion: held-out traces from the planted process
  mn::Log held;
  for (int i = 0; i < 300; ++i) held.push_back(i % 2 ? mn::Trace{"a", "x", "b"} : mn::Trace{"c", "x", "d"});
  const cypha::eval::LciDecision dec = mn::promote_lci(col, fin, log, held, P);
  CHECK(dec.lci >= 0.95);

  // merge: p q always together
  mn::Model mm;
  const int p = mm.add_node({"p"}), q = mm.add_node({"q"});
  mm.add_edge(0, p);
  mm.add_edge(p, q);
  mm.add_edge(q, 1);
  const mn::Log lpq = repeat({"p", "q"}, 50);
  const auto cand = mn::candidate_edits(mm, mn::replay(mm, lpq));
  bool has_merge = false;
  for (const mn::Edit& e : cand) has_merge = has_merge || (e.kind == mn::EditKind::Merge && e.u == p && e.v == q);
  CHECK(has_merge);
  mn::MdlParams Pm;
  Pm.alphabet = mn::alphabet_of(lpq);
  const mn::Model merged = mn::apply_merge(mm, p, q);
  CHECK(merged.nodes.size() == 3 && merged.nodes[2].labels == (std::vector<std::string>{"p", "q"}));
  CHECK(mn::description_length(merged, lpq, Pm) < mn::description_length(mm, lpq, Pm));
  CHECK(mn::trigrams(merged) == mn::trigrams(mm));  // the same structure, expanded
}

/// Planted processes and defects: the generator keeps the automaton deterministic; repairs are read on trigrams.
static void test_planted() {
  for (std::uint64_t s = 1; s <= 20; ++s) {
    Rng r(s);
    const mn::Planted P = mn::make_planted(r, 8, 2);
    for (int u = 0; u < static_cast<int>(P.model.nodes.size()); ++u) {
      std::set<std::string> first;
      double ps = 0;
      for (int v : P.model.nodes[static_cast<std::size_t>(u)].out) {
        first.insert(P.model.nodes[static_cast<std::size_t>(v)].labels.front());
        ps += P.prob.at({u, v});
      }
      CHECK(first.size() == P.model.nodes[static_cast<std::size_t>(u)].out.size());
      if (!P.model.nodes[static_cast<std::size_t>(u)].out.empty()) CHECK_NEAR(ps, 1.0, 1e-9);
    }
  }
  Rng r(77);
  const mn::Planted P = mn::make_planted(r, 8, 0);
  std::map<std::pair<int, int>, int> wc;
  mn::make_log(P, r, 200, 0.0, &wc);
  mn::Model m = P.model;
  mn::Defect d;
  CHECK(mn::apply_defect(P.model, m, mn::DefectKind::SpuriousEdge, r, wc, d));
  CHECK(!d.delta.empty());
  CHECK(!mn::repaired(d, mn::trigrams(P.model), mn::trigrams(m)));
  CHECK(mn::repaired(d, mn::trigrams(P.model), mn::trigrams(P.model)));
}

// ---------------------------------------------------------------------------
// The calibration map (HANDOFF 5.23)
// ---------------------------------------------------------------------------

/// A base model whose P for a process moves to the label it has just learned: a map fitted after the label would see
/// the learned P, not the P that was used.
class StepModel : public ApplicabilityModel {
 public:
  ApplicabilityQuery query(const std::string&, const std::string&, const WorkingMemory&) override {
    ApplicabilityQuery q;
    q.p = p_;
    return q;
  }
  void label(const std::string&, const std::string&, const WorkingMemory&, bool positive) override { p_ = positive ? 0.9 : 0.1; }

 private:
  double p_ = 0.5;
};

static void test_calibration() {
  auto base = std::make_shared<StepModel>();
  bd::CalibratedApplicability cal(base, bd::CalibratedApplicability::Map::Isotonic, 4, 2);
  WorkingMemory wm;
  cal.label("p", "c", wm, true);   // P before: 0.5
  cal.label("p", "c", wm, false);  // P before: 0.9
  cal.label("p", "c", wm, true);   // P before: 0.1
  CHECK(cal.raw_seen().size() == 3);
  CHECK_NEAR(cal.raw_seen()[0], 0.5, 1e-12);
  CHECK_NEAR(cal.raw_seen()[1], 0.9, 1e-12);
  CHECK_NEAR(cal.raw_seen()[2], 0.1, 1e-12);
  CHECK(!cal.fitted());
  cal.label("p", "c", wm, false);
  CHECK(cal.fitted());

  // isotonic on overconfident raw P: truth = raw / 2
  Rng r(11);
  std::vector<double> raw;
  std::vector<int> y;
  for (int i = 0; i < 20000; ++i) {
    const double p = uniform01(r);
    raw.push_back(p);
    y.push_back(uniform01(r) < p / 2 ? 1 : 0);
  }
  cypha::eval::IsotonicCalibrator iso;
  iso.fit(raw, y);
  CHECK_NEAR(iso.predict(0.6), 0.3, 0.05);
  CHECK_NEAR(iso.predict(0.2), 0.1, 0.05);
  // Platt recovers a planted map sigmoid(2 logit(p) - 1)
  std::vector<double> raw2;
  std::vector<int> y2;
  for (int i = 0; i < 20000; ++i) {
    const double p = 0.02 + 0.96 * uniform01(r);
    const double t = 1.0 / (1.0 + std::exp(-(2.0 * bd::CalibratedApplicability::logit(p) - 1.0)));
    raw2.push_back(p);
    y2.push_back(uniform01(r) < t ? 1 : 0);
  }
  double a = 0, b = 0;
  bd::CalibratedApplicability::platt_fit(raw2, y2, a, b);
  CHECK_NEAR(a, 2.0, 0.1);
  CHECK_NEAR(b, -1.0, 0.1);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: thought_k6_test PROCESSES_JSON\n");
    return 2;
  }
  const auto lib = std::make_shared<const Library>(Library::load(argv[1]));
  test_chunk_rules();
  test_dual_process(lib);
  test_miner();
  test_mdl();
  test_split_and_merge();
  test_planted();
  test_calibration();
  std::printf("thought_k6_test: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
