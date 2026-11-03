// K4: known-answer checks of the thought primitives bound to Cypha's libraries (bindings.hpp), the rule-10 guard, the
// applicability hook, and the library with every op bound (docs/reports/K4_K5_THOUGHT_BINDINGS.md).
//
// Usage: thought_bindings_test <docs/thought/processes.json>
//
// Author: Odin Loch
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "cypha/cyphalm/fm_infinigram.hpp"
#include "cypha/thought/bindings.hpp"

using namespace cypha::thought;
namespace bd = cypha::thought::bind;
namespace ev = cypha::eval;

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

static Step mk(const std::string& op, std::vector<std::string> next = {}) {
  Step s;
  s.id = op;
  s.op = op;
  s.next = std::move(next);
  return s;
}

static std::string corpus() {
  std::string t;
  for (int i = 0; i < 60; ++i) t += "a cat sat on the mat. a dog lay by one rug. ";
  return t;
}

// ---------------------------------------------------------------------------
// CyphaLM bindings: perceive / read, recall_case (with the ∞-gram), generate
// ---------------------------------------------------------------------------

static void test_cyphalm(bd::Toolkit& tk, const std::string& text) {
  StepContext ctx;
  Rng rng(7);
  ctx.rng = &rng;
  {  // perceive: in-distribution text is far less surprising than random letters
    WorkingMemory a, b;
    a.set("text", std::string("a cat sat on the mat. "));
    b.set("text", std::string("qzx vkj wpq frt glm. "));
    const StepResult ra = bd::do_perceive(tk, mk("perceive"), ctx, a);
    const StepResult rb = bd::do_perceive(tk, mk("read"), ctx, b);
    CHECK(ra.ok && rb.ok);
    const double ba = a.get("surprise")->at("bpc").get<double>(), bb = b.get("surprise")->at("bpc").get<double>();
    CHECK(std::isfinite(ba) && std::isfinite(bb));
    CHECK(ba < 1.0);
    CHECK(bb > 4.0);
    CHECK(a.get("surprise")->at("n").get<int>() == 22);
    // the binding's number is the model's own: bits of each byte under the mixture
    const bd::Surprise s = bd::surprise(tk.lm, "", "a cat sat on the mat. ");
    CHECK_NEAR(ba, s.bpc, 1e-12);
  }
  {  // recall_case: CyphaLM's greedy continuation, its confidence and the corpus match
    WorkingMemory wm;
    wm.set("prompt", std::string("a cat sat on the "));
    const StepResult r = bd::do_recall(tk, mk("recall_case"), ctx, wm);
    CHECK(r.ok);
    const Json& rc = *wm.get("recall");
    CHECK(rc.at("text").get<std::string>().rfind("mat.", 0) == 0);
    CHECK(rc.at("source").get<std::string>() == "cyphalm");
    CHECK(rc.at("mean_lp").get<double>() > -0.5);
    CHECK(rc.at("match_n").get<std::size_t>() == 17);  // the whole prompt occurs
    WorkingMemory un;
    un.set("prompt", std::string("purple elephants "));
    bd::do_recall(tk, mk("recall_case"), ctx, un);
    CHECK(un.get("recall")->at("match_n").get<std::size_t>() < 4);
    CHECK(un.get("recall")->at("prompt_bpc").get<double>() > rc.at("prompt_bpc").get<double>() + 2.0);
    bd::Toolkit none;
    WorkingMemory w2;
    w2.set("prompt", std::string("x"));
    const StepResult rn = bd::do_recall(none, mk("recall_case"), ctx, w2);
    CHECK(!rn.ok && rn.impasse);  // no CyphaLM, no System 1
  }
  {  // generate: seeded, CyphaLM writes every byte
    WorkingMemory a, b, c;
    for (WorkingMemory* w : {&a, &b, &c}) w->set("prompt", std::string("a "));
    Rng r1(11), r2(11), r3(12);
    StepContext c1, c2, c3;
    c1.rng = &r1;
    c2.rng = &r2;
    c3.rng = &r3;
    bd::do_generate(tk, mk("generate"), c1, a);
    bd::do_generate(tk, mk("generate"), c2, b);
    bd::do_generate(tk, mk("generate"), c3, c);
    const std::string ga = a.get("generated")->at("text").get<std::string>();
    CHECK(ga.size() == static_cast<std::size_t>(tk.gen_bytes));
    CHECK(ga == b.get("generated")->at("text").get<std::string>());
    CHECK(ga != c.get("generated")->at("text").get<std::string>());
    CHECK(a.get("generated")->at("source").get<std::string>() == "cyphalm");
    WorkingMemory d;  // without a prompt: the candidates pass through
    d.set("candidates", Json::array({"h1", "h2"}));
    bd::Toolkit none;
    bd::do_generate(none, mk("generate"), c1, d);
    CHECK(d.get("generated")->size() == 2);
  }
  (void)text;
}

// ---------------------------------------------------------------------------
// knowledge: retrieve, score / evaluate, classify, communicate and the rule-10 guard
// ---------------------------------------------------------------------------

static void test_knowledge() {
  bd::Toolkit tk;
  StepContext ctx;
  tk.facts["capital|Algeria"] = ev::opinion_from_counts(12.0, 0.0);
  {  // retrieve: an opinion; in answer mode it may only gate
    WorkingMemory d, a, m;
    d.set("query", std::string("capital|Algeria"));
    a.set("query", std::string("capital|Algeria"));
    a.set("mode", std::string("answer"));
    m.set("query", std::string("capital|Atlantis"));
    CHECK(bd::do_retrieve(tk, mk("retrieve"), ctx, d).ok);
    CHECK(bd::do_retrieve(tk, mk("retrieve"), ctx, a).ok);
    CHECK(!d.get("opinion")->at("gate_only").get<bool>());
    CHECK(a.get("opinion")->at("gate_only").get<bool>());
    CHECK_NEAR(d.get("opinion")->at("P").get<double>(), tk.facts["capital|Algeria"].P, 1e-12);
    CHECK(!bd::do_retrieve(tk, mk("retrieve"), ctx, m).ok);
  }
  {  // score: ECS fusion, the same opinion as fuse()
    WorkingMemory wm;
    wm.set("evidence", Json::array({{{"stance", "support"}, {"trust_a", 20.0}, {"trust_b", 1.0}, {"extraction", 0.9}},
                                    {{"stance", "support"}, {"trust_a", 50.0}, {"trust_b", 1.0}, {"extraction", 1.0}},
                                    {{"stance", "oppose"}, {"trust_a", 1.0}, {"trust_b", 3.0}, {"extraction", 0.8}}}));
    CHECK(bd::do_score(tk, mk("score"), ctx, wm).ok);
    std::vector<ev::EvidenceItem> it = {ev::make_evidence(ev::Stance::Support, {20.0, 1.0}, 0.9),
                                        ev::make_evidence(ev::Stance::Support, {50.0, 1.0}, 1.0),
                                        ev::make_evidence(ev::Stance::Oppose, {1.0, 3.0}, 0.8)};
    const ev::Opinion o = ev::fuse(it);
    CHECK_NEAR(wm.get("opinion")->at("P").get<double>(), o.P, 1e-12);
    CHECK_NEAR(wm.get("opinion")->at("d").get<double>(), o.d, 1e-12);
    CHECK(wm.get("opinion")->at("grade").get<std::string>() == ev::grade_name(o.grade));
    WorkingMemory none;
    CHECK(!bd::do_score(tk, mk("evaluate"), ctx, none).ok);
  }
  {  // classify: the epistemic state of an opinion, and an attached classifier
    WorkingMemory v, k, c;
    v.set("opinion", bd::opinion_json(ev::opinion_from_counts(0.0, 0.0)));
    bd::do_classify(tk, mk("classify"), ctx, v);
    CHECK(v.str("state") == "Vacuous");
    const ev::Opinion strong = ev::opinion_from_counts(40.0, 0.0);
    k.set("opinion", bd::opinion_json(strong));
    bd::do_classify(tk, mk("classify"), ctx, k);
    CHECK(k.str("state") == ev::state_name(ev::classify(strong)));
    CHECK(k.str("state") == "Known");
    c.set("opinion", bd::opinion_json(ev::opinion_from_counts(2.0, 9.0)));
    bd::do_classify(tk, mk("classify"), ctx, c);
    CHECK(c.str("state") == "Contested");
    bd::Toolkit tc;
    tc.classifier = [](const std::vector<double>& x) {
      bd::ClassifierOut o;
      o.label = x[0] > 0 ? "pos" : "neg";
      o.confidence = 0.8;
      o.ood = std::fabs(x[0]) > 10;
      return o;
    };
    WorkingMemory f, g;
    f.set("features", Json::array({1.0, 2.0}));
    g.set("features", Json::array({50.0, 2.0}));
    CHECK(bd::do_classify(tc, mk("classify"), ctx, f).ok);
    CHECK(f.get("class")->at("label").get<std::string>() == "pos");
    CHECK(!bd::do_classify(tc, mk("classify"), ctx, g).ok);  // OOD
    CHECK(g.get("class")->at("ood").get<bool>());
  }
  {  // communicate: cards, reason codes, and rule 10
    bd::Toolkit t2;
    WorkingMemory a;
    a.set("mode", std::string("answer"));
    a.set("answer", Json{{"text", "Algiers"}, {"source", "cyphalm"}, {"confidence", 0.93}});
    bd::do_communicate(t2, mk("communicate"), ctx, a);
    CHECK(a.get("card")->at("kind").get<std::string>() == "answer");
    CHECK(a.get("card")->at("text").get<std::string>() == "Algiers");
    WorkingMemory b;
    b.set("mode", std::string("answer"));
    b.set("answer", Json{{"text", "Algiers"}, {"source", "factstore"}, {"confidence", 0.99}});
    bd::do_communicate(t2, mk("communicate"), ctx, b);
    CHECK(b.get("card")->at("kind").get<std::string>() == "idk");
    CHECK(b.get("card")->value("note", std::string()) == "rule10");
    CHECK(b.get("card")->at("text").get<std::string>().find("<|unknown reason=NO_EVIDENCE|>") != std::string::npos);
    CHECK(t2.rule10_refusals == 1);
    WorkingMemory c;  // discovery mode: working text, not an answer
    c.set("answer", Json{{"text", "a hypothesis"}, {"source", "G2"}});
    bd::do_communicate(t2, mk("explain"), ctx, c);
    CHECK(c.get("card")->at("kind").get<std::string>() == "answer");
    WorkingMemory d;
    d.set("mode", std::string("answer"));
    d.set("idk", true);
    d.set("reason", std::string("UNKNOWN_ENTITY"));
    bd::do_communicate(t2, mk("communicate"), ctx, d);
    CHECK(d.get("card")->at("reason").get<std::string>() == "UNKNOWN_ENTITY");
    CHECK(d.get("card")->at("text").get<std::string>().find("<|unknown reason=UNKNOWN_ENTITY|>") != std::string::npos);
    WorkingMemory e;
    e.set("report", Json{{"top", "H2"}});
    bd::do_communicate(t2, mk("communicate"), ctx, e);
    CHECK(e.get("card")->at("kind").get<std::string>() == "report");
    // K4b: a planted bypass, 100 answers from the fact store in answer mode: every one becomes an IDK card
    bd::Toolkit t3;
    int idk = 0;
    for (int i = 0; i < 100; ++i) {
      WorkingMemory w;
      w.set("mode", std::string("answer"));
      w.set("answer", Json{{"text", "v" + std::to_string(i)}, {"source", i % 2 ? "factstore" : "retrieval"}});
      bd::do_communicate(t3, mk("communicate"), ctx, w);
      idk += w.get("card")->at("kind").get<std::string>() == "idk";
    }
    CHECK(idk == 100);
    CHECK(t3.rule10_refusals == 100);
  }
}

// ---------------------------------------------------------------------------
// hypothesis tests, updates, decisions, LCI
// ---------------------------------------------------------------------------

static void test_hypothesis_and_decisions() {
  bd::Toolkit tk;
  StepContext ctx;
  {  // two-way: s = c = 0.9, positive: BF 9, 0.5 -> 0.9
    WorkingMemory wm;
    wm.set("outcome", std::string("positive"));
    wm.set("test", Json{{"s", 0.9}, {"c", 0.9}});
    wm.set("posterior", 0.5);
    const StepResult r = bd::do_test(tk, mk("test"), ctx, wm);
    CHECK(r.ok);
    CHECK_NEAR(wm.num("bf"), 9.0, 1e-9);
    CHECK_NEAR(wm.num("posterior"), 0.9, 1e-9);
    CHECK(r.gain && std::fabs(*r.gain - 0.4) < 1e-9);
    wm.set("outcome", std::string("negative"));
    const StepResult n = bd::do_test(tk, mk("verify"), ctx, wm);
    CHECK(!n.ok);
    CHECK_NEAR(wm.num("posterior"), 0.5, 1e-9);  // 0.9 with BF 1/9
    const StepResult f = bd::do_test(tk, mk("falsify"), ctx, wm);
    CHECK(f.ok);  // the disconfirmation search found evidence against
  }
  {  // S7b three-way likelihoods: inconclusive has BF t[2]/f[2]
    WorkingMemory wm;
    wm.set("outcome", std::string("inconclusive"));
    wm.set("test", Json{{"t", {0.6, 0.1, 0.3}}, {"f", {0.05, 0.6, 0.35}}});
    wm.set("posterior", 0.5);
    bd::do_test(tk, mk("test"), ctx, wm);
    CHECK_NEAR(wm.num("bf"), 0.3 / 0.35, 1e-12);
    CHECK_NEAR(wm.num("posterior"), ev::update_posterior(0.5, 0.3 / 0.35), 1e-12);
    WorkingMemory nw;  // no hypothesis wiring: a plain check
    nw.set("check", false);
    CHECK(!bd::do_test(tk, mk("test"), ctx, nw).ok);
  }
  {  // update: a Beta and a pending Bayes factor
    WorkingMemory wm;
    wm.set("beta", Json{{"a", 2.0}, {"b", 1.0}});
    wm.set("result", 1.0);
    bd::do_update(tk, mk("update"), ctx, wm);
    CHECK_NEAR(wm.get("beta")->at("a").get<double>(), 3.0, 0);
    WorkingMemory w2;
    w2.set("posterior", 0.5);
    w2.set("bf", 3.0);
    w2.set("pending_bf", true);
    bd::do_update(tk, mk("update"), ctx, w2);
    CHECK_NEAR(w2.num("posterior"), 0.75, 1e-12);
    CHECK(!w2.has("pending_bf"));
  }
  {  // decide: a guard, the threshold factory, the LCI factory
    WorkingMemory wm;
    wm.set("guard", std::string("slow"));
    const StepResult r = bd::do_decide(tk, mk("decide", {"answer", "slow"}), ctx, wm);
    CHECK(r.chosen_next && *r.chosen_next == "slow");
    CHECK(!wm.has("guard"));
    wm.set("guard", std::string("nowhere"));
    CHECK(!bd::do_decide(tk, mk("decide", {"answer", "slow"}), ctx, wm).chosen_next);
    const Primitive th = bd::decide_threshold("conf", 0.8, "answer", "idk");
    WorkingMemory w1, w2;
    w1.set("conf", 0.85);
    w2.set("conf", 0.5);
    CHECK(*th(mk("decide"), ctx, w1).chosen_next == "answer");
    CHECK(*th(mk("decide"), ctx, w2).chosen_next == "idk");
    const Primitive lc = bd::decide_lci("promote", "hold", 0.0);
    WorkingMemory good, flat;
    Json gd = Json::array(), fd = Json::array();
    for (int i = 0; i < 400; ++i) {
      gd.push_back(0.05 + 0.01 * ((i % 5) - 2));
      fd.push_back(0.02 * ((i % 3) - 1));
    }
    good.set("lci_diffs", gd);
    good.set("lci_gain_dev", 0.05);
    good.set("lci_gain_ext", 0.05);
    flat.set("lci_diffs", fd);
    flat.set("lci_gain_dev", 0.05);
    flat.set("lci_gain_ext", 0.0);
    CHECK(*lc(mk("decide"), ctx, good).chosen_next == "promote");
    CHECK(good.num("lci") >= 0.95);
    CHECK(*lc(mk("decide"), ctx, flat).chosen_next == "hold");
  }
  {  // monitor: CUSUM, k = 0.5, h = 5; in control it stays at 0; a +2 shift alarms on the 4th observation
    WorkingMemory wm;
    for (int i = 0; i < 50; ++i) {
      wm.set("obs", 0.0);
      bd::do_monitor(tk, mk("monitor"), ctx, wm);
    }
    CHECK(!wm.get("alarm")->get<bool>());
    CHECK_NEAR(wm.get("cusum")->at("S").get<double>(), 0.0, 0);
    int at = -1;
    for (int i = 1; i <= 10 && at < 0; ++i) {
      wm.set("obs", 2.0);
      bd::do_monitor(tk, mk("monitor"), ctx, wm);
      if (wm.get("alarm")->get<bool>()) at = i;
    }
    CHECK(at == 4);  // S = 1.5, 3, 4.5, 6
  }
}

// ---------------------------------------------------------------------------
// select, rank, estimate, combine, transform, compare, infer, ask, record, defaults
// ---------------------------------------------------------------------------

static void test_misc() {
  bd::Toolkit tk;
  Rng rng(3);
  StepContext ctx;
  ctx.rng = &rng;
  {  // select among options, by Thompson draws, and from the gap map by VOI
    WorkingMemory wm;
    wm.set("options", Json::array({{{"id", "a"}, {"score", 0.2}}, {{"id", "b"}, {"score", 0.9}}, {{"id", "c"}, {"score", 0.5}}}));
    bd::do_select(tk, mk("select"), ctx, wm);
    CHECK(wm.get("chosen")->at("id").get<std::string>() == "b");
    bd::do_rank(tk, mk("rank"), ctx, wm);
    CHECK(wm.get("ranked")->at(0).at("id").get<std::string>() == "b");
    CHECK(wm.get("ranked")->at(2).at("id").get<std::string>() == "a");
    int first = 0;
    for (int i = 0; i < 200; ++i) {
      WorkingMemory t;
      t.set("options", Json::array({{{"id", "good"}, {"a", 100.0}, {"b", 1.0}}, {{"id", "bad"}, {"a", 1.0}, {"b", 100.0}}}));
      bd::do_select(tk, mk("select"), ctx, t);
      first += t.get("chosen")->at("id").get<std::string>() == "good";
    }
    CHECK(first == 200);
    ev::GapMap gm;
    gm.open(ev::GapKind::Schema, "Algeria", "capital").p = 0.5;
    gm.open(ev::GapKind::Schema, "Chad", "capital").p = 0.98;
    ev::Gap& g3 = gm.open(ev::GapKind::Schema, "Peru", "capital");
    g3.p = 0.5;
    g3.impact = 5.0;
    gm.score();
    bd::Toolkit tg;
    tg.gaps = &gm;
    WorkingMemory gw;
    CHECK(bd::do_select(tg, mk("select"), ctx, gw).ok);
    CHECK(gw.get("chosen")->at("entity").get<std::string>() == gm.ranked().front()->entity);
  }
  {  // estimate: Wilson 8 / 10 -> (0.490162, 0.943318)
    WorkingMemory wm;
    wm.set("counts", Json{{"k", 8}, {"n", 10}});
    bd::do_estimate(tk, mk("estimate"), ctx, wm);
    CHECK_NEAR(wm.get("estimate")->at("lo").get<double>(), 0.490162, 1e-6);
    CHECK_NEAR(wm.get("estimate")->at("hi").get<double>(), 0.943318, 1e-6);
  }
  {  // combine: lognormal 90% intervals multiply in log space
    WorkingMemory a, b;
    a.set("factors", Json::array({{{"lo", 1.0}, {"hi", 100.0}}}));
    bd::do_combine(tk, mk("combine"), ctx, a);
    CHECK_NEAR(a.get("estimate")->at("median").get<double>(), 10.0, 1e-9);
    CHECK_NEAR(a.get("estimate")->at("lo").get<double>(), 1.0, 1e-9);
    CHECK_NEAR(a.get("estimate")->at("hi").get<double>(), 100.0, 1e-9);
    b.set("factors", Json::array({{{"lo", 10.0}, {"hi", 10.0}}, {{"lo", 1.0}, {"hi", 100.0}}, {{"lo", 1.0}, {"hi", 100.0}}}));
    bd::do_combine(tk, mk("combine"), ctx, b);
    CHECK_NEAR(b.get("estimate")->at("median").get<double>(), 1000.0, 1e-6);
    CHECK_NEAR(std::log10(b.get("estimate")->at("hi").get<double>() / 1000.0), std::sqrt(2.0) * 2.0 / 2.0 * 1.0, 1e-9);
  }
  {  // transform: isotonic calibration
    ev::IsotonicCalibrator cal;
    cal.fit({0.1, 0.2, 0.3, 0.8, 0.9}, {0, 0, 1, 1, 1});
    bd::Toolkit tc;
    tc.calibrator = &cal;
    WorkingMemory wm;
    wm.set("raw", 0.9);
    bd::do_transform(tc, mk("transform"), ctx, wm);
    CHECK_NEAR(wm.num("calibrated"), cal.predict(0.9), 1e-12);
    CHECK_NEAR(wm.num("calibrated"), 1.0, 1e-12);
  }
  {  // compare: cosine of mean word vectors; numbers
    ev::WordVectors wv;
    wv.set({"cat", "dog", "car"}, {10, 10, 10}, {1, 0, 0, 0.8f, 0.6f, 0, 0, 0, 1}, 3, false);
    bd::Toolkit tv;
    tv.vectors = &wv;
    WorkingMemory a, b, c;
    a.set("a", std::string("cat"));
    a.set("b", std::string("cat"));
    b.set("a", std::string("cat"));
    b.set("b", std::string("dog"));
    c.set("a", std::string("cat"));
    c.set("b", std::string("car"));
    bd::do_compare(tv, mk("compare"), ctx, a);
    bd::do_compare(tv, mk("compare"), ctx, b);
    bd::do_compare(tv, mk("match"), ctx, c);
    CHECK_NEAR(a.num("similarity"), 1.0, 1e-6);
    CHECK_NEAR(b.num("similarity"), 0.8, 1e-6);
    CHECK_NEAR(c.num("similarity"), 0.0, 1e-6);
    WorkingMemory n;
    n.set("a", 8.0);
    n.set("b", 10.0);
    bd::do_compare(tk, mk("compare"), ctx, n);
    CHECK_NEAR(n.num("similarity"), 0.8, 1e-12);
  }
  {  // infer: derivation factor, depth cap 3
    WorkingMemory a, b;
    a.set("premises", Json::array({0.9, 0.8}));
    a.set("rule_reliability", 0.9);
    a.set("depth", 2);
    CHECK(bd::do_infer(tk, mk("infer"), ctx, a).ok);
    CHECK_NEAR(a.num("derived"), 0.648, 1e-12);
    b.set("premises", Json::array({0.9}));
    b.set("depth", 4);
    CHECK(!bd::do_infer(tk, mk("infer"), ctx, b).ok);
    CHECK(b.str("reason") == "BEYOND_DEPTH");
    CHECK_NEAR(b.num("derived"), 0.0, 0);
  }
  {  // ask: queued, never sent; record: the journal
    bd::Toolkit tq;
    const auto path = (std::filesystem::temp_directory_path() / "thought_ask_test.jsonl").string();
    std::filesystem::remove(path);
    tq.ask_path = path;
    StepContext c2;
    c2.process = "impasse";
    c2.step = "choose";
    WorkingMemory wm;
    wm.set("question", std::string("Is Brussels the City of Brussels?"));
    bd::do_ask(tq, mk("ask"), c2, wm);
    CHECK(tq.ask_queue.size() == 1);
    CHECK(tq.ask_queue[0].at("from").get<std::string>() == "impasse.choose");
    CHECK(tq.ask_queue[0].at("status").get<std::string>() == "pending");
    std::ifstream f(path);
    std::string line;
    CHECK(std::getline(f, line) && line.find("City of Brussels") != std::string::npos);
    std::filesystem::remove(path);
    wm.set("last_value", 3);
    bd::do_record(tq, mk("record"), c2, wm);
    CHECK(tq.journal.size() == 1 && tq.journal[0].at("value").get<int>() == 3);
    CHECK_NEAR(bd::do_default(tq, mk("simulate"), c2, wm).cost, 1.0, 0);
  }
}

// ---------------------------------------------------------------------------
// the library with every op bound, end-to-end runs, and the applicability hook
// ---------------------------------------------------------------------------

struct FixedApp : ApplicabilityModel {
  std::map<std::string, double> p;
  std::map<std::string, bool> ood;
  std::vector<std::pair<std::string, bool>> labels;
  ApplicabilityQuery query(const std::string& pid, const std::string&, const WorkingMemory&) override {
    ApplicabilityQuery q;
    q.p = p.count(pid) ? p[pid] : 0.5;
    q.ood = ood.count(pid) ? ood[pid] : false;
    return q;
  }
  void label(const std::string& pid, const std::string&, const WorkingMemory&, bool positive) override {
    labels.push_back({pid, positive});
  }
};

static void test_library(const std::string& path, bd::Toolkit& tk) {
  auto lib = std::make_shared<const Library>(Library::load(path));
  CHECK(lib->processes().size() >= 144);
  CHECK(lib->validate().empty());
  Interpreter it(lib, Config{}, 5);
  bd::bind_all(it, tk);
  std::size_t with_unbound = 0;
  for (const Process& p : lib->processes()) with_unbound += !it.unbound(p.id).empty();
  CHECK(with_unbound == 0);  // K4a
  std::printf("bound: %zu processes, %zu with an unbound op\n", lib->processes().size(), with_unbound);

  // every process runs to an end with the default toolkit (no step errors, no unbound ops)
  std::size_t done = 0, errors = 0;
  for (const Process& p : lib->processes()) {
    WorkingMemory wm;
    RunOptions opt;
    opt.budget = 200;
    const RunResult r = it.run(p.id, "smoke", wm, opt);
    errors += r.trace.outcome == Outcome::Error;
    done += r.trace.outcome == Outcome::Done || r.trace.outcome == Outcome::Budget || r.trace.outcome == Outcome::Voc ||
            r.trace.outcome == Outcome::MaxSteps;
  }
  CHECK(errors == 0);
  std::printf("smoke: %zu of %zu processes ran to an end with the bound toolkit, %zu errors\n", done,
              lib->processes().size(), errors);

  {  // cooperative_reply end to end, answer mode: CyphaLM's answer goes out, a fact-store answer does not
    WorkingMemory a, b;
    for (WorkingMemory* w : {&a, &b}) {
      w->set("mode", std::string("answer"));
      w->set("check", true);
    }
    a.set("answer", Json{{"text", "mat"}, {"source", "cyphalm"}, {"confidence", 0.9}});
    b.set("answer", Json{{"text", "mat"}, {"source", "factstore"}, {"confidence", 0.9}});
    const RunResult ra = it.run("cooperative_reply", "qa", a);
    const RunResult rb = it.run("cooperative_reply", "qa", b);
    CHECK(ra.trace.outcome == Outcome::Done && rb.trace.outcome == Outcome::Done);
    CHECK(a.get("card")->at("kind").get<std::string>() == "answer");
    CHECK(b.get("card")->at("kind").get<std::string>() == "idk");
  }
  {  // dual_process: System 1 is CyphaLM's recall; the gate (bound to its confidence) answers or goes slow
    Interpreter i2(lib, Config{}, 9);
    bd::bind_all(i2, tk);
    i2.bind("dual_process.gate", bd::decide_threshold("fast_conf", -1.0, "answer", "slow"));
    i2.bind("dual_process.fast", [&tk](const Step& s, StepContext& c, WorkingMemory& wm) {
      StepResult r = bd::do_recall(tk, s, c, wm);
      wm.set("fast_conf", -wm.get("recall")->at("prompt_bpc").get<double>());
      return r;
    });
    WorkingMemory seen, unseen;
    seen.set("prompt", std::string("a dog lay by one "));
    unseen.set("prompt", std::string("purple elephants "));
    const RunResult r1 = i2.run("dual_process", "qa", seen);
    const RunResult r2 = i2.run("dual_process", "qa", unseen);
    CHECK(r1.trace.steps.size() >= 2 && r1.trace.steps[1].next == "answer");
    CHECK(r2.trace.steps.size() >= 2 && r2.trace.steps[1].next == "slow");
  }
  {  // the applicability hook: unset is the Beta; set, it decides, OOD routes to IMPASSE, labels arrive
    auto app = std::make_shared<FixedApp>();
    app->p["cooperative_reply"] = 0.05;
    app->ood["five_whys"] = true;
    Interpreter i3(lib, Config{}, 1);
    bd::bind_all(i3, tk);
    i3.set_applicability(app);
    WorkingMemory w;
    const RunResult na = i3.run("cooperative_reply", "c", w);
    CHECK(na.impasse() && na.trace.note == "not applicable");
    const RunResult oo = i3.run("five_whys", "c", w);
    CHECK(oo.impasse() && oo.trace.note == "ood");
    w.set("check", true);
    const RunResult ok = i3.run("tote", "c", w);
    CHECK(!ok.impasse());
    CHECK(!app->labels.empty() && app->labels.back().first == "tote");
    Goal g{"c", {"cooperative_reply", "five_whys", "tote"}};
    const SolveResult s = i3.solve(g, w);
    CHECK(!s.impasse && s.applicable.size() == 1 && s.chosen == "tote");
    app->p["tote"] = 0.0;
    const SolveResult s2 = i3.solve(g, w);
    CHECK(s2.impasse);
    i3.set_applicability(nullptr);  // back to the Beta: all three applicable at Beta(1,1)
    const SolveResult s3 = i3.solve(g, w);
    CHECK(s3.applicable.size() == 3);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: thought_bindings_test <docs/thought/processes.json>\n");
    return 2;
  }
  const std::string text = corpus();
  auto lm = bd::train_lm(text);
  cypha::cyphalm::FmInfiniGram fm(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
  bd::Toolkit tk;
  tk.lm = lm->view();
  tk.gen_bytes = 40;
  tk.longest_match = [&fm](const std::string& p) {
    std::size_t n = 0;
    fm.longest_suffix(reinterpret_cast<const std::uint8_t*>(p.data()), p.size(), 64, n);
    return n;
  };
  test_cyphalm(tk, text);
  test_knowledge();
  test_hypothesis_and_decisions();
  test_misc();
  test_library(argv[1], tk);
  std::printf("thought_bindings_test: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
