// K4: the thought library's primitives bound to Cypha's own libraries (docs/THOUGHT.md section 7, 12;
// docs/reports/K4_K5_THOUGHT_BINDINGS.md).
//
// Header-only. Each binding reads and writes named working-memory keys, so one binding serves every process that
// uses the op; a process step can still be bound on its own ("<process>.<step>"), which wins.
//
//   perceive, read       CyphaLM: bits per byte and the largest per-byte surprise of wm "text" (packed_trie.hpp)
//   recall_case          CyphaLM's System 1: its greedy continuation of wm "prompt", its mean log2 p, and the
//                        longest corpus match of the prompt when an index hook is given (FmInfiniGram)
//   generate             CyphaLM sampling of wm "prompt"; without a prompt, wm "candidates" are passed on
//   retrieve             a fact-store opinion for wm "query"; in answer mode the opinion may gate, never answer
//   score, evaluate      ECS fusion of wm "evidence" into an opinion and grade (ecs.hpp)
//   classify             the epistemic state of wm "opinion" (epistemic.hpp), or the attached classifier on
//                        wm "features" (the Cypha DIF classifier: bindings_cypha.hpp)
//   explain, communicate cards with reason codes; the rule-10 guard (PLAN rule 10): in answer mode an answer is sent
//                        only when its source is "cyphalm", otherwise it becomes an "I don't know" card
//   test, verify, falsify a hypothesis test's result and Bayes factor, S7b three-way likelihoods (hypothesis.hpp)
//   update               posterior from a Bayes factor, or a Beta from a result
//   decide               a guard set by an earlier step (wm "guard"); factories for threshold and LCI decisions
//   monitor              CUSUM on wm "obs" with the alarm in wm "alarm"; an e-process when configured (lci.hpp)
//   select, rank         the highest value of information among the gap map's open gaps (gap_map.hpp), or among
//                        wm "options"; Thompson draws when options carry Beta counts
//   estimate             a proportion with its Wilson interval
//   combine              log-space product of 90% intervals (Fermi), or opinion fusion
//   transform            isotonic calibration (ecs.hpp IsotonicCalibrator)
//   compare, match       cosine of mean word vectors (scribe.hpp WordVectors), or a numeric difference
//   infer                the derivation factor of a premise chain, depth cap 3 in answer mode (ecs.hpp)
//   ask                  the Ask-Odin queue: appended, never sent (no teacher inference here)
//   record               an append-only journal (the experiment-DB hook)
//   simulate, measure, decompose, act, schedule, wait: the caller's callbacks; by default the step is recorded
//
// Rule 10 (PLAN): a thought process answers a question only through CyphaLM. Traces are working, not answers.
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cypha/cyphalm/packed_trie.hpp"
#include "cypha/eval/ecs.hpp"
#include "cypha/eval/epistemic.hpp"
#include "cypha/eval/gap_map.hpp"
#include "cypha/eval/hypothesis.hpp"
#include "cypha/eval/lci.hpp"
#include "cypha/eval/scribe.hpp"
#include "cypha/thought/interpreter.hpp"

namespace cypha::thought::bind {

namespace ev = cypha::eval;
namespace trie = cypha::cyphalm::trie;

// ---------------------------------------------------------------------------
// CyphaLM: a view over a packed-trie model (Model or PackedModel) with its mixer weights
// ---------------------------------------------------------------------------

struct LmView {
  /// Next-byte distribution after hist[0..len) (Mode::Synth, the learned mixture, as `sample` uses it).
  std::function<void(const std::uint8_t* hist, std::size_t len, double* P)> dist;
  bool ready() const { return static_cast<bool>(dist); }
};

template <class ModelT>
LmView make_lm(const ModelT& m, const double w[6], std::size_t follower_limit) {
  std::array<double, 6> ws{};
  for (int i = 0; i < 6; ++i) ws[static_cast<std::size_t>(i)] = w[i];
  LmView v;
  v.dist = [&m, ws, follower_limit](const std::uint8_t* hist, std::size_t len, double* P) {
    double e[5][256];
    bool matched = false;
    unsigned pred = 0;
    m.fill_experts(hist, len, follower_limit, trie::Mode::Synth, e, matched, pred, false);
    trie::mix_dist(ws.data(), e, matched, pred, P);
  };
  return v;
}

/// A small CyphaLM trained on `text` (online mixer weights, compacted match), for tests and small runs. The model
/// keeps a pointer to its own copy of the text.
struct OwnedLm {
  std::vector<std::uint8_t> data;
  trie::Model model;
  double w[6]{};
  LmView view() const { return make_lm(model, w, data.size()); }
};

inline std::unique_ptr<OwnedLm> train_lm(const std::string& text, const trie::Config& cfg = trie::Config{}) {
  auto lm = std::make_unique<OwnedLm>();
  lm->data.assign(text.begin(), text.end());
  lm->model.verbose = false;
  lm->model.data = lm->data.data();
  trie::Config c = cfg;
  c.reserve_nodes = std::min<std::size_t>(c.reserve_nodes, 4 * lm->data.size() + 16);
  c.reserve_fols = std::min<std::size_t>(c.reserve_fols, 4 * lm->data.size() + 16);
  c.match_bits = std::min<unsigned>(c.match_bits, 16u);
  lm->model.init(c);
  trie::init_weights(lm->w, c);
  trie::train_span(lm->model, lm->data, lm->w, 0, lm->data.size(), c);
  lm->model.compact_match();
  return lm;
}

struct Surprise {
  double bpc = 0.0;       // mean -log2 p over the text
  double max_bits = 0.0;  // the largest single-byte surprise
  std::size_t n = 0;
};

inline Surprise surprise(const LmView& lm, const std::string& context, const std::string& text) {
  Surprise s;
  std::vector<std::uint8_t> seq(context.begin(), context.end());
  double P[256];
  for (unsigned char c : text) {
    lm.dist(seq.data(), seq.size(), P);
    const double bits = -std::log2(std::max(P[c], 1e-300));
    s.bpc += bits;
    s.max_bits = std::max(s.max_bits, bits);
    ++s.n;
    seq.push_back(c);
  }
  if (s.n) s.bpc /= static_cast<double>(s.n);
  return s;
}

struct Recall {
  std::string text;
  double mean_lp = 0.0;     // mean log2 p of the bytes CyphaLM chose
  double prompt_bpc = 0.0;  // the prompt's own surprise: how familiar the situation is to CyphaLM
  std::size_t match_n = 0;
};

/// Greedy continuation of `prompt` for at most `max_bytes`, stopping after one of `stops`.
inline Recall recall(const LmView& lm, const std::string& prompt, int max_bytes, const std::string& stops = "\n.") {
  Recall r;
  std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
  double P[256];
  double lp = 0.0;
  for (int t = 0; t < max_bytes; ++t) {
    lm.dist(seq.data(), seq.size(), P);
    int pick = 0;
    for (int s = 1; s < 256; ++s)
      if (P[s] > P[pick]) pick = s;
    lp += std::log2(std::max(P[pick], 1e-300));
    seq.push_back(static_cast<std::uint8_t>(pick));
    r.text.push_back(static_cast<char>(pick));
    if (stops.find(static_cast<char>(pick)) != std::string::npos) break;
  }
  if (!r.text.empty()) r.mean_lp = lp / static_cast<double>(r.text.size());
  r.prompt_bpc = surprise(lm, "", prompt).bpc;
  return r;
}

/// Seeded sampling at a temperature (CyphaLM writes every byte).
inline std::string sample(const LmView& lm, const std::string& prompt, int n, Rng& rng, double temperature = 0.8) {
  std::vector<std::uint8_t> seq(prompt.begin(), prompt.end());
  std::string out;
  double P[256];
  for (int t = 0; t < n; ++t) {
    lm.dist(seq.data(), seq.size(), P);
    double z = 0.0;
    double q[256];
    for (int s = 0; s < 256; ++s) {
      q[s] = std::pow(std::max(P[s], 0.0), 1.0 / temperature);
      z += q[s];
    }
    double u = uniform01(rng) * z;
    int pick = 255;
    for (int s = 0; s < 256; ++s) {
      u -= q[s];
      if (u <= 0.0) {
        pick = s;
        break;
      }
    }
    seq.push_back(static_cast<std::uint8_t>(pick));
    out.push_back(static_cast<char>(pick));
  }
  return out;
}

// ---------------------------------------------------------------------------
// JSON helpers for Cypha's evaluation types
// ---------------------------------------------------------------------------

inline Json opinion_json(const ev::Opinion& o) {
  return Json{{"r", o.r}, {"s", o.s}, {"b", o.b}, {"d", o.d}, {"u", o.u}, {"P", o.P}, {"lo", o.lo},
              {"hi", o.hi}, {"grade", ev::grade_name(o.grade)}};
}

inline ev::Opinion opinion_from_json(const Json& j) {
  return ev::opinion_from_counts(j.value("r", 0.0), j.value("s", 0.0));
}

inline ev::Stance stance_from(const std::string& s) {
  if (s == "oppose") return ev::Stance::Oppose;
  if (s == "competing") return ev::Stance::CompetingValue;
  return ev::Stance::Support;
}

inline ev::TestOutcome outcome_from(const std::string& s) {
  if (s == "positive") return ev::TestOutcome::Positive;
  if (s == "negative") return ev::TestOutcome::Negative;
  return ev::TestOutcome::Inconclusive;
}

/// The message part of an "I don't know" card (DISCOVERY 3.5): the message and the tag.
inline std::string idk_text(ev::ReasonCode c) {
  return std::string("I don't know: ") + ev::reason_meaning(c) + ". <|unknown reason=" + ev::reason_name(c) + "|>";
}

// ---------------------------------------------------------------------------
// The toolkit: what the bindings read from
// ---------------------------------------------------------------------------

struct ClassifierOut {
  std::string label;
  double confidence = 0.0;
  bool ood = false;
};

struct Toolkit {
  // CyphaLM
  LmView lm;
  int recall_bytes = 64;
  int gen_bytes = 64;
  double temperature = 0.8;
  std::function<std::size_t(const std::string&)> longest_match;  // the ∞-gram's longest suffix match of a prompt
  // knowledge (discovery mode; in answer mode it may gate, never answer)
  std::map<std::string, ev::Opinion> facts;  // key -> opinion
  ev::GapMap* gaps = nullptr;
  // classifier (the Cypha DIF classifier through bindings_cypha.hpp, or any other)
  std::function<ClassifierOut(const std::vector<double>&)> classifier;
  // calibration and vectors
  const ev::IsotonicCalibrator* calibrator = nullptr;
  const ev::WordVectors* vectors = nullptr;
  // CUSUM defaults (standardised units)
  double cusum_k = 0.5, cusum_h = 5.0;
  // ask and record: kept in memory; written to files when paths are set
  std::vector<Json> ask_queue, journal;
  std::string ask_path, journal_path;
  // rule-10 log: answers refused because their source was not CyphaLM
  std::size_t rule10_refusals = 0;
};

inline void append_jsonl(const std::string& path, const Json& j) {
  if (path.empty()) return;
  std::ofstream f(path, std::ios::app);
  f << j.dump() << "\n";
}

inline bool answer_mode(const WorkingMemory& wm) { return wm.str("mode") == "answer"; }

// ---------------------------------------------------------------------------
// The bindings, one function each (unit-testable), and bind_all
// ---------------------------------------------------------------------------

inline StepResult do_perceive(Toolkit& tk, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  if (!tk.lm.ready() || !wm.has("text")) {
    r.ok = wm.has("text");
    r.value = wm.has("text") ? *wm.get("text") : Json();
    r.cost = 1.0;
    return r;
  }
  const std::string text = wm.str("text");
  const Surprise s = surprise(tk.lm, wm.str("context"), text);
  r.value = Json{{"bpc", s.bpc}, {"max_bits", s.max_bits}, {"n", s.n}};
  wm.set("surprise", r.value);
  r.cost = 1.0 + static_cast<double>(text.size()) / 1024.0;
  return r;
}

inline StepResult do_recall(Toolkit& tk, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  if (!tk.lm.ready() || !wm.has("prompt")) {
    r.ok = false;
    r.impasse = !tk.lm.ready();
    return r;
  }
  const std::string prompt = wm.str("prompt");
  Recall rc = recall(tk.lm, prompt, tk.recall_bytes);
  if (tk.longest_match) rc.match_n = tk.longest_match(prompt);
  r.value = Json{{"text", rc.text},           {"mean_lp", rc.mean_lp}, {"prompt_bpc", rc.prompt_bpc},
                 {"match_n", rc.match_n},      {"source", "cyphalm"}};
  wm.set("recall", r.value);
  r.cost = 1.0 + static_cast<double>(rc.text.size()) / 64.0;
  return r;
}

inline StepResult do_generate(Toolkit& tk, const Step&, StepContext& ctx, WorkingMemory& wm) {
  StepResult r;
  if (tk.lm.ready() && wm.has("prompt") && ctx.rng) {
    const std::string g = sample(tk.lm, wm.str("prompt"), tk.gen_bytes, *ctx.rng, tk.temperature);
    r.value = Json{{"text", g}, {"source", "cyphalm"}};
    wm.set("generated", r.value);
    r.cost = 1.0 + static_cast<double>(g.size()) / 64.0;
    return r;
  }
  if (const Json* c = wm.get("candidates")) {
    r.value = *c;
    wm.set("generated", *c);
  }
  r.cost = 1.0;
  return r;
}

inline StepResult do_retrieve(Toolkit& tk, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  const std::string key = wm.str("query");
  const auto it = tk.facts.find(key);
  if (key.empty() || it == tk.facts.end()) {
    r.ok = false;
    return r;
  }
  Json o = opinion_json(it->second);
  o["key"] = key;
  o["gate_only"] = answer_mode(wm);  // rule 10: in answer mode it gates, never answers
  r.value = o;
  wm.set("opinion", o);
  return r;
}

inline StepResult do_score(Toolkit&, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  const Json* ev_list = wm.get("evidence");
  if (!ev_list || !ev_list->is_array()) {
    r.ok = false;
    return r;
  }
  std::vector<ev::EvidenceItem> items;
  for (const Json& e : *ev_list) {
    ev::BetaPosterior src{e.value("trust_a", 1.0), e.value("trust_b", 1.0)};
    items.push_back(ev::make_evidence(stance_from(e.value("stance", std::string("support"))), src,
                                      e.value("extraction", 1.0), e.value("omega", 1.0)));
  }
  const ev::Opinion o = ev::fuse(items);
  r.value = opinion_json(o);
  wm.set("opinion", r.value);
  return r;
}

inline StepResult do_classify(Toolkit& tk, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  if (const Json* f = wm.get("features"); f && f->is_array() && tk.classifier) {
    std::vector<double> x;
    for (const Json& v : *f) x.push_back(v.get<double>());
    const ClassifierOut c = tk.classifier(x);
    r.value = Json{{"label", c.label}, {"confidence", c.confidence}, {"ood", c.ood}};
    wm.set("class", r.value);
    r.ok = !c.ood;
    return r;
  }
  if (const Json* o = wm.get("opinion")) {
    std::optional<ev::Opinion> rival;
    if (const Json* rv = wm.get("rival")) rival = opinion_from_json(*rv);
    const ev::EpistemicState s = ev::classify(opinion_from_json(*o), rival);
    r.value = Json(ev::state_name(s));
    wm.set("state", r.value);
    return r;
  }
  r.ok = false;
  return r;
}

/// Cards (epistemic.hpp) and the rule-10 guard. With wm "answer" {text, source, confidence}: an answer card when not
/// in answer mode or when the source is CyphaLM; otherwise an "I don't know" card (NO_EVIDENCE: nothing CyphaLM has
/// read supports it) and the refusal is counted. Without an answer: an "I don't know" card with wm "reason"
/// (default NO_EVIDENCE) when wm "idk" is set, else a report card of wm "report".
inline StepResult do_communicate(Toolkit& tk, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  Json card;
  const Json* ans = wm.get("answer");
  const bool idk_asked = wm.has("idk") && wm.get("idk")->is_boolean() && wm.get("idk")->get<bool>();
  if (ans && ans->is_object() && !idk_asked) {
    const std::string src = ans->value("source", std::string());
    if (answer_mode(wm) && src != "cyphalm") {
      ++tk.rule10_refusals;
      card = Json{{"kind", "idk"}, {"reason", ev::reason_name(ev::ReasonCode::NoEvidence)},
                  {"text", idk_text(ev::ReasonCode::NoEvidence)}, {"note", "rule10"}};
    } else {
      card = Json{{"kind", "answer"}, {"text", ans->value("text", std::string())},
                  {"confidence", ans->value("confidence", 0.0)}, {"source", src}};
    }
  } else if (idk_asked || answer_mode(wm)) {
    ev::ReasonCode code = ev::ReasonCode::NoEvidence;
    if (auto c = ev::reason_from_name(wm.str("reason"))) code = *c;
    card = Json{{"kind", "idk"}, {"reason", ev::reason_name(code)}, {"text", idk_text(code)}};
  } else {
    card = Json{{"kind", "report"}, {"content", wm.has("report") ? *wm.get("report") : Json()}};
  }
  r.value = card;
  wm.set("card", card);
  return r;
}

/// A hypothesis test's result: wm "outcome" (positive | negative | inconclusive) and the test's likelihoods in wm
/// "test" {t: [3], f: [3]} or {s, c}; wm "posterior" is moved by the Bayes factor. ok = a positive result (a
/// `falsify` step is ok when the result is negative: the disconfirmation search found evidence against).
inline StepResult do_test(Toolkit&, const Step& step, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  if (!wm.has("outcome")) {  // a test with no hypothesis wiring: pass when wm "check" is true
    const Json* c = wm.get("check");
    r.ok = c && c->is_boolean() ? c->get<bool>() : true;
    return r;
  }
  const ev::TestOutcome o = outcome_from(wm.str("outcome"));
  ev::HypTest t;
  if (const Json* tj = wm.get("test")) {
    if (tj->contains("t") && tj->contains("f")) {
      for (int i = 0; i < 3; ++i) {
        t.likelihoods.t[i] = tj->at("t").at(static_cast<std::size_t>(i)).get<double>();
        t.likelihoods.f[i] = tj->at("f").at(static_cast<std::size_t>(i)).get<double>();
      }
    } else {
      t.sensitivity = tj->value("s", 0.9);
      t.specificity = tj->value("c", 0.9);
    }
  }
  const double bf = ev::bayes_factor(o, t);
  const double prior = wm.num("posterior", 0.5);
  const double post = ev::update_posterior(prior, bf);
  wm.set("bf", bf);
  wm.set("posterior", post);
  r.value = Json{{"outcome", ev::test_outcome_name(o)}, {"bf", bf}, {"posterior", post}};
  r.ok = step.op == "falsify" ? o == ev::TestOutcome::Negative : o == ev::TestOutcome::Positive;
  r.gain = std::fabs(post - prior);
  return r;
}

inline StepResult do_update(Toolkit&, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  if (wm.has("bf") && wm.has("posterior") && wm.has("pending_bf")) {
    const double post = ev::update_posterior(wm.num("posterior"), wm.num("bf"));
    wm.set("posterior", post);
    wm.erase("pending_bf");
    r.value = post;
    return r;
  }
  if (const Json* b = wm.get("beta"); b && b->is_object() && wm.has("result")) {
    ev::BetaPosterior bp{b->value("a", 1.0), b->value("b", 1.0)};
    bp.observe(wm.num("result") >= 0.5);
    wm.set("beta", Json{{"a", bp.alpha}, {"b", bp.beta}});
    r.value = bp.mean();
    return r;
  }
  return r;
}

/// A guard set by an earlier step: wm "guard" names the branch (it must be in step.next, else the learned choice).
inline StepResult do_decide(Toolkit&, const Step& step, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 0.0;
  const std::string g = wm.str("guard");
  if (!g.empty() && std::find(step.next.begin(), step.next.end(), g) != step.next.end()) r.chosen_next = g;
  wm.erase("guard");
  return r;
}

/// Factory: branch `hi` when wm[key] >= tau, else `lo`.
inline Primitive decide_threshold(const std::string& key, double tau, const std::string& hi, const std::string& lo) {
  return [key, tau, hi, lo](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    r.chosen_next = wm.num(key) >= tau ? hi : lo;
    return r;
  };
}

/// Factory: an LCI promotion decision on wm "lci_diffs" (per-item gains): branch `promote` when LCI >= 0.95.
inline Primitive decide_lci(const std::string& promote, const std::string& hold, double eps = 0.0) {
  return [promote, hold, eps](const Step&, StepContext&, WorkingMemory& wm) {
    StepResult r;
    ev::LciInputs in;
    if (const Json* d = wm.get("lci_diffs"))
      for (const Json& v : *d) in.test_diffs.push_back(v.get<double>());
    in.eps = eps;
    in.gain_dev = wm.num("lci_gain_dev", 0.0);
    in.gain_ext = wm.num("lci_gain_ext", 0.0);
    in.seed_gains = {in.gain_ext, in.gain_ext, in.gain_ext};
    const ev::LciDecision d = ev::compute_lci(in);
    wm.set("lci", d.lci);
    r.value = d.lci;
    r.chosen_next = d.meets_threshold ? promote : hold;
    return r;
  };
}

/// CUSUM (upper, standardised) on wm "obs": S <- max(0, S + (x - mu0)/sigma - k); alarm when S > h. State in wm
/// "cusum" {S, mu0, sigma, k, h}.
inline StepResult do_monitor(Toolkit& tk, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  Json st = wm.has("cusum") ? *wm.get("cusum") : Json::object();
  const double mu0 = st.value("mu0", 0.0), sigma = st.value("sigma", 1.0);
  const double k = st.value("k", tk.cusum_k), h = st.value("h", tk.cusum_h);
  double S = st.value("S", 0.0);
  if (wm.has("obs")) S = std::max(0.0, S + (wm.num("obs") - mu0) / sigma - k);
  st["S"] = S;
  st["mu0"] = mu0;
  st["sigma"] = sigma;
  st["k"] = k;
  st["h"] = h;
  wm.set("cusum", st);
  const bool alarm = S > h;
  wm.set("alarm", alarm);
  r.value = Json{{"S", S}, {"alarm", alarm}};
  return r;
}

/// The gap map's best open gap by value of information; or the best of wm "options" [{id, score}] (Thompson draws
/// when an option carries Beta counts {a, b}).
inline StepResult do_select(Toolkit& tk, const Step&, StepContext& ctx, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  if (const Json* opts = wm.get("options"); opts && opts->is_array() && !opts->empty()) {
    double best = -1e300;
    Json pick;
    for (const Json& o : *opts) {
      double v = o.value("score", 0.0);
      if (o.contains("a") && o.contains("b") && ctx.rng) v = Beta{o.at("a").get<double>(), o.at("b").get<double>()}.sample(*ctx.rng);
      if (v > best) {
        best = v;
        pick = o;
      }
    }
    r.value = pick;
    wm.set("chosen", pick);
    return r;
  }
  if (tk.gaps) {
    for (const ev::Gap* g : tk.gaps->ranked()) {
      if (g->status == ev::GapStatus::Closed || g->status == ev::GapStatus::Parked) continue;
      r.value = Json{{"id", g->id}, {"entity", g->entity}, {"relation", g->relation}, {"voi", g->voi}};
      wm.set("chosen", r.value);
      return r;
    }
  }
  r.ok = false;
  return r;
}

inline StepResult do_rank(Toolkit&, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  const Json* opts = wm.get("options");
  if (!opts || !opts->is_array()) {
    r.ok = false;
    return r;
  }
  std::vector<Json> v(opts->begin(), opts->end());
  std::stable_sort(v.begin(), v.end(),
                   [](const Json& a, const Json& b) { return a.value("score", 0.0) > b.value("score", 0.0); });
  r.value = Json(v);
  wm.set("ranked", r.value);
  return r;
}

inline StepResult do_estimate(Toolkit&, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  if (const Json* c = wm.get("counts"); c && c->is_object()) {
    const double k = c->value("k", 0.0), n = c->value("n", 0.0);
    const double p = n > 0 ? k / n : 0.0;
    r.value = Json{{"p", p}, {"lo", ev::wilson_lower(k, n)}, {"hi", ev::wilson_upper(k, n)}, {"n", n}};
    wm.set("estimate", r.value);
    return r;
  }
  return r;
}

/// Fermi: wm "factors" [{lo, hi}] are 90% intervals of positive quantities, each lognormal; their product's median and
/// 90% interval: mu = sum (ln lo + ln hi) / 2, sigma^2 = sum ((ln hi - ln lo) / (2 * 1.6449))^2.
inline StepResult do_combine(Toolkit&, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  if (const Json* f = wm.get("factors"); f && f->is_array() && !f->empty()) {
    constexpr double z90 = 1.6448536269514722;
    double mu = 0.0, var = 0.0;
    for (const Json& x : *f) {
      const double lo = std::log(x.at("lo").get<double>()), hi = std::log(x.at("hi").get<double>());
      mu += 0.5 * (lo + hi);
      const double s = (hi - lo) / (2.0 * z90);
      var += s * s;
    }
    const double sd = std::sqrt(var);
    r.value = Json{{"median", std::exp(mu)}, {"lo", std::exp(mu - z90 * sd)}, {"hi", std::exp(mu + z90 * sd)}};
    wm.set("estimate", r.value);
    return r;
  }
  if (wm.has("evidence")) {
    Toolkit tk;
    Step s;
    StepContext c;
    return do_score(tk, s, c, wm);
  }
  return r;
}

inline StepResult do_transform(Toolkit& tk, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  if (wm.has("raw")) {
    const double raw = wm.num("raw");
    const double cal = tk.calibrator && tk.calibrator->fitted() ? tk.calibrator->predict(raw) : raw;
    wm.set("calibrated", cal);
    r.value = cal;
  }
  return r;
}

inline std::vector<std::string> words_of(const std::string& s) {
  std::vector<std::string> out;
  ev::tokenize_words(ev::scribe_detail::lower_ascii(s), out);
  return out;
}

/// Cosine of the mean word vectors of wm "a" and wm "b" (strings), or 1 - |a - b| / max(|a|, |b|) for numbers.
inline StepResult do_compare(Toolkit& tk, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  const Json* a = wm.get("a");
  const Json* b = wm.get("b");
  if (!a || !b) {
    r.ok = false;
    return r;
  }
  double sim = 0.0;
  if (a->is_number() && b->is_number()) {
    const double x = a->get<double>(), y = b->get<double>();
    const double m = std::max(std::fabs(x), std::fabs(y));
    sim = m > 0 ? 1.0 - std::fabs(x - y) / m : 1.0;
  } else if (a->is_string() && b->is_string() && tk.vectors && tk.vectors->size() > 0) {
    const std::size_t d = tk.vectors->dim();
    auto mean = [&](const std::string& s, std::vector<double>& v) {
      v.assign(d, 0.0);
      int n = 0;
      for (const std::string& w : words_of(s)) {
        const int i = tk.vectors->index(w);
        if (i < 0) continue;
        const float* row = tk.vectors->row(i);
        for (std::size_t k = 0; k < d; ++k) v[k] += row[k];
        ++n;
      }
      return n;
    };
    std::vector<double> va, vb;
    if (mean(a->get<std::string>(), va) > 0 && mean(b->get<std::string>(), vb) > 0) {
      double dot = 0, na = 0, nb = 0;
      for (std::size_t k = 0; k < d; ++k) {
        dot += va[k] * vb[k];
        na += va[k] * va[k];
        nb += vb[k] * vb[k];
      }
      sim = na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : 0.0;
    }
  } else {
    r.ok = false;
    return r;
  }
  wm.set("similarity", sim);
  r.value = sim;
  return r;
}

/// The derivation factor of wm "premises" (their ECS P) through a rule of reliability wm "rule_reliability" at depth
/// wm "depth" (default 1). Depth above 3 derives nothing (BEYOND_DEPTH), as in answer mode (ecs.hpp).
inline StepResult do_infer(Toolkit&, const Step&, StepContext&, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  std::vector<double> ps;
  if (const Json* p = wm.get("premises"))
    for (const Json& v : *p) ps.push_back(v.get<double>());
  const int depth = static_cast<int>(wm.num("depth", 1));
  const double f = ev::derivation_factor(ps, wm.num("rule_reliability", 1.0), depth);
  wm.set("derived", f);
  r.value = f;
  if (depth > ev::kMaxDerivationDepth) {
    r.ok = false;
    wm.set("reason", std::string(ev::reason_name(ev::ReasonCode::BeyondDepth)));
  }
  return r;
}

inline StepResult do_ask(Toolkit& tk, const Step&, StepContext& ctx, WorkingMemory& wm) {
  StepResult r;
  r.cost = 1.0;
  Json q{{"id", tk.ask_queue.size()}, {"from", ctx.process + "." + ctx.step}, {"question", wm.str("question")},
         {"status", "pending"}};
  tk.ask_queue.push_back(q);
  append_jsonl(tk.ask_path, q);
  r.value = Json{{"queued", q["id"]}};
  return r;
}

inline StepResult do_record(Toolkit& tk, const Step&, StepContext& ctx, WorkingMemory& wm) {
  StepResult r;
  r.cost = 0.0;
  Json e{{"at", ctx.process + "." + ctx.step},
         {"value", wm.has("last_value") ? *wm.get("last_value") : Json()}};
  tk.journal.push_back(e);
  append_jsonl(tk.journal_path, e);
  return r;
}

/// The default for ops whose meaning depends on the process (simulate, measure, decompose, act, schedule, wait): the
/// step is recorded and costs 1. Processes that need them bind their own step callbacks.
inline StepResult do_default(Toolkit&, const Step&, StepContext&, WorkingMemory&) {
  StepResult r;
  r.cost = 1.0;
  return r;
}

/// Bind every primitive of the library to the toolkit. The toolkit must outlive the interpreter's runs.
inline void bind_all(Interpreter& it, Toolkit& tk) {
  using Fn = StepResult (*)(Toolkit&, const Step&, StepContext&, WorkingMemory&);
  const std::vector<std::pair<std::string, Fn>> table = {
      {"perceive", do_perceive},   {"read", do_perceive},       {"recall_case", do_recall},
      {"generate", do_generate},   {"retrieve", do_retrieve},   {"score", do_score},
      {"evaluate", do_score},      {"classify", do_classify},   {"explain", do_communicate},
      {"communicate", do_communicate}, {"test", do_test},       {"verify", do_test},
      {"falsify", do_test},        {"update", do_update},       {"decide", do_decide},
      {"monitor", do_monitor},     {"select", do_select},       {"rank", do_rank},
      {"estimate", do_estimate},   {"combine", do_combine},     {"transform", do_transform},
      {"compare", do_compare},     {"match", do_compare},       {"infer", do_infer},
      {"ask", do_ask},             {"record", do_record},       {"simulate", do_default},
      {"measure", do_default},     {"decompose", do_default},   {"act", do_default},
      {"schedule", do_default},    {"wait", do_default}};
  for (const auto& [op, fn] : table)
    it.bind(op, [&tk, fn](const Step& s, StepContext& c, WorkingMemory& wm) { return fn(tk, s, c, wm); });
}

}  // namespace cypha::thought::bind
