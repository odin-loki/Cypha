// K7: process mining and MDL structural edits (docs/THOUGHT.md section 6; docs/reports/K6_K7_CHUNKING_AND_MINING.md).
//
// Header-only.
//   count_log / heuristics_miner   directly-follows counts |a > b| over traces with a virtual start "^" and end "$";
//                                  dependency a => b = (|a>b| - |b>a|) / (|a>b| + |b>a| + 1); an edge when
//                                  a => b >= 0.9 and |a>b| >= 10 (THOUGHT 6.1), plus the heuristics miner's self-loop
//                                  and length-2 loop measures (Weijters & van der Aalst 2003).
//   Model                          a process as a probabilistic automaton over step labels: node 0 is the start "^",
//                                  node 1 the end "$"; a node carries a label (or a short sequence, for a merged step);
//                                  a node's successors have distinct first labels, so a trace replays along one path.
//   description_length             L(model) + L(traces | model) in bits (THOUGHT 6.4). Transition probabilities are
//                                  the counts + 0.5 per allowed edge and per escape. A transition the model does not
//                                  allow takes the escape: its probability plus log2(|labels| + 1) bits to name the
//                                  label; replay restarts at the first node whose labels match there.
//   edit_search                    greedy MDL edits on training traces: remove an edge, add an edge for an escape seen
//                                  at least 5 times, split a node by dividing its incoming edges in two, merge two steps
//                                  always taken together.
//   promote_lci                    an edited model replaces its start model only if LCI >= 0.95 on held-out traces
//                                  (THOUGHT 6.5; lci.hpp).
//
// Mined processes and edits are hypotheses: they enter the library tainted (THOUGHT 9).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../eval/lci.hpp"
#include "cypha/thought/learners.hpp"

namespace cypha::thought::mine {

using Trace = std::vector<std::string>;
using Log = std::vector<Trace>;
using Edge = std::pair<std::string, std::string>;
using EdgeSet = std::set<Edge>;

inline const std::string kStart = "^";
inline const std::string kEnd = "$";

// ---------------------------------------------------------------------------
// The heuristics miner (THOUGHT 6.1)
// ---------------------------------------------------------------------------

struct Counts {
  std::map<Edge, int> df;   // |a > b|, with ^ before the first event and $ after the last
  std::map<Edge, int> aba;  // |a b a|: a, then b != a, then a again
  std::set<std::string> labels;
};

inline Counts count_log(const Log& log) {
  Counts c;
  for (const Trace& t : log) {
    std::vector<std::string> s;
    s.reserve(t.size() + 2);
    s.push_back(kStart);
    s.insert(s.end(), t.begin(), t.end());
    s.push_back(kEnd);
    for (const std::string& x : t) c.labels.insert(x);
    for (std::size_t i = 0; i + 1 < s.size(); ++i) ++c.df[{s[i], s[i + 1]}];
    for (std::size_t i = 0; i + 2 < s.size(); ++i)
      if (s[i] == s[i + 2] && s[i] != s[i + 1]) ++c.aba[{s[i], s[i + 1]}];
  }
  return c;
}

inline int count_of(const std::map<Edge, int>& m, const std::string& a, const std::string& b) {
  const auto it = m.find({a, b});
  return it == m.end() ? 0 : it->second;
}

/// a => b; for a == b the self-loop measure |a>a| / (|a>a| + 1). `noise_floor` reverse counts are ignored (K7
/// amendment 2, miner M2; 0 is THOUGHT 6.1's measure).
inline double dependency(const Counts& c, const std::string& a, const std::string& b, int noise_floor = 0) {
  const double ab = count_of(c.df, a, b);
  if (a == b) return ab / (ab + 1.0);
  const double ba = std::max(0, count_of(c.df, b, a) - noise_floor);
  return (ab - ba) / (ab + ba + 1.0);
}

/// The length-2 loop measure (|aba| + |bab|) / (|aba| + |bab| + 1).
inline double loop2(const Counts& c, const std::string& a, const std::string& b) {
  const double s = count_of(c.aba, a, b) + count_of(c.aba, b, a);
  return s / (s + 1.0);
}

struct MinerRules {
  double dep = 0.9;
  int min_count = 10;
  double loop_dep = 0.9;
  int loop_min = 10;
  int noise_floor = 0;  // K7 amendment 2: M0 (THOUGHT) 0.9 / 0; M1 0.8 / 0; M2 0.9 / 1
};

inline MinerRules miner_variant(const std::string& name) {
  MinerRules r;
  if (name == "M1") r.dep = 0.8;
  if (name == "M2") r.noise_floor = 1;
  return r;
}

inline EdgeSet heuristics_miner(const Counts& c, const MinerRules& r = {}) {
  EdgeSet e;
  for (const auto& [ab, n] : c.df) {
    if (ab.first == ab.second) {
      if (n >= r.loop_min && dependency(c, ab.first, ab.second) >= r.loop_dep) e.insert(ab);
    } else if (n >= r.min_count && dependency(c, ab.first, ab.second, r.noise_floor) >= r.dep) {
      e.insert(ab);
    }
  }
  for (const auto& [ab, n] : c.aba) {
    (void)n;
    const std::string& a = ab.first;
    const std::string& b = ab.second;
    const int s = count_of(c.aba, a, b) + count_of(c.aba, b, a);
    if (s >= r.loop_min && loop2(c, a, b) >= r.loop_dep && count_of(c.df, a, b) > 0 && count_of(c.df, b, a) > 0) {
      e.insert({a, b});
      e.insert({b, a});
    }
  }
  return e;
}

inline EdgeSet heuristics_miner(const Log& log, const MinerRules& r = {}) { return heuristics_miner(count_log(log), r); }

// ---------------------------------------------------------------------------
// The model: a probabilistic automaton over step labels
// ---------------------------------------------------------------------------

struct Node {
  std::vector<std::string> labels;  // one label, or a sequence for a merged step
  std::vector<int> out;             // successor node ids
};

class Model {
 public:
  static constexpr int kStartNode = 0;
  static constexpr int kEndNode = 1;

  Model() {
    nodes.push_back(Node{{kStart}, {}});
    nodes.push_back(Node{{kEnd}, {}});
  }

  std::vector<Node> nodes;

  int add_node(std::vector<std::string> labels) {
    nodes.push_back(Node{std::move(labels), {}});
    return static_cast<int>(nodes.size()) - 1;
  }
  bool has_edge(int u, int v) const {
    const auto& o = nodes[static_cast<std::size_t>(u)].out;
    return std::find(o.begin(), o.end(), v) != o.end();
  }
  /// The successor of u whose first label is `label`, or -1.
  int succ_with_label(int u, const std::string& label) const {
    for (int v : nodes[static_cast<std::size_t>(u)].out)
      if (nodes[static_cast<std::size_t>(v)].labels.front() == label) return v;
    return -1;
  }
  /// Adding u -> v keeps the automaton deterministic (no other successor with v's first label).
  bool can_add(int u, int v) const {
    if (u == kEndNode || v == kStartNode || has_edge(u, v)) return false;
    return succ_with_label(u, nodes[static_cast<std::size_t>(v)].labels.front()) < 0;
  }
  void add_edge(int u, int v) {
    if (!has_edge(u, v)) nodes[static_cast<std::size_t>(u)].out.push_back(v);
  }
  void remove_edge(int u, int v) {
    auto& o = nodes[static_cast<std::size_t>(u)].out;
    o.erase(std::remove(o.begin(), o.end(), v), o.end());
  }
  std::size_t edge_count() const {
    std::size_t n = 0;
    for (const Node& x : nodes) n += x.out.size();
    return n;
  }
  /// Remove node v (not start or end): drop its edges and renumber.
  void remove_node(int v) {
    for (Node& x : nodes) {
      x.out.erase(std::remove(x.out.begin(), x.out.end(), v), x.out.end());
      for (int& w : x.out)
        if (w > v) --w;
    }
    nodes.erase(nodes.begin() + v);
  }
  /// The label edges of a model whose nodes carry one distinct label each (K7a): (label u, label v).
  EdgeSet label_edges() const {
    EdgeSet e;
    for (const Node& x : nodes)
      for (int v : x.out) e.insert({x.labels.back(), nodes[static_cast<std::size_t>(v)].labels.front()});
    return e;
  }
};

// ---------------------------------------------------------------------------
// Replay and description length (THOUGHT 6.4)
// ---------------------------------------------------------------------------

struct Replay {
  std::map<std::pair<int, int>, double> trans;              // u -> v
  std::map<int, double> esc;                                // escapes at u (-1: lost, no node)
  std::map<std::pair<int, std::string>, double> esc_label;  // escapes at u with the event's label
  std::map<int, double> restart_into;                       // restarts landing on node v
  std::map<std::tuple<int, int, int>, double> tri;          // p -> v -> w (w = -1: an escape at v)
  std::vector<double> trace_bits;                           // filled by score()
};

/// Does node v's label sequence match the events from `pos`?
inline bool matches(const Model& m, int v, const std::vector<std::string>& ev, std::size_t pos) {
  const auto& L = m.nodes[static_cast<std::size_t>(v)].labels;
  if (pos + L.size() > ev.size()) return false;
  for (std::size_t k = 0; k < L.size(); ++k)
    if (ev[pos + k] != L[k]) return false;
  return true;
}

/// The node replay restarts at after an escape: the first node (start excluded) whose labels match from `pos`.
inline int restart_node(const Model& m, const std::vector<std::string>& ev, std::size_t pos) {
  for (int v = 1; v < static_cast<int>(m.nodes.size()); ++v)
    if (matches(m, v, ev, pos)) return v;
  return -1;
}

/// A callback per replay event: (from node, to node or -1 for an escape, event label at the escape, trace index).
template <class F>
inline void replay_trace(const Model& m, const Trace& t, F&& on) {
  std::vector<std::string> ev(t);
  ev.push_back(kEnd);
  int cur = Model::kStartNode;
  int prev = -2;  // -2: no regular predecessor (the start, or after an escape)
  std::size_t pos = 0;
  while (pos < ev.size()) {
    if (cur >= 0) {
      const int v = m.succ_with_label(cur, ev[pos]);
      if (v >= 0 && matches(m, v, ev, pos)) {
        on(prev, cur, v, std::string());
        prev = cur;
        cur = v;
        pos += m.nodes[static_cast<std::size_t>(v)].labels.size();
        if (v == Model::kEndNode) return;
        continue;
      }
    }
    on(prev, cur, -1, ev[pos]);  // escape at cur (cur = -1: lost)
    const int w = restart_node(m, ev, pos);
    prev = -2;
    if (w >= 0) {
      on(-3, -3, w, std::string());  // restart marker
      cur = w;
      pos += m.nodes[static_cast<std::size_t>(w)].labels.size();
      if (w == Model::kEndNode) return;
    } else {
      cur = -1;
      pos += 1;
    }
  }
}

inline Replay replay(const Model& m, const Log& log) {
  Replay r;
  for (const Trace& t : log)
    replay_trace(m, t, [&](int p, int u, int v, const std::string& lab) {
      if (u == -3) {
        r.restart_into[v] += 1;
        return;
      }
      if (v >= 0) {
        r.trans[{u, v}] += 1;
        if (p >= 0) r.tri[{p, u, v}] += 1;
      } else {
        r.esc[u] += 1;
        if (u >= 0) r.esc_label[{u, lab}] += 1;
        if (p >= 0) r.tri[{p, u, -1}] += 1;
      }
    });
  return r;
}

struct MdlParams {
  double w = 0.5;          // bits per free parameter per log2(visits): 0.5 is BIC
  double alphabet = 2.0;   // |labels| + 1 (the end symbol included); set from the training log
  double edge_factor = 2.0;  // an edge costs edge_factor log2(nodes + 2) bits (K7 amendment 3, E1: 1)
};

/// Transition probabilities from training counts: (c + 0.5) / (N_u + 0.5 (deg_u + 1)).
struct Probs {
  const Model* m = nullptr;
  const Replay* tr = nullptr;
  double p_edge(int u, int v) const {
    const double c = get(tr->trans, {u, v});
    return (c + 0.5) / (visits(u) + 0.5 * (static_cast<double>(m->nodes[static_cast<std::size_t>(u)].out.size()) + 1.0));
  }
  double p_esc(int u) const {
    if (u < 0) return 1.0;
    const auto it = tr->esc.find(u);
    const double c = it == tr->esc.end() ? 0.0 : it->second;
    return (c + 0.5) / (visits(u) + 0.5 * (static_cast<double>(m->nodes[static_cast<std::size_t>(u)].out.size()) + 1.0));
  }
  double visits(int u) const {
    double n = 0;
    for (int v : m->nodes[static_cast<std::size_t>(u)].out) n += get(tr->trans, {u, v});
    const auto it = tr->esc.find(u);
    if (it != tr->esc.end()) n += it->second;
    return n;
  }
  static double get(const std::map<std::pair<int, int>, double>& mm, std::pair<int, int> k) {
    const auto it = mm.find(k);
    return it == mm.end() ? 0.0 : it->second;
  }
};

/// L(traces | model) in bits for `eval`, with probabilities from the training replay `tr`; per-trace bits kept.
inline double trace_bits(const Model& m, const Replay& tr, const Log& eval, const MdlParams& P,
                         std::vector<double>* per_trace = nullptr) {
  Probs pr{&m, &tr};
  const double name_bits = std::log2(P.alphabet);
  double total = 0;
  if (per_trace) per_trace->assign(eval.size(), 0.0);
  for (std::size_t i = 0; i < eval.size(); ++i) {
    double b = 0;
    replay_trace(m, eval[i], [&](int, int u, int v, const std::string&) {
      if (u == -3) return;
      if (v >= 0) b -= std::log2(pr.p_edge(u, v));
      else b += -std::log2(pr.p_esc(u)) + name_bits;
    });
    total += b;
    if (per_trace) (*per_trace)[i] = b;
  }
  return total;
}

/// L(model) in bits: labels, edges, and w log2(visits) per free parameter of each node.
inline double model_bits(const Model& m, const Replay& tr, const MdlParams& P) {
  Probs pr{&m, &tr};
  const double name_bits = std::log2(P.alphabet);
  const double edge_bits = P.edge_factor * std::log2(static_cast<double>(m.nodes.size()) + 2.0);
  double b = 0;
  for (std::size_t v = 2; v < m.nodes.size(); ++v) b += static_cast<double>(m.nodes[v].labels.size()) * name_bits;
  b += static_cast<double>(m.edge_count()) * edge_bits;
  for (std::size_t u = 0; u < m.nodes.size(); ++u) {
    const double deg = static_cast<double>(m.nodes[u].out.size());
    if (deg <= 0) continue;
    b += P.w * std::log2(std::max(2.0, pr.visits(static_cast<int>(u)))) * deg;
  }
  return b;
}

inline double description_length(const Model& m, const Log& train, const MdlParams& P) {
  const Replay r = replay(m, train);
  return model_bits(m, r, P) + trace_bits(m, r, train, P);
}

inline double alphabet_of(const Log& log) {
  std::set<std::string> s;
  for (const Trace& t : log) s.insert(t.begin(), t.end());
  return static_cast<double>(s.size()) + 2.0;  // the labels, the end symbol, and one for "unknown"
}

// ---------------------------------------------------------------------------
// Edits (THOUGHT 6.4)
// ---------------------------------------------------------------------------

enum class EditKind { RemoveEdge, AddEdge, Split, Merge };

inline const char* edit_name(EditKind k) {
  switch (k) {
    case EditKind::RemoveEdge: return "remove_edge";
    case EditKind::AddEdge: return "add_edge";
    case EditKind::Split: return "split";
    case EditKind::Merge: return "merge";
  }
  return "?";
}

struct Edit {
  EditKind kind = EditKind::RemoveEdge;
  int u = -1, v = -1;
  std::vector<int> group;  // Split: the predecessors moved to the copy
  std::string text;
};

/// Split v: a copy v' with v's labels and out-edges; the predecessors in `group` now point at v'.
inline Model apply_split(const Model& m, int v, const std::vector<int>& group) {
  Model x = m;
  const int c = x.add_node(x.nodes[static_cast<std::size_t>(v)].labels);
  x.nodes[static_cast<std::size_t>(c)].out = x.nodes[static_cast<std::size_t>(v)].out;
  for (int p : group) {
    x.remove_edge(p, v);
    x.nodes[static_cast<std::size_t>(p)].out.push_back(c);
  }
  return x;
}

/// Merge a -> b into one step (a always followed by b, b always entered from a).
inline Model apply_merge(const Model& m, int a, int b) {
  Model x = m;
  std::vector<std::string> lab = x.nodes[static_cast<std::size_t>(a)].labels;
  lab.insert(lab.end(), x.nodes[static_cast<std::size_t>(b)].labels.begin(), x.nodes[static_cast<std::size_t>(b)].labels.end());
  x.nodes[static_cast<std::size_t>(a)].labels = lab;
  x.nodes[static_cast<std::size_t>(a)].out = x.nodes[static_cast<std::size_t>(b)].out;
  for (int& w : x.nodes[static_cast<std::size_t>(a)].out)
    if (w == b) w = a;  // a self-loop on b becomes one on the merged step
  x.remove_node(b);
  return x;
}

inline Model apply_edit(const Model& m, const Edit& e) {
  Model x = m;
  switch (e.kind) {
    case EditKind::RemoveEdge: x.remove_edge(e.u, e.v); return x;
    case EditKind::AddEdge: x.add_edge(e.u, e.v); return x;
    case EditKind::Split: return apply_split(m, e.v, e.group);
    case EditKind::Merge: return apply_merge(m, e.u, e.v);
  }
  return x;
}

inline std::string node_text(const Model& m, int v) {
  std::string s;
  for (const std::string& l : m.nodes[static_cast<std::size_t>(v)].labels) s += (s.empty() ? "" : "+") + l;
  return s + "#" + std::to_string(v);
}

/// The candidate edits on a training replay.
inline std::vector<Edit> candidate_edits(const Model& m, const Replay& r, int add_min = 5, int split_max_preds = 8) {
  std::vector<Edit> c;
  for (int u = 0; u < static_cast<int>(m.nodes.size()); ++u)
    for (int v : m.nodes[static_cast<std::size_t>(u)].out)
      c.push_back({EditKind::RemoveEdge, u, v, {}, "remove " + node_text(m, u) + "->" + node_text(m, v)});
  for (const auto& [ul, n] : r.esc_label) {
    if (n < add_min) continue;
    const int u = ul.first;
    for (int v = 1; v < static_cast<int>(m.nodes.size()); ++v)
      if (m.nodes[static_cast<std::size_t>(v)].labels.front() == ul.second && m.can_add(u, v))
        c.push_back({EditKind::AddEdge, u, v, {}, "add " + node_text(m, u) + "->" + node_text(m, v)});
  }
  for (int v = 2; v < static_cast<int>(m.nodes.size()); ++v) {
    std::vector<int> preds;
    for (int p = 0; p < static_cast<int>(m.nodes.size()); ++p)
      if (m.has_edge(p, v) && p != v) preds.push_back(p);
    if (preds.size() < 2 || static_cast<int>(preds.size()) > split_max_preds) continue;
    const std::uint32_t K = static_cast<std::uint32_t>(preds.size());
    for (std::uint32_t mask = 1; mask < (1u << (K - 1)); ++mask) {  // the last predecessor stays with v
      std::vector<int> g;
      for (std::uint32_t i = 0; i < K; ++i)
        if (mask >> i & 1u) g.push_back(preds[i]);
      std::string t = "split " + node_text(m, v) + " {";
      for (int p : g) t += node_text(m, p) + " ";
      c.push_back({EditKind::Split, -1, v, g, t + "}"});
    }
  }
  for (int a = 2; a < static_cast<int>(m.nodes.size()); ++a)
    for (int b : m.nodes[static_cast<std::size_t>(a)].out) {
      if (b < 2 || b == a) continue;
      if (m.nodes[static_cast<std::size_t>(a)].out.size() != 1) continue;  // a always goes to b
      bool only_a = true;
      for (int p = 0; p < static_cast<int>(m.nodes.size()) && only_a; ++p)
        if (p != a && m.has_edge(p, b)) only_a = false;
      if (!only_a) continue;
      const auto e = r.esc.find(a);
      if (e != r.esc.end() && e->second > 0) continue;  // a has escaped: not always followed by b
      const auto ri = r.restart_into.find(b);
      if (ri != r.restart_into.end() && ri->second > 0) continue;
      c.push_back({EditKind::Merge, a, b, {}, "merge " + node_text(m, a) + "+" + node_text(m, b)});
    }
  return c;
}

struct EditStep {
  Edit edit;
  double dl_before = 0, dl_after = 0;  // training description length
  Model after;
};

/// Bits of `eval` under `m`, with probabilities fitted on `fit`.
inline double fold_bits(const Model& m, const Log& fit, const Log& eval, const MdlParams& P) {
  const Replay r = replay(m, fit);
  return trace_bits(m, r, eval, P);
}

/// Greedy MDL edit search: apply the candidate with the largest drop in description length until none lowers it.
/// With a validation fold (K7 amendment 3, E2) an edit must also lower the fold's bits (probabilities fitted on
/// `train`); the candidates are tried in order of their drop in description length.
inline std::vector<EditStep> edit_search(const Model& start, const Log& train, const MdlParams& P, int max_edits = 30,
                                         const Log* valid = nullptr) {
  std::vector<EditStep> steps;
  Model cur = start;
  double cur_dl = description_length(cur, train, P);
  double cur_val = valid ? fold_bits(cur, train, *valid, P) : 0.0;
  for (int it = 0; it < max_edits; ++it) {
    const Replay r = replay(cur, train);
    const std::vector<Edit> cand = candidate_edits(cur, r);
    std::vector<std::pair<double, std::size_t>> drops;
    std::vector<Model> models(cand.size());
    for (std::size_t i = 0; i < cand.size(); ++i) {
      models[i] = apply_edit(cur, cand[i]);
      const double dl = description_length(models[i], train, P);
      if (dl < cur_dl - 1e-9) drops.push_back({dl, i});
    }
    std::stable_sort(drops.begin(), drops.end());
    int bi = -1;
    double best = cur_dl, best_val = cur_val;
    for (const auto& [dl, i] : drops) {
      if (valid) {
        const double vb = fold_bits(models[i], train, *valid, P);
        if (!(vb < cur_val - 1e-9)) continue;
        best_val = vb;
      }
      best = dl;
      bi = static_cast<int>(i);
      break;
    }
    if (bi < 0) break;
    steps.push_back({cand[static_cast<std::size_t>(bi)], cur_dl, best, models[static_cast<std::size_t>(bi)]});
    cur = models[static_cast<std::size_t>(bi)];
    cur_dl = best;
    cur_val = best_val;
  }
  return steps;
}

/// Held-out bits per trace of `m`, with probabilities fitted on `train`.
inline std::vector<double> heldout_bits(const Model& m, const Log& train, const Log& held, const MdlParams& P) {
  const Replay r = replay(m, train);
  std::vector<double> v;
  trace_bits(m, r, held, P, &v);
  return v;
}

/// THOUGHT 6.5: the edited model replaces the start model only if LCI >= 0.95 on held-out traces. Gains are bits
/// saved per trace; gamma compares the held-out gain with the training gain; sigma reads three thirds of the held-out.
inline eval::LciDecision promote_lci(const Model& start, const Model& edited, const Log& train, const Log& held,
                                     const MdlParams& P) {
  const std::vector<double> h0 = heldout_bits(start, train, held, P), h1 = heldout_bits(edited, train, held, P);
  const std::vector<double> t0 = heldout_bits(start, train, train, P), t1 = heldout_bits(edited, train, train, P);
  eval::LciInputs in;
  for (std::size_t i = 0; i < h0.size(); ++i) in.test_diffs.push_back(h0[i] - h1[i]);
  double gt = 0;
  for (std::size_t i = 0; i < t0.size(); ++i) gt += t0[i] - t1[i];
  in.gain_dev = t0.empty() ? 0.0 : gt / static_cast<double>(t0.size());
  double gh = 0;
  for (double d : in.test_diffs) gh += d;
  in.gain_ext = in.test_diffs.empty() ? 0.0 : gh / static_cast<double>(in.test_diffs.size());
  const std::size_t n = in.test_diffs.size();
  for (int k = 0; k < 3; ++k) {
    double s = 0;
    std::size_t c = 0;
    for (std::size_t i = static_cast<std::size_t>(k) * n / 3; i < static_cast<std::size_t>(k + 1) * n / 3; ++i, ++c) s += in.test_diffs[i];
    in.seed_gains.push_back(c ? s / static_cast<double>(c) : 0.0);
  }
  return eval::compute_lci(in);
}

// ---------------------------------------------------------------------------
// Planted processes, traces, defects (the K7 benchmarks)
// ---------------------------------------------------------------------------

struct Planted {
  Model model;
  std::map<std::pair<int, int>, double> prob;  // branch probabilities
};

/// A planted process: n_labels labels; `reuse` extra nodes copy an existing label (K7b); a backbone with 1-3
/// successors per non-final node (distinct first labels); a back edge with probability p_loop; flat-Dirichlet
/// branch probabilities.
inline Planted make_planted(Rng& rng, int n_labels, int reuse, double p_loop = 0.3) {
  Planted P;
  Model& m = P.model;
  const int n = n_labels + reuse;
  std::vector<int> ids;
  for (int i = 0; i < n; ++i) {
    std::string lab;
    if (i < n_labels) lab = "s" + std::to_string(i);
    else lab = m.nodes[static_cast<std::size_t>(2 + static_cast<int>(uniform01(rng) * n_labels))].labels.front();
    ids.push_back(m.add_node({lab}));
  }
  for (int i = n - 1; i > 0; --i) {  // a random order of the nodes
    const int j = static_cast<int>(uniform01(rng) * (i + 1));
    std::swap(ids[static_cast<std::size_t>(i)], ids[static_cast<std::size_t>(j)]);
  }
  m.add_edge(Model::kStartNode, ids[0]);
  for (int k = 0; k < n; ++k) {
    const int u = ids[static_cast<std::size_t>(k)];
    const int nxt = k + 1 < n ? ids[static_cast<std::size_t>(k + 1)] : Model::kEndNode;
    m.add_edge(u, nxt);
    const int extra = static_cast<int>(uniform01(rng) * 3);  // 0-2 more successors
    for (int e = 0; e < extra; ++e) {
      int v;
      if (uniform01(rng) < 0.15) v = Model::kEndNode;
      else {
        const int span = n - (k + 1);
        if (span <= 1) continue;
        v = ids[static_cast<std::size_t>(k + 2 + static_cast<int>(uniform01(rng) * (span - 1)))];
      }
      if (m.can_add(u, v)) m.add_edge(u, v);
    }
  }
  if (uniform01(rng) < p_loop && n >= 3) {  // one back edge
    for (int tries = 0; tries < 20; ++tries) {
      const int a = 1 + static_cast<int>(uniform01(rng) * (n - 1));
      const int b = static_cast<int>(uniform01(rng) * a);
      const int u = ids[static_cast<std::size_t>(a)], v = ids[static_cast<std::size_t>(b)];
      if (m.can_add(u, v)) {
        m.add_edge(u, v);
        break;
      }
    }
  }
  for (int u = 0; u < static_cast<int>(m.nodes.size()); ++u) {
    const auto& o = m.nodes[static_cast<std::size_t>(u)].out;
    std::vector<double> g(o.size());
    double s = 0;
    for (double& x : g) s += (x = gamma_sample(rng, 1.0));
    for (std::size_t i = 0; i < o.size(); ++i) P.prob[{u, o[i]}] = s > 0 ? g[i] / s : 1.0 / static_cast<double>(o.size());
  }
  return P;
}

/// A random walk of at most `cap` events; `walk_counts` (optional) gets the planted edges traversed.
inline Trace walk(const Planted& P, Rng& rng, int cap = 60, std::map<std::pair<int, int>, int>* walk_counts = nullptr) {
  Trace t;
  int cur = Model::kStartNode;
  while (static_cast<int>(t.size()) < cap) {
    const auto& o = P.model.nodes[static_cast<std::size_t>(cur)].out;
    if (o.empty()) break;
    double u = uniform01(rng), acc = 0;
    int nxt = o.back();
    for (int v : o) {
      acc += P.prob.at({cur, v});
      if (u < acc) {
        nxt = v;
        break;
      }
    }
    if (walk_counts) ++(*walk_counts)[{cur, nxt}];
    if (nxt == Model::kEndNode) break;
    t.push_back(P.model.nodes[static_cast<std::size_t>(nxt)].labels.front());
    cur = nxt;
  }
  return t;
}

/// One random corruption: an adjacent swap, an inserted random label, or a deleted event.
inline void corrupt(Trace& t, const std::vector<std::string>& labels, Rng& rng) {
  const double u = uniform01(rng);
  if (u < 1.0 / 3.0 && t.size() >= 2) {
    const std::size_t i = static_cast<std::size_t>(uniform01(rng) * static_cast<double>(t.size() - 1));
    std::swap(t[i], t[i + 1]);
  } else if (u < 2.0 / 3.0 || t.empty()) {
    const std::size_t i = static_cast<std::size_t>(uniform01(rng) * static_cast<double>(t.size() + 1));
    t.insert(t.begin() + static_cast<std::ptrdiff_t>(i), labels[static_cast<std::size_t>(uniform01(rng) * static_cast<double>(labels.size()))]);
  } else {
    const std::size_t i = static_cast<std::size_t>(uniform01(rng) * static_cast<double>(t.size()));
    t.erase(t.begin() + static_cast<std::ptrdiff_t>(i));
  }
}

inline Log make_log(const Planted& P, Rng& rng, int n, double noise, std::map<std::pair<int, int>, int>* walk_counts = nullptr) {
  std::vector<std::string> labels;
  for (std::size_t v = 2; v < P.model.nodes.size(); ++v) labels.push_back(P.model.nodes[v].labels.front());
  std::sort(labels.begin(), labels.end());
  labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
  Log log;
  for (int i = 0; i < n; ++i) {
    Trace t = walk(P, rng, 60, walk_counts);
    if (uniform01(rng) < noise) corrupt(t, labels, rng);
    log.push_back(std::move(t));
  }
  return log;
}

/// The set of label trigrams along every path p -> u -> s (expanded through merged steps): a fingerprint of the
/// process's structure that does not depend on node numbering or on merges.
inline std::set<std::string> trigrams(const Model& m) {
  std::set<std::string> out;
  for (int u = 0; u < static_cast<int>(m.nodes.size()); ++u) {
    std::vector<int> preds;
    for (int p = 0; p < static_cast<int>(m.nodes.size()); ++p)
      if (m.has_edge(p, u)) preds.push_back(p);
    for (int p : preds)
      for (int s : m.nodes[static_cast<std::size_t>(u)].out) {
        std::vector<std::string> seq;
        seq.push_back(m.nodes[static_cast<std::size_t>(p)].labels.back());
        for (const std::string& l : m.nodes[static_cast<std::size_t>(u)].labels) seq.push_back(l);
        seq.push_back(m.nodes[static_cast<std::size_t>(s)].labels.front());
        for (std::size_t i = 0; i + 2 < seq.size(); ++i) out.insert(seq[i] + "\x1f" + seq[i + 1] + "\x1f" + seq[i + 2]);
      }
  }
  return out;
}

enum class DefectKind { RemovedEdge, SpuriousEdge, Collapse };

inline const char* defect_name(DefectKind k) {
  switch (k) {
    case DefectKind::RemovedEdge: return "removed_edge";
    case DefectKind::SpuriousEdge: return "spurious_edge";
    case DefectKind::Collapse: return "collapse";
  }
  return "?";
}

/// Collapse y into x (same label): y's in-edges go to x; y's out-edges join x unless x already has a successor
/// with that first label.
inline Model collapse(const Model& m, int x, int y) {
  Model c = m;
  for (Node& n : c.nodes)
    for (int& w : n.out)
      if (w == y) w = x;
  for (Node& n : c.nodes) {  // a predecessor of both now lists x twice
    std::vector<int> o;
    for (int w : n.out)
      if (std::find(o.begin(), o.end(), w) == o.end()) o.push_back(w);
    n.out = o;
  }
  for (int w : m.nodes[static_cast<std::size_t>(y)].out) {
    const int t = w == y ? x : w;
    if (t != Model::kStartNode && c.succ_with_label(x, c.nodes[static_cast<std::size_t>(t)].labels.front()) < 0) c.add_edge(x, t);
  }
  c.remove_node(y);
  return c;
}

struct Defect {
  DefectKind kind;
  std::set<std::string> delta;  // the trigrams this defect alone changes against the planted model
  std::string text;
};

/// Apply one defect of `kind` (eligible picks drawn uniformly); returns false when none is eligible. Node ids must still
/// match `planted` and `walk_counts` for the edge defects, so a collapse (which renumbers) is applied last.
inline bool apply_defect(const Model& planted, Model& m, DefectKind kind, Rng& rng,
                         const std::map<std::pair<int, int>, int>& walk_counts, Defect& d) {
  d.kind = kind;
  const auto tri_before = trigrams(m);
  if (kind == DefectKind::RemovedEdge) {
    std::vector<std::pair<int, int>> el;
    for (int u = 0; u < static_cast<int>(m.nodes.size()); ++u)
      for (int v : m.nodes[static_cast<std::size_t>(u)].out) {
        const auto it = walk_counts.find({u, v});
        if (u >= 2 && v >= 2 && it != walk_counts.end() && it->second >= 20) el.push_back({u, v});
      }
    if (el.empty()) return false;
    const auto e = el[static_cast<std::size_t>(uniform01(rng) * static_cast<double>(el.size()))];
    m.remove_edge(e.first, e.second);
    d.text = "removed " + node_text(m, e.first) + "->" + node_text(m, e.second);
  } else if (kind == DefectKind::SpuriousEdge) {
    std::vector<std::pair<int, int>> el;
    for (int u = 2; u < static_cast<int>(m.nodes.size()); ++u)
      for (int v = 2; v < static_cast<int>(m.nodes.size()); ++v)
        if (u != v && m.can_add(u, v) && !(planted.nodes.size() == m.nodes.size() && planted.has_edge(u, v)))
          el.push_back({u, v});
    if (el.empty()) return false;
    const auto e = el[static_cast<std::size_t>(uniform01(rng) * static_cast<double>(el.size()))];
    m.add_edge(e.first, e.second);
    d.text = "spurious " + node_text(m, e.first) + "->" + node_text(m, e.second);
  } else {
    std::vector<std::pair<int, int>> el;
    for (int x = 2; x < static_cast<int>(m.nodes.size()); ++x)
      for (int y = x + 1; y < static_cast<int>(m.nodes.size()); ++y)
        if (m.nodes[static_cast<std::size_t>(x)].labels == m.nodes[static_cast<std::size_t>(y)].labels) el.push_back({x, y});
    if (el.empty()) return false;
    const auto e = el[static_cast<std::size_t>(uniform01(rng) * static_cast<double>(el.size()))];
    d.text = "collapse " + node_text(m, e.first) + "<-" + node_text(m, e.second);
    m = collapse(m, e.first, e.second);
  }
  const auto tri_after = trigrams(m);
  std::set_symmetric_difference(tri_before.begin(), tri_before.end(), tri_after.begin(), tri_after.end(),
                                std::inserter(d.delta, d.delta.end()));
  return true;
}

/// A defect is repaired when the final model agrees with the planted one on every trigram the defect changed.
inline bool repaired(const Defect& d, const std::set<std::string>& planted_tri, const std::set<std::string>& final_tri) {
  if (d.delta.empty()) return true;
  for (const std::string& t : d.delta)
    if ((planted_tri.count(t) != 0) != (final_tri.count(t) != 0)) return false;
  return true;
}

}  // namespace cypha::thought::mine
