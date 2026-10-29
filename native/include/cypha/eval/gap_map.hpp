// Gap map for Cypha's discovery track (docs/DISCOVERY.md sections 3.2, 3.5 and 6.3; Track S item S3).
//
// Header-only. Includes hypothesis.hpp (EIG and priority, 6.3) and through it epistemic.hpp (ReasonCode); standard
// library only.
//
//   3.5  gap_id(): the id CyphaLM writes in its tag; parse_unknown_tag(): the <|unknown ...|> tag, fields in any order
//   3.2  SchemaCell: the expectation f(T, R); Gap and GapMap (schema, demand, contradiction and certainty gaps)
//   6.3  GapMap::score(): EIG of a placeholder test, Impact = demand, priority = EIG (1 + Impact) / Cost
//
// The gap map is bookkeeping about what the store lacks. It never answers a question (PLAN rule 10).
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cypha/eval/hypothesis.hpp"

namespace cypha::eval {

// ---------------------------------------------------------------------------
// 3.5 The tag
// ---------------------------------------------------------------------------

/// The gap id of 3.5 as CyphaLM writes it (cyphalm_trie's gap_id): FNV-1a 32 over "entity|relation", low 16 bits, "G-%04X".
/// 16 bits collide often in a large map, so the map's key is (kind, entity, relation) and this id is a label.
inline std::string gap_id(const std::string& entity, const std::string& relation) {
  std::uint32_t h = 2166136261u;
  for (unsigned char c : entity + "|" + relation) h = (h ^ c) * 16777619u;
  char b[16];
  std::snprintf(b, sizeof b, "G-%04X", static_cast<unsigned>(h & 0xffffu));
  return b;
}

/// The reason code with this name (one of the 11 of 3.1), or none.
inline std::optional<ReasonCode> reason_from_name(const std::string& s) {
  for (std::size_t i = 0; i < kReasonCodeCount; ++i)
    if (s == reason_name(static_cast<ReasonCode>(i))) return static_cast<ReasonCode>(i);
  return std::nullopt;
}

/// A parsed <|unknown reason=CODE entity="..." relation=REL gap=ID|> tag. The tag runs from "<|unknown" to the first "|>"
/// (or the end of the text when it is never closed). Fields may come in any order. A quoted value that is not closed
/// inside the tag is left empty.
struct UnknownTag {
  bool found = false;
  bool closed = false;
  std::string reason, entity, relation, gap;
  std::optional<ReasonCode> code;

  bool complete() const { return found && closed && code.has_value() && !entity.empty() && !relation.empty() && !gap.empty(); }
  /// The tag's gap id is the one its own entity and relation give.
  bool id_consistent() const { return complete() && gap == gap_id(entity, relation); }
};

inline UnknownTag parse_unknown_tag(const std::string& text) {
  UnknownTag t;
  const std::size_t b = text.find("<|unknown");
  if (b == std::string::npos) return t;
  t.found = true;
  const std::size_t e = text.find("|>", b + 9);
  t.closed = e != std::string::npos;
  const std::string body = text.substr(b + 9, t.closed ? e - b - 9 : std::string::npos);
  for (std::size_t i = 0; i < body.size();) {  // key=value or key="value", left to right; the first of each key counts
    while (i < body.size() && body[i] == ' ') ++i;
    const std::size_t eq = body.find('=', i);
    if (eq == std::string::npos) break;
    const std::string k = body.substr(i, eq - i);
    std::string v;
    std::size_t j = eq + 1;
    if (j < body.size() && body[j] == '"') {
      const std::size_t q = body.find('"', j + 1);
      if (q == std::string::npos) break;  // cut off inside the quotes: no value, and nothing after it can be trusted
      v = body.substr(j + 1, q - j - 1);
      j = q + 1;
    } else {
      while (j < body.size() && body[j] != ' ') ++j;
      v = body.substr(eq + 1, j - eq - 1);
    }
    std::string* dst = k == "reason" ? &t.reason : k == "entity" ? &t.entity : k == "relation" ? &t.relation : k == "gap" ? &t.gap : nullptr;
    if (dst && dst->empty()) *dst = v;
    i = j;
  }
  t.code = reason_from_name(t.reason);
  return t;
}

// ---------------------------------------------------------------------------
// 3.2 The gap map
// ---------------------------------------------------------------------------

/// The kinds the current stores can show. Lexical, structural, coverage and prediction gaps need stores that do not
/// exist yet. Fragile (S9b): a held fact that rests on one source (S9's single point of failure); last, so the others'
/// keys and order are unchanged.
enum class GapKind { Schema, Demand, Contradiction, Certainty, Fragile };
/// Open: a rung of the asking ladder (P1) can run. Queued: open and asked about (a tag queued it, 3.5). Parked: no
/// automatic rung can fill it now. Closed: filled through ECS.
enum class GapStatus { Open, Queued, Parked, Closed };

inline const char* gap_kind_name(GapKind k) {
  switch (k) {
    case GapKind::Schema: return "schema";
    case GapKind::Demand: return "demand";
    case GapKind::Contradiction: return "contradiction";
    case GapKind::Certainty: return "certainty";
    case GapKind::Fragile: return "fragile";
  }
  return "?";
}

inline const char* gap_status_name(GapStatus s) {
  switch (s) {
    case GapStatus::Open: return "open";
    case GapStatus::Queued: return "queued";
    case GapStatus::Parked: return "parked";
    case GapStatus::Closed: return "closed";
  }
  return "?";
}

/// f(T, R): of the entities of type T, the share with at least one value of R. An entity of type T lacking R is a schema
/// gap with P(exists) ~ f(T, R).
struct SchemaCell {
  std::size_t entities = 0;
  std::size_t with_value = 0;
  double f() const { return entities ? static_cast<double>(with_value) / static_cast<double>(entities) : 0.0; }
};

struct Gap {
  std::string id;  // gap_id(entity, relation)
  GapKind kind = GapKind::Schema;
  GapStatus status = GapStatus::Open;
  std::string entity, type, relation;
  double p_exists = 0.0;  // prior that a value exists (schema gaps: f(T, R))
  double p = 0.5;         // probability of the binary hypothesis a test would check (set by the caller per kind)
  int demand = 0;         // "I don't know" tags that named this slot
  double eig = 0.0, impact = 0.0, cost = 1.0, voi = 0.0;
  std::string note;
};

/// The test behind VOI until S7 measures each test kind: sensitivity, specificity and cost are placeholders.
struct VoiConfig {
  double sensitivity = 0.9;
  double specificity = 0.9;
  double cost = 1.0;
  double demand_weight = 1.0;  // Impact = weight x demand; below 1 when tags are known to name the wrong slot
};

class GapMap {
 public:
  /// Entity key normaliser; the default lowers ASCII letters. A caller with its own normaliser sets it before adding gaps.
  std::function<std::string(const std::string&)> norm = [](const std::string& s) {
    std::string o = s;
    for (char& c : o) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return o;
  };

  /// The gap of this kind for (entity, relation): created on first use, returned after.
  Gap& open(GapKind kind, const std::string& entity, const std::string& relation, const std::string& type = "") {
    const std::string k = key(kind, entity, relation);
    auto it = index_.find(k);
    if (it != index_.end()) return gaps_[it->second];
    Gap g;
    g.id = gap_id(entity, relation);
    g.kind = kind;
    g.entity = entity;
    g.relation = relation;
    g.type = type;
    index_[k] = gaps_.size();
    gaps_.push_back(std::move(g));
    return gaps_.back();
  }

  Gap* find(GapKind kind, const std::string& entity, const std::string& relation) {
    auto it = index_.find(key(kind, entity, relation));
    return it == index_.end() ? nullptr : &gaps_[it->second];
  }

  /// 3.5: a tag adds demand +1 to its slot's gap and queues it for gap closing. The slot's schema gap is used first, then
  /// a contradiction or certainty gap; with none of them a demand gap is opened. A tag with no entity or no relation
  /// cannot name a slot and returns nullptr.
  Gap* add_demand(const UnknownTag& t, const std::string& type = "") {
    if (t.entity.empty() || t.relation.empty()) return nullptr;
    Gap* g = nullptr;
    for (GapKind k : {GapKind::Schema, GapKind::Contradiction, GapKind::Certainty, GapKind::Demand})
      if (!g) g = find(k, t.entity, t.relation);
    if (!g) {
      g = &open(GapKind::Demand, t.entity, t.relation, type);
      g->p_exists = g->p = 0.5;  // nothing is known about this slot
    }
    ++g->demand;
    if (g->status == GapStatus::Open) g->status = GapStatus::Queued;
    return g;
  }

  /// S9b: a held fact that a single source holds up (S9's single point of failure) as a fragile gap. A slot that already
  /// has a contradiction or certainty gap gets none (those already ask about it); a slot keeps its first fragile gap.
  /// p is the fact's P, p_exists 1, demand 0. Returns the gap, or nullptr when the slot is skipped.
  Gap* add_fragile(const std::string& entity, const std::string& relation, double p, const std::string& type = "",
                   const std::string& note = "") {
    if (entity.empty() || relation.empty()) return nullptr;
    if (find(GapKind::Contradiction, entity, relation) || find(GapKind::Certainty, entity, relation)) return nullptr;
    if (Gap* g = find(GapKind::Fragile, entity, relation)) return g;
    Gap& g = open(GapKind::Fragile, entity, relation, type);
    g.p = p;
    g.p_exists = 1.0;
    g.note = note;
    return &g;
  }

  /// 6.3 for every gap: EIG of the placeholder test on the gap's p, Impact = weight x demand, I+ = 0, priority = VOI.
  void score(const VoiConfig& c = {}) {
    for (Gap& g : gaps_) {
      g.eig = expected_information_gain(g.p, c.sensitivity, c.specificity).eig;
      g.impact = c.demand_weight * g.demand;
      g.cost = c.cost;
      g.voi = voi_priority(g.eig, g.impact, 0.0, g.cost);
    }
  }

  /// Gaps by VOI, highest first (ties: demand, then kind, entity, relation).
  std::vector<const Gap*> ranked() const {
    std::vector<const Gap*> v;
    for (const Gap& g : gaps_) v.push_back(&g);
    std::stable_sort(v.begin(), v.end(), [](const Gap* a, const Gap* b) {
      if (a->voi != b->voi) return a->voi > b->voi;
      if (a->demand != b->demand) return a->demand > b->demand;
      if (a->kind != b->kind) return a->kind < b->kind;
      if (a->entity != b->entity) return a->entity < b->entity;
      return a->relation < b->relation;
    });
    return v;
  }

  const std::deque<Gap>& gaps() const { return gaps_; }
  std::size_t size() const { return gaps_.size(); }

 private:
  std::string key(GapKind kind, const std::string& entity, const std::string& relation) const {
    return std::to_string(static_cast<int>(kind)) + "\x1f" + norm(entity) + "\x1f" + relation;
  }

  std::deque<Gap> gaps_;  // a deque: references returned by open() stay valid
  std::unordered_map<std::string, std::size_t> index_;
};

}  // namespace cypha::eval
