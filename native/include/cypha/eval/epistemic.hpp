// Epistemic layer for Cypha's discovery track (docs/DISCOVERY.md sections 2.1-2.3, 3.1, 3.3 and 10).
//
// Header-only. Includes ecs.hpp and reuses its Opinion and Grade types; standard library only.
//
//   2.1  EpistemicState and classify(opinion, optional rival opinion)
//   2.2  split_uncertainty(): lack of evidence versus mixed evidence, with documented thresholds
//   2.3  degree_of_conflict(): DC = |P_A - P_B| (1 - u_A)(1 - u_B)
//   3.1  ReasonCode (11 codes, priority order), ReasonBreakdown, explain_unknown(), first_reason()
//   3.3  unseen estimators: Good-Turing missing mass, Chao1, Chapman capture-recapture, Heaps' law fit
//   10   plain-text cards: answer, "I don't know", hypothesis, hypothetical
//
// Every threshold that the spec leaves open is a named field with a default and a comment saying where the
// number comes from. Nothing here reads or writes a store; the caller supplies the statistics.
//
// Author: Odin Loch
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cypha/eval/ecs.hpp"

namespace cypha::eval {

// UTF-8 pieces used by the cards, spelled as bytes so the output does not depend on the compiler's execution charset.
namespace epi_text {
inline constexpr const char* kEmDash = "\xE2\x80\x94";   // U+2014
inline constexpr const char* kEnDash = "\xE2\x80\x93";   // U+2013
inline constexpr const char* kMidDot = "\xC2\xB7";       // U+00B7
inline constexpr const char* kLeq = "\xE2\x89\xA4";      // U+2264
inline constexpr const char* kArrow = "\xE2\x86\x92";    // U+2192
inline constexpr const char* kEllipsis = "\xE2\x80\xA6"; // U+2026
}  // namespace epi_text

// ---------------------------------------------------------------------------
// 2.3 Degree of conflict
// ---------------------------------------------------------------------------

/// DC(A, B) = |P_A - P_B| (1 - u_A)(1 - u_B). High only when both opinions are confident and they disagree.
/// A and B must be opinions about the same proposition (two sources, two times); u is clamped to [0, 1].
inline double degree_of_conflict(double p_a, double u_a, double p_b, double u_b) {
  const double ua = detail::ecs_clamp01(u_a);
  const double ub = detail::ecs_clamp01(u_b);
  return std::fabs(p_a - p_b) * (1.0 - ua) * (1.0 - ub);
}

inline double degree_of_conflict(const Opinion& a, const Opinion& b) {
  return degree_of_conflict(a.P, a.u, b.P, b.u);
}

// ---------------------------------------------------------------------------
// 2.1 Epistemic states
// ---------------------------------------------------------------------------

enum class EpistemicState { Known, Believed, Contested, Ignorant, Vacuous, Hypothesis, Refuted };

inline const char* state_name(EpistemicState s) {
  switch (s) {
    case EpistemicState::Known: return "Known";
    case EpistemicState::Believed: return "Believed";
    case EpistemicState::Contested: return "Contested";
    case EpistemicState::Ignorant: return "Ignorant";
    case EpistemicState::Vacuous: return "Vacuous";
    case EpistemicState::Hypothesis: return "Hypothesis";
    case EpistemicState::Refuted: return "Refuted";
  }
  return "?";
}

/// The thresholds of the 2.1 table. Defaults are the table's numbers.
struct ClassifyConfig {
  double contested_d = 0.5;    // Contested: d >= 0.5 ...
  double contested_dc = 0.3;   // ... or DC with a rival value >= 0.3
  double ignorant_u = 0.8;     // Ignorant: u >= 0.8
  double vacuous_u = 1.0 - 1e-9;  // Vacuous: no evidence at all (u = 1, with a tolerance for rounding)
  /// The table has no row for Grade Very low with u < 0.8 and d < 0.5 (P < 0.5, some evidence, leaning
  /// against). Such an item must not be answered from, so by default it is treated as a gap (Ignorant).
  EpistemicState very_low_state = EpistemicState::Ignorant;
};

/// State of a fact, pattern, rule, concept link or sense from its ECS opinion and, optionally, the opinion of
/// a rival value for the same functional slot. Checked in this order, first match wins:
///   Vacuous   u = 1
///   Contested d >= 0.5 (or Grade Contradicted), or DC(opinion, rival) >= 0.3
///   Ignorant  u >= 0.8
///   Known     Grade High
///   Believed  Grade Moderate or Low
///   otherwise (Grade Very low) cfg.very_low_state
/// The order matters only where the conditions overlap. Vacuous, Ignorant and the two Contested routes cannot
/// overlap each other (u >= 0.8 forces d <= 0.2 and DC <= 0.2). Contested does overlap Known and Believed, and
/// wins: a Known item with a confident rival is reported as contested, never answered.
/// Hypothesis and Refuted are not derivable from an opinion; they are properties of the hypothesis store.
inline EpistemicState classify(const Opinion& o, const std::optional<Opinion>& rival = std::nullopt,
                               const ClassifyConfig& cfg = {}) {
  if (o.u >= cfg.vacuous_u) return EpistemicState::Vacuous;
  if (o.d >= cfg.contested_d || o.grade == Grade::Contradicted) return EpistemicState::Contested;
  if (rival && degree_of_conflict(o, *rival) >= cfg.contested_dc) return EpistemicState::Contested;
  if (o.u >= cfg.ignorant_u) return EpistemicState::Ignorant;
  if (o.grade == Grade::High) return EpistemicState::Known;
  if (o.grade == Grade::Moderate || o.grade == Grade::Low) return EpistemicState::Believed;
  return cfg.very_low_state;
}

/// State of an item that lives in the hypothesis store: Refuted once its tests went against it.
inline EpistemicState hypothesis_state(bool refuted) {
  return refuted ? EpistemicState::Refuted : EpistemicState::Hypothesis;
}

/// The "May be used for" column of the 2.1 table.
struct StateUse {
  bool answer = false;               // may be given as an answer
  bool answer_states_grade = false;   // ... and then the grade must be stated (Believed)
  bool premise = false;              // may be a premise for inference
  bool discovery_only = false;       // Hypothesis: discovery mode only, never an answer
  bool blocks_regeneration = false;  // Refuted: kept as a negative result
};

inline StateUse state_use(EpistemicState s, Grade g) {
  StateUse u;
  switch (s) {
    case EpistemicState::Known: u.answer = true; u.premise = true; break;
    case EpistemicState::Believed:
      u.answer = true;
      u.answer_states_grade = true;
      u.premise = (g == Grade::Moderate);  // "not as an inference premise unless Moderate"
      break;
    case EpistemicState::Hypothesis: u.discovery_only = true; break;
    case EpistemicState::Refuted: u.blocks_regeneration = true; break;
    case EpistemicState::Contested:
    case EpistemicState::Ignorant:
    case EpistemicState::Vacuous: break;  // never
  }
  return u;
}

// ---------------------------------------------------------------------------
// 2.2 Two kinds of uncertainty
// ---------------------------------------------------------------------------

enum class UncertaintyKind {
  OneSided,        // plenty of evidence and it points one way
  LackOfEvidence,  // u high: it has not seen enough -> look for evidence (P1 gap closing)
  MixedEvidence,   // u low, P near 0.5, b and d both substantial -> look for a hidden variable (P4)
  Unclear          // u between the two bands: neither signal is strong enough to name
};

inline const char* uncertainty_name(UncertaintyKind k) {
  switch (k) {
    case UncertaintyKind::OneSided: return "one-sided";
    case UncertaintyKind::LackOfEvidence: return "lack of evidence";
    case UncertaintyKind::MixedEvidence: return "mixed evidence";
    case UncertaintyKind::Unclear: return "unclear";
  }
  return "?";
}

/// The action 2.2 attaches to each kind.
inline const char* uncertainty_action(UncertaintyKind k) {
  switch (k) {
    case UncertaintyKind::LackOfEvidence: return "look for evidence (gap closing, P1)";
    case UncertaintyKind::MixedEvidence: return "look for a hidden variable: another time, sense, entity or qualifier (P4)";
    case UncertaintyKind::OneSided: return "none needed";
    case UncertaintyKind::Unclear: return "collect more evidence, then re-split";
  }
  return "?";
}

/// Documented thresholds for the 2.2 split. The spec says "u high" and "u low, P near 0.5, b and d both
/// substantial" without numbers; these are the choices.
struct UncertaintyConfig {
  double lack_u = 0.5;         // u >= 0.5: under about two effective observations at W = 2
  double settled_u = 0.3;      // u <= 0.3: the Moderate grade's own cut-off (PLAN 7.3)
  double mixed_min_side = 0.2; // b >= 0.2 and d >= 0.2: both sides substantial
  double mixed_p_band = 0.2;   // |P - 0.5| <= 0.2: P near 0.5
};

struct UncertaintySplit {
  UncertaintyKind kind = UncertaintyKind::Unclear;
  double lack_of_evidence = 1.0;  // = u, in [0, 1]
  double mixed_evidence = 0.0;    // = 2 min(b, d), in [0, 1]: large only when b and d are both large
};

inline UncertaintySplit split_uncertainty(const Opinion& o, const UncertaintyConfig& cfg = {}) {
  UncertaintySplit s;
  s.lack_of_evidence = detail::ecs_clamp01(o.u);
  s.mixed_evidence = detail::ecs_clamp01(2.0 * std::min(o.b, o.d));
  if (o.u >= cfg.lack_u) {
    s.kind = UncertaintyKind::LackOfEvidence;
  } else if (o.u <= cfg.settled_u) {
    const bool mixed = o.b >= cfg.mixed_min_side && o.d >= cfg.mixed_min_side &&
                       std::fabs(o.P - 0.5) <= cfg.mixed_p_band;
    s.kind = mixed ? UncertaintyKind::MixedEvidence : UncertaintyKind::OneSided;
  } else {
    s.kind = UncertaintyKind::Unclear;
  }
  return s;
}

// ---------------------------------------------------------------------------
// 3.1 Reason codes
// ---------------------------------------------------------------------------

/// The eleven codes of 3.1, in the order in which the first one that applies is reported.
enum class ReasonCode {
  OutOfScope = 0,
  Unparseable,
  AmbiguousQuestion,
  UnknownEntity,
  AmbiguousEntity,
  NoEvidence,
  Conflict,
  LowCertainty,
  Stale,
  BeyondDepth,
  HypothesisOnly
};

inline constexpr std::size_t kReasonCodeCount = 11;

inline const char* reason_name(ReasonCode c) {
  switch (c) {
    case ReasonCode::OutOfScope: return "OUT_OF_SCOPE";
    case ReasonCode::Unparseable: return "UNPARSEABLE";
    case ReasonCode::AmbiguousQuestion: return "AMBIGUOUS_QUESTION";
    case ReasonCode::UnknownEntity: return "UNKNOWN_ENTITY";
    case ReasonCode::AmbiguousEntity: return "AMBIGUOUS_ENTITY";
    case ReasonCode::NoEvidence: return "NO_EVIDENCE";
    case ReasonCode::Conflict: return "CONFLICT";
    case ReasonCode::LowCertainty: return "LOW_CERTAINTY";
    case ReasonCode::Stale: return "STALE";
    case ReasonCode::BeyondDepth: return "BEYOND_DEPTH";
    case ReasonCode::HypothesisOnly: return "HYPOTHESIS_ONLY";
  }
  return "?";
}

/// What each code means, in the words of the "Detected by" column.
inline const char* reason_meaning(ReasonCode c) {
  switch (c) {
    case ReasonCode::OutOfScope: return "the question or its topic is outside what the store covers";
    case ReasonCode::Unparseable: return "no question pattern matches with P >= 0.5";
    case ReasonCode::AmbiguousQuestion: return "two or more relations fit the question about equally";
    case ReasonCode::UnknownEntity: return "no concept links to the mention";
    case ReasonCode::AmbiguousEntity: return "two or more concepts fit the mention";
    case ReasonCode::NoEvidence: return "the slot is empty (vacuous or ignorant)";
    case ReasonCode::Conflict: return "the slot is contested";
    case ReasonCode::LowCertainty: return "a value exists but its confidence is below the cell threshold";
    case ReasonCode::Stale: return "the value is past its relation's half-life";
    case ReasonCode::BeyondDepth: return "answerable only by inference deeper than the depth cap";
    case ReasonCode::HypothesisOnly: return "the only candidate is in the hypothesis store";
  }
  return "?";
}

/// Shannon entropy in bits of a non-negative weight vector (normalised first; empty or all-zero gives 0).
inline double entropy_bits(const std::vector<double>& weights) {
  double total = 0.0;
  for (double w : weights) total += std::max(0.0, w);
  if (!(total > 0.0)) return 0.0;
  double h = 0.0;
  for (double w : weights) {
    if (w <= 0.0) continue;
    const double p = w / total;
    h -= p * std::log2(p);
  }
  return h;
}

/// Thresholds for deciding which codes apply. 3.1 names the statistic but, apart from the ones marked
/// "from 3.1", not the number.
struct ReasonThresholds {
  double ood_score = 0.5;             // OOD score above this -> OUT_OF_SCOPE
  double parse_p = 0.5;               // from 3.1: UNPARSEABLE when P(parse) < 0.5
  double ambiguity_ratio = 0.5;       // "close in probability": second-best >= 0.5 x best ...
  double ambiguity_min_second = 0.15; // ... and the second-best is itself a real candidate (>= 0.15 after normalising)
  double link_p = 0.5;                // UNKNOWN_ENTITY when the best link P is below this
  double stale_theta = 0.5;           // from 3.1: STALE when theta < 0.5
  int depth_cap = kMaxDerivationDepth;  // from 3.1 / PLAN 7.1: 3
  ClassifyConfig classify;            // thresholds for the slot's state
};

/// Everything the caller knows about why a question could not be answered. A field left empty means "not
/// measured" and never makes a code apply (except an empty slot, which is itself the NO_EVIDENCE signal).
struct ReasonInputs {
  // OUT_OF_SCOPE
  std::optional<double> ood_score;
  bool ood_flag = false;                   // Cypha's own OOD flag, when it gives a flag and not a score
  bool topic_outside_every_domain = false;
  // UNPARSEABLE
  std::optional<double> p_parse;           // best question-pattern parse probability
  // AMBIGUOUS_QUESTION
  std::vector<double> relation_probs;      // distribution over the relations that fit the question
  // UNKNOWN_ENTITY / AMBIGUOUS_ENTITY
  std::optional<double> best_link_p;       // best entity-link probability
  std::vector<double> link_probs;          // distribution over link candidates for the mention
  // NO_EVIDENCE / CONFLICT
  std::optional<Opinion> slot;             // the slot's opinion; empty = no value stored
  std::optional<Opinion> rival;            // a rival value's opinion for DC
  // LOW_CERTAINTY
  std::optional<double> answer_conf;       // conf(answer) from PLAN 7.4
  std::optional<double> conf_threshold;    // the cell's abstain threshold
  // STALE
  std::optional<double> theta;             // time factor of the value
  // BEYOND_DEPTH
  int required_depth = 0;                  // depth of the shortest derivation, 0 = none needed or none known
  // HYPOTHESIS_ONLY
  bool only_in_hypothesis_store = false;
};

/// One row of the breakdown: the 3.1 statistic for one code, the threshold it was compared with, and whether
/// the code applies.
struct ReasonComponent {
  ReasonCode code = ReasonCode::OutOfScope;
  bool measured = false;
  bool applies = false;
  double statistic = 0.0;  // OUT_OF_SCOPE: OOD score. UNPARSEABLE: 1 - P(parse). AMBIGUOUS_*: entropy in bits.
                           // UNKNOWN_ENTITY: best link P. NO_EVIDENCE: u of the slot (1 if empty).
                           // CONFLICT: DC. LOW_CERTAINTY: conf. STALE: theta. BEYOND_DEPTH: required depth.
                           // HYPOTHESIS_ONLY: 1 if the only candidate is a hypothesis, else 0.
  double threshold = 0.0;  // the cut it was compared with, in the statistic's own units (0 when it is a flag)
  double aux = 0.0;        // AMBIGUOUS_*: second-best / best ratio. CONFLICT: d of the slot. Else 0.
};

struct ReasonBreakdown {
  std::array<ReasonComponent, kReasonCodeCount> part{};
  EpistemicState slot_state = EpistemicState::Vacuous;  // state of the slot (Vacuous when empty)
  bool has_reason = false;
  ReasonCode first = ReasonCode::NoEvidence;            // valid only when has_reason

  const ReasonComponent& operator[](ReasonCode c) const { return part[static_cast<std::size_t>(c)]; }
  ReasonComponent& operator[](ReasonCode c) { return part[static_cast<std::size_t>(c)]; }
};

/// Whether a probability distribution has two close candidates, and its entropy.
struct AmbiguityStat {
  double entropy_bits = 0.0;
  double p1 = 0.0;  // best, after normalising to sum 1
  double p2 = 0.0;  // second best
  double ratio = 0.0;  // p2 / p1
  bool ambiguous = false;
};

inline AmbiguityStat ambiguity_of(const std::vector<double>& probs, const ReasonThresholds& t = {}) {
  AmbiguityStat a;
  double total = 0.0;
  for (double p : probs) total += std::max(0.0, p);
  if (!(total > 0.0)) return a;
  a.entropy_bits = entropy_bits(probs);
  for (double p : probs) {
    const double q = std::max(0.0, p) / total;
    if (q > a.p1) { a.p2 = a.p1; a.p1 = q; }
    else if (q > a.p2) { a.p2 = q; }
  }
  a.ratio = a.p1 > 0.0 ? a.p2 / a.p1 : 0.0;
  a.ambiguous = a.p2 >= t.ambiguity_min_second && a.ratio >= t.ambiguity_ratio;
  return a;
}

/// Fills every component of the breakdown. Notes on where the 3.1 table needed a decision:
///  - NO_EVIDENCE ("the slot is empty") does not apply when the only candidate is a hypothesis, nor when a
///    derivation exists but is deeper than the cap. Otherwise HYPOTHESIS_ONLY and BEYOND_DEPTH, which are
///    listed after it, could never be reported: their slot is always empty in the fact store.
///  - LOW_CERTAINTY and STALE need a stored value, so an empty slot never triggers them.
///  - A statistic that was not supplied leaves its code unmeasured and not applying.
inline ReasonBreakdown explain_unknown(const ReasonInputs& in, const ReasonThresholds& t = {}) {
  ReasonBreakdown bd;
  for (std::size_t i = 0; i < kReasonCodeCount; ++i) bd.part[i].code = static_cast<ReasonCode>(i);

  {  // OUT_OF_SCOPE
    ReasonComponent& c = bd[ReasonCode::OutOfScope];
    c.threshold = t.ood_score;
    c.measured = in.ood_score.has_value() || in.ood_flag || in.topic_outside_every_domain;
    c.statistic = in.ood_score ? *in.ood_score : ((in.ood_flag || in.topic_outside_every_domain) ? 1.0 : 0.0);
    c.applies = in.ood_flag || in.topic_outside_every_domain || (in.ood_score && *in.ood_score > t.ood_score);
  }
  {  // UNPARSEABLE
    ReasonComponent& c = bd[ReasonCode::Unparseable];
    c.threshold = 1.0 - t.parse_p;
    c.measured = in.p_parse.has_value();
    if (in.p_parse) {
      c.statistic = 1.0 - detail::ecs_clamp01(*in.p_parse);
      c.applies = *in.p_parse < t.parse_p;
    }
  }
  {  // AMBIGUOUS_QUESTION
    ReasonComponent& c = bd[ReasonCode::AmbiguousQuestion];
    c.threshold = t.ambiguity_ratio;
    c.measured = !in.relation_probs.empty();
    const AmbiguityStat a = ambiguity_of(in.relation_probs, t);
    c.statistic = a.entropy_bits;
    c.aux = a.ratio;
    c.applies = a.ambiguous;
  }
  {  // UNKNOWN_ENTITY
    ReasonComponent& c = bd[ReasonCode::UnknownEntity];
    c.threshold = t.link_p;
    c.measured = in.best_link_p.has_value();
    if (in.best_link_p) {
      c.statistic = *in.best_link_p;
      c.applies = *in.best_link_p < t.link_p;
    }
  }
  {  // AMBIGUOUS_ENTITY
    ReasonComponent& c = bd[ReasonCode::AmbiguousEntity];
    c.threshold = t.ambiguity_ratio;
    c.measured = !in.link_probs.empty();
    const AmbiguityStat a = ambiguity_of(in.link_probs, t);
    c.statistic = a.entropy_bits;
    c.aux = a.ratio;
    c.applies = a.ambiguous;
  }

  // The slot's state drives the next three codes.
  bd.slot_state = in.slot ? classify(*in.slot, in.rival, t.classify) : EpistemicState::Vacuous;
  const bool has_value = in.slot.has_value();
  {  // NO_EVIDENCE
    ReasonComponent& c = bd[ReasonCode::NoEvidence];
    c.measured = true;
    c.statistic = in.slot ? in.slot->u : 1.0;
    c.threshold = t.classify.ignorant_u;
    const bool empty_or_weak = !has_value || bd.slot_state == EpistemicState::Vacuous ||
                               bd.slot_state == EpistemicState::Ignorant;
    const bool something_else_exists = in.only_in_hypothesis_store || in.required_depth > t.depth_cap;
    c.applies = empty_or_weak && !something_else_exists;
  }
  {  // CONFLICT
    ReasonComponent& c = bd[ReasonCode::Conflict];
    c.measured = has_value;
    c.threshold = t.classify.contested_dc;
    if (in.slot) {
      c.statistic = in.rival ? degree_of_conflict(*in.slot, *in.rival) : 0.0;
      c.aux = in.slot->d;
    }
    c.applies = has_value && bd.slot_state == EpistemicState::Contested;
  }
  {  // LOW_CERTAINTY
    ReasonComponent& c = bd[ReasonCode::LowCertainty];
    c.measured = has_value && in.answer_conf.has_value() && in.conf_threshold.has_value();
    if (c.measured) {
      c.statistic = *in.answer_conf;
      c.threshold = *in.conf_threshold;
      c.applies = *in.answer_conf < *in.conf_threshold;
    }
  }
  {  // STALE
    ReasonComponent& c = bd[ReasonCode::Stale];
    c.threshold = t.stale_theta;
    c.measured = has_value && in.theta.has_value();
    if (c.measured) {
      c.statistic = *in.theta;
      c.applies = *in.theta < t.stale_theta;
    }
  }
  {  // BEYOND_DEPTH
    ReasonComponent& c = bd[ReasonCode::BeyondDepth];
    c.measured = in.required_depth > 0;
    c.statistic = static_cast<double>(in.required_depth);
    c.threshold = static_cast<double>(t.depth_cap);
    c.applies = in.required_depth > t.depth_cap;
  }
  {  // HYPOTHESIS_ONLY
    ReasonComponent& c = bd[ReasonCode::HypothesisOnly];
    c.measured = true;
    c.statistic = in.only_in_hypothesis_store ? 1.0 : 0.0;
    c.applies = in.only_in_hypothesis_store;
  }

  for (const ReasonComponent& c : bd.part) {
    if (c.applies) {
      bd.has_reason = true;
      bd.first = c.code;
      break;
    }
  }
  return bd;
}

/// The first code, in the 3.1 priority order, that applies; empty when none does (the question is answerable).
inline std::optional<ReasonCode> first_reason(const ReasonBreakdown& bd) {
  for (const ReasonComponent& c : bd.part)
    if (c.applies) return c.code;
  return std::nullopt;
}

/// Convenience: breakdown and pick in one call.
inline std::optional<ReasonCode> pick_reason(const ReasonInputs& in, const ReasonThresholds& t = {}) {
  return first_reason(explain_unknown(in, t));
}

// ---------------------------------------------------------------------------
// 3.3 Estimating the unseen
// ---------------------------------------------------------------------------

/// Counts of how often each distinct item was seen. Zero entries are ignored.
struct AbundanceStats {
  std::size_t s_obs = 0;  // distinct items seen
  std::size_t total = 0;  // N: total observations
  std::size_t n1 = 0;     // items seen exactly once
  std::size_t n2 = 0;     // items seen exactly twice
};

inline AbundanceStats abundance_stats(const std::vector<std::size_t>& counts) {
  AbundanceStats a;
  for (std::size_t c : counts) {
    if (c == 0) continue;
    ++a.s_obs;
    a.total += c;
    if (c == 1) ++a.n1;
    else if (c == 2) ++a.n2;
  }
  return a;
}

/// Good-Turing missing mass n1 / N: the probability that the next item read is new. With no observations at
/// all (N = 0) everything is unseen, so the result is 1.
inline double good_turing_missing_mass(std::size_t n1, std::size_t total) {
  if (total == 0) return 1.0;
  return static_cast<double>(n1) / static_cast<double>(total);
}

inline double good_turing_missing_mass(const std::vector<std::size_t>& counts) {
  const AbundanceStats a = abundance_stats(counts);
  return good_turing_missing_mass(a.n1, a.total);
}

/// Chao1 richness: S_obs + n1^2 / (2 n2); when n2 = 0, S_obs + n1 (n1 - 1) / 2.
inline double chao1(std::size_t s_obs, std::size_t n1, std::size_t n2) {
  const double s = static_cast<double>(s_obs);
  const double f1 = static_cast<double>(n1);
  const double f2 = static_cast<double>(n2);
  if (n2 == 0) return s + f1 * (f1 - 1.0) / 2.0;
  return s + f1 * f1 / (2.0 * f2);
}

struct Chao1Result {
  AbundanceStats stats;
  double estimate = 0.0;  // S-hat, the estimated total number of distinct items
  double unseen = 0.0;    // S-hat - S_obs
  double coverage = 1.0;  // S_obs / S-hat (1 when S-hat = 0)
};

inline Chao1Result chao1_from_counts(const std::vector<std::size_t>& counts) {
  Chao1Result r;
  r.stats = abundance_stats(counts);
  r.estimate = chao1(r.stats.s_obs, r.stats.n1, r.stats.n2);
  r.unseen = r.estimate - static_cast<double>(r.stats.s_obs);
  r.coverage = r.estimate > 0.0 ? static_cast<double>(r.stats.s_obs) / r.estimate : 1.0;
  return r;
}

struct ChapmanResult {
  double estimate = 0.0;  // N-hat = (nA + 1)(nB + 1) / (m + 1) - 1
  double variance = 0.0;  // Seber's variance of the Chapman estimator
  double known = 0.0;     // nA + nB - m: facts found by at least one source
  double coverage = 0.0;  // known / N-hat, clamped to [0, 1]; a lower bound when the sources are correlated
};

/// Chapman capture-recapture. n_a and n_b are the facts found by two independent sources, m by both
/// (m is clamped to min(n_a, n_b)). It underestimates when the sources are correlated, so treat the estimate
/// as a lower bound and the coverage as an upper bound.
inline ChapmanResult chapman(double n_a, double n_b, double m) {
  n_a = std::max(0.0, n_a);
  n_b = std::max(0.0, n_b);
  m = std::max(0.0, std::min(m, std::min(n_a, n_b)));
  ChapmanResult r;
  r.estimate = (n_a + 1.0) * (n_b + 1.0) / (m + 1.0) - 1.0;
  r.variance = (n_a + 1.0) * (n_b + 1.0) * (n_a - m) * (n_b - m) / ((m + 1.0) * (m + 1.0) * (m + 2.0));
  r.known = n_a + n_b - m;
  r.coverage = r.estimate > 0.0 ? detail::ecs_clamp01(r.known / r.estimate) : 1.0;
  return r;
}

struct HeapsFit {
  double K = 0.0;
  double beta = 0.0;
  double r2 = 0.0;       // R^2 of the fit in log space
  std::size_t n = 0;     // points used (N > 0 and V > 0)
  bool ok = false;       // false when fewer than two usable points with distinct N
};

/// Heaps' law V = K N^beta, fitted by ordinary least squares on ln V = ln K + beta ln N. Points with N <= 0 or
/// V <= 0 are skipped.
inline HeapsFit fit_heaps(const std::vector<double>& tokens, const std::vector<double>& vocab) {
  HeapsFit f;
  const std::size_t m = std::min(tokens.size(), vocab.size());
  double sx = 0.0, sy = 0.0;
  std::vector<std::pair<double, double>> pts;
  for (std::size_t i = 0; i < m; ++i) {
    if (!(tokens[i] > 0.0) || !(vocab[i] > 0.0)) continue;
    pts.emplace_back(std::log(tokens[i]), std::log(vocab[i]));
    sx += pts.back().first;
    sy += pts.back().second;
  }
  f.n = pts.size();
  if (pts.size() < 2) return f;
  const double n = static_cast<double>(pts.size());
  const double mx = sx / n;
  const double my = sy / n;
  double sxx = 0.0, sxy = 0.0, syy = 0.0;
  for (const auto& p : pts) {
    sxx += (p.first - mx) * (p.first - mx);
    sxy += (p.first - mx) * (p.second - my);
    syy += (p.second - my) * (p.second - my);
  }
  if (!(sxx > 0.0)) return f;
  f.beta = sxy / sxx;
  f.K = std::exp(my - f.beta * mx);
  f.r2 = syy > 0.0 ? (sxy * sxy) / (sxx * syy) : 1.0;
  f.ok = true;
  return f;
}

/// Expected vocabulary after reading `tokens` tokens.
inline double heaps_vocab(const HeapsFit& f, double tokens) {
  return tokens > 0.0 ? f.K * std::pow(tokens, f.beta) : 0.0;
}

/// Expected new words from reading `step` more tokens (default one million) after `tokens_now`.
inline double heaps_new_words(const HeapsFit& f, double tokens_now, double step = 1e6) {
  return heaps_vocab(f, tokens_now + step) - heaps_vocab(f, tokens_now);
}

// ---------------------------------------------------------------------------
// 10 Cards
// ---------------------------------------------------------------------------

namespace detail {

inline std::string epi_fmt(const char* fmt, double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, fmt, v);
  return buf;
}

inline std::size_t utf8_width(const std::string& s) {
  std::size_t n = 0;
  for (unsigned char c : s)
    if ((c & 0xC0) != 0x80) ++n;
  return n;
}

/// `text`, then spaces up to display column `width`, then a single space at least.
inline std::string pad_for_tag(const std::string& text, std::size_t width = 33) {
  std::string s = text;
  const std::size_t w = utf8_width(text);
  s.append(w < width ? width - w : 1, ' ');
  return s;
}

inline std::string with_period(const std::string& s) {
  if (s.empty()) return s;
  const char last = s.back();
  return (last == '.' || last == '!' || last == '?') ? s : s + ".";
}

}  // namespace detail

/// Answer card. Fill the opinion fields by hand or with fill_card_opinion().
struct AnswerCard {
  std::string question;
  std::string answer;
  EpistemicState state = EpistemicState::Known;
  double p = 0.0;
  double p_lo = 0.0;
  double p_hi = 0.0;
  Grade grade = Grade::High;
  std::string because;                 // omitted when empty
  int independent_sources = 0;
  std::string contradictions = "none";
  double fragility = 0.0;
  std::string weakest_link;            // what it is; the line is omitted when empty
  double weakest_link_p = 1.0;
  std::string would_change_mind;       // omitted when empty
};

inline void fill_card_opinion(AnswerCard& card, const Opinion& o, const std::optional<Opinion>& rival = std::nullopt,
                              const ClassifyConfig& cfg = {}) {
  card.state = classify(o, rival, cfg);
  card.p = o.P;
  card.p_lo = o.lo;
  card.p_hi = o.hi;
  card.grade = o.grade;
}

inline std::string format_answer_card(const AnswerCard& c) {
  using namespace epi_text;
  std::string s = "Q: " + c.question + "\n";
  s += detail::pad_for_tag(c.answer) + "[" + state_name(c.state) + " " + kMidDot + " P " + detail::epi_fmt("%.2f", c.p) + " (" +
       detail::epi_fmt("%.2f", c.p_lo) + kEnDash + detail::epi_fmt("%.2f", c.p_hi) + ") " + kMidDot + " " + grade_name(c.grade) + "]\n";
  if (!c.because.empty()) s += "Because: " + c.because + "\n";
  s += "Independent sources: " + std::to_string(c.independent_sources) + " " + kMidDot + " Contradictions: " + c.contradictions + " " +
       kMidDot + " Fragility: " + detail::epi_fmt("%.1f", c.fragility) + "\n";
  if (!c.weakest_link.empty()) s += "Weakest link: " + c.weakest_link + " (" + detail::epi_fmt("%.2f", c.weakest_link_p) + ")\n";
  if (!c.would_change_mind.empty()) s += "Would change my mind: " + c.would_change_mind + "\n";
  return s;
}

/// "I don't know" card.
struct UnknownCard {
  std::string question;
  ReasonCode reason = ReasonCode::NoEvidence;
  std::string why;                       // omitted when empty
  std::string what_i_can_say;            // omitted when empty
  std::string what_would_let_me_answer;  // omitted when empty
  std::string gap_id;                    // e.g. "G-0412"; the line is omitted when empty
  int demand_delta = 1;
  std::string breakdown;                 // optional lines from describe_breakdown(); omitted when empty
};

inline std::string format_unknown_card(const UnknownCard& c) {
  std::string s = "Q: " + c.question + "\n";
  s += detail::pad_for_tag("I don't know.") + "[" + reason_name(c.reason) + "]\n";
  if (!c.why.empty()) s += "Why: " + c.why + "\n";
  if (!c.what_i_can_say.empty()) s += "What I can say: " + c.what_i_can_say + "\n";
  if (!c.what_would_let_me_answer.empty()) s += "What would let me answer: " + c.what_would_let_me_answer + "\n";
  if (!c.breakdown.empty()) s += c.breakdown + (c.breakdown.back() == '\n' ? "" : "\n");
  if (!c.gap_id.empty())
    s += "Logged as gap " + c.gap_id + " (demand " + (c.demand_delta >= 0 ? "+" : "") + std::to_string(c.demand_delta) + ").\n";
  return s;
}

/// One line per code that was measured, in priority order, marking those that apply.
inline std::string describe_breakdown(const ReasonBreakdown& bd) {
  std::string s = "Breakdown (first applicable code wins):\n";
  for (const ReasonComponent& c : bd.part) {
    if (!c.measured && !c.applies) continue;
    s += std::string("  ") + (c.applies ? "*" : " ") + " " + reason_name(c.code) + ": statistic " + detail::epi_fmt("%.3f", c.statistic) +
         ", threshold " + detail::epi_fmt("%.3f", c.threshold) + "\n";
  }
  return s;
}

/// Hypothesis card (discovery mode only). Always starts with the SPECULATION label.
struct HypothesisCard {
  std::string statement;
  std::string from;             // generator and the rule or analogy it applied; may span lines
  double plausibility = 0.0;
  double novelty_bits = 0.0;
  std::string impact;           // e.g. "fills a schema gap (demand 0)"
  std::string tests_planned;
  std::string status = "queued";
};

inline std::string hypothesis_label() { return std::string("SPECULATION ") + epi_text::kEmDash + " not knowledge."; }

inline std::string format_hypothesis_card(const HypothesisCard& c) {
  using namespace epi_text;
  std::string s = hypothesis_label() + "\n";
  s += c.statement + "\n";
  if (!c.from.empty()) s += "From: " + c.from + "\n";
  s += "Plausibility " + detail::epi_fmt("%.2f", c.plausibility) + " " + kMidDot + " Novelty " + detail::epi_fmt("%.1f", c.novelty_bits) +
       " bits";
  if (!c.impact.empty()) s += std::string(" ") + kMidDot + " Impact: " + c.impact;
  s += "\n";
  if (!c.tests_planned.empty()) s += "Tests planned: " + c.tests_planned + "\n";
  s += "Status: " + c.status + "\n";
  return s;
}

/// Hypothetical (what-if) card. Always starts with the HYPOTHETICAL label and ends with "Nothing here is stored."
struct HypotheticalCard {
  std::string assumption;
  int depth = kMaxDerivationDepth;
  int dropped_count = 0;                           // derived facts that no longer hold
  std::vector<std::string> dropped_examples;       // a few of them
  std::vector<std::string> contradictions_with_known;  // each: the Known fact the assumption collides with
};

inline std::string hypothetical_label(const std::string& assumption) {
  return std::string("HYPOTHETICAL ") + epi_text::kEmDash + " assuming: " + detail::with_period(assumption);
}

inline std::string format_hypothetical_card(const HypotheticalCard& c) {
  using namespace epi_text;
  std::string s = hypothetical_label(c.assumption) + "\n";
  s += std::string("Consequences (depth ") + kLeq + " " + std::to_string(c.depth) + "): ";
  if (c.dropped_count <= 0) {
    s += "no derived facts drop";
  } else {
    s += std::to_string(c.dropped_count) + (c.dropped_count == 1 ? " derived fact drops" : " derived facts drop");
    if (!c.dropped_examples.empty()) {
      s += " (";
      for (std::size_t i = 0; i < c.dropped_examples.size(); ++i) s += (i ? ", " : "") + c.dropped_examples[i];
      if (static_cast<std::size_t>(c.dropped_count) > c.dropped_examples.size()) s += std::string(" ") + kEllipsis;
      s += ")";
    }
  }
  s += "; ";
  const std::size_t k = c.contradictions_with_known.size();
  if (k == 0) {
    s += "no contradiction with Known facts.\n";
  } else {
    s += std::to_string(k) + (k == 1 ? " contradiction" : " contradictions") + " with a Known fact: ";
    for (std::size_t i = 0; i < k; ++i) s += (i ? "; " : "") + c.contradictions_with_known[i];
    s += std::string(" ") + kArrow + " the assumption conflicts with Known facts.\n";
  }
  s += "Nothing here is stored.\n";
  return s;
}

}  // namespace cypha::eval
