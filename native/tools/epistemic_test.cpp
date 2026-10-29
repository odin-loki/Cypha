// Known-answer tests for cypha/eval/epistemic.hpp (docs/DISCOVERY.md sections 2.1-2.3, 3.1, 3.3, 10), hypothesis.hpp
// (S5: 4, 6.3, 7.2-7.5; invariants I1, I4, I7, I8; S7: 7.1-7.3, I5, I6; S7b: three-way likelihoods, the batch step
// among the supported), gap_map.hpp (S3: 3.2, 3.5), gen_links.hpp (S8b: G6, G7), gen_rules.hpp (S8a: G2, G14), sources.hpp
// (S8c: evidence sources), worlds.hpp (S9) and curiosity.hpp (S10); S7c's batch rules.
// Returns nonzero if any check fails.
#include "cypha/eval/abc_closed.hpp"
#include "cypha/eval/curiosity.hpp"
#include "cypha/eval/epistemic.hpp"
#include "cypha/eval/gap_map.hpp"
#include "cypha/eval/gen_concepts.hpp"
#include "cypha/eval/gen_deep.hpp"
#include "cypha/eval/gen_laws.hpp"
#include "cypha/eval/gen_analogy.hpp"
#include "cypha/eval/gen_links.hpp"
#include "cypha/eval/gen_rules.hpp"
#include "cypha/eval/hypothesis.hpp"
#include "cypha/eval/sources.hpp"
#include "cypha/eval/unknown_signals.hpp"
#include "cypha/eval/worlds.hpp"
#include "cypha/eval/wd_grader.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace cypha::eval;

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

#define CHECK_STR(x, y)                                                          \
  do {                                                                           \
    ++g_checks;                                                                  \
    const std::string _x = (x), _y = (y);                                        \
    if (_x != _y) {                                                              \
      ++g_failed;                                                                \
      std::printf("FAIL %s:%d  %s\n--- got ---\n%s\n--- expected ---\n%s\n", __FILE__, __LINE__, #x, _x.c_str(), _y.c_str()); \
    }                                                                            \
  } while (0)

static bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

// ---------------------------------------------------------------------------
// 2.3 degree of conflict
// ---------------------------------------------------------------------------
static void test_degree_of_conflict() {
  // Two confident, disagreeing opinions about one proposition.
  // A: r=18, s=0, W=2 -> b .9, d 0, u .1, P .95.  B: r=0, s=8 -> b 0, d .8, u .2, P .1.
  const Opinion a = opinion_from_counts(18, 0);
  const Opinion b = opinion_from_counts(0, 8);
  CHECK_NEAR(a.P, 0.95, 1e-12);
  CHECK_NEAR(a.u, 0.1, 1e-12);
  CHECK_NEAR(b.P, 0.1, 1e-12);
  CHECK_NEAR(b.u, 0.2, 1e-12);
  // |0.95 - 0.1| * (1 - 0.1)(1 - 0.2) = 0.85 * 0.9 * 0.8
  CHECK_NEAR(degree_of_conflict(a, b), 0.85 * 0.9 * 0.8, 1e-12);
  CHECK_NEAR(degree_of_conflict(b, a), 0.85 * 0.9 * 0.8, 1e-12);  // symmetric
  CHECK_NEAR(degree_of_conflict(0.95, 0.1, 0.1, 0.2), 0.612, 1e-12);
  // Agreement is no conflict; a vacuous opinion (u = 1) never conflicts.
  CHECK_NEAR(degree_of_conflict(a, a), 0.0, 1e-15);
  CHECK_NEAR(degree_of_conflict(a, opinion_from_counts(0, 0)), 0.0, 1e-15);
  // Two confident opinions that disagree completely: P 1 vs P 0, u 0 -> DC 1.
  CHECK_NEAR(degree_of_conflict(1.0, 0.0, 0.0, 0.0), 1.0, 1e-15);
  // Disagreement without confidence stays low: P .9 vs .1 but u .9 each -> 0.8 * 0.1 * 0.1.
  CHECK_NEAR(degree_of_conflict(0.9, 0.9, 0.1, 0.9), 0.008, 1e-12);
}

// ---------------------------------------------------------------------------
// 2.1 classify
// ---------------------------------------------------------------------------
static void test_classify() {
  // Known: Grade High. Beta(41, 1) has 5% quantile 0.05^(1/41) = 0.9295 >= 0.90.
  const Opinion known = opinion_from_counts(40, 0);
  CHECK(known.grade == Grade::High);
  CHECK(classify(known) == EpistemicState::Known);
  // Believed: Moderate (r 18: P .95, u .1, but lower bound .854 < .90) and Low (r 1: P .667).
  const Opinion moderate = opinion_from_counts(18, 0);
  CHECK(moderate.grade == Grade::Moderate);
  CHECK(classify(moderate) == EpistemicState::Believed);
  const Opinion low = opinion_from_counts(1, 0);
  CHECK(low.grade == Grade::Low);
  CHECK(classify(low) == EpistemicState::Believed);
  // Vacuous: no evidence at all.
  CHECK(classify(opinion_from_counts(0, 0)) == EpistemicState::Vacuous);
  // Ignorant: u >= 0.8. r = 0.2 gives u = 2 / 2.2 = 0.909; r = 1 gives u = 0.667 (Believed).
  CHECK(classify(opinion_from_counts(0.2, 0)) == EpistemicState::Ignorant);
  CHECK(classify(opinion_from_counts(0, 0.2)) == EpistemicState::Ignorant);
  CHECK(classify(opinion_from_counts(1, 0)) == EpistemicState::Believed);
  // u exactly 0.8: r = 0.5, s = 0, W = 2 -> u = 2 / 2.5.
  CHECK(classify(opinion_from_counts(0.5, 0)) == EpistemicState::Ignorant);
  // Contested by d: d = s / (r + s + 2). s = 2 gives d = 0.5 exactly (>= 0.5 counts); s = 1 gives d = 0.333.
  CHECK(classify(opinion_from_counts(0, 2)) == EpistemicState::Contested);
  CHECK(classify(opinion_from_counts(0, 3)) == EpistemicState::Contested);
  CHECK(classify(opinion_from_counts(0, 1)) != EpistemicState::Contested);
  // Contested by DC with a rival value: DC >= 0.3.
  //   moderate (P .95, u .1) against (0, 4): P .167, u .333 -> DC .783 * .9 * .667 = .47 -> Contested
  //   moderate against (1, 3): P .333, u .333 -> DC .617 * .9 * .667 = .37 -> Contested
  //   moderate against (2, 2): P .5, u .333 -> DC .45 * .9 * .667 = .27 -> not contested
  CHECK_NEAR(degree_of_conflict(moderate, opinion_from_counts(0, 4)), (0.95 - 1.0 / 6.0) * 0.9 * (2.0 / 3.0), 1e-12);
  CHECK(classify(moderate, opinion_from_counts(0, 4)) == EpistemicState::Contested);
  CHECK(classify(moderate, opinion_from_counts(1, 3)) == EpistemicState::Contested);
  CHECK_NEAR(degree_of_conflict(moderate, opinion_from_counts(2, 2)), 0.45 * 0.9 * (2.0 / 3.0), 1e-12);
  CHECK(classify(moderate, opinion_from_counts(2, 2)) == EpistemicState::Believed);
  // A Known item with a confident rival is Contested, not Known.
  CHECK(classify(known, opinion_from_counts(0, 40)) == EpistemicState::Contested);
  // A rival that agrees changes nothing.
  CHECK(classify(known, opinion_from_counts(40, 0)) == EpistemicState::Known);
  // A hand-set DC threshold is honoured.
  ClassifyConfig strict;
  strict.contested_dc = 0.2;
  CHECK(classify(moderate, opinion_from_counts(2, 2), strict) == EpistemicState::Contested);
  // Grade Very low with u < 0.8 and d < 0.5 has no row in the table: r 1, s 2 -> b .2, d .4, u .4, P .4.
  const Opinion vlow = opinion_from_counts(1, 2);
  CHECK(vlow.grade == Grade::VeryLow);
  CHECK(classify(vlow) == EpistemicState::Ignorant);  // default policy: a gap, never an answer
  ClassifyConfig lenient;
  lenient.very_low_state = EpistemicState::Believed;
  CHECK(classify(vlow, std::nullopt, lenient) == EpistemicState::Believed);
  // Hypothesis store states.
  CHECK(hypothesis_state(false) == EpistemicState::Hypothesis);
  CHECK(hypothesis_state(true) == EpistemicState::Refuted);
  // Names.
  CHECK_STR(state_name(EpistemicState::Known), "Known");
  CHECK_STR(state_name(EpistemicState::Refuted), "Refuted");
  // "May be used for".
  CHECK(state_use(EpistemicState::Known, Grade::High).answer && state_use(EpistemicState::Known, Grade::High).premise);
  CHECK(state_use(EpistemicState::Believed, Grade::Moderate).premise);
  CHECK(!state_use(EpistemicState::Believed, Grade::Low).premise);
  CHECK(state_use(EpistemicState::Believed, Grade::Low).answer && state_use(EpistemicState::Believed, Grade::Low).answer_states_grade);
  CHECK(!state_use(EpistemicState::Contested, Grade::Low).answer);
  CHECK(!state_use(EpistemicState::Ignorant, Grade::Low).answer);
  CHECK(!state_use(EpistemicState::Vacuous, Grade::Low).answer);
  CHECK(state_use(EpistemicState::Hypothesis, Grade::Low).discovery_only && !state_use(EpistemicState::Hypothesis, Grade::Low).answer);
  CHECK(state_use(EpistemicState::Refuted, Grade::Low).blocks_regeneration);
}

// ---------------------------------------------------------------------------
// 2.2 lack of evidence versus mixed evidence
// ---------------------------------------------------------------------------
static void test_split_uncertainty() {
  // Lack of evidence: r = 0.5 -> u = 0.8.
  const UncertaintySplit lack = split_uncertainty(opinion_from_counts(0.5, 0));
  CHECK(lack.kind == UncertaintyKind::LackOfEvidence);
  CHECK_NEAR(lack.lack_of_evidence, 0.8, 1e-12);
  // u = 0.5 exactly (r = s = 1, W = 2) is lack of evidence (>= 0.5).
  CHECK(split_uncertainty(opinion_from_counts(1, 1)).kind == UncertaintyKind::LackOfEvidence);
  // Mixed evidence: r = s = 20 -> b = d = 0.476, u = 0.048, P = 0.5.
  const UncertaintySplit mixed = split_uncertainty(opinion_from_counts(20, 20));
  CHECK(mixed.kind == UncertaintyKind::MixedEvidence);
  CHECK_NEAR(mixed.mixed_evidence, 2.0 * 20.0 / 42.0, 1e-12);
  CHECK(mixed.lack_of_evidence < 0.05);
  // One-sided: lots of evidence for.
  CHECK(split_uncertainty(opinion_from_counts(40, 0)).kind == UncertaintyKind::OneSided);
  CHECK(split_uncertainty(opinion_from_counts(0, 40)).kind == UncertaintyKind::OneSided);
  // Lots of evidence, lopsided 30:10 -> P .71 is outside the 0.5 +- 0.2 band: one-sided.
  CHECK(split_uncertainty(opinion_from_counts(30, 10)).kind == UncertaintyKind::OneSided);
  // In-between u (r 2, s 1: u = 0.4): unclear.
  CHECK(split_uncertainty(opinion_from_counts(2, 1)).kind == UncertaintyKind::Unclear);
  // The action attached to each kind.
  CHECK(std::string(uncertainty_action(UncertaintyKind::LackOfEvidence)).find("P1") != std::string::npos);
  CHECK(std::string(uncertainty_action(UncertaintyKind::MixedEvidence)).find("P4") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 3.1 reason codes
// ---------------------------------------------------------------------------
static ReasonInputs good_inputs() {
  ReasonInputs in;
  in.ood_score = 0.1;
  in.p_parse = 0.9;
  in.relation_probs = {0.9, 0.05, 0.05};
  in.best_link_p = 0.95;
  in.link_probs = {0.95, 0.05};
  in.slot = opinion_from_counts(40, 0);  // Known
  in.answer_conf = 0.9;
  in.conf_threshold = 0.8;
  in.theta = 0.9;
  in.required_depth = 1;
  return in;
}

static void test_reason_codes() {
  // Eleven codes, in the 3.1 order.
  static const char* const names[kReasonCodeCount] = {
      "OUT_OF_SCOPE", "UNPARSEABLE",   "AMBIGUOUS_QUESTION", "UNKNOWN_ENTITY", "AMBIGUOUS_ENTITY", "NO_EVIDENCE",
      "CONFLICT",     "LOW_CERTAINTY", "STALE",              "BEYOND_DEPTH",   "HYPOTHESIS_ONLY"};
  CHECK(kReasonCodeCount == 11);
  for (std::size_t i = 0; i < kReasonCodeCount; ++i) CHECK_STR(reason_name(static_cast<ReasonCode>(i)), names[i]);
  CHECK(static_cast<int>(ReasonCode::OutOfScope) == 0);
  CHECK(static_cast<int>(ReasonCode::HypothesisOnly) == 10);
  CHECK(static_cast<int>(ReasonCode::AmbiguousQuestion) < static_cast<int>(ReasonCode::NoEvidence));
  CHECK(static_cast<int>(ReasonCode::NoEvidence) < static_cast<int>(ReasonCode::Conflict));

  // Answerable: nothing applies.
  CHECK(!pick_reason(good_inputs()).has_value());

  // Cumulative triggers, lowest priority first. Each new trigger must take over as the first code.
  ReasonInputs in = good_inputs();
  in.only_in_hypothesis_store = true;
  CHECK(pick_reason(in) == ReasonCode::HypothesisOnly);
  in.required_depth = 5;  // > cap of 3
  CHECK(pick_reason(in) == ReasonCode::BeyondDepth);
  in.theta = 0.3;  // < 0.5
  CHECK(pick_reason(in) == ReasonCode::Stale);
  in.answer_conf = 0.5;  // < 0.8
  CHECK(pick_reason(in) == ReasonCode::LowCertainty);
  in.rival = opinion_from_counts(0, 40);  // confident rival against a Known value
  CHECK(pick_reason(in) == ReasonCode::Conflict);
  in.link_probs = {0.5, 0.45};  // two concepts fit the mention
  CHECK(pick_reason(in) == ReasonCode::AmbiguousEntity);
  in.best_link_p = 0.2;
  CHECK(pick_reason(in) == ReasonCode::UnknownEntity);
  in.relation_probs = {0.45, 0.4, 0.15};  // two relations fit
  CHECK(pick_reason(in) == ReasonCode::AmbiguousQuestion);
  in.p_parse = 0.3;  // < 0.5
  CHECK(pick_reason(in) == ReasonCode::Unparseable);
  in.ood_score = 0.9;
  CHECK(pick_reason(in) == ReasonCode::OutOfScope);

  // The breakdown holds every component's statistic, whether or not it was the one picked.
  const ReasonBreakdown bd = explain_unknown(in);
  CHECK(bd.has_reason && bd.first == ReasonCode::OutOfScope);
  CHECK_NEAR(bd[ReasonCode::OutOfScope].statistic, 0.9, 1e-12);
  CHECK_NEAR(bd[ReasonCode::Unparseable].statistic, 0.7, 1e-12);      // 1 - P(parse)
  CHECK_NEAR(bd[ReasonCode::AmbiguousQuestion].statistic, entropy_bits({0.45, 0.4, 0.15}), 1e-12);
  CHECK_NEAR(bd[ReasonCode::AmbiguousQuestion].aux, 0.4 / 0.45, 1e-12);
  CHECK_NEAR(bd[ReasonCode::UnknownEntity].statistic, 0.2, 1e-12);
  CHECK_NEAR(bd[ReasonCode::AmbiguousEntity].statistic, entropy_bits({0.5, 0.45}), 1e-12);
  CHECK_NEAR(bd[ReasonCode::Conflict].statistic, degree_of_conflict(opinion_from_counts(40, 0), opinion_from_counts(0, 40)), 1e-12);
  CHECK_NEAR(bd[ReasonCode::LowCertainty].statistic, 0.5, 1e-12);
  CHECK_NEAR(bd[ReasonCode::LowCertainty].threshold, 0.8, 1e-12);
  CHECK_NEAR(bd[ReasonCode::Stale].statistic, 0.3, 1e-12);
  CHECK_NEAR(bd[ReasonCode::BeyondDepth].statistic, 5.0, 1e-12);
  CHECK_NEAR(bd[ReasonCode::BeyondDepth].threshold, 3.0, 1e-12);
  CHECK(bd[ReasonCode::HypothesisOnly].applies);
  CHECK(!bd[ReasonCode::NoEvidence].applies);  // a value exists
  CHECK(bd.slot_state == EpistemicState::Contested);
  CHECK(first_reason(bd) == ReasonCode::OutOfScope);

  // NO_EVIDENCE: empty slot, nothing else going on.
  ReasonInputs empty;
  CHECK(pick_reason(empty) == ReasonCode::NoEvidence);
  CHECK_NEAR(explain_unknown(empty)[ReasonCode::NoEvidence].statistic, 1.0, 1e-12);
  CHECK(!explain_unknown(empty)[ReasonCode::Unparseable].measured);  // not supplied, so it never applies
  // ... or a slot that is Ignorant (u >= 0.8) or Vacuous.
  ReasonInputs weak = good_inputs();
  weak.slot = opinion_from_counts(0.2, 0);
  CHECK(pick_reason(weak) == ReasonCode::NoEvidence);
  weak.slot = opinion_from_counts(0, 0);
  CHECK(pick_reason(weak) == ReasonCode::NoEvidence);
  // A thin but not ignorant slot (u .667) is not NO_EVIDENCE; with a low confidence it is LOW_CERTAINTY.
  ReasonInputs thin = good_inputs();
  thin.slot = opinion_from_counts(1, 0);
  thin.answer_conf = 0.4;
  CHECK(pick_reason(thin) == ReasonCode::LowCertainty);
  // An empty slot whose only candidate is a hypothesis is HYPOTHESIS_ONLY, not NO_EVIDENCE.
  ReasonInputs hyp;
  hyp.only_in_hypothesis_store = true;
  CHECK(pick_reason(hyp) == ReasonCode::HypothesisOnly);
  // An empty slot reachable only by a derivation deeper than the cap is BEYOND_DEPTH.
  ReasonInputs deep;
  deep.required_depth = 4;
  CHECK(pick_reason(deep) == ReasonCode::BeyondDepth);
  deep.required_depth = 3;  // at the cap is allowed; with an empty slot there is simply no evidence
  CHECK(pick_reason(deep) == ReasonCode::NoEvidence);
  // Contested by d alone (no rival): CONFLICT.
  ReasonInputs contested = good_inputs();
  contested.slot = opinion_from_counts(0, 3);
  CHECK(pick_reason(contested) == ReasonCode::Conflict);
  CHECK_NEAR(explain_unknown(contested)[ReasonCode::Conflict].aux, 0.6, 1e-12);  // d
  // OOD flag and "topic outside every domain" each trigger OUT_OF_SCOPE.
  ReasonInputs flag = good_inputs();
  flag.ood_flag = true;
  CHECK(pick_reason(flag) == ReasonCode::OutOfScope);
  ReasonInputs topic = good_inputs();
  topic.topic_outside_every_domain = true;
  CHECK(pick_reason(topic) == ReasonCode::OutOfScope);
  // P(parse) exactly 0.5 still parses ("P >= 0.5"); just under does not.
  ReasonInputs parse = good_inputs();
  parse.p_parse = 0.5;
  CHECK(!pick_reason(parse).has_value());
  parse.p_parse = 0.49;
  CHECK(pick_reason(parse) == ReasonCode::Unparseable);

  // Ambiguity statistic.
  const AmbiguityStat half = ambiguity_of({0.5, 0.5});
  CHECK_NEAR(half.entropy_bits, 1.0, 1e-12);
  CHECK(half.ambiguous);
  CHECK(!ambiguity_of({0.9, 0.1}).ambiguous);
  CHECK(ambiguity_of({0.4, 0.3, 0.3}).ambiguous);
  CHECK(!ambiguity_of(std::vector<double>(100, 1.0)).ambiguous);  // 100 equal tiny candidates: no real second
  CHECK_NEAR(entropy_bits(std::vector<double>(8, 1.0)), 3.0, 1e-12);
  CHECK_NEAR(entropy_bits({1.0}), 0.0, 1e-15);
  CHECK_NEAR(entropy_bits({}), 0.0, 1e-15);
  // Breakdown text.
  CHECK(describe_breakdown(bd).find("* OUT_OF_SCOPE") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 3.3 unseen estimators
// ---------------------------------------------------------------------------
static void test_estimators() {
  // Frequency list: 10 distinct items seen {1,1,1,1,2,2,3,4,5,7} times. S_obs 10, N 27, n1 4, n2 2.
  const std::vector<std::size_t> counts = {1, 1, 1, 1, 2, 2, 3, 4, 5, 7};
  const AbundanceStats st = abundance_stats(counts);
  CHECK(st.s_obs == 10 && st.total == 27 && st.n1 == 4 && st.n2 == 2);
  CHECK_NEAR(good_turing_missing_mass(counts), 4.0 / 27.0, 1e-15);
  CHECK_NEAR(good_turing_missing_mass(4, 27), 4.0 / 27.0, 1e-15);
  // Chao1 = 10 + 4^2 / (2 * 2) = 14.
  const Chao1Result c = chao1_from_counts(counts);
  CHECK_NEAR(c.estimate, 14.0, 1e-12);
  CHECK_NEAR(c.unseen, 4.0, 1e-12);
  CHECK_NEAR(c.coverage, 10.0 / 14.0, 1e-12);
  CHECK_NEAR(chao1(10, 4, 2), 14.0, 1e-12);
  // n2 = 0: S_obs + n1 (n1 - 1) / 2. {1,1,1,3,4,6}: S_obs 6, n1 3 -> 6 + 3 = 9.
  const Chao1Result c0 = chao1_from_counts({1, 1, 1, 3, 4, 6});
  CHECK(c0.stats.n2 == 0);
  CHECK_NEAR(c0.estimate, 9.0, 1e-12);
  CHECK_NEAR(good_turing_missing_mass({1, 1, 1, 3, 4, 6}), 3.0 / 16.0, 1e-15);
  // n1 = 1 and n2 = 0: S_obs + 0. No singletons: S_obs, coverage 1.
  CHECK_NEAR(chao1(5, 1, 0), 5.0, 1e-12);
  CHECK_NEAR(chao1_from_counts({2, 3, 4}).estimate, 3.0, 1e-12);
  CHECK_NEAR(chao1_from_counts({2, 3, 4}).coverage, 1.0, 1e-15);
  // Zero entries are ignored; an empty list has missing mass 1 by convention.
  CHECK_NEAR(good_turing_missing_mass({0, 1, 0, 1, 2}), 2.0 / 4.0, 1e-15);
  CHECK_NEAR(good_turing_missing_mass(std::vector<std::size_t>{}), 1.0, 1e-15);
  // All singletons: every next item is new.
  CHECK_NEAR(good_turing_missing_mass({1, 1, 1, 1}), 1.0, 1e-15);

  // Chapman: n_A 50, n_B 40, m 10 -> (51)(41)/11 - 1 = 2091/11 - 1 = 189.0909...
  // (The brief quoted 185.45, which is 51 * 40 / 11: it drops the +1 on n_B. The 3.3 formula gives 189.09.)
  const ChapmanResult ch = chapman(50, 40, 10);
  CHECK_NEAR(ch.estimate, 2091.0 / 11.0 - 1.0, 1e-12);
  CHECK_NEAR(ch.estimate, 189.0909090909, 1e-9);
  CHECK_NEAR(ch.known, 80.0, 1e-12);
  CHECK_NEAR(ch.coverage, 80.0 / (2091.0 / 11.0 - 1.0), 1e-12);
  CHECK_NEAR(ch.variance, 51.0 * 41.0 * 40.0 * 30.0 / (121.0 * 12.0), 1e-9);
  CHECK(ch.estimate > ch.known && ch.estimate < 200.0);  // between the union and Lincoln-Petersen (50 * 40 / 10)
  // No overlap at all: (51)(41) - 1 = 2090.
  CHECK_NEAR(chapman(50, 40, 0).estimate, 2090.0, 1e-9);
  // Full overlap: both sources found the same 30 facts -> N-hat = 31 * 31 / 31 - 1 = 30, coverage 1.
  CHECK_NEAR(chapman(30, 30, 30).estimate, 30.0, 1e-12);
  CHECK_NEAR(chapman(30, 30, 30).coverage, 1.0, 1e-12);
  CHECK_NEAR(chapman(30, 30, 30).variance, 0.0, 1e-12);

  // Heaps: exact points from V = 31.6228 N^0.5 recover K and beta.
  {
    std::vector<double> n = {1e3, 1e4, 1e5, 1e6};
    std::vector<double> v;
    for (double x : n) v.push_back(10.0 * std::pow(x, 0.6));
    const HeapsFit f = fit_heaps(n, v);
    CHECK(f.ok && f.n == 4);
    CHECK_NEAR(f.K, 10.0, 1e-9);
    CHECK_NEAR(f.beta, 0.6, 1e-12);
    CHECK_NEAR(f.r2, 1.0, 1e-12);
    CHECK_NEAR(heaps_vocab(f, 1e7), 10.0 * std::pow(1e7, 0.6), 1e-3);
  }
  // Two points: (10, 100) and (1000, 1000) -> beta = ln 10 / ln 100 = 0.5, K = 100 / sqrt(10) = 31.6228.
  {
    const HeapsFit f = fit_heaps({10.0, 1000.0}, {100.0, 1000.0});
    CHECK(f.ok);
    CHECK_NEAR(f.beta, 0.5, 1e-12);
    CHECK_NEAR(f.K, 100.0 / std::sqrt(10.0), 1e-9);
  }
  // Least squares in log space with noise: three points on ln V = 1 + 0.5 ln N, one nudged. Compare with
  // the closed form on the log points.
  {
    const std::vector<double> n = {1.0, std::exp(1.0), std::exp(2.0)};
    const std::vector<double> v = {std::exp(1.0), std::exp(2.0), std::exp(2.0)};  // ln V = 1, 2, 2 at ln N = 0, 1, 2
    const HeapsFit f = fit_heaps(n, v);
    // slope = ((0-1)(1-5/3) + (1-1)(2-5/3) + (2-1)(2-5/3)) / ((0-1)^2 + 0 + 1) = (0.6667 + 0 + 0.3333) / 2 = 0.5
    CHECK_NEAR(f.beta, 0.5, 1e-12);
    CHECK_NEAR(std::log(f.K), 5.0 / 3.0 - 0.5, 1e-12);  // intercept = mean(y) - slope * mean(x)
    CHECK(f.r2 > 0.0 && f.r2 < 1.0);
  }
  // New words per million tokens: K = 1, beta = 0.5, from zero: sqrt(1e6) = 1000.
  {
    const HeapsFit f = fit_heaps({100.0, 10000.0}, {10.0, 100.0});
    CHECK_NEAR(f.K, 1.0, 1e-9);
    CHECK_NEAR(f.beta, 0.5, 1e-12);
    CHECK_NEAR(heaps_new_words(f, 0.0), 1000.0, 1e-6);
    CHECK_NEAR(heaps_new_words(f, 1e6), std::sqrt(2e6) - 1000.0, 1e-6);
  }
  // Degenerate input: fewer than two usable points, or all the same N.
  CHECK(!fit_heaps({100.0}, {10.0}).ok);
  CHECK(!fit_heaps({100.0, 100.0}, {10.0, 12.0}).ok);
  CHECK(!fit_heaps({0.0, -5.0, 100.0}, {10.0, 12.0, 20.0}).ok);  // only one positive point
}

// S4: clumped samples (pages as units). Three pages: P1 {a:3, b:1}, P2 {a:1, c:2}, P3 {d:1}. Items a b c d: counts
// 4 1 2 1 (N 8), pages 2 1 1 1 (T 3, U 5); Q1 = 3 (b c d), Q2 = 1 (a); observations of one-page items 1 + 2 + 1 = 4.
static void test_unseen_s4() {
  const ClumpedStats s = clumped_stats({4, 1, 2, 1}, {2, 1, 1, 1}, 3);
  CHECK(s.s_obs == 4 && s.total == 8 && s.units == 3 && s.incidences == 5);
  CHECK(s.q1 == 3 && s.q2 == 1 && s.q1_mass == 4);
  CHECK_NEAR(unit_missing_mass(s), 0.5, 1e-15);       // c's two observations sit on one page: new for a new page
  CHECK_NEAR(good_turing_missing_mass({4, 1, 2, 1}), 0.25, 1e-15);  // n1 / N counts c as seen twice
  CHECK_NEAR(incidence_missing_mass(s), 0.6, 1e-15);
  // Zero entries are ignored; nothing read means everything is new.
  const ClumpedStats z = clumped_stats({0, 4, 1, 2, 1, 0}, {0, 2, 1, 1, 1, 0}, 3);
  CHECK(z.s_obs == 4 && z.q1_mass == 4 && z.incidences == 5);
  CHECK_NEAR(unit_missing_mass(ClumpedStats{}), 1.0, 1e-15);
  CHECK_NEAR(incidence_missing_mass(ClumpedStats{}), 1.0, 1e-15);
  // Every item on one page: every observation of a new page is predicted new.
  CHECK_NEAR(unit_missing_mass(clumped_stats({5, 2}, {1, 1}, 2)), 1.0, 1e-15);
  // Chao extrapolation, incidence form: S_obs 4, Q1 3, Q2 1, T 3. Q0 = (2/3) 9/2 = 3; S(3 + t) = 4 + 3 [1 - 0.75^t].
  CHECK_NEAR(chao_extrapolated_richness(4, 3, 1, 3, 1.0), 4.75, 1e-12);
  CHECK_NEAR(chao_extrapolated_richness(4, 3, 1, 3, 2.0), 4.0 + 3.0 * (1.0 - 0.5625), 1e-12);
  CHECK_NEAR(chao_extrapolated_richness(4, 3, 1, 3, 1e9), 7.0, 1e-9);  // the bias-corrected Chao1
  CHECK_NEAR(chao_extrapolated_richness(4, 3, 1, 3, 0.0), 4.0, 1e-15);
  // Q2 = 0: Q0 = (2/3) 3 2 / 2 = 2; S(3 + 1) = 4 + 2 [1 - (1 - 3 / 9)] = 4 + 2 / 3.
  CHECK_NEAR(chao_extrapolated_richness(4, 3, 0, 3, 1.0), 4.0 + 2.0 / 3.0, 1e-12);
  // No singletons: nothing more is expected.
  CHECK_NEAR(chao_extrapolated_richness(4, 0, 2, 3, 10.0), 4.0, 1e-15);
  CHECK_NEAR(chao_extrapolated_richness(0, 0, 0, 0, 10.0), 0.0, 1e-15);
  // It grows with more reading and stays below its limit.
  double prev = 4.0;
  for (double t : {1.0, 3.0, 10.0, 100.0}) {
    const double v = chao_extrapolated_richness(4, 3, 1, 3, t);
    CHECK(v > prev && v < 7.0);
    prev = v;
  }
}

// ---------------------------------------------------------------------------
// 10 cards
// ---------------------------------------------------------------------------
static void test_cards() {
  using namespace epi_text;
  const std::string dot = kMidDot, dash = kEnDash, em = kEmDash;

  // Answer card, the example of section 10.
  AnswerCard a;
  a.question = "What is the capital of Algeria?";
  a.answer = "Algiers.";
  a.state = EpistemicState::Known;
  a.p = 0.97;
  a.p_lo = 0.92;
  a.p_hi = 0.99;
  a.grade = Grade::High;
  a.because = "infobox \"capital\" on the Algeria page; 3 sentences of the form \"Algiers, the capital of " + std::string(kEllipsis) + "\"";
  a.independent_sources = 2;
  a.contradictions = "none";
  a.fragility = 0.0;
  a.weakest_link = "entity link \"Algeria\"";
  a.weakest_link_p = 0.99;
  a.would_change_mind = "a current source naming another capital";
  CHECK_STR(format_answer_card(a),
            "Q: What is the capital of Algeria?\n"
            "Algiers.                         [Known " + dot + " P 0.97 (0.92" + dash + "0.99) " + dot + " High]\n"
            "Because: infobox \"capital\" on the Algeria page; 3 sentences of the form \"Algiers, the capital of " + kEllipsis + "\"\n"
            "Independent sources: 2 " + dot + " Contradictions: none " + dot + " Fragility: 0.0\n"
            "Weakest link: entity link \"Algeria\" (0.99)\n"
            "Would change my mind: a current source naming another capital\n");
  // Filling the opinion fields from an Opinion.
  AnswerCard f;
  f.question = "q";
  f.answer = "a";
  fill_card_opinion(f, opinion_from_counts(40, 0));
  CHECK(f.state == EpistemicState::Known && f.grade == Grade::High);
  CHECK_NEAR(f.p, 40.0 / 42.0 + 0.5 * 2.0 / 42.0, 1e-12);
  CHECK(format_answer_card(f).find("[Known") != std::string::npos);
  fill_card_opinion(f, opinion_from_counts(40, 0), opinion_from_counts(0, 40));
  CHECK(f.state == EpistemicState::Contested);
  // Optional lines are dropped when empty.
  AnswerCard bare;
  bare.question = "q";
  bare.answer = "a";
  const std::string bare_text = format_answer_card(bare);
  CHECK(bare_text.find("Because") == std::string::npos && bare_text.find("Weakest") == std::string::npos &&
        bare_text.find("Would change") == std::string::npos);

  // "I don't know" card, the example of section 10.
  UnknownCard u;
  u.question = "Who wrote Hamlet?";
  u.reason = ReasonCode::NoEvidence;
  u.why = "I have no source about \"Hamlet\"; it is outside the text I have read.";
  u.what_i_can_say = "\"author\" is a well-covered relation for books here (precision 0.90).";
  u.what_would_let_me_answer = "any source about Hamlet.";
  u.gap_id = "G-0412";
  u.demand_delta = 1;
  CHECK_STR(format_unknown_card(u),
            "Q: Who wrote Hamlet?\n"
            "I don't know.                    [NO_EVIDENCE]\n"
            "Why: I have no source about \"Hamlet\"; it is outside the text I have read.\n"
            "What I can say: \"author\" is a well-covered relation for books here (precision 0.90).\n"
            "What would let me answer: any source about Hamlet.\n"
            "Logged as gap G-0412 (demand +1).\n");
  CHECK(format_unknown_card(u).find("[NO_EVIDENCE]") != std::string::npos);
  u.reason = ReasonCode::BeyondDepth;
  CHECK(format_unknown_card(u).find("[BEYOND_DEPTH]") != std::string::npos);

  // Hypothesis card: the label comes first.
  HypothesisCard h;
  h.statement = "Atlas Shrugged was probably first published in English.";
  h.from = "G2 rule induction: author(B, P) AND nationality(P, United States) => original_language(B, English)\n"
           "      (support 41, PCA confidence 0.88), applied to author(Atlas Shrugged, Ayn Rand)";
  h.plausibility = 0.71;
  h.novelty_bits = 1.9;
  h.impact = "fills a schema gap (demand 0)";
  h.tests_planned = "T4 corpus search (for and against), T5 Wikidata, T1 consistency";
  h.status = "queued";
  const std::string ht = format_hypothesis_card(h);
  CHECK(starts_with(ht, "SPECULATION " + em + " not knowledge.\n"));
  CHECK(starts_with(ht, hypothesis_label()));
  CHECK(ht.find("Plausibility 0.71 " + dot + " Novelty 1.9 bits " + dot + " Impact: fills a schema gap (demand 0)\n") != std::string::npos);
  CHECK(ht.find("Tests planned: T4 corpus search (for and against), T5 Wikidata, T1 consistency\n") != std::string::npos);
  CHECK(ht.find("Status: queued\n") != std::string::npos);
  CHECK(ht.find("From: G2 rule induction") != std::string::npos);
  // The label is there even when everything else is empty.
  CHECK(starts_with(format_hypothesis_card(HypothesisCard{}), "SPECULATION " + em + " not knowledge."));

  // Hypothetical card, the example of section 10.
  HypotheticalCard w;
  w.assumption = "Berlin is not the capital of Germany";
  w.depth = 3;
  w.dropped_count = 4;
  w.dropped_examples = {"seat of the Bundestag"};
  w.contradictions_with_known = {"the Bundestag sits in Berlin"};
  const std::string wt = format_hypothetical_card(w);
  CHECK(starts_with(wt, "HYPOTHETICAL " + em + " assuming: Berlin is not the capital of Germany.\n"));
  CHECK(starts_with(wt, hypothetical_label("Berlin is not the capital of Germany.")));  // no doubled full stop
  CHECK_STR(wt,
            "HYPOTHETICAL " + em + " assuming: Berlin is not the capital of Germany.\n"
            "Consequences (depth " + kLeq + " 3): 4 derived facts drop (seat of the Bundestag " + kEllipsis + "); 1 contradiction with a Known fact: "
            "the Bundestag sits in Berlin " + kArrow + " the assumption conflicts with Known facts.\n"
            "Nothing here is stored.\n");
  // No consequences found.
  HypotheticalCard w0;
  w0.assumption = "The Moon is made of cheese.";
  const std::string w0t = format_hypothetical_card(w0);
  CHECK(starts_with(w0t, "HYPOTHETICAL " + em + " assuming: The Moon is made of cheese.\n"));
  CHECK(w0t.find("no derived facts drop; no contradiction with Known facts.") != std::string::npos);
  CHECK(w0t.find("Nothing here is stored.") != std::string::npos);
}


// ---------------------------------------------------------------------------
// S5 hypothesis store (hypothesis.hpp): 6.3 VOI, 4.2 lifecycle, 4.3 tiers, 7.2 updating, I1, I4, I7, I8
// ---------------------------------------------------------------------------
static void test_voi_and_tiers() {
  // EIG at p .5 with s = c = .9: P(pos) .5, p+ .9, p- .1, EIG = 1 - H(.9) = 0.5310044.
  const InfoGain g = expected_information_gain(0.5, 0.9, 0.9);
  CHECK_NEAR(g.p_pos, 0.5, 1e-12);
  CHECK_NEAR(g.p_plus, 0.9, 1e-12);
  CHECK_NEAR(g.p_minus, 0.1, 1e-12);
  CHECK_NEAR(g.eig, 0.5310044064107189, 1e-12);
  CHECK_NEAR(binary_entropy_bits(0.9), 1.0 - 0.5310044064107189, 1e-12);
  // p .2, s .9, c .8: P(pos) .34, p+ .5294, p- .0303, EIG .2534771.
  const InfoGain g2 = expected_information_gain(0.2, 0.9, 0.8);
  CHECK_NEAR(g2.p_pos, 0.34, 1e-12);
  CHECK_NEAR(g2.eig, 0.253477110345284, 1e-12);
  CHECK_NEAR(expected_information_gain(0.25, 1.0, 1.0).eig, 0.8112781244591328, 1e-12);  // a perfect test gains H(p)
  CHECK_NEAR(expected_information_gain(0.3, 0.7, 0.3).eig, 0.0, 1e-12);                // s = 1 - c tells nothing
  CHECK_NEAR(expected_information_gain(0.0, 0.9, 0.9).eig, 0.0, 1e-15);
  CHECK_NEAR(expected_information_gain(1.0, 0.9, 0.9).eig, 0.0, 1e-15);
  CHECK(expected_information_gain(0.5, 0.9, 0.9).eig > expected_information_gain(0.95, 0.9, 0.9).eig);
  // Priority = EIG (1 + Impact)(1 + max(0, I)) / Cost; a negative I and a zero cost are neutral.
  CHECK_NEAR(voi_priority(0.5, 2.0, 1.0, 4.0), 0.5 * 3.0 * 2.0 / 4.0, 1e-12);
  CHECK_NEAR(voi_priority(0.5, 2.0, -1.0, 0.0), 1.5, 1e-12);
  // 6.1 P0.
  CHECK_NEAR(plausibility_prior(0.8, 0.5), 0.4, 1e-12);
  CHECK_NEAR(plausibility_prior(0.8, 0.5, 0.5), 0.2, 1e-12);
  CHECK_NEAR(plausibility_prior(0.8, 0.5, 0.0), 0.0, 1e-15);
  // 4.3 tiers, on both sides of each cut, and the starting budget shares.
  CHECK(tier_of(0.3) == Tier::Conjecture);
  CHECK(tier_of(0.2999) == Tier::Speculation);
  CHECK(tier_of(0.05) == Tier::Speculation);
  CHECK(tier_of(0.0499) == Tier::Wild);
  CHECK_NEAR(tier_budget_share(Tier::Conjecture) + tier_budget_share(Tier::Speculation) + tier_budget_share(Tier::Wild), 1.0, 1e-12);
  CHECK_NEAR(tier_budget_share(Tier::Wild), 0.05, 1e-12);
  CHECK_STR(tier_name(Tier::Wild), "wild");
  // 7.2 Bayes factors and odds updating.
  CHECK_NEAR(bayes_factor(true, 0.9, 0.9), 9.0, 1e-9);
  CHECK_NEAR(bayes_factor(false, 0.9, 0.9), 1.0 / 9.0, 1e-9);
  CHECK_NEAR(update_posterior(0.5, 9.0), 0.9, 1e-12);
  CHECK_NEAR(update_posterior(0.7, 9.0), 21.0 / 22.0, 1e-12);
  CHECK_NEAR(update_posterior(update_posterior(0.5, 9.0), 9.0), 81.0 / 82.0, 1e-12);
  CHECK(std::isfinite(bayes_factor(true, 1.0, 1.0)) && bayes_factor(false, 1.0, 1.0) > 0.0);
}

static void test_lifecycle() {
  using S = HypState;
  // Legal arrows of 4.2.
  CHECK(legal_transition(S::Proposed, S::Scored));
  CHECK(legal_transition(S::Scored, S::Queued));
  CHECK(legal_transition(S::Queued, S::UnderTest));
  CHECK(legal_transition(S::Queued, S::Parked));
  for (S to : {S::Supported, S::Refuted, S::Parked, S::Irreducible, S::Queued}) CHECK(legal_transition(S::UnderTest, to));
  CHECK(legal_transition(S::Supported, S::Promoted));
  CHECK(legal_transition(S::Supported, S::Queued));
  CHECK(legal_transition(S::Parked, S::Scored));
  for (S from : {S::Proposed, S::Scored, S::Queued, S::Supported, S::Refuted, S::Parked, S::Irreducible}) CHECK(legal_transition(from, S::Retired));
  // Illegal ones: skipping a step, leaving a final state, undoing a refutation.
  CHECK(!legal_transition(S::Proposed, S::Queued));
  CHECK(!legal_transition(S::Queued, S::Supported));
  CHECK(!legal_transition(S::Scored, S::Promoted));
  CHECK(!legal_transition(S::Refuted, S::Promoted));
  CHECK(!legal_transition(S::Refuted, S::Queued));
  CHECK(!legal_transition(S::Irreducible, S::Queued));
  CHECK(!legal_transition(S::Parked, S::Queued));
  CHECK(!legal_transition(S::Supported, S::Refuted));
  CHECK(!legal_transition(S::UnderTest, S::Retired));
  CHECK(!legal_transition(S::UnderTest, S::Promoted));
  for (S to : {S::Proposed, S::Scored, S::Queued, S::Retired}) CHECK(!legal_transition(S::Promoted, to));
  for (S to : {S::Proposed, S::Scored, S::Queued, S::Promoted}) CHECK(!legal_transition(S::Retired, to));
  CHECK_STR(hyp_state_name(S::UnderTest), "under test");

  // Through the store: scoring sets P0, the tier and the history; an illegal move changes nothing.
  HypothesisStore st;
  Hypothesis h;
  h.id = "h-irr";
  h.key = "founded|Atlantis|-9600";
  h.generator = "G18";
  for (int i = 0; i < 6; ++i) {
    HypTest t;
    t.kind = "T8 prospective " + std::to_string(i);
    t.source = "later data";
    t.independent = true;
    t.sensitivity = 0.51;  // nearly uninformative: BF 1.02 for a positive
    t.specificity = 0.5;
    h.tests.push_back(t);
  }
  CHECK(st.propose(h));
  CHECK(!st.propose(h));  // the id is taken
  CHECK(!st.transition("h-irr", S::Queued, "skip scoring"));
  CHECK(st.find("h-irr")->state == S::Proposed);
  CHECK(st.score("h-irr", 0.5));
  CHECK(st.find("h-irr")->tier == Tier::Conjecture);
  CHECK_NEAR(st.find("h-irr")->posterior, 0.5, 1e-15);
  CHECK(st.transition("h-irr", S::Queued, "queued"));
  // Irreducible: five tests that move the posterior .5 -> .5247 (less than .05 in total).
  const double expect[5] = {0.504950495049505, 0.5099000196039992, 0.5148476039293464, 0.5197922798118908, 0.5247330813155342};
  for (int i = 0; i < 5; ++i) {
    CHECK(st.transition("h-irr", S::UnderTest, "test"));
    const auto r = st.record_result("h-irr", static_cast<std::size_t>(i), true);
    CHECK(r.has_value());
    CHECK_NEAR(st.find("h-irr")->posterior, expect[i], 1e-12);
    CHECK(*r == (i < 4 ? S::Queued : S::Irreducible));
  }
  CHECK(!st.transition("h-irr", S::UnderTest, "more"));  // irreducible: stop spending on it
  CHECK(!st.record_result("h-irr", 5, true).has_value());  // not under test
  CHECK(st.transition("h-irr", S::Retired, "stop"));
  const Hypothesis* hi = st.find("h-irr");
  CHECK(hi->history.size() == 1 + 1 + 1 + 5 * 2 + 1);  // proposed, scored, queued, 5 x (under test, result), retired
  CHECK(hi->history.back().from == S::Irreducible && hi->history.back().to == S::Retired);
  // Parked hypotheses are re-scored when the store changes.
  Hypothesis p;
  p.id = "h-park";
  p.key = "capital|Mu|?";
  p.generator = "G4";
  CHECK(st.propose(p));
  CHECK(st.score("h-park", 0.01));
  CHECK(st.find("h-park")->tier == Tier::Wild);  // below the floor: still queued, in the wild tier
  CHECK(st.transition("h-park", S::Queued, "queued"));
  CHECK(st.transition("h-park", S::Parked, "no feasible test"));
  CHECK(st.rescore_parked() == 1);
  CHECK(st.find("h-park")->state == S::Scored);
  CHECK(st.score("h-park", 0.12));  // re-scored in place
  CHECK(st.find("h-park")->state == S::Scored && st.find("h-park")->tier == Tier::Speculation);
  CHECK(st.find("h-park")->history.back().cause == "re-scored");
}

// I1: answer mode never returns an item with the hypothetical taint, directly or through derivation (and I10: contested
// items are never answers). Also the firewall's read and write rules of section 1.
static void test_firewall_I1() {
  CHECK(may_read(Mode::Answer, StoreKind::Facts));
  CHECK(!may_read(Mode::Answer, StoreKind::Hypotheses));
  CHECK(!may_read(Mode::Answer, StoreKind::Journal));
  for (StoreKind s : {StoreKind::Facts, StoreKind::Hypotheses, StoreKind::GapMap, StoreKind::Journal}) CHECK(may_read(Mode::Discovery, s));
  CHECK(!may_write(Mode::Discovery, StoreKind::Facts));
  CHECK(may_write(Mode::Discovery, StoreKind::Hypotheses) && may_write(Mode::Discovery, StoreKind::GapMap) && may_write(Mode::Discovery, StoreKind::Journal));
  CHECK(!may_write(Mode::Answer, StoreKind::Facts));
  CHECK(!may_write(Mode::Answer, StoreKind::Hypotheses));
  CHECK(may_write(Mode::Answer, StoreKind::GapMap));  // demand from a "don't know" tag

  const Opinion known = opinion_from_counts(40, 0);
  KnowledgeStore ks;
  ks.add({"A", known, std::nullopt, false, {}});
  ks.add({"H", known, std::nullopt, true, {}});                  // a hypothesis in discovery's working view
  ks.add({"D1", known, std::nullopt, false, {"A", "H"}});        // laundering attempt: arrives untainted
  ks.add({"D2", known, std::nullopt, false, {"D1"}});
  ks.add({"D3", known, std::nullopt, false, {"A"}});
  ks.add({"X", known, std::nullopt, false, {"missing"}});        // premise not in the store: fails closed
  ks.add({"C1", known, std::nullopt, false, {}});
  ks.add({"C2", known, std::nullopt, false, {"C1"}});
  CHECK(ks.find("D1")->hypothetical);  // inherited at derivation
  CHECK(ks.find("D2")->hypothetical);
  CHECK(!ks.find("D3")->hypothetical);
  CHECK(ks.find("X")->hypothetical);
  CHECK(ks.tainted("H") && ks.tainted("D1") && ks.tainted("D2") && ks.tainted("X"));
  CHECK(!ks.tainted("A") && !ks.tainted("D3"));
  CHECK(ks.answerable("A") != nullptr);
  CHECK(ks.answerable("D3") != nullptr);
  for (const char* id : {"H", "D1", "D2", "X", "nope"}) CHECK(ks.answerable(id) == nullptr);
  // Taint that arrives later still reaches the derived item at answer time: C1 is replaced by a hypothetical version.
  CHECK(ks.answerable("C2") != nullptr);
  ks.add({"C1", known, std::nullopt, true, {}});
  CHECK(!ks.find("C2")->hypothetical);  // its own flag is still clear ...
  CHECK(ks.tainted("C2"));               // ... but the derivation is followed
  CHECK(ks.answerable("C2") == nullptr);
  // A derivation cycle fails closed.
  ks.add({"Y1", known, std::nullopt, false, {"Y2"}});
  ks.add({"Y2", known, std::nullopt, false, {"Y1"}});
  CHECK(ks.tainted("Y2") && ks.answerable("Y2") == nullptr);
  // States: contested (I10), ignorant and vacuous items are not answers even when untainted; Believed is.
  ks.add({"K", known, opinion_from_counts(0, 40), false, {}});
  ks.add({"I", opinion_from_counts(0.2, 0), std::nullopt, false, {}});
  ks.add({"V", opinion_from_counts(0, 0), std::nullopt, false, {}});
  ks.add({"B", opinion_from_counts(1, 0), std::nullopt, false, {}});
  CHECK(ks.answerable("K") == nullptr && ks.answerable("I") == nullptr && ks.answerable("V") == nullptr);
  CHECK(ks.answerable("B") != nullptr);
  std::string view;
  for (const KnowledgeItem* it : ks.answer_view()) view += it->id + " ";
  CHECK_STR(view, "A B D3 ");
  for (const KnowledgeItem* it : ks.answer_view()) CHECK(!ks.tainted(it->id));
}

// I4: promotion only with evidence independent of the generator. 7.4: the generator's support is the base rate only.
static void test_promotion_I4() {
  HypothesisStore st;
  KnowledgeStore ks;
  // A teacher hypothesis whose every test comes from the teacher (one even flagged "independent"): the posterior passes
  // .9 but it never becomes supported, so it cannot be promoted.
  Hypothesis t;
  t.id = "h-teacher";
  t.key = "author|Atlas Shrugged|Ayn Rand";
  t.generator = "G16";
  for (int i = 0; i < 3; ++i) {
    HypTest x;
    x.kind = "T7 teacher";
    x.source = "G16";
    x.independent = (i == 0);  // mislabelled: the source is the generator itself
    x.sensitivity = x.specificity = 0.95;
    t.tests.push_back(x);
  }
  CHECK(st.propose(t));
  CHECK(st.score("h-teacher", 0.5));
  CHECK(st.transition("h-teacher", HypState::Queued, "queued"));
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(st.transition("h-teacher", HypState::UnderTest, "test"));
    CHECK(st.record_result("h-teacher", i, true) == HypState::Queued);
  }
  CHECK(st.find("h-teacher")->posterior > 0.999);
  CHECK(!has_independent_evidence(*st.find("h-teacher")));
  CHECK(!independent_evidence(*st.find("h-teacher"), st.find("h-teacher")->tests[0]));
  CHECK(!st.promote("h-teacher", ks).has_value());  // not supported
  CHECK(ks.find("h-teacher") == nullptr);
  CHECK(st.find("h-teacher")->hypothetical);
  CHECK_NEAR(st.generator_reliability("G16").alpha, 1.0, 1e-15);

  // A rule-induction hypothesis: one dependent positive (its own support restated) and independent positives.
  Hypothesis h;
  h.id = "h-rule";
  h.key = "language|Atlas Shrugged|English";
  h.generator = "G2";
  h.rule = "author-nationality-language";
  HypTest dep;
  dep.kind = "G2 support";
  dep.source = "G2";
  dep.independent = false;
  h.tests.push_back(dep);
  for (int i = 0; i < 5; ++i) {
    HypTest x;
    x.kind = i % 2 ? "T4 corpus search" : "T5 Wikidata";
    x.source = i % 2 ? "corpus" : "wikidata";
    x.independent = true;
    x.disconfirmation = true;
    h.tests.push_back(x);
  }
  CHECK(st.propose(h));
  CHECK(st.score("h-rule", 0.7));
  CHECK(st.transition("h-rule", HypState::Queued, "queued"));
  CHECK(st.transition("h-rule", HypState::UnderTest, "test"));
  CHECK(st.record_result("h-rule", 0, true) == HypState::Queued);  // .7 -> .9545 but the evidence is the generator's
  CHECK_NEAR(st.find("h-rule")->posterior, 21.0 / 22.0, 1e-12);
  CHECK(st.transition("h-rule", HypState::UnderTest, "test"));
  CHECK(st.record_result("h-rule", 1, true) == HypState::Supported);
  // ECS on one independent item, a = P0 = .7, W = 2: b 1/3, u 2/3, P .8, grade Low -> Candidate, not promoted.
  const auto op1 = st.promote("h-rule", ks);
  CHECK(op1.has_value());
  CHECK_NEAR(op1->r, 1.0, 1e-12);  // the dependent positive is not an ECS item
  CHECK_NEAR(op1->P, 0.8, 1e-12);
  CHECK(admission_decision(*op1) != Admission::Admit);
  CHECK(st.find("h-rule")->state == HypState::Supported);
  CHECK(ks.find("h-rule") == nullptr);
  for (std::size_t i = 2; i <= 5; ++i) {
    CHECK(st.transition("h-rule", HypState::Queued, "more evidence"));
    CHECK(st.transition("h-rule", HypState::UnderTest, "test"));
    CHECK(st.record_result("h-rule", i, true) == HypState::Supported);
  }
  // Five independent items: b 5/7, u 2/7, P 5/7 + .7 x 2/7 = .9143, Moderate -> Admit -> promoted.
  const auto op5 = st.promote("h-rule", ks);
  CHECK(op5.has_value());
  CHECK_NEAR(op5->r, 5.0, 1e-12);
  CHECK_NEAR(op5->base_rate, 0.7, 1e-12);
  CHECK_NEAR(op5->P, 5.0 / 7.0 + 0.7 * 2.0 / 7.0, 1e-12);
  CHECK(op5->grade == Grade::Moderate);
  CHECK(st.find("h-rule")->state == HypState::Promoted);
  CHECK(!st.find("h-rule")->hypothetical);
  CHECK(ks.answerable("h-rule") != nullptr);
  CHECK_NEAR(st.generator_reliability("G2").alpha, 2.0, 1e-15);
  CHECK_NEAR(st.rule_reliability("author-nationality-language").alpha, 2.0, 1e-15);
  CHECK(!st.promote("h-rule", ks).has_value());  // only once
  // Taint and promotion: an item derived from the working copy before promotion stays tainted; one derived after is clean.
  KnowledgeStore wk;
  wk.add({"h-rule", opinion_from_counts(40, 0), std::nullopt, true, {}});
  wk.add({"pre", opinion_from_counts(40, 0), std::nullopt, false, {"h-rule"}});
  wk.add({"h-rule", *op5, std::nullopt, false, {}});
  wk.add({"post", opinion_from_counts(40, 0), std::nullopt, false, {"h-rule"}});
  CHECK(wk.answerable("pre") == nullptr);
  CHECK(wk.answerable("post") != nullptr);
}

// I7: refuted hypotheses are kept and block regeneration for the cooling period.
static void test_refutation_I7() {
  HypothesisStore st;
  Hypothesis h;
  h.id = "h-ref";
  h.key = "capital|Australia|Sydney";
  h.statement = "Sydney is the capital of Australia.";
  h.generator = "G7";
  h.rule = "path:largest_city";
  HypTest x;
  x.kind = "T5 Wikidata";
  x.source = "wikidata";
  x.independent = true;
  x.disconfirmation = true;
  h.tests = {x, x};
  CHECK(st.propose(h));
  CHECK(st.score("h-ref", 0.3));
  CHECK(st.transition("h-ref", HypState::Queued, "queued"));
  CHECK(st.transition("h-ref", HypState::UnderTest, "test"));
  CHECK(st.record_result("h-ref", 0, false) == HypState::Refuted);  // .3 -> 1/22 = .04545 < .05
  const Hypothesis* r = st.find("h-ref");
  CHECK_NEAR(r->posterior, 1.0 / 22.0, 1e-12);
  CHECK(r->refuted_cycle == 0);
  CHECK(r->tests[0].done && !r->tests[0].positive);  // kept with its test
  CHECK_NEAR(st.generator_reliability("G7").beta, 2.0, 1e-15);
  CHECK_NEAR(st.rule_reliability("path:largest_city").beta, 2.0, 1e-15);
  CHECK(!st.transition("h-ref", HypState::Queued, "try again"));
  CHECK(!st.transition("h-ref", HypState::Promoted, "promote anyway"));
  // Regeneration of the same key is blocked for 10 cycles; a different key is not.
  Hypothesis again = h;
  again.id = "h-ref-2";
  CHECK(st.blocked("capital|Australia|Sydney"));
  CHECK(!st.propose(again));
  Hypothesis other = h;
  other.id = "h-other";
  other.key = "capital|Australia|Canberra";
  CHECK(st.propose(other));
  CHECK(!st.blocked("capital|Australia|Sydney", true));  // unless new evidence arrives
  CHECK(st.transition("h-ref", HypState::Retired, "negative result"));
  for (int c = 1; c <= 9; ++c) {
    st.advance_cycle();
    CHECK(st.blocked("capital|Australia|Sydney"));  // retired, still blocking
  }
  CHECK(!st.propose(again));
  st.advance_cycle();
  CHECK(st.cycle() == 10);
  CHECK(!st.blocked("capital|Australia|Sydney"));
  CHECK(st.propose(again));
  // Nothing was deleted: the refuted record is still there, with its tests and its history.
  CHECK(st.all().size() == 3);
  CHECK(st.find("h-ref") != nullptr && st.find("h-ref")->state == HypState::Retired && st.find("h-ref")->refuted_cycle == 0);
  CHECK(st.find("h-ref")->history.size() >= 5);
  CHECK(st.find("h-ref-2")->state == HypState::Proposed && st.find("h-ref-2")->hypothetical);
}

// I8: discovery output is always labelled (SPECULATION / HYPOTHETICAL).
static void test_labels_I8() {
  const std::string label = std::string("SPECULATION ") + "\xE2\x80\x94" + " not knowledge.";
  HypothesisStore st;
  int n = 0;
  for (double p0 : {0.6, 0.1, 0.01}) {
    Hypothesis h;
    h.id = "h" + std::to_string(n++);
    h.key = h.id;
    h.statement = "Atlantis had a capital.";
    h.generator = "G20";
    CHECK(st.propose(h));
    CHECK(starts_with(discovery_card(*st.find(h.id)), label));  // proposed, no tier yet
    CHECK(st.score(h.id, p0));
    const std::string card = discovery_card(*st.find(h.id));
    CHECK(starts_with(card, label));
    CHECK(is_labelled(card));
    CHECK(card.find(std::string("(") + tier_name(tier_of(p0)) + " tier)") != std::string::npos);
  }
  CHECK(discovery_card(*st.find("h2")).find("Status: scored (wild tier)") != std::string::npos);
  CHECK(starts_with(discovery_card(Hypothesis{}), label));  // an empty record is labelled too
  // Free text from discovery mode gets the label once; labelled text is left alone.
  const std::string t = label_discovery_text("Atlantis probably had a capital.");
  CHECK(starts_with(t, label + "\n") && t.find("Atlantis probably had a capital.") != std::string::npos);
  CHECK_STR(label_discovery_text(t), t);
  HypotheticalCard w;
  w.assumption = "Berlin is not the capital of Germany";
  const std::string wt = format_hypothetical_card(w);
  CHECK(is_labelled(wt));
  CHECK_STR(label_discovery_text(wt), wt);
  // An answer card is not discovery output and carries no label.
  AnswerCard a;
  a.question = "Q";
  a.answer = "A";
  CHECK(!is_labelled(format_answer_card(a)));
  CHECK(!is_labelled(""));
}

// ---------------------------------------------------------------------------
// S6 scoring (hypothesis.hpp 6.1, 6.2): the triple store, base rate, the link predictor, novelty, compression gain,
// graph distance (bridging), impact, interestingness. Every expected value is worked by hand in the comments.
// ---------------------------------------------------------------------------
static void test_scoring_S6() {
  CHECK_NEAR(novelty_bits(0.25), 2.0, 1e-12);
  CHECK_NEAR(novelty_bits(1.0), 0.0, 1e-12);
  CHECK_NEAR(novelty_bits(0.0), -std::log2(1e-12), 1e-9);  // floored, never infinite
  CHECK_NEAR(hypothesis_impact(1, 2, 3, 0.5), 4.5, 1e-12);
  const PoolStats ps = pool_stats({1, 2, 3, 4});  // mean 2.5, population sd sqrt(1.25)
  CHECK_NEAR(ps.mean, 2.5, 1e-12);
  CHECK_NEAR(ps.sd, std::sqrt(1.25), 1e-12);
  CHECK_NEAR(zscore(4, ps), 1.5 / std::sqrt(1.25), 1e-12);
  CHECK_NEAR(zscore(7, PoolStats{7, 0}), 0.0, 1e-12);  // no spread: no signal
  CHECK_NEAR(interestingness(1, 2, 3, 4), 10.0, 1e-12);

  // The store: duplicates and empty fields are ignored.
  TripleStore st;
  CHECK(st.add("z1", "occ", "poet"));
  CHECK(!st.add("z1", "occ", "poet"));
  CHECK(!st.add("z1", "", "poet"));
  st.add("z1", "nat", "French");
  st.add("z1", "era", "modern");
  st.add("z2", "occ", "poet");
  st.add("z2", "nat", "French");
  st.add("z3", "occ", "painter");
  st.add("z3", "nat", "French");
  st.add("z3", "era", "modern");
  st.add("s", "occ", "poet");
  st.add("s", "era", "modern");
  CHECK(st.size() == 10);
  CHECK(st.count("nat", "French") == 3 && st.count("nat") == 3 && st.values("nat") == 1 && st.count("occ") == 4);
  CHECK(st.has("z2", "nat", "French") && !st.has("s", "nat", "French"));
  CHECK(st.has_relation("s", "era") && !st.has_relation("s", "nat"));
  CHECK(st.facts_of("s").size() == 2 && st.facts_of("nobody").empty());
  CHECK(st.subjects_with("occ", "poet").size() == 3);
  CHECK(st.neighbours("poet").size() == 3 && st.neighbours("s").size() == 2);

  // Base rate: (c + 1) / (n + values + 1). nat=French: (3 + 1) / (3 + 2) = 0.8; unseen German: 1/5. Left out of its own
  // count (self): (2 + 1) / (2 + 2) = 0.75.
  CHECK_NEAR(base_rate(st, "nat", "French"), 0.8, 1e-12);
  CHECK_NEAR(base_rate(st, "nat", "German"), 0.2, 1e-12);
  CHECK_NEAR(base_rate(st, "nat", "French", 1.0, true), 0.75, 1e-12);

  // The link predictor, m = 1. h = (s, nat, French), b = 0.8 (logit ln 4). Body (occ, poet): other subjects z1, z2 both
  // have a nat, both French: c = (2 + 0.8) / 3 = 14/15 (odds 14). Body (era, modern): z1, z3, both French: the same c.
  // Naive Bayes: odds = 4 x (14/4)^2 = 49, so P = 49/50 = 0.98.
  const LinkPredictor lp(st, {});
  CHECK_NEAR(lp.prob("s", "nat", "French"), 0.98, 1e-9);
  // Without (era, modern) in the body (skip it): one rule, P = 14/15.
  const TripleStore::Fact era{"era", "modern"};
  CHECK_NEAR(lp.prob("s", "nat", "French", &era), 14.0 / 15.0, 1e-9);
  // Shrinkage m = 2, one rule: c = (2 + 2 x 0.8) / (2 + 2) = 0.9.
  CHECK_NEAR(LinkPredictor(st, {2.0, true}).prob("s", "nat", "French", &era), 0.9, 1e-9);
  // A subject with no facts: the base rate.
  CHECK_NEAR(lp.prob("nobody", "nat", "French"), 0.8, 1e-9);
  // Compression gain of a hypothesis on a subject with no other facts: only its own cost, log2 0.8.
  CHECK_NEAR(compression_gain_bits(lp, st, "nobody", "nat", "French"), std::log2(0.8), 1e-9);
  // ... and on s: its own cost, plus what it saves on s's two facts. Both terms are finite; h explains its co-facts here,
  // so the gain exceeds its own cost.
  const double dl = compression_gain_bits(lp, st, "s", "nat", "French");
  CHECK(std::isfinite(dl) && dl > std::log2(0.98));

  // A copy rule. bp and dp: z4 and z5 have both, z4's are the same (Paris), z5's differ. h = (s2, bp, Lyon) with s2's
  // dp = Lyon. b(bp = Lyon) = (0 + 1) / (2 + 3) = 0.2; no other subject has dp = Lyon, so only the copy rule fires:
  // c = (1 + 0.2) / (2 + 1) = 0.4, and P = 0.4. Without copy rules, P = b = 0.2.
  TripleStore cp;
  cp.add("z4", "bp", "Paris");
  cp.add("z4", "dp", "Paris");
  cp.add("z5", "bp", "Rome");
  cp.add("z5", "dp", "Milan");
  cp.add("s2", "dp", "Lyon");
  CHECK_NEAR(LinkPredictor(cp, {}).prob("s2", "bp", "Lyon"), 0.4, 1e-9);
  CHECK_NEAR(LinkPredictor(cp, {1.0, false}).prob("s2", "bp", "Lyon"), 0.2, 1e-9);

  // Adamic-Adar (S6b's path term): z1 and z2 share poet (neighbours z1, z2, s) and French (z1, z2, z3), each of degree 3:
  // 2 / ln 3. s and French share no neighbour: 0.
  CHECK_NEAR(adamic_adar(st, "z1", "z2"), 2.0 / std::log(3.0), 1e-12);
  CHECK_NEAR(adamic_adar(st, "s", "French"), 0.0, 1e-12);

  // Graph distance (bridging): s - poet - z1 - French; nothing links Lyon to the first store.
  CHECK(graph_distance(st, "s", "s") == 0);
  CHECK(graph_distance(st, "s", "poet") == 1);
  CHECK(graph_distance(st, "s", "French") == 3);
  CHECK(graph_distance(st, "s", "French", 2) == 3);  // beyond the cap: cap + 1
  CHECK(graph_distance(st, "s", "Lyon") == 7);
}

// ---------------------------------------------------------------------------
// S3 gap map (gap_map.hpp): the tag, the gap id, schema cells, demand, VOI
// ---------------------------------------------------------------------------
// S7 (DISCOVERY 7.1-7.3, I5, I6; docs/reports/S7_HYPOTHESIS_TESTS.md): three-way results, e-values and BH batch
// control, measured s and c, T3's held-out rule check, T1's functional check, and the pre-registration log.
static Hypothesis s7_hyp(const std::string& id, bool with_disconfirmation) {
  Hypothesis h;
  h.id = id;
  h.key = "capital|Atlantis|" + id;
  h.generator = "planted-store";
  HypTest t1;
  t1.kind = "T1 consistency", t1.source = "store", t1.predicted = "positive", t1.refuted_if = "negative";
  t1.disconfirmation = with_disconfirmation, t1.sensitivity = 0.75, t1.specificity = 0.92;
  HypTest t4;
  t4.kind = "T4 corpus", t4.source = "corpus", t4.predicted = "positive", t4.refuted_if = "negative";
  t4.independent = true, t4.sensitivity = 0.99, t4.specificity = 0.8;
  h.tests = {t1, t4};
  return h;
}

static void test_tests_S7() {
  // Three-way Bayes factors and the e-value.
  CHECK_NEAR(bayes_factor(TestOutcome::Positive, 0.9, 0.8), 4.5, 1e-12);
  CHECK_NEAR(bayes_factor(TestOutcome::Negative, 0.9, 0.8), 0.125, 1e-12);
  CHECK_NEAR(bayes_factor(TestOutcome::Inconclusive, 0.9, 0.8), 1.0, 1e-15);
  CHECK_STR(test_outcome_name(TestOutcome::Inconclusive), "inconclusive");
  {
    Hypothesis h;
    h.tests.resize(4);
    h.tests[0].done = true, h.tests[0].bayes_factor = 4.5;
    h.tests[1].done = true, h.tests[1].bayes_factor = 0.125;
    h.tests[2].done = true, h.tests[2].inconclusive = true, h.tests[2].bayes_factor = 1.0;
    h.tests[3].bayes_factor = 1000.0;  // not run: not counted
    CHECK_NEAR(e_value(h), 0.5625, 1e-12);
  }
  // BH on p = 1/e: p {0.001, 0.05, 1, 1} at q 0.1 over m = 4 rejects the first two.
  const auto rej = bh_reject_e({1000.0, 20.0, 1.0, 0.5}, 0.1);
  CHECK(rej.size() == 4 && rej[0] && rej[1] && !rej[2] && !rej[3]);
  CHECK(bh_reject_e({}, 0.1).empty());

  // Measured s and c: true 8 pos / 2 neg / 10 inc, false 1 pos / 9 neg / 10 inc.
  TestTally tt;
  for (int i = 0; i < 8; ++i) tt.add(true, TestOutcome::Positive);
  for (int i = 0; i < 2; ++i) tt.add(true, TestOutcome::Negative);
  for (int i = 0; i < 10; ++i) tt.add(true, TestOutcome::Inconclusive), tt.add(false, TestOutcome::Inconclusive);
  tt.add(false, TestOutcome::Positive);
  for (int i = 0; i < 9; ++i) tt.add(false, TestOutcome::Negative);
  CHECK_NEAR(tt.sensitivity(), 0.8, 1e-12);
  CHECK_NEAR(tt.specificity(), 0.9, 1e-12);
  CHECK_NEAR(tt.applicable(), 0.5, 1e-12);
  CHECK_NEAR(tt.youden(), 0.7, 1e-12);
  CHECK_NEAR(tt.s_lo(), 0.49016247153664183, 1e-12);
  CHECK_NEAR(tt.s_hi(), 0.9433178485456247, 1e-12);
  CHECK_NEAR(tt.youden_lo(), 0.08601244474140346, 1e-12);
  CHECK_NEAR(tt.mutual_information_bits(), 0.19865630487474323, 1e-12);
  CHECK_NEAR(TestTally{}.mutual_information_bits(), 0.0, 1e-15);
  TestTally blind;  // a test that says the same whatever the truth carries no information
  for (int i = 0; i < 5; ++i) blind.add(true, TestOutcome::Positive), blind.add(false, TestOutcome::Positive);
  CHECK_NEAR(blind.mutual_information_bits(), 0.0, 1e-12);
  CHECK_NEAR(blind.youden(), 0.0, 1e-12);

  // T3: Wilson lower 0.596 >= 0.3 positive; upper 0.278 < 0.5 negative; 3 of 6 inconclusive; fewer than 3 inconclusive.
  CHECK(rule_holdout_test(9, 10) == TestOutcome::Positive);
  CHECK(rule_holdout_test(0, 10) == TestOutcome::Negative);
  CHECK(rule_holdout_test(3, 6) == TestOutcome::Inconclusive);
  CHECK(rule_holdout_test(2, 2) == TestOutcome::Inconclusive);
  // T1's functional check.
  TripleStore st;
  st.add("Atlantis", "capital", "Poseidonia");
  CHECK(functional_conflict(st, "Atlantis", "capital", "Basileia"));
  CHECK(!functional_conflict(st, "Atlantis", "capital", "Poseidonia"));
  CHECK(!functional_conflict(st, "Lemuria", "capital", "Basileia"));

  // I6: a plan with no disconfirmation search is refused; so is an empty plan.
  PreregLog log;
  std::string why;
  CHECK(!log.register_plan(s7_hyp("h-noneg", false), &why));
  CHECK(why.find("I6") == 0);
  Hypothesis empty = s7_hyp("h-empty", true);
  empty.tests.clear();
  CHECK(!log.register_plan(empty));
  CHECK(log.entries().empty());
  // I5: results only under a registered plan, once each; no plan change after a result; amendments before it stay visible.
  CHECK(!log.record("h-a", 0, TestOutcome::Positive, &why));  // not registered
  CHECK(why.find("I5") == 0);
  CHECK(log.register_plan(s7_hyp("h-a", true)));
  CHECK(log.register_plan(s7_hyp("h-a", true)));  // an amendment before any result: allowed, and kept in the log
  CHECK(log.entries().size() == 2);
  CHECK(log.record("h-a", 0, TestOutcome::Positive));
  CHECK(!log.record("h-a", 0, TestOutcome::Negative));  // recorded once
  CHECK(!log.record("h-a", 2, TestOutcome::Positive));  // not in the plan
  CHECK(!log.register_plan(s7_hyp("h-a", true), &why));  // a plan cannot change after a result
  CHECK(why.find("I5") == 0);
  CHECK(log.record("h-a", 1, TestOutcome::Inconclusive));
  CHECK(log.has_result("h-a") && !log.has_result("h-"));
  CHECK(log.plan("h-a") && log.plan("h-a")->size() == 2 && (*log.plan("h-a"))[1].kind == "T4 corpus");
  CHECK_NEAR((*log.plan("h-a"))[0].specificity, 0.92, 1e-15);
  CHECK(log.entries().size() == 4);
  CHECK(log.verify());
  // The log as stored: it reads back intact, and an edit, a dropped line or a swap breaks the chain.
  const std::string tsv = log.serialize();
  PreregLog back = PreregLog::parse(tsv);
  CHECK(back.verify() && back.entries().size() == 4);
  CHECK(back.registered("h-a") && !back.record("h-a", 0, TestOutcome::Negative));  // its results come back too
  CHECK(!back.register_plan(s7_hyp("h-a", true)));
  std::string edited = tsv;
  const std::size_t at = edited.find("0 positive");
  CHECK(at != std::string::npos);
  edited.replace(at, 10, "0 negative");  // the recorded outcome changed after the fact
  CHECK(!PreregLog::parse(edited).verify());
  std::string plan_edit = tsv;
  plan_edit.replace(plan_edit.find("0.92"), 4, "0.99");  // the registered specificity changed
  CHECK(!PreregLog::parse(plan_edit).verify());
  const std::size_t nl1 = tsv.find('\n'), nl2 = tsv.find('\n', nl1 + 1);
  CHECK(!PreregLog::parse(tsv.substr(0, nl1 + 1) + tsv.substr(nl2 + 1)).verify());  // the second line dropped
  CHECK(!PreregLog::parse(tsv.substr(nl1 + 1, nl2 - nl1) + tsv.substr(0, nl1 + 1) + tsv.substr(nl2 + 1)).verify());  // swapped
  {
    Hypothesis tab = s7_hyp("h\twith\ttabs", true);  // escaping keeps an awkward id intact
    PreregLog l2;
    CHECK(l2.register_plan(tab) && l2.record(tab.id, 0, TestOutcome::Negative));
    const PreregLog b2 = PreregLog::parse(l2.serialize());
    CHECK(b2.verify() && b2.registered(tab.id) && b2.has_result(tab.id));
  }

  // The store with the log: a test runs only under its registered plan; inconclusive is BF 1 and no evidence (I4).
  HypothesisStore hs;
  PreregLog hl;
  CHECK(hs.propose(s7_hyp("h-s", true)));
  CHECK(hs.score("h-s", 0.5));
  CHECK(hs.transition("h-s", HypState::Queued, "scored"));
  CHECK(hs.transition("h-s", HypState::UnderTest, "test"));
  CHECK(!hs.record_outcome("h-s", 0, TestOutcome::Positive, &hl).has_value());  // not pre-registered
  CHECK(!hs.find("h-s")->tests[0].done);
  CHECK(hl.register_plan(*hs.find("h-s")));
  auto r = hs.record_outcome("h-s", 0, TestOutcome::Positive, &hl);  // T1 positive: BF 0.75 / 0.08 = 9.375
  CHECK(r.has_value() && *r == HypState::Queued);  // posterior 0.904 but T1 is the store: not independent
  CHECK_NEAR(hs.find("h-s")->posterior, 9.375 / 10.375, 1e-12);
  CHECK(!hs.record_outcome("h-s", 0, TestOutcome::Positive, &hl).has_value());  // not under test, and recorded
  CHECK(hs.transition("h-s", HypState::UnderTest, "test"));
  r = hs.record_outcome("h-s", 1, TestOutcome::Inconclusive, &hl);
  CHECK(r.has_value() && *r == HypState::Queued);
  const Hypothesis* hp = hs.find("h-s");
  CHECK(hp->tests[1].done && hp->tests[1].inconclusive);
  CHECK_NEAR(hp->tests[1].bayes_factor, 1.0, 1e-15);
  CHECK_NEAR(hp->posterior, 9.375 / 10.375, 1e-12);
  CHECK(!has_independent_evidence(*hp));  // the corpus test ran but could not tell
  CHECK_NEAR(e_value(*hp), 9.375, 1e-12);
  // With an independent positive it is supported.
  CHECK(hs.propose(s7_hyp("h-t", true)));
  CHECK(hs.score("h-t", 0.7));
  CHECK(hs.transition("h-t", HypState::Queued, "scored"));
  CHECK(hl.register_plan(*hs.find("h-t")));
  CHECK(hs.transition("h-t", HypState::UnderTest, "test"));
  CHECK(hs.record_outcome("h-t", 1, TestOutcome::Positive, &hl) == HypState::Supported);  // BF 0.99 / 0.2 = 4.95
  CHECK_NEAR(hs.find("h-t")->posterior, 0.7 * 4.95 / (0.7 * 4.95 + 0.3), 1e-12);
  CHECK(has_independent_evidence(*hs.find("h-t")));
  CHECK(hl.verify());
}

// S7b (docs/reports/S7_HYPOTHESIS_TESTS.md, "S7b"): three-way likelihoods and the batch step among the supported.
static void test_s7b() {
  // likelihoods_from: the add-one mean of each row over (positive, negative, inconclusive).
  TestTally tl;
  for (int i = 0; i < 6; ++i) tl.add(true, TestOutcome::Positive);
  tl.add(true, TestOutcome::Negative);
  for (int i = 0; i < 3; ++i) tl.add(true, TestOutcome::Inconclusive);
  tl.add(false, TestOutcome::Positive);
  for (int i = 0; i < 4; ++i) tl.add(false, TestOutcome::Negative);
  for (int i = 0; i < 5; ++i) tl.add(false, TestOutcome::Inconclusive);
  const OutcomeLikelihoods L = likelihoods_from(tl);
  CHECK(L.set() && !OutcomeLikelihoods{}.set());
  CHECK_NEAR(L.t[0], 7.0 / 13, 1e-12);
  CHECK_NEAR(L.t[1], 2.0 / 13, 1e-12);
  CHECK_NEAR(L.t[2], 4.0 / 13, 1e-12);
  CHECK_NEAR(L.f[0], 2.0 / 13, 1e-12);
  CHECK_NEAR(L.f[1], 5.0 / 13, 1e-12);
  CHECK_NEAR(L.f[2], 6.0 / 13, 1e-12);
  // A test's BF: S7's two-way factor without likelihoods; the likelihood ratio with them, inconclusive included.
  HypTest t;
  t.sensitivity = 0.9, t.specificity = 0.8;
  CHECK_NEAR(bayes_factor(TestOutcome::Positive, t), 4.5, 1e-12);
  CHECK_NEAR(bayes_factor(TestOutcome::Inconclusive, t), 1.0, 1e-15);
  t.likelihoods = L;
  CHECK_NEAR(bayes_factor(TestOutcome::Positive, t), 3.5, 1e-12);
  CHECK_NEAR(bayes_factor(TestOutcome::Negative, t), 0.4, 1e-12);
  CHECK_NEAR(bayes_factor(TestOutcome::Inconclusive, t), 4.0 / 6, 1e-12);
  {
    HypTest tz = t;
    tz.likelihoods.f[0] = 0.0;  // a zero likelihood is clamped: the factor is large, never infinite
    CHECK_NEAR(bayes_factor(TestOutcome::Positive, tz), (7.0 / 13) / 1e-6, 1e-3);
  }
  // In the store: silence moves the posterior by its ratio, and is still no independent evidence (I4).
  Hypothesis h = s7_hyp("h-3", true);
  h.tests[0].likelihoods = L, h.tests[1].likelihoods = L;
  HypothesisStore hs;
  PreregLog log;
  CHECK(hs.propose(h) && hs.score("h-3", 0.5) && hs.transition("h-3", HypState::Queued, "scored"));
  CHECK(log.register_plan(*hs.find("h-3")));
  CHECK(hs.transition("h-3", HypState::UnderTest, "test"));
  auto r = hs.record_outcome("h-3", 0, TestOutcome::Inconclusive, &log);
  CHECK(r.has_value() && *r == HypState::Queued);
  CHECK_NEAR(hs.find("h-3")->posterior, (4.0 / 6) / (1 + 4.0 / 6), 1e-12);
  CHECK(hs.find("h-3")->tests[0].inconclusive && !has_independent_evidence(*hs.find("h-3")));
  CHECK(hs.transition("h-3", HypState::UnderTest, "test"));
  r = hs.record_outcome("h-3", 1, TestOutcome::Positive, &log);  // odds 2/3 x 3.5 = 7/3: 0.7, queued
  CHECK(r.has_value() && *r == HypState::Queued);
  CHECK_NEAR(hs.find("h-3")->posterior, 0.7, 1e-12);
  CHECK_NEAR(hs.find("h-3")->tests[1].bayes_factor, 3.5, 1e-12);  // record_result uses the three-way factor
  CHECK_NEAR(e_value(*hs.find("h-3")), (4.0 / 6) * 3.5, 1e-12);
  CHECK(log.verify());
  {  // a posterior above 0.9 from silence alone is never supported
    OutcomeLikelihoods sil;
    sil.t[0] = 0.1, sil.t[1] = 0.1, sil.t[2] = 0.8, sil.f[0] = 0.1, sil.f[1] = 0.1, sil.f[2] = 0.08;  // silence: BF 10
    Hypothesis g = s7_hyp("h-sil", true);
    g.tests[0].likelihoods = sil, g.tests[1].likelihoods = sil;
    HypothesisStore gs;
    CHECK(gs.propose(g) && gs.score("h-sil", 0.5) && gs.transition("h-sil", HypState::Queued, "scored"));
    CHECK(gs.transition("h-sil", HypState::UnderTest, "test"));
    CHECK(gs.record_outcome("h-sil", 0, TestOutcome::Inconclusive) == HypState::Queued);
    CHECK(gs.transition("h-sil", HypState::UnderTest, "test"));
    const auto q = gs.record_outcome("h-sil", 1, TestOutcome::Inconclusive);
    CHECK(q.has_value() && *q != HypState::Supported);
    CHECK(gs.find("h-sil")->posterior > 0.9);
  }
  {  // without likelihoods an inconclusive result leaves the posterior exactly as it was (S7)
    Hypothesis g = s7_hyp("h-2w", true);
    HypothesisStore gs;
    CHECK(gs.propose(g) && gs.score("h-2w", 0.37) && gs.transition("h-2w", HypState::Queued, "scored"));
    CHECK(gs.transition("h-2w", HypState::UnderTest, "test"));
    CHECK(gs.record_outcome("h-2w", 1, TestOutcome::Inconclusive) == HypState::Queued);
    CHECK(gs.find("h-2w")->posterior == 0.37);
  }
  {  // the likelihoods are registered with the plan (I5): an edit of one breaks the chain
    PreregLog a, b;
    Hypothesis g1 = s7_hyp("p1", true), g2 = g1;
    g2.tests[0].likelihoods = L;
    CHECK(a.register_plan(g1) && b.register_plan(g2));
    const std::string s1 = a.serialize(), s2 = b.serialize();
    CHECK(s2.size() > s1.size());
    const std::size_t at = s2.find("0.53846");
    CHECK(at != std::string::npos);
    std::string edited = s2;
    if (at != std::string::npos) edited[at + 6] = edited[at + 6] == '9' ? '8' : '9';
    CHECK(PreregLog::parse(s2).verify() && !PreregLog::parse(edited).verify());
  }
  // The batch step among the supported: BH at q 0.1 on p = 1/e over the selected only.
  {
    const std::vector<double> e = {1000, 1e9, 50, 2, 1.5, 400};
    const std::vector<bool> sel = {true, false, true, true, true, true};
    // Selected p: 0.001, 0.02, 0.5, 0.667, 0.0025; m 5, thresholds 0.02, 0.04, 0.06, 0.08, 0.1: the three smallest pass.
    const auto rej = bh_reject_e_among(e, sel, 0.1);
    CHECK(rej.size() == 6);
    CHECK(rej[0] && !rej[1] && rej[2] && !rej[3] && !rej[4] && rej[5]);  // never the unselected one, however large its e
    const auto all = bh_reject_e(e, 0.1);  // the whole batch (m 6) rejects the unselected one too
    CHECK(all[0] && all[1] && all[2] && !all[3] && !all[4] && all[5]);
    const auto none = bh_reject_e_among(e, std::vector<bool>(6, false), 0.1);
    CHECK(none.size() == 6 && !none[0] && !none[1]);
  }
}

// ---------------------------------------------------------------------------
// S8b: G6 neighbourhood scores, the one-pass link scorer, G7 path ranking (gen_links.hpp)
// ---------------------------------------------------------------------------

static void test_links_S8b() {
  // G6: a and b share two private neighbours; a and c share only a hub. RA (1 / deg) penalises the hub more than AA does.
  {
    TripleStore st;
    st.add("a", "r", "p1"), st.add("b", "r", "p1"), st.add("a", "r", "p2"), st.add("b", "r", "p2");
    st.add("a", "r", "hub"), st.add("c", "r", "hub");
    for (int i = 0; i < 20; ++i) st.add("x" + std::to_string(i), "r", "hub");
    const auto ns = neighbour_scores(st, "a");
    CHECK(ns.count("b") && ns.count("c"));
    CHECK_NEAR(ns.at("b").cn, 2.0, 1e-12);
    CHECK_NEAR(ns.at("c").cn, 1.0, 1e-12);
    CHECK_NEAR(ns.at("b").aa, adamic_adar(st, "a", "b"), 1e-12);  // AA agrees with S6's adamic_adar
    CHECK_NEAR(ns.at("c").aa, adamic_adar(st, "a", "c"), 1e-12);
    CHECK_NEAR(ns.at("b").ra, 2.0 * 0.5, 1e-12);                   // p1 and p2 each have degree 2
    CHECK_NEAR(ns.at("c").ra, 1.0 / 22.0, 1e-12);                  // the hub has 22 neighbours
    CHECK(ns.at("c").ra / ns.at("c").aa < ns.at("b").ra / ns.at("b").aa);  // degree normalisation hits the hub hardest
    CHECK_NEAR(ns.at("b").jaccard, 2.0 / 3.0, 1e-12);              // |{p1,p2}| / |{p1,p2,hub}|
    CHECK(!ns.count("a"));
  }
  // The one-pass link scorer gives LinkPredictor::prob's numbers, with and without copy rules.
  {
    TripleStore st;
    std::mt19937_64 rng(7);
    const char* rel[] = {"born", "lives", "job", "nat"};
    for (int i = 0; i < 60; ++i)
      for (int k = 0; k < 4; ++k)
        if (rng() % 3) st.add("s" + std::to_string(i), rel[k], "v" + std::to_string(rng() % 6));
    for (const bool copy : {true, false}) {
      const LinkPredictor lp(st, {1.0, copy});
      const LinkScorer ls(st, {1.0, copy});
      double worst = 0.0;
      int compared = 0;
      for (int i = 0; i < 60; ++i)
        for (const char* r : rel) {
          std::vector<std::string> cands;
          for (int v = 0; v < 6; ++v)
            if (!st.has("s" + std::to_string(i), r, "v" + std::to_string(v))) cands.push_back("v" + std::to_string(v));
          const auto ps = ls.probs("s" + std::to_string(i), r, cands);
          for (std::size_t c = 0; c < cands.size(); ++c, ++compared)
            worst = std::max(worst, std::fabs(ps[c] - lp.prob("s" + std::to_string(i), r, cands[c])));
        }
      CHECK(compared > 500);
      CHECK_NEAR(worst, 0.0, 1e-12);
    }
  }
  // Logistic regression: one informative feature gets a positive weight and orders the rows.
  {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    for (int i = 0; i < 40; ++i) X.push_back({static_cast<double>(i % 10), 1.0}), y.push_back(i % 10 >= 5 ? 1 : 0);
    y[3] = 1, y[17] = 0;  // not separable, so the fit is finite
    Logistic lr;
    lr.fit(X, y, 1.0);
    CHECK(lr.ok);
    CHECK(lr.w[0] > 0.0);
    CHECK(lr.prob({9.0, 1.0}) > 0.8 && lr.prob({0.0, 1.0}) < 0.2);
    Logistic one;
    one.fit({{1.0}, {2.0}}, {1, 1});  // one label only: no fit
    CHECK(!one.ok);
  }
  // G7: the planted rule born_in(X, C) and city_in(C, K) => nationality(X, K). People are born in 6 cities of 3
  // countries; one person's nationality is missing. Path ranking puts the right country first and explains it by the
  // path born_in > city_in. The walk probabilities are exact on a small chain.
  {
    TripleStore st;
    const char* country[] = {"K0", "K1", "K2"};
    for (int c = 0; c < 6; ++c) st.add("C" + std::to_string(c), "city_in", country[c % 3]);
    for (int i = 0; i < 90; ++i) {
      const int c = i % 6;
      st.add("P" + std::to_string(i), "born_in", "C" + std::to_string(c));
      if (i != 4) st.add("P" + std::to_string(i), "nationality", country[c % 3]);
    }
    // Popularity favours K0 (a dozen extra K0 nationals with no birthplace), so the base rate alone would be wrong for P4.
    for (int i = 0; i < 12; ++i) st.add("Q" + std::to_string(i), "nationality", "K0");
    const RelGraph g(st);
    PathRanker pr(st, g, {});
    const auto& M = pr.fit("nationality");
    CHECK(M.ok);
    bool has_rule_path = false;
    for (const auto& p : M.paths) has_rule_path = has_rule_path || pr.path_name(p) == "born_in > city_in";
    CHECK(has_rule_path);
    std::vector<int> best;
    const std::vector<std::string> cands = {"K0", "K1", "K2"};
    const auto sc = pr.score("P4", "nationality", cands, &best);  // P4 is born in C4, in K1
    CHECK(sc[1] > sc[0] && sc[1] > sc[2]);
    CHECK(best[1] >= 0 && pr.path_name(M.paths[static_cast<std::size_t>(best[1])]) == "born_in > city_in");
    CHECK(base_rate(st, "nationality", "K0") > base_rate(st, "nationality", "K1"));
    // The walk along born_in > city_in from P4 reaches K1 with probability 1; along born_in alone, C4 with 1.
    const int p4 = g.node("P4");
    const auto d = pr.walks(p4, {{g.forward("born_in"), g.forward("city_in"), -1}, {g.forward("born_in"), -1, -1}}, SkipEdge{});
    CHECK_NEAR(d[0].at(g.node("K1")), 1.0, 1e-12);
    CHECK(d[0].size() == 1);
    CHECK_NEAR(d[1].at(g.node("C4")), 1.0, 1e-12);
    // Back along born_in^-1 from C4: 15 people, 1/15 each; leaving out the edge P4-C4 removes P4 and gives 1/14.
    const int c4 = g.node("C4"), inv = g.forward("born_in") ^ 1;
    const auto e = pr.walks(c4, {{inv, -1, -1}}, SkipEdge{});
    CHECK(e[0].size() == 15);
    CHECK_NEAR(e[0].at(p4), 1.0 / 15.0, 1e-12);
    const auto f = pr.walks(c4, {{inv, -1, -1}}, SkipEdge{p4, g.forward("born_in"), c4});
    CHECK(f[0].size() == 14 && !f[0].count(p4));
    CHECK_NEAR(f[0].begin()->second, 1.0 / 14.0, 1e-12);
  }
  // A length-1 path: everyone works where they live, so works_in is predicted by the path lives_in alone.
  {
    TripleStore st;
    for (int i = 0; i < 40; ++i) {
      st.add("W" + std::to_string(i), "lives_in", "T" + std::to_string(i % 4));
      if (i != 6) st.add("W" + std::to_string(i), "works_in", "T" + std::to_string(i % 4));
    }
    for (int i = 0; i < 10; ++i) st.add("V" + std::to_string(i), "works_in", "T0");  // popularity favours T0
    const RelGraph g(st);
    PathRanker pr(st, g, {});
    const auto& M = pr.fit("works_in");
    CHECK(M.ok);
    bool one = false;
    for (const auto& p : M.paths) one = one || pr.path_name(p) == "lives_in";
    CHECK(one);
    const auto sc = pr.score("W6", "works_in", {"T0", "T1", "T2", "T3"});  // W6 lives in T2
    CHECK(sc[2] > sc[0] && sc[2] > sc[1] && sc[2] > sc[3]);
  }
  // The hub guard: a walk does not go on through an intermediate node with more edges of the step's type than `hub`.
  {
    TripleStore st;
    st.add("a", "nat", "H");
    for (int i = 0; i < 30; ++i) st.add("p" + std::to_string(i), "nat", "H"), st.add("p" + std::to_string(i), "born", "T");
    const RelGraph g(st);
    PathRanker::Config small;
    small.hub = 10;
    PathRanker guarded(st, g, small), open(st, g, {});
    const std::array<int, 3> path{g.forward("nat"), g.forward("nat") ^ 1, g.forward("born")};
    CHECK(guarded.walks(g.node("a"), {path}, SkipEdge{})[0].empty());       // H has 31 nat^-1 edges, over 10
    CHECK_NEAR(open.walks(g.node("a"), {path}, SkipEdge{})[0].at(g.node("T")), 30.0 / 31.0, 1e-12);  // a itself has no born edge
  }
}

// ---------------------------------------------------------------------------
// S8a: G2 rule induction and G14 contradiction resolution (gen_rules.hpp)
// ---------------------------------------------------------------------------

static void test_gen_rules_S8a() {
  // The binomial tail.
  CHECK_NEAR(binomial_upper_tail(8, 10, 0.5), 56.0 / 1024.0, 1e-12);
  CHECK_NEAR(binomial_upper_tail(0, 5, 0.2), 1.0, 1e-15);
  CHECK_NEAR(binomial_upper_tail(6, 5, 0.2), 0.0, 1e-15);
  CHECK_NEAR(binomial_upper_tail(5, 5, 0.2), std::pow(0.2, 5), 1e-15);

  TripleStore st;
  // A planted const rule: 30 painters, all artists. Another 20 people have occupation(lawyer) and a random other value.
  for (int i = 0; i < 30; ++i) {
    const std::string x = "P" + std::to_string(i);
    st.add(x, "occupation", "Painter");
    if (i < 27) st.add(x, "occupation", "Artist");  // 27 of 30: PCA 0.9
  }
  for (int i = 0; i < 20; ++i) st.add("L" + std::to_string(i), "occupation", "Lawyer"), st.add("L" + std::to_string(i), "occupation", "V" + std::to_string(i));
  // A spurious rule: 100 subjects; 80% are "common"; the first 50 also have tag z. tag(z) => kind(common) has support 40 and
  // PCA 0.8, but common's base rate is 0.8, so it is no better than chance and BH keeps it out.
  for (int i = 0; i < 100; ++i) {
    const std::string x = "C" + std::to_string(i);
    st.add(x, "kind", i % 5 ? "common" : "rare" + std::to_string(i));
    if (i < 50) st.add(x, "tag", "z");
  }
  // An inverse rule: 15 offices, successor and predecessor both stated.
  for (int i = 0; i < 15; ++i) {
    st.add("A" + std::to_string(i), "successor", "B" + std::to_string(i));
    st.add("B" + std::to_string(i), "predecessor", "A" + std::to_string(i));
  }
  // A path rule: 12 people born in a town whose country is their nationality. One more is born there, nationality unknown.
  for (int i = 0; i < 12; ++i) {
    const std::string town = "Town" + std::to_string(i % 4), land = "Land" + std::to_string(i % 4);
    st.add("H" + std::to_string(i), "birth_place", town);
    st.add("H" + std::to_string(i), "nationality", land);
    if (i < 4) st.add(town, "country", land);
  }
  st.add("H99", "birth_place", "Town1");
  st.add("H99", "occupation", "Painter");  // G2 should propose Artist for H99

  const RuleIndex ix(st);
  const std::vector<Rule> rules = mine_rules(ix);
  auto find = [&](const std::string& text) -> const Rule* {
    for (const auto& R : rules)
      if (R.text() == text) return &R;
    return nullptr;
  };
  const Rule* planted = find("occupation(X, Painter) => occupation(X, Artist)");
  CHECK(planted != nullptr);
  if (planted) {
    CHECK(planted->kept);
    CHECK_NEAR(planted->support, 27, 0);
    CHECK_NEAR(planted->pca_body, 31, 0);  // the 30 painters and H99
    CHECK_NEAR(planted->body, 31, 0);
    CHECK_NEAR(planted->pca_conf(), 27.0 / 31.0, 1e-12);
    CHECK_NEAR(planted->head_coverage(), 1.0, 1e-12);  // every Artist fact
    CHECK_NEAR(planted->p0, 27.0 / 98.0, 1e-12);        // Artist's share of the 98 occupation facts
    CHECK(planted->p_value < 1e-10);
    CHECK(rule_t3(*planted) != TestOutcome::Negative);
  }
  const Rule* spurious = find("tag(X, z) => kind(X, common)");
  CHECK(spurious != nullptr);
  if (spurious) {
    CHECK_NEAR(spurious->support, 40, 0);
    CHECK_NEAR(spurious->pca_conf(), 0.8, 1e-12);
    CHECK_NEAR(spurious->p0, 0.8, 1e-12);  // common's share of kind
    CHECK(spurious->p_value > 0.3);
    CHECK(!spurious->kept);  // PCA >= 0.5 and support >= 10, but no better than its base rate
  }
  const Rule* inv = find("predecessor(Y, X) => successor(X, Y)");
  CHECK(inv != nullptr && inv->kept && inv->support == 15 && inv->pca_conf() == 1.0);
  const Rule* path = find("birth_place(X, Z) & country(Z, Y) => nationality(X, Y)");
  CHECK(path != nullptr && path->kept && path->support == 12 && path->kind == RuleKind::Path);
  if (path) {  // H99 is born in Town1 but has no nationality: in the body, outside the PCA body
    CHECK_NEAR(path->body, 13, 0);
    CHECK_NEAR(path->pca_body, 12, 0);
    CHECK_NEAR(path->pca_conf(), 1.0, 1e-12);
    CHECK_NEAR(path->std_conf(), 12.0 / 13.0, 1e-12);
  }
  for (const auto& R : rules) CHECK(!R.kept || (R.support >= 10 && R.pca_conf() >= 0.5));

  const RuleSet rs(ix, rules);
  const auto c = rs.candidates("H99", "occupation");
  CHECK(!c.empty() && c[0].o == "Artist");
  if (!c.empty()) CHECK_NEAR(c[0].score, 27.0 / 31.0, 1e-12);
  const auto n = rs.candidates("H99", "nationality");
  CHECK(n.size() == 1 && n[0].o == "Land1");
  const auto succ = rs.candidates("B3", "successor");  // no successor stated for B3; the inverse needs a predecessor of someone
  CHECK(succ.empty());
  // Consequences (T2): if H99 is a painter, the planted rule makes it an artist.
  bool has_artist = false;
  for (const auto& q : rs.consequences("H99", "occupation", "Painter")) has_artist = has_artist || (q.s == "H99" && q.r == "occupation" && q.o == "Artist");
  CHECK(has_artist);
  bool has_succ = false;  // predecessor(Q, P) => successor(P, Q)
  for (const auto& q : rs.consequences("Q", "predecessor", "P")) has_succ = has_succ || (q.s == "P" && q.r == "successor" && q.o == "Q");
  CHECK(has_succ);
  // Noisy-or and the maximum.
  const RuleSet rs_max(ix, rules, false);
  CHECK(!rs_max.candidates("H99", "occupation").empty());

  // G14.
  ContestedSlot slot{"Al Gore", "birth_place", "Washington, D.C.", "Carthage", 1, 2};
  ContestFacts f;
  f.a_contains_b = true;
  CHECK(explain_contest(slot, f).kind == HiddenVar::Granularity);
  ContestFacts d;
  d.is_date = true, d.day_gap = 10;
  CHECK(explain_contest(slot, d).kind == HiddenVar::Calendar);
  d.day_gap = 400;
  CHECK(explain_contest(slot, d).kind == HiddenVar::SourceError);
  ContestFacts tv;
  tv.time_varying = true;
  const auto et = explain_contest(slot, tv);
  CHECK(et.kind == HiddenVar::Time && et.bits == 3.0);
  ContestFacts er;
  er.t4_a = 1, er.t4_b = 0;
  const auto e1 = explain_contest(slot, er);
  CHECK(e1.kind == HiddenVar::SourceError && e1.wrong == 1 && e1.bits == 4.0);
  er.t4_a = 0;
  CHECK(explain_contest(slot, er).wrong == 0);  // no page evidence: b has more sources, so a is wrong
  slot.sources_a = slot.sources_b = 1;
  CHECK(explain_contest(slot, er).wrong == 1);  // a tie: the later value
  CHECK_STR(hidden_var_name(HiddenVar::Granularity), "granularity");
  CHECK_STR(rule_kind_name(RuleKind::Inverse), "inverse");
}

// ---------------------------------------------------------------------------
// S7c: batch rules that state their guarantee (weighted e-BH, conformal selection, null-model p-values), the hybrid
// ---------------------------------------------------------------------------

static void test_s7c() {
  // Weighted e-BH: equal weights are plain e-BH (bh_reject_e) on every random batch; a zero weight is never rejected.
  {
    std::mt19937_64 g(7);
    std::exponential_distribution<double> ex(0.05);
    int same = 0;
    for (int b = 0; b < 200; ++b) {
      std::vector<double> e(40), w(40, 2.5);
      for (double& x : e) x = ex(g);
      same += weighted_e_bh(e, w, 0.1) == bh_reject_e(e, 0.1);
    }
    CHECK(same == 200);
    // m 5, q 0.1: e-BH rejects the k largest with e_(k) >= 50 / k: 100 and 50 (50 >= 25), not 12 (12 < 16.7).
    const std::vector<double> e{100, 50, 12, 1, 0.5};
    const auto r = weighted_e_bh(e, {1, 1, 1, 1, 1}, 0.1);
    CHECK(r[0] && r[1] && !r[2] && !r[3] && !r[4]);
    CHECK(r == bh_reject_e(e, 0.1));
    // Weights 0.5, 0.5, 4, 0, 0 (mean 1): w e = 50, 25, 48, 0, 0; sorted 50, 48, 25 against 50, 25, 16.7: all three.
    const auto rw = weighted_e_bh(e, {0.5, 0.5, 4, 0, 0}, 0.1);
    CHECK(rw[0] && rw[1] && rw[2] && !rw[3] && !rw[4]);
    // Weights are rescaled to mean 1, so their scale does not matter; all-zero weights reject nothing.
    CHECK(weighted_e_bh(e, {5, 5, 40, 0, 0}, 0.1) == rw);
    const auto none = weighted_e_bh(e, {0, 0, 0, 0, 0}, 0.1);
    CHECK(!none[0] && !none[1] && !none[2]);
  }
  // Conformal p-values: (1 + #{false calibration hypotheses with score >= s}) / (n + 1), n counting the true ones too.
  {
    const std::vector<double> cs{0.1, 0.5, 0.9, 0.95};
    const std::vector<bool> ct{false, false, true, false};
    const auto p = conformal_selection_pvalues(cs, ct, {0.92, 0.99, 0.5, 0.9});
    CHECK_NEAR(p[0], 2.0 / 5, 1e-12);  // 0.95 only
    CHECK_NEAR(p[1], 1.0 / 5, 1e-12);  // none
    CHECK_NEAR(p[2], 3.0 / 5, 1e-12);  // 0.5 (a tie counts) and 0.95
    CHECK_NEAR(p[3], 2.0 / 5, 1e-12);  // the true 0.9 is not counted
  }
  // Conformal selection's finite-sample guarantee, by simulation: scores N(1.5, 1) when true, N(0, 1) when false, a
  // calibration set and a batch drawn the same way. Mixed (30% true) and all-null batches: the realised FDR stays at or
  // below q = 0.1 (3 standard errors).
  for (const double pi1 : {0.3, 0.0}) {
    std::mt19937_64 g(pi1 > 0 ? 11 : 13);
    std::normal_distribution<double> n01(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double s = 0, s2 = 0, sel = 0;
    const int reps = 2000;
    for (int r = 0; r < reps; ++r) {
      std::vector<double> cs(150), bs(150);
      std::vector<bool> ct(150), bt(150);
      for (int i = 0; i < 150; ++i) {
        ct[i] = u(g) < pi1, bt[i] = u(g) < pi1;
        cs[i] = n01(g) + (ct[i] ? 1.5 : 0.0), bs[i] = n01(g) + (bt[i] ? 1.5 : 0.0);
      }
      const auto pick = conformal_select(cs, ct, bs, 0.1);
      double R = 0, V = 0;
      for (int i = 0; i < 150; ++i)
        if (pick[i]) R += 1, V += bt[i] ? 0 : 1;
      const double fdp = R > 0 ? V / R : 0.0;
      s += fdp, s2 += fdp * fdp, sel += R;
    }
    const double fdr = s / reps, se = std::sqrt(std::max(0.0, (s2 / reps - fdr * fdr) / (reps - 1)));
    CHECK(fdr <= 0.1 + 3 * se);
    if (pi1 > 0) CHECK(sel / reps > 5.0);  // and it does select
  }
  // Null-model p-values: one test, f = (0.1, 0.8, 0.1), t = (0.8, 0.1, 0.1); not independent, so the loop never stops early.
  {
    HypTest t;
    t.independent = false;
    t.likelihoods.t[0] = 0.8, t.likelihoods.t[1] = 0.1, t.likelihoods.t[2] = 0.1;
    t.likelihoods.f[0] = 0.1, t.likelihoods.f[1] = 0.8, t.likelihoods.f[2] = 0.1;
    const std::vector<HypTest> plan{t};
    CHECK_NEAR(null_model_pvalue(plan, 0.5, 8.0), 0.1, 1e-12);     // only "positive" reaches e 8
    CHECK_NEAR(null_model_pvalue(plan, 0.5, 1.0), 0.2, 1e-12);     // positive or inconclusive
    CHECK_NEAR(null_model_pvalue(plan, 0.5, 0.0), 1.0, 1e-12);     // every path
    CHECK_NEAR(null_model_pvalue(plan, 0.5, 100.0), 0.0, 1e-12);   // none
    HypTest two_way = t;
    two_way.likelihoods = OutcomeLikelihoods{};
    CHECK_NEAR(null_model_pvalue({two_way}, 0.5, 8.0), 1.0, 1e-12);  // no null model: 1
    // The loop's stopping counts: two independent copies of t from prior 0.7. A first "positive" reaches 0.949 and stops at
    // e 8, so e 64 is never reached; e >= 8 is positive (0.1) or inconclusive then positive (0.01). Without stopping the
    // answers would be 0.03 and 0.01.
    HypTest ti = t;
    ti.independent = true;
    CHECK_NEAR(null_model_pvalue({ti, ti}, 0.7, 8.0), 0.11, 1e-12);
    CHECK_NEAR(null_model_pvalue({ti, ti}, 0.7, 64.0), 0.0, 1e-12);
    // Super-uniform under the model, with the loop's stopping: three tests, the second independent; draws from the null.
    HypTest a = t, c = t;
    a.independent = true;
    c.likelihoods.t[0] = 0.6, c.likelihoods.t[1] = 0.2, c.likelihoods.t[2] = 0.2;
    c.likelihoods.f[0] = 0.2, c.likelihoods.f[1] = 0.5, c.likelihoods.f[2] = 0.3;
    const std::vector<HypTest> p3{t, a, c};
    std::mt19937_64 g(17);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const HypothesisConfig cfg;
    int below05 = 0, below20 = 0;
    const int reps = 20000;
    for (int r = 0; r < reps; ++r) {
      double post = 0.7, e = 1;
      bool ind = false;
      for (const HypTest& x : p3) {
        const double v = u(g);
        const int o = v < x.likelihoods.f[0] ? 0 : v < x.likelihoods.f[0] + x.likelihoods.f[1] ? 1 : 2;
        const TestOutcome out = o == 0 ? TestOutcome::Positive : o == 1 ? TestOutcome::Negative : TestOutcome::Inconclusive;
        const double bf = bayes_factor(out, x);
        post = update_posterior(post, bf), e *= bf, ind = ind || (x.independent && o != 2);
        if ((post > cfg.supported_p && ind) || post < cfg.refuted_p) break;
      }
      const double p = null_model_pvalue(p3, 0.7, e, cfg);
      below05 += p <= 0.05, below20 += p <= 0.20;
    }
    CHECK(below05 <= 0.05 * reps + 3 * std::sqrt(0.05 * reps));
    CHECK(below20 <= 0.20 * reps + 3 * std::sqrt(0.20 * reps));
  }
  // The hybrid's tiers: kept and supported is Known, supported only is Believed, never Known without support.
  {
    const auto t = hybrid_tiers({true, true, false, false}, {true, false, true, false});
    CHECK(t[0] == 2 && t[1] == 1 && t[2] == 0 && t[3] == 0);
  }
  // The chosen rule (K4 as the Known tier), planted cases. One independent test, t = (0.8, 0.1, 0.1), f = (0.01, 0.89, 0.1):
  // "positive" from prior 0.7 is supported at e 80 with null p 0.01. In a batch of 4 (2 supported) BH's bar for 2 is
  // 0.05, so both are Known; in a batch of 100 it is 0.002, so both stay Believed. Unsupported is never Known.
  {
    HypTest k;
    k.independent = true;
    k.likelihoods.t[0] = 0.8, k.likelihoods.t[1] = 0.1, k.likelihoods.t[2] = 0.1;
    k.likelihoods.f[0] = 0.01, k.likelihoods.f[1] = 0.89, k.likelihoods.f[2] = 0.1;
    CHECK_NEAR(null_model_pvalue({k}, 0.7, 80.0), 0.01, 1e-12);
    const std::vector<std::vector<HypTest>> plans4(4, std::vector<HypTest>{k});
    const auto t4 = hybrid_null_model_tiers(plans4, {0.7, 0.7, 0.7, 0.7}, {80, 80, 80, 1}, {true, true, false, false});
    CHECK(t4.size() == 4 && t4[0] == 2 && t4[1] == 2 && t4[2] == 0 && t4[3] == 0);
    std::vector<double> e100(100, 1.0);
    std::vector<bool> s100(100, false);
    e100[0] = e100[1] = 80, s100[0] = s100[1] = true;
    const auto t100 = hybrid_null_model_tiers(std::vector<std::vector<HypTest>>(100, {k}), std::vector<double>(100, 0.7), e100, s100);
    CHECK(t100[0] == 1 && t100[1] == 1 && t100[2] == 0);
    // A plan without three-way likelihoods has no null model: Believed at most.
    HypTest two_way = k;
    two_way.likelihoods = OutcomeLikelihoods{};
    const auto tw = hybrid_null_model_tiers({{two_way}, {k}}, {0.7, 0.7}, {80, 80}, {true, true});
    CHECK(tw[0] == 1 && tw[1] == 2);
  }
  // The chosen rule's guarantee, by simulation under its model: batches of 200 hypotheses, priors 0.3, 0.5 or 0.7, the
  // three-test plan above with the loop's stopping, outcomes drawn from t or f. 30% true or all false: the realised FDR
  // among the Known stays at or below q = 0.1 (3 standard errors), and with true members it promotes some.
  {
    HypTest t, a, c;
    t.independent = false, a.independent = true, c.independent = true;
    t.likelihoods.t[0] = 0.8, t.likelihoods.t[1] = 0.1, t.likelihoods.t[2] = 0.1;
    t.likelihoods.f[0] = 0.1, t.likelihoods.f[1] = 0.8, t.likelihoods.f[2] = 0.1;
    a.likelihoods = t.likelihoods;
    c.likelihoods.t[0] = 0.6, c.likelihoods.t[1] = 0.2, c.likelihoods.t[2] = 0.2;
    c.likelihoods.f[0] = 0.05, c.likelihoods.f[1] = 0.65, c.likelihoods.f[2] = 0.3;
    const std::vector<HypTest> plan{t, a, c};
    const HypothesisConfig cfg;
    for (const double pi1 : {0.3, 0.0}) {
      std::mt19937_64 g(pi1 > 0 ? 29 : 31);
      std::uniform_real_distribution<double> u(0.0, 1.0);
      const int reps = 400, m = 200;
      double sum = 0, sum2 = 0, known = 0;
      for (int r = 0; r < reps; ++r) {
        std::vector<double> pri(m), e(m);
        std::vector<bool> sup(m), tru(m);
        for (int i = 0; i < m; ++i) {
          pri[i] = i % 3 == 0 ? 0.3 : i % 3 == 1 ? 0.5 : 0.7;
          tru[i] = u(g) < pi1;
          double post = pri[i], ev = 1;
          bool ind = false;
          for (const HypTest& x : plan) {
            const double* pr = tru[i] ? x.likelihoods.t : x.likelihoods.f;
            const double v = u(g);
            const int o = v < pr[0] ? 0 : v < pr[0] + pr[1] ? 1 : 2;
            const TestOutcome out = o == 0 ? TestOutcome::Positive : o == 1 ? TestOutcome::Negative : TestOutcome::Inconclusive;
            const double bf = bayes_factor(out, x);
            post = update_posterior(post, bf), ev *= bf, ind = ind || (x.independent && o != 2);
            if ((post > cfg.supported_p && ind) || post < cfg.refuted_p) break;
          }
          e[i] = ev, sup[i] = post > cfg.supported_p && ind;
        }
        const auto tiers = hybrid_null_model_tiers(std::vector<std::vector<HypTest>>(m, plan), pri, e, sup, 0.1, cfg);
        int k = 0, bad = 0;
        for (int i = 0; i < m; ++i)
          if (tiers[i] == 2) ++k, bad += !tru[i];
        const double fdp = k ? static_cast<double>(bad) / k : 0.0;
        sum += fdp, sum2 += fdp * fdp, known += k;
      }
      const double fdr = sum / reps, se = std::sqrt(std::max(0.0, sum2 / reps - fdr * fdr) / reps);
      CHECK(fdr <= 0.1 + 3 * se);
      if (pi1 > 0) CHECK(known / reps > 5.0);
    }
  }
}

// ---------------------------------------------------------------------------
// S9: possible worlds, T2 through worlds, reductio, fragility, the card; I9 (worlds are never written back)
// ---------------------------------------------------------------------------
static Rule s9_rule(RuleKind k, const std::string& head, const std::string& head_o, const std::string& r1, const std::string& c1,
                    const std::string& r2, bool inv1, bool inv2, double support, double pca_body, bool kept = true) {
  Rule R;
  R.kind = k, R.head = head, R.head_o = head_o, R.r1 = r1, R.c1 = c1, R.r2 = r2, R.inv1 = inv1, R.inv2 = inv2;
  R.support = support, R.pca_body = pca_body, R.body = pca_body, R.kept = kept;
  return R;
}

static void test_worlds_S9() {
  TripleStore st;
  st.add("Town1", "country", "Land1");
  st.add("Town2", "country", "Land2");
  st.add("Bob", "nationality", "Land2");
  st.add("Cat", "birth_place", "Town1");
  st.add("Cat", "nationality", "Land1");
  st.add("Dan", "occupation", "Artist");
  st.add("Fay", "language", "Lang9");
  st.add("Ida", "birth_place", "Town2");
  st.add("Jo", "birth_place", "Town2");
  st.add("Jo", "passport", "Land2");
  const std::vector<Rule> rules = {
      s9_rule(RuleKind::Const, "occupation", "Artist", "occupation", "Painter", "", false, false, 9, 10),       // 0.9
      s9_rule(RuleKind::Const, "field", "Art", "occupation", "Artist", "", false, false, 10, 10),               // 1.0
      s9_rule(RuleKind::Path, "nationality", "", "birth_place", "", "country", false, false, 12, 12),          // 1.0
      s9_rule(RuleKind::Same, "citizenship", "", "nationality", "", "", false, false, 17, 20),                 // 0.85
      s9_rule(RuleKind::Const, "language", "Lang2", "citizenship", "Land2", "", false, false, 8, 10),          // 0.8
      s9_rule(RuleKind::Same, "nationality", "", "passport", "", "", false, false, 9, 10),                    // 0.9
      s9_rule(RuleKind::Inverse, "successor", "", "predecessor", "", "", false, false, 19, 20),               // 0.95
      s9_rule(RuleKind::Inverse, "predecessor", "", "successor", "", "", false, false, 18, 20),               // 0.9
      s9_rule(RuleKind::Const, "field", "Law", "occupation", "Painter", "", false, false, 2, 10, false),      // not kept
  };
  const RuleBook book(rules);
  CHECK(book.rules().size() == 8);  // the rule that was not kept is left out
  const std::uint64_t fp0 = store_fingerprint(st);
  const std::size_t n0 = st.size();

  // Functional slots: nationality and language. A fact conflicts when its slot holds other values (the fact itself aside).
  auto functional = [](const std::string& r) { return r == "nationality" || r == "language"; };
  ConflictChecks cc;
  cc.held = [&](const World& w, const Triple& t, Triple& culprit, std::string& kind) {
    if (!functional(t.r)) return false;
    bool other = false;
    for (const auto& f : w.held_facts_of(t.s)) {
      if (f.r != t.r || f.o == t.o) continue;
      if (!other) culprit = Triple{t.s, t.r, f.o};
      other = true;
    }
    if (other) kind = "functional";
    return other;
  };
  cc.pair = [&](const Triple& a, const Triple& b) { return functional(a.r) && a.r == b.r && a.o != b.o; };

  // The overlay: assumptions are held in the world, retractions hide stored facts, the store is untouched.
  {
    World w(st);
    w.add("Ann", "occupation", "Painter");
    w.retract("Cat", "nationality", "Land1");
    CHECK(w.held(Triple{"Ann", "occupation", "Painter"}));
    CHECK(!w.held(Triple{"Cat", "nationality", "Land1"}));
    CHECK(st.has("Cat", "nationality", "Land1") && !st.has("Ann", "occupation", "Painter"));
    CHECK(w.values("Cat", "nationality").empty());
    CHECK(w.values("Ann", "occupation").size() == 1);
    CHECK(w.subjects_with("occupation", "Painter").size() == 1);
  }

  // Forward chaining, depth and confidence: Painter -> Artist (0.9) -> field Art (0.9 x 1.0).
  {
    World w(st);
    w.add("Ann", "occupation", "Painter");
    const ChainResult r = chain(w, book, ChainConfig{3, 0.8}, cc);
    CHECK(r.derived.size() == 2 && r.confirmed.empty() && r.contradictions.empty() && r.direct.empty());
    if (r.derived.size() == 2) {
      CHECK(r.derived[0].t == (Triple{"Ann", "occupation", "Artist"}) && r.derived[0].depth == 1);
      CHECK_NEAR(r.derived[0].conf, 0.9, 1e-12);
      CHECK(r.derived[1].t == (Triple{"Ann", "field", "Art"}) && r.derived[1].depth == 2);
      CHECK_NEAR(r.derived[1].conf, 0.9, 1e-12);
      CHECK(r.derived[1].premises.size() == 1 && r.derived[1].premises[0] == r.derived[0].t);
    }
    World w1(st);
    w1.add("Ann", "occupation", "Painter");
    CHECK(chain(w1, book, ChainConfig{1, 0.8}, cc).derived.size() == 1);
  }
  // A path, a same and a const rule: depth 3, confidence 1.0 x 0.85 x 0.8. Depth 2 stops one short.
  {
    World w(st);
    w.add("Eve", "birth_place", "Town2");
    const ChainResult r = chain(w, book, ChainConfig{3, 0.8}, cc);
    CHECK(r.derived.size() == 3);
    if (r.derived.size() == 3) {
      CHECK(r.derived[0].t == (Triple{"Eve", "nationality", "Land2"}) && r.derived[0].premises.size() == 2);
      CHECK(r.derived[0].premises.size() == 2 && r.derived[0].premises[1] == (Triple{"Town2", "country", "Land2"}));
      CHECK(r.derived[2].t == (Triple{"Eve", "language", "Lang2"}) && r.derived[2].depth == 3);
      CHECK_NEAR(r.derived[1].conf, 0.85, 1e-12);
      CHECK_NEAR(r.derived[2].conf, 0.85 * 0.8, 1e-12);
    }
    World w2(st);
    w2.add("Eve", "birth_place", "Town2");
    CHECK(chain(w2, book, ChainConfig{2, 0.8}, cc).derived.size() == 2);
    // The path rule from its second atom: a new country for Town1 reaches Cat (born there).
    World w3(st);
    w3.add("Town1", "country", "Land3");
    const ChainResult r3 = chain(w3, book, ChainConfig{1, 0.8}, cc);
    CHECK(r3.derived.size() == 1 && r3.derived[0].t == (Triple{"Cat", "nationality", "Land3"}));
    CHECK(r3.contradictions.size() == 1 && r3.contradictions[0].culprit == (Triple{"Cat", "nationality", "Land1"}));
    // The inverse rule.
    World w4(st);
    w4.add("Q", "predecessor", "P");
    const ChainResult r4 = chain(w4, book, ChainConfig{3, 0.8}, cc);
    CHECK(r4.derived.size() == 1 && r4.derived[0].t == (Triple{"P", "successor", "Q"}));
    // At depth 2 the inverse rule leads back to the assumption: held, so a confirmation, but not the store's.
    CHECK(r4.confirmed.size() == 1 && r4.confirmed[0].t == (Triple{"Q", "predecessor", "P"}) && !r4.confirmed[0].by_store);
    CHECK(t2_world(st, book, Triple{"Q", "predecessor", "P"}, ChainConfig{}, cc) == TestOutcome::Inconclusive);
  }

  // T2 through worlds. Positive: Dan painter -> Dan artist, which the store holds.
  {
    ChainResult r;
    CHECK(t2_world(st, book, Triple{"Dan", "occupation", "Painter"}, ChainConfig{}, cc, &r) == TestOutcome::Positive);
    CHECK(r.confirmed.size() == 1 && r.confirmed[0].t == (Triple{"Dan", "occupation", "Artist"}) && r.derived.empty());
    CHECK(r.confirmed.size() == 1 && r.confirmed[0].by_store);
    // Negative: Bob born in Town1 makes him Land1's, but the store holds Land2 in a functional slot.
    CHECK(t2_world(st, book, Triple{"Bob", "birth_place", "Town1"}, ChainConfig{}, cc, &r) == TestOutcome::Negative);
    CHECK(r.contradictions.size() == 1);
    if (!r.contradictions.empty()) {
      CHECK(r.contradictions[0].fact == (Triple{"Bob", "nationality", "Land1"}));
      CHECK(r.contradictions[0].culprit == (Triple{"Bob", "nationality", "Land2"}));
      CHECK_STR(r.contradictions[0].kind, "functional");
      CHECK(!r.contradictions[0].among_derived);
    }
    // Below the confidence floor a clash is not a contradiction: Fay's derived language (0.68) clashes with Lang9.
    CHECK(t2_world(st, book, Triple{"Fay", "birth_place", "Town2"}, ChainConfig{3, 0.8}, cc, &r) == TestOutcome::Inconclusive);
    CHECK(r.derived.size() == 3 && r.contradictions.empty());
    CHECK(t2_world(st, book, Triple{"Fay", "birth_place", "Town2"}, ChainConfig{3, 0.6}, cc) == TestOutcome::Negative);
    CHECK(t2_world(st, book, Triple{"Fay", "birth_place", "Town2"}, ChainConfig{2, 0.6}, cc) == TestOutcome::Inconclusive);
    // h's own conflict is T1's: reported apart, not a T2 negative.
    CHECK(t2_world(st, book, Triple{"Bob", "nationality", "Land1"}, ChainConfig{}, cc, &r) == TestOutcome::Inconclusive);
    CHECK(r.direct.size() == 1 && r.contradictions.empty());
    // Two derived values in one functional slot.
    World w(st);
    w.add("Hal", "birth_place", "Town1");
    w.add("Hal", "birth_place", "Town2");
    const ChainResult rp = chain(w, book, ChainConfig{1, 0.8}, cc);
    CHECK(rp.contradictions.size() == 1 && rp.contradictions[0].among_derived && rp.contradictions[0].kind == "derived pair");
    // A planted bug would show: with no checks there is nothing to contradict.
    CHECK(t2_world(st, book, Triple{"Bob", "birth_place", "Town1"}, ChainConfig{}, ConflictChecks{}) == TestOutcome::Inconclusive);
  }

  // Reductio: minimal conflicting subsets, supersets pruned.
  const std::vector<Triple> as = {{"Bob", "birth_place", "Town1"}, {"Hal", "birth_place", "Town1"},
                                  {"Hal", "birth_place", "Town2"}, {"Ann", "occupation", "Painter"}};
  const auto mcs = minimal_conflicting_subsets(st, book, as, ChainConfig{}, cc);
  CHECK(mcs.size() == 2);
  if (mcs.size() == 2) {
    CHECK(mcs[0] == std::vector<std::size_t>({0}));
    CHECK(mcs[1] == std::vector<std::size_t>({1, 2}));
  }
  CHECK(minimal_conflicting_subsets(st, book, {{"Ann", "occupation", "Painter"}}, ChainConfig{}, cc).empty());

  // Consequences that disappear when a fact is retracted: Ida's three; Jo keeps his through the passport rule.
  {
    World w(st);
    w.retract("Ida", "birth_place", "Town2");
    const auto v = vanished(w, book, ChainConfig{});
    CHECK(v.size() == 3);
    if (v.size() == 3) CHECK(v[2] == (Triple{"Ida", "language", "Lang2"}));
    World wj(st);
    wj.retract("Jo", "birth_place", "Town2");
    CHECK(vanished(wj, book, ChainConfig{}).empty());
    CHECK(derivable(wj, book, Triple{"Jo", "language", "Lang2"}, 3));
    CHECK(!derivable(wj, book, Triple{"Jo", "language", "Lang2"}, 2));
  }

  // The card: labelled, data, ends with the closing line once.
  {
    World w(st);
    w.add("Bob", "birth_place", "Town1");
    const ChainResult r = chain(w, book, ChainConfig{}, cc);
    WorldCard c;
    c.added = w.assumed_added();
    c.consequences = r.derived;
    c.contradictions = r.contradictions;
    c.minimal_conflicts = {{as[0]}};
    const std::string text = format_world_card(c);
    CHECK(is_labelled(text));
    CHECK(starts_with(text, "HYPOTHETICAL"));
    const std::string close = "Nothing here is stored.\n";
    CHECK(text.size() > close.size() && text.compare(text.size() - close.size(), close.size(), close) == 0);
    CHECK(text.find("Nothing here is stored.") == text.rfind("Nothing here is stored."));
    CHECK(text.find("Minimal conflicting subsets: {Bob birth_place Town1}") != std::string::npos);
    CHECK(text.find("Bob nationality Land1 vs Bob nationality Land2 (functional)") != std::string::npos);
    CHECK(text.find("Tainted:") != std::string::npos);
  }

  // I9: no world writes back. The store is byte-identical after every world above, write_back refuses, and what a world
  // exports is tainted, so answer mode never sees it even with a confident opinion.
  {
    World w(st);
    w.add("Eve", "birth_place", "Town2");
    w.retract("Cat", "nationality", "Land1");
    chain(w, book, ChainConfig{}, cc);
    TripleStore target = st;
    CHECK(!w.write_back(target));
    CHECK(store_fingerprint(target) == fp0 && target.size() == n0);
    CHECK(store_fingerprint(st) == fp0 && st.size() == n0);
    CHECK(!st.has("Eve", "nationality", "Land2") && st.has("Cat", "nationality", "Land1"));
    KnowledgeStore ks;
    ks.add(KnowledgeItem{"store:" + Triple{"Town2", "country", "Land2"}.key(), opinion_from_counts(100, 0), std::nullopt, false, {}});
    const auto items = w.export_items();
    CHECK(items.size() == 4);  // one assumption, three derived facts
    for (auto it : items) {
      CHECK(it.hypothetical);
      it.opinion = opinion_from_counts(100, 0);
      if (!it.premises.empty()) it.hypothetical = false;  // a planted bug: a derived item's flag dropped. Its premises still carry the taint.
      ks.add(it);
    }
    for (const auto& it : items) CHECK(ks.tainted(it.id));
    CHECK(ks.answer_view().size() == 1);  // only the observed store item
    // The fingerprint does catch a change: one more triple, or one changed object.
    TripleStore more = st;
    more.add("Eve", "nationality", "Land2");
    CHECK(store_fingerprint(more) != fp0);
    TripleStore other;
    for (const auto& [s, fs] : st.subjects())
      for (const auto& f : fs) other.add(s, f.r, s == "Bob" ? "Land9" : f.o);
    CHECK(other.size() == n0 && store_fingerprint(other) != fp0);
  }

  // Fragility (7.6). tau = Beta(20, 1)'s mean for every source; W = 2, a = 0.5.
  {
    const double tau = prior_infobox().mean();
    auto src = [&](std::size_t id, double t) { SourceItem s; s.source = id; s.item.tau = t; return s; };
    std::vector<FragFact> F(12);
    for (std::size_t i = 0; i < F.size(); ++i) F[i].s = "S" + std::to_string(i), F[i].r = "r", F[i].o = "o";
    F[0].sources = {src(0, tau)};                       // one source: a single point of failure
    F[1].sources = {src(1, tau), src(2, tau)};          // two sources
    F[2].sources = {src(3, tau)};                       // a source and a strong derivation from fact 0
    F[2].derivations = {FactDerivation{0, {0}, 0.9}};
    F[3].sources = {src(4, tau)};                       // a source and a weak derivation from fact 0
    F[3].derivations = {FactDerivation{0, {0}, 0.5}};
    F[4].sources = {src(5, tau), src(6, tau), src(7, tau)};  // a functional slot, rival fact 5
    F[4].rivals = {5}, F[4].phi = 1.0;
    F[5].sources = {src(8, tau)};
    F[5].rivals = {4}, F[5].phi = 1.0;
    F[6].sources = {src(9, tau)};                       // source 9 holds facts 6-9
    F[6].hard = 1;                                      // a hard violation: contested
    F[7].sources = {src(9, tau)};
    F[8].sources = {src(9, tau)};
    F[9].sources = {src(9, tau), src(10, tau)};
    F[10].sources = {src(11, 0.3)};                     // a weak source and a derivation from fact 7
    F[10].derivations = {FactDerivation{0, {7}, 1.0}};
    F[11].sources = {src(12, tau)};
    const FragilityModel M(F);
    const Retraction none;
    const double r1 = tau, P1 = (r1 + 1.0) / (r1 + 2.0);  // b + a u with W = 2, a = 0.5
    CHECK_NEAR(M.direct(0, none).P, P1, 1e-12);
    CHECK(M.state(0, none) == EpistemicState::Believed);
    CHECK(M.top_source(0) && *M.top_source(0) == 0);
    auto f0 = M.fragility(0);
    CHECK(f0.worlds == 1 && f0.drops == 1 && f0.top_dropped && f0.value() == 1.0);
    auto f1 = M.fragility(1);
    CHECK(f1.worlds == 1 && f1.drops == 0);
    // Fact 2: r = tau + 0.9 P1. Neither world drops it.
    const double r2 = tau + 0.9 * P1;
    CHECK_NEAR(M.full(2, none).P, (r2 + 1.0) / (r2 + 2.0), 1e-12);
    CHECK(M.weakest_premise(2) && *M.weakest_premise(2) == 0);
    auto f2 = M.fragility(2);
    CHECK(f2.worlds == 2 && f2.premise_world && f2.drops == 0);
    // Fact 3: with its source gone only 0.5 P1 is left: u >= 0.8, Ignorant. Half fragile.
    auto f3 = M.fragility(3);
    CHECK(f3.worlds == 2 && f3.drops == 1 && f3.top_dropped && !f3.premise_dropped);
    Retraction r3;
    r3.sources.insert(4);
    CHECK(M.state(3, r3) == EpistemicState::Ignorant);
    // Rivals: fact 4 is believed against its weaker rival; fact 5 is not held (Very low -> Ignorant).
    CHECK(M.state(4, none) == EpistemicState::Believed);
    CHECK(M.state(5, none) == EpistemicState::Ignorant);
    CHECK(M.fragility(5).worlds == 0);
    CHECK_NEAR(M.direct(4, none).s, tau, 1e-12);  // phi x the rival's mass
    Retraction r8;
    r8.sources.insert(8);
    CHECK(M.direct(4, r8).P > M.direct(4, none).P);
    CHECK(M.dropped_by(r8).empty());
    // A hard violation: d >= 0.5, Contested.
    CHECK(M.state(6, none) == EpistemicState::Contested);
    // Blast radius of source 9: facts 7 and 8 (their only source) and fact 10 one level on (its derivation's premise).
    Retraction r9;
    r9.sources.insert(9);
    CHECK(M.dropped_by(r9) == std::vector<std::size_t>({7, 8, 10}));
    CHECK(M.state(9, r9) == EpistemicState::Believed);
    CHECK(M.state(7, r9) == EpistemicState::Vacuous);
    // The drop table.
    CHECK(drops(EpistemicState::Known, EpistemicState::Believed));
    CHECK(!drops(EpistemicState::Believed, EpistemicState::Believed));
    CHECK(drops(EpistemicState::Believed, EpistemicState::Contested));
    CHECK(!drops(EpistemicState::Ignorant, EpistemicState::Vacuous));
  }
  CHECK(store_fingerprint(st) == fp0 && st.size() == n0);  // I9 again, after everything
}

// ---------------------------------------------------------------------------
// S8d: support needs a positive independent result; sources calibrated on generator errors (docs/reports/S8D_PROMOTION.md)
// ---------------------------------------------------------------------------

static void test_decide_S8d() {
  namespace src = cypha::eval::sources;
  // Two source tests with three-way likelihoods: positive BF 30, negative BF 0.2, inconclusive BF 0.65 / 0.74.
  auto hyp = [](const std::string& id) {
    Hypothesis h;
    h.id = id, h.key = "capital|" + id + "|X", h.statement = id + " capital X", h.generator = "G2";
    for (const char* k : {"DB-IB", "GN"}) {
      HypTest t;
      t.kind = std::string("S8d ") + k, t.source = k, t.independent = true, t.disconfirmation = true;
      t.likelihoods.t[0] = 0.30, t.likelihoods.t[1] = 0.05, t.likelihoods.t[2] = 0.65;
      t.likelihoods.f[0] = 0.01, t.likelihoods.f[1] = 0.25, t.likelihoods.f[2] = 0.74;
      h.tests.push_back(t);
    }
    return h;
  };
  HypothesisStore hs;
  auto start = [&](const Hypothesis& h, double p0) {
    CHECK(hs.propose(h));
    CHECK(hs.score(h.id, p0));
    CHECK(hs.transition(h.id, HypState::Queued, "scored"));
    CHECK(hs.transition(h.id, HypState::UnderTest, "test"));
  };
  // 1. Negative independent results only: never supported, even with the posterior above 0.9 (S8c's India capital Mumbai).
  start(hyp("h-neg"), 0.999);
  auto r = hs.record_outcome("h-neg", 0, TestOutcome::Negative);
  CHECK(r.has_value() && *r == HypState::Queued);
  const Hypothesis* hn = hs.find("h-neg");
  CHECK_NEAR(hn->posterior, 199.8 / 200.8, 1e-12);  // odds 999 x 0.2
  CHECK(has_independent_evidence(*hn));               // the negative is independent evidence ...
  CHECK(!has_supporting_independent_evidence(*hn));   // ... but not evidence for h
  CHECK(decide(*hn) == HypState::Queued);
  CHECK(hs.transition("h-neg", HypState::UnderTest, "test"));
  r = hs.record_outcome("h-neg", 1, TestOutcome::Negative);
  CHECK(r.has_value() && *r == HypState::Queued);
  CHECK_NEAR(hs.find("h-neg")->posterior, 39.96 / 40.96, 1e-12);
  // 2. A positive independent result supports.
  start(hyp("h-pos"), 0.5);
  r = hs.record_outcome("h-pos", 0, TestOutcome::Positive);
  CHECK(r.has_value() && *r == HypState::Supported);
  CHECK(has_supporting_independent_evidence(*hs.find("h-pos")));
  // 3. Negative, then positive: supported once the positive comes and the posterior is above 0.9 (odds 9 x 0.2 x 30 = 54).
  start(hyp("h-np"), 0.9);
  r = hs.record_outcome("h-np", 0, TestOutcome::Negative);
  CHECK(r.has_value() && *r == HypState::Queued);
  CHECK(hs.transition("h-np", HypState::UnderTest, "test"));
  r = hs.record_outcome("h-np", 1, TestOutcome::Positive);
  CHECK(r.has_value() && *r == HypState::Supported);
  CHECK_NEAR(hs.find("h-np")->posterior, 54.0 / 55.0, 1e-12);
  const Hypothesis* hnp = hs.find("h-np");
  CHECK(independent_evidence(*hnp, hnp->tests[0]) && !hnp->tests[0].positive);  // the negative still counts for ECS (Oppose)
  // 4. A positive from the generator's own source is not independent (I4).
  Hypothesis g = hyp("h-gen");
  g.generator = "DB-IB";
  start(g, 0.5);
  r = hs.record_outcome("h-gen", 0, TestOutcome::Positive);
  CHECK(r.has_value() && *r == HypState::Queued && hs.find("h-gen")->posterior > 0.9);
  // 5. A positive from a test not marked independent (a source under S8d's calibration bar) never supports.
  Hypothesis w = hyp("h-weak");
  w.tests[0].independent = false;
  start(w, 0.5);
  r = hs.record_outcome("h-weak", 0, TestOutcome::Positive);
  CHECK(r.has_value() && *r == HypState::Queued && hs.find("h-weak")->posterior > 0.9);
  CHECK(hs.transition("h-weak", HypState::UnderTest, "test"));
  r = hs.record_outcome("h-weak", 1, TestOutcome::Positive);  // then an independent one
  CHECK(r.has_value() && *r == HypState::Supported);
  // 6. An inconclusive result never supports.
  start(hyp("h-inc"), 0.95);
  r = hs.record_outcome("h-inc", 0, TestOutcome::Inconclusive);
  CHECK(r.has_value() && *r == HypState::Queued && hs.find("h-inc")->posterior > 0.9);
  CHECK(!has_supporting_independent_evidence(*hs.find("h-inc")));
  // 7. The null model's walk stops as decide does: after a negative it goes on. From 0.999 with e_obs = 6 (negative then
  // positive), P(e >= 6 | false) = P(pos) + P(neg) P(pos) + P(inc) P(pos) = 0.01 + 0.0025 + 0.0074. Before S8d the walk
  // stopped at the negative (posterior 0.995 with "independent evidence"), and the middle path was missing (0.0174).
  const Hypothesis plan_h = hyp("h-plan");
  CHECK_NEAR(null_model_pvalue(plan_h.tests, 0.999, 6.0), 0.0199, 1e-12);
  CHECK_NEAR(null_model_pvalue(plan_h.tests, 0.999, 30.0), 0.01, 1e-12);  // only the first positive reaches 30
  // 8. Calibration helpers (sources.hpp): a cell under 10 graded true or false uses the pooled tally.
  TestTally small, big, pooled;
  small.t_pos = 5, small.t_inc = 4, small.f_neg = 20, small.f_inc = 30;  // 9 true
  big.t_pos = 30, big.t_inc = 106, big.f_pos = 1, big.f_neg = 100, big.f_inc = 164;  // DB-IB-like: 30 of 31 positives true
  pooled = big;
  bool own = true;
  const OutcomeLikelihoods Ls = src::calibrated_likelihoods(small, pooled, 10.0, &own);
  CHECK(!own && Ls.t[0] == likelihoods_from(pooled).t[0] && Ls.f[0] == likelihoods_from(pooled).f[0]);
  const OutcomeLikelihoods Lb = src::calibrated_likelihoods(big, small, 10.0, &own);
  CHECK(own);
  CHECK_NEAR(src::positive_lr(Lb), (31.0 / 139.0) / (2.0 / 268.0), 1e-12);
  CHECK(src::may_carry_promotion(Lb));
  TestTally lead;  // WP-LEAD-like: positive on 25 of 136 true and 18 of 265 false, never negative
  lead.t_pos = 25, lead.t_inc = 111, lead.f_pos = 18, lead.f_inc = 247;
  const OutcomeLikelihoods Ll = likelihoods_from(lead);
  CHECK_NEAR(src::positive_lr(Ll), (26.0 / 139.0) / (19.0 / 268.0), 1e-12);  // 2.64
  CHECK(!src::may_carry_promotion(Ll));
  CHECK(src::positive_lr(OutcomeLikelihoods{}) == 0.0 && !src::may_carry_promotion(OutcomeLikelihoods{}));
}

// ---------------------------------------------------------------------------
// S4b: unknown-unknown signals and their alarm rule (unknown_signals.hpp)
// ---------------------------------------------------------------------------

static void test_signals_S4b() {
  // The conformal p-value, ties randomised by v.
  const std::vector<double> ref = {1, 2, 3, 4};
  CHECK_NEAR(conformal_p(ref, 5.0, 0.5), 0.1, 1e-15);
  CHECK_NEAR(conformal_p(ref, 0.0, 0.5), 0.9, 1e-15);
  CHECK_NEAR(conformal_p(ref, 3.0, 0.0), 0.2, 1e-15);  // one greater, the tie not counted
  CHECK_NEAR(conformal_p(std::vector<double>{1, 1, 1}, 1.0, 0.5), 0.5, 1e-15);
  // All-tied scores (U's many zero scores): p = v, so the event rate stays alpha.
  {
    const std::vector<double> flat(300, 0.0);
    int ev = 0;
    for (int i = 0; i < 20000; ++i) ev += conformal_p(flat, 0.0, signal_uniform(signal_unit_id("t", static_cast<std::size_t>(i)))) < 0.05;
    CHECK_NEAR(ev / 20000.0, 0.05, 0.006);
  }
  // Exchangeable continuous scores: P(p < 0.05) = 0.05.
  {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> N(0.0, 1.0);
    std::vector<double> w(499);
    int ev = 0;
    for (int r = 0; r < 4000; ++r) {
      for (double& x : w) x = N(rng);
      ev += conformal_p(w, N(rng), signal_uniform(static_cast<std::uint64_t>(r))) < 0.05;
    }
    CHECK_NEAR(ev / 4000.0, 0.05, 0.012);
  }
  // Shiryaev–Roberts: E[L] = 1 under p0 (the martingale), the known path of an all-event stream, a flat stream silent.
  SignalRule rule;
  CHECK_NEAR(rule.alpha * (rule.p1 / rule.alpha) + (1 - rule.alpha) * ((1 - rule.p1) / (1 - rule.alpha)), 1.0, 1e-15);
  {
    SRDetector sr(rule);
    int at = 0;
    for (int t = 1; t <= 10 && !at; ++t)
      if (sr.step(true)) at = t;
    CHECK(at == 7);  // R: 5, 30, 155, 780, 3905, 19530, 97655 >= 20000
    CHECK_NEAR(sr.alarm_R, 97655.0, 1e-6);
    CHECK(sr.alarm_units == 7 && sr.alarm_events == 7 && sr.R == 0.0);
    SRDetector flat(rule);
    int alarms = 0;
    for (int t = 0; t < 100000; ++t) alarms += flat.step(false);
    CHECK(alarms == 0);
    const double q = 0.75 / 0.95;
    CHECK_NEAR(flat.R, q / (1 - q), 1e-9);  // the fixed point, 3.75, far below A
  }
  // P(an alarm within L units) <= L / A with independent events at p0: 2,000 streams of 1,000 units, bound 0.05.
  {
    std::mt19937_64 rng(11);
    std::bernoulli_distribution B(0.05);
    int with = 0;
    for (int s = 0; s < 2000; ++s) {
      SRDetector sr(rule);
      bool any = false;
      for (int t = 0; t < 1000; ++t) any |= sr.step(B(rng));
      with += any;
    }
    CHECK(with / 2000.0 <= 0.06);
  }
  // Streams with two strata: stratum 0 ~ N(5, 1), stratum 1 ~ N(0, 1). The unshifted stream stays near alpha and
  // (almost) silent; a planted step change of stratum-1 units at 3 (inside stratum 0's range) is detected fast.
  std::mt19937_64 rng(20261029);
  std::normal_distribution<double> N(0.0, 1.0);
  auto make = [&](std::size_t n, const std::string& set, double shift1) {
    std::vector<SignalUnit> v;
    for (std::size_t i = 0; i < n; ++i) {
      const int k = static_cast<int>(i % 2);
      v.push_back({(k == 0 ? 5.0 : shift1) + N(rng), k, signal_unit_id(set, i)});
    }
    return v;
  };
  const auto C = make(2000, "C", 0.0), S = make(1500, "S", 0.0);
  std::vector<SignalUnit> pool;
  for (std::size_t i = 0; i < 400; ++i) pool.push_back({3.0 + N(rng), 1, signal_unit_id("P", i)});
  for (bool running : {true, false}) {
    const SignalRun run = signal_run(C, S, 2, running, rule);
    CHECK(run.s_scored == S.size() && run.r_before.size() == S.size() && run.s_event.size() == S.size());
    CHECK(running ? run.c_scored == C.size() - 2 * rule.min_ref : run.c_scored == 0);  // the warm-up: 100 per stratum
    const double rate = static_cast<double>(run.s_events) / static_cast<double>(run.s_scored);
    CHECK(rate > 0.03 && rate < 0.07);
    std::size_t s_alarms = 0;
    for (const auto& a : run.alarms) s_alarms += a.in_test;
    CHECK(s_alarms <= 1);
    std::vector<int> delays, delays_half, none;
    std::mt19937_64 r2(5);
    for (int rep = 0; rep < 100; ++rep) {
      const std::size_t t0 = 100 + static_cast<std::size_t>(rep) * 12;
      delays.push_back(signal_planted(C, S, pool, 2, running, rule, run, t0, 1.0, 200, r2).delay);
      delays_half.push_back(signal_planted(C, S, pool, 2, running, rule, run, t0, 0.5, 200, r2).delay);
      none.push_back(signal_planted(C, S, S, 2, running, rule, run, t0, 1.0, 200, r2).delay);  // "plants" from S itself
    }
    const auto st = signal_delay_stats(delays, 200), sh = signal_delay_stats(delays_half, 200), sn = signal_delay_stats(none, 200);
    CHECK(st.within50 == 100 && st.median >= 6 && st.median <= 12);  // P(event) ~ 0.9: about 7-8 units
    CHECK(sh.within_window == 100 && sh.median > st.median);
    CHECK(sn.within_window <= 10);  // no shift: an alarm within 200 units at most rarely (bound 200 / A = 0.01 each)
    CHECK(signal_perm_alarms_per_k(run, rule, 200, 3) <= 0.1);
  }
  // The running reference a shift meets is the last cap units of each stratum read before it.
  {
    SignalRule r = rule;
    r.cap = 50;
    const SignalRef at = signal_ref_at(C, S, 2, true, r, 10);
    CHECK(at.w[0].size() == 50 && at.w[1].size() == 50);
    CHECK(at.w[0].back() == S[8].score && at.w[1].back() == S[9].score && at.w[1].front() == C[C.size() - 89].score);
    CHECK(signal_ref_at(C, S, 2, false, r, 10).w[0].size() == 1000);  // fixed: all of C's stratum
  }
  // Delay statistics: misses count as window + 1.
  {
    const auto s = signal_delay_stats({1, 10, 0, 60}, 200);
    CHECK(s.reps == 4 && s.within50 == 2 && s.within_window == 3);
    CHECK_NEAR(s.median, 60.0, 0);
    CHECK_NEAR(s.p90, 201.0, 0);
  }
  // U's score: -ln P(X <= x), X ~ Binomial(n, q).
  CHECK_NEAR(binom_lower_tail_score(10, 0, 0.1), -10 * std::log(0.9), 1e-12);
  CHECK_NEAR(binom_lower_tail_score(3, 1, 0.5), std::log(2.0), 1e-12);
  CHECK_NEAR(binom_lower_tail_score(40, 3, 0.2), -std::log(0.02846209445974458), 1e-10);  // P(X <= 3)
  CHECK_NEAR(binom_lower_tail_score(5, 5, 0.3), 0.0, 0);
  CHECK_NEAR(binom_lower_tail_score(5, 0, 0.0), 0.0, 0);
  CHECK(binom_lower_tail_score(20, 0, 0.3) > binom_lower_tail_score(10, 0, 0.3));
  // The page types the store's reader skips (pool L).
  CHECK(signal_store_skips("List of cities in Japan") && signal_store_skips("Mercury (disambiguation)"));
  CHECK(!signal_store_skips("Listening") && !signal_store_skips("Lists of rivers") && !signal_store_skips("Aristotle"));
  CHECK(signal_unit_id("C", 1) != signal_unit_id("S", 1) && signal_unit_id("C", 1) == signal_unit_id("C", 1));
}

// ---------------------------------------------------------------------------
// S8c: evidence sources and their reliability (sources.hpp)
// ---------------------------------------------------------------------------

static void test_sources_S8c() {
  namespace src = cypha::eval::sources;
  // Names: accents folded, "(...)" and a comma tail dropped, "the" and " language" / " music" removed.
  CHECK_STR(src::norm_name("São Paulo"), "sao paulo");
  CHECK_STR(src::norm_name("Chicago, Illinois"), "chicago");
  CHECK_STR(src::norm_name("The Beatles (band)"), "beatles");
  CHECK_STR(src::norm_name("English language"), src::norm_name("English"));
  CHECK_STR(src::norm_name("Rock music"), "rock");
  CHECK_STR(src::fold_lower("Ærøskøbing Łódź"), "aroskobing lodz");  // a ligature folds to one letter
  // Values: years and days.
  const src::Value d = src::make_value("1879-03-14"), y = src::make_value("1879"), bc = src::make_value("-490");
  CHECK(d.year == 1879 && d.day == "1879-03-14" && y.year == 1879 && y.day.empty() && bc.year == -490);
  CHECK(src::make_value("Paris").year == -100000);
  CHECK(src::same_value("birth_year", d, y) && src::same_value("birth_date", d, src::make_value("1879-03-14")));
  CHECK(!src::same_value("birth_date", d, src::make_value("1879-03-15")) && src::same_value("birth_date", d, y));
  CHECK(src::same_value("capital", src::make_value("New Delhi", "Q987"), src::make_value("Delhi, India", "Q987")));
  CHECK(!src::same_value("capital", src::make_value("Delhi", "Q1353"), src::make_value("New Delhi", "Q987")));  // items differ
  CHECK(src::same_value("capital", src::make_value("New Delhi"), src::make_value("new delhi", "Q987")));  // one item: by name
  // A table's three-way answer: positive, negative only for a single-valued relation, inconclusive when not covered.
  src::Table t;
  t.add("India", "capital", "New Delhi", "Q987");
  t.add("India", "language", "Hindi");
  t.add_alias("Bharat", "India");
  CHECK(src::answer(t, "India", "capital", src::make_value("New Delhi")) == TestOutcome::Positive);
  CHECK(src::answer(t, "India", "capital", src::make_value("Mumbai")) == TestOutcome::Negative);
  CHECK(src::answer(t, "India", "language", src::make_value("English")) == TestOutcome::Inconclusive);  // multi-valued
  CHECK(src::answer(t, "India", "currency", src::make_value("Rupee")) == TestOutcome::Inconclusive);
  CHECK(src::answer(t, "Bharat", "capital", src::make_value("Mumbai")) == TestOutcome::Negative);    // through the alias
  CHECK(src::answer(t, "Atlantis", "capital", src::make_value("Mumbai")) == TestOutcome::Inconclusive);
  // Homonyms stay apart: a subject keeps its "(...)" and comma parts (values drop them).
  t.add("Georgia (country)", "capital", "Tbilisi");
  t.add("Georgia (U.S. state)", "capital", "Atlanta");
  t.add("Hanover", "country", "Germany");
  CHECK_STR(src::norm_subject("Hanover, New Hampshire"), "hanover new hampshire");
  CHECK(src::answer(t, "Georgia (U.S. state)", "capital", src::make_value("Tbilisi")) == TestOutcome::Negative);
  CHECK(src::answer(t, "Georgia (country)", "capital", src::make_value("Tbilisi")) == TestOutcome::Positive);
  CHECK(src::answer(t, "Hanover, New Hampshire", "country", src::make_value("United States")) == TestOutcome::Inconclusive);
  // The lead paragraph: positive when named on word boundaries, never negative.
  src::LeadText lt;
  lt.add("Albert Einstein", "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist.");
  CHECK(lt.answer("Albert Einstein", "birth_year", src::make_value("1879")) == TestOutcome::Positive);
  CHECK(lt.answer("Albert Einstein", "death_year", src::make_value("1956")) == TestOutcome::Inconclusive);
  CHECK(lt.answer("Albert Einstein", "occupation", src::make_value("Physicist")) == TestOutcome::Positive);
  CHECK(lt.answer("Albert Einstein", "occupation", src::make_value("Chemist")) == TestOutcome::Inconclusive);
  CHECK(lt.answer("Albert Einstein", "occupation", src::make_value("theoretical physicist")) == TestOutcome::Positive);
  CHECK(lt.answer("Albert Einstein", "nationality", src::make_value("Germ")) == TestOutcome::Inconclusive);  // word boundaries
  CHECK(lt.answer("Nobody", "birth_year", src::make_value("1879")) == TestOutcome::Inconclusive);
  // Dawid-Skene recovers planted three-way likelihoods and the share of true items with no reference. When one source
  // copies another, the fit follows the pair and overrates it (the report's caveat): it is no test of independence.
  std::mt19937_64 rng(20261029);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  const double pt[5][3] = {{0.80, 0.05, 0.15}, {0.60, 0.10, 0.30}, {0.40, 0.00, 0.60}, {0.70, 0.10, 0.20}, {0.50, 0.20, 0.30}};
  const double pf[5][3] = {{0.03, 0.70, 0.27}, {0.05, 0.50, 0.45}, {0.05, 0.00, 0.95}, {0.10, 0.60, 0.30}, {0.04, 0.66, 0.30}};
  auto draw = [&](const double* p) { const double u = U(rng); return u < p[0] ? 0 : u < p[0] + p[1] ? 1 : 2; };
  std::vector<std::vector<int>> Y, Yc;
  for (int i = 0; i < 20000; ++i) {
    const bool truth = U(rng) < 0.3;
    std::vector<int> y(5);
    for (int j = 0; j < 5; ++j) y[j] = draw(truth ? pt[j] : pf[j]);
    Y.push_back(y);
    std::vector<int> yc = y;
    yc[4] = U(rng) < 0.8 ? y[0] : draw(truth ? pt[4] : pf[4]);  // source 4 copies source 0 eight times in ten
    Yc.push_back(yc);
  }
  const src::DSFit m = src::dawid_skene(Y, 5);
  CHECK_NEAR(m.prior, 0.3, 0.02);
  for (int j = 0; j < 5; ++j)
    for (int k = 0; k < 3; ++k) {
      CHECK_NEAR(m.t[j][k], pt[j][k], 0.03);
      CHECK_NEAR(m.f[j][k], pf[j][k], 0.03);
    }
  const std::vector<double> post = src::ds_posterior(m, Y);
  double brier = 0;
  for (std::size_t i = 0; i < post.size(); ++i) brier += post[i] * (1 - post[i]);
  CHECK(brier / static_cast<double>(post.size()) < 0.15);
  const src::DSFit mc = src::dawid_skene(Yc, 5);
  CHECK(mc.t[0][0] > pt[0][0] + 0.05 && mc.f[0][0] < pf[0][0]);  // the copied source looks better than it is
  CHECK(src::outcome_index(TestOutcome::Negative) == 1 && src::single_valued("birth_place") && !src::single_valued("genre"));
}

// ---------------------------------------------------------------------------
// S10: the curiosity scheduler (curiosity.hpp)
// ---------------------------------------------------------------------------

static void test_curiosity_S10() {
  namespace cu = cypha::eval::curiosity;
  // Random numbers: the same seed draws the same; the draws have the right means.
  {
    cu::Rng a(7), b(7), c(8);
    bool same = true, differ = false;
    for (int i = 0; i < 100; ++i) {
      const std::uint64_t x = a.next();
      same = same && x == b.next();
      differ = differ || x != c.next();
    }
    CHECK(same && differ);
    cu::Rng r(11);
    double sb = 0, sg = 0, su = 0, umin = 1, umax = 0;
    bool in_range = true;
    const int n = 40000;
    for (int i = 0; i < n; ++i) {
      sb += r.beta(2.0, 5.0), sg += r.gamma(0.5);
      const double u = r.uniform();
      su += u, umin = std::min(umin, u), umax = std::max(umax, u);
      in_range = in_range && r.below(7) < 7;
    }
    CHECK_NEAR(sb / n, 2.0 / 7.0, 0.006);
    CHECK_NEAR(sg / n, 0.5, 0.02);
    CHECK_NEAR(su / n, 0.5, 0.01);
    CHECK(umin >= 0.0 && umax < 1.0 && in_range && r.below(0) == 0);
    CHECK(cu::mix_seed(1, 2, 3) != cu::mix_seed(1, 2, 4) && cu::mix_seed(1, 2, 3) != cu::mix_seed(2, 1, 3));
  }
  // The value of a pull.
  CHECK_NEAR(cu::certainty_gained(0.3, 0.95), 0.65, 1e-12);
  CHECK(cu::certainty_gained(0.98, 0.95) == 0.0);
  CHECK_NEAR(cu::bits_removed(0.5, 0.95), 1.0 - binary_entropy_bits(0.95), 1e-12);
  CHECK(cu::bits_removed(0.95, 0.5) == 0.0);
  CHECK(cu::pull_reward(false, 1.0, 0.3, 0.95) == 0.0);
  CHECK_NEAR(cu::pull_reward(true, 2.0, 0.3, 0.95), 1.3, 1e-12);

  // A simulated bandit: arm k pays reward 1 with probability p[k] at cost c[k].
  auto simulate = [](const std::vector<cu::Arm>& arms, const cu::SchedulerConfig& cfg, const std::vector<double>& p,
                     const std::vector<double>& c, int pulls, std::uint64_t seed) {
    cu::Scheduler s(arms, cfg, seed);
    cu::Rng env(seed + 99);
    const std::vector<bool> all(arms.size(), true);
    for (int i = 0; i < pulls; ++i) {
      const int a = s.choose(all);
      if (a < 0) break;
      const bool win = env.uniform() < p[static_cast<std::size_t>(a)];
      s.update(static_cast<std::size_t>(a), c[static_cast<std::size_t>(a)], win ? 1.0 : 0.0, win ? 0.5 : 0.0, win);
    }
    return s;
  };
  const std::vector<cu::Arm> two = {{"a", "G1", Tier::Conjecture}, {"b", "G1", Tier::Conjecture}};
  cu::SchedulerConfig ts;
  ts.floors = false, ts.retire = false;
  {  // Thompson finds the better arm
    const cu::Scheduler s = simulate(two, ts, {0.4, 0.1}, {1, 1}, 2000, 5);
    CHECK(s.state(0).pulls / s.pulls() > 0.85);
  }
  {  // ... per unit cost: 0.4 at cost 5 (0.08 per unit) loses to 0.2 at cost 1
    const cu::Scheduler s = simulate(two, ts, {0.4, 0.2}, {5, 1}, 3000, 6);
    CHECK(s.state(1).pulls / s.pulls() > 0.75);
    CHECK_NEAR(s.expected_cost(0), (s.state(0).cost + s.spent() / s.pulls()) / (s.state(0).pulls + 1.0), 1e-12);
  }
  {  // the uniform scheduler gives every arm the same share and never looks at rewards
    cu::SchedulerConfig u;
    u.policy = cu::Policy::Uniform, u.floors = false, u.retire = false;
    const std::vector<cu::Arm> three = {{"a", "G1", Tier::Conjecture}, {"b", "G1", Tier::Conjecture}, {"c", "G2", Tier::Wild}};
    const cu::Scheduler s = simulate(three, u, {0.9, 0.0, 0.0}, {1, 1, 1}, 6000, 7);
    for (std::size_t a = 0; a < 3; ++a) CHECK_NEAR(s.state(a).pulls / 6000.0, 1.0 / 3.0, 0.03);
  }
  {  // floors: a useless generator keeps 2% of the cost spent, the wild tier 5%
    const std::vector<cu::Arm> arms = {{"good", "G1", Tier::Conjecture}, {"bad", "G2", Tier::Conjecture}, {"wild", "G1", Tier::Wild}};
    cu::SchedulerConfig f = ts;
    f.floors = true;
    const cu::Scheduler s = simulate(arms, f, {0.5, 0.0, 0.0}, {1, 1, 1}, 4000, 8);
    const double bad = s.state(1).cost / s.spent(), wild = s.state(2).cost / s.spent();
    CHECK(bad >= 0.019 && bad < 0.04);
    CHECK(wild >= 0.049 && wild < 0.08);
    const cu::Scheduler n = simulate(arms, ts, {0.5, 0.0, 0.0}, {1, 1, 1}, 4000, 8);
    CHECK(n.state(1).cost / n.spent() < 0.015 && n.state(2).cost / n.spent() < 0.015);
  }
  {  // retirement: Wilson upper bound of the promotion rate below 0.05 after at least 200 pulls
    cu::SchedulerConfig r;
    r.policy = cu::Policy::Uniform, r.floors = false, r.retire = true;
    const cu::Scheduler s = simulate(two, r, {0.5, 0.0}, {1, 1}, 2000, 9);
    CHECK(s.state(1).retired && s.state(1).pulls == 200.0 && !s.state(0).retired);
    cu::Scheduler t({{"x", "G1", Tier::Wild}, {"y", "G1", Tier::Wild}, {"z", "G1", Tier::Wild}}, r, 1);
    for (int i = 0; i < 199; ++i) t.update(0, 1, 0, 0, false);
    CHECK(!t.state(0).retired);
    t.update(0, 1, 0, 0, false);
    CHECK(t.state(0).retired);
    for (int i = 0; i < 200; ++i) t.update(1, 1, 0, 0, i % 22 == 0), t.update(2, 1, 0, 0, i % 70 == 0);  // 10 / 200 (upper 0.090) and 3 / 200 (0.043)
    CHECK(!t.state(1).retired && t.state(2).retired);
    CHECK(t.choose({true, true, true}) == 1);
  }
  {  // learning progress: bits per unit cost over the last 10 pulls; the bonus multiplies the score
    cu::SchedulerConfig l = ts;
    l.lp_weight = 1.0;
    cu::Scheduler s({{"a", "G1", Tier::Conjecture}}, l, 3);
    CHECK(s.learning_progress(0) == 0.0 && s.expected_cost(0) == 1.0);
    for (int i = 1; i <= 15; ++i) s.update(0, 2, 0, 2.0 * i, false);
    CHECK_NEAR(s.learning_progress(0), 10.5, 1e-12);
    CHECK_NEAR(s.score(0, 0.2), 0.2 * 0.5 / s.expected_cost(0) * 11.5, 1e-12);  // no promotion yet: reward size 0.5
    CHECK(s.choose({false}) == -1);
  }
  {  // amendment 1: the promotion probability's prior comes from the pool; the reward size is a mean of its own
    cu::Scheduler s({{"a", "G1", Tier::Conjecture}, {"b", "G1", Tier::Conjecture}}, ts, 3);
    CHECK_NEAR(s.pooled_rate(), 0.5, 1e-12);
    for (int i = 0; i < 10; ++i) s.update(0, 1, i == 0 ? 0.6 : i == 1 ? 0.8 : 0.0, 0, i < 2), s.update(1, 1, 5.0, 0, false);
    CHECK_NEAR(s.pooled_rate(), 3.0 / 22.0, 1e-12);
    CHECK_NEAR(s.beta_of(0).first, 2.0 * 3.0 / 22.0 + 2.0, 1e-12);
    CHECK_NEAR(s.beta_of(0).second, 2.0 * 19.0 / 22.0 + 8.0, 1e-12);
    CHECK_NEAR(s.reward_size(0), (1.4 + 0.7) / 3.0, 1e-12);
    CHECK_NEAR(s.reward_size(1), 0.7, 1e-12);  // a reward without a promotion pays nothing
    CHECK(s.state(1).reward == 0.0);
  }
  // Replay on a recorded pool.
  {
    std::vector<cu::PoolItem> pool;
    for (int i = 0; i < 10; ++i) {
      cu::PoolItem it;
      it.arm = 0, it.cost = 2, it.prior = 0.1 * i;
      pool.push_back(it);
    }
    pool[3].supported = true, pool[3].graded = 1, pool[3].prior = 0.3, pool[3].posterior = 0.95;   // +0.65
    pool[5].supported = true, pool[5].graded = 0, pool[5].prior = 0.5, pool[5].posterior = 0.92;   // -0.42
    pool[6].supported = true, pool[6].graded = -1, pool[6].prior = 0.2, pool[6].posterior = 0.97;  // ungraded: 0
    pool[7].graded = 1, pool[7].posterior = 0.6;                                                   // not supported: 0
    for (auto& it : pool) it.reward = cu::pull_reward(it.supported, it.impact, it.prior, it.posterior);
    const std::vector<cu::Arm> one = {{"a", "G1", Tier::Conjecture}};
    cu::SchedulerConfig u;
    u.policy = cu::Policy::Uniform, u.floors = false, u.retire = false;
    const auto ord = cu::arm_orders(pool, 1, cu::WithinArm::Random, 4);
    std::vector<std::size_t> sorted = ord[0];
    std::sort(sorted.begin(), sorted.end());
    bool perm = sorted.size() == 10;
    for (std::size_t i = 0; perm && i < 10; ++i) perm = sorted[i] == i;
    CHECK(perm);
    const cu::ReplayResult all = cu::replay(pool, one, ord, u, 1e9, 1);
    CHECK(all.pulls == 10.0 && all.cost == 20.0 && all.promoted == 3.0 && all.graded == 2.0 && all.graded_false == 1.0);
    CHECK_NEAR(all.value, 0.65 - 0.42, 1e-12);
    CHECK_NEAR(all.brier, 0.05 * 0.05 + 0.92 * 0.92, 1e-12);
    const cu::ReplayResult part = cu::replay(pool, one, ord, u, 5.0, 1);
    CHECK(part.pulls == 3.0 && part.cost == 6.0);  // the pull that crosses the budget completes
    // orders: P0 first, demand before everything; with equal priors P0 first is the random order (common random numbers)
    pool[1].demand = 1;
    const auto pf = cu::arm_orders(pool, 1, cu::WithinArm::PriorFirst, 4)[0];
    CHECK(pf[0] == 1 && pf[1] == 9 && pf[2] == 8);
    std::vector<cu::PoolItem> flat = pool;
    for (auto& it : flat) it.prior = 0.5, it.demand = 0;
    CHECK(cu::arm_orders(flat, 1, cu::WithinArm::PriorFirst, 4)[0] == cu::arm_orders(flat, 1, cu::WithinArm::Random, 4)[0]);
    CHECK(cu::arm_orders(flat, 1, cu::WithinArm::Random, 4)[0] != cu::arm_orders(flat, 1, cu::WithinArm::Random, 5)[0]);
  }
  {  // a stratified bootstrap keeps each arm's size; at the whole budget with retirement off, U and Thompson tie
    std::vector<cu::PoolItem> pool;
    cu::Rng g(21);
    for (int i = 0; i < 300; ++i) {
      cu::PoolItem it;
      it.arm = i % 3, it.cost = 1 + static_cast<double>(g.below(3)), it.prior = g.uniform();
      it.supported = g.uniform() < (it.arm == 0 ? 0.3 : 0.02);
      it.posterior = it.supported ? 0.95 : 0.02, it.graded = g.uniform() < 0.8 ? (g.uniform() < 0.9 ? 1 : 0) : -1;
      it.reward = cu::pull_reward(it.supported, 1.0, it.prior, it.posterior);
      pool.push_back(it);
    }
    const auto bs = cu::stratified_bootstrap(pool, 3, 9);
    int n[3] = {0, 0, 0};
    for (const auto& it : bs) ++n[it.arm];
    CHECK(bs.size() == 300 && n[0] == 100 && n[1] == 100 && n[2] == 100);
    const std::vector<cu::Arm> arms = {{"a", "G1", Tier::Conjecture}, {"b", "G1", Tier::Speculation}, {"c", "G2", Tier::Wild}};
    double tot = 0;
    for (const auto& it : pool) tot += it.cost;
    cu::SchedulerConfig u, t;
    u.policy = cu::Policy::Uniform, u.floors = false, u.retire = false;
    t.retire = false;
    const auto ord = cu::arm_orders(pool, 3, cu::WithinArm::Random, 2);
    const cu::ReplayResult ru = cu::replay(pool, arms, ord, u, tot, 3), rt = cu::replay(pool, arms, ord, t, tot, 4);
    CHECK(ru.pulls == 300.0 && rt.pulls == 300.0 && ru.cost == rt.cost);
    CHECK_NEAR(ru.value, rt.value, 1e-9);
    // at a quarter of the budget Thompson finds the arm that promotes
    const cu::ReplayResult qu = cu::replay(pool, arms, ord, u, 0.25 * tot, 3), qt = cu::replay(pool, arms, ord, t, 0.25 * tot, 4);
    CHECK(qt.arm_pulls[0] > qu.arm_pulls[0] && qt.promoted > qu.promoted);
  }
}

// ---------------------------------------------------------------------------
// S11a: G9 concept formation and G11 open relations (gen_concepts.hpp)
// ---------------------------------------------------------------------------

static void test_concepts_S11a() {
  namespace cc = cypha::eval::concepts;
  // LCM against brute force: every closed set with an extent >= minsup and a non-empty intent, each exactly once.
  for (int seed = 0; seed < 6; ++seed) {
    std::mt19937_64 rng(20261103 + seed);
    std::vector<cc::Fact3> facts;
    for (int e = 0; e < 30; ++e)
      for (int a = 0; a < 8; ++a)
        if (rng() % 100 < 35) facts.push_back({"e" + std::to_string(100 + e), a < 4 ? "r0" : "r1", "v" + std::to_string(a)});
    const cc::FormalContext C = cc::build_context(facts, 1.0);
    const double minsup = 3.0;
    const auto cs = cc::closed_concepts(C, minsup);
    const int m = static_cast<int>(C.attr_r.size());
    std::vector<std::vector<int>> brute;
    for (int mask = 1; mask < (1 << m); ++mask) {
      std::vector<int> I, E;
      for (int a = 0; a < m; ++a)
        if (mask >> a & 1) I.push_back(a);
      for (std::size_t e = 0; e < C.entities.size(); ++e) {
        bool all = true;
        for (int a : I) all = all && std::binary_search(C.attrs[e].begin(), C.attrs[e].end(), a);
        if (all) E.push_back(static_cast<int>(e));
      }
      if (static_cast<double>(E.size()) >= minsup && cc::closure_of(C, E) == I) brute.push_back(I);
    }
    std::vector<std::vector<int>> got;
    bool ext_ok = true;
    for (const auto& c : cs) {
      got.push_back(c.intent);
      ext_ok = ext_ok && cc::closure_of(C, c.extent) == c.intent && static_cast<double>(c.extent.size()) >= minsup;
    }
    std::sort(brute.begin(), brute.end());
    std::sort(got.begin(), got.end());
    CHECK(got == brute);  // no closed set missed, none twice
    CHECK(ext_ok);
    CHECK(!brute.empty());
  }
  // the support guard and the stop
  {
    std::vector<cc::Fact3> f = {{"a", "r", "x"}, {"b", "r", "x"}, {"c", "r", "y"}};
    const cc::FormalContext C = cc::build_context(f, 2.0);
    CHECK(C.attr_r.size() == 1 && C.entities.size() == 2 && C.attribute("r", "x") == 0 && C.attribute("r", "y") == -1);
    bool trunc = false;
    std::vector<cc::Fact3> g;
    for (int e = 0; e < 12; ++e)
      for (int a = 0; a < 6; ++a)
        if ((e + a) % 3 != 0) g.push_back({"e" + std::to_string(e), "r", "v" + std::to_string(a)});
    const auto all = cc::closed_concepts(cc::build_context(g, 1.0), 1.0);
    const auto cut = cc::closed_concepts(cc::build_context(g, 1.0), 1.0, 3, &trunc);
    CHECK(all.size() > 3 && cut.size() == 3 && trunc);
  }
  // Jaccard counts members outside the context in the union only
  CHECK_NEAR(cc::jaccard({1, 2, 3}, {2, 3}, 4.0), 2.0 / 5.0, 1e-12);
  CHECK_NEAR(cc::jaccard({1, 2}, {1, 2}), 1.0, 1e-12);
  CHECK_NEAR(cc::jaccard({}, {}, 0.0), 0.0, 1e-12);

  // Candidate classes and class rules on a built example. AB: job x and nat y (30, 29 of them genre z, the other one
  // genre w; 15 lang q, 15 lang p); AB2: job x and nat y with no genre at all (5, lang p); A: job x only (20, 4 genre z);
  // B: nat y only (20, 8 genre z). Everyone in A and B holds a genre and a lang.
  std::vector<cc::Fact3> f;
  auto add = [&](const std::string& e, bool job, bool nat, int genre, bool q) {  // genre: 1 z, 0 w, -1 none
    if (job) f.push_back({e, "job", "x"});
    if (nat) f.push_back({e, "nat", "y"});
    if (genre >= 0) f.push_back({e, "genre", genre ? "z" : "w"});
    f.push_back({e, "lang", q ? "q" : "p"});
  };
  for (int i = 0; i < 30; ++i) add("ab" + std::to_string(i), true, true, i < 29 ? 1 : 0, i < 15);
  for (int i = 0; i < 5; ++i) add("abn" + std::to_string(i), true, true, -1, false);
  for (int i = 0; i < 20; ++i) add("a" + std::to_string(i), true, false, i < 4 ? 1 : 0, i < 10);
  for (int i = 0; i < 20; ++i) add("b" + std::to_string(i), false, true, i < 8 ? 1 : 0, i < 10);
  const cc::FormalContext C = cc::build_context(f, 3.0);
  CHECK(C.entities.size() == 75 && C.attr_r.size() == 6);
  const auto cs = cc::closed_concepts(C, 3.0);
  const auto cand = cc::candidate_classes(C, cs, 5.0, 0.9);
  const int job = C.attribute("job", "x"), nat = C.attribute("nat", "y"), z = C.attribute("genre", "z"), lp = C.attribute("lang", "p");
  std::size_t jn = cs.size();
  for (std::size_t k = 0; k < cs.size(); ++k)
    if (cs[k].intent == std::vector<int>{std::min(job, nat), std::max(job, nat)}) jn = k;
  CHECK(jn < cs.size() && cs[jn].extent.size() == 35);
  CHECK(std::find(cand.begin(), cand.end(), jn) != cand.end());
  for (std::size_t k : cand) CHECK(cs[k].intent.size() >= 2);
  // a conjunction whose extent a single attribute already names is not a candidate
  {
    std::vector<cc::Fact3> h;
    for (int i = 0; i < 12; ++i) h.push_back({"e" + std::to_string(i), "r", "u"}), h.push_back({"e" + std::to_string(i), "s", "t"});
    h.push_back({"e12", "r", "u"}), h.push_back({"e13", "s", "t"});  // r = u and s = t 13 each, the pair 12: Jaccard 0.92
    const cc::FormalContext H = cc::build_context(h, 1.0);
    const auto hc = cc::closed_concepts(H, 10.0);
    CHECK(cc::candidate_classes(H, hc, 10.0, 0.9).empty());
    CHECK(cc::candidate_classes(H, hc, 10.0, 0.95).size() == 1);
  }
  cc::ClassRuleConfig cfg;
  cfg.min_support = 5.0;
  const auto rules = cc::class_rules(C, cs, cand, cfg, [](const std::string& e) { return e.size() >= 3 && e[0] == 'a' && e[1] == 'b' && (e.back() - '0') % 2 == 0; });
  const cc::ClassRule* rz = nullptr;
  const cc::ClassRule* rp = nullptr;
  for (const auto& R : rules) {
    if (R.cls == jn && R.head == z) rz = &R;
    if (R.cls == jn && R.head == lp) rp = &R;
  }
  CHECK(rz != nullptr && rp != nullptr);
  if (rz && rp) {
    CHECK(rz->support == 29.0 && rz->pca_body == 30.0);   // AB2 holds no genre: outside the PCA body
    CHECK_NEAR(rz->pca_conf(), 29.0 / 30.0, 1e-12);
    CHECK_NEAR(rz->baseline, 37.0 / 50.0, 1e-12);         // the better single attribute (nat y), not job x's 33/50
    CHECK_NEAR(rz->p_value, binomial_upper_tail(29.0, 30.0, 0.74), 1e-12);
    CHECK(rz->kept);
    CHECK_NEAR(cc::single_pca(C, job, z), 33.0 / 50.0, 1e-12);
    CHECK(rp->support == 20.0 && rp->pca_body == 35.0 && !rp->kept);  // 0.571 against 30/55: no gain
    CHECK(rz->hid_support == 15.0 && rz->hid_pca == 15.0);  // hidden: AB and AB2 with an even last digit; AB2 holds no genre
    CHECK(cc::class_rule_t3(*rz) == TestOutcome::Positive);
  }

  // G11: connectors, the phrase filter, clusters and their grade
  CHECK_STR(cc::connector_of("[S] was educated at the [O]"), "S ... educated at O");
  CHECK_STR(cc::connector_of("[S] attended [O]"), "S ... attended O");
  CHECK_STR(cc::connector_of("[O] , the capital of [S]"), "O , the ... S");  // the token next to [O] is ",", not an article
  CHECK_STR(cc::connector_of("[O] the capital of [S]"), "O capital of ... S");
  CHECK_STR(cc::connector_of("[S] the [O]"), "");
  CHECK_STR(cc::connector_of("[S] [O]"), "");
  CHECK(cc::has_word("[S] won the [O]") && !cc::has_word("[S] ( # , [O]") && cc::has_word("[O] ) , also known as [S]"));
  std::vector<std::array<std::string, 3>> recs;
  for (int i = 0; i < 6; ++i) recs.push_back({"[S] attended [O]", "p" + std::to_string(i), "school" + std::to_string(i)});
  recs.push_back({"[S] attended [O]", "p0", "school0"});  // a repeat counts once
  for (int i = 0; i < 4; ++i) recs.push_back({"[S] liked [O]", "p" + std::to_string(i), "x"});
  const auto cl = cc::open_clusters(recs, 5);
  CHECK(cl.size() == 1 && cl[0].key == "[S] attended [O]" && cl[0].pairs.size() == 6);
  auto props = [](const std::string& s, const std::string& o) {
    std::vector<std::string> v;
    if (s != "p5") v.push_back(s == "p4" ? "employer" : "educated_at");
    if (o == "school0") v.push_back("educated_at");  // a property listed twice for one pair counts once
    return v;
  };
  auto gradable = [](const std::string& s) { return s != "p3"; };
  const cc::ClusterGrade g = cc::grade_cluster(cl[0], props, gradable);
  CHECK_STR(g.prop, "educated_at");
  CHECK(g.gradable == 5.0 && g.hits == 3.0);
  const cc::ClusterGrade gf = cc::grade_cluster(cl[0], props, gradable, "employer");
  CHECK(gf.gradable == 5.0 && gf.hits == 1.0);

  // S11d: the rediscovery tally (S11a's gate and the strict count), fresh classes' split, G11's rule for the skip.
  {
    cc::RediscoveryTally t;
    t.add({0.62, 0.70, 0.10, 0.05});  // rediscovered, redundant: not strict
    t.add({0.55, 0.20, 0.55, 0.00});  // rediscovered and strict, also by a kept class
    t.add({0.50, 0.49, 0.00, 0.50});  // at the bar: rediscovered and strict; the null reaches the bar too
    t.add({0.49, 0.90, 0.00, 0.10});  // just under the bar, redundant
    CHECK(t.n == 4.0 && t.red == 3.0 && t.strict == 2.0 && t.redun == 2.0 && t.kept == 1.0 && t.null == 1.0);
    CHECK(!t.gate_met() && !t.strict_met());  // 3 of 4 against 1 of 4: the intervals overlap
    cc::RediscoveryTally big;
    for (int i = 0; i < 40; ++i) big.add({i < 30 ? 0.6 : 0.1, i < 25 ? 0.6 : 0.1, 0.0, 0.0});
    CHECK(big.red == 30.0 && big.strict == 5.0 && big.null == 0.0);
    CHECK(big.gate_met());                // 30 / 40 against 0 / 40
    CHECK(!big.strict_met());             // 5 / 40: lower bound 0.055 < the null's upper bound 0.088
    cc::RediscoveryTally none;
    CHECK(!none.gate_met() && !none.strict_met());
    CHECK(cc::fresh_class_is_dev(0) && !cc::fresh_class_is_dev(std::uint64_t{1} << 21) && cc::fresh_class_is_dev(std::uint64_t{1} << 19));
    CHECK(cc::adopt_g11_skip(8, 58, 107) && !cc::adopt_g11_skip(7, 90, 100) && !cc::adopt_g11_skip(12, 57, 107));
    CHECK(cc::adopt_g11_skip(9, 60, 100) && !cc::adopt_g11_skip(9, 0, 0));
  }
}

// ---------------------------------------------------------------------------
// S4b-b: a link inside a date candidate is not an entity mention for U (unknown_signals.hpp)
// ---------------------------------------------------------------------------

static void test_date_links_S4bb() {
  using SP = std::vector<std::pair<int, int>>;
  CHECK(span_within(2, 3, SP{{1, 4}}));             // "[[1882]]" inside "January 18 , 1882"
  CHECK(span_within(1, 4, SP{{1, 4}}));             // the link is the whole date
  CHECK(span_within(0, 1, SP{{0, 1}}));             // a lone linked year, its own one-token date
  CHECK(!span_within(0, 2, SP{{1, 4}}));            // overlaps the date but starts before it ("[[War of 1812]]")
  CHECK(!span_within(3, 5, SP{{1, 4}}));            // overlaps the date but ends after it
  CHECK(!span_within(2, 3, SP{}));                  // no date candidates
  CHECK(span_within(5, 6, SP{{0, 2}, {4, 7}}));     // inside the second of two dates
  CHECK(!span_within(2, 5, SP{{0, 3}, {3, 6}}));    // across two dates: inside neither
  CHECK(!span_within(7, 8, SP{{0, 2}, {4, 7}}));    // just after a date
}

// ---------------------------------------------------------------------------
// S11c: G13 anomalies, G3 abduction, G4 and G5 analogy (gen_analogy.hpp)
// ---------------------------------------------------------------------------

static void test_analogy_S11c() {
  namespace an = cypha::eval::analogy;
  // the empirical tail, add-one
  {
    an::EmpiricalTail t;
    for (int v = 1; v <= 99; ++v) t.add(v);
    CHECK_NEAR(t.upper_p(99), 2.0 / 100.0, 1e-12);
    CHECK_NEAR(t.upper_p(100), 1.0 / 100.0, 1e-12);
    CHECK_NEAR(t.upper_p(0), 1.0, 1e-12);
  }
  const an::YearOf year_of = [](const std::string& v, int& y) {
    if (v.empty()) return false;
    const bool bc = v.size() > 3 && v.compare(v.size() - 3, 3, " BC") == 0;
    const std::string d = v.substr(0, bc ? v.size() - 3 : std::min<std::size_t>(4, v.size()));
    for (char c : d)
      if (c < '0' || c > '9') return false;
    y = std::stoi(d) * (bc ? -1 : 1);
    return y != 0;
  };
  // ---- G13: one store with each kind
  TripleStore st;
  st.add("A", "birth_year", "1900"), st.add("A", "death_year", "1880");                // A1
  st.add("B", "birth_date", "1850-01-01"), st.add("B", "birth_year", "1851");          // A3
  st.add("Aristotle", "birth_year", "384"), st.add("Aristotle", "death_year", "322 BC");  // A1, a lost BC sign
  st.add("Texas", "capital", "Austin"), st.add("Austin", "country", "United States"), st.add("Texas", "located_in", "United States");  // A4, a state
  st.add("Narnia", "capital", "Paris"), st.add("Paris", "country", "France");          // A4
  st.add("Org", "founder", "P"), st.add("Org", "founded", "1900"), st.add("P", "birth_year", "1950");  // A5
  for (int i = 0; i < 40; ++i) {
    const std::string s = "L" + std::to_string(i);
    st.add(s, "birth_year", "1800"), st.add(s, "death_year", std::to_string(1860 + i % 20));
  }
  st.add("Old", "birth_year", "1700"), st.add("Old", "death_year", "1850");            // A2: 150 years
  st.add("Mid", "birth_year", "1800"), st.add("Mid", "death_year", "1880");            // 80 years: not in the tail
  for (int i = 0; i < 40; ++i) {
    const std::string a = "Sa" + std::to_string(i), b = "Sb" + std::to_string(i);
    st.add(a, "birth_year", "1900"), st.add(b, "birth_year", std::to_string(1900 + i)), st.add(a, "spouse", b);
  }
  st.add("Sx", "birth_year", "1900"), st.add("Sy", "birth_year", "1970"), st.add("Sx", "spouse", "Sy");  // a 70-year gap
  const auto A = an::find_anomalies(st, year_of);
  std::map<std::string, int> kinds;
  for (const auto& a : A) ++kinds[an::anomaly_kind_name(a.kind)];
  CHECK(kinds["A1_death_before_birth"] == 2 && kinds["A3_year_mismatch"] == 1 && kinds["A4_capital_elsewhere"] == 2 &&
        kinds["A5_founder_timing"] == 1 && kinds["A2_lifespan_tail"] == 1);
  CHECK(kinds["A6_spouse_gap"] == 0);  // 41 gaps: add-one p cannot reach 0.005
  {
    an::AnomalyConfig cfg;
    cfg.tail_p = 0.05;
    int n6 = 0;
    for (const auto& a : an::find_anomalies(st, year_of, cfg)) {
      if (a.kind != an::AnomalyKind::SpouseGap) continue;
      ++n6;
      CHECK(a.facts[0].o == "Sy" && a.value == 70 && a.link == 0);
    }
    CHECK(n6 == 1);
  }
  auto find = [&](an::AnomalyKind k, const std::string& s) {
    for (const auto& a : A)
      if (a.kind == k && a.facts[0].s == s) return a;
    return an::Anomaly{};
  };
  const an::Anomaly a1 = find(an::AnomalyKind::DeathBeforeBirth, "A");
  CHECK(a1.facts.size() == 2 && a1.facts[0].r == "birth_year" && a1.facts[1].r == "death_year");
  if (a1.facts.size() == 2) {
    an::AnomalyEvidence ev;
    ev.year_of = year_of;
    auto x = an::explain_anomaly(a1, ev);  // no sources, no page: the registered order names the death
    CHECK(x.verdict == an::Verdict::Error && x.wrong == 1 && x.why == "registered order");
    CHECK_NEAR(x.bits, 4.0, 1e-12);
    CHECK(std::isinf(x.bits_unusual));
    ev.sources = [](const an::FactRef& f) { return f.r == "birth_year" ? 1.0 : 2.0; };  // the birth has fewer sources
    x = an::explain_anomaly(a1, ev);
    CHECK(x.wrong == 0 && x.why == "fewer sources");
    ev.sources = [](const an::FactRef&) { return 1.0; };
    ev.page = [](const an::FactRef& f) { return f.r == "death_year" ? 1 : 0; };  // the page states the death
    x = an::explain_anomaly(a1, ev);
    CHECK(x.wrong == 0 && x.why == "page states the others");
  }
  if (find(an::AnomalyKind::DeathBeforeBirth, "Aristotle").facts.size() == 2 && find(an::AnomalyKind::YearMismatch, "B").facts.size() == 2) {
    an::AnomalyEvidence ev;
    ev.year_of = year_of;
    const auto x = an::explain_anomaly(find(an::AnomalyKind::DeathBeforeBirth, "Aristotle"), ev);
    CHECK(x.verdict == an::Verdict::Qualifier && x.wrong == 0 && x.why == "lost BC sign");
    CHECK_NEAR(x.bits, 1.0, 1e-12);
    const auto y = an::explain_anomaly(find(an::AnomalyKind::YearMismatch, "B"), ev);
    CHECK(y.verdict == an::Verdict::Error && y.wrong == 0);  // the year relation, by the registered order
  }
  {
    an::AnomalyEvidence ev;
    ev.year_of = year_of;
    ev.contains = [&](const std::string& o, const std::string& i) { return st.has(i, "located_in", o); };
    ev.ambiguous = [](const std::string& n) { return n == "Paris"; };
    const auto tx = an::explain_anomaly(find(an::AnomalyKind::CapitalElsewhere, "Texas"), ev);
    CHECK(tx.verdict == an::Verdict::Qualifier && tx.wrong == -1);
    const auto nr = an::explain_anomaly(find(an::AnomalyKind::CapitalElsewhere, "Narnia"), ev);
    CHECK(nr.verdict == an::Verdict::DifferentEntity && nr.wrong == 0);
    CHECK_NEAR(nr.bits, 2.0, 1e-12);
    ev.ambiguous = nullptr;
    const auto ne = an::explain_anomaly(find(an::AnomalyKind::CapitalElsewhere, "Narnia"), ev);
    CHECK(ne.verdict == an::Verdict::Error && ne.wrong == 0);
    const an::Anomaly a2 = find(an::AnomalyKind::LifespanTail, "Old");
    CHECK(a2.value == 150);
    const auto t2 = an::explain_anomaly(a2, ev);
    CHECK(t2.verdict == an::Verdict::Error && t2.bits_unusual > t2.bits_error);  // the tail costs more than an error
    const an::Anomaly a5 = find(an::AnomalyKind::FounderTiming, "Org");
    CHECK(a5.facts.size() == 3 && a5.linked == "P" && a5.value == 50);
  }
  // ---- G3: abduction through a const rule and a path rule
  {
    TripleStore g;
    for (int i = 0; i < 30; ++i) {
      const std::string p = "poet" + std::to_string(i);
      g.add(p, "occupation", "Poet"), g.add(p, "occupation", "Writer");
    }
    for (int i = 0; i < 60; ++i) g.add("painter" + std::to_string(i), "occupation", "Painter");
    for (int i = 0; i < 20; ++i) {
      const std::string x = "x" + std::to_string(i), c = "c" + std::to_string(i), k = i % 2 ? "K1" : "K0";
      g.add(x, "birth_place", c), g.add(c, "country", k), g.add(x, "nationality", k);
    }
    g.add("X", "occupation", "Writer");
    g.add("Y", "nationality", "K0"), g.add("Y", "occupation", "Painter");
    const RuleIndex ix(g);
    const RuleSet rs(ix, mine_rules(ix), true);
    CHECK(!an::derivable(rs, "X", "occupation", "Writer"));
    const auto ex = an::abduce(ix, rs, "X", "occupation", "Writer", [](const an::FactRef&) { return 3.0; });
    const auto rf = an::ranked_facts(ex);
    CHECK(!rf.empty() && rf.front().first == (an::FactRef{"X", "occupation", "Poet"}));
    CHECK(!ex.empty() && ex.front().bits >= 3.0 && ex.front().bits < 3.5);
    CHECK(an::derivable(rs, "poet0", "occupation", "Writer"));
    // the path rule birth_place(X, Z) & country(Z, Y) => nationality(X, Y): the missing birth place, c0 cheapest
    auto bits = [](const an::FactRef& f) { return f.o == "c0" ? 1.0 : 5.0; };
    const auto rp = an::ranked_facts(an::abduce(ix, rs, "Y", "nationality", "K0", bits));
    CHECK(!rp.empty() && rp.front().first == (an::FactRef{"Y", "birth_place", "c0"}));
    bool odd = false;
    for (const auto& [f, b] : rp) odd = odd || f.o == "c1";  // c1 is in K1: never proposed for K0
    CHECK(!odd);
    TripleStore g2 = g;
    g2.add("X", "occupation", "Poet");
    const RuleIndex ix2(g2);
    const RuleSet rs2(ix2, mine_rules(ix2), true);
    CHECK(an::derivable(rs2, "X", "occupation", "Writer"));
    CHECK(an::abduce(ix2, rs2, "X", "occupation", "Writer", [](const an::FactRef&) { return 3.0; }).empty() ||
          an::ranked_facts(an::abduce(ix2, rs2, "X", "occupation", "Writer", [](const an::FactRef&) { return 3.0; })).front().first.o != "Poet");
    // S11e: the rule term's weight. The parts of an explanation's bits, and the order at w = 1 is S11c's order.
    bool parts = !ex.empty();
    for (const auto& e : ex) parts = parts && e.fact_bits == 3.0 && e.rule_bits >= 0 && e.bits == e.fact_bits + e.rule_bits;
    CHECK(parts);
    const auto w1 = an::ranked_facts_weighted(ex, 1.0);
    bool same = w1.size() == rf.size();
    for (std::size_t i = 0; same && i < rf.size(); ++i) same = w1[i].first == rf[i].first && w1[i].second == rf[i].second;
    CHECK(same);
    const auto rp1 = an::abduce(ix, rs, "Y", "nationality", "K0", bits);
    const auto rpw = an::ranked_facts_weighted(rp1, 1.0), rpr = an::ranked_facts(rp1);
    bool same2 = rpw.size() == rpr.size();
    for (std::size_t i = 0; same2 && i < rpr.size(); ++i) same2 = rpw[i].first == rpr[i].first;
    CHECK(same2);
    const auto ex0 = an::abduce(ix, rs, "X", "occupation", "Writer", [](const an::FactRef&) { return 3.0; }, 5000, 0.0);
    CHECK(!ex0.empty() && ex0.front().bits == 3.0 && ex0.front().rule_bits == ex.front().rule_bits);
    // a cheap fact through a weak rule against a dearer fact through a strong rule: the weight decides
    std::vector<an::Explanation> ew(3);
    ew[0].add = {{"t", "r", "a"}}, ew[0].fact_bits = 2.0, ew[0].rule_bits = 3.0, ew[0].support = 10;
    ew[1].add = {{"t", "r", "b"}}, ew[1].fact_bits = 3.0, ew[1].rule_bits = 0.5, ew[1].support = 10;
    ew[2].add = {{"t", "r", "c"}}, ew[2].fact_bits = 2.0, ew[2].rule_bits = 3.0, ew[2].support = 20;  // ties a, more support
    const auto o1 = an::ranked_facts_weighted(ew, 1.0), o0 = an::ranked_facts_weighted(ew, 0.0), oh = an::ranked_facts_weighted(ew, 0.5);
    CHECK(o1.size() == 3 && o1[0].first.o == "b" && o1[1].first.o == "c" && o1[2].first.o == "a");
    CHECK(o0.size() == 3 && o0[0].first.o == "c" && o0[1].first.o == "a" && o0[2].first.o == "b");
    CHECK(oh.size() == 3 && oh[0].first.o == "b");
    CHECK_NEAR(o1.empty() ? 0.0 : o1[0].second, 3.5, 1e-12);
    // a rule with PCA below 1: the weight scales its bits (w = 0: the fact's bits alone; w = 0.5: half the rule's)
    {
      TripleStore gw;
      for (int i = 0; i < 30; ++i) {
        const std::string p = "bard" + std::to_string(i);
        gw.add(p, "occupation", "Bard"), gw.add(p, "occupation", "Singer");
      }
      for (int i = 0; i < 10; ++i) gw.add("lonebard" + std::to_string(i), "occupation", "Bard");
      for (int i = 0; i < 60; ++i) gw.add("smith" + std::to_string(i), "occupation", "Smith");
      gw.add("Z", "occupation", "Singer");
      const RuleIndex ixw(gw);
      const RuleSet rsw(ixw, mine_rules(ixw), true);
      auto three = [](const an::FactRef&) { return 3.0; };
      const auto e1 = an::abduce(ixw, rsw, "Z", "occupation", "Singer", three);
      const auto e0 = an::abduce(ixw, rsw, "Z", "occupation", "Singer", three, 5000, 0.0);
      const auto eh = an::abduce(ixw, rsw, "Z", "occupation", "Singer", three, 5000, 0.5);
      CHECK(!e1.empty() && e1.front().rule_bits > 0.1);
      CHECK(!e0.empty() && e0.front().bits == 3.0);
      CHECK(!e1.empty() && !eh.empty() && std::fabs(eh.front().bits - (3.0 + 0.5 * e1.front().rule_bits)) < 1e-12);
    }
    // one fact, two explanations: its best one counts
    std::vector<an::Explanation> e2(2);
    e2[0].add = {{"t", "r", "a"}}, e2[0].fact_bits = 2.0, e2[0].rule_bits = 4.0;
    e2[1].add = {{"t", "r", "a"}}, e2[1].fact_bits = 2.0, e2[1].rule_bits = 1.0;
    const auto r2 = an::ranked_facts_weighted(e2, 1.0);
    CHECK(r2.size() == 1 && r2[0].second == 3.0);
  }
  // ---- G4: equality and path roles, the guards
  {
    TripleStore g;
    g.add("S1", "capital", "a1"), g.add("S1", "largest_city", "a1"), g.add("S1", "currency", "Euro"), g.add("S1", "language", "French"),
        g.add("S1", "located_in", "Europe");
    g.add("T", "capital", "t1"), g.add("T", "currency", "Euro"), g.add("T", "language", "French"), g.add("T", "located_in", "Europe");
    g.add("S2", "largest_city", "s2c"), g.add("S2", "currency", "Euro"), g.add("S2", "language", "French");  // two aligned: no analog
    const auto an1 = an::find_analogs(g, "T", "largest_city");
    CHECK(an1.size() == 1 && an1.front().s == "S1" && an1.front().aligned == 4 && an1.front().identical == 3);
    const auto tr0 = an::transfer(g, an1, "T", "largest_city", 0.0);
    CHECK(tr0.size() == 1 && tr0.front().first == "t1");
    CHECK_NEAR(tr0.empty() ? 0.0 : tr0.front().second, 4.0, 1e-12);
    const auto tr1 = an::transfer(g, an1, "T", "largest_city", 1.0);
    bool has_a1 = false, has_t1 = false;
    for (const auto& [o, s] : tr1) has_a1 = has_a1 || o == "a1", has_t1 = has_t1 || o == "t1";
    CHECK(has_a1 && has_t1 && tr1.size() == 2);
    an::AnalogConfig strict;
    strict.min_identical = 4;
    CHECK(an::find_analogs(g, "T", "largest_city", strict).empty());
    TripleStore h;
    h.add("Sp", "birth_place", "Lyon"), h.add("Lyon", "country", "France"), h.add("Sp", "nationality", "France");
    h.add("Sp", "occupation", "Writer"), h.add("Sp", "genre", "Novel"), h.add("Sp", "language", "French");
    h.add("Tp", "birth_place", "Munich"), h.add("Munich", "country", "Germany");
    h.add("Tp", "occupation", "Writer"), h.add("Tp", "genre", "Novel"), h.add("Tp", "language", "French");
    const auto an2 = an::find_analogs(h, "Tp", "nationality");
    CHECK(an2.size() == 1 && an2.front().higher == 1 && an2.front().sys == 5);
    const auto tr2 = an::transfer(h, an2, "Tp", "nationality", 0.0);
    CHECK(!tr2.empty() && tr2.front().first == "Germany");
  }
  // ---- G5: words, entity vectors and the relation offset
  {
    CHECK(an::WordVectors::words_of("New York City") == (std::vector<std::string>{"new", "york", "city"}));
    CHECK(an::WordVectors::words_of("C\xc3\xb4te d'Ivoire") == (std::vector<std::string>{"c\xc3\xb4te", "d", "ivoire"}));
    CHECK(an::WordVectors::words_of("\xc3\x89ire") == (std::vector<std::string>{"\xc3\xa9ire"}));
    an::WordVectors wv;
    wv.add("france", {1, 0, 0, 0}), wv.add("germany", {0, 1, 0, 0}), wv.add("italy", {0, 0, 1, 0});
    wv.add("paris", {1, 0, 0, 1}), wv.add("berlin", {0, 1, 0, 1}), wv.add("rome", {0, 0, 1, 1}), wv.add("venice", {0, 0, 1, -1});
    CHECK(wv.size() == 7 && wv.dim() == 4 && wv.entity("Atlantis").empty());
    CHECK_NEAR(an::cosine(wv.entity("Paris"), {1, 0, 0, 1}), 1.0, 1e-6);
    TripleStore g;
    g.add("France", "capital", "Paris"), g.add("Germany", "capital", "Berlin"), g.add("Italy", "language", "Italian");
    const an::VectorAnalogy va(g, wv);
    CHECK(va.has("capital") && va.facts_used("capital") == 2.0 && !va.has("language"));
    const auto q = va.query("Italy", "capital");
    double r = 0, p = 0, bn = 0, vn = 0;
    CHECK(va.score(q, "Rome", r) && va.score(q, "Paris", p) && va.score(q, "Berlin", bn) && va.score(q, "Venice", vn));
    CHECK(r > p + 0.1 && r > bn + 0.1 && r > vn + 0.1);  // the offset, not Italy's own vector, separates Rome from Venice
    double z = 0;
    CHECK(!va.score(q, "Atlantis", z) && va.query("Atlantis", "capital").empty());
  }
}

// ---------------------------------------------------------------------------
// S9b and S7d: fragile gaps, a promotion's supporting sources, and where K4's guarantee and K3's part
// ---------------------------------------------------------------------------

static void test_fragile_S9b() {
  // Fragile gaps (S9b): one per slot, none where a contradiction or certainty gap already asks about the slot.
  GapMap m;
  m.open(GapKind::Certainty, "Auguste Rodin", "death_date", "person").p = 0.97;
  m.open(GapKind::Contradiction, "Bob Dylan", "birth_place", "person").p = 0.6;
  Gap* f = m.add_fragile("Agamemnon", "birth_place", 0.66, "person", "Mycenae");
  CHECK(f != nullptr && f->kind == GapKind::Fragile && f->p == 0.66 && f->p_exists == 1.0 && f->demand == 0 && f->status == GapStatus::Open);
  CHECK_STR(gap_kind_name(GapKind::Fragile), "fragile");
  CHECK(m.add_fragile("Auguste Rodin", "death_date", 0.66) == nullptr);  // the certainty gap asks already
  CHECK(m.add_fragile("Bob Dylan", "birth_place", 0.66) == nullptr);     // the contradiction gap asks already
  CHECK(m.add_fragile("AGAMEMNON", "birth_place", 0.9) == f && f->p == 0.66 && m.size() == 3);  // one per slot, the first kept
  CHECK(m.add_fragile("", "birth_place", 0.5) == nullptr && m.add_fragile("Troy", "", 0.5) == nullptr);
  CHECK(m.find(GapKind::Schema, "Agamemnon", "birth_place") == nullptr);  // its own kind: no other kind's key
  // A demand tag never lands on a fragile gap (S3's order: schema, contradiction, certainty, demand).
  const Gap* d = m.add_demand(parse_unknown_tag("<|unknown reason=NO_FACT entity=\"Agamemnon\" relation=birth_place gap=G-0000|>"));
  CHECK(d != nullptr && d->kind == GapKind::Demand && f->demand == 0);
  m.score();
  CHECK_NEAR(f->eig, expected_information_gain(0.66, 0.9, 0.9).eig, 1e-12);
  // Ties in VOI keep the older kinds first: the fragile kind is last.
  GapMap t;
  t.open(GapKind::Certainty, "B", "capital").p = 0.66;
  t.add_fragile("A", "capital", 0.66);
  t.score();
  CHECK(t.ranked()[0]->kind == GapKind::Certainty && t.ranked()[1]->kind == GapKind::Fragile);

  // A promotion's supporting sources (S7d): positive, independent, not the generator's own; each once, in test order.
  Hypothesis h;
  h.generator = "G2";
  auto test = [](const char* src, bool indep, bool pos, bool inc) {
    HypTest x;
    x.source = src, x.independent = indep, x.done = true, x.positive = pos, x.inconclusive = inc;
    return x;
  };
  h.tests = {test("store", false, true, false), test("wikidata", true, true, false), test("corpus", true, false, true)};
  CHECK(supporting_independent_sources(h) == std::vector<std::string>{"wikidata"});  // rests on Wikidata alone
  h.tests[2] = test("corpus", true, true, false);
  CHECK((supporting_independent_sources(h) == std::vector<std::string>{"wikidata", "corpus"}));
  h.tests.push_back(test("wikidata", true, true, false));
  CHECK(supporting_independent_sources(h).size() == 2);  // a source counts once
  h.tests = {test("G2", true, true, false), test("corpus", true, false, false)};
  CHECK(supporting_independent_sources(h).empty());  // the generator's own result, and a negative one, carry nothing
  h.tests[1].done = false, h.tests[1].positive = true;
  CHECK(supporting_independent_sources(h).empty());  // a test not run carries nothing

  // Where K4 and K3 part (S7d 5a): one independent test whose stated null likelihoods understate how often false
  // hypotheses pass it. K4's p-values rest on the stated model, so its false share goes over q; conformal selection
  // calibrated on draws from the real distribution stays under it.
  {
    HypTest k;
    k.kind = "T", k.source = "corpus", k.independent = true;
    k.likelihoods.t[0] = 0.8, k.likelihoods.t[1] = 0.05, k.likelihoods.t[2] = 0.15;
    k.likelihoods.f[0] = 0.02, k.likelihoods.f[1] = 0.5, k.likelihoods.f[2] = 0.48;  // stated: false passes 2% of the time
    const double real_false_pos = 0.25;                                               // real: 25%
    const std::vector<HypTest> plan = {k};
    std::mt19937_64 g(20261029);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto draw = [&](bool truth, double& post, double& e, bool& sup) {
      const double u = U(g);
      const int o = truth ? (u < 0.8 ? 0 : u < 0.85 ? 1 : 2) : (u < real_false_pos ? 0 : u < 0.6 ? 1 : 2);
      const TestOutcome out = o == 0 ? TestOutcome::Positive : o == 1 ? TestOutcome::Negative : TestOutcome::Inconclusive;
      e = bayes_factor(out, k);
      post = update_posterior(0.5, e);
      sup = post > 0.9 && out == TestOutcome::Positive;
    };
    double fdp4 = 0, fdp3 = 0;
    const int reps = 200, m = 200;
    for (int r = 0; r < reps; ++r) {
      std::vector<std::vector<HypTest>> plans(m, plan);
      std::vector<double> pri(m, 0.5), e(m), post(m), cs, bs;
      std::vector<bool> sup(m), truth(m), ct;
      for (int i = 0; i < m; ++i) {
        truth[i] = i < m / 4;  // a batch where most are false
        bool s2 = false;
        double p2 = 0, e2 = 1;
        draw(truth[i], post[i], e[i], s2);
        sup[i] = s2;
        bs.push_back(s2 ? post[i] : -1.0);
        const bool tc = U(g) < 0.25;  // the calibration set: drawn the same way
        draw(tc, p2, e2, s2);
        cs.push_back(s2 ? p2 : -1.0), ct.push_back(tc);
      }
      const std::vector<int> tiers = hybrid_null_model_tiers(plans, pri, e, sup, 0.1);
      const std::vector<bool> k3 = conformal_select(cs, ct, bs, 0.1);
      double n4 = 0, f4 = 0, n3 = 0, f3 = 0;
      for (int i = 0; i < m; ++i) {
        if (tiers[i] == 2) n4 += 1, f4 += !truth[i];
        if (k3[i] && sup[i]) n3 += 1, f3 += !truth[i];
      }
      fdp4 += n4 > 0 ? f4 / n4 : 0.0, fdp3 += n3 > 0 ? f3 / n3 : 0.0;
    }
    CHECK(fdp4 / reps > 0.2);   // K4 trusts the stated model and over-promotes
    CHECK(fdp3 / reps <= 0.12); // K3 keeps its guarantee (q 0.1, Monte Carlo slack)
  }
}

// ---------------------------------------------------------------------------
// S12a: G15 numeric laws (gen_laws.hpp); G1 deep deduction, G20 the wild tier (gen_deep.hpp); S12c: items 1-3 (G19 removed)
// ---------------------------------------------------------------------------

static void test_laws_deep_S12a() {
  namespace lw = cypha::eval::laws;
  namespace dp = cypha::eval::deep;
  // ---- robust statistics and the chi-square tail
  CHECK_NEAR(lw::median_of({4, 1, 3, 2}), 2.5, 1e-12);
  CHECK_NEAR(lw::robust_sd({1, 2, 3, 4, 100}), 1.4826, 1e-9);  // one outlier does not move it
  CHECK_NEAR(lw::chi2_lower(2.0, 2.0), 1.0 - std::exp(-1.0), 1e-10);
  CHECK_NEAR(lw::chi2_lower(3.841458820694124, 1.0), 0.95, 1e-6);
  CHECK_NEAR(lw::chi2_lower(30.0, 30.0) + (1.0 - lw::chi2_lower(30.0, 30.0)), 1.0, 1e-12);
  CHECK(lw::chi2_lower(1e-3, 39.0) < 1e-40);
  // ---- the exponent vectors BACON's walk reaches (first positive, gcd 1)
  const auto e2 = lw::detail::exponent_vectors(2, 3), e3 = lw::detail::exponent_vectors(3, 2);
  CHECK(e2.size() == 14 && e3.size() == 28);
  bool canon = true;
  for (const auto& e : e2) canon = canon && e[0] > 0 && std::gcd(e[0], e[1]) == 1;
  CHECK(canon);
  CHECK(lw::detail::in_span({{1, -1, 0}}, {2, -2, 0}) && !lw::detail::in_span({{1, -1, 0}}, {1, 0, -1}));
  CHECK(lw::detail::in_span({{1, -1, 0}, {0, 1, -1}}, {1, 0, -1}));  // the third is implied by the first two
  // ---- a table: density = population / area (5% noise-free), and a second unit of area
  {
    lw::NumTable t;
    t.type = "toy countries";
    const std::size_t A = t.add_attr("area", lw::Scale::Ratio), M = t.add_attr("area_mi2", lw::Scale::Ratio), P = t.add_attr("population", lw::Scale::Ratio),
                      D = t.add_attr("density", lw::Scale::Ratio), H = t.add_attr("hdi", lw::Scale::Ratio);
    std::mt19937_64 rng(7);
    for (int i = 0; i < 60; ++i) {
      const std::size_t e = t.add_entity("country" + std::to_string(i));
      const double area = std::exp(std::normal_distribution<double>(11, 2)(rng)), pop = std::exp(std::normal_distribution<double>(15, 2)(rng));
      t.v[A][e] = area, t.v[M][e] = area / 2.58999, t.v[P][e] = pop, t.v[D][e] = pop / area * std::exp(std::normal_distribution<double>(0, 0.01)(rng));
      t.v[H][e] = std::uniform_real_distribution<double>(0.3, 0.95)(rng);
    }
    t.v[D][3] = 1e9;  // a typing error: robust statistics keep the law
    const auto laws = lw::search_laws(t);
    int kept = 0, dens = 0, conv = 0, hdi = 0, redundant = 0;
    for (const auto& L : laws) {
      redundant += L.redundant;
      if (!L.kept) continue;
      ++kept;
      if (L.attr == std::vector<std::size_t>{A, M} && L.exp == std::vector<int>{1, -1}) {
        ++conv;
        CHECK_NEAR(L.constant, 2.58999, 1e-6);
      }
      if (L.attr == std::vector<std::size_t>{A, P, D} && L.exp == std::vector<int>{1, -1, 1}) {
        ++dens;
        CHECK_NEAR(L.constant, 1.0, 0.01);
      }
      for (std::size_t a : L.attr) hdi += a == H;
    }
    CHECK(conv == 1 && dens == 1 && hdi == 0);
    CHECK(kept == 2 && redundant >= 1);  // density through area_mi2 is implied by the two, so it is redundant
  }
  // ---- interval attributes: differences and orders, never ratios; the order null is independence (amendment 1)
  {
    lw::NumTable t;
    t.type = "toy persons";
    const std::size_t B = t.add_attr("birth", lw::Scale::Interval), Z = t.add_attr("z", lw::Scale::Interval), S = t.add_attr("shifted", lw::Scale::Interval);
    std::mt19937_64 rng(11);
    for (int i = 0; i < 80; ++i) {
      const std::size_t e = t.add_entity("p" + std::to_string(i));
      const double b = std::floor(std::uniform_real_distribution<double>(-500, 1950)(rng));
      t.v[B][e] = b, t.v[Z][e] = b + 30;                                       // Z = birth + 30
      t.v[S][e] = std::floor(std::uniform_real_distribution<double>(1800, 2600)(rng));  // later than almost every birth, no link
    }
    const auto laws = lw::search_laws(t);
    bool diff = false, order_bz = false, order_bs = false, power = false;
    for (const auto& L : laws) {
      power = power || L.kind == lw::LawKind::Power;
      if (!L.kept) continue;
      if (L.kind == lw::LawKind::Diff && L.attr == std::vector<std::size_t>{B, Z}) {
        diff = true;
        CHECK_NEAR(L.constant, 30.0, 1e-9);
      }
      if (L.kind == lw::LawKind::Order && L.attr == std::vector<std::size_t>{B, Z}) order_bz = true;
      if (L.kind == lw::LawKind::Order && L.attr == std::vector<std::size_t>{B, S}) order_bs = true;
    }
    CHECK(diff && order_bz && !power);
    CHECK(!order_bs);  // birth <= shifted holds for almost every entity, but no more than for unrelated pairs (a coin null keeps it)
  }
  // ---- the held-out third: a law that fails on held-out entities is not kept
  {
    lw::NumTable t;
    const std::size_t X = t.add_attr("x", lw::Scale::Ratio), Y = t.add_attr("y", lw::Scale::Ratio);
    std::mt19937_64 rng(13);
    std::size_t held = 0;
    for (int i = 0; i < 120; ++i) {
      const std::string name = "h" + std::to_string(i);
      const std::size_t e = t.add_entity(name);
      const double x = std::exp(std::normal_distribution<double>(3, 1)(rng));
      t.v[X][e] = x;
      t.v[Y][e] = lw::heldout_entity(name) ? x * std::exp(std::normal_distribution<double>(0, 1)(rng)) : 2 * x;
      held += lw::heldout_entity(name);
    }
    CHECK(held > 20 && held < 70);
    bool kept = false, screened = false;
    for (const auto& L : lw::search_laws(t))
      if (L.attr == std::vector<std::size_t>{X, Y} && L.exp == std::vector<int>{1, -1}) kept = L.kept, screened = L.heldout_ok;
    CHECK(!kept && !screened);
  }
  // ---- the planted gate's pieces: a planted law is found, its null finds nothing
  {
    std::vector<lw::PersonYears> persons;
    for (int i = 0; i < 100; ++i) persons.push_back({"person" + std::to_string(i), 1500.0 + 4 * i, 1560.0 + 4 * i + (i % 7)});
    int found = 0, nulls = 0;
    for (auto kind : {lw::PlantKind::Ratio, lw::PlantKind::Product, lw::PlantKind::Power, lw::PlantKind::Scale, lw::PlantKind::Offset})
      for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        const lw::Planted P = lw::plant(kind, 0.0, seed * 977, persons, 40, 0.0);
        found += lw::planted_found(P, lw::search_laws(P.table));
        nulls += lw::false_laws(P, lw::search_laws(P.null_table), true) > 0;
      }
    CHECK(found == 15 && nulls == 0);
    const lw::Planted Q = lw::plant(lw::PlantKind::Ratio, 0.0, 5, persons, 40, 0.0);
    CHECK(Q.table.entities.size() == 40 && Q.table.attrs.size() == 5 && Q.null_table.v[Q.z] != Q.table.v[Q.z]);
    CHECK_NEAR(lw::implied_c(lw::PlantKind::Power, 1.0 / (3.0 * 3.0)), 3.0, 1e-12);
  }
  // ---- G1: chaining two rules through a derived premise
  {
    TripleStore st;
    st.add("x1", "occupation", "poet");
    st.add("x2", "occupation", "writer");
    std::vector<Rule> rules(2);
    rules[0].kind = RuleKind::Const, rules[0].r1 = "occupation", rules[0].c1 = "poet", rules[0].head = "occupation", rules[0].head_o = "writer";
    rules[0].support = 8, rules[0].pca_body = 10, rules[0].kept = true;
    rules[1].kind = RuleKind::Const, rules[1].r1 = "occupation", rules[1].c1 = "writer", rules[1].head = "occupation", rules[1].head_o = "author";
    rules[1].support = 5, rules[1].pca_body = 10, rules[1].kept = true;
    const RuleBook book(rules);
    const auto d = dp::deep_chain(st, book, 6);
    std::map<std::string, std::pair<int, double>> got;
    for (const auto& x : d) got[x.t.s + "|" + x.t.o] = {x.depth, x.conf};
    CHECK(got.size() == 3);
    CHECK(got["x1|writer"].first == 1 && got["x1|author"].first == 2 && got["x2|author"].first == 1);
    CHECK_NEAR(got["x1|author"].second, 0.8 * 0.5, 1e-12);  // the product of the rules' PCA confidences
    CHECK(dp::deep_chain(st, book, 1).size() == 2);         // depth 1 is G2's own prediction
  }
  // ---- S12c item 1: a power law's spread on the attribute it predicts least precisely, T^(1/min|e|)
  {
    lw::NumTable t;
    const std::size_t X = t.add_attr("x", lw::Scale::Ratio), Z = t.add_attr("z", lw::Scale::Ratio), Y = t.add_attr("y", lw::Scale::Ratio);
    std::mt19937_64 rng(17);
    for (int i = 0; i < 400; ++i) {
      const std::size_t e = t.add_entity("w" + std::to_string(i));
      const double x = std::exp(std::normal_distribution<double>(4, 1)(rng));
      t.v[X][e] = x;
      t.v[Z][e] = 3.0 * std::pow(x, 1.5) * std::exp(std::normal_distribution<double>(0, 0.06)(rng));  // Kepler-like, 6% noise on z
      t.v[Y][e] = 2.0 * x * std::exp(std::normal_distribution<double>(0, 0.02)(rng));
    }
    lw::LawConfig on;
    on.power_min_exp = true;
    const auto off = lw::search_laws(t), onl = lw::search_laws(t, on);
    double s_off = -1, s_on = -1, y_off = -1, y_on = -1;
    bool kept_off = false, kept_on = false;
    for (const auto& L : off) {
      if (L.attr == std::vector<std::size_t>{X, Z} && L.exp == std::vector<int>{3, -2}) s_off = L.spread, kept_off = L.kept;
      if (L.attr == std::vector<std::size_t>{X, Y} && L.exp == std::vector<int>{1, -1}) y_off = L.spread;
    }
    for (const auto& L : onl) {
      if (L.attr == std::vector<std::size_t>{X, Z} && L.exp == std::vector<int>{3, -2}) s_on = L.spread, kept_on = L.kept;
      if (L.attr == std::vector<std::size_t>{X, Y} && L.exp == std::vector<int>{1, -1}) y_on = L.spread;
    }
    CHECK(s_off > 0.1 && !kept_off);             // the term carries z's noise twice
    CHECK(s_on < 0.1 && kept_on);                // z, the least precisely predicted attribute, is within the bar
    CHECK(std::fabs(s_on / s_off - 0.5) < 0.05);  // divided by min |e| = 2, not by max |e| = 3
    CHECK(y_off > 0 && y_on == y_off);            // a law with an exponent of 1 is unchanged
  }
  // ---- S12c item 2: orders at 90% with a margin over independence
  {
    lw::NumTable t;
    const std::size_t B = t.add_attr("birth", lw::Scale::Interval), Z = t.add_attr("z", lw::Scale::Interval), S = t.add_attr("shifted", lw::Scale::Interval);
    std::mt19937_64 rng(19);
    int bad = 0;
    for (int i = 0; i < 80; ++i) {
      const std::string name = "q" + std::to_string(i);
      const std::size_t e = t.add_entity(name);
      const double b = std::floor(std::uniform_real_distribution<double>(-500, 1950)(rng));
      t.v[B][e] = b;
      t.v[Z][e] = b + 20 + std::floor(std::uniform_real_distribution<double>(0, 40)(rng));
      if (bad < 6 && !lw::heldout_entity(name)) t.v[Z][e] = b - 100, ++bad;  // 6 typing errors (7.5%), none held out
      t.v[S][e] = std::floor(std::uniform_real_distribution<double>(1800, 2600)(rng));
    }
    CHECK(bad == 6);
    lw::LawConfig rel;
    rel.order_share = 0.90, rel.order_heldout_viol = 0.20, rel.order_margin = 0.20;
    auto kept_order = [&](const std::vector<lw::Law>& laws, std::size_t x, std::size_t y) {
      for (const auto& L : laws)
        if (L.kind == lw::LawKind::Order && L.attr == std::vector<std::size_t>{x, y}) return L.kept;
      return false;
    };
    const auto d = lw::search_laws(t), r = lw::search_laws(t, rel);
    CHECK(!kept_order(d, B, Z) && kept_order(r, B, Z));  // 74 of 80 (0.925): below 95%, above 90% and far above independence
    CHECK(!kept_order(r, B, S));                         // high share, but no more than for unrelated pairs: the margin holds it out
  }
  {
    // an order whose share (0.95) passes the binomial test against independence (p0 about 0.8) but not the margin
    lw::NumTable t;
    const std::size_t B = t.add_attr("birth", lw::Scale::Interval), W = t.add_attr("w", lw::Scale::Interval);
    std::mt19937_64 rng(23);
    int bad = 0;
    for (int i = 0; i < 80; ++i) {
      const std::string name = "m" + std::to_string(i);
      const std::size_t e = t.add_entity(name);
      const double b = std::floor(std::uniform_real_distribution<double>(-500, 1950)(rng));
      t.v[B][e] = b;
      t.v[W][e] = b + std::floor(std::uniform_real_distribution<double>(600, 1200)(rng));
      if (bad < 4 && !lw::heldout_entity(name)) t.v[W][e] = b - 100, ++bad;  // 4 of 80 below: share 0.95
    }
    lw::LawConfig no_margin, margin;
    no_margin.order_share = margin.order_share = 0.90, no_margin.order_heldout_viol = margin.order_heldout_viol = 0.20;
    margin.order_margin = 0.30;
    const lw::Law* o = nullptr;
    const auto a = lw::search_laws(t, no_margin), m = lw::search_laws(t, margin);
    for (const auto& L : a)
      if (L.kind == lw::LawKind::Order && L.attr == std::vector<std::size_t>{B, W}) o = &L;
    CHECK(bad == 4 && o != nullptr);
    if (o) {
      CHECK(o->p0 > 0.7 && o->p0 < 0.9 && o->inlier - o->p0 < 0.30);  // beats independence, but by less than the margin
      CHECK(o->kept);                                                  // without the margin: kept on its p-value
    }
    bool kept_m = false;
    for (const auto& L : m)
      if (L.kind == lw::LawKind::Order && L.attr == std::vector<std::size_t>{B, W}) kept_m = L.kept;
    CHECK(!kept_m);  // the margin holds it out
  }
  // ---- S12c item 3: G1's prior by depth
  {
    dp::DepthCalib cal;
    for (int i = 0; i < 8; ++i) cal.add(2, i < 2);
    for (int i = 0; i < 10; ++i) cal.add(4, false);
    for (int i = 0; i < 200; ++i) cal.add(6, false);
    CHECK_NEAR(cal.prior(2), 3.0 / 10.0, 1e-12);
    CHECK_NEAR(cal.prior(4), 1.0 / 12.0, 1e-12);
    CHECK_NEAR(cal.prior(5), 0.5, 1e-12);  // no counts at depth 5
    CHECK_NEAR(dp::g1_prior(dp::G1Prior::Chain, 5, 0.45, cal), 0.45, 1e-12);
    CHECK_NEAR(dp::g1_prior(dp::G1Prior::Chain, 1, 0.999, cal), 0.98, 1e-12);  // clamped
    CHECK(std::isnan(dp::g1_prior(dp::G1Prior::Cap3, 4, 0.45, cal)));
    CHECK_NEAR(dp::g1_prior(dp::G1Prior::Cap3, 3, 0.45, cal), 0.45, 1e-12);
    CHECK_NEAR(dp::g1_prior(dp::G1Prior::Depth, 4, 0.45, cal), 1.0 / 12.0, 1e-12);
    CHECK_NEAR(dp::g1_prior(dp::G1Prior::Depth, 2, 0.999, cal), 0.3, 1e-12);
    CHECK_NEAR(dp::g1_prior(dp::G1Prior::Depth, 6, 0.6, cal), 0.02, 1e-12);  // 1/202, clamped to the floor
  }
  // ---- G20: the wild tier's clusters, budget and support rule
  {
    TripleStore st;
    st.add("Alice", "birth_year", "1900");
    st.add("Alice", "nationality", "France");
    st.add("Lyon", "located_in", "France");
    CHECK_STR(dp::cluster_of(st, "Alice"), "person|France");
    CHECK_STR(dp::cluster_of(st, "Lyon"), "place|none");
    CHECK_STR(dp::cluster_of(st, "Nowhere"), "value|none");
    TripleStore g;
    std::mt19937_64 rng(3);
    std::vector<std::pair<std::string, std::string>> slots;
    for (int i = 0; i < 40; ++i) {
      const std::string p = "p" + std::to_string(i);
      g.add(p, "birth_year", std::to_string(1800 + i));
      g.add(p, "nationality", i % 2 ? "France" : "Italy");
      g.add(p, "occupation", i % 3 ? "Painter" : "Poet");
      if (i < 30) g.add(p, "birth_place", i % 2 ? "Paris" : "Rome");
      else slots.push_back({p, "birth_place"});
    }
    g.add("Paris", "country", "France");
    g.add("Rome", "country", "Italy");
    const LinkPredictor lp(g, LinkPredictor::Config{});
    const auto w1 = dp::wild_tier(g, lp, slots, 5, 99), w2 = dp::wild_tier(g, lp, slots, 5, 99);
    CHECK(w1.size() <= 5 && w1.size() == w2.size());
    bool same = true, rule_ok = true;
    for (std::size_t i = 0; i < w1.size(); ++i) {
      same = same && w1[i].t == w2[i].t;
      rule_ok = rule_ok && w1[i].lp >= 2.0 * w1[i].base && dp::cluster_of(g, w1[i].t.o) != dp::cluster_of(g, w1[i].t.s) && !g.has(w1[i].t.s, w1[i].t.r, w1[i].t.o);
    }
    CHECK(same && rule_ok);
  }
}

// ---------------------------------------------------------------------------
// S6c: G12's closed form, documents holding both left out (abc_closed.hpp)
// ---------------------------------------------------------------------------

static void test_abc_closed_S6c() {
  namespace ab = cypha::eval::abc;
  // Entities: A 0, C 1, B1 2, B2 3, B3 4, X 5, Y 6. Document 4 holds both A and C (and B1, B3).
  const std::vector<std::vector<std::uint32_t>> docs = {{0, 2}, {2, 1}, {0, 3}, {3, 1}, {0, 1, 2, 4}, {4, 5}, {2, 5}, {6}};
  const ab::Cooc c = ab::Cooc::build(docs, 7);
  CHECK(c.n_docs() == 8 && c.df(0) == 3 && c.df(2) == 4 && c.df(6) == 1);
  CHECK(c.shared(0, 1) == std::vector<std::uint32_t>{4} && c.shared(0, 5).empty());
  const std::vector<char> all(7, 1);
  const ab::Closed cl(c, all, 200, 2);
  // Left out: document 4. N' 7, n'_A 2, n'_C 2; B1 has 3 documents left (0, 1, 6), B2 has 2; B3 is in no other
  // document of A, so the strong link inside document 4 counts for nothing.
  const double b1 = std::log(7.0 / 3.0) * std::log(7.0 / 6.0) * std::log(7.0 / 6.0);
  const double b2 = std::log(7.0 / 2.0) * std::log(7.0 / 4.0) * std::log(7.0 / 4.0);
  const ab::ClosedResult r = cl.score(0, 1);
  CHECK_NEAR(r.score, b1 + b2, 1e-12);
  CHECK(r.links == 2 && r.left_out == 1);
  CHECK(r.top.size() == 2 && r.top[0].first == 3 && r.top[1].first == 2);
  CHECK_NEAR(cl.score(1, 0).score, r.score, 1e-12);  // symmetric
  CHECK(cl.score(0, 0).score == 0.0 && cl.score(0, 0).links == 0);
  // A pair that never shares a document is S11b's closed form: (A, X) over all 8 documents. B1's link to X has
  // PPMI ln(8 * 1 / (4 * 2)) = 0, so only B3 counts.
  const ab::ClosedResult rx = cl.score(0, 5);
  CHECK(rx.left_out == 0 && rx.links == 1);
  CHECK_NEAR(rx.score, std::log(4.0) * std::log(8.0 / 6.0) * std::log(2.0), 1e-12);
  // The hub cap is inclusive (B1 with 3 documents stays at cap 3, goes at cap 2); min_df; usable.
  CHECK_NEAR(ab::Closed(c, all, 3, 2).score(0, 1).score, b1 + b2, 1e-12);
  CHECK_NEAR(ab::Closed(c, all, 2, 2).score(0, 1).score, b2, 1e-12);
  CHECK_NEAR(ab::Closed(c, all, 200, 3).score(0, 1).score, b1, 1e-12);
  std::vector<char> no_b2 = all;
  no_b2[3] = 0;
  CHECK_NEAR(ab::Closed(c, no_b2, 200, 2).score(0, 1).score, b1, 1e-12);
  // The counters are reset between calls: the same pair twice, and another pair in between.
  CHECK_NEAR(cl.score(0, 1).score, r.score, 1e-15);
  // An entity whose only documents are shared ones has nothing left.
  const ab::Cooc c2 = ab::Cooc::build({{0, 1}, {0, 1, 2}, {2}}, 3);
  CHECK(ab::Closed(c2, std::vector<char>(3, 1), 200, 1).score(0, 1).score == 0.0);
  CHECK_NEAR(ab::ppmi(8, 1, 2, 2), std::log(2.0), 1e-15);
  CHECK(ab::ppmi(8, 1, 4, 2) == 0.0 && ab::ppmi(8, 0, 1, 1) == 0.0);
  // Titles kept out of every role, and wiki links.
  for (const char* t : {"1952", "300 BC", "AD 79", "1990s", "19th century", "March 4", "4 March", "June", "List of rivers"}) CHECK(ab::date_or_list_title(t));
  for (const char* t : {"Paris", "War of 1812", "1984 (novel)", "May Day"}) CHECK(!ab::date_or_list_title(t));
  const auto lt = ab::link_targets("[[Paris|the city]] and [[france#History]], [[Category:X]] [[:fr:Paris]] [[new_york]]");
  CHECK(lt.size() == 3 && lt[0] == "Paris" && lt[1] == "France" && lt[2] == "New york");
  const std::unordered_map<std::string, std::string> red = {{"UK", "United Kingdom"}, {"U.K.", "UK"}};
  CHECK(ab::resolve("U.K.", red) == "United Kingdom" && ab::resolve("Paris", red) == "Paris");
}

// ---------------------------------------------------------------------------
// S4d: the fresh split of the pages after kSignalPlainFrom, and the second alarm threshold (unknown_signals.hpp)
// ---------------------------------------------------------------------------

static void test_threshold_S4d() {
  // The fresh split: P (S4c's control) q % 4 = 0, P2 q % 4 = 2, the fresh stream F q % 4 in {1, 3}.
  CHECK(signal_fresh_part(0) == SignalFreshPart::P && signal_fresh_part(1) == SignalFreshPart::F && signal_fresh_part(2) == SignalFreshPart::P2);
  CHECK(signal_fresh_part(3) == SignalFreshPart::F && signal_fresh_part(4) == SignalFreshPart::P && signal_fresh_part(1001) == SignalFreshPart::F);
  {
    std::size_t np = 0, nf = 0, n2 = 0;
    for (std::size_t q = 0; q < 4000; ++q) {
      const SignalFreshPart x = signal_fresh_part(q);
      np += x == SignalFreshPart::P, nf += x == SignalFreshPart::F, n2 += x == SignalFreshPart::P2;
    }
    CHECK(np == 1000 && n2 == 1000 && nf == 2000);
  }
  // The second threshold.
  CHECK(kSignalAHigh == 100000.0);
  SignalRule lo, hi;
  hi.A = kSignalAHigh;
  CHECK(lo.A == 20000.0 && hi.alpha == lo.alpha && hi.p1 == lo.p1 && hi.cap == lo.cap && hi.min_ref == lo.min_ref);
  // All events: R 5, 30, 155, 780, 3905, 19530, 97655, 488280. A = 20,000 alarms at the 7th, A = 100,000 at the 8th.
  {
    SRDetector a(lo), b(hi);
    int at_a = 0, at_b = 0;
    for (int t = 1; t <= 10; ++t) {
      if (!at_a && a.step(true)) at_a = t;
      if (!at_b && b.step(true)) at_b = t;
    }
    CHECK(at_a == 7 && at_b == 8);
    CHECK_NEAR(b.alarm_R, 488280.0, 1e-6);
  }
  // On any event sequence the higher threshold alarms no more often and no sooner (both start at R = 0).
  {
    std::mt19937_64 rng(20261029);
    std::bernoulli_distribution B(0.12);
    int fewer = 0, ok_count = 0, ok_first = 0;
    for (int r = 0; r < 300; ++r) {
      SRDetector a(lo), b(hi);
      int na = 0, nb = 0, fa = 0, fb = 0;
      for (int t = 1; t <= 3000; ++t) {
        const bool e = B(rng);
        if (a.step(e)) ++na, fa = fa ? fa : t;
        if (b.step(e)) ++nb, fb = fb ? fb : t;
      }
      ok_count += nb <= na;
      ok_first += fb == 0 || (fa > 0 && fa <= fb);
      fewer += nb < na;
    }
    CHECK(ok_count == 300 && ok_first == 300 && fewer > 0);
  }
  // P(an alarm within L units) <= L / A at p0: 2,000 streams of 5,000 units, bound 0.05 at A = 100,000 (0.25 at 20,000).
  {
    std::mt19937_64 rng(13);
    std::bernoulli_distribution B(0.05);
    int with_lo = 0, with_hi = 0;
    for (int st = 0; st < 2000; ++st) {
      SRDetector a(lo), b(hi);
      bool any_a = false, any_b = false;
      for (int t = 0; t < 5000; ++t) {
        const bool e = B(rng);
        any_a |= a.step(e), any_b |= b.step(e);
      }
      with_lo += any_a, with_hi += any_b;
    }
    CHECK(with_hi / 2000.0 <= 0.06 && with_hi < with_lo && with_lo / 2000.0 <= 0.27);
  }
  // The same replicates at both thresholds (pattern_learn seeds both from the signal's name): with no alarm before the
  // plants, both runs carry the same R into each replicate, so A = 100,000 detects no sooner, and about one event later.
  {
    std::mt19937_64 rng(20261029);
    std::normal_distribution<double> N(0.0, 1.0);
    auto make = [&](std::size_t n, const std::string& set) {
      std::vector<SignalUnit> v;
      for (std::size_t i = 0; i < n; ++i) v.push_back({N(rng), 0, signal_unit_id(set, i)});
      return v;
    };
    const auto C = make(1000, "C"), S = make(1200, "S");
    std::vector<SignalUnit> pool;
    for (std::size_t i = 0; i < 300; ++i) pool.push_back({2.5 + N(rng), 0, signal_unit_id("P", i)});
    const SignalRun rl = signal_run(C, S, 1, true, lo), rh = signal_run(C, S, 1, true, hi);
    CHECK(rl.alarms.empty() && rh.alarms.empty() && rl.r_before == rh.r_before);
    std::vector<int> dl, dh;
    int paired = 0;
    for (int rep = 0; rep < 100; ++rep) {
      const std::size_t t0 = 100 + static_cast<std::size_t>(rep) * 9;
      std::mt19937_64 ra(1000 + static_cast<std::uint64_t>(rep)), rb(1000 + static_cast<std::uint64_t>(rep));
      const int a = signal_planted(C, S, pool, 1, true, lo, rl, t0, 1.0, 200, ra).delay;
      const int b = signal_planted(C, S, pool, 1, true, hi, rh, t0, 1.0, 200, rb).delay;
      dl.push_back(a), dh.push_back(b);
      paired += b == 0 || (a > 0 && a <= b);
    }
    CHECK(paired == 100);
    const auto sl = signal_delay_stats(dl, 200), sh = signal_delay_stats(dh, 200);
    CHECK(sl.within50 >= 95 && sh.within50 >= 95);
    CHECK(sh.median >= sl.median && sh.median <= sl.median + 4);
  }
}

// ---------------------------------------------------------------------------
// S8e: the grader with Wikidata's names and places (wd_grader.hpp; docs/reports/S8E_GRADER_AND_FRESH_SET.md)
// ---------------------------------------------------------------------------

static void test_wd_grader_S8e() {
  namespace wg = cypha::eval::wdgrade;
  using V = wg::Verdict;
  wg::Tables T;
  // Toy items: City of Brussels (alias Brussels) in Belgium; Los Angeles in California in the United States; Century City in
  // Los Angeles; the United Kingdom (demonym British) with England (demonym English) inside it.
  T.add_name("Q239", "City of Brussels"), T.add_name("Q239", "Brussels"), T.add_name("Q31", "Belgium"), T.add_name("Q31", "Belgian");
  T.add_name("Q65", "Los Angeles"), T.add_name("Q65", "L.A."), T.add_name("Q99", "California"), T.add_name("Q30", "United States of America");
  T.add_name("Q30", "American"), T.add_name("Q1054483", "Century City"), T.add_name("Q145", "United Kingdom"), T.add_name("Q145", "British");
  T.add_name("Q21", "England"), T.add_name("Q21", "English"), T.add_name("Q25265", "Felidae");
  for (const char* a : {"Q31"}) T.add_ancestor("Q239", a);
  for (const char* a : {"Q99", "Q30"}) T.add_ancestor("Q65", a);
  for (const char* a : {"Q65", "Q99", "Q30"}) T.add_ancestor("Q1054483", a);
  T.add_ancestor("Q99", "Q30"), T.add_ancestor("Q21", "Q145");
  for (const char* c : {"Q31", "Q30", "Q145", "Q21"}) T.country_like.insert(c), T.coarse.insert(c);
  T.coarse.insert("Q99");  // a first-level subdivision
  auto n = [](const char* x) { return cypha::eval::sources::norm_name(x); };
  // Rule 2, names: an alias, a demonym, the label; case and punctuation as norm_name has them.
  CHECK(wg::match(T, "capital", "Q239", n("Brussels")) == V::Match);
  CHECK(wg::match(T, "capital", "Q239", n("brussels")) == V::Match);
  CHECK(wg::match(T, "nationality", "Q30", n("American")) == V::Match);
  CHECK(wg::match(T, "family", "Q25265", n("Felidae")) == V::Match);
  CHECK(wg::match(T, "capital", "Q239", n("Antwerp")) == V::NoMatch);
  // Rule 3, a containing place: not coarse for birth / death place, location and headquarters; any for located_in.
  CHECK(wg::match(T, "death_place", "Q1054483", n("Los Angeles")) == V::Match);
  CHECK(wg::match(T, "birth_place", "Q1054483", n("L.A.")) == V::Match);
  CHECK(wg::match(T, "death_place", "Q1054483", n("California")) == V::NoMatch);  // coarse
  CHECK(wg::match(T, "death_place", "Q1054483", n("United States of America")) == V::NoMatch);
  CHECK(wg::match(T, "located_in", "Q1054483", n("California")) == V::Match);  // located_in is transitive
  CHECK(wg::match(T, "capital", "Q1054483", n("Los Angeles")) == V::NoMatch);  // capital: names only
  // Rule 4, a finer place: ungraded for the place relations, no match elsewhere.
  CHECK(wg::match(T, "birth_place", "Q65", n("Century City")) == V::Finer);
  CHECK(wg::match(T, "location", "Q65", n("Century City")) == V::Finer);
  CHECK(wg::match(T, "capital", "Q65", n("Century City")) == V::NoMatch);
  // Rule 5, a country-like place inside the truth, for country and nationality only.
  CHECK(wg::match(T, "nationality", "Q145", n("English")) == V::Match);
  CHECK(wg::match(T, "country", "Q145", n("England")) == V::Match);
  CHECK(wg::match(T, "country", "Q30", n("California")) == V::NoMatch);  // inside, not country-like
  CHECK(wg::match(T, "nationality", "Q21", n("British")) == V::NoMatch);  // the wider one does not count
  CHECK(wg::match(T, "occupation", "Q145", n("English")) == V::NoMatch);
  // Empty items or names never match; closed-world relations are the fresh set's.
  CHECK(wg::match(T, "capital", "", n("Brussels")) == V::NoMatch && wg::match(T, "capital", "Q239", "") == V::NoMatch);
  CHECK(wg::closed_world("nationality") && wg::closed_world("successor") && wg::closed_world("class") && !wg::closed_world("occupation") &&
        !wg::closed_world("birth_place"));
  CHECK(wg::place_up("headquarters") && !wg::place_up("capital") && wg::country_relation("nationality") && !wg::country_relation("capital"));
  // The rule that decided a verdict (for the audit).
  int k = -1;
  CHECK(wg::match(T, "capital", "Q239", n("Brussels"), &k) == V::Match && k == 2);
  CHECK(wg::match(T, "death_place", "Q1054483", n("Los Angeles"), &k) == V::Match && k == 3);
  CHECK(wg::match(T, "birth_place", "Q65", n("Century City"), &k) == V::Finer && k == 4);
  CHECK(wg::match(T, "nationality", "Q145", n("English"), &k) == V::Match && k == 5);
  CHECK(wg::match(T, "capital", "Q239", n("Antwerp"), &k) == V::NoMatch && k == 0);
  // A slot's grade (rules 1 and 7): +1 true, 0 false, -1 ungraded; without tables, S8c's grader.
  {
    namespace src = cypha::eval::sources;
    using Vals = std::vector<std::pair<src::Value, int>>;
    T.add_name("Q12892", "Antwerp"), T.add_name("Q12892", "Antwerpen");
    const Vals cap = {{src::make_value("City of Brussels", "Q239"), 1}, {src::make_value("Antwerpen", "Q12892"), 0}};
    const char* w = "";
    auto mv = [](const char* x) { return src::make_value(x); };
    CHECK(wg::grade_slot(&T, "capital", cap, mv("City of Brussels"), &w) == 1 && std::string(w) == "shown");
    CHECK(wg::grade_slot(&T, "capital", cap, mv("Brussels"), &w) == 1 && std::string(w) == "rule2");
    CHECK(wg::grade_slot(nullptr, "capital", cap, mv("Brussels"), &w) == 0 && std::string(w) == "single-valued");  // S8c's grader
    CHECK(wg::grade_slot(&T, "capital", cap, mv("Antwerp"), &w) == 0 && std::string(w) == "drawn-false");  // the false value's alias
    CHECK(wg::grade_slot(nullptr, "capital", cap, mv("Antwerp")) == 0);  // single-valued there
    CHECK(wg::grade_slot(&T, "capital", cap, mv("Ghent"), &w) == 0 && std::string(w) == "single-valued");
    const Vals nat = {{src::make_value("United Kingdom", "Q145"), 1}};
    CHECK(wg::grade_slot(&T, "nationality", nat, mv("British"), &w) == 1 && std::string(w) == "rule2");
    CHECK(wg::grade_slot(&T, "nationality", nat, mv("English"), &w) == 1 && std::string(w) == "rule5");
    CHECK(wg::grade_slot(&T, "nationality", nat, mv("American"), &w) == 0 && std::string(w) == "closed-world");
    CHECK(wg::grade_slot(nullptr, "nationality", nat, mv("American"), &w) == -1 && std::string(w) == "open");
    const Vals occ = {{src::make_value("actor", "Q33999"), 1}};
    CHECK(wg::grade_slot(&T, "occupation", occ, mv("singer"), &w) == -1 && std::string(w) == "open");  // not closed-world
    const Vals born = {{src::make_value("Los Angeles", "Q65"), 1}};
    CHECK(wg::grade_slot(&T, "birth_place", born, mv("Century City"), &w) == -1 && std::string(w) == "finer");  // before single-valued
    CHECK(wg::grade_slot(nullptr, "birth_place", born, mv("Century City")) == 0);
    CHECK(wg::grade_slot(&T, "location", born, mv("Century City"), &w) == -1 && std::string(w) == "finer");  // before closed-world
    CHECK(wg::grade_slot(&T, "location", born, mv("California"), &w) == 0 && std::string(w) == "closed-world");  // coarse
    const Vals dob = {{src::make_value("1950-01-02"), 1}};
    CHECK(wg::grade_slot(&T, "birth_date", dob, mv("1950-01-02"), &w) == 1 && std::string(w) == "shown");
    CHECK(wg::grade_slot(&T, "birth_date", dob, mv("1951"), &w) == 0 && std::string(w) == "single-valued");
    CHECK(wg::grade_slot(&T, "capital", {}, mv("Brussels")) == -1);
  }
  // Loading the tables from files: names normalised, ancestors and flags read.
  const std::filesystem::path tmp = std::filesystem::temp_directory_path() / ("cypha_s8e_wdg_" + std::to_string(std::random_device{}()));
  std::filesystem::create_directories(tmp);
  const std::string dir = tmp.string();
  {
    std::ofstream(dir + "/names.tsv") << "item\tnames\nQ239\tCity of Brussels|Brussels\nQ31\tBelgium\nQ1\tThe Beatles\n";
    std::ofstream(dir + "/anc.tsv") << "item\tancestors\nQ239\tQ31\n";
    std::ofstream(dir + "/flags.tsv") << "item\tcountry_like\tcoarse\nQ31\t1\t1\n";
  }
  wg::Tables L;
  CHECK(L.load(dir));
  CHECK(L.named("Q239", "brussels") && L.named("Q1", "beatles") && L.inside("Q239", "Q31") && L.country_like.count("Q31") && L.coarse.count("Q31"));
  CHECK(L.by_name.count("brussels") && L.by_name.at("brussels").size() == 1 && L.by_name.at("brussels")[0] == "Q239");
  CHECK(wg::match(L, "death_place", "Q239", n("Belgium")) == V::NoMatch);  // its one ancestor is coarse
  wg::Tables E;
  CHECK(!E.load(dir + "/missing") && E.empty());
  std::filesystem::remove_all(tmp);
}

static void test_gap_map() {
  // The gap id CyphaLM writes (cyphalm_trie dktext rows and its dev tags).
  CHECK_STR(gap_id("Honduras", "capital"), "G-AEC1");
  CHECK_STR(gap_id("Chess", "definition"), "G-CFEC");
  CHECK_STR(gap_id("Antares", "definition"), "G-1482");
  // A training tag, as dktext writes it.
  UnknownTag a = parse_unknown_tag(
      "I don't know. I have nothing about Honduras in what I've read. <|unknown reason=UNKNOWN_ENTITY entity=\"Honduras\" relation=capital gap=G-AEC1|>");
  CHECK(a.found && a.closed && a.complete() && a.id_consistent());
  CHECK(a.code == ReasonCode::UnknownEntity);
  CHECK_STR(a.entity, "Honduras");
  CHECK_STR(a.relation, "capital");
  // Fields in another order (A1c's later frames put the entity last).
  UnknownTag b = parse_unknown_tag("x <|unknown reason=UNKNOWN_ENTITY relation=definition gap=G-1482 entity=\"Antares\"|> y");
  CHECK(b.complete() && b.id_consistent());
  CHECK_STR(b.entity, "Antares");
  // A field name inside the quoted entity does not fool the parser.
  UnknownTag q = parse_unknown_tag("<|unknown reason=NO_EVIDENCE entity=\"the relation=x of Y\" relation=author gap=G-0000|>");
  CHECK_STR(q.entity, "the relation=x of Y");
  CHECK_STR(q.relation, "author");
  CHECK(q.complete() && !q.id_consistent());
  // Cut off inside the entity: no entity, not closed, not complete.
  UnknownTag c = parse_unknown_tag("<|unknown reason=UNKNOWN_ENTITY entity=\"Eden Projecond the hosts bleed when pierced");
  CHECK(c.found && !c.closed && c.entity.empty() && !c.complete());
  CHECK_STR(c.reason, "UNKNOWN_ENTITY");
  // Unknown or empty reason names.
  CHECK(!parse_unknown_tag("<|unknown reason= entity=\"X\" relation=r gap=G-0000|>").code.has_value());
  CHECK(!parse_unknown_tag("<|unknown reason=UN entity=\"X\" relation=r gap=G-0000|>").complete());
  CHECK(!parse_unknown_tag("I don't know.").found);
  // A complete tag whose id belongs to another slot.
  UnknownTag d = parse_unknown_tag("<|unknown reason=NO_EVIDENCE entity=\"Hamlet\" relation=author gap=G-0412|>");
  CHECK(d.complete() && !d.id_consistent());
  CHECK_STR(gap_id("Hamlet", "author"), "G-785D");
  for (std::size_t i = 0; i < kReasonCodeCount; ++i)
    CHECK(reason_from_name(reason_name(static_cast<ReasonCode>(i))) == static_cast<ReasonCode>(i));

  // f(T, R).
  CHECK_NEAR((SchemaCell{4, 3}).f(), 0.75, 1e-15);
  CHECK_NEAR(SchemaCell{}.f(), 0.0, 1e-15);

  // The map: a schema gap, demand on it, a new demand gap, a parked gap, VOI.
  GapMap m;
  Gap& s = m.open(GapKind::Schema, "Atlantis", "capital", "country");
  s.p_exists = s.p = 0.9;
  CHECK_STR(s.id, "G-72B1");
  CHECK(&m.open(GapKind::Schema, "Atlantis", "capital") == &m.gaps()[0]);  // same gap
  CHECK(m.size() == 1);
  CHECK(m.find(GapKind::Schema, "ATLANTIS", "capital") != nullptr);       // keys are normalised
  CHECK(m.find(GapKind::Demand, "Atlantis", "capital") == nullptr);
  Gap* g1 = m.add_demand(parse_unknown_tag("<|unknown reason=NO_EVIDENCE entity=\"atlantis\" relation=capital gap=G-0000|>"));
  CHECK(g1 == &m.gaps()[0]);
  CHECK(g1->demand == 1 && g1->status == GapStatus::Queued && g1->kind == GapKind::Schema);
  Gap* g2 = m.add_demand(a, "unknown");
  CHECK(g2 != nullptr && g2->kind == GapKind::Demand && g2->status == GapStatus::Queued && g2->demand == 1);
  CHECK_NEAR(g2->p, 0.5, 1e-15);
  CHECK_STR(g2->type, "unknown");
  CHECK(m.add_demand(a)->demand == 2);  // the same slot again: demand +1, no new gap
  CHECK(m.size() == 2);
  CHECK(m.add_demand(c) == nullptr);    // a cut-off tag names no slot
  Gap& k = m.open(GapKind::Contradiction, "Bob Dylan", "birth_place", "person");
  k.p = 0.55;
  k.status = GapStatus::Parked;
  CHECK(m.add_demand(parse_unknown_tag("<|unknown reason=CONFLICT entity=\"Bob Dylan\" relation=birth_place gap=G-0000|>"))->kind == GapKind::Contradiction);
  CHECK(m.find(GapKind::Contradiction, "Bob Dylan", "birth_place")->status == GapStatus::Parked);  // demand does not unpark
  Gap& cert = m.open(GapKind::Certainty, "Auguste Rodin", "death_date", "person");
  cert.p = 0.974;
  m.score();
  CHECK_NEAR(m.find(GapKind::Schema, "Atlantis", "capital")->eig, expected_information_gain(0.9, 0.9, 0.9).eig, 1e-15);
  CHECK_NEAR(m.find(GapKind::Demand, "Honduras", "capital")->eig, 0.5310044064107189, 1e-12);
  CHECK_NEAR(m.find(GapKind::Demand, "Honduras", "capital")->voi, 0.5310044064107189 * 3.0, 1e-12);  // demand 2
  CHECK_NEAR(m.find(GapKind::Certainty, "Auguste Rodin", "death_date")->voi, expected_information_gain(0.974, 0.9, 0.9).eig, 1e-15);
  const auto rk = m.ranked();
  CHECK(rk.size() == 4);
  CHECK(rk.front()->entity == "Honduras");
  CHECK(rk.back()->entity == "Auguste Rodin");
  for (std::size_t i = 1; i < rk.size(); ++i) CHECK(rk[i - 1]->voi >= rk[i]->voi);
  CHECK_STR(gap_kind_name(GapKind::Contradiction), "contradiction");
  CHECK_STR(gap_status_name(GapStatus::Queued), "queued");
}

// ---------------------------------------------------------------------------
// S4c: the out-of-scope signal read from prose words (unknown_signals.hpp)
// ---------------------------------------------------------------------------

static void test_scope_S4c() {
  // The list rule: '-', a bullet, or digits then '.' or ')' at the start of a line (after blanks).
  const std::string lines = "- apples\n  - pears\n\xE2\x80\xA2 plums\n1. one\n12) two\n1990 was a year\nText - here\n\n7";
  CHECK(ood_list_line(lines, 0));
  CHECK(ood_list_line(lines, lines.find("  - pears")));
  CHECK(ood_list_line(lines, lines.find("\xE2\x80\xA2")));
  CHECK(ood_list_line(lines, lines.find("1. one")));
  CHECK(ood_list_line(lines, lines.find("12) two")));
  CHECK(!ood_list_line(lines, lines.find("1990 was")));
  CHECK(!ood_list_line(lines, lines.find("Text")));
  CHECK(!ood_list_line(lines, lines.find("\n\n7") + 1));  // an empty line
  CHECK(!ood_list_line(lines, lines.size() - 1));          // digits with nothing after them
  CHECK(!ood_list_line(lines, lines.size()));

  // The reference: F by count, ties by the word; the other words' share last.
  OodRef ref;
  for (const char* w : {"the", "the", "the", "the", "of", "of", "and", "and", "cat", "dog"}) ref.add(w);
  ref.finish(3);
  CHECK(ref.N == 10 && ref.top.size() == 3);
  CHECK_STR(ref.top[0], "the");
  CHECK_STR(ref.top[1], "and");  // ties (2 each) by the word
  CHECK_STR(ref.top[2], "of");
  CHECK_NEAR(ref.share[0], 0.4, 1e-12);
  CHECK_NEAR(ref.share[3], 0.2, 1e-12);
  CHECK(ref.in_top("of") && !ref.in_top("cat"));
  CHECK_NEAR(ref.freq("cat"), 0.1, 1e-12);
  CHECK_NEAR(ref.freq("emu"), 0.0, 1e-12);

  // O1: a window with the reference's shares has G 0; a window of one word has G = 2 ln(1 / p).
  const std::vector<std::string> same = {"the", "the", "the", "the", "of", "of", "and", "and", "cat", "emu"};
  CHECK_NEAR(ood_register(same, ref), 0.0, 1e-12);
  CHECK_NEAR(ood_register(std::vector<std::string>(10, "of"), ref), 2.0 * std::log(1.0 / 0.2), 1e-12);
  CHECK_NEAR(ood_register(std::vector<std::string>(10, "zebra"), ref), 2.0 * std::log(1.0 / 0.2), 1e-12);  // the other bin
  CHECK(ood_register({}, ref) == 0.0);

  // O2: the mean -ln p over words outside F, with p = (n + 0.5) / (N + 0.5 (V + 1)); V = 5 distinct words.
  const double den = 10.0 + 0.5 * 6.0;
  CHECK_NEAR(ood_rare({"the", "cat", "emu"}, ref), (-std::log(1.5 / den) - std::log(0.5 / den)) / 2.0, 1e-12);
  CHECK(ood_rare({"the", "of"}, ref) == 0.0);

  // O3: the page vector skips F and words with no vector; distance 1 - the best cosine.
  const float vcat[2] = {1.0f, 0.0f}, vdog[2] = {0.0f, 1.0f};
  auto vec = [&](const std::string& w) -> const float* { return w == "cat" ? vcat : w == "dog" ? vdog : nullptr; };
  const auto pc = ood_page_vector({"the", "cat", "emu"}, ref, 2, vec);
  CHECK(pc.size() == 2);
  CHECK_NEAR(pc[0], 1.0, 1e-12);
  CHECK(ood_page_vector({"the", "emu"}, ref, 2, vec).empty());
  const auto pcd = ood_page_vector({"cat", "dog"}, ref, 2, vec);  // equal weights (equal shares): the diagonal
  CHECK_NEAR(pcd[0], std::sqrt(0.5), 1e-12);
  CHECK_NEAR(pcd[1], std::sqrt(0.5), 1e-12);
  const std::vector<std::vector<double>> bank = {{0.0, 1.0}, {std::sqrt(0.5), std::sqrt(0.5)}};
  CHECK_NEAR(ood_topic_distance(pc, bank), 1.0 - std::sqrt(0.5), 1e-12);
  CHECK_NEAR(ood_topic_distance(pcd, bank), 0.0, 1e-12);
  CHECK(ood_topic_distance({}, bank) == 1.0 && ood_topic_distance(pc, {}) == 1.0);

  // O4: windows of one fixed composition give mean = share and sd at its binomial floor; such a window scores 0.
  {
    OodRef r2;
    std::vector<std::vector<std::string>> pages;
    for (int p = 0; p < 3; ++p) {
      std::vector<std::string> w;
      for (int k = 0; k < 10; ++k)
        for (const char* x : {"a", "a", "b", "c", "d"}) w.push_back(x), r2.add(x);
      pages.push_back(w);
    }
    r2.finish(2);
    OodProfile pr;
    pr.build(r2, pages, 2, 5);
    CHECK(pr.windows == 30 && pr.mu.size() == 3);
    CHECK_NEAR(pr.mu[pr.bin.at("a")], 0.4, 1e-12);
    CHECK_NEAR(pr.sd[pr.bin.at("a")], std::sqrt(0.4 * 0.6 / 5.0), 1e-12);
    CHECK_NEAR(ood_delta({"a", "b", "a", "c", "d"}, pr), 0.0, 1e-12);
    const double z = (std::fabs(0.0 - 0.4) / std::sqrt(0.24 / 5) + std::fabs(0.0 - 0.2) / std::sqrt(0.16 / 5) + std::fabs(1.0 - 0.4) / std::sqrt(0.24 / 5)) / 3.0;
    CHECK_NEAR(ood_delta({"c", "c", "d", "e", "f"}, pr), z, 1e-12);
  }

  // O5: TF-IDF novelty against the bank pages; F's words never count.
  {
    OodTfidf tf;
    tf.build(ref, {{"apple", "banana", "the"}, {"cherry", "date", "date"}});
    CHECK(tf.pages == 2);
    CHECK_NEAR(tf.novelty({"apple", "banana", "of"}, ref), 0.0, 1e-12);
    CHECK_NEAR(tf.novelty({"zebra", "yak"}, ref), 1.0, 1e-12);
    CHECK_NEAR(tf.novelty({"the", "of"}, ref), 1.0, 1e-12);
    const double ia = std::log(3.0 / 2.0) + 1.0, iz = std::log(3.0) + 1.0;  // apple on 1 of 2 pages; zebra on none
    CHECK_NEAR(tf.novelty({"apple", "zebra"}, ref), 1.0 - ia / std::sqrt(ia * ia + iz * iz) * std::sqrt(0.5), 1e-12);
  }

  // End to end: windows drawn from the reference language are silent; a planted language with another register
  // (function words used in other shares) is detected by O1 through S4b's rule.
  {
    std::mt19937_64 rng(4041);
    const std::vector<std::string> fwords = {"the", "of", "and", "in", "to", "a", "is", "was", "you", "can"};
    const std::vector<double> pa = {0.30, 0.18, 0.14, 0.12, 0.10, 0.08, 0.04, 0.03, 0.005, 0.005};  // wiki-like
    const std::vector<double> pb = {0.12, 0.05, 0.10, 0.06, 0.12, 0.10, 0.12, 0.01, 0.16, 0.16};    // another register
    auto draw = [&](const std::vector<double>& p, std::size_t n, double content) {
      std::discrete_distribution<int> fd(p.begin(), p.end());
      std::uniform_real_distribution<double> U(0, 1);
      std::uniform_int_distribution<int> cw(0, 4999);
      std::vector<std::string> w;
      for (std::size_t i = 0; i < n; ++i) w.push_back(U(rng) < content ? "w" + std::to_string(cw(rng)) : fwords[static_cast<std::size_t>(fd(rng))]);
      return w;
    };
    OodRef r;
    for (int i = 0; i < 400; ++i)
      for (const auto& w : draw(pa, 200, 0.5)) r.add(w);
    r.finish(10);
    std::vector<SignalUnit> C, S, A, B;
    for (int i = 0; i < 600; ++i) C.push_back({ood_register(draw(pa, kOodWindow, 0.5), r), 0, signal_unit_id("C", static_cast<std::size_t>(i))});
    for (int i = 0; i < 1500; ++i) S.push_back({ood_register(draw(pa, kOodWindow, 0.5), r), 0, signal_unit_id("S", static_cast<std::size_t>(i))});
    for (int i = 0; i < 400; ++i) A.push_back({ood_register(draw(pa, kOodWindow, 0.5), r), 0, signal_unit_id("A", static_cast<std::size_t>(i))});
    for (int i = 0; i < 400; ++i) B.push_back({ood_register(draw(pb, kOodWindow, 0.5), r), 0, signal_unit_id("B", static_cast<std::size_t>(i))});
    const SignalRule rule;
    const SignalRun run = signal_run(C, S, 1, true, rule);
    CHECK(run.alarms.size() <= 1);
    std::size_t hitB = 0, hitA = 0;
    std::uniform_int_distribution<std::size_t> T(100, S.size() - 200);
    for (int rep = 0; rep < 100; ++rep) {
      const std::size_t t0 = T(rng);
      const SignalDetection db = signal_planted(C, S, B, 1, true, rule, run, t0, 1.0, 200, rng);
      const SignalDetection da = signal_planted(C, S, A, 1, true, rule, run, t0, 1.0, 200, rng);
      hitB += db.delay > 0 && db.delay <= 50;
      hitA += da.delay > 0 && da.delay <= 50;
    }
    CHECK(hitB >= 90);  // the other register: detected within 50 units
    CHECK(hitA <= 5);   // more of the reference language: silent
  }
}

int main() {
  test_degree_of_conflict();
  test_classify();
  test_split_uncertainty();
  test_reason_codes();
  test_estimators();
  test_unseen_s4();
  test_cards();
  test_voi_and_tiers();
  test_lifecycle();
  test_firewall_I1();
  test_promotion_I4();
  test_refutation_I7();
  test_labels_I8();
  test_scoring_S6();
  test_tests_S7();
  test_s7b();
  test_links_S8b();
  test_gen_rules_S8a();
  test_s7c();
  test_worlds_S9();
  test_decide_S8d();
  test_signals_S4b();
  test_sources_S8c();
  test_gap_map();
  test_scope_S4c();
  test_curiosity_S10();
  test_concepts_S11a();
  test_date_links_S4bb();
  test_analogy_S11c();
  test_fragile_S9b();
  test_laws_deep_S12a();
  test_abc_closed_S6c();
  test_threshold_S4d();
  test_wd_grader_S8e();
  std::printf("epistemic_test: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
