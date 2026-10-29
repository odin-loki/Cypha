// Known-answer tests for cypha/eval/epistemic.hpp (docs/DISCOVERY.md sections 2.1-2.3, 3.1, 3.3, 10), hypothesis.hpp
// (S5: 4, 6.3, 7.2-7.5; invariants I1, I4, I7, I8; S7: 7.1-7.3, I5, I6; S7b: three-way likelihoods, the batch step
// among the supported), gap_map.hpp (S3: 3.2, 3.5), gen_links.hpp (S8b: G6, G7), gen_rules.hpp (S8a: G2, G14), sources.hpp
// (S8c: evidence sources) and worlds.hpp (S9); S7c's batch rules.
// Returns nonzero if any check fails.
#include "cypha/eval/epistemic.hpp"
#include "cypha/eval/gap_map.hpp"
#include "cypha/eval/gen_links.hpp"
#include "cypha/eval/gen_rules.hpp"
#include "cypha/eval/hypothesis.hpp"
#include "cypha/eval/sources.hpp"
#include "cypha/eval/unknown_signals.hpp"
#include "cypha/eval/worlds.hpp"

#include <cmath>
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
  std::printf("epistemic_test: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
