// Known-answer tests for cypha/eval/epistemic.hpp (docs/DISCOVERY.md sections 2.1-2.3, 3.1, 3.3, 10), hypothesis.hpp
// (S5: 4, 6.3, 7.2-7.5; invariants I1, I4, I7, I8) and gap_map.hpp (S3: 3.2, 3.5).
// Returns nonzero if any check fails.
#include "cypha/eval/epistemic.hpp"
#include "cypha/eval/gap_map.hpp"
#include "cypha/eval/hypothesis.hpp"

#include <cmath>
#include <cstdio>
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
  test_gap_map();
  std::printf("epistemic_test: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
