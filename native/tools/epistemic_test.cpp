// Known-answer tests for cypha/eval/epistemic.hpp (docs/DISCOVERY.md sections 2.1-2.3, 3.1, 3.3, 10).
// Returns nonzero if any check fails.
#include "cypha/eval/epistemic.hpp"

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

int main() {
  test_degree_of_conflict();
  test_classify();
  test_split_uncertainty();
  test_reason_codes();
  test_estimators();
  test_cards();
  std::printf("epistemic_test: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
