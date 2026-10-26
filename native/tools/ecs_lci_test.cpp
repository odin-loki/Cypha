// Known-answer tests for cypha/eval/ecs.hpp and cypha/eval/lci.hpp (PLAN sections 7 and 8).
// Returns nonzero if any check fails.
#include "cypha/eval/ecs.hpp"
#include "cypha/eval/lci.hpp"

#include <cmath>
#include <cstdio>
#include <random>
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

static void test_beta_and_wilson() {
  CHECK_NEAR(beta_mean(2, 5), 2.0 / 7.0, 1e-12);
  CHECK_NEAR(beta_variance(2, 5), 10.0 / (49.0 * 8.0), 1e-12);
  // Beta(2,2) CDF = 3x^2 - 2x^3.
  CHECK_NEAR(reg_inc_beta(0.25, 2, 2), 3 * 0.0625 - 2 * 0.015625, 1e-12);
  CHECK_NEAR(reg_inc_beta(0.5, 3.5, 3.5), 0.5, 1e-12);
  // Beta(2,5): CDF = 1 - [(1-x)^6 + 6x(1-x)^5]; median ~ 0.26445.
  CHECK_NEAR(beta_quantile(0.5, 2, 5), 0.26445, 2e-5);
  CHECK_NEAR(reg_inc_beta(beta_quantile(0.9, 2, 5), 2, 5), 0.9, 1e-10);
  // Beta(1,1) is uniform.
  const Interval u = beta_credible_interval(1, 1, 0.9);
  CHECK_NEAR(u.lo, 0.05, 1e-9);
  CHECK_NEAR(u.hi, 0.95, 1e-9);
  // Beta(41,1): CDF = x^41, 5% quantile = 0.05^(1/41).
  CHECK_NEAR(beta_quantile(0.05, 41, 1), std::pow(0.05, 1.0 / 41.0), 1e-10);
  // Wilson 90/100.
  CHECK_NEAR(wilson_lower(90, 100), 0.8256, 5e-4);
  CHECK_NEAR(wilson_upper(90, 100), 0.9448, 5e-4);
  CHECK_NEAR(wilson_lower(0, 0), 0.0, 1e-12);
  CHECK(wilson_upper(100, 100) <= 1.0 && wilson_lower(100, 100) > 0.96);
  // Source-trust priors and updates.
  CHECK_NEAR(prior_wikidata().mean(), 50.0 / 51.0, 1e-12);
  CHECK_NEAR(prior_distilled_nn().mean(), 0.25, 1e-12);
  BetaPosterior t = prior_unknown_source();
  t.observe(true);
  t.observe(true);
  t.observe(false);
  CHECK_NEAR(t.mean(), 3.0 / 5.0, 1e-12);
}

static void test_evidence_factors() {
  CHECK_NEAR(independence_weight(1), 1.0, 1e-12);
  CHECK_NEAR(independence_weight(3), (1.0 + std::log(3.0)) / 3.0, 1e-12);
  CHECK_NEAR(3.0 * independence_weight(3), 1.0 + std::log(3.0), 1e-12);  // cluster total 1 + ln n
  const std::vector<double> w = independence_weights({7, 7, 7, 9});
  CHECK_NEAR(w[0], independence_weight(3), 1e-12);
  CHECK_NEAR(w[3], 1.0, 1e-12);
  CHECK_NEAR(time_decay(10.0, 10.0), std::exp(-1.0), 1e-12);
  CHECK_NEAR(time_decay(5.0, std::numeric_limits<double>::infinity()), 1.0, 1e-12);
  CHECK_NEAR(time_decay(-3.0, 10.0), 1.0, 1e-12);
  CHECK_NEAR(derivation_factor({0.9, 0.8}, 0.95, 2), 0.9 * 0.8 * 0.95, 1e-12);
  CHECK_NEAR(derivation_factor({0.9}, 0.95, 0), 1.0, 1e-12);       // observed, not derived
  CHECK_NEAR(derivation_factor({0.99}, 0.99, 3), 0.99 * 0.99, 1e-12);
  CHECK_NEAR(derivation_factor({0.99}, 0.99, 4), 0.0, 1e-12);      // depth cap 3
  CHECK(derivation_depth({0, 2, 1}) == 3);
  EvidenceItem e;
  e.omega = 0.5;
  e.tau = 0.9;
  e.pi = 0.8;
  e.theta = 0.7;
  e.delta = 0.6;
  CHECK_NEAR(e.mass(), 0.5 * 0.9 * 0.8 * 0.7 * 0.6, 1e-12);
  const EvidenceItem m = make_evidence(Stance::Support, prior_wikidata(), 0.9);
  CHECK_NEAR(m.mass(), (50.0 / 51.0) * 0.9, 1e-12);
}

static void test_fusion_and_grades() {
  // No evidence: P = a, u = 1.
  {
    FusionConfig c;
    c.base_rate = 0.5;
    const Opinion o = fuse({}, c);
    CHECK_NEAR(o.P, 0.5, 1e-12);
    CHECK_NEAR(o.u, 1.0, 1e-12);
    CHECK_NEAR(o.b, 0.0, 1e-12);
    CHECK_NEAR(o.d, 0.0, 1e-12);
    CHECK(o.grade == Grade::Low);  // P >= 0.5
    c.base_rate = 0.3;
    const Opinion o2 = fuse({}, c);
    CHECK_NEAR(o2.P, 0.3, 1e-12);
    CHECK_NEAR(o2.u, 1.0, 1e-12);
    CHECK(o2.grade == Grade::VeryLow);
    CHECK_NEAR(o2.alpha, 0.6, 1e-12);  // Beta(aW, (1-a)W)
    CHECK_NEAR(o2.beta, 1.4, 1e-12);
  }
  // Strong positive evidence: 40 unit-mass items -> High.
  {
    std::vector<EvidenceItem> items(40);
    const Opinion o = fuse(items);
    CHECK_NEAR(o.r, 40.0, 1e-12);
    CHECK_NEAR(o.b, 40.0 / 42.0, 1e-12);
    CHECK_NEAR(o.u, 2.0 / 42.0, 1e-12);
    CHECK_NEAR(o.P, 40.0 / 42.0 + 0.5 * 2.0 / 42.0, 1e-12);  // = 41/42 = Beta(41,1) mean
    CHECK_NEAR(o.lo, std::pow(0.05, 1.0 / 41.0), 1e-9);
    CHECK(o.grade == Grade::High);
    CHECK(admission_decision(o) == Admission::Admit);
    CHECK(usable_as_premise(o));
  }
  // Moderate: 20 items -> lower bound 0.867 < 0.9, P 0.9545, u 0.09.
  {
    std::vector<EvidenceItem> items(20);
    const Opinion o = fuse(items);
    CHECK(o.grade == Grade::Moderate);
    CHECK(admission_decision(o) == Admission::Admit);
    CHECK(!usable_as_premise(o));
  }
  // Contradicted functional fact: 1 support vs 10 items for a different value,
  // phi = 0.95 -> s = 9.5, d = 9.5/12.5 = 0.76.
  {
    std::vector<EvidenceItem> items(1);
    for (int i = 0; i < 10; ++i) {
      EvidenceItem c;
      c.stance = Stance::CompetingValue;
      items.push_back(c);
    }
    FusionConfig cfg;
    cfg.phi = 0.95;
    const Opinion o = fuse(items, cfg);
    CHECK_NEAR(o.s, 9.5, 1e-12);
    CHECK_NEAR(o.d, 9.5 / 12.5, 1e-12);
    CHECK(o.grade == Grade::Contradicted);
    CHECK(admission_decision(o) == Admission::Reject);
    // Non-functional relation (phi = 0, e.g. "child"): competing values do not count against it.
    cfg.phi = 0.0;
    const Opinion o2 = fuse(items, cfg);
    CHECK_NEAR(o2.s, 0.0, 1e-12);
    CHECK_NEAR(o2.P, (1.0 + 0.5 * 2.0) / 3.0, 1e-12);
    CHECK(o2.grade == Grade::Low);  // P = 0.667, u = 0.667
    CHECK(admission_decision(o2) == Admission::Candidate);
  }
  // Constraint penalties: one hard violation adds 5, one soft adds 1.
  {
    FusionConfig cfg;
    cfg.hard_violations = 1;
    const Opinion o = fuse({}, cfg);
    CHECK_NEAR(o.s, 5.0, 1e-12);
    CHECK_NEAR(o.d, 5.0 / 7.0, 1e-12);
    CHECK(o.grade == Grade::Contradicted);
    cfg.hard_violations = 0;
    cfg.soft_violations = 2;
    CHECK_NEAR(fuse({}, cfg).s, 2.0, 1e-12);
  }
  // Opposing evidence is not scaled by phi.
  {
    EvidenceItem opp;
    opp.stance = Stance::Oppose;
    FusionConfig cfg;
    cfg.phi = 0.0;
    CHECK_NEAR(fuse({opp, opp}, cfg).s, 2.0, 1e-12);
  }
  // Grade boundaries.
  CHECK(grade_from(0.6, 0.5, 0.2, 0.2) == Grade::Low);
  CHECK(grade_from(0.8, 0.2, 0.0, 0.7) == Grade::Moderate);
  CHECK(grade_from(0.8, 0.4, 0.0, 0.5) == Grade::Low);
  CHECK(grade_from(0.4, 0.5, 0.1, 0.1) == Grade::VeryLow);
  CHECK(grade_from(0.4, 0.2, 0.5, 0.1) == Grade::Contradicted);
  CHECK(std::string(grade_name(Grade::VeryLow)) == "Very low");
}

static void test_answer_confidence_and_abstain() {
  CHECK_NEAR(answer_confidence(0.9, 0.95, {0.9, 0.8}, 0.99), 0.9 * 0.95 * 0.9 * 0.8 * 0.99, 1e-12);
  const std::vector<double> conf = {0.95, 0.9, 0.8, 0.7, 0.6};
  const std::vector<int> ok = {1, 1, 1, 0, 1};
  const std::vector<RiskCoveragePoint> curve = risk_coverage_curve(conf, ok);
  CHECK(curve.size() == 5);
  CHECK_NEAR(curve[3].accuracy, 0.75, 1e-12);
  CHECK_NEAR(curve[4].coverage, 1.0, 1e-12);
  const AbstainChoice a = abstain_threshold(conf, ok, 0.9);
  CHECK(a.found);
  CHECK_NEAR(a.threshold, 0.8, 1e-12);
  CHECK_NEAR(a.coverage, 0.6, 1e-12);
  const AbstainChoice b = abstain_threshold(conf, ok, 0.8);
  CHECK(b.found && std::fabs(b.threshold - 0.6) < 1e-12);  // all five answered, accuracy 0.8
  CHECK(!abstain_threshold({0.9}, {0}, 0.9).found);
  CHECK(use_in_answer(0.8, 0.8) && !use_in_answer(0.79, 0.8));
}

static void test_calibration() {
  // Perfectly calibrated: in each bin accuracy equals the stated probability exactly.
  std::vector<double> p;
  std::vector<int> y;
  for (int bin = 0; bin < 10; ++bin) {
    const double conf = (bin + 0.5) / 10.0;  // 0.05, 0.15, ..., 0.95
    for (int i = 0; i < 20; ++i) {
      p.push_back(conf);
      y.push_back(i < static_cast<int>(std::lround(conf * 20.0)) ? 1 : 0);
    }
  }
  CHECK_NEAR(expected_calibration_error(p, y), 0.0, 1e-9);
  const std::vector<ReliabilityBin> bins = reliability_bins(p, y);
  CHECK(bins.size() == 10 && bins[3].count == 20);
  CHECK_NEAR(bins[3].mean_conf, 0.35, 1e-12);
  CHECK_NEAR(bins[3].accuracy, 0.35, 1e-12);
  // Overconfident: p = 0.9, 50% right -> ECE 0.4.
  std::vector<double> p2(100, 0.9);
  std::vector<int> y2(100, 0);
  for (int i = 0; i < 50; ++i) y2[static_cast<std::size_t>(i)] = 1;
  CHECK_NEAR(expected_calibration_error(p2, y2), 0.4, 1e-12);
  // p = 1.0 lands in the last bin.
  CHECK(reliability_bins({1.0}, {1}).back().count == 1);
  // Brier.
  CHECK_NEAR(brier_score({1.0, 0.0}, {1, 0}), 0.0, 1e-12);
  CHECK_NEAR(brier_score({0.5, 0.5}, {1, 0}), 0.25, 1e-12);
  CHECK_NEAR(brier_score({0.9}, {0}), 0.81, 1e-12);

  // PAV on a known sequence.
  const std::vector<double> f = isotonic_pav({1, 3, 2, 3, 5, 4, 6});
  const std::vector<double> want = {1, 2.5, 2.5, 3, 4.5, 4.5, 6};
  CHECK(f.size() == want.size());
  for (std::size_t i = 0; i < want.size() && i < f.size(); ++i) CHECK_NEAR(f[i], want[i], 1e-12);
  // Fully decreasing pools to the mean; weights respected.
  const std::vector<double> g = isotonic_pav({3, 2, 1});
  CHECK_NEAR(g[0], 2.0, 1e-12);
  CHECK_NEAR(g[2], 2.0, 1e-12);
  const std::vector<double> gw = isotonic_pav({3, 1}, {1, 3});
  CHECK_NEAR(gw[0], 1.5, 1e-12);  // (3*1 + 1*3) / 4 = 1.5

  // Isotonic calibrator: monotone, repairs overconfidence.
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::vector<double> rp;
  std::vector<int> ry;
  for (int i = 0; i < 4000; ++i) {
    const double raw = U(rng);
    const double truth = raw * raw;  // true probability is raw^2: raw is overconfident
    rp.push_back(raw);
    ry.push_back(U(rng) < truth ? 1 : 0);
  }
  IsotonicCalibrator iso;
  iso.fit(rp, ry);
  CHECK(iso.fitted());
  bool mono = true;
  double prev = -1.0;
  for (int i = 0; i <= 100; ++i) {
    const double v = iso.predict(i / 100.0);
    mono = mono && (v >= prev - 1e-12);
    prev = v;
  }
  CHECK(mono);
  std::vector<double> cal;
  for (double v : rp) cal.push_back(iso.predict(v));
  CHECK(expected_calibration_error(cal, ry) < expected_calibration_error(rp, ry));
  CHECK(expected_calibration_error(cal, ry) < 0.05);
}

static void test_admission_rules() {
  CHECK(!pattern_admissible(19, 19));       // fewer than 20 checks
  CHECK(pattern_admissible(20, 20));        // Wilson lower of 20/20 ~ 0.839
  CHECK(!pattern_admissible(14, 20));       // 0.70 precision, lower bound ~0.48
  CHECK(pattern_admissible(95, 100));       // lower bound ~0.888
  const AuditOutcome good = weekly_audit(98, 100);
  CHECK(!good.pause_admission);
  CHECK_NEAR(good.precision, 0.98, 1e-12);
  const AuditOutcome bad = weekly_audit(94, 100);
  CHECK(bad.pause_admission);  // lower bound ~0.875 < 0.90
  // Admit needs both P >= 0.8 and grade >= Moderate; high P with large u stays a candidate.
  Opinion o;
  o.P = 0.85;
  o.grade = Grade::Low;
  CHECK(admission_decision(o) == Admission::Candidate);
  o.grade = Grade::Moderate;
  CHECK(admission_decision(o) == Admission::Admit);
  o.P = 0.6;
  CHECK(admission_decision(o) == Admission::Candidate);
  o.P = 0.49;
  CHECK(admission_decision(o) == Admission::Reject);
}

static void test_bootstrap() {
  std::mt19937_64 rng(11);
  std::normal_distribution<double> N01(0.0, 1.0);
  // Clear gain: mean 0.05, sd 0.1 over 500 items, eps 0.005 -> P ~ 1.
  std::vector<double> gain;
  for (int i = 0; i < 500; ++i) gain.push_back(0.05 + 0.1 * N01(rng));
  const BootstrapResult a = bayesian_bootstrap(gain, 0.005);
  CHECK(a.p_gt > 0.999);
  CHECK(a.n_eff == 500 && a.draws == 4000);
  CHECK(a.lo90 < a.mean && a.mean < a.hi90);
  // Same seed -> identical result; different seed -> still ~1.
  CHECK_NEAR(bayesian_bootstrap(gain, 0.005).p_gt, a.p_gt, 0.0);
  // Zero mean difference (symmetric +/-1) -> P ~ 0.5.
  std::vector<double> zero;
  for (int i = 0; i < 400; ++i) zero.push_back(i % 2 ? 1.0 : -1.0);
  const BootstrapResult z = bayesian_bootstrap(zero, 0.0);
  CHECK(z.p_gt > 0.44 && z.p_gt < 0.56);
  CHECK_NEAR(z.mean, 0.0, 0.02);
  // A clear loss gives P ~ 0.
  std::vector<double> loss;
  for (int i = 0; i < 500; ++i) loss.push_back(-0.05 + 0.1 * N01(rng));
  CHECK(bayesian_bootstrap(loss, 0.0).p_gt < 0.001);
  // Cluster-level resampling: 20 clusters of 25 near-duplicates with a cluster-wide
  // effect. Item-level bootstrap is overconfident, cluster-level is honest.
  std::vector<double> cd;
  std::vector<int> cid;
  for (int k = 0; k < 20; ++k) {
    const double effect = 0.02 + 0.2 * N01(rng);
    for (int i = 0; i < 25; ++i) {
      cd.push_back(effect);
      cid.push_back(k);
    }
  }
  const BootstrapResult item_level = bayesian_bootstrap(cd, 0.0);
  const BootstrapResult cluster_level = bayesian_bootstrap(cd, 0.0, cid);
  CHECK(cluster_level.n_eff == 20);
  CHECK(cluster_level.n_items == 500);
  CHECK((cluster_level.hi90 - cluster_level.lo90) > 2.0 * (item_level.hi90 - item_level.lo90));
  // n_eff cap.
  CHECK_NEAR(n_eff_cap(199), 0.5, 0.0);
  CHECK_NEAR(n_eff_cap(200), 1.0, 0.0);
  // size mismatch throws
  bool threw = false;
  try {
    bayesian_bootstrap({1.0, 2.0}, 0.0, {1});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

static std::vector<double> noisy(double mean, double sd, int n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> N01(0.0, 1.0);
  std::vector<double> v;
  for (int i = 0; i < n; ++i) v.push_back(mean + sd * N01(rng));
  return v;
}

static void test_lci_components() {
  // Canaries.
  CanaryInput ok{"D01", noisy(0.0, 0.05, 400, 1), {}, 0.01};
  CanaryInput bad{"T-QA", noisy(-0.05, 0.05, 400, 2), {}, 0.01};
  const CanaryResult rok = canary_noninferiority(ok);
  const CanaryResult rbad = canary_noninferiority(bad);
  CHECK(rok.pass && rok.p_noninferior >= 0.95);
  CHECK(!rbad.pass && rbad.p_noninferior < 0.01);
  CHECK_NEAR(canary_r({rok}), 1.0, 0.0);
  CHECK_NEAR(canary_r({rok, rbad}), 0.0, 0.0);
  CHECK_NEAR(canary_r({}), 1.0, 0.0);
  CHECK(!canary_noninferiority(CanaryInput{"empty", {}, {}, 0.01}).pass);
  // Generalisation gamma.
  CHECK_NEAR(generalisation_gamma(0.5, 1.0), 1.0, 1e-12);   // G = 0.5 -> 1
  CHECK_NEAR(generalisation_gamma(0.25, 1.0), 0.5, 1e-12);  // G = 0.25 -> 0.5
  CHECK_NEAR(generalisation_gamma(0.0, 1.0), 0.0, 1e-12);   // memorisation
  CHECK_NEAR(generalisation_gamma(3.0, 1.0), 1.0, 1e-12);   // clipped
  CHECK_NEAR(generalisation_gamma(-1.0, 1.0), 0.0, 1e-12);
  CHECK_NEAR(generalisation_gamma(1.0, 0.0), 0.0, 1e-12);
  // Calibration factor.
  CHECK_NEAR(calibration_factor(0.04, 0.045, 0.10, 0.10), 1.0, 0.0);
  CHECK_NEAR(calibration_factor(0.04, 0.06, 0.10, 0.10), 0.5, 0.0);
  CHECK_NEAR(calibration_factor(0.04, 0.04, 0.10, 0.11), 0.5, 0.0);
  // Stability.
  CHECK_NEAR(stability_sigma({0.01, 0.02, 0.015}), 1.0, 0.0);
  CHECK_NEAR(stability_sigma({0.01, -0.02, 0.015}), 0.5, 0.0);
  CHECK_NEAR(stability_sigma({0.01, 0.02}), 0.5, 0.0);  // fewer than 3 seeds
  // Cost.
  CHECK_NEAR(cost_kappa(100, 200, 5, 10), 1.0, 0.0);
  CHECK_NEAR(cost_kappa(300, 200, 5, 10), 0.0, 0.0);
  CHECK_NEAR(cost_kappa(100, 200, 50, 10), 0.0, 0.0);
}

static LciInputs good_inputs(int n_items) {
  LciInputs in;
  in.test_diffs = noisy(0.08, 0.1, n_items, 5);
  in.eps = 0.005;
  in.canaries.push_back(CanaryInput{"D01", noisy(0.0, 0.05, 400, 1), {}, 0.01});
  in.canaries.push_back(CanaryInput{"D10A", noisy(0.002, 0.05, 400, 3), {}, 0.01});
  in.gain_dev = 0.08;
  in.gain_ext = 0.06;
  in.ece_before = 0.04;
  in.ece_after = 0.04;
  in.brier_before = 0.10;
  in.brier_after = 0.09;
  in.seed_gains = {0.078, 0.082, 0.08};
  in.ram = 100;
  in.ram_budget = 200;
  in.latency = 5;
  in.latency_budget = 10;
  return in;
}

static void test_lci_decision() {
  const LciDecision d = compute_lci(good_inputs(500));
  CHECK(d.p_gain > 0.999);
  CHECK_NEAR(d.r, 1.0, 0.0);
  CHECK_NEAR(d.gamma, 1.0, 1e-12);
  CHECK_NEAR(d.calibration, 1.0, 0.0);
  CHECK_NEAR(d.sigma, 1.0, 0.0);
  CHECK_NEAR(d.kappa, 1.0, 0.0);
  CHECK(!d.n_eff_capped && d.n_eff == 500);
  CHECK_NEAR(d.lci, d.product, 1e-15);
  CHECK(d.lci >= 0.95 && d.meets_threshold);
  CHECK(promote(d, true) && !promote(d, false));
  CHECK(d.components().size() == 9 + 2);  // 9 named components + one per canary
  // n_eff < 200: capped at 0.5 even with an overwhelming gain.
  const LciDecision small = compute_lci(good_inputs(150));
  CHECK(small.p_gain > 0.999);
  CHECK(small.n_eff_capped && small.n_eff == 150);
  CHECK(small.product > 0.9);
  CHECK(small.lci <= 0.5 + 1e-15);
  CHECK(!small.meets_threshold);
  // Cluster-aware n_eff: 1000 items in 150 clusters -> n_eff 150 -> capped.
  LciInputs ci = good_inputs(1000);
  for (int i = 0; i < 1000; ++i) ci.test_clusters.push_back(i % 150);
  const LciDecision cd = compute_lci(ci);
  CHECK(cd.n_items == 1000 && cd.n_eff == 150 && cd.n_eff_capped);
  CHECK(cd.lci <= 0.5 + 1e-15);
  // A failing canary zeroes LCI.
  LciInputs bad = good_inputs(500);
  bad.canaries.push_back(CanaryInput{"T-QA", noisy(-0.05, 0.05, 400, 9), {}, 0.01});
  const LciDecision bd = compute_lci(bad);
  CHECK_NEAR(bd.r, 0.0, 0.0);
  CHECK_NEAR(bd.lci, 0.0, 0.0);
  // Memorisation: big dev gain, no external gain -> gamma 0 -> LCI 0.
  LciInputs mem = good_inputs(500);
  mem.gain_ext = 0.0;
  CHECK_NEAR(compute_lci(mem).lci, 0.0, 0.0);
  // Cost over budget -> LCI 0. Unstable seeds -> halved.
  LciInputs cost = good_inputs(500);
  cost.ram = 500;
  CHECK_NEAR(compute_lci(cost).lci, 0.0, 0.0);
  LciInputs unstable = good_inputs(500);
  unstable.seed_gains = {0.05, -0.01, 0.04};
  const LciDecision ud = compute_lci(unstable);
  CHECK_NEAR(ud.lci, 0.5 * ud.p_gain, 1e-12);
  CHECK(!ud.meets_threshold);
}

static void test_multiplicity() {
  // Alpha-spending schedule.
  CHECK_NEAR(alpha_spend(1), 0.05 * 0.1, 1e-15);
  CHECK_NEAR(alpha_spend(2), 0.05 * 0.1 * 0.9, 1e-15);
  CHECK_NEAR(alpha_spend(10), 0.05 * 0.1 * std::pow(0.9, 9), 1e-15);
  double sum = 0.0;
  for (std::size_t i = 1; i <= 400; ++i) sum += alpha_spend(i);
  CHECK_NEAR(sum, 0.05, 1e-9);
  AlphaSpender sp;
  CHECK_NEAR(sp.next(), 0.005, 1e-15);
  CHECK_NEAR(sp.next(), 0.0045, 1e-15);
  CHECK(sp.count() == 2);
  CHECK_NEAR(sp.spent(), 0.05 * (1.0 - 0.81), 1e-15);
  CHECK_NEAR(sp.spent() + sp.remaining(), 0.05, 1e-15);
  sp.reset();
  CHECK(sp.count() == 0);

  // E-process: consistent gain accumulates evidence past 1/alpha_1 = 200.
  EProcess win(1.0, 0.0);
  for (int i = 0; i < 200; ++i) win.update(0.3);
  CHECK(win.e_value() > 200.0);
  CHECK(win.rejects_at(alpha_spend(1)));
  CHECK(multiplicity_check(win, 1).pass);
  CHECK_NEAR(multiplicity_check(win, 1).needed, 200.0, 1e-9);
  // Zero-mean symmetric stream: e-value never grows (each +/- pair multiplies by < 1).
  EProcess nul(1.0, 0.0);
  double max_e = 0.0;
  for (int i = 0; i < 2000; ++i) {
    nul.update(i % 2 ? 0.3 : -0.3);
    max_e = std::max(max_e, nul.e_value());
  }
  CHECK(max_e <= 1.2);
  CHECK(!nul.rejects_at(0.05));
  // A regression never produces evidence for promotion (e stays below 1; the
  // small-lambda components of the mixture lose little, so it does not go to 0).
  EProcess neg(1.0, 0.0);
  for (int i = 0; i < 100; ++i) neg.update(-0.2);
  CHECK(neg.e_value() < 0.5);
  CHECK(!neg.rejects_at(0.5));
  CHECK_NEAR(e_to_p(250.0), 0.004, 1e-12);
  CHECK_NEAR(e_to_p(0.3), 1.0, 0.0);
  // Anytime validity (empirical): under a true null the sup of E exceeds 1/alpha rarely.
  {
    std::mt19937_64 rng(2024);
    int crossings = 0;
    const int trials = 400;
    for (int t = 0; t < trials; ++t) {
      EProcess e(1.0, 0.0);
      bool crossed = false;
      for (int i = 0; i < 500 && !crossed; ++i) {
        e.update((rng() & 1) ? 0.5 : -0.5);
        crossed = e.rejects_at(0.05);
      }
      crossings += crossed ? 1 : 0;
    }
    CHECK(crossings <= 0.05 * trials);  // bound is 5%; typically ~0
  }
  // Out-of-range input rejected.
  bool threw = false;
  try {
    EProcess(1.0, 0.0).update(1.5);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  // Benjamini-Hochberg on a known p-vector (m = 6, q = 0.05): thresholds k*q/m =
  // .00833 .01667 .025 .03333 .04167 .05; sorted p = .005 .011 .02 .04 .13 .5 ->
  // largest k with p_(k) <= threshold is 3.
  const std::vector<double> p = {0.04, 0.005, 0.5, 0.02, 0.13, 0.011};
  const BhResult bh = benjamini_hochberg(p, 0.05);
  CHECK(bh.n_rejected == 3);
  const std::vector<bool> want = {false, true, false, true, false, true};
  CHECK(bh.reject == want);
  CHECK_NEAR(bh.p_threshold, 0.02, 1e-15);
  // Step-up: p_(2) fails its own threshold but p_(3) passes, so all three are rejected.
  const BhResult su = benjamini_hochberg({0.001, 0.03, 0.03, 0.9}, 0.05);
  // thresholds .0125 .025 .0375 .05 -> p_(3)=0.03 <= .0375 so k = 3.
  CHECK(su.n_rejected == 3);
  CHECK(benjamini_hochberg({0.2, 0.3}, 0.05).n_rejected == 0);
  CHECK(benjamini_hochberg({}, 0.05).n_rejected == 0);
  const std::vector<double> adj = bh_adjusted(p);
  CHECK_NEAR(adj[1], 0.03, 1e-12);        // 0.005 * 6 / 1
  CHECK_NEAR(adj[5], 0.033, 1e-12);       // 0.011 * 6 / 2
  CHECK_NEAR(adj[3], 0.04, 1e-12);        // 0.02 * 6 / 3
  CHECK_NEAR(adj[0], 0.06, 1e-12);        // 0.04 * 6 / 4
  for (std::size_t i = 0; i < p.size(); ++i) CHECK((adj[i] <= 0.05) == bh.reject[i]);
}

static void test_thresholdout() {
  // Within the threshold: dev value comes back untouched and no budget is spent.
  Thresholdout to(0.04, 1e-4, 3, 1);
  const Thresholdout::Answer a = to.query(0.80, 0.81);
  CHECK(!a.used_test && !a.budget_exhausted);
  CHECK_NEAR(a.value, 0.80, 0.0);
  CHECK(to.budget_remaining() == 3);
  // Large disagreement: test value plus Laplace noise, one unit of budget spent.
  const Thresholdout::Answer b = to.query(0.80, 0.60);
  CHECK(b.used_test);
  CHECK_NEAR(b.value, 0.60, 0.01);
  CHECK(b.value != 0.60);  // noisy
  CHECK(to.budget_remaining() == 2);
  to.query(0.8, 0.6);
  const Thresholdout::Answer c = to.query(0.8, 0.6);
  CHECK(c.used_test && to.exhausted());
  const Thresholdout::Answer d = to.query(0.8, 0.8);
  CHECK(d.budget_exhausted && std::isnan(d.value));
  to.rotate(5);
  CHECK(!to.exhausted() && to.budget_remaining() == 5);
  // Statistics: Laplace noise has mean ~0 and scale ~sigma (mean |xi| ~ sigma).
  Thresholdout many(0.0, 0.01, 100000, 99);
  double s = 0.0, sa = 0.0;
  int k = 0;
  for (int i = 0; i < 20000; ++i) {
    const Thresholdout::Answer r = many.query(0.0, 1.0);  // always released
    if (r.used_test) {
      s += r.value - 1.0;
      sa += std::fabs(r.value - 1.0);
      ++k;
    }
  }
  CHECK(k == 20000);
  CHECK_NEAR(s / k, 0.0, 5e-4);
  CHECK_NEAR(sa / k, 0.01, 5e-4);
}

static void test_bwt_and_curve() {
  // R[i][j]: performance on task j after stage i.
  const std::vector<std::vector<double>> R = {
      {0.90, 0.10, 0.10}, {0.80, 0.85, 0.10}, {0.70, 0.75, 0.95}};
  CHECK_NEAR(backward_transfer(R), ((0.70 - 0.90) + (0.75 - 0.85)) / 2.0, 1e-12);  // -0.15
  CHECK_NEAR(backward_transfer({{0.9}}), 0.0, 0.0);
  CHECK_NEAR(backward_transfer({0.7, 0.75}, {0.9, 0.85}), -0.15, 1e-12);
  const std::vector<std::vector<double>> no_forgetting = {{0.9, 0.0}, {0.9, 0.8}};
  CHECK_NEAR(backward_transfer(no_forgetting), 0.0, 1e-12);

  // Learning-curve fit recovers known parameters.
  std::vector<double> n, L;
  for (double x = 100; x <= 51200; x *= 2) {
    n.push_back(x);
    L.push_back(0.2 + 2.0 * std::pow(x, -0.5));
  }
  const LearningCurveFit f = fit_learning_curve(n, L);
  CHECK(f.ok);
  CHECK_NEAR(f.Linf, 0.2, 1e-6);
  CHECK_NEAR(f.a, 2.0, 1e-4);
  CHECK_NEAR(f.b, 0.5, 1e-5);
  CHECK(f.rmse < 1e-6);
  CHECK_NEAR(f.predict(1e6), 0.2 + 2.0 * std::pow(1e6, -0.5), 1e-6);
  CHECK_NEAR(f.n_for(0.25), std::pow((0.25 - 0.2) / 2.0, -2.0), 1.0);  // 2.5e... = 640000
  CHECK(std::isinf(f.n_for(0.1)));  // below the asymptote: unreachable
  // Off-grid exponent, accuracy-style curve (a < 0), with a little noise.
  std::mt19937_64 rng(3);
  std::normal_distribution<double> N01(0.0, 1.0);
  std::vector<double> n2, L2;
  for (double x = 50; x <= 100000; x *= 1.5) {
    n2.push_back(x);
    L2.push_back(0.9 - 1.5 * std::pow(x, -0.337) + 0.0005 * N01(rng));
  }
  const LearningCurveFit f2 = fit_learning_curve(n2, L2);
  CHECK(f2.ok);
  CHECK_NEAR(f2.b, 0.337, 0.03);
  CHECK_NEAR(f2.Linf, 0.9, 0.02);
  CHECK(f2.a < 0.0);
  CHECK(!fit_learning_curve({1, 2}, {1, 2}).ok);  // too few points
}

int main() {
  test_beta_and_wilson();
  test_evidence_factors();
  test_fusion_and_grades();
  test_answer_confidence_and_abstain();
  test_calibration();
  test_admission_rules();
  test_bootstrap();
  test_lci_components();
  test_lci_decision();
  test_multiplicity();
  test_thresholdout();
  test_bwt_and_curve();
  std::printf("ecs_lci_test: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}
