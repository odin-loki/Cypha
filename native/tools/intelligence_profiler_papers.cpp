/// Papers II–V scenarios for intelligence profiler: applications, landscape, epistemic loop, soft world.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/intelligence/causal_graph.hpp"
#include "cypha/intelligence/epistemic_threshold.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/measurers.hpp"
#include "cypha/intelligence/profile_from_model.hpp"
#include "cypha/intelligence/profile_guided_loss.hpp"
#include "cypha/intelligence/self_correcting_infer.hpp"
#include "cypha/intelligence/soft_world_monitor.hpp"

namespace {

namespace fs = std::filesystem;

constexpr double kTol = 1e-5;

bool near(double a, double b, double tol = kTol) {
  return std::abs(a - b) <= tol;
}

void test_paper_ii_applications() {
  const auto target = cypha::intelligence::IntelligenceProfiler::critical_targets();
  cypha::intelligence::ProfileObservation at_target;
  at_target.alpha = target[0];
  at_target.d_eff = target[1];
  at_target.sigma_branch = target[2];
  at_target.tau = target[3];
  at_target.r_eu = target[4];
  at_target.lipschitz = target[5];
  at_target.calibration = target[6];

  const double nav_loss =
      cypha::intelligence::IntelligenceProfiler::navigation_loss(at_target);
  assert(near(nav_loss, 0.0));

  const auto ffn =
      cypha::intelligence::IntelligenceProfiler::landscape_reference(
          cypha::intelligence::LandscapeSystemClass::SimpleFfn);
  const auto flags = cypha::intelligence::IntelligenceProfiler::predict_failure_modes(ffn);
  assert(flags.high_lipschitz);
  assert(flags.low_tau);
  assert(flags.low_r_eu);

  const auto cypha =
      cypha::intelligence::IntelligenceProfiler::landscape_reference(
          cypha::intelligence::LandscapeSystemClass::CyphaAugmented);
  const double dist =
      cypha::intelligence::IntelligenceProfiler::profile_distance_normalized(ffn, cypha);
  assert(dist > 0.2 && dist < 1.0);
}

void test_paper_iii_landscape() {
  const auto transformer =
      cypha::intelligence::IntelligenceProfiler::landscape_reference(
          cypha::intelligence::LandscapeSystemClass::LargeTransformer);
  const auto cypha =
      cypha::intelligence::IntelligenceProfiler::landscape_reference(
          cypha::intelligence::LandscapeSystemClass::CyphaAugmented);
  const auto human =
      cypha::intelligence::IntelligenceProfiler::landscape_reference(
          cypha::intelligence::LandscapeSystemClass::HumanMedian);

  const double kappa_transformer =
      cypha::intelligence::IntelligenceProfiler::criticality_score_for(transformer);
  const double kappa_cypha =
      cypha::intelligence::IntelligenceProfiler::criticality_score_for(cypha);
  assert(kappa_cypha > kappa_transformer);

  assert(cypha::intelligence::IntelligenceProfiler::dominates(cypha, transformer));
  assert(!cypha::intelligence::IntelligenceProfiler::dominates(transformer, cypha));
  assert(!cypha::intelligence::IntelligenceProfiler::dominates(cypha, human));

  const double dist_to_human =
      cypha::intelligence::IntelligenceProfiler::profile_distance_normalized(cypha, human);
  assert(dist_to_human < 0.2);
}

void test_paper_iv_epistemic_threshold() {
  cypha::intelligence::EpistemicThreshold threshold(0.5, 5.0);
  assert(threshold.should_correct(0.8));
  assert(!threshold.should_correct(0.3));

  const double before = threshold.threshold();
  threshold.update(0.8, true);
  assert(threshold.threshold() < before + 0.1);

  cypha::intelligence::EpistemicThreshold raised(0.5, 5.0);
  const double start = raised.threshold();
  raised.update(0.6, false);
  assert(raised.threshold() > start);
}

void test_paper_v_soft_world() {
  cypha::intelligence::SoftWorldMonitor monitor;
  assert(monitor.maturation_level() == 0.0);

  monitor.record_acquisition(0.8, 0.3);
  assert(monitor.query_quality() > 0.4);
  assert(monitor.maturation_level() > 0.4);

  monitor.record_simulation(0.25);
  assert(monitor.maturation_level() > 0.4);

  const auto soft_world =
      cypha::intelligence::IntelligenceProfiler::landscape_reference(
          cypha::intelligence::LandscapeSystemClass::SoftWorldCypha);
  const auto self_correcting =
      cypha::intelligence::IntelligenceProfiler::landscape_reference(
          cypha::intelligence::LandscapeSystemClass::SelfCorrectingCypha);
  const double kappa_soft =
      cypha::intelligence::IntelligenceProfiler::criticality_score_for(soft_world);
  const double kappa_self =
      cypha::intelligence::IntelligenceProfiler::criticality_score_for(self_correcting);
  assert(kappa_soft > kappa_self);
  assert(soft_world.tau > self_correcting.tau);
}

void test_paper_iv_self_correcting_infer() {
  const fs::path root = cypha::bench::repo_root();
  assert(cypha::intelligence::profile_from_reference_fixture(root).criticality_score() > 0.0);

  cypha::CyphaInferModel model = cypha::intelligence::load_reference_model_from_fixture(root);
  const std::vector<double> x = cypha::intelligence::reference_fixture_first_input(root);
  cypha::intelligence::EpistemicThreshold threshold(0.5, 5.0);
  cypha::CyphaInferOptions opt{};
  opt.deliberation_lo = 0.2;
  opt.deliberation_hi = 0.8;
  const auto result = cypha::intelligence::self_correcting_infer(
      model, x.data(), static_cast<int>(x.size()), opt, threshold, 3);
  assert(result.correction_passes >= 1);
  assert(!result.infer.label.empty());
}

void test_extended_measurers_and_batch() {
  const std::vector<double> input = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  const std::vector<double> perturb = {0.01, 0.11, 0.21, 0.31, 0.41, 0.51};
  const std::vector<double> output = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2};
  const std::vector<double> output_pert = {0.25, 0.45, 0.65, 0.85, 1.05, 1.25};

  const double sigma =
      cypha::intelligence::compute_branching_ratio_sensitivity(output.data(), output_pert.data(),
                                                               perturb.data(), 3, 2);
  assert(sigma > 0.0 && sigma <= 1.0);

  const double lipschitz =
      cypha::intelligence::compute_lipschitz_sensitivity(output.data(), output_pert.data(), 3, 2);
  assert(lipschitz > 0.0 && lipschitz <= 1.0);

  std::vector<double> sequence;
  for (int t = 0; t < 16; ++t) {
    const double v = std::sin(static_cast<double>(t) * 0.3);
    sequence.push_back(v);
    sequence.push_back(v * 0.5);
  }
  const double tau = cypha::intelligence::compute_memory_depth_normalized(
      sequence.data(), 16, 2, 8, 512);
  assert(tau > 0.1);

  cypha::intelligence::IntelligenceProfiler profiler;
  cypha::intelligence::ProfileBatch batch;
  batch.input = input.data();
  batch.output = output.data();
  batch.perturbed_input = perturb.data();
  batch.perturbed_output = output_pert.data();
  batch.n_samples = 3;
  batch.n_dims = 2;
  batch.sequence = sequence.data();
  batch.n_timesteps = 16;
  batch.tau_max_lag = 8;
  profiler.update_from_batch(batch);

  const auto matrix = profiler.get_profile_matrix();
  assert(matrix[static_cast<std::size_t>(cypha::intelligence::ProfileStatistic::SigmaBranch)][0] >
         0.0);
  assert(matrix[static_cast<std::size_t>(cypha::intelligence::ProfileStatistic::Tau)][0] > 0.1);
}

void test_paper_v_causal_graph() {
  cypha::intelligence::CausalGraphMonitor graph;
  cypha::intelligence::ProfileObservation a;
  a.alpha = 0.45;
  a.tau = 0.55;
  a.r_eu = 0.6;
  cypha::intelligence::ProfileObservation b = a;
  b.r_eu = 0.75;
  graph.observe_profile(a);
  graph.observe_profile(b);
  graph.record_simulation(0.2);
  graph.simulation_step(0.85, 0.55, 0.3);
  graph.simulation_step(0.7, 0.4, 0.25);
  cypha::intelligence::ProfileObservation traj_obs = b;
  traj_obs.r_eu = 0.72;
  graph.run_simulation_trajectory(3, traj_obs, 0.25);
  const auto j = graph.to_json();
  assert(j.contains("edges"));
  assert(j.at("step_count").get<int>() >= 5);
  const auto traj = graph.trajectory_json();
  assert(traj.at("trajectory").is_array());
  assert(traj.at("trajectory").size() >= 5);
  assert(j.at("soft_world").at("maturation_level").get<double>() >= 0.0);
}

void test_paper_v_causal_graph_correlation_edges() {
  using cypha::intelligence::CausalGraphMonitor;
  using cypha::intelligence::ProfileObservation;

  // Single observation: cannot estimate a correlation from one point, so both
  // online-estimated edges must report weight 0 (n < 2), unlike the old fixed-formula
  // implementation which produced a confident-looking nonzero number from one sample.
  {
    CausalGraphMonitor single;
    ProfileObservation obs;
    obs.alpha = 0.72;
    obs.calibration = 0.65;
    obs.tau = 0.6;
    obs.r_eu = 0.4;
    single.observe_profile(obs);
    assert(single.alpha_calibration_n() == 1);
    assert(near(single.alpha_calibration_correlation(), 0.0, 1e-9));
    assert(near(single.tau_r_eu_correlation(), 0.0, 1e-9));
    const auto j = single.to_json();
    bool found_zero_alpha_edge = false;
    for (const auto& e : j.at("edges")) {
      if (e.at("from") == "alpha" && e.at("to") == "calibration") {
        assert(near(e.at("weight").get<double>(), 0.0, 1e-9));
        found_zero_alpha_edge = true;
      }
    }
    assert(found_zero_alpha_edge);
  }

  // Strongly positively correlated alpha/calibration history (moving together) must
  // drive the estimated edge weight toward 1; tau/r_eu held anti-correlated in the same
  // pass must drive that edge weight toward 1 as well (magnitude, sign-agnostic).
  {
    CausalGraphMonitor corr;
    for (int i = 0; i < 12; ++i) {
      ProfileObservation obs;
      const double t = static_cast<double>(i) / 11.0;
      obs.alpha = 0.2 + 0.6 * t;         // rising
      obs.calibration = 0.3 + 0.5 * t;   // rising with alpha -> strong positive correlation
      obs.tau = 0.2 + 0.6 * t;           // rising
      obs.r_eu = 0.8 - 0.6 * t;          // falling -> strong negative correlation with tau
      corr.observe_profile(obs);
    }
    assert(corr.alpha_calibration_n() == 12);
    assert(corr.alpha_calibration_correlation() > 0.95);
    assert(corr.tau_r_eu_correlation() < -0.95);
    const auto j = corr.to_json();
    const auto& est = j.at("edge_estimation");
    assert(est.at("alpha_calibration_n").get<int>() == 12);
    for (const auto& e : j.at("edges")) {
      if (e.at("from") == "alpha" && e.at("to") == "calibration") {
        assert(e.at("weight").get<double>() > 0.95);
      }
      if (e.at("from") == "tau" && e.at("to") == "r_eu") {
        assert(e.at("weight").get<double>() > 0.95);
      }
    }
  }

  // Uncorrelated (constant) history: variance is ~0 so the correlation is undefined and
  // must fall back to the documented 0.0, not a spurious value.
  {
    CausalGraphMonitor flat;
    for (int i = 0; i < 8; ++i) {
      ProfileObservation obs;
      obs.alpha = 0.5;
      obs.calibration = 0.5;
      obs.tau = 0.5;
      obs.r_eu = 0.5;
      flat.observe_profile(obs);
    }
    assert(near(flat.alpha_calibration_correlation(), 0.0, 1e-9));
    assert(near(flat.tau_r_eu_correlation(), 0.0, 1e-9));
  }
}

void test_paper_v_soft_world_simulation_step() {
  cypha::intelligence::SoftWorldMonitor monitor;
  monitor.simulation_step(0.8, 0.5, 0.2);
  assert(monitor.query_quality() > 0.2);
  assert(monitor.maturation_level() > 0.2);
}

void test_paper_iv_profile_guided_loss() {
  const auto cfg = cypha::intelligence::default_profile_guided_loss_config();
  cypha::intelligence::ProfileObservation obs;
  obs.alpha = 0.75;
  obs.d_eff = 0.15;
  obs.sigma_branch = 0.35;
  obs.tau = 0.05;
  obs.r_eu = 0.10;
  obs.lipschitz = 0.70;
  obs.calibration = 0.45;
  const auto loss = cypha::intelligence::compute_profile_guided_loss(obs, cfg);
  assert(loss.alpha_penalty > 0.0);
  assert(loss.d_eff_penalty > 0.0);
  assert(loss.sigma_branch_penalty > 0.0);
  assert(loss.tau_penalty > 0.0);
  assert(loss.r_eu_penalty > 0.0);
  assert(loss.lipschitz_penalty > 0.0);
  assert(loss.calibration_penalty > 0.0);
  assert(near(loss.navigation_loss_total,
              cypha::intelligence::IntelligenceProfiler::navigation_loss(obs)));
  assert(near(loss.total, loss.alpha_penalty + loss.d_eff_penalty + loss.sigma_branch_penalty +
                            loss.tau_penalty + loss.r_eu_penalty + loss.lipschitz_penalty +
                            loss.calibration_penalty));

  const auto grad =
      cypha::intelligence::compute_profile_guided_loss_grad(obs, cfg, /*gria_field_dim=*/8);
  assert(grad.d_alpha_uniform != 0.0);
  assert(grad.d_logit_uniform != 0.0);
  assert(grad.d_gria_input.size() == 8U);
  assert(grad.d_gria_input[0] > 0.0);
}

void test_adaptive_navigation_lambdas() {
  const auto base = cypha::intelligence::default_profile_guided_loss_config();
  const double low_kappa = 0.30;
  const double high_kappa = 0.85;
  const double target = 0.89;
  const auto scaled_low =
      cypha::intelligence::scale_profile_guided_loss_config(base, low_kappa, target);
  const auto scaled_high =
      cypha::intelligence::scale_profile_guided_loss_config(base, high_kappa, target);
  const double scale_low = std::clamp(1.0 - low_kappa / target, 0.1, 1.0);
  const double scale_high = std::clamp(1.0 - high_kappa / target, 0.1, 1.0);
  assert(scale_low > scale_high);
  assert(near(scaled_low.lambda_alpha, base.lambda_alpha * scale_low));
  assert(near(scaled_high.lambda_alpha, base.lambda_alpha * scale_high));
  assert(scaled_low.lambda_alpha > scaled_high.lambda_alpha);
  assert(scaled_low.lambda_tau > scaled_high.lambda_tau);
}

void test_kappa_trajectory_navigation_lambdas() {
  const auto base = cypha::intelligence::default_profile_guided_loss_config();
  cypha::intelligence::KappaTrajectoryState state;
  const auto step1 = cypha::intelligence::scale_profile_guided_loss_from_trajectory(
      base, 0.40, state, 0.89, 8);
  const auto step2 = cypha::intelligence::scale_profile_guided_loss_from_trajectory(
      base, 0.35, state, 0.89, 8);
  assert(state.sample_count == 2);
  assert(step2.lambda_alpha >= step1.lambda_alpha);
}

void test_per_stat_deviation_navigation_lambdas() {
  const auto base = cypha::intelligence::default_profile_guided_loss_config();
  cypha::intelligence::ProfileObservation obs;
  obs.alpha = 0.20;
  obs.d_eff = base.target_d_eff;
  const auto scaled =
      cypha::intelligence::scale_profile_guided_loss_by_stat_deviation(base, obs, 0.5);
  assert(scaled.lambda_alpha > base.lambda_alpha);
  assert(near(scaled.lambda_d_eff, base.lambda_d_eff));
}

void test_kappa_ceiling_navigation_lambdas() {
  const auto base = cypha::intelligence::default_profile_guided_loss_config();
  cypha::intelligence::ProfileObservation obs;
  obs.alpha = 0.75;
  obs.d_eff = 0.15;
  obs.sigma_branch = 0.35;
  obs.tau = 0.55;
  obs.r_eu = 0.60;
  obs.lipschitz = 0.70;
  obs.calibration = 0.45;
  cypha::intelligence::AdaptiveNavigationOptions opts;
  opts.use_kappa_ceiling_lambdas = true;
  opts.target_kappa = 0.89;
  const auto below = cypha::intelligence::resolve_adaptive_profile_guided_config(
      base, obs, opts, nullptr);
  assert(near(below.lambda_alpha, base.lambda_alpha));
  obs.alpha = 0.95;
  obs.d_eff = 0.85;
  obs.tau = 0.90;
  obs.r_eu = 0.88;
  const auto above = cypha::intelligence::resolve_adaptive_profile_guided_config(
      base, obs, opts, nullptr);
  assert(above.lambda_alpha < base.lambda_alpha);
  assert(above.lambda_tau < base.lambda_tau);
}

void test_hidden_nudge_grad() {
  const auto cfg = cypha::intelligence::default_profile_guided_loss_config();
  cypha::intelligence::ProfileObservation obs;
  obs.d_eff = cfg.target_d_eff - 0.2;
  const auto grad =
      cypha::intelligence::compute_profile_guided_loss_grad(obs, cfg, /*gria_field_dim=*/4);
  assert(grad.d_h_hidden_uniform != 0.0);
  assert(std::signbit(grad.d_h_hidden_uniform) == std::signbit(grad.d_logit_uniform));
}

void test_eigenvalue_participation_ratio() {
  const std::vector<double> activations{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const double proxy = cypha::intelligence::compute_participation_ratio(
      activations.data(), 3, 3, cypha::intelligence::ParticipationRatioMethod::VarianceProxy);
  const double eigen = cypha::intelligence::compute_participation_ratio(
      activations.data(), 3, 3,
      cypha::intelligence::ParticipationRatioMethod::CovarianceEigenvalue);
  assert(proxy > 0.0 && eigen > 0.0);
  assert(near(proxy, eigen, 0.15));
}

// Phase 0 (hidden-dim scale-up plan, docs/reports/HIDDEN_DIM_SCALE_PLAN.md §4.3/§7):
// the eigenvalue-fidelity D_eff method must stay accurate (not silently fall back to
// VarianceProxy) above 256 dims, since that's exactly the LSTM hidden-width range the
// scale-up plan needs to trust. Covers both correctness (TraceFrobenius must match the
// Jacobi eigenvalue result within tolerance where Jacobi is still affordable) and the
// >256-dim path (where CovarianceEigenvalue now delegates to TraceFrobenius internally
// instead of degrading to VarianceProxy).
void test_trace_frobenius_participation_ratio() {
  // Small case: TraceFrobenius should agree tightly with the Jacobi eigenvalue method,
  // since (Sigma lambda)^2 / Sigma(lambda^2) == trace(C)^2 / trace(C^2) exactly for a
  // symmetric covariance matrix -- this is an algebraic identity, not an approximation.
  const std::vector<double> small_activations{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const double small_eigen = cypha::intelligence::compute_participation_ratio(
      small_activations.data(), 3, 3,
      cypha::intelligence::ParticipationRatioMethod::CovarianceEigenvalue);
  const double small_trace_frob = cypha::intelligence::compute_participation_ratio(
      small_activations.data(), 3, 3,
      cypha::intelligence::ParticipationRatioMethod::TraceFrobenius);
  assert(near(small_eigen, small_trace_frob, 1e-6));

  // Large case: n_dims=300 is above the old hard `n_dims > 256` guard
  // (native/src/intelligence/measurers.cpp) that used to silently swap in the cheaper,
  // different VarianceProxy metric with no signal. Use a deterministic LCG so the
  // activations have realistic cross-dimension correlation structure (not degenerate
  // like a pure identity matrix), and keep n_samples small (48) to match the LSTM
  // hidden-state ring buffer cap (`kLstmHiddenHistoryMax`, cyphalm_model.hpp).
  constexpr int kDims = 300;
  constexpr int kSamples = 48;
  std::vector<double> big(static_cast<std::size_t>(kDims * kSamples), 0.0);
  std::uint64_t state = 0x2545F4914F6CDD1DULL;
  auto next_uniform = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((state >> 11) & 0xFFFFFFFFFFFFFULL) /
           static_cast<double>(1ULL << 52);
  };
  std::vector<double> shared_factor(static_cast<std::size_t>(kSamples), 0.0);
  for (int r = 0; r < kSamples; ++r) {
    shared_factor[static_cast<std::size_t>(r)] = next_uniform() - 0.5;
  }
  for (int r = 0; r < kSamples; ++r) {
    for (int d = 0; d < kDims; ++d) {
      const double idio = next_uniform() - 0.5;
      // Mix a shared low-rank factor into every 4th dimension so the covariance matrix
      // has genuine off-diagonal structure (a pure-noise matrix would trivially make
      // every method agree, which would not exercise the eigenvalue-vs-trace path).
      const double coupling = (d % 4 == 0) ? 0.8 : 0.0;
      big[static_cast<std::size_t>(r * kDims + d)] =
          idio + coupling * shared_factor[static_cast<std::size_t>(r)];
    }
  }

  const double big_variance_proxy = cypha::intelligence::compute_participation_ratio(
      big.data(), kSamples, kDims, cypha::intelligence::ParticipationRatioMethod::VarianceProxy);
  const double big_eigen = cypha::intelligence::compute_participation_ratio(
      big.data(), kSamples, kDims,
      cypha::intelligence::ParticipationRatioMethod::CovarianceEigenvalue);
  const double big_trace_frob = cypha::intelligence::compute_participation_ratio(
      big.data(), kSamples, kDims,
      cypha::intelligence::ParticipationRatioMethod::TraceFrobenius);

  assert(big_variance_proxy > 0.0 && big_variance_proxy <= 1.0);
  assert(big_eigen > 0.0 && big_eigen <= 1.0);
  assert(big_trace_frob > 0.0 && big_trace_frob <= 1.0);
  // Above the old n_dims > 256 guard, CovarianceEigenvalue must now delegate to
  // TraceFrobenius exactly (same call path, no diagonalization) -- not silently return
  // the (numerically different) VarianceProxy value.
  assert(near(big_eigen, big_trace_frob, 1e-9));
}

void test_kappa_ceiling_strength() {
  const auto base = cypha::intelligence::default_profile_guided_loss_config();
  cypha::intelligence::ProfileObservation obs;
  obs.alpha = 0.95;
  obs.d_eff = 0.85;
  obs.tau = 0.90;
  obs.r_eu = 0.88;
  cypha::intelligence::AdaptiveNavigationOptions opts;
  opts.use_kappa_ceiling_lambdas = true;
  opts.target_kappa = 0.83;
  opts.kappa_ceiling_strength = 3.0;
  opts.kappa_ceiling_min_scale = 0.35;
  const auto scaled =
      cypha::intelligence::resolve_adaptive_profile_guided_config(base, obs, opts, nullptr);
  assert(scaled.lambda_alpha < base.lambda_alpha * 0.6);
}

void test_kappa_excess_grad_nudge() {
  cypha::intelligence::ProfileObservation obs;
  obs.alpha = 0.95;
  obs.d_eff = 0.85;
  obs.tau = 0.55;
  const auto grad_in_margin =
      cypha::intelligence::kappa_excess_grad_nudge(obs, 0.85, 0.84, 1.5, 0.02);
  assert(grad_in_margin.d_alpha_uniform == 0.0);
  const auto grad = cypha::intelligence::kappa_excess_grad_nudge(obs, 0.90, 0.84, 1.5, 0.02);
  assert(grad.d_alpha_uniform < 0.0);
}

void test_kappa_trajectory_ceiling() {
  const auto base = cypha::intelligence::default_profile_guided_loss_config();
  cypha::intelligence::KappaTrajectoryState state;
  state.ema_kappa = 0.88;
  state.prev_ema_kappa = 0.85;
  state.sample_count = 2;
  const auto with_ceiling = cypha::intelligence::scale_profile_guided_loss_from_trajectory(
      base, 0.90, state, 0.84, 8, true);
  cypha::intelligence::KappaTrajectoryState state2 = state;
  const auto without_ceiling = cypha::intelligence::scale_profile_guided_loss_from_trajectory(
      base, 0.90, state2, 0.84, 8, false);
  assert(with_ceiling.lambda_alpha < without_ceiling.lambda_alpha);
}

void test_kappa_kernel_blend_scale() {
  const double at_target =
      cypha::intelligence::scale_kernel_blend_from_kappa(0.25, 0.83, 0.83, 0.08);
  assert(near(at_target, 0.25, 1e-9));
  const double high_kappa =
      cypha::intelligence::scale_kernel_blend_from_kappa(0.25, 0.92, 0.83, 0.08);
  assert(high_kappa < 0.25 && high_kappa >= 0.08);
}

void test_kappa_navigation_warmup_scale() {
  const double at_target =
      cypha::intelligence::scale_navigation_warmup_from_kappa(0.8, 0.83, 0.83, 0.35, 0.65);
  assert(near(at_target, 0.8, 1e-9));
  const double high_kappa =
      cypha::intelligence::scale_navigation_warmup_from_kappa(0.8, 0.92, 0.83, 0.35, 0.65);
  assert(high_kappa < 0.8 && high_kappa >= 0.65);
}

}  // namespace

int main() {
  test_paper_ii_applications();
  test_paper_iii_landscape();
  test_paper_iv_epistemic_threshold();
  test_paper_iv_self_correcting_infer();
  test_paper_v_soft_world();
  test_paper_v_soft_world_simulation_step();
  test_paper_v_causal_graph();
  test_paper_v_causal_graph_correlation_edges();
  test_paper_iv_profile_guided_loss();
  test_adaptive_navigation_lambdas();
  test_kappa_trajectory_navigation_lambdas();
  test_per_stat_deviation_navigation_lambdas();
  test_kappa_ceiling_navigation_lambdas();
  test_kappa_ceiling_strength();
  test_kappa_excess_grad_nudge();
  test_kappa_kernel_blend_scale();
  test_kappa_navigation_warmup_scale();
  test_kappa_trajectory_ceiling();
  test_eigenvalue_participation_ratio();
  test_trace_frobenius_participation_ratio();
  test_hidden_nudge_grad();
  test_extended_measurers_and_batch();
  std::puts("intelligence_profiler_papers: PASS");
  return 0;
}
