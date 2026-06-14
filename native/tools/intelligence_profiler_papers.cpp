/// Papers II–V scenarios for intelligence profiler: applications, landscape, epistemic loop, soft world.
#include <cassert>
#include <cmath>
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

void test_paper_v_soft_world_simulation_step() {
  cypha::intelligence::SoftWorldMonitor monitor;
  monitor.simulation_step(0.8, 0.5, 0.2);
  assert(monitor.query_quality() > 0.2);
  assert(monitor.maturation_level() > 0.2);
}

void test_paper_iv_profile_guided_loss() {
  cypha::intelligence::ProfileObservation obs;
  obs.r_eu = 0.8;
  obs.tau = 0.2;
  const auto loss = cypha::intelligence::compute_profile_guided_loss(obs);
  assert(loss.r_eu_penalty > 0.0);
  assert(loss.tau_penalty > 0.0);
  assert(near(loss.total, loss.r_eu_penalty + loss.tau_penalty));
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
  test_paper_iv_profile_guided_loss();
  test_extended_measurers_and_batch();
  std::puts("intelligence_profiler_papers: PASS");
  return 0;
}
