#include "cypha/intelligence/profile_from_model.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/intelligence/criticality_vector.hpp"
#include "cypha/intelligence/intelligence_profile_json.hpp"
#include "cypha/intelligence/causal_graph.hpp"
#include "cypha/intelligence/measurers.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"

namespace cypha::intelligence {

namespace {

struct ParityBin {
  std::uint32_t n{0};
  std::uint32_t d{0};
  std::uint32_t k{0};
  std::uint32_t field_dim{0};
  double temperature{0.0};
  double eps{0.0};
  std::vector<double> f_field;
  std::vector<double> x;
  std::vector<double> exp_probs;
};

ParityBin load_parity_bin(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  std::vector<std::uint8_t> sidecar((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (sidecar.size() < 8 + 4 * 5 + 8 * 2) {
    throw std::runtime_error("native_parity.bin too small");
  }
  if (std::memcmp(sidecar.data(), "CYPHNP01", 8) != 0) {
    throw std::runtime_error("native_parity.bin: bad magic (expected CYPHNP01)");
  }

  ParityBin bin;
  std::size_t o = 8;
  std::uint32_t ver = 0;
  std::memcpy(&ver, sidecar.data() + o, 4);
  o += 4;
  std::memcpy(&bin.n, sidecar.data() + o, 4);
  o += 4;
  std::memcpy(&bin.d, sidecar.data() + o, 4);
  o += 4;
  std::memcpy(&bin.k, sidecar.data() + o, 4);
  o += 4;
  std::memcpy(&bin.field_dim, sidecar.data() + o, 4);
  o += 4;
  std::memcpy(&bin.temperature, sidecar.data() + o, 8);
  o += 8;
  std::memcpy(&bin.eps, sidecar.data() + o, 8);
  o += 8;
  if (ver != 1u && ver != 2u) {
    throw std::runtime_error("native_parity.bin: unsupported version");
  }

  const std::size_t core = o + static_cast<std::size_t>(bin.d) * bin.field_dim * 8u +
                           static_cast<std::size_t>(bin.n) * bin.d * 8u +
                           static_cast<std::size_t>(bin.n) * bin.k * 8u * 2u +
                           static_cast<std::size_t>(bin.n) * 8u;
  const std::size_t need_v2 = core + static_cast<std::size_t>(bin.n) * 8u * 2u;
  if (ver == 1u && sidecar.size() < core) {
    throw std::runtime_error("native_parity.bin: truncated payload");
  }
  if (ver == 2u && sidecar.size() < need_v2) {
    throw std::runtime_error("native_parity.bin: truncated payload (v2 tail)");
  }

  bin.f_field.resize(static_cast<std::size_t>(bin.d) * bin.field_dim);
  std::memcpy(bin.f_field.data(), sidecar.data() + o, bin.f_field.size() * sizeof(double));
  o += bin.f_field.size() * sizeof(double);

  bin.x.resize(static_cast<std::size_t>(bin.n) * bin.d);
  std::memcpy(bin.x.data(), sidecar.data() + o, bin.x.size() * sizeof(double));
  o += bin.x.size() * sizeof(double);
  o += static_cast<std::size_t>(bin.n) * bin.k * 8u;  // skip exp_llr

  bin.exp_probs.resize(static_cast<std::size_t>(bin.n) * bin.k);
  std::memcpy(bin.exp_probs.data(), sidecar.data() + o, bin.exp_probs.size() * sizeof(double));
  return bin;
}

ProfileObservation mean_observation(const IntelligenceProfiler& profiler) {
  const auto matrix = profiler.get_profile_matrix();
  ProfileObservation obs;
  obs.alpha = matrix[0][0];
  obs.d_eff = matrix[1][0];
  obs.sigma_branch = matrix[2][0];
  obs.tau = matrix[3][0];
  obs.r_eu = matrix[4][0];
  obs.lipschitz = matrix[5][0];
  obs.calibration = matrix[6][0];
  return obs;
}

}  // namespace

cypha::CyphaInferModel load_reference_model_from_fixture(const std::filesystem::path& repo_root) {
  const std::filesystem::path root = repo_root.empty() ? cypha::bench::repo_root() : repo_root;
  const std::filesystem::path cypha = root / "fixtures" / "reference.cypha";
  const ParityBin bin = load_parity_bin(root / "fixtures" / "native_parity.bin");
  const cypha::CNode root_node = cypha::load_cypha_file(cypha.string().c_str());
  return cypha::CyphaInferModel::from_root(root_node, bin.f_field.data(), static_cast<int>(bin.field_dim));
}

std::vector<double> reference_fixture_first_input(const std::filesystem::path& repo_root) {
  const std::filesystem::path root = repo_root.empty() ? cypha::bench::repo_root() : repo_root;
  const ParityBin bin = load_parity_bin(root / "fixtures" / "native_parity.bin");
  std::vector<double> row(static_cast<std::size_t>(bin.d));
  if (!bin.x.empty()) {
    std::memcpy(row.data(), bin.x.data(), row.size() * sizeof(double));
  }
  return row;
}

IntelligenceProfiler profile_from_reference_fixture(const std::filesystem::path& repo_root,
                                                    const std::filesystem::path& cypha_path) {
  const std::filesystem::path root = repo_root.empty() ? cypha::bench::repo_root() : repo_root;
  const std::filesystem::path cypha =
      cypha_path.empty() ? root / "fixtures" / "reference.cypha" : cypha_path;
  const std::filesystem::path bin_path = root / "fixtures" / "native_parity.bin";

  const ParityBin bin = load_parity_bin(bin_path);
  const cypha::CNode root_node = cypha::load_cypha_file(cypha.string().c_str());
  cypha::CyphaInferModel model =
      cypha::CyphaInferModel::from_root(root_node, bin.f_field.data(), static_cast<int>(bin.field_dim));
  if (model.d_latent != static_cast<int>(bin.d)) {
    throw std::runtime_error("d_latent mismatch between model and native_parity.bin");
  }
  if (static_cast<int>(model.labels.size()) != static_cast<int>(bin.k)) {
    throw std::runtime_error("class count mismatch between model and native_parity.bin");
  }

  std::vector<double> h;
  cypha::batch_encode(model, bin.x.data(), static_cast<int>(bin.n), h);

  std::vector<double> h_pert(h.size());
  for (std::size_t i = 0; i < h.size(); ++i) {
    h_pert[i] = h[i] + 0.01 * std::sin(static_cast<double>(i));
  }

  std::vector<double> conf;
  std::vector<int> correct;
  conf.reserve(bin.n);
  correct.reserve(bin.n);
  const int d_lat = model.d_latent;
  const int k = static_cast<int>(bin.k);
  for (std::uint32_t i = 0; i < bin.n; ++i) {
    const cypha::InferAtHResult inf =
        cypha::infer_at_h(model, h.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(d_lat), {});
    conf.push_back(inf.confidence);

    int y_gt = 0;
    const double* prow = bin.exp_probs.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(k);
    for (int j = 1; j < k; ++j) {
      if (prow[static_cast<std::size_t>(j)] > prow[static_cast<std::size_t>(y_gt)]) {
        y_gt = j;
      }
    }
    correct.push_back(inf.label == model.labels[static_cast<std::size_t>(y_gt)] ? 1 : 0);
  }

  IntelligenceProfiler profiler;
  ProfileBatch batch;
  batch.input = bin.x.data();
  batch.output = h.data();
  batch.n_samples = static_cast<int>(bin.n);
  batch.n_dims = static_cast<int>(bin.d);
  batch.perturbed_input = bin.x.data();
  batch.perturbed_output = h_pert.data();
  batch.confidences = conf.data();
  batch.correct = correct.data();
  batch.n_labels = static_cast<int>(bin.n);
  batch.epistemic_var = 0.25;
  batch.aleatoric_var = 0.15;
  profiler.update_from_batch(batch);
  return profiler;
}

nlohmann::json intelligence_profile_report_json(const IntelligenceProfiler& profiler) {
  nlohmann::json root = intelligence_profile_to_json(profiler);
  const ProfileObservation obs = mean_observation(profiler);
  root["navigation_loss"] = IntelligenceProfiler::navigation_loss(obs);

  // Causal graph fidelity is read from `profiler`'s own persistent, bench-run-scoped
  // `CausalGraphMonitor` (see `IntelligenceProfiler::causal_graph()`), not a fresh
  // single-observation monitor. See docs/reports/SOFT_WORLD_CAUSAL_GRAPH_PLAN.md §9/§9.7 for
  // the full rationale. Before §9.7, this call path constructed a brand-new `CausalGraphMonitor`
  // on every call and fed it exactly one observation (`run_simulation_trajectory` only calls
  // `observe_profile` once), so `causal_fidelity()` was guaranteed 0.0 here (n=1 on both
  // estimated edges) and `apply_causal_fidelity` was a guaranteed no-op. As of §9.7, callers
  // that have access to per-step history during training/eval (e.g.
  // `LmIntelligenceMonitor::flush_to_profiler`) feed real, genuinely-varying (alpha,
  // calibration)/(tau, r_eu) checkpoints into this same persistent monitor as they go, so by the
  // time a report is generated the monitor may already have accumulated >=2 observations and
  // `causal_fidelity()` can be non-degenerate. Callers that never feed the profiler
  // incrementally (e.g. `profile_from_reference_fixture`'s single `update_from_batch` call) are
  // unaffected: their persistent monitor still only ever sees the one observation added below,
  // so `causal_fidelity()` remains exactly 0.0 and this path's kappa values stay bit-identical to
  // before this pass -- backward compatibility is preserved by construction, not by a special
  // case.
  CausalGraphMonitor& causal = profiler.causal_graph();
  causal.run_simulation_trajectory(4, obs);
  const double causal_fidelity = causal.causal_fidelity();

  root["causal_fidelity"] = causal_fidelity;
  root["criticality_score"] =
      IntelligenceProfiler::apply_causal_fidelity(root["criticality_score"].get<double>(), causal_fidelity);
  root["criticality_score_obs"] =
      IntelligenceProfiler::apply_causal_fidelity(IntelligenceProfiler::criticality_score_for(obs), causal_fidelity);

  const auto flags = IntelligenceProfiler::predict_failure_modes(obs);
  root["failure_modes"] = {
      {"low_calibration", flags.low_calibration},
      {"high_lipschitz", flags.high_lipschitz},
      {"low_tau", flags.low_tau},
      {"explosive_branching", flags.explosive_branching},
      {"damped_branching", flags.damped_branching},
      {"low_d_eff", flags.low_d_eff},
      {"low_r_eu", flags.low_r_eu},
      {"extreme_alpha", flags.extreme_alpha},
  };

  nlohmann::json landscape = nlohmann::json::object();
  landscape["simple_ffn"] = nlohmann::json{
      {"kappa", IntelligenceProfiler::criticality_score_for(
                    IntelligenceProfiler::landscape_reference(LandscapeSystemClass::SimpleFfn))}};
  landscape["large_transformer"] = nlohmann::json{
      {"kappa",
       IntelligenceProfiler::criticality_score_for(IntelligenceProfiler::landscape_reference(
           LandscapeSystemClass::LargeTransformer))}};
  landscape["cypha_augmented"] = nlohmann::json{
      {"kappa", IntelligenceProfiler::criticality_score_for(
                    IntelligenceProfiler::landscape_reference(LandscapeSystemClass::CyphaAugmented))}};
  landscape["human_median"] = nlohmann::json{
      {"kappa", IntelligenceProfiler::criticality_score_for(
                    IntelligenceProfiler::landscape_reference(LandscapeSystemClass::HumanMedian))}};
  root["landscape_kappa"] = landscape;

  root["causal_graph"] = causal.to_json();
  root["criticality_vector"] = criticality_vector_to_json(profiler.criticality_vector());
  return root;
}

}  // namespace cypha::intelligence
