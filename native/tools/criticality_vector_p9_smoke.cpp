/// Phase 9 smoke: CriticalityVector telemetry is read-only — inference byte-identical with monitor on vs off.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cypha/bench/bench_paths.hpp"
#include "cypha/infer_cpu.hpp"
#include "cypha/intelligence/criticality_vector.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/profile_from_model.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/regression_stub.hpp"

namespace {

struct InferSnapshot {
  std::string label;
  double confidence = 0.0;
  std::vector<double> scores;
};

InferSnapshot infer_once(cypha::CyphaInferModel& model, const double* h) {
  const cypha::InferAtHResult inf = cypha::infer_at_h(model, h, {});
  InferSnapshot snap;
  snap.label = inf.label;
  snap.confidence = inf.confidence;
  snap.scores = inf.llrs;
  return snap;
}

bool snapshots_equal(const InferSnapshot& a, const InferSnapshot& b) {
  if (a.label != b.label) {
    return false;
  }
  if (std::memcmp(&a.confidence, &b.confidence, sizeof(double)) != 0) {
    return false;
  }
  if (a.scores.size() != b.scores.size()) {
    return false;
  }
  return std::memcmp(a.scores.data(), b.scores.data(), a.scores.size() * sizeof(double)) == 0;
}

bool field_named(const cypha::intelligence::CriticalityVector& vec, const char* name) {
  for (const auto& f : vec.fields) {
    if (f.name == name && f.available) {
      return std::isfinite(f.value);
    }
  }
  return false;
}

}  // namespace

int main() {
  try {
    const auto repo = cypha::bench::repo_root();
    cypha::CyphaInferModel model = cypha::intelligence::load_reference_model_from_fixture(repo);
    const std::vector<double> x = cypha::intelligence::reference_fixture_first_input(repo);

    std::vector<double> h;
    cypha::batch_encode(model, x.data(), 1, h);

    const InferSnapshot baseline = infer_once(model, h.data());

    cypha::intelligence::IntelligenceProfiler profiler =
        cypha::intelligence::profile_from_reference_fixture(repo);

    cypha::intelligence::CriticalityHotInput hot;
    const std::vector<double> route_probs = {0.4, 0.35, 0.25};
    hot.routing_entropy =
        cypha::regression::mke_routing_entropy(route_probs.data(), static_cast<int>(route_probs.size()), 1e-12);
    hot.dead_expert_fraction = 0.0;
    hot.anomaly_score = 0.42;
    hot.ood_rate = 0.0;

    const cypha::intelligence::CriticalityVector vec =
        profiler.criticality_vector(hot, {}, cypha::intelligence::CriticalityBuildOptions{});

    if (!field_named(vec, "alpha")) {
      std::puts("criticality_vector_p9_smoke: FAIL (alpha missing)");
      return 1;
    }
    if (!field_named(vec, "anomaly_score")) {
      std::puts("criticality_vector_p9_smoke: FAIL (anomaly_score missing)");
      return 1;
    }
    if (!field_named(vec, "routing_entropy")) {
      std::puts("criticality_vector_p9_smoke: FAIL (routing_entropy missing)");
      return 1;
    }
    if (vec.mid_enabled) {
      std::puts("criticality_vector_p9_smoke: FAIL (mid enabled by default)");
      return 1;
    }

    const auto json = cypha::intelligence::criticality_vector_to_json(vec);
    if (!json.contains("fields") || !json["fields"].is_array() || json["fields"].empty()) {
      std::puts("criticality_vector_p9_smoke: FAIL (empty REST fields)");
      return 1;
    }

    const InferSnapshot after_monitor = infer_once(model, h.data());
    if (!snapshots_equal(baseline, after_monitor)) {
      std::puts("criticality_vector_p9_smoke: FAIL (inference changed after monitor read)");
      return 1;
    }

    cypha::intelligence::CriticalityBuildOptions mid_opts;
    mid_opts.enable_mid = true;
    mid_opts.step = 0;
    const cypha::intelligence::CriticalityVector mid_vec =
        profiler.criticality_vector(hot, {}, mid_opts);
    if (!mid_vec.mid_enabled) {
      std::puts("criticality_vector_p9_smoke: FAIL (mid gate did not open)");
      return 1;
    }

    const InferSnapshot after_mid = infer_once(model, h.data());
    if (!snapshots_equal(baseline, after_mid)) {
      std::puts("criticality_vector_p9_smoke: FAIL (inference changed after mid estimators)");
      return 1;
    }

    std::puts("criticality_vector_p9_smoke: PASS");
    return 0;
  } catch (const std::exception& ex) {
    std::printf("criticality_vector_p9_smoke: FAIL (%s)\n", ex.what());
    return 1;
  }
}
