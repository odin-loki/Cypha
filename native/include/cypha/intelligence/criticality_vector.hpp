#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha::intelligence {

class IntelligenceProfiler;
class SoftWorldMonitor;

/// Tier-1 = distance-from-bound; Tier-2 = distance-from-critical-point (Phase 9).
enum class CriticalityTier {
  Tier1 = 1,
  Tier2 = 2,
};

/// Update cadence: hot every step, mid every N steps, cold on demand.
enum class CriticalityCadence {
  Hot = 0,
  Mid = 1,
  Cold = 2,
};

/// One monitored gauge with bound/target semantics encoded explicitly.
struct CriticalityField {
  std::string name;
  double value = 0.0;
  CriticalityTier tier = CriticalityTier::Tier2;
  /// Tier-2: critical target; Tier-1: acceptable bound (distance = max(0, value - bound)).
  double target_or_bound = 0.0;
  CriticalityCadence cadence = CriticalityCadence::Hot;
  /// ``|value - target|`` for Tier-2; ``max(0, value - bound)`` for Tier-1.
  double distance = 0.0;
  /// False when the upstream signal was unavailable (e.g. MoE routing entropy without router).
  bool available = true;
};

/// Unified runtime criticality readout (read-only telemetry surface).
struct CriticalityVector {
  std::vector<CriticalityField> fields;
  bool mid_enabled = false;
};

/// Optional hot-path inputs not carried in ``IntelligenceProfiler`` alone.
struct CriticalityHotInput {
  std::optional<double> routing_entropy;
  std::optional<double> dead_expert_fraction;
  double anomaly_score = 0.0;
  double drift_score = 0.0;
  double nig_confidence = 0.5;
  double effective_sample_size = 1.0;
  double ood_rate = 0.0;
};

/// Mid-cadence estimator inputs (stochastic / subsampled).
struct CriticalityMidInput {
  std::optional<double> spectral_radius;
  std::optional<double> kernel_frobenius_error;
  std::optional<double> forgetting_canary;
};

/// Gate for mid-cadence fields (default off — zero inference impact).
struct CriticalityBuildOptions {
  bool enable_mid = false;
  std::int64_t step = 0;
  int mid_subsample_interval = 64;
};

const char* criticality_tier_label(CriticalityTier tier);
const char* criticality_cadence_label(CriticalityCadence cadence);

double criticality_field_distance(double value, CriticalityTier tier, double target_or_bound);

/// Populate hot fields from profiler NIG state + optional session extras.
CriticalityVector build_criticality_vector(const IntelligenceProfiler& profiler,
                                           const CriticalityHotInput& hot = {},
                                           const CriticalityMidInput& mid = {},
                                           const CriticalityBuildOptions& opts = {});

/// Hot snapshot from ``SoftWorldMonitor`` maturation / query quality (Paper V).
CriticalityHotInput hot_input_from_soft_world(const SoftWorldMonitor& monitor,
                                              double drift_score = 0.0, double ood_rate = 0.0);

nlohmann::json criticality_vector_to_json(const CriticalityVector& vec);

}  // namespace cypha::intelligence
