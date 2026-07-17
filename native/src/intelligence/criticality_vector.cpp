#include "cypha/intelligence/criticality_vector.hpp"

#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

#include "cypha/env.hpp"
#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/soft_world_monitor.hpp"

namespace cypha::intelligence {

namespace {

constexpr double kAlphaTarget = 0.5;
constexpr double kRoutingEntropyTarget = 0.8;
constexpr double kDeadExpertTarget = 0.0;
constexpr double kSpectralRadiusTarget = 1.0;
constexpr double kNigConfidenceTarget = 0.82;
constexpr double kEssTarget = 8.0;
constexpr double kOodRateBound = 0.05;
constexpr double kKernelErrBound = 0.01;
constexpr double kForgettingBound = 0.001;
constexpr double kAnomalyTarget = 0.5;
constexpr double kDriftTarget = 0.0;

CriticalityField make_field(const char* name, double value, CriticalityTier tier,
                            double target_or_bound, CriticalityCadence cadence, bool available = true) {
  CriticalityField f;
  f.name = name;
  f.value = value;
  f.tier = tier;
  f.target_or_bound = target_or_bound;
  f.cadence = cadence;
  f.available = available;
  f.distance = criticality_field_distance(value, tier, target_or_bound);
  return f;
}

double mean_nig_lambda(const IntelligenceProfiler& profiler) {
  const auto counts = profiler.get_statistic_update_counts();
  double sum = 0.0;
  int n = 0;
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    if (counts[i] > 0) {
      sum += static_cast<double>(counts[i]);
      n += 1;
    }
  }
  if (n <= 0) {
    return 1.0;
  }
  return sum / static_cast<double>(n);
}

double mean_nig_confidence(const IntelligenceProfiler& profiler) {
  const auto matrix = profiler.get_profile_matrix();
  double sum = 0.0;
  int n = 0;
  for (std::size_t i = 0; i < kProfileStatisticCount; ++i) {
    const double epistemic = matrix[i][1];
    if (std::isfinite(epistemic) && epistemic >= 0.0) {
      sum += 1.0 / (1.0 + epistemic);
      n += 1;
    }
  }
  if (n <= 0) {
    return 0.5;
  }
  return sum / static_cast<double>(n);
}

bool mid_gate_open(const CriticalityBuildOptions& opts) {
  if (!opts.enable_mid) {
    return false;
  }
  const int interval = std::max(1, opts.mid_subsample_interval);
  return opts.step % static_cast<std::int64_t>(interval) == 0;
}

}  // namespace

const char* criticality_tier_label(CriticalityTier tier) {
  switch (tier) {
    case CriticalityTier::Tier1:
      return "T1";
    case CriticalityTier::Tier2:
      return "T2";
  }
  return "T2";
}

const char* criticality_cadence_label(CriticalityCadence cadence) {
  switch (cadence) {
    case CriticalityCadence::Hot:
      return "hot";
    case CriticalityCadence::Mid:
      return "mid";
    case CriticalityCadence::Cold:
      return "cold";
  }
  return "hot";
}

double criticality_field_distance(double value, CriticalityTier tier, double target_or_bound) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  if (tier == CriticalityTier::Tier1) {
    return std::max(0.0, value - target_or_bound);
  }
  return std::abs(value - target_or_bound);
}

bool criticality_vector_enabled() {
  if (const std::optional<std::string> raw = cypha::env_get("CYPHA_CRITICALITY"); raw.has_value()) {
    if (raw->empty()) {
      return true;
    }
    const char c0 = (*raw)[0];
    if (c0 == '0') {
      return false;
    }
    if (c0 == 'f' || c0 == 'F') {
      return false;
    }
    if (c0 == 'n' || c0 == 'N') {
      return false;
    }
  }
  return true;
}

CriticalityVector build_criticality_vector(const IntelligenceProfiler& profiler,
                                           const CriticalityHotInput& hot,
                                           const CriticalityMidInput& mid,
                                           const CriticalityBuildOptions& opts) {
  const auto matrix = profiler.get_profile_matrix();
  const double alpha = matrix[0][0];

  CriticalityVector out;
  out.mid_enabled = mid_gate_open(opts);

  out.fields.push_back(make_field("alpha", alpha, CriticalityTier::Tier2, kAlphaTarget,
                                  CriticalityCadence::Hot));

  if (hot.routing_entropy.has_value()) {
    out.fields.push_back(make_field("routing_entropy", *hot.routing_entropy, CriticalityTier::Tier2,
                                    kRoutingEntropyTarget, CriticalityCadence::Hot));
  } else {
    out.fields.push_back(make_field("routing_entropy", 0.0, CriticalityTier::Tier2, kRoutingEntropyTarget,
                                    CriticalityCadence::Hot, false));
  }

  if (hot.dead_expert_fraction.has_value()) {
    out.fields.push_back(make_field("dead_expert_fraction", *hot.dead_expert_fraction,
                                    CriticalityTier::Tier2, kDeadExpertTarget, CriticalityCadence::Hot));
  } else {
    out.fields.push_back(make_field("dead_expert_fraction", 0.0, CriticalityTier::Tier2,
                                    kDeadExpertTarget, CriticalityCadence::Hot, false));
  }

  out.fields.push_back(make_field("anomaly_score", hot.anomaly_score, CriticalityTier::Tier2,
                                  kAnomalyTarget, CriticalityCadence::Hot));
  out.fields.push_back(make_field("drift_score", hot.drift_score, CriticalityTier::Tier2, kDriftTarget,
                                  CriticalityCadence::Hot));

  const double nig_conf =
      hot.nig_confidence > 0.0 ? hot.nig_confidence : mean_nig_confidence(profiler);
  out.fields.push_back(make_field("nig_confidence", nig_conf, CriticalityTier::Tier2,
                                  kNigConfidenceTarget, CriticalityCadence::Hot));

  const double ess =
      hot.effective_sample_size > 0.0 ? hot.effective_sample_size : mean_nig_lambda(profiler);
  out.fields.push_back(make_field("effective_sample_size", ess, CriticalityTier::Tier2, kEssTarget,
                                  CriticalityCadence::Hot));
  out.fields.push_back(make_field("ood_rate", hot.ood_rate, CriticalityTier::Tier1, kOodRateBound,
                                  CriticalityCadence::Hot));

  if (out.mid_enabled) {
    if (mid.spectral_radius.has_value()) {
      out.fields.push_back(make_field("spectral_radius", *mid.spectral_radius, CriticalityTier::Tier2,
                                      kSpectralRadiusTarget, CriticalityCadence::Mid));
    } else {
      out.fields.push_back(make_field("spectral_radius", alpha, CriticalityTier::Tier2,
                                      kSpectralRadiusTarget, CriticalityCadence::Mid, false));
    }
    if (mid.kernel_frobenius_error.has_value()) {
      out.fields.push_back(make_field("kernel_frobenius_error", *mid.kernel_frobenius_error,
                                      CriticalityTier::Tier1, kKernelErrBound, CriticalityCadence::Mid));
    } else {
      out.fields.push_back(make_field("kernel_frobenius_error", 0.0, CriticalityTier::Tier1,
                                      kKernelErrBound, CriticalityCadence::Mid, false));
    }
    if (mid.forgetting_canary.has_value()) {
      out.fields.push_back(make_field("forgetting_canary", *mid.forgetting_canary, CriticalityTier::Tier1,
                                      kForgettingBound, CriticalityCadence::Mid));
    } else {
      out.fields.push_back(make_field("forgetting_canary", 0.0, CriticalityTier::Tier1, kForgettingBound,
                                      CriticalityCadence::Mid, false));
    }
  }

  return out;
}

CriticalityHotInput hot_input_from_soft_world(const SoftWorldMonitor& monitor, double drift_score,
                                              double ood_rate) {
  CriticalityHotInput hot;
  hot.drift_score = drift_score;
  hot.ood_rate = ood_rate;
  hot.nig_confidence = monitor.maturation_level();
  hot.anomaly_score = 1.0 - monitor.query_quality();
  return hot;
}

nlohmann::json criticality_vector_to_json(const CriticalityVector& vec) {
  nlohmann::json fields = nlohmann::json::array();
  for (const CriticalityField& f : vec.fields) {
    fields.push_back(nlohmann::json{
        {"name", f.name},
        {"value", f.value},
        {"tier", criticality_tier_label(f.tier)},
        {"target_or_bound", f.target_or_bound},
        {"cadence", criticality_cadence_label(f.cadence)},
        {"distance", f.distance},
        {"available", f.available},
    });
  }
  return nlohmann::json{
      {"fields", fields},
      {"mid_enabled", vec.mid_enabled},
  };
}

}  // namespace cypha::intelligence
