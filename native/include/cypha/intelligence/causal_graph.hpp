#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/intelligence/intelligence_profiler.hpp"
#include "cypha/intelligence/soft_world_monitor.hpp"

namespace cypha::intelligence {

/// Paper V causal graph stub: tracks P-space edges and soft-world maturation signals.
struct CausalEdge {
  std::string from;
  std::string to;
  double weight = 0.0;
};

class CausalGraphMonitor {
 public:
  CausalGraphMonitor();

  /// Record a directed causal influence (weight in [0, 1]).
  void record_edge(std::string from, std::string to, double weight);

  /// Ingest profiler observation and update default Paper V edges + soft-world monitor.
  void observe_profile(const ProfileObservation& obs);

  /// Record acquisition / simulation events into the embedded soft-world monitor.
  void record_acquisition(double r_eu_before, double r_eu_after);
  void record_simulation(double resolution);

  const SoftWorldMonitor& soft_world() const { return soft_world_; }
  const std::vector<CausalEdge>& edges() const { return edges_; }

  nlohmann::json to_json() const;

 private:
  SoftWorldMonitor soft_world_;
  std::vector<CausalEdge> edges_;
  ProfileObservation last_obs_{};
  bool has_last_obs_{false};
};

}  // namespace cypha::intelligence
