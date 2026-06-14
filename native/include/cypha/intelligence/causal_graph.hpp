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

/// One acquisition → simulation → maturation cycle (Paper V trajectory entry).
struct SimulationStepEvent {
  double r_eu_before = 0.0;
  double r_eu_after = 0.0;
  double resolution = 0.0;
  double maturation_level = 0.0;
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

  /// Full Paper V cycle: acquisition, simulation resolution, maturation snapshot.
  void simulation_step(double r_eu_before, double r_eu_after, double resolution);

  const SoftWorldMonitor& soft_world() const { return soft_world_; }
  const std::vector<CausalEdge>& edges() const { return edges_; }
  const std::vector<SimulationStepEvent>& trajectory() const { return trajectory_; }

  nlohmann::json to_json() const;
  nlohmann::json trajectory_json() const;

 private:
  SoftWorldMonitor soft_world_;
  std::vector<CausalEdge> edges_;
  std::vector<SimulationStepEvent> trajectory_;
  ProfileObservation last_obs_{};
  bool has_last_obs_{false};
};

}  // namespace cypha::intelligence
