#pragma once

#include "cypha/intelligence/nig_statistic_state.hpp"

namespace cypha::intelligence {

/// Tracks soft-world maturation via acquisition and simulation resolution (Paper V §6).
class SoftWorldMonitor {
 public:
  SoftWorldMonitor();

  /// ``query_value = r_eu_before - r_eu_after`` after uncertainty-driven acquisition.
  void record_acquisition(double r_eu_before, double r_eu_after);

  /// ``resolution`` is epistemic uncertainty reduced by simulation.
  void record_simulation(double resolution);

  /// Acquisition then simulation in one Paper V maturation step.
  void simulation_step(double r_eu_before, double r_eu_after, double resolution);

  /// Running mean of positive resolution events (world model quality proxy).
  double maturation_level() const { return world_model_nig_.mean(); }

  /// Running mean of query usefulness.
  double query_quality() const { return query_quality_nig_.mean(); }

 private:
  NigStatisticState world_model_nig_;
  NigStatisticState query_quality_nig_;
};

}  // namespace cypha::intelligence
