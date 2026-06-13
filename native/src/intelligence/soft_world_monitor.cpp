#include "cypha/intelligence/soft_world_monitor.hpp"

#include <algorithm>

namespace cypha::intelligence {

SoftWorldMonitor::SoftWorldMonitor() : world_model_nig_(0.0, 1.0, 2.0, 0.1), query_quality_nig_(0.0, 1.0, 2.0, 0.1) {}

void SoftWorldMonitor::record_acquisition(double r_eu_before, double r_eu_after) {
  const double query_value = std::max(0.0, r_eu_before - r_eu_after);
  query_quality_nig_.update(query_value);
  if (query_value > 0.0) {
    world_model_nig_.update(query_value);
  }
}

void SoftWorldMonitor::record_simulation(double resolution) {
  const double clipped = std::max(0.0, resolution);
  if (clipped > 0.0) {
    world_model_nig_.update(clipped);
  }
}

}  // namespace cypha::intelligence
