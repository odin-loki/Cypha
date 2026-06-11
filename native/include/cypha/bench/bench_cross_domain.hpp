#pragma once

#include "cypha/bench/bench_report_json.hpp"

namespace cypha::bench {

ProfileJson run_uncertainty_calibration();
ProfileJson run_online_adaptation();
ProfileJson run_forgetting_resistance();
ProfileJson run_alpha_spectrum_global();

/// Run all four cross-domain analyses and write ``cross_*.json`` tables.
void run_all_cross_domain();

}  // namespace cypha::bench
