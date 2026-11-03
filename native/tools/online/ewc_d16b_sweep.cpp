/// D16B EWC strength sweep: prints baseline + several `ewc_lambda` settings (each with/without
/// NIG world-field protection) so the before/after forgetting_score / task-B-accuracy trade-off
/// can be inspected directly, without going through `cypha_bench_run` (which always finishes by
/// calling `build_report()` / touching `bench/BASELINE_REPORT.md` -- explicitly off-limits for
/// this investigation). See native/include/cypha/bench/bench_domains.hpp `run_d16_ewc_sweep` and
/// docs/reports/EWC_D16B_SCOPING_2026-07-12.md.
#include <cstdio>
#include <cstdlib>

#include <nlohmann/json.hpp>

#include "cypha/bench/bench_domains.hpp"

int main() {
  const nlohmann::json result = cypha::bench::run_d16_ewc_sweep();
  std::printf("%s\n", result.dump(2).c_str());

  const auto& rows = result.at("rows");
  if (!rows.is_array() || rows.empty()) {
    std::fprintf(stderr, "ewc_d16b_sweep: FAIL empty rows\n");
    return 1;
  }
  const double baseline_forgetting = rows.at(0).at("forgetting_score").get<double>();
  if (rows.at(0).at("ewc_lambda").get<double>() != 0.0) {
    std::fprintf(stderr, "ewc_d16b_sweep: FAIL first row is not the baseline (ewc_lambda != 0)\n");
    return 1;
  }

  // Every EWC-on row must carry a nonzero final penalty -- i.e. the growable-D prefix fix
  // (native/src/ewc_regularizer.cpp `squared_penalty`/`pull_toward_anchor`) is actually engaging,
  // not silently no-op'ing the moment task B introduces classes task A never saw (the pre-fix
  // bug this sweep exists to regression-test against).
  int checked = 0;
  for (const auto& row : rows) {
    const double lambda = row.at("ewc_lambda").get<double>();
    if (lambda <= 0.0) continue;
    const double penalty = row.at("ewc_penalty_final").get<double>();
    if (!(penalty > 0.0)) {
      std::fprintf(stderr,
                   "ewc_d16b_sweep: FAIL ewc_penalty_final == %.6g for lambda=%.3g protect_world_field=%s "
                   "(growable-D prefix fix regressed)\n",
                   penalty, lambda, row.at("protect_world_field").get<bool>() ? "true" : "false");
      return 1;
    }
    ++checked;
  }
  if (checked == 0) {
    std::fprintf(stderr, "ewc_d16b_sweep: FAIL no EWC-on rows found\n");
    return 1;
  }

  std::printf("ewc_d16b_sweep: baseline_forgetting=%.4f (%d EWC-on settings, all penalty>0) PASS\n",
              baseline_forgetting, checked);
  return 0;
}
