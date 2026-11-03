#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha::bench {

using DomainJson = nlohmann::json;

struct DomainSpec {
    std::string tag;
    std::string module_path;
    std::function<DomainJson()> run;
};

void set_tool_dir(const std::filesystem::path& dir);
void set_ssm_diagnose(bool enabled);
int domain_number(const std::string& tag);
std::vector<DomainSpec> all_domains();

/// D16B EWC strength sweep (baseline + several `ewc_lambda` settings, each with/without NIG
/// world-field protection), reporting task A forgetting_score and task B accuracy at each
/// setting. Standalone -- not registered in `all_domains()`, does not call `finalize_domain`, and
/// never writes to `bench/report/tables/` or `bench/BASELINE_REPORT.md`. See
/// native/tools/ewc_d16b_sweep.cpp and docs/reports/EWC_D16B_SCOPING_2026-07-12.md.
DomainJson run_d16_ewc_sweep();

/// D15C FGSM-proxy robustness curve (accuracy vs perturbation epsilon). MC3 Addendum 2.
DomainJson run_d15_fgsm_robustness_curve(const std::vector<double>& epsilons, int max_eval = -1,
                                          std::uint64_t seed = 42);

}  // namespace cypha::bench
