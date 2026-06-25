#pragma once



#include <cstdint>



#include <nlohmann/json.hpp>



#include "cypha/cyphalm/cyphalm_config.hpp"

#include "cypha/intelligence/profile_guided_loss.hpp"



namespace cypha::intelligence {

class IntelligenceProfiler;

}



namespace cypha::cyphalm {



/// Enable all existing math modules for hybrid LM (safe combination preset).

void apply_math_integration_preset(CyphaLMConfig& cfg);



/// Alias for ``apply_math_integration_preset`` (CLI / bench entry point).

inline void apply_math_integration(CyphaLMConfig& cfg) { apply_math_integration_preset(cfg); }



struct MathIntegrationExportOptions {

    const cypha::intelligence::KappaTrajectoryState* kappa_trajectory = nullptr;

    std::uint32_t step_count = 0;

};



nlohmann::json export_math_integration_report(

    const cypha::intelligence::IntelligenceProfiler& profiler, const CyphaLMConfig& cfg,

    const MathIntegrationExportOptions& opts = {});



}  // namespace cypha::cyphalm

