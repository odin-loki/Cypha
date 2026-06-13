#pragma once

#include <nlohmann/json.hpp>

#include "cypha/intelligence/intelligence_profiler.hpp"

namespace cypha::intelligence {

/// Export native intelligence profiler state as REST/diagnostics JSON.
nlohmann::json intelligence_profile_to_json(const IntelligenceProfiler& profiler);

}  // namespace cypha::intelligence
