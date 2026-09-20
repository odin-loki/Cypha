#pragma once

#include <string>

namespace cypha::cyphalm {

class CyphaLMModel;

/// Save ``{base}.json`` (config metadata) + ``{base}.hpbin`` (HPCP v1 predictor state).
void save_cyphalm_model(const CyphaLMModel& model, const std::string& base_path);

/// Load checkpoint from ``{path}.json`` (and sibling ``.hpbin`` when present).
CyphaLMModel load_cyphalm_model(const std::string& json_path);

}  // namespace cypha::cyphalm
