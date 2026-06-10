#pragma once

#include <string>

namespace cypha::cyphalm {

class CyphaLMModel;

/// Save ``{base}.json`` + ``{base}.npz`` (Python-compatible layout; GRIA stored as low-rank U/V).
void save_cyphalm_model(const CyphaLMModel& model, const std::string& base_path);

/// Load checkpoint from ``{path}.json`` or ``path`` when suffix is ``.json``.
CyphaLMModel load_cyphalm_model(const std::string& json_path);

}  // namespace cypha::cyphalm
