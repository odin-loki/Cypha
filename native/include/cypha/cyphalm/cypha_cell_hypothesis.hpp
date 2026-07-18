#pragma once

#include <string>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"

namespace cypha::cyphalm {

struct CellVariantSpec {
    std::string id;
    std::string name;
    int tier = 0;
    bool runnable = true;
    std::string bench_mode;
    std::string notes;
};

/// All cell-hypothesis variants (B0–B2, H01–H23, U01–U10 unified-context).
const std::vector<CellVariantSpec>& all_cell_variants();

/// Lookup by id (e.g. ``H06``). Returns nullptr when unknown.
const CellVariantSpec* find_cell_variant(const std::string& id);

/// Apply bench mode plus variant-specific ``CyphaLMConfig`` flags. Throws on unknown id.
void apply_cell_variant(const std::string& id, CyphaLMConfig& cfg);

}  // namespace cypha::cyphalm
