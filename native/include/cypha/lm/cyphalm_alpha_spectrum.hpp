#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace cypha::cyphalm {

class CyphaLMModel;

struct AlphaSpectrumSnapshot {
    std::vector<double> gria_projection_alpha;
    std::vector<double> expert_alpha;
    double mean_alpha = 0.0;
    double mean_expert_alpha = 0.0;
    double fraction_near_edge_of_chaos = 0.0;
    double fraction_experts_near_edge_of_chaos = 0.0;
};

std::vector<nlohmann::json> alpha_spectrum_track(CyphaLMModel& model, int n_steps,
                                                 const std::vector<int>& train_data);

}  // namespace cypha::cyphalm
