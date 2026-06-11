#include "cypha/cyphalm/cyphalm_alpha_spectrum.hpp"

#include <cmath>
#include <numeric>

#include "cypha/cyphalm/cyphalm_model.hpp"

namespace cypha::cyphalm {

namespace {

double mean_vec(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

}  // namespace

std::vector<nlohmann::json> alpha_spectrum_track(CyphaLMModel& model, int n_steps,
                                                 const std::vector<int>& train_data) {
    std::vector<int> data = train_data;
    if (data.empty()) data.push_back(0);

    std::vector<nlohmann::json> rows;
    rows.reserve(static_cast<std::size_t>(std::max(0, n_steps)));
    for (int step = 0; step < n_steps; ++step) {
        const int denom = std::max(static_cast<int>(data.size()) - 1, 1);
        const int a = step % denom;
        if (data.size() > 1) {
            model.train_step(static_cast<std::uint32_t>(data[static_cast<std::size_t>(a)]),
                             static_cast<std::uint32_t>(data[static_cast<std::size_t>(a + 1)]));
        }
        const auto snap = model.alpha_spectrum_snapshot();
        nlohmann::json row = {
            {"step", step},
            {"mean_alpha", snap.mean_alpha},
            {"fraction_near_edge_of_chaos", snap.fraction_near_edge_of_chaos},
            {"gria_alpha_mean", mean_vec(snap.gria_projection_alpha)},
        };
        if (!snap.expert_alpha.empty()) {
            row["expert_alpha_mean"] = mean_vec(snap.expert_alpha);
        } else {
            row["expert_alpha_mean"] = nullptr;
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

}  // namespace cypha::cyphalm
