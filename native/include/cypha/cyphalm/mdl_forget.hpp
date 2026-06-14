#pragma once

#include <cmath>
#include <vector>

namespace cypha::cyphalm {

/// H12: project hidden state onto L2 ball (MDL-style norm cap).
inline void mdl_forget_project(std::vector<double>& h, double max_norm) {
    if (max_norm <= 0.0 || h.empty()) {
        return;
    }
    double sq = 0.0;
    for (double v : h) {
        sq += v * v;
    }
    const double norm = std::sqrt(sq);
    if (norm > max_norm) {
        const double scale = max_norm / norm;
        for (double& v : h) {
            v *= scale;
        }
    }
}

}  // namespace cypha::cyphalm
