#pragma once

#include <algorithm>
#include <cmath>

namespace cypha::cyphalm {

/// Clamp a scalar to [0, 1] for Sheffer-style soft logic.
inline double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

/// Sheffer NAND on clamped doubles: eml(x, y) = ~(x & y) with soft AND.
inline double eml_nand(double x, double y) {
    const double xc = clamp01(x);
    const double yc = clamp01(y);
    return 1.0 - xc * yc;
}

/// Derivatives of ``eml_nand`` w.r.t. raw inputs (interior of clamp only).
inline void eml_nand_grad(double x, double y, double grad_out, double& grad_x, double& grad_y) {
    const double xc = clamp01(x);
    const double yc = clamp01(y);
    grad_x = 0.0;
    grad_y = 0.0;
    if (x > 0.0 && x < 1.0) {
        grad_x = -yc * grad_out;
    }
    if (y > 0.0 && y < 1.0) {
        grad_y = -xc * grad_out;
    }
}

}  // namespace cypha::cyphalm
