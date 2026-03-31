#pragma once

#include <cstddef>

namespace cypha::detail {

constexpr std::size_t kBesselN = 16384;
constexpr double kBesselX0 = 1e-6;
constexpr double kBesselX1 = 120.0;

extern const double kBesselX[kBesselN];
extern const double kBesselK2K1[kBesselN];
/// K₀(x)/K₁(x) on the same grid as ``kBesselX`` (for GIG E[V], λ=-1).
extern const double kBesselK0K1[kBesselN];

}  // namespace cypha::detail
