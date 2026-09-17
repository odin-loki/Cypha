#pragma once

#include <cstddef>

namespace cypha::detail {

constexpr std::size_t kBesselN = 16384;
constexpr double kBesselX0 = 1e-6;
constexpr double kBesselX1 = 120.0;

extern const double kBesselX[kBesselN];
/// No longer read by any code path: both backends and the CUDA device path build K_2/K_1 from
/// ``kBesselK0K1`` via the exact recurrence ``K_2/K_1 = 2/x + K_0/K_1``, because interpolating the
/// 2/x pole directly was the largest numerical error in the kernel (audit finding L9). Retained
/// deliberately -- ``scripts/gen_gig_k0k1_fit.py --validate`` checks the two columns against the
/// recurrence, which is a standing cross-check on the table itself.
extern const double kBesselK2K1[kBesselN];
/// K₀(x)/K₁(x) on the same grid as ``kBesselX`` (for GIG E[V], λ=-1).
extern const double kBesselK0K1[kBesselN];

}  // namespace cypha::detail
