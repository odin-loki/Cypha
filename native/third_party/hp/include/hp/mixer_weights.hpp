#pragma once
//
// hp/mixer_weights.hpp — compile-time mixer weight representation.
//
// Weights are Q16 fixed-point. HP_MIXER_W16 stores (weight >> 1) in int16,
// doubling on read; this preserves the full ±(1<<HP_MIXER_CLAMP_BITS) range
// while halving mixer weight RAM. Verified byte-identical on proxy corpora.

#include <cstdint>
#include <type_traits>

#include "hp/features.hpp"

namespace hp {

inline constexpr int kMixerClamp = (1 << 16);

/// Mixer learning rates run in 1/16ths (Q4): the update is
/// (x * rate_q4) >> 18, which equals (x * rate) >> 14 for rate_q4 = 16 rate,
/// so trained integer rates are bit-identical and a serve-time scale can go
/// below 1 (MixerNet::set_rate_scale).
inline constexpr int kMixerRateFrac = 4;
inline constexpr int kMixerRateShift = 14 + kMixerRateFrac;

using MixerWt = std::int32_t;
inline std::int32_t mixer_wt_expand(MixerWt w) { return w; }
inline MixerWt mixer_wt_pack(std::int32_t w) { return w; }

using MixerSt = std::int16_t;

}  // namespace hp
