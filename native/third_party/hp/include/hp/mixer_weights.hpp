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

inline constexpr int kMixerClamp = (1 << HP_MIXER_CLAMP_BITS);

#if HP_MIXER_W16
using MixerWt = std::int16_t;
inline std::int32_t mixer_wt_expand(MixerWt w) {
    return static_cast<std::int32_t>(w) << 1;
}
inline MixerWt mixer_wt_pack(std::int32_t w) {
    return static_cast<MixerWt>(w >> 1);
}
#else
using MixerWt = std::int32_t;
inline std::int32_t mixer_wt_expand(MixerWt w) { return w; }
inline MixerWt mixer_wt_pack(std::int32_t w) { return w; }
#endif

#if HP_MIXER_ST16
using MixerSt = std::int16_t;
#else
using MixerSt = std::int32_t;
#endif

}  // namespace hp
