#pragma once
//
// Integer mixer dots. HP_XSIMD=0 uses scalar loops; HP_XSIMD=1 (default) uses SSE4.1.
// int64 add is associative so the vectorized sum matches the scalar
// sum exactly. axpy_shift_clamp stays scalar (update is less hot).

#include <cstdint>

#include "hp/features.hpp"
#include "hp/int_math.hpp"
#include "hp/mixer_weights.hpp"

#if HP_XSIMD
#ifndef __SSE4_1__
#error HP_XSIMD=1 requires SSE4.1 (compile with -msse4.1)
#endif
#include <smmintrin.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <xsimd/xsimd.hpp>
#endif

namespace hp {

inline std::int64_t dot_i32(const std::int32_t* a, const std::int32_t* b, int n) {
#if HP_XSIMD
    std::int64_t sum = 0;
    int i = 0;
#ifdef __AVX2__
    using batch32_8 = xsimd::batch<std::int32_t, xsimd::avx2>;
    __m256i acc8 = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        const __m256i va = batch32_8::load_unaligned(a + i);
        const __m256i vb = batch32_8::load_unaligned(b + i);
        const __m256i even = _mm256_mul_epi32(va, vb);
        const __m256i odd = _mm256_mul_epi32(_mm256_srli_epi64(va, 32),
                                            _mm256_srli_epi64(vb, 32));
        acc8 = _mm256_add_epi64(acc8, _mm256_add_epi64(even, odd));
    }
    {
        const __m128i lo = _mm256_castsi256_si128(acc8);
        const __m128i hi = _mm256_extracti128_si256(acc8, 1);
        const __m128i s = _mm_add_epi64(lo, hi);
        const __m128i h = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
        sum += _mm_cvtsi128_si64(h);
    }
#endif
    using batch32_4 = xsimd::batch<std::int32_t, xsimd::sse4_1>;
    __m128i acc4 = _mm_setzero_si128();
    for (; i + 4 <= n; i += 4) {
        const __m128i va = batch32_4::load_unaligned(a + i);
        const __m128i vb = batch32_4::load_unaligned(b + i);
        const __m128i even = _mm_mul_epi32(va, vb);
        const __m128i odd = _mm_mul_epi32(_mm_srli_epi64(va, 32),
                                         _mm_srli_epi64(vb, 32));
        acc4 = _mm_add_epi64(acc4, _mm_add_epi64(even, odd));
    }
    {
        const __m128i h = _mm_add_epi64(acc4, _mm_unpackhi_epi64(acc4, acc4));
        sum += _mm_cvtsi128_si64(h);
    }
    for (; i < n; ++i) sum += static_cast<std::int64_t>(a[i]) * b[i];
    return sum;
#else
    std::int64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += static_cast<std::int64_t>(a[i]) * b[i];
    return sum;
#endif
}

inline std::int64_t dot_i32_i16(const std::int32_t* w, const std::int16_t* st, int n) {
#if HP_XSIMD
    std::int64_t sum = 0;
    int i = 0;
    using batch32_4 = xsimd::batch<std::int32_t, xsimd::sse4_1>;
    for (; i + 4 <= n; i += 4) {
        const __m128i ws = batch32_4::load_unaligned(w + i);
        const __m128i ss =
            _mm_cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(st + i)));
        const __m128i prod = _mm_mullo_epi32(ws, ss);
        alignas(16) std::int32_t parts[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(parts), prod);
        sum += static_cast<std::int64_t>(parts[0]) + parts[1] + parts[2] + parts[3];
    }
    for (; i < n; ++i) sum += static_cast<std::int64_t>(w[i]) * st[i];
    return sum;
#else
    std::int64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += static_cast<std::int64_t>(w[i]) * st[i];
    return sum;
#endif
}

inline std::int64_t dot_mixer_wt(const MixerWt* w, const MixerSt* st, int n) {
#if HP_XSIMD && !HP_MIXER_W16 && HP_MIXER_ST16
    return dot_i32_i16(reinterpret_cast<const std::int32_t*>(w),
                       reinterpret_cast<const std::int16_t*>(st), n);
#elif HP_XSIMD && !HP_MIXER_W16 && !HP_MIXER_ST16
    return dot_i32(reinterpret_cast<const std::int32_t*>(w),
                   reinterpret_cast<const std::int32_t*>(st), n);
#else
    std::int64_t sum = 0;
    for (int i = 0; i < n; ++i)
        sum += static_cast<std::int64_t>(mixer_wt_expand(w[i])) * st[i];
    return sum;
#endif
}

inline void axpy_mixer_wt(MixerWt* w, const MixerSt* st, int n, std::int32_t err,
                          std::int32_t l1, std::int64_t energy = 0) {
#if HP_MIXER_NLMS
    // NLMS: scale the step by ETYP/||st||^2 so the effective step size is
    // invariant to how many experts are currently saturated. Integer-exact.
    const std::int64_t den = energy + HP_NLMS_EPS;
#else
    (void)energy;
#endif
    for (int i = 0; i < n; ++i) {
#if HP_MIXER_NLMS
        const std::int32_t dw = static_cast<std::int32_t>(
            ((static_cast<std::int64_t>(st[i]) * err * l1 *
              static_cast<std::int64_t>(HP_NLMS_ETYP)) / den) >> 14);
#else
        const std::int32_t dw = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(st[i]) * err * l1) >> 14);
#endif
        w[i] = mixer_wt_pack(
            clamp_int(mixer_wt_expand(w[i]) + dw, -kMixerClamp, kMixerClamp));
    }
}

}  // namespace hp
