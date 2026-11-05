#pragma once

/// The copy expert (A1c, PLAN B9; docs/MODULES.md, module lm): the next byte of the longest earlier repeat of the
/// context's last bytes, inside a window of the current context. Moved here from tools/lm/cyphalm_trie.cpp unchanged
/// (2026-11-03), so the content mixer (cypha/online/content_mixer.hpp) and the samplers share one copy.
/// copy_bucket gives the mixer's bucket of a match length; copy_clip zeroes the lambdas of the short-match buckets.

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define CYPHA_COPY_SSE2 1
#endif

namespace cypha::cyphalm {

// Copy expert (A1c, PLAN B9): the longest suffix of s[0..i), 3 to 64 bytes, that also ends at an earlier j inside the
// last 16,384 bytes of the current context (the prompt and the answer so far, or the stream being scored). It predicts
// s[j], from the most recent of the longest matches. Returns -1 (n = 0) when there is none. The window was 4,096 bytes
// until B4C (B4B chose 16,384; a Q/A prompt and its answer fit inside either).
//
// O3 (docs/reports/OPT_BOOLEAN.md): the same answer, faster. The scan visits j from i - 1 down to lo + 1 and keeps the
// first j whose match length L beats the best so far, n. A j can beat n only if s[j - 1] = s[i - 1] (else L = 0) and
// s[j - 1 - n] = s[i - 1 - n] (else L <= n), and only if j - lo > n and n < 64 (L is at most j - lo and at most 64).
// So the scan skips every other j, and finds the rest 64 positions at a time: SSE2 byte compares give a 64-bit mask,
// walked from its top bit (the highest j first, as before). Each j that passes is measured exactly as before.
inline constexpr std::size_t kCopyWin = 16384;
inline constexpr int kCopyBuckets = 8;
inline int copy_pred(const std::uint8_t* s, std::size_t i, int& n, std::size_t* at = nullptr) {  // DK2: at = the source j
    n = 0;
    int pred = -1;
    const std::size_t lo = i > kCopyWin ? i - kCopyWin : 0;
    const auto measure = [&](std::size_t j) {  // the old loop's body for one j
        int L = 0;
        while (L < 64 && j - lo > static_cast<std::size_t>(L) && s[j - 1 - L] == s[i - 1 - L]) ++L;
        if (L > n) {
            n = L;
            pred = s[j];
            if (at) *at = j;
        }
    };
    if (i >= lo + 2) {
        const std::uint8_t c0 = s[i - 1];
        std::size_t j = i - 1;  // the next j to visit, downwards
#ifdef CYPHA_COPY_SSE2
        // Block: j - 63 .. j, positions p = j' - 1 in [j - 64, j - 1]; byte k of a load at s + j - 64 is j' = j - 63 + k.
        // Both loads stay inside [lo, i - 1): j - 64 - n >= lo and j <= i - 1.
        while (n < 64 && j >= lo + 64 + static_cast<std::size_t>(n)) {
            const std::size_t nn = static_cast<std::size_t>(n);
            const __m128i v0 = _mm_set1_epi8(static_cast<char>(c0)), vn = _mm_set1_epi8(static_cast<char>(s[i - 1 - nn]));
            const std::uint8_t* a = s + (j - 64);
            const std::uint8_t* b = a - nn;
            std::uint64_t m = 0;
            for (int q = 0; q < 4; ++q) {
                const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + 16 * q));
                const __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 16 * q));
                const unsigned mq = static_cast<unsigned>(_mm_movemask_epi8(_mm_and_si128(_mm_cmpeq_epi8(x, v0), _mm_cmpeq_epi8(y, vn))));
                m |= static_cast<std::uint64_t>(mq) << (16 * q);
            }
            while (m) {  // highest j first; a j whose bit was set with a smaller n is still measured exactly
                const int k = 63 - std::countl_zero(m);
                m &= ~(std::uint64_t{1} << k);
                measure(j - 63 + static_cast<std::size_t>(k));
            }
            j -= 64;
        }
#endif
        for (; n < 64 && j > lo + static_cast<std::size_t>(n); --j)
            if (s[j - 1] == c0 && s[j - 1 - static_cast<std::size_t>(n)] == s[i - 1 - static_cast<std::size_t>(n)]) measure(j);
    }
    if (n < 3) n = 0, pred = -1;
    return pred;
}
// O3: the bucket of a match length, as the chain of compares n == 0 ? 0 : n < 4 ? 1 : n < 6 ? 2 : n < 8 ? 3 : n < 12 ? 4
// : n < 16 ? 5 : n < 32 ? 6 : 7 gives it for every int n, read from a table of n = -1 .. 64 (n < 0 gives 1, n > 64 gives 7).
inline int copy_bucket(int n) {
    static constexpr std::uint8_t kB[66] = {1, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
                                            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                                            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    return kB[std::clamp(n, -1, 64) + 1];
}
// --copy-min m (B4B): copy only from matches of at least m bytes; the lambdas of buckets starting below m become 0.
inline void copy_clip(double* lam, int m) {
    static const int lo[kCopyBuckets] = {0, 3, 4, 6, 8, 12, 16, 32};
    for (int b = 0; b < kCopyBuckets; ++b)
        if (lo[b] < m) lam[b] = 0;
}

}  // namespace cypha::cyphalm
