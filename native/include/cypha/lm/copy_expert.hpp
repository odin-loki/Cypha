#pragma once

/// The copy expert (A1c, PLAN B9; docs/MODULES.md, module lm): the next byte of the longest earlier repeat of the
/// context's last bytes, inside a window of the current context. Moved here from tools/lm/cyphalm_trie.cpp unchanged
/// (2026-11-03), so the content mixer (cypha/online/content_mixer.hpp) and the samplers share one copy.
/// copy_bucket gives the mixer's bucket of a match length; copy_clip zeroes the lambdas of the short-match buckets.

#include <cstddef>
#include <cstdint>

namespace cypha::cyphalm {

// Copy expert (A1c, PLAN B9): the longest suffix of s[0..i), 3 to 64 bytes, that also ends at an earlier j inside the
// last 16,384 bytes of the current context (the prompt and the answer so far, or the stream being scored). It predicts
// s[j], from the most recent of the longest matches. Returns -1 (n = 0) when there is none. The window was 4,096 bytes
// until B4C (B4B chose 16,384; a Q/A prompt and its answer fit inside either).
inline constexpr std::size_t kCopyWin = 16384;
inline constexpr int kCopyBuckets = 8;
inline int copy_pred(const std::uint8_t* s, std::size_t i, int& n, std::size_t* at = nullptr) {  // DK2: at = the source j
    n = 0;
    int pred = -1;
    const std::size_t lo = i > kCopyWin ? i - kCopyWin : 0;
    for (std::size_t j = i; j-- > lo + 1;) {
        int L = 0;
        while (L < 64 && j - lo > static_cast<std::size_t>(L) && s[j - 1 - L] == s[i - 1 - L]) ++L;
        if (L > n) {
            n = L;
            pred = s[j];
            if (at) *at = j;
        }
    }
    if (n < 3) n = 0, pred = -1;
    return pred;
}
inline int copy_bucket(int n) {
    return n == 0 ? 0 : n < 4 ? 1 : n < 6 ? 2 : n < 8 ? 3 : n < 12 ? 4 : n < 16 ? 5 : n < 32 ? 6 : 7;
}
// --copy-min m (B4B): copy only from matches of at least m bytes; the lambdas of buckets starting below m become 0.
inline void copy_clip(double* lam, int m) {
    static const int lo[kCopyBuckets] = {0, 3, 4, 6, 8, 12, 16, 32};
    for (int b = 0; b < kCopyBuckets; ++b)
        if (lo[b] < m) lam[b] = 0;
}

}  // namespace cypha::cyphalm
