#pragma once
//
// hp/stat_gates.hpp — mixer weight-set selector used by gate24.
//
// argmax_bin: index (0..15) of the most confident of the first 16 experts.
// The other Stats-for-Compression selectors (rec_branch3, disp_var,
// mm_len2, shape6, agree) did not pay on enwik8 and were removed.
//
// Integer-only. Encoder and decoder share the same bytes.

#include <cstdint>
#include <vector>

#include "hp/int_math.hpp"
#include "hp/models.hpp"

namespace hp {

inline int argmax_bin(const int* p12, int n) {
    int best_i = 0, best_a = -1;
    for (int i = 0; i < n && i < 16; ++i) {
        const int a = p12[i] > 2048 ? p12[i] - 2048 : 2048 - p12[i];
        if (a > best_a) {
            best_a = a;
            best_i = i;
        }
    }
    return best_i;
}

}  // namespace hp
