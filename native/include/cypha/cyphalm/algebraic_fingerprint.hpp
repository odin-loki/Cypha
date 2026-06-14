#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace cypha::cyphalm {

/// H22: Izaac-style algebraic fingerprint tag from field state.
inline double algebraic_fingerprint(const double* field, int dim) {
    if (dim <= 0 || field == nullptr) {
        return 0.0;
    }
    static constexpr std::uint32_t kPrimes[8] = {2, 3, 5, 7, 11, 13, 17, 19};
    double acc = 0.0;
    for (int i = 0; i < dim; ++i) {
        const std::uint32_t p = kPrimes[static_cast<std::size_t>(i) % 8u];
        acc += field[i] * static_cast<double>(p);
    }
    return std::tanh(acc * 0.01);
}

inline void mix_algebraic_fingerprint(std::vector<double>& target, const double* field, int dim,
                                      double weight = 0.02) {
    if (target.empty() || dim <= 0 || field == nullptr) {
        return;
    }
    const double tag = algebraic_fingerprint(field, dim);
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] += weight * tag * std::sin(static_cast<double>(i) * 0.17 + tag);
    }
}

}  // namespace cypha::cyphalm
