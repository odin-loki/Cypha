#include "cypha/cyphalm/ca_state_cell.hpp"

#include <cmath>
#include <cstdint>

namespace cypha::cyphalm {

namespace {

constexpr std::uint8_t kRule110 = 0b01101110;

}  // namespace

bool CAStateCell::rule110_bit(bool left, bool center, bool right) {
    const int pattern = (left ? 4 : 0) + (center ? 2 : 0) + (right ? 1 : 0);
    return (kRule110 >> pattern) & 1u;
}

std::vector<double> CAStateCell::step_rule110(const std::vector<double>& h, double scale) {
    if (h.empty()) {
        return {};
    }
    const int n = static_cast<int>(h.size());
    std::vector<bool> bits(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        bits[static_cast<std::size_t>(i)] = h[static_cast<std::size_t>(i)] >= 0.0;
    }
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    const double mag = scale > 0.0 ? scale : 1.0;
    for (int i = 0; i < n; ++i) {
        const bool left = bits[static_cast<std::size_t>((i - 1 + n) % n)];
        const bool center = bits[static_cast<std::size_t>(i)];
        const bool right = bits[static_cast<std::size_t>((i + 1) % n)];
        out[static_cast<std::size_t>(i)] = rule110_bit(left, center, right) ? mag : -mag;
    }
    return out;
}

}  // namespace cypha::cyphalm
