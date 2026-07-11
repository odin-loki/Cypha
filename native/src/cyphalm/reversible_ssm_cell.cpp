#include "cypha/cyphalm/reversible_ssm_cell.hpp"

#include <cmath>

namespace cypha::cyphalm {

namespace {

std::vector<double> tanh_vec(const std::vector<double>& v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i] = std::tanh(v[i]);
    }
    return out;
}

}  // namespace

void ReversibleSSMCell::reset() {
    last_x_.clear();
    last_delta_.clear();
    last_y_.clear();
    has_pair_ = false;
}

std::vector<double> ReversibleSSMCell::forward(const std::vector<double>& x,
                                               const std::vector<double>& delta) {
    const std::size_t n = x.size();
    last_x_ = x;
    last_delta_.assign(n, 0.0);
    const auto f_delta = tanh_vec(delta);
    last_y_.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double d = i < delta.size() ? delta[i] : 0.0;
        last_delta_[i] = d;
        last_y_[i] = x[i] + f_delta[i];
    }
    has_pair_ = !last_y_.empty();
    return last_y_;
}

// Exact inverse of forward(): forward computes y_i = x_i + tanh(delta_i) and caches delta_i
// verbatim in last_delta_ (not re-derived from anything else). Substituting that identical
// cached delta_i back in gives:
//     x_hat_i = y_i - tanh(delta_i) = (x_i + tanh(delta_i)) - tanh(delta_i) = x_i
// exactly (up to floating-point rounding of the two additions/subtraction — tanh(delta_i) is
// evaluated once in forward() and delta_i is stored as-is, so there is no re-evaluation drift).
// This is why the "stub" name was misleading: there is no missing piece here (delta does not
// need to be reconstructed from anything — it is simply cached), and every call site below
// only ever calls this after forward() populated last_delta_/last_y_ for the *same* delta.
std::vector<double> ReversibleSSMCell::reconstruct() const {
    if (!has_pair_) {
        return {};
    }
    std::vector<double> x_hat(last_y_.size(), 0.0);
    for (std::size_t i = 0; i < last_y_.size(); ++i) {
        const double d = i < last_delta_.size() ? last_delta_[i] : 0.0;
        x_hat[i] = last_y_[i] - std::tanh(d);
    }
    return x_hat;
}

}  // namespace cypha::cyphalm
