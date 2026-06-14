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

std::vector<double> ReversibleSSMCell::backward_stub() const {
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
