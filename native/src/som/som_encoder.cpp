#include "cypha/som/som_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cypha::som {

namespace {

double l2_norm(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    const int n = static_cast<int>(std::min(a.size(), b.size()));
    for (int i = 0; i < n; ++i) {
        const double d = a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)];
        s += d * d;
    }
    for (std::size_t i = static_cast<std::size_t>(n); i < a.size(); ++i) {
        s += a[i] * a[i];
    }
    for (std::size_t i = static_cast<std::size_t>(n); i < b.size(); ++i) {
        s += b[i] * b[i];
    }
    return std::sqrt(s);
}

std::vector<double> resize_or_copy(const std::vector<double>& z, int d_in) {
    if (static_cast<int>(z.size()) == d_in) {
        return z;
    }
    std::vector<double> out(static_cast<std::size_t>(d_in), 0.0);
    for (int i = 0; i < d_in; ++i) {
        out[static_cast<std::size_t>(i)] = z[static_cast<std::size_t>(i % static_cast<int>(z.size()))];
    }
    return out;
}

}  // namespace

OnlineSOMEncoder::OnlineSOMEncoder(int d_in, OnlineSOMConfig cfg)
    : d_in_(d_in),
      k_(std::max(1, cfg.k)),
      n_units_(k_ * k_),
      eta0_(cfg.eta0),
      sigma0_(cfg.sigma0),
      T_(std::max(1, cfg.T)) {
    if (d_in_ <= 0) {
        throw std::invalid_argument("OnlineSOMEncoder: d_in must be positive");
    }
    std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.seed));
    std::normal_distribution<double> gauss(0.0, 1.0);

    W_.resize(static_cast<std::size_t>(n_units_), std::vector<double>(static_cast<std::size_t>(d_in_)));
    for (int u = 0; u < n_units_; ++u) {
        for (int j = 0; j < d_in_; ++j) {
            W_[static_cast<std::size_t>(u)][static_cast<std::size_t>(j)] = gauss(rng) * 0.1;
        }
    }

    positions_.reserve(static_cast<std::size_t>(n_units_));
    for (int i = 0; i < k_; ++i) {
        for (int j = 0; j < k_; ++j) {
            positions_.emplace_back(static_cast<double>(i), static_cast<double>(j));
        }
    }
}

std::vector<double> OnlineSOMEncoder::encode(const std::vector<double>& z, bool train) {
    const std::vector<double> x = resize_or_copy(z, d_in_);

    int bmu = 0;
    double best = std::numeric_limits<double>::infinity();
    for (int u = 0; u < n_units_; ++u) {
        const double d = l2_norm(x, W_[static_cast<std::size_t>(u)]);
        if (d < best) {
            best = d;
            bmu = u;
        }
    }

    if (train) {
        const double decay = std::exp(-static_cast<double>(t_) / static_cast<double>(T_));
        const double eta = eta0_ * decay;
        const double sigma = sigma0_ * decay;
        const auto& bmu_pos = positions_[static_cast<std::size_t>(bmu)];
        const double denom = 2.0 * sigma * sigma + 1e-12;

        for (int u = 0; u < n_units_; ++u) {
            const auto& pos = positions_[static_cast<std::size_t>(u)];
            const double dx = pos.first - bmu_pos.first;
            const double dy = pos.second - bmu_pos.second;
            const double d2 = dx * dx + dy * dy;
            const double h = std::exp(-d2 / denom);
            auto& w = W_[static_cast<std::size_t>(u)];
            for (int j = 0; j < d_in_; ++j) {
                w[static_cast<std::size_t>(j)] += eta * h * (x[static_cast<std::size_t>(j)] - w[static_cast<std::size_t>(j)]);
            }
        }
        ++t_;
    }

    return W_[static_cast<std::size_t>(bmu)];
}

std::vector<std::vector<double>> OnlineSOMEncoder::batch_encode(const std::vector<std::vector<double>>& Z,
                                                                bool train) {
    std::vector<std::vector<double>> out;
    out.reserve(Z.size());
    for (const auto& row : Z) {
        out.push_back(encode(row, train));
    }
    return out;
}

}  // namespace cypha::som
