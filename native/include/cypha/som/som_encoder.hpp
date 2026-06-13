#pragma once

/// Minimal online SOM grid over encoder features (U2; off by default).

#include <cstdint>
#include <random>
#include <vector>

namespace cypha::som {

struct OnlineSOMConfig {
    int k{16};
    double eta0{0.3};
    double sigma0{4.0};
    int T{10000};
    std::uint64_t seed{42};
};

class OnlineSOMEncoder {
public:
    OnlineSOMEncoder(int d_in, OnlineSOMConfig cfg = {});

    int dim_in() const { return d_in_; }
    int grid_k() const { return k_; }
    int n_units() const { return n_units_; }
    int step_count() const { return t_; }

    /// BMU weight vector (copy). When train=true, updates weights with Gaussian neighborhood.
    std::vector<double> encode(const std::vector<double>& z, bool train = true);

    std::vector<std::vector<double>> batch_encode(const std::vector<std::vector<double>>& Z, bool train = true);

private:
    int d_in_;
    int k_;
    int n_units_;
    double eta0_;
    double sigma0_;
    int T_;
    int t_{0};
    std::vector<std::vector<double>> W_;
    std::vector<std::pair<double, double>> positions_;
};

}  // namespace cypha::som
