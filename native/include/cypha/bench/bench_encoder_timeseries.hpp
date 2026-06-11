#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cypha::bench {

/// Mirrors ``TimeSeriesEncoder`` (statistical + rFFT features).
class TimeSeriesEncoder {
  public:
    explicit TimeSeriesEncoder(int window_size = 50, int n_fft_coeffs = 10);

    int feature_dim() const { return feature_dim_; }

    std::vector<float> encode_window(const std::vector<double>& window) const;
    std::pair<std::vector<std::vector<double>>, std::vector<int>> sliding_windows(const std::vector<double>& series,
                                                                                int step = 1) const;
    std::vector<float> encode_series(const std::vector<double>& series) const;

  private:
    int window_size_;
    int n_fft_coeffs_;
    int feature_dim_;
};

struct EcgSplit {
    std::string source;
    std::vector<std::vector<double>> x_train;
    std::vector<int> y_train;
    std::vector<std::vector<double>> x_test;
    std::vector<int> y_test;
};

EcgSplit load_ecg5000(std::uint64_t seed);

struct FinancialWindowDataset {
    std::vector<std::vector<double>> x;
    std::vector<int> y;
};

FinancialWindowDataset load_financial_returns(std::uint64_t seed);

}  // namespace cypha::bench
