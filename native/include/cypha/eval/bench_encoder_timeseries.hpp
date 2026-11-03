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

/// Per-series z-score (mean 0, std 1); no-op on empty / near-constant series.
void zscore_series_inplace(std::vector<double>& series);

/// First temporal difference; empty input → empty output.
std::vector<double> series_first_diff(const std::vector<double>& series);

struct D10EcgFeatureConfig {
    bool enrich{false};
    bool zscore_series{true};
    bool include_diff{true};
    bool full_window{true};
    bool standardize_features{true};
    int n_fft{32};
    int train_passes{44};
};

D10EcgFeatureConfig d10_ecg_feature_config_from_env(const std::string& data_source);

/// Encode one ECG series; when ``cfg.enrich`` concatenates raw + diff features.
std::vector<float> encode_ecg_series(const TimeSeriesEncoder& enc, const std::vector<double>& series,
                                     const D10EcgFeatureConfig& cfg);

EcgSplit load_ecg5000(std::uint64_t seed);

struct FinancialWindowDataset {
    std::vector<std::vector<double>> x;
    std::vector<int> y;
};

FinancialWindowDataset load_financial_returns(std::uint64_t seed);

}  // namespace cypha::bench
