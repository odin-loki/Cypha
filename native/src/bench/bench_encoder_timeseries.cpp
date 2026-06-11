#include "cypha/bench/bench_encoder_timeseries.hpp"

#include "cypha/bench/bench_paths.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace cypha::bench {

namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.14159265358979323846;

double percentile(const std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::vector<double> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    const double rank = p * static_cast<double>(sorted.size() - 1);
    const int lo = static_cast<int>(std::floor(rank));
    const int hi = static_cast<int>(std::ceil(rank));
    const double frac = rank - static_cast<double>(lo);
    return sorted[static_cast<std::size_t>(lo)] * (1.0 - frac) + sorted[static_cast<std::size_t>(hi)] * frac;
}

void rfft_magnitudes(const std::vector<double>& window, int n_coeffs, std::vector<double>& out) {
    const int n = static_cast<int>(window.size());
    out.resize(static_cast<std::size_t>(n_coeffs));
    for (int k = 0; k < n_coeffs; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (int j = 0; j < n; ++j) {
            const double angle = -2.0 * kPi * static_cast<double>(k) * static_cast<double>(j) / static_cast<double>(n);
            re += window[static_cast<std::size_t>(j)] * std::cos(angle);
            im += window[static_cast<std::size_t>(j)] * std::sin(angle);
        }
        out[static_cast<std::size_t>(k)] = std::sqrt(re * re + im * im);
    }
}

EcgSplit synthetic_ecg5000(std::uint64_t seed, int n_train, int n_test, int length, int n_classes) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<double> noise(0.0, 0.05);
    auto make = [&](int n) {
        std::vector<std::vector<double>> x;
        std::vector<int> y;
        x.reserve(static_cast<std::size_t>(n));
        y.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const int cls = i % n_classes;
            std::vector<double> wave(static_cast<std::size_t>(length));
            for (int t = 0; t < length; ++t) {
                const double tt = 4.0 * kPi * static_cast<double>(t) / static_cast<double>(length);
                wave[static_cast<std::size_t>(t)] =
                    std::sin(tt + cls) + 0.3 * std::sin(3.0 * tt + cls * 0.5) + noise(rng);
            }
            x.push_back(std::move(wave));
            y.push_back(cls + 1);
        }
        return std::pair{x, y};
    };
    const auto tr = make(n_train);
    const auto te = make(n_test);
    EcgSplit split;
    split.source = "synthetic";
    split.x_train = tr.first;
    split.y_train = tr.second;
    split.x_test = te.first;
    split.y_test = te.second;
    return split;
}

bool load_ecg_file(const fs::path& path, std::vector<std::vector<double>>& x, std::vector<int>& y) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<double> vals;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ' ')) {
            if (cell.empty()) continue;
            vals.push_back(std::stod(cell));
        }
        if (vals.size() < 2) continue;
        y.push_back(static_cast<int>(vals[0]));
        vals.erase(vals.begin());
        x.push_back(std::move(vals));
    }
    return !x.empty();
}

}  // namespace

TimeSeriesEncoder::TimeSeriesEncoder(int window_size, int n_fft_coeffs)
    : window_size_(window_size), n_fft_coeffs_(n_fft_coeffs), feature_dim_(8 + n_fft_coeffs) {}

std::vector<float> TimeSeriesEncoder::encode_window(const std::vector<double>& window) const {
    if (static_cast<int>(window.size()) != window_size_) {
        throw std::runtime_error("TimeSeriesEncoder: window size mismatch");
    }
    std::vector<float> features;
    features.reserve(static_cast<std::size_t>(feature_dim_));
    double sum = 0.0;
    double min_v = window[0];
    double max_v = window[0];
    for (double v : window) {
        sum += v;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    const double mean = sum / static_cast<double>(window.size());
    double var_acc = 0.0;
    for (double v : window) var_acc += (v - mean) * (v - mean);
    const double stdv = std::sqrt(var_acc / static_cast<double>(window.size()));
    features.push_back(static_cast<float>(mean));
    features.push_back(static_cast<float>(stdv));
    features.push_back(static_cast<float>(min_v));
    features.push_back(static_cast<float>(max_v));
    features.push_back(static_cast<float>(percentile(window, 0.25)));
    features.push_back(static_cast<float>(percentile(window, 0.75)));
    double mad = 0.0;
    for (std::size_t i = 1; i < window.size(); ++i) {
        mad += std::abs(window[i] - window[i - 1]);
    }
    features.push_back(static_cast<float>(mad / static_cast<double>(window.size() - 1)));
    int changes = 0;
    for (std::size_t i = 1; i < window.size(); ++i) {
        if ((window[i] > 0.0) != (window[i - 1] > 0.0)) ++changes;
    }
    features.push_back(static_cast<float>(changes));

    std::vector<double> fft;
    rfft_magnitudes(window, n_fft_coeffs_, fft);
    for (double c : fft) features.push_back(static_cast<float>(c));
    return features;
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> TimeSeriesEncoder::sliding_windows(
    const std::vector<double>& series, int step) const {
    std::vector<std::vector<double>> rows;
    std::vector<int> indices;
    for (int i = window_size_; i < static_cast<int>(series.size()); i += step) {
        std::vector<double> window(series.begin() + (i - window_size_), series.begin() + i);
        const auto feat = encode_window(window);
        std::vector<double> row(feat.size());
        for (std::size_t k = 0; k < feat.size(); ++k) row[k] = static_cast<double>(feat[k]);
        rows.push_back(std::move(row));
        indices.push_back(i);
    }
    return {rows, indices};
}

std::vector<float> TimeSeriesEncoder::encode_series(const std::vector<double>& series) const {
    if (static_cast<int>(series.size()) == window_size_) return encode_window(series);
    if (static_cast<int>(series.size()) > window_size_) {
        const int step = std::max(1, static_cast<int>(series.size()) / window_size_);
        const auto [windows, _] = sliding_windows(series, step);
        if (windows.empty()) {
            std::vector<double> padded = series;
            padded.resize(static_cast<std::size_t>(window_size_), series.back());
            return encode_window(padded);
        }
        std::vector<float> mean(static_cast<std::size_t>(windows.front().size()), 0.0f);
        for (const auto& row : windows) {
            for (std::size_t j = 0; j < row.size(); ++j) mean[j] += static_cast<float>(row[j]);
        }
        for (float& m : mean) m /= static_cast<float>(windows.size());
        return mean;
    }
    std::vector<double> padded = series;
    padded.resize(static_cast<std::size_t>(window_size_), series.empty() ? 0.0 : series.back());
    return encode_window(padded);
}

EcgSplit load_ecg5000(std::uint64_t seed) {
    const fs::path train_p = data_dir() / "ecg5000" / "ECG5000_TRAIN.txt";
    const fs::path test_p = data_dir() / "ecg5000" / "ECG5000_TEST.txt";
    if (fs::is_regular_file(train_p) && fs::is_regular_file(test_p)) {
        EcgSplit split;
        split.source = "ecg5000";
        if (!load_ecg_file(train_p, split.x_train, split.y_train) || !load_ecg_file(test_p, split.x_test, split.y_test)) {
            return synthetic_ecg5000(seed, 500, 450, 140, 5);
        }
        return split;
    }
    return synthetic_ecg5000(seed, 500, 450, 140, 5);
}

FinancialWindowDataset load_financial_returns(std::uint64_t seed) {
    constexpr int kWindow = 20;
    constexpr int kTickers = 4;
    TimeSeriesEncoder enc(kWindow);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed + 3));
    std::normal_distribution<double> noise(0.0, 0.01);
    FinancialWindowDataset ds;
    for (int ti = 0; ti < kTickers; ++ti) {
        std::vector<double> series(static_cast<std::size_t>(800));
        series[0] = 0.0;
        for (int i = 1; i < 800; ++i) {
            series[static_cast<std::size_t>(i)] = series[static_cast<std::size_t>(i - 1)] + noise(rng);
        }
        std::vector<double> log_returns(static_cast<std::size_t>(799));
        for (int i = 0; i < 799; ++i) {
            log_returns[static_cast<std::size_t>(i)] =
                std::log(series[static_cast<std::size_t>(i + 1)] + 1e-12) -
                std::log(series[static_cast<std::size_t>(i)] + 1e-12);
        }
        const auto [windows, indices] = enc.sliding_windows(log_returns, 1);
        for (std::size_t wi = 0; wi < windows.size(); ++wi) {
            ds.x.push_back(windows[wi]);
            const int idx = indices[wi];
            ds.y.push_back(log_returns[static_cast<std::size_t>(idx)] > 0.0 ? 1 : 0);
        }
    }
    return ds;
}

}  // namespace cypha::bench
