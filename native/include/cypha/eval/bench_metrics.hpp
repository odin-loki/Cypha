#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cypha::bench {

double accuracy(const std::vector<std::string>& y_true, const std::vector<std::string>& y_pred);
double accuracy_int(const std::vector<int>& y_true, const std::vector<int>& y_pred);

/// Macro-averaged F1 over classes present in ``y_true``.
double f1_macro(const std::vector<std::string>& y_true, const std::vector<std::string>& y_pred);

/// Mean per-class recall (sklearn ``balanced_accuracy_score``).
double balanced_accuracy(const std::vector<std::string>& y_true, const std::vector<std::string>& y_pred);

double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred);
double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred);
double r2(const std::vector<double>& y_true, const std::vector<double>& y_pred);

/// Bits per character from total negative log-likelihood (natural log) and character count.
double bpc_from_nll(double total_nll, std::size_t n_chars);

/// Safe Spearman rank correlation (returns 0 when undefined).
double safe_spearman(const std::vector<double>& a, const std::vector<double>& b);

/// ROC-AUC via Mann-Whitney U (NaN when only one class present).
double safe_auroc(const std::vector<int>& y_true, const std::vector<double>& scores);

/// Expected Calibration Error (equal-width confidence bins). Returns NaN when n <= 0.
double expected_calibration_error(const std::vector<double>& confidences,
                                  const std::vector<double>& correct, int n_bins = 15);

/// CRPS for a single Gaussian forecast N(mu, sigma^2) at observation y. Returns NaN when undefined.
double crps_gaussian(double y, double mu, double sigma);

/// Mean CRPS over paired observations and Gaussian forecasts (same length vectors).
double crps_gaussian_mean(const std::vector<double>& y_true, const std::vector<double>& mu,
                          const std::vector<double>& sigma);

/// Fraction of y_true inside [mu - z*sigma, mu + z*sigma]. Default z=1.645 is 90% two-sided normal.
double predictive_interval_coverage(const std::vector<double>& y_true, const std::vector<double>& mu,
                                    const std::vector<double>& sigma, double z = 1.645);

/// Lag-1 sample autocorrelation of regression residuals (y_true - y_pred). Returns 0 when undefined.
double residual_autocorr_lag1(const std::vector<double>& y_true, const std::vector<double>& y_pred);

/// Spectral flatness (geometric / arithmetic mean of DFT power bins) on regression residuals.
double residual_spectral_flatness(const std::vector<double>& y_true, const std::vector<double>& y_pred);

/// Summary of per-sample |top1 − top2| logit margins (distance from decision boundary proxy).
struct MarginDistribution {
    double mean = 0.0;
    double p50 = 0.0;
    double p10 = 0.0;
};

/// |top-1 − top-2| logit gap for one LLR row (0 when k <= 1).
double logit_margin_top2(const double* llr_row, int k);

/// Mean / p50 / p10 of |margin| over samples (NaN fields when empty).
MarginDistribution margin_distribution(const std::vector<double>& margins);

}  // namespace cypha::bench
