#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cypha::bench {

double accuracy(const std::vector<std::string>& y_true, const std::vector<std::string>& y_pred);
double accuracy_int(const std::vector<int>& y_true, const std::vector<int>& y_pred);

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

}  // namespace cypha::bench
