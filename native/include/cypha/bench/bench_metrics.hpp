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

}  // namespace cypha::bench
