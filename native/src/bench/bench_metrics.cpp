#include "cypha/bench/bench_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace cypha::bench {

namespace {

double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

std::vector<double> ranks(const std::vector<double>& x) {
    const std::size_t n = x.size();
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
    std::vector<double> r(n);
    for (std::size_t i = 0; i < n;) {
        std::size_t j = i + 1;
        while (j < n && x[idx[j]] == x[idx[i]]) ++j;
        const double avg_rank = 0.5 * static_cast<double>(i + j - 1) + 1.0;
        for (std::size_t k = i; k < j; ++k) r[idx[k]] = avg_rank;
        i = j;
    }
    return r;
}

double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / static_cast<double>(v.size() - 1));
}

}  // namespace

double accuracy(const std::vector<std::string>& y_true, const std::vector<std::string>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size()) return 0.0;
    std::size_t correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

double accuracy_int(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size()) return 0.0;
    std::size_t correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size()) return std::numeric_limits<double>::quiet_NaN();
    double acc = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double d = y_true[i] - y_pred[i];
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(y_true.size()));
}

double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size()) return std::numeric_limits<double>::quiet_NaN();
    double acc = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) acc += std::abs(y_true[i] - y_pred[i]);
    return acc / static_cast<double>(y_true.size());
}

double r2(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() < 2 || y_true.size() != y_pred.size()) return std::numeric_limits<double>::quiet_NaN();
    const double y_mean = mean(y_true);
    double ss_tot = 0.0;
    double ss_res = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        ss_tot += (y_true[i] - y_mean) * (y_true[i] - y_mean);
        const double d = y_true[i] - y_pred[i];
        ss_res += d * d;
    }
    if (ss_tot <= 1e-24) return 0.0;
    return 1.0 - ss_res / ss_tot;
}

double bpc_from_nll(double total_nll, std::size_t n_chars) {
    if (n_chars == 0) return std::numeric_limits<double>::quiet_NaN();
    return total_nll / (static_cast<double>(n_chars) * std::log(2.0));
}

double safe_auroc(const std::vector<int>& y_true, const std::vector<double>& scores) {
    if (y_true.empty() || y_true.size() != scores.size()) return std::numeric_limits<double>::quiet_NaN();
    int n_pos = 0;
    int n_neg = 0;
    for (int y : y_true) {
        if (y != 0) ++n_pos;
        else ++n_neg;
    }
    if (n_pos == 0 || n_neg == 0) return std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> score_ranks = ranks(scores);
    double rank_sum_pos = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] != 0) rank_sum_pos += score_ranks[i];
    }
    const double u = rank_sum_pos - static_cast<double>(n_pos) * static_cast<double>(n_pos + 1) / 2.0;
    return u / (static_cast<double>(n_pos) * static_cast<double>(n_neg));
}

double expected_calibration_error(const std::vector<double>& confidences,
                                  const std::vector<double>& correct, int n_bins) {
    const int n = static_cast<int>(confidences.size());
    if (n <= 0 || confidences.size() != correct.size() || n_bins <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double ece = 0.0;
    for (int b = 0; b < n_bins; ++b) {
        const double lo = static_cast<double>(b) / static_cast<double>(n_bins);
        const double hi = static_cast<double>(b + 1) / static_cast<double>(n_bins);
        double sum_w = 0.0;
        double sum_c = 0.0;
        double sum_corr = 0.0;
        for (int i = 0; i < n; ++i) {
            if (confidences[static_cast<std::size_t>(i)] >= lo &&
                confidences[static_cast<std::size_t>(i)] < hi) {
                sum_w += 1.0;
                sum_c += confidences[static_cast<std::size_t>(i)];
                sum_corr += correct[static_cast<std::size_t>(i)];
            }
        }
        if (sum_w > 0.0) {
            ece += sum_w * std::abs(sum_c / sum_w - sum_corr / sum_w) / static_cast<double>(n);
        }
    }
    return ece;
}

double safe_spearman(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() < 2 || a.size() != b.size()) return 0.0;
    if (stddev(a) < 1e-12 || stddev(b) < 1e-12) return 0.0;
    const std::vector<double> ra = ranks(a);
    const std::vector<double> rb = ranks(b);
    const double ma = mean(ra);
    const double mb = mean(rb);
    double num = 0.0;
    double da = 0.0;
    double db = 0.0;
    for (std::size_t i = 0; i < ra.size(); ++i) {
        const double xa = ra[i] - ma;
        const double xb = rb[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    const double denom = std::sqrt(da * db);
    if (denom < 1e-24) return 0.0;
    const double rho = num / denom;
    return std::isfinite(rho) ? rho : 0.0;
}

}  // namespace cypha::bench
