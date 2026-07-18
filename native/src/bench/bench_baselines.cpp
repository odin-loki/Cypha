#include "cypha/bench/bench_baselines.hpp"

#include "cypha/bench/bench_metrics.hpp"
#include "cypha/regression.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace cypha::bench {

namespace {

double sigmoid(double z) {
    if (z >= 0.0) {
        const double ez = std::exp(-z);
        return 1.0 / (1.0 + ez);
    }
    const double ez = std::exp(z);
    return ez / (1.0 + ez);
}

std::vector<std::string> unique_labels_sorted(const std::vector<std::string>& labels) {
    std::unordered_set<std::string> seen;
    std::vector<std::string> out;
    for (const auto& y : labels) {
        if (seen.insert(y).second) out.push_back(y);
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool fit_binary_logistic(const std::vector<std::vector<double>>& x, const std::vector<int>& y01,
                         std::vector<double>& w, double& bias, double C, int max_iter) {
    if (x.empty() || x.size() != y01.size()) return false;
    const int n = static_cast<int>(x.size());
    const int d = static_cast<int>(x.front().size());
    w.assign(static_cast<std::size_t>(d), 0.0);
    bias = 0.0;
    const double reg = 1.0 / std::max(C, 1e-12);
    const double lr = 0.1;
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> grad_w(static_cast<std::size_t>(d), 0.0);
        double grad_b = 0.0;
        for (int i = 0; i < n; ++i) {
            double z = bias;
            for (int j = 0; j < d; ++j) z += w[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            const double p = sigmoid(z);
            const double err = p - static_cast<double>(y01[static_cast<std::size_t>(i)]);
            for (int j = 0; j < d; ++j) {
                grad_w[static_cast<std::size_t>(j)] +=
                    err * x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            }
            grad_b += err;
        }
        for (int j = 0; j < d; ++j) {
            grad_w[static_cast<std::size_t>(j)] =
                grad_w[static_cast<std::size_t>(j)] / static_cast<double>(n) + reg * w[static_cast<std::size_t>(j)];
            w[static_cast<std::size_t>(j)] -= lr * grad_w[static_cast<std::size_t>(j)];
        }
        bias -= lr * (grad_b / static_cast<double>(n));
    }
    return true;
}

std::vector<double> predict_binary_proba(const std::vector<std::vector<double>>& x, const std::vector<double>& w,
                                         double bias) {
    std::vector<double> out;
    out.reserve(x.size());
    const int d = static_cast<int>(w.size());
    for (const auto& row : x) {
        double z = bias;
        for (int j = 0; j < d; ++j) z += w[static_cast<std::size_t>(j)] * row[static_cast<std::size_t>(j)];
        out.push_back(sigmoid(z));
    }
    return out;
}

RegressionScores ridge_fit_predict(const std::vector<std::vector<double>>& train_x,
                                   const std::vector<double>& train_y,
                                   const std::vector<std::vector<double>>& test_x,
                                   const std::vector<double>& test_y, double alpha) {
    RegressionScores out;
    if (train_x.empty() || test_x.empty()) {
        out.rmse = std::numeric_limits<double>::quiet_NaN();
        out.mae = std::numeric_limits<double>::quiet_NaN();
        out.r2 = std::numeric_limits<double>::quiet_NaN();
        return out;
    }
    const int n = static_cast<int>(train_x.size());
    const int d = static_cast<int>(train_x.front().size());
    const int d_feat = d;
    std::vector<double> phi(static_cast<std::size_t>(n * (d_feat + 1)));
    for (int i = 0; i < n; ++i) {
        double* row = phi.data() + static_cast<std::size_t>(i * (d_feat + 1));
        for (int j = 0; j < d; ++j) row[j] = train_x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        row[d_feat] = 1.0;
    }
    std::vector<double> coef(static_cast<std::size_t>(d_feat + 1));
    if (!cypha::regression::ridge_fit_bias(phi.data(), n, d_feat, alpha, train_y.data(), coef.data())) {
        out.rmse = std::numeric_limits<double>::quiet_NaN();
        out.mae = std::numeric_limits<double>::quiet_NaN();
        out.r2 = std::numeric_limits<double>::quiet_NaN();
        return out;
    }
    std::vector<double> preds;
    preds.reserve(test_x.size());
    for (const auto& xrow : test_x) {
        double y = coef[static_cast<std::size_t>(d_feat)];
        for (int j = 0; j < d; ++j) y += xrow[static_cast<std::size_t>(j)] * coef[static_cast<std::size_t>(j)];
        preds.push_back(y);
    }
    return regression_scores(test_y, preds);
}

}  // namespace

ClassificationScores classification_scores(const std::vector<std::string>& y_true,
                                           const std::vector<std::string>& y_pred) {
    ClassificationScores out;
    out.accuracy = accuracy(y_true, y_pred);
    out.f1_macro = f1_macro(y_true, y_pred);
    return out;
}

RegressionScores regression_scores(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    return RegressionScores{rmse(y_true, y_pred), mae(y_true, y_pred), r2(y_true, y_pred)};
}

RegressionScores ridge_baseline(const std::vector<std::vector<double>>& train_x, const std::vector<double>& train_y,
                                const std::vector<std::vector<double>>& test_x, const std::vector<double>& test_y,
                                double alpha) {
    return ridge_fit_predict(train_x, train_y, test_x, test_y, alpha);
}

ClassificationScores logistic_regression_baseline(const std::vector<std::vector<double>>& train_x,
                                                const std::vector<std::string>& train_y,
                                                const std::vector<std::vector<double>>& test_x,
                                                const std::vector<std::string>& test_y) {
    ClassificationScores out{};
    const auto classes = unique_labels_sorted(train_y);
    if (classes.size() < 2 || train_x.empty() || test_x.empty()) {
        if (!classes.empty() && !test_y.empty()) {
            std::vector<std::string> preds(test_y.size(), classes.front());
            return classification_scores(test_y, preds);
        }
        return out;
    }

    if (classes.size() == 2) {
        std::vector<int> y01(train_y.size());
        for (std::size_t i = 0; i < train_y.size(); ++i) y01[i] = train_y[i] == classes[1] ? 1 : 0;
        std::vector<double> w;
        double bias = 0.0;
        if (!fit_binary_logistic(train_x, y01, w, bias, 1.0, 500)) return out;
        const std::vector<double> prob = predict_binary_proba(test_x, w, bias);
        std::vector<std::string> preds;
        preds.reserve(prob.size());
        for (double p : prob) preds.push_back(p >= 0.5 ? classes[1] : classes[0]);
        return classification_scores(test_y, preds);
    }

    std::vector<std::vector<double>> class_scores(test_x.size(), std::vector<double>(classes.size(), 0.0));
    for (std::size_t c = 0; c < classes.size(); ++c) {
        std::vector<int> y01(train_y.size());
        for (std::size_t i = 0; i < train_y.size(); ++i) y01[i] = train_y[i] == classes[c] ? 1 : 0;
        std::vector<double> w;
        double bias = 0.0;
        if (!fit_binary_logistic(train_x, y01, w, bias, 1.0, 500)) continue;
        const std::vector<double> prob = predict_binary_proba(test_x, w, bias);
        for (std::size_t i = 0; i < prob.size(); ++i) class_scores[i][c] = prob[i];
    }
    std::vector<std::string> preds;
    preds.reserve(test_x.size());
    for (const auto& scores : class_scores) {
        const auto best = std::max_element(scores.begin(), scores.end());
        preds.push_back(classes[static_cast<std::size_t>(best - scores.begin())]);
    }
    return classification_scores(test_y, preds);
}

ProfileJson regression_scores_json(const RegressionScores& s) {
    return ProfileJson{{"rmse", s.rmse}, {"mae", s.mae}, {"r2", s.r2}};
}

ProfileJson classification_scores_json(const ClassificationScores& s) {
    return ProfileJson{{"accuracy", s.accuracy}, {"f1_macro", s.f1_macro}};
}

ProfileJson offline_regression_baselines_json(const std::vector<std::vector<double>>& train_x,
                                              const std::vector<double>& train_y,
                                              const std::vector<std::vector<double>>& test_x,
                                              const std::vector<double>& test_y) {
    const RegressionScores ridge = ridge_baseline(train_x, train_y, test_x, test_y, 1.0);
    return ProfileJson{{"ridge", regression_scores_json(ridge)}};
}

ProfileJson offline_classification_baselines_json(const std::vector<std::vector<double>>& train_x,
                                                  const std::vector<std::string>& train_y,
                                                  const std::vector<std::vector<double>>& test_x,
                                                  const std::vector<std::string>& test_y) {
    const ClassificationScores lr = logistic_regression_baseline(train_x, train_y, test_x, test_y);
    return ProfileJson{{"logistic_regression", classification_scores_json(lr)}};
}

}  // namespace cypha::bench
