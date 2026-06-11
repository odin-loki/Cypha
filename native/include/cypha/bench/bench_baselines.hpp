#pragma once

#include "cypha/bench/bench_report_json.hpp"

#include <string>
#include <vector>

namespace cypha::bench {

struct ClassificationScores {
    double accuracy{0.0};
    double f1_macro{0.0};
};

struct RegressionScores {
    double rmse{0.0};
    double mae{0.0};
    double r2{0.0};
};

ClassificationScores classification_scores(const std::vector<std::string>& y_true,
                                           const std::vector<std::string>& y_pred);

RegressionScores regression_scores(const std::vector<double>& y_true, const std::vector<double>& y_pred);

/// sklearn Ridge(alpha=1.0) equivalent on standardized features.
RegressionScores ridge_baseline(const std::vector<std::vector<double>>& train_x,
                              const std::vector<double>& train_y,
                              const std::vector<std::vector<double>>& test_x,
                              const std::vector<double>& test_y, double alpha = 1.0);

/// sklearn LogisticRegression(max_iter=500, C=1.0, l2) one-vs-rest equivalent.
ClassificationScores logistic_regression_baseline(const std::vector<std::vector<double>>& train_x,
                                                const std::vector<std::string>& train_y,
                                                const std::vector<std::vector<double>>& test_x,
                                                const std::vector<std::string>& test_y);

ProfileJson offline_regression_baselines_json(const std::vector<std::vector<double>>& train_x,
                                              const std::vector<double>& train_y,
                                              const std::vector<std::vector<double>>& test_x,
                                              const std::vector<double>& test_y);

ProfileJson offline_classification_baselines_json(const std::vector<std::vector<double>>& train_x,
                                                  const std::vector<std::string>& train_y,
                                                  const std::vector<std::vector<double>>& test_x,
                                                  const std::vector<std::string>& test_y);

ProfileJson regression_scores_json(const RegressionScores& s);
ProfileJson classification_scores_json(const ClassificationScores& s);

}  // namespace cypha::bench
