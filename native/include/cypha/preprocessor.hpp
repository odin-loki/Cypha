#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cypha {

/// Matches `Preprocessor` in `cypha_studio/core/dataset.py` + `PREPROCESSOR_CONTRACT.md`.
struct PreprocessorState {
  bool scale{true};
  int pca_dim{-1};
  int rff_dim{-1};
  double rff_gamma{1.0};
  /// Median pairwise distance heuristic (Python ``RFFEncoder.auto_gamma``).
  bool auto_rff_gamma{false};
  /// Cross-validated γ grid search (Python ``RFFEncoder.auto_gamma_cv``). Takes precedence over
  /// ``auto_rff_gamma`` when both are set.
  bool auto_rff_gamma_cv{false};
  /// Phase 5: structured orthogonal RFF (SORF/Fastfood) instead of iid Gaussian rows.
  bool rff_sorf{false};
  int seed{42};
  std::vector<double> mean;
  std::vector<double> stddev;
  std::vector<std::vector<double>> pca_components;
  std::vector<double> pca_mean;
  std::vector<std::vector<double>> rff_w;
  std::vector<double> rff_b;
  bool fitted{false};
  int input_dim{0};
  int output_dim{0};

  /// Throws on missing required keys / bad shapes.
  static PreprocessorState from_json_file(const char* path);
  static PreprocessorState from_json_string(std::string_view json);

  [[nodiscard]] std::vector<double> transform_one(const std::vector<double>& x) const;

  /// Fit from ``n_rows``×``n_cols`` row-major design matrix (matches Python ``Preprocessor.fit`` for
  /// scale + PCA + optional RFF). RFF weights use NumPy-compatible ``default_rng(seed)``.
  /// When ``y_rowmajor`` is non-null and ``auto_rff_gamma_cv`` is set, γ is chosen via hold-out ridge
  /// CV on ``y`` (Python ``auto_gamma_cv`` semantics). Otherwise CV uses 5-fold reconstruction MSE on ``X``.
  void fit_from_design_matrix(const std::vector<double>& row_major, int n_rows, int n_cols,
                              const std::vector<double>* y_rowmajor = nullptr, int y_cols = 1);
};

/// Default γ grid (matches Python ``RFFEncoder.auto_gamma_cv``).
const std::vector<double>& default_rff_gamma_cv_grid();

/// Select RFF γ by cross-validated ridge score on temporary RFF features.
/// With ``y_rowmajor``: single 20% hold-out, ridge without bias, ``reg=1e-5``.
/// Without ``y``: 5-fold CV minimizing mean reconstruction MSE of ``X`` from RFF features.
double estimate_rff_gamma_cv(const std::vector<double>& x_rowmajor, int n_rows, int n_cols, int rff_dim,
                             int seed, const std::vector<double>* y_rowmajor = nullptr, int y_cols = 1);

}  // namespace cypha
