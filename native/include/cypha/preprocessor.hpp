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
  /// **scale + PCA** only). ``rff_dim`` must be unset (≤0); RFF weights must be produced in Python.
  void fit_from_design_matrix(const std::vector<double>& row_major, int n_rows, int n_cols);
};

}  // namespace cypha
