#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace cypha {

// RFC4180-style CSV parse matching Python ``csv.reader`` (Excel dialect: comma, ``"`` quote, ``""`` escape,
// newlines allowed inside quoted fields).
std::vector<std::vector<std::string>> parse_csv_utf8(std::string_view text, char delimiter);

struct CsvDenseSpec {
  bool has_header = true;
  char delimiter = ',';
  /// When ``target_col_name`` is empty: column index in each full CSV row. Negative indices count from
  /// the end (``-1`` = last column), matching Python ``CSVDataset.from_file``.
  int target_col_index = -1;
  /// When non-empty and ``has_header`` is true, target column is resolved by exact header match (first
  /// occurrence). Ignores ``target_col_index``.
  std::string target_col_name;
  /// Non-empty: use these integer columns (after optional name resolution below is skipped).
  std::vector<int> feature_col_indices;
  /// When non-empty and ``has_header`` is true, feature columns are resolved by exact header match,
  /// preserving order. Ignores ``feature_col_indices``.
  std::vector<std::string> feature_col_names;
  bool regression = false;
};

struct CsvDenseResult {
  int n_rows = 0;
  int n_features = 0;
  std::vector<double> x_rowmajor;
  std::vector<std::string> y_class;
  std::vector<double> y_regression;
};

// Load numeric feature matrix + targets from a UTF-8 CSV file. Skips empty rows like ``CSVDataset``.
CsvDenseResult load_csv_dense(const std::filesystem::path& path, const CsvDenseSpec& spec);

}  // namespace cypha
