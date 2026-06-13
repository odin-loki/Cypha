#pragma once

#include <filesystem>
#include <fstream>
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

/// Count non-empty data rows (after header). Stops at ``max_rows`` when ``max_rows > 0`` (0 = all rows).
int count_csv_dense_rows(const std::filesystem::path& path, const CsvDenseSpec& spec, int max_rows = 0);

/// Incrementally reads dense CSV rows without materialising the full matrix in memory.
class CsvDenseChunkReader {
 public:
  CsvDenseChunkReader(const std::filesystem::path& path, const CsvDenseSpec& spec, int max_rows = 0,
                      int chunk_rows = 4096);

  /// Reads up to ``chunk_rows`` data rows into ``out``. Sets ``global_row_start`` to the file row index
  /// of the first row in ``out``. Returns false when no more rows (``out.n_rows`` may be 0).
  bool read_chunk(CsvDenseResult& out, int& global_row_start);

  int n_features() const { return n_features_; }

 private:
  std::ifstream file_;
  CsvDenseSpec spec_;
  int n_features_{0};
  int tgt_col_{0};
  std::vector<int> feat_cols_;
  int max_rows_{0};
  int chunk_rows_{4096};
  int rows_read_{0};
  int next_global_row_{0};
  bool eof_{false};
  std::vector<std::string> primed_row_;
  bool has_primed_row_{false};
};

}  // namespace cypha
