#include "cypha/csv_ingest.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cypha {
namespace {

void parse_double_strict(const std::string& s, double* out) {
  const char* p = s.c_str();
  char* end = nullptr;
  *out = std::strtod(p, &end);
  if (end == p) {
    throw std::runtime_error("invalid float field: " + s);
  }
  while (*end == ' ' || *end == '\t') {
    ++end;
  }
  if (*end != '\0') {
    throw std::runtime_error("invalid float field: " + s);
  }
}

int header_column_index(const std::vector<std::string>& header, const std::string& name) {
  for (std::size_t i = 0; i < header.size(); ++i) {
    if (header[i] == name) {
      return static_cast<int>(i);
    }
  }
  throw std::runtime_error("csv: column name not found: " + name);
}

int normalize_col_index(int c, int ncols) {
  if (ncols < 1) {
    return c;
  }
  if (c < 0) {
    c += ncols;
  }
  return c;
}

}  // namespace

std::vector<std::vector<std::string>> parse_csv_utf8(std::string_view t, char delim) {
  std::vector<std::vector<std::string>> rows;
  const size_t n = t.size();
  size_t i = 0;
  auto at = [&](size_t j) -> char { return j < n ? t[j] : '\0'; };

  while (i < n) {
    std::vector<std::string> row;
    bool row_done = false;
    while (!row_done) {
      std::string field;
      if (i < n && at(i) == '"') {
        ++i;
        while (i < n) {
          if (at(i) == '"') {
            if (i + 1 < n && at(i + 1) == '"') {
              field += '"';
              i += 2;
            } else {
              ++i;
              break;
            }
          } else {
            field += at(i);
            ++i;
          }
        }
      } else {
        while (i < n && at(i) != delim && at(i) != '\n' && at(i) != '\r') {
          field += at(i);
          ++i;
        }
      }
      row.push_back(std::move(field));
      if (i >= n) {
        row_done = true;
      } else if (at(i) == delim) {
        ++i;
      } else if (at(i) == '\r') {
        ++i;
        if (i < n && at(i) == '\n') {
          ++i;
        }
        row_done = true;
      } else if (at(i) == '\n') {
        ++i;
        row_done = true;
      }
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

CsvDenseResult load_csv_dense(const std::filesystem::path& path, const CsvDenseSpec& spec) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open csv: " + path.string());
  }
  std::ostringstream buf;
  buf << f.rdbuf();
  std::string text = buf.str();

  std::vector<std::vector<std::string>> rows = parse_csv_utf8(text, spec.delimiter);
  std::size_t idx = 0;
  std::vector<std::string> header_row;
  if (spec.has_header) {
    if (idx >= rows.size()) {
      throw std::runtime_error("csv: missing header row");
    }
    header_row = rows[idx];
    ++idx;
  }
  while (idx < rows.size() && rows[idx].empty()) {
    ++idx;
  }
  if (idx >= rows.size()) {
    throw std::runtime_error("csv: no data rows");
  }

  std::vector<std::vector<std::string>> data;
  data.push_back(std::move(rows[idx]));
  ++idx;
  for (; idx < rows.size(); ++idx) {
    if (!rows[idx].empty()) {
      data.push_back(std::move(rows[idx]));
    }
  }

  const int ncols = static_cast<int>(data[0].size());
  if (spec.has_header && static_cast<int>(header_row.size()) != ncols) {
    throw std::runtime_error("csv: header column count != data column count");
  }

  int tgt = 0;
  if (!spec.target_col_name.empty()) {
    if (!spec.has_header) {
      throw std::runtime_error("csv: target_col_name requires has_header=true");
    }
    tgt = header_column_index(header_row, spec.target_col_name);
  } else {
    tgt = normalize_col_index(spec.target_col_index, ncols);
  }
  if (tgt < 0 || tgt >= ncols) {
    throw std::runtime_error("csv: target column index out of range");
  }

  std::vector<int> feat;
  if (!spec.feature_col_names.empty()) {
    if (!spec.has_header) {
      throw std::runtime_error("csv: feature_col_names requires has_header=true");
    }
    feat.reserve(spec.feature_col_names.size());
    for (const std::string& name : spec.feature_col_names) {
      feat.push_back(header_column_index(header_row, name));
    }
  } else if (!spec.feature_col_indices.empty()) {
    feat.reserve(spec.feature_col_indices.size());
    for (int c : spec.feature_col_indices) {
      int cc = normalize_col_index(c, ncols);
      feat.push_back(cc);
    }
  } else {
    for (int c = 0; c < ncols; ++c) {
      if (c != tgt) {
        feat.push_back(c);
      }
    }
  }

  for (int c : feat) {
    if (c < 0 || c >= ncols || c == tgt) {
      throw std::runtime_error("csv: invalid feature column indices");
    }
  }

  for (const auto& row : data) {
    if (static_cast<int>(row.size()) != ncols) {
      throw std::runtime_error("csv: ragged row");
    }
  }

  CsvDenseResult out;
  out.n_rows = static_cast<int>(data.size());
  out.n_features = static_cast<int>(feat.size());
  out.x_rowmajor.reserve(static_cast<std::size_t>(out.n_rows) * feat.size());
  if (spec.regression) {
    out.y_regression.reserve(static_cast<std::size_t>(out.n_rows));
  } else {
    out.y_class.reserve(static_cast<std::size_t>(out.n_rows));
  }

  for (const auto& row : data) {
    for (int c : feat) {
      double v = 0.0;
      parse_double_strict(row[static_cast<std::size_t>(c)], &v);
      out.x_rowmajor.push_back(v);
    }
    if (spec.regression) {
      double yv = 0.0;
      parse_double_strict(row[static_cast<std::size_t>(tgt)], &yv);
      out.y_regression.push_back(yv);
    } else {
      out.y_class.push_back(row[static_cast<std::size_t>(tgt)]);
    }
  }
  return out;
}

}  // namespace cypha
