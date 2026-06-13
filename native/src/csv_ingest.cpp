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

struct CsvColumnLayout {
  int ncols{0};
  int tgt{0};
  std::vector<int> feat;
};

CsvColumnLayout resolve_column_layout(const std::vector<std::string>& header_row,
                                      const std::vector<std::string>& first_data_row, const CsvDenseSpec& spec) {
  const int ncols = static_cast<int>(first_data_row.size());
  if (spec.has_header && static_cast<int>(header_row.size()) != ncols) {
    throw std::runtime_error("csv: header column count != data column count");
  }

  CsvColumnLayout layout;
  layout.ncols = ncols;
  if (!spec.target_col_name.empty()) {
    if (!spec.has_header) {
      throw std::runtime_error("csv: target_col_name requires has_header=true");
    }
    layout.tgt = header_column_index(header_row, spec.target_col_name);
  } else {
    layout.tgt = normalize_col_index(spec.target_col_index, ncols);
  }
  if (layout.tgt < 0 || layout.tgt >= ncols) {
    throw std::runtime_error("csv: target column index out of range");
  }

  if (!spec.feature_col_names.empty()) {
    if (!spec.has_header) {
      throw std::runtime_error("csv: feature_col_names requires has_header=true");
    }
    layout.feat.reserve(spec.feature_col_names.size());
    for (const std::string& name : spec.feature_col_names) {
      layout.feat.push_back(header_column_index(header_row, name));
    }
  } else if (!spec.feature_col_indices.empty()) {
    layout.feat.reserve(spec.feature_col_indices.size());
    for (int c : spec.feature_col_indices) {
      layout.feat.push_back(normalize_col_index(c, ncols));
    }
  } else {
    for (int c = 0; c < ncols; ++c) {
      if (c != layout.tgt) {
        layout.feat.push_back(c);
      }
    }
  }

  for (int c : layout.feat) {
    if (c < 0 || c >= ncols || c == layout.tgt) {
      throw std::runtime_error("csv: invalid feature column indices");
    }
  }
  return layout;
}

void append_dense_row(const std::vector<std::string>& row, const CsvColumnLayout& layout, const CsvDenseSpec& spec,
                      CsvDenseResult& out) {
  if (static_cast<int>(row.size()) != layout.ncols) {
    throw std::runtime_error("csv: ragged row");
  }
  for (int c : layout.feat) {
    double v = 0.0;
    parse_double_strict(row[static_cast<std::size_t>(c)], &v);
    out.x_rowmajor.push_back(v);
  }
  if (spec.regression) {
    double yv = 0.0;
    parse_double_strict(row[static_cast<std::size_t>(layout.tgt)], &yv);
    out.y_regression.push_back(yv);
  } else {
    out.y_class.push_back(row[static_cast<std::size_t>(layout.tgt)]);
  }
  ++out.n_rows;
}

bool read_csv_row(std::istream& in, char delim, std::vector<std::string>& row) {
  row.clear();
  if (!in || in.peek() == std::char_traits<char>::eof()) {
    return false;
  }

  bool row_done = false;
  while (!row_done) {
    std::string field;
    const int ch = in.peek();
    if (ch == std::char_traits<char>::eof()) {
      row_done = true;
      break;
    }
    if (static_cast<char>(ch) == '"') {
      in.get();
      while (in) {
        const int c = in.get();
        if (c == std::char_traits<char>::eof()) {
          break;
        }
        if (static_cast<char>(c) == '"') {
          if (in.peek() == '"') {
            in.get();
            field += '"';
          } else {
            break;
          }
        } else {
          field += static_cast<char>(c);
        }
      }
    } else {
      while (in) {
        const int c = in.peek();
        if (c == std::char_traits<char>::eof()) {
          break;
        }
        const char cc = static_cast<char>(c);
        if (cc == delim || cc == '\n' || cc == '\r') {
          break;
        }
        field += cc;
        in.get();
      }
    }
    row.push_back(std::move(field));

    if (!in || in.peek() == std::char_traits<char>::eof()) {
      row_done = true;
    } else {
      const int c = in.peek();
      if (static_cast<char>(c) == delim) {
        in.get();
      } else if (static_cast<char>(c) == '\r') {
        in.get();
        if (in.peek() == '\n') {
          in.get();
        }
        row_done = true;
      } else if (static_cast<char>(c) == '\n') {
        in.get();
        row_done = true;
      }
    }
  }

  if (row.empty()) {
    return false;
  }
  return true;
}

bool read_next_data_row(std::istream& in, char delim, std::vector<std::string>& row) {
  while (read_csv_row(in, delim, row)) {
    if (!row.empty()) {
      return true;
    }
  }
  return false;
}

class CsvDenseStreamCore {
 public:
  CsvDenseStreamCore(const std::filesystem::path& path, const CsvDenseSpec& spec)
      : spec_(spec) {
    file_.open(path, std::ios::binary);
    if (!file_) {
      throw std::runtime_error("cannot open csv: " + path.string());
    }

    std::vector<std::string> row;
    if (spec_.has_header) {
      if (!read_next_data_row(file_, spec_.delimiter, row)) {
        throw std::runtime_error("csv: missing header row");
      }
      header_row_ = std::move(row);
    }

    if (!read_next_data_row(file_, spec_.delimiter, row)) {
      throw std::runtime_error("csv: no data rows");
    }
    first_data_row_ = std::move(row);
    layout_ = resolve_column_layout(header_row_, first_data_row_, spec_);
    primed_row_ = first_data_row_;
    has_primed_row_ = true;
  }

  bool read_data_row(std::vector<std::string>& row) {
    if (has_primed_row_) {
      row = std::move(primed_row_);
      has_primed_row_ = false;
      return true;
    }
    return read_next_data_row(file_, spec_.delimiter, row);
  }

  const CsvColumnLayout& layout() const { return layout_; }

 private:
  CsvDenseSpec spec_;
  std::ifstream file_;
  std::vector<std::string> header_row_;
  std::vector<std::string> first_data_row_;
  CsvColumnLayout layout_;
  std::vector<std::string> primed_row_;
  bool has_primed_row_{false};
};

}  // namespace

std::vector<std::vector<std::string>> parse_csv_utf8(std::string_view text, char delimiter) {
  std::vector<std::vector<std::string>> rows;
  const size_t n = text.size();
  size_t i = 0;
  auto at = [&](size_t j) -> char { return j < n ? text[j] : '\0'; };

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
        while (i < n && at(i) != delimiter && at(i) != '\n' && at(i) != '\r') {
          field += at(i);
          ++i;
        }
      }
      row.push_back(std::move(field));
      if (i >= n) {
        row_done = true;
      } else if (at(i) == delimiter) {
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

  const CsvColumnLayout layout = resolve_column_layout(header_row, data[0], spec);

  CsvDenseResult out;
  out.n_features = static_cast<int>(layout.feat.size());
  out.x_rowmajor.reserve(static_cast<std::size_t>(data.size()) * layout.feat.size());
  if (spec.regression) {
    out.y_regression.reserve(data.size());
  } else {
    out.y_class.reserve(data.size());
  }

  for (const auto& row : data) {
    append_dense_row(row, layout, spec, out);
  }
  return out;
}

int count_csv_dense_rows(const std::filesystem::path& path, const CsvDenseSpec& spec, int max_rows) {
  CsvDenseStreamCore stream(path, spec);
  int count = 0;
  std::vector<std::string> row;
  while (stream.read_data_row(row)) {
    ++count;
    if (max_rows > 0 && count >= max_rows) {
      break;
    }
  }
  return count;
}

CsvDenseChunkReader::CsvDenseChunkReader(const std::filesystem::path& path, const CsvDenseSpec& spec, int max_rows,
                                         int chunk_rows)
    : spec_(spec), max_rows_(max_rows), chunk_rows_(chunk_rows > 0 ? chunk_rows : 4096) {
  file_.open(path, std::ios::binary);
  if (!file_) {
    throw std::runtime_error("cannot open csv: " + path.string());
  }

  std::vector<std::string> row;
  std::vector<std::string> header_row;
  if (spec_.has_header) {
    if (!read_next_data_row(file_, spec_.delimiter, row)) {
      throw std::runtime_error("csv: missing header row");
    }
    header_row = std::move(row);
  }
  if (!read_next_data_row(file_, spec_.delimiter, row)) {
    throw std::runtime_error("csv: no data rows");
  }

  const CsvColumnLayout layout = resolve_column_layout(header_row, row, spec_);
  n_features_ = static_cast<int>(layout.feat.size());
  tgt_col_ = layout.tgt;
  feat_cols_ = layout.feat;

  primed_row_ = std::move(row);
  has_primed_row_ = true;
  next_global_row_ = 0;
}

bool CsvDenseChunkReader::read_chunk(CsvDenseResult& out, int& global_row_start) {
  if (eof_) {
    out = {};
    global_row_start = next_global_row_;
    return false;
  }

  out = {};
  out.n_features = n_features_;
  global_row_start = next_global_row_;

  CsvColumnLayout layout;
  layout.ncols = 0;
  layout.tgt = tgt_col_;
  layout.feat = feat_cols_;

  auto consume_row = [&](const std::vector<std::string>& row) {
    if (layout.ncols == 0) {
      layout.ncols = static_cast<int>(row.size());
    }
    append_dense_row(row, layout, spec_, out);
    ++rows_read_;
    ++next_global_row_;
  };

  if (has_primed_row_) {
    consume_row(primed_row_);
    has_primed_row_ = false;
  }

  std::vector<std::string> row;
  while (out.n_rows < chunk_rows_) {
    if (max_rows_ > 0 && rows_read_ >= max_rows_) {
      eof_ = true;
      break;
    }
    if (!read_next_data_row(file_, spec_.delimiter, row)) {
      eof_ = true;
      break;
    }
    consume_row(row);
  }

  if (out.n_rows == 0) {
    eof_ = true;
    return false;
  }
  return true;
}

}  // namespace cypha
