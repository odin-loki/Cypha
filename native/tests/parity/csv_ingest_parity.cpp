// Native CSV dense load vs Python ``CSVDataset.from_file`` golden (``fixtures/csv_ingest/``).
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "cypha/csv_ingest.hpp"

namespace fs = std::filesystem;

namespace {

void read_json_file(const fs::path& p, nlohmann::json& out) {
  std::ifstream f(p);
  if (!f) {
    throw std::runtime_error("cannot open " + p.string());
  }
  std::stringstream buf;
  buf << f.rdbuf();
  out = nlohmann::json::parse(buf.str());
}

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

int run_case(const fs::path& fixture_dir, const nlohmann::json& c) {
  const std::string csv_name = c.at("csv").get<std::string>();
  cypha::CsvDenseSpec spec;
  spec.has_header = c.value("has_header", true);
  std::string delim = c.value("delimiter", ",");
  if (delim.size() != 1) {
    std::cerr << "delimiter must be a single character\n";
    return 1;
  }
  spec.delimiter = delim[0];
  spec.target_col_name = c.value("target_col_name", std::string{});
  if (spec.target_col_name.empty()) {
    spec.target_col_index = c.at("target_col_index").get<int>();
  } else {
    spec.target_col_index = -1;
  }
  if (c.contains("feature_col_names") && c["feature_col_names"].is_array() && !c["feature_col_names"].empty()) {
    for (const auto& v : c.at("feature_col_names")) {
      spec.feature_col_names.push_back(v.get<std::string>());
    }
  } else if (c.contains("feature_col_indices") && !c["feature_col_indices"].is_null()) {
    for (const auto& v : c.at("feature_col_indices")) {
      spec.feature_col_indices.push_back(v.get<int>());
    }
  }
  const std::string task = c.at("task").get<std::string>();
  spec.regression = (task == "regression");
  if (!spec.regression && task != "classification") {
    std::cerr << "task must be classification or regression\n";
    return 1;
  }

  cypha::CsvDenseResult got =
      cypha::load_csv_dense(fixture_dir / csv_name, spec);

  nlohmann::json jexp;
  read_json_file(fixture_dir / c.at("expected").get<std::string>(), jexp);

  constexpr double kTol = 1e-12;
  if (got.n_rows != jexp.at("n_rows").get<int>()) {
    std::cerr << "n_rows mismatch\n";
    return 1;
  }
  if (got.n_features != jexp.at("n_features").get<int>()) {
    std::cerr << "n_features mismatch\n";
    return 1;
  }
  const auto& xexp = jexp.at("x_rowmajor");
  if (got.x_rowmajor.size() != xexp.size()) {
    std::cerr << "x_rowmajor length mismatch\n";
    return 1;
  }
  for (std::size_t i = 0; i < got.x_rowmajor.size(); ++i) {
    if (!near_eq(got.x_rowmajor[i], xexp[i].get<double>(), kTol)) {
      std::cerr << "x mismatch at " << i << "\n";
      return 1;
    }
  }
  if (spec.regression) {
    const auto& yexp = jexp.at("y_regression");
    if (got.y_regression.size() != yexp.size()) {
      std::cerr << "y_regression length mismatch\n";
      return 1;
    }
    for (std::size_t i = 0; i < got.y_regression.size(); ++i) {
      if (!near_eq(got.y_regression[i], yexp[i].get<double>(), kTol)) {
        std::cerr << "y regression mismatch at " << i << "\n";
        return 1;
      }
    }
  } else {
    const auto& yexp = jexp.at("y_class");
    if (got.y_class.size() != yexp.size()) {
      std::cerr << "y_class length mismatch\n";
      return 1;
    }
    for (std::size_t i = 0; i < got.y_class.size(); ++i) {
      if (got.y_class[i] != yexp[i].get<std::string>()) {
        std::cerr << "y class mismatch at " << i << "\n";
        return 1;
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: csv_ingest_parity <fixtures/csv_ingest>\n";
      return 2;
    }
    fs::path fixture_dir = fs::path(argv[1]);
    nlohmann::json jroot;
    read_json_file(fixture_dir / "cases.json", jroot);
    for (const auto& c : jroot.at("cases")) {
      int rc = run_case(fixture_dir, c);
      if (rc != 0) {
        return rc;
      }
    }
    std::cout << "csv_ingest parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "csv_ingest_parity: " << e.what() << "\n";
    return 1;
  }
}
