// Native PreprocessorState::fit_from_design_matrix vs Python golden (scale + PCA + RFF).
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/preprocessor.hpp"

namespace fs = std::filesystem;

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

void read_json_file(const fs::path& p, nlohmann::json& out) {
  std::ifstream f(p);
  if (!f) {
    throw std::runtime_error("cannot open " + p.string());
  }
  std::stringstream buf;
  buf << f.rdbuf();
  out = nlohmann::json::parse(buf.str());
}

void align_pca_rows(cypha::PreprocessorState& pre, const nlohmann::json& exp_pc) {
  for (std::size_t k = 0; k < pre.pca_components.size(); ++k) {
    double dot = 0.0;
    const auto& row = exp_pc[k];
    for (std::size_t j = 0; j < pre.pca_components[k].size(); ++j) {
      dot += pre.pca_components[k][j] * row[j].get<double>();
    }
    if (dot < 0.0) {
      for (double& t : pre.pca_components[k]) {
        t = -t;
      }
    }
  }
}

bool vec_near(const std::vector<double>& a, const nlohmann::json& jexp, double tol) {
  if (a.size() != jexp.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!near_eq(a[i], jexp[i].get<double>(), tol)) {
      return false;
    }
  }
  return true;
}

bool mat_near(const std::vector<std::vector<double>>& a, const nlohmann::json& jexp, double tol) {
  if (a.size() != jexp.size()) {
    return false;
  }
  for (std::size_t r = 0; r < a.size(); ++r) {
    if (a[r].size() != jexp[r].size()) {
      return false;
    }
    for (std::size_t c = 0; c < a[r].size(); ++c) {
      if (!near_eq(a[r][c], jexp[r][c].get<double>(), tol)) {
        return false;
      }
    }
  }
  return true;
}

int run_one_case(const fs::path& dir) {
  nlohmann::json jd;
  read_json_file(dir / "design.json", jd);
  nlohmann::json jexp;
  read_json_file(dir / "expected_preprocessor.json", jexp);
  nlohmann::json jprobe;
  read_json_file(dir / "probe.json", jprobe);

  const int n = jd.at("n_rows").get<int>();
  const int d = jd.at("n_cols").get<int>();
  std::vector<double> data = jd.at("rowmajor").get<std::vector<double>>();
  if (static_cast<int>(data.size()) != n * d) {
    throw std::runtime_error("design rowmajor size mismatch");
  }

  cypha::PreprocessorState pre;
  pre.scale = jd.value("scale", true);
  pre.pca_dim = jd.contains("pca_dim") && !jd["pca_dim"].is_null() ? jd["pca_dim"].get<int>() : -1;
  pre.rff_dim = jd.contains("rff_dim") && !jd["rff_dim"].is_null() ? jd["rff_dim"].get<int>() : -1;
  pre.seed = jd.value("seed", 42);
  pre.rff_gamma = jd.value("rff_gamma", 1.0);
  pre.auto_rff_gamma = jd.value("auto_rff_gamma", false);
  pre.auto_rff_gamma_cv = jd.value("auto_rff_gamma_cv", false);

  pre.fit_from_design_matrix(data, n, d);

  constexpr double kTol = 1e-9;
  if (pre.input_dim != jexp.at("input_dim").get<int>()) {
    std::cerr << dir << ": input_dim mismatch\n";
    return 1;
  }
  if (pre.output_dim != jexp.at("output_dim").get<int>()) {
    std::cerr << dir << ": output_dim mismatch\n";
    return 1;
  }
  if (!pre.fitted) {
    std::cerr << dir << ": native not fitted\n";
    return 1;
  }

  if (pre.scale) {
    if (!vec_near(pre.mean, jexp.at("mean"), kTol)) {
      std::cerr << dir << ": mean mismatch\n";
      return 1;
    }
    if (!vec_near(pre.stddev, jexp.at("std"), kTol)) {
      std::cerr << dir << ": std mismatch\n";
      return 1;
    }
  } else {
    if (!jexp.at("mean").is_null() || !jexp.at("std").is_null()) {
      std::cerr << dir << ": expected null mean/std when scale=false\n";
      return 1;
    }
    if (!pre.mean.empty() || !pre.stddev.empty()) {
      std::cerr << dir << ": native should leave mean/std empty when scale=false\n";
      return 1;
    }
  }

  if (!pre.pca_components.empty()) {
    align_pca_rows(pre, jexp.at("pca_components"));
    if (!vec_near(pre.pca_mean, jexp.at("pca_mean"), kTol)) {
      std::cerr << dir << ": pca_mean mismatch\n";
      return 1;
    }
    if (!mat_near(pre.pca_components, jexp.at("pca_components"), kTol)) {
      std::cerr << dir << ": pca_components mismatch\n";
      return 1;
    }
  } else if (!jexp.at("pca_components").is_null()) {
    std::cerr << dir << ": expected PCA components but native fit produced none\n";
    return 1;
  }

  if (!pre.rff_w.empty()) {
    if (!mat_near(pre.rff_w, jexp.at("rff_W"), kTol)) {
      std::cerr << dir << ": rff_W mismatch\n";
      return 1;
    }
    if (!vec_near(pre.rff_b, jexp.at("rff_b"), kTol)) {
      std::cerr << dir << ": rff_b mismatch\n";
      return 1;
    }
  } else if (!jexp.at("rff_W").is_null()) {
    std::cerr << dir << ": expected RFF weights but native fit produced none\n";
    return 1;
  }

  for (const auto& pr : jprobe.at("probes")) {
    std::vector<double> x;
    for (const auto& v : pr.at("x")) {
      x.push_back(v.get<double>());
    }
    std::vector<double> y = pre.transform_one(x);
    const auto& yexp = pr.at("expected");
    if (y.size() != yexp.size()) {
      std::cerr << dir << ": probe output len mismatch\n";
      return 1;
    }
    for (std::size_t i = 0; i < y.size(); ++i) {
      if (!near_eq(y[i], yexp[i].get<double>(), kTol)) {
        std::cerr << dir << ": probe transform mismatch at " << i << " got " << y[i] << " exp " << yexp[i]
                  << "\n";
        return 1;
      }
    }
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: preprocessor_fit_golden <fixture_dir> [fixture_dir ...]\n"
                   "  e.g. fixtures/preprocessor_fit fixtures/preprocessor_fit_no_scale\n";
      return 2;
    }
    for (int i = 1; i < argc; ++i) {
      fs::path dir = fs::path(argv[i]);
      int rc = run_one_case(dir);
      if (rc != 0) {
        return rc;
      }
    }
    std::cout << "preprocessor_fit parity OK (" << (argc - 1) << " case(s))\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "preprocessor_fit_golden: " << e.what() << "\n";
    return 1;
  }
}
