#include "cypha/preprocessor.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace cypha {

namespace {

void need(bool ok, const char* msg) {
  if (!ok) {
    throw std::runtime_error(msg);
  }
}

std::vector<double> read_vec1(const nlohmann::json& j, const char* key, bool allow_null) {
  if (!j.contains(key) || j[key].is_null()) {
    if (allow_null) {
      return {};
    }
    throw std::runtime_error(std::string("missing or null: ") + key);
  }
  const auto& a = j[key];
  need(a.is_array(), "expected array");
  std::vector<double> out;
  out.reserve(a.size());
  for (const auto& v : a) {
    out.push_back(v.get<double>());
  }
  return out;
}

std::vector<std::vector<double>> read_mat(const nlohmann::json& j, const char* key, bool allow_null) {
  if (!j.contains(key) || j[key].is_null()) {
    if (allow_null) {
      return {};
    }
    throw std::runtime_error(std::string("missing or null: ") + key);
  }
  const auto& a = j[key];
  need(a.is_array(), "expected 2d array");
  std::vector<std::vector<double>> out;
  out.reserve(a.size());
  for (const auto& row : a) {
    need(row.is_array(), "expected 2d array rows");
    std::vector<double> r;
    r.reserve(row.size());
    for (const auto& v : row) {
      r.push_back(v.get<double>());
    }
    out.push_back(std::move(r));
  }
  return out;
}

}  // namespace

PreprocessorState PreprocessorState::from_json_string(std::string_view json) {
  auto j = nlohmann::json::parse(json);
  PreprocessorState s;
  s.scale = j.value("scale", true);
  if (j.contains("pca_dim") && !j["pca_dim"].is_null()) {
    s.pca_dim = j["pca_dim"].get<int>();
  } else {
    s.pca_dim = -1;
  }
  if (j.contains("rff_dim") && !j["rff_dim"].is_null()) {
    s.rff_dim = j["rff_dim"].get<int>();
  } else {
    s.rff_dim = -1;
  }
  s.rff_gamma = j.value("rff_gamma", 1.0);
  s.seed = j.value("seed", 42);
  s.mean = read_vec1(j, "mean", true);
  s.stddev = read_vec1(j, "std", true);
  s.pca_components = read_mat(j, "pca_components", true);
  s.pca_mean = read_vec1(j, "pca_mean", true);
  s.rff_w = read_mat(j, "rff_W", true);
  s.rff_b = read_vec1(j, "rff_b", true);
  s.fitted = j.value("fitted", false);
  s.input_dim = j.value("input_dim", 0);
  s.output_dim = j.value("output_dim", 0);
  if (!s.fitted) {
    throw std::runtime_error("preprocessor.json: fitted must be true for inference");
  }
  return s;
}

PreprocessorState PreprocessorState::from_json_file(const char* path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error(std::string("Cannot open ") + path);
  }
  std::stringstream buf;
  buf << f.rdbuf();
  return from_json_string(buf.str());
}

std::vector<double> PreprocessorState::transform_one(const std::vector<double>& x) const {
  if (!fitted) {
    throw std::runtime_error("Preprocessor not fitted");
  }
  if (static_cast<int>(x.size()) != input_dim) {
    throw std::runtime_error("transform_one: input dim mismatch");
  }
  std::vector<double> X = x;

  if (scale && !mean.empty()) {
    need(static_cast<int>(mean.size()) == input_dim && static_cast<int>(stddev.size()) == input_dim,
         "scale mean/std length");
    for (int i = 0; i < input_dim; ++i) {
      X[static_cast<std::size_t>(i)] = (X[static_cast<std::size_t>(i)] - mean[static_cast<std::size_t>(i)]) /
                                       stddev[static_cast<std::size_t>(i)];
    }
  }

  if (!pca_components.empty()) {
    int d_in = static_cast<int>(X.size());
    need(static_cast<int>(pca_mean.size()) == d_in, "pca_mean length");
    int k = static_cast<int>(pca_components.size());
    need(k > 0 && static_cast<int>(pca_components[0].size()) == d_in, "pca_components shape");
    std::vector<double> out(static_cast<std::size_t>(k), 0.0);
    for (int r = 0; r < k; ++r) {
      double acc = 0.0;
      for (int c = 0; c < d_in; ++c) {
        acc += (X[static_cast<std::size_t>(c)] - pca_mean[static_cast<std::size_t>(c)]) *
               pca_components[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
      }
      out[static_cast<std::size_t>(r)] = acc;
    }
    X = std::move(out);
  }

  if (!rff_w.empty()) {
    int d_in = static_cast<int>(X.size());
    int D = static_cast<int>(rff_w.size());
    need(D == rff_dim, "rff_w rows vs rff_dim");
    need(static_cast<int>(rff_b.size()) == D, "rff_b length");
    need(D > 0 && static_cast<int>(rff_w[0].size()) == d_in, "rff_w cols");
    const double s = std::sqrt(2.0 / static_cast<double>(D));
    std::vector<double> out(static_cast<std::size_t>(D), 0.0);
    for (int r = 0; r < D; ++r) {
      double dot = rff_b[static_cast<std::size_t>(r)];
      for (int c = 0; c < d_in; ++c) {
        dot += X[static_cast<std::size_t>(c)] * rff_w[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
      }
      out[static_cast<std::size_t>(r)] = s * std::cos(dot);
    }
    X = std::move(out);
  }

  if (output_dim > 0 && static_cast<int>(X.size()) != output_dim) {
    throw std::runtime_error("transform_one: output dim mismatch vs output_dim");
  }
  return X;
}

}  // namespace cypha
