#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"
#include "cypha/similarity_index.hpp"

namespace fs = std::filesystem;

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

std::vector<double> flatten_f_field(const nlohmann::json& j) {
  std::vector<double> o;
  for (const auto& row : j) {
    for (const auto& v : row) {
      o.push_back(v.get<double>());
    }
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: similarity_index_golden <fixtures/similarity_index>\n";
      return 2;
    }
    fs::path dir = fs::path(argv[1]);
    std::ifstream sf(dir / "sidecar.json");
    if (!sf) {
      throw std::runtime_error("cannot open sidecar.json");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    auto j = nlohmann::json::parse(buf.str());

    fs::path cypha_path = dir / "reference.cypha";
    fs::path ff_path = dir / "f_field.json";
    cypha::CNode root_node = cypha::load_cypha_file(cypha_path.string().c_str());
    const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
    int fd = static_cast<int>(fh.shape[0]);
    const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
    int d = static_cast<int>(enc.shape[0]);

    std::ifstream jf(ff_path);
    if (!jf) {
      throw std::runtime_error("cannot open f_field.json");
    }
    std::stringstream fj;
    fj << jf.rdbuf();
    std::vector<double> fflat = flatten_f_field(nlohmann::json::parse(fj.str()));

    cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);
    cypha::SimilarityIndex idx(infer);
    constexpr double kTol = 1e-9;

    for (const auto& ex : j.at("add_examples")) {
      std::vector<double> x = ex.at("x").get<std::vector<double>>();
      nlohmann::json md = ex.value("metadata", nlohmann::json(nullptr));
      idx.add(x.data(), d, md);
    }

    {
      std::vector<double> x1 = j.at("sim_x1").get<std::vector<double>>();
      std::vector<double> x2 = j.at("sim_x2").get<std::vector<double>>();
      double got = idx.similarity(x1.data(), d, x2.data());
      double exp = j.at("expected_similarity").get<double>();
      if (!near_eq(got, exp, kTol)) {
        std::cerr << "similarity got " << got << " expected " << exp << "\n";
        return 1;
      }
    }

    {
      std::vector<double> xq = j.at("query_x").get<std::vector<double>>();
      int k = j.at("query_k").get<int>();
      auto hits = idx.query(xq.data(), d, k, true);
      const auto& exp = j.at("expected_query");
      if (hits.size() != exp.size()) {
        std::cerr << "query result count mismatch\n";
        return 1;
      }
      for (std::size_t i = 0; i < hits.size(); ++i) {
        if (hits[i].index != exp[i].at("index").get<int>()) {
          std::cerr << "query index mismatch at " << i << "\n";
          return 1;
        }
        if (!near_eq(hits[i].similarity, exp[i].at("similarity").get<double>(), kTol)) {
          std::cerr << "query similarity mismatch at " << i << "\n";
          return 1;
        }
      }
    }

    if (j.contains("batch_query")) {
      const auto& bq = j.at("batch_query");
      const int n = bq.at("n").get<int>();
      std::vector<double> xs = bq.at("x_rowmajor").get<std::vector<double>>();
      int k = bq.at("k").get<int>();
      auto batch = idx.query_batch(xs.data(), n, d, k);
      const auto& exp = bq.at("expected");
      if (batch.size() != exp.size()) {
        std::cerr << "batch query count mismatch\n";
        return 1;
      }
      for (std::size_t m = 0; m < batch.size(); ++m) {
        if (batch[m].size() != exp[m].size()) {
          std::cerr << "batch query row size mismatch\n";
          return 1;
        }
        for (std::size_t r = 0; r < batch[m].size(); ++r) {
          if (batch[m][r].index != exp[m][r].at("index").get<int>()) {
            std::cerr << "batch index mismatch\n";
            return 1;
          }
          if (!near_eq(batch[m][r].similarity, exp[m][r].at("similarity").get<double>(), kTol)) {
            std::cerr << "batch similarity mismatch\n";
            return 1;
          }
        }
      }
    }

    std::cout << "similarity_index parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "similarity_index_golden: " << e.what() << "\n";
    return 1;
  }
}
