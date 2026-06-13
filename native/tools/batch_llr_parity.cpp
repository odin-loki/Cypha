// batch_llr_from_x vs fixtures/batch_llr/sidecar.json (LLR from expected.npz)
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
      std::cerr << "usage: batch_llr_parity <fixtures/batch_llr/sidecar.json>\n";
      return 2;
    }
    fs::path side = fs::path(argv[1]);
    fs::path root = side.parent_path().parent_path();
    std::ifstream sf(side);
    if (!sf) {
      throw std::runtime_error("cannot open sidecar");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    auto j = nlohmann::json::parse(buf.str());
    constexpr double kTol = 1e-9;

    const int n = j.at("n").get<int>();
    const int d_in = j.at("d_in").get<int>();
    const int K = j.at("K").get<int>();
    std::vector<double> x = j.at("x_rowmajor").get<std::vector<double>>();
    std::vector<double> exp_llr = j.at("expected_llr_rowmajor").get<std::vector<double>>();
    if (static_cast<int>(x.size()) != n * d_in || static_cast<int>(exp_llr.size()) != n * K) {
      std::cerr << "bad sidecar sizes\n";
      return 1;
    }

    fs::path cypha_path = root / "reference.cypha";
    fs::path ff_path = root / "f_field.json";
    cypha::CNode root_node = cypha::load_cypha_file(cypha_path.string().c_str());
    const cypha::CNode& fh = cypha::map_get_required(root_node, "field_h");
    int fd = static_cast<int>(fh.shape[0]);
    const cypha::CNode& enc = cypha::map_get_required(root_node, "enc_W");
    int d = static_cast<int>(enc.shape[0]);
    if (d_in != d) {
      std::cerr << "d_in mismatch enc_W\n";
      return 1;
    }

    std::ifstream jf(ff_path);
    if (!jf) {
      throw std::runtime_error("cannot open f_field.json");
    }
    std::stringstream fj;
    fj << jf.rdbuf();
    std::vector<double> fflat = flatten_f_field(nlohmann::json::parse(fj.str()));
    if (static_cast<int>(fflat.size()) != d * fd) {
      throw std::runtime_error("f_field size mismatch");
    }

    cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);
    if (static_cast<int>(infer.labels.size()) != K) {
      std::cerr << "K mismatch labels\n";
      return 1;
    }

    std::vector<double> llr;
    cypha::batch_llr_from_x(infer, x.data(), n, llr);
    for (std::size_t i = 0; i < exp_llr.size(); ++i) {
      if (!near_eq(llr[i], exp_llr[i], kTol)) {
        std::cerr << "LLR mismatch at " << i << " got " << llr[i] << " exp " << exp_llr[i] << "\n";
        return 1;
      }
    }

    std::cout << "batch_llr parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "batch_llr_parity: " << e.what() << "\n";
    return 1;
  }
}
