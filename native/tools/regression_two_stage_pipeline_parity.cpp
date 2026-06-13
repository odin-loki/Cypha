// Native CyphaDIF LLR + RFF stage-2 + two-stage combine vs fixtures/two_stage_pipeline/sidecar.json
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
#include "cypha/regression_stub.hpp"

namespace fs = std::filesystem;

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

std::string read_all(const char* path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }
  std::stringstream b;
  b << f.rdbuf();
  return b.str();
}

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
      std::cerr << "usage: regression_two_stage_pipeline_parity <fixtures/two_stage_pipeline/sidecar.json>\n";
      return 2;
    }
    if (cypha::regression::native_regression_milestone() < 5) {
      std::cerr << "native_regression_milestone expected >= 5\n";
      return 1;
    }
    fs::path side = fs::path(argv[1]);
    fs::path root = side.parent_path().parent_path();
    auto j = nlohmann::json::parse(read_all(argv[1]));
    constexpr double kTol = 1e-8;
    constexpr double kTolLlr = 1e-7;

    fs::path cypha_path = root / "reference.cypha";
    fs::path ff_path = root / "f_field.json";
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
    if (static_cast<int>(fflat.size()) != d * fd) {
      throw std::runtime_error("f_field size mismatch");
    }

    cypha::CyphaInferModel infer = cypha::CyphaInferModel::from_root(root_node, fflat.data(), fd);

    const int d_in = j.at("d_in").get<int>();
    const int D2 = j.at("D2").get<int>();
    const int K = j.at("K").get<int>();
    if (d_in != d) {
      std::cerr << "sidecar d_in mismatch model\n";
      return 1;
    }
    if (static_cast<int>(infer.labels.size()) != K) {
      std::cerr << "sidecar K mismatch model.labels\n";
      return 1;
    }

    std::vector<double> x;
    for (const auto& v : j.at("x")) {
      x.push_back(v.get<double>());
    }
    if (static_cast<int>(x.size()) != d_in) {
      throw std::runtime_error("bad x length");
    }

    std::vector<double> encW = j.at("enc2_W").get<std::vector<double>>();
    std::vector<double> encb = j.at("enc2_b").get<std::vector<double>>();
    std::vector<double> w1 = j.at("w1").get<std::vector<double>>();
    std::vector<double> w2 = j.at("w2").get<std::vector<double>>();
    if (static_cast<int>(encW.size()) != D2 * d_in || static_cast<int>(encb.size()) != D2 ||
        static_cast<int>(w1.size()) != K + d_in || static_cast<int>(w2.size()) != D2) {
      std::cerr << "bad tensor sizes in sidecar\n";
      return 1;
    }

    std::vector<double> llr_check;
    {
      std::vector<double> h;
      cypha::batch_encode(infer, x.data(), 1, h);
      cypha::score_matrix_use_field(infer, h.data(), 1, llr_check);
    }
    std::vector<double> llr_exp = j.at("expected_llr").get<std::vector<double>>();
    for (int i = 0; i < K; ++i) {
      if (!near_eq(llr_check[static_cast<std::size_t>(i)], llr_exp[static_cast<std::size_t>(i)], kTolLlr)) {
        std::cerr << "LLR mismatch at " << i << " got " << llr_check[static_cast<std::size_t>(i)] << " exp "
                  << llr_exp[static_cast<std::size_t>(i)] << "\n";
        return 1;
      }
    }

    const double yp = cypha::regression::two_stage_dif_predict_with_clf(
        infer, x.data(), d_in, encW.data(), encb.data(), D2, w1.data(), j.at("b1").get<double>(), w2.data(),
        j.at("b2").get<double>(), j.at("y_mean").get<double>(), j.at("y_std").get<double>());

    if (!near_eq(yp, j.at("expected_y").get<double>(), kTol)) {
      std::cerr << "two_stage pipeline y mismatch got " << yp << " exp " << j.at("expected_y").get<double>() << "\n";
      return 1;
    }

    std::cout << "regression two_stage pipeline parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "regression_two_stage_pipeline_parity: " << e.what() << "\n";
    return 1;
  }
}
