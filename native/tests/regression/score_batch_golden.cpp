// score_batch_golden — cypha::accel batch_encode + score_matrix vs
// fixtures/score_batch/sidecar.json (cypha_accel fused LLR path).
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/accel_backend.hpp"

namespace fs = std::filesystem;

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: score_batch_golden <fixtures/score_batch/sidecar.json>\n";
      return 2;
    }
    fs::path side = fs::path(argv[1]);
    std::ifstream sf(side);
    if (!sf) {
      throw std::runtime_error("cannot open sidecar");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    auto j = nlohmann::json::parse(buf.str());
    constexpr double kTol = 1e-9;

    const int n = j.at("n").get<int>();
    const int d = j.at("d").get<int>();
    const int K = j.at("K").get<int>();
    std::vector<double> F = j.at("F_rowmajor").get<std::vector<double>>();
    std::vector<double> W = j.at("W_enc_rowmajor").get<std::vector<double>>();
    std::vector<double> mu0 = j.at("mu0").get<std::vector<double>>();
    std::vector<double> inv_v = j.at("inv_v").get<std::vector<double>>();
    std::vector<double> D = j.at("D_rowmajor").get<std::vector<double>>();
    std::vector<double> D_sq = j.at("D_sq").get<std::vector<double>>();
    std::vector<double> u_k = j.at("u_k").get<std::vector<double>>();
    std::vector<double> ctx = j.at("ctx").get<std::vector<double>>();
    std::vector<double> exp_H = j.at("expected_H_rowmajor").get<std::vector<double>>();
    std::vector<double> exp_llr = j.at("expected_LLR_rowmajor").get<std::vector<double>>();

    if (static_cast<int>(F.size()) != n * d || static_cast<int>(W.size()) != d * d ||
        static_cast<int>(mu0.size()) != d || static_cast<int>(inv_v.size()) != d ||
        static_cast<int>(D.size()) != K * d || static_cast<int>(D_sq.size()) != K ||
        static_cast<int>(u_k.size()) != K || static_cast<int>(ctx.size()) != K ||
        static_cast<int>(exp_H.size()) != n * d || static_cast<int>(exp_llr.size()) != n * K) {
      std::cerr << "bad sidecar sizes\n";
      return 1;
    }

    cypha::accel::init();
    std::cout << "accel backend: " << cypha::accel::device_info() << "\n";

    std::vector<double> H(static_cast<std::size_t>(n * d));
    std::vector<double> llr(static_cast<std::size_t>(n * K));

    cypha::accel::batch_encode(F.data(), n, d, W.data(), H.data());
    for (std::size_t i = 0; i < exp_H.size(); ++i) {
      if (!near_eq(H[i], exp_H[i], kTol)) {
        std::cerr << "H mismatch at " << i << " got " << H[i] << " exp " << exp_H[i] << "\n";
        cypha::accel::shutdown();
        return 1;
      }
    }

    cypha::accel::score_matrix(H.data(), n, d, K, mu0.data(), inv_v.data(), D.data(), D_sq.data(),
                               u_k.data(), ctx.data(), llr.data());
    for (std::size_t i = 0; i < exp_llr.size(); ++i) {
      if (!near_eq(llr[i], exp_llr[i], kTol)) {
        std::cerr << "LLR mismatch at " << i << " got " << llr[i] << " exp " << exp_llr[i] << "\n";
        cypha::accel::shutdown();
        return 1;
      }
    }

    cypha::accel::shutdown();
    std::cout << "score_batch parity OK (project_features + fused_score_llr)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "score_batch_golden: " << e.what() << "\n";
    return 1;
  }
}
