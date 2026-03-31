// two_stage_dif_ridge_fit_from_llr vs parity_fixtures/two_stage_ridge_fit/sidecar.json
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/regression_stub.hpp"

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

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: regression_two_stage_ridge_fit_parity <parity_fixtures/two_stage_ridge_fit/sidecar.json>\n";
      return 2;
    }
    if (cypha::regression::native_regression_milestone() < 7) {
      std::cerr << "native_regression_milestone expected >= 7\n";
      return 1;
    }
    auto j = nlohmann::json::parse(read_all(argv[1]));
    constexpr double kTol = 1e-9;

    const int n = j.at("n").get<int>();
    const int K = j.at("K").get<int>();
    const int d_in = j.at("d_in").get<int>();
    const int D2 = j.at("D2").get<int>();
    std::vector<double> llr = j.at("llr_rowmajor").get<std::vector<double>>();
    std::vector<double> X = j.at("X_rowmajor").get<std::vector<double>>();
    std::vector<double> y_raw = j.at("y_raw").get<std::vector<double>>();
    std::vector<double> encW = j.at("enc2_W").get<std::vector<double>>();
    std::vector<double> encb = j.at("enc2_b").get<std::vector<double>>();
    if (static_cast<int>(llr.size()) != n * K || static_cast<int>(X.size()) != n * d_in ||
        static_cast<int>(y_raw.size()) != n || static_cast<int>(encW.size()) != D2 * d_in ||
        static_cast<int>(encb.size()) != D2) {
      std::cerr << "bad tensor sizes\n";
      return 1;
    }

    std::vector<double> w1(static_cast<std::size_t>(K + d_in));
    double b1 = 0.0;
    std::vector<double> w2(static_cast<std::size_t>(D2));
    double b2 = 0.0;
    if (!cypha::regression::two_stage_dif_ridge_fit_from_llr(
            llr.data(), n, K, X.data(), d_in, y_raw.data(), j.at("y_mean").get<double>(),
            j.at("y_std").get<double>(), j.at("lam1").get<double>(), j.at("lam2").get<double>(), encW.data(),
            encb.data(), D2, w1.data(), &b1, w2.data(), &b2)) {
      std::cerr << "two_stage_dif_ridge_fit_from_llr failed\n";
      return 1;
    }

    std::vector<double> ew1 = j.at("expected_w1").get<std::vector<double>>();
    for (std::size_t i = 0; i < ew1.size(); ++i) {
      if (!near_eq(w1[i], ew1[i], kTol)) {
        std::cerr << "w1 mismatch at " << i << "\n";
        return 1;
      }
    }
    if (!near_eq(b1, j.at("expected_b1").get<double>(), kTol)) {
      std::cerr << "b1 mismatch\n";
      return 1;
    }
    std::vector<double> ew2 = j.at("expected_w2").get<std::vector<double>>();
    for (std::size_t i = 0; i < ew2.size(); ++i) {
      if (!near_eq(w2[i], ew2[i], kTol)) {
        std::cerr << "w2 mismatch at " << i << "\n";
        return 1;
      }
    }
    if (!near_eq(b2, j.at("expected_b2").get<double>(), kTol)) {
      std::cerr << "b2 mismatch\n";
      return 1;
    }

    std::vector<double> phi(static_cast<std::size_t>(n * D2));
    cypha::regression::rff_encode_batch_rowmajor(X.data(), n, d_in, encW.data(), encb.data(), D2, phi.data());
    std::vector<double> yhat_exp = j.at("expected_yn_hat").get<std::vector<double>>();
    std::vector<double> yhat_batch(static_cast<std::size_t>(n));
    cypha::regression::two_stage_dif_predict_batch(llr.data(), n, K, X.data(), d_in, w1.data(), b1, phi.data(), D2,
                                                   w2.data(), b2, 0.0, 1.0, yhat_batch.data());
    for (int i = 0; i < n; ++i) {
      if (!near_eq(yhat_batch[static_cast<std::size_t>(i)], yhat_exp[static_cast<std::size_t>(i)], kTol)) {
        std::cerr << "train yn_hat batch mismatch at " << i << "\n";
        return 1;
      }
    }

    std::cout << "regression two_stage ridge_fit parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "regression_two_stage_ridge_fit_parity: " << e.what() << "\n";
    return 1;
  }
}
