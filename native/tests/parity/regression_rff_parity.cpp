// CTest: RFF encode + ridge/bias + MKE expert dots vs fixtures/rff_regression/sidecar.json
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
      std::cerr << "usage: regression_rff_parity <fixtures/rff_regression/sidecar.json>\n";
      return 2;
    }
    if (cypha::regression::native_regression_milestone() < 3) {
      std::cerr << "native_regression_milestone expected >= 3\n";
      return 1;
    }
    auto j = nlohmann::json::parse(read_all(argv[1]));
    constexpr double kTol = 1e-8;

    const auto& rr = j.at("rff_ridge");
    const int n = rr.at("n").get<int>();
    const int d_in = rr.at("d_in").get<int>();
    const int D = rr.at("D").get<int>();
    const double lam = rr.at("lam").get<double>();
    const double y_mean = rr.at("y_mean").get<double>();
    const double y_std = rr.at("y_std").get<double>();
    std::vector<double> X = rr.at("X").get<std::vector<double>>();
    std::vector<double> W = rr.at("W").get<std::vector<double>>();
    std::vector<double> b = rr.at("b").get<std::vector<double>>();
    std::vector<double> y_raw = rr.at("y_raw").get<std::vector<double>>();
    std::vector<double> exp_phi = rr.at("expected_phi_rowmajor").get<std::vector<double>>();
    std::vector<double> exp_coef = rr.at("expected_coef").get<std::vector<double>>();
    std::vector<double> exp_yp = rr.at("expected_y_pred").get<std::vector<double>>();

    if (static_cast<int>(X.size()) != n * d_in || static_cast<int>(W.size()) != D * d_in ||
        static_cast<int>(b.size()) != D) {
      std::cerr << "rff_ridge: bad array sizes\n";
      return 1;
    }

    std::vector<double> phi(static_cast<std::size_t>(n * D));
    cypha::regression::rff_encode_batch_rowmajor(X.data(), n, d_in, W.data(), b.data(), D, phi.data());
    for (std::size_t i = 0; i < exp_phi.size(); ++i) {
      if (!near_eq(phi[i], exp_phi[i], kTol)) {
        std::cerr << "phi mismatch at " << i << "\n";
        return 1;
      }
    }

    std::vector<double> yn(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      yn[static_cast<std::size_t>(i)] = (y_raw[static_cast<std::size_t>(i)] - y_mean) / y_std;
    }
    std::vector<double> coef(static_cast<std::size_t>(D + 1));
    if (!cypha::regression::ridge_fit_bias(phi.data(), n, D, lam, yn.data(), coef.data())) {
      std::cerr << "ridge_fit_bias failed\n";
      return 1;
    }
    for (std::size_t i = 0; i < exp_coef.size(); ++i) {
      if (!near_eq(coef[i], exp_coef[i], kTol)) {
        std::cerr << "coef mismatch at " << i << " got " << coef[i] << " exp " << exp_coef[i] << "\n";
        return 1;
      }
    }

    std::vector<double> pred_norm(static_cast<std::size_t>(n));
    cypha::regression::linear_predict_with_bias(phi.data(), n, D, coef.data(), pred_norm.data());
    for (int i = 0; i < n; ++i) {
      const double yp = pred_norm[static_cast<std::size_t>(i)] * y_std + y_mean;
      if (!near_eq(yp, exp_yp[static_cast<std::size_t>(i)], kTol)) {
        std::cerr << "y_pred mismatch at " << i << "\n";
        return 1;
      }
    }

    const auto& md = j.at("mke_dots");
    const int d_feat = md.at("d_feat").get<int>();
    const int K = md.at("K").get<int>();
    std::vector<double> ph1 = md.at("phi").get<std::vector<double>>();
    std::vector<double> wexp = md.at("W_experts_rowmajor").get<std::vector<double>>();
    std::vector<double> exp_dots = md.at("expected_dots").get<std::vector<double>>();
    if (static_cast<int>(ph1.size()) != d_feat || static_cast<int>(wexp.size()) != K * d_feat ||
        static_cast<int>(exp_dots.size()) != K) {
      std::cerr << "mke_dots: bad sizes\n";
      return 1;
    }
    std::vector<double> dots(static_cast<std::size_t>(K));
    cypha::regression::mke_expert_linear_dots(ph1.data(), d_feat, K, wexp.data(), dots.data());
    for (int k = 0; k < K; ++k) {
      if (!near_eq(dots[static_cast<std::size_t>(k)], exp_dots[static_cast<std::size_t>(k)], kTol)) {
        std::cerr << "mke dot mismatch at " << k << "\n";
        return 1;
      }
    }

    // Combine dots with mixture (sanity: same as predict_mixture_scalar on scalar mus).
    std::vector<double> probs(static_cast<std::size_t>(K), 1.0 / static_cast<double>(K));
    std::vector<double> varz(static_cast<std::size_t>(K), 0.01);
    double y_mix = 0.0;
    double u_mix = 0.0;
    cypha::regression::predict_mixture_scalar(probs.data(), dots.data(), varz.data(),
                                                static_cast<std::size_t>(K), y_mix, u_mix);
    double exp_y = 0.0;
    double exp_v = 0.0;
    for (int k = 0; k < K; ++k) {
      exp_y += probs[static_cast<std::size_t>(k)] * exp_dots[static_cast<std::size_t>(k)];
      exp_v += probs[static_cast<std::size_t>(k)] * varz[static_cast<std::size_t>(k)];
    }
    if (!near_eq(y_mix, exp_y, kTol) || !near_eq(u_mix, std::sqrt(std::max(exp_v, 0.0)), kTol)) {
      std::cerr << "mixture on mke dots mismatch\n";
      return 1;
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
