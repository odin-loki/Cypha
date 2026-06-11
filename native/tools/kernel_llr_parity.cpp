// kernel_llr_parity vs parity_fixtures/kernel_llr/sidecar.json
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"

namespace {

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

void compare_vec(const std::vector<double>& a, const std::vector<double>& b, const char* name, double atol) {
  if (a.size() != b.size()) {
    throw std::runtime_error(std::string("size mismatch: ") + name);
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!near_eq(a[i], b[i], atol)) {
      std::cerr << name << "[" << i << "] got " << a[i] << " expected " << b[i] << "\n";
      throw std::runtime_error(std::string("mismatch: ") + name);
    }
  }
}

cypha::KernelMemory load_kernel_from_json(const nlohmann::json& st) {
  const int feat_dim = st.at("feat_dim").get<int>();
  const int M = st.at("M").get<int>();
  cypha::KernelMemory km(feat_dim, M, 0);
  std::map<std::string, std::vector<double>> weights;
  for (const auto& pr : st.at("weights").items()) {
    weights[pr.key()] = pr.value().get<std::vector<double>>();
  }
  std::vector<double> basis = st.at("basis_rowmajor").get<std::vector<double>>();
  km.load_state(st.at("n_basis").get<int>(), st.at("n_seen").get<int>(), basis.data(), M, weights);
  return km;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: kernel_llr_parity <parity_fixtures/kernel_llr/sidecar.json>\n";
      return 2;
    }
    std::ifstream sf(argv[1]);
    if (!sf) {
      throw std::runtime_error("cannot open sidecar");
    }
    std::stringstream buf;
    buf << sf.rdbuf();
    const auto j = nlohmann::json::parse(buf.str());
    constexpr double kTol = 1e-9;

    const auto& st = j.at("kernel_state");
    cypha::KernelMemory km = load_kernel_from_json(st);

    const int feat_dim = st.at("feat_dim").get<int>();
    const int M = st.at("M").get<int>();
    const int n_test = j.at("n_test").get<int>();
    const int K = j.at("K").get<int>();
    const double blend = j.at("blend").get<double>();
    std::vector<std::string> labels = j.at("labels").get<std::vector<std::string>>();

    std::vector<double> h = j.at("h_test_rowmajor").get<std::vector<double>>();
    if (static_cast<int>(h.size()) != n_test * feat_dim) {
      throw std::runtime_error("h_test size mismatch");
    }

    std::vector<double> exp_phi = j.at("expected_phi_rowmajor").get<std::vector<double>>();
    std::vector<double> exp_kernel = j.at("expected_kernel_scores_rowmajor").get<std::vector<double>>();
    std::vector<double> linear = j.at("linear_llr_rowmajor").get<std::vector<double>>();
    std::vector<double> exp_blend = j.at("expected_blended_rowmajor").get<std::vector<double>>();

    std::vector<double> phi(M, 0.0);
    std::vector<double> got_phi;
    got_phi.reserve(static_cast<std::size_t>(n_test * M));
    std::vector<double> got_kernel;
    got_kernel.reserve(static_cast<std::size_t>(n_test * K));
    std::vector<double> kernel_row(static_cast<std::size_t>(K));

    for (int i = 0; i < n_test; ++i) {
      const double* row = h.data() + static_cast<std::size_t>(i * feat_dim);
      km.phi(row, phi);
      got_phi.insert(got_phi.end(), phi.begin(), phi.end());
      km.score_all(row, labels, kernel_row);
      got_kernel.insert(got_kernel.end(), kernel_row.begin(), kernel_row.end());
    }
    compare_vec(got_phi, exp_phi, "phi", kTol);
    compare_vec(got_kernel, exp_kernel, "kernel_scores", kTol);

    std::vector<double> blended = linear;
    for (int i = 0; i < n_test; ++i) {
      for (int k = 0; k < K; ++k) {
        const std::size_t idx = static_cast<std::size_t>(i * K + k);
        blended[idx] = (1.0 - blend) * linear[idx] + blend * got_kernel[idx];
      }
    }
    compare_vec(blended, exp_blend, "blended_llr", kTol);

    // Update step with fixed reservoir index.
    const auto& up = j.at("update_step");
    cypha::KernelMemory km_up = load_kernel_from_json(st);
    std::vector<double> h_up = up.at("h").get<std::vector<double>>();
    const std::string label = up.at("label").get<std::string>();
    std::vector<std::string> all_labels = up.at("all_labels").get<std::vector<std::string>>();
    const double lr = up.at("lr").get<double>();
    const int n_basis_before = up.at("n_basis_before").get<int>();
    const int n_basis_after = up.at("n_basis_after").get<int>();
    km_up.update(h_up.data(), label, all_labels, lr);
    if (km_up.n_basis() != n_basis_after) {
      std::cerr << "n_basis after update: got " << km_up.n_basis() << " expected " << n_basis_after
                << " (before " << n_basis_before << ")\n";
      return 1;
    }

    for (const auto& pr : up.at("weights_after").items()) {
      auto it = km_up.weights().find(pr.key());
      if (it == km_up.weights().end()) {
        throw std::runtime_error("missing weight key after update");
      }
      std::vector<double> exp_w = pr.value().get<std::vector<double>>();
      compare_vec(it->second, exp_w, pr.key().c_str(), kTol);
    }

    std::cout << "kernel_llr parity OK\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "kernel_llr_parity: " << e.what() << "\n";
    return 1;
  }
}
