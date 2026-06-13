// Smoke: KernelMemory export_snapshot / import_snapshot round-trip.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

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

}  // namespace

int main() {
  try {
    constexpr int d = 4;
    constexpr int M = 16;
    cypha::KernelMemory km(d, M, 42);
    const std::vector<std::string> labels = {"pos", "neg"};
    for (int i = 0; i < 8; ++i) {
      const double h[d] = {0.1 * static_cast<double>(i), 0.2, 0.3, 0.4};
      km.update(h, labels[static_cast<std::size_t>(i % 2)], labels, 0.05);
    }
    if (km.n_basis() < 4) {
      throw std::runtime_error("basis too thin");
    }

    const double h_test[d] = {0.5, 0.4, 0.3, 0.2};
    std::vector<double> scores_before;
    km.score_all(h_test, labels, scores_before);

    const cypha::KernelMemory::Snapshot snap = km.export_snapshot();
    cypha::KernelMemory km2(d, M, 99);
    km2.import_snapshot(snap);

    std::vector<double> scores_after;
    km2.score_all(h_test, labels, scores_after);
    compare_vec(scores_after, scores_before, "kernel scores", 1e-12);

    std::vector<double> phi_before;
    std::vector<double> phi_after;
    km.phi(h_test, phi_before);
    km2.phi(h_test, phi_after);
    compare_vec(phi_after, phi_before, "phi", 1e-12);

    std::cout << "kernel_snapshot_roundtrip OK (n_basis=" << km.n_basis() << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "kernel_snapshot_roundtrip FAIL: " << e.what() << "\n";
    return 1;
  }
}
