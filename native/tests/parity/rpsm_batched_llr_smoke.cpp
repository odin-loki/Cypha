// rpsm_batched_llr_smoke — rpsm_score_matrix_batched vs score_matrix_use_field on
// fixtures/reference.cypha + native_parity.bin encoded rows.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "cypha/infer_cpu.hpp"
#include "cypha/load_cypha.hpp"

namespace {

constexpr const char* kUsage =
    "usage: rpsm_batched_llr_smoke <reference.cypha> <native_parity.bin>\n";

void read_all(const char* path, std::vector<std::uint8_t>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error(std::string("Cannot open ") + path);
  }
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << kUsage;
      return 2;
    }
    std::vector<std::uint8_t> sidecar;
    read_all(argv[2], sidecar);
    if (sidecar.size() < 8 + 4 * 5 + 8 * 2) {
      throw std::runtime_error("native_parity.bin too small");
    }
    if (std::memcmp(sidecar.data(), "CYPHNP01", 8) != 0) {
      throw std::runtime_error("native_parity.bin: bad magic (expected CYPHNP01)");
    }
    std::uint32_t ver = 0;
    std::uint32_t n = 0;
    std::uint32_t d = 0;
    std::uint32_t k = 0;
    std::uint32_t field_dim = 0;
    std::size_t o = 8;
    std::memcpy(&ver, sidecar.data() + o, 4);
    o += 4;
    std::memcpy(&n, sidecar.data() + o, 4);
    o += 4;
    std::memcpy(&d, sidecar.data() + o, 4);
    o += 4;
    std::memcpy(&k, sidecar.data() + o, 4);
    o += 4;
    std::memcpy(&field_dim, sidecar.data() + o, 4);
    o += 4;
    double temperature = 0.0;
    double eps = 0.0;
    std::memcpy(&temperature, sidecar.data() + o, 8);
    o += 8;
    std::memcpy(&eps, sidecar.data() + o, 8);
    o += 8;
    if (ver != 1u && ver != 2u) {
      throw std::runtime_error("native_parity.bin: unsupported version");
    }
    const std::size_t core = o + static_cast<std::size_t>(d) * field_dim * 8u +
                             static_cast<std::size_t>(n) * d * 8u +
                             static_cast<std::size_t>(n) * k * 8u * 2u + static_cast<std::size_t>(n) * 8u;
    const std::size_t need_v1 = core;
    const std::size_t need_v2 = core + static_cast<std::size_t>(n) * 8u * 2u;
    if (ver == 1u && sidecar.size() < need_v1) {
      throw std::runtime_error("native_parity.bin: truncated payload");
    }
    if (ver == 2u && sidecar.size() < need_v2) {
      throw std::runtime_error("native_parity.bin: truncated payload (v2 tail)");
    }
    (void)eps;

    const double* f_field = reinterpret_cast<const double*>(sidecar.data() + o);
    o += static_cast<std::size_t>(d) * field_dim * 8u;
    const double* x_in = reinterpret_cast<const double*>(sidecar.data() + o);
    o += static_cast<std::size_t>(n) * d * 8u;

    cypha::CNode root = cypha::load_cypha_file(argv[1]);
    cypha::CyphaInferModel model =
        cypha::CyphaInferModel::from_root(root, f_field, static_cast<int>(field_dim));
    if (std::abs(model.temperature - temperature) > 1e-15) {
      throw std::runtime_error("temperature mismatch between .cypha and native_parity.bin");
    }
    if (model.d_latent != static_cast<int>(d)) {
      throw std::runtime_error("d_latent mismatch");
    }
    if (static_cast<int>(model.labels.size()) != static_cast<int>(k)) {
      throw std::runtime_error("class count mismatch");
    }

    std::vector<double> H;
    cypha::batch_encode(model, x_in, static_cast<int>(n), H);

    std::vector<double> llr_ref;
    cypha::score_matrix_use_field(model, H.data(), static_cast<int>(n), llr_ref);

    std::vector<double> llr_batched;
    cypha::rpsm_score_matrix_batched(model, H.data(), static_cast<int>(n), llr_batched);

    if (llr_ref.size() != llr_batched.size()) {
      std::cerr << "LLR size mismatch ref=" << llr_ref.size() << " batched=" << llr_batched.size() << "\n";
      return 1;
    }

    constexpr double kTol = 1e-12;
    for (std::size_t i = 0; i < llr_ref.size(); ++i) {
      if (!near_eq(llr_ref[i], llr_batched[i], kTol)) {
        std::cerr << "batched LLR mismatch at " << i << " ref=" << llr_ref[i] << " batched=" << llr_batched[i]
                  << "\n";
        return 1;
      }
    }

    std::cout << "rpsm_batched_llr_smoke OK (n=" << n << ", d=" << d << ", k=" << k << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rpsm_batched_llr_smoke: " << e.what() << "\n";
    return 1;
  }
}
