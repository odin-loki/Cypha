// Regression guard for the kernel-LLR confidence invariant.
//
// At kernel_blend = 0.0 the blended LLR vector is, by construction, bit-identical to the linear
// one: out.llrs[k] = (1 - 0)*lin + 0*ker. The label and the softmax are therefore unchanged, so
// classify_at_h MUST return the same confidence whether or not use_kernel_llr is set.
//
// This failed before the fix: the kernel branch multiplied out.disc by world_gate, and the tail
// of classify_at_h multiplied by world_gate a second time, so the kernel path returned
// disc*gate^2 against the linear path's disc*gate.
//
// Exit 0 on success, 1 on failure. Registered as CTest `native_kernel_gate_invariant`.

#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "cypha/infer_cpu.hpp"
#include "cypha/kernel_memory.hpp"

namespace {

// A small, fully-specified model whose world gate is strictly below 1, which is the regime the
// defect lives in (gate == 1 hides it).
cypha::CyphaInferModel make_model(int d, int K) {
  cypha::CyphaInferModel m;
  m.d_latent = d;
  m.field_dim = 0;
  m.labels.reserve(static_cast<std::size_t>(K));
  for (int k = 0; k < K; ++k) {
    m.labels.push_back("c" + std::to_string(k));
  }
  m.n_obs.assign(static_cast<std::size_t>(K), 50.0);
  m.class_pi.assign(static_cast<std::size_t>(K), 1.0 / static_cast<double>(K));
  m.class_n_comp.assign(static_cast<std::size_t>(K), 1);

  // Class offsets: one separated direction per class.
  m.D.assign(static_cast<std::size_t>(K) * static_cast<std::size_t>(d), 0.0);
  for (int k = 0; k < K; ++k) {
    m.D[static_cast<std::size_t>(k) * static_cast<std::size_t>(d) + static_cast<std::size_t>(k % d)] = 1.5;
  }

  m.mu_world.assign(static_cast<std::size_t>(d), 0.0);
  m.inv_v.assign(static_cast<std::size_t>(d), 1.0);
  m.v_mean = 1.0;
  m.temperature = 1.0;
  m.has_mahal_ema = true;
  m.mahal_ema = 1.0;
  m.mahal_std_ema = 0.5;
  return m;
}

bool check(const char* what, double a, double b, double tol) {
  const double diff = std::fabs(a - b);
  if (diff <= tol) {
    std::printf("  ok    %-42s %.12f == %.12f\n", what, a, b);
    return true;
  }
  std::printf("  FAIL  %-42s %.12f != %.12f  (diff %.3e)\n", what, a, b, diff);
  return false;
}

}  // namespace

int main() {
  const int d = 8;
  const int K = 4;
  const cypha::CyphaInferModel m = make_model(d, K);

  // An input far enough from the world mean that world_gate < 1.
  // world_gate < 1 requires mahal_per_dim / r_base > ~2.51; with d=8 and inv_v=1 that needs
  // sum((h-mu)^2) > ~20.
  std::vector<double> h(static_cast<std::size_t>(d), 0.0);
  h[0] = 7.0;
  h[1] = 3.0;

  // RFF basis: n_basis() is fixed at construction, so the >= 4 guard is satisfied without
  // any training, and the basis is deterministic for a fixed seed.
  const cypha::KernelMemory km = cypha::KernelMemory::make_rff(d, /*M=*/32, /*gamma=*/0.5, /*seed=*/1234);

  bool ok = true;
  if (km.n_basis() < 4) {
    std::printf("  FAIL  kernel basis too small (%d); the guarded branch would not run\n", km.n_basis());
    return 1;
  }

  const cypha::ClassifyAtHResult lin =
      cypha::classify_at_h(m, h.data(), nullptr, m.temperature, m.mahal_ema, m.mahal_std_ema, 1.0, 1.0, true,
                           nullptr, /*use_kernel_llr=*/false, /*kernel_blend=*/0.0, nullptr);

  const cypha::ClassifyAtHResult ker =
      cypha::classify_at_h(m, h.data(), nullptr, m.temperature, m.mahal_ema, m.mahal_std_ema, 1.0, 1.0, true,
                           &km, /*use_kernel_llr=*/true, /*kernel_blend=*/0.0, nullptr);

  std::printf("kernel-LLR confidence invariant (kernel_blend = 0.0)\n");
  std::printf("  world_gate = %.12f  (must be < 1 for this test to bite)\n", lin.world_gate);
  if (!(lin.world_gate < 1.0)) {
    std::printf("  FAIL  world_gate == 1: the test input no longer exercises the gated path\n");
    return 1;
  }

  // The blend is a no-op, so every downstream quantity must match exactly.
  ok &= check("label", static_cast<double>(lin.label == ker.label), 1.0, 0.0);
  ok &= check("world_gate", lin.world_gate, ker.world_gate, 0.0);
  ok &= check("disc", lin.disc, ker.disc, 1e-12);
  ok &= check("confidence", lin.confidence, ker.confidence, 1e-12);

  // Guard the specific regression: confidence must not be the gate applied twice.
  const double squared = lin.disc * lin.world_gate * lin.world_gate;
  if (std::fabs(ker.confidence - squared) < 1e-12 && std::fabs(lin.world_gate - 1.0) > 1e-9) {
    std::printf("  FAIL  kernel confidence equals disc*gate^2 (%.12f) - the gate is applied twice\n", squared);
    ok = false;
  }

  std::printf("%s\n", ok ? "PASS" : "FAILED");
  return ok ? 0 : 1;
}
