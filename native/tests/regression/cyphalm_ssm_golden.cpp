#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "cypha/cyphalm/cellai_ssm.hpp"
#include "cypha/cyphalm/hybrid_blend_cap.hpp"
#include "cypha/cyphalm/memory_policy.hpp"

namespace {

#include "cyphalm_ssm_golden.inc"

bool near_eq(double a, double b, double atol) { return std::abs(a - b) <= atol; }

}  // namespace

int main() {
  try {
    cypha::cyphalm::CellAISSMConfig cfg;
    cfg.d_input = 64;
    cfg.d_state = 128;
    cfg.tau_fast = 10.0;
    cfg.tau_slow = 100.0;
    cfg.n_layers = 2;
    cfg.seed = 42;
    cfg.use_spectral_pde = true;
    cfg.use_multiscale = true;
    cfg.use_sparse_hebbian = true;

    cypha::cyphalm::CellAISSM ssm(cfg);
    ssm.set_projection_weights(0, std::vector<double>(kWFast0, kWFast0 + kWFast0_len),
                               std::vector<double>(kWSlow0, kWSlow0 + kWSlow0_len));
    ssm.set_projection_weights(1, std::vector<double>(kWFast1, kWFast1 + kWFast1_len),
                               std::vector<double>(kWSlow1, kWSlow1 + kWSlow1_len));

    if (ssm.context_dim() != kGoldenContext_len) {
      std::cerr << "context_dim mismatch: got " << ssm.context_dim() << " expected "
                << kGoldenContext_len << "\n";
      return 1;
    }

    std::vector<double> e_t(kGoldenInput, kGoldenInput + kGoldenInput_len);
    const auto ctx = ssm.step(e_t);
    if (static_cast<int>(ctx.size()) != kGoldenContext_len) {
      std::cerr << "output size mismatch\n";
      return 1;
    }

    constexpr double kAtol = 1e-8;
    for (int i = 0; i < kGoldenContext_len; ++i) {
      if (!near_eq(ctx[static_cast<std::size_t>(i)], kGoldenContext[i], kAtol)) {
        std::cerr << "context[" << i << "] got " << ctx[static_cast<std::size_t>(i)] << " expected "
                  << kGoldenContext[i] << "\n";
        return 1;
      }
    }

    cypha::cyphalm::CyphaLMNativeConfig native_cfg;
    if (native_cfg.bptt_steps != 256 || !native_cfg.train_ssm) {
      std::cerr << "CyphaLMNativeConfig defaults mismatch\n";
      return 1;
    }
    const double capped = cypha::cyphalm::capped_hybrid_alpha(0.9, native_cfg);
    if (!near_eq(capped, 0.5, 1e-12)) {
      std::cerr << "capped_hybrid_alpha mismatch\n";
      return 1;
    }

    ssm.reset();
    cypha::cyphalm::apply_memory_policy(cypha::cyphalm::MemoryPolicy::CARRY_SLOW, ssm);
    for (const auto& h : ssm.h_states()) {
      for (double v : h) {
        if (v != 0.0) {
          std::cerr << "CARRY_SLOW: fast state not zeroed\n";
          return 1;
        }
      }
    }

    std::cout << "cyphalm_ssm_golden ok (context_dim=" << kGoldenContext_len << ")\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "cyphalm_ssm_golden error: " << ex.what() << "\n";
    return 1;
  }
}
