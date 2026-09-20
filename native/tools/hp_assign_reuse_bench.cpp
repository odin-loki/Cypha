/// Benchmark: clone_from vs scratch assign_from reuse for full-vocab byte log-probs.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"
#include "cypha/cyphalm/hp_backend.hpp"

namespace {

using Clock = std::chrono::steady_clock;

long vm_hwm_kb() {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (f == nullptr) {
        return -1;
    }
    char line[256];
    long kb = -1;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        if (std::sscanf(line, "VmHWM: %ld kB", &kb) == 1) {
            break;
        }
    }
    std::fclose(f);
    return kb;
}

double elapsed_ms(Clock::time_point t0, int iters) {
    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    return (sec / static_cast<double>(iters)) * 1000.0;
}

}  // namespace

int main() {
    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = 16;
    cypha::cyphalm::CyphaLMModel model(cfg);

    std::mt19937 rng(9001);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    model.reset_context();
    for (int i = 0; i < 512; ++i) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(byte_dist(rng)));
    }

    auto& hp = model.hp_backend();
    constexpr int kWarm = 1;
    constexpr int kIters = 3;

    for (int w = 0; w < kWarm; ++w) {
        (void)hp.next_byte_log_probs_legacy(256);
        (void)hp.next_byte_log_probs_assign_reuse(256);
    }

    const long hwm_before_kb = vm_hwm_kb();

    Clock::time_point t0 = Clock::now();
    for (int i = 0; i < kIters; ++i) {
        (void)hp.next_byte_log_probs_legacy(256);
    }
    const double clone_ms = elapsed_ms(t0, kIters);
    const long hwm_after_clone_kb = vm_hwm_kb();

    t0 = Clock::now();
    for (int i = 0; i < kIters; ++i) {
        (void)hp.next_byte_log_probs_assign_reuse(256);
    }
    const double assign_ms = elapsed_ms(t0, kIters);
    const long hwm_after_assign_kb = vm_hwm_kb();

    std::printf("{\n");
    std::printf("  \"table_bits\": %d,\n", cfg.hp_table_bits);
    std::printf("  \"vocab_size\": %d,\n", cfg.vocab_size);
    std::printf("  \"iters\": %d,\n", kIters);
    std::printf("  \"next_byte_log_probs_legacy_ms\": %.3f,\n", clone_ms);
    std::printf("  \"next_byte_log_probs_assign_reuse_ms\": %.3f,\n", assign_ms);
    if (clone_ms > 0.0) {
        std::printf("  \"assign_over_clone_latency_ratio\": %.4f,\n", assign_ms / clone_ms);
    }
    std::printf("  \"vm_hwm_kb_before_bench\": %ld,\n", hwm_before_kb);
    std::printf("  \"vm_hwm_kb_after_clone\": %ld,\n", hwm_after_clone_kb);
    std::printf("  \"vm_hwm_kb_after_assign\": %ld\n", hwm_after_assign_kb);
    std::printf("}\n");
    return 0;
}
