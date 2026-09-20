/// Microbench: next_byte_log_probs latency after delta undo bit-tree.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "cypha/cyphalm/cyphalm_config.hpp"
#include "cypha/cyphalm/cyphalm_model.hpp"

#if defined(__linux__)
#include <fstream>
#endif

namespace {

long rss_kb() {
#if defined(__linux__)
    std::ifstream in("/proc/self/status");
    std::string line;
    long kb = -1;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &kb);
            break;
        }
    }
    return kb;
#else
    return -1;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    int table_bits = 16;
    int iters = 3;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--table-bits" && i + 1 < argc) {
            table_bits = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        }
    }

    cypha::cyphalm::CyphaLMConfig cfg;
    cypha::cyphalm::apply_hp_production_recipe(cfg);
    cfg.vocab_size = 256;
    cfg.hp_table_bits = table_bits;
    cypha::cyphalm::CyphaLMModel model(cfg);

    std::mt19937 rng(42);
    model.reset_context();
    for (int i = 0; i < 256; ++i) {
        model.hp_backend().consume_byte(static_cast<std::uint8_t>(rng() & 0xff));
    }

    const long rss_before = rss_kb();
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        (void)model.hp_backend().next_byte_log_probs_bit_tree(256);
    }
    const double us = std::chrono::duration<double>(Clock::now() - t0).count() * 1e6 /
                      static_cast<double>(iters);

    std::printf("hp_inference_bench OK table_bits=%d iters=%d next_byte_log_probs_bit_tree_us=%.0f "
                "vm_rss_kb=%ld\n",
                table_bits, iters, us, rss_before);
    return 0;
}
