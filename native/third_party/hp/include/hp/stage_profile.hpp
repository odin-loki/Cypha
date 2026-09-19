#pragma once
// Optional per-stage timers for hp::Predictor (Cypha algorithm profiling).
// Enable with -DHP_CYPHA_STAGE_PROFILE=1 when building profile tools.

#if HP_CYPHA_STAGE_PROFILE

#include <chrono>
#include <cstdint>
#include <cstdio>

namespace hp {

struct StageProfile {
    std::uint64_t predict_calls = 0;
    std::uint64_t update_calls = 0;
    std::uint64_t ctx_chain_ns = 0;
    std::uint64_t match_ns = 0;
    std::uint64_t extra_experts_ns = 0;
    std::uint64_t mixer_gates_ns = 0;
    std::uint64_t mixer_mix_ns = 0;
    std::uint64_t apm_ns = 0;
    std::uint64_t update_ns = 0;

    void reset() { *this = {}; }

    void report(std::FILE* f) const {
        const auto total = ctx_chain_ns + match_ns + extra_experts_ns + mixer_gates_ns +
                           mixer_mix_ns + apm_ns;
        if (total == 0) return;
        std::fprintf(f, "stage_profile predict_calls=%llu update_calls=%llu\n",
                     (unsigned long long)predict_calls, (unsigned long long)update_calls);
        auto row = [&](const char* name, std::uint64_t ns) {
            std::fprintf(f, "  %-16s %10.2f ms  %5.1f%%\n", name, ns / 1e6,
                         100.0 * static_cast<double>(ns) / static_cast<double>(total));
        };
        row("ctx_chain", ctx_chain_ns);
        row("match", match_ns);
        row("extra_experts", extra_experts_ns);
        row("mixer_gates", mixer_gates_ns);
        row("mixer_mix", mixer_mix_ns);
        row("apm", apm_ns);
        row("update", update_ns);
    }
};

inline StageProfile& stage_profile() {
    static StageProfile g;
    return g;
}

class StageTimer {
 public:
    explicit StageTimer(std::uint64_t& acc) : acc_(acc), t0_(std::chrono::steady_clock::now()) {}
    ~StageTimer() {
        acc_ += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0_)
                .count());
    }

 private:
    std::uint64_t& acc_;
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace hp

#endif  // HP_CYPHA_STAGE_PROFILE
